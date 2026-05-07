# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union, Sized

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
import re
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url
# from trl import GRPOTrainer

from accelerate.utils import is_peft_model, set_seed
import PIL.Image

import copy
from torch.utils.data import Sampler
import warnings

if is_peft_available():
    from peft import PeftConfig, get_peft_model, PeftModel

if is_wandb_available():
    import wandb

from open_r1.vlm_modules.vlm_module import VLMBaseModule
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility.
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count

# class ShardedRepeatRandomSampler(RepeatRandomSampler):
#     """
#     在 RepeatRandomSampler 基础上做 rank 级切分：
#     - world_size / rank: 全局并行度与本进程序号
#     - set_epoch(e): 每个 epoch 变更随机种子，避免各 rank 的 shuffle 一致
#     """
#     def __init__(
#         self,
#         data_source: Sized,
#         mini_repeat_count: int,
#         batch_size: int,
#         repeat_count: int = 1,
#         seed: Optional[int] = None,
#         world_size: int = 1,
#         rank: int = 0,
#     ):
#         super().__init__(
#             data_source=data_source,
#             mini_repeat_count=mini_repeat_count,
#             batch_size=batch_size,
#             repeat_count=repeat_count,
#             seed=seed,
#         )
#         self.world_size = int(world_size)
#         self.rank = int(rank)
#         assert 0 <= self.rank < self.world_size

#         # Trainer 会在每个 epoch 调 set_epoch（如果 sampler 有该方法）
#         self._epoch = 0

#     def set_epoch(self, epoch: int):
#         self._epoch = int(epoch)
#         if self.seed is not None:
#             # 让不同 epoch+rank 的随机序列都不同
#             self.generator.manual_seed(self.seed + epoch * 1000 + self.rank)

#     def __iter__(self):
#         # 复用父类逻辑先得到“全局顺序”
#         # 注意：父类的 generator 已经在 set_epoch 时按 (seed + epoch + rank) 重置
#         global_iter = super().__iter__()

#         # 关键：对“样本序列”的**位置**做切分（round-robin）
#         # 让每个 rank 只拿属于自己的那部分
#         # 这里按“生成次序”的位置切：pos % world_size == rank
#         for pos, idx in enumerate(global_iter):
#             if (pos % self.world_size) == self.rank:
#                 yield idx



class MASKED_VLMGRPOTrainer_CSPO(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        reward_weights: Optional[list[float]] = None,
        args: GRPOConfig = None,
        vlm_module: VLMBaseModule = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        freeze_vision_modules: Optional[bool] = False,
        attn_implementation: str = "flash_attention_2",
        torch_dtype: str = "bfloat16",
        **kwargs,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
        
        self.vlm_module = vlm_module

        self.reward_weights = reward_weights
        
        self.debug_mask = False

        if os.getenv("METHOD")=="CSPO":
            self.CSPO = True # component reward
        else:
            self.CSPO = False

        self.align_reward_name = "align_reward"

        # 组件权重初值（和=a=7），你原先是全1，这里也OK；确保 sum==a 即可
        self.component_reward_weights = None

        # 累计器（微批累计用）
        self._comp_sum = None
        self._comp_cnt = 0

        # 超参（也可作为函数入参传）
        self._weight_beta = 4.0
        self._weight_a    = 7.0
        self._weight_lam  = 0.9

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        # FIXME
        # Remember to modify it in the invernvl
        model_init_kwargs["attn_implementation"] = attn_implementation
        if model_init_kwargs.get("torch_dtype") is None:
            model_init_kwargs["torch_dtype"] = torch_dtype
        
        assert isinstance(model, str), "model must be a string in the current implementation"
        model_id = model
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            pass  # torch_dtype is already a torch.dtype or "auto" or None
        elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
            torch_dtype = getattr(torch, torch_dtype)
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )
        # Disable caching if gradient checkpointing is enabled (not supported)
        model_init_kwargs["use_cache"] = (
            False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )
        model_cls = self.vlm_module.get_model_class(model_id, model_init_kwargs)
        # If `model_id` points to a PEFT adapter directory (contains adapter_config.json),
        # load the base model first and then wrap it with the saved adapter weights.
        loaded_from_adapter = False
        if is_peft_available() and os.path.isdir(model_id) and os.path.exists(os.path.join(model_id, "adapter_config.json")):
            print("Loading PEFT model from adapter directory...")
            try:
                peft_cfg = PeftConfig.from_pretrained(model_id)
                base_name = peft_cfg.base_model_name_or_path or model_id
                base_model = model_cls.from_pretrained(base_name, **model_init_kwargs)
                model = PeftModel.from_pretrained(base_model, model_id, is_trainable=True)
                loaded_from_adapter = True
            except Exception as e:
                print(f"Fallback to normal loading due to: {e}")
                model = model_cls.from_pretrained(model_id, **model_init_kwargs)
        else:
            model = model_cls.from_pretrained(model_id, **model_init_kwargs)

        # LoRA
        self.vision_modules_keywords = self.vlm_module.get_vision_modules_keywords()
        if peft_config is not None and not loaded_from_adapter:
            print("Applying LoRA...")
            def find_all_linear_names(model, multimodal_keywords):
                cls = torch.nn.Linear
                lora_module_names = set()
                for name, module in model.named_modules():
                    # LoRA is not applied to the vision modules
                    if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                        continue
                    if isinstance(module, cls):
                        lora_module_names.add(name)
                for m in lora_module_names:  # needed for 16-bit
                    if "embed_tokens" in m:
                        lora_module_names.remove(m)
                return list(lora_module_names)
            target_modules = find_all_linear_names(model, self.vision_modules_keywords)
            peft_config.target_modules = target_modules
            model = get_peft_model(model, peft_config)

        # kws = list(self.vlm_module.get_vision_modules_keywords())
        # print("[vision keywords]", kws)
        # import sys
        # sys.exit(0)

        # Freeze vision modules
        if freeze_vision_modules:
            print("Freezing vision modules...")
            for n, p in model.named_parameters():
                if any(keyword in n for keyword in self.vision_modules_keywords):
                    p.requires_grad = False
        # Compute the number of trainable parameters and print the parameter that is trainable
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        total_params = sum(p.numel() for p in trainable_params)
        # for n, p in model.named_parameters():
        #     if p.requires_grad:
        #         print(n, p.shape)
        print(f"Total trainable parameters: {total_params}")

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            self.ref_model = model_cls.from_pretrained(model_id, **model_init_kwargs)
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)

        # Processing class
        if processing_class is None:
            processing_cls = self.vlm_module.get_processing_class()
            processing_class = processing_cls.from_pretrained(model_id, trust_remote_code=model_init_kwargs.get("trust_remote_code", None))
            for component, processing_keyword in self.vlm_module.get_custom_processing_keywords():
                if processing_keyword in kwargs:
                    # If we cannot find component in processing_class, return the processing_class itself
                    processing_component = getattr(processing_class, component, processing_class)
                    setattr(processing_component, processing_keyword, kwargs[processing_keyword])
            if getattr(processing_class, "tokenizer",  None) is not None:
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
            else:
                assert isinstance(processing_class, PreTrainedTokenizerBase), "processing_class must be an instance of PreTrainedTokenizerBase if it has no tokenizer attribute"
                pad_token_id = processing_class.pad_token_id

        self.vlm_module.post_model_init(model, processing_class)
        self.vlm_module.post_model_init(self.ref_model, processing_class)

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_prompt_length = None
        if args.max_prompt_length is not None:
            warnings.warn("Setting max_prompt_length is currently not supported, it has been set to None")

        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,  
            temperature=1.0,
            pad_token_id=pad_token_id,
        )
        if hasattr(self.vlm_module, "get_eos_token_id"): # For InternVL
            self.generation_config.eos_token_id = self.vlm_module.get_eos_token_id(processing_class)
        self.beta = args.beta
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            # if self.is_deepspeed_enabled:
            if is_deepspeed_zero3_enabled():
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)
    
    def _ensure_component_buffers(self, device, dtype, a=7.0):
        # component weights：和为 a 的均匀初始化（首次）
        if getattr(self, "component_reward_weights", None) is None:
            if os.getenv("WEIGHTED")=="true":
                #这里需要从log文件中读取
                weights_list =  [1.9841009378433228, 0.5595909953117371, 1.3075982332229614, 0.7750168442726135, 0.4022205173969269, 0.5600659847259521, 1.4114054441452026]
                self.component_reward_weights = torch.tensor(weights_list, device=device, dtype=dtype)
            else:
                self.component_reward_weights = torch.full((7,), a / 7.0, device=device, dtype=dtype)
            #self.component_reward_weights = torch.full((7,), a / 7.0, device=device, dtype=dtype)

        # 微批累计器（首次）
        if getattr(self, "_comp_sum", None) is None:
            self._comp_sum = torch.zeros(7, device=device, dtype=dtype)
            self._comp_cnt = 0
    
    '''
    def _load_scaler(self, checkpoint):
        print("🚫 Skipping GradScaler state loading.")
        return
    
    def _load_optimizer_and_scheduler(self, checkpoint):
        print("🚫 Skipping optimizer and scheduler state loading from checkpoint.")
        # 不执行加载，等价于重新初始化优化器和 scheduler
        return
    '''
    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            model.gradient_checkpointing_enable()
            try:
                # For InternVL; these operations are copied from the original training script of InternVL
                model.language_model.config.use_cache = False
                model.vision_model.gradient_checkpointing = True
                model.vision_model.encoder.gradient_checkpointing = True
                model.language_model._set_gradient_checkpointing()
                # This line is necessary, otherwise the `model.gradient_checkpointing_enable()` will be executed during the training process, leading to an error since InternVL does not support this operation.
                args.gradient_checkpointing = False
            except:
                pass

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model
    
    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, **custom_multimodal_inputs):
        logits = model(input_ids=input_ids, attention_mask=attention_mask, **custom_multimodal_inputs).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    def _get_per_token_entropy(self, model, input_ids, attention_mask, **custom_multimodal_inputs):
        """
        计算每个位置的熵 H(p_t) = -sum_v p_t(v) log p_t(v)（自然对数，单位 nats）。
        为了降低峰值显存，按 batch 的每一行循环计算（与 _get_per_token_logps 的写法一致）。
        返回形状: (B, L-1)，已经做了与 _get_per_token_logps 同样的对齐（去掉首 token）。
        """
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, **custom_multimodal_inputs).logits  # (B,L,V)
            logits = logits[:, :-1, :]                 # 对齐下一 token 预测
            entropies = []
            for logits_row in logits:                  # (L-1, V)
                log_probs = torch.log_softmax(logits_row, dim=-1)     # (L-1,V)
                probs = torch.exp(log_probs)                          # (L-1,V)
                H = -(probs * log_probs).sum(dim=-1)                  # (L-1,)
                entropies.append(H)
            return torch.stack(entropies)              # (B, L-1)

    def _get_per_token_logps_and_entropy(self, model, input_ids, attention_mask, **custom_multimodal_inputs):
        # 一次 forward
        logits = model(input_ids=input_ids, attention_mask=attention_mask, **custom_multimodal_inputs).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it

        per_token_logps = []
        per_token_entropy = []

        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
            
            with torch.no_grad():
                probs = torch.exp(log_probs)                          # (L-1,V)
                H = -(probs * log_probs).sum(dim=-1)                  # (L-1,)
                per_token_entropy.append(H)

        return torch.stack(per_token_logps), torch.stack(per_token_entropy)  # (B,L-1),(B,L-1)


    def _prepare_inputs(self, inputs):
        # Simple pass-through, just like original
        return inputs

    def _get_key_from_inputs(self, x, key):
        ele = x.get(key, None)
        assert ele is not None, f"The key {key} is not found in the input"
        if isinstance(ele, list):
            return [e for e in ele]
        else:
            return [ele]


    # ========= helpers =========
    def iter_table_spans(self, text: str):
        pat = re.compile(
            r"""
            \\begin\s*\{\s*table\s*\}
            (?:\s*\[[^\]]*\])?
            (.*?)
            \\end\s*\{\s*table\s*\}
            """,
            re.VERBOSE | re.DOTALL
        )
        for m in pat.finditer(text):
            yield m.span(0)

    def parse_align_spans(self, text: str):
        spans = []
        pat = re.compile(
            r"""
            \\begin
            \s*\{\s*
            (?:tabularx?|tabular\*)
            \s*\}\s*
            (?:\[[^\]]*\]\s*)?
            \{(?P<spec>[^\}]*)\}
            """,
            re.VERBOSE
        )
        for ts, te in self.iter_table_spans(text):
            chunk = text[ts:te]
            for m in pat.finditer(chunk):
                s, e = m.span(0)
                spans.append((ts + s, ts + e))
        return spans

    def parse_horizontal_line_spans(self, text: str):
        spans = []
        pat = re.compile(
            r"""
            (\\hline\s*\\hline)
            |(\\cmidrule\s*(?:\([^)]+\))?\s*\{[^}]*\})
            |(\\cline\s*\{[^}]*\})
            |(\\hhline\s*\{[^}]*\})
            |(\\toprule|\\midrule|\\bottomrule)
            |(\\hline)
            """,
            re.VERBOSE
        )
        for ts, te in self.iter_table_spans(text):
            chunk = text[ts:te]
            for m in pat.finditer(chunk):
                s, e = m.span(0)
                spans.append((ts + s, ts + e))
        return spans

    def parse_structure_token_spans(self, text: str):
        spans = []
        pat = re.compile(
            r"""
            (?<!\\)&                                   # 未转义 &
            |(\\\\)                                    # 行分隔
            |(\\multicolumn\s*\{[^}]*\}\s*\{[^}]*\})   # 仅前两个参数
            |(\\multirow\s*\{[^}]*\}\s*\{[^}]*\})      # 仅前两个参数
            """,
            re.VERBOSE
        )
        for ts, te in self.iter_table_spans(text):
            chunk = text[ts:te]
            for m in pat.finditer(chunk):
                s, e = m.span(0)
                spans.append((ts + s, ts + e))
        return spans

    def parse_caption_spans(self, text: str):
        spans = []
        pat = re.compile(
            r"""
            \\caption
            \s*
            \{[^}]*\}
            """,
            re.VERBOSE
        )
        for ts, te in self.iter_table_spans(text):
            chunk = text[ts:te]
            for m in pat.finditer(chunk):
                s, e = m.span(0)
                spans.append((ts + s, ts + e))
        return spans
    
    # === 在 LatexTableParser 类内新增该函数 ===
    def parse_text_content_style_spans(self, text,align,hlines,struct,caps):
        """
        只在 \begin{table}...\end{table} 内，返回“文本内容与风格”的全局 spans：
        = table 内减去以下已知成分后的剩余区间（过滤纯空白） =
        - 对齐起始：\begin{tabular...}{...}
        - 横线命令：\hline、\cline{..}、\hhline{..}、\toprule、\midrule、\bottomrule、\cmidrule(..){..}
        - 结构 token：未转义 &、\\\\、\multicolumn{..}{..}（仅前两参）、\multirow{..}{..}（仅前两参）
        - caption：\caption{...}
        - 非内容：\begin{table}[...]/\end{table}、\centering
        """
        import re

        # 仅 table 内：begin/end table 与 \centering 也作为排除项
        begin_table_pat = re.compile(r"""\\begin\s*\{\s*table\s*\}(?:\s*\[[^\]]*\])?""", re.VERBOSE)
        end_table_pat   = re.compile(r"""\\end\s*\{\s*table\s*\}""", re.VERBOSE)
        centering_pat   = re.compile(r"""\\centering\b""", re.VERBOSE)
        end_tabular_pat = re.compile(r"""\\end\s*\{\s*(?:tabularx?|tabular\*)\s*\}""", re.VERBOSE)


        # ---- 工具函数 ----
        def merge_spans(spans):
            if not spans: return []
            spans = sorted(spans)
            out = [spans[0]]
            for s, e in spans[1:]:
                ls, le = out[-1]
                if s <= le:
                    out[-1] = (ls, max(le, e))
                else:
                    out.append((s, e))
            return out

        def subtract_spans(start, end, masks):
            """从 [start,end) 中减去 masks（已合并且有序），返回剩余区间列表"""
            gaps = []
            cur = start
            for ms, me in masks:
                if me <= cur: 
                    continue
                if ms >= end:
                    break
                if ms > cur:
                    gaps.append((cur, min(ms, end)))
                cur = max(cur, me)
                if cur >= end:
                    break
            if cur < end:
                gaps.append((cur, end))
            return gaps

        results = []
        # 针对每个 table 环境，计算排除区间 → 做差
        for ts, te in self.iter_table_spans(text):
            chunk = text[ts:te]

            # table 内的 begin/end/centering 相对区间
            extra_excludes_rel = []
            extra_excludes_rel += [m.span(0) for m in begin_table_pat.finditer(chunk)]
            extra_excludes_rel += [m.span(0) for m in end_table_pat.finditer(chunk)]
            extra_excludes_rel += [m.span(0) for m in centering_pat.finditer(chunk)]
            extra_excludes_rel += [m.span(0) for m in end_tabular_pat.finditer(chunk)]
            # 映射到全局
            extra_excludes_abs = [(ts + s, ts + e) for s, e in extra_excludes_rel]

            # 收集所有要排除的区间（全局）
            excludes_abs = []
            excludes_abs += align
            excludes_abs += hlines
            excludes_abs += struct
            excludes_abs += caps
            excludes_abs += extra_excludes_abs

            masks = merge_spans([sp for sp in excludes_abs if ts <= sp[0] < te or ts < sp[1] <= te])
            # 做差得到内容区间（仍是全局坐标）
            gaps = subtract_spans(ts, te, masks)

             # 不过滤纯空白，只裁掉左侧的换行符
            for gs, ge in gaps:
                seg = text[gs:ge]
                # 去掉左侧换行符
                left_trim = len(seg) - len(seg.lstrip("\n"))
                # 去掉右侧换行符
                right_trim = len(seg) - len(seg.rstrip("\n"))
                s2 = gs + left_trim
                e2 = ge - right_trim
                if s2 < e2:  # 仍然有内容
                    results.append((s2, e2))

        return merge_spans(results)

    def parse_preamble_spans(self, text):
        """
        提取从 \documentclass{<class_name>} 到 \begin{document}（含二者本身）的代码段。
        - 兼容 \documentclass 可选参数：\documentclass[...]{<class_name>}
        - 兼容空格与换行
        - 若找不到对应的 \begin{document}，则跳过该起点

        Returns:
            List[(start_idx, end_idx)]
        """
        spans = []

        # 起点：\documentclass[...]{class_name}
        preamble_pat = re.compile(
            rf"""
            \\documentclass                 # \documentclass
            \s*                             # 可选空白
            (?:\[[^\]]*\]\s*)?              # 可选方括号参数 [...]
            \{{\s*{re.escape("article")}\s*\}}
            """,
            re.VERBOSE | re.IGNORECASE
        )

        # 终点：\begin{document}
        begin_doc_pat = re.compile(
            r"""
            \\begin
            \s*\{\s*document\s*\}
            """,
            re.VERBOSE | re.IGNORECASE
        )

        for m in preamble_pat.finditer(text):
            start = m.start()  # 包含 \documentclass...
            # 从当前起点之后寻找最近的 \begin{document}
            end_m = begin_doc_pat.search(text, pos=m.end())
            if end_m:
                end = end_m.end()  # 包含 \begin{document}
                spans.append((start, end))
            # 若未找到 \begin{document}，则忽略该起点

        return spans


    @staticmethod
    def spans_to_token_mask_from_offsets(spans, offsets):
        import re
        """
        spans: List[(S,E)] 字符区间
        offsets: 一条样本的 offset_mapping
        返回 (len_i,) float mask
        """
        if isinstance(offsets, torch.Tensor):
            offs = offsets.tolist()
        else:
            offs = offsets
        T = len(offs)
        mask = torch.zeros(T, dtype=torch.float32)
        for t, (s, e) in enumerate(offs):
            for S, E in spans:
                if max(s, S) < min(e, E):  # 区间交集
                    mask[t] = 1.0
                    break
        return mask
    
    def build_component_mask(self, component_masks_list, completion_mask, C, device):
        component_mask = torch.nn.utils.rnn.pad_sequence(component_masks_list, batch_first=True, padding_value=0.0).to(device)
        if component_mask.size(1) > C:
            component_mask = component_mask[:, :C]
        elif component_mask.size(1) < C:
            component_mask = torch.nn.functional.pad(component_mask, (0, C - component_mask.size(1)), value=0.0)
        component_mask = component_mask * completion_mask

        return component_mask

    def validate_component_mask(
        self,
        component_name: str,
        component_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completions_text: list,
        C: int,
        completion_mask: torch.Tensor = None,
        sample_idx: int = 0,
        expected_regex: str = None,
        warn_when_present_but_zero: bool = True,
    ):
        """
        通用的 component mask 校验与调试打印。

        Args:
            component_name (str): 组件名（如 "align", "horizontal_line", "structure", "caption", "content", "vertical_line"...）
            component_mask (Tensor): [B, C'] 的 mask（内部会按 C 做截断/对齐检查，只用于校验与打印，不修改原张量）
            completion_ids (Tensor): [B, L] 生成的 token ids
            completions_text (List[str]): batch 内每条样本对应的原始文本
            C (int): 期望的可视化 / 校验长度（通常与生成长度对齐）
            completion_mask (Tensor, optional): [B, C]，用于检查 mask 未越界到非生成区域
            sample_idx (int): debug 打印用的样本下标（仅在 self.debug_mask=True 生效）
            expected_regex (str, optional): 若提供则覆盖内置的默认 regex
            warn_when_present_but_zero (bool): 文本里命中正则但 mask 全 0 时是否告警
        """

        # ---------- 1) 形状与长度检查 ----------
        B = completion_ids.size(0)
        if component_mask.dim() != 2:
            print(f"[WARN] `{component_name}` mask dim={component_mask.dim()} != 2")
            return

        if component_mask.size(0) != B:
            print(f"[WARN] `{component_name}` mask batch={component_mask.size(0)} != completion_ids batch={B}")

        C_mask = component_mask.size(1)
        if C_mask != C:
            print(f"[INFO] `{component_name}` mask length={C_mask} != expected C={C} (仅提示，不修改原张量)")

        # ---------- 2) 与完成度 mask 的一致性检查（可选） ----------
        if completion_mask is not None:
            if completion_mask.size(0) != B or completion_mask.size(1) != C:
                print(f"[WARN] `completion_mask` shape={tuple(completion_mask.shape)} != (B={B}, C={C})")
            else:
                try:
                    # 只截断到可比对范围
                    C_cmp = min(component_mask.size(1), completion_mask.size(1))
                    bad = (component_mask[:, :C_cmp] * (1 - completion_mask[:, :C_cmp])).abs().sum().item()
                    if bad > 0:
                        print(f"[WARN] `{component_name}` mask has {bad} positions outside completion_mask==1")
                except Exception as e:
                    print(f"[WARN] `{component_name}` vs completion_mask check failed: {e}")

        # ---------- 3) 文本存在性 vs. mask=0 的告警 ----------
        # 为不同组件给一个默认的“应当出现”的正则（粗略启发式）
        default_patterns = {
            "align":           r"\\begin\{tabular\}",
            "vertical_line":   r"\\begin\{tabular\}.*\|",  # 只要对齐串里有竖线
            "horizontal_line": r"(\\hline|\\toprule|\\midrule|\\bottomrule|\\cline|\\cmidrule|\\hhline)",
            "structure":       r"(\\multicolumn|\\multirow|&|\\\\)",
            "caption":         r"\\caption\s*\{",
            # content 面向“除对齐/线型/结构/caption外的文本”，很难给唯一正则，默认不做存在性告警
            "content":         None,
            "preamble":        None,
        }
        pat = expected_regex if expected_regex is not None else default_patterns.get(component_name, None)
        if warn_when_present_but_zero and pat:
            try:
                regex = re.compile(pat, flags=re.DOTALL)
                for i, txt in enumerate(completions_text):
                    if regex.search(txt) and component_mask[i, :C].sum().item() == 0:
                        print(f"[WARN] sample {i} hits `{component_name}` pattern but {component_name}_mask=0 — skipping this reward.")
            except Exception as e:
                print(f"[WARN] `{component_name}` presence regex failed to compile/run: {e}")

        # ---------- 4) 调试打印（与原逻辑一致，受 self.debug_mask 控制） ----------
        if getattr(self, "debug_mask", True):
        #if True:
            i = int(sample_idx)
            if not (0 <= i < B):
                print(f"[WARN] sample_idx={i} out of range (B={B})")
                return

            try:
                enc_i = self.processing_class.tokenizer(
                    [completions_text[i]],
                    return_tensors="pt",
                    padding=True,
                    padding_side="right",
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                # 这里保持你原来的“用生成的 ids 来还原 token 序列”的方式
                tokens = self.processing_class.tokenizer.convert_ids_to_tokens(completion_ids[i].tolist())
                offs = enc_i["offset_mapping"][0].tolist()
                cm = component_mask[i].tolist()

                print(f"=== DEBUG ({component_name} mask) ===")
                print(f"completion_ids[i]: {completion_ids[i]}")
                re_tokenized = enc_i["input_ids"][0]
                print(f'enc_i["input_ids"][0]: {re_tokenized}')

                gen_ids = completion_ids[i].detach().cpu()
                ret_ids_full = enc_i["input_ids"][0].detach().cpu()
                m = min(gen_ids.size(0), ret_ids_full.size(0))
                print(gen_ids[:m] == ret_ids_full[:m])

                print(f"mask_length: {len(cm)}")
                print(f"generate_length: {completion_ids[i].size(0)}")
                print(f"re_tokenized_length: {re_tokenized.size(0)}")

                # 逐 token 可视化（截到 C 以内）
                print(f"--- token/offset/mask for `{component_name}` ---")
                for tkn, (s, e), mk in zip(tokens[:C], offs[:C], cm[:C]):
                    frag = completions_text[i][s:e] if s < e else ""
                    flag = "*" if mk > 0.5 else " "
                    print(f"{flag} tok={repr(tkn):>12s} off=({s:>3},{e:<3}) frag={repr(frag)}")
                print("=======================================")
            except Exception as e:
                print(f"[WARN] debug print for `{component_name}` failed: {e}")

    def _accumulate_component_means(self, comp_scores):
        """微批内累计组件分数，等到 step 边界再统一计算权重。"""
        # comp_scores: (B_total, 7) —— 你已经在前面 gather 过
        self._comp_sum = self._comp_sum + comp_scores.sum(dim=0)
        self._comp_cnt += comp_scores.size(0)

    def _finalize_and_update_component_weights(self):
        """在梯度累积边界（accelerator.sync_gradients=True）时，被调用一次，更新 EMA 权重。"""
        if self._comp_cnt == 0:
            return  # 可能这个step没有有效样本

        means = self._comp_sum / self._comp_cnt  # (7,)
        # Softmax 反向权重（β, a）
        logits = -self._weight_beta * means
        exps = torch.exp(logits - logits.max())  # 数值稳定
        new_w = self._weight_a * exps / exps.sum()  # (7,), 和为 a

        # EMA 融合旧权重
        prev = self.component_reward_weights.to(means.device, dtype=means.dtype)
        updated = self._weight_lam * prev + (1.0 - self._weight_lam) * new_w

        # # （可选）广播到所有 rank，确保一致
        # if torch.distributed.is_initialized():
        #     # 只让 rank0 作为源
        #     if self.accelerator.is_main_process:
        #         src_tensor = updated
        #     else:
        #         src_tensor = torch.zeros_like(updated)
        #     torch.distributed.broadcast(src_tensor, src=0)
        #     updated = src_tensor

        # 保存并切断计算图
        self.component_reward_weights = updated.detach()

        # 只在主进程打印一次
        #if self.accelerator.is_main_process:
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            #应该写入不带日期的path，即不用replace，而是只取log_path的目录，然后拼接上component_weights.jsonl
            #应该先找到当前step的位置，从当前step开始覆盖写入
            #应该把当前self.component_reward_weights赋值给self.state.component_reward_weights
            with open(log_path.replace(".txt", "_component_weights.jsonl"), "a", encoding='utf-8') as f:
                log_obj = {
                        "step": int(self.state.global_step),   # 或者你用的 step 变量
                        "component_weights": self.component_reward_weights.detach().cpu().numpy().tolist()
                    }
                f.write(json.dumps(log_obj, ensure_ascii=False) + "\n")

        # 重置累计器
        self._comp_sum.zero_()
        self._comp_cnt = 0

    # ========= 重写的方法 =========

    def _generate_and_score_completions(self, inputs, model):
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = self.vlm_module.prepare_prompt(self.processing_class, inputs)

        # 处理图像（保持原始逻辑，包括 resize 部分）
        images = []
        for x in inputs:
            if "image" in x:
                imgs = self._get_key_from_inputs(x, "image")
            elif "image_path" in x and x["image_path"] is not None:
                from PIL import Image
                imgs = [Image.open(p) for p in self._get_key_from_inputs(x, "image_path")]
            else:
                imgs = []

            for img in imgs:
                try:
                    w, h = img.size
                    if w < 28 or h < 28:
                        if w < h:
                            new_w = 28
                            new_h = int(h * (28 / w))
                        else:
                            new_h = 28
                            new_w = int(w * (28 / h))
                        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                except:
                    pass
                images.append(img)

        # 准备 prompt 输入
        prompt_inputs = self.vlm_module.prepare_model_inputs(
            self.processing_class,
            prompts_text,
            images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        # 生成
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            generate_returned_result = unwrapped_model.generate(
                **{k: v for k, v in prompt_inputs.items() if k not in self.vlm_module.get_non_generate_params()},
                generation_config=self.generation_config,
            )
            prompt_length = prompt_ids.size(1)
            if not self.vlm_module.is_embeds_input():
                prompt_completion_ids = generate_returned_result
                prompt_ids = prompt_completion_ids[:, :prompt_length]
                completion_ids = prompt_completion_ids[:, prompt_length:]
            else:
                completion_ids = generate_returned_result
                prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        # 构造 completion_mask
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        has_eos = is_eos.any(dim=1)
        eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).to(torch.float32)

        #大小写不敏感
        # 拼 attention_mask
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # 计算 old_per_token_logps / ref_per_token_logps（保持不变）
        multimodal_keywords = self.vlm_module.get_custom_multimodal_keywords()
        multimodal_inputs = {k: prompt_inputs[k] if k in prompt_inputs else None for k in multimodal_keywords}
        with torch.no_grad():
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    model, prompt_completion_ids, attention_mask, **multimodal_inputs
                )
                old_per_token_logps = old_per_token_logps[:, prompt_length - 1:]
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, **multimodal_inputs
                )
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        model, prompt_completion_ids, attention_mask, **multimodal_inputs
                    )
        if ref_per_token_logps is not None:
            ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]

        # 解码 completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        #print(f"len(completions):{len(completions)}")

        completions_text = completions
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # ========= 新增: align_mask =========
        if self.CSPO:
            comp_enc = self.processing_class.tokenizer(
                completions_text,
                return_tensors="pt",
                padding=True,
                padding_side="right",
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            offsets_list = comp_enc["offset_mapping"]  # 注意：用字典取，不再 pop

            C = completion_ids.size(1)
            preamble_masks_list = []
            align_masks_list = []
            horizontal_line_masks_list = []
            structure_masks_list = []
            caption_masks_list = []
            content_masks_list = []
            for txt, off in zip(completions_text, offsets_list):
                preamble_spans = self.parse_preamble_spans(txt)
                mask_i = self.spans_to_token_mask_from_offsets(preamble_spans, off)
                preamble_masks_list.append(mask_i)

                align_spans = self.parse_align_spans(txt)
                mask_i = self.spans_to_token_mask_from_offsets(align_spans, off)
                align_masks_list.append(mask_i)

                horizontal_line_spans = self.parse_horizontal_line_spans(txt)
                mask_i = self.spans_to_token_mask_from_offsets(horizontal_line_spans, off)
                horizontal_line_masks_list.append(mask_i)
                
                structure_spans = self.parse_structure_token_spans(txt)
                mask_i = self.spans_to_token_mask_from_offsets(structure_spans, off)
                structure_masks_list.append(mask_i)

                caption_spans = self.parse_caption_spans(txt)
                mask_i = self.spans_to_token_mask_from_offsets(caption_spans, off)
                caption_masks_list.append(mask_i)

                content_spans = self.parse_text_content_style_spans(txt,align_spans,horizontal_line_spans,structure_spans,caption_spans)
                mask_i = self.spans_to_token_mask_from_offsets(content_spans, off)
                content_masks_list.append(mask_i)



            # align_mask = torch.nn.utils.rnn.pad_sequence(align_masks_list, batch_first=True, padding_value=0.0).to(device)
            # if align_mask.size(1) > C:
            #     align_mask = align_mask[:, :C]
            # elif align_mask.size(1) < C:
            #     align_mask = torch.nn.functional.pad(align_mask, (0, C - align_mask.size(1)), value=0.0)
            # align_mask = align_mask * completion_mask

            preamble_mask = self.build_component_mask(preamble_masks_list, completion_mask, C, device)
            align_mask = self.build_component_mask(align_masks_list, completion_mask, C, device)
            horizontal_line_mask = self.build_component_mask(horizontal_line_masks_list, completion_mask, C, device)
            structure_mask = self.build_component_mask(structure_masks_list, completion_mask, C, device)
            caption_mask = self.build_component_mask(caption_masks_list, completion_mask, C, device)
            content_mask = self.build_component_mask(content_masks_list, completion_mask, C, device)
            vertical_line_mask = align_mask


            # 校验
            if getattr(self, "debug_mask", True):
            #if True:
                print(f"=== Validating component masks (CSPO) ===") 
                self.validate_component_mask("preamble",        preamble_mask,        completion_ids, completions_text, C, completion_mask)
                self.validate_component_mask("align",           align_mask,           completion_ids, completions_text, C, completion_mask)
                self.validate_component_mask("horizontal_line", horizontal_line_mask, completion_ids, completions_text, C, completion_mask)
                self.validate_component_mask("structure",       structure_mask,       completion_ids, completions_text, C, completion_mask)
                self.validate_component_mask("caption",         caption_mask,         completion_ids, completions_text, C, completion_mask)
                self.validate_component_mask("content",         content_mask,         completion_ids, completions_text, C, completion_mask)
                self.validate_component_mask("vertical_line",   vertical_line_mask,   completion_ids, completions_text, C, completion_mask)

        else:
            preamble_mask = completion_mask
            align_mask = completion_mask
            horizontal_line_mask = completion_mask
            structure_mask = completion_mask
            caption_mask = completion_mask
            content_mask = completion_mask
            vertical_line_mask = completion_mask


        # B = completion_ids.size(0)
        # #assert align_mask.shape == (B, C)
        # #assert torch.all((align_mask * (1 - completion_mask)) == 0)
        # for i, txt in enumerate(completions_text):
        #     if r"\begin{tabular}" in txt and align_mask[i].sum().item() == 0:
        #         print(f"[WARN] sample {i} has \\begin{{tabular}} but align_mask=0 — skipping align reward for this sample.")
        # # === 调试打印: token/offset/mask ===
        # if getattr(self, "debug_mask", True):
        #     i = 0
        #     enc_i = self.processing_class.tokenizer(
        #         [completions_text[i]],
        #         return_tensors="pt",
        #         padding=True,
        #         padding_side="right",
        #         add_special_tokens=False,
        #         return_offsets_mapping=True,
        #     )
        #     #tokens = self.processing_class.tokenizer.convert_ids_to_tokens(enc_i["input_ids"][0].tolist())
        #     tokens = self.processing_class.tokenizer.convert_ids_to_tokens(completion_ids[i].tolist())
        #     offs = enc_i["offset_mapping"][0].tolist()
        #     am = align_mask[i].tolist()

        #     print(f"completion_ids[i]:{completion_ids[i]}")
        #     re_tokenized = enc_i["input_ids"][0]
        #     print(f"enc_i[\"input_ids\"][0]:{re_tokenized}")
        #     gen_ids = completion_ids[i].detach().cpu()                 # 统一搬到 CPU
        #     ret_ids_full = enc_i["input_ids"][0].detach().cpu()        # 统一搬到 CPU
        #     print(gen_ids[:min(gen_ids.size(0),ret_ids_full.size(0))]==ret_ids_full[:min(gen_ids.size(0),ret_ids_full.size(0))])

        #     print(f"mask_length:{len(am)}")
        #     print(f"generate_length:{completion_ids[i].size(0)}")
        #     print(f"re_tokenized_length:{re_tokenized.size(0)}")

        #     print("=== DEBUG (align mask) ===")
        #     for tkn, (s, e), mk in zip(tokens[:C], offs[:C], am[:C]):
        #         frag = completions_text[i][s:e] if s < e else ""
        #         flag = "*" if mk > 0.5 else " "
        #         print(f"{flag} tok={repr(tkn):>12s} off=({s:>3},{e:<3}) frag={repr(frag)}")
        #     print("==========================")

        # Compute the rewards
        # No need to duplicate prompts as we're not generating multiple completions per prompt

        # 固定组件顺序，避免“靠记忆”
        _COMPONENT_ORDER = ["align", "vertical_lines", "horizontal_lines", "structure", "caption", "content", "preamble"]

        def _extract_component_scores_tensor(component_rewards_batch, device):
            """
            将 batch 级组件字典列表 -> [B, 6] 张量；列顺序见 _COMPONENT_ORDER
            """
            aliases = {
                "align":           ["alignment", "align"],
                "vertical_lines":  ["vertical lines", "vertical line", "vertical bar", "vertical bars", "vlines", "vline", "vertical"],
                "horizontal_lines":["horizontal lines", "horizontal line", "horizontal bar", "horizontal bars", "hlines", "hline", "horizontal"],
                "structure":       ["structure", "struct"],
                "caption":         ["caption", "captions"],
                "content":         ["text content and style", "content", "text content", "content and style",
                                    "text&style", "text/style", "text-content-and-style"],
                "preamble":       ["preamble"],
            }
            def get_score(d, keys):
                if not isinstance(d, dict):
                    return 0.0
                low = {str(k).lower(): v for k, v in d.items()}
                for k in keys:
                    if k in low and isinstance(low[k], dict) and "score" in low[k]:
                        try:
                            return float(low[k]["score"])
                        except Exception:
                            return 0.0
                return 0.0

            rows = []
            for d in component_rewards_batch:
                rows.append([get_score(d, aliases[name]) for name in _COMPONENT_ORDER])
            return torch.tensor(rows, dtype=torch.float32, device=device)  # [B, 6]


        # ====================== 主流程 ======================
        B_local = len(prompts)
        N_funcs = len(self.reward_funcs)

        rewards_per_func = torch.zeros(B_local, N_funcs, device=device)

        # 仅记录“组件奖励字典”；其数值不写入 rewards_per_func（该列置 0）
        component_rewards_batch = None  # List[dict]（长度=B_local）；若没出现则最终补 [{}]*B_local

        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)

            else:
                # 复用除 prompt/completion 外的字段
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        reward_kwargs[key].extend([example[key]])

                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                reward_func_name = reward_func.__name__


                if reward_func_name == "component_reward":
                    component_rewards_batch = output_reward_func #目前默认权重为1
                    # 组件列不计入矩阵（需要贡献总分的话可在后续把六分量线性组合加入）
                    rewards_per_func[:, i] = 0.0
                else:
                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)


        # 本地将组件字典 -> [B_local, 6]
        if component_rewards_batch is None:
            comp_scores_local = torch.zeros(B_local, 7, dtype=torch.float32, device=device)
        else:
            comp_scores_local = _extract_component_scores_tensor(component_rewards_batch, device)  # [B_local, 6]


        # ===== Gather across processes（仅 Tensor）=====
        #print(f"before gather: rewards_per_func:{rewards_per_func}")
        rewards_per_func = self.accelerator.gather(rewards_per_func)   # [B_total, N_funcs]
        comp_scores      = self.accelerator.gather(comp_scores_local)  # [B_total, 7]
        #print(f"after gather: rewards_per_func:{rewards_per_func}")
        

        compile_idx_list = [i for i, f in enumerate(self.reward_funcs) if "compile" in f.__name__]
        if compile_idx_list:
            compile_idx = compile_idx_list[0]
            compile_results = rewards_per_func[:, compile_idx]
            comp_scores[:, 6] = torch.where(compile_results > 0, 
                                        torch.ones_like(comp_scores[:, 6]), 
                                        comp_scores[:, 6])

        
        self._ensure_component_buffers(device=comp_scores.device, dtype=comp_scores.dtype, a=7.0)
        component_reward_weights_current = self.component_reward_weights

        if os.getenv("WEIGHTED")=="true":
            
            # 1) 先累计（每个微批都调用）
            self._accumulate_component_means(comp_scores)      # <— 不更新，只累计

            # 2) 用“当前缓存的”权重乘上本批 comp_scores（注意：这一步不会改变权重）
            #comp_scores = comp_scores * self.component_reward_weights  # (B_total, 7)

            # 3) 仅在“累积边界”更新一次（Accelerate：True 表示这一次要 step 了）
            if self.accelerator.sync_gradients:
                self._finalize_and_update_component_weights()  # <— 只在边界统一算 softmax+EMA


        # ====== 组件六分量（顺序与 _COMPONENT_ORDER 一致）======
        align_rewards            = comp_scores[:, 0]  # (B_total,)
        vertical_lines_rewards   = comp_scores[:, 1]
        horizontal_lines_rewards = comp_scores[:, 2]
        structure_rewards        = comp_scores[:, 3]
        caption_rewards          = comp_scores[:, 4]
        content_rewards          = comp_scores[:, 5]
        preamble_rewards         = comp_scores[:, 6]

        if "no-style" in os.getenv("EXP_NAME"):
            align_rewards = torch.zeros_like(align_rewards)
            vertical_lines_rewards = torch.zeros_like(vertical_lines_rewards)
            horizontal_lines_rewards = torch.zeros_like(horizontal_lines_rewards)             
        if "no-structure" in os.getenv("EXP_NAME"):
            structure_rewards = torch.zeros_like(structure_rewards)
        if "no-content" in os.getenv("EXP_NAME"):
            caption_rewards = torch.zeros_like(caption_rewards)
            content_rewards = torch.zeros_like(content_rewards)
        if "no-preamble" in os.getenv("EXP_NAME"):
            preamble_rewards = torch.zeros_like(preamble_rewards)

        

        # ====== 轨迹分数：非组件的全局列之和（组件列已置 0）======
        if self.CSPO is False:
            #GRPO直接在计算reward时加权重
            if os.getenv("COMPONENT")=="false":
                traj_rewards = rewards_per_func.sum(dim=1) * (self.reward_weights[0]+self.reward_weights[1])
            else:
                if len(self.reward_weights)>1:
                    traj_rewards = rewards_per_func.sum(dim=1) * (self.reward_weights[0]+self.reward_weights[1]) + (comp_scores * component_reward_weights_current).sum(dim=1) * self.reward_weights[-1]
                elif len(self.reward_weights)==1:
                    traj_rewards = (comp_scores * component_reward_weights_current).sum(dim=1)
        else:
            #CSPO对advantage加权重
            traj_rewards = rewards_per_func.sum(dim=1)

        # ====== 分别做 grouped normalize，得到两路 advantages（标量/样本级）=====
        def _group_norm(x: torch.Tensor) -> torch.Tensor:
            # x: (B_total,) = (batch*num_generations,)
            m = x.view(-1, self.num_generations).mean(dim=1)   # (batch,)
            s = x.view(-1, self.num_generations).std(dim=1)    # (batch,)
            m = m.repeat_interleave(self.num_generations, dim=0)
            s = s.repeat_interleave(self.num_generations, dim=0)
            return (x - m) / (s + 1e-4), s

        #print(f"traj_rewards.size():{traj_rewards.size()}")
        adv_traj_all,  std_adv_traj_all  = _group_norm(traj_rewards)
        #print(f"adv_traj_all.size():{adv_traj_all.size()}")
        #import sys
        #sys.exit(0)

        zero_like = lambda t: torch.zeros_like(t)
        # 每个组件的 advantage：
        if self.CSPO is False:
            adv_align_all,   std_adv_align_all   = zero_like(align_rewards),            zero_like(align_rewards)
            adv_vlines_all,  std_adv_vlines_all  = zero_like(vertical_lines_rewards),   zero_like(vertical_lines_rewards)
            adv_hlines_all,  std_adv_hlines_all  = zero_like(horizontal_lines_rewards), zero_like(horizontal_lines_rewards)
            adv_struct_all,  std_adv_struct_all  = zero_like(structure_rewards),        zero_like(structure_rewards)
            adv_caption_all, std_adv_caption_all = zero_like(caption_rewards),          zero_like(caption_rewards)
            adv_content_all, std_adv_content_all = zero_like(content_rewards),          zero_like(content_rewards)
            adv_preamble_all,std_adv_preamble_all = zero_like(preamble_rewards),        zero_like(preamble_rewards)
        else:
            adv_align_all, std_adv_align_all = _group_norm(align_rewards)
            adv_vlines_all,  std_adv_vlines_all = _group_norm(vertical_lines_rewards)
            adv_hlines_all,  std_adv_hlines_all = _group_norm(horizontal_lines_rewards)
            adv_struct_all,  std_adv_struct_all = _group_norm(structure_rewards)
            adv_caption_all, std_adv_caption_all = _group_norm(caption_rewards)
            adv_content_all, std_adv_content_all = _group_norm(content_rewards)
            adv_preamble_all, std_adv_preamble_all = _group_norm(preamble_rewards)

            

        # 仅取本进程 slice
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )

        if self.CSPO is True:
            if len(self.reward_weights)>1:
                adv_traj  = adv_traj_all[process_slice] * (self.reward_weights[0] + self.reward_weights[1])  # (B_local,)
            elif len(self.reward_weights)==1:
                adv_traj  = adv_traj_all[process_slice]
        else:
            adv_traj  = adv_traj_all[process_slice]
        adv_align = adv_align_all[process_slice] * component_reward_weights_current[0] * self.reward_weights[-1] # (B_local,)
        adv_vlines  = adv_vlines_all[process_slice] * component_reward_weights_current[1] * self.reward_weights[-1]
        adv_hlines  = adv_hlines_all[process_slice] * component_reward_weights_current[2] * self.reward_weights[-1]
        adv_struct  = adv_struct_all[process_slice] * component_reward_weights_current[3] * self.reward_weights[-1]
        adv_caption = adv_caption_all[process_slice] * component_reward_weights_current[4] * self.reward_weights[-1]
        adv_content = adv_content_all[process_slice] * component_reward_weights_current[5] * self.reward_weights[-1]
        adv_preamble = adv_preamble_all[process_slice] * component_reward_weights_current[6] * self.reward_weights[-1]

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__

            if reward_func_name != 'component_reward':
                self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())
                #self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item()/self.reward_weights[i])

        # 记录统计信息（可选）
        self._metrics["reward/global"].append(self.accelerator.gather_for_metrics(traj_rewards).mean().item())
        self._metrics["reward/global_std"].append(self.accelerator.gather_for_metrics(std_adv_traj_all).mean().item())
        
        comp_idx = {
            "align": 0,
            "vlines": 1,
            "hlines": 2,
            "struct": 3,
            "caption": 4,
            "content": 5,
            "preamble": 6,
        }
        
        if self.CSPO:
            for name, reward_vec, std_vec in [
                ("align",   align_rewards,            std_adv_align_all),
                ("vlines",  vertical_lines_rewards,   std_adv_vlines_all),
                ("hlines",  horizontal_lines_rewards, std_adv_hlines_all),
                ("struct",  structure_rewards,        std_adv_struct_all),
                ("caption", caption_rewards,          std_adv_caption_all),
                ("content", content_rewards,          std_adv_content_all),
                ("preamble", preamble_rewards,          std_adv_preamble_all),
            ]:
                idx = comp_idx[name]
                self._metrics[f"reward/{name}"].append(self.accelerator.gather_for_metrics(reward_vec).mean().item())
                self._metrics[f"reward/{name}_std"].append(self.accelerator.gather_for_metrics(std_vec).mean().item())
                # if os.getenv("WEIGHTED")=="true":
                #     self._metrics[f"reward/{name}"].append(self.accelerator.gather_for_metrics(reward_vec).mean().item()/self.reward_weights[-1]/component_reward_weights_current[idx].item())
                #     self._metrics[f"reward/{name}_std"].append(self.accelerator.gather_for_metrics(std_vec).mean().item()/self.reward_weights[-1]/component_reward_weights_current[idx].item())
                # else:
                #     self._metrics[f"reward/{name}"].append(self.accelerator.gather_for_metrics(reward_vec).mean().item()/self.reward_weights[-1])
                #     self._metrics[f"reward/{name}_std"].append(self.accelerator.gather_for_metrics(std_vec).mean().item()/self.reward_weights[-1])
        else:
             for name, reward_vec in [
                ("align",   align_rewards),
                ("vlines",  vertical_lines_rewards),
                ("hlines",  horizontal_lines_rewards),
                ("struct",  structure_rewards),
                ("caption", caption_rewards),
                ("content", content_rewards),
                ("preamble", preamble_rewards),
            ]:
                idx = comp_idx[name]
                self._metrics[f"reward/{name}"].append(self.accelerator.gather_for_metrics(reward_vec).mean().item())
                # if os.getenv("WEIGHTED")=="true":
                #     self._metrics[f"reward/{name}"].append(self.accelerator.gather_for_metrics(reward_vec).mean().item()/self.reward_weights[-1]/component_reward_weights_current[idx].item())
                # else:
                #     self._metrics[f"reward/{name}"].append(self.accelerator.gather_for_metrics(reward_vec).mean().item()/self.reward_weights[-1])
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "multimodal_inputs": multimodal_inputs,
            "preamble_mask": preamble_mask,
            "align_mask": align_mask,          # (B, C)
            "vlines_mask": vertical_line_mask,          # (B, C)
            "hlines_mask": horizontal_line_mask,          # (B, C)
            "struct_mask": structure_mask,          # (B, C)
            "caption_mask": caption_mask,          # (B, C)
            "content_mask": content_mask,          # (B, C)
            "adv_align": adv_align,            # (B,)
            "adv_vlines": adv_vlines,            # (B,)
            "adv_hlines": adv_hlines,            # (B,)
            "adv_struct": adv_struct,            # (B,)
            "adv_caption": adv_caption,            # (B,)
            "adv_content": adv_content,            # (B,)
            "adv_preamble": adv_preamble,            # (B,)
            "adv_traj": adv_traj,              # (B,)
        }


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        # 缓存
        if self.state.global_step % self.num_iterations == 0:
            inputs = self._generate_and_score_completions(inputs, model)
            self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
        else:
            inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
        self._step += 1

        # 取张量
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        multimodal_inputs = inputs["multimodal_inputs"]
        adv_traj   = inputs["adv_traj"].to(torch.float32)       # (B,)


        # 拼接
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # logps & entropy
        per_token_logps, per_token_entropy = self._get_per_token_logps_and_entropy(
            model, input_ids, attention_mask, **multimodal_inputs
        )
        per_token_logps = per_token_logps[:, prompt_ids.size(1) - 1:]  # (B, C)

        # === 计算并记录熵：只统计 completion 段 ===
        with torch.no_grad():
            #per_token_entropy = self._get_per_token_entropy(model, input_ids, attention_mask, **multimodal_inputs)
            # 对齐到 completion 段（与 logps 同样的切法：prompt_ids.size(1)-1 起始）
            per_token_entropy = per_token_entropy[:, prompt_ids.size(1) - 1:]  # (B, |completion|)
            # 用 completion_mask 作加权平均，得到每条样本的平均熵（nats）
            sample_entropy = (per_token_entropy * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1.0)
            # 多进程安全聚合并取均值
            mean_entropy = self.accelerator.gather_for_metrics(sample_entropy).float().mean().item()
            # 可选：转为 bits（除以 ln 2）
            mean_entropy_bits = mean_entropy / 0.6931471805599453

        # 记录到度量（wandb 会自动接收）
        self._metrics["entropy/nats"].append(mean_entropy)
        self._metrics["entropy/bits"].append(mean_entropy_bits)

        # old logps
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()

        # ratio
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        def distribute_advantage(mask: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
            """
            把样本级 advantage 分配到 mask=1 的 token 上
            mask: (B,C)，adv: (B,)
            return: (B,C)
            """
            counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)   # 防止除 0
            return (adv.unsqueeze(1) / counts) * mask
        
        if self.CSPO:
            # ===== masks (B, C) =====
            preamble_mask = inputs["preamble_mask"].to(torch.float32)     # (B, C)
            align_mask    = inputs["align_mask"].to(torch.float32)     # (B, C)
            vlines_mask   = inputs["vlines_mask"].to(torch.float32)    # (B, C)
            hlines_mask   = inputs["hlines_mask"].to(torch.float32)    # (B, C)
            struct_mask   = inputs["struct_mask"].to(torch.float32)    # (B, C)
            caption_mask  = inputs["caption_mask"].to(torch.float32)   # (B, C)
            content_mask  = inputs["content_mask"].to(torch.float32)   # (B, C)

            # ===== advantages (B,) =====
            adv_align     = inputs["adv_align"].to(torch.float32)      # (B,)
            adv_vlines    = inputs["adv_vlines"].to(torch.float32)     # (B,)
            adv_hlines    = inputs["adv_hlines"].to(torch.float32)     # (B,)
            adv_struct    = inputs["adv_struct"].to(torch.float32)     # (B,)
            adv_caption   = inputs["adv_caption"].to(torch.float32)    # (B,)
            adv_content   = inputs["adv_content"].to(torch.float32)    # (B,)
            adv_preamble     = inputs["adv_preamble"].to(torch.float32)      # (B,)

            adv_traj_token = distribute_advantage(completion_mask, adv_traj)    # (B,C)
            adv_align_token = distribute_advantage(align_mask, adv_align)    # (B,C)
            adv_vlines_token = distribute_advantage(vlines_mask, adv_vlines)   # (B,C)
            adv_hlines_token = distribute_advantage(hlines_mask, adv_hlines)   # (B,C)
            adv_struct_token = distribute_advantage(struct_mask, adv_struct)   # (B,C)
            adv_caption_token = distribute_advantage(caption_mask, adv_caption)  # (B,C)
            adv_content_token = distribute_advantage(content_mask, adv_content)  # (B,C)
            adv_preamble_token = distribute_advantage(preamble_mask, adv_preamble)    # (B,C)

            # 合并两路
            if len(self.reward_weights)>1:
                advantages_token = (
                    adv_traj_token
                    + adv_align_token
                    + adv_vlines_token
                    + adv_hlines_token
                    + adv_struct_token
                    + adv_caption_token
                    + adv_content_token
                    + adv_preamble_token
                )  # (B,C)
            else:
                advantages_token = (
                    adv_align_token
                    + adv_vlines_token
                    + adv_hlines_token
                    + adv_struct_token
                    + adv_caption_token
                    + adv_content_token
                    + adv_preamble_token
                )  # (B,C)
        else:
            adv_traj_token = distribute_advantage(completion_mask, adv_traj)    # (B,C)
            advantages_token = adv_traj_token

        # PPO 损失
        per_token_loss1 = coef_1 * advantages_token
        per_token_loss2 = coef_2 * advantages_token
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # KL（若启用，仍对所有有效 token 计算）
        if self.beta > 0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            per_sample_kl = (per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1.0)
            per_sample_loss = per_token_loss.sum(dim=1)
            per_sample_loss = per_sample_loss + self.beta * per_sample_kl

            # Log KL divergence
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1.0)).mean()
            self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        loss = per_sample_loss.mean()

        # Log clip ratio
        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())

        return loss


    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))

    def _get_train_sampler(self) -> Sampler:
        """Returns a sampler that ensures proper data sampling for GRPO training."""
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        
        return RepeatRandomSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations,
            seed=self.args.seed,
        )

    # def _get_train_sampler(self) -> Sampler:
    #     """
    #     正确的做法：
    #     - batch_size 只用“本地”的：per_device * grad_accum
    #     - 采样按 rank 做全局分片，避免重复
    #     - num_generations 保持用 mini_repeat_count 做组内重复
    #     """
    #     # 本地（每个 rank）有效 batch 大小（不乘 world_size）
    #     local_effective_bsz = (
    #         self.args.per_device_train_batch_size
    #         * self.args.gradient_accumulation_steps
    #     )
    #     if local_effective_bsz % self.num_generations != 0:
    #         raise ValueError(
    #             f"per_device_train_batch_size({self.args.per_device_train_batch_size}) * "
    #             f"gradient_accumulation_steps({self.args.gradient_accumulation_steps}) "
    #             f"must be divisible by num_generations({self.num_generations})."
    #         )

    #     return ShardedRepeatRandomSampler(
    #         data_source=self.train_dataset,
    #         mini_repeat_count=self.num_generations,
    #         batch_size=local_effective_bsz // self.num_generations,
    #         repeat_count=self.num_iterations,
    #         seed=self.args.seed,
    #         world_size=self.accelerator.num_processes,   # ← 全局并行度
    #         rank=self.accelerator.process_index,         # ← 本进程序号
    #     )


    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        """Returns a sampler for evaluation."""
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )
