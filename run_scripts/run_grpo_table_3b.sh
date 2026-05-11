# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export REPO_HOME="${PROJECT_ROOT}"
echo "REPO_HOME: $REPO_HOME"
# on remote
data_paths="data/grpo_trainset5000.jsonl" 
image_folders="data/tabletex"
model_path="LLaMA-Factory/saves/qwen2_5_vl_sft_3b/checkpoint-156"
is_reward_customized_from_vlm_module=True
echo "data_paths: $data_paths"
echo "image_folders: $image_folders"

export EXP_NAME="3B-lora-sft156-lr5e-6-GRPO-lr1e-5" # TODO: change this to your own experiment name
TASK_TYPE="table"
cd ${REPO_HOME}/src/open-r1-multimodal

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL

export CRITIC_MODEL="gpt-4o-2" #"gpt-4o-0806-global" 

export METHOD="GRPO"

export WEIGHTED="false"

export TRAJECTORY="false"

export COMPONENT="false"

# create the run directory and log file
mkdir -p ${REPO_HOME}/runs/${EXP_NAME}/log
export LOG_PATH="${REPO_HOME}/runs/${EXP_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"
# MAX_STEPS=1200 # TODO: change this to your own max steps


# export WANDB_DISABLED=true
#export CUDA_VISIBLE_DEVICES=1,2
torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12349" \
  src/open_r1/table_grpo.py \
    --use_vllm False \
    --output_dir "model/3B-lora-GRPO" \
    --resume_from_checkpoint True \
    --model_name_or_path $model_path \
    --data_file_paths $data_paths \
    --image_folders $image_folders \
    --is_reward_customized_from_vlm_module $is_reward_customized_from_vlm_module \
    --task_type $TASK_TYPE \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --num_train_epochs 2 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --run_name ${EXP_NAME} \
    --data_seed 42 \
    --save_steps 10 \
    --save_total_limit 3 \
    --num_generations 8 \
    --max_completion_length 4096 \
    --reward_funcs accuracy compile\
    --reward_weights 0.5 0.5\
    --beta 0.04 \
    --report_to wandb \
    --dataset-name this_is_not_used \
    --deepspeed ${REPO_HOME}/src/open-r1-multimodal/local_scripts/zero2.json \
    --learning_rate 1e-5 \
    --use_peft true \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --freeze_vision_modules true

echo "Training completed for ${EXP_NAME}"
