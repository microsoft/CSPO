# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import random
import datetime
from pathlib import Path
from collections import defaultdict
import torch.distributed as dist
import argparse
import sys
import warnings

from open_r1.vlm_modules.teds_utils import compute_teds_score
from compile_utils import process_tex_file
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank) 
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=720000)
    )
    return local_rank, dist.get_world_size(), dist.get_rank()

def compile_reward(completions, **kwargs):
        import os
        from datetime import datetime
        import time

        contents = [completion[0]["content"] for completion in completions]

        def extract_article_tex(text):
            """
            从原始文本中提取以 \\documentclass{article} 开始、以 \\end{document} 结束的 LaTeX 文档片段。
            """
            import re

            ARTICLE_TEX_PATTERN = re.compile(
                r"""
                \\documentclass             # \documentclass
                \s*                         # 可有空白
                (?:\[[^\]]*\]\s*)?          # 可选 [options]
                \{ \s* article \s* \}       # {article}，允许内部空白
                .*?                         # 中间所有内容（非贪婪）
                \\end \s* \{ \s* document \s* \}  # \end{document}，允许空白
                """,
                re.IGNORECASE | re.DOTALL | re.VERBOSE
            )

            cleaned = re.sub(r"```+\s*(?:latex|tex)?\s*", "", text, flags=re.IGNORECASE)

            m = ARTICLE_TEX_PATTERN.search(cleaned)
            return m.group(0).strip() if m else None
        
        def save_tex_file(latex_code, output_dir='compile', image_filename='default.png'):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(image_filename)[0]
            tex_path = os.path.join(output_dir, f"{base_name}.tex")
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(latex_code)
        
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        for content in contents:
            #pred_latex = extract_latex_from_response(content)
            pred_latex = extract_article_tex(content)
            if pred_latex is None:
                reward = 0.0    #如果格式不匹配（format_acc遗漏了）则奖励为0
            else:
                rank = int(os.getenv("RANK", "0"))   # torch DDP 会自动设置 RANK 环境变量
                ts = int(time.time() * 1e6)          # 微秒级时间戳
                unique_name = f"sample_r{rank}_{ts}.png"

                save_tex_file(latex_code=pred_latex, output_dir=f"compile/{rank}", image_filename=unique_name)
                reward = process_tex_file(output_dir=f"compile/{rank}", image_filename=unique_name)

            rewards.append(reward)
        return rewards

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="data/testset4000.jsonl")
parser.add_argument("--image", type=str, default="data/tabletex")
parser.add_argument("--model_path", type=str, default="model/Qwen2.5-VL-3B-CSPO")
parser.add_argument("--run_name", type=str, default="CSPO")
parser.add_argument("--output_dir", type=str, default="src/eval/output")
args = parser.parse_args()

local_rank, world_size, rank = setup_distributed()
device = f"cuda:{local_rank}"
print(f"Process {rank} using {device}")

# Config
BSZ = 4
MODEL_PATH = args.model_path
ds_path = args.dataset
RUN_NAME = args.run_name
IMAGE_DIR = args.image
OUTPUT_DIR = args.output_dir
OUTPUT_PATH = f"{OUTPUT_DIR}/{RUN_NAME}"
QUESTION_TEMPLATE = "Please generate complete LaTeX code for the table in the image, including the table body and the full preamble."
partial_output_path = f"{OUTPUT_DIR}/{RUN_NAME}/partial_rank{rank}.jsonl"
progress_path = f"{OUTPUT_DIR}/{RUN_NAME}/progress_rank{rank}.txt"

# Ensure output directory exists
Path(os.path.dirname(partial_output_path)).mkdir(parents=True, exist_ok=True)

# Load model and processor
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": local_rank}
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# Load dataset and split
data = [json.loads(line) for line in open(ds_path, "r", encoding="utf-8")]
random.seed(42)
#data = data[:2000]  # 默认采样2000条

per_rank_data = len(data) // world_size
start_idx = rank * per_rank_data
end_idx = start_idx + per_rank_data if rank < world_size - 1 else len(data)
rank_data = data[start_idx:end_idx]

# Identify main_rank (for tqdm)
output_lens = [None] * world_size
dist.all_gather_object(output_lens, len(rank_data))
main_rank = output_lens.index(max(output_lens))
dist.barrier()

# Load completed indices
completed_indices = set()
if os.path.exists(progress_path):
    with open(progress_path, "r") as f:
        completed_indices = set(int(line.strip()) for line in f if line.strip().isdigit())

# Start inference
messages = []
for x in rank_data:
    img_path = os.path.join(IMAGE_DIR, x["image_path"])
    messages.append([{
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{img_path}"},
            {"type": "text", "text": QUESTION_TEMPLATE}
        ]
    }])

processed_count = 0

for i in tqdm(range(0, len(messages), BSZ), disable=rank != main_rank):
    batch_messages = messages[i:i + BSZ]

    # 过滤掉已完成的样本
    valid_batch = []
    valid_indices = []
    for j, msg in enumerate(batch_messages):
        global_idx = i + j
        if global_idx not in completed_indices:
            valid_batch.append(msg)
            valid_indices.append(global_idx)

    if len(valid_batch) == 0:
        continue

    # 准备输入
    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in valid_batch]
    image_inputs, video_inputs = process_vision_info(valid_batch)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        padding_side="left",
        return_tensors="pt",
    ).to(device)

    # 生成
    generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=8192, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    outputs = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # 保存结果
    for idx, output in zip(valid_indices, outputs):
        input_example = rank_data[idx]
        ground_truth = input_example.get('tex_code', '')
        subject = input_example.get('subject', 'unknown')

        try:
            _, teds_acc = compute_teds_score(output, ground_truth)
            compile_acc = compile_reward([[{"content":output}]])[0]
        except Exception as e:
            print(f"[Rank {rank}] Error at index {idx}: {e}")
            teds_acc = -1

        result = {
            'index': idx,
            'image': input_example['image_path'],
            'ground_truth': ground_truth,
            'model_output': output,
            'TEDS acc': teds_acc,
            "Compile acc": compile_acc,
            'subject': subject
        }
        with open(partial_output_path, "a") as fout, open(progress_path, "a") as pf:
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            pf.write(f"{idx}\n")

        processed_count += 1

print(f"[Rank {rank}] Finished processing {processed_count} examples.")
dist.barrier()
dist.destroy_process_group()
