import os
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, default="CSPO")
parser.add_argument("--output_dir", type=str, default="src/eval/output")
args = parser.parse_args()

# Config
RUN_NAME = args.run_name
OUTPUT_DIR = args.output_dir
OUTPUT_PATH = f"{OUTPUT_DIR}/{RUN_NAME}"

output_file = os.path.join(OUTPUT_PATH, "merged.jsonl")

# 如果目录存在 merged.jsonl 文件，写入数据
with open(output_file, 'w', encoding='utf-8') as fout:
    for filename in sorted(os.listdir(OUTPUT_PATH)):
        if filename.endswith(".jsonl") and filename != "merged.jsonl":  # 避免合并自己
            filepath = os.path.join(OUTPUT_PATH, filename)
            with open(filepath, 'r', encoding='utf-8') as fin:
                for line in fin:
                    fout.write(line)

print(f"合并完成: {output_file}")


def compute_compile_acc_mean(jsonl_path, key="Compile acc"):
    total = 0.0
    count = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                # 如果有坏行，直接跳过
                continue

            if key not in item:
                continue

            value = item[key]
            try:
                value = float(value)
            except (TypeError, ValueError):
                # 无法转换为数值就跳过
                continue

            total += value
            count += 1

    if count == 0:
        return None  # 没有有效数据

    return total / count * 100, count


teds_acc, teds_count = compute_compile_acc_mean(output_file, key="TEDS acc")
print(f"Average TEDS acc: {teds_acc:.2f}%, valid samples: {teds_count}")
compile_acc, compile_count = compute_compile_acc_mean(output_file, key="Compile acc")
print(f"Average Compile acc: {compile_acc:.2f}%, valid samples: {compile_count}")

#写入文件
with open(os.path.join(OUTPUT_PATH, "result.txt"), "w", encoding="utf-8") as f:
    f.write(f"Average TEDS acc: {teds_acc:.2f}%, valid samples: {teds_count}\n")
    f.write(f"Average Compile acc: {compile_acc:.2f}%, valid samples: {compile_count}\n")