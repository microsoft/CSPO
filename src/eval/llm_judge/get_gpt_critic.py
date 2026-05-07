import json
import os

WORKING_HOME_DIR="src/eval/llm_judge/result/CSPO"

merged_jsonl_path = os.path.join(WORKING_HOME_DIR, "merged.jsonl")

if not os.path.exists(merged_jsonl_path):
    with open(merged_jsonl_path, 'w', encoding='utf-8') as fout:
        for filename in sorted(os.listdir(WORKING_HOME_DIR)):
            if filename.endswith(".jsonl") and filename != "merged.jsonl":  # 避免合并自己
                filepath = os.path.join(WORKING_HOME_DIR, filename)
                with open(filepath, 'r', encoding='utf-8') as fin:
                    for line in fin:
                        fout.write(line)

    print(f"合并完成: {merged_jsonl_path}")

# 初始化统计变量
score_sums = {"Content": 0, "Structure": 0, "Line": 0, "Alignment": 0, "Cell Style": 0}
count = 0

# 组合统计
count_content_structure_1 = 0
count_line_align_style_1 = 0
count_all_five_1 = 0

output_dir = os.path.dirname(os.path.abspath(merged_jsonl_path))
output_path = os.path.join(output_dir, "score_avg.txt")

# 读取 JSONL 文件并累计得分
with open(merged_jsonl_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        score = data.get("score", {})

        # 如果 score 中任何一个需要的 key 是 None，则跳过该行
        if any(score.get(key) is None for key in score_sums):
            continue

        # 累加总分
        for key in score_sums:
            score_sums[key] += score.get(key, 0)

        # 组合判断
        if score.get("Content") == 1 and score.get("Structure") == 1:
            count_content_structure_1 += 1

        if (
            score.get("Line") == 1 and
            score.get("Alignment") == 1 and
            score.get("Cell Style") == 1
        ):
            count_line_align_style_1 += 1

        if all(score.get(key) == 1 for key in score_sums):
            count_all_five_1 += 1

        count += 1

# 写入统计结果到文件
with open(output_path, 'w') as out_file:
    if count > 0:
        out_file.write("平均分数：\n")
        for key in score_sums:
            avg = score_sums[key] / count
            out_file.write(f"{key}: {avg:.4f}\n")

        out_file.write("\n组合得分为1的比例：\n")
        out_file.write(f"Content + Structure 都为1: {count_content_structure_1}/{count} = {count_content_structure_1 / count:.4f}\n")
        out_file.write(f"Line + Alignment + Cell Style 都为1: {count_line_align_style_1}/{count} = {count_line_align_style_1 / count:.4f}\n")
        out_file.write(f"全部5项都为1: {count_all_five_1}/{count} = {count_all_five_1 / count:.4f}\n")
    else:
        out_file.write("未找到有效数据。\n")

print(f"wrote to: {output_path}")