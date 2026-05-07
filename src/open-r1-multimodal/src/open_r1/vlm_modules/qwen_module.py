from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch

from open_r1.vlm_modules.vlm_module import VLMBaseModule

import os
import subprocess
import re
from .teds_utils import compute_table_similarity, compute_teds_score
from .compile_utils import process_tex_file
from open_r1.vlm_modules.agents.ReviewAgent import ReviewAgent
from open_r1.vlm_modules.agents.GradedReviewAgent import GradedReviewAgent
import time

class Qwen2VLModule(VLMBaseModule):
    def __init__(self):
        super().__init__()

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id, model_init_kwargs):
        return Qwen2_5_VLForConditionalGeneration
        # if (
        #     model_id == "Qwen/Qwen-VL-Chat"
        #     or model_id.startswith("Qwen2.5-VL")
        #     or "qwen2_5_vl" in model_id.lower()  # ✅ 添加这个
        # ):
        #     return Qwen2_5_VLForConditionalGeneration
        # else:
        #     raise ValueError(f"Unsupported model: {model_id}")
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        return AutoProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_grid_thw']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return [('image_processor', 'max_pixels'), ('image_processor', 'min_pixels')]
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        # FIXME
        # This could only process pure-multimodal or pure-text inputs
        if len(images) > 0:
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        else:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        return prompt_inputs
    
    @staticmethod
    def get_question_template(task_type: str):
        match task_type:
            case "table":
                return "Please generate complete LaTeX code for the table in the image, including the table body and the full preamble."
                #return "Please generate LaTeX code for the table shown in the image, including the preamble and the table body. Ensure that the appearance of the generated table matches the image exactly. First, output the thinking process in <think> </think> tags: describing (1) table structure, (2) character style, (3) line type, and other details; analyze the required LaTeX macros and packages to be included.  Then, output the final answer in <answer> </answer> tags."
                #return "Please generate LaTeX code for the table shown in the image, including the preamble and the table body. First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
                #return "Please generate LaTeX code for the table in the image."
            case "rec":
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
            case "ic":
                return "{Question} First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> json format answer here </answer>"
            case "odLength":
                SYSTEM_PROMPT = (
                    #"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
                    "First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
                    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
                    "<think> reasoning process here </think><answer> answer here </answer>"
                )
                return SYSTEM_PROMPT + '\n' + "{Question}"
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
            
    @staticmethod
    def format_reward_rec(completions, **kwargs):
        """Check if the Qwen model output matches a specific format."""
        import re
        import os
        from datetime import datetime
        pattern = r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]

        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Format reward -------------\n")
                for content, match in zip(completion_contents, matches):
                    f.write(f"Content: {content}\n")
                    f.write(f"Has format: {bool(match)}\n")
        return [1.0 if match else 0.0 for match in matches]
    
    
    @staticmethod
    def iou_reward(completions, solution, **kwargs):
        """Calculate IoU reward between predicted bounding box from Qwen model and ground truth bounding box."""
        import re
        import os
        from datetime import datetime
        import json
        def iou(box1, box2):
            inter_x1 = max(box1[0], box2[0])
            inter_y1 = max(box1[1], box2[1])
            inter_x2 = min(box1[2]-1, box2[2]-1)
            inter_y2 = min(box1[3]-1, box2[3]-1)
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
            else:
                inter = 0
            union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
            return float(inter)/union
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
        for content, sol in zip(contents, solution):
            sol = re.findall(answer_tag_pattern, sol, re.DOTALL)[-1]
            sol = json.loads(sol.strip())
            reward = 0.0
            # Try symbolic verification first
            try:
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()
                    bbox_match = re.search(bbox_pattern, content_answer)
                    if bbox_match:
                        bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                        # if iou(bbox, sol) > 0.5:
                        #     reward = 1.0
                        reward = iou(bbox, sol)
            except Exception:
                pass  # Continue to next verification method if this fails
                    
            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
                problem = kwargs.get("problem")[0]
                if reward <= 1.0:  # this condition can be changed for debug
                    with open(log_path, "a", encoding='utf-8') as f:
                        f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                        f.write(f"image_path: {image_path}\n")
                        f.write(f"problem: {problem}\n")
                        f.write(f"Content: {content}\n")
                        f.write(f"Solution: {sol}\n") 
        return rewards
    
    @staticmethod
    def compile_reward(completions, **kwargs):
        """Check if the Qwen model output can be successfully compiled to pdf."""
        import os
        from datetime import datetime

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
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                #将文件名加上compile后缀，以区分不同reward的日志
                log_path = log_path.replace(".txt", "_compile.txt")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
                if reward <= 1.0:  # this condition can be changed for debug
                    with open(log_path, "a", encoding='utf-8') as f:
                        f.write(f"------------- {current_time} Compile reward: {reward} -------------\n")
                        f.write(f"image_path: {image_path}\n")
                        f.write(f"Content: {pred_latex}\n")
        return rewards
    
    @staticmethod
    def format_reward_table(completions, **kwargs):
        """Check if the Qwen model output matches a specific format."""
        import re
        import os
        from datetime import datetime

        contents = [completion[0]["content"] for completion in completions]

        #pattern = r"<think>.*?</think>\s*<answer>\s*```(?:[\s\S]*?)```\s*</answer>"
        pattern = r"<think>.*?</think>\s*<answer>\s*(?:```)?([\s\S]*?)(?:```)?\s*</answer>"
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in contents]

        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
                image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
                f.write(f"------------- {current_time} Format reward -------------\n")
                for content, match in zip(contents, matches):
                    f.write(f"image_path: {image_path}\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Has format: {bool(match)}\n")
        return [1.0 if match else 0.0 for match in matches]
    
    @staticmethod
    def TEDS_reward(completions, solution, **kwargs):
        """Calculate TEDS reward between predicted latex code from Qwen model and ground truth latex code."""
        import os
        from datetime import datetime

        contents = [completion[0]["content"] for completion in completions]

        def extract_latex_from_response(response: str) -> str:
            #matches = re.findall(r"<answer>\s*```(?:latex)?\s*(.*?)\s*```\s*</answer>", response, re.DOTALL)
            matches = re.findall(r"<answer>\s*(?:```(?:latex)?\s*)?(.*?)(?:\s*```)?\s*</answer>", response, re.DOTALL)
            return matches[-1].strip() if matches else None

        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        for content, gt_latex in zip(contents, solution):
            # Try symbolic verification first
            #pred_latex = extract_latex_from_response(content)
            pred_latex=content
            if pred_latex is None:
                reward = 0.0    #如果格式不匹配（format_acc遗漏了）则奖励为0
            else:
                #pred_latex = repr(pred_latex)
                edit_distance, reward = compute_teds_score(pred_latex, gt_latex)
            
            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
                #if reward <= 1.0:  # this condition can be changed for debug
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"image_path: {image_path}\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"gt_latex: {gt_latex}\n") 
        return rewards
    
    @staticmethod
    def component_reward(completions, solution, **kwargs):
        """Calculate TEDS reward between predicted latex code from Qwen model and ground truth latex code."""
        import os
        from datetime import datetime

        contents = [completion[0]["content"] for completion in completions]


        def extract_tables(tex_content: str):
            """
            从 LaTeX 文本中提取第一个表格环境的内容。
            优先匹配 \begin{table}...\end{table}，
            如果没有 \end{table}，则退化为匹配 \begin{table} 及之后的所有内容。
            """
            # 优先匹配完整 table 环境
            table_pattern = re.compile(r'\\begin{table}.*?\\end{table}', re.DOTALL)
            match = table_pattern.search(tex_content)
            if match:
                return match.group(0)

            # 如果没有匹配到完整的 \end{table}，则匹配从 \begin{table} 开始到结尾
            fallback_pattern = re.compile(r'\\begin{table}.*', re.DOTALL)
            match_fallback = fallback_pattern.search(tex_content)
            if match_fallback:
                return match_fallback.group(0)

            return ""


        def extract_modified_code(text):
            tag = "<answer>"
            idx = text.find(tag)
            if idx == -1:
                idx = 0  # 没有找到标签
            modified_code = extract_tables(text[idx + len(tag):].strip())
            return modified_code

        def remove_pagestyle_empty(content: str) -> str:
            """
            删除 content 中的 \pagestyle{empty} 及其后紧跟的空白字符
            """
            pattern = r'\\pagestyle\{empty\}\s*'
            return re.sub(pattern, '', content)
        
        def clean_adjustbox(text):
            # 定义需要移除的模式
            # 1. \begin{adjustbox}{...} - 考虑到参数可能变化，使用正则匹配整个括号内容
            # 2. \end{adjustbox}
            # 3. \usepackage{adjustbox}
            
            patterns = [
                r'\\begin\{adjustbox\}\{[^}]*\}\n?',  # 如果 \begin{adjustbox} 后面紧跟换行，也一并移除
                r'\\end\{adjustbox\}\n?',
                r'\\usepackage\{adjustbox\}\n?'
            ]
            
            for pattern in patterns:
                # 使用 re.sub 将匹配到的内容替换为空字符串
                text = re.sub(pattern, '', text)
            
            # 去除因删除代码可能留下的多余空行（可选）
            text = re.sub(r'\n\s*\n', '\n\n', text).strip()
            
            return text


        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        for content, gt_latex in zip(contents, solution):
            # Try symbolic verification first
            #pred_latex = extract_tables(content)
            pred_latex = remove_pagestyle_empty(content)
            gt_latex = remove_pagestyle_empty(gt_latex)
            
            pred_latex = clean_adjustbox(pred_latex)
            gt_latex = clean_adjustbox(gt_latex)

            score = {}
            agents = {
                "ReviewAgent": ReviewAgent("gpt-4o"),
            }
            review_agent = agents["ReviewAgent"]
            for i in range(0,5):
                raw_response, score = review_agent.act(
                    gt_code=gt_latex, pred_code=pred_latex)
                if score is not None and score != {} and raw_response != "" and raw_response is not None:
                    break
            #if "No difference." in content and gt_latex["have difference"] is True:
            #    reward = reward * 0.5

            #print(f"teds:{teds}, ori_teds:{ori_teds}")
            if not review_agent.is_valid_score_dict(score):
                score = review_agent.coerce_score_dict(score)
            
            rewards.append(score)

            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                #将文件名加上component后缀，以区分不同reward的日志
                log_path = log_path.replace(".txt", "_component.txt")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} component reward:-------------\n")
                    f.write(f"image_path: {image_path}\n")
                    f.write(f"score: {score}\n")
                    f.write(f"pred_latex: {pred_latex}\n")
                    f.write(f"gt_latex: {gt_latex}\n") 
        return rewards

    @staticmethod
    def select_reward_func(func: str, task_type: str, threshold: float = None):
        def discretized_teds_reward(threshold: float):
            raw_func = Qwen2VLModule.TEDS_reward
            def wrapped_teds_reward(*args, **kwargs):
                raw_rewards = raw_func(*args, **kwargs)
                return raw_rewards, [1.0 if r >= threshold else 0.0 for r in raw_rewards]
            return wrapped_teds_reward

        if func == "accuracy":
            match task_type:
                case "table":
                    if threshold is not None:
                        return discretized_teds_reward(threshold)
                    return Qwen2VLModule.TEDS_reward
                case "rec":
                    return Qwen2VLModule.iou_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")

        elif func == "format":
            match task_type:
                case "table":
                    return Qwen2VLModule.format_reward_table
                case "rec":
                    return Qwen2VLModule.format_reward_rec
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")

        elif func == "compile":
            match task_type:
                case "table":
                    return Qwen2VLModule.compile_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "component":
            match task_type:
                case "table":
                    return Qwen2VLModule.component_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        else:
            raise ValueError(f"Unsupported reward function: {func}")
