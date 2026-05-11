# Portions of this file are adapted from METAL, which declares the MIT License in its README.
# This version has been substantially modified for CSPO.
# Modifications Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing

## Metal agents
from agents.CriticAgent import CriticAgent

from agents.table_utils.teds_utils import compute_teds_score

from agents.utils import *

import torch
import gc

# utils
def setup_logger(output_path, output_suffix):
    logger = logging.getLogger(output_suffix)
    logger.setLevel(logging.INFO)

    log_file = os.path.join(output_path, f"{output_suffix}.log")
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def format_log_critique(message):
    msg = "             >> =======================================================\n"
    for line in message.split("\n"):
        msg += f"                                               {line}\n"
    msg += "                                             >> ======================================================="
    return msg

def extract_tables(tex_content):
    """
    从 LaTeX 文本中提取所有表格环境的内容
    """
    table_pattern = re.compile(r'\\begin{table}.*?\\end{table}', re.DOTALL)
    match = table_pattern.search(tex_content)
    return match.group(0) if match else None

def extract_tabular(tex_content):
    """
    从 LaTeX 文本中提取所有表格环境的内容
    """
    #tex_content = remove_commented_sections(tex_content)  # 移除被注释的内容
    table_pattern = re.compile(r'\\begin{tabular}.*?\\end{tabular}', re.DOTALL)
    match = table_pattern.search(tex_content)
    return match.group(0) if match else None

def extract_modified_code(text):
    tag = "<answer>"
    idx = text.find(tag)
    if idx == -1:
        idx = 0  # 没有找到标签
    modified_code = extract_tables(text[idx + len(tag):].strip())
    return modified_code

def remove_at_braces(s: str) -> str:
    """去除字符串中的 @{ }"""
    if not isinstance(s, str):
        return s
    return s.replace("@{}", "")

def process_data_table(data_item, config, model, max_iterations, max_compile_iterations, need_regions_correction="True"):
    log = []
    
    output_suffix = os.path.splitext(os.path.basename(data_item["image"]))[0]
    output_path = f"{config['working_dir']}{output_suffix}/"
    os.makedirs(output_path, exist_ok=True)
    result_dir = config['result_dir']
    image_path = data_item["image"]
    
    logger = setup_logger(output_path, output_suffix)
    
    try:
        logger.info(f"## Processing data {output_suffix} ##")
        
        ground_truth_code = data_item["ground_truth"]
        
        agents = {
            "CriticAgent": CriticAgent(model, **config.get('agent_kwargs', {})),
        }
        critic_agent = agents["CriticAgent"]
        generated_code = data_item["model_output"]
        if r"\begin{table}" in ground_truth_code and "no_caption" not in image_path:
            generated_code = extract_tables(generated_code)
        else:
            generated_code = extract_tabular(generated_code)
        
        if "no_caption" in image_path:
            ground_truth_code = extract_tabular(ground_truth_code)

        generated_code = remove_at_braces(generated_code)
        ground_truth_code = remove_at_braces(ground_truth_code)

        for i in range(0,10):
            raw_response, score = critic_agent.act(
                gt_code=ground_truth_code, pred_code=generated_code)
            valid_score = False
            if raw_response is not None and raw_response!="" and score!={}:
                # 定义需要的键
                required_keys = {"Content", "Structure", "Line", "Alignment", "Cell Style"}
                # 这里的 score 是你的字典
                if (all(k in score for k in required_keys) and 
                    all(v in [0, 1] for v in score.values())):
                    valid_score = True
            if valid_score:
                break
            else:       
                with open("debug/log.txt", "a", encoding="utf-8") as f:
                    f.write("image_path:\n")
                    f.write(image_path)
                    f.write("ground_truth_code:\n")
                    f.write(ground_truth_code + "\n\n")
                    f.write("generated_code:\n")
                    f.write(generated_code + "\n\n")
                    f.write("raw_response:\n")
                    f.write(str(raw_response) + "\n\n")
                    f.write("=" * 80 + "\n\n")  # 分隔符，方便阅读
        
        gc.collect()
        torch.cuda.empty_cache()

        
        with open(f"{output_path}{output_suffix}_log.json", "w") as f:
                    json.dump(log, f, indent=2)
        result = {
            "image_path": image_path,
            "TEDS acc":data_item["TEDS acc"],
            "ground_truth_code":ground_truth_code,
            "generated_code": generated_code,
            "raw_response": raw_response,
            "score": score
        }
        
        # 追加写入 jsonl 文件
        with open(result_dir, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        return {"data": output_suffix, "log": log, "flag":"success"}
    
    except Exception as e:
        logger.error(f"## Failed to process data {output_suffix}: {e} ##", exc_info=True)
        log.append({
            "data": output_suffix,
            "agent": "Error",
            "error": str(e),
        })
        return {"data": output_suffix, "log": log, "flag":"error"}    
    



class MultiAgentSystem:
    def __init__(self, model, data, n_process, config, **kwargs):
        self.model = model
        self.data = data
        self.n_process = n_process
        self.config = config
        self.config.update(kwargs)
        self.system = config.get("agent_kwargs",{}).get("system", "Metal")
        self.max_iterations = config["max_iterations"]
        self.max_compile_iterations  = config["max_compile_iterations"]
        self.logs = []
        
        self.main_logger = logging.getLogger("MultiAgentSystem")
        self.main_logger.setLevel(logging.INFO)
        main_log_file = os.path.join(self.config["working_dir"], f"{self.config['exp_name']}_main.log")
        fh = logging.FileHandler(main_log_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.main_logger.addHandler(fh)
    
    def run(self):
        num_workers = self.n_process
        self.main_logger.info(f"Prcoessing {len(self.data)} data using {num_workers} workers.")
        
        ctx = multiprocessing.get_context('spawn')
        
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            if self.system == "table":
                func = partial(process_data_table, config=self.config, model=self.model, max_iterations=self.max_iterations, max_compile_iterations=self.max_compile_iterations)

            future_to_data = {executor.submit(func, data_item): data_item for data_item in self.data}
            
            for future in as_completed(future_to_data):
                data_item = future_to_data[future]
                try:
                    result = future.result()
                    self.logs.append(result)
                    flag = result["flag"]
                    improvment = result.get("improvement", 0)
                    if flag == "improved":
                        self.main_logger.info(f"{result['data']} Completed. Result: {flag} (+{improvment})")
                    else:
                        self.main_logger.info(f"{result['data']} Completed. Result: {flag}")
                except Exception as exc:
                    data_suffix = os.path.splitext(os.path.basename(data_item["image"]))[0]
                    self.main_logger.error(f"## Failed to process data {data_suffix}: {exc} ##", exc_info=True)
                    self.logs.append({
                        "data": data_suffix,
                        "log": [{
                            "agent": "Error",
                            "error": str(exc)
                        }],
                        "flag": "error"
                    })
        

        final_log_path = os.path.join(self.config["working_dir"], f"{self.config['exp_name']}_logs.json")
        with open(final_log_path, "w") as f:
            json.dump(self.logs, f, indent=2)
        
        self.main_logger.info(f"Processing complete. Logs saved to {final_log_path}")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="../dataset")
    parser.add_argument("--working_dir", type=str, default="working")
    parser.add_argument("--result_dir", type=str, default="")
    parser.add_argument("--exp_name", type=str, default="exp1")
    parser.add_argument("--max_iter", type=int, default=3)
    parser.add_argument("--max_compile_iter", type=int, default=1)
    parser.add_argument("--need_regions_correction", type=str, default="True")
    parser.add_argument("--n_process", type=int, default=8)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--data_range", type=str, default="100-200")
    parser.add_argument("--agent_kwargs", type=json.loads, default="{}")
    args = parser.parse_args()
    
    working_dir = args.working_dir
    result_dir = args.result_dir
    exp_name = args.exp_name
    max_iterations = args.max_iter
    max_compile_iterations = args.max_compile_iter
    need_regions_correction = args.need_regions_correction
    n_process = args.n_process
    model = args.model
    data_range = args.data_range.split("-")
    start, end = int(data_range[0]), int(data_range[1])
    kwargs = args.agent_kwargs
    
    #img_parent_dir = args.dataset_dir
    test_data_path = args.dataset_dir
    with open(test_data_path) as f:
        test_data = [json.loads(line) for line in f]
    data = test_data[start:end]
    
    os.makedirs(working_dir, exist_ok=True)
    
    config = {
        "exp_name": exp_name,
        "working_dir": working_dir,
        "result_dir": result_dir,
        "max_iterations": max_iterations,
        "max_compile_iterations": max_compile_iterations,
        "need_regions_correction":need_regions_correction,
        "agent_kwargs": kwargs,
    }
    
    mas = MultiAgentSystem(model, data, n_process, config)
    mas.run()
