# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .Agent import *
from .utils import *

__init__ = """"""

class CriticAgent(Agent):
  def __init__(self, model, **kwargs):
    super().__init__("CriticAgent", model, **kwargs)
    
  def get_agent_info(self):
    info = """This agent is responsible for handling code that fails to compile.
    It detects syntax or LaTeX compilation errors and attempts to fix the code automatically.

    The input of this agent is:
    - img_path: a path to the image that the code should generate
    - code: the generated code that failed to compile
    - output_path: a path to the output directory
    - output_prefix: a prefix for the output files
    - output_suffix: a suffix for the output files

    The output of this agent is:
    - raw_response: the raw response from the model
    - fixed_code: the corrected version of the code
    - fixed_code_path: the path to the corrected code file
    - fixed_image_path: a path to the re-rendered image after code fixing
    - combined_image_path: a path to the image comparing the original and fixed outputs
    """
    for line in info.split("\n"):
      print("|", line)
    return info
  
  
  def extract_scores_from_output(self, output_text):
        """
        从模型输出中提取五个评分项的 score 值。
        
        Args:
            output_text (str): 模型输出文本，包含 JSON 格式评分信息。
        
        Returns:
            dict: 形如 {"Content": 1, "Structure": 0, ...} 的评分字典。
        """
        aspects = ["Content", "Structure", "Line", "Alignment", "Cell Style"]
        scores = {}

        for aspect in aspects:
            # 正则匹配形如 '"Content": { "analysis": "...", "score": 1 }'
            pattern = rf'"{aspect}"\s*:\s*{{.*?"score"\s*:\s*(\d).*?}}'
            match = re.search(pattern, output_text, re.DOTALL)
            if match:
                scores[aspect] = int(match.group(1))
            else:
                scores[aspect] = None  # 或者设为 0 / raise Exception
        
        return scores
  
  def clean_adjustbox(self,text):
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
  
  def prompt_assembler(self, gt_code, pred_code):
        from agents.prompts.critic_prompt import critic_prompt, critic_prompt_qwen, critic_prompt_gemini
        if "qwen" in self.model_name or "gpt-5.2" in self.model_name or "deepseek" in self.model_name:
            prompt = critic_prompt_qwen.format(gt_code=gt_code, pred_code=pred_code)
        elif "gemini" in self.model_name:
            gt_code = self.clean_adjustbox(gt_code)
            pred_code = self.clean_adjustbox(pred_code) 
            prompt = critic_prompt_gemini.format(gt_code=gt_code, pred_code=pred_code)
        else:
            prompt = critic_prompt.format(gt_code=gt_code, pred_code=pred_code)
        return prompt
    
  def act(self, gt_code, pred_code):
      #system_prompt = "You are tasked with generating code from the given image. NOTE: The only two external packages allowed for generated code is matplotlib and numpy."

      generate_prompt = self.prompt_assembler(gt_code, pred_code)
      raw_response = self.critic_generate(generate_prompt)
      print("critic raw response:")
      print(raw_response)

      score = self.extract_scores_from_output(raw_response)
      print("score")
      print(score)
      return raw_response, score
