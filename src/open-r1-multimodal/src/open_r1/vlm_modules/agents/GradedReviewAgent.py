from .Agent import *
from .utils import *

__init__ = """"""

class GradedReviewAgent(Agent):
  def __init__(self, model):
    super().__init__("GradedReviewAgent", model)
    self._REQUIRED_KEYS = [
        "Alignment",
        "Vertical Lines",
        "Horizontal Lines",
        "Structure",
        "Caption",
        "Text Content and Style",
        "Preamble"
        ]
    self._LOWER_TO_CANONICAL = {k.lower(): k for k in self._REQUIRED_KEYS}

    
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
  
  # def prompt_assembler(self, compliation_failed_code):
  #       from agents.prompts.compilation_fix_prompt import compilation_fix_prompt
  #       prompt = [compilation_fix_prompt[0], compilation_fix_prompt[1].format(compliation_failed_code=compliation_failed_code)]
  #       return prompt

  def is_valid_score_dict(self, score_dict):
      """
      检查 GPT 提取出的评分字典是否为合法结构（键大小写不敏感）。
      
      参数:
          score_dict (dict): 提取出的 JSON 字典。
      
      返回:
          bool: True 表示结构合法，False 表示非法。
      """

      # 预期的字段名（统一小写用于对比）
      required_keys = [
          "alignment",
          "vertical lines",
          "horizontal lines",
          "structure",
          "caption",
          "text content and style",
          "preamble"
      ]

      if not isinstance(score_dict, dict):
          return False

      # 将所有 key 转为小写做映射（同时保留原键）
      lowered_dict = {k.lower(): v for k, v in score_dict.items()}

      for key in required_keys:
          if key not in lowered_dict:
              return False

          section = lowered_dict[key]
          if not isinstance(section, dict):
              return False

          # 内部键统一小写处理
          inner_keys = {ik.lower(): iv for ik, iv in section.items()}

          if "reason" not in inner_keys or "score" not in inner_keys:
              return False

          # 检查 reason 是字符串
          if not isinstance(inner_keys["reason"], str):
              return False

          # 检查 score 是整数且为 0 或 1 or 2 or 3
          if inner_keys["score"] not in [0, 1, 2, 3]:
              return False

      return True
  # ======== 你需要的“自动修复/填充”逻辑 ========
  def coerce_score_dict(self, score_dict):
        """
        将不合规的 score_dict 变为合规：
        - 如果为 None：生成完整的随机 0,1,2,3 合规字典；
        - 如果为 dict：保留已有、合规的内容；缺项或不合规项随机填充并修正为合规结构。
        """
        import random
        def rand_item():
            val = random.randint(0, 3)
            return {"reason": "autofilled", "score": val}

        # 情况 1：None -> 直接生成完整随机
        if score_dict is None:
            return {k: rand_item() for k in self._REQUIRED_KEYS}

        result = {}

        # 先把已有的（尽量）规范化并拷贝到 result
        if isinstance(score_dict, dict):
            # 输入字典的 key 做小写映射到规范名
            lowered = {str(k).lower(): v for k, v in score_dict.items()}
            for req_lower, canonical in self._LOWER_TO_CANONICAL.items():
                if req_lower in lowered and isinstance(lowered[req_lower], dict):
                    section = lowered[req_lower]
                    # 取出 reason / score，做最小修复
                    reason = section.get("reason")
                    score_val = section.get("score")

                    # 修正 reason
                    if not isinstance(reason, str):
                        reason = "autofilled"

                    # 修正 score（不是 0,1,2,3 就随机）
                    if not isinstance(score_val, int) or score_val not in (0, 1, 2, 3):
                        score_val = random.randint(0, 3)

                    result[canonical] = {"reason": reason, "score": score_val}
                else:
                    # 缺项则随机填充
                    result[canonical] = rand_item()
        else:
            # 非 dict（例如字符串/列表等） -> 全量随机
            result = {k: rand_item() for k in self._REQUIRED_KEYS}

        return result

  def extract_scores_from_output(self, output_text):
    """
    提取 GPT-4o 输出中的 JSON 字典部分，并将其转换为 Python 字典。
    
    参数:
        output_text (str): GPT-4o 输出的完整文本。
    
    返回:
        dict: 提取并解析后的字典对象。
    
    异常:
        如果无法找到合法的 JSON 字典，将抛出 ValueError。
    """
    # 尝试用正则提取最外层 {...} 的 JSON 内容
    match = re.search(r'\{[\s\S]*\}', output_text)
    
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            return None
    else:
        return None
  
  def prompt_assembler(self, gt_code, pred_code):
        from open_r1.vlm_modules.agents.prompts.review_prompt import graded_review_prompt
        prompt = graded_review_prompt.format(gt_code=gt_code, pred_code=pred_code)
        return prompt
    
  def act(self, gt_code, pred_code):
      #system_prompt = "You are tasked with generating code from the given image. NOTE: The only two external packages allowed for generated code is matplotlib and numpy."

      generate_prompt = self.prompt_assembler(gt_code, pred_code)
      raw_response = self.critic_generate(generate_prompt)
      #print("critic raw response:")
      #print(raw_response)

      score = self.extract_scores_from_output(raw_response)
      if score is not None and self.is_valid_score_dict(score):
        return raw_response, score
      else:
        return raw_response, None
      