# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# critic_prompt = r"""You will be given two LaTeX table codes: a Ground Truth version and a Predicted version.

# Your task is to compare their rendered results across five aspects: Content, Structure, Line, Alignment and Cell Style. For each aspect, provide:
# - "analysis": a short explanation of whether the prediction is correct
# - "score": 1 if correct, 0 if there is any noticeable error


# [EVALUATION CRITERIA]

# 1. Content
# Check if all textual contents in the table are identical. Minor differences in LaTeX syntax (e.g., alternative math formatting) are acceptable as long as the rendered result is the same.
# **Errors include**: missing, extra, or incorrect content.

# 2. Structure
# Check if the structure matches: row/column counts, merged cells (`\multicolumn`, `\multirow`), and cell positions.  
# **Errors include**: missing, extra, misaligned, or wrongly merged cells/rows/columns.

# 3. Line 
# Check if horizontal and vertical lines are consistent in placement, numbers and style (e.g., `\hline`, `\hline \hline`, `\cline`, `\toprule`).  
# **Errors include**: missing, excessive, misplaced, or stylistically inconsistent lines.

# 4. Alignment
# Check if cell/column text alignment (left `l`, center `c`, right `r`) matches.  
# **Errors include**: misalignment that changes visual layout.

# 5. Cell Style
# Check if text styles (bold, italic, underline, color) and background colors match.  
# **Errors include**: missing or incorrect style.


# [TABLE CODES]

# Ground Truth Code:
# ```latex
# {gt_code}
# ```

# Predicted Code:
# ```latex
# {pred_code}
# ```


# [OUTPUT FORMAT]
# Please return your judgment as a JSON dictionary like the following:
# {{
#   "Content": {{
#     "analysis": "...",
#     "score": 0
#   }},
#   "Structure": {{
#     "analysis": "...",
#     "score": 1
#   }},
#   "Line": {{
#     "analysis": "...",
#     "score": 0
#   }},
#   "Alignment": {{
#     "analysis": "...",
#     "score": 1
#   }},
#   "Cell Style": {{
#     "analysis": "...",
#     "score": 1
#   }}
# }}
# """

critic_prompt = r"""You will be given two LaTeX table codes: a Ground Truth version and a Predicted version.

Your task is to compare their rendered results across five aspects: Content, Structure, Line, Alignment and Cell Style. For each aspect, provide:
- "analysis": a short explanation of whether the prediction is correct
- "score": 1 if correct, 0 if there is any noticeable error


[EVALUATION CRITERIA]

1. Content
Check if all textual contents in the table are identical. Minor differences in LaTeX syntax (e.g., alternative math formatting) are acceptable as long as the rendered result is the same.

2. Structure
Check if the structure matches: row/column counts, merged cells (`\multicolumn`, `\multirow`), and cell positions.  

3. Line 
Check if horizontal and vertical lines are consistent in placement, numbers and style (e.g., `\hline`, `\hline \hline`, `\cline`, `\toprule`).  

4. Alignment
Check if cell/column text alignment (left `l`, center `c`, right `r`) matches.  

5. Cell Style
Check if text styles (e.g., bold, italic, underline, color) and background colors match.  

[TABLE CODES]

Ground Truth Code:
```latex
{gt_code}
```

Predicted Code:
```latex
{pred_code}
```


[OUTPUT FORMAT]
Please return your judgment as a JSON dictionary like the following:
{{
  "Content": {{
    "analysis": "...",
    "score": 0
  }},
  "Structure": {{
    "analysis": "...",
    "score": 1
  }},
  "Line": {{
    "analysis": "...",
    "score": 0
  }},
  "Alignment": {{
    "analysis": "...",
    "score": 1
  }},
  "Cell Style": {{
    "analysis": "...",
    "score": 1
  }}
}}
"""

critic_prompt_qwen = r"""You will be given two LaTeX table codes: a Ground Truth version and a Predicted version.

Your task is to compare their rendered results across five aspects: Content, Structure, Line, Alignment and Cell Style. For each aspect, provide:
- "analysis": a short explanation of whether the prediction is correct
- "score": 1 if correct, 0 if there is any noticeable error

[Evaluation Guidelines]
Evaluate each dimension **independently**. An error in one category must **not** influence the score of another. 
* *Example:* Do not penalize "Content" for an "Cell Style" error.

[EVALUATION CRITERIA]

1. Content
Check if all textual contents in the table are identical. Minor differences in LaTeX syntax (e.g., alternative math formatting) are acceptable as long as the rendered result is the same.
Note: dollar signs around the reward values can be ignored, for example, $0.5$ and 0.5 are equivalent.

2. Structure
Check if the structure matches: row/column counts, merged cells (`\multicolumn`, `\multirow`), and cell positions.  

3. Line 
Check if horizontal and vertical lines are consistent in placement, numbers and style (e.g., `\hline`, `\hline \hline`, `\cline`, `\toprule`).  
Note: "\hline\n\hline" and "\hline \hline"  is equivalent to "\hline\hline".

4. Alignment
Check if cell/column text alignment (left `l`, center `c`, right `r`) matches.  

5. Cell Style
Check if text styles (e.g., bold, italic, underline, color) and background colors match.  
Note: dollar signs around the reward values can be ignored, for example, $0.5$ and 0.5 are equivalent.

[TABLE CODES]

Ground Truth Code:
```latex
{gt_code}
```

Predicted Code:
```latex
{pred_code}
```


[OUTPUT FORMAT]
Please return your judgment as a JSON dictionary like the following:
{{
  "Content": {{
    "analysis": "...",
    "score": 0
  }},
  "Structure": {{
    "analysis": "...",
    "score": 1
  }},
  "Line": {{
    "analysis": "...",
    "score": 0
  }},
  "Alignment": {{
    "analysis": "...",
    "score": 1
  }},
  "Cell Style": {{
    "analysis": "...",
    "score": 1
  }}
}}
"""

critic_prompt_gemini = r"""You will be given two LaTeX table codes: a Ground Truth version and a Predicted version.

Your task is to compare their rendered results across five aspects: Content, Structure, Line, Alignment and Cell Style. For each aspect, provide:
- "analysis": a short explanation of whether the prediction is correct
- "score": 1 if correct, 0 if there is any noticeable error

[Evaluation Guidelines]
Evaluate each dimension **independently**. An error in one category must **not** influence the score of another. 
* *Example:* Do not penalize "Content" for a "Cell Style" error.

[EVALUATION CRITERIA]

1. Content
Check if all textual contents in the table are identical. Focus solely on semantic consistency. Disregard any differences in space, text formatting, or LaTeX syntax.
Note: 
* dollar signs around the values can be ignored, for example, $0.5$ and 0.5 are equivalent.
* Do not penalize "Content" for a "Cell Style" error, for example, \texttt{{c}} and c have the same content.
* empty and `~` are equivalent in content.

2. Structure
Check if the structure matches: row/column counts, merged cells (`\multicolumn`, `\multirow`), and cell positions.  

3. Line 
Check if horizontal and vertical lines are consistent in placement, numbers and style (e.g., `\hline`, `\hline \hline`, `\cline`, `\toprule`).  
Note: "\hline\n\hline" and "\hline \hline"  is equivalent to "\hline\hline".

4. Alignment
Check if cell/column text alignment (left `l`, center `c`, right `r`) matches.  

5. Cell Style
Check if text styles (e.g., bold, italic, underline, color) and background colors match.  
Note: Math mode and plain text share the exact same cell style (e.g., "$0.5$" is strictly equivalent to "0.5").

[TABLE CODES]

Ground Truth Code:
```latex
{gt_code}
```

Predicted Code:
```latex
{pred_code}
```


[OUTPUT FORMAT]
Please return your judgment as a JSON dictionary like the following:
{{
  "Content": {{
    "analysis": "...",
    "score": 0
  }},
  "Structure": {{
    "analysis": "...",
    "score": 1
  }},
  "Line": {{
    "analysis": "...",
    "score": 0
  }},
  "Alignment": {{
    "analysis": "...",
    "score": 1
  }},
  "Cell Style": {{
    "analysis": "...",
    "score": 1
  }}
}}
"""
