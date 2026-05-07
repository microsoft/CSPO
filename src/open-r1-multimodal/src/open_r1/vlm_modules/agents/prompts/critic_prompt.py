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