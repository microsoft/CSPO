graded_review_prompt = r"""You are a reviewer specializing in LaTeX table code.
Please compare the following ground truth LaTeX table code and the predicted LaTeX table code, and assign a score to the predicted code.

### Scoring Scale
For each item, assign a score from 0 to 3 based on the following general criteria:
3 = Identical or visually indistinguishable from the reference.
2 = Differences are clearly noticeable upon inspection, but they affect only limited parts of the table and do not substantially change its overall appearance or interpretability.
1 = Differences are prominent and affect a substantial portion of the table’s presentation, but the underlying information remains intact and understandable.
0 = Differences compromise correctness, completeness, or interpretability of the information, or cause compilation failure.

### Evaluation Guidelines
Evaluate each dimension **independently**. An error in one category must **not** influence the score of another. Stick strictly to the scope defined in the "Reference" for each dimension.
* *Example:* Do not penalize "Structure" for an "Alignment" error.

### Scoring Criteria and Corresponding Code Sections:
1. Alignment
Reference: \begin{{tabular}}{{…}}, including the type, order, and number of c, r, and l columns.
2. Vertical Lines
Reference: \begin{{tabular}}{{…}}, including '|' and '||'.
3. Horizontal Lines
Reference: Commands such as: \hline, \hline \hline, \cline, \hhline, \toprule, \midrule, \bottomrule, \cmidrule(lr){{i-j}}. The type, number, and position of the lines must match exactly.
4. Structure
Reference: Tokens such as &, \\, \multicolumn, and \multirow.
5. Caption
Reference: The content of \caption{{…}} and its position (whether it appears before or after the table).
6. Text Content and Style
Reference: The cell content and styling, excluding alignment, line types, structure, and caption. Minor formatting differences (e.g., $10$ vs 10) should be considered correct. However, missing or incorrect content, or inconsistencies that affect semantics or formatting (e.g., \textbf{{10}} vs 10) should be considered incorrect.
7.Preamble
Reference: Only check commands before \begin{{document}} that are necessary for successful compilation, such as \usepackage{{...}}. Missing non-essential commands that do not affect compilability is acceptable and should not be penalized.

Please return your judgment as a JSON dictionary in the following format:
{{
  "Alignment": {{
    "reason": "...",
    "score": 3
  }},
  "Vertical Lines": {{
    "reason": "...",
    "score": 2
  }},
  "Horizontal Lines": {{
    "reason": "...",
    "score": 1
  }},
  "Structure": {{
    "reason": "...",
    "score": 0
  }},
  "Caption": {{
    "reason": "...",
    "score": 3
  }},
  "Text Content and Style": {{
    "reason": "...",
    "score": 2
  }},
  "Preamble": {{
    "reason": "...",
    "score": 3
  }}
}}

Ground truth code:
```latex
{gt_code}
```

Predicted code:
```latex
{pred_code}
```
"""

review_prompt = r"""You are a reviewer specializing in LaTeX table code.
Please compare the following ground truth LaTeX table code and the predicted LaTeX table code, and assign a score to the predicted code.
Scoring Criteria and Corresponding Code Sections:
1. Alignment
Reference: \begin{{tabular}}{{…}}, including the type, order, and number of c, r, and l columns.
2. Vertical Lines
Reference: \begin{{tabular}}{{…}}, including '|' and '||'.
3. Horizontal Lines
Reference: Commands such as: \hline, \hline \hline, \cline, \hhline, \toprule, \midrule, \bottomrule, \cmidrule(lr){{i-j}}. The type, number, and position of the lines must match exactly.
4. Structure
Reference: Tokens such as &, \\, \multicolumn, and \multirow.
5. Caption
Reference: The content of \caption{{…}} and its position (whether it appears before or after the table).
6. Text Content and Style
Reference: The cell content and styling, excluding alignment, line types, structure, and caption. Minor formatting differences (e.g., $10$ vs 10) should be considered correct. However, missing or incorrect content, or inconsistencies that affect semantics or formatting (e.g., \textbf{{10}} vs 10) should be considered incorrect.
7.Preamble
Reference: Only check commands before \begin{{document}} that are necessary for successful compilation, such as \usepackage{{...}}. Missing non-essential commands that do not affect compilability is acceptable and should not be penalized.

Each item should be scored as either 1 (correct) or 0 (incorrect), along with a reason justifying the score.
Please return your judgment as a JSON dictionary in the following format:
{{
  "Alignment": {{
    "reason": "...",
    "score": 1
  }},
  "Vertical Lines": {{
    "reason": "...",
    "score": 0
  }},
  "Horizontal Lines": {{
    "reason": "...",
    "score": 1
  }},
  "Structure": {{
    "reason": "...",
    "score": 1
  }},
  "Caption": {{
    "reason": "...",
    "score": 1
  }},
  "Text Content and Style": {{
    "reason": "...",
    "score": 1
  }},
  "Preamble": {{
    "reason": "...",
    "score": 0
  }}
}}

Ground truth code:
```latex
{gt_code}
```

Predicted code:
```latex
{pred_code}
```
"""

tabular_review_prompt = r"""You are a reviewer specializing in LaTeX table code.
Please compare the following ground truth LaTeX table code and the predicted LaTeX table code, and assign a score to the predicted code.
Scoring Criteria and Corresponding Code Sections:
1. Alignment
Reference: \begin{{tabular}}{{…}}, including the type, order, and number of c, r, and l columns.
2. Vertical Lines
Reference: \begin{{tabular}}{{…}}, including '|' and '||'.
3. Horizontal Lines
Reference: Commands such as: \hline, \hline \hline, \cline, \hhline, \toprule, \midrule, \bottomrule, \cmidrule(lr){{i-j}}. The type, number, and position of the lines must match exactly.
4. Structure
Reference: Tokens such as &, \\, \multicolumn, and \multirow.
5. Text Content and Style
Reference: The cell content and styling, excluding alignment, line types, structure, and caption. Minor formatting differences (e.g., $10$ vs 10) should be considered correct. However, missing or incorrect content, or inconsistencies that affect semantics or formatting (e.g., \textbf{{10}} vs 10) should be considered incorrect.

Each item should be scored as either 1 (correct) or 0 (incorrect), along with a reason justifying the score.
Please return your judgment as a JSON dictionary in the following format:
{{
  "Alignment": {{
    "reason": "...",
    "score": 1
  }},
  "Vertical Lines": {{
    "reason": "...",
    "score": 0
  }},
  "Horizontal Lines": {{
    "reason": "...",
    "score": 1
  }},
  "Structure": {{
    "reason": "...",
    "score": 1
  }},
  "Text Content and Style": {{
    "reason": "...",
    "score": 1
  }}
}}

Ground truth code:
```latex
{gt_code}
```

Predicted code:
```latex
{pred_code}
```
"""
