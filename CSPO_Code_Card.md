# CSPO

## Overview

CSPO is a code framework for structured Table-to-LaTeX generation that alleviates reward ambiguity during reinforcement learning-based training, enabling LLMs to generate accurate and compilable LaTeX representations of structured tables. 

See related model ([3B](https://yunfanyang1/Qwen2.5-VL-3B-CSPO) / [7B](https://yunfanyang1/Qwen2.5-VL-7B-CSPO)) and [TableTex dataset](https://huggingface.co/datasets/yunfanyang1/TableTex).

---

## WHAT CAN CSPO do

CSPO was developed to alleviate reward ambiguity in structured Table-to-LaTeX generation by introducing a Component-Specific Policy Optimization (CSPO) framework that leverages component-wise feedback as structured reward signals during RL-based training. This enables LLMs to generate accurate and compilable LaTeX representations of structured tables. We also designed fine-grained evaluation metrics for diagnostic evaluation of generation quality.

A detailed discussion of CSPO, including how it was developed and tested, can be found in our paper *CSPO: Alleviating Reward Ambiguity for Structured Table-to-LaTeX Generation* ([https://arxiv.org/pdf/2604.10918](https://arxiv.org/pdf/2604.10918)). 

---

## INTENDED USEs

* CSPO is suited for research on structured document understanding, Table-to-LaTeX generation, and reinforcement learning from structured feedback for code-structured output tasks. 
* CSPO is being shared with the research community to facilitate reproduction of our results and foster further research in this area.
* CSPO is intended to be used by domain experts who are independently capable of evaluating the quality of outputs before acting on them.

---

## OUT-of-scope uses

* CSPO is not well suited for general-purpose LaTeX document generation beyond structured table conversion, or for domains requiring high-stakes accuracy without human verification. 
* We do not recommend using CSPO in commercial or real-world applications without further testing and development. It is being released for research purposes.
* CSPO was not designed or evaluated for all possible downstream purposes. Developers should consider its inherent limitations as they select use cases, and evaluate and mitigate accuracy, safety, and fairness concerns specific to each intended downstream use.
* Without further testing and development, CSPO should not be used in sensitive domains where inaccurate outputs could suggest actions that lead to injury or negatively impact an individual's legal, financial, or life opportunities.
* We do not recommend using CSPO in the context of high-risk decision making (e.g. in law enforcement, legal, finance, or healthcare).

---

## How to get started

please refer to the README in our [GitHub repository](https://github.com/microsoft/CSPO). 

### Conda / Miniconda

Create and activate the CSPO environment:

```bash
conda create -n cspo python=3.12
conda activate cspo
```

### Install TeX Live

Our training and evaluation pipeline requires LaTeX compilation, such as `pdflatex`.
Please install TeX Live from the official website:

- TeX Live: [TeX Live Official Website](https://tug.org/texlive/)
- Installation Guide: [TeX Live Documentation](https://www.tug.org/texlive/doc.html)

We recommend installing the full TeX Live distribution.

On Ubuntu/Debian systems, also install the following system dependencies:

```bash
sudo apt-get update
sudo apt-get install -y ghostscript poppler-utils
```

### Install Python Dependencies

```bash
cd CSPO
bash setup.sh
```

## Evaluation

Models trained using the CSPO method were evaluated on their ability to generate compilable and accurate LaTeX code from structured table inputs, measured by structure, style, and content fidelity to ground-truth LaTeX.

In our experiments, we used Qwen2.5-VL-3B, Qwen2.5-VL-7B as the base models for finetuning and GPT-4o as an automatic judge for component-specific rewarding. Results may vary if CSPO is used with a different base and judge models, based on their unique design, configuration and training.

A detailed discussion of our evaluation methods and results can be found in our paper (https://arxiv.org/pdf/2604.10918).

### EVALUATION METHODS

We used a hierarchical, multi-level assessment framework that combines global metrics with fine-grained, component-wise diagnostics to measure table image to LaTeX code performance.

Global Evaluation – focuses on overall quality
* TEDS (Tree Edit Distance Similarity): measures structural and semantic similarity between generated and reference LaTeX.
* Compilation Rate (R): checks whether the generated LaTeX code compiles successfully (executable).

Fine-grained Evaluation – decomposes performance into key components, each evaluated at the table level (binary correctness)
* Structural correctness (S): layout fidelity (e.g., rows, columns, merged cells).
* Content fidelity (C): accuracy of cell values.
* Stylistic fidelity (Y): formatting consistency, including line styles, alignment, and cell formatting.

Composite Metric – Overall Fidelity (OF)
* `OF = S ∧ C ∧ Y ∧ R`: requires simultaneous correctness in structure, content, style, and compilability.

Instead of relying solely on a single aggregated score, the evaluation explicitly disentangles structure, content, and style, enabling more diagnostic and interpretable analysis.

### EVALUATION RESULTS

Overall performance
* Models trained using the CSPO method achieve the best Overall Fidelity (OF) and strong TEDS, outperforming both general MLLMs and specialized baselines, indicating higher end-to-end table reconstruction quality.

Improvement over RL baselines
* Compared to GRPO and VSGRPO, CSPO consistently improves OF, showing that component-specific rewards and credit assignment are more effective than global reward designs.

---

## LIMITATIONS

* Overall Fidelity (OF) remains limited: the best model (Qwen2.5-VL-7B-CSPO) achieves ~53% OF on the in-domain TableTex-test set, leaving significant room for improvement.
* Models trained using CSPO were not evaluated on non-English tables; performance on multilingual content is unknown.
* CSPO was developed for research and experimental purposes. Further testing and validation are needed before considering its application in commercial or real-world scenarios.
* CSPO was designed and tested using the English language. Performance in other languages may vary and should be assessed by someone who is both an expert in the expected outputs and a native speaker of that language.
* Outputs generated by AI may include factual errors, fabrication, or speculation. Users are responsible for assessing the accuracy of generated content. All decisions leveraging outputs of the system should be made with human oversight and not be based solely on system outputs.
* CSPO inherits any biases, errors, or omissions produced by its base model. Developers are advised to choose an appropriate base LLM/MLLM carefully, depending on the intended use case.

---

## BEST PRACTICES 

Better performance can be achieved by training CSPO with sufficient epochs. We found two epochs lead to better performance than one epoch. 

---

## LICENSE 

**MIT License** Nothing disclosed here, including the Out of Scope Uses section, should be interpreted as or deemed a restriction or modification to the license the code is released under. 

---

## TRADEMARKS 

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies. 

---

## CONTACT 

This research was conducted by members of Microsoft Research. We welcome feedback and collaboration from our audience. If you have suggestions, questions, or observe unexpected/offensive behavior in our technology, please contact us at culan@microsoft.com. 

If the team receives reports of undesired behavior or identifies issues independently, we will update this repository with appropriate mitigations.