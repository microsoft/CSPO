# Dataset card for TableTex v1.0

## Overview

TableTex is a research dataset derived from structured tables and their corresponding LaTeX representations collected from publicly available arXiv papers. It was developed to support research on structured table-to-LaTeX generation. See related training [code](https://github.com/microsoft/CSPO) and model ([3B](https://yunfanyang1/Qwen2.5-VL-3B-CSPO) / [7B](https://yunfanyang1/Qwen2.5-VL-7B-CSPO)). 

A detailed discussion of TableTex, including how it was developed and evaluated, can be found in our paper *CSPO: Alleviating Reward Ambiguity for Structured Table-to-LaTeX Generation* (https://arxiv.org/pdf/2604.10918).

### Intended uses

* TableTex is best suited for evaluating the table image to LaTeX code generation quality.
* TableTex is being shared with the research community to facilitate reproduction of our results and foster further research in this area.

### Out-of-scope uses

* TableTex is not well suited for other tasks unrelated with table image understanding.
* We do not recommend using TableTex in commercial or real-world applications without further testing and development. It is being released for research purposes.
* We do not recommend using TableTex in the context of high-risk decision making (e.g. in law enforcement, legal, finance, or healthcare).

---

## Dataset Details

### Dataset Contents

TableTex consists of 19,000 instance pairs of table images and renderable LaTeX codes. Each instance includes a table image and the corresponding renderable LaTeX code. 

To ensure legal compliance, we restrict our dataset to content released under licenses that explicitly permit reuse. Specifically, all included data fall under one of the following licenses: CC BY, CC BY-SA, CC0, and CC BY-NC.

---

## Data Creation & Processing

* **Collection Source:** TableTex was collected exclusively from publicly available papers hosted on arXiv. The data that was used to create TableTex consisted of table images and LaTeX codes.
* **People & Identifiers:** TableTex is not believed to contain information that could be used to directly or indirectly identify a person.
* **Sensitive or harmful content:** TableTex is not believed to contain information that might be considered sensitive or private. The dataset consists exclusively of table images and corresponding LaTeX code extracted from publicly available scientific papers on arXiv.
* **Other processing:** The data was not manually annotated. TableTex pairs are constructed by automatically extracting table LaTeX code directly from arXiv paper source files and rendering them into table images using a LaTeX compiler. No additional human or AI annotation was applied.

---

## How to get started

To begin using TableTex, please refer to the associated GitHub repository (https://github.com/microsoft/CSPO) and model release ([3B](https://yunfanyang1/Qwen2.5-VL-3B-CSPO) and [7B](https://yunfanyang1/Qwen2.5-VL-7B-CSPO)). 

The dataset is provided as paired table image files and corresponding LaTeX code files. Users can load the dataset using standard Python data loading utilities. See the related [code](https://github.com/microsoft/CSPO) and model ([3B](https://yunfanyang1/Qwen2.5-VL-3B-CSPO) / [7B](https://yunfanyang1/Qwen2.5-VL-7B-CSPO)) for usage examples.

---

## Validation

To assess how effective TableTex would be at its intended purpose, our team evaluated models trained on the dataset using hierarchical evaluation metrics, including TEDS (Tree-Edit-Distance-based Similarity), Compilation Rate, and fine-grained metrics covering Structural Correctness (S), Content Fidelity (C), and Stylistic Fidelity (Y).

Specifically, we trained multimodal large language models (Qwen2.5-VL-3B and Qwen2.5-VL-7B) on TableTex-train (15,000 samples) and evaluated them on TableTex-test (4,000 samples), as well as two out-of-domain benchmarks: DocGenome-table-1k and Table2LaTeX-test-simple.

Our CSPO model achieved an Overall Fidelity (OF) of 45.2% on the 3B model and 53.0% on the 7B model, consistently outperforming baseline models trained with GRPO and other reward strategies. A detailed discussion of our validation methods and results can be found in our paper: *CSPO: Alleviating Reward Ambiguity for Structured Table-to-LaTeX Generation* (https://arxiv.org/pdf/2604.10918).

---

## Limitations

* Training dataset size is relatively small (15,000 samples); performance may degrade on tables with rare layouts, complex merged cells, or formatting styles not well represented in TableTex.
* TableTex was developed for research and experimental purposes. Further testing and validation are needed before considering its application in commercial or real-world scenarios.
* TableTex consists of English language instances only.
* TableTex should not be used to train models in highly regulated domains where inaccurate or incomplete model outputs could suggest actions that lead to injury or negatively impact an individual's legal, financial, or life opportunities.

---

## Best Practices

* We recommend splitting the data into a training set (TableTex-train, 15,000 samples) and a test set (TableTex-test, 4,000 samples). We randomly partition that in our experiments and provide partition manner in our code.
* It is the user’s responsibility to ensure that the use of TableTex complies with relevant data protection regulations and organizational guidelines.

---

## License

**CC BY**

All the data were collected from arXiv papers under the following licenses: CC BY, CC BY-SA, CC0, and CC BY-NC.

---

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. 

Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.

---

## Contact

This research was conducted by members of Microsoft Research. We welcome feedback and collaboration from our audience. 

If you have suggestions, questions, or observe unexpected/problematic data in our dataset, please contact us at culan@microsoft.com. If the team receives reports of undesired content or identifies issues independently, we will update this repository with appropriate mitigations.