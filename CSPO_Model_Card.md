# CSPO (Qwen2.5-VL-3B-CSPO and Qwen2.5-VL-7B-CSPO)

## Model summary

| Metric/Field | Description |
| :--- | :--- |
| **Developer** | Microsoft Corporation, "Authorized representative: Microsoft Ireland Operations Limited 70 Sir John Rogerson’s Quay, Dublin 2, D02 R296, Ireland", Instructions: Do not modify this text |
| **Description** | CSPO is a fine-tuned multimodal model for structured table-to-LaTeX generation, built upon the Qwen2.5-VL model. The model is trained on the TableTex dataset, which consists of paired table images and LaTeX code extracted from publicly available arXiv papers under permissive licenses. It is post-trained using a reinforcement learning framework that introduces component-specific rewards and token-level credit assignment to mitigate reward ambiguity and improve structural, content, and stylistic fidelity. |
| **Model architecture** | Multimodal decoder-only transformer (Qwen2.5-VL) with image-text inputs and text outputs. |
| **Parameters** | 3B, 7B |
| **Inputs** | Image (table images) and text (structured prompts or instructions); input size is bounded by the context length and dynamic-resolution vision encoder of the base model (Qwen2.5-VL). |
| **Context length** | 32,768 tokens |
| **Outputs** | Text (LaTeX code); output length is limited by the context length. |
| **Public data summary (or summaries)** | The training data is based on our constructed TableTex dataset, constructed from publicly available LaTeX sources and corresponding rendered table images extracted from arXiv papers under permissive licenses. |
| **Training Dates** | Nov 2025 to Dec 2025 |
| **Release date, Release date in the EU (if different)** | Not yet released |
| **License** | Open source. |
| **Model dependencies** | Qwen2.5-VL-3B-Instruct, Qwen2.5-VL-7B-Instruct |
| **List and link to any additional related assets** | TabeTex dataset, Finetuning framework |
| **Acceptable use policy** | N/A |

---

## 1. Model overview

The CSPO-trained models (Qwen2.5-VL-3B-CSPO and Qwen2.5-VL-7B-CSPO) are fine-tuned to convert table images into executable LaTeX code. They are built upon the Qwen2.5-VL family (Qwen/Qwen2.5-VL-3B-Instruct and Qwen/Qwen2.5-VL-7B-Instruct), which are general-purpose vision-language models for image-text understanding. More details about the base models are available at: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct and https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct.

The models are adapted for structured table-to-LaTeX generation using a two-stage post-training pipeline consisting of supervised fine-tuning (SFT) followed by reinforcement learning (RL). This fine-tuning introduces improved capability for structured generation through a Component-Specific Policy Optimization (CSPO) framework. By using component-specific rewards and token-level credit assignment, this approach mitigates reward ambiguity that arises when heterogeneous aspects (e.g., structure, content, and style) are aggregated into a single reward signal, enabling more precise optimization for generating LaTeX tables with accurate structure, content, and style.

Potential limitations include limited evaluation on non-academic or non-English table data.

### 1.1 Alignment approach

The model employs a reinforcement learning–based alignment approach using automated compiler feedback derived from the TableTex dataset. Instead of relying on human preference annotations, the alignment signals are based on LaTeX compilation outcomes (e.g., success or failure) and component-level evaluations of generated outputs, including structure, content, and style. This approach targets the specific safety objective of output reliability and verifiability in code-structured generation tasks, helping reduce invalid or non-compilable LaTeX outputs by grounding the training process in executable feedback.

---

## 2. Usage

### 2.1 Primary use cases

CSPO is designed for structured Table-to-LaTeX generation tasks in academic and research settings. Its primary use case is converting structured table data (e.g., from scientific papers, technical reports) into compilable LaTeX source code. The model excels at tasks requiring high structural fidelity, including preserving row/column layout, merged cells, alignment, and formatting conventions consistent with academic LaTeX standards. It is suitable for use by researchers and developers building document automation tools, scientific publishing workflows, and structured data rendering pipelines.

### 2.2 Out-of-scope use cases

CSPO is not designed for general-purpose LaTeX document generation beyond structured table conversion. The model has not been evaluated for multilingual table content and may underperform on non-English table data. CSPO should not be used in high-stakes or safety-critical applications, including legal, financial, medical, or regulatory document generation, without extensive additional validation. It is not suitable for production deployment without further testing, red-teaming, and safety evaluation.

### 2.3 Distribution channels

If released, the CSPO models (Qwen2.5-VL-3B-CSPO and Qwen2.5-VL-7B-CSPO) may be distributed through public access via open-source repositories (e.g., Hugging Face or GitHub).

### 2.4 Input formats

The model accepts multimodal inputs consisting of a table image and a textual instruction. The input is typically provided as an image accompanied by a prompt describing the task, such as converting the table into LaTeX code, and follows an instruction-based interaction format consistent with Qwen2.5-VL. For example, a typical input may include a table image and a prompt like: “Please generate complete LaTeX code for the table in the image, including the table body and the full preamble.”

### 2.5 Technical requirements and integration guidance

For the CSPO model, a GPU with 40 GB VRAM is generally sufficient for inference under standard deployment settings. Hardware requirements may vary depending on numerical precision, input image resolution, and output length. PyTorch 2.0+ and HuggingFace Transformers are required. A functional LaTeX distribution (e.g., TeX Live with pdflatex) is recommended for validating generated outputs.

The model can be integrated into document processing pipelines, scientific publishing workflows, or research automation systems. It is not suitable for real-time consumer applications or autonomous decision-making systems without human oversight. Supported platforms include Linux-based server environments; compatibility with Windows and macOS is not guaranteed.

### 2.6 Responsible AI considerations

CSPO is a research model and has not undergone systematic safety red-teaming. Its outputs are LaTeX code rather than natural language, which limits certain categories of harmful content generation; however, malformed or incorrect LaTeX outputs could cause downstream rendering errors in automated pipelines. The model may inherit biases present in the TableTex training dataset, which is derived from arXiv papers and may over-represent certain academic domains, table styles, and formatting conventions. CSPO has not been evaluated for fairness across different table content types, languages, or domains beyond academic scientific tables.

Developers integrating CSPO into applications should implement compiler-based output validation, apply human review for high-stakes document generation, and evaluate performance on their specific use case before deployment.

---

## 3. Quality and performance evaluation

CSPO was evaluated on the TableTex benchmark using a hierarchical, multi-level assessment framework designed for structured table image to LaTeX generation tasks. Global metrics include TEDS (Tree Edit Distance Similarity) for structural and semantic similarity, and Compilation Rate (R) for executable output correctness. Fine-grained evaluation covers Structural correctness (S), Content fidelity (C), and Stylistic fidelity (Y), each assessed at the table level as binary correctness. The composite Overall Fidelity (OF = S ∧ C ∧ Y ∧ R) metric requires simultaneous correctness across all dimensions.

CSPO demonstrates improved performance over supervised fine-tuning (SFT) baselines and standard reinforcement learning approaches on overall fidelity and fine-grained metrics, including structural correctness, content fidelity, and stylistic fidelity. These evaluations primarily reflect performance on academic table image data derived from arXiv and do not cover broader tasks such as open-domain reasoning or multilingual generation. Detailed results are reported in the accompanying ACL’2026 paper.

### 3.1 Safety evaluation and red-teaming

As a research-oriented model specialized in table-to-LaTeX conversion, no formal red-teaming or broad adversarial safety evaluations were conducted prior to release. The primary evaluation method used to assess safety is the integration of an automated LaTeX compiler feedback loop, which quantitatively ensures that the model's outputs are structurally valid and executable, thereby reducing the risk of generating malformed or non-compilable code that could disrupt downstream document pipelines.

The risk of generating disallowed content—such as sexual, violent, hateful, or self-harm content—is significantly reduced because the model's objective is restricted to structured table reconstruction rather than free-form natural language generation. Furthermore, the training data (TableTex) is derived from publicly available academic sources under permissive licenses (CC BY, CC BY-SA, CC0, and CC BY-NC), helping reduce risks related to copyright and intellectual property (IP). No specific evaluations for jailbreak robustness were conducted, given the model’s constrained generation scope.

Developers are strongly advised to implement appropriate safeguards when integrating CSPO into downstream applications. These should include sandboxed LaTeX execution environments to mitigate potential code execution risks, additional compiler-based validation, and human review for high-risk or consequential use cases where data accuracy is critical.

---

## 4. Data overview

### 4.1 Training, testing, and validation datasets

To learn more about the training data used for Qwen2.5-VL-3B-CSPO and Qwen2.5-VL-7B-CSPO, please refer to the full data card: https://huggingface.co/datasets/yunfanyang1/TableTex.

#### 4.1.1 Size of dataset and characteristics

We construct TableTex, a benchmark comprising 19000 pairs of table images and renderable LaTeX codes. In order to support complete table generation, each table code contains:
* (i) necessary package declarations to ensure its compilation correctness;
* (ii) the caption of table[^1];
* (iii) the body of the table that preserves the rich styles and formatting of the original table.

[^1]: Existing datasets usually lack table captions and sample-wise compilation package specifications, which we include for table completeness.

The dataset is curated from scientific articles collected from arXiv, where renderable table code is directly extracted from LaTeX sources. The articles span six major categories—Computer Science, Mathematics, Economics, Electrical Engineering and Systems Science, Quantitative Finance, and Statistics—covering publications from 2012 to 2025. Each extracted LaTeX snippet is rendered into an image, resulting in tables with diverse aspect ratios, resolutions, and visual layouts.

* **4.1.2.A Text training data size:** Less than 1 billion tokens
* **4.1.2.B Text training data content:** LaTeX table code extracted from publicly available academic papers on arXiv, covering multiple domains including Computer Science, Mathematics, Economics, Electrical Engineering and Systems Science, Quantitative Finance, and Statistics. License-filtered to CC BY, CC BY-SA, CC0, and CC BY-NC.
* **4.1.2.C Image training data size:** Less than 1 million images
* **4.1.2.D Image training data content:** Rendered images of tables extracted from publicly available academic papers on arXiv, covering multiple domains including Computer Science, Mathematics, Economics, Electrical Engineering and Systems Science, Quantitative Finance, and Statistics.
* **4.1.3.A Audio training data size:** Not applicable. Audio data is not part of the training data
* **4.1.3.B Audio training data content:** Not applicable.
* **4.1.4.A Video training data size:** Not applicable. Video data is not part of the training data
* **4.1.4.B Video training data content:** Not applicable.
* **4.1.5.A Other training data size:** Not applicable.
* **4.1.5.B Other training data content:** Not applicable.
* **4.1.6 Latest date of data (acquisition/collection for model training):** 28-Nov-2025
* **4.1.7 Is data collection ongoing to update the model with new data collection after deployment?** No
* **4.1.8 Date the training dataset was first used to train the model:** Nov 2025
* **4.1.9 Rationale or purpose of data selection:** The TableTex dataset was selected as the most appropriate publicly available source for structured table-to-LaTeX generation research. It provides paired table images and corresponding LaTeX code extracted from arXiv papers under permissive licenses, enabling reproducible research and replication of scientific findings.

### 4.2 List of data sources

* **4.2.1 Publicly available datasets**
  * **4.2.2 Have you used publicly available datasets to train the model?** No
* **4.2.2 Private non-publicly available datasets obtained from third parties**
  * **4.2.2.1 Datasets commercially licensed by rightsholders or their representatives**
    * **4.2.2.1A Have you concluded transactional commercial licensing agreement(s) with rightsholder(s) or with their representatives?** No
  * **4.2.2.2 Private datasets obtained from other third parties**
    * **4.2.2.2.A Have you obtained private datasets from third parties that are not licensed as described in Section 2.2.1, such as data obtained from providers of private databases, or data intermediaries?** No

### 4.3 Personal Data

* **4.3.1 Was personal data used to train the model?** Microsoft follows applicable laws and best practices pertaining to personal data. [Instructions: Do not modify this text]

### 4.4 Synthetic data

* **4.4.1 Was any synthetic AI-generated data used to train the model?** No

### 4.4 Data processing aspects

* **4.4.1 Respect of reservation of rights from text and data mining exception or limitation**
  * **4.4.1A Does this dataset include any data protected by copyright, trademark, or patent?** Microsoft follows applicable laws and best practices for processing data protected by copyright, trademark, or patent.
* **4.4.2 Other information**
  * **4.4.2A Does the dataset include information about consumer groups without revealing individual consumer identities?** Microsoft follows applicable laws and best practices for protecting consumer identities.
  * **4.4.2B Was the dataset cleaned or modified before model training?** Yes

---

## 5. Contact

Requests for additional information can be directed to MSFTAIActRequest@microsoft.com.

**Authorized representative:** Microsoft Ireland Operations Limited 70 Sir John Rogerson’s Quay, Dublin 2, D02 R296, Ireland