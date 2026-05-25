# Transparency Documentation

**Project:** CSPO: Alleviating Reward Ambiguity for Structured Table-to-LaTeX Generation

---

## DATASET DETAILS

### DATASET CONTENTS

TableTex consists of 19,000 instances of paired data, where each data point represents a table image and its corresponding LaTeX code.

Each instance includes:
* A rendered table image
* The corresponding LaTeX source code, including structure, textual content, stylistic elements, caption, and required packages.

Each instance is associated with a ground-truth LaTeX sequence. The data was collected and generated between 2012 and 2025 from scientific articles.

---

## DATA CREATION & PROCESSING

TableTex was created by adapting existing data sources from scientific articles on arXiv. The data consists of LaTeX source files containing tables. The original data was collected by arXiv as part of its open-access repository.

**Creation process:**
* Extract LaTeX table code
* Render tables into images
* Pair images with LaTeX code

The dataset was created using automated pipelines. No arbitrary web crawling was performed; data is restricted to licensed arXiv sources.

---

## PEOPLE & IDENTIFIERS

* Data points do not correspond to individual people’s behaviors or personal attributes.
* The dataset does not include data pertaining to children.
* No personally identifiable information (PII) is included.
* No anonymization was required.
* The dataset is not believed to contain identifiable information.

---

## SENSITIVE OR HARMFUL CONTENT

* The dataset does not contain sensitive personal information such as racial, political, religious, health, financial, or criminal data.
* It consists solely of scientific tables and LaTeX code.

---

## ADDITIONAL NOTES

* Data is sourced from arXiv papers under permissive licenses (CC BY, CC BY-SA, CC0, and CC BY-NC).
* Usage is restricted to research purposes in compliance with licensing.
* Images are synthetically rendered from LaTeX.