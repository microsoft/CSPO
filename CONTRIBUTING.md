# Contributing

This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately, for example with a label or comment.
Simply follow the instructions provided by the bot. You will only need to do this once
across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## How to Contribute

We welcome issues and pull requests related to:

- Bug fixes in CSPO training, reward computation, or evaluation.
- Improvements to documentation, setup instructions, or examples.
- Reproducibility improvements for SFT, CSPO, GRPO, or fine-grained evaluation.
- Small, focused extensions that are aligned with table image-to-LaTeX generation.

Before opening a pull request, please:

- Check whether a related issue or pull request already exists.
- Keep the change focused and avoid unrelated refactors.
- Avoid committing datasets, model checkpoints, generated outputs, logs, credentials, or local environment files.
- Preserve upstream license headers and attribution notices where present.

## Development Setup

Create and activate the environment:

```bash
conda create -n cspo python=3.12
conda activate cspo
```

Install system dependencies used by the LaTeX compilation pipeline:

```bash
sudo apt-get update
sudo apt-get install -y ghostscript poppler-utils
```

Install Python dependencies:

```bash
bash setup.sh
```

If you need SFT workflows, install LLaMA-Factory extras:

```bash
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
cd ..
```

## Coding Guidance

- Follow the existing style and structure of the file you are modifying.
- Keep scripts configurable through command-line arguments or environment variables rather than hard-coded local paths.
- Do not add API keys, tokens, local absolute paths, logs, checkpoints, or generated data to the repository.
- When adapting code from another project, preserve license headers where present and add attribution when needed.
- Prefer small pull requests that are easy to review.

## Testing and Validation

Please run the relevant workflow for your change when possible:

- For training script changes, verify that the target script starts correctly and resolves paths from the repository root.
- For inference changes, run a small inference pass with `src/eval/inference_CSPO.py` if data and checkpoints are available.
- For metric changes, run `src/eval/merge_and_eval.py` or the fine-grained evaluation scripts on a small sample.
- For documentation-only changes, check that links and commands point to existing files.

If you cannot run a workflow because datasets, checkpoints, or API credentials are unavailable, please mention that in the pull request.

## Roadmap

Planned areas of work include:

- Releasing training and evaluation datasets.
- Releasing pretrained CSPO checkpoints.
- Improving documentation for reproducing SFT, CSPO, GRPO, and fine-grained evaluation.
- Extending component-aware optimization and evaluation for structured generation tasks.
