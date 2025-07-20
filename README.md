# LLM Evaluation

This repository contains scripts for evaluating language models on the [Omni-MATH](https://huggingface.co/datasets/KbsdJames/Omni-MATH) dataset.

## Preparing the dataset

Use `scripts/prepare_math_dataset.py` to download and filter the dataset. Only examples with a `difficulty` of 7 or more are kept. The output file path can be configured with the `INPUT_PATH` environment variable.

```bash
export INPUT_PATH=/eval/input/math.jsonl
python scripts/prepare_math_dataset.py
```

## Running evaluation

`scripts/openrouter_eval.py` runs inference using OpenRouter and logs the accuracy. The script reads configuration from environment variables such as `MODEL_NAME`, `OUTPUT_BASE_DIR` and `INFO_LOG_FILE`.

Example:

```bash
export EXPERIMENT_ID=basemodel
export MODEL_NAME=qwen/qwen3-32b:free
export INPUT_PATH=/eval/input/math.jsonl
export OUTPUT_BASE_DIR=/eval/output/${EXPERIMENT_ID}
export INFO_LOG_FILE=${OUTPUT_BASE_DIR}/info.txt
export TEMPERATURE=0
python scripts/openrouter_eval.py
```

Make sure `OPENROUTER_API_KEY` is set in the environment to allow API access.

## Development setup

Install runtime requirements and the additional development tools:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install
```

With the hooks installed, formatting and lint checks are run automatically on
each commit.
