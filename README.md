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
export OPENROUTER_API_KEY=your_openrouter_key
python scripts/openrouter_eval.py
```

Make sure `OPENROUTER_API_KEY` is set in the environment to allow API access.

## OpenRouter evaluation with round_test.sh

`round_test.sh` can be modified to call `scripts/openrouter_eval.py` so that multiple evaluation prompts are executed in sequence. Example environment setup:

```bash
export MODEL_NAME=qwen/qwen3-32b:free
export INPUT_PATH=/eval/input/math.jsonl
export EXPERIMENT_ID=basemodel
export OUTPUT_BASE_DIR=/eval/output/${EXPERIMENT_ID}
export INFO_LOG_FILE=${OUTPUT_BASE_DIR}/info.txt
export TEMPERATURE=0
export OPENROUTER_API_KEY=your_openrouter_key
./round_test.sh
```

The script will generate one output file for each prompt in the array and log the overall accuracy.
