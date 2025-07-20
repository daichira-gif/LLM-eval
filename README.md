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

To evaluate a local model instead, execute `round_test.sh`. When `USE_OPENROUTER=true` is set the script will invoke `openrouter_eval.py`; otherwise it runs the local inference scripts bundled in this repository.

## Environment variables

Several scripts read configuration from environment variables:

- `MODEL_NAME` – model identifier or path. Used by both `round_test.sh` and `openrouter_eval.py`.
- `INPUT_PATH` – path to the evaluation dataset. Defaults to `./input/math.jsonl`.
- `OUTPUT_BASE_DIR` – directory in which result files are created. Defaults to `./output/${EXPERIMENT_ID}`.
- `EXPERIMENT_ID` – experiment name appended to output files. Defaults to `basemodel`.
- `TEMPERATURE` – sampling temperature (default `0`).
- `LORA` and `GPU_MEMORY_UTILIZATION` – options for local evaluation scripts.
- `USE_OPENROUTER` – if set to `true`, `round_test.sh` runs `openrouter_eval.py` instead of local scripts.
- `INFO_LOG_FILE` – log file path (default `${OUTPUT_BASE_DIR}/info.txt`).
## Development setup

Install runtime requirements and the additional development tools:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install
```

With the hooks installed, formatting and lint checks are run automatically on
