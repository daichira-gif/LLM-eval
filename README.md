# LLM Evaluation

This repository contains scripts for evaluating language models on the [Omni-MATH](https://huggingface.co/datasets/KbsdJames/Omni-MATH) dataset.

## Preparing the dataset

Use `scripts/prepare_math_dataset.py` to download and filter the dataset. Only examples with a `difficulty` of 7 or more are kept. The output file path can be configured with the `INPUT_PATH` environment variable.

```bash
export INPUT_PATH=/eval/input/math.jsonl
python scripts/prepare_math_dataset.py
```

## Running evaluation

Evaluation behavior is controlled entirely through environment variables.

### Required variables

- `MODEL_NAME` – Local model path or OpenRouter model identifier.
- `INPUT_PATH` – Path to the JSONL dataset.
- `OUTPUT_BASE_DIR` – Directory where outputs are written.
- `EXPERIMENT_ID` – Label appended to output files.
- `INFO_LOG_FILE` – Log file path (defaults to `$OUTPUT_BASE_DIR/info.txt`).
- `TEMPERATURE` – Sampling temperature.
- `USE_OPENROUTER` – When set to `true`, `scripts/openrouter_eval.py` is used. Otherwise the local scripts are run.
- `OPENROUTER_API_KEY` – Required only when `USE_OPENROUTER=true`.

### Example (OpenRouter)

```bash
export USE_OPENROUTER=true
export EXPERIMENT_ID=basemodel
export MODEL_NAME=qwen/qwen3-32b:free
export INPUT_PATH=/eval/input/math.jsonl
export OUTPUT_BASE_DIR=/eval/output/${EXPERIMENT_ID}
export INFO_LOG_FILE=${OUTPUT_BASE_DIR}/info.txt
export TEMPERATURE=0
python round_test.sh
```

### Example (local)

```bash
export MODEL_NAME=/path/to/local/model
export INPUT_PATH=/eval/input/math.jsonl
export EXPERIMENT_ID=basemodel
python round_test.sh
```

When `USE_OPENROUTER=true`, ensure `OPENROUTER_API_KEY` is available in the environment.
