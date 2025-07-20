#!/bin/sh

ARRAY=(main zeroshot instruct_format_ja instruct_format_en instruct_format_mix step_by_step_ja step_by_step_en step_by_step_mix roleplay_ja roleplay_en deep_breath_ja deep_breath_en re-read_ja re-read_en re-read_mix echo_ja echo_en step_back_ja)

# Experiment configuration
EXPERIMENT_ID="${EXPERIMENT_ID:-basemodel}"
MODEL_NAME="${MODEL_NAME:?MODEL_NAME is not set}"
INPUT_PATH="${INPUT_PATH:-./input/math.jsonl}"
USE_OPENROUTER="${USE_OPENROUTER:-false}"

# Optional parameters for local inference
lora=${ENABLE_LORA:-False}
gpu=${GPU_MEMORY_UTILIZATION:-0.9}
temperature=${TEMPERATURE:-0}

OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-./output/${EXPERIMENT_ID}}"
INFO_LOG_FILE="${INFO_LOG_FILE:-${OUTPUT_BASE_DIR}/info.txt}"

mkdir -p "${OUTPUT_BASE_DIR}"

if [ "${USE_OPENROUTER}" = "true" ]; then
  python scripts/openrouter_eval.py
else
  for arg in ${ARRAY[@]}
  do
    output_dir="${OUTPUT_BASE_DIR}/${arg}-${EXPERIMENT_ID}.jsonl"
    python scripts/$arg.py $MODEL_NAME \
      --input_path=$INPUT_PATH \
      --output_path=$output_dir \
      --enable_lora=$lora \
      --gpu_memory_utilization=$gpu \
      --temperature=$temperature
    python scripts/eval.py --input_path=$output_dir --log_path=$INFO_LOG_FILE
  done
fi
