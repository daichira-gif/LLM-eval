#!/bin/sh

ARRAY=(main zeroshot instruct_format_ja instruct_format_en instruct_format_mix step_by_step_ja step_by_step_en step_by_step_mix roleplay_ja roleplay_en deep_breath_ja deep_breath_en re-read_ja re-read_en re-read_mix echo_ja echo_en step_back_ja)

# Experiment name used for output file names.
EXPERIMENT_ID="${EXPERIMENT_ID:-basemodel}"
# Model path or identifier. Can be overridden with the MODEL_NAME environment variable.
MODEL_NAME="${MODEL_NAME:-llm-jp/llm-jp-3-13b-instruct2}"
# Path to the input dataset.
INPUT_PATH="${INPUT_PATH:-./input/math.jsonl}"
LORA="${LORA:-False}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
TEMPERATURE="${TEMPERATURE:-0}"

# Base directory for outputs
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-./output/${EXPERIMENT_ID}}"

if [ -e "${OUTPUT_BASE_DIR}" ]; then
  echo "${OUTPUT_BASE_DIR}は存在します。"
else
  mkdir -p "${OUTPUT_BASE_DIR}"
  echo "${OUTPUT_BASE_DIR}を作成しました。"
fi

INFO_LOG_FILE="${INFO_LOG_FILE:-${OUTPUT_BASE_DIR}/info.txt}"

if [ "${USE_OPENROUTER}" = "true" ]; then
  python scripts/openrouter_eval.py
else
  for arg in ${ARRAY[@]}
  do
    output_dir="${OUTPUT_BASE_DIR}/${arg}-${EXPERIMENT_ID}.jsonl"
    python scripts/$arg.py $MODEL_NAME \
      --input_path=$INPUT_PATH \
      --output_path=$output_dir \
      --enable_lora=$LORA \
      --gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
      --temperature=$TEMPERATURE
    python scripts/eval.py --input_path=$output_dir --log_path=$INFO_LOG_FILE

  done
fi
