#!/bin/sh

ARRAY=(main zeroshot instruct_format_ja instruct_format_en instruct_format_mix step_by_step_ja step_by_step_en step_by_step_mix roleplay_ja roleplay_en deep_breath_ja deep_breath_en re-read_ja re-read_en re-read_mix echo_ja echo_en step_back_ja)
number="basemodel"
model="llm-jp/llm-jp-3-13b-instruct2"
input="./input/math.jsonl"
lora=False
gpu=0.9
temperature=0

output_home="./output/$number"

if [ -e ${output_home} ]; then
  echo "${output_home}は存在します。"
else
  mkdir "${output_home}"
  echo "${output_home}を作成しました。"
fi

INFO_LOG_FILE="${output_home}/info.txt"

for arg in ${ARRAY[@]}
do
  output_dir="${output_home}/${arg}-${number}.jsonl"
  python scripts/$arg.py $model \
  --input_path=$input \
  --output_path=$output_dir \
  --enable_lora=$lora \
  --gpu_memory_utilization=$gpu \
  --temperature=$temperature
  python scripts/eval.py --input_path=$output_dir --log_path=$INFO_LOG_FILE
done
