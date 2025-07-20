import os
import json
import argparse
from vllm import LLM, SamplingParams


def extract_boxed_answer_rev(text: str) -> str:
    """
    テキスト中から最初の \boxed{...} の中身（ネストを考慮）を抽出する。
    例: r"\boxed{\frac{\pi}{6}}" -> "\frac{\pi}{6}"
    """
    key = r"\boxed{"
    start_idx = text.find(key)
    if start_idx == -1:
        return ""
    # \boxed{ の直後の位置を開始位置とする
    start_idx += len(key)
    brace_count = 1  # 最初の { を既にカウント
    i = start_idx
    while i < len(text) and brace_count > 0:
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
        i += 1
    # i-1 が閉じ括弧に対応する位置
    return text[start_idx : i - 1].strip().replace(",", "")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("model_path", type=str)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--enable_lora", type=str, default=False)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.6)

    args = parser.parse_args()

    if args.enable_lora == True:
        llm = LLM(
            model=args.model_path,
            enable_lora=True,
            tensor_parallel_size=args.num_gpus,
            seed=0,
        )
    else:
        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=args.num_gpus,
            seed=0,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

    instruction_prompt = "Q:"
    answer_prompt = "\nA:"

    PROCESS_USER_PROMPT = (
        'Please ensure your response begins with "<reasoning>\n". '
        "Please reason, and put your final answer within \\boxed{}. "
        '回答は必ず "<reasoning>\n" で始まっていることを確認してください。'
        "理由を述べ、最終的な回答を \\boxed{} 内に記入してください。"
    )
    with open(args.input_path, "r", encoding="utf-8") as f:
        data = list(map(json.loads, f))

    messages_list = []
    for d in data:
        user_messages = [
            {
                "role": "user",
                "content": PROCESS_USER_PROMPT
                + instruction_prompt
                + d["text"]
                + answer_prompt,
            }
        ]
        messages_list.append(user_messages)

    sampling_params1 = SamplingParams(
        max_tokens=2048,
        temperature=args.temperature,
    )

    outputs = llm.chat(messages_list, sampling_params=sampling_params1)
    for i, output in enumerate(outputs):
        # \boxed{...} の中身を抽出する関数で回答を取得
        boxed_answer = extract_boxed_answer_rev(output.outputs[0].text)
        data[i]["response"] = boxed_answer
        data[i]["processed"] = output.outputs[0].text

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
