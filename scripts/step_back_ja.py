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
    formula_prompt = "公式:"
    answer_prompt = "\nA:"

    STEP_BACK_PROMPT = "この質問の背後にある数学の公式は何ですか？"

    PROCESS_USER_PROMPT = (
        '回答は必ず "<reasoning>\n" で始まっていることを確認してください。'
        "理由を述べ、最終的な回答を \\boxed{} 内に記入してください。"
    )
    COT_ZEROS_PROMPT = "ステップバイステップで考えてみましょう。"
    with open(args.input_path, "r", encoding="utf-8") as f:
        data = list(map(json.loads, f))

    messages_list0 = []
    for d in data:
        user_messages0 = [
            {
                "role": "user",
                "content": STEP_BACK_PROMPT + instruction_prompt + d["text"],
            }
        ]

        messages_list0.append(user_messages0)

    sampling_params0 = SamplingParams(
        max_tokens=128,
        temperature=args.temperature,
    )

    outputs0 = llm.chat(messages_list0, sampling_params=sampling_params0)
    for i, output in enumerate(outputs0):
        # \boxed{...} の中身を抽出する関数で回答を取得
        data[i]["formula"] = output.outputs[0].text

    messages_list1 = []
    for d in data:
        question_text = instruction_prompt + d["text"] + "\n"
        formula_text = formula_prompt + d["formula"] + "\n"
        answer_text = answer_prompt + COT_ZEROS_PROMPT
        user_messages1 = [
            {
                "role": "user",
                "content": PROCESS_USER_PROMPT
                + question_text
                + formula_text
                + answer_text,
            }
        ]

        messages_list1.append(user_messages1)

    sampling_params1 = SamplingParams(
        max_tokens=2048,
        temperature=args.temperature,
    )

    outputs1 = llm.chat(messages_list1, sampling_params=sampling_params1)
    for i, output in enumerate(outputs1):
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
