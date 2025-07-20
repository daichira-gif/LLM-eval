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
    parser.add_argument("--enable_lora", action="store_true")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.6)

    args = parser.parse_args()

    if args.enable_lora:
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

    ROLL_USER_PROMPT = "From now on, you are an excellent math teacher and always teach your students math problems correctly. And I am one of your students."
    ROLL_ASSISTANT_PROMPT = (
        "That's great to hear! As your math teacher, I'll do my best to explain mathematical concepts correctly so that you can understand them easily."
        "Feel free to ask any math problems or questions you have, and I'll be glad to assist you. Let’s dive into the world of mathematics and explore its wonders together!"
    )

    PROCESS_USER_PROMPT = (
        'Please ensure your response begins with "<reasoning>\n". '
        "Please reason, and put your final answer within \\boxed{}. "
    )
    COT_ZEROS_PROMPT = "Let's think step by step."
    with open(args.input_path, "r", encoding="utf-8") as f:
        data = list(map(json.loads, f))

    messages_list = []
    #    messages_list2 = []
    for d in data:
        user_messages = [
            {"role": "user", "content": ROLL_USER_PROMPT},
            {"role": "assistant", "content": ROLL_ASSISTANT_PROMPT},
            {
                "role": "user",
                "content": PROCESS_USER_PROMPT
                + instruction_prompt
                + d["text"]
                + answer_prompt
                + COT_ZEROS_PROMPT,
            },
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
