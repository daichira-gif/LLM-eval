import os
import json
import argparse
from openai import OpenAI

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
    return text[start_idx:i-1].strip().replace(",","")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("model_name", type=str, help="The model identifier on OpenRouter (e.g., google/gemini-pro)")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.6)

    args = parser.parse_args()
 
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )
            
    instruction_prompt = "Q:"
    answer_prompt = "\nA:"
    re_read_prompt = "もう一度問題を読んでみましょう:"

    PROCESS_USER_PROMPT_EN = (
    "Please ensure your response begins with \"<reasoning>\n\". "
    "Please reason, and put your final answer within \\boxed{}. "
)
    PROCESS_USER_PROMPT = (
    "回答は必ず \"<reasoning>\n\" で始まっていることを確認してください。"
    "理由を述べ、最終的な回答を \\boxed{} 内に記入してください。"
)
    COT_ZEROS_PROMPT = (
    "ステップバイステップで考えてみましょう"
)
    ECHO_PROMPT = (
    "問題を繰り返した後、ステップバイステップで考えてみましょう。"
)
    
    with open(args.input_path, "r", encoding="utf-8") as f:
        data = list(map(json.loads, f))

    for d in data:
        # Construct the 3 prompts for the current data item 'd'
        instruction_text_en = PROCESS_USER_PROMPT_EN + instruction_prompt + d["text"]
        instruction_text = PROCESS_USER_PROMPT + instruction_prompt + d["text"]
        re_read_text = re_read_prompt + d["text"]
        answer_text = answer_prompt + COT_ZEROS_PROMPT
        echo_text = answer_prompt + ECHO_PROMPT

        # Prompt 1: echoprompt(ja)
        user_messages1 = instruction_text + echo_text
        # Prompt 2: re-reading(ja)
        user_messages2 = instruction_text + re_read_text + answer_text
        # Prompt 3: re-reading(mix)
        user_messages3 = instruction_text_en + instruction_text + answer_text

        # API Call 1
        completion1 = client.chat.completions.create(
            model=args.model_name,
            messages=[{"role": "user", "content": user_messages1}],
            max_tokens=2048,
            temperature=args.temperature,
        )
        response_text1 = completion1.choices[0].message.content
        d["answer1"] = extract_boxed_answer_rev(response_text1)

        # API Call 2
        completion2 = client.chat.completions.create(
            model=args.model_name,
            messages=[{"role": "user", "content": user_messages2}],
            max_tokens=2048,
            temperature=args.temperature,
        )
        response_text2 = completion2.choices[0].message.content
        d["answer2"] = extract_boxed_answer_rev(response_text2)

        # API Call 3
        completion3 = client.chat.completions.create(
            model=args.model_name,
            messages=[{"role": "user", "content": user_messages3}],
            max_tokens=2048,
            temperature=args.temperature,
        )
        response_text3 = completion3.choices[0].message.content
        d["answer3"] = extract_boxed_answer_rev(response_text3)

        # Logic for final response selection remains the same
        d["response"] = d["answer2"]
        if d["answer1"] == d["answer3"]:
            d["response"] = d["answer3"]

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
