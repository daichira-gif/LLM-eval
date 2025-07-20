import os
import json
import logging
from logging import getLogger, FileHandler, Formatter
from openai import OpenAI

# Helper functions from eval.py

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string

def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1

def extract_boxed_answer_rev(text: str) -> str:
    key = r"\\boxed{"
    start_idx = text.find(key)
    if start_idx == -1:
        return ""
    start_idx += len(key)
    brace_count = 1
    i = start_idx
    while i < len(text) and brace_count > 0:
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
        i += 1
    return text[start_idx:i-1].strip().replace(",", "")


def main():
    model_name = os.environ.get("MODEL_NAME")
    input_path = os.environ.get("INPUT_PATH")
    temperature = float(os.environ.get("TEMPERATURE", 0))
    output_base = os.environ.get("OUTPUT_BASE_DIR", "output")
    info_log_file = os.environ.get("INFO_LOG_FILE", os.path.join(output_base, "info.txt"))
    experiment_id = os.environ.get("EXPERIMENT_ID", "exp")
    output_path = os.path.join(output_base, f"openrouter-{experiment_id}.jsonl")

    os.makedirs(output_base, exist_ok=True)

    logger = getLogger("openrouter_eval")
    logger.setLevel(logging.INFO)
    fh = FileHandler(info_log_file, "a")
    fh.setFormatter(Formatter('%(name)s - %(message)s'))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ.get("OPENROUTER_API_KEY"))

    PROCESS_USER_PROMPT = (
        "回答は必ず \"<reasoning>\\n\" で始まっていることを確認してください。"
        "理由を述べ、最終的な回答を \\boxed{} 内に記入してください。"
    )
    instruction_prompt = "Q:"
    answer_prompt = "\nA:"

    with open(input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    correct = 0
    total = 0

    for d in data:
        total += 1
        user_messages = PROCESS_USER_PROMPT + instruction_prompt + d["text"] + answer_prompt
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": user_messages}],
            max_tokens=2048,
            temperature=temperature,
        )
        response_text = completion.choices[0].message.content
        answer = extract_boxed_answer_rev(response_text)
        d["response"] = answer
        d["processed"] = response_text
        if is_equiv(d["gold"].strip(), answer.strip()):
            correct += 1
            logger.info(f"正解しました。 id: {d['id']}, gold: {d['gold']}, response: {answer}")

    accuracy = correct / total if total else 0
    logger.info(f"評価 input: {total}, output: {correct}, accuracy: {accuracy}")

    with open(output_path, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
