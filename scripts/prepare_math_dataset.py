import os
import json
from datasets import load_dataset


def main():
    output_path = os.environ.get("INPUT_PATH", "input/math.jsonl")
    dataset = load_dataset("KbsdJames/Omni-MATH", split="test")

    formatted = []
    idx = 1
    for item in dataset:
        try:
            difficulty = int(item.get("difficulty", 0))
        except Exception:
            try:
                difficulty = int(float(item.get("difficulty", 0)))
            except Exception:
                difficulty = 0
        if difficulty < 7:
            continue
        formatted.append(
            {
                "id": idx,
                "text": item["question"],
                "gold": str(item["solution"]),
                "response": "",
                "type": item.get("domain", ""),
                "level": difficulty,
            }
        )
        idx += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in formatted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
