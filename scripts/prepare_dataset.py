import json
from datasets import load_dataset

def main():
    # Load the dataset from Hugging Face
    dataset = load_dataset("KbsdJames/Omni-MATH", split="train")

    # Prepare the data in the desired format
    formatted_data = []
    for i, item in enumerate(dataset):
        formatted_item = {
            "id": i + 1,
            "text": item["query"],
            "gold": str(item["solution"]),
            "response": "",
            "type": item["domain"],
            "level": item["difficulty"]
        }
        formatted_data.append(formatted_item)

    # Write the formatted data to a jsonl file
    with open("input/omni_math.jsonl", "w", encoding="utf-8") as f:
        for item in formatted_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
