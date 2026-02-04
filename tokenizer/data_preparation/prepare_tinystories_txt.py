import json
import os

from tqdm import tqdm

from datasets import load_dataset

OUT_DIR = "../../datasets/tokenization/"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PATH = os.path.join(OUT_DIR, "tinystories_train.jsonl")


def normalize_text(text: str) -> str:
    """
    Normalize raw text:
    - unify line endings
    - collapse all whitespace into single spaces
    - strip leading/trailing spaces
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = " ".join(text.split())
    return text


def main():
    # Load TinyStories in streaming mode to avoid loading the full dataset into memory
    ds = load_dataset(
        "roneneldan/TinyStories",
        split="train",
        streaming=True,
    )

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for ex in tqdm(ds, desc="writing tinystories_train.jsonl"):
            text = ex.get("text", "")
            if not isinstance(text, str):
                continue

            text = normalize_text(text)
            if not text:
                continue

            # Write one JSON object per line (JSONL format)
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    print("Done.")
    print("Output:", OUT_PATH)


if __name__ == "__main__":
    main()
