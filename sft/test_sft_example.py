#!/usr/bin/env python3

import json

from build_sft_example import IGNORE_INDEX, build_sft_example
from tokenizers import Tokenizer


def main():
    tok = Tokenizer.from_file("../tokenizer_32k/tokenizer.json")

    # Load one canonical sample
    with open("../dataset/mixed_v2/ultrachat.canon.jsonl", encoding="utf-8") as f:
        sample = json.loads(f.readline())

    ex = build_sft_example(tok, sample["messages"], keep_rendered_text=True)

    print("Rendered text (first 60 lines):")
    lines = ex.rendered_text.splitlines()
    print("\n".join(lines[:60]))

    # Show how many tokens are supervised
    supervised = sum(1 for x in ex.labels if x != IGNORE_INDEX)
    print(f"\nTotal tokens: {len(ex.input_ids)}")
    print(f"Supervised tokens (assistant spans): {supervised}")

    # Print a short aligned preview: token + label_mask
    tokens = tok.decode(ex.input_ids[:200])
    print("\nDecoded preview (first 800 chars):")
    print(tokens[:800])


if __name__ == "__main__":
    main()
