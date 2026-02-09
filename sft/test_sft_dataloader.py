#!/usr/bin/env python3

"""
Quick test for SFTJsonlDataset + DataLoader.
"""

from sft_dataset import IGNORE_INDEX, SFTJsonlDataset, collate_sft_batch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader


def main():
    tok = Tokenizer.from_file("../tokenizer_32k/tokenizer.json")

    # IMPORTANT: you must choose a pad_token_id.
    # If your tokenizer has a dedicated PAD token, use it.
    # Otherwise, a common pragmatic choice is to reuse EOS id.
    pad_id = tok.token_to_id("[PAD]")
    if pad_id is None:
        pad_id = tok.token_to_id("[EOS]")
    if pad_id is None:
        # Last resort: 0
        pad_id = 0

    ds = SFTJsonlDataset(
        jsonl_path="../dataset/mixed_v2/ultrachat.canon.jsonl",
        tokenizer=tok,
        max_length=1024,
        limit=50,
    )

    dl = DataLoader(
        ds,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda b: collate_sft_batch(b, pad_token_id=pad_id),
    )

    batch = next(iter(dl))
    print("input_ids:", batch["input_ids"].shape)
    print("labels:", batch["labels"].shape)
    print("attention_mask:", batch["attention_mask"].shape)

    # Sanity checks
    supervised = (batch["labels"] != IGNORE_INDEX).sum().item()
    total = batch["labels"].numel()
    print(f"supervised tokens in batch: {supervised}/{total}")

    # Show a tiny decoded preview from sample 0
    ids0 = batch["input_ids"][0].tolist()
    text0 = tok.decode([i for i in ids0 if i != pad_id])
    print("\nDecoded sample 0 (first 800 chars):")
    print(text0[:800])

    print("bos_id =", tok.token_to_id("[BOS]"))
    print("eos_id =", tok.token_to_id("[EOS]"))
    print("pad_id =", pad_id)


if __name__ == "__main__":
    main()
