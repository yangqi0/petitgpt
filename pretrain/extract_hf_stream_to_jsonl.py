#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_hf_stream_to_jsonl.py

Stream a HuggingFace dataset (no full download) and write JSONL:
  {"text": "..."} per line.

Requirements:
  pip install datasets

Examples:

  # OpenWebMath
  python extract_hf_stream_to_jsonl.py \
    --dataset open-web-math/open-web-math \
    --split train \
    --text_field text \
    --out datasets/raw/openwebmath.raw.jsonl \
    --max_rows 2000000 \
    --shuffle_buffer 20000 \
    --seed 1234

  # ProofPile-2 (algebraic-stack subset)
  python extract_hf_stream_to_jsonl.py \
    --dataset EleutherAI/proof-pile-2 \
    --subset algebraic-stack \
    --split train \
    --text_field text \
    --out datasets/raw/proofpile2_algebraic.raw.jsonl \
    --max_rows 800000 \
    --shuffle_buffer 20000 \
    --seed 1234
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import islice

from datasets import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="HF dataset id, e.g. open-web-math/open-web-math")
    ap.add_argument("--subset", default="", help="Optional subset/config name, e.g. algebraic-stack")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text_field", default="text")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_rows", type=int, default=0, help="0 means all rows (may be huge).")
    ap.add_argument("--shuffle_buffer", type=int, default=0, help="0 disables streaming shuffle.")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.subset:
        ds = load_dataset(args.dataset, args.subset, split=args.split, streaming=True)
    else:
        ds = load_dataset(args.dataset, split=args.split, streaming=True)

    if args.shuffle_buffer and args.shuffle_buffer > 0:
        ds = ds.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)

    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        it = ds if args.max_rows <= 0 else islice(ds, args.max_rows)
        for row in it:
            t = row.get(args.text_field, None)
            if not isinstance(t, str):
                continue
            t = t.strip()
            if not t:
                continue
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
            n += 1
            if n % 200000 == 0:
                print(f"[progress] {n} rows", flush=True)

    print(f"[done] wrote {n} rows -> {args.out}")


if __name__ == "__main__":
    main()
