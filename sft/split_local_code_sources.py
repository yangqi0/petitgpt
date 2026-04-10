#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split a local canonical JSONL source into train/val JSONL files.

Input rows are expected to look like:
  {"messages": [...], ...}

This is meant for small, high-value local code datasets such as:
  - local_mbpp_simple.canon.jsonl
  - local_code_fix_simple.canon.jsonl

Why this exists:
  For tiny local sources, letting the main SFT prepare script auto-build val first can
  consume too much of the source into validation, leaving too little for training.
  This script makes the split explicit and reproducible.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_train", required=True)
    ap.add_argument("--out_val", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--val_ratio", type=float, default=0.08)
    ap.add_argument(
        "--max_val_examples",
        type=int,
        default=0,
        help="Optional hard cap on validation examples. 0 means no cap.",
    )
    args = ap.parse_args()

    rows = read_jsonl(args.in_jsonl)
    if not rows:
        raise SystemExit(f"empty input: {args.in_jsonl}")

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    n_total = len(rows)
    n_val = max(1, int(round(n_total * args.val_ratio)))
    if args.max_val_examples > 0:
        n_val = min(n_val, args.max_val_examples)
    n_val = min(n_val, n_total - 1)

    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    write_jsonl(args.out_train, train_rows)
    write_jsonl(args.out_val, val_rows)

    print(f"input={args.in_jsonl}")
    print(f"total={n_total}")
    print(f"train={len(train_rows)} -> {args.out_train}")
    print(f"val={len(val_rows)} -> {args.out_val}")


if __name__ == "__main__":
    main()
