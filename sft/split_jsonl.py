#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
split_jsonl.py

Split a JSONL file into train/val.

Features:
- Streaming (doesn't load all JSON into memory).
- Deterministic split with a seed.
- Two modes:
  1) --val_ratio (approximate, binomial; one-pass)
  2) --val_n (exact count; two-pass, still memory-light)

Usage examples:

# 1) Ratio split (fast, one-pass)
python split_jsonl.py \
  --input datasets/sft/ultrachat.canon.jsonl \
  --out_train datasets/sft/ultrachat.train.canon.jsonl \
  --out_val datasets/sft/ultrachat.val.canon.jsonl \
  --val_ratio 0.002 \
  --seed 1234

# 2) Exact N for val (two-pass)
python split_jsonl.py \
  --input datasets/sft/ultrachat.canon.jsonl \
  --out_train datasets/sft/ultrachat.train.canon.jsonl \
  --out_val datasets/sft/ultrachat.val.canon.jsonl \
  --val_n 2000 \
  --seed 1234
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Set


def count_lines(path: str) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def choose_val_indices(total: int, val_n: int, seed: int) -> Set[int]:
    """
    Choose exactly val_n distinct indices in [0, total).
    Uses random.sample on range(total), which is memory-light for moderate totals.
    For very large totals (tens of millions), you'd want reservoir sampling.
    """
    if val_n < 0 or val_n > total:
        raise ValueError(f"val_n must be in [0, {total}], got {val_n}")
    rng = random.Random(seed)
    return set(rng.sample(range(total), val_n))


def atomic_write_replace(path: str, tmp_path: str) -> None:
    os.replace(tmp_path, path)


def split_ratio_one_pass(
    inp: str, out_train: str, out_val: str, val_ratio: float, seed: int
) -> None:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("--val_ratio must be between 0 and 1 (exclusive).")

    rng = random.Random(seed)

    t_tmp = out_train + ".tmp"
    v_tmp = out_val + ".tmp"

    n_total = 0
    n_train = 0
    n_val = 0

    with open(inp, "r", encoding="utf-8") as fi, \
         open(t_tmp, "w", encoding="utf-8") as ft, \
         open(v_tmp, "w", encoding="utf-8") as fv:
        for line in fi:
            n_total += 1
            if rng.random() < val_ratio:
                fv.write(line)
                n_val += 1
            else:
                ft.write(line)
                n_train += 1

    atomic_write_replace(out_train, t_tmp)
    atomic_write_replace(out_val, v_tmp)

    print(f"[*] Done (ratio mode). total={n_total} train={n_train} val={n_val} val_ratio≈{(n_val/max(1,n_total)):.6f}")


def split_exact_n_two_pass(
    inp: str, out_train: str, out_val: str, val_n: int, seed: int
) -> None:
    total = count_lines(inp)
    print(f"[*] Counted lines: total={total}")

    val_idx = choose_val_indices(total=total, val_n=val_n, seed=seed)
    print(f"[*] Selected val indices: {len(val_idx)} (exact)")

    t_tmp = out_train + ".tmp"
    v_tmp = out_val + ".tmp"

    n_train = 0
    n_val = 0

    with open(inp, "r", encoding="utf-8") as fi, \
         open(t_tmp, "w", encoding="utf-8") as ft, \
         open(v_tmp, "w", encoding="utf-8") as fv:
        for i, line in enumerate(fi):
            if i in val_idx:
                fv.write(line)
                n_val += 1
            else:
                ft.write(line)
                n_train += 1

    atomic_write_replace(out_train, t_tmp)
    atomic_write_replace(out_val, v_tmp)

    print(f"[*] Done (exact-N mode). total={total} train={n_train} val={n_val} val_ratio={(n_val/max(1,total)):.6f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input JSONL")
    ap.add_argument("--out_train", required=True, help="output train JSONL")
    ap.add_argument("--out_val", required=True, help="output val JSONL")
    ap.add_argument("--seed", type=int, default=1234)

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--val_ratio", type=float, default=None, help="validation ratio (one-pass, approximate)")
    g.add_argument("--val_n", type=int, default=None, help="exact number of validation lines (two-pass)")

    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_train) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_val) or ".", exist_ok=True)

    if args.val_ratio is not None:
        split_ratio_one_pass(args.input, args.out_train, args.out_val, args.val_ratio, args.seed)
    else:
        split_exact_n_two_pass(args.input, args.out_train, args.out_val, args.val_n, args.seed)


if __name__ == "__main__":
    main()
