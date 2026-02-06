#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Near-duplicate check with fixed-length token windows over uint16 .bin shards.

We hash windows of length W (e.g., 256) with stride S (e.g., 64).
This catches "same content but shifted block boundaries".

Example:
  python pretrain/check_bin_dup_windows.py \
    --dir datasets/pretrain_mix/train \
    --window 256 \
    --stride 64 \
    --max_windows 3000000 \
    --topk 20
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from collections import Counter

import numpy as np


def list_bins(d: str) -> list[Path]:
    p = Path(d)
    files = sorted([x for x in p.glob("*.bin") if x.is_file()])
    if not files:
        raise FileNotFoundError(f"No .bin files under: {d}")
    return files


def hash_u16(arr_u16: np.ndarray) -> int:
    h = hashlib.blake2b(arr_u16.tobytes(order="C"), digest_size=8).digest()
    return int.from_bytes(h, "little", signed=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, required=True)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--max_windows", type=int, default=0, help="0 = no cap")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    W = int(args.window)
    S = max(1, int(args.stride))
    cap = int(args.max_windows)

    shard_paths = list_bins(args.dir)
    print(f"[*] Scanning: {args.dir}")
    print(f"    - shards: {len(shard_paths)}")
    print(f"    - window: {W} tokens")
    print(f"    - stride: {S} tokens")
    print(f"    - max_windows: {cap:,}" if cap > 0 else "    - max_windows: (no cap)")

    counts = Counter()
    total_raw_tokens = 0
    total_windows_possible = 0
    processed = 0

    for sp in shard_paths:
        mm = np.memmap(sp, dtype=np.uint16, mode="r")
        n = int(mm.shape[0])
        total_raw_tokens += n

        if n < W:
            continue
        total_windows_possible += 1 + (n - W) // S

        for start in range(0, n - W + 1, S):
            win = mm[start : start + W]
            hv = hash_u16(np.asarray(win, dtype=np.uint16))
            counts[hv] += 1
            processed += 1
            if cap > 0 and processed >= cap:
                break
        if cap > 0 and processed >= cap:
            break

    unique = len(counts)
    dup_windows = sum(c - 1 for c in counts.values() if c > 1)
    total_hashed = sum(counts.values())
    dup_rate = (dup_windows / total_hashed) if total_hashed > 0 else 0.0

    print("\n[*] Raw stats:")
    print(f"    - raw tokens total: {total_raw_tokens:,}")
    print(f"    - windows possible (approx): {total_windows_possible:,}")
    print(f"    - hashed windows: {total_hashed:,}")
    print(f"    - unique window hashes: {unique:,}")
    print(f"    - duplicate windows (extra copies): {dup_windows:,}")
    print(f"    - duplicate rate (exact, by window): {dup_rate:.6%}")

    if args.topk > 0 and unique > 0:
        print(f"\n[*] Top-{args.topk} most frequent windows (hash,count):")
        for hv, c in counts.most_common(args.topk):
            print(f"    {hv:016x}  {c}")

    print("\n[*] Notes:")
    print("    - This is still exact hashing, but at smaller window granularity.")
    print("    - If this rate is noticeably higher than block-level, you have shifted/boilerplate repetition.")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
