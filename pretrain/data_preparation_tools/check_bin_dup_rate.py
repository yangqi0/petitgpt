#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check duplicate rate of packed uint16 .bin shards (block = seq_len + 1).

- Each .bin is a continuous uint16 token stream.
- We consider "blocks" of fixed length (seq_len+1).
- Compute hash per block and report exact-duplicate rate.

Example:
  python pretrain/check_bin_dup_rate.py \
    --dir datasets/pretrain_mix/train \
    --seq_len 1024 \
    --stride 1 \
    --max_blocks 200000 \
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


def hash_block_u16(arr_u16: np.ndarray) -> int:
    """
    arr_u16: shape [block], dtype=uint16
    Return 64-bit int hash.
    """
    # Use stable 64-bit digest (8 bytes) to keep memory small.
    h = hashlib.blake2b(arr_u16.tobytes(order="C"), digest_size=8).digest()
    return int.from_bytes(h, "little", signed=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, required=True, help="directory containing *.bin")
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=1, help="hash every N blocks (1 = all)")
    ap.add_argument(
        "--max_blocks",
        type=int,
        default=0,
        help="cap total processed blocks across all shards (0 = no cap)",
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=20,
        help="show top-k most frequent hashes",
    )
    ap.add_argument(
        "--dtype",
        type=str,
        default="uint16",
        choices=["uint16"],
        help="token dtype on disk",
    )
    args = ap.parse_args()

    shard_paths = list_bins(args.dir)
    block = args.seq_len + 1
    stride = max(1, int(args.stride))
    max_blocks = int(args.max_blocks)

    print(f"[*] Scanning: {args.dir}")
    print(f"    - shards: {len(shard_paths)}")
    print(f"    - block size: {block} tokens (seq_len+1)")
    print(f"    - stride: {stride}")
    if max_blocks > 0:
        print(f"    - max_blocks: {max_blocks:,}")
    else:
        print(f"    - max_blocks: (no cap)")

    counts = Counter()
    processed = 0
    total_full_blocks = 0
    total_raw_tokens = 0

    for sp in shard_paths:
        mm = np.memmap(sp, dtype=np.uint16, mode="r")
        n_tokens = int(mm.shape[0])
        total_raw_tokens += n_tokens

        n_blocks = n_tokens // block
        total_full_blocks += n_blocks

        # iterate blocks
        for bi in range(0, n_blocks, stride):
            start = bi * block
            blk = mm[start : start + block]
            # blk is a memmap view; hashing needs bytes -> ok
            hv = hash_block_u16(np.asarray(blk, dtype=np.uint16))
            counts[hv] += 1

            processed += 1
            if max_blocks > 0 and processed >= max_blocks:
                break

        if max_blocks > 0 and processed >= max_blocks:
            break

    unique = len(counts)
    dup_blocks = sum(c - 1 for c in counts.values() if c > 1)
    total_hashed = sum(counts.values())
    dup_rate = (dup_blocks / total_hashed) if total_hashed > 0 else 0.0

    print("\n[*] Raw stats:")
    print(f"    - raw tokens total: {total_raw_tokens:,}")
    print(f"    - full blocks total (all shards): {total_full_blocks:,}")
    if stride == 1 and (max_blocks == 0):
        print(f"    - hashed blocks: {total_hashed:,} (ALL full blocks)")
    else:
        print(f"    - hashed blocks: {total_hashed:,} (sampled by stride/cap)")
    print(f"    - unique block hashes: {unique:,}")
    print(f"    - duplicate blocks (extra copies): {dup_blocks:,}")
    print(f"    - duplicate rate (exact, by block): {dup_rate:.6%}")

    # Show top-k frequent hashes
    if args.topk > 0 and unique > 0:
        print(f"\n[*] Top-{args.topk} most frequent blocks (hash,count):")
        for hv, c in counts.most_common(args.topk):
            print(f"    {hv:016x}  {c}")

    # A quick interpretation hint
    print("\n[*] Notes:")
    print("    - This measures exact duplicates at block granularity (seq_len+1 tokens).")
    print("    - If you used random start sampling in dataset, runtime duplicates seen by training may differ.")
    print("    - For finer-grain duplication (e.g., 256-token windows), need a sliding-window variant.")


if __name__ == "__main__":
    # Avoid MKL/OMP oversubscription surprises on some boxes
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
