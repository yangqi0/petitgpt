#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_tokenizer_mix.py

Create a mixed JSONL file ({"text": ...} per line) for tokenizer training by
sampling from multiple source JSONL files with user-specified proportions.

Why this exists
---------------
Passing multiple files to tokenizer.train_from_iterator(...) does NOT enforce
ratios. It simply streams examples in file order. This script lets you:

1) Decide a total sample size (e.g., 1,000,000 lines)
2) Allocate per-source counts using ratios (e.g., 57/10/10/18/5)
3) Uniformly sample that many lines from each source (without replacement)
4) Merge and (optionally) shuffle the final mixture

Sampling method
---------------
We use reservoir sampling (single pass, uniform without replacement) per file.
This is "correct" sampling but requires storing the sampled lines in memory
for each source. With 46GB RAM, ~1M lines is typically fine for tokenizer corpora,
but you can disable shuffling or reduce total size if memory is tight.

Input assumptions
-----------------
- Each input file is JSONL with at least: {"text": "..."} per line.
- Lines may be large; we do not parse JSON to keep it fast.
  (We only validate a small prefix if --validate_json is set.)

Typical usage
-------------
# 1M total lines with ratios (57/10/10/18/5)
python make_tokenizer_mix.py \
  --total 1000000 \
  --out datasets/tokenization/tokenizer_mix_1m.jsonl \
  --spec datasets/tokenization/fineweb_edu.tok.v1.jsonl:0.57 \
  --spec datasets/tokenization/wiki.clean.v3.jsonl:0.10 \
  --spec datasets/tokenization/cosmo_math.jsonl:0.10 \
  --spec datasets/tokenization/code_starcoder_py200k.jsonl:0.18 \
  --spec datasets/tokenization/tinystories_train.jsonl:0.05 \
  --seed 1234 \
  --shuffle

# If you prefer explicit counts instead of ratios:
python make_tokenizer_mix.py \
  --out datasets/tokenization/tokenizer_mix.jsonl \
  --spec datasets/tokenization/fineweb_edu.tok.v1.jsonl:570000 \
  --spec datasets/tokenization/wiki.clean.v3.jsonl:100000 \
  --spec datasets/tokenization/cosmo_math.jsonl:100000 \
  --spec datasets/tokenization/code_starcoder_py200k.jsonl:180000 \
  --spec datasets/tokenization/tinystories_train.jsonl:50000 \
  --seed 1234 --shuffle
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class Spec:
    path: str
    value: float  # ratio (0<value<=1) or count (>=1)


def parse_spec(s: str) -> Spec:
    """
    Parse --spec PATH:VALUE where VALUE is either:
      - ratio in (0,1]  (e.g., 0.57)
      - integer count   (e.g., 570000)
    """
    if ":" not in s:
        raise ValueError(f"Invalid --spec '{s}'. Expected format PATH:VALUE")
    path, val = s.rsplit(":", 1)
    path = path.strip()
    val = val.strip()
    if not path:
        raise ValueError(f"Invalid --spec '{s}': empty path")
    try:
        fval = float(val)
    except ValueError as e:
        raise ValueError(f"Invalid --spec '{s}': VALUE must be a number") from e
    return Spec(path=path, value=fval)


def is_ratio(v: float) -> bool:
    return 0.0 < v <= 1.0 and not float(v).is_integer()


def is_count(v: float) -> bool:
    return v >= 1.0 and float(v).is_integer()


def resolve_counts(specs: List[Spec], total: int | None) -> List[Tuple[str, int]]:
    """
    Convert specs into concrete per-file sample counts.
    If all VALUE look like ratios, --total must be provided.
    If all VALUE look like counts, --total is ignored.
    Mixed ratios+counts is disallowed to avoid ambiguity.
    """
    kinds = []
    for sp in specs:
        if is_ratio(sp.value):
            kinds.append("ratio")
        elif is_count(sp.value):
            kinds.append("count")
        else:
            # This catches ratios like 1.0 (ambiguous: could be 100% ratio or count=1)
            # We disambiguate by treating 1.0 as ratio only if --total is provided and all are <=1.0.
            kinds.append("ambiguous")

    if any(k == "ambiguous" for k in kinds):
        # Allow 1.0 as ratio if ALL values are <=1.0 and --total is provided
        if total is not None and all(sp.value <= 1.0 for sp in specs):
            kinds = ["ratio"] * len(specs)
        else:
            raise ValueError(
                "Ambiguous VALUE found (e.g., 1.0). Use explicit integer counts (e.g., 100000) "
                "or provide --total and ratios strictly less than 1.0 for all but possibly one."
            )

    if len(set(kinds)) != 1:
        raise ValueError("Do not mix ratios and counts in --spec. Use all ratios or all counts.")

    if kinds[0] == "count":
        return [(sp.path, int(sp.value)) for sp in specs]

    # ratio mode
    if total is None:
        raise ValueError("Ratio specs require --total.")
    ratios = [sp.value for sp in specs]
    s = sum(ratios)
    if not (0.999 <= s <= 1.001):
        raise ValueError(f"Ratios must sum to ~1.0. Got sum={s:.6f}")

    # Largest remainder method to make counts sum exactly to total
    raw = [r * total for r in ratios]
    floors = [int(math.floor(x)) for x in raw]
    remainder = total - sum(floors)
    fracs = sorted([(raw[i] - floors[i], i) for i in range(len(specs))], reverse=True)
    counts = floors[:]
    for k in range(remainder):
        counts[fracs[k][1]] += 1

    return [(specs[i].path, counts[i]) for i in range(len(specs))]


def reservoir_sample_jsonl(path: str, k: int, rng: random.Random, validate_json: bool) -> List[str]:
    """
    Reservoir sample k lines uniformly from a JSONL file.

    Returns a list of raw JSONL lines (including trailing '\n').

    Notes:
    - Single pass over the file, no need to know file length.
    - Uniform without replacement over lines.
    - If the file has fewer than k valid lines, returns all lines and warns.
    """
    if k <= 0:
        return []

    reservoir: List[str] = []
    seen = 0
    kept = 0
    bad = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line:
                continue

            # Optional lightweight JSON validation (can be slow on very large corpora)
            if validate_json:
                try:
                    obj = json.loads(line)
                    if not isinstance(obj, dict) or "text" not in obj or not isinstance(obj["text"], str):
                        bad += 1
                        continue
                except json.JSONDecodeError:
                    bad += 1
                    continue

            seen += 1

            if kept < k:
                reservoir.append(line if line.endswith("\n") else (line + "\n"))
                kept += 1
                continue

            # Replace elements with decreasing probability
            j = rng.randint(0, seen - 1)  # inclusive
            if j < k:
                reservoir[j] = line if line.endswith("\n") else (line + "\n")

    if kept < k:
        print(
            f"[WARN] {path}: requested k={k} but only got {kept} valid lines "
            f"(seen={seen}, bad={bad}). Using all {kept}.",
            file=sys.stderr,
        )
    else:
        print(f"[OK] {path}: sampled k={k} lines (seen={seen}, bad={bad})")

    return reservoir


def write_jsonl(lines: List[str], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--spec",
        action="append",
        required=True,
        help="Repeated. Format PATH:VALUE where VALUE is either ratio (e.g., 0.57) or count (e.g., 570000).",
    )
    ap.add_argument("--total", type=int, default=None, help="Total lines (required if VALUE are ratios).")
    ap.add_argument("--out", type=str, required=True, help="Output mixed JSONL path.")
    ap.add_argument("--seed", type=int, default=1234, help="Random seed.")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle the merged mixture in-memory.")
    ap.add_argument(
        "--validate_json",
        action="store_true",
        help="Validate each sampled line is JSON with a string 'text' field (slower).",
    )
    ap.add_argument(
        "--max_in_memory_lines",
        type=int,
        default=2_000_000,
        help="Safety limit for --shuffle. If total lines exceeds this, we refuse to shuffle.",
    )
    args = ap.parse_args()

    specs = [parse_spec(s) for s in args.spec]
    resolved = resolve_counts(specs, total=args.total)

    # Print plan
    planned_total = sum(k for _, k in resolved)
    print("Sampling plan:")
    for p, k in resolved:
        print(f"  - {p}: {k}")
    print(f"Planned total: {planned_total}")

    rng = random.Random(args.seed)

    # Sample each source
    all_lines: List[str] = []
    for path, k in resolved:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing input file: {path}")
        lines = reservoir_sample_jsonl(path, k=k, rng=rng, validate_json=args.validate_json)
        all_lines.extend(lines)

    print(f"Collected total lines: {len(all_lines)}")

    # Shuffle if requested
    if args.shuffle:
        if len(all_lines) > args.max_in_memory_lines:
            raise RuntimeError(
                f"Refusing to shuffle {len(all_lines)} lines in-memory (limit={args.max_in_memory_lines}). "
                "Either reduce --total or increase --max_in_memory_lines or omit --shuffle."
            )
        rng.shuffle(all_lines)
        print("[OK] Shuffled mixture in-memory")

    write_jsonl(all_lines, args.out)
    print(f"[OK] Wrote: {args.out}")


if __name__ == "__main__":
    main()
