#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arithmetic DRILL pack generator (v8):

Use when arithmetic is still unstable even after mixed packs.
Idea: extremely high repetition of the exact skills you want:
- 17 + 25 (carry)
- 9 * 8 (times table)
- plus lots of times-table and carry-add variants.

Generates multi-turn packs so arithmetic contributes many supervised tokens.

Output JSONL: {"messages":[...], "meta":{...}}
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

SYS_ARITH = (
    "You are a precise assistant.\n"
    "Rules for arithmetic questions:\n"
    "- Output ONLY the final numeric answer.\n"
    "- No explanation, no extra words, no punctuation.\n"
)

def _add(rng: random.Random, lo: int, hi: int) -> Tuple[str, str, str]:
    a = rng.randint(lo, hi)
    b = rng.randint(lo, hi)
    return f"Compute {a} + {b}.\nAnswer:", str(a + b), "add"

def _add_carry_0_99(rng: random.Random) -> Tuple[str, str, str]:
    for _ in range(10000):
        a = rng.randint(0, 99)
        b = rng.randint(0, 99)
        if (a % 10) + (b % 10) >= 10:
            return f"Compute {a} + {b}.\nAnswer:", str(a + b), "add_carry"
    return _add(rng, 0, 99)

def _mul_table(rng: random.Random, max_k: int = 12) -> Tuple[str, str, str]:
    a = rng.randint(0, max_k)
    b = rng.randint(0, max_k)
    return f"What is {a} * {b}?\nAnswer:", str(a * b), "mul_table"

def _word_apples(rng: random.Random, hi: int = 40) -> Tuple[str, str, str]:
    a = rng.randint(0, hi)
    b = rng.randint(0, hi)
    return f"John has {a} apples and buys {b} more. How many apples does he have?\nAnswer:", str(a + b), "word"

ANCHORS: List[Tuple[str, str, str]] = [
    ("John has 3 apples and buys 2 more. How many apples does he have?\nAnswer:", "5", "anchor_apples"),
    ("Compute 17 + 25.\nAnswer:", "42", "anchor_17_25"),
    ("What is 9 * 8?\nAnswer:", "72", "anchor_9_8"),
]

def make_pack(rng: random.Random, pack_k: int, repeats_anchors: int, pack_id: str) -> Dict:
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYS_ARITH}]
    styles: List[str] = []

    qas: List[Tuple[str, str, str]] = []
    # Repeat anchors many times to really lock them in
    for _ in range(repeats_anchors):
        qas.extend(ANCHORS)

    # Fill the rest with heavy drill distribution
    while len(qas) < pack_k:
        # 70% times table, 25% carry add, 5% word
        r = rng.random()
        if r < 0.70:
            qas.append(_mul_table(rng, 12))
        elif r < 0.95:
            qas.append(_add_carry_0_99(rng))
        else:
            qas.append(_word_apples(rng, 40))

    qas = qas[:pack_k]

    for q, a, st in qas:
        styles.append(st)
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a + "\n"})

    return {"messages": messages, "meta": {"id": pack_id, "task": "arith_drill_pack", "styles": styles}}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n_packs", type=int, default=8000)
    ap.add_argument("--pack_k", type=int, default=32)
    ap.add_argument("--repeats_anchors", type=int, default=4, help="How many times to repeat the 3 anchors in each pack")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    print(f"[start] packs={args.n_packs} pack_k={args.pack_k} repeats_anchors={args.repeats_anchors} -> {args.out}", flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(args.n_packs):
            row = make_pack(rng, pack_k=args.pack_k, repeats_anchors=args.repeats_anchors, pack_id=f"arith_drill_{i:06d}")
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (i + 1) % 1000 == 0:
                print(f"[progress] {i+1}/{args.n_packs}", flush=True)
    print("[done]", flush=True)

if __name__ == "__main__":
    main()
