#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arithmetic anchor-shuffle DRILL (v9):

Fixes a failure mode of v8:
- Repeating anchors in the same order can let the model learn a *sequence* (5,42,72,5,42,72...)
  without conditioning strongly on the question.
v9 shuffles anchor order every repetition, forcing Q->A association.

You can generate:
- Pure anchor packs (pack_k = 3 * repeats_anchors)  -> memorize bench quickly
- Anchor+practice packs (larger pack_k)             -> reinforce times table + carry-add

Output JSONL: {"messages":[...], "meta":{...}}
"""

from __future__ import annotations
import argparse, json, os, random
from typing import Dict, List, Tuple

SYS_ARITH = (
    "You are a precise assistant.\n"
    "Rules for arithmetic questions:\n"
    "- Output ONLY the final numeric answer.\n"
    "- No explanation, no extra words, no punctuation.\n"
)

ANCHORS: List[Tuple[str, str, str]] = [
    ("John has 3 apples and buys 2 more. How many apples does he have?\nAnswer:", "5", "anchor_apples"),
    ("Compute 17 + 25.\nAnswer:", "42", "anchor_17_25"),
    ("What is 9 * 8?\nAnswer:", "72", "anchor_9_8"),
]

def _add_carry_0_99(rng: random.Random) -> Tuple[str, str, str]:
    for _ in range(20000):
        a = rng.randint(0, 99)
        b = rng.randint(0, 99)
        if (a % 10) + (b % 10) >= 10:
            return f"Compute {a} + {b}.\nAnswer:", str(a + b), "add_carry"
    a = rng.randint(0, 99)
    b = rng.randint(0, 99)
    return f"Compute {a} + {b}.\nAnswer:", str(a + b), "add"

def _mul_table(rng: random.Random, max_k: int = 12) -> Tuple[str, str, str]:
    a = rng.randint(0, max_k)
    b = rng.randint(0, max_k)
    return f"What is {a} * {b}?\nAnswer:", str(a * b), "mul_table"

def _word_apples(rng: random.Random, hi: int = 40) -> Tuple[str, str, str]:
    a = rng.randint(0, hi)
    b = rng.randint(0, hi)
    return f"John has {a} apples and buys {b} more. How many apples does he have?\nAnswer:", str(a + b), "word"

def sample_practice(rng: random.Random) -> Tuple[str, str, str]:
    r = rng.random()
    if r < 0.65:
        return _mul_table(rng, 12)
    if r < 0.95:
        return _add_carry_0_99(rng)
    return _word_apples(rng, 40)

def make_pack(rng: random.Random, pack_k: int, repeats_anchors: int, pack_id: str) -> Dict:
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYS_ARITH}]
    styles: List[str] = []
    qas: List[Tuple[str, str, str]] = []

    for r in range(repeats_anchors):
        tmp = ANCHORS[:]  # copy
        rng.shuffle(tmp)  # shuffle anchor order each repetition
        qas.extend(tmp)

    while len(qas) < pack_k:
        qas.append(sample_practice(rng))

    qas = qas[:pack_k]

    for q, a, st in qas:
        styles.append(st)
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a + "\n"})

    return {"messages": messages, "meta": {"id": pack_id, "task": "arith_anchor_shuffle_pack", "styles": styles}}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n_packs", type=int, default=12000)
    ap.add_argument("--pack_k", type=int, default=12, help="Total QAs per pack (>= 3*repeats_anchors)")
    ap.add_argument("--repeats_anchors", type=int, default=4)
    args = ap.parse_args()

    if args.pack_k < 3 * args.repeats_anchors:
        raise ValueError("pack_k must be >= 3*repeats_anchors (anchors only)")

    rng = random.Random(args.seed)
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    print(f"[start] packs={args.n_packs} pack_k={args.pack_k} repeats_anchors={args.repeats_anchors} -> {args.out}", flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(args.n_packs):
            row = make_pack(rng, args.pack_k, args.repeats_anchors, f"arith_shuf_{i:06d}")
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (i + 1) % 1000 == 0:
                print(f"[progress] {i+1}/{args.n_packs}", flush=True)
    print("[done]", flush=True)

if __name__ == "__main__":
    main()
