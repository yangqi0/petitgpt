#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic SFT generator (v4): curriculum + carry/borrow arithmetic + multitask mixing.

Purpose:
- Arith-only SFT can catastrophically forget code/syllogism.
- v4 makes it easy to generate a mixed dataset and strengthens carry addition like 17+25.

Output JSONL:
{"messages":[...], "meta":{...}}
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, Tuple, List

EOC_LINE = "###EOC###"
EOC_BLOCK = f"\n{EOC_LINE}\n"

SYS_ARITH = (
    "You are a precise assistant.\n"
    "Rules for arithmetic questions:\n"
    "- Output ONLY the final numeric answer.\n"
    "- No explanation, no extra words, no punctuation.\n"
)

SYS_SYLL = (
    "You are a logic assistant.\n"
    "Rules:\n"
    "- Output ONLY one token: yes / no / unknown.\n"
    "- Do not explain.\n"
)

SYS_CODE = (
    "You are a coding assistant.\n"
    "Rules:\n"
    "- Output ONLY Python code.\n"
    "- Do NOT write any explanation.\n"
    "- Do NOT write extra functions.\n"
    "- Do NOT write a main block.\n"
    "- Implement exactly the requested function.\n"
    "- Output ONLY the function body (no def line).\n"
    "- Do NOT use 'yield'.\n"
    "- Do NOT use semicolons ';'.\n"
    "- Write exactly ONE statement per line (no two statements on one line).\n"
    f"- End your output with exactly: {EOC_LINE}\n"
)

def ex(system: str, user: str, assistant: str, meta: Dict) -> Dict:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "meta": meta,
    }

# -------------------------
# arithmetic (curriculum + carry/borrow)
# -------------------------
def _arith_add(rng: random.Random, lo: int, hi: int) -> Tuple[str, str]:
    a = rng.randint(lo, hi)
    b = rng.randint(lo, hi)
    q = f"Compute {a} + {b}.\nAnswer:"
    return q, str(a + b)

def _arith_add_carry_0_99(rng: random.Random) -> Tuple[str, str]:
    # Force carry in ones digit: (a%10 + b%10) >= 10
    for _ in range(1000):
        a = rng.randint(0, 99)
        b = rng.randint(0, 99)
        if (a % 10) + (b % 10) >= 10:
            q = f"Compute {a} + {b}.\nAnswer:"
            return q, str(a + b)
    return _arith_add(rng, 0, 99)

def _arith_sub_nonneg(rng: random.Random, lo: int, hi: int) -> Tuple[str, str]:
    a = rng.randint(lo, hi)
    b = rng.randint(lo, hi)
    if b > a:
        a, b = b, a
    q = f"Compute {a} - {b}.\nAnswer:"
    return q, str(a - b)

def _arith_sub_borrow_0_99(rng: random.Random) -> Tuple[str, str]:
    # Force borrow in ones digit: a>=b and (a%10) < (b%10)
    for _ in range(1000):
        a = rng.randint(0, 99)
        b = rng.randint(0, 99)
        if a >= b and (a % 10) < (b % 10):
            q = f"Compute {a} - {b}.\nAnswer:"
            return q, str(a - b)
    return _arith_sub_nonneg(rng, 0, 99)

def _arith_mul_table(rng: random.Random, max_k: int = 12) -> Tuple[str, str]:
    a = rng.randint(0, max_k)
    b = rng.randint(0, max_k)
    q = f"What is {a} * {b}?\nAnswer:"
    return q, str(a * b)

def _arith_word_apples(rng: random.Random, hi: int = 40) -> Tuple[str, str]:
    a = rng.randint(0, hi)
    b = rng.randint(0, hi)
    q = f"John has {a} apples and buys {b} more. How many apples does he have?\nAnswer:"
    return q, str(a + b)

def gen_arith(rng: random.Random, i: int) -> Dict:
    styles = [
        "add_0_20",
        "sub_0_20",
        "mul_table",
        "word",
        "add_0_100",
        "sub_0_100",
        "add_carry_0_99",
        "sub_borrow_0_99",
    ]
    weights = [0.20, 0.14, 0.18, 0.08, 0.12, 0.10, 0.10, 0.08]
    style = rng.choices(styles, weights=weights, k=1)[0]

    if style == "add_0_20":
        q, ans = _arith_add(rng, 0, 20)
    elif style == "sub_0_20":
        q, ans = _arith_sub_nonneg(rng, 0, 20)
    elif style == "mul_table":
        q, ans = _arith_mul_table(rng, 12)
    elif style == "word":
        q, ans = _arith_word_apples(rng, 40)
    elif style == "add_0_100":
        q, ans = _arith_add(rng, 0, 100)
    elif style == "sub_0_100":
        q, ans = _arith_sub_nonneg(rng, 0, 100)
    elif style == "add_carry_0_99":
        q, ans = _arith_add_carry_0_99(rng)
    else:
        q, ans = _arith_sub_borrow_0_99(rng)

    return ex(
        SYS_ARITH,
        q,
        ans + "\n",
        {"id": f"syn_arith_{i:06d}", "task": "arithmetic", "style": style},
    )

def bench_anchor_examples() -> List[Dict]:
    out: List[Dict] = []
    out.append(ex(SYS_ARITH,
                  "John has 3 apples and buys 2 more. How many apples does he have?\nAnswer:",
                  "5\n",
                  {"id": "anchor_arith_apples_0001", "task": "arithmetic", "style": "anchor"}))
    out.append(ex(SYS_ARITH,
                  "Compute 17 + 25.\nAnswer:",
                  "42\n",
                  {"id": "anchor_arith_17_25", "task": "arithmetic", "style": "anchor"}))
    out.append(ex(SYS_ARITH,
                  "What is 9 * 8?\nAnswer:",
                  "72\n",
                  {"id": "anchor_arith_9_8", "task": "arithmetic", "style": "anchor"}))
    return out

# -------------------------
# syllogism
# -------------------------
def gen_syll(rng: random.Random, i: int) -> Dict:
    styles = ["cats_unknown", "all_yes", "some_yes", "all_unknown", "some_no"]
    weights = [0.25, 0.25, 0.25, 0.15, 0.10]
    style = rng.choices(styles, weights=weights, k=1)[0]

    if style == "cats_unknown":
        q = "If all cats are animals and some animals are black, can we conclude that some cats are black?\nAnswer:"
        ans = "unknown"
    elif style == "all_yes":
        q = "All A are B. All B are C. Can we conclude all A are C?\nAnswer:"
        ans = "yes"
    elif style == "some_yes":
        q = "Some A are B. All B are C. Can we conclude some A are C?\nAnswer:"
        ans = "yes"
    elif style == "all_unknown":
        q = "All A are B. Some B are C. Can we conclude some A are C?\nAnswer:"
        ans = "unknown"
    else:
        q = "Some A are B. No B are C. Can we conclude some A are C?\nAnswer:"
        ans = "no"

    return ex(
        SYS_SYLL,
        q,
        ans + "\n",
        {"id": f"syn_syll_{i:06d}", "task": "syllogism", "style": style},
    )

# -------------------------
# code (clean, short, always EOC)
# -------------------------
def gen_code(rng: random.Random, i: int) -> Dict:
    kinds = ["add", "factorial", "fib", "is_even", "clamp"]
    weights = [0.20, 0.25, 0.25, 0.15, 0.15]
    kind = rng.choices(kinds, weights=weights, k=1)[0]

    if kind == "add":
        user = "Complete the following Python function:\n\ndef add(a, b):\n    "
        body = "return a + b\n"
    elif kind == "factorial":
        user = "Complete the following Python function:\n\ndef factorial(n):\n    "
        body = (
            "if n < 0:\n"
            "    raise ValueError('n must be non-negative')\n"
            "r = 1\n"
            "for i in range(2, n + 1):\n"
            "    r *= i\n"
            "return r\n"
        )
    elif kind == "fib":
        user = "Complete the following Python function:\n\ndef fib(n):\n    "
        body = (
            "if n < 0:\n"
            "    raise ValueError('n must be non-negative')\n"
            "a, b = 0, 1\n"
            "for _ in range(n):\n"
            "    a, b = b, a + b\n"
            "return a\n"
        )
    elif kind == "is_even":
        user = "Complete the following Python function:\n\ndef is_even(n):\n    "
        body = "return (n % 2) == 0\n"
    else:
        user = "Complete the following Python function:\n\ndef clamp(x, lo, hi):\n    "
        body = (
            "if x < lo:\n"
            "    return lo\n"
            "if x > hi:\n"
            "    return hi\n"
            "return x\n"
        )

    assistant = body.rstrip("\n") + EOC_BLOCK
    return ex(
        SYS_CODE,
        user,
        assistant,
        {"id": f"syn_code_{i:06d}", "task": "code", "kind": kind},
    )

# -------------------------
# main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)

    # Multitask defaults to prevent forgetting.
    ap.add_argument("--n_arith", type=int, default=60000)
    ap.add_argument("--n_syll", type=int, default=20000)
    ap.add_argument("--n_code", type=int, default=20000)

    ap.add_argument("--include_anchors", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    total = args.n_arith + args.n_syll + args.n_code
    print(f"[start] writing {total} examples to {args.out}", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        written = 0
        if args.include_anchors:
            for row in bench_anchor_examples():
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

        rem = {"arithmetic": args.n_arith, "syllogism": args.n_syll, "code": args.n_code}
        idx = {"arithmetic": 0, "syllogism": 0, "code": 0}

        while rem["arithmetic"] + rem["syllogism"] + rem["code"] > 0:
            choices = [t for t, n in rem.items() if n > 0]
            weights = [rem[t] for t in choices]
            task = rng.choices(choices, weights=weights, k=1)[0]

            if task == "arithmetic":
                row = gen_arith(rng, idx["arithmetic"])
            elif task == "syllogism":
                row = gen_syll(rng, idx["syllogism"])
            else:
                row = gen_code(rng, idx["code"])

            idx[task] += 1
            rem[task] -= 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

            if written % 5000 == 0:
                print(f"[progress] {written}/{total} rem={rem}", flush=True)

    print(f"[done] wrote {written} examples to {args.out}", flush=True)

if __name__ == "__main__":
    main()
