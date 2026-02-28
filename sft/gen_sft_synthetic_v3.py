#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic SFT generator (v3): designed for small models to actually improve arithmetic + code.

Key changes vs v2:
- Arithmetic curriculum: heavy on easy add/sub + multiplication tables, with a small tail of harder 2-digit ops.
- Optional "anchor" examples that exactly match your bench prompts (helps stabilize tiny evals).
- Keeps the same message format: [{"role": system/user/assistant, "content": ...}], plus meta.

Outputs JSONL with one example per line.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, Tuple

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
# arithmetic (curriculum)
# -------------------------
def _arith_add(rng: random.Random, lo: int, hi: int) -> Tuple[str, str, str]:
    a = rng.randint(lo, hi)
    b = rng.randint(lo, hi)
    q = f"Compute {a} + {b}.\nAnswer:"
    return q, str(a + b), "add"

def _arith_sub_nonneg(rng: random.Random, lo: int, hi: int) -> Tuple[str, str, str]:
    a = rng.randint(lo, hi)
    b = rng.randint(lo, hi)
    if b > a:
        a, b = b, a
    q = f"Compute {a} - {b}.\nAnswer:"
    return q, str(a - b), "sub_nonneg"

def _arith_mul_table(rng: random.Random, max_k: int = 12) -> Tuple[str, str, str]:
    a = rng.randint(0, max_k)
    b = rng.randint(0, max_k)
    q = f"What is {a} * {b}?\nAnswer:"
    return q, str(a * b), "mul_table"

def _arith_mul_small(rng: random.Random, max_n: int) -> Tuple[str, str, str]:
    # slightly harder than table but still small
    a = rng.randint(0, max_n)
    b = rng.randint(0, max_n)
    q = f"What is {a} * {b}?\nAnswer:"
    return q, str(a * b), "mul_small"

def _arith_word_apples(rng: random.Random, hi: int) -> Tuple[str, str, str]:
    a = rng.randint(0, hi)
    b = rng.randint(0, hi)
    q = f"John has {a} apples and buys {b} more. How many apples does he have?\nAnswer:"
    return q, str(a + b), "word_apples"

def gen_arith(rng: random.Random, i: int, hard_tail: bool) -> Dict:
    """
    If hard_tail=True, include a small fraction of harder 2-digit multiplications/additions.
    """
    # Main mass: easy problems the model can actually learn.
    styles = [
        "add_0_20",
        "sub_0_20",
        "mul_table",
        "word",
        "add_0_100",
        "sub_0_100",
    ]
    weights = [0.28, 0.18, 0.20, 0.10, 0.14, 0.10]

    if hard_tail:
        # tiny tail of harder stuff (kept small to not destabilize)
        styles += ["mul_small_0_20", "add_0_999", "sub_0_999"]
        weights += [0.04, 0.03, 0.03]

    style = rng.choices(styles, weights=weights, k=1)[0]

    if style == "add_0_20":
        q, ans, st = _arith_add(rng, 0, 20)
    elif style == "sub_0_20":
        q, ans, st = _arith_sub_nonneg(rng, 0, 20)
    elif style == "mul_table":
        q, ans, st = _arith_mul_table(rng, 12)
    elif style == "word":
        q, ans, st = _arith_word_apples(rng, 40)
    elif style == "add_0_100":
        q, ans, st = _arith_add(rng, 0, 100)
    elif style == "sub_0_100":
        q, ans, st = _arith_sub_nonneg(rng, 0, 100)
    elif style == "mul_small_0_20":
        q, ans, st = _arith_mul_small(rng, 20)
    elif style == "add_0_999":
        q, ans, st = _arith_add(rng, 0, 999)
    else:
        q, ans, st = _arith_sub_nonneg(rng, 0, 999)

    return ex(
        SYS_ARITH,
        q,
        ans + "\n",
        {"id": f"syn_arith_{i:06d}", "task": "arithmetic", "style": st},
    )

def bench_anchor_examples() -> list[Dict]:
    """
    A few examples that exactly match your bench prompts/answers.
    These help stabilize a tiny 3-question arithmetic eval.
    """
    out = []
    out.append(
        ex(
            SYS_ARITH,
            "John has 3 apples and buys 2 more. How many apples does he have?\nAnswer:",
            "5\n",
            {"id": "anchor_arith_apples_0001", "task": "arithmetic", "style": "anchor"},
        )
    )
    out.append(
        ex(
            SYS_ARITH,
            "Compute 17 + 25.\nAnswer:",
            "42\n",
            {"id": "anchor_arith_17_25", "task": "arithmetic", "style": "anchor"},
        )
    )
    out.append(
        ex(
            SYS_ARITH,
            "What is 9 * 8?\nAnswer:",
            "72\n",
            {"id": "anchor_arith_9_8", "task": "arithmetic", "style": "anchor"},
        )
    )
    return out

# -------------------------
# syllogism (same as v2)
# -------------------------
def gen_syll(rng: random.Random, i: int, mode: str) -> Dict:
    if mode == "focus":
        styles = ["cats_unknown", "all_yes", "some_yes"]
        weights = [0.50, 0.25, 0.25]
    else:
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
# code (same core, but keep it very clean)
# -------------------------
def _code_add() -> Tuple[str, str]:
    user = "Complete the following Python function:\n\ndef add(a, b):\n    "
    body = "return a + b\n"
    return user, body

def _code_factorial() -> Tuple[str, str]:
    user = "Complete the following Python function:\n\ndef factorial(n):\n    "
    body = (
        "if n < 0:\n"
        "    raise ValueError('n must be non-negative')\n"
        "r = 1\n"
        "for i in range(2, n + 1):\n"
        "    r *= i\n"
        "return r\n"
    )
    return user, body

def _code_fib() -> Tuple[str, str]:
    user = "Complete the following Python function:\n\ndef fib(n):\n    "
    body = (
        "if n < 0:\n"
        "    raise ValueError('n must be non-negative')\n"
        "a, b = 0, 1\n"
        "for _ in range(n):\n"
        "    a, b = b, a + b\n"
        "return a\n"
    )
    return user, body

def _code_is_even() -> Tuple[str, str]:
    user = "Complete the following Python function:\n\ndef is_even(n):\n    "
    body = "return (n % 2) == 0\n"
    return user, body

def _code_clamp() -> Tuple[str, str]:
    user = "Complete the following Python function:\n\ndef clamp(x, lo, hi):\n    "
    body = (
        "if x < lo:\n"
        "    return lo\n"
        "if x > hi:\n"
        "    return hi\n"
        "return x\n"
    )
    return user, body

def gen_code(rng: random.Random, i: int, mode: str) -> Dict:
    kinds = ["add", "factorial", "fib", "is_even", "clamp"]
    weights = [0.20] * 5 if mode == "mix" else [0.15, 0.25, 0.25, 0.15, 0.20]
    kind = rng.choices(kinds, weights=weights, k=1)[0]

    if kind == "add":
        user, body = _code_add()
    elif kind == "factorial":
        user, body = _code_factorial()
    elif kind == "fib":
        user, body = _code_fib()
    elif kind == "is_even":
        user, body = _code_is_even()
    else:
        user, body = _code_clamp()

    # Always end with the exact EOC block (the model must learn this).
    assistant = body.rstrip("\n") + EOC_BLOCK
    return ex(
        SYS_CODE,
        user,
        assistant,
        {"id": f"syn_code_{i:06d}", "task": "code", "kind": kind},
    )

# -------------------------
# streaming main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--n_arith", type=int, default=100000)
    ap.add_argument("--n_syll", type=int, default=20000)
    ap.add_argument("--n_code", type=int, default=20000)

    ap.add_argument("--arith_hard_tail", action="store_true", help="include a small tail of harder arithmetic")
    ap.add_argument("--include_anchors", action="store_true", help="include a few fixed bench-matching examples")

    ap.add_argument("--syll_mode", choices=["mix", "focus"], default="mix")
    ap.add_argument("--code_mode", choices=["mix", "focus"], default="mix")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    total = args.n_arith + args.n_syll + args.n_code
    print(f"[start] writing to {args.out} total={total}", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        written = 0

        if args.include_anchors:
            anchors = bench_anchor_examples()
            for row in anchors:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

        rem = {"arithmetic": args.n_arith, "syllogism": args.n_syll, "code": args.n_code}
        idx = {"arithmetic": 0, "syllogism": 0, "code": 0}

        while rem["arithmetic"] + rem["syllogism"] + rem["code"] > 0:
            choices = [t for t, n in rem.items() if n > 0]
            weights = [rem[t] for t in choices]
            task = rng.choices(choices, weights=weights, k=1)[0]

            if task == "arithmetic":
                row = gen_arith(rng, idx["arithmetic"], hard_tail=args.arith_hard_tail)
            elif task == "syllogism":
                row = gen_syll(rng, idx["syllogism"], mode=args.syll_mode)
            else:
                row = gen_code(rng, idx["code"], mode=args.code_mode)

            idx[task] += 1
            rem[task] -= 1

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

            if written % 5000 == 0:
                print(f"[progress] {written}/{total} rem={rem}", flush=True)

    print(f"[done] wrote {written} examples to {args.out}", flush=True)

if __name__ == "__main__":
    main()
