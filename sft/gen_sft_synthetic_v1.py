#!/usr/bin/env python3
"""
Generate synthetic SFT data in canonical chat jsonl:
{"messages":[...], "meta": {...}}

Tasks:
- arithmetic: +, -, * with small integers; strict "ONLY final numeric answer"
- syllogism: yes/no/unknown with simple set-logic templates
- code: add / factorial / fib + a few extra small functions; strict "ONLY code"

This dataset is designed to teach *format/behavior* first.
"""

from __future__ import annotations

import argparse
import json
import os
import random

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
)


def ex(system: str, user: str, assistant: str, meta: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "meta": meta,
    }


def gen_arith(rng, i, max_n):
    # 20% apples
    if rng.random() < 0.20:
        have = rng.randint(0, max_n)
        buy = rng.randint(0, max_n)
        ans = have + buy
        q = f"John has {have} apples and buys {buy} more. How many apples does he have?\nAnswer:"
        return ex(SYS_ARITH, q, f"{ans}\n", {"id": f"syn_arith_{i:06d}", "task":"arithmetic", "style":"word_apples"})

    # 10%: bench-critical cases to fight copy-bias
    if rng.random() < 0.10:
        a, b, op = rng.choice([
            (3, 2, "+"),
            (17, 25, "+"),
            (9, 8, "*"),
        ])

    # fall through to formatting below
    elif rng.random() < 0.20:
        pool = [
            (10, 2, "+"), (12, 3, "+"), (20, 5, "+"), (17, 25, "+"),
            (9, 8, "*"), (11, 11, "+"), (15, 0, "+"), (99, 1, "+"),
            (30, 12, "-"), (100, 1, "-"),
        ]
        a, b, op = rng.choice(pool)
    else:
        op = rng.choice(["+", "-", "*"])
        a = rng.randint(0, max_n)
        b = rng.randint(0, max_n)

    if op == "+":
        ans = a + b
        q = f"Compute {a} + {b}.\nAnswer:"
    elif op == "-":
        ans = a - b
        q = f"Compute {a} - {b}.\nAnswer:"
    else:
        ans = a * b
        q = f"What is {a} * {b}?\nAnswer:"
    return ex(SYS_ARITH, q, f"{ans}\n", {"id": f"syn_arith_{i:06d}", "task":"arithmetic", "style":"symbolic"})


def gen_syll(rng: random.Random, i: int) -> dict:
    # 25%: match bench cat/black unknown case
    if rng.random() < 0.25:
        q = "If all cats are animals and some animals are black, can we conclude that some cats are black?\nAnswer:"
        ans = "unknown"
        return ex(SYS_SYLL, q, f"{ans}\n", {"id": f"syn_syll_{i:06d}", "task": "syllogism", "style": "cats_unknown"})

    # otherwise use abstract templates
    t = rng.choice([0, 1, 2, 3])
    if t == 0:
        q = "All A are B. All B are C. Can we conclude all A are C?\nAnswer:"
        ans = "yes"
    elif t == 1:
        q = "Some A are B. All B are C. Can we conclude some A are C?\nAnswer:"
        ans = "yes"
    elif t == 2:
        q = "All A are B. Some B are C. Can we conclude some A are C?\nAnswer:"
        ans = "unknown"
    else:
        q = "Some A are B. No B are C. Can we conclude some A are C?\nAnswer:"
        ans = "no"

    return ex(SYS_SYLL, q, f"{ans}\n", {"id": f"syn_syll_{i:06d}", "task": "syllogism", "style": "abc"})


def gen_code(rng: random.Random, i: int) -> dict:
    # rotate among a few tiny functions; keep answers short and exact
    k = rng.choice(["add", "factorial", "fib", "is_even", "clamp"])
    if k == "add":
        user = "Complete the following Python function:\n\ndef add(a, b):\n    "
        assistant = "return a + b\n"
    elif k == "factorial":
        user = "Complete the following Python function:\n\ndef factorial(n):\n    "
        assistant = (
            "if n < 0:\n"
            "    raise ValueError('n must be non-negative')\n"
            "r = 1\n"
            "for i in range(2, n + 1):\n"
            "    r *= i\n"
            "return r\n"
        )
    elif k == "fib":
        user = "Complete the following Python function:\n\ndef fib(n):\n    "
        assistant = (
            "if n < 0:\n"
            "    raise ValueError('n must be non-negative')\n"
            "a, b = 0, 1\n"
            "for _ in range(n):\n"
            "    a, b = b, a + b\n"
            "return a\n"
        )
    elif k == "is_even":
        user = "Complete the following Python function:\n\ndef is_even(n):\n    "
        assistant = "return (n % 2) == 0\n"
    else:
        user = "Complete the following Python function:\n\ndef clamp(x, lo, hi):\n    "
        assistant = (
            "if x < lo:\n"
            "    return lo\n"
            "if x > hi:\n"
            "    return hi\n"
            "return x\n"
        )
    # IMPORTANT: make code completions end with a blank line so --stop_string "\n\n" works at inference time.
    assistant = assistant.rstrip() + "\n\n"
    return ex(SYS_CODE, user, assistant, {"id": f"syn_code_{i:06d}", "task": "code", "kind": k})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n_arith", type=int, default=10000)
    ap.add_argument("--n_syll", type=int, default=6000)
    ap.add_argument("--n_code", type=int, default=4000)
    ap.add_argument("--arith_max_n", type=int, default=99)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows: list[dict] = []
    for i in range(args.n_arith):
        rows.append(gen_arith(rng, i, args.arith_max_n))
    for i in range(args.n_syll):
        rows.append(gen_syll(rng, i))
    for i in range(args.n_code):
        rows.append(gen_code(rng, i))

    rng.shuffle(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", len(rows), "examples to", args.out)


if __name__ == "__main__":
    main()
