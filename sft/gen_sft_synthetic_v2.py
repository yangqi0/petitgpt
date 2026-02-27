#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

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
# arithmetic
# -------------------------
def _arith_symbolic(rng: random.Random, max_n: int) -> Tuple[str, str, str]:
    op = rng.choice(["+", "-", "*"])
    a = rng.randint(0, max_n)
    b = rng.randint(0, max_n)
    if op == "+":
        q = f"Compute {a} + {b}.\nAnswer:"
        ans = str(a + b)
        style = "symbolic"
    elif op == "-":
        q = f"Compute {a} - {b}.\nAnswer:"
        ans = str(a - b)
        style = "symbolic"
    else:
        q = f"What is {a} * {b}?\nAnswer:"
        ans = str(a * b)
        style = "symbolic"
    return q, ans, style

def _arith_abc(rng: random.Random, max_n: int) -> Tuple[str, str, str]:
    op = rng.choice(["+", "-", "*"])
    a = rng.randint(0, max_n)
    b = rng.randint(0, max_n)
    if op == "+":
        q = f"Let A = {a} and B = {b}. Compute A + B.\nAnswer:"
        ans = str(a + b)
    elif op == "-":
        q = f"Let A = {a} and B = {b}. Compute A - B.\nAnswer:"
        ans = str(a - b)
    else:
        q = f"Let A = {a} and B = {b}. Compute A * B.\nAnswer:"
        ans = str(a * b)
    return q, ans, "abc"

def _arith_apples(rng: random.Random, max_n: int) -> Tuple[str, str, str]:
    a = rng.randint(0, max_n)
    b = rng.randint(0, max_n)
    q = f"John has {a} apples and buys {b} more. How many apples does he have?\nAnswer:"
    ans = str(a + b)
    return q, ans, "word_apples"

def gen_arith(rng: random.Random, i: int, max_n: int) -> Dict:
    style = rng.choices(
        ["symbolic", "abc", "word_apples"],
        weights=[0.70, 0.20, 0.10],
        k=1,
    )[0]
    if style == "symbolic":
        q, ans, style2 = _arith_symbolic(rng, max_n)
    elif style == "abc":
        q, ans, style2 = _arith_abc(rng, max_n)
    else:
        q, ans, style2 = _arith_apples(rng, max_n)
    return ex(
        SYS_ARITH,
        q,
        ans + "\n",
        {"id": f"syn_arith_{i:06d}", "task": "arithmetic", "style": style2},
    )

# -------------------------
# syllogism
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
# code
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
    weights = [0.20, 0.20, 0.20, 0.20, 0.20] if mode == "mix" else [0.15, 0.25, 0.25, 0.15, 0.20]
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
    ap.add_argument("--n_arith", type=int, default=10000)
    ap.add_argument("--n_syll", type=int, default=6000)
    ap.add_argument("--n_code", type=int, default=4000)
    ap.add_argument("--arith_max_n", type=int, default=99)
    ap.add_argument("--syll_mode", choices=["mix", "focus"], default="mix")
    ap.add_argument("--code_mode", choices=["mix", "focus"], default="mix")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    total = args.n_arith + args.n_syll + args.n_code
    print(f"[start] writing to {args.out} total={total}", flush=True)

    # Open file immediately so you can see it exists right away.
    with open(args.out, "w", encoding="utf-8") as f:
        # Interleave tasks by sampling according to remaining counts (so no huge blocks).
        rem = {"arithmetic": args.n_arith, "syllogism": args.n_syll, "code": args.n_code}
        idx = {"arithmetic": 0, "syllogism": 0, "code": 0}
        written = 0

        while rem["arithmetic"] + rem["syllogism"] + rem["code"] > 0:
            choices = [t for t, n in rem.items() if n > 0]
            weights = [rem[t] for t in choices]
            task = rng.choices(choices, weights=weights, k=1)[0]

            if task == "arithmetic":
                row = gen_arith(rng, idx["arithmetic"], args.arith_max_n)
            elif task == "syllogism":
                row = gen_syll(rng, idx["syllogism"], mode=args.syll_mode)
            else:
                row = gen_code(rng, idx["code"], mode=args.code_mode)

            idx[task] += 1
            rem[task] -= 1

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

            if written % 2000 == 0:
                print(f"[progress] {written}/{total} rem={rem}", flush=True)

    print(f"[done] wrote {total} examples to {args.out}", flush=True)

if __name__ == "__main__":
    main()
