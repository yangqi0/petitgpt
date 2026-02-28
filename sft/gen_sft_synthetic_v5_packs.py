#!/usr/bin/env python3
"""
Synthetic SFT generator (v5): token-balanced multitask via multi-turn "packs".

Why:
- In your trainer, loss is averaged over supervised tokens.
- Code samples contain ~10-50x more supervised tokens than arithmetic/syllogism,
  so even if arithmetic is 60% by *examples*, it's tiny by *tokens* and won't improve.

v5 fixes this by packing many arithmetic (and optionally syllogism) turns into ONE training example,
so arithmetic contributes comparable supervised-token mass without breaking the eval format.

Output JSONL:
{"messages":[{"role":"system","content":...},{"role":"user","content":...},{"role":"assistant","content":...}, ...],
 "meta":{...}}
"""

from __future__ import annotations

import argparse
import json
import os
import random

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


# -------------------------
# arithmetic primitives
# -------------------------
def _arith_add(rng: random.Random, lo: int, hi: int) -> tuple[str, str, str]:
    a = rng.randint(lo, hi)
    b = rng.randint(lo, hi)
    q = f"Compute {a} + {b}.\nAnswer:"
    return q, str(a + b), "add"


def _arith_add_carry_0_99(rng: random.Random) -> tuple[str, str, str]:
    for _ in range(2000):
        a = rng.randint(0, 99)
        b = rng.randint(0, 99)
        if (a % 10) + (b % 10) >= 10:
            q = f"Compute {a} + {b}.\nAnswer:"
            return q, str(a + b), "add_carry"
    return _arith_add(rng, 0, 99)


def _arith_sub_nonneg(rng: random.Random, lo: int, hi: int) -> tuple[str, str, str]:
    a = rng.randint(lo, hi)
    b = rng.randint(lo, hi)
    if b > a:
        a, b = b, a
    q = f"Compute {a} - {b}.\nAnswer:"
    return q, str(a - b), "sub"


def _arith_mul_table(rng: random.Random, max_k: int = 12) -> tuple[str, str, str]:
    a = rng.randint(0, max_k)
    b = rng.randint(0, max_k)
    q = f"What is {a} * {b}?\nAnswer:"
    return q, str(a * b), "mul_table"


def _arith_word_apples(rng: random.Random, hi: int = 40) -> tuple[str, str, str]:
    a = rng.randint(0, hi)
    b = rng.randint(0, hi)
    q = f"John has {a} apples and buys {b} more. How many apples does he have?\nAnswer:"
    return q, str(a + b), "word_apples"


def arith_anchor_triplet() -> list[tuple[str, str, str]]:
    return [
        (
            "John has 3 apples and buys 2 more. How many apples does he have?\nAnswer:",
            "5",
            "anchor_apples",
        ),
        ("Compute 17 + 25.\nAnswer:", "42", "anchor_17_25"),
        ("What is 9 * 8?\nAnswer:", "72", "anchor_9_8"),
    ]


def sample_arith_qa(rng: random.Random) -> tuple[str, str, str]:
    styles = [
        "add_0_20",
        "sub_0_20",
        "mul_table",
        "word",
        "add_carry_0_99",
        "add_0_100",
        "sub_0_100",
    ]
    weights = [0.18, 0.12, 0.20, 0.08, 0.22, 0.12, 0.08]
    style = rng.choices(styles, weights=weights, k=1)[0]
    if style == "add_0_20":
        return _arith_add(rng, 0, 20)
    if style == "sub_0_20":
        return _arith_sub_nonneg(rng, 0, 20)
    if style == "mul_table":
        return _arith_mul_table(rng, 12)
    if style == "word":
        return _arith_word_apples(rng, 40)
    if style == "add_carry_0_99":
        return _arith_add_carry_0_99(rng)
    if style == "add_0_100":
        return _arith_add(rng, 0, 100)
    return _arith_sub_nonneg(rng, 0, 100)


def make_arith_pack(rng: random.Random, pack_k: int, pack_id: str, include_anchors: bool) -> dict:
    messages: list[dict[str, str]] = [{"role": "system", "content": SYS_ARITH}]
    styles: list[str] = []

    qas: list[tuple[str, str, str]] = []
    if include_anchors:
        qas.extend(arith_anchor_triplet())

    while len(qas) < pack_k:
        qas.append(sample_arith_qa(rng))

    qas = qas[:pack_k]
    for q, a, st in qas:
        styles.append(st)
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a + "\n"})

    return {
        "messages": messages,
        "meta": {"id": pack_id, "task": "arithmetic_pack", "styles": styles},
    }


# -------------------------
# syllogism pack (optional)
# -------------------------
def sample_syll(rng: random.Random) -> tuple[str, str, str]:
    styles = ["cats_unknown", "all_yes", "some_yes", "all_unknown", "some_no"]
    weights = [0.35, 0.20, 0.20, 0.15, 0.10]
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

    return q, ans, style


def make_syll_pack(rng: random.Random, pack_k: int, pack_id: str) -> dict:
    messages: list[dict[str, str]] = [{"role": "system", "content": SYS_SYLL}]
    styles: list[str] = []
    for _ in range(pack_k):
        q, a, st = sample_syll(rng)
        styles.append(st)
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a + "\n"})
    return {
        "messages": messages,
        "meta": {"id": pack_id, "task": "syllogism_pack", "styles": styles},
    }


# -------------------------
# code (single-turn)
# -------------------------
def make_code_ex(rng: random.Random, ex_id: str) -> dict:
    kinds = ["add", "factorial", "fib"]
    weights = [0.30, 0.35, 0.35]
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
    else:
        user = "Complete the following Python function:\n\ndef fib(n):\n    "
        body = (
            "if n < 0:\n"
            "    raise ValueError('n must be non-negative')\n"
            "a, b = 0, 1\n"
            "for _ in range(n):\n"
            "    a, b = b, a + b\n"
            "return a\n"
        )

    assistant = body.rstrip("\n") + EOC_BLOCK
    return {
        "messages": [
            {"role": "system", "content": SYS_CODE},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "meta": {"id": ex_id, "task": "code", "kind": kind},
    }


# -------------------------
# main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)

    # Token-balanced defaults (for your trainer settings):
    # - arithmetic packs carry the token mass
    # - small amount of code to prevent forgetting (code has many tokens per example)
    # - medium syll packs to keep format
    ap.add_argument("--n_arith_packs", type=int, default=15000)
    ap.add_argument("--arith_pack_k", type=int, default=16)
    ap.add_argument("--arith_pack_include_anchors", action="store_true")

    ap.add_argument("--n_syll_packs", type=int, default=5000)
    ap.add_argument("--syll_pack_k", type=int, default=8)

    ap.add_argument("--n_code", type=int, default=5000)

    args = ap.parse_args()
    rng = random.Random(args.seed)

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    total = args.n_arith_packs + args.n_syll_packs + args.n_code
    print(f"[start] writing {total} examples to {args.out}", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        rem = {"arith": args.n_arith_packs, "syll": args.n_syll_packs, "code": args.n_code}
        idx = {"arith": 0, "syll": 0, "code": 0}
        written = 0

        while rem["arith"] + rem["syll"] + rem["code"] > 0:
            choices = [k for k, n in rem.items() if n > 0]
            weights = [rem[k] for k in choices]
            kind = rng.choices(choices, weights=weights, k=1)[0]

            if kind == "arith":
                row = make_arith_pack(
                    rng,
                    pack_k=args.arith_pack_k,
                    pack_id=f"arith_pack_{idx['arith']:06d}",
                    include_anchors=args.arith_pack_include_anchors,
                )
            elif kind == "syll":
                row = make_syll_pack(
                    rng,
                    pack_k=args.syll_pack_k,
                    pack_id=f"syll_pack_{idx['syll']:06d}",
                )
            else:
                row = make_code_ex(rng, ex_id=f"code_{idx['code']:06d}")

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            idx[kind] += 1
            rem[kind] -= 1
            written += 1

            if written % 2000 == 0:
                print(f"[progress] {written}/{total} rem={rem}", flush=True)

    print("[done]", flush=True)


if __name__ == "__main__":
    main()
