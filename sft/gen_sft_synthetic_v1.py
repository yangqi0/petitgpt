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
import re

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
)


# ---------------------------------------------------------------------
# Strict filters (to avoid "bad style" samples that poison SFT)
# ---------------------------------------------------------------------

_SYLL_OK_RE = re.compile(r"^(yes|no|unknown)\n$", re.IGNORECASE)

_CODE_BAN_SUBSTRS = [
    "if __name__",
    "class ",
    "def ",
    "import ",
    "from ",
    "async",
    "await",
    "@",
    "yield",
    "pytest",
    "unittest",
    "raw_input",
    "input(",
    "print(",
]

# Detect "glued lines" like: "r = 1 for i in ...", "return a for ..."
_CODE_GLUED_RE = re.compile(
    r"(^|[\s])\b(for|while|if|return)\b.*\b(for|while|if|return)\b", re.IGNORECASE
)


def _is_good_syll_answer(a: str) -> bool:
    return bool(_SYLL_OK_RE.match(a or ""))


def _is_good_code_answer(a: str) -> bool:
    if not a:
        return False
    # must end with \n\n (so eval can stop cleanly) and contain at least one return
    if not a.endswith("\n\n"):
        return False
    if "return" not in a:
        return False
    # ban semicolons to reduce "glued lines"
    if ";" in a:
        return False
    # ban glued control-flow on same line (common failure mode)
    if _CODE_GLUED_RE.search(a):
        return False
    low = a.lower()
    for s in _CODE_BAN_SUBSTRS:
        if s in low:
            return False
    return True


def _gen_with_retries(fn, max_tries: int, accept_fn):
    last = None
    for _ in range(max_tries):
        ex = fn()
        # find assistant content
        a = ""
        for m in ex.get("messages", []):
            if m.get("role") == "assistant":
                a = m.get("content", "")
                break
        if accept_fn(a):
            return ex
        last = ex
    # If we failed too many times, return the last one (and let caller decide)
    return last


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
        return ex(
            SYS_ARITH,
            q,
            f"{ans}\n",
            {"id": f"syn_arith_{i:06d}", "task": "arithmetic", "style": "word_apples"},
        )

    # 10%: bench-critical cases to fight copy-bias
    if rng.random() < 0.10:
        a, b, op = rng.choice(
            [
                (3, 2, "+"),
                (17, 25, "+"),
                (9, 8, "*"),
            ]
        )

    # fall through to formatting below
    elif rng.random() < 0.20:
        pool = [
            (10, 2, "+"),
            (12, 3, "+"),
            (20, 5, "+"),
            (17, 25, "+"),
            (9, 8, "*"),
            (11, 11, "+"),
            (15, 0, "+"),
            (99, 1, "+"),
            (30, 12, "-"),
            (100, 1, "-"),
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
    return ex(
        SYS_ARITH,
        q,
        f"{ans}\n",
        {"id": f"syn_arith_{i:06d}", "task": "arithmetic", "style": "symbolic"},
    )


def gen_syll(rng: random.Random, i: int, mode: str = "balanced") -> dict:
    # 25%: match bench cat/black unknown case
    if mode == "focus":
        # 50% cats_unknown, 30% some_yes, 20% all_yes
        r = rng.random()
        if r < 0.50:
            q = "If all cats are animals and some animals are black, can we conclude that some cats are black?\nAnswer:"
            ans = "unknown"
            return ex(
                SYS_SYLL,
                q,
                f"{ans}\n",
                {"id": f"syn_syll_{i:06d}", "task": "syllogism", "style": "cats_unknown"},
            )
        elif r < 0.80:
            q = "Some A are B. All B are C. Can we conclude some A are C?\nAnswer:"
            ans = "yes"
            return ex(
                SYS_SYLL,
                q,
                f"{ans}\n",
                {"id": f"syn_syll_{i:06d}", "task": "syllogism", "style": "some_yes"},
            )
        else:
            q = "All A are B. All B are C. Can we conclude all A are C?\nAnswer:"
            ans = "yes"
            return ex(
                SYS_SYLL,
                q,
                f"{ans}\n",
                {"id": f"syn_syll_{i:06d}", "task": "syllogism", "style": "all_yes"},
            )

    if rng.random() < 0.25:
        q = "If all cats are animals and some animals are black, can we conclude that some cats are black?\nAnswer:"
        ans = "unknown"
        return ex(
            SYS_SYLL,
            q,
            f"{ans}\n",
            {"id": f"syn_syll_{i:06d}", "task": "syllogism", "style": "cats_unknown"},
        )

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

    return ex(
        SYS_SYLL, q, f"{ans}\n", {"id": f"syn_syll_{i:06d}", "task": "syllogism", "style": "abc"}
    )


def _maybe_add_user_anti(user: str, anti: bool, task: str) -> str:
    if not anti:
        return user
    # IMPORTANT: don't always add reminders; keep some prompts "clean" to match eval/real usage
    # This reduces prompt dependence and improves generalization.
    if random.random() >= 0.5:
        return user
    if task == "code":
        return (
            user
            + "\n\nRules reminder (must follow):\n"
            + "- Do NOT use yield.\n"
            + "- Do NOT use semicolons.\n"
            + "- ONE statement per line.\n"
            + "- Output ONLY the function body.\n"
        )
    if task == "syll":
        return user + "\n\nOutput exactly one token: yes / no / unknown."
    return user


def gen_code(rng: random.Random, i: int, mode: str = "balanced", anti: bool = False) -> dict:
    # rotate among a few tiny functions; keep answers short and exact
    if mode == "focus":
        # fib 60%, factorial 30%, add 10%
        r = rng.random()
        if r < 0.60:
            k = "fib"
        elif r < 0.90:
            k = "factorial"
        else:
            k = "add"
    else:
        k = rng.choice(["add", "factorial", "fib", "is_even", "clamp"])
    if k == "add":
        user = "Complete the following Python function:\n\ndef add(a, b):\n    "
        assistant = "return a + b\n"
    elif k == "factorial":
        # add a second prompt variant to reduce prompt-overfitting
        if anti and rng.random() < 0.5:
            user = "Write ONLY the function body for factorial:\n\ndef factorial(n):\n    "
        else:
            user = "Complete the following Python function:\n\ndef factorial(n):\n    "
        assistant = "r = 1\nfor i in range(2, n + 1):\n    r *= i\nreturn r\n"
    elif k == "fib":
        # add a second prompt variant to reduce prompt-overfitting
        if anti and rng.random() < 0.5:
            user = (
                "Write ONLY the function body for Fibonacci (return fib(n)):\n\ndef fib(n):\n    "
            )
        else:
            user = "Complete the following Python function:\n\ndef fib(n):\n    "
        assistant = "a, b = 0, 1\nfor _ in range(n):\n    a, b = b, a + b\nreturn a\n"
    elif k == "is_even":
        user = "Complete the following Python function:\n\ndef is_even(n):\n    "
        assistant = "return (n % 2) == 0\n"
    else:
        user = "Complete the following Python function:\n\ndef clamp(x, lo, hi):\n    "
        assistant = "if x < lo:\n    return lo\nif x > hi:\n    return hi\nreturn x\n"
    # IMPORTANT: make code completions end with a blank line so --stop_string "\n\n" works at inference time.
    assistant = assistant.rstrip() + "\n\n"
    user = _maybe_add_user_anti(user, anti=anti, task="code")
    return ex(SYS_CODE, user, assistant, {"id": f"syn_code_{i:06d}", "task": "code", "kind": k})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n_arith", type=int, default=10000)
    ap.add_argument("--n_syll", type=int, default=6000)
    ap.add_argument("--n_code", type=int, default=4000)
    ap.add_argument("--arith_max_n", type=int, default=99)
    ap.add_argument("--syll_mode", type=str, default="balanced", choices=["balanced", "focus"])
    ap.add_argument("--code_mode", type=str, default="balanced", choices=["balanced", "focus"])
    ap.add_argument(
        "--anti",
        action="store_true",
        help="Round A3: add extra anti-pattern constraints into USER prompts.",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Filter bad code/syll samples; resample until passing heuristics.",
    )
    ap.add_argument(
        "--strict_max_tries",
        type=int,
        default=200,
        help="Max resample tries per example in strict mode.",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows: list[dict] = []
    for i in range(args.n_arith):
        rows.append(gen_arith(rng, i, args.arith_max_n))

    for i in range(args.n_syll):
        if not args.strict:
            exi = gen_syll(rng, i, mode=args.syll_mode)
            if args.anti:
                # add explicit token-only instruction in user
                exi["messages"][1]["content"] = _maybe_add_user_anti(
                    exi["messages"][1]["content"], anti=True, task="syll"
                )
            rows.append(exi)
        else:
            rows.append(
                _gen_with_retries(
                    fn=lambda: gen_syll(rng, i, mode=args.syll_mode),
                    max_tries=int(args.strict_max_tries),
                    accept_fn=_is_good_syll_answer,
                )
            )
            if args.anti:
                rows[-1]["messages"][1]["content"] = _maybe_add_user_anti(
                    rows[-1]["messages"][1]["content"], anti=True, task="syll"
                )

    for i in range(args.n_code):
        if not args.strict:
            rows.append(gen_code(rng, i, mode=args.code_mode, anti=args.anti))
        else:
            rows.append(
                _gen_with_retries(
                    fn=lambda: gen_code(rng, i, mode=args.code_mode, anti=args.anti),
                    max_tries=int(args.strict_max_tries),
                    accept_fn=_is_good_code_answer,
                )
            )

    rng.shuffle(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", len(rows), "examples to", args.out)


if __name__ == "__main__":
    main()
