#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import random
import textwrap
from pathlib import Path
from typing import Dict, List

SYSTEM = "You are a helpful assistant."

PROMPT_TEMPLATES = [
    "Fix the bug in this Python code and return only the corrected code:\n\n```python\n{buggy}\n```",
    "Repair this Python function. Do not explain anything. Return only one Python code block:\n\n```python\n{buggy}\n```",
]

TASKS = [
    {
        "name": "reverse_string",
        "correct": """
        def reverse_string(s):
            return s[::-1]
        """,
        "bugs": [
            """
            def reverse_string(s):
                out = ''
                for ch in s:
                    out = out + ch
                return out
            """,
            """
            def reverse_string(s):
                return s
            """,
        ],
    },
    {
        "name": "dedup_preserve_order",
        "correct": """
        def dedup_preserve_order(items):
            out = []
            for item in items:
                if item not in out:
                    out.append(item)
            return out
        """,
        "bugs": [
            """
            def dedup_preserve_order(items):
                return list(set(items))
            """,
            """
            def dedup_preserve_order(items):
                out = []
                for item in items:
                    if item in out:
                        out.append(item)
                return out
            """,
        ],
    },
    {
        "name": "count_words",
        "correct": """
        import re
        def count_words(text):
            words = re.findall(r"[A-Za-z0-9']+", text.lower())
            out = {}
            for w in words:
                out[w] = out.get(w, 0) + 1
            return out
        """,
        "bugs": [
            """
            def count_words(text):
                words = text.split()
                return len(words)
            """,
            """
            def count_words(text):
                out = {}
                for w in text.split():
                    out[w] = 1
                return out
            """,
        ],
    },
    {
        "name": "running_sum",
        "correct": """
        def running_sum(nums):
            total = 0
            out = []
            for x in nums:
                total += x
                out.append(total)
            return out
        """,
        "bugs": [
            """
            def running_sum(nums):
                total = 0
                out = []
                for x in nums:
                    out.append(total)
                return out
            """,
            """
            def running_sum(nums):
                return nums
            """,
        ],
    },
    {
        "name": "running_max",
        "correct": """
        def running_max(nums):
            out = []
            current = None
            for x in nums:
                if current is None or x > current:
                    current = x
                out.append(current)
            return out
        """,
        "bugs": [
            """
            def running_max(nums):
                out = []
                current = 0
                for x in nums:
                    if x < current:
                        current = x
                    out.append(current)
                return out
            """,
        ],
    },
    {
        "name": "safe_divide",
        "correct": """
        def safe_divide(a, b):
            return None if b == 0 else a / b
        """,
        "bugs": [
            """
            def safe_divide(a, b):
                return a / b
            """,
            """
            def safe_divide(a, b):
                if b == 0:
                    return 0
                return a * b
            """,
        ],
    },
    {
        "name": "lowercase_keys",
        "correct": """
        def lowercase_keys(d):
            return {k.lower() if isinstance(k, str) else k: v for k, v in d.items()}
        """,
        "bugs": [
            """
            def lowercase_keys(d):
                return d.lower()
            """,
        ],
    },
    {
        "name": "group_by_length",
        "correct": """
        def group_by_length(words):
            out = {}
            for w in words:
                out.setdefault(len(w), []).append(w)
            return out
        """,
        "bugs": [
            """
            def group_by_length(words):
                return {w: len(w) for w in words}
            """,
        ],
    },
]


def normalize(code: str) -> str:
    return textwrap.dedent(code).strip() + "\n"


def code_block(code: str) -> str:
    code = normalize(code)
    ast.parse(code)
    return f"```python\n{code.rstrip()}\n```"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for task in TASKS:
        name = task["name"]
        correct_block = code_block(task["correct"])
        for bug_idx, buggy in enumerate(task["bugs"]):
            buggy_norm = normalize(buggy)
            base_id = f"codefix_{name}_bug{bug_idx}"
            for rep in range(max(1, args.repeat)):
                tmpl = PROMPT_TEMPLATES[rep % len(PROMPT_TEMPLATES)]
                rows.append(
                    {
                        "messages": [
                            {"role": "system", "content": SYSTEM},
                            {
                                "role": "user",
                                "content": tmpl.format(buggy=buggy_norm.rstrip()),
                            },
                            {"role": "assistant", "content": correct_block},
                        ],
                        "meta": {
                            "dataset": "local_code_fix_simple",
                            "base_id": base_id,
                            "variant_id": f"{base_id}_v{rep}",
                            "task_name": name,
                            "bug_index": bug_idx,
                        },
                    }
                )

    rng.shuffle(rows)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} examples -> {out_path}")


if __name__ == "__main__":
    main()
