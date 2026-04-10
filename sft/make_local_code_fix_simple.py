#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import random
import textwrap
from pathlib import Path

SYSTEM = "You are a helpful assistant."

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
                    out += ch
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
            seen = set()
            out = []
            for item in items:
                if item not in seen:
                    seen.add(item)
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
            if b == 0:
                return None
            return a / b
        """,
        "bugs": [
            """
            def safe_divide(a, b):
                return a / b
            """,
            """
            def safe_divide(a, b):
                if a == 0:
                    return None
                return a / b
            """,
        ],
    },
    {
        "name": "get_common_elements",
        "correct": """
        def get_common_elements(a, b):
            bset = set(b)
            return [x for x in a if x in bset]
        """,
        "bugs": [
            """
            def get_common_elements(a, b):
                return [x for x in a if x not in b]
            """,
        ],
    },
    {
        "name": "swap_first_last",
        "correct": """
        def swap_first_last(items):
            if len(items) < 2:
                return items[:]
            out = items[:]
            out[0], out[-1] = out[-1], out[0]
            return out
        """,
        "bugs": [
            """
            def swap_first_last(items):
                if len(items) < 2:
                    return items
                items[0] = items[-1]
                return items
            """,
        ],
    },
    {
        "name": "lowercase_keys",
        "correct": """
        def lowercase_keys(d):
            return {str(k).lower(): v for k, v in d.items()}
        """,
        "bugs": [
            """
            def lowercase_keys(d):
                return {k: v for k, v in d.items()}
            """,
        ],
    },
    {
        "name": "flatten_once",
        "correct": """
        def flatten_once(items):
            out = []
            for x in items:
                if isinstance(x, list):
                    out.extend(x)
                else:
                    out.append(x)
            return out
        """,
        "bugs": [
            """
            def flatten_once(items):
                out = []
                for x in items:
                    out.append(x)
                return out
            """,
        ],
    },
    {
        "name": "chunk_list",
        "correct": """
        def chunk_list(items, size):
            if size <= 0:
                raise ValueError('size must be positive')
            return [items[i:i + size] for i in range(0, len(items), size)]
        """,
        "bugs": [
            """
            def chunk_list(items, size):
                return [items[:size]]
            """,
        ],
    },
    {
        "name": "binary_search",
        "correct": """
        def binary_search(nums, target):
            left, right = 0, len(nums) - 1
            while left <= right:
                mid = (left + right) // 2
                if nums[mid] == target:
                    return mid
                if nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            return -1
        """,
        "bugs": [
            """
            def binary_search(nums, target):
                for i, x in enumerate(nums):
                    if x > target:
                        return i
                return -1
            """,
        ],
    },
]

PROMPT_TEMPLATES = [
    "Fix the bug in this Python code and return only the corrected code:\n\n{buggy}",
    "Correct this Python function. Reply with only one Python code block:\n\n{buggy}",
    "The following Python code is wrong. Fix it and return only the corrected code:\n\n{buggy}",
    "Repair this Python function. Do not explain anything. Return only the corrected code:\n\n{buggy}",
]


def normalize(code: str) -> str:
    return textwrap.dedent(code).strip() + "\n"


def to_code_block(code: str) -> str:
    return f"```python\n{normalize(code)}```"


def ast_ok(code: str) -> bool:
    try:
        ast.parse(normalize(code))
        return True
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--repeat", type=int, default=28)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    examples = []
    for task in TASKS:
        correct = normalize(task["correct"])
        assert ast_ok(correct)
        for buggy in task["bugs"]:
            buggy_n = normalize(buggy)
            assert ast_ok(buggy_n)
            for _ in range(args.repeat):
                tmpl = rng.choice(PROMPT_TEMPLATES)
                user = tmpl.format(buggy=to_code_block(buggy_n))
                assistant = to_code_block(correct)
                examples.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": assistant},
                    ]
                })

    rng.shuffle(examples)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"wrote {len(examples)} examples -> {out_path}")


if __name__ == "__main__":
    main()
