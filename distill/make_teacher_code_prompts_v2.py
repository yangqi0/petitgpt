#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a large code teacher prompt set with hidden tests in metadata.

Each row contains:
- id
- task_type
- prompt
- meta: {
    "family": ...,
    "entrypoint": ...,
    "tests": [...],
  }

The verifier uses meta.tests to run minimal semantic checks.

Comments are in English by design.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import random
from typing import Any


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def t(args=None, kwargs=None, expected=None, raises=None):
    return {
        "args": [] if args is None else args,
        "kwargs": {} if kwargs is None else kwargs,
        "expected": expected,
        "raises": raises,
    }


def bug_reverse_string() -> str:
    return "def reverse_string(s):\n    out = ''\n    for ch in s:\n        out = out + ch\n    return out\n"


def bug_dedup() -> str:
    return "def dedup_preserve_order(items):\n    seen = set()\n    out = []\n    for x in items:\n        if x in seen:\n            out.append(x)\n        seen.add(x)\n    return out\n"


def bug_count_words() -> str:
    return "def count_words(text):\n    words = text.lower().split()\n    freq = {}\n    for w in words:\n        freq[w] = freq.get(w, 0) + 1\n    return list(freq.items())\n"


def bug_binary_search() -> str:
    return "def binary_search(nums, target):\n    left, right = 0, len(nums) - 1\n    while left < right:\n        mid = (left + right) // 2\n        if nums[mid] < target:\n            right = mid - 1\n        else:\n            left = mid + 1\n    return -1\n"


def bug_safe_divide() -> str:
    return "def safe_divide(a, b):\n    return a // b\n"


def build_task_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    specs.append({
        "family": "fib_iterative",
        "entrypoint": "fib",
        "write_prompts": [
            "Write a Python function fib(n) that returns the n-th Fibonacci number iteratively. Raise ValueError if n < 0.",
            "Write a Python function fib(n) that returns the nth Fibonacci number using an iterative loop. Raise ValueError for negative n.",
            "Implement fib(n) in Python. It should return the n-th Fibonacci number iteratively and raise ValueError if n is negative.",
        ],
        "tests": [
            t(args=[0], expected=0),
            t(args=[1], expected=1),
            t(args=[7], expected=13),
            t(args=[10], expected=55),
            t(args=[-1], raises="ValueError"),
        ],
    })

    specs.append({
        "family": "factorial",
        "entrypoint": "factorial",
        "write_prompts": [
            "Write a Python function factorial(n) that returns n!. Raise ValueError if n < 0.",
            "Implement factorial(n) in Python. Return n factorial and raise ValueError for negative inputs.",
            "Write a Python function factorial(n). It should compute n! and raise ValueError if n is negative.",
        ],
        "tests": [
            t(args=[0], expected=1),
            t(args=[1], expected=1),
            t(args=[5], expected=120),
            t(args=[7], expected=5040),
            t(args=[-3], raises="ValueError"),
        ],
    })

    specs.append({
        "family": "reverse_string",
        "entrypoint": "reverse_string",
        "write_prompts": [
            "Write a Python function reverse_string(s) that returns the reversed string.",
            "Implement reverse_string(s) in Python so it returns the input string in reverse order.",
            "Write a Python function reverse_string(s). It should return a reversed copy of the string.",
        ],
        "fix_prompt": "Fix this Python function so it returns the reversed string correctly:\n\n{buggy}",
        "buggy": bug_reverse_string(),
        "tests": [
            t(args=["abc"], expected="cba"),
            t(args=[""], expected=""),
            t(args=["A b"], expected="b A"),
        ],
    })

    specs.append({
        "family": "dedup_preserve_order",
        "entrypoint": "dedup_preserve_order",
        "write_prompts": [
            "Write a Python function dedup_preserve_order(items) that removes duplicates while keeping the first occurrence order.",
            "Implement dedup_preserve_order(items). It should remove duplicates and preserve the original first-seen order.",
            "Write dedup_preserve_order(items) in Python. Return a list with duplicates removed, keeping the first occurrence of each item.",
        ],
        "fix_prompt": "Fix this Python function so it removes duplicates while preserving first occurrence order:\n\n{buggy}",
        "buggy": bug_dedup(),
        "tests": [
            t(args=[[1, 2, 1, 3, 2]], expected=[1, 2, 3]),
            t(args=[["a", "b", "a", "c"]], expected=["a", "b", "c"]),
            t(args=[[]], expected=[]),
        ],
    })

    specs.append({
        "family": "count_words",
        "entrypoint": "count_words",
        "write_prompts": [
            "Write a Python function count_words(text) that returns a dictionary of lowercase word frequencies. Ignore punctuation like commas and periods.",
            "Implement count_words(text). Return a dictionary of lowercase word counts and ignore punctuation such as commas and periods.",
            "Write a Python function count_words(text). It should count lowercase words and ignore punctuation like commas and periods.",
        ],
        "fix_prompt": "Fix this Python function so it returns a dictionary of lowercase word frequencies and ignores punctuation:\n\n{buggy}",
        "buggy": bug_count_words(),
        "tests": [
            t(args=["Hi, hi."], expected={"hi": 2}),
            t(args=["One fish, two fish."], expected={"one": 1, "fish": 2, "two": 1}),
            t(args=[""], expected={}),
        ],
    })

    specs.append({
        "family": "gcd",
        "entrypoint": "gcd",
        "write_prompts": [
            "Write a Python function gcd(a, b) that returns the greatest common divisor of two integers.",
            "Implement gcd(a, b) in Python. Return the greatest common divisor of the two integers.",
            "Write a Python function gcd(a, b). It should compute the greatest common divisor.",
        ],
        "tests": [
            t(args=[12, 18], expected=6),
            t(args=[7, 5], expected=1),
            t(args=[0, 9], expected=9),
        ],
    })

    specs.append({
        "family": "lcm",
        "entrypoint": "lcm",
        "write_prompts": [
            "Write a Python function lcm(a, b) that returns the least common multiple of two integers.",
            "Implement lcm(a, b) in Python. Return the least common multiple of the two integers.",
            "Write a Python function lcm(a, b). It should compute the least common multiple.",
        ],
        "tests": [
            t(args=[4, 6], expected=12),
            t(args=[7, 5], expected=35),
            t(args=[0, 9], expected=0),
        ],
    })

    specs.append({
        "family": "linear_search",
        "entrypoint": "linear_search",
        "write_prompts": [
            "Write a Python function linear_search(nums, target) that returns the index of the first occurrence of target, or -1 if not found.",
            "Implement linear_search(nums, target). Return the index of the first matching element, or -1 if the target is absent.",
            "Write linear_search(nums, target) in Python. Return the first index of target or -1 when target is not present.",
        ],
        "tests": [
            t(args=[[5, 1, 5, 2], 5], expected=0),
            t(args=[[1, 2, 3], 4], expected=-1),
            t(args=[[], 1], expected=-1),
        ],
    })

    specs.append({
        "family": "binary_search",
        "entrypoint": "binary_search",
        "write_prompts": [
            "Write a Python function binary_search(nums, target) that returns the index of target in a sorted list, or -1 if not found.",
            "Implement binary_search(nums, target). The input list is sorted. Return the index of target or -1 if it is missing.",
            "Write binary_search(nums, target) in Python for a sorted list. Return the target index or -1 when not found.",
        ],
        "fix_prompt": "Fix this Python function so it correctly performs binary search on a sorted list:\n\n{buggy}",
        "buggy": bug_binary_search(),
        "tests": [
            t(args=[[1, 3, 5, 7], 1], expected=0),
            t(args=[[1, 3, 5, 7], 7], expected=3),
            t(args=[[1, 3, 5, 7], 4], expected=-1),
            t(args=[[], 4], expected=-1),
        ],
    })

    specs.append({
        "family": "chunk_list",
        "entrypoint": "chunk_list",
        "write_prompts": [
            "Write a Python function chunk_list(items, size) that splits a list into chunks of length size. Raise ValueError if size <= 0.",
            "Implement chunk_list(items, size). Return a list of chunks of length size, and raise ValueError when size <= 0.",
            "Write chunk_list(items, size) in Python. Split the list into chunks and raise ValueError if size is not positive.",
        ],
        "tests": [
            t(args=[[1, 2, 3, 4, 5], 2], expected=[[1, 2], [3, 4], [5]]),
            t(args=[[], 3], expected=[]),
            t(args=[[1, 2], 0], raises="ValueError"),
        ],
    })

    specs.append({
        "family": "merge_sorted_lists",
        "entrypoint": "merge_sorted_lists",
        "write_prompts": [
            "Write a Python function merge_sorted_lists(a, b) that merges two sorted lists into one sorted list.",
            "Implement merge_sorted_lists(a, b). Both inputs are sorted. Return a merged sorted list.",
            "Write merge_sorted_lists(a, b) in Python. Merge two sorted lists and return the combined sorted result.",
        ],
        "tests": [
            t(args=[[1, 3, 5], [2, 4, 6]], expected=[1, 2, 3, 4, 5, 6]),
            t(args=[[], [1, 2]], expected=[1, 2]),
            t(args=[[1, 2], []], expected=[1, 2]),
        ],
    })

    specs.append({
        "family": "remove_none",
        "entrypoint": "remove_none",
        "write_prompts": [
            "Write a Python function remove_none(items) that returns a new list without None values.",
            "Implement remove_none(items). Return a list that keeps all elements except None.",
            "Write remove_none(items) in Python. It should remove None values and keep everything else.",
        ],
        "tests": [
            t(args=[[1, None, 2, None, 3]], expected=[1, 2, 3]),
            t(args=[[None, None]], expected=[]),
            t(args=[[]], expected=[]),
        ],
    })

    specs.append({
        "family": "safe_divide",
        "entrypoint": "safe_divide",
        "write_prompts": [
            "Write a Python function safe_divide(a, b) that returns a / b, but returns None if b == 0.",
            "Implement safe_divide(a, b). Return a divided by b, except return None when b is zero.",
            "Write safe_divide(a, b) in Python. It should return the quotient, or None if division by zero would occur.",
        ],
        "fix_prompt": "Fix this Python function so it returns a / b and returns None when b == 0:\n\n{buggy}",
        "buggy": bug_safe_divide(),
        "tests": [
            t(args=[8, 2], expected=4.0),
            t(args=[7, 0], expected=None),
            t(args=[5, 2], expected=2.5),
        ],
    })

    specs.append({
        "family": "flatten_once",
        "entrypoint": "flatten_once",
        "write_prompts": [
            "Write a Python function flatten_once(items) that flattens a list by one level only.",
            "Implement flatten_once(items). Flatten nested lists by exactly one level.",
            "Write flatten_once(items) in Python. It should flatten one nesting level and leave deeper nesting intact.",
        ],
        "tests": [
            t(args=[[[1, 2], [3], 4]], expected=[1, 2, 3, 4]),
            t(args=[[[1, [2]], 3]], expected=[1, [2], 3]),
            t(args=[[1, 2, 3]], expected=[1, 2, 3]),
        ],
    })

    specs.append({
        "family": "top_k",
        "entrypoint": "top_k",
        "write_prompts": [
            "Write a Python function top_k(nums, k) that returns the k largest numbers in descending order.",
            "Implement top_k(nums, k). Return the k largest values sorted from largest to smallest.",
            "Write top_k(nums, k) in Python. It should return the k largest numbers in descending order.",
        ],
        "tests": [
            t(args=[[3, 1, 5, 2, 4], 2], expected=[5, 4]),
            t(args=[[3, 1], 5], expected=[3, 1]),
            t(args=[[], 3], expected=[]),
        ],
    })

    specs.append({
        "family": "sum_even_numbers",
        "entrypoint": "sum_even_numbers",
        "write_prompts": [
            "Write a Python function sum_even_numbers(nums) that returns the sum of all even integers in the list.",
            "Implement sum_even_numbers(nums). Return the sum of the even values only.",
            "Write sum_even_numbers(nums) in Python. It should add only the even integers and ignore odd ones.",
        ],
        "tests": [
            t(args=[[1, 2, 3, 4, 5, 6]], expected=12),
            t(args=[[1, 3, 5]], expected=0),
            t(args=[[]], expected=0),
        ],
    })

    specs.append({
        "family": "normalize_spaces",
        "entrypoint": "normalize_spaces",
        "write_prompts": [
            "Write a Python function normalize_spaces(text) that trims leading and trailing whitespace and replaces repeated internal whitespace with a single space.",
            "Implement normalize_spaces(text). Strip surrounding whitespace and collapse internal whitespace to one space.",
            "Write normalize_spaces(text) in Python. It should strip outer whitespace and reduce repeated internal spaces to a single space.",
        ],
        "tests": [
            t(args=["  hello   world  "], expected="hello world"),
            t(args=["a\tb\nc"], expected="a b c"),
            t(args=["   "], expected=""),
        ],
    })

    return specs


def build_rows(n_target: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    specs = build_task_specs()
    rows: list[dict[str, Any]] = []
    counter = 1

    # First pass: one instance per wording variant.
    for spec in specs:
        for prompt in spec.get("write_prompts", []):
            rows.append(
                {
                    "id": f"code_write_{counter:06d}",
                    "task_type": "code_write",
                    "prompt": prompt,
                    "meta": {
                        "family": spec["family"],
                        "entrypoint": spec["entrypoint"],
                        "tests": copy.deepcopy(spec["tests"]),
                    },
                }
            )
            counter += 1

        if spec.get("fix_prompt"):
            rows.append(
                {
                    "id": f"code_fix_{counter:06d}",
                    "task_type": "code_fix",
                    "prompt": spec["fix_prompt"].format(buggy=spec["buggy"]),
                    "meta": {
                        "family": spec["family"],
                        "entrypoint": spec["entrypoint"],
                        "tests": copy.deepcopy(spec["tests"]),
                    },
                }
            )
            counter += 1

    # Expand by randomized paraphrase wrappers if a larger target is requested.
    wrappers = [
        "{base}",
        "{base} Output only one Python code block.",
        "{base} Use only the Python standard library.",
        "{base} Keep the solution simple and correct.",
        "{base} Do not include any explanation; output only code.",
        "{base} Handle edge cases correctly.",
    ]

    while len(rows) < n_target:
        spec = rng.choice(specs)
        base = rng.choice(spec.get("write_prompts", []))
        wrapped = rng.choice(wrappers).format(base=base)
        rows.append(
            {
                "id": f"code_write_{counter:06d}",
                "task_type": "code_write",
                "prompt": wrapped,
                "meta": {
                    "family": spec["family"],
                    "entrypoint": spec["entrypoint"],
                    "tests": copy.deepcopy(spec["tests"]),
                },
            }
        )
        counter += 1

        if len(rows) >= n_target:
            break

        if spec.get("fix_prompt") and rng.random() < 0.35:
            rows.append(
                {
                    "id": f"code_fix_{counter:06d}",
                    "task_type": "code_fix",
                    "prompt": spec["fix_prompt"].format(buggy=spec["buggy"]),
                    "meta": {
                        "family": spec["family"],
                        "entrypoint": spec["entrypoint"],
                        "tests": copy.deepcopy(spec["tests"]),
                    },
                }
            )
            counter += 1

    rng.shuffle(rows)
    return rows[:n_target]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_target", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rows = build_rows(args.n_target, args.seed)
    write_jsonl(args.out, rows)
    print(f"Wrote {len(rows)} prompts to {args.out}")


if __name__ == "__main__":
    main()
