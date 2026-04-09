#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List

DEFAULT_SYSTEM = "You are a helpful assistant."

BUG_TEMPLATES: List[Dict[str, str]] = [
    {
        "name": "reverse_string",
        "user": "Fix this Python function so it returns the reversed string correctly. Return only a Python code block.",
        "broken": "def reverse_string(s):\n    out = ''\n    for ch in s:\n        out = out + ch\n    return out\n",
        "fixed": "def reverse_string(s):\n    return s[::-1]\n",
    },
    {
        "name": "dedup_preserve_order",
        "user": "Fix this Python function so it removes duplicates while preserving the first occurrence order. Return only a Python code block.",
        "broken": "def dedup_preserve_order(items):\n    return list(set(items))\n",
        "fixed": "def dedup_preserve_order(items):\n    seen = set()\n    out = []\n    for item in items:\n        if item not in seen:\n            seen.add(item)\n            out.append(item)\n    return out\n",
    },
    {
        "name": "count_words",
        "user": "Fix this Python function so it returns lowercase word frequencies and ignores commas and periods. Return only a Python code block.",
        "broken": "def count_words(text):\n    words = text.split()\n    return len(words)\n",
        "fixed": "import re\n\ndef count_words(text):\n    words = re.findall(r\"[A-Za-z0-9']+\", text.lower())\n    counts = {}\n    for word in words:\n        counts[word] = counts.get(word, 0) + 1\n    return counts\n",
    },
    {
        "name": "running_sum",
        "user": "Fix this Python function so it returns cumulative sums. Return only a Python code block.",
        "broken": "def running_sum(nums):\n    total = 0\n    out = []\n    for x in nums:\n        out.append(total)\n    return out\n",
        "fixed": "def running_sum(nums):\n    total = 0\n    out = []\n    for x in nums:\n        total += x\n        out.append(total)\n    return out\n",
    },
    {
        "name": "running_max",
        "user": "Fix this Python function so it returns the running maximum correctly. Return only a Python code block.",
        "broken": "def running_max(nums):\n    out = []\n    current = 0\n    for x in nums:\n        if x < current:\n            current = x\n        out.append(current)\n    return out\n",
        "fixed": "def running_max(nums):\n    out = []\n    current = None\n    for x in nums:\n        if current is None or x > current:\n            current = x\n        out.append(current)\n    return out\n",
    },
    {
        "name": "safe_divide",
        "user": "Fix this Python function so it returns None when the divisor is zero. Return only a Python code block.",
        "broken": "def safe_divide(a, b):\n    return a / b\n",
        "fixed": "def safe_divide(a, b):\n    return None if b == 0 else a / b\n",
    },
    {
        "name": "swap_first_last",
        "user": "Fix this Python function so it swaps the first and last items of a list. Return only a Python code block.",
        "broken": "def swap_first_last(items):\n    if len(items) < 2:\n        return items\n    items[0] = items[-1]\n    return items\n",
        "fixed": "def swap_first_last(items):\n    if len(items) < 2:\n        return items\n    items = list(items)\n    items[0], items[-1] = items[-1], items[0]\n    return items\n",
    },
    {
        "name": "get_common_elements",
        "user": "Fix this Python function so it returns the common elements of two lists while preserving the order from the first list. Return only a Python code block.",
        "broken": "def get_common_elements(a, b):\n    return list(set(a) & set(b))\n",
        "fixed": "def get_common_elements(a, b):\n    bset = set(b)\n    return [x for x in a if x in bset]\n",
    },
    {
        "name": "lowercase_keys",
        "user": "Fix this Python function so it returns a new dictionary with lowercase string keys. Return only a Python code block.",
        "broken": "def lowercase_keys(d):\n    for k in d:\n        d[k.lower()] = d[k]\n    return d\n",
        "fixed": "def lowercase_keys(d):\n    out = {}\n    for k, v in d.items():\n        out[k.lower() if isinstance(k, str) else k] = v\n    return out\n",
    },
    {
        "name": "binary_search",
        "user": "Fix this Python function so it performs binary search correctly on a sorted list and returns the index or -1. Return only a Python code block.",
        "broken": "def binary_search(nums, target):\n    left, right = 0, len(nums) - 1\n    while left < right:\n        mid = (left + right) // 2\n        if nums[mid] == target:\n            return mid\n        if nums[mid] < target:\n            right = mid - 1\n        else:\n            left = mid + 1\n    return -1\n",
        "fixed": "def binary_search(nums, target):\n    left, right = 0, len(nums) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if nums[mid] == target:\n            return mid\n        if nums[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n",
    },
    {
        "name": "is_prime",
        "user": "Fix this Python function so it correctly checks whether an integer is prime. Return only a Python code block.",
        "broken": "def is_prime(n):\n    if n < 2:\n        return True\n    for i in range(2, n):\n        if n % i == 0:\n            return False\n    return False\n",
        "fixed": "def is_prime(n):\n    if n < 2:\n        return False\n    i = 2\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 1\n    return True\n",
    },
    {
        "name": "gcd",
        "user": "Fix this Python function so it returns the greatest common divisor using the Euclidean algorithm. Return only a Python code block.",
        "broken": "def gcd(a, b):\n    while b:\n        a = b\n    return a\n",
        "fixed": "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return abs(a)\n",
    },
]

PROMPT_VARIANTS = [
    "Fix this Python code. Return only a Python code block.",
    "Repair the bug in this function. Return only the corrected Python code.",
    "Correct the function below. Return only a Python code block.",
]


def build_messages(user: str, broken: str, fixed: str) -> Dict[str, object]:
    full_user = f"{user}\n\n```python\n{broken.rstrip()}\n```"
    return {
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": full_user},
            {"role": "assistant", "content": f"```python\n{fixed.rstrip()}\n```"},
        ],
        "meta": {
            "bucket": "B_code",
            "dataset": "local_code_fix_simple",
            "split": "train",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--repeat", type=int, default=8, help="How many prompt variants to generate per bug family.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    rows: List[Dict[str, object]] = []
    for item in BUG_TEMPLATES:
        for _ in range(args.repeat):
            prefix = rng.choice(PROMPT_VARIANTS)
            user = prefix + "\n\n" + item["user"]
            rows.append(build_messages(user, item["broken"], item["fixed"]))

    rng.shuffle(rows)
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} examples -> {args.out_jsonl}")


if __name__ == "__main__":
    main()
