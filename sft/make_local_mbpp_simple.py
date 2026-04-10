#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import random
from pathlib import Path

SYSTEM = "You are a helpful assistant."

TASKS = [
    ("reverse_string", "Write a Python function reverse_string(s) that returns the reversed string."),
    ("dedup_preserve_order", "Write a Python function dedup_preserve_order(items) that removes duplicates while keeping the first occurrence order."),
    ("count_words", "Write a Python function count_words(text) that returns a dictionary of lowercase word frequencies."),
    ("running_sum", "Write a Python function running_sum(nums) that returns cumulative sums."),
    ("running_max", "Write a Python function running_max(nums) that returns the running maximum values."),
    ("safe_divide", "Write a Python function safe_divide(a, b) that returns None if b is zero, otherwise a / b."),
    ("swap_first_last", "Write a Python function swap_first_last(items) that returns a new list with the first and last items swapped."),
    ("get_common_elements", "Write a Python function get_common_elements(a, b) that returns the elements of a that also appear in b, preserving a's order."),
    ("lowercase_keys", "Write a Python function lowercase_keys(d) that returns a new dictionary with lowercase string keys."),
    ("binary_search", "Write a Python function binary_search(nums, target) for a sorted list, returning the index of target or -1."),
    ("is_prime", "Write a Python function is_prime(n) that returns True if n is prime and False otherwise."),
    ("gcd", "Write a Python function gcd(a, b) using the Euclidean algorithm."),
    ("flatten_once", "Write a Python function flatten_once(items) that flattens one nesting level of lists."),
    ("chunk_list", "Write a Python function chunk_list(items, size) that splits a list into chunks of the given size."),
    ("extract_urls", "Write a Python function extract_urls(text) that returns all URLs starting with http:// or https://."),
    ("merge_counts", "Write a Python function merge_counts(a, b) that merges two dictionaries by summing values of shared keys."),
    ("filter_even", "Write a Python function filter_even(nums) that returns only the even numbers from a list."),
    ("title_case_words", "Write a Python function title_case_words(words) that returns a new list with each word title-cased."),
    ("group_by_length", "Write a Python function group_by_length(words) that returns a dictionary mapping word length to the list of words with that length."),
    ("sum_csv_column", "Write a Python function sum_csv_column(path, column_name) that uses csv.DictReader and returns the sum of one numeric column."),
]

SOLUTIONS = {
    "reverse_string": "def reverse_string(s):\n    return s[::-1]\n",
    "dedup_preserve_order": "def dedup_preserve_order(items):\n    seen = set()\n    out = []\n    for item in items:\n        if item not in seen:\n            seen.add(item)\n            out.append(item)\n    return out\n",
    "count_words": "import re\n\ndef count_words(text):\n    words = re.findall(r\"[A-Za-z0-9']+\", text.lower())\n    out = {}\n    for w in words:\n        out[w] = out.get(w, 0) + 1\n    return out\n",
    "running_sum": "def running_sum(nums):\n    total = 0\n    out = []\n    for x in nums:\n        total += x\n        out.append(total)\n    return out\n",
    "running_max": "def running_max(nums):\n    out = []\n    current = None\n    for x in nums:\n        if current is None or x > current:\n            current = x\n        out.append(current)\n    return out\n",
    "safe_divide": "def safe_divide(a, b):\n    return None if b == 0 else a / b\n",
    "swap_first_last": "def swap_first_last(items):\n    if len(items) < 2:\n        return list(items)\n    out = list(items)\n    out[0], out[-1] = out[-1], out[0]\n    return out\n",
    "get_common_elements": "def get_common_elements(a, b):\n    bset = set(b)\n    return [x for x in a if x in bset]\n",
    "lowercase_keys": "def lowercase_keys(d):\n    out = {}\n    for k, v in d.items():\n        out[k.lower() if isinstance(k, str) else k] = v\n    return out\n",
    "binary_search": "def binary_search(nums, target):\n    left, right = 0, len(nums) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if nums[mid] == target:\n            return mid\n        if nums[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n",
    "is_prime": "def is_prime(n):\n    if n < 2:\n        return False\n    i = 2\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 1\n    return True\n",
    "gcd": "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return abs(a)\n",
    "flatten_once": "def flatten_once(items):\n    out = []\n    for x in items:\n        if isinstance(x, list):\n            out.extend(x)\n        else:\n            out.append(x)\n    return out\n",
    "chunk_list": "def chunk_list(items, size):\n    if size <= 0:\n        raise ValueError('size must be positive')\n    return [items[i:i + size] for i in range(0, len(items), size)]\n",
    "extract_urls": "import re\n\ndef extract_urls(text):\n    return re.findall(r'https?://[^\\s]+', text)\n",
    "merge_counts": "def merge_counts(a, b):\n    out = dict(a)\n    for k, v in b.items():\n        out[k] = out.get(k, 0) + v\n    return out\n",
    "filter_even": "def filter_even(nums):\n    return [x for x in nums if x % 2 == 0]\n",
    "title_case_words": "def title_case_words(words):\n    return [w.title() for w in words]\n",
    "group_by_length": "def group_by_length(words):\n    out = {}\n    for w in words:\n        out.setdefault(len(w), []).append(w)\n    return out\n",
    "sum_csv_column": "import csv\n\ndef sum_csv_column(path, column_name):\n    total = 0.0\n    with open(path, newline='', encoding='utf-8') as f:\n        reader = csv.DictReader(f)\n        for row in reader:\n            value = row.get(column_name, '')\n            if value is None or str(value).strip() == '':\n                continue\n            total += float(value)\n    return total\n",
}

PROMPT_TEMPLATES = [
    "{task}\n\nReturn only a Python code block.",
    "Implement this in Python. Do not explain anything.\n\n{task}\n\nReturn only a Python code block.",
    "Solve this basic Python task. Reply with only one Python code block.\n\n{task}",
    "Write clear Python code for the following. Return only the code block.\n\n{task}",
]


def ast_ok(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--variants_per_problem", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=12)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, task in TASKS:
        code = SOLUTIONS[name]
        assert ast_ok(code)
        for _ in range(args.repeats):
            tmpls = PROMPT_TEMPLATES[:]
            rng.shuffle(tmpls)
            for tmpl in tmpls[: max(1, args.variants_per_problem)]:
                user = tmpl.format(task=task)
                rows.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": f"```python\n{code.strip()}\n```"},
                    ]
                })

    rng.shuffle(rows)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} examples -> {out_path}")


if __name__ == "__main__":
    main()
