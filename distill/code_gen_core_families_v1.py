from __future__ import annotations

import argparse
import random
from typing import Any, Dict, List

from code_utils import family_specs, stable_id, write_jsonl

PROMPTS = {
    "safe_divide": [
        "Write a Python function `safe_divide(a, b, default=0.0)`.\n\nReturn `a / b` when `b` is not zero.\nIf `b` is zero, return `default`.\n\nDo not use imports.\nReturn only Python code.",
        "Implement `safe_divide(a, b, default=0.0)` in Python. Return `a / b` unless `b` is zero, in which case return `default`. Return only Python code.",
        "Create a small helper function called `safe_divide(a, b, default=0.0)` that safely divides by `b`. If `b` is zero, return `default`. Return only Python code.",
    ],
    "running_sum": [
        "Implement a Python function `running_sum(nums)`.\n\nIt should return a new list where each element is the cumulative sum up to that position.\n\nExamples:\n[1, 2, 3] -> [1, 3, 6]\n[] -> []\n\nDo not modify the input list.\nReturn only Python code.",
        "Write a Python function `running_sum(nums)` that returns cumulative sums from left to right. Return only Python code.",
        "Solve this small Python task: define `running_sum(nums)` to return the running totals of the input list. Return only Python code.",
    ],
    "running_max": [
        "Write a Python function `running_max(nums)`.\n\nReturn a list where each position contains the maximum value seen so far from left to right.\n\nDo not use imports.\nReturn only Python code.",
        "Implement `running_max(nums)` in Python. It should return prefix maximums. Return only Python code.",
        "Create a Python function `running_max(nums)` that tracks the largest value seen so far at each step. Return only Python code.",
    ],
    "dedup_preserve_order": [
        "Create a Python function `dedup_preserve_order(items)`.\n\nReturn a list with duplicates removed while keeping the first occurrence of each item.\n\nDo not use imports.\nReturn only Python code.",
        "Implement `dedup_preserve_order(items)` in Python. Keep the first copy of each item and preserve the original order. Return only Python code.",
        "Write a small Python helper `dedup_preserve_order(items)` that removes duplicates without changing first-seen order. Return only Python code.",
    ],
    "count_words": [
        "Implement `count_words(text)` in Python.\n\nConvert the text to lowercase, split on whitespace, and return a dictionary mapping each word to its count.\n\nAssume the input text contains only letters and spaces.\nReturn only Python code.",
        "Write a Python function `count_words(text)` that lowercases the text, splits on whitespace, and counts words. Return only Python code.",
        "Define `count_words(text)` in Python. Use lowercase words and whitespace splitting only. Return only Python code.",
    ],
    "lowercase_keys": [
        "Write a Python function `lowercase_keys(d)`.\n\nReturn a new dictionary with all string keys converted to lowercase.\nKeep the values unchanged.\n\nIf two keys become the same after lowercasing, keep the later one.\n\nReturn only Python code.",
        "Implement `lowercase_keys(d)` in Python. Lowercase every string key and keep values unchanged. If keys collide after lowercasing, the later one should win. Return only Python code.",
        "Create a Python helper `lowercase_keys(d)` that returns a new dictionary with lowercase string keys. Return only Python code.",
    ],
    "flatten_once": [
        "Implement a Python function `flatten_once(items)`.\n\nIf an element is a list, extend the result with its elements.\nOtherwise, append the element as is.\n\nOnly flatten one level.\nReturn only Python code.",
        "Write `flatten_once(items)` in Python. Flatten only one list level and leave other values unchanged. Return only Python code.",
        "Create a small Python function `flatten_once(items)` that expands list elements one level deep. Return only Python code.",
    ],
    "reverse_string": [
        "Write a Python function `reverse_string(s)` that returns the reversed string.\n\nReturn only Python code.",
        "Implement `reverse_string(s)` in Python. It should return `s` reversed. Return only Python code.",
        "Define a Python function `reverse_string(s)` that returns the characters of the string in reverse order. Return only Python code.",
    ],
    "is_prime": [
        "Implement `is_prime(n)` in Python.\n\nReturn True if `n` is prime, otherwise return False.\nFor n < 2, return False.\n\nDo not use imports.\nReturn only Python code.",
        "Write a Python function `is_prime(n)` that checks whether an integer is prime. For numbers less than 2, return False. Return only Python code.",
        "Create a simple Python helper `is_prime(n)` for primality testing. Return only Python code.",
    ],
    "clamp": [
        "Write a Python function `clamp(x, low, high)`.\n\nReturn `low` if `x` is smaller than `low`, `high` if `x` is larger than `high`, otherwise return `x`.\nReturn only Python code.",
        "Implement `clamp(x, low, high)` in Python. Keep `x` inside the inclusive range `[low, high]`. Return only Python code.",
    ],
    "remove_none": [
        "Write a Python function `remove_none(items)`.\n\nReturn a new list that contains all items except `None`.\nReturn only Python code.",
        "Implement `remove_none(items)` in Python. Remove every `None` value and keep the remaining order unchanged. Return only Python code.",
    ],
    "merge_counts": [
        "Implement `merge_counts(a, b)` in Python.\n\nBoth inputs are dictionaries from strings to integers.\nReturn a new dictionary that adds the counts for matching keys.\nReturn only Python code.",
        "Write a Python function `merge_counts(a, b)` that combines two count dictionaries by summing values for the same key. Return only Python code.",
    ],
}

def expand_records(instances_per_family: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    specs = family_specs()
    rows: List[Dict[str, Any]] = []
    for family, spec in specs.items():
        prompts = PROMPTS[family]
        for i in range(instances_per_family):
            prompt = prompts[i % len(prompts)]
            rows.append({
                "id": stable_id("fam", family, str(i)),
                "source": "core_families_v1",
                "source_key": f"{family}_{i}",
                "family": family,
                "raw_prompt": prompt,
                "canonical_prompt": prompt,
                "entry_point": spec["entry_point"],
                "tests": spec["tests"],
                "meta": {"from_family_template": True, "family_index": i},
            })
    rng.shuffle(rows)
    return rows

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--instances_per_family", type=int, default=80)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    rows = expand_records(args.instances_per_family, args.seed)
    write_jsonl(args.out_jsonl, rows)
    print(f"Wrote {len(rows)} family prompts to {args.out_jsonl}")

if __name__ == "__main__":
    main()
