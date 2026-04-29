from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def norm(s: str) -> str:
    return " ".join(s.lower().split())

BASE_RULES = [
    "Return only Python code.",
    "Do not use imports. Return only Python code.",
    "Use simple Python. Return only Python code.",
    "Define exactly one top-level function. Return only Python code.",
    "Prefer clear loops and conditionals. Return only Python code.",
    "Do not include examples, tests, or explanations in the answer. Return only Python code.",
]

TEMPLATES = [
    "Write a Python function `{signature}`. {desc} {edge} {example} {rules}",
    "Implement `{signature}` in Python. {desc} {edge} {rules}",
    "Create a small helper function called `{name}`. {desc} {edge} {example} {rules}",
    "Solve this short Python task by defining `{signature}`. {desc} {rules}",
    "Define `{signature}`. {desc} {edge} {rules}",
    "Write `{name}` as a Python function. {desc} {example} {rules}",
    "Implement the function `{signature}`. {desc} {edge} {rules}",
    "Create `{signature}` for this simple Python task. {desc} {example} {rules}",
    "Write a beginner-friendly implementation of `{signature}`. {desc} {edge} {rules}",
    "Define a function named `{name}` with the signature `{signature}`. {desc} {example} {rules}",
]

FAMILIES: Dict[str, Dict[str, Any]] = {
    "safe_divide": {
        "signature": "safe_divide(a, b, default=0.0)",
        "entry_point": "safe_divide",
        "descs": [
            "Return `a / b` when `b` is not zero, and return `default` when `b` is zero.",
            "Divide `a` by `b`, but avoid division by zero by returning `default` if `b == 0`.",
            "Return the quotient of `a` and `b`; if the denominator is zero, return the fallback value `default`.",
            "Use a simple conditional so that zero denominators return `default` instead of causing an error.",
            "Handle normal division and the zero-denominator case in a small, direct function.",
            "If `b` is zero, return `default`; otherwise return the result of `a / b`.",
        ],
        "edges": [
            "The default value should be used exactly as passed by the caller.",
            "Assume the inputs are numbers.",
            "Do not use exception handling for this task.",
            "Negative numbers should work naturally.",
            "The function should not print anything.",
            "Keep the behavior simple and deterministic.",
        ],
        "examples": [
            "For example, `safe_divide(6, 2)` should return `3.0` or `3`, and `safe_divide(1, 0)` should return `0.0`.",
            "For example, `safe_divide(5, 0, -1)` should return `-1`.",
            "For example, `safe_divide(-9, 3)` should return `-3.0` or `-3`.",
            "",
        ],
        "tests": [
            "assert safe_divide(6, 2) == 3",
            "assert safe_divide(1, 0) == 0.0",
            "assert safe_divide(1, 0, 7) == 7",
            "assert safe_divide(-9, 3) == -3",
            "assert abs(safe_divide(1.5, 0.5) - 3.0) < 1e-9",
        ],
    },
    "running_sum": {
        "signature": "running_sum(nums)",
        "entry_point": "running_sum",
        "descs": [
            "Return a new list where each element is the cumulative sum up to that position.",
            "Build the prefix sums of `nums` from left to right and return them as a list.",
            "For each position, store the sum of all values seen so far.",
            "Return the running total after each element in the input list.",
            "Compute cumulative sums without modifying the input list.",
            "Transform a list of numbers into a list of prefix sums.",
        ],
        "edges": [
            "An empty input list should return an empty list.",
            "The input may contain negative numbers.",
            "Do not modify `nums`.",
            "A single-element list should return a single-element list.",
            "Use a simple loop.",
            "The output should have the same length as the input.",
        ],
        "examples": [
            "For example, `[1, 2, 3]` should become `[1, 3, 6]`.",
            "For example, `[3, -1, 2]` should become `[3, 2, 4]`.",
            "For example, `[]` should return `[]`.",
            "",
        ],
        "tests": [
            "assert running_sum([]) == []",
            "assert running_sum([1]) == [1]",
            "assert running_sum([1, 2, 3]) == [1, 3, 6]",
            "assert running_sum([3, -1, 2]) == [3, 2, 4]",
            "assert running_sum([0, 0, 0]) == [0, 0, 0]",
        ],
    },
    "running_max": {
        "signature": "running_max(nums)",
        "entry_point": "running_max",
        "descs": [
            "Return a list where each element is the maximum value seen so far from left to right.",
            "Compute prefix maximums for the input list.",
            "For every position, store the largest value encountered up to that point.",
            "Return the running maximum after each number in `nums`.",
            "Build a new list of cumulative maximum values.",
            "Scan the list from left to right and track the largest value so far.",
        ],
        "edges": [
            "An empty list should return an empty list.",
            "The input may contain negative numbers.",
            "Do not modify the input list.",
            "The output list should have the same length as the input.",
            "Use simple Python control flow.",
            "A single value should return a one-item list.",
        ],
        "examples": [
            "For example, `[1, 3, 2, 5]` should become `[1, 3, 3, 5]`.",
            "For example, `[-2, -5, -1]` should become `[-2, -2, -1]`.",
            "For example, `[]` should return `[]`.",
            "",
        ],
        "tests": [
            "assert running_max([]) == []",
            "assert running_max([1]) == [1]",
            "assert running_max([1, 3, 2, 5]) == [1, 3, 3, 5]",
            "assert running_max([-2, -5, -1]) == [-2, -2, -1]",
        ],
    },
    "dedup_preserve_order": {
        "signature": "dedup_preserve_order(items)",
        "entry_point": "dedup_preserve_order",
        "descs": [
            "Return a new list with duplicate items removed while keeping the first occurrence of each item.",
            "Remove repeated values but preserve the original order of the first appearances.",
            "Keep only the first time each item appears in the list.",
            "Deduplicate the list without changing the order of the remaining elements.",
            "Return the unique items in their original first-seen order.",
            "Scan from left to right and keep items that have not appeared before.",
        ],
        "edges": [
            "Assume all items are hashable.",
            "An empty list should return an empty list.",
            "Do not modify the input list.",
            "Repeated strings and repeated integers should both work.",
            "Use a simple loop and a set of seen values.",
            "The returned list should contain the original values, not converted strings.",
        ],
        "examples": [
            "For example, `[1, 2, 1, 3, 2]` should return `[1, 2, 3]`.",
            "For example, `['a', 'b', 'a']` should return `['a', 'b']`.",
            "For example, `[1, 1, 1]` should return `[1]`.",
            "",
        ],
        "tests": [
            "assert dedup_preserve_order([]) == []",
            "assert dedup_preserve_order([1, 2, 1, 3, 2]) == [1, 2, 3]",
            "assert dedup_preserve_order(['a', 'b', 'a']) == ['a', 'b']",
            "assert dedup_preserve_order([1, 1, 1]) == [1]",
        ],
    },
    "count_words": {
        "signature": "count_words(text)",
        "entry_point": "count_words",
        "descs": [
            "Convert the text to lowercase, split on whitespace, and return a dictionary mapping each word to its count.",
            "Count how many times each lowercase word appears after splitting the text by whitespace.",
            "Return word frequencies using lowercase words as keys.",
            "Build a dictionary of word counts from the input string.",
            "Use `lower()` and whitespace splitting to count words.",
            "Return an empty dictionary for text with no words.",
        ],
        "edges": [
            "Assume the text contains only letters and spaces.",
            "Multiple spaces should be handled naturally by `split()`.",
            "Case should not matter.",
            "The function should not remove punctuation beyond the stated assumption.",
            "Use a normal dictionary.",
            "Do not print anything.",
        ],
        "examples": [
            "For example, `'a a b'` should return `{'a': 2, 'b': 1}`.",
            "For example, `'One one TWO'` should return `{'one': 2, 'two': 1}`.",
            "For example, an empty string should return `{}`.",
            "",
        ],
        "tests": [
            "assert count_words('') == {}",
            "assert count_words('a') == {'a': 1}",
            "assert count_words('a a b') == {'a': 2, 'b': 1}",
            "assert count_words('One one TWO') == {'one': 2, 'two': 1}",
            "assert count_words('a   b   a') == {'a': 2, 'b': 1}",
        ],
    },
    "lowercase_keys": {
        "signature": "lowercase_keys(d)",
        "entry_point": "lowercase_keys",
        "descs": [
            "Return a new dictionary with all string keys converted to lowercase and the values unchanged.",
            "Make a shallow copy of the dictionary where every key is lowercased.",
            "Convert each key with `.lower()` and keep the corresponding value.",
            "Return a dictionary whose keys are lowercase versions of the input keys.",
            "Lowercase all keys without changing the values.",
            "Build a new dictionary instead of modifying the input dictionary.",
        ],
        "edges": [
            "If two keys become the same after lowercasing, the later value should overwrite the earlier one.",
            "Assume all keys are strings.",
            "The function should be shallow; do not modify nested values.",
            "An empty dictionary should return an empty dictionary.",
            "Keep the values exactly as they are.",
            "Use simple dictionary assignment.",
        ],
        "examples": [
            "For example, `{'A': 1}` should return `{'a': 1}`.",
            "For example, `{'A': 1, 'a': 3}` should return `{'a': 3}`.",
            "For example, `{}` should return `{}`.",
            "",
        ],
        "tests": [
            "assert lowercase_keys({}) == {}",
            "assert lowercase_keys({'A': 1}) == {'a': 1}",
            "assert lowercase_keys({'A': 1, 'b': 2}) == {'a': 1, 'b': 2}",
            "assert lowercase_keys({'A': 1, 'a': 3}) == {'a': 3}",
        ],
    },
    "flatten_once": {
        "signature": "flatten_once(items)",
        "entry_point": "flatten_once",
        "descs": [
            "Return a new list where elements that are lists are expanded by one level.",
            "Flatten only one list level and leave non-list values unchanged.",
            "If an element is a list, extend the result with its elements; otherwise append it as is.",
            "Perform a one-level flatten operation on a mixed list.",
            "Expand nested list elements only once.",
            "Return a shallow flattened version of the input list.",
        ],
        "edges": [
            "Do not recursively flatten deeper lists.",
            "Only flatten values that are lists, not tuples or strings.",
            "An empty list should return an empty list.",
            "Empty inner lists should simply add nothing.",
            "Do not modify the input list.",
            "Use a simple loop.",
        ],
        "examples": [
            "For example, `[1, [2, 3], 4]` should return `[1, 2, 3, 4]`.",
            "For example, `[1, [2, [3]], 4]` should return `[1, 2, [3], 4]`.",
            "For example, `[1, [], 2]` should return `[1, 2]`.",
            "",
        ],
        "tests": [
            "assert flatten_once([]) == []",
            "assert flatten_once([1, [2, 3], 4]) == [1, 2, 3, 4]",
            "assert flatten_once([[1], [2], [3]]) == [1, 2, 3]",
            "assert flatten_once([1, [], 2]) == [1, 2]",
            "assert flatten_once([1, [2, [3]], 4]) == [1, 2, [3], 4]",
        ],
    },
    "reverse_string": {
        "signature": "reverse_string(s)",
        "entry_point": "reverse_string",
        "descs": [
            "Return the input string reversed.",
            "Create and return a reversed version of `s`.",
            "Reverse the characters of the string and return the result.",
            "Return a new string with the characters in opposite order.",
            "Implement a simple string reversal function.",
            "Return `s` from end to beginning.",
        ],
        "edges": [
            "An empty string should return an empty string.",
            "A one-character string should return itself.",
            "Spaces should be reversed like any other character.",
            "Unicode characters should work naturally.",
            "Do not print anything.",
            "Keep the implementation simple.",
        ],
        "examples": [
            "For example, `'abc'` should return `'cba'`.",
            "For example, `'ab cd'` should return `'dc ba'`.",
            "For example, `'你好'` should return `'好你'`.",
            "",
        ],
        "tests": [
            "assert reverse_string('') == ''",
            "assert reverse_string('a') == 'a'",
            "assert reverse_string('abc') == 'cba'",
            "assert reverse_string('ab cd') == 'dc ba'",
            "assert reverse_string('你好') == '好你'",
        ],
    },
    "is_prime": {
        "signature": "is_prime(n)",
        "entry_point": "is_prime",
        "descs": [
            "Return `True` if `n` is prime, otherwise return `False`.",
            "Check whether the integer `n` is a prime number.",
            "Return a boolean indicating whether `n` has exactly two positive divisors.",
            "Implement a simple primality test for an integer.",
            "For numbers less than 2, return `False`; otherwise test divisibility.",
            "Determine whether `n` is prime using clear Python code.",
        ],
        "edges": [
            "For `n < 2`, return `False`.",
            "The input is an integer.",
            "Do not use imports.",
            "It is fine to use a simple loop up to the square root or equivalent.",
            "Small primes such as 2 and 3 should return `True`.",
            "Composite squares such as 49 should return `False`.",
        ],
        "examples": [
            "For example, `is_prime(2)` should return `True`, and `is_prime(4)` should return `False`.",
            "For example, `is_prime(29)` should return `True`.",
            "For example, `is_prime(1)` should return `False`.",
            "",
        ],
        "tests": [
            "assert is_prime(-3) is False",
            "assert is_prime(0) is False",
            "assert is_prime(1) is False",
            "assert is_prime(2) is True",
            "assert is_prime(3) is True",
            "assert is_prime(4) is False",
            "assert is_prime(29) is True",
            "assert is_prime(49) is False",
        ],
    },
    "merge_counts": {
        "signature": "merge_counts(a, b)",
        "entry_point": "merge_counts",
        "descs": [
            "Return a new dictionary that combines two count dictionaries by summing values for matching keys.",
            "Merge two dictionaries from strings to integers, adding counts when a key appears in both.",
            "Create a combined count dictionary from `a` and `b`.",
            "Copy the counts from `a`, then add the counts from `b`.",
            "Return the total counts for all keys appearing in either input dictionary.",
            "Do not modify either input dictionary while combining their counts.",
        ],
        "edges": [
            "Both inputs are dictionaries from strings to integers.",
            "Keys that appear in only one dictionary should keep their count.",
            "If a key appears in both dictionaries, add the two values.",
            "An empty dictionary should work normally.",
            "Return a new dictionary.",
            "Use simple dictionary operations.",
        ],
        "examples": [
            "For example, `{'a': 1}` and `{'a': 2, 'b': 3}` should return `{'a': 3, 'b': 3}`.",
            "For example, `{}` and `{'x': 4}` should return `{'x': 4}`.",
            "",
        ],
        "tests": [
            "assert merge_counts({}, {}) == {}",
            "assert merge_counts({'a': 1}, {}) == {'a': 1}",
            "assert merge_counts({}, {'x': 4}) == {'x': 4}",
            "assert merge_counts({'a': 1}, {'a': 2, 'b': 3}) == {'a': 3, 'b': 3}",
        ],
    },
    "clamp": {
        "signature": "clamp(x, low, high)",
        "entry_point": "clamp",
        "descs": [
            "Return `low` if `x` is smaller than `low`, `high` if `x` is larger than `high`, otherwise return `x`.",
            "Keep `x` inside the inclusive range `[low, high]`.",
            "Limit a value so it cannot go below `low` or above `high`.",
            "Return the nearest boundary when `x` is outside the allowed range.",
            "Implement a simple clamp function.",
            "Return `x` unchanged when it is already between `low` and `high`.",
        ],
        "edges": [
            "Assume `low <= high`.",
            "The range is inclusive.",
            "Values equal to the boundaries should return unchanged.",
            "The inputs are numbers.",
            "Use simple conditionals.",
            "Do not print anything.",
        ],
        "examples": [
            "For example, `clamp(5, 1, 10)` should return `5`.",
            "For example, `clamp(-2, 0, 10)` should return `0`.",
            "For example, `clamp(12, 0, 10)` should return `10`.",
            "",
        ],
        "tests": [
            "assert clamp(5, 1, 10) == 5",
            "assert clamp(-2, 0, 10) == 0",
            "assert clamp(12, 0, 10) == 10",
            "assert clamp(0, 0, 10) == 0",
            "assert clamp(10, 0, 10) == 10",
        ],
    },
    "remove_none": {
        "signature": "remove_none(items)",
        "entry_point": "remove_none",
        "descs": [
            "Return a new list with every `None` value removed.",
            "Remove `None` entries while keeping all other items in their original order.",
            "Filter out values that are exactly `None`.",
            "Keep the order of the remaining items unchanged.",
            "Build a list containing all items except `None`.",
            "Do not remove falsey values such as `0`, `False`, or an empty string unless they are actually `None`.",
        ],
        "edges": [
            "An empty list should return an empty list.",
            "Do not modify the input list.",
            "Only remove values that are `None`.",
            "The function should keep `0`, `False`, and `''`.",
            "Use a simple loop or list comprehension.",
            "Return a new list.",
        ],
        "examples": [
            "For example, `[1, None, 2]` should return `[1, 2]`.",
            "For example, `[None, None]` should return `[]`.",
            "For example, `[0, None, False, '']` should return `[0, False, '']`.",
            "",
        ],
        "tests": [
            "assert remove_none([]) == []",
            "assert remove_none([1, None, 2]) == [1, 2]",
            "assert remove_none([None, None]) == []",
            "assert remove_none([0, None, False, '']) == [0, False, '']",
        ],
    },
}

def build_prompt(fam: str, spec: Dict[str, Any], template: str, desc: str, edge: str, example: str, rules: str) -> str:
    prompt = template.format(
        signature=spec["signature"],
        name=spec["entry_point"],
        desc=desc.strip(),
        edge=edge.strip(),
        example=example.strip(),
        rules=rules.strip(),
    )
    prompt = " ".join(prompt.split())
    return prompt.strip()

def generate_for_family(fam: str, spec: Dict[str, Any], per_family: int, rng: random.Random) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    attempts = 0

    combos = []
    for template in TEMPLATES:
        for desc in spec["descs"]:
            for edge in spec["edges"]:
                for example in spec["examples"]:
                    for rules in BASE_RULES:
                        combos.append((template, desc, edge, example, rules))
    rng.shuffle(combos)

    for template, desc, edge, example, rules in combos:
        attempts += 1
        prompt = build_prompt(fam, spec, template, desc, edge, example, rules)
        key = norm(prompt)
        if key in seen:
            continue
        seen.add(key)
        idx = len(rows)
        rows.append({
            "id": f"corev2_{fam}_{idx:04d}",
            "source": "core_families_v2",
            "family": fam,
            "subfamily": fam,
            "entry_point": spec["entry_point"],
            "canonical_prompt": prompt,
            "tests": spec["tests"],
            "test_list": spec["tests"],
            "constraints": {
                "single_top_level_function": True,
                "no_imports": True,
                "no_classes": True,
                "no_explanations": True,
                "return_only_python_code": True,
                "max_lines": 30,
            },
            "meta": {
                "variant_index": idx,
                "signature": spec["signature"],
                "generator_version": "core_family_prompts_v2",
            },
        })
        if len(rows) >= per_family:
            break

    if len(rows) < per_family:
        raise RuntimeError(f"Only generated {len(rows)} unique prompts for {fam}, requested {per_family}")

    return rows

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--per_family", type=int, default=60)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows: List[Dict[str, Any]] = []

    for fam, spec in FAMILIES.items():
        rows.extend(generate_for_family(fam, spec, args.per_family, rng))

    write_jsonl(args.out_jsonl, rows)

    print(f"Wrote {len(rows)} rows -> {args.out_jsonl}")
    print(f"Families: {len(FAMILIES)}")
    print(f"per_family: {args.per_family}")
    for fam in FAMILIES:
        print(f"  {fam}: {sum(1 for r in rows if r['family'] == fam)}")

if __name__ == "__main__":
    main()
