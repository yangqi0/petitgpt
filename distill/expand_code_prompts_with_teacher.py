#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Use an OpenAI-compatible teacher endpoint to expand code prompts, while
preserving hidden tests and entrypoints for later semantic verification.

This is intentionally hybrid:
- prompt wording is teacher-generated
- correctness is still enforced by known task families + hidden tests
"""

from __future__ import annotations

import argparse
import copy
import json
import copy
import os
import random
import re
from typing import Any

from openai import OpenAI


DEFAULT_SYSTEM = "You are a helpful assistant."


def t(args=None, kwargs=None, expected=None, raises=None):
    return {
        "args": [] if args is None else args,
        "kwargs": {} if kwargs is None else kwargs,
        "expected": expected,
        "raises": raises,
    }


def task_specs() -> list[dict[str, Any]]:
    return [
        {
            "family": "fib_iterative",
            "entrypoint": "fib",
            "style": "write",
            "spec_text": "Write a Python function fib(n) that returns the n-th Fibonacci number iteratively. Raise ValueError if n < 0.",
            "tests": [t(args=[0], expected=0), t(args=[1], expected=1), t(args=[7], expected=13), t(args=[-1], raises="ValueError")],
        },
        {
            "family": "factorial",
            "entrypoint": "factorial",
            "style": "write",
            "spec_text": "Write a Python function factorial(n) that returns n!. Raise ValueError if n < 0.",
            "tests": [t(args=[0], expected=1), t(args=[5], expected=120), t(args=[-3], raises="ValueError")],
        },
        {
            "family": "reverse_string",
            "entrypoint": "reverse_string",
            "style": "fix",
            "spec_text": "Fix a Python function reverse_string(s) so that it returns the reversed string correctly.",
            "buggy": "def reverse_string(s):\n    out = ''\n    for ch in s:\n        out = out + ch\n    return out\n",
            "tests": [t(args=["abc"], expected="cba"), t(args=[""], expected=""), t(args=["A b"], expected="b A")],
        },
        {
            "family": "dedup_preserve_order",
            "entrypoint": "dedup_preserve_order",
            "style": "write",
            "spec_text": "Write a Python function dedup_preserve_order(items) that removes duplicates while keeping the first occurrence order.",
            "tests": [t(args=[[1, 2, 1, 3, 2]], expected=[1, 2, 3]), t(args=[[]], expected=[])],
        },
        {
            "family": "count_words",
            "entrypoint": "count_words",
            "style": "write",
            "spec_text": "Write a Python function count_words(text) that returns a dictionary of lowercase word frequencies. Ignore punctuation like commas and periods.",
            "tests": [t(args=["Hi, hi."], expected={"hi": 2}), t(args=["One fish, two fish."], expected={"one": 1, "fish": 2, "two": 1}), t(args=[""], expected={})],
        },
        {
            "family": "binary_search",
            "entrypoint": "binary_search",
            "style": "write",
            "spec_text": "Write a Python function binary_search(nums, target) that returns the index of target in a sorted list, or -1 if not found.",
            "tests": [t(args=[[1, 3, 5, 7], 1], expected=0), t(args=[[1, 3, 5, 7], 7], expected=3), t(args=[[1, 3, 5, 7], 4], expected=-1), t(args=[[], 4], expected=-1)],
        },
        {
            "family": "chunk_list",
            "entrypoint": "chunk_list",
            "style": "write",
            "spec_text": "Write a Python function chunk_list(items, size) that splits a list into chunks of length size. Raise ValueError if size <= 0.",
            "tests": [t(args=[[1, 2, 3, 4, 5], 2], expected=[[1, 2], [3, 4], [5]]), t(args=[[], 3], expected=[]), t(args=[[1, 2], 0], raises="ValueError")],
        },
        {
            "family": "merge_sorted_lists",
            "entrypoint": "merge_sorted_lists",
            "style": "write",
            "spec_text": "Write a Python function merge_sorted_lists(a, b) that merges two sorted lists into one sorted list.",
            "tests": [t(args=[[1, 3, 5], [2, 4, 6]], expected=[1, 2, 3, 4, 5, 6]), t(args=[[], [1, 2]], expected=[1, 2])],
        },
        {
            "family": "safe_divide",
            "entrypoint": "safe_divide",
            "style": "write",
            "spec_text": "Write a Python function safe_divide(a, b) that returns a / b, but returns None if b == 0.",
            "tests": [t(args=[8, 2], expected=4.0), t(args=[7, 0], expected=None), t(args=[5, 2], expected=2.5)],
        },
        {
            "family": "normalize_spaces",
            "entrypoint": "normalize_spaces",
            "style": "write",
            "spec_text": "Write a Python function normalize_spaces(text) that trims leading and trailing whitespace and replaces repeated internal whitespace with a single space.",
            "tests": [t(args=["  hello   world  "], expected="hello world"), t(args=["a\tb\nc"], expected="a b c"), t(args=["   "], expected="")],
        },
    ]


def parse_single_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


def normalize_prompt(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def read_existing_prompts(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                out.add(normalize_prompt(row.get("prompt", "")))
            except Exception:
                continue
    return out


def build_request(spec: dict[str, Any]) -> str:
    if spec["style"] == "write":
        return (
            "Create ONE new user prompt for a simple Python coding task.\n"
            "Constraints:\n"
            "- Keep it short and clear.\n"
            "- Preserve the exact required function name.\n"
            "- Preserve the core behavior and edge cases.\n"
            "- Do not include any answer.\n"
            "- Do not include markdown.\n"
            "Return exactly one JSON object with keys: task_type, prompt.\n"
            'Set task_type to "code_write".\n'
            f"Required function name: {spec['entrypoint']}\n"
            f"Core specification: {spec['spec_text']}\n"
        )

    if spec["style"] == "fix":
        return (
            "Create ONE new user prompt for a Python bug-fixing task.\n"
            "Constraints:\n"
            "- Keep it short and clear.\n"
            "- Preserve the exact required function name.\n"
            "- Include the buggy code inside the prompt.\n"
            "- Do not include any answer.\n"
            "- Do not include markdown outside the buggy code if not needed.\n"
            "Return exactly one JSON object with keys: task_type, prompt.\n"
            'Set task_type to "code_fix".\n'
            f"Required function name: {spec['entrypoint']}\n"
            f"Core specification: {spec['spec_text']}\n"
            f"Buggy code:\n{spec['buggy']}\n"
        )

    raise ValueError(spec["style"])


def validate_row(row: dict[str, Any], allowed_task_types: set[str]) -> bool:
    if not isinstance(row, dict):
        return False
    if row.get("task_type") not in allowed_task_types:
        return False
    prompt = row.get("prompt")
    if not isinstance(prompt, str) or len(prompt.strip()) < 20:
        return False
    if "```python" in prompt and "def " not in prompt:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_target", type=int, default=3000)
    ap.add_argument("--base_url", required=True)
    ap.add_argument("--api_key", default="EMPTY")
    ap.add_argument("--model", required=True)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=260)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max_attempts", type=int, default=100000)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    rng = random.Random(args.seed)
    specs = task_specs()

    seen = read_existing_prompts(args.out)
    rows: list[dict[str, Any]] = []

    attempts = 0
    while len(rows) < args.n_target and attempts < args.max_attempts:
        attempts += 1
        spec = copy.deepcopy(rng.choice(specs))
        req = build_request(spec)

        resp = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": DEFAULT_SYSTEM + "\n\nReturn exactly one JSON object and nothing else."},
                {"role": "user", "content": req},
            ],
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
        )
        text = resp.choices[0].message.content.strip()
        obj = parse_single_json_object(text)
        if obj is None or not validate_row(obj, {"code_write", "code_fix"}):
            continue

        norm = normalize_prompt(obj["prompt"])
        if norm in seen:
            continue

        row = {
            "id": f"{obj['task_type']}_{len(rows)+1:06d}",
            "task_type": obj["task_type"],
            "prompt": obj["prompt"].strip(),
            "meta": {
                "family": spec["family"],
                "entrypoint": spec["entrypoint"],
                "tests": spec["tests"],
            },
        }
        rows.append(row)
        seen.add(norm)

        if len(rows) % 100 == 0:
            print(f"[progress] accepted={len(rows)} attempts={attempts}")

    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} prompts to {args.out}")
    print(f"Attempts: {attempts}")


if __name__ == "__main__":
    main()
