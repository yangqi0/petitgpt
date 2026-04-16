#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a small local canonical JSONL from MBPP without relying on a dataset loading script.
Compatible with datasets>=4 where dataset scripts are no longer supported.

Output rows look like:
  {"messages": [...], "meta": {...}}
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from typing import Any, Dict, List

from datasets import load_dataset
from huggingface_hub import hf_hub_download

# def load_mbpp_subset_local(subset: str):
#     if subset == "sanitized":
#         filename = "sanitized/mbpp-test.parquet"
#     elif subset == "full":
#         filename = "full/mbpp-test.parquet"
#     else:
#         raise ValueError(f"unsupported subset: {subset}")

#     local_path = hf_hub_download(
#         repo_id="Muennighoff/mbpp",
#         repo_type="dataset",
#         filename=filename,
#         revision="refs/convert/parquet",
#     )
#     return load_dataset("parquet", data_files={"test": local_path}, split="test")

def load_mbpp_subset_local(subset: str):
    return load_dataset("google-research-datasets/mbpp", subset, split="test")

PROMPT_TEMPLATES = [
    "Write a Python function for this task. Return only one Python code block.\n\n{prompt}",
    "Solve this basic Python task. Reply with only one Python code block.\n\n{prompt}",
]

PARQUET_URLS = {
    "sanitized": "https://huggingface.co/datasets/Muennighoff/mbpp/resolve/main/sanitized/test/0000.parquet",
    "full": "https://huggingface.co/datasets/Muennighoff/mbpp/resolve/main/full/test/0000.parquet",
}


def normalize_code(code: str) -> str:
    code = code.replace("\r\n", "\n").strip()
    return code


def only_one_code_block(code: str) -> str:
    return f"```python\n{code.rstrip()}\n```"


def pick_prompt(ex: Dict[str, Any]) -> str:
    prompt = ex.get("prompt") or ex.get("text") or ""
    prompt = str(prompt).strip()
    return prompt


def maybe_skip(ex: Dict[str, Any]) -> bool:
    prompt = pick_prompt(ex).lower()
    code = str(ex.get("code", ""))
    if not prompt or not code.strip():
        return True
    # keep it basic/compact
    banned = [
        "class ", "django", "flask", "tensorflow", "pytorch", "plotly", "matplotlib",
        "beautifulsoup", "requests", "sqlalchemy", "asyncio", "socket", "subprocess",
    ]
    if any(x in prompt for x in banned):
        return True
    if len(code.splitlines()) > 40:
        return True
    return False


def build_rows(ds, seed: int, max_examples: int, variants_per_problem: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    items = list(ds)
    rng.shuffle(items)

    for ex in items:
        if maybe_skip(ex):
            continue
        prompt = pick_prompt(ex)
        code = normalize_code(str(ex["code"]))
        task_id = ex.get("task_id")
        templates = PROMPT_TEMPLATES[: max(1, min(variants_per_problem, len(PROMPT_TEMPLATES)))]
        for i, tmpl in enumerate(templates):
            user = tmpl.format(prompt=prompt)
            row = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": only_one_code_block(code)},
                ],
                "meta": {
                    "source": "local_mbpp_simple",
                    "task_id": task_id,
                    "base_id": f"mbpp::{task_id}",
                    "variant_id": i,
                    "subset": "sanitized",
                },
            }
            rows.append(row)
            if len(rows) >= max_examples:
                return rows
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--subset", choices=["sanitized", "full"], default="sanitized")
    ap.add_argument("--split", default="test", help="Kept for CLI compatibility; MBPP only has test here.")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max_examples", type=int, default=2000)
    ap.add_argument("--variants_per_problem", type=int, default=2)
    args = ap.parse_args()

    # url = PARQUET_URLS[args.subset]
    # if args.subset == "sanitized":
    #     url = "https://huggingface.co/datasets/Muennighoff/mbpp/resolve/main/sanitized/test/0000.parquet"
    # elif args.subset == "full":
    #     url = "https://huggingface.co/datasets/Muennighoff/mbpp/resolve/main/full/test/0000.parquet"
    # else:
    #     raise ValueError(f"unsupported subset: {args.subset}")

    # ds = load_dataset("parquet", data_files={"test": url}, split="test")
    ds = load_mbpp_subset_local(args.subset)
    rows = build_rows(ds, seed=args.seed, max_examples=args.max_examples, variants_per_problem=args.variants_per_problem)
    if not rows:
        raise SystemExit("no rows produced")
    write_jsonl(args.out_jsonl, rows)
    print(f"loaded={len(ds)} examples from MBPP dataset")
    print(f"wrote={len(rows)} -> {args.out_jsonl}")


if __name__ == "__main__":
    main()
