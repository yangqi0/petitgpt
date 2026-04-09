#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
from typing import Any, Dict, Optional

from datasets import load_dataset

DEFAULT_SYSTEM = "You are a helpful assistant."


def norm_newlines(s: str) -> str:
    return (s or "").replace("\r\n", "\n").replace("\r", "\n")


def prompt_to_instruction(prompt: str) -> str:
    p = norm_newlines(prompt).strip()
    p = re.sub(r"^Write a function\s+", "Write a Python function ", p, flags=re.I)
    p = re.sub(r"^Write a\s+", "Write a Python ", p, flags=re.I)
    if "Return only a Python code block." not in p:
        p = p.rstrip() + "\n\nReturn only a Python code block."
    return p


def ast_ok(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False


def normalize_code(code: str) -> str:
    return norm_newlines(code).replace("\t", "    ").strip()


def first_function_name(code: str) -> Optional[str]:
    try:
        tree = ast.parse(code)
    except Exception:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    return None


def keep_example(row: Dict[str, Any], max_prompt_chars: int, max_code_chars: int) -> bool:
    prompt = str(row.get("prompt") or row.get("text") or "")
    code = str(row.get("code") or "")
    if not prompt.strip() or not code.strip():
        return False
    if len(prompt.strip()) > max_prompt_chars or len(code.strip()) > max_code_chars:
        return False
    code = normalize_code(code)
    if not ast_ok(code):
        return False
    if not first_function_name(code):
        return False
    banned = ["matplotlib", "tensorflow", "torch", "pandas", "numpy"]
    low = (prompt + "\n" + code).lower()
    if any(b in low for b in banned):
        return False
    return True


def build_messages(prompt: str, code: str) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": prompt_to_instruction(prompt)},
            {"role": "assistant", "content": f"```python\n{code}\n```"},
        ],
        "meta": {
            "bucket": "B_code",
            "dataset": "local_mbpp_simple",
            "split": "train",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--subset", default="sanitized", choices=["sanitized", "full"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max_examples", type=int, default=2000)
    ap.add_argument("--max_prompt_chars", type=int, default=600)
    ap.add_argument("--max_code_chars", type=int, default=900)
    args = ap.parse_args()

    ds = load_dataset("Muennighoff/mbpp", args.subset, split=args.split)
    idxs = list(range(len(ds)))
    random.Random(args.seed).shuffle(idxs)

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    wrote = 0
    seen = set()
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for i in idxs:
            row = dict(ds[i])
            if not keep_example(row, args.max_prompt_chars, args.max_code_chars):
                continue
            prompt = str(row.get("prompt") or row.get("text") or "")
            code = normalize_code(str(row.get("code") or ""))
            sig = (prompt.strip().lower(), code.strip())
            if sig in seen:
                continue
            seen.add(sig)
            ex = build_messages(prompt, code)
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            wrote += 1
            if args.max_examples and wrote >= args.max_examples:
                break

    print(f"wrote {wrote} examples -> {args.out_jsonl}")


if __name__ == "__main__":
    main()
