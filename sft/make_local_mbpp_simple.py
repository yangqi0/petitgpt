#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

SYSTEM = "You are a helpful assistant."

PROMPT_TEMPLATES = [
    "Write a Python function for this task. Return only one Python code block.\n\n{prompt}",
    "Solve this basic Python task. Reply with only one Python code block.\n\n{prompt}",
]


def first_function_name(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except Exception:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node.name
    return None


def safe_code_block(code: str) -> str | None:
    code = code.strip()
    try:
        ast.parse(code)
    except Exception:
        return None
    if not first_function_name(code):
        return None
    return f"```python\n{code}\n```"


def clean_prompt(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\n?assert .*", "", text)
    text = re.sub(r"\n?>>>.*", "", text)
    text = re.sub(r"\n?Examples?:.*", "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--subset", default="sanitized", choices=["sanitized", "full"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max_examples", type=int, default=2000)
    ap.add_argument("--variants_per_problem", type=int, default=2)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    ds = load_dataset("Muennighoff/mbpp", args.subset, split=args.split)
    rows = [dict(ds[i]) for i in range(len(ds))]
    rng.shuffle(rows)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    seen_pairs: set[tuple[str, str]] = set()

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            prompt = clean_prompt(str(row.get("text") or row.get("prompt") or ""))
            code = str(row.get("code") or row.get("canonical_solution") or "").strip()
            if not prompt or not code:
                continue
            code_block = safe_code_block(code)
            if code_block is None:
                continue

            raw_id = row.get("task_id") or row.get("problem_id") or row.get("id")
            if raw_id is None:
                raw_id = written
            base_id = f"mbpp_{raw_id}"

            tmpls = PROMPT_TEMPLATES[:]
            rng.shuffle(tmpls)
            tmpls = tmpls[: max(1, args.variants_per_problem)]
            for j, tmpl in enumerate(tmpls):
                user = tmpl.format(prompt=prompt)
                key = (base_id, user)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                ex = {
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": code_block},
                    ],
                    "meta": {
                        "dataset": "local_mbpp_simple",
                        "base_id": base_id,
                        "variant_id": f"{base_id}_v{j}",
                        "source": "mbpp",
                    },
                }
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                written += 1
                if written >= args.max_examples:
                    break
            if written >= args.max_examples:
                break

    print(f"wrote {written} examples -> {out_path}")


if __name__ == "__main__":
    main()
