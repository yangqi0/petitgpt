from __future__ import annotations

import argparse
from typing import Any, Dict, List

from code_utils import openai_compatible_chat, read_jsonl, write_jsonl

SYSTEM_PROMPT = """You are creating training answers for a very small Python assistant.

Write simple, beginner-friendly Python.
Prefer explicit loops and conditionals over clever tricks.
Do not use imports, classes, exceptions, decorators, recursion, or external libraries.
Define exactly one top-level function with the requested name.
Do not include tests, examples, explanations, markdown, or backticks.
Return only valid Python code.
"""

def max_tokens_for_row(row: Dict[str, Any]) -> int:
    return 180 if row.get("source") in {"mbpp", "apps_intro"} else 160

def raw_mode(rows: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        answer = openai_compatible_chat(
            api_base=args.api_base,
            model=args.model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=row["canonical_prompt"],
            temperature=args.temperature,
            max_tokens=max_tokens_for_row(row),
        ).strip()
        out.append({
            "id": row["id"],
            "source": row["source"],
            "source_key": row.get("source_key"),
            "family": row.get("family"),
            "prompt": row["canonical_prompt"],
            "answer_raw": answer,
            "entry_point": row["entry_point"],
            "tests": row["tests"],
            "meta": {**row.get("meta", {}), "generation_round": 1, "temperature": args.temperature},
        })
    return out

def repair_mode(rows: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        prompt = f"""Your previous solution failed verification.

Task:
{row['prompt']}

Failure summary:
""" + "\n".join(f"- {x}" for x in row["repair_reasons"]) + "\n\nRewrite the solution from scratch.\nReturn only valid Python code."
        answer = openai_compatible_chat(
            api_base=args.api_base,
            model=args.model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=args.temperature,
            max_tokens=180,
        ).strip()
        out.append({
            "id": row["id"],
            "source": row["source"],
            "source_key": row.get("source_key"),
            "family": row.get("family"),
            "prompt": row["prompt"],
            "answer_repaired": answer,
            "entry_point": row["entry_point"],
            "tests": row["tests"],
            "repair_reasons": row["repair_reasons"],
            "meta": {**row.get("meta", {}), "generation_round": 2, "temperature": args.temperature},
        })
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["raw", "repair"], required=True)
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--api_base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--temperature", type=float, default=0.15)
    args = ap.parse_args()

    rows = read_jsonl(args.in_jsonl)
    out = raw_mode(rows, args) if args.mode == "raw" else repair_mode(rows, args)
    write_jsonl(args.out_jsonl, out)
    print(f"Wrote {len(out)} rows to {args.out_jsonl}")

if __name__ == "__main__":
    main()
