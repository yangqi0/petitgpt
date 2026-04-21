from __future__ import annotations

import argparse
from typing import Any, Dict, List

from general_utils import openai_compatible_chat, read_jsonl, write_jsonl

SYSTEM_PROMPT = """You are writing training answers for a very small assistant.

Follow the user's format exactly.
Keep the answer short, clean, and direct.
Do not add explanations, notes, prefaces, code, markdown fences, placeholders, or multiple alternatives.
Do not mention the instructions.
If the task asks for bullets, return only bullets.
If the task asks for a rewrite, return only the rewritten text.
If the task asks for an email, write one short realistic email and nothing else.
If the task asks for a short explanation, use simple everyday language.
"""

def max_tokens_for_family(family: str) -> int:
    return {
        "rewrite_style": 60,
        "summary_bullets": 90,
        "explain_compare": 90,
        "email_message": 140,
    }.get(family, 120)

def raw_mode(rows: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        max_toks = max_tokens_for_family(row["family"])
        answer = openai_compatible_chat(
            api_base=args.api_base,
            model=args.model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=row["canonical_prompt"],
            temperature=args.temperature,
            max_tokens=max_toks,
        ).strip()
        out.append({
            "id": row["id"],
            "family": row["family"],
            "subfamily": row["subfamily"],
            "prompt": row["canonical_prompt"],
            "answer_raw": answer,
            "constraints": row["constraints"],
            "meta": {
                **row.get("meta", {}),
                "source": row["source"],
                "source_key": row.get("source_key"),
                "generation_round": 1,
                "temperature": args.temperature,
                "max_new_tokens": max_toks,
            },
        })
    return out

def repair_mode(rows: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        prompt = f"""The previous answer failed formatting checks.

Task:
{row['prompt']}

Previous answer:
{row['bad_answer']}

Problems:
""" + "\n".join(f"- {x}" for x in row["repair_reasons"]) + "\n\nRewrite the answer from scratch.\nReturn only the final answer."
        max_toks = max_tokens_for_family(row["family"])
        answer = openai_compatible_chat(
            api_base=args.api_base,
            model=args.model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=args.temperature,
            max_tokens=max_toks,
        ).strip()
        out.append({
            "id": row["id"],
            "family": row["family"],
            "subfamily": row["subfamily"],
            "prompt": row["prompt"],
            "answer_repaired": answer,
            "repair_from": row["bad_answer"],
            "repair_reasons": row["repair_reasons"],
            "constraints": row["constraints"],
            "meta": {
                **row.get("meta", {}),
                "generation_round": 2,
                "temperature": args.temperature,
                "max_new_tokens": max_toks,
            },
        })
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["raw", "repair"], required=True)
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--api_base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--temperature", type=float, default=0.3)
    args = ap.parse_args()

    rows = read_jsonl(args.in_jsonl)
    if args.mode == "raw":
        out = raw_mode(rows, args)
    else:
        out = repair_mode(rows, args)

    write_jsonl(args.out_jsonl, out)
    print(f"Wrote {len(out)} rows to {args.out_jsonl}")

if __name__ == "__main__":
    main()
