from __future__ import annotations

import argparse
from typing import Any, Dict, List

from general_utils import openai_compatible_chat, read_jsonl, write_jsonl

BASE_SYSTEM = """You are writing training answers for a very small assistant.

Follow the user's task exactly.
Keep the answer short, clean, and direct.
Do not add explanations, notes, prefaces, code, markdown fences, placeholders, or multiple alternatives.
Do not mention the instructions.
"""

FAMILY_RULES = {
    "rewrite_style": """For rewrite tasks:
- Return only the rewritten text.
- Use one short paragraph.
- Do not use bullets, numbering, headings, or explanations.
- Do not say "Here is" or "Rewritten text".
""",
    "summary_bullets": """For summary tasks:
- Return exactly the requested number of bullet points.
- Start each bullet with "- ".
- Do not add an introduction, conclusion, heading, or extra paragraph.
- Keep each bullet short.
""",
    "email_message": """For email/message tasks:
- Return exactly one short realistic email or message.
- If the task asks for an email, include a greeting and a sign-off.
- If the task asks for a short message/text/note, keep it short and do not force a formal email.
- Do not use placeholders such as [Name] or [Company].
- Do not use bullets.
""",
    "explain_compare": """For explanation/comparison tasks:
- Return a short paragraph of 2 or 3 sentences.
- Do not use bullets, numbered lists, headings, markdown, or examples unless the task explicitly asks.
- Use simple everyday language.
""",
}

def max_tokens_for_family(family: str) -> int:
    return {
        "rewrite_style": 60,
        "summary_bullets": 90,
        "explain_compare": 90,
        "email_message": 140,
    }.get(family, 120)

def build_system_prompt(family: str) -> str:
    return BASE_SYSTEM + "\n" + FAMILY_RULES.get(family, "")

def build_user_prompt(row: Dict[str, Any]) -> str:
    family = row["family"]
    constraints = row.get("constraints", {})
    task = row["canonical_prompt"]

    extra: List[str] = []

    if family == "summary_bullets" and constraints.get("exact_bullets") is not None:
        n = constraints["exact_bullets"]
        extra.append(f"Return exactly {n} bullet points. Each line must start with '- '. Return no other text.")

    if family == "explain_compare":
        max_s = constraints.get("max_sentences", 3)
        extra.append(f"Return 2 or 3 plain sentences, and no more than {max_s} sentences. Do not use bullets or numbered lists.")

    if family == "rewrite_style":
        extra.append("Return only the rewritten text in one short paragraph. Do not explain the changes.")

    if family == "email_message":
        if constraints.get("must_be_email"):
            extra.append("Write a short email with a greeting and a sign-off. Do not use placeholders.")
        elif constraints.get("must_be_short_message"):
            extra.append("Write a short workplace message only. Do not add a greeting or sign-off unless natural.")

    if extra:
        return task + "\n\nOutput rules:\n" + "\n".join(f"- {x}" for x in extra)

    return task

def raw_mode(rows: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        family = row["family"]
        max_toks = max_tokens_for_family(family)
        answer = openai_compatible_chat(
            api_base=args.api_base,
            model=args.model,
            system_prompt=build_system_prompt(family),
            user_prompt=build_user_prompt(row),
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
                "teacher_generate_version": "v1_1",
            },
        })
    return out

def repair_mode(rows: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        family = row["family"]
        prompt = f"""The previous answer failed formatting checks.

Task:
{row['prompt']}

Previous answer:
{row['bad_answer']}

Problems:
""" + "\n".join(f"- {x}" for x in row["repair_reasons"]) + "\n\nRewrite the answer from scratch. Return only the final answer."
        max_toks = max_tokens_for_family(family)
        answer = openai_compatible_chat(
            api_base=args.api_base,
            model=args.model,
            system_prompt=build_system_prompt(family),
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
                "teacher_generate_version": "v1_1",
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
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    rows = read_jsonl(args.in_jsonl)
    out = raw_mode(rows, args) if args.mode == "raw" else repair_mode(rows, args)
    write_jsonl(args.out_jsonl, out)
    print(f"Wrote {len(out)} rows to {args.out_jsonl}")

if __name__ == "__main__":
    main()
