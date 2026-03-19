#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List

from openai import OpenAI

DEFAULT_SYSTEM = "You are a helpful assistant."


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def build_format_instruction(task_type: str) -> str:
    if task_type == "general_email":
        return (
            "Write a short professional email.\n"
            "Output only the email.\n"
            "Use this format:\n"
            "Subject: <short subject>\n\n"
            "Hello ..., \n\n"
            "<body>\n\n"
            "Best,\n"
            "<name>\n"
        )

    if task_type == "general_summary":
        return (
            "Follow the output format exactly.\n"
            "If the user asks for 3 bullet points, output exactly 3 bullet points.\n"
            "Do not add any introduction or extra explanation."
        )

    if task_type == "general_rewrite":
        return (
            "Output only the rewritten text.\n"
            "Do not explain changes.\n"
            "Do not add quotation marks unless necessary."
        )

    if task_type == "general_explain":
        return (
            "Explain clearly in at most 5 short sentences.\n"
            "Do not use bullet points.\n"
            "Do not include code."
        )

    if task_type == "general_checklist":
        return (
            "Output a numbered list.\n"
            "Use exactly the number of items requested.\n"
            "Keep each item short."
        )

    if task_type in ("code_write", "code_fix", "code_script"):
        return (
            "Output only one Python code block.\n"
            "Do not include any explanation.\n"
            "Do not include multiple code blocks.\n"
            "Prefer simple, correct, executable code.\n"
            "Prefer the Python standard library."
        )

    if task_type == "code_explain_then_code":
        return (
            "Use exactly this format:\n"
            "Explanation:\n"
            "<2-3 short sentences>\n\n"
            "```python\n"
            "<code>\n"
            "```\n"
            "Do not add any extra text."
        )

    if task_type.startswith("math_"):
        return (
            "Use exactly this format:\n"
            "Brief steps:\n"
            "1. ...\n"
            "2. ...\n\n"
            "Final answer: ...\n"
            "Keep the steps brief.\n"
            "Do not use long chain-of-thought.\n"
            "Do not use LaTeX boxes."
        )

    return "Answer clearly and briefly."


def build_messages(prompt: str, task_type: str) -> List[Dict[str, str]]:
    fmt = build_format_instruction(task_type)
    user = prompt.strip()
    system = DEFAULT_SYSTEM + "\n\n" + "Follow the requested format exactly.\n" + fmt
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def generate_with_teacher(
    messages,
    teacher_name,
    temperature,
    top_p,
    max_new_tokens,
    n,
    reasoning_effort="low",
    sleep_s=0.0,
):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # iterate n times to get n responses, instead of using n=... in the API call, to avoid getting multiple responses in one output which can be hard to parse.
    outs = []
    for _ in range(n):
        response = client.responses.create(
            model=teacher_name,
            instructions=messages[0]["content"],  # system
            input=messages[1]["content"],  # user
            reasoning={"effort": reasoning_effort},
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_new_tokens,
        )
        outs.append(response.output_text.strip())
        if sleep_s > 0:
            time.sleep(sleep_s)
    return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument(
        "--model",
        required=True,
        default="gpt-5.4",
        help="Model name for the teacher (e.g., 'gpt-5.4', 'gpt-5.3-codex')",
    )
    ap.add_argument(
        "--reasoning_effort",
        default="low",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="Reasoning effort level for the teacher model (e.g., 'low', 'medium', 'high')",
    )
    ap.add_argument(
        "--sleep_s",
        type=float,
        default=0.0,
        help="Seconds to sleep between API calls to avoid rate limits",
    )
    args = ap.parse_args()

    rows = read_jsonl(args.in_jsonl)
    ensure_dir(args.out_jsonl)

    total = 0
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for ex in rows:
            prompt_id = ex["id"]
            task_type = ex["task_type"]
            prompt = ex["prompt"]

            messages = build_messages(prompt, task_type)
            responses = generate_with_teacher(
                messages=messages,
                teacher_name=args.model,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                n=args.n,
                reasoning_effort=args.reasoning_effort,
                sleep_s=args.sleep_s,
            )

            for cid, resp in enumerate(responses):
                out = {
                    "prompt_id": prompt_id,
                    "task_type": task_type,
                    "prompt": prompt,
                    "candidate_id": cid,
                    "response": resp,
                    "teacher": args.model,
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                total += 1

    print(f"Wrote {total} candidates to {args.out_jsonl}")


if __name__ == "__main__":
    main()
