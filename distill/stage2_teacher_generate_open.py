#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate teacher responses using an OpenAI-compatible chat endpoint.

This is intentionally endpoint-agnostic:
- local vLLM
- text-generation-inference with OpenAI-compatible proxy
- OpenRouter-style compatible endpoints
- other OpenAI-compatible servers

Input JSONL row fields:
- id
- task_type
- prompt
- meta (optional)

Output JSONL row fields:
- prompt_id
- task_type
- prompt
- candidate_id
- response
- teacher
- meta

Comments are in English by design.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

from openai import OpenAI


DEFAULT_SYSTEM = "You are a helpful assistant."


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def build_format_instruction(task_type: str, meta: dict[str, Any] | None) -> str:
    meta = meta or {}

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
        bullet_count = int(meta.get("bullet_count", 0) or 0)
        if bullet_count > 0:
            return (
                f"Output exactly {bullet_count} bullet points.\n"
                "Do not add any introduction, conclusion, or extra explanation.\n"
                "Do not include code."
            )
        return (
            "Output one short paragraph only.\n"
            "Do not add any introduction or extra explanation.\n"
            "Do not include code."
        )

    if task_type == "general_rewrite":
        return (
            "Output only the rewritten text.\n"
            "Do not explain changes.\n"
            "Do not include code.\n"
            "Do not add quotation marks unless necessary."
        )

    if task_type == "general_explain":
        max_sentences = int(meta.get("max_sentences", 5) or 5)
        return (
            f"Explain clearly in at most {max_sentences} short sentences.\n"
            "Do not use bullet points.\n"
            "Do not include code."
        )

    if task_type == "general_checklist":
        item_count = meta.get("item_count")
        if item_count:
            return (
                "Output a numbered list.\n"
                f"Use exactly {int(item_count)} items.\n"
                "Keep each item short.\n"
                "Do not include code."
            )
        return (
            "Output a numbered list.\n"
            "Keep each item short.\n"
            "Do not include code."
        )

    if task_type in ("code_write", "code_fix", "code_script"):
        entrypoint = (meta or {}).get("entrypoint", "")
        ep_line = f"The function name must be exactly `{entrypoint}`.\n" if entrypoint else ""
        return (
            "Output only one Python code block.\n"
            "Do not include any explanation.\n"
            "Do not include multiple code blocks.\n"
            "Prefer simple, correct, executable code.\n"
            "Prefer the Python standard library.\n"
            + ep_line
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

    return "Answer clearly and briefly."


def get_response_text(resp: Any) -> str:
    # Chat Completions path
    if hasattr(resp, "choices") and resp.choices:
        msg = resp.choices[0].message
        if msg is not None and getattr(msg, "content", None) is not None:
            return msg.content.strip()

    # Fallback to dict-like
    try:
        return resp["choices"][0]["message"]["content"].strip()
    except Exception:
        pass

    raise RuntimeError("Could not extract text from model response.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base_url", required=True)
    ap.add_argument("--api_key", default="EMPTY")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--sleep_s", type=float, default=0.0)
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    rows = read_jsonl(args.in_jsonl)
    ensure_dir(args.out_jsonl)

    total = 0
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for ex in rows:
            prompt_id = ex["id"]
            task_type = ex["task_type"]
            prompt = ex["prompt"]
            meta = ex.get("meta", {})

            system = DEFAULT_SYSTEM + "\n\n" + "Follow the requested format exactly.\n" + build_format_instruction(task_type, meta)
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]

            for cid in range(args.n):
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_new_tokens,
                )
                text = get_response_text(resp)

                out = {
                    "prompt_id": prompt_id,
                    "task_type": task_type,
                    "prompt": prompt,
                    "candidate_id": cid,
                    "response": text,
                    "teacher": args.model,
                    "meta": meta,
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                total += 1

                if args.sleep_s > 0:
                    time.sleep(args.sleep_s)

    print(f"Wrote {total} candidates to {args.out_jsonl}")


if __name__ == "__main__":
    main()
