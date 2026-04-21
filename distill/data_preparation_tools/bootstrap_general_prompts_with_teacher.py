#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Use an OpenAI-compatible teacher endpoint to bootstrap many general prompts.

This script asks the teacher to produce ONE new prompt at a time in JSON.
That is slower than batched JSON generation, but it is much more robust and
easier to debug.

Output row schema:
{
  "id": "...",
  "task_type": "general_email|general_summary|general_rewrite|general_explain|general_checklist",
  "prompt": "...",
  "meta": {...}
}
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from typing import Any

from openai import OpenAI


DEFAULT_SYSTEM = "You are a helpful assistant."
TASK_TYPES = [
    "general_email",
    "general_summary",
    "general_rewrite",
    "general_explain",
    "general_checklist",
]


EMAIL_TOPICS = [
    "asking for an update after a job interview",
    "asking whether a submitted code test has been reviewed",
    "rescheduling a meeting to next week",
    "confirming that documents were received",
    "asking for clarification about a project deadline",
    "asking for feedback on a submitted project",
    "asking whether a role is still open",
    "requesting a short extension for a project deadline",
    "asking for a convenient time for a follow-up meeting",
    "checking whether travel reimbursement paperwork was received",
    "asking whether there are any next steps after a screening call",
    "checking the status of a bug report you submitted",
]

SUMMARY_DOMAINS = [
    "a short paragraph about Python and data analysis",
    "a short explanation of version control",
    "a short explanation of unit testing",
    "a short explanation of data cleaning",
    "a short explanation of gradient descent",
    "a short explanation of APIs",
    "a short explanation of debugging",
    "a short explanation of why readable code matters",
]

REWRITE_SEEDS = [
    "Send me the file today.",
    "I want to know what is going on with my application.",
    "Can you fix this bug ASAP?",
    "Your report is late and needs changes.",
    "I do not understand your instructions.",
    "Please answer me quickly.",
    "The meeting time does not work for me.",
    "This explanation is hard to follow.",
    "This draft is confusing and too long.",
]

EXPLAIN_CONCEPTS = [
    "the difference between a Python list and a tuple",
    "the difference between a function and a method in Python",
    "what debugging means",
    "what an API is",
    "what a CSV file is",
    "what version control is",
    "why readable variable names matter",
    "what overfitting means in machine learning",
    "what a tokenizer does in a language model pipeline",
    "why binary search requires sorted input",
]

CHECKLIST_TOPICS = [
    "preparing for a Python coding interview",
    "debugging a function that returns the wrong result",
    "writing cleaner Python code",
    "organizing a small coding project",
    "reviewing a pull request",
    "preparing a small project for GitHub",
    "writing a clear bug report",
    "cleaning a small CSV dataset",
]


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


def normalize_prompt(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


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


def build_user_request(task_type: str, seed: str, rng: random.Random) -> tuple[str, dict[str, Any]]:
    if task_type == "general_email":
        meta = {"format": "email"}
        req = (
            "Create ONE new high-quality user prompt for an email-writing task.\n"
            "Constraints:\n"
            "- It should be realistic and professional.\n"
            "- It should request a short email.\n"
            "- It should not mention any answer format.\n"
            "- It should be clearly different from the seed topic.\n"
            "Return exactly one JSON object with keys: task_type, prompt, meta.\n"
            f'Set task_type to "general_email".\n'
            f'Set meta to {json.dumps(meta)}.\n'
            f"Seed topic: {seed}\n"
        )
        return req, meta

    if task_type == "general_summary":
        bullet_count = rng.choice([0, 3])
        meta = {"bullet_count": bullet_count}
        req = (
            "Create ONE new high-quality user prompt for a summary task.\n"
            "Constraints:\n"
            "- Include a short source text inside the prompt.\n"
            "- Keep the source text practical and non-fiction.\n"
            "- If bullet_count is 3, ask for exactly 3 bullets.\n"
            "- If bullet_count is 0, ask for one short paragraph.\n"
            "- Do not include the answer.\n"
            "Return exactly one JSON object with keys: task_type, prompt, meta.\n"
            f'Set task_type to "general_summary".\n'
            f'Set meta to {json.dumps(meta)}.\n'
            f"Seed domain: {seed}\n"
        )
        return req, meta

    if task_type == "general_rewrite":
        meta = {}
        req = (
            "Create ONE new high-quality user prompt for a rewrite task.\n"
            "Constraints:\n"
            "- Include a short original sentence or paragraph inside the prompt.\n"
            "- Ask to rewrite it to be more polite, more professional, shorter, clearer, or friendlier.\n"
            "- Do not include the rewritten answer.\n"
            "Return exactly one JSON object with keys: task_type, prompt, meta.\n"
            f'Set task_type to "general_rewrite".\n'
            f'Set meta to {json.dumps(meta)}.\n'
            f"Seed text: {seed}\n"
        )
        return req, meta

    if task_type == "general_explain":
        meta = {"max_sentences": 5}
        req = (
            "Create ONE new high-quality user prompt for a beginner-friendly explanation task.\n"
            "Constraints:\n"
            "- Ask for a simple explanation.\n"
            "- Keep it suitable for a beginner.\n"
            "- Keep the prompt practical, technical, or study-related.\n"
            "- Do not include the answer.\n"
            "Return exactly one JSON object with keys: task_type, prompt, meta.\n"
            f'Set task_type to "general_explain".\n'
            f'Set meta to {json.dumps(meta)}.\n'
            f"Seed concept: {seed}\n"
        )
        return req, meta

    if task_type == "general_checklist":
        item_count = rng.choice([4, 5, 6])
        meta = {"item_count": item_count}
        req = (
            "Create ONE new high-quality user prompt for a checklist or numbered-list task.\n"
            "Constraints:\n"
            "- Ask for a numbered list.\n"
            "- Keep the topic practical and technical or professional.\n"
            "- Do not include the answer.\n"
            "Return exactly one JSON object with keys: task_type, prompt, meta.\n"
            f'Set task_type to "general_checklist".\n'
            f'Set meta to {json.dumps(meta)}.\n'
            f"Seed topic: {seed}\n"
        )
        return req, meta

    raise ValueError(f"unknown task_type: {task_type}")


def choose_seed(task_type: str, rng: random.Random) -> str:
    if task_type == "general_email":
        return rng.choice(EMAIL_TOPICS)
    if task_type == "general_summary":
        return rng.choice(SUMMARY_DOMAINS)
    if task_type == "general_rewrite":
        return rng.choice(REWRITE_SEEDS)
    if task_type == "general_explain":
        return rng.choice(EXPLAIN_CONCEPTS)
    if task_type == "general_checklist":
        return rng.choice(CHECKLIST_TOPICS)
    raise ValueError(task_type)


def validate_row(row: dict[str, Any], task_type: str) -> bool:
    if not isinstance(row, dict):
        return False
    if row.get("task_type") != task_type:
        return False
    prompt = row.get("prompt")
    if not isinstance(prompt, str) or len(prompt.strip()) < 20:
        return False
    if "```" in prompt:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_target", type=int, default=2000)
    ap.add_argument("--base_url", required=True)
    ap.add_argument("--api_key", default="EMPTY")
    ap.add_argument("--model", required=True)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=220)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max_attempts", type=int, default=100000)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    rng = random.Random(args.seed)

    seen = read_existing_prompts(args.out)
    rows: list[dict[str, Any]] = []

    attempts = 0
    while len(rows) < args.n_target and attempts < args.max_attempts:
        attempts += 1
        task_type = rng.choice(TASK_TYPES)
        seed = choose_seed(task_type, rng)
        user_req, meta = build_user_request(task_type, seed, rng)

        resp = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": DEFAULT_SYSTEM + "\n\nReturn exactly one JSON object and nothing else."},
                {"role": "user", "content": user_req},
            ],
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
        )
        text = resp.choices[0].message.content.strip()
        obj = parse_single_json_object(text)
        if obj is None or not validate_row(obj, task_type):
            continue

        norm = normalize_prompt(obj["prompt"])
        if norm in seen:
            continue

        row = {
            "id": f"{task_type}_{len(rows)+1:06d}",
            "task_type": task_type,
            "prompt": obj["prompt"].strip(),
            "meta": obj.get("meta", meta) or meta,
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
