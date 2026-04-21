#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a canonical SFT dataset from:
- general teacher raw outputs
- verified code teacher outputs

Output rows match the current SFT trainer expectation:
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "meta": {...}
}

Comments are in English by design.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from typing import Any


DEFAULT_SYSTEM = "You are a helpful assistant."


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    if not path:
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_sentences(text: str) -> int:
    parts = re.split(r"[.!?]+(?:\s+|$)", text.strip())
    return len([p for p in parts if p.strip()])


def count_bullets(text: str) -> int:
    return len(re.findall(r"(?m)^\s*[-*•]\s+", text))


def count_numbered_items(text: str) -> int:
    return len(re.findall(r"(?m)^\s*\d+[.)]\s+", text))


def looks_like_email(text: str) -> bool:
    return "Subject:" in text and ("Hello" in text or "Hi" in text) and ("Best," in text or "Regards," in text or "Sincerely," in text)


def validate_general(row: dict[str, Any]) -> tuple[bool, str]:
    task_type = row["task_type"]
    text = (row.get("response") or "").strip()
    meta = row.get("meta", {}) or {}

    if not text:
        return False, "empty"

    if "```" in text:
        return False, "contains_code_block"

    if len(text) < 8:
        return False, "too_short"

    if task_type == "general_email":
        return (looks_like_email(text), "bad_email_format" if not looks_like_email(text) else "ok")

    if task_type == "general_summary":
        bullet_count = int(meta.get("bullet_count", 0) or 0)
        if bullet_count > 0:
            return (count_bullets(text) == bullet_count, "wrong_bullet_count")
        return (count_sentences(text) <= 4, "summary_too_long")

    if task_type == "general_rewrite":
        return (count_sentences(text) <= 6, "rewrite_too_long")

    if task_type == "general_explain":
        max_sentences = int(meta.get("max_sentences", 5) or 5)
        return (count_sentences(text) <= max_sentences + 1, "too_many_sentences")

    if task_type == "general_checklist":
        item_count = meta.get("item_count")
        if item_count:
            got = count_numbered_items(text)
            return (got == int(item_count), "wrong_item_count")
        return (count_numbered_items(text) > 0, "missing_numbered_list")

    return True, "ok"


def to_canonical(prompt: str, answer: str, bucket: str, task_type: str, source: str, prompt_id: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": prompt.strip()},
            {"role": "assistant", "content": answer.rstrip()},
        ],
        "meta": {
            "bucket": bucket,
            "source": source,
            "task_type": task_type,
            "prompt_id": prompt_id,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--general_raw", required=True)
    ap.add_argument("--code_verified", required=True)
    ap.add_argument("--out_train", required=True)
    ap.add_argument("--out_val", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max_general", type=int, default=0, help="0 means keep all")
    ap.add_argument("--max_code", type=int, default=0, help="0 means keep all")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    general_rows = read_jsonl(args.general_raw)
    code_rows = read_jsonl(args.code_verified)

    general_kept: list[dict[str, Any]] = []
    for row in general_rows:
        ok, _ = validate_general(row)
        if not ok:
            continue
        general_kept.append(
            to_canonical(
                prompt=row["prompt"],
                answer=row["response"],
                bucket="A_general",
                task_type=row["task_type"],
                source="teacher_general",
                prompt_id=row["prompt_id"],
            )
        )

    code_kept: list[dict[str, Any]] = []
    for row in code_rows:
        code_kept.append(
            to_canonical(
                prompt=row["prompt"],
                answer=row["response"],
                bucket="B_code",
                task_type=row["task_type"],
                source="teacher_code",
                prompt_id=row["prompt_id"],
            )
        )

    rng.shuffle(general_kept)
    rng.shuffle(code_kept)

    if args.max_general > 0:
        general_kept = general_kept[: args.max_general]
    if args.max_code > 0:
        code_kept = code_kept[: args.max_code]

    all_rows = general_kept + code_kept
    rng.shuffle(all_rows)

    n_val = max(1, int(len(all_rows) * args.val_ratio)) if all_rows else 0
    val_rows = all_rows[:n_val]
    train_rows = all_rows[n_val:]

    write_jsonl(args.out_train, train_rows)
    write_jsonl(args.out_val, val_rows)

    print(f"General kept: {len(general_kept)}")
    print(f"Code kept:    {len(code_kept)}")
    print(f"Train rows:    {len(train_rows)}")
    print(f"Val rows:      {len(val_rows)}")
    print(f"Wrote train -> {args.out_train}")
    print(f"Wrote val   -> {args.out_val}")


if __name__ == "__main__":
    main()
