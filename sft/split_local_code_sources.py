#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Group-aware train/val split for small, high-value local code datasets.

Key change vs. the old script:
  - split by base_id / task_id / problem_id / template_id groups
  - NEVER random-split individual augmented rows from the same base problem
  - this prevents train/val leakage when local data contains prompt variants or repeats

Expected input rows are canonical chat JSONL, e.g.:
  {
    "messages": [...],
    "meta": {"base_id": "mbpp_123", ...}
  }

If no explicit base identifier is found, the script falls back to a stable hash built
from the first user message + first assistant message.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def first_content(messages: List[Dict[str, Any]], role: str) -> str:
    for m in messages or []:
        if m.get("role") == role:
            return str(m.get("content", "")).strip()
    return ""


def infer_group_id(row: Dict[str, Any]) -> str:
    meta = row.get("meta") or {}
    for key in (
        "base_id",
        "task_id",
        "problem_id",
        "template_id",
        "group_id",
        "source_id",
    ):
        val = meta.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, int):
            return str(val)

    user = first_content(row.get("messages", []), "user")
    assistant = first_content(row.get("messages", []), "assistant")
    sig = user + "\n\n===ASSISTANT===\n\n" + assistant
    return "hash_" + hashlib.sha1(sig.encode("utf-8")).hexdigest()


def choose_val_group_ids(
    group_ids: List[str],
    groups: Dict[str, List[Dict[str, Any]]],
    rng: random.Random,
    val_ratio: float,
    max_val_examples: int,
) -> set[str]:
    gids = group_ids[:]
    rng.shuffle(gids)

    n_total = sum(len(groups[g]) for g in gids)
    target = max(1, int(round(n_total * val_ratio)))
    if max_val_examples > 0:
        target = min(target, max_val_examples)
    # always leave something for train
    target = min(target, max(1, n_total - 1))

    chosen: set[str] = set()
    acc = 0
    for gid in gids:
        gsz = len(groups[gid])
        # keep at least one entire group for train
        if acc >= target:
            break
        if acc + gsz > target and acc > 0:
            # stop once we'd overshoot too much; whole-group split only
            break
        chosen.add(gid)
        acc += gsz

    if not chosen:
        chosen.add(gids[0])
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_train", required=True)
    ap.add_argument("--out_val", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--val_ratio", type=float, default=0.08)
    ap.add_argument(
        "--max_val_examples",
        type=int,
        default=0,
        help="Optional hard cap on validation examples. 0 means no cap.",
    )
    ap.add_argument(
        "--group_key",
        default="auto",
        help="Reserved for future use. Current script auto-detects meta.base_id/task_id/etc.",
    )
    args = ap.parse_args()

    rows = read_jsonl(args.in_jsonl)
    if not rows:
        raise SystemExit(f"empty input: {args.in_jsonl}")

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        gid = infer_group_id(row)
        groups[gid].append(row)

    group_ids = list(groups.keys())
    if len(group_ids) == 1:
        raise SystemExit(
            "only one group found; cannot make a leakage-safe train/val split. "
            "Please generate more distinct base problems first."
        )

    rng = random.Random(args.seed)
    val_group_ids = choose_val_group_ids(
        group_ids, groups, rng, args.val_ratio, args.max_val_examples
    )

    train_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    for gid, grows in groups.items():
        if gid in val_group_ids:
            val_rows.extend(grows)
        else:
            train_rows.extend(grows)

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    write_jsonl(args.out_train, train_rows)
    write_jsonl(args.out_val, val_rows)

    print(f"input={args.in_jsonl}")
    print(f"total_rows={len(rows)}")
    print(f"total_groups={len(group_ids)}")
    print(f"train_groups={len(group_ids) - len(val_group_ids)}")
    print(f"val_groups={len(val_group_ids)}")
    print(f"train={len(train_rows)} -> {args.out_train}")
    print(f"val={len(val_rows)} -> {args.out_val}")


if __name__ == "__main__":
    main()
