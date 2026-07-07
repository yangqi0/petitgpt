#!/usr/bin/env python3

"""
Build a prompts-only JSONL bank for GRPO (grpo/grpo.py).

GRPO does not need chosen/rejected pairs (like DPO) or teacher answers (like
distillation) — it generates its own completions and scores them with a reward
function. So its data is just prompts, optionally carrying the fields a reward
needs (`tests`/`entry_point` for the `code` reward, `reference` for the
`reference_*` rewards). This script assembles such a bank from the project's own
local data, so it runs with no downloads:

- `--code_bank a.jsonl` (repeatable): canonical code-prompt records produced by
  the distillation pipeline (fields `canonical_prompt`/`prompt`, `entry_point`,
  `tests`) -> code-RLVR prompts. This is the natural source for `--reward code`.
- `--messages a.jsonl` (repeatable): SFT/distill-style `{"messages": [...]}`
  records -> prompts (everything up to and including the last user turn; any
  trailing assistant answer is dropped). Carries over `reference`/`answer`/
  `tests`/`entry_point` if present.

Deduplicates by prompt, optionally filters by prompt token length, and writes
`{out_dir}/train.jsonl` and `{out_dir}/val.jsonl`.

Example:
    python grpo/prepare_grpo_data.py \\
      --code_bank dataset/distill/code_canonical_prompts.jsonl \\
      --tokenizer_path tokenizer/tokenizer.json --out_dir datasets/grpo \\
      --max_prompt_tokens 384 --val_ratio 0.05
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any

from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# -------------------------
# Record conversion (pure, unit-tested)
# -------------------------
def code_bank_record_to_prompt(rec: dict[str, Any], tag: str = "[Code] ") -> dict | None:
    """Canonical code-prompt record -> GRPO prompt with tests for `code` reward.

    Returns None if the record lacks a prompt or unit tests (both required for a
    verifiable code reward).
    """
    prompt = (
        rec.get("canonical_prompt") or rec.get("prompt") or rec.get("raw_prompt") or ""
    ).strip()
    tests = rec.get("tests") or []
    if not prompt or not tests:
        return None
    out: dict[str, Any] = {
        "messages": [{"role": "user", "content": (tag + prompt) if tag else prompt}],
        "tests": list(tests),
        "meta": {**(rec.get("meta") or {}), "source": "code_bank"},
    }
    entry = rec.get("entry_point") or (rec.get("meta") or {}).get("entry_point")
    if entry:
        out["entry_point"] = entry
    return out


def messages_record_to_prompt(rec: dict[str, Any]) -> dict | None:
    """`{"messages": [...]}` record -> prompt up to and including the last user
    turn (drops a trailing assistant answer). Carries reward-relevant fields."""
    msgs = rec.get("messages") or []
    last_user = max(
        (i for i, m in enumerate(msgs) if (m.get("role") or "").strip().lower() == "user"),
        default=-1,
    )
    if last_user < 0:
        return None
    out: dict[str, Any] = {
        "messages": msgs[: last_user + 1],
        "meta": {**(rec.get("meta") or {}), "source": "messages"},
    }
    for key in ("reference", "answer", "tests", "entry_point"):
        if key in rec and rec[key] not in (None, "", []):
            out[key] = rec[key]
    return out


def _prompt_key(rec: dict[str, Any]) -> str:
    return json.dumps(rec["messages"], sort_keys=True, ensure_ascii=False)


def dedup_by_prompt(records: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for r in records:
        key = _prompt_key(r)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def prompt_token_len(rec: dict[str, Any], tok: Tokenizer) -> int:
    text = "\n".join(m.get("content", "") for m in rec["messages"])
    return len(tok.encode(text).ids)


def filter_by_prompt_tokens(
    records: list[dict], tok: Tokenizer, min_tokens: int, max_tokens: int
) -> list[dict]:
    out: list[dict] = []
    for r in records:
        n = prompt_token_len(r, tok)
        if n < min_tokens:
            continue
        if max_tokens > 0 and n > max_tokens:
            continue
        out.append(r)
    return out


def split_train_val(
    records: list[dict], val_ratio: float, seed: int
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    recs = list(records)
    rng.shuffle(recs)
    n_val = int(len(recs) * val_ratio)
    return recs[n_val:], recs[:n_val]


# -------------------------
# IO
# -------------------------
def read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--code_bank",
        action="append",
        default=[],
        help="canonical code-prompt jsonl (repeatable) -> code-RLVR prompts",
    )
    ap.add_argument(
        "--messages",
        action="append",
        default=[],
        help="SFT/distill-style messages jsonl (repeatable) -> prompts-only",
    )
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--code_tag", default="[Code] ", help="prefix prepended to code prompts")
    ap.add_argument("--min_prompt_tokens", type=int, default=1)
    ap.add_argument("--max_prompt_tokens", type=int, default=384, help="0 disables the upper bound")
    ap.add_argument("--limit", type=int, default=0, help="cap total prompts (0 = no cap)")
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if not args.code_bank and not args.messages:
        ap.error("provide at least one of --code_bank / --messages")

    os.makedirs(args.out_dir, exist_ok=True)
    tok = Tokenizer.from_file(args.tokenizer_path)

    records: list[dict] = []
    for path in args.code_bank:
        n0 = len(records)
        for rec in read_jsonl(path):
            g = code_bank_record_to_prompt(rec, tag=args.code_tag)
            if g is not None:
                records.append(g)
        print(f"[code_bank] {path}: +{len(records) - n0} prompts")
    for path in args.messages:
        n0 = len(records)
        for rec in read_jsonl(path):
            g = messages_record_to_prompt(rec)
            if g is not None:
                records.append(g)
        print(f"[messages] {path}: +{len(records) - n0} prompts")

    before = len(records)
    records = dedup_by_prompt(records)
    print(f"[dedup] {before} -> {len(records)}")

    before = len(records)
    records = filter_by_prompt_tokens(records, tok, args.min_prompt_tokens, args.max_prompt_tokens)
    print(
        f"[filter tokens in [{args.min_prompt_tokens}, {args.max_prompt_tokens or '∞'}]] "
        f"{before} -> {len(records)}"
    )

    if args.limit and len(records) > args.limit:
        random.Random(args.seed).shuffle(records)
        records = records[: args.limit]
        print(f"[limit] -> {len(records)}")

    train, val = split_train_val(records, args.val_ratio, args.seed)
    write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train)
    write_jsonl(os.path.join(args.out_dir, "val.jsonl"), val)
    n_code = sum(1 for r in train + val if "tests" in r)
    print(f"[done] train={len(train)} val={len(val)} (with unit tests: {n_code}) -> {args.out_dir}")


if __name__ == "__main__":
    main()
