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

from src.chat_template import (  # noqa: E402
    DEFAULT_SYSTEM,
    encode_prompt as chat_encode_prompt,
    load_chat_tokenizer,
    prepare_prompt_messages,
    truncate_chat_sequence,
)


# -------------------------
# Record conversion (pure, unit-tested)
# -------------------------
def code_bank_record_to_prompt(
    rec: dict[str, Any],
    tag: str = "[Code] ",
    default_system: str = DEFAULT_SYSTEM,
) -> dict | None:
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
    messages = prepare_prompt_messages(
        [{"role": "user", "content": (tag + prompt) if tag else prompt}],
        default_system,
    )
    out: dict[str, Any] = {
        "messages": messages,
        "tests": list(tests),
        "meta": {**(rec.get("meta") or {}), "source": "code_bank"},
    }
    entry = rec.get("entry_point") or (rec.get("meta") or {}).get("entry_point")
    if entry:
        out["entry_point"] = entry
    return out


def messages_record_to_prompt(
    rec: dict[str, Any],
    default_system: str = DEFAULT_SYSTEM,
) -> dict | None:
    """Convert a valid SFT record to an explicit, canonical USER-ending prompt.

    A final assistant answer is removed only through the shared explicit helper;
    malformed roles or empty content fail rather than being silently skipped.
    """
    messages = rec.get("messages")
    if messages is None:
        return None
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    prompt_messages = prepare_prompt_messages(messages, default_system)
    out: dict[str, Any] = {
        "messages": prompt_messages,
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


def prompt_token_len(
    rec: dict[str, Any],
    tok: Tokenizer,
    default_system: str = DEFAULT_SYSTEM,
) -> int:
    """Canonical prompt length including BOS, system/role tokens, and cue."""
    ids = chat_encode_prompt(
        tok, rec["messages"], default_system, mode="full_context"
    )
    return len(ids)


def filter_by_prompt_tokens(
    records: list[dict],
    tok: Tokenizer,
    min_tokens: int,
    max_tokens: int,
    *,
    max_prompt_len: int,
    default_system: str = DEFAULT_SYSTEM,
    rejected: list[dict[str, Any]] | None = None,
) -> list[dict]:
    """Apply the same structural encoding/truncation budget as GRPO training."""
    out: list[dict] = []
    for index, record in enumerate(records):
        try:
            ids = chat_encode_prompt(
                tok, record["messages"], default_system, mode="full_context"
            )
            kept_ids, _ = truncate_chat_sequence(
                ids, labels=None, max_len=max_prompt_len
            )
            token_count = len(kept_ids)
            if token_count < min_tokens:
                raise ValueError(
                    f"canonical prompt has {token_count} tokens, below minimum {min_tokens}"
                )
            if max_tokens > 0 and token_count > max_tokens:
                raise ValueError(
                    f"canonical prompt has {token_count} tokens, above maximum {max_tokens}"
                )
        except (KeyError, TypeError, ValueError) as exc:
            if rejected is not None:
                rejected.append(
                    {
                        "record_index": index,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            continue
        out.append(record)
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
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--default_system", default=DEFAULT_SYSTEM)
    ap.add_argument("--min_prompt_tokens", type=int, default=1)
    ap.add_argument("--max_prompt_tokens", type=int, default=384, help="0 disables the upper bound")
    ap.add_argument("--limit", type=int, default=0, help="cap total prompts (0 = no cap)")
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if not args.code_bank and not args.messages:
        ap.error("provide at least one of --code_bank / --messages")
    if not 0 < args.max_new_tokens < args.seq_len:
        ap.error("--max_new_tokens must satisfy 0 < value < --seq_len")
    if args.min_prompt_tokens < 1:
        ap.error("--min_prompt_tokens must be positive")
    if args.max_prompt_tokens < 0:
        ap.error("--max_prompt_tokens must be non-negative")
    max_prompt_len = args.seq_len - args.max_new_tokens

    os.makedirs(args.out_dir, exist_ok=True)
    tok = load_chat_tokenizer(args.tokenizer_path)

    records: list[dict] = []
    for path in args.code_bank:
        n0 = len(records)
        for rec in read_jsonl(path):
            g = code_bank_record_to_prompt(
                rec, tag=args.code_tag, default_system=args.default_system
            )
            if g is not None:
                records.append(g)
        print(f"[code_bank] {path}: +{len(records) - n0} prompts")
    for path in args.messages:
        n0 = len(records)
        for rec in read_jsonl(path):
            g = messages_record_to_prompt(
                rec, default_system=args.default_system
            )
            if g is not None:
                records.append(g)
        print(f"[messages] {path}: +{len(records) - n0} prompts")

    before = len(records)
    records = dedup_by_prompt(records)
    print(f"[dedup] {before} -> {len(records)}")

    before = len(records)
    rejected: list[dict[str, Any]] = []
    records = filter_by_prompt_tokens(
        records,
        tok,
        args.min_prompt_tokens,
        args.max_prompt_tokens,
        max_prompt_len=max_prompt_len,
        default_system=args.default_system,
        rejected=rejected,
    )
    print(
        f"[canonical token preflight: trainer_budget={max_prompt_len}, "
        f"filter=[{args.min_prompt_tokens}, {args.max_prompt_tokens or '∞'}]] "
        f"{before} -> {len(records)} (rejected={len(rejected)})"
    )
    preflight_report = {
        "schema_version": 1,
        "kind": "petitgpt_grpo_prepare_preflight",
        "status": "passed" if records else "failed",
        "seq_len": args.seq_len,
        "max_new_tokens": args.max_new_tokens,
        "max_prompt_len": max_prompt_len,
        "input_records": before,
        "accepted_records": len(records),
        "rejected_records": len(rejected),
        "errors": rejected[:50],
        "errors_truncated": len(rejected) > 50,
    }
    with open(
        os.path.join(args.out_dir, "preflight_report.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(preflight_report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    if not records:
        raise ValueError("no GRPO prompts survived canonical token-budget preflight")

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
