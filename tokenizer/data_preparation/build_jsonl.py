#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build JSONL files for tokenizer training (and optionally pretrain text staging).

Output format (one JSON per line):
  {"text": "..."}  # whitespace preserved

Why this script exists
----------------------
Tokenizer quality depends heavily on whitespace + punctuation + structural patterns
(newlines, indentation, markdown, code, JSON, etc.). Therefore:

- DO NOT collapse whitespace (no " ".join(split())).
- DO NOT strip text fields aggressively.
- Keep only minimal normalization: unify newline styles, remove invalid control chars.

This script exports a mixed corpus from:
- FineWeb (general web)
- UltraChat (chat-style)
- OASST1 (assistant chat)
- Dolly (instruction)

It uses streaming=True and dataset.shuffle(buffer_size=...) for approximate random sampling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Iterable

from datasets import load_dataset
from tqdm import tqdm


# -------------------------
# Minimal, safe normalization
# -------------------------
def normalize_text_preserve_ws(s: str) -> str:
    """
    Minimal normalization that preserves whitespace structure.

    - Unify Windows/Mac newlines to '\n'
    - Remove NUL bytes and a few problematic control chars
    - Keep tabs/spaces/newlines exactly as-is otherwise
    """
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # Remove NUL and a couple of rarely useful control chars
    s = s.replace("\x00", "")
    s = s.replace("\u000b", "")  # vertical tab
    s = s.replace("\u000c", "")  # form feed
    return s


def is_reasonable_text(s: str, min_chars: int, max_chars: int) -> bool:
    """
    Quick filters to avoid junk and extreme outliers.
    """
    if not isinstance(s, str):
        return False
    if len(s) < min_chars:
        return False
    if len(s) > max_chars:
        return False
    # Must contain at least one non-whitespace character
    if not any(not ch.isspace() for ch in s):
        return False
    return True


def write_jsonl(
    ex_iter: Iterable[Any],
    out_path: Path,
    max_examples: int,
    get_text_fn: Callable[[Any], str],
    min_chars: int,
    max_chars: int,
) -> int:
    """
    Stream examples, extract text, lightly normalize, filter, and write JSONL lines:
      {"text": "..."}
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ex in tqdm(ex_iter, desc=f"writing {out_path.name}"):
            if n >= max_examples:
                break

            text = get_text_fn(ex)
            if not isinstance(text, str):
                continue

            text = normalize_text_preserve_ws(text)
            if not is_reasonable_text(text, min_chars=min_chars, max_chars=max_chars):
                continue

            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n += 1

    return n


# -------------------------
# Dataset-specific extractors
# -------------------------
def fineweb_to_text(ex: dict[str, Any]) -> str:
    return ex.get("text", "") or ""


def ultrachat_to_text(ex: dict[str, Any]) -> str:
    """
    UltraChat 200k typically has 'messages': [{role, content}, ...]
    We keep a simple stable format without stripping message content.
    """
    msgs = ex.get("messages", None)
    if isinstance(msgs, list) and msgs:
        parts: list[str] = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = m.get("role", "")
            content = m.get("content", "")
            if isinstance(content, str) and content != "":
                parts.append(f"{role}:\n{content}")
        if parts:
            return "\n\n".join(parts)
    return ex.get("text", "") or ""


def oasst_to_text(ex: dict[str, Any]) -> str:
    """
    OASST1 rows are single messages. We expose role + content.
    """
    role = ex.get("role", "")
    text = ex.get("text", "")
    if isinstance(text, str) and text != "":
        return f"{role}:\n{text}"
    return ""


def dolly_to_text(ex: dict[str, Any]) -> str:
    """
    Dolly fields: instruction / context / response
    Keep newlines between sections; do not strip.
    """
    instr = ex.get("instruction", "") or ""
    ctx = ex.get("context", "") or ""
    resp = ex.get("response", "") or ""
    parts: list[str] = []
    if instr:
        parts.append("Instruction:\n" + instr)
    if ctx:
        parts.append("Context:\n" + ctx)
    if resp:
        parts.append("Answer:\n" + resp)
    return "\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default=".", help="Output directory for JSONL.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for streaming shuffle.")

    # Export sizes (tokenizer training)
    ap.add_argument("--n_fineweb", type=int, default=800_000)
    ap.add_argument("--n_ultrachat", type=int, default=200_000)
    ap.add_argument("--n_oasst1", type=int, default=80_000)
    ap.add_argument("--n_dolly", type=int, default=30_000)

    # Streaming shuffle buffer (bigger => better randomness, more RAM)
    ap.add_argument("--shuffle_buffer", type=int, default=50_000)

    # Text filters
    ap.add_argument("--min_chars", type=int, default=1)
    ap.add_argument("--max_chars", type=int, default=20_000)

    # FineWeb config
    ap.add_argument("--fineweb_config", type=str, default="sample-10BT")

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) FineWeb (general web)
    fineweb = load_dataset(
        "HuggingFaceFW/fineweb",
        name=args.fineweb_config,
        split="train",
        streaming=True,
    )
    # Approximate random order for streaming
    fineweb = fineweb.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    n1 = write_jsonl(
        fineweb,
        out_dir / "fineweb_sample.jsonl",
        max_examples=args.n_fineweb,
        get_text_fn=fineweb_to_text,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )

    # 2) UltraChat 200k (chat-style)
    ultrachat = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split="train_sft",
        streaming=True,
    )
    ultrachat = ultrachat.shuffle(seed=args.seed + 1, buffer_size=args.shuffle_buffer)

    n2 = write_jsonl(
        ultrachat,
        out_dir / "ultrachat_200k.jsonl",
        max_examples=args.n_ultrachat,
        get_text_fn=ultrachat_to_text,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )

    # 3) OASST1
    oasst = load_dataset("OpenAssistant/oasst1", split="train", streaming=True)
    oasst = oasst.shuffle(seed=args.seed + 2, buffer_size=args.shuffle_buffer)

    n3 = write_jsonl(
        oasst,
        out_dir / "oasst1.jsonl",
        max_examples=args.n_oasst1,
        get_text_fn=oasst_to_text,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )

    # 4) Dolly 15k
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train", streaming=True)
    dolly = dolly.shuffle(seed=args.seed + 3, buffer_size=args.shuffle_buffer)

    n4 = write_jsonl(
        dolly,
        out_dir / "dolly_15k.jsonl",
        max_examples=args.n_dolly,
        get_text_fn=dolly_to_text,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )

    print("Done. Exported examples:")
    print("  FineWeb :", n1)
    print("  UltraChat:", n2)
    print("  OASST1  :", n3)
    print("  Dolly   :", n4)
    print("Output dir:", out_dir)


if __name__ == "__main__":
    main()
