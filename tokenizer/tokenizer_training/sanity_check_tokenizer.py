#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sanity checks for a trained Byte-Level BPE tokenizer.

This script is intentionally strict about whitespace round-trip because
newline/indentation are important for:
- pretraining (web + code)
- chat/SFT formatting
- markdown and JSON/code blocks

Checks
------
1) Strict round-trip tests on whitespace-sensitive strings.
2) Qualitative inspection on chat/markdown/code snippets.
3) Compression ratio (characters per token) on a sampled corpus.
4) Special token presence + (optional) strict ID assertions.
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Any

from tokenizers import Tokenizer


SPECIAL_CANDIDATES = {
    "pad": ["[PAD]", "<pad>"],
    "unk": ["[UNK]", "<unk>"],
    "bos": ["[BOS]", "<bos>", "<s>"],
    "eos": ["[EOS]", "<eos>", "</s>"],
}


def _json_loads(line: str) -> Any | None:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def strict_roundtrip(tok: Tokenizer, s: str) -> None:
    enc = tok.encode(s)
    dec = tok.decode(enc.ids)
    if dec != s:
        raise AssertionError(
            "Round-trip failed!\n"
            f"orig={repr(s)}\n"
            f"tokens={enc.tokens}\n"
            f"ids={enc.ids}\n"
            f"dec ={repr(dec)}\n"
        )


def find_special_ids(tok: Tokenizer) -> dict[str, tuple[int | None, str | None]]:
    out: dict[str, tuple[int | None, str | None]] = {}
    for key, names in SPECIAL_CANDIDATES.items():
        found_id = None
        found_name = None
        for name in names:
            tid = tok.token_to_id(name)
            if tid is not None:
                found_id = tid
                found_name = name
                break
        out[key] = (found_id, found_name)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer.json")
    ap.add_argument(
        "--jsonl",
        nargs="*",
        default=[],
        help="Optional JSONL files for compression test (expects field 'text' by default).",
    )
    ap.add_argument("--field", type=str, default="text", help="Text field for compression test.")
    ap.add_argument("--n_samples", type=int, default=2000, help="Max samples for compression test.")
    ap.add_argument(
        "--strict_special_ids",
        action="store_true",
        help="Assert [PAD]=0,[UNK]=1,[BOS]=2,[EOS]=3 (enable only if you require fixed IDs).",
    )
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)

    # --- 1) Strict round-trip tests (this catches the prefix-space issue immediately) ---
    tests = [
        "\n",
        "\n\n",
        " ",
        "  ",
        "\t",
        "a\nb",
        "\nHello",
        "Hello\n",
        "```python\nprint('hi')\n```",
        "{\n  \"a\": 1,\n  \"b\": [2, 3]\n}\n",
        "User:\nHi\n\nAssistant:\nHello!\n",
    ]
    for s in tests:
        strict_roundtrip(tok, s)
    print("[OK] strict round-trip tests passed")

    # --- 2) Qualitative inspection ---
    qualitative = [
        "Once upon a time, a robot said: \"Hello!\"",
        "### Task\n- Step 1: ...\n- Step 2: ...\n",
        "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
        "User: Summarize the following.\n\nAssistant:",
        "The following is a news report:\n\n",
    ]
    for i, s in enumerate(qualitative, start=1):
        enc = tok.encode(s)
        dec = tok.decode(enc.ids)
        print(f"\n--- Qualitative #{i} ---")
        print("orig:", repr(s))
        print("tokens (first 60):", enc.tokens[:60])
        print("ids    (first 60):", enc.ids[:60])
        print("dec :", repr(dec))

    # --- 3) Compression ratio (chars per token) ---
    if args.jsonl:
        texts: list[str] = []
        for p in args.jsonl:
            with open(p, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.rstrip("\r\n")
                    if not line:
                        continue
                    obj = _json_loads(line)
                    if not isinstance(obj, dict):
                        continue
                    t = obj.get(args.field)
                    if isinstance(t, str) and t != "":
                        texts.append(t)

        if texts:
            samples = random.sample(texts, min(args.n_samples, len(texts)))
            total_chars = 0
            total_tokens = 0
            for s in samples:
                total_chars += len(s)
                total_tokens += len(tok.encode(s).ids)
            ratio = total_chars / max(total_tokens, 1)
            print(f"\n[OK] avg chars/token on sample = {ratio:.3f}  (n={len(samples)})")
        else:
            print("\n[WARN] compression test skipped (no usable texts found)")

    # --- 4) Special tokens ---
    spec = find_special_ids(tok)
    print("\nSpecial token IDs (first match wins):")
    for k, (tid, name) in spec.items():
        print(f"{k}: {tid} ({name})")

    if args.strict_special_ids:
        assert tok.token_to_id("[PAD]") == 0
        assert tok.token_to_id("[UNK]") == 1
        assert tok.token_to_id("[BOS]") == 2
        assert tok.token_to_id("[EOS]") == 3
        print("[OK] strict special token IDs verified")


if __name__ == "__main__":
    main()
