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
import os
import random
import sys
from typing import Any

from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.special_tokens import SPECIAL_TOKEN_IDS  # noqa: E402


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
        "--report_json",
        type=str,
        default="",
        help="Write the collected stats (vocab size, special IDs, per-file chars/token) "
        "to this JSON file — blog/README source material that otherwise only lands in stdout. "
        "Suggested: tokenizer/tokenizer_report.json",
    )
    ap.add_argument(
        "--strict_special_ids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Assert the src/special_tokens.py ID layout (default ON; the whole "
        "pipeline hardcodes it). --no-strict_special_ids to disable.",
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
        # unicode that a normalizer (e.g. NFKC) would silently rewrite:
        "ｆｕｌｌ－ｗｉｄｔｈ：１２３ and ①②③ café ﬁle",
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

    # --- 3) Compression ratio (chars per token), overall + per input file ---
    # Per-file ratios let you report e.g. English vs Python chars/token separately
    # by passing one jsonl per domain.
    compression: dict[str, dict[str, float | int]] = {}
    if args.jsonl:
        all_chars = 0
        all_tokens = 0
        for p in args.jsonl:
            texts: list[str] = []
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
            if not texts:
                print(f"\n[WARN] compression test skipped for {p} (no usable texts)")
                continue
            samples = random.sample(texts, min(args.n_samples, len(texts)))
            n_chars = sum(len(s) for s in samples)
            n_tokens = sum(len(tok.encode(s).ids) for s in samples)
            ratio = n_chars / max(n_tokens, 1)
            compression[p] = {"chars_per_token": ratio, "n_samples": len(samples)}
            all_chars += n_chars
            all_tokens += n_tokens
            print(f"\n[OK] {p}: avg chars/token = {ratio:.3f}  (n={len(samples)})")
        if all_tokens:
            compression["__overall__"] = {
                "chars_per_token": all_chars / all_tokens,
                "n_samples": sum(int(v["n_samples"]) for v in compression.values()),
            }
            print(f"[OK] overall avg chars/token = {all_chars / all_tokens:.3f}")

    # --- 4) Special tokens ---
    print("\nSpecial token IDs:")
    for name in SPECIAL_TOKEN_IDS:
        print(f"{name}: {tok.token_to_id(name)}")

    if args.strict_special_ids:
        for name, expected in SPECIAL_TOKEN_IDS.items():
            got = tok.token_to_id(name)
            assert got == expected, f"{name}: expected id {expected}, got {got}"
        print("[OK] strict special token IDs verified")

    # --- 5) Injection hardening (how the pipeline loads the tokenizer) ---
    tok.encode_special_tokens = True
    probe = tok.encode("literal [EOS] and <|assistant|> in text").ids
    special_ids = set(SPECIAL_TOKEN_IDS.values())
    hit = [i for i in probe if i in special_ids]
    assert not hit, f"special ids {hit} leaked from plain text with encode_special_tokens=True"
    print("[OK] literal special-token strings do not inject real ids")

    # --- 6) Optional JSON report (blog/README source material) ---
    if args.report_json:
        report = {
            "tokenizer": args.tokenizer,
            "vocab_size": tok.get_vocab_size(),
            "special_token_ids": {name: tok.token_to_id(name) for name in SPECIAL_TOKEN_IDS},
            "strict_special_ids_ok": bool(args.strict_special_ids),
            "roundtrip_ok": True,
            "injection_hardening_ok": True,
            "compression": compression,
        }
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[OK] wrote report to {args.report_json}")


if __name__ == "__main__":
    main()
