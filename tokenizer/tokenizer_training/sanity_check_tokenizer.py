#!/usr/bin/env python3

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
4) Complete canonical production contract (32k, 7 specials, BPE/ByteLevel).
5) Literal-special-token injection hardening and atomic release manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from typing import Any

from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.special_tokens import (  # noqa: E402
    CANONICAL_VOCAB_SIZE,
    SPECIAL_TOKEN_IDS,
    assert_tokenizer_contract,
)

LEGACY_CONTRACT_WARNING = (
    "LEGACY ONLY: full production tokenizer-contract validation is disabled; "
    "this result must not be used to release a tokenizer or start production training."
)


def _json_loads(line: str) -> Any | None:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _encode_raw_text(tok: Tokenizer, text: str):
    """Encode ordinary text only after literal-special-token hardening is active."""
    if not bool(tok.encode_special_tokens):
        raise RuntimeError(
            "raw-text encoding requires tokenizer.encode_special_tokens=True"
        )
    return tok.encode(text)


def _per_file_sample_seed(base_seed: int, path: str) -> int:
    identity = str(Path(path).resolve())
    payload = json.dumps(
        [int(base_seed), identity], separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _sample_texts(
    texts: list[str],
    *,
    limit: int,
    base_seed: int,
    path: str,
) -> tuple[list[str], int]:
    """Select a stable, per-file sample without replacement."""
    file_seed = _per_file_sample_seed(base_seed, path)
    sample_size = min(int(limit), len(texts))
    indices = sorted(random.Random(file_seed).sample(range(len(texts)), sample_size))
    return [texts[index] for index in indices], file_seed


def _atomic_write_json(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    """Atomically replace a JSON report with a flushed same-directory temp file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=str(target.parent),
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, target)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _contract_report(
    tokenizer_json: dict[str, Any],
    tok: Tokenizer,
    *,
    strict_enabled: bool,
) -> dict[str, Any]:
    model = tokenizer_json.get("model") or {}
    actual_special_ids = {
        name: tok.token_to_id(name) for name in SPECIAL_TOKEN_IDS
    }
    return {
        "strict_validation_enabled": bool(strict_enabled),
        "production_contract_ok": bool(strict_enabled),
        "legacy_contract_bypass": not bool(strict_enabled),
        "requirements": {
            "vocab_size": CANONICAL_VOCAB_SIZE,
            "special_token_ids": dict(SPECIAL_TOKEN_IDS),
            "model": {"type": "BPE", "unk_token": "[UNK]"},
            "normalizer": None,
            "post_processor": None,
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False},
            "decoder": {"type": "ByteLevel"},
            "raw_text_encode_special_tokens": True,
        },
        "observed": {
            "vocab_size": int(tok.get_vocab_size(with_added_tokens=True)),
            "special_token_ids": actual_special_ids,
            "model": {
                "type": model.get("type"),
                "unk_token": model.get("unk_token"),
            },
            "normalizer": tokenizer_json.get("normalizer"),
            "post_processor": tokenizer_json.get("post_processor"),
            "pre_tokenizer": tokenizer_json.get("pre_tokenizer"),
            "decoder": tokenizer_json.get("decoder"),
            "raw_text_encode_special_tokens": bool(tok.encode_special_tokens),
        },
    }


def strict_roundtrip(tok: Tokenizer, s: str) -> None:
    enc = _encode_raw_text(tok, s)
    dec = tok.decode(enc.ids)
    if dec != s:
        raise AssertionError(
            "Round-trip failed!\n"
            f"orig={repr(s)}\n"
            f"tokens={enc.tokens}\n"
            f"ids={enc.ids}\n"
            f"dec ={repr(dec)}\n"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer.json")
    ap.add_argument(
        "--jsonl",
        nargs="*",
        default=[],
        help="Optional JSONL files for compression test (expects field text by default).",
    )
    ap.add_argument("--field", type=str, default="text", help="Text field for compression test.")
    ap.add_argument("--n_samples", type=int, default=2000, help="Max samples per input file.")
    ap.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base seed for deterministic, independent per-file sampling.",
    )
    ap.add_argument(
        "--report_json",
        type=str,
        default="",
        help="Atomically write the release manifest and compression statistics.",
    )
    ap.add_argument(
        "--strict_special_ids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate the complete canonical production tokenizer contract (default ON). "
        "--no-strict_special_ids is LEGACY ONLY and never production-ready.",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    if int(args.seed) < 0:
        raise ValueError("--seed must be non-negative")
    if int(args.n_samples) <= 0:
        raise ValueError("--n_samples must be positive")

    tokenizer_path = Path(args.tokenizer)
    tokenizer_sha256 = _sha256_file(tokenizer_path)
    with open(tokenizer_path, encoding="utf-8") as f:
        tokenizer_json = json.load(f)

    tok = Tokenizer.from_file(str(tokenizer_path))
    # Must be active before every ordinary encode below, including round-trip tests.
    tok.encode_special_tokens = True
    if not bool(tok.encode_special_tokens):
        raise RuntimeError("failed to enable raw-text special-token injection hardening")

    legacy_warning: str | None = None
    if args.strict_special_ids:
        assert_tokenizer_contract(tokenizer_path)
        print("[OK] complete canonical production tokenizer contract verified")
    else:
        legacy_warning = LEGACY_CONTRACT_WARNING
        print(f"[WARN] {legacy_warning}", file=sys.stderr)

    # --- 1) Strict round-trip tests ---
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
        "ｆｕｌｌ－ｗｉｄｔｈ：１２３ and ①②③ café ﬁle",
    ]
    for text in tests:
        strict_roundtrip(tok, text)
    print("[OK] strict round-trip tests passed")

    # --- 2) Qualitative inspection ---
    qualitative = [
        "Once upon a time, a robot said: \"Hello!\"",
        "### Task\n- Step 1: ...\n- Step 2: ...\n",
        "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n"
        "        a, b = b, a + b\n    return a\n",
        "User: Summarize the following.\n\nAssistant:",
        "The following is a news report:\n\n",
    ]
    for index, text in enumerate(qualitative, start=1):
        enc = _encode_raw_text(tok, text)
        dec = tok.decode(enc.ids)
        print(f"\n--- Qualitative #{index} ---")
        print("orig:", repr(text))
        print("tokens (first 60):", enc.tokens[:60])
        print("ids    (first 60):", enc.ids[:60])
        print("dec :", repr(dec))

    # --- 3) Compression ratio, overall + per input file ---
    compression: dict[str, dict[str, float | int | str]] = {}
    all_chars = 0
    all_tokens = 0
    all_samples = 0
    for path in args.jsonl:
        texts: list[str] = []
        with open(path, encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.rstrip("\r\n")
                if not line:
                    continue
                obj = _json_loads(line)
                if not isinstance(obj, dict):
                    continue
                text = obj.get(args.field)
                if isinstance(text, str) and text:
                    texts.append(text)
        if not texts:
            print(f"\n[WARN] compression test skipped for {path} (no usable texts)")
            continue

        samples, file_seed = _sample_texts(
            texts,
            limit=int(args.n_samples),
            base_seed=int(args.seed),
            path=path,
        )
        n_chars = sum(len(text) for text in samples)
        n_tokens = sum(len(_encode_raw_text(tok, text).ids) for text in samples)
        ratio = n_chars / max(n_tokens, 1)
        sample_sha256 = hashlib.sha256(
            json.dumps(samples, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        compression[path] = {
            "chars_per_token": ratio,
            "population_size": len(texts),
            "n_samples": len(samples),
            "n_chars": n_chars,
            "n_tokens": n_tokens,
            "sample_seed": file_seed,
            "sample_sha256": sample_sha256,
        }
        all_chars += n_chars
        all_tokens += n_tokens
        all_samples += len(samples)
        print(f"\n[OK] {path}: avg chars/token = {ratio:.3f}  (n={len(samples)})")

    if all_tokens:
        compression["__overall__"] = {
            "chars_per_token": all_chars / all_tokens,
            "n_samples": all_samples,
            "n_chars": all_chars,
            "n_tokens": all_tokens,
        }
        print(f"[OK] overall avg chars/token = {all_chars / all_tokens:.3f}")

    # --- 4) Special-token visibility ---
    actual_special_ids = {name: tok.token_to_id(name) for name in SPECIAL_TOKEN_IDS}
    print("\nSpecial token IDs:")
    for name, token_id in actual_special_ids.items():
        print(f"{name}: {token_id}")

    # --- 5) Literal-token injection probe ---
    probe = _encode_raw_text(tok, "literal [EOS] and <|assistant|> in text").ids
    special_ids = set(SPECIAL_TOKEN_IDS.values())
    hit = [token_id for token_id in probe if token_id in special_ids]
    assert not hit, f"special ids {hit} leaked from plain text with encode_special_tokens=True"
    print("[OK] literal special-token strings do not inject real ids")

    # --- 6) Release/report manifest ---
    contract = _contract_report(
        tokenizer_json,
        tok,
        strict_enabled=bool(args.strict_special_ids),
    )
    report: dict[str, Any] = {
        "schema_version": 2,
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": tokenizer_sha256,
        "vocab_size": int(tok.get_vocab_size(with_added_tokens=True)),
        "special_token_ids": actual_special_ids,
        "strict_special_ids_ok": bool(args.strict_special_ids),
        "production_ready": bool(args.strict_special_ids),
        "legacy_warning": legacy_warning,
        "roundtrip_ok": True,
        "injection_hardening_ok": True,
        "sampling": {
            "seed": int(args.seed),
            "max_samples_per_file": int(args.n_samples),
            "field": str(args.field),
            "method": "SHA256(base_seed,resolved_path) per file; sorted sample without replacement",
        },
        "contract": contract,
        "compression": compression,
    }
    if args.report_json:
        _atomic_write_json(args.report_json, report)
        print(f"[OK] atomically wrote report to {args.report_json}")
    return report


if __name__ == "__main__":
    main()
