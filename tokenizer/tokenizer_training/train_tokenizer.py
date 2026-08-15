#!/usr/bin/env python3

"""
Train a Byte-Level BPE tokenizer for LLM pretraining/SFT.

Design goals
------------
1) Preserve whitespace AND unicode exactly (no normalizer, do NOT strip text
   fields), so decode(encode(x)) == x. Mainstream code tokenizers (GPT-4,
   Llama) do not normalize; NFKC silently rewrote full-width/compatibility
   characters, which corrupts string literals in code.
2) Avoid forced prefix-space behavior by default (add_prefix_space=False).
3) Keep BOS/EOS insertion out of the tokenizer (no post_processor by default).
   Add special tokens by ID exactly once in the data pipeline:
   - pretrain/build_pretrain_shards.py wraps each document as [BOS] doc [EOS];
     documents are separated by those special tokens ONLY (never "\\n\\n").
   - SFT/distill/DPO/GRPO use the token-level chat template in
     src/chat_template.py: [BOS] <|system|> ... <|user|> ... <|assistant|> ... [EOS]
     with a supervised EOS after every assistant answer.
4) Special tokens are single-source-of-truth in src/special_tokens.py:
   [PAD]=0 [UNK]=1 [BOS]=2 [EOS]=3 <|system|>=4 <|user|>=5 <|assistant|>=6.

Supported input JSONL formats
-----------------------------
Each line is a JSON object. This script can extract text from:
- Plain fields: {"text": "..."} or {"prompt": "...", "response": "..."}
- Chat messages: {"messages": [{"role":"user","content":"..."}, ...]}

Usage example
-------------
python train_tokenizer.py \
  --data datasets/tokenization/fineweb_sample.jsonl datasets/tokenization/tinystories_train.jsonl \
  --fields text \
  --vocab_size 32000 \
  --out_dir tokenizer \
  --min_freq 2
"""

from __future__ import annotations

import argparse
import atexit
from collections.abc import Iterator
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pretrain.build_pretrain_shards import (  # noqa: E402
    _write_json_atomic,
    clean_text,
    cleaned_text_sha256,
    load_exclusion_hash_manifest,
)
from src.special_tokens import (  # noqa: E402
    CANONICAL_VOCAB_SIZE,
    SPECIAL_TOKEN_IDS,
    SPECIAL_TOKENS,
    assert_tokenizer_contract,
)


def _json_loads(line: str) -> Any | None:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def _string_or_none(x: Any) -> str | None:
    return x if isinstance(x, str) and x != "" else None


def _render_messages(messages: list[dict[str, Any]]) -> str | None:
    """
    Convert chat-style messages into a single training string for BPE.

    Role special tokens are NOT written here on purpose — they are added
    tokens, invisible to BPE merge learning; this is just corpus text.
    """
    parts: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = _string_or_none(m.get("role"))
        content = _string_or_none(m.get("content"))
        if role is None or content is None:
            continue
        # Use a simple, stable delimiter. Do NOT strip content.
        parts.append(f"{role}:\n{content}")
    if not parts:
        return None
    return "\n\n".join(parts)


def iter_texts(
    paths: list[str],
    fields: list[str],
    messages_key: str = "messages",
    allow_messages: bool = True,
    exclusion_hash_sets: list[frozenset[str]] | None = None,
    exclusion_cleaning: dict[str, Any] | None = None,
    exclusion_stats: dict[str, Any] | None = None,
) -> Iterator[str]:
    """
    Stream text samples from one or more JSONL files.

    Important: we preserve whitespace and do NOT call strip() on text fields.
    """
    hash_sets = exclusion_hash_sets or []
    if hash_sets and exclusion_cleaning is None:
        raise ValueError("exclusion_cleaning is required with exclusion hash sets")
    stats = exclusion_stats if exclusion_stats is not None else {}
    stats.setdefault("considered_samples", 0)
    stats.setdefault("excluded_samples", 0)
    stats.setdefault("yielded_samples", 0)
    stats.setdefault("matched_per_manifest", [0 for _ in hash_sets])

    def should_exclude(text: str) -> bool:
        stats["considered_samples"] += 1
        if not hash_sets:
            stats["yielded_samples"] += 1
            return False
        assert exclusion_cleaning is not None
        required = {
            "strip_leading_noise",
            "normalize_quotes",
            "underscores_policy",
            "min_chars",
            "min_ascii_ratio",
        }
        if set(exclusion_cleaning) != required:
            raise ValueError(
                "exclusion manifest cleaning contract has unexpected keys: "
                f"{sorted(exclusion_cleaning)}"
            )
        cleaned = clean_text(
            text,
            strip_leading_noise=bool(exclusion_cleaning["strip_leading_noise"]),
            normalize_quotes=bool(exclusion_cleaning["normalize_quotes"]),
            underscores_policy=str(exclusion_cleaning["underscores_policy"]),
            min_chars=int(exclusion_cleaning["min_chars"]),
            min_ascii_ratio=float(exclusion_cleaning["min_ascii_ratio"]),
        )
        if cleaned is not None:
            value = cleaned_text_sha256(cleaned)
            matched = [index for index, hashes in enumerate(hash_sets) if value in hashes]
            if matched:
                stats["excluded_samples"] += 1
                for index in matched:
                    stats["matched_per_manifest"][index] += 1
                return True
        stats["yielded_samples"] += 1
        return False

    for p in paths:
        with open(p, encoding="utf-8") as f:
            for raw_line in f:
                # Only remove the trailing newline(s) from the JSONL file, not leading spaces.
                line = raw_line.rstrip("\r\n")
                if not line:
                    continue

                obj = _json_loads(line)
                if obj is None or not isinstance(obj, dict):
                    continue

                # 1) Chat-style: {"messages":[...]}
                if allow_messages and messages_key in obj and isinstance(obj[messages_key], list):
                    rendered = _render_messages(obj[messages_key])
                    if rendered is not None:
                        if not should_exclude(rendered):
                            yield rendered
                        continue

                # 2) Plain fields: concatenate in the given order
                parts: list[str] = []
                for k in fields:
                    v = _string_or_none(obj.get(k))
                    if v is not None:
                        parts.append(v)

                if parts:
                    text = "\n".join(parts)
                    if not should_exclude(text):
                        yield text


def build_tokenizer(add_prefix_space: bool) -> Tokenizer:
    """
    Build a Byte-Level BPE tokenizer.

    No normalizer: byte-level fallback already handles rare unicode, and any
    normalization (e.g. NFKC) breaks decode(encode(x)) == x for code.
    """
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=add_prefix_space)
    tok.decoder = ByteLevelDecoder()
    return tok


def maybe_add_post_processor(tok: Tokenizer, enabled: bool) -> None:
    """
    Optionally add BOS/EOS automatically.

    Recommended default: disabled. Add BOS/EOS exactly once in your data pipeline.
    """
    if not enabled:
        return
    bos_id = tok.token_to_id("[BOS]")
    eos_id = tok.token_to_id("[EOS]")
    assert bos_id is not None and eos_id is not None
    tok.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [EOS] $B:1 [EOS]:1",
        special_tokens=[("[BOS]", bos_id), ("[EOS]", eos_id)],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="Input JSONL files.")
    ap.add_argument(
        "--fields",
        nargs="+",
        default=["text"],
        help="JSON fields to concatenate when not using chat messages.",
    )
    ap.add_argument("--messages_key", type=str, default="messages", help="Chat messages key.")
    ap.add_argument("--no_messages", action="store_true", help="Disable chat messages parsing.")
    ap.add_argument("--vocab_size", type=int, default=32000, help="Target vocabulary size.")
    ap.add_argument("--min_freq", type=int, default=2, help="Minimum token frequency.")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory.")
    ap.add_argument(
        "--add_prefix_space",
        action="store_true",
        help="If set, use ByteLevel(add_prefix_space=True). Not recommended for strict round-trip.",
    )
    ap.add_argument(
        "--add_bos_eos_post_processor",
        action="store_true",
        help="If set, tokenizer will automatically add BOS/EOS via post_processor (usually avoid).",
    )
    ap.add_argument(
        "--strict_special_ids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Assert the src/special_tokens.py ID layout (default ON: all downstream "
        "scripts hardcode these IDs). --no-strict_special_ids to disable.",
    )
    ap.add_argument(
        "--model_max_length",
        type=int,
        default=2048,
        help="Write into tokenizer_config.json (informational).",
    )
    ap.add_argument(
        "--legacy_allow_noncanonical_contract",
        action="store_true",
        help=(
            "DEBUG/TEST ONLY: permit non-32k, prefix-space, post-processor, "
            "non-strict IDs, or missing reference exclusions. The manifest is "
            "marked noncanonical and production consumers reject it."
        ),
    )
    ap.add_argument(
        "--exclude_hash_manifest",
        type=Path,
        action="append",
        default=[],
        help=(
            "Repeatable pre-tokenizer reference-reserve exclusion manifest. "
            "Matching samples never reach BPE training."
        ),
    )
    args = ap.parse_args()

    contract_issues: list[str] = []
    if int(args.vocab_size) != CANONICAL_VOCAB_SIZE:
        contract_issues.append(
            f"vocab_size={args.vocab_size}, expected {CANONICAL_VOCAB_SIZE}"
        )
    if bool(args.add_prefix_space):
        contract_issues.append("add_prefix_space=True")
    if bool(args.add_bos_eos_post_processor):
        contract_issues.append("automatic BOS/EOS post-processor enabled")
    if not bool(args.strict_special_ids):
        contract_issues.append("strict special-token IDs disabled")
    if not args.exclude_hash_manifest:
        contract_issues.append("reference-reserve exclusion manifest missing")
    if contract_issues and not args.legacy_allow_noncanonical_contract:
        raise SystemExit(
            "noncanonical tokenizer release refused: " + "; ".join(contract_issues)
        )

    final_out_dir = Path(args.out_dir)
    if os.path.lexists(final_out_dir):
        raise FileExistsError(
            f"refusing to replace existing tokenizer output path: {final_out_dir}"
        )
    final_out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{final_out_dir.name}.building-",
            dir=str(final_out_dir.parent),
        )
    )

    def cleanup_staging() -> None:
        shutil.rmtree(staging_dir, ignore_errors=True)

    atexit.register(cleanup_staging)

    exclusion_hash_sets: list[frozenset[str]] = []
    exclusion_manifests: list[dict[str, Any]] = []
    exclusion_cleaning: dict[str, Any] | None = None
    for path in args.exclude_hash_manifest:
        hashes, metadata = load_exclusion_hash_manifest(
            path,
            expected_cleaning=exclusion_cleaning,
        )
        if exclusion_cleaning is None:
            exclusion_cleaning = metadata["cleaning"]
        exclusion_hash_sets.append(hashes)
        exclusion_manifests.append(metadata)

    tok = build_tokenizer(add_prefix_space=args.add_prefix_space)

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_freq,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    allow_messages = not args.no_messages
    exclusion_stats: dict[str, Any] = {}
    iterator = iter_texts(
        paths=args.data,
        fields=args.fields,
        messages_key=args.messages_key,
        allow_messages=allow_messages,
        exclusion_hash_sets=exclusion_hash_sets,
        exclusion_cleaning=exclusion_cleaning,
        exclusion_stats=exclusion_stats,
    )
    tok.train_from_iterator(iterator, trainer=trainer)

    # --- Validate special tokens ---
    tok2id = {t: tok.token_to_id(t) for t in SPECIAL_TOKENS}
    for t, tid in tok2id.items():
        assert tid is not None, f"Missing special token in vocab: {t}"
    assert len(set(tok2id.values())) == len(tok2id), f"Special token IDs are not unique: {tok2id}"

    if args.strict_special_ids:
        for t, expected in SPECIAL_TOKEN_IDS.items():
            assert tok2id[t] == expected, f"Expected {t}={expected}, got {tok2id[t]}"
    else:
        print("Special token IDs:", tok2id)

    maybe_add_post_processor(tok, enabled=args.add_bos_eos_post_processor)

    # --- Save artifacts ---
    tokenizer_path = str(staging_dir / "tokenizer.json")
    tok.save(tokenizer_path)
    if not contract_issues:
        assert_tokenizer_contract(tokenizer_path)

    with open(staging_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "model_max_length": args.model_max_length,
                "padding_side": "right",
                "truncation_side": "right",
                "bos_token": "[BOS]",
                "eos_token": "[EOS]",
                "unk_token": "[UNK]",
                "pad_token": "[PAD]",
                "additional_special_tokens": ["<|system|>", "<|user|>", "<|assistant|>"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(staging_dir / "special_tokens_map.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "bos_token": "[BOS]",
                "eos_token": "[EOS]",
                "unk_token": "[UNK]",
                "pad_token": "[PAD]",
                "additional_special_tokens": ["<|system|>", "<|user|>", "<|assistant|>"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    tokenizer_sha256 = hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest()
    for index, metadata in enumerate(exclusion_manifests):
        metadata["matched_samples"] = exclusion_stats["matched_per_manifest"][index]
    release_manifest = {
        "schema_version": 2,
        "kind": "petitgpt_tokenizer_release",
        "status": "complete",
        "contract": {
            "canonical": not contract_issues,
            "issues": contract_issues,
            "legacy_allow_noncanonical_contract": bool(
                args.legacy_allow_noncanonical_contract
            ),
        },
        "publication": "sibling_staging_then_atomic_rename",
        "tokenizer_sha256": tokenizer_sha256,
        "vocab_size": tok.get_vocab_size(),
        "special_token_ids": tok2id,
        "training": {
            "data": list(args.data),
            "fields": list(args.fields),
            "messages_key": args.messages_key,
            "allow_messages": allow_messages,
            "vocab_size_target": args.vocab_size,
            "min_frequency": args.min_freq,
            "add_prefix_space": args.add_prefix_space,
            "post_processor_enabled": args.add_bos_eos_post_processor,
        },
        "reference_reserve_exclusion": {
            "enabled": bool(exclusion_hash_sets),
            "hash_algorithm": (
                exclusion_manifests[0]["hash_algorithm"]
                if exclusion_manifests
                else None
            ),
            "cleaning": exclusion_cleaning,
            "manifest_count": len(exclusion_manifests),
            "union_hash_count": len(set().union(*exclusion_hash_sets))
            if exclusion_hash_sets
            else 0,
            "considered_samples": exclusion_stats["considered_samples"],
            "excluded_samples": exclusion_stats["excluded_samples"],
            "yielded_samples": exclusion_stats["yielded_samples"],
            "manifests": exclusion_manifests,
        },
    }
    _write_json_atomic(
        staging_dir / "tokenizer_release_manifest.json", release_manifest
    )

    if os.path.lexists(final_out_dir):
        raise FileExistsError(
            f"output path appeared during tokenizer training: {final_out_dir}"
        )
    os.rename(staging_dir, final_out_dir)
    atexit.unregister(cleanup_staging)

    print("Saved tokenizer to:", final_out_dir)
    print("vocab_size =", tok.get_vocab_size())
    print("add_prefix_space =", args.add_prefix_space)
    print("post_processor(BOS/EOS) =", args.add_bos_eos_post_processor)


if __name__ == "__main__":
    main()
