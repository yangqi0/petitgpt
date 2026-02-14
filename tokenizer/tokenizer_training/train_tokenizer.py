#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a Byte-Level BPE tokenizer for LLM pretraining/SFT.

Design goals
------------
1) Preserve whitespace exactly (do NOT strip text fields).
2) Avoid forced prefix-space behavior by default (add_prefix_space=False),
   so decode(encode(x)) == x for whitespace-sensitive strings like "\\n".
3) Keep BOS/EOS insertion out of the tokenizer (no post_processor by default).
   Add BOS/EOS in your dataset/sharding pipeline exactly once.

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
  --vocab_size 16000 \
  --out_dir tokenizer \
  --min_freq 2 \
  --strict_special_ids
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterator
from typing import Any

from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]


def _json_loads(line: str) -> Any | None:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def _string_or_none(x: Any) -> str | None:
    return x if isinstance(x, str) and x != "" else None


def _render_messages(messages: list[dict[str, Any]]) -> str | None:
    """
    Convert chat-style messages into a single training string.

    We intentionally keep formatting minimal and whitespace-stable.
    If you later standardize a chat template, you can update this function.
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
) -> Iterator[str]:
    """
    Stream text samples from one or more JSONL files.

    Important: we preserve whitespace and do NOT call strip() on text fields.
    """
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for lineno, raw_line in enumerate(f, start=1):
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
                        yield rendered
                        continue

                # 2) Plain fields: concatenate in the given order
                parts: list[str] = []
                for k in fields:
                    v = _string_or_none(obj.get(k))
                    if v is not None:
                        parts.append(v)

                if parts:
                    yield "\n".join(parts)


def build_tokenizer(add_prefix_space: bool) -> Tokenizer:
    """
    Build a Byte-Level BPE tokenizer.

    - NFKC helps normalize common unicode variants (full-width, compatibility chars).
    - ByteLevel makes it robust to code/symbols/rare unicode by falling back to bytes.
    """
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.normalizer = NFKC()
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
    ap.add_argument("--vocab_size", type=int, default=16000, help="Target vocabulary size.")
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
        action="store_true",
        help="Assert [PAD]=0,[UNK]=1,[BOS]=2,[EOS]=3. Enable if downstream code assumes fixed IDs.",
    )
    ap.add_argument(
        "--model_max_length",
        type=int,
        default=2048,
        help="Write into tokenizer_config.json (informational).",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tok = build_tokenizer(add_prefix_space=args.add_prefix_space)

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_freq,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    allow_messages = not args.no_messages
    iterator = iter_texts(
        paths=args.data,
        fields=args.fields,
        messages_key=args.messages_key,
        allow_messages=allow_messages,
    )
    tok.train_from_iterator(iterator, trainer=trainer)

    # --- Validate special tokens ---
    tok2id = {t: tok.token_to_id(t) for t in SPECIAL_TOKENS}
    for t, tid in tok2id.items():
        assert tid is not None, f"Missing special token in vocab: {t}"
    assert len(set(tok2id.values())) == len(tok2id), f"Special token IDs are not unique: {tok2id}"

    if args.strict_special_ids:
        assert tok2id["[PAD]"] == 0, f"Expected [PAD]=0, got {tok2id['[PAD]']}"
        assert tok2id["[UNK]"] == 1, f"Expected [UNK]=1, got {tok2id['[UNK]']}"
        assert tok2id["[BOS]"] == 2, f"Expected [BOS]=2, got {tok2id['[BOS]']}"
        assert tok2id["[EOS]"] == 3, f"Expected [EOS]=3, got {tok2id['[EOS]']}"
    else:
        print("Special token IDs:", tok2id)

    maybe_add_post_processor(tok, enabled=args.add_bos_eos_post_processor)

    # --- Save artifacts ---
    tokenizer_path = os.path.join(args.out_dir, "tokenizer.json")
    tok.save(tokenizer_path)

    with open(os.path.join(args.out_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
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
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(os.path.join(args.out_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "bos_token": "[BOS]",
                "eos_token": "[EOS]",
                "unk_token": "[UNK]",
                "pad_token": "[PAD]",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Saved tokenizer to:", args.out_dir)
    print("vocab_size =", tok.get_vocab_size())
    print("add_prefix_space =", args.add_prefix_space)
    print("post_processor(BOS/EOS) =", args.add_bos_eos_post_processor)


if __name__ == "__main__":
    main()
