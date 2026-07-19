"""Canonical special-token IDs for the whole pipeline.

Every training/data script hardcodes PAD=0, UNK=1, BOS=2, EOS=3 (see CLAUDE.md).
`assert_special_token_ids` turns that silent assumption into a loud startup
check: if the tokenizer at --tokenizer_path is ever retrained and the IDs move,
scripts fail immediately instead of training with a misaligned loss mask or a
broken EOS stop condition.
"""

from __future__ import annotations

import json

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

SPECIAL_TOKEN_IDS: dict[str, int] = {
    "[PAD]": PAD_ID,
    "[UNK]": UNK_ID,
    "[BOS]": BOS_ID,
    "[EOS]": EOS_ID,
}


def assert_special_token_ids(tokenizer_path: str) -> None:
    """Validate the hardcoded special-token IDs against a tokenizer.json file.

    Reads the file as plain JSON (no `tokenizers` import needed) so it is cheap
    to call from any script entry point.
    """
    with open(tokenizer_path, encoding="utf-8") as f:
        obj = json.load(f)
    vocab = (obj.get("model") or {}).get("vocab") or {}
    added = {t.get("content"): t.get("id") for t in obj.get("added_tokens") or []}
    for token, expected in SPECIAL_TOKEN_IDS.items():
        got = added.get(token, vocab.get(token))
        if got != expected:
            raise ValueError(
                f"special token {token!r} has id {got!r} in {tokenizer_path}, but the "
                f"pipeline hardcodes {expected}. Retrain the tokenizer with "
                f"--strict_special_ids (default) or reconcile src/special_tokens.py."
            )
