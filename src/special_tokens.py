"""Canonical special-token IDs for the whole pipeline.

Every training/data script hardcodes these IDs (see CLAUDE.md):

    [PAD]=0  [UNK]=1  [BOS]=2  [EOS]=3
    <|system|>=4  <|user|>=5  <|assistant|>=6

The three role tokens delimit chat turns at the *token* level (see
src/chat_template.py) — BPE can never merge across a special token, so
training-time and inference-time encodings agree by construction.

`assert_special_token_ids` turns the silent assumption into a loud startup
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
SYSTEM_ID = 4
USER_ID = 5
ASSISTANT_ID = 6

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
SYSTEM_TOKEN = "<|system|>"
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"

SPECIAL_TOKEN_IDS: dict[str, int] = {
    PAD_TOKEN: PAD_ID,
    UNK_TOKEN: UNK_ID,
    BOS_TOKEN: BOS_ID,
    EOS_TOKEN: EOS_ID,
    SYSTEM_TOKEN: SYSTEM_ID,
    USER_TOKEN: USER_ID,
    ASSISTANT_TOKEN: ASSISTANT_ID,
}

# Ordered by ID — the exact list BpeTrainer must receive so IDs come out right.
SPECIAL_TOKENS: list[str] = sorted(SPECIAL_TOKEN_IDS, key=SPECIAL_TOKEN_IDS.get)


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
                f"tokenizer/tokenizer_training/train_tokenizer.py (its --strict_special_ids "
                f"default enforces this layout) or reconcile src/special_tokens.py."
            )
