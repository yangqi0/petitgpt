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
from os import PathLike

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
SYSTEM_ID = 4
USER_ID = 5
ASSISTANT_ID = 6
CANONICAL_VOCAB_SIZE = 32_000

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
    """Validate the exact registered-special-token contract.

    Reads the file as plain JSON (no `tokenizers` import needed) so it is cheap
    to call from any script entry point. Merely finding the strings in the BPE
    vocabulary is insufficient: all seven must be registered as special tokens,
    and no eighth registered special token is allowed.
    """
    with open(tokenizer_path, encoding="utf-8") as f:
        obj = json.load(f)

    special_entries = [
        entry
        for entry in (obj.get("added_tokens") or [])
        if isinstance(entry, dict) and entry.get("special") is True
    ]
    registered_tokens = [entry.get("content") for entry in special_entries]
    if len(special_entries) != len(SPECIAL_TOKEN_IDS) or set(registered_tokens) != set(
        SPECIAL_TOKEN_IDS
    ):
        raise ValueError(
            f"{tokenizer_path} must register exactly these seven special tokens: "
            f"{list(SPECIAL_TOKEN_IDS)}; got {registered_tokens!r}"
        )

    added = {entry["content"]: entry.get("id") for entry in special_entries}
    for token, expected in SPECIAL_TOKEN_IDS.items():
        got = added.get(token)
        if got != expected:
            raise ValueError(
                f"special token {token!r} has id {got!r} in {tokenizer_path}, but the "
                f"pipeline hardcodes {expected}. Retrain the tokenizer with "
                f"tokenizer/tokenizer_training/train_tokenizer.py (its --strict_special_ids "
                f"default enforces this layout) or reconcile src/special_tokens.py."
            )


def assert_tokenizer_contract(tokenizer_path: str | PathLike[str]) -> None:
    """Fail fast unless ``tokenizer.json`` is the canonical production artifact.

    This is deliberately stricter than :func:`assert_special_token_ids`, which
    remains useful for tiny unit-test tokenizers. Production entry points must
    call this function so a valid-looking seven-token map cannot hide an old
    vocabulary, normalizer, prefix-space rule, or automatic BOS/EOS processor.
    """
    path = str(tokenizer_path)
    assert_special_token_ids(path)
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)

    model = obj.get("model") or {}
    if model.get("type") != "BPE":
        raise ValueError(f"{path} must use a BPE model; got {model.get('type')!r}")
    if model.get("unk_token") != UNK_TOKEN:
        raise ValueError(
            f"{path} BPE unk_token must be {UNK_TOKEN!r}; got {model.get('unk_token')!r}"
        )

    model_vocab = model.get("vocab") or {}
    token_ids = set(model_vocab.values())
    token_ids.update(
        entry.get("id")
        for entry in (obj.get("added_tokens") or [])
        if isinstance(entry, dict) and isinstance(entry.get("id"), int)
    )
    expected_ids = set(range(CANONICAL_VOCAB_SIZE))
    if token_ids != expected_ids:
        raise ValueError(
            f"{path} runtime vocab IDs must be exactly 0..{CANONICAL_VOCAB_SIZE - 1} "
            f"(vocab_size={CANONICAL_VOCAB_SIZE}); got {len(token_ids)} unique IDs"
        )

    if obj.get("normalizer") is not None:
        raise ValueError(f"{path} must not configure a tokenizer normalizer")
    if obj.get("post_processor") is not None:
        raise ValueError(f"{path} must not configure an automatic BOS/EOS post-processor")

    pre = obj.get("pre_tokenizer") or {}
    if pre.get("type") != "ByteLevel" or pre.get("add_prefix_space") is not False:
        raise ValueError(
            f"{path} pre_tokenizer must be ByteLevel(add_prefix_space=False); got {pre!r}"
        )
    decoder = obj.get("decoder") or {}
    if decoder.get("type") != "ByteLevel":
        raise ValueError(f"{path} decoder must be ByteLevel; got {decoder!r}")
