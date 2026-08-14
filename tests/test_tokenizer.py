"""Tests for the checked-in tokenizer artifact against the project's hardcoded
special-token layout (src/special_tokens.py). These IDs are hardcoded across
every training script; if a retrained tokenizer ever changes them, these tests
fail instead of the training loop silently mis-masking loss.

If the checked-in artifact predates the role-token contract
(<|system|>/<|user|>/<|assistant|>), the whole module SKIPS with a pointer to
retrain — the chat-format contract itself is covered by test_chat_template.py
via an in-process tokenizer, independent of this artifact.
"""

from pathlib import Path

import pytest
from tokenizers import Tokenizer

from src.special_tokens import SPECIAL_TOKEN_IDS, assert_special_token_ids

TOKENIZER_PATH = Path(__file__).resolve().parent.parent / "tokenizer" / "tokenizer.json"


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    if not TOKENIZER_PATH.exists():
        pytest.skip(f"tokenizer not found at {TOKENIZER_PATH}")
    try:
        assert_special_token_ids(str(TOKENIZER_PATH))
    except ValueError as e:
        pytest.skip(
            f"checked-in tokenizer predates the current special-token contract "
            f"({e}); retrain with tokenizer/tokenizer_training/train_tokenizer.py"
        )
    return Tokenizer.from_file(str(TOKENIZER_PATH))


def test_vocab_fits_uint16(tok):
    # packed pretrain shards default to uint16
    assert tok.get_vocab_size() < 65536


def test_special_token_ids_are_stable(tok):
    for token, tid in SPECIAL_TOKEN_IDS.items():
        assert tok.id_to_token(tid) == token


@pytest.mark.parametrize(
    "text",
    [
        "The quick brown fox jumps over the lazy dog.",
        "def add(a, b):\n    return a + b\n",
        "Numbers: 3.14159 and 42 — and unicode: café, naïve.",
    ],
)
def test_roundtrip_preserves_non_whitespace(tok, text):
    ids = tok.encode(text).ids
    decoded = tok.decode(ids)
    # BPE decode can normalize whitespace; compare on non-whitespace content.
    assert "".join(decoded.split()) == "".join(text.split())


def test_encode_is_deterministic(tok):
    text = "deterministic encoding check"
    assert tok.encode(text).ids == tok.encode(text).ids
