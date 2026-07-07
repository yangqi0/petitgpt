"""Tests for the checked-in tokenizer and the project's hardcoded special-token
IDs (BOS=2, EOS=3, PAD=0). These IDs are hardcoded across every training script;
if a retrained tokenizer ever changes them, these tests fail instead of the
training loop silently mis-masking loss."""

from pathlib import Path

import pytest
from tokenizers import Tokenizer

TOKENIZER_PATH = Path(__file__).resolve().parent.parent / "tokenizer" / "tokenizer.json"

# Hardcoded in pretrain/SFT/distill/DPO scripts (see CLAUDE.md).
PAD_ID, BOS_ID, EOS_ID = 0, 2, 3


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    if not TOKENIZER_PATH.exists():
        pytest.skip(f"tokenizer not found at {TOKENIZER_PATH}")
    return Tokenizer.from_file(str(TOKENIZER_PATH))


def test_vocab_size_is_32k(tok):
    # 32k vocab fits in uint16, which the packed pretrain shards rely on.
    assert tok.get_vocab_size() == 32000
    assert tok.get_vocab_size() < 65536


def test_special_token_ids_are_stable(tok):
    assert tok.id_to_token(PAD_ID) == "[PAD]"
    assert tok.id_to_token(BOS_ID) == "[BOS]"
    assert tok.id_to_token(EOS_ID) == "[EOS]"


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
