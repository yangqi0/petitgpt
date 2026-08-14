"""Shared fixtures/helpers for the test suite. All tests run on CPU with tiny
models — no GPU required."""

import json

import pytest
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
import torch

from src.chat_template import configure_chat_tokenizer
from src.model import GPTConfig
from src.special_tokens import SPECIAL_TOKENS

_TINY_CORPUS = [
    "Hello world, how are you today?",
    "The quick brown fox jumps over the lazy dog.",
    "def add(a, b):\n    return a + b\n",
    "It is four. Hi there. Say hi. Bye now.",
    "You are a helpful assistant.",
    "What is two plus two? Write a poem about rain.",
    "System user assistant roles and [EOS] literal text.",
]


@pytest.fixture(scope="session")
def chat_tok() -> Tokenizer:
    """A tiny chat tokenizer trained in-process, mirroring
    tokenizer/tokenizer_training/train_tokenizer.py (byte-level BPE, no
    normalizer, the 7 special tokens at their canonical IDs). Keeps the test
    suite independent of the checked-in tokenizer artifact."""
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=400,
        min_frequency=1,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(_TINY_CORPUS, trainer=trainer)
    for i, t in enumerate(SPECIAL_TOKENS):
        assert tok.token_to_id(t) == i, f"special {t} got id {tok.token_to_id(t)}"
    return configure_chat_tokenizer(tok)


@pytest.fixture(scope="session")
def production_chat_tok(chat_tok: Tokenizer) -> Tokenizer:
    """Canonical 32k variant used by production data-pipeline guards."""
    payload = json.loads(chat_tok.to_str())
    vocab = payload["model"]["vocab"]
    next_id = max(vocab.values()) + 1
    for token_id in range(next_id, 32_000):
        vocab[f"__unused_test_token_{token_id}__"] = token_id
    tok = Tokenizer.from_str(json.dumps(payload))
    assert tok.get_vocab_size(with_added_tokens=True) == 32_000
    return tok


@pytest.fixture(autouse=True)
def _deterministic():
    """Make every test deterministic and fast."""
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(False)


@pytest.fixture
def tiny_cfg() -> GPTConfig:
    """A model small enough to build/forward/backward in milliseconds on CPU."""
    return GPTConfig(
        vocab_size=256,
        n_layers=3,
        d_model=64,
        n_heads=4,
        d_ff=160,
        max_seq_len=64,
        dropout=0.0,
        tie_embeddings=True,
    )
