"""Shared fixtures/helpers for the test suite. All tests run on CPU with tiny
models — no GPU required."""

import pytest
import torch

from src.model import GPTConfig


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
