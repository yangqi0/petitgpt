"""Canonical BOS/EOS inference contract tests."""

from __future__ import annotations

import pytest

from pretrain.sample import _assert_sampling_contract
from src.special_tokens import BOS_ID, EOS_ID


def test_sampling_contract_accepts_canonical_ids(production_chat_tok, tmp_path):
    path = tmp_path / "tokenizer.json"
    production_chat_tok.save(str(path))
    _assert_sampling_contract(str(path), eos_id=EOS_ID, add_bos=True, bos_id=BOS_ID)


@pytest.mark.parametrize(
    ("eos_id", "add_bos", "bos_id"),
    [
        (-1, True, BOS_ID),
        (EOS_ID, False, BOS_ID),
        (EOS_ID, True, 99),
    ],
)
def test_sampling_contract_rejects_noncanonical_controls(
    production_chat_tok, tmp_path, eos_id, add_bos, bos_id
):
    path = tmp_path / "tokenizer.json"
    production_chat_tok.save(str(path))
    with pytest.raises(ValueError, match="canonical sampling requires"):
        _assert_sampling_contract(
            str(path),
            eos_id=eos_id,
            add_bos=add_bos,
            bos_id=bos_id,
        )
