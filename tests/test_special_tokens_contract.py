"""Regression tests for the exact seven-special-token repository contract."""

from __future__ import annotations

import json

import pytest

from src.special_tokens import assert_special_token_ids, assert_tokenizer_contract


def _save(chat_tok, path):
    chat_tok.save(str(path))
    return path


def test_exact_canonical_special_set_passes(chat_tok, tmp_path):
    path = _save(chat_tok, tmp_path / "tokenizer.json")
    assert_special_token_ids(str(path))


def test_full_production_contract_passes(production_chat_tok, tmp_path):
    path = _save(production_chat_tok, tmp_path / "tokenizer.json")
    assert_tokenizer_contract(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("normalizer", {"type": "NFKC"}, "must not configure.*normalizer"),
        (
            "post_processor",
            {"type": "TemplateProcessing", "single": [], "pair": []},
            "must not configure.*post-processor",
        ),
    ],
)
def test_full_contract_rejects_normalizer_or_postprocessor(
    production_chat_tok, tmp_path, field, value, message
):
    path = _save(production_chat_tok, tmp_path / "tokenizer.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        assert_tokenizer_contract(path)


def test_full_contract_rejects_prefix_space(production_chat_tok, tmp_path):
    path = _save(production_chat_tok, tmp_path / "tokenizer.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["pre_tokenizer"]["add_prefix_space"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="add_prefix_space=False"):
        assert_tokenizer_contract(path)


def test_full_contract_rejects_non_32k_vocab(chat_tok, tmp_path):
    path = _save(chat_tok, tmp_path / "tokenizer.json")
    with pytest.raises(ValueError, match="vocab_size=32000"):
        assert_tokenizer_contract(path)


def test_extra_registered_special_is_rejected(chat_tok, tmp_path):
    path = _save(chat_tok, tmp_path / "tokenizer.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["added_tokens"].append(
        {
            "id": 399,
            "content": "<|extra|>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly these seven"):
        assert_special_token_ids(str(path))


def test_role_marker_must_be_registered_special(chat_tok, tmp_path):
    path = _save(chat_tok, tmp_path / "tokenizer.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    role = next(
        entry for entry in payload["added_tokens"] if entry["content"] == "<|assistant|>"
    )
    role["special"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly these seven"):
        assert_special_token_ids(str(path))
