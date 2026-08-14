"""Tests for grpo/prepare_grpo_data.py record conversion and assembly."""

import pytest

from grpo.prepare_grpo_data import (
    code_bank_record_to_prompt,
    dedup_by_prompt,
    filter_by_prompt_tokens,
    messages_record_to_prompt,
    prompt_token_len,
    split_train_val,
)
from src.chat_template import DEFAULT_SYSTEM, encode_prompt as chat_encode_prompt


def test_code_bank_conversion_carries_tests_and_entry_point():
    rec = {
        "canonical_prompt": "Write a function that adds two numbers.",
        "entry_point": "add",
        "tests": ["assert add(1, 2) == 3"],
        "meta": {"from_mbpp": True},
    }
    out = code_bank_record_to_prompt(rec, tag="[Code] ")
    assert out["messages"] == [
        {"role": "system", "content": DEFAULT_SYSTEM},
        {"role": "user", "content": "[Code] Write a function that adds two numbers."},
    ]
    assert out["tests"] == ["assert add(1, 2) == 3"]
    assert out["entry_point"] == "add"
    assert out["meta"]["source"] == "code_bank"
    assert out["meta"]["from_mbpp"] is True


def test_code_bank_rejects_missing_tests_or_prompt():
    assert code_bank_record_to_prompt({"canonical_prompt": "do x"}) is None  # no tests
    assert code_bank_record_to_prompt({"tests": ["assert f()"]}) is None  # no prompt


def test_code_bank_entry_point_from_meta():
    rec = {"prompt": "p", "tests": ["assert g(1)==1"], "meta": {"entry_point": "g"}}
    out = code_bank_record_to_prompt(rec, tag="")
    assert out["entry_point"] == "g"
    assert out["messages"][-1]["content"] == "p"


def test_messages_conversion_drops_trailing_assistant():
    rec = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "SHOULD BE DROPPED"},
        ],
        "reference": "hello",
        "meta": {"bucket": "gen"},
    }
    out = messages_record_to_prompt(rec)
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["system", "user"]  # assistant dropped
    assert all(m["content"] != "SHOULD BE DROPPED" for m in out["messages"])
    assert out["reference"] == "hello"
    assert out["meta"]["source"] == "messages"


def test_messages_conversion_keeps_multiturn_up_to_last_user():
    rec = {
        "messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
    }
    out = messages_record_to_prompt(rec)
    assert [m["content"] for m in out["messages"]] == [
        DEFAULT_SYSTEM,
        "q1",
        "a1",
        "q2",
    ]


def test_messages_conversion_rejects_missing_user_or_illegal_roles():
    assert messages_record_to_prompt({}) is None
    with pytest.raises(ValueError, match="at least one non-empty user"):
        messages_record_to_prompt({"messages": [{"role": "system", "content": "s"}]})
    with pytest.raises(ValueError, match="at least one non-empty user"):
        messages_record_to_prompt({"messages": []})
    with pytest.raises(ValueError, match="role order"):
        messages_record_to_prompt({
            "messages": [
                {"role": "user", "content": "one"},
                {"role": "user", "content": "two"},
            ]
        })


def test_dedup_by_prompt():
    a = {"messages": [{"role": "user", "content": "same"}]}
    b = {"messages": [{"role": "user", "content": "same"}]}
    c = {"messages": [{"role": "user", "content": "different"}]}
    out = dedup_by_prompt([a, b, c])
    assert len(out) == 2


def test_prompt_token_length_uses_canonical_roles_and_system(chat_tok):
    record = code_bank_record_to_prompt({"prompt": "short prompt", "tests": ["assert True"]})
    canonical = chat_encode_prompt(
        chat_tok, record["messages"], default_system="", mode="full_context"
    )
    assert prompt_token_len(record, chat_tok, default_system="") == len(canonical)
    assert len(canonical) > len(chat_tok.encode("short prompt").ids)


def test_token_filter_rejects_prompt_that_cannot_fit_latest_user(chat_tok):
    record = code_bank_record_to_prompt({
        "prompt": "latest user content must remain completely intact",
        "tests": ["assert True"],
    })
    full = chat_encode_prompt(chat_tok, record["messages"], default_system="", mode="full_context")
    rejected = []
    kept = filter_by_prompt_tokens(
        [record],
        chat_tok,
        min_tokens=1,
        max_tokens=0,
        max_prompt_len=len(full) - 1,
        default_system="",
        rejected=rejected,
    )
    assert kept == []
    assert len(rejected) == 1
    assert "latest user-led suffix" in rejected[0]["error"]


def test_split_train_val_sizes_and_disjoint():
    recs = [{"messages": [{"role": "user", "content": f"q{i}"}]} for i in range(20)]
    train, val = split_train_val(recs, val_ratio=0.25, seed=0)
    assert len(val) == 5
    assert len(train) == 15
    train_keys = {m["messages"][0]["content"] for m in train}
    val_keys = {m["messages"][0]["content"] for m in val}
    assert train_keys.isdisjoint(val_keys)


def test_split_is_deterministic():
    recs = [{"messages": [{"role": "user", "content": f"q{i}"}]} for i in range(20)]
    assert split_train_val(recs, 0.25, seed=7) == split_train_val(recs, 0.25, seed=7)


def test_prepared_prompts_are_valid_grpo_input(chat_tok):
    """A converted record should be consumable by the GRPO trainer's encoder."""
    grpo = pytest.importorskip("grpo.grpo")
    from src.special_tokens import ASSISTANT_ID, BOS_ID, USER_ID

    rec = code_bank_record_to_prompt({
        "canonical_prompt": "add two ints",
        "entry_point": "add",
        "tests": ["assert add(1,1)==2"],
    })
    ids = grpo.encode_prompt(
        chat_tok, rec["messages"], default_system="You are helpful.", max_prompt_len=128
    )
    assert ids[0] == BOS_ID
    assert USER_ID in ids
    assert ids[-1] == ASSISTANT_ID
