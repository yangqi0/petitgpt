"""Tests for grpo/prepare_grpo_data.py record conversion and assembly."""

import pytest

from grpo.prepare_grpo_data import (
    code_bank_record_to_prompt,
    dedup_by_prompt,
    messages_record_to_prompt,
    split_train_val,
)


def test_code_bank_conversion_carries_tests_and_entry_point():
    rec = {
        "canonical_prompt": "Write a function that adds two numbers.",
        "entry_point": "add",
        "tests": ["assert add(1, 2) == 3"],
        "meta": {"from_mbpp": True},
    }
    out = code_bank_record_to_prompt(rec, tag="[Code] ")
    assert out["messages"] == [
        {"role": "user", "content": "[Code] Write a function that adds two numbers."}
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
    assert out["messages"][0]["content"] == "p"


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
    assert [m["content"] for m in out["messages"]] == ["q1", "a1", "q2"]


def test_messages_conversion_none_without_user():
    assert messages_record_to_prompt({"messages": [{"role": "system", "content": "s"}]}) is None
    assert messages_record_to_prompt({"messages": []}) is None


def test_dedup_by_prompt():
    a = {"messages": [{"role": "user", "content": "same"}]}
    b = {"messages": [{"role": "user", "content": "same"}]}
    c = {"messages": [{"role": "user", "content": "different"}]}
    out = dedup_by_prompt([a, b, c])
    assert len(out) == 2


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


def test_prepared_prompts_are_valid_grpo_input():
    """A converted record should be consumable by the GRPO trainer's renderer."""
    grpo = pytest.importorskip("grpo.grpo")
    rec = code_bank_record_to_prompt({
        "canonical_prompt": "add two ints",
        "entry_point": "add",
        "tests": ["assert add(1,1)==2"],
    })
    text = grpo.render_prompt_text(rec["messages"], default_system="You are helpful.")
    assert "User:" in text and text.rstrip().endswith("Assistant:")
