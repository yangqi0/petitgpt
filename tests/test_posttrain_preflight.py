"""Fail-before-GPU post-training data validation tests."""

import json

import pytest

from dpo.dpo import preflight_dpo_record
from grpo.grpo import preflight_grpo_record
from sft.train_sft import preflight_sft_record
from src.posttrain_preflight import (
    require_preflight_passed,
    run_jsonl_preflight,
)


def test_jsonl_preflight_records_all_rejects_before_raising(tmp_path):
    data = tmp_path / "rows.jsonl"
    data.write_text(
        '{"value": 2}\n'
        "\n"
        "not-json\n"
        "[]\n"
        '{"value": -1}\n',
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"

    def validate(_split, record):
        if record["value"] < 0:
            raise ValueError("negative value")
        return {"value_sum": record["value"]}

    report = run_jsonl_preflight(
        stage="test",
        datasets={"train": data},
        validate_record=validate,
        report_path=report_path,
        max_recorded_errors=2,
    )

    assert report["status"] == "failed"
    assert report["total_records"] == 5
    assert report["total_valid"] == 1
    assert report["total_rejected"] == 4
    assert report["total_errors"] == 4
    assert report["errors_truncated"] is True
    assert report["splits"]["train"]["metrics"]["value_sum"] == 2
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    with pytest.raises(ValueError, match="preflight failed"):
        require_preflight_passed(report)


def test_jsonl_preflight_passes_and_aggregates_metrics(tmp_path):
    data = tmp_path / "rows.jsonl"
    data.write_text('{"n": 2}\n{"n": 3}\n', encoding="utf-8")
    report = run_jsonl_preflight(
        stage="test",
        datasets={"train": data},
        validate_record=lambda _split, record: {"tokens": record["n"]},
        report_path=tmp_path / "report.json",
    )
    require_preflight_passed(report)
    assert report["status"] == "passed"
    assert report["total_valid"] == 2
    assert report["splits"]["train"]["metrics"]["tokens"] == 5


def test_sft_preflight_requires_complete_assistant_turn(chat_tok):
    valid = {
        "messages": [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]
    }
    metrics = preflight_sft_record(
        "train",
        valid,
        tok=chat_tok,
        seq_len=64,
        default_system="Policy",
    )
    assert metrics["supervised_tokens"] >= 2

    with pytest.raises(ValueError, match="must end with a non-empty assistant"):
        preflight_sft_record(
            "train",
            {"messages": [{"role": "user", "content": "Question"}]},
            tok=chat_tok,
            seq_len=64,
            default_system="Policy",
        )


def test_dpo_preflight_rejects_whitespace_completion(chat_tok):
    with pytest.raises(ValueError, match="chosen completion.*non-whitespace"):
        preflight_dpo_record(
            "train",
            {
                "messages": [{"role": "user", "content": "Question"}],
                "chosen": " \r\n ",
                "rejected": "Valid answer",
            },
            tok=chat_tok,
            seq_len=64,
            default_system="Policy",
        )


def test_grpo_preflight_uses_reserved_generation_budget(chat_tok):
    example = {"messages": [{"role": "user", "content": "A complete prompt"}]}
    full = preflight_grpo_record(
        "train",
        example,
        tok=chat_tok,
        max_prompt_len=64,
        default_system="Policy",
    )
    with pytest.raises(ValueError, match="latest user-led suffix"):
        preflight_grpo_record(
            "train",
            example,
            tok=chat_tok,
            max_prompt_len=full["encoded_prompt_tokens"] - 1,
            default_system="Policy",
        )
