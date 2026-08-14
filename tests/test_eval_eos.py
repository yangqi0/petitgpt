from __future__ import annotations

import json
import math

import pytest
import torch

from pretrain.eval_eos import (
    _build_prompt_ids,
    assert_exact_tokenizer_contract,
    atomic_write_json,
    classify_generation,
    finalize_teacher_forced_statistics,
    merge_teacher_forced_statistics,
    read_generation_cases,
    summarize_generation_records,
    teacher_forced_batch_statistics,
)
from src.special_tokens import SPECIAL_TOKEN_IDS


def _metric_fixture():
    # Six eligible positions: two true EOS targets and four internal targets.
    # EOS is top-1 at one of the true ends and one internal position.
    labels = torch.tensor([[3, 1, 2, 3], [0, 3, 4, 1]])
    loss_mask = torch.tensor([[1, 1, 0, 1], [1, 0, 1, 1]], dtype=torch.float32)
    logits = torch.tensor(
        [
            [
                [0.0, -1.0, -2.0, 3.0, 0.5],
                [0.0, 3.0, -1.0, 2.0, -2.0],
                [1.0, 0.0, 3.0, -1.0, 0.0],
                [4.0, 0.0, 0.0, 1.0, 0.0],
            ],
            [
                [0.0, -1.0, -2.0, 2.0, 1.0],
                [0.0, 0.0, 0.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, -1.0, 3.0],
                [0.0, 2.0, 0.0, -1.0, 0.0],
            ],
        ],
        dtype=torch.float32,
    )
    return logits, labels, loss_mask


def test_teacher_forced_statistics_have_exact_counts_and_probabilities():
    logits, labels, loss_mask = _metric_fixture()
    stats = teacher_forced_batch_statistics(logits, labels, loss_mask, eos_id=3)
    result = finalize_teacher_forced_statistics(stats)

    assert result["counts"] == {
        "serialized_positions": 8,
        "eligible_positions": 6,
        "masked_positions": 2,
        "true_document_end_positions": 2,
        "eligible_internal_positions": 4,
        "masked_eos_target_positions": 1,
        "true_eos_top1_positions": 1,
        "internal_eos_top1_positions": 1,
    }
    assert result["metrics"]["eos_top1_accuracy"] == 0.5
    assert result["metrics"]["internal_eos_top1_false_positive_rate"] == 0.25

    expected_p = torch.softmax(logits, dim=-1)[..., 3]
    target_p = torch.softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    eligible = loss_mask > 0
    true_end = (labels == 3) & (loss_mask > 0)
    internal = (labels != 3) & (loss_mask > 0)
    expected_ce = -torch.log(expected_p[true_end]).mean().item()
    expected_overall_ce = -torch.log(target_p[eligible]).mean().item()
    expected_non_eos_ce = -torch.log(target_p[internal]).mean().item()
    assert result["metrics"]["overall_supervised_cross_entropy"] == pytest.approx(
        expected_overall_ce
    )
    assert result["metrics"]["overall_supervised_perplexity"] == pytest.approx(
        math.exp(expected_overall_ce)
    )
    assert result["metrics"]["non_eos_cross_entropy"] == pytest.approx(expected_non_eos_ce)
    assert result["metrics"]["non_eos_perplexity"] == pytest.approx(
        math.exp(expected_non_eos_ce)
    )
    assert result["metrics"]["eos_only_cross_entropy"] == pytest.approx(expected_ce)
    assert result["metrics"]["mean_p_eos_true_document_end"] == pytest.approx(
        expected_p[true_end].mean().item()
    )
    assert result["metrics"]["mean_p_eos_eligible_internal"] == pytest.approx(
        expected_p[internal].mean().item()
    )


def test_teacher_forced_statistics_merge_is_additive_and_handles_empty_denominators():
    logits, labels, loss_mask = _metric_fixture()
    first = teacher_forced_batch_statistics(logits[:1], labels[:1], loss_mask[:1])
    second = teacher_forced_batch_statistics(logits[1:], labels[1:], loss_mask[1:])
    merged = finalize_teacher_forced_statistics(
        merge_teacher_forced_statistics([first, second])
    )
    whole = finalize_teacher_forced_statistics(
        teacher_forced_batch_statistics(logits, labels, loss_mask)
    )
    assert merged["counts"] == whole["counts"]
    assert merged["additive_sums"] == pytest.approx(whole["additive_sums"])
    assert merged["metrics"] == pytest.approx(whole["metrics"])

    no_end_logits = torch.zeros(1, 2, 5)
    no_end_labels = torch.tensor([[1, 2]])
    no_end_mask = torch.ones(1, 2)
    no_end = finalize_teacher_forced_statistics(
        teacher_forced_batch_statistics(no_end_logits, no_end_labels, no_end_mask)
    )
    assert no_end["metrics"]["eos_only_cross_entropy"] is None
    assert no_end["metrics"]["eos_top1_accuracy"] is None
    assert no_end["metrics"]["mean_p_eos_true_document_end"] is None
    assert math.isfinite(no_end["metrics"]["mean_p_eos_eligible_internal"])


def test_teacher_forced_statistics_reject_bad_shapes_and_bad_eos_id():
    logits = torch.zeros(1, 2, 5)
    labels = torch.zeros(1, 2, dtype=torch.long)
    mask = torch.ones(1, 2)
    with pytest.raises(ValueError, match="shape"):
        teacher_forced_batch_statistics(logits[0], labels, mask)
    with pytest.raises(ValueError, match="must match"):
        teacher_forced_batch_statistics(logits, labels[:, :1], mask)
    with pytest.raises(ValueError, match="outside vocab"):
        teacher_forced_batch_statistics(logits, labels, mask, eos_id=5)
    labels[0, 0] = 7
    with pytest.raises(ValueError, match="labels contains"):
        teacher_forced_batch_statistics(logits, labels, mask)


def test_generation_classification_observes_early_eos_instead_of_suppressing_it():
    early = classify_generation([9, 3], eos_id=3, min_tokens_before_eos=2)
    on_time = classify_generation([9, 8, 3], eos_id=3, min_tokens_before_eos=2)
    failure = classify_generation([9, 8, 7], eos_id=3, min_tokens_before_eos=2)

    assert early == {
        "stopped_by_eos": True,
        "failure_to_stop": False,
        "premature_eos": True,
        "first_eos_index": 1,
        "tokens_before_eos": 1,
        "tokens_after_first_eos": 0,
        "generated_tokens_including_eos": 2,
        "min_tokens_before_eos": 2,
    }
    assert on_time["stopped_by_eos"] and not on_time["premature_eos"]
    assert failure["failure_to_stop"] and not failure["premature_eos"]
    assert failure["tokens_before_eos"] == 3


def test_generation_summary_counts_rates_and_lengths():
    records = [
        classify_generation([3], min_tokens_before_eos=1),
        classify_generation([7, 8, 3], min_tokens_before_eos=2),
        classify_generation([7, 8, 9, 10], min_tokens_before_eos=2),
    ]
    summary = summarize_generation_records(records)
    assert summary["counts"] == {
        "examples": 3,
        "stopped_by_eos": 2,
        "premature_eos": 1,
        "failure_to_stop": 1,
    }
    assert summary["rates"]["eos_stop_rate"] == pytest.approx(2 / 3)
    assert summary["rates"]["premature_eos_rate"] == pytest.approx(1 / 3)
    assert summary["rates"]["premature_eos_rate_given_eos_stop"] == 0.5
    assert summary["rates"]["failure_to_stop_rate"] == pytest.approx(1 / 3)
    assert summary["length_excluding_terminal_eos"]["min"] == 0
    assert summary["length_excluding_terminal_eos"]["max"] == 4
    assert summary["length_excluding_terminal_eos"]["mean"] == 2.0
    assert summary["stopped_length_excluding_terminal_eos"]["mean"] == 1.0


def test_generation_jsonl_requires_per_example_threshold_and_forbids_ids(tmp_path):
    good = tmp_path / "good.jsonl"
    good.write_text(
        '\n'.join(
            [
                json.dumps(
                    {"id": "base", "prompt": "A complete paragraph", "min_tokens_before_eos": 4}
                ),
                json.dumps(
                    {
                        "id": "chat",
                        "messages": [{"role": "user", "content": "Explain gravity."}],
                        "min_tokens_before_eos": 8,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    cases = read_generation_cases(good, max_new_tokens=32)
    assert [case["id"] for case in cases] == ["base", "chat"]

    missing_threshold = tmp_path / "missing.jsonl"
    missing_threshold.write_text('{"prompt":"x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="must be frozen per example"):
        read_generation_cases(missing_threshold, max_new_tokens=32)

    injected_ids = tmp_path / "ids.jsonl"
    injected_ids.write_text(
        '{"prompt_ids":[2,5,6],"min_tokens_before_eos":1}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="forbidden"):
        read_generation_cases(injected_ids, max_new_tokens=32)


def test_atomic_json_output_replaces_complete_document(tmp_path):
    destination = tmp_path / "nested" / "report.json"
    atomic_write_json(destination, {"version": 1, "unicode": "文档"})
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "version": 1,
        "unicode": "文档",
    }
    atomic_write_json(destination, {"version": 2})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"version": 2}
    assert not list(destination.parent.glob("*.tmp"))


def test_exact_tokenizer_contract_rejects_an_eighth_special(tmp_path):
    vocab = dict(SPECIAL_TOKEN_IDS)
    vocab.update({f"token-{i}": i for i in range(7, 32_000)})
    added = [
        {"id": token_id, "content": token, "special": True}
        for token, token_id in SPECIAL_TOKEN_IDS.items()
    ]
    path = tmp_path / "tokenizer.json"
    path.write_text(
        json.dumps({"model": {"vocab": vocab}, "added_tokens": added}),
        encoding="utf-8",
    )
    assert_exact_tokenizer_contract(path)

    added.append({"id": 99, "content": "<|extra|>", "special": True})
    path.write_text(
        json.dumps({"model": {"vocab": vocab}, "added_tokens": added}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"exactly.*seven"):
        assert_exact_tokenizer_contract(path)


def test_raw_prompt_cannot_encode_to_a_control_id():
    class Encoding:
        ids = [11, 3, 12]

    class LeakyTokenizer:
        def encode(self, _text):
            return Encoding()

    case = {
        "id": "injection",
        "prompt": "literal [EOS] in untrusted text",
        "messages": None,
    }
    with pytest.raises(RuntimeError, match="encoded to control IDs"):
        _build_prompt_ids(LeakyTokenizer(), case, default_system="")
