"""R5 regressions for authoritative LR loss-series reconstruction.

All evidence is synthetic and CPU-only. These tests publish or fully reseal the same immutable
result/receipt/terminal chains used by the R4 executor tests; they never construct a model or
perform an optimizer update.
"""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any

import pytest

import pretrain.pilot_contract_v2_3 as C
import pretrain.pilot_runner_v2_3 as R

tests_dir = str(Path(__file__).resolve().parent)
if tests_dir not in sys.path:
    sys.path.insert(0, tests_dir)
R4 = importlib.import_module("test_pilot_executor_r4")


def _calm_losses() -> dict[str, float]:
    return {str(update): 3.0 for update in range(1, C.LR_RUN_UPDATES + 1)}


def _divergent_losses() -> dict[str, float]:
    losses = _calm_losses()
    for update in range(61, C.LR_RUN_UPDATES + 1):
        losses[str(update)] = 6.0
    return losses


def _canonical_detail(losses: dict[str, float]) -> dict[str, Any]:
    return C.sustained_divergence({int(update): loss for update, loss in losses.items()})


def _published_valid_chain(tmp_path: Path):
    session = R4._artifact_bound_session(tmp_path / "authorized")  # noqa: SLF001
    candidates, plan = R4._published_lr_plan(session, [2e-4])  # noqa: SLF001
    candidate = candidates[0]
    R4._publish_synthetic_lr_chain(candidate, session, plan, eligible=True)  # noqa: SLF001
    return session, candidates, candidate, plan


def _rebind(
    session: R.ExecutionSession,
    candidates: list[dict[str, Any]],
    candidate: dict[str, Any],
    plan: dict[str, Any],
    mutate: Callable[[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]], None],
) -> None:
    R4._rebind_candidate_chains(session, candidates, mutate)  # noqa: SLF001
    R4._assert_chain_is_rehashed(session, candidate, plan)  # noqa: SLF001


def test_valid_complete_loss_series_remains_admissible(tmp_path: Path) -> None:
    session, _, candidate, plan = _published_valid_chain(tmp_path)

    admitted = R.load_completed_result(session, planned=candidate, plan=plan)

    expected_detail = _canonical_detail(_calm_losses())
    assert admitted["eligible"] is True
    assert admitted["all_losses_finite"] is True
    assert admitted["sustained_divergence"] is False
    assert admitted["divergence_detail"] == expected_detail


def test_divergent_raw_series_falsely_labelled_healthy_is_rejected(tmp_path: Path) -> None:
    session, candidates, candidate, plan = _published_valid_chain(tmp_path)

    def forge_raw_losses(_candidate, result, _meta, _terminal):
        result["losses_by_update"] = _divergent_losses()

    _rebind(session, candidates, candidate, plan, forge_raw_losses)

    with pytest.raises(
        R.BindingFailure,
        match=r"stored sustained_divergence False disagrees.*True",
    ):
        R.load_completed_result(session, planned=candidate, plan=plan)


def test_healthy_raw_series_falsely_labelled_divergent_is_rejected(tmp_path: Path) -> None:
    session, candidates, candidate, plan = _published_valid_chain(tmp_path)

    def forge_stored_summary(_candidate, result, _meta, terminal):
        result["sustained_divergence"] = True
        result["divergence_detail"] = _canonical_detail(_divergent_losses())
        result["eligible"] = False
        result["eligibility_failures"] = ["sustained_divergence"]
        result["terminal_status"] = "CANDIDATE_INELIGIBLE"
        terminal["terminal_status"] = "CANDIDATE_INELIGIBLE"

    _rebind(session, candidates, candidate, plan, forge_stored_summary)

    with pytest.raises(
        R.BindingFailure,
        match=r"stored sustained_divergence True disagrees.*False",
    ):
        R.load_completed_result(session, planned=candidate, plan=plan)


def test_healthy_raw_series_cannot_be_excluded_by_stored_finiteness(tmp_path: Path) -> None:
    session, candidates, candidate, plan = _published_valid_chain(tmp_path)

    def forge_stored_summary(_candidate, result, _meta, terminal):
        result["all_losses_finite"] = False
        result["eligible"] = False
        result["eligibility_failures"] = ["non_finite_loss"]
        result["terminal_status"] = "CANDIDATE_INELIGIBLE"
        terminal["terminal_status"] = "CANDIDATE_INELIGIBLE"

    _rebind(session, candidates, candidate, plan, forge_stored_summary)

    with pytest.raises(
        R.BindingFailure,
        match=r"stored all_losses_finite False disagrees.*True",
    ):
        R.load_completed_result(session, planned=candidate, plan=plan)


def test_null_raw_loss_hidden_by_stored_finite_boolean_is_rejected(tmp_path: Path) -> None:
    session, candidates, candidate, plan = _published_valid_chain(tmp_path)

    def hide_null(_candidate, result, _meta, _terminal):
        result["losses_by_update"]["100"] = None

    _rebind(session, candidates, candidate, plan, hide_null)

    with pytest.raises(R.BindingFailure, match=r"non-numeric value.*100"):
        R.load_completed_result(session, planned=candidate, plan=plan)


def test_divergence_detail_must_equal_the_canonical_recomputation(tmp_path: Path) -> None:
    session, candidates, candidate, plan = _published_valid_chain(tmp_path)

    def forge_detail(_candidate, result, _meta, _terminal):
        result["divergence_detail"]["threshold"] += 1.0

    _rebind(session, candidates, candidate, plan, forge_detail)

    with pytest.raises(R.BindingFailure, match="stored divergence_detail disagrees"):
        R.load_completed_result(session, planned=candidate, plan=plan)


def test_divergence_detail_rejects_boolean_integer_type_confusion(tmp_path: Path) -> None:
    session, candidates, candidate, plan = _published_valid_chain(tmp_path)

    def forge_detail_type(_candidate, result, _meta, _terminal):
        result["divergence_detail"]["diverged"] = 0

    _rebind(session, candidates, candidate, plan, forge_detail_type)

    with pytest.raises(R.BindingFailure, match="stored divergence_detail disagrees"):
        R.load_completed_result(session, planned=candidate, plan=plan)


def _remove_update(_candidate, result, _meta, _terminal):
    del result["losses_by_update"]["100"]


def _add_extra_update(_candidate, result, _meta, _terminal):
    result["losses_by_update"]["201"] = 3.0


def _add_duplicate_update_alias(_candidate, result, _meta, _terminal):
    result["losses_by_update"]["0100"] = 3.0


def _add_oversized_decimal_update(_candidate, result, _meta, _terminal):
    result["losses_by_update"]["1" * 5000] = 3.0


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_remove_update, r"exactly updates 1\.\.200; missing=\[100\]"),
        (_add_extra_update, r"exactly updates 1\.\.200; missing=\[\], extra=\[201\]"),
        (_add_duplicate_update_alias, r"duplicate update.*100"),
        (_add_oversized_decimal_update, r"malformed update key"),
    ],
    ids=(
        "missing_update",
        "extra_update",
        "duplicate_update_alias",
        "oversized_decimal_update",
    ),
)
def test_loss_series_requires_exactly_one_entry_for_updates_1_to_200(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]], None],
    message: str,
) -> None:
    session, candidates, candidate, plan = _published_valid_chain(tmp_path)
    _rebind(session, candidates, candidate, plan, mutate)

    with pytest.raises(R.BindingFailure, match=message):
        R.load_completed_result(session, planned=candidate, plan=plan)


@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param("3.0", id="string"),
        pytest.param(True, id="boolean"),
        pytest.param(None, id="null"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive_infinity"),
        pytest.param(float("-inf"), id="negative_infinity"),
    ],
)
def test_raw_loss_values_must_be_numeric_finite_non_booleans(invalid: Any) -> None:
    losses: dict[str, Any] = _calm_losses()
    losses["100"] = invalid
    result = {
        "candidate_id": "lr_0p0002_seed1",
        "losses_by_update": losses,
        "all_losses_finite": True,
        "sustained_divergence": False,
        "divergence_detail": _canonical_detail(_calm_losses()),
    }

    with pytest.raises(R.BindingFailure, match=r"(non-numeric|non-finite) value"):
        R.verify_recomputed_lr_loss_result(result)


def test_literal_duplicate_json_update_key_is_rejected_after_full_chain_rehash(
    tmp_path: Path,
) -> None:
    session, _, candidate, plan = _published_valid_chain(tmp_path)
    output = Path(candidate["output_dir"])
    result_path = output / "result.json"
    original = result_path.read_bytes()
    marker = b'"losses_by_update":{'
    assert original.count(marker) == 1
    forged = original.replace(marker, marker + b'"100":3.0,', 1)
    result_sha256 = hashlib.sha256(forged).hexdigest()
    result_path.write_bytes(forged)
    result_path.with_suffix(".sha256").write_text(
        f"{result_sha256}  {result_path.name}\n", encoding="utf-8"
    )

    ledger = session.ledger
    with ledger._lock():  # noqa: SLF001
        ledger._reload_locked()  # noqa: SLF001
        receipt = ledger.state["candidate_receipts"][0]
        receipt["result_sha256"] = result_sha256
        receipt_body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        receipt_sha256 = hashlib.sha256(C.canonical_json_bytes(receipt_body)).hexdigest()
        receipt["receipt_sha256"] = receipt_sha256
        ledger.state["receipt_chain_head_sha256"] = receipt_sha256
        ledger._require_structural_invariants(ledger.state)  # noqa: SLF001
        ledger._write(ledger.state)  # noqa: SLF001

    terminal_path = R.terminal_result_path(output)
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["result_sha256"] = result_sha256
    terminal["ledger_receipt_sha256"] = receipt_sha256
    R4._adversarially_replace_immutable(terminal_path, terminal)  # noqa: SLF001

    R4._assert_chain_is_rehashed(session, candidate, plan)  # noqa: SLF001
    with pytest.raises(R.BindingFailure, match="bytes are not canonical JSON"):
        R.load_completed_result(session, planned=candidate, plan=plan)


def test_finite_raw_losses_with_nonfinite_derived_threshold_are_rejected(
    tmp_path: Path,
) -> None:
    session, candidates, candidate, plan = _published_valid_chain(tmp_path)

    def overflow_threshold(_candidate, result, _meta, _terminal):
        result["losses_by_update"] = {
            str(update): 1e308 for update in range(1, C.LR_RUN_UPDATES + 1)
        }

    _rebind(session, candidates, candidate, plan, overflow_threshold)

    with pytest.raises(R.BindingFailure, match="divergence_detail is not canonical JSON"):
        R.load_completed_result(session, planned=candidate, plan=plan)
