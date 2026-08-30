"""Focused R4 regressions for exact report-derived views (no training)."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from pretrain import pilot_contract_v2_3 as C, pilot_runner_v2_3 as R

# Reuse the established synthetic artifact-chain builders without copying their large fixtures.
tests_dir = str(Path(__file__).resolve().parent)
if tests_dir not in sys.path:
    sys.path.insert(0, tests_dir)
LEGACY = importlib.import_module("test_pilot_contract_v2_3")


def _rewrite_report(path: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    mutate(report)
    body = C.canonical_json_bytes(report)
    path.write_bytes(body)
    path.with_suffix(".sha256").write_text(
        f"{hashlib.sha256(body).hexdigest()}  {path.name}\n", encoding="utf-8"
    )


def _restore_report(path: Path, body: bytes, sidecar: bytes) -> None:
    path.write_bytes(body)
    path.with_suffix(".sha256").write_bytes(sidecar)


def _replace_immutable_json(path: Path, document: dict[str, Any]) -> str:
    body = C.canonical_json_bytes(document)
    digest = hashlib.sha256(body).hexdigest()
    path.write_bytes(body)
    path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def _publish_full_lr_chain(session: R.ExecutionSession) -> None:
    LEGACY._frozen_mb_report(session, micro_bsz=8, compile_on=False)
    scores = {2e-4: 3.0, 3e-4: 3.5, 4e-4: 3.9, 1e-4: 2.5}

    def launcher(candidate: dict[str, Any], given: R.ExecutionSession, plan: dict[str, Any]):
        LEGACY._write_result(
            candidate,
            given,
            plan,
            LEGACY._bound_lr_result(candidate, given, plan, scores[candidate["peak_lr"]]),
        )

    R.orchestrate_phase_muon_lr(session, launcher=launcher)


def test_mb_report_rejects_report_only_ledger_and_unknown_fields(tmp_path: Path) -> None:
    session = LEGACY._fake_session(tmp_path)
    LEGACY._frozen_mb_report(session)
    path = session.output_root / R.MB_REPORT_FILENAME
    original = path.read_bytes()
    original_sidecar = path.with_suffix(".sha256").read_bytes()
    mutations = (
        lambda report: report["ledger"].__setitem__(
            "completed_updates", report["ledger"]["completed_updates"] + 1
        ),
        lambda report: report.__setitem__("report_only_extension", "not-derived"),
    )
    for mutate in mutations:
        try:
            _rewrite_report(path, mutate)
            with pytest.raises(R.BindingFailure, match="raw candidate evidence"):
                R.load_authoritative_mb_report(session)
        finally:
            _restore_report(path, original, original_sidecar)


def test_every_lr_report_rejects_non_derived_full_document_changes(tmp_path: Path) -> None:
    session = LEGACY._fake_session(tmp_path)
    _publish_full_lr_chain(session)
    cases = (
        (
            R.LR_INITIAL_REPORT_FILENAME,
            R.load_authoritative_lr_initial_report,
            lambda report: report["ledger"].__setitem__(
                "completed_updates", report["ledger"]["completed_updates"] + 1
            ),
        ),
        (
            R.LR_CONFIRMATION_REPORT_FILENAME,
            R.load_authoritative_lr_confirmation_report,
            lambda report: report.__setitem__("report_only_extension", "not-derived"),
        ),
        (
            R.LR_EDGE_REPORT_FILENAME,
            R.load_authoritative_lr_edge_report,
            lambda report: report["ledger"].__setitem__("receipt_chain_head_sha256", "0" * 64),
        ),
        (
            R.LR_REPORT_FILENAME,
            R.load_authoritative_lr_final_report,
            lambda report: report.__setitem__("report_only_extension", "not-derived"),
        ),
    )

    # The current ledger has advanced beyond INITIAL/CONFIRMATION. Their canonical reports must
    # still validate from historical receipts, not from the latest aggregate snapshot.
    for _, loader, _ in cases:
        loader(session)

    for filename, loader, mutate in cases:
        path = session.output_root / filename
        original = path.read_bytes()
        original_sidecar = path.with_suffix(".sha256").read_bytes()
        try:
            _rewrite_report(path, mutate)
            with pytest.raises(R.BindingFailure, match="raw candidate evidence"):
                loader(session)
        finally:
            _restore_report(path, original, original_sidecar)


def test_complete_mb_failure_window_keeps_its_recomputed_metric(tmp_path: Path) -> None:
    session = LEGACY._fake_session(tmp_path)
    candidates, plan = LEGACY._published_mb_plan(session)
    candidate = candidates[0]
    result = LEGACY._bound_mb_result(
        candidate,
        session,
        plan,
        all_losses_finite=False,
    )
    LEGACY._write_result(candidate, session, plan, result)
    admitted = R.load_completed_result(session, planned=candidate, plan=plan)
    assert admitted["eligible"] is False
    assert admitted["median_update_tokens_per_second"] == pytest.approx(
        C.mb_median_update_tokens_per_second(result["update_timings"])
    )


def test_resealed_result_cannot_forge_its_pre_finalization_ledger_snapshot(
    tmp_path: Path,
) -> None:
    session = LEGACY._fake_session(tmp_path)
    candidates, plan = LEGACY._published_mb_plan(session)
    candidate = candidates[0]
    LEGACY._write_result(
        candidate,
        session,
        plan,
        LEGACY._bound_mb_result(candidate, session, plan),
    )

    output_dir = Path(candidate["output_dir"])
    result_path = output_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    snapshot = result["ledger_snapshot"]
    unit = C.TRAINED_TOKENS_PER_UPDATE
    snapshot["completed_updates"] -= 1
    snapshot["completed_tokens"]["MB"] -= unit
    snapshot["completed_tokens"]["GLOBAL"] -= unit
    snapshot["active_candidate"]["candidate_completed_updates"] -= 1
    resealed_result_sha256 = _replace_immutable_json(result_path, result)

    # Reseal the only receipt and chain head around the changed result bytes. This keeps the
    # durable ledger structurally valid and avoids a trivial stale-result-hash rejection.
    ledger_path = session.ledger.path
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert len(ledger["candidate_receipts"]) == 1
    receipt = ledger["candidate_receipts"][0]
    receipt["result_sha256"] = resealed_result_sha256
    receipt_body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    resealed_receipt_sha256 = hashlib.sha256(C.canonical_json_bytes(receipt_body)).hexdigest()
    receipt["receipt_sha256"] = resealed_receipt_sha256
    ledger["receipt_chain_head_sha256"] = resealed_receipt_sha256
    ledger_path.write_bytes(C.canonical_json_bytes(ledger))

    terminal_path = R.terminal_result_path(output_dir)
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["result_sha256"] = resealed_result_sha256
    terminal["ledger_receipt_sha256"] = resealed_receipt_sha256
    _replace_immutable_json(terminal_path, terminal)

    # The terminal, result hash, receipt, and ledger chain are mutually consistent; only the
    # result's claimed pre-finalization snapshot is false.
    validated_terminal = R.read_terminal_result(
        output_dir,
        expected=R.terminal_expectations(session, planned=candidate, plan=plan),
        ledger=session.ledger,
    )
    assert validated_terminal["result_sha256"] == resealed_result_sha256
    assert validated_terminal["ledger_receipt_sha256"] == resealed_receipt_sha256
    with pytest.raises(R.BindingFailure, match="receipt-derived pre-finalization"):
        R.load_completed_result(session, planned=candidate, plan=plan)
