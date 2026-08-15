from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

import pretrain.manual_review_python_p1 as manual_review
from pretrain.manual_review_python_p1 import (
    ArmInputs,
    ManualReviewConfig,
    ManualReviewError,
    ReviewItem,
    VerifiedInputs,
)


def _synthetic_verified_inputs() -> VerifiedInputs:
    items: list[ReviewItem] = []
    ordinal = 1
    for arm in ("primary", "stack_comparison"):
        for automatic_outcome in ("keep", "reject"):
            for sample in range(12):
                items.append(
                    ReviewItem(
                        review_id=f"mrv2-{ordinal:04d}",
                        arm=arm,
                        automatic_outcome=automatic_outcome,
                        selection_rank_sha256=f"{ordinal:064x}",
                        presentation_sha256=f"{ordinal + 1000:064x}",
                        raw=(
                            "raw-source-secret "
                            f"synthetic_{arm}_{automatic_outcome}_{sample} = {ordinal}\n"
                        ).encode(),
                    )
                )
                ordinal += 1

    spec_sha256 = "a" * 64
    spec = {
        "manual_attestation": {"review_session_id": "synthetic-session-01"},
        "outputs": {
            "queue_stem": "python-p1-manual-review-queue",
            "attestation_stem": "python-p1-manual-review-attestation",
            "result_stem": "python-p1-manual-review-result",
        },
    }
    queue = {
        "schema_version": 2,
        "kind": manual_review.QUEUE_KIND,
        "status": "READY_FOR_BLINDED_REVIEW",
        "decision_scope": manual_review.DECISION_SCOPE,
        "review_session_id": spec["manual_attestation"]["review_session_id"],
        "spec_sha256": spec_sha256,
        "record_count": 48,
        "records": [{"review_id": item.review_id} for item in items],
    }
    return VerifiedInputs(
        spec=spec,
        spec_sha256=spec_sha256,
        items=tuple(items),
        queue=queue,
        sensitive_values=frozenset({"raw-source-secret", "synthetic-private-identifier"}),
    )


def _synthetic_config(root: Path, *, session_name: str = "session") -> ManualReviewConfig:
    arm = lambda role: ArmInputs(  # noqa: E731 - compact inert synthetic config.
        role=role,
        manifest=root / f"unused-{role}-manifest.json",
        collection_report=root / f"unused-{role}-collection.json",
        replay_report=root / f"unused-{role}-replay.json",
        analysis_report=root / f"unused-{role}-analysis.json",
        cache_dir=root / f"unused-{role}-cache",
    )
    return ManualReviewConfig(
        spec_path=root / "unused-spec.json",
        policy_path=root / "unused-policy.json",
        comparison_report=root / "unused-comparison.json",
        primary=arm("primary"),
        stack_comparison=arm("stack_comparison"),
        session_dir=root / session_name,
        expected_generator_commit="0" * 40,
        enforce_environment=False,
        enforce_frozen_spec=False,
    )


@pytest.fixture
def synthetic_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ManualReviewConfig, VerifiedInputs]:
    verified = _synthetic_verified_inputs()
    config = _synthetic_config(tmp_path)
    monkeypatch.setattr(manual_review, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(manual_review, "verify_inputs", lambda _config: verified)
    return config, verified


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _expected_label(item: ReviewItem) -> str:
    return "MANUAL_KEEP" if item.automatic_outcome == "keep" else "MANUAL_REJECT"


def _assert_private_modes(session_dir: Path) -> None:
    assert stat.S_IMODE(session_dir.stat().st_mode) == 0o700
    for path in session_dir.iterdir():
        assert path.is_file()
        assert not path.is_symlink()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_prepare_review_unblind_persists_only_blinded_and_aggregate_data(
    synthetic_review: tuple[ManualReviewConfig, VerifiedInputs],
):
    config, verified = synthetic_review
    expected_by_id = {item.review_id: _expected_label(item) for item in verified.items}
    shown: list[tuple[str, str]] = []

    queue_path = manual_review.prepare_queue(config)
    queue = _read_json(queue_path)
    assert queue["record_count"] == 48
    assert all(set(record) == {"review_id"} for record in queue["records"])  # type: ignore[union-attr]
    assert [record["review_id"] for record in queue["records"]] == [  # type: ignore[index]
        item.review_id for item in verified.items
    ]
    _assert_private_modes(config.session_dir)

    def label_provider(review_id: str, source: str) -> str:
        shown.append((review_id, source))
        return expected_by_id[review_id]

    attestation_path = manual_review.review_and_seal(
        config,
        label_provider=label_provider,
    )
    assert len(shown) == 48
    assert [review_id for review_id, _ in shown] == [item.review_id for item in verified.items]
    assert all("raw-source-secret" in source for _, source in shown)
    attestation = _read_json(attestation_path)
    assert len(attestation["records"]) == 48  # type: ignore[arg-type]
    assert all(
        set(record) == {"review_id", "label"}
        for record in attestation["records"]  # type: ignore[union-attr]
    )
    assert (
        len(list(config.session_dir.glob("python-p1-manual-review-attestation.sha256-*.json"))) == 1
    )
    with pytest.raises(ManualReviewError):
        manual_review.review_and_seal(config, label_provider=label_provider)

    result_path = manual_review.unblind_and_publish(config)
    result = _read_json(result_path)
    assert result["status"] == "CLASSIFICATION_SPOT_CHECK_MATCHED"
    assert result["manual_gate_passed"] is True
    assert result["reviewed_records"] == 48
    assert result["manual_label_mismatches"] == 0
    assert result["unreviewable_records"] == 0
    assert "records" not in result
    assert "review_id" not in result_path.read_text(encoding="utf-8")
    assert result["individual_truth_rows_persisted"] == 0
    assert result["source_selection_result"] is None
    assert len(list(config.session_dir.glob("python-p1-manual-review-result.sha256-*.json"))) == 1
    with pytest.raises(ManualReviewError):
        manual_review.unblind_and_publish(config)

    persisted = b"\n".join(path.read_bytes() for path in config.session_dir.iterdir())
    for item in verified.items:
        assert item.raw not in persisted
        assert item.selection_rank_sha256.encode() not in persisted
        assert item.presentation_sha256.encode() not in persisted
    assert b'"automatic_outcome"' not in persisted
    assert b'"selection_rank_sha256"' not in persisted
    assert b'"presentation_sha256"' not in persisted
    _assert_private_modes(config.session_dir)


@pytest.mark.parametrize("exception_label", ["MANUAL_REJECT", "UNREVIEWABLE"])
def test_disagreement_or_unreviewable_blocks_manual_gate(
    synthetic_review: tuple[ManualReviewConfig, VerifiedInputs],
    exception_label: str,
):
    config, verified = synthetic_review
    first_id = verified.items[0].review_id

    manual_review.prepare_queue(config)

    def label_provider(review_id: str, _source: str) -> str:
        if review_id == first_id:
            return exception_label
        item = next(item for item in verified.items if item.review_id == review_id)
        return _expected_label(item)

    manual_review.review_and_seal(config, label_provider=label_provider)
    result = _read_json(manual_review.unblind_and_publish(config))
    assert result["status"] == "BLOCKED_MANUAL_REVIEW_EXCEPTION"
    assert result["manual_gate_passed"] is False
    assert result["manual_label_mismatches"] == 1
    assert result["unreviewable_records"] == (exception_label == "UNREVIEWABLE")
    assert result["source_selection_result"] is None
    assert result["p1b_authorized"] is False


def test_zero_network_audit_guard_rejects_socket_and_non_git_subprocess_without_publish(
    tmp_path: Path,
):
    output_dir = tmp_path / "must-remain-empty"
    output_dir.mkdir()
    code = r"""
import pathlib
import socket
import subprocess
import sys

from pretrain.manual_review_python_p1 import ManualReviewError, install_zero_network_guards

install_zero_network_guards()
blocked = []
try:
    socket.socket()
except ManualReviewError:
    blocked.append("socket")
try:
    subprocess.run(["true"], check=False)
except ManualReviewError:
    blocked.append("subprocess")
if blocked != ["socket", "subprocess"] or any(pathlib.Path(sys.argv[1]).iterdir()):
    raise SystemExit(9)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, os.fspath(output_dir)],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert not list(output_dir.iterdir())


def test_sensitive_review_dataclasses_have_opaque_repr():
    item = ReviewItem(
        review_id="mrv2-0001",
        arm="secret-arm",
        automatic_outcome="secret-outcome",
        selection_rank_sha256="1" * 64,
        presentation_sha256="2" * 64,
        raw=b"secret-raw-source",
    )
    verified = VerifiedInputs(
        spec={"secret-spec": True},
        spec_sha256="3" * 64,
        items=(item,),
        queue={"secret-queue": True},
        sensitive_values=frozenset({"secret-identifier"}),
    )
    for value in (item, verified):
        rendered = repr(value)
        assert "secret" not in rendered
        assert "raw" not in rendered
        assert "arm" not in rendered
        assert "outcome" not in rendered


def test_non_tty_review_fails_before_opening_or_displaying_source(
    monkeypatch: pytest.MonkeyPatch,
):
    opened: list[object] = []
    monkeypatch.setattr(manual_review.os, "isatty", lambda _fd: False)
    monkeypatch.setattr(manual_review.os, "open", lambda *args, **kwargs: opened.append(args))
    with pytest.raises(ManualReviewError, match="controlling TTY"):
        manual_review._tty_labels((
            manual_review.BlindReviewItem(review_id="mrv2-0001", raw=b"secret"),
        ))
    assert opened == []


def test_each_phase_rejects_unexpected_session_files_without_publishing(
    synthetic_review: tuple[ManualReviewConfig, VerifiedInputs],
):
    config, verified = synthetic_review
    expected_by_id = {item.review_id: _expected_label(item) for item in verified.items}

    manual_review.prepare_queue(config)
    unexpected = config.session_dir / "unexpected.txt"
    unexpected.write_text("synthetic only", encoding="utf-8")
    os.chmod(unexpected, 0o600)
    with pytest.raises(ManualReviewError, match="not pristine"):
        manual_review.review_and_seal(
            config,
            label_provider=lambda review_id, _source: expected_by_id[review_id],
        )
    assert manual_review.ATTESTATION_MARKER not in {
        path.name for path in config.session_dir.iterdir()
    }

    unexpected.unlink()
    manual_review.review_and_seal(
        config,
        label_provider=lambda review_id, _source: expected_by_id[review_id],
    )
    unexpected.write_text("synthetic only", encoding="utf-8")
    os.chmod(unexpected, 0o600)
    with pytest.raises(ManualReviewError, match="not pristine"):
        manual_review.unblind_and_publish(config)
    assert manual_review.RESULT_MARKER not in {path.name for path in config.session_dir.iterdir()}


def test_prepare_refuses_preexisting_session_namespace(
    synthetic_review: tuple[ManualReviewConfig, VerifiedInputs],
):
    config, _ = synthetic_review
    config.session_dir.mkdir(mode=0o700)
    (config.session_dir / "unexpected.txt").write_text("synthetic only", encoding="utf-8")
    with pytest.raises(ManualReviewError, match="already exists"):
        manual_review.prepare_queue(config)
    assert (config.session_dir / "unexpected.txt").read_text(encoding="utf-8") == "synthetic only"
