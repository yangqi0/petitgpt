from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

import pretrain.analyze_python_p1 as analyzer
from pretrain.analyze_python_p1 import AnalysisConfig
import pretrain.collect_python_p1 as collector
import pretrain.compare_python_p1 as comparator
from pretrain.compare_python_p1 import ComparisonConfig
from pretrain.inspect_python_sources import HttpJsonResponse
from pretrain.manual_review_io import canonical_json_bytes
import pretrain.manual_review_python_p1 as manual_review
from pretrain.manual_review_python_p1 import (
    ArmInputs,
    ManualReviewConfig,
    ManualReviewError,
)

_TEST_BOOTSTRAP_RESAMPLES = 8


class _FakeRowsServer:
    """Dataset-server shaped transport over an in-memory synthetic corpus."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        revision: str,
        dataset_config: str,
    ) -> None:
        self._rows = rows
        self._revision = revision
        self._dataset_config = dataset_config
        self._features = [
            {"name": name, "type": {"dtype": "string", "_type": "Value"}} for name in rows[0]
        ]

    def __call__(self, url: str, timeout_seconds: float) -> HttpJsonResponse:
        assert timeout_seconds == collector.HTTP_TIMEOUT_SECONDS
        parsed = urlparse(url)
        endpoint = parsed.path.rsplit("/", 1)[-1]
        query = parse_qs(parsed.query)
        if endpoint == "size":
            payload = {
                "size": {
                    "splits": [
                        {
                            "config": self._dataset_config,
                            "split": "train",
                            "num_rows": len(self._rows),
                        }
                    ]
                }
            }
        elif endpoint == "rows":
            offset = int(query["offset"][0])
            length = int(query["length"][0])
            payload = {
                "features": self._features,
                "rows": [
                    {
                        "row_idx": index,
                        "row": dict(self._rows[index]),
                        "truncated_cells": [],
                    }
                    for index in range(offset, offset + length)
                ],
            }
        else:  # pragma: no cover - the collector has a fixed endpoint contract.
            raise AssertionError(f"unexpected endpoint {endpoint!r}")
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return HttpJsonResponse(
            payload=payload,
            headers={"x-revision": self._revision},
            body_sha256=hashlib.sha256(body).hexdigest(),
            http_status=200,
            body_bytes=len(body),
            request_latency_seconds=0.001,
        )


@dataclass(frozen=True)
class _SyntheticArm:
    inputs: ArmInputs
    manifest: dict[str, Any]
    analysis: dict[str, Any]
    records: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _SyntheticStudy:
    root: Path
    policy_path: Path
    policy_sha256: str
    comparison_path: Path
    primary: _SyntheticArm
    stack: _SyntheticArm
    spec_path: Path
    spec: dict[str, Any]
    config: ManualReviewConfig
    expected_items: tuple[tuple[str, str, str, str, str], ...]


def _addressed_json(directory: Path, stem: str, value: dict[str, Any]) -> Path:
    data = canonical_json_bytes(value)
    digest = hashlib.sha256(data).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}.sha256-{digest}.json"
    path.write_bytes(data)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_source(index: int) -> bytes:
    if index % 5 == 0:
        prefix = f"# syntax-reject {index}\ndef broken_{index}(:\n".encode()
        return prefix + b"#" + b"b" * (240 - len(prefix) - 1)
    if index % 7 == 0:
        prefix = (
            f"# auto-generated; do not edit {index}\ndef generated_{index}():\n    return {index}\n"
        ).encode()
        return prefix + b"#" + b"g" * (240 - len(prefix) - 1)
    prefix = (
        f'"""synthetic module {index}"""\n'
        f"# synthetic comment {index}\n"
        f"def synthetic_{index}(value):\n    return value + {index}\n"
    ).encode()
    return prefix + b"#" + b"k" * (240 - len(prefix) - 1)


def _synthetic_corpus(*, role: str) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    rows: list[dict[str, Any]] = []
    raw_by_blob: dict[str, bytes] = {}
    for index in range(500):
        blob_id = f"synthetic-{role}-blob-{index:04d}"
        raw = _synthetic_source(index)
        row: dict[str, Any] = {
            "blob_id": blob_id,
            "repo_name": f"synthetic-{role}-repository-{index % 23}",
            "path": f"synthetic/src/module_{index:04d}.py",
            "length_bytes": len(raw),
            "score": 4.5,
            "int_score": 4,
        }
        if role == "stack_comparison":
            row.update({
                "detected_licenses": ["MIT"],
                "language": "Python",
                "license_type": "permissive",
                "src_encoding": "utf-8",
            })
        rows.append(row)
        raw_by_blob[blob_id] = raw
    return rows, raw_by_blob


def _arm_policy(role: str) -> tuple[str, dict[str, Any]]:
    policy = analyzer.p1_policy_template()
    if role == "primary":
        return "smollm_python_edu", policy["arms"][0]
    if role == "stack_comparison":
        return "stack_edu_python", policy["arms"][1]
    raise AssertionError(role)


def _productionize_collection(
    *,
    injected_manifest_path: Path,
    injected_report_path: Path,
    cache_dir: Path,
    output_dir: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    manifest = json.loads(injected_manifest_path.read_bytes())
    report = json.loads(injected_report_path.read_bytes())
    manifest["backend_provenance"] = analyzer.PRODUCTION_BACKEND_CONTRACT
    manifest["cache_origin_contract"] = analyzer.PRODUCTION_CACHE_ORIGIN_CONTRACT

    index_names: set[str] = set()
    raw_names: set[str] = set()
    for entry in manifest["selected_blobs"]:
        blob_id = entry["metadata"]["blob_id"]
        index_name = f"index-{hashlib.sha256(blob_id.encode()).hexdigest()}.json"
        index_path = cache_dir / index_name
        index = json.loads(index_path.read_bytes())
        index["origin"] = analyzer.PRODUCTION_CACHE_ORIGIN_CONTRACT
        index_path.write_bytes(
            json.dumps(
                index,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            + b"\n"
        )
        index_names.add(index_name)
        raw_names.add(entry["cache_object"])
    assert len(index_names) == 300
    assert len(raw_names) == 300
    assert {path.name for path in cache_dir.iterdir()} == index_names | raw_names

    manifest_path = _addressed_json(output_dir, "collection-manifest", manifest)
    manifest_sha = _sha256(manifest_path)
    report["backend_provenance"] = analyzer.PRODUCTION_BACKEND_CONTRACT
    report["cache_origin_contract"] = analyzer.PRODUCTION_CACHE_ORIGIN_CONTRACT
    report["manifest"] = {"path": str(manifest_path), "sha256": manifest_sha}
    report_path = _addressed_json(output_dir, "collection-report", report)
    return manifest_path, report_path, manifest


def _population(records: tuple[dict[str, Any], ...]) -> dict[str, int]:
    outcomes = [manual_review._automatic_outcome(record) for record in records]
    counts = Counter(outcome for outcome, _reviewable in outcomes)
    reviewable_reject = sum(reviewable for _outcome, reviewable in outcomes)
    return {
        "total": len(records),
        "keep": counts["keep"],
        "reject": counts["reject"],
        "reviewable_reject": reviewable_reject,
        "nonreviewable_reject": counts["reject"] - reviewable_reject,
    }


def _build_arm(
    *,
    root: Path,
    role: str,
    policy_path: Path,
    policy_sha256: str,
) -> _SyntheticArm:
    adapter, arm_policy = _arm_policy(role)
    rows, raw_by_blob = _synthetic_corpus(role=role)
    injected_root = root / role / "injected"
    cache_dir = root / role / "cache"
    config = collector.P1Config(
        expected_revision=arm_policy["expected_revision"],
        adapter=adapter,
        dataset=arm_policy["dataset"],
        dataset_config=arm_policy["dataset_config"],
        split=arm_policy["split"],
        output_dir=injected_root,
        cache_dir=cache_dir,
        policy_path=policy_path,
        expected_policy_sha256=policy_sha256,
        workers=collector.MAX_WORKERS,
        enforce_ignored_paths=False,
    )
    injected = collector.collect_python_p1(
        config,
        transport=_FakeRowsServer(
            rows,
            revision=arm_policy["expected_revision"],
            dataset_config=arm_policy["dataset_config"],
        ),
        blob_fetcher=raw_by_blob.__getitem__,
        clock=lambda: 1.0,
        sleeper=lambda _seconds: None,
    )
    frozen_root = root / role / "frozen"
    manifest_path, collection_report_path, manifest = _productionize_collection(
        injected_manifest_path=Path(injected["manifest"]),
        injected_report_path=Path(injected["report"]),
        cache_dir=cache_dir,
        output_dir=frozen_root,
    )

    replay_config = replace(
        config,
        output_dir=root / role / "replay",
        replay_manifest=manifest_path,
    )
    replay = collector._offline_replay(
        replay_config,
        policy_binding=manifest["policy_binding"],
        clock=lambda: 2.0,
    )
    replay_path = _addressed_json(replay_config.output_dir, "replay-report", replay)

    records: list[dict[str, Any]] = []
    analysis = analyzer.build_python_p1_analysis(
        AnalysisConfig(
            collection_manifest=manifest_path,
            policy_path=policy_path,
            policy_sha256=policy_sha256,
            cache_dir=cache_dir,
            output_dir=root / role / "analysis",
            expected_arm=role,
            enforce_ignored_paths=False,
        ),
        verified_records_out=records,
    )
    analysis_path = _addressed_json(root / role / "analysis", "python-p1-analysis", analysis)
    return _SyntheticArm(
        inputs=ArmInputs(
            role=role,
            manifest=manifest_path,
            collection_report=collection_report_path,
            replay_report=replay_path,
            analysis_report=analysis_path,
            cache_dir=cache_dir,
        ),
        manifest=manifest,
        analysis=analysis,
        records=tuple(records),
    )


def _expected_items(
    primary: _SyntheticArm, stack: _SyntheticArm
) -> tuple[tuple[str, str, str, str, str], ...]:
    sampled: list[tuple[str, str, str]] = []
    for arm_name, arm in (("primary", primary), ("stack_comparison", stack)):
        frames: dict[str, list[str]] = {"keep": [], "reject": []}
        for record in arm.records:
            outcome, reviewable = manual_review._automatic_outcome(record)
            if outcome == "keep" or reviewable:
                blob_id = record["metadata"]["blob_id"]
                rank = hashlib.sha256(f"{analyzer.COLLECTION_SEED}\0{blob_id}".encode()).hexdigest()
                frames[outcome].append(rank)
        for outcome in manual_review.AUTOMATIC_OUTCOMES:
            sampled.extend((arm_name, outcome, rank) for rank in sorted(frames[outcome])[:12])
    presented = sorted(
        (
            hashlib.sha256(
                manual_review.PRESENTATION_DOMAIN + b"\0" + rank.encode("ascii")
            ).hexdigest(),
            arm,
            outcome,
            rank,
        )
        for arm, outcome, rank in sampled
    )
    return tuple(
        (f"mrv2-{ordinal:04d}", arm, outcome, rank, presentation)
        for ordinal, (presentation, arm, outcome, rank) in enumerate(presented, start=1)
    )


def _build_spec(
    *,
    policy_sha256: str,
    primary: _SyntheticArm,
    stack: _SyntheticArm,
    comparison_path: Path,
    output_namespace: str,
) -> dict[str, Any]:
    def arm_binding(arm: _SyntheticArm) -> dict[str, Any]:
        return {
            "collection_manifest_sha256": _sha256(arm.inputs.manifest),
            "collection_report_sha256": _sha256(arm.inputs.collection_report),
            "replay_report_sha256": _sha256(arm.inputs.replay_report),
            "analysis_sha256": _sha256(arm.inputs.analysis_report),
            "expected_population": _population(arm.records),
        }

    return {
        "schema_version": 2,
        "kind": "petitgpt_python_p1_blinded_manual_review_policy",
        "status": "FROZEN_BEFORE_INDIVIDUAL_REVIEW",
        "decision_scope": manual_review.DECISION_SCOPE,
        "inputs": {
            "frozen_p1_policy_sha256": policy_sha256,
            "primary": arm_binding(primary),
            "stack_comparison": arm_binding(stack),
            "comparison_sha256": _sha256(comparison_path),
        },
        "outputs": {"exact_output_namespace": output_namespace},
        "sampling": {
            "presentation_domain_ascii": manual_review.PRESENTATION_DOMAIN.decode("ascii"),
            "presentation_separator_hex": "00",
            "selected_records": 48,
        },
        "outcomes": {
            "reviewable_records_per_outcome_per_arm": 12,
            "gate_order": list(analyzer.FULL_GATE_ORDER),
        },
        "manual_attestation": {
            "allowed_labels": list(manual_review.ALLOWED_LABELS),
            "all_48_labels_required": True,
            "review_session_id": "synthetic-verifier-session",
        },
        "validation": {
            "network_access": False,
            "expected_cache_indexes_per_arm": 300,
            "expected_raw_objects_per_arm": 300,
        },
    }


@pytest.fixture(scope="module")
def synthetic_study(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("manual-review-verifier")
    previous_root = manual_review.PROJECT_ROOT
    previous_resamples = analyzer.BOOTSTRAP_RESAMPLES
    manual_review.PROJECT_ROOT = root
    analyzer.BOOTSTRAP_RESAMPLES = _TEST_BOOTSTRAP_RESAMPLES
    try:
        policy_path = root / "policy.json"
        policy_path.write_bytes(canonical_json_bytes(analyzer.p1_policy_template()))
        policy_sha = _sha256(policy_path)
        primary = _build_arm(
            root=root,
            role="primary",
            policy_path=policy_path,
            policy_sha256=policy_sha,
        )
        stack = _build_arm(
            root=root,
            role="stack_comparison",
            policy_path=policy_path,
            policy_sha256=policy_sha,
        )
        comparison = comparator.build_python_p1_comparison(
            ComparisonConfig(
                primary_report=primary.inputs.analysis_report,
                stack_report=stack.inputs.analysis_report,
                policy_sha256=policy_sha,
                output_dir=root / "comparison",
                enforce_ignored_output=False,
            ),
            verified_reports=(
                (primary.analysis, _sha256(primary.inputs.analysis_report)),
                (stack.analysis, _sha256(stack.inputs.analysis_report)),
            ),
        )
        comparison_path = _addressed_json(root / "comparison", "python-p1-comparison", comparison)
        output_namespace = "session-does-not-exist"
        spec = _build_spec(
            policy_sha256=policy_sha,
            primary=primary,
            stack=stack,
            comparison_path=comparison_path,
            output_namespace=output_namespace,
        )
        spec_path = root / "manual-review-v2-synthetic-spec.json"
        spec_path.write_bytes(canonical_json_bytes(spec))
        config = ManualReviewConfig(
            spec_path=spec_path,
            policy_path=policy_path,
            comparison_report=comparison_path,
            primary=primary.inputs,
            stack_comparison=stack.inputs,
            session_dir=root / output_namespace,
            expected_generator_commit="0" * 40,
            enforce_environment=False,
            enforce_frozen_spec=False,
        )
        yield _SyntheticStudy(
            root=root,
            policy_path=policy_path,
            policy_sha256=policy_sha,
            comparison_path=comparison_path,
            primary=primary,
            stack=stack,
            spec_path=spec_path,
            spec=spec,
            config=config,
            expected_items=_expected_items(primary, stack),
        )
    finally:
        manual_review.PROJECT_ROOT = previous_root
        analyzer.BOOTSTRAP_RESAMPLES = previous_resamples


def test_verify_inputs_revalidates_all_synthetic_objects_and_builds_blinded_queue(
    synthetic_study: _SyntheticStudy,
    monkeypatch: pytest.MonkeyPatch,
):
    reads: Counter[str] = Counter()
    original = manual_review.read_regular_file_at_nofollow

    def counted_read(directory_fd: int, name: str, *, max_bytes: int | None = None) -> bytes:
        if name.startswith("index-"):
            reads["index"] += 1
        elif name.startswith("raw-sha256-"):
            reads["raw"] += 1
        return original(directory_fd, name, max_bytes=max_bytes)

    monkeypatch.setattr(manual_review, "read_regular_file_at_nofollow", counted_read)
    verified = manual_review.verify_inputs(synthetic_study.config)

    assert reads == {"index": 600, "raw": 600}
    assert len(verified.items) == 48
    assert len({item.review_id for item in verified.items}) == 48
    assert Counter((item.arm, item.automatic_outcome) for item in verified.items) == {
        ("primary", "keep"): 12,
        ("primary", "reject"): 12,
        ("stack_comparison", "keep"): 12,
        ("stack_comparison", "reject"): 12,
    }
    observed_items = tuple(
        (
            item.review_id,
            item.arm,
            item.automatic_outcome,
            item.selection_rank_sha256,
            item.presentation_sha256,
        )
        for item in verified.items
    )
    assert observed_items == synthetic_study.expected_items
    assert verified.queue["record_count"] == 48
    assert verified.queue["records"] == [
        {"review_id": f"mrv2-{ordinal:04d}"} for ordinal in range(1, 49)
    ]
    assert all(set(row) == {"review_id"} for row in verified.queue["records"])


def test_verify_inputs_rejects_cache_index_symlink(
    synthetic_study: _SyntheticStudy,
):
    first_blob = synthetic_study.primary.manifest["selected_blobs"][0]["metadata"]["blob_id"]
    index_path = synthetic_study.primary.inputs.cache_dir / (
        f"index-{hashlib.sha256(first_blob.encode()).hexdigest()}.json"
    )
    backup = index_path.with_name(f"{index_path.name}.regular-backup")
    index_path.rename(backup)
    index_path.symlink_to(backup.name)
    try:
        with pytest.raises(ManualReviewError, match="no-follow snapshot failed"):
            manual_review.verify_inputs(synthetic_study.config)
    finally:
        index_path.unlink()
        backup.rename(index_path)


def test_verify_inputs_rejects_rebound_selection_rank_tamper(
    synthetic_study: _SyntheticStudy,
):
    tamper_root = synthetic_study.root / "rank-tamper"
    manifest = json.loads(canonical_json_bytes(synthetic_study.primary.manifest))
    manifest["selected_blobs"][0]["selection_rank_sha256"] = "0" * 64
    manifest_path = _addressed_json(tamper_root, "collection-manifest", manifest)

    collection_report = json.loads(synthetic_study.primary.inputs.collection_report.read_bytes())
    collection_report["manifest"] = {
        "path": str(manifest_path),
        "sha256": _sha256(manifest_path),
    }
    collection_report_path = _addressed_json(tamper_root, "collection-report", collection_report)
    replay_report = json.loads(synthetic_study.primary.inputs.replay_report.read_bytes())
    replay_report["input_manifest"] = str(manifest_path)
    replay_report["input_manifest_sha256"] = _sha256(manifest_path)
    replay_report_path = _addressed_json(tamper_root, "replay-report", replay_report)

    spec = json.loads(canonical_json_bytes(synthetic_study.spec))
    primary_binding = spec["inputs"]["primary"]
    primary_binding["collection_manifest_sha256"] = _sha256(manifest_path)
    primary_binding["collection_report_sha256"] = _sha256(collection_report_path)
    primary_binding["replay_report_sha256"] = _sha256(replay_report_path)
    spec_path = tamper_root / "spec.json"
    spec_path.write_bytes(canonical_json_bytes(spec))
    config = replace(
        synthetic_study.config,
        spec_path=spec_path,
        primary=replace(
            synthetic_study.config.primary,
            manifest=manifest_path,
            collection_report=collection_report_path,
            replay_report=replay_report_path,
        ),
    )
    with pytest.raises(ManualReviewError, match="replay semantic validation failed"):
        manual_review.verify_inputs(config)


def test_verify_inputs_rejects_population_drift(
    synthetic_study: _SyntheticStudy,
):
    spec = json.loads(canonical_json_bytes(synthetic_study.spec))
    spec["inputs"]["primary"]["expected_population"]["keep"] += 1
    spec_path = synthetic_study.root / "population-tamper-spec.json"
    spec_path.write_bytes(canonical_json_bytes(spec))
    with pytest.raises(ManualReviewError, match="population disagrees with spec"):
        manual_review.verify_inputs(replace(synthetic_study.config, spec_path=spec_path))


def test_verify_inputs_rejects_analysis_byte_mismatch(
    synthetic_study: _SyntheticStudy,
):
    tamper_dir = synthetic_study.root / "analysis-byte-tamper"
    tamper_dir.mkdir()
    tampered_path = tamper_dir / synthetic_study.primary.inputs.analysis_report.name
    tampered_path.write_bytes(synthetic_study.primary.inputs.analysis_report.read_bytes() + b" ")
    config = replace(
        synthetic_study.config,
        primary=replace(synthetic_study.config.primary, analysis_report=tampered_path),
    )
    with pytest.raises(ManualReviewError, match="analysis report SHA-256 mismatch"):
        manual_review.verify_inputs(config)
