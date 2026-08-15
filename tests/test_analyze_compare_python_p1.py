from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

import pretrain.analyze_python_p1 as analyzer
from pretrain.analyze_python_p1 import (
    PRODUCTION_BACKEND_CONTRACT,
    PRODUCTION_CACHE_ORIGIN_CONTRACT,
    AnalysisConfig,
    AnalysisError,
    analyze_python_p1,
    load_analysis_report,
    p1_policy_template,
)
import pretrain.collect_python_p1 as collector
import pretrain.compare_python_p1 as comparator
from pretrain.compare_python_p1 import (
    ComparisonConfig,
    ComparisonError,
    compare_python_p1,
)
from pretrain.inspect_python_sources import deterministic_windows


def _json_bytes(value: dict, *, compact: bool = False) -> bytes:
    if compact:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            + b"\n"
        )
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode()
        + b"\n"
    )


def _write_policy(directory: Path, *, compact: bool = False) -> tuple[Path, str]:
    data = _json_bytes(p1_policy_template(), compact=compact)
    path = directory / ("policy-compact.json" if compact else "policy.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest()


def _write_addressed_manifest(directory: Path, value: dict) -> Path:
    data = _json_bytes(value)
    digest = hashlib.sha256(data).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"collection-manifest.sha256-{digest}.json"
    path.write_bytes(data)
    return path


def _valid_source(index: int) -> bytes:
    prefix = (
        f'"""module documentation {index}"""\n'
        f"# private-source-comment-{index}\n"
        f"def private_function_{index}(value):\n"
        f"    return value + {index}\n"
    ).encode()
    return prefix + b"#" + b"p" * (220 - len(prefix) - 1)


def _arm_spec(role: str) -> tuple[str, dict]:
    policy = p1_policy_template()
    if role == "primary":
        return "smollm_python_edu", policy["arms"][0]
    if role == "stack_comparison":
        return "stack_edu_python", policy["arms"][1]
    raise AssertionError(role)


def _write_collection(
    root: Path,
    *,
    role: str,
    policy_sha: str,
    oversize_rows: int = 200,
    stack_optionals: bool = False,
    common_score_rows: int = 500,
) -> tuple[Path, Path]:
    adapter, arm = _arm_spec(role)
    cache_dir = root / f"{role}-private-cache"
    cache_dir.mkdir(parents=True)
    eligible_rows = 500 - oversize_rows
    assert eligible_rows >= 300
    assert 0 <= common_score_rows <= 500
    windows = deterministic_windows(
        total_rows=1_000,
        sample_rows=500,
        window_size=10,
        seed=20_250_814,
    )
    sampled_row_indices = [
        row_idx
        for window in windows
        for row_idx in range(window.offset, window.offset + window.length)
    ]
    rows: list[dict] = []
    raw_by_blob: dict[str, bytes] = {}
    for index, row_idx in enumerate(sampled_row_indices):
        blob_id = f"{role}-blob-{index:04d}"
        if index >= eligible_rows:
            raw = None
            length = 100_001
        elif index == 2:
            raw = b"\xff" + b"x" * 219
            length = len(raw)
        elif index == 3:
            raw = b"# auto-generated; do not edit\nvalue = 1\n" + b"#" + b"g" * 177
            length = len(raw)
        elif index == 5:
            raw = b"def broken(:\n" + b"#" + b"b" * 206
            length = len(raw)
        elif index == 1:
            raw = _valid_source(0)
            length = len(raw)
        else:
            raw = _valid_source(index)
            length = len(raw)
        row = {
            "row_idx": row_idx,
            "blob_id": blob_id,
            "repo_name": f"private-repository-{index % 7}",
            "path": (
                f"private/vendor/file_{index}.py" if index == 4 else f"private/src/file_{index}.py"
            ),
            "length_bytes": length,
            "score": 4.5 if index < common_score_rows else 3.5,
            "int_score": 4 if index < common_score_rows else 3,
        }
        if stack_optionals:
            row.update({
                "detected_licenses": ["MIT"],
                "license_type": "permissive",
                "src_encoding": "utf-8",
            })
        rows.append(row)
        if raw is not None:
            raw_by_blob[blob_id] = raw

    eligible = sorted(
        rows[:eligible_rows],
        key=lambda row: (
            hashlib.sha256(f"20250814\0{row['blob_id']}".encode()).hexdigest(),
            row["blob_id"],
        ),
    )
    selected_rows = eligible[:300]
    entries: list[dict] = []
    for row in selected_rows:
        raw = raw_by_blob[row["blob_id"]]
        selection_rank = hashlib.sha256(f"20250814\0{row['blob_id']}".encode()).hexdigest()
        if row["blob_id"].endswith("-0010"):
            entries.append({
                "metadata": row,
                "selection_rank_sha256": selection_rank,
                "fetch_outcome": "failed",
                "error_category": "not_found",
                "fetch_attempts": 1,
                "fetch_attempt_latencies_seconds": [0.001],
                "fetch_latency_seconds": 0.001,
                "fidelity_outcome": "not_available",
                "decode_outcome": "not_attempted",
            })
            continue
        if row["blob_id"].endswith("-0011"):
            raw += b"x"
        raw_sha = hashlib.sha256(raw).hexdigest()
        relative = Path(f"raw-sha256-{raw_sha}.raw")
        raw_path = cache_dir / relative
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        if not raw_path.exists():
            raw_path.write_bytes(raw)
        index_key = hashlib.sha256(row["blob_id"].encode()).hexdigest()
        index_path = cache_dir / f"index-{index_key}.json"
        index_path.write_bytes(
            _json_bytes(
                {
                    "schema_version": collector.CACHE_INDEX_SCHEMA_VERSION,
                    "blob_id": row["blob_id"],
                    "raw_sha256": raw_sha,
                    "raw_bytes": len(raw),
                    "raw_path": str(relative),
                    "origin": PRODUCTION_CACHE_ORIGIN_CONTRACT,
                },
                compact=True,
            )
        )
        try:
            raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            decode_outcome = "decode_failed"
        else:
            decode_outcome = "decoded"
        fidelity_outcome = (
            "length_matches_metadata"
            if len(raw) == row["length_bytes"]
            else "metadata_length_mismatch"
        )
        entry = {
            "metadata": row,
            "selection_rank_sha256": selection_rank,
            "fetch_outcome": "success",
            "fetch_attempts": 1,
            "fetch_attempt_latencies_seconds": [0.001],
            "fetch_latency_seconds": 0.001,
            "error_category": None,
            "raw_sha256": raw_sha,
            "raw_bytes": len(raw),
            "cache_object": str(relative),
            "cache_origin_verified": False,
            "fidelity_outcome": fidelity_outcome,
            "decode_outcome": (
                decode_outcome if fidelity_outcome == "length_matches_metadata" else "not_attempted"
            ),
        }
        if fidelity_outcome == "length_matches_metadata":
            entry["quality"] = {}
        entries.append(entry)

    normalized_fields = {
        "blob_id": "blob_id",
        "length_bytes": "length_bytes",
        "repo_name": "repo_name",
        "path": "path",
        "score": "score",
        "int_score": "int_score",
    }
    if stack_optionals:
        normalized_fields.update({
            "detected_licenses": "detected_licenses",
            "license_type": "license_type",
            "src_encoding": "src_encoding",
        })
    manifest = {
        "schema_version": 1,
        "kind": "petitgpt_python_p1_collection",
        "decision_scope": "SOURCE_COMPARISON_NOT_FINAL_TOKEN_APPROVAL",
        "backend_provenance": PRODUCTION_BACKEND_CONTRACT,
        "backend_accounting": {"swh": {"cache_origin_verified": True}},
        "cache_origin_contract": PRODUCTION_CACHE_ORIGIN_CONTRACT,
        "policy_binding": {
            "path": "policy.json",
            "sha256": policy_sha,
            "expected_sha256": policy_sha,
            "schema_version": 1,
            "kind": "petitgpt_python_p1_matched_source_policy",
            "status": "FROZEN_BEFORE_COLLECTION",
            "decision_scope": "SOURCE_COMPARISON_NOT_FINAL_TOKEN_APPROVAL",
            "arm_tuple": [
                arm["name"],
                arm["dataset"],
                arm["dataset_config"],
                arm["split"],
                arm["expected_revision"],
            ],
        },
        "contract": {
            "metadata_rows": 500,
            "windows": 50,
            "rows_per_window": 10,
            "selected_distinct_blobs": 300,
            "seed": 20_250_814,
            "selection_gates": [
                "length_bytes<=100000",
                "distinct_blob_id",
                "stable_hash_rank",
            ],
            "explicitly_not_selection_gates": [
                "score",
                "minimum_size",
                "repository_cap",
            ],
            "no_backfill_after_content_selection": True,
            "source_code_exposed_in_manifest": False,
        },
        "input": {
            "adapter": adapter,
            "dataset": arm["dataset"],
            "dataset_config": arm["dataset_config"],
            "split": arm["split"],
            "expected_revision": arm["expected_revision"],
        },
        "schema": {
            "upstream_features": [],
            "upstream_fields": sorted(normalized_fields.values()),
            "normalized_field_map": normalized_fields,
        },
        "sampling": {
            "total_rows": 1_000,
            "windows": [asdict(window) for window in windows],
        },
        "metadata_rows": rows,
        "selection": {"selected": 300},
        "selected_blobs": entries,
        "hf_evidence": {},
    }
    manifest_path = _write_addressed_manifest(root, manifest)
    return manifest_path, cache_dir


def _analyze_arm(
    root: Path,
    *,
    role: str,
    policy_path: Path,
    policy_sha: str,
    stack_optionals: bool,
    common_score_rows: int = 500,
) -> Path:
    manifest, cache = _write_collection(
        root,
        role=role,
        policy_sha=policy_sha,
        stack_optionals=stack_optionals,
        common_score_rows=common_score_rows,
    )
    return analyze_python_p1(
        AnalysisConfig(
            collection_manifest=manifest,
            policy_path=policy_path,
            policy_sha256=policy_sha,
            cache_dir=cache,
            output_dir=root / f"{role}-analysis",
            expected_arm=role,
            enforce_ignored_paths=False,
        )
    )


def test_analyzer_is_private_two_stage_and_preserves_optional_presence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(analyzer, "BOOTSTRAP_RESAMPLES", 200)
    policy_path, policy_sha = _write_policy(tmp_path)
    primary_path = _analyze_arm(
        tmp_path / "primary",
        role="primary",
        policy_path=policy_path,
        policy_sha=policy_sha,
        stack_optionals=False,
    )
    stack_path = _analyze_arm(
        tmp_path / "stack",
        role="stack_comparison",
        policy_path=policy_path,
        policy_sha=policy_sha,
        stack_optionals=True,
    )
    primary, _ = load_analysis_report(primary_path)
    stack, _ = load_analysis_report(stack_path)

    assert primary["decision_scope"] == "SOURCE_COMPARISON_NOT_FINAL_TOKEN_APPROVAL"
    assert primary["network_access"] is False
    assert primary["source_code_exposed"] is False
    assert primary["inputs"]["private_cache"]["objects_verified"] == 299
    assert primary["inputs"]["private_cache"]["cache_indexes_verified"] == 299
    assert primary["inputs"]["private_cache"]["production_origin_contract_verified"] is True
    assert primary["inputs"]["private_cache"]["cache_hit_origin_evidence_verified"] is True
    assert primary["full_hard_gate_funnel"]["considered_documents"] == 300
    assert (
        primary["full_hard_gate_funnel"]["ordered_gate_funnel"][0]["rejected_at_gate_documents"]
        == 1
    )
    assert primary["collection_outcomes"]["fetch_failed"] == 1
    assert primary["collection_outcomes"]["fidelity_length_mismatch"] == 1
    assert primary["collection_outcomes"]["strict_utf8_not_evaluable"] == 2
    assert primary["collection_outcomes"]["strict_utf8_failed"] >= 1
    assert (
        "strict_utf8_conditional_on_fetch_fraction_wilson_95" not in primary["collection_outcomes"]
    )
    strict_conditional = primary["collection_outcomes"][
        "strict_utf8_conditional_on_fetch_and_fidelity"
    ]
    assert strict_conditional["numerator"] == primary["collection_outcomes"]["strict_utf8_success"]
    assert strict_conditional["denominator"] == 298
    assert (
        strict_conditional["denominator"]
        == primary["collection_outcomes"]["fidelity_length_matches"]
    )
    assert strict_conditional["fraction_wilson_95"]["estimate"] == pytest.approx(
        strict_conditional["numerator"] / strict_conditional["denominator"]
    )
    fidelity_gate = next(
        step
        for step in primary["full_hard_gate_funnel"]["ordered_gate_funnel"]
        if step["gate"] == "fidelity_length_matches"
    )
    assert fidelity_gate["input_documents"] == 299
    assert fidelity_gate["rejected_at_gate_documents"] == 1
    assert fidelity_gate["passed_documents"] == 298
    assert (
        primary["decoded_quality_metrics"]["decoded_documents"]
        == (primary["collection_outcomes"]["strict_utf8_success"])
    )
    assert primary["duplicates"]["selected_raw_sha256"]["duplicate_groups"] >= 1
    assert primary["duplicates"]["decoded_ast_canonical_fingerprint"]["duplicate_groups"] >= 1
    assert primary["concentration"]["metadata_repository_rows_and_bytes"]["values_exposed"] is False

    primary_presence = primary["provenance_presence"]["metadata_sample"]
    stack_presence = stack["provenance_presence"]["metadata_sample"]
    assert primary_presence["detected_licenses"]["classification"] == (
        "ABSENT_UPSTREAM_NORMALIZED_SCHEMA"
    )
    assert primary_presence["derived"]["license_or_detected_nonempty"]["rows"] == 0
    assert stack_presence["detected_licenses"]["nonempty_rows"] == 500
    assert stack_presence["src_encoding"]["nonempty_rows"] == 500
    assert stack_presence["derived"]["license_or_detected_nonempty"]["rows"] == 500

    overall = primary["score_slices_and_pretokenizer_yield"]["all_scores"]
    assert overall["metadata_stage"]["sampled_rows"] == 500
    assert overall["metadata_stage"]["eligible_unique_within_max_size_rows"] == 300
    assert overall["content_stage_conditional_on_eligible"]["sampled_documents"] == 300
    projected = overall["projected"]["retained_documents"]["estimate"]
    assert projected < 600
    assert projected < 0.7 * overall["metadata_stage"]["population_documents"]
    assert overall["scope"] == ("PRE_TOKENIZER_TWO_STAGE_SENSITIVITY_NOT_A_CANONICAL_TOKEN_COUNT")
    assert overall["combined_uncertainty_envelope_calibration"] == (
        "HEURISTIC_NOT_A_CALIBRATED_JOINT_95_PERCENT_CONFIDENCE_INTERVAL"
    )
    assert "lower_95" not in overall["projected"]["retained_documents"]
    assert "heuristic_lower_uncertainty_envelope" in overall["projected"]["retained_documents"]
    assert primary["score_slices_and_pretokenizer_yield"]["score_gte_3"]["available"]
    assert primary["score_slices_and_pretokenizer_yield"]["score_gte_4"]["available"]

    serialized = primary_path.read_bytes() + stack_path.read_bytes()
    for source_value in (
        b"private_function_",
        b"private-source-comment",
        b"private-repository-",
        b"private/src/file_",
        b"private/vendor/file_",
    ):
        assert source_value not in serialized


def test_analyzer_rejects_manifest_cache_and_policy_tamper_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(analyzer, "BOOTSTRAP_RESAMPLES", 200)
    policy_path, policy_sha = _write_policy(tmp_path)
    manifest, cache = _write_collection(tmp_path / "arm", role="primary", policy_sha=policy_sha)
    manifest_value = json.loads(manifest.read_text())
    test_only = json.loads(json.dumps(manifest_value))
    test_only["backend_provenance"]["production"] = False
    test_only["backend_provenance"]["test_only"] = True
    test_only["backend_provenance"]["mode"] = "test_only_injected"
    test_only_path = _write_addressed_manifest(tmp_path / "test-only", test_only)
    with pytest.raises(AnalysisError, match="exact production backend"):
        analyze_python_p1(
            AnalysisConfig(
                collection_manifest=test_only_path,
                policy_path=policy_path,
                policy_sha256=policy_sha,
                cache_dir=cache,
                output_dir=tmp_path / "test-only-output",
                expected_arm="primary",
                enforce_ignored_paths=False,
            )
        )

    binding_drift = json.loads(json.dumps(manifest_value))
    binding_drift["policy_binding"]["sha256"] = "0" * 64
    binding_drift_path = _write_addressed_manifest(tmp_path / "binding-drift", binding_drift)
    with pytest.raises(AnalysisError, match="policy binding"):
        analyze_python_p1(
            AnalysisConfig(
                collection_manifest=binding_drift_path,
                policy_path=policy_path,
                policy_sha256=policy_sha,
                cache_dir=cache,
                output_dir=tmp_path / "binding-drift-output",
                expected_arm="primary",
                enforce_ignored_paths=False,
            )
        )

    origin_drift = json.loads(json.dumps(manifest_value))
    origin_drift["cache_origin_contract"]["production"] = False
    origin_drift_path = _write_addressed_manifest(tmp_path / "origin-drift", origin_drift)
    with pytest.raises(AnalysisError, match="exact production SWH origin"):
        analyze_python_p1(
            AnalysisConfig(
                collection_manifest=origin_drift_path,
                policy_path=policy_path,
                policy_sha256=policy_sha,
                cache_dir=cache,
                output_dir=tmp_path / "origin-drift-output",
                expected_arm="primary",
                enforce_ignored_paths=False,
            )
        )

    accounting_drift = json.loads(json.dumps(manifest_value))
    accounting_drift["backend_accounting"]["swh"]["cache_origin_verified"] = False
    accounting_drift_path = _write_addressed_manifest(
        tmp_path / "accounting-drift", accounting_drift
    )
    with pytest.raises(AnalysisError, match="backend accounting lacks verified cache-origin"):
        analyze_python_p1(
            AnalysisConfig(
                collection_manifest=accounting_drift_path,
                policy_path=policy_path,
                policy_sha256=policy_sha,
                cache_dir=cache,
                output_dir=tmp_path / "accounting-drift-output",
                expected_arm="primary",
                enforce_ignored_paths=False,
            )
        )

    cache_hit_drift = json.loads(json.dumps(manifest_value))
    successful_entry = next(
        entry for entry in cache_hit_drift["selected_blobs"] if entry["fetch_outcome"] == "success"
    )
    successful_entry["cache_origin_verified"] = True
    cache_hit_drift_path = _write_addressed_manifest(tmp_path / "cache-hit-drift", cache_hit_drift)
    with pytest.raises(AnalysisError, match="cache-origin verification evidence drift"):
        analyze_python_p1(
            AnalysisConfig(
                collection_manifest=cache_hit_drift_path,
                policy_path=policy_path,
                policy_sha256=policy_sha,
                cache_dir=cache,
                output_dir=tmp_path / "cache-hit-drift-output",
                expected_arm="primary",
                enforce_ignored_paths=False,
            )
        )

    window_drift = json.loads(json.dumps(manifest_value))
    window_drift["sampling"]["windows"][0]["offset"] += 1
    window_drift_path = _write_addressed_manifest(tmp_path / "window-drift", window_drift)
    with pytest.raises(AnalysisError, match="deterministic sampling windows drifted"):
        analyze_python_p1(
            AnalysisConfig(
                collection_manifest=window_drift_path,
                policy_path=policy_path,
                policy_sha256=policy_sha,
                cache_dir=cache,
                output_dir=tmp_path / "window-drift-output",
                expected_arm="primary",
                enforce_ignored_paths=False,
            )
        )

    row_order_drift = json.loads(json.dumps(manifest_value))
    row_order_drift["metadata_rows"][0], row_order_drift["metadata_rows"][1] = (
        row_order_drift["metadata_rows"][1],
        row_order_drift["metadata_rows"][0],
    )
    row_order_drift_path = _write_addressed_manifest(tmp_path / "row-order-drift", row_order_drift)
    with pytest.raises(AnalysisError, match="metadata row order drifted"):
        analyze_python_p1(
            AnalysisConfig(
                collection_manifest=row_order_drift_path,
                policy_path=policy_path,
                policy_sha256=policy_sha,
                cache_dir=cache,
                output_dir=tmp_path / "row-order-drift-output",
                expected_arm="primary",
                enforce_ignored_paths=False,
            )
        )

    category_drift = json.loads(json.dumps(manifest_value))
    failed_entry = next(
        entry for entry in category_drift["selected_blobs"] if entry["fetch_outcome"] == "failed"
    )
    failed_entry["error_category"] = "private source text must never become a Counter key"
    category_drift_path = _write_addressed_manifest(tmp_path / "category-drift", category_drift)
    category_output = tmp_path / "category-drift-output"
    with pytest.raises(AnalysisError, match="non-allowlisted error category"):
        analyze_python_p1(
            AnalysisConfig(
                collection_manifest=category_drift_path,
                policy_path=policy_path,
                policy_sha256=policy_sha,
                cache_dir=cache,
                output_dir=category_output,
                expected_arm="primary",
                enforce_ignored_paths=False,
            )
        )
    assert not list(category_output.glob("*.json"))

    first = next(
        entry for entry in manifest_value["selected_blobs"] if entry["fetch_outcome"] == "success"
    )
    index_key = hashlib.sha256(first["metadata"]["blob_id"].encode()).hexdigest()
    index_path = cache / f"index-{index_key}.json"
    original_index = index_path.read_bytes()

    index_path.unlink()
    with pytest.raises(AnalysisError, match="missing production cache index"):
        analyze_python_p1(
            AnalysisConfig(
                collection_manifest=manifest,
                policy_path=policy_path,
                policy_sha256=policy_sha,
                cache_dir=cache,
                output_dir=tmp_path / "missing-index-output",
                expected_arm="primary",
                enforce_ignored_paths=False,
            )
        )
    index_path.write_bytes(original_index)

    legacy_index = json.loads(original_index)
    legacy_index["schema_version"] = 1
    index_path.write_bytes(_json_bytes(legacy_index, compact=True))
    with pytest.raises(AnalysisError, match="production cache index verification failed"):
        analyze_python_p1(
            AnalysisConfig(
                collection_manifest=manifest,
                policy_path=policy_path,
                policy_sha256=policy_sha,
                cache_dir=cache,
                output_dir=tmp_path / "legacy-index-output",
                expected_arm="primary",
                enforce_ignored_paths=False,
            )
        )
    index_path.write_bytes(original_index)

    wrong_origin_index = json.loads(original_index)
    wrong_origin_index["origin"]["production"] = False
    index_path.write_bytes(_json_bytes(wrong_origin_index, compact=True))
    with pytest.raises(AnalysisError, match="production cache index verification failed"):
        analyze_python_p1(
            AnalysisConfig(
                collection_manifest=manifest,
                policy_path=policy_path,
                policy_sha256=policy_sha,
                cache_dir=cache,
                output_dir=tmp_path / "wrong-origin-index-output",
                expected_arm="primary",
                enforce_ignored_paths=False,
            )
        )
    index_path.write_bytes(original_index)

    raw_path = cache / first["cache_object"]
    raw_path.write_bytes(raw_path.read_bytes() + b"tamper")
    output = tmp_path / "tamper-output"
    with pytest.raises(AnalysisError, match="production cache index verification failed"):
        analyze_python_p1(
            AnalysisConfig(
                collection_manifest=manifest,
                policy_path=policy_path,
                policy_sha256=policy_sha,
                cache_dir=cache,
                output_dir=output,
                expected_arm="primary",
                enforce_ignored_paths=False,
            )
        )
    assert not list(output.glob("*.json"))

    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(AnalysisError, match="manifest SHA mismatch"):
        analyze_python_p1(
            AnalysisConfig(
                collection_manifest=manifest,
                policy_path=policy_path,
                policy_sha256=policy_sha,
                cache_dir=cache,
                output_dir=tmp_path / "manifest-tamper-output",
                expected_arm="primary",
                enforce_ignored_paths=False,
            )
        )

    wrong_policy = p1_policy_template()
    wrong_policy["analysis"]["comparison"]["inconclusive_absolute_rate_difference_below"] = 0.11
    wrong_data = _json_bytes(wrong_policy)
    wrong_path = tmp_path / "wrong-policy.json"
    wrong_path.write_bytes(wrong_data)
    with pytest.raises(AnalysisError, match="does not match the frozen"):
        analyzer._load_policy(wrong_path, hashlib.sha256(wrong_data).hexdigest())
    with pytest.raises(AnalysisError, match="policy SHA mismatch"):
        analyzer._load_policy(policy_path, "0" * 64)


def test_comparator_is_matched_inconclusive_aware_and_never_approves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(analyzer, "BOOTSTRAP_RESAMPLES", 200)
    policy_path, policy_sha = _write_policy(tmp_path)
    primary_path = _analyze_arm(
        tmp_path / "primary",
        role="primary",
        policy_path=policy_path,
        policy_sha=policy_sha,
        stack_optionals=False,
    )
    stack_path = _analyze_arm(
        tmp_path / "stack",
        role="stack_comparison",
        policy_path=policy_path,
        policy_sha=policy_sha,
        stack_optionals=True,
    )
    comparison_path = compare_python_p1(
        ComparisonConfig(
            primary_report=primary_path,
            stack_report=stack_path,
            policy_sha256=policy_sha,
            output_dir=tmp_path / "comparison",
            enforce_ignored_output=False,
        )
    )
    comparison = json.loads(comparison_path.read_text())
    assert comparison["automatic_source_approval"] is False
    assert comparison["source_selection_result"] is None
    assert comparison["matched_contract"]["one_expected_arm_each"] is True
    assert "fetch_success_selected" in comparison["rate_deltas"]
    quality = comparison["rate_deltas"]["strict_utf8_selected"]
    assert quality["absolute_delta_below_10_percentage_points"] is True
    assert quality["source_intervals_overlap"] is True
    assert quality["interpretation"] == "INCONCLUSIVE_LT_10PP_OR_INTERVAL_OVERLAP"
    license_delta = comparison["rate_deltas"]["metadata_license_provenance_field_coverage"]
    assert license_delta["delta_percentage_points"] == 100
    assert license_delta["interpretation"] == (
        "PROVENANCE_FIELD_COVERAGE_ONLY_NOT_LICENSE_CLEARANCE"
    )
    assert license_delta["field_coverage_statistical_status"] == (
        "FIELD_COVERAGE_RATE_DIFFERENCE_DETECTED_NOT_LICENSE_QUALITY"
    )
    assert license_delta["directional_conclusion"] is None
    assert (
        comparison["pretokenizer_yield_deltas"]["common_score_gte_4_at_4_chars_per_content_token"][
            "scope"
        ]
        == "PRE_TOKENIZER_SENSITIVITY_NOT_MEASURED_CANONICAL_TOKENS"
    )
    assert all(
        value["interpretation"] == "DESCRIPTIVE_HEURISTIC_ONLY"
        and value["directional_conclusion"] is None
        for value in comparison["pretokenizer_yield_deltas"].values()
    )
    assert b"private_function_" not in comparison_path.read_bytes()

    with pytest.raises(ComparisonError, match="expected exactly one 'primary' arm"):
        compare_python_p1(
            ComparisonConfig(
                primary_report=stack_path,
                stack_report=primary_path,
                policy_sha256=policy_sha,
                output_dir=tmp_path / "swapped",
                enforce_ignored_output=False,
            )
        )


def test_common_score_comparison_requires_100_content_documents_per_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(analyzer, "BOOTSTRAP_RESAMPLES", 200)
    policy_path, policy_sha = _write_policy(tmp_path)
    primary_path = _analyze_arm(
        tmp_path / "primary-small-common",
        role="primary",
        policy_path=policy_path,
        policy_sha=policy_sha,
        stack_optionals=False,
        common_score_rows=50,
    )
    stack_path = _analyze_arm(
        tmp_path / "stack-full-common",
        role="stack_comparison",
        policy_path=policy_path,
        policy_sha=policy_sha,
        stack_optionals=True,
        common_score_rows=500,
    )
    comparison_path = compare_python_p1(
        ComparisonConfig(
            primary_report=primary_path,
            stack_report=stack_path,
            policy_sha256=policy_sha,
            output_dir=tmp_path / "small-common-comparison",
            enforce_ignored_output=False,
        )
    )
    comparison = json.loads(comparison_path.read_text())
    contract = comparison["matched_contract"]
    assert contract["primary_common_score_content_documents"] == 50
    assert contract["stack_common_score_content_documents"] == 300
    assert contract["minimum_common_score_content_documents_per_arm"] == 100
    assert contract["common_score_slice_sufficient"] is False
    related = [
        comparison["rate_deltas"]["common_score_gte_4_content_hard_retention"],
        comparison["rate_deltas"]["common_score_gte_4_metadata_eligibility"],
        comparison["pretokenizer_yield_deltas"]["common_score_gte_4_at_4_chars_per_content_token"],
    ]
    assert all(
        result["interpretation"] == "INCONCLUSIVE_INSUFFICIENT_COMMON_SLICE"
        and result["directional_conclusion"] is None
        and result["automatic_resampling_or_backfill"] is False
        and result["required_follow_up"] == "SEPARATELY_PRE_FREEZE_P1B_BEFORE_ADDITIONAL_COLLECTION"
        for result in related
    )
    assert (
        comparison["pretokenizer_yield_deltas"]["all_scores_at_4_chars_per_content_token"][
            "interpretation"
        ]
        == "DESCRIPTIVE_HEURISTIC_ONLY"
    )
    assert comparison["automatic_source_approval"] is False


def test_output_directories_are_git_ignored_fail_closed_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    policy_path, policy_sha = _write_policy(tmp_path)
    manifest, cache = _write_collection(tmp_path / "arm", role="primary", policy_sha=policy_sha)
    checked_labels: list[str] = []

    def analyzer_gate(path: Path, *, label: str) -> None:
        checked_labels.append(label)
        if label == "analysis output directory":
            raise AnalysisError("analysis output directory is not Git-ignored")

    monkeypatch.setattr(analyzer, "_require_git_ignored", analyzer_gate)
    with pytest.raises(AnalysisError, match="analysis output directory is not Git-ignored"):
        analyze_python_p1(
            AnalysisConfig(
                collection_manifest=manifest,
                policy_path=policy_path,
                policy_sha256=policy_sha,
                cache_dir=cache,
                output_dir=tmp_path / "unignored-analysis-output",
                expected_arm="primary",
            )
        )
    assert checked_labels == ["private cache", "analysis output directory"]

    def comparator_gate(path: Path, *, label: str) -> None:
        raise AnalysisError(f"{label} is not Git-ignored")

    monkeypatch.setattr(comparator, "_require_git_ignored", comparator_gate)
    with pytest.raises(ComparisonError, match="comparison output directory is not Git-ignored"):
        compare_python_p1(
            ComparisonConfig(
                primary_report=tmp_path / "not-read-primary.json",
                stack_report=tmp_path / "not-read-stack.json",
                policy_sha256="0" * 64,
                output_dir=tmp_path / "unignored-comparison-output",
            )
        )


def test_comparator_rejects_policy_mismatch_and_report_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(analyzer, "BOOTSTRAP_RESAMPLES", 200)
    policy_path, policy_sha = _write_policy(tmp_path)
    compact_policy, compact_sha = _write_policy(tmp_path, compact=True)
    assert compact_sha != policy_sha
    primary_path = _analyze_arm(
        tmp_path / "primary",
        role="primary",
        policy_path=policy_path,
        policy_sha=policy_sha,
        stack_optionals=False,
    )
    stack_path = _analyze_arm(
        tmp_path / "stack",
        role="stack_comparison",
        policy_path=compact_policy,
        policy_sha=compact_sha,
        stack_optionals=True,
    )
    with pytest.raises(ComparisonError, match="policy_sha256"):
        compare_python_p1(
            ComparisonConfig(
                primary_report=primary_path,
                stack_report=stack_path,
                policy_sha256=policy_sha,
                output_dir=tmp_path / "policy-mismatch",
                enforce_ignored_output=False,
            )
        )

    stack_path.write_bytes(stack_path.read_bytes() + b"tamper")
    with pytest.raises(ComparisonError, match="analysis report SHA mismatch"):
        compare_python_p1(
            ComparisonConfig(
                primary_report=primary_path,
                stack_report=stack_path,
                policy_sha256=policy_sha,
                output_dir=tmp_path / "report-tamper",
                enforce_ignored_output=False,
            )
        )
