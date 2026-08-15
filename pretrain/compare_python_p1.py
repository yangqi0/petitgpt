#!/usr/bin/env python3

"""Compare matched offline Python P1 reports without selecting a source."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import math
from pathlib import Path
import re
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pretrain.analyze_python_p1 import (  # noqa: E402
    ANALYSIS_KIND,
    ARM_ADAPTERS,
    MINIMUM_COMMON_SCORE_CONTENT_DOCUMENTS_PER_ARM,
    PRODUCTION_BACKEND_CONTRACT,
    PRODUCTION_CACHE_ORIGIN_CONTRACT,
    AnalysisError,
    _publish_addressed_json,
    _require_git_ignored,
    load_analysis_report,
)

COMPARISON_KIND = "petitgpt_python_p1_matched_comparison"
DECISION_SCOPE = "SOURCE_COMPARISON_NOT_FINAL_TOKEN_APPROVAL"
INCONCLUSIVE_RATE_DELTA = 0.10
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ComparisonError(RuntimeError):
    """Matched analysis reports cannot be compared safely."""


@dataclass(frozen=True)
class ComparisonConfig:
    primary_report: Path
    stack_report: Path
    policy_sha256: str
    output_dir: Path
    enforce_ignored_output: bool = True


def _nested(value: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise ComparisonError(f"analysis report lacks {'.'.join(path)}")
        current = current[key]
    return current


def _validate_interval(value: Any, *, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ComparisonError(f"{label} must be an interval object")
    try:
        estimate = float(value["estimate"])
        lower = float(value["lower"])
        upper = float(value["upper"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ComparisonError(f"{label} has invalid estimate/lower/upper") from exc
    if not all(math.isfinite(number) for number in (estimate, lower, upper)):
        raise ComparisonError(f"{label} contains non-finite values")
    if not 0 <= lower <= estimate <= upper <= 1:
        raise ComparisonError(f"{label} is not a valid probability interval")
    return {"estimate": estimate, "lower": lower, "upper": upper}


def _validate_projection(value: Any, *, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ComparisonError(f"{label} must be a projection interval object")
    try:
        estimate = float(value["estimate"])
        lower = float(value["heuristic_lower_uncertainty_envelope"])
        upper = float(value["heuristic_upper_uncertainty_envelope"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ComparisonError(
            f"{label} has invalid estimate/heuristic uncertainty envelope"
        ) from exc
    if not all(math.isfinite(number) for number in (estimate, lower, upper)):
        raise ComparisonError(f"{label} contains non-finite values")
    if not 0 <= lower <= estimate <= upper:
        raise ComparisonError(f"{label} is not an ordered non-negative interval")
    return {"estimate": estimate, "lower": lower, "upper": upper}


def _validate_report(
    report: Mapping[str, Any],
    *,
    expected_role: str,
    expected_policy_sha256: str,
) -> None:
    if report.get("kind") != ANALYSIS_KIND or report.get("schema_version") != 1:
        raise ComparisonError("unsupported analyzer report kind/version")
    if report.get("decision_scope") != DECISION_SCOPE:
        raise ComparisonError("analyzer report decision scope drifted")
    if report.get("automatic_source_approval") is not False:
        raise ComparisonError("analyzer report must forbid automatic source approval")
    if report.get("network_access") is not False or report.get("source_code_exposed") is not False:
        raise ComparisonError("analyzer report violates offline/privacy contract")
    expected_arm = {"role": expected_role, "adapter": ARM_ADAPTERS[expected_role]}
    if report.get("arm") != expected_arm:
        raise ComparisonError(
            f"expected exactly one {expected_role!r} arm with adapter "
            f"{ARM_ADAPTERS[expected_role]!r}"
        )
    contract = report.get("matched_contract")
    if not isinstance(contract, Mapping):
        raise ComparisonError("analyzer report lacks matched_contract")
    expected_contract = {
        "seed": 20_250_814,
        "metadata_rows": 500,
        "selected_blobs": 300,
        "policy_sha256": expected_policy_sha256,
        "minimum_common_score_content_documents_per_arm": (
            MINIMUM_COMMON_SCORE_CONTENT_DOCUMENTS_PER_ARM
        ),
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise ComparisonError(
                f"{expected_role} report matched contract {key!r} differs: "
                f"expected {expected!r}, got {contract.get(key)!r}"
            )
    inputs = report.get("inputs")
    if not isinstance(inputs, Mapping) or not isinstance(inputs.get("policy"), Mapping):
        raise ComparisonError("analyzer report lacks input policy identity")
    if inputs["policy"].get("sha256") != expected_policy_sha256:
        raise ComparisonError("analyzer report input policy SHA disagrees with contract")
    if inputs["policy"].get("expected_sha256") != expected_policy_sha256:
        raise ComparisonError("analyzer report expected policy SHA disagrees with contract")
    if inputs.get("collection_backend_provenance") != PRODUCTION_BACKEND_CONTRACT:
        raise ComparisonError("analyzer report lacks exact production backend provenance")
    if inputs.get("collection_cache_origin_contract") != PRODUCTION_CACHE_ORIGIN_CONTRACT:
        raise ComparisonError("analyzer report lacks exact production cache-origin provenance")
    private_cache = inputs.get("private_cache")
    if (
        not isinstance(private_cache, Mapping)
        or private_cache.get("production_origin_contract_verified") is not True
        or private_cache.get("cache_hit_origin_evidence_verified") is not True
        or private_cache.get("cache_indexes_verified") != private_cache.get("objects_verified")
    ):
        raise ComparisonError("analyzer report lacks complete production cache-index evidence")


def _rate_delta(
    primary: Mapping[str, Any], stack: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    primary_interval = _validate_interval(primary, label=f"primary {label}")
    stack_interval = _validate_interval(stack, label=f"stack {label}")
    delta = stack_interval["estimate"] - primary_interval["estimate"]
    delta_lower = stack_interval["lower"] - primary_interval["upper"]
    delta_upper = stack_interval["upper"] - primary_interval["lower"]
    overlap = not (
        primary_interval["upper"] < stack_interval["lower"]
        or stack_interval["upper"] < primary_interval["lower"]
    )
    below_margin = abs(delta) < INCONCLUSIVE_RATE_DELTA
    inconclusive = below_margin or overlap
    reasons: list[str] = []
    if below_margin:
        reasons.append("absolute point delta is below 10 percentage points")
    if overlap:
        reasons.append("95% intervals overlap")
    return {
        "direction": "stack_minus_primary",
        "primary": primary_interval,
        "stack": stack_interval,
        "delta_percentage_points": delta * 100,
        "conservative_delta_interval_percentage_points": {
            "lower_95": delta_lower * 100,
            "upper_95": delta_upper * 100,
        },
        "absolute_delta_below_10_percentage_points": below_margin,
        "source_intervals_overlap": overlap,
        "interpretation": (
            "INCONCLUSIVE_LT_10PP_OR_INTERVAL_OVERLAP"
            if inconclusive
            else "DIRECTIONAL_DIFFERENCE_NOT_SOURCE_APPROVAL"
        ),
        "inconclusive_reasons": reasons,
    }


def _yield_scenario(report: Mapping[str, Any], *, score_slice: str) -> Mapping[str, Any]:
    scenarios = _nested(
        report,
        ("score_slices_and_pretokenizer_yield", score_slice, "serialized_token_sensitivity"),
    )
    if not isinstance(scenarios, list):
        raise ComparisonError(f"{score_slice} token sensitivity must be a list")
    matches = [
        scenario
        for scenario in scenarios
        if isinstance(scenario, Mapping)
        and scenario.get("assumed_characters_per_content_token") == 4.0
    ]
    if len(matches) != 1:
        raise ComparisonError(f"{score_slice} must have exactly one 4.0 chars/token scenario")
    return _nested(matches[0], ("projected_serialized_token_equivalent",))


def _yield_delta(
    primary: Mapping[str, Any], stack: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    primary_interval = _validate_projection(primary, label=f"primary {label}")
    stack_interval = _validate_projection(stack, label=f"stack {label}")
    overlap = not (
        primary_interval["upper"] < stack_interval["lower"]
        or stack_interval["upper"] < primary_interval["lower"]
    )
    return {
        "primary": primary_interval,
        "stack": stack_interval,
        "signed_point_delta_stack_minus_primary": (
            stack_interval["estimate"] - primary_interval["estimate"]
        ),
        "heuristic_delta_uncertainty_envelope": {
            "lower": stack_interval["lower"] - primary_interval["upper"],
            "upper": stack_interval["upper"] - primary_interval["lower"],
        },
        "heuristic_source_envelopes_overlap": overlap,
        "interpretation": "DESCRIPTIVE_HEURISTIC_ONLY",
        "directional_conclusion": None,
        "uncertainty_calibration": ("NOT_A_CALIBRATED_JOINT_95_PERCENT_CONFIDENCE_INTERVAL"),
        "scope": "PRE_TOKENIZER_SENSITIVITY_NOT_MEASURED_CANONICAL_TOKENS",
    }


def _common_score_sample_size(report: Mapping[str, Any], *, label: str) -> int:
    score_slice = _nested(report, ("score_slices_and_pretokenizer_yield", "score_gte_4"))
    if not isinstance(score_slice, Mapping) or not isinstance(score_slice.get("available"), bool):
        raise ComparisonError(f"{label} common score slice availability is malformed")
    if score_slice["available"] is False:
        return 0
    sampled = _nested(score_slice, ("content_stage_conditional_on_eligible", "sampled_documents"))
    if isinstance(sampled, bool) or not isinstance(sampled, int) or sampled < 0:
        raise ComparisonError(f"{label} common score slice sample count is invalid")
    return sampled


def _insufficient_common_slice(
    *, primary_documents: int, stack_documents: int, metric_kind: str
) -> dict[str, Any]:
    return {
        "interpretation": "INCONCLUSIVE_INSUFFICIENT_COMMON_SLICE",
        "directional_conclusion": None,
        "metric_kind": metric_kind,
        "primary_sampled_documents": primary_documents,
        "stack_sampled_documents": stack_documents,
        "minimum_required_per_arm": MINIMUM_COMMON_SCORE_CONTENT_DOCUMENTS_PER_ARM,
        "automatic_resampling_or_backfill": False,
        "required_follow_up": "SEPARATELY_PRE_FREEZE_P1B_BEFORE_ADDITIONAL_COLLECTION",
        "note": (
            "This P1 comparator must not auto-sample. Any larger common-score experiment "
            "requires a separately pre-frozen P1b design and new run identity."
        ),
    }


def build_python_p1_comparison(
    config: ComparisonConfig,
    *,
    verified_reports: tuple[
        tuple[Mapping[str, Any], str],
        tuple[Mapping[str, Any], str],
    ]
    | None = None,
) -> dict[str, Any]:
    """Validate and build a comparison without publishing an artifact."""
    if not _SHA256_RE.fullmatch(config.policy_sha256):
        raise ComparisonError("policy_sha256 must be an exact lowercase SHA-256")
    if config.enforce_ignored_output:
        try:
            _require_git_ignored(config.output_dir, label="comparison output directory")
        except AnalysisError as exc:
            raise ComparisonError(str(exc)) from exc
    if verified_reports is None:
        try:
            primary, primary_sha = load_analysis_report(config.primary_report)
            stack, stack_sha = load_analysis_report(config.stack_report)
        except AnalysisError as exc:
            raise ComparisonError(str(exc)) from exc
    else:
        (primary, primary_sha), (stack, stack_sha) = verified_reports
    _validate_report(
        primary,
        expected_role="primary",
        expected_policy_sha256=config.policy_sha256,
    )
    _validate_report(
        stack,
        expected_role="stack_comparison",
        expected_policy_sha256=config.policy_sha256,
    )

    base_rate_paths = {
        "fetch_success_selected": (
            "collection_outcomes",
            "fetch_success_fraction_wilson_95",
        ),
        "strict_utf8_selected": (
            "collection_outcomes",
            "strict_utf8_fraction_wilson_95",
        ),
        "all_score_content_hard_retention": (
            "score_slices_and_pretokenizer_yield",
            "all_scores",
            "content_stage_conditional_on_eligible",
            "retained_fraction_wilson_95",
        ),
        "all_score_metadata_eligibility": (
            "score_slices_and_pretokenizer_yield",
            "all_scores",
            "metadata_stage",
            "eligible_fraction_wilson_95",
        ),
        "metadata_license_provenance_field_coverage": (
            "provenance_presence",
            "metadata_sample",
            "derived",
            "license_or_detected_nonempty",
            "fraction_wilson_95",
        ),
        "metadata_declared_encoding_presence": (
            "provenance_presence",
            "metadata_sample",
            "derived",
            "encoding_declared_nonempty",
            "fraction_wilson_95",
        ),
        "metadata_core_provenance_complete": (
            "provenance_presence",
            "metadata_sample",
            "derived",
            "core_blob_repo_path_complete",
            "fraction_wilson_95",
        ),
    }
    common_rate_paths = {
        "common_score_gte_4_content_hard_retention": (
            "score_slices_and_pretokenizer_yield",
            "score_gte_4",
            "content_stage_conditional_on_eligible",
            "retained_fraction_wilson_95",
        ),
        "common_score_gte_4_metadata_eligibility": (
            "score_slices_and_pretokenizer_yield",
            "score_gte_4",
            "metadata_stage",
            "eligible_fraction_wilson_95",
        ),
    }
    rate_deltas = {
        name: _rate_delta(
            _nested(primary, path),
            _nested(stack, path),
            label=name,
        )
        for name, path in base_rate_paths.items()
    }
    license_coverage = rate_deltas["metadata_license_provenance_field_coverage"]
    coverage_rate_status = license_coverage.pop("interpretation")
    license_coverage["field_coverage_statistical_status"] = (
        "FIELD_COVERAGE_RATE_DIFFERENCE_DETECTED_NOT_LICENSE_QUALITY"
        if coverage_rate_status == "DIRECTIONAL_DIFFERENCE_NOT_SOURCE_APPROVAL"
        else "FIELD_COVERAGE_RATE_DIFFERENCE_INCONCLUSIVE_NOT_LICENSE_QUALITY"
    )
    license_coverage["interpretation"] = "PROVENANCE_FIELD_COVERAGE_ONLY_NOT_LICENSE_CLEARANCE"
    license_coverage["directional_conclusion"] = None
    license_coverage["legal_inference"] = (
        "Automated field presence does not establish legal permission; field absence does not "
        "establish that no license exists. Removal and terms gates remain independent and block "
        "production until satisfied."
    )
    primary_common_documents = _common_score_sample_size(primary, label="primary")
    stack_common_documents = _common_score_sample_size(stack, label="stack")
    common_slice_sufficient = (
        primary_common_documents >= MINIMUM_COMMON_SCORE_CONTENT_DOCUMENTS_PER_ARM
        and stack_common_documents >= MINIMUM_COMMON_SCORE_CONTENT_DOCUMENTS_PER_ARM
    )
    if common_slice_sufficient:
        rate_deltas.update({
            name: _rate_delta(
                _nested(primary, path),
                _nested(stack, path),
                label=name,
            )
            for name, path in common_rate_paths.items()
        })
    else:
        rate_deltas.update({
            name: _insufficient_common_slice(
                primary_documents=primary_common_documents,
                stack_documents=stack_common_documents,
                metric_kind="RATE",
            )
            for name in common_rate_paths
        })
    yield_deltas = {
        "all_scores_at_4_chars_per_content_token": _yield_delta(
            _yield_scenario(primary, score_slice="all_scores"),
            _yield_scenario(stack, score_slice="all_scores"),
            label="all_scores_at_4_chars_per_content_token",
        ),
    }
    if common_slice_sufficient:
        yield_deltas["common_score_gte_4_at_4_chars_per_content_token"] = _yield_delta(
            _yield_scenario(primary, score_slice="score_gte_4"),
            _yield_scenario(stack, score_slice="score_gte_4"),
            label="common_score_gte_4_at_4_chars_per_content_token",
        )
    else:
        yield_deltas["common_score_gte_4_at_4_chars_per_content_token"] = (
            _insufficient_common_slice(
                primary_documents=primary_common_documents,
                stack_documents=stack_common_documents,
                metric_kind="PRETOKENIZER_YIELD",
            )
        )
    report = {
        "schema_version": 1,
        "kind": COMPARISON_KIND,
        "status": "complete",
        "decision_scope": DECISION_SCOPE,
        "automatic_source_approval": False,
        "source_selection_result": None,
        "network_access": False,
        "source_code_exposed": False,
        "policy_sha256": config.policy_sha256,
        "expected_policy_sha256": config.policy_sha256,
        "inputs": {
            "primary": {"filename": config.primary_report.name, "sha256": primary_sha},
            "stack_comparison": {
                "filename": config.stack_report.name,
                "sha256": stack_sha,
            },
        },
        "matched_contract": {
            "seed": 20_250_814,
            "metadata_rows_per_arm": 500,
            "selected_blobs_per_arm": 300,
            "one_expected_arm_each": True,
            "common_score_slice_minimum_int_score": 4,
            "minimum_common_score_content_documents_per_arm": (
                MINIMUM_COMMON_SCORE_CONTENT_DOCUMENTS_PER_ARM
            ),
            "primary_common_score_content_documents": primary_common_documents,
            "stack_common_score_content_documents": stack_common_documents,
            "common_score_slice_sufficient": common_slice_sufficient,
            "rate_inconclusive_margin_percentage_points": 10.0,
        },
        "rate_deltas": rate_deltas,
        "pretokenizer_yield_deltas": yield_deltas,
        "interpretation": {
            "rate_rule": (
                "A rate difference is inconclusive when its absolute point delta is below "
                "10 percentage points or its 95% intervals overlap."
            ),
            "yield_rule": (
                "Projected-yield uncertainty envelopes are uncalibrated pre-tokenizer "
                "sensitivity only. Overlap and non-overlap are descriptive; neither produces "
                "a directional conclusion or source approval."
            ),
            "common_score_slice_rule": (
                "Each arm requires at least 100 sampled score>=4 content documents. An "
                "underfilled arm makes all related comparisons inconclusive; no automatic "
                "backfill is allowed, and a larger study requires separately pre-frozen P1b."
            ),
            "license_rule": (
                "PROVENANCE_FIELD_COVERAGE_ONLY_NOT_LICENSE_CLEARANCE: automated license-field "
                "presence does not establish legal permission, absence does not establish no "
                "license, and independent removal/terms gates still block production."
            ),
            "metadata_interval_limitation": (
                "Wilson intervals treat the 500 rows as iid even though sampling uses fifty "
                "contiguous ten-row clusters; correlation can make intervals optimistic."
            ),
            "manual_review_required": True,
        },
        "privacy": {
            "source_text_fields": 0,
            "repository_or_path_values": 0,
            "individual_identity_hashes": 0,
        },
    }
    return report


def compare_python_p1(config: ComparisonConfig) -> Path:
    """Validate, compare, and publish exactly one report for each P1 arm."""
    report = build_python_p1_comparison(config)
    return _publish_addressed_json(config.output_dir, "python-p1-comparison", report)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare matched primary and Stack-Edu Python P1 analysis reports."
    )
    parser.add_argument("--primary_report", type=Path, required=True)
    parser.add_argument("--stack_report", type=Path, required=True)
    parser.add_argument("--expected_policy_sha256", dest="policy_sha256", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    path = compare_python_p1(
        ComparisonConfig(
            primary_report=args.primary_report,
            stack_report=args.stack_report,
            policy_sha256=args.policy_sha256,
            output_dir=args.output_dir,
        )
    )
    print(path)


if __name__ == "__main__":
    main()
