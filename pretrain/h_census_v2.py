#!/usr/bin/env python3
"""Stage-H canonical accounting census v2 and candidate-plan generator.

Stage H measures and predicts. It publishes no physical candidate views: the frozen original
releases remain the only document-byte authority, and everything Stage H emits is metadata,
row-index sets and accounting.

Its canonical output is deliberately labelled NON_AUTHORITATIVE_FEASIBILITY_REPLAY. Stage I later
reruns the same shared replay core against the same frozen bytes and must reproduce the branch,
selection fingerprint, selected token totals, crossing identity, overshoot and ownership matrix
exactly; any mismatch is a hard stop.

Resume is not supported. A run either completes and publishes atomically, or publishes nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pretrain.build_pretrain_shards import (  # noqa: E402
    clean_text,
    cleaned_text_sha256,
    encode_with_accounting,
    load_tokenizer,
)
from pretrain.select_pretrain_documents import canonical_document_fingerprint  # noqa: E402
from pretrain.stage_i_graph_v2 import (  # noqa: E402
    GraphError,
    InputBinding,
    SourceGraph,
    canonical_json_bytes,
    load_source_graph,
    sha256_file,
    validate_eligibility_index,
)
from pretrain.stage_i_replay_v2 import (  # noqa: E402
    CandidateRecord,
    ReplayError,
    ownership_matrix,
    replay,
    score_to_bits,
)
from src.special_tokens import BOS_ID, EOS_ID, SPECIAL_TOKEN_IDS  # noqa: E402

CENSUS_SCHEMA = "petitgpt-h-census-v2"
PLAN_SCHEMA = "petitgpt-h-candidate-plan-v2"
OUTPUT_LABEL = "NON_AUTHORITATIVE_FEASIBILITY_REPLAY"
SPECIAL_IDS = frozenset(SPECIAL_TOKEN_IDS.values())
RESUME_SUPPORTED = False


class CensusError(RuntimeError):
    """Fail-closed census condition. Nothing is published when this is raised."""


def read_score(row: dict[str, Any], accessor: dict[str, Any], which: str) -> Any:
    """Read a score strictly through the release-pinned accessor.

    The FineWeb release stores the LITERAL key ``"metadata.int_score"`` inside its ``metadata``
    object. Generic dotted-path splitting silently fails there, so the accessor names a container
    and an exact key and nothing else is attempted.
    """
    spec = accessor.get(which)
    if spec is None:
        return None
    container = spec.get("container")
    key = spec.get("key")
    if container is None or key is None:
        raise CensusError(f"accessor for {which} is incomplete")
    holder = row.get(container)
    if not isinstance(holder, dict):
        raise CensusError(f"accessor container {container!r} missing or not an object")
    if key not in holder:
        raise CensusError(f"accessor key {key!r} missing from {container!r}")
    value = holder[key]
    expected = spec.get("json_type")
    if expected == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise CensusError(f"{key!r} must be a JSON integer, got {type(value).__name__}")
    elif expected == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CensusError(f"{key!r} must be a JSON number, got {type(value).__name__}")
    return value


def scan_binding(
    binding: InputBinding, tokenizer_path: str
) -> tuple[list[CandidateRecord], dict[str, int]]:
    """Stream one frozen release once and emit its eligible candidate metadata records."""
    excluded = set(validate_eligibility_index(binding).tolist())
    tokenizer = load_tokenizer(tokenizer_path)
    accessor = binding.schema_accessor
    cleaning = binding.cleaning_contract
    records: list[CandidateRecord] = []
    counters = {
        "physical_rows": 0,
        "d2_d3_excluded_rows": 0,
        "empty_or_non_string_text": 0,
        "empty_after_cleaning": 0,
        "eligible_rows": 0,
    }
    digest = hashlib.sha256()

    with open(binding.documents_path, "rb", buffering=8 * 1024 * 1024) as handle:
        for row_index, raw in enumerate(handle):
            digest.update(raw)
            counters["physical_rows"] += 1
            if row_index in excluded:
                counters["d2_d3_excluded_rows"] += 1
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise CensusError(
                    f"{binding.input_binding_id}:{row_index}: malformed JSON"
                ) from exc
            if not isinstance(row, dict):
                raise CensusError(f"{binding.input_binding_id}:{row_index}: row is not an object")
            text = row.get(binding.text_field)
            if text is None:
                raise CensusError(
                    f"{binding.input_binding_id}:{row_index}: missing {binding.text_field!r}"
                )
            if not isinstance(text, str):
                raise CensusError(
                    f"{binding.input_binding_id}:{row_index}: {binding.text_field!r} is not a string"
                )
            if not text:
                counters["empty_or_non_string_text"] += 1
                continue
            cleaned = clean_text(
                text,
                strip_leading_noise=bool(cleaning["strip_leading_noise"]),
                normalize_quotes=bool(cleaning["normalize_quotes"]),
                underscores_policy=str(cleaning["underscores_policy"]),
                min_chars=int(cleaning["min_chars"]),
                min_ascii_ratio=float(cleaning["min_ascii_ratio"]),
            )
            if cleaned is None:
                counters["empty_after_cleaning"] += 1
                continue
            ids, content, boundary = encode_with_accounting(
                tokenizer, cleaned, add_bos=True, add_eos=True, bos_id=BOS_ID, eos_id=EOS_ID
            )
            if not ids or ids[0] != BOS_ID or ids[-1] != EOS_ID or boundary != 2:
                raise CensusError(f"{binding.input_binding_id}:{row_index}: framing violation")
            leaked = SPECIAL_IDS.intersection(ids[1:-1])
            if leaked:
                raise CensusError(
                    f"{binding.input_binding_id}:{row_index}: text encoded to special ids {sorted(leaked)}"
                )
            int_score = read_score(row, accessor, "integer_score")
            raw_cont = read_score(row, accessor, "continuous_score")
            records.append(
                CandidateRecord(
                    input_binding_id=binding.input_binding_id,
                    binding_ordinal=0,
                    row_index=row_index,
                    cleaned_sha256=cleaned_text_sha256(cleaned),
                    canonical_fingerprint=canonical_document_fingerprint(cleaned),
                    serialized_tokens=content + boundary,
                    int_score=None if int_score is None else int(int_score),
                    score_bits=None if raw_cont is None else score_to_bits(raw_cont),
                )
            )
            counters["eligible_rows"] += 1

    if counters["physical_rows"] != binding.total_physical_rows:
        raise CensusError(f"{binding.input_binding_id}: physical row count mismatch")
    if counters["d2_d3_excluded_rows"] != binding.excluded_rows:
        raise CensusError(f"{binding.input_binding_id}: eligibility exclusions not fully consumed")
    if digest.hexdigest() != binding.documents_sha256:
        raise CensusError(f"{binding.input_binding_id}: documents SHA-256 mismatch")
    return records, counters


def build_census(graph: SourceGraph, tokenizer_path: str, exclusion_path: Path) -> dict[str, Any]:
    """Run the full metadata census and feasibility replay. Publishes nothing by itself."""
    exclusion = json.loads(exclusion_path.read_text(encoding="utf-8"))
    if exclusion.get("exclusion_scope") != "entire_pre_tokenizer_reserved_pool":
        raise CensusError("reference exclusion manifest is not the whole-reserve scope")
    identities = set(exclusion["hashes"])
    if len(identities) != int(exclusion["hash_count"]):
        raise CensusError("reference exclusion hash_count disagrees with the hash list")

    by_binding: dict[str, list[CandidateRecord]] = {}
    counters: dict[str, dict[str, int]] = {}
    for binding_id in sorted(graph.bindings):
        records, stats = scan_binding(graph.bindings[binding_id], tokenizer_path)
        by_binding[binding_id] = records
        counters[binding_id] = stats

    results = replay(graph, by_binding, identities)
    return {
        "schema_version": CENSUS_SCHEMA,
        "status": "COMPLETE",
        "output_label": OUTPUT_LABEL,
        "resume_supported": RESUME_SUPPORTED,
        "graph_sha256": graph.graph_sha256,
        "seed": graph.seed,
        "bound_authorities": graph.bound_authorities,
        "reference_exclusion_identities": len(identities),
        "binding_counters": counters,
        "nodes": [r.as_canonical() for r in results],
        "ownership_matrix": ownership_matrix(results),
        "hard_stop_required": any(not r.feasible for r in results),
        "totals": {
            "eligible_rows": sum(c["eligible_rows"] for c in counters.values()),
            "physical_rows": sum(c["physical_rows"] for c in counters.values()),
            "selected_serialized_tokens": sum(r.selected_serialized_tokens for r in results),
            "selected_identities": sum(r.selected_identities for r in results),
        },
    }


def publish_atomic(out_dir: Path, census: dict[str, Any]) -> Path:
    """Unique run directory, sibling staging, COMPLETE written last, atomic rename."""
    fingerprint = hashlib.sha256(canonical_json_bytes(census)).hexdigest()[:16]
    final = out_dir / f"run-{census['graph_sha256'][:16]}-{fingerprint}"
    if final.exists():
        raise CensusError(f"run directory already exists, refusing to overwrite: {final}")
    out_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{final.name}.staging-", dir=str(out_dir)))
    try:
        (staging / "census.json").write_bytes(canonical_json_bytes(census))
        (staging / "COMPLETE").write_text(fingerprint + "\n", encoding="ascii")
        os.replace(staging, final)
    except BaseException:
        for path in sorted(staging.rglob("*"), reverse=True):
            path.unlink(missing_ok=True)
        staging.rmdir()
        raise
    return final


def generate_candidate_plan(
    graph_path: Path, graph: SourceGraph, tools: dict[str, Path]
) -> dict[str, Any]:
    """A candidate plan that cannot authorize itself."""
    return {
        "schema_version": PLAN_SCHEMA,
        "authorization_status": "NOT_AUTHORIZED",
        "authorization_note": (
            "This plan carries no owner authorization and cannot create one. Stage-H production "
            "must be invoked with --expected-plan-sha256 supplied externally by the owner."
        ),
        "graph_path": str(graph_path),
        "graph_sha256": graph.graph_sha256,
        "seed": graph.seed,
        "resume_supported": RESUME_SUPPORTED,
        "bound_authorities": graph.bound_authorities,
        "tool_sha256": {name: sha256_file(path) for name, path in sorted(tools.items())},
        "input_bindings": {
            key: {
                "documents_sha256": b.documents_sha256,
                "documents_size_bytes": b.documents_size_bytes,
                "release_manifest_sha256": b.release_manifest_sha256,
                "eligibility_index_sha256": b.eligibility_index_sha256,
                "total_physical_rows": b.total_physical_rows,
                "expected_eligible_rows": b.expected_eligible_rows,
                "schema_accessor_id": b.schema_accessor.get("accessor_id"),
            }
            for key, b in sorted(graph.bindings.items())
        },
        "node_order": [n.source_id for n in graph.nodes],
        "h_publishes_physical_views": False,
    }


def require_authorized_plan(plan_path: Path, expected_sha256: str | None) -> dict[str, Any]:
    """Refuse to proceed without an externally supplied, matching plan SHA-256."""
    if not expected_sha256:
        raise CensusError("--expected-plan-sha256 is required; a plan may not authorize itself")
    actual = sha256_file(plan_path)
    if actual != expected_sha256:
        raise CensusError(f"plan SHA-256 mismatch: expected {expected_sha256}, got {actual}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise CensusError("plan schema_version mismatch")
    return plan


def main(argv: list[str] | None = None) -> int:
    # Checked before argparse so a resume attempt fails on its own terms rather than as a
    # generic usage error.
    if "--resume" in (argv if argv is not None else sys.argv[1:]):
        raise CensusError("resume is not supported")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--expected-plan-sha256", type=str, default=None)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--reference-exclusion", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    require_authorized_plan(args.plan, args.expected_plan_sha256)
    graph = load_source_graph(args.graph, verify_hashes=True)
    census = build_census(graph, str(args.tokenizer), args.reference_exclusion)
    final = publish_atomic(args.out_dir, census)
    print(f"published {final}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CensusError, GraphError, ReplayError) as exc:
        print(f"FAIL-CLOSED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
