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

Authorisation is a capability, not a comparison. An owner-supplied expected plan digest is checked
against the plan bytes, those same bytes are parsed under a closed schema, and every field in them
is compared with the graph, the tokenizer, the exclusion and eligibility authorities, and the
bytes of this implementation. Only the resulting :class:`AuthorizedRunContext` can publish. A
caller reaching ``census_body`` or ``replay`` directly still gets an answer, but it has no path to
a canonical directory, a COMPLETE marker or a production final name.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
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

# The frozen selector's private canonical-JSON helper is imported rather than reimplemented: the
# duplicate representative key must be byte-identical to v1's, and re-deriving the serialisation
# here would be a second definition that could drift. select_pretrain_documents.py is frozen and
# hash-pinned in the implementation bundle, so this name cannot move underneath us.
from pretrain.select_pretrain_documents import (  # noqa: E402
    SELECTION_METADATA_FIELD,
    _canonical_json_bytes,
    canonical_document_fingerprint,
)
from pretrain.stage_i_graph_v2 import (  # noqa: E402
    BOUND_AUTHORITY_KEYS,
    GraphError,
    InputBinding,
    SourceGraph,
    StrictJSONError,
    canonical_json_bytes,
    load_source_graph,
    open_authoritative,
    read_authoritative_bytes,
    strict_json_object,
)
from pretrain.stage_i_replay_v2 import (  # noqa: E402
    NODE_RESULT_SCHEMA,
    CandidateRecord,
    ReplayError,
    ownership_matrix,
    replay,
    representative_key,
    score_to_bits,
)
from src.special_tokens import BOS_ID, EOS_ID, SPECIAL_TOKEN_IDS  # noqa: E402

CENSUS_SCHEMA = "petitgpt-h-census-v2"
PLAN_SCHEMA = "petitgpt-h-candidate-plan-v3"
RUN_IDENTITY_SCHEMA = "petitgpt-h-run-identity-v1"
BUNDLE_SCHEMA = "petitgpt-h-implementation-bundle-v1"
OUTPUT_LABEL = "NON_AUTHORITATIVE_FEASIBILITY_REPLAY"
COMPLETE_MARKER = "COMPLETE"
CENSUS_FILENAME = "census.json"
SPECIAL_IDS = frozenset(SPECIAL_TOKEN_IDS.values())
RESUME_SUPPORTED = False
EXCLUSION_SCOPE = "entire_pre_tokenizer_reserved_pool"

#: Every module whose bytes can change the canonical result. The bundle digest over these files
#: is part of the run identity, so a changed implementation cannot reuse a run directory.
IMPLEMENTATION_BUNDLE_FILES = (
    "pretrain/build_pretrain_shards.py",
    "pretrain/h_census_v2.py",
    "pretrain/select_pretrain_documents.py",
    "pretrain/stage_i_graph_v2.py",
    "pretrain/stage_i_replay_v2.py",
    "src/special_tokens.py",
)

#: Authority name -> the graph ``bound_authorities`` key that pins its bytes.
AUTHORITY_KEYS = {
    "d2_d3_eligibility_manifest": "d2_d3_eligibility_manifest_sha256",
    "g2_release_manifest": "g2_manifest_sha256",
    "g_release_manifest": "g_manifest_sha256",
    "hq_policy": "hq_policy_sha256",
    "reference_exclusion": "g2_exclusion_manifest_sha256",
    "selector_v1": "selector_v1_sha256_preserved",
    "stage_e_allocation": "stage_e_allocation_sha256",
    "tokenizer": "tokenizer_sha256",
}

#: Repo-relative locations the plan generator resolves; the plan records them explicitly and the
#: authorisation boundary re-derives every digest from the recorded path.
DEFAULT_AUTHORITY_PATHS = {
    "d2_d3_eligibility_manifest": "runs/l1_production_2026-08-20/eligibility/eligibility_manifest.json",
    "g2_release_manifest": "runs/g2_production_2026-08-21/release/manifest.json",
    "g_release_manifest": "runs/g_production_2026-08-21/release/manifest.json",
    "hq_policy": "runs/h_production_2026-08-21/policy/stage_b_hq_policy_v1.json",
    "reference_exclusion": "runs/g2_production_2026-08-21/release/exclusion_hash_manifest.json",
    "selector_v1": "pretrain/select_pretrain_documents.py",
    "stage_e_allocation": "runs/stage_e_2026-08-20/allocation_contract.json",
    "tokenizer": "runs/g_production_2026-08-21/release/tokenizer.json",
}


class CensusError(RuntimeError):
    """Fail-closed census condition. Nothing is published when this is raised."""


class AuthorizationError(CensusError):
    """The run is not authorised. No production capability is created."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CensusError(message)


# --------------------------------------------------------------------- implementation identity


def file_sha256(path: Path) -> str:
    _, digest = read_authoritative_bytes(path, max_bytes=1 << 31)
    return digest


def implementation_files(repo_root: Path) -> dict[str, str]:
    """SHA-256 of every module whose bytes determine the canonical result."""
    digests: dict[str, str] = {}
    for relative in IMPLEMENTATION_BUNDLE_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise CensusError(f"implementation bundle member missing: {relative}")
        digests[relative] = file_sha256(path)
    return digests


def implementation_bundle_sha256(files: dict[str, str]) -> str:
    """One digest over the exact member list and their digests."""
    payload = {"schema_version": BUNDLE_SCHEMA, "files": dict(sorted(files.items()))}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def run_identity(
    *, plan_sha256: str, graph_sha256: str, bundle_sha256: str, census_schema: str = CENSUS_SCHEMA
) -> str:
    """The run's name. A changed plan, graph or implementation is a different run, always."""
    payload = {
        "schema_version": RUN_IDENTITY_SCHEMA,
        "candidate_plan_sha256": plan_sha256,
        "owner_graph_sha256": graph_sha256,
        "implementation_bundle_sha256": bundle_sha256,
        "census_schema_version": census_schema,
        "node_result_schema_version": NODE_RESULT_SCHEMA,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


# --------------------------------------------------------------------- score accessor


def read_score(row: dict[str, Any], accessor: dict[str, Any], which: str) -> Any:
    """Read a score strictly through the release-pinned accessor.

    The FineWeb release stores the LITERAL key ``"metadata.int_score"`` inside its ``metadata``
    object. Generic dotted-path splitting silently fails there, so the accessor names a container
    and an exact key and nothing else is attempted. The declared JSON type is enforced exactly:
    an integer where a float is declared is a schema disagreement, not a convenience.
    """
    spec = accessor.get(which)
    if spec is None:
        return None
    container = spec["container"]
    key = spec["key"]
    holder = row.get(container)
    if not isinstance(holder, dict):
        raise CensusError(f"accessor container {container!r} missing or not an object")
    if key not in holder:
        raise CensusError(f"accessor key {key!r} missing from {container!r}")
    value = holder[key]
    expected = spec["json_type"]
    if isinstance(value, bool):
        raise CensusError(f"{key!r} must be a JSON {expected}, got boolean")
    if expected == "int":
        if not isinstance(value, int):
            raise CensusError(f"{key!r} must be a JSON integer, got {type(value).__name__}")
    elif expected == "float":
        if not isinstance(value, float):
            raise CensusError(
                f"{key!r} must be a JSON float, got {type(value).__name__}; the accessor "
                "distinguishes integer and continuous scores and will not coerce between them"
            )
    return value


# --------------------------------------------------------------------- scanning


def scan_binding(
    binding: InputBinding, tokenizer_path: str, excluded_rows: Any
) -> tuple[list[CandidateRecord], dict[str, int]]:
    """Stream one frozen release once and emit its eligible candidate metadata records.

    The corpus is opened exactly once. Its digest is computed from the same descriptor the rows
    are parsed from and compared with the frozen ``documents_sha256`` at the end, and the opened
    object's device, inode and size are compared before and after. There is no path reopened
    between validating and consuming, because there is no second open.
    """
    excluded = set(int(row) for row in excluded_rows.tolist())
    _require(
        len(excluded) == binding.excluded_rows,
        f"{binding.input_binding_id}: eligibility rows are not unique",
    )
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

    with open_authoritative(binding.documents_path, buffering=8 * 1024 * 1024) as (handle, _id):
        for row_index, raw in enumerate(handle):
            digest.update(raw)
            counters["physical_rows"] += 1
            if row_index in excluded:
                counters["d2_d3_excluded_rows"] += 1
                continue
            where = f"{binding.input_binding_id}:{row_index}"
            try:
                row = strict_json_object(raw, where=where)
            except StrictJSONError as exc:
                raise CensusError(str(exc)) from exc
            if SELECTION_METADATA_FIELD in row:
                raise CensusError(f"{where}: reserved field {SELECTION_METADATA_FIELD!r} present")
            text = row.get(binding.text_field)
            if text is None:
                raise CensusError(f"{where}: missing {binding.text_field!r}")
            if not isinstance(text, str):
                raise CensusError(f"{where}: {binding.text_field!r} is not a string")
            if not text:
                counters["empty_or_non_string_text"] += 1
                continue
            cleaned = clean_text(
                text,
                strip_leading_noise=cleaning["strip_leading_noise"],
                normalize_quotes=cleaning["normalize_quotes"],
                underscores_policy=cleaning["underscores_policy"],
                min_chars=cleaning["min_chars"],
                min_ascii_ratio=cleaning["min_ascii_ratio"],
            )
            if cleaned is None:
                counters["empty_after_cleaning"] += 1
                continue
            ids, content, boundary = encode_with_accounting(
                tokenizer, cleaned, add_bos=True, add_eos=True, bos_id=BOS_ID, eos_id=EOS_ID
            )
            if not ids or ids[0] != BOS_ID or ids[-1] != EOS_ID or boundary != 2:
                raise CensusError(f"{where}: framing violation")
            leaked = SPECIAL_IDS.intersection(ids[1:-1])
            if leaked:
                raise CensusError(f"{where}: text encoded to special ids {sorted(leaked)}")
            int_score = read_score(row, accessor, "integer_score")
            raw_cont = read_score(row, accessor, "continuous_score")
            try:
                record_digest = hashlib.sha256(_canonical_json_bytes(row)).hexdigest()
            except ValueError as exc:
                raise CensusError(f"{where}: {exc}") from exc
            records.append(
                CandidateRecord(
                    input_binding_id=binding.input_binding_id,
                    row_index=row_index,
                    cleaned_sha256=cleaned_text_sha256(cleaned),
                    canonical_fingerprint=canonical_document_fingerprint(cleaned),
                    serialized_tokens=content + boundary,
                    representative_key=representative_key(
                        raw_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        input_record_sha256=record_digest,
                    ),
                    int_score=int_score,
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


def load_reference_exclusion(path: Path, *, expected_sha256: str | None = None) -> set[str]:
    """Load the whole-reserve exclusion identities from one verified descriptor."""
    payload, digest = read_authoritative_bytes(path, max_bytes=1 << 31)
    if expected_sha256 is not None and digest != expected_sha256:
        raise CensusError(f"reference exclusion manifest SHA mismatch at {path}")
    manifest = strict_json_object(payload, where="reference exclusion manifest")
    if manifest.get("exclusion_scope") != EXCLUSION_SCOPE:
        raise CensusError("reference exclusion manifest is not the whole-reserve scope")
    hashes = manifest.get("hashes")
    if not isinstance(hashes, list):
        raise CensusError("reference exclusion manifest has no hash list")
    identities = set(hashes)
    count = manifest.get("hash_count")
    if isinstance(count, bool) or not isinstance(count, int):
        raise CensusError("reference exclusion hash_count must be an exact integer")
    if len(identities) != count or len(hashes) != count:
        raise CensusError("reference exclusion hash_count disagrees with the hash list")
    return identities


def census_body(
    graph: SourceGraph, tokenizer_path: str, reference_exclusion: set[str]
) -> dict[str, Any]:
    """Run the metadata census and feasibility replay. Publishes nothing and authorises nothing.

    This is the pure helper: it is callable without an authorisation context, and what it returns
    carries no run identity, so :func:`validate_complete_census` will refuse it and it can never
    reach a canonical directory.
    """
    by_binding: dict[str, list[CandidateRecord]] = {}
    counters: dict[str, dict[str, int]] = {}
    for binding_id in sorted(graph.bindings):
        records, stats = scan_binding(
            graph.bindings[binding_id],
            tokenizer_path,
            graph.validated_eligibility_rows(binding_id),
        )
        by_binding[binding_id] = records
        counters[binding_id] = stats

    results = replay(graph, by_binding, reference_exclusion)
    return {
        "schema_version": CENSUS_SCHEMA,
        "output_label": OUTPUT_LABEL,
        "resume_supported": RESUME_SUPPORTED,
        "graph_sha256": graph.graph_sha256,
        "seed": graph.seed,
        "bound_authorities": dict(sorted(graph.bound_authorities.items())),
        "reference_exclusion_identities": len(reference_exclusion),
        "binding_counters": {k: dict(sorted(v.items())) for k, v in sorted(counters.items())},
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


# --------------------------------------------------------------------- authorisation


@dataclass(frozen=True)
class AuthorizedRunContext:
    """The single capability that can publish canonical Stage-H output.

    Holding one of these is proof that an owner-supplied digest matched the plan bytes, that
    those exact bytes parsed under the closed plan schema, and that every field in them agreed
    with the frozen graph, the bound authorities and this implementation's bytes.
    """

    repo_root: Path
    plan_path: Path
    plan_sha256: str
    plan: dict[str, Any]
    graph: SourceGraph
    graph_sha256: str
    bundle_sha256: str
    bundle_files: dict[str, str]
    authority_paths: dict[str, Path]
    authority_sha256: dict[str, str]
    identity: str = field(default="")

    @property
    def tokenizer_path(self) -> Path:
        return self.authority_paths["tokenizer"]

    @property
    def reference_exclusion_path(self) -> Path:
        return self.authority_paths["reference_exclusion"]

    def as_canonical(self) -> dict[str, Any]:
        return {
            "run_identity": self.identity,
            "candidate_plan_sha256": self.plan_sha256,
            "owner_graph_sha256": self.graph_sha256,
            "implementation_bundle_sha256": self.bundle_sha256,
            "implementation_files": dict(sorted(self.bundle_files.items())),
            "authority_sha256": dict(sorted(self.authority_sha256.items())),
            "census_schema_version": CENSUS_SCHEMA,
            "node_result_schema_version": NODE_RESULT_SCHEMA,
            "plan_schema_version": PLAN_SCHEMA,
            "authorization_status": "OWNER_SUPPLIED_EXPECTED_PLAN_SHA256",
        }

    def revalidate(self) -> None:
        """Re-derive every authority digest from disk; used immediately before publication."""
        _, plan_digest = read_authoritative_bytes(self.plan_path, max_bytes=1 << 31)
        if plan_digest != self.plan_sha256:
            raise AuthorizationError("plan bytes changed after authorisation")
        _, graph_digest = read_authoritative_bytes(Path(self.plan["graph_path"]), max_bytes=1 << 31)
        if graph_digest != self.graph_sha256:
            raise AuthorizationError("owner graph bytes changed after authorisation")
        current = implementation_files(self.repo_root)
        if current != self.bundle_files:
            raise AuthorizationError("implementation bytes changed after authorisation")
        for name, path in sorted(self.authority_paths.items()):
            if file_sha256(path) != self.authority_sha256[name]:
                raise AuthorizationError(f"authority {name!r} changed after authorisation")
        # Rebind every per-binding authority too. The release manifests and eligibility indexes are
        # small, so re-deriving them here costs nothing and closes the window between the scan and
        # the publication. The 151.6 GB document corpus is deliberately not re-hashed: the
        # authoritative scan already hashed it on the descriptor it consumed, and re-reading it
        # from the path afterwards would be the very hash-then-reopen pattern this removes. What is
        # rechecked is that each documents file is still the same opened object at the same size.
        for key in sorted(self.graph.bindings):
            binding = self.graph.bindings[key]
            with open_authoritative(binding.documents_path) as (_stream, identity):
                if identity.st_size != binding.documents_size_bytes:
                    raise AuthorizationError(f"{key}: documents size changed after authorisation")
            if file_sha256(binding.release_manifest_path) != binding.release_manifest_sha256:
                raise AuthorizationError(f"{key}: release manifest changed after authorisation")
            if file_sha256(binding.eligibility_index_path) != binding.eligibility_index_sha256:
                raise AuthorizationError(f"{key}: eligibility index changed after authorisation")


def _plan_binding_projection(binding: InputBinding) -> dict[str, Any]:
    return {
        "documents_path": str(binding.documents_path),
        "documents_sha256": binding.documents_sha256,
        "documents_size_bytes": binding.documents_size_bytes,
        "eligibility_index_path": str(binding.eligibility_index_path),
        "eligibility_index_sha256": binding.eligibility_index_sha256,
        "expected_eligible_rows": binding.expected_eligible_rows,
        "release_manifest_path": str(binding.release_manifest_path),
        "release_manifest_sha256": binding.release_manifest_sha256,
        "schema_accessor_id": binding.schema_accessor["accessor_id"],
        "total_physical_rows": binding.total_physical_rows,
    }


def _plan_node_projection(graph: SourceGraph) -> list[dict[str, Any]]:
    return [
        {
            "input_binding_ids": list(node.input_binding_ids),
            "selection_mode": node.selection_mode,
            "source_id": node.source_id,
            "stage": node.stage,
            "stage_priority": node.stage_priority,
            "target_serialized_tokens": node.target_serialized_tokens,
        }
        for node in graph.nodes
    ]


_PLAN_REQUIRED_KEYS = {
    "authorities",
    "authorization_note",
    "authorization_status",
    "bound_authorities",
    "census_schema_version",
    "graph_path",
    "graph_sha256",
    "h_publishes_physical_views",
    "implementation_bundle_sha256",
    "implementation_commit",
    "implementation_files",
    "input_bindings",
    "node_order",
    "node_result_schema_version",
    "nodes",
    "resume_supported",
    "schema_version",
    "seed",
}


def _parse_plan(payload: bytes, plan_path: Path) -> dict[str, Any]:
    """Strictly parse the exact approved bytes under the closed plan schema."""
    plan = strict_json_object(payload, where=f"candidate plan {plan_path}")
    # The schema version is checked first so a superseded plan is refused as the wrong schema
    # rather than as an incidental key mismatch.
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise AuthorizationError(
            f"candidate plan: schema_version must be {PLAN_SCHEMA}, "
            f"got {plan.get('schema_version')!r}"
        )
    unknown = sorted(set(plan) - _PLAN_REQUIRED_KEYS)
    if unknown:
        raise AuthorizationError(f"candidate plan: unknown keys {unknown}")
    missing = sorted(_PLAN_REQUIRED_KEYS - set(plan))
    if missing:
        raise AuthorizationError(f"candidate plan: missing required keys {missing}")
    if plan["census_schema_version"] != CENSUS_SCHEMA:
        raise AuthorizationError("candidate plan: census_schema_version mismatch")
    if plan["node_result_schema_version"] != NODE_RESULT_SCHEMA:
        raise AuthorizationError("candidate plan: node_result_schema_version mismatch")
    if plan["authorization_status"] != "NOT_AUTHORIZED":
        raise AuthorizationError(
            "candidate plan: authorization_status must remain NOT_AUTHORIZED; a plan cannot "
            "authorise itself and a self-labelled authorisation is exactly the forgery this "
            "boundary exists to refuse"
        )
    if plan["resume_supported"] is not False:
        raise AuthorizationError("candidate plan: resume_supported must be false")
    if plan["h_publishes_physical_views"] is not False:
        raise AuthorizationError("candidate plan: H must not publish physical candidate views")
    for key in (
        "graph_path",
        "graph_sha256",
        "implementation_bundle_sha256",
        "implementation_commit",
        "authorization_note",
    ):
        if not isinstance(plan[key], str) or not plan[key]:
            raise AuthorizationError(f"candidate plan: {key} must be a non-empty string")
    if isinstance(plan["seed"], bool) or not isinstance(plan["seed"], int):
        raise AuthorizationError("candidate plan: seed must be an exact integer")
    for key in ("bound_authorities", "authorities", "implementation_files", "input_bindings"):
        if not isinstance(plan[key], dict):
            raise AuthorizationError(f"candidate plan: {key} must be an object")
    for key in ("nodes", "node_order"):
        if not isinstance(plan[key], list):
            raise AuthorizationError(f"candidate plan: {key} must be an array")
    return plan


def authorize_run(
    *,
    plan_path: Path,
    expected_plan_sha256: str | None,
    repo_root: Path | None = None,
) -> AuthorizedRunContext:
    """Build the production capability, or refuse.

    The order matters. The digest is taken over bytes already held in memory and the plan is
    parsed from those same bytes, so an approved digest can never certify one file while a second
    read interprets another.
    """
    if not expected_plan_sha256:
        raise AuthorizationError(
            "--expected-plan-sha256 is required; a plan may not authorise itself"
        )
    if (
        not isinstance(expected_plan_sha256, str)
        or len(expected_plan_sha256) != 64
        or set(expected_plan_sha256) - set("0123456789abcdef")
    ):
        raise AuthorizationError(
            "expected plan SHA-256 must be a lowercase 64-character hex digest"
        )
    repo_root = Path(repo_root or ROOT).resolve()

    payload, actual = read_authoritative_bytes(plan_path, max_bytes=1 << 31)
    if actual != expected_plan_sha256:
        raise AuthorizationError(
            f"plan SHA-256 mismatch: expected {expected_plan_sha256}, got {actual}"
        )
    plan = _parse_plan(payload, plan_path)

    graph_path = Path(plan["graph_path"])
    graph = load_source_graph(
        graph_path, verify_hashes=True, expected_graph_sha256=plan["graph_sha256"]
    )
    if graph.graph_sha256 != plan["graph_sha256"]:
        raise AuthorizationError("owner graph SHA-256 disagrees with the approved plan")
    if graph.seed != plan["seed"]:
        raise AuthorizationError("owner graph seed disagrees with the approved plan")
    if dict(sorted(graph.bound_authorities.items())) != dict(
        sorted(plan["bound_authorities"].items())
    ):
        raise AuthorizationError("bound authorities disagree with the approved plan")

    bundle_files = implementation_files(repo_root)
    if bundle_files != plan["implementation_files"]:
        raise AuthorizationError(
            "implementation files disagree with the approved plan; this is a different tool"
        )
    bundle_sha = implementation_bundle_sha256(bundle_files)
    if bundle_sha != plan["implementation_bundle_sha256"]:
        raise AuthorizationError("implementation bundle SHA-256 disagrees with the approved plan")

    authority_paths: dict[str, Path] = {}
    authority_sha256: dict[str, str] = {}
    declared = plan["authorities"]
    if sorted(declared) != sorted(AUTHORITY_KEYS):
        raise AuthorizationError(
            f"candidate plan: authorities must be exactly {sorted(AUTHORITY_KEYS)}"
        )
    for name, bound_key in sorted(AUTHORITY_KEYS.items()):
        entry = declared[name]
        if not isinstance(entry, dict) or sorted(entry) != ["path", "sha256"]:
            raise AuthorizationError(f"candidate plan: authority {name!r} must be {{path, sha256}}")
        path = Path(entry["path"])
        if not path.is_absolute():
            path = repo_root / path
        digest = file_sha256(path)
        if digest != entry["sha256"]:
            raise AuthorizationError(f"authority {name!r} bytes disagree with the approved plan")
        if digest != graph.bound_authorities[bound_key]:
            raise AuthorizationError(
                f"authority {name!r} bytes disagree with the owner graph's {bound_key}"
            )
        authority_paths[name] = path
        authority_sha256[name] = digest
    if set(AUTHORITY_KEYS.values()) != set(BOUND_AUTHORITY_KEYS):
        raise AuthorizationError("authority table does not cover every bound authority")

    expected_bindings = {
        key: _plan_binding_projection(binding) for key, binding in sorted(graph.bindings.items())
    }
    if plan["input_bindings"] != expected_bindings:
        raise AuthorizationError("input bindings disagree with the approved plan")
    if plan["nodes"] != _plan_node_projection(graph):
        raise AuthorizationError("node targets or order disagree with the approved plan")
    if plan["node_order"] != [node.source_id for node in graph.nodes]:
        raise AuthorizationError("node order disagrees with the approved plan")

    identity = run_identity(
        plan_sha256=actual, graph_sha256=graph.graph_sha256, bundle_sha256=bundle_sha
    )
    return AuthorizedRunContext(
        repo_root=repo_root,
        plan_path=plan_path,
        plan_sha256=actual,
        plan=plan,
        graph=graph,
        graph_sha256=graph.graph_sha256,
        bundle_sha256=bundle_sha,
        bundle_files=bundle_files,
        authority_paths=authority_paths,
        authority_sha256=authority_sha256,
        identity=identity,
    )


def build_census(context: AuthorizedRunContext) -> dict[str, Any]:
    """Run the authorised census and stamp it with the run identity it belongs to."""
    if not isinstance(context, AuthorizedRunContext):
        raise AuthorizationError("build_census requires an AuthorizedRunContext")
    exclusion = load_reference_exclusion(
        context.reference_exclusion_path,
        expected_sha256=context.authority_sha256["reference_exclusion"],
    )
    body = census_body(context.graph, str(context.tokenizer_path), exclusion)
    body["status"] = "COMPLETE"
    body["authorization"] = context.as_canonical()
    return body


# --------------------------------------------------------------------- validation & publication


def validate_complete_census(census: Any, context: AuthorizedRunContext) -> None:
    """Everything that must hold before the word COMPLETE may be written anywhere.

    A result that is incomplete, malformed, mis-schemaed or internally inconsistent fails here,
    before any staging file exists, so no partial artefact can be labelled complete.
    """
    if not isinstance(census, dict):
        raise CensusError("census must be an object")
    required = {
        "authorization",
        "binding_counters",
        "bound_authorities",
        "graph_sha256",
        "hard_stop_required",
        "nodes",
        "output_label",
        "ownership_matrix",
        "reference_exclusion_identities",
        "resume_supported",
        "schema_version",
        "seed",
        "status",
        "totals",
    }
    unknown = sorted(set(census) - required)
    _require(not unknown, f"census: unknown keys {unknown}")
    missing = sorted(required - set(census))
    _require(not missing, f"census: missing required keys {missing}")
    _require(census["schema_version"] == CENSUS_SCHEMA, "census: wrong schema_version")
    _require(census["status"] == "COMPLETE", f"census: status is {census['status']!r}")
    _require(census["output_label"] == OUTPUT_LABEL, "census: wrong output_label")
    _require(census["resume_supported"] is False, "census: resume must be disabled")
    _require(census["graph_sha256"] == context.graph_sha256, "census: graph SHA disagreement")
    _require(census["seed"] == context.graph.seed, "census: seed disagreement")
    _require(
        census["authorization"] == context.as_canonical(),
        "census: authorization block does not match the authorised run context",
    )
    _require(isinstance(census["hard_stop_required"], bool), "census: hard_stop_required not bool")

    nodes = census["nodes"]
    _require(isinstance(nodes, list) and nodes, "census: nodes must be a non-empty array")
    expected_order = [node.source_id for node in context.graph.nodes]
    _require(
        [n.get("source_id") for n in nodes] == expected_order,
        "census: node results do not cover the graph's nodes in execution order",
    )
    targets = {node.source_id: node.target_serialized_tokens for node in context.graph.nodes}
    for node in nodes:
        _validate_node_result(node, targets)

    counters = census["binding_counters"]
    _require(isinstance(counters, dict), "census: binding_counters must be an object")
    _require(
        sorted(counters) == sorted(context.graph.bindings),
        "census: binding counters do not cover exactly the graph's bindings",
    )
    for key, stats in sorted(counters.items()):
        binding = context.graph.bindings[key]
        _require(
            stats["physical_rows"] == binding.total_physical_rows,
            f"census: {key} physical row count is not final",
        )
        _require(
            stats["d2_d3_excluded_rows"] == binding.excluded_rows,
            f"census: {key} eligibility exclusions are not fully consumed",
        )
        accounted = (
            stats["d2_d3_excluded_rows"]
            + stats["empty_or_non_string_text"]
            + stats["empty_after_cleaning"]
            + stats["eligible_rows"]
        )
        _require(
            accounted == stats["physical_rows"],
            f"census: {key} row dispositions do not account for every physical row",
        )

    totals = census["totals"]
    _require(
        totals["physical_rows"] == sum(c["physical_rows"] for c in counters.values()),
        "census: totals.physical_rows disagrees with the per-binding counters",
    )
    _require(
        totals["eligible_rows"] == sum(c["eligible_rows"] for c in counters.values()),
        "census: totals.eligible_rows disagrees with the per-binding counters",
    )
    _require(
        totals["selected_identities"] == sum(n["selected_identities"] for n in nodes),
        "census: totals.selected_identities disagrees with the node results",
    )
    _require(
        totals["selected_serialized_tokens"] == sum(n["selected_serialized_tokens"] for n in nodes),
        "census: totals.selected_serialized_tokens disagrees with the node results",
    )
    _require(
        census["hard_stop_required"] == any(not n["feasible"] for n in nodes),
        "census: hard_stop_required disagrees with the node feasibility states",
    )

    try:
        encoded = canonical_json_bytes(census)
    except ValueError as exc:
        raise CensusError(f"census is not canonically serialisable: {exc}") from exc
    _require(
        canonical_json_bytes(json.loads(encoded.decode("utf-8"))) == encoded,
        "census: canonical serialisation is not a fixed point",
    )


def _validate_node_result(node: Any, targets: dict[str, int]) -> None:
    _require(isinstance(node, dict), "census: node result must be an object")
    _require(
        node.get("canonical_schema_version") == NODE_RESULT_SCHEMA,
        "census: node result carries the wrong canonical schema version",
    )
    source_id = node["source_id"]
    target = targets[source_id]
    _require(
        node["target_serialized_tokens"] == target,
        f"census: {source_id} target disagrees with the frozen graph",
    )
    for key in (
        "pre_exclusion_unique_identities",
        "g2_excluded_identities",
        "prior_commit_excluded_identities",
        "post_exclusion_candidate_identities",
        "post_exclusion_candidate_serialized_tokens",
        "selected_identities",
        "selected_serialized_tokens",
        "residual_identities",
        "residual_serialized_tokens",
        "actual_overshoot_tokens",
    ):
        value = node[key]
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"census: {source_id}.{key} must be a non-negative exact integer",
        )
    _require(isinstance(node["feasible"], bool), f"census: {source_id}.feasible must be a boolean")
    _require(
        node["prior_commit_excluded_identities"] == sum(node["exclusions_by_owner"].values()),
        f"census: {source_id} ownership exclusions do not sum to the reported total",
    )
    _require(
        node["pre_exclusion_unique_identities"]
        == node["g2_excluded_identities"]
        + node["prior_commit_excluded_identities"]
        + node["post_exclusion_candidate_identities"],
        f"census: {source_id} candidate accounting does not close",
    )
    _require(
        node["residual_identities"]
        == node["post_exclusion_candidate_identities"] - node["selected_identities"],
        f"census: {source_id} residual identities do not close",
    )
    _require(
        node["residual_serialized_tokens"]
        == node["post_exclusion_candidate_serialized_tokens"] - node["selected_serialized_tokens"],
        f"census: {source_id} residual tokens do not close",
    )
    _require(
        node["feasible"] == (node["selected_serialized_tokens"] >= target),
        f"census: {source_id} feasibility disagrees with its selected mass",
    )
    expected_overshoot = node["selected_serialized_tokens"] - target if node["feasible"] else 0
    _require(
        node["actual_overshoot_tokens"] == expected_overshoot,
        f"census: {source_id} overshoot is not the measured value",
    )
    if node["feasible"]:
        _require(
            isinstance(node["crossing_identity"], str),
            f"census: {source_id} is feasible but reports no crossing document",
        )
        _require(
            0 < node["crossing_document_serialized_tokens"] <= node["selected_serialized_tokens"],
            f"census: {source_id} crossing mass is outside its selected mass",
        )
    else:
        _require(
            node["crossing_identity"] is None
            and node["crossing_document_serialized_tokens"] is None,
            f"census: {source_id} is infeasible but reports a crossing document",
        )
    evidence = node["boundary_evidence"]
    _require(isinstance(evidence, dict), f"census: {source_id} boundary evidence must be an object")


def final_run_dir(out_dir: Path, context: AuthorizedRunContext) -> Path:
    return out_dir / f"run-{context.identity[:32]}"


def _staging_prefix(context: AuthorizedRunContext) -> str:
    return f".run-{context.identity[:32]}.staging-"


def publish_atomic(context: AuthorizedRunContext, census: dict[str, Any], out_dir: Path) -> Path:
    """Validate, stage, verify the staged bytes, mark COMPLETE last, then rename atomically.

    Ordering is the whole point. Nothing is written until the census has passed strict validation
    and every authority has been re-derived from disk; the marker is written after the payload and
    verified read-back; and the directory only becomes discoverable under its final name once it
    already contains a complete, self-consistent run. A failure at any point removes the staging
    tree and leaves any previous state untouched.
    """
    if not isinstance(context, AuthorizedRunContext):
        raise AuthorizationError("publication requires an AuthorizedRunContext")
    validate_complete_census(census, context)
    context.revalidate()

    final = final_run_dir(out_dir, context)
    if final.exists():
        raise CensusError(f"run directory already exists, refusing to overwrite: {final}")
    out_dir.mkdir(parents=True, exist_ok=True)
    stale = sorted(out_dir.glob(_staging_prefix(context) + "*"))
    if stale:
        raise CensusError(
            f"stale staging directories for this run identity must be removed first: {stale}"
        )

    payload = canonical_json_bytes(census)
    marker = canonical_json_bytes({
        "marker": COMPLETE_MARKER,
        "run_identity": context.identity,
        "candidate_plan_sha256": context.plan_sha256,
        "owner_graph_sha256": context.graph_sha256,
        "implementation_bundle_sha256": context.bundle_sha256,
        "census_sha256": hashlib.sha256(payload).hexdigest(),
        "census_schema_version": CENSUS_SCHEMA,
    })
    staging = Path(tempfile.mkdtemp(prefix=_staging_prefix(context), dir=str(out_dir)))
    try:
        _write_durable(staging / CENSUS_FILENAME, payload)
        written = (staging / CENSUS_FILENAME).read_bytes()
        if written != payload:
            raise CensusError("staged census bytes do not match what was serialised")
        # COMPLETE is written last, and only now: everything it attests to already exists.
        _write_durable(staging / COMPLETE_MARKER, marker)
        _fsync_dir(staging)
        os.replace(staging, final)
        _fsync_dir(out_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final


def _write_durable(path: Path, payload: bytes) -> None:
    with open(path, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_published_run(final: Path, *, expected_run_identity: str | None = None) -> dict[str, Any]:
    """Strict consumer: refuse anything that is not a complete, self-consistent published run."""
    if not final.is_dir():
        raise CensusError(f"{final}: not a published run directory")
    marker_path = final / COMPLETE_MARKER
    census_path = final / CENSUS_FILENAME
    if not marker_path.is_file():
        raise CensusError(f"{final}: no COMPLETE marker; the run is not publishable output")
    payload, census_digest = read_authoritative_bytes(census_path, max_bytes=1 << 31)
    marker_bytes, _ = read_authoritative_bytes(marker_path, max_bytes=1 << 20)
    marker = strict_json_object(marker_bytes, where=f"{final}/{COMPLETE_MARKER}")
    if marker.get("marker") != COMPLETE_MARKER:
        raise CensusError(f"{final}: malformed COMPLETE marker")
    if marker.get("census_sha256") != census_digest:
        raise CensusError(f"{final}: COMPLETE marker does not describe the published census")
    census = strict_json_object(payload, where=f"{final}/{CENSUS_FILENAME}")
    if census.get("schema_version") != CENSUS_SCHEMA or census.get("status") != "COMPLETE":
        raise CensusError(f"{final}: published census is not a COMPLETE {CENSUS_SCHEMA}")
    authorization = census.get("authorization")
    if not isinstance(authorization, dict):
        raise CensusError(f"{final}: published census carries no run identity")
    for key in (
        "run_identity",
        "candidate_plan_sha256",
        "owner_graph_sha256",
        "implementation_bundle_sha256",
    ):
        if marker.get(key) != authorization.get(key):
            raise CensusError(f"{final}: COMPLETE marker and census disagree on {key}")
    if final.name != f"run-{authorization['run_identity'][:32]}":
        raise CensusError(f"{final}: directory name does not match its own run identity")
    if expected_run_identity is not None and authorization["run_identity"] != expected_run_identity:
        raise CensusError(f"{final}: run identity is not the expected one")
    return census


# --------------------------------------------------------------------- candidate plan


def generate_candidate_plan(
    *,
    graph_path: Path,
    graph: SourceGraph,
    repo_root: Path,
    implementation_commit: str,
    authority_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    """A candidate plan that cannot authorise itself, emitting every canonical field.

    Every field the authorisation boundary reads is produced here, so the sealed owner-facing
    plan is reproducible byte-for-byte from this generator and nothing has to be added by hand.
    """
    paths = dict(authority_paths or DEFAULT_AUTHORITY_PATHS)
    if sorted(paths) != sorted(AUTHORITY_KEYS):
        raise CensusError(f"authority paths must be exactly {sorted(AUTHORITY_KEYS)}")
    authorities: dict[str, Any] = {}
    for name, relative in sorted(paths.items()):
        path = Path(relative)
        if not path.is_absolute():
            path = repo_root / path
        digest = file_sha256(path)
        bound = graph.bound_authorities[AUTHORITY_KEYS[name]]
        if digest != bound:
            raise CensusError(
                f"authority {name!r} at {path} hashes to {digest}, but the owner graph binds {bound}"
            )
        authorities[name] = {"path": relative, "sha256": digest}

    files = implementation_files(repo_root)
    return {
        "schema_version": PLAN_SCHEMA,
        "authorization_status": "NOT_AUTHORIZED",
        "authorization_note": (
            "This plan carries no owner authorization and cannot create one. Stage-H production "
            "must be invoked with --expected-plan-sha256 supplied externally by the owner, and "
            "that authorization applies only to these exact plan bytes, this graph, these "
            "authorities and this implementation bundle."
        ),
        "census_schema_version": CENSUS_SCHEMA,
        "node_result_schema_version": NODE_RESULT_SCHEMA,
        "graph_path": str(graph_path),
        "graph_sha256": graph.graph_sha256,
        "seed": graph.seed,
        "resume_supported": RESUME_SUPPORTED,
        "h_publishes_physical_views": False,
        "bound_authorities": dict(sorted(graph.bound_authorities.items())),
        "authorities": authorities,
        "implementation_commit": implementation_commit,
        "implementation_files": files,
        "implementation_bundle_sha256": implementation_bundle_sha256(files),
        "input_bindings": {
            key: _plan_binding_projection(binding)
            for key, binding in sorted(graph.bindings.items())
        },
        "nodes": _plan_node_projection(graph),
        "node_order": [node.source_id for node in graph.nodes],
    }


def _git_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise CensusError(f"cannot determine implementation commit: {result.stderr.strip()}")
    return result.stdout.strip()


# --------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    # Checked before argparse so a resume attempt fails on its own terms rather than as a
    # generic usage error, in every spelling.
    for token in raw_argv:
        if token == "--resume" or token.startswith("--resume=") or token.startswith("--resume-"):
            raise CensusError("resume is not supported")

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan_parser = sub.add_parser("plan", help="generate an unauthorized candidate plan")
    plan_parser.add_argument("--graph", type=Path, required=True)
    plan_parser.add_argument("--out", type=Path, required=True)
    plan_parser.add_argument("--repo-root", type=Path, default=Path(ROOT))
    plan_parser.add_argument("--implementation-commit", type=str, default=None)

    run_parser = sub.add_parser("run", help="run the authorized census and publish it")
    run_parser.add_argument("--plan", type=Path, required=True)
    run_parser.add_argument("--expected-plan-sha256", type=str, default=None)
    run_parser.add_argument("--out-dir", type=Path, required=True)
    run_parser.add_argument("--repo-root", type=Path, default=Path(ROOT))

    args = parser.parse_args(raw_argv)

    if args.command == "plan":
        repo_root = args.repo_root.resolve()
        graph = load_source_graph(args.graph, verify_hashes=True)
        plan = generate_candidate_plan(
            graph_path=args.graph,
            graph=graph,
            repo_root=repo_root,
            implementation_commit=args.implementation_commit or _git_head(repo_root),
        )
        payload = canonical_json_bytes(plan)
        if args.out.exists():
            raise CensusError(f"candidate plan already exists, refusing to overwrite: {args.out}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        _write_durable(args.out, payload)
        print(f"{hashlib.sha256(payload).hexdigest()}  {args.out}")
        return 0

    context = authorize_run(
        plan_path=args.plan,
        expected_plan_sha256=args.expected_plan_sha256,
        repo_root=args.repo_root,
    )
    census = build_census(context)
    final = publish_atomic(context, census, args.out_dir)
    print(f"published {final}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CensusError, GraphError, ReplayError) as exc:
        print(f"FAIL-CLOSED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
