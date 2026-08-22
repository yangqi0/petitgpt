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
from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import MappingProxyType
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
    BRANCH_DEPENDENT,
    STAGE_PRIORITY,
    FileIdentity,
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
    BOUNDARY_EVIDENCE_FIELDS,
    NODE_RESULT_FIELDS,
    NODE_RESULT_SCHEMA,
    REPRESENTATIVE_RULE,
    CandidateRecord,
    ReplayError,
    bits_to_score,
    ownership_matrix,
    replay,
    representative_key,
    score_to_bits,
)
from src.special_tokens import BOS_ID, EOS_ID, SPECIAL_TOKEN_IDS  # noqa: E402

CENSUS_SCHEMA = "petitgpt-h-census-v2"
PLAN_SCHEMA = "petitgpt-h-candidate-plan-v4"
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

_HEX = frozenset("0123456789abcdef")
_CENSUS_FIELDS = frozenset({
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
})
_AUTHORIZATION_FIELDS = frozenset({
    "run_identity",
    "candidate_plan_sha256",
    "owner_graph_sha256",
    "implementation_bundle_sha256",
    "implementation_files",
    "authority_sha256",
    "census_schema_version",
    "node_result_schema_version",
    "plan_schema_version",
    "authorization_status",
})
_COUNTER_FIELDS = frozenset({
    "physical_rows",
    "d2_d3_excluded_rows",
    "empty_or_non_string_text",
    "empty_after_cleaning",
    "eligible_rows",
})
_TOTAL_FIELDS = frozenset({
    "eligible_rows",
    "physical_rows",
    "selected_serialized_tokens",
    "selected_identities",
})
_MARKER_FIELDS = frozenset({
    "marker",
    "run_identity",
    "candidate_plan_sha256",
    "owner_graph_sha256",
    "implementation_bundle_sha256",
    "census_sha256",
    "census_schema_version",
})
_AUTHORIZED_PROVENANCE = object()


class CensusError(RuntimeError):
    """Fail-closed census condition. Nothing is published when this is raised."""


class AuthorizationError(CensusError):
    """The run is not authorised. No production capability is created."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CensusError(message)


def _closed_object(value: Any, fields: frozenset[str], where: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{where}: must be an object")
    _require(all(type(key) is str for key in value), f"{where}: keys must be strings")
    unknown = sorted(set(value) - fields)
    missing = sorted(fields - set(value))
    _require(not unknown, f"{where}: unknown keys {unknown}")
    _require(not missing, f"{where}: missing required keys {missing}")
    return value


def _exact_int(value: Any, where: str, *, minimum: int = 0) -> int:
    _require(type(value) is int, f"{where}: must be an exact integer")
    _require(value >= minimum, f"{where}: must be at least {minimum}")
    return value


def _exact_str(value: Any, where: str, *, nonempty: bool = True) -> str:
    _require(type(value) is str, f"{where}: must be a string")
    _require(not nonempty or bool(value), f"{where}: must not be empty")
    return value


def _hex_string(value: Any, length: int, where: str) -> str:
    value = _exact_str(value, where)
    _require(
        len(value) == length and not (set(value) - _HEX),
        f"{where}: must be a lowercase {length}-character hex string",
    )
    return value


def _optional_hex(value: Any, length: int, where: str) -> str | None:
    if value is None:
        return None
    return _hex_string(value, length, where)


def _resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path


def _freeze_context_value(value: Any) -> Any:
    """Copy and recursively freeze JSON-derived authorization state."""
    if type(value) is dict:
        return MappingProxyType({key: _freeze_context_value(child) for key, child in value.items()})
    if type(value) is list:
        return tuple(_freeze_context_value(child) for child in value)
    return value


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
    binding: InputBinding,
    tokenizer_path: str,
    excluded_rows: Any,
    *,
    expected_documents_identity: FileIdentity | None = None,
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

    with open_authoritative(binding.documents_path, buffering=8 * 1024 * 1024) as (
        handle,
        opened_identity,
    ):
        if (
            expected_documents_identity is not None
            and opened_identity != expected_documents_identity
        ):
            raise CensusError(f"{binding.input_binding_id}: documents identity changed before scan")
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
    graph: SourceGraph,
    tokenizer_path: str,
    reference_exclusion: set[str],
    *,
    expected_binding_identities: Mapping[str, FileIdentity] | None = None,
) -> dict[str, Any]:
    """Run the metadata census and feasibility replay. Publishes nothing and authorises nothing.

    This is the pure helper: it is callable without an authorisation context, and what it returns
    carries no run identity, so :func:`validate_complete_census` will refuse it and it can never
    reach a canonical directory.
    """
    by_binding: dict[str, list[CandidateRecord]] = {}
    counters: dict[str, dict[str, int]] = {}
    identities = (
        graph.binding_identities
        if expected_binding_identities is None
        else expected_binding_identities
    )
    if identities is None or set(identities) != set(graph.bindings):
        raise CensusError(
            "documents identities must cover exactly the validated input bindings before scan"
        )
    for binding_id in sorted(graph.bindings):
        records, stats = scan_binding(
            graph.bindings[binding_id],
            tokenizer_path,
            graph.validated_eligibility_rows(binding_id),
            expected_documents_identity=identities[binding_id],
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
    plan: Mapping[str, Any]
    graph: SourceGraph
    graph_sha256: str
    bundle_sha256: str
    bundle_files: Mapping[str, str]
    authority_paths: Mapping[str, Path]
    authority_sha256: Mapping[str, str]
    implementation_commit: str
    identity: str = field(default="")
    _provenance: object | None = field(default=None, init=False, repr=False, compare=False)
    _plan_snapshot: bytes = field(default=b"", init=False, repr=False, compare=False)
    _binding_identity_snapshot: tuple[tuple[str, FileIdentity], ...] = field(
        default=(), init=False, repr=False, compare=False
    )

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
        _require_authorized_context(self, "revalidation")
        plan_payload, plan_digest = read_authoritative_bytes(self.plan_path, max_bytes=1 << 31)
        if plan_digest != self.plan_sha256:
            raise AuthorizationError("plan bytes changed after authorisation")
        current_plan = _parse_plan(plan_payload, self.plan_path)
        if canonical_json_bytes(current_plan) != self._plan_snapshot:
            raise AuthorizationError("parsed plan changed after authorisation")
        if current_plan["implementation_commit"] != self.implementation_commit:
            raise AuthorizationError("plan implementation commit changed after authorisation")
        runtime_commit = _git_head(self.repo_root)
        if runtime_commit != self.implementation_commit:
            raise AuthorizationError(
                "runtime repository HEAD changed after authorisation: "
                f"expected {self.implementation_commit}, got {runtime_commit}"
            )

        graph_path = _resolve_repo_path(self.repo_root, current_plan["graph_path"])
        _, graph_digest = read_authoritative_bytes(graph_path, max_bytes=1 << 31)
        if graph_digest != self.graph_sha256:
            raise AuthorizationError("owner graph bytes changed after authorisation")
        if (
            self.graph.graph_sha256 != self.graph_sha256
            or current_plan["graph_sha256"] != self.graph_sha256
            or current_plan["seed"] != self.graph.seed
            or current_plan["bound_authorities"]
            != dict(sorted(self.graph.bound_authorities.items()))
        ):
            raise AuthorizationError("in-memory owner graph disagrees with the approved plan")
        expected_bindings = {
            key: _plan_binding_projection(binding)
            for key, binding in sorted(self.graph.bindings.items())
        }
        if current_plan["input_bindings"] != expected_bindings:
            raise AuthorizationError("in-memory input bindings disagree with the approved plan")
        if current_plan["nodes"] != _plan_node_projection(self.graph) or current_plan[
            "node_order"
        ] != [node.source_id for node in self.graph.nodes]:
            raise AuthorizationError("in-memory node graph disagrees with the approved plan")

        frozen_bundle_files = dict(self.bundle_files)
        if (
            current_plan["implementation_files"] != frozen_bundle_files
            or current_plan["implementation_bundle_sha256"] != self.bundle_sha256
            or implementation_bundle_sha256(frozen_bundle_files) != self.bundle_sha256
        ):
            raise AuthorizationError("implementation snapshot disagrees with the approved plan")
        current = implementation_files(self.repo_root)
        if current != frozen_bundle_files:
            raise AuthorizationError("implementation bytes changed after authorisation")

        declared_authorities = current_plan["authorities"]
        expected_authority_paths: dict[str, Path] = {}
        expected_authority_sha256: dict[str, str] = {}
        for name in sorted(AUTHORITY_KEYS):
            entry = declared_authorities[name]
            expected_authority_paths[name] = _resolve_repo_path(self.repo_root, entry["path"])
            expected_authority_sha256[name] = entry["sha256"]
        if expected_authority_paths != dict(
            self.authority_paths
        ) or expected_authority_sha256 != dict(self.authority_sha256):
            raise AuthorizationError("authority snapshot disagrees with the approved plan")
        for name, path in sorted(expected_authority_paths.items()):
            if file_sha256(path) != expected_authority_sha256[name]:
                raise AuthorizationError(f"authority {name!r} changed after authorisation")

        expected_identities = dict(self._binding_identity_snapshot)
        if (
            self.graph.binding_identities is None
            or set(expected_identities) != set(self.graph.bindings)
            or dict(self.graph.binding_identities) != expected_identities
        ):
            raise AuthorizationError(
                "documents identities do not cover exactly the authorized input bindings"
            )
        # The 151.6 GB corpus is not re-hashed. The graph captured full descriptor identities while
        # validating each binding, the scan consumed those same identities, and this final check
        # proves that every path still names that validated object before publication.
        for key, expected_identity in sorted(expected_identities.items()):
            binding = self.graph.bindings[key]
            with open_authoritative(binding.documents_path) as (_stream, identity):
                pass
            if identity.st_size != binding.documents_size_bytes:
                raise AuthorizationError(f"{key}: documents size changed after authorisation")
            if identity != expected_identity:
                raise AuthorizationError(f"{key}: documents identity changed after authorisation")
            if file_sha256(binding.release_manifest_path) != binding.release_manifest_sha256:
                raise AuthorizationError(f"{key}: release manifest changed after authorisation")
            if file_sha256(binding.eligibility_index_path) != binding.eligibility_index_sha256:
                raise AuthorizationError(f"{key}: eligibility index changed after authorisation")


def _require_authorized_context(context: Any, operation: str) -> AuthorizedRunContext:
    if (
        type(context) is not AuthorizedRunContext
        or context._provenance is not _AUTHORIZED_PROVENANCE
        or not context._plan_snapshot
        or not context._binding_identity_snapshot
    ):
        raise AuthorizationError(
            f"{operation} requires an AuthorizedRunContext produced by authorize_run"
        )
    return context


def _make_authorized_context(**values: Any) -> AuthorizedRunContext:
    graph = values["graph"]
    if graph.binding_identities is None or set(graph.binding_identities) != set(graph.bindings):
        raise AuthorizationError(
            "authorization did not validate identities for exactly every input binding"
        )
    plan_snapshot = canonical_json_bytes(values["plan"])
    identity_snapshot = tuple(sorted(graph.binding_identities.items()))
    frozen_values = dict(values)
    frozen_values["plan"] = _freeze_context_value(dict(values["plan"]))
    frozen_values["bundle_files"] = MappingProxyType(dict(values["bundle_files"]))
    frozen_values["authority_paths"] = MappingProxyType(dict(values["authority_paths"]))
    frozen_values["authority_sha256"] = MappingProxyType(dict(values["authority_sha256"]))
    context = AuthorizedRunContext(**frozen_values)
    object.__setattr__(context, "_provenance", _AUTHORIZED_PROVENANCE)
    object.__setattr__(context, "_plan_snapshot", plan_snapshot)
    object.__setattr__(context, "_binding_identity_snapshot", identity_snapshot)
    return context


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
    commit = plan["implementation_commit"]
    if len(commit) != 40 or set(commit) - _HEX:
        raise AuthorizationError(
            "candidate plan: implementation_commit must be a lowercase 40-character Git SHA"
        )
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

    runtime_commit = _git_head(repo_root)
    if runtime_commit != plan["implementation_commit"]:
        raise AuthorizationError(
            "candidate plan implementation_commit does not match runtime repository HEAD: "
            f"expected {plan['implementation_commit']}, got {runtime_commit}"
        )

    graph_path = _resolve_repo_path(repo_root, plan["graph_path"])
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
    return _make_authorized_context(
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
        implementation_commit=runtime_commit,
        identity=identity,
    )


def build_census(context: AuthorizedRunContext) -> dict[str, Any]:
    """Run the authorised census and stamp it with the run identity it belongs to."""
    _require_authorized_context(context, "build_census")
    context.revalidate()
    exclusion = load_reference_exclusion(
        context.reference_exclusion_path,
        expected_sha256=context.authority_sha256["reference_exclusion"],
    )
    body = census_body(
        context.graph,
        str(context.tokenizer_path),
        exclusion,
        expected_binding_identities=dict(context._binding_identity_snapshot),
    )
    body["status"] = "COMPLETE"
    body["authorization"] = context.as_canonical()
    return body


# --------------------------------------------------------------------- validation & publication


def _validate_score_evidence(bits_hex: Any, score_hex: Any, where: str, *, required: bool) -> None:
    if bits_hex is None or score_hex is None:
        _require(
            bits_hex is None and score_hex is None,
            f"{where}: score bits and score hex must be jointly present or absent",
        )
        _require(not required, f"{where}: exact-score evidence is required")
        return
    bits = _hex_string(bits_hex, 16, f"{where}.score_bits_hex")
    score = _exact_str(score_hex, f"{where}.score_hex")
    try:
        decoded = bits_to_score(int(bits, 16))
    except ReplayError as exc:
        raise CensusError(f"{where}: invalid binary64 score evidence: {exc}") from exc
    _require(decoded.hex() == score, f"{where}: score hex does not match the exact binary64 bits")


def _validate_contract_node(
    node: Any,
    *,
    graph_node: Any | None,
    earlier_sources: set[str],
) -> None:
    obj = _closed_object(node, NODE_RESULT_FIELDS, "census.node")
    _require(
        _exact_str(obj["canonical_schema_version"], "census.node.canonical_schema_version")
        == NODE_RESULT_SCHEMA,
        "census: node result carries the wrong canonical schema version",
    )
    source_id = _exact_str(obj["source_id"], "census.node.source_id")
    stage = _exact_str(obj["stage"], f"census: {source_id}.stage")
    _require(stage in {"stage_a", "stage_b"}, f"census: {source_id}.stage is invalid")
    branch = _exact_str(obj["branch"], f"census: {source_id}.branch")
    _require(
        branch in {"ORDINARY", "PRIMARY_GE4", "FALLBACK_RANKED_GE3"},
        f"census: {source_id}.branch is invalid",
    )
    mode = _exact_str(obj["selection_mode"], f"census: {source_id}.selection_mode")
    _require(
        mode in {"SEEDED_HASH", "EXACT_SCORE_DESC_SHA_ASC"},
        f"census: {source_id}.selection_mode is invalid",
    )
    target = _exact_int(
        obj["target_serialized_tokens"],
        f"census: {source_id}.target_serialized_tokens",
        minimum=1,
    )

    if graph_node is not None:
        _require(source_id == graph_node.source_id, "census: node order disagrees with graph")
        _require(stage == graph_node.stage, f"census: {source_id} stage disagrees with graph")
        _require(
            target == graph_node.target_serialized_tokens,
            f"census: {source_id} target disagrees with the frozen graph",
        )
        if graph_node.selection_mode == BRANCH_DEPENDENT:
            _require(
                branch in {"PRIMARY_GE4", "FALLBACK_RANKED_GE3"},
                f"census: {source_id} must report the selected graph branch",
            )
            branch_spec = (
                graph_node.branch_primary if branch == "PRIMARY_GE4" else graph_node.branch_fallback
            )
            _require(
                mode == branch_spec["selection_mode"],
                f"census: {source_id} mode disagrees with its selected graph branch",
            )
        else:
            _require(branch == "ORDINARY", f"census: {source_id} has an invalid ordinary branch")
            _require(
                mode == graph_node.selection_mode,
                f"census: {source_id} mode disagrees with graph",
            )

    integer_fields = (
        "pre_exclusion_unique_identities",
        "g2_excluded_identities",
        "prior_commit_excluded_identities",
        "post_exclusion_candidate_identities",
        "post_exclusion_candidate_serialized_tokens",
        "selected_identities",
        "selected_serialized_tokens",
        "actual_overshoot_tokens",
        "residual_identities",
        "residual_serialized_tokens",
    )
    for key in integer_fields:
        _exact_int(obj[key], f"census: {source_id}.{key}")

    _hex_string(obj["selection_fingerprint"], 64, f"census: {source_id}.selection_fingerprint")
    crossing_identity = _optional_hex(
        obj["crossing_identity"], 64, f"census: {source_id}.crossing_identity"
    )
    crossing_tokens = obj["crossing_document_serialized_tokens"]
    if crossing_tokens is not None:
        _exact_int(
            crossing_tokens,
            f"census: {source_id}.crossing_document_serialized_tokens",
            minimum=1,
        )
    _require(type(obj["feasible"]) is bool, f"census: {source_id}.feasible must be a boolean")

    exclusions = obj["exclusions_by_owner"]
    _require(type(exclusions) is dict, f"census: {source_id}.exclusions_by_owner must be an object")
    _require(
        all(type(owner) is str and owner for owner in exclusions),
        f"census: {source_id}.exclusions_by_owner keys must be non-empty strings",
    )
    for owner, count in exclusions.items():
        _exact_int(count, f"census: {source_id}.exclusions_by_owner.{owner}", minimum=1)
        _require(
            owner in earlier_sources,
            f"census: {source_id} claims an exclusion owned by a non-earlier node {owner!r}",
        )

    _require(
        obj["prior_commit_excluded_identities"] == sum(exclusions.values()),
        f"census: {source_id} ownership exclusions do not sum to the reported total",
    )
    _require(
        obj["pre_exclusion_unique_identities"]
        == obj["g2_excluded_identities"]
        + obj["prior_commit_excluded_identities"]
        + obj["post_exclusion_candidate_identities"],
        f"census: {source_id} candidate accounting does not close",
    )
    _require(
        obj["selected_identities"] <= obj["post_exclusion_candidate_identities"],
        f"census: {source_id} selected identities exceed candidates",
    )
    _require(
        obj["selected_serialized_tokens"] <= obj["post_exclusion_candidate_serialized_tokens"],
        f"census: {source_id} selected tokens exceed candidates",
    )
    for count_key, mass_key in (
        ("post_exclusion_candidate_identities", "post_exclusion_candidate_serialized_tokens"),
        ("selected_identities", "selected_serialized_tokens"),
        ("residual_identities", "residual_serialized_tokens"),
    ):
        _require(
            (obj[count_key] == 0) == (obj[mass_key] == 0),
            f"census: {source_id} {count_key} and {mass_key} disagree on empty population",
        )
    _require(
        obj["residual_identities"]
        == obj["post_exclusion_candidate_identities"] - obj["selected_identities"],
        f"census: {source_id} residual identities do not close",
    )
    _require(
        obj["residual_serialized_tokens"]
        == obj["post_exclusion_candidate_serialized_tokens"] - obj["selected_serialized_tokens"],
        f"census: {source_id} residual tokens do not close",
    )
    feasible = obj["feasible"]
    _require(
        feasible == (obj["selected_serialized_tokens"] >= target),
        f"census: {source_id} feasibility disagrees with its selected mass",
    )
    expected_overshoot = obj["selected_serialized_tokens"] - target if feasible else 0
    _require(
        obj["actual_overshoot_tokens"] == expected_overshoot,
        f"census: {source_id} overshoot is not the measured value",
    )
    if feasible:
        _require(
            crossing_identity is not None and crossing_tokens is not None,
            f"census: {source_id} is feasible but reports no crossing document",
        )
        _require(
            crossing_tokens <= obj["selected_serialized_tokens"],
            f"census: {source_id} crossing mass is outside its selected mass",
        )
        _require(
            obj["actual_overshoot_tokens"] < crossing_tokens,
            f"census: {source_id} overshoot is not bounded by the whole crossing document",
        )
    else:
        _require(
            crossing_identity is None and crossing_tokens is None,
            f"census: {source_id} is infeasible but reports a crossing document",
        )
        _require(
            obj["selected_identities"] == obj["post_exclusion_candidate_identities"]
            and obj["selected_serialized_tokens"]
            == obj["post_exclusion_candidate_serialized_tokens"],
            f"census: {source_id} infeasible selection did not exhaust its candidates",
        )

    evidence = _closed_object(
        obj["boundary_evidence"],
        frozenset(BOUNDARY_EVIDENCE_FIELDS),
        f"census: {source_id}.boundary_evidence",
    )
    _require(
        _exact_str(
            evidence["representative_rule"],
            f"census: {source_id}.boundary_evidence.representative_rule",
        )
        == REPRESENTATIVE_RULE,
        f"census: {source_id} boundary representative rule is invalid",
    )
    crossing_rank = _optional_hex(
        evidence["crossing_selection_rank"],
        64,
        f"census: {source_id}.boundary_evidence.crossing_selection_rank",
    )
    if not feasible:
        _require(crossing_rank is None, f"census: {source_id} has crossing rank without crossing")
    elif mode == "SEEDED_HASH":
        _require(crossing_rank is not None, f"census: {source_id} is missing crossing rank")
    else:
        _require(crossing_rank is None, f"census: {source_id} exact-score crossing has hash rank")
    _validate_score_evidence(
        evidence["crossing_score_bits_hex"],
        evidence["crossing_score_hex"],
        f"census: {source_id}.boundary_evidence.crossing",
        required=feasible and mode == "EXACT_SCORE_DESC_SHA_ASC",
    )
    if not feasible:
        _require(
            evidence["crossing_score_bits_hex"] is None and evidence["crossing_score_hex"] is None,
            f"census: {source_id} has crossing score without crossing",
        )

    next_identity = _optional_hex(
        evidence["next_unselected_identity"],
        64,
        f"census: {source_id}.boundary_evidence.next_unselected_identity",
    )
    next_tokens = evidence["next_unselected_serialized_tokens"]
    next_rank = _optional_hex(
        evidence["next_unselected_selection_rank"],
        64,
        f"census: {source_id}.boundary_evidence.next_unselected_selection_rank",
    )
    has_next = obj["residual_identities"] > 0
    if has_next:
        _require(next_identity is not None, f"census: {source_id} is missing next identity")
        _exact_int(
            next_tokens,
            f"census: {source_id}.boundary_evidence.next_unselected_serialized_tokens",
            minimum=1,
        )
        _require(
            next_tokens <= obj["residual_serialized_tokens"],
            f"census: {source_id} next document exceeds residual mass",
        )
        if mode == "SEEDED_HASH":
            _require(next_rank is not None, f"census: {source_id} is missing next hash rank")
        else:
            _require(next_rank is None, f"census: {source_id} exact-score next item has hash rank")
    else:
        _require(
            next_identity is None and next_tokens is None and next_rank is None,
            f"census: {source_id} has next-unselected evidence without residual candidates",
        )
    _validate_score_evidence(
        evidence["next_unselected_score_bits_hex"],
        evidence["next_unselected_score_hex"],
        f"census: {source_id}.boundary_evidence.next_unselected",
        required=has_next and mode == "EXACT_SCORE_DESC_SHA_ASC",
    )
    if not has_next:
        _require(
            evidence["next_unselected_score_bits_hex"] is None
            and evidence["next_unselected_score_hex"] is None,
            f"census: {source_id} has next score without a residual candidate",
        )


def _validate_complete_contract(census: Any, context: AuthorizedRunContext | None = None) -> None:
    if context is not None:
        _require_authorized_context(context, "census validation")
    obj = _closed_object(census, _CENSUS_FIELDS, "census")
    _require(
        _exact_str(obj["schema_version"], "census.schema_version") == CENSUS_SCHEMA,
        "census: wrong schema_version",
    )
    _require(_exact_str(obj["status"], "census.status") == "COMPLETE", "census: wrong status")
    _require(
        _exact_str(obj["output_label"], "census.output_label") == OUTPUT_LABEL,
        "census: wrong output_label",
    )
    _require(obj["resume_supported"] is False, "census: resume must be disabled")
    graph_sha = _hex_string(obj["graph_sha256"], 64, "census.graph_sha256")
    _exact_int(obj["seed"], "census.seed")
    _exact_int(
        obj["reference_exclusion_identities"],
        "census.reference_exclusion_identities",
    )
    _require(
        type(obj["hard_stop_required"]) is bool,
        "census: hard_stop_required must be a boolean",
    )

    authorization = _closed_object(
        obj["authorization"], _AUTHORIZATION_FIELDS, "census.authorization"
    )
    for key in (
        "run_identity",
        "candidate_plan_sha256",
        "owner_graph_sha256",
        "implementation_bundle_sha256",
    ):
        _hex_string(authorization[key], 64, f"census.authorization.{key}")
    _require(
        graph_sha == authorization["owner_graph_sha256"],
        "census: graph SHA disagrees with authorization",
    )
    files = _closed_object(
        authorization["implementation_files"],
        frozenset(IMPLEMENTATION_BUNDLE_FILES),
        "census.authorization.implementation_files",
    )
    for relative, digest in files.items():
        _hex_string(digest, 64, f"census.authorization.implementation_files.{relative}")
    _require(
        implementation_bundle_sha256(files) == authorization["implementation_bundle_sha256"],
        "census: implementation bundle digest is internally inconsistent",
    )
    authority_sha = _closed_object(
        authorization["authority_sha256"],
        frozenset(AUTHORITY_KEYS),
        "census.authorization.authority_sha256",
    )
    for name, digest in authority_sha.items():
        _hex_string(digest, 64, f"census.authorization.authority_sha256.{name}")
    _require(
        _exact_str(
            authorization["census_schema_version"],
            "census.authorization.census_schema_version",
        )
        == CENSUS_SCHEMA,
        "census: authorization census schema mismatch",
    )
    _require(
        _exact_str(
            authorization["node_result_schema_version"],
            "census.authorization.node_result_schema_version",
        )
        == NODE_RESULT_SCHEMA,
        "census: authorization node schema mismatch",
    )
    _require(
        _exact_str(
            authorization["plan_schema_version"],
            "census.authorization.plan_schema_version",
        )
        == PLAN_SCHEMA,
        "census: authorization plan schema mismatch",
    )
    _require(
        _exact_str(
            authorization["authorization_status"],
            "census.authorization.authorization_status",
        )
        == "OWNER_SUPPLIED_EXPECTED_PLAN_SHA256",
        "census: invalid authorization status",
    )
    expected_identity = run_identity(
        plan_sha256=authorization["candidate_plan_sha256"],
        graph_sha256=authorization["owner_graph_sha256"],
        bundle_sha256=authorization["implementation_bundle_sha256"],
        census_schema=authorization["census_schema_version"],
    )
    _require(
        authorization["run_identity"] == expected_identity,
        "census: authoritative run identity is internally inconsistent",
    )

    bound = _closed_object(
        obj["bound_authorities"],
        frozenset(BOUND_AUTHORITY_KEYS),
        "census.bound_authorities",
    )
    for key, digest in bound.items():
        _hex_string(digest, 64, f"census.bound_authorities.{key}")
    for name, bound_key in AUTHORITY_KEYS.items():
        _require(
            bound[bound_key] == authority_sha[name],
            f"census: bound authority {bound_key} disagrees with authorization",
        )

    counters = obj["binding_counters"]
    _require(
        type(counters) is dict and counters, "census: binding_counters must be a non-empty object"
    )
    _require(
        all(type(binding_id) is str and binding_id for binding_id in counters),
        "census: binding counter keys must be non-empty strings",
    )
    for binding_id, stats_value in counters.items():
        stats = _closed_object(
            stats_value, _COUNTER_FIELDS, f"census.binding_counters.{binding_id}"
        )
        for key in _COUNTER_FIELDS:
            _exact_int(stats[key], f"census.binding_counters.{binding_id}.{key}")
        accounted = (
            stats["d2_d3_excluded_rows"]
            + stats["empty_or_non_string_text"]
            + stats["empty_after_cleaning"]
            + stats["eligible_rows"]
        )
        _require(
            accounted == stats["physical_rows"],
            f"census: {binding_id} row dispositions do not account for every physical row",
        )

    nodes = obj["nodes"]
    _require(type(nodes) is list and nodes, "census: nodes must be a non-empty array")
    if context is not None:
        _require(graph_sha == context.graph_sha256, "census: graph SHA disagreement")
        _require(obj["seed"] == context.graph.seed, "census: seed disagreement")
        _require(
            authorization == context.as_canonical(),
            "census: authorization block does not match the authorised run context",
        )
        _require(
            bound == context.graph.bound_authorities,
            "census: bound authorities disagree with the authorised graph",
        )
        _require(
            set(counters) == set(context.graph.bindings),
            "census: binding counters do not cover exactly the graph's bindings",
        )
        _require(
            len(nodes) == len(context.graph.nodes),
            "census: node results do not cover the graph's nodes",
        )
        graph_nodes: list[Any | None] = list(context.graph.nodes)
    else:
        graph_nodes = [None] * len(nodes)

    earlier_sources: set[str] = set()
    for index, (node, graph_node) in enumerate(zip(nodes, graph_nodes, strict=True)):
        source = node.get("source_id") if type(node) is dict else None
        _require(
            type(source) is str and source not in earlier_sources,
            f"census: node {index} source_id must be a unique string",
        )
        _validate_contract_node(
            node,
            graph_node=graph_node,
            earlier_sources=earlier_sources,
        )
        earlier_sources.add(source)
    execution_order = [(STAGE_PRIORITY[node["stage"]], node["source_id"]) for node in nodes]
    _require(
        execution_order == sorted(execution_order),
        "census: nodes are not in canonical execution order",
    )

    if context is not None:
        for binding_id, stats in counters.items():
            binding = context.graph.bindings[binding_id]
            _require(
                stats["physical_rows"] == binding.total_physical_rows,
                f"census: {binding_id} physical row count is not final",
            )
            _require(
                stats["d2_d3_excluded_rows"] == binding.excluded_rows,
                f"census: {binding_id} eligibility exclusions are not fully consumed",
            )

    ownership = obj["ownership_matrix"]
    _require(type(ownership) is dict, "census: ownership_matrix must be an object")
    _require(
        all(type(source) is str and source for source in ownership),
        "census: ownership_matrix keys must be non-empty strings",
    )
    for source, owners in ownership.items():
        _require(
            type(owners) is dict,
            f"census: ownership_matrix.{source} must be an object",
        )
        _require(
            all(type(owner) is str and owner for owner in owners),
            f"census: ownership_matrix.{source} keys must be non-empty strings",
        )
        for owner, count in owners.items():
            _exact_int(count, f"census: ownership_matrix.{source}.{owner}", minimum=1)
    expected_ownership = {
        node["source_id"]: node["exclusions_by_owner"]
        for node in nodes
        if node["exclusions_by_owner"]
    }
    _require(
        ownership == expected_ownership,
        "census: ownership_matrix disagrees with node ownership exclusions",
    )

    totals = _closed_object(obj["totals"], _TOTAL_FIELDS, "census.totals")
    for key in _TOTAL_FIELDS:
        _exact_int(totals[key], f"census.totals.{key}")
    _require(
        totals["physical_rows"] == sum(stats["physical_rows"] for stats in counters.values()),
        "census: totals.physical_rows disagrees with the per-binding counters",
    )
    _require(
        totals["eligible_rows"] == sum(stats["eligible_rows"] for stats in counters.values()),
        "census: totals.eligible_rows disagrees with the per-binding counters",
    )
    _require(
        totals["selected_identities"] == sum(node["selected_identities"] for node in nodes),
        "census: totals.selected_identities disagrees with the node results",
    )
    _require(
        totals["selected_serialized_tokens"]
        == sum(node["selected_serialized_tokens"] for node in nodes),
        "census: totals.selected_serialized_tokens disagrees with the node results",
    )
    _require(
        obj["hard_stop_required"] == any(not node["feasible"] for node in nodes),
        "census: hard_stop_required disagrees with the node feasibility states",
    )

    try:
        encoded = canonical_json_bytes(obj)
    except (TypeError, ValueError) as exc:
        raise CensusError(f"census is not canonically serialisable: {exc}") from exc
    _require(
        canonical_json_bytes(json.loads(encoded.decode("utf-8"))) == encoded,
        "census: canonical serialisation is not a fixed point",
    )


def validate_complete_census(census: Any, context: AuthorizedRunContext | None = None) -> None:
    """Everything that must hold before the word COMPLETE may be written anywhere.

    A result that is incomplete, malformed, mis-schemaed or internally inconsistent fails here,
    before any staging file exists, so no partial artefact can be labelled complete.
    """
    _validate_complete_contract(census, context)


def _complete_marker(census: dict[str, Any], census_sha256: str) -> dict[str, Any]:
    authorization = census["authorization"]
    return {
        "marker": COMPLETE_MARKER,
        "run_identity": authorization["run_identity"],
        "candidate_plan_sha256": authorization["candidate_plan_sha256"],
        "owner_graph_sha256": authorization["owner_graph_sha256"],
        "implementation_bundle_sha256": authorization["implementation_bundle_sha256"],
        "census_sha256": census_sha256,
        "census_schema_version": CENSUS_SCHEMA,
    }


def _validate_complete_marker(marker: Any, census: dict[str, Any], census_sha256: str) -> None:
    obj = _closed_object(marker, _MARKER_FIELDS, "COMPLETE marker")
    _require(
        _exact_str(obj["marker"], "COMPLETE marker.marker") == COMPLETE_MARKER,
        "COMPLETE marker: wrong marker literal",
    )
    for key in (
        "run_identity",
        "candidate_plan_sha256",
        "owner_graph_sha256",
        "implementation_bundle_sha256",
        "census_sha256",
    ):
        _hex_string(obj[key], 64, f"COMPLETE marker.{key}")
    _require(
        _exact_str(
            obj["census_schema_version"],
            "COMPLETE marker.census_schema_version",
        )
        == CENSUS_SCHEMA,
        "COMPLETE marker: census schema mismatch",
    )
    _require(
        obj == _complete_marker(census, census_sha256),
        "COMPLETE marker and census disagree",
    )


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
    _require_authorized_context(context, "publication")

    # Finalize every hash-bearing byte before the last identity check. Nothing has been written
    # yet, so a revalidation failure cannot leave observable output.
    validate_complete_census(census, context)
    payload = canonical_json_bytes(census)
    marker = canonical_json_bytes(_complete_marker(census, hashlib.sha256(payload).hexdigest()))
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
    if marker.get("census_sha256") != census_digest:
        raise CensusError(f"{final}: COMPLETE marker does not describe the published census")
    census = strict_json_object(payload, where=f"{final}/{CENSUS_FILENAME}")
    _require(
        canonical_json_bytes(census) == payload,
        f"{final}: census bytes are not canonical",
    )
    _require(
        canonical_json_bytes(marker) == marker_bytes,
        f"{final}: COMPLETE marker bytes are not canonical",
    )
    validate_complete_census(census)
    _validate_complete_marker(marker, census, census_digest)
    authorization = census["authorization"]
    if final.name != f"run-{authorization['run_identity'][:32]}":
        raise CensusError(f"{final}: directory name does not match its own run identity")
    if expected_run_identity is not None:
        _hex_string(expected_run_identity, 64, "expected_run_identity")
        if authorization["run_identity"] != expected_run_identity:
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
    if (
        type(implementation_commit) is not str
        or len(implementation_commit) != 40
        or set(implementation_commit) - _HEX
    ):
        raise CensusError("implementation_commit must be a lowercase 40-character Git SHA")

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
    commit = result.stdout.strip()
    if len(commit) != 40 or set(commit) - _HEX:
        raise CensusError(
            f"runtime repository HEAD is not a lowercase 40-character Git SHA: {commit!r}"
        )
    return commit


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
