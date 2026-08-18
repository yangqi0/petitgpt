#!/usr/bin/env python3

"""Production candidate builder for the seven frozen Non-Python Gate C sources.

This is the production-scale sibling of the accepted bounded Parquet C0 diagnostic
(``pretrain/corpus_gate_c_parquet.py``).  That module keeps its frozen semantics untouched: its
``SAMPLER_VERSION``, its ``release_kind="c0_diagnostic"``, its structural ceilings and its
already-published immutable releases are not modified, lifted or reused here.

What this module adds is a *complete sequential traversal* of a deterministic, exactly-described
scope of the pinned Parquet objects, with every row in that scope evaluated by the frozen
source-local Gate C filter.  There is no sampler: the C0 PPS/Feistel plan is deliberately not
imported, because a production candidate population must not inherit a diagnostic sampling claim.

Two honest scope modes, both recorded in every manifest:

    full_source   — every pinned Parquet file of the source, traversed completely.
                    full_source_traversal = true
    file_prefix   — the first K pinned files in ascending path order, each traversed completely.
                    full_source_traversal = false
                    candidate_scope names K and the exact file list hash.

A ``file_prefix`` release is a complete population *of its scope* and is never described as the
complete source population, and never as a row-level representative sample of it.

Out of scope here, and asserted as such in every manifest: cross-source near-dedup, benchmark
decontamination, reference-reserve exclusion, canonical-token counting, final source quotas and
the Stage A / Stage B split.  Those are later stages that consume this output.

Resource accounting follows the same contract as the Python full builder: quantities this
transport cannot measure are published as ``null`` with an explicit ``*_measured = false``, never
as zero.
"""

from __future__ import annotations

import argparse
import array
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pretrain.corpus_gate_c import (  # noqa: E402
    CHECKPOINT_NAME,
    CHECKSUMS_NAME,
    DOCUMENTS_NAME,
    MANIFEST_NAME,
    SOURCES,
    GateCError,
    PlannedInterruption,
    _atomic_replace_bytes,
    _canonical_json_bytes,
    _ensure_git_ignored,
    _ensure_revision,
    _ensure_under_workspace,
    _fsync_directory,
    _natural_id,
    _publish_directory,
    _require_nonnegative_int,
    _sha256_bytes,
    _source_record_id,
    _staging_path,
    _write_new_file,
    canonical_jsonl_record_bytes,
    evaluate_row,
)
from pretrain.corpus_gate_c_parquet import (  # noqa: E402
    PARQUET_BINDINGS,
    READ_BLOCK_BYTES,
    SOURCE_COLUMNS,
    GateCParquetError,
    HubObjectStore,
    ObjectStore,
    ParquetBinding,
    _parquet_schema_hash,
)

__all__ = [
    "CandidateScope",
    "GateCProductionError",
    "HubObjectStore",
    "PlannedInterruption",
    "ProductionConfig",
    "ResourceAccounting",
    "build_production_candidates",
    "build_scoped_transport_manifest",
    "diagnose_release",
    "main",
    "verify_release",
]

PRODUCTION_TOOL_SCHEMA_VERSION = "petitgpt-corpus-gate-c-nonpython-production-v1"
PRODUCTION_SPEC_VERSION = "nonpython-gate-c-production-candidate-spec-2026-08-18"
RELEASE_KIND = "nonpython_gate_c_production_candidate"

TRANSPORT_MANIFEST_NAME = "transport_manifest.json"
DEFAULT_CHECKPOINT_EVERY = 50_000
DEFAULT_BATCH_ROWS = 512
MAX_WALL_SECONDS = 24 * 60 * 60

FULL_SOURCE = "full_source"
FILE_PREFIX = "file_prefix"


class GateCProductionError(GateCParquetError):
    """A fail-closed Non-Python production candidate contract error."""


def filter_contract(source_key: str) -> dict[str, Any]:
    """The frozen source-local filter this builder inherits verbatim from Gate C."""
    spec = SOURCES[source_key]
    return {
        "source_key": spec.key,
        "body_path": spec.body_path,
        "min_bytes": spec.min_bytes,
        "max_bytes": spec.max_bytes,
        "required_schema": spec.required_schema_map,
        "metadata_paths": list(spec.metadata_paths),
        "natural_id_path": spec.natural_id_path,
        "evaluator": "pretrain.corpus_gate_c.evaluate_row",
        "inherited_from": "pretrain/corpus_gate_c.py",
    }


def filter_contract_sha256(source_key: str) -> str:
    return _sha256_bytes(_canonical_json_bytes(filter_contract(source_key)))


# --------------------------------------------------------------------------------------
# Scope
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateScope:
    """An exactly-described deterministic build scope."""

    mode: str
    file_count: int
    total_file_count: int
    file_list_sha256: str

    def __post_init__(self) -> None:
        if self.mode not in {FULL_SOURCE, FILE_PREFIX}:
            raise GateCProductionError(f"unknown candidate scope mode: {self.mode}")
        if not 1 <= self.file_count <= self.total_file_count:
            raise GateCProductionError("scope file count is out of range")
        if self.mode == FULL_SOURCE and self.file_count != self.total_file_count:
            raise GateCProductionError("full_source scope must cover every pinned file")

    @property
    def full_source_traversal(self) -> bool:
        return self.mode == FULL_SOURCE

    def to_json(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "files_in_scope": self.file_count,
            "pinned_files_total": self.total_file_count,
            "file_list_sha256": self.file_list_sha256,
            "description": (
                "every pinned Parquet file of the source, each traversed completely"
                if self.full_source_traversal
                else (
                    f"the first {self.file_count} of {self.total_file_count} pinned Parquet files "
                    "in ascending path order, each traversed completely"
                )
            ),
            "row_level_sampling": False,
            "representativeness_claim": (
                "complete population of the pinned source"
                if self.full_source_traversal
                else (
                    "complete population of the named file prefix only; this is NOT the complete "
                    "source population and NOT a row-level representative sample of it"
                )
            ),
        }


# --------------------------------------------------------------------------------------
# Honest resource accounting
# --------------------------------------------------------------------------------------


@dataclass
class ResourceAccounting:
    """Exactly measured I/O quantities plus an explicit record of what is not measured."""

    parquet_files_opened: int = 0
    parquet_footers_read: int = 0
    row_groups_read: int = 0
    objects_downloaded: int = 0
    object_cache_reuses: int = 0
    object_integrity_verifications: int = 0
    downloaded_bytes: int = 0
    resume_reread_row_groups: int = 0

    MEASURED_FIELDS = (
        "parquet_files_opened",
        "parquet_footers_read",
        "row_groups_read",
        "objects_downloaded",
        "object_cache_reuses",
        "object_integrity_verifications",
        "downloaded_bytes",
        "resume_reread_row_groups",
    )
    UNMEASURED_FIELDS = ("network_request_count",)

    def measured(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.MEASURED_FIELDS}

    def to_json(self) -> dict[str, Any]:
        return {
            "measured": self.measured(),
            "measured_fields": list(self.MEASURED_FIELDS),
            "unmeasured": {name: None for name in self.UNMEASURED_FIELDS},
            "unmeasured_fields": list(self.UNMEASURED_FIELDS),
            "network_request_count": None,
            "network_request_count_measured": False,
            "network_request_count_note": (
                "The huggingface_hub transport does not expose an HTTP request count. An unknown "
                "quantity is published as null, never as zero and never approximated by counting "
                "one request per file or row group."
            ),
            "downloaded_bytes_note": (
                "Bytes actually transferred for pinned objects. Each in-scope object is fetched "
                "once as a complete file, verified against its pinned size and LFS SHA-256, read "
                "from local disk, and then released."
            ),
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> ResourceAccounting:
        measured = value.get("measured") or {}
        if not isinstance(measured, Mapping):
            raise GateCProductionError("checkpoint resource accounting is invalid")
        return cls(**{
            name: _require_nonnegative_int(measured.get(name, 0), name)
            for name in cls.MEASURED_FIELDS
        })


@dataclass
class ProductionCounters:
    """Traversal accounting for one production candidate build."""

    scanned: int = 0
    accepted: int = 0
    rejected: int = 0
    accepted_text_bytes: int = 0
    rejections: Counter = field(default_factory=Counter)
    diagnostics: Counter = field(default_factory=Counter)

    def to_json(self) -> dict[str, Any]:
        return {
            "scanned": self.scanned,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "accepted_text_bytes": self.accepted_text_bytes,
            "rejections": dict(sorted(self.rejections.items())),
            "diagnostics": dict(sorted(self.diagnostics.items())),
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> ProductionCounters:
        maps = {}
        for name in ("rejections", "diagnostics"):
            entry = value.get(name) or {}
            if not isinstance(entry, Mapping):
                raise GateCProductionError("checkpoint counter maps are invalid")
            maps[name] = Counter(dict(entry))
        return cls(
            scanned=_require_nonnegative_int(value.get("scanned"), "scanned"),
            accepted=_require_nonnegative_int(value.get("accepted"), "accepted"),
            rejected=_require_nonnegative_int(value.get("rejected"), "rejected"),
            accepted_text_bytes=_require_nonnegative_int(
                value.get("accepted_text_bytes"), "accepted_text_bytes"
            ),
            rejections=maps["rejections"],
            diagnostics=maps["diagnostics"],
        )


@dataclass(frozen=True)
class ProductionConfig:
    """Production candidate build parameters.

    There is no document cap and no scan cap: completeness *of the declared scope* is the point.
    ``max_wall_seconds`` is operational only — a time-capped run leaves a resumable checkpoint and
    publishes nothing.
    """

    source_key: str
    output_dir: Path
    work_dir: Path
    cache_dir: Path | None = None
    scope_files: int | None = None
    keep_object_cache: bool = False
    max_wall_seconds: float = float(MAX_WALL_SECONDS)
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY
    batch_rows: int = DEFAULT_BATCH_ROWS
    stop_after_documents: int | None = None


# --------------------------------------------------------------------------------------
# Scoped transport manifest
# --------------------------------------------------------------------------------------


def build_scoped_transport_manifest(
    source_key: str,
    store: ObjectStore,
    *,
    scope_files: int | None,
    resources: ResourceAccounting | None = None,
) -> dict[str, Any]:
    """Pin the complete object list, and read footers for exactly the files in scope.

    The full object list is always enumerated so the manifest can state honestly how large the
    pinned source is and what fraction the scope covers.  Footers are read only for in-scope
    files, because reading thousands of out-of-scope footers buys nothing.
    """
    if source_key not in PARQUET_BINDINGS or source_key not in SOURCES:
        raise GateCProductionError(f"unknown source: {source_key}")
    binding = PARQUET_BINDINGS[source_key]
    spec = SOURCES[source_key]
    _ensure_revision(spec.revision)
    _ensure_revision(binding.parquet_revision)

    objects = store.list_objects(binding)
    total = len(objects)
    if total == 0:
        raise GateCProductionError(f"no pinned Parquet objects for {source_key}")
    count = total if scope_files is None else min(scope_files, total)
    if count < 1:
        raise GateCProductionError("scope must contain at least one pinned file")
    mode = FULL_SOURCE if count == total else FILE_PREFIX
    in_scope = objects[:count]

    from pretrain.corpus_gate_c_parquet import read_footer

    entries = []
    schema_hashes: set[str] = set()
    for obj in in_scope:
        entry = read_footer(store, binding, obj)
        if resources is not None:
            resources.parquet_footers_read += 1
        schema_hashes.add(str(entry.schema_hash))
        entries.append(entry)
    if len(schema_hashes) != 1:
        raise GateCProductionError(f"Parquet schema differs between files for {source_key}")

    file_list_sha = _sha256_bytes(
        _canonical_json_bytes([{"path": o.path, "size": o.size, "oid": o.oid} for o in in_scope])
    )
    scope = CandidateScope(
        mode=mode, file_count=count, total_file_count=total, file_list_sha256=file_list_sha
    )
    oids_are_sha256 = all(_is_sha256(obj.oid) for obj in in_scope)
    manifest = {
        "tool_schema_version": PRODUCTION_TOOL_SCHEMA_VERSION,
        "spec_version": PRODUCTION_SPEC_VERSION,
        "source_key": source_key,
        "dataset": spec.dataset,
        "config": spec.dataset_config,
        "split": spec.split,
        "data_revision": spec.revision,
        "parquet_repo_id": binding.repo_id,
        "parquet_revision": binding.parquet_revision,
        "path_prefix": binding.path_prefix,
        "transport": "huggingface_hub_parquet_sequential",
        "pinned_file_count": total,
        "pinned_total_bytes": sum(obj.size for obj in objects),
        "scope": scope.to_json(),
        "scope_file_count": count,
        "scope_bytes": sum(obj.size for obj in in_scope),
        "scope_rows": sum(entry.rows or 0 for entry in entries),
        "scope_row_group_count": sum(len(entry.row_group_rows or ()) for entry in entries),
        "schema_hash": next(iter(schema_hashes)),
        "footers_read_for_scope": True,
        "files": [entry.to_json() for entry in entries],
        "all_pinned_paths_sha256": _sha256_bytes(
            _canonical_json_bytes([obj.path for obj in objects])
        ),
        # Building this manifest is a *preflight* step: it lists the pinned objects and reads
        # their Parquet footers, so it alone computes no full-file digest.  Production traversal
        # is a different thing and is described separately, because claiming here that nothing is
        # ever downloaded or hashed in full would contradict what the build actually does.
        "preflight_full_file_sha256_computed": False,
        "production_traversal_object_integrity": {
            "objects_materialized_completely_before_row_evaluation": True,
            "expected_size_verified": True,
            "local_full_file_sha256_computed": True,
            "pinned_oids_are_sha256": oids_are_sha256,
            "local_sha256_compared_with_pinned_oid": oids_are_sha256,
            "note": (
                "Production traversal downloads each in-scope object in full, verifies its byte "
                "size against the pinned record and computes its local SHA-256 before any row of "
                "that object is evaluated; the digest is additionally compared with the pinned "
                "LFS oid whenever that oid is itself a SHA-256. Any mismatch aborts the build."
            ),
        },
        "identity_note": (
            "Pinned identity is the Hub LFS OID plus blob etag at the pinned commit. Constructing "
            "this transport manifest is a preflight step that reads only Parquet footers, so the "
            "preflight itself claims no local full-file SHA-256. Production traversal does "
            "materialize and hash every in-scope object completely before evaluating any of its "
            "rows; see production_traversal_object_integrity."
        ),
    }
    manifest["manifest_sha256"] = _sha256_bytes(_canonical_json_bytes(manifest))
    return manifest


def scoped_transport_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return _sha256_bytes(_canonical_json_bytes(body))


def verify_scoped_transport_manifest(manifest: Mapping[str, Any]) -> str:
    recorded = manifest.get("manifest_sha256")
    actual = scoped_transport_manifest_sha256(manifest)
    if recorded != actual:
        raise GateCProductionError("transport manifest checksum mismatch")
    return actual


def scope_from_manifest(manifest: Mapping[str, Any]) -> CandidateScope:
    scope = manifest["scope"]
    return CandidateScope(
        mode=str(scope["mode"]),
        file_count=int(scope["files_in_scope"]),
        total_file_count=int(scope["pinned_files_total"]),
        file_list_sha256=str(scope["file_list_sha256"]),
    )


# --------------------------------------------------------------------------------------
# Sequential row-group traversal
# --------------------------------------------------------------------------------------


def fetch_pinned_object(
    store: ObjectStore,
    binding: ParquetBinding,
    file_path: str,
    destination: Path,
    *,
    expected_size: int,
    expected_oid: str,
    resources: ResourceAccounting,
    max_attempts: int = 4,
) -> Path:
    """Materialize one complete pinned Parquet object locally and verify its identity.

    Reading Parquet straight off the Hub filesystem issues a fresh HTTP request per read block.
    At production scale that is tens of thousands of connections per source, which exhausts the
    pod's ephemeral ports and DNS (observed as ``[Errno 99] Cannot assign requested address`` and
    ``[Errno -3] Temporary failure in name resolution``) long before a file finishes.  One bulk
    transfer per file, verified and then read from local disk, is both far cheaper and the only
    shape that completes.  Files are released by the caller once traversed, so the cache holds one
    object at a time rather than the whole source.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == expected_size:
        digest, size = _sha256_file(destination)
        resources.object_integrity_verifications += 1
        if size == expected_size and (not _is_sha256(expected_oid) or digest == expected_oid):
            resources.object_cache_reuses += 1
            return destination
        destination.unlink()

    partial = destination.parent / f".{destination.name}.partial"
    last: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            written = 0
            with store.open(binding, file_path) as reader, open(partial, "wb") as local:
                while True:
                    block = reader.read(READ_BLOCK_BYTES)
                    if not block:
                        break
                    local.write(block)
                    written += len(block)
                local.flush()
                os.fsync(local.fileno())
            if written != expected_size:
                raise GateCProductionError(
                    f"pinned object transfer is short: {file_path} "
                    f"({written} of {expected_size} bytes)"
                )
            os.replace(partial, destination)
            _fsync_directory(destination.parent)
            resources.objects_downloaded += 1
            resources.downloaded_bytes += written
            break
        except GateCError:
            partial.unlink(missing_ok=True)
            raise
        except Exception as exc:  # noqa: BLE001 - bounded retry of the same pinned object
            last = exc
            partial.unlink(missing_ok=True)
            if attempt + 1 >= max_attempts:
                raise GateCProductionError(
                    f"pinned Parquet object unavailable after bounded retries: {file_path}"
                ) from last
            time.sleep(float(4 * (attempt + 1)))

    digest, size = _sha256_file(destination)
    resources.object_integrity_verifications += 1
    if size != expected_size:
        raise GateCProductionError(f"pinned object size mismatch after fetch: {file_path}")
    if _is_sha256(expected_oid) and digest != expected_oid:
        raise GateCProductionError(
            f"pinned object SHA-256 does not match the pinned LFS oid: {file_path}"
        )
    return destination


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with open(path, "rb") as handle:
        while True:
            block = handle.read(READ_BLOCK_BYTES)
            if not block:
                break
            digest.update(block)
            total += len(block)
    return digest.hexdigest(), total


@contextmanager
def open_local_parquet(
    local_path: Path,
    file_path: str,
    *,
    expected_schema_hash: str,
    expected_row_group_rows: Sequence[int],
) -> Iterator[Any]:
    """Open a locally materialized pinned object and validate its pinned topology."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(local_path, pre_buffer=False, buffer_size=READ_BLOCK_BYTES)
    metadata = parquet.metadata
    if metadata.num_row_groups != len(expected_row_group_rows):
        raise GateCProductionError(
            f"pinned row-group count drifted: {file_path} "
            f"({metadata.num_row_groups} != {len(expected_row_group_rows)})"
        )
    for index, rows in enumerate(expected_row_group_rows):
        if metadata.row_group(index).num_rows != int(rows):
            raise GateCProductionError(f"pinned row-group row count drifted: {file_path}#{index}")
    observed = _parquet_schema_hash(metadata)
    if observed != expected_schema_hash:
        raise GateCProductionError(
            f"Parquet schema drift in {file_path}: expected {expected_schema_hash}, "
            f"observed {observed}"
        )
    try:
        yield parquet
    finally:
        parquet.close()


def read_row_group_rows(
    parquet: Any,
    file_path: str,
    row_group: int,
    columns: Sequence[str],
    *,
    start_row: int = 0,
    batch_rows: int = DEFAULT_BATCH_ROWS,
) -> list[dict[str, Any]]:
    """Read one row group from an already-open Parquet file, optionally from ``start_row``."""
    rows: list[dict[str, Any]] = []
    seen = 0
    for batch in parquet.iter_batches(
        batch_size=batch_rows, row_groups=[row_group], columns=list(columns)
    ):
        records = batch.to_pylist()
        if seen + len(records) <= start_row:
            seen += len(records)
            continue
        for record in records:
            if seen >= start_row:
                rows.append(record)
            seen += 1
    return rows


# --------------------------------------------------------------------------------------
# Configuration, fingerprint, checkpoint
# --------------------------------------------------------------------------------------


def _validate_config(config: ProductionConfig) -> None:
    if config.source_key not in SOURCES or config.source_key not in PARQUET_BINDINGS:
        raise GateCProductionError(f"unknown source: {config.source_key}")
    if config.scope_files is not None and config.scope_files < 1:
        raise GateCProductionError("scope_files must be >= 1 when set")
    if (
        not math.isfinite(config.max_wall_seconds)
        or not 0 < config.max_wall_seconds <= MAX_WALL_SECONDS
    ):
        raise GateCProductionError(f"max_wall_seconds must be in (0, {MAX_WALL_SECONDS}]")
    if not 1 <= config.checkpoint_every <= 10_000_000:
        raise GateCProductionError("checkpoint_every must be in [1, 10000000]")
    if not 1 <= config.batch_rows <= 65_536:
        raise GateCProductionError("batch_rows must be in [1, 65536]")
    if config.stop_after_documents is not None and config.stop_after_documents < 1:
        raise GateCProductionError("stop_after_documents must be >= 1 when set")
    paths = [config.output_dir, config.work_dir, _staging_path(config.output_dir)]
    if config.cache_dir is not None:
        paths.append(config.cache_dir)
    for path in paths:
        _ensure_git_ignored(path)


def run_fingerprint(config: ProductionConfig, transport_sha: str) -> str:
    return _sha256_bytes(
        _canonical_json_bytes({
            "tool_schema_version": PRODUCTION_TOOL_SCHEMA_VERSION,
            "spec_version": PRODUCTION_SPEC_VERSION,
            "release_kind": RELEASE_KIND,
            "source_key": config.source_key,
            "filter_contract_sha256": filter_contract_sha256(config.source_key),
            "transport_manifest_sha256": transport_sha,
            "output_dir": str(_ensure_under_workspace(config.output_dir)),
        })
    )


def _new_checkpoint(fingerprint: str, transport_sha: str) -> dict[str, Any]:
    return {
        "tool_schema_version": PRODUCTION_TOOL_SCHEMA_VERSION,
        "run_fingerprint": fingerprint,
        "transport_manifest_sha256": transport_sha,
        "next_file_index": 0,
        "next_row_group": 0,
        "next_row_in_group": 0,
        "counters": ProductionCounters().to_json(),
        "resources": ResourceAccounting().to_json(),
        "per_file": {},
        "documents_sha256": _sha256_bytes(b""),
        "documents_bytes": 0,
        "resume_count": 0,
        "completed": False,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _checkpoint_payload(state: Mapping[str, Any]) -> bytes:
    return _canonical_json_bytes({k: v for k, v in state.items() if k != "checksum"})


def _write_checkpoint(path: Path, state: Mapping[str, Any]) -> None:
    body = {key: value for key, value in state.items() if key != "checksum"}
    body["checksum"] = _sha256_bytes(_checkpoint_payload(body))
    _atomic_replace_bytes(path, json.dumps(body, indent=2, sort_keys=True).encode() + b"\n")


def _read_checkpoint(path: Path, fingerprint: str, transport_sha: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise GateCProductionError(f"checkpoint is unreadable: {path}") from exc
    try:
        state = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateCProductionError("checkpoint is corrupted: not strict UTF-8 JSON") from exc
    if not isinstance(state, dict):
        raise GateCProductionError("checkpoint is corrupted: not an object")
    checksum = state.get("checksum")
    if not isinstance(checksum, str) or checksum != _sha256_bytes(_checkpoint_payload(state)):
        raise GateCProductionError("checkpoint checksum mismatch; refusing to resume")
    if state.get("tool_schema_version") != PRODUCTION_TOOL_SCHEMA_VERSION:
        raise GateCProductionError("checkpoint tool schema version mismatch")
    if state.get("run_fingerprint") != fingerprint:
        raise GateCProductionError("checkpoint run fingerprint mismatch; refusing to resume")
    if state.get("transport_manifest_sha256") != transport_sha:
        raise GateCProductionError("checkpoint transport manifest mismatch; refusing to resume")
    for name in (
        "next_file_index",
        "next_row_group",
        "next_row_in_group",
        "documents_bytes",
        "resume_count",
    ):
        _require_nonnegative_int(state.get(name), name)
    if not isinstance(state.get("documents_sha256"), str):
        raise GateCProductionError("checkpoint field is invalid: documents_sha256")
    if not isinstance(state.get("completed"), bool):
        raise GateCProductionError("checkpoint field is invalid: completed")
    if not isinstance(state.get("per_file"), Mapping):
        raise GateCProductionError("checkpoint field is invalid: per_file")
    ProductionCounters.from_json(state.get("counters") or {})
    ResourceAccounting.from_json(state.get("resources") or {})
    return state


def restore_and_rebuild_dedup(
    documents_path: Path, state: Mapping[str, Any]
) -> tuple[set[str], set[str], Any, int]:
    """Truncate to the committed prefix and rebuild dedup state from it.

    At production scale the seen-id / seen-hash sets are too large to serialize into a checkpoint
    rewritten many times, so they are not stored: they are exactly the keys of the committed
    candidate prefix.  A duplicate rejection never enters the sets, so the rebuild is exact.
    """
    expected_bytes = int(state["documents_bytes"])
    expected_sha = str(state["documents_sha256"])
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    digest = hashlib.sha256()

    if not documents_path.exists():
        if expected_bytes != 0:
            raise GateCProductionError(
                "checkpoint references accepted documents but the candidate file is missing"
            )
        _write_new_file(documents_path, b"")
        return seen_ids, seen_hashes, digest, 0

    actual_size = documents_path.stat().st_size
    if actual_size < expected_bytes:
        raise GateCProductionError("candidate file is shorter than its committed prefix")

    remaining = expected_bytes
    with open(documents_path, "rb") as handle:
        for line in handle:
            if remaining <= 0:
                break
            if len(line) > remaining:
                raise GateCProductionError("committed prefix does not end on a record boundary")
            remaining -= len(line)
            digest.update(line)
            if not line.strip():
                continue
            record = json.loads(line.decode("utf-8", errors="strict"))
            seen_ids.add(record["source_record_id"])
            seen_hashes.add(record["text_sha256"])
    if remaining != 0:
        raise GateCProductionError("committed prefix could not be read in full")
    if digest.hexdigest() != expected_sha:
        raise GateCProductionError("candidate prefix does not match the committed checkpoint hash")
    if actual_size != expected_bytes:
        with open(documents_path, "r+b") as handle:
            handle.truncate(expected_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(documents_path.parent)
    return seen_ids, seen_hashes, digest, expected_bytes


# --------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------


def _object_integrity_claim(
    transport: Mapping[str, Any],
    per_file: Mapping[str, Mapping[str, Any]],
    files_in_scope: int,
) -> dict[str, Any]:
    """What the traversal actually did to every in-scope object, stated for the release.

    The transport manifest is built by a preflight that reads footers only.  The release must not
    inherit that preflight's "nothing is downloaded in full" framing, because production traversal
    materializes and hashes each in-scope object completely before evaluating any of its rows.
    """
    contract = transport["production_traversal_object_integrity"]
    verified = sum(1 for entry in per_file.values() if entry.get("integrity_verified") is True)
    return {
        "objects_materialized_completely_before_row_evaluation": True,
        "expected_size_verified": True,
        "local_full_file_sha256_computed": True,
        "pinned_oids_are_sha256": bool(contract["pinned_oids_are_sha256"]),
        "local_sha256_compared_with_pinned_oid": bool(
            contract["local_sha256_compared_with_pinned_oid"]
        ),
        "scope_files_integrity_verified": verified,
        "all_scope_files_integrity_verified": verified == files_in_scope,
        "note": (
            "Every file in the declared scope was transferred in full, size-checked and SHA-256 "
            "hashed locally before a single one of its rows reached the frozen Gate C filter. "
            "The preflight transport manifest, which reads Parquet footers only, is a separate "
            "step and its preflight_full_file_sha256_computed flag describes only that step."
        ),
    }


def _make_manifest(
    config: ProductionConfig,
    transport: Mapping[str, Any],
    state: Mapping[str, Any],
    counters: ProductionCounters,
    resources: ResourceAccounting,
    *,
    fingerprint: str,
    wall_seconds: float,
    documents_sha256: str,
    documents_bytes: int,
) -> dict[str, Any]:
    spec = SOURCES[config.source_key]
    scope = scope_from_manifest(transport)
    per_file = {key: state["per_file"][key] for key in sorted(state["per_file"])}
    return {
        "tool_schema_version": PRODUCTION_TOOL_SCHEMA_VERSION,
        "spec_version": PRODUCTION_SPEC_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "release_kind": RELEASE_KIND,
        "release_description": (
            "A production candidate population for one frozen Non-Python Gate C source, produced "
            "by complete sequential traversal of an exactly-described deterministic scope. This "
            "is a candidate pool, not a final training corpus and not a Stage A/B selection."
        ),
        "source": {
            "key": spec.key,
            "dataset": spec.dataset,
            "config": spec.dataset_config,
            "split": spec.split,
            "revision": spec.revision,
            "license": spec.license,
            "parquet_repo_id": transport["parquet_repo_id"],
            "parquet_revision": transport["parquet_revision"],
            "path_prefix": transport["path_prefix"],
            "transport": transport["transport"],
        },
        "candidate_scope": scope.to_json(),
        "full_source_traversal": scope.full_source_traversal,
        "pinned_file_count": transport["pinned_file_count"],
        "pinned_total_bytes": transport["pinned_total_bytes"],
        "scope_bytes": transport["scope_bytes"],
        "scope_rows": transport["scope_rows"],
        "scope_byte_share_of_pinned_source": (
            transport["scope_bytes"] / transport["pinned_total_bytes"]
            if transport["pinned_total_bytes"]
            else None
        ),
        "transport_manifest_sha256": transport["manifest_sha256"],
        "transport_manifest_file": TRANSPORT_MANIFEST_NAME,
        "parquet_schema_hash": transport["schema_hash"],
        "object_integrity": _object_integrity_claim(transport, per_file, scope.file_count),
        "filter_contract": filter_contract(config.source_key),
        "filter_contract_sha256": filter_contract_sha256(config.source_key),
        "sampler": None,
        "sampler_note": (
            "No sampler is used. Every row of every file in the declared scope is evaluated. The "
            "frozen C0 diagnostic sampler is deliberately not imported so that no diagnostic "
            "sampling claim is inherited by a production candidate population."
        ),
        "run_fingerprint": fingerprint,
        "accounting": counters.to_json(),
        "yield_rate": (counters.accepted / counters.scanned) if counters.scanned else None,
        "proxy_tokens_4_bytes_per_token": counters.accepted_text_bytes / 4.0,
        "proxy_token_label": "NOT CANONICAL TOKEN COUNT",
        "proxy_token_note": (
            "4.0 bytes/token is the frozen planning proxy. No tokenizer exists at Gate C, so this "
            "is not a canonical token count and must not be used as one."
        ),
        "resources": resources.to_json(),
        "per_file": per_file,
        "files_complete": sum(1 for entry in per_file.values() if entry["complete"]),
        "files_in_scope": scope.file_count,
        "all_scope_files_complete": (
            sum(1 for entry in per_file.values() if entry["complete"]) == scope.file_count
        ),
        "resume_count": _require_nonnegative_int(state.get("resume_count"), "resume_count"),
        "wall_seconds": round(wall_seconds, 3),
        "documents_file": DOCUMENTS_NAME,
        "documents_sha256": documents_sha256,
        "documents_bytes": documents_bytes,
        "gate_c_scope": {
            "chat_conversion": False,
            "textual_document_separator": False,
            "tokenizer_counting": False,
            "bos_eos_inserted": False,
            "document_truncation": False,
            "cross_source_near_dedup": False,
            "intra_source_near_dedup": False,
            "benchmark_decontamination": False,
            "reference_reserve_exclusion": False,
            "stage_a_stage_b_split_performed": False,
            "final_source_quota_applied": False,
        },
        "hard_stops": {
            "near_dedup_started": False,
            "benchmark_decontamination_started": False,
            "reference_reserve_started": False,
            "tokenizer_trained": False,
            "canonical_token_counting_started": False,
            "final_allocation_frozen": False,
            "final_shards_built": False,
            "model_training_started": False,
        },
        "promotion_eligible": False,
        "promotion_eligible_rationale": (
            "Promotion requires near-dedup, benchmark decontamination, reference-reserve "
            "exclusion, canonical-token accounting and a frozen allocation, none of which exist "
            "yet. This release is an input to those stages, not their output."
        ),
    }


# --------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------


def _new_file_entry(path: str, rows: int, row_groups: int, size: int) -> dict[str, Any]:
    return {
        "path": path,
        "pinned_rows": rows,
        "pinned_row_groups": row_groups,
        "pinned_bytes": size,
        "scanned": 0,
        "accepted": 0,
        "rejected": 0,
        "accepted_text_bytes": 0,
        "integrity_verified": False,
        "complete": False,
    }


def build_production_candidates(
    config: ProductionConfig,
    *,
    store: ObjectStore | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Traverse the declared scope completely and publish one production candidate release."""
    _validate_config(config)
    store = store if store is not None else HubObjectStore()
    spec = SOURCES[config.source_key]
    binding = PARQUET_BINDINGS[config.source_key]
    columns = SOURCE_COLUMNS[config.source_key]

    output_dir = _ensure_under_workspace(config.output_dir)
    if output_dir.exists():
        raise GateCProductionError(f"refusing to overwrite published output: {output_dir}")
    work_dir = _ensure_under_workspace(config.work_dir)
    cache_dir = _ensure_under_workspace(
        config.cache_dir if config.cache_dir is not None else config.work_dir / "objects"
    )
    staging = _staging_path(output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)

    resources = ResourceAccounting()
    transport_path = work_dir / TRANSPORT_MANIFEST_NAME
    if transport_path.exists():
        transport = json.loads(transport_path.read_bytes().decode("utf-8"))
        verify_scoped_transport_manifest(transport)
        # A resume reuses the pinned scope rather than re-listing the source, so a *different*
        # requested scope must fail closed instead of being silently ignored: the operator would
        # otherwise ask for K files and receive a release built over the earlier scope.
        pinned_total = int(transport["pinned_file_count"])
        requested = (
            pinned_total if config.scope_files is None else min(config.scope_files, pinned_total)
        )
        if int(transport["scope_file_count"]) != requested:
            raise GateCProductionError(
                "requested scope does not match the persisted transport manifest: "
                f"requested {requested} files, pinned scope has {transport['scope_file_count']}"
            )
        if transport["source_key"] != config.source_key:
            raise GateCProductionError(
                "persisted transport manifest belongs to a different source; refusing to resume"
            )
    else:
        transport = build_scoped_transport_manifest(
            config.source_key, store, scope_files=config.scope_files, resources=resources
        )
        _write_new_file(
            transport_path, json.dumps(transport, indent=2, sort_keys=True).encode() + b"\n"
        )
    transport_sha = transport["manifest_sha256"]
    scope = scope_from_manifest(transport)
    files = transport["files"]

    fingerprint = run_fingerprint(config, transport_sha)
    checkpoint_path = work_dir / CHECKPOINT_NAME
    documents_path = staging / DOCUMENTS_NAME

    resumed = checkpoint_path.exists()
    if resumed:
        state = _read_checkpoint(checkpoint_path, fingerprint, transport_sha)
        state["resume_count"] = int(state["resume_count"]) + 1
    else:
        state = _new_checkpoint(fingerprint, transport_sha)

    seen_ids, seen_hashes, documents_digest, documents_bytes = restore_and_rebuild_dedup(
        documents_path, state
    )

    counters = ProductionCounters.from_json(state["counters"])
    stored = ResourceAccounting.from_json(state["resources"])
    for name in ResourceAccounting.MEASURED_FIELDS:
        setattr(resources, name, getattr(resources, name) + getattr(stored, name))
    per_file: dict[str, dict[str, Any]] = {
        key: dict(value) for key, value in (state.get("per_file") or {}).items()
    }
    next_file_index = int(state["next_file_index"])
    next_row_group = int(state["next_row_group"])
    next_row_in_group = int(state["next_row_in_group"])
    completed = bool(state["completed"])

    start = clock()
    handle = open(documents_path, "ab")

    def commit(is_complete: bool) -> None:
        handle.flush()
        os.fsync(handle.fileno())
        state["next_file_index"] = next_file_index
        state["next_row_group"] = next_row_group
        state["next_row_in_group"] = next_row_in_group
        state["counters"] = counters.to_json()
        state["resources"] = resources.to_json()
        state["per_file"] = per_file
        state["documents_sha256"] = documents_digest.hexdigest()
        state["documents_bytes"] = documents_bytes
        state["completed"] = is_complete
        state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _write_checkpoint(checkpoint_path, state)

    global_row_base = 0
    for index in range(next_file_index):
        global_row_base += int(files[index]["rows"] or 0)

    try:
        if completed:
            stop_reason = "scope_complete"
        else:
            stop_reason = "scope_complete"
            since_checkpoint = 0
            stop = False
            for file_index in range(next_file_index, scope.file_count):
                if file_index != next_file_index:
                    raise GateCProductionError("file cursor desynchronized from traversal order")
                entry_meta = files[file_index]
                path = str(entry_meta["path"])
                row_group_rows = list(entry_meta["row_group_rows"] or ())
                entry = per_file.setdefault(
                    path,
                    _new_file_entry(
                        path,
                        int(entry_meta["rows"] or 0),
                        len(row_group_rows),
                        int(entry_meta["size"]),
                    ),
                )
                rows_before_group = sum(row_group_rows[:next_row_group])

                # One bulk transfer per file, then read from local disk: see
                # fetch_pinned_object() for why per-block Hub reads cannot survive at this scale.
                local_path = fetch_pinned_object(
                    store,
                    binding,
                    path,
                    cache_dir / Path(path).name,
                    expected_size=int(entry_meta["size"]),
                    expected_oid=str(entry_meta.get("oid") or ""),
                    resources=resources,
                )
                entry["integrity_verified"] = True
                resources.parquet_files_opened += 1
                with open_local_parquet(
                    local_path,
                    path,
                    expected_schema_hash=str(transport["schema_hash"]),
                    expected_row_group_rows=[int(value) for value in row_group_rows],
                ) as parquet:
                    for group_index in range(next_row_group, len(row_group_rows)):
                        if group_index != next_row_group:
                            raise GateCProductionError("row-group cursor desynchronized")
                        rows = read_row_group_rows(
                            parquet,
                            path,
                            group_index,
                            columns,
                            start_row=next_row_in_group,
                            batch_rows=config.batch_rows,
                        )
                        resources.row_groups_read += 1
                        if next_row_in_group > 0:
                            resources.resume_reread_row_groups += 1

                        offset = next_row_in_group
                        for position, row in enumerate(rows):
                            row_in_group = offset + position
                            global_row = global_row_base + rows_before_group + row_in_group
                            next_row_in_group = row_in_group + 1
                            counters.scanned += 1
                            entry["scanned"] += 1

                            decision = evaluate_row(row, spec)
                            if not decision.accepted:
                                counters.rejected += 1
                                entry["rejected"] += 1
                                counters.rejections[decision.reason or "unspecified"] += 1
                            else:
                                natural = _natural_id(row, spec, global_row)
                                record_id = _source_record_id(spec.key, natural)
                                text = decision.text or ""
                                encoded = text.encode("utf-8")
                                text_sha = _sha256_bytes(encoded)
                                if record_id in seen_ids:
                                    counters.rejected += 1
                                    entry["rejected"] += 1
                                    counters.rejections["duplicate_source_record_id"] += 1
                                elif text_sha in seen_hashes:
                                    counters.rejected += 1
                                    entry["rejected"] += 1
                                    counters.rejections["duplicate_text_sha256"] += 1
                                else:
                                    seen_ids.add(record_id)
                                    seen_hashes.add(text_sha)
                                    payload = canonical_jsonl_record_bytes({
                                        "source_key": spec.key,
                                        "source_record_id": record_id,
                                        "natural_id": natural,
                                        "text": text,
                                        "text_sha256": text_sha,
                                        "text_bytes": len(encoded),
                                        "file_path": path,
                                        "file_index": file_index,
                                        "row_group": group_index,
                                        "row_in_group": row_in_group,
                                        "global_row_index": global_row,
                                        "metadata": dict(decision.metadata),
                                        "diagnostics": list(decision.diagnostics),
                                        "provenance": {
                                            "dataset": spec.dataset,
                                            "config": spec.dataset_config,
                                            "split": spec.split,
                                            "revision": spec.revision,
                                            "license": spec.license,
                                            "parquet_repo_id": binding.repo_id,
                                            "parquet_revision": binding.parquet_revision,
                                            "parquet_file": path,
                                            "transport": "huggingface_hub_parquet_sequential",
                                        },
                                    })
                                    handle.write(payload)
                                    documents_digest.update(payload)
                                    documents_bytes += len(payload)
                                    counters.accepted += 1
                                    entry["accepted"] += 1
                                    counters.accepted_text_bytes += len(encoded)
                                    entry["accepted_text_bytes"] += len(encoded)
                                    for name in decision.diagnostics:
                                        counters.diagnostics[name] += 1
                                    since_checkpoint += 1

                            if config.stop_after_documents is not None and (
                                counters.accepted >= config.stop_after_documents
                            ):
                                stop_reason = "stop_after_documents"
                                stop = True
                                break
                        if stop:
                            break
                        if clock() - start > config.max_wall_seconds:
                            stop_reason = "time_cap"
                            stop = True
                            next_row_group = group_index + 1
                            next_row_in_group = 0
                            break
                        rows_before_group += int(row_group_rows[group_index])
                        next_row_group = group_index + 1
                        next_row_in_group = 0
                        if since_checkpoint >= config.checkpoint_every:
                            commit(False)
                            since_checkpoint = 0

                if stop:
                    break
                entry["complete"] = True
                global_row_base += int(entry_meta["rows"] or 0)
                next_file_index = file_index + 1
                next_row_group = 0
                next_row_in_group = 0
                commit(False)
                since_checkpoint = 0
                if not config.keep_object_cache:
                    # Hold one pinned object at a time rather than the whole source.
                    local_path.unlink(missing_ok=True)
            else:
                completed = True
        commit(completed)
    finally:
        handle.close()

    incomplete = [
        str(files[index]["path"])
        for index in range(scope.file_count)
        if not per_file.get(str(files[index]["path"]), {}).get("complete")
    ]
    if incomplete or not completed:
        return {
            "published": False,
            "stop_reason": stop_reason,
            "source_key": config.source_key,
            "work_dir": str(work_dir),
            "staging_dir": str(staging),
            "incomplete_files": len(incomplete),
            "files_complete": scope.file_count - len(incomplete),
            "files_in_scope": scope.file_count,
            "scanned": counters.scanned,
            "accepted": counters.accepted,
            "rejected": counters.rejected,
            "accepted_text_bytes": counters.accepted_text_bytes,
            "next_file_index": next_file_index,
            "next_row_group": next_row_group,
            "next_row_in_group": next_row_in_group,
            "resume_count": int(state["resume_count"]),
            "resumed": resumed,
            "documents_sha256": documents_digest.hexdigest(),
            "resources": resources.to_json(),
        }

    wall_seconds = clock() - start
    manifest = _make_manifest(
        config,
        transport,
        state,
        counters,
        resources,
        fingerprint=fingerprint,
        wall_seconds=wall_seconds,
        documents_sha256=documents_digest.hexdigest(),
        documents_bytes=documents_bytes,
    )
    _write_new_file(
        staging / MANIFEST_NAME, json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    )
    _write_new_file(
        staging / TRANSPORT_MANIFEST_NAME,
        json.dumps(transport, indent=2, sort_keys=True).encode() + b"\n",
    )
    digest = hashlib.sha256()
    with open(staging / DOCUMENTS_NAME, "rb") as reader:
        while True:
            block = reader.read(READ_BLOCK_BYTES)
            if not block:
                break
            digest.update(block)
    if digest.hexdigest() != manifest["documents_sha256"]:
        raise GateCProductionError("staged candidate file does not match its incremental digest")
    _write_new_file(
        staging / CHECKSUMS_NAME,
        "".join(
            f"{_sha256_bytes((staging / name).read_bytes())}  {name}\n"
            if name != DOCUMENTS_NAME
            else f"{digest.hexdigest()}  {name}\n"
            for name in (DOCUMENTS_NAME, MANIFEST_NAME, TRANSPORT_MANIFEST_NAME)
        ).encode(),
    )
    _publish_directory(staging, output_dir)

    return {
        "published": True,
        "source_key": config.source_key,
        "output_dir": str(output_dir),
        "release_kind": RELEASE_KIND,
        "candidate_scope": scope.to_json()["description"],
        "full_source_traversal": scope.full_source_traversal,
        "files_in_scope": scope.file_count,
        "pinned_file_count": transport["pinned_file_count"],
        "stop_reason": stop_reason,
        "scanned": counters.scanned,
        "accepted": counters.accepted,
        "rejected": counters.rejected,
        "accepted_text_bytes": counters.accepted_text_bytes,
        "proxy_tokens_4_bytes_per_token": counters.accepted_text_bytes / 4.0,
        "yield_rate": (counters.accepted / counters.scanned) if counters.scanned else None,
        "wall_seconds": round(wall_seconds, 3),
        "resume_count": int(state["resume_count"]),
        "resumed": resumed,
        "documents_sha256": digest.hexdigest(),
        "manifest_sha256": _sha256_bytes((output_dir / MANIFEST_NAME).read_bytes()),
        "rejections": dict(sorted(counters.rejections.items())),
        "resources": resources.to_json(),
    }


# --------------------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------------------


def _verify_object_integrity(manifest: Mapping[str, Any], transport: Mapping[str, Any]) -> None:
    """Reject a release whose metadata contradicts what production traversal actually does.

    A release built before this contract existed claimed, at the top level of its transport
    manifest, that no pinned object is downloaded in full and that no local full-file SHA-256 is
    computed — while its traversal was in fact materializing, size-checking and hashing every
    in-scope object before evaluating a row.  That claim is refused rather than reinterpreted.
    """
    if "local_full_file_sha256_computed" in transport:
        raise GateCProductionError(
            "transport manifest carries the superseded top-level "
            "local_full_file_sha256_computed claim, which contradicts the production traversal "
            "integrity contract; rebuild the release with the current builder"
        )
    if transport.get("preflight_full_file_sha256_computed") is not False:
        raise GateCProductionError(
            "transport manifest must record preflight_full_file_sha256_computed=false: "
            "manifest construction reads Parquet footers only"
        )
    contract = transport.get("production_traversal_object_integrity")
    if not isinstance(contract, Mapping):
        raise GateCProductionError(
            "transport manifest does not describe production traversal object integrity"
        )
    claim = manifest.get("object_integrity")
    if not isinstance(claim, Mapping):
        raise GateCProductionError("release manifest does not declare object_integrity")
    for flag in (
        "objects_materialized_completely_before_row_evaluation",
        "expected_size_verified",
        "local_full_file_sha256_computed",
    ):
        if contract.get(flag) is not True:
            raise GateCProductionError(
                f"transport manifest must record production object integrity: {flag}"
            )
        if claim.get(flag) is not True:
            raise GateCProductionError(f"release object_integrity must record: {flag}")
    for flag in ("pinned_oids_are_sha256", "local_sha256_compared_with_pinned_oid"):
        if not isinstance(contract.get(flag), bool) or claim.get(flag) != contract.get(flag):
            raise GateCProductionError(
                f"release object_integrity disagrees with the transport contract: {flag}"
            )

    files_in_scope = int(manifest["files_in_scope"])
    verified = sum(
        1 for entry in manifest["per_file"].values() if entry.get("integrity_verified") is True
    )
    if verified != files_in_scope:
        raise GateCProductionError(
            "a published release must integrity-verify every in-scope object: "
            f"{verified} of {files_in_scope}"
        )
    if claim.get("scope_files_integrity_verified") != files_in_scope:
        raise GateCProductionError(
            "object_integrity.scope_files_integrity_verified does not match the scope"
        )
    if claim.get("all_scope_files_integrity_verified") is not True:
        raise GateCProductionError(
            "object_integrity.all_scope_files_integrity_verified must be true"
        )
    measured = manifest["resources"]["measured"]
    if int(measured["object_integrity_verifications"]) < files_in_scope:
        raise GateCProductionError(
            "measured object_integrity_verifications is below one per in-scope object"
        )


def verify_release(output_dir: Path) -> dict[str, Any]:
    """Recompute every load-bearing invariant of a published production candidate release."""
    output_dir = _ensure_under_workspace(output_dir)
    manifest_bytes = (output_dir / MANIFEST_NAME).read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8", errors="strict"))

    if manifest.get("tool_schema_version") != PRODUCTION_TOOL_SCHEMA_VERSION:
        raise GateCProductionError("release was not produced by this tool schema version")
    if manifest.get("release_kind") != RELEASE_KIND:
        raise GateCProductionError(f"release_kind must be {RELEASE_KIND}")
    source_key = manifest["source"]["key"]
    if manifest.get("filter_contract_sha256") != filter_contract_sha256(source_key):
        raise GateCProductionError("release filter contract does not match the frozen Gate C spec")
    if manifest.get("sampler") is not None:
        raise GateCProductionError("a production candidate release must not carry a sampler claim")
    for flag in ("promotion_eligible",):
        if manifest.get(flag) is not False:
            raise GateCProductionError(f"{flag} must be false in a Gate C candidate release")
    for name, value in manifest["gate_c_scope"].items():
        if value is not False:
            raise GateCProductionError(f"gate_c_scope.{name} must be false")
    for name, value in manifest["hard_stops"].items():
        if value is not False:
            raise GateCProductionError(f"hard_stops.{name} must be false")
    if not manifest["all_scope_files_complete"]:
        raise GateCProductionError("a published release must have every in-scope file complete")
    if manifest["files_complete"] != manifest["files_in_scope"]:
        raise GateCProductionError("file completion accounting does not match the scope")

    transport = json.loads((output_dir / TRANSPORT_MANIFEST_NAME).read_bytes().decode("utf-8"))
    verify_scoped_transport_manifest(transport)
    if transport["manifest_sha256"] != manifest["transport_manifest_sha256"]:
        raise GateCProductionError("release transport manifest does not match the build manifest")
    scope = scope_from_manifest(transport)
    if scope.full_source_traversal != manifest["full_source_traversal"]:
        raise GateCProductionError("scope traversal claim disagrees with the transport manifest")
    _verify_object_integrity(manifest, transport)
    pinned_paths = {str(entry["path"]) for entry in transport["files"]}

    spec = SOURCES[source_key]
    digest = hashlib.sha256()
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    per_file: Counter = Counter()
    accepted = 0
    accepted_text_bytes = 0
    total_bytes = 0

    with open(output_dir / DOCUMENTS_NAME, "rb") as handle:
        for line in handle:
            digest.update(line)
            total_bytes += len(line)
            if not line.strip():
                continue
            if not line.endswith(b"\n"):
                raise GateCProductionError("candidate file ends with an unterminated record")
            decoded = line[:-1].decode("utf-8", errors="strict")
            if len(decoded.splitlines()) != 1:
                raise GateCProductionError("physical JSONL framing is ambiguous for a record")
            record = json.loads(decoded)
            encoded = record["text"].encode("utf-8")
            if _sha256_bytes(encoded) != record["text_sha256"]:
                raise GateCProductionError("candidate record text_sha256 was tampered with")
            if len(encoded) != record["text_bytes"]:
                raise GateCProductionError("candidate record text_bytes does not match its text")
            if not spec.min_bytes <= len(encoded) <= spec.max_bytes:
                raise GateCProductionError("candidate record violates the frozen byte band")
            if record["source_key"] != source_key:
                raise GateCProductionError("candidate record carries a foreign source key")
            if record["file_path"] not in pinned_paths:
                raise GateCProductionError("candidate release references an unpinned file")
            if record["source_record_id"] in seen_ids:
                raise GateCProductionError("candidate release contains a duplicate record id")
            if record["text_sha256"] in seen_hashes:
                raise GateCProductionError("candidate release contains a duplicate text hash")
            seen_ids.add(record["source_record_id"])
            seen_hashes.add(record["text_sha256"])
            per_file[record["file_path"]] += 1
            accepted += 1
            accepted_text_bytes += len(encoded)

    if digest.hexdigest() != manifest["documents_sha256"]:
        raise GateCProductionError("documents.jsonl does not match manifest documents_sha256")
    if total_bytes != manifest["documents_bytes"]:
        raise GateCProductionError("documents.jsonl length does not match the manifest")
    if accepted != manifest["accounting"]["accepted"]:
        raise GateCProductionError("published record count does not match the manifest accounting")
    if accepted_text_bytes != manifest["accounting"]["accepted_text_bytes"]:
        raise GateCProductionError("published text bytes do not match the manifest accounting")
    for path, entry in manifest["per_file"].items():
        if per_file.get(path, 0) != entry["accepted"]:
            raise GateCProductionError(f"per-file accepted count does not match documents: {path}")
    for entry in (output_dir / CHECKSUMS_NAME).read_text().splitlines():
        recorded, name = entry.split("  ", 1)
        actual = hashlib.sha256()
        with open(output_dir / name, "rb") as reader:
            while True:
                block = reader.read(READ_BLOCK_BYTES)
                if not block:
                    break
                actual.update(block)
        if actual.hexdigest() != recorded:
            raise GateCProductionError(f"MANIFEST.sha256 mismatch for {name}")

    return {
        "output_dir": str(output_dir),
        "source_key": source_key,
        "release_kind": manifest["release_kind"],
        "full_source_traversal": manifest["full_source_traversal"],
        "candidate_scope": manifest["candidate_scope"]["description"],
        "files_in_scope": manifest["files_in_scope"],
        "accepted": accepted,
        "accepted_text_bytes": accepted_text_bytes,
        "distinct_record_ids": len(seen_ids),
        "distinct_text_hashes": len(seen_hashes),
        "documents_sha256": digest.hexdigest(),
        "manifest_sha256": _sha256_bytes(manifest_bytes),
    }


# --------------------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------------------


def _quantiles(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    count = len(ordered)

    def pick(fraction: float) -> float:
        return ordered[min(count - 1, int(fraction * count))]

    return {
        "n": count,
        "min": ordered[0],
        "p05": pick(0.05),
        "p25": pick(0.25),
        "median": pick(0.50),
        "p75": pick(0.75),
        "p95": pick(0.95),
        "p99": pick(0.99),
        "max": ordered[-1],
        "mean": sum(ordered) / count,
    }


def _top(counter: Counter, limit: int = 20) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit])


def diagnose_release(output_dir: Path) -> dict[str, Any]:
    """Streaming composition diagnostics over a published production candidate release."""
    output_dir = _ensure_under_workspace(output_dir)
    manifest = json.loads((output_dir / MANIFEST_NAME).read_bytes().decode("utf-8"))

    lengths = array.array("i")
    per_file: Counter = Counter()
    per_file_bytes: Counter = Counter()
    diagnostics: Counter = Counter()
    metadata_values: dict[str, Counter] = {}
    total = 0
    total_bytes = 0

    with open(output_dir / DOCUMENTS_NAME, "rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line.decode("utf-8", errors="strict"))
            total += 1
            text_bytes = int(record["text_bytes"])
            total_bytes += text_bytes
            lengths.append(min(text_bytes, 2**31 - 1))
            per_file[record["file_path"]] += 1
            per_file_bytes[record["file_path"]] += text_bytes
            for name in record.get("diagnostics") or []:
                diagnostics[name] += 1
            for key, value in (record.get("metadata") or {}).items():
                if isinstance(value, str) and len(value) <= 64:
                    metadata_values.setdefault(key, Counter())[value] += 1

    return {
        "tool_schema_version": PRODUCTION_TOOL_SCHEMA_VERSION,
        "spec_version": PRODUCTION_SPEC_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "release": {
            "output_dir": str(output_dir),
            "source_key": manifest["source"]["key"],
            "release_kind": manifest["release_kind"],
            "full_source_traversal": manifest["full_source_traversal"],
            "candidate_scope": manifest["candidate_scope"],
            "documents_sha256": manifest["documents_sha256"],
            "run_fingerprint": manifest["run_fingerprint"],
        },
        "documents": total,
        "accepted_text_bytes": total_bytes,
        "proxy_tokens_4_bytes_per_token": total_bytes / 4.0,
        "proxy_token_label": "NOT CANONICAL TOKEN COUNT",
        "canonical_token_counting_performed": False,
        "length_bytes": _quantiles(lengths),
        "file_distribution": {
            "documents": dict(sorted(per_file.items())),
            "accepted_text_bytes": dict(sorted(per_file_bytes.items())),
            "files_covered": len(per_file),
        },
        "diagnostics": dict(sorted(diagnostics.items())),
        "metadata_value_histograms": {
            key: _top(counter, limit=15)
            for key, counter in sorted(metadata_values.items())
            if len(counter) <= 512
        },
        "rejection_histogram": manifest["accounting"]["rejections"],
        "yield_rate": manifest["yield_rate"],
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PetitGPT Non-Python Gate C production candidate builder"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser(
        "preflight", help="pin the scoped transport manifest without reading any row"
    )
    preflight.add_argument("--source", required=True, choices=sorted(PARQUET_BINDINGS))
    preflight.add_argument("--scope-files", type=int, default=None)
    preflight.add_argument("--out", required=True, type=Path)

    build = subparsers.add_parser("build", help="build one production candidate release")
    build.add_argument("--source", required=True, choices=sorted(PARQUET_BINDINGS))
    build.add_argument("--output-dir", required=True, type=Path)
    build.add_argument("--work-dir", required=True, type=Path)
    build.add_argument("--cache-dir", type=Path, default=None)
    build.add_argument("--keep-object-cache", action="store_true")
    build.add_argument(
        "--scope-files",
        type=int,
        default=None,
        help="traverse only the first K pinned files; omit for a full-source traversal",
    )
    build.add_argument("--max-wall-seconds", type=float, default=float(MAX_WALL_SECONDS))
    build.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    build.add_argument("--batch-rows", type=int, default=DEFAULT_BATCH_ROWS)
    build.add_argument("--stop-after-documents", type=int, default=None)

    verify = subparsers.add_parser("verify", help="verify a published production release")
    verify.add_argument("--output-dir", required=True, type=Path)

    diagnose = subparsers.add_parser("diagnose", help="composition diagnostics for a release")
    diagnose.add_argument("--output-dir", required=True, type=Path)
    diagnose.add_argument("--out", required=True, type=Path)
    return parser


def _write_json_output(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    out = _ensure_under_workspace(path)
    _ensure_git_ignored(out)
    if out.exists():
        raise GateCProductionError(f"refusing to overwrite output: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_new_file(out, json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n")
    return {"out": str(out), "sha256": _sha256_bytes(out.read_bytes())}


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "preflight":
            manifest = build_scoped_transport_manifest(
                args.source, HubObjectStore(), scope_files=args.scope_files
            )
            result = _write_json_output(args.out, manifest)
            result.update({
                "source_key": manifest["source_key"],
                "pinned_file_count": manifest["pinned_file_count"],
                "pinned_total_bytes": manifest["pinned_total_bytes"],
                "scope_file_count": manifest["scope_file_count"],
                "scope_bytes": manifest["scope_bytes"],
                "scope_rows": manifest["scope_rows"],
                "full_source_traversal": manifest["scope"]["mode"] == FULL_SOURCE,
            })
        elif args.command == "build":
            result = build_production_candidates(
                ProductionConfig(
                    source_key=args.source,
                    output_dir=args.output_dir,
                    work_dir=args.work_dir,
                    cache_dir=args.cache_dir,
                    keep_object_cache=args.keep_object_cache,
                    scope_files=args.scope_files,
                    max_wall_seconds=args.max_wall_seconds,
                    checkpoint_every=args.checkpoint_every,
                    batch_rows=args.batch_rows,
                    stop_after_documents=args.stop_after_documents,
                )
            )
        elif args.command == "verify":
            result = verify_release(args.output_dir)
        else:
            diagnostics = diagnose_release(args.output_dir)
            result = _write_json_output(args.out, diagnostics)
            result.update({
                "documents": diagnostics["documents"],
                "proxy_tokens_4_bytes_per_token": diagnostics["proxy_tokens_4_bytes_per_token"],
                "canonical_token_counting_performed": False,
            })
    except GateCError as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    except OSError as exc:
        print(json.dumps({"error": f"io error: {exc}"}, indent=2), file=sys.stderr)
        return 2
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError) as exc:
        print(
            json.dumps({"error": f"malformed release artifact: {exc}"}, indent=2), file=sys.stderr
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
