#!/usr/bin/env python3

"""Gate C v2: pinned official Parquet transport plus a deterministic dispersed cluster scan.

This helper exists because the v1 builder in ``pretrain/corpus_gate_c.py`` traverses the Hugging
Face Dataset Server strictly sequentially from a scalar row cursor, which makes every real C0
release a contiguous head of the split.  See ``GATE_C_V2_DESIGN_REVIEW.md`` for the confirmed
finding.

What lives here: the pinned Parquet topology manifest, the seeded cluster sampler, the
persisted selection plan, the v2 checkpoint contract, and the Parquet body transport.

What does **not** live here: source bindings and source-local filters.  Those stay frozen in
``pretrain/corpus_gate_c.py`` and are imported, never copied.  ``pretrain/corpus_gate_e.py`` is a
different frozen tool and is neither imported nor modified.

Out of scope at this gate, asserted in every manifest: chat conversion, textual document
separators, tokenizer counting, BOS/EOS insertion, cross-source near-dedup, benchmark
decontamination, reference-reserve exclusion.  Every release this module can produce is
``release_kind = c0_diagnostic`` with ``promotion_eligible = false``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The frozen v1 module owns the source bindings, the filter layer and the atomic-IO primitives.
# Importing them here is deliberate reuse: no source filter is reimplemented in this file.
from pretrain.corpus_gate_c import (  # noqa: E402
    SOURCES,
    BuildCounters,
    Decision,
    GateCError,
    SourceSpec,
    _canonical_json_bytes,
    _ensure_git_ignored,
    _ensure_revision,
    _ensure_under_workspace,
    _fsync_directory,
    _publish_directory,
    _sha256_bytes,
    _source_record_id,
    _write_new_file,
    canonical_jsonl_record_bytes,
    evaluate_row,
)

TOOL_SCHEMA_VERSION = "petitgpt-corpus-gate-c-v2"
SAMPLER_VERSION = "pps-rowgroup-fixed-head-cluster-v2"
ALLOWED_SAMPLER_CLAIM = (
    "deterministic seed-sensitive dispersed Parquet cluster diagnostic, PPS at file/row-group "
    "layers where recorded, without replacement at selected row-group-unit scope, not a "
    "row-level representative sampler"
)
SPEC_VERSION = "nonpython-gate-c-source-spec-2026-08-16"

TRANSPORT = "huggingface_hub_parquet"

MANIFEST_NAME = "manifest.json"
DOCUMENTS_NAME = "documents.jsonl"
CHECKSUMS_NAME = "MANIFEST.sha256"
CHECKPOINT_NAME = "checkpoint.json"
TRANSPORT_MANIFEST_NAME = "transport_manifest.json"
SELECTION_PLAN_NAME = "selection_plan.json"

READ_BLOCK_BYTES = 1024 * 1024
MAX_UNITS = 4096
MAX_ACCEPTED_DOCUMENTS = 8192
MAX_SCANNED_RECORDS = 50_000
DEFAULT_UNITS = 32
FEISTEL_ROUNDS = 4


class GateCParquetError(GateCError):
    """A fail-closed Gate C v2 contract error."""


class PlannedInterruption(RuntimeError):
    """Raised by tests to interrupt a v2 build at an exact point."""


# --------------------------------------------------------------------------------------
# Pinned Parquet bindings
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ParquetBinding:
    """Where the official Parquet bytes for a frozen source binding actually live.

    ``parquet_revision`` differs from ``SourceSpec.revision`` only when the pinned data revision
    stores no Parquet natively and the official Hub Parquet export must be used instead.  Both are
    recorded; neither is inferred at run time.
    """

    source_key: str
    repo_id: str
    parquet_revision: str
    path_prefix: str
    footer_policy: str  # "complete" | "selected_files_only"
    export_note: str = ""

    @property
    def is_export_branch(self) -> bool:
        return self.parquet_revision != SOURCES[self.source_key].revision


PARQUET_BINDINGS: dict[str, ParquetBinding] = {
    "fineweb_edu_dedup": ParquetBinding(
        source_key="fineweb_edu_dedup",
        repo_id="HuggingFaceTB/smollm-corpus",
        parquet_revision="3ba9d605774198c5868892d7a8deda78031a781f",
        path_prefix="fineweb-edu-dedup/",
        footer_policy="selected_files_only",
        export_note="native Parquet at the pinned data revision. official pinned Hub Parquet path used for this diagnostic; full source-population equivalence is outside this correction task",
    ),
    "dclm_edu": ParquetBinding(
        source_key="dclm_edu",
        repo_id="HuggingFaceTB/dclm-edu",
        parquet_revision="dbad8ad71224482740cd9c9d353591adbf62fe04",
        path_prefix="data/",
        footer_policy="selected_files_only",
        export_note="native Parquet; very large row groups observed. official pinned Hub Parquet path used for this diagnostic; full source-population equivalence is outside this correction task",
    ),
    "finewiki_en": ParquetBinding(
        source_key="finewiki_en",
        repo_id="HuggingFaceFW/finewiki",
        parquet_revision="8bd13e72e6a002407649b3e898535f42ceb1aeb9",
        path_prefix="data/enwiki/",
        footer_policy="complete",
        export_note="native Parquet for the en config. official pinned Hub Parquet path used for this diagnostic; full source-population equivalence is outside this correction task",
    ),
    "pes2o": ParquetBinding(
        source_key="pes2o",
        repo_id="allenai/dolmino-mix-1124",
        parquet_revision="c58ab4b6ff990115e1ff3121953754ee2bc29501",
        path_prefix="pes2o/partial-train/",
        footer_policy="complete",
        export_note=(
            "the official Hub Parquet conversion path refs/convert/parquet is used for this "
            "diagnostic. The 'partial-train' directory name is not interpreted here. official pinned Hub Parquet path used for this diagnostic; full source-population equivalence is outside this correction task"
        ),
    ),
    "stackexchange": ParquetBinding(
        source_key="stackexchange",
        repo_id="allenai/dolmino-mix-1124",
        parquet_revision="c58ab4b6ff990115e1ff3121953754ee2bc29501",
        path_prefix="stackexchange/partial-train/",
        footer_policy="complete",
        export_note=(
            "official Hub Parquet conversion path refs/convert/parquet used for this diagnostic. "
            "official pinned Hub Parquet path used for this diagnostic; full source-population equivalence is outside this correction task"
        ),
    ),
    "cosmopedia_v2": ParquetBinding(
        source_key="cosmopedia_v2",
        repo_id="HuggingFaceTB/smollm-corpus",
        parquet_revision="3ba9d605774198c5868892d7a8deda78031a781f",
        path_prefix="cosmopedia-v2/",
        footer_policy="selected_files_only",
        export_note="native Parquet. official pinned Hub Parquet path used for this diagnostic; full source-population equivalence is outside this correction task",
    ),
    "finephrase_tutorial": ParquetBinding(
        source_key="finephrase_tutorial",
        repo_id="HuggingFaceFW/finephrase",
        parquet_revision="78cf4a5ed0099214979c094c963e699c19163838",
        path_prefix="tutorial/",
        footer_policy="selected_files_only",
        export_note="native Parquet for the tutorial config. official pinned Hub Parquet path used for this diagnostic; full source-population equivalence is outside this correction task",
    ),
}

# Columns each source actually needs.  Projection is what makes a bounded Parquet read affordable.
SOURCE_COLUMNS: dict[str, tuple[str, ...]] = {
    "fineweb_edu_dedup": ("id", "text", "metadata"),
    "dclm_edu": (
        "id",
        "text",
        "edu_int_score",
        "edu_score",
        "fasttext_score",
        "url",
        "language",
        "language_score",
    ),
    "finewiki_en": (
        "id",
        "text",
        "title",
        "page_id",
        "url",
        "version",
        "wikidata_id",
        "wikiname",
        "in_language",
        "date_modified",
        "has_math",
        "bytes_html",
    ),
    "pes2o": ("id", "text", "metadata", "source", "version", "added", "created"),
    "stackexchange": ("id", "text", "metadata", "source", "version"),
    "cosmopedia_v2": ("text", "format", "audience", "seed_data", "token_length"),
    "finephrase_tutorial": (
        "id",
        "text",
        "rollout_results",
        "url",
        "dump",
        "dataset",
        "score",
        "int_score",
        "language",
        "language_score",
        "token_count",
    ),
}


# --------------------------------------------------------------------------------------
# Transport: a small injectable object-store interface so tests need no network
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RemoteObject:
    """One pinned remote Parquet object."""

    path: str
    size: int
    oid: str
    etag: str

    def to_json(self) -> dict[str, Any]:
        return {"path": self.path, "size": self.size, "oid": self.oid, "etag": self.etag}


class _CountingReader(io.RawIOBase):
    """Wraps a random-access file and counts the bytes actually pulled over the wire."""

    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self.bytes_read = 0
        self.read_calls = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self._handle.read(size)
        self.bytes_read += len(chunk)
        self.read_calls += 1
        return chunk

    def readinto(self, buffer: Any) -> int:  # noqa: D102 - RawIOBase protocol
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._handle.seek(offset, whence)

    def tell(self) -> int:
        return self._handle.tell()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    @property
    def closed(self) -> bool:  # pyarrow inspects this
        return False


class ObjectStore:
    """Minimal pinned-object interface: list a prefix, and open one object for ranged reads."""

    def list_objects(self, binding: ParquetBinding) -> tuple[RemoteObject, ...]:
        raise NotImplementedError

    @contextmanager
    def open(self, binding: ParquetBinding, path: str) -> Iterator[_CountingReader]:
        raise NotImplementedError


class HubObjectStore(ObjectStore):
    """Official Hugging Face Hub transport: HfApi metadata plus HfFileSystem ranged reads."""

    def __init__(self, *, block_bytes: int = READ_BLOCK_BYTES, max_attempts: int = 3) -> None:
        self.block_bytes = block_bytes
        self.max_attempts = max_attempts
        self._api = None
        self._fs = None

    def _hub(self) -> tuple[Any, Any]:
        if self._api is None or self._fs is None:
            from huggingface_hub import HfApi, HfFileSystem

            self._api = HfApi()
            self._fs = HfFileSystem()
        return self._api, self._fs

    def list_objects(self, binding: ParquetBinding) -> tuple[RemoteObject, ...]:
        api, _ = self._hub()
        info = api.repo_info(
            binding.repo_id,
            revision=binding.parquet_revision,
            repo_type="dataset",
            files_metadata=True,
        )
        objects = []
        for sibling in info.siblings:
            name = sibling.rfilename
            if not name.startswith(binding.path_prefix) or not name.endswith(".parquet"):
                continue
            lfs = getattr(sibling, "lfs", None)
            oid = ""
            if lfs is not None:
                oid = getattr(lfs, "sha256", None) or (
                    lfs.get("sha256", "") if isinstance(lfs, Mapping) else ""
                )
            objects.append(
                RemoteObject(
                    path=name,
                    size=int(sibling.size or 0),
                    oid=str(oid or ""),
                    etag=str(getattr(sibling, "blob_id", "") or ""),
                )
            )
        if not objects:
            raise GateCParquetError(f"no pinned Parquet objects under {binding.path_prefix}")
        return tuple(sorted(objects, key=lambda item: item.path))

    @contextmanager
    def open(self, binding: ParquetBinding, path: str) -> Iterator[_CountingReader]:
        _, fs = self._hub()
        uri = f"datasets/{binding.repo_id}@{binding.parquet_revision}/{path}"
        last: BaseException | None = None
        for attempt in range(self.max_attempts):
            try:
                handle = fs.open(uri, "rb", block_size=self.block_bytes)
                break
            except Exception as exc:  # noqa: BLE001 - bounded retry of the same pinned object
                last = exc
                if attempt + 1 >= self.max_attempts:
                    raise GateCParquetError(
                        f"pinned object unavailable after bounded retries: {path}"
                    ) from last
                time.sleep(float(2**attempt))
        reader = _CountingReader(handle)
        try:
            yield reader
        finally:
            handle.close()


# --------------------------------------------------------------------------------------
# Transport manifest
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FileEntry:
    """One pinned Parquet file plus whatever exact topology has been read for it."""

    obj: RemoteObject
    rows: int | None = None
    row_group_rows: tuple[int, ...] | None = None
    schema_hash: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            **self.obj.to_json(),
            "rows": self.rows,
            "row_group_rows": list(self.row_group_rows) if self.row_group_rows else None,
            "row_group_count": len(self.row_group_rows) if self.row_group_rows else None,
            "schema_hash": self.schema_hash,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> FileEntry:
        groups = value.get("row_group_rows")
        return cls(
            obj=RemoteObject(
                path=str(value["path"]),
                size=int(value["size"]),
                oid=str(value["oid"]),
                etag=str(value["etag"]),
            ),
            rows=value.get("rows"),
            row_group_rows=tuple(int(x) for x in groups) if groups else None,
            schema_hash=value.get("schema_hash"),
        )


def _parquet_schema_hash(metadata: Any) -> str:
    return _sha256_bytes(str(metadata.schema.to_arrow_schema()).encode("utf-8"))


def read_footer(store: ObjectStore, binding: ParquetBinding, obj: RemoteObject) -> FileEntry:
    """Read one Parquet footer and return the exact row/row-group topology of that file."""
    import pyarrow.parquet as pq

    with store.open(binding, obj.path) as reader:
        parquet = pq.ParquetFile(reader, pre_buffer=False, buffer_size=READ_BLOCK_BYTES)
        metadata = parquet.metadata
        groups = tuple(metadata.row_group(i).num_rows for i in range(metadata.num_row_groups))
        entry = FileEntry(
            obj=obj,
            rows=int(metadata.num_rows),
            row_group_rows=groups,
            schema_hash=_parquet_schema_hash(metadata),
        )
    if entry.rows != sum(groups):
        raise GateCParquetError(f"row-group rows do not sum to file rows: {obj.path}")
    return entry


def build_transport_manifest(
    source_key: str,
    store: ObjectStore,
    *,
    footer_policy: str | None = None,
    max_footer_files: int | None = None,
) -> dict[str, Any]:
    """Build the pinned Parquet topology manifest for one frozen source binding."""
    if source_key not in PARQUET_BINDINGS:
        raise GateCParquetError(f"unknown source: {source_key}")
    binding = PARQUET_BINDINGS[source_key]
    spec = SOURCES[source_key]
    _ensure_revision(spec.revision)
    _ensure_revision(binding.parquet_revision)
    policy = footer_policy or binding.footer_policy
    if policy not in {"complete", "selected_files_only"}:
        raise GateCParquetError(f"unknown footer policy: {policy}")

    objects = store.list_objects(binding)
    entries: list[FileEntry] = [FileEntry(obj=obj) for obj in objects]
    if policy == "complete":
        limit = len(objects) if max_footer_files is None else min(max_footer_files, len(objects))
        if limit < len(objects):
            raise GateCParquetError("complete footer policy cannot be truncated")
        entries = [read_footer(store, binding, obj) for obj in objects]
        schema_hashes = {entry.schema_hash for entry in entries}
        if len(schema_hashes) != 1:
            raise GateCParquetError(f"Parquet schema differs between files for {source_key}")

    complete = policy == "complete"
    total_rows = sum(entry.rows or 0 for entry in entries) if complete else None
    manifest = {
        "tool_schema_version": TOOL_SCHEMA_VERSION,
        "spec_version": SPEC_VERSION,
        "source_key": source_key,
        "dataset": spec.dataset,
        "config": spec.dataset_config,
        "split": spec.split,
        "data_revision": spec.revision,
        "parquet_repo_id": binding.repo_id,
        "parquet_revision": binding.parquet_revision,
        "parquet_is_export_branch": binding.is_export_branch,
        "parquet_export_note": binding.export_note,
        "path_prefix": binding.path_prefix,
        "transport": TRANSPORT,
        "file_manifest_complete": True,
        "production_row_group_manifest_complete": complete,
        "footer_policy": policy,
        "file_count": len(entries),
        "total_bytes": sum(entry.obj.size for entry in entries),
        "total_rows": total_rows,
        "row_group_count": (
            sum(len(entry.row_group_rows or ()) for entry in entries) if complete else None
        ),
        "files": [entry.to_json() for entry in entries],
        "local_full_file_sha256_computed": False,
        "local_full_file_sha256_note": (
            "no pinned object was fully downloaded, so no local full-file SHA-256 is claimed; "
            "identity is the Hub LFS OID plus blob etag at the pinned commit"
        ),
        "upstream_population_equivalence": "not_audited_in_this_task",
    }
    manifest["manifest_sha256"] = _sha256_bytes(_canonical_json_bytes(manifest))
    return manifest


def transport_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return _sha256_bytes(_canonical_json_bytes(body))


def verify_transport_manifest(manifest: Mapping[str, Any]) -> str:
    """Checksum **and** semantics.

    This delegates to :func:`validate_transport_manifest_v2` so the build path, the plan validator
    and the release verifier all share one transport contract; a checksum-consistent identity
    forgery cannot pass through any of them.
    """
    return validate_transport_manifest_v2(manifest)


# --------------------------------------------------------------------------------------
# Deterministic sampler
# --------------------------------------------------------------------------------------


def _prf(key: bytes, *parts: Any) -> int:
    payload = b"\x00".join([key] + [str(part).encode("utf-8") for part in parts])
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def feistel_permutation(index: int, domain: int, key: bytes, rounds: int = FEISTEL_ROUNDS) -> int:
    """A keyed pseudorandom permutation of ``range(domain)`` with cycle walking.

    O(1) memory and O(1) time per index, so a traversal order over billions of rows never has to be
    materialised.  Being a permutation is what gives sampling without replacement for free.
    """
    if domain <= 0:
        raise GateCParquetError("permutation domain must be positive")
    if not 0 <= index < domain:
        raise GateCParquetError("permutation index out of range")
    if domain == 1:
        return 0
    half = max(1, (domain - 1).bit_length() // 2 + 1)
    mask = (1 << half) - 1
    value = index
    for _ in range(domain):  # cycle walking: bounded, and in practice one or two iterations
        left = value >> half
        right = value & mask
        for round_index in range(rounds):
            left, right = right, left ^ (_prf(key, round_index, right) & mask)
        candidate = (left << half) | right
        if candidate < domain:
            return candidate
        value = candidate
    raise GateCParquetError("cycle walking failed to converge")


def _sampler_key(seed: int, transport_sha: str, source_key: str) -> bytes:
    return hashlib.sha256(
        f"{SAMPLER_VERSION}\x00{source_key}\x00{transport_sha}\x00{seed}".encode()
    ).digest()


def _weighted_pick(weights: Sequence[int], draw: int) -> int:
    """Pick an index with probability proportional to ``weights`` from a uniform ``draw``."""
    total = sum(weights)
    if total <= 0:
        raise GateCParquetError("weighted pick needs a positive total weight")
    target = draw % total
    cumulative = 0
    for index, weight in enumerate(weights):
        cumulative += weight
        if target < cumulative:
            return index
    raise AssertionError("weighted pick must select an index")


@dataclass(frozen=True)
class PlanUnit:
    """One planned read: a contiguous head slice of one row group of one pinned file."""

    order: int
    file_path: str
    file_oid: str
    row_group: int
    row_group_rows: int
    file_row_offset: int
    rows_to_read: int

    def to_json(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "file_path": self.file_path,
            "file_oid": self.file_oid,
            "row_group": self.row_group,
            "row_group_rows": self.row_group_rows,
            "file_row_offset": self.file_row_offset,
            "rows_to_read": self.rows_to_read,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> PlanUnit:
        return cls(
            order=int(value["order"]),
            file_path=str(value["file_path"]),
            file_oid=str(value["file_oid"]),
            row_group=int(value["row_group"]),
            row_group_rows=int(value["row_group_rows"]),
            file_row_offset=int(value["file_row_offset"]),
            rows_to_read=int(value["rows_to_read"]),
        )


def build_selection_plan(
    manifest: Mapping[str, Any],
    *,
    seed: int,
    units: int,
    rows_per_unit: int,
    store: ObjectStore | None = None,
) -> dict[str, Any]:
    """Draw a deterministic, seed-sensitive, without-replacement dispersed cluster plan.

    Three stages:

    * stage 1 forces a distinct-file first pass, drawing files without replacement with probability
      proportional to exact row counts when the row-group inventory is complete, and to the exact
      Hub **compressed byte size** otherwise. A compressed byte size is a proxy, never an exact row
      weight;
    * stage 2 draws one **row group** inside each resolved file with probability proportional to its
      exact row count within that file;
    * stage 3 reads a contiguous **head slice** of that row group, always starting at row 0.

    What this is NOT: because stage 3 only ever reaches the first ``rows_per_unit`` rows of a
    selected row group, every deeper row has inclusion probability exactly zero. This is therefore
    a dispersed cluster diagnostic, not a row-level representative sampler, and no global exact row
    weighting is established. The seed varies which files and row groups are drawn; it does not vary
    the within-row-group start offset.
    """
    transport_sha = verify_transport_manifest(manifest)
    if not 1 <= units <= MAX_UNITS:
        raise GateCParquetError(f"units must be in [1, {MAX_UNITS}]")
    if rows_per_unit < 1:
        raise GateCParquetError("rows_per_unit must be positive")
    if type(seed) is not int or seed < 0:
        raise GateCParquetError("seed must be a non-negative integer")

    entries = [FileEntry.from_json(item) for item in manifest["files"]]
    if not entries:
        raise GateCParquetError("transport manifest has no files")
    # Validated as an exact bool by validate_transport_manifest_v2 above.
    complete = manifest["production_row_group_manifest_complete"]
    key = _sampler_key(seed, transport_sha, str(manifest["source_key"]))

    # Stage 1: distinct files, PPS, without replacement, in a seeded order.
    weights = [
        (entry.rows if complete and entry.rows is not None else entry.obj.size) for entry in entries
    ]
    if any(weight <= 0 for weight in weights):
        raise GateCParquetError("every pinned file needs a positive sampling weight")
    remaining = list(range(len(entries)))
    remaining_weights = list(weights)
    file_order: list[int] = []
    wanted_files = min(units, len(entries))
    for step in range(wanted_files):
        pick = _weighted_pick(remaining_weights, _prf(key, "file", step))
        file_order.append(remaining[pick])
        remaining.pop(pick)
        remaining_weights.pop(pick)

    binding = PARQUET_BINDINGS[str(manifest["source_key"])]
    resolved: dict[int, FileEntry] = {}
    footer_bytes = 0
    plan_units: list[PlanUnit] = []
    for order in range(units):
        if order < len(file_order):
            # The first units cover distinct files, so file coverage is guaranteed.
            file_index = file_order[order]
        else:
            # Extra units are drawn PPS with replacement over every file, so that overall
            # P(row group) is proportional to its row count rather than to its file's turn.
            file_index = _weighted_pick(weights, _prf(key, "extra", order))
        entry = resolved.get(file_index)
        if entry is None:
            entry = entries[file_index]
            if entry.row_group_rows is None:
                if store is None:
                    raise GateCParquetError(
                        "an incomplete row-group inventory needs an object store to read footers"
                    )
                entry = read_footer(store, binding, entry.obj)
                footer_bytes += 1
            resolved[file_index] = entry
        groups = entry.row_group_rows or ()
        if not groups:
            raise GateCParquetError(f"file has no row groups: {entry.obj.path}")
        used = {unit.row_group for unit in plan_units if unit.file_path == entry.obj.path}
        available = [index for index in range(len(groups)) if index not in used]
        if not available:
            continue
        pick = _weighted_pick([groups[index] for index in available], _prf(key, "rg", order))
        row_group = available[pick]
        plan_units.append(
            PlanUnit(
                order=len(plan_units),
                file_path=entry.obj.path,
                file_oid=entry.obj.oid,
                row_group=row_group,
                row_group_rows=groups[row_group],
                file_row_offset=sum(groups[:row_group]),
                rows_to_read=min(rows_per_unit, groups[row_group]),
            )
        )

    if not plan_units:
        raise GateCParquetError("selection plan is empty")

    plan = {
        "tool_schema_version": TOOL_SCHEMA_VERSION,
        "sampler_version": SAMPLER_VERSION,
        "source_key": manifest["source_key"],
        "dataset": manifest["dataset"],
        "config": manifest["config"],
        "split": manifest["split"],
        "data_revision": manifest["data_revision"],
        "parquet_revision": manifest["parquet_revision"],
        "transport_manifest_sha256": transport_sha,
        "production_row_group_manifest_complete": complete,
        "file_weighting": "exact_row_counts" if complete else "hub_file_byte_size_proxy",
        "file_weighting_exact_for_rows": complete,
        "forced_distinct_file_first_pass": True,
        "row_group_weighting": "exact_row_count_within_resolved_file",
        "within_row_group_selection": "contiguous_head_slice",
        "within_row_group_start": 0,
        "within_row_group_seed_sensitive": False,
        "without_replacement_scope": "selected_file_row_group_unit",
        "row_level_nonzero_inclusion_for_all_rows": False,
        "row_level_equal_inclusion_proven": False,
        "global_exact_row_weighting": False,
        "representative_sampler": False,
        "allowed_claim": ALLOWED_SAMPLER_CLAIM,
        "seed": seed,
        "requested_units": units,
        "rows_per_unit": rows_per_unit,
        "planned_units": len(plan_units),
        "distinct_files": len({unit.file_path for unit in plan_units}),
        "distinct_row_groups": len({(u.file_path, u.row_group) for u in plan_units}),
        "planned_rows": sum(unit.rows_to_read for unit in plan_units),
        # Per-file topology pinned by exact path: a body read validates a file against ITS OWN
        # recorded schema hash, never against some other file's.
        "resolved_file_topology": {
            resolved[i].obj.path: {
                "rows": resolved[i].rows,
                "row_group_rows": list(resolved[i].row_group_rows or ()),
                "row_group_count": len(resolved[i].row_group_rows or ()),
                "schema_hash": resolved[i].schema_hash,
            }
            for i in sorted(resolved)
        },
        "units": [unit.to_json() for unit in plan_units],
    }
    plan["selection_plan_sha256"] = _sha256_bytes(_canonical_json_bytes(plan))
    return plan


def selection_plan_sha256(plan: Mapping[str, Any]) -> str:
    body = {key: value for key, value in plan.items() if key != "selection_plan_sha256"}
    return _sha256_bytes(_canonical_json_bytes(body))


_SHA256_HEX_RE = re.compile(r"\A[0-9a-f]{64}\Z")

# The frozen Option-B semantics. A plan that is internally checksum-consistent but disagrees with
# any of these is still rejected: a recomputed checksum must never buy a stale or forged contract.
_FROZEN_PLAN_SEMANTICS: dict[str, Any] = {
    "forced_distinct_file_first_pass": True,
    "row_group_weighting": "exact_row_count_within_resolved_file",
    "within_row_group_selection": "contiguous_head_slice",
    "within_row_group_start": 0,
    "within_row_group_seed_sensitive": False,
    "without_replacement_scope": "selected_file_row_group_unit",
    "row_level_nonzero_inclusion_for_all_rows": False,
    "row_level_equal_inclusion_proven": False,
    "global_exact_row_weighting": False,
    "representative_sampler": False,
}
# Frozen release-manifest constants. The generator and the validator both read these, so a legal
# manifest keeps byte-identical output while the validator gains a single source of truth.
ACCEPTED_COVERAGE_DERIVATION = "rebuilt from the committed documents.jsonl provenance fields"
TRANSPORT_COVERAGE_DERIVATION = "accumulated in the checkpoint at each unit open"
DEPRECATED_COVERAGE_NOTE = (
    "retained only for compatibility with readers of the pre-correction manifest. "
    "These two fields mirror accepted_document_coverage, i.e. coverage of the "
    "ACCEPTED DOCUMENTS, not of the transport reads. Use the two explicit blocks."
)
FROZEN_SAMPLER_STATIC_FIELDS: dict[str, Any] = {
    "row_level_nonzero_inclusion_for_all_rows": False,
    "row_level_equal_inclusion_proven": False,
    "global_exact_row_weighting": False,
    "representative_sampler": False,
    "allowed_claim": ALLOWED_SAMPLER_CLAIM,
}
FROZEN_RESOURCE_STATIC_FIELDS: dict[str, Any] = {
    "wire_bytes": None,
    "wire_bytes_measured": False,
    "metadata_bytes": None,
    "footer_bytes": None,
    "network_byte_cap_status": "partially_verified",
    "cap_enforcement": "pre_unit_soft_cap_may_overshoot_one_unit",
    "gpu_api_called": False,
}
FROZEN_GATE_C_SCOPE: dict[str, bool] = {
    "chat_conversion": False,
    "textual_document_separator": False,
    "tokenizer_counting": False,
    "bos_eos_inserted": False,
    "cross_source_near_dedup": False,
    "benchmark_decontamination": False,
    "reference_reserve_exclusion": False,
}
FROZEN_HARD_STOPS: dict[str, bool] = {
    "production_candidate_quota_authorized": False,
    "bulk_candidate_quota_started": False,
    "tokenizer_trained": False,
    "gate_r_started": False,
    "final_shards_built": False,
    "model_training_started": False,
}

_PLAN_BINDING_FIELDS = (
    "source_key",
    "dataset",
    "config",
    "split",
    "data_revision",
    "parquet_revision",
    "production_row_group_manifest_complete",
)


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GateCParquetError(f"{label} must be a JSON object, got {type(value).__name__}")
    return value


def _as_list(value: Any, label: str, *, allow_empty: bool = False) -> list:
    if not isinstance(value, list):
        raise GateCParquetError(f"{label} must be a list, got {type(value).__name__}")
    if not value and not allow_empty:
        raise GateCParquetError(f"{label} must not be empty")
    return value


def _require_field(mapping: Mapping[str, Any], field: str, label: str) -> Any:
    if field not in mapping:
        raise GateCParquetError(f"{label} is missing the required field {field!r}")
    return mapping[field]


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise GateCParquetError(f"{label} must be an integer >= {minimum}, got {value!r}")
    return value


def _require_number(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise GateCParquetError(f"{label} must be a finite number >= 0, got {value!r}")
    return float(value)


def _require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise GateCParquetError(f"{label} must be a non-empty string, got {value!r}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_HEX_RE.match(value):
        raise GateCParquetError(f"{label} must be a 64-hex lowercase SHA-256, got {value!r}")
    return value


def _require_exact_mapping(observed: Any, expected: Mapping[str, Any], label: str) -> None:
    """Exact key set, exact types and exact values -- an extra contradictory key never slips by."""
    mapping = _as_mapping(observed, label)
    if set(mapping) != set(expected):
        missing = sorted(set(expected) - set(mapping))
        extra = sorted(set(mapping) - set(expected))
        raise GateCParquetError(f"{label} key set is wrong: missing={missing} unexpected={extra}")
    for key, want in expected.items():
        _require_exact(mapping, key, want, label=label)


def _require_exact(
    plan: Mapping[str, Any], field: str, expected: Any, *, label: str = "selection plan"
) -> None:
    """Exact identity, not truthiness: ``1`` must not pass where ``True`` is required."""
    if field not in plan:
        raise GateCParquetError(f"{label} is missing the frozen field {field!r}")
    observed = plan[field]
    if type(observed) is not type(expected) or observed != expected:
        raise GateCParquetError(
            f"{label} field {field!r} is {observed!r}, expected exactly {expected!r}"
        )


_FOOTER_POLICIES = ("complete", "selected_files_only")


def validate_transport_manifest_v2(manifest: Mapping[str, Any]) -> str:
    """Single authority for the transport-manifest contract: checksum plus frozen identity.

    Every failure is a ``GateCParquetError``; a malformed or missing field never leaks a
    ``KeyError``/``TypeError``/``AttributeError`` to the caller or to the CLI.
    """
    L = "transport manifest"
    manifest = _as_mapping(manifest, L)

    # --- checksum first ---
    recorded = _require_field(manifest, "manifest_sha256", L)
    _require_sha256(recorded, f"{L}.manifest_sha256")
    computed = transport_manifest_sha256(manifest)
    if recorded != computed:
        raise GateCParquetError("transport manifest checksum mismatch")

    # --- frozen source identity ---
    source_key = _require_str(_require_field(manifest, "source_key", L), f"{L}.source_key")
    if source_key not in SOURCES or source_key not in PARQUET_BINDINGS:
        raise GateCParquetError(f"{L}.source_key {source_key!r} is not a frozen source binding")
    spec = SOURCES[source_key]
    binding = PARQUET_BINDINGS[source_key]
    for field, expected in (
        ("tool_schema_version", TOOL_SCHEMA_VERSION),
        ("spec_version", SPEC_VERSION),
        ("source_key", binding.source_key),
        ("dataset", spec.dataset),
        ("config", spec.dataset_config),
        ("split", spec.split),
        ("data_revision", spec.revision),
        ("parquet_repo_id", binding.repo_id),
        ("parquet_revision", binding.parquet_revision),
        ("parquet_is_export_branch", binding.is_export_branch),
        ("path_prefix", binding.path_prefix),
        ("transport", TRANSPORT),
    ):
        _require_exact(manifest, field, expected, label=L)

    # --- completeness fields: exact bool, never a truthy coercion ---
    _require_exact(manifest, "file_manifest_complete", True, label=L)
    complete = _require_field(manifest, "production_row_group_manifest_complete", L)
    if type(complete) is not bool:
        raise GateCParquetError(
            f"{L}.production_row_group_manifest_complete must be a bool, got "
            f"{type(complete).__name__} {complete!r}"
        )
    footer_policy = _require_str(_require_field(manifest, "footer_policy", L), f"{L}.footer_policy")
    if footer_policy not in _FOOTER_POLICIES:
        raise GateCParquetError(
            f"{L}.footer_policy must be one of {_FOOTER_POLICIES}, got {footer_policy!r}"
        )
    # The binding's own default footer policy is deliberately NOT enforced: a bounded run may
    # legitimately request an override. Only the internal equivalence is a contract.
    if (footer_policy == "complete") is not complete:
        raise GateCParquetError(
            f"{L}.footer_policy {footer_policy!r} contradicts "
            f"production_row_group_manifest_complete={complete!r}"
        )

    # --- files field: shape and uniqueness only; the plan validator owns per-file topology ---
    files = _as_list(_require_field(manifest, "files", L), f"{L}.files")
    seen: set[str] = set()
    for index, item in enumerate(files):
        entry = _as_mapping(item, f"{L}.files[{index}]")
        path = _require_str(
            _require_field(entry, "path", f"{L}.files[{index}]"), f"{L}.files[{index}].path"
        )
        if path in seen:
            raise GateCParquetError(f"{L}.files lists a duplicate path: {path}")
        seen.add(path)
    return computed


def verify_selection_plan(plan: Mapping[str, Any]) -> str:
    """Checksum only. Callers that act on a plan must use ``validate_selection_plan``."""
    plan = _as_mapping(plan, "selection plan")
    if plan.get("selection_plan_sha256") != selection_plan_sha256(plan):
        raise GateCParquetError("selection plan checksum mismatch")
    return str(plan["selection_plan_sha256"])


def validate_selection_plan(
    plan: Mapping[str, Any],
    transport_manifest: Mapping[str, Any],
    *,
    config: BuildV2Config | None = None,
) -> str:
    """The single authority every caller uses before acting on a selection plan.

    Runs before any checkpoint restore or mutation, before persisted-input writing, before the
    first body open, before publication, and inside ``verify_release_v2`` — so the build path and
    the release-verification path can never drift apart. Every failure is a ``GateCParquetError``;
    no ``assert`` is used, so the checks survive ``python -O``.
    """
    plan = _as_mapping(plan, "selection plan")
    transport_manifest = _as_mapping(transport_manifest, "transport manifest")
    plan_sha = verify_selection_plan(plan)
    transport_sha = verify_transport_manifest(transport_manifest)

    # --- 5.1 plan version and frozen Option-B semantics ---
    _require_exact(plan, "tool_schema_version", TOOL_SCHEMA_VERSION)
    _require_exact(plan, "sampler_version", SAMPLER_VERSION)
    for field, expected in _FROZEN_PLAN_SEMANTICS.items():
        _require_exact(plan, field, expected)
    _require_exact(plan, "allowed_claim", ALLOWED_SAMPLER_CLAIM)
    complete = transport_manifest["production_row_group_manifest_complete"]
    expected_weighting = "exact_row_counts" if complete else "hub_file_byte_size_proxy"
    _require_exact(plan, "file_weighting", expected_weighting)
    _require_exact(plan, "file_weighting_exact_for_rows", complete)

    # --- 5.2 binding to the exact transport manifest and to the build config ---
    _require_exact(plan, "transport_manifest_sha256", transport_sha)
    for field in _PLAN_BINDING_FIELDS:
        if field not in transport_manifest:
            raise GateCParquetError(f"transport manifest is missing {field!r}")
        _require_exact(plan, field, transport_manifest[field])
    if config is not None:
        for field, expected in (
            ("source_key", config.source.key),
            ("seed", config.seed),
            ("requested_units", config.units),
            ("rows_per_unit", config.rows_per_unit),
        ):
            _require_exact(plan, field, expected)

    manifest_files = {}
    for index, item in enumerate(
        _as_list(
            _require_field(transport_manifest, "files", "transport manifest"),
            "transport manifest files",
        )
    ):
        item = _as_mapping(item, f"transport manifest files[{index}]")
        path = _require_str(
            _require_field(item, "path", f"transport manifest files[{index}]"),
            f"transport manifest files[{index}].path",
        )
        if path in manifest_files:
            raise GateCParquetError(f"transport manifest lists a duplicate file path: {path}")
        manifest_files[path] = item

    # --- 5.3 internally derived fields, recomputed rather than trusted ---
    raw_units = _as_list(_require_field(plan, "units", "selection plan"), "selection plan units")
    units = []
    for index, item in enumerate(raw_units):
        item = _as_mapping(item, f"selection plan units[{index}]")
        try:
            units.append(PlanUnit.from_json(item))
        except GateCParquetError:
            raise
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise GateCParquetError(
                f"selection plan units[{index}] is malformed: {type(exc).__name__}: {exc}"
            ) from exc
    _require_exact(plan, "planned_units", len(units))
    requested = _require_field(plan, "requested_units", "selection plan")
    if type(requested) is not int or requested < 1:
        raise GateCParquetError(
            f"selection plan requested_units must be a positive integer, got {requested!r}"
        )
    if len(units) > requested:
        raise GateCParquetError(
            f"selection plan has {len(units)} units, more than the requested {requested}"
        )
    rows_per_unit = _require_field(plan, "rows_per_unit", "selection plan")
    if type(rows_per_unit) is not int or rows_per_unit < 1:
        raise GateCParquetError("selection plan rows_per_unit must be a positive integer")
    _require_exact(plan, "distinct_files", len({unit.file_path for unit in units}))
    _require_exact(plan, "distinct_row_groups", len({(u.file_path, u.row_group) for u in units}))
    _require_exact(plan, "planned_rows", sum(unit.rows_to_read for unit in units))
    seen_units: set[tuple[str, int]] = set()
    for index, unit in enumerate(units):
        if unit.order != index:
            raise GateCParquetError(f"selection plan unit {index} declares order {unit.order}")
        key = (unit.file_path, unit.row_group)
        if key in seen_units:
            raise GateCParquetError(f"selection plan repeats unit {key}")
        seen_units.add(key)
        if unit.rows_to_read < 1:
            raise GateCParquetError(f"selection plan unit {index} reads no rows")
        if unit.rows_to_read != min(rows_per_unit, unit.row_group_rows):
            raise GateCParquetError(
                f"selection plan unit {index} rows_to_read {unit.rows_to_read} != "
                f"min({rows_per_unit}, {unit.row_group_rows})"
            )
        entry = manifest_files.get(unit.file_path)
        if entry is None:
            raise GateCParquetError(
                f"selection plan unit {index} references {unit.file_path}, which is not in the "
                "transport manifest"
            )
        if entry["oid"] != unit.file_oid:
            raise GateCParquetError(
                f"selection plan unit {index} file_oid does not match the transport manifest for "
                f"{unit.file_path}"
            )

    # The declared distinct-file first pass must match the plan's actual shape.
    first_pass = units[: min(len(units), len(manifest_files))]
    if len({unit.file_path for unit in first_pass}) != len(first_pass):
        raise GateCParquetError(
            "selection plan declares a forced distinct-file first pass, but its leading units "
            "repeat a file"
        )

    # --- 5.4 exact-path resolved_file_topology ---
    topology = _as_mapping(
        _require_field(plan, "resolved_file_topology", "selection plan"),
        "selection plan resolved_file_topology",
    )
    selected_paths = {unit.file_path for unit in units}
    if set(topology) != selected_paths:
        missing = sorted(selected_paths - set(topology))
        extra = sorted(set(topology) - selected_paths)
        raise GateCParquetError(
            f"resolved_file_topology does not cover exactly the selected files: "
            f"missing={missing} unexpected={extra}"
        )
    for path in sorted(selected_paths):
        record = topology[path]
        if not isinstance(record, Mapping):
            raise GateCParquetError(f"resolved_file_topology[{path!r}] is not an object")
        rows = record.get("rows")
        group_rows = record.get("row_group_rows")
        group_count = record.get("row_group_count")
        schema_hash = record.get("schema_hash")
        if type(rows) is not int or rows < 1:
            raise GateCParquetError(f"resolved_file_topology[{path!r}].rows is invalid")
        if not isinstance(group_rows, list) or not group_rows:
            raise GateCParquetError(
                f"resolved_file_topology[{path!r}].row_group_rows is empty or invalid"
            )
        if any(type(n) is not int or n < 1 for n in group_rows):
            raise GateCParquetError(
                f"resolved_file_topology[{path!r}].row_group_rows has a non-positive entry"
            )
        if group_count != len(group_rows):
            raise GateCParquetError(
                f"resolved_file_topology[{path!r}].row_group_count {group_count} != "
                f"{len(group_rows)} row groups"
            )
        if rows != sum(group_rows):
            raise GateCParquetError(
                f"resolved_file_topology[{path!r}].rows {rows} != sum of row groups "
                f"{sum(group_rows)}"
            )
        if not isinstance(schema_hash, str) or not _SHA256_HEX_RE.match(schema_hash):
            raise GateCParquetError(
                f"resolved_file_topology[{path!r}].schema_hash must be a 64-hex SHA-256, got "
                f"{schema_hash!r}"
            )
        if complete:
            entry = manifest_files[path]
            if (
                entry.get("rows") != rows
                or list(entry.get("row_group_rows") or []) != group_rows
                or entry.get("row_group_count") != group_count
                or entry.get("schema_hash") != schema_hash
            ):
                raise GateCParquetError(
                    f"resolved_file_topology[{path!r}] disagrees with the complete transport "
                    "manifest entry for that exact file"
                )
    for index, unit in enumerate(units):
        record = topology[unit.file_path]
        group_rows = record["row_group_rows"]
        if not 0 <= unit.row_group < len(group_rows):
            raise GateCParquetError(
                f"selection plan unit {index} row group {unit.row_group} is out of range for "
                f"{unit.file_path}"
            )
        if unit.row_group_rows != group_rows[unit.row_group]:
            raise GateCParquetError(
                f"selection plan unit {index} row_group_rows {unit.row_group_rows} != topology "
                f"{group_rows[unit.row_group]}"
            )
        if unit.file_row_offset != sum(group_rows[: unit.row_group]):
            raise GateCParquetError(
                f"selection plan unit {index} file_row_offset {unit.file_row_offset} != "
                f"{sum(group_rows[: unit.row_group])}"
            )
    return plan_sha


def expected_schema_hash_for(plan: Mapping[str, Any], file_path: str) -> str:
    """Resolve the validated, non-null expected schema hash for one exact file path."""
    record = (plan.get("resolved_file_topology") or {}).get(file_path)
    schema_hash = (record or {}).get("schema_hash")
    if not isinstance(schema_hash, str) or not _SHA256_HEX_RE.match(schema_hash):
        raise GateCParquetError(
            f"no validated schema hash is pinned for {file_path}; refusing to read its body"
        )
    return schema_hash


# --------------------------------------------------------------------------------------
# Global row identity
# --------------------------------------------------------------------------------------


def source_global_row_index(
    manifest: Mapping[str, Any], file_path: str, row_in_file: int
) -> int | None:
    """Stable source-global row index under the canonical (sorted-path) file order.

    Returns ``None`` when the row-group inventory is incomplete, because a global index cannot be
    computed exactly without every preceding file's row count.  Guessing one is forbidden.
    """
    offset = 0
    for item in manifest["files"]:
        if item["path"] == file_path:
            return offset + row_in_file
        rows = item.get("rows")
        if rows is None:
            return None
        offset += int(rows)
    raise GateCParquetError(f"file is not in the transport manifest: {file_path}")


# --------------------------------------------------------------------------------------
# Checkpoint v2
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BuildV2Config:
    source: SourceSpec
    output_dir: Path
    work_dir: Path
    target_documents: int
    max_scanned: int
    max_transfer_bytes: int
    max_wall_seconds: float
    seed: int
    units: int = DEFAULT_UNITS
    rows_per_unit: int = 128
    stop_after_documents: int | None = None
    checkpoint_every: int = 32
    format_quota: Mapping[str, int] | None = None
    forum_deny: tuple[str, ...] = ()
    forum_cap: int | None = None
    max_scanned_rationale: str = ""


def _run_fingerprint_v2(config: BuildV2Config, transport_sha: str, plan_sha: str) -> str:
    return _sha256_bytes(
        _canonical_json_bytes({
            "tool_schema_version": TOOL_SCHEMA_VERSION,
            "sampler_version": SAMPLER_VERSION,
            "spec_version": SPEC_VERSION,
            "source_key": config.source.key,
            "source_filter_spec": {
                "body_path": config.source.body_path,
                "min_bytes": config.source.min_bytes,
                "max_bytes": config.source.max_bytes,
                "required_schema": config.source.required_schema_map,
            },
            "transport_manifest_sha256": transport_sha,
            "selection_plan_sha256": plan_sha,
            "seed": config.seed,
            "units": config.units,
            "rows_per_unit": config.rows_per_unit,
            "target_documents": config.target_documents,
            "max_scanned": config.max_scanned,
            "format_quota": dict(config.format_quota or {}),
            "forum_deny": list(config.forum_deny),
            "forum_cap": config.forum_cap,
            "output_dir": str(_ensure_under_workspace(config.output_dir)),
        })
    )


def _new_checkpoint_v2(fingerprint: str, transport_sha: str, plan_sha: str) -> dict[str, Any]:
    return {
        "tool_schema_version": TOOL_SCHEMA_VERSION,
        "sampler_version": SAMPLER_VERSION,
        "run_fingerprint": fingerprint,
        "transport_manifest_sha256": transport_sha,
        "selection_plan_sha256": plan_sha,
        "unit_cursor": 0,
        "unit_row_cursor": 0,
        "visited_global_rows": 0,
        "counters": BuildCounters().to_json(),
        "seen_record_ids": [],
        "seen_text_sha256": [],
        "seen_global_row_index": [],
        "documents_sha256": _sha256_bytes(b""),
        "documents_bytes": 0,
        "resume_count": 0,
        "exhausted": False,
        "accepted_by_format": {},
        "accepted_by_forum": {},
        # Correction B: transport-read coverage must survive a resume, so it is accumulated in the
        # checkpoint rather than in a set that is re-initialised on every invocation.
        "transport_files_opened": [],
        "transport_row_groups_opened": [],
        "planned_units_attempted": 0,
        "planned_units_completed": 0,
        # Correction F: resource accounting is cumulative across invocations.
        "cumulative_body_reader_exposed_bytes": 0,
        "cumulative_body_reader_read_calls": 0,
        "cumulative_build_wall_seconds": 0.0,
    }


def _checkpoint_payload(state: Mapping[str, Any]) -> bytes:
    return _canonical_json_bytes({k: v for k, v in state.items() if k != "checksum"})


def _write_checkpoint(path: Path, state: Mapping[str, Any]) -> None:
    body = {key: value for key, value in state.items() if key != "checksum"}
    body["checksum"] = _sha256_bytes(_checkpoint_payload(body))
    temporary = path.parent / f".{path.name}.tmp"
    if temporary.exists():
        temporary.unlink()
    _write_new_file(temporary, json.dumps(body, indent=2, sort_keys=True).encode() + b"\n")
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _read_checkpoint(path: Path, fingerprint: str, transport_sha: str, plan_sha: str):
    raw = path.read_bytes()
    try:
        state = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateCParquetError("checkpoint is corrupted: not strict UTF-8 JSON") from exc
    if not isinstance(state, dict):
        raise GateCParquetError("checkpoint is corrupted: not an object")
    version = state.get("tool_schema_version")
    if version != TOOL_SCHEMA_VERSION:
        raise GateCParquetError(
            f"refusing to resume a {version!r} checkpoint with {TOOL_SCHEMA_VERSION!r}; "
            "there is no automatic migration"
        )
    if state.get("checksum") != _sha256_bytes(_checkpoint_payload(state)):
        raise GateCParquetError("checkpoint checksum mismatch; refusing to resume")
    if state.get("sampler_version") != SAMPLER_VERSION:
        raise GateCParquetError("checkpoint sampler version mismatch; refusing to resume")
    if state.get("transport_manifest_sha256") != transport_sha:
        raise GateCParquetError("transport manifest drifted; refusing to resume")
    if state.get("selection_plan_sha256") != plan_sha:
        raise GateCParquetError("selection plan drifted; refusing to resume")
    if state.get("run_fingerprint") != fingerprint:
        raise GateCParquetError("checkpoint run fingerprint mismatch; refusing to resume")
    return state


def _persist_and_verify_inputs(
    work_dir: Path, transport_manifest: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, str]:
    """Persist the exact transport manifest and selection plan BEFORE any body byte is read.

    First run writes them atomically with a ``.sha256`` sidecar. A resume re-reads them and
    requires an exact byte match, so a build can never read bodies against inputs that differ from
    the ones already on disk, and an existing file is never silently overwritten.
    """
    recorded: dict[str, str] = {}
    for name, value in (
        (TRANSPORT_MANIFEST_NAME, transport_manifest),
        (SELECTION_PLAN_NAME, plan),
    ):
        payload = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
        digest = _sha256_bytes(payload)
        target = work_dir / name
        sidecar = work_dir / f"{name}.sha256"
        if target.exists():
            existing = target.read_bytes()
            if existing != payload:
                raise GateCParquetError(
                    f"persisted {name} differs from the inputs of this build; refusing to overwrite"
                )
            if not sidecar.exists() or sidecar.read_text().split("  ")[0] != digest:
                raise GateCParquetError(f"persisted {name}.sha256 does not match {name}")
        else:
            temporary = work_dir / f".{name}.tmp"
            if temporary.exists():
                temporary.unlink()
            _write_new_file(temporary, payload)
            os.replace(temporary, target)
            side_tmp = work_dir / f".{name}.sha256.tmp"
            if side_tmp.exists():
                side_tmp.unlink()
            _write_new_file(side_tmp, f"{digest}  {name}\n".encode())
            os.replace(side_tmp, sidecar)
            _fsync_directory(work_dir)
        recorded[name] = digest
    return recorded


def _restore_documents(path: Path, state: Mapping[str, Any]) -> None:
    expected_bytes = int(state["documents_bytes"])
    expected_sha = str(state["documents_sha256"])
    if not path.exists():
        if expected_bytes:
            raise GateCParquetError("checkpoint references documents but the file is missing")
        _write_new_file(path, b"")
        return
    actual = path.read_bytes()
    if len(actual) < expected_bytes:
        raise GateCParquetError("document file is shorter than its committed checkpoint prefix")
    if _sha256_bytes(actual[:expected_bytes]) != expected_sha:
        raise GateCParquetError("document prefix does not match the committed checkpoint hash")
    if len(actual) != expected_bytes:
        with open(path, "r+b") as handle:
            handle.truncate(expected_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)


# --------------------------------------------------------------------------------------
# Parquet body reads
# --------------------------------------------------------------------------------------


def read_unit_rows(
    store: ObjectStore,
    binding: ParquetBinding,
    unit: PlanUnit,
    columns: Sequence[str],
    *,
    expected_schema_hash: str,
    start_row: int = 0,
) -> tuple[list[dict[str, Any]], int, dict[str, str]]:
    """Read one planned unit and return ``(rows, bytes_fetched, arrow_schema_map)``."""
    import pyarrow.parquet as pq

    with store.open(binding, unit.file_path) as reader:
        parquet = pq.ParquetFile(reader, pre_buffer=False, buffer_size=READ_BLOCK_BYTES)
        metadata = parquet.metadata
        if unit.row_group >= metadata.num_row_groups:
            raise GateCParquetError(f"planned row group is missing: {unit.file_path}")
        if metadata.row_group(unit.row_group).num_rows != unit.row_group_rows:
            raise GateCParquetError(
                f"planned row-group row count drifted: {unit.file_path}#{unit.row_group}"
            )
        schema_hash = _parquet_schema_hash(metadata)
        # No fail-open path: a missing or malformed pin is rejected before any comparison.
        if not isinstance(expected_schema_hash, str) or not _SHA256_HEX_RE.match(
            expected_schema_hash
        ):
            raise GateCParquetError(
                f"refusing to read {unit.file_path}: expected schema hash is not a pinned SHA-256"
            )
        if schema_hash != expected_schema_hash:
            raise GateCParquetError(
                f"Parquet schema drift in {unit.file_path}: expected {expected_schema_hash}, "
                f"observed {schema_hash}"
            )
        arrow_schema = metadata.schema.to_arrow_schema()
        rows: list[dict[str, Any]] = []
        wanted = max(0, unit.rows_to_read - start_row)
        seen = 0
        for batch in parquet.iter_batches(
            batch_size=min(256, max(1, wanted)),
            row_groups=[unit.row_group],
            columns=list(columns),
        ):
            for record in batch.to_pylist():
                if seen >= start_row + wanted:
                    break
                if seen >= start_row:
                    rows.append(record)
                seen += 1
            if seen >= start_row + wanted:
                break
        fetched = reader.bytes_read
    schema_map = {name: str(arrow_schema.field(name).type) for name in arrow_schema.names}
    return rows, fetched, schema_map


# --------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------


def reconstruct_accepted_document_coverage(
    documents_path: Path, total_rows: int | None
) -> dict[str, Any]:
    """Rebuild accepted-document coverage deterministically from the committed JSONL.

    This is a fact about the published records, not about this invocation's transport activity, so
    it stays correct after any number of resumes.
    """
    files: Counter = Counter()
    row_groups: Counter = Counter()
    positions: list[int] = []
    count = 0
    if documents_path.exists():
        for line in documents_path.read_bytes().split(b"\n"):
            if not line:
                continue
            record = json.loads(line.decode("utf-8", errors="strict"))
            provenance = record["provenance"]
            files[provenance["parquet_file"]] += 1
            row_groups[f"{provenance['parquet_file']}#{provenance['row_group']}"] += 1
            position = provenance.get("source_global_row_index")
            if position is not None:
                positions.append(position)
            count += 1
    deciles = None
    if positions and total_rows:
        buckets = Counter(min(9, (p * 10) // total_rows) for p in positions)
        deciles = {str(k): buckets[k] for k in sorted(buckets)}
    return {
        "accepted_documents": count,
        "accepted_file_count": len(files),
        "accepted_row_group_count": len(row_groups),
        "accepted_files": sorted(files),
        "accepted_row_groups": sorted(row_groups),
        "accepted_file_histogram": dict(files.most_common()),
        "accepted_row_group_histogram": dict(row_groups.most_common()),
        "global_row_min": min(positions) if positions else None,
        "global_row_max": max(positions) if positions else None,
        "global_row_decile_histogram": deciles,
        "global_row_index_available": bool(positions) and len(positions) == count,
    }


def _validate_v2_config(config: BuildV2Config) -> None:
    if not 1 <= config.target_documents <= MAX_ACCEPTED_DOCUMENTS:
        raise GateCParquetError(f"target_documents must be in [1, {MAX_ACCEPTED_DOCUMENTS}]")
    if not config.target_documents <= config.max_scanned <= MAX_SCANNED_RECORDS:
        raise GateCParquetError(f"max_scanned must be in [target_documents, {MAX_SCANNED_RECORDS}]")
    if config.max_transfer_bytes < 1:
        raise GateCParquetError("max_transfer_bytes must be positive")
    if not math.isfinite(config.max_wall_seconds) or config.max_wall_seconds <= 0:
        raise GateCParquetError("max_wall_seconds must be positive")
    if config.stop_after_documents is not None and not (
        1 <= config.stop_after_documents <= config.target_documents
    ):
        raise GateCParquetError("stop_after_documents must be in [1, target_documents]")
    for path in (config.output_dir, config.work_dir):
        _ensure_git_ignored(path)


def build_c0v2(
    config: BuildV2Config,
    transport_manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
    store: ObjectStore,
    *,
    clock: Callable[[], float] = time.perf_counter,
    unit_hook: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Traverse the frozen selection plan, filter with the frozen v1 filters, publish atomically."""
    _validate_v2_config(config)
    output_dir = _ensure_under_workspace(config.output_dir)
    if output_dir.exists():
        raise GateCParquetError(f"refusing to overwrite published output: {output_dir}")
    work_dir = _ensure_under_workspace(config.work_dir)

    # Strict, single-authority validation happens BEFORE the work directory is created, before any
    # checkpoint restore or mutation, before persisted-input writing, and before the first body
    # open. An invalid plan therefore opens zero bodies and persists nothing load-bearing.
    transport_sha = verify_transport_manifest(transport_manifest)
    plan_sha = validate_selection_plan(plan, transport_manifest, config=config)

    work_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _run_fingerprint_v2(config, transport_sha, plan_sha)
    checkpoint_path = work_dir / CHECKPOINT_NAME
    documents_path = work_dir / DOCUMENTS_NAME
    resumed = checkpoint_path.exists()
    if resumed:
        state = _read_checkpoint(checkpoint_path, fingerprint, transport_sha, plan_sha)
        state["resume_count"] = int(state["resume_count"]) + 1
    else:
        state = _new_checkpoint_v2(fingerprint, transport_sha, plan_sha)
    _restore_documents(documents_path, state)

    counters = BuildCounters.from_json(state["counters"])
    seen_record_ids = set(state["seen_record_ids"])
    seen_text_hashes = set(state["seen_text_sha256"])
    seen_global_rows = set(state["seen_global_row_index"])
    accepted_by_format: Counter = Counter(state.get("accepted_by_format") or {})
    accepted_by_forum: Counter = Counter(state.get("accepted_by_forum") or {})
    documents_digest = hashlib.sha256(documents_path.read_bytes())
    documents_bytes = int(state["documents_bytes"])
    exposed_bytes = int(state["cumulative_body_reader_exposed_bytes"])
    read_calls = int(state["cumulative_body_reader_read_calls"])
    prior_wall = float(state["cumulative_build_wall_seconds"])
    unit_cursor = int(state["unit_cursor"])
    unit_row_cursor = int(state["unit_row_cursor"])

    units = [PlanUnit.from_json(item) for item in plan["units"]]
    binding = PARQUET_BINDINGS[config.source.key]
    columns = SOURCE_COLUMNS[config.source.key]
    # Correction D: inputs are persisted and verified before the first body open.
    persisted_inputs = _persist_and_verify_inputs(work_dir, transport_manifest, plan)

    # Correction B: transport-read coverage accumulates across resumes.
    transport_files: set[str] = set(state["transport_files_opened"])
    transport_row_groups: set[str] = set(state["transport_row_groups_opened"])
    units_attempted = int(state["planned_units_attempted"])
    units_completed = int(state["planned_units_completed"])
    invocation_exposed_bytes = 0
    invocation_read_calls = 0
    start = clock()
    handle = open(documents_path, "ab")

    def commit(is_exhausted: bool) -> None:
        handle.flush()
        os.fsync(handle.fileno())
        state["unit_cursor"] = unit_cursor
        state["unit_row_cursor"] = unit_row_cursor
        state["counters"] = counters.to_json()
        state["seen_record_ids"] = sorted(seen_record_ids)
        state["seen_text_sha256"] = sorted(seen_text_hashes)
        state["seen_global_row_index"] = sorted(seen_global_rows)
        state["documents_sha256"] = documents_digest.hexdigest()
        state["documents_bytes"] = documents_bytes
        state["exhausted"] = is_exhausted
        state["accepted_by_format"] = dict(accepted_by_format)
        state["accepted_by_forum"] = dict(accepted_by_forum)
        state["transport_files_opened"] = sorted(transport_files)
        state["transport_row_groups_opened"] = sorted(transport_row_groups)
        state["planned_units_attempted"] = units_attempted
        state["planned_units_completed"] = units_completed
        state["cumulative_body_reader_exposed_bytes"] = exposed_bytes
        state["cumulative_body_reader_read_calls"] = read_calls
        state["cumulative_build_wall_seconds"] = round(prior_wall + (clock() - start), 3)
        state["visited_global_rows"] = len(seen_global_rows)
        _write_checkpoint(checkpoint_path, state)

    try:
        if counters.accepted >= config.target_documents:
            stop_reason = "target_reached"
        elif unit_cursor >= len(units):
            stop_reason = "plan_exhausted"
        else:
            stop_reason = "plan_exhausted"
            since_checkpoint = 0
            while unit_cursor < len(units):
                # Checked at the unit boundary only: a single unit may overshoot the cap.
                if exposed_bytes > config.max_transfer_bytes:
                    stop_reason = "byte_cap"
                    break
                if clock() - start > config.max_wall_seconds:
                    stop_reason = "time_cap"
                    break
                unit = units[unit_cursor]
                if unit_hook is not None:
                    unit_hook(unit_cursor)
                units_attempted += 1
                transport_files.add(unit.file_path)
                transport_row_groups.add(f"{unit.file_path}#{unit.row_group}")
                rows, fetched, _schema_map = read_unit_rows(
                    store,
                    binding,
                    unit,
                    columns,
                    expected_schema_hash=expected_schema_hash_for(plan, unit.file_path),
                    start_row=unit_row_cursor,
                )
                exposed_bytes += fetched
                invocation_exposed_bytes += fetched
                read_calls += 1
                invocation_read_calls += 1
                counters.response_bytes += fetched
                counters.request_count += 1

                stopped_inside_unit = False
                for offset, row in enumerate(rows):
                    row_in_unit = unit_row_cursor + offset
                    row_in_file = unit.file_row_offset + row_in_unit
                    global_index = source_global_row_index(
                        transport_manifest, unit.file_path, row_in_file
                    )
                    counters.scanned += 1
                    decision = evaluate_row(row, config.source)
                    if decision.accepted:
                        decision = _apply_quotas(
                            decision, config, accepted_by_format, accepted_by_forum
                        )
                    if not decision.accepted:
                        counters.rejected += 1
                        counters.rejections[decision.reason or "unspecified"] += 1
                    else:
                        natural_id = _natural_identity(
                            row, config.source, global_index, unit, row_in_file
                        )
                        record_id = _source_record_id(config.source.key, natural_id)
                        text = decision.text or ""
                        text_bytes = text.encode("utf-8")
                        text_sha = _sha256_bytes(text_bytes)
                        if record_id in seen_record_ids:
                            counters.rejected += 1
                            counters.rejections["duplicate_source_record_id"] += 1
                        elif text_sha in seen_text_hashes:
                            counters.rejected += 1
                            counters.rejections["duplicate_text_sha256"] += 1
                        elif global_index is not None and global_index in seen_global_rows:
                            counters.rejected += 1
                            counters.rejections["duplicate_global_row_index"] += 1
                        else:
                            seen_record_ids.add(record_id)
                            seen_text_hashes.add(text_sha)
                            if global_index is not None:
                                seen_global_rows.add(global_index)
                            line = canonical_jsonl_record_bytes({
                                "source_key": config.source.key,
                                "source_record_id": record_id,
                                "natural_id": natural_id,
                                "text": text,
                                "text_sha256": text_sha,
                                "text_bytes": len(text_bytes),
                                "metadata": dict(decision.metadata),
                                "provenance": {
                                    "dataset": config.source.dataset,
                                    "config": config.source.dataset_config,
                                    "split": config.source.split,
                                    "revision": config.source.revision,
                                    "license": config.source.license,
                                    "transport": TRANSPORT,
                                    "parquet_repo_id": binding.repo_id,
                                    "parquet_revision": binding.parquet_revision,
                                    "parquet_file": unit.file_path,
                                    "parquet_file_oid": unit.file_oid,
                                    "row_group": unit.row_group,
                                    "row_in_group": row_in_unit,
                                    "row_in_file": row_in_file,
                                    "source_global_row_index": global_index,
                                    "transport_manifest_sha256": transport_sha,
                                    "selection_plan_sha256": plan_sha,
                                },
                            })
                            handle.write(line)
                            documents_digest.update(line)
                            documents_bytes += len(line)
                            counters.accepted += 1
                            counters.accepted_text_bytes += len(text_bytes)
                            for name in decision.diagnostics:
                                counters.diagnostics[name] += 1
                            document_format = decision.metadata.get("format")
                            if isinstance(document_format, str):
                                accepted_by_format[document_format] += 1
                            forum = decision.metadata.get("metadata.forum")
                            if isinstance(forum, str) and forum:
                                accepted_by_forum[forum] += 1
                            since_checkpoint += 1

                    if counters.accepted >= config.target_documents:
                        stop_reason = "target_reached"
                        unit_row_cursor = row_in_unit + 1
                        stopped_inside_unit = True
                        break
                    if config.stop_after_documents is not None and (
                        counters.accepted >= config.stop_after_documents
                    ):
                        stop_reason = "stop_after_documents"
                        unit_row_cursor = row_in_unit + 1
                        stopped_inside_unit = True
                        break
                    if counters.scanned >= config.max_scanned:
                        stop_reason = "scan_cap"
                        unit_row_cursor = row_in_unit + 1
                        stopped_inside_unit = True
                        break

                if stopped_inside_unit:
                    break
                units_completed += 1
                unit_cursor += 1
                unit_row_cursor = 0
                if since_checkpoint >= config.checkpoint_every:
                    commit(False)
                    since_checkpoint = 0
        exhausted = stop_reason == "plan_exhausted"
        commit(exhausted)
    finally:
        handle.close()

    if stop_reason == "stop_after_documents":
        return {
            "published": False,
            "stop_reason": stop_reason,
            "work_dir": str(work_dir),
            "accepted": counters.accepted,
            "scanned": counters.scanned,
            "body_reader_exposed_bytes": invocation_exposed_bytes,
            "cumulative_body_reader_exposed_bytes": exposed_bytes,
            "unit_cursor": unit_cursor,
            "unit_row_cursor": unit_row_cursor,
            "resume_count": int(state["resume_count"]),
            "resumed": resumed,
            "documents_sha256": documents_digest.hexdigest(),
        }

    wall_seconds = clock() - start
    manifest = _make_manifest_v2(
        config,
        state,
        counters,
        transport_manifest=transport_manifest,
        plan=plan,
        fingerprint=fingerprint,
        stop_reason=stop_reason,
        wall_seconds=wall_seconds,
        documents_sha256=documents_digest.hexdigest(),
        documents_bytes=documents_bytes,
        documents_path=documents_path,
        transport_files=transport_files,
        transport_row_groups=transport_row_groups,
        units_attempted=units_attempted,
        units_completed=units_completed,
        exposed_bytes=exposed_bytes,
        read_calls=read_calls,
        invocation_exposed_bytes=invocation_exposed_bytes,
        invocation_read_calls=invocation_read_calls,
        prior_wall=prior_wall,
        persisted_inputs=persisted_inputs,
        accepted_by_format=accepted_by_format,
        accepted_by_forum=accepted_by_forum,
    )

    staging = output_dir.parent / f".{output_dir.name}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copyfile(documents_path, staging / DOCUMENTS_NAME)
    _write_new_file(
        staging / MANIFEST_NAME, json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    )
    _write_new_file(
        staging / TRANSPORT_MANIFEST_NAME,
        json.dumps(transport_manifest, indent=2, sort_keys=True).encode() + b"\n",
    )
    _write_new_file(
        staging / SELECTION_PLAN_NAME,
        json.dumps(plan, indent=2, sort_keys=True).encode() + b"\n",
    )
    _write_new_file(
        staging / CHECKSUMS_NAME,
        "".join(
            f"{_sha256_bytes((staging / name).read_bytes())}  {name}\n"
            for name in (
                DOCUMENTS_NAME,
                MANIFEST_NAME,
                TRANSPORT_MANIFEST_NAME,
                SELECTION_PLAN_NAME,
            )
        ).encode(),
    )
    _publish_directory(staging, output_dir)

    published_sha = _sha256_bytes((output_dir / DOCUMENTS_NAME).read_bytes())
    if published_sha != manifest["documents_sha256"]:
        raise GateCParquetError("published document file does not match its manifest checksum")

    return {
        "published": True,
        "output_dir": str(output_dir),
        "stop_reason": stop_reason,
        "release_kind": manifest["release_kind"],
        "promotion_eligible": manifest["promotion_eligible"],
        "accepted": counters.accepted,
        "scanned": counters.scanned,
        "rejected": counters.rejected,
        "body_reader_exposed_bytes": invocation_exposed_bytes,
        "cumulative_body_reader_exposed_bytes": exposed_bytes,
        "wall_seconds": round(wall_seconds, 3),
        "resume_count": int(state["resume_count"]),
        "resumed": resumed,
        "transport_files_opened": len(transport_files),
        "transport_row_groups_opened": len(transport_row_groups),
        "documents_sha256": published_sha,
        "manifest_sha256": _sha256_bytes((output_dir / MANIFEST_NAME).read_bytes()),
        "rejections": dict(sorted(counters.rejections.items())),
    }


def _apply_quotas(decision, config, accepted_by_format, accepted_by_forum):
    """Apply the C0-only stratification quotas; never a quality judgement."""
    if config.format_quota:
        document_format = decision.metadata.get("format")
        if isinstance(document_format, str):
            allowed = config.format_quota.get(document_format)
            if allowed is None:
                return Decision(accepted=False, reason="format_not_in_c0_quota")
            if accepted_by_format[document_format] >= allowed:
                return Decision(accepted=False, reason="format_quota_full")
    forum = decision.metadata.get("metadata.forum")
    if isinstance(forum, str) and forum:
        if forum in config.forum_deny:
            return Decision(accepted=False, reason="forum_denied_c0_policy")
        if config.forum_cap is not None and accepted_by_forum[forum] >= config.forum_cap:
            return Decision(accepted=False, reason="forum_cap_full")
    return decision


def _natural_identity(
    row: Mapping[str, Any],
    source: SourceSpec,
    global_index: int | None,
    unit: PlanUnit,
    row_in_file: int,
) -> str:
    """Stable identity: the source's own ID when it has one, else an exact positional identity."""
    if source.natural_id_path:
        value = row.get(source.natural_id_path)
        if isinstance(value, str) and value:
            return value
        if type(value) is int:
            return str(value)
        raise GateCParquetError(f"source record is missing its natural id: {source.key}")
    if global_index is not None:
        return f"grow:{global_index}"
    return f"file:{unit.file_path}:row:{row_in_file}"


def _make_manifest_v2(
    config: BuildV2Config,
    state: Mapping[str, Any],
    counters: BuildCounters,
    *,
    transport_manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
    fingerprint: str,
    stop_reason: str,
    wall_seconds: float,
    documents_sha256: str,
    documents_bytes: int,
    documents_path: Path,
    transport_files: set[str],
    transport_row_groups: set[str],
    units_attempted: int,
    units_completed: int,
    exposed_bytes: int,
    read_calls: int,
    invocation_exposed_bytes: int,
    invocation_read_calls: int,
    prior_wall: float,
    persisted_inputs: Mapping[str, str],
    accepted_by_format: Mapping[str, int],
    accepted_by_forum: Mapping[str, int],
) -> dict[str, Any]:
    source = config.source
    accepted = counters.accepted
    diagnostics = dict(counters.diagnostics)
    total_rows = transport_manifest.get("total_rows")
    accepted_cov = reconstruct_accepted_document_coverage(documents_path, total_rows)
    return {
        "tool_schema_version": TOOL_SCHEMA_VERSION,
        "sampler_version": SAMPLER_VERSION,
        "spec_version": SPEC_VERSION,
        "release_kind": "c0_diagnostic",
        "promotion_eligible": False,
        "promotion_eligible_rationale": (
            "every release this module can produce is a bounded C0 diagnostic; production "
            "candidate mode is not implemented and reaching the target does not make it promotable"
        ),
        "run_fingerprint": fingerprint,
        "source": {
            "key": source.key,
            "dataset": source.dataset,
            "config": source.dataset_config,
            "split": source.split,
            "revision": source.revision,
            "license": source.license,
            "body_path": source.body_path,
            "min_bytes": source.min_bytes,
            "max_bytes": source.max_bytes,
            "transport": TRANSPORT,
            "parquet_repo_id": transport_manifest["parquet_repo_id"],
            "parquet_revision": transport_manifest["parquet_revision"],
            "parquet_is_export_branch": transport_manifest["parquet_is_export_branch"],
        },
        "transport_manifest_sha256": transport_manifest["manifest_sha256"],
        "selection_plan_sha256": plan["selection_plan_sha256"],
        "production_row_group_manifest_complete": transport_manifest[
            "production_row_group_manifest_complete"
        ],
        "sampler": {
            "version": SAMPLER_VERSION,
            "seed": config.seed,
            "units_planned": plan["planned_units"],
            "rows_per_unit": config.rows_per_unit,
            "file_weighting": plan["file_weighting"],
            "file_weighting_exact_for_rows": plan["file_weighting_exact_for_rows"],
            "within_row_group_selection": plan["within_row_group_selection"],
            "within_row_group_start": plan["within_row_group_start"],
            "within_row_group_seed_sensitive": plan["within_row_group_seed_sensitive"],
            "without_replacement_scope": plan["without_replacement_scope"],
            **FROZEN_SAMPLER_STATIC_FIELDS,
        },
        "caps": {
            "target_documents": config.target_documents,
            "max_scanned": config.max_scanned,
            "max_transfer_bytes": config.max_transfer_bytes,
            "max_wall_seconds": config.max_wall_seconds,
            "max_scanned_rationale": config.max_scanned_rationale,
        },
        "c0_policy": {
            "format_quota": dict(config.format_quota or {}),
            "forum_deny": list(config.forum_deny),
            "forum_cap": config.forum_cap,
        },
        "accounting": counters.to_json(),
        "yield_rate": (accepted / counters.scanned) if counters.scanned else None,
        "diagnostic_shares": {
            "structured_abstract_share": (
                diagnostics.get("structured_abstract", 0) / accepted if accepted else None
            ),
            "story_family_share": (
                diagnostics.get("format_family_story", 0) / accepted if accepted else None
            ),
            "declarative_numbered_list_share": (
                diagnostics.get("declarative_numbered_list", 0) / accepted if accepted else None
            ),
        },
        "accepted_by_format": dict(sorted(accepted_by_format.items())),
        "accepted_by_forum": dict(sorted(accepted_by_forum.items())),
        # Correction B: two separately named coverages. The accepted-document coverage is
        # deterministically rebuilt from the committed documents.jsonl, so it never depends on a
        # per-invocation set. The transport-read coverage accumulates across resumes and can be
        # larger, because a unit may be opened and yield no accepted document.
        "accepted_document_coverage": {
            **accepted_cov,
            "source_file_count": transport_manifest["file_count"],
            "source_total_rows": total_rows,
            "derivation": ACCEPTED_COVERAGE_DERIVATION,
        },
        "transport_read_coverage": {
            "files_opened": len(transport_files),
            "row_groups_opened": len(transport_row_groups),
            "files_opened_list": sorted(transport_files),
            "row_groups_opened_list": sorted(transport_row_groups),
            "planned_units_attempted": units_attempted,
            "planned_units_completed": units_completed,
            "accumulates_across_resumes": True,
            "derivation": TRANSPORT_COVERAGE_DERIVATION,
        },
        "coverage": {
            "deprecated": True,
            "deprecated_note": DEPRECATED_COVERAGE_NOTE,
            "files_touched": accepted_cov["accepted_file_count"],
            "row_groups_touched": accepted_cov["accepted_row_group_count"],
        },
        "stop_reason": stop_reason,
        "exhausted": bool(state["exhausted"]),
        "resume_count": int(state["resume_count"]),
        "current_invocation_wall_seconds": round(wall_seconds, 3),
        "cumulative_build_wall_seconds": round(prior_wall + wall_seconds, 3),
        # Correction F: honest resource accounting. Wire bytes are NOT measured in this build.
        "resource_accounting": {
            "body_reader_exposed_bytes": invocation_exposed_bytes,
            "body_reader_read_calls": invocation_read_calls,
            "cumulative_body_reader_exposed_bytes": exposed_bytes,
            "cumulative_body_reader_read_calls": read_calls,
            **FROZEN_RESOURCE_STATIC_FIELDS,
        },
        "persisted_inputs_before_first_body_read": dict(persisted_inputs),
        "documents_file": DOCUMENTS_NAME,
        "documents_sha256": documents_sha256,
        "documents_bytes": documents_bytes,
        "gate_c_scope": dict(FROZEN_GATE_C_SCOPE),
        "hard_stops": dict(FROZEN_HARD_STOPS),
    }


def _read_release_json(output_dir: Path, name: str) -> tuple[Mapping[str, Any], bytes]:
    """Read one required release JSON file, converting every input-shape failure to GateCError."""
    path = output_dir / name
    if not path.is_file():
        raise GateCParquetError(f"published release is missing {name}")
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateCParquetError(f"{name} is not strict UTF-8 JSON: {exc}") from exc
    return _as_mapping(value, name), payload


def validate_release_manifest_v2(
    manifest: Mapping[str, Any],
    *,
    output_dir: Path,
    transport_manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
    documents_bytes: bytes,
) -> dict[str, Any]:
    """The single authority for the release-manifest contract.

    It validates only what ``_make_manifest_v2`` already records, and only relationships the
    generator genuinely guarantees; it does not extend the release schema. Every failure is a
    ``GateCParquetError``.
    """
    manifest = _as_mapping(manifest, "release manifest")
    L = "release manifest"

    # --- 5.2 top-level contract ---
    _require_exact(manifest, "tool_schema_version", TOOL_SCHEMA_VERSION, label=L)
    _require_exact(manifest, "sampler_version", SAMPLER_VERSION, label=L)
    _require_exact(manifest, "spec_version", SPEC_VERSION, label=L)
    _require_exact(manifest, "release_kind", "c0_diagnostic", label=L)
    _require_exact(manifest, "promotion_eligible", False, label=L)
    _require_exact(manifest, "documents_file", DOCUMENTS_NAME, label=L)

    transport_sha = verify_transport_manifest(transport_manifest)
    plan_sha = validate_selection_plan(plan, transport_manifest)
    _require_exact(manifest, "transport_manifest_sha256", transport_sha, label=L)
    _require_exact(manifest, "selection_plan_sha256", plan_sha, label=L)
    _require_exact(
        manifest,
        "production_row_group_manifest_complete",
        transport_manifest["production_row_group_manifest_complete"],
        label=L,
    )

    # --- 5.3 source binding, cross-checked against every authority ---
    source = _as_mapping(_require_field(manifest, "source", L), f"{L}.source")
    key = _require_str(_require_field(source, "key", f"{L}.source"), f"{L}.source.key")
    if key not in SOURCES or key not in PARQUET_BINDINGS:
        raise GateCParquetError(f"{L}.source.key {key!r} is not a frozen source binding")
    spec = SOURCES[key]
    binding = PARQUET_BINDINGS[key]
    for field, expected in (
        ("key", plan["source_key"]),
        ("dataset", spec.dataset),
        ("config", spec.dataset_config),
        ("split", spec.split),
        ("revision", spec.revision),
        ("license", spec.license),
        ("body_path", spec.body_path),
        ("min_bytes", spec.min_bytes),
        ("max_bytes", spec.max_bytes),
        ("transport", TRANSPORT),
        ("parquet_repo_id", binding.repo_id),
        ("parquet_revision", binding.parquet_revision),
        ("parquet_is_export_branch", binding.is_export_branch),
    ):
        _require_exact(source, field, expected, label=f"{L}.source")
    for field in ("dataset", "config", "split"):
        _require_exact(source, field, transport_manifest[field], label=f"{L}.source")
    _require_exact(source, "revision", transport_manifest["data_revision"], label=f"{L}.source")

    # --- 5.4 sampler block ---
    sampler = _as_mapping(_require_field(manifest, "sampler", L), f"{L}.sampler")
    for field, expected in (
        ("version", SAMPLER_VERSION),
        ("seed", plan["seed"]),
        ("units_planned", plan["planned_units"]),
        ("rows_per_unit", plan["rows_per_unit"]),
        ("file_weighting", plan["file_weighting"]),
        ("file_weighting_exact_for_rows", plan["file_weighting_exact_for_rows"]),
        ("within_row_group_selection", plan["within_row_group_selection"]),
        ("within_row_group_start", plan["within_row_group_start"]),
        ("within_row_group_seed_sensitive", plan["within_row_group_seed_sensitive"]),
        ("without_replacement_scope", plan["without_replacement_scope"]),
    ):
        _require_exact(sampler, field, expected, label=f"{L}.sampler")
    for field, expected in FROZEN_SAMPLER_STATIC_FIELDS.items():
        _require_exact(sampler, field, expected, label=f"{L}.sampler")

    # --- 5.5 documents and accounting, recomputed from the published bytes ---
    byte_lines = [line for line in documents_bytes.split(b"\n") if line]
    try:
        decoded_documents = documents_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise GateCParquetError(f"documents.jsonl is not strict UTF-8: {exc}") from exc
    str_lines = [line for line in decoded_documents.splitlines() if line]
    if len(byte_lines) != len(str_lines):
        raise GateCParquetError(
            f"physical JSONL framing is ambiguous: {len(byte_lines)} byte-delimited records vs "
            f"{len(str_lines)} str.splitlines() records"
        )
    record_ids: set[str] = set()
    text_hashes: set[str] = set()
    global_rows: set[int] = set()
    by_format: Counter = Counter()
    by_forum: Counter = Counter()
    accepted_text_bytes = 0
    for index, line in enumerate(byte_lines):
        label = f"documents.jsonl[{index}]"
        try:
            record = json.loads(line.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GateCParquetError(f"{label} is not strict UTF-8 JSON: {exc}") from exc
        record = _as_mapping(record, label)
        text = _require_field(record, "text", label)
        if not isinstance(text, str):
            raise GateCParquetError(f"{label}.text must be a string")
        encoded = text.encode("utf-8")
        _require_exact(record, "text_bytes", len(encoded), label=label)
        _require_exact(record, "text_sha256", _sha256_bytes(encoded), label=label)
        record_id = _require_str(
            _require_field(record, "source_record_id", label), f"{label}.source_record_id"
        )
        if record_id in record_ids:
            raise GateCParquetError("published release contains a duplicate source record id")
        if record["text_sha256"] in text_hashes:
            raise GateCParquetError("published release contains a duplicate text hash")
        provenance = _as_mapping(_require_field(record, "provenance", label), f"{label}.provenance")
        _require_exact(provenance, "transport", TRANSPORT, label=f"{label}.provenance")
        position = provenance.get("source_global_row_index")
        if position is not None:
            if type(position) is not int:
                raise GateCParquetError(f"{label}.provenance.source_global_row_index is not an int")
            if position in global_rows:
                raise GateCParquetError("published release contains a duplicate global row index")
            global_rows.add(position)
        record_ids.add(record_id)
        text_hashes.add(record["text_sha256"])
        accepted_text_bytes += len(encoded)
        metadata = _as_mapping(_require_field(record, "metadata", label), f"{label}.metadata")
        document_format = metadata.get("format")
        if isinstance(document_format, str):
            by_format[document_format] += 1
        forum = metadata.get("metadata.forum")
        if isinstance(forum, str) and forum:
            by_forum[forum] += 1
    accepted = len(byte_lines)

    accounting = _as_mapping(_require_field(manifest, "accounting", L), f"{L}.accounting")
    for field in (
        "scanned",
        "accepted",
        "rejected",
        "accepted_text_bytes",
        "response_bytes",
        "request_count",
    ):
        _require_int(
            _require_field(accounting, field, f"{L}.accounting"), f"{L}.accounting.{field}"
        )
    _require_exact(accounting, "accepted", accepted, label=f"{L}.accounting")
    _require_exact(accounting, "accepted_text_bytes", accepted_text_bytes, label=f"{L}.accounting")
    rejections = _as_mapping(
        _require_field(accounting, "rejections", f"{L}.accounting"), f"{L}.accounting.rejections"
    )
    if accounting["scanned"] != accounting["accepted"] + accounting["rejected"]:
        raise GateCParquetError(
            f"{L}.accounting scanned {accounting['scanned']} != accepted + rejected"
        )
    if sum(rejections.values()) != accounting["rejected"]:
        raise GateCParquetError(
            f"{L}.accounting rejection counters sum to {sum(rejections.values())}, "
            f"not {accounting['rejected']}"
        )
    yield_rate = manifest.get("yield_rate")
    if accounting["scanned"]:
        expected_yield = accounting["accepted"] / accounting["scanned"]
        if type(yield_rate) not in (int, float) or abs(yield_rate - expected_yield) > 1e-12:
            raise GateCParquetError(
                f"{L}.yield_rate {yield_rate!r} != accepted/scanned {expected_yield!r}"
            )
    elif yield_rate is not None:
        raise GateCParquetError(f"{L}.yield_rate must be null when nothing was scanned")
    _require_exact_mapping(
        _require_field(manifest, "accepted_by_format", L),
        dict(by_format),
        f"{L}.accepted_by_format",
    )
    _require_exact_mapping(
        _require_field(manifest, "accepted_by_forum", L), dict(by_forum), f"{L}.accepted_by_forum"
    )

    # --- 5.6 accepted-document coverage, rebuilt from the published documents ---
    rebuilt = reconstruct_accepted_document_coverage(
        output_dir / DOCUMENTS_NAME, transport_manifest.get("total_rows")
    )
    accepted_cov = _as_mapping(
        _require_field(manifest, "accepted_document_coverage", L),
        f"{L}.accepted_document_coverage",
    )
    for field, expected in rebuilt.items():
        _require_exact(accepted_cov, field, expected, label=f"{L}.accepted_document_coverage")
    _require_exact(
        accepted_cov,
        "source_file_count",
        transport_manifest["file_count"],
        label=f"{L}.accepted_document_coverage",
    )
    _require_exact(
        accepted_cov,
        "source_total_rows",
        transport_manifest.get("total_rows"),
        label=f"{L}.accepted_document_coverage",
    )
    _require_exact(
        accepted_cov,
        "derivation",
        ACCEPTED_COVERAGE_DERIVATION,
        label=f"{L}.accepted_document_coverage",
    )

    # --- 5.7 transport-read coverage: only the internal contract the build really guarantees ---
    transport_cov = _as_mapping(
        _require_field(manifest, "transport_read_coverage", L), f"{L}.transport_read_coverage"
    )
    TL = f"{L}.transport_read_coverage"
    files_list = _as_list(
        _require_field(transport_cov, "files_opened_list", TL),
        f"{TL}.files_opened_list",
        allow_empty=True,
    )
    groups_list = _as_list(
        _require_field(transport_cov, "row_groups_opened_list", TL),
        f"{TL}.row_groups_opened_list",
        allow_empty=True,
    )
    for name, values in (
        ("files_opened_list", files_list),
        ("row_groups_opened_list", groups_list),
    ):
        if any(not isinstance(v, str) for v in values):
            raise GateCParquetError(f"{TL}.{name} must contain strings")
        if values != sorted(set(values)):
            raise GateCParquetError(f"{TL}.{name} must be a sorted list of unique values")
    _require_exact(transport_cov, "files_opened", len(files_list), label=TL)
    _require_exact(transport_cov, "row_groups_opened", len(groups_list), label=TL)
    manifest_paths = {item["path"] for item in transport_manifest["files"]}
    for path in files_list:
        if path not in manifest_paths:
            raise GateCParquetError(f"{TL}.files_opened_list has {path}, not in the manifest")
    plan_units = {f"{u['file_path']}#{u['row_group']}" for u in plan["units"]}
    for entry in groups_list:
        if entry not in plan_units:
            raise GateCParquetError(f"{TL}.row_groups_opened_list has {entry}, not a planned unit")
        if entry.rsplit("#", 1)[0] not in files_list:
            raise GateCParquetError(
                f"{TL}.row_groups_opened_list has {entry} whose file is not in files_opened_list"
            )
    attempted = _require_int(
        _require_field(transport_cov, "planned_units_attempted", TL),
        f"{TL}.planned_units_attempted",
    )
    completed = _require_int(
        _require_field(transport_cov, "planned_units_completed", TL),
        f"{TL}.planned_units_completed",
    )
    # NOTE: attempted is cumulative across resumes and may legitimately exceed the number of
    # unique planned units, because an interrupted unit is attempted again after a resume.
    if completed > attempted:
        raise GateCParquetError(f"{TL} completed {completed} > attempted {attempted}")
    if completed > plan["planned_units"]:
        raise GateCParquetError(
            f"{TL} completed {completed} > plan planned_units {plan['planned_units']}"
        )
    _require_exact(transport_cov, "accumulates_across_resumes", True, label=TL)
    _require_exact(transport_cov, "derivation", TRANSPORT_COVERAGE_DERIVATION, label=TL)

    # --- 5.8 deprecated mirror ---
    deprecated = _as_mapping(_require_field(manifest, "coverage", L), f"{L}.coverage")
    _require_exact(deprecated, "deprecated", True, label=f"{L}.coverage")
    _require_exact(deprecated, "deprecated_note", DEPRECATED_COVERAGE_NOTE, label=f"{L}.coverage")
    _require_exact(
        deprecated, "files_touched", rebuilt["accepted_file_count"], label=f"{L}.coverage"
    )
    _require_exact(
        deprecated, "row_groups_touched", rebuilt["accepted_row_group_count"], label=f"{L}.coverage"
    )

    # --- 5.9 resource accounting ---
    resource = _as_mapping(
        _require_field(manifest, "resource_accounting", L), f"{L}.resource_accounting"
    )
    RL = f"{L}.resource_accounting"
    current_bytes = _require_int(
        _require_field(resource, "body_reader_exposed_bytes", RL), f"{RL}.body_reader_exposed_bytes"
    )
    current_calls = _require_int(
        _require_field(resource, "body_reader_read_calls", RL), f"{RL}.body_reader_read_calls"
    )
    cum_bytes = _require_int(
        _require_field(resource, "cumulative_body_reader_exposed_bytes", RL),
        f"{RL}.cumulative_body_reader_exposed_bytes",
    )
    cum_calls = _require_int(
        _require_field(resource, "cumulative_body_reader_read_calls", RL),
        f"{RL}.cumulative_body_reader_read_calls",
    )
    if cum_bytes < current_bytes or cum_calls < current_calls:
        raise GateCParquetError(f"{RL} cumulative counters are below the current invocation")
    for field, expected in FROZEN_RESOURCE_STATIC_FIELDS.items():
        _require_exact(resource, field, expected, label=RL)
    _require_exact(accounting, "response_bytes", cum_bytes, label=f"{L}.accounting")
    _require_exact(accounting, "request_count", cum_calls, label=f"{L}.accounting")
    current_wall = _require_number(
        _require_field(manifest, "current_invocation_wall_seconds", L),
        f"{L}.current_invocation_wall_seconds",
    )
    cumulative_wall = _require_number(
        _require_field(manifest, "cumulative_build_wall_seconds", L),
        f"{L}.cumulative_build_wall_seconds",
    )
    if cumulative_wall < current_wall:
        raise GateCParquetError(f"{L}.cumulative_build_wall_seconds < current invocation")
    _require_int(_require_field(manifest, "resume_count", L), f"{L}.resume_count")

    # --- 5.10 persisted inputs are the SHA-256 of the published file bytes ---
    persisted = _as_mapping(
        _require_field(manifest, "persisted_inputs_before_first_body_read", L),
        f"{L}.persisted_inputs_before_first_body_read",
    )
    PL = f"{L}.persisted_inputs_before_first_body_read"
    if set(persisted) != {TRANSPORT_MANIFEST_NAME, SELECTION_PLAN_NAME}:
        raise GateCParquetError(f"{PL} must record exactly the two persisted inputs")
    for name in (TRANSPORT_MANIFEST_NAME, SELECTION_PLAN_NAME):
        digest = _require_sha256(persisted[name], f"{PL}[{name}]")
        actual = _sha256_bytes((output_dir / name).read_bytes())
        if digest != actual:
            raise GateCParquetError(
                f"{PL}[{name}] {digest} != SHA-256 of the published file bytes {actual}"
            )

    # --- 5.11 scope and hard stops ---
    _require_exact_mapping(
        _require_field(manifest, "gate_c_scope", L), FROZEN_GATE_C_SCOPE, f"{L}.gate_c_scope"
    )
    _require_exact_mapping(
        _require_field(manifest, "hard_stops", L), FROZEN_HARD_STOPS, f"{L}.hard_stops"
    )

    return {
        "accepted": accepted,
        "accepted_text_bytes": accepted_text_bytes,
        "distinct_global_rows": len(global_rows),
        "accepted_document_coverage": rebuilt,
        "plan_sha256": plan_sha,
        "transport_manifest_sha256": transport_sha,
    }


def verify_release_v2(output_dir: Path) -> dict[str, Any]:
    """Re-read a published v2 release and verify every recorded checksum and invariant."""
    output_dir = _ensure_under_workspace(output_dir)
    manifest, manifest_bytes = _read_release_json(output_dir, MANIFEST_NAME)
    transport, _ = _read_release_json(output_dir, TRANSPORT_MANIFEST_NAME)
    plan, _ = _read_release_json(output_dir, SELECTION_PLAN_NAME)
    for name in (DOCUMENTS_NAME, CHECKSUMS_NAME):
        if not (output_dir / name).is_file():
            raise GateCParquetError(f"published release is missing {name}")
    documents_bytes = (output_dir / DOCUMENTS_NAME).read_bytes()
    _require_exact(
        manifest, "documents_sha256", _sha256_bytes(documents_bytes), label="release manifest"
    )
    _require_exact(manifest, "documents_bytes", len(documents_bytes), label="release manifest")

    summary = validate_release_manifest_v2(
        manifest,
        output_dir=output_dir,
        transport_manifest=transport,
        plan=plan,
        documents_bytes=documents_bytes,
    )
    accepted = summary["accepted"]
    global_rows_count = summary["distinct_global_rows"]

    for entry in (output_dir / CHECKSUMS_NAME).read_text().splitlines():
        if not entry:
            continue
        if "  " not in entry:
            raise GateCParquetError(f"{CHECKSUMS_NAME} line is malformed: {entry!r}")
        digest, name = entry.split("  ", 1)
        target = output_dir / name
        if not target.is_file():
            raise GateCParquetError(f"{CHECKSUMS_NAME} references a missing file: {name}")
        if _sha256_bytes(target.read_bytes()) != digest:
            raise GateCParquetError(f"MANIFEST.sha256 mismatch for {name}")
    return {
        "output_dir": str(output_dir),
        "accepted": accepted,
        "release_kind": manifest["release_kind"],
        "promotion_eligible": manifest["promotion_eligible"],
        "distinct_global_rows": global_rows_count,
        "release_manifest_validated": True,
        "accepted_file_count": manifest["accepted_document_coverage"]["accepted_file_count"],
        "accepted_row_group_count": manifest["accepted_document_coverage"][
            "accepted_row_group_count"
        ],
        "transport_files_opened": manifest["transport_read_coverage"]["files_opened"],
        "transport_row_groups_opened": manifest["transport_read_coverage"]["row_groups_opened"],
        "jsonl_byte_lines_equal_str_lines": True,
        "representative_sampler": manifest["sampler"]["representative_sampler"],
        "documents_sha256": manifest["documents_sha256"],
        "manifest_sha256": _sha256_bytes(manifest_bytes),
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    temporary = path.parent / f".{path.name}.tmp"
    if temporary.exists():
        temporary.unlink()
    _write_new_file(temporary, payload)
    os.replace(temporary, path)
    _fsync_directory(path.parent)
    digest = _sha256_bytes(payload)
    _write_new_file(path.with_suffix(path.suffix + ".sha256"), f"{digest}  {path.name}\n".encode())
    return digest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PetitGPT Gate C v2 Parquet transport")
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("manifest", help="build a pinned Parquet topology manifest")
    manifest.add_argument("--source", required=True, choices=sorted(PARQUET_BINDINGS))
    manifest.add_argument("--out", required=True, type=Path)
    manifest.add_argument("--footer-policy", choices=["complete", "selected_files_only"])

    plan = sub.add_parser(
        "plan", help="freeze a deterministic seeded dispersed cluster selection plan"
    )
    plan.add_argument("--transport-manifest", required=True, type=Path)
    plan.add_argument("--out", required=True, type=Path)
    plan.add_argument("--seed", required=True, type=int)
    plan.add_argument("--units", type=int, default=DEFAULT_UNITS)
    plan.add_argument("--rows-per-unit", type=int, default=128)

    verify = sub.add_parser("verify", help="verify a published C0-v2 release")
    verify.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "manifest":
            store = HubObjectStore()
            value = build_transport_manifest(args.source, store, footer_policy=args.footer_policy)
            digest = _atomic_write_json(args.out, value)
            result = {
                "out": str(args.out),
                "sha256": digest,
                "file_count": value["file_count"],
                "total_rows": value["total_rows"],
                "production_row_group_manifest_complete": value[
                    "production_row_group_manifest_complete"
                ],
            }
        elif args.command == "plan":
            manifest = json.loads(args.transport_manifest.read_text())
            value = build_selection_plan(
                manifest,
                seed=args.seed,
                units=args.units,
                rows_per_unit=args.rows_per_unit,
                store=HubObjectStore(),
            )
            digest = _atomic_write_json(args.out, value)
            result = {
                "out": str(args.out),
                "sha256": digest,
                "planned_units": value["planned_units"],
                "distinct_files": value["distinct_files"],
                "planned_rows": value["planned_rows"],
            }
        else:
            result = verify_release_v2(args.output_dir)
    except GateCError as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
