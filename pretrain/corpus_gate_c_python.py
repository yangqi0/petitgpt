#!/usr/bin/env python3

"""Bounded Python Gate C C0 diagnostic builder for Common Pile ``stackv2_edu_filtered``.

This is a *diagnostic-only* sibling of the frozen non-Python Gate C builders.  It answers a
narrow set of questions about the pinned Python shards — can we bind them, stream them under
explicit caps, filter them deterministically, resume safely, publish immutably, and what does the
post-gate yield / score distribution look like — and nothing else.

Deliberately out of scope here, and asserted as such in every manifest: production candidate
quota, canonical-token accounting, tokenizer training, near-dedup, benchmark decontamination,
reference-reserve exclusion, chat conversion, textual document separators, and BOS/EOS insertion.

Transport is intentionally *independent* of ``pretrain/corpus_gate_c_parquet.py``: the pinned
revision of this dataset contains no Parquet at all, only 95 gzip JSON-lines shards, of which
exactly 12 are Python.  Only the genuinely transport-independent helpers of
``pretrain/corpus_gate_c.py`` (atomic publication, checkpoint framing, strict UTF-8, coarse
repetition, canonical JSONL serialization) are reused; the frozen non-Python modules are neither
modified nor re-purposed.

Every release this module can produce is ``release_kind="c0_diagnostic"`` with
``promotion_eligible=false``.  The per-shard compressed-prefix ceiling makes a full read of the
Python corpus structurally impossible with this tool, so a C0 run can never become a disguised
production build.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any
import zlib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pretrain.corpus_gate_c import (  # noqa: E402
    CHECKPOINT_NAME,
    CHECKSUMS_NAME,
    DOCUMENTS_NAME,
    MANIFEST_NAME,
    GateCError,
    PlannedInterruption,
    _atomic_replace_bytes,
    _canonical_json_bytes,
    _ensure_git_ignored,
    _ensure_revision,
    _ensure_under_workspace,
    _fsync_directory,
    _pathological_repetition,
    _publish_directory,
    _require_nonnegative_int,
    _sha256_bytes,
    _staging_path,
    _strict_utf8_bytes,
    _utc_now,
    _write_new_file,
    canonical_jsonl_record_bytes,
)

__all__ = [
    "PYTHON_SOURCE",
    "BuildConfig",
    "GateCPythonError",
    "HubShardStore",
    "LocalShardStore",
    "PlannedInterruption",
    "PythonSourceSpec",
    "ShardSpec",
    "build_candidates",
    "diagnose_release",
    "evaluate_record",
    "main",
    "verify_release",
]

TOOL_SCHEMA_VERSION = "petitgpt-corpus-gate-c-python-v1"
SPEC_VERSION = "python-gate-c-c0-source-spec-2026-08-17"

DIAGNOSTICS_NAME = "score_diagnostics.json"

# --------------------------------------------------------------------------------------
# Frozen C0 ceilings
#
# These are hard structural ceilings, not the parameters of any particular run.  The per-shard
# compressed prefix ceiling (256 MiB) times 12 shards is 3 GiB against an 11.72 GB pinned Python
# corpus, so this tool cannot read the Python corpus even if every cap is set to its maximum.  The
# accepted-document ceiling likewise keeps any release two orders of magnitude below the frozen
# 0.60B-token Python quota.
# --------------------------------------------------------------------------------------

MAX_ACCEPTED_DOCUMENTS = 60_000
MAX_SCANNED_RECORDS = 400_000
MAX_SHARD_COMPRESSED_BYTES = 256 * 1024 * 1024
MAX_RESPONSE_BYTES = 4 * 1024 * 1024 * 1024
MAX_WALL_SECONDS = 6 * 60 * 60
MAX_STRIDE = 4096
DEFAULT_CHECKPOINT_EVERY = 1024

READ_BLOCK_BYTES = 8 * 1024 * 1024
MAX_DECOMPRESS_CHUNK_BYTES = 64 * 1024 * 1024

# --------------------------------------------------------------------------------------
# Frozen Python mechanical filter parameters (§6 of the authorized C0 scope)
# --------------------------------------------------------------------------------------

PYTHON_MIN_BYTES = 200
PYTHON_MAX_BYTES = 8192
PYTHON_MAX_COMMENT_BLANK_FRACTION = 0.70

# Applied in this exact order; the accepted set is order-independent (every gate is conjunctive)
# but the rejection histogram is not, so the order is frozen and republished in every manifest.
FILTER_ORDER = (
    "record_shape",
    "language_not_python",
    "generated",
    "vendor",
    "strict_utf8",
    "size_band",
    "ast_parse",
    "repetition",
    "comment_blank_fraction",
    "duplicate_source_record_id",
    "duplicate_text_sha256",
)

# Rejections that happen before a record has ever been parsed into an object, so the per-shard
# frozen-leaf-set check cannot have run yet.
_PRE_SCHEMA_REJECTS = frozenset({"line_not_utf8", "malformed_json", "record_not_object"})

# The Stage A / Stage B ratio is frozen, but at Gate C the canonical tokenizer does not exist, so
# this module may only ever apply it to *bytes* and must say so in every artifact it writes.
STAGE_B_MASS_SHARE = 5.0 / 12.0


class GateCPythonError(GateCError):
    """A fail-closed Python Gate C contract error."""


# --------------------------------------------------------------------------------------
# Frozen source binding
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardSpec:
    """One pinned gzip JSON-lines shard at the pinned revision."""

    path: str
    size: int
    lfs_sha256: str
    blob_id: str

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size": self.size,
            "lfs_sha256": self.lfs_sha256,
            "blob_id": self.blob_id,
        }


@dataclass(frozen=True)
class RemoteShard:
    """One shard as the live transport reports it."""

    path: str
    size: int
    lfs_sha256: str
    blob_id: str


@dataclass(frozen=True)
class PythonSourceSpec:
    """An exact, frozen, Python-only Gate C binding.

    ``required_schema`` freezes the load-bearing JSON leaves with their exact JSON types.
    ``frozen_leaf_set`` freezes the *complete* leaf set of a record so an upstream schema change
    fails closed instead of being silently ignored.  ``nullable_paths`` are leaves that exist on
    every record but legitimately carry ``null``; none of them is load-bearing.
    """

    key: str
    dataset: str
    dataset_config: str
    split: str
    revision: str
    body_path: str
    language: str
    license: str
    shards: tuple[ShardSpec, ...]
    revision_shard_count: int
    required_schema: tuple[tuple[str, str], ...]
    frozen_leaf_set: tuple[str, ...]
    nullable_paths: tuple[str, ...]
    natural_id_path: str
    score_path: str
    min_bytes: int = PYTHON_MIN_BYTES
    max_bytes: int = PYTHON_MAX_BYTES
    max_comment_blank_fraction: float = PYTHON_MAX_COMMENT_BLANK_FRACTION

    @property
    def required_schema_map(self) -> dict[str, str]:
        return dict(self.required_schema)

    @property
    def shard_paths(self) -> tuple[str, ...]:
        return tuple(shard.path for shard in self.shards)

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "dataset": self.dataset,
            "config": self.dataset_config,
            "split": self.split,
            "revision": self.revision,
            "body_path": self.body_path,
            "language": self.language,
            "license": self.license,
            "revision_shard_count": self.revision_shard_count,
            "python_shard_count": len(self.shards),
            "shards": [shard.to_json() for shard in self.shards],
            "min_bytes": self.min_bytes,
            "max_bytes": self.max_bytes,
            "max_comment_blank_fraction": self.max_comment_blank_fraction,
            "natural_id_path": self.natural_id_path,
            "score_path": self.score_path,
            "transport": "huggingface_hub_gzip_jsonl_prefix",
        }


def _shard(path: str, size: int, lfs_sha256: str, blob_id: str) -> ShardSpec:
    return ShardSpec(path=path, size=size, lfs_sha256=lfs_sha256, blob_id=blob_id)


# The 12 Python shards of the pinned revision, bound by exact path, size, LFS SHA-256 and Git blob
# id.  Identity is re-verified against the live listing on every build; any drift fails closed.
PYTHON_SHARDS: tuple[ShardSpec, ...] = (
    _shard(
        "stack-edu-0073.json.gz",
        976494926,
        "b04f81e60a5397b1ade2099e1640ef5c1b105abfeb61c57e6b2e7c4dacef836e",
        "da6d5fe5eac38d2b68226abd73783fa87f2df532",
    ),
    _shard(
        "stack-edu-0074.json.gz",
        974511426,
        "bb655343ccf1200b022e8e7d118a41cdc22b6c0c66ea664e849d3c1e64af4018",
        "63540cbcba8732b4a71974906163a67913727ea9",
    ),
    _shard(
        "stack-edu-0075.json.gz",
        977861558,
        "b59b65a638bb26fffc45db87f8da363a36435c80f6d8276b37fc81c9fe53e5a4",
        "bcde01091d3594d824f27ebcefe1adda49d858c9",
    ),
    _shard(
        "stack-edu-0076.json.gz",
        976451737,
        "3e48dfa8ef1cf594b1a7042f66862b8946477eeb284d24b158e1d822bf765ffe",
        "17f54411b6007879a6863150fccee069105f28cf",
    ),
    _shard(
        "stack-edu-0077.json.gz",
        971788667,
        "d0ca3bcfb871b717855383bfef6500b0c5ba5333b739b5acae94ca12ce5fb36d",
        "d6a531ca93526aeaec3475e8a71cc9da194f68c4",
    ),
    _shard(
        "stack-edu-0078.json.gz",
        972955403,
        "b70cd0d934d518d6523e3296c85bc004ad43590edbb8ea82e0751dc72c786c13",
        "0137bcc4852ee286d4a25a4ed8b9c9efafe5b277",
    ),
    _shard(
        "stack-edu-0079.json.gz",
        970216639,
        "7eeb9e77151c3766af1b611e3230869cde9833a47ea8c06749187b295b863dc4",
        "be117618173dbf69e6c56e55b4aa9bf0772be316",
    ),
    _shard(
        "stack-edu-0080.json.gz",
        971712713,
        "7847a264bc3abcfd02fb68bab0a2afb5ff099fa1dc670a8cb0d87f0d5d5d5608",
        "ce9c821e8431bb708455f6d245f15ff3a0db3051",
    ),
    _shard(
        "stack-edu-0081.json.gz",
        978858699,
        "21ce2f93b1c92e40f3afdedefb7c560de79ca92fde299ef52d91d7be74644577",
        "8ab3833cc0bfbb1a61986e3ee8a7d2f04f79f742",
    ),
    _shard(
        "stack-edu-0082.json.gz",
        974994520,
        "65a78e66c9548fa2f2668c560d37cc4576f4307183582d345e3aa8ef88a1656b",
        "93e3849433af47e2c17f0088265faea709b0ab16",
    ),
    _shard(
        "stack-edu-0083.json.gz",
        983678048,
        "7396ecba90289dbb242cd86aed8c180b0449e5e72f3f60f9838453f2e07ee1fa",
        "71ca73f5ee8490485b32502554a91b56e4752f16",
    ),
    _shard(
        "stack-edu-0084.json.gz",
        987435422,
        "5e2ad7d634976f90694a2f644922a6fc92b54d624958440eea532a7ae2bda3b8",
        "f8ee8f8b5ee0994fb5edbccee8ac156fe957bffa",
    ),
)

# Load-bearing leaves, pinned to their exact JSON type.  "number" accepts int or float because JSON
# has one numeric type and a whole-valued score may serialize either way.
PYTHON_REQUIRED_SCHEMA: tuple[tuple[str, str], ...] = (
    ("added", "str"),
    ("created", "str"),
    ("id", "str"),
    ("int_score", "int"),
    ("metadata.blob_id", "str"),
    ("metadata.content_id", "str"),
    ("metadata.detected_licenses", "list"),
    ("metadata.extension", "str"),
    ("metadata.is_generated", "bool"),
    ("metadata.is_vendor", "bool"),
    ("metadata.language", "str"),
    ("metadata.length_bytes", "int"),
    ("metadata.license", "str"),
    ("metadata.license_type", "str"),
    ("metadata.path", "str"),
    ("metadata.provenance", "str"),
    ("metadata.repo_name", "str"),
    ("metadata.src_encoding", "str"),
    ("score", "number"),
    ("source", "str"),
    ("text", "str"),
)

# The complete 37-leaf record shape at the pinned revision.  Anything added or removed upstream
# fails the build closed rather than being filtered around.
PYTHON_FROZEN_LEAF_SET: tuple[str, ...] = (
    "added",
    "created",
    "id",
    "int_score",
    "metadata.blob_id",
    "metadata.branch_name",
    "metadata.committer_date",
    "metadata.content_id",
    "metadata.detected_licenses",
    "metadata.directory_id",
    "metadata.extension",
    "metadata.filename",
    "metadata.fork_events_count",
    "metadata.gha_created_at",
    "metadata.gha_event_created_at",
    "metadata.gha_language",
    "metadata.gha_license_id",
    "metadata.github_id",
    "metadata.is_generated",
    "metadata.is_vendor",
    "metadata.language",
    "metadata.length_bytes",
    "metadata.license",
    "metadata.license_type",
    "metadata.path",
    "metadata.provenance",
    "metadata.repo_name",
    "metadata.revision_date",
    "metadata.revision_id",
    "metadata.snapshot_id",
    "metadata.src_encoding",
    "metadata.star_events_count",
    "metadata.url",
    "metadata.visit_date",
    "score",
    "source",
    "text",
)

PYTHON_NULLABLE_PATHS: tuple[str, ...] = (
    "metadata.gha_created_at",
    "metadata.gha_event_created_at",
    "metadata.gha_language",
    "metadata.gha_license_id",
    "metadata.github_id",
)

PYTHON_SOURCE = PythonSourceSpec(
    key="common_pile_stackv2_edu_python",
    dataset="common-pile/stackv2_edu_filtered",
    dataset_config="default",
    split="train",
    revision="c354dbe88469a1153e97c6a63ac50591849654de",
    body_path="text",
    language="Python",
    license="per-record metadata.license (Software Heritage permissive subset)",
    shards=PYTHON_SHARDS,
    revision_shard_count=95,
    required_schema=PYTHON_REQUIRED_SCHEMA,
    frozen_leaf_set=PYTHON_FROZEN_LEAF_SET,
    nullable_paths=PYTHON_NULLABLE_PATHS,
    natural_id_path="id",
    score_path="score",
)

# Provenance metadata retained on every accepted document for later selection stages.
RETAINED_METADATA_PATHS: tuple[str, ...] = (
    "score",
    "int_score",
    "metadata.repo_name",
    "metadata.license",
    "metadata.license_type",
    "metadata.detected_licenses",
    "metadata.path",
    "metadata.extension",
    "metadata.blob_id",
    "metadata.content_id",
    "metadata.provenance",
    "metadata.length_bytes",
    "metadata.src_encoding",
    "metadata.star_events_count",
    "metadata.fork_events_count",
    "metadata.language",
)


# --------------------------------------------------------------------------------------
# Build configuration and accounting
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BuildConfig:
    """Bounded Python Gate C C0 build parameters."""

    source: PythonSourceSpec
    output_dir: Path
    work_dir: Path
    target_documents: int
    max_scanned: int
    max_shard_records: int
    max_shard_compressed_bytes: int
    max_response_bytes: int
    max_wall_seconds: float
    stride: int
    seed: int
    stop_after_documents: int | None = None
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY


@dataclass
class BuildCounters:
    """Scanned/accepted/rejected/byte accounting plus structured rejection counters."""

    scanned: int = 0
    accepted: int = 0
    rejected: int = 0
    accepted_text_bytes: int = 0
    stride_skipped: int = 0
    compressed_bytes: int = 0
    decompressed_bytes: int = 0
    resume_reread_compressed_bytes: int = 0
    request_count: int = 0
    rejections: Counter = field(default_factory=Counter)
    field_violations: Counter = field(default_factory=Counter)
    diagnostics: Counter = field(default_factory=Counter)

    def to_json(self) -> dict[str, Any]:
        return {
            "scanned": self.scanned,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "accepted_text_bytes": self.accepted_text_bytes,
            "stride_skipped": self.stride_skipped,
            "compressed_bytes": self.compressed_bytes,
            "decompressed_bytes": self.decompressed_bytes,
            "resume_reread_compressed_bytes": self.resume_reread_compressed_bytes,
            "request_count": self.request_count,
            "rejections": dict(sorted(self.rejections.items())),
            "field_violations": dict(sorted(self.field_violations.items())),
            "diagnostics": dict(sorted(self.diagnostics.items())),
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> BuildCounters:
        maps = {}
        for name in ("rejections", "field_violations", "diagnostics"):
            entry = value.get(name) or {}
            if not isinstance(entry, Mapping):
                raise GateCPythonError("checkpoint counter maps are invalid")
            maps[name] = Counter(dict(entry))
        return cls(
            scanned=_require_nonnegative_int(value.get("scanned"), "scanned"),
            accepted=_require_nonnegative_int(value.get("accepted"), "accepted"),
            rejected=_require_nonnegative_int(value.get("rejected"), "rejected"),
            accepted_text_bytes=_require_nonnegative_int(
                value.get("accepted_text_bytes"), "accepted_text_bytes"
            ),
            stride_skipped=_require_nonnegative_int(value.get("stride_skipped"), "stride_skipped"),
            compressed_bytes=_require_nonnegative_int(
                value.get("compressed_bytes"), "compressed_bytes"
            ),
            decompressed_bytes=_require_nonnegative_int(
                value.get("decompressed_bytes"), "decompressed_bytes"
            ),
            resume_reread_compressed_bytes=_require_nonnegative_int(
                value.get("resume_reread_compressed_bytes"), "resume_reread_compressed_bytes"
            ),
            request_count=_require_nonnegative_int(value.get("request_count"), "request_count"),
            rejections=maps["rejections"],
            field_violations=maps["field_violations"],
            diagnostics=maps["diagnostics"],
        )


# --------------------------------------------------------------------------------------
# Transport: pinned gzip JSON-lines shard prefixes
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ScannedRecord:
    """One physical JSONL record read from one pinned shard prefix."""

    shard_index: int
    shard_path: str
    record_index: int
    line: bytes


class ShardStore:
    """Transport interface: list pinned shard identities and stream bounded shard prefixes."""

    def list_shards(self, source: PythonSourceSpec) -> tuple[RemoteShard, ...]:
        raise NotImplementedError

    def revision_file_count(self, source: PythonSourceSpec) -> int:
        raise NotImplementedError

    def open_prefix(self, source: PythonSourceSpec, path: str, max_bytes: int) -> Iterator[bytes]:
        raise NotImplementedError


class HubShardStore(ShardStore):
    """Official Hugging Face Hub transport, pinned to the exact revision.

    ``huggingface_hub`` is imported lazily so the CPU-only contract test suite never needs it.
    """

    def __init__(self, *, block_bytes: int = READ_BLOCK_BYTES, max_attempts: int = 3) -> None:
        self.block_bytes = block_bytes
        self.max_attempts = max_attempts
        self._api = None
        self._fs = None
        self._siblings: dict[str, tuple[RemoteShard, ...]] = {}

    def _hub(self) -> tuple[Any, Any]:
        if self._api is None or self._fs is None:
            from huggingface_hub import HfApi, HfFileSystem

            self._api = HfApi()
            self._fs = HfFileSystem()
        return self._api, self._fs

    def _repo_shards(self, source: PythonSourceSpec) -> tuple[RemoteShard, ...]:
        cached = self._siblings.get(source.revision)
        if cached is not None:
            return cached
        api, _ = self._hub()
        info = api.repo_info(
            source.dataset,
            revision=source.revision,
            repo_type="dataset",
            files_metadata=True,
        )
        if getattr(info, "sha", None) != source.revision:
            raise GateCPythonError(
                "Hugging Face resolved revision does not match the pinned commit"
            )
        shards = []
        for sibling in info.siblings:
            name = sibling.rfilename
            if not name.endswith(".json.gz"):
                continue
            lfs = getattr(sibling, "lfs", None)
            oid = ""
            if lfs is not None:
                oid = getattr(lfs, "sha256", None) or (
                    lfs.get("sha256", "") if isinstance(lfs, Mapping) else ""
                )
            shards.append(
                RemoteShard(
                    path=name,
                    size=int(sibling.size or 0),
                    lfs_sha256=str(oid or ""),
                    blob_id=str(getattr(sibling, "blob_id", "") or ""),
                )
            )
        resolved = tuple(sorted(shards, key=lambda item: item.path))
        self._siblings[source.revision] = resolved
        return resolved

    def revision_file_count(self, source: PythonSourceSpec) -> int:
        return len(self._repo_shards(source))

    def list_shards(self, source: PythonSourceSpec) -> tuple[RemoteShard, ...]:
        wanted = set(source.shard_paths)
        return tuple(item for item in self._repo_shards(source) if item.path in wanted)

    def open_prefix(self, source: PythonSourceSpec, path: str, max_bytes: int) -> Iterator[bytes]:
        _, fs = self._hub()
        uri = f"datasets/{source.dataset}@{source.revision}/{path}"
        handle = None
        last: BaseException | None = None
        for attempt in range(self.max_attempts):
            try:
                handle = fs.open(uri, "rb", block_size=self.block_bytes)
                break
            except Exception as exc:  # noqa: BLE001 - bounded retry of the same pinned object
                last = exc
                if attempt + 1 >= self.max_attempts:
                    raise GateCPythonError(
                        f"pinned shard unavailable after bounded retries: {path}"
                    ) from last
                time.sleep(float(2**attempt))
        try:
            remaining = max_bytes
            while remaining > 0:
                block = handle.read(min(self.block_bytes, remaining))
                if not block:
                    break
                remaining -= len(block)
                yield block
        finally:
            handle.close()


class LocalShardStore(ShardStore):
    """Local-file transport for contract tests: identical framing, zero network."""

    def __init__(self, root: Path, *, block_bytes: int = 1 << 16) -> None:
        self.root = Path(root)
        self.block_bytes = block_bytes

    def _describe(self, path: str) -> RemoteShard:
        target = self.root / path
        if not target.exists():
            raise GateCPythonError(f"pinned shard is missing from the local store: {path}")
        payload = target.read_bytes()
        return RemoteShard(
            path=path,
            size=len(payload),
            lfs_sha256=_sha256_bytes(payload),
            blob_id=hashlib.sha1(payload, usedforsecurity=False).hexdigest(),
        )

    def revision_file_count(self, source: PythonSourceSpec) -> int:
        return len(sorted(self.root.glob("*.json.gz")))

    def list_shards(self, source: PythonSourceSpec) -> tuple[RemoteShard, ...]:
        return tuple(self._describe(path) for path in sorted(source.shard_paths))

    def open_prefix(self, source: PythonSourceSpec, path: str, max_bytes: int) -> Iterator[bytes]:
        with open(self.root / path, "rb") as handle:
            remaining = max_bytes
            while remaining > 0:
                block = handle.read(min(self.block_bytes, remaining))
                if not block:
                    break
                remaining -= len(block)
                yield block


def assert_shard_scope(
    source: PythonSourceSpec, live: Sequence[RemoteShard], *, revision_file_count: int
) -> None:
    """Fail closed unless the live listing reproduces the frozen Python shard scope exactly."""
    if len(source.shards) != 12:
        raise GateCPythonError("the Python shard scope must be exactly the 12 pinned shards")
    if revision_file_count != source.revision_shard_count:
        raise GateCPythonError(
            "pinned revision shard count drifted: "
            f"expected {source.revision_shard_count}, live {revision_file_count}"
        )
    live_by_path = {item.path: item for item in live}
    if set(live_by_path) != set(source.shard_paths):
        raise GateCPythonError("live listing does not contain exactly the pinned Python shards")
    for shard in source.shards:
        observed = live_by_path[shard.path]
        if observed.size != shard.size:
            raise GateCPythonError(f"pinned shard size drifted: {shard.path}")
        if observed.lfs_sha256 != shard.lfs_sha256:
            raise GateCPythonError(f"pinned shard LFS SHA-256 drifted: {shard.path}")
        if observed.blob_id != shard.blob_id:
            raise GateCPythonError(f"pinned shard blob id drifted: {shard.path}")


def _record_leaves(value: Any, prefix: str = "") -> set[str]:
    leaves: set[str] = set()
    for key, item in value.items():
        path = f"{prefix}{key}"
        if isinstance(item, dict):
            leaves |= _record_leaves(item, f"{path}.")
        else:
            leaves.add(path)
    return leaves


def assert_schema(source: PythonSourceSpec, observed: Iterable[str]) -> None:
    """Fail closed on any upstream leaf-set drift at the pinned revision."""
    observed_set = set(observed)
    frozen = set(source.frozen_leaf_set)
    if observed_set != frozen:
        added = sorted(observed_set - frozen)
        missing = sorted(frozen - observed_set)
        raise GateCPythonError(
            f"record schema drifted at the pinned revision: added={added} missing={missing}"
        )


def iter_shard_records(
    chunks: Iterable[bytes],
    *,
    start_record_index: int = 0,
    expect_complete: bool = False,
) -> Iterator[tuple[int, bytes, int, int]]:
    """Yield ``(record_index, line, consumed_compressed, produced_decompressed)`` per record.

    Only complete ``\\n``-terminated physical records are yielded; a record truncated by the
    compressed-prefix cap is never emitted, which is what makes a bounded prefix read deterministic.
    """
    decompressor = zlib.decompressobj(31)
    buffer = bytearray()
    index = 0
    pending_compressed = 0
    pending_decompressed = 0
    saw_any = False
    for chunk in chunks:
        saw_any = True
        pending_compressed += len(chunk)
        try:
            produced = decompressor.decompress(chunk, MAX_DECOMPRESS_CHUNK_BYTES)
            while decompressor.unconsumed_tail:
                produced += decompressor.decompress(
                    decompressor.unconsumed_tail, MAX_DECOMPRESS_CHUNK_BYTES
                )
        except zlib.error as exc:
            raise GateCPythonError(f"gzip stream is corrupt: {exc}") from exc
        pending_decompressed += len(produced)
        buffer += produced
        while True:
            newline = buffer.find(b"\n")
            if newline < 0:
                break
            line = bytes(buffer[:newline])
            del buffer[: newline + 1]
            current = index
            index += 1
            if current < start_record_index:
                continue
            yield current, line, pending_compressed, pending_decompressed
            pending_compressed = 0
            pending_decompressed = 0
    if expect_complete:
        if saw_any and not decompressor.eof:
            raise GateCPythonError("gzip stream ended before its terminator")
        if buffer.strip():
            raise GateCPythonError("shard ended with an unterminated JSONL record")
    if pending_compressed or pending_decompressed:
        yield -1, b"", pending_compressed, pending_decompressed


# --------------------------------------------------------------------------------------
# Frozen Python mechanical filters
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    """A single record's Gate C outcome."""

    accepted: bool
    reason: str | None = None
    detail: str | None = None
    text: str | None = None
    text_bytes: bytes | None = None
    record: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: tuple[str, ...] = ()


def _reject(reason: str, detail: str | None = None) -> Decision:
    return Decision(accepted=False, reason=reason, detail=detail)


def _get_path(record: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    current: Any = record
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _type_ok(value: Any, expected: str) -> bool:
    if expected == "str":
        return isinstance(value, str)
    if expected == "bool":
        return isinstance(value, bool)
    if expected == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "list":
        return isinstance(value, list)
    raise GateCPythonError(f"unsupported pinned JSON type: {expected}")


def comment_blank_fraction(text: str) -> float:
    """The frozen Gate E definition: line-based comment+blank share, empty text counts as 0.0.

    Reproduced verbatim from the Gate E P1b probe so C0 numbers stay comparable with the
    already-closed ``comment+blank > 0.70 ~ 0.9% of rows`` evidence.
    """
    lines = text.splitlines()
    total = len(lines) or 1
    comments = sum(1 for line in lines if line.lstrip().startswith("#"))
    blanks = sum(1 for line in lines if not line.strip())
    return (comments + blanks) / total


def _ast_features(tree: ast.AST) -> tuple[str, ...]:
    diagnostics = []
    if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree)):
        diagnostics.append("has_function")
    if any(isinstance(node, ast.ClassDef) for node in ast.walk(tree)):
        diagnostics.append("has_class")
    documented = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    if any(
        isinstance(node, documented) and ast.get_docstring(node) is not None
        for node in ast.walk(tree)
    ):
        diagnostics.append("has_docstring")
    return tuple(diagnostics)


def evaluate_record(record: Mapping[str, Any], source: PythonSourceSpec) -> Decision:
    """Apply the frozen Python mechanical filter to one parsed record."""
    if not isinstance(record, Mapping):
        return _reject("record_not_object")

    for path, expected in source.required_schema:
        present, value = _get_path(record, path)
        if not present:
            return _reject("missing_field", path)
        if value is None:
            return _reject("null_field", path)
        if not _type_ok(value, expected):
            return _reject("field_type", path)

    if record["metadata"]["language"] != source.language:
        return _reject("language_not_python")
    if record["metadata"]["is_generated"]:
        return _reject("generated")
    if record["metadata"]["is_vendor"]:
        return _reject("vendor")

    text = record["text"]
    encoded = _strict_utf8_bytes(text)
    if encoded is None:
        return _reject("strict_utf8")
    if len(encoded) < source.min_bytes:
        return _reject("size_band_short")
    if len(encoded) > source.max_bytes:
        return _reject("size_band_long")

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _reject("ast_parse")
    except (ValueError, MemoryError, RecursionError):
        return _reject("ast_parse_hostile")

    if _pathological_repetition(text):
        return _reject("repetition")

    fraction = comment_blank_fraction(text)
    if fraction > source.max_comment_blank_fraction:
        return _reject("comment_blank_fraction")

    metadata: dict[str, Any] = {}
    for path in RETAINED_METADATA_PATHS:
        present, value = _get_path(record, path)
        metadata[path] = value if present else None
    metadata["derived.comment_blank_fraction"] = fraction
    metadata["derived.line_count"] = len(text.splitlines())

    return Decision(
        accepted=True,
        text=text,
        text_bytes=encoded,
        record=record,
        metadata=metadata,
        diagnostics=_ast_features(tree),
    )


# --------------------------------------------------------------------------------------
# Configuration validation, fingerprint and checkpoint
# --------------------------------------------------------------------------------------


def _validate_config(config: BuildConfig) -> None:
    source = config.source
    _ensure_revision(source.revision)
    if len(source.shards) != 12:
        raise GateCPythonError("the Python shard scope must be exactly the 12 pinned shards")
    if len(set(source.shard_paths)) != len(source.shards):
        raise GateCPythonError("the pinned Python shard list contains a duplicate path")
    if not 1 <= config.target_documents <= MAX_ACCEPTED_DOCUMENTS:
        raise GateCPythonError(f"target_documents must be in [1, {MAX_ACCEPTED_DOCUMENTS}]")
    if not config.target_documents <= config.max_scanned <= MAX_SCANNED_RECORDS:
        raise GateCPythonError(f"max_scanned must be in [target_documents, {MAX_SCANNED_RECORDS}]")
    if not 1 <= config.max_shard_records <= config.max_scanned:
        raise GateCPythonError("max_shard_records must be in [1, max_scanned]")
    if not 1 <= config.max_shard_compressed_bytes <= MAX_SHARD_COMPRESSED_BYTES:
        raise GateCPythonError(
            f"max_shard_compressed_bytes must be in [1, {MAX_SHARD_COMPRESSED_BYTES}]"
        )
    if not 1 <= config.max_response_bytes <= MAX_RESPONSE_BYTES:
        raise GateCPythonError(f"max_response_bytes must be in [1, {MAX_RESPONSE_BYTES}]")
    if (
        not math.isfinite(config.max_wall_seconds)
        or not 0 < config.max_wall_seconds <= MAX_WALL_SECONDS
    ):
        raise GateCPythonError(f"max_wall_seconds must be in (0, {MAX_WALL_SECONDS}]")
    if not 1 <= config.stride <= MAX_STRIDE:
        raise GateCPythonError(f"stride must be in [1, {MAX_STRIDE}]")
    if type(config.seed) is not int or config.seed < 0:
        raise GateCPythonError("seed must be a non-negative integer")
    if config.stop_after_documents is not None and not (
        1 <= config.stop_after_documents <= config.target_documents
    ):
        raise GateCPythonError("stop_after_documents must be in [1, target_documents]")
    if not 1 <= config.checkpoint_every <= MAX_ACCEPTED_DOCUMENTS:
        raise GateCPythonError(f"checkpoint_every must be in [1, {MAX_ACCEPTED_DOCUMENTS}]")
    for path in (config.output_dir, config.work_dir, _staging_path(config.output_dir)):
        _ensure_git_ignored(path)


def run_fingerprint(config: BuildConfig) -> str:
    """Bind every semantic build input; operational stop caps and cadence are excluded."""
    source = config.source
    return _sha256_bytes(
        _canonical_json_bytes({
            "tool_schema_version": TOOL_SCHEMA_VERSION,
            "spec_version": SPEC_VERSION,
            "source": source.to_json(),
            "required_schema": source.required_schema_map,
            "frozen_leaf_set": list(source.frozen_leaf_set),
            "filter_order": list(FILTER_ORDER),
            "target_documents": config.target_documents,
            "max_scanned": config.max_scanned,
            "max_shard_records": config.max_shard_records,
            "max_shard_compressed_bytes": config.max_shard_compressed_bytes,
            "stride": config.stride,
            "seed": config.seed,
            "output_dir": str(_ensure_under_workspace(config.output_dir)),
        })
    )


def _new_checkpoint(fingerprint: str) -> dict[str, Any]:
    return {
        "tool_schema_version": TOOL_SCHEMA_VERSION,
        "run_fingerprint": fingerprint,
        "next_shard_index": 0,
        "next_record_index": 0,
        "counters": BuildCounters().to_json(),
        "per_shard": {},
        "seen_record_ids": [],
        "seen_text_sha256": [],
        "documents_sha256": _sha256_bytes(b""),
        "documents_bytes": 0,
        "resume_count": 0,
        "completed": False,
        "live_schema": [],
        "shard_identity": [],
        "updated_at": _utc_now(),
    }


def _checkpoint_payload(state: Mapping[str, Any]) -> bytes:
    return _canonical_json_bytes({k: v for k, v in state.items() if k != "checksum"})


def _write_checkpoint(path: Path, state: Mapping[str, Any]) -> None:
    body = {key: value for key, value in state.items() if key != "checksum"}
    body["checksum"] = _sha256_bytes(_checkpoint_payload(body))
    _atomic_replace_bytes(path, json.dumps(body, indent=2, sort_keys=True).encode() + b"\n")


def _read_checkpoint(path: Path, fingerprint: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise GateCPythonError(f"checkpoint is unreadable: {path}") from exc
    try:
        state = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateCPythonError("checkpoint is corrupted: not strict UTF-8 JSON") from exc
    if not isinstance(state, dict):
        raise GateCPythonError("checkpoint is corrupted: not an object")
    checksum = state.get("checksum")
    if not isinstance(checksum, str) or checksum != _sha256_bytes(_checkpoint_payload(state)):
        raise GateCPythonError("checkpoint checksum mismatch; refusing to resume")
    if state.get("tool_schema_version") != TOOL_SCHEMA_VERSION:
        raise GateCPythonError("checkpoint tool schema version mismatch")
    if state.get("run_fingerprint") != fingerprint:
        raise GateCPythonError("checkpoint run fingerprint mismatch; refusing to resume")
    for name in ("next_shard_index", "next_record_index", "documents_bytes", "resume_count"):
        _require_nonnegative_int(state.get(name), name)
    for name in ("seen_record_ids", "seen_text_sha256", "live_schema", "shard_identity"):
        if not isinstance(state.get(name), list):
            raise GateCPythonError(f"checkpoint field is invalid: {name}")
    if not isinstance(state.get("documents_sha256"), str):
        raise GateCPythonError("checkpoint field is invalid: documents_sha256")
    if not isinstance(state.get("completed"), bool):
        raise GateCPythonError("checkpoint field is invalid: completed")
    if not isinstance(state.get("per_shard"), Mapping):
        raise GateCPythonError("checkpoint field is invalid: per_shard")
    BuildCounters.from_json(state.get("counters") or {})
    return state


def _restore_documents_to_checkpoint(path: Path, state: Mapping[str, Any]) -> None:
    """Restore ``documents.jsonl`` to exactly the committed checkpoint prefix.

    Records written after the last checkpoint are uncommitted by definition; dropping them is what
    makes an interruption before a checkpoint safe and duplicate-free on resume.
    """
    expected_bytes = int(state["documents_bytes"])
    expected_sha = str(state["documents_sha256"])
    if not path.exists():
        if expected_bytes != 0:
            raise GateCPythonError(
                "checkpoint references accepted documents but the file is missing"
            )
        _write_new_file(path, b"")
        return
    actual = path.read_bytes()
    if len(actual) < expected_bytes:
        raise GateCPythonError(
            "accepted-document file is shorter than its committed checkpoint prefix"
        )
    if _sha256_bytes(actual[:expected_bytes]) != expected_sha:
        raise GateCPythonError(
            "accepted-document prefix does not match the committed checkpoint hash"
        )
    if len(actual) != expected_bytes:
        with open(path, "r+b") as handle:
            handle.truncate(expected_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)


# --------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------


def _make_manifest(
    config: BuildConfig,
    state: Mapping[str, Any],
    counters: BuildCounters,
    *,
    fingerprint: str,
    stop_reason: str,
    wall_seconds: float,
    documents_sha256: str,
    documents_bytes: int,
) -> dict[str, Any]:
    source = config.source
    per_shard = state.get("per_shard") or {}
    covered = sum(1 for entry in per_shard.values() if int(entry.get("scanned", 0)) > 0)
    return {
        "tool_schema_version": TOOL_SCHEMA_VERSION,
        "spec_version": SPEC_VERSION,
        "generated_at": _utc_now(),
        # Nothing this builder produces is ever promotable, and the release says so itself rather
        # than leaving promotability to be inferred from the stop reason.
        "release_kind": "c0_diagnostic",
        "promotion_eligible": False,
        "production_candidate_quota_authorized": False,
        "full_python_candidate_build_authorized": False,
        "promotion_eligible_rationale": (
            "This tool implements bounded C0 diagnostics only. Its per-shard compressed-prefix "
            "ceiling makes a full read of the 11.72 GB pinned Python corpus structurally "
            "impossible, and no canonical-token accounting exists at Gate C, so no release it "
            "publishes can satisfy the frozen 0.60B-token Python quota."
        ),
        "run_fingerprint": fingerprint,
        "source": source.to_json(),
        "required_schema": source.required_schema_map,
        "nullable_paths": list(source.nullable_paths),
        "live_schema": sorted(state.get("live_schema") or []),
        "schema_verified": bool(state.get("live_schema")),
        "shard_identity_verified": bool(state.get("shard_identity")),
        "shard_identity": list(state.get("shard_identity") or []),
        "traversal": {
            "method": "bounded_deterministic_strided_per_shard_prefix",
            "description": (
                "Shards are traversed in ascending pinned path order. Each shard is streamed from "
                "compressed offset 0 (a single-member gzip stream cannot be seeked), physical "
                "JSONL records are counted from 0, and only records with "
                "record_index % stride == 0 are parsed and filtered. Each shard stops at the first "
                "of its evaluated-record cap or its compressed-prefix cap."
            ),
            "stride": config.stride,
            "row_level_representative_sampler": False,
            "representativeness_caveat": (
                "This is a bounded head window of each shard, not a row-level representative "
                "sample of the Python corpus. Yields and distributions are honest for the traversed "
                "window only."
            ),
        },
        "filter_order": list(FILTER_ORDER),
        "filter_parameters": {
            "language": source.language,
            "min_bytes": source.min_bytes,
            "max_bytes": source.max_bytes,
            "max_comment_blank_fraction": source.max_comment_blank_fraction,
            "comment_blank_fraction_definition": (
                "(lines whose lstrip() starts with '#' + blank lines) / max(len(splitlines()), 1)"
            ),
            "ast": "python3 ast.parse",
            "exact_text_dedup": "sha256 of the strict UTF-8 document text",
            "record_identity": source.natural_id_path,
        },
        "caps": {
            "target_documents": config.target_documents,
            "max_scanned": config.max_scanned,
            "max_shard_records": config.max_shard_records,
            "max_shard_compressed_bytes": config.max_shard_compressed_bytes,
            "max_response_bytes": config.max_response_bytes,
            "max_wall_seconds": config.max_wall_seconds,
            "checkpoint_every": config.checkpoint_every,
        },
        "c0_ceilings": {
            "max_accepted_documents": MAX_ACCEPTED_DOCUMENTS,
            "max_scanned_records": MAX_SCANNED_RECORDS,
            "max_shard_compressed_bytes": MAX_SHARD_COMPRESSED_BYTES,
            "pinned_python_compressed_bytes": sum(shard.size for shard in source.shards),
        },
        "seed": config.seed,
        "accounting": counters.to_json(),
        "yield_rate": (counters.accepted / counters.scanned) if counters.scanned else None,
        "per_shard": {key: per_shard[key] for key in sorted(per_shard)},
        "shards_covered": covered,
        "shards_pinned": len(source.shards),
        "stop_reason": stop_reason,
        "next_shard_index": state["next_shard_index"],
        "next_record_index": state["next_record_index"],
        "completed": bool(state["completed"]),
        "resume_count": _require_nonnegative_int(state.get("resume_count"), "resume_count"),
        "wall_seconds": round(wall_seconds, 3),
        "documents_file": DOCUMENTS_NAME,
        "documents_sha256": documents_sha256,
        "documents_bytes": documents_bytes,
        "provisional_byte_weighted_only": True,
        "canonical_token_split_performed": False,
        "gate_c_scope": {
            "chat_conversion": False,
            "textual_document_separator": False,
            "tokenizer_counting": False,
            "bos_eos_inserted": False,
            "document_truncation": False,
            "repository_aware_split": False,
            "cross_source_near_dedup": False,
            "benchmark_decontamination": False,
            "reference_reserve_exclusion": False,
            "stage_a_stage_b_split_performed": False,
        },
        "hard_stops": {
            "bulk_candidate_quota_started": False,
            "tokenizer_trained": False,
            "gate_r_started": False,
            "final_shards_built": False,
            "model_training_started": False,
        },
    }


# --------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------


def _new_shard_entry() -> dict[str, Any]:
    return {
        "scanned": 0,
        "accepted": 0,
        "rejected": 0,
        "accepted_text_bytes": 0,
        "records_seen": 0,
        "stride_skipped": 0,
        "compressed_bytes": 0,
        "decompressed_bytes": 0,
        "stop_reason": None,
        "completed": False,
    }


def build_candidates(
    config: BuildConfig,
    *,
    store: ShardStore | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Scan, filter, checkpoint and atomically publish one bounded Python C0 diagnostic release."""
    _validate_config(config)
    source = config.source
    store = store if store is not None else HubShardStore()

    output_dir = _ensure_under_workspace(config.output_dir)
    if output_dir.exists():
        raise GateCPythonError(f"refusing to overwrite published output: {output_dir}")
    work_dir = _ensure_under_workspace(config.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = run_fingerprint(config)
    checkpoint_path = work_dir / CHECKPOINT_NAME
    documents_path = work_dir / DOCUMENTS_NAME

    resumed = checkpoint_path.exists()
    if resumed:
        state = _read_checkpoint(checkpoint_path, fingerprint)
        state["resume_count"] = int(state["resume_count"]) + 1
    else:
        state = _new_checkpoint(fingerprint)
    _restore_documents_to_checkpoint(documents_path, state)

    live = store.list_shards(source)
    assert_shard_scope(source, live, revision_file_count=store.revision_file_count(source))
    state["shard_identity"] = [
        {
            "path": item.path,
            "size": item.size,
            "lfs_sha256": item.lfs_sha256,
            "blob_id": item.blob_id,
        }
        for item in sorted(live, key=lambda entry: entry.path)
    ]

    counters = BuildCounters.from_json(state["counters"])
    per_shard: dict[str, dict[str, Any]] = {
        key: dict(value) for key, value in (state.get("per_shard") or {}).items()
    }
    seen_record_ids = set(state["seen_record_ids"])
    seen_text_hashes = set(state["seen_text_sha256"])
    live_schema: list[str] = list(state.get("live_schema") or [])
    documents_digest = hashlib.sha256(documents_path.read_bytes())
    documents_bytes = int(state["documents_bytes"])
    next_shard_index = int(state["next_shard_index"])
    next_record_index = int(state["next_record_index"])
    completed = bool(state["completed"])

    start = clock()
    handle = open(documents_path, "ab")

    def commit(is_complete: bool) -> None:
        handle.flush()
        os.fsync(handle.fileno())
        state["next_shard_index"] = next_shard_index
        state["next_record_index"] = next_record_index
        state["counters"] = counters.to_json()
        state["per_shard"] = per_shard
        state["seen_record_ids"] = sorted(seen_record_ids)
        state["seen_text_sha256"] = sorted(seen_text_hashes)
        state["documents_sha256"] = documents_digest.hexdigest()
        state["documents_bytes"] = documents_bytes
        state["completed"] = is_complete
        state["live_schema"] = sorted(live_schema)
        state["updated_at"] = _utc_now()
        _write_checkpoint(checkpoint_path, state)

    try:
        if counters.accepted >= config.target_documents:
            stop_reason = "target_reached"
        elif completed:
            stop_reason = "all_shard_windows_completed"
        else:
            stop_reason = "all_shard_windows_completed"
            since_checkpoint = 0
            stop = False
            for shard_index in range(next_shard_index, len(source.shards)):
                if shard_index != next_shard_index:
                    raise GateCPythonError("shard cursor desynchronized from the traversal order")
                shard = source.shards[shard_index]
                entry = per_shard.setdefault(shard.path, _new_shard_entry())
                resume_offset = next_record_index
                budget = min(
                    config.max_shard_compressed_bytes,
                    max(0, config.max_response_bytes - counters.compressed_bytes),
                )
                if budget <= 0:
                    stop_reason = "byte_cap"
                    break
                shard_stop: str | None = None
                schema_checked = False
                first_yield = True
                stream = iter_shard_records(
                    store.open_prefix(source, shard.path, budget),
                    start_record_index=resume_offset,
                    expect_complete=budget >= shard.size,
                )
                for record_index, line, used_compressed, used_decompressed in stream:
                    counters.compressed_bytes += used_compressed
                    counters.decompressed_bytes += used_decompressed
                    entry["compressed_bytes"] += used_compressed
                    entry["decompressed_bytes"] += used_decompressed
                    if first_yield and resume_offset > 0 and record_index >= 0:
                        # Everything consumed to reach the checkpointed cursor is a re-read: a
                        # single-member gzip stream cannot be seeked, so resume pays for the prefix.
                        counters.resume_reread_compressed_bytes += used_compressed
                    first_yield = False
                    if record_index < 0:
                        break
                    if record_index < next_record_index:
                        raise GateCPythonError(
                            "transport produced a record before the checkpointed cursor"
                        )
                    next_record_index = record_index + 1
                    entry["records_seen"] = max(entry["records_seen"], next_record_index)

                    if counters.compressed_bytes > config.max_response_bytes:
                        shard_stop = "byte_cap"
                        stop_reason = "byte_cap"
                        stop = True
                        break
                    if clock() - start > config.max_wall_seconds:
                        shard_stop = "time_cap"
                        stop_reason = "time_cap"
                        stop = True
                        break
                    if record_index % config.stride != 0:
                        counters.stride_skipped += 1
                        entry["stride_skipped"] += 1
                        continue

                    counters.scanned += 1
                    entry["scanned"] += 1
                    decision = _evaluate_line(
                        line, source, live_schema, check_schema=not schema_checked
                    )
                    # A record that never parsed cannot have been schema checked, so the check
                    # stays armed until one record of this shard actually reaches it.
                    if decision.reason not in _PRE_SCHEMA_REJECTS:
                        schema_checked = True
                    if not decision.accepted:
                        counters.rejected += 1
                        entry["rejected"] += 1
                        reason = decision.reason or "unspecified"
                        counters.rejections[reason] += 1
                        if decision.detail:
                            counters.field_violations[f"{reason}:{decision.detail}"] += 1
                    else:
                        record = decision.record or {}
                        natural_id = str(record[source.natural_id_path])
                        record_id = f"{source.key}:{natural_id}"
                        encoded = decision.text_bytes or b""
                        text_sha = _sha256_bytes(encoded)
                        if record_id in seen_record_ids:
                            counters.rejected += 1
                            entry["rejected"] += 1
                            counters.rejections["duplicate_source_record_id"] += 1
                        elif text_sha in seen_text_hashes:
                            counters.rejected += 1
                            entry["rejected"] += 1
                            counters.rejections["duplicate_text_sha256"] += 1
                        else:
                            seen_record_ids.add(record_id)
                            seen_text_hashes.add(text_sha)
                            line_bytes = canonical_jsonl_record_bytes({
                                "source_key": source.key,
                                "source_record_id": record_id,
                                "natural_id": natural_id,
                                "text": decision.text,
                                "text_sha256": text_sha,
                                "text_bytes": len(encoded),
                                "shard_index": shard_index,
                                "shard_path": shard.path,
                                "record_index": record_index,
                                "metadata": dict(decision.metadata),
                                "diagnostics": list(decision.diagnostics),
                                "provenance": {
                                    "dataset": source.dataset,
                                    "config": source.dataset_config,
                                    "split": source.split,
                                    "revision": source.revision,
                                    "shard": shard.path,
                                    "shard_lfs_sha256": shard.lfs_sha256,
                                    "license": _get_path(record, "metadata.license")[1],
                                    "transport": "huggingface_hub_gzip_jsonl_prefix",
                                },
                            })
                            handle.write(line_bytes)
                            documents_digest.update(line_bytes)
                            documents_bytes += len(line_bytes)
                            counters.accepted += 1
                            entry["accepted"] += 1
                            counters.accepted_text_bytes += len(encoded)
                            entry["accepted_text_bytes"] += len(encoded)
                            for name in decision.diagnostics:
                                counters.diagnostics[name] += 1
                            since_checkpoint += 1

                    if counters.accepted >= config.target_documents:
                        shard_stop = "target_reached"
                        stop_reason = "target_reached"
                        stop = True
                        break
                    if config.stop_after_documents is not None and (
                        counters.accepted >= config.stop_after_documents
                    ):
                        shard_stop = "stop_after_documents"
                        stop_reason = "stop_after_documents"
                        stop = True
                        break
                    if counters.scanned >= config.max_scanned:
                        shard_stop = "scan_cap"
                        stop_reason = "scan_cap"
                        stop = True
                        break
                    if entry["scanned"] >= config.max_shard_records:
                        shard_stop = "shard_record_cap"
                        break
                    if since_checkpoint >= config.checkpoint_every:
                        commit(False)
                        since_checkpoint = 0
                stream.close()

                if shard_stop is None:
                    shard_stop = (
                        "shard_compressed_cap"
                        if entry["compressed_bytes"] >= budget
                        else "shard_exhausted"
                    )
                entry["stop_reason"] = shard_stop
                if stop:
                    break
                entry["completed"] = True
                next_shard_index = shard_index + 1
                next_record_index = 0
            else:
                completed = True
        commit(completed)
    finally:
        handle.close()

    if stop_reason == "stop_after_documents":
        return {
            "published": False,
            "stop_reason": stop_reason,
            "work_dir": str(work_dir),
            "accepted": counters.accepted,
            "scanned": counters.scanned,
            "rejected": counters.rejected,
            "compressed_bytes": counters.compressed_bytes,
            "next_shard_index": next_shard_index,
            "next_record_index": next_record_index,
            "resume_count": int(state["resume_count"]),
            "resumed": resumed,
            "documents_sha256": documents_digest.hexdigest(),
        }

    wall_seconds = clock() - start
    manifest = _make_manifest(
        config,
        state,
        counters,
        fingerprint=fingerprint,
        stop_reason=stop_reason,
        wall_seconds=wall_seconds,
        documents_sha256=documents_digest.hexdigest(),
        documents_bytes=documents_bytes,
    )

    staging = _staging_path(output_dir)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copyfile(documents_path, staging / DOCUMENTS_NAME)
    _write_new_file(
        staging / MANIFEST_NAME, json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    )
    _write_new_file(
        staging / CHECKSUMS_NAME,
        "".join(
            f"{_sha256_bytes((staging / name).read_bytes())}  {name}\n"
            for name in (DOCUMENTS_NAME, MANIFEST_NAME)
        ).encode(),
    )
    _publish_directory(staging, output_dir)

    published_sha = _sha256_bytes((output_dir / DOCUMENTS_NAME).read_bytes())
    if published_sha != manifest["documents_sha256"]:
        raise GateCPythonError("published document file does not match its manifest checksum")

    return {
        "published": True,
        "output_dir": str(output_dir),
        "release_kind": "c0_diagnostic",
        "promotion_eligible": False,
        "stop_reason": stop_reason,
        "accepted": counters.accepted,
        "scanned": counters.scanned,
        "rejected": counters.rejected,
        "accepted_text_bytes": counters.accepted_text_bytes,
        "compressed_bytes": counters.compressed_bytes,
        "decompressed_bytes": counters.decompressed_bytes,
        "shards_covered": manifest["shards_covered"],
        "wall_seconds": round(wall_seconds, 3),
        "resume_count": int(state["resume_count"]),
        "resumed": resumed,
        "next_shard_index": next_shard_index,
        "next_record_index": next_record_index,
        "documents_sha256": published_sha,
        "manifest_sha256": _sha256_bytes((output_dir / MANIFEST_NAME).read_bytes()),
        "rejections": dict(sorted(counters.rejections.items())),
    }


def _evaluate_line(
    line: bytes,
    source: PythonSourceSpec,
    live_schema: list[str],
    *,
    check_schema: bool,
) -> Decision:
    """Decode one physical JSONL record and apply the frozen filter.

    ``check_schema`` is set for the first evaluated record of every shard: the complete leaf set
    is compared with the frozen record shape and any drift fails the build closed.  Per-record
    field problems inside an otherwise-unchanged shape stay controlled rejects.
    """
    try:
        decoded = line.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return _reject("line_not_utf8")
    try:
        record = json.loads(decoded)
    except json.JSONDecodeError:
        return _reject("malformed_json")
    if not isinstance(record, dict):
        return _reject("record_not_object")
    if check_schema:
        leaves = _record_leaves(record)
        assert_schema(source, leaves)
        if not live_schema:
            live_schema.extend(sorted(leaves))
    return evaluate_record(record, source)


# --------------------------------------------------------------------------------------
# Release verification
# --------------------------------------------------------------------------------------


def verify_release(output_dir: Path) -> dict[str, Any]:
    """Re-read a published release and verify every recorded checksum and invariant."""
    output_dir = _ensure_under_workspace(output_dir)
    manifest_bytes = (output_dir / MANIFEST_NAME).read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8", errors="strict"))
    if manifest.get("tool_schema_version") != TOOL_SCHEMA_VERSION:
        raise GateCPythonError("release was not produced by this tool schema version")
    if manifest.get("release_kind") != "c0_diagnostic":
        raise GateCPythonError("release_kind must be c0_diagnostic")
    if manifest.get("promotion_eligible") is not False:
        raise GateCPythonError("a C0 diagnostic release can never be promotion eligible")
    if manifest.get("production_candidate_quota_authorized") is not False:
        raise GateCPythonError("production candidate quota is not authorized")
    if manifest.get("canonical_token_split_performed") is not False:
        raise GateCPythonError("no canonical token split may be claimed at Gate C")
    documents_bytes = (output_dir / DOCUMENTS_NAME).read_bytes()
    documents_sha = _sha256_bytes(documents_bytes)
    if documents_sha != manifest["documents_sha256"]:
        raise GateCPythonError("documents.jsonl does not match manifest documents_sha256")
    if len(documents_bytes) != manifest["documents_bytes"]:
        raise GateCPythonError("documents.jsonl length does not match the manifest")
    record_ids: set[str] = set()
    text_hashes: set[str] = set()
    shards: Counter = Counter()
    accepted = 0
    accepted_text_bytes = 0
    byte_lines = [line for line in documents_bytes.split(b"\n") if line]
    str_lines = [line for line in documents_bytes.decode("utf-8").splitlines() if line]
    if len(byte_lines) != len(str_lines):
        raise GateCPythonError(
            f"physical JSONL framing is ambiguous: {len(byte_lines)} byte-delimited records vs "
            f"{len(str_lines)} str.splitlines() records"
        )
    pinned = {entry["path"] for entry in manifest["source"]["shards"]}
    for line in byte_lines:
        record = json.loads(line.decode("utf-8", errors="strict"))
        encoded = record["text"].encode("utf-8")
        if _sha256_bytes(encoded) != record["text_sha256"]:
            raise GateCPythonError("candidate record text_sha256 was tampered with")
        if len(encoded) != record["text_bytes"]:
            raise GateCPythonError("candidate record text_bytes does not match its text")
        if record["source_record_id"] in record_ids:
            raise GateCPythonError("published release contains a duplicate source record id")
        if record["text_sha256"] in text_hashes:
            raise GateCPythonError("published release contains a duplicate text hash")
        if record["shard_path"] not in pinned:
            raise GateCPythonError("published release references an unpinned shard")
        if record["metadata"]["metadata.language"] != manifest["source"]["language"]:
            raise GateCPythonError("published release contains a non-Python document")
        record_ids.add(record["source_record_id"])
        text_hashes.add(record["text_sha256"])
        shards[record["shard_path"]] += 1
        accepted += 1
        accepted_text_bytes += len(encoded)
    if accepted != manifest["accounting"]["accepted"]:
        raise GateCPythonError("published record count does not match the manifest accounting")
    if accepted_text_bytes != manifest["accounting"]["accepted_text_bytes"]:
        raise GateCPythonError("published text bytes do not match the manifest accounting")
    for entry in (output_dir / CHECKSUMS_NAME).read_text().splitlines():
        digest, name = entry.split("  ", 1)
        if _sha256_bytes((output_dir / name).read_bytes()) != digest:
            raise GateCPythonError(f"MANIFEST.sha256 mismatch for {name}")
    return {
        "output_dir": str(output_dir),
        "release_kind": manifest["release_kind"],
        "promotion_eligible": manifest["promotion_eligible"],
        "accepted": accepted,
        "accepted_text_bytes": accepted_text_bytes,
        "shards_covered": len(shards),
        "documents_sha256": documents_sha,
        "manifest_sha256": _sha256_bytes(manifest_bytes),
    }


# --------------------------------------------------------------------------------------
# Score / distribution diagnostics over an immutable release
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


def _histogram(values: Iterable[float], *, width: float) -> dict[str, int]:
    counts: Counter = Counter()
    for value in values:
        counts[f"{math.floor(value / width) * width:.4f}"] += 1
    return dict(sorted(counts.items(), key=lambda item: float(item[0])))


def _top(counter: Counter, limit: int = 20) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit])


def diagnose_release(output_dir: Path) -> dict[str, Any]:
    """Compute the C0 score / length / provenance diagnostics from an immutable release.

    The Stage A / Stage B cutoff reported here is **byte weighted and provisional**.  The canonical
    tokenizer does not exist at Gate C, so the final canonical-token split cannot be, and is not,
    performed.
    """
    verified = verify_release(output_dir)
    output_dir = _ensure_under_workspace(output_dir)
    manifest_bytes = (output_dir / MANIFEST_NAME).read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8", errors="strict"))
    documents_bytes = (output_dir / DOCUMENTS_NAME).read_bytes()

    entries: list[tuple[float, int, str]] = []
    lengths: list[int] = []
    scores: list[float] = []
    int_scores: Counter = Counter()
    fractions: list[float] = []
    lines_counts: list[int] = []
    licenses: Counter = Counter()
    license_types: Counter = Counter()
    repos: Counter = Counter()
    shards: Counter = Counter()
    shard_bytes: Counter = Counter()
    diagnostics: Counter = Counter()
    repo_present = 0
    license_present = 0
    detected_present = 0
    total = 0

    for line in documents_bytes.split(b"\n"):
        if not line:
            continue
        record = json.loads(line.decode("utf-8", errors="strict"))
        metadata = record["metadata"]
        total += 1
        text_bytes = int(record["text_bytes"])
        score = float(metadata["score"])
        entries.append((score, text_bytes, record["text_sha256"]))
        lengths.append(text_bytes)
        scores.append(score)
        int_scores[str(metadata["int_score"])] += 1
        fractions.append(float(metadata["derived.comment_blank_fraction"]))
        lines_counts.append(int(metadata["derived.line_count"]))
        repo = metadata.get("metadata.repo_name")
        if isinstance(repo, str) and repo:
            repo_present += 1
            repos[repo] += 1
        licence = metadata.get("metadata.license")
        if isinstance(licence, str) and licence:
            license_present += 1
            licenses[licence] += 1
        license_type = metadata.get("metadata.license_type")
        if isinstance(license_type, str) and license_type:
            license_types[license_type] += 1
        detected = metadata.get("metadata.detected_licenses")
        if isinstance(detected, list) and detected:
            detected_present += 1
        shards[record["shard_path"]] += 1
        shard_bytes[record["shard_path"]] += text_bytes
        for name in record.get("diagnostics") or []:
            diagnostics[name] += 1

    total_bytes = sum(lengths)
    # Frozen ranking rule: descending continuous score, SHA-256 ascending as the stable tie-break.
    ranked = sorted(entries, key=lambda item: (-item[0], item[2]))
    target = STAGE_B_MASS_SHARE * total_bytes
    cumulative = 0
    cutoff_score: float | None = None
    included_docs = 0
    for score, text_bytes, _ in ranked:
        cumulative += text_bytes
        included_docs += 1
        cutoff_score = score
        if cumulative >= target:
            break
    strictly_above = sum(
        1 for score, _, _ in ranked if cutoff_score is not None and score > cutoff_score
    )
    strictly_above_bytes = sum(
        text_bytes
        for score, text_bytes, _ in ranked
        if cutoff_score is not None and score > cutoff_score
    )

    return {
        "tool_schema_version": TOOL_SCHEMA_VERSION,
        "spec_version": SPEC_VERSION,
        "generated_at": _utc_now(),
        "release": {
            "output_dir": str(output_dir),
            "release_kind": manifest["release_kind"],
            "promotion_eligible": manifest["promotion_eligible"],
            "documents_sha256": verified["documents_sha256"],
            "manifest_sha256": verified["manifest_sha256"],
            "revision": manifest["source"]["revision"],
            "run_fingerprint": manifest["run_fingerprint"],
        },
        "documents": total,
        "accepted_text_bytes": total_bytes,
        "length_bytes": _quantiles(lengths),
        "length_histogram_256b": _histogram(lengths, width=256.0),
        "line_count": _quantiles(lines_counts),
        "comment_blank_fraction": _quantiles(fractions),
        "score": _quantiles(scores),
        "score_histogram_0p25": _histogram(scores, width=0.25),
        "int_score_histogram": dict(sorted(int_scores.items())),
        "provisional_stage_b_cutoff": {
            "provisional_byte_weighted_only": True,
            "canonical_token_split_performed": False,
            "stage_b_mass_share": STAGE_B_MASS_SHARE,
            "rule": "rank by descending score, SHA-256 ascending tie-break, accumulate text bytes",
            "target_bytes": target,
            "cutoff_score": cutoff_score,
            "included_documents": included_docs,
            "included_bytes": cumulative,
            "included_byte_share": (cumulative / total_bytes) if total_bytes else None,
            "documents_strictly_above_cutoff": strictly_above,
            "bytes_strictly_above_cutoff": strictly_above_bytes,
            "tie_mass_at_cutoff_bytes": cumulative - strictly_above_bytes,
            "caveat": (
                "Byte mass is not canonical token mass. This cutoff is a Gate C diagnostic only "
                "and must be recomputed on canonical tokens once the tokenizer release exists."
            ),
        },
        "provenance_availability": {
            "repo_name_present": repo_present,
            "repo_name_present_rate": (repo_present / total) if total else None,
            "distinct_repos": len(repos),
            "top_repo_share": (max(repos.values()) / total) if total and repos else None,
            "license_present": license_present,
            "license_present_rate": (license_present / total) if total else None,
            "detected_licenses_present": detected_present,
            "detected_licenses_present_rate": (detected_present / total) if total else None,
        },
        "license_histogram": _top(licenses),
        "license_type_histogram": dict(sorted(license_types.items())),
        "top_repos": _top(repos),
        "shard_coverage": {
            "documents": dict(sorted(shards.items())),
            "accepted_text_bytes": dict(sorted(shard_bytes.items())),
            "shards_covered": len(shards),
            "shards_pinned": manifest["shards_pinned"],
        },
        "ast_diagnostics": {
            "has_function": diagnostics.get("has_function", 0),
            "has_function_share": (diagnostics.get("has_function", 0) / total) if total else None,
            "has_class": diagnostics.get("has_class", 0),
            "has_docstring": diagnostics.get("has_docstring", 0),
            "has_docstring_share": (diagnostics.get("has_docstring", 0) / total) if total else None,
        },
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PetitGPT Common Pile Python Gate C C0 diagnostic builder"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build one bounded C0 diagnostic release")
    build.add_argument("--output-dir", required=True, type=Path)
    build.add_argument("--work-dir", required=True, type=Path)
    build.add_argument("--target-documents", required=True, type=int)
    build.add_argument("--max-scanned", required=True, type=int)
    build.add_argument("--max-shard-records", required=True, type=int)
    build.add_argument("--max-shard-compressed-bytes", required=True, type=int)
    build.add_argument("--max-response-bytes", required=True, type=int)
    build.add_argument("--max-wall-seconds", required=True, type=float)
    build.add_argument("--stride", required=True, type=int)
    build.add_argument("--seed", required=True, type=int)
    build.add_argument("--stop-after-documents", type=int, default=None)
    build.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)

    verify = subparsers.add_parser("verify", help="verify a published C0 diagnostic release")
    verify.add_argument("--output-dir", required=True, type=Path)

    diagnose = subparsers.add_parser(
        "diagnose", help="compute score/length/provenance diagnostics for a published release"
    )
    diagnose.add_argument("--output-dir", required=True, type=Path)
    diagnose.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "build":
            result = build_candidates(
                BuildConfig(
                    source=PYTHON_SOURCE,
                    output_dir=args.output_dir,
                    work_dir=args.work_dir,
                    target_documents=args.target_documents,
                    max_scanned=args.max_scanned,
                    max_shard_records=args.max_shard_records,
                    max_shard_compressed_bytes=args.max_shard_compressed_bytes,
                    max_response_bytes=args.max_response_bytes,
                    max_wall_seconds=args.max_wall_seconds,
                    stride=args.stride,
                    seed=args.seed,
                    stop_after_documents=args.stop_after_documents,
                    checkpoint_every=args.checkpoint_every,
                )
            )
        elif args.command == "verify":
            result = verify_release(args.output_dir)
        else:
            diagnostics = diagnose_release(args.output_dir)
            out = _ensure_under_workspace(args.out)
            _ensure_git_ignored(out)
            if out.exists():
                raise GateCPythonError(f"refusing to overwrite diagnostics output: {out}")
            out.parent.mkdir(parents=True, exist_ok=True)
            _write_new_file(out, json.dumps(diagnostics, indent=2, sort_keys=True).encode() + b"\n")
            result = {
                "out": str(out),
                "sha256": _sha256_bytes(out.read_bytes()),
                "documents": diagnostics["documents"],
                "provisional_byte_weighted_only": True,
                "canonical_token_split_performed": False,
            }
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
