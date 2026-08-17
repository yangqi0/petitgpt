#!/usr/bin/env python3

"""Full Python Gate C candidate builder for Common Pile ``stackv2_edu_filtered``.

This is the production-scale sibling of the accepted bounded C0 diagnostic
(``pretrain/corpus_gate_c_python.py``).  It traverses **every physical record of all 12 pinned
Python shards** and publishes the complete mechanically accepted candidate population.

It deliberately does not reuse the C0 release type or the C0 ceilings.  The C0 builder keeps its
frozen ``release_kind="c0_diagnostic"`` semantics and its structural caps; this module imports the
frozen Python *contracts* from it — the pinned source binding, the mechanical filter, the record
evaluator, the comment/blank definition — and adds a full-traversal transport, a full-scale resume
design, and a distinct release identity.

What this module produces is a **candidate registry**, not the final Python training corpus:

    full_source_traversal              = true
    final_quota_selection_performed    = false
    canonical_token_counting_performed = false
    stage_a_stage_b_final_split        = false
    promotion_to_final_training_corpus = false

Out of scope here, and asserted as such in every manifest: canonical-token counting, the frozen
0.60B quota selection, the Stage A / Stage B split, near-dedup, benchmark decontamination,
reference-reserve exclusion, chat conversion, textual separators, BOS/EOS insertion, and document
truncation.

Resource accounting is honest by construction: quantities this transport cannot measure are
reported as ``null`` with an explicit ``*_measured = false`` flag, never as zero.
"""

from __future__ import annotations

import argparse
import array
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
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
    _publish_directory,
    _require_nonnegative_int,
    _sha256_bytes,
    _staging_path,
    _write_new_file,
    canonical_jsonl_record_bytes,
)
from pretrain.corpus_gate_c_python import (  # noqa: E402
    FILTER_ORDER,
    PYTHON_FROZEN_LEAF_SET,
    PYTHON_MAX_BYTES,
    PYTHON_MAX_COMMENT_BLANK_FRACTION,
    PYTHON_MIN_BYTES,
    PYTHON_NULLABLE_PATHS,
    PYTHON_REQUIRED_SCHEMA,
    PYTHON_SOURCE,
    STAGE_B_MASS_SHARE,
    Decision,
    GateCPythonError,
    PythonSourceSpec,
    RemoteShard,
    ShardSpec,
    _record_leaves,
    assert_schema,
    assert_shard_scope,
    evaluate_record,
)

__all__ = [
    "FullBuildConfig",
    "GateCPythonFullError",
    "HubShardSource",
    "LocalShardSource",
    "PlannedInterruption",
    "ResourceAccounting",
    "build_full_candidates",
    "diagnose_release",
    "main",
    "verify_release",
]

FULL_TOOL_SCHEMA_VERSION = "petitgpt-corpus-gate-c-python-full-v1"
FULL_SPEC_VERSION = "python-gate-c-full-candidate-spec-2026-08-17"
RELEASE_KIND = "python_gate_c_full_candidate"

# The frozen filter contract this builder inherits verbatim from the accepted C0 module.  The hash
# is recorded in every manifest so a later reader can prove which filter produced the release.
FILTER_CONTRACT = {
    "language": PYTHON_SOURCE.language,
    "min_bytes": PYTHON_MIN_BYTES,
    "max_bytes": PYTHON_MAX_BYTES,
    "max_comment_blank_fraction": PYTHON_MAX_COMMENT_BLANK_FRACTION,
    "filter_order": list(FILTER_ORDER),
    "required_schema": dict(PYTHON_REQUIRED_SCHEMA),
    "frozen_leaf_set": list(PYTHON_FROZEN_LEAF_SET),
    "nullable_paths": list(PYTHON_NULLABLE_PATHS),
    "inherited_from": "pretrain/corpus_gate_c_python.py",
}
FILTER_CONTRACT_SHA256 = _sha256_bytes(_canonical_json_bytes(FILTER_CONTRACT))

READ_BLOCK_BYTES = 8 * 1024 * 1024
DEFAULT_CHECKPOINT_EVERY = 50_000
MAX_WALL_SECONDS = 24 * 60 * 60
MAX_DECOMPRESS_CHUNK_BYTES = 64 * 1024 * 1024

# Rejections raised before a record has ever been parsed into an object.
_PRE_SCHEMA_REJECTS = frozenset({"line_not_utf8", "malformed_json", "record_not_object"})


class GateCPythonFullError(GateCPythonError):
    """A fail-closed full Python Gate C contract error."""


# --------------------------------------------------------------------------------------
# Honest resource accounting
# --------------------------------------------------------------------------------------


@dataclass
class ResourceAccounting:
    """Exactly measured I/O quantities plus an explicit record of what is *not* measured.

    The contract this type exists to enforce: ``unknown != zero`` and ``unmeasured != measured``.
    The Hugging Face Hub client does not expose an HTTP request count, so that quantity is
    published as ``null`` with ``network_request_count_measured=false`` — never fabricated by, for
    example, counting one request per shard.
    """

    shard_download_count: int = 0
    shard_cache_reuse_count: int = 0
    shard_open_count: int = 0
    shard_integrity_verification_count: int = 0
    downloaded_compressed_bytes: int = 0
    compressed_source_bytes: int = 0
    decompressed_bytes: int = 0
    resume_reread_compressed_bytes: int = 0
    resume_reread_decompressed_bytes: int = 0
    integrity_hashed_compressed_bytes: int = 0

    MEASURED_FIELDS = (
        "shard_download_count",
        "shard_cache_reuse_count",
        "shard_open_count",
        "shard_integrity_verification_count",
        "downloaded_compressed_bytes",
        "compressed_source_bytes",
        "decompressed_bytes",
        "resume_reread_compressed_bytes",
        "resume_reread_decompressed_bytes",
        "integrity_hashed_compressed_bytes",
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
                "one request per shard."
            ),
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> ResourceAccounting:
        measured = value.get("measured") or {}
        if not isinstance(measured, Mapping):
            raise GateCPythonFullError("checkpoint resource accounting is invalid")
        return cls(**{
            name: _require_nonnegative_int(measured.get(name, 0), name)
            for name in cls.MEASURED_FIELDS
        })


@dataclass
class FullBuildCounters:
    """Traversal accounting for a complete pinned-source pass."""

    physical_records_seen: int = 0
    evaluated: int = 0
    accepted: int = 0
    rejected: int = 0
    accepted_text_bytes: int = 0
    rejections: Counter = field(default_factory=Counter)
    field_violations: Counter = field(default_factory=Counter)
    diagnostics: Counter = field(default_factory=Counter)

    def to_json(self) -> dict[str, Any]:
        return {
            "physical_records_seen": self.physical_records_seen,
            "evaluated": self.evaluated,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "accepted_text_bytes": self.accepted_text_bytes,
            "rejections": dict(sorted(self.rejections.items())),
            "field_violations": dict(sorted(self.field_violations.items())),
            "diagnostics": dict(sorted(self.diagnostics.items())),
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> FullBuildCounters:
        maps = {}
        for name in ("rejections", "field_violations", "diagnostics"):
            entry = value.get(name) or {}
            if not isinstance(entry, Mapping):
                raise GateCPythonFullError("checkpoint counter maps are invalid")
            maps[name] = Counter(dict(entry))
        return cls(
            physical_records_seen=_require_nonnegative_int(
                value.get("physical_records_seen"), "physical_records_seen"
            ),
            evaluated=_require_nonnegative_int(value.get("evaluated"), "evaluated"),
            accepted=_require_nonnegative_int(value.get("accepted"), "accepted"),
            rejected=_require_nonnegative_int(value.get("rejected"), "rejected"),
            accepted_text_bytes=_require_nonnegative_int(
                value.get("accepted_text_bytes"), "accepted_text_bytes"
            ),
            rejections=maps["rejections"],
            field_violations=maps["field_violations"],
            diagnostics=maps["diagnostics"],
        )


@dataclass(frozen=True)
class FullBuildConfig:
    """Full Python candidate build parameters.

    There is deliberately no document cap, no scan cap and no per-shard prefix cap: completeness
    of the traversal is the point of this builder.  ``max_wall_seconds`` is operational only — a
    time-capped run leaves a resumable checkpoint and publishes nothing.
    """

    source: PythonSourceSpec
    output_dir: Path
    work_dir: Path
    cache_dir: Path
    max_wall_seconds: float = float(MAX_WALL_SECONDS)
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY
    keep_shard_cache: bool = False
    stop_after_documents: int | None = None


# --------------------------------------------------------------------------------------
# Transport: complete pinned shard objects
# --------------------------------------------------------------------------------------


def sha256_file(path: Path, *, block_bytes: int = READ_BLOCK_BYTES) -> tuple[str, int]:
    """Return ``(hex digest, byte length)`` for a complete local object."""
    digest = hashlib.sha256()
    total = 0
    with open(path, "rb") as handle:
        while True:
            block = handle.read(block_bytes)
            if not block:
                break
            digest.update(block)
            total += len(block)
    return digest.hexdigest(), total


class ShardSource:
    """Transport interface: pinned identity plus complete-object fetch."""

    def list_shards(self, source: PythonSourceSpec) -> tuple[RemoteShard, ...]:
        raise NotImplementedError

    def revision_file_count(self, source: PythonSourceSpec) -> int:
        raise NotImplementedError

    def fetch(self, source: PythonSourceSpec, shard: ShardSpec, destination: Path) -> int:
        """Write the complete pinned object to ``destination``; return bytes transferred."""
        raise NotImplementedError


class HubShardSource(ShardSource):
    """Official Hugging Face Hub transport pinned to the exact revision.

    ``huggingface_hub`` is imported lazily so the CPU-only contract suite never needs it.
    """

    def __init__(self, *, block_bytes: int = READ_BLOCK_BYTES, max_attempts: int = 4) -> None:
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
            raise GateCPythonFullError(
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

    def fetch(self, source: PythonSourceSpec, shard: ShardSpec, destination: Path) -> int:
        _, fs = self._hub()
        uri = f"datasets/{source.dataset}@{source.revision}/{shard.path}"
        partial = destination.parent / f".{destination.name}.partial"
        last: BaseException | None = None
        for attempt in range(self.max_attempts):
            written = 0
            try:
                with fs.open(uri, "rb", block_size=self.block_bytes) as remote:
                    with open(partial, "wb") as local:
                        while True:
                            block = remote.read(self.block_bytes)
                            if not block:
                                break
                            local.write(block)
                            written += len(block)
                        local.flush()
                        os.fsync(local.fileno())
                if written != shard.size:
                    raise GateCPythonFullError(
                        f"pinned shard transfer is short: {shard.path} "
                        f"({written} of {shard.size} bytes)"
                    )
                os.replace(partial, destination)
                _fsync_directory(destination.parent)
                return written
            except Exception as exc:  # noqa: BLE001 - bounded retry of the same pinned object
                last = exc
                partial.unlink(missing_ok=True)
                if attempt + 1 >= self.max_attempts:
                    raise GateCPythonFullError(
                        f"pinned shard unavailable after bounded retries: {shard.path}"
                    ) from last
                time.sleep(float(2**attempt))
        raise GateCPythonFullError(f"pinned shard could not be fetched: {shard.path}")


class LocalShardSource(ShardSource):
    """Local-file transport for contract tests: identical framing and integrity rules."""

    def __init__(self, root: Path, *, block_bytes: int = 1 << 16) -> None:
        self.root = Path(root)
        self.block_bytes = block_bytes

    def _describe(self, path: str) -> RemoteShard:
        target = self.root / path
        if not target.exists():
            raise GateCPythonFullError(f"pinned shard is missing from the local store: {path}")
        digest, size = sha256_file(target, block_bytes=self.block_bytes)
        return RemoteShard(
            path=path,
            size=size,
            lfs_sha256=digest,
            blob_id=hashlib.sha1(target.read_bytes(), usedforsecurity=False).hexdigest(),
        )

    def revision_file_count(self, source: PythonSourceSpec) -> int:
        return len(sorted(self.root.glob("*.json.gz")))

    def list_shards(self, source: PythonSourceSpec) -> tuple[RemoteShard, ...]:
        return tuple(self._describe(path) for path in sorted(source.shard_paths))

    def fetch(self, source: PythonSourceSpec, shard: ShardSpec, destination: Path) -> int:
        partial = destination.parent / f".{destination.name}.partial"
        shutil.copyfile(self.root / shard.path, partial)
        os.replace(partial, destination)
        _fsync_directory(destination.parent)
        return destination.stat().st_size


def ensure_verified_shard(
    source: PythonSourceSpec,
    shard: ShardSpec,
    cache_dir: Path,
    store: ShardSource,
    resources: ResourceAccounting,
) -> Path:
    """Return a local path holding the complete pinned object, verified against its frozen identity.

    A usable cache entry is reused rather than re-downloaded, but it is verified either way: the
    object that feeds the filter is always one whose full compressed SHA-256 equals the pinned LFS
    digest.  No document is ever emitted from an unverified object.
    """
    target = cache_dir / shard.path
    if target.exists():
        digest, size = sha256_file(target)
        resources.shard_integrity_verification_count += 1
        resources.integrity_hashed_compressed_bytes += size
        if digest == shard.lfs_sha256 and size == shard.size:
            resources.shard_cache_reuse_count += 1
            return target
        target.unlink()

    transferred = store.fetch(source, shard, target)
    resources.shard_download_count += 1
    resources.downloaded_compressed_bytes += transferred

    digest, size = sha256_file(target)
    resources.shard_integrity_verification_count += 1
    resources.integrity_hashed_compressed_bytes += size
    if size != shard.size:
        raise GateCPythonFullError(
            f"pinned shard size mismatch after fetch: {shard.path} ({size} != {shard.size})"
        )
    if digest != shard.lfs_sha256:
        raise GateCPythonFullError(
            f"pinned shard SHA-256 does not match the frozen LFS object: {shard.path}"
        )
    return target


def iter_full_shard_records(
    path: Path,
    *,
    start_record_index: int = 0,
    block_bytes: int = READ_BLOCK_BYTES,
) -> Iterator[tuple[int, bytes, int, int]]:
    """Yield every complete record of a *complete* gzip JSONL object, in physical order.

    Unlike the C0 prefix reader this never accepts a truncated object: the stream must reach a
    gzip terminator and must not end mid-record.  Concatenated gzip members are decoded as one
    logical stream, so a multi-member object cannot silently lose its trailing members.

    Yields ``(record_index, line, consumed_compressed, produced_decompressed)`` and finally a
    ``(-1, b"", residual_compressed, residual_decompressed)`` sentinel so byte accounting stays
    exact even when the last block produced no record.
    """
    decompressor = zlib.decompressobj(31)
    buffer = bytearray()
    index = 0
    pending_compressed = 0
    pending_decompressed = 0
    saw_any = False
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(block_bytes)
            if not chunk:
                break
            saw_any = True
            pending_compressed += len(chunk)
            data = chunk
            while data:
                try:
                    produced = decompressor.decompress(data, MAX_DECOMPRESS_CHUNK_BYTES)
                except zlib.error as exc:
                    raise GateCPythonFullError(f"gzip stream is corrupt: {exc}") from exc
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
                if decompressor.eof:
                    # Concatenated gzip members form one legal object; a trailing member must
                    # never be silently dropped, so decoding continues with a fresh stream.
                    remainder = decompressor.unused_data
                    if remainder:
                        decompressor = zlib.decompressobj(31)
                        data = remainder
                    else:
                        data = b""
                else:
                    data = decompressor.unconsumed_tail
    if not saw_any:
        raise GateCPythonFullError(f"pinned shard object is empty: {path}")
    if not decompressor.eof:
        raise GateCPythonFullError(f"gzip stream ended before its terminator: {path}")
    if buffer.strip():
        raise GateCPythonFullError(f"shard ended with an unterminated JSONL record: {path}")
    if pending_compressed or pending_decompressed:
        yield -1, b"", pending_compressed, pending_decompressed


def evaluate_line(
    line: bytes,
    source: PythonSourceSpec,
    live_schema: list[str],
    *,
    check_schema: bool,
) -> Decision:
    """Decode one physical JSONL record and apply the frozen Python filter.

    The filter itself is ``pretrain.corpus_gate_c_python.evaluate_record`` verbatim; only the
    transport-level decoding lives here.
    """
    try:
        decoded = line.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return Decision(accepted=False, reason="line_not_utf8")
    try:
        record = json.loads(decoded)
    except json.JSONDecodeError:
        return Decision(accepted=False, reason="malformed_json")
    if not isinstance(record, dict):
        return Decision(accepted=False, reason="record_not_object")
    if check_schema:
        leaves = _record_leaves(record)
        assert_schema(source, leaves)
        if not live_schema:
            live_schema.extend(sorted(leaves))
    return evaluate_record(record, source)


# --------------------------------------------------------------------------------------
# Configuration, fingerprint and checkpoint
# --------------------------------------------------------------------------------------


def _validate_config(config: FullBuildConfig) -> None:
    source = config.source
    _ensure_revision(source.revision)
    if len(source.shards) != 12:
        raise GateCPythonFullError("the Python shard scope must be exactly the 12 pinned shards")
    if len(set(source.shard_paths)) != len(source.shards):
        raise GateCPythonFullError("the pinned Python shard list contains a duplicate path")
    if (
        not math.isfinite(config.max_wall_seconds)
        or not 0 < config.max_wall_seconds <= MAX_WALL_SECONDS
    ):
        raise GateCPythonFullError(f"max_wall_seconds must be in (0, {MAX_WALL_SECONDS}]")
    if not 1 <= config.checkpoint_every <= 10_000_000:
        raise GateCPythonFullError("checkpoint_every must be in [1, 10000000]")
    if config.stop_after_documents is not None and config.stop_after_documents < 1:
        raise GateCPythonFullError("stop_after_documents must be >= 1 when set")
    for path in (
        config.output_dir,
        config.work_dir,
        config.cache_dir,
        _staging_path(config.output_dir),
    ):
        _ensure_git_ignored(path)


def run_fingerprint(config: FullBuildConfig) -> str:
    """Bind every semantic build input; operational caps and cadence are excluded."""
    source = config.source
    return _sha256_bytes(
        _canonical_json_bytes({
            "tool_schema_version": FULL_TOOL_SCHEMA_VERSION,
            "spec_version": FULL_SPEC_VERSION,
            "release_kind": RELEASE_KIND,
            "source": source.to_json(),
            "filter_contract_sha256": FILTER_CONTRACT_SHA256,
            "full_source_traversal": True,
            "output_dir": str(_ensure_under_workspace(config.output_dir)),
        })
    )


def _new_checkpoint(fingerprint: str) -> dict[str, Any]:
    return {
        "tool_schema_version": FULL_TOOL_SCHEMA_VERSION,
        "filter_contract_sha256": FILTER_CONTRACT_SHA256,
        "run_fingerprint": fingerprint,
        "next_shard_index": 0,
        "next_record_index": 0,
        "counters": FullBuildCounters().to_json(),
        "resources": ResourceAccounting().to_json(),
        "per_shard": {},
        "documents_sha256": _sha256_bytes(b""),
        "documents_bytes": 0,
        "resume_count": 0,
        "completed": False,
        "live_schema": [],
        "shard_identity": [],
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
        raise GateCPythonFullError(f"checkpoint is unreadable: {path}") from exc
    try:
        state = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateCPythonFullError("checkpoint is corrupted: not strict UTF-8 JSON") from exc
    if not isinstance(state, dict):
        raise GateCPythonFullError("checkpoint is corrupted: not an object")
    checksum = state.get("checksum")
    if not isinstance(checksum, str) or checksum != _sha256_bytes(_checkpoint_payload(state)):
        raise GateCPythonFullError("checkpoint checksum mismatch; refusing to resume")
    if state.get("tool_schema_version") != FULL_TOOL_SCHEMA_VERSION:
        raise GateCPythonFullError("checkpoint tool schema version mismatch")
    if state.get("filter_contract_sha256") != FILTER_CONTRACT_SHA256:
        raise GateCPythonFullError("checkpoint filter contract mismatch; refusing to resume")
    if state.get("run_fingerprint") != fingerprint:
        raise GateCPythonFullError("checkpoint run fingerprint mismatch; refusing to resume")
    for name in ("next_shard_index", "next_record_index", "documents_bytes", "resume_count"):
        _require_nonnegative_int(state.get(name), name)
    for name in ("live_schema", "shard_identity"):
        if not isinstance(state.get(name), list):
            raise GateCPythonFullError(f"checkpoint field is invalid: {name}")
    if not isinstance(state.get("documents_sha256"), str):
        raise GateCPythonFullError("checkpoint field is invalid: documents_sha256")
    if not isinstance(state.get("completed"), bool):
        raise GateCPythonFullError("checkpoint field is invalid: completed")
    if not isinstance(state.get("per_shard"), Mapping):
        raise GateCPythonFullError("checkpoint field is invalid: per_shard")
    FullBuildCounters.from_json(state.get("counters") or {})
    ResourceAccounting.from_json(state.get("resources") or {})
    return state


def restore_and_rebuild_dedup(
    documents_path: Path, state: Mapping[str, Any]
) -> tuple[set[str], set[str], Any, int]:
    """Truncate to the committed prefix and rebuild the dedup sets from it.

    At full scale the seen-id / seen-hash sets are far too large to serialize into a checkpoint
    that is rewritten many times, so they are *not* stored: they are exactly the keys of the
    committed candidate prefix and are rebuilt by reading it.  Duplicate rejections never enter
    the sets, so a rebuild from accepted documents alone is exact.
    """
    expected_bytes = int(state["documents_bytes"])
    expected_sha = str(state["documents_sha256"])
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    digest = hashlib.sha256()

    if not documents_path.exists():
        if expected_bytes != 0:
            raise GateCPythonFullError(
                "checkpoint references accepted documents but the candidate file is missing"
            )
        _write_new_file(documents_path, b"")
        return seen_ids, seen_hashes, digest, 0

    actual_size = documents_path.stat().st_size
    if actual_size < expected_bytes:
        raise GateCPythonFullError("candidate file is shorter than its committed checkpoint prefix")

    remaining = expected_bytes
    with open(documents_path, "rb") as handle:
        for line in handle:
            if remaining <= 0:
                break
            if len(line) > remaining:
                raise GateCPythonFullError(
                    "committed checkpoint prefix does not end on a record boundary"
                )
            remaining -= len(line)
            digest.update(line)
            if not line.strip():
                continue
            record = json.loads(line.decode("utf-8", errors="strict"))
            seen_ids.add(record["source_record_id"])
            seen_hashes.add(record["text_sha256"])
    if remaining != 0:
        raise GateCPythonFullError("committed checkpoint prefix could not be read in full")
    if digest.hexdigest() != expected_sha:
        raise GateCPythonFullError("candidate prefix does not match the committed checkpoint hash")
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


def _new_shard_entry() -> dict[str, Any]:
    return {
        "physical_records": 0,
        "evaluated": 0,
        "accepted": 0,
        "rejected": 0,
        "accepted_text_bytes": 0,
        "compressed_bytes": 0,
        "decompressed_bytes": 0,
        "pinned_compressed_size": 0,
        "integrity_verified": False,
        "complete": False,
    }


def _make_manifest(
    config: FullBuildConfig,
    state: Mapping[str, Any],
    counters: FullBuildCounters,
    resources: ResourceAccounting,
    *,
    fingerprint: str,
    wall_seconds: float,
    documents_sha256: str,
    documents_bytes: int,
) -> dict[str, Any]:
    source = config.source
    per_shard = {key: state["per_shard"][key] for key in sorted(state["per_shard"])}
    complete = [key for key, entry in per_shard.items() if entry["complete"]]
    return {
        "tool_schema_version": FULL_TOOL_SCHEMA_VERSION,
        "spec_version": FULL_SPEC_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "release_kind": RELEASE_KIND,
        "release_description": (
            "The complete mechanically accepted Python candidate population from the frozen "
            "Common Pile source. This is a candidate registry, not the final Python training "
            "corpus."
        ),
        "full_source_traversal": True,
        "final_quota_selection_performed": False,
        "canonical_token_counting_performed": False,
        "stage_a_stage_b_final_split_performed": False,
        "promotion_to_final_training_corpus": False,
        "promotion_eligible": False,
        "promotion_eligible_rationale": (
            "Promotion to the final training corpus requires canonical-token accounting, the "
            "frozen 0.60B Python quota selection and the Stage A / Stage B split, none of which "
            "exist at Gate C. This release is the input to those stages, not their output."
        ),
        "supersedes_c0_diagnostic": False,
        "c0_relationship": (
            "Independent of the accepted bounded C0 diagnostic release, which keeps its own "
            "release_kind and structural ceilings. The frozen filter contract is imported from "
            "the C0 module unchanged."
        ),
        "run_fingerprint": fingerprint,
        "source": source.to_json(),
        "shard_identity": list(state.get("shard_identity") or []),
        "shard_identity_verified": bool(state.get("shard_identity")),
        "shard_integrity_policy": (
            "Every shard is fetched or reused as a complete object and its full compressed "
            "SHA-256 is verified against the pinned LFS digest before any record from it is "
            "evaluated. Prefix reads are never treated as complete shards."
        ),
        "live_schema": sorted(state.get("live_schema") or []),
        "schema_verified": bool(state.get("live_schema")),
        "filter_contract": FILTER_CONTRACT,
        "filter_contract_sha256": FILTER_CONTRACT_SHA256,
        "traversal": {
            "method": "complete_sequential_physical_traversal",
            "description": (
                "Shards are traversed in ascending pinned path order; every physical JSONL record "
                "of every shard is evaluated. There is no stride, no sampling, no document cap "
                "and no per-shard prefix cap."
            ),
            "stride": 1,
            "sampling": False,
            "shards_complete": len(complete),
            "shards_pinned": len(source.shards),
            "all_shards_complete": len(complete) == len(source.shards),
        },
        "caps": {
            "max_wall_seconds": config.max_wall_seconds,
            "checkpoint_every": config.checkpoint_every,
            "document_cap": None,
            "scan_cap": None,
            "per_shard_compressed_cap": None,
        },
        "accounting": counters.to_json(),
        "yield_rate": (counters.accepted / counters.evaluated) if counters.evaluated else None,
        "proxy_tokens_4_bytes_per_token": counters.accepted_text_bytes / 4.0,
        "proxy_token_note": (
            "4.0 bytes/token is the frozen byte proxy used for planning. This is NOT a canonical "
            "token count; no tokenizer exists at Gate C."
        ),
        "resources": resources.to_json(),
        "per_shard": per_shard,
        "shards_complete": len(complete),
        "shards_pinned": len(source.shards),
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
            "ast_boundary_split": False,
            "comment_docstring_stripping": False,
            "provenance_header_injection": False,
            "cross_source_near_dedup": False,
            "benchmark_decontamination": False,
            "reference_reserve_exclusion": False,
            "stage_a_stage_b_split_performed": False,
        },
        "hard_stops": {
            "final_0p60b_selection_started": False,
            "canonical_token_counting_started": False,
            "tokenizer_trained": False,
            "gate_r_started": False,
            "final_shards_built": False,
            "cuda_smoke_started": False,
            "model_training_started": False,
        },
    }


# --------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------


def build_full_candidates(
    config: FullBuildConfig,
    *,
    store: ShardSource | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Traverse every pinned Python shard completely and publish the full candidate release.

    The candidate file is written directly into the staging directory that is later atomically
    renamed into place, so publication never copies the (multi-gigabyte) candidate corpus.
    """
    _validate_config(config)
    source = config.source
    store = store if store is not None else HubShardSource()

    output_dir = _ensure_under_workspace(config.output_dir)
    if output_dir.exists():
        raise GateCPythonFullError(f"refusing to overwrite published output: {output_dir}")
    work_dir = _ensure_under_workspace(config.work_dir)
    cache_dir = _ensure_under_workspace(config.cache_dir)
    staging = _staging_path(output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)

    fingerprint = run_fingerprint(config)
    checkpoint_path = work_dir / CHECKPOINT_NAME
    documents_path = staging / DOCUMENTS_NAME

    resumed = checkpoint_path.exists()
    if resumed:
        state = _read_checkpoint(checkpoint_path, fingerprint)
        state["resume_count"] = int(state["resume_count"]) + 1
    else:
        state = _new_checkpoint(fingerprint)

    seen_ids, seen_hashes, documents_digest, documents_bytes = restore_and_rebuild_dedup(
        documents_path, state
    )

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

    counters = FullBuildCounters.from_json(state["counters"])
    resources = ResourceAccounting.from_json(state["resources"])
    per_shard: dict[str, dict[str, Any]] = {
        key: dict(value) for key, value in (state.get("per_shard") or {}).items()
    }
    live_schema: list[str] = list(state.get("live_schema") or [])
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
        state["resources"] = resources.to_json()
        state["per_shard"] = per_shard
        state["documents_sha256"] = documents_digest.hexdigest()
        state["documents_bytes"] = documents_bytes
        state["completed"] = is_complete
        state["live_schema"] = sorted(live_schema)
        state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _write_checkpoint(checkpoint_path, state)

    try:
        if completed:
            stop_reason = "all_shards_complete"
        else:
            stop_reason = "all_shards_complete"
            since_checkpoint = 0
            stop = False
            for shard_index in range(next_shard_index, len(source.shards)):
                if shard_index != next_shard_index:
                    raise GateCPythonFullError(
                        "shard cursor desynchronized from the traversal order"
                    )
                shard = source.shards[shard_index]
                entry = per_shard.setdefault(shard.path, _new_shard_entry())
                entry["pinned_compressed_size"] = shard.size
                resume_offset = next_record_index

                local_path = ensure_verified_shard(source, shard, cache_dir, store, resources)
                entry["integrity_verified"] = True
                resources.shard_open_count += 1

                schema_checked = False
                first_yield = True
                for (
                    record_index,
                    line,
                    used_compressed,
                    used_decompressed,
                ) in iter_full_shard_records(local_path, start_record_index=resume_offset):
                    resources.compressed_source_bytes += used_compressed
                    resources.decompressed_bytes += used_decompressed
                    entry["compressed_bytes"] += used_compressed
                    entry["decompressed_bytes"] += used_decompressed
                    if first_yield and resume_offset > 0 and record_index >= 0:
                        # A single-member gzip stream cannot be seeked, so reaching a mid-shard
                        # cursor costs a prefix re-read. Account for it rather than hide it.
                        resources.resume_reread_compressed_bytes += used_compressed
                        resources.resume_reread_decompressed_bytes += used_decompressed
                    first_yield = False
                    if record_index < 0:
                        break
                    if record_index < next_record_index:
                        raise GateCPythonFullError(
                            "transport produced a record before the checkpointed cursor"
                        )
                    next_record_index = record_index + 1
                    counters.physical_records_seen += 1
                    entry["physical_records"] += 1
                    counters.evaluated += 1
                    entry["evaluated"] += 1

                    decision = evaluate_line(
                        line, source, live_schema, check_schema=not schema_checked
                    )
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
                                    "transport": "huggingface_hub_complete_gzip_jsonl",
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
                    if clock() - start > config.max_wall_seconds:
                        stop_reason = "time_cap"
                        stop = True
                        break
                    if since_checkpoint >= config.checkpoint_every:
                        commit(False)
                        since_checkpoint = 0

                if stop:
                    break
                entry["complete"] = True
                next_shard_index = shard_index + 1
                next_record_index = 0
                commit(False)
                since_checkpoint = 0
                if not config.keep_shard_cache:
                    local_path.unlink(missing_ok=True)
            else:
                completed = True
        commit(completed)
    finally:
        handle.close()

    incomplete = [
        shard.path for shard in source.shards if not per_shard.get(shard.path, {}).get("complete")
    ]
    if incomplete or not completed:
        # A partial traversal is never published: an unpublished checkpoint is resumable, whereas
        # a published "complete" release that silently omitted shards would be unrecoverable.
        return {
            "published": False,
            "stop_reason": stop_reason,
            "work_dir": str(work_dir),
            "staging_dir": str(staging),
            "incomplete_shards": incomplete,
            "shards_complete": len(source.shards) - len(incomplete),
            "accepted": counters.accepted,
            "evaluated": counters.evaluated,
            "rejected": counters.rejected,
            "accepted_text_bytes": counters.accepted_text_bytes,
            "next_shard_index": next_shard_index,
            "next_record_index": next_record_index,
            "resume_count": int(state["resume_count"]),
            "resumed": resumed,
            "documents_sha256": documents_digest.hexdigest(),
            "resources": resources.to_json(),
        }

    wall_seconds = clock() - start
    manifest = _make_manifest(
        config,
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
    documents_sha, documents_size = sha256_file(staging / DOCUMENTS_NAME)
    if documents_sha != manifest["documents_sha256"] or documents_size != documents_bytes:
        raise GateCPythonFullError("staged candidate file does not match its incremental digest")
    _write_new_file(
        staging / CHECKSUMS_NAME,
        (
            f"{documents_sha}  {DOCUMENTS_NAME}\n"
            f"{_sha256_bytes((staging / MANIFEST_NAME).read_bytes())}  {MANIFEST_NAME}\n"
        ).encode(),
    )
    _publish_directory(staging, output_dir)

    return {
        "published": True,
        "output_dir": str(output_dir),
        "release_kind": RELEASE_KIND,
        "promotion_eligible": False,
        "stop_reason": stop_reason,
        "shards_complete": len(source.shards),
        "physical_records_seen": counters.physical_records_seen,
        "evaluated": counters.evaluated,
        "accepted": counters.accepted,
        "rejected": counters.rejected,
        "accepted_text_bytes": counters.accepted_text_bytes,
        "proxy_tokens_4_bytes_per_token": counters.accepted_text_bytes / 4.0,
        "yield_rate": (counters.accepted / counters.evaluated) if counters.evaluated else None,
        "wall_seconds": round(wall_seconds, 3),
        "resume_count": int(state["resume_count"]),
        "resumed": resumed,
        "documents_sha256": documents_sha,
        "manifest_sha256": _sha256_bytes((output_dir / MANIFEST_NAME).read_bytes()),
        "rejections": dict(sorted(counters.rejections.items())),
        "resources": resources.to_json(),
    }


# --------------------------------------------------------------------------------------
# Verification (streaming; the candidate file does not fit comfortably in memory)
# --------------------------------------------------------------------------------------


def verify_release(output_dir: Path) -> dict[str, Any]:
    """Recompute every load-bearing invariant of a published full candidate release."""
    output_dir = _ensure_under_workspace(output_dir)
    manifest_bytes = (output_dir / MANIFEST_NAME).read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8", errors="strict"))

    if manifest.get("tool_schema_version") != FULL_TOOL_SCHEMA_VERSION:
        raise GateCPythonFullError("release was not produced by this tool schema version")
    if manifest.get("release_kind") != RELEASE_KIND:
        raise GateCPythonFullError(f"release_kind must be {RELEASE_KIND}")
    if manifest.get("filter_contract_sha256") != FILTER_CONTRACT_SHA256:
        raise GateCPythonFullError("release filter contract does not match this builder")
    for flag in (
        "final_quota_selection_performed",
        "canonical_token_counting_performed",
        "stage_a_stage_b_final_split_performed",
        "promotion_to_final_training_corpus",
        "promotion_eligible",
        "canonical_token_split_performed",
    ):
        if manifest.get(flag) is not False:
            raise GateCPythonFullError(f"{flag} must be false in a Gate C candidate release")
    if manifest.get("full_source_traversal") is not True:
        raise GateCPythonFullError("full_source_traversal must be true")
    if manifest["shards_complete"] != manifest["shards_pinned"]:
        raise GateCPythonFullError("a published release must have every pinned shard complete")
    for path, entry in manifest["per_shard"].items():
        if not entry.get("complete"):
            raise GateCPythonFullError(f"shard is marked incomplete in a published release: {path}")
        if not entry.get("integrity_verified"):
            raise GateCPythonFullError(f"shard integrity was never verified: {path}")

    pinned = {entry["path"] for entry in manifest["source"]["shards"]}
    documents_path = output_dir / DOCUMENTS_NAME
    digest = hashlib.sha256()
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    per_shard: Counter = Counter()
    accepted = 0
    accepted_text_bytes = 0
    total_bytes = 0

    with open(documents_path, "rb") as handle:
        for line in handle:
            digest.update(line)
            total_bytes += len(line)
            if not line.strip():
                continue
            if not line.endswith(b"\n"):
                raise GateCPythonFullError("candidate file ends with an unterminated record")
            decoded = line[:-1].decode("utf-8", errors="strict")
            # Byte framing and str framing must agree: a raw U+0085/2028/2029 inside a record
            # would split it for a str.splitlines() reader.
            if len(decoded.splitlines()) != 1:
                raise GateCPythonFullError("physical JSONL framing is ambiguous for a record")
            record = json.loads(decoded)
            encoded = record["text"].encode("utf-8")
            if _sha256_bytes(encoded) != record["text_sha256"]:
                raise GateCPythonFullError("candidate record text_sha256 was tampered with")
            if len(encoded) != record["text_bytes"]:
                raise GateCPythonFullError("candidate record text_bytes does not match its text")
            if not PYTHON_MIN_BYTES <= len(encoded) <= PYTHON_MAX_BYTES:
                raise GateCPythonFullError("candidate record violates the frozen byte band")
            if record["metadata"]["metadata.language"] != manifest["source"]["language"]:
                raise GateCPythonFullError("candidate release contains a non-Python document")
            if record["shard_path"] not in pinned:
                raise GateCPythonFullError("candidate release references an unpinned shard")
            if record["source_record_id"] in seen_ids:
                raise GateCPythonFullError("candidate release contains a duplicate record id")
            if record["text_sha256"] in seen_hashes:
                raise GateCPythonFullError("candidate release contains a duplicate text hash")
            seen_ids.add(record["source_record_id"])
            seen_hashes.add(record["text_sha256"])
            per_shard[record["shard_path"]] += 1
            accepted += 1
            accepted_text_bytes += len(encoded)

    if digest.hexdigest() != manifest["documents_sha256"]:
        raise GateCPythonFullError("documents.jsonl does not match manifest documents_sha256")
    if total_bytes != manifest["documents_bytes"]:
        raise GateCPythonFullError("documents.jsonl length does not match the manifest")
    if accepted != manifest["accounting"]["accepted"]:
        raise GateCPythonFullError("published record count does not match the manifest accounting")
    if accepted_text_bytes != manifest["accounting"]["accepted_text_bytes"]:
        raise GateCPythonFullError("published text bytes do not match the manifest accounting")
    for path, entry in manifest["per_shard"].items():
        if per_shard.get(path, 0) != entry["accepted"]:
            raise GateCPythonFullError(f"per-shard accepted count does not match documents: {path}")
    for entry in (output_dir / CHECKSUMS_NAME).read_text().splitlines():
        recorded, name = entry.split("  ", 1)
        actual, _ = sha256_file(output_dir / name)
        if actual != recorded:
            raise GateCPythonFullError(f"MANIFEST.sha256 mismatch for {name}")

    return {
        "output_dir": str(output_dir),
        "release_kind": manifest["release_kind"],
        "full_source_traversal": True,
        "shards_complete": manifest["shards_complete"],
        "accepted": accepted,
        "accepted_text_bytes": accepted_text_bytes,
        "distinct_record_ids": len(seen_ids),
        "distinct_text_hashes": len(seen_hashes),
        "documents_sha256": digest.hexdigest(),
        "manifest_sha256": _sha256_bytes(manifest_bytes),
    }


# --------------------------------------------------------------------------------------
# Full-population diagnostics
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
    """Full-population diagnostics plus the provisional byte-weighted Stage B boundary.

    The cutoff is computed exactly rather than approximately: the upstream score lies on a
    discrete grid, so an exact score -> (count, bytes) histogram locates the boundary bucket, and
    a second pass resolves the SHA-256 tie-break inside that single bucket only.  Memory stays
    bounded regardless of population size.
    """
    output_dir = _ensure_under_workspace(output_dir)
    manifest = json.loads((output_dir / MANIFEST_NAME).read_bytes().decode("utf-8"))
    documents_path = output_dir / DOCUMENTS_NAME

    lengths = array.array("i")
    line_counts = array.array("i")
    fractions = array.array("d")
    score_counts: Counter = Counter()
    score_bytes: Counter = Counter()
    int_scores: Counter = Counter()
    licenses: Counter = Counter()
    license_types: Counter = Counter()
    repos: Counter = Counter()
    shard_docs: Counter = Counter()
    shard_bytes: Counter = Counter()
    diagnostics: Counter = Counter()
    repo_present = 0
    license_present = 0
    detected_present = 0
    total = 0
    total_bytes = 0

    with open(documents_path, "rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line.decode("utf-8", errors="strict"))
            metadata = record["metadata"]
            total += 1
            text_bytes = int(record["text_bytes"])
            total_bytes += text_bytes
            lengths.append(text_bytes)
            line_counts.append(int(metadata["derived.line_count"]))
            fractions.append(float(metadata["derived.comment_blank_fraction"]))
            score = float(metadata["score"])
            score_counts[score] += 1
            score_bytes[score] += text_bytes
            int_scores[str(metadata["int_score"])] += 1
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
            shard_docs[record["shard_path"]] += 1
            shard_bytes[record["shard_path"]] += text_bytes
            for name in record.get("diagnostics") or []:
                diagnostics[name] += 1

    target = STAGE_B_MASS_SHARE * total_bytes
    cumulative = 0
    cumulative_docs = 0
    cutoff_score: float | None = None
    for score in sorted(score_counts, reverse=True):
        if cumulative + score_bytes[score] >= target:
            cutoff_score = score
            break
        cumulative += score_bytes[score]
        cumulative_docs += score_counts[score]
    strictly_above_bytes = cumulative
    strictly_above_docs = cumulative_docs

    tie_included_docs = 0
    tie_included_bytes = 0
    if cutoff_score is not None:
        # Resolve the frozen SHA-256 ascending tie-break inside the boundary bucket only.
        bucket: list[tuple[str, int]] = []
        with open(documents_path, "rb") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line.decode("utf-8", errors="strict"))
                if float(record["metadata"]["score"]) == cutoff_score:
                    bucket.append((record["text_sha256"], int(record["text_bytes"])))
        bucket.sort()
        for _, text_bytes in bucket:
            if cumulative >= target:
                break
            cumulative += text_bytes
            tie_included_bytes += text_bytes
            tie_included_docs += 1

    included_docs = strictly_above_docs + tie_included_docs

    return {
        "tool_schema_version": FULL_TOOL_SCHEMA_VERSION,
        "spec_version": FULL_SPEC_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "release": {
            "output_dir": str(output_dir),
            "release_kind": manifest["release_kind"],
            "documents_sha256": manifest["documents_sha256"],
            "run_fingerprint": manifest["run_fingerprint"],
            "revision": manifest["source"]["revision"],
            "shards_complete": manifest["shards_complete"],
        },
        "documents": total,
        "accepted_text_bytes": total_bytes,
        "proxy_tokens_4_bytes_per_token": total_bytes / 4.0,
        "proxy_token_label": "NOT CANONICAL TOKEN COUNT",
        "canonical_token_counting_performed": False,
        "length_bytes": _quantiles(lengths),
        "line_count": _quantiles(line_counts),
        "comment_blank_fraction": _quantiles(fractions),
        "score": _score_quantiles(score_counts, total) if total else {"n": 0},
        "score_histogram": {f"{score:.6f}": count for score, count in sorted(score_counts.items())},
        "score_byte_mass": {f"{score:.6f}": score_bytes[score] for score in sorted(score_bytes)},
        "distinct_scores": len(score_counts),
        "int_score_histogram": dict(sorted(int_scores.items())),
        "provisional_stage_b_cutoff": {
            "provisional_byte_weighted_only": True,
            "canonical_token_split_performed": False,
            "final_stage_a_stage_b_split_performed": False,
            "stage_b_mass_share": STAGE_B_MASS_SHARE,
            "rule": (
                "rank by descending continuous score with a SHA-256 ascending tie-break, "
                "accumulate accepted UTF-8 text bytes until 5/12 of the mass is reached"
            ),
            "target_bytes": target,
            "cutoff_score": cutoff_score,
            "included_documents": included_docs,
            "included_bytes": cumulative,
            "included_byte_share": (cumulative / total_bytes) if total_bytes else None,
            "documents_strictly_above_cutoff": strictly_above_docs,
            "bytes_strictly_above_cutoff": strictly_above_bytes,
            "documents_admitted_by_tie_break": tie_included_docs,
            "bytes_admitted_by_tie_break": tie_included_bytes,
            "caveat": (
                "Byte mass is not canonical token mass. This is evidence only: it must be "
                "recomputed on canonical tokens once the tokenizer release exists, and it must "
                "never be frozen into code as a fixed score threshold."
            ),
        },
        "provenance_availability": {
            "repo_name_present": repo_present,
            "repo_name_present_rate": (repo_present / total) if total else None,
            "distinct_repos": len(repos),
            "top_repo_share": (max(repos.values()) / total) if total and repos else None,
            "top_10_repo_share": (
                sum(sorted(repos.values(), reverse=True)[:10]) / total if total and repos else None
            ),
            "license_present": license_present,
            "license_present_rate": (license_present / total) if total else None,
            "detected_licenses_present": detected_present,
            "detected_licenses_present_rate": (detected_present / total) if total else None,
        },
        "license_histogram": _top(licenses, limit=30),
        "license_type_histogram": dict(sorted(license_types.items())),
        "top_repos": _top(repos, limit=25),
        "shard_distribution": {
            "documents": dict(sorted(shard_docs.items())),
            "accepted_text_bytes": dict(sorted(shard_bytes.items())),
            "shards_covered": len(shard_docs),
            "shards_pinned": manifest["shards_pinned"],
        },
        "ast_diagnostics": {
            "has_function": diagnostics.get("has_function", 0),
            "has_function_share": (diagnostics.get("has_function", 0) / total) if total else None,
            "has_class": diagnostics.get("has_class", 0),
            "has_class_share": (diagnostics.get("has_class", 0) / total) if total else None,
            "has_docstring": diagnostics.get("has_docstring", 0),
            "has_docstring_share": (diagnostics.get("has_docstring", 0) / total) if total else None,
        },
        "exact_dedup": {
            "duplicate_source_record_id_rejects": manifest["accounting"]["rejections"].get(
                "duplicate_source_record_id", 0
            ),
            "duplicate_text_sha256_rejects": manifest["accounting"]["rejections"].get(
                "duplicate_text_sha256", 0
            ),
        },
        "rejection_histogram": manifest["accounting"]["rejections"],
    }


def _score_quantiles(score_counts: Counter, total: int) -> dict[str, Any]:
    """Exact quantiles from a discrete score histogram, without materializing the population."""
    ordered = sorted(score_counts)
    cumulative = 0
    marks = {
        "min": 0,
        "p05": int(0.05 * total),
        "p25": int(0.25 * total),
        "median": int(0.50 * total),
        "p75": int(0.75 * total),
        "p95": int(0.95 * total),
        "p99": int(0.99 * total),
        "max": total - 1,
    }
    result: dict[str, Any] = {"n": total}
    pending = sorted(marks.items(), key=lambda item: item[1])
    position = 0
    for score in ordered:
        cumulative += score_counts[score]
        while position < len(pending) and pending[position][1] < cumulative:
            result[pending[position][0]] = score
            position += 1
    while position < len(pending):
        result[pending[position][0]] = ordered[-1]
        position += 1
    result["mean"] = sum(score * count for score, count in score_counts.items()) / total
    return result


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PetitGPT full Common Pile Python Gate C candidate builder"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build the full Python candidate release")
    build.add_argument("--output-dir", required=True, type=Path)
    build.add_argument("--work-dir", required=True, type=Path)
    build.add_argument("--cache-dir", required=True, type=Path)
    build.add_argument("--max-wall-seconds", type=float, default=float(MAX_WALL_SECONDS))
    build.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    build.add_argument("--keep-shard-cache", action="store_true")
    build.add_argument("--stop-after-documents", type=int, default=None)

    verify = subparsers.add_parser("verify", help="verify a published full candidate release")
    verify.add_argument("--output-dir", required=True, type=Path)

    diagnose = subparsers.add_parser(
        "diagnose", help="full-population diagnostics for a published release"
    )
    diagnose.add_argument("--output-dir", required=True, type=Path)
    diagnose.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "build":
            result = build_full_candidates(
                FullBuildConfig(
                    source=PYTHON_SOURCE,
                    output_dir=args.output_dir,
                    work_dir=args.work_dir,
                    cache_dir=args.cache_dir,
                    max_wall_seconds=args.max_wall_seconds,
                    checkpoint_every=args.checkpoint_every,
                    keep_shard_cache=args.keep_shard_cache,
                    stop_after_documents=args.stop_after_documents,
                )
            )
        elif args.command == "verify":
            result = verify_release(args.output_dir)
        else:
            diagnostics = diagnose_release(args.output_dir)
            out = _ensure_under_workspace(args.out)
            _ensure_git_ignored(out)
            if out.exists():
                raise GateCPythonFullError(f"refusing to overwrite diagnostics output: {out}")
            out.parent.mkdir(parents=True, exist_ok=True)
            _write_new_file(out, json.dumps(diagnostics, indent=2, sort_keys=True).encode() + b"\n")
            result = {
                "out": str(out),
                "sha256": _sha256_bytes(out.read_bytes()),
                "documents": diagnostics["documents"],
                "proxy_tokens_4_bytes_per_token": diagnostics["proxy_tokens_4_bytes_per_token"],
                "canonical_token_counting_performed": False,
                "provisional_byte_weighted_only": True,
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
