#!/usr/bin/env python3
"""Stage-M output v1: canonical schema-3 packing, publication and strict validation.

One stage stream in, one canonical schema-3 packed release out. The binary representation is
not new work -- it is the existing canonical format that ``PackedBinDataset`` already reads --
so this module reuses the frozen writer geometry and the frozen release validator rather than
restating either.

What is Stage-M-specific:

* the stream is the accepted Stage-I records for one stage, framed ``[BOS] content [EOS]`` with
  no separator, concatenated so adjacent documents meet as ``[EOS][BOS]``;
* exactly ``retained_stored_token_ids = q*T + 1`` token IDs are written. Writing the final
  lookahead token and stopping is equivalent to writing all ``N`` and letting the reader drop
  the tail -- same blocks, same bytes per block -- but it makes the published release
  self-describing: a correct release has *no* reader-side tail left to drop;
* publication is staged and renamed atomically, and the canonical schema-3 completion state
  (``meta.json`` with ``status="complete"``) is written only after every shard is durable and
  every accounting check has passed.

There is deliberately no redundant ``COMPLETE`` file: schema 3 already has a canonical
completion authority and adding a second one would create two things that can disagree.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import ctypes
from dataclasses import dataclass, field
import errno
import hashlib
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from typing import Any

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pretrain.dataset_pretrain import validate_shard_release  # noqa: E402
from pretrain.stage_m_contract_v1 import (  # noqa: E402
    REQUIRED_BYTE_ORDER,
    StreamAccounting,
    canonical_json_bytes,
    file_sha256,
    require_int,
)
from src.special_tokens import BOS_ID, CANONICAL_VOCAB_SIZE, EOS_ID, SPECIAL_TOKEN_IDS  # noqa: E402

# R1-G: explicit little-endian, so the emitted bytes are defined by the dtype rather than by
# the host. The environment contract additionally refuses a non-little-endian runtime, which is
# what keeps this in agreement with the frozen schema-3 reader's native uint16.
STORAGE_DTYPE = np.dtype("<u2")
DEFAULT_SHARD_TOKENS = 10_000_000
SPLIT = "train"
MANIFEST_FILENAME = "meta.json"
FAILURE_FILENAME = "meta.failed.json"


class StageMOutputError(RuntimeError):
    """Controlled failure while packing, publishing or validating a Stage-M release."""


def _require(condition: object, message: str) -> None:
    if not condition:
        raise StageMOutputError(message)


def shard_basename(index: int) -> str:
    return f"shard_{index:05d}.bin"


def canonical_contract_block() -> dict[str, Any]:
    """The exact contract object the frozen schema-3 validator requires, field for field."""
    return {
        "mode": "canonical",
        "canonical": True,
        "legacy_allow_noncanonical_contract": False,
        "issues": [],
        "expected_special_token_ids": dict(SPECIAL_TOKEN_IDS),
        "expected_vocab_size": CANONICAL_VOCAB_SIZE,
        "actual_vocab_size": CANONICAL_VOCAB_SIZE,
        "add_bos": True,
        "add_eos": True,
        "bos_id": BOS_ID,
        "eos_id": EOS_ID,
        "doc_sep": "",
    }


@dataclass
class ShardWriter:
    """Fixed-geometry uint16 shard writer with bounded memory.

    Every shard but the last carries exactly ``shard_tokens`` IDs, because that is the geometry
    the frozen release validator recomputes. The buffer never exceeds one shard.
    """

    directory: Path
    shard_tokens: int
    records: list[dict[str, Any]] = field(default_factory=list)
    _buffer: list[int] = field(default_factory=list)
    _index: int = 0
    _total: int = 0

    def __post_init__(self) -> None:
        require_int(self.shard_tokens, field="shard_tokens", minimum=1)
        _require(
            sys.byteorder == REQUIRED_BYTE_ORDER,
            f"Stage-M v1 canonical output requires a {REQUIRED_BYTE_ORDER}-endian runtime, "
            f"got {sys.byteorder}",
        )
        self.directory.mkdir(parents=True, exist_ok=True)

    def extend(self, ids: Sequence[int]) -> None:
        self._buffer.extend(ids)
        while len(self._buffer) >= self.shard_tokens:
            self._flush(self.shard_tokens)

    def _flush(self, count: int) -> None:
        chunk = self._buffer[:count]
        del self._buffer[:count]
        array = np.asarray(chunk, dtype=STORAGE_DTYPE)
        _require(
            array.shape == (count,),
            f"shard buffer geometry error: {array.shape} != ({count},)",
        )
        name = shard_basename(self._index)
        path = self.directory / name
        with open(path, "wb") as handle:
            array.tofile(handle)
            handle.flush()
            os.fsync(handle.fileno())
        size_bytes = int(path.stat().st_size)
        _require(
            size_bytes == count * STORAGE_DTYPE.itemsize,
            f"{name}: byte geometry mismatch after write",
        )
        self.records.append({
            "path": (Path(SPLIT) / name).as_posix(),
            "size_bytes": size_bytes,
            "token_count": int(count),
            "sha256": file_sha256(path),
        })
        self._index += 1
        self._total += count

    def close(self) -> None:
        if self._buffer:
            self._flush(len(self._buffer))
        _require(self.records, "release must contain at least one shard")
        # R1-F: every shard file's bytes are already fsynced; this persists the directory
        # entries that name them.
        fsync_dir(self.directory)

    @property
    def total_tokens(self) -> int:
        return self._total

    @property
    def shard_count(self) -> int:
        return len(self.records)


def assert_token_ids(ids: Iterable[int], *, label: str) -> None:
    for value in ids:
        _require(
            isinstance(value, int) and 0 <= value < CANONICAL_VOCAB_SIZE,
            f"{label}: token id outside the canonical 32k range: {value!r}",
        )


@dataclass(frozen=True)
class PackedStream:
    """The realized packing result for one stage stream."""

    stage: str
    directory: Path
    accounting: StreamAccounting
    documents: int
    shard_records: tuple[Mapping[str, Any], ...]
    shard_tokens: int

    @property
    def shard_count(self) -> int:
        return len(self.shard_records)


def pack_stream(
    *,
    stage: str,
    documents: Iterable[Sequence[int]],
    accounting: StreamAccounting,
    directory: Path,
    shard_tokens: int = DEFAULT_SHARD_TOKENS,
) -> PackedStream:
    """Concatenate framed documents into one stream and stop at the retained-token boundary.

    ``documents`` yields the already-framed ``[BOS] content [EOS]`` id sequence per document, in
    Stage-M consumption order. The writer stops after ``retained_stored_token_ids`` IDs; the
    remaining ``tail_transitions`` IDs of the final document are the frozen dropped tail.
    """
    limit = accounting.retained_stored_token_ids
    writer = ShardWriter(directory=directory, shard_tokens=shard_tokens)
    emitted = 0
    consumed_serialized = 0
    doc_count = 0

    for ids in documents:
        doc_count += 1
        consumed_serialized += len(ids)
        if emitted >= limit:
            continue
        take = min(len(ids), limit - emitted)
        writer.extend(ids[:take])
        emitted += take
    writer.close()

    _require(
        writer.total_tokens == limit,
        f"{stage}: emitted {writer.total_tokens} stored token IDs, expected {limit}",
    )
    _require(
        consumed_serialized == accounting.input_serialized_tokens,
        f"{stage}: consumed {consumed_serialized} serialized input tokens, expected "
        f"{accounting.input_serialized_tokens}",
    )
    expected_shards = -(-limit // shard_tokens)
    _require(
        writer.shard_count == expected_shards,
        f"{stage}: produced {writer.shard_count} shards, expected {expected_shards}",
    )
    return PackedStream(
        stage=stage,
        directory=directory,
        accounting=accounting,
        documents=doc_count,
        shard_records=tuple(writer.records),
        shard_tokens=shard_tokens,
    )


def build_release_meta(
    packed: PackedStream,
    *,
    tokenizer_path: str,
    tokenizer_sha256: str,
    stage_m_binding: Mapping[str, Any],
    reference_exclusion: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the canonical schema-3 release manifest for one packed stage stream.

    The frozen validator requires an exact ``shard_files`` key set and a ``val_by_source`` map
    whose keys it must agree with. A Stage-M release holds no validation split at all -- the
    reference validation set is the separately frozen G2 release -- so both are present and
    empty rather than absent, which is the only shape the frozen validator accepts for
    "there is no validation here".
    """
    accounting = packed.accounting
    emitted = accounting.retained_stored_token_ids
    return {
        "schema_version": 3,
        "status": "complete",
        "tokenizer_path": tokenizer_path,
        "tokenizer_sha256": tokenizer_sha256,
        "dtype": "uint16",
        "vocab_size": CANONICAL_VOCAB_SIZE,
        "contract": canonical_contract_block(),
        "legacy_flags": {
            "allow_noncanonical_contract": False,
            "replay_on_exhaustion": False,
        },
        "source_exhaustion_policy": "fail_fast",
        "reference_validation_exclusion": dict(reference_exclusion),
        "stage_m": dict(stage_m_binding),
        "shard_tokens": int(packed.shard_tokens),
        "train_tokens": emitted,
        "train_shards": int(packed.shard_count),
        "val_tokens": 0,
        "val_shards": 0,
        "val_shard_tokens": int(packed.shard_tokens),
        "val_ratio": 0.0,
        "documents": int(packed.documents),
        "stage_m_accounting": accounting.as_canonical(),
        "accounting": {
            "train": {
                "documents": int(packed.documents),
                "content_tokens": accounting.input_serialized_tokens - 2 * int(packed.documents),
                "boundary_tokens": 2 * int(packed.documents),
                "serialized_tokens": accounting.input_serialized_tokens,
                "separator_tokens": 0,
                "emitted_shard_tokens": emitted,
            },
            "val": {
                "documents": 0,
                "content_tokens": 0,
                "boundary_tokens": 0,
                "serialized_tokens": 0,
                "separator_tokens": 0,
                "emitted_shard_tokens": 0,
            },
        },
        "val_by_source": {},
        "shard_files": {
            "hash_algorithm": "sha256",
            "train": [dict(record) for record in packed.shard_records],
            "val": [],
            "val_by_source": {},
        },
    }


def write_manifest(directory: Path, meta: Mapping[str, Any]) -> str:
    """Write the canonical completion object durably and return its digest."""
    payload = canonical_json_bytes(meta)
    path = directory / MANIFEST_FILENAME
    with open(path, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def write_failure_marker(directory: Path, reason: str) -> None:
    """Mark a staging directory unusable. A marked release can never load."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes({"status": "failed", "reason": reason})
    path = directory / FAILURE_FILENAME
    with open(path, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def fsync_dir(path: Path) -> None:
    """Durably persist a directory entry, so a created file is findable after a crash.

    Writing and fsyncing a file persists its *contents*; the directory entry that names it is a
    separate write. R1-F: the nested ``train/`` directory holding the shards must be synced
    after the shards are finalized, before the staging root is synced and before the rename
    that publishes the release.
    """
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


_fsync_dir = fsync_dir


# R2-F. `os.rename` on a directory REPLACES an empty destination, so `exists()` followed by
# `rename()` is a check-then-act with a real window: a destination created between the two calls
# is silently consumed. The kernel primitive that closes it is renameat2(RENAME_NOREPLACE),
# which either creates the destination or fails with EEXIST, with no window in between.
RENAME_NOREPLACE = 1
AT_FDCWD = -100
# renameat2 syscall numbers for the architectures this project supports.
_RENAMEAT2_SYSCALL = {"x86_64": 316, "aarch64": 276}


class AtomicPublicationUnsupported(StageMOutputError):
    """No atomic no-replace rename is available. Publication stops rather than weakening."""


def _renameat2_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish ``source`` as ``destination``, never replacing an existing object."""
    libc = ctypes.CDLL(None, use_errno=True)
    src = os.fsencode(str(source))
    dst = os.fsencode(str(destination))

    handler = getattr(libc, "renameat2", None)
    if handler is not None:
        handler.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        handler.restype = ctypes.c_int
        ctypes.set_errno(0)
        result = handler(AT_FDCWD, src, AT_FDCWD, dst, RENAME_NOREPLACE)
    else:
        number = _RENAMEAT2_SYSCALL.get(platform.machine())
        if number is None:
            raise AtomicPublicationUnsupported(
                "no atomic no-replace rename is available on this platform "
                f"({platform.machine()}); refusing to publish with weaker semantics"
            )
        libc.syscall.argtypes = [
            ctypes.c_long,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        libc.syscall.restype = ctypes.c_long
        ctypes.set_errno(0)
        result = libc.syscall(number, AT_FDCWD, src, AT_FDCWD, dst, RENAME_NOREPLACE)

    if result == 0:
        return
    code = ctypes.get_errno()
    if code == errno.EEXIST or code == errno.ENOTEMPTY:
        raise StageMOutputError(f"refusing to replace an existing Stage-M release: {destination}")
    if code in (errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP):
        # The kernel or the filesystem does not implement RENAME_NOREPLACE. Falling back to a
        # replacing rename would reintroduce exactly the defect this function exists to remove.
        raise AtomicPublicationUnsupported(
            "RENAME_NOREPLACE is not supported by this kernel or filesystem "
            f"(errno {code}); refusing to publish with weaker semantics"
        )
    raise StageMOutputError(
        f"atomic publication failed: {source} -> {destination}: [Errno {code}] {os.strerror(code)}"
    )


def publish_release_atomic(staging: Path, destination: Path) -> Path:
    """Publish a fully built staging directory atomically, never replacing a destination.

    There is deliberately no ``exists()`` pre-check driving the decision: the kernel makes the
    create-or-fail choice in one operation. A destination that appears between the last fsync
    and the publication is left untouched and this call fails cleanly, leaving the staging
    directory intact for the caller to discard.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # R1-F ordering, preserved: nested shard directories first, then the staging root, then the
    # publication, then the destination's parent.
    for child in sorted(p for p in staging.iterdir() if p.is_dir()):
        fsync_dir(child)
    fsync_dir(staging)
    _renameat2_noreplace(staging, destination)
    fsync_dir(destination.parent)
    return destination


def validate_published_release(release_root: Path) -> dict[str, Any]:
    """Strict consumer check: the frozen validator, on the published bytes.

    This is the same function the production trainer calls before opening a memmap, so passing
    it means a real ``PackedBinDataset`` will accept the release.
    """
    result = validate_shard_release(Path(release_root) / SPLIT)
    _require(
        result.get("dtype") == "uint16",
        f"published release dtype must be uint16, got {result.get('dtype')!r}",
    )
    return result


def verify_release_against_accounting(
    release_root: Path, accounting: StreamAccounting
) -> dict[str, Any]:
    """Prove a published release matches the frozen expected accounting, then read it back.

    The release is validated by the frozen validator first, then the derived block geometry is
    checked against the plan: sequence count, model input positions, and the requirement that a
    correct Stage-M release leaves the reader no tail at all.
    """
    result = validate_published_release(release_root)
    emitted = int(result["expected_tokens"])
    _require(
        emitted == accounting.retained_stored_token_ids,
        f"{accounting.stage}: published {emitted} stored token IDs, plan expects "
        f"{accounting.retained_stored_token_ids}",
    )
    transitions = emitted - 1
    sequences = transitions // accounting.seq_len
    _require(
        sequences == accounting.training_sequences,
        f"{accounting.stage}: published stream yields {sequences} sequences, plan expects "
        f"{accounting.training_sequences}",
    )
    _require(
        sequences * accounting.seq_len == accounting.model_input_positions,
        f"{accounting.stage}: model input positions disagree with the plan",
    )
    _require(
        transitions - sequences * accounting.seq_len == 0,
        f"{accounting.stage}: a published Stage-M release must leave no reader-side tail",
    )
    return result


def staging_directory(destination: Path) -> Path:
    """A sibling staging directory, so the rename into place stays within one filesystem."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=str(destination.parent))
    )


def discard_staging(staging: Path) -> None:
    shutil.rmtree(staging, ignore_errors=True)


__all__ = [
    "DEFAULT_SHARD_TOKENS",
    "FAILURE_FILENAME",
    "MANIFEST_FILENAME",
    "SPLIT",
    "STORAGE_DTYPE",
    "PackedStream",
    "ShardWriter",
    "AtomicPublicationUnsupported",
    "RENAME_NOREPLACE",
    "StageMOutputError",
    "assert_token_ids",
    "build_release_meta",
    "canonical_contract_block",
    "discard_staging",
    "fsync_dir",
    "pack_stream",
    "publish_release_atomic",
    "shard_basename",
    "staging_directory",
    "validate_published_release",
    "verify_release_against_accounting",
    "write_failure_marker",
    "write_manifest",
]
