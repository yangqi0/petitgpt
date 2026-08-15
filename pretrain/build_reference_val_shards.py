#!/usr/bin/env python3

"""Build one immutable, cross-stage reference-validation corpus.

Each source is scanned once. Documents are ranked by a seeded hash of cleaned
content, and the smallest stable-hash prefix that reaches the requested
serialized-token target is retained. This makes selection independent of JSONL
order while keeping memory proportional to the small reference set, not the
input corpus.

The output directory contains combined ``val/`` shards, matching
``val_by_source/<name>/`` shards, ``manifest.json``, and the
``exclusion_hash_manifest.json`` that every production train/control shard
build must consume. The manifest binds every shard by ordered relative path,
byte size, token count, and SHA-256.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pretrain.build_pretrain_shards import (  # noqa: E402
    CLEANED_TEXT_HASH_ALGORITHM,
    EXCLUSION_MANIFEST_KIND,
    _tokenizer_vocab_size,
    _write_json_atomic,
    ascii_ratio,
    assert_token_ids_ok,
    clean_text,
    cleaned_text_sha256,
    cleaning_contract,
    encode_with_accounting,
    file_fingerprint,
    load_exclusion_hash_manifest,
    load_tokenizer,
    validate_build_contract,
)
from src.special_tokens import BOS_ID, EOS_ID, SPECIAL_TOKEN_IDS  # noqa: E402

SELECTION_ALGORITHM = "blake2b-128-seed-plus-cleaned-sha256-v1"
MANIFEST_NAME = "manifest.json"
EXCLUSION_MANIFEST_NAME = "exclusion_hash_manifest.json"
RESERVE_MANIFEST_NAME = "reserve_manifest.json"
RESERVE_MANIFEST_KIND = "petitgpt_reference_validation_reserve"
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]")
_SPECIAL_IDS = frozenset(SPECIAL_TOKEN_IDS.values())


class ReferenceSourceExhaustedError(RuntimeError):
    """A reference source cannot meet its complete-document token target."""


@dataclass(frozen=True)
class ReferenceSource:
    path: Path
    target_serialized_tokens: int
    output_name: str


@dataclass(frozen=True)
class Candidate:
    selection_rank: int
    selection_rank_hex: str
    cleaned_text_sha256: str
    ids: tuple[int, ...]
    content_tokens: int
    boundary_tokens: int
    cleaned_chars: int

    @property
    def serialized_tokens(self) -> int:
        return len(self.ids)


@dataclass(frozen=True)
class ReserveCandidate:
    selection_rank: int
    selection_rank_hex: str
    cleaned_text_sha256: str
    cleaned_utf8_bytes: int


def _selection_rank(clean_hash: str, seed: int) -> tuple[int, str]:
    digest = hashlib.blake2b(digest_size=16, person=b"PetitGPT-ref-v1")
    digest.update(str(int(seed)).encode("ascii"))
    digest.update(b"\0")
    digest.update(bytes.fromhex(clean_hash))
    raw = digest.digest()
    return int.from_bytes(raw, "big", signed=False), raw.hex()


def _safe_source_names(paths: list[Path]) -> dict[str, str]:
    resolved = [str(path.resolve()) for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("duplicate --source paths are not allowed")

    bases = [(_SAFE_NAME_RE.sub("_", path.stem) or "source") for path in paths]
    base_counts = {base: bases.count(base) for base in set(bases)}
    result: dict[str, str] = {}
    for path, resolved_path, base in zip(paths, resolved, bases, strict=True):
        name = base
        if base_counts[base] > 1:
            suffix = hashlib.sha256(resolved_path.encode("utf-8")).hexdigest()[:10]
            name = f"{base}_{suffix}"
        result[str(path)] = name
    if len(set(result.values())) != len(result):
        raise AssertionError("failed to derive unique source output names")
    return result


def parse_reference_sources(source_args: list[str]) -> list[ReferenceSource]:
    """Parse repeatable ``path:target_serialized_tokens`` source arguments."""
    parsed: list[tuple[Path, int]] = []
    for value in source_args:
        if ":" not in value:
            raise ValueError(
                "--source must be path:target_serialized_tokens, "
                f"got {value!r}"
            )
        raw_path, raw_target = value.rsplit(":", 1)
        path = Path(raw_path.strip())
        try:
            target = int(raw_target.strip())
        except ValueError as exc:
            raise ValueError(f"invalid serialized-token target in --source {value!r}") from exc
        if target <= 0:
            raise ValueError(f"reference source target must be > 0, got {target}")
        parsed.append((path, target))

    if not parsed:
        raise ValueError("at least one --source is required")
    names = _safe_source_names([path for path, _ in parsed])
    return [
        ReferenceSource(
            path=path,
            target_serialized_tokens=target,
            output_name=names[str(path)],
        )
        for path, target in parsed
    ]


def _empty_scan_stats() -> dict[str, int]:
    return {
        "lines": 0,
        "blank_lines": 0,
        "malformed_json": 0,
        "non_object": 0,
        "missing_or_non_string_text": 0,
        "empty_text": 0,
        "dropped_empty_after_cleaning": 0,
        "dropped_short": 0,
        "dropped_ascii": 0,
        "eligible_documents": 0,
        "duplicate_selected_candidates": 0,
        "not_in_reserved_pool": 0,
    }


def _reserve_source(
    source: ReferenceSource,
    *,
    seed: int,
    target_cleaned_utf8_bytes: int,
    strip_leading_noise: bool,
    normalize_quotes: bool,
    underscores_policy: str,
    min_chars: int,
    min_ascii_ratio: float,
) -> tuple[list[ReserveCandidate], dict[str, int]]:
    """Select a tokenizer-independent, stable, deliberately oversized pool."""
    if not source.path.is_file():
        raise FileNotFoundError(
            f"reference source does not exist or is not a file: {source.path}"
        )

    heap: list[tuple[int, int, ReserveCandidate]] = []
    selected_hashes: set[str] = set()
    selected_bytes = 0
    stats = _empty_scan_stats()
    with open(source.path, encoding="utf-8") as f:
        for line in f:
            stats["lines"] += 1
            if not line.strip():
                stats["blank_lines"] += 1
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats["malformed_json"] += 1
                continue
            if not isinstance(obj, dict):
                stats["non_object"] += 1
                continue
            raw_text = obj.get("text")
            if not isinstance(raw_text, str):
                stats["missing_or_non_string_text"] += 1
                continue
            if not raw_text:
                stats["empty_text"] += 1
                continue
            cleaned = clean_text(
                raw_text,
                strip_leading_noise=strip_leading_noise,
                normalize_quotes=normalize_quotes,
                underscores_policy=underscores_policy,
                min_chars=0,
                min_ascii_ratio=0.0,
            )
            if cleaned is None:
                stats["dropped_empty_after_cleaning"] += 1
                continue
            if min_chars > 0 and len(cleaned) < min_chars:
                stats["dropped_short"] += 1
                continue
            if min_ascii_ratio > 0.0 and ascii_ratio(cleaned) < min_ascii_ratio:
                stats["dropped_ascii"] += 1
                continue

            stats["eligible_documents"] += 1
            clean_hash = cleaned_text_sha256(cleaned)
            if clean_hash in selected_hashes:
                stats["duplicate_selected_candidates"] += 1
                continue
            rank, rank_hex = _selection_rank(clean_hash, seed)
            candidate = ReserveCandidate(
                selection_rank=rank,
                selection_rank_hex=rank_hex,
                cleaned_text_sha256=clean_hash,
                cleaned_utf8_bytes=len(cleaned.encode("utf-8")),
            )
            heapq.heappush(heap, (-rank, -int(clean_hash, 16), candidate))
            selected_hashes.add(clean_hash)
            selected_bytes += candidate.cleaned_utf8_bytes
            while len(heap) > 1:
                worst = heap[0][2]
                if selected_bytes - worst.cleaned_utf8_bytes < target_cleaned_utf8_bytes:
                    break
                _, _, removed = heapq.heappop(heap)
                selected_hashes.remove(removed.cleaned_text_sha256)
                selected_bytes -= removed.cleaned_utf8_bytes

    if selected_bytes < target_cleaned_utf8_bytes:
        raise ReferenceSourceExhaustedError(
            f"reference reserve source exhausted before byte quota: {source.path}; "
            f"target_cleaned_utf8_bytes={target_cleaned_utf8_bytes}; "
            f"available_selected_cleaned_utf8_bytes={selected_bytes}"
        )
    selected = [item[2] for item in heap]
    selected.sort(key=lambda item: (item.selection_rank, item.cleaned_text_sha256))
    return selected, stats


def reserve_reference_candidates(
    *,
    sources: list[ReferenceSource],
    out_dir: Path,
    seed: int,
    reserve_bytes_per_target_token: float = 32.0,
    strip_leading_noise: bool = False,
    normalize_quotes: bool = False,
    underscores_policy: str = "keep",
    min_chars: int = 0,
    min_ascii_ratio: float = 0.0,
) -> dict[str, Any]:
    """Reserve candidates before tokenizer training and publish their hashes."""
    if not sources:
        raise ValueError("at least one reference source is required")
    if reserve_bytes_per_target_token <= 0:
        raise ValueError("reserve_bytes_per_target_token must be > 0")
    if out_dir.exists():
        raise FileExistsError(
            f"reference reserve output must not already exist (immutable build): {out_dir}"
        )
    if underscores_policy not in {"keep", "space", "remove"}:
        raise ValueError(f"invalid underscores_policy: {underscores_policy!r}")
    if min_chars < 0 or not 0.0 <= min_ascii_ratio <= 1.0:
        raise ValueError("invalid reference cleaning filter values")

    source_paths = [str(source.path.resolve()) for source in sources]
    if len(set(source_paths)) != len(source_paths):
        raise ValueError("duplicate reference source paths are not allowed")
    active_cleaning = cleaning_contract(
        strip_leading_noise=strip_leading_noise,
        normalize_quotes=normalize_quotes,
        underscores_policy=underscores_policy,
        min_chars=min_chars,
        min_ascii_ratio=min_ascii_ratio,
    )
    by_output_name = {source.output_name: source for source in sources}
    if len(by_output_name) != len(sources):
        raise ValueError("duplicate reference source output names are not allowed")

    source_meta: dict[str, Any] = {}
    all_hash_sources: dict[str, set[str]] = {}
    for output_name in sorted(by_output_name):
        source = by_output_name[output_name]
        byte_target = max(
            1,
            math.ceil(
                source.target_serialized_tokens
                * float(reserve_bytes_per_target_token)
            ),
        )
        selected, scan = _reserve_source(
            source,
            seed=seed,
            target_cleaned_utf8_bytes=byte_target,
            strip_leading_noise=strip_leading_noise,
            normalize_quotes=normalize_quotes,
            underscores_policy=underscores_policy,
            min_chars=min_chars,
            min_ascii_ratio=min_ascii_ratio,
        )
        for item in selected:
            all_hash_sources.setdefault(item.cleaned_text_sha256, set()).add(output_name)
        realized_bytes = sum(item.cleaned_utf8_bytes for item in selected)
        source_meta[output_name] = {
            "path": str(source.path),
            "resolved_path": str(source.path.resolve()),
            "source_fingerprint": file_fingerprint(source.path),
            "final_target_serialized_tokens": source.target_serialized_tokens,
            "reserve_target_cleaned_utf8_bytes": byte_target,
            "reserve_realized_cleaned_utf8_bytes": realized_bytes,
            "reserve_documents": len(selected),
            "scan": scan,
            "reserved_documents": [
                {
                    "selection_rank": item.selection_rank_hex,
                    "cleaned_text_sha256": item.cleaned_text_sha256,
                    "cleaned_utf8_bytes": item.cleaned_utf8_bytes,
                }
                for item in selected
            ],
        }

    hashes = sorted(all_hash_sources)
    reserve_manifest = {
        "schema_version": 1,
        "status": "complete",
        "kind": RESERVE_MANIFEST_KIND,
        "immutable": True,
        "tokenizer_independent": True,
        "selection": {
            "algorithm": SELECTION_ALGORITHM,
            "seed": int(seed),
            "reserve_bytes_per_target_token": float(
                reserve_bytes_per_target_token
            ),
            "input_order_independent": True,
        },
        "cleaning": active_cleaning,
        "sources": source_meta,
        "outputs": {"exclusion_hash_manifest": EXCLUSION_MANIFEST_NAME},
        "unique_reserved_hashes": len(hashes),
    }
    exclusion_manifest = {
        "schema_version": 1,
        "kind": EXCLUSION_MANIFEST_KIND,
        "hash_algorithm": CLEANED_TEXT_HASH_ALGORITHM,
        "membership_basis": "cleaned document text encoded as UTF-8",
        "cleaning": active_cleaning,
        "hash_count": len(hashes),
        "hashes": hashes,
        "hash_sources": {
            value: sorted(all_hash_sources[value]) for value in hashes
        },
        "reference_reserve_manifest": RESERVE_MANIFEST_NAME,
    }

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{out_dir.name}.tmp-", dir=str(out_dir.parent))
    )
    try:
        _write_json_atomic(staging / EXCLUSION_MANIFEST_NAME, exclusion_manifest)
        _write_json_atomic(staging / RESERVE_MANIFEST_NAME, reserve_manifest)
        os.replace(staging, out_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return reserve_manifest


def _select_source(
    source: ReferenceSource,
    *,
    tok: Any,
    vocab_size: int | None,
    dtype: np.dtype,
    seed: int,
    strip_leading_noise: bool,
    normalize_quotes: bool,
    underscores_policy: str,
    min_chars: int,
    min_ascii_ratio: float,
    allowed_hashes: frozenset[str] | None = None,
) -> tuple[list[Candidate], dict[str, int]]:
    """Return the stable-hash prefix sufficient for one source quota."""
    if not source.path.is_file():
        raise FileNotFoundError(
            f"reference source does not exist or is not a file: {source.path}"
        )

    # Python's heap is a min-heap. Negative keys put the worst retained rank at
    # heap[0], allowing it to be evicted once all better candidates meet quota.
    heap: list[tuple[int, int, Candidate]] = []
    selected_hashes: set[str] = set()
    selected_tokens = 0
    stats = _empty_scan_stats()
    found_allowed_hashes: set[str] = set()

    with open(source.path, encoding="utf-8") as f:
        for line in f:
            stats["lines"] += 1
            if not line.strip():
                stats["blank_lines"] += 1
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats["malformed_json"] += 1
                continue
            if not isinstance(obj, dict):
                stats["non_object"] += 1
                continue
            raw_text = obj.get("text")
            if not isinstance(raw_text, str):
                stats["missing_or_non_string_text"] += 1
                continue
            if not raw_text:
                stats["empty_text"] += 1
                continue

            cleaned = clean_text(
                raw_text,
                strip_leading_noise=strip_leading_noise,
                normalize_quotes=normalize_quotes,
                underscores_policy=underscores_policy,
                min_chars=0,
                min_ascii_ratio=0.0,
            )
            if cleaned is None:
                stats["dropped_empty_after_cleaning"] += 1
                continue
            if min_chars > 0 and len(cleaned) < min_chars:
                stats["dropped_short"] += 1
                continue
            if min_ascii_ratio > 0.0 and ascii_ratio(cleaned) < min_ascii_ratio:
                stats["dropped_ascii"] += 1
                continue

            stats["eligible_documents"] += 1
            clean_hash = cleaned_text_sha256(cleaned)
            if allowed_hashes is not None:
                if clean_hash not in allowed_hashes:
                    stats["not_in_reserved_pool"] += 1
                    continue
                found_allowed_hashes.add(clean_hash)
            if clean_hash in selected_hashes:
                stats["duplicate_selected_candidates"] += 1
                continue

            ids, content_tokens, boundary_tokens = encode_with_accounting(
                tok,
                cleaned,
                add_bos=True,
                add_eos=True,
                bos_id=BOS_ID,
                eos_id=EOS_ID,
            )
            assert_token_ids_ok(
                ids,
                vocab_size=vocab_size,
                dtype=dtype,
                src=str(source.path),
                text_preview=cleaned[:200],
            )
            injected = _SPECIAL_IDS.intersection(ids[1:-1])
            if injected:
                raise ValueError(
                    "ordinary corpus text encoded to true special-token IDs "
                    f"{sorted(injected)} in {source.path}; literal-special injection refused"
                )

            rank, rank_hex = _selection_rank(clean_hash, seed)
            candidate = Candidate(
                selection_rank=rank,
                selection_rank_hex=rank_hex,
                cleaned_text_sha256=clean_hash,
                ids=tuple(ids),
                content_tokens=content_tokens,
                boundary_tokens=boundary_tokens,
                cleaned_chars=len(cleaned),
            )
            clean_int = int(clean_hash, 16)
            heapq.heappush(heap, (-rank, -clean_int, candidate))
            selected_hashes.add(clean_hash)
            selected_tokens += candidate.serialized_tokens

            while len(heap) > 1:
                worst = heap[0][2]
                if selected_tokens - worst.serialized_tokens < source.target_serialized_tokens:
                    break
                _, _, removed = heapq.heappop(heap)
                selected_hashes.remove(removed.cleaned_text_sha256)
                selected_tokens -= removed.serialized_tokens

    if allowed_hashes is not None:
        missing = allowed_hashes.difference(found_allowed_hashes)
        stats["reserved_hashes_expected"] = len(allowed_hashes)
        stats["reserved_hashes_found"] = len(found_allowed_hashes)
        if missing:
            preview = sorted(missing)[:3]
            raise ReferenceSourceExhaustedError(
                f"reserved documents disappeared from source {source.path}: "
                f"missing={len(missing)}, preview={preview}"
            )

    if selected_tokens < source.target_serialized_tokens:
        raise ReferenceSourceExhaustedError(
            f"reference source exhausted before quota: {source.path}; "
            f"target_serialized_tokens={source.target_serialized_tokens}; "
            f"available_selected_serialized_tokens={selected_tokens}; "
            f"eligible_documents={stats['eligible_documents']}"
        )

    selected = [item[2] for item in heap]
    selected.sort(key=lambda item: (item.selection_rank, item.cleaned_text_sha256))
    if sum(item.serialized_tokens for item in selected) != selected_tokens:
        raise AssertionError("reference selection token accounting mismatch")
    return selected, stats


def _write_bin_atomic(
    path: Path,
    ids: list[int],
    dtype: np.dtype,
    *,
    release_root: Path,
) -> dict[str, Any]:
    """Atomically write one shard and fingerprint the exact byte image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    arr = np.asarray(ids, dtype=dtype)
    sha256 = hashlib.sha256(memoryview(arr).cast("B")).hexdigest()
    with open(tmp, "wb") as f:
        arr.tofile(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    if path.stat().st_size != arr.nbytes:
        raise OSError(
            f"short shard write for {path}: "
            f"actual={path.stat().st_size}, expected={arr.nbytes}"
        )
    return {
        "path": path.relative_to(release_root).as_posix(),
        "size_bytes": int(arr.nbytes),
        "token_count": int(arr.size),
        "sha256": sha256,
    }


class _ShardWriter:
    def __init__(
        self,
        directory: Path,
        *,
        release_root: Path,
        shard_tokens: int,
        dtype: np.dtype,
    ):
        self.directory = directory
        self.release_root = release_root
        self.shard_tokens = int(shard_tokens)
        self.dtype = dtype
        self.buffer: list[int] = []
        self.shards = 0
        self.tokens = 0
        self.files: list[dict[str, Any]] = []

    def append(self, ids: tuple[int, ...]) -> None:
        self.buffer.extend(ids)
        while len(self.buffer) >= self.shard_tokens:
            chunk = self.buffer[: self.shard_tokens]
            del self.buffer[: self.shard_tokens]
            self._write(chunk)

    def finish(self) -> None:
        if self.buffer:
            chunk = self.buffer
            self.buffer = []
            self._write(chunk)

    def _write(self, chunk: list[int]) -> None:
        path = self.directory / f"shard_{self.shards:05d}.bin"
        record = _write_bin_atomic(
            path,
            chunk,
            self.dtype,
            release_root=self.release_root,
        )
        self.files.append(record)
        self.tokens += int(record["token_count"])
        self.shards += 1


def _selected_document_meta(candidate: Candidate) -> dict[str, Any]:
    return {
        "selection_rank": candidate.selection_rank_hex,
        "cleaned_text_sha256": candidate.cleaned_text_sha256,
        "cleaned_chars": candidate.cleaned_chars,
        "content_tokens": candidate.content_tokens,
        "boundary_tokens": candidate.boundary_tokens,
        "serialized_tokens": candidate.serialized_tokens,
    }


def load_reference_reserve(
    reserve_manifest_path: Path,
) -> tuple[
    list[ReferenceSource],
    dict[str, frozenset[str]],
    frozenset[str],
    dict[str, Any],
]:
    """Load a phase-one reserve and verify it against its exclusion manifest."""
    try:
        before = reserve_manifest_path.stat()
        raw_reserve = reserve_manifest_path.read_bytes()
        after = reserve_manifest_path.stat()
        before_identity = (
            before.st_size,
            before.st_mtime_ns,
            before.st_ino,
            before.st_dev,
        )
        after_identity = (
            after.st_size,
            after.st_mtime_ns,
            after.st_ino,
            after.st_dev,
        )
        if before_identity != after_identity:
            raise RuntimeError(
                f"reference reserve manifest changed while reading: {reserve_manifest_path}"
            )
        reserve = json.loads(raw_reserve)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot read reference reserve manifest {reserve_manifest_path}: {exc}"
        ) from exc
    if not isinstance(reserve, dict) or reserve.get("kind") != RESERVE_MANIFEST_KIND:
        raise ValueError(f"not a PetitGPT reference reserve: {reserve_manifest_path}")
    if reserve.get("status") != "complete" or reserve.get("tokenizer_independent") is not True:
        raise ValueError("reference reserve is incomplete or not tokenizer-independent")
    selection = reserve.get("selection")
    if not isinstance(selection, dict) or selection.get("algorithm") != SELECTION_ALGORITHM:
        raise ValueError("reference reserve selection contract mismatch")
    cleaning = reserve.get("cleaning")
    if not isinstance(cleaning, dict):
        raise ValueError("reference reserve cleaning contract is missing")
    expected_cleaning_keys = {
        "strip_leading_noise",
        "normalize_quotes",
        "underscores_policy",
        "min_chars",
        "min_ascii_ratio",
    }
    if set(cleaning) != expected_cleaning_keys:
        raise ValueError(
            "reference reserve cleaning contract has unexpected keys: "
            f"{sorted(cleaning)}"
        )
    raw_sources = reserve.get("sources")
    if not isinstance(raw_sources, dict) or not raw_sources:
        raise ValueError("reference reserve has no sources")

    sources: list[ReferenceSource] = []
    reserved_by_source: dict[str, frozenset[str]] = {}
    union: set[str] = set()
    selection_seed = int(selection.get("seed", 0))
    for output_name in sorted(raw_sources):
        if (
            not output_name
            or output_name in {".", ".."}
            or Path(output_name).name != output_name
            or _SAFE_NAME_RE.search(output_name) is not None
        ):
            raise ValueError(f"unsafe reserve source output name: {output_name!r}")
        item = raw_sources[output_name]
        if not isinstance(item, dict):
            raise ValueError(f"invalid reserve source entry: {output_name!r}")
        raw_resolved_path = item.get("resolved_path")
        if not isinstance(raw_resolved_path, str) or not raw_resolved_path:
            raise ValueError(f"reserve source resolved path is missing: {output_name!r}")
        path = Path(raw_resolved_path)
        if not path.is_absolute():
            raise ValueError(f"reserve source path is not absolute: {output_name!r}")
        target = int(item.get("final_target_serialized_tokens", 0))
        if target <= 0:
            raise ValueError(f"invalid reserve path/target for {output_name!r}")
        documents = item.get("reserved_documents")
        if not isinstance(documents, list) or not documents:
            raise ValueError(f"reserve source has no candidate documents: {output_name!r}")
        hashes: list[str] = []
        for document in documents:
            value = document.get("cleaned_text_sha256") if isinstance(document, dict) else None
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(f"invalid reserved hash for {output_name!r}: {value!r}")
            expected_rank = _selection_rank(value, selection_seed)[1]
            if document.get("selection_rank") != expected_rank:
                raise ValueError(f"reserved selection rank mismatch for {output_name!r}")
            if int(document.get("cleaned_utf8_bytes", 0)) <= 0:
                raise ValueError(f"invalid reserved byte count for {output_name!r}")
            hashes.append(value)
        if len(set(hashes)) != len(hashes):
            raise ValueError(f"duplicate reserved hashes for {output_name!r}")
        sources.append(
            ReferenceSource(
                path=path,
                target_serialized_tokens=target,
                output_name=output_name,
            )
        )
        reserved_by_source[output_name] = frozenset(hashes)
        union.update(hashes)

    outputs = reserve.get("outputs")
    exclusion_name = outputs.get("exclusion_hash_manifest") if isinstance(outputs, dict) else None
    if not isinstance(exclusion_name, str) or Path(exclusion_name).name != exclusion_name:
        raise ValueError("reference reserve exclusion manifest path is invalid")
    exclusion_path = reserve_manifest_path.parent / exclusion_name
    exclusion_hashes, exclusion_meta = load_exclusion_hash_manifest(
        exclusion_path,
        expected_cleaning=cleaning,
    )
    if exclusion_hashes != union:
        raise ValueError(
            "reference reserve and exclusion hash manifest disagree: "
            f"reserve={len(union)}, exclusion={len(exclusion_hashes)}"
        )
    provenance = {
        "reserve_manifest_path": str(reserve_manifest_path),
        "reserve_manifest_size_bytes": len(raw_reserve),
        "reserve_manifest_sha256": hashlib.sha256(raw_reserve).hexdigest(),
        "reserve_exclusion": exclusion_meta,
        "selection": selection,
        "cleaning": cleaning,
    }
    return sources, reserved_by_source, exclusion_hashes, provenance


def build_reference_validation(
    *,
    sources: list[ReferenceSource],
    out_dir: Path,
    tokenizer_path: str,
    shard_tokens: int,
    seed: int,
    strip_leading_noise: bool = False,
    normalize_quotes: bool = False,
    underscores_policy: str = "keep",
    min_chars: int = 0,
    min_ascii_ratio: float = 0.0,
    reserved_hashes_by_source: dict[str, frozenset[str]] | None = None,
    exclusion_hashes_override: frozenset[str] | None = None,
    reserve_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and atomically publish the fixed cross-stage validation set."""
    if not sources:
        raise ValueError("at least one reference source is required")
    if shard_tokens <= 0:
        raise ValueError("shard_tokens must be > 0")
    if out_dir.exists():
        raise FileExistsError(
            f"reference output must not already exist (immutable build): {out_dir}"
        )
    if underscores_policy not in {"keep", "space", "remove"}:
        raise ValueError(f"invalid underscores_policy: {underscores_policy!r}")
    if min_chars < 0:
        raise ValueError("min_chars must be >= 0")
    if not 0.0 <= min_ascii_ratio <= 1.0:
        raise ValueError("min_ascii_ratio must be in [0,1]")

    source_paths = [str(source.path.resolve()) for source in sources]
    if len(set(source_paths)) != len(source_paths):
        raise ValueError("duplicate reference source paths are not allowed")
    output_names = [source.output_name for source in sources]
    if len(set(output_names)) != len(output_names):
        raise ValueError("duplicate reference source output names are not allowed")

    contract = validate_build_contract(
        tokenizer_path,
        add_bos=True,
        add_eos=True,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        doc_sep="",
    )
    tok = load_tokenizer(tokenizer_path)
    vocab_size = _tokenizer_vocab_size(tok)
    dtype = np.uint32 if (vocab_size is not None and vocab_size > 65535) else np.uint16
    active_cleaning = cleaning_contract(
        strip_leading_noise=strip_leading_noise,
        normalize_quotes=normalize_quotes,
        underscores_policy=underscores_policy,
        min_chars=min_chars,
        min_ascii_ratio=min_ascii_ratio,
    )

    selected_by_source: dict[str, list[Candidate]] = {}
    scan_by_source: dict[str, dict[str, int]] = {}
    by_output_name = {source.output_name: source for source in sources}
    if reserved_hashes_by_source is not None:
        if set(reserved_hashes_by_source) != set(by_output_name):
            raise ValueError("reserved hash source names do not match final sources")
    for output_name in sorted(by_output_name):
        source = by_output_name[output_name]
        selected, scan_stats = _select_source(
            source,
            tok=tok,
            vocab_size=vocab_size,
            dtype=dtype,
            seed=seed,
            strip_leading_noise=strip_leading_noise,
            normalize_quotes=normalize_quotes,
            underscores_policy=underscores_policy,
            min_chars=min_chars,
            min_ascii_ratio=min_ascii_ratio,
            allowed_hashes=(
                reserved_hashes_by_source[output_name]
                if reserved_hashes_by_source is not None
                else None
            ),
        )
        selected_by_source[output_name] = selected
        scan_by_source[output_name] = scan_stats

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{out_dir.name}.tmp-", dir=str(out_dir.parent))
    )
    try:
        combined_writer = _ShardWriter(
            staging / "val",
            release_root=staging,
            shard_tokens=shard_tokens,
            dtype=dtype,
        )
        per_source_output: dict[str, dict[str, Any]] = {}
        per_source_shard_files: dict[str, list[dict[str, Any]]] = {}
        all_hash_sources: dict[str, set[str]] = {}
        total_documents = 0
        total_content_tokens = 0
        total_boundary_tokens = 0
        total_serialized_tokens = 0

        for output_name in sorted(selected_by_source):
            source = by_output_name[output_name]
            writer = _ShardWriter(
                staging / "val_by_source" / output_name,
                release_root=staging,
                shard_tokens=shard_tokens,
                dtype=dtype,
            )
            candidates = selected_by_source[output_name]
            for candidate in candidates:
                combined_writer.append(candidate.ids)
                writer.append(candidate.ids)
                all_hash_sources.setdefault(candidate.cleaned_text_sha256, set()).add(
                    output_name
                )
            writer.finish()

            content_tokens = sum(item.content_tokens for item in candidates)
            boundary_tokens = sum(item.boundary_tokens for item in candidates)
            serialized_tokens = sum(item.serialized_tokens for item in candidates)
            if writer.tokens != serialized_tokens:
                raise AssertionError(f"per-source shard accounting mismatch: {output_name}")
            per_source_shard_files[output_name] = writer.files
            per_source_output[output_name] = {
                "path": str(source.path),
                "resolved_path": str(source.path.resolve()),
                "source_fingerprint": file_fingerprint(source.path),
                "target_serialized_tokens": source.target_serialized_tokens,
                "realized": {
                    "documents": len(candidates),
                    "content_tokens": content_tokens,
                    "boundary_tokens": boundary_tokens,
                    "serialized_tokens": serialized_tokens,
                    "target_overshoot_tokens": (
                        serialized_tokens - source.target_serialized_tokens
                    ),
                    "shards": writer.shards,
                },
                "scan": scan_by_source[output_name],
                "selected_documents": [
                    _selected_document_meta(item) for item in candidates
                ],
            }
            total_documents += len(candidates)
            total_content_tokens += content_tokens
            total_boundary_tokens += boundary_tokens
            total_serialized_tokens += serialized_tokens

        combined_writer.finish()
        if combined_writer.tokens != total_serialized_tokens:
            raise AssertionError("combined reference shard accounting mismatch")

        selected_hashes = frozenset(all_hash_sources)
        exclusion_hashes = (
            exclusion_hashes_override
            if exclusion_hashes_override is not None
            else selected_hashes
        )
        if not selected_hashes.issubset(exclusion_hashes):
            raise AssertionError("final reference documents are missing from exclusion set")
        hashes = sorted(exclusion_hashes)
        exclusion_hash_sources: dict[str, list[str]] = {}
        if reserved_hashes_by_source is not None:
            for value in hashes:
                exclusion_hash_sources[value] = sorted(
                    output_name
                    for output_name, values in reserved_hashes_by_source.items()
                    if value in values
                )
        else:
            exclusion_hash_sources = {
                value: sorted(all_hash_sources[value]) for value in hashes
            }
        exclusion_manifest = {
            "schema_version": 1,
            "kind": EXCLUSION_MANIFEST_KIND,
            "hash_algorithm": CLEANED_TEXT_HASH_ALGORITHM,
            "membership_basis": "cleaned document text encoded as UTF-8",
            "cleaning": active_cleaning,
            "hash_count": len(hashes),
            "hashes": hashes,
            "hash_sources": exclusion_hash_sources,
            "reference_manifest": MANIFEST_NAME,
            "exclusion_scope": (
                "entire_pre_tokenizer_reserved_pool"
                if reserved_hashes_by_source is not None
                else "final_reference_documents_only"
            ),
        }
        _write_json_atomic(staging / EXCLUSION_MANIFEST_NAME, exclusion_manifest)

        manifest = {
            "schema_version": 2,
            "status": "complete",
            "kind": "petitgpt_cross_stage_reference_validation",
            "immutable": True,
            "tokenizer_path": tokenizer_path,
            "tokenizer_sha256": hashlib.sha256(
                Path(tokenizer_path).read_bytes()
            ).hexdigest(),
            "vocab_size": vocab_size,
            "dtype": "uint32" if dtype == np.uint32 else "uint16",
            "contract": contract,
            "selection": {
                "algorithm": SELECTION_ALGORITHM,
                "seed": int(seed),
                "membership_basis": "cleaned document text SHA-256",
                "input_order_independent": True,
                "whole_documents_only": True,
                "quota_unit": "serialized tokens (content + BOS + EOS)",
                "restricted_to_pre_tokenizer_reserve": (
                    reserved_hashes_by_source is not None
                ),
            },
            "reserve_provenance": reserve_provenance,
            "cleaning": active_cleaning,
            "packing": {
                "document_form": "[BOS] content [EOS]",
                "textual_document_separator": None,
                "shard_tokens": int(shard_tokens),
            },
            "accounting": {
                "documents": total_documents,
                "content_tokens": total_content_tokens,
                "boundary_tokens": total_boundary_tokens,
                "serialized_tokens": total_serialized_tokens,
                "emitted_shard_tokens": combined_writer.tokens,
                "combined_shards": combined_writer.shards,
            },
            "outputs": {
                "combined_val": "val",
                "val_by_source": "val_by_source",
                "exclusion_hash_manifest": EXCLUSION_MANIFEST_NAME,
            },
            "sources": per_source_output,
            "shard_files": {
                "hash_algorithm": "sha256",
                "val": combined_writer.files,
                "val_by_source": per_source_shard_files,
            },
            "unique_selected_hashes": len(selected_hashes),
            "unique_exclusion_hashes": len(hashes),
            "cross_source_duplicate_selected_documents": (
                total_documents - len(selected_hashes)
            ),
        }
        _write_json_atomic(staging / MANIFEST_NAME, manifest)
        os.replace(staging, out_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def finalize_reference_validation(
    *,
    reserve_manifest_path: Path,
    out_dir: Path,
    tokenizer_path: str,
    shard_tokens: int,
) -> dict[str, Any]:
    """Finalize exact token quotas strictly inside a pre-tokenizer reserve."""
    sources, reserved_by_source, exclusion_hashes, provenance = (
        load_reference_reserve(reserve_manifest_path)
    )
    selection = provenance["selection"]
    cleaning = provenance["cleaning"]
    return build_reference_validation(
        sources=sources,
        out_dir=out_dir,
        tokenizer_path=tokenizer_path,
        shard_tokens=shard_tokens,
        seed=int(selection["seed"]),
        strip_leading_noise=bool(cleaning["strip_leading_noise"]),
        normalize_quotes=bool(cleaning["normalize_quotes"]),
        underscores_policy=str(cleaning["underscores_policy"]),
        min_chars=int(cleaning["min_chars"]),
        min_ascii_ratio=float(cleaning["min_ascii_ratio"]),
        reserved_hashes_by_source=reserved_by_source,
        exclusion_hashes_override=exclusion_hashes,
        reserve_provenance=provenance,
    )


def _add_cleaning_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--strip_leading_noise", action="store_true")
    parser.add_argument("--normalize_quotes", action="store_true")
    parser.add_argument(
        "--underscores_policy",
        choices=["keep", "space", "remove"],
        default="keep",
    )
    parser.add_argument("--min_chars", type=int, default=0)
    parser.add_argument("--min_ascii_ratio", type=float, default=0.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Two-phase immutable reference validation: reserve before tokenizer "
            "training, then finalize exact token quotas with the frozen tokenizer."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    reserve_parser = subparsers.add_parser(
        "reserve", help="Tokenizer-independent phase; run before tokenizer training."
    )
    reserve_parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="PATH:TARGET_SERIALIZED_TOKENS",
        help="Repeat once per source/domain.",
    )
    reserve_parser.add_argument("--out_dir", type=Path, required=True)
    reserve_parser.add_argument("--seed", type=int, default=20250814)
    reserve_parser.add_argument(
        "--reserve_bytes_per_target_token",
        type=float,
        default=32.0,
        help=(
            "Oversized pre-tokenizer reserve budget. If finalization is short, "
            "increase this and repeat reserve before training the tokenizer."
        ),
    )
    _add_cleaning_args(reserve_parser)

    finalize_parser = subparsers.add_parser(
        "finalize", help="Token-count and shard the frozen reserve after tokenizer training."
    )
    finalize_parser.add_argument("--reserve_manifest", type=Path, required=True)
    finalize_parser.add_argument("--out_dir", type=Path, required=True)
    finalize_parser.add_argument("--tokenizer_path", required=True)
    finalize_parser.add_argument("--shard_tokens", type=int, default=2_000_000)
    args = parser.parse_args()

    try:
        if args.command == "reserve":
            sources = parse_reference_sources(args.source)
            manifest = reserve_reference_candidates(
                sources=sources,
                out_dir=args.out_dir,
                seed=args.seed,
                reserve_bytes_per_target_token=args.reserve_bytes_per_target_token,
                strip_leading_noise=args.strip_leading_noise,
                normalize_quotes=args.normalize_quotes,
                underscores_policy=args.underscores_policy,
                min_chars=args.min_chars,
                min_ascii_ratio=args.min_ascii_ratio,
            )
        else:
            manifest = finalize_reference_validation(
                reserve_manifest_path=args.reserve_manifest,
                out_dir=args.out_dir,
                tokenizer_path=args.tokenizer_path,
                shard_tokens=args.shard_tokens,
            )
    except (OSError, ValueError, ReferenceSourceExhaustedError) as exc:
        raise SystemExit(str(exc)) from exc

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
