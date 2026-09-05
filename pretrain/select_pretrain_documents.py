#!/usr/bin/env python3

"""Deterministically select disjoint pretraining documents by token quota.

This is the scalable, upstream selection layer for production corpus builds.
It scans JSONL sources in a fixed priority order (Stage B, then Stage A, then
control), retains the smallest seeded-hash prefix that reaches each source's
whole-document serialized-token quota, and writes the selected JSONL streams.

Selection state lives in SQLite rather than Python sets.  Memory is therefore
bounded by one input document, one tokenizer encoding, and SQLite's configured
page cache.  Exact raw/cleaned/canonical identities are globally unique across
the committed selection, so a lower-priority source can never reuse a document
already selected by a higher-priority source.

Near-duplicate removal is intentionally not attempted here.  Input candidate
pools must already have passed the project's upstream near-deduplication step.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import tempfile
from typing import Any
import uuid

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pretrain.build_pretrain_shards import (  # noqa: E402
    CLEANED_TEXT_HASH_ALGORITHM,
    EXCLUSION_MANIFEST_KIND,
    _file_sha256,
    _tokenizer_vocab_size,
    _write_json_atomic,
    ascii_ratio,
    assert_token_ids_ok,
    clean_text,
    cleaning_contract,
    encode_with_accounting,
    load_exclusion_hash_manifest,
    load_tokenizer,
)
from src.special_tokens import (  # noqa: E402
    BOS_ID,
    CANONICAL_VOCAB_SIZE,
    EOS_ID,
    SPECIAL_TOKEN_IDS,
    assert_tokenizer_contract,
)

SPEC_SCHEMA_VERSION = 1
REGISTRY_SCHEMA_VERSION = 2
MANIFEST_KIND = "petitgpt_pretrain_document_selection"
DATABASE_NAME = "selection_registry.sqlite"
MANIFEST_NAME = "manifest.json"
AUDIT_NAME = "audit_exact_intersections.json"
SELECTION_METADATA_FIELD = "_petitgpt_selection"
SELECTION_ALGORITHM = "sha256-seed-stage-source-canonical-v1"
RAW_HASH_ALGORITHM = "sha256-raw-text-utf8-v1"
CLEANED_HASH_ALGORITHM = CLEANED_TEXT_HASH_ALGORITHM
CANONICAL_FINGERPRINT_ALGORITHM = "sha256-domain-separated-cleaned-text-v1"
STAGE_PRIORITY = {"stage_b": 0, "stage_a": 1, "control": 2}
STAGE_ORDER = tuple(STAGE_PRIORITY)
_SOURCE_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
_SPECIAL_IDS = frozenset(SPECIAL_TOKEN_IDS.values())


class SourceExhaustedError(RuntimeError):
    """A source cannot meet its quota after exact-hash exclusions."""


class HashCollisionError(RuntimeError):
    """Two non-identical documents produced an identity-hash collision."""


class SourceMutationError(RuntimeError):
    """An input source changed while it was being scanned."""


@dataclass(frozen=True)
class CleaningSpec:
    strip_leading_noise: bool = False
    normalize_quotes: bool = False
    underscores_policy: str = "keep"
    min_chars: int = 0
    min_ascii_ratio: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "strip_leading_noise": self.strip_leading_noise,
            "normalize_quotes": self.normalize_quotes,
            "underscores_policy": self.underscores_policy,
            "min_chars": self.min_chars,
            "min_ascii_ratio": self.min_ascii_ratio,
        }


@dataclass(frozen=True)
class SourceSpec:
    stage: str
    source_id: str
    path: Path
    target_serialized_tokens: int

    @property
    def priority_key(self) -> tuple[int, str]:
        return STAGE_PRIORITY[self.stage], self.source_id


@dataclass(frozen=True)
class SelectionSpec:
    seed: int
    text_field: str
    cleaning: CleaningSpec
    sources: tuple[SourceSpec, ...]


@dataclass(frozen=True)
class DocumentIdentity:
    raw_sha256: str
    cleaned_sha256: str
    canonical_fingerprint: str
    raw_utf8_bytes: int
    cleaned_utf8_bytes: int


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"record cannot be serialized as strict canonical JSON: {exc}") from exc
    return serialized.encode("ascii")


def canonical_document_fingerprint(cleaned_text: str) -> str:
    """Return a domain-separated canonical document fingerprint."""
    cleaned_bytes = cleaned_text.encode("utf-8", errors="strict")
    digest = hashlib.sha256()
    digest.update(b"PetitGPT-canonical-document-v1\0")
    digest.update(len(cleaned_bytes).to_bytes(8, "big", signed=False))
    digest.update(cleaned_bytes)
    return digest.hexdigest()


def document_identity(raw_text: str, cleaned_text: str) -> DocumentIdentity:
    raw_bytes = raw_text.encode("utf-8", errors="strict")
    cleaned_bytes = cleaned_text.encode("utf-8", errors="strict")
    return DocumentIdentity(
        raw_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        cleaned_sha256=hashlib.sha256(cleaned_bytes).hexdigest(),
        canonical_fingerprint=canonical_document_fingerprint(cleaned_text),
        raw_utf8_bytes=len(raw_bytes),
        cleaned_utf8_bytes=len(cleaned_bytes),
    )


def selection_rank(
    *,
    seed: int,
    stage: str,
    source_id: str,
    canonical_fingerprint: str,
) -> str:
    """Return the fixed-width rank used for order-independent selection."""
    digest = hashlib.sha256()
    digest.update(b"PetitGPT-pretrain-selection-v1\0")
    digest.update(str(seed).encode("ascii"))
    digest.update(b"\0")
    digest.update(stage.encode("ascii"))
    digest.update(b"\0")
    digest.update(source_id.encode("ascii"))
    digest.update(b"\0")
    digest.update(bytes.fromhex(canonical_fingerprint))
    return digest.hexdigest()


def _strict_object(path: Path) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            value = json.load(
                f,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant {value!r}")
                ),
            )
    except (OSError, json.JSONDecodeError, UnicodeError, ValueError) as exc:
        raise ValueError(f"cannot read selection spec {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("selection spec must be a JSON object")
    return value


def _check_keys(value: dict[str, Any], allowed: set[str], *, where: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown {where} keys: {unknown}")


def load_selection_spec(path: Path) -> SelectionSpec:
    """Load and strictly validate a v1 selection specification."""
    raw = _strict_object(path)
    _check_keys(
        raw,
        {"schema_version", "seed", "text_field", "cleaning", "sources"},
        where="top-level selection spec",
    )
    schema_version = raw.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != SPEC_SCHEMA_VERSION
    ):
        raise ValueError(
            f"selection spec schema_version must be {SPEC_SCHEMA_VERSION}, got {schema_version!r}"
        )

    seed = raw.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or not (0 <= seed < 2**63):
        raise ValueError("selection spec seed must be an integer in [0, 2**63)")
    text_field = raw.get("text_field", "text")
    if not isinstance(text_field, str) or not text_field:
        raise ValueError("selection spec text_field must be a non-empty string")
    if text_field == SELECTION_METADATA_FIELD:
        raise ValueError(f"text_field cannot be reserved field {SELECTION_METADATA_FIELD!r}")

    raw_cleaning = raw.get("cleaning", {})
    if not isinstance(raw_cleaning, dict):
        raise ValueError("selection spec cleaning must be an object")
    _check_keys(
        raw_cleaning,
        {
            "strip_leading_noise",
            "normalize_quotes",
            "underscores_policy",
            "min_chars",
            "min_ascii_ratio",
        },
        where="cleaning",
    )
    strip_leading_noise = raw_cleaning.get("strip_leading_noise", False)
    normalize_quotes = raw_cleaning.get("normalize_quotes", False)
    if not isinstance(strip_leading_noise, bool):
        raise ValueError("cleaning.strip_leading_noise must be boolean")
    if not isinstance(normalize_quotes, bool):
        raise ValueError("cleaning.normalize_quotes must be boolean")
    underscores_policy = raw_cleaning.get("underscores_policy", "keep")
    if underscores_policy not in {"keep", "space", "remove"}:
        raise ValueError("cleaning.underscores_policy must be keep, space, or remove")
    min_chars = raw_cleaning.get("min_chars", 0)
    if not isinstance(min_chars, int) or isinstance(min_chars, bool) or min_chars < 0:
        raise ValueError("cleaning.min_chars must be a non-negative integer")
    min_ascii_ratio = raw_cleaning.get("min_ascii_ratio", 0.0)
    if (
        not isinstance(min_ascii_ratio, (int, float))
        or isinstance(min_ascii_ratio, bool)
        or not (0.0 <= float(min_ascii_ratio) <= 1.0)
    ):
        raise ValueError("cleaning.min_ascii_ratio must be in [0, 1]")
    cleaning = CleaningSpec(
        strip_leading_noise=strip_leading_noise,
        normalize_quotes=normalize_quotes,
        underscores_policy=underscores_policy,
        min_chars=min_chars,
        min_ascii_ratio=float(min_ascii_ratio),
    )

    raw_sources = raw.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("selection spec sources must be a non-empty list")
    base_dir = path.resolve().parent
    sources: list[SourceSpec] = []
    source_ids: set[str] = set()
    resolved_paths: set[str] = set()
    for index, raw_source in enumerate(raw_sources):
        if not isinstance(raw_source, dict):
            raise ValueError(f"sources[{index}] must be an object")
        _check_keys(
            raw_source,
            {"stage", "source_id", "path", "target_serialized_tokens"},
            where=f"sources[{index}]",
        )
        stage = raw_source.get("stage")
        if stage not in STAGE_PRIORITY:
            raise ValueError(
                f"sources[{index}].stage must be one of {list(STAGE_ORDER)}, got {stage!r}"
            )
        source_id = raw_source.get("source_id")
        if not isinstance(source_id, str) or _SOURCE_ID_RE.fullmatch(source_id) is None:
            raise ValueError(f"sources[{index}].source_id must match {_SOURCE_ID_RE.pattern!r}")
        if source_id in source_ids:
            raise ValueError(f"duplicate source_id {source_id!r}")
        source_ids.add(source_id)
        raw_path = raw_source.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"sources[{index}].path must be a non-empty string")
        source_path = Path(raw_path)
        if not source_path.is_absolute():
            source_path = base_dir / source_path
        source_path = source_path.resolve()
        resolved = str(source_path)
        if resolved in resolved_paths:
            raise ValueError(f"duplicate resolved source path {resolved!r}")
        resolved_paths.add(resolved)
        if not source_path.is_file():
            raise FileNotFoundError(f"source does not exist or is not a file: {source_path}")
        target = raw_source.get("target_serialized_tokens")
        if not isinstance(target, int) or isinstance(target, bool) or target <= 0:
            raise ValueError(
                f"sources[{index}].target_serialized_tokens must be a positive integer"
            )
        sources.append(
            SourceSpec(
                stage=stage,
                source_id=source_id,
                path=source_path,
                target_serialized_tokens=target,
            )
        )

    sources.sort(key=lambda source: source.priority_key)
    return SelectionSpec(
        seed=seed,
        text_field=text_field,
        cleaning=cleaning,
        sources=tuple(sources),
    )


def _stat_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(getattr(stat, "st_ino", 0)),
        int(getattr(stat, "st_dev", 0)),
    )


def _file_snapshot(path: Path) -> tuple[tuple[int, int, int, int], str]:
    before = _stat_identity(path)
    sha256 = _file_sha256(path)
    after = _stat_identity(path)
    if before != after:
        raise SourceMutationError(f"file changed while hashing: {path}")
    return after, sha256


def _assert_file_snapshot(
    path: Path,
    snapshot: tuple[tuple[int, int, int, int], str],
) -> None:
    expected_stat, expected_sha256 = snapshot
    actual_stat, actual_sha256 = _file_snapshot(path)
    if actual_stat != expected_stat or actual_sha256 != expected_sha256:
        raise SourceMutationError(f"audited input changed during selection: {path}")


def _open_registry(path: Path, *, sqlite_cache_mb: int) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(f"PRAGMA cache_size=-{sqlite_cache_mb * 1024}")
    connection.executescript(
        """
        CREATE TABLE sources (
            source_id TEXT PRIMARY KEY,
            stage TEXT NOT NULL,
            stage_priority INTEGER NOT NULL,
            input_path TEXT NOT NULL UNIQUE,
            target_serialized_tokens INTEGER NOT NULL,
            input_sha256 TEXT,
            input_size_bytes INTEGER,
            stats_json TEXT,
            selected_documents INTEGER,
            selected_content_tokens INTEGER,
            selected_boundary_tokens INTEGER,
            selected_serialized_tokens INTEGER,
            output_relative_path TEXT,
            output_sha256 TEXT,
            output_size_bytes INTEGER
        );

        CREATE TABLE documents (
            canonical_fingerprint TEXT PRIMARY KEY,
            cleaned_sha256 TEXT NOT NULL UNIQUE,
            raw_sha256 TEXT NOT NULL UNIQUE,
            input_record_sha256 TEXT NOT NULL,
            source_id TEXT NOT NULL REFERENCES sources(source_id),
            stage TEXT NOT NULL,
            committed INTEGER NOT NULL CHECK (committed IN (0, 1)),
            selection_rank TEXT NOT NULL,
            raw_utf8_bytes INTEGER NOT NULL,
            cleaned_utf8_bytes INTEGER NOT NULL,
            content_tokens INTEGER NOT NULL,
            boundary_tokens INTEGER NOT NULL,
            serialized_tokens INTEGER NOT NULL,
            payload_json TEXT NOT NULL
        );

        CREATE INDEX documents_candidate_rank
            ON documents(committed, source_id, selection_rank DESC,
                         canonical_fingerprint DESC);
        CREATE INDEX documents_source_output
            ON documents(source_id, committed, selection_rank,
                         canonical_fingerprint);

        CREATE TABLE exclusions (
            source_id TEXT NOT NULL REFERENCES sources(source_id),
            owner_source_id TEXT NOT NULL REFERENCES sources(source_id),
            documents INTEGER NOT NULL,
            PRIMARY KEY (source_id, owner_source_id)
        );

        CREATE TABLE reference_exclusion_manifests (
            manifest_id INTEGER PRIMARY KEY,
            input_path TEXT NOT NULL UNIQUE,
            input_sha256 TEXT NOT NULL,
            input_size_bytes INTEGER NOT NULL,
            kind TEXT NOT NULL,
            hash_algorithm TEXT NOT NULL,
            hash_count INTEGER NOT NULL
        );

        CREATE TABLE reference_exclusion_hashes (
            cleaned_sha256 TEXT PRIMARY KEY
        );

        CREATE TABLE reference_exclusion_memberships (
            cleaned_sha256 TEXT NOT NULL
                REFERENCES reference_exclusion_hashes(cleaned_sha256),
            manifest_id INTEGER NOT NULL
                REFERENCES reference_exclusion_manifests(manifest_id),
            PRIMARY KEY (cleaned_sha256, manifest_id)
        );

        CREATE TABLE reference_exclusion_matches (
            manifest_id INTEGER NOT NULL
                REFERENCES reference_exclusion_manifests(manifest_id),
            source_id TEXT NOT NULL REFERENCES sources(source_id),
            documents INTEGER NOT NULL,
            PRIMARY KEY (manifest_id, source_id)
        );

        CREATE VIEW selections AS
            SELECT * FROM documents WHERE committed = 1;
        CREATE VIEW candidates AS
            SELECT * FROM documents WHERE committed = 0;
        """
    )
    return connection


def _initialize_reference_exclusions(
    connection: sqlite3.Connection,
    *,
    manifest_paths: tuple[Path, ...],
    cleaning: CleaningSpec,
) -> int:
    expected_cleaning = cleaning_contract(**cleaning.as_dict())
    for manifest_id, path in enumerate(manifest_paths, start=1):
        hashes, metadata = load_exclusion_hash_manifest(
            path,
            expected_cleaning=expected_cleaning,
        )
        connection.execute(
            """
            INSERT INTO reference_exclusion_manifests(
                manifest_id, input_path, input_sha256, input_size_bytes,
                kind, hash_algorithm, hash_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                manifest_id,
                str(path),
                metadata["manifest_sha256"],
                metadata["manifest_size_bytes"],
                metadata["kind"],
                metadata["hash_algorithm"],
                metadata["hash_count"],
            ),
        )
        for cleaned_hash in sorted(hashes):
            connection.execute(
                "INSERT OR IGNORE INTO reference_exclusion_hashes(cleaned_sha256) VALUES (?)",
                (cleaned_hash,),
            )
            connection.execute(
                """
                INSERT INTO reference_exclusion_memberships(cleaned_sha256, manifest_id)
                VALUES (?, ?)
                """,
                (cleaned_hash, manifest_id),
            )
        connection.commit()
    union_count = int(
        connection.execute("SELECT COUNT(*) FROM reference_exclusion_hashes").fetchone()[0]
    )
    if union_count == 0:
        raise ValueError("reference exclusion manifests have an empty hash union")
    return union_count


def _identity_row(
    connection: sqlite3.Connection,
    identity: DocumentIdentity,
) -> sqlite3.Row | None:
    """Find an identity and fail if any exact-hash collision is observable."""
    by_canonical = connection.execute(
        "SELECT * FROM documents WHERE canonical_fingerprint = ?",
        (identity.canonical_fingerprint,),
    ).fetchone()
    if by_canonical is not None and by_canonical["cleaned_sha256"] != identity.cleaned_sha256:
        raise HashCollisionError(
            "canonical fingerprint collision: identical fingerprint maps to distinct "
            "cleaned SHA-256 values"
        )

    by_cleaned = connection.execute(
        "SELECT * FROM documents WHERE cleaned_sha256 = ?",
        (identity.cleaned_sha256,),
    ).fetchone()
    if (
        by_cleaned is not None
        and by_cleaned["canonical_fingerprint"] != identity.canonical_fingerprint
    ):
        raise HashCollisionError(
            "cleaned SHA-256 collision: identical digest maps to distinct canonical fingerprints"
        )

    by_raw = connection.execute(
        "SELECT * FROM documents WHERE raw_sha256 = ?",
        (identity.raw_sha256,),
    ).fetchone()
    if by_raw is not None and (
        by_raw["canonical_fingerprint"] != identity.canonical_fingerprint
        or by_raw["cleaned_sha256"] != identity.cleaned_sha256
    ):
        raise HashCollisionError(
            "raw SHA-256 collision: identical raw digest maps to a distinct document"
        )
    return by_canonical or by_cleaned or by_raw


def _empty_scan_stats() -> dict[str, int]:
    return {
        "lines": 0,
        "blank_lines": 0,
        "parsed_documents": 0,
        "dropped_empty_after_cleaning": 0,
        "dropped_short": 0,
        "dropped_ascii": 0,
        "eligible_documents": 0,
        "excluded_reference_validation": 0,
        "excluded_by_prior_selection": 0,
        "duplicate_candidates": 0,
        "representative_updates": 0,
        "skipped_by_rank": 0,
        "candidate_inserts": 0,
        "candidate_evictions": 0,
    }


def _clean_for_selection(raw_text: str, cleaning: CleaningSpec) -> tuple[str | None, str]:
    cleaned = clean_text(
        raw_text,
        strip_leading_noise=cleaning.strip_leading_noise,
        normalize_quotes=cleaning.normalize_quotes,
        underscores_policy=cleaning.underscores_policy,
        min_chars=0,
        min_ascii_ratio=0.0,
    )
    if cleaned is None:
        return None, "empty"
    if cleaning.min_chars > 0 and len(cleaned) < cleaning.min_chars:
        return None, "short"
    if cleaning.min_ascii_ratio > 0.0 and ascii_ratio(cleaned) < cleaning.min_ascii_ratio:
        return None, "ascii"
    return cleaned, "eligible"


def _strict_jsonl_record(
    raw_line: bytes, *, source: SourceSpec, line_number: int
) -> dict[str, Any]:
    try:
        line = raw_line.decode("utf-8", errors="strict")
        record = json.loads(
            line,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value!r}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid strict JSONL at {source.path}:{line_number}: {exc}") from exc
    if not isinstance(record, dict):
        raise ValueError(f"JSONL record at {source.path}:{line_number} is not an object")
    return record


def _output_payload(
    record: dict[str, Any],
    *,
    text_field: str,
    cleaned_text: str,
    source: SourceSpec,
    identity: DocumentIdentity,
    record_sha256: str,
    rank: str,
    content_tokens: int,
    boundary_tokens: int,
) -> str:
    if SELECTION_METADATA_FIELD in record:
        raise ValueError(
            f"input record contains reserved provenance field {SELECTION_METADATA_FIELD!r}"
        )
    output = dict(record)
    output[text_field] = cleaned_text
    output[SELECTION_METADATA_FIELD] = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "stage": source.stage,
        "source_id": source.source_id,
        "raw_sha256": identity.raw_sha256,
        "cleaned_sha256": identity.cleaned_sha256,
        "canonical_fingerprint": identity.canonical_fingerprint,
        "input_record_sha256": record_sha256,
        "selection_rank": rank,
        "raw_utf8_bytes": identity.raw_utf8_bytes,
        "cleaned_utf8_bytes": identity.cleaned_utf8_bytes,
        "content_tokens": content_tokens,
        "boundary_tokens": boundary_tokens,
        "serialized_tokens": content_tokens + boundary_tokens,
    }
    return _canonical_json_bytes(output).decode("ascii")


def _worst_candidate(
    connection: sqlite3.Connection,
    source_id: str,
) -> sqlite3.Row | None:
    return connection.execute(
        """
        SELECT canonical_fingerprint, selection_rank, serialized_tokens
        FROM documents
        WHERE committed = 0 AND source_id = ?
        ORDER BY selection_rank DESC, canonical_fingerprint DESC
        LIMIT 1
        """,
        (source_id,),
    ).fetchone()


def _candidate_totals(
    connection: sqlite3.Connection,
    source_id: str,
) -> tuple[int, int]:
    row = connection.execute(
        """
        SELECT COUNT(*) AS documents, COALESCE(SUM(serialized_tokens), 0) AS tokens
        FROM documents WHERE committed = 0 AND source_id = ?
        """,
        (source_id,),
    ).fetchone()
    assert row is not None
    return int(row["documents"]), int(row["tokens"])


def _record_exclusion(
    connection: sqlite3.Connection,
    *,
    source_id: str,
    owner_source_id: str,
) -> None:
    connection.execute(
        """
        INSERT INTO exclusions(source_id, owner_source_id, documents)
        VALUES (?, ?, 1)
        ON CONFLICT(source_id, owner_source_id)
        DO UPDATE SET documents = documents + 1
        """,
        (source_id, owner_source_id),
    )


def _record_reference_exclusion(
    connection: sqlite3.Connection,
    *,
    source_id: str,
    manifest_ids: list[int],
) -> None:
    for manifest_id in manifest_ids:
        connection.execute(
            """
            INSERT INTO reference_exclusion_matches(manifest_id, source_id, documents)
            VALUES (?, ?, 1)
            ON CONFLICT(manifest_id, source_id)
            DO UPDATE SET documents = documents + 1
            """,
            (manifest_id, source_id),
        )


def _select_source(
    connection: sqlite3.Connection,
    source: SourceSpec,
    *,
    spec: SelectionSpec,
    tokenizer: Any,
    vocab_size: int,
    commit_every: int,
) -> dict[str, Any]:
    dangling = connection.execute("SELECT COUNT(*) FROM documents WHERE committed = 0").fetchone()[
        0
    ]
    if dangling:
        raise AssertionError("registry contains candidates from a previous source")

    stats = _empty_scan_stats()
    candidate_documents = 0
    candidate_tokens = 0
    source_digest = hashlib.sha256()
    before_stat = _stat_identity(source.path)

    with open(source.path, "rb") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            source_digest.update(raw_line)
            stats["lines"] += 1
            if not raw_line.strip():
                stats["blank_lines"] += 1
                continue
            record = _strict_jsonl_record(
                raw_line,
                source=source,
                line_number=line_number,
            )
            stats["parsed_documents"] += 1
            if SELECTION_METADATA_FIELD in record:
                raise ValueError(
                    f"reserved field collision at {source.path}:{line_number}: "
                    f"{SELECTION_METADATA_FIELD!r}"
                )
            raw_text = record.get(spec.text_field)
            if not isinstance(raw_text, str):
                raise ValueError(
                    f"record at {source.path}:{line_number} has no string {spec.text_field!r} field"
                )
            cleaned_text, disposition = _clean_for_selection(raw_text, spec.cleaning)
            if disposition == "empty":
                stats["dropped_empty_after_cleaning"] += 1
                continue
            if disposition == "short":
                stats["dropped_short"] += 1
                continue
            if disposition == "ascii":
                stats["dropped_ascii"] += 1
                continue
            assert cleaned_text is not None
            stats["eligible_documents"] += 1

            try:
                identity = document_identity(raw_text, cleaned_text)
            except UnicodeError as exc:
                raise ValueError(
                    f"invalid Unicode text at {source.path}:{line_number}: {exc}"
                ) from exc
            reference_manifest_ids = [
                int(row["manifest_id"])
                for row in connection.execute(
                    """
                    SELECT manifest_id FROM reference_exclusion_memberships
                    WHERE cleaned_sha256 = ? ORDER BY manifest_id
                    """,
                    (identity.cleaned_sha256,),
                )
            ]
            if reference_manifest_ids:
                leaked_owner = connection.execute(
                    "SELECT source_id FROM documents WHERE cleaned_sha256 = ?",
                    (identity.cleaned_sha256,),
                ).fetchone()
                if leaked_owner is not None:
                    raise AssertionError(
                        "reference validation hash was selected before exclusion; "
                        f"owner={leaked_owner['source_id']!r}"
                    )
                stats["excluded_reference_validation"] += 1
                _record_reference_exclusion(
                    connection,
                    source_id=source.source_id,
                    manifest_ids=reference_manifest_ids,
                )
                if stats["lines"] % commit_every == 0:
                    connection.commit()
                continue

            record_bytes = _canonical_json_bytes(record)
            record_sha256 = hashlib.sha256(record_bytes).hexdigest()
            existing = _identity_row(connection, identity)
            rank = selection_rank(
                seed=spec.seed,
                stage=source.stage,
                source_id=source.source_id,
                canonical_fingerprint=identity.canonical_fingerprint,
            )

            if existing is not None and int(existing["committed"]) == 1:
                stats["excluded_by_prior_selection"] += 1
                _record_exclusion(
                    connection,
                    source_id=source.source_id,
                    owner_source_id=str(existing["source_id"]),
                )
                if stats["lines"] % commit_every == 0:
                    connection.commit()
                continue

            if existing is not None:
                if existing["source_id"] != source.source_id:
                    raise AssertionError("uncommitted candidate belongs to another source")
                stats["duplicate_candidates"] += 1
                old_representative = (
                    str(existing["raw_sha256"]),
                    str(existing["input_record_sha256"]),
                )
                new_representative = (identity.raw_sha256, record_sha256)
                if new_representative < old_representative:
                    payload = _output_payload(
                        record,
                        text_field=spec.text_field,
                        cleaned_text=cleaned_text,
                        source=source,
                        identity=identity,
                        record_sha256=record_sha256,
                        rank=rank,
                        content_tokens=int(existing["content_tokens"]),
                        boundary_tokens=int(existing["boundary_tokens"]),
                    )
                    try:
                        connection.execute(
                            """
                            UPDATE documents
                            SET raw_sha256 = ?, input_record_sha256 = ?,
                                raw_utf8_bytes = ?, payload_json = ?
                            WHERE canonical_fingerprint = ?
                            """,
                            (
                                identity.raw_sha256,
                                record_sha256,
                                identity.raw_utf8_bytes,
                                payload,
                                identity.canonical_fingerprint,
                            ),
                        )
                    except sqlite3.IntegrityError as exc:
                        raise HashCollisionError(
                            f"identity collision while updating duplicate representative: {exc}"
                        ) from exc
                    stats["representative_updates"] += 1
                if stats["lines"] % commit_every == 0:
                    connection.commit()
                continue

            if candidate_tokens >= source.target_serialized_tokens:
                worst = _worst_candidate(connection, source.source_id)
                if worst is None:
                    raise AssertionError("quota reached without a retained candidate")
                if (rank, identity.canonical_fingerprint) > (
                    str(worst["selection_rank"]),
                    str(worst["canonical_fingerprint"]),
                ):
                    stats["skipped_by_rank"] += 1
                    if stats["lines"] % commit_every == 0:
                        connection.commit()
                    continue

            ids, content_tokens, boundary_tokens = encode_with_accounting(
                tokenizer,
                cleaned_text,
                add_bos=True,
                add_eos=True,
                bos_id=BOS_ID,
                eos_id=EOS_ID,
            )
            assert_token_ids_ok(
                ids,
                vocab_size=vocab_size,
                dtype=np.dtype(np.uint16),
                src=source.source_id,
                text_preview=cleaned_text[:200],
            )
            if not ids or ids[0] != BOS_ID or ids[-1] != EOS_ID or boundary_tokens != 2:
                raise AssertionError("serialized document does not satisfy [BOS] content [EOS]")
            injected = sorted(set(ids[1:-1]) & _SPECIAL_IDS)
            if injected:
                raise AssertionError(
                    "document content encoded to reserved special-token IDs; "
                    f"literal-special injection guard failed for IDs {injected}"
                )
            serialized_tokens = content_tokens + boundary_tokens
            if len(ids) != serialized_tokens:
                raise AssertionError("content/boundary/serialized token accounting mismatch")

            payload = _output_payload(
                record,
                text_field=spec.text_field,
                cleaned_text=cleaned_text,
                source=source,
                identity=identity,
                record_sha256=record_sha256,
                rank=rank,
                content_tokens=content_tokens,
                boundary_tokens=boundary_tokens,
            )
            try:
                connection.execute(
                    """
                    INSERT INTO documents(
                        canonical_fingerprint, cleaned_sha256, raw_sha256,
                        input_record_sha256, source_id, stage, committed,
                        selection_rank, raw_utf8_bytes, cleaned_utf8_bytes,
                        content_tokens, boundary_tokens, serialized_tokens,
                        payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        identity.canonical_fingerprint,
                        identity.cleaned_sha256,
                        identity.raw_sha256,
                        record_sha256,
                        source.source_id,
                        source.stage,
                        rank,
                        identity.raw_utf8_bytes,
                        identity.cleaned_utf8_bytes,
                        content_tokens,
                        boundary_tokens,
                        serialized_tokens,
                        payload,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise HashCollisionError(
                    f"identity collision while inserting candidate: {exc}"
                ) from exc
            stats["candidate_inserts"] += 1
            candidate_documents += 1
            candidate_tokens += serialized_tokens

            while candidate_documents > 1:
                worst = _worst_candidate(connection, source.source_id)
                if worst is None:
                    raise AssertionError("candidate accounting lost its worst row")
                worst_tokens = int(worst["serialized_tokens"])
                if candidate_tokens - worst_tokens < source.target_serialized_tokens:
                    break
                connection.execute(
                    "DELETE FROM documents WHERE canonical_fingerprint = ? AND committed = 0",
                    (worst["canonical_fingerprint"],),
                )
                candidate_documents -= 1
                candidate_tokens -= worst_tokens
                stats["candidate_evictions"] += 1

            if stats["lines"] % commit_every == 0:
                connection.commit()

    after_stat = _stat_identity(source.path)
    if before_stat != after_stat:
        raise SourceMutationError(f"source changed while being scanned: {source.path}")
    source_sha256 = source_digest.hexdigest()
    independently_verified_sha256 = _file_sha256(source.path)
    verification_stat = _stat_identity(source.path)
    if after_stat != verification_stat:
        raise SourceMutationError(
            f"source changed during independent SHA-256 verification: {source.path}"
        )
    if source_sha256 != independently_verified_sha256:
        raise SourceMutationError(
            "streaming input SHA-256 disagrees with an independent full-file hash; "
            f"source may have changed during scanning: {source.path}"
        )
    connection.commit()
    db_documents, db_tokens = _candidate_totals(connection, source.source_id)
    if db_documents != candidate_documents or db_tokens != candidate_tokens:
        raise AssertionError(
            "candidate accounting disagrees with SQLite: "
            f"memory=({candidate_documents}, {candidate_tokens}), "
            f"database=({db_documents}, {db_tokens})"
        )
    if candidate_tokens < source.target_serialized_tokens:
        raise SourceExhaustedError(
            f"source {source.source_id!r} exhausted at {candidate_tokens} serialized "
            f"tokens, below target {source.target_serialized_tokens}; reference-validation "
            "documents and exact duplicates owned by higher-priority selections are unavailable"
        )

    accounting = connection.execute(
        """
        SELECT COUNT(*) AS documents,
               COALESCE(SUM(content_tokens), 0) AS content_tokens,
               COALESCE(SUM(boundary_tokens), 0) AS boundary_tokens,
               COALESCE(SUM(serialized_tokens), 0) AS serialized_tokens
        FROM documents WHERE committed = 0 AND source_id = ?
        """,
        (source.source_id,),
    ).fetchone()
    assert accounting is not None
    connection.execute(
        "UPDATE documents SET committed = 1 WHERE committed = 0 AND source_id = ?",
        (source.source_id,),
    )
    connection.execute(
        """
        UPDATE sources
        SET input_sha256 = ?, input_size_bytes = ?, stats_json = ?,
            selected_documents = ?, selected_content_tokens = ?,
            selected_boundary_tokens = ?, selected_serialized_tokens = ?
        WHERE source_id = ?
        """,
        (
            source_sha256,
            before_stat[0],
            json.dumps(stats, sort_keys=True, separators=(",", ":")),
            int(accounting["documents"]),
            int(accounting["content_tokens"]),
            int(accounting["boundary_tokens"]),
            int(accounting["serialized_tokens"]),
            source.source_id,
        ),
    )
    connection.commit()
    return {
        "input_sha256": source_sha256,
        "input_size_bytes": before_stat[0],
        "scan": stats,
        "selected": {
            "documents": int(accounting["documents"]),
            "content_tokens": int(accounting["content_tokens"]),
            "boundary_tokens": int(accounting["boundary_tokens"]),
            "serialized_tokens": int(accounting["serialized_tokens"]),
            "serialized_token_overshoot": (
                int(accounting["serialized_tokens"]) - source.target_serialized_tokens
            ),
        },
    }


def _write_selected_jsonl(
    connection: sqlite3.Connection,
    *,
    staging_dir: Path,
    source: SourceSpec,
) -> dict[str, Any]:
    relative = Path("selected") / source.stage / f"{source.source_id}.jsonl"
    destination = staging_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    digest = hashlib.sha256()
    lines = 0
    size_bytes = 0
    cursor = connection.execute(
        """
        SELECT payload_json FROM documents
        WHERE committed = 1 AND source_id = ?
        ORDER BY selection_rank, canonical_fingerprint
        """,
        (source.source_id,),
    )
    with open(temporary, "wb") as output_file:
        while True:
            rows = cursor.fetchmany(512)
            if not rows:
                break
            for row in rows:
                encoded = str(row["payload_json"]).encode("ascii") + b"\n"
                output_file.write(encoded)
                digest.update(encoded)
                lines += 1
                size_bytes += len(encoded)
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temporary, destination)
    output_sha256 = digest.hexdigest()
    connection.execute(
        """
        UPDATE sources
        SET output_relative_path = ?, output_sha256 = ?, output_size_bytes = ?
        WHERE source_id = ?
        """,
        (str(relative), output_sha256, size_bytes, source.source_id),
    )
    return {
        "relative_path": str(relative),
        "sha256": output_sha256,
        "size_bytes": size_bytes,
        "documents": lines,
    }


def _audit_exact_intersections(
    connection: sqlite3.Connection,
    *,
    sources: tuple[SourceSpec, ...],
) -> dict[str, Any]:
    columns = {
        "raw_sha256": RAW_HASH_ALGORITHM,
        "cleaned_sha256": CLEANED_HASH_ALGORITHM,
        "canonical_fingerprint": CANONICAL_FINGERPRINT_ALGORITHM,
    }
    pairs: list[dict[str, Any]] = []
    all_zero = True
    for left_index, left in enumerate(sources):
        for right in sources[left_index + 1 :]:
            counts: dict[str, int] = {}
            for column in columns:
                count = connection.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM selections AS left_docs
                    JOIN selections AS right_docs
                      ON left_docs.{column} = right_docs.{column}
                    WHERE left_docs.source_id = ? AND right_docs.source_id = ?
                    """,  # noqa: S608 - column is selected from a fixed local mapping
                    (left.source_id, right.source_id),
                ).fetchone()[0]
                counts[column] = int(count)
                all_zero = all_zero and count == 0
            pairs.append({
                "left_source_id": left.source_id,
                "left_stage": left.stage,
                "right_source_id": right.source_id,
                "right_stage": right.stage,
                "intersection_counts": counts,
            })
    selected_reference_intersection = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM selections AS selected
            JOIN reference_exclusion_hashes AS reference
              ON selected.cleaned_sha256 = reference.cleaned_sha256
            """
        ).fetchone()[0]
    )
    all_zero = all_zero and selected_reference_intersection == 0
    audit = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "status": "passed" if all_zero else "failed",
        "all_exact_intersections_zero": all_zero,
        "identity_algorithms": columns,
        "pairwise_sources": pairs,
        "reference_validation": {
            "available_identity": "cleaned_sha256",
            "selected_reference_intersection": selected_reference_intersection,
            "intersection_zero": selected_reference_intersection == 0,
            "raw_and_canonical_reference_identities_available": False,
            "note": (
                "the canonical reference exclusion schema carries cleaned SHA-256; "
                "zero cleaned identity intersection proves no exact cleaned document leak"
            ),
        },
    }
    if not all_zero:
        raise AssertionError("exact-hash intersection audit found cross-source leakage")
    return audit


def _reference_exclusion_summary(
    connection: sqlite3.Connection,
    *,
    cleaning: CleaningSpec,
) -> dict[str, Any]:
    manifests: list[dict[str, Any]] = []
    for row in connection.execute(
        """
        SELECT manifest_id, input_path, input_sha256, input_size_bytes,
               kind, hash_algorithm, hash_count
        FROM reference_exclusion_manifests ORDER BY manifest_id
        """
    ):
        manifest_id = int(row["manifest_id"])
        matched_per_source = {
            str(match["source_id"]): int(match["documents"])
            for match in connection.execute(
                """
                SELECT source_id, documents FROM reference_exclusion_matches
                WHERE manifest_id = ? ORDER BY source_id
                """,
                (manifest_id,),
            )
        }
        manifests.append({
            "path": str(row["input_path"]),
            "sha256": str(row["input_sha256"]),
            "size_bytes": int(row["input_size_bytes"]),
            "kind": str(row["kind"]),
            "hash_algorithm": str(row["hash_algorithm"]),
            "hash_count": int(row["hash_count"]),
            "matched_documents": sum(matched_per_source.values()),
            "matched_per_source": matched_per_source,
        })
    union_count = int(
        connection.execute("SELECT COUNT(*) FROM reference_exclusion_hashes").fetchone()[0]
    )
    membership_count = int(
        connection.execute("SELECT COUNT(*) FROM reference_exclusion_memberships").fetchone()[0]
    )
    selected_intersection = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM selections AS selected
            JOIN reference_exclusion_hashes AS reference
              ON selected.cleaned_sha256 = reference.cleaned_sha256
            """
        ).fetchone()[0]
    )
    return {
        "required": True,
        "kind": EXCLUSION_MANIFEST_KIND,
        "hash_algorithm": CLEANED_TEXT_HASH_ALGORITHM,
        "cleaning": cleaning.as_dict(),
        "manifest_count": len(manifests),
        "union_hash_count": union_count,
        "cross_manifest_duplicate_memberships": membership_count - union_count,
        "selected_reference_intersection": selected_intersection,
        "intersection_zero": selected_intersection == 0,
        "manifests": manifests,
    }


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            # Some filesystems do not support directory fsync. All payload
            # files were already individually flushed before publication.
            pass
    finally:
        os.close(descriptor)


def build_selection_registry(
    *,
    spec_path: Path,
    tokenizer_path: Path,
    out_dir: Path,
    exclude_hash_manifests: tuple[Path, ...],
    sqlite_cache_mb: int = 256,
    commit_every: int = 1_000,
) -> dict[str, Any]:
    """Build and atomically publish a deterministic selection registry."""
    if sqlite_cache_mb <= 0:
        raise ValueError("sqlite_cache_mb must be > 0")
    if commit_every <= 0:
        raise ValueError("commit_every must be > 0")
    if not exclude_hash_manifests:
        raise ValueError(
            "at least one --exclude_hash_manifest is required so selected training "
            "documents can be proven disjoint from reference validation"
        )
    spec_path = spec_path.resolve()
    tokenizer_path = tokenizer_path.resolve()
    out_dir = out_dir.resolve()
    exclude_hash_manifests = tuple(path.resolve() for path in exclude_hash_manifests)
    if len(set(exclude_hash_manifests)) != len(exclude_hash_manifests):
        raise ValueError("duplicate --exclude_hash_manifest paths are not allowed")
    for manifest_path in exclude_hash_manifests:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"reference exclusion manifest does not exist: {manifest_path}")
    if os.path.lexists(out_dir):
        raise FileExistsError(f"refusing to overwrite existing output path: {out_dir}")
    if not spec_path.is_file():
        raise FileNotFoundError(f"selection spec does not exist: {spec_path}")
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"tokenizer does not exist: {tokenizer_path}")

    spec_snapshot = _file_snapshot(spec_path)
    tokenizer_snapshot = _file_snapshot(tokenizer_path)
    reference_snapshots = tuple((path, _file_snapshot(path)) for path in exclude_hash_manifests)
    spec = load_selection_spec(spec_path)
    assert_tokenizer_contract(tokenizer_path)
    tokenizer = load_tokenizer(str(tokenizer_path))
    _assert_file_snapshot(spec_path, spec_snapshot)
    _assert_file_snapshot(tokenizer_path, tokenizer_snapshot)
    for path, snapshot in reference_snapshots:
        _assert_file_snapshot(path, snapshot)
    vocab_size = _tokenizer_vocab_size(tokenizer)
    if vocab_size != CANONICAL_VOCAB_SIZE:
        raise ValueError(
            f"tokenizer runtime vocab_size must be {CANONICAL_VOCAB_SIZE}, got {vocab_size}"
        )

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.selecting-", dir=out_dir.parent))
    connection: sqlite3.Connection | None = None
    current_source: str | None = None
    published = False
    try:
        database_path = staging_dir / DATABASE_NAME
        connection = _open_registry(database_path, sqlite_cache_mb=sqlite_cache_mb)
        for source in spec.sources:
            connection.execute(
                """
                INSERT INTO sources(
                    source_id, stage, stage_priority, input_path,
                    target_serialized_tokens
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    source.source_id,
                    source.stage,
                    STAGE_PRIORITY[source.stage],
                    str(source.path),
                    source.target_serialized_tokens,
                ),
            )
        connection.commit()
        reference_union_count = _initialize_reference_exclusions(
            connection,
            manifest_paths=exclude_hash_manifests,
            cleaning=spec.cleaning,
        )

        source_results: dict[str, dict[str, Any]] = {}
        for source in spec.sources:
            current_source = source.source_id
            source_results[source.source_id] = _select_source(
                connection,
                source,
                spec=spec,
                tokenizer=tokenizer,
                vocab_size=vocab_size,
                commit_every=commit_every,
            )

        outputs: dict[str, dict[str, Any]] = {}
        for source in spec.sources:
            outputs[source.source_id] = _write_selected_jsonl(
                connection,
                staging_dir=staging_dir,
                source=source,
            )
        connection.commit()

        audit = _audit_exact_intersections(connection, sources=spec.sources)
        _write_json_atomic(staging_dir / AUDIT_NAME, audit)
        audit_snapshot = _file_snapshot(staging_dir / AUDIT_NAME)
        audit_sha256 = audit_snapshot[1]
        audit_size_bytes = audit_snapshot[0][0]

        sources_manifest: list[dict[str, Any]] = []
        for source in spec.sources:
            exclusions = {
                str(row["owner_source_id"]): int(row["documents"])
                for row in connection.execute(
                    """
                    SELECT owner_source_id, documents FROM exclusions
                    WHERE source_id = ? ORDER BY owner_source_id
                    """,
                    (source.source_id,),
                )
            }
            result = source_results[source.source_id]
            sources_manifest.append({
                "stage": source.stage,
                "source_id": source.source_id,
                "input_path": str(source.path),
                "target_serialized_tokens": source.target_serialized_tokens,
                **result,
                "exact_hash_exclusions_by_owner": exclusions,
                "output": outputs[source.source_id],
            })

        reference_exclusion = _reference_exclusion_summary(
            connection,
            cleaning=spec.cleaning,
        )
        if reference_exclusion["union_hash_count"] != reference_union_count:
            raise AssertionError("reference exclusion union accounting changed during selection")
        if not reference_exclusion["intersection_zero"]:
            raise AssertionError("selected documents intersect reference validation exclusions")
        _assert_file_snapshot(spec_path, spec_snapshot)
        _assert_file_snapshot(tokenizer_path, tokenizer_snapshot)
        for path, snapshot in reference_snapshots:
            _assert_file_snapshot(path, snapshot)

        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"SQLite integrity_check failed: {integrity}")
        database_rows = connection.execute("SELECT COUNT(*) FROM selections").fetchone()[0]
        dangling_candidates = connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        if dangling_candidates:
            raise AssertionError("registry publication attempted with uncommitted candidates")
        connection.commit()
        connection.close()
        connection = None
        database_snapshot = _file_snapshot(database_path)
        database_sha256 = database_snapshot[1]
        database_size_bytes = database_snapshot[0][0]

        manifest = {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "kind": MANIFEST_KIND,
            "status": "complete",
            "priority_order": list(STAGE_ORDER),
            "source_order": [source.source_id for source in spec.sources],
            "selection": {
                "algorithm": SELECTION_ALGORITHM,
                "seed": spec.seed,
                "whole_documents": True,
                "quota_unit": "serialized_tokens",
                "serialized_document_contract": "[BOS] content [EOS]",
                "boundary_tokens_per_document": 2,
                "input_order_independent": True,
            },
            "identity": {
                "raw_sha256": RAW_HASH_ALGORITHM,
                "cleaned_sha256": CLEANED_HASH_ALGORITHM,
                "canonical_fingerprint": CANONICAL_FINGERPRINT_ALGORITHM,
            },
            "cleaning": spec.cleaning.as_dict(),
            "text_field": spec.text_field,
            "near_deduplication": {
                "performed_by_this_tool": False,
                "required_upstream": True,
                "contract": "all candidate pools must be near-deduplicated before selection",
            },
            "reference_validation_exclusion": reference_exclusion,
            "tokenizer": {
                "path": str(tokenizer_path),
                "sha256": tokenizer_snapshot[1],
                "size_bytes": tokenizer_snapshot[0][0],
                "vocab_size": vocab_size,
                "special_token_ids": dict(SPECIAL_TOKEN_IDS),
                "automatic_bos_eos": False,
                "literal_special_tokens_encoded_as_text": True,
            },
            "spec": {
                "path": str(spec_path),
                "sha256": spec_snapshot[1],
                "size_bytes": spec_snapshot[0][0],
                "schema_version": SPEC_SCHEMA_VERSION,
            },
            "bounded_memory": {
                "registry_backend": "sqlite",
                "sqlite_cache_mb": sqlite_cache_mb,
                "commit_every_lines": commit_every,
                "python_identity_set": False,
                "selection_state_scope": "on_disk_plus_one_document",
                "reference_manifest_loading": (
                    "transient memory bounded by each canonical exclusion JSON hash list"
                ),
            },
            "database": {
                "relative_path": DATABASE_NAME,
                "sha256": database_sha256,
                "size_bytes": database_size_bytes,
                "selected_document_rows": int(database_rows),
                "uncommitted_candidate_rows": int(dangling_candidates),
                "integrity_check": integrity,
            },
            "audit": {
                "relative_path": AUDIT_NAME,
                "sha256": audit_sha256,
                "size_bytes": audit_size_bytes,
                "all_exact_intersections_zero": True,
                "selected_reference_intersection": audit["reference_validation"][
                    "selected_reference_intersection"
                ],
            },
            "stage_accounting": _stage_accounting_from_sources(sources_manifest),
            "sources": sources_manifest,
            "publication": {
                "mode": "sibling_staging_then_atomic_rename",
                "requested_out_dir": str(out_dir),
            },
        }
        _write_json_atomic(staging_dir / MANIFEST_NAME, manifest)
        _fsync_directory(staging_dir)
        if os.path.lexists(out_dir):
            raise FileExistsError(f"output path appeared during build: {out_dir}")
        os.rename(staging_dir, out_dir)
        published = True
        _fsync_directory(out_dir.parent)
        return manifest
    except BaseException as exc:
        if connection is not None:
            connection.close()
        failure = {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "kind": MANIFEST_KIND,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "current_source_id": current_source,
            "requested_out_dir": str(out_dir),
            "production_directory_published": published,
        }
        failure_path = out_dir.with_name(f"{out_dir.name}.failed-{uuid.uuid4().hex}.json")
        try:
            _write_json_atomic(failure_path, failure)
        except OSError:
            pass
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def _stage_accounting_from_sources(
    sources: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    result = {
        stage: {
            "sources": 0,
            "documents": 0,
            "content_tokens": 0,
            "boundary_tokens": 0,
            "serialized_tokens": 0,
        }
        for stage in STAGE_ORDER
    }
    for source in sources:
        target = result[str(source["stage"])]
        selected = source["selected"]
        target["sources"] += 1
        for key in ("documents", "content_tokens", "boundary_tokens", "serialized_tokens"):
            target[key] += int(selected[key])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select deterministic, exact-hash-disjoint pretraining JSONL streams "
            "with SQLite-bounded memory."
        )
    )
    parser.add_argument("--spec", type=Path, required=True, help="Selection spec JSON (v1).")
    parser.add_argument(
        "--tokenizer_path",
        type=Path,
        required=True,
        help="Final canonical 32k/seven-special tokenizer.json.",
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--exclude_hash_manifest",
        type=Path,
        action="append",
        required=True,
        help=(
            "Repeatable reference reserve/final exclusion_hash_manifest.json; "
            "at least one non-empty manifest is mandatory."
        ),
    )
    parser.add_argument(
        "--sqlite_cache_mb",
        type=int,
        default=256,
        help="SQLite page-cache ceiling in MiB (default: 256).",
    )
    parser.add_argument(
        "--commit_every",
        type=int,
        default=1_000,
        help="Commit on-disk candidate state every N scanned lines.",
    )
    args = parser.parse_args()
    build_selection_registry(
        spec_path=args.spec,
        tokenizer_path=args.tokenizer_path,
        out_dir=args.out_dir,
        exclude_hash_manifests=tuple(args.exclude_hash_manifest),
        sqlite_cache_mb=args.sqlite_cache_mb,
        commit_every=args.commit_every,
    )


if __name__ == "__main__":
    main()
