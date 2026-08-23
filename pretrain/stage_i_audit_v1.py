#!/usr/bin/env python3
"""Streaming physical-realization audit for Stage I, shared by publication and the consumer.

Codex's independent review of I-tooling-v1 found two related defects: COMPLETE could be published
while the manifest's token totals disagreed with the records actually on disk, and the strict
consumer could fully iterate a published realization without enforcing the physical invariants.
Both had the same root cause -- accounting was trusted from an in-memory object rather than
derived from the bytes.

This module is the single answer to both. It derives every number from the staged or published
records themselves and returns a deterministic summary. The publisher runs it before COMPLETE
exists; the consumer runs the same function after. Because there is one implementation, the two
interpretations cannot drift.

Two design constraints shape everything here:

* **Bounded memory.** A production realization is ~13.8M records and ~56 GB. Nothing in this
  module may scale with the size of a shard, a node or the realization. Records stream one line at
  a time; anything needing a global view (identity uniqueness, per-node ordinal continuity,
  per-node fingerprints) goes through a deterministic disk-backed external sort with a bounded
  in-memory chunk.
* **Derive, never trust.** Expected *semantic* values -- what the selection should have been --
  come from the bound plan and the accepted Stage-H prediction, never from the records. What this
  module derives is only what is *physically there*, so the two can be compared meaningfully.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import heapq
import os
from pathlib import Path
from typing import Any

from pretrain.stage_i_graph_v2 import STAGE_PRIORITY, canonical_json_bytes, open_authoritative

AUDIT_SCHEMA = "petitgpt-stage-i-realization-audit-v1"

# The frozen selection fingerprint's domain, restated here so the streaming reproduction is
# byte-identical. stage_i_select_v1.py is NOT modified: this module reproduces its output.
SELECTION_FINGERPRINT_DOMAIN = b"PetitGPT-stage-i-selection-fingerprint-v1\0"

# The NEW order-sensitive commitment. Explicit and versioned, and deliberately a separate value
# from the frozen fingerprint rather than a redefinition of it.
SELECTION_SEQUENCE_SCHEMA = "petitgpt-stage-i-selection-sequence-v1"
SELECTION_SEQUENCE_DOMAIN = b"PetitGPT-stage-i-selection-sequence-v1\0"

# Bounded working state for the external sort. Deliberately small enough that the audit's peak
# memory is a property of this constant rather than of the realization's size.
DEFAULT_SORT_CHUNK_LINES = 200_000

# Bounded read window for streaming a shard. Exposed so a regression can force many small reads
# over a spool/shard larger than the window and prove the path never slurps a whole file.
DEFAULT_READ_WINDOW_BYTES = 1 << 20

# Bounded external-merge fan-in. The merge proceeds in generations so the number of simultaneously
# open runs -- and therefore file descriptors and heap entries -- is fixed regardless of how many
# spill runs the input produced.
MAX_MERGE_FANIN = 8


def _split_key(line: str, key_arity: int) -> tuple[str, ...]:
    return tuple(line.rstrip("\n").split("\t")[:key_arity])


def _split_row(line: str, key_arity: int) -> tuple[tuple[str, ...], str]:
    parts = line.rstrip("\n").split("\t")
    return tuple(parts[:key_arity]), "\t".join(parts[key_arity:])


class AuditError(RuntimeError):
    """Fail-closed physical-realization audit condition."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


# --------------------------------------------------------------------- bounded line streaming


def stream_lines(
    path: Path, *, read_window_bytes: int = DEFAULT_READ_WINDOW_BYTES
) -> Iterator[bytes]:
    """Yield newline-framed records from a file, bounded.

    Reads at most ``read_window_bytes`` per call and holds at most one line plus one partial tail.
    There is deliberately no code path that materialises the whole file: a `read()` over a
    multi-gigabyte spool was the reviewed defect, not a detail.
    """
    _require(read_window_bytes > 0, "read window must be positive")
    with open_authoritative(path) as (handle, _identity):
        pending = b""
        while True:
            chunk = handle.read(read_window_bytes)
            if not chunk:
                break
            pending += chunk
            start = 0
            while True:
                index = pending.find(b"\n", start)
                if index < 0:
                    break
                yield pending[start:index]
                start = index + 1
            pending = pending[start:]
        _require(
            pending == b"",
            f"{path}: file does not end with a newline; last record is truncated",
        )


class ShardReader:
    """Stream one shard's lines while hashing and counting exactly the bytes consumed.

    A class rather than a generator handing back closures: the digest and byte count are results
    of the read, so they belong on an object the caller interrogates afterwards. Leaking them
    through loop variables leaves them undefined for an empty file and reads as an accident.
    """

    def __init__(self, path: Path, *, read_window_bytes: int = DEFAULT_READ_WINDOW_BYTES) -> None:
        _require(read_window_bytes > 0, "read window must be positive")
        self.path = path
        self.read_window_bytes = read_window_bytes
        self._digest = hashlib.sha256()
        self._bytes = 0
        self._reads = 0
        self._finished = False

    def __iter__(self) -> Iterator[bytes]:
        with open_authoritative(self.path) as (handle, _identity):
            pending = b""
            while True:
                chunk = handle.read(self.read_window_bytes)
                if not chunk:
                    break
                self._reads += 1
                self._digest.update(chunk)
                self._bytes += len(chunk)
                pending += chunk
                start = 0
                while True:
                    index = pending.find(b"\n", start)
                    if index < 0:
                        break
                    yield pending[start:index]
                    start = index + 1
                pending = pending[start:]
            _require(
                pending == b"",
                f"{self.path}: file does not end with a newline; last record is truncated",
            )
        self._finished = True

    @property
    def sha256(self) -> str:
        _require(self._finished, "shard digest requested before the shard was fully read")
        return self._digest.hexdigest()

    @property
    def bytes_read(self) -> int:
        _require(self._finished, "shard byte count requested before the shard was fully read")
        return self._bytes

    @property
    def read_calls(self) -> int:
        """How many bounded reads the stream actually took; used by the streaming regression."""
        return self._reads


# --------------------------------------------------------------------- deterministic external sort


@dataclass
class ExternalSorter:
    """Deterministic disk-backed sort with bounded memory.

    Used for the three global questions the audit cannot answer in one streaming pass: is every
    cleaned identity unique across the whole realization, are a node's selection ordinals exactly
    the contiguous domain, and what is a node's fingerprint over its identities in sorted order.

    Holding 13.8M identities as Python objects would cost several GB for no reason; chunking to
    temp files and k-way merging costs one bounded buffer and a deterministic amount of scratch
    disk that is deleted on completion.
    """

    work_dir: Path
    chunk_lines: int = DEFAULT_SORT_CHUNK_LINES
    _buffer: list[tuple] = field(default_factory=list, init=False, repr=False)
    _runs: list[Path] = field(default_factory=list, init=False, repr=False)
    _count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        _require(self.chunk_lines > 0, "sort chunk size must be positive")
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def add(self, key: tuple, payload: str) -> None:
        self._buffer.append((key, payload))
        self._count += 1
        if len(self._buffer) >= self.chunk_lines:
            self._spill()

    def _spill(self) -> None:
        if not self._buffer:
            return
        self._buffer.sort()
        path = self.work_dir / f"run-{len(self._runs):06d}.tsv"
        with open(path, "w", encoding="utf-8") as handle:
            for key, payload in self._buffer:
                handle.write("\t".join(str(part) for part in key) + "\t" + payload + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._runs.append(path)
        self._buffer.clear()

    def _merge_runs(self, runs: list[Path], key_arity: int, generation: int) -> list[Path]:
        """One deterministic merge generation over at most ``MAX_MERGE_FANIN`` runs at a time."""
        merged: list[Path] = []
        for group_index in range(0, len(runs), MAX_MERGE_FANIN):
            group = runs[group_index : group_index + MAX_MERGE_FANIN]
            if len(group) == 1:
                merged.append(group[0])
                continue
            out = self.work_dir / f"merge-{generation:03d}-{len(merged):06d}.tsv"
            handles = [open(path, encoding="utf-8") for path in group]
            try:
                with open(out, "w", encoding="utf-8") as sink:
                    for line in heapq.merge(*handles, key=lambda ln: _split_key(ln, key_arity)):
                        sink.write(line)
                    sink.flush()
                    os.fsync(sink.fileno())
            finally:
                for handle in handles:
                    handle.close()
            for path in group:
                path.unlink(missing_ok=True)
            merged.append(out)
        return merged

    def sorted_items(self, key_arity: int) -> Iterator[tuple[tuple[str, ...], str]]:
        """Merge every run in sorted order with a bounded number of open files.

        R1 opened every spill run simultaneously, so file descriptors and merge buffers scaled
        with the number of runs, which itself scales with the data. Merging in generations of at
        most ``MAX_MERGE_FANIN`` keeps the open-file count and the heap fixed no matter how many
        runs the input produced.
        """
        self._spill()
        runs = list(self._runs)
        generation = 0
        while len(runs) > MAX_MERGE_FANIN:
            runs = self._merge_runs(runs, key_arity, generation)
            generation += 1
        self._runs = list(runs)
        if not runs:
            return
        handles = [open(path, encoding="utf-8") for path in runs]
        try:
            for line in heapq.merge(*handles, key=lambda ln: _split_key(ln, key_arity)):
                yield _split_row(line, key_arity)
        finally:
            for handle in handles:
                handle.close()

    def close(self) -> None:
        self._buffer.clear()
        for path in self._runs:
            path.unlink(missing_ok=True)
        self._runs.clear()

    @property
    def count(self) -> int:
        return self._count


# --------------------------------------------------------------------- audit result


# Ordinals are sorted as zero-padded strings so lexical order equals numeric order. Without this
# "10" would sort before "9" and the contiguity check would pass on a permuted domain.
_ORDINAL_WIDTH = 20


def _pad(value: int) -> str:
    _require(0 <= value < 10**_ORDINAL_WIDTH, "ordinal out of representable range")
    return str(value).zfill(_ORDINAL_WIDTH)


@dataclass(frozen=True)
class NodeAudit:
    """What one node's records physically are, derived only from the bytes on disk."""

    source_id: str
    stage: str
    records: int
    content_tokens: int
    serialized_tokens: int
    selection_fingerprint: str
    selection_sequence_commitment: str
    selection_ordinal_count: int
    input_binding_ids: tuple[str, ...]

    def as_canonical(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "stage": self.stage,
            "records": self.records,
            "content_tokens": self.content_tokens,
            "serialized_tokens": self.serialized_tokens,
            "selection_fingerprint": self.selection_fingerprint,
            "selection_sequence_commitment": self.selection_sequence_commitment,
            "selection_ordinal_count": self.selection_ordinal_count,
            "input_binding_ids": list(self.input_binding_ids),
        }


@dataclass(frozen=True)
class RealizationAudit:
    """The deterministic physical truth of a staged or published realization."""

    schema_version: str
    records: int
    content_tokens: int
    serialized_tokens: int
    shards: int
    unique_cleaned_identities: int
    per_shard: tuple[Mapping[str, Any], ...]
    nodes: tuple[NodeAudit, ...]

    def as_canonical(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "records": self.records,
            "content_tokens": self.content_tokens,
            "serialized_tokens": self.serialized_tokens,
            "shards": self.shards,
            "unique_cleaned_identities": self.unique_cleaned_identities,
            "per_shard": [dict(entry) for entry in self.per_shard],
            "nodes": [node.as_canonical() for node in self.nodes],
        }

    def digest(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.as_canonical())).hexdigest()

    def node(self, source_id: str) -> NodeAudit:
        for entry in self.nodes:
            if entry.source_id == source_id:
                return entry
        raise AuditError(f"audit has no node {source_id!r}")


# --------------------------------------------------------------------- the audit itself


def audit_realization(
    documents_dir: Path,
    shard_names: Sequence[str],
    work_dir: Path,
    *,
    validate_record: Callable[[Any], Mapping[str, Any]],
    parse_record: Callable[[bytes], Mapping[str, Any]],
    node_binding_projection: Mapping[str, Sequence[str]],
    read_window_bytes: int = DEFAULT_READ_WINDOW_BYTES,
    sort_chunk_lines: int = DEFAULT_SORT_CHUNK_LINES,
) -> RealizationAudit:
    """Derive the physical truth of a realization, in bounded memory, from the records themselves.

    ``validate_record`` and ``parse_record`` are injected rather than imported so that this module
    sits below the output-schema module in the dependency order and the two cannot import each
    other. The schema stays owned by the output contract; the physical accounting stays here.

    Enforced in the single streaming pass: canonical record bytes (re-serialising the parsed record
    must reproduce the exact line), closed record schema, strictly ascending physical order across
    the whole realization including shard boundaries, per-shard record counts, and per-shard byte
    length and digest computed from the same bytes that were consumed.

    Enforced through the bounded external sorts: global ``cleaned_text_sha256`` uniqueness, and per
    node the exact contiguous ``selection_ordinal_within_node`` domain ``0..n-1`` with no gap,
    duplicate or extra, plus the node's fingerprint over its identities.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    identity_sorter = ExternalSorter(work_dir / "sort-identity", sort_chunk_lines)
    ordinal_sorter = ExternalSorter(work_dir / "sort-ordinal", sort_chunk_lines)
    fingerprint_sorter = ExternalSorter(work_dir / "sort-fingerprint", sort_chunk_lines)

    try:
        per_shard: list[dict[str, Any]] = []
        total_records = 0
        total_content = 0
        total_serialized = 0
        node_records: dict[str, int] = {}
        node_content: dict[str, int] = {}
        node_serialized: dict[str, int] = {}
        node_stage: dict[str, str] = {}
        node_bindings: dict[str, set[str]] = {}
        authorized_bindings = {
            source_id: frozenset(bindings)
            for source_id, bindings in node_binding_projection.items()
        }
        previous_key: tuple[int, str, str, int] | None = None

        for name in shard_names:
            path = documents_dir / name
            _require(path.is_file(), f"shard {name} is missing from {documents_dir}")
            shard_records = 0
            reader = ShardReader(path, read_window_bytes=read_window_bytes)

            for line in reader:
                record = parse_record(line)
                validate_record(record)
                # Canonical bytes: the line on disk must be exactly the canonical serialisation of
                # what it parses to. This is what makes a shard digest a statement about content
                # rather than about incidental whitespace or key order.
                _require(
                    canonical_json_bytes(record) == line + b"\n",
                    f"{name}: record bytes are not canonical at physical position {total_records}",
                )
                key = (
                    STAGE_PRIORITY[record["stage"]],
                    record["source_id"],
                    record["input_binding_id"],
                    record["stable_input_record_ordinal"],
                )
                _require(
                    previous_key is None or key > previous_key,
                    f"{name}: records are out of canonical physical order at position "
                    f"{total_records}",
                )
                previous_key = key

                source_id = record["source_id"]
                stage = record["stage"]
                known = node_stage.setdefault(source_id, stage)
                _require(
                    known == stage,
                    f"node {source_id} appears under two stages: {known!r} and {stage!r}",
                )
                # R2-B: the node must be authorized and the binding must be authorized FOR THAT
                # NODE. R1 used input_binding_id only as a sort key, so a record naming a binding
                # nobody declared, or a real binding attached to the wrong node, went straight
                # through the publisher and the consumer.
                allowed = authorized_bindings.get(source_id)
                _require(
                    allowed is not None,
                    f"record names node {source_id!r}, which is not in the authorized "
                    "node/binding projection",
                )
                _require(
                    record["input_binding_id"] in allowed,
                    f"node {source_id} is not authorized to draw from input binding "
                    f"{record['input_binding_id']!r}; authorized: {sorted(allowed)}",
                )
                node_bindings.setdefault(source_id, set()).add(record["input_binding_id"])

                identity = record["cleaned_text_sha256"]
                ordinal = record["selection_ordinal_within_node"]

                identity_sorter.add((identity,), source_id)
                ordinal_sorter.add((source_id, _pad(ordinal)), identity)
                fingerprint_sorter.add((source_id, identity), "")

                node_records[source_id] = node_records.get(source_id, 0) + 1
                node_content[source_id] = (
                    node_content.get(source_id, 0) + record["content_token_count"]
                )
                node_serialized[source_id] = (
                    node_serialized.get(source_id, 0) + record["serialized_token_count"]
                )
                total_records += 1
                total_content += record["content_token_count"]
                total_serialized += record["serialized_token_count"]
                shard_records += 1

            _require(shard_records > 0, f"shard {name} contains no records")
            per_shard.append({
                "name": name,
                "records": shard_records,
                "bytes": reader.bytes_read,
                "sha256": reader.sha256,
            })

        unique = _check_global_identity_uniqueness(identity_sorter)
        # One sorted pass now yields BOTH the contiguity check and the order-sensitive commitment,
        # because the ordinal sorter carries the identity that sat at each ordinal.
        sequences = _selection_sequences(ordinal_sorter, node_stage)
        fingerprints = _node_fingerprints(fingerprint_sorter, node_records)

        _require(
            set(sequences) == set(node_records) == set(fingerprints),
            "internal: node sets disagree between the streaming pass and the sorted passes",
        )

        nodes = tuple(
            NodeAudit(
                source_id=source_id,
                stage=node_stage[source_id],
                records=node_records[source_id],
                content_tokens=node_content[source_id],
                serialized_tokens=node_serialized[source_id],
                selection_fingerprint=fingerprints[source_id],
                selection_sequence_commitment=sequences[source_id],
                selection_ordinal_count=node_records[source_id],
                input_binding_ids=tuple(sorted(node_bindings.get(source_id, ()))),
            )
            for source_id in sorted(node_records, key=lambda s: (STAGE_PRIORITY[node_stage[s]], s))
        )
        return RealizationAudit(
            schema_version=AUDIT_SCHEMA,
            records=total_records,
            content_tokens=total_content,
            serialized_tokens=total_serialized,
            shards=len(per_shard),
            unique_cleaned_identities=unique,
            per_shard=tuple(per_shard),
            nodes=nodes,
        )
    finally:
        identity_sorter.close()
        ordinal_sorter.close()
        fingerprint_sorter.close()


def _check_global_identity_uniqueness(sorter: ExternalSorter) -> int:
    """Every cleaned identity must appear exactly once across the whole realization."""
    unique = 0
    previous: str | None = None
    for (identity,), owner in sorter.sorted_items(1):
        if previous is not None and identity == previous:
            raise AuditError(
                f"cleaned identity {identity} appears more than once in the realization "
                f"(second occurrence owned by {owner!r}); global ownership is violated"
            )
        previous = identity
        unique += 1
    return unique


class StreamingNodeFingerprint:
    """Reproduce ``selection_fingerprint_v1`` byte-for-byte from a sorted stream.

    The frozen definition is
    ``sha256(DOMAIN || count_be64 || concat(bytes.fromhex(id) for id in sorted(ids)))``.
    The count comes first, which is why R1 buffered the whole list: it did not know the count until
    it had them all. But the streaming pass already counts each node's records, so the count can be
    supplied up front and the identities fed in as they arrive from the external sort.

    This changes nothing about the fingerprint's value or meaning -- it is the same bytes in the
    same order -- only about how much memory it takes to compute. The equivalence is pinned
    directly against ``selection_fingerprint_v1`` in the regressions.
    """

    def __init__(self, count: int) -> None:
        _require(count >= 0, "fingerprint count must not be negative")
        self._expected = count
        self._seen = 0
        self._previous: str | None = None
        self._digest = hashlib.sha256(SELECTION_FINGERPRINT_DOMAIN)
        self._digest.update(count.to_bytes(8, "big"))

    def update(self, identity: str) -> None:
        _require(
            self._previous is None or identity >= self._previous,
            "streaming fingerprint requires identities in ascending order",
        )
        self._previous = identity
        self._seen += 1
        _require(self._seen <= self._expected, "more identities than the declared count")
        self._digest.update(bytes.fromhex(identity))

    def hexdigest(self) -> str:
        _require(
            self._seen == self._expected,
            f"streaming fingerprint saw {self._seen} identities, expected {self._expected}",
        )
        return self._digest.hexdigest()


class StreamingSequenceCommitment:
    """Order-sensitive commitment to a node's exact ``ordinal -> identity`` sequence.

    R1 checked only that a node's ordinals formed the contiguous domain, and the frozen selection
    fingerprint hashes a *sorted set*, so permuting which identity sits at which ordinal changed
    nothing either check could see. Codex published `[1, 0, 2, 3]` through both.

    This is the missing commitment, added alongside the frozen fingerprint rather than replacing
    it: the fingerprint still answers "which identities were selected", and this answers "in
    exactly what order". Each pair is folded in as it arrives, so it is order-sensitive by
    construction and costs one hash object per node.
    """

    def __init__(self, *, source_id: str, stage: str) -> None:
        self._count = 0
        self._expected_ordinal = 0
        self._digest = hashlib.sha256(SELECTION_SEQUENCE_DOMAIN)
        self._digest.update(SELECTION_SEQUENCE_SCHEMA.encode("ascii"))
        self._digest.update(b"\0")
        self._digest.update(source_id.encode("utf-8"))
        self._digest.update(b"\0")
        self._digest.update(stage.encode("utf-8"))
        self._digest.update(b"\0")

    def update(self, ordinal: int, identity: str) -> None:
        _require(
            ordinal == self._expected_ordinal,
            f"selection sequence is not contiguous: expected ordinal {self._expected_ordinal}, "
            f"found {ordinal}",
        )
        self._expected_ordinal += 1
        self._count += 1
        self._digest.update(ordinal.to_bytes(8, "big"))
        self._digest.update(bytes.fromhex(identity))

    def hexdigest(self) -> str:
        # The count is folded in last so the digest also commits to the sequence's length; a
        # truncated or extended sequence cannot collide with a correct one.
        final = self._digest.copy()
        final.update(b"\0")
        final.update(self._count.to_bytes(8, "big"))
        return final.hexdigest()

    @property
    def count(self) -> int:
        return self._count


def selection_sequence_commitment(
    *, source_id: str, stage: str, pairs: Iterable[tuple[int, str]]
) -> str:
    """Convenience wrapper: the commitment over an in-order ``(ordinal, identity)`` sequence.

    Used by Stage-I Pass 1 to state the expected sequence *before* anything is materialized, so
    the value the audit must reproduce does not come from the physical records it is checking.
    """
    commitment = StreamingSequenceCommitment(source_id=source_id, stage=stage)
    for ordinal, identity in pairs:
        commitment.update(ordinal, identity)
    return commitment.hexdigest()


def _node_fingerprints(sorter: ExternalSorter, counts: Mapping[str, int]) -> dict[str, str]:
    """Recompute each node's frozen fingerprint from a sorted stream, holding no identity list."""
    fingerprints: dict[str, str] = {}
    current: str | None = None
    streaming: StreamingNodeFingerprint | None = None
    for (source_id, identity), _rest in sorter.sorted_items(2):
        if source_id != current:
            if current is not None and streaming is not None:
                fingerprints[current] = streaming.hexdigest()
            current = source_id
            _require(source_id in counts, f"fingerprint stream saw unknown node {source_id!r}")
            streaming = StreamingNodeFingerprint(counts[source_id])
        streaming.update(identity)
    if current is not None and streaming is not None:
        fingerprints[current] = streaming.hexdigest()
    return fingerprints


def _selection_sequences(sorter: ExternalSorter, stages: Mapping[str, str]) -> dict[str, str]:
    """Reconstruct each node's order-sensitive sequence commitment from the sorted ordinals."""
    commitments: dict[str, str] = {}
    current: str | None = None
    builder: StreamingSequenceCommitment | None = None
    for (source_id, padded), identity in sorter.sorted_items(2):
        if source_id != current:
            if current is not None and builder is not None:
                commitments[current] = builder.hexdigest()
            current = source_id
            _require(source_id in stages, f"sequence stream saw unknown node {source_id!r}")
            builder = StreamingSequenceCommitment(source_id=source_id, stage=stages[source_id])
        builder.update(int(padded), identity)
    if current is not None and builder is not None:
        commitments[current] = builder.hexdigest()
    return commitments


__all__ = [
    "AUDIT_SCHEMA",
    "DEFAULT_READ_WINDOW_BYTES",
    "DEFAULT_SORT_CHUNK_LINES",
    "AuditError",
    "ExternalSorter",
    "NodeAudit",
    "RealizationAudit",
    "MAX_MERGE_FANIN",
    "SELECTION_SEQUENCE_DOMAIN",
    "SELECTION_SEQUENCE_SCHEMA",
    "StreamingNodeFingerprint",
    "StreamingSequenceCommitment",
    "audit_realization",
    "selection_sequence_commitment",
    "stream_lines",
    "ShardReader",
]
