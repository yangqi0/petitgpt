#!/usr/bin/env python3
"""Stage-M contract core v1: schemas, accounting, commitments, environment, byte bundle.

Stage M turns the *accepted* Stage-I document realization into canonical schema-3 packed
training shards. This module holds the parts of that contract that must be closed and
versioned: the canonical hash-bearing encoding, the exact packing accounting, the input
sequence commitment, the frozen runtime environment, and the implementation-byte bundle.

Nothing here reads Stage-I or writes a release; those live in ``stage_m_input_v1`` and
``stage_m_output_v1``. Keeping the closed definitions in one dependency-free module is what
lets the candidate plan bind them by bytes.

Accounting (owner-frozen, DECISIONS D-145/D-148). For one stage stream carrying ``N``
serialized input tokens at context length ``T``::

    q                         = (N - 1) // T
    training_sequences        = q
    model_input_positions     = q * T
    retained_stored_token_ids = q * T + 1
    tail_transitions          = (N - 1) - q * T
    final_lookahead_tokens    = 1
    padding_tokens            = 0
    invariant                 N == model_input_positions + tail_transitions + 1

The final lookahead token is the ``T + 1``-th token of the last window: it is the supervised
next-token label for the last model input position. It is deliberately *not* padding, *not* a
tail transition and *not* a model input position, and the six quantities are reported under six
distinct names so that no consumer has to guess which one an ambiguous ``training_tokens`` meant.

Two stage streams are packed independently, so each drops its own tail. There is no stream in
which Stage B is concatenated onto Stage A.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from types import MappingProxyType
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --------------------------------------------------------------------- schema names

CANDIDATE_PLAN_SCHEMA = "petitgpt-m-candidate-plan-v1"
INPUT_SEQUENCE_COMMITMENT_SCHEMA = "petitgpt-stage-m-input-sequence-commitment-v1"
ACCOUNTING_SCHEMA = "petitgpt-stage-m-accounting-v1"
BUNDLE_SCHEMA = "petitgpt-m-implementation-bundle-v1"
RELEASE_PROFILE_SCHEMA = "petitgpt-stage-m-release-profile-v1"
ORDERING_CONTRACT_ID = "ACCEPTED_STAGE_I_PHYSICAL_ORDER_V1"

STAGE_I_RECORD_SCHEMA = "petitgpt-stage-i-document-v1"
STAGE_I_MANIFEST_SCHEMA = "petitgpt-stage-i-manifest-v1"

STAGE_STREAMS = ("stage_a", "stage_b")

# The frozen final-model contract Stage-M output must be packed for (CLAUDE.md; ARCH D-037).
# Declared here as closed constants rather than imported from src.model, so the Stage-M closure
# does not drag torch onto the packing path; seq_len and vocab_size are the two fields Stage M
# can and does verify against the real tokenizer and the real window geometry.
MODEL_CONTRACT = MappingProxyType({
    "n_layers": 30,
    "d_model": 576,
    "n_heads": 9,
    "n_kv_heads": 3,
    "d_ff": 1536,
    "seq_len": 2048,
    "vocab_size": 32_000,
})
SEQ_LEN = 2048

# Packing semantics, frozen by data_design.md section 17 and DECISIONS D-015/D-022. Restated as
# a bound object so the candidate plan commits to them rather than implying them.
PACKING_SEMANTICS = MappingProxyType({
    "document_form": "[BOS] content [EOS]",
    "textual_document_separator": None,
    "adjacent_document_boundary": "[EOS][BOS]",
    "framing_applied_by": "stage_m",
    "framing_applied_times": 1,
    "read_length": 2049,
    "model_input_length": 2048,
    "label_length": 2048,
    "stride": 2048,
    "stride_t_plus_1_forbidden": True,
    "label_shift_owner": "dataset_consumer",
    "documents_may_span_blocks": True,
    "multiple_documents_per_block": True,
    "padding": "none",
    "tail_policy": "one dropped tail per stage stream",
    "stage_stream_count": 2,
    "stage_streams_concatenated": False,
})

# --------------------------------------------------------------------- errors


class StageMError(RuntimeError):
    """Controlled Stage-M failure. Never raised for a condition a caller can ignore."""


def require(condition: object, message: str) -> None:
    if not condition:
        raise StageMError(message)


# --------------------------------------------------------------------- canonical encoding

_HEX64 = frozenset("0123456789abcdef")


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical hash-bearing serialisation: UTF-8, sorted keys, fixed separators, one newline.

    Deliberately identical in behaviour to the Stage-I encoding, but defined here so the
    Stage-M closure does not depend on a Stage-I module for its own wire contract.
    """
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_hex(canonical_json_bytes(value))


def file_sha256(path: Path, *, chunk: int = 1 << 20) -> str:
    """Hash a regular non-symlink file, rejecting a size change during the read."""
    resolved = Path(path)
    require(not resolved.is_symlink(), f"refusing to hash a symlink: {resolved}")
    require(resolved.is_file(), f"not a regular file: {resolved}")
    before = resolved.stat()
    digest = hashlib.sha256()
    with open(resolved, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    after = resolved.stat()
    require(
        (before.st_size, before.st_mtime_ns, before.st_ino)
        == (after.st_size, after.st_mtime_ns, after.st_ino),
        f"file changed while hashing: {resolved}",
    )
    return digest.hexdigest()


def validated_sha256(value: object, *, field: str) -> str:
    require(
        isinstance(value, str) and len(value) == 64 and set(value) <= _HEX64,
        f"{field} must be a lowercase 64-hex SHA-256, got {value!r}",
    )
    assert isinstance(value, str)
    return value


def require_int(value: object, *, field: str, minimum: int = 0) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
        f"{field} must be an int >= {minimum}, got {value!r}",
    )
    assert isinstance(value, int)
    return value


# --------------------------------------------------------------------- accounting


@dataclass(frozen=True)
class StreamAccounting:
    """The six distinct packing quantities for one stage stream. No collapsed alias exists."""

    stage: str
    seq_len: int
    input_serialized_tokens: int
    training_sequences: int
    model_input_positions: int
    retained_stored_token_ids: int
    tail_transitions: int
    final_lookahead_tokens: int
    padding_tokens: int

    def as_canonical(self) -> dict[str, Any]:
        return {
            "schema_version": ACCOUNTING_SCHEMA,
            "stage": self.stage,
            "seq_len": self.seq_len,
            "input_serialized_tokens": self.input_serialized_tokens,
            "training_sequences": self.training_sequences,
            "model_input_positions": self.model_input_positions,
            "retained_stored_token_ids": self.retained_stored_token_ids,
            "tail_transitions": self.tail_transitions,
            "final_lookahead_tokens": self.final_lookahead_tokens,
            "padding_tokens": self.padding_tokens,
        }


def stream_accounting(stage: str, input_serialized_tokens: int, seq_len: int) -> StreamAccounting:
    """Derive the frozen accounting for one stream. The only place these formulas exist."""
    require(stage in STAGE_STREAMS, f"unknown stage stream: {stage!r}")
    n = require_int(input_serialized_tokens, field="input_serialized_tokens", minimum=1)
    t = require_int(seq_len, field="seq_len", minimum=1)
    require(n >= t + 1, f"{stage}: stream too short to form one window: N={n}, T={t}")

    q = (n - 1) // t
    model_input_positions = q * t
    accounting = StreamAccounting(
        stage=stage,
        seq_len=t,
        input_serialized_tokens=n,
        training_sequences=q,
        model_input_positions=model_input_positions,
        retained_stored_token_ids=model_input_positions + 1,
        tail_transitions=(n - 1) - model_input_positions,
        final_lookahead_tokens=1,
        padding_tokens=0,
    )
    # The invariant is checked here rather than trusted, because every downstream release
    # number is derived from this object.
    require(
        accounting.input_serialized_tokens
        == accounting.model_input_positions
        + accounting.tail_transitions
        + accounting.final_lookahead_tokens,
        f"{stage}: accounting invariant violated: "
        f"N={n} != {accounting.model_input_positions}+{accounting.tail_transitions}+1",
    )
    require(accounting.tail_transitions < t, f"{stage}: tail must be shorter than one stride")
    return accounting


def total_accounting(streams: Sequence[StreamAccounting]) -> dict[str, Any]:
    """Sum the per-stream quantities. Sums are reported, never used to re-derive a stream."""
    require(bool(streams), "total_accounting requires at least one stream")
    return {
        "schema_version": ACCOUNTING_SCHEMA,
        "streams": [s.stage for s in streams],
        "total_input_serialized_tokens": sum(s.input_serialized_tokens for s in streams),
        "total_training_sequences": sum(s.training_sequences for s in streams),
        "total_model_input_positions": sum(s.model_input_positions for s in streams),
        "total_retained_stored_token_ids": sum(s.retained_stored_token_ids for s in streams),
        "total_tail_transitions": sum(s.tail_transitions for s in streams),
        "total_final_lookahead_tokens": sum(s.final_lookahead_tokens for s in streams),
        "total_padding_tokens": 0,
    }


# --------------------------------------------------------------------- input commitment


def canonical_record_commitment_payload(
    *,
    stage: str,
    ordinal: int,
    source_id: str,
    input_binding_id: str,
    stable_input_record_ordinal: int,
    canonical_fingerprint: str,
    cleaned_text_sha256: str,
    raw_sha256: str,
    input_record_sha256: str,
    selection_ordinal_within_node: int,
    content_token_count: int,
    serialized_token_count: int,
) -> bytes:
    """Closed canonical encoding of one consumed record, in consumption position.

    ``ordinal`` is the record's 0-based position in the Stage-M consumption order for its
    stream, so reordering two records changes the payload of both. Every identity and count a
    substitution could target is bound explicitly. This is a closed versioned JSON encoding,
    never a Python ``repr``.
    """
    return canonical_json_bytes({
        "schema_version": INPUT_SEQUENCE_COMMITMENT_SCHEMA,
        "stage": stage,
        "ordinal": ordinal,
        "source_id": source_id,
        "input_binding_id": input_binding_id,
        "stable_input_record_ordinal": stable_input_record_ordinal,
        "canonical_fingerprint": canonical_fingerprint,
        "cleaned_text_sha256": cleaned_text_sha256,
        "raw_sha256": raw_sha256,
        "input_record_sha256": input_record_sha256,
        "selection_ordinal_within_node": selection_ordinal_within_node,
        "content_token_count": content_token_count,
        "serialized_token_count": serialized_token_count,
    })


class InputSequenceCommitment:
    """Streaming commitment over one stage stream's exact consumed record sequence.

    Records are folded in consumption order into a running SHA-256, then sealed with the
    stream's totals. Omission, duplication, stage reassignment, reordering, record-byte
    substitution, identity substitution and token-count substitution each move the digest:
    the first four because ``ordinal``/``stage``/count are in the payload, the last three
    because the identities and counts are.
    """

    def __init__(self, stage: str) -> None:
        require(stage in STAGE_STREAMS, f"unknown stage stream: {stage!r}")
        self.stage = stage
        self._digest = hashlib.sha256()
        self._digest.update(
            canonical_json_bytes({
                "schema_version": INPUT_SEQUENCE_COMMITMENT_SCHEMA,
                "header": "stage-m-input-sequence",
                "stage": stage,
            })
        )
        self.record_count = 0
        self.serialized_tokens = 0
        self.content_tokens = 0
        self._sealed: str | None = None

    def update(
        self, payload: bytes, *, serialized_token_count: int, content_token_count: int
    ) -> None:
        require(self._sealed is None, "commitment already sealed")
        self._digest.update(payload)
        self.record_count += 1
        self.serialized_tokens += int(serialized_token_count)
        self.content_tokens += int(content_token_count)

    def seal(self) -> str:
        """Bind the running digest to the stream totals and return the commitment."""
        if self._sealed is not None:
            return self._sealed
        require(self.record_count > 0, f"{self.stage}: commitment over an empty stream")
        trailer = canonical_json_bytes({
            "schema_version": INPUT_SEQUENCE_COMMITMENT_SCHEMA,
            "trailer": "stage-m-input-sequence",
            "stage": self.stage,
            "record_count": self.record_count,
            "serialized_tokens": self.serialized_tokens,
            "content_tokens": self.content_tokens,
        })
        self._digest.update(trailer)
        self._sealed = self._digest.hexdigest()
        return self._sealed

    def as_canonical(self) -> dict[str, Any]:
        return {
            "schema_version": INPUT_SEQUENCE_COMMITMENT_SCHEMA,
            "stage": self.stage,
            "commitment": self.seal(),
            "record_count": self.record_count,
            "serialized_tokens": self.serialized_tokens,
            "content_tokens": self.content_tokens,
        }


# --------------------------------------------------------------------- environment


REQUIRED_PYTHON_EXECUTABLE = "/workspace/petitgpt/.venv/bin/python"
REQUIRED_PYTHON_VERSION = "3.10.12"
REQUIRED_TOKENIZERS_VERSION = "0.22.2"


@dataclass(frozen=True)
class Environment:
    """Every runtime package that can materially change Stage-M output bytes.

    NumPy is bound as well as tokenizers: Stage M writes its shards through a NumPy uint16
    buffer, so the array library is part of the byte-producing path, not an incidental import.
    """

    python_executable: str
    python_version: str
    tokenizers_version: str
    numpy_version: str

    def as_canonical(self) -> dict[str, str]:
        return {
            "python_executable": self.python_executable,
            "python_version": self.python_version,
            "tokenizers_version": self.tokenizers_version,
            "numpy_version": self.numpy_version,
        }


def current_environment() -> Environment:
    import numpy
    import tokenizers

    return Environment(
        python_executable=str(Path(sys.executable)),
        python_version=platform.python_version(),
        tokenizers_version=str(tokenizers.__version__),
        numpy_version=str(numpy.__version__),
    )


def verify_environment(environment: Environment) -> None:
    require(
        environment.python_executable == REQUIRED_PYTHON_EXECUTABLE,
        f"Stage-M requires {REQUIRED_PYTHON_EXECUTABLE}, got {environment.python_executable}",
    )
    require(
        environment.python_version == REQUIRED_PYTHON_VERSION,
        f"Stage-M requires Python {REQUIRED_PYTHON_VERSION}, got {environment.python_version}",
    )
    require(
        environment.tokenizers_version == REQUIRED_TOKENIZERS_VERSION,
        f"Stage-M requires tokenizers {REQUIRED_TOKENIZERS_VERSION}, "
        f"got {environment.tokenizers_version}",
    )


# --------------------------------------------------------------------- implementation bundle

# The Stage-M producer closure. This is an explicit list, not a glob or a filename convention:
# membership cannot drift with the working directory or with what happens to be importable.
# Every entry is rehashed from disk at runtime; git HEAD is provenance metadata only.
M_IMPLEMENTATION_BUNDLE_FILES = (
    "pretrain/build_pretrain_shards.py",
    "pretrain/dataset_pretrain.py",
    "pretrain/stage_m_contract_v1.py",
    "pretrain/stage_m_input_v1.py",
    "pretrain/stage_m_output_v1.py",
    "pretrain/stage_m_realize_v1.py",
    "src/__init__.py",
    "src/special_tokens.py",
)

# The Stage-P native-provenance validator closure, kept deliberately separate from the producer
# bundle: the candidate-M plan binds what produces and validates Stage-M output, while a later
# Stage-P run plan binds the then-reviewed P implementation and the accepted M release.
P_NATIVE_IMPLEMENTATION_BUNDLE_FILES = (
    "pretrain/dataset_pretrain.py",
    "pretrain/stage_m_contract_v1.py",
    "pretrain/stage_m_input_v1.py",
    "pretrain/stage_m_output_v1.py",
    "pretrain/stage_p_native_provenance_v1.py",
    "src/__init__.py",
    "src/special_tokens.py",
)


def bundle_files(repo_root: Path, members: Sequence[str]) -> dict[str, str]:
    """Rehash every declared member from the executing installation, right now."""
    digests: dict[str, str] = {}
    for relative in members:
        path = repo_root / relative
        require(path.is_file(), f"implementation bundle member missing: {relative}")
        digests[relative] = file_sha256(path)
    return digests


def bundle_sha256(files: Mapping[str, str], *, schema: str = BUNDLE_SCHEMA) -> str:
    """One digest over the exact member list and their digests.

    The payload is the sorted mapping, so the digest binds repository-relative PATH as well as
    content: moving identical bytes to another member name moves this value, and a member that
    disappears cannot be compensated for by one that arrives.
    """
    return canonical_sha256({"schema_version": schema, "files": dict(sorted(files.items()))})


def m_implementation_bundle(repo_root: Path) -> tuple[dict[str, str], str]:
    files = bundle_files(repo_root, M_IMPLEMENTATION_BUNDLE_FILES)
    return files, bundle_sha256(files)


def p_native_implementation_bundle(repo_root: Path) -> tuple[dict[str, str], str]:
    files = bundle_files(repo_root, P_NATIVE_IMPLEMENTATION_BUNDLE_FILES)
    return files, bundle_sha256(files)


# --------------------------------------------------------------------- release profile

# Frozen for the M-v1 release only (DECISIONS D-148). These key spellings and the uint16 width
# were not independently owner-frozen historically; this profile freezes them for M-v1.
RELEASE_PROFILE = MappingProxyType({
    "schema_version": RELEASE_PROFILE_SCHEMA,
    "manifest_schema_family": "canonical-schema-3",
    "manifest_schema_version": 3,
    "manifest_filename": "meta.json",
    "completion_object_kind": "meta.json:status=complete",
    "storage_dtype": "uint16",
    "shard_basename_format": "shard_{index:05d}.bin",
    "split": "train",
    "stage_streams": list(STAGE_STREAMS),
    "padding": "none",
    "publication": "staged-and-atomic-rename",
    "shard_integrity_fields": ["path", "size_bytes", "token_count", "sha256"],
    "consumer": "PackedBinDataset",
})


def resolve_repo_root(explicit: Path | None = None) -> Path:
    """Resolve the executing installation root, never the process working directory."""
    installation = Path(ROOT).resolve()
    if explicit is None:
        return installation
    candidate = Path(explicit).expanduser().resolve()
    require(
        candidate == installation,
        f"repo_root must name the executing Stage-M installation {installation}, got {candidate}",
    )
    return candidate


def iter_sorted(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(values))


__all__ = [
    "ACCOUNTING_SCHEMA",
    "BUNDLE_SCHEMA",
    "CANDIDATE_PLAN_SCHEMA",
    "INPUT_SEQUENCE_COMMITMENT_SCHEMA",
    "MODEL_CONTRACT",
    "M_IMPLEMENTATION_BUNDLE_FILES",
    "ORDERING_CONTRACT_ID",
    "PACKING_SEMANTICS",
    "P_NATIVE_IMPLEMENTATION_BUNDLE_FILES",
    "RELEASE_PROFILE",
    "RELEASE_PROFILE_SCHEMA",
    "SEQ_LEN",
    "REQUIRED_PYTHON_EXECUTABLE",
    "REQUIRED_PYTHON_VERSION",
    "REQUIRED_TOKENIZERS_VERSION",
    "STAGE_I_MANIFEST_SCHEMA",
    "STAGE_I_RECORD_SCHEMA",
    "STAGE_STREAMS",
    "Environment",
    "InputSequenceCommitment",
    "StageMError",
    "StreamAccounting",
    "bundle_files",
    "bundle_sha256",
    "canonical_json_bytes",
    "canonical_record_commitment_payload",
    "canonical_sha256",
    "current_environment",
    "file_sha256",
    "iter_sorted",
    "m_implementation_bundle",
    "p_native_implementation_bundle",
    "require",
    "require_int",
    "resolve_repo_root",
    "sha256_hex",
    "stream_accounting",
    "total_accounting",
    "validated_sha256",
    "verify_environment",
]
