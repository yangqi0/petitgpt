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
CANDIDATE_PLAN_CONTRACT_SCHEMA = "petitgpt-m-candidate-plan-contract-v1"
SHARED_EXCLUSION_AUTHORITY_SCHEMA = "petitgpt-stage-m-shared-exclusion-authority-v1"

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
# R1-G. Stage-M writes uint16 shards and the frozen schema-3 consumer reads them back with
# numpy's *native* uint16, so producer and consumer only agree while the host is
# little-endian. Rather than change the canonical `dtype: "uint16"` spelling that existing
# schema-3 consumers already assume, the byte order is made explicit and bound: the writer
# emits explicit little-endian "<u2" and the environment contract refuses to run anywhere the
# native order is not little. On a big-endian host M-v1 production stops instead of silently
# producing shards the reader would byte-swap.
REQUIRED_BYTE_ORDER = "little"


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
    byte_order: str

    def as_canonical(self) -> dict[str, str]:
        return {
            "python_executable": self.python_executable,
            "python_version": self.python_version,
            "tokenizers_version": self.tokenizers_version,
            "numpy_version": self.numpy_version,
            "byte_order": self.byte_order,
        }


def current_environment() -> Environment:
    import numpy
    import tokenizers

    return Environment(
        python_executable=str(Path(sys.executable)),
        python_version=platform.python_version(),
        tokenizers_version=str(tokenizers.__version__),
        numpy_version=str(numpy.__version__),
        byte_order=sys.byteorder,
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
    require(
        environment.byte_order == REQUIRED_BYTE_ORDER,
        f"Stage-M v1 canonical uint16 output requires a {REQUIRED_BYTE_ORDER}-endian runtime, "
        f"got {environment.byte_order}",
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
    "storage_byte_order": REQUIRED_BYTE_ORDER,
    "storage_dtype_explicit": "<u2",
    "shard_basename_format": "shard_{index:05d}.bin",
    "split": "train",
    "stage_streams": list(STAGE_STREAMS),
    "padding": "none",
    "publication": "staged-and-atomic-rename",
    "shard_integrity_fields": ["path", "size_bytes", "token_count", "sha256"],
    "consumer": "PackedBinDataset",
})


# --------------------------------------------------------------------- candidate-plan contract


def _exact(actual: Any, expected: Any, *, field: str) -> None:
    require(
        actual == expected,
        f"candidate plan {field}: got {actual!r}, contract requires {expected!r}",
    )


def validate_candidate_plan_contract(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Closed semantic validation of a candidate-M plan. R2-B.

    Matching the reviewed plan digest proves *which bytes* were reviewed; it does not prove the
    bytes describe the one permitted Stage-M contract. A plan whose digest is correct but whose
    declarations contradict the frozen contract -- a different stride, a shuffle enabled, a
    big-endian profile -- would previously have resolved fine because the runtime is hard-coded
    to do the right thing regardless. That divergence between what a plan *says* and what the
    code *does* is exactly what a reviewer reading the plan would be misled by, so it is a
    contract error here.

    Returns a summary including the number of leaf fields actually validated.
    """
    checked = 0

    def check(actual: Any, expected: Any, field: str) -> None:
        nonlocal checked
        checked += 1
        _exact(actual, expected, field=field)

    require(isinstance(plan, Mapping), "candidate plan must be a JSON object")

    # --- identity and posture -----------------------------------------------------------
    check(plan.get("schema_version"), CANDIDATE_PLAN_SCHEMA, "schema_version")
    check(plan.get("authorization_status"), "NOT_AUTHORIZED", "authorization_status")
    check(plan.get("text_field"), "training_text", "text_field")
    check(plan.get("legacy_orchestration_used"), False, "legacy_orchestration_used")
    checked += 1
    require(
        isinstance(plan.get("authorization_note"), str) and plan["authorization_note"],
        "candidate plan authorization_note must be a non-empty string",
    )

    # --- frozen contracts, compared whole ------------------------------------------------
    check(dict(plan.get("model_contract") or {}), dict(MODEL_CONTRACT), "model_contract")
    check(dict(plan.get("packing_semantics") or {}), dict(PACKING_SEMANTICS), "packing_semantics")

    ordering = plan.get("ordering_contract")
    require(isinstance(ordering, Mapping), "candidate plan ordering_contract must be an object")
    assert isinstance(ordering, Mapping)
    check(ordering.get("policy"), ORDERING_CONTRACT_ID, "ordering_contract.policy")
    for flag in (
        "weighted_interleave",
        "selection_rank_reorder",
        "source_quota_scheduling",
        "hash_shuffle",
        "new_random_permutation",
    ):
        check(ordering.get(flag), False, f"ordering_contract.{flag}")
    check(
        ordering.get("only_shuffle"),
        "training-time block-ID permutation",
        "ordering_contract.only_shuffle",
    )
    checked += 1
    require(
        isinstance(ordering.get("rule"), str) and ordering["rule"],
        "candidate plan ordering_contract.rule must be a non-empty string",
    )

    # --- release profile ------------------------------------------------------------------
    profile = plan.get("release_profile")
    require(isinstance(profile, Mapping), "candidate plan release_profile must be an object")
    assert isinstance(profile, Mapping)
    for field, expected in RELEASE_PROFILE.items():
        check(profile.get(field), expected, f"release_profile.{field}")
    shard_tokens = require_int(
        profile.get("shard_tokens"), field="release_profile.shard_tokens", minimum=1
    )
    checked += 1

    # --- environment -----------------------------------------------------------------------
    environment = plan.get("environment_contract")
    require(isinstance(environment, Mapping), "candidate plan environment_contract must be object")
    assert isinstance(environment, Mapping)
    check(
        sorted(environment),
        sorted([
            "python_executable",
            "python_version",
            "tokenizers_version",
            "numpy_version",
            "byte_order",
        ]),
        "environment_contract field set",
    )
    check(
        environment.get("python_executable"),
        REQUIRED_PYTHON_EXECUTABLE,
        "environment_contract.python_executable",
    )
    check(
        environment.get("python_version"),
        REQUIRED_PYTHON_VERSION,
        "environment_contract.python_version",
    )
    check(
        environment.get("tokenizers_version"),
        REQUIRED_TOKENIZERS_VERSION,
        "environment_contract.tokenizers_version",
    )
    check(environment.get("byte_order"), REQUIRED_BYTE_ORDER, "environment_contract.byte_order")
    checked += 1
    require(
        isinstance(environment.get("numpy_version"), str) and environment["numpy_version"],
        "candidate plan environment_contract.numpy_version must be a non-empty string",
    )

    # --- implementation identity ------------------------------------------------------------
    files = plan.get("implementation_files")
    require(isinstance(files, Mapping), "candidate plan implementation_files must be an object")
    assert isinstance(files, Mapping)
    check(sorted(files), sorted(M_IMPLEMENTATION_BUNDLE_FILES), "implementation_files key set")
    for name, digest in files.items():
        validated_sha256(digest, field=f"implementation_files[{name}]")
        checked += 1
    check(
        bundle_sha256(dict(files)),
        plan.get("implementation_bundle_sha256"),
        "implementation_bundle_sha256 (recomputed from implementation_files)",
    )
    checked += 1
    require(
        isinstance(plan.get("implementation_commit"), str) and plan["implementation_commit"],
        "candidate plan implementation_commit must be a non-empty string",
    )

    # --- resources ---------------------------------------------------------------------------
    resources = plan.get("resources")
    require(isinstance(resources, Mapping), "candidate plan resources must be an object")
    assert isinstance(resources, Mapping)
    check(sorted(resources), ["reference_exclusion_manifest", "tokenizer"], "resources key set")
    tokenizer = resources["tokenizer"]
    require(isinstance(tokenizer, Mapping), "candidate plan resources.tokenizer must be object")
    validated_sha256(tokenizer.get("sha256"), field="resources.tokenizer.sha256")
    require_int(tokenizer.get("size_bytes"), field="resources.tokenizer.size_bytes", minimum=1)
    checked += 2
    exclusion = plan_exclusion_authority(plan)
    checked += 2

    # --- accepted Stage-I authority ------------------------------------------------------------
    accepted = plan.get("accepted_stage_i")
    require(isinstance(accepted, Mapping), "candidate plan accepted_stage_i must be an object")
    assert isinstance(accepted, Mapping)
    for field in (
        "run_identity",
        "manifest_sha256",
        "completion_object_sha256",
        "layer2_expected_result_sha256",
    ):
        validated_sha256(accepted.get(field), field=f"accepted_stage_i.{field}")
        checked += 1
    validated_sha256(
        plan.get("accepted_stage_i_identity_sha256"), field="accepted_stage_i_identity_sha256"
    )
    check(
        accepted.get("record_schema_version"),
        STAGE_I_RECORD_SCHEMA,
        "accepted_stage_i.record_schema_version",
    )
    check(
        accepted.get("manifest_schema_version"),
        STAGE_I_MANIFEST_SCHEMA,
        "accepted_stage_i.manifest_schema_version",
    )
    check(
        accepted.get("completion_object_kind"),
        "stage_i:COMPLETE",
        "accepted_stage_i.completion_object_kind",
    )
    inventory = accepted.get("shard_inventory")
    require(
        isinstance(inventory, list) and bool(inventory),
        "candidate plan accepted_stage_i.shard_inventory must be a non-empty list",
    )
    assert isinstance(inventory, list)
    check(len(inventory), accepted.get("shard_count"), "accepted_stage_i.shard_count")
    summed_records = 0
    for index, entry in enumerate(inventory):
        require(isinstance(entry, Mapping), f"shard_inventory[{index}] must be an object")
        validated_sha256(entry.get("sha256"), field=f"shard_inventory[{index}].sha256")
        summed_records += require_int(
            entry.get("records"), field=f"shard_inventory[{index}].records", minimum=1
        )
        require_int(entry.get("bytes"), field=f"shard_inventory[{index}].bytes", minimum=1)
    check(
        summed_records,
        accepted.get("total_records"),
        "accepted_stage_i.total_records (summed inventory)",
    )
    total_records = require_int(accepted.get("total_records"), field="total_records", minimum=1)
    total_serialized = require_int(
        accepted.get("total_serialized_tokens"), field="total_serialized_tokens", minimum=1
    )
    total_content = require_int(
        accepted.get("total_content_tokens"), field="total_content_tokens", minimum=1
    )
    check(
        total_content + 2 * total_records,
        total_serialized,
        "accepted_stage_i content + 2*records == serialized",
    )

    membership = accepted.get("stage_membership")
    require(isinstance(membership, Mapping), "accepted_stage_i.stage_membership must be object")
    assert isinstance(membership, Mapping)
    check(sorted(membership), sorted(STAGE_STREAMS), "accepted_stage_i.stage_membership key set")

    # --- stage streams and accounting ------------------------------------------------------
    streams = plan.get("stage_streams")
    require(isinstance(streams, Mapping), "candidate plan stage_streams must be an object")
    assert isinstance(streams, Mapping)
    check(sorted(streams), sorted(STAGE_STREAMS), "stage_streams key set")
    check(len(streams), 2, "stage stream count")
    check(
        int(PACKING_SEMANTICS["stage_stream_count"]),
        len(streams),
        "packing_semantics.stage_stream_count vs stage_streams",
    )
    check(list(profile["stage_streams"]), list(STAGE_STREAMS), "release_profile.stage_streams")

    seq_len = int(MODEL_CONTRACT["seq_len"])
    check(seq_len, SEQ_LEN, "model_contract.seq_len")
    check(int(PACKING_SEMANTICS["model_input_length"]), seq_len, "packing model_input_length == T")
    check(int(PACKING_SEMANTICS["stride"]), seq_len, "packing stride == T")
    check(int(PACKING_SEMANTICS["read_length"]), seq_len + 1, "packing read_length == T+1")
    check(int(PACKING_SEMANTICS["label_length"]), seq_len, "packing label_length == T")

    accountings: list[StreamAccounting] = []
    for stage in STAGE_STREAMS:
        entry = streams[stage]
        require(isinstance(entry, Mapping), f"stage_streams.{stage} must be an object")
        assert isinstance(entry, Mapping)
        check(
            entry.get("input_sequence_commitment_schema"),
            INPUT_SEQUENCE_COMMITMENT_SCHEMA,
            f"stage_streams.{stage}.input_sequence_commitment_schema",
        )
        validated_sha256(
            entry.get("input_sequence_commitment"),
            field=f"stage_streams.{stage}.input_sequence_commitment",
        )
        records = require_int(
            entry.get("input_record_count"), field=f"{stage}.input_record_count", minimum=1
        )
        serialized = require_int(
            entry.get("input_serialized_tokens"),
            field=f"{stage}.input_serialized_tokens",
            minimum=1,
        )
        content = require_int(
            entry.get("input_content_tokens"), field=f"{stage}.input_content_tokens", minimum=1
        )
        check(content + 2 * records, serialized, f"{stage} content + 2*records == serialized")
        declared = membership[stage]
        check(int(declared["records"]), records, f"{stage} membership records vs stream")
        check(
            int(declared["serialized_tokens"]),
            serialized,
            f"{stage} membership serialized tokens vs stream",
        )
        derived = stream_accounting(stage, serialized, seq_len)
        check(
            dict(entry.get("expected_accounting") or {}),
            derived.as_canonical(),
            f"stage_streams.{stage}.expected_accounting (recomputed)",
        )
        accountings.append(derived)
        checked += 4

    check(
        sum(a.input_serialized_tokens for a in accountings),
        total_serialized,
        "stage streams sum to accepted serialized tokens",
    )
    check(
        sum(int(membership[s]["records"]) for s in STAGE_STREAMS),
        total_records,
        "stage membership sums to accepted record count",
    )
    check(
        dict(plan.get("expected_totals") or {}),
        total_accounting(accountings),
        "expected_totals (recomputed)",
    )

    return {
        "schema_version": CANDIDATE_PLAN_CONTRACT_SCHEMA,
        "validated_field_count": checked,
        "shard_tokens": shard_tokens,
        "seq_len": seq_len,
        "exclusion_authority": exclusion,
        "stage_streams": list(STAGE_STREAMS),
    }


# --------------------------------------------------------------------- shared exclusion authority


def shared_exclusion_authority(
    *, sha256s: Sequence[str], hash_count: int, source: str
) -> dict[str, Any]:
    """The one canonical normalization of a reference-exclusion authority (R2-A).

    Every artifact that must share the exclusion contract -- the candidate-M plan, both Stage-M
    releases, the G tokenizer release and the G2 reference release -- is reduced to this same
    shape before comparison, so there is a single definition rather than one per call site.
    """
    normalized = tuple(
        sorted({validated_sha256(v, field=f"{source}.exclusion.sha256") for v in sha256s})
    )
    require(bool(normalized), f"{source}: declares no reference exclusion manifest")
    return {
        "schema_version": SHARED_EXCLUSION_AUTHORITY_SCHEMA,
        "source": source,
        "manifest_sha256s": list(normalized),
        "manifest_count": len(normalized),
        "hash_count": require_int(hash_count, field=f"{source}.exclusion.hash_count", minimum=1),
    }


def require_agreeing_exclusion_authorities(
    authorities: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Prove every supplied exclusion authority agrees, and return the agreed one.

    Digest set and hash count must both match. Comparing only the digests would let a release
    claim a different number of excluded identities under the same manifest name.
    """
    require(len(authorities) >= 2, "exclusion agreement needs at least two authorities")
    reference = authorities[0]
    digests = tuple(reference["manifest_sha256s"])
    count = int(reference["hash_count"])
    disagreements = [
        f"{a['source']}(sha={list(a['manifest_sha256s'])}, hash_count={a['hash_count']})"
        for a in authorities
        if tuple(a["manifest_sha256s"]) != digests or int(a["hash_count"]) != count
    ]
    require(
        not disagreements,
        "shared reference-exclusion authorities disagree: "
        f"expected sha={list(digests)}, hash_count={count}; offending {disagreements}",
    )
    return {
        "schema_version": SHARED_EXCLUSION_AUTHORITY_SCHEMA,
        "manifest_sha256s": list(digests),
        "manifest_count": len(digests),
        "hash_count": count,
        "sources": [a["source"] for a in authorities],
    }


def plan_exclusion_authority(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the candidate-M plan's own exclusion authority. R2-A: it must be compared too."""
    resources = plan.get("resources")
    require(isinstance(resources, Mapping), "plan.resources must be an object")
    assert isinstance(resources, Mapping)
    entry = resources.get("reference_exclusion_manifest")
    require(
        isinstance(entry, Mapping),
        "plan.resources.reference_exclusion_manifest must be an object",
    )
    assert isinstance(entry, Mapping)
    return shared_exclusion_authority(
        sha256s=[str(entry.get("sha256"))],
        hash_count=entry.get("hash_count"),
        source="candidate_m_plan",
    )


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
    "REQUIRED_BYTE_ORDER",
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
    "CANDIDATE_PLAN_CONTRACT_SCHEMA",
    "SHARED_EXCLUSION_AUTHORITY_SCHEMA",
    "canonical_record_commitment_payload",
    "plan_exclusion_authority",
    "validate_candidate_plan_contract",
    "require_agreeing_exclusion_authorities",
    "shared_exclusion_authority",
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
