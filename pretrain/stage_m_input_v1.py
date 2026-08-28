#!/usr/bin/env python3
"""Stage-M input v1: bind the accepted Stage-I publication and consume it natively.

Stage I is closed and accepted; this module treats it as an immutable authority and never
reinterprets its selection. What it does is narrow:

* bind the complete accepted publication -- run identity, Layer-2 digest, manifest digest,
  canonical completion object, the full shard inventory with per-shard path/bytes/records/SHA,
  the accepted totals, record count and two-stage membership -- and revalidate it from disk;
* stream the records in the accepted Stage-I *physical* order, with bounded memory;
* reject a malformed record rather than skipping it.

Ordering is the owner decision ``ACCEPTED_STAGE_I_PHYSICAL_ORDER_V1`` (DECISIONS D-145): each
stage stream is the accepted publication filtered to that stage, with the accepted relative
order preserved exactly. There is no weighted interleave, no selection-rank reorder, no
source-quota scheduling, no hash shuffle and no new random permutation here -- and no code path
that could introduce one, because the only iteration order in this module is the shard
inventory order declared by the accepted manifest.

The text field is ``training_text``. There is deliberately no fallback to ``"text"``: the
legacy packer's silent default is what made a Stage-I shard yield zero documents, and a silent
zero is exactly the failure this module refuses to reproduce.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pretrain.stage_m_contract_v1 import (  # noqa: E402
    STAGE_I_MANIFEST_SCHEMA,
    STAGE_I_RECORD_SCHEMA,
    STAGE_STREAMS,
    InputSequenceCommitment,
    canonical_record_commitment_payload,
    file_sha256,
    require_int,
    validated_sha256,
)

TEXT_FIELD = "training_text"
COMPLETION_FILENAME = "COMPLETE"
MANIFEST_FILENAME = "manifest.json"
DOCUMENTS_DIRNAME = "documents"

# The closed Stage-I record schema, exactly as Stage-I publishes it. An unknown, missing or
# mistyped field is a controlled error, never an ignored one.
RECORD_FIELDS: Mapping[str, type] = MappingProxyType({
    "canonical_fingerprint": str,
    "cleaned_text_sha256": str,
    "content_token_count": int,
    "input_binding_id": str,
    "input_record_sha256": str,
    "raw_sha256": str,
    "schema_version": str,
    "selection_ordinal_within_node": int,
    "serialized_token_count": int,
    "source_id": str,
    "stable_input_record_ordinal": int,
    "stage": str,
    TEXT_FIELD: str,
})

# Stage-I frames every document as [BOS] content [EOS]; the two boundary tokens are the whole
# difference between the two counts, and a record that disagrees is rejected.
BOUNDARY_TOKENS_PER_DOCUMENT = 2


class StageIInputError(RuntimeError):
    """Controlled failure while binding or reading the accepted Stage-I publication."""


def _require(condition: object, message: str) -> None:
    if not condition:
        raise StageIInputError(message)


@dataclass(frozen=True)
class AcceptedStageI:
    """A fully revalidated binding to one accepted Stage-I publication."""

    run_dir: Path
    manifest_path: Path
    manifest_sha256: str
    completion_path: Path
    completion_sha256: str
    run_identity: str
    layer2_sha256: str
    record_schema_version: str
    manifest_schema_version: str
    records_per_shard: int
    shard_inventory: tuple[Mapping[str, Any], ...]
    total_records: int
    total_serialized_tokens: int
    total_content_tokens: int
    stage_membership: Mapping[str, Mapping[str, int]]
    shard_bytes_verified: bool

    @property
    def shard_count(self) -> int:
        return len(self.shard_inventory)

    def as_canonical(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir.as_posix(),
            "run_identity": self.run_identity,
            "layer2_expected_result_sha256": self.layer2_sha256,
            "manifest_sha256": self.manifest_sha256,
            "completion_object_sha256": self.completion_sha256,
            "completion_object_kind": "stage_i:COMPLETE",
            "record_schema_version": self.record_schema_version,
            "manifest_schema_version": self.manifest_schema_version,
            "records_per_shard": self.records_per_shard,
            "shard_count": self.shard_count,
            "shard_inventory": [dict(entry) for entry in self.shard_inventory],
            "total_records": self.total_records,
            "total_serialized_tokens": self.total_serialized_tokens,
            "total_content_tokens": self.total_content_tokens,
            "stage_membership": {
                stage: dict(counts) for stage, counts in sorted(self.stage_membership.items())
            },
        }

    def require_physically_verified(self, context: str) -> None:
        """Refuse to let an unverified binding reach a production path.

        ``shard_bytes_verified`` is only ever True when every declared shard was hashed from
        disk. The diagnostic loader cannot set it, so a diagnostic binding cannot derive an
        authorizable plan, resolve an authorization, consume input for publication, or feed the
        Stage-P native chain.
        """
        _require(
            self.shard_bytes_verified,
            f"{context}: refusing an accepted Stage-I binding whose {self.shard_count} shard "
            "files were not physically verified against their authoritative SHA-256 digests",
        )

    def identity_sha256(self) -> str:
        """One digest over the whole accepted binding, inventory and digests included."""
        from pretrain.stage_m_contract_v1 import canonical_sha256

        return canonical_sha256(self.as_canonical())


def _read_json(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    _require(not path.is_symlink(), f"{label} must not be a symlink: {path}")
    _require(path.is_file(), f"{label} is missing: {path}")
    digest = file_sha256(path)
    with open(path, encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise StageIInputError(f"{label} is not valid JSON: {path}: {exc}") from exc
    _require(isinstance(payload, dict), f"{label} must be a JSON object: {path}")
    return payload, digest


def load_accepted_stage_i(
    run_dir: Path,
    *,
    expected_run_identity: str | None = None,
    expected_manifest_sha256: str | None = None,
    expected_completion_sha256: str | None = None,
    expected_layer2_sha256: str | None = None,
    expected_records: int | None = None,
    expected_serialized_tokens: int | None = None,
    expected_shard_count: int | None = None,
) -> AcceptedStageI:
    """Bind an accepted Stage-I publication, hashing every declared shard file from disk.

    D-147 requires the whole publication to be bound, not only its top-level hashes, so this
    function always re-reads and re-hashes all declared shards. There is deliberately no
    ``verify_shard_bytes`` switch: a boolean whose default is False is exactly how physical
    validation silently stops happening on the production path.

    Every expectation the caller supplies is checked against freshly read bytes. Passing none
    of them still validates internal consistency: the completion object must agree with the
    manifest it names, the inventory must be exactly the files present, the declared totals
    must equal the summed inventory, and every shard's bytes must hash to its manifest digest.

    Use :func:`inspect_accepted_stage_i_metadata_only` for cheap diagnostics; the binding it
    returns cannot reach any production path.
    """
    return _load_accepted_stage_i(
        run_dir,
        expected_run_identity=expected_run_identity,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_completion_sha256=expected_completion_sha256,
        expected_layer2_sha256=expected_layer2_sha256,
        expected_records=expected_records,
        expected_serialized_tokens=expected_serialized_tokens,
        expected_shard_count=expected_shard_count,
        verify_shard_bytes=True,
    )


def inspect_accepted_stage_i_metadata_only(
    run_dir: Path,
    *,
    expected_run_identity: str | None = None,
    expected_manifest_sha256: str | None = None,
    expected_completion_sha256: str | None = None,
    expected_layer2_sha256: str | None = None,
    expected_records: int | None = None,
    expected_serialized_tokens: int | None = None,
    expected_shard_count: int | None = None,
) -> AcceptedStageI:
    """Diagnostic-only binding that skips per-shard hashing.

    Everything except the shard-byte digests is still checked. The result carries
    ``shard_bytes_verified=False``, and every production entry point calls
    :meth:`AcceptedStageI.require_physically_verified`, so this binding cannot derive or
    authorize a candidate plan, publish Stage-M output, or yield ``full_chain_validated``.
    """
    return _load_accepted_stage_i(
        run_dir,
        expected_run_identity=expected_run_identity,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_completion_sha256=expected_completion_sha256,
        expected_layer2_sha256=expected_layer2_sha256,
        expected_records=expected_records,
        expected_serialized_tokens=expected_serialized_tokens,
        expected_shard_count=expected_shard_count,
        verify_shard_bytes=False,
    )


def _load_accepted_stage_i(
    run_dir: Path,
    *,
    expected_run_identity: str | None = None,
    expected_manifest_sha256: str | None = None,
    expected_completion_sha256: str | None = None,
    expected_layer2_sha256: str | None = None,
    expected_records: int | None = None,
    expected_serialized_tokens: int | None = None,
    expected_shard_count: int | None = None,
    verify_shard_bytes: bool,
) -> AcceptedStageI:
    """Shared implementation. Private: the two public spellings above are the whole API."""
    run_dir = Path(run_dir).expanduser().resolve()
    _require(run_dir.is_dir(), f"accepted Stage-I run directory is missing: {run_dir}")

    manifest_path = run_dir / MANIFEST_FILENAME
    completion_path = run_dir / COMPLETION_FILENAME
    manifest, manifest_sha256 = _read_json(manifest_path, label="Stage-I manifest")
    completion, completion_sha256 = _read_json(completion_path, label="Stage-I COMPLETE")

    _require(
        completion.get("marker") == "COMPLETE",
        f"Stage-I completion object is not a COMPLETE marker: {completion_path}",
    )
    _require(
        completion.get("manifest_sha256") == manifest_sha256,
        "Stage-I COMPLETE does not bind the manifest actually on disk: "
        f"complete={completion.get('manifest_sha256')!r}, actual={manifest_sha256}",
    )
    _require(
        manifest.get("schema_version") == STAGE_I_MANIFEST_SCHEMA,
        f"unexpected Stage-I manifest schema: {manifest.get('schema_version')!r}",
    )
    _require(
        manifest.get("record_schema_version") == STAGE_I_RECORD_SCHEMA,
        f"unexpected Stage-I record schema: {manifest.get('record_schema_version')!r}",
    )

    run_block = manifest.get("stage_i_run")
    _require(isinstance(run_block, dict), "Stage-I manifest has no stage_i_run object")
    assert isinstance(run_block, dict)
    run_identity = validated_sha256(run_block.get("run_identity"), field="stage_i_run.run_identity")
    layer2 = validated_sha256(
        run_block.get("post_pass1_result_identity_sha256"),
        field="stage_i_run.post_pass1_result_identity_sha256",
    )
    _require(
        completion.get("stage_i_run_identity") == run_identity,
        "Stage-I COMPLETE run identity disagrees with the manifest",
    )

    totals = manifest.get("totals")
    _require(isinstance(totals, dict), "Stage-I manifest has no totals object")
    assert isinstance(totals, dict)
    total_records = require_int(totals.get("records"), field="totals.records", minimum=1)
    total_serialized = require_int(
        totals.get("serialized_tokens"), field="totals.serialized_tokens", minimum=1
    )
    total_content = require_int(
        totals.get("content_tokens"), field="totals.content_tokens", minimum=1
    )
    _require(
        total_content + BOUNDARY_TOKENS_PER_DOCUMENT * total_records == total_serialized,
        "Stage-I totals are internally inconsistent: "
        f"content {total_content} + 2*{total_records} != serialized {total_serialized}",
    )

    records_per_shard = require_int(
        manifest.get("records_per_shard"), field="records_per_shard", minimum=1
    )
    raw_shards = manifest.get("shards")
    _require(
        isinstance(raw_shards, list) and bool(raw_shards),
        "Stage-I manifest has no shard inventory",
    )
    assert isinstance(raw_shards, list)

    documents_dir = run_dir / DOCUMENTS_DIRNAME
    _require(documents_dir.is_dir(), f"Stage-I documents directory is missing: {documents_dir}")

    inventory: list[Mapping[str, Any]] = []
    summed_records = 0
    for index, raw in enumerate(raw_shards):
        label = f"Stage-I manifest.shards[{index}]"
        _require(isinstance(raw, dict), f"{label} must be an object")
        assert isinstance(raw, dict)
        name = raw.get("name")
        _require(
            isinstance(name, str) and name and "/" not in name and not name.startswith("."),
            f"{label}.name is invalid: {name!r}",
        )
        assert isinstance(name, str)
        records = require_int(raw.get("records"), field=f"{label}.records", minimum=1)
        size_bytes = require_int(raw.get("bytes"), field=f"{label}.bytes", minimum=1)
        digest = validated_sha256(raw.get("sha256"), field=f"{label}.sha256")

        path = documents_dir / name
        _require(not path.is_symlink(), f"{label} must be a regular non-symlink file: {path}")
        _require(path.is_file(), f"{label} names a missing shard: {path}")
        actual_bytes = int(path.stat().st_size)
        _require(
            actual_bytes == size_bytes,
            f"{label} byte-size mismatch: actual={actual_bytes}, manifest={size_bytes}",
        )
        if verify_shard_bytes:
            actual_sha = file_sha256(path)
            _require(
                actual_sha == digest,
                f"{label} SHA-256 mismatch: actual={actual_sha}, manifest={digest}",
            )
        summed_records += records
        inventory.append(
            MappingProxyType({
                "name": name,
                "records": records,
                "bytes": size_bytes,
                "sha256": digest,
            })
        )

    _require(
        summed_records == total_records,
        f"Stage-I inventory record sum {summed_records} != declared totals.records {total_records}",
    )
    on_disk = sorted(entry.name for entry in documents_dir.iterdir())
    declared = sorted(str(entry["name"]) for entry in inventory)
    _require(
        on_disk == declared,
        "Stage-I documents directory does not match the declared inventory: "
        f"extra={sorted(set(on_disk) - set(declared))[:4]}, "
        f"missing={sorted(set(declared) - set(on_disk))[:4]}",
    )

    membership: dict[str, dict[str, int]] = {}
    for node in manifest.get("nodes") or []:
        _require(isinstance(node, dict), "Stage-I manifest.nodes entries must be objects")
        assert isinstance(node, dict)
        stage = node.get("stage")
        _require(stage in STAGE_STREAMS, f"Stage-I node has an unknown stage: {stage!r}")
        assert isinstance(stage, str)
        bucket = membership.setdefault(stage, {"records": 0, "serialized_tokens": 0})
        bucket["records"] += require_int(
            node.get("selected_identities"), field="node.selected_identities", minimum=1
        )
        bucket["serialized_tokens"] += require_int(
            node.get("selected_serialized_tokens"),
            field="node.selected_serialized_tokens",
            minimum=1,
        )
    _require(
        sorted(membership) == sorted(STAGE_STREAMS),
        f"accepted Stage-I must declare exactly {list(STAGE_STREAMS)}, got {sorted(membership)}",
    )
    _require(
        sum(v["records"] for v in membership.values()) == total_records,
        "Stage-I per-stage record counts do not sum to the accepted record count",
    )
    _require(
        sum(v["serialized_tokens"] for v in membership.values()) == total_serialized,
        "Stage-I per-stage serialized tokens do not sum to the accepted total",
    )

    if expected_run_identity is not None:
        _require(
            run_identity == expected_run_identity,
            f"accepted Stage-I run identity mismatch: actual={run_identity}, "
            f"expected={expected_run_identity}",
        )
    if expected_layer2_sha256 is not None:
        _require(
            layer2 == expected_layer2_sha256,
            f"accepted Stage-I Layer-2 digest mismatch: actual={layer2}, "
            f"expected={expected_layer2_sha256}",
        )
    if expected_manifest_sha256 is not None:
        _require(
            manifest_sha256 == expected_manifest_sha256,
            f"accepted Stage-I manifest digest mismatch: actual={manifest_sha256}, "
            f"expected={expected_manifest_sha256}",
        )
    if expected_completion_sha256 is not None:
        _require(
            completion_sha256 == expected_completion_sha256,
            f"accepted Stage-I completion digest mismatch: actual={completion_sha256}, "
            f"expected={expected_completion_sha256}",
        )
    if expected_records is not None:
        _require(
            total_records == expected_records,
            f"accepted Stage-I record count mismatch: actual={total_records}, "
            f"expected={expected_records}",
        )
    if expected_serialized_tokens is not None:
        _require(
            total_serialized == expected_serialized_tokens,
            f"accepted Stage-I serialized-token mismatch: actual={total_serialized}, "
            f"expected={expected_serialized_tokens}",
        )
    if expected_shard_count is not None:
        _require(
            len(inventory) == expected_shard_count,
            f"accepted Stage-I shard-count mismatch: actual={len(inventory)}, "
            f"expected={expected_shard_count}",
        )

    return AcceptedStageI(
        run_dir=run_dir,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        completion_path=completion_path,
        completion_sha256=completion_sha256,
        run_identity=run_identity,
        layer2_sha256=layer2,
        record_schema_version=STAGE_I_RECORD_SCHEMA,
        manifest_schema_version=str(manifest.get("schema_version")),
        records_per_shard=records_per_shard,
        shard_inventory=tuple(inventory),
        total_records=total_records,
        total_serialized_tokens=total_serialized,
        total_content_tokens=total_content,
        stage_membership=MappingProxyType({
            stage: MappingProxyType(dict(counts)) for stage, counts in membership.items()
        }),
        shard_bytes_verified=bool(verify_shard_bytes),
    )


def validate_record(raw: object, *, label: str) -> dict[str, Any]:
    """Validate one Stage-I record against the closed schema. Malformed is an error, not a skip."""
    _require(isinstance(raw, dict), f"{label} must be a JSON object")
    assert isinstance(raw, dict)
    actual = set(raw)
    expected = set(RECORD_FIELDS)
    _require(
        actual == expected,
        f"{label} field set mismatch: unexpected={sorted(actual - expected)}, "
        f"missing={sorted(expected - actual)}",
    )
    for field, kind in RECORD_FIELDS.items():
        value = raw[field]
        if kind is int:
            _require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                f"{label}.{field} must be a non-negative int, got {value!r}",
            )
        else:
            _require(isinstance(value, str), f"{label}.{field} must be a string")
    _require(
        raw["schema_version"] == STAGE_I_RECORD_SCHEMA,
        f"{label}.schema_version must be {STAGE_I_RECORD_SCHEMA!r}",
    )
    _require(raw["stage"] in STAGE_STREAMS, f"{label}.stage is unknown: {raw['stage']!r}")
    _require(raw[TEXT_FIELD] != "", f"{label}.{TEXT_FIELD} must be non-empty")
    _require(
        raw["serialized_token_count"] == raw["content_token_count"] + BOUNDARY_TOKENS_PER_DOCUMENT,
        f"{label} token accounting is inconsistent: serialized "
        f"{raw['serialized_token_count']} != content {raw['content_token_count']} + 2",
    )
    for field in (
        "canonical_fingerprint",
        "cleaned_text_sha256",
        "raw_sha256",
        "input_record_sha256",
    ):
        validated_sha256(raw[field], field=f"{label}.{field}")
    return raw


def iter_accepted_records(
    accepted: AcceptedStageI,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Stream ``(stage, record)`` in the accepted Stage-I physical order, bounded memory.

    The iteration order is the manifest's declared shard inventory order and, within a shard,
    file order. That *is* the accepted physical order, so per-stage relative order follows by
    construction: filtering a sequence never reorders what it keeps.

    Every shard is re-hashed while it is read and compared to its manifest digest at end of
    file. Hashing at bind time proves what was there when the binding was made; hashing again
    while consuming proves the bytes actually consumed are those same bytes, which is what
    catches a replacement between authorization and use.
    """
    import hashlib

    accepted.require_physically_verified("streaming accepted Stage-I records")
    documents_dir = accepted.run_dir / DOCUMENTS_DIRNAME
    seen_records = 0
    for entry in accepted.shard_inventory:
        name = str(entry["name"])
        path = documents_dir / name
        digest = hashlib.sha256()
        shard_records = 0
        with open(path, "rb") as handle:
            for line_no, raw_line in enumerate(handle):
                digest.update(raw_line)
                stripped = raw_line.strip()
                if not stripped:
                    raise StageIInputError(f"{name}:{line_no + 1}: blank line in a Stage-I shard")
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise StageIInputError(f"{name}:{line_no + 1}: invalid JSON: {exc}") from exc
                record = validate_record(parsed, label=f"{name}:{line_no + 1}")
                shard_records += 1
                seen_records += 1
                yield str(record["stage"]), record
        if shard_records != int(entry["records"]):
            raise StageIInputError(
                f"{name}: record count mismatch: actual={shard_records}, "
                f"manifest={entry['records']}"
            )
        if digest.hexdigest() != str(entry["sha256"]):
            raise StageIInputError(
                f"{name}: shard bytes do not match the accepted manifest digest during the "
                "streaming read"
            )
    if seen_records != accepted.total_records:
        raise StageIInputError(
            f"streamed {seen_records} records, accepted publication declares "
            f"{accepted.total_records}"
        )


def derive_input_sequence_commitments(
    accepted: AcceptedStageI,
) -> dict[str, InputSequenceCommitment]:
    """One bounded-memory streaming pass producing both per-stage commitments.

    Each record contributes to exactly one stream, at its consumption ordinal in that stream.
    The per-stage totals are then proved against the accepted publication's own per-stage
    membership, so a commitment can never be sealed over a stream that silently lost or gained
    records relative to what Stage I published.
    """
    accepted.require_physically_verified("deriving input sequence commitments")
    commitments = {stage: InputSequenceCommitment(stage) for stage in STAGE_STREAMS}
    ordinals = dict.fromkeys(STAGE_STREAMS, 0)

    for stage, record in iter_accepted_records(accepted):
        ordinal = ordinals[stage]
        payload = canonical_record_commitment_payload(
            stage=stage,
            ordinal=ordinal,
            source_id=str(record["source_id"]),
            input_binding_id=str(record["input_binding_id"]),
            stable_input_record_ordinal=int(record["stable_input_record_ordinal"]),
            canonical_fingerprint=str(record["canonical_fingerprint"]),
            cleaned_text_sha256=str(record["cleaned_text_sha256"]),
            raw_sha256=str(record["raw_sha256"]),
            input_record_sha256=str(record["input_record_sha256"]),
            selection_ordinal_within_node=int(record["selection_ordinal_within_node"]),
            content_token_count=int(record["content_token_count"]),
            serialized_token_count=int(record["serialized_token_count"]),
        )
        commitments[stage].update(
            payload,
            serialized_token_count=int(record["serialized_token_count"]),
            content_token_count=int(record["content_token_count"]),
        )
        ordinals[stage] = ordinal + 1

    for stage, commitment in commitments.items():
        declared = accepted.stage_membership[stage]
        _require(
            commitment.record_count == declared["records"],
            f"{stage}: streamed {commitment.record_count} records, accepted publication "
            f"declares {declared['records']}",
        )
        _require(
            commitment.serialized_tokens == declared["serialized_tokens"],
            f"{stage}: streamed {commitment.serialized_tokens} serialized tokens, accepted "
            f"publication declares {declared['serialized_tokens']}",
        )
        commitment.seal()

    total = sum(c.record_count for c in commitments.values())
    _require(
        total == accepted.total_records,
        f"per-stage record counts sum to {total}, accepted publication declares "
        f"{accepted.total_records}",
    )
    return commitments


def commitments_as_canonical(
    commitments: Mapping[str, InputSequenceCommitment],
) -> dict[str, Any]:
    return {stage: commitments[stage].as_canonical() for stage in sorted(commitments)}


__all__ = [
    "BOUNDARY_TOKENS_PER_DOCUMENT",
    "COMPLETION_FILENAME",
    "DOCUMENTS_DIRNAME",
    "MANIFEST_FILENAME",
    "RECORD_FIELDS",
    "TEXT_FIELD",
    "AcceptedStageI",
    "StageIInputError",
    "commitments_as_canonical",
    "derive_input_sequence_commitments",
    "inspect_accepted_stage_i_metadata_only",
    "iter_accepted_records",
    "load_accepted_stage_i",
    "validate_record",
]
