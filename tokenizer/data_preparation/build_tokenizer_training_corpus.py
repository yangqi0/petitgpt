#!/usr/bin/env python3
"""Build the frozen PetitGPT Stage-F tokenizer-training corpus.

The production CLI exposes only an output directory and worker count. Owner-frozen
semantics are constants, not runtime knobs.

There are intentionally two text views. The historical L1 view applies
text.strip("\n\r") only to test membership in the frozen exclusion identity set.
The F payload view is the decoded source JSON text unchanged. F hashes, ranks,
accounts, and publishes that exact tokenizer-visible UTF-8 text.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import struct
import sys
import time
from typing import Any

import numpy as np

try:
    import ujson
except ImportError:  # pragma: no cover
    ujson = None


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

SCHEMA_VERSION = "petitgpt-f-tokenizer-corpus-v1"
ELIGIBILITY_KIND = "petitgpt_reference_reserve_eligibility"
EXCLUSION_KIND = "petitgpt_reference_validation_exclusions"
EXCLUSION_ALGORITHM = "sha256-cleaned-text-utf8-v1"
RANK_ALGORITHM = "BLAKE2b-128"
RANK_PERSON = b"PetitGPT-F-v1"
RANK_SEED_ASCII = b"20250814"
OUTPUT_FIELDS = (
    "text",
    "canonical_source",
    "canonical_release_id",
    "physical_row_index",
    "cleaned_text_sha256",
)
TEXT_START = b'"text":"'
TEXT_END = b'","text_bytes":'
SPOOL_HEADER = struct.Struct(">16s32sBQQQ")
SELECTION_RECORD = struct.Struct(">16s32sBQQ")
SELECTION_DTYPE = np.dtype(
    [
        ("rank", "V16"),
        ("sha", "V32"),
        ("release_ordinal", "u1"),
        ("physical_row_index", ">u8"),
        ("cleaned_utf8_bytes", ">u8"),
    ],
    align=False,
)
assert SPOOL_HEADER.size == 73
assert SELECTION_RECORD.size == 65
assert SELECTION_DTYPE.itemsize == 65

EXPECTED_L1_CLEANING = {
    "strip_leading_noise": False,
    "normalize_quotes": False,
    "underscores_policy": "keep",
    "min_chars": 0,
    "min_ascii_ratio": 0.0,
}


@dataclass(frozen=True)
class BucketSpec:
    canonical_name: str
    slug: str
    target_bytes: int
    member_release_ids: tuple[str, ...]


@dataclass(frozen=True)
class FileBinding:
    role: str
    path: Path
    sha256: str
    size_bytes: int | None = None


@dataclass(frozen=True)
class ReleaseSpec:
    canonical_release_id: str
    bucket_name: str
    source_path: Path
    documents_sha256: str
    documents_file_bytes: int
    physical_rows: int
    eligible_rows: int
    excluded_rows: int
    excluded_rows_path: Path
    excluded_rows_sha256: str
    release_manifest_path: Path | None = None
    release_manifest_sha256: str | None = None
    aux_path: Path | None = None
    aux_sha256: str | None = None
    expected_l1_matches: int | None = None
    expected_post_l1_occurrences: int | None = None
    expected_post_l1_utf8_bytes: int | None = None


@dataclass(frozen=True)
class BuildContract:
    buckets: tuple[BucketSpec, ...]
    releases: tuple[ReleaseSpec, ...]
    l1_exclusion_manifest_path: Path
    l1_exclusion_manifest_sha256: str
    l1_exclusion_hash_count: int
    static_bindings: tuple[FileBinding, ...] = ()
    expected_pool_post_l1_occurrences: int | None = None
    expected_pool_post_l1_utf8_bytes: int | None = None
    authoritative_global_post_l1_occurrences: int | None = None
    frozen_stage_e_releases: tuple[tuple[str, str], ...] = ()
    l1_checkpoint_commit: str | None = None


PRODUCTION_BUCKETS = (
    BucketSpec("FineWeb", "fineweb", 5_000_000_000, ("fineweb_edu_dedup",)),
    BucketSpec("DCLM", "dclm", 2_000_000_000, ("dclm_edu",)),
    BucketSpec("Wikipedia", "wikipedia", 1_000_000_000, ("finewiki_en",)),
    BucketSpec("Python", "python", 1_500_000_000, ("python_gate_c_full",)),
    BucketSpec(
        "structured/tutorial",
        "structured_tutorial",
        500_000_000,
        ("cosmopedia_v2", "finephrase_tutorial"),
    ),
)

STAGE_E_PATH = PROJECT_ROOT / "runs/stage_e_2026-08-20/allocation_contract.json"
STAGE_E_SHA256 = "b9fbfd7484c5f21a2b68e1e4913b6ea8c0c5ab10cc28014c60b8bd32a22da05b"
ELIGIBILITY_PATH = (
    PROJECT_ROOT / "runs/l1_production_2026-08-20/eligibility/eligibility_manifest.json"
)
ELIGIBILITY_SHA256 = "e3e5e8bfc0cc18524254ed00196f03ad7af3ae72617489e83713ff883b765dce"
L1_EXCLUSION_PATH = (
    PROJECT_ROOT / "runs/l1_production_2026-08-20/reference_reserve_v1/exclusion_hash_manifest.json"
)
L1_EXCLUSION_SHA256 = "7e768eb992456cca9b7ba64dd6fda0410f87843faff72d573027139a917e1dd4"
L1_CLEANER_PATH = PROJECT_ROOT / "pretrain/build_pretrain_shards.py"
L1_CLEANER_SHA256 = "710cf55a8dc4eec05470f43b2ff2c4fd966dda619ff333e45368b6f6e660d56d"
F_PREFLIGHT_CAPACITY_PATH = (
    PROJECT_ROOT / "runs/f_preflight_2026-08-21/evidence/capacity_reproduction.json"
)
F_PREFLIGHT_CAPACITY_SHA256 = "390662914278ebf4e6c4aa688588a07d3e84101b7bcc2d713e9d8e42dbfa4eb6"

PRODUCTION_RELEASES: dict[str, dict[str, Any]] = {
    "cosmopedia_v2": {
        "bucket": "structured/tutorial",
        "source": "runs/nonpython_gate_c_production_2026-08-17/"
        "cosmopedia_v2_11files/release/documents.jsonl",
        "source_size": 16_895_732_499,
        "documents_sha256": "fbb5b7aed9289b6d5de541ede7beb766e4a829d01e3fdf013dd94d5eb079e960",
        "release_manifest": "runs/nonpython_gate_c_production_2026-08-17/"
        "cosmopedia_v2_11files/release/manifest.json",
        "release_manifest_sha256": "d4355dde603b31e7d97d203574635e806ea3add8130583ec880e2812093aaf8c",
        "aux": "runs/d2_production_2026-08-19/d2_1_signatures/aux.cosmopedia_v2.npy",
        "aux_sha256": "7239750f1b9d85a776618bd64dab85ebd8a3692871ab543271bda1bc248b6839",
        "physical_rows": 3_545_224,
        "eligible_rows": 3_541_290,
        "excluded_rows": 3_934,
        "l1_matches": 13_538,
        "post_l1_occurrences": 3_527_752,
        "post_l1_utf8_bytes": 13_332_243_908,
    },
    "dclm_edu": {
        "bucket": "DCLM",
        "source": "runs/nonpython_gate_c_production_2026-08-17/"
        "dclm_edu_14files/release/documents.jsonl",
        "source_size": 22_492_715_741,
        "documents_sha256": "80b491f927d93ca2176563a654c926bc12f8457a8d6ee4cbb2b5716c8125662b",
        "release_manifest": "runs/nonpython_gate_c_production_2026-08-17/"
        "dclm_edu_14files/release/manifest.json",
        "release_manifest_sha256": "2c20d30328367d18e1f5cdb359c2bb3964871be07df480085e4d9de1245b7f9e",
        "aux": "runs/d2_production_2026-08-19/d2_1_signatures/aux.dclm_edu.npy",
        "aux_sha256": "4e42ffeeface8d00a9195978ce098d0cc196f1ffb4643dfe26bef4cfc99a7d77",
        "physical_rows": 3_291_114,
        "eligible_rows": 3_187_368,
        "excluded_rows": 103_746,
        "l1_matches": 11_514,
        "post_l1_occurrences": 3_175_854,
        "post_l1_utf8_bytes": 17_882_807_219,
    },
    "finephrase_tutorial": {
        "bucket": "structured/tutorial",
        "source": "runs/nonpython_gate_c_production_2026-08-17/"
        "finephrase_tutorial_28files/release/documents.jsonl",
        "source_size": 5_355_647_008,
        "documents_sha256": "49b22f305aab1398ad9969171f5a7c0ae6bbae02ac543a59cacb9693961f4cac",
        "release_manifest": "runs/nonpython_gate_c_production_2026-08-17/"
        "finephrase_tutorial_28files/release/manifest.json",
        "release_manifest_sha256": "23d27ba2d4e0c1a5f178d197d9a013918af80fb308fbeeb8f5f826e5dac5e1bb",
        "aux": "runs/d2_production_2026-08-19/d2_1_signatures/aux.finephrase_tutorial.npy",
        "aux_sha256": "c38001693d056f1dd7ed84c3f2e6331b26c24b862dc2d31f1bbc0be6d28f8243",
        "physical_rows": 1_589_946,
        "eligible_rows": 1_589_945,
        "excluded_rows": 1,
        "l1_matches": 6_124,
        "post_l1_occurrences": 1_583_821,
        "post_l1_utf8_bytes": 3_251_869_104,
    },
    "fineweb_edu_dedup": {
        "bucket": "FineWeb",
        "source": "runs/nonpython_gate_c_production_2026-08-17/"
        "fineweb_edu_dedup_14files/release/documents.jsonl",
        "source_size": 64_410_467_081,
        "documents_sha256": "c5bc4f688e7e1170452deda83a1f91e577d0b49c3d2074877d63d810100c630f",
        "release_manifest": "runs/nonpython_gate_c_production_2026-08-17/"
        "fineweb_edu_dedup_14files/release/manifest.json",
        "release_manifest_sha256": "857099ba628b63de5c110cd6ba330cfda527814c60332707ee1b4154bed5f163",
        "aux": "runs/d2_production_2026-08-19/d2_1_signatures/aux.fineweb_edu_dedup.npy",
        "aux_sha256": "4aea143de2dc4a1a2ae4933cde3ca3f7a055074a1e7d15c12705a3e8fef3357d",
        "physical_rows": 11_310_174,
        "eligible_rows": 11_078_026,
        "excluded_rows": 232_148,
        "l1_matches": 14_524,
        "post_l1_occurrences": 11_063_502,
        "post_l1_utf8_bytes": 49_051_634_551,
    },
    "finewiki_en": {
        "bucket": "Wikipedia",
        "source": "runs/nonpython_gate_c_production_2026-08-17/"
        "finewiki_en_7files/release/documents.jsonl",
        "source_size": 12_927_403_994,
        "documents_sha256": "743e6eeade3f594097a312ecec2cf0778b9fc64cced421f8da2843f51699a134",
        "release_manifest": "runs/nonpython_gate_c_production_2026-08-17/"
        "finewiki_en_7files/release/manifest.json",
        "release_manifest_sha256": "315a5df7f114b7457a94280df11a8e5ea2a0084a573c2a12ee6122d042a4c232",
        "aux": "runs/d2_production_2026-08-19/d2_1_signatures/aux.finewiki_en.npy",
        "aux_sha256": "55322f8723efcb0d10bb725fd87dc461be89ddec143bfe8f0ccb8886ecdfb473",
        "physical_rows": 2_631_702,
        "eligible_rows": 2_631_467,
        "excluded_rows": 235,
        "l1_matches": 16_910,
        "post_l1_occurrences": 2_614_557,
        "post_l1_utf8_bytes": 9_986_627_334,
    },
    "python_gate_c_full": {
        "bucket": "Python",
        "source": "runs/python_gate_c_full_2026-08-17/release/documents.jsonl",
        "source_size": 21_698_689_238,
        "documents_sha256": "8fffeee998bb1f0e5e7b85cf609c612e614cba0de05f1bf60fba87ff2c71d997",
        "release_manifest": "runs/python_gate_c_full_2026-08-17/release/manifest.json",
        "release_manifest_sha256": "365053b12704e2cf94ca7540ee9cfaa7e200a7456a924b170e615d876ac088b6",
        "aux": None,
        "aux_sha256": None,
        "physical_rows": 5_470_037,
        "eligible_rows": 5_469_783,
        "excluded_rows": 254,
        "l1_matches": 26_243,
        "post_l1_occurrences": 5_443_540,
        "post_l1_utf8_bytes": 13_303_867_921,
    },
}

_WORKER_L1_HASHES: frozenset[bytes] = frozenset()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def f_payload(text: str) -> tuple[bytes, int]:
    encoded = text.encode("utf-8")
    return hashlib.sha256(encoded).digest(), len(encoded)


def l1_identity_digest(text: str) -> bytes:
    identity_text = text.strip("\n\r")
    if not identity_text:
        raise ValueError("eligible text is empty under frozen L1 identity cleaning")
    return hashlib.sha256(identity_text.encode("utf-8")).digest()


def rank_digest(bucket_name: str, cleaned_text_sha256: bytes) -> bytes:
    if len(cleaned_text_sha256) != 32:
        raise ValueError("cleaned_text_sha256 must be exactly 32 raw bytes")
    message = RANK_SEED_ASCII + b"\0" + bucket_name.encode("utf-8") + b"\0" + cleaned_text_sha256
    return hashlib.blake2b(message, digest_size=16, person=RANK_PERSON).digest()


def encode_output_row(
    text: str,
    bucket_name: str,
    release_id: str,
    row_index: int,
    cleaned_sha: bytes,
) -> bytes:
    row = {
        "text": text,
        "canonical_source": bucket_name,
        "canonical_release_id": release_id,
        "physical_row_index": row_index,
        "cleaned_text_sha256": cleaned_sha.hex(),
    }
    return (
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def load_json_object(line: bytes, context: str) -> dict[str, Any]:
    try:
        value = ujson.loads(line) if ujson is not None else json.loads(line)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{context}: JSON row is not an object")
    return value


def load_text(line: bytes, context: str) -> str:
    text = load_json_object(line, context).get("text")
    if not isinstance(text, str):
        raise TypeError(f"{context}: text is not a string")
    return text


def expect_file(binding: FileBinding) -> dict[str, Any]:
    path = binding.path
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{binding.role}: missing regular file: {path}")
    size = path.stat().st_size
    if binding.size_bytes is not None and size != binding.size_bytes:
        raise RuntimeError(
            f"{binding.role}: size mismatch: expected {binding.size_bytes}, got {size}"
        )
    actual = sha256_file(path)
    if actual != binding.sha256:
        raise RuntimeError(
            f"{binding.role}: SHA-256 mismatch: expected {binding.sha256}, got {actual}"
        )
    return {"path": str(path), "sha256": actual, "size_bytes": size}


def validate_contract_shape(contract: BuildContract) -> dict[str, dict[str, int]]:
    if not contract.buckets or not contract.releases:
        raise ValueError("contract requires buckets and releases")
    names = [item.canonical_name for item in contract.buckets]
    slugs = [item.slug for item in contract.buckets]
    if len(names) != len(set(names)) or len(slugs) != len(set(slugs)):
        raise ValueError("bucket names and slugs must be unique")
    if any(item.target_bytes <= 0 for item in contract.buckets):
        raise ValueError("bucket targets must be positive")
    release_ids = [item.canonical_release_id for item in contract.releases]
    if len(release_ids) != len(set(release_ids)):
        raise ValueError("release IDs must be unique")

    release_set = set(release_ids)
    ordinals: dict[str, dict[str, int]] = {}
    claimed: list[str] = []
    by_release = {item.canonical_release_id: item for item in contract.releases}
    for bucket in contract.buckets:
        if bucket.member_release_ids != tuple(sorted(bucket.member_release_ids)):
            raise ValueError(f"{bucket.canonical_name}: member releases must be lexical")
        if not bucket.member_release_ids or not set(bucket.member_release_ids) <= release_set:
            raise ValueError(f"{bucket.canonical_name}: unknown member release")
        ordinals[bucket.canonical_name] = {
            release_id: index for index, release_id in enumerate(bucket.member_release_ids)
        }
        claimed.extend(bucket.member_release_ids)
        for release_id in bucket.member_release_ids:
            if by_release[release_id].bucket_name != bucket.canonical_name:
                raise ValueError(f"{release_id}: bucket mismatch")
    if sorted(claimed) != sorted(release_ids):
        raise ValueError("each release must belong to exactly one bucket")
    return ordinals


def load_l1_exclusions(contract: BuildContract) -> frozenset[bytes]:
    expect_file(
        FileBinding(
            "L1 exclusion manifest",
            contract.l1_exclusion_manifest_path,
            contract.l1_exclusion_manifest_sha256,
        )
    )
    value = json.loads(contract.l1_exclusion_manifest_path.read_text(encoding="utf-8"))
    if value.get("kind") != EXCLUSION_KIND:
        raise ValueError("unexpected L1 exclusion manifest kind")
    if value.get("hash_algorithm") != EXCLUSION_ALGORITHM:
        raise ValueError("unexpected L1 exclusion algorithm")
    if value.get("membership_basis") != "cleaned document text encoded as UTF-8":
        raise ValueError("unexpected L1 membership basis")
    if value.get("cleaning") != EXPECTED_L1_CLEANING:
        raise ValueError("unexpected frozen L1 cleaning object")
    hashes = value.get("hashes")
    if not isinstance(hashes, list) or value.get("hash_count") != len(hashes):
        raise ValueError("L1 hash_count mismatch")
    if len(hashes) != contract.l1_exclusion_hash_count:
        raise ValueError("L1 hash count differs from contract")
    if hashes != sorted(hashes) or len(hashes) != len(set(hashes)):
        raise ValueError("L1 hashes must be sorted and unique")
    try:
        decoded = tuple(bytes.fromhex(item) for item in hashes)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid L1 SHA-256 identity") from exc
    if any(len(item) != 32 for item in decoded):
        raise ValueError("invalid L1 identity width")
    return frozenset(decoded)


def validate_release_control(spec: ReleaseSpec) -> dict[str, Any]:
    if not spec.source_path.is_file() or spec.source_path.is_symlink():
        raise FileNotFoundError(f"{spec.canonical_release_id}: source is missing")
    if spec.source_path.stat().st_size != spec.documents_file_bytes:
        raise RuntimeError(f"{spec.canonical_release_id}: source size mismatch")
    index_meta = expect_file(
        FileBinding(
            f"{spec.canonical_release_id} D2/D3 index",
            spec.excluded_rows_path,
            spec.excluded_rows_sha256,
            spec.excluded_rows * 4,
        )
    )
    indices = np.fromfile(spec.excluded_rows_path, dtype="<u4")
    if len(indices) != spec.excluded_rows:
        raise RuntimeError(f"{spec.canonical_release_id}: excluded count mismatch")
    if len(indices) and (
        int(indices[-1]) >= spec.physical_rows or np.any(indices[1:] <= indices[:-1])
    ):
        raise RuntimeError(f"{spec.canonical_release_id}: invalid exclusion indices")
    if spec.eligible_rows != spec.physical_rows - spec.excluded_rows:
        raise RuntimeError(f"{spec.canonical_release_id}: eligibility arithmetic mismatch")

    result: dict[str, Any] = {
        "source": {
            "path": str(spec.source_path),
            "sha256": spec.documents_sha256,
            "size_bytes": spec.documents_file_bytes,
        },
        "d2_d3_excluded_rows": spec.excluded_rows,
        "d2_d3_eligible_rows": spec.eligible_rows,
        "excluded_row_index": index_meta,
    }
    if spec.release_manifest_path is not None:
        if spec.release_manifest_sha256 is None:
            raise ValueError(f"{spec.canonical_release_id}: missing release manifest SHA")
        result["release_manifest"] = expect_file(
            FileBinding(
                f"{spec.canonical_release_id} release manifest",
                spec.release_manifest_path,
                spec.release_manifest_sha256,
            )
        )
    if spec.aux_path is not None:
        if spec.aux_sha256 is None:
            raise ValueError(f"{spec.canonical_release_id}: missing aux SHA")
        result["d2_aux_acceleration"] = expect_file(
            FileBinding(
                f"{spec.canonical_release_id} D2 aux",
                spec.aux_path,
                spec.aux_sha256,
            )
        )
        aux = np.load(spec.aux_path, mmap_mode="r")
        if aux.dtype.names != ("nb", "sha") or aux.dtype.itemsize != 40:
            raise RuntimeError(f"{spec.canonical_release_id}: unexpected aux dtype")
        if len(aux) != spec.physical_rows:
            raise RuntimeError(f"{spec.canonical_release_id}: aux row count mismatch")
    elif spec.aux_sha256 is not None:
        raise ValueError(f"{spec.canonical_release_id}: aux SHA without path")
    return result


def init_worker(exclusion_hashes: tuple[bytes, ...]) -> None:
    global _WORKER_L1_HASHES
    _WORKER_L1_HASHES = frozenset(exclusion_hashes)


def has_json_boundary_newline(line: bytes, context: str) -> bool:
    start = line.find(TEXT_START)
    end = line.rfind(TEXT_END)
    if start < 0 or end < start:
        raise ValueError(f"{context}: unexpected frozen non-Python JSONL layout")
    text_start = start + len(TEXT_START)
    return line[text_start : text_start + 2] in (b"\\n", b"\\r") or line[end - 2 : end] in (
        b"\\n",
        b"\\r",
    )


def scan_release(
    spec: ReleaseSpec,
    release_ordinal: int,
    phase: int,
    boundary_bin: int | None,
    spool_root: Path | None,
) -> dict[str, Any]:
    if phase not in (1, 2):
        raise ValueError("phase must be 1 or 2")
    if phase == 2 and (boundary_bin is None or spool_root is None):
        raise ValueError("phase 2 needs boundary and spool root")
    started = time.monotonic()
    excluded = np.fromfile(spec.excluded_rows_path, dtype="<u4")
    excluded_position = 0
    aux = np.load(spec.aux_path, mmap_mode="r") if spec.aux_path is not None else None
    source_digest = hashlib.sha256()
    hist_documents = [0] * 256
    hist_bytes = [0] * 256
    physical_rows = 0
    eligible_rows = 0
    l1_matches = 0
    l1_match_payload_bytes = 0
    post_l1_occurrences = 0
    post_l1_bytes = 0
    boundary_reparsed_rows = 0
    spool_handles: dict[int, Any] = {}
    spooled_documents = 0
    spooled_bytes = 0
    first_candidate_above_boundary = None

    try:
        with open(spec.source_path, "rb", buffering=8 * 1024 * 1024) as source:
            for row_index, line in enumerate(source):
                source_digest.update(line)
                physical_rows += 1
                if excluded_position < len(excluded) and row_index == int(
                    excluded[excluded_position]
                ):
                    excluded_position += 1
                    continue
                eligible_rows += 1
                context = f"{spec.canonical_release_id}:{row_index}"
                raw_text: str | None = None
                if aux is None:
                    raw_text = load_text(line, context)
                    cleaned_sha, cleaned_bytes = f_payload(raw_text)
                    identity_sha = l1_identity_digest(raw_text)
                else:
                    record = aux[row_index].tobytes()
                    cleaned_bytes = int.from_bytes(record[:8], "little", signed=True)
                    cleaned_sha = record[8:40]
                    if cleaned_bytes < 0 or len(cleaned_sha) != 32:
                        raise RuntimeError(f"{context}: invalid aux record")
                    if has_json_boundary_newline(line, context):
                        boundary_reparsed_rows += 1
                        raw_text = load_text(line, context)
                        actual_sha, actual_bytes = f_payload(raw_text)
                        if (actual_sha, actual_bytes) != (cleaned_sha, cleaned_bytes):
                            raise RuntimeError(f"{context}: aux/text mismatch")
                        identity_sha = l1_identity_digest(raw_text)
                    else:
                        identity_sha = cleaned_sha

                if identity_sha in _WORKER_L1_HASHES:
                    l1_matches += 1
                    l1_match_payload_bytes += cleaned_bytes
                    continue

                candidate_rank = rank_digest(spec.bucket_name, cleaned_sha)
                rank_bin = candidate_rank[0]
                hist_documents[rank_bin] += 1
                hist_bytes[rank_bin] += cleaned_bytes
                post_l1_occurrences += 1
                post_l1_bytes += cleaned_bytes

                full_candidate_key = (
                    candidate_rank,
                    cleaned_sha,
                    release_ordinal,
                    row_index,
                    cleaned_bytes,
                )
                if phase == 2 and rank_bin > boundary_bin:
                    if (
                        first_candidate_above_boundary is None
                        or full_candidate_key[:4] < first_candidate_above_boundary[:4]
                    ):
                        first_candidate_above_boundary = full_candidate_key

                if phase == 2 and rank_bin <= boundary_bin:
                    if raw_text is None:
                        raw_text = load_text(line, context)
                        actual_sha, actual_bytes = f_payload(raw_text)
                        if (actual_sha, actual_bytes) != (cleaned_sha, cleaned_bytes):
                            raise RuntimeError(f"{context}: aux/text mismatch")
                    payload = encode_output_row(
                        raw_text,
                        spec.bucket_name,
                        spec.canonical_release_id,
                        row_index,
                        cleaned_sha,
                    )
                    handle = spool_handles.get(rank_bin)
                    if handle is None:
                        assert spool_root is not None
                        path = (
                            spool_root
                            / spec.canonical_release_id
                            / f"{spec.bucket_name.replace('/', '_')}.{rank_bin:03d}.spool"
                        )
                        path.parent.mkdir(parents=True, exist_ok=True)
                        handle = open(path, "wb", buffering=8 * 1024 * 1024)
                        spool_handles[rank_bin] = handle
                    handle.write(
                        SPOOL_HEADER.pack(
                            candidate_rank,
                            cleaned_sha,
                            release_ordinal,
                            row_index,
                            cleaned_bytes,
                            len(payload),
                        )
                    )
                    handle.write(payload)
                    spooled_documents += 1
                    spooled_bytes += cleaned_bytes
            source_stat = os.fstat(source.fileno())
            source_stat_identity = (
                source_stat.st_dev,
                source_stat.st_ino,
                source_stat.st_size,
                source_stat.st_mtime_ns,
                source_stat.st_ctime_ns,
            )
    finally:
        for handle in spool_handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()

    if excluded_position != len(excluded):
        raise RuntimeError(f"{spec.canonical_release_id}: exclusion index not consumed")
    if physical_rows != spec.physical_rows:
        raise RuntimeError(f"{spec.canonical_release_id}: physical row mismatch")
    if eligible_rows != spec.eligible_rows:
        raise RuntimeError(f"{spec.canonical_release_id}: eligible row mismatch")
    actual_source_sha = source_digest.hexdigest()
    if actual_source_sha != spec.documents_sha256:
        raise RuntimeError(f"{spec.canonical_release_id}: source SHA-256 mismatch")
    for label, actual, expected in (
        ("L1 matches", l1_matches, spec.expected_l1_matches),
        ("post-L1 occurrences", post_l1_occurrences, spec.expected_post_l1_occurrences),
        ("post-L1 UTF-8 bytes", post_l1_bytes, spec.expected_post_l1_utf8_bytes),
    ):
        if expected is not None and actual != expected:
            raise RuntimeError(f"{spec.canonical_release_id}: {label} {actual} != {expected}")
    return {
        "canonical_release_id": spec.canonical_release_id,
        "bucket_name": spec.bucket_name,
        "phase": phase,
        "source_documents_sha256": actual_source_sha,
        "source_stat_identity": source_stat_identity,
        "physical_rows": physical_rows,
        "d2_d3_eligible_rows": eligible_rows,
        "l1_match_occurrences": l1_matches,
        "l1_match_tokenizer_visible_utf8_bytes": l1_match_payload_bytes,
        "post_l1_occurrences": post_l1_occurrences,
        "post_l1_tokenizer_visible_utf8_bytes": post_l1_bytes,
        "hist_documents": hist_documents,
        "hist_tokenizer_visible_utf8_bytes": hist_bytes,
        "boundary_reparsed_rows": boundary_reparsed_rows,
        "spooled_documents": spooled_documents,
        "spooled_tokenizer_visible_utf8_bytes": spooled_bytes,
        "first_candidate_above_boundary": first_candidate_above_boundary,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def run_scan_phase(
    contract: BuildContract,
    ordinals: dict[str, dict[str, int]],
    exclusions: frozenset[bytes],
    phase: int,
    boundaries: dict[str, int] | None,
    spool_root: Path | None,
    workers: int,
) -> list[dict[str, Any]]:
    jobs = []
    for spec in contract.releases:
        ordinal = ordinals[spec.bucket_name][spec.canonical_release_id]
        boundary = None if boundaries is None else boundaries[spec.bucket_name]
        jobs.append((spec, ordinal, phase, boundary, spool_root))
    if workers == 1:
        init_worker(tuple(exclusions))
        results = [scan_release(*job) for job in jobs]
    else:
        if "fork" not in multiprocessing.get_all_start_methods():
            raise RuntimeError("parallel scans require fork")
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=init_worker,
            initargs=(tuple(exclusions),),
        ) as executor:
            futures = {
                executor.submit(scan_release, *job): job[0].canonical_release_id for job in jobs
            }
            results = []
            for future in as_completed(futures):
                release_id = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    raise RuntimeError(f"{release_id}: scan phase {phase} failed") from exc
    return sorted(results, key=lambda item: item["canonical_release_id"])


def bucket_histograms(
    contract: BuildContract, scan_results: Iterable[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    result = {
        bucket.canonical_name: {
            "documents": [0] * 256,
            "bytes": [0] * 256,
            "post_l1_occurrences": 0,
            "post_l1_bytes": 0,
            "l1_matches": 0,
        }
        for bucket in contract.buckets
    }
    for release in scan_results:
        bucket = result[release["bucket_name"]]
        bucket["documents"] = [
            a + b for a, b in zip(bucket["documents"], release["hist_documents"], strict=True)
        ]
        bucket["bytes"] = [
            a + b
            for a, b in zip(
                bucket["bytes"], release["hist_tokenizer_visible_utf8_bytes"], strict=True
            )
        ]
        bucket["post_l1_occurrences"] += release["post_l1_occurrences"]
        bucket["post_l1_bytes"] += release["post_l1_tokenizer_visible_utf8_bytes"]
        bucket["l1_matches"] += release["l1_match_occurrences"]
    return result


def choose_boundaries(
    contract: BuildContract, histograms: dict[str, dict[str, Any]]
) -> dict[str, int]:
    boundaries = {}
    for bucket in contract.buckets:
        histogram = histograms[bucket.canonical_name]
        if histogram["post_l1_bytes"] < bucket.target_bytes:
            raise RuntimeError(
                f"{bucket.canonical_name}: capacity below target "
                f"({histogram['post_l1_bytes']} < {bucket.target_bytes})"
            )
        cumulative = 0
        for rank_bin, value in enumerate(histogram["bytes"]):
            if cumulative + value >= bucket.target_bytes:
                boundaries[bucket.canonical_name] = rank_bin
                break
            cumulative += value
        else:  # pragma: no cover
            raise AssertionError("capacity check failed to yield a boundary")
    return boundaries


def key_json(
    key: tuple[bytes, bytes, int, int, int],
    ordinal_to_release: dict[int, str],
) -> dict[str, Any]:
    rank, cleaned_sha, release_ordinal, row_index, cleaned_bytes = key
    return {
        "rank128": rank.hex(),
        "cleaned_text_sha256": cleaned_sha.hex(),
        "canonical_release_id": ordinal_to_release[release_ordinal],
        "physical_row_index": row_index,
        "cleaned_utf8_bytes": cleaned_bytes,
    }


def read_spools(paths: list[Path], expected_bin: int) -> tuple[bytes, list[tuple[Any, ...]]]:
    blob = b"".join(path.read_bytes() for path in paths)
    records = []
    position = 0
    while position < len(blob):
        if len(blob) - position < SPOOL_HEADER.size:
            raise RuntimeError("truncated spool header")
        values = SPOOL_HEADER.unpack_from(blob, position)
        rank, cleaned_sha, release_ordinal, row_index, cleaned_bytes, payload_bytes = values
        payload_start = position + SPOOL_HEADER.size
        payload_end = payload_start + payload_bytes
        if payload_end > len(blob):
            raise RuntimeError("truncated spool payload")
        if rank[0] != expected_bin:
            raise RuntimeError("spool record in wrong rank bin")
        records.append((
            rank,
            cleaned_sha,
            release_ordinal,
            row_index,
            cleaned_bytes,
            payload_start,
            payload_bytes,
        ))
        position = payload_end
    return blob, records


def occurrence_set_fingerprint(bucket: BucketSpec, index_path: Path) -> tuple[str, bytes]:
    values = np.fromfile(index_path, dtype=SELECTION_DTYPE)
    ordered = np.sort(values, order=("release_ordinal", "physical_row_index"), kind="stable")
    canonical = ordered.tobytes()
    digest = hashlib.sha256()
    name = bucket.canonical_name.encode("utf-8")
    digest.update(b"PetitGPT-F-selected-occurrence-set-v1\0")
    digest.update(len(name).to_bytes(2, "big"))
    digest.update(name)
    digest.update(len(ordered).to_bytes(8, "big"))
    digest.update(canonical)
    return digest.hexdigest(), canonical


def finalize_bucket(
    bucket: BucketSpec,
    boundary_bin: int,
    spool_root: Path,
    staging: Path,
    expected_documents_by_bin: list[int],
    expected_bytes_by_bin: list[int],
    first_candidate_above_boundary: tuple[bytes, bytes, int, int, int] | None,
) -> dict[str, Any]:
    output_path = staging / f"{bucket.slug}.jsonl"
    index_dir = staging / "indices"
    index_dir.mkdir(exist_ok=True)
    index_path = index_dir / f"{bucket.slug}.selection.idx"
    release_to_ordinal = {
        release_id: index for index, release_id in enumerate(bucket.member_release_ids)
    }
    ordinal_to_release = {value: key for key, value in release_to_ordinal.items()}
    output_digest = hashlib.sha256()
    index_digest = hashlib.sha256()
    sequence_digest = hashlib.sha256(b"PetitGPT-F-selected-occurrence-sequence-v1\0")
    selected_documents = 0
    realized_bytes = 0
    output_file_bytes = 0
    last_selected = None
    first_unselected = None
    previous_key = None

    with (
        open(output_path, "wb", buffering=8 * 1024 * 1024) as output,
        open(index_path, "wb", buffering=8 * 1024 * 1024) as index,
    ):
        reached = False
        for rank_bin in range(boundary_bin + 1):
            paths = [
                spool_root
                / release_id
                / f"{bucket.canonical_name.replace('/', '_')}.{rank_bin:03d}.spool"
                for release_id in bucket.member_release_ids
            ]
            paths = [path for path in paths if path.is_file()]
            blob, records = read_spools(paths, rank_bin)
            if (
                len(records) != expected_documents_by_bin[rank_bin]
                or sum(record[4] for record in records) != expected_bytes_by_bin[rank_bin]
            ):
                raise RuntimeError(f"{bucket.canonical_name}: incomplete rank-bin spool {rank_bin}")
            records.sort(key=lambda item: item[:4])
            for record in records:
                rank, cleaned_sha, release_ordinal, row_index, cleaned_bytes, start, length = record
                sort_key = (rank, cleaned_sha, release_ordinal, row_index)
                if previous_key is not None and sort_key <= previous_key:
                    raise RuntimeError(f"{bucket.canonical_name}: non-ascending rank key")
                previous_key = sort_key
                full_key = (rank, cleaned_sha, release_ordinal, row_index, cleaned_bytes)
                if reached:
                    if first_unselected is None:
                        first_unselected = full_key
                    continue
                payload = memoryview(blob)[start : start + length]
                output.write(payload)
                output_digest.update(payload)
                output_file_bytes += length
                selected_record = SELECTION_RECORD.pack(
                    rank, cleaned_sha, release_ordinal, row_index, cleaned_bytes
                )
                index.write(selected_record)
                index_digest.update(selected_record)
                sequence_digest.update(selected_record)
                selected_documents += 1
                realized_bytes += cleaned_bytes
                last_selected = full_key
                if realized_bytes >= bucket.target_bytes:
                    reached = True
            del blob
            for path in paths:
                path.unlink()
        if first_unselected is None:
            first_unselected = first_candidate_above_boundary
        if not reached or last_selected is None:
            raise RuntimeError(f"{bucket.canonical_name}: target not reached")
        if first_unselected is not None and first_unselected[:4] <= last_selected[:4]:
            raise RuntimeError(f"{bucket.canonical_name}: invalid first-unselected boundary")
        output.flush()
        os.fsync(output.fileno())
        index.flush()
        os.fsync(index.fileno())

    set_sha, canonical_set = occurrence_set_fingerprint(bucket, index_path)
    last_bytes = last_selected[4]
    if not (realized_bytes - last_bytes < bucket.target_bytes <= realized_bytes):
        raise RuntimeError(f"{bucket.canonical_name}: boundary invariant failed")
    return {
        "canonical_name": bucket.canonical_name,
        "canonical_name_utf8_hex": bucket.canonical_name.encode("utf-8").hex(),
        "slug": bucket.slug,
        "member_release_ids": list(bucket.member_release_ids),
        "target_cleaned_utf8_bytes": bucket.target_bytes,
        "selected_documents": selected_documents,
        "realized_cleaned_utf8_bytes": realized_bytes,
        "overshoot_bytes": realized_bytes - bucket.target_bytes,
        "last_document_cleaned_utf8_bytes": last_bytes,
        "last_selected": key_json(last_selected, ordinal_to_release),
        "first_unselected": (
            None if first_unselected is None else key_json(first_unselected, ordinal_to_release)
        ),
        "output": {
            "path": output_path.name,
            "size_bytes": output_file_bytes,
            "sha256": output_digest.hexdigest(),
            "rows": selected_documents,
        },
        "selection_index": {
            "path": str(index_path.relative_to(staging)),
            "record_schema": (
                "rank128[16] || cleaned_text_sha256[32] || release_ordinal[u8] || "
                "physical_row_index[u64be] || cleaned_utf8_bytes[u64be]"
            ),
            "record_bytes": SELECTION_RECORD.size,
            "records": selected_documents,
            "size_bytes": index_path.stat().st_size,
            "sha256": index_digest.hexdigest(),
            "release_ordinal": release_to_ordinal,
        },
        "selected_occurrence_sequence_sha256": sequence_digest.hexdigest(),
        "selected_occurrence_set_sha256": set_sha,
        "_canonical_set_records": canonical_set,
    }


def self_verify_outputs(
    staging: Path,
    buckets: tuple[BucketSpec, ...],
    bucket_results: list[dict[str, Any]],
    exclusions: frozenset[bytes],
) -> dict[str, Any]:
    total_rows = 0
    total_text_bytes = 0
    l1_intersection = 0
    by_name = {item["canonical_name"]: item for item in bucket_results}
    for bucket in buckets:
        expected = by_name[bucket.canonical_name]
        output_path = staging / expected["output"]["path"]
        index_path = staging / expected["selection_index"]["path"]
        ordinal_to_release = {
            value: key for key, value in expected["selection_index"]["release_ordinal"].items()
        }
        output_digest = hashlib.sha256()
        index_digest = hashlib.sha256()
        rows = 0
        text_bytes = 0
        previous_key = None
        with (
            open(output_path, "rb", buffering=8 * 1024 * 1024) as output,
            open(index_path, "rb", buffering=8 * 1024 * 1024) as index,
        ):
            for line in output:
                output_digest.update(line)
                row = load_json_object(line, f"{bucket.canonical_name}:output:{rows}")
                if tuple(row) != OUTPUT_FIELDS:
                    raise RuntimeError(f"{bucket.canonical_name}: output schema/order mismatch")
                text = row["text"]
                if not isinstance(text, str):
                    raise RuntimeError(f"{bucket.canonical_name}: output text type mismatch")
                cleaned_sha, cleaned_bytes = f_payload(text)
                if row["cleaned_text_sha256"] != cleaned_sha.hex():
                    raise RuntimeError(f"{bucket.canonical_name}: output text SHA mismatch")
                if row["canonical_source"] != bucket.canonical_name:
                    raise RuntimeError(f"{bucket.canonical_name}: canonical source mismatch")
                release_id = row["canonical_release_id"]
                if release_id not in bucket.member_release_ids:
                    raise RuntimeError(f"{bucket.canonical_name}: release membership mismatch")
                row_index = row["physical_row_index"]
                if not isinstance(row_index, int) or row_index < 0:
                    raise RuntimeError(f"{bucket.canonical_name}: row index mismatch")
                rank = rank_digest(bucket.canonical_name, cleaned_sha)
                key = (rank, cleaned_sha, release_id, row_index)
                if previous_key is not None and key <= previous_key:
                    raise RuntimeError(f"{bucket.canonical_name}: output order mismatch")
                previous_key = key
                raw_index = index.read(SELECTION_RECORD.size)
                if len(raw_index) != SELECTION_RECORD.size:
                    raise RuntimeError(f"{bucket.canonical_name}: short selection index")
                index_digest.update(raw_index)
                irank, isha, ordinal, irow, ibytes = SELECTION_RECORD.unpack(raw_index)
                if (
                    irank != rank
                    or isha != cleaned_sha
                    or ordinal_to_release.get(ordinal) != release_id
                    or irow != row_index
                    or ibytes != cleaned_bytes
                ):
                    raise RuntimeError(f"{bucket.canonical_name}: output/index mismatch")
                if l1_identity_digest(text) in exclusions:
                    l1_intersection += 1
                rows += 1
                text_bytes += cleaned_bytes
            if index.read(1):
                raise RuntimeError(f"{bucket.canonical_name}: extra index records")
        if output_digest.hexdigest() != expected["output"]["sha256"]:
            raise RuntimeError(f"{bucket.canonical_name}: output SHA mismatch")
        if output_path.stat().st_size != expected["output"]["size_bytes"]:
            raise RuntimeError(f"{bucket.canonical_name}: output size mismatch")
        if index_digest.hexdigest() != expected["selection_index"]["sha256"]:
            raise RuntimeError(f"{bucket.canonical_name}: selection index SHA mismatch")
        if index_path.stat().st_size != expected["selection_index"]["size_bytes"]:
            raise RuntimeError(f"{bucket.canonical_name}: selection index size mismatch")
        if (
            rows != expected["selected_documents"]
            or text_bytes != expected["realized_cleaned_utf8_bytes"]
        ):
            raise RuntimeError(f"{bucket.canonical_name}: accounting mismatch")
        total_rows += rows
        total_text_bytes += text_bytes
    if l1_intersection:
        raise RuntimeError(f"selected output intersects L1: {l1_intersection}")
    return {
        "status": "PASS",
        "staging_verified_before_publication": True,
        "output_schema_and_field_order": list(OUTPUT_FIELDS),
        "output_rank_order": "strict ascending full frozen tuple",
        "selected_documents": total_rows,
        "cleaned_utf8_bytes": total_text_bytes,
        "l1_exclusion_intersection": 0,
    }


def write_atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def git_head() -> str | None:
    head_path = PROJECT_ROOT / ".git/HEAD"
    if not head_path.is_file():
        return None
    value = head_path.read_text(encoding="ascii").strip()
    if value.startswith("ref: "):
        ref_path = PROJECT_ROOT / ".git" / value[5:]
        if not ref_path.is_file():
            return None
        value = ref_path.read_text(encoding="ascii").strip()
    return value if len(value) == 40 else None


def make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def source_stat_identities(contract: BuildContract) -> dict[str, tuple[int, ...]]:
    result = {}
    for spec in contract.releases:
        source_stat = os.stat(spec.source_path, follow_symlinks=False)
        result[spec.canonical_release_id] = (
            source_stat.st_dev,
            source_stat.st_ino,
            source_stat.st_size,
            source_stat.st_mtime_ns,
            source_stat.st_ctime_ns,
        )
    return result


def validate_static_bindings(contract: BuildContract) -> dict[str, Any]:
    result = {}
    for binding in contract.static_bindings:
        if binding.role in result:
            raise ValueError(f"duplicate static binding role: {binding.role}")
        result[binding.role] = expect_file(binding)
    return result


def build_corpus(
    contract: BuildContract,
    out_dir: Path,
    *,
    workers: int = 1,
    make_immutable: bool = True,
) -> dict[str, Any]:
    if workers < 1 or workers > len(contract.releases):
        raise ValueError("workers must be between 1 and release count")
    ordinals = validate_contract_shape(contract)
    out_dir = out_dir.resolve()
    staging = out_dir.with_name(f".{out_dir.name}.staging")
    if os.path.lexists(out_dir):
        raise FileExistsError(f"immutable release exists: {out_dir}")
    if os.path.lexists(staging):
        raise FileExistsError(f"stale staging requires inspection: {staging}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    builder_sha_before = sha256_file(SCRIPT_PATH)
    static_before = validate_static_bindings(contract)
    exclusions = load_l1_exclusions(contract)
    controls_before = {
        spec.canonical_release_id: validate_release_control(spec)
        for spec in sorted(contract.releases, key=lambda item: item.canonical_release_id)
    }
    staging.mkdir()
    work = staging / "_work"
    spool_root = work / "spools"
    spool_root.mkdir(parents=True)
    try:
        pass1 = run_scan_phase(contract, ordinals, exclusions, 1, None, None, workers)
        hist1 = bucket_histograms(contract, pass1)
        pool_occurrences = sum(item["post_l1_occurrences"] for item in pass1)
        pool_bytes = sum(item["post_l1_tokenizer_visible_utf8_bytes"] for item in pass1)
        if (
            contract.expected_pool_post_l1_occurrences is not None
            and pool_occurrences != contract.expected_pool_post_l1_occurrences
        ):
            raise RuntimeError("F pool occurrence total mismatch")
        if (
            contract.expected_pool_post_l1_utf8_bytes is not None
            and pool_bytes != contract.expected_pool_post_l1_utf8_bytes
        ):
            raise RuntimeError("F pool UTF-8 byte total mismatch")
        boundaries = choose_boundaries(contract, hist1)

        pass2 = run_scan_phase(contract, ordinals, exclusions, 2, boundaries, spool_root, workers)
        hist2 = bucket_histograms(contract, pass2)
        if hist1 != hist2:
            raise RuntimeError("pass-1/pass-2 population mismatch")
        pass2_source_stats = {
            item["canonical_release_id"]: tuple(item["source_stat_identity"]) for item in pass2
        }
        source_stats_after_scan = source_stat_identities(contract)
        if pass2_source_stats != source_stats_after_scan:
            raise RuntimeError("frozen source changed immediately after scan")

        static_after = validate_static_bindings(contract)
        exclusions_after = load_l1_exclusions(contract)
        controls_after = {
            spec.canonical_release_id: validate_release_control(spec)
            for spec in sorted(contract.releases, key=lambda item: item.canonical_release_id)
        }
        if (
            static_before != static_after
            or controls_before != controls_after
            or exclusions != exclusions_after
        ):
            raise RuntimeError("frozen input changed during production")
        builder_sha_after = sha256_file(SCRIPT_PATH)
        if builder_sha_after != builder_sha_before:
            raise RuntimeError("builder implementation changed during production")

        bucket_results = []
        aggregate_set_digest = hashlib.sha256(b"PetitGPT-F-selected-occurrence-sets-v1\0")
        first_above_by_bucket = {}
        for bucket in contract.buckets:
            candidates = [
                item["first_candidate_above_boundary"]
                for item in pass2
                if item["bucket_name"] == bucket.canonical_name
                and item["first_candidate_above_boundary"] is not None
            ]
            first_above_by_bucket[bucket.canonical_name] = (
                min(candidates, key=lambda item: item[:4]) if candidates else None
            )
        for bucket in contract.buckets:
            result = finalize_bucket(
                bucket,
                boundaries[bucket.canonical_name],
                spool_root,
                staging,
                hist2[bucket.canonical_name]["documents"],
                hist2[bucket.canonical_name]["bytes"],
                first_above_by_bucket[bucket.canonical_name],
            )
            result["boundary_bin"] = boundaries[bucket.canonical_name]
            result["candidate_capacity_documents"] = hist2[bucket.canonical_name][
                "post_l1_occurrences"
            ]
            result["candidate_capacity_cleaned_utf8_bytes"] = hist2[bucket.canonical_name][
                "post_l1_bytes"
            ]
            canonical_set = result.pop("_canonical_set_records")
            name = bucket.canonical_name.encode("utf-8")
            aggregate_set_digest.update(len(name).to_bytes(2, "big"))
            aggregate_set_digest.update(name)
            aggregate_set_digest.update(len(canonical_set).to_bytes(8, "big"))
            aggregate_set_digest.update(canonical_set)
            bucket_results.append(result)
        shutil.rmtree(work)

        self_verification = self_verify_outputs(
            staging, contract.buckets, bucket_results, exclusions
        )
        static_final = validate_static_bindings(contract)
        exclusions_final = load_l1_exclusions(contract)
        controls_final = {
            spec.canonical_release_id: validate_release_control(spec)
            for spec in sorted(contract.releases, key=lambda item: item.canonical_release_id)
        }
        if (
            static_final != static_after
            or exclusions_final != exclusions_after
            or controls_final != controls_after
            or source_stat_identities(contract) != source_stats_after_scan
        ):
            raise RuntimeError("frozen input changed before publication")
        builder_sha_final = sha256_file(SCRIPT_PATH)
        if builder_sha_final != builder_sha_after:
            raise RuntimeError("builder implementation changed before publication")
        total_target = sum(item.target_bytes for item in contract.buckets)
        total_selected = sum(item["selected_documents"] for item in bucket_results)
        total_realized = sum(item["realized_cleaned_utf8_bytes"] for item in bucket_results)
        release_stats = {item["canonical_release_id"]: item for item in pass2}
        by_spec = {item.canonical_release_id: item for item in contract.releases}
        release_manifest = []
        for release_id in sorted(release_stats):
            stats = release_stats[release_id]
            spec = by_spec[release_id]
            release_manifest.append({
                "canonical_release_id": release_id,
                "bucket_name": spec.bucket_name,
                **controls_final[release_id],
                "measured": {
                    key: stats[key]
                    for key in (
                        "physical_rows",
                        "d2_d3_eligible_rows",
                        "l1_match_occurrences",
                        "l1_match_tokenizer_visible_utf8_bytes",
                        "post_l1_occurrences",
                        "post_l1_tokenizer_visible_utf8_bytes",
                        "boundary_reparsed_rows",
                    )
                },
            })

        builder_sha = builder_sha_final
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE_SELF_VERIFIED",
            "immutable_publication": True,
            "contract": {
                "bucket_order": [item.canonical_name for item in contract.buckets],
                "total_target_cleaned_utf8_bytes": total_target,
                "rank": {
                    "algorithm": RANK_ALGORITHM,
                    "digest_bytes": 16,
                    "person_ascii": RANK_PERSON.decode("ascii"),
                    "seed_ascii_decimal": RANK_SEED_ASCII.decode("ascii"),
                    "message": (
                        'ASCII decimal seed "20250814" || NUL || canonical bucket '
                        "name UTF-8 || NUL || raw 32-byte cleaned_text_sha256 digest"
                    ),
                    "ordering_tuple": [
                        "rank128",
                        "cleaned_text_sha256",
                        "canonical_release_id",
                        "physical_row_index",
                    ],
                    "direction": "ascending independently inside each bucket",
                },
                "text_contract": {
                    "source_field": "text",
                    "transform": "identity",
                    "encoding": "UTF-8",
                    "byte_accounting": "len(text.encode('utf-8'))",
                    "cleaned_text_sha256": (
                        "SHA-256 of exact tokenizer-visible source text UTF-8 bytes"
                    ),
                    "strip": False,
                    "nfkc": False,
                    "quote_normalization": False,
                    "whitespace_collapse": False,
                    "separator_added": False,
                    "bos_added": False,
                    "eos_added": False,
                },
                "l1_exclusion_identity_view": {
                    "purpose": "membership only; not F payload/rank/accounting text",
                    "algorithm": EXCLUSION_ALGORITHM,
                    "transform": "strip leading/trailing U+000A and U+000D only",
                    "optional_flags": EXPECTED_L1_CLEANING,
                },
                "filter_order": [
                    "physical frozen occurrence",
                    "D2/D3 logical eligibility",
                    "L1 exclusion identity membership",
                    "lossless F text and cleaned_text_sha256",
                    "rank",
                ],
                "occurrence_semantics": (
                    "preserve eligible logical occurrences; no new deduplication"
                ),
                "selection": (
                    "whole-document ascending prefix first reaching/exceeding target; "
                    "no truncation, underfill, or cross-bucket refill"
                ),
                "output_jsonl": {
                    "encoding": "UTF-8",
                    "ensure_ascii": False,
                    "compact_separators": [",", ":"],
                    "framing": "one JSON object plus LF; framing LF is not text",
                    "field_order": list(OUTPUT_FIELDS),
                    "row_order": "full frozen rank tuple ascending",
                },
            },
            "input_bindings": {
                "l1_checkpoint_commit": contract.l1_checkpoint_commit,
                "static_files": static_final,
                "l1_exclusion": {
                    "path": str(contract.l1_exclusion_manifest_path),
                    "sha256": contract.l1_exclusion_manifest_sha256,
                    "unique_identities": len(exclusions),
                },
                "frozen_stage_e_release_documents_sha256": dict(contract.frozen_stage_e_releases),
                "releases": release_manifest,
                "authoritative_global_post_l1_occurrences": (
                    contract.authoritative_global_post_l1_occurrences
                ),
                "f_pool_post_l1_occurrences": pool_occurrences,
                "f_pool_post_l1_tokenizer_visible_utf8_bytes": pool_bytes,
            },
            "selection": {
                "buckets": bucket_results,
                "total_target_cleaned_utf8_bytes": total_target,
                "total_selected_documents": total_selected,
                "total_realized_cleaned_utf8_bytes": total_realized,
                "total_overshoot_bytes": total_realized - total_target,
                "selected_occurrence_set_sha256": aggregate_set_digest.hexdigest(),
                "selected_occurrence_set_fingerprint_contract": (
                    "SHA256 domain-separated fixed bucket sequence; each bucket uses "
                    "selection records sorted by release_ordinal and physical_row_index"
                ),
                "l1_exclusion_intersection": 0,
            },
            "verification": self_verification,
            "implementation": {
                "builder_path": str(SCRIPT_PATH.relative_to(PROJECT_ROOT)),
                "builder_sha256": builder_sha,
                "git_head_at_build": git_head(),
                "workers": workers,
                "algorithm": ("two-pass 8-bit rank radix boundary with exact full-key bin sort"),
                "resume_implemented": False,
                "publication": ("sibling staging, self-verification, manifest-last, atomic rename"),
            },
        }
        fingerprint_projection = {
            "schema_version": manifest["schema_version"],
            "contract": manifest["contract"],
            "input_bindings": manifest["input_bindings"],
            "selection": manifest["selection"],
            "implementation": {
                "builder_path": manifest["implementation"]["builder_path"],
                "builder_sha256": builder_sha,
                "algorithm": manifest["implementation"]["algorithm"],
            },
        }
        manifest["run_fingerprint_sha256"] = hashlib.sha256(
            canonical_json_bytes(fingerprint_projection)
        ).hexdigest()

        checksums = []
        for path in sorted(staging.rglob("*")):
            if path.is_file() and path.name not in {"manifest.json", "SHA256SUMS"}:
                checksums.append(f"{sha256_file(path)}  {path.relative_to(staging)}")
        checksum_path = staging / "SHA256SUMS"
        with open(checksum_path, "w", encoding="ascii") as handle:
            handle.write("\n".join(checksums) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        write_atomic_json(staging / "manifest.json", manifest)
        fsync_directory(staging / "indices")
        fsync_directory(staging)
        if make_immutable:
            make_read_only(staging)
        if os.path.lexists(out_dir):
            raise FileExistsError(f"immutable release appeared during build: {out_dir}")
        os.replace(staging, out_dir)
        fsync_directory(out_dir.parent)
        return manifest
    except BaseException:
        if staging.exists():
            for path in staging.rglob("*"):
                try:
                    mode = 0o700 if path.is_dir() else 0o600
                    path.chmod(mode)
                except OSError:
                    pass
            try:
                staging.chmod(0o700)
                shutil.rmtree(staging)
            except OSError:
                pass
        raise


def resolve_bound_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_file():
        return path
    frozen_root = Path("/workspace/petitgpt")
    try:
        relative = path.relative_to(frozen_root)
    except ValueError as exc:
        raise FileNotFoundError(f"bound source unavailable: {path}") from exc
    relocated = PROJECT_ROOT / relative
    if not relocated.is_file():
        raise FileNotFoundError(f"bound source unavailable: {path} or {relocated}")
    return relocated


def production_contract() -> BuildContract:
    fixed_bindings = (
        FileBinding("stage_e_allocation_contract", STAGE_E_PATH, STAGE_E_SHA256),
        FileBinding("d2_d3_eligibility_manifest", ELIGIBILITY_PATH, ELIGIBILITY_SHA256),
        FileBinding("l1_identity_cleaner_implementation", L1_CLEANER_PATH, L1_CLEANER_SHA256),
        FileBinding(
            "f_preflight_capacity_reproduction",
            F_PREFLIGHT_CAPACITY_PATH,
            F_PREFLIGHT_CAPACITY_SHA256,
        ),
    )
    for binding in fixed_bindings:
        expect_file(binding)
    stage_e = json.loads(STAGE_E_PATH.read_text(encoding="utf-8"))
    if stage_e.get("contract_status") != "FROZEN":
        raise ValueError("Stage E contract is not frozen")
    frozen_releases = tuple(
        sorted(
            (item["release_key"], item["documents_sha256"])
            for item in stage_e["input_binding"]["releases"]
        )
    )
    if len(frozen_releases) != 8:
        raise ValueError("Stage E must bind eight releases")

    eligibility = json.loads(ELIGIBILITY_PATH.read_text(encoding="utf-8"))
    if eligibility.get("kind") != ELIGIBILITY_KIND or eligibility.get("schema_version") != 1:
        raise ValueError("unexpected eligibility manifest")
    by_release = {
        item["release_key"]: (raw_path, item) for raw_path, item in eligibility["files"].items()
    }
    if set(by_release) != {item[0] for item in frozen_releases}:
        raise ValueError("eligibility and Stage E universes differ")

    releases = []
    for release_id, frozen in sorted(PRODUCTION_RELEASES.items()):
        raw_path, eligible = by_release[release_id]
        expected = {
            "documents_sha256": frozen["documents_sha256"],
            "total_rows": frozen["physical_rows"],
            "eligible_rows": frozen["eligible_rows"],
            "excluded_rows": frozen["excluded_rows"],
        }
        for key, wanted in expected.items():
            if eligible.get(key) != wanted:
                raise ValueError(f"{release_id}: eligibility {key} mismatch")
        source_path = resolve_bound_path(raw_path)
        if source_path.resolve() != (PROJECT_ROOT / frozen["source"]).resolve():
            raise ValueError(f"{release_id}: source path mismatch")
        releases.append(
            ReleaseSpec(
                canonical_release_id=release_id,
                bucket_name=frozen["bucket"],
                source_path=source_path,
                documents_sha256=frozen["documents_sha256"],
                documents_file_bytes=frozen["source_size"],
                physical_rows=frozen["physical_rows"],
                eligible_rows=frozen["eligible_rows"],
                excluded_rows=frozen["excluded_rows"],
                excluded_rows_path=(
                    ELIGIBILITY_PATH.parent / eligible["excluded_row_indices_file"]
                ),
                excluded_rows_sha256=eligible["excluded_row_indices_sha256"],
                release_manifest_path=PROJECT_ROOT / frozen["release_manifest"],
                release_manifest_sha256=frozen["release_manifest_sha256"],
                aux_path=(None if frozen["aux"] is None else PROJECT_ROOT / frozen["aux"]),
                aux_sha256=frozen["aux_sha256"],
                expected_l1_matches=frozen["l1_matches"],
                expected_post_l1_occurrences=frozen["post_l1_occurrences"],
                expected_post_l1_utf8_bytes=frozen["post_l1_utf8_bytes"],
            )
        )
    return BuildContract(
        buckets=PRODUCTION_BUCKETS,
        releases=tuple(releases),
        l1_exclusion_manifest_path=L1_EXCLUSION_PATH,
        l1_exclusion_manifest_sha256=L1_EXCLUSION_SHA256,
        l1_exclusion_hash_count=172_483,
        static_bindings=fixed_bindings,
        expected_pool_post_l1_occurrences=27_409_026,
        expected_pool_post_l1_utf8_bytes=106_809_050_037,
        authoritative_global_post_l1_occurrences=29_966_831,
        frozen_stage_e_releases=frozen_releases,
        l1_checkpoint_commit="90e14a34c882d75b67e10b5d2b0623c99cfee14e",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Final immutable release directory; final and sibling staging must not exist.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Parallel release scans, 1 through 6; does not alter semantics.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.monotonic()
    print("[F] validating frozen contract", flush=True)
    contract = production_contract()
    print("[F] building staged corpus", flush=True)
    manifest = build_corpus(contract, args.out_dir, workers=args.workers)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "release": str(args.out_dir.resolve()),
                "selected_documents": manifest["selection"]["total_selected_documents"],
                "realized_cleaned_utf8_bytes": manifest["selection"][
                    "total_realized_cleaned_utf8_bytes"
                ],
                "run_fingerprint_sha256": manifest["run_fingerprint_sha256"],
                "elapsed_seconds": round(time.monotonic() - started, 3),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
