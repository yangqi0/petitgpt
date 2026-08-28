"""Synthetic accepted-Stage-I publications for the Stage-M / Stage-P native tests.

The builder deliberately interleaves stages *and* sources in the physical order, so a packer
that grouped by stage globally, grouped by source, sorted by selection rank, or interleaved by
weight would produce an observably different stream than the accepted physical order.
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from pretrain.build_pretrain_shards import encode_with_accounting
from pretrain.stage_m_contract_v1 import canonical_json_bytes
from src.special_tokens import BOS_ID, EOS_ID, SPECIAL_TOKENS

RECORD_SCHEMA = "petitgpt-stage-i-document-v1"
MANIFEST_SCHEMA = "petitgpt-stage-i-manifest-v1"

_CORPUS = [
    "Hello world, how are you today? The quick brown fox jumps over the lazy dog.",
    "def add(a, b):\n    return a + b\n",
    "Wikipedia is a free online encyclopedia written collaboratively.",
    "Stack exchange answers often contain code blocks and prose together.",
    "A tutorial paragraph explaining a concept step by step, with examples.",
]


def tiny_tokenizer() -> Tokenizer:
    """A tiny byte-level BPE with the seven specials at canonical IDs, trained in-process."""
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=500,
        min_frequency=1,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(_CORPUS, trainer=trainer)
    for index, token in enumerate(SPECIAL_TOKENS):
        assert tok.token_to_id(token) == index
    tok.encode_special_tokens = True
    return tok


def framed_ids(tok: Tokenizer, text: str) -> list[int]:
    ids, _content, _boundary = encode_with_accounting(
        tok, text, add_bos=True, add_eos=True, bos_id=BOS_ID, eos_id=EOS_ID
    )
    return ids


def make_record(
    tok: Tokenizer, *, stage: str, source_id: str, binding: str, ordinal: int, rank: int, text: str
) -> dict[str, Any]:
    ids, content, _boundary = encode_with_accounting(
        tok, text, add_bos=True, add_eos=True, bos_id=BOS_ID, eos_id=EOS_ID
    )
    cleaned = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "canonical_fingerprint": hashlib.sha256(f"cf:{text}".encode()).hexdigest(),
        "cleaned_text_sha256": cleaned,
        "content_token_count": content,
        "input_binding_id": binding,
        "input_record_sha256": hashlib.sha256(f"ir:{text}".encode()).hexdigest(),
        "raw_sha256": hashlib.sha256(f"raw:{text}".encode()).hexdigest(),
        "schema_version": RECORD_SCHEMA,
        "selection_ordinal_within_node": rank,
        "serialized_token_count": len(ids),
        "source_id": source_id,
        "stable_input_record_ordinal": ordinal,
        "stage": stage,
        "training_text": text,
    }


def interleaved_records(tok: Tokenizer, *, per_group: int = 3) -> list[dict[str, Any]]:
    """Records whose physical order interleaves both stages and several sources."""
    groups = [
        ("stage_b", "b_web", "ib_web"),
        ("stage_a", "a_web", "ib_web"),
        ("stage_b", "b_code", "ib_code"),
        ("stage_a", "a_wiki", "ib_wiki"),
        ("stage_b", "b_web", "ib_web"),
        ("stage_a", "a_web", "ib_web"),
    ]
    records: list[dict[str, Any]] = []
    counters: dict[tuple[str, str], int] = {}
    for group_index, (stage, source_id, binding) in enumerate(groups):
        for offset in range(per_group):
            key = (stage, source_id)
            rank = counters.get(key, 0)
            counters[key] = rank + 1
            text = f"{_CORPUS[(group_index + offset) % len(_CORPUS)]} [g{group_index}o{offset}]"
            records.append(
                make_record(
                    tok,
                    stage=stage,
                    source_id=source_id,
                    binding=binding,
                    ordinal=group_index * 100 + offset,
                    # Selection rank deliberately disagrees with physical order.
                    rank=(per_group - offset) * 10 + group_index,
                    text=text,
                )
            )
    return records


def write_accepted_stage_i(
    root: Path,
    records: Sequence[dict[str, Any]],
    *,
    records_per_shard: int = 4,
    run_identity: str | None = None,
) -> Path:
    """Materialize a valid accepted Stage-I publication for ``records`` in the given order."""
    root = Path(root)
    documents = root / "documents"
    documents.mkdir(parents=True, exist_ok=True)

    shards: list[dict[str, Any]] = []
    for index in range(0, len(records), records_per_shard):
        chunk = records[index : index + records_per_shard]
        name = f"documents-{index // records_per_shard:05d}.jsonl"
        payload = b"".join(canonical_json_bytes(record) for record in chunk)
        (documents / name).write_bytes(payload)
        shards.append({
            "name": name,
            "records": len(chunk),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })

    nodes: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = (str(record["stage"]), str(record["source_id"]))
        node = nodes.setdefault(
            key,
            {
                "stage": key[0],
                "source_id": key[1],
                "selected_identities": 0,
                "selected_serialized_tokens": 0,
            },
        )
        node["selected_identities"] += 1
        node["selected_serialized_tokens"] += int(record["serialized_token_count"])

    total_records = len(records)
    total_serialized = sum(int(r["serialized_token_count"]) for r in records)
    total_content = sum(int(r["content_token_count"]) for r in records)
    identity = (
        run_identity
        or hashlib.sha256(
            canonical_json_bytes([r["cleaned_text_sha256"] for r in records])
        ).hexdigest()
    )

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "record_schema_version": RECORD_SCHEMA,
        "records_per_shard": records_per_shard,
        "shards": shards,
        "nodes": sorted(nodes.values(), key=lambda n: (n["stage"], n["source_id"])),
        "totals": {
            "records": total_records,
            "serialized_tokens": total_serialized,
            "content_tokens": total_content,
            "unique_cleaned_identities": total_records,
            "shards": len(shards),
        },
        "stage_i_run": {
            "run_identity": identity,
            "post_pass1_result_identity_sha256": hashlib.sha256(
                f"layer2:{identity}".encode()
            ).hexdigest(),
        },
    }
    manifest_bytes = canonical_json_bytes(manifest)
    (root / "manifest.json").write_bytes(manifest_bytes)
    (root / "COMPLETE").write_bytes(
        canonical_json_bytes({
            "marker": "COMPLETE",
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "stage_i_run_identity": identity,
        })
    )
    return root


def write_exclusion_manifest(path: Path, *, hash_count: int = 3) -> Path:
    """Write a reference-exclusion manifest.

    With the default ``hash_count=3`` this is a standalone fixture. Pass ``shared=True`` via
    :func:`write_shared_exclusion_manifest` to emit bytes identical to the reserve exclusion
    that tests/test_plan_pretrain_run.py's provenance fixture creates, so the candidate-M plan,
    both Stage-M releases and the G/G2 authorities all bind one artifact -- which is what the
    real pipeline does and what the R2-A shared-authority comparison requires.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        canonical_json_bytes({
            "kind": "petitgpt_reference_validation_exclusions",
            "hash_algorithm": "sha256-cleaned-text-utf8-v1",
            "hash_count": hash_count,
            "hashes": [hashlib.sha256(f"x{i}".encode()).hexdigest() for i in range(hash_count)],
        })
    )
    return path


# Byte-for-byte the payload tests/test_plan_pretrain_run.py::_write_full_provenance writes as
# its reserve exclusion. Kept here so the native end-to-end fixture can bind exactly one
# exclusion artifact across the plan, both releases and the G/G2 authorities.
SHARED_EXCLUSION_HASHES = ["1" * 64, "2" * 64]
SHARED_EXCLUSION_CLEANING = {
    "strip_leading_noise": False,
    "normalize_quotes": False,
    "underscores_policy": "keep",
    "min_chars": 0,
    "min_ascii_ratio": 0.0,
}


def write_shared_exclusion_manifest(path: Path) -> Path:
    """Emit the exact reserve-exclusion bytes the legacy provenance fixture also writes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "schema_version": 1,
            "kind": "petitgpt_reference_validation_exclusions",
            "hash_algorithm": "sha256-cleaned-text-utf8-v1",
            "cleaning": SHARED_EXCLUSION_CLEANING,
            "hash_count": len(SHARED_EXCLUSION_HASHES),
            "hashes": SHARED_EXCLUSION_HASHES,
        }),
        encoding="utf-8",
    )
    return path


def save_tokenizer(tok: Tokenizer, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(path))
    return path


def read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def e2e_records(tok, count: int = 400):
    """Records whose physical order interleaves both stages and two sources."""
    stages = ("stage_b", "stage_a", "stage_b", "stage_a")
    return [
        make_record(
            tok,
            stage=stages[i % len(stages)],
            source_id=f"{stages[i % len(stages)][-1]}_src{i % 2}",
            binding=f"ib_src{i % 2}",
            ordinal=i,
            rank=count - i,
            text=(
                "The quick brown fox jumps over the lazy dog while a tutorial paragraph "
                f"explains a concept step by step with examples number {i}."
            ),
        )
        for i in range(count)
    ]


def same_size_mutation(path: Path) -> None:
    """Flip one payload byte without changing the file length."""
    data = bytearray(Path(path).read_bytes())
    for index, byte in enumerate(data):
        if byte not in (0x0A, 0x22, 0x7B, 0x7D):
            data[index] = 0x41 if byte != 0x41 else 0x42
            break
    Path(path).write_bytes(bytes(data))


# --------------------------------------------------------------------- accepted G / G2 / L1

CANONICAL_EXCLUSION_HASHES = ["1" * 64, "2" * 64]
CANONICAL_CLEANING = {
    "strip_leading_noise": False,
    "normalize_quotes": False,
    "underscores_policy": "keep",
    "min_chars": 0,
    "min_ascii_ratio": 0.0,
}
L1_EXCLUSION_RELATIVE = (
    "runs/l1_production_2026-08-20/reference_reserve_v1/exclusion_hash_manifest.json"
)
G_MANIFEST_RELATIVE = "runs/g_production_2026-08-21/release/tokenizer_release_manifest.json"
G2_MANIFEST_RELATIVE = "runs/g2_production_2026-08-21/release/manifest.json"


def write_accepted_exclusion_authorities(
    root: Path, *, tokenizer_source: Path | None = None
) -> dict[str, Any]:
    """Write a synthetic canonical L1 artifact plus the accepted G and G2 manifests naming it.

    Mirrors the real shape: one L1 exclusion artifact, named identically by the accepted G
    tokenizer release and the accepted G2 reference release, with each manifest carrying its own
    independently derived hash count.
    """
    root = Path(root)
    artifact = root / L1_EXCLUSION_RELATIVE
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps({
            "schema_version": 1,
            "kind": "petitgpt_reference_validation_exclusions",
            "hash_algorithm": "sha256-cleaned-text-utf8-v1",
            "cleaning": CANONICAL_CLEANING,
            "hash_count": len(CANONICAL_EXCLUSION_HASHES),
            "hashes": CANONICAL_EXCLUSION_HASHES,
        }),
        encoding="utf-8",
    )
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    size = artifact.stat().st_size
    entry = {
        "enabled": True,
        "manifest_path": L1_EXCLUSION_RELATIVE,
        "manifest_resolved": str(artifact),
        "manifest_size_bytes": size,
        "manifest_sha256": digest,
        "kind": "petitgpt_reference_validation_exclusions",
        "hash_algorithm": "sha256-cleaned-text-utf8-v1",
        "hash_count": len(CANONICAL_EXCLUSION_HASHES),
        "cleaning": CANONICAL_CLEANING,
    }

    g = root / G_MANIFEST_RELATIVE
    g.parent.mkdir(parents=True, exist_ok=True)
    # R4-C: the accepted G release names its own canonical runtime artifact, and the canonical
    # tokenizer path is derived from that name relative to the manifest's own directory.
    tokenizer_path = g.parent / "tokenizer.json"
    if tokenizer_source is not None:
        tokenizer_path.write_bytes(Path(tokenizer_source).read_bytes())
    elif not tokenizer_path.exists():
        tokenizer_path.write_text("{}", encoding="utf-8")
    tokenizer_sha = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    g.write_text(
        json.dumps({
            "schema_version": 2,
            "kind": "petitgpt_tokenizer_release",
            "tokenizer_sha256": tokenizer_sha,
            "artifacts": {
                "canonical_runtime_artifact": "tokenizer.json",
                "authoritative_manifest": "tokenizer_release_manifest.json",
            },
            "reference_reserve_exclusion": {
                "enabled": True,
                "hash_algorithm": "sha256-cleaned-text-utf8-v1",
                "cleaning": CANONICAL_CLEANING,
                "manifest_count": 1,
                "union_hash_count": len(CANONICAL_EXCLUSION_HASHES),
                "manifests": [dict(entry)],
            },
        }),
        encoding="utf-8",
    )

    g2 = root / G2_MANIFEST_RELATIVE
    g2.parent.mkdir(parents=True, exist_ok=True)
    g2.write_text(
        json.dumps({
            "schema_version": 2,
            "kind": "petitgpt_cross_stage_reference_validation",
            "reserve_provenance": {"reserve_exclusion": dict(entry)},
        }),
        encoding="utf-8",
    )
    return {
        "artifact_path": L1_EXCLUSION_RELATIVE,
        "artifact_sha256": digest,
        "artifact_size_bytes": size,
        "artifact_schema_version": 1,
        "kind": "petitgpt_reference_validation_exclusions",
        "hash_algorithm": "sha256-cleaned-text-utf8-v1",
        "derived_count": len(CANONICAL_EXCLUSION_HASHES),
        "artifact_abs": artifact,
        "g_manifest": g,
        "g2_manifest": g2,
        "entry": entry,
        "tokenizer_path": str(tokenizer_path.relative_to(root)),
        "tokenizer_sha256": tokenizer_sha,
        "tokenizer_abs": tokenizer_path,
    }


def restore_canonical_exclusion_block(meta_path: Path, canonical: dict) -> None:
    """Rewrite one Stage-M release's exclusion block to name the canonical L1 artifact.

    tests/test_plan_pretrain_run.py::_write_full_provenance replaces this block with the legacy
    selector-v1 shape as a side effect of building the G/G2 fixture, so the native flow restores
    it. The two exclusion payloads are byte-identical, so only the block's shape is restored,
    never its identity.
    """
    payload = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    payload["reference_validation_exclusion"] = {
        "enabled": True,
        "manifest_count": 1,
        "union_hash_count": canonical["derived_count"],
        "enforced_at_stage": "stage_i",
        "reapplied_by_stage_m": False,
        "canonical_artifact_path": canonical["artifact_path"],
        "canonical_artifact_sha256": canonical["artifact_sha256"],
        "canonical_artifact_schema_version": canonical["artifact_schema_version"],
        "kind": canonical["kind"],
        "hash_algorithm": canonical["hash_algorithm"],
        "manifests": [
            {
                "enabled": True,
                "path": canonical["artifact_path"],
                "manifest_sha256": canonical["artifact_sha256"],
                "hash_count": canonical["derived_count"],
                "kind": canonical["kind"],
                "hash_algorithm": canonical["hash_algorithm"],
                "schema_version": canonical["artifact_schema_version"],
            }
        ],
    }
    Path(meta_path).write_text(json.dumps(payload), encoding="utf-8")
