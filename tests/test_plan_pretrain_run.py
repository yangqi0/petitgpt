from __future__ import annotations

import hashlib
import json
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from pretrain.dataset_pretrain import PackedBinDataset
from pretrain.plan_pretrain_run import (
    build_checkpoint_milestones,
    build_run_plan,
    main,
)
from pretrain.run_plan_contract import (
    load_run_plan_binding,
    validate_run_plan_dataset,
    validate_run_plan_validation_dataset,
)
from src.special_tokens import BOS_ID, EOS_ID, SPECIAL_TOKEN_IDS


def _shard_records(root, directory):
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "token_count": path.stat().st_size // np.dtype(np.uint16).itemsize,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(directory.glob("*.bin"))
    ]


def _write_stage(
    tmp_path,
    name: str,
    chunks: list[list[int]],
    *,
    exclusion_sha256: str = "c" * 64,
):
    root = tmp_path / name
    stage = root / "train"
    stage.mkdir(parents=True)
    val = root / "val"
    val.mkdir()
    for index, chunk in enumerate(chunks):
        np.asarray(chunk, dtype=np.uint16).tofile(stage / f"shard_{index:05d}.bin")
        np.asarray(chunk, dtype=np.uint16).tofile(val / f"shard_{index:05d}.bin")

    total_tokens = sum(len(chunk) for chunk in chunks)
    shard_tokens = max(len(chunk) for chunk in chunks)
    meta = {
        "schema_version": 3,
        "status": "complete",
        "dtype": "uint16",
        "vocab_size": 32_000,
        "tokenizer_sha256": "d" * 64,
        "contract": {
            "mode": "canonical",
            "canonical": True,
            "legacy_allow_noncanonical_contract": False,
            "issues": [],
            "expected_special_token_ids": dict(SPECIAL_TOKEN_IDS),
            "expected_vocab_size": 32_000,
            "actual_vocab_size": 32_000,
            "add_bos": True,
            "add_eos": True,
            "bos_id": BOS_ID,
            "eos_id": EOS_ID,
            "doc_sep": "",
        },
        "legacy_flags": {
            "allow_noncanonical_contract": False,
            "replay_on_exhaustion": False,
        },
        "source_exhaustion_policy": "fail_fast",
        "reference_validation_exclusion": {
            "enabled": True,
            "manifest_count": 1,
            "union_hash_count": 2,
            "manifests": [
                {
                    "enabled": True,
                    "manifest_sha256": exclusion_sha256,
                    "hash_count": 2,
                }
            ],
        },
        "shard_tokens": shard_tokens,
        "val_shard_tokens": shard_tokens,
        "train_shards": len(chunks),
        "train_tokens": total_tokens,
        "val_shards": len(chunks),
        "val_tokens": total_tokens,
        "accounting": {
            "train": {"emitted_shard_tokens": total_tokens},
            "val": {"emitted_shard_tokens": total_tokens},
        },
        "val_by_source": {},
        "shard_files": {
            "hash_algorithm": "sha256",
            "train": _shard_records(root, stage),
            "val": _shard_records(root, val),
            "val_by_source": {},
        },
    }
    (root / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return stage


def _write_full_provenance(
    tmp_path,
    stage_a,
    stage_b,
):
    cleaning = {
        "strip_leading_noise": False,
        "normalize_quotes": False,
        "underscores_policy": "keep",
        "min_chars": 0,
        "min_ascii_ratio": 0.0,
    }
    hash_algorithm = "sha256-cleaned-text-utf8-v1"
    exclusion_kind = "petitgpt_reference_validation_exclusions"
    reserved_hashes = ["1" * 64, "2" * 64]

    reserve_root = tmp_path / "reference_reserve"
    reserve_root.mkdir()
    reserve_exclusion = reserve_root / "reserve_exclusion.json"
    reserve_exclusion_payload = {
        "schema_version": 1,
        "kind": exclusion_kind,
        "hash_algorithm": hash_algorithm,
        "cleaning": cleaning,
        "hash_count": len(reserved_hashes),
        "hashes": reserved_hashes,
    }
    reserve_exclusion.write_text(
        json.dumps(reserve_exclusion_payload),
        encoding="utf-8",
    )
    exclusion_sha256 = hashlib.sha256(reserve_exclusion.read_bytes()).hexdigest()
    exclusion_size_bytes = reserve_exclusion.stat().st_size
    reserve_manifest = reserve_root / "reserve_manifest.json"
    reserve_manifest_payload = {
        "schema_version": 1,
        "status": "complete",
        "kind": "petitgpt_reference_validation_reserve",
        "immutable": True,
        "tokenizer_independent": True,
        "selection": {},
        "cleaning": cleaning,
        "sources": {
            "reference.jsonl": {
                "reserved_documents": [{"cleaned_text_sha256": value} for value in reserved_hashes]
            }
        },
        "outputs": {"exclusion_hash_manifest": reserve_exclusion.name},
        "unique_reserved_hashes": len(reserved_hashes),
    }
    reserve_manifest.write_text(json.dumps(reserve_manifest_payload), encoding="utf-8")
    reserve_sha256 = hashlib.sha256(reserve_manifest.read_bytes()).hexdigest()
    reserve_size_bytes = reserve_manifest.stat().st_size
    exclusion_metadata = {
        "enabled": True,
        "manifest_path": str(reserve_exclusion),
        "manifest_resolved": str(reserve_exclusion.resolve()),
        "manifest_sha256": exclusion_sha256,
        "manifest_size_bytes": exclusion_size_bytes,
        "kind": exclusion_kind,
        "hash_algorithm": hash_algorithm,
        "hash_count": len(reserved_hashes),
        "cleaning": cleaning,
    }

    tokenizer_root = tmp_path / "tokenizer_release"
    tokenizer_root.mkdir()
    tokenizer_json = tokenizer_root / "tokenizer.json"
    tokenizer_json.write_bytes(b"{}")
    tokenizer_sha256 = hashlib.sha256(tokenizer_json.read_bytes()).hexdigest()
    tokenizer_size_bytes = tokenizer_json.stat().st_size

    selection_root = tmp_path / "selection"
    selection_root.mkdir()
    selection_inputs_root = tmp_path / "selection_inputs"
    selection_inputs_root.mkdir()
    selection_sources = []
    for stage_name, stage_dir, source_id in (
        ("stage_a", stage_a, "source_a"),
        ("stage_b", stage_b, "source_b"),
    ):
        input_path = selection_inputs_root / f"{source_id}.jsonl"
        input_path.write_text(
            json.dumps({"text": f"{stage_name} candidate"}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
        input_size_bytes = input_path.stat().st_size

        relative = f"selected/{stage_name}/{source_id}.jsonl"
        output = selection_root / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "text": f"{stage_name} document",
                    "_petitgpt_selection": {
                        "stage": stage_name,
                        "source_id": source_id,
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        output_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
        output_stat = output.stat()
        selection_sources.append({
            "stage": stage_name,
            "source_id": source_id,
            "input_path": str(input_path),
            "input_sha256": input_sha256,
            "input_size_bytes": input_size_bytes,
            "output": {
                "relative_path": relative,
                "sha256": output_sha256,
                "size_bytes": output_stat.st_size,
                "documents": 1,
            },
        })

        meta_path = stage_dir.parent / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["tokenizer_sha256"] = tokenizer_sha256
        meta["reference_validation_exclusion"] = {
            "enabled": True,
            "hash_algorithm": hash_algorithm,
            "cleaning": cleaning,
            "manifest_count": 1,
            "union_hash_count": len(reserved_hashes),
            "manifests": [dict(exclusion_metadata)],
        }
        meta["sources"] = [{"path": str(output), "weight": 1.0}]
        meta["source_fingerprints"] = {
            str(output): {
                "path": str(output),
                "resolved": str(output.resolve()),
                "size": output_stat.st_size,
                "mtime": output_stat.st_mtime,
                "mtime_ns": output_stat.st_mtime_ns,
                "inode": output_stat.st_ino,
                "device": output_stat.st_dev,
                "sha256": output_sha256,
            }
        }
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

    tokenizer_release = {
        "schema_version": 2,
        "kind": "petitgpt_tokenizer_release",
        "status": "complete",
        "contract": {
            "canonical": True,
            "issues": [],
            "legacy_allow_noncanonical_contract": False,
        },
        "publication": "sibling_staging_then_atomic_rename",
        "tokenizer_sha256": tokenizer_sha256,
        "vocab_size": 32_000,
        "special_token_ids": dict(SPECIAL_TOKEN_IDS),
        "training": {
            "vocab_size_target": 32_000,
            "add_prefix_space": False,
            "post_processor_enabled": False,
        },
        "reference_reserve_exclusion": {
            "enabled": True,
            "hash_algorithm": hash_algorithm,
            "cleaning": cleaning,
            "manifest_count": 1,
            "union_hash_count": len(reserved_hashes),
            "manifests": [dict(exclusion_metadata)],
        },
    }
    tokenizer_manifest = tokenizer_root / "tokenizer_release_manifest.json"
    tokenizer_manifest.write_text(json.dumps(tokenizer_release), encoding="utf-8")

    selection_spec = tmp_path / "selection_spec.json"
    selection_spec.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    spec_sha256 = hashlib.sha256(selection_spec.read_bytes()).hexdigest()
    spec_size_bytes = selection_spec.stat().st_size

    audit_path = selection_root / "audit_exact_intersections.json"
    audit_payload = {
        "schema_version": 2,
        "status": "passed",
        "all_exact_intersections_zero": True,
        "identity_algorithms": {
            "raw_sha256": "sha256-raw-text-utf8-v1",
            "cleaned_sha256": hash_algorithm,
            "canonical_fingerprint": "sha256-domain-separated-cleaned-text-v1",
        },
        "pairwise_sources": [
            {
                "left_source_id": "source_a",
                "left_stage": "stage_a",
                "right_source_id": "source_b",
                "right_stage": "stage_b",
                "intersection_counts": {
                    "raw_sha256": 0,
                    "cleaned_sha256": 0,
                    "canonical_fingerprint": 0,
                },
            }
        ],
        "reference_validation": {
            "selected_reference_intersection": 0,
            "intersection_zero": True,
        },
    }
    audit_path.write_text(json.dumps(audit_payload), encoding="utf-8")
    audit_sha256 = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    audit_size_bytes = audit_path.stat().st_size

    database_path = selection_root / "selection_registry.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE sources (
                source_id TEXT PRIMARY KEY,
                stage TEXT NOT NULL,
                output_relative_path TEXT NOT NULL,
                output_sha256 TEXT NOT NULL,
                output_size_bytes INTEGER NOT NULL
            );
            CREATE TABLE documents (
                canonical_fingerprint TEXT NOT NULL,
                cleaned_sha256 TEXT NOT NULL,
                raw_sha256 TEXT NOT NULL,
                source_id TEXT NOT NULL,
                committed INTEGER NOT NULL
            );
            CREATE VIEW selections AS
                SELECT * FROM documents WHERE committed = 1;
            CREATE VIEW candidates AS
                SELECT * FROM documents WHERE committed = 0;
            CREATE TABLE reference_exclusion_hashes (
                cleaned_sha256 TEXT PRIMARY KEY
            );
            """
        )
        for index, source in enumerate(selection_sources, start=3):
            output = source["output"]
            connection.execute(
                "INSERT INTO sources VALUES (?, ?, ?, ?, ?)",
                (
                    source["source_id"],
                    source["stage"],
                    output["relative_path"],
                    output["sha256"],
                    output["size_bytes"],
                ),
            )
            connection.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, 1)",
                (
                    f"{index}" * 64,
                    f"{index + 2}" * 64,
                    f"{index + 4}" * 64,
                    source["source_id"],
                ),
            )
        connection.executemany(
            "INSERT INTO reference_exclusion_hashes VALUES (?)",
            [(value,) for value in reserved_hashes],
        )
    database_sha256 = hashlib.sha256(database_path.read_bytes()).hexdigest()
    database_size_bytes = database_path.stat().st_size

    selection_manifest_payload = {
        "schema_version": 2,
        "kind": "petitgpt_pretrain_document_selection",
        "status": "complete",
        "publication": {"mode": "sibling_staging_then_atomic_rename"},
        "reference_validation_exclusion": {
            "required": True,
            "kind": exclusion_kind,
            "hash_algorithm": hash_algorithm,
            "cleaning": cleaning,
            "manifest_count": 1,
            "union_hash_count": len(reserved_hashes),
            "cross_manifest_duplicate_memberships": 0,
            "selected_reference_intersection": 0,
            "intersection_zero": True,
            "manifests": [
                {
                    "path": str(reserve_exclusion),
                    "sha256": exclusion_sha256,
                    "size_bytes": exclusion_size_bytes,
                    "kind": exclusion_kind,
                    "hash_algorithm": hash_algorithm,
                    "hash_count": len(reserved_hashes),
                    "matched_documents": 0,
                    "matched_per_source": {},
                }
            ],
        },
        "tokenizer": {
            "path": str(tokenizer_json),
            "sha256": tokenizer_sha256,
            "size_bytes": tokenizer_size_bytes,
            "vocab_size": 32_000,
            "special_token_ids": dict(SPECIAL_TOKEN_IDS),
            "automatic_bos_eos": False,
            "literal_special_tokens_encoded_as_text": True,
        },
        "spec": {
            "path": str(selection_spec),
            "sha256": spec_sha256,
            "size_bytes": spec_size_bytes,
            "schema_version": 1,
        },
        "audit": {
            "relative_path": audit_path.name,
            "sha256": audit_sha256,
            "size_bytes": audit_size_bytes,
            "all_exact_intersections_zero": True,
            "selected_reference_intersection": 0,
        },
        "database": {
            "relative_path": database_path.name,
            "sha256": database_sha256,
            "size_bytes": database_size_bytes,
            "selected_document_rows": len(selection_sources),
            "integrity_check": "ok",
            "uncommitted_candidate_rows": 0,
        },
        "sources": selection_sources,
    }
    selection_manifest = selection_root / "manifest.json"
    selection_manifest.write_text(json.dumps(selection_manifest_payload), encoding="utf-8")

    reference_root = tmp_path / "reference"
    reference_val = reference_root / "val"
    reference_val.mkdir(parents=True)
    np.asarray([BOS_ID, 7, 8, EOS_ID], dtype=np.uint16).tofile(reference_val / "shard_00000.bin")
    stage_contract = json.loads((stage_a.parent / "meta.json").read_text(encoding="utf-8"))[
        "contract"
    ]
    reference_manifest_payload = {
        "schema_version": 2,
        "kind": "petitgpt_cross_stage_reference_validation",
        "status": "complete",
        "immutable": True,
        "tokenizer_sha256": tokenizer_sha256,
        "vocab_size": 32_000,
        "dtype": "uint16",
        "contract": stage_contract,
        "selection": {"restricted_to_pre_tokenizer_reserve": True},
        "reserve_provenance": {
            "reserve_manifest_path": str(reserve_manifest),
            "reserve_manifest_sha256": reserve_sha256,
            "reserve_manifest_size_bytes": reserve_size_bytes,
            "reserve_exclusion": dict(exclusion_metadata),
            "selection": {},
            "cleaning": cleaning,
        },
        "packing": {"shard_tokens": 4},
        "accounting": {
            "serialized_tokens": 4,
            "emitted_shard_tokens": 4,
            "combined_shards": 1,
        },
        "outputs": {
            "combined_val": "val",
            "val_by_source": "val_by_source",
        },
        "sources": {},
        "shard_files": {
            "hash_algorithm": "sha256",
            "val": _shard_records(reference_root, reference_val),
            "val_by_source": {},
        },
    }
    reference_manifest = reference_root / "manifest.json"
    reference_manifest.write_text(json.dumps(reference_manifest_payload), encoding="utf-8")
    return {
        "reference_val_dir": reference_val,
        "tokenizer_release_manifest": tokenizer_manifest,
        "selection_manifest": selection_manifest,
        "tokenizer_sha256": tokenizer_sha256,
    }


def _refresh_selection_evidence_for_source(
    selection_manifest,
    manifest,
    *,
    source_id,
):
    source = next(item for item in manifest["sources"] if item["source_id"] == source_id)

    audit_path = selection_manifest.parent / manifest["audit"]["relative_path"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    for pair in audit["pairwise_sources"]:
        if pair["left_source_id"] == source_id:
            pair["left_stage"] = source["stage"]
        if pair["right_source_id"] == source_id:
            pair["right_stage"] = source["stage"]
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    manifest["audit"]["sha256"] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    manifest["audit"]["size_bytes"] = audit_path.stat().st_size

    database_path = selection_manifest.parent / manifest["database"]["relative_path"]
    output = source["output"]
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            UPDATE sources
            SET stage = ?, output_relative_path = ?, output_sha256 = ?,
                output_size_bytes = ?
            WHERE source_id = ?
            """,
            (
                source["stage"],
                output["relative_path"],
                output["sha256"],
                output["size_bytes"],
                source_id,
            ),
        )
    manifest["database"]["sha256"] = hashlib.sha256(database_path.read_bytes()).hexdigest()
    manifest["database"]["size_bytes"] = database_path.stat().st_size


def _build_full_plan(stage_a, stage_b, provenance, *, stage_b_selection_stage="stage_b"):
    return build_run_plan(
        stage_a_dir=stage_a,
        stage_b_dir=stage_b,
        seq_len=2,
        micro_bsz=1,
        grad_accum=2,
        warmup_steps=1,
        decay_fraction=0.5,
        stage_b_selection_stage=stage_b_selection_stage,
        reference_val_dir=provenance["reference_val_dir"],
        tokenizer_release_manifest=provenance["tokenizer_release_manifest"],
        selection_manifest=provenance["selection_manifest"],
    )


def test_exact_step_boundaries_and_no_replacement_accounting(tmp_path):
    # A: 26 tokens -> 25 transitions -> 6 blocks + one global tail.
    stage_a = _write_stage(tmp_path, "a", [list(range(13)), list(range(13, 26))])
    # B: 22 tokens -> 21 transitions -> 5 blocks + one global tail.
    stage_b = _write_stage(tmp_path, "b", [list(range(22))])

    plan = build_run_plan(
        stage_a_dir=stage_a,
        stage_b_dir=stage_b,
        seq_len=4,
        micro_bsz=1,
        grad_accum=2,
        warmup_steps=1,
        decay_fraction=0.4,
    )

    assert plan["batch"] == {
        "sequences_per_optimizer_step": 2,
        "serialized_target_positions_per_optimizer_step": 8,
    }
    assert plan["boundaries"] == {
        "stage_a_start_step": 0,
        "stage_a_stop_step": 3,
        "stage_b_start_step": 3,
        "stage_b_global_stop_step": 5,
        "schedule_total_steps": 5,
    }
    assert plan["wsd_candidate"]["decay_steps"] == 2
    assert plan["wsd_candidate"]["decay_start_step"] == 3
    assert plan["wsd_candidate"]["decay_end_step"] == 5

    a = plan["stages"]["stage_a"]
    assert a["full_blocks"] == a["consumed_blocks"] == 6
    assert a["dropped_batch_alignment_blocks"] == 0
    assert a["consumed_serialized_target_positions"] == 24
    assert a["global_tail_transitions"] == 1
    assert a["unconsumed_transitions_total"] == 1
    assert a["block_coverage_fraction"] == 1.0

    b = plan["stages"]["stage_b"]
    assert b["full_blocks"] == 5
    assert b["consumed_blocks"] == 4
    assert b["dropped_batch_alignment_blocks"] == 1
    assert b["dropped_batch_alignment_transitions"] == 4
    assert b["global_tail_transitions"] == 1
    assert b["unconsumed_transitions_total"] == 5
    assert b["block_coverage_fraction"] == pytest.approx(0.8)
    assert b["total_transition_coverage_fraction"] == pytest.approx(16 / 21)

    assert plan["totals"]["consumed_blocks"] == 10
    assert plan["totals"]["dropped_batch_alignment_blocks"] == 1
    assert plan["totals"]["dropped_batch_alignment_transitions"] == 4
    assert plan["totals"]["global_tail_transitions"] == 2
    assert plan["totals"]["consumed_serialized_target_positions"] == 40
    assert plan["invariants"]["implicit_replay"] is False
    assert len(a["dataset"]["release_validation"]["manifest_sha256"]) == 64
    assert len(b["dataset"]["release_validation"]["manifest_sha256"]) == 64
    provenance = plan["release_provenance"]
    assert provenance["shared_reference_exclusion_manifest_sha256s"] == ["c" * 64]
    assert len(provenance["stage_a"]["manifest_sha256"]) == 64
    assert len(provenance["stage_b"]["manifest_sha256"]) == 64

    milestones = plan["checkpoint_milestones"]
    assert milestones["absolute_steps"] == [3, 4, 5]
    assert milestones["cli_save_steps"] == "3,4,5"
    assert len(milestones["entries"]) == len(milestones["absolute_steps"])
    reasons_at_step = {
        entry["absolute_step"]: set(entry["reasons"]) for entry in milestones["entries"]
    }
    assert reasons_at_step[3] >= {
        "stage_a_end",
        "stage_a_exposure_1_end",
        "wsd_decay_start",
    }
    assert reasons_at_step[4] == {"stage_b_midpoint"}
    assert reasons_at_step[5] >= {"nominal_1b_tokens", "stage_b_end"}
    assert all(
        target["horizon_clamped"] for target in milestones["nominal_cumulative_token_targets"]
    )


def test_rejects_different_stage_reference_exclusion_sha_sets(tmp_path):
    stage_a = _write_stage(
        tmp_path,
        "a",
        [list(range(17))],
        exclusion_sha256="a" * 64,
    )
    stage_b = _write_stage(
        tmp_path,
        "b",
        [list(range(17))],
        exclusion_sha256="b" * 64,
    )

    with pytest.raises(RuntimeError, match="different reference exclusion"):
        build_run_plan(
            stage_a_dir=stage_a,
            stage_b_dir=stage_b,
            seq_len=2,
            micro_bsz=1,
            grad_accum=2,
            warmup_steps=1,
            decay_fraction=0.5,
        )


def test_milestone_helper_rounds_up_and_deduplicates_all_reasons():
    milestones = build_checkpoint_milestones(
        stage_specs=(
            {
                "name": "stage_a",
                "start_step": 0,
                "planned_optimizer_steps": 4,
                "unique_blocks": 8,
                "completed_full_exposures": 1,
            },
            {
                "name": "stage_b",
                "start_step": 4,
                "planned_optimizer_steps": 4,
                "unique_blocks": 8,
                "completed_full_exposures": 1,
            },
        ),
        sequences_per_optimizer_step=2,
        consumed_transitions_per_optimizer_step=10,
        decay_start_step=4,
        nominal_token_targets=(
            ("nominal_10", 10),
            ("nominal_21", 21),
            ("nominal_40", 40),
            ("nominal_100", 100),
        ),
    )

    assert milestones["absolute_steps"] == [1, 3, 4, 6, 8]
    records = {
        target["reason"]: target for target in milestones["nominal_cumulative_token_targets"]
    }
    assert records["nominal_10"]["absolute_step"] == 1
    assert records["nominal_21"]["absolute_step"] == 3
    assert records["nominal_21"]["delta_consumed_transitions"] == 9
    assert records["nominal_40"]["absolute_step"] == 4
    assert records["nominal_100"]["absolute_step"] == 8
    assert records["nominal_100"]["horizon_clamped"] is True
    assert records["nominal_100"]["delta_consumed_transitions"] == -20

    entries = {entry["absolute_step"]: entry for entry in milestones["entries"]}
    assert set(entries[4]["reasons"]) == {
        "nominal_40",
        "stage_a_end",
        "stage_a_exposure_1_end",
        "wsd_decay_start",
    }
    assert set(entries[8]["reasons"]) == {
        "nominal_100",
        "stage_b_end",
        "stage_b_exposure_1_end",
    }
    assert milestones["exposure_epoch_endpoints"] == [
        {
            "reason": "stage_a_exposure_1_end",
            "stage": "stage_a",
            "exposure_index": 1,
            "boundary_consumed_blocks": 8,
            "absolute_step": 4,
            "actual_stage_consumed_blocks": 8,
            "optimizer_batch_overshoot_blocks": 0,
        },
        {
            "reason": "stage_b_exposure_1_end",
            "stage": "stage_b",
            "exposure_index": 1,
            "boundary_consumed_blocks": 8,
            "absolute_step": 8,
            "actual_stage_consumed_blocks": 8,
            "optimizer_batch_overshoot_blocks": 0,
        },
    ]


def test_explicit_multi_exposure_plan_floors_once_per_expanded_stage(tmp_path):
    stage_a = _write_stage(tmp_path, "a", [list(range(13)), list(range(13, 26))])
    stage_b = _write_stage(tmp_path, "b", [list(range(22))])

    plan = build_run_plan(
        stage_a_dir=stage_a,
        stage_b_dir=stage_b,
        seq_len=4,
        micro_bsz=1,
        grad_accum=2,
        warmup_steps=1,
        decay_fraction=0.4,
        stage_a_exposures=2,
        stage_b_exposures=3,
    )

    assert plan["plan_type"] == "deterministic_explicit_multi_exposure_stage_a_b"
    assert plan["invariants"]["implicit_replay"] is False
    assert plan["invariants"]["explicit_replay"] is True
    assert plan["boundaries"] == {
        "stage_a_start_step": 0,
        "stage_a_stop_step": 6,
        "stage_b_start_step": 6,
        "stage_b_global_stop_step": 13,
        "schedule_total_steps": 13,
    }
    assert plan["wsd_candidate"]["decay_start_step"] == 7

    stage_a_plan = plan["stages"]["stage_a"]
    assert stage_a_plan["unique_blocks"] == 6
    assert stage_a_plan["candidate_exposure_blocks"] == 12
    assert stage_a_plan["consumed_replay_blocks"] == 6
    assert stage_a_plan["completed_full_exposures"] == 2
    assert stage_a_plan["partial_exposure_blocks"] == 0

    stage_b_plan = plan["stages"]["stage_b"]
    assert stage_b_plan["unique_blocks"] == 5
    assert stage_b_plan["candidate_exposure_blocks"] == 15
    assert stage_b_plan["consumed_exposure_blocks"] == 14
    assert stage_b_plan["dropped_batch_alignment_blocks"] == 1
    assert stage_b_plan["consumed_replay_blocks"] == 9
    assert stage_b_plan["completed_full_exposures"] == 2
    assert stage_b_plan["partial_exposure_blocks"] == 4
    assert stage_b_plan["realized_mean_exposures_per_unique_block"] == pytest.approx(2.8)

    totals = plan["totals"]
    assert totals["unique_blocks"] == 11
    assert totals["candidate_exposure_blocks"] == 27
    assert totals["consumed_exposure_blocks"] == 26
    assert totals["planned_replay_blocks"] == 16
    assert totals["consumed_replay_blocks"] == 15
    assert totals["unique_block_coverage_fraction"] == 1.0
    assert totals["exposure_block_coverage_fraction"] == pytest.approx(26 / 27)
    assert totals["unconsumed_transitions_total"] == 9

    milestones = plan["checkpoint_milestones"]
    assert milestones["absolute_steps"] == [3, 6, 7, 9, 10, 11, 13]
    assert milestones["absolute_steps"] == sorted(set(milestones["absolute_steps"]))
    exposure_endpoints = {
        endpoint["reason"]: endpoint for endpoint in milestones["exposure_epoch_endpoints"]
    }
    assert set(exposure_endpoints) == {
        "stage_a_exposure_1_end",
        "stage_a_exposure_2_end",
        "stage_b_exposure_1_end",
        "stage_b_exposure_2_end",
    }
    assert exposure_endpoints["stage_a_exposure_1_end"]["absolute_step"] == 3
    assert exposure_endpoints["stage_a_exposure_2_end"]["absolute_step"] == 6
    assert exposure_endpoints["stage_b_exposure_1_end"] == {
        "reason": "stage_b_exposure_1_end",
        "stage": "stage_b",
        "exposure_index": 1,
        "boundary_consumed_blocks": 5,
        "absolute_step": 9,
        "actual_stage_consumed_blocks": 6,
        "optimizer_batch_overshoot_blocks": 1,
    }
    assert exposure_endpoints["stage_b_exposure_2_end"]["absolute_step"] == 11
    entry_at_stage_a_end = next(
        entry for entry in milestones["entries"] if entry["absolute_step"] == 6
    )
    assert set(entry_at_stage_a_end["reasons"]) >= {
        "stage_a_end",
        "stage_a_exposure_2_end",
    }
    assert plan["boundaries"]["stage_a_stop_step"] == 6
    assert plan["boundaries"]["stage_b_global_stop_step"] == 13


def test_cli_atomically_writes_machine_readable_plan(tmp_path, capsys, monkeypatch):
    stage_a = _write_stage(tmp_path, "a", [list(range(17))])
    stage_b = _write_stage(tmp_path, "b", [list(range(17))])
    provenance = _write_full_provenance(tmp_path, stage_a, stage_b)
    monkeypatch.setattr("pretrain.plan_pretrain_run.assert_tokenizer_contract", lambda _path: None)
    output = tmp_path / "nested" / "run_plan.json"

    assert (
        main([
            "--stage_a_dir",
            str(stage_a),
            "--stage_b_dir",
            str(stage_b),
            "--seq_len",
            "2",
            "--micro_bsz",
            "1",
            "--grad_accum",
            "2",
            "--warmup_steps",
            "1",
            "--decay_fraction",
            "0.5",
            "--reference_val_dir",
            str(provenance["reference_val_dir"]),
            "--tokenizer_release_manifest",
            str(provenance["tokenizer_release_manifest"]),
            "--selection_manifest",
            str(provenance["selection_manifest"]),
            "--out_json",
            str(output),
        ])
        == 0
    )

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 3
    assert saved["boundaries"]["stage_a_stop_step"] == 4
    assert saved["plan_type"] == "deterministic_no_replacement_stage_a_b"
    assert saved["inputs"]["stage_a_exposures"] == 1
    assert saved["inputs"]["stage_b_exposures"] == 1
    assert saved["boundaries"]["stage_b_global_stop_step"] == 8
    release = saved["release_provenance"]
    assert release["full_chain_validated"] is True
    assert release["shared_tokenizer_sha256"] == provenance["tokenizer_sha256"]
    for artifact in ("reference_validation", "tokenizer_release", "selection"):
        assert len(release[artifact]["manifest_sha256"]) == 64
    assert len(release["reference_validation"]["reserve_manifest_sha256"]) == 64
    assert release["source_bindings"]["validated"] is True
    assert len(release["source_bindings"]["stage_a"]) == 1
    assert len(release["source_bindings"]["stage_b"]) == 1
    assert saved["checkpoint_milestones"]["cli_save_steps"] == "4,6,8"
    launch_args = SimpleNamespace(
        run_plan_json=str(output),
        run_plan_stage="stage_a",
        strict_resume_contract=True,
        seq_len=2,
        micro_bsz=1,
        grad_accum=2,
        warmup_steps=1,
        data_stage_start_step=0,
        max_steps=4,
        schedule_total_steps=8,
        lr_schedule="wsd",
        decay_start_step=4,
        decay_end_step=8,
        allow_schedule_branch=False,
        save_steps=[4, 6, 8],
        val_dir=str(provenance["reference_val_dir"]),
    )
    binding = load_run_plan_binding(
        launch_args,
        train_dir=stage_a,
        val_dir=provenance["reference_val_dir"],
        tokenizer_sha256=provenance["tokenizer_sha256"],
    )
    assert binding is not None
    train_ds = PackedBinDataset(stage_a, seq_len=2, require_release_manifest=True)
    reference_ds = PackedBinDataset(
        provenance["reference_val_dir"], seq_len=2, require_release_manifest=True
    )
    validate_run_plan_dataset(binding, train_ds)
    validate_run_plan_validation_dataset(binding, reference_ds)
    stdout = capsys.readouterr().out
    assert "A_stop=4 B_stop=8 decay=4:8" in stdout
    assert "save_steps=4,6,8" in stdout
    assert not list(output.parent.glob(".*.tmp"))


def test_full_provenance_inputs_are_all_or_none(tmp_path):
    stage_a = _write_stage(tmp_path, "a", [list(range(17))])
    stage_b = _write_stage(tmp_path, "b", [list(range(17))])

    with pytest.raises(ValueError, match="must be supplied together"):
        build_run_plan(
            stage_a_dir=stage_a,
            stage_b_dir=stage_b,
            seq_len=2,
            micro_bsz=1,
            grad_accum=2,
            warmup_steps=1,
            decay_fraction=0.5,
            reference_val_dir=tmp_path / "reference" / "val",
        )


def test_full_provenance_rejects_exclusion_sha_mismatch(tmp_path, monkeypatch):
    stage_a = _write_stage(tmp_path, "a", [list(range(17))])
    stage_b = _write_stage(tmp_path, "b", [list(range(17))])
    provenance = _write_full_provenance(tmp_path, stage_a, stage_b)
    monkeypatch.setattr("pretrain.plan_pretrain_run.assert_tokenizer_contract", lambda _path: None)
    path = provenance["selection_manifest"]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["reference_validation_exclusion"]["manifests"][0]["sha256"] = "a" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="SHA-256 disagrees"):
        _build_full_plan(stage_a, stage_b, provenance)


def test_full_provenance_rejects_tokenizer_sha_mismatch(tmp_path, monkeypatch):
    stage_a = _write_stage(tmp_path, "a", [list(range(17))])
    stage_b = _write_stage(tmp_path, "b", [list(range(17))])
    provenance = _write_full_provenance(tmp_path, stage_a, stage_b)
    monkeypatch.setattr("pretrain.plan_pretrain_run.assert_tokenizer_contract", lambda _path: None)
    path = provenance["selection_manifest"]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["tokenizer"]["sha256"] = "a" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="selection tokenizer SHA-256 disagrees"):
        _build_full_plan(stage_a, stage_b, provenance)


def test_full_provenance_rejects_source_sha_mismatch(tmp_path, monkeypatch):
    stage_a = _write_stage(tmp_path, "a", [list(range(17))])
    stage_b = _write_stage(tmp_path, "b", [list(range(17))])
    provenance = _write_full_provenance(tmp_path, stage_a, stage_b)
    monkeypatch.setattr("pretrain.plan_pretrain_run.assert_tokenizer_contract", lambda _path: None)
    meta_path = stage_a.parent / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    next(iter(meta["source_fingerprints"].values()))["sha256"] = "a" * 64
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(RuntimeError, match="source SHA-256 disagrees"):
        _build_full_plan(stage_a, stage_b, provenance)


@pytest.mark.parametrize("failure_mode", ["legacy", "missing"])
def test_full_provenance_rejects_legacy_or_missing_selection_manifest(
    tmp_path,
    monkeypatch,
    failure_mode,
):
    stage_a = _write_stage(tmp_path, "a", [list(range(17))])
    stage_b = _write_stage(tmp_path, "b", [list(range(17))])
    provenance = _write_full_provenance(tmp_path, stage_a, stage_b)
    monkeypatch.setattr("pretrain.plan_pretrain_run.assert_tokenizer_contract", lambda _path: None)
    path = provenance["selection_manifest"]
    if failure_mode == "legacy":
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["schema_version"] = 0
        path.write_text(json.dumps(manifest), encoding="utf-8")
        expected_error = RuntimeError
        message = "schema_version"
    else:
        path.unlink()
        expected_error = FileNotFoundError
        message = "selection manifest"

    with pytest.raises(expected_error, match=message):
        _build_full_plan(stage_a, stage_b, provenance)


def test_full_provenance_rejects_missing_audit_evidence(tmp_path, monkeypatch):
    stage_a = _write_stage(tmp_path, "a", [list(range(17))])
    stage_b = _write_stage(tmp_path, "b", [list(range(17))])
    provenance = _write_full_provenance(tmp_path, stage_a, stage_b)
    monkeypatch.setattr("pretrain.plan_pretrain_run.assert_tokenizer_contract", lambda _path: None)
    selection_path = provenance["selection_manifest"]
    manifest = json.loads(selection_path.read_text(encoding="utf-8"))
    (selection_path.parent / manifest["audit"]["relative_path"]).unlink()

    with pytest.raises(FileNotFoundError, match="exact-intersection audit"):
        _build_full_plan(stage_a, stage_b, provenance)


def test_full_provenance_rejects_same_size_selected_output_mutation(tmp_path, monkeypatch):
    stage_a = _write_stage(tmp_path, "a", [list(range(17))])
    stage_b = _write_stage(tmp_path, "b", [list(range(17))])
    provenance = _write_full_provenance(tmp_path, stage_a, stage_b)
    monkeypatch.setattr("pretrain.plan_pretrain_run.assert_tokenizer_contract", lambda _path: None)
    selection_path = provenance["selection_manifest"]
    manifest = json.loads(selection_path.read_text(encoding="utf-8"))
    relative = manifest["sources"][0]["output"]["relative_path"]
    selected_output = selection_path.parent / relative
    original = selected_output.read_bytes()
    mutated = bytes([original[0] ^ 1]) + original[1:]
    assert len(mutated) == len(original)
    selected_output.write_bytes(mutated)

    with pytest.raises(RuntimeError, match="output SHA-256 disagrees"):
        _build_full_plan(stage_a, stage_b, provenance)


def test_full_provenance_rejects_recorded_database_size_mismatch(tmp_path, monkeypatch):
    stage_a = _write_stage(tmp_path, "a", [list(range(17))])
    stage_b = _write_stage(tmp_path, "b", [list(range(17))])
    provenance = _write_full_provenance(tmp_path, stage_a, stage_b)
    monkeypatch.setattr("pretrain.plan_pretrain_run.assert_tokenizer_contract", lambda _path: None)
    selection_path = provenance["selection_manifest"]
    manifest = json.loads(selection_path.read_text(encoding="utf-8"))
    manifest["database"]["size_bytes"] += 1
    selection_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="database size disagrees"):
        _build_full_plan(stage_a, stage_b, provenance)


def test_full_provenance_recomputes_audit_intersection_counts(tmp_path, monkeypatch):
    stage_a = _write_stage(tmp_path, "a", [list(range(17))])
    stage_b = _write_stage(tmp_path, "b", [list(range(17))])
    provenance = _write_full_provenance(tmp_path, stage_a, stage_b)
    monkeypatch.setattr("pretrain.plan_pretrain_run.assert_tokenizer_contract", lambda _path: None)
    selection_path = provenance["selection_manifest"]
    manifest = json.loads(selection_path.read_text(encoding="utf-8"))
    audit_path = selection_path.parent / manifest["audit"]["relative_path"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["pairwise_sources"][0]["intersection_counts"]["raw_sha256"] = 1
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    manifest["audit"]["sha256"] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    manifest["audit"]["size_bytes"] = audit_path.stat().st_size
    selection_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="non-zero exact intersection"):
        _build_full_plan(stage_a, stage_b, provenance)


def test_full_provenance_recomputes_sqlite_reference_intersection(tmp_path, monkeypatch):
    stage_a = _write_stage(tmp_path, "a", [list(range(17))])
    stage_b = _write_stage(tmp_path, "b", [list(range(17))])
    provenance = _write_full_provenance(tmp_path, stage_a, stage_b)
    monkeypatch.setattr("pretrain.plan_pretrain_run.assert_tokenizer_contract", lambda _path: None)
    selection_path = provenance["selection_manifest"]
    manifest = json.loads(selection_path.read_text(encoding="utf-8"))
    database_path = selection_path.parent / manifest["database"]["relative_path"]
    with sqlite3.connect(database_path) as connection:
        selected_hash = connection.execute(
            "SELECT cleaned_sha256 FROM selections LIMIT 1"
        ).fetchone()[0]
        connection.execute(
            "INSERT INTO reference_exclusion_hashes VALUES (?)",
            (selected_hash,),
        )
    manifest["database"]["sha256"] = hashlib.sha256(database_path.read_bytes()).hexdigest()
    manifest["database"]["size_bytes"] = database_path.stat().st_size
    selection_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="intersects reference exclusion"):
        _build_full_plan(stage_a, stage_b, provenance)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"seq_len": 0}, "seq_len must be positive"),
        ({"micro_bsz": 0}, "micro_bsz must be positive"),
        ({"grad_accum": -1}, "grad_accum must be positive"),
        ({"warmup_steps": -1}, "warmup_steps must be >= 0"),
        ({"decay_fraction": 0.0}, "decay_fraction"),
        ({"decay_fraction": float("nan")}, "decay_fraction"),
        ({"stage_a_exposures": 0}, "stage_a_exposures must be positive"),
        ({"stage_b_exposures": -1}, "stage_b_exposures must be positive"),
    ],
)
def test_rejects_invalid_numeric_inputs(tmp_path, override, message):
    stage_a = _write_stage(tmp_path, "a", [list(range(33))])
    stage_b = _write_stage(tmp_path, "b", [list(range(33))])
    kwargs = dict(
        stage_a_dir=stage_a,
        stage_b_dir=stage_b,
        seq_len=2,
        micro_bsz=1,
        grad_accum=2,
        warmup_steps=1,
        decay_fraction=0.5,
    )
    kwargs.update(override)
    with pytest.raises(ValueError, match=message):
        build_run_plan(**kwargs)


def test_rejects_partial_optimizer_step_only_stage(tmp_path):
    stage_a = _write_stage(tmp_path, "a", [list(range(8))])
    stage_b = _write_stage(tmp_path, "b", [list(range(20))])
    with pytest.raises(ValueError, match="fewer than one optimizer step"):
        build_run_plan(
            stage_a_dir=stage_a,
            stage_b_dir=stage_b,
            seq_len=4,
            micro_bsz=2,
            grad_accum=1,
            warmup_steps=0,
            decay_fraction=0.5,
        )


def test_rejects_wsd_decay_that_intrudes_into_stage_a(tmp_path):
    stage_a = _write_stage(tmp_path, "a", [list(range(33))])
    stage_b = _write_stage(tmp_path, "b", [list(range(17))])
    with pytest.raises(ValueError, match="before Stage B"):
        build_run_plan(
            stage_a_dir=stage_a,
            stage_b_dir=stage_b,
            seq_len=2,
            micro_bsz=1,
            grad_accum=2,
            warmup_steps=1,
            decay_fraction=0.75,
        )


def test_rejects_unknown_stage_b_selection_cohort(tmp_path):
    stage_a = _write_stage(tmp_path, "a", [list(range(17))])
    stage_b = _write_stage(tmp_path, "b", [list(range(17))])
    with pytest.raises(ValueError, match="stage_b_selection_stage"):
        build_run_plan(
            stage_a_dir=stage_a,
            stage_b_dir=stage_b,
            seq_len=2,
            micro_bsz=1,
            grad_accum=2,
            warmup_steps=1,
            decay_fraction=0.5,
            stage_b_selection_stage="bogus",
        )


def test_control_selection_cohort_binds_stage_b_source_bytes(tmp_path, monkeypatch):
    stage_a = _write_stage(tmp_path, "a", [list(range(17))])
    stage_b = _write_stage(tmp_path, "b", [list(range(17))])
    provenance = _write_full_provenance(tmp_path, stage_a, stage_b)
    monkeypatch.setattr("pretrain.plan_pretrain_run.assert_tokenizer_contract", lambda _path: None)
    path = provenance["selection_manifest"]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    premium = next(source for source in manifest["sources"] if source["stage"] == "stage_b")
    premium["stage"] = "control"
    _refresh_selection_evidence_for_source(path, manifest, source_id=premium["source_id"])
    path.write_text(json.dumps(manifest), encoding="utf-8")

    plan = _build_full_plan(stage_a, stage_b, provenance, stage_b_selection_stage="control")

    assert plan["inputs"]["stage_b_selection_stage"] == "control"
    release = plan["release_provenance"]
    assert release["stage_b_selection_stage"] == "control"
    assert release["selection"]["stage_b_selection_stage"] == "control"
    assert release["source_bindings"]["stage_b_selection_stage"] == "control"
    assert release["source_bindings"]["stage_b"][0]["source_id"] == "source_b"


def test_control_selection_cohort_rejects_stage_b_source_mismatch(tmp_path, monkeypatch):
    stage_a = _write_stage(tmp_path, "a", [list(range(17))])
    stage_b = _write_stage(tmp_path, "b", [list(range(17))])
    provenance = _write_full_provenance(tmp_path, stage_a, stage_b)
    monkeypatch.setattr("pretrain.plan_pretrain_run.assert_tokenizer_contract", lambda _path: None)
    path = provenance["selection_manifest"]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    control_source = next(source for source in manifest["sources"] if source["stage"] == "stage_b")
    control_output = path.parent / "selected/control/source_b.jsonl"
    control_output.parent.mkdir(parents=True)
    control_output.write_text('{"text":"different control bytes"}\n', encoding="utf-8")
    stat = control_output.stat()
    control_source["stage"] = "control"
    control_source["output"] = {
        "relative_path": "selected/control/source_b.jsonl",
        "sha256": hashlib.sha256(control_output.read_bytes()).hexdigest(),
        "size_bytes": stat.st_size,
        "documents": 1,
    }
    _refresh_selection_evidence_for_source(
        path,
        manifest,
        source_id=control_source["source_id"],
    )
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="sources do not exactly match"):
        _build_full_plan(stage_a, stage_b, provenance, stage_b_selection_stage="control")
