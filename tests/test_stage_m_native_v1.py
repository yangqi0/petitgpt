"""Stage-M Stage-I-native tooling contract tests.

Bounded synthetic fixtures only: no real 26 GB corpus is produced anywhere in this file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pretrain.dataset_pretrain import PackedBinDataset
import pretrain.stage_m_contract_v1 as contract
from pretrain.stage_m_contract_v1 import (
    CANDIDATE_PLAN_SCHEMA,
    M_IMPLEMENTATION_BUNDLE_FILES,
    MODEL_CONTRACT,
    ORDERING_CONTRACT_ID,
    SEQ_LEN,
    STAGE_STREAMS,
    InputSequenceCommitment,
    canonical_json_bytes,
    canonical_record_commitment_payload,
    m_implementation_bundle,
    stream_accounting,
    total_accounting,
)
from pretrain.stage_m_input_v1 import (
    StageIInputError,
    derive_input_sequence_commitments,
    iter_accepted_records,
    load_accepted_stage_i,
    validate_record,
)
from pretrain.stage_m_output_v1 import (
    StageMOutputError,
    build_release_meta,
    pack_stream,
    publish_release_atomic,
    validate_published_release,
    verify_release_against_accounting,
    write_manifest,
)
import pretrain.stage_m_realize_v1 as realize
from src.special_tokens import BOS_ID, EOS_ID
from tests._stage_m_fixtures import (
    framed_ids,
    interleaved_records,
    make_record,
    read_json,
    save_tokenizer,
    tiny_tokenizer,
    write_accepted_exclusion_authorities,
    write_accepted_stage_i,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def tok():
    return tiny_tokenizer()


@pytest.fixture
def accepted(tmp_path, tok):
    records = interleaved_records(tok, per_group=3)
    return write_accepted_stage_i(tmp_path / "stage_i", records), records


@pytest.fixture
def big_accepted(tmp_path, big_records):
    """An accepted publication large enough that both streams exceed one 2048-token window."""
    return write_accepted_stage_i(
        tmp_path / "stage_i_big", big_records, records_per_shard=64
    ), big_records


@pytest.fixture
def big_records(tok):
    """Enough records that both streams exceed one 2048-token window."""
    records: list[dict[str, Any]] = []
    stages = ("stage_b", "stage_a", "stage_b", "stage_a")
    sources = ("s_one", "s_two")
    for index in range(400):
        stage = stages[index % len(stages)]
        source = sources[index % len(sources)]
        records.append(
            make_record(
                tok,
                stage=stage,
                source_id=f"{stage[-1]}_{source}",
                binding=f"ib_{source}",
                ordinal=index,
                rank=400 - index,
                text=(
                    "The quick brown fox jumps over the lazy dog while a tutorial paragraph "
                    f"explains a concept step by step with examples number {index}."
                ),
            )
        )
    return records


@pytest.fixture
def relaxed(monkeypatch, tmp_path, tok):
    """A plan-generation environment usable with the tiny fixture tokenizer.

    The tokenizer-contract call and the frozen interpreter check are stubbed because a 500-token
    fixture tokenizer is not the canonical 32k release and CI is not the frozen pod. Both are
    asserted separately: ``test_realize_calls_the_canonical_tokenizer_contract`` proves the call
    happens, and ``test_frozen_environment_constants`` pins the real constants.
    """
    calls: dict[str, int] = {"tokenizer_contract": 0, "verify_environment": 0}

    def _contract(path):
        calls["tokenizer_contract"] += 1

    def _env(environment):
        calls["verify_environment"] += 1

    monkeypatch.setattr(realize, "assert_tokenizer_contract", _contract)
    monkeypatch.setattr(realize, "verify_environment", _env)
    tokenizer_path = save_tokenizer(tok, tmp_path / "tok" / "tokenizer.json")
    # R3: the canonical exclusion authority comes from accepted G and G2, so the fixture writes
    # both manifests and the single L1 artifact they name.
    canonical = write_accepted_exclusion_authorities(tmp_path)
    monkeypatch.setattr(realize, "resolve_repo_root", lambda explicit=None: tmp_path.resolve())
    return {
        "calls": calls,
        "tokenizer_relative": str(tokenizer_path.relative_to(tmp_path)),
        "canonical_exclusion": canonical,
        "repo_root": tmp_path.resolve(),
    }


def _bundle_into(root: Path) -> None:
    """Copy the real implementation bundle under ``root`` so byte binding is testable there."""
    for relative in M_IMPLEMENTATION_BUNDLE_FILES:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPO_ROOT / relative).read_bytes())


def _make_plan(accepted_dir: Path, relaxed_env, out: Path, *, shard_tokens: int = 4096) -> Path:
    _bundle_into(relaxed_env["repo_root"])
    argv = [
        "plan",
        "--accepted-stage-i-dir",
        str(accepted_dir),
        "--tokenizer",
        relaxed_env["tokenizer_relative"],
        "--out",
        str(out),
        "--shard-tokens",
        str(shard_tokens),
        "--implementation-commit",
        "0" * 40,
    ]
    assert realize.main(argv) == 0
    return out


# --------------------------------------------------------------------- 13.1 record and order


def test_training_text_is_consumed_and_text_is_not_substituted(accepted, tok):
    accepted_dir, records = accepted
    binding = load_accepted_stage_i(accepted_dir)
    streamed = [record for _stage, record in iter_accepted_records(binding)]
    assert [r["training_text"] for r in streamed] == [r["training_text"] for r in records]
    assert all("text" not in r for r in streamed)


def test_a_record_carrying_text_instead_of_training_text_is_rejected(tmp_path, tok):
    record = make_record(
        tok, stage="stage_a", source_id="a", binding="ib", ordinal=0, rank=0, text="hello"
    )
    record["text"] = record.pop("training_text")
    with pytest.raises(StageIInputError, match="field set mismatch"):
        validate_record(record, label="r")


def test_every_record_is_consumed_exactly_once_and_stage_membership_is_exact(accepted):
    accepted_dir, records = accepted
    binding = load_accepted_stage_i(accepted_dir)
    seen: list[tuple[str, str]] = []
    for stage, record in iter_accepted_records(binding):
        seen.append((stage, str(record["cleaned_text_sha256"])))
    assert len(seen) == len(records) == binding.total_records
    assert len(set(seen)) == len(seen)
    for stage in STAGE_STREAMS:
        expected = [r for r in records if r["stage"] == stage]
        assert sum(1 for s, _ in seen if s == stage) == len(expected)
        assert binding.stage_membership[stage]["records"] == len(expected)


def test_per_stage_relative_order_is_the_accepted_physical_order(accepted):
    accepted_dir, records = accepted
    binding = load_accepted_stage_i(accepted_dir)
    per_stage: dict[str, list[str]] = {stage: [] for stage in STAGE_STREAMS}
    for stage, record in iter_accepted_records(binding):
        per_stage[stage].append(str(record["cleaned_text_sha256"]))
    for stage in STAGE_STREAMS:
        expected = [str(r["cleaned_text_sha256"]) for r in records if r["stage"] == stage]
        assert per_stage[stage] == expected


def test_fixture_physically_interleaves_stages_and_sources(accepted):
    """Guard the guard: an order-insensitive packer must be observable with this fixture."""
    _accepted_dir, records = accepted
    stages = [r["stage"] for r in records]
    sources = [r["source_id"] for r in records]
    assert len(set(stages)) == 2
    assert any(a != b for a, b in zip(stages, stages[1:], strict=False))
    assert len(set(sources)) > 2
    per_stage_ranks = [
        r["selection_ordinal_within_node"] for r in records if r["stage"] == "stage_a"
    ]
    assert per_stage_ranks != sorted(per_stage_ranks), "rank order must differ from physical"


def test_stage_streams_are_separate_and_not_concatenated(accepted, tok, tmp_path):
    accepted_dir, records = accepted
    binding = load_accepted_stage_i(accepted_dir)
    for stage in STAGE_STREAMS:
        docs = [framed_ids(tok, r["training_text"]) for r in records if r["stage"] == stage]
        total = sum(len(d) for d in docs)
        assert total == binding.stage_membership[stage]["serialized_tokens"]


# --------------------------------------------------------------------- 13.2 no legacy path


def test_native_module_never_imports_legacy_orchestration():
    import ast

    source = (REPO_ROOT / "pretrain" / "stage_m_realize_v1.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "pretrain.build_pretrain_shards":
            imported.update(alias.name for alias in node.names)
    assert imported == {"encode_with_accounting", "load_tokenizer"}, imported


@pytest.mark.parametrize(
    "symbol",
    [
        "write_shards",
        "choose_src_by_remaining",
        "_allocate_weighted_tokens",
        "is_validation_holdout",
        "route_document",
        "handle_source_exhaustion",
    ],
)
def test_no_legacy_orchestration_symbol_is_referenced_on_the_native_path(symbol):
    """AST-level: the native modules never *call* legacy orchestration.

    Deliberately not a substring scan -- the docstrings name what they exclude, and
    ``val_ratio`` is a legitimate schema-3 manifest key. Only real identifier references count.
    """
    import ast

    for name in (
        "stage_m_realize_v1",
        "stage_m_input_v1",
        "stage_m_output_v1",
        "stage_m_contract_v1",
    ):
        tree = ast.parse((REPO_ROOT / "pretrain" / f"{name}.py").read_text(encoding="utf-8"))
        referenced: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                referenced.add(node.id)
            elif isinstance(node, ast.Attribute):
                referenced.add(node.attr)
            elif isinstance(node, ast.alias):
                referenced.add(node.name.split(".")[-1])
            elif isinstance(node, ast.keyword) and node.arg:
                referenced.add(node.arg)
        assert symbol not in referenced, f"{name} references legacy {symbol}"


def test_native_path_takes_only_two_primitives_from_the_legacy_module():
    """val_ratio / min_val_tokens_per_source never reach the native path as parameters."""
    import ast

    for name in ("stage_m_realize_v1", "stage_m_input_v1", "stage_m_output_v1"):
        tree = ast.parse((REPO_ROOT / "pretrain" / f"{name}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.arg):
                assert node.arg not in {"val_ratio", "min_val_tokens_per_source"}
            if isinstance(node, ast.keyword) and node.arg:
                assert node.arg not in {"val_ratio", "min_val_tokens_per_source"}


def test_native_release_has_no_validation_reserve(big_accepted, relaxed, tmp_path, tok):
    accepted_dir, _records = big_accepted
    plan_path = _make_plan(accepted_dir, relaxed, tmp_path / "plan.json")
    plan = read_json(plan_path)
    assert plan["legacy_orchestration_used"] is False
    assert "val_ratio" not in plan
    assert "min_val_tokens_per_source" not in plan


# --------------------------------------------------------------------- 13.3 packing semantics


def test_framing_is_applied_exactly_once_with_no_separator(tok):
    ids = framed_ids(tok, "Hello world")
    assert ids[0] == BOS_ID and ids[-1] == EOS_ID
    assert ids.count(BOS_ID) == 1 and ids.count(EOS_ID) == 1


def test_adjacent_documents_meet_as_eos_then_bos(tmp_path, tok):
    docs = [framed_ids(tok, f"document number {i}") for i in range(6)]
    total = sum(len(d) for d in docs)
    accounting = stream_accounting("stage_a", total, 8)
    packed = pack_stream(
        stage="stage_a",
        documents=iter(docs),
        accounting=accounting,
        directory=tmp_path / "train",
        shard_tokens=64,
    )
    stream = np.concatenate([
        np.fromfile(tmp_path / "train" / Path(r["path"]).name, dtype=np.uint16)
        for r in packed.shard_records
    ])
    eos_positions = [i for i, v in enumerate(stream) if v == EOS_ID and i + 1 < len(stream)]
    assert eos_positions
    for position in eos_positions:
        assert stream[position + 1] == BOS_ID


def test_multiple_documents_share_a_block_and_a_document_spans_blocks(tmp_path, tok):
    """Prove both packing properties from the emitted stream, not from arithmetic.

    A block covers ``[i*T, i*T + T)`` of the virtual stream. Each document occupies a known
    half-open token range, so "several documents share a block" and "a document crosses a block
    boundary" are both decidable by interval arithmetic against the ranges actually written.
    """
    seq_len = 8
    docs = [framed_ids(tok, f"tiny {i}") for i in range(20)]
    docs.append(framed_ids(tok, "a deliberately much longer document " * 6))
    docs += [framed_ids(tok, f"tail {i}") for i in range(20)]
    total = sum(len(d) for d in docs)
    accounting = stream_accounting("stage_a", total, seq_len)
    assert accounting.training_sequences >= 3

    packed = pack_stream(
        stage="stage_a",
        documents=iter(docs),
        accounting=accounting,
        directory=tmp_path / "train",
        shard_tokens=32,
    )
    assert packed.documents == len(docs)

    stream = np.concatenate([
        np.fromfile(tmp_path / "train" / Path(r["path"]).name, dtype=np.uint16)
        for r in packed.shard_records
    ]).tolist()
    assert len(stream) == accounting.retained_stored_token_ids

    ranges, offset = [], 0
    for ids in docs:
        start, end = offset, offset + len(ids)
        offset = end
        if start < len(stream):
            ranges.append((start, min(end, len(stream))))

    shared = 0
    for index in range(accounting.training_sequences):
        lo, hi = index * seq_len, (index + 1) * seq_len
        if len([r for r in ranges if r[0] < hi and r[1] > lo]) > 1:
            shared += 1
    spanning = sum(1 for start, end in ranges if (start // seq_len) != ((end - 1) // seq_len))
    assert shared > 0, "no packed block contains more than one document"
    assert spanning > 0, "no document crosses a block boundary"

    boundaries = [
        i for i, value in enumerate(stream[:-1]) if value == EOS_ID and stream[i + 1] != EOS_ID
    ]
    assert boundaries, "the packed stream has no document boundary to check"
    for position in boundaries:
        assert stream[position + 1] == BOS_ID, "a separator token appeared between documents"
    assert {start for start, _end in ranges} == {
        i for i, value in enumerate(stream) if value == BOS_ID
    }


@pytest.mark.parametrize("t", [4, 8, 2048])
@pytest.mark.parametrize("delta", [0, 1, 2])
@pytest.mark.parametrize("multiple", [1, 2])
def test_exact_off_by_one_accounting(t, delta, multiple):
    n = multiple * t + delta
    if n < t + 1:
        with pytest.raises(contract.StageMError, match="too short"):
            stream_accounting("stage_a", n, t)
        return
    accounting = stream_accounting("stage_a", n, t)
    q = (n - 1) // t
    assert accounting.training_sequences == q
    assert accounting.model_input_positions == q * t
    assert accounting.retained_stored_token_ids == q * t + 1
    assert accounting.tail_transitions == (n - 1) - q * t
    assert accounting.final_lookahead_tokens == 1
    assert accounting.padding_tokens == 0
    assert n == (
        accounting.model_input_positions
        + accounting.tail_transitions
        + accounting.final_lookahead_tokens
    )


def test_n_equal_to_one_is_rejected():
    with pytest.raises(contract.StageMError):
        stream_accounting("stage_a", 1, 2048)


def test_two_streams_drop_two_independent_tails():
    a = stream_accounting("stage_a", 10_000_003_234, SEQ_LEN)
    b = stream_accounting("stage_b", 3_000_004_240, SEQ_LEN)
    assert a.tail_transitions == 161
    assert b.tail_transitions == 1679
    totals = total_accounting([a, b])
    assert totals["total_tail_transitions"] == 1840
    assert totals["total_final_lookahead_tokens"] == 2
    assert totals["total_padding_tokens"] == 0


def test_no_padding_is_ever_written(tmp_path, tok):
    docs = [framed_ids(tok, f"pad check {i}") for i in range(20)]
    total = sum(len(d) for d in docs)
    accounting = stream_accounting("stage_a", total, 8)
    packed = pack_stream(
        stage="stage_a",
        documents=iter(docs),
        accounting=accounting,
        directory=tmp_path / "train",
        shard_tokens=32,
    )
    stream = np.concatenate([
        np.fromfile(tmp_path / "train" / Path(r["path"]).name, dtype=np.uint16)
        for r in packed.shard_records
    ])
    assert len(stream) == accounting.retained_stored_token_ids
    # PAD is id 0 and is never emitted by the packer.
    assert 0 not in set(stream.tolist())


# --------------------------------------------------------------------- 13.4 accounting values


def test_frozen_real_accounting_constants_without_materializing_the_corpus():
    a = stream_accounting("stage_a", 10_000_003_234, SEQ_LEN)
    b = stream_accounting("stage_b", 3_000_004_240, SEQ_LEN)
    assert (
        a.training_sequences,
        a.model_input_positions,
        a.retained_stored_token_ids,
        a.tail_transitions,
        a.final_lookahead_tokens,
        a.padding_tokens,
    ) == (4_882_814, 10_000_003_072, 10_000_003_073, 161, 1, 0)
    assert (
        b.training_sequences,
        b.model_input_positions,
        b.retained_stored_token_ids,
        b.tail_transitions,
        b.final_lookahead_tokens,
        b.padding_tokens,
    ) == (1_464_845, 3_000_002_560, 3_000_002_561, 1679, 1, 0)
    totals = total_accounting([a, b])
    assert totals["total_input_serialized_tokens"] == 13_000_007_474
    assert totals["total_training_sequences"] == 6_347_659
    assert totals["total_model_input_positions"] == 13_000_005_632
    assert totals["total_retained_stored_token_ids"] == 13_000_005_634
    assert totals["total_tail_transitions"] == 1840


def test_six_quantities_are_reported_under_six_distinct_names():
    fields = set(stream_accounting("stage_a", 100_000, 2048).as_canonical())
    for name in (
        "input_serialized_tokens",
        "retained_stored_token_ids",
        "model_input_positions",
        "training_sequences",
        "tail_transitions",
        "final_lookahead_tokens",
        "padding_tokens",
    ):
        assert name in fields
    assert "training_tokens" not in fields


# --------------------------------------------------------------------- 13.5 commitments


def _commitment(records, stage):
    commitment = InputSequenceCommitment(stage)
    for ordinal, record in enumerate(r for r in records if r["stage"] == stage):
        commitment.update(
            canonical_record_commitment_payload(
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
            ),
            serialized_token_count=int(record["serialized_token_count"]),
            content_token_count=int(record["content_token_count"]),
        )
    return commitment.seal()


def test_commitment_detects_omission_duplication_reorder_and_substitution(accepted, tok):
    _accepted_dir, records = accepted
    base = _commitment(records, "stage_a")

    omitted = [r for r in records if r is not next(x for x in records if x["stage"] == "stage_a")]
    assert _commitment(omitted, "stage_a") != base

    duplicated = list(records)
    duplicated.insert(0, next(r for r in records if r["stage"] == "stage_a"))
    assert _commitment(duplicated, "stage_a") != base

    stage_a_indices = [i for i, r in enumerate(records) if r["stage"] == "stage_a"]
    reordered = list(records)
    i, j = stage_a_indices[0], stage_a_indices[1]
    reordered[i], reordered[j] = reordered[j], reordered[i]
    assert _commitment(reordered, "stage_a") != base

    restaged = [dict(r) for r in records]
    restaged[stage_a_indices[0]]["stage"] = "stage_b"
    assert _commitment(restaged, "stage_a") != base

    for field, value in (
        ("cleaned_text_sha256", "0" * 64),
        ("canonical_fingerprint", "1" * 64),
        ("raw_sha256", "2" * 64),
        ("input_record_sha256", "3" * 64),
        ("serialized_token_count", 99999),
    ):
        mutated = [dict(r) for r in records]
        mutated[stage_a_indices[0]][field] = value
        assert _commitment(mutated, "stage_a") != base, field


def test_commitment_is_not_a_python_repr():
    payload = canonical_record_commitment_payload(
        stage="stage_a",
        ordinal=0,
        source_id="s",
        input_binding_id="b",
        stable_input_record_ordinal=1,
        canonical_fingerprint="a" * 64,
        cleaned_text_sha256="b" * 64,
        raw_sha256="c" * 64,
        input_record_sha256="d" * 64,
        selection_ordinal_within_node=2,
        content_token_count=3,
        serialized_token_count=5,
    )
    parsed = json.loads(payload.decode("utf-8"))
    assert parsed["schema_version"] == contract.INPUT_SEQUENCE_COMMITMENT_SCHEMA
    assert payload.endswith(b"\n")


def test_derived_commitments_match_a_hand_rolled_stream(accepted):
    accepted_dir, records = accepted
    binding = load_accepted_stage_i(accepted_dir)
    derived = derive_input_sequence_commitments(binding)
    for stage in STAGE_STREAMS:
        assert derived[stage].seal() == _commitment(records, stage)


def test_record_byte_substitution_in_a_shard_is_rejected(accepted):
    accepted_dir, _records = accepted
    binding = load_accepted_stage_i(accepted_dir)
    shard = accepted_dir / "documents" / str(binding.shard_inventory[0]["name"])
    payload = shard.read_bytes().replace(b"quick", b"slowl", 1)
    shard.write_bytes(payload)
    with pytest.raises(StageIInputError, match="do not match the accepted manifest digest"):
        list(iter_accepted_records(binding))


# --------------------------------------------------------------------- 13.6 accepted-I binding


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expected_run_identity": "0" * 64},
        {"expected_manifest_sha256": "1" * 64},
        {"expected_completion_sha256": "2" * 64},
        {"expected_layer2_sha256": "3" * 64},
        {"expected_records": 999},
        {"expected_serialized_tokens": 12345},
        {"expected_shard_count": 99},
    ],
)
def test_accepted_binding_rejects_a_changed_identity(accepted, kwargs):
    accepted_dir, _records = accepted
    with pytest.raises(StageIInputError):
        load_accepted_stage_i(accepted_dir, **kwargs)


def test_accepted_binding_rejects_a_tampered_shard_digest(accepted):
    accepted_dir, _records = accepted
    manifest = read_json(accepted_dir / "manifest.json")
    manifest["shards"][0]["sha256"] = "0" * 64
    (accepted_dir / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(StageIInputError, match="COMPLETE does not bind"):
        load_accepted_stage_i(accepted_dir)


def test_accepted_binding_rejects_a_shard_removed_from_disk(accepted):
    accepted_dir, _records = accepted
    binding = load_accepted_stage_i(accepted_dir)
    (accepted_dir / "documents" / str(binding.shard_inventory[-1]["name"])).unlink()
    with pytest.raises(StageIInputError, match="names a missing shard"):
        load_accepted_stage_i(accepted_dir)


def test_accepted_binding_rejects_an_extra_undeclared_shard(accepted):
    accepted_dir, _records = accepted
    (accepted_dir / "documents" / "documents-99999.jsonl").write_text("{}\n")
    with pytest.raises(StageIInputError, match="does not match the declared inventory"):
        load_accepted_stage_i(accepted_dir)


def test_accepted_binding_always_verifies_shard_bytes(accepted):
    """R1-A: the production loader hashes every shard; there is no switch to turn that off."""
    accepted_dir, _records = accepted
    binding = load_accepted_stage_i(accepted_dir)
    assert binding.shard_bytes_verified is True
    shard = accepted_dir / "documents" / str(binding.shard_inventory[0]["name"])
    data = bytearray(shard.read_bytes())
    data[0:1] = b" "
    shard.write_bytes(bytes(data))
    with pytest.raises(StageIInputError, match="SHA-256 mismatch"):
        load_accepted_stage_i(accepted_dir)


# --------------------------------------------------------------------- 13.7 implementation bytes


def test_declared_bundle_is_the_real_local_dependency_closure():
    import ast

    roots = ["pretrain/stage_m_realize_v1.py"]
    seen: set[str] = set()
    stack = list(roots)
    while stack:
        rel = stack.pop()
        if rel in seen:
            continue
        seen.add(rel)
        tree = ast.parse((REPO_ROOT / rel).read_text(encoding="utf-8"), filename=rel)
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
            for dotted in names:
                parts = dotted.split(".")
                for depth in range(1, len(parts)):
                    init = REPO_ROOT.joinpath(*parts[:depth], "__init__.py")
                    if init.is_file():
                        candidate = str(init.relative_to(REPO_ROOT))
                        if candidate not in seen:
                            stack.append(candidate)
                module = REPO_ROOT.joinpath(*parts).with_suffix(".py")
                if module.is_file():
                    candidate = str(module.relative_to(REPO_ROOT))
                    if candidate not in seen:
                        stack.append(candidate)
    assert sorted(seen) == sorted(M_IMPLEMENTATION_BUNDLE_FILES)


def test_unbound_load_bearing_module_count_is_zero():
    files, _digest = m_implementation_bundle(REPO_ROOT)
    assert set(files) == set(M_IMPLEMENTATION_BUNDLE_FILES)
    assert len(files) == len(M_IMPLEMENTATION_BUNDLE_FILES)


def test_bundle_digest_binds_path_as_well_as_content():
    files, digest = m_implementation_bundle(REPO_ROOT)
    moved = dict(files)
    key = sorted(moved)[0]
    moved["pretrain/somewhere_else.py"] = moved.pop(key)
    assert contract.bundle_sha256(moved) != digest


def test_bundle_membership_is_an_explicit_list_not_a_convention():
    source = (REPO_ROOT / "pretrain" / "stage_m_contract_v1.py").read_text(encoding="utf-8")
    assert "glob(" not in source
    assert "iterdir()" not in source


@pytest.mark.parametrize("member", list(M_IMPLEMENTATION_BUNDLE_FILES))
def test_a_single_changed_implementation_byte_invalidates_a_fixed_plan(
    member, big_accepted, relaxed, tmp_path
):
    accepted_dir, _records = big_accepted
    plan_path = _make_plan(accepted_dir, relaxed, tmp_path / "plan.json")
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    realize.authorize_plan(plan_path, digest, relaxed["repo_root"])

    target = relaxed["repo_root"] / member
    target.write_bytes(target.read_bytes() + b"\n# byte change\n")
    with pytest.raises(contract.StageMError, match="implementation"):
        realize.authorize_plan(plan_path, digest, relaxed["repo_root"])


# --------------------------------------------------------------------- 13.8 environment


def test_frozen_environment_constants():
    assert contract.REQUIRED_PYTHON_EXECUTABLE == "/workspace/petitgpt/.venv/bin/python"
    assert contract.REQUIRED_PYTHON_VERSION == "3.10.12"
    assert contract.REQUIRED_TOKENIZERS_VERSION == "0.22.2"


@pytest.mark.parametrize(
    "field,value",
    [
        ("python_executable", "/usr/bin/python3"),
        ("python_version", "3.12.1"),
        ("tokenizers_version", "0.21.0"),
        ("byte_order", "big"),
    ],
)
def test_environment_mismatch_is_rejected(field, value):
    kwargs = {
        "python_executable": contract.REQUIRED_PYTHON_EXECUTABLE,
        "python_version": contract.REQUIRED_PYTHON_VERSION,
        "tokenizers_version": contract.REQUIRED_TOKENIZERS_VERSION,
        "numpy_version": "2.2.6",
        "byte_order": contract.REQUIRED_BYTE_ORDER,
    }
    kwargs[field] = value
    with pytest.raises(contract.StageMError):
        contract.verify_environment(contract.Environment(**kwargs))


def test_numpy_is_a_bound_runtime_dependency():
    assert "numpy_version" in contract.current_environment().as_canonical()


def test_plan_binds_the_running_environment_and_rejects_a_different_one(
    big_accepted, relaxed, tmp_path, monkeypatch
):
    accepted_dir, _records = big_accepted
    plan_path = _make_plan(accepted_dir, relaxed, tmp_path / "plan.json")
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    realize.authorize_plan(plan_path, digest, relaxed["repo_root"])

    other = contract.Environment(
        python_executable="/usr/bin/python3",
        python_version="3.12.1",
        tokenizers_version="0.21.0",
        numpy_version="1.0.0",
        byte_order="little",
    )
    monkeypatch.setattr(realize, "current_environment", lambda: other)
    with pytest.raises(contract.StageMError, match="environment"):
        realize.authorize_plan(plan_path, digest, relaxed["repo_root"])


def test_tokenizer_substitution_is_rejected(big_accepted, relaxed, tmp_path):
    accepted_dir, _records = big_accepted
    plan_path = _make_plan(accepted_dir, relaxed, tmp_path / "plan.json")
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    tokenizer = relaxed["repo_root"] / relaxed["tokenizer_relative"]
    tokenizer.write_bytes(tokenizer.read_bytes() + b" ")
    with pytest.raises(contract.StageMError, match="tokenizer"):
        realize.authorize_plan(plan_path, digest, relaxed["repo_root"])


def test_realize_calls_the_canonical_tokenizer_contract(big_accepted, relaxed, tmp_path):
    accepted_dir, _records = big_accepted
    plan_path = _make_plan(accepted_dir, relaxed, tmp_path / "plan.json")
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    realize.authorize_plan(plan_path, digest, relaxed["repo_root"])
    assert relaxed["calls"]["tokenizer_contract"] >= 1


def test_model_and_seq_len_contract_are_bound(big_accepted, relaxed, tmp_path):
    accepted_dir, _records = big_accepted
    plan_path = _make_plan(accepted_dir, relaxed, tmp_path / "plan.json")
    plan = read_json(plan_path)
    assert plan["model_contract"] == dict(MODEL_CONTRACT)
    assert plan["model_contract"]["seq_len"] == SEQ_LEN == 2048
    assert plan["model_contract"]["vocab_size"] == 32_000

    plan["model_contract"]["n_layers"] = 31
    plan_path.write_bytes(canonical_json_bytes(plan))
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    with pytest.raises(contract.StageMError, match="model_contract"):
        realize.authorize_plan(plan_path, digest, relaxed["repo_root"])


# --------------------------------------------------------------------- 13.9 candidate plan


def test_candidate_plan_is_byte_identical_three_times(big_accepted, relaxed, tmp_path):
    accepted_dir, _records = big_accepted
    digests = []
    for index in range(3):
        out = tmp_path / f"plan_{index}.json"
        _make_plan(accepted_dir, relaxed, out)
        digests.append(hashlib.sha256(out.read_bytes()).hexdigest())
    assert len(set(digests)) == 1


def test_candidate_plan_generation_is_cwd_independent(big_accepted, relaxed, tmp_path, monkeypatch):
    accepted_dir, _records = big_accepted
    first = tmp_path / "a.json"
    _make_plan(accepted_dir, relaxed, first)
    elsewhere = tmp_path / "unrelated"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    second = tmp_path / "b.json"
    _make_plan(accepted_dir, relaxed, second)
    assert first.read_bytes() == second.read_bytes()


def test_candidate_plan_is_not_authorized(big_accepted, relaxed, tmp_path):
    accepted_dir, _records = big_accepted
    plan_path = _make_plan(accepted_dir, relaxed, tmp_path / "plan.json")
    plan = read_json(plan_path)
    assert plan["authorization_status"] == "NOT_AUTHORIZED"
    assert plan["schema_version"] == CANDIDATE_PLAN_SCHEMA
    assert plan["ordering_contract"]["policy"] == ORDERING_CONTRACT_ID


def test_a_self_declared_authorization_is_refused(big_accepted, relaxed, tmp_path):
    accepted_dir, _records = big_accepted
    plan_path = _make_plan(accepted_dir, relaxed, tmp_path / "plan.json")
    plan = read_json(plan_path)
    plan["authorization_status"] = "AUTHORIZED"
    plan_path.write_bytes(canonical_json_bytes(plan))
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    with pytest.raises(contract.StageMError, match="self-declared authorization"):
        realize.authorize_plan(plan_path, digest, relaxed["repo_root"])


def test_a_wrong_expected_digest_is_refused(big_accepted, relaxed, tmp_path):
    accepted_dir, _records = big_accepted
    plan_path = _make_plan(accepted_dir, relaxed, tmp_path / "plan.json")
    with pytest.raises(contract.StageMError, match="digest mismatch"):
        realize.authorize_plan(plan_path, "0" * 64, relaxed["repo_root"])


def test_plan_will_not_overwrite_an_existing_file(big_accepted, relaxed, tmp_path):
    accepted_dir, _records = big_accepted
    out = tmp_path / "plan.json"
    _make_plan(accepted_dir, relaxed, out)
    with pytest.raises(contract.StageMError, match="refusing to overwrite"):
        _make_plan(accepted_dir, relaxed, out)


def test_repo_root_resolution_rejects_an_unrelated_root(tmp_path):
    with pytest.raises(contract.StageMError, match="must name the executing"):
        contract.resolve_repo_root(tmp_path)
    assert contract.resolve_repo_root(None) == REPO_ROOT
    assert contract.resolve_repo_root(REPO_ROOT) == REPO_ROOT


# --------------------------------------------------------------------- 13.10 publication


def _release(tmp_path, tok, records, stage, *, shard_tokens=512, seq_len=8):
    docs = [framed_ids(tok, r["training_text"]) for r in records if r["stage"] == stage]
    total = sum(len(d) for d in docs)
    accounting = stream_accounting(stage, total, seq_len)
    staging = tmp_path / "staging"
    packed = pack_stream(
        stage=stage,
        documents=iter(docs),
        accounting=accounting,
        directory=staging / "train",
        shard_tokens=shard_tokens,
    )
    meta = build_release_meta(
        packed,
        tokenizer_path="tokenizer.json",
        tokenizer_sha256="a" * 64,
        stage_m_binding={"stage": stage},
        reference_exclusion={
            "enabled": True,
            "manifest_count": 1,
            "union_hash_count": 3,
            "manifests": [{"enabled": True, "manifest_sha256": "b" * 64, "hash_count": 3}],
        },
    )
    write_manifest(staging, meta)
    return staging, accounting, meta


def test_published_release_validates_and_matches_accounting(tmp_path, tok, big_records):
    staging, accounting, _meta = _release(tmp_path, tok, big_records, "stage_a")
    result = verify_release_against_accounting(staging, accounting)
    assert result["dtype"] == "uint16"
    published = publish_release_atomic(staging, tmp_path / "release")
    verify_release_against_accounting(published, accounting)


def test_publication_never_replaces_an_existing_release(tmp_path, tok, big_records):
    staging, _accounting, _meta = _release(tmp_path, tok, big_records, "stage_a")
    destination = tmp_path / "release"
    destination.mkdir()
    with pytest.raises(StageMOutputError, match="refusing to replace"):
        publish_release_atomic(staging, destination)


def test_incorrect_token_total_is_rejected(tmp_path, tok, big_records):
    staging, accounting, meta = _release(tmp_path, tok, big_records, "stage_a")
    meta["train_tokens"] += 1
    meta["accounting"]["train"]["emitted_shard_tokens"] += 1
    write_manifest(staging, meta)
    with pytest.raises(RuntimeError):
        verify_release_against_accounting(staging, accounting)


def test_incorrect_sequence_count_is_rejected(tmp_path, tok, big_records):
    staging, accounting, _meta = _release(tmp_path, tok, big_records, "stage_a")
    # A different declared input size moves both the retained-token boundary and the sequence
    # count, so the published stream can no longer satisfy the claimed accounting.
    wrong = stream_accounting("stage_a", accounting.input_serialized_tokens + 4096, 8)
    assert wrong.training_sequences != accounting.training_sequences
    with pytest.raises(StageMOutputError):
        verify_release_against_accounting(staging, wrong)


def test_per_shard_hash_tampering_is_rejected(tmp_path, tok, big_records):
    staging, accounting, _meta = _release(tmp_path, tok, big_records, "stage_a")
    shard = sorted((staging / "train").glob("shard_*.bin"))[0]
    data = bytearray(shard.read_bytes())
    data[0] ^= 0xFF
    shard.write_bytes(bytes(data))
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify_release_against_accounting(staging, accounting)


def test_partial_output_is_not_represented_as_complete(tmp_path, tok, big_records):
    staging, _accounting, _meta = _release(tmp_path, tok, big_records, "stage_a")
    (staging / "meta.json").unlink()
    with pytest.raises((RuntimeError, FileNotFoundError)):
        validate_published_release(staging)


def test_shard_geometry_is_fixed_except_for_the_last(tmp_path, tok, big_records):
    staging, _accounting, meta = _release(tmp_path, tok, big_records, "stage_a", shard_tokens=256)
    counts = [record["token_count"] for record in meta["shard_files"]["train"]]
    assert all(count == 256 for count in counts[:-1])
    assert 0 < counts[-1] <= 256


# --------------------------------------------------------------------- 13.13 physical consumer


@pytest.mark.parametrize("stage", list(STAGE_STREAMS))
def test_packed_bin_dataset_consumes_the_release_with_t_plus_1_stride_t(
    tmp_path, tok, big_records, stage
):
    seq_len = 8
    staging, accounting, _meta = _release(
        tmp_path, tok, big_records, stage, shard_tokens=256, seq_len=seq_len
    )
    published = publish_release_atomic(staging, tmp_path / f"release_{stage}")
    dataset = PackedBinDataset(
        str(published / "train"), seq_len=seq_len, require_release_manifest=True
    )
    stats = dataset.stats()
    assert stats["window_size"] == seq_len + 1
    assert stats["block_stride"] == seq_len
    assert stats["n_blocks"] == accounting.training_sequences
    assert stats["tail_transitions"] == 0
    assert len(dataset) == accounting.training_sequences
    input_ids, labels, loss_mask = dataset[0]
    assert input_ids.shape[0] == seq_len
    assert labels.shape[0] == seq_len
    assert loss_mask.shape[0] == seq_len
    assert dataset.total_raw_tokens == accounting.retained_stored_token_ids
