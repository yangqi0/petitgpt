"""R4 regressions for the artifact-only pilot executor and terminal protocol.

Every test is CPU-only and uses synthetic JSON/ledger state.  Nothing here constructs a
PetitGPT model, opens the accepted packed datasets, or performs an optimizer update.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path

import pytest

import pretrain.pilot_contract_v2_3 as C
import pretrain.pilot_runner_v2_3 as R

_OBSOLETE_EXECUTION_CAPABILITIES = (
    "WorkerAuthority",
    "_WORKER_AUTHORITY_MINT",
    "_require_worker_authority",
    "_run_updates",
    "_train_phase_mb",
    "_train_phase_lr",
    "REAL_TRAINING_ENTRYPOINTS",
    "execute_validated_candidate",
    "build_pilot_model",
    "build_pilot_optimizer",
)


@pytest.mark.parametrize("name", _OBSOLETE_EXECUTION_CAPABILITIES)
def test_caller_constructed_execution_capabilities_are_absent(name):
    assert not hasattr(R, name), name


def test_the_sole_executor_accepts_only_canonical_artifact_paths():
    parameters = inspect.signature(R.execute_candidate_from_artifact_paths).parameters

    assert set(parameters) == set(R.REAL_WORKER_ARTIFACT_INPUTS) | {"gpu_required"}
    assert parameters["gpu_required"].default is True
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in parameters.values()
    )
    for forbidden in (
        "authority",
        "authorized",
        "backend",
        "candidate",
        "context",
        "execution",
        "progress",
        "session",
        "validated",
    ):
        assert forbidden not in parameters


def _enclosing_functions(node, parents):
    names = []
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(current.name)
        current = parents.get(current)
    return names


def test_validation_lexically_precedes_every_real_candidate_training_operation():
    source = Path(R.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=R.__file__)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    executor = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "execute_candidate_from_artifact_paths"
    )
    validation_calls = [
        node
        for node in ast.walk(executor)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "validate_worker_execution"
    ]
    assert len(validation_calls) == 1
    validation_line = validation_calls[0].lineno

    local_definitions = {
        node.name: node
        for node in ast.walk(executor)
        if isinstance(node, ast.FunctionDef) and node is not executor
    }
    assert {"construct_model", "construct_optimizer", "run_updates", "train_candidate"} <= set(
        local_definitions
    )
    assert validation_line < min(
        local_definitions[name].lineno
        for name in ("construct_model", "construct_optimizer", "run_updates", "train_candidate")
    )

    real_steps = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "step"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "optimizer"
    ]
    assert len(real_steps) == 1
    assert _enclosing_functions(real_steps[0], parents)[:2] == [
        "run_updates",
        "execute_candidate_from_artifact_paths",
    ]
    assert validation_line < real_steps[0].lineno

    model_calls = [
        node
        for node in ast.walk(executor)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"GPT", "build_optimizer", "PackedBinDataset"}
    ]
    assert model_calls
    assert all(validation_line < node.lineno for node in model_calls)


def test_full_release_byte_validation_is_upstream_of_executor_construction():
    source = Path(R.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=R.__file__)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    release_validator = functions["verify_accepted_release"]
    artifact_gate = functions["validate_execution_artifacts"]
    worker_gate = functions["validate_worker_execution"]
    executor = functions["execute_candidate_from_artifact_paths"]

    canonical_release_calls = [
        node
        for node in ast.walk(release_validator)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "validate_shard_release"
    ]
    assert len(canonical_release_calls) == 1

    accepted_release_calls = sorted(
        (
            node
            for node in ast.walk(artifact_gate)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "verify_accepted_release"
        ),
        key=lambda node: node.lineno,
    )
    assert [
        call.args[0].value
        for call in accepted_release_calls
        if call.args and isinstance(call.args[0], ast.Constant)
    ] == ["stage_a", "stage_b"]

    artifact_gate_calls = [
        node
        for node in ast.walk(worker_gate)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "validate_execution_artifacts"
    ]
    executor_gate_calls = [
        node
        for node in ast.walk(executor)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "validate_worker_execution"
    ]
    assert len(artifact_gate_calls) == len(executor_gate_calls) == 1

    construction_calls = [
        node
        for node in ast.walk(executor)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"GPT", "build_optimizer", "PackedBinDataset"}
    ]
    assert construction_calls
    assert executor_gate_calls[0].lineno < min(node.lineno for node in construction_calls)
    assert release_validator.lineno < artifact_gate.lineno < worker_gate.lineno < executor.lineno


def test_validation_failure_cannot_create_or_modify_candidate_artifacts(tmp_path, monkeypatch):
    root = tmp_path / "authorized"
    specs = root / R.SPEC_DIRNAME
    specs.mkdir(parents=True)
    spec = specs / "candidate.json"
    plan = root / R.PHASE_MB_PLAN_FILENAME
    spec.write_bytes(b"immutable candidate spec")
    plan.write_bytes(b"immutable phase plan")
    candidate_output = root / "candidate"
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    calls = []

    def refuse(**kwargs):
        calls.append(kwargs)
        raise R.BindingFailure("synthetic prevalidation refusal")

    monkeypatch.setattr(R, "validate_worker_execution", refuse)
    kwargs = {
        "authorization_path": tmp_path / "authorization.json",
        "session_manifest_path": root / R.SESSION_FILENAME,
        "phase_plan_path": plan,
        "candidate_spec_path": spec,
        "pilot_index_manifest_path": tmp_path / R.PILOT_INDEX_MANIFEST_FILENAME,
        "accepted_stage_a_path": Path(R.ROOT) / R.ACCEPTED_STAGE_A,
        "accepted_stage_b_path": Path(R.ROOT) / R.ACCEPTED_STAGE_B,
        "ledger_path": root / R.LEDGER_FILENAME,
        "candidate_output_path": candidate_output,
        "gpu_required": False,
    }

    with pytest.raises(R.BindingFailure, match="synthetic prevalidation refusal"):
        R.execute_candidate_from_artifact_paths(**kwargs)

    assert len(calls) == 1
    assert calls[0] == kwargs
    after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert not candidate_output.exists()
    assert not spec.with_suffix(".terminal.json").exists()
    assert not R.terminal_result_path(candidate_output).exists()


def _ledger_identity(root):
    return {
        "contract_sha256": "a" * 64,
        "implementation_head": "b" * 40,
        "execution_bundle_sha256": "c" * 64,
        "pilot_index_manifest_file_sha256": "d" * 64,
        "authorization_sha256": "e" * 64,
        "session_id": "f" * 64,
        "authorized_output_root": str(root.resolve()),
        "authorized_scope": "FULL_V2_3_PILOT",
    }


def _ceilings():
    return {
        "MB": R.PHASE_CEILINGS["MB"],
        "LR": R.PHASE_CEILINGS["LR"],
        "GLOBAL": R.GLOBAL_PILOT_TOKEN_CEILING,
    }


@pytest.fixture
def sealed_terminal(tmp_path):
    root = tmp_path / "authorized"
    root.mkdir()
    ledger = R.TokenLedger(root / R.LEDGER_FILENAME, _ledger_identity(root), _ceilings())
    expected = {
        "phase": "MB",
        "candidate_id": "mb_micro16_compileoff",
        "peak_lr": 3e-4,
        "planned_updates": 2,
        "candidate_spec_sha256": "1" * 64,
        "phase_plan_sha256": "2" * 64,
        "session_id": ledger.identity["session_id"],
        "session_sha256": "3" * 64,
        "authorization_sha256": ledger.identity["authorization_sha256"],
        "ledger_identity": dict(ledger.identity),
    }
    ledger.begin_candidate(expected)
    ledger.reserve("MB")
    ledger.complete("MB")
    receipt = ledger.finalize_candidate(
        terminal_status="CANDIDATE_INELIGIBLE",
        run_meta_sha256="4" * 64,
        result_sha256="5" * 64,
    )
    terminal = {
        "schema_version": R.TERMINAL_RESULT_SCHEMA,
        "terminal_status": "CANDIDATE_INELIGIBLE",
        "error_class": "RuntimeError",
        "error_message": "synthetic candidate-local failure",
        **expected,
        "ledger_receipt_sha256": receipt["receipt_sha256"],
        "reserved_updates": receipt["candidate_reserved_updates"],
        "completed_updates": receipt["candidate_completed_updates"],
        "reserved_tokens": receipt["after_reserved_tokens"],
        "completed_tokens": receipt["after_completed_tokens"],
        "ledger_reserved_updates": receipt["after_reserved_updates"],
        "ledger_completed_updates": receipt["after_completed_updates"],
        "run_meta_sha256": receipt["run_meta_sha256"],
        "result_sha256": receipt["result_sha256"],
    }
    assert set(terminal) == set(R.TERMINAL_RESULT_FIELDS)
    assert R.validate_terminal_document(terminal, expected=expected, ledger=ledger) == terminal
    return terminal, expected, ledger


def _string_token_map(document):
    document["reserved_tokens"] = "not a token map"


def _negative_completed_token(document):
    document["completed_tokens"]["MB"] = -R.TRAINED_TOKENS_PER_UPDATE


def _wrong_phase(document):
    document["phase"] = "LR"


def _wrong_candidate(document):
    document["candidate_id"] = "some_other_candidate"


def _malformed_sha(document):
    document["candidate_spec_sha256"] = "not-a-sha256"


def _wrong_ledger_identity(document):
    document["ledger_identity"]["session_id"] = "wrong-session"


def _unknown_receipt(document):
    document["ledger_receipt_sha256"] = "9" * 64


def _receipt_accounting_mismatch(document):
    zero = {bucket: 0 for bucket in R.LEDGER_BUCKETS}
    document["reserved_updates"] = 0
    document["completed_updates"] = 0
    document["reserved_tokens"] = dict(zero)
    document["completed_tokens"] = dict(zero)
    document["ledger_reserved_updates"] = 0
    document["ledger_completed_updates"] = 0


@pytest.mark.parametrize(
    "mutate",
    [
        _string_token_map,
        _negative_completed_token,
        _wrong_phase,
        _wrong_candidate,
        _malformed_sha,
        _wrong_ledger_identity,
        _unknown_receipt,
        _receipt_accounting_mismatch,
    ],
    ids=lambda mutate: mutate.__name__.removeprefix("_"),
)
def test_strict_terminal_validation_rejects_malformed_or_unbound_evidence(sealed_terminal, mutate):
    terminal, expected, ledger = sealed_terminal
    forged = copy.deepcopy(terminal)
    mutate(forged)

    with pytest.raises(R.PhaseAbort):
        R.validate_terminal_document(forged, expected=expected, ledger=ledger)


def _synthetic_session(root):
    identity = _ledger_identity(root)
    ledger = R.TokenLedger(root / R.LEDGER_FILENAME, identity, _ceilings())
    return R.ExecutionSession(
        {
            "authorized_root": root,
            "scope": identity["authorized_scope"],
            "session_id": identity["session_id"],
        },
        ledger,
    )


def _published_mb_plan(root, session):
    candidates = R.plan_phase_mb(output_root=root)
    published = R.publish_phase_plan(
        root=root,
        plan_kind="PHASE_MB_PLAN",
        session_sha256="6" * 64,
        session_id=session.session_id,
        candidates=candidates,
        derived_from={"source": "the frozen P-PILOT-CONTRACT-V2.3 Phase-MB grid"},
    )
    return candidates, published


def test_complete_mb_membership_rejects_a_truncated_plan(tmp_path):
    root = tmp_path / "authorized"
    root.mkdir()
    session = _synthetic_session(root)
    _, published = _published_mb_plan(root, session)
    truncated = copy.deepcopy(published["plan"])
    truncated["candidates"] = truncated["candidates"][:1]
    truncated["candidate_ids"] = truncated["candidate_ids"][:1]
    truncated["candidate_spec_sha256s"] = truncated["candidate_spec_sha256s"][:1]

    with pytest.raises(R.BindingFailure, match="complete contract-derived set"):
        R.validate_complete_phase_plan_membership(
            plan=truncated, session=session, authorized_root=root
        )


def test_complete_mb_membership_rejects_an_arbitrary_rehashed_spec(tmp_path):
    root = tmp_path / "authorized"
    root.mkdir()
    session = _synthetic_session(root)
    _, published = _published_mb_plan(root, session)
    forged = copy.deepcopy(published["plan"])
    entry = forged["candidates"][0]
    spec_path = root / entry["candidate_spec_relpath"]
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["candidate"]["updates"] = 1
    forged_bytes = C.canonical_json_bytes(spec)
    spec_path.write_bytes(forged_bytes)
    forged_sha256 = hashlib.sha256(forged_bytes).hexdigest()
    entry["candidate_spec_sha256"] = forged_sha256
    forged["candidate_spec_sha256s"][0] = forged_sha256

    with pytest.raises(R.BindingFailure, match="spec bytes differ from the contract derivation"):
        R.validate_complete_phase_plan_membership(
            plan=forged, session=session, authorized_root=root
        )


def _artifact_bound_session(root):
    root = root.resolve()
    root.mkdir()
    identity = _ledger_identity(root)
    validated = {
        "authorized_root": root,
        "scope": identity["authorized_scope"],
        "session_id": identity["session_id"],
        "authorization_sha256": identity["authorization_sha256"],
        "observed": {
            "contract_sha256": identity["contract_sha256"],
            "head": identity["implementation_head"],
            "execution_bundle_sha256": identity["execution_bundle_sha256"],
        },
        "serialized_index_lists_digest": "6" * 64,
        "pilot_index_manifest_file_sha256": identity["pilot_index_manifest_file_sha256"],
        "fingerprint": {"fingerprint_sha256": "7" * 64},
    }
    ledger = R.TokenLedger(root / R.LEDGER_FILENAME, identity, _ceilings())
    return R.ExecutionSession(validated, ledger, session_sha256="8" * 64)


def _published_lr_plan(session, peak_lrs):
    candidates = R.plan_phase_lr(
        output_root=session.output_root,
        micro_bsz=8,
        compile_on=False,
        peak_lrs=peak_lrs,
        seed_label="seed-1",
    )
    plan = R.publish_phase_plan(
        root=session.output_root,
        plan_kind="PHASE_LR_INITIAL_PLAN",
        session_sha256=session.session_sha256,
        session_id=session.session_id,
        candidates=candidates,
        derived_from={"source": "synthetic R4 admission fixture"},
    )
    return candidates, plan


def _candidate_spec_sha256(plan, candidate):
    return plan["specs"][candidate["candidate_id"]]["candidate_spec_sha256"]


def _synthetic_run_meta(candidate, session, plan):
    bindings = dict(R._session_bindings(session))  # noqa: SLF001
    return {
        "schema_version": R.RUN_META_SCHEMA,
        "contract_version": C.CONTRACT_VERSION,
        "phase": candidate["phase"],
        "candidate_id": candidate["candidate_id"],
        "seed_label": candidate["seed_label"],
        "peak_lr": candidate["peak_lr"],
        "micro_bsz": candidate["micro_bsz"],
        "grad_accum": candidate["grad_accum"],
        "compile": candidate["compile"],
        "model_seed": candidate["model_init_seed"],
        "train_order_seed": candidate["train_order_seed"],
        "output_dir": str(Path(candidate["output_dir"]).resolve()),
        "candidate_spec_sha256": _candidate_spec_sha256(plan, candidate),
        "phase_plan_sha256": plan["plan_sha256"],
        "session_sha256": session.session_sha256,
        "session_id": session.session_id,
        "contract_sha256": bindings["contract_sha256"],
        "implementation_head": bindings["implementation_head"],
        "execution_bundle_sha256": bindings["execution_bundle_sha256"],
        "pilot_index_manifest_file_sha256": bindings["pilot_index_manifest_file_sha256"],
        "runtime_fingerprint_sha256": bindings["runtime_fingerprint_sha256"],
        "authorization_sha256": bindings["authorization_sha256"],
        "lr_configuration": {
            "peak_lr": candidate["peak_lr"],
            "warmup_updates": candidate["warmup_updates"],
            "updates": candidate["updates"],
        },
        "ledger_identity": dict(session.ledger.identity),
    }


def _synthetic_lr_result(candidate, session, plan, *, eligible):
    loss = 3.0
    weight = 1024.0
    losses_by_update = {str(update): loss for update in range(1, C.LR_RUN_UPDATES + 1)}
    divergence_detail = C.sustained_divergence({int(k): v for k, v in losses_by_update.items()})
    result = {
        "schema_version": R.CANDIDATE_RESULT_SCHEMA,
        "contract_version": C.CONTRACT_VERSION,
        "phase": candidate["phase"],
        "candidate_id": candidate["candidate_id"],
        "seed_label": candidate["seed_label"],
        "peak_lr": candidate["peak_lr"],
        "micro_bsz": candidate["micro_bsz"],
        "grad_accum": candidate["grad_accum"],
        "compile": candidate["compile"],
        "output_dir": str(Path(candidate["output_dir"]).resolve()),
        "candidate_spec_sha256": _candidate_spec_sha256(plan, candidate),
        "phase_plan_sha256": plan["plan_sha256"],
        "session_sha256": session.session_sha256,
        "ledger_identity": dict(session.ledger.identity),
        "completed_updates": candidate["updates"] if eligible else 0,
        "losses_by_update": losses_by_update,
        "all_losses_finite": True,
        "all_grad_norms_finite": True,
        "all_parameters_finite": True,
        "muon_momentum_states_present": True,
        "aux_adamw_states_present": True,
        "grouping_matches_contract": True,
        "all_lr_ratios_are_one": True,
        "eval_stage_a_numerator": loss * weight,
        "eval_stage_a_weight": weight,
        "eval_stage_b_numerator": loss * weight,
        "eval_stage_b_weight": weight,
        "eval_loss_stage_a": loss,
        "eval_loss_stage_b": loss,
        "score": loss,
        "sustained_divergence": False,
        "divergence_detail": divergence_detail,
        "compile_evidence": {
            "compile_requested": False,
            "forward_invocations_match_geometry": True,
            "invoked_uncompiled_module": True,
            "realized_module_is_optimized_module": False,
            "dynamo_unique_graphs": 0,
        },
        "canonical_compile_path": True,
    }
    result.update(dict(R._session_bindings(session)))  # noqa: SLF001
    recomputed_eligible, failures = C.lr_candidate_eligible(result)
    assert recomputed_eligible is eligible
    result["eligible"] = eligible
    result["eligibility_failures"] = list(failures)
    result["terminal_status"] = "SUCCESS" if eligible else "CANDIDATE_INELIGIBLE"
    return result


def _advance_synthetic_candidate(ledger, candidate, session, plan, completed_updates):
    ledger.begin_candidate({
        "phase": candidate["phase"],
        "candidate_id": candidate["candidate_id"],
        "peak_lr": candidate["peak_lr"],
        "planned_updates": candidate["updates"],
        "candidate_spec_sha256": _candidate_spec_sha256(plan, candidate),
        "phase_plan_sha256": plan["plan_sha256"],
        "session_id": session.session_id,
        "session_sha256": session.session_sha256,
        "authorization_sha256": session.validated["authorization_sha256"],
    })
    with ledger._lock():  # noqa: SLF001
        ledger._reload_locked()  # noqa: SLF001
        tokens = completed_updates * R.TRAINED_TOKENS_PER_UPDATE
        for bucket in (candidate["phase"], "GLOBAL"):
            ledger.state["reserved_tokens"][bucket] += tokens
            ledger.state["completed_tokens"][bucket] += tokens
        ledger.state["reserved_updates"] += completed_updates
        ledger.state["completed_updates"] += completed_updates
        active = ledger.state["active_candidate"]
        active["candidate_reserved_updates"] = completed_updates
        active["candidate_completed_updates"] = completed_updates
        ledger._require_structural_invariants(ledger.state)  # noqa: SLF001
        ledger._write(ledger.state)  # noqa: SLF001


def _publish_synthetic_lr_chain(candidate, session, plan, *, eligible):
    result = _synthetic_lr_result(candidate, session, plan, eligible=eligible)
    _advance_synthetic_candidate(
        session.ledger,
        candidate,
        session,
        plan,
        completed_updates=result["completed_updates"],
    )
    output = Path(candidate["output_dir"])
    output.mkdir(parents=True)
    meta_sha256 = R.write_immutable_artifact(
        output / "run_meta.json", _synthetic_run_meta(candidate, session, plan)
    )
    result["run_meta_sha256"] = meta_sha256
    result["ledger_snapshot"] = session.ledger.snapshot()
    result_sha256 = R.write_immutable_artifact(output / "result.json", result)
    receipt = session.ledger.finalize_candidate(
        terminal_status=result["terminal_status"],
        run_meta_sha256=meta_sha256,
        result_sha256=result_sha256,
    )
    R.write_terminal_result(
        output,
        {
            "terminal_status": result["terminal_status"],
            "error_class": None,
            "error_message": None,
            "phase": candidate["phase"],
            "candidate_id": candidate["candidate_id"],
            "peak_lr": candidate["peak_lr"],
            "planned_updates": candidate["updates"],
            "candidate_spec_sha256": result["candidate_spec_sha256"],
            "phase_plan_sha256": plan["plan_sha256"],
            "session_id": session.session_id,
            "session_sha256": session.session_sha256,
            "authorization_sha256": session.validated["authorization_sha256"],
            "ledger_identity": dict(session.ledger.identity),
            "ledger_receipt_sha256": receipt["receipt_sha256"],
            "reserved_updates": receipt["candidate_reserved_updates"],
            "completed_updates": receipt["candidate_completed_updates"],
            "reserved_tokens": receipt["after_reserved_tokens"],
            "completed_tokens": receipt["after_completed_tokens"],
            "ledger_reserved_updates": receipt["after_reserved_updates"],
            "ledger_completed_updates": receipt["after_completed_updates"],
            "run_meta_sha256": meta_sha256,
            "result_sha256": result_sha256,
        },
    )


def _adversarially_replace_immutable(path, document):
    body = C.canonical_json_bytes(document)
    path.write_bytes(body)
    digest = hashlib.sha256(body).hexdigest()
    path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def _rebind_candidate_chains(session, candidates, mutate):
    artifacts = {}
    for candidate in candidates:
        output = Path(candidate["output_dir"])
        result_path = output / "result.json"
        meta_path = output / "run_meta.json"
        terminal_path = R.terminal_result_path(output)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        mutate(candidate, result, meta, terminal)
        meta_sha256 = _adversarially_replace_immutable(meta_path, meta)
        result["run_meta_sha256"] = meta_sha256
        artifacts[candidate["candidate_id"]] = {
            "result": result,
            "result_path": result_path,
            "meta": meta,
            "meta_sha256": meta_sha256,
            "terminal": terminal,
            "terminal_path": terminal_path,
        }

    ledger = session.ledger
    with ledger._lock():  # noqa: SLF001
        ledger._reload_locked()  # noqa: SLF001
        previous = None
        rebound_receipts = []
        for original in ledger.state["candidate_receipts"]:
            receipt = dict(original)
            artifact = artifacts[receipt["candidate_id"]]
            terminal = artifact["terminal"]
            receipt["previous_receipt_sha256"] = previous
            receipt["peak_lr"] = terminal["peak_lr"]
            receipt["terminal_status"] = terminal["terminal_status"]
            receipt["run_meta_sha256"] = artifact["meta_sha256"]
            artifact["result"]["ledger_snapshot"] = R._expected_result_ledger_snapshot(  # noqa: SLF001
                session, receipt
            )
            artifact["result_sha256"] = _adversarially_replace_immutable(
                artifact["result_path"], artifact["result"]
            )
            receipt["result_sha256"] = artifact["result_sha256"]
            body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
            receipt["receipt_sha256"] = hashlib.sha256(C.canonical_json_bytes(body)).hexdigest()
            terminal["ledger_receipt_sha256"] = receipt["receipt_sha256"]
            previous = receipt["receipt_sha256"]
            rebound_receipts.append(receipt)
        ledger.state["candidate_receipts"] = rebound_receipts
        ledger.state["receipt_chain_head_sha256"] = previous
        ledger._require_structural_invariants(ledger.state)  # noqa: SLF001
        ledger._write(ledger.state)  # noqa: SLF001

    for artifact in artifacts.values():
        terminal = artifact["terminal"]
        terminal["run_meta_sha256"] = artifact["meta_sha256"]
        terminal["result_sha256"] = artifact["result_sha256"]
        artifact["terminal_sha256"] = _adversarially_replace_immutable(
            artifact["terminal_path"], terminal
        )
    return artifacts


def _assert_chain_is_rehashed(session, candidate, plan, *, expected_peak_lr=None):
    output = Path(candidate["output_dir"])
    result, result_sha256 = R.read_immutable_artifact(
        output / "result.json", schema_version=R.CANDIDATE_RESULT_SCHEMA
    )
    meta, meta_sha256 = R.read_immutable_artifact(
        output / "run_meta.json", schema_version=R.RUN_META_SCHEMA
    )
    expected = R.terminal_expectations(session, planned=candidate, plan=plan)
    if expected_peak_lr is not None:
        expected["peak_lr"] = expected_peak_lr
    terminal = R.read_terminal_result(output, expected=expected, ledger=session.ledger)
    assert result["run_meta_sha256"] == terminal["run_meta_sha256"] == meta_sha256
    assert terminal["result_sha256"] == result_sha256
    assert result["peak_lr"] == meta["peak_lr"] == terminal["peak_lr"]
    assert meta["lr_configuration"]["peak_lr"] == result["peak_lr"]
    receipt = session.ledger.receipt(terminal["ledger_receipt_sha256"])
    assert receipt["result_sha256"] == result_sha256
    assert result["ledger_snapshot"] == R._expected_result_ledger_snapshot(  # noqa: SLF001
        session, receipt
    )
    return result


def test_swapped_lr_peak_labels_are_rejected_after_every_chain_hash_is_rebound(tmp_path):
    session = _artifact_bound_session(tmp_path / "authorized")
    candidates, plan = _published_lr_plan(session, C.LR_GRID_SEED1)
    for candidate in candidates:
        _publish_synthetic_lr_chain(candidate, session, plan, eligible=True)
        assert R.load_completed_result(session, planned=candidate, plan=plan)["eligible"] is True

    first, second = candidates[:2]
    swapped = {
        first["candidate_id"]: second["peak_lr"],
        second["candidate_id"]: first["peak_lr"],
    }

    def swap_peak(candidate, result, meta, terminal):
        if candidate["candidate_id"] not in swapped:
            return
        peak_lr = swapped[candidate["candidate_id"]]
        result["peak_lr"] = peak_lr
        meta["peak_lr"] = peak_lr
        meta["lr_configuration"]["peak_lr"] = peak_lr
        terminal["peak_lr"] = peak_lr

    _rebind_candidate_chains(session, candidates, swap_peak)
    for candidate in candidates:
        expected_peak_lr = swapped.get(candidate["candidate_id"], candidate["peak_lr"])
        _assert_chain_is_rehashed(session, candidate, plan, expected_peak_lr=expected_peak_lr)

    for candidate in (first, second):
        with pytest.raises(
            R.BindingFailure,
            match=r"result artifact peak_lr .* does not match planned",
        ):
            R.load_completed_result(session, planned=candidate, plan=plan)
    for candidate in candidates[2:]:
        assert R.load_completed_result(session, planned=candidate, plan=plan)["eligible"] is True


@pytest.mark.parametrize(
    ("honest_eligible", "forged_eligible"),
    [(True, False), (False, True)],
    ids=("true_to_false", "false_to_true"),
)
def test_stored_lr_eligibility_flip_is_rejected_after_full_chain_rehash(
    tmp_path, honest_eligible, forged_eligible
):
    session = _artifact_bound_session(tmp_path / "authorized")
    candidates, plan = _published_lr_plan(session, [2e-4])
    candidate = candidates[0]
    _publish_synthetic_lr_chain(candidate, session, plan, eligible=honest_eligible)
    assert (
        R.load_completed_result(session, planned=candidate, plan=plan)["eligible"]
        is honest_eligible
    )

    def flip_stored_eligible(_candidate, result, _meta, _terminal):
        result["eligible"] = forged_eligible

    _rebind_candidate_chains(session, candidates, flip_stored_eligible)
    rewritten = _assert_chain_is_rehashed(session, candidate, plan)
    assert rewritten["eligible"] is forged_eligible

    with pytest.raises(
        R.BindingFailure,
        match=(
            f"stored eligible={forged_eligible!r} disagrees with "
            f"recomputed verdict {honest_eligible!r}"
        ),
    ):
        R.load_completed_result(session, planned=candidate, plan=plan)
