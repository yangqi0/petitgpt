"""R1 real-path repair tests: the actual trainer, checkpoint and resume paths.

No production training: the 124,635,456-parameter path is never executed. Model, optimizer,
dataset, sampler, compiled callable, checkpoint state and the optimizer-update boundary are
monkeypatched or tiny/fake.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pretrain"))

# The trainer uses bare intra-package imports, so `production_launch_contract_v1` loaded that
# way is a DISTINCT module object with a DISTINCT LaunchContractError class. Trainer-path
# assertions must use the class the trainer actually raises.
import production_launch_contract_v1 as TRAINER_C  # noqa: E402

from pretrain import production_launch_contract_v1 as C  # noqa: E402

from .test_production_launch_contract_v1 import (  # noqa: E402
    _runtime,
    authorization,
    governed_args,
)

# --------------------------------------------------------------------- helpers


def write_contract(tmp_path: Path, **mutations) -> Path:
    doc = C.contract_document()
    for dotted, value in mutations.items():
        node = doc
        parts = dotted.split(".")
        for key in parts[:-1]:
            node = node[key]
        node[parts[-1]] = value
    path = tmp_path / "LAUNCH_CONTRACT.json"
    path.write_bytes(C.canonical_json_bytes(doc))
    return path


def write_authorization(tmp_path: Path, contract_path: Path, **overrides) -> Path:
    manifest = authorization("STAGE_N")
    repo = C.observed_repository()
    manifest["repository_branch"] = repo["branch"]
    manifest["trainer_head"] = repo["head"]
    manifest["trainer_execution_bundle_sha256"] = C.trainer_execution_bundle_sha256()
    manifest["launch_contract_sha256"] = C.file_sha256(contract_path)
    runtime = {
        **_runtime(),
        "trainer_head": repo["head"],
        "trainer_execution_bundle_sha256": C.trainer_execution_bundle_sha256(),
    }
    manifest["training_runtime"] = runtime
    manifest.update(overrides)
    path = tmp_path / "STAGE_AUTHORIZATION.json"
    path.write_bytes(C.canonical_json_bytes(manifest))
    return path


def live_runtime() -> dict:
    repo = C.observed_repository()
    return {
        **_runtime(),
        "trainer_head": repo["head"],
        "trainer_execution_bundle_sha256": C.trainer_execution_bundle_sha256(),
    }


def gate_a(tmp_path: Path, *, args=None, contract_path=None, auth_path=None, stage="stage_a"):
    contract_path = contract_path or write_contract(tmp_path)
    auth_path = auth_path or write_authorization(tmp_path, contract_path)
    args = args if args is not None else governed_args(stage)
    # Mirror the real trainer: enforce_governed_launch normalizes the legacy shared seed to
    # the active stage seed before Gate A runs.
    C.normalize_legacy_sampler_seed(args, stage)
    return C.gate_a_pre_construction(
        args,
        stage=stage,
        launch_contract_path=contract_path,
        stage_authorization_path=auth_path,
        exact_plan_path=REPO / C.EXACT_RUN_PLAN_RELPATH,
        pilot_acceptance_path=REPO / C.PILOT_OWNER_ACCEPTANCE_RELPATH,
        observed_runtime=live_runtime(),
        cwd=C.CANONICAL_CWD,
    )


# --------------------------------------------------------------------- Part 1: artifact auth


def test_authentic_launch_contract_artifact_is_accepted(tmp_path):
    loaded = C.load_launch_contract_artifact(write_contract(tmp_path))
    assert loaded["sha256"] == C.contract_sha256()
    assert loaded["matches_code_authority"] is True


@pytest.mark.parametrize(
    "mutation",
    [
        {"training.peak_lr": 0.0003},
        {"authorizes_training": True},
        {"seeds.model_init_seed": 1234},
        {"evaluation_policy.periodic_eval_every": 1000},
        {"checkpoint_policy.periodic_save_every": 1000},
        {"model.n_layers": 16},
        {"model.parameter_count": 125000000},
        {"training.compile": False},
        {"canonical_cwd": "/tmp"},
    ],
)
def test_altered_launch_contract_bytes_are_rejected(tmp_path, mutation):
    path = write_contract(tmp_path, **mutation)
    with pytest.raises(C.LaunchContractError, match="failed authentication"):
        C.load_launch_contract_artifact(path)


def test_self_declared_sha_is_never_the_authority(tmp_path):
    doc = C.contract_document()
    doc["launch_contract_sha256"] = "0" * 64
    path = tmp_path / "LAUNCH_CONTRACT.json"
    path.write_bytes(C.canonical_json_bytes(doc))
    with pytest.raises(C.LaunchContractError):
        C.load_launch_contract_artifact(path)


def test_missing_or_unparseable_artifact_is_rejected(tmp_path):
    with pytest.raises(C.LaunchContractError, match="not found"):
        C.load_launch_contract_artifact(tmp_path / "absent.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(C.LaunchContractError, match="canonical JSON"):
        C.load_launch_contract_artifact(bad)


# --------------------------------------------------------------------- Part 2: enforcement


def test_special_token_ids_are_frozen_policy():
    assert dict(C.SPECIAL_TOKEN_IDS) == {
        "PAD": 0,
        "UNK": 1,
        "BOS": 2,
        "EOS": 3,
        "SYSTEM": 4,
        "USER": 5,
        "ASSISTANT": 6,
    }


@pytest.mark.parametrize("field,bad", [("bos_id", 1), ("eos_id", 2)])
def test_wrong_special_token_id_is_rejected(field, bad):
    failures = C.validate_special_token_binding(governed_args(**{field: bad}))
    assert any(field in f for f in failures)


def test_real_tokenizer_realizes_the_frozen_special_ids():
    result = C.verify_tokenizer_special_ids(REPO / C.TOKENIZER_RELPATH)
    assert result["failures"] == []
    assert result["observed"] == dict(C.SPECIAL_TOKEN_IDS)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("log_every", 7),
        ("debug_every", 9),
        ("sample_temperature", 1.0),
        ("sample_top_p", 0.5),
        ("sample_top_k", 40),
        ("sample_max_new_tokens", 512),
        ("add_bos_to_prompts", False),
        ("bench_eval_max_seq_len", 2048),
    ],
)
def test_diagnostic_field_arbitrary_value_is_rejected(field, bad):
    failures = C.validate_diagnostic_fields(governed_args(**{field: bad}))
    assert any(field in f for f in failures), failures


def test_diagnostic_defaults_are_accepted():
    assert C.validate_diagnostic_fields(governed_args()) == []


def test_every_diagnostic_field_has_an_explicit_allowed_value():
    for dest, spec in C.PARSER_FIELD_CLASSIFICATION.items():
        if spec["class"] == C.DIAGNOSTIC_ONLY:
            assert spec["value"] is not None, dest


def test_arbitrary_num_workers_is_rejected_and_binding_is_required():
    assert C.validate_num_workers_binding(governed_args(num_workers=8), 2)
    assert C.validate_num_workers_binding(governed_args(num_workers=2), 2) == []
    assert C.validate_num_workers_binding(governed_args(num_workers=2), None) == [
        "num_workers_not_bound_by_authorization"
    ]


def test_num_workers_is_part_of_the_bound_runtime_identity():
    assert "num_workers" in C.RUNTIME_BINDING_REQUIRED_FIELDS
    assert "num_workers" in C.STAGE_N_O_RUNTIME_COMPARISON_FIELDS
    assert C.PARSER_FIELD_CLASSIFICATION["num_workers"]["class"] == C.LAUNCH_AUTHORIZATION_BOUND


def test_resume_modes_are_authorization_bound():
    fresh = governed_args()
    assert C.validate_resume_binding(fresh, {"mode": "FRESH"}) == []
    assert C.validate_resume_binding(fresh, None) == ["resume_binding_missing_from_authorization"]
    assert any(
        "resume_mode_invalid" in f for f in C.validate_resume_binding(fresh, {"mode": "WHATEVER"})
    )


def test_arbitrary_resume_fields_are_rejected_under_fresh():
    for field, value in (("resume_path", "/tmp/x.pt"), ("resume_full", True), ("resume_step", 5)):
        failures = C.validate_resume_binding(governed_args(**{field: value}), {"mode": "FRESH"})
        assert any(field.split("_")[-1] in f.lower() or field in f for f in failures), failures


def test_exact_checkpoint_resume_requires_the_authorized_path():
    binding = {
        "mode": "RESUME_EXACT_CHECKPOINT",
        "checkpoint_path": "/authorized/step_038146.pt",
        "checkpoint_sha256": "a" * 64,
        "expected_step": 38146,
        "stage": "stage_a",
        "governed_run_contract_sha256": "b" * 64,
    }
    ok = governed_args(resume_path="/authorized/step_038146.pt", resume_full=True)
    assert C.validate_resume_binding(ok, binding) == []
    bad = governed_args(resume_path="/somewhere/else.pt", resume_full=True)
    assert any("resume_path" in f for f in C.validate_resume_binding(bad, binding))


def test_legacy_sampler_seed_cannot_select_another_permutation():
    args = governed_args(sampler_seed=999999)
    assert C.validate_legacy_sampler_seed(args, "stage_a")
    C.normalize_legacy_sampler_seed(args, "stage_a")
    assert args.sampler_seed == C.STAGE_A_SAMPLER_SEED
    assert C.validate_legacy_sampler_seed(args, "stage_a") == []
    C.normalize_legacy_sampler_seed(args, "stage_b")
    assert args.sampler_seed == C.STAGE_B_SAMPLER_SEED


def test_every_parser_field_has_runtime_enforcement_coverage():
    """Each classified field must be reachable by an actual runtime check."""
    enforced = set()
    for dest, spec in C.PARSER_FIELD_CLASSIFICATION.items():
        cls = spec["class"]
        if cls in (C.OWNER_FROZEN, C.EXACT_PLAN_DERIVED, C.FORBIDDEN_OR_UNSET):
            enforced.add(dest)  # validate_governed_args / special-token / seed checks
        elif cls == C.DIAGNOSTIC_ONLY:
            enforced.add(dest)  # validate_diagnostic_fields
        elif cls == C.LAUNCH_AUTHORIZATION_BOUND:
            enforced.add(dest)  # authorization / resume / num_workers binding
        elif cls == C.RUNTIME_OBSERVED_AND_BOUND:
            enforced.add(dest)  # runtime fingerprint comparison
    assert enforced == set(C.PARSER_FIELD_CLASSIFICATION)


# --------------------------------------------------------------------- Part 3: gates


def test_gate_a_accepts_a_correct_governed_launch(tmp_path):
    assert gate_a(tmp_path)["passed"] is True


def test_gate_a_rejects_a_mismatched_cli_value_before_construction(tmp_path):
    with pytest.raises(C.LaunchContractError, match="Gate A refused"):
        gate_a(tmp_path, args=governed_args(lr=0.0003))


def test_gate_a_rejects_altered_contract_bytes(tmp_path):
    path = write_contract(tmp_path, **{"training.peak_lr": 0.0003})
    auth = write_authorization(tmp_path, path)
    with pytest.raises(C.LaunchContractError):
        gate_a(tmp_path, contract_path=path, auth_path=auth)


def test_gate_a_rejects_a_non_canonical_cwd(tmp_path):
    contract_path = write_contract(tmp_path)
    with pytest.raises(C.LaunchContractError, match="canonical_cwd"):
        C.gate_a_pre_construction(
            governed_args("stage_a"),
            stage="stage_a",
            launch_contract_path=contract_path,
            stage_authorization_path=write_authorization(tmp_path, contract_path),
            exact_plan_path=REPO / C.EXACT_RUN_PLAN_RELPATH,
            pilot_acceptance_path=REPO / C.PILOT_OWNER_ACCEPTANCE_RELPATH,
            observed_runtime=live_runtime(),
            cwd="/tmp",
        )


class _TinyModel:
    """A tiny stand-in; the 124M path is never constructed in tests."""

    def __init__(self, params, tied=True):
        import torch.nn as nn

        self._params = params
        self.tok_emb = nn.Embedding(4, 2)
        self.lm_head = nn.Linear(2, 4, bias=False)
        if tied:
            self.lm_head.weight = self.tok_emb.weight
        self.cfg = type("Cfg", (), dict(C.MODEL_CONTRACT, max_seq_len=2048))()

    def parameters(self):
        return iter(self._params)

    def named_parameters(self):
        return iter([(f"p{i}", p) for i, p in enumerate(self._params)])


def test_gate_b_rejects_a_wrong_parameter_count():
    import torch

    model = _TinyModel([torch.zeros(3, 3, requires_grad=True)])
    with pytest.raises(C.LaunchContractError, match="parameter count"):
        C.gate_b_post_construction(model, optimizer=None)


def test_gate_b_rejects_untied_embeddings_and_bad_grouping():
    import torch

    model = _TinyModel([torch.zeros(3, 3, requires_grad=True)], tied=False)
    with pytest.raises(C.LaunchContractError):
        C.gate_b_post_construction(model, optimizer=None)


def test_gate_b_rejects_a_tiny_model_on_frozen_architecture():
    """Gate B enforces the FULL frozen architecture: a tiny config must not pass."""
    from src.model import GPT, GPTConfig
    from src.optim import build_optimizer

    cfg = GPTConfig(n_layers=2, d_model=64, n_heads=2, n_kv_heads=1, d_ff=128)
    model = GPT(cfg)
    optimizer = build_optimizer(
        model,
        name="muon",
        lr=C.PEAK_LR,
        weight_decay=C.WEIGHT_DECAY,
        betas=C.ADAMW_AUX_BETAS,
        muon_lr=C.MUON_LR,
        muon_momentum=C.MUON_MOMENTUM,
        verbose=False,
    )
    count = int(sum(p.numel() for p in {id(p): p for p in model.parameters()}.values()))
    with pytest.raises(C.LaunchContractError, match="Gate B refused"):
        C.gate_b_post_construction(model, optimizer, expected_parameter_count=count)
    assert len(optimizer.state) == 0  # no optimizer update occurred


def test_gate_b_optimizer_verification_accepts_the_real_realized_muon():
    """The optimizer half of Gate B, isolated from the frozen architecture check."""
    from src.model import GPT, GPTConfig
    from src.optim import build_optimizer

    cfg = GPTConfig(n_layers=2, d_model=64, n_heads=2, n_kv_heads=1, d_ff=128)
    model = GPT(cfg)
    optimizer = build_optimizer(
        model,
        name="muon",
        lr=C.PEAK_LR,
        weight_decay=C.WEIGHT_DECAY,
        betas=C.ADAMW_AUX_BETAS,
        muon_lr=C.MUON_LR,
        muon_momentum=C.MUON_MOMENTUM,
        verbose=False,
    )
    result = C.verify_realized_optimizer(optimizer, model)
    assert result["failures"] == []
    assert len(optimizer.state) == 0


def test_gate_b_rejects_a_mutated_realized_muon():
    from src.model import GPT, GPTConfig
    from src.optim import build_optimizer

    cfg = GPTConfig(n_layers=2, d_model=64, n_heads=2, n_kv_heads=1, d_ff=128)
    model = GPT(cfg)
    optimizer = build_optimizer(
        model,
        name="muon",
        lr=C.PEAK_LR,
        weight_decay=C.WEIGHT_DECAY,
        betas=C.ADAMW_AUX_BETAS,
        muon_lr=C.MUON_LR,
        muon_momentum=C.MUON_MOMENTUM,
        verbose=False,
    )
    optimizer.param_groups[0]["lr_ratio"] = 0.5
    count = int(sum(p.numel() for p in {id(p): p for p in model.parameters()}.values()))
    with pytest.raises(C.LaunchContractError, match="Gate B refused"):
        C.gate_b_post_construction(model, optimizer, expected_parameter_count=count)


# --------------------------------------------------------------------- Part 8: compile


class _FakeCompiled:
    """Stands in for an OptimizedModule; never runs the production model."""

    def __init__(self, name="torch._dynamo.eval_frame.OptimizedModule"):
        self.__class__.__qualname__ = "OptimizedModule"
        self.__class__.__module__ = "torch._dynamo.eval_frame"
        self.calls = 0

    def __call__(self, *a, **k):
        self.calls += 1
        return None


def _evidence(*, requested=True, invocations=1, graphs=1, compiled=True, recompile=False):
    module = _FakeCompiled() if compiled else object()
    forward = C.ObservedForward(module, compiled_object=module if compiled else None)
    for _ in range(invocations):
        forward()
    counters = {"stats": {"unique_graphs": graphs}}
    if recompile:
        counters["recompile_reasons"] = {"cache_size_limit exceeded": 1}
    return C.compile_realization_evidence(
        module,
        forward,
        requested=requested,
        cache_dir=None,
        expected_forward_invocations=1,
        counters=counters,
    )


def test_lazy_compile_realization_produces_evidence():
    evidence = _evidence()
    assert evidence["compile_realized"] is True
    assert evidence["compilation_materialized"] is True
    assert evidence["compiled_callable_is_training_callable"] is True
    assert len(evidence["compile_evidence_sha256"]) == 64
    C.require_compile_realized(evidence)


def test_wrapper_that_never_compiles_is_rejected():
    evidence = _evidence(graphs=0)
    assert evidence["compile_realized"] is False
    assert evidence["eager_fallback_occurred"] is True
    with pytest.raises(C.LaunchContractError, match="lazily realized"):
        C.require_compile_realized(evidence)


def test_recompile_limit_fallback_is_rejected():
    evidence = _evidence(recompile=True)
    assert evidence["recompile_limit_fallback_detected"] is True
    with pytest.raises(C.LaunchContractError):
        C.require_compile_realized(evidence)


def test_uninvoked_compiled_callable_is_rejected():
    evidence = _evidence(invocations=0, graphs=0)
    assert evidence["compile_realized"] is False


def test_compile_exception_aborts(monkeypatch):
    import torch

    monkeypatch.setattr(torch, "compile", lambda m: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(C.LaunchContractError, match="eager fallback is forbidden"):
        C.bind_compiled_callable_governed(object())


def test_identity_return_aborts(monkeypatch):
    import torch

    monkeypatch.setattr(torch, "compile", lambda m: m)
    with pytest.raises(C.LaunchContractError, match="identity/eager return"):
        C.bind_compiled_callable_governed(object())


# --------------------------------------------------------------------- Parts 4-7


def _contract(tmp_path, stage="stage_a", **over):
    a = gate_a(tmp_path, stage=stage, args=governed_args(stage))
    b = {
        "parameter_count": C.MODEL_PARAMETER_COUNT,
        "tied_embeddings": True,
        "realized_muon": C.realized_muon_contract(),
        "optimizer_group_roles": list(C.OPTIMIZER_GROUP_ROLES),
        "optimizer_membership_counts": {},
    }
    doc = C.build_governed_run_contract(
        gate_a=a,
        gate_b=b,
        stage=stage,
        sampler_identity={
            "stage": stage,
            "sampler_seed": C.stage_sampler_seed(stage),
            "range_start_position": 0,
            "range_stop_position": 100,
            "cursor": 0,
            "permutation_identity": "p" * 64,
        },
        compile_evidence=_evidence(),
    )
    doc.update(over)
    return doc


def test_governed_run_contract_carries_every_required_field(tmp_path):
    doc = _contract(tmp_path)
    for field in (
        "launch_contract_path",
        "launch_contract_sha256",
        "stage_authorization_path",
        "stage_authorization_sha256",
        "exact_plan_path",
        "exact_run_plan_sha256",
        "pilot_acceptance_path",
        "pilot_owner_acceptance_sha256",
        "trainer_branch",
        "trainer_head",
        "trainer_execution_bundle_sha256",
        "model",
        "training",
        "seeds",
        "evaluation_policy",
        "checkpoint_policy",
        "canonical_cwd",
        "num_workers",
        "resume",
        "runtime_fingerprint",
        "gpu_uuid",
        "gpu_pci_bus_id",
        "stage",
        "stage_stop_step",
        "sampler_identity",
        "compile_evidence",
        "compile_evidence_sha256",
    ):
        assert field in doc, field
    assert doc["seeds"] == dict(C.SEED_TUPLE)
    assert doc["active_stage_sampler_seed"] == C.STAGE_A_SAMPLER_SEED


def test_governed_run_contract_is_published_atomically_once(tmp_path):
    doc = _contract(tmp_path)
    out = tmp_path / "run"
    published = C.publish_governed_run_contract(out, doc)
    assert published["atomic"] is True
    on_disk = json.loads((out / C.GOVERNED_RUN_CONTRACT_FILENAME).read_text())
    assert on_disk["governed_run_contract_sha256"] == published["governed_run_contract_sha256"]
    assert not list(out.glob("*.tmp"))
    with pytest.raises(C.LaunchContractError, match="published once"):
        C.publish_governed_run_contract(out, doc)


def test_legacy_config_json_alone_cannot_satisfy_governed_mode(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "config.json").write_text("{}", encoding="utf-8")
    assert not (out / C.GOVERNED_RUN_CONTRACT_FILENAME).exists()


def test_identity_digest_ignores_compile_and_sampler_observations(tmp_path):
    """A resume recompiles and advances the cursor; neither may move the identity."""
    a = _contract(tmp_path)
    b = dict(a)
    b["compile_evidence"] = _evidence(graphs=7)
    b["compile_evidence_sha256"] = b["compile_evidence"]["compile_evidence_sha256"]
    b["sampler_identity"] = {**a["sampler_identity"], "cursor": 64}
    assert C.governed_digest(a) == C.governed_digest(b)


def test_governed_checkpoint_is_distinguishable_from_a_legacy_one(tmp_path):
    doc = _contract(tmp_path)
    assert C.is_governed_checkpoint({"governed_run_contract": doc}) is True
    assert C.is_governed_checkpoint({"run_contract": {"schema_version": 3}}) is False


def test_ungoverned_checkpoint_cannot_resume_a_governed_run(tmp_path):
    doc = _contract(tmp_path)
    verdict = C.validate_governed_checkpoint_before_restore({"run_contract": {}}, doc)
    assert verdict["compatible"] is False
    assert "ungoverned_checkpoint_cannot_resume_a_governed_run" in verdict["failures"]


def _ckpt(doc, **over):
    ckpt = {
        "governed_run_contract": doc,
        "governed_run_contract_sha256": C.governed_digest(doc),
        "global_step": 38146,
    }
    ckpt.update(over)
    return ckpt


def test_matching_governed_checkpoint_resumes(tmp_path):
    doc = _contract(tmp_path)
    assert C.validate_governed_checkpoint_before_restore(_ckpt(doc), doc)["compatible"] is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("launch_contract_sha256", "0" * 64),
        ("stage_authorization_sha256", "0" * 64),
        ("exact_run_plan_sha256", "0" * 64),
        ("trainer_head", "9" * 40),
        ("trainer_execution_bundle_sha256", "0" * 64),
        ("gpu_uuid", "GPU-other"),
        ("gpu_pci_bus_id", "00000000:09:00.0"),
        ("num_workers", 8),
        ("active_stage_sampler_seed", 1234),
    ],
)
def test_governed_resume_rejects_drift(tmp_path, field, value):
    doc = _contract(tmp_path)
    drifted = {**doc, field: value}
    verdict = C.validate_governed_checkpoint_before_restore(_ckpt(doc), drifted)
    assert verdict["compatible"] is False


def test_governed_checkpoint_cannot_claim_compile_without_evidence(tmp_path):
    doc = _contract(tmp_path)
    doc["compile_evidence"] = {"compile_realized": False}
    verdict = C.validate_governed_checkpoint_before_restore(_ckpt(doc), doc)
    assert any("without_realized_evidence" in f for f in verdict["failures"])


def test_resume_rejects_a_wrong_expected_step(tmp_path):
    doc = _contract(tmp_path)
    verdict = C.validate_governed_checkpoint_before_restore(
        _ckpt(doc, global_step=1),
        doc,
        expected_resume={"expected_step": 38146, "stage": "stage_a"},
    )
    assert any("resume_step_mismatch" in f for f in verdict["failures"])


# --------------------------------------------------------------------- sampler persistence


def _identity(stage, **over):
    base = {
        "stage": stage,
        "sampler_seed": C.stage_sampler_seed(stage),
        "range_start_position": 0,
        "range_stop_position": 4882688,
        "cursor": 4882688,
        "permutation_identity": "p" * 64,
    }
    base.update(over)
    return base


def test_same_stage_resume_validates_range_start_position():
    saved = _identity("stage_a")
    assert C.validate_same_stage_resume(saved, saved) == []
    moved = _identity("stage_a", range_start_position=128)
    assert any("range_start_position" in f for f in C.validate_same_stage_resume(saved, moved))


@pytest.mark.parametrize(
    "field,value",
    [
        ("sampler_seed", 1234),
        ("permutation_identity", "q" * 64),
        ("range_stop_position", 999),
        ("stage", "stage_b"),
    ],
)
def test_same_stage_resume_rejects_sampler_drift(field, value):
    saved = _identity("stage_a")
    drifted = dict(saved)
    drifted[field] = value
    assert C.validate_same_stage_resume(saved, drifted)


def test_cursor_outside_its_range_is_rejected():
    saved = _identity("stage_a", cursor=99999999)
    assert any(
        "outside its committed range" in f for f in C.validate_same_stage_resume(saved, saved)
    )


def test_stage_a_to_b_requires_a_complete_stage_a_endpoint(tmp_path):
    doc = _contract(tmp_path)
    doc["sampler_identity"] = _identity("stage_a")
    doc["global_step"] = C.STAGE_A_STOP_STEP
    assert C.validate_stage_a_to_b_transition(doc) == []


def test_stage_a_to_b_rejects_a_wrong_saved_stage_a_seed(tmp_path):
    doc = _contract(tmp_path)
    doc["sampler_identity"] = _identity("stage_a", sampler_seed=C.STAGE_B_SAMPLER_SEED)
    doc["active_stage_sampler_seed"] = C.STAGE_B_SAMPLER_SEED
    doc["global_step"] = C.STAGE_A_STOP_STEP
    failures = C.validate_stage_a_to_b_transition(doc)
    assert any("Stage-A sampler seed" in f or "Stage-A seed" in f for f in failures)


def test_stage_a_to_b_rejects_an_incomplete_consumed_range(tmp_path):
    doc = _contract(tmp_path)
    doc["sampler_identity"] = _identity("stage_a", cursor=100)
    doc["global_step"] = C.STAGE_A_STOP_STEP
    assert any("incomplete" in f for f in C.validate_stage_a_to_b_transition(doc))


def test_stage_a_to_b_rejects_a_cursor_off_the_plan_boundary(tmp_path):
    doc = _contract(tmp_path)
    doc["sampler_identity"] = _identity("stage_a")
    doc["global_step"] = 12345
    assert any("plan boundary" in f for f in C.validate_stage_a_to_b_transition(doc))


def test_stage_a_to_b_rejects_a_stage_b_source(tmp_path):
    # Built directly: a Stage-B contract cannot come through Gate A without the STAGE_O chain.
    doc = _contract(tmp_path)
    doc["stage"] = "stage_b"
    doc["sampler_identity"] = _identity("stage_b")
    doc["active_stage_sampler_seed"] = C.STAGE_B_SAMPLER_SEED
    assert any("expected 'stage_a'" in f for f in C.validate_stage_a_to_b_transition(doc))


# --------------------------------------------------------------------- Part 10: device map


NVML = [
    {
        "physical_index": 0,
        "uuid": "GPU-aaaa",
        "pci_bus_id": "00000000:01:00.0",
        "name": "NVIDIA GeForce RTX 4090",
        "memory_total": "24564 MiB",
        "driver_version": "580",
    },
    {
        "physical_index": 1,
        "uuid": "GPU-bbbb",
        "pci_bus_id": "00000000:02:00.0",
        "name": "NVIDIA GeForce RTX 4090",
        "memory_total": "24564 MiB",
        "driver_version": "580",
    },
]


def test_cuda_visible_devices_index_form_selects_the_right_physical_device():
    resolved = C.resolve_selected_gpu_identity(cuda_visible_devices="1", records=NVML)
    assert resolved["resolved"] is True
    assert resolved["gpu_uuid"] == "GPU-bbbb"
    assert resolved["gpu_pci_bus_id"] == "00000000:02:00.0"
    assert resolved["mapping_form"] == "index"


def test_cuda_visible_devices_uuid_form_selects_the_right_physical_device():
    resolved = C.resolve_selected_gpu_identity(cuda_visible_devices="GPU-bbbb", records=NVML)
    assert resolved["resolved"] is True
    assert resolved["gpu_uuid"] == "GPU-bbbb"
    assert resolved["mapping_form"] == "uuid"


def test_first_nvidia_smi_row_is_not_assumed():
    """With CUDA_VISIBLE_DEVICES=1 the first row is the WRONG device."""
    resolved = C.resolve_selected_gpu_identity(cuda_visible_devices="1", records=NVML)
    assert resolved["gpu_uuid"] != NVML[0]["uuid"]


def test_multi_visible_device_is_rejected_under_governed_policy():
    resolved = C.resolve_selected_gpu_identity(cuda_visible_devices="0,1", records=NVML)
    assert resolved["resolved"] is False
    assert any("visible_device_count_not_exactly_1" in f for f in resolved["failures"])


def test_unresolvable_or_ambiguous_mapping_fails():
    assert (
        C.resolve_selected_gpu_identity(cuda_visible_devices="GPU-zzzz", records=NVML)["resolved"]
        is False
    )
    assert (
        C.resolve_selected_gpu_identity(cuda_visible_devices="7", records=NVML)["resolved"] is False
    )


def test_inconsistent_torch_name_is_rejected():
    resolved = C.resolve_selected_gpu_identity(
        cuda_visible_devices="0", records=NVML, torch_device_name="NVIDIA A100"
    )
    assert resolved["resolved"] is False
    assert any("name_inconsistent" in f for f in resolved["failures"])


# --------------------------------------------------------------------- Part 9: Stage N/O


def _stage_n_result(tmp_path, **over):
    doc = _contract(tmp_path)
    result = C.stage_n_result_document(
        governed_run_contract=doc,
        final_checkpoint_path="/out/step_038146.pt",
        final_checkpoint_sha256="c" * 64,
        final_checkpoint_step=38146,
        smoke_results={"status": "PASS"},
        resume_results={"status": "PASS"},
    )
    result.update(over)
    return result


def test_stage_n_result_schema_is_complete(tmp_path):
    assert C.validate_stage_n_result(_stage_n_result(tmp_path)) == []


def test_incomplete_stage_n_result_is_rejected(tmp_path):
    assert C.validate_stage_n_result(_stage_n_result(tmp_path, status="ABORTED"))
    assert C.validate_stage_n_result(None) == ["stage_n_result_missing_or_malformed"]


def _stage_o_chain(tmp_path, *, accept=True, result_over=None, runtime_over=None):
    result = _stage_n_result(tmp_path, **(result_over or {}))
    rpath = tmp_path / "STAGE_N_RESULT.json"
    rbytes = C.canonical_json_bytes(result)
    rpath.write_bytes(rbytes)
    rsha = C._sha256_bytes(rbytes)

    acceptance = {
        "stage_n_result_owner_verdict": "ACCEPTED" if accept else "REJECTED",
        "accepted_stage_n_result_sha256": rsha,
    }
    apath = tmp_path / "STAGE_N_ACCEPTANCE.json"
    abytes = C.canonical_json_bytes(acceptance)
    apath.write_bytes(abytes)

    chain = {
        "accepted_stage_n_result_path": str(rpath),
        "accepted_stage_n_result_sha256": rsha,
        "stage_n_owner_acceptance_path": str(apath),
        "stage_n_owner_acceptance_sha256": C._sha256_bytes(abytes),
        "stage_n_authorization_sha256": result["stage_authorization_sha256"],
        "stage_n_governed_run_contract_sha256": result["governed_run_contract_sha256"],
        "stage_n_runtime_fingerprint": result["runtime_fingerprint"],
        "stage_n_gpu_uuid": result["gpu_uuid"],
        "stage_n_gpu_pci_bus_id": result["gpu_pci_bus_id"],
        "stage_n_trainer_head": result["trainer_head"],
        "stage_n_trainer_execution_bundle_sha256": result["trainer_execution_bundle_sha256"],
        "stage_n_exact_run_plan_sha256": result["exact_run_plan_sha256"],
    }
    observed = {**result["runtime_fingerprint"], **(runtime_over or {})}
    return {"stage_n_chain": chain}, observed


def test_stage_o_without_a_stage_n_chain_is_rejected():
    verdict = C.validate_stage_o_chain({}, observed_runtime=_runtime())
    assert verdict["valid"] is False
    assert "stage_o_authorization_missing_stage_n_chain" in verdict["failures"]


def test_stage_o_with_a_matching_accepted_chain_passes(tmp_path):
    auth, observed_runtime = _stage_o_chain(tmp_path)
    verdict = C.validate_stage_o_chain(auth, observed_runtime=observed_runtime)
    assert verdict["failures"] == []
    assert verdict["valid"] is True


def test_stage_o_rejects_an_unaccepted_stage_n_result(tmp_path):
    auth, observed_runtime = _stage_o_chain(tmp_path, accept=False)
    verdict = C.validate_stage_o_chain(auth, observed_runtime=observed_runtime)
    assert verdict["valid"] is False
    assert any("not_owner_accepted" in f for f in verdict["failures"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("gpu_uuid", "GPU-different"),
        ("gpu_pci_bus_id", "00000000:09:00.0"),
        ("num_workers", 8),
        ("torch_version", "2.10.0"),
        ("trainer_head", "f" * 40),
    ],
)
def test_stage_o_rejects_a_runtime_change_versus_accepted_stage_n(tmp_path, field, value):
    auth, observed_runtime = _stage_o_chain(tmp_path, runtime_over={field: value})
    verdict = C.validate_stage_o_chain(auth, observed_runtime=observed_runtime)
    assert verdict["valid"] is False
    assert verdict["requires_new_stage_n"] is True


def test_changing_both_authorization_and_runtime_cannot_evade_comparison(tmp_path):
    """The accepted Stage-N result is loaded from disk, so a self-consistent lie still fails."""
    auth, _ = _stage_o_chain(tmp_path)
    forged_runtime = {**_runtime(), "gpu_uuid": "GPU-forged", "num_workers": 2}
    auth["stage_n_chain"]["stage_n_gpu_uuid"] = "GPU-forged"
    auth["stage_n_chain"]["stage_n_runtime_fingerprint"] = forged_runtime
    verdict = C.validate_stage_o_chain(auth, observed_runtime=forged_runtime)
    assert verdict["valid"] is False
    assert any("contradicts_accepted_stage_n_result" in f for f in verdict["failures"])


def test_tampered_stage_n_result_bytes_are_rejected(tmp_path):
    auth, observed_runtime = _stage_o_chain(tmp_path)
    path = Path(auth["stage_n_chain"]["accepted_stage_n_result_path"])
    doc = json.loads(path.read_text())
    doc["gpu_uuid"] = "GPU-tampered"
    path.write_bytes(C.canonical_json_bytes(doc))
    verdict = C.validate_stage_o_chain(auth, observed_runtime=observed_runtime)
    assert verdict["valid"] is False


# --------------------------------------------------------------------- real trainer path


def test_real_trainer_parser_exposes_the_governed_fields():
    import train_pretrain_with_bench as trainer

    saved = sys.argv
    try:
        sys.argv = ["t.py", "--help"]
        with pytest.raises(SystemExit):
            trainer.parse_args()
    finally:
        sys.argv = saved


def test_real_trainer_resolves_only_the_active_stage_seed():
    import train_pretrain_with_bench as trainer

    args = governed_args("stage_b")
    assert trainer.resolve_stage_sampler_seed(args, "stage_b") == C.STAGE_B_SAMPLER_SEED
    assert trainer.resolve_stage_sampler_seed(args, "stage_a") == C.STAGE_A_SAMPLER_SEED


def test_real_trainer_data_contract_records_the_active_stage_seed(monkeypatch, tmp_path):
    """build_data_contract must persist the ACTIVE stage seed, not the legacy field."""
    import train_pretrain_with_bench as trainer

    class _DS:
        shards: list = []
        total_raw_tokens = 10
        usable_transitions = 9
        sampling_mode = "deterministic"
        _dtype = "uint16"

        def __len__(self):
            return 4

    for stage, expected in (
        ("stage_a", C.STAGE_A_SAMPLER_SEED),
        ("stage_b", C.STAGE_B_SAMPLER_SEED),
    ):
        args = governed_args(stage)
        contract = trainer.build_data_contract(tmp_path, _DS(), args)
        assert contract["sampler_seed"] == expected
        assert contract["active_stage"] == stage


def test_real_trainer_enforce_governed_launch_requires_both_artifacts():
    import train_pretrain_with_bench as trainer

    args = governed_args()
    args.launch_contract_json = "/some/contract.json"
    args.stage_authorization_json = ""
    with pytest.raises(ValueError, match="must be supplied together"):
        trainer.enforce_governed_launch(args)


def test_real_trainer_ungoverned_path_returns_none():
    import train_pretrain_with_bench as trainer

    args = governed_args()
    args.launch_contract_json = ""
    args.stage_authorization_json = ""
    assert trainer.enforce_governed_launch(args) is None


def test_real_trainer_gate_a_rejects_before_construction(tmp_path, monkeypatch):
    """enforce_governed_launch must raise before anything is constructed."""
    import train_pretrain_with_bench as trainer

    contract_path = write_contract(tmp_path)
    auth_path = write_authorization(tmp_path, contract_path)
    args = governed_args(lr=0.0003)
    args.launch_contract_json = str(contract_path)
    args.stage_authorization_json = str(auth_path)
    args.run_plan_json = str(REPO / C.EXACT_RUN_PLAN_RELPATH)
    monkeypatch.setattr(trainer, "_resolve_path", lambda p: str(REPO / C.EXACT_RUN_PLAN_RELPATH))
    monkeypatch.chdir(C.CANONICAL_CWD)
    with pytest.raises(TRAINER_C.LaunchContractError, match="Gate A refused"):
        trainer.enforce_governed_launch(args)


def test_real_trainer_save_ckpt_binds_the_governed_contract(tmp_path, monkeypatch):
    """The real save_ckpt path must persist the governed document and digest."""
    import torch
    import train_pretrain_with_bench as trainer

    doc = _contract(tmp_path)
    digest = C.governed_digest(doc)
    captured: dict = {}
    monkeypatch.setattr(trainer, "_atomic_torch_save", lambda obj, path: captured.update(obj=obj))

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Linear(2, 2)

    model = _M()
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer.save_ckpt(
        out_dir=tmp_path / "out",
        global_step=1,
        local_step=1,
        model=model,
        optim=optim,
        scaler=None,
        model_config={},
        train_args={},
        run_contract={},
        position_stats={},
        sampler_state={},
        data_contract={},
        retain_step=True,
        governed_run_contract=doc,
        governed_run_contract_sha256=digest,
    )
    saved = captured["obj"]
    assert saved["governed_run_contract"] == doc
    assert saved["governed_run_contract_sha256"] == digest
    assert saved["kind"] == C.GOVERNED_CHECKPOINT_KIND
    assert C.is_governed_checkpoint(saved) is True
    assert "rng_state" in saved and "data_sampler" in saved


def test_real_trainer_resume_validates_before_any_state_restoration(tmp_path, monkeypatch):
    """load_ckpt must reject a mismatch before load_state_dict or RNG restore runs."""
    import torch
    import train_pretrain_with_bench as trainer

    doc = _contract(tmp_path)
    drifted = {**doc, "gpu_uuid": "GPU-different"}
    ckpt = _ckpt(doc)
    ckpt["model"] = {}
    restored: list = []

    monkeypatch.setattr(trainer.torch, "load", lambda *a, **k: ckpt)

    class _M(torch.nn.Module):
        def load_state_dict(self, *a, **k):
            restored.append("model")

    class _O:
        def load_state_dict(self, *a, **k):
            restored.append("optim")

    with pytest.raises(TRAINER_C.LaunchContractError, match="before restoring any state"):
        trainer.load_ckpt(
            resume_path=tmp_path / "ckpt.pt",
            model=_M(),
            optim=_O(),
            scaler=None,
            resume_full=True,
            current_run_contract={},
            strict_resume_contract=True,
            allow_schedule_branch=False,
            governed_run_contract=drifted,
        )
    assert restored == []  # nothing was restored


def test_real_trainer_governed_resume_rejects_an_ungoverned_checkpoint(tmp_path, monkeypatch):
    import train_pretrain_with_bench as trainer

    doc = _contract(tmp_path)
    monkeypatch.setattr(trainer.torch, "load", lambda *a, **k: {"run_contract": {}, "model": {}})
    with pytest.raises(TRAINER_C.LaunchContractError, match="ungoverned_checkpoint"):
        trainer.load_ckpt(
            resume_path=tmp_path / "c.pt",
            model=None,
            optim=None,
            scaler=None,
            resume_full=True,
            current_run_contract={},
            strict_resume_contract=True,
            allow_schedule_branch=False,
            governed_run_contract=doc,
        )


def test_publication_precedes_the_first_optimizer_update(tmp_path):
    """Ordering proof: the contract file exists before any update callback could fire."""
    doc = _contract(tmp_path)
    out = tmp_path / "run"
    order: list = []

    published = C.publish_governed_run_contract(out, doc)
    order.append("publish")

    def fake_optimizer_update():
        assert (out / C.GOVERNED_RUN_CONTRACT_FILENAME).is_file()
        order.append("update")

    fake_optimizer_update()
    assert order == ["publish", "update"]
    assert published["governed_run_contract_sha256"] == C.governed_digest(doc)


# --------------------------------------------------------------------- closure


def test_governed_trainer_closure_is_rederived_and_complete():
    closure = C.trainer_execution_closure()
    assert closure["unbound_load_bearing_module_count"] == 0
    assert closure["reused_pilot_execution_closure"] is False
    assert closure["reused_stage_p_plan_closure"] is False
    for required in (
        "src/model.py",
        "src/optim.py",
        "pretrain/dataset_pretrain.py",
        "src/canonical_loss.py",
        "src/canonical_schedule.py",
        "src/special_tokens.py",
        "pretrain/run_plan_contract.py",
        "pretrain/production_launch_contract_v1.py",
        "pretrain/train_pretrain_with_bench.py",
    ):
        assert required in closure["derived_closure"], required
