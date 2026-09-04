"""Opt-in actual-N2 rehearsal that stops at the true pre-forward boundary.

The default test suite skips this diagnostic because it opens the 1 GiB terminal N2
checkpoint and constructs the production GPT and Muon on the governed CUDA device.  The
explicit rehearsal still performs no model forward, compile realization, backward,
optimizer step, or N3 publication.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import pytest

RUN_REHEARSAL = os.environ.get("PETITGPT_RUN_N3_ACTUAL_OBJECT_REHEARSAL") == "1"


@pytest.mark.skipif(
    not RUN_REHEARSAL,
    reason="set PETITGPT_RUN_N3_ACTUAL_OBJECT_REHEARSAL=1 for the governed CUDA rehearsal",
)
def test_actual_n2_reaches_true_precompile_boundary(tmp_path, monkeypatch):
    import torch

    successor_root = Path(__file__).resolve().parents[1]
    historical_root = Path("/workspace/petitgpt").resolve()
    assert Path.cwd().resolve() == historical_root
    assert (
        successor_root
        == Path("/workspace/petitgpt_stage_n_result_publication_recovery_v1").resolve()
    )

    if str(successor_root) not in sys.path:
        sys.path.insert(0, str(successor_root))
    from pretrain import stage_n_successor_head_compatibility_bridge_v1 as B

    launch = B.launch
    assert (
        Path(B.__file__).resolve()
        == (successor_root / "pretrain/stage_n_successor_head_compatibility_bridge_v1.py").resolve()
    )

    observed_successor = B.observe_successor_identity()
    manifest = B.build_semantic_comparison_manifest(
        historical_root,
        successor_root,
        successor_head=observed_successor["head"],
        successor_trainer_bundle_sha256=observed_successor["trainer_execution_bundle_sha256"],
    )
    assert manifest["semantic_isolation_pass"] is True
    assert manifest["core_training_semantics_files_changed"] == []
    manifest_path = tmp_path / "SEMANTIC_COMPARISON_ACTUAL_REHEARSAL.json"
    manifest_path.write_bytes(B.canonical_json_bytes(manifest))

    output_root = tmp_path / "n3-real-destination-must-remain-absent"
    candidate = B.n3_authorization_template(
        successor_head=observed_successor["head"],
        successor_trainer_bundle_sha256=observed_successor["trainer_execution_bundle_sha256"],
        bridge_tool_bundle_sha256=observed_successor["bridge_tool_bundle_sha256"],
        successor_runtime_fingerprint_sha256=observed_successor["runtime"][
            "runtime_fingerprint_sha256"
        ],
        semantic_comparison_manifest_path=manifest_path,
        semantic_comparison_manifest_sha256=B.file_sha256(manifest_path),
        output_root=output_root,
    )
    candidate.update(
        authorization_status="AUTHORIZED",
        authorizes_bridge_execution=True,
        authorized_by="R3 bounded actual-object rehearsal",
        authorized_at="2026-09-04T00:00:00Z",
    )
    authorization_path = tmp_path / "SCRATCH_AUTHORIZED_ACTUAL_REHEARSAL.json"
    authorization_path.write_bytes(B.canonical_json_bytes(candidate))
    manifest_artifact_sha256 = B.file_sha256(manifest_path)
    scratch_authorization_sha256 = B.file_sha256(authorization_path)

    source_path = Path(B.HISTORICAL_ARTIFACTS["n2_terminal_checkpoint"]["path"])
    expected_source_sha256 = str(B.HISTORICAL_ARTIFACTS["n2_terminal_checkpoint"]["sha256"])
    source_sha256_before = B.file_sha256(source_path)
    assert source_sha256_before == expected_source_sha256
    assert not output_root.exists()

    model_module = B._load_exact_successor_module("_petitgpt_successor_src_model", "src/model.py")
    optimizer_module = B._load_exact_successor_module(
        "_petitgpt_successor_src_optim", "src/optim.py"
    )
    canonical_model = sys.modules["src.model"]
    canonical_optimizer = sys.modules["src.optim"]
    assert canonical_model is model_module
    assert canonical_optimizer is optimizer_module

    real_boundary_derivation = launch.derive_stage_n_completion_boundary

    def raise_reviewed_launch_error(_document):
        raise launch.LaunchContractError("R3 canonical handler probe")

    monkeypatch.setattr(
        launch,
        "derive_stage_n_completion_boundary",
        raise_reviewed_launch_error,
    )
    canonical_handler_result = B.validate_completion_boundary({})
    assert canonical_handler_result == ["completion_boundary_derivation_failed:LaunchContractError"]
    ForeignLaunchContractError = type("LaunchContractError", (RuntimeError,), {})

    def raise_foreign_launch_error(_document):
        raise ForeignLaunchContractError("R3 foreign handler probe")

    monkeypatch.setattr(
        launch,
        "derive_stage_n_completion_boundary",
        raise_foreign_launch_error,
    )
    with pytest.raises(ForeignLaunchContractError):
        B.validate_completion_boundary({})
    monkeypatch.setattr(
        launch,
        "derive_stage_n_completion_boundary",
        real_boundary_derivation,
    )
    reviewed_exception_caught = True
    unrelated_exception_caught_as_reviewed = False

    events: list[str] = []
    objects: dict[str, object] = {}
    records: dict[str, object] = {}
    forbidden = {
        "model_forward": 0,
        "compile_realization": 0,
        "causal_diagnostic": 0,
        "tensor_backward": 0,
        "autograd_backward": 0,
        "autograd_grad": 0,
        "muon_step": 0,
        "base_optimizer_step": 0,
        "checkpoint_save": 0,
        "json_publication": 0,
        "directory_publication": 0,
        "staging_directory": 0,
    }

    def trap(label):
        def fail(*_args, **_kwargs):
            forbidden[label] += 1
            raise AssertionError(f"forbidden actual-object rehearsal operation:{label}")

        return fail

    real_validate = B.validate_bridge_authorization

    def tracked_validate(*args, **kwargs):
        verdict = real_validate(*args, **kwargs)
        records["canonical_preflight"] = {
            "authorized": verdict.get("authorized"),
            "failures": list(verdict.get("failures") or []),
            "source_head": verdict.get("source_head"),
            "successor_head": verdict.get("successor_head"),
            "observed_historical_identity": verdict.get("observed_historical_identity"),
            "observed_successor_identity": verdict.get("observed_successor_identity"),
        }
        assert verdict["authorized"] is True
        assert verdict["failures"] == []
        events.append("historical_source_chain_and_preflight_validated")
        return verdict

    monkeypatch.setattr(B, "validate_bridge_authorization", tracked_validate)

    real_read_checkpoint = B._read_bound_checkpoint_snapshot

    def tracked_read_checkpoint(path, *, expected_sha256, label):
        snapshot, document = real_read_checkpoint(
            path,
            expected_sha256=expected_sha256,
            label=label,
        )
        if Path(path).resolve() == source_path.resolve():
            records["source_checkpoint"] = {
                "path": str(Path(path).resolve()),
                "expected_sha256": expected_sha256,
                "snapshot_sha256": hashlib.sha256(snapshot).hexdigest(),
                "label": label,
                "global_step": document.get("global_step"),
            }
            events.append("actual_n2_checkpoint_read_and_deserialized")
        return snapshot, document

    monkeypatch.setattr(B, "_read_bound_checkpoint_snapshot", tracked_read_checkpoint)

    real_config_from_checkpoint = model_module.gpt_config_from_checkpoint_dict

    def tracked_config_from_checkpoint(document):
        cfg = real_config_from_checkpoint(document)
        assert type(cfg) is model_module.GPTConfig
        records["config"] = {
            "type": f"{type(cfg).__module__}.{type(cfg).__qualname__}",
            "exact_canonical_identity": type(cfg) is canonical_model.GPTConfig,
        }
        events.append("real_successor_config_constructed")
        return cfg

    monkeypatch.setattr(
        model_module,
        "gpt_config_from_checkpoint_dict",
        tracked_config_from_checkpoint,
    )

    real_model_init = model_module.GPT.__init__

    def tracked_model_init(self, *args, **kwargs):
        real_model_init(self, *args, **kwargs)
        objects["model_constructed"] = self
        assert type(self) is model_module.GPT
        assert type(self.cfg) is model_module.GPTConfig
        events.append("real_successor_model_constructed")

    monkeypatch.setattr(model_module.GPT, "__init__", tracked_model_init)

    real_model_load_state = model_module.GPT.load_state_dict

    def tracked_model_load_state(self, state_dict, *args, **kwargs):
        result = real_model_load_state(self, state_dict, *args, **kwargs)
        records["model_state_load"] = {
            "strict": kwargs.get("strict") is True,
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
        }
        events.append("actual_model_state_loaded_strictly")
        return result

    monkeypatch.setattr(model_module.GPT, "load_state_dict", tracked_model_load_state)

    real_build_optimizer = optimizer_module.build_optimizer

    def tracked_build_optimizer(*args, **kwargs):
        optimizer = real_build_optimizer(*args, **kwargs)
        objects["optimizer"] = optimizer
        assert type(optimizer) is optimizer_module.Muon
        events.append("real_successor_optimizer_constructed")
        return optimizer

    monkeypatch.setattr(optimizer_module, "build_optimizer", tracked_build_optimizer)

    real_optimizer_load_state = torch.optim.Optimizer.load_state_dict

    def tracked_optimizer_load_state(self, state_dict):
        result = real_optimizer_load_state(self, state_dict)
        if type(self) is optimizer_module.Muon:
            records["optimizer_state_load"] = {
                "actual_type": f"{type(self).__module__}.{type(self).__qualname__}",
                "returned_none": result is None,
            }
            events.append("actual_optimizer_state_loaded")
        return result

    monkeypatch.setattr(
        torch.optim.Optimizer,
        "load_state_dict",
        tracked_optimizer_load_state,
    )

    real_verify_optimizer = launch.verify_realized_optimizer

    def tracked_verify_optimizer(optimizer, model=None):
        verdict = real_verify_optimizer(optimizer, model)
        records["realized_muon_verdict"] = copy.deepcopy(verdict)
        assert verdict["matches_governed_realization"] is True
        assert verdict["failures"] == []
        events.append("actual_realized_muon_verification_passed")
        return verdict

    monkeypatch.setattr(launch, "verify_realized_optimizer", tracked_verify_optimizer)

    real_gate_b = launch.gate_b_post_construction

    def tracked_gate_b(model, optimizer, **kwargs):
        verdict = real_gate_b(model, optimizer, **kwargs)
        records["gate_b"] = copy.deepcopy(verdict)
        assert verdict["passed"] is True
        events.append("actual_gate_b_passed")
        return verdict

    monkeypatch.setattr(launch, "gate_b_post_construction", tracked_gate_b)

    real_restore_rng = B._restore_canonical_rng_state

    def tracked_restore_rng(state):
        result = real_restore_rng(state)
        events.append("actual_rng_state_restored")
        return result

    monkeypatch.setattr(B, "_restore_canonical_rng_state", tracked_restore_rng)

    real_live_state_proof = B._live_restored_state_proof

    def tracked_live_state_proof(*args, **kwargs):
        proof = real_live_state_proof(*args, **kwargs)
        records["live_state_proof"] = copy.deepcopy(proof)
        assert proof["model_tensors_compared"] == B.EXPECTED_MODEL_TENSOR_COUNT
        events.append("actual_precompile_live_state_proof_passed")
        return proof

    monkeypatch.setattr(B, "_live_restored_state_proof", tracked_live_state_proof)

    real_compile_stance = launch.enforce_compile_fail_closed_stance

    def tracked_compile_stance():
        stance = real_compile_stance()
        records["fail_closed_compile_stance"] = copy.deepcopy(stance)
        events.append("compile_fail_closed_stance_checked")
        return stance

    monkeypatch.setattr(launch, "enforce_compile_fail_closed_stance", tracked_compile_stance)

    cache_paths: dict[str, object] = {}
    cache_cleanup_roots: list[Path] = []
    real_isolated_cache = launch.isolated_inductor_cache

    def tracked_isolated_cache(run_token):
        cache_cleanup_roots.append(
            (Path(tempfile.gettempdir()) / f"petitgpt_governed_inductor_{run_token}").resolve()
        )
        cache = real_isolated_cache(run_token)
        cache_paths.update(cache)
        records["isolated_empty_compile_cache"] = copy.deepcopy(cache)
        events.append("isolated_empty_compile_cache_prepared")
        return cache

    monkeypatch.setattr(launch, "isolated_inductor_cache", tracked_isolated_cache)

    real_reset_dynamo_counters = launch.reset_dynamo_counters

    def tracked_reset_dynamo_counters():
        result = real_reset_dynamo_counters()
        events.append("dynamo_counters_reset_before_compile_entry")
        return result

    monkeypatch.setattr(launch, "reset_dynamo_counters", tracked_reset_dynamo_counters)

    class _PreCompileBoundary(BaseException):
        pass

    real_torch_compile = torch.compile
    compile_entry_sentinel_calls = 0

    def stop_at_true_boundary(model, *_args, **_kwargs):
        nonlocal compile_entry_sentinel_calls
        compile_entry_sentinel_calls += 1
        objects["model_at_boundary"] = model
        records["model_at_boundary"] = {
            "type": f"{type(model).__module__}.{type(model).__qualname__}",
            "config_type": f"{type(model.cfg).__module__}.{type(model.cfg).__qualname__}",
            "model_exact_canonical_identity": type(model) is canonical_model.GPT,
            "config_exact_canonical_identity": type(model.cfg) is canonical_model.GPTConfig,
        }
        events.append("true_pre_forward_torch_compile_entry_sentinel_reached")
        raise _PreCompileBoundary("intentional true pre-forward rehearsal stop")

    assert callable(real_torch_compile)
    monkeypatch.setattr(torch, "compile", stop_at_true_boundary)
    monkeypatch.setattr(model_module.GPT, "forward", trap("model_forward"))
    monkeypatch.setattr(
        launch,
        "realize_compile_production_shape",
        trap("compile_realization"),
    )
    monkeypatch.setattr(B, "_causal_diagnostic", trap("causal_diagnostic"))
    monkeypatch.setattr(torch.Tensor, "backward", trap("tensor_backward"))
    monkeypatch.setattr(torch.autograd, "backward", trap("autograd_backward"))
    monkeypatch.setattr(torch.autograd, "grad", trap("autograd_grad"))
    monkeypatch.setattr(optimizer_module.Muon, "step", trap("muon_step"))
    monkeypatch.setattr(torch.optim.Optimizer, "step", trap("base_optimizer_step"))
    monkeypatch.setattr(B, "_default_checkpoint_saver", trap("checkpoint_save"))
    monkeypatch.setattr(B, "_atomic_publish_json", trap("json_publication"))
    monkeypatch.setattr(B, "_rename_directory_noreplace", trap("directory_publication"))
    monkeypatch.setattr(tempfile, "mkdtemp", trap("staging_directory"))

    missing = launch._dig({}, ("unavailable",))
    assert missing is launch._MISSING
    assert type(missing) is launch._Missing
    observed_forward = launch.ObservedForward(lambda: None)
    assert type(observed_forward) is launch.ObservedForward
    assert observed_forward.invocations == 0
    assert reviewed_exception_caught
    assert not unrelated_exception_caught_as_reviewed

    old_cache_environment = {
        name: os.environ.get(name) for name in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
    }
    reached_boundary = False
    try:
        B.execute_authorized_bridge(
            authorization_path=authorization_path,
            semantic_comparison_manifest_path=manifest_path,
        )
    except _PreCompileBoundary:
        reached_boundary = True
    finally:
        for name, previous in old_cache_environment.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous
        for cache_root in cache_cleanup_roots:
            temp_root = Path(tempfile.gettempdir()).resolve()
            assert cache_root.parent == temp_root
            assert cache_root.name.startswith("petitgpt_governed_inductor_")
            if cache_root.exists():
                shutil.rmtree(cache_root)
        authorization_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)

    assert reached_boundary
    assert compile_entry_sentinel_calls == 1
    assert all(count == 0 for count in forbidden.values()), forbidden
    assert not output_root.exists()
    assert source_path.is_file()
    source_sha256_after = B.file_sha256(source_path)
    assert source_sha256_after == source_sha256_before == expected_source_sha256

    model = objects["model_at_boundary"]
    optimizer = objects["optimizer"]
    assert model is objects["model_constructed"]
    assert type(model) is canonical_model.GPT is model_module.GPT
    assert type(model.cfg) is canonical_model.GPTConfig is model_module.GPTConfig
    assert type(optimizer) is canonical_optimizer.Muon is optimizer_module.Muon
    assert sys.modules["_petitgpt_successor_src_model"] is canonical_model
    assert sys.modules["_petitgpt_successor_src_optim"] is canonical_optimizer
    assert sys.modules["_petitgpt_successor_production_launch_contract_v1"] is launch

    import pretrain
    import src

    assert src.model is canonical_model
    assert src.optim is canonical_optimizer
    assert pretrain.production_launch_contract_v1 is launch

    required_order = [
        "historical_source_chain_and_preflight_validated",
        "actual_n2_checkpoint_read_and_deserialized",
        "real_successor_config_constructed",
        "real_successor_model_constructed",
        "actual_model_state_loaded_strictly",
        "real_successor_optimizer_constructed",
        "actual_optimizer_state_loaded",
        "actual_realized_muon_verification_passed",
        "actual_gate_b_passed",
        "actual_rng_state_restored",
        "actual_precompile_live_state_proof_passed",
        "compile_fail_closed_stance_checked",
        "isolated_empty_compile_cache_prepared",
        "dynamo_counters_reset_before_compile_entry",
        "true_pre_forward_torch_compile_entry_sentinel_reached",
    ]
    assert events == required_order
    assert records["model_state_load"] == {
        "strict": True,
        "missing_keys": [],
        "unexpected_keys": [],
    }
    assert records["optimizer_state_load"]["returned_none"] is True
    assert records["gate_b"]["passed"] is True
    assert records["realized_muon_verdict"]["failures"] == []
    assert records["live_state_proof"] == {
        "model_parameters_bitwise_identical": True,
        "optimizer_state_equivalent": True,
        "scaler_state_equivalent": True,
        "rng_state_preserved": True,
        "all_parameter_grads_absent": True,
        "model_tensors_compared": B.EXPECTED_MODEL_TENSOR_COUNT,
    }

    def module_record(module, canonical_name, private_name):
        parent_name, _, child_name = canonical_name.rpartition(".")
        parent = sys.modules[parent_name]
        source = Path(module.__file__).resolve()
        return {
            "module_id": id(module),
            "canonical_module_id": id(sys.modules[canonical_name]),
            "private_module_id": id(sys.modules[private_name]),
            "parent_attribute_module_id": id(parent.__dict__[child_name]),
            "all_bindings_same_object": (
                module
                is sys.modules[canonical_name]
                is sys.modules[private_name]
                is parent.__dict__[child_name]
            ),
            "path": str(source),
            "sha256": B.file_sha256(source),
        }

    records.update({
        "schema_version": "petitgpt-n3-r3-actual-object-precompile-rehearsal-v1",
        "events": events,
        "required_event_order": required_order,
        "true_pre_forward_sentinel_reached": reached_boundary,
        "torch_compile_entry_sentinel_calls": compile_entry_sentinel_calls,
        "actual_torch_compile_implementation_called": False,
        "forbidden_operation_counts": forbidden,
        "source_checkpoint_sha256_before": source_sha256_before,
        "source_checkpoint_sha256_after": source_sha256_after,
        "destination_absent_before_and_after": not output_root.exists(),
        "reviewed_exception_caught": reviewed_exception_caught,
        "unrelated_exception_caught_as_reviewed": unrelated_exception_caught_as_reviewed,
        "missing_type_exact": type(missing) is launch._Missing,
        "observed_forward_type_exact": type(observed_forward) is launch.ObservedForward,
        "observed_forward_invocations": observed_forward.invocations,
        "model_module": module_record(
            model_module,
            "src.model",
            "_petitgpt_successor_src_model",
        ),
        "optimizer_module": module_record(
            optimizer_module,
            "src.optim",
            "_petitgpt_successor_src_optim",
        ),
        "launch_module": module_record(
            launch,
            "pretrain.production_launch_contract_v1",
            "_petitgpt_successor_production_launch_contract_v1",
        ),
        "model_type_object_id": id(type(model)),
        "canonical_gpt_object_id": id(canonical_model.GPT),
        "config_type_object_id": id(type(model.cfg)),
        "canonical_gpt_config_object_id": id(canonical_model.GPTConfig),
        "optimizer_type_object_id": id(type(optimizer)),
        "canonical_muon_object_id": id(canonical_optimizer.Muon),
        "successor_head": observed_successor["head"],
        "successor_trainer_bundle_sha256": observed_successor["trainer_execution_bundle_sha256"],
        "bridge_tool_bundle_sha256": observed_successor["bridge_tool_bundle_sha256"],
        "semantic_comparison_manifest_sha256": manifest_artifact_sha256,
        "scratch_authorization_sha256": scratch_authorization_sha256,
        "scratch_authorization_disposition": "deleted before rehearsal test completion",
    })
    assert not authorization_path.exists()
    assert not manifest_path.exists()
    print(
        "PETITGPT_R3_ACTUAL_OBJECT_REHEARSAL="
        + json.dumps(records, sort_keys=True, separators=(",", ":"))
    )

    objects.clear()
    del model, optimizer
    torch.cuda.empty_cache()
