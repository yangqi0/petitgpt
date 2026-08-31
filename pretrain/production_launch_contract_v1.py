"""P-PRODUCTION-LAUNCH-CONTRACT-V1: the governed Stage-N/Stage-O launch contract.

The accepted exact Stage-P run plan fixes *data identity and schedule geometry*. It does
not, and by design cannot, fix the learning rate, the optimizer, precision, compile mode,
seeds, evaluation policy or checkpoint policy: those are launch-time bindings. This module
is the adjacent owner-frozen contract that completes the plan without touching its bytes.

Nothing here authorizes training. Execution additionally requires an external
``authorization_status="AUTHORIZED"`` manifest scoped to exactly one stage, which this
repository cannot serialize. ``validate_authorization`` is the single gate that manifest
passes through, so no tracked code change is needed for that transition.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import MappingProxyType
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONTRACT_VERSION = "P-PRODUCTION-LAUNCH-CONTRACT-V1"
CONTRACT_SCHEMA = "petitgpt-production-launch-contract-v1"
AUTHORIZATION_SCHEMA = "petitgpt-production-launch-authorization-v1"
RUN_CONTRACT_SCHEMA = "petitgpt-production-governed-run-contract-v1"
EXECUTION_BUNDLE_SCHEMA = "petitgpt-production-trainer-execution-bundle-v1"


class LaunchContractError(RuntimeError):
    """A governed-launch binding failure. Never downgraded, never warned past."""


def require(condition: object, message: str) -> None:
    if not condition:
        raise LaunchContractError(message)


# --------------------------------------------------------------- accepted Stage-P inputs

EXACT_RUN_PLAN_RELPATH = (
    "runs/p_pilot_acceptance_and_exact_run_plan_v1_2026-08-31/plan/EXACT_RUN_PLAN.json"
)
EXACT_RUN_PLAN_SHA256 = "d673089447b4240ad7d5f7fd97dbf5d57567ad68bfffcc708a08f345fd25c117"
PLAN_GENERATION_HEAD = "4306f1db60b2c283f504404627e74f921c601800"
STAGE_P_PLAN_IMPLEMENTATION_BUNDLE_SHA256 = (
    "44d0982c2c0853c035d95528a043ca1cc48d60bdc9f9beb3e47a9d3b148f8f9f"
)
PILOT_OWNER_ACCEPTANCE_RELPATH = (
    "runs/p_pilot_acceptance_and_exact_run_plan_v1_2026-08-31/evidence/"
    "PILOT_RESULT_OWNER_ACCEPTANCE.json"
)
PILOT_OWNER_ACCEPTANCE_SHA256 = "ce5f0366f0f4f276b7ab802006930e3a01c605c023adab6317f0e17755079391"

ACCEPTED_STAGE_A_TRAIN_RELPATH = "runs/m_production_v1_2026-08-29/release/stage_a/train"
ACCEPTED_STAGE_B_TRAIN_RELPATH = "runs/m_production_v1_2026-08-29/release/stage_b/train"
ACCEPTED_VAL_RELPATH = "runs/g2_production_2026-08-21/release/val"

TOKENIZER_RELPATH = "runs/g_production_2026-08-21/release/tokenizer.json"
TOKENIZER_SHA256 = "d8f84df58928023edebd809e152b3b38a0dac53b9f887bd2455f427661e9b9ce"

CANONICAL_CWD = ROOT

# --------------------------------------------------------------- owner-frozen model

MODEL_CONTRACT = MappingProxyType({
    "n_layers": 30,
    "d_model": 576,
    "n_heads": 9,
    "n_kv_heads": 3,
    "d_ff": 1536,
    "seq_len": 2048,
    "vocab_size": 32000,
    "dropout": 0.0,
    "tie_embeddings": True,
})
MODEL_PARAMETER_COUNT = 124_635_456
MODEL_ARCHITECTURE_FEATURES = (
    "GQA",
    "RoPE",
    "RMSNorm",
    "SwiGLU",
    "SDPA",
    "tied_embeddings",
    "bf16",
)

# --------------------------------------------------------------- owner-frozen training

OPTIMIZER = "muon"
MUON_LR = 0.0
MUON_MOMENTUM = 0.95
PEAK_LR = 0.0006
MIN_LR_RATIO = 0.10
MICRO_BSZ = 8
GRAD_ACCUM = 16
EFFECTIVE_BATCH_TOKENS = 262144
SEQUENCES_PER_UPDATE = 128
COMPILE = True
PRECISION = "bf16"
WARMUP_STEPS = 500
DECAY_FRACTION = 0.10
LR_SCHEDULE = "wsd"
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
ADAMW_AUX_BETAS = (0.9, 0.95)
ADAMW_AUX_EPS = 1e-8
AUX_ADAMW_DECAY_WEIGHT_DECAY = 0.1
AUX_ADAMW_NO_DECAY_WEIGHT_DECAY = 0.0
MUON_NESTEROV = True
MUON_NS_STEPS = 5
MUON_LR_RATIO = 1.0
RMS_MATCHING_CONSTANT = 0.2
NEWTON_SCHULZ_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
OPTIMIZER_GROUP_ROLES = ("muon_matrices", "aux_adamw_decay", "aux_adamw_no_decay")
ROLE_WEIGHT_DECAY = MappingProxyType({
    "muon_matrices": WEIGHT_DECAY,
    "aux_adamw_decay": AUX_ADAMW_DECAY_WEIGHT_DECAY,
    "aux_adamw_no_decay": AUX_ADAMW_NO_DECAY_WEIGHT_DECAY,
})

STAGE_ORDER = ("stage_a", "stage_b")
OPTIMIZER_RESET_AT_A_B = False
SCHEDULER_RESET_AT_A_B = False

# --------------------------------------------------------------- owner-frozen seeds

MODEL_INIT_SEED = 20260831
STAGE_A_SAMPLER_SEED = 20260832
STAGE_B_SAMPLER_SEED = 20260833
VALIDATION_SEED = 20260834

SEED_TUPLE = MappingProxyType({
    "model_init_seed": MODEL_INIT_SEED,
    "stage_a_sampler_seed": STAGE_A_SAMPLER_SEED,
    "stage_b_sampler_seed": STAGE_B_SAMPLER_SEED,
    "validation_seed": VALIDATION_SEED,
})
PILOT_SEEDS_ARE_NOT_PRODUCTION_SEEDS = True
_PILOT_SEEDS = (20260829, 20260830)


def stage_sampler_seed(stage: str) -> int:
    """The one sampler seed a stage may use. There is no shared mutable seed."""
    require(stage in STAGE_ORDER, f"unknown stage {stage!r}; expected one of {STAGE_ORDER}")
    return STAGE_A_SAMPLER_SEED if stage == "stage_a" else STAGE_B_SAMPLER_SEED


def worker_rng_seed(base_seed: int, *, purpose: str) -> int:
    """Deterministically derive a loader/worker RNG seed from a frozen seed.

    The trainer already offsets DataLoader generators by +17 (train/val) and +19
    (per-source val); this reproduces that derivation so it is recorded rather than
    implicit.
    """
    offsets = {"train_loader": 17, "val_loader": 17, "val_by_source_loader": 19}
    require(purpose in offsets, f"unknown worker RNG purpose {purpose!r}")
    return int(base_seed) + offsets[purpose]


# --------------------------------------------------------------- owner-frozen evaluation

EVALUATION_MILESTONES = (500, 3815, 11445, 22889, 38146, 38147, 43868, 43870, 44631, 49590)
PERIODIC_EVAL_EVERY = 0  # 0 == DISABLED
BENCHMARK_EVAL_EVERY = 0  # 0 == DISABLED
VALIDATION_FULL_SET = True
VALIDATION_SAMPLES = 0  # 0 == the complete validation release
VALIDATION_SAMPLES_PER_SOURCE = 0
VALIDATION_SHUFFLE = False
VALIDATION_LOSS_REDUCTION = "canonical_global_token_mean"
VALIDATION_EOS_WEIGHT = 1.0

EVALUATION_POLICY = MappingProxyType({
    "mode": "explicit_milestones_only",
    "evaluation_steps": tuple(EVALUATION_MILESTONES),
    "periodic_eval_every": PERIODIC_EVAL_EVERY,
    "periodic_eval_status": "DISABLED",
    "benchmark_eval_every": BENCHMARK_EVAL_EVERY,
    "benchmark_eval_status": "DISABLED",
    "validation_release": ACCEPTED_VAL_RELPATH,
    "validation_seed": VALIDATION_SEED,
    "validation_full_set": VALIDATION_FULL_SET,
    "validation_samples": VALIDATION_SAMPLES,
    "validation_samples_per_source": VALIDATION_SAMPLES_PER_SOURCE,
    "validation_shuffle": VALIDATION_SHUFFLE,
    "validation_order": "deterministic_no_shuffle",
    "loss_reduction": VALIDATION_LOSS_REDUCTION,
    "eos_weight": VALIDATION_EOS_WEIGHT,
    "random_validation_subset": False,
})

# --------------------------------------------------------------- owner-frozen checkpoints

CHECKPOINT_MILESTONES = (3815, 11445, 22889, 38146, 38147, 43868, 43870, 44631, 49590)
PERIODIC_SAVE_EVERY = 0  # 0 == DISABLED
EXTRA_CHECKPOINTS_FORBIDDEN = True

CHECKPOINT_POLICY = MappingProxyType({
    "mode": "exact_plan_milestones_only",
    "save_steps": tuple(CHECKPOINT_MILESTONES),
    "periodic_save_every": PERIODIC_SAVE_EVERY,
    "periodic_save_status": "DISABLED",
    "extra_checkpoints": "FORBIDDEN",
    "cli_serialization": "repeated_--save_steps_flags_from_numeric_plan_list",
})


def save_steps_cli_flags(steps: Sequence[int] | None = None) -> list[str]:
    """Mechanically derive the governed repeated-flag serialization.

    ``load_run_plan_binding`` applies ``int()`` to each appended ``--save_steps`` element,
    so the governed launch never passes the planner's comma-joined string as one argument.
    """
    values = tuple(CHECKPOINT_MILESTONES if steps is None else steps)
    flags: list[str] = []
    for step in values:
        flags.extend(("--save_steps", str(int(step))))
    return flags


def eval_steps_cli_flags(steps: Sequence[int] | None = None) -> list[str]:
    values = tuple(EVALUATION_MILESTONES if steps is None else steps)
    flags: list[str] = []
    for step in values:
        flags.extend(("--eval_steps", str(int(step))))
    return flags


# --------------------------------------------------------------- exact-plan geometry

STAGE_A_START_STEP = 0
STAGE_A_STOP_STEP = 38146
STAGE_B_START_STEP = 38146
STAGE_B_GLOBAL_STOP_STEP = 49590
SCHEDULE_TOTAL_STEPS = 49590
DECAY_START_STEP = 44631
DECAY_END_STEP = 49590

STAGE_BOUNDARIES = MappingProxyType({
    "stage_a": {"start_step": STAGE_A_START_STEP, "stop_step": STAGE_A_STOP_STEP},
    "stage_b": {"start_step": STAGE_B_START_STEP, "stop_step": STAGE_B_GLOBAL_STOP_STEP},
})

# --------------------------------------------------------------- runtime binding

RUNTIME_BINDING_REQUIRED_FIELDS = (
    "gpu_uuid",
    "gpu_pci_bus_id",
    "gpu_name",
    "visible_cuda_device_count",
    "total_vram_bytes",
    "compute_capability",
    "driver_version",
    "cuda_runtime_version",
    "torch_version",
    "python_version",
    "numpy_version",
    "trainer_head",
    "trainer_execution_bundle_sha256",
    "canonical_cwd",
)
GPU_PRODUCT_AGNOSTIC_UNTIL_AUTHORIZATION = True
STAGE_N_TO_STAGE_O_CONTINUITY = (
    "Stage N must run on the exact runtime intended for Stage O. If any bound runtime "
    "field changes after Stage N, the Stage-N authorization and result become insufficient "
    "for Stage O and Stage N must be rerun. There is no materiality exception."
)
MATERIALITY_EXCEPTION_IMPLEMENTED = False

# --------------------------------------------------------------- parser classification

OWNER_FROZEN = "owner_frozen"
EXACT_PLAN_DERIVED = "exact_plan_derived"
LAUNCH_AUTHORIZATION_BOUND = "launch_authorization_bound"
RUNTIME_OBSERVED_AND_BOUND = "runtime_observed_and_bound"
DIAGNOSTIC_ONLY = "diagnostic_only_with_explicit_allowed_value"
FORBIDDEN_OR_UNSET = "forbidden_or_unset"

FIELD_CLASSES = (
    OWNER_FROZEN,
    EXACT_PLAN_DERIVED,
    LAUNCH_AUTHORIZATION_BOUND,
    RUNTIME_OBSERVED_AND_BOUND,
    DIAGNOSTIC_ONLY,
    FORBIDDEN_OR_UNSET,
)

# Every mutable trainer parser dest, classified exactly once. `value` is the governed
# required value where one exists; None means "bound elsewhere" (plan/authorization/runtime).
PARSER_FIELD_CLASSIFICATION: Mapping[str, Mapping[str, Any]] = MappingProxyType({
    # ---- data / output paths: bound by the authorization + exact plan ----
    "train_dir": {"class": EXACT_PLAN_DERIVED, "value": None, "affects": ("data order",)},
    "val_dir": {"class": OWNER_FROZEN, "value": ACCEPTED_VAL_RELPATH, "affects": ("evaluation",)},
    "out_dir": {"class": LAUNCH_AUTHORIZATION_BOUND, "value": None, "affects": ("checkpointing",)},
    "samples_dir": {
        "class": LAUNCH_AUTHORIZATION_BOUND,
        "value": None,
        "affects": ("checkpointing",),
    },
    "tokenizer_path": {
        "class": OWNER_FROZEN,
        "value": TOKENIZER_RELPATH,
        "affects": ("model bytes", "data order", "evaluation"),
    },
    # ---- model contract ----
    "vocab_size": {"class": OWNER_FROZEN, "value": 32000, "affects": ("model bytes",)},
    "seq_len": {"class": OWNER_FROZEN, "value": 2048, "affects": ("model bytes", "data order")},
    "layers": {"class": OWNER_FROZEN, "value": 30, "affects": ("model bytes",)},
    "d_model": {"class": OWNER_FROZEN, "value": 576, "affects": ("model bytes",)},
    "n_heads": {"class": OWNER_FROZEN, "value": 9, "affects": ("model bytes",)},
    "n_kv_heads": {"class": OWNER_FROZEN, "value": 3, "affects": ("model bytes",)},
    "d_ff": {"class": OWNER_FROZEN, "value": 1536, "affects": ("model bytes",)},
    "dropout": {
        "class": OWNER_FROZEN,
        "value": 0.0,
        "affects": ("model bytes", "gradients", "randomness"),
    },
    "bos_id": {"class": OWNER_FROZEN, "value": 2, "affects": ("gradients",)},
    "eos_id": {"class": OWNER_FROZEN, "value": 3, "affects": ("gradients",)},
    # ---- loss masking ----
    "no_mask_bos_in_loss": {"class": FORBIDDEN_OR_UNSET, "value": False, "affects": ("gradients",)},
    "mask_last_label_in_loss": {
        "class": FORBIDDEN_OR_UNSET,
        "value": False,
        "affects": ("gradients",),
    },
    "no_mask_last_label_in_loss": {
        "class": FORBIDDEN_OR_UNSET,
        "value": False,
        "affects": ("gradients",),
    },
    "eos_weight": {"class": OWNER_FROZEN, "value": 1.0, "affects": ("gradients",)},
    "eos_weight_warmup_steps": {"class": OWNER_FROZEN, "value": 0, "affects": ("gradients",)},
    # ---- precision / geometry ----
    "precision": {"class": OWNER_FROZEN, "value": PRECISION, "affects": ("precision", "gradients")},
    "micro_bsz": {
        "class": OWNER_FROZEN,
        "value": MICRO_BSZ,
        "affects": ("gradients", "data order"),
    },
    "grad_accum": {
        "class": OWNER_FROZEN,
        "value": GRAD_ACCUM,
        "affects": ("gradients", "data order"),
    },
    # ---- optimizer / schedule ----
    "lr": {
        "class": OWNER_FROZEN,
        "value": PEAK_LR,
        "affects": ("optimizer", "learning-rate schedule"),
    },
    "weight_decay": {"class": OWNER_FROZEN, "value": WEIGHT_DECAY, "affects": ("optimizer",)},
    "optimizer": {"class": OWNER_FROZEN, "value": OPTIMIZER, "affects": ("optimizer",)},
    "muon_lr": {"class": OWNER_FROZEN, "value": MUON_LR, "affects": ("optimizer",)},
    "muon_momentum": {"class": OWNER_FROZEN, "value": MUON_MOMENTUM, "affects": ("optimizer",)},
    "warmup_steps": {
        "class": OWNER_FROZEN,
        "value": WARMUP_STEPS,
        "affects": ("learning-rate schedule",),
    },
    "lr_schedule": {
        "class": OWNER_FROZEN,
        "value": LR_SCHEDULE,
        "affects": ("learning-rate schedule",),
    },
    "schedule_total_steps": {
        "class": EXACT_PLAN_DERIVED,
        "value": SCHEDULE_TOTAL_STEPS,
        "affects": ("learning-rate schedule",),
    },
    "decay_start_step": {
        "class": EXACT_PLAN_DERIVED,
        "value": DECAY_START_STEP,
        "affects": ("learning-rate schedule",),
    },
    "decay_end_step": {
        "class": EXACT_PLAN_DERIVED,
        "value": DECAY_END_STEP,
        "affects": ("learning-rate schedule",),
    },
    "min_lr_ratio": {
        "class": OWNER_FROZEN,
        "value": MIN_LR_RATIO,
        "affects": ("learning-rate schedule",),
    },
    "max_steps": {"class": EXACT_PLAN_DERIVED, "value": None, "affects": ("stop boundary",)},
    "data_stage_start_step": {
        "class": EXACT_PLAN_DERIVED,
        "value": None,
        "affects": ("data order", "stop boundary"),
    },
    "grad_clip": {"class": OWNER_FROZEN, "value": GRAD_CLIP, "affects": ("gradients",)},
    # ---- run plan binding ----
    "run_plan_json": {
        "class": EXACT_PLAN_DERIVED,
        "value": EXACT_RUN_PLAN_RELPATH,
        "affects": ("data order", "stop boundary"),
    },
    "run_plan_stage": {
        "class": LAUNCH_AUTHORIZATION_BOUND,
        "value": None,
        "affects": ("data order", "stop boundary"),
    },
    # ---- randomness ----
    "seed": {
        "class": OWNER_FROZEN,
        "value": MODEL_INIT_SEED,
        "affects": ("model bytes", "randomness"),
    },
    "sampler_seed": {
        "class": FORBIDDEN_OR_UNSET,
        "value": None,
        "affects": ("data order", "randomness"),
        "note": "legacy shared mutable seed; the governed path uses the "
        "per-stage seeds and never this field",
    },
    "stage_a_sampler_seed": {
        "class": OWNER_FROZEN,
        "value": STAGE_A_SAMPLER_SEED,
        "affects": ("data order", "randomness"),
    },
    "stage_b_sampler_seed": {
        "class": OWNER_FROZEN,
        "value": STAGE_B_SAMPLER_SEED,
        "affects": ("data order", "randomness"),
    },
    "val_seed": {
        "class": OWNER_FROZEN,
        "value": VALIDATION_SEED,
        "affects": ("evaluation", "randomness"),
    },
    # ---- validation controls ----
    "val_samples": {"class": OWNER_FROZEN, "value": VALIDATION_SAMPLES, "affects": ("evaluation",)},
    "val_samples_per_source": {
        "class": OWNER_FROZEN,
        "value": VALIDATION_SAMPLES_PER_SOURCE,
        "affects": ("evaluation",),
    },
    # ---- cadence ----
    "log_every": {"class": DIAGNOSTIC_ONLY, "value": 20, "affects": ()},
    "eval_every": {"class": OWNER_FROZEN, "value": PERIODIC_EVAL_EVERY, "affects": ("evaluation",)},
    "eval_steps": {
        "class": OWNER_FROZEN,
        "value": tuple(EVALUATION_MILESTONES),
        "affects": ("evaluation",),
    },
    "save_every": {
        "class": OWNER_FROZEN,
        "value": PERIODIC_SAVE_EVERY,
        "affects": ("checkpointing",),
    },
    "save_steps": {
        "class": EXACT_PLAN_DERIVED,
        "value": tuple(CHECKPOINT_MILESTONES),
        "affects": ("checkpointing",),
    },
    "debug_every": {"class": DIAGNOSTIC_ONLY, "value": 500, "affects": ()},
    # ---- benchmark eval: disabled ----
    "bench_eval_path": {"class": FORBIDDEN_OR_UNSET, "value": "", "affects": ("evaluation",)},
    "bench_eval_every": {
        "class": OWNER_FROZEN,
        "value": BENCHMARK_EVAL_EVERY,
        "affects": ("evaluation",),
    },
    "bench_eval_script": {
        "class": DIAGNOSTIC_ONLY,
        "value": "pretrain/eval_bench_v5.py",
        "affects": (),
    },
    "bench_eval_out_dir": {"class": FORBIDDEN_OR_UNSET, "value": "", "affects": ()},
    "bench_eval_max_seq_len": {"class": DIAGNOSTIC_ONLY, "value": 1024, "affects": ()},
    "bench_eval_max_new_tokens": {"class": DIAGNOSTIC_ONLY, "value": 192, "affects": ()},
    "bench_eval_min_new_tokens": {"class": DIAGNOSTIC_ONLY, "value": 1, "affects": ()},
    "bench_eval_ban_first_steps": {"class": DIAGNOSTIC_ONLY, "value": 4, "affects": ()},
    # ---- sampling: diagnostic text generation only, never gradients ----
    "add_bos_to_prompts": {"class": DIAGNOSTIC_ONLY, "value": True, "affects": ()},
    "sample_temperature": {"class": DIAGNOSTIC_ONLY, "value": 0.7, "affects": ()},
    "sample_top_p": {"class": DIAGNOSTIC_ONLY, "value": 0.9, "affects": ()},
    "sample_top_k": {"class": DIAGNOSTIC_ONLY, "value": 0, "affects": ()},
    "sample_max_new_tokens": {"class": DIAGNOSTIC_ONLY, "value": 256, "affects": ()},
    "sample_min_new_tokens": {"class": DIAGNOSTIC_ONLY, "value": 0, "affects": ()},
    # ---- runtime ----
    "num_workers": {
        "class": RUNTIME_OBSERVED_AND_BOUND,
        "value": None,
        "affects": ("data order", "randomness"),
    },
    "compile": {"class": OWNER_FROZEN, "value": COMPILE, "affects": ("compile behavior",)},
    # ---- resume ----
    "resume_path": {"class": LAUNCH_AUTHORIZATION_BOUND, "value": None, "affects": ("resume",)},
    "resume_full": {"class": LAUNCH_AUTHORIZATION_BOUND, "value": None, "affects": ("resume",)},
    "resume_step": {"class": LAUNCH_AUTHORIZATION_BOUND, "value": None, "affects": ("resume",)},
    "allow_weights_only_resume": {
        "class": FORBIDDEN_OR_UNSET,
        "value": False,
        "affects": ("resume",),
    },
    "allow_data_branch": {
        "class": FORBIDDEN_OR_UNSET,
        "value": False,
        "affects": ("resume", "data order"),
    },
    "allow_schedule_branch": {
        "class": FORBIDDEN_OR_UNSET,
        "value": False,
        "affects": ("resume", "learning-rate schedule"),
    },
    "strict_resume_contract": {"class": OWNER_FROZEN, "value": True, "affects": ("resume",)},
    # ---- governed-launch inputs ----
    "launch_contract_json": {
        "class": OWNER_FROZEN,
        "value": None,
        "affects": ("runtime identity",),
    },
    "stage_authorization_json": {
        "class": LAUNCH_AUTHORIZATION_BOUND,
        "value": None,
        "affects": ("runtime identity",),
    },
})

AFFECT_DOMAINS = (
    "model bytes",
    "optimizer",
    "learning-rate schedule",
    "gradients",
    "data order",
    "evaluation",
    "checkpointing",
    "resume",
    "precision",
    "compile behavior",
    "randomness",
    "runtime identity",
    "stop boundary",
)


def classify_parser_namespace(parser_dests: Sequence[str]) -> dict[str, Any]:
    """Every mutable parser field must be classified exactly once."""
    dests = [d for d in dict.fromkeys(parser_dests) if d != "help"]
    classified = set(PARSER_FIELD_CLASSIFICATION)
    unclassified = sorted(set(dests) - classified)
    unknown = sorted(classified - set(dests))
    by_class: dict[str, list[str]] = {c: [] for c in FIELD_CLASSES}
    for dest in sorted(set(dests) & classified):
        by_class[PARSER_FIELD_CLASSIFICATION[dest]["class"]].append(dest)
    return {
        "parser_field_count": len(dests),
        "classified_count": len(set(dests) & classified),
        "unclassified_fields": unclassified,
        "classified_but_absent_from_parser": unknown,
        "complete": not unclassified,
        "by_class": by_class,
    }


# --------------------------------------------------------------- realized Muon binding


def realized_muon_contract() -> dict[str, Any]:
    """Bind the COMPLETE realized Muon configuration from the frozen ``src/optim.py``."""
    import inspect

    from src.optim import ADAM_PARAM_NAME_KEYS, Muon, build_optimizer

    defaults = {k: v.default for k, v in inspect.signature(build_optimizer).parameters.items()}
    return {
        "optimizer": OPTIMIZER,
        "muon_lr": MUON_LR,
        "muon_lr_resolution": (
            "build_optimizer: muon_lr = float(muon_lr) if muon_lr and muon_lr > 0 else "
            "float(lr). With --muon_lr 0.0 the Muon matrix groups reuse the scheduled main "
            "--lr and lr_ratio is exactly 1.0."
        ),
        "muon_momentum": MUON_MOMENTUM,
        "nesterov": MUON_NESTEROV,
        "newton_schulz_steps": defaults.get("ns_steps", MUON_NS_STEPS),
        "newton_schulz_coefficients": list(NEWTON_SCHULZ_COEFFICIENTS),
        "rms_matching": {
            "formula": "adjusted_lr = lr * 0.2 * sqrt(max(fan_in, fan_out))",
            "constant": RMS_MATCHING_CONSTANT,
        },
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "aux_adamw_betas": list(ADAMW_AUX_BETAS),
        "aux_adamw_eps": ADAMW_AUX_EPS,
        "role_weight_decay": dict(ROLE_WEIGHT_DECAY),
        "group_roles": list(OPTIMIZER_GROUP_ROLES),
        "lr_ratio_required_on_every_group": MUON_LR_RATIO,
        "adam_param_name_keys": list(ADAM_PARAM_NAME_KEYS),
        "membership_rule": {
            "muon_matrices": "2D params not matched by ADAM_PARAM_NAME_KEYS",
            "aux_adamw_decay": f"params whose name contains any of {list(ADAM_PARAM_NAME_KEYS)}",
            "aux_adamw_no_decay": "params with ndim < 2 (norm gains, biases)",
        },
        "single_optimizer_instance": True,
        "muon_class": Muon.__name__,
    }


def verify_realized_optimizer(optimizer: Any, model: Any | None = None) -> dict[str, Any]:
    """Verify a constructed optimizer matches the governed realization exactly."""
    from src.optim import ADAM_PARAM_NAME_KEYS, Muon

    failures: list[str] = []
    groups = list(optimizer.param_groups)
    if not isinstance(optimizer, Muon):
        failures.append(f"optimizer must be the Muon instance, got {type(optimizer).__name__}")

    roles: list[str] = []
    for group in groups:
        if group.get("use_muon"):
            roles.append("muon_matrices")
        else:
            params = list(group.get("params") or [])
            roles.append(
                "aux_adamw_no_decay"
                if params and all(int(getattr(p, "ndim", 2)) < 2 for p in params)
                else "aux_adamw_decay"
            )
    if sorted(roles) != sorted(OPTIMIZER_GROUP_ROLES):
        failures.append(f"realized group roles {sorted(roles)} != {sorted(OPTIMIZER_GROUP_ROLES)}")

    for index, (role, group) in enumerate(zip(roles, groups, strict=True)):
        if "lr_ratio" not in group:
            failures.append(f"optimizer group {index} ({role}) is missing lr_ratio")
        elif float(group["lr_ratio"]) != MUON_LR_RATIO:
            failures.append(f"{role} lr_ratio must be {MUON_LR_RATIO}, got {group['lr_ratio']!r}")
        expected_wd = ROLE_WEIGHT_DECAY.get(role)
        if expected_wd is not None and float(group.get("weight_decay", -1)) != float(expected_wd):
            failures.append(
                f"{role} weight_decay must be {expected_wd}, got {group.get('weight_decay')!r}"
            )
        if group.get("use_muon"):
            if float(group.get("momentum", -1)) != MUON_MOMENTUM:
                failures.append(f"Muon momentum must be {MUON_MOMENTUM}")
            if group.get("nesterov") is not MUON_NESTEROV:
                failures.append("Muon group must use Nesterov momentum")
            if int(group.get("ns_steps", -1)) != MUON_NS_STEPS:
                failures.append(f"Newton-Schulz steps must be {MUON_NS_STEPS}")
        else:
            if tuple(group.get("betas", ())) != ADAMW_AUX_BETAS:
                failures.append(f"auxiliary AdamW betas must be {ADAMW_AUX_BETAS}")
            if float(group.get("eps", -1)) != ADAMW_AUX_EPS:
                failures.append(f"auxiliary AdamW eps must be {ADAMW_AUX_EPS}")

    membership: dict[str, list[str]] = {}
    if model is not None:
        by_id = {id(p): n for n, p in model.named_parameters() if p.requires_grad}
        expected: dict[str, list[str]] = {r: [] for r in OPTIMIZER_GROUP_ROLES}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2:
                expected["aux_adamw_no_decay"].append(name)
            elif any(k in name for k in ADAM_PARAM_NAME_KEYS):
                expected["aux_adamw_decay"].append(name)
            else:
                expected["muon_matrices"].append(name)
        seen: list[int] = []
        for role, group in zip(roles, groups, strict=True):
            names = [by_id[id(p)] for p in group["params"] if id(p) in by_id]
            membership[role] = sorted(names)
            seen.extend(id(p) for p in group["params"])
        if len(seen) != len(set(seen)):
            failures.append("a parameter appears in more than one optimizer group")
        if sorted(set(by_id) - set(seen)):
            failures.append("a trainable parameter is in no optimizer group")
        for role in OPTIMIZER_GROUP_ROLES:
            if membership.get(role, []) != sorted(expected[role]):
                failures.append(f"{role} membership differs from the frozen grouping rule")

    return {
        "group_roles": roles,
        "membership": membership,
        "failures": failures,
        "matches_governed_realization": not failures,
    }


# --------------------------------------------------------------- serialization


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_root() -> Path:
    return Path(ROOT).resolve()


def contract_document() -> dict[str, Any]:
    """The complete governed launch contract. Authorizes nothing on its own."""
    return {
        "schema_version": CONTRACT_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "authorization_status": "NOT_AUTHORIZED",
        "authorizes_training": False,
        "accepted_stage_p_inputs": {
            "exact_run_plan_relpath": EXACT_RUN_PLAN_RELPATH,
            "exact_run_plan_sha256": EXACT_RUN_PLAN_SHA256,
            "plan_generation_head": PLAN_GENERATION_HEAD,
            "stage_p_plan_implementation_bundle_sha256": STAGE_P_PLAN_IMPLEMENTATION_BUNDLE_SHA256,
            "pilot_owner_acceptance_relpath": PILOT_OWNER_ACCEPTANCE_RELPATH,
            "pilot_owner_acceptance_sha256": PILOT_OWNER_ACCEPTANCE_SHA256,
            "exact_plan_bytes_are_immutable": True,
            "exact_plan_is_not_a_training_authorization": True,
        },
        "data": {
            "stage_a_train_relpath": ACCEPTED_STAGE_A_TRAIN_RELPATH,
            "stage_b_train_relpath": ACCEPTED_STAGE_B_TRAIN_RELPATH,
            "validation_relpath": ACCEPTED_VAL_RELPATH,
            "tokenizer_relpath": TOKENIZER_RELPATH,
            "tokenizer_sha256": TOKENIZER_SHA256,
        },
        "model": {
            **dict(MODEL_CONTRACT),
            "parameter_count": MODEL_PARAMETER_COUNT,
            "architecture_features": list(MODEL_ARCHITECTURE_FEATURES),
        },
        "training": {
            "optimizer": OPTIMIZER,
            "muon_lr": MUON_LR,
            "muon_momentum": MUON_MOMENTUM,
            "peak_lr": PEAK_LR,
            "min_lr_ratio": MIN_LR_RATIO,
            "micro_bsz": MICRO_BSZ,
            "grad_accum": GRAD_ACCUM,
            "effective_batch_tokens": EFFECTIVE_BATCH_TOKENS,
            "sequences_per_update": SEQUENCES_PER_UPDATE,
            "compile": COMPILE,
            "precision": PRECISION,
            "warmup_steps": WARMUP_STEPS,
            "decay_fraction": DECAY_FRACTION,
            "lr_schedule": LR_SCHEDULE,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "stage_order": list(STAGE_ORDER),
            "optimizer_reset_at_a_b": OPTIMIZER_RESET_AT_A_B,
            "scheduler_reset_at_a_b": SCHEDULER_RESET_AT_A_B,
        },
        "optimizer_realization": realized_muon_contract(),
        "boundaries": {
            "stage_a_start_step": STAGE_A_START_STEP,
            "stage_a_stop_step": STAGE_A_STOP_STEP,
            "stage_b_start_step": STAGE_B_START_STEP,
            "stage_b_global_stop_step": STAGE_B_GLOBAL_STOP_STEP,
            "schedule_total_steps": SCHEDULE_TOTAL_STEPS,
            "decay_start_step": DECAY_START_STEP,
            "decay_end_step": DECAY_END_STEP,
        },
        "seeds": {
            **dict(SEED_TUPLE),
            "pilot_seeds_are_not_production_seeds": PILOT_SEEDS_ARE_NOT_PRODUCTION_SEEDS,
            "rejected_pilot_seeds": list(_PILOT_SEEDS),
            "worker_rng_derivation": {
                "train_loader": "stage sampler seed + 17",
                "val_loader": "validation seed + 17",
                "val_by_source_loader": "validation seed + 19",
            },
            "global_rng_governed_by": "model_init_seed (python random, numpy, torch cpu+cuda)",
        },
        "evaluation_policy": {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in EVALUATION_POLICY.items()
        },
        "checkpoint_policy": {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in CHECKPOINT_POLICY.items()
        },
        "canonical_cwd": CANONICAL_CWD,
        "runtime_binding": {
            "required_fields": list(RUNTIME_BINDING_REQUIRED_FIELDS),
            "gpu_product_agnostic_until_authorization": GPU_PRODUCT_AGNOSTIC_UNTIL_AUTHORIZATION,
            "stage_n_to_stage_o_continuity": STAGE_N_TO_STAGE_O_CONTINUITY,
            "materiality_exception_implemented": MATERIALITY_EXCEPTION_IMPLEMENTED,
        },
        "parser_field_classification": {
            dest: {
                "class": spec["class"],
                "value": (
                    list(spec["value"]) if isinstance(spec["value"], tuple) else spec["value"]
                ),
                "affects": list(spec.get("affects", ())),
                **({"note": spec["note"]} if "note" in spec else {}),
            }
            for dest, spec in sorted(PARSER_FIELD_CLASSIFICATION.items())
        },
        "field_classes": list(FIELD_CLASSES),
        "affect_domains": list(AFFECT_DOMAINS),
    }


def contract_sha256() -> str:
    return _sha256_bytes(canonical_json_bytes(contract_document()))


# --------------------------------------------------------------- authorization

ALLOWED_SCOPES = ("STAGE_N", "STAGE_O")
AUTHORIZATION_REQUIRED_FIELDS = (
    "schema_version",
    "authorization_status",
    "allowed_scope",
    "repository_branch",
    "trainer_head",
    "trainer_execution_bundle_sha256",
    "launch_contract_sha256",
    "exact_run_plan_sha256",
    "pilot_owner_acceptance_sha256",
    "allowed_output_root",
    "canonical_cwd",
    "training_runtime",
)


def authorization_template() -> dict[str, Any]:
    """Always NOT_AUTHORIZED. This repository cannot publish an AUTHORIZED manifest."""
    return {
        "schema_version": AUTHORIZATION_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "authorization_status": "NOT_AUTHORIZED",
        "allowed_scope": None,
        "allowed_scope_values": list(ALLOWED_SCOPES),
        "repository_branch": None,
        "trainer_head": None,
        "trainer_execution_bundle_sha256": None,
        "launch_contract_sha256": None,
        "exact_run_plan_sha256": EXACT_RUN_PLAN_SHA256,
        "pilot_owner_acceptance_sha256": PILOT_OWNER_ACCEPTANCE_SHA256,
        "allowed_output_root": None,
        "canonical_cwd": CANONICAL_CWD,
        "training_runtime": None,
        "authorized_by": None,
        "authorized_at": None,
        "note": (
            "One stage per authorization. Stage O is never authorized by a Stage-N "
            "authorization: it requires its own manifest bound to the same runtime Stage N "
            "actually ran on. If any bound runtime field changed after Stage N, Stage N must "
            "be rerun. No tracked code change is required for the AUTHORIZED transition: "
            "validate_authorization consumes this same schema."
        ),
    }


def check_training_runtime_binding(runtime: Mapping[str, Any] | None) -> list[str]:
    """Every runtime field an authorization must pin. GPU product is not policy."""
    if not isinstance(runtime, Mapping):
        return ["training_runtime_missing_or_malformed"]
    failures = [
        f"training_runtime_missing_field:{field}"
        for field in RUNTIME_BINDING_REQUIRED_FIELDS
        if field not in runtime
    ]
    for field in ("gpu_uuid", "gpu_pci_bus_id", "gpu_name"):
        value = runtime.get(field)
        if field in runtime and (not isinstance(value, str) or not value.strip()):
            failures.append(f"training_runtime_{field}_invalid")
    count = runtime.get("visible_cuda_device_count")
    if "visible_cuda_device_count" in runtime and (
        not isinstance(count, int) or isinstance(count, bool) or count != 1
    ):
        failures.append("visible_cuda_device_count_must_equal_1")
    if runtime.get("canonical_cwd") != CANONICAL_CWD:
        failures.append("training_runtime_canonical_cwd_mismatch")
    return failures


def validate_authorization(
    manifest: Mapping[str, Any] | None,
    *,
    requested_scope: str,
    observed: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an external stage authorization. The caller refuses on authorized=False."""
    require(requested_scope in ALLOWED_SCOPES, f"unknown scope {requested_scope!r}")
    if manifest is None:
        return {
            "authorized": False,
            "failures": ["authorization_missing"],
            "requested_scope": requested_scope,
        }
    failures: list[str] = []
    missing = [f for f in AUTHORIZATION_REQUIRED_FIELDS if f not in manifest]
    if missing:
        failures.append(f"missing_fields:{sorted(missing)}")
    if manifest.get("schema_version") != AUTHORIZATION_SCHEMA:
        failures.append("authorization_schema_mismatch")
    if manifest.get("contract_version") != CONTRACT_VERSION:
        failures.append("contract_version_mismatch")
    if manifest.get("authorization_status") != "AUTHORIZED":
        failures.append("authorization_status_not_authorized")

    scope = manifest.get("allowed_scope")
    if scope not in ALLOWED_SCOPES:
        failures.append("allowed_scope_invalid")
    elif scope != requested_scope:
        # A Stage-N authorization can never execute Stage O, and vice versa.
        failures.append(f"scope_mismatch:authorized={scope},requested={requested_scope}")

    for field, key in (
        ("repository_branch", "branch"),
        ("trainer_head", "head"),
        ("trainer_execution_bundle_sha256", "trainer_execution_bundle_sha256"),
        ("launch_contract_sha256", "launch_contract_sha256"),
        ("exact_run_plan_sha256", "exact_run_plan_sha256"),
        ("pilot_owner_acceptance_sha256", "pilot_owner_acceptance_sha256"),
        ("allowed_output_root", "output_root"),
        ("canonical_cwd", "canonical_cwd"),
    ):
        if key in observed and manifest.get(field) != observed.get(key):
            failures.append(f"mismatch:{field}")

    failures.extend(check_training_runtime_binding(manifest.get("training_runtime")))

    observed_runtime = observed.get("training_runtime")
    if isinstance(observed_runtime, Mapping) and isinstance(
        manifest.get("training_runtime"), Mapping
    ):
        bound = manifest["training_runtime"]
        for field in RUNTIME_BINDING_REQUIRED_FIELDS:
            if field in observed_runtime and bound.get(field) != observed_runtime.get(field):
                failures.append(f"runtime_mismatch:{field}")

    failures = list(dict.fromkeys(failures))
    return {
        "authorized": not failures,
        "failures": failures,
        "allowed_scope": scope,
        "requested_scope": requested_scope,
    }


def require_authorization(
    manifest: Mapping[str, Any] | None,
    *,
    requested_scope: str,
    observed: Mapping[str, Any],
) -> dict[str, Any]:
    verdict = validate_authorization(manifest, requested_scope=requested_scope, observed=observed)
    require(
        verdict["authorized"],
        f"governed launch refused under {CONTRACT_VERSION}: " + ", ".join(verdict["failures"]),
    )
    return verdict


# --------------------------------------------------------------- execution closure

TRAINER_CLOSURE_ROOTS = (
    "pretrain/train_pretrain_with_bench.py",
    "pretrain/production_launch_contract_v1.py",
    "pretrain/run_plan_contract.py",
    "pretrain/dataset_pretrain.py",
)

PILOT_EXECUTION_CLOSURE = (
    "pretrain/dataset_pretrain.py",
    "pretrain/pilot_contract_v2_3.py",
    "pretrain/pilot_runner_v2_3.py",
    "src/__init__.py",
    "src/canonical_loss.py",
    "src/canonical_schedule.py",
    "src/model.py",
    "src/optim.py",
    "src/special_tokens.py",
)
STAGE_P_PLAN_CLOSURE = (
    "pretrain/dataset_pretrain.py",
    "pretrain/plan_pretrain_run.py",
    "pretrain/run_plan_contract.py",
    "pretrain/stage_m_contract_v1.py",
    "pretrain/stage_m_input_v1.py",
    "pretrain/stage_m_output_v1.py",
    "pretrain/stage_p_native_provenance_v1.py",
    "src/__init__.py",
    "src/special_tokens.py",
)


def trainer_execution_closure(root: Path | None = None) -> dict[str, Any]:
    """Derive the governed trainer's complete load-bearing local closure by AST walk.

    Neither the pilot execution closure nor the Stage-P plan closure is reused: this walk
    starts from the trainer roots and re-derives everything reachable, including the model,
    optimizer, dataset, loss, schedule, sampler, evaluation, checkpoint/resume, run-plan
    loading, special tokens and launch-contract validation modules.
    """
    base = Path(root) if root is not None else repo_root()

    def resolve(mod: str) -> list[str]:
        out, parts = [], mod.split(".")
        for i in range(1, len(parts)):
            q = base.joinpath(*parts[:i], "__init__.py")
            if q.is_file():
                out.append(str(q.relative_to(base)))
        f = base.joinpath(*parts).with_suffix(".py")
        if f.is_file():
            out.append(str(f.relative_to(base)))
        pkg = base.joinpath(*parts, "__init__.py")
        if pkg.is_file():
            out.append(str(pkg.relative_to(base)))
        return out

    def resolve_bare(mod: str) -> list[str]:
        """The trainer uses bare intra-package imports resolved via pretrain/ on sys.path."""
        f = base / "pretrain" / f"{mod.split('.')[0]}.py"
        return [str(f.relative_to(base))] if f.is_file() else []

    seen: set[str] = set()
    graph: dict[str, list[str]] = {}
    external: set[str] = set()
    stack = list(TRAINER_CLOSURE_ROOTS)
    while stack:
        rel = stack.pop()
        if rel in seen:
            continue
        seen.add(rel)
        tree = ast.parse((base / rel).read_text(encoding="utf-8"), filename=rel)
        deps: set[str] = set()
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            for n in names:
                hits = resolve(n) or resolve_bare(n)
                if hits:
                    deps.update(hits)
                else:
                    external.add(n.split(".")[0])
        graph[rel] = sorted(deps)
        stack.extend(d for d in deps if d not in seen)

    closure = sorted(seen)
    files = {c: file_sha256(base / c) for c in closure}
    digest = _sha256_bytes(
        canonical_json_bytes({
            "schema_version": EXECUTION_BUNDLE_SCHEMA,
            "files": dict(sorted(files.items())),
        })
    )
    unbound = [c for c in closure if c not in files]
    return {
        "bundle_schema_version": EXECUTION_BUNDLE_SCHEMA,
        "roots": list(TRAINER_CLOSURE_ROOTS),
        "derived_closure": closure,
        "derived_closure_count": len(closure),
        "files": files,
        "external_non_repository_modules": sorted(external),
        "local_import_graph": graph,
        "unbound_load_bearing_modules": unbound,
        "unbound_load_bearing_module_count": len(unbound),
        "TRAINER_EXECUTION_BUNDLE_SHA256": digest,
        "reused_pilot_execution_closure": closure == sorted(PILOT_EXECUTION_CLOSURE),
        "reused_stage_p_plan_closure": closure == sorted(STAGE_P_PLAN_CLOSURE),
    }


def trainer_execution_bundle_sha256() -> str:
    return trainer_execution_closure()["TRAINER_EXECUTION_BUNDLE_SHA256"]


# --------------------------------------------------------------- git / runtime observation


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root()), *args], capture_output=True, text=True, check=False
    ).stdout.strip()


def observed_repository() -> dict[str, Any]:
    return {"branch": _git("rev-parse", "--abbrev-ref", "HEAD"), "head": _git("rev-parse", "HEAD")}


def observed_training_runtime() -> dict[str, Any]:
    """Observe the live runtime for authorization comparison. Never a policy choice."""
    import platform

    import numpy as np
    import torch

    gpu: dict[str, Any] = {
        "visible_cuda_device_count": int(torch.cuda.device_count())
        if torch.cuda.is_available()
        else 0,
        "gpu_uuid": None,
        "gpu_pci_bus_id": None,
        "gpu_name": None,
        "total_vram_bytes": None,
        "compute_capability": None,
        "driver_version": None,
        "cuda_runtime_version": torch.version.cuda,
    }
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        index = int(torch.cuda.current_device())
        props = torch.cuda.get_device_properties(index)
        gpu.update({
            "gpu_name": torch.cuda.get_device_name(index),
            "total_vram_bytes": int(props.total_memory),
            "compute_capability": f"{props.major}.{props.minor}",
        })
        query = (
            subprocess
            .run(
                [
                    "nvidia-smi",
                    "--query-gpu=uuid,pci.bus_id,driver_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            .stdout.strip()
            .splitlines()
        )
        if query:
            parts = [p.strip() for p in query[0].split(",")]
            if len(parts) >= 3:
                gpu["gpu_uuid"], gpu["gpu_pci_bus_id"], gpu["driver_version"] = parts[:3]
    repo = observed_repository()
    return {
        **gpu,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "trainer_head": repo["head"],
        "trainer_execution_bundle_sha256": trainer_execution_bundle_sha256(),
        "canonical_cwd": CANONICAL_CWD,
    }


# --------------------------------------------------------------- the enforcement gate


def _as_int_list(values: object) -> list[int]:
    if values is None:
        return []
    if isinstance(values, (str, int)):
        values = [values]
    out: list[int] = []
    for value in values:  # type: ignore[union-attr]
        if isinstance(value, int):
            out.append(int(value))
            continue
        for field in str(value).split(","):
            field = field.strip()
            if field:
                out.append(int(field))
    return out


def _cmp(failures: list[str], name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        failures.append(f"{name}: expected {expected!r}, got {actual!r}")


def validate_governed_args(args: argparse.Namespace, *, stage: str) -> list[str]:
    """Every governed CLI value, checked before any model or optimizer exists."""
    require(stage in STAGE_ORDER, f"unknown stage {stage!r}")
    f: list[str] = []

    # ---- model contract ----
    _cmp(f, "layers", int(getattr(args, "layers", -1)), MODEL_CONTRACT["n_layers"])
    _cmp(f, "d_model", int(getattr(args, "d_model", -1)), MODEL_CONTRACT["d_model"])
    _cmp(f, "n_heads", int(getattr(args, "n_heads", -1)), MODEL_CONTRACT["n_heads"])
    _cmp(f, "n_kv_heads", int(getattr(args, "n_kv_heads", -1)), MODEL_CONTRACT["n_kv_heads"])
    _cmp(f, "d_ff", int(getattr(args, "d_ff", -1)), MODEL_CONTRACT["d_ff"])
    _cmp(f, "seq_len", int(getattr(args, "seq_len", -1)), MODEL_CONTRACT["seq_len"])
    _cmp(f, "vocab_size", int(getattr(args, "vocab_size", -1)), MODEL_CONTRACT["vocab_size"])
    _cmp(f, "dropout", float(getattr(args, "dropout", -1.0)), MODEL_CONTRACT["dropout"])

    # ---- optimizer / schedule ----
    _cmp(f, "optimizer", str(getattr(args, "optimizer", "")), OPTIMIZER)
    _cmp(f, "lr", float(getattr(args, "lr", -1.0)), PEAK_LR)
    _cmp(f, "muon_lr", float(getattr(args, "muon_lr", -1.0)), MUON_LR)
    _cmp(f, "muon_momentum", float(getattr(args, "muon_momentum", -1.0)), MUON_MOMENTUM)
    _cmp(f, "min_lr_ratio", float(getattr(args, "min_lr_ratio", -1.0)), MIN_LR_RATIO)
    _cmp(f, "warmup_steps", int(getattr(args, "warmup_steps", -1)), WARMUP_STEPS)
    _cmp(f, "lr_schedule", str(getattr(args, "lr_schedule", "")), LR_SCHEDULE)
    _cmp(f, "weight_decay", float(getattr(args, "weight_decay", -1.0)), WEIGHT_DECAY)
    _cmp(f, "grad_clip", float(getattr(args, "grad_clip", -1.0)), GRAD_CLIP)
    _cmp(f, "micro_bsz", int(getattr(args, "micro_bsz", -1)), MICRO_BSZ)
    _cmp(f, "grad_accum", int(getattr(args, "grad_accum", -1)), GRAD_ACCUM)
    _cmp(f, "precision", str(getattr(args, "precision", "")), PRECISION)
    if bool(getattr(args, "compile", False)) is not COMPILE:
        f.append(f"compile: expected {COMPILE!r}, got {getattr(args, 'compile', None)!r}")

    # effective batch is derived, never independently supplied
    realized_tokens = (
        int(getattr(args, "micro_bsz", 0))
        * int(getattr(args, "grad_accum", 0))
        * int(getattr(args, "seq_len", 0))
    )
    _cmp(f, "effective_batch_tokens", realized_tokens, EFFECTIVE_BATCH_TOKENS)

    # ---- exact-plan geometry ----
    _cmp(
        f,
        "schedule_total_steps",
        int(getattr(args, "schedule_total_steps", -1)),
        SCHEDULE_TOTAL_STEPS,
    )
    _cmp(f, "decay_start_step", int(getattr(args, "decay_start_step", -1)), DECAY_START_STEP)
    _cmp(f, "decay_end_step", int(getattr(args, "decay_end_step", -1)), DECAY_END_STEP)
    _cmp(f, "run_plan_stage", str(getattr(args, "run_plan_stage", "")), stage)
    _cmp(
        f,
        "data_stage_start_step",
        int(getattr(args, "data_stage_start_step", -1)),
        STAGE_BOUNDARIES[stage]["start_step"],
    )
    _cmp(f, "max_steps", int(getattr(args, "max_steps", -1)), STAGE_BOUNDARIES[stage]["stop_step"])

    # ---- seeds ----
    _cmp(f, "seed", int(getattr(args, "seed", -1)), MODEL_INIT_SEED)
    _cmp(f, "val_seed", int(getattr(args, "val_seed", -1)), VALIDATION_SEED)
    _cmp(
        f,
        "stage_a_sampler_seed",
        int(getattr(args, "stage_a_sampler_seed", -1)),
        STAGE_A_SAMPLER_SEED,
    )
    _cmp(
        f,
        "stage_b_sampler_seed",
        int(getattr(args, "stage_b_sampler_seed", -1)),
        STAGE_B_SAMPLER_SEED,
    )
    if STAGE_A_SAMPLER_SEED == STAGE_B_SAMPLER_SEED:
        f.append("stage_a_sampler_seed and stage_b_sampler_seed must remain distinct")
    for name, value in (
        ("seed", int(getattr(args, "seed", -1))),
        ("stage_a_sampler_seed", int(getattr(args, "stage_a_sampler_seed", -1))),
        ("stage_b_sampler_seed", int(getattr(args, "stage_b_sampler_seed", -1))),
        ("val_seed", int(getattr(args, "val_seed", -1))),
    ):
        if value in _PILOT_SEEDS:
            f.append(f"{name}: pilot seed {value} is not a production seed")

    # ---- evaluation policy ----
    eval_steps = _as_int_list(getattr(args, "eval_steps", []))
    if tuple(eval_steps) != tuple(EVALUATION_MILESTONES):
        f.append(f"eval_steps: expected exactly {list(EVALUATION_MILESTONES)}, got {eval_steps}")
    _cmp(f, "eval_every", int(getattr(args, "eval_every", -1)), PERIODIC_EVAL_EVERY)
    _cmp(f, "bench_eval_every", int(getattr(args, "bench_eval_every", -1)), BENCHMARK_EVAL_EVERY)
    if str(getattr(args, "bench_eval_path", "") or "").strip():
        f.append("bench_eval_path must be unset: benchmark evaluation is DISABLED")
    _cmp(f, "val_samples", int(getattr(args, "val_samples", -1)), VALIDATION_SAMPLES)
    _cmp(
        f,
        "val_samples_per_source",
        int(getattr(args, "val_samples_per_source", -1)),
        VALIDATION_SAMPLES_PER_SOURCE,
    )
    _cmp(f, "eos_weight", float(getattr(args, "eos_weight", -1.0)), VALIDATION_EOS_WEIGHT)
    _cmp(f, "eos_weight_warmup_steps", int(getattr(args, "eos_weight_warmup_steps", -1)), 0)

    # ---- checkpoint policy ----
    save_steps = _as_int_list(getattr(args, "save_steps", []))
    if tuple(save_steps) != tuple(CHECKPOINT_MILESTONES):
        f.append(f"save_steps: expected exactly {list(CHECKPOINT_MILESTONES)}, got {save_steps}")
    _cmp(f, "save_every", int(getattr(args, "save_every", -1)), PERIODIC_SAVE_EVERY)

    # ---- loss masking / escape hatches ----
    for flag in (
        "mask_last_label_in_loss",
        "no_mask_bos_in_loss",
        "no_mask_last_label_in_loss",
        "allow_weights_only_resume",
        "allow_data_branch",
        "allow_schedule_branch",
    ):
        if bool(getattr(args, flag, False)):
            f.append(f"{flag} is forbidden on the governed path")
    if not bool(getattr(args, "strict_resume_contract", True)):
        f.append("strict_resume_contract must remain enabled on the governed path")

    # ---- legacy shared sampler seed must not steer a governed run ----
    if getattr(args, "sampler_seed", None) is not None:
        legacy = int(args.sampler_seed)
        if legacy in (STAGE_A_SAMPLER_SEED, STAGE_B_SAMPLER_SEED):
            f.append(
                "legacy --sampler_seed must not carry a governed stage seed; the governed "
                "path reads only --stage_a_sampler_seed / --stage_b_sampler_seed"
            )
    return f


def validate_governed_launch(
    args: argparse.Namespace,
    *,
    stage: str,
    launch_contract: Mapping[str, Any] | None,
    authorization: Mapping[str, Any] | None,
    exact_plan_sha256: str,
    pilot_owner_acceptance_sha256: str,
    observed_runtime: Mapping[str, Any] | None = None,
    cwd: str | None = None,
) -> dict[str, Any]:
    """THE gate. Runs before any model, optimizer, sampler or dataset is constructed."""
    failures: list[str] = []

    resolved_cwd = str(Path(cwd if cwd is not None else os.getcwd()).resolve())
    if resolved_cwd != CANONICAL_CWD:
        failures.append(
            f"canonical_cwd: governed launch must run from {CANONICAL_CWD}, got {resolved_cwd}"
        )

    expected_contract_sha = contract_sha256()
    if launch_contract is None:
        failures.append("launch_contract_missing")
    else:
        if launch_contract.get("schema_version") != CONTRACT_SCHEMA:
            failures.append("launch_contract_schema_mismatch")
        supplied = launch_contract.get("launch_contract_sha256")
        if supplied is not None and supplied != expected_contract_sha:
            failures.append("launch_contract_sha256_mismatch")

    if exact_plan_sha256 != EXACT_RUN_PLAN_SHA256:
        failures.append(
            f"exact_run_plan_sha256: expected {EXACT_RUN_PLAN_SHA256}, got {exact_plan_sha256}"
        )
    if pilot_owner_acceptance_sha256 != PILOT_OWNER_ACCEPTANCE_SHA256:
        failures.append("pilot_owner_acceptance_sha256_mismatch")

    failures.extend(validate_governed_args(args, stage=stage))

    repo = observed_repository()
    observed = {
        "branch": repo["branch"],
        "head": repo["head"],
        "trainer_execution_bundle_sha256": trainer_execution_bundle_sha256(),
        "launch_contract_sha256": expected_contract_sha,
        "exact_run_plan_sha256": exact_plan_sha256,
        "pilot_owner_acceptance_sha256": pilot_owner_acceptance_sha256,
        "output_root": str(Path(str(getattr(args, "out_dir", ""))).resolve())
        if getattr(args, "out_dir", "")
        else None,
        "canonical_cwd": resolved_cwd,
    }
    if observed_runtime is not None:
        observed["training_runtime"] = dict(observed_runtime)

    scope = "STAGE_N" if stage == "stage_a" else "STAGE_O"
    verdict = validate_authorization(authorization, requested_scope=scope, observed=observed)
    failures.extend(verdict["failures"])

    failures = list(dict.fromkeys(failures))
    return {
        "governed": True,
        "stage": stage,
        "requested_scope": scope,
        "launch_contract_sha256": expected_contract_sha,
        "trainer_execution_bundle_sha256": observed["trainer_execution_bundle_sha256"],
        "observed": {k: v for k, v in observed.items() if k != "training_runtime"},
        "authorization_verdict": verdict,
        "failures": failures,
        "authorized": not failures,
    }


def require_governed_launch(**kwargs: Any) -> dict[str, Any]:
    """Fail closed before model construction. Never warns, never degrades."""
    result = validate_governed_launch(**kwargs)
    require(
        result["authorized"],
        f"governed launch refused under {CONTRACT_VERSION}:\n  - "
        + "\n  - ".join(result["failures"]),
    )
    return result


# --------------------------------------------------------------- governed run contract


def governed_run_contract(
    *,
    stage: str,
    launch_contract_sha256: str,
    stage_authorization_sha256: str,
    exact_plan_sha256: str,
    pilot_owner_acceptance_sha256: str,
    trainer_head: str,
    trainer_execution_bundle_sha256: str,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """The normalized artifact published atomically BEFORE the first optimizer update."""
    require(stage in STAGE_ORDER, f"unknown stage {stage!r}")
    return {
        "schema_version": RUN_CONTRACT_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "stage": stage,
        "scope": "STAGE_N" if stage == "stage_a" else "STAGE_O",
        "launch_contract_sha256": launch_contract_sha256,
        "stage_authorization_sha256": stage_authorization_sha256,
        "exact_run_plan_sha256": exact_plan_sha256,
        "pilot_owner_acceptance_sha256": pilot_owner_acceptance_sha256,
        "trainer_head": trainer_head,
        "trainer_execution_bundle_sha256": trainer_execution_bundle_sha256,
        "model": {**dict(MODEL_CONTRACT), "parameter_count": MODEL_PARAMETER_COUNT},
        "training": {
            "optimizer": OPTIMIZER,
            "muon_lr": MUON_LR,
            "muon_momentum": MUON_MOMENTUM,
            "peak_lr": PEAK_LR,
            "min_lr_ratio": MIN_LR_RATIO,
            "micro_bsz": MICRO_BSZ,
            "grad_accum": GRAD_ACCUM,
            "effective_batch_tokens": EFFECTIVE_BATCH_TOKENS,
            "compile": COMPILE,
            "precision": PRECISION,
            "warmup_steps": WARMUP_STEPS,
            "decay_fraction": DECAY_FRACTION,
            "lr_schedule": LR_SCHEDULE,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
        },
        "seeds": dict(SEED_TUPLE),
        "stage_sampler_seed": stage_sampler_seed(stage),
        "evaluation_policy": {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in EVALUATION_POLICY.items()
        },
        "checkpoint_policy": {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in CHECKPOINT_POLICY.items()
        },
        "completed_evaluation_milestones": [],
        "runtime": dict(runtime),
        "gpu_uuid": runtime.get("gpu_uuid"),
        "gpu_pci_bus_id": runtime.get("gpu_pci_bus_id"),
        "canonical_cwd": CANONICAL_CWD,
        "stage_start_step": STAGE_BOUNDARIES[stage]["start_step"],
        "stage_stop_step": STAGE_BOUNDARIES[stage]["stop_step"],
        "optimizer_reset_at_a_b": OPTIMIZER_RESET_AT_A_B,
        "scheduler_reset_at_a_b": SCHEDULER_RESET_AT_A_B,
    }


def governed_run_contract_sha256(contract: Mapping[str, Any]) -> str:
    payload = {k: v for k, v in contract.items() if k != "completed_evaluation_milestones"}
    return _sha256_bytes(canonical_json_bytes(payload))


GOVERNED_RESUME_IMMUTABLE_FIELDS = (
    "launch_contract_sha256",
    "stage_authorization_sha256",
    "exact_run_plan_sha256",
    "pilot_owner_acceptance_sha256",
    "trainer_head",
    "trainer_execution_bundle_sha256",
    "model",
    "training",
    "seeds",
    "stage_sampler_seed",
    "evaluation_policy",
    "checkpoint_policy",
    "gpu_uuid",
    "gpu_pci_bus_id",
    "canonical_cwd",
    "stage",
    "stage_start_step",
    "stage_stop_step",
)


def validate_governed_resume(
    checkpoint_contract: Mapping[str, Any], current_contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Resume rejects any drift. No CLI flag may override a checkpoint-bound value."""
    failures = [
        f"resume_mismatch:{field}"
        for field in GOVERNED_RESUME_IMMUTABLE_FIELDS
        if checkpoint_contract.get(field) != current_contract.get(field)
    ]
    a = governed_run_contract_sha256(checkpoint_contract)
    b = governed_run_contract_sha256(current_contract)
    if a != b:
        failures.append("resume_mismatch:governed_run_contract_sha256")
    return {
        "compatible": not failures,
        "failures": failures,
        "checkpoint_run_contract_sha256": a,
        "current_run_contract_sha256": b,
    }


# --------------------------------------------------------------- compile fail-closed


def bind_compiled_callable(model: Any, *, compile_requested: bool) -> dict[str, Any]:
    """Compile fail-closed. A governed run never continues eagerly after a compile failure.

    Returns evidence that the compiled callable is the callable training will use; the
    caller must use the returned ``module``.
    """
    import torch

    if not compile_requested:
        return {
            "compile_requested": False,
            "compiled": False,
            "module": model,
            "compiled_callable_is_training_callable": True,
        }
    require(
        callable(getattr(torch, "compile", None)),
        "governed run requires compile=true but torch.compile is unavailable",
    )
    try:
        compiled = torch.compile(model)
    except Exception as exc:  # noqa: BLE001 - fail closed, never fall back to eager
        raise LaunchContractError(
            f"governed run requires compile=true and torch.compile failed: {exc!r}; "
            "eager fallback is forbidden and the run is aborted"
        ) from exc
    require(
        compiled is not model,
        "torch.compile returned the original eager module; a governed run may not claim "
        "compile=true after an eager fallback",
    )
    return {
        "compile_requested": True,
        "compiled": True,
        "module": compiled,
        "compiled_callable_type": type(compiled).__name__,
        "eager_module_type": type(model).__name__,
        "compiled_callable_is_training_callable": True,
        "eager_fallback_occurred": False,
    }


def assert_compile_binding(run_contract: Mapping[str, Any], evidence: Mapping[str, Any]) -> None:
    """A governed contract may never claim compile=true after an eager fallback."""
    claimed = bool((run_contract.get("training") or {}).get("compile"))
    if claimed:
        require(
            bool(evidence.get("compiled")) and not evidence.get("eager_fallback_occurred", False),
            "run contract claims compile=true but no compiled callable was bound",
        )
        require(
            bool(evidence.get("compiled_callable_is_training_callable")),
            "the compiled callable is not the callable used for training",
        )


__all__ = [
    "CONTRACT_VERSION",
    "CONTRACT_SCHEMA",
    "AUTHORIZATION_SCHEMA",
    "RUN_CONTRACT_SCHEMA",
    "LaunchContractError",
    "ALLOWED_SCOPES",
    "EVALUATION_MILESTONES",
    "CHECKPOINT_MILESTONES",
    "PARSER_FIELD_CLASSIFICATION",
    "SEED_TUPLE",
    "authorization_template",
    "classify_parser_namespace",
    "contract_document",
    "contract_sha256",
    "governed_run_contract",
    "governed_run_contract_sha256",
    "require_governed_launch",
    "save_steps_cli_flags",
    "eval_steps_cli_flags",
    "stage_sampler_seed",
    "trainer_execution_closure",
    "trainer_execution_bundle_sha256",
    "validate_authorization",
    "validate_governed_args",
    "validate_governed_launch",
    "validate_governed_resume",
    "verify_realized_optimizer",
    "bind_compiled_callable",
    "assert_compile_binding",
    "observed_training_runtime",
    "worker_rng_seed",
    "realized_muon_contract",
]
