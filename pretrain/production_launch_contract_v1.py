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
    "num_workers",
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
    # Owner clarification 1: not freely mutable. The authorization binds the exact value, it
    # enters the runtime fingerprint and governed run contract, and Stage O must use the value
    # Stage N accepted unless Stage N is rerun.
    "num_workers": {
        "class": LAUNCH_AUTHORIZATION_BOUND,
        "value": None,
        "affects": ("data order", "randomness", "runtime identity"),
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
        "class": LAUNCH_AUTHORIZATION_BOUND,
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
    "resume",
)


# Owner clarification 2: exactly two resume modes, both authorization-bound.
RESUME_MODES = ("FRESH", "RESUME_EXACT_CHECKPOINT")
RESUME_REQUIRED_FIELDS_BY_MODE = MappingProxyType({
    "FRESH": (),
    "RESUME_EXACT_CHECKPOINT": (
        "checkpoint_path",
        "checkpoint_sha256",
        "expected_step",
        "stage",
        "governed_run_contract_sha256",
    ),
})

# Canonical special-token IDs are part of the frozen policy, not a tokenizer detail.
SPECIAL_TOKEN_IDS = MappingProxyType({
    "PAD": 0,
    "UNK": 1,
    "BOS": 2,
    "EOS": 3,
    "SYSTEM": 4,
    "USER": 5,
    "ASSISTANT": 6,
})
CANONICAL_BOS_ID = SPECIAL_TOKEN_IDS["BOS"]
CANONICAL_EOS_ID = SPECIAL_TOKEN_IDS["EOS"]


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


def observed_training_runtime(*, num_workers: int | None = None) -> dict[str, Any]:
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
        # R1 Part 10: the FIRST nvidia-smi row is not the selected device in general.
        # Resolve Torch's selected logical device through CUDA_VISIBLE_DEVICES (index or
        # UUID form) to its physical NVML record, and refuse an ambiguous mapping.
        resolved = resolve_selected_gpu_identity(
            torch_device_name=gpu["gpu_name"], selected_index=index
        )
        gpu["selected_device_resolution"] = resolved
        if resolved.get("resolved"):
            gpu["gpu_uuid"] = resolved["gpu_uuid"]
            gpu["gpu_pci_bus_id"] = resolved["gpu_pci_bus_id"]
            gpu["driver_version"] = resolved["driver_version"]
            gpu["selected_physical_index"] = resolved["selected_physical_index"]
        else:
            gpu["gpu_uuid"] = None
            gpu["gpu_pci_bus_id"] = None
            gpu["selected_device_resolution_failures"] = resolved.get("failures", [])
    repo = observed_repository()
    return {
        **gpu,
        # Owner clarification 1: num_workers is part of the bound runtime identity.
        "num_workers": num_workers,
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

    # ---- legacy shared sampler seed ----
    # Owner clarification 4 (R1) supersedes the earlier "must not equal a stage seed" rule:
    # the legacy field is mechanically normalized to the ACTIVE stage seed and then validated,
    # so it can never select a different permutation. validate_legacy_sampler_seed owns that
    # check; keeping the inverted rule here would make a correctly normalized run unlaunchable.
    f.extend(validate_legacy_sampler_seed(args, stage))
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


# ===================================================================================
# R1 real-path repair
# ===================================================================================

# --------------------------------------------------------------- Part 10: device identity


def _nvml_records() -> list[dict[str, Any]]:
    """Every physical GPU nvidia-smi reports, in physical index order."""
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    records = []
    for line in query.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            records.append({
                "physical_index": int(parts[0]),
                "uuid": parts[1],
                "pci_bus_id": parts[2],
                "name": parts[3],
                "memory_total": parts[4],
                "driver_version": parts[5],
            })
    return records


def resolve_selected_gpu_identity(
    *,
    cuda_visible_devices: str | None = None,
    records: Sequence[Mapping[str, Any]] | None = None,
    torch_device_name: str | None = None,
    selected_index: int = 0,
) -> dict[str, Any]:
    """Resolve Torch's selected logical device to its physical NVML record.

    The first ``nvidia-smi`` row is NOT the selected device in general.
    ``CUDA_VISIBLE_DEVICES`` reorders and filters the visible set, and it may be written as
    indices or as UUIDs. This resolves the logical index Torch actually selected through that
    mapping, and refuses to guess when the mapping is ambiguous or inconsistent.
    """
    physical = [dict(r) for r in (records if records is not None else _nvml_records())]
    raw = (
        cuda_visible_devices
        if cuda_visible_devices is not None
        else os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    failures: list[str] = []

    if not physical:
        return {
            "resolved": False,
            "failures": ["no_physical_gpu_records"],
            "cuda_visible_devices": raw,
            "physical_record_count": 0,
        }

    if raw is None or not str(raw).strip():
        # No filtering: logical order equals physical order.
        visible = physical
        mapping_form = "unset_all_physical_visible"
    else:
        tokens = [t.strip() for t in str(raw).split(",") if t.strip()]
        mapping_form = "index" if all(t.isdigit() for t in tokens) else "uuid"
        visible = []
        for token in tokens:
            if token.isdigit():
                match = [r for r in physical if r["physical_index"] == int(token)]
            else:
                # UUID form; NVML accepts a full or unambiguous prefix.
                match = [r for r in physical if r["uuid"] == token]
                if not match:
                    match = [r for r in physical if r["uuid"].startswith(token)]
            if len(match) != 1:
                failures.append(
                    f"cuda_visible_devices_token_unresolvable:{token}"
                    if not match
                    else f"cuda_visible_devices_token_ambiguous:{token}"
                )
            else:
                visible.append(match[0])

    if failures:
        return {
            "resolved": False,
            "failures": failures,
            "cuda_visible_devices": raw,
            "mapping_form": mapping_form,
            "physical_record_count": len(physical),
        }

    if selected_index < 0 or selected_index >= len(visible):
        return {
            "resolved": False,
            "failures": [f"selected_logical_index_out_of_visible_range:{selected_index}"],
            "cuda_visible_devices": raw,
            "mapping_form": mapping_form,
            "visible_device_count": len(visible),
        }

    chosen = visible[selected_index]
    if torch_device_name is not None and chosen["name"] != torch_device_name:
        failures.append(
            f"selected_device_name_inconsistent:nvml={chosen['name']!r},torch={torch_device_name!r}"
        )
    if len(visible) != REQUIRED_TRAINING_DEVICE_COUNT:
        failures.append(f"visible_device_count_not_exactly_1:{len(visible)}")

    return {
        "resolved": not failures,
        "failures": failures,
        "cuda_visible_devices": raw,
        "mapping_form": mapping_form,
        "physical_record_count": len(physical),
        "visible_device_count": len(visible),
        "selected_logical_index": selected_index,
        "selected_physical_index": chosen["physical_index"],
        "gpu_uuid": chosen["uuid"],
        "gpu_pci_bus_id": chosen["pci_bus_id"],
        "gpu_name": chosen["name"],
        "driver_version": chosen["driver_version"],
    }


REQUIRED_TRAINING_DEVICE_COUNT = 1


# --------------------------------------------------------------- Part 1: artifact auth

# Every load-bearing contract field the supplied artifact must reproduce exactly. A supplied
# document is never trusted: each of these is compared against the code authority, so an
# altered peak_lr, seed, cadence, model value or authorizes_training flag fails here.
LAUNCH_CONTRACT_AUTHORITATIVE_PATHS = (
    ("schema_version",),
    ("contract_version",),
    ("authorization_status",),
    ("authorizes_training",),
    ("accepted_stage_p_inputs",),
    ("data",),
    ("model",),
    ("training",),
    ("boundaries",),
    ("seeds",),
    ("evaluation_policy",),
    ("checkpoint_policy",),
    ("canonical_cwd",),
    ("runtime_binding",),
    ("parser_field_classification",),
)


def _dig(doc: Mapping[str, Any], path: Sequence[str]) -> Any:
    node: Any = doc
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            return _MISSING
        node = node[key]
    return node


class _Missing:
    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "<missing>"


_MISSING = _Missing()


def load_launch_contract_artifact(path: str | Path) -> dict[str, Any]:
    """Authenticate a launch-contract artifact from its ACTUAL bytes on disk.

    Never trusts an in-memory dictionary or a self-declared SHA. The bytes are read, parsed
    as canonical JSON, hashed, and every load-bearing field is compared with the code
    authority. Any divergence raises before the caller can construct anything.
    """
    artifact = Path(path)
    if not artifact.is_file():
        raise LaunchContractError(f"launch contract artifact not found: {artifact}")
    body = artifact.read_bytes()
    try:
        supplied = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise LaunchContractError(
            f"launch contract artifact is not readable canonical JSON: {exc}"
        ) from exc
    if not isinstance(supplied, Mapping):
        raise LaunchContractError("launch contract artifact is not a JSON object")

    observed_sha = _sha256_bytes(body)
    authority = contract_document()
    authority_sha = contract_sha256()

    failures: list[str] = []
    for path_tuple in LAUNCH_CONTRACT_AUTHORITATIVE_PATHS:
        want = _dig(authority, path_tuple)
        got = _dig(supplied, path_tuple)
        if isinstance(got, _Missing):
            failures.append(f"launch_contract_missing_field:{'.'.join(path_tuple)}")
        elif got != want:
            failures.append(f"launch_contract_field_mismatch:{'.'.join(path_tuple)}")

    # The canonical bytes must round-trip to the code authority exactly.
    if observed_sha != authority_sha:
        failures.append("launch_contract_sha256_does_not_match_code_authority")

    # A self-declared SHA is allowed to exist but is never the authority.
    declared = supplied.get("launch_contract_sha256")
    if declared is not None and declared != authority_sha:
        failures.append("launch_contract_self_declared_sha256_mismatch")

    if failures:
        raise LaunchContractError(
            "launch contract artifact failed authentication:\n  - " + "\n  - ".join(failures)
        )
    return {
        "path": str(artifact),
        "sha256": observed_sha,
        "document": dict(supplied),
        "matches_code_authority": True,
    }


# --------------------------------------------------------------- Part 2: full enforcement


def validate_special_token_binding(args: argparse.Namespace) -> list[str]:
    """Canonical special-token IDs are frozen policy and must be reproduced exactly."""
    failures: list[str] = []
    _cmp(failures, "bos_id", int(getattr(args, "bos_id", -1)), CANONICAL_BOS_ID)
    _cmp(failures, "eos_id", int(getattr(args, "eos_id", -1)), CANONICAL_EOS_ID)
    return failures


def verify_tokenizer_special_ids(tokenizer_path: str | Path) -> dict[str, Any]:
    """Confirm the real tokenizer realizes the frozen special-token identity."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer_path))
    observed = {
        name: tok.token_to_id(literal)
        for name, literal in (
            ("PAD", "[PAD]"),
            ("UNK", "[UNK]"),
            ("BOS", "[BOS]"),
            ("EOS", "[EOS]"),
            ("SYSTEM", "<|system|>"),
            ("USER", "<|user|>"),
            ("ASSISTANT", "<|assistant|>"),
        )
    }
    failures = [
        f"special_token_id_mismatch:{name}: expected {want}, got {observed.get(name)!r}"
        for name, want in SPECIAL_TOKEN_IDS.items()
        if observed.get(name) != want
    ]
    return {"observed": observed, "failures": failures, "matches": not failures}


def validate_diagnostic_fields(args: argparse.Namespace) -> list[str]:
    """A diagnostic classification is not permission to accept arbitrary values."""
    failures: list[str] = []
    for dest, spec in PARSER_FIELD_CLASSIFICATION.items():
        if spec["class"] != DIAGNOSTIC_ONLY:
            continue
        expected = spec["value"]
        if expected is None:
            failures.append(f"diagnostic_field_without_explicit_allowed_value:{dest}")
            continue
        if not hasattr(args, dest):
            continue
        actual = getattr(args, dest)
        if isinstance(expected, bool):
            ok = bool(actual) is expected
        elif isinstance(expected, float):
            ok = float(actual) == expected
        elif isinstance(expected, int):
            ok = int(actual) == expected
        else:
            ok = str(actual) == str(expected)
        if not ok:
            failures.append(f"{dest}: expected {expected!r}, got {actual!r}")
    return failures


def validate_resume_binding(
    args: argparse.Namespace, resume_binding: Mapping[str, Any] | None
) -> list[str]:
    """Resume controls are authorization-bound; no arbitrary CLI override is permitted."""
    failures: list[str] = []
    if not isinstance(resume_binding, Mapping):
        return ["resume_binding_missing_from_authorization"]
    mode = resume_binding.get("mode")
    if mode not in RESUME_MODES:
        return [f"resume_mode_invalid:{mode!r}"]
    for field in RESUME_REQUIRED_FIELDS_BY_MODE[mode]:
        if resume_binding.get(field) in (None, ""):
            failures.append(f"resume_binding_missing_field:{field}")

    cli_path = str(getattr(args, "resume_path", "") or "").strip()
    cli_full = bool(getattr(args, "resume_full", False))
    cli_step = int(getattr(args, "resume_step", -1))

    if mode == "FRESH":
        if cli_path:
            failures.append("resume_mode FRESH forbids --resume_path")
        if cli_full:
            failures.append("resume_mode FRESH forbids --resume_full")
        if cli_step != -1:
            failures.append("resume_mode FRESH forbids --resume_step")
    else:
        want_path = str(resume_binding.get("checkpoint_path", ""))
        if cli_path != want_path:
            failures.append(
                f"resume_path: expected the authorized checkpoint {want_path!r}, got {cli_path!r}"
            )
        if not cli_full:
            failures.append("RESUME_EXACT_CHECKPOINT requires --resume_full")
        want_step = int(resume_binding.get("expected_step", -1))
        if cli_step not in (-1, want_step):
            failures.append(f"resume_step: expected {want_step} (or unset), got {cli_step}")
    return failures


def validate_num_workers_binding(
    args: argparse.Namespace, authorized_num_workers: Any
) -> list[str]:
    """num_workers is not freely mutable: the authorization binds the exact value."""
    if authorized_num_workers is None:
        return ["num_workers_not_bound_by_authorization"]
    if not isinstance(authorized_num_workers, int) or isinstance(authorized_num_workers, bool):
        return [f"num_workers_binding_invalid:{authorized_num_workers!r}"]
    actual = int(getattr(args, "num_workers", -1))
    if actual != authorized_num_workers:
        return [f"num_workers: expected {authorized_num_workers}, got {actual}"]
    if actual < 0:
        return [f"num_workers must be >= 0, got {actual}"]
    return []


def normalize_legacy_sampler_seed(args: argparse.Namespace, stage: str) -> int:
    """Owner clarification 4: the legacy field never selects a different permutation.

    In a governed run the legacy ``--sampler_seed`` is mechanically normalized to the active
    stage seed. It is not an independent authority and cannot choose another permutation.
    """
    active = stage_sampler_seed(stage)
    args.sampler_seed = active
    return active


def validate_legacy_sampler_seed(args: argparse.Namespace, stage: str) -> list[str]:
    active = stage_sampler_seed(stage)
    legacy = getattr(args, "sampler_seed", None)
    if legacy is None:
        return []
    if int(legacy) != active:
        return [
            f"legacy sampler_seed {int(legacy)} would select a different permutation than the "
            f"active {stage} seed {active}; it must be normalized or unset"
        ]
    return []


# --------------------------------------------------------------- Part 8: compile realization


class ObservedForward:
    """The exact callable the training loop invokes, wrapped so the choice becomes evidence.

    Reuses the independently accepted V2.3 pilot observation shape: compile is proven by
    which object was actually invoked and how often, never by a caller Boolean.
    """

    __slots__ = ("target", "compiled_object", "invocations")

    def __init__(self, target: Any, *, compiled_object: Any = None) -> None:
        self.target = target
        self.compiled_object = compiled_object
        self.invocations = 0

    @property
    def invoked_compiled_callable(self) -> bool:
        return self.compiled_object is not None and self.target is self.compiled_object

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.invocations += 1
        return self.target(*args, **kwargs)


def dynamo_counters_snapshot() -> dict[str, Any]:
    """TorchDynamo's own counters. Diagnostics that become evidence, never a dependency."""
    try:
        import torch

        raw = torch._dynamo.utils.counters
        return {k: dict(v) for k, v in raw.items()}
    except Exception:  # noqa: BLE001
        return {}


def reset_dynamo_counters() -> None:
    try:
        import torch

        torch._dynamo.utils.counters.clear()
    except Exception:  # noqa: BLE001
        pass


def compile_realization_evidence(
    module: Any,
    forward: ObservedForward,
    *,
    requested: bool,
    cache_dir: str | Path | None,
    expected_forward_invocations: int,
    counters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Structured evidence that compile was LAZILY REALIZED, not merely requested.

    ``torch.compile`` returns a wrapper eagerly and compiles on first call, so a distinct
    wrapper proves nothing. Realization is proven by Dynamo actually producing a graph, or by
    Inductor leaving artifacts on disk, after the compiled callable was invoked.
    """
    module_type = f"{type(module).__module__}.{type(module).__qualname__}"
    is_optimized_module = "OptimizedModule" in module_type
    snapshot = dict(counters) if counters is not None else dynamo_counters_snapshot()
    unique_graphs = int(snapshot.get("stats", {}).get("unique_graphs", 0))
    graph_breaks = int(sum(int(v) for v in snapshot.get("graph_break", {}).values()))
    # A recompile-limit fallback silently drops back to eager; Dynamo records it.
    recompile_reasons = dict(snapshot.get("recompile_reasons", {}) or {})
    recompile_limit_hit = any(
        "cache_size_limit" in str(k) or "recompile_limit" in str(k)
        for k in list(recompile_reasons) + list(snapshot.get("unimplemented", {}) or {})
    )

    cache = Path(cache_dir) if cache_dir is not None else None
    artifacts = [p for p in cache.rglob("*") if p.is_file()] if cache and cache.is_dir() else []
    invoked_compiled = bool(forward.invoked_compiled_callable)
    invocations_match = int(forward.invocations) == int(expected_forward_invocations)
    compilation_materialized = bool(unique_graphs > 0 or artifacts)

    failures: list[str] = []
    if requested:
        if not invoked_compiled:
            failures.append("invoked_callable_is_not_the_compiled_object")
        if not is_optimized_module:
            failures.append("realized_module_is_not_an_OptimizedModule")
        if not compilation_materialized:
            failures.append("compile_never_materialized_lazily_eager_fallback")
        if not invocations_match:
            failures.append(
                f"forward_invocations {forward.invocations} != expected "
                f"{expected_forward_invocations}"
            )
        if recompile_limit_hit:
            failures.append("recompile_limit_fallback_to_eager_detected")
    else:
        if invoked_compiled or is_optimized_module or unique_graphs:
            failures.append("unrequested_compiled_path")

    evidence = {
        "schema_version": "petitgpt-governed-compile-evidence-v1",
        "compile_requested": bool(requested),
        "realized_module_type": module_type,
        "realized_module_is_optimized_module": is_optimized_module,
        "invoked_callable_type": (
            f"{type(forward.target).__module__}.{type(forward.target).__qualname__}"
        ),
        "invoked_compiled_callable": invoked_compiled,
        "forward_invocations": int(forward.invocations),
        "expected_forward_invocations": int(expected_forward_invocations),
        "forward_invocations_match": invocations_match,
        "dynamo_unique_graphs": unique_graphs,
        "dynamo_graph_breaks": graph_breaks,
        "recompile_limit_fallback_detected": recompile_limit_hit,
        "inductor_cache_dir": str(cache) if cache else None,
        "inductor_artifact_count": len(artifacts),
        "compilation_materialized": compilation_materialized,
        "eager_fallback_occurred": bool(requested and not compilation_materialized),
        "compiled_callable_is_training_callable": invoked_compiled,
        "failures": failures,
        "compile_realized": not failures,
    }
    evidence["compile_evidence_sha256"] = _sha256_bytes(
        canonical_json_bytes({k: v for k, v in evidence.items() if k != "failures"})
    )
    return evidence


def require_compile_realized(evidence: Mapping[str, Any]) -> None:
    """Fail closed. No governed optimizer update may precede realized compile evidence."""
    require(
        bool(evidence.get("compile_realized")),
        "governed run requires compile=true to be lazily realized before the first "
        "optimizer update; observed failures: " + ", ".join(evidence.get("failures", [])),
    )


def bind_compiled_callable_governed(model: Any) -> dict[str, Any]:
    """Compile eagerly-wrap, fail closed on raise or identity return. Realization comes later."""
    import torch

    require(
        callable(getattr(torch, "compile", None)),
        "governed run requires compile=true but torch.compile is unavailable",
    )
    reset_dynamo_counters()
    try:
        compiled = torch.compile(model)
    except Exception as exc:  # noqa: BLE001 - fail closed; eager fallback is forbidden
        raise LaunchContractError(
            f"governed run requires compile=true and torch.compile raised: {exc!r}; "
            "eager fallback is forbidden and the run is aborted"
        ) from exc
    require(
        compiled is not model,
        "torch.compile returned the original eager module; a governed run may not claim "
        "compile=true after an identity/eager return",
    )
    return {"compiled_module": compiled, "eager_module": model}


# --------------------------------------------------------------- Part 3: the two gates


def gate_a_pre_construction(
    args: argparse.Namespace,
    *,
    stage: str,
    launch_contract_path: str | Path,
    stage_authorization_path: str | Path,
    exact_plan_path: str | Path,
    pilot_acceptance_path: str | Path,
    observed_runtime: Mapping[str, Any],
    cwd: str | None = None,
) -> dict[str, Any]:
    """Gate A: everything checkable BEFORE a model, optimizer, sampler or dataset exists."""
    require(stage in STAGE_ORDER, f"unknown stage {stage!r}")
    failures: list[str] = []

    resolved_cwd = str(Path(cwd if cwd is not None else os.getcwd()).resolve())
    if resolved_cwd != CANONICAL_CWD:
        failures.append(f"canonical_cwd: expected {CANONICAL_CWD}, got {resolved_cwd}")

    # 1. authenticate the launch-contract artifact from its actual bytes
    contract_artifact = load_launch_contract_artifact(launch_contract_path)

    # 2. authorization, from bytes
    auth_path = Path(stage_authorization_path)
    if not auth_path.is_file():
        raise LaunchContractError(f"stage authorization not found: {auth_path}")
    auth_bytes = auth_path.read_bytes()
    authorization = json.loads(auth_bytes.decode("utf-8"))
    authorization_sha = _sha256_bytes(auth_bytes)

    # 3. accepted immutable inputs, from bytes
    plan_sha = file_sha256(exact_plan_path)
    acceptance_sha = file_sha256(pilot_acceptance_path)
    if plan_sha != EXACT_RUN_PLAN_SHA256:
        failures.append(f"exact_run_plan_sha256: expected {EXACT_RUN_PLAN_SHA256}, got {plan_sha}")
    if acceptance_sha != PILOT_OWNER_ACCEPTANCE_SHA256:
        failures.append("pilot_owner_acceptance_sha256_mismatch")

    # 4. the authorization must agree with the artifact it claims to authorize
    if authorization.get("launch_contract_sha256") != contract_artifact["sha256"]:
        failures.append("authorization_launch_contract_sha256_mismatch")

    # 5. every governed CLI value
    failures.extend(validate_governed_args(args, stage=stage))
    failures.extend(validate_special_token_binding(args))
    failures.extend(validate_diagnostic_fields(args))
    failures.extend(validate_legacy_sampler_seed(args, stage))
    failures.extend(
        validate_num_workers_binding(
            args, (authorization.get("training_runtime") or {}).get("num_workers")
        )
    )
    failures.extend(validate_resume_binding(args, authorization.get("resume")))

    # 6. authorization vs observed runtime
    scope = "STAGE_N" if stage == "stage_a" else "STAGE_O"
    repo = observed_repository()
    observed = {
        "branch": repo["branch"],
        "head": repo["head"],
        "trainer_execution_bundle_sha256": trainer_execution_bundle_sha256(),
        "launch_contract_sha256": contract_artifact["sha256"],
        "exact_run_plan_sha256": plan_sha,
        "pilot_owner_acceptance_sha256": acceptance_sha,
        "output_root": (
            str(Path(str(getattr(args, "out_dir", ""))).resolve())
            if getattr(args, "out_dir", "")
            else None
        ),
        "canonical_cwd": resolved_cwd,
        "training_runtime": dict(observed_runtime),
    }
    verdict = validate_authorization(authorization, requested_scope=scope, observed=observed)
    failures.extend(verdict["failures"])

    # 7. Stage O additionally requires the accepted Stage-N chain
    if scope == "STAGE_O":
        failures.extend(
            validate_stage_o_chain(authorization, observed_runtime=observed_runtime)["failures"]
        )

    failures = list(dict.fromkeys(failures))
    require(
        not failures,
        f"Gate A refused the governed launch under {CONTRACT_VERSION}:\n  - "
        + "\n  - ".join(failures),
    )
    return {
        "gate": "A",
        "stage": stage,
        "scope": scope,
        "launch_contract_path": contract_artifact["path"],
        "launch_contract_sha256": contract_artifact["sha256"],
        "stage_authorization_path": str(auth_path),
        "stage_authorization_sha256": authorization_sha,
        "exact_plan_path": str(exact_plan_path),
        "exact_run_plan_sha256": plan_sha,
        "pilot_acceptance_path": str(pilot_acceptance_path),
        "pilot_owner_acceptance_sha256": acceptance_sha,
        "trainer_head": repo["head"],
        "trainer_branch": repo["branch"],
        "trainer_execution_bundle_sha256": observed["trainer_execution_bundle_sha256"],
        "resume": dict(authorization.get("resume") or {}),
        "num_workers": (authorization.get("training_runtime") or {}).get("num_workers"),
        "runtime": dict(observed_runtime),
        "authorization": authorization,
        "passed": True,
    }


def gate_b_post_construction(
    model: Any,
    optimizer: Any,
    *,
    expected_parameter_count: int = MODEL_PARAMETER_COUNT,
) -> dict[str, Any]:
    """Gate B: the ACTUAL constructed model and optimizer, before any forward or update."""
    failures: list[str] = []

    unique = {id(p): p for p in model.parameters()}
    actual = int(sum(p.numel() for p in unique.values()))
    if actual != expected_parameter_count:
        failures.append(f"model parameter count: expected {expected_parameter_count}, got {actual}")

    cfg = getattr(model, "cfg", None) or getattr(model, "config", None)
    architecture: dict[str, Any] = {}
    if cfg is not None:
        for field, want in MODEL_CONTRACT.items():
            attr = "max_seq_len" if field == "seq_len" else field
            got = getattr(cfg, attr, _MISSING)
            architecture[field] = got if not isinstance(got, _Missing) else None
            if isinstance(got, _Missing):
                failures.append(f"model config missing {attr}")
            elif got != want:
                failures.append(f"model {field}: expected {want}, got {got}")

    tok_emb = getattr(model, "tok_emb", None)
    lm_head = getattr(model, "lm_head", None)
    tied = (
        tok_emb is not None
        and lm_head is not None
        and getattr(lm_head, "weight", None) is getattr(tok_emb, "weight", None)
    )
    if not tied:
        failures.append("tied embeddings are not realized on the constructed model")

    if optimizer is None:
        grouping = {
            "failures": ["governed run requires a constructed optimizer"],
            "group_roles": [],
            "membership": {},
        }
    else:
        grouping = verify_realized_optimizer(optimizer, model)
    failures.extend(grouping["failures"])

    failures = list(dict.fromkeys(failures))
    require(
        not failures,
        f"Gate B refused the governed run under {CONTRACT_VERSION} before any training "
        f"forward or optimizer update:\n  - " + "\n  - ".join(failures),
    )
    return {
        "gate": "B",
        "parameter_count": actual,
        "architecture": architecture,
        "tied_embeddings": tied,
        "optimizer_group_roles": grouping["group_roles"],
        "optimizer_membership_counts": {
            role: len(names) for role, names in grouping["membership"].items()
        },
        "realized_muon": realized_muon_contract(),
        "passed": True,
    }


# --------------------------------------------------------------- Parts 4-7: run contract

GOVERNED_RUN_CONTRACT_FILENAME = "GOVERNED_RUN_CONTRACT.json"
GOVERNED_CHECKPOINT_KIND = "PETITGPT_GOVERNED_V1"


def build_governed_run_contract(
    *,
    gate_a: Mapping[str, Any],
    gate_b: Mapping[str, Any],
    stage: str,
    sampler_identity: Mapping[str, Any] | None = None,
    compile_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The normalized governed run contract, built from real gate results."""
    require(stage in STAGE_ORDER, f"unknown stage {stage!r}")
    runtime = dict(gate_a["runtime"])
    return {
        "schema_version": RUN_CONTRACT_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "governed": True,
        "kind": GOVERNED_CHECKPOINT_KIND,
        "stage": stage,
        "scope": gate_a["scope"],
        "launch_contract_path": gate_a["launch_contract_path"],
        "launch_contract_sha256": gate_a["launch_contract_sha256"],
        "stage_authorization_path": gate_a["stage_authorization_path"],
        "stage_authorization_sha256": gate_a["stage_authorization_sha256"],
        "exact_plan_path": gate_a["exact_plan_path"],
        "exact_run_plan_sha256": gate_a["exact_run_plan_sha256"],
        "pilot_acceptance_path": gate_a["pilot_acceptance_path"],
        "pilot_owner_acceptance_sha256": gate_a["pilot_owner_acceptance_sha256"],
        "trainer_branch": gate_a["trainer_branch"],
        "trainer_head": gate_a["trainer_head"],
        "trainer_execution_bundle_sha256": gate_a["trainer_execution_bundle_sha256"],
        "model": {
            **dict(MODEL_CONTRACT),
            "parameter_count": gate_b["parameter_count"],
            "tied_embeddings": gate_b["tied_embeddings"],
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
            "compile": COMPILE,
            "precision": PRECISION,
            "warmup_steps": WARMUP_STEPS,
            "decay_fraction": DECAY_FRACTION,
            "lr_schedule": LR_SCHEDULE,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
        },
        "optimizer_realization": gate_b["realized_muon"],
        "seeds": dict(SEED_TUPLE),
        "active_stage_sampler_seed": stage_sampler_seed(stage),
        "special_token_ids": dict(SPECIAL_TOKEN_IDS),
        "evaluation_policy": {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in EVALUATION_POLICY.items()
        },
        "checkpoint_policy": {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in CHECKPOINT_POLICY.items()
        },
        "canonical_cwd": CANONICAL_CWD,
        "num_workers": gate_a["num_workers"],
        "resume": dict(gate_a["resume"]),
        "runtime_fingerprint": runtime,
        "gpu_uuid": runtime.get("gpu_uuid"),
        "gpu_pci_bus_id": runtime.get("gpu_pci_bus_id"),
        "stage_start_step": STAGE_BOUNDARIES[stage]["start_step"],
        "stage_stop_step": STAGE_BOUNDARIES[stage]["stop_step"],
        "optimizer_reset_at_a_b": OPTIMIZER_RESET_AT_A_B,
        "scheduler_reset_at_a_b": SCHEDULER_RESET_AT_A_B,
        "sampler_identity": dict(sampler_identity) if sampler_identity is not None else None,
        "compile_evidence": dict(compile_evidence) if compile_evidence is not None else None,
        "compile_evidence_sha256": (
            compile_evidence.get("compile_evidence_sha256")
            if compile_evidence is not None
            else None
        ),
        "completed_evaluation_milestones": [],
        "completed_checkpoint_milestones": [],
    }


# Fields a resume may never differ on. `sampler_identity` moves legitimately as training
# advances, so it is validated separately against the stage/range/cursor rules.
GOVERNED_IMMUTABLE_FIELDS = (
    "schema_version",
    "contract_version",
    "kind",
    "stage",
    "scope",
    "launch_contract_sha256",
    "stage_authorization_sha256",
    "exact_run_plan_sha256",
    "pilot_owner_acceptance_sha256",
    "trainer_branch",
    "trainer_head",
    "trainer_execution_bundle_sha256",
    "model",
    "training",
    "optimizer_realization",
    "seeds",
    "active_stage_sampler_seed",
    "special_token_ids",
    "evaluation_policy",
    "checkpoint_policy",
    "canonical_cwd",
    "num_workers",
    "gpu_uuid",
    "gpu_pci_bus_id",
    "stage_start_step",
    "stage_stop_step",
)


def governed_digest(contract: Mapping[str, Any]) -> str:
    """Digest over the immutable governed identity only."""
    return _sha256_bytes(
        canonical_json_bytes({k: contract.get(k) for k in GOVERNED_IMMUTABLE_FIELDS})
    )


def publish_governed_run_contract(out_dir: str | Path, contract: Mapping[str, Any]) -> dict:
    """Atomically publish the finalized contract BEFORE the first optimizer update."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / GOVERNED_RUN_CONTRACT_FILENAME
    require(
        not path.exists(),
        f"{GOVERNED_RUN_CONTRACT_FILENAME} already exists at {path}; a governed run contract "
        "is published once and a rerun requires a new authorized output root",
    )
    payload = dict(contract)
    payload["governed_run_contract_sha256"] = governed_digest(contract)
    body = canonical_json_bytes(payload)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "wb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return {
        "path": str(path),
        "governed_run_contract_sha256": payload["governed_run_contract_sha256"],
        "file_sha256": _sha256_bytes(body),
        "atomic": True,
    }


def sampler_identity_document(stage: str, sampler: Any) -> dict[str, Any]:
    """Record the ACTIVE stage sampler seed and its exact permutation position."""
    require(stage in STAGE_ORDER, f"unknown stage {stage!r}")
    seed = stage_sampler_seed(stage)
    start = int(getattr(sampler, "start_position", 0))
    stop = int(getattr(sampler, "end_position", 0))
    cursor = int(getattr(sampler, "committed_position", start))
    identity = {
        "stage": stage,
        "sampler_seed": seed,
        "range_start_position": start,
        "range_stop_position": stop,
        "cursor": cursor,
        "consumed": max(0, cursor - start),
        "remaining": max(0, stop - cursor),
    }
    identity["permutation_identity"] = _sha256_bytes(
        canonical_json_bytes({
            "stage": stage,
            "sampler_seed": seed,
            "range_start_position": start,
            "range_stop_position": stop,
        })
    )
    return identity


def validate_same_stage_resume(saved: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    """Same-stage resume validates range_start_position as well as seed/permutation/cursor."""
    failures: list[str] = []
    for field in (
        "stage",
        "sampler_seed",
        "permutation_identity",
        "range_start_position",
        "range_stop_position",
    ):
        if saved.get(field) != current.get(field):
            failures.append(
                f"sampler_resume_mismatch:{field}: checkpoint {saved.get(field)!r} != "
                f"current {current.get(field)!r}"
            )
    saved_cursor = int(saved.get("cursor", -1))
    start = int(saved.get("range_start_position", 0))
    stop = int(saved.get("range_stop_position", 0))
    if not start <= saved_cursor <= stop:
        failures.append(
            f"sampler cursor {saved_cursor} outside its committed range [{start}, {stop}]"
        )
    return failures


def validate_stage_a_to_b_transition(
    stage_a_checkpoint_contract: Mapping[str, Any],
    *,
    plan_boundary_step: int = STAGE_A_STOP_STEP,
) -> list[str]:
    """A -> B is legal only from a complete Stage-A endpoint at the exact plan boundary."""
    failures: list[str] = []
    if stage_a_checkpoint_contract.get("stage") != "stage_a":
        failures.append(
            f"stage_a_to_b: source checkpoint stage is "
            f"{stage_a_checkpoint_contract.get('stage')!r}, expected 'stage_a'"
        )
    saved_seed = stage_a_checkpoint_contract.get("active_stage_sampler_seed")
    if saved_seed != STAGE_A_SAMPLER_SEED:
        failures.append(
            f"stage_a_to_b: saved Stage-A sampler seed {saved_seed!r} != {STAGE_A_SAMPLER_SEED}"
        )
    identity = stage_a_checkpoint_contract.get("sampler_identity") or {}
    if identity.get("sampler_seed") != STAGE_A_SAMPLER_SEED:
        failures.append("stage_a_to_b: sampler_identity does not carry the Stage-A seed")
    cursor = int(identity.get("cursor", -1))
    stop = int(identity.get("range_stop_position", -2))
    if cursor != stop:
        failures.append(
            f"stage_a_to_b: Stage-A consumed range incomplete (cursor {cursor} != stop {stop})"
        )
    saved_step = int(stage_a_checkpoint_contract.get("global_step", -1))
    if saved_step not in (-1, int(plan_boundary_step)):
        failures.append(
            f"stage_a_to_b: source cursor is at step {saved_step}, not the plan boundary "
            f"{plan_boundary_step}"
        )
    return failures


def is_governed_checkpoint(ckpt: Mapping[str, Any]) -> bool:
    contract = ckpt.get("governed_run_contract")
    return isinstance(contract, Mapping) and contract.get("kind") == GOVERNED_CHECKPOINT_KIND


def validate_governed_checkpoint_before_restore(
    ckpt: Mapping[str, Any],
    current_contract: Mapping[str, Any],
    *,
    expected_resume: Mapping[str, Any] | None = None,
    current_sampler_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate metadata BEFORE any executable state is restored.

    Runs before ``model.load_state_dict``, ``optimizer.load_state_dict``, scaler restore and
    RNG restore, so a mismatched checkpoint can never partially mutate the process.
    """
    failures: list[str] = []
    if not is_governed_checkpoint(ckpt):
        return {
            "compatible": False,
            "failures": ["ungoverned_checkpoint_cannot_resume_a_governed_run"],
        }
    saved = ckpt["governed_run_contract"]

    # A governed checkpoint may never claim compile=true without realized evidence.
    if bool((saved.get("training") or {}).get("compile")):
        saved_evidence = saved.get("compile_evidence") or {}
        if not saved_evidence.get("compile_realized"):
            failures.append("governed_checkpoint_claims_compile_true_without_realized_evidence")

    for field in GOVERNED_IMMUTABLE_FIELDS:
        if saved.get(field) != current_contract.get(field):
            failures.append(f"governed_resume_mismatch:{field}")

    saved_digest = ckpt.get("governed_run_contract_sha256") or saved.get(
        "governed_run_contract_sha256"
    )
    recomputed = governed_digest(saved)
    if saved_digest != recomputed:
        failures.append("governed_run_contract_digest_does_not_match_its_own_document")
    if recomputed != governed_digest(current_contract):
        failures.append("governed_run_contract_digest_mismatch")

    if expected_resume is not None:
        want_step = expected_resume.get("expected_step")
        if want_step is not None and int(ckpt.get("global_step", -1)) != int(want_step):
            failures.append(
                f"resume_step_mismatch: checkpoint at {ckpt.get('global_step')}, "
                f"authorization expects {want_step}"
            )
        want_stage = expected_resume.get("stage")
        if want_stage is not None and saved.get("stage") != want_stage:
            failures.append(f"resume_stage_mismatch: expected {want_stage}")
        want_digest = expected_resume.get("governed_run_contract_sha256")
        if want_digest is not None and recomputed != want_digest:
            failures.append("resume_governed_run_contract_sha256_mismatch")

    if current_sampler_identity is not None:
        failures.extend(
            validate_same_stage_resume(
                saved.get("sampler_identity") or {}, current_sampler_identity
            )
        )

    failures = list(dict.fromkeys(failures))
    return {
        "compatible": not failures,
        "failures": failures,
        "checkpoint_governed_run_contract_sha256": recomputed,
        "current_governed_run_contract_sha256": governed_digest(current_contract),
    }


# --------------------------------------------------------------- Part 9: Stage-N/O chain

STAGE_N_RESULT_SCHEMA = "petitgpt-stage-n-result-v1"
STAGE_N_RESULT_REQUIRED_FIELDS = (
    "schema_version",
    "status",
    "stage_authorization_sha256",
    "launch_contract_sha256",
    "exact_run_plan_sha256",
    "pilot_owner_acceptance_sha256",
    "trainer_head",
    "trainer_execution_bundle_sha256",
    "governed_run_contract_sha256",
    "final_checkpoint_path",
    "final_checkpoint_sha256",
    "final_checkpoint_step",
    "runtime_fingerprint",
    "gpu_uuid",
    "gpu_pci_bus_id",
    "num_workers",
    "smoke_results",
    "resume_results",
)

# A Stage-O authorization must carry the accepted Stage-N chain. Without these it fails.
STAGE_O_REQUIRED_CHAIN_FIELDS = (
    "accepted_stage_n_result_path",
    "accepted_stage_n_result_sha256",
    "stage_n_owner_acceptance_path",
    "stage_n_owner_acceptance_sha256",
    "stage_n_authorization_sha256",
    "stage_n_governed_run_contract_sha256",
    "stage_n_runtime_fingerprint",
    "stage_n_gpu_uuid",
    "stage_n_gpu_pci_bus_id",
    "stage_n_trainer_head",
    "stage_n_trainer_execution_bundle_sha256",
    "stage_n_exact_run_plan_sha256",
)

# Runtime fields Stage O must match against the ACCEPTED Stage-N result, not merely against
# its own authorization. Changing both the authorization and the runtime cannot evade this.
STAGE_N_O_RUNTIME_COMPARISON_FIELDS = (
    "gpu_uuid",
    "gpu_pci_bus_id",
    "gpu_name",
    "num_workers",
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


def stage_n_result_document(
    *,
    governed_run_contract: Mapping[str, Any],
    final_checkpoint_path: str,
    final_checkpoint_sha256: str,
    final_checkpoint_step: int,
    smoke_results: Mapping[str, Any],
    resume_results: Mapping[str, Any],
) -> dict[str, Any]:
    """The machine-readable Stage-N result a future Stage-O authorization must bind."""
    runtime = dict(governed_run_contract.get("runtime_fingerprint") or {})
    return {
        "schema_version": STAGE_N_RESULT_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "status": "COMPLETE",
        "stage": "stage_a",
        "scope": "STAGE_N",
        "stage_authorization_sha256": governed_run_contract["stage_authorization_sha256"],
        "launch_contract_sha256": governed_run_contract["launch_contract_sha256"],
        "exact_run_plan_sha256": governed_run_contract["exact_run_plan_sha256"],
        "pilot_owner_acceptance_sha256": governed_run_contract["pilot_owner_acceptance_sha256"],
        "trainer_head": governed_run_contract["trainer_head"],
        "trainer_execution_bundle_sha256": governed_run_contract["trainer_execution_bundle_sha256"],
        "governed_run_contract_sha256": governed_digest(governed_run_contract),
        "final_checkpoint_path": str(final_checkpoint_path),
        "final_checkpoint_sha256": str(final_checkpoint_sha256),
        "final_checkpoint_step": int(final_checkpoint_step),
        "runtime_fingerprint": runtime,
        "gpu_uuid": runtime.get("gpu_uuid"),
        "gpu_pci_bus_id": runtime.get("gpu_pci_bus_id"),
        "num_workers": governed_run_contract.get("num_workers"),
        "smoke_results": dict(smoke_results),
        "resume_results": dict(resume_results),
    }


def validate_stage_n_result(document: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(document, Mapping):
        return ["stage_n_result_missing_or_malformed"]
    failures = [
        f"stage_n_result_missing_field:{field}"
        for field in STAGE_N_RESULT_REQUIRED_FIELDS
        if field not in document
    ]
    if document.get("schema_version") != STAGE_N_RESULT_SCHEMA:
        failures.append("stage_n_result_schema_mismatch")
    if document.get("status") != "COMPLETE":
        failures.append(f"stage_n_result_status_not_complete:{document.get('status')!r}")
    return failures


def validate_stage_o_chain(
    authorization: Mapping[str, Any],
    *,
    observed_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Stage O must bind, load and match the ACCEPTED Stage-N chain.

    Loading the accepted Stage-N result from disk is what stops a rewritten Stage-O
    authorization plus a changed runtime from agreeing with each other and evading comparison.
    """
    failures: list[str] = []
    chain = authorization.get("stage_n_chain")
    if not isinstance(chain, Mapping):
        return {
            "valid": False,
            "failures": ["stage_o_authorization_missing_stage_n_chain"],
        }
    for field in STAGE_O_REQUIRED_CHAIN_FIELDS:
        if chain.get(field) in (None, ""):
            failures.append(f"stage_o_chain_missing_field:{field}")
    if failures:
        return {"valid": False, "failures": failures}

    # The accepted Stage-N result, from bytes.
    result_path = Path(str(chain["accepted_stage_n_result_path"]))
    if not result_path.is_file():
        return {
            "valid": False,
            "failures": [f"accepted_stage_n_result_not_found:{result_path}"],
        }
    result_bytes = result_path.read_bytes()
    result_sha = _sha256_bytes(result_bytes)
    if result_sha != chain["accepted_stage_n_result_sha256"]:
        failures.append("accepted_stage_n_result_sha256_mismatch")
    stage_n = json.loads(result_bytes.decode("utf-8"))
    failures.extend(validate_stage_n_result(stage_n))

    # The owner acceptance of that exact result, from bytes.
    acceptance_path = Path(str(chain["stage_n_owner_acceptance_path"]))
    if not acceptance_path.is_file():
        failures.append(f"stage_n_owner_acceptance_not_found:{acceptance_path}")
    else:
        acceptance_bytes = acceptance_path.read_bytes()
        if _sha256_bytes(acceptance_bytes) != chain["stage_n_owner_acceptance_sha256"]:
            failures.append("stage_n_owner_acceptance_sha256_mismatch")
        acceptance = json.loads(acceptance_bytes.decode("utf-8"))
        if acceptance.get("accepted_stage_n_result_sha256") != result_sha:
            failures.append("stage_n_owner_acceptance_does_not_accept_this_result")
        if acceptance.get("stage_n_result_owner_verdict") != "ACCEPTED":
            failures.append("stage_n_result_not_owner_accepted")

    # The chain's claims must equal the accepted result itself.
    for chain_field, result_field in (
        ("stage_n_authorization_sha256", "stage_authorization_sha256"),
        ("stage_n_governed_run_contract_sha256", "governed_run_contract_sha256"),
        ("stage_n_gpu_uuid", "gpu_uuid"),
        ("stage_n_gpu_pci_bus_id", "gpu_pci_bus_id"),
        ("stage_n_trainer_head", "trainer_head"),
        ("stage_n_trainer_execution_bundle_sha256", "trainer_execution_bundle_sha256"),
        ("stage_n_exact_run_plan_sha256", "exact_run_plan_sha256"),
    ):
        if chain.get(chain_field) != stage_n.get(result_field):
            failures.append(f"stage_o_chain_contradicts_accepted_stage_n_result:{chain_field}")

    # The CURRENTLY observed runtime must equal the runtime Stage N actually ran on.
    accepted_runtime = dict(stage_n.get("runtime_fingerprint") or {})
    for field in STAGE_N_O_RUNTIME_COMPARISON_FIELDS:
        if field in observed_runtime and accepted_runtime.get(field) != observed_runtime.get(field):
            failures.append(
                f"stage_o_runtime_differs_from_accepted_stage_n:{field}: "
                f"stage_n={accepted_runtime.get(field)!r}, observed={observed_runtime.get(field)!r}"
            )

    failures = list(dict.fromkeys(failures))
    return {
        "valid": not failures,
        "failures": failures,
        "accepted_stage_n_result_sha256": result_sha,
        "requires_new_stage_n": bool([
            f for f in failures if f.startswith("stage_o_runtime_differs_from_accepted_stage_n")
        ]),
    }
