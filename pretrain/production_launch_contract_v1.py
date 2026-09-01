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
import contextlib
import hashlib
import json
import math
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
    # R2 Part 1: authorization-bound to an exact path that must resolve INSIDE the
    # authorized governed output root. Not arbitrary even when sampling is disabled.
    "samples_dir": {
        "class": LAUNCH_AUTHORIZATION_BOUND,
        "value": None,
        "affects": ("checkpointing", "runtime identity"),
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
    # R2 Part 1: benchmark evaluation is DISABLED, so any non-empty value fails Gate A.
    "bench_eval_out_dir": {
        "class": FORBIDDEN_OR_UNSET,
        "value": "",
        "affects": ("evaluation",),
    },
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
    foreign_count = 0
    if model is not None:
        # R2 Part 2: EXACT membership. Unknown optimizer parameters are never filtered out
        # before comparison -- a foreign Parameter must surface as a failure, not vanish.
        by_id = {id(p): n for n, p in model.named_parameters() if p.requires_grad}
        all_model_ids = {id(p) for p in model.parameters()}
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
            names: list[str] = []
            for param in group["params"]:
                pid = id(param)
                seen.append(pid)
                if pid in by_id:
                    names.append(by_id[pid])
                else:
                    foreign_count += 1
                    names.append(f"<foreign:{pid}>")
            membership[role] = sorted(names)

        if foreign_count:
            failures.append(
                f"{foreign_count} optimizer parameter(s) are not trainable model parameters"
            )
        duplicates = len(seen) - len(set(seen))
        if duplicates:
            failures.append(f"{duplicates} parameter(s) appear in more than one optimizer group")
        missing = sorted(set(by_id) - set(seen))
        if missing:
            failures.append(f"{len(missing)} trainable parameter(s) are in no optimizer group")
        untracked = sorted(all_model_ids - set(by_id) - set(seen))
        if untracked:
            failures.append(
                f"{len(untracked)} model parameter(s) are neither trainable nor optimized"
            )
        for role in OPTIMIZER_GROUP_ROLES:
            if membership.get(role, []) != sorted(expected[role]):
                failures.append(f"{role} membership differs from the frozen grouping rule")

    return {
        "group_roles": roles,
        "membership": membership,
        "foreign_parameter_count": foreign_count,
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
    "allowed_samples_dir",
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

# A resume authorization is a statement about an immutable SOURCE invocation, not merely
# permission to open a checkpoint path. Gate A enforces these fields for every governed
# RESUME_EXACT_CHECKPOINT launch. They intentionally remain separate from the command-line
# binding tuple above so callers can distinguish CLI shape validation from the stronger
# on-disk authority-chain verification.
RESUME_SOURCE_AUTHORITY_REQUIRED_FIELDS = (
    "source_stage_authorization_path",
    "source_stage_authorization_sha256",
    "source_invocation_run_contract_path",
    "source_invocation_run_contract_sha256",
    "source_base_governed_identity_digest",
    "source_checkpoint_path",
    "source_checkpoint_sha256",
    "source_checkpoint_step",
    "source_checkpoint_stage",
    "source_active_stage",
    "source_sampler_seed",
    "source_permutation_identity",
    "source_range_start_position",
    "source_invocation_range_start_position",
    "source_range_stop_position",
    "source_cursor",
)

A_TO_B_SOURCE_REQUIRED_FIELDS = RESUME_SOURCE_AUTHORITY_REQUIRED_FIELDS

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
        "allowed_samples_dir": None,
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
    document = {
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
    # R2 Part 3: the COMPLETE runtime document gets one cryptographic identity, so the
    # immutable governed identity binds the whole fingerprint rather than a chosen subset.
    document["runtime_fingerprint_sha256"] = runtime_fingerprint_sha256(document)
    return document


def runtime_fingerprint_sha256(document: Mapping[str, Any]) -> str:
    """SHA-256 over the complete runtime document, excluding its own self-hash."""
    payload = {k: v for k, v in document.items() if k != "runtime_fingerprint_sha256"}
    return _sha256_bytes(canonical_json_bytes(payload))


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


def cuda_logical_device_identity(index: int = 0) -> dict[str, Any]:
    """The physical identity of Torch/CUDA LOGICAL device ``index``, from CUDA itself.

    R2 Part 16: a numeric ``CUDA_VISIBLE_DEVICES`` ordinal is NOT an NVML index, and mapping
    one to the other because both are numeric is wrong. CUDA already knows which physical
    device it selected, so the UUID and PCI identity are read from
    ``torch.cuda.get_device_properties`` and only THEN matched against NVML.
    """
    import torch

    props = torch.cuda.get_device_properties(index)
    raw_uuid = str(getattr(props, "uuid", "") or "")
    uuid = raw_uuid if raw_uuid.startswith("GPU-") else (f"GPU-{raw_uuid}" if raw_uuid else "")
    domain = int(getattr(props, "pci_domain_id", 0))
    bus = int(getattr(props, "pci_bus_id", 0))
    device = int(getattr(props, "pci_device_id", 0))
    return {
        "logical_index": int(index),
        "gpu_uuid": uuid,
        "gpu_pci_bus_id": f"{domain:08X}:{bus:02X}:{device:02X}.0",
        "gpu_name": str(props.name),
        "total_vram_bytes": int(props.total_memory),
        "compute_capability": f"{props.major}.{props.minor}",
        "source": "torch.cuda.get_device_properties",
    }


def resolve_selected_gpu_identity(
    *,
    cuda_visible_devices: str | None = None,
    records: Sequence[Mapping[str, Any]] | None = None,
    torch_device_name: str | None = None,
    selected_index: int = 0,
    cuda_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Match CUDA's own logical-device identity to exactly one physical NVML record.

    The mapping is driven by the CUDA-reported UUID/PCI of logical device 0, never by
    interpreting a ``CUDA_VISIBLE_DEVICES`` ordinal as an NVML index. It fails closed when
    the identity is ambiguous, unmatched, or contradicted by NVML.
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

    identity = dict(cuda_identity) if cuda_identity is not None else None
    if identity is None:
        try:
            identity = cuda_logical_device_identity(selected_index)
        except Exception as exc:  # noqa: BLE001
            return {
                "resolved": False,
                "failures": [f"cuda_logical_device_identity_unavailable:{exc!r}"],
                "cuda_visible_devices": raw,
            }

    cvd_form = (
        "unset"
        if raw is None or not str(raw).strip()
        else (
            "index"
            if all(t.strip().isdigit() for t in str(raw).split(",") if t.strip())
            else "uuid"
        )
    )
    visible_count = (
        len(physical) if cvd_form == "unset" else len([t for t in str(raw).split(",") if t.strip()])
    )

    # Match NVML by the CUDA-reported UUID first, then by PCI identity. Never by ordinal.
    cuda_uuid = str(identity.get("gpu_uuid") or "")
    cuda_pci = str(identity.get("gpu_pci_bus_id") or "")
    by_uuid = [r for r in physical if cuda_uuid and r["uuid"] == cuda_uuid]
    if not by_uuid and cuda_uuid:
        by_uuid = [r for r in physical if r["uuid"].startswith(cuda_uuid)]
    by_pci = [r for r in physical if cuda_pci and r["pci_bus_id"].upper() == cuda_pci.upper()]

    # R3 Part 16: when BOTH CUDA-derived identities exist they must resolve to the SAME
    # single physical device. `by_uuid or by_pci` was a fallback that let an unknown UUID be
    # rescued by a matching PCI (or vice versa); that is now a hard failure.
    have_uuid, have_pci = bool(cuda_uuid), bool(cuda_pci)
    if have_uuid and have_pci:
        if len(by_uuid) != 1 or len(by_pci) != 1:
            failures.append(
                "cuda_logical_device_unmatched_in_nvml"
                if not (by_uuid and by_pci)
                else f"cuda_logical_device_ambiguous_in_nvml:uuid={len(by_uuid)},pci={len(by_pci)}"
            )
            matched: list[dict[str, Any]] = []
        elif by_uuid[0]["uuid"] != by_pci[0]["uuid"]:
            failures.append(
                f"cuda_uuid_and_pci_resolve_to_different_devices:"
                f"uuid->{by_uuid[0]['uuid']}, pci->{by_pci[0]['uuid']}"
            )
            matched = []
        else:
            matched = by_uuid
    else:
        matched = by_uuid or by_pci
        if len(matched) != 1:
            failures.append(
                "cuda_logical_device_unmatched_in_nvml"
                if not matched
                else f"cuda_logical_device_ambiguous_in_nvml:{len(matched)}"
            )

    # R4 Part 19: a UUID-form CUDA_VISIBLE_DEVICES value is an identity assertion, not a
    # convenient prefix selector. Governed execution therefore requires one complete UUID
    # token which is exposed independently by CUDA logical device 0 and names exactly one
    # physical record. In particular, values such as ``GPU-`` (or any other shortened CUDA
    # prefix) may be accepted by the CUDA runtime, but are not sufficient evidence here.
    if cvd_form == "uuid":
        tokens = [t.strip() for t in str(raw).split(",") if t.strip()]
        if len(tokens) != 1:
            failures.append(f"cuda_visible_devices_uuid_token_count_not_exactly_1:{len(tokens)}")
        elif not have_uuid:
            failures.append("cuda_visible_devices_uuid_unverifiable_without_cuda_uuid")
        else:
            token = tokens[0]
            exact_physical = [r for r in physical if r["uuid"] == token]
            if not token.startswith("GPU-") or len(token) <= len("GPU-"):
                failures.append(f"cuda_visible_devices_uuid_not_full_identity:{token!r}")
            if len(exact_physical) != 1:
                failures.append(
                    "cuda_visible_devices_uuid_not_unique_physical_identity:"
                    f"cvd={token},matches={len(exact_physical)}"
                )
            if cuda_uuid != token:
                failures.append(
                    f"cuda_visible_devices_uuid_contradicts_logical_device_0:"
                    f"cvd={token}, cuda={cuda_uuid}"
                )

    if failures:
        return {
            "resolved": False,
            "failures": failures,
            "cuda_visible_devices": raw,
            "mapping_form": cvd_form,
            "cuda_identity": identity,
            "physical_record_count": len(physical),
        }

    chosen = matched[0]
    if torch_device_name is not None and chosen["name"] != torch_device_name:
        failures.append(
            f"selected_device_name_inconsistent:nvml={chosen['name']!r},torch={torch_device_name!r}"
        )
    if chosen["pci_bus_id"].upper() != cuda_pci.upper():
        failures.append(f"cuda_pci_disagrees_with_nvml:cuda={cuda_pci},nvml={chosen['pci_bus_id']}")
    if visible_count != REQUIRED_TRAINING_DEVICE_COUNT:
        failures.append(f"visible_device_count_not_exactly_1:{visible_count}")

    return {
        "resolved": not failures,
        "failures": failures,
        "cuda_visible_devices": raw,
        "mapping_form": cvd_form,
        "mapping_method": "cuda_logical_device_identity_matched_to_nvml_by_uuid_then_pci",
        "physical_record_count": len(physical),
        "visible_device_count": visible_count,
        "selected_logical_index": selected_index,
        "selected_physical_index": chosen["physical_index"],
        "cuda_identity": identity,
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
    args: argparse.Namespace,
    resume_binding: Mapping[str, Any] | None,
    *,
    require_source_authority: bool = False,
    transition: str | None = None,
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

    if mode == "RESUME_EXACT_CHECKPOINT" and require_source_authority:
        required_source_fields = (
            A_TO_B_SOURCE_REQUIRED_FIELDS
            if transition == "A_TO_B"
            else RESUME_SOURCE_AUTHORITY_REQUIRED_FIELDS
        )
        for field in required_source_fields:
            if resume_binding.get(field) in (None, ""):
                failures.append(f"resume_binding_missing_source_field:{field}")

        for field in (
            "source_stage_authorization_sha256",
            "source_invocation_run_contract_sha256",
            "source_base_governed_identity_digest",
            "source_checkpoint_sha256",
        ):
            value = resume_binding.get(field)
            if value not in (None, "") and not _is_sha256(value):
                failures.append(f"resume_binding_invalid_source_sha256:{field}")

        for field in (
            "source_stage_authorization_path",
            "source_invocation_run_contract_path",
            "source_checkpoint_path",
        ):
            value = resume_binding.get(field)
            if value not in (None, "") and not Path(str(value)).is_absolute():
                failures.append(f"resume_binding_source_path_not_absolute:{field}")

        for field in (
            "source_checkpoint_step",
            "source_sampler_seed",
            "source_range_start_position",
            "source_invocation_range_start_position",
            "source_range_stop_position",
            "source_cursor",
        ):
            value = resume_binding.get(field)
            if value not in (None, "") and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                failures.append(f"resume_binding_invalid_source_integer:{field}")

        for current_field, source_field in (
            ("checkpoint_path", "source_checkpoint_path"),
            ("checkpoint_sha256", "source_checkpoint_sha256"),
            ("expected_step", "source_checkpoint_step"),
            ("stage", "source_checkpoint_stage"),
        ):
            current = resume_binding.get(current_field)
            source = resume_binding.get(source_field)
            if current not in (None, "") and source not in (None, "") and current != source:
                failures.append(
                    f"resume_binding_source_alias_mismatch:{current_field}!={source_field}"
                )

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


def validate_samples_dir_binding(
    args: argparse.Namespace, authorized_samples_dir: Any, authorized_output_root: Any
) -> list[str]:
    """R2 Part 1: samples_dir is an exact authorized path INSIDE the governed output root."""
    if authorized_samples_dir in (None, ""):
        return ["samples_dir_not_bound_by_authorization"]
    actual = str(getattr(args, "samples_dir", "") or "").strip()
    if not actual:
        return ["samples_dir must be the exact contract-authorized value, not empty"]
    failures: list[str] = []
    resolved = Path(actual).expanduser().resolve()
    expected = Path(str(authorized_samples_dir)).expanduser().resolve()
    if resolved != expected:
        failures.append(f"samples_dir: expected {expected}, got {resolved}")
    if authorized_output_root not in (None, ""):
        root = Path(str(authorized_output_root)).expanduser().resolve()
        if resolved != root and root not in resolved.parents:
            failures.append(
                f"samples_dir must resolve inside the authorized output root {root}: {resolved}"
            )
    return failures


def validate_bench_eval_out_dir(args: argparse.Namespace) -> list[str]:
    """R3 Part 1: STRICT. Only None or the parser's canonical empty string is permitted.

    Whitespace is not normalized into permission: an explicitly supplied "   ", "\t" or
    "\n" is a supplied value and is forbidden, exactly like any other non-empty string.
    """
    value = getattr(args, "bench_eval_out_dir", "")
    if value is None or value == "":
        return []
    return [
        f"bench_eval_out_dir must be exactly None or the canonical empty string in governed "
        f"execution (benchmark evaluation is DISABLED), got {value!r}"
    ]


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
    finalize: bool = True,
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
    }
    if not finalize:
        evidence["failures"] = failures
        return evidence
    return finalize_compile_evidence(evidence, additional_failures=failures)


COMPILE_EVIDENCE_SELF_HASH_FIELD = "compile_evidence_sha256"
COMPILE_EVIDENCE_UNHASHED_FIELDS = (COMPILE_EVIDENCE_SELF_HASH_FIELD,)


def seal_compile_evidence(evidence: Mapping[str, Any]) -> str:
    """The one canonical compile-evidence self-hash convention.

    SHA-256 covers the canonical JSON of every field except the self-hash itself. Failures
    and the final verdict are load-bearing evidence: adding or removing either must
    invalidate the seal.
    """
    return _sha256_bytes(
        canonical_json_bytes({
            k: v for k, v in evidence.items() if k not in COMPILE_EVIDENCE_UNHASHED_FIELDS
        })
    )


COMPILE_REALIZATION_SUBFACTS = (
    "compile_requested",
    "invoked_compiled_callable",
    "realized_module_is_optimized_module",
    "compilation_materialized",
    "forward_invocations_match",
    "compiled_callable_is_training_callable",
)

CAUSAL_DIAGNOSTIC_CHECK_POS = 128
CAUSAL_DIAGNOSTIC_DELTA_POS = 8
CAUSAL_DIAGNOSTIC_SEQ_LEN = CAUSAL_DIAGNOSTIC_CHECK_POS + CAUSAL_DIAGNOSTIC_DELTA_POS + 1
CAUSAL_LEAK_MAX_ABS_TOLERANCE = 1.0e-5

COMPILE_PRODUCTION_REQUIRED_FACTS = (
    "probe_geometry",
    "probe_signature",
    "production_shape_probe",
    "isolated_cache",
    "cache_was_empty_before_realization",
    "precompile_causal_diagnostic",
    "fail_closed_stance",
    "post_realization_stance",
)


def _derived_compile_failures(evidence: Mapping[str, Any]) -> list[str]:
    """Derive compile validity from observations rather than a caller verdict Boolean."""
    missing = [
        field
        for field in (*COMPILE_REALIZATION_SUBFACTS, *COMPILE_PRODUCTION_REQUIRED_FACTS)
        if field not in evidence
    ]
    failures = [f"compile_evidence_missing_subfact:{field}" for field in missing]
    if missing:
        return failures

    if not bool(evidence.get("compile_requested")):
        failures.append("compile_was_not_requested")
    if not bool(evidence.get("invoked_compiled_callable")):
        failures.append("invoked_callable_is_not_the_compiled_object")
    if not bool(evidence.get("realized_module_is_optimized_module")):
        failures.append("realized_module_is_not_an_OptimizedModule")

    raw_materialized = bool(
        int(evidence.get("dynamo_unique_graphs", 0) or 0) > 0
        or int(evidence.get("inductor_artifact_count", 0) or 0) > 0
    )
    if bool(evidence.get("compilation_materialized")) != raw_materialized:
        failures.append("compilation_materialized_contradicts_graph_and_cache_facts")
    if not raw_materialized:
        failures.append("compile_never_materialized_lazily_eager_fallback")

    raw_invocations_match = int(evidence.get("forward_invocations", -1)) == int(
        evidence.get("expected_forward_invocations", -2)
    )
    if bool(evidence.get("forward_invocations_match")) != raw_invocations_match:
        failures.append("forward_invocations_match_contradicts_invocation_counts")
    if not raw_invocations_match:
        failures.append("forward_invocations_do_not_match_expected")

    if bool(evidence.get("recompile_limit_fallback_detected")):
        failures.append("recompile_limit_fallback_to_eager_detected")
    if bool(evidence.get("eager_fallback_occurred")):
        failures.append("eager_fallback_occurred")
    if not bool(evidence.get("compiled_callable_is_training_callable")):
        failures.append("compiled_callable_is_not_training_callable")

    geometry = evidence.get("probe_geometry")
    geometry_is_production = bool(
        isinstance(geometry, Mapping)
        and isinstance(geometry.get("micro_bsz"), int)
        and not isinstance(geometry.get("micro_bsz"), bool)
        and geometry.get("micro_bsz") == MICRO_BSZ
        and isinstance(geometry.get("seq_len"), int)
        and not isinstance(geometry.get("seq_len"), bool)
        and geometry.get("seq_len") == MODEL_CONTRACT["seq_len"]
    )
    if not geometry_is_production:
        failures.append("compile_probe_not_at_exact_production_geometry")
    if bool(evidence.get("production_shape_probe")) != geometry_is_production:
        failures.append("production_shape_probe_contradicts_probe_geometry")
    if not bool(evidence.get("production_shape_probe")):
        failures.append("compile_probe_not_at_production_shape")

    signature = evidence.get("probe_signature")
    if not isinstance(signature, Mapping):
        failures.append("compile_probe_signature_missing_or_malformed")
    else:
        exact_signature = bool(
            signature.get("device_type") == "cuda"
            and signature.get("module_training_mode") is True
            and signature.get("grad_enabled") is True
            and signature.get("autocast_enabled") is True
            and signature.get("autocast") == "bf16"
            and signature.get("autocast_dtype") == "torch.bfloat16"
            and signature.get("input_dtype") == "torch.int64"
            and signature.get("input_shape") == [MICRO_BSZ, MODEL_CONTRACT["seq_len"]]
            and signature.get("output_requires_grad") is True
            and signature.get("optimizer_step_taken") is False
        )
        if signature.get("grad_enabled") is not True:
            failures.append("compile_probe_was_not_grad_enabled")
        if signature.get("module_training_mode") is not True:
            failures.append("compile_probe_module_was_not_in_training_mode")
        if (
            signature.get("device_type") != "cuda"
            or signature.get("autocast_enabled") is not True
            or signature.get("autocast") != "bf16"
            or signature.get("autocast_dtype") != "torch.bfloat16"
        ):
            failures.append("compile_probe_did_not_observe_canonical_cuda_bf16_autocast")
        if signature.get("input_dtype") != "torch.int64" or signature.get("input_shape") != [
            MICRO_BSZ,
            MODEL_CONTRACT["seq_len"],
        ]:
            failures.append("compile_probe_input_signature_mismatch")
        if signature.get("output_requires_grad") is not True:
            failures.append("compile_probe_output_did_not_require_grad")
        if signature.get("optimizer_step_taken") is not False:
            failures.append("compile_probe_took_an_optimizer_step")
        if bool(signature.get("matches_training_forward_signature")) != exact_signature:
            failures.append("training_forward_signature_claim_contradicts_observations")
        if not exact_signature:
            failures.append("compile_probe_did_not_match_training_forward_signature")

    cache = evidence.get("isolated_cache")
    cache_valid = bool(
        isinstance(cache, Mapping)
        and cache.get("isolated") is True
        and cache.get("was_empty_before_realization") is True
        and isinstance(cache.get("cache_dir"), str)
        and bool(cache.get("cache_dir"))
        and isinstance(cache.get("triton_cache_dir"), str)
        and bool(cache.get("triton_cache_dir"))
        and evidence.get("inductor_cache_dir") == cache.get("cache_dir")
        and evidence.get("cache_was_empty_before_realization") is True
    )
    if not cache_valid:
        failures.append("inductor_cache_not_isolated_or_not_empty")

    causal = evidence.get("precompile_causal_diagnostic")
    if not isinstance(causal, Mapping):
        failures.append("precompile_causal_diagnostic_missing_or_malformed")
    else:
        difference = causal.get("max_abs_difference")
        tolerance = causal.get("max_abs_tolerance")
        numeric_difference = (
            isinstance(difference, (int, float))
            and not isinstance(difference, bool)
            and math.isfinite(float(difference))
        )
        within_tolerance = bool(
            numeric_difference
            and tolerance == CAUSAL_LEAK_MAX_ABS_TOLERANCE
            and float(difference) <= CAUSAL_LEAK_MAX_ABS_TOLERANCE
        )
        causal_valid = bool(
            causal.get("executed") is True
            and causal.get("used_uncompiled_base_model") is True
            and causal.get("executed_before_training_compile_realization") is True
            and causal.get("grad_enabled") is False
            and causal.get("input_shape") == [1, CAUSAL_DIAGNOSTIC_SEQ_LEN]
            and causal.get("check_pos") == CAUSAL_DIAGNOSTIC_CHECK_POS
            and causal.get("delta_pos") == CAUSAL_DIAGNOSTIC_DELTA_POS
            and causal.get("mode_before") == "train"
            and causal.get("mode_after") == "train"
            and causal.get("mode_restored") is True
            and causal.get("within_tolerance") is within_tolerance
            and within_tolerance
        )
        if causal.get("used_uncompiled_base_model") is not True:
            failures.append("causal_diagnostic_did_not_use_uncompiled_base_model")
        if causal.get("mode_restored") is not True or causal.get("mode_after") != causal.get(
            "mode_before"
        ):
            failures.append("causal_diagnostic_did_not_restore_training_mode")
        if not within_tolerance:
            failures.append("causal_diagnostic_exceeded_max_abs_tolerance")
        if not causal_valid:
            failures.append("precompile_causal_diagnostic_facts_invalid")

    fail_closed = evidence.get("fail_closed_stance")
    if not (
        isinstance(fail_closed, Mapping)
        and fail_closed.get("suppress_errors") is False
        and fail_closed.get("fail_on_recompile_limit_hit") is True
        and fail_closed.get("set_stance_available") is True
    ):
        failures.append("compile_fail_closed_configuration_not_verified")

    post_stance = evidence.get("post_realization_stance")
    if not (
        isinstance(post_stance, Mapping)
        and post_stance.get("armed") is True
        and post_stance.get("stance") == "fail_on_recompile"
    ):
        failures.append("fail_on_recompile_stance_not_armed")

    # Stable de-duplication makes the final sealed representation deterministic.
    return list(dict.fromkeys(failures))


def finalize_compile_evidence(
    observations: Mapping[str, Any], *, additional_failures: Sequence[str] = ()
) -> dict[str, Any]:
    """Build the immutable final evidence document and seal it exactly once.

    Callers must pass an unsealed observation document. Discovering another fact after this
    function returns requires rebuilding from observations, not mutating the sealed result.
    """
    require(
        COMPILE_EVIDENCE_SELF_HASH_FIELD not in observations,
        "compile evidence must be finalized from an unsealed observation document",
    )
    require(
        "verdict" not in observations and "compile_realized" not in observations,
        "compile verdict fields may only be populated by finalization",
    )
    evidence = dict(observations)
    observed = evidence.pop("failures", [])
    require(isinstance(observed, list), "compile evidence failures must be a list")
    derived = _derived_compile_failures(evidence)
    recorded = [str(value) for value in [*observed, *additional_failures]]
    failures = list(dict.fromkeys([*recorded, *derived]))
    success = not failures
    evidence["failures"] = failures
    evidence["verdict"] = "PASS" if success else "FAIL"
    evidence["compile_realized"] = success
    evidence[COMPILE_EVIDENCE_SELF_HASH_FIELD] = seal_compile_evidence(evidence)
    return evidence


def require_compile_realized(evidence: Mapping[str, Any]) -> None:
    """Fail closed. No governed optimizer update may precede realized compile evidence.

    R3 Part 18: the ``compile_realized`` verdict is not taken on trust. It is re-derived from
    the sub-facts recorded in the same document, and an eager fallback is rejected outright.
    A document asserting ``compile_realized`` while its own sub-facts say compile never
    materialized is a contradiction, and a contradiction must abort the run rather than be
    read as the answer it claims.
    """
    document_failures = verify_compile_evidence_document(evidence)
    require(
        not document_failures,
        "governed compile was not lazily realized: its evidence sub-facts failed "
        "sealed-document verification: " + ", ".join(document_failures),
    )
    derived_failures = _derived_compile_failures(evidence)
    recorded_failures = list(evidence.get("failures") or [])
    rederived = not derived_failures and not recorded_failures
    require(
        rederived and evidence.get("verdict") == "PASS" and bool(evidence.get("compile_realized")),
        "governed run requires compile=true to be lazily realized before the first "
        "optimizer update; observed failures: "
        + ", ".join([*recorded_failures, *derived_failures]),
    )


COMPILE_PROBE_MICRO_BSZ = MICRO_BSZ  # production shape, not batch 1
COMPILE_PROBE_SEQ_LEN = MODEL_CONTRACT["seq_len"]


def enforce_compile_fail_closed_stance() -> dict[str, Any]:
    """R2 Part 10: supported PyTorch 2.11 fail-closed compile configuration.

    ``suppress_errors=False`` makes a compile error raise instead of degrading to eager;
    ``fail_on_recompile_limit_hit=True`` turns a recompile-limit fallback into a hard error;
    ``torch.compiler.set_stance("fail_on_recompile")`` makes any later recompile a hard
    error. All three are supported public/stable configuration in this runtime.
    """
    import torch
    import torch._dynamo as dynamo

    applied: dict[str, Any] = {}
    require(
        hasattr(dynamo.config, "suppress_errors"),
        "governed compile requires dynamo.config.suppress_errors in this runtime",
    )
    dynamo.config.suppress_errors = False
    applied["suppress_errors"] = dynamo.config.suppress_errors

    if hasattr(dynamo.config, "fail_on_recompile_limit_hit"):
        dynamo.config.fail_on_recompile_limit_hit = True
        applied["fail_on_recompile_limit_hit"] = dynamo.config.fail_on_recompile_limit_hit

    stance_available = hasattr(torch.compiler, "set_stance")
    applied["set_stance_available"] = stance_available
    require(
        stance_available or "fail_on_recompile_limit_hit" in applied,
        "no supported deterministic fail-closed compile mechanism is available in this "
        "runtime; refusing to weaken compile=true semantics",
    )
    return applied


def arm_fail_on_recompile() -> dict[str, Any]:
    """Arm ``fail_on_recompile`` AFTER initial realization, so later recompiles abort."""
    import torch

    if not hasattr(torch.compiler, "set_stance"):
        return {"armed": False, "reason": "set_stance_unavailable"}
    torch.compiler.set_stance("fail_on_recompile")
    return {"armed": True, "stance": "fail_on_recompile"}


def isolated_inductor_cache(run_token: str) -> dict[str, Any]:
    """R2 Part 9: a process/run-specific Inductor cache that must be empty beforehand.

    Pre-existing artifacts in a shared cache can never stand in for evidence that THIS
    process realized compilation.
    """
    import tempfile

    root = Path(tempfile.gettempdir()) / f"petitgpt_governed_inductor_{run_token}"
    require(
        not root.exists() or not any(root.rglob("*")),
        f"governed compile requires an empty isolated Inductor cache, found artifacts at {root}",
    )
    root.mkdir(parents=True, exist_ok=True)
    triton = root / "triton"
    triton.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(root)
    os.environ["TRITON_CACHE_DIR"] = str(triton)
    return {
        "cache_dir": str(root),
        "triton_cache_dir": str(triton),
        "was_empty_before_realization": True,
        "isolated": True,
    }


def governed_autocast(device: Any):
    """The canonical bf16 autocast context the governed training forward runs under."""
    import torch

    device_type = getattr(device, "type", "cpu")
    if device_type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def realize_compile_production_shape(
    compiled_callable: Any,
    *,
    device: Any,
    micro_bsz: int = COMPILE_PROBE_MICRO_BSZ,
    seq_len: int = COMPILE_PROBE_SEQ_LEN,
    vocab_size: int = MODEL_CONTRACT["vocab_size"],
    cache: Mapping[str, Any] | None = None,
    finalize: bool = True,
) -> dict[str, Any]:
    """Realize compile at the frozen production input geometry.

    Batch-1 is insufficient: the shape the training loop actually compiles for is
    ``micro_bsz x seq_len``. This invokes the real compiled callable with gradients enabled
    and governed bf16 autocast, exactly like the training forward, but takes no optimizer
    step. Gate C requests an unsealed draft so it can add the already-observed causal and
    stance facts before the one final seal.
    """
    import torch

    forward = ObservedForward(compiled_callable, compiled_object=compiled_callable)
    before = dynamo_counters_snapshot()
    reset_dynamo_counters()

    # R3 Part 9: match the ACTUAL governed training forward signature. Training runs
    # grad-enabled under bf16 autocast, so a no_grad probe would capture a different graph
    # and the first real forward would recompile. No optimizer step is taken; the autograd
    # graph this builds is discarded immediately after evidence capture.
    probe = torch.randint(
        0, int(vocab_size), (int(micro_bsz), int(seq_len)), dtype=torch.long, device=device
    )
    device_type = str(getattr(device, "type", "cpu"))
    module_training_mode = bool(getattr(compiled_callable, "training", False))
    with torch.enable_grad(), governed_autocast(device):
        observed_grad_enabled = bool(torch.is_grad_enabled())
        observed_autocast_enabled = bool(torch.is_autocast_enabled(device_type))
        observed_autocast_dtype = (
            str(torch.get_autocast_dtype(device_type)) if observed_autocast_enabled else None
        )
        output = forward(probe)
    if getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize()
    realized_dtype = str(getattr(output, "dtype", None))
    requires_grad = bool(getattr(output, "requires_grad", False))
    del output  # discard the autograd graph; nothing is backpropagated

    after = dynamo_counters_snapshot()
    evidence = compile_realization_evidence(
        compiled_callable,
        forward,
        requested=True,
        cache_dir=(cache or {}).get("cache_dir"),
        expected_forward_invocations=1,
        counters=after,
        finalize=False,
    )
    evidence["probe_geometry"] = {"micro_bsz": int(micro_bsz), "seq_len": int(seq_len)}
    evidence["probe_signature"] = {
        "device_type": device_type,
        "module_training_mode": module_training_mode,
        "grad_enabled": observed_grad_enabled,
        "autocast_enabled": observed_autocast_enabled,
        "autocast": "bf16" if observed_autocast_dtype == "torch.bfloat16" else "none",
        "autocast_dtype": observed_autocast_dtype,
        "input_dtype": str(probe.dtype),
        "input_shape": list(probe.shape),
        "output_dtype": realized_dtype,
        "output_requires_grad": requires_grad,
        "optimizer_step_taken": False,
        "matches_training_forward_signature": bool(
            device_type == "cuda"
            and module_training_mode
            and observed_grad_enabled
            and observed_autocast_enabled
            and observed_autocast_dtype == "torch.bfloat16"
            and probe.dtype == torch.int64
            and list(probe.shape) == [MICRO_BSZ, MODEL_CONTRACT["seq_len"]]
            and requires_grad
        ),
    }
    evidence["production_shape_probe"] = (
        int(micro_bsz) == MICRO_BSZ and int(seq_len) == MODEL_CONTRACT["seq_len"]
    )
    evidence["counters_before"] = before
    evidence["isolated_cache"] = dict(cache) if cache else None
    evidence["cache_was_empty_before_realization"] = bool(
        (cache or {}).get("was_empty_before_realization")
    )
    if not evidence["production_shape_probe"]:
        evidence.setdefault("failures", []).append("compile_probe_not_at_production_shape")
    if cache is not None and not evidence["cache_was_empty_before_realization"]:
        evidence.setdefault("failures", []).append("inductor_cache_not_isolated_or_not_empty")
    return finalize_compile_evidence(evidence) if finalize else evidence


def verify_compile_evidence_document(evidence: Mapping[str, Any] | None) -> list[str]:
    """Recompute the seal and independently derive the verdict from recorded sub-facts."""
    if not isinstance(evidence, Mapping):
        return ["compile_evidence_missing_or_malformed"]
    failures: list[str] = []
    if evidence.get("schema_version") != "petitgpt-governed-compile-evidence-v1":
        failures.append("compile_evidence_schema_mismatch")
    for field in (
        *COMPILE_REALIZATION_SUBFACTS,
        "dynamo_unique_graphs",
        "inductor_artifact_count",
        "forward_invocations",
        "expected_forward_invocations",
        "failures",
        "verdict",
        "compile_realized",
        "compile_evidence_sha256",
    ):
        if field not in evidence:
            failures.append(f"compile_evidence_missing_field:{field}")
    stored = evidence.get(COMPILE_EVIDENCE_SELF_HASH_FIELD)
    recomputed = seal_compile_evidence(evidence)
    if stored != recomputed:
        failures.append("compile_evidence_sha256_does_not_match_its_own_document")

    recorded = evidence.get("failures")
    if not isinstance(recorded, list):
        failures.append("compile_evidence_failures_is_not_a_list")
        recorded_failures: list[Any] = ["malformed_failures"]
    else:
        recorded_failures = recorded
    derived_failures = _derived_compile_failures(evidence)
    failures.extend(derived_failures)
    expected_success = not recorded_failures and not derived_failures
    expected_verdict = "PASS" if expected_success else "FAIL"
    if evidence.get("verdict") != expected_verdict:
        failures.append("compile_evidence_verdict_contradicts_verified_subfacts")
    if bool(evidence.get("compile_realized")) != expected_success:
        failures.append("compile_realized_contradicts_verified_subfacts")
    if not expected_success:
        failures.append("compile_evidence_does_not_represent_a_realized_compile")
    return list(dict.fromkeys(failures))


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


SCOPE_REQUIRED_TRANSITION = MappingProxyType({"STAGE_N": None, "STAGE_O": "A_TO_B"})


def required_transition_for_authorized_source(
    authorization: Mapping[str, Any],
    *,
    scope: str,
    verified_source_authority: Mapping[str, Any] | None = None,
) -> tuple[str | None, list[str]]:
    """Derive the only legal transition from scope and authenticated source stage.

    A first Stage-O invocation consumes Stage A and therefore requires ``A_TO_B``. A later
    Stage-O crash restart consumes a Stage-B checkpoint and must use ordinary exact
    same-stage comparison. The operator-provided declaration is never used to choose between
    those rules; the independently loaded and verified source run contract is.
    """
    if scope not in SCOPE_REQUIRED_TRANSITION:
        return None, [f"transition_declaration_for_unknown_scope:{scope!r}"]

    resume = authorization.get("resume") or {}
    if resume.get("mode") != "RESUME_EXACT_CHECKPOINT":
        return SCOPE_REQUIRED_TRANSITION[scope], []
    if verified_source_authority is None:
        # Compatibility for direct declaration-schema checks. Gate A always supplies the
        # source-verification result for an exact resume.
        return SCOPE_REQUIRED_TRANSITION[scope], []
    if not bool(verified_source_authority.get("verified")):
        return None, ["transition_declaration_requires_verified_source_authority"]
    source_contract = verified_source_authority.get("source_invocation_run_contract")
    if not isinstance(source_contract, Mapping):
        return None, ["transition_declaration_verified_source_contract_missing"]
    source_stage = source_contract.get("stage")
    if scope == "STAGE_N":
        if source_stage != "stage_a":
            return None, [f"transition_source_stage_invalid_for_scope:{scope}:{source_stage!r}"]
        return None, []
    if source_stage == "stage_a":
        return "A_TO_B", []
    if source_stage == "stage_b":
        return None, []
    return None, [f"transition_source_stage_invalid_for_scope:{scope}:{source_stage!r}"]


def validate_transition_declaration(
    authorization: Mapping[str, Any],
    *,
    scope: str,
    verified_source_authority: Mapping[str, Any] | None = None,
) -> list[str]:
    """The transition kind is source-derived, never operator-selectable.

    ``A_TO_B`` is the only declaration that suppresses same-stage invocation-identity
    matching. If a STAGE_N authorization could declare it, a same-stage resume would silently
    be judged by the transition rule instead -- the weaker one for that case. Likewise, a
    Stage-B restart must not inherit Stage O's initial Stage-A handoff rule.
    """
    declared = authorization.get("transition")
    expected, failures = required_transition_for_authorized_source(
        authorization,
        scope=scope,
        verified_source_authority=verified_source_authority,
    )
    if failures:
        return failures
    if declared != expected:
        return [
            f"transition_declaration_invalid_for_scope:{scope} requires "
            f"transition={expected!r}, authorization declares {declared!r}"
        ]
    return []


def validate_stage_n_completion_binding(
    authorization: Mapping[str, Any], args: argparse.Namespace, *, stage: str
) -> list[str]:
    """Validate an optional future Stage-N completion choice during Gate A.

    The block does not authorize training by itself. If present, however, it must bind the
    already frozen invocation stop before the one-shot GRC can be published.
    """
    completion = authorization.get("stage_n_completion")
    if completion is None:
        return []
    if stage != "stage_a":
        return ["stage_n_completion_is_only_valid_for_stage_a"]
    if not isinstance(completion, Mapping):
        return ["stage_n_completion_must_be_an_object"]
    expected = completion.get("expected_final_step")
    if not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0:
        return ["stage_n_completion_expected_final_step_invalid"]
    invocation_stop = int(getattr(args, "max_steps", -1))
    frozen_stop = int(STAGE_BOUNDARIES[stage]["stop_step"])
    failures: list[str] = []
    if expected != invocation_stop:
        failures.append("stage_n_completion_expected_final_step_differs_from_max_steps")
    if expected != frozen_stop:
        failures.append("stage_n_completion_expected_final_step_differs_from_frozen_stop")
    return failures


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

    # ``allowed_output_root`` binds the stable governed production run. Actual output paths
    # are invocation-specific and can only be derived after the authorization bytes have a
    # digest, avoiding a circular authorization that embeds its own hash-derived path.
    output_binding = validate_invocation_output_binding(
        args,
        run_root=authorization.get("allowed_output_root"),
        authorized_samples_dir=authorization.get("allowed_samples_dir"),
        stage=stage,
        authorization_sha256=authorization_sha,
    )
    failures.extend(output_binding["failures"])
    invocation_paths = output_binding.get("layout") or {}

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
    failures.extend(
        validate_resume_binding(
            args,
            authorization.get("resume"),
            require_source_authority=True,
            transition=authorization.get("transition"),
        )
    )
    failures.extend(validate_bench_eval_out_dir(args))
    failures.extend(validate_stage_n_completion_binding(authorization, args, stage=stage))

    # R2 Part 5: when the authorization pins an exact checkpoint, its BYTES are verified here,
    # long before any executable state could be restored.
    resume_binding = authorization.get("resume") or {}
    checkpoint_verification: dict[str, Any] | None = None
    source_authority: dict[str, Any] | None = None
    if resume_binding.get("mode") == "RESUME_EXACT_CHECKPOINT":
        checkpoint_verification = verify_authorized_checkpoint_bytes(
            resume_binding, resume_binding.get("checkpoint_path", "")
        )
        failures.extend(checkpoint_verification["failures"])
        source_authority = verify_resume_source_authority(
            resume_binding,
            transition=authorization.get("transition"),
        )
        failures.extend(source_authority["failures"])

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
        "output_root": invocation_paths.get("governed_run_root"),
        "canonical_cwd": resolved_cwd,
        "training_runtime": dict(observed_runtime),
    }
    verdict = validate_authorization(authorization, requested_scope=scope, observed=observed)
    failures.extend(verdict["failures"])

    # 7. Stage O additionally requires the accepted Stage-N chain. The value carried into
    # checkpoint loading is reconstructed from that accepted result, never re-read from a
    # separately selectable authorization mapping.
    stage_o_chain_verdict: dict[str, Any] | None = None
    if scope == "STAGE_O":
        stage_o_chain_verdict = validate_stage_o_chain(
            authorization, observed_runtime=observed_runtime
        )
        failures.extend(stage_o_chain_verdict["failures"])

    # 8. R3 Part 18: the declared transition selects WHICH resume rule applies, so it is not
    # free text. Declaring A_TO_B makes validate_governed_checkpoint_before_restore skip
    # invocation-identity matching entirely; an unvalidated field must never be able to
    # choose the weaker rule. It is therefore pinned to the scope.
    failures.extend(
        validate_transition_declaration(
            authorization,
            scope=scope,
            verified_source_authority=source_authority,
        )
    )

    failures = list(dict.fromkeys(failures))
    require(
        not failures,
        f"Gate A refused the governed launch under {CONTRACT_VERSION}:\n  - "
        + "\n  - ".join(failures),
    )

    # No output consumer has been constructed yet. Replace the non-circular run-root CLI
    # alias with the only canonical per-invocation paths before returning to trainer.main.
    args.out_dir = invocation_paths["out_dir"]
    args.samples_dir = invocation_paths["samples_dir"]
    return {
        "gate": "A",
        "stage": stage,
        "scope": scope,
        "launch_contract_path": contract_artifact["path"],
        "launch_contract_sha256": contract_artifact["sha256"],
        "stage_authorization_path": str(auth_path.expanduser().resolve()),
        "stage_authorization_sha256": authorization_sha,
        "exact_plan_path": str(exact_plan_path),
        "exact_run_plan_sha256": plan_sha,
        "pilot_acceptance_path": str(pilot_acceptance_path),
        "pilot_owner_acceptance_sha256": acceptance_sha,
        "trainer_head": repo["head"],
        "trainer_branch": repo["branch"],
        "trainer_execution_bundle_sha256": observed["trainer_execution_bundle_sha256"],
        "resume": dict(
            (stage_o_chain_verdict or {}).get("derived_resume") or authorization.get("resume") or {}
        ),
        "authorized_checkpoint_verification": checkpoint_verification,
        "verified_source_authority": source_authority,
        "num_workers": (authorization.get("training_runtime") or {}).get("num_workers"),
        "governed_run_root": invocation_paths["governed_run_root"],
        "invocation_root": invocation_paths["invocation_root"],
        "out_dir": invocation_paths["out_dir"],
        "samples_dir": invocation_paths["samples_dir"],
        "transition": authorization.get("transition"),
        "source_stage_a": dict(authorization.get("source_stage_a") or {}),
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
        "governed_run_root": gate_a.get("governed_run_root"),
        "invocation_root": gate_a.get("invocation_root"),
        "out_dir": gate_a.get("out_dir"),
        "samples_dir": gate_a.get("samples_dir"),
        "resume": dict(gate_a["resume"]),
        "runtime_fingerprint": runtime,
        "runtime_fingerprint_sha256": (
            runtime.get("runtime_fingerprint_sha256") or runtime_fingerprint_sha256(runtime)
        ),
        "compile_intent": COMPILE,
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
    }


DYNAMIC_CHECKPOINT_STATE_FIELDS = (
    "active_stage",
    "active_stage_sampler_seed",
    "permutation_identity",
    "range_start_position",
    "invocation_range_start_position",
    "range_stop_position",
    "cursor",
    "global_step",
    "completed_evaluation_milestones",
    "completed_checkpoint_milestones",
    "rng_state",
    "compile_evidence",
    "compile_evidence_sha256",
)


CUDA_RNG_STATE_NUM_BYTES = 16


def validate_cuda_rng_states(
    cuda_states: Any,
    *,
    require_live_validation: bool = False,
) -> list[str]:
    """Validate the one-device CUDA Philox state without touching global generators.

    The frozen PyTorch 2.11 CUDA runtime serializes a CUDA generator as exactly 16 bytes.
    When CUDA is present we additionally round-trip the bytes through an isolated temporary
    generator. Governed save/restore callers require that live check; offline artifact
    inspection may use the exact-format structural check when CUDA is unavailable.
    """
    failure = "governed_checkpoint_dynamic_state_torch_cuda_rng_not_restorable"
    if not isinstance(cuda_states, (list, tuple)) or len(cuda_states) != 1:
        return ["governed_checkpoint_dynamic_state_torch_cuda_rng_count_mismatch"]

    try:
        import torch

        value = cuda_states[0]
        if not (
            isinstance(value, torch.Tensor)
            and value.device.type == "cpu"
            and value.dtype == torch.uint8
            and value.ndim == 1
            and value.is_contiguous()
            and value.numel() == CUDA_RNG_STATE_NUM_BYTES
        ):
            return [failure]

        if not torch.cuda.is_available():
            return [failure] if require_live_validation else []
        if torch.cuda.device_count() != 1:
            return ["governed_checkpoint_dynamic_state_torch_cuda_rng_count_mismatch"]

        generator = torch.Generator(device="cuda:0")
        generator.set_state(value)
        if not torch.equal(generator.get_state(), value):
            return [failure]
    except (ImportError, RuntimeError, TypeError):
        return [failure]
    return []


def validate_restorable_rng_state(
    rng_state: Mapping[str, Any] | None,
    *,
    require_live_cuda_validation: bool = False,
) -> list[str]:
    """Validate full RNG payloads without mutating any process-global generator."""
    if not isinstance(rng_state, Mapping):
        return ["governed_checkpoint_dynamic_state_rng_state_missing"]

    failures: list[str] = []
    for stream in ("python", "numpy", "torch_cpu", "torch_cuda"):
        if stream not in rng_state or rng_state.get(stream) is None:
            failures.append(f"governed_checkpoint_dynamic_state_rng_stream_missing:{stream}")
    if failures:
        return failures

    try:
        import random

        random.Random().setstate(rng_state["python"])
    except (TypeError, ValueError):
        failures.append("governed_checkpoint_dynamic_state_python_rng_not_restorable")

    try:
        import numpy as np

        np.random.RandomState().set_state(rng_state["numpy"])
    except (TypeError, ValueError):
        failures.append("governed_checkpoint_dynamic_state_numpy_rng_not_restorable")

    try:
        import torch

        cpu_state = rng_state["torch_cpu"]
        if not (
            isinstance(cpu_state, torch.Tensor)
            and cpu_state.device.type == "cpu"
            and cpu_state.dtype == torch.uint8
            and cpu_state.ndim == 1
            and cpu_state.numel() > 0
        ):
            raise TypeError
        torch.Generator(device="cpu").set_state(cpu_state)
    except (RuntimeError, TypeError):
        failures.append("governed_checkpoint_dynamic_state_torch_cpu_rng_not_restorable")

    failures.extend(
        validate_cuda_rng_states(
            rng_state["torch_cuda"],
            require_live_validation=require_live_cuda_validation,
        )
    )
    return failures


def build_checkpoint_state(
    *,
    stage: str,
    sampler: Any,
    global_step: int,
    completed_evaluation_milestones: Sequence[int],
    completed_checkpoint_milestones: Sequence[int],
    rng_state: Mapping[str, Any] | None,
    compile_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Dynamic state derived from the LIVE sampler/trainer, never launch-time summaries."""
    require(
        isinstance(global_step, int) and not isinstance(global_step, bool) and global_step >= 0,
        f"governed checkpoint global_step must be a non-negative exact int, got {global_step!r}",
    )
    identity = sampler_identity_document(stage, sampler)
    rng_failures = validate_restorable_rng_state(rng_state)
    require(
        not rng_failures,
        "governed checkpoint requires full restorable RNG state: " + ", ".join(rng_failures),
    )
    compile_failures = verify_compile_evidence_document(compile_evidence)
    require(
        not compile_failures,
        "governed checkpoint requires current verified compile evidence: "
        + ", ".join(compile_failures),
    )
    completed_evaluations = sorted({int(v) for v in completed_evaluation_milestones})
    completed_checkpoints = sorted({int(v) for v in completed_checkpoint_milestones})
    expected_evaluations = [v for v in EVALUATION_MILESTONES if v <= int(global_step)]
    expected_checkpoints = [v for v in CHECKPOINT_MILESTONES if v <= int(global_step)]
    require(
        completed_evaluations == expected_evaluations,
        "governed checkpoint completed evaluation milestones must equal the frozen prefix "
        f"through step {global_step}: expected {expected_evaluations}, "
        f"got {completed_evaluations}",
    )
    require(
        completed_checkpoints == expected_checkpoints,
        "governed checkpoint completed checkpoint milestones must equal the frozen prefix "
        f"through step {global_step}: expected {expected_checkpoints}, "
        f"got {completed_checkpoints}",
    )
    return {
        "schema_version": "petitgpt-governed-checkpoint-state-v1",
        "active_stage": stage,
        "active_stage_sampler_seed": identity["sampler_seed"],
        "permutation_identity": identity["permutation_identity"],
        "range_start_position": identity["range_start_position"],
        "invocation_range_start_position": identity["invocation_range_start_position"],
        "range_stop_position": identity["range_stop_position"],
        "cursor": identity["cursor"],
        "consumed": identity["consumed"],
        "remaining": identity["remaining"],
        "global_step": int(global_step),
        "completed_evaluation_milestones": completed_evaluations,
        "completed_checkpoint_milestones": completed_checkpoints,
        "rng_state": dict(rng_state),
        "rng_state_present": True,
        "rng_state_streams": sorted(rng_state),
        "compile_evidence": dict(compile_evidence),
        "compile_evidence_sha256": compile_evidence.get("compile_evidence_sha256"),
    }


def validate_governed_checkpoint_state(
    state: Mapping[str, Any] | None,
    *,
    governed_run_contract: Mapping[str, Any] | None = None,
    checkpoint_global_step: int | None = None,
    require_live_cuda_validation: bool = False,
) -> list[str]:
    """Validate that governed dynamic state is internally exact and fully restorable."""
    if not isinstance(state, Mapping):
        return ["governed_checkpoint_dynamic_state_missing"]
    failures: list[str] = []
    for field in DYNAMIC_CHECKPOINT_STATE_FIELDS:
        if field not in state or state.get(field) is None:
            failures.append(f"governed_checkpoint_dynamic_state_missing_field:{field}")
    if state.get("schema_version") != "petitgpt-governed-checkpoint-state-v1":
        failures.append("governed_checkpoint_dynamic_state_schema_mismatch")

    stage = state.get("active_stage")
    if stage not in STAGE_ORDER:
        failures.append(f"governed_checkpoint_dynamic_state_stage_invalid:{stage!r}")
    else:
        seed = state.get("active_stage_sampler_seed")
        if seed != stage_sampler_seed(str(stage)):
            failures.append("governed_checkpoint_dynamic_state_sampler_seed_mismatch")
        stop = state.get("range_stop_position")
        if isinstance(seed, int) and isinstance(stop, int):
            expected_permutation = permutation_identity(str(stage), seed, stop)
            if state.get("permutation_identity") != expected_permutation:
                failures.append("governed_checkpoint_dynamic_state_permutation_mismatch")

    position_fields = (
        "range_start_position",
        "invocation_range_start_position",
        "cursor",
        "range_stop_position",
    )
    if any(
        not isinstance(state.get(field), int) or isinstance(state.get(field), bool)
        for field in position_fields
    ):
        failures.append("governed_checkpoint_dynamic_state_sampler_positions_invalid")
    else:
        start = state["range_start_position"]
        invocation_start = state["invocation_range_start_position"]
        cursor = state["cursor"]
        stop = state["range_stop_position"]
        if start != 0 or not start <= invocation_start <= cursor <= stop:
            failures.append("governed_checkpoint_dynamic_state_sampler_range_invalid")

    if checkpoint_global_step is not None and state.get("global_step") != int(
        checkpoint_global_step
    ):
        failures.append("governed_checkpoint_dynamic_state_global_step_mismatch")

    global_step = state.get("global_step")
    if not isinstance(global_step, int) or isinstance(global_step, bool) or global_step < 0:
        failures.append("governed_checkpoint_dynamic_state_global_step_invalid")

    milestone_policies = (
        ("completed_evaluation_milestones", EVALUATION_MILESTONES),
        ("completed_checkpoint_milestones", CHECKPOINT_MILESTONES),
    )
    for field, allowed_milestones in milestone_policies:
        values = state.get(field)
        if not isinstance(values, list) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            failures.append(f"governed_checkpoint_dynamic_state_invalid:{field}")
        elif values != sorted(set(values)):
            failures.append(f"governed_checkpoint_dynamic_state_not_canonical:{field}")
        else:
            unknown = [value for value in values if value not in allowed_milestones]
            if unknown:
                failures.append(
                    f"governed_checkpoint_dynamic_state_unknown_milestones:{field}:{unknown}"
                )
            if isinstance(global_step, int) and not isinstance(global_step, bool):
                expected = [value for value in allowed_milestones if value <= global_step]
                if values != expected:
                    failures.append(
                        f"governed_checkpoint_dynamic_state_milestone_prefix_mismatch:{field}"
                    )

    failures.extend(
        validate_restorable_rng_state(
            state.get("rng_state"),
            require_live_cuda_validation=require_live_cuda_validation,
        )
    )

    compile_required = bool(
        ((governed_run_contract or {}).get("training") or {}).get("compile", COMPILE)
    )
    if compile_required:
        evidence = state.get("compile_evidence")
        failures.extend(verify_compile_evidence_document(evidence))
        if isinstance(evidence, Mapping) and state.get("compile_evidence_sha256") != evidence.get(
            "compile_evidence_sha256"
        ):
            failures.append("governed_checkpoint_dynamic_state_compile_sha256_mismatch")
    return list(dict.fromkeys(failures))


# Fields a resume may never differ on. `sampler_identity` moves legitimately as training
# advances, so it is validated separately against the stage/range/cursor rules.
# ---------------------------------------------------------------------------------
# R3: two explicit identity layers.
#
# BASE_GOVERNED_IDENTITY is what must NOT change across the whole governed production run.
# INVOCATION_IDENTITY is what each separately authorized invocation (fresh, resume, or an
# explicitly authorized A->B transition) legitimately binds for itself.
#
# Collapsing these into one immutable set is what made a legitimate authorized Stage-A ->
# Stage-B transition impossible: the stage, its seed, its authorization and its output
# subdirectory all change by design, while the base identity does not.
# ---------------------------------------------------------------------------------

BASE_GOVERNED_IDENTITY_FIELDS = (
    "schema_version",
    "contract_version",
    "kind",
    "launch_contract_sha256",
    "exact_run_plan_sha256",
    "pilot_owner_acceptance_sha256",
    "trainer_branch",
    "trainer_head",
    "trainer_execution_bundle_sha256",
    "model",
    "training",
    "optimizer_realization",
    "seeds",
    "special_token_ids",
    "evaluation_policy",
    "checkpoint_policy",
    "canonical_cwd",
    "num_workers",
    "runtime_fingerprint_sha256",
    "gpu_uuid",
    "gpu_pci_bus_id",
    "compile_intent",
)

INVOCATION_IDENTITY_FIELDS = (
    "stage_authorization_sha256",
    "stage",
    "scope",
    "governed_run_root",
    "invocation_root",
    "out_dir",
    "samples_dir",
    "resume",
    "active_stage_sampler_seed",
    "stage_start_step",
    "stage_stop_step",
)


def base_governed_identity(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {k: contract.get(k) for k in BASE_GOVERNED_IDENTITY_FIELDS}


# R3 Part 18: the INVOCATION fields that must match across a SAME-STAGE restart.
#
# ``stage_authorization_sha256`` and ``resume`` are deliberately excluded. A crash restart is
# a new invocation: it is authorized by a new file (new SHA) whose resume binding names the
# checkpoint, while the checkpoint it resumes was written under a FRESH-mode authorization.
# Requiring those two to match made every same-stage restart structurally impossible -- a
# multi-day Stage-A run could never be resumed after an interruption. The continuity that
# actually matters is enforced instead: identical stage, scope, output roots, sampler seed
# and absolute boundaries, plus the full BASE identity, plus the exact cursor rule.
SAME_STAGE_INVOCATION_MATCH_FIELDS = tuple(
    f
    for f in INVOCATION_IDENTITY_FIELDS
    if f
    not in {
        "stage_authorization_sha256",
        "invocation_root",
        "out_dir",
        "samples_dir",
        "resume",
    }
)


def base_governed_identity_sha256(contract: Mapping[str, Any]) -> str:
    return _sha256_bytes(canonical_json_bytes(base_governed_identity(contract)))


def invocation_identity(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {k: contract.get(k) for k in INVOCATION_IDENTITY_FIELDS}


def invocation_identity_sha256(contract: Mapping[str, Any]) -> str:
    return _sha256_bytes(canonical_json_bytes(invocation_identity(contract)))


# Retained for same-stage resume, where BOTH layers must be identical.
GOVERNED_IMMUTABLE_FIELDS = (
    "schema_version",
    "contract_version",
    "kind",
    "stage",
    "scope",
    "runtime_fingerprint_sha256",
    "compile_intent",
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
    "samples_dir",
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


SAMPLER_REQUIRED_ATTRS = (
    "seed",
    "range_start_position",
    "end_position",
    "committed_position",
)


def sampler_identity_document(stage: str, sampler: Any) -> dict[str, Any]:
    """Record canonical stage range plus the LIVE invocation's exact start/cursor.

    A restart's real sampler begins at the recovered cursor, while the permutation's canonical
    stage range still begins at zero. Persisting only one overloaded "range start" either made
    restart impossible or made a restarted Stage A ineligible for A->B. Both values are
    load-bearing and independently authenticated.
    """
    require(stage in STAGE_ORDER, f"unknown stage {stage!r}")
    expected_seed = stage_sampler_seed(stage)
    missing = [a for a in SAMPLER_REQUIRED_ATTRS if not hasattr(sampler, a)]
    require(
        not missing,
        f"sampler is missing required position attributes {missing}; a governed sampler "
        f"identity may never default a missing range start to zero",
    )
    seed = int(sampler.seed)
    require(
        seed == expected_seed,
        f"live sampler seed {seed} differs from frozen {stage} seed {expected_seed}",
    )
    invocation_start = int(sampler.range_start_position)
    canonical_start = 0
    stop = int(sampler.end_position)
    cursor = int(sampler.committed_position)
    require(
        canonical_start <= invocation_start <= cursor <= stop,
        "live sampler positions must satisfy canonical_start <= invocation_start <= "
        f"cursor <= stop, got {canonical_start}, {invocation_start}, {cursor}, {stop}",
    )
    identity = {
        "stage": stage,
        "sampler_seed": seed,
        "range_start_position": canonical_start,
        "invocation_range_start_position": invocation_start,
        "range_stop_position": stop,
        "cursor": cursor,
        "consumed": max(0, cursor - canonical_start),
        "remaining": max(0, stop - cursor),
    }
    identity["permutation_identity"] = permutation_identity(stage, seed, stop)
    return identity


def permutation_identity(stage: str, sampler_seed: int, range_stop_position: int) -> str:
    """Identify the PERMUTATION, not the invocation that is walking it.

    ``ResumablePermutationSampler`` derives each epoch's permutation from ``seed`` and the
    epoch index alone; ``range_start_position`` is per-invocation bookkeeping for the planned
    remainder. Keying this digest on the range start would therefore have made a legitimate
    same-stage resume -- which necessarily restarts its range at the recovered cursor --
    look like a different permutation, and every crash restart would have been rejected.
    """
    return _sha256_bytes(
        canonical_json_bytes({
            "stage": stage,
            "sampler_seed": int(sampler_seed),
            "range_stop_position": int(range_stop_position),
        })
    )


# Fields that identify the canonical stage permutation and must be bit-identical across a
# same-stage resume. The separate invocation range start legitimately advances to the saved
# cursor and is checked below.
SAME_STAGE_EXACT_FIELDS = (
    "stage",
    "sampler_seed",
    "permutation_identity",
    "range_start_position",
    "range_stop_position",
)


def expected_sampler_identity_for_resume(
    stage: str,
    *,
    expected_step: int,
    data_stage_start_step: int,
    micro_bsz: int,
    grad_accum: int,
    planned_stage_samples: int,
) -> dict[str, Any]:
    """R3 Part 5: the expected sampler identity, DERIVED before the checkpoint is opened.

    The training sampler is constructed after the resume completes, so the expectation cannot
    be read off the sampler object. It does not need to be: every field is a function of the
    frozen policy plus the authorized resume step, so it can be computed first and used to
    validate the checkpoint before any state is restored.

    ``ResumablePermutationSampler`` is built with ``start_position=stage_sample_position`` and
    ``num_samples=planned_stage_samples - stage_sample_position``, and its committed cursor
    begins at its range start. This reproduces exactly that derivation.
    """
    require(stage in STAGE_ORDER, f"unknown stage {stage!r}")
    require(int(expected_step) >= 0, f"expected_step must be non-negative, got {expected_step}")
    require(
        int(expected_step) >= int(data_stage_start_step),
        f"expected_step {expected_step} precedes data_stage_start_step {data_stage_start_step}",
    )
    samples_per_step = int(micro_bsz) * int(grad_accum)
    require(samples_per_step > 0, "micro_bsz * grad_accum must be positive")
    local_step = int(expected_step) - int(data_stage_start_step)
    start = local_step * samples_per_step
    stop = int(planned_stage_samples)
    require(
        start <= stop,
        f"derived sampler range start {start} exceeds the planned stage budget {stop}",
    )
    seed = stage_sampler_seed(stage)
    identity = {
        "stage": stage,
        "sampler_seed": seed,
        "range_start_position": 0,
        "invocation_range_start_position": start,
        "range_stop_position": stop,
        "cursor": start,
        "consumed": start,
        "remaining": stop - start,
    }
    identity["permutation_identity"] = permutation_identity(stage, seed, stop)
    return identity


def validate_same_stage_resume(
    saved: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    expected_global_step: int | None = None,
    saved_global_step: int | None = None,
) -> list[str]:
    """R3 Part 4: EXACT equality, including the cursor.

    "The cursor lies inside its range" is not sufficient: an unequal but in-range cursor
    means the checkpoint and the reconstructed sampler disagree about how much has been
    consumed, and that must fail before any state is restored.
    """
    failures: list[str] = []
    for field in SAME_STAGE_EXACT_FIELDS:
        if field not in saved or saved.get(field) is None:
            failures.append(f"sampler_resume_missing_saved_field:{field}")
            continue
        if field not in current or current.get(field) is None:
            failures.append(f"sampler_resume_missing_current_field:{field}")
            continue
        if saved[field] != current[field]:
            failures.append(
                f"sampler_resume_mismatch:{field}: checkpoint {saved[field]!r} != "
                f"current {current[field]!r}"
            )
    for field in (
        "cursor",
        "range_start_position",
        "invocation_range_start_position",
        "range_stop_position",
    ):
        if saved.get(field) is None:
            failures.append(f"sampler_resume_missing_saved_field:{field}")
        if current.get(field) is None:
            failures.append(f"sampler_resume_missing_current_field:{field}")

    if not failures:
        saved_cursor = int(saved["cursor"])
        saved_start = int(saved["range_start_position"])
        saved_invocation_start = int(saved["invocation_range_start_position"])
        stop = int(saved["range_stop_position"])
        cur_start = int(current["invocation_range_start_position"])
        cur_cursor = int(current["cursor"])

        # EXACT continuity: the resuming invocation must begin precisely where the checkpoint
        # committed -- no earlier (replay) and no later (skipped data). This is stricter than
        # the "cursor lies inside its range" test it replaces, while still permitting the
        # range start to move from 0 to the recovered cursor as the trainer really does.
        if cur_start != saved_cursor:
            failures.append(
                f"sampler_resume_discontinuity: checkpoint committed {saved_cursor} but the "
                f"resuming sampler starts at {cur_start}"
            )
        if cur_cursor != saved_cursor:
            failures.append(
                f"sampler_resume_mismatch:cursor: checkpoint {saved_cursor} != current {cur_cursor}"
            )
        if not saved_start <= saved_invocation_start <= saved_cursor <= stop:
            failures.append(
                "sampler positions are inconsistent: canonical start "
                f"{saved_start}, invocation start {saved_invocation_start}, cursor "
                f"{saved_cursor}, stop {stop}"
            )
    if expected_global_step is not None:
        if saved_global_step is None:
            failures.append("sampler_resume_missing_saved_global_step")
        elif int(saved_global_step) != int(expected_global_step):
            failures.append(
                f"sampler_resume_mismatch:global_step: checkpoint {saved_global_step} != "
                f"expected {expected_global_step}"
            )
    return failures


def validate_governed_operational_sampler_state(
    sampler_state: Mapping[str, Any] | None,
    dynamic_state: Mapping[str, Any] | None,
    data_contract: Mapping[str, Any] | None = None,
) -> list[str]:
    """Cross-authenticate the executable sampler state against governed identity.

    ``data_sampler`` is what the trainer actually loads, while
    ``governed_checkpoint_state`` is the authenticated continuity document. Both must
    describe the same live sampler before a governed checkpoint can be saved, restored, or
    accepted as Stage-N evidence.
    """
    if not isinstance(sampler_state, Mapping):
        return ["governed_checkpoint_operational_sampler_state_missing"]
    if not isinstance(dynamic_state, Mapping):
        return ["governed_checkpoint_dynamic_sampler_state_missing"]

    failures: list[str] = []
    required_operational_fields = (
        "version",
        "data_length",
        "seed",
        "range_start_position",
        "committed_position",
        "end_position",
    )
    for field in required_operational_fields:
        value = sampler_state.get(field)
        if type(value) is not int:
            failures.append(f"governed_checkpoint_operational_sampler_invalid:{field}")

    if type(sampler_state.get("version")) is int and sampler_state.get("version") != 2:
        failures.append("governed_checkpoint_operational_sampler_version_mismatch")
    if type(sampler_state.get("data_length")) is int and sampler_state.get("data_length") <= 0:
        failures.append("governed_checkpoint_operational_sampler_data_length_invalid")
    if not isinstance(data_contract, Mapping):
        failures.append("governed_checkpoint_data_contract_missing")
    else:
        contract_length = data_contract.get("dataset_length")
        if type(contract_length) is not int or contract_length <= 0:
            failures.append("governed_checkpoint_data_contract_dataset_length_invalid")
        elif (
            type(sampler_state.get("data_length")) is int
            and sampler_state.get("data_length") != contract_length
        ):
            failures.append("governed_checkpoint_sampler_data_length_contract_mismatch")

    if all(
        type(sampler_state.get(field)) is int
        for field in ("range_start_position", "committed_position", "end_position")
    ):
        start = sampler_state["range_start_position"]
        committed = sampler_state["committed_position"]
        stop = sampler_state["end_position"]
        if not 0 <= start <= committed <= stop:
            failures.append("governed_checkpoint_operational_sampler_range_invalid")

    for operational_field, dynamic_field in (
        ("seed", "active_stage_sampler_seed"),
        ("range_start_position", "invocation_range_start_position"),
        ("committed_position", "cursor"),
        ("end_position", "range_stop_position"),
    ):
        operational_value = sampler_state.get(operational_field)
        dynamic_value = dynamic_state.get(dynamic_field)
        if type(dynamic_value) is not int:
            failures.append(f"governed_checkpoint_dynamic_sampler_invalid:{dynamic_field}")
        elif type(operational_value) is int and operational_value != dynamic_value:
            failures.append(
                "governed_checkpoint_operational_dynamic_sampler_mismatch:"
                f"{operational_field}:{dynamic_field}"
            )

    return list(dict.fromkeys(failures))


def validate_governed_checkpoint_resume_envelope(
    checkpoint: Mapping[str, Any] | None,
) -> list[str]:
    """Require the complete payload that the governed real trainer can resume."""
    if not isinstance(checkpoint, Mapping):
        return ["governed_checkpoint_resume_envelope_missing"]

    failures: list[str] = []
    for field in ("model", "optim", "scaler"):
        if field not in checkpoint:
            failures.append(f"governed_checkpoint_resume_payload_missing:{field}")
    if "model" in checkpoint and not isinstance(checkpoint.get("model"), Mapping):
        failures.append("governed_checkpoint_resume_payload_invalid:model")
    if "optim" in checkpoint and not isinstance(checkpoint.get("optim"), Mapping):
        failures.append("governed_checkpoint_resume_payload_invalid:optim")
    scaler = checkpoint.get("scaler")
    if "scaler" in checkpoint and scaler is not None and not isinstance(scaler, Mapping):
        failures.append("governed_checkpoint_resume_payload_invalid:scaler")

    run_contract = checkpoint.get("run_contract")
    if not isinstance(run_contract, Mapping):
        failures.append("governed_checkpoint_run_contract_missing")
    else:
        if run_contract.get("schema_version") != 3:
            failures.append("governed_checkpoint_run_contract_schema_mismatch")
        state = checkpoint.get("governed_checkpoint_state")
        if isinstance(state, Mapping):
            if run_contract.get("sampler_seed") != state.get("active_stage_sampler_seed"):
                failures.append("governed_checkpoint_run_contract_sampler_seed_mismatch")
            run_plan = run_contract.get("run_plan")
            if not isinstance(run_plan, Mapping):
                failures.append("governed_checkpoint_run_plan_binding_missing")
            else:
                if run_plan.get("stage") != state.get("active_stage"):
                    failures.append("governed_checkpoint_run_plan_stage_mismatch")
                if run_plan.get("plan_sha256") != EXACT_RUN_PLAN_SHA256:
                    failures.append("governed_checkpoint_run_plan_sha256_mismatch")

    state = checkpoint.get("governed_checkpoint_state")
    data_contract = checkpoint.get("data_contract")
    if not isinstance(data_contract, Mapping):
        failures.append("governed_checkpoint_data_contract_missing")
    else:
        fingerprint = data_contract.get("fingerprint")
        if not _is_sha256(fingerprint):
            failures.append("governed_checkpoint_data_contract_fingerprint_invalid")
        if data_contract.get("sampling_mode") != "deterministic":
            failures.append("governed_checkpoint_data_contract_sampling_mode_mismatch")
        if data_contract.get("samples_per_optimizer_step") != SEQUENCES_PER_UPDATE:
            failures.append("governed_checkpoint_data_contract_batch_geometry_mismatch")
        if isinstance(state, Mapping):
            if data_contract.get("sampler_seed") != state.get("active_stage_sampler_seed"):
                failures.append("governed_checkpoint_data_contract_sampler_seed_mismatch")
            if data_contract.get("active_stage") != state.get("active_stage"):
                failures.append("governed_checkpoint_data_contract_stage_mismatch")
        governed_contract = checkpoint.get("governed_run_contract")
        if isinstance(governed_contract, Mapping) and data_contract.get(
            "data_stage_start_step"
        ) != governed_contract.get("stage_start_step"):
            failures.append("governed_checkpoint_data_contract_stage_start_mismatch")

    failures.extend(
        validate_governed_operational_sampler_state(
            checkpoint.get("data_sampler"), state, data_contract
        )
    )
    return list(dict.fromkeys(failures))


STAGE_A_TERMINAL_RANGE_STOP = 4882688  # Stage-A consumed blocks at the plan boundary


def validate_stage_a_to_b_transition(
    source_contract: Mapping[str, Any],
    source_state: Mapping[str, Any],
    *,
    source_binding: Mapping[str, Any] | None,
    plan_boundary_step: int = STAGE_A_STOP_STEP,
) -> list[str]:
    """A -> B is legal only from a completely bound, proven Stage-A endpoint.

    There is no optional expected-value path. Every source authority field is mandatory and
    every sampler comparison is unconditional. This validator is used instead of -- never
    alongside -- the same-stage comparison for an intentional stage change.
    """
    failures: list[str] = []

    if not isinstance(source_binding, Mapping):
        failures.append("stage_a_to_b: mandatory source binding missing")
        source_binding = {}
    for field in A_TO_B_SOURCE_REQUIRED_FIELDS:
        if source_binding.get(field) in (None, ""):
            failures.append(f"stage_a_to_b: source binding missing required field:{field}")

    if source_contract.get("kind") != GOVERNED_CHECKPOINT_KIND:
        failures.append(
            f"stage_a_to_b: source is not a governed checkpoint "
            f"(kind={source_contract.get('kind')!r})"
        )
    if source_contract.get("stage") != "stage_a":
        failures.append(
            f"stage_a_to_b: source contract stage is {source_contract.get('stage')!r}, "
            f"expected 'stage_a'"
        )
    if source_contract.get("active_stage_sampler_seed") != STAGE_A_SAMPLER_SEED:
        failures.append(
            f"stage_a_to_b: contract Stage-A seed is "
            f"{source_contract.get('active_stage_sampler_seed')!r}, "
            f"expected {STAGE_A_SAMPLER_SEED}"
        )

    required = (
        "active_stage",
        "active_stage_sampler_seed",
        "permutation_identity",
        "range_start_position",
        "invocation_range_start_position",
        "range_stop_position",
        "cursor",
        "global_step",
    )
    for field in required:
        if field not in source_state or source_state.get(field) is None:
            failures.append(f"stage_a_to_b: source state missing required field:{field}")
    if failures:
        return list(dict.fromkeys(failures))

    if source_state["active_stage"] != "stage_a":
        failures.append(
            f"stage_a_to_b: source state active_stage is {source_state['active_stage']!r}, "
            f"expected 'stage_a'"
        )
    if source_state["active_stage_sampler_seed"] != STAGE_A_SAMPLER_SEED:
        failures.append(
            f"stage_a_to_b: source state Stage-A seed is "
            f"{source_state['active_stage_sampler_seed']!r}, expected {STAGE_A_SAMPLER_SEED}"
        )
    canonical_permutation = permutation_identity(
        "stage_a", STAGE_A_SAMPLER_SEED, STAGE_A_TERMINAL_RANGE_STOP
    )
    if source_state["permutation_identity"] != canonical_permutation:
        failures.append("stage_a_to_b: Stage-A permutation identity mismatch")
    if int(source_state["range_start_position"]) != 0:
        failures.append(
            f"stage_a_to_b: Stage-A range_start_position is "
            f"{source_state['range_start_position']}, expected 0"
        )
    if int(source_state["range_stop_position"]) != STAGE_A_TERMINAL_RANGE_STOP:
        failures.append(
            f"stage_a_to_b: Stage-A range_stop_position is "
            f"{source_state['range_stop_position']}, expected {STAGE_A_TERMINAL_RANGE_STOP}"
        )

    for state_field, binding_field in (
        ("active_stage", "source_active_stage"),
        ("active_stage_sampler_seed", "source_sampler_seed"),
        ("permutation_identity", "source_permutation_identity"),
        ("range_start_position", "source_range_start_position"),
        (
            "invocation_range_start_position",
            "source_invocation_range_start_position",
        ),
        ("range_stop_position", "source_range_stop_position"),
        ("cursor", "source_cursor"),
        ("global_step", "source_checkpoint_step"),
    ):
        if source_state.get(state_field) != source_binding.get(binding_field):
            failures.append(f"stage_a_to_b: source binding mismatch:{state_field}!={binding_field}")
    if source_contract.get("stage") != source_binding.get("source_checkpoint_stage"):
        failures.append("stage_a_to_b: source contract stage differs from source binding")

    cursor = int(source_state["cursor"])
    stop = int(source_state["range_stop_position"])
    start = int(source_state["range_start_position"])
    if cursor != stop:
        failures.append(
            f"stage_a_to_b: Stage-A consumed range incomplete (cursor {cursor} != stop {stop})"
        )
    if stop <= start:
        failures.append(f"stage_a_to_b: Stage-A range is empty or inverted [{start}, {stop}]")
    if int(source_state["global_step"]) != int(plan_boundary_step):
        failures.append(
            f"stage_a_to_b: source step is {source_state['global_step']}, not the plan "
            f"boundary {plan_boundary_step}"
        )
    return list(dict.fromkeys(failures))


def verify_authorized_checkpoint_bytes(
    resume_binding: Mapping[str, Any], checkpoint_path: str | Path
) -> dict[str, Any]:
    """Hash the checkpoint on disk and require the authorized SHA.

    Path and step alone are not trusted: a changed byte must fail before any executable
    state is restored.
    """
    path = Path(checkpoint_path)
    failures: list[str] = []
    if not path.is_file():
        return {"verified": False, "failures": [f"authorized_checkpoint_not_found:{path}"]}

    authorized_path = str(resume_binding.get("checkpoint_path", ""))
    if authorized_path and Path(authorized_path).expanduser().resolve() != path.resolve():
        failures.append(
            f"resume path is not the authorized checkpoint: authorized={authorized_path}, "
            f"opened={path}"
        )
    observed = file_sha256(path)
    authorized_sha = str(resume_binding.get("checkpoint_sha256", ""))
    if observed != authorized_sha:
        failures.append(
            f"authorized checkpoint SHA-256 mismatch: authorized={authorized_sha}, "
            f"observed={observed}"
        )
    return {
        "verified": not failures,
        "failures": failures,
        "observed_sha256": observed,
        "authorized_sha256": authorized_sha,
        "path": str(path),
    }


def verify_resume_source_authority(
    resume_binding: Mapping[str, Any],
    *,
    transition: str | None,
) -> dict[str, Any]:
    """Authenticate the immutable source invocation from independent on-disk bytes.

    This result is carried from Gate A into pre-restore validation. A checkpoint document and
    a new authorization can therefore never establish continuity merely by agreeing with one
    another: both must agree with the already-published source authorization and invocation
    contract named by the new authorization.
    """
    failures: list[str] = []
    required = (
        A_TO_B_SOURCE_REQUIRED_FIELDS
        if transition == "A_TO_B"
        else RESUME_SOURCE_AUTHORITY_REQUIRED_FIELDS
    )
    missing = [field for field in required if resume_binding.get(field) in (None, "")]
    if missing:
        return {
            "verified": False,
            "failures": [f"source_authority_missing_field:{field}" for field in missing],
        }

    def load_json_artifact(path_field: str, sha_field: str, label: str) -> tuple[Path, dict]:
        path = Path(str(resume_binding[path_field])).expanduser().resolve()
        if not path.is_file():
            failures.append(f"source_authority_{label}_not_found:{path}")
            return path, {}
        body = path.read_bytes()
        observed_sha = _sha256_bytes(body)
        if observed_sha != resume_binding.get(sha_field):
            failures.append(
                f"source_authority_{label}_sha256_mismatch: expected "
                f"{resume_binding.get(sha_field)}, observed {observed_sha}"
            )
        try:
            document = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            failures.append(f"source_authority_{label}_not_canonical_json:{path}")
            return path, {}
        if not isinstance(document, dict):
            failures.append(f"source_authority_{label}_not_a_json_object:{path}")
            return path, {}
        return path, document

    auth_path, source_authorization = load_json_artifact(
        "source_stage_authorization_path",
        "source_stage_authorization_sha256",
        "authorization",
    )
    contract_path, source_contract = load_json_artifact(
        "source_invocation_run_contract_path",
        "source_invocation_run_contract_sha256",
        "invocation_contract",
    )

    if source_authorization:
        if source_authorization.get("schema_version") != AUTHORIZATION_SCHEMA:
            failures.append("source_authority_authorization_schema_mismatch")
        if source_authorization.get("authorization_status") != "AUTHORIZED":
            failures.append("source_authority_authorization_not_authorized")

    source_authorization_sha = str(resume_binding["source_stage_authorization_sha256"])
    if source_contract:
        if source_contract.get("schema_version") != RUN_CONTRACT_SCHEMA:
            failures.append("source_authority_run_contract_schema_mismatch")
        if source_contract.get("kind") != GOVERNED_CHECKPOINT_KIND:
            failures.append("source_authority_run_contract_kind_mismatch")
        if source_contract.get("stage_authorization_sha256") != source_authorization_sha:
            failures.append("source_authority_run_contract_authorization_sha256_mismatch")
        recorded_auth_path = (
            Path(str(source_contract.get("stage_authorization_path", ""))).expanduser().resolve()
        )
        if recorded_auth_path != auth_path:
            failures.append("source_authority_run_contract_authorization_path_mismatch")

        semantic_digest = governed_digest(source_contract)
        if source_contract.get("governed_run_contract_sha256") != semantic_digest:
            failures.append("source_authority_run_contract_semantic_digest_mismatch")
        if resume_binding.get("governed_run_contract_sha256") != semantic_digest:
            failures.append("source_authority_resume_run_contract_digest_mismatch")

        source_base_digest = base_governed_identity_sha256(source_contract)
        if resume_binding.get("source_base_governed_identity_digest") != source_base_digest:
            failures.append("source_authority_base_governed_identity_digest_mismatch")

        run_root = Path(str(source_contract.get("governed_run_root", ""))).expanduser().resolve()
        expected_invocation_root = invocation_directory(
            run_root,
            str(source_contract.get("stage", "")),
            source_authorization_sha,
        )
        if contract_path.name != GOVERNED_RUN_CONTRACT_FILENAME:
            failures.append("source_authority_run_contract_filename_mismatch")
        if contract_path.parent != expected_invocation_root:
            failures.append("source_authority_run_contract_location_mismatch")
        for field in ("invocation_root", "out_dir"):
            recorded = Path(str(source_contract.get(field, ""))).expanduser().resolve()
            if recorded != expected_invocation_root:
                failures.append(f"source_authority_run_contract_{field}_mismatch")
        source_samples = Path(str(source_contract.get("samples_dir", ""))).expanduser().resolve()
        if source_samples != expected_invocation_root / "samples":
            failures.append("source_authority_run_contract_samples_dir_mismatch")
        authorized_root = (
            Path(str(source_authorization.get("allowed_output_root", ""))).expanduser().resolve()
        )
        if authorized_root != run_root:
            failures.append("source_authority_authorized_run_root_mismatch")
        authorized_samples = (
            Path(str(source_authorization.get("allowed_samples_dir", ""))).expanduser().resolve()
        )
        if authorized_samples != run_root / "samples":
            failures.append("source_authority_authorized_samples_root_mismatch")

        # Hashing both artifacts proves their bytes, but does not prove that the two
        # independently loaded documents describe the same invocation. Cross-bind every
        # authorization field that the published contract records from Gate A.
        if source_authorization:
            for authorization_field, contract_field in (
                ("allowed_scope", "scope"),
                ("repository_branch", "trainer_branch"),
                ("trainer_head", "trainer_head"),
                ("trainer_execution_bundle_sha256", "trainer_execution_bundle_sha256"),
                ("launch_contract_sha256", "launch_contract_sha256"),
                ("exact_run_plan_sha256", "exact_run_plan_sha256"),
                ("pilot_owner_acceptance_sha256", "pilot_owner_acceptance_sha256"),
                ("canonical_cwd", "canonical_cwd"),
            ):
                if source_authorization.get(authorization_field) != source_contract.get(
                    contract_field
                ):
                    failures.append(
                        "source_authority_authorization_run_contract_mismatch:"
                        f"{authorization_field}!={contract_field}"
                    )
            if source_authorization.get("training_runtime") != source_contract.get(
                "runtime_fingerprint"
            ):
                failures.append("source_authority_authorization_run_contract_runtime_mismatch")
            if (source_authorization.get("resume") or {}) != (source_contract.get("resume") or {}):
                failures.append("source_authority_authorization_run_contract_resume_mismatch")

            source_scope = source_authorization.get("allowed_scope")
            expected_source_scope = {
                "stage_a": "STAGE_N",
                "stage_b": "STAGE_O",
            }.get(source_contract.get("stage"))
            if source_scope != expected_source_scope:
                failures.append("source_authority_source_stage_scope_mismatch")
            if transition == "A_TO_B" and source_scope != "STAGE_N":
                failures.append("source_authority_a_to_b_requires_stage_n_source_scope")

        sampler = source_contract.get("sampler_identity")
        if not isinstance(sampler, Mapping):
            failures.append("source_authority_run_contract_sampler_identity_missing")
        else:
            for binding_field, sampler_field in (
                ("source_active_stage", "stage"),
                ("source_sampler_seed", "sampler_seed"),
                ("source_permutation_identity", "permutation_identity"),
                ("source_range_start_position", "range_start_position"),
                (
                    "source_invocation_range_start_position",
                    "invocation_range_start_position",
                ),
                ("source_range_stop_position", "range_stop_position"),
            ):
                if resume_binding.get(binding_field) != sampler.get(sampler_field):
                    failures.append(f"source_authority_sampler_binding_mismatch:{binding_field}")

    checkpoint_path = Path(str(resume_binding["source_checkpoint_path"])).expanduser().resolve()
    if not checkpoint_path.is_file():
        failures.append(f"source_authority_checkpoint_not_found:{checkpoint_path}")
        checkpoint_sha = None
    else:
        checkpoint_sha = file_sha256(checkpoint_path)
        if checkpoint_sha != resume_binding.get("source_checkpoint_sha256"):
            failures.append("source_authority_checkpoint_sha256_mismatch")
    if checkpoint_path.parent != contract_path.parent:
        failures.append("source_authority_checkpoint_not_in_source_invocation_root")

    for current_field, source_field in (
        ("checkpoint_path", "source_checkpoint_path"),
        ("checkpoint_sha256", "source_checkpoint_sha256"),
        ("expected_step", "source_checkpoint_step"),
        ("stage", "source_checkpoint_stage"),
    ):
        if resume_binding.get(current_field) != resume_binding.get(source_field):
            failures.append(
                f"source_authority_resume_alias_mismatch:{current_field}!={source_field}"
            )

    if source_contract:
        if source_contract.get("stage") != resume_binding.get("source_checkpoint_stage"):
            failures.append("source_authority_checkpoint_stage_contract_mismatch")
        sampler_seed = int(resume_binding["source_sampler_seed"])
        range_stop = int(resume_binding["source_range_stop_position"])
        canonical_permutation = permutation_identity(
            str(resume_binding["source_active_stage"]), sampler_seed, range_stop
        )
        if resume_binding.get("source_permutation_identity") != canonical_permutation:
            failures.append("source_authority_permutation_identity_not_canonical")

    if transition == "A_TO_B":
        exact_a_to_b = {
            "source_checkpoint_stage": "stage_a",
            "source_active_stage": "stage_a",
            "source_sampler_seed": STAGE_A_SAMPLER_SEED,
            "source_range_start_position": 0,
            "source_range_stop_position": STAGE_A_TERMINAL_RANGE_STOP,
            "source_cursor": STAGE_A_TERMINAL_RANGE_STOP,
            "source_checkpoint_step": STAGE_A_STOP_STEP,
        }
        for field, expected in exact_a_to_b.items():
            if resume_binding.get(field) != expected:
                failures.append(
                    f"source_authority_a_to_b_endpoint_mismatch:{field}: expected {expected!r}, "
                    f"got {resume_binding.get(field)!r}"
                )

    return {
        "verified": not failures,
        "failures": list(dict.fromkeys(failures)),
        "source_authorization_path": str(auth_path),
        "source_authorization_sha256": source_authorization_sha,
        "source_authorization": source_authorization,
        "source_invocation_run_contract_path": str(contract_path),
        "source_invocation_run_contract_sha256": resume_binding.get(
            "source_invocation_run_contract_sha256"
        ),
        "source_invocation_run_contract": source_contract,
        "source_base_governed_identity_digest": (
            base_governed_identity_sha256(source_contract) if source_contract else None
        ),
        "source_checkpoint_path": str(checkpoint_path),
        "source_checkpoint_sha256": checkpoint_sha,
    }


def is_governed_checkpoint(ckpt: Mapping[str, Any]) -> bool:
    contract = ckpt.get("governed_run_contract")
    return isinstance(contract, Mapping) and contract.get("kind") == GOVERNED_CHECKPOINT_KIND


# R2 Part 7: fields that legitimately DIFFER across an intentional Stage-A -> Stage-B
# transition. Comparing these with the ordinary same-stage rule would reject the legal
# transition, so they are excluded only when stage_transition == "A_TO_B".
A_TO_B_PERMITTED_DIFFERENCES = (
    "stage",
    "scope",
    "stage_start_step",
    "stage_stop_step",
    # The active stage seed legitimately changes 20260832 -> 20260833 across the transition;
    # omitting it here would reject the legal A -> B handoff.
    "active_stage_sampler_seed",
)


def validate_governed_checkpoint_before_restore(
    ckpt: Mapping[str, Any],
    current_contract: Mapping[str, Any],
    *,
    expected_resume: Mapping[str, Any] | None = None,
    current_sampler_identity: Mapping[str, Any] | None = None,
    stage_transition: str | None = None,
    expected_global_step: int | None = None,
    verified_source_authority: Mapping[str, Any] | None = None,
    require_live_cuda_validation: bool = False,
) -> dict[str, Any]:
    """Validate metadata BEFORE any executable state is restored.

    R3: BASE_GOVERNED_IDENTITY must always match. INVOCATION_IDENTITY must match on a
    same-stage resume, but an explicitly authorized ``A_TO_B`` transition legitimately
    changes it, and is validated by the dedicated transition rule instead.
    """
    failures: list[str] = []
    if not is_governed_checkpoint(ckpt):
        return {
            "compatible": False,
            "failures": ["ungoverned_checkpoint_cannot_resume_a_governed_run"],
        }
    saved = ckpt["governed_run_contract"]
    saved_state = ckpt.get("governed_checkpoint_state") or {}
    a_to_b = stage_transition == "A_TO_B"
    failures.extend(
        validate_governed_checkpoint_state(
            saved_state,
            governed_run_contract=saved,
            checkpoint_global_step=ckpt.get("global_step"),
            require_live_cuda_validation=require_live_cuda_validation,
        )
    )
    failures.extend(validate_governed_checkpoint_resume_envelope(ckpt))

    # ---- source authority: every governed resume is continuity from immutable bytes ----
    exact_resume = (expected_resume or {}).get("mode") == "RESUME_EXACT_CHECKPOINT"
    if exact_resume:
        if not isinstance(verified_source_authority, Mapping) or not bool(
            verified_source_authority.get("verified")
        ):
            failures.append("verified_source_authority_required_before_restore")
        else:
            source = verified_source_authority.get("source_invocation_run_contract")
            if not isinstance(source, Mapping):
                failures.append("verified_source_invocation_run_contract_missing")
            else:
                source_base = base_governed_identity_sha256(source)
                if source_base != verified_source_authority.get(
                    "source_base_governed_identity_digest"
                ):
                    failures.append("verified_source_base_digest_internal_mismatch")
                if source_base != (expected_resume or {}).get(
                    "source_base_governed_identity_digest"
                ):
                    failures.append("resume_source_base_digest_mismatch")
                if base_governed_identity_sha256(current_contract) != source_base:
                    failures.append("current_base_identity_differs_from_verified_source")
                if base_governed_identity_sha256(saved) != source_base:
                    failures.append("checkpoint_base_identity_differs_from_verified_source")

                source_document = dict(source)
                source_document.pop("governed_run_contract_sha256", None)
                saved_document = dict(saved)
                saved_document.pop("governed_run_contract_sha256", None)
                if canonical_json_bytes(saved_document) != canonical_json_bytes(source_document):
                    failures.append("checkpoint_contract_differs_from_verified_source_artifact")

                if current_contract.get("governed_run_root") != source.get("governed_run_root"):
                    failures.append("current_governed_run_root_differs_from_verified_source")

            for state_field, resume_field in (
                ("active_stage", "source_active_stage"),
                ("active_stage_sampler_seed", "source_sampler_seed"),
                ("permutation_identity", "source_permutation_identity"),
                ("range_start_position", "source_range_start_position"),
                (
                    "invocation_range_start_position",
                    "source_invocation_range_start_position",
                ),
                ("range_stop_position", "source_range_stop_position"),
                ("cursor", "source_cursor"),
                ("global_step", "source_checkpoint_step"),
            ):
                if saved_state.get(state_field) != (expected_resume or {}).get(resume_field):
                    failures.append(f"checkpoint_state_differs_from_source_authority:{state_field}")
            if ckpt.get("global_step") != (expected_resume or {}).get("source_checkpoint_step"):
                failures.append("checkpoint_step_differs_from_source_authority")
            if saved.get("stage") != (expected_resume or {}).get("source_checkpoint_stage"):
                failures.append("checkpoint_stage_differs_from_source_authority")

    # ---- layer 1: BASE identity, always ----
    for field in BASE_GOVERNED_IDENTITY_FIELDS:
        if saved.get(field) != current_contract.get(field):
            failures.append(f"base_governed_identity_mismatch:{field}")
    if base_governed_identity_sha256(saved) != base_governed_identity_sha256(current_contract):
        failures.append("base_governed_identity_sha256_mismatch")

    # ---- layer 2: INVOCATION identity ----
    if a_to_b:
        # Legitimately different by design; the transition rule below is the authority.
        pass
    else:
        for field in SAME_STAGE_INVOCATION_MATCH_FIELDS:
            if saved.get(field) != current_contract.get(field):
                failures.append(f"invocation_identity_mismatch:{field}")

    saved_digest = ckpt.get("governed_run_contract_sha256") or saved.get(
        "governed_run_contract_sha256"
    )
    recomputed = governed_digest(saved)
    if saved_digest != recomputed:
        failures.append("governed_run_contract_digest_does_not_match_its_own_document")

    # ---- compile evidence must verify by schema, SHA and verdict ----
    if bool((saved.get("training") or {}).get("compile")):
        stored_evidence = saved_state.get("compile_evidence") or saved.get("compile_evidence")
        failures.extend(verify_compile_evidence_document(stored_evidence))

    # ---- sampler / transition ----
    saved_sampler = {
        "stage": saved_state.get("active_stage"),
        "sampler_seed": saved_state.get("active_stage_sampler_seed"),
        "permutation_identity": saved_state.get("permutation_identity"),
        "range_start_position": saved_state.get("range_start_position"),
        "invocation_range_start_position": saved_state.get("invocation_range_start_position"),
        "range_stop_position": saved_state.get("range_stop_position"),
        "cursor": saved_state.get("cursor"),
    }
    if a_to_b:
        failures.extend(
            validate_stage_a_to_b_transition(
                saved,
                saved_state,
                source_binding=expected_resume,
            )
        )
    else:
        if current_sampler_identity is None:
            failures.append(
                "governed_resume_requires_the_expected_sampler_identity_before_restoration"
            )
        else:
            failures.extend(
                validate_same_stage_resume(
                    saved_sampler,
                    current_sampler_identity,
                    expected_global_step=expected_global_step,
                    saved_global_step=saved_state.get("global_step"),
                )
            )

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

    failures = list(dict.fromkeys(failures))
    return {
        "compatible": not failures,
        "failures": failures,
        "stage_transition": stage_transition,
        "saved_sampler_state": saved_sampler,
        "base_governed_identity_sha256": base_governed_identity_sha256(saved),
        "invocation_identity_sha256": invocation_identity_sha256(saved),
        "checkpoint_governed_run_contract_sha256": recomputed,
        "current_governed_run_contract_sha256": governed_digest(current_contract),
    }


# --------------------------------------------------------------- Part 8: invocation layout


def invocation_directory(run_root: str | Path, stage: str, authorization_sha256: str) -> Path:
    """Return the one deterministic directory for an authenticated invocation.

    Each separately authorized invocation publishes its own single-publication run contract,
    so a legitimate Stage-B or resume process never collides with the artifact the previous
    invocation already wrote. The complete authorization digest is used: the directory is an
    identity artifact, so truncation is unnecessary.
    """
    require(stage in STAGE_ORDER, f"unknown stage {stage!r}")
    require(
        _is_sha256(authorization_sha256),
        f"invocation directory requires a sha256 authorization id, got {authorization_sha256!r}",
    )
    return Path(run_root).expanduser().resolve() / f"{stage}_{authorization_sha256}"


def invocation_layout(
    run_root: str | Path, stage: str, authorization_sha256: str
) -> dict[str, str]:
    """Derive every invocation-scoped output from the authority-bound run root."""
    root = Path(run_root).expanduser().resolve()
    invocation_root = invocation_directory(root, stage, authorization_sha256)
    return {
        "governed_run_root": str(root),
        "invocation_root": str(invocation_root),
        "out_dir": str(invocation_root),
        "samples_dir": str(invocation_root / "samples"),
    }


def validate_invocation_output_binding(
    args: argparse.Namespace,
    *,
    run_root: Any,
    authorized_samples_dir: Any,
    stage: str,
    authorization_sha256: str,
) -> dict[str, Any]:
    """Validate operator inputs, then return the canonical derived invocation paths.

    ``allowed_output_root`` authorizes the production RUN root. The authorization cannot
    embed an invocation path derived from its own eventual file hash without becoming
    circular, so Gate A accepts the bound run-root spelling as an input alias and replaces it
    with the derived invocation path before main constructs any output consumer.
    """
    failures: list[str] = []
    if run_root in (None, ""):
        return {"failures": ["allowed_output_root_missing"], "layout": None}
    layout = invocation_layout(str(run_root), stage, authorization_sha256)
    root = Path(layout["governed_run_root"])
    invocation_root = Path(layout["invocation_root"])
    declared_samples = root / "samples"

    requested_out = Path(str(getattr(args, "out_dir", ""))).expanduser().resolve()
    if requested_out not in (root, invocation_root):
        failures.append(
            "out_dir must name the authorized governed run root or its deterministic "
            f"invocation root: got {requested_out}"
        )

    if authorized_samples_dir not in (None, ""):
        bound_samples = Path(str(authorized_samples_dir)).expanduser().resolve()
        if bound_samples != declared_samples:
            failures.append(
                "allowed_samples_dir must be the non-circular run-root samples placeholder "
                f"{declared_samples}, got {bound_samples}"
            )

    requested_samples = Path(str(getattr(args, "samples_dir", ""))).expanduser().resolve()
    derived_samples = Path(layout["samples_dir"])
    if requested_samples not in (declared_samples, derived_samples):
        failures.append(
            "samples_dir must name the authorized run-root placeholder or deterministic "
            f"invocation samples path: got {requested_samples}"
        )
    try:
        requested_samples.relative_to(root)
    except ValueError:
        failures.append(f"samples_dir escapes governed run root {root}: {requested_samples}")
    return {"failures": list(dict.fromkeys(failures)), "layout": layout}


def publish_invocation_run_contract(
    contract_or_run_root: Mapping[str, Any] | str | Path,
    contract: Mapping[str, Any] | None = None,
    *,
    gate_a: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish exactly once at the contract-derived invocation path.

    The two-positional-argument form is retained only for compatibility with local callers;
    the supplied root must already equal the root embedded by Gate A and therefore cannot
    become an independent identity source. The real trainer uses the one-document form.
    """
    if contract is None:
        require(isinstance(contract_or_run_root, Mapping), "run contract document is required")
        document = dict(contract_or_run_root)
        supplied_root = None
    else:
        document = dict(contract)
        supplied_root = Path(str(contract_or_run_root)).expanduser().resolve()

    stage = str(document["stage"])
    bound_root = Path(str(document.get("governed_run_root", ""))).expanduser().resolve()
    require(str(document.get("governed_run_root", "")), "run contract has no governed_run_root")
    if supplied_root is not None:
        require(
            supplied_root == bound_root,
            f"caller run root {supplied_root} differs from authority-bound root {bound_root}",
        )
    if gate_a is not None:
        require(
            Path(str(gate_a.get("governed_run_root", ""))).expanduser().resolve() == bound_root,
            "Gate-A run root differs from the normalized contract",
        )

    directory = invocation_directory(bound_root, stage, str(document["stage_authorization_sha256"]))
    require(
        Path(str(document.get("invocation_root", ""))).expanduser().resolve() == directory,
        "run contract invocation_root is not deterministically derived",
    )
    require(
        Path(str(document.get("out_dir", ""))).expanduser().resolve() == directory,
        "run contract out_dir is not the deterministic invocation root",
    )
    require(
        Path(str(document.get("samples_dir", ""))).expanduser().resolve() == directory / "samples",
        "run contract samples_dir is not derived from the invocation root",
    )

    published = publish_governed_run_contract(directory, document)
    published["run_root"] = str(bound_root)
    published["invocation_dir"] = str(directory)
    published["base_governed_identity_sha256"] = base_governed_identity_sha256(document)
    published["invocation_identity_sha256"] = invocation_identity_sha256(document)
    return published


# --------------------------------------------------------------- Part 9: Stage-N/O chain

STAGE_N_RESULT_SCHEMA = "petitgpt-stage-n-result-v1"
STAGE_N_RUNTIME_ARTIFACT_SCHEMA = "petitgpt-stage-n-runtime-fingerprint-v1"
STAGE_N_RUNTIME_ARTIFACT_KIND = "STAGE_N_RUNTIME_FINGERPRINT"
STAGE_N_CHECK_RESULT_SCHEMA = "petitgpt-stage-n-check-result-v1"
STAGE_N_SMOKE_RESULT_KIND = "STAGE_N_SMOKE_CHECK"
STAGE_N_RESUME_RESULT_KIND = "STAGE_N_RESUME_CHECK"
STAGE_N_RESUME_EVIDENCE_SCHEMA = "petitgpt-stage-n-resume-evidence-v1"
STAGE_N_RESUME_EVIDENCE_KIND = "STAGE_N_VERIFIED_RESUME_EXECUTION"
STAGE_N_RUNTIME_FILENAME = "STAGE_N_RUNTIME_FINGERPRINT.json"
STAGE_N_SMOKE_RESULT_FILENAME = "STAGE_N_SMOKE_RESULT.json"
STAGE_N_RESUME_RESULT_FILENAME = "STAGE_N_RESUME_RESULT.json"
STAGE_N_RESUME_EVIDENCE_FILENAME = "STAGE_N_RESUME_EVIDENCE.json"
STAGE_N_RESULT_REQUIRED_FIELDS = (
    "schema_version",
    "status",
    "stage",
    "scope",
    "stage_authorization_path",
    "stage_authorization_sha256",
    "launch_contract_sha256",
    "exact_plan_path",
    "exact_run_plan_sha256",
    "pilot_acceptance_path",
    "pilot_owner_acceptance_sha256",
    "trainer_head",
    "trainer_execution_bundle_sha256",
    "governed_run_contract_path",
    "governed_run_contract_artifact_sha256",
    "governed_run_contract_sha256",
    "base_governed_identity_sha256",
    "final_checkpoint_path",
    "final_checkpoint_sha256",
    "final_checkpoint_step",
    "runtime_fingerprint",
    "runtime_fingerprint_path",
    "runtime_fingerprint_sha256",
    "runtime_fingerprint_artifact_sha256",
    "gpu_uuid",
    "gpu_pci_bus_id",
    "num_workers",
    "smoke_results_path",
    "smoke_results_sha256",
    "smoke_results",
    "resume_results_path",
    "resume_results_sha256",
    "resume_results",
    "final_sampler_permutation_identity",
    "final_sampler_range_start_position",
    "final_sampler_invocation_range_start_position",
    "final_sampler_range_stop_position",
    "final_sampler_cursor",
)

# A Stage-O authorization must carry the accepted Stage-N chain. Without these it fails.
STAGE_O_REQUIRED_CHAIN_FIELDS = (
    "accepted_stage_n_result_path",
    "accepted_stage_n_result_sha256",
    "stage_n_owner_acceptance_path",
    "stage_n_owner_acceptance_sha256",
    "stage_n_authorization_sha256",
    "stage_n_governed_run_contract_sha256",
    "stage_n_governed_run_contract_artifact_sha256",
    "stage_n_runtime_fingerprint",
    "stage_n_runtime_fingerprint_sha256",
    "stage_n_gpu_uuid",
    "stage_n_gpu_pci_bus_id",
    "stage_n_trainer_head",
    "stage_n_trainer_execution_bundle_sha256",
    "stage_n_exact_run_plan_sha256",
    # R2 Part 14: the exact checkpoint Stage O will resume from.
    "stage_n_final_checkpoint_path",
    "stage_n_final_checkpoint_sha256",
    "stage_n_final_checkpoint_step",
    "stage_n_runtime_fingerprint_path",
    "stage_n_runtime_fingerprint_artifact_sha256",
)

# Runtime fields Stage O must match against the ACCEPTED Stage-N result, not merely against
# its own authorization. Changing both the authorization and the runtime cannot evade this.
STAGE_N_O_RUNTIME_COMPARISON_FIELDS = (
    "gpu_uuid",
    "gpu_pci_bus_id",
    "gpu_name",
    "visible_cuda_device_count",
    "total_vram_bytes",
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


def _paths_equal(recorded: Any, actual: str | Path) -> bool:
    if not isinstance(recorded, str) or not recorded.strip():
        return False
    return Path(recorded).resolve() == Path(actual).resolve()


def _load_json_mapping_artifact(
    path: str | Path, *, label: str, failures: list[str]
) -> dict[str, Any] | None:
    candidate = Path(path)
    if not candidate.is_file():
        failures.append(f"{label}_not_found:{candidate}")
        return None
    try:
        value = json.loads(candidate.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        failures.append(f"{label}_not_valid_json:{type(exc).__name__}")
        return None
    if not isinstance(value, Mapping):
        failures.append(f"{label}_json_not_an_object")
        return None
    return dict(value)


def _atomic_publish_json(path: str | Path, document: Mapping[str, Any]) -> dict[str, Any]:
    """Publish one canonical JSON artifact exactly once."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    require(not target.exists(), f"artifact already exists at {target}")
    body = canonical_json_bytes(document)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)
    return {"path": str(target), "sha256": _sha256_bytes(body), "atomic": True}


def _publish_or_verify_identical_json(
    path: str | Path, document: Mapping[str, Any]
) -> dict[str, Any]:
    """Publish canonical JSON, or accept an exact prior publication after a retry.

    A process can die after publishing one member of the Stage-N evidence/result pair.  A
    retry must be able to finish that pair, but it must never bless a pre-existing document
    whose bytes differ from the document reconstructed from the still-authenticated inputs.
    """
    target = Path(path)
    expected = canonical_json_bytes(document)
    if target.exists():
        require(target.is_file(), f"artifact path is not a file: {target}")
        observed = target.read_bytes()
        require(
            observed == expected,
            f"contradictory pre-existing artifact at {target}",
        )
        return {
            "path": str(target),
            "sha256": _sha256_bytes(observed),
            "atomic": True,
            "reused": True,
        }
    published = _atomic_publish_json(target, document)
    published["reused"] = False
    return published


def _checkpoint_values_equal(left: Any, right: Any) -> bool:
    """Exact recursive equality for deserialized optimizer/model/RNG state."""
    try:
        import torch

        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            return (
                isinstance(left, torch.Tensor)
                and isinstance(right, torch.Tensor)
                and left.dtype == right.dtype
                and tuple(left.shape) == tuple(right.shape)
                and torch.equal(left, right)
            )
    except ImportError:  # pragma: no cover - checkpoint validation already needs torch
        pass
    try:
        import numpy as np

        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            return (
                isinstance(left, np.ndarray)
                and isinstance(right, np.ndarray)
                and left.dtype == right.dtype
                and left.shape == right.shape
                and bool(np.array_equal(left, right))
            )
    except ImportError:  # pragma: no cover - governed training requires NumPy
        pass
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        if set(left) != set(right):
            return False
        return all(_checkpoint_values_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            return False
        return all(_checkpoint_values_equal(a, b) for a, b in zip(left, right, strict=True))
    if type(left) is not type(right):
        return False
    try:
        result = left == right
    except Exception:  # noqa: BLE001 - unequal opaque state must fail closed
        return False
    return result if isinstance(result, bool) else bool(result)


def stage_n_runtime_artifact_document(runtime: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(runtime)
    fingerprint_sha = raw.get("runtime_fingerprint_sha256") or runtime_fingerprint_sha256(raw)
    require(
        fingerprint_sha == runtime_fingerprint_sha256(raw),
        "runtime fingerprint self-hash is invalid",
    )
    return {
        "schema_version": STAGE_N_RUNTIME_ARTIFACT_SCHEMA,
        "kind": STAGE_N_RUNTIME_ARTIFACT_KIND,
        "runtime_fingerprint": raw,
        "runtime_fingerprint_sha256": fingerprint_sha,
    }


def publish_stage_n_runtime_artifact(
    invocation_dir: str | Path, runtime: Mapping[str, Any]
) -> dict[str, Any]:
    return _atomic_publish_json(
        Path(invocation_dir) / STAGE_N_RUNTIME_FILENAME,
        stage_n_runtime_artifact_document(runtime),
    )


def stage_n_check_result_document(
    *,
    kind: str,
    stage_authorization_sha256: str,
    governed_run_contract_sha256: str,
    checkpoint_path: str | Path,
    checkpoint_step: int,
    runtime_fingerprint_sha256: str,
    evidence_artifact_path: str | Path,
) -> dict[str, Any]:
    """Build a structured smoke/resume result from actual local evidence bytes."""
    require(
        kind in (STAGE_N_SMOKE_RESULT_KIND, STAGE_N_RESUME_RESULT_KIND),
        f"unknown Stage-N check kind {kind!r}",
    )
    checkpoint = Path(checkpoint_path)
    evidence = Path(evidence_artifact_path)
    require(checkpoint.is_file(), f"Stage-N check checkpoint not found: {checkpoint}")
    require(evidence.is_file(), f"Stage-N check evidence not found: {evidence}")
    return {
        "schema_version": STAGE_N_CHECK_RESULT_SCHEMA,
        "kind": kind,
        "status": "PASS",
        "stage_authorization_sha256": stage_authorization_sha256,
        "governed_run_contract_sha256": governed_run_contract_sha256,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_step": int(checkpoint_step),
        "runtime_fingerprint_sha256": runtime_fingerprint_sha256,
        "evidence_artifact_path": str(evidence),
        "evidence_artifact_sha256": file_sha256(evidence),
    }


STAGE_N_CHECK_RESULT_REQUIRED_FIELDS = (
    "schema_version",
    "kind",
    "status",
    "stage_authorization_sha256",
    "governed_run_contract_sha256",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_step",
    "runtime_fingerprint_sha256",
    "evidence_artifact_path",
    "evidence_artifact_sha256",
)


def _validate_stage_n_resume_evidence(
    evidence: Mapping[str, Any],
    *,
    expected_authorization_sha256: str,
    expected_governed_run_contract_sha256: str,
    expected_checkpoint_path: str | Path | None,
    expected_checkpoint_sha256: str | None,
    expected_checkpoint_step: int | None,
    expected_authorization_path: str | Path | None,
    expected_governed_run_contract_path: str | Path | None,
) -> list[str]:
    """Reopen every source/current artifact named by generated resume evidence."""
    failures: list[str] = []
    required = (
        "schema_version",
        "kind",
        "status",
        "source_authority_verified",
        "source_checkpoint_bytes_verified",
        "source_stage_authorization_path",
        "source_stage_authorization_sha256",
        "source_governed_run_contract_path",
        "source_governed_run_contract_artifact_sha256",
        "source_governed_run_contract_sha256",
        "source_checkpoint_path",
        "source_checkpoint_sha256",
        "source_checkpoint_step",
        "resume_stage_authorization_path",
        "resume_stage_authorization_sha256",
        "resume_governed_run_contract_path",
        "resume_governed_run_contract_artifact_sha256",
        "resume_governed_run_contract_sha256",
        "resume_final_checkpoint_path",
        "resume_final_checkpoint_sha256",
        "resume_start_step",
        "completed_step",
        "optimizer_updates",
    )
    for field in required:
        if evidence.get(field) in (None, ""):
            failures.append(f"stage_n_resume_evidence_missing_field:{field}")
    for field, expected in (
        ("schema_version", STAGE_N_RESUME_EVIDENCE_SCHEMA),
        ("kind", STAGE_N_RESUME_EVIDENCE_KIND),
        ("status", "PASS"),
        ("source_authority_verified", True),
        ("source_checkpoint_bytes_verified", True),
        ("source_stage_authorization_sha256", expected_authorization_sha256),
        ("source_governed_run_contract_sha256", expected_governed_run_contract_sha256),
        ("source_checkpoint_sha256", expected_checkpoint_sha256),
        ("source_checkpoint_step", expected_checkpoint_step),
        ("resume_start_step", expected_checkpoint_step),
        ("completed_step", expected_checkpoint_step),
        ("optimizer_updates", 0),
    ):
        if expected is not None and evidence.get(field) != expected:
            failures.append(f"stage_n_resume_evidence_{field}_mismatch")
    if expected_authorization_path is not None and not _paths_equal(
        evidence.get("source_stage_authorization_path"), expected_authorization_path
    ):
        failures.append("stage_n_resume_evidence_source_authorization_path_mismatch")
    if expected_governed_run_contract_path is not None and not _paths_equal(
        evidence.get("source_governed_run_contract_path"),
        expected_governed_run_contract_path,
    ):
        failures.append("stage_n_resume_evidence_source_grc_path_mismatch")
    if expected_checkpoint_path is not None and not _paths_equal(
        evidence.get("source_checkpoint_path"), expected_checkpoint_path
    ):
        failures.append("stage_n_resume_evidence_source_checkpoint_path_mismatch")

    loaded: dict[str, dict[str, Any] | None] = {}
    for label, path_field, sha_field in (
        (
            "source_authorization",
            "source_stage_authorization_path",
            "source_stage_authorization_sha256",
        ),
        (
            "source_grc",
            "source_governed_run_contract_path",
            "source_governed_run_contract_artifact_sha256",
        ),
        (
            "resume_authorization",
            "resume_stage_authorization_path",
            "resume_stage_authorization_sha256",
        ),
        (
            "resume_grc",
            "resume_governed_run_contract_path",
            "resume_governed_run_contract_artifact_sha256",
        ),
    ):
        value = evidence.get(path_field)
        if not isinstance(value, str) or not value.strip():
            continue
        path = Path(value)
        loaded[label] = _load_json_mapping_artifact(
            path, label=f"stage_n_resume_evidence_{label}", failures=failures
        )
        if path.is_file() and file_sha256(path) != evidence.get(sha_field):
            failures.append(f"stage_n_resume_evidence_{label}_sha256_mismatch")

    source_auth = loaded.get("source_authorization")
    source_grc = loaded.get("source_grc")
    resume_auth = loaded.get("resume_authorization")
    resume_grc = loaded.get("resume_grc")
    for label, auth in (("source", source_auth), ("resume", resume_auth)):
        if auth is not None:
            if auth.get("schema_version") != AUTHORIZATION_SCHEMA:
                failures.append(f"stage_n_resume_evidence_{label}_authorization_schema_mismatch")
            if auth.get("authorization_status") != "AUTHORIZED":
                failures.append(f"stage_n_resume_evidence_{label}_authorization_not_authorized")
            if auth.get("allowed_scope") != "STAGE_N":
                failures.append(f"stage_n_resume_evidence_{label}_authorization_scope_mismatch")
    if source_grc is not None:
        for field, expected in (
            ("schema_version", RUN_CONTRACT_SCHEMA),
            ("kind", GOVERNED_CHECKPOINT_KIND),
            ("governed", True),
            ("stage", "stage_a"),
            ("scope", "STAGE_N"),
        ):
            if source_grc.get(field) != expected:
                failures.append(f"stage_n_resume_evidence_source_grc_{field}_mismatch")
        if governed_digest(source_grc) != expected_governed_run_contract_sha256:
            failures.append("stage_n_resume_evidence_source_grc_semantic_digest_mismatch")
        if source_grc.get("governed_run_contract_sha256") != governed_digest(source_grc):
            failures.append("stage_n_resume_evidence_source_grc_self_digest_mismatch")
        if source_auth is not None:
            source_auth_path = Path(str(evidence["source_stage_authorization_path"]))
            if source_grc.get("stage_authorization_sha256") != file_sha256(source_auth_path):
                failures.append("stage_n_resume_evidence_source_grc_authorization_sha_mismatch")
            if not _paths_equal(source_grc.get("stage_authorization_path"), source_auth_path):
                failures.append("stage_n_resume_evidence_source_grc_authorization_path_mismatch")

    resume_binding: Mapping[str, Any] | None = None
    if resume_auth is not None and resume_grc is not None:
        resume_auth_path = Path(str(evidence["resume_stage_authorization_path"]))
        resume_auth_sha = file_sha256(resume_auth_path)
        resume_root = resume_auth.get("allowed_output_root")
        if not isinstance(resume_root, str) or not resume_root.strip():
            failures.append("stage_n_resume_evidence_resume_run_root_missing")
        else:
            resume_invocation = invocation_directory(resume_root, "stage_a", resume_auth_sha)
            resume_grc_path = Path(str(evidence["resume_governed_run_contract_path"]))
            resume_checkpoint_path = Path(str(evidence["resume_final_checkpoint_path"]))
            if (
                resume_grc_path.resolve()
                != (resume_invocation / GOVERNED_RUN_CONTRACT_FILENAME).resolve()
            ):
                failures.append("stage_n_resume_evidence_resume_grc_topology_mismatch")
            if (
                expected_checkpoint_step is not None
                and resume_checkpoint_path.resolve()
                != (resume_invocation / f"step_{expected_checkpoint_step:06d}.pt").resolve()
            ):
                failures.append("stage_n_resume_evidence_resume_checkpoint_topology_mismatch")
        if resume_grc.get("stage_authorization_sha256") != resume_auth_sha:
            failures.append("stage_n_resume_evidence_resume_grc_authorization_sha_mismatch")
        if not _paths_equal(resume_grc.get("stage_authorization_path"), resume_auth_path):
            failures.append("stage_n_resume_evidence_resume_grc_authorization_path_mismatch")
        for field, expected in (
            ("schema_version", RUN_CONTRACT_SCHEMA),
            ("kind", GOVERNED_CHECKPOINT_KIND),
            ("governed", True),
            ("stage", "stage_a"),
            ("scope", "STAGE_N"),
        ):
            if resume_grc.get(field) != expected:
                failures.append(f"stage_n_resume_evidence_resume_grc_{field}_mismatch")
        if resume_grc.get("governed_run_contract_sha256") != governed_digest(resume_grc):
            failures.append("stage_n_resume_evidence_resume_grc_self_digest_mismatch")
        if evidence.get("resume_governed_run_contract_sha256") != governed_digest(resume_grc):
            failures.append("stage_n_resume_evidence_resume_grc_semantic_digest_mismatch")
        resume_binding = resume_auth.get("resume")
        if not isinstance(resume_binding, Mapping):
            failures.append("stage_n_resume_evidence_resume_binding_missing")
            resume_binding = None
        else:
            for field, expected in (
                ("checkpoint_path", expected_checkpoint_path),
                ("checkpoint_sha256", expected_checkpoint_sha256),
                ("expected_step", expected_checkpoint_step),
            ):
                observed = resume_binding.get(field)
                if field == "checkpoint_path" and expected is not None:
                    if not _paths_equal(observed, expected):
                        failures.append(f"stage_n_resume_evidence_resume_binding_{field}_mismatch")
                elif expected is not None and observed != expected:
                    failures.append(f"stage_n_resume_evidence_resume_binding_{field}_mismatch")
            if resume_grc.get("resume") != resume_binding:
                failures.append("stage_n_resume_evidence_resume_auth_grc_binding_mismatch")
            verified_current_source = verify_resume_source_authority(
                resume_binding, transition=None
            )
            failures.extend(
                f"stage_n_resume_evidence:{failure}"
                for failure in verified_current_source["failures"]
            )
            for observed, expected, label in (
                (
                    verified_current_source.get("source_authorization_path"),
                    expected_authorization_path,
                    "source_authorization_path",
                ),
                (
                    verified_current_source.get("source_invocation_run_contract_path"),
                    expected_governed_run_contract_path,
                    "source_grc_path",
                ),
                (
                    verified_current_source.get("source_checkpoint_path"),
                    expected_checkpoint_path,
                    "source_checkpoint_path",
                ),
            ):
                if expected is not None and not _paths_equal(observed, expected):
                    failures.append(f"stage_n_resume_evidence_verified_{label}_mismatch")

    checkpoints: dict[str, Mapping[str, Any]] = {}
    for label, path_field, sha_field in (
        ("source_checkpoint", "source_checkpoint_path", "source_checkpoint_sha256"),
        (
            "resume_final_checkpoint",
            "resume_final_checkpoint_path",
            "resume_final_checkpoint_sha256",
        ),
    ):
        value = evidence.get(path_field)
        if not isinstance(value, str) or not value.strip():
            continue
        path = Path(value)
        if not path.is_file():
            failures.append(f"stage_n_resume_evidence_{label}_not_found:{path}")
            continue
        if file_sha256(path) != evidence.get(sha_field):
            failures.append(f"stage_n_resume_evidence_{label}_sha256_mismatch")
        try:
            import torch

            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001 - evidence must fail closed
            failures.append(f"stage_n_resume_evidence_{label}_not_loadable:{type(exc).__name__}")
            continue
        if not isinstance(checkpoint, Mapping):
            failures.append(f"stage_n_resume_evidence_{label}_not_mapping")
            continue
        checkpoints[label] = checkpoint

    for label, contract in (
        ("source_checkpoint", source_grc),
        ("resume_final_checkpoint", resume_grc),
    ):
        checkpoint = checkpoints.get(label)
        if checkpoint is None or contract is None:
            continue
        if (
            checkpoint.get("kind") != GOVERNED_CHECKPOINT_KIND
            or checkpoint.get("global_step") != expected_checkpoint_step
        ):
            failures.append(f"stage_n_resume_evidence_{label}_semantics_mismatch")
        embedded = checkpoint.get("governed_run_contract")
        if not isinstance(embedded, Mapping):
            failures.append(f"stage_n_resume_evidence_{label}_grc_missing")
        else:
            expected_document = dict(contract)
            expected_document.pop("governed_run_contract_sha256", None)
            embedded_document = dict(embedded)
            embedded_document.pop("governed_run_contract_sha256", None)
            if embedded_document != expected_document:
                failures.append(f"stage_n_resume_evidence_{label}_grc_not_exact")
        if checkpoint.get("governed_run_contract_sha256") != governed_digest(contract):
            failures.append(f"stage_n_resume_evidence_{label}_grc_digest_mismatch")
        failures.extend(
            f"stage_n_resume_evidence_{label}:{failure}"
            for failure in validate_governed_checkpoint_state(
                checkpoint.get("governed_checkpoint_state"),
                governed_run_contract=contract,
                checkpoint_global_step=expected_checkpoint_step,
            )
        )
        failures.extend(
            f"stage_n_resume_evidence_{label}:{failure}"
            for failure in validate_governed_checkpoint_resume_envelope(checkpoint)
        )

    source_checkpoint = checkpoints.get("source_checkpoint")
    resume_checkpoint = checkpoints.get("resume_final_checkpoint")
    if source_grc is not None and resume_grc is not None:
        if base_governed_identity_sha256(source_grc) != base_governed_identity_sha256(resume_grc):
            failures.append("stage_n_resume_evidence_base_identity_changed")
    if source_checkpoint is not None and resume_checkpoint is not None:
        # The second phase is specifically a zero-update proof. Model and optimizer state,
        # RNG streams, completed milestone prefixes and committed cursor may not move. The
        # new authorization/GRC and the invocation range start are authenticated separately.
        for field in ("model", "optim", "scaler", "run_contract", "data_contract"):
            if field not in source_checkpoint or field not in resume_checkpoint:
                failures.append(f"stage_n_resume_evidence_checkpoint_missing:{field}")
            elif not _checkpoint_values_equal(source_checkpoint[field], resume_checkpoint[field]):
                failures.append(f"stage_n_resume_evidence_zero_update_mismatch:{field}")
        for field in ("global_step", "local_step"):
            if source_checkpoint.get(field) != resume_checkpoint.get(field):
                failures.append(f"stage_n_resume_evidence_zero_update_mismatch:{field}")

        source_state = source_checkpoint.get("governed_checkpoint_state")
        resume_state = resume_checkpoint.get("governed_checkpoint_state")
        if isinstance(source_state, Mapping) and isinstance(resume_state, Mapping):

            def sampler_identity_from_dynamic(
                state: Mapping[str, Any],
            ) -> dict[str, Any]:
                return {
                    "stage": state.get("active_stage"),
                    "sampler_seed": state.get("active_stage_sampler_seed"),
                    "permutation_identity": state.get("permutation_identity"),
                    "range_start_position": state.get("range_start_position"),
                    "invocation_range_start_position": state.get("invocation_range_start_position"),
                    "range_stop_position": state.get("range_stop_position"),
                    "cursor": state.get("cursor"),
                }

            failures.extend(
                f"stage_n_resume_evidence:{failure}"
                for failure in validate_same_stage_resume(
                    sampler_identity_from_dynamic(source_state),
                    sampler_identity_from_dynamic(resume_state),
                    expected_global_step=expected_checkpoint_step,
                    saved_global_step=source_checkpoint.get("global_step"),
                )
            )
            if isinstance(resume_binding, Mapping):
                for state_field, binding_field in (
                    ("active_stage", "source_active_stage"),
                    ("active_stage_sampler_seed", "source_sampler_seed"),
                    ("permutation_identity", "source_permutation_identity"),
                    ("range_start_position", "source_range_start_position"),
                    (
                        "invocation_range_start_position",
                        "source_invocation_range_start_position",
                    ),
                    ("range_stop_position", "source_range_stop_position"),
                    ("cursor", "source_cursor"),
                    ("global_step", "source_checkpoint_step"),
                ):
                    if source_state.get(state_field) != resume_binding.get(binding_field):
                        failures.append(
                            f"stage_n_resume_evidence_source_state_binding_mismatch:{state_field}"
                        )
            for label, state, contract in (
                ("source", source_state, source_grc),
                ("resume", resume_state, resume_grc),
            ):
                sampler = (contract or {}).get("sampler_identity")
                if not isinstance(sampler, Mapping):
                    failures.append(f"stage_n_resume_evidence_{label}_grc_sampler_missing")
                    continue
                for state_field, sampler_field in (
                    ("active_stage", "stage"),
                    ("active_stage_sampler_seed", "sampler_seed"),
                    ("permutation_identity", "permutation_identity"),
                    ("range_start_position", "range_start_position"),
                    (
                        "invocation_range_start_position",
                        "invocation_range_start_position",
                    ),
                    ("range_stop_position", "range_stop_position"),
                ):
                    if state.get(state_field) != sampler.get(sampler_field):
                        failures.append(
                            f"stage_n_resume_evidence_{label}_state_grc_sampler_mismatch:"
                            f"{state_field}"
                        )
            for field in (
                "rng_state",
                "completed_evaluation_milestones",
                "completed_checkpoint_milestones",
            ):
                if not _checkpoint_values_equal(source_state.get(field), resume_state.get(field)):
                    failures.append(f"stage_n_resume_evidence_zero_update_mismatch:{field}")
    return list(dict.fromkeys(failures))


def validate_stage_n_check_result(
    document: Mapping[str, Any] | None,
    *,
    expected_kind: str,
    expected_authorization_sha256: str,
    expected_governed_run_contract_sha256: str,
    expected_runtime_fingerprint_sha256: str,
    expected_checkpoint_path: str | Path | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_checkpoint_step: int | None = None,
    expected_authorization_path: str | Path | None = None,
    expected_governed_run_contract_path: str | Path | None = None,
) -> list[str]:
    label = "smoke" if expected_kind == STAGE_N_SMOKE_RESULT_KIND else "resume"
    if not isinstance(document, Mapping):
        return [f"stage_n_{label}_result_missing_or_malformed"]
    failures = [
        f"stage_n_{label}_result_missing_field:{field}"
        for field in STAGE_N_CHECK_RESULT_REQUIRED_FIELDS
        if document.get(field) in (None, "")
    ]
    expected = {
        "schema_version": STAGE_N_CHECK_RESULT_SCHEMA,
        "kind": expected_kind,
        "status": "PASS",
        "stage_authorization_sha256": expected_authorization_sha256,
        "governed_run_contract_sha256": expected_governed_run_contract_sha256,
        "runtime_fingerprint_sha256": expected_runtime_fingerprint_sha256,
    }
    for field, value in expected.items():
        if document.get(field) != value:
            failures.append(f"stage_n_{label}_result_{field}_mismatch")
    for field in (
        "stage_authorization_sha256",
        "governed_run_contract_sha256",
        "checkpoint_sha256",
        "runtime_fingerprint_sha256",
        "evidence_artifact_sha256",
    ):
        if not _is_sha256(document.get(field)):
            failures.append(f"stage_n_{label}_result_malformed_sha256:{field}")
    step = document.get("checkpoint_step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        failures.append(f"stage_n_{label}_result_checkpoint_step_invalid")
    if expected_checkpoint_path is not None and not _paths_equal(
        document.get("checkpoint_path"), expected_checkpoint_path
    ):
        failures.append(f"stage_n_{label}_result_checkpoint_path_mismatch")
    if (
        expected_checkpoint_sha256 is not None
        and document.get("checkpoint_sha256") != expected_checkpoint_sha256
    ):
        failures.append(f"stage_n_{label}_result_checkpoint_sha256_parent_mismatch")
    if expected_checkpoint_step is not None and document.get("checkpoint_step") != int(
        expected_checkpoint_step
    ):
        failures.append(f"stage_n_{label}_result_checkpoint_step_mismatch")
    for prefix in ("checkpoint", "evidence_artifact"):
        path_value = document.get(f"{prefix}_path")
        if isinstance(path_value, str) and path_value.strip():
            path = Path(path_value)
            if not path.is_file():
                failures.append(f"stage_n_{label}_result_{prefix}_not_found:{path}")
            elif file_sha256(path) != document.get(f"{prefix}_sha256"):
                failures.append(f"stage_n_{label}_result_{prefix}_sha256_mismatch")
    if expected_kind == STAGE_N_RESUME_RESULT_KIND:
        evidence_value = document.get("evidence_artifact_path")
        evidence_failures: list[str] = []
        evidence = (
            _load_json_mapping_artifact(
                evidence_value,
                label="stage_n_resume_evidence",
                failures=evidence_failures,
            )
            if isinstance(evidence_value, str) and evidence_value.strip()
            else None
        )
        failures.extend(evidence_failures)
        if evidence is not None:
            failures.extend(
                _validate_stage_n_resume_evidence(
                    evidence,
                    expected_authorization_sha256=expected_authorization_sha256,
                    expected_governed_run_contract_sha256=(expected_governed_run_contract_sha256),
                    expected_checkpoint_path=expected_checkpoint_path,
                    expected_checkpoint_sha256=expected_checkpoint_sha256,
                    expected_checkpoint_step=expected_checkpoint_step,
                    expected_authorization_path=expected_authorization_path,
                    expected_governed_run_contract_path=(expected_governed_run_contract_path),
                )
            )
    return list(dict.fromkeys(failures))


def publish_stage_n_check_result(
    invocation_dir: str | Path,
    *,
    kind: str,
    authorization_path: str | Path,
    governed_run_contract_path: str | Path,
    runtime_fingerprint_path: str | Path,
    checkpoint_path: str | Path,
    checkpoint_step: int,
    evidence_artifact_path: str | Path,
) -> dict[str, Any]:
    """Build, validate, and atomically publish one canonical Stage-N check result.

    Independent smoke tooling uses this public boundary; it never needs the private atomic
    writer and cannot select an alternate result location or checkpoint identity.
    """
    require(
        kind in (STAGE_N_SMOKE_RESULT_KIND, STAGE_N_RESUME_RESULT_KIND),
        f"unknown Stage-N check kind {kind!r}",
    )
    failures: list[str] = []
    auth_path = Path(authorization_path)
    grc_path = Path(governed_run_contract_path)
    runtime_path = Path(runtime_fingerprint_path)
    checkpoint = Path(checkpoint_path)
    evidence = Path(evidence_artifact_path)
    auth = _load_json_mapping_artifact(
        auth_path, label="stage_n_check_authorization", failures=failures
    )
    grc = _load_json_mapping_artifact(
        grc_path, label="stage_n_check_governed_run_contract", failures=failures
    )
    runtime_artifact = _load_json_mapping_artifact(
        runtime_path, label="stage_n_check_runtime", failures=failures
    )
    require(
        not failures and auth is not None and grc is not None and runtime_artifact is not None,
        "refusing to publish Stage-N check from missing/malformed authority artifacts:\n  - "
        + "\n  - ".join(failures),
    )
    auth_sha = file_sha256(auth_path)
    completion = auth.get("stage_n_completion")
    require(
        isinstance(completion, Mapping)
        and completion.get("expected_final_step") == int(checkpoint_step),
        "Stage-N check step differs from its source authorization",
    )
    canonical_invocation = invocation_directory(
        str(auth.get("allowed_output_root", "")), "stage_a", auth_sha
    )
    require(
        Path(invocation_dir).expanduser().resolve() == canonical_invocation,
        "Stage-N check output is not the canonical source invocation",
    )
    for label, actual, canonical in (
        (
            "GRC",
            grc_path,
            canonical_invocation / GOVERNED_RUN_CONTRACT_FILENAME,
        ),
        (
            "runtime",
            runtime_path,
            canonical_invocation / STAGE_N_RUNTIME_FILENAME,
        ),
        (
            "checkpoint",
            checkpoint,
            canonical_invocation / f"step_{int(checkpoint_step):06d}.pt",
        ),
    ):
        require(
            actual.expanduser().resolve() == canonical.resolve(),
            f"Stage-N check {label} is outside the canonical source invocation",
        )
    require(evidence.is_file(), f"Stage-N check evidence not found: {evidence}")
    try:
        evidence.expanduser().resolve().relative_to(canonical_invocation)
    except ValueError as exc:
        raise LaunchContractError(
            "Stage-N check evidence escapes the canonical source invocation"
        ) from exc
    require(
        grc.get("stage_authorization_sha256") == auth_sha
        and _paths_equal(grc.get("stage_authorization_path"), auth_path),
        "Stage-N check GRC differs from its source authorization",
    )
    semantic_digest = governed_digest(grc)
    require(
        grc.get("governed_run_contract_sha256") == semantic_digest,
        "Stage-N check GRC semantic digest is invalid",
    )
    raw_runtime = runtime_artifact.get("runtime_fingerprint")
    require(isinstance(raw_runtime, Mapping), "Stage-N check runtime artifact is malformed")
    runtime_sha = runtime_fingerprint_sha256(raw_runtime)
    require(
        runtime_artifact.get("schema_version") == STAGE_N_RUNTIME_ARTIFACT_SCHEMA
        and runtime_artifact.get("kind") == STAGE_N_RUNTIME_ARTIFACT_KIND
        and runtime_artifact.get("runtime_fingerprint_sha256") == runtime_sha
        and grc.get("runtime_fingerprint_sha256") == runtime_sha,
        "Stage-N check runtime artifact differs from its source GRC",
    )
    check = stage_n_check_result_document(
        kind=kind,
        stage_authorization_sha256=auth_sha,
        governed_run_contract_sha256=semantic_digest,
        checkpoint_path=checkpoint,
        checkpoint_step=int(checkpoint_step),
        runtime_fingerprint_sha256=runtime_sha,
        evidence_artifact_path=evidence,
    )
    check_failures = validate_stage_n_check_result(
        check,
        expected_kind=kind,
        expected_authorization_sha256=auth_sha,
        expected_governed_run_contract_sha256=semantic_digest,
        expected_runtime_fingerprint_sha256=runtime_sha,
        expected_checkpoint_path=checkpoint,
        expected_checkpoint_sha256=file_sha256(checkpoint),
        expected_checkpoint_step=int(checkpoint_step),
        expected_authorization_path=auth_path,
        expected_governed_run_contract_path=grc_path,
    )
    require(
        not check_failures,
        "refusing to publish an invalid Stage-N check result:\n  - "
        + "\n  - ".join(check_failures),
    )
    filename = (
        STAGE_N_SMOKE_RESULT_FILENAME
        if kind == STAGE_N_SMOKE_RESULT_KIND
        else STAGE_N_RESUME_RESULT_FILENAME
    )
    published = _publish_or_verify_identical_json(canonical_invocation / filename, check)
    published["document"] = check
    return published


def publish_stage_n_resume_check_from_verified_invocation(
    *,
    verified_source_authority: Mapping[str, Any],
    authorized_checkpoint_verification: Mapping[str, Any],
    source_resume_binding: Mapping[str, Any],
    resume_authorization_path: str | Path,
    resume_governed_run_contract_path: str | Path,
    resume_final_checkpoint_path: str | Path,
    completed_step: int,
) -> dict[str, Any]:
    """Publish the canonical resume check only after a real verified resume reaches its end.

    This is the operational second phase of Stage-N completion. The result is written into
    the immutable SOURCE invocation and binds its authorization, GRC, checkpoint, and runtime;
    the later resume authorization is evidence that those source bytes were actually reopened
    and restored successfully. It does not authorize Stage N or Stage O.
    """
    require(
        bool(verified_source_authority.get("verified")),
        "Stage-N resume evidence requires verified source authority",
    )
    require(
        bool(authorized_checkpoint_verification.get("verified")),
        "Stage-N resume evidence requires verified source checkpoint bytes",
    )
    source_authorization = verified_source_authority.get("source_authorization")
    source_contract = verified_source_authority.get("source_invocation_run_contract")
    require(
        isinstance(source_authorization, Mapping) and isinstance(source_contract, Mapping),
        "Stage-N resume evidence requires source authorization and GRC artifacts",
    )

    source_auth_path = Path(str(verified_source_authority["source_authorization_path"]))
    source_grc_path = Path(str(verified_source_authority["source_invocation_run_contract_path"]))
    source_checkpoint = Path(str(verified_source_authority["source_checkpoint_path"]))
    source_auth_sha = file_sha256(source_auth_path)
    source_invocation = invocation_directory(
        str(source_authorization.get("allowed_output_root", "")),
        "stage_a",
        source_auth_sha,
    )
    require(
        source_grc_path.resolve() == (source_invocation / GOVERNED_RUN_CONTRACT_FILENAME).resolve(),
        "verified Stage-N source GRC is outside its canonical invocation",
    )
    expected_step = source_resume_binding.get("source_checkpoint_step")
    require(
        isinstance(expected_step, int)
        and not isinstance(expected_step, bool)
        and int(completed_step) == expected_step,
        "resume check completion step differs from the verified source checkpoint",
    )
    completion = source_authorization.get("stage_n_completion")
    require(
        isinstance(completion, Mapping) and completion.get("expected_final_step") == expected_step,
        "source Stage-N authorization does not bind this completion step",
    )
    require(
        source_checkpoint.resolve()
        == (source_invocation / f"step_{expected_step:06d}.pt").resolve(),
        "verified Stage-N source checkpoint is outside its canonical invocation",
    )
    require(
        file_sha256(source_checkpoint) == source_resume_binding.get("source_checkpoint_sha256"),
        "verified Stage-N source checkpoint bytes changed before resume evidence publication",
    )

    resume_auth_path = Path(resume_authorization_path)
    resume_grc_path = Path(resume_governed_run_contract_path)
    resume_checkpoint = Path(resume_final_checkpoint_path)
    for label, path in (
        ("resume authorization", resume_auth_path),
        ("resume GRC", resume_grc_path),
        ("resume final checkpoint", resume_checkpoint),
    ):
        require(path.is_file(), f"Stage-N {label} artifact not found: {path}")

    resume_grc_failures: list[str] = []
    resume_grc = _load_json_mapping_artifact(
        resume_grc_path,
        label="stage_n_resume_publication_current_grc",
        failures=resume_grc_failures,
    )
    require(
        not resume_grc_failures and resume_grc is not None,
        "Stage-N current resume GRC is malformed:\n  - " + "\n  - ".join(resume_grc_failures),
    )

    evidence_path = source_invocation / STAGE_N_RESUME_EVIDENCE_FILENAME
    evidence = {
        "schema_version": STAGE_N_RESUME_EVIDENCE_SCHEMA,
        "kind": STAGE_N_RESUME_EVIDENCE_KIND,
        "status": "PASS",
        "source_authority_verified": True,
        "source_checkpoint_bytes_verified": True,
        "source_stage_authorization_path": str(source_auth_path),
        "source_stage_authorization_sha256": source_auth_sha,
        "source_governed_run_contract_path": str(source_grc_path),
        "source_governed_run_contract_artifact_sha256": file_sha256(source_grc_path),
        "source_governed_run_contract_sha256": governed_digest(source_contract),
        "source_checkpoint_path": str(source_checkpoint),
        "source_checkpoint_sha256": file_sha256(source_checkpoint),
        "source_checkpoint_step": expected_step,
        "resume_stage_authorization_path": str(resume_auth_path),
        "resume_stage_authorization_sha256": file_sha256(resume_auth_path),
        "resume_governed_run_contract_path": str(resume_grc_path),
        "resume_governed_run_contract_artifact_sha256": file_sha256(resume_grc_path),
        "resume_governed_run_contract_sha256": governed_digest(resume_grc),
        "resume_final_checkpoint_path": str(resume_checkpoint),
        "resume_final_checkpoint_sha256": file_sha256(resume_checkpoint),
        "resume_start_step": expected_step,
        "completed_step": int(completed_step),
        "optimizer_updates": 0,
    }
    evidence_failures = _validate_stage_n_resume_evidence(
        evidence,
        expected_authorization_sha256=source_auth_sha,
        expected_governed_run_contract_sha256=governed_digest(source_contract),
        expected_checkpoint_path=source_checkpoint,
        expected_checkpoint_sha256=file_sha256(source_checkpoint),
        expected_checkpoint_step=expected_step,
        expected_authorization_path=source_auth_path,
        expected_governed_run_contract_path=source_grc_path,
    )
    require(
        not evidence_failures,
        "refusing to publish invalid Stage-N resume evidence:\n  - "
        + "\n  - ".join(evidence_failures),
    )
    evidence_publication = _publish_or_verify_identical_json(evidence_path, evidence)
    result_publication = publish_stage_n_check_result(
        source_invocation,
        kind=STAGE_N_RESUME_RESULT_KIND,
        authorization_path=source_auth_path,
        governed_run_contract_path=source_grc_path,
        runtime_fingerprint_path=source_invocation / STAGE_N_RUNTIME_FILENAME,
        checkpoint_path=source_checkpoint,
        checkpoint_step=expected_step,
        evidence_artifact_path=evidence_path,
    )
    return {
        "evidence": evidence_publication,
        "result": result_publication,
        "source_invocation_dir": str(source_invocation),
    }


def stage_n_result_document(
    *,
    governed_run_contract: Mapping[str, Any],
    governed_run_contract_path: str | Path | None = None,
    final_checkpoint_path: str,
    final_checkpoint_sha256: str,
    final_checkpoint_step: int,
    smoke_results: Mapping[str, Any],
    smoke_results_path: str | Path | None = None,
    resume_results: Mapping[str, Any],
    resume_results_path: str | Path | None = None,
    runtime_fingerprint_path: str | Path | None = None,
    final_sampler_state: Mapping[str, Any],
) -> dict[str, Any]:
    """The machine-readable Stage-N result a future Stage-O authorization must bind."""
    for field in (
        "permutation_identity",
        "range_start_position",
        "invocation_range_start_position",
        "range_stop_position",
        "cursor",
    ):
        require(
            final_sampler_state.get(field) is not None,
            f"a Stage-N result must record the final sampler {field}; without it the A->B "
            f"source checks cannot be populated and would silently pass",
        )
    runtime = dict(governed_run_contract.get("runtime_fingerprint") or {})
    return {
        "schema_version": STAGE_N_RESULT_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "status": "COMPLETE",
        "stage": "stage_a",
        "scope": "STAGE_N",
        "stage_authorization_path": governed_run_contract.get("stage_authorization_path"),
        "stage_authorization_sha256": governed_run_contract["stage_authorization_sha256"],
        "launch_contract_sha256": governed_run_contract["launch_contract_sha256"],
        "exact_plan_path": governed_run_contract.get("exact_plan_path"),
        "exact_run_plan_sha256": governed_run_contract["exact_run_plan_sha256"],
        "pilot_acceptance_path": governed_run_contract.get("pilot_acceptance_path"),
        "pilot_owner_acceptance_sha256": governed_run_contract["pilot_owner_acceptance_sha256"],
        "trainer_head": governed_run_contract["trainer_head"],
        "trainer_execution_bundle_sha256": governed_run_contract["trainer_execution_bundle_sha256"],
        "governed_run_contract_path": (
            str(governed_run_contract_path) if governed_run_contract_path is not None else None
        ),
        "governed_run_contract_artifact_sha256": (
            file_sha256(governed_run_contract_path)
            if governed_run_contract_path is not None and Path(governed_run_contract_path).is_file()
            else None
        ),
        "governed_run_contract_sha256": governed_digest(governed_run_contract),
        "base_governed_identity_sha256": base_governed_identity_sha256(governed_run_contract),
        "final_checkpoint_path": str(final_checkpoint_path),
        "final_checkpoint_sha256": str(final_checkpoint_sha256),
        "final_checkpoint_step": int(final_checkpoint_step),
        "runtime_fingerprint": runtime,
        "runtime_fingerprint_sha256": (
            runtime.get("runtime_fingerprint_sha256") or runtime_fingerprint_sha256(runtime)
        ),
        "runtime_fingerprint_artifact_sha256": (
            file_sha256(runtime_fingerprint_path)
            if runtime_fingerprint_path is not None and Path(runtime_fingerprint_path).is_file()
            else None
        ),
        "runtime_fingerprint_path": (
            str(runtime_fingerprint_path) if runtime_fingerprint_path is not None else None
        ),
        "gpu_uuid": runtime.get("gpu_uuid"),
        "gpu_pci_bus_id": runtime.get("gpu_pci_bus_id"),
        "num_workers": governed_run_contract.get("num_workers"),
        "smoke_results_path": str(smoke_results_path) if smoke_results_path is not None else None,
        "smoke_results_sha256": (
            file_sha256(smoke_results_path)
            if smoke_results_path is not None and Path(smoke_results_path).is_file()
            else None
        ),
        "smoke_results": dict(smoke_results),
        "resume_results_path": (
            str(resume_results_path) if resume_results_path is not None else None
        ),
        "resume_results_sha256": (
            file_sha256(resume_results_path)
            if resume_results_path is not None and Path(resume_results_path).is_file()
            else None
        ),
        "resume_results": dict(resume_results),
        # R3 Part 18: without these, derive_stage_o_resume_binding could not populate the
        # source_* fields, and the strongest A->B sampler checks stayed inert (their guards
        # are `is not None`, so an absent expectation silently passed).
        "final_sampler_permutation_identity": final_sampler_state["permutation_identity"],
        "final_sampler_range_start_position": int(final_sampler_state["range_start_position"]),
        "final_sampler_invocation_range_start_position": int(
            final_sampler_state["invocation_range_start_position"]
        ),
        "final_sampler_range_stop_position": int(final_sampler_state["range_stop_position"]),
        "final_sampler_cursor": int(final_sampler_state["cursor"]),
    }


STAGE_N_SHA256_FIELDS = (
    "runtime_fingerprint_artifact_sha256",
    "governed_run_contract_artifact_sha256",
    "stage_authorization_sha256",
    "launch_contract_sha256",
    "exact_run_plan_sha256",
    "pilot_owner_acceptance_sha256",
    "trainer_execution_bundle_sha256",
    "governed_run_contract_sha256",
    "final_checkpoint_sha256",
    "runtime_fingerprint_sha256",
    "base_governed_identity_sha256",
    "smoke_results_sha256",
    "resume_results_sha256",
)
STAGE_N_NON_EMPTY_FIELDS = (
    *STAGE_N_SHA256_FIELDS,
    "trainer_head",
    "stage_authorization_path",
    "exact_plan_path",
    "pilot_acceptance_path",
    "governed_run_contract_path",
    "final_checkpoint_path",
    "runtime_fingerprint_path",
    "smoke_results_path",
    "resume_results_path",
    "gpu_uuid",
    "gpu_pci_bus_id",
)


def validate_stage_n_result(
    document: Mapping[str, Any] | None, *, require_artifacts: bool = False
) -> list[str]:
    """R2 Part 12: status=COMPLETE never makes an empty result valid.

    Every load-bearing field must be present, non-null, non-empty and well-formed.
    """
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

    for field in STAGE_N_NON_EMPTY_FIELDS:
        value = document.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            failures.append(f"stage_n_result_empty_field:{field}")
    for field in STAGE_N_SHA256_FIELDS:
        value = document.get(field)
        if isinstance(value, str) and value.strip() and not _is_sha256(value):
            failures.append(f"stage_n_result_malformed_sha256:{field}")

    step = document.get("final_checkpoint_step")
    if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
        failures.append("stage_n_result_final_checkpoint_step_invalid")
    workers = document.get("num_workers")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 0:
        failures.append("stage_n_result_num_workers_invalid")

    runtime = document.get("runtime_fingerprint")
    if not isinstance(runtime, Mapping) or not runtime:
        failures.append("stage_n_result_runtime_fingerprint_empty")
    else:
        for field in RUNTIME_BINDING_REQUIRED_FIELDS:
            if runtime.get(field) in (None, ""):
                failures.append(f"stage_n_result_runtime_missing:{field}")
        stored_sha = document.get("runtime_fingerprint_sha256")
        if stored_sha and stored_sha != runtime_fingerprint_sha256(runtime):
            failures.append("stage_n_result_runtime_fingerprint_sha256_mismatch")

    for block in ("smoke_results", "resume_results"):
        value = document.get(block)
        if not isinstance(value, Mapping) or not value:
            failures.append(f"stage_n_result_{block}_empty")
    failures.extend(
        validate_stage_n_check_result(
            document.get("smoke_results"),
            expected_kind=STAGE_N_SMOKE_RESULT_KIND,
            expected_authorization_sha256=str(document.get("stage_authorization_sha256") or ""),
            expected_governed_run_contract_sha256=str(
                document.get("governed_run_contract_sha256") or ""
            ),
            expected_runtime_fingerprint_sha256=str(
                document.get("runtime_fingerprint_sha256") or ""
            ),
            expected_checkpoint_path=document.get("final_checkpoint_path"),
            expected_checkpoint_sha256=document.get("final_checkpoint_sha256"),
            expected_checkpoint_step=(
                int(document["final_checkpoint_step"])
                if isinstance(document.get("final_checkpoint_step"), int)
                and not isinstance(document.get("final_checkpoint_step"), bool)
                else None
            ),
            expected_authorization_path=document.get("stage_authorization_path"),
            expected_governed_run_contract_path=document.get("governed_run_contract_path"),
        )
    )
    failures.extend(
        validate_stage_n_check_result(
            document.get("resume_results"),
            expected_kind=STAGE_N_RESUME_RESULT_KIND,
            expected_authorization_sha256=str(document.get("stage_authorization_sha256") or ""),
            expected_governed_run_contract_sha256=str(
                document.get("governed_run_contract_sha256") or ""
            ),
            expected_runtime_fingerprint_sha256=str(
                document.get("runtime_fingerprint_sha256") or ""
            ),
            expected_checkpoint_path=document.get("final_checkpoint_path"),
            expected_checkpoint_sha256=document.get("final_checkpoint_sha256"),
            expected_checkpoint_step=(
                int(document["final_checkpoint_step"])
                if isinstance(document.get("final_checkpoint_step"), int)
                and not isinstance(document.get("final_checkpoint_step"), bool)
                else None
            ),
            expected_authorization_path=document.get("stage_authorization_path"),
            expected_governed_run_contract_path=document.get("governed_run_contract_path"),
        )
    )

    if require_artifacts:
        path = document.get("final_checkpoint_path")
        if isinstance(path, str) and path.strip():
            candidate = Path(path)
            if not candidate.is_file():
                failures.append(f"stage_n_result_final_checkpoint_absent:{path}")
            elif file_sha256(candidate) != document.get("final_checkpoint_sha256"):
                failures.append("stage_n_result_final_checkpoint_sha256_mismatch")
    return list(dict.fromkeys(failures))


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value.lower())
    )


STAGE_N_RESULT_FILENAME = "STAGE_N_RESULT.json"


def publish_stage_n_result(out_dir: str | Path, document: Mapping[str, Any]) -> dict[str, Any]:
    """R2 Part 13: validate, then publish the Stage-N result atomically, exactly once."""
    failures = validate_stage_n_result(document, require_artifacts=True)
    require(
        not failures,
        "refusing to publish an invalid Stage-N result:\n  - " + "\n  - ".join(failures),
    )
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / STAGE_N_RESULT_FILENAME
    require(not path.exists(), f"{STAGE_N_RESULT_FILENAME} already exists at {path}")
    body = canonical_json_bytes(document)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "wb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return {
        "path": str(path),
        "stage_n_result_sha256": _sha256_bytes(body),
        "atomic": True,
        "owner_acceptance_required": True,
        "status": "PUBLISHED_AWAITING_OWNER_ACCEPTANCE",
    }


def _validate_stage_o_chain_r3(
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

    # R3 Part 14: the accepted Stage-N runtime ARTIFACT is loaded from disk and rehashed;
    # the redundant raw copy inside the Stage-O authorization is never trusted.
    failures.extend(validate_stage_o_runtime_artifact(chain, stage_n, observed_runtime))

    # R2 Part 15: the COMPLETE runtime fingerprint SHA must match, not only chosen fields.
    accepted_fp_sha = stage_n.get("runtime_fingerprint_sha256")
    if chain.get("stage_n_runtime_fingerprint_sha256") != accepted_fp_sha:
        failures.append("stage_o_chain_runtime_fingerprint_sha256_contradicts_accepted_result")
    observed_fp_sha = observed_runtime.get("runtime_fingerprint_sha256") or (
        runtime_fingerprint_sha256(observed_runtime) if observed_runtime else None
    )
    if accepted_fp_sha and observed_fp_sha and accepted_fp_sha != observed_fp_sha:
        failures.append("stage_o_runtime_differs_from_accepted_stage_n:runtime_fingerprint_sha256")

    # The chain's claims must equal the accepted result itself.
    for chain_field, result_field in (
        ("stage_n_authorization_sha256", "stage_authorization_sha256"),
        ("stage_n_governed_run_contract_sha256", "governed_run_contract_sha256"),
        ("stage_n_gpu_uuid", "gpu_uuid"),
        ("stage_n_gpu_pci_bus_id", "gpu_pci_bus_id"),
        ("stage_n_trainer_head", "trainer_head"),
        ("stage_n_trainer_execution_bundle_sha256", "trainer_execution_bundle_sha256"),
        ("stage_n_exact_run_plan_sha256", "exact_run_plan_sha256"),
        ("stage_n_final_checkpoint_path", "final_checkpoint_path"),
        ("stage_n_final_checkpoint_sha256", "final_checkpoint_sha256"),
        ("stage_n_final_checkpoint_step", "final_checkpoint_step"),
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

    # R2 Part 14: the authorized Stage-O resume binding must BE the accepted Stage-N
    # checkpoint, and the bytes on disk must hash to the accepted SHA.
    resume = authorization.get("resume") or {}
    if resume.get("mode") != "RESUME_EXACT_CHECKPOINT":
        failures.append("stage_o_requires_resume_mode_RESUME_EXACT_CHECKPOINT")
    else:
        for resume_field, chain_field in (
            ("checkpoint_path", "stage_n_final_checkpoint_path"),
            ("checkpoint_sha256", "stage_n_final_checkpoint_sha256"),
            ("expected_step", "stage_n_final_checkpoint_step"),
        ):
            if resume.get(resume_field) != chain.get(chain_field):
                failures.append(
                    f"stage_o_resume_binding_is_not_the_accepted_stage_n_checkpoint:{resume_field}"
                )
        ckpt_path = resume.get("checkpoint_path")
        if isinstance(ckpt_path, str) and ckpt_path.strip():
            candidate = Path(ckpt_path)
            if not candidate.is_file():
                failures.append(f"stage_o_resume_checkpoint_absent:{ckpt_path}")
            elif file_sha256(candidate) != resume.get("checkpoint_sha256"):
                failures.append("stage_o_resume_checkpoint_bytes_do_not_match_accepted_sha256")

    failures = list(dict.fromkeys(failures))
    return {
        "valid": not failures,
        "failures": failures,
        "accepted_stage_n_result_sha256": result_sha,
        "requires_new_stage_n": bool([
            f for f in failures if f.startswith("stage_o_runtime_differs_from_accepted_stage_n")
        ]),
    }


# --------------------------------------------------------------- R3 Stage-N/O binding


def validate_stage_n_checkpoint_artifact(
    checkpoint_path: str | Path,
    *,
    result: Mapping[str, Any],
    governed_run_contract: Mapping[str, Any],
    expected_final_step: int,
) -> list[str]:
    """Open the final checkpoint and validate its governed semantics, not only its bytes."""
    failures: list[str] = []
    path = Path(checkpoint_path)
    if not path.is_file():
        return [f"stage_n_final_checkpoint_not_found:{path}"]
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - malformed/unpicklable artifacts fail closed
        return [f"stage_n_final_checkpoint_not_loadable:{type(exc).__name__}"]
    if not isinstance(checkpoint, Mapping):
        return ["stage_n_final_checkpoint_not_a_mapping"]

    if checkpoint.get("kind") != GOVERNED_CHECKPOINT_KIND:
        failures.append("stage_n_final_checkpoint_kind_mismatch")
    step = checkpoint.get("global_step")
    if not isinstance(step, int) or isinstance(step, bool) or step != int(expected_final_step):
        failures.append(
            f"stage_n_final_checkpoint_step_mismatch:checkpoint={step!r},"
            f"expected={expected_final_step}"
        )

    expected_digest = governed_digest(governed_run_contract)
    if checkpoint.get("governed_run_contract_sha256") != expected_digest:
        failures.append("stage_n_final_checkpoint_run_contract_digest_mismatch")
    checkpoint_contract = checkpoint.get("governed_run_contract")
    if not isinstance(checkpoint_contract, Mapping):
        failures.append("stage_n_final_checkpoint_governed_run_contract_missing")
        checkpoint_contract = {}
    else:
        if governed_digest(checkpoint_contract) != expected_digest:
            failures.append("stage_n_final_checkpoint_embedded_contract_digest_mismatch")
        expected_contract = dict(governed_run_contract)
        expected_contract.pop("governed_run_contract_sha256", None)
        actual_contract = dict(checkpoint_contract)
        actual_contract.pop("governed_run_contract_sha256", None)
        if actual_contract != expected_contract:
            failures.append("stage_n_final_checkpoint_embedded_contract_not_exact_artifact")
        for field, expected in (
            ("kind", GOVERNED_CHECKPOINT_KIND),
            ("stage", "stage_a"),
            ("scope", "STAGE_N"),
            ("stage_authorization_sha256", result.get("stage_authorization_sha256")),
            ("runtime_fingerprint_sha256", result.get("runtime_fingerprint_sha256")),
        ):
            if checkpoint_contract.get(field) != expected:
                failures.append(f"stage_n_final_checkpoint_embedded_contract_mismatch:{field}")
        expected_base = base_governed_identity_sha256(governed_run_contract)
        if base_governed_identity_sha256(checkpoint_contract) != expected_base:
            failures.append("stage_n_final_checkpoint_base_identity_mismatch")
        if result.get("base_governed_identity_sha256") != expected_base:
            failures.append("stage_n_result_base_identity_mismatch")

    state = checkpoint.get("governed_checkpoint_state")
    if not isinstance(state, Mapping):
        return list(dict.fromkeys([*failures, "stage_n_final_checkpoint_dynamic_state_missing"]))
    failures.extend(
        validate_governed_checkpoint_state(
            state,
            governed_run_contract=governed_run_contract,
            checkpoint_global_step=int(expected_final_step),
        )
    )
    failures.extend(validate_governed_checkpoint_resume_envelope(checkpoint))
    expected_dynamic = {
        "active_stage": "stage_a",
        "active_stage_sampler_seed": STAGE_A_SAMPLER_SEED,
        "global_step": int(expected_final_step),
        "permutation_identity": result.get("final_sampler_permutation_identity"),
        "range_start_position": result.get("final_sampler_range_start_position"),
        "invocation_range_start_position": result.get(
            "final_sampler_invocation_range_start_position"
        ),
        "range_stop_position": result.get("final_sampler_range_stop_position"),
        "cursor": result.get("final_sampler_cursor"),
    }
    for field, expected in expected_dynamic.items():
        if state.get(field) != expected:
            failures.append(f"stage_n_final_checkpoint_dynamic_state_mismatch:{field}")
    start = state.get("range_start_position")
    stop = state.get("range_stop_position")
    cursor = state.get("cursor")
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (start, stop, cursor)):
        failures.append("stage_n_final_checkpoint_sampler_range_not_integer")
    elif not (0 <= int(start) <= int(cursor) <= int(stop)):
        failures.append("stage_n_final_checkpoint_sampler_range_invalid")
    elif state.get("permutation_identity") != permutation_identity(
        "stage_a", STAGE_A_SAMPLER_SEED, int(stop)
    ):
        failures.append("stage_n_final_checkpoint_sampler_permutation_not_canonical")

    # The result and checkpoint cannot collude on a different range. The invocation contract
    # was published before training and independently authenticates the stage, seed,
    # permutation, canonical range, and invocation start. Only its cursor is expected to
    # advance by completion time.
    contract_sampler = governed_run_contract.get("sampler_identity")
    if not isinstance(contract_sampler, Mapping):
        failures.append("stage_n_governed_run_contract_sampler_identity_missing")
    else:
        for state_field, contract_field in (
            ("active_stage", "stage"),
            ("active_stage_sampler_seed", "sampler_seed"),
            ("permutation_identity", "permutation_identity"),
            ("range_start_position", "range_start_position"),
            ("invocation_range_start_position", "invocation_range_start_position"),
            ("range_stop_position", "range_stop_position"),
        ):
            if state.get(state_field) != contract_sampler.get(contract_field):
                failures.append(
                    "stage_n_final_checkpoint_dynamic_state_contract_sampler_mismatch:"
                    f"{state_field}"
                )
        initial_cursor = contract_sampler.get("cursor")
        if (
            not isinstance(initial_cursor, int)
            or isinstance(initial_cursor, bool)
            or not isinstance(cursor, int)
            or isinstance(cursor, bool)
            or cursor < initial_cursor
        ):
            failures.append("stage_n_final_checkpoint_cursor_did_not_advance_from_contract")

    stage_start = governed_run_contract.get("stage_start_step")
    if isinstance(stage_start, int) and not isinstance(stage_start, bool):
        expected_cursor = (int(expected_final_step) - stage_start) * SEQUENCES_PER_UPDATE
        if state.get("cursor") != expected_cursor:
            failures.append("stage_n_final_checkpoint_cursor_step_mismatch")

    compile_evidence = state.get("compile_evidence")
    failures.extend(verify_compile_evidence_document(compile_evidence))
    if isinstance(compile_evidence, Mapping):
        if state.get("compile_evidence_sha256") != compile_evidence.get("compile_evidence_sha256"):
            failures.append("stage_n_final_checkpoint_compile_evidence_sha256_mismatch")
        if compile_evidence != governed_run_contract.get("compile_evidence"):
            failures.append("stage_n_final_checkpoint_compile_evidence_not_current_process")
    return list(dict.fromkeys(failures))


def _validate_stage_n_result_against_artifacts_r3(
    result: Mapping[str, Any],
    *,
    authorization: Mapping[str, Any],
    authorization_path: str | Path,
    governed_run_contract: Mapping[str, Any],
    governed_run_contract_path: str | Path,
    runtime_fingerprint_path: str | Path,
    checkpoint_path: str | Path,
) -> list[str]:
    """R3 Part 12: semantic identity, not merely well-formed strings.

    Every load-bearing value is compared against the ACTUAL Stage-N authorization, governed
    run contract, checkpoint and runtime artifacts on disk. A wrong-but-well-formed SHA, a
    malformed trainer HEAD, a wrong checkpoint step or a contradictory UUID/PCI is rejected.
    """
    failures = validate_stage_n_result(result, require_artifacts=True)

    if result.get("scope") != "STAGE_N":
        failures.append(f"stage_n_result_scope_mismatch:{result.get('scope')!r}")
    if result.get("stage") not in ("stage_a", "STAGE_N"):
        failures.append(f"stage_n_result_stage_mismatch:{result.get('stage')!r}")

    head = result.get("trainer_head")
    if not (
        isinstance(head, str)
        and len(head) == 40
        and all(c in "0123456789abcdef" for c in head.lower())
    ):
        failures.append(f"stage_n_result_malformed_trainer_head:{head!r}")

    # ---- against the authorization ----
    auth_path = Path(authorization_path)
    if not auth_path.is_file():
        failures.append(f"stage_n_authorization_not_found:{auth_path}")
    else:
        observed_auth_sha = file_sha256(auth_path)
        if result.get("stage_authorization_sha256") != observed_auth_sha:
            failures.append("stage_n_result_authorization_sha256_does_not_match_the_artifact")
    for field, key in (
        ("launch_contract_sha256", "launch_contract_sha256"),
        ("exact_run_plan_sha256", "exact_run_plan_sha256"),
        ("pilot_owner_acceptance_sha256", "pilot_owner_acceptance_sha256"),
    ):
        if authorization.get(key) is not None and result.get(field) != authorization.get(key):
            failures.append(f"stage_n_result_contradicts_authorization:{field}")

    # ---- against the governed run contract ----
    grc_path = Path(governed_run_contract_path)
    if not grc_path.is_file():
        failures.append(f"stage_n_governed_run_contract_not_found:{grc_path}")
    elif result.get("governed_run_contract_sha256") != governed_digest(governed_run_contract):
        failures.append("stage_n_result_governed_run_contract_sha256_mismatch")
    for field in (
        "trainer_head",
        "trainer_execution_bundle_sha256",
        "num_workers",
        "gpu_uuid",
        "gpu_pci_bus_id",
    ):
        if governed_run_contract.get(field) != result.get(field):
            failures.append(f"stage_n_result_contradicts_run_contract:{field}")
    # R3 Part 12: a wrong checkpoint step must be rejected. Stage N ends at the governed stage
    # stop boundary, so the run contract is the authority -- not merely "a positive integer".
    expected_step = governed_run_contract.get("stage_stop_step")
    if expected_step is not None and int(result.get("final_checkpoint_step", -1)) != int(
        expected_step
    ):
        failures.append(
            f"stage_n_result_final_checkpoint_step_contradicts_run_contract: "
            f"result={result.get('final_checkpoint_step')!r}, "
            f"stage_stop_step={expected_step!r}"
        )

    # ---- against the runtime artifact (Part 14) ----
    rt_path = Path(runtime_fingerprint_path)
    if not rt_path.is_file():
        failures.append(f"stage_n_runtime_fingerprint_artifact_not_found:{rt_path}")
    else:
        if result.get("runtime_fingerprint_artifact_sha256") != file_sha256(rt_path):
            failures.append("stage_n_result_runtime_artifact_sha256_mismatch")
        raw = json.loads(rt_path.read_bytes().decode("utf-8"))
        if runtime_fingerprint_sha256(raw) != result.get("runtime_fingerprint_sha256"):
            failures.append("stage_n_runtime_artifact_document_sha256_mismatch")
        for field in ("gpu_uuid", "gpu_pci_bus_id", "num_workers"):
            if raw.get(field) != result.get(field):
                failures.append(f"stage_n_runtime_artifact_contradicts_result:{field}")

    # ---- against the checkpoint ----
    ckpt = Path(checkpoint_path)
    if not ckpt.is_file():
        failures.append(f"stage_n_final_checkpoint_not_found:{ckpt}")
    else:
        if result.get("final_checkpoint_sha256") != file_sha256(ckpt):
            failures.append("stage_n_result_final_checkpoint_sha256_does_not_match_bytes")
        if Path(str(result.get("final_checkpoint_path"))).resolve() != ckpt.resolve():
            failures.append("stage_n_result_final_checkpoint_path_mismatch")
    return list(dict.fromkeys(failures))


def validate_stage_n_result_against_artifacts(
    result: Mapping[str, Any],
    *,
    authorization_path: str | Path,
    governed_run_contract_path: str | Path,
    runtime_fingerprint_path: str | Path,
    checkpoint_path: str | Path,
    smoke_results_path: str | Path,
    resume_results_path: str | Path,
    exact_plan_path: str | Path | None = None,
    pilot_acceptance_path: str | Path | None = None,
    authorization: Mapping[str, Any] | None = None,
    governed_run_contract: Mapping[str, Any] | None = None,
    expected_final_step: int | None = None,
) -> list[str]:
    """Authenticate a Stage-N COMPLETE result against every actual bound artifact."""
    failures = validate_stage_n_result(result, require_artifacts=True)
    recorded_final_step = result.get("final_checkpoint_step")
    final_step: int | None = (
        int(recorded_final_step)
        if isinstance(recorded_final_step, int) and not isinstance(recorded_final_step, bool)
        else None
    )
    if expected_final_step is not None:
        if isinstance(expected_final_step, int) and not isinstance(expected_final_step, bool):
            if final_step is not None and final_step != int(expected_final_step):
                failures.append("stage_n_result_final_checkpoint_step_mismatch")
            final_step = int(expected_final_step)
        else:
            failures.append("stage_n_expected_final_checkpoint_step_invalid")
    if result.get("scope") != "STAGE_N":
        failures.append(f"stage_n_result_scope_mismatch:{result.get('scope')!r}")
    if result.get("stage") != "stage_a":
        failures.append(f"stage_n_result_stage_mismatch:{result.get('stage')!r}")
    head = result.get("trainer_head")
    if not (
        isinstance(head, str)
        and len(head) == 40
        and all(c in "0123456789abcdef" for c in head.lower())
    ):
        failures.append(f"stage_n_result_malformed_trainer_head:{head!r}")

    auth_path = Path(authorization_path)
    if not _paths_equal(result.get("stage_authorization_path"), auth_path):
        failures.append("stage_n_result_authorization_path_mismatch")
    auth = _load_json_mapping_artifact(auth_path, label="stage_n_authorization", failures=failures)
    expected_invocation_root: Path | None = None
    if auth is not None:
        if authorization is not None and dict(authorization) != auth:
            failures.append("stage_n_supplied_authorization_differs_from_artifact")
        if result.get("stage_authorization_sha256") != file_sha256(auth_path):
            failures.append("stage_n_result_authorization_sha256_does_not_match_the_artifact")
        for field, expected in (
            ("schema_version", AUTHORIZATION_SCHEMA),
            ("authorization_status", "AUTHORIZED"),
            ("allowed_scope", "STAGE_N"),
        ):
            if auth.get(field) != expected:
                failures.append(f"stage_n_authorization_{field}_mismatch")
        completion = auth.get("stage_n_completion")
        if not isinstance(completion, Mapping):
            failures.append("stage_n_authorization_completion_binding_missing")
        else:
            authorized_step = completion.get("expected_final_step")
            if (
                not isinstance(authorized_step, int)
                or isinstance(authorized_step, bool)
                or authorized_step <= 0
            ):
                failures.append("stage_n_authorization_expected_final_step_invalid")
            elif final_step is not None and authorized_step != final_step:
                failures.append("stage_n_authorization_expected_final_step_mismatch")
            else:
                final_step = int(authorized_step)

        run_root = auth.get("allowed_output_root")
        if not isinstance(run_root, str) or not run_root.strip():
            failures.append("stage_n_authorization_allowed_output_root_missing")
        elif auth_path.is_file():
            expected_invocation_root = invocation_directory(
                run_root, "stage_a", file_sha256(auth_path)
            )

    if expected_invocation_root is not None and final_step is not None:
        topology = (
            (
                "governed_run_contract",
                Path(governed_run_contract_path),
                expected_invocation_root / GOVERNED_RUN_CONTRACT_FILENAME,
            ),
            (
                "runtime_fingerprint",
                Path(runtime_fingerprint_path),
                expected_invocation_root / STAGE_N_RUNTIME_FILENAME,
            ),
            (
                "final_checkpoint",
                Path(checkpoint_path),
                expected_invocation_root / f"step_{final_step:06d}.pt",
            ),
            (
                "smoke_result",
                Path(smoke_results_path),
                expected_invocation_root / STAGE_N_SMOKE_RESULT_FILENAME,
            ),
            (
                "resume_result",
                Path(resume_results_path),
                expected_invocation_root / STAGE_N_RESUME_RESULT_FILENAME,
            ),
        )
        for label, actual, canonical in topology:
            if actual.expanduser().resolve() != canonical.resolve():
                failures.append(f"stage_n_{label}_not_at_canonical_invocation_path")

    grc_path = Path(governed_run_contract_path)
    if not _paths_equal(result.get("governed_run_contract_path"), grc_path):
        failures.append("stage_n_result_governed_run_contract_path_mismatch")
    grc = _load_json_mapping_artifact(
        grc_path, label="stage_n_governed_run_contract", failures=failures
    )
    if grc is not None:
        if governed_run_contract is not None and any(
            grc.get(key) != value for key, value in governed_run_contract.items()
        ):
            failures.append("stage_n_supplied_run_contract_differs_from_artifact")
        if result.get("governed_run_contract_artifact_sha256") != file_sha256(grc_path):
            failures.append("stage_n_result_governed_run_contract_artifact_sha256_mismatch")
        semantic_digest = governed_digest(grc)
        if grc.get("governed_run_contract_sha256") != semantic_digest:
            failures.append("stage_n_governed_run_contract_self_digest_mismatch")
        if result.get("governed_run_contract_sha256") != semantic_digest:
            failures.append("stage_n_result_governed_run_contract_sha256_mismatch")
        for field, expected in (
            ("schema_version", RUN_CONTRACT_SCHEMA),
            ("kind", GOVERNED_CHECKPOINT_KIND),
            ("governed", True),
            ("stage", "stage_a"),
            ("scope", "STAGE_N"),
        ):
            if grc.get(field) != expected:
                failures.append(f"stage_n_governed_run_contract_{field}_mismatch")
        if expected_invocation_root is not None:
            expected_root = expected_invocation_root.parent
            if Path(str(grc.get("governed_run_root", ""))).expanduser().resolve() != expected_root:
                failures.append("stage_n_governed_run_contract_governed_run_root_mismatch")
            for field in ("invocation_root", "out_dir"):
                if Path(str(grc.get(field, ""))).expanduser().resolve() != expected_invocation_root:
                    failures.append(f"stage_n_governed_run_contract_{field}_topology_mismatch")
            if Path(str(grc.get("samples_dir", ""))).expanduser().resolve() != (
                expected_invocation_root / "samples"
            ):
                failures.append("stage_n_governed_run_contract_samples_dir_topology_mismatch")
        if result.get("base_governed_identity_sha256") != base_governed_identity_sha256(grc):
            failures.append("stage_n_result_base_governed_identity_sha256_mismatch")
        stage_stop = grc.get("stage_stop_step")
        if (
            isinstance(stage_stop, int)
            and not isinstance(stage_stop, bool)
            and final_step is not None
            and final_step != stage_stop
        ):
            failures.append("stage_n_final_checkpoint_step_contradicts_run_contract_stop")
        for field in (
            "stage_authorization_sha256",
            "launch_contract_sha256",
            "exact_run_plan_sha256",
            "pilot_owner_acceptance_sha256",
            "trainer_head",
            "trainer_execution_bundle_sha256",
            "num_workers",
            "gpu_uuid",
            "gpu_pci_bus_id",
            "runtime_fingerprint_sha256",
        ):
            if result.get(field) != grc.get(field):
                failures.append(f"stage_n_result_contradicts_run_contract:{field}")
        if auth is not None:
            if grc.get("stage_authorization_sha256") != file_sha256(auth_path):
                failures.append("stage_n_run_contract_authorization_sha256_mismatch")
            if not _paths_equal(grc.get("stage_authorization_path"), auth_path):
                failures.append("stage_n_run_contract_authorization_path_mismatch")
            for field in (
                "launch_contract_sha256",
                "exact_run_plan_sha256",
                "pilot_owner_acceptance_sha256",
                "trainer_head",
                "trainer_execution_bundle_sha256",
            ):
                if auth.get(field) != grc.get(field):
                    failures.append(f"stage_n_authorization_run_contract_mismatch:{field}")
            if auth.get("training_runtime") != grc.get("runtime_fingerprint"):
                failures.append("stage_n_authorization_run_contract_runtime_mismatch")

    plan_path = Path(exact_plan_path or (grc or {}).get("exact_plan_path") or "")
    if not _paths_equal(result.get("exact_plan_path"), plan_path):
        failures.append("stage_n_result_exact_plan_path_mismatch")
    plan = _load_json_mapping_artifact(plan_path, label="stage_n_exact_plan", failures=failures)
    if plan is not None:
        if file_sha256(plan_path) != EXACT_RUN_PLAN_SHA256:
            failures.append("stage_n_exact_plan_sha256_mismatch")
        if result.get("exact_run_plan_sha256") != file_sha256(plan_path):
            failures.append("stage_n_result_exact_plan_sha256_mismatch")
        if plan.get("schema_version") != 3 or plan.get("plan_type") != (
            "deterministic_no_replacement_stage_a_b"
        ):
            failures.append("stage_n_exact_plan_schema_or_kind_mismatch")

    pilot_path = Path(pilot_acceptance_path or (grc or {}).get("pilot_acceptance_path") or "")
    if not _paths_equal(result.get("pilot_acceptance_path"), pilot_path):
        failures.append("stage_n_result_pilot_acceptance_path_mismatch")
    pilot = _load_json_mapping_artifact(
        pilot_path, label="stage_n_pilot_acceptance", failures=failures
    )
    if pilot is not None:
        if file_sha256(pilot_path) != PILOT_OWNER_ACCEPTANCE_SHA256:
            failures.append("stage_n_pilot_acceptance_sha256_mismatch")
        if result.get("pilot_owner_acceptance_sha256") != file_sha256(pilot_path):
            failures.append("stage_n_result_pilot_acceptance_sha256_mismatch")
        if (
            pilot.get("schema_version") != "petitgpt-pilot-result-owner-acceptance-v1"
            or pilot.get("PILOT_RESULT_OWNER_VERDICT") != "ACCEPTED"
            or pilot.get("this_artifact_authorizes_training") is not False
        ):
            failures.append("stage_n_pilot_acceptance_schema_or_kind_mismatch")

    runtime_path = Path(runtime_fingerprint_path)
    if not _paths_equal(result.get("runtime_fingerprint_path"), runtime_path):
        failures.append("stage_n_result_runtime_fingerprint_path_mismatch")
    runtime_artifact = _load_json_mapping_artifact(
        runtime_path, label="stage_n_runtime_fingerprint_artifact", failures=failures
    )
    runtime: dict[str, Any] | None = None
    if runtime_artifact is not None:
        if result.get("runtime_fingerprint_artifact_sha256") != file_sha256(runtime_path):
            failures.append("stage_n_result_runtime_artifact_sha256_mismatch")
        if runtime_artifact.get("schema_version") != STAGE_N_RUNTIME_ARTIFACT_SCHEMA:
            failures.append("stage_n_runtime_artifact_schema_mismatch")
        if runtime_artifact.get("kind") != STAGE_N_RUNTIME_ARTIFACT_KIND:
            failures.append("stage_n_runtime_artifact_kind_mismatch")
        raw_runtime = runtime_artifact.get("runtime_fingerprint")
        if not isinstance(raw_runtime, Mapping):
            failures.append("stage_n_runtime_artifact_fingerprint_missing")
        else:
            runtime = dict(raw_runtime)
            document_sha = runtime_fingerprint_sha256(runtime)
            if runtime_artifact.get("runtime_fingerprint_sha256") != document_sha:
                failures.append("stage_n_runtime_artifact_document_sha256_mismatch")
            if result.get("runtime_fingerprint_sha256") != document_sha:
                failures.append("stage_n_result_runtime_fingerprint_sha256_mismatch")
            if result.get("runtime_fingerprint") != runtime:
                failures.append("stage_n_result_runtime_fingerprint_content_mismatch")
            if grc is not None and grc.get("runtime_fingerprint") != runtime:
                failures.append("stage_n_run_contract_runtime_artifact_mismatch")
            for field in ("gpu_uuid", "gpu_pci_bus_id", "num_workers"):
                if result.get(field) != runtime.get(field):
                    failures.append(f"stage_n_runtime_artifact_contradicts_result:{field}")

    for label, actual_path, expected_kind, result_field in (
        ("smoke", Path(smoke_results_path), STAGE_N_SMOKE_RESULT_KIND, "smoke_results"),
        ("resume", Path(resume_results_path), STAGE_N_RESUME_RESULT_KIND, "resume_results"),
    ):
        if not _paths_equal(result.get(f"{label}_results_path"), actual_path):
            failures.append(f"stage_n_result_{label}_results_path_mismatch")
        check = _load_json_mapping_artifact(
            actual_path, label=f"stage_n_{label}_result_artifact", failures=failures
        )
        if check is not None:
            if result.get(f"{label}_results_sha256") != file_sha256(actual_path):
                failures.append(f"stage_n_result_{label}_results_sha256_mismatch")
            if result.get(result_field) != check:
                failures.append(f"stage_n_result_{label}_results_content_mismatch")
            failures.extend(
                validate_stage_n_check_result(
                    check,
                    expected_kind=expected_kind,
                    expected_authorization_sha256=str(
                        result.get("stage_authorization_sha256") or ""
                    ),
                    expected_governed_run_contract_sha256=str(
                        result.get("governed_run_contract_sha256") or ""
                    ),
                    expected_runtime_fingerprint_sha256=str(
                        result.get("runtime_fingerprint_sha256") or ""
                    ),
                    expected_checkpoint_path=checkpoint_path,
                    expected_checkpoint_sha256=(
                        file_sha256(checkpoint_path) if Path(checkpoint_path).is_file() else None
                    ),
                    expected_checkpoint_step=final_step,
                    expected_authorization_path=authorization_path,
                    expected_governed_run_contract_path=governed_run_contract_path,
                )
            )
            if expected_invocation_root is not None:
                evidence_value = check.get("evidence_artifact_path")
                if isinstance(evidence_value, str) and evidence_value.strip():
                    try:
                        Path(evidence_value).expanduser().resolve().relative_to(
                            expected_invocation_root
                        )
                    except ValueError:
                        failures.append(
                            f"stage_n_{label}_evidence_escapes_canonical_invocation_root"
                        )

    if final_step is None:
        failures.append("stage_n_result_final_checkpoint_step_invalid")
        final_step = -1
    checkpoint = Path(checkpoint_path)
    if not _paths_equal(result.get("final_checkpoint_path"), checkpoint):
        failures.append("stage_n_result_final_checkpoint_path_mismatch")
    if not checkpoint.is_file():
        failures.append(f"stage_n_final_checkpoint_not_found:{checkpoint}")
    else:
        if result.get("final_checkpoint_sha256") != file_sha256(checkpoint):
            failures.append("stage_n_result_final_checkpoint_sha256_does_not_match_bytes")
        if grc is not None:
            failures.extend(
                validate_stage_n_checkpoint_artifact(
                    checkpoint,
                    result=result,
                    governed_run_contract=grc,
                    expected_final_step=int(final_step),
                )
            )
    return list(dict.fromkeys(failures))


def publish_stage_n_completion(
    out_dir: str | Path,
    *,
    governed_run_contract: Mapping[str, Any],
    governed_run_contract_path: str | Path,
    authorization: Mapping[str, Any],
    authorization_path: str | Path,
    runtime_fingerprint_path: str | Path,
    final_checkpoint_path: str | Path,
    final_checkpoint_step: int,
    smoke_results_path: str | Path,
    resume_results_path: str | Path,
    smoke_results: Mapping[str, Any] | None = None,
    resume_results: Mapping[str, Any] | None = None,
    final_sampler_state: Mapping[str, Any],
) -> dict[str, Any]:
    """The canonical Stage-N completion publication path.

    Reconstructs the result from the ACTUAL bound artifacts, validates it against them,
    publishes atomically once, and returns the SHA for independent owner review. It never
    marks anything AUTHORIZED and never advances to Stage O.
    """
    auth_path = Path(authorization_path)
    require(auth_path.is_file(), f"Stage-N authorization not found: {auth_path}")
    require(
        file_sha256(auth_path) == governed_run_contract.get("stage_authorization_sha256"),
        "Stage-N authorization bytes differ from the governed invocation authority",
    )
    completion = authorization.get("stage_n_completion")
    require(
        isinstance(completion, Mapping)
        and isinstance(completion.get("expected_final_step"), int)
        and not isinstance(completion.get("expected_final_step"), bool),
        "Stage-N authorization must bind stage_n_completion.expected_final_step",
    )
    authorized_step = int(completion["expected_final_step"])
    require(
        int(final_checkpoint_step) == authorized_step,
        "caller final checkpoint step differs from the Stage-N authorization",
    )
    canonical_invocation = invocation_directory(
        str(authorization.get("allowed_output_root", "")),
        "stage_a",
        file_sha256(auth_path),
    )
    require(
        Path(out_dir).expanduser().resolve() == canonical_invocation,
        "Stage-N result output is not the authorization-derived invocation root",
    )
    ckpt = Path(final_checkpoint_path)
    runtime_path = Path(runtime_fingerprint_path)
    smoke_path = Path(smoke_results_path)
    resume_path = Path(resume_results_path)
    load_failures: list[str] = []
    runtime_artifact = _load_json_mapping_artifact(
        runtime_path, label="stage_n_runtime_fingerprint_artifact", failures=load_failures
    )
    smoke = _load_json_mapping_artifact(
        smoke_path, label="stage_n_smoke_result_artifact", failures=load_failures
    )
    resume = _load_json_mapping_artifact(
        resume_path, label="stage_n_resume_result_artifact", failures=load_failures
    )
    require(
        not load_failures
        and runtime_artifact is not None
        and smoke is not None
        and resume is not None,
        "refusing to construct a Stage-N result from missing/malformed artifacts:\n  - "
        + "\n  - ".join(load_failures),
    )
    raw_runtime = runtime_artifact.get("runtime_fingerprint")
    require(
        isinstance(raw_runtime, Mapping),
        "Stage-N runtime artifact has no runtime_fingerprint object",
    )
    if smoke_results is not None:
        require(dict(smoke_results) == smoke, "supplied smoke result differs from its artifact")
    if resume_results is not None:
        require(dict(resume_results) == resume, "supplied resume result differs from its artifact")
    result = stage_n_result_document(
        governed_run_contract=governed_run_contract,
        governed_run_contract_path=governed_run_contract_path,
        final_checkpoint_path=str(ckpt),
        final_checkpoint_sha256=file_sha256(ckpt),
        final_checkpoint_step=int(final_checkpoint_step),
        smoke_results=smoke,
        smoke_results_path=smoke_path,
        resume_results=resume,
        resume_results_path=resume_path,
        runtime_fingerprint_path=runtime_path,
        final_sampler_state=final_sampler_state,
    )
    result["stage_authorization_path"] = str(Path(authorization_path))
    result["runtime_fingerprint"] = dict(raw_runtime)
    result["runtime_fingerprint_sha256"] = runtime_fingerprint_sha256(raw_runtime)

    failures = validate_stage_n_result_against_artifacts(
        result,
        authorization=authorization,
        authorization_path=authorization_path,
        governed_run_contract=governed_run_contract,
        governed_run_contract_path=governed_run_contract_path,
        runtime_fingerprint_path=runtime_path,
        checkpoint_path=ckpt,
        smoke_results_path=smoke_path,
        resume_results_path=resume_path,
        exact_plan_path=governed_run_contract.get("exact_plan_path"),
        pilot_acceptance_path=governed_run_contract.get("pilot_acceptance_path"),
        expected_final_step=int(final_checkpoint_step),
    )
    require(
        not failures,
        "refusing to publish a Stage-N result that does not bind its artifacts:\n  - "
        + "\n  - ".join(failures),
    )
    published = publish_stage_n_result(out_dir, result)
    published["hard_stop"] = "STOPPED_FOR_INDEPENDENT_OWNER_REVIEW"
    published["stage_o_authorized"] = False
    return published


def derive_stage_o_resume_binding(stage_n_result: Mapping[str, Any]) -> dict[str, Any]:
    """R3 Part 15: Stage-O resume fields are DERIVED from the accepted Stage-N chain.

    They are never separately caller-selected, so a Stage-O authorization cannot point at a
    different checkpoint than the one Stage N actually ended on.
    """
    return {
        "mode": "RESUME_EXACT_CHECKPOINT",
        "checkpoint_path": stage_n_result["final_checkpoint_path"],
        "checkpoint_sha256": stage_n_result["final_checkpoint_sha256"],
        "expected_step": int(stage_n_result["final_checkpoint_step"]),
        "stage": "stage_a",
        "governed_run_contract_sha256": stage_n_result["governed_run_contract_sha256"],
        "source_stage_authorization_path": stage_n_result["stage_authorization_path"],
        "source_stage_authorization_sha256": stage_n_result["stage_authorization_sha256"],
        "source_invocation_run_contract_path": stage_n_result["governed_run_contract_path"],
        "source_invocation_run_contract_sha256": stage_n_result[
            "governed_run_contract_artifact_sha256"
        ],
        "source_base_governed_identity_digest": stage_n_result["base_governed_identity_sha256"],
        "source_checkpoint_path": stage_n_result["final_checkpoint_path"],
        "source_checkpoint_sha256": stage_n_result["final_checkpoint_sha256"],
        "source_checkpoint_step": int(stage_n_result["final_checkpoint_step"]),
        "source_checkpoint_stage": "stage_a",
        "source_active_stage": "stage_a",
        "source_sampler_seed": STAGE_A_SAMPLER_SEED,
        "source_permutation_identity": stage_n_result["final_sampler_permutation_identity"],
        "source_range_start_position": stage_n_result["final_sampler_range_start_position"],
        "source_invocation_range_start_position": stage_n_result[
            "final_sampler_invocation_range_start_position"
        ],
        "source_range_stop_position": stage_n_result["final_sampler_range_stop_position"],
        "source_cursor": stage_n_result["final_sampler_cursor"],
    }


def validate_stage_o_runtime_artifact(
    chain: Mapping[str, Any], stage_n_result: Mapping[str, Any], observed_runtime: Mapping[str, Any]
) -> list[str]:
    """R3 Part 14: load the accepted Stage-N runtime artifact from disk and rehash it.

    A redundant raw copy inside the Stage-O authorization is never trusted: the artifact
    named by the accepted result is opened, its file SHA and canonical document SHA are both
    required to match, and its load-bearing fields must agree with the observed runtime.
    """
    failures: list[str] = []
    path_value = stage_n_result.get("runtime_fingerprint_path")
    if not path_value:
        return ["stage_o_chain_missing_stage_n_runtime_fingerprint_path"]
    path = Path(str(path_value))
    if not _paths_equal(chain.get("stage_n_runtime_fingerprint_path"), path):
        failures.append("stage_o_chain_runtime_artifact_path_contradicts_accepted_result")
    if not path.is_file():
        return [f"accepted_stage_n_runtime_artifact_not_found:{path}"]

    artifact_sha = file_sha256(path)
    if artifact_sha != stage_n_result.get("runtime_fingerprint_artifact_sha256"):
        failures.append("accepted_stage_n_runtime_artifact_file_sha256_mismatch")
    if artifact_sha != chain.get("stage_n_runtime_fingerprint_artifact_sha256"):
        failures.append("stage_o_chain_runtime_artifact_sha256_contradicts_accepted_result")
    artifact = _load_json_mapping_artifact(
        path, label="accepted_stage_n_runtime_artifact", failures=failures
    )
    if artifact is None:
        return list(dict.fromkeys(failures))
    if artifact.get("schema_version") != STAGE_N_RUNTIME_ARTIFACT_SCHEMA:
        failures.append("accepted_stage_n_runtime_artifact_schema_mismatch")
    if artifact.get("kind") != STAGE_N_RUNTIME_ARTIFACT_KIND:
        failures.append("accepted_stage_n_runtime_artifact_kind_mismatch")
    raw = artifact.get("runtime_fingerprint")
    if not isinstance(raw, Mapping):
        failures.append("accepted_stage_n_runtime_artifact_fingerprint_missing")
        return list(dict.fromkeys(failures))
    document_sha = runtime_fingerprint_sha256(raw)
    if artifact.get("runtime_fingerprint_sha256") != document_sha:
        failures.append("accepted_stage_n_runtime_artifact_self_sha256_mismatch")
    if document_sha != stage_n_result.get("runtime_fingerprint_sha256"):
        failures.append("accepted_stage_n_runtime_artifact_document_sha256_mismatch")
    if stage_n_result.get("runtime_fingerprint") != raw:
        failures.append("accepted_stage_n_runtime_artifact_content_mismatch")
    for field in STAGE_N_O_RUNTIME_COMPARISON_FIELDS:
        if field in observed_runtime and raw.get(field) != observed_runtime.get(field):
            failures.append(
                f"stage_o_runtime_differs_from_accepted_stage_n_artifact:{field}: "
                f"stage_n={raw.get(field)!r}, observed={observed_runtime.get(field)!r}"
            )
    return list(dict.fromkeys(failures))


def validate_stage_o_chain(
    authorization: Mapping[str, Any],
    *,
    observed_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate the accepted Stage-N chain and derive the only Stage-O resume binding."""
    failures: list[str] = []
    chain = authorization.get("stage_n_chain")
    if not isinstance(chain, Mapping):
        return {
            "valid": False,
            "failures": ["stage_o_authorization_missing_stage_n_chain"],
            "derived_resume": None,
        }
    for field in STAGE_O_REQUIRED_CHAIN_FIELDS:
        if chain.get(field) in (None, ""):
            failures.append(f"stage_o_chain_missing_field:{field}")
    if failures:
        return {"valid": False, "failures": failures, "derived_resume": None}

    result_path = Path(str(chain["accepted_stage_n_result_path"]))
    stage_n = _load_json_mapping_artifact(
        result_path, label="accepted_stage_n_result", failures=failures
    )
    result_sha = file_sha256(result_path) if result_path.is_file() else None
    if result_sha != chain.get("accepted_stage_n_result_sha256"):
        failures.append("accepted_stage_n_result_sha256_mismatch")
    if stage_n is None:
        return {
            "valid": False,
            "failures": list(dict.fromkeys(failures)),
            "accepted_stage_n_result_sha256": result_sha,
            "derived_resume": None,
            "requires_new_stage_n": False,
        }

    failures.extend(validate_stage_n_result(stage_n))
    failures.extend(
        validate_stage_n_result_against_artifacts(
            stage_n,
            authorization_path=str(stage_n.get("stage_authorization_path") or ""),
            governed_run_contract_path=str(stage_n.get("governed_run_contract_path") or ""),
            exact_plan_path=str(stage_n.get("exact_plan_path") or ""),
            pilot_acceptance_path=str(stage_n.get("pilot_acceptance_path") or ""),
            runtime_fingerprint_path=str(stage_n.get("runtime_fingerprint_path") or ""),
            checkpoint_path=str(stage_n.get("final_checkpoint_path") or ""),
            smoke_results_path=str(stage_n.get("smoke_results_path") or ""),
            resume_results_path=str(stage_n.get("resume_results_path") or ""),
            expected_final_step=int(stage_n.get("final_checkpoint_step", -1)),
        )
    )

    acceptance_path = Path(str(chain["stage_n_owner_acceptance_path"]))
    acceptance = _load_json_mapping_artifact(
        acceptance_path, label="stage_n_owner_acceptance", failures=failures
    )
    if acceptance_path.is_file() and file_sha256(acceptance_path) != chain.get(
        "stage_n_owner_acceptance_sha256"
    ):
        failures.append("stage_n_owner_acceptance_sha256_mismatch")
    if acceptance is not None:
        if acceptance.get("accepted_stage_n_result_sha256") != result_sha:
            failures.append("stage_n_owner_acceptance_does_not_accept_this_result")
        if acceptance.get("stage_n_result_owner_verdict") != "ACCEPTED":
            failures.append("stage_n_result_not_owner_accepted")

    failures.extend(validate_stage_o_runtime_artifact(chain, stage_n, observed_runtime))
    accepted_fp_sha = stage_n.get("runtime_fingerprint_sha256")
    if chain.get("stage_n_runtime_fingerprint_sha256") != accepted_fp_sha:
        failures.append("stage_o_chain_runtime_fingerprint_sha256_contradicts_accepted_result")
    observed_fp_sha = observed_runtime.get("runtime_fingerprint_sha256") or (
        runtime_fingerprint_sha256(observed_runtime) if observed_runtime else None
    )
    if accepted_fp_sha and observed_fp_sha and accepted_fp_sha != observed_fp_sha:
        failures.append("stage_o_runtime_differs_from_accepted_stage_n:runtime_fingerprint_sha256")

    for chain_field, result_field in (
        ("stage_n_authorization_sha256", "stage_authorization_sha256"),
        ("stage_n_governed_run_contract_sha256", "governed_run_contract_sha256"),
        (
            "stage_n_governed_run_contract_artifact_sha256",
            "governed_run_contract_artifact_sha256",
        ),
        ("stage_n_runtime_fingerprint_sha256", "runtime_fingerprint_sha256"),
        ("stage_n_runtime_fingerprint_artifact_sha256", "runtime_fingerprint_artifact_sha256"),
        ("stage_n_gpu_uuid", "gpu_uuid"),
        ("stage_n_gpu_pci_bus_id", "gpu_pci_bus_id"),
        ("stage_n_trainer_head", "trainer_head"),
        ("stage_n_trainer_execution_bundle_sha256", "trainer_execution_bundle_sha256"),
        ("stage_n_exact_run_plan_sha256", "exact_run_plan_sha256"),
        ("stage_n_final_checkpoint_path", "final_checkpoint_path"),
        ("stage_n_final_checkpoint_sha256", "final_checkpoint_sha256"),
        ("stage_n_final_checkpoint_step", "final_checkpoint_step"),
    ):
        if chain.get(chain_field) != stage_n.get(result_field):
            failures.append(f"stage_o_chain_contradicts_accepted_stage_n_result:{chain_field}")

    accepted_runtime = dict(stage_n.get("runtime_fingerprint") or {})
    for field in STAGE_N_O_RUNTIME_COMPARISON_FIELDS:
        if field in observed_runtime and accepted_runtime.get(field) != observed_runtime.get(field):
            failures.append(
                f"stage_o_runtime_differs_from_accepted_stage_n:{field}: "
                f"stage_n={accepted_runtime.get(field)!r}, observed={observed_runtime.get(field)!r}"
            )

    stage_n_resume = derive_stage_o_resume_binding(stage_n)
    supplied_resume = authorization.get("resume")
    selected_resume: Mapping[str, Any] = stage_n_resume
    source_stage = (
        supplied_resume.get("source_checkpoint_stage")
        if isinstance(supplied_resume, Mapping)
        else None
    )
    if source_stage == "stage_a":
        if supplied_resume != stage_n_resume:
            failures.append(
                "stage_o_authorization_resume_is_not_mechanically_derived_stage_n_resume"
            )
    elif source_stage == "stage_b" and isinstance(supplied_resume, Mapping):
        # A later Stage-O restart still authenticates the accepted Stage-N provenance, but
        # its executable continuation is mechanically bound to the already-published Stage-B
        # source invocation. Replacing that with the old Stage-N checkpoint would replay the
        # A->B transition and lose Stage-B progress.
        verified_stage_b = verify_resume_source_authority(supplied_resume, transition=None)
        failures.extend(verified_stage_b["failures"])
        stage_b_source = verified_stage_b.get("source_invocation_run_contract")
        if not isinstance(stage_b_source, Mapping):
            failures.append("stage_o_stage_b_restart_source_grc_missing")
        else:
            if stage_b_source.get("stage") != "stage_b":
                failures.append("stage_o_stage_b_restart_source_stage_mismatch")
            if stage_b_source.get("scope") != "STAGE_O":
                failures.append("stage_o_stage_b_restart_source_scope_mismatch")
        if supplied_resume.get("source_base_governed_identity_digest") != stage_n.get(
            "base_governed_identity_sha256"
        ):
            failures.append("stage_o_stage_b_restart_base_identity_differs_from_stage_n")
        selected_resume = dict(supplied_resume)
    else:
        failures.append("stage_o_resume_source_stage_must_be_stage_a_or_stage_b")

    checkpoint = Path(str(selected_resume["checkpoint_path"]))
    if not checkpoint.is_file():
        failures.append(f"stage_o_resume_checkpoint_absent:{checkpoint}")
    elif file_sha256(checkpoint) != selected_resume["checkpoint_sha256"]:
        failures.append("stage_o_resume_checkpoint_bytes_do_not_match_accepted_sha256")

    failures = list(dict.fromkeys(failures))
    return {
        "valid": not failures,
        "failures": failures,
        "accepted_stage_n_result_sha256": result_sha,
        "derived_resume": dict(selected_resume),
        "requires_new_stage_n": bool([
            failure
            for failure in failures
            if failure.startswith("stage_o_runtime_differs_from_accepted_stage_n")
        ]),
    }
