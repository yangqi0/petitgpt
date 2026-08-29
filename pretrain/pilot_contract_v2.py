#!/usr/bin/env python3
"""P-PILOT-CONTRACT-V2.2: the frozen pre-GPU pilot authority, as executable contract.

This module is the machine-readable half of ``docs/PILOT_CONTRACT_V2_2.md``. It holds the
owner-frozen constants and the *pure* decision logic -- candidate grids, learning-rate
schedules, eligibility predicates, selection and tie-break rules, deterministic pilot index
generation, and the budget ceilings -- with no training, no CUDA and no I/O side effects.

V2.2 supersedes PLAYBOOK.md sections 11.1-11.3, the earlier incomplete pilot protocol, the
Pro V2 proposal, and the V2.1 text wherever V2.2 edits it. It does NOT supersede the final model
architecture, the tokenizer, accepted Stage-I / Stage-M data, the Stage-A-then-Stage-B order,
the continuous WSD timeline, production data identities, or the existing Stage-P provenance
policy.

Nothing in this module can authorize a pilot launch: ``authorization_template`` always
serializes ``authorization_status="NOT_AUTHORIZED"``, and ``require_launch_authorization``
raises unconditionally.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from statistics import median
import sys
from types import MappingProxyType
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# SUPERSEDED IN PART by P-PILOT-CONTRACT-V2.3 (pretrain/pilot_contract_v2_3.py): the
# optimizer decision here (adamw) no longer governs. Retained unchanged for history and
# for the non-optimizer freezes V2.3 restates.
CONTRACT_VERSION = "P-PILOT-CONTRACT-V2.2"
CONTRACT_SCHEMA = "petitgpt-pilot-contract-v2.2"
AUTHORIZATION_TEMPLATE_SCHEMA = "petitgpt-pilot-authorization-template-v1"
BASE_FINGERPRINT_SCHEMA = "petitgpt-pilot-base-runtime-fingerprint-v1"
RUN_META_SCHEMA = "petitgpt-pilot-run-meta-v1"
PILOT_INDEX_SCHEMA = "petitgpt-pilot-indices-v1"

SUPERSEDES = (
    "PLAYBOOK.md 11.1",
    "PLAYBOOK.md 11.2",
    "PLAYBOOK.md 11.3",
    "incomplete-earlier-pilot-protocol",
    "pro-v2-proposal",
    "fable-v2.1-where-v2.2-edits-it",
)
DOES_NOT_SUPERSEDE = (
    "final-model-architecture",
    "tokenizer",
    "accepted-stage-i-data",
    "accepted-stage-m-data",
    "stage-a-then-stage-b-order",
    "continuous-wsd-timeline",
    "production-data-identities",
    "stage-p-provenance-policy",
)


class PilotContractError(RuntimeError):
    """Controlled contract failure. Never raised for a condition a caller may ignore."""


def require(condition: object, message: str) -> None:
    if not condition:
        raise PilotContractError(message)


# --------------------------------------------------------------------- 2.1 hardware

HARDWARE = MappingProxyType({
    "training_authority_gpu_class": "NVIDIA GeForce RTX 4090 24GB class",
    "rtx_4000_ada_training_authority": "NONE",
    "stage_n_requirement": "must run on the exact Stage-O pod instance and base fingerprint",
    "base_fingerprint_change_after_stage_n": "rerun Stage N before Stage O",
    "runtime_estimate_authority": "Stage-N measurements on the actual Stage-O pod",
})


def gpu_has_training_authority(gpu_name: object) -> bool:
    """Only an RTX 4090 24GB-class device may run a load-bearing training pilot."""
    return isinstance(gpu_name, str) and "4090" in gpu_name


# --------------------------------------------------------------------- 2.2 effective batch

EFFECTIVE_BATCH_TOKENS = 262144
SEQUENCES_PER_OPTIMIZER_UPDATE = 128
SEQ_LEN = 2048

require(
    SEQUENCES_PER_OPTIMIZER_UPDATE * SEQ_LEN == EFFECTIVE_BATCH_TOKENS,
    "effective batch constants are internally inconsistent",
)


def frozen_grad_accum(micro_bsz: int) -> int:
    """FROZEN_GRAD_ACCUM = 128 / FROZEN_MICRO_BSZ, exactly."""
    micro_bsz = int(micro_bsz)
    require(micro_bsz > 0, f"micro_bsz must be positive, got {micro_bsz}")
    require(
        SEQUENCES_PER_OPTIMIZER_UPDATE % micro_bsz == 0,
        f"micro_bsz {micro_bsz} does not divide "
        f"{SEQUENCES_PER_OPTIMIZER_UPDATE} sequences per optimizer update",
    )
    return SEQUENCES_PER_OPTIMIZER_UPDATE // micro_bsz


# --------------------------------------------------------------------- 2.3 Phase MB

MB_MICRO_BSZ_ORDER = (16, 8, 4, 2, 1)
MB_PROBE_PEAK_LR = 3e-4
MB_PROBE_WARMUP_UPDATES = 10
MB_PROBE_UPDATES = 40
MB_MEASURED_FIRST_UPDATE = 11
MB_MODEL_INIT_SEED = 20260829
MB_VRAM_RESERVED_FRACTION_CEILING = 0.90
MB_TIE_THROUGHPUT_RELATIVE = 0.03
MB_TIE_VRAM_MIB = 256


def mb_candidate_grid() -> tuple[dict[str, Any], ...]:
    """The ten unconditional probes, in the frozen run order.

    All ten run. D-037 and any earlier measurement may NOT exclude a candidate.
    """
    grid: list[dict[str, Any]] = []
    for micro_bsz in MB_MICRO_BSZ_ORDER:
        for compile_on in (False, True):
            grid.append({
                "candidate_id": f"mb_micro{micro_bsz}_compile{'on' if compile_on else 'off'}",
                "phase": "MB",
                "micro_bsz": micro_bsz,
                "grad_accum": frozen_grad_accum(micro_bsz),
                "compile": compile_on,
                "updates": MB_PROBE_UPDATES,
                "peak_lr": MB_PROBE_PEAK_LR,
                "warmup_updates": MB_PROBE_WARMUP_UPDATES,
                "model_init_seed": MB_MODEL_INIT_SEED,
            })
    require(len(grid) == 10, "the Phase-MB grid must hold exactly ten probes")
    return tuple(grid)


def mb_lr(update: int) -> float:
    """lr(u) = 3e-4 * min(u / 10, 1.0), for the update about to be applied (u >= 1)."""
    update = int(update)
    require(update >= 1, f"optimizer updates are 1-based, got {update}")
    return MB_PROBE_PEAK_LR * min(update / MB_PROBE_WARMUP_UPDATES, 1.0)


def mb_candidate_eligible(
    result: Mapping[str, Any], physical_vram_bytes: int
) -> tuple[bool, tuple[str, ...]]:
    """Every eligibility condition from 2.3, evaluated together so all failures are reported."""
    failures: list[str] = []
    if int(result.get("completed_updates", -1)) != MB_PROBE_UPDATES:
        failures.append("did_not_complete_40_updates")
    if result.get("oom") or result.get("uncontrolled_exception"):
        failures.append("oom_or_uncontrolled_exception")
    if not result.get("all_losses_finite", False):
        failures.append("non_finite_loss")
    if not result.get("all_grad_norms_finite", False):
        failures.append("non_finite_grad_norm")
    if not result.get("adamw_state_complete", False):
        failures.append("adamw_state_incomplete")
    reserved = result.get("max_memory_reserved_bytes")
    if reserved is None or int(physical_vram_bytes) <= 0:
        failures.append("vram_measurement_missing")
    elif int(reserved) > MB_VRAM_RESERVED_FRACTION_CEILING * int(physical_vram_bytes):
        failures.append("vram_reserved_above_90_percent")
    if result.get("compile") and not result.get("canonical_compile_path", False):
        failures.append("compile_silent_fallback")
    return (not failures), tuple(failures)


def mb_select(results: Sequence[Mapping[str, Any]], physical_vram_bytes: int) -> dict[str, Any]:
    """Deterministic Phase-MB selection with the frozen tie-break ladder."""
    eligible = []
    for r in results:
        ok, failures = mb_candidate_eligible(r, physical_vram_bytes)
        if ok:
            eligible.append(r)
    if not eligible:
        return {"outcome": "PHASE_MB_ABORT", "reason": "no eligible candidate", "eligible": 0}

    fastest = max(float(r["median_tokens_per_sec"]) for r in eligible)
    tied = [
        r
        for r in eligible
        if (fastest - float(r["median_tokens_per_sec"])) / fastest <= MB_TIE_THROUGHPUT_RELATIVE
    ]
    tie_break = "fastest_unique" if len(tied) == 1 else "throughput_tie"

    if len(tied) > 1:
        lowest = min(int(r["max_memory_reserved_bytes"]) for r in tied)
        window = MB_TIE_VRAM_MIB * 1024 * 1024
        vram_tied = [r for r in tied if int(r["max_memory_reserved_bytes"]) - lowest <= window]
        if len(vram_tied) == 1:
            tied, tie_break = vram_tied, "lowest_peak_reserved_vram"
        else:
            tied = vram_tied
            compile_off = [r for r in tied if not r.get("compile")]
            if compile_off and len(compile_off) != len(tied):
                tied, tie_break = compile_off, "compile_off_preferred"
            else:
                tied = compile_off or tied
                if len(tied) > 1:
                    biggest = max(int(r["micro_bsz"]) for r in tied)
                    tied = [r for r in tied if int(r["micro_bsz"]) == biggest]
                    tie_break = "larger_micro_bsz"
                else:
                    tie_break = "compile_off_preferred"

    require(len(tied) == 1, f"Phase-MB tie-break did not resolve to one candidate: {len(tied)}")
    winner = tied[0]
    return {
        "outcome": "PHASE_MB_FROZEN",
        "tie_break": tie_break,
        "eligible": len(eligible),
        "FROZEN_MICRO_BSZ": int(winner["micro_bsz"]),
        "FROZEN_GRAD_ACCUM": frozen_grad_accum(int(winner["micro_bsz"])),
        "FROZEN_COMPILE": bool(winner.get("compile")),
        "winner_candidate_id": winner.get("candidate_id"),
        "winner_median_tokens_per_sec": float(winner["median_tokens_per_sec"]),
    }


# --------------------------------------------------------------------- 2.4 pilot indices

PILOT_INDEX_SEED = 20260829
PILOT_TRAIN_ORDER_SEED_1 = 20260829
PILOT_TRAIN_ORDER_SEED_2 = 20260830
PILOT_MODEL_SEED_1 = 20260829
PILOT_MODEL_SEED_2 = 20260830
STAGE_A_EVAL_COUNT = 4096
STAGE_A_TRAIN_COUNT = 131072
STAGE_B_EVAL_COUNT = 4096


def generate_pilot_indices(stage_a_blocks: int, stage_b_blocks: int) -> dict[str, Any]:
    """The frozen deterministic draw, in the exact contract order.

    NumPy Generator(PCG64(20260829)); one generator, three draws in order:
      1. Stage-A eval, 4096 without replacement from 0..stage_a_blocks-1
      2. Stage-A train, 131072 without replacement from the REMAINING Stage-A indices
      3. Stage-B eval, 4096 without replacement, continuing the same generator
    """
    import numpy as np

    stage_a_blocks, stage_b_blocks = int(stage_a_blocks), int(stage_b_blocks)
    require(stage_a_blocks > 0 and stage_b_blocks > 0, "block universes must be positive")
    require(
        stage_a_blocks >= STAGE_A_EVAL_COUNT + STAGE_A_TRAIN_COUNT,
        f"Stage A has {stage_a_blocks} blocks, needs at least "
        f"{STAGE_A_EVAL_COUNT + STAGE_A_TRAIN_COUNT}",
    )
    require(stage_b_blocks >= STAGE_B_EVAL_COUNT, "Stage B is too small for the eval draw")

    rng = np.random.Generator(np.random.PCG64(PILOT_INDEX_SEED))
    a_eval = rng.choice(stage_a_blocks, size=STAGE_A_EVAL_COUNT, replace=False)
    remaining = np.setdiff1d(np.arange(stage_a_blocks, dtype=np.int64), a_eval, assume_unique=False)
    a_train = rng.choice(remaining, size=STAGE_A_TRAIN_COUNT, replace=False)
    b_eval = rng.choice(stage_b_blocks, size=STAGE_B_EVAL_COUNT, replace=False)

    stage_a_eval = sorted(int(v) for v in a_eval)
    stage_a_train = [int(v) for v in a_train]  # draw order, NOT sorted
    stage_b_eval = sorted(int(v) for v in b_eval)

    require(len(set(stage_a_eval)) == STAGE_A_EVAL_COUNT, "Stage-A eval draw is not unique")
    require(len(set(stage_a_train)) == STAGE_A_TRAIN_COUNT, "Stage-A train draw is not unique")
    require(len(set(stage_b_eval)) == STAGE_B_EVAL_COUNT, "Stage-B eval draw is not unique")
    require(
        not (set(stage_a_eval) & set(stage_a_train)),
        "Stage-A train and eval sets must be disjoint",
    )
    require(all(0 <= v < stage_a_blocks for v in stage_a_eval), "Stage-A eval out of range")
    require(all(0 <= v < stage_a_blocks for v in stage_a_train), "Stage-A train out of range")
    require(all(0 <= v < stage_b_blocks for v in stage_b_eval), "Stage-B eval out of range")

    return {
        "schema_version": PILOT_INDEX_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "generator": "numpy.random.Generator(PCG64)",
        "seed": PILOT_INDEX_SEED,
        "stage_a_blocks": stage_a_blocks,
        "stage_b_blocks": stage_b_blocks,
        "stage_a_eval": stage_a_eval,
        "stage_a_train": stage_a_train,
        "stage_b_eval": stage_b_eval,
        "stage_a_eval_sha256": _index_sha256(stage_a_eval),
        "stage_a_train_sha256": _index_sha256(stage_a_train),
        "stage_b_eval_sha256": _index_sha256(stage_b_eval),
    }


def _index_sha256(values: Sequence[int]) -> str:
    """Canonical digest of an index list: newline-joined decimal integers, UTF-8."""
    payload = ("\n".join(str(int(v)) for v in values) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def train_order(stage_a_train: Sequence[int], seed: int) -> list[int]:
    """Per-run permutation of the fixed Stage-A train set (PCG64(seed))."""
    import numpy as np

    seed = int(seed)
    require(
        seed in (PILOT_TRAIN_ORDER_SEED_1, PILOT_TRAIN_ORDER_SEED_2),
        f"train-order seed must be {PILOT_TRAIN_ORDER_SEED_1} or "
        f"{PILOT_TRAIN_ORDER_SEED_2}, got {seed}",
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    order = rng.permutation(len(stage_a_train))
    return [int(stage_a_train[i]) for i in order]


SEED_SEMANTICS = MappingProxyType({
    "seed-1": {"model_init": PILOT_MODEL_SEED_1, "train_order": PILOT_TRAIN_ORDER_SEED_1},
    "seed-2": {"model_init": PILOT_MODEL_SEED_2, "train_order": PILOT_TRAIN_ORDER_SEED_2},
})


# --------------------------------------------------------------------- 2.5 optimizer

FROZEN_OPTIMIZER = "adamw"
ADAMW_BETAS = (0.9, 0.95)
ADAMW_GRAD_CLIP = 1.0
MUON_IN_SCOPE = False


def realized_adamw_config() -> dict[str, Any]:
    """Bind the COMPLETE realized AdamW configuration from src/optim.py and the trainer.

    Every value is read from the frozen repository; none is invented here. A field with two
    legitimate unfrozen values would be an owner ambiguity, not a default to pick.
    """
    import inspect

    from src.optim import ADAM_PARAM_NAME_KEYS, build_optimizer

    sig = inspect.signature(build_optimizer)
    defaults = {k: v.default for k, v in sig.parameters.items()}
    return {
        "optimizer": FROZEN_OPTIMIZER,
        "betas": list(ADAMW_BETAS),
        "betas_authority": "hard-coded at the trainer's build_optimizer call site",
        "eps": defaults["eps"],
        "eps_authority": "src.optim.build_optimizer default; not exposed by the trainer CLI",
        "weight_decay": defaults["weight_decay"],
        "weight_decay_authority": "trainer --weight_decay default, single frozen value",
        "grad_clip": ADAMW_GRAD_CLIP,
        "grad_clip_authority": "trainer --grad_clip default",
        "fused_on_cuda": True,
        "fused_authority": "build_optimizer sets fused=all(p.is_cuda), with a documented "
        "RuntimeError/TypeError fallback to non-fused",
        "param_group_membership": {
            "no_decay": "p.ndim < 2 (norm gains, biases); weight_decay 0.0",
            "adam_decay_names": list(ADAM_PARAM_NAME_KEYS),
            "adamw_grouping": "matrix_params + adam_decay_params share one weight_decay group; "
            "no_decay_params form the second group",
            "lr_ratio": 1.0,
        },
        "tied_weight_handling": "named_parameters() deduplicates tok_emb/lm_head",
        "explicit_cli_requirement": "--optimizer adamw must be passed explicitly on every "
        "command; the repository default is muon",
        "muon_in_scope": MUON_IN_SCOPE,
    }


# --------------------------------------------------------------------- 2.6 Phase LR

LR_GRID_SEED1 = (2e-4, 3e-4, 4e-4, 6e-4)
LR_RUN_UPDATES = 400
LR_WARMUP_UPDATES = 50
LR_BASELINE_WINDOW = (81, 100)
LR_GUARD_FIRST_UPDATE = 101
LR_GUARD_WINDOW = 20
LR_GUARD_MULTIPLIER = 1.5
LR_SCORE_WEIGHTS = (10, 3)
LR_TIE_RELATIVE = 0.005
LR_EDGE_LOW = {2e-4: 1e-4}
LR_EDGE_HIGH = {6e-4: 8e-4}
LR_RUN_CEILING = 8


def lr_schedule(update: int, candidate_peak_lr: float) -> float:
    """lr(u) = candidate_peak_lr * min(u / 50, 1.0); constant after warmup."""
    update = int(update)
    require(update >= 1, f"optimizer updates are 1-based, got {update}")
    return float(candidate_peak_lr) * min(update / LR_WARMUP_UPDATES, 1.0)


def lr_score(loss_a: float, loss_b: float) -> float:
    """SCORE = (10 * loss_A + 3 * loss_B) / 13."""
    wa, wb = LR_SCORE_WEIGHTS
    return (wa * float(loss_a) + wb * float(loss_b)) / (wa + wb)


def sustained_divergence(losses_by_update: Mapping[int, float]) -> dict[str, Any]:
    """The frozen rolling-median guard.

    BASELINE = median of updates 81..100. For every COMPLETE 20-update window beginning at or
    after update 101, the window median must not exceed 1.5 x BASELINE.
    """
    base = [
        float(losses_by_update[u])
        for u in range(LR_BASELINE_WINDOW[0], LR_BASELINE_WINDOW[1] + 1)
        if u in losses_by_update
    ]
    require(
        len(base) == LR_BASELINE_WINDOW[1] - LR_BASELINE_WINDOW[0] + 1,
        "baseline window 81..100 is incomplete",
    )
    baseline = median(base)
    threshold = LR_GUARD_MULTIPLIER * baseline
    last_update = max(losses_by_update) if losses_by_update else 0
    violations: list[dict[str, Any]] = []
    start = LR_GUARD_FIRST_UPDATE
    while start + LR_GUARD_WINDOW - 1 <= last_update:
        window = [losses_by_update.get(u) for u in range(start, start + LR_GUARD_WINDOW)]
        if any(v is None for v in window):
            start += 1
            continue
        wm = median(float(v) for v in window)
        if wm > threshold:
            violations.append({"window_start": start, "window_median": wm})
        start += 1
    return {
        "baseline": baseline,
        "threshold": threshold,
        "violations": violations,
        "diverged": bool(violations),
    }


def lr_candidate_eligible(result: Mapping[str, Any]) -> tuple[bool, tuple[str, ...]]:
    failures: list[str] = []
    if int(result.get("completed_updates", -1)) != LR_RUN_UPDATES:
        failures.append("did_not_complete_400_updates")
    if not result.get("all_losses_finite", False):
        failures.append("non_finite_loss")
    if not result.get("all_grad_norms_finite", False):
        failures.append("non_finite_grad_norm")
    for key in ("eval_loss_stage_a", "eval_loss_stage_b"):
        v = result.get(key)
        if v is None or not math.isfinite(float(v)):
            failures.append(f"non_finite_{key}")
    if result.get("sustained_divergence", False):
        failures.append("sustained_divergence")
    return (not failures), tuple(failures)


def _lowest_lr_among_ties(
    candidates: Sequence[Mapping[str, Any]], score_key: str
) -> Mapping[str, Any]:
    best = min(float(c[score_key]) for c in candidates)
    tied = (
        [c for c in candidates if (float(c[score_key]) - best) / best <= LR_TIE_RELATIVE]
        if best > 0
        else [c for c in candidates if float(c[score_key]) == best]
    )
    return min(tied, key=lambda c: float(c["peak_lr"]))


def lr_select_seed1(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Seed-1 selection: lowest eligible SCORE; ties within 0.5% resolved by lower LR."""
    eligible = [r for r in results if lr_candidate_eligible(r)[0]]
    if len(eligible) < 2:
        return {
            "outcome": "PHASE_LR_ABORT",
            "reason": f"fewer than two eligible seed-1 grid runs ({len(eligible)})",
            "eligible": len(eligible),
        }
    winner = _lowest_lr_among_ties(eligible, "score")
    return {
        "outcome": "SEED1_WINNER",
        "eligible": len(eligible),
        "winner_peak_lr": float(winner["peak_lr"]),
        "winner_score": float(winner["score"]),
    }


def confirmation_neighbor(winner_lr: float, grid: Sequence[float] = LR_GRID_SEED1) -> float:
    """The adjacent grid candidate to re-run under seed-2: lower, or higher if at the minimum."""
    ordered = sorted(float(v) for v in grid)
    winner_lr = float(winner_lr)
    require(winner_lr in ordered, f"{winner_lr} is not a seed-1 grid point")
    i = ordered.index(winner_lr)
    return ordered[i + 1] if i == 0 else ordered[i - 1]


def lr_confirm(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Seed-2 confirmation. FINAL_SCORE = mean(seed-1 SCORE, seed-2 SCORE)."""
    usable = []
    for p in pairs:
        if p.get("seed2_eligible", False) and p.get("seed1_eligible", True):
            usable.append({
                "peak_lr": float(p["peak_lr"]),
                "final_score": (float(p["seed1_score"]) + float(p["seed2_score"])) / 2.0,
            })
    if not usable:
        return {"outcome": "PHASE_LR_ABORT", "reason": "both confirmation candidates ineligible"}
    if len(usable) == 1:
        only = usable[0]
        return {
            "outcome": "PHASE_LR_FROZEN",
            "rule": "single_eligible_confirmation_candidate",
            "FROZEN_PEAK_LR": only["peak_lr"],
            "final_score": only["final_score"],
        }
    winner = _lowest_lr_among_ties(usable, "final_score")
    return {
        "outcome": "PHASE_LR_FROZEN",
        "rule": "final_score_lowest_ties_to_lower_lr",
        "FROZEN_PEAK_LR": float(winner["peak_lr"]),
        "final_score": float(winner["final_score"]),
    }


def edge_expansion_candidate(final_winner_lr: float) -> float | None:
    """At most one bounded edge expansion: 2e-4 -> compare 1e-4; 6e-4 -> compare 8e-4."""
    lr = float(final_winner_lr)
    if lr in LR_EDGE_LOW:
        return LR_EDGE_LOW[lr]
    if lr in LR_EDGE_HIGH:
        return LR_EDGE_HIGH[lr]
    return None


# --------------------------------------------------------------------- 2.7 production schedule

FROZEN_WARMUP_STEPS = 500
FROZEN_WARMUP_STEPS_AUTHORITY = "convention-frozen by V2.2; NOT pilot-derived"
OWNER_DECAY_INTENT_FRACTION_OF_TOTAL = 0.10
PLANNER_DECAY_FRACTION_INPUT = 0.10
MIN_LR_RATIO = 0.10

DECAY_ENCODING = MappingProxyType({
    "planner_flag": "--decay_fraction",
    "planner_flag_scope": "fraction of schedule_total_steps (= stage_a_steps + stage_b_steps)",
    "decay_steps_formula": "max(1, ceil(schedule_total_steps * decay_fraction))",
    "decay_start_formula": "schedule_total_steps - decay_steps",
    "decay_end_step": "schedule_total_steps (hard-coded by the planner)",
    "planner_guard": "raises if decay_start < stage_a_steps, i.e. decay must lie inside Stage B",
    "owner_intent_fraction": OWNER_DECAY_INTENT_FRACTION_OF_TOTAL,
    "planner_input_literal": PLANNER_DECAY_FRACTION_INPUT,
    "min_lr_ratio": MIN_LR_RATIO,
    "min_lr_flag": "--min_lr_ratio (trainer), default 0.1; must be passed explicitly",
})


def verify_decay_encoding(
    plan_boundaries: Mapping[str, Any], wsd: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify a planner-emitted plan satisfies the owner decay intent.

    All integers come from the planner. No boundary is computed here and none may become
    authority; this only checks the planner's own output against the frozen intent.
    """
    total = int(plan_boundaries["schedule_total_steps"])
    stage_b_start = int(plan_boundaries["stage_b_start_step"])
    stage_b_stop = int(plan_boundaries["stage_b_global_stop_step"])
    decay_start = int(wsd["decay_start_step"])
    decay_end = int(wsd["decay_end_step"])
    expected_steps = max(1, math.ceil(total * OWNER_DECAY_INTENT_FRACTION_OF_TOTAL))
    expected_start = total - expected_steps
    checks = {
        "decay_end_equals_stage_b_global_stop": decay_end == stage_b_stop,
        "decay_end_equals_schedule_total": decay_end == total,
        "decay_start_within_one_step_of_intent": abs(decay_start - expected_start) <= 1,
        "decay_interval_wholly_in_stage_b": decay_start >= stage_b_start,
    }
    return {
        "checks": checks,
        "all_passed": all(checks.values()),
        "expected_decay_start_step": expected_start,
        "observed_decay_start_step": decay_start,
        "rounding_difference_steps": abs(decay_start - expected_start),
        "schedule_total_steps": total,
    }


# --------------------------------------------------------------------- 2.8 budget

PHASE_MB_TRAINED_TOKEN_CEILING = 105_000_000
PHASE_LR_RUN_CEILING = LR_RUN_CEILING
GLOBAL_PILOT_TRAINED_TOKEN_CEILING = 1_000_000_000


def phase_mb_projected_tokens() -> int:
    """Ten probes x 40 updates x 262144 tokens."""
    return len(mb_candidate_grid()) * MB_PROBE_UPDATES * EFFECTIVE_BATCH_TOKENS


def phase_lr_projected_tokens(runs: int) -> int:
    return int(runs) * LR_RUN_UPDATES * EFFECTIVE_BATCH_TOKENS


def budget_status(mb_tokens: int, lr_tokens: int, lr_runs: int) -> dict[str, Any]:
    total = int(mb_tokens) + int(lr_tokens)
    breaches = []
    if int(mb_tokens) > PHASE_MB_TRAINED_TOKEN_CEILING:
        breaches.append("phase_mb_token_ceiling")
    if int(lr_runs) > PHASE_LR_RUN_CEILING:
        breaches.append("phase_lr_run_ceiling")
    if total > GLOBAL_PILOT_TRAINED_TOKEN_CEILING:
        breaches.append("global_pilot_token_ceiling")
    return {
        "phase_mb_tokens": int(mb_tokens),
        "phase_lr_tokens": int(lr_tokens),
        "phase_lr_runs": int(lr_runs),
        "total_tokens": total,
        "breaches": breaches,
        "outcome": "PILOT_ABORT" if breaches else "WITHIN_BUDGET",
        "ceiling_may_be_increased_inside_this_contract": False,
    }


# --------------------------------------------------------------------- 2.9 checkpoint isolation

CHECKPOINT_ISOLATION = MappingProxyType({
    "candidates_always_start_fresh": True,
    "pilot_checkpoint_may_initialize_another_candidate": False,
    "pilot_checkpoint_may_initialize_stage_n": False,
    "pilot_checkpoint_may_initialize_stage_o": False,
    "recovery_checkpoints_resume_same_candidate_only": True,
    "recovery_checkpoint_required_bindings": (
        "candidate_config",
        "seed",
        "pilot_index_hashes",
        "contract_sha256",
        "implementation_head",
        "runtime_fingerprint_sha256",
    ),
})


def reject_pilot_checkpoint_as_initialization(purpose: str) -> None:
    """A pilot checkpoint may never initialize another candidate, Stage N, or Stage O."""
    raise PilotContractError(
        f"P-PILOT-CONTRACT-V2.2 forbids initializing {purpose!r} from a pilot checkpoint; "
        "pilot candidates always start from fresh initialization"
    )


# --------------------------------------------------------------------- authorization


def authorization_template() -> dict[str, Any]:
    """Always NOT_AUTHORIZED. This module cannot serialize an authorized launch."""
    return {
        "schema_version": AUTHORIZATION_TEMPLATE_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "authorization_status": "NOT_AUTHORIZED",
        "authorized_implementation_head": None,
        "authorized_contract_sha256": None,
        "authorized_by": None,
        "authorized_at": None,
        "note": (
            "The final implementation HEAD is bound later by a separate owner authorization "
            "manifest after independent review. No tooling in this repository may set "
            "authorization_status to AUTHORIZED on its own."
        ),
    }


def require_launch_authorization(_manifest: Mapping[str, Any] | None = None) -> None:
    """Unconditional refusal: this segment cannot authorize a pilot launch."""
    raise PilotContractError(
        "pilot launch is NOT_AUTHORIZED under P-PILOT-CONTRACT-V2.2 materialization; "
        "an external owner authorization manifest bound to a reviewed implementation HEAD "
        "is required before any Phase-MB or Phase-LR run may execute"
    )


# --------------------------------------------------------------------- canonical serialization


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def contract_document() -> dict[str, Any]:
    """The complete machine-readable V2.2 contract."""
    return {
        "schema_version": CONTRACT_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "supersedes": list(SUPERSEDES),
        "does_not_supersede": list(DOES_NOT_SUPERSEDE),
        "authorization_status": "NOT_AUTHORIZED",
        "hardware": dict(HARDWARE),
        "effective_batch": {
            "EFFECTIVE_BATCH_TOKENS": EFFECTIVE_BATCH_TOKENS,
            "SEQUENCES_PER_OPTIMIZER_UPDATE": SEQUENCES_PER_OPTIMIZER_UPDATE,
            "SEQ_LEN": SEQ_LEN,
            "FROZEN_GRAD_ACCUM_rule": "128 / FROZEN_MICRO_BSZ",
        },
        "phase_mb": {
            "grid": [dict(c) for c in mb_candidate_grid()],
            "MB_PROBE_PEAK_LR": MB_PROBE_PEAK_LR,
            "MB_PROBE_WARMUP_UPDATES": MB_PROBE_WARMUP_UPDATES,
            "updates": MB_PROBE_UPDATES,
            "measured_updates": [MB_MEASURED_FIRST_UPDATE, MB_PROBE_UPDATES],
            "model_init_seed": MB_MODEL_INIT_SEED,
            "vram_reserved_fraction_ceiling": MB_VRAM_RESERVED_FRACTION_CEILING,
            "tie_throughput_relative": MB_TIE_THROUGHPUT_RELATIVE,
            "tie_vram_mib": MB_TIE_VRAM_MIB,
            "all_probes_unconditional": True,
            "prior_measurements_may_exclude_a_candidate": False,
        },
        "pilot_indices": {
            "seed": PILOT_INDEX_SEED,
            "generator": "numpy.random.Generator(PCG64)",
            "stage_a_eval_count": STAGE_A_EVAL_COUNT,
            "stage_a_train_count": STAGE_A_TRAIN_COUNT,
            "stage_b_eval_count": STAGE_B_EVAL_COUNT,
            "draw_order": ["stage_a_eval", "stage_a_train", "stage_b_eval"],
            "serialization": {
                "stage_a_eval": "sorted ascending",
                "stage_a_train": "draw order",
                "stage_b_eval": "sorted ascending",
            },
            "seed_semantics": {k: dict(v) for k, v in SEED_SEMANTICS.items()},
        },
        "optimizer": realized_adamw_config(),
        "phase_lr": {
            "grid_seed1": list(LR_GRID_SEED1),
            "updates": LR_RUN_UPDATES,
            "warmup_updates": LR_WARMUP_UPDATES,
            "baseline_window": list(LR_BASELINE_WINDOW),
            "guard_first_update": LR_GUARD_FIRST_UPDATE,
            "guard_window": LR_GUARD_WINDOW,
            "guard_multiplier": LR_GUARD_MULTIPLIER,
            "score_weights": {
                "stage_a": LR_SCORE_WEIGHTS[0],
                "stage_b": LR_SCORE_WEIGHTS[1],
                "formula": "(10*loss_A + 3*loss_B)/13",
            },
            "tie_relative": LR_TIE_RELATIVE,
            "edge_expansion": {"2e-4": 1e-4, "6e-4": 8e-4, "at_most_once": True},
            "run_ceiling": LR_RUN_CEILING,
        },
        "production_schedule": {
            "FROZEN_WARMUP_STEPS": FROZEN_WARMUP_STEPS,
            "FROZEN_WARMUP_STEPS_AUTHORITY": FROZEN_WARMUP_STEPS_AUTHORITY,
            "OWNER_DECAY_INTENT_FRACTION_OF_TOTAL": OWNER_DECAY_INTENT_FRACTION_OF_TOTAL,
            "PLANNER_DECAY_FRACTION_INPUT": PLANNER_DECAY_FRACTION_INPUT,
            "MIN_LR_RATIO": MIN_LR_RATIO,
            "decay_encoding": dict(DECAY_ENCODING),
            "boundary_authority": "canonical planner output only; no hand-computed boundary",
        },
        "budget": {
            "PHASE_MB_TRAINED_TOKEN_CEILING": PHASE_MB_TRAINED_TOKEN_CEILING,
            "PHASE_LR_RUN_CEILING": PHASE_LR_RUN_CEILING,
            "GLOBAL_PILOT_TRAINED_TOKEN_CEILING": GLOBAL_PILOT_TRAINED_TOKEN_CEILING,
            "phase_mb_projected_tokens": phase_mb_projected_tokens(),
            "ceiling_may_be_increased_inside_this_contract": False,
        },
        "checkpoint_isolation": {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in CHECKPOINT_ISOLATION.items()
        },
    }


def contract_sha256() -> str:
    return hashlib.sha256(canonical_json_bytes(contract_document())).hexdigest()


__all__ = [
    "CONTRACT_VERSION",
    "CONTRACT_SCHEMA",
    "PilotContractError",
    "EFFECTIVE_BATCH_TOKENS",
    "SEQUENCES_PER_OPTIMIZER_UPDATE",
    "SEQ_LEN",
    "frozen_grad_accum",
    "mb_candidate_grid",
    "mb_lr",
    "mb_candidate_eligible",
    "mb_select",
    "generate_pilot_indices",
    "train_order",
    "SEED_SEMANTICS",
    "realized_adamw_config",
    "FROZEN_OPTIMIZER",
    "lr_schedule",
    "lr_score",
    "sustained_divergence",
    "lr_candidate_eligible",
    "lr_select_seed1",
    "confirmation_neighbor",
    "lr_confirm",
    "edge_expansion_candidate",
    "FROZEN_WARMUP_STEPS",
    "OWNER_DECAY_INTENT_FRACTION_OF_TOTAL",
    "PLANNER_DECAY_FRACTION_INPUT",
    "MIN_LR_RATIO",
    "verify_decay_encoding",
    "budget_status",
    "phase_mb_projected_tokens",
    "phase_lr_projected_tokens",
    "CHECKPOINT_ISOLATION",
    "reject_pilot_checkpoint_as_initialization",
    "authorization_template",
    "require_launch_authorization",
    "contract_document",
    "contract_sha256",
    "canonical_json_bytes",
    "gpu_has_training_authority",
]
