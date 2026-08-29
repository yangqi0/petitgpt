#!/usr/bin/env python3
"""P-PILOT-CONTRACT-V2.3: Muon-frozen pilot authority, as executable contract.

V2.3 supersedes the V2.2 **optimizer** decision. The owner froze the optimizer family directly
(``FREEZE_MUON_DIRECTLY``); no AdamW-vs-Muon comparison is run. Everything V2.2 froze outside the
optimizer -- effective batch, Phase-MB grid shape, deterministic pilot indices, production warmup,
the continuous WSD family, the accepted data -- is retained unchanged and re-stated here so this
contract is sufficient by itself for execution.

This module is the machine-readable half of ``docs/PILOT_CONTRACT_V2_3.md``. It is pure: frozen
constants, schedules, eligibility predicates, selection ladders, the token ledger arithmetic, the
authorization interface and the runtime gates. It performs no training and touches no GPU.

The gitignored local files CLAUDE.md / DECISIONS.md / PLAYBOOK.md / RETRAIN_PLAN.md are **not
load-bearing** for pilot execution: no executable decision here reads their bytes.
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

CONTRACT_VERSION = "P-PILOT-CONTRACT-V2.3"
CONTRACT_SCHEMA = "petitgpt-pilot-contract-v2.3"
AUTHORIZATION_SCHEMA = "petitgpt-pilot-authorization-v2.3"
BASE_FINGERPRINT_SCHEMA = "petitgpt-pilot-base-runtime-fingerprint-v2.3"
RUN_META_SCHEMA = "petitgpt-pilot-run-meta-v2.3"
PILOT_INDEX_SCHEMA = "petitgpt-pilot-indices-v2.3"
TOKEN_LEDGER_SCHEMA = "petitgpt-pilot-token-ledger-v2.3"
PILOT_CHECKPOINT_KIND = "PILOT_V2_3"

SUPERSEDES = MappingProxyType({
    "P-PILOT-CONTRACT-V2.2": (
        "optimizer decision only: V2.2 froze adamw and required an optimizer-family "
        "comparison; V2.3 freezes muon directly and removes that comparison. Every "
        "non-optimizer V2.2 freeze is retained and restated here."
    ),
})
RETAINED_FROM_V2_2 = (
    "effective-batch geometry",
    "phase-mb candidate grid shape",
    "deterministic pilot-index algorithm",
    "production warmup 500",
    "continuous WSD family",
    "owner decay intent 0.10",
)
NOT_REOPENED = (
    "model architecture",
    "tokenizer",
    "stage-i data",
    "stage-m data",
    "stage-a/b order",
    "effective batch decision",
    "production warmup decision",
    "continuous WSD family",
)


class PilotContractError(RuntimeError):
    """Controlled contract failure. Never raised for a condition a caller may ignore."""


def require(condition: object, message: str) -> None:
    if not condition:
        raise PilotContractError(message)


# --------------------------------------------------------------------- geometry (retained)

EFFECTIVE_BATCH_TOKENS = 262144
SEQUENCES_PER_OPTIMIZER_UPDATE = 128
SEQ_LEN = 2048
require(
    SEQUENCES_PER_OPTIMIZER_UPDATE * SEQ_LEN == EFFECTIVE_BATCH_TOKENS,
    "effective batch constants are internally inconsistent",
)


def frozen_grad_accum(micro_bsz: int) -> int:
    micro_bsz = int(micro_bsz)
    require(micro_bsz > 0, f"micro_bsz must be positive, got {micro_bsz}")
    require(
        SEQUENCES_PER_OPTIMIZER_UPDATE % micro_bsz == 0,
        f"micro_bsz {micro_bsz} does not divide {SEQUENCES_PER_OPTIMIZER_UPDATE} exactly",
    )
    return SEQUENCES_PER_OPTIMIZER_UPDATE // micro_bsz


# --------------------------------------------------------------------- optimizer (V2.3)

OWNER_OPTIMIZER_VERDICT = "FREEZE_MUON_DIRECTLY"
FROZEN_OPTIMIZER = "muon"
MUON_LR_ARG = 0.0
MUON_LR_POLICY = "muon_lr=0.0 means the Muon matrix groups reuse the scheduled main --lr"
MUON_MOMENTUM = 0.95
ADAMW_AUX_BETAS = (0.9, 0.95)
ADAMW_AUX_EPS = 1e-8
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
OPTIMIZER_FAMILY_COMPARISON_REQUIRED = False


def realized_muon_config() -> dict[str, Any]:
    """Bind the COMPLETE realized Muon configuration from the frozen ``src/optim.py``.

    Read from the implementation, never asserted from prose. ``src/optim.py`` is the grouping
    and mechanics authority; this function materializes what it actually does.
    """
    import inspect

    from src.optim import ADAM_PARAM_NAME_KEYS, Muon, build_optimizer

    sig = inspect.signature(build_optimizer)
    d = {k: v.default for k, v in sig.parameters.items()}
    muon_defaults = inspect.signature(Muon.__init__)
    return {
        "optimizer": FROZEN_OPTIMIZER,
        "cli": {
            "--optimizer": "muon",
            "--muon_lr": MUON_LR_ARG,
            "--muon_momentum": MUON_MOMENTUM,
            "--lr": "the single searched axis; scheduled",
            "--weight_decay": WEIGHT_DECAY,
            "--grad_clip": GRAD_CLIP,
            "explicit_optimizer_flag_required": True,
        },
        "muon_lr_resolution": (
            "build_optimizer: muon_lr = float(muon_lr) if muon_lr and muon_lr > 0 else "
            "float(lr); ratio = muon_lr / lr. With --muon_lr 0.0 this yields muon_lr == lr and "
            "ratio == 1.0, so there is no separate Muon-LR search dimension."
        ),
        "param_groups": [
            {
                "name": "muon_matrices",
                "use_muon": True,
                "membership": "2D params not matched by ADAM_PARAM_NAME_KEYS",
                "lr": "muon_lr (== main lr when --muon_lr 0.0)",
                "weight_decay": d["weight_decay"],
                "momentum": MUON_MOMENTUM,
                "nesterov": True,
                "ns_steps": d["ns_steps"],
                "lr_ratio": 1.0,
                "state_keys": ["momentum_buffer"],
            },
            {
                "name": "aux_adamw_decay",
                "use_muon": False,
                "membership": f"params whose name contains any of {list(ADAM_PARAM_NAME_KEYS)}",
                "lr": "main lr",
                "weight_decay": d["weight_decay"],
                "betas": list(ADAMW_AUX_BETAS),
                "eps": d["eps"],
                "lr_ratio": 1.0,
                "state_keys": ["step", "exp_avg", "exp_avg_sq"],
            },
            {
                "name": "aux_adamw_no_decay",
                "use_muon": False,
                "membership": "params with ndim < 2 (norm gains, biases)",
                "lr": "main lr",
                "weight_decay": 0.0,
                "betas": list(ADAMW_AUX_BETAS),
                "eps": d["eps"],
                "lr_ratio": 1.0,
                "state_keys": ["step", "exp_avg", "exp_avg_sq"],
            },
        ],
        "adam_param_name_keys": list(ADAM_PARAM_NAME_KEYS),
        "rms_matching": {
            "formula": "adjusted_lr = lr * 0.2 * sqrt(max(fan_in, fan_out))",
            "source": "Moonlight (Liu et al., 2025); matches the orthogonalized update RMS to a "
            "typical AdamW update so AdamW-tuned lr/weight_decay transfer directly",
            "newton_schulz_steps": d["ns_steps"],
            "newton_schulz_coefficients": [3.4445, -4.7750, 2.0315],
        },
        "weight_decay_style": "decoupled in both halves: p.mul_(1 - lr * wd) for Muon groups",
        "muon_group_requires_2d": "Muon.__init__ raises for any non-2D param in a use_muon group",
        "single_optimizer_instance": True,
        "checkpoint_schema_unchanged": (
            "both halves live in one Muon instance, so optimizer.state_dict() stays one object"
        ),
        "tied_weight_handling": "named_parameters() deduplicates tok_emb/lm_head",
        "muon_defaults_signature": str(muon_defaults),
        "grad_clip": GRAD_CLIP,
    }


def verify_realized_grouping(optimizer: Any) -> dict[str, Any]:
    """Verify a constructed optimizer matches the frozen V2.3 realization."""
    groups = list(optimizer.param_groups)
    muon_groups = [g for g in groups if g.get("use_muon")]
    aux_groups = [g for g in groups if not g.get("use_muon")]
    ratios = [float(g.get("lr_ratio", 1.0)) for g in groups]
    failures: list[str] = []
    if len(muon_groups) != 1:
        failures.append(f"expected exactly one Muon group, got {len(muon_groups)}")
    if not aux_groups:
        failures.append("expected at least one auxiliary AdamW group")
    if any(r != 1.0 for r in ratios):
        failures.append(f"every group lr_ratio must be 1.0, got {ratios}")
    for g in muon_groups:
        if float(g.get("momentum", -1)) != MUON_MOMENTUM:
            failures.append(f"Muon momentum must be {MUON_MOMENTUM}, got {g.get('momentum')}")
        if not all(getattr(p, "ndim", 0) == 2 for p in g["params"]):
            failures.append("Muon group contains a non-2D parameter")
    return {
        "group_count": len(groups),
        "muon_group_count": len(muon_groups),
        "aux_adamw_group_count": len(aux_groups),
        "lr_ratios": ratios,
        "all_lr_ratios_are_one": all(r == 1.0 for r in ratios),
        "failures": failures,
        "matches_frozen_realization": not failures,
    }


def verify_optimizer_state(optimizer: Any) -> dict[str, Any]:
    """Every expected per-group state type must be instantiated."""
    missing: list[str] = []
    for g in optimizer.param_groups:
        expected = ["momentum_buffer"] if g.get("use_muon") else ["exp_avg", "exp_avg_sq", "step"]
        for p in g["params"]:
            state = optimizer.state.get(p, {})
            for key in expected:
                if key not in state:
                    missing.append(f"{'muon' if g.get('use_muon') else 'aux_adamw'}:{key}")
    return {"missing_state_keys": sorted(set(missing)), "all_states_instantiated": not missing}


# --------------------------------------------------------------------- Phase MB (retained)

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
    grid = []
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
                "optimizer": FROZEN_OPTIMIZER,
                "muon_lr": MUON_LR_ARG,
                "muon_momentum": MUON_MOMENTUM,
            })
    require(len(grid) == 10, "the Phase-MB grid must hold exactly ten probes")
    return tuple(grid)


MB_REQUIRED_CANDIDATE_IDS = tuple(c["candidate_id"] for c in mb_candidate_grid())


def mb_lr(update: int) -> float:
    update = int(update)
    require(update >= 1, f"optimizer updates are 1-based, got {update}")
    return MB_PROBE_PEAK_LR * min(update / MB_PROBE_WARMUP_UPDATES, 1.0)


def mb_candidate_eligible(
    result: Mapping[str, Any], physical_vram_bytes: int
) -> tuple[bool, tuple[str, ...]]:
    failures: list[str] = []
    if int(result.get("completed_updates", -1)) != MB_PROBE_UPDATES:
        failures.append("did_not_complete_40_updates")
    if result.get("oom") or result.get("uncontrolled_exception"):
        failures.append("oom_or_uncontrolled_exception")
    if not result.get("all_losses_finite", False):
        failures.append("non_finite_loss")
    if not result.get("all_grad_norms_finite", False):
        failures.append("non_finite_grad_norm")
    if not result.get("all_optimizer_states_instantiated", False):
        failures.append("optimizer_state_incomplete")
    if not result.get("grouping_matches_contract", False):
        failures.append("optimizer_grouping_mismatch")
    if not result.get("all_lr_ratios_are_one", False):
        failures.append("lr_ratio_not_one")
    reserved = result.get("max_memory_reserved_bytes")
    if reserved is None or int(physical_vram_bytes) <= 0:
        failures.append("vram_measurement_missing")
    elif int(reserved) > MB_VRAM_RESERVED_FRACTION_CEILING * int(physical_vram_bytes):
        failures.append("vram_reserved_above_90_percent")
    if result.get("compile") and not result.get("canonical_compile_path", False):
        failures.append("compile_silent_fallback")
    return (not failures), tuple(failures)


def require_complete_mb_grid(results: Sequence[Mapping[str, Any]]) -> None:
    """No caller-supplied subset may masquerade as a complete grid."""
    ids = [r.get("candidate_id") for r in results]
    require(len(ids) == len(set(ids)), f"duplicate Phase-MB candidate identity in {ids}")
    missing = [c for c in MB_REQUIRED_CANDIDATE_IDS if c not in set(ids)]
    require(not missing, f"Phase-MB grid incomplete; missing {missing}")
    unknown = [c for c in ids if c not in set(MB_REQUIRED_CANDIDATE_IDS)]
    require(not unknown, f"unknown Phase-MB candidate identity: {unknown}")


def mb_select(results: Sequence[Mapping[str, Any]], physical_vram_bytes: int) -> dict[str, Any]:
    require_complete_mb_grid(results)
    eligible = [r for r in results if mb_candidate_eligible(r, physical_vram_bytes)[0]]
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
        tied = [r for r in tied if int(r["max_memory_reserved_bytes"]) - lowest <= window]
        if len(tied) == 1:
            tie_break = "lowest_peak_reserved_vram"
        else:
            off = [r for r in tied if not r.get("compile")]
            if off and len(off) != len(tied):
                tied, tie_break = off, "compile_off_preferred"
            else:
                tied = off or tied
                if len(tied) > 1:
                    biggest = max(int(r["micro_bsz"]) for r in tied)
                    tied = [r for r in tied if int(r["micro_bsz"]) == biggest]
                    tie_break = "larger_micro_bsz"
                else:
                    tie_break = "compile_off_preferred"
    require(len(tied) == 1, f"Phase-MB tie-break did not resolve: {len(tied)} remain")
    w = tied[0]
    return {
        "outcome": "PHASE_MB_FROZEN",
        "tie_break": tie_break,
        "eligible": len(eligible),
        "FROZEN_MICRO_BSZ": int(w["micro_bsz"]),
        "FROZEN_GRAD_ACCUM": frozen_grad_accum(int(w["micro_bsz"])),
        "FROZEN_COMPILE": bool(w.get("compile")),
        "winner_candidate_id": w.get("candidate_id"),
        "winner_median_tokens_per_sec": float(w["median_tokens_per_sec"]),
    }


# --------------------------------------------------------------------- indices (retained)

PILOT_INDEX_SEED = 20260829
PILOT_TRAIN_ORDER_SEED_1 = 20260829
PILOT_TRAIN_ORDER_SEED_2 = 20260830
PILOT_MODEL_SEED_1 = 20260829
PILOT_MODEL_SEED_2 = 20260830
STAGE_A_EVAL_COUNT = 4096
STAGE_A_TRAIN_COUNT = 131072
STAGE_B_EVAL_COUNT = 4096
REQUIRED_NUMPY_VERSION = "2.2.6"

SEED_SEMANTICS = MappingProxyType({
    "seed-1": {"model_init": PILOT_MODEL_SEED_1, "train_order": PILOT_TRAIN_ORDER_SEED_1},
    "seed-2": {"model_init": PILOT_MODEL_SEED_2, "train_order": PILOT_TRAIN_ORDER_SEED_2},
})


def require_numpy_version() -> str:
    """Real execution requires NumPy exactly 2.2.6."""
    import numpy as np

    require(
        np.__version__ == REQUIRED_NUMPY_VERSION,
        f"P-PILOT-CONTRACT-V2.3 requires NumPy exactly {REQUIRED_NUMPY_VERSION}, "
        f"found {np.__version__}",
    )
    return np.__version__


def _index_sha256(values: Sequence[int]) -> str:
    return hashlib.sha256(
        ("\n".join(str(int(v)) for v in values) + "\n").encode("utf-8")
    ).hexdigest()


def generate_pilot_indices(stage_a_blocks: int, stage_b_blocks: int) -> dict[str, Any]:
    """The retained V2.2 deterministic draw: PCG64(20260829), three draws in order."""
    import numpy as np

    require_numpy_version()
    stage_a_blocks, stage_b_blocks = int(stage_a_blocks), int(stage_b_blocks)
    require(
        stage_a_blocks >= STAGE_A_EVAL_COUNT + STAGE_A_TRAIN_COUNT,
        f"Stage A has {stage_a_blocks} blocks, needs {STAGE_A_EVAL_COUNT + STAGE_A_TRAIN_COUNT}",
    )
    require(stage_b_blocks >= STAGE_B_EVAL_COUNT, "Stage B is too small for the eval draw")
    rng = np.random.Generator(np.random.PCG64(PILOT_INDEX_SEED))
    a_eval = rng.choice(stage_a_blocks, size=STAGE_A_EVAL_COUNT, replace=False)
    remaining = np.setdiff1d(np.arange(stage_a_blocks, dtype=np.int64), a_eval)
    a_train = rng.choice(remaining, size=STAGE_A_TRAIN_COUNT, replace=False)
    b_eval = rng.choice(stage_b_blocks, size=STAGE_B_EVAL_COUNT, replace=False)
    stage_a_eval = sorted(int(v) for v in a_eval)
    stage_a_train = [int(v) for v in a_train]
    stage_b_eval = sorted(int(v) for v in b_eval)
    require(
        not (set(stage_a_eval) & set(stage_a_train)), "Stage-A train and eval sets must be disjoint"
    )
    return {
        "schema_version": PILOT_INDEX_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "generator": "numpy.random.Generator(PCG64)",
        "seed": PILOT_INDEX_SEED,
        "numpy_version": REQUIRED_NUMPY_VERSION,
        "stage_a_blocks": stage_a_blocks,
        "stage_b_blocks": stage_b_blocks,
        "stage_a_eval": stage_a_eval,
        "stage_a_train": stage_a_train,
        "stage_b_eval": stage_b_eval,
        "stage_a_eval_sha256": _index_sha256(stage_a_eval),
        "stage_a_train_sha256": _index_sha256(stage_a_train),
        "stage_b_eval_sha256": _index_sha256(stage_b_eval),
    }


def train_order(stage_a_train: Sequence[int], seed: int) -> list[int]:
    import numpy as np

    seed = int(seed)
    require(
        seed in (PILOT_TRAIN_ORDER_SEED_1, PILOT_TRAIN_ORDER_SEED_2),
        f"train-order seed must be one of the two frozen seeds, got {seed}",
    )
    order = np.random.Generator(np.random.PCG64(seed)).permutation(len(stage_a_train))
    return [int(stage_a_train[i]) for i in order]


# --------------------------------------------------------------------- Phase Muon-LR (V2.3)

LR_GRID_SEED1 = (2e-4, 3e-4, 4e-4)
LR_RUN_UPDATES = 200
LR_WARMUP_UPDATES = 25
LR_BASELINE_WINDOW = (41, 60)
LR_GUARD_WINDOW = 20
LR_GUARD_FINAL_UPDATE_RANGE = (80, 200)
LR_GUARD_MULTIPLIER = 1.5
LR_SCORE_WEIGHTS = (10, 3)
LR_TIE_RELATIVE = 0.005
LR_EDGE = MappingProxyType({2e-4: 1e-4, 4e-4: 6e-4})
LR_BLOCKS_PER_RUN = LR_RUN_UPDATES * SEQUENCES_PER_OPTIMIZER_UPDATE
LR_TOKENS_PER_RUN = LR_RUN_UPDATES * EFFECTIVE_BATCH_TOKENS
require(LR_BLOCKS_PER_RUN == 25600, "Phase-LR block accounting drifted")
require(LR_TOKENS_PER_RUN == 52428800, "Phase-LR token accounting drifted")


def lr_schedule(update: int, candidate_peak_lr: float) -> float:
    update = int(update)
    require(update >= 1, f"optimizer updates are 1-based, got {update}")
    return float(candidate_peak_lr) * min(update / LR_WARMUP_UPDATES, 1.0)


def lr_score(loss_a: float, loss_b: float) -> float:
    wa, wb = LR_SCORE_WEIGHTS
    return (wa * float(loss_a) + wb * float(loss_b)) / (wa + wb)


def sustained_divergence(losses_by_update: Mapping[int, float]) -> dict[str, Any]:
    """BASELINE = median(41..60); every complete 20-update window ENDING in 80..200 is checked."""
    base = [
        float(losses_by_update[u])
        for u in range(LR_BASELINE_WINDOW[0], LR_BASELINE_WINDOW[1] + 1)
        if u in losses_by_update
    ]
    require(
        len(base) == LR_BASELINE_WINDOW[1] - LR_BASELINE_WINDOW[0] + 1,
        "baseline window 41..60 is incomplete",
    )
    baseline = median(base)
    threshold = LR_GUARD_MULTIPLIER * baseline
    violations = []
    lo, hi = LR_GUARD_FINAL_UPDATE_RANGE
    for final_update in range(lo, hi + 1):
        start = final_update - LR_GUARD_WINDOW + 1
        window = [losses_by_update.get(u) for u in range(start, final_update + 1)]
        if any(v is None for v in window):
            continue
        wm = median(float(v) for v in window)
        if wm > threshold:
            violations.append({"window_final_update": final_update, "window_median": wm})
    return {
        "baseline": baseline,
        "threshold": threshold,
        "violations": violations,
        "diverged": bool(violations),
    }


def lr_candidate_eligible(result: Mapping[str, Any]) -> tuple[bool, tuple[str, ...]]:
    failures: list[str] = []
    if int(result.get("completed_updates", -1)) != LR_RUN_UPDATES:
        failures.append("did_not_complete_200_updates")
    for flag, name in (
        ("all_losses_finite", "non_finite_loss"),
        ("all_grad_norms_finite", "non_finite_grad_norm"),
        ("all_parameters_finite", "non_finite_parameter"),
        ("muon_momentum_states_present", "missing_muon_momentum_state"),
        ("aux_adamw_states_present", "missing_aux_adamw_state"),
        ("grouping_matches_contract", "optimizer_grouping_mismatch"),
        ("all_lr_ratios_are_one", "lr_ratio_not_one"),
    ):
        if not result.get(flag, False):
            failures.append(name)
    for key in ("eval_loss_stage_a", "eval_loss_stage_b"):
        v = result.get(key)
        if v is None or not math.isfinite(float(v)):
            failures.append(f"non_finite_{key}")
    if result.get("sustained_divergence", False):
        failures.append("sustained_divergence")
    return (not failures), tuple(failures)


def _tie_to_lower_lr(candidates: Sequence[Mapping[str, Any]], key: str) -> Mapping[str, Any]:
    best = min(float(c[key]) for c in candidates)
    if best > 0:
        tied = [c for c in candidates if (float(c[key]) - best) / best <= LR_TIE_RELATIVE]
    else:
        tied = [c for c in candidates if float(c[key]) == best]
    return min(tied, key=lambda c: float(c["peak_lr"]))


def require_complete_lr_grid(results: Sequence[Mapping[str, Any]]) -> None:
    lrs = [float(r["peak_lr"]) for r in results]
    require(len(lrs) == len(set(lrs)), f"duplicate Phase-LR candidate LR in {lrs}")
    missing = [lr for lr in LR_GRID_SEED1 if lr not in set(lrs)]
    require(not missing, f"Phase-LR initial grid incomplete; missing {missing}")


def lr_select_seed1(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require_complete_lr_grid(results)
    eligible = [r for r in results if lr_candidate_eligible(r)[0]]
    if len(eligible) < 2:
        return {
            "outcome": "PHASE_MUON_LR_ABORT",
            "reason": f"fewer than two eligible seed-1 candidates ({len(eligible)})",
            "eligible": len(eligible),
        }
    w = _tie_to_lower_lr(eligible, "score")
    return {
        "outcome": "SEED1_WINNER",
        "eligible": len(eligible),
        "winner_peak_lr": float(w["peak_lr"]),
        "winner_score": float(w["score"]),
    }


def confirmation_neighbor(winner_lr: float) -> float:
    """Adjacent lower initial-grid candidate; the higher one if the winner is the minimum."""
    ordered = sorted(float(v) for v in LR_GRID_SEED1)
    winner_lr = float(winner_lr)
    require(winner_lr in ordered, f"{winner_lr} is not an initial-grid point")
    i = ordered.index(winner_lr)
    return ordered[i + 1] if i == 0 else ordered[i - 1]


def lr_confirm(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    usable = [
        {
            "peak_lr": float(p["peak_lr"]),
            "final_score": (float(p["seed1_score"]) + float(p["seed2_score"])) / 2.0,
        }
        for p in pairs
        if p.get("seed2_eligible", False) and p.get("seed1_eligible", True)
    ]
    if not usable:
        return {
            "outcome": "PHASE_MUON_LR_ABORT",
            "reason": "both seed-2 confirmation runs ineligible",
        }
    if len(usable) == 1:
        only = usable[0]
        return {
            "outcome": "CONFIRMED",
            "rule": "single_eligible_confirmation_candidate",
            "confirmed_peak_lr": only["peak_lr"],
            "final_score": only["final_score"],
        }
    w = _tie_to_lower_lr(usable, "final_score")
    return {
        "outcome": "CONFIRMED",
        "rule": "final_score_lowest_ties_to_lower_lr",
        "confirmed_peak_lr": float(w["peak_lr"]),
        "final_score": float(w["final_score"]),
    }


def edge_candidate(confirmed_lr: float) -> float | None:
    """2e-4 -> 1e-4; 4e-4 -> 6e-4; 3e-4 -> no expansion. At most one, ever."""
    return LR_EDGE.get(float(confirmed_lr))


def lr_resolve_edge(
    *,
    incumbent_lr: float,
    incumbent_final_score: float,
    edge_lr: float | None,
    edge_seed1_eligible: bool | None = None,
    edge_seed2_eligible: bool | None = None,
    edge_seed1_score: float | None = None,
    edge_seed2_score: float | None = None,
) -> dict[str, Any]:
    """Bounded single edge expansion. Both edge runs must be individually eligible."""
    if edge_lr is None:
        return {
            "outcome": "PHASE_MUON_LR_FROZEN",
            "rule": "no_edge_expansion_defined",
            "FROZEN_PEAK_LR": float(incumbent_lr),
            "final_score": float(incumbent_final_score),
            "second_expansion_permitted": False,
        }
    if not (edge_seed1_eligible and edge_seed2_eligible):
        return {
            "outcome": "PHASE_MUON_LR_FROZEN",
            "rule": "edge_not_comparison_eligible_incumbent_remains",
            "FROZEN_PEAK_LR": float(incumbent_lr),
            "final_score": float(incumbent_final_score),
            "second_expansion_permitted": False,
        }
    edge_final = (float(edge_seed1_score) + float(edge_seed2_score)) / 2.0
    w = _tie_to_lower_lr(
        [
            {"peak_lr": float(incumbent_lr), "final_score": float(incumbent_final_score)},
            {"peak_lr": float(edge_lr), "final_score": edge_final},
        ],
        "final_score",
    )
    return {
        "outcome": "PHASE_MUON_LR_FROZEN",
        "rule": "edge_compared_lowest_ties_to_lower_lr",
        "FROZEN_PEAK_LR": float(w["peak_lr"]),
        "final_score": float(w["final_score"]),
        "edge_final_score": edge_final,
        "second_expansion_permitted": False,
    }


# --------------------------------------------------------------------- schedule (retained)

FROZEN_WARMUP_STEPS = 500
FROZEN_WARMUP_STEPS_AUTHORITY = "convention-frozen; NOT pilot-derived"
OWNER_DECAY_INTENT_FRACTION_OF_TOTAL = 0.10
PLANNER_DECAY_FRACTION_INPUT = 0.10
PRODUCTION_MIN_LR_INTENT_RATIO = 0.10
EXPECTED_SCHEDULE_TOTAL_STEPS = 49590
EXPECTED_DECAY_UPDATES = 4959
EXPECTED_DECAY_INTERVAL = (44631, 49590)


def expected_planner_geometry(stage_a_blocks: int, stage_b_blocks: int) -> dict[str, Any]:
    """Projection from the planner's own formulas. Never an authority; the plan is."""
    a = int(stage_a_blocks) // SEQUENCES_PER_OPTIMIZER_UPDATE
    b = int(stage_b_blocks) // SEQUENCES_PER_OPTIMIZER_UPDATE
    total = a + b
    decay_steps = max(1, math.ceil(total * PLANNER_DECAY_FRACTION_INPUT))
    return {
        "stage_a_stop_step": a,
        "stage_b_start_step": a,
        "schedule_total_steps": total,
        "decay_steps": decay_steps,
        "decay_start_step": total - decay_steps,
        "decay_end_step": total,
        "matches_expected": (
            total == EXPECTED_SCHEDULE_TOTAL_STEPS
            and decay_steps == EXPECTED_DECAY_UPDATES
            and (total - decay_steps, total) == EXPECTED_DECAY_INTERVAL
        ),
        "authority": "canonical planner output only; this is a projection",
    }


def wsd_lr(
    update: int,
    *,
    peak_lr: float,
    warmup_steps: int,
    decay_start_step: int,
    decay_end_step: int,
    min_lr_ratio: float = PRODUCTION_MIN_LR_INTENT_RATIO,
) -> float:
    """The canonical discrete WSD value for a 0-based optimizer update index.

    Mirrors the trainer's schedule: linear warmup, constant stable region, then linear decay to
    ``min_lr_ratio * peak_lr`` at ``decay_end_step``. The endpoint is the *mathematical* end;
    the last update actually applied is ``decay_end_step - 1``.
    """
    update = int(update)
    if update < warmup_steps:
        return peak_lr * (update + 1) / warmup_steps if warmup_steps > 0 else peak_lr
    if update < decay_start_step:
        return peak_lr
    span = max(1, int(decay_end_step) - int(decay_start_step))
    progress = min(1.0, (update - decay_start_step) / span)
    floor = peak_lr * float(min_lr_ratio)
    return peak_lr + (floor - peak_lr) * progress


def final_lr_semantics(
    *, peak_lr: float, warmup_steps: int, decay_start_step: int, decay_end_step: int
) -> dict[str, Any]:
    """Discrete final-LR semantics: endpoint intent vs the last update actually applied."""
    kw = {
        "peak_lr": peak_lr,
        "warmup_steps": warmup_steps,
        "decay_start_step": decay_start_step,
        "decay_end_step": decay_end_step,
    }
    last_applied_update = int(decay_end_step) - 1
    mathematical_endpoint = peak_lr * PRODUCTION_MIN_LR_INTENT_RATIO
    span = max(1, int(decay_end_step) - int(decay_start_step))
    return {
        "PRODUCTION_MIN_LR_INTENT_RATIO": PRODUCTION_MIN_LR_INTENT_RATIO,
        "mathematical_endpoint_step": int(decay_end_step),
        "mathematical_endpoint_lr": mathematical_endpoint,
        "last_applied_optimizer_update": last_applied_update,
        "last_applied_lr": wsd_lr(last_applied_update, **kw),
        "one_step_before_end_progress": (span - 1) / span,
        "warmup_boundary_update": int(warmup_steps),
        "lr_at_warmup_boundary": wsd_lr(int(warmup_steps), **kw),
        "lr_at_decay_start": wsd_lr(int(decay_start_step), **kw),
        "note": (
            "The last applied update is decay_end_step - 1, so its LR is the exact "
            "canonical one-step-before-end value, NOT numerically identical to the "
            "endpoint floor. Trainer scheduler mathematics must not be altered to force "
            "literal last-update equality to the floor."
        ),
    }


# --------------------------------------------------------------------- token ledger (V2.3)

PHASE_MB_TOKEN_CEILING = 105_000_000
PHASE_MB_EXPECTED_MAX = 104_857_600
PHASE_MUON_LR_TOKEN_CEILING = 370_000_000
GLOBAL_PILOT_TOKEN_CEILING = 500_000_000
PHASE_CEILINGS = MappingProxyType({"MB": PHASE_MB_TOKEN_CEILING, "LR": PHASE_MUON_LR_TOKEN_CEILING})


def check_update_within_ceilings(
    *,
    phase: str,
    phase_tokens_so_far: int,
    global_tokens_so_far: int,
    tokens_this_update: int = EFFECTIVE_BATCH_TOKENS,
) -> dict[str, Any]:
    """Called BEFORE every optimizer update: may this update be executed?"""
    require(phase in PHASE_CEILINGS, f"unknown phase {phase!r}")
    phase_after = int(phase_tokens_so_far) + int(tokens_this_update)
    global_after = int(global_tokens_so_far) + int(tokens_this_update)
    breaches = []
    if phase_after > PHASE_CEILINGS[phase]:
        breaches.append(f"phase_{phase}_token_ceiling")
    if global_after > GLOBAL_PILOT_TOKEN_CEILING:
        breaches.append("global_pilot_token_ceiling")
    return {
        "phase": phase,
        "phase_tokens_after": phase_after,
        "global_tokens_after": global_after,
        "breaches": breaches,
        "may_execute": not breaches,
        "outcome": "PILOT_ABORT" if breaches else "WITHIN_BUDGET",
    }


def phase_mb_projected_tokens() -> int:
    return len(mb_candidate_grid()) * MB_PROBE_UPDATES * EFFECTIVE_BATCH_TOKENS


require(
    phase_mb_projected_tokens() == PHASE_MB_EXPECTED_MAX,
    "Phase-MB projection drifted from the frozen expected maximum",
)


# --------------------------------------------------------------------- runtime gates

REQUIRED_GPU_NAME_EXACT = "NVIDIA GeForce RTX 4090"
REQUIRED_GPU_MIN_VRAM_MIB = 22000
REQUIRED_GPU_MAX_VRAM_MIB = 26000


def check_training_authority(gpu: Mapping[str, Any]) -> dict[str, Any]:
    """The exact RTX 4090 gate. A substring check alone is explicitly insufficient."""
    failures: list[str] = []
    name = gpu.get("name")
    if not isinstance(name, str) or name.strip() != REQUIRED_GPU_NAME_EXACT:
        failures.append(f"gpu_name_not_exactly_{REQUIRED_GPU_NAME_EXACT!r}")
    vram = gpu.get("total_vram_mib")
    if not isinstance(vram, int) or not (
        REQUIRED_GPU_MIN_VRAM_MIB <= vram <= REQUIRED_GPU_MAX_VRAM_MIB
    ):
        failures.append("vram_not_24gb_class")
    if not gpu.get("cuda_available"):
        failures.append("cuda_unavailable")
    if not gpu.get("bf16_supported"):
        failures.append("bf16_unsupported")
    return {
        "training_authority": "GRANTED" if not failures else "NONE",
        "failures": failures,
        "granted": not failures,
    }


def require_training_authority(gpu: Mapping[str, Any]) -> None:
    result = check_training_authority(gpu)
    require(
        result["granted"],
        f"GPU has no training authority under {CONTRACT_VERSION}: " + ", ".join(result["failures"]),
    )


# --------------------------------------------------------------------- checkpoint isolation

CHECKPOINT_ISOLATION = MappingProxyType({
    "checkpoint_kind": PILOT_CHECKPOINT_KIND,
    "candidates_always_start_fresh": True,
    "required_identity_fields": (
        "checkpoint_kind",
        "phase",
        "candidate_id",
        "seed_label",
        "contract_sha256",
        "implementation_head",
        "execution_bundle_sha256",
        "pilot_index_manifest_sha256",
        "runtime_fingerprint_sha256",
    ),
    "resume_requires_exact_match": (
        "candidate_id",
        "seed_label",
        "contract_sha256",
        "implementation_head",
        "pilot_index_manifest_sha256",
    ),
    "may_initialize_another_candidate": False,
    "may_initialize_stage_n": False,
    "may_initialize_stage_o": False,
})


def reject_pilot_checkpoint_as_initialization(purpose: str) -> None:
    raise PilotContractError(
        f"{CONTRACT_VERSION} forbids initializing {purpose!r} from a pilot checkpoint "
        f"(checkpoint_kind={PILOT_CHECKPOINT_KIND}); pilot candidates always start fresh"
    )


def check_pilot_resume(
    checkpoint: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Resume is allowed only for the exact same candidate under the exact same bindings."""
    failures = []
    if checkpoint.get("checkpoint_kind") != PILOT_CHECKPOINT_KIND:
        failures.append("not_a_pilot_v2_3_checkpoint")
    for field in CHECKPOINT_ISOLATION["resume_requires_exact_match"]:
        if checkpoint.get(field) != candidate.get(field):
            failures.append(f"mismatch:{field}")
    return {"may_resume": not failures, "failures": failures}


# --------------------------------------------------------------------- authorization

ALLOWED_SCOPES = ("PHASE_MB_ONLY", "FULL_V2_3_PILOT")
AUTHORIZATION_REQUIRED_FIELDS = (
    "schema_version",
    "authorization_status",
    "allowed_scope",
    "repository_branch",
    "repository_head",
    "contract_version",
    "contract_sha256",
    "execution_implementation_bundle_sha256",
    "pilot_index_manifest_sha256",
    "accepted_stage_a_meta_sha256",
    "accepted_stage_b_meta_sha256",
    "allowed_output_root",
    "pilot_trained_token_ceiling",
)


def authorization_template() -> dict[str, Any]:
    """Always NOT_AUTHORIZED. This repository cannot publish an AUTHORIZED manifest."""
    return {
        "schema_version": AUTHORIZATION_SCHEMA,
        "authorization_status": "NOT_AUTHORIZED",
        "allowed_scope": None,
        "allowed_scope_values": list(ALLOWED_SCOPES),
        "repository_branch": None,
        "repository_head": None,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": None,
        "execution_implementation_bundle_sha256": None,
        "pilot_index_manifest_sha256": None,
        "accepted_stage_a_meta_sha256": None,
        "accepted_stage_b_meta_sha256": None,
        "allowed_output_root": None,
        "pilot_trained_token_ceiling": GLOBAL_PILOT_TOKEN_CEILING,
        "authorized_by": None,
        "authorized_at": None,
        "note": (
            "The owner publishes a separate AUTHORIZED manifest binding the exact reviewed "
            "HEAD after independent review. No tracked code change is required for that "
            "transition: validate_authorization() below consumes this same schema."
        ),
    }


def validate_authorization(
    manifest: Mapping[str, Any] | None, *, requested_scope: str, observed: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate an external authorization manifest against the observed runtime.

    Returns a verdict; the caller refuses on ``authorized=False``. This is the single gate a
    later AUTHORIZED manifest passes through, so no code edit is needed for that transition.
    """
    require(requested_scope in ALLOWED_SCOPES, f"unknown scope {requested_scope!r}")
    if manifest is None:
        return {"authorized": False, "failures": ["authorization_missing"]}
    failures: list[str] = []
    missing = [f for f in AUTHORIZATION_REQUIRED_FIELDS if f not in manifest]
    if missing:
        failures.append(f"missing_fields:{sorted(missing)}")
    if manifest.get("schema_version") != AUTHORIZATION_SCHEMA:
        failures.append("authorization_schema_mismatch")
    if manifest.get("authorization_status") != "AUTHORIZED":
        failures.append("authorization_status_not_authorized")
    if manifest.get("contract_version") != CONTRACT_VERSION:
        failures.append("contract_version_mismatch")
    scope = manifest.get("allowed_scope")
    if scope not in ALLOWED_SCOPES:
        failures.append("allowed_scope_invalid")
    elif scope == "PHASE_MB_ONLY" and requested_scope != "PHASE_MB_ONLY":
        failures.append("requested_scope_exceeds_authorization")
    for field, key in (
        ("repository_branch", "branch"),
        ("repository_head", "head"),
        ("contract_sha256", "contract_sha256"),
        ("execution_implementation_bundle_sha256", "execution_bundle_sha256"),
        ("pilot_index_manifest_sha256", "pilot_index_manifest_sha256"),
        ("accepted_stage_a_meta_sha256", "stage_a_meta_sha256"),
        ("accepted_stage_b_meta_sha256", "stage_b_meta_sha256"),
        ("allowed_output_root", "output_root"),
    ):
        if key in observed and manifest.get(field) != observed.get(key):
            failures.append(f"mismatch:{field}")
    ceiling = manifest.get("pilot_trained_token_ceiling")
    if not isinstance(ceiling, int) or ceiling > GLOBAL_PILOT_TOKEN_CEILING:
        failures.append("token_ceiling_invalid_or_above_contract")
    return {
        "authorized": not failures,
        "failures": failures,
        "allowed_scope": scope,
        "requested_scope": requested_scope,
    }


# --------------------------------------------------------------------- serialization


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def contract_document() -> dict[str, Any]:
    return {
        "schema_version": CONTRACT_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "supersedes": dict(SUPERSEDES),
        "retained_from_v2_2": list(RETAINED_FROM_V2_2),
        "not_reopened": list(NOT_REOPENED),
        "authorization_status": "NOT_AUTHORIZED",
        "owner_optimizer_verdict": OWNER_OPTIMIZER_VERDICT,
        "optimizer_family_comparison_required": OPTIMIZER_FAMILY_COMPARISON_REQUIRED,
        "ignored_local_authority_runtime_role": "NONE",
        "effective_batch": {
            "EFFECTIVE_BATCH_TOKENS": EFFECTIVE_BATCH_TOKENS,
            "SEQUENCES_PER_OPTIMIZER_UPDATE": SEQUENCES_PER_OPTIMIZER_UPDATE,
            "SEQ_LEN": SEQ_LEN,
            "FROZEN_GRAD_ACCUM_rule": "128 / FROZEN_MICRO_BSZ",
        },
        "optimizer": realized_muon_config(),
        "phase_mb": {
            "grid": [dict(c) for c in mb_candidate_grid()],
            "required_candidate_ids": list(MB_REQUIRED_CANDIDATE_IDS),
            "peak_lr": MB_PROBE_PEAK_LR,
            "warmup_updates": MB_PROBE_WARMUP_UPDATES,
            "updates": MB_PROBE_UPDATES,
            "measured_updates": [MB_MEASURED_FIRST_UPDATE, MB_PROBE_UPDATES],
            "model_init_seed": MB_MODEL_INIT_SEED,
            "vram_reserved_fraction_ceiling": MB_VRAM_RESERVED_FRACTION_CEILING,
            "tie_throughput_relative": MB_TIE_THROUGHPUT_RELATIVE,
            "tie_vram_mib": MB_TIE_VRAM_MIB,
        },
        "pilot_indices": {
            "seed": PILOT_INDEX_SEED,
            "numpy_version": REQUIRED_NUMPY_VERSION,
            "generator": "numpy.random.Generator(PCG64)",
            "stage_a_eval_count": STAGE_A_EVAL_COUNT,
            "stage_a_train_count": STAGE_A_TRAIN_COUNT,
            "stage_b_eval_count": STAGE_B_EVAL_COUNT,
            "draw_order": ["stage_a_eval", "stage_a_train", "stage_b_eval"],
            "universes_derived_from": "the accepted releases at runtime",
            "seed_semantics": {k: dict(v) for k, v in SEED_SEMANTICS.items()},
        },
        "phase_muon_lr": {
            "grid_seed1": list(LR_GRID_SEED1),
            "updates": LR_RUN_UPDATES,
            "warmup_updates": LR_WARMUP_UPDATES,
            "blocks_per_run": LR_BLOCKS_PER_RUN,
            "tokens_per_run": LR_TOKENS_PER_RUN,
            "baseline_window": list(LR_BASELINE_WINDOW),
            "guard_window": LR_GUARD_WINDOW,
            "guard_final_update_range": list(LR_GUARD_FINAL_UPDATE_RANGE),
            "guard_multiplier": LR_GUARD_MULTIPLIER,
            "score_formula": "(10*loss_A + 3*loss_B)/13",
            "tie_relative": LR_TIE_RELATIVE,
            "edge": {
                "2e-4": 1e-4,
                "4e-4": 6e-4,
                "3e-4": None,
                "requires_both_seeds_eligible": True,
                "second_expansion_permitted": False,
            },
            "search_axis": "main scheduled --lr only",
        },
        "production_schedule": {
            "FROZEN_WARMUP_STEPS": FROZEN_WARMUP_STEPS,
            "FROZEN_WARMUP_STEPS_AUTHORITY": FROZEN_WARMUP_STEPS_AUTHORITY,
            "OWNER_DECAY_INTENT_FRACTION_OF_TOTAL": OWNER_DECAY_INTENT_FRACTION_OF_TOTAL,
            "PLANNER_DECAY_FRACTION_INPUT": PLANNER_DECAY_FRACTION_INPUT,
            "PRODUCTION_MIN_LR_INTENT_RATIO": PRODUCTION_MIN_LR_INTENT_RATIO,
            "expected_schedule_total_steps": EXPECTED_SCHEDULE_TOTAL_STEPS,
            "expected_decay_updates": EXPECTED_DECAY_UPDATES,
            "expected_decay_interval": list(EXPECTED_DECAY_INTERVAL),
            "boundary_authority": "canonical planner output only",
        },
        "token_budget": {
            "PHASE_MB_TOKEN_CEILING": PHASE_MB_TOKEN_CEILING,
            "PHASE_MB_EXPECTED_MAX": PHASE_MB_EXPECTED_MAX,
            "PHASE_MUON_LR_TOKEN_CEILING": PHASE_MUON_LR_TOKEN_CEILING,
            "GLOBAL_PILOT_TOKEN_CEILING": GLOBAL_PILOT_TOKEN_CEILING,
            "checked_before_every_update": True,
            "persisted_after_every_update": True,
        },
        "runtime_gate": {
            "required_gpu_name_exact": REQUIRED_GPU_NAME_EXACT,
            "required_vram_mib_range": [REQUIRED_GPU_MIN_VRAM_MIB, REQUIRED_GPU_MAX_VRAM_MIB],
            "substring_check_insufficient": True,
            "required_numpy_version": REQUIRED_NUMPY_VERSION,
        },
        "checkpoint_isolation": {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in CHECKPOINT_ISOLATION.items()
        },
        "authorization": {
            "schema": AUTHORIZATION_SCHEMA,
            "allowed_scopes": list(ALLOWED_SCOPES),
            "required_fields": list(AUTHORIZATION_REQUIRED_FIELDS),
        },
    }


def contract_sha256() -> str:
    return hashlib.sha256(canonical_json_bytes(contract_document())).hexdigest()
