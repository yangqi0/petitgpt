#!/usr/bin/env python3

"""
Direct Preference Optimization (DPO) post-training for petitgpt.

Starts from an SFT or distill checkpoint (schema: keys "model" + "cfg"/"config"),
builds a frozen reference copy of that same checkpoint, and optimizes the policy
on preference pairs with the standard DPO loss (Rafailov et al., 2023):

    L = -log sigmoid( beta * [ (logpi_w - logref_w) - (logpi_l - logref_l) ] )

where w/l are the chosen/rejected completions and logpi_*/logref_* are the
summed log-probs of the completion tokens under the policy / reference model.

Expected data format (JSONL, one example per line), matching the token-level
chat template in src/chat_template.py (shared with SFT/distill/GRPO):

    {"messages": [{"role": "system", "content": "..."},
                   {"role": "user", "content": "..."}],
     "chosen": "the preferred assistant response",
     "rejected": "the dispreferred assistant response"}

`messages` is the shared prompt context (system/user turns, and any earlier
assistant turns for multi-turn prompts); `chosen`/`rejected` are plain
assistant-completion strings appended after the `<|assistant|>` cue, each
ending with a supervised EOS (so DPO logps include the stop decision).

Example:
    python dpo/dpo.py \\
      --train_jsonl datasets/dpo/train.jsonl --val_jsonl datasets/dpo/val.jsonl \\
      --out_dir outputs/dpo_run --tokenizer_path tokenizer/tokenizer.json \\
      --init_ckpt outputs/sft_v6_general_code/step_003500.pt \\
      --seq_len 1024 --micro_bsz 2 --grad_accum 8 --lr 5e-6 --beta 0.1
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import copy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
from tokenizers import Tokenizer
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.chat_template import (  # noqa: E402
    IGNORE_INDEX,
    decode_completion,
    encode_completion,
    encode_prompt,
    load_chat_tokenizer,
    pad_or_truncate,
    truncate_chat_sequence,
)
from src.model import GPT, GPTConfig, gpt_config_from_checkpoint_dict  # noqa: E402
from src.optim import build_optimizer  # noqa: E402
from src.posttrain_preflight import (  # noqa: E402
    require_preflight_passed,
    run_jsonl_preflight,
)
from src.posttrain_resume import (  # noqa: E402
    DeterministicEpochBatchSampler,
    build_resume_contract_base,
    capture_rng_state,
    make_loader_generator,
    require_resume_step,
    restore_rng_state,
    restore_training_state,
    resume_contract_for_step,
    validate_resume_contract,
    validate_training_controls,
)
from src.special_tokens import EOS_ID, PAD_ID  # noqa: E402
from src.tracking import Tracker  # noqa: E402

FIXED_PROMPTS = [
    "[Code] Write a Python function running_sum(nums) that returns cumulative sums.",
    "[Code] Write a Python function lowercase_keys(d) that returns a new dictionary with lowercase string keys.",
    "[General] Write a short polite email asking for an update on a job application after an interview.",
    "[General] Rewrite this to be more concise: 'I am writing this email in order to ask whether it would be possible to move our meeting to Friday afternoon.'",
]


# -------------------------
# Dataset: jsonl offsets
# -------------------------
class JsonlOffsetsDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        self.offsets: list[int] = []
        with open(path, "rb") as f:
            off = 0
            for line in f:
                self.offsets.append(off)
                off += len(line)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        off = self.offsets[idx]
        with open(self.path, "rb") as f:
            f.seek(off)
            line = f.readline().decode("utf-8")
        return json.loads(line)


# -------------------------
# Preference-pair example building (token-level template, src/chat_template.py)
# -------------------------
def preflight_dpo_record(
    _split: str,
    example: dict[str, Any],
    *,
    tok: Tokenizer,
    seq_len: int,
    default_system: str,
) -> dict[str, int]:
    """Validate both preference branches with the production token contract."""
    messages = example.get("messages")
    if not isinstance(messages, list):
        raise ValueError("missing messages list")

    metrics = {
        "encoded_tokens": 0,
        "retained_tokens": 0,
        "supervised_tokens": 0,
        "truncated_branches": 0,
    }
    for field in ("chosen", "rejected"):
        completion = example.get(field)
        if not isinstance(completion, str) or not completion.strip():
            raise ValueError(f"{field} completion must be non-empty and non-whitespace")
        ids, labels = encode_completion(tok, messages, completion, default_system)
        kept_ids, kept_labels = truncate_chat_sequence(ids, labels, max_len=seq_len)
        assert kept_labels is not None
        supervised = sum(label != IGNORE_INDEX for label in kept_labels)
        if supervised < 2:
            raise ValueError(
                f"{field} branch must retain assistant content plus supervised EOS"
            )
        metrics["encoded_tokens"] += len(ids)
        metrics["retained_tokens"] += len(kept_ids)
        metrics["supervised_tokens"] += supervised
        metrics["truncated_branches"] += int(len(kept_ids) != len(ids))
    return metrics


def build_completion_example(
    messages: list[dict[str, str]],
    completion: str,
    tok: Tokenizer,
    seq_len: int,
    pad_id: int,
    default_system: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode (prompt context + completion) and return (input_ids, labels), with
    labels=-100 everywhere except the completion tokens + their SUPERVISED EOS
    (so DPO logps include the stop decision). Matches SFT's encoding exactly."""
    ids, labels = encode_completion(tok, messages, completion, default_system)
    ids, labels = pad_or_truncate(ids, labels, seq_len, pad_id)
    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def collate_fn_builder(
    tok: Tokenizer,
    seq_len: int,
    pad_id: int,
    default_system: str,
    debug_first_batch: bool,
):
    printed = {"done": False}

    def collate(batch: list[dict[str, Any]]):
        xs_c, ys_c, xs_r, ys_r = [], [], [], []
        for ex in batch:
            messages = ex.get("messages") or []
            chosen = ex.get("chosen", "")
            rejected = ex.get("rejected", "")
            if not messages:
                raise ValueError("missing messages")
            if not chosen or not rejected:
                raise ValueError("missing chosen/rejected completion")

            xc, yc = build_completion_example(
                messages, chosen, tok, seq_len, pad_id, default_system
            )
            xr, yr = build_completion_example(
                messages, rejected, tok, seq_len, pad_id, default_system
            )
            xs_c.append(xc)
            ys_c.append(yc)
            xs_r.append(xr)
            ys_r.append(yr)

        batch_out = {
            "input_ids_chosen": torch.stack(xs_c, dim=0),
            "labels_chosen": torch.stack(ys_c, dim=0),
            "input_ids_rejected": torch.stack(xs_r, dim=0),
            "labels_rejected": torch.stack(ys_r, dim=0),
        }

        if debug_first_batch and not printed["done"]:
            printed["done"] = True
            lc0 = batch_out["labels_chosen"][0]
            lr0 = batch_out["labels_rejected"][0]
            print(f"[dbg] chosen supervised tokens(sample0): {int((lc0 != -100).sum().item())}")
            print(f"[dbg] rejected supervised tokens(sample0): {int((lr0 != -100).sum().item())}")
            idx = (lc0 != -100).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() > 0:
                dec = tok.decode(batch_out["input_ids_chosen"][0, idx].tolist())
                print(f"[dbg] decoded chosen span(first 300 chars): {dec[:300]}")

        return batch_out

    return collate


# -------------------------
# DPO loss
# -------------------------
def sequence_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum of log-probs of the supervised (label != -100) next-token targets,
    per example. logits: [B,T,V], labels: [B,T]."""
    B, T, V = logits.size()
    logits2 = logits[:, :-1, :].float()
    labels2 = labels[:, 1:]
    # per-token NLL; ignore_index positions contribute exactly 0 to the sum
    nll = F.cross_entropy(
        logits2.reshape(-1, V),
        labels2.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(B, T - 1)
    return -nll.sum(dim=1)


def get_batch_logps(
    model: torch.nn.Module,
    input_ids_chosen: torch.Tensor,
    labels_chosen: torch.Tensor,
    input_ids_rejected: torch.Tensor,
    labels_rejected: torch.Tensor,
    autocast_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single fused forward pass over chosen+rejected (concatenated on the
    batch dim) for one model, returning (chosen_logps, rejected_logps)."""
    b = input_ids_chosen.size(0)
    input_ids = torch.cat([input_ids_chosen, input_ids_rejected], dim=0)
    labels = torch.cat([labels_chosen, labels_rejected], dim=0)
    with torch.autocast(
        device_type="cuda", dtype=autocast_dtype, enabled=(autocast_dtype is not None)
    ):
        logits = model(input_ids)
    logps = sequence_logps(logits, labels)
    return logps[:b], logps[b:]


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-example DPO loss + implicit rewards. Returns (losses[B], chosen_rewards[B], rejected_rewards[B])."""
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = (beta * (policy_chosen_logps - ref_chosen_logps)).detach()
    rejected_rewards = (beta * (policy_rejected_logps - ref_rejected_logps)).detach()
    return losses, chosen_rewards, rejected_rewards


# -------------------------
# Checkpoints
# -------------------------
def save_checkpoint_atomic(path: str, obj: dict[str, Any]) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_ckpt(path: str) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def build_model_from_ckpt(ckpt: dict[str, Any], vocab_size: int, seq_len: int, device: str) -> tuple[GPT, GPTConfig]:
    cfg_dict = ckpt.get("cfg") or ckpt.get("config")
    if not isinstance(cfg_dict, dict):
        raise RuntimeError("checkpoint missing 'cfg'/'config' dict")
    cfg_dict = dict(cfg_dict)
    cfg_dict["vocab_size"] = vocab_size
    cfg_dict["max_seq_len"] = seq_len
    cfg = gpt_config_from_checkpoint_dict(cfg_dict)
    model = GPT(cfg).to(device)

    sd = ckpt.get("model")
    if sd is None:
        raise RuntimeError("checkpoint missing 'model'")
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k[len("_orig_mod.") :]: v for k, v in sd.items()}

    model.load_state_dict(sd, strict=True)
    return model, cfg


# -------------------------
# Sampling (qualitative check; policy only, no reference comparison)
# -------------------------
@torch.no_grad()
def sample_from_prompt(
    model: torch.nn.Module,
    tok: Tokenizer,
    prompt_ids: list[int],
    device: str,
    seq_len: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """Generate from an already-encoded prompt (ending in the <|assistant|>
    cue) and decode ONLY the generated tokens."""
    model.eval()
    prompt_ids, _ = truncate_chat_sequence(
        prompt_ids, labels=None, max_len=seq_len - 1
    )

    ids = torch.tensor(prompt_ids, device=device, dtype=torch.long)[None, :]
    prompt_len = ids.size(1)
    generation_steps = min(max_new_tokens, seq_len - prompt_len)
    for _ in range(generation_steps):
        logits = model(ids)[0, -1, :].float()

        if temperature <= 0:
            nxt = int(torch.argmax(logits).item())
        else:
            logits = logits / temperature
            if top_p and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cum = torch.cumsum(probs, dim=-1)
                mask_sorted = cum > top_p
                mask_sorted[0] = False
                mask = torch.zeros_like(mask_sorted)
                mask.scatter_(0, sorted_idx, mask_sorted)
                logits = torch.where(mask, torch.full_like(logits, -float("inf")), logits)
            probs = torch.softmax(logits, dim=-1)
            nxt = int(torch.multinomial(probs, num_samples=1).item())

        ids = torch.cat([ids, torch.tensor([[nxt]], device=device, dtype=torch.long)], dim=1)
        if nxt == EOS_ID:
            break

    out = decode_completion(tok, ids[0, prompt_len:].tolist())
    model.train()
    return out


def emit_samples(
    policy: torch.nn.Module,
    tok: Tokenizer,
    samples_dir: str,
    step_tag: str,
    device: str,
    seq_len: int,
    default_system: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> None:
    Path(samples_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(samples_dir, f"{step_tag}.txt")
    lines: list[str] = [f"step={step_tag}\n", "=" * 80 + "\n"]
    for i, q in enumerate(FIXED_PROMPTS, start=1):
        prompt_ids = encode_prompt(
            tok, [{"role": "user", "content": q}], default_system, mode="last_user"
        )
        ans = sample_from_prompt(policy, tok, prompt_ids, device, seq_len, max_new_tokens, temperature, top_p)
        lines.append(f"[Q{i}] {q}\n[A{i}] {ans}\n" + "-" * 80 + "\n")
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"[sample] wrote {out_path}")


# -------------------------
# Eval
# -------------------------
@torch.no_grad()
def evaluate(
    policy: torch.nn.Module,
    reference: torch.nn.Module,
    val_loader: DataLoader,
    device: str,
    autocast_dtype: torch.dtype | None,
    beta: float,
    max_batches: int,
) -> dict[str, float]:
    policy.eval()
    losses, margins, accs = [], [], []
    for j, batch in enumerate(val_loader):
        if j >= max_batches:
            break
        in_c = batch["input_ids_chosen"].to(device, non_blocking=True)
        lb_c = batch["labels_chosen"].to(device, non_blocking=True)
        in_r = batch["input_ids_rejected"].to(device, non_blocking=True)
        lb_r = batch["labels_rejected"].to(device, non_blocking=True)

        pc, pr = get_batch_logps(policy, in_c, lb_c, in_r, lb_r, autocast_dtype)
        rc, rr = get_batch_logps(reference, in_c, lb_c, in_r, lb_r, autocast_dtype)
        loss_vec, chosen_rewards, rejected_rewards = dpo_loss(pc, pr, rc, rr, beta)

        losses.append(float(loss_vec.mean().item()))
        margins.append(float((chosen_rewards - rejected_rewards).mean().item()))
        accs.append(float((chosen_rewards > rejected_rewards).float().mean().item()))

    policy.train()
    n = max(1, len(losses))
    return {
        "val_loss": sum(losses) / n,
        "val_reward_margin": sum(margins) / n,
        "val_reward_acc": sum(accs) / n,
    }


def validate_dpo_args(args: argparse.Namespace) -> None:
    validate_training_controls(
        args,
        positive_fields=(
            "micro_bsz",
            "grad_accum",
            "eval_batches",
            "sample_max_new_tokens",
        ),
        nonnegative_fields=(
            "num_workers",
            "eval_every",
            "save_every",
            "sample_every",
        ),
    )
    if args.seq_len <= 1:
        raise ValueError("--seq_len must be greater than 1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tokenizer_path", required=True)

    ap.add_argument("--init_ckpt", required=True, help="SFT or distill checkpoint to init policy+reference from")
    ap.add_argument("--ref_ckpt", default="", help="Optional separate reference checkpoint (default: deep-copy of the initialized policy)")
    ap.add_argument(
        "--resume",
        default="",
        help="Exact continuation from this run's own DPO checkpoint. Requires matching "
        "arguments, input bytes, runtime, policy/optimizer/scaler, all RNG streams, loop "
        "state, and deterministic data cursor. --init_ckpt must still point at the original "
        "starting checkpoint; use --init_ckpt without --resume for a weights-only new run. "
        "The frozen reference is rebuilt from it, never from the resumed policy.",
    )

    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--micro_bsz", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)

    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--optimizer", choices=["muon", "adamw"], default="muon",
                    help="muon: Muon on hidden matrices + AdamW on embeddings/norms (default). adamw: AdamW everywhere.")
    ap.add_argument("--muon_lr", type=float, default=0.0,
                    help="LR for Muon matrix groups (<=0: reuse --lr; Muon update RMS is matched to AdamW's).")
    ap.add_argument("--muon_momentum", type=float, default=0.95)
    ap.add_argument("--warmup_steps", type=int, default=150)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--grad_clip", type=float, default=1.0, help="0 disables grad clipping")
    ap.add_argument("--beta", type=float, default=0.1, help="DPO temperature; higher = closer to reference")

    ap.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="bf16")
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_batches", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=200)

    ap.add_argument("--default_system", default="You are a helpful assistant.")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--debug_first_batch", action="store_true")

    ap.add_argument("--sample_every", type=int, default=200)
    ap.add_argument("--samples_dir", type=str, default="")
    ap.add_argument("--sample_max_new_tokens", type=int, default=192)
    ap.add_argument("--sample_temperature", type=float, default=0.7)
    ap.add_argument("--sample_top_p", type=float, default=0.9)

    args = ap.parse_args()

    validate_dpo_args(args)
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    tok = load_chat_tokenizer(args.tokenizer_path)
    tracker = Tracker(args.out_dir)
    tracker.log_run_start(vars(args), args.tokenizer_path)
    vocab_size = tok.get_vocab_size()
    pad_id = PAD_ID

    report = run_jsonl_preflight(
        stage="dpo",
        datasets={"train": args.train_jsonl, "val": args.val_jsonl},
        validate_record=lambda split, record: preflight_dpo_record(
            split,
            record,
            tok=tok,
            seq_len=args.seq_len,
            default_system=args.default_system,
        ),
        report_path=os.path.join(args.out_dir, "posttrain_preflight.json"),
        metadata={"seq_len": args.seq_len, "default_system": args.default_system},
    )
    tracker.log(
        "preflight",
        0,
        status=report["status"],
        records=report["total_records"],
        valid=report["total_valid"],
        rejected=report["total_rejected"],
    )
    require_preflight_passed(report)

    resume_checkpoint: Mapping[str, Any] | None = None
    start_step = 0
    if args.resume:
        loaded_resume = load_ckpt(args.resume)
        if not isinstance(loaded_resume, Mapping):
            raise RuntimeError("--resume checkpoint must be a mapping")
        resume_checkpoint = loaded_resume
        start_step = require_resume_step(
            resume_checkpoint,
            stage="dpo",
            weights_only_hint="--init_ckpt",
        )

    resume_inputs: dict[str, str] = {
        "tokenizer": args.tokenizer_path,
        "train_jsonl": args.train_jsonl,
        "val_jsonl": args.val_jsonl,
        "init_ckpt": args.init_ckpt,
    }
    if args.ref_ckpt:
        resume_inputs["ref_ckpt"] = args.ref_ckpt
    resume_contract_base = build_resume_contract_base(
        stage="dpo",
        args=vars(args),
        input_paths=resume_inputs,
        dataset_size=int(report["splits"]["train"]["records"]),
        batch_size=args.micro_bsz,
        batches_per_step=args.grad_accum,
        seed=args.seed,
    )
    if resume_checkpoint is not None:
        validate_resume_contract(
            resume_checkpoint,
            resume_contract_for_step(resume_contract_base, start_step),
            weights_only_hint="--init_ckpt",
        )

    print(f"[*] loading policy init from: {args.init_ckpt}")
    init_ckpt = load_ckpt(args.init_ckpt)
    policy, cfg = build_model_from_ckpt(init_ckpt, vocab_size, args.seq_len, device)

    if args.ref_ckpt:
        print(f"[*] loading reference from: {args.ref_ckpt}")
        ref_ckpt = load_ckpt(args.ref_ckpt)
        reference, _ = build_model_from_ckpt(ref_ckpt, vocab_size, args.seq_len, device)
    else:
        print("[*] reference = frozen deep-copy of the initialized policy")
        reference = copy.deepcopy(policy)

    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    optimizer = build_optimizer(
        policy,
        name=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        muon_lr=args.muon_lr,
        muon_momentum=args.muon_momentum,
    )

    use_fp16 = args.precision == "fp16" and device == "cuda"
    use_bf16 = args.precision == "bf16" and device == "cuda"
    autocast_dtype = torch.float16 if use_fp16 else (torch.bfloat16 if use_bf16 else None)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    if resume_checkpoint is not None:
        restore_training_state(
            resume_checkpoint,
            model=policy,
            optimizer=optimizer,
            scaler=scaler,
            use_fp16=use_fp16,
        )
        print(f"[*] resumed policy from: {args.resume} at step={start_step} "
              f"(frozen reference stays the one built from --init_ckpt/--ref_ckpt)")

    train_ds = JsonlOffsetsDataset(args.train_jsonl)
    val_ds = JsonlOffsetsDataset(args.val_jsonl)
    print(f"[*] dataset: train_lines={len(train_ds)} val_lines={len(val_ds)}")
    print(
        f"[*] effective_pairs/step = micro_bsz({args.micro_bsz}) * grad_accum({args.grad_accum})"
        f" = {args.micro_bsz * args.grad_accum}"
    )

    train_batch_sampler = DeterministicEpochBatchSampler(
        len(train_ds),
        args.micro_bsz,
        seed=args.seed,
        start_batch=start_step * args.grad_accum,
        drop_last=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_batch_sampler,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn_builder(tok, args.seq_len, pad_id, args.default_system, args.debug_first_batch),
        generator=make_loader_generator(args.seed, 1),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.micro_bsz,
        shuffle=False,
        num_workers=max(0, args.num_workers // 2),
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn_builder(tok, args.seq_len, pad_id, args.default_system, False),
        drop_last=False,
        generator=make_loader_generator(args.seed, 2),
    )

    def get_lr(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        t = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        t = min(max(t, 0.0), 1.0)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * t))

    samples_dir = args.samples_dir or os.path.join(args.out_dir, "samples")

    policy.train()
    t0 = time.time()
    running_loss = 0.0
    running_margin = 0.0
    running_acc = 0.0
    step = start_step
    if resume_checkpoint is not None:
        loop_state = resume_checkpoint.get("loop_state")
        if not isinstance(loop_state, Mapping):
            raise RuntimeError("DPO exact resume checkpoint lacks loop_state")
        try:
            running_loss = float(loop_state["running_loss"])
            running_margin = float(loop_state["running_margin"])
            running_acc = float(loop_state["running_acc"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("DPO exact resume has invalid loop_state") from exc
        if not all(
            math.isfinite(value)
            for value in (running_loss, running_margin, running_acc)
        ):
            raise RuntimeError("DPO exact resume loop_state must be finite")
        restore_rng_state(resume_checkpoint["rng_state"])

    last_saved_step: int | None = None

    def save_training_checkpoint(checkpoint_step: int) -> None:
        nonlocal last_saved_step
        if last_saved_step == checkpoint_step:
            return
        ckpt = {
            "step": checkpoint_step,
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if use_fp16 else None,
            "cfg": asdict(cfg),
            "args": vars(args),
            "kind": "dpo",
            "resume_contract": resume_contract_for_step(
                resume_contract_base, checkpoint_step
            ),
            "rng_state": capture_rng_state(),
            "loop_state": {
                "running_loss": running_loss,
                "running_margin": running_margin,
                "running_acc": running_acc,
            },
        }
        retained_path = os.path.join(
            args.out_dir, f"step_{checkpoint_step:06d}.pt"
        )
        save_checkpoint_atomic(retained_path, ckpt)
        save_checkpoint_atomic(os.path.join(args.out_dir, "latest.pt"), ckpt)
        last_saved_step = checkpoint_step
        print(f"[ckpt] saved {retained_path}")

    train_iter = iter(train_loader)

    while step < args.max_steps:
        optimizer.zero_grad(set_to_none=True)
        micro_loss = 0.0
        micro_margin = 0.0
        micro_acc = 0.0

        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr * pg.get("lr_ratio", 1.0)

        for _ in range(args.grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            in_c = batch["input_ids_chosen"].to(device, non_blocking=True)
            lb_c = batch["labels_chosen"].to(device, non_blocking=True)
            in_r = batch["input_ids_rejected"].to(device, non_blocking=True)
            lb_r = batch["labels_rejected"].to(device, non_blocking=True)

            pc, pr = get_batch_logps(policy, in_c, lb_c, in_r, lb_r, autocast_dtype)
            with torch.no_grad():
                rc, rr = get_batch_logps(reference, in_c, lb_c, in_r, lb_r, autocast_dtype)

            loss_vec, chosen_rewards, rejected_rewards = dpo_loss(pc, pr, rc, rr, args.beta)
            loss = loss_vec.mean() / args.grad_accum

            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            micro_loss += float(loss.item())
            micro_margin += float((chosen_rewards - rejected_rewards).mean().item()) / args.grad_accum
            micro_acc += float((chosen_rewards > rejected_rewards).float().mean().item()) / args.grad_accum

        if args.grad_clip > 0:
            if use_fp16:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)

        if use_fp16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        step += 1
        running_loss += micro_loss
        running_margin += micro_margin
        running_acc += micro_acc

        if step % 20 == 0:
            dt = time.time() - t0
            print(
                f"[train] step={step} loss={running_loss / 20:.4f} "
                f"reward_margin={running_margin / 20:.4f} reward_acc={running_acc / 20:.4f} "
                f"lr={lr:.2e} dt={dt:.1f}s"
            )
            tracker.log(
                "train",
                step,
                loss=running_loss / 20,
                reward_margin=running_margin / 20,
                reward_acc=running_acc / 20,
                lr=lr,
            )
            running_loss = 0.0
            running_margin = 0.0
            running_acc = 0.0
            t0 = time.time()

        if args.eval_every > 0 and step % args.eval_every == 0:
            metrics = evaluate(policy, reference, val_loader, device, autocast_dtype, args.beta, args.eval_batches)
            print(
                f"[eval] step={step} val_loss={metrics['val_loss']:.4f} "
                f"val_reward_margin={metrics['val_reward_margin']:.4f} "
                f"val_reward_acc={metrics['val_reward_acc']:.4f}"
            )
            tracker.log("val", step, **{k: float(v) for k, v in metrics.items()})
            tracker.render()

        if args.sample_every and args.sample_every > 0 and step % args.sample_every == 0:
            emit_samples(
                policy,
                tok,
                samples_dir,
                f"step_{step:06d}",
                device,
                args.seq_len,
                args.default_system,
                args.sample_max_new_tokens,
                args.sample_temperature,
                args.sample_top_p,
            )

        if args.save_every > 0 and step % args.save_every == 0:
            save_training_checkpoint(step)

    save_training_checkpoint(step)
    tracker.render()
    print("[done]")


if __name__ == "__main__":
    main()
