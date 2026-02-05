# pretrain/train_pretrain.py
# Minimal, robust GPT pretraining script for petitgpt.
# - Works when launched from either project root or pretrain/ directory
# - Supports grad accumulation, bf16/fp16 autocast, torch.compile
# - Saves clean checkpoints (including model config) and periodic text samples
# - Computes loss only over valid tokens (labels != -100)
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
# Make imports work no matter where the script is launched from.
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_pretrain import PackedBinDataset  # noqa: E402
from src.model import GPT, GPTConfig  # noqa: E402
from sample import generate_default_samples  # noqa: E402

# -----------------------------------------------------------------------------
# Performance toggles
# -----------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
except Exception:
    pass


def _resolve_dir(p: str) -> str:
    """Resolve a directory path relative to project root if needed."""
    path = Path(p)
    if path.exists():
        return str(path)
    alt = PROJECT_ROOT / p
    if alt.exists():
        return str(alt)
    # Also try one-level up (common when launching from pretrain/)
    alt2 = (PROJECT_ROOT / "pretrain" / p).resolve()
    if alt2.exists():
        return str(alt2)
    return str(path)


def masked_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Cross entropy over non-masked tokens.
    logits: [B, T, V]
    labels: [B, T] with -100 meaning "ignore"
    Returns a scalar (mean over valid tokens).
    """
    B, T, V = logits.shape
    logits_2d = logits.reshape(B * T, V)
    labels_1d = labels.reshape(B * T)
    return F.cross_entropy(logits_2d, labels_1d, ignore_index=-100, reduction="mean")


def save_ckpt(
    out_dir: Path,
    global_step: int,
    local_step: int,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    model_config: Dict,
    train_args: Dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_path = out_dir / "latest.pt"
    step_path = out_dir / f"step_{global_step:06d}.pt"

    # If model is torch.compile'd, unwrap to save clean keys (no _orig_mod.* prefix).
    model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model

    ckpt = {
        "model": model_to_save.state_dict(),
        "optim": optim.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "global_step": global_step,
        "local_step": local_step,
        "config": model_config,  # IMPORTANT: used by sampling / reload
        "train_args": train_args,
    }
    torch.save(ckpt, step_path)
    torch.save(ckpt, latest_path)


def load_ckpt(
    resume_path: Path,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    resume_full: bool,
    map_location: str = "cpu",
) -> Tuple[int, int]:
    ckpt = torch.load(resume_path, map_location=map_location)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # Handle old compiled checkpoints that may have "_orig_mod." prefixes.
    if isinstance(state, dict) and any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in state.items()}

    model.load_state_dict(state, strict=True)

    global_step = int(ckpt.get("global_step", 0)) if isinstance(ckpt, dict) else 0
    local_step = int(ckpt.get("local_step", 0)) if isinstance(ckpt, dict) else 0

    if resume_full and isinstance(ckpt, dict):
        if "optim" in ckpt and ckpt["optim"] is not None:
            optim.load_state_dict(ckpt["optim"])
        if scaler is not None and "scaler" in ckpt and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])

    return global_step, local_step


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--train_dir", type=str, required=True)
    ap.add_argument("--val_dir", type=str, required=True)

    # Output
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--tb_dir", type=str, required=True)  # kept for compatibility; not required to exist
    ap.add_argument("--samples_dir", type=str, required=True)
    ap.add_argument("--tokenizer_path", type=str, required=True)

    # Model
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--n_heads", type=int, default=12)
    ap.add_argument("--d_ff", type=int, default=3072)
    ap.add_argument("--dropout", type=float, default=0.0)

    # Special tokens (must match tokenizer)
    ap.add_argument("--bos_id", type=int, default=2)
    ap.add_argument("--eos_id", type=int, default=3)

    # Train
    ap.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--micro_bsz", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--max_steps", type=int, default=80000)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--save_every", type=int, default=1000)

    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)

    # Sampling during training
    ap.add_argument("--add_bos", action="store_true", help="Add BOS to prompts in sampling only")
    ap.add_argument("--sample_temperature", type=float, default=0.7)
    ap.add_argument("--sample_top_p", type=float, default=0.9)
    ap.add_argument("--sample_top_k", type=int, default=0)
    ap.add_argument("--sample_max_new_tokens", type=int, default=128)

    # Resume
    ap.add_argument("--resume_path", type=str, default="")
    ap.add_argument("--resume_full", action="store_true", help="Also resume optimizer/scaler states")

    # torch.compile
    ap.add_argument("--compile", action="store_true")

    return ap.parse_args()


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_schedule(step: int, warmup_steps: int, base_lr: float) -> float:
    if warmup_steps <= 0:
        return base_lr
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


@torch.no_grad()
def eval_one_epoch(
    model: torch.nn.Module,
    dl: DataLoader,
    device: torch.device,
    precision: str,
    max_batches: int = 50,
) -> float:
    model.eval()
    ac_dtype = torch.bfloat16 if precision == "bf16" else (torch.float16 if precision == "fp16" else None)

    losses = []
    it = iter(dl)
    for _ in range(max_batches):
        try:
            input_ids, labels = next(it)
        except StopIteration:
            break
        input_ids = input_ids.to(device, dtype=torch.long, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)

        if ac_dtype is not None:
            with torch.autocast("cuda", dtype=ac_dtype):
                logits = model(input_ids)
                loss = masked_ce_loss(logits, labels)
        else:
            logits = model(input_ids)
            loss = masked_ce_loss(logits, labels)

        losses.append(float(loss.item()))

    model.train()
    return float(sum(losses) / max(1, len(losses)))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Resolve paths robustly
    train_dir = Path(_resolve_dir(args.train_dir))
    val_dir = Path(_resolve_dir(args.val_dir))
    out_dir = Path(_resolve_dir(args.out_dir))
    samples_dir = Path(_resolve_dir(args.samples_dir))
    tok_path = Path(_resolve_dir(args.tokenizer_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] PROJECT_ROOT={PROJECT_ROOT}")
    print(f"[env] device={device}, precision={args.precision}")
    print(f"[env] OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '(unset)')}")
    print(f"[env] MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS', '(unset)')}")

    cfg = GPTConfig(
        vocab_size=args.vocab_size,
        n_layers=args.layers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        tie_embeddings=True,
    )

    model = GPT(cfg).to(device)

    use_fp16 = args.precision == "fp16"
    ac_dtype = torch.bfloat16 if args.precision == "bf16" else (torch.float16 if use_fp16 else None)
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    # Data
    train_ds = PackedBinDataset(str(train_dir), seq_len=args.seq_len)
    val_ds = PackedBinDataset(str(val_dir), seq_len=args.seq_len)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.micro_bsz,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.micro_bsz,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Resume (optional)
    global_step = 0
    local_step = 0
    if args.resume_path:
        resume_path = Path(_resolve_dir(args.resume_path))
        global_step, local_step = load_ckpt(
            resume_path=resume_path,
            model=model,
            optim=optim,
            scaler=scaler if use_fp16 else None,
            resume_full=args.resume_full,
        )
        print(f"[resume] loaded {resume_path} (global_step={global_step}, local_step={local_step})")

    # compile (after loading weights)
    if args.compile:
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
            print("[compile] torch.compile enabled")
        except Exception as e:
            print(f"[compile] failed, continuing without compile: {e}")

    model.train()
    data_iter = iter(train_dl)

    tokens_seen = 0
    window_tokens = 0
    t0 = time.time()

    while local_step < args.max_steps:
        # LR warmup
        lr = lr_schedule(global_step, args.warmup_steps, args.lr)
        for pg in optim.param_groups:
            pg["lr"] = lr

        optim.zero_grad(set_to_none=True)
        accum_loss = 0.0
        accum_loss_raw = 0.0

        for _ in range(args.grad_accum):
            try:
                input_ids, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(train_dl)
                input_ids, labels = next(data_iter)

            input_ids = input_ids.to(device, dtype=torch.long, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)

            ntok = int((labels != -100).sum().item())
            tokens_seen += ntok
            window_tokens += ntok

            if ac_dtype is not None:
                with torch.autocast("cuda", dtype=ac_dtype):
                    logits = model(input_ids)
                    loss_raw = masked_ce_loss(logits, labels)
                    loss = loss_raw / args.grad_accum
            else:
                logits = model(input_ids)
                loss_raw = masked_ce_loss(logits, labels)
                loss = loss_raw / args.grad_accum

            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += float(loss.detach().item())
            accum_loss_raw += float(loss_raw.detach().item())

        # Clip and step
        if args.grad_clip and args.grad_clip > 0:
            if use_fp16:
                scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))

        if use_fp16:
            scaler.step(optim)
            scaler.update()
        else:
            optim.step()

        global_step += 1
        local_step += 1

        if global_step % args.log_every == 0:
            dt = time.time() - t0
            tok_s = window_tokens / max(dt, 1e-6)
            print(
                f"[train] step={global_step} loss={accum_loss:.4f} loss_raw={accum_loss_raw/args.grad_accum:.4f} "
                f"lr={lr:.2e} tok/s={tok_s:.0f}"
            )
            t0 = time.time()
            window_tokens = 0

        if global_step % args.eval_every == 0:
            val_loss = eval_one_epoch(model, val_dl, device, args.precision, max_batches=50)
            print(f"[eval] step={global_step} val_loss={val_loss:.4f}")

            # Write samples
            samples_dir.mkdir(parents=True, exist_ok=True)
            out_path = samples_dir / f"step_{global_step:06d}.txt"
            try:
                generate_default_samples(
                    model=model,
                    tokenizer_path=str(tok_path),
                    device=device,
                    max_seq_len=args.seq_len,
                    precision=args.precision,
                    out_path=out_path,
                    temperature=args.sample_temperature,
                    top_p=args.sample_top_p,
                    top_k=args.sample_top_k,
                    max_new_tokens=args.sample_max_new_tokens,
                    eos_id=args.eos_id,
                    add_bos=args.add_bos,
                    bos_id=args.bos_id,
                    min_new_tokens=0,
                    greedy=False,
                    debug=True,
                )
                print(f"[sample] wrote {out_path}")
            except Exception as e:
                print(f"[sample] failed: {e}")

        if global_step % args.save_every == 0:
            save_ckpt(
                out_dir=out_dir,
                global_step=global_step,
                local_step=local_step,
                model=model,
                optim=optim,
                scaler=scaler if use_fp16 else None,
                model_config=asdict(cfg),
                train_args=vars(args),
            )
            print(f"[ckpt] saved step {global_step} to {out_dir}")

    # Final save
    save_ckpt(
        out_dir=out_dir,
        global_step=global_step,
        local_step=local_step,
        model=model,
        optim=optim,
        scaler=scaler if use_fp16 else None,
        model_config=asdict(cfg),
        train_args=vars(args),
    )
    print(f"[done] saved final checkpoint to {out_dir}")


if __name__ == "__main__":
    main()
