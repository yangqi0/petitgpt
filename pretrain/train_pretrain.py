# pretrain/train_pretrain.py
# Minimal, robust GPT pretraining script for petitgpt.
# - MiniMind-style token-level loss_mask (explicit mask and weighted reduction)
# - BOS is masked from loss; EOS can be down-weighted to prevent early-EOS collapse
# - grad accumulation, bf16/fp16 autocast, torch.compile, clean checkpoints
# - periodic evaluation + sample generation + milestone checkpoints
from __future__ import annotations

import argparse
import json
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


def _resolve_path(p: str) -> str:
    """Resolve a path relative to project root if needed."""
    path = Path(p)
    if path.exists():
        return str(path)
    alt = PROJECT_ROOT / p
    if alt.exists():
        return str(alt)
    return str(path)


def lr_schedule(step: int, warmup_steps: int, base_lr: float) -> float:
    """Simple linear warmup then constant LR (stable baseline)."""
    if warmup_steps <= 0:
        return base_lr
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _autocast_dtype(precision: str) -> Optional[torch.dtype]:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return None


def masked_weighted_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    eos_id: int,
    eos_weight: float,
) -> torch.Tensor:
    """
    MiniMind-style loss:
      - per-token CE (reduction='none')
      - multiply by loss_mask
      - optionally down-weight EOS targets to prevent early-EOS collapse
      - normalize by sum of weights (NOT by B*T)
    """
    B, T, V = logits.shape
    logits_2d = logits.reshape(B * T, V)
    labels_1d = labels.reshape(B * T)
    mask_1d = loss_mask.reshape(B * T)

    # Per-token CE, ignore_index is NOT used here (mask controls everything).
    per_tok = F.cross_entropy(logits_2d, labels_1d, reduction="none")  # [B*T]

    weights = mask_1d
    if eos_weight != 1.0:
        eos_m = (labels_1d == int(eos_id)).float()
        # Scale EOS weights (mask already includes BOS removal etc.)
        weights = weights * (1.0 + eos_m * (float(eos_weight) - 1.0))

    denom = weights.sum().clamp_min(1.0)
    return (per_tok * weights).sum() / denom


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

    # Unwrap torch.compile for clean keys.
    model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model

    ckpt = {
        "model": model_to_save.state_dict(),
        "optim": optim.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "global_step": global_step,
        "local_step": local_step,
        "config": model_config,
        "train_args": train_args,
    }

    # Always update latest
    torch.save(ckpt, out_dir / "latest.pt")

    # Also write milestone-style checkpoint by step (useful for comparing samples)
    torch.save(ckpt, out_dir / f"step_{global_step:06d}.pt")


def load_ckpt(
    resume_path: Path,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    resume_full: bool,
) -> Tuple[int, int]:
    ckpt = torch.load(resume_path, map_location="cpu")
    state = ckpt["model"]

    # Handle legacy compiled prefixes.
    if any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k[len("_orig_mod."):]: v for k, v in state.items()}

    model.load_state_dict(state, strict=True)

    global_step = int(ckpt.get("global_step", 0))
    local_step = int(ckpt.get("local_step", 0))

    if resume_full:
        if "optim" in ckpt and ckpt["optim"] is not None:
            optim.load_state_dict(ckpt["optim"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])

    return global_step, local_step


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dl: DataLoader,
    device: torch.device,
    precision: str,
    eos_id: int,
    eos_weight: float,
    max_batches: int = 50,
) -> float:
    model.eval()
    ac_dtype = _autocast_dtype(precision)

    losses = []
    it = iter(dl)
    for _ in range(max_batches):
        try:
            batch = next(it)
        except StopIteration:
            break

        # Support (x,y) or (x,y,mask)
        if len(batch) == 2:
            input_u16, labels_u16 = batch
            loss_mask = torch.ones_like(labels_u16, dtype=torch.float32)
        else:
            input_u16, labels_u16, loss_mask = batch

        input_ids = input_u16.to(device, dtype=torch.long, non_blocking=True)
        labels = labels_u16.to(device, dtype=torch.long, non_blocking=True)
        loss_mask = loss_mask.to(device, dtype=torch.float32, non_blocking=True)

        if ac_dtype is not None:
            with torch.autocast("cuda", dtype=ac_dtype):
                logits = model(input_ids)
                loss = masked_weighted_ce_loss(logits, labels, loss_mask, eos_id=eos_id, eos_weight=eos_weight)
        else:
            logits = model(input_ids)
            loss = masked_weighted_ce_loss(logits, labels, loss_mask, eos_id=eos_id, eos_weight=eos_weight)

        losses.append(float(loss.item()))

    model.train()
    return float(sum(losses) / max(1, len(losses)))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)

    # Output
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--samples_dir", required=True)
    ap.add_argument("--tokenizer_path", required=True)

    # Model
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--n_heads", type=int, default=12)
    ap.add_argument("--d_ff", type=int, default=3072)
    ap.add_argument("--dropout", type=float, default=0.0)

    # Special tokens
    ap.add_argument("--bos_id", type=int, default=2)
    ap.add_argument("--eos_id", type=int, default=3)

    # Loss shaping
    ap.add_argument("--mask_bos_in_loss", action="store_true", help="Mask BOS targets using loss_mask (recommended).")
    ap.add_argument("--mask_last_label_in_loss", action="store_true", help="Mask last label position in each block.")
    ap.add_argument(
        "--eos_weight",
        type=float,
        default=0.2,
        help="Down-weight EOS targets in the loss (e.g., 0.2). Set 1.0 to disable.",
    )
    ap.add_argument(
        "--eos_weight_warmup_steps",
        type=int,
        default=0,
        help="If >0, use eos_weight for first N steps, then switch to 1.0.",
    )

    # Train
    ap.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--micro_bsz", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--max_steps", type=int, default=80000)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)

    # Logging / eval / save
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--milestone_every", type=int, default=10000)

    # Sampling during training
    ap.add_argument("--add_bos_to_prompts", action="store_true")
    ap.add_argument("--sample_temperature", type=float, default=0.7)
    ap.add_argument("--sample_top_p", type=float, default=0.9)
    ap.add_argument("--sample_top_k", type=int, default=0)
    ap.add_argument("--sample_max_new_tokens", type=int, default=256)
    ap.add_argument(
        "--sample_min_new_tokens",
        type=int,
        default=32,
        help="Prevent 'empty samples' by not allowing EOS before N tokens.",
    )

    # Resume
    ap.add_argument("--resume_path", type=str, default="")
    ap.add_argument("--resume_full", action="store_true")

    # torch.compile
    ap.add_argument("--compile", action="store_true")

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_dir = Path(_resolve_path(args.train_dir))
    val_dir = Path(_resolve_path(args.val_dir))
    out_dir = Path(_resolve_path(args.out_dir))
    samples_dir = Path(_resolve_path(args.samples_dir))
    tok_path = Path(_resolve_path(args.tokenizer_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "This script expects a CUDA GPU."

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
    ac_dtype = _autocast_dtype(args.precision)
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    # Data (MiniMind-style loss_mask from dataset)
    train_ds = PackedBinDataset(
        str(train_dir),
        seq_len=args.seq_len,
        bos_id=args.bos_id,
        eos_id=args.eos_id,
        mask_bos_in_loss=args.mask_bos_in_loss,
        mask_last_label_in_loss=args.mask_last_label_in_loss,
    )
    val_ds = PackedBinDataset(
        str(val_dir),
        seq_len=args.seq_len,
        bos_id=args.bos_id,
        eos_id=args.eos_id,
        mask_bos_in_loss=args.mask_bos_in_loss,
        mask_last_label_in_loss=args.mask_last_label_in_loss,
    )

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
        resume_path = Path(_resolve_path(args.resume_path))
        global_step, local_step = load_ckpt(
            resume_path=resume_path,
            model=model,
            optim=optim,
            scaler=scaler if use_fp16 else None,
            resume_full=args.resume_full,
        )
        print(f"[resume] loaded {resume_path} (global_step={global_step}, local_step={local_step})")

    # Compile (after loading)
    if args.compile:
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
            print("[compile] torch.compile enabled")
        except Exception as e:
            print(f"[compile] torch.compile failed: {e}")

    # Save config snapshot
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps({**vars(args), "model_cfg": asdict(cfg)}, indent=2),
        encoding="utf-8",
    )

    model.train()
    data_iter = iter(train_dl)

    t_window = time.time()
    window_tokens = 0

    while local_step < args.max_steps:
        # LR warmup (stable baseline)
        lr = lr_schedule(global_step, args.warmup_steps, args.lr)
        for pg in optim.param_groups:
            pg["lr"] = lr

        optim.zero_grad(set_to_none=True)
        accum_loss_scaled = 0.0
        accum_loss_raw = 0.0

        # EOS weight schedule
        cur_eos_weight = float(args.eos_weight)
        if args.eos_weight_warmup_steps and global_step >= int(args.eos_weight_warmup_steps):
            cur_eos_weight = 1.0

        for _ in range(args.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_dl)
                batch = next(data_iter)

            if len(batch) == 2:
                input_u16, labels_u16 = batch
                loss_mask = torch.ones_like(labels_u16, dtype=torch.float32)
            else:
                input_u16, labels_u16, loss_mask = batch

            input_ids = input_u16.to(device, dtype=torch.long, non_blocking=True)
            labels = labels_u16.to(device, dtype=torch.long, non_blocking=True)
            loss_mask = loss_mask.to(device, dtype=torch.float32, non_blocking=True)

            window_tokens += int(loss_mask.sum().item())

            if ac_dtype is not None:
                with torch.autocast("cuda", dtype=ac_dtype):
                    logits = model(input_ids)
                    loss_raw = masked_weighted_ce_loss(
                        logits=logits,
                        labels=labels,
                        loss_mask=loss_mask,
                        eos_id=args.eos_id,
                        eos_weight=cur_eos_weight,
                    )
                    loss = loss_raw / args.grad_accum
            else:
                logits = model(input_ids)
                loss_raw = masked_weighted_ce_loss(
                    logits=logits,
                    labels=labels,
                    loss_mask=loss_mask,
                    eos_id=args.eos_id,
                    eos_weight=cur_eos_weight,
                )
                loss = loss_raw / args.grad_accum

            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss_scaled += float(loss.detach().item())
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

        # Logging
        if global_step % args.log_every == 0:
            dt = time.time() - t_window
            tok_s = window_tokens / max(dt, 1e-6)
            # True CE ~= mean of loss_raw over accumulation steps
            true_ce = accum_loss_raw / max(1, args.grad_accum)
            print(
                f"[train] step={global_step} loss_raw={true_ce:.4f} "
                f"(eos_w={cur_eos_weight:g}) lr={lr:.2e} tok/s={tok_s:.0f}"
            )
            t_window = time.time()
            window_tokens = 0

        # Eval + samples
        if global_step % args.eval_every == 0:
            val_loss = evaluate(
                model=model,
                dl=val_dl,
                device=device,
                precision=args.precision,
                eos_id=args.eos_id,
                eos_weight=cur_eos_weight if args.eos_weight_warmup_steps else float(args.eos_weight),
                max_batches=50,
            )
            print(f"[eval] step={global_step} val_loss={val_loss:.4f}")

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
                    add_bos=args.add_bos_to_prompts,
                    bos_id=args.bos_id,
                    min_new_tokens=args.sample_min_new_tokens,
                    greedy=False,
                    debug=True,
                )
                print(f"[sample] wrote {out_path}")
            except Exception as e:
                print(f"[sample] failed: {e}")

        # Save checkpoints
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
            print(f"[ckpt] saved latest + step_{global_step:06d}.pt to {out_dir}")

        # Extra milestone (every N steps)
        if args.milestone_every > 0 and (global_step % args.milestone_every == 0):
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
            print(f"[milestone] saved step_{global_step:06d}.pt")

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
