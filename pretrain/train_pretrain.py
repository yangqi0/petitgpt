# pretrain/train_pretrain.py
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# -----------------------------------------------------------------------------
# Make project root importable so we can import src.model from pretrain/
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset_pretrain import PackedBinDataset  # local to pretrain/
from src.model import GPT, GPTConfig           # in /src


# -----------------------------------------------------------------------------
# Performance toggles (safe defaults for training)
# -----------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Prefer faster SDPA kernels where possible
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy with ignore_index=-100.
    logits: [B, T, V]
    labels: [B, T]
    """
    b, t, v = logits.shape
    return F.cross_entropy(
        logits.view(b * t, v),
        labels.view(b * t),
        ignore_index=-100,
        reduction="mean",
    )


def get_lr(step: int, max_steps: int, base_lr: float, warmup_steps: int) -> float:
    """Linear warmup + cosine decay."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
    progress = min(1.0, max(0.0, progress))
    return base_lr * 0.5 * (1.0 + math.cos(progress * math.pi))


def _autocast_dtype(precision: str) -> Optional[torch.dtype]:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return None


def save_ckpt(
    out_dir: Path,
    step: int,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    cfg: Dict,
) -> None:
    """Save a fault-tolerant checkpoint to out_dir/latest.pt (atomic replace)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "cfg": cfg,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

    tmp = out_dir / "latest.tmp.pt"
    final = out_dir / "latest.pt"
    torch.save(ckpt, tmp)
    os.replace(tmp, final)


def load_ckpt(
    path: Path,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
) -> int:
    """Load checkpoint and return step."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    optim.load_state_dict(ckpt["optim"])

    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    if ckpt.get("torch_rng") is not None:
        torch.set_rng_state(ckpt["torch_rng"])
    if torch.cuda.is_available() and ckpt.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(ckpt["cuda_rng"])

    return int(ckpt["step"])


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dl: DataLoader,
    device: torch.device,
    precision: str,
    max_batches: int = 50,
) -> float:
    """Compute mean validation loss on at most max_batches."""
    model.eval()
    losses = []

    ac_dtype = _autocast_dtype(precision)

    for bi, (input_ids, labels) in enumerate(dl):
        if bi >= max_batches:
            break

        input_ids = input_ids.to(device, dtype=torch.long, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)

        if ac_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=ac_dtype):
                logits = model(input_ids)
                loss = masked_ce_loss(logits, labels)
        else:
            logits = model(input_ids)
            loss = masked_ce_loss(logits, labels)

        losses.append(float(loss.item()))

    model.train()
    return float(sum(losses) / max(1, len(losses)))


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_dir", required=True, help="Path to token shards train/ directory")
    ap.add_argument("--val_dir", required=True, help="Path to token shards val/ directory")
    ap.add_argument("--out_dir", default="checkpoints/pretrain_120m")
    ap.add_argument("--tb_dir", default="runs/pretrain_120m")

    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--n_heads", type=int, default=12)
    ap.add_argument("--d_ff", type=int, default=3072)
    ap.add_argument("--dropout", type=float, default=0.0)

    ap.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--micro_bsz", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=16)

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=500)

    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--compile", action="store_true", help="Enable torch.compile (recommended for long runs)")

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "This script expects a CUDA GPU."

    # Optional hints: reduce CPU oversubscription when using many DataLoader workers
    print(f"[env] OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '(unset)')}")
    print(f"[env] MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS', '(unset)')}")

    # Build model config
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

    # Instantiate model (do NOT compile yet; compile after optional resume load)
    model = GPT(cfg).to(device)

    # Optimizer
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # GradScaler only for fp16
    use_fp16 = args.precision == "fp16"
    scaler: Optional[GradScaler] = GradScaler("cuda", enabled=use_fp16)

    # Datasets / loaders
    train_ds = PackedBinDataset(args.train_dir, seq_len=args.seq_len)
    val_ds = PackedBinDataset(args.val_dir, seq_len=args.seq_len)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.micro_bsz,
        shuffle=False,               # sequential read is much faster for memmap shards
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=args.micro_bsz,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=2,
    )

    out_dir = Path(args.out_dir)
    tb_dir = Path(args.tb_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    # Save config once
    (out_dir / "config.json").write_text(
        json.dumps({**vars(args), "model_cfg": asdict(cfg)}, indent=2),
        encoding="utf-8",
    )

    # Resume (load BEFORE compile)
    step = 0
    ckpt_path = out_dir / "latest.pt"
    if args.resume and ckpt_path.exists():
        step = load_ckpt(ckpt_path, model, optim, scaler if use_fp16 else None)
        print(f"[resume] Loaded {ckpt_path} at step={step}")

    # Compile after resume load (best practice)
    if args.compile:
        model = torch.compile(model)
        print("[compile] torch.compile enabled")

    writer = SummaryWriter(log_dir=str(tb_dir))

    # Training state
    model.train()
    t0 = time.time()
    tokens_seen = 0

    # Sliding window throughput
    window_tokens = 0
    window_t0 = time.time()

    ac_dtype = _autocast_dtype(args.precision)
    data_iter = iter(train_dl)

    while step < args.max_steps:
        optim.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(args.grad_accum):
            try:
                input_ids, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(train_dl)
                input_ids, labels = next(data_iter)

            # Cast to long on GPU (dataset can return uint16 on CPU)
            input_ids = input_ids.to(device, dtype=torch.long, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)

            ntok = int((labels != -100).sum().item())
            tokens_seen += ntok
            window_tokens += ntok

            if ac_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=ac_dtype):
                    logits = model(input_ids)
                    loss = masked_ce_loss(logits, labels) / args.grad_accum
            else:
                logits = model(input_ids)
                loss = masked_ce_loss(logits, labels) / args.grad_accum

            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += float(loss.item())

        # Update LR
        lr = get_lr(step, args.max_steps, args.lr, args.warmup_steps)
        for pg in optim.param_groups:
            pg["lr"] = lr

        # Step optimizer
        if use_fp16:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optim)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()

        # Logging (window + avg throughput)
        if (step + 1) % args.log_every == 0:
            dt_total = time.time() - t0
            dt_win = time.time() - window_t0
            tok_s_win = window_tokens / max(1e-9, dt_win)
            tok_s_avg = tokens_seen / max(1e-9, dt_total)

            writer.add_scalar("train/loss", accum_loss, step + 1)
            writer.add_scalar("train/lr", lr, step + 1)
            writer.add_scalar("train/tokens_per_s_win", tok_s_win, step + 1)
            writer.add_scalar("train/tokens_per_s_avg", tok_s_avg, step + 1)

            print(
                f"step {step+1:6d} | loss {accum_loss:.4f} | lr {lr:.2e} "
                f"| tok/s(win) {tok_s_win:,.0f} | tok/s(avg) {tok_s_avg:,.0f}"
            )

            window_tokens = 0
            window_t0 = time.time()

        # Eval
        if (step + 1) % args.eval_every == 0:
            val_loss = evaluate(model, val_dl, device, precision=args.precision, max_batches=50)
            writer.add_scalar("val/loss", val_loss, step + 1)
            print(f"[eval] step {step+1} | val_loss {val_loss:.4f}")

        # Save
        if (step + 1) % args.save_every == 0:
            save_ckpt(out_dir, step + 1, model, optim, scaler if use_fp16 else None, cfg=vars(args))
            print(f"[ckpt] saved {out_dir/'latest.pt'} at step {step+1}")

        step += 1

    # Final save
    save_ckpt(out_dir, step, model, optim, scaler if use_fp16 else None, cfg=vars(args))
    print("[done] training finished")
    writer.close()


if __name__ == "__main__":
    main()
