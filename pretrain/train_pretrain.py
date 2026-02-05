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
from typing import Dict, Optional, Tuple

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

from dataset_pretrain import PackedBinDataset
from src.model import GPT, GPTConfig
from sample import generate_default_samples

# -----------------------------------------------------------------------------
# Performance toggles
# -----------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
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
    """Linear warmup + cosine decay to (almost) zero."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
    progress = min(1.0, max(0.0, progress))
    return base_lr * 0.5 * (1.0 + math.cos(progress * math.pi))


def autocast_dtype(precision: str) -> Optional[torch.dtype]:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return None


def save_ckpt_atomic(path: Path, payload: Dict) -> None:
    """Atomically write checkpoint (tmp -> replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.pt")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _strip_prefix(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """Remove prefix from state_dict keys if present."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            out[k] = v
    return out


def _add_prefix(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """Add prefix to all state_dict keys."""
    return {prefix + k: v for k, v in state.items()}


def normalize_state_dict_keys_for_model(
    model: torch.nn.Module,
    ckpt_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Make checkpoint state_dict keys compatible with current model state_dict keys.

    Common case:
      - torch.compile saves keys with "_orig_mod." prefix.
      - non-compiled model expects keys without that prefix.

    This function:
      - detects whether ckpt has "_orig_mod." prefix
      - detects whether model expects "_orig_mod." prefix
      - converts accordingly
    """
    model_keys = list(model.state_dict().keys())
    ckpt_keys = list(ckpt_state.keys())

    model_has_prefix = len(model_keys) > 0 and model_keys[0].startswith("_orig_mod.")
    ckpt_has_prefix = len(ckpt_keys) > 0 and ckpt_keys[0].startswith("_orig_mod.")

    if ckpt_has_prefix and not model_has_prefix:
        return _strip_prefix(ckpt_state, "_orig_mod.")
    if (not ckpt_has_prefix) and model_has_prefix:
        return _add_prefix(ckpt_state, "_orig_mod.")
    return ckpt_state


def save_ckpt(
    out_dir: Path,
    global_step: int,
    local_step: int,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    cfg: Dict,
) -> None:
    """
    Save latest.pt each time. Also save milestone every 10k global steps.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "global_step": global_step,
        "step": local_step,
        "model": (model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()),
        "optim": optim.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "cfg": cfg,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

    save_ckpt_atomic(out_dir / "latest.pt", ckpt)

    if global_step > 0 and (global_step % 10_000 == 0):
        save_ckpt_atomic(out_dir / f"step_{global_step:06d}.pt", ckpt)


def load_full_ckpt(
    path: Path,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
) -> Tuple[int, int]:
    """
    Load full checkpoint (model + optim + scaler + rng).
    Returns (global_step, local_step).
    """
    ckpt = torch.load(path, map_location="cpu")

    model_state = normalize_state_dict_keys_for_model(model, ckpt["model"])
    model.load_state_dict(model_state, strict=True)

    optim.load_state_dict(ckpt["optim"])

    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    if ckpt.get("torch_rng") is not None:
        torch.set_rng_state(ckpt["torch_rng"])
    if torch.cuda.is_available() and ckpt.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(ckpt["cuda_rng"])

    global_step = int(ckpt.get("global_step", ckpt.get("step", 0)))
    local_step = int(ckpt.get("step", 0))
    return global_step, local_step


def load_weights_only(path: Path, model: torch.nn.Module) -> int:
    """
    Load only model weights. Return previous global step for step_offset.
    Compatible with compiled/non-compiled checkpoints.
    """
    ckpt = torch.load(path, map_location="cpu")
    model_state = normalize_state_dict_keys_for_model(model, ckpt["model"])
    model.load_state_dict(model_state, strict=True)
    prev_global = int(ckpt.get("global_step", ckpt.get("step", 0)))
    return prev_global


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
    ac_dtype = autocast_dtype(precision)

    for bi, (input_ids, labels) in enumerate(dl):
        if bi >= max_batches:
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
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--out_dir", default="checkpoints/pretrain_120m")
    ap.add_argument("--tb_dir", default="runs/pretrain_120m")

    # Sampling
    ap.add_argument("--tokenizer_path", default="../tokenizer/tokenizer.json")
    ap.add_argument("--samples_dir", default="../samples/pretrain")
    ap.add_argument("--sample_temperature", type=float, default=0.9)
    ap.add_argument("--sample_top_p", type=float, default=0.95)
    ap.add_argument("--sample_max_new_tokens", type=int, default=200)
    ap.add_argument("--bos_id", type=int, default=2)
    ap.add_argument("--eos_id", type=int, default=3)
    ap.add_argument("--add_bos", action="store_true")

    # Model
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--n_heads", type=int, default=12)
    ap.add_argument("--d_ff", type=int, default=3072)
    ap.add_argument("--dropout", type=float, default=0.0)

    # Training
    ap.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--micro_bsz", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--max_steps", type=int, default=20000)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--save_every", type=int, default=1000)

    # Resume (scheme A supported)
    ap.add_argument("--resume_path", type=str, default="")
    ap.add_argument("--resume_weights_only", action="store_true")
    ap.add_argument("--resume_full", action="store_true")

    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--compile", action="store_true")

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "This script expects a CUDA GPU."

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
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    use_fp16 = args.precision == "fp16"
    scaler: Optional[GradScaler] = GradScaler("cuda", enabled=use_fp16)

    # Data
    train_ds = PackedBinDataset(args.train_dir, seq_len=args.seq_len)
    val_ds = PackedBinDataset(args.val_dir, seq_len=args.seq_len)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.micro_bsz,
        shuffle=False,
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

    # ---- debug one batch ----
    input_ids, labels = next(iter(train_dl))
    print("input_ids shape", input_ids.shape, "labels shape", labels.shape)
    print("labels != -100 ratio:", (labels != -100).float().mean().item())
    print("eos in input ratio:", (input_ids == args.eos_id).float().mean().item())
    print("eos in labels ratio:", (labels == args.eos_id).float().mean().item())
    print("first 50 input ids:", input_ids[0, :50].tolist())
    print("first 50 labels:", labels[0, :50].tolist())
    raise SystemExit


    out_dir = Path(args.out_dir)
    tb_dir = Path(args.tb_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    (out_dir / "config.json").write_text(
        json.dumps({**vars(args), "model_cfg": asdict(cfg)}, indent=2),
        encoding="utf-8",
    )

    # Resume
    local_step = 0
    step_offset = 0

    if args.resume_path:
        ckpt_path = Path(args.resume_path)
        assert ckpt_path.exists(), f"resume_path not found: {ckpt_path}"

        if args.resume_full:
            gstep, lstep = load_full_ckpt(ckpt_path, model, optim, scaler if use_fp16 else None)
            step_offset = gstep - lstep
            local_step = lstep
            print(f"[resume_full] loaded {ckpt_path} global_step={gstep} local_step={lstep}")

        elif args.resume_weights_only:
            prev_global = load_weights_only(ckpt_path, model)
            step_offset = prev_global
            local_step = 0
            print(f"[resume_weights_only] loaded weights from {ckpt_path} prev_global_step={prev_global} (restart schedule)")

        else:
            raise ValueError("If --resume_path is set, choose either --resume_full or --resume_weights_only.")

    # Compile AFTER resume (important)
    if args.compile:
        model = torch.compile(model)
        print("[compile] torch.compile enabled")

    writer = SummaryWriter(log_dir=str(tb_dir))

    model.train()
    t0 = time.time()
    tokens_seen = 0
    window_tokens = 0
    window_t0 = time.time()

    ac_dtype = autocast_dtype(args.precision)
    data_iter = iter(train_dl)

    while local_step < args.max_steps:
        optim.zero_grad(set_to_none=True)
        accum_loss = 0.0  # scaled loss summed across micro-steps (≈ mean loss)
        accum_loss_raw = 0.0  # unscaled CE summed across micro-steps (for logging)

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
                    loss_raw = masked_ce_loss(logits, labels)  # real CE
                    loss = loss_raw / args.grad_accum          # for backward
            else:
                logits = model(input_ids)
                loss_raw = masked_ce_loss(logits, labels)
                loss = loss_raw / args.grad_accum

            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += float(loss.item())
            accum_loss_raw += float(loss_raw.detach().item())

        lr = get_lr(local_step, args.max_steps, args.lr, args.warmup_steps)
        for pg in optim.param_groups:
            pg["lr"] = lr

        if use_fp16:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optim)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()

        global_step = step_offset + (local_step + 1)

        if (local_step + 1) % args.log_every == 0:
            dt_total = time.time() - t0
            dt_win = time.time() - window_t0
            tok_s_win = window_tokens / max(1e-9, dt_win)
            tok_s_avg = tokens_seen / max(1e-9, dt_total)

            writer.add_scalar("train/loss", accum_loss, global_step)
            writer.add_scalar("train/loss_raw", accum_loss_raw / args.grad_accum, global_step)
            writer.add_scalar("train/lr", lr, global_step)
            writer.add_scalar("train/tokens_per_s_win", tok_s_win, global_step)
            writer.add_scalar("train/tokens_per_s_avg", tok_s_avg, global_step)

            print(
                f"global_step {global_step:7d} | local_step {local_step+1:7d} "
                f"| loss {accum_loss:.4f} | lr {lr:.2e} "
                f"| tok/s(win) {tok_s_win:,.0f} | tok/s(avg) {tok_s_avg:,.0f}"
            )

            window_tokens = 0
            window_t0 = time.time()

        if (local_step + 1) % args.eval_every == 0:
            val_loss = evaluate(model, val_dl, device, precision=args.precision, max_batches=50)
            writer.add_scalar("val/loss", val_loss, global_step)
            print(f"[eval] global_step {global_step} | val_loss {val_loss:.4f}")

            samples_dir = Path(args.samples_dir)
            out_path = samples_dir / f"step_{global_step:06d}.txt"
            try:
                generate_default_samples(
                    model=model,
                    tokenizer_path=args.tokenizer_path,
                    device=device,
                    max_seq_len=args.seq_len,
                    precision=args.precision,
                    out_path=out_path,
                    temperature=args.sample_temperature,
                    top_p=args.sample_top_p,
                    max_new_tokens=args.sample_max_new_tokens,
                    eos_id=args.eos_id,
                    add_bos=args.add_bos,
                    bos_id=args.bos_id,
                )
                print(f"[sample] wrote {out_path}")
            except Exception as e:
                print(f"[sample] failed: {e}")

        if (local_step + 1) % args.save_every == 0:
            save_ckpt(
                out_dir=out_dir,
                global_step=global_step,
                local_step=local_step + 1,
                model=model,
                optim=optim,
                scaler=scaler if use_fp16 else None,
                cfg=vars(args),
            )
            print(f"[ckpt] saved {out_dir/'latest.pt'} at global_step {global_step}")

        local_step += 1

    final_global = step_offset + local_step
    save_ckpt(
        out_dir=out_dir,
        global_step=final_global,
        local_step=local_step,
        model=model,
        optim=optim,
        scaler=scaler if use_fp16 else None,
        cfg=vars(args),
    )
    print(f"[done] training finished at global_step={final_global}")
    writer.close()


if __name__ == "__main__":
    main()
