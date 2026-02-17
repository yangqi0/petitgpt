# pretrain/train_pretrain.py
# Robust GPT pretraining for petitgpt (causal LM)
# - PackedBinDataset: returns (input_ids_u16, labels_u16, loss_mask_f32/u8)
# - token-level loss mask (MiniMind-style): weighted reduction by sum(mask)
# - optional EOS down-weight warmup to avoid early EOS collapse
# - gradient accumulation, bf16/fp16 autocast, grad clip, checkpoints, eval, samples
# - debug: dataset stats, label shift sanity, future-leak checks

from __future__ import annotations

import os
import argparse
import json
import sys
import time
import random
import signal
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
# Prefer a fixed sampler implementation if present.
try:
    from sample_fixed import generate_default_samples  # type: ignore  # noqa: E402
except Exception:
    from sample import generate_default_samples  # noqa: E402
from src.model import GPT, GPTConfig  # noqa: E402

# -----------------------------------------------------------------------------
# Performance toggles
# -----------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
except Exception:
    pass


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def infer_vocab_size_from_tokenizer_json(path: str) -> int:
    """Infer vocab size from HF tokenizers' tokenizer.json."""
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    model = obj.get("model", {})
    vocab = model.get("vocab", None)
    if isinstance(vocab, dict):
        return len(vocab)
    if isinstance(vocab, list):
        return len(vocab)

    added = obj.get("added_tokens", [])
    if isinstance(added, list) and added:
        return max(int(t.get("id", -1)) for t in added) + 1

    raise ValueError(f"Cannot infer vocab_size from tokenizer.json: {path}")


def _resolve_path(p: str) -> str:
    path = Path(p)
    if path.exists():
        return str(path)
    alt = PROJECT_ROOT / p
    if alt.exists():
        return str(alt)
    return str(path)



def _resolve_resume_path(resume_path: str, out_dir: Path, resume_step: int = -1) -> Path | None:
    """Resolve a checkpoint path.

    Accepts:
      - a direct .pt file path
      - a directory containing latest.pt and/or step_XXXXXX.pt
      - a relative path (resolved against PROJECT_ROOT)
    """
    if not resume_path:
        return None

    p = Path(_resolve_path(resume_path))

    # If user passed an output directory, select a checkpoint inside it.
    if p.is_dir():
        if resume_step is not None and resume_step >= 0:
            cand = p / f"step_{resume_step:06d}.pt"
            if cand.exists():
                return cand
            # Sometimes users keep checkpoints under out_dir/ckpt or similar; try a shallow search.
            cand2 = next(p.glob(f"**/step_{resume_step:06d}.pt"), None)
            if cand2 is not None and cand2.exists():
                return cand2

        latest = p / "latest.pt"
        if latest.exists():
            return latest

        # Fallback to the largest step_*.pt
        steps = sorted(p.glob("step_*.pt"))
        if steps:
            return steps[-1]
        return None

    # If it's a file, use it.
    if p.is_file():
        return p

    # If not found, try interpreting it relative to out_dir.
    p2 = out_dir / resume_path
    if p2.is_file():
        return p2

    return None
def set_seed(seed: int) -> None:
    import random

# -----------------------------------------------------------------------------
# Graceful Ctrl+C (SIGINT): request stop and save a checkpoint at a safe boundary.
# -----------------------------------------------------------------------------
_STOP_REQUESTED = False


def _handle_sigint(signum, frame):  # pragma: no cover
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print("\n[interrupt] SIGINT received (Ctrl+C). Will save and stop at next safe point...", flush=True)


try:
    signal.signal(signal.SIGINT, _handle_sigint)
except Exception:
    pass

    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None

    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_schedule(step: int, warmup_steps: int, base_lr: float) -> float:
    """Linear warmup then constant LR (stable baseline)."""
    if warmup_steps <= 0:
        return base_lr
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    return base_lr


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
    *,
    eos_id: int,
    eos_weight: float,
) -> torch.Tensor:
    """
    Token-level CE:
      - per-token CE (reduction='none')
      - multiply by loss_mask
      - optionally down-weight EOS targets
      - normalize by sum(weights) (NOT by B*T)
    """
    B, T, V = logits.shape
    l = logits.reshape(B * T, V)
    y = labels.reshape(B * T)
    m = loss_mask.reshape(B * T)

    # cross_entropy expects int64 targets
    per = F.cross_entropy(l, y, reduction="none")  # [B*T]
    w = m
    if eos_weight != 1.0:
        eos_m = (y == int(eos_id)).to(w.dtype)
        w = w * (1.0 + eos_m * (float(eos_weight) - 1.0))

    denom = w.sum().clamp_min(1.0)
    return (per * w).sum() / denom


@torch.no_grad()
def masked_ce_128_debug(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    tt: int = 128,
) -> float:
    """Same as training loss (mask-weighted), but only first tt tokens."""
    B, T, V = logits.shape
    t = min(T, int(tt))
    l = logits[:, :t, :].reshape(-1, V).float()
    y = labels[:, :t].reshape(-1)
    m = loss_mask[:, :t].reshape(-1).float()
    per = F.cross_entropy(l, y, reduction="none")
    return float((per * m).sum().item() / m.sum().clamp_min(1.0).item())


# -----------------------------------------------------------------------------
# Checkpoints
# -----------------------------------------------------------------------------
def _atomic_torch_save(obj: dict, final_path: Path) -> None:
    """
    Atomically save a torch checkpoint:
      1) write to final_path.with_suffix(final_path.suffix + ".tmp")
      2) flush + fsync
      3) os.replace(tmp, final)
    This prevents half-written latest.pt when the filesystem is flaky.
    """
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")

    # best effort: remove stale tmp
    try:
        if tmp_path.exists():
            tmp_path.unlink()
    except Exception:
        pass

    # Use a real file handle so we can fsync.
    # torch.save can accept a file-like object.
    with open(tmp_path, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())

    # Atomic replace on POSIX
    os.replace(tmp_path, final_path)

def save_ckpt(
    out_dir: Path,
    global_step: int,
    local_step: int,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    model_config: Dict,
    train_args: Dict,
) -> None:
    """
    Robust checkpoint saving:
      - writes latest.pt and step_XXXXXX.pt atomically
      - avoids corrupted latest.pt on interruption / flaky disks
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model

    ckpt = {
        "model": model_to_save.state_dict(),
        "optim": optim.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "global_step": int(global_step),
        "local_step": int(local_step),
        "config": model_config,
        "train_args": train_args,
        "saved_at_unix": int(time.time()),
    }

    latest_path = out_dir / "latest.pt"
    step_path = out_dir / f"step_{global_step:06d}.pt"

    try:
        _atomic_torch_save(ckpt, latest_path)
        _atomic_torch_save(ckpt, step_path)
    except Exception as e:
        # If saving fails, don't leave tmp files around (best effort)
        for p in [latest_path, step_path]:
            tmp = p.with_suffix(p.suffix + ".tmp")
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
        raise RuntimeError(f"[ckpt] save failed: {e}") from e

def load_ckpt(
    resume_path: Path,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    resume_full: bool,
) -> Tuple[int, int]:
    ckpt = torch.load(resume_path, map_location="cpu")
    state = ckpt["model"]

    # handle accidental _orig_mod prefix
    if any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k[len("_orig_mod.") :]: v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    global_step = int(ckpt.get("global_step", 0))
    local_step = int(ckpt.get("local_step", 0))

    if resume_full:
        if ckpt.get("optim") is not None:
            optim.load_state_dict(ckpt["optim"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])

    return global_step, local_step


# -----------------------------------------------------------------------------
# Debug sanity checks (causality + label shift)
# -----------------------------------------------------------------------------
@torch.no_grad()
def causal_leak_check(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    device: torch.device,
    *,
    vocab_size: int,
    check_pos: int = 128,
    delta_pos: int = 8,
) -> float:
    """
    Perturb ONE token at position (check_pos + delta_pos) and measure how much
    the logits on prefix [0:check_pos] change. For a strictly causal model,
    this should be ~0.
    """
    model.eval()
    x = input_ids.to(device, non_blocking=True)
    logits1 = model(x).float()

    x2 = x.clone()
    p = min(x2.shape[1] - 1, int(check_pos + delta_pos))
    x2[:, p] = (x2[:, p] + 123) % int(vocab_size)
    logits2 = model(x2).float()

    diff = (logits1[:, :check_pos, :] - logits2[:, :check_pos, :]).abs().max().item()
    print(f"[dbg] local_future_leak_check max_abs_diff={diff:.6f} (expect ~0)")
    model.train()
    return float(diff)


@torch.no_grad()
def label_shift_sanity(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
) -> float:
    """
    Check labels are next-token targets:
        labels[t] == input_ids[t+1]  (for supervised positions).
    """
    m = (loss_mask[:, :-1] > 0)
    if m.sum().item() == 0:
        acc = 0.0
    else:
        ok = ((labels[:, :-1] == input_ids[:, 1:]) & m).float().sum().item()
        tot = m.float().sum().item()
        acc = ok / max(1.0, tot)
    print(f"[dbg] label_shift_sanity next-token match over supervised: {acc:.6f}")
    return float(acc)


# -----------------------------------------------------------------------------
# Eval + dataset stats
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dl: DataLoader,
    device: torch.device,
    precision: str,
    *,
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
                loss = masked_weighted_ce_loss(
                    logits, labels, loss_mask, eos_id=eos_id, eos_weight=eos_weight
                )
        else:
            logits = model(input_ids)
            loss = masked_weighted_ce_loss(
                logits, labels, loss_mask, eos_id=eos_id, eos_weight=eos_weight
            )

        losses.append(float(loss.item()))

    model.train()
    return float(sum(losses) / max(1, len(losses)))


def _estimate_dataset_stats(ds: PackedBinDataset, seq_len: int, *, name: str) -> None:
    """
    Best-effort dataset stats printer.
    Works with PackedBinDataset used in this project (bin shards + fixed block_size = seq_len+1).
    """
    shards = getattr(ds, "shards", None)
    shard_lens = getattr(ds, "shard_lens", None)

    if isinstance(shards, list) and isinstance(shard_lens, list) and len(shards) == len(shard_lens):
        raw_tokens = int(sum(int(x) for x in shard_lens))
        block_size = int(seq_len + 1)
        n_blocks = raw_tokens // block_size
        epoch_tokens = n_blocks * int(seq_len)

        print(f"[*] {name} dataset stats:")
        print(f"    - shards: {len(shards)}")
        print(f"    - raw tokens in .bin (sum of shard lengths): {raw_tokens:,}")
        print(f"    - block size (seq_len+1): {block_size}")
        print(f"    - full blocks (n_blocks): {n_blocks:,}")
        print(f"    - epoch-equivalent tokens (n_blocks*seq_len): {epoch_tokens:,}")
    else:
        # fallback
        n = int(len(ds))
        print(f"[*] {name} dataset stats (fallback): len={n:,} blocks, approx tokens={n*seq_len:,}")

    # sample a few batches to estimate mask density and EOS fraction
    try:
        # sample 32 blocks from dataset (deterministic indices)
        n_samp = min(32, int(len(ds)))
        mask_sum = 0.0
        eos_sum = 0.0
        tot = 0.0
        for i in range(n_samp):
            batch = ds[i]
            if len(batch) == 2:
                _, labels_u16 = batch
                loss_mask = torch.ones_like(labels_u16, dtype=torch.float32)
            else:
                _, labels_u16, loss_mask = batch

            m = loss_mask.float()
            y = labels_u16.long()

            mask_sum += float(m.sum().item())
            tot += float(m.numel())
            # try to get eos_id from dataset if present
            eos_id = int(getattr(ds, "eos_id", 3))
            eos_sum += float(((y == eos_id) & (m > 0)).float().sum().item())

        avg_sup = mask_sum / max(1.0, float(n_samp))
        eos_frac = eos_sum / max(1.0, mask_sum)
        print(f"    - avg supervised tokens per block (sampled): {avg_sup:.1f} / {seq_len}")
        print(f"    - avg EOS fraction over supervised tokens (sampled): {eos_frac:.4f}")
    except Exception as e:
        print(f"    - (could not sample mask/EOS stats: {e})")


# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------
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
    ap.add_argument("--no_mask_bos_in_loss", action="store_true")
    ap.add_argument("--no_mask_last_label_in_loss", action="store_true")
    ap.add_argument("--eos_weight", type=float, default=1.0)
    ap.add_argument("--eos_weight_warmup_steps", type=int, default=0)

    # Train
    ap.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--micro_bsz", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--max_steps", type=int, default=100000)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)

    # Logging / eval / save
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--debug_every", type=int, default=500)

    # Sampling during training
    ap.add_argument("--add_bos_to_prompts", action="store_true")
    ap.add_argument("--sample_temperature", type=float, default=0.7)
    ap.add_argument("--sample_top_p", type=float, default=0.9)
    ap.add_argument("--sample_top_k", type=int, default=0)
    ap.add_argument("--sample_max_new_tokens", type=int, default=256)
    ap.add_argument("--sample_min_new_tokens", type=int, default=32)

    # stop strings for sampling (can be passed multiple times)
    ap.add_argument(
        "--stop_string",
        action="append",
        default=[],
        help="Stop generation when ANY of these strings appears (can be repeated).",
    )
    ap.add_argument(
        "--strip_stop",
        action="store_true",
        help="If set, remove the stop string from the printed sample output.",
    )

    # Resume
    ap.add_argument("--resume_path", type=str, default="")
    ap.add_argument("--resume_full", action="store_true")
    ap.add_argument("--resume_step", type=int, default=-1, help="If resume_path is a directory, load step_XXXXXX.pt. -1: auto (prefer latest.pt)")

    # torch.compile
    ap.add_argument("--compile", action="store_true")

    return ap.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
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

    # vocab_size from tokenizer.json
    try:
        inferred_vs = infer_vocab_size_from_tokenizer_json(str(tok_path))
    except Exception as e:
        print(f"[warn] failed to infer vocab_size from tokenizer.json: {e}")
        inferred_vs = int(args.vocab_size)

    if inferred_vs != int(args.vocab_size):
        print(f"[info] override vocab_size: args={args.vocab_size} -> tokenizer={inferred_vs}")

    cfg = GPTConfig(
        vocab_size=int(inferred_vs),
        n_layers=int(args.layers),
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        d_ff=int(args.d_ff),
        max_seq_len=int(args.seq_len),
        dropout=float(args.dropout),
        tie_embeddings=True,
    )

    model = GPT(cfg).to(device)

    use_fp16 = args.precision == "fp16"
    ac_dtype = _autocast_dtype(args.precision)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        betas=(0.9, 0.95),
    )

    # Data
    train_ds = PackedBinDataset(
        str(train_dir),
        seq_len=int(args.seq_len),
        bos_id=int(args.bos_id),
        eos_id=int(args.eos_id),
        mask_bos_in_loss=not bool(args.no_mask_bos_in_loss),
        mask_last_label_in_loss=not bool(args.no_mask_last_label_in_loss),
    )
    val_ds = PackedBinDataset(
        str(val_dir),
        seq_len=int(args.seq_len),
        bos_id=int(args.bos_id),
        eos_id=int(args.eos_id),
        mask_bos_in_loss=not bool(args.no_mask_bos_in_loss),
        mask_last_label_in_loss=not bool(args.no_mask_last_label_in_loss),
    )

    # dataset stats (once, before dataloaders)
    _estimate_dataset_stats(train_ds, int(args.seq_len), name="train")
    _estimate_dataset_stats(val_ds, int(args.seq_len), name="val")
    print(f"[*] model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    train_dl = DataLoader(
        train_ds,
        batch_size=int(args.micro_bsz),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=int(args.micro_bsz),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
    )

    # Resume
    global_step = 0
    local_step = 0
    resolved = None
    if args.resume_path:
        resolved = _resolve_resume_path(
            args.resume_path, out_dir=out_dir, resume_step=int(args.resume_step)
        )

    if resolved is None:
        if args.resume_path:
            print(f"[resume] WARNING: could not resolve resume_path={args.resume_path!r} "
                f"(resume_step={args.resume_step}). Starting from scratch.")
    else:
        global_step, local_step = load_ckpt(
            resume_path=resolved,
            model=model,
            optim=optim,
            scaler=scaler if use_fp16 else None,
            resume_full=bool(args.resume_full),
        )
        print(f"[resume] loaded {resolved} (global_step={global_step}, local_step={local_step})")

    # Compile
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
    window_sup_tokens_est = 0  # estimated supervised tokens (avoid GPU sync)

    while local_step < int(args.max_steps):
        lr = lr_schedule(global_step, int(args.warmup_steps), float(args.lr))
        for pg in optim.param_groups:
            pg["lr"] = lr

        # Stop requested via Ctrl+C? Save and exit cleanly.
        if _STOP_REQUESTED:
            print("[interrupt] Saving checkpoint and exiting...")
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
            print(f"[interrupt] saved latest + step_{global_step:06d}.pt to {out_dir}")
            return

        optim.zero_grad(set_to_none=True)

        # EOS weight schedule:
        # If eos_weight_warmup_steps > 0, linearly anneal eos_weight -> 1.0 over that many steps.
        # (Use eos_weight < 1.0 to down-weight EOS early and avoid EOS-collapse.)
        cur_eos_weight = float(args.eos_weight)
        warm = int(args.eos_weight_warmup_steps)
        if warm > 0 and cur_eos_weight != 1.0:
            if global_step >= warm:
                cur_eos_weight = 1.0
            else:
                t = float(global_step) / float(warm)
                cur_eos_weight = cur_eos_weight + t * (1.0 - cur_eos_weight)

        accum_loss_raw = 0.0

        for micro in range(int(args.grad_accum)):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_dl)
                batch = next(data_iter)

            if len(batch) == 2:
                input_u16, labels_u16 = batch
                loss_mask_cpu = torch.ones_like(labels_u16, dtype=torch.float32)
            else:
                input_u16, labels_u16, loss_mask_cpu = batch

            # estimated supervised tokens (cheap)
            window_sup_tokens_est += int(loss_mask_cpu.float().sum().item())

            input_ids = input_u16.to(device, dtype=torch.long, non_blocking=True)
            labels = labels_u16.to(device, dtype=torch.long, non_blocking=True)
            loss_mask = loss_mask_cpu.to(device, dtype=torch.float32, non_blocking=True)

            # forward + loss
            if ac_dtype is not None:
                with torch.autocast("cuda", dtype=ac_dtype):
                    logits = model(input_ids)
                    loss_raw = masked_weighted_ce_loss(
                        logits=logits,
                        labels=labels,
                        loss_mask=loss_mask,
                        eos_id=int(args.eos_id),
                        eos_weight=float(cur_eos_weight),
                    )
                    loss = loss_raw / float(args.grad_accum)
            else:
                logits = model(input_ids)
                loss_raw = masked_weighted_ce_loss(
                    logits=logits,
                    labels=labels,
                    loss_mask=loss_mask,
                    eos_id=int(args.eos_id),
                    eos_weight=float(cur_eos_weight),
                )
                loss = loss_raw / float(args.grad_accum)

            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss_raw += float(loss_raw.detach().item())

            # Debug (once per debug interval, on first microbatch)
            if (global_step % int(args.debug_every) == 0) and (micro == 0):
                lm = float(logits.float().mean().item())
                ls = float(logits.float().std().item())
                mce = masked_ce_128_debug(logits, labels, loss_mask, tt=128)

                # How much supervision is active?
                m = loss_mask
                eos_frac = float((((labels == int(args.eos_id)) & (m > 0)).float().sum().item()) / m.sum().clamp_min(1.0).item())

                # masked top-1 acc (rough)
                pred = logits.argmax(dim=-1)
                hit = ((pred == labels) & (m > 0)).sum().item()
                tot = (m > 0).sum().item()
                top1 = float(hit / max(1.0, tot))

                print(f"[dbg] step={global_step} logits_mean={lm:.4f} logits_std={ls:.4f} masked_ce_128={mce:.6f}")
                print(f"[dbg] mask_mean={float(m.mean().item()):.6f} mask_sum={float(m.sum().item()):.1f}")
                print(f"[dbg] labels min/max: {int(labels.min().item())} {int(labels.max().item())}")
                print(f"[dbg] eos_frac_supervised: {eos_frac:.6f}")
                print(f"[dbg] masked_top1_acc: {top1:.6f}")

                ce_nomask = F.cross_entropy(
                    logits[:, :128, :].reshape(-1, logits.shape[-1]).float(),
                    labels[:, :128].reshape(-1),
                    reduction="mean",
                )
                print(f"[dbg] ce_nomask_128: {float(ce_nomask.item()):.6f}")

                # critical correctness checks
                _ = causal_leak_check(model, input_ids, device, vocab_size=cfg.vocab_size, check_pos=128, delta_pos=8)
                _ = label_shift_sanity(input_ids, labels, loss_mask)

        # grad clip
        if float(args.grad_clip) > 0:
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

        if _STOP_REQUESTED:
            print("[interrupt] Saving checkpoint and exiting...")
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
            print(f"[interrupt] saved latest + step_{global_step:06d}.pt to {out_dir}")
            return

        # Logging
        if global_step % int(args.log_every) == 0:
            dt = time.time() - t_window
            tok_s = window_sup_tokens_est / max(dt, 1e-6)
            mean_loss_raw = accum_loss_raw / max(1, int(args.grad_accum))
            print(
                f"[train] step={global_step} loss={mean_loss_raw:.4f} "
                f"(eos_w={cur_eos_weight:g}) lr={lr:.2e} tok/s={tok_s:.0f}"
            )
            t_window = time.time()
            window_sup_tokens_est = 0

        # Eval + samples
        if global_step % int(args.eval_every) == 0:
            val_loss = evaluate(
                model=model,
                dl=val_dl,
                device=device,
                precision=args.precision,
                eos_id=int(args.eos_id),
                eos_weight=float(cur_eos_weight),
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
                    max_seq_len=int(args.seq_len),
                    precision=args.precision,
                    out_path=out_path,
                    temperature=float(args.sample_temperature),
                    top_p=float(args.sample_top_p),
                    top_k=int(args.sample_top_k),
                    max_new_tokens=int(args.sample_max_new_tokens),
                    min_new_tokens=int(args.sample_min_new_tokens),
                    eos_id=int(args.eos_id),
                    add_bos=bool(args.add_bos_to_prompts),
                    bos_id=int(args.bos_id),
                    greedy=False,
                    debug=True,
                    stop_strings=(args.stop_string if len(args.stop_string) > 0 else None),
                    strip_stop=args.strip_stop,
                )
                print(f"[sample] wrote {out_path}")
            except Exception as e:
                print(f"[sample] failed: {e}")

        # Save checkpoints
        if global_step % int(args.save_every) == 0:
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
