# pretrain/train_pretrain.py
# Minimal, robust GPT pretraining script for petitgpt.
# Key features:
# - Causal LM next-token objective with explicit token-level loss_mask
# - Loss is computed in FP32 for numerical stability (even under bf16/fp16)
# - Optional EOS down-weighting early to prevent EOS collapse
# - Gradient accumulation, bf16/fp16 autocast, optional torch.compile
# - Atomic checkpoints (avoid partial writes) + periodic eval + sample generation
# - Strong debug sanity checks (shift correctness + causal leakage checks)

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
from sample import generate_default_samples  # noqa: E402
from src.model import GPT, GPTConfig  # noqa: E402

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


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _resolve_path(p: str) -> str:
    path = Path(p)
    if path.exists():
        return str(path)
    alt = PROJECT_ROOT / p
    if alt.exists():
        return str(alt)
    return str(path)


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_schedule(step: int, warmup_steps: int, base_lr: float) -> float:
    """Linear warmup then constant LR (stable baseline)."""
    if warmup_steps <= 0:
        return base_lr
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


def _autocast_dtype(precision: str) -> Optional[torch.dtype]:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return None


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
        return max(t.get("id", -1) for t in added) + 1

    raise ValueError(f"Cannot infer vocab_size from tokenizer.json: {path}")


# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------
def masked_weighted_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    eos_id: int,
    eos_weight: float,
) -> torch.Tensor:
    """
    Token-level CE:
      - per-token CE (reduction='none') computed in FP32
      - multiply by loss_mask (float32)
      - optionally down-weight EOS targets
      - normalize by sum of weights (NOT by B*T)
    """
    B, T, V = logits.shape
    l = logits.reshape(B * T, V).float()  # FP32 loss math
    y = labels.reshape(B * T)
    m = loss_mask.reshape(B * T).float()

    per = F.cross_entropy(l, y, reduction="none")  # [B*T], FP32
    w = m
    if eos_weight != 1.0:
        eos_m = (y == int(eos_id)).float()
        w = w * (1.0 + eos_m * (float(eos_weight) - 1.0))

    denom = w.sum().clamp_min(1.0)
    return (per * w).sum() / denom


@torch.no_grad()
def masked_ce_128_debug(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    tt: int = 128,
) -> float:
    """Same loss "shape" as training loss (mask-weighted), but only first tt tokens."""
    B, T, V = logits.shape
    t = min(T, tt)
    l = logits[:, :t, :].reshape(-1, V).float()
    y = labels[:, :t].reshape(-1)
    m = loss_mask[:, :t].reshape(-1).float()
    per = F.cross_entropy(l, y, reduction="none")
    return float((per * m).sum().item() / m.sum().clamp_min(1.0).item())


# -----------------------------------------------------------------------------
# Debug sanity checks
# -----------------------------------------------------------------------------
@torch.no_grad()
def causal_leak_check(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    device: torch.device,
    vocab_size: int,
    check_pos: int = 128,
    perturb_pos: Optional[int] = None,
) -> float:
    """
    Check that changing a token at position `perturb_pos` does NOT change logits
    for positions strictly before `check_pos`.

    - For a correct causal transformer: logits[:, :check_pos, :] are a function
      of input_ids[:, :check_pos], so perturbing at position >= check_pos should
      not affect that prefix.
    - To catch *local* off-by-one leakage (t can see t+1), we default:
        perturb_pos = min(T-1, check_pos)
      and compare logits up to check_pos-1.
    """
    model.eval()
    x = input_ids.to(device)
    T = x.shape[1]
    cp = int(min(max(1, check_pos), T - 1))  # compare up to cp-1
    if perturb_pos is None:
        perturb_pos = cp
    p = int(min(max(0, perturb_pos), T - 1))

    logits1 = model(x).float()

    x2 = x.clone()
    x2[:, p] = (x2[:, p] + 123) % int(vocab_size)
    logits2 = model(x2).float()

    diff = (logits1[:, :cp, :] - logits2[:, :cp, :]).abs().max().item()
    print(f"[dbg] causal_leak_check pos={p} prefix<{cp} max_abs_diff={diff:.6f}")

    model.train()
    return float(diff)

@torch.no_grad()
def local_future_leak_check(model, input_ids, device, vocab_size: int, n_checks: int = 16):
    """
    Check whether logits at position i change when we perturb token at position i+1.
    If they do (beyond tiny numeric noise), you have a 1-step future leak.
    """
    model.eval()
    x = input_ids.to(device)
    B, T = x.shape
    logits1 = model(x).float()

    # pick some i positions away from edges
    # (avoid i near end; need i+1 valid)
    idxs = torch.linspace(8, T - 10, steps=n_checks).long().tolist()

    max_diff = 0.0
    for i in idxs:
        x2 = x.clone()
        # perturb token at i+1 only
        x2[:, i + 1] = (x2[:, i + 1] + 123) % vocab_size
        logits2 = model(x2).float()
        # compare ONLY position i
        diff = (logits1[:, i, :] - logits2[:, i, :]).abs().max().item()
        max_diff = max(max_diff, diff)

    print(f"[dbg] local_future_leak_check max_abs_diff={max_diff:.6f} (expect ~0)")
    model.train()
    return max_diff

@torch.no_grad()
def label_shift_sanity(input_ids: torch.Tensor, labels: torch.Tensor, loss_mask: torch.Tensor):
    """
    For a standard causal LM pack:
      labels[:, t] should equal input_ids[:, t+1] for supervised positions (except last).
    This checks a few positions.
    """
    B, T = input_ids.shape
    # compare on prefix to avoid boundary noise
    tmax = min(T - 2, 256)
    # positions where we have supervision and also t+1 valid
    m = (loss_mask[:, :tmax] > 0)
    if m.sum().item() == 0:
        print("[dbg] label_shift_sanity: no supervised tokens in range")
        return

    target = input_ids[:, 1:tmax+1]
    lab = labels[:, :tmax]
    ok = ((lab == target) & m).float().sum().item()
    tot = m.float().sum().item()
    print(f"[dbg] label_shift_sanity next-token match over supervised: {ok/tot:.6f}")

@torch.no_grad()
def shift_sanity_check(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
) -> Dict[str, float]:
    """
    Verify dataset alignment:
      - labels should be input_ids shifted left by 1 (next-token prediction)
      - labels should almost never equal input_ids at the same position
    """
    x = input_ids
    y = labels
    m = (loss_mask > 0).float()

    same_pos = ((x == y).float() * m).sum().item() / m.sum().clamp_min(1.0).item()

    if x.shape[1] > 1:
        m2 = (loss_mask[:, :-1] > 0).float()
        nxt = ((y[:, :-1] == x[:, 1:]).float() * m2).sum().item() / m2.sum().clamp_min(1.0).item()
    else:
        nxt = 0.0

    return {"same_pos_frac": float(same_pos), "next_token_match_frac": float(nxt)}


# -----------------------------------------------------------------------------
# Checkpoint I/O (atomic save)
# -----------------------------------------------------------------------------
def _atomic_torch_save(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


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
    out_dir.mkdir(parents=True, exist_ok=True)
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

    _atomic_torch_save(ckpt, out_dir / "latest.pt")
    _atomic_torch_save(ckpt, out_dir / f"step_{global_step:06d}.pt")


def load_ckpt(
    resume_path: Path,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    resume_full: bool,
) -> Tuple[int, int]:
    ckpt = torch.load(resume_path, map_location="cpu")
    state = ckpt["model"]

    if any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k[len("_orig_mod."):]: v for k, v in state.items()}

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
# Eval
# -----------------------------------------------------------------------------
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
        else:
            logits = model(input_ids)

        loss = masked_weighted_ce_loss(logits, labels, loss_mask, eos_id=eos_id, eos_weight=eos_weight)
        losses.append(float(loss.item()))

    model.train()
    return float(sum(losses) / max(1, len(losses)))


# -----------------------------------------------------------------------------
# Dataset stats (lightweight)
# -----------------------------------------------------------------------------
@torch.no_grad()
def print_dataset_stats(
    name: str,
    ds: PackedBinDataset,
    seq_len: int,
    micro_bsz: int,
    grad_accum: int,
    sample_blocks: int = 256,
) -> None:
    raw_tokens = int(getattr(ds, "total_tokens", 0))
    block = int(seq_len + 1)
    n_blocks = int(raw_tokens // block)
    epoch_tokens = int(n_blocks * seq_len)
    tokens_per_step = int(micro_bsz * grad_accum * seq_len)
    steps_per_epoch = int(epoch_tokens // max(1, tokens_per_step))

    n = min(len(ds), sample_blocks)
    sup_sum = 0.0
    eos_sum = 0.0
    for i in range(n):
        _, y, m = ds[i]
        m = m.float()
        sup_sum += float(m.sum().item())
        eos_sum += float(((y == ds.eos_id).float() * m).sum().item())

    avg_sup = sup_sum / max(1.0, float(n))
    eos_frac = eos_sum / max(1.0, sup_sum)

    print(f"[*] {name} dataset stats:")
    print(f"    - shards: {len(ds.shards)}")
    print(f"    - raw tokens in .bin (sum of shard lengths): {raw_tokens:,}")
    print(f"    - block size (seq_len+1): {block}")
    print(f"    - full blocks (n_blocks): {n_blocks:,}")
    print(f"    - epoch-equivalent tokens (n_blocks*seq_len): {epoch_tokens:,}")
    print(f"    - tokens per step (micro*accum*seq): {tokens_per_step:,}")
    print(f"    - steps per epoch (approx): {steps_per_epoch:,}")
    print(f"    - avg supervised tokens per block (sampled): {avg_sup:.1f} / {seq_len}")
    print(f"    - avg EOS fraction over supervised tokens (sampled): {eos_frac:.4f}")


# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)

    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--samples_dir", required=True)
    ap.add_argument("--tokenizer_path", required=True)

    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--n_heads", type=int, default=12)
    ap.add_argument("--d_ff", type=int, default=3072)
    ap.add_argument("--dropout", type=float, default=0.0)

    ap.add_argument("--bos_id", type=int, default=2)
    ap.add_argument("--eos_id", type=int, default=3)

    ap.add_argument("--no_mask_bos_in_loss", action="store_true")
    ap.add_argument("--no_mask_last_label_in_loss", action="store_true")
    ap.add_argument("--eos_weight", type=float, default=1.0)
    ap.add_argument("--eos_weight_warmup_steps", type=int, default=0)

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

    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--debug_every", type=int, default=500)

    ap.add_argument("--debug_shift_check", action="store_true")
    ap.add_argument("--debug_causal_check", action="store_true")
    ap.add_argument("--debug_causal_pos", type=int, default=512)
    ap.add_argument("--debug_causal_prefix", type=int, default=512)

    ap.add_argument("--add_bos_to_prompts", action="store_true")
    ap.add_argument("--sample_temperature", type=float, default=0.7)
    ap.add_argument("--sample_top_p", type=float, default=0.9)
    ap.add_argument("--sample_top_k", type=int, default=0)
    ap.add_argument("--sample_max_new_tokens", type=int, default=256)
    ap.add_argument("--sample_min_new_tokens", type=int, default=32)

    ap.add_argument("--resume_path", type=str, default="")
    ap.add_argument("--resume_full", action="store_true")

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

    inferred_vs = args.vocab_size
    try:
        inferred_vs = infer_vocab_size_from_tokenizer_json(str(tok_path))
        if inferred_vs != args.vocab_size:
            print(f"[info] override vocab_size: args={args.vocab_size} -> tokenizer={inferred_vs}")
    except Exception as e:
        print(f"[warn] failed to infer vocab_size from tokenizer.json: {e}")

    cfg = GPTConfig(
        vocab_size=inferred_vs,
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
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    train_ds = PackedBinDataset(
        str(train_dir),
        seq_len=args.seq_len,
        bos_id=args.bos_id,
        eos_id=args.eos_id,
        mask_bos_in_loss=not args.no_mask_bos_in_loss,
        mask_last_label_in_loss=not args.no_mask_last_label_in_loss,
    )
    val_ds = PackedBinDataset(
        str(val_dir),
        seq_len=args.seq_len,
        bos_id=args.bos_id,
        eos_id=args.eos_id,
        mask_bos_in_loss=not args.no_mask_bos_in_loss,
        mask_last_label_in_loss=not args.no_mask_last_label_in_loss,
    )

    print_dataset_stats("train", train_ds, args.seq_len, args.micro_bsz, args.grad_accum)
    print_dataset_stats("val", val_ds, args.seq_len, args.micro_bsz, args.grad_accum)
    print(f"[*] model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

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

    if args.compile:
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
            print("[compile] torch.compile enabled")
        except Exception as e:
            print(f"[compile] torch.compile failed: {e}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps({**vars(args), "model_cfg": asdict(cfg)}, indent=2),
        encoding="utf-8",
    )

    model.train()
    data_iter = iter(train_dl)

    t_window = time.time()
    window_sup_tokens_est = 0

    while local_step < args.max_steps:
        lr = lr_schedule(global_step, args.warmup_steps, args.lr)
        for pg in optim.param_groups:
            pg["lr"] = lr

        cur_eos_weight = float(args.eos_weight)
        if args.eos_weight_warmup_steps and global_step >= int(args.eos_weight_warmup_steps):
            cur_eos_weight = 1.0

        optim.zero_grad(set_to_none=True)
        accum_loss_raw = 0.0

        for micro in range(args.grad_accum):
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

            window_sup_tokens_est += int(loss_mask_cpu.sum().item())

            input_ids = input_u16.to(device, dtype=torch.long, non_blocking=True)
            labels = labels_u16.to(device, dtype=torch.long, non_blocking=True)
            loss_mask = loss_mask_cpu.to(device, dtype=torch.float32, non_blocking=True)

            if ac_dtype is not None:
                with torch.autocast("cuda", dtype=ac_dtype):
                    logits = model(input_ids)
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

            accum_loss_raw += float(loss_raw.detach().item())

            if (global_step % args.debug_every == 0) and (micro == 0):
                lm = float(logits.float().mean().item())
                ls = float(logits.float().std().item())
                mce = masked_ce_128_debug(logits, labels, loss_mask, tt=128)

                m = loss_mask
                y = labels
                pred = logits.argmax(dim=-1)
                hit = ((pred == y) & (m > 0)).sum().item()
                tot = (m > 0).sum().item()
                top1 = float(hit / max(1.0, tot))

                eos_frac = float((((y == args.eos_id).float() * (m > 0).float()).sum().item()) / max(1.0, tot))

                B, T, V = logits.shape
                tt = min(T, 128)
                ce_nomask = F.cross_entropy(
                    logits[:, :tt, :].reshape(-1, V).float(),
                    y[:, :tt].reshape(-1),
                    reduction="mean",
                )

                print(f"[dbg] step={global_step} logits_mean={lm:.4f} logits_std={ls:.4f} masked_ce_128={mce:.6f}")
                print(f"[dbg] mask_mean={float(m.mean().item()):.6f} mask_sum={float(m.sum().item()):.1f}")
                print(f"[dbg] labels min/max: {int(y.min().item())} {int(y.max().item())}")
                print(f"[dbg] eos_frac_supervised: {eos_frac:.6f}")
                print(f"[dbg] masked_top1_acc: {top1:.6f}")
                print(f"[dbg] ce_nomask_128: {float(ce_nomask.item()):.6f}")

                if args.debug_shift_check:
                    s = shift_sanity_check(input_ids, labels, loss_mask)
                    print(f"[dbg] shift_check same_pos_frac={s['same_pos_frac']:.6f} next_token_match_frac={s['next_token_match_frac']:.6f}")

                if args.debug_causal_check:
                    _ = causal_leak_check(
                        model=model,
                        input_ids=input_ids,
                        device=device,
                        vocab_size=cfg.vocab_size,
                        check_pos=int(args.debug_causal_prefix),
                        perturb_pos=int(args.debug_causal_pos),
                    )

                _ = local_future_leak_check(model, input_ids, device, vocab_size=cfg.vocab_size, n_checks=16)

                label_shift_sanity(input_ids, labels, loss_mask)

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
            dt = time.time() - t_window
            tok_s = window_sup_tokens_est / max(dt, 1e-6)
            mean_loss_raw = accum_loss_raw / max(1, args.grad_accum)
            print(
                f"[train] step={global_step} loss={mean_loss_raw:.4f} "
                f"(eos_w={cur_eos_weight:g}) lr={lr:.2e} tok/s={tok_s:.0f}"
            )
            t_window = time.time()
            window_sup_tokens_est = 0

        if global_step % args.eval_every == 0:
            val_loss = evaluate(
                model=model,
                dl=val_dl,
                device=device,
                precision=args.precision,
                eos_id=args.eos_id,
                eos_weight=cur_eos_weight,
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

        if global_step % args.save_every == 0:
            try:
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
            except Exception as e:
                print(f"[ckpt] save failed: {e}")

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
