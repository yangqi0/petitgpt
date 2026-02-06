# pretrain/train_pretrain.py
# Minimal, robust GPT pretraining script for petitgpt.
# Key properties:
# - MiniMind-style token-level loss_mask (explicit mask and weighted reduction)
# - BOS masked from loss; EOS can be down-weighted early to prevent collapse
# - IMPORTANT: cross-entropy is always computed in FP32 (even under bf16/fp16 autocast)
# - grad accumulation, bf16/fp16 autocast, torch.compile, clean checkpoints
# - periodic evaluation + sample generation + checkpoints

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

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


def infer_vocab_size_from_tokenizer_json(path: str) -> int:
    """Infer vocab size from HF tokenizers' tokenizer.json."""
    with open(path, encoding="utf-8") as f:
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


def _autocast_dtype(precision: str) -> torch.dtype | None:
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
    Token-level CE (MiniMind style):
      - per-token CE (reduction='none')
      - multiply by loss_mask
      - optionally down-weight EOS targets
      - normalize by sum of weights (NOT by B*T)

    IMPORTANT:
      - Cross-entropy is always computed in FP32 for numerical stability.
        This prevents bf16/fp16 autocast from "shrinking" CE and giving fake near-zero loss.
    """
    B, T, V = logits.shape
    l = logits.reshape(B * T, V).float()          # force FP32 CE
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
    """
    Debug-only masked CE over the first tt positions, computed in FP32.
    (This matches the "mask mouth" of training loss, but does NOT include eos_weight.)
    """
    B, T, V = logits.shape
    t = min(T, tt)
    l = logits[:, :t, :].reshape(-1, V).float()
    y = labels[:, :t].reshape(-1)
    m = loss_mask[:, :t].reshape(-1).float()
    per = F.cross_entropy(l, y, reduction="none")
    denom = m.sum().clamp_min(1.0)
    return float((per * m).sum().item() / denom.item())


@torch.no_grad()
def causal_leak_check(
    model,
    input_ids: torch.Tensor,
    device: torch.device,
    vocab_size: int,
    check_pos: int = 128,
) -> float:
    """
    If the model is causal, changing a future token should NOT change logits of earlier positions.
    We check that logits[:, :check_pos] are invariant to edits in the suffix.
    """
    model.eval()
    x = input_ids.to(device)
    logits1 = model(x).float()

    x2 = x.clone()
    t = x2.shape[1] - 5
    x2[:, t] = (x2[:, t] + 123) % int(vocab_size)
    logits2 = model(x2).float()

    diff = (logits1[:, :check_pos, :] - logits2[:, :check_pos, :]).abs().max().item()
    print(f"[dbg] causal_leak_check max_abs_diff(prefix_logits)={diff:.6f}")
    model.train()
    return float(diff)


def save_ckpt(
    out_dir: Path,
    global_step: int,
    local_step: int,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    model_config: dict,
    train_args: dict,
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

    torch.save(ckpt, out_dir / "latest.pt")
    torch.save(ckpt, out_dir / f"step_{global_step:06d}.pt")


def load_ckpt(
    resume_path: Path,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    resume_full: bool,
) -> tuple[int, int]:
    ckpt = torch.load(resume_path, map_location="cpu")
    state = ckpt["model"]

    # Handle compiled prefix if present
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

        loss = masked_weighted_ce_loss(
            logits, labels, loss_mask, eos_id=eos_id, eos_weight=eos_weight
        )
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
    ap.add_argument("--no_mask_bos_in_loss", action="store_true")
    ap.add_argument("--no_mask_last_label_in_loss", action="store_true")
    ap.add_argument("--eos_weight", type=float, default=0.2)
    ap.add_argument("--eos_weight_warmup_steps", type=int, default=0)

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
    ap.add_argument("--debug_every", type=int, default=500)

    # Sampling during training
    ap.add_argument("--add_bos_to_prompts", action="store_true")
    ap.add_argument("--sample_temperature", type=float, default=0.7)
    ap.add_argument("--sample_top_p", type=float, default=0.9)
    ap.add_argument("--sample_top_k", type=int, default=0)
    ap.add_argument("--sample_max_new_tokens", type=int, default=256)
    ap.add_argument("--sample_min_new_tokens", type=int, default=32)

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

    # vocab_size from tokenizer.json
    try:
        inferred_vs = infer_vocab_size_from_tokenizer_json(str(tok_path))
    except Exception as e:
        print(f"[warn] failed to infer vocab_size from tokenizer.json: {e}")
        inferred_vs = args.vocab_size
    if inferred_vs != args.vocab_size:
        print(f"[info] override vocab_size: args={args.vocab_size} -> tokenizer={inferred_vs}")

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

    # Data
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

    # === Dataset statistics (for PackedBinDataset) ===
    def _dataset_stats(
        ds: PackedBinDataset,
        name: str,
        seq_len: int,
        micro_bsz: int,
        grad_accum: int,
        sample_blocks: int = 8,
    ):
        block = seq_len + 1

        raw_tokens = int(sum(getattr(ds, "_lens", []))) if hasattr(ds, "_lens") else 0
        n_blocks = int(getattr(ds, "n_blocks", len(ds)))
        epoch_tokens = n_blocks * seq_len

        tokens_per_step = micro_bsz * grad_accum * seq_len
        steps_per_epoch = epoch_tokens // max(1, tokens_per_step)

        print(f"[*] {name} dataset stats:")
        if hasattr(ds, "shards"):
            try:
                print(f"    - shards: {len(ds.shards)}")
            except Exception:
                pass
        if raw_tokens > 0:
            print(f"    - raw tokens in .bin (sum of shard lengths): {raw_tokens:,}")
            print(f"    - block size (seq_len+1): {block}")
        print(f"    - full blocks (n_blocks): {n_blocks:,}")
        print(f"    - epoch-equivalent tokens (n_blocks*seq_len): {epoch_tokens:,}")
        print(f"    - tokens per step (micro*accum*seq): {tokens_per_step:,}")
        print(f"    - steps per epoch (approx): {steps_per_epoch:,}")

        # Sample a few blocks to estimate supervision density / EOS rate under the mask rules
        kmax = min(len(ds), max(1, sample_blocks))
        sup = []
        eos_frac = []
        eos_id = int(getattr(ds, "eos_id", 3))
        for k in range(kmax):
            x, y, m = ds[k]  # CPU
            mpos = (m > 0).float()
            sup_tok = float(mpos.sum().item())
            sup.append(sup_tok)
            if sup_tok > 0:
                eos_frac.append(float(((y == eos_id).float() * mpos).sum().item() / sup_tok))

        if sup:
            avg_sup = sum(sup) / len(sup)
            print(f"    - avg supervised tokens per block (sampled): {avg_sup:.1f} / {seq_len}")
            if avg_sup < 0.5 * seq_len:
                print("    [WARN] Supervised tokens per block is very low; check your loss_mask rules.")
        if eos_frac:
            avg_eos = sum(eos_frac) / len(eos_frac)
            print(f"    - avg EOS fraction over supervised tokens (sampled): {avg_eos:.4f}")

    _dataset_stats(train_ds, "train", args.seq_len, args.micro_bsz, args.grad_accum)
    _dataset_stats(val_ds, "val", args.seq_len, args.micro_bsz, args.grad_accum)

    n_params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[*] model params: {n_params_m:.2f}M")
    # === end dataset statistics ===

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

    # Resume
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
    window_sup_tokens_est = 0  # supervised tokens estimate from CPU mask (cheap)

    while local_step < args.max_steps:
        lr = lr_schedule(global_step, args.warmup_steps, args.lr)
        for pg in optim.param_groups:
            pg["lr"] = lr

        optim.zero_grad(set_to_none=True)

        # EOS weight schedule
        cur_eos_weight = float(args.eos_weight)
        if args.eos_weight_warmup_steps and global_step >= int(args.eos_weight_warmup_steps):
            cur_eos_weight = 1.0

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

            # CPU-side sanity check (first 50 steps, first microbatch)
            if global_step < 50 and micro == 0:
                tmp = input_u16.to(torch.int32)
                max_id = int(tmp.max().item())
                min_id = int(tmp.min().item())
                if min_id < 0 or max_id >= cfg.vocab_size:
                    print(
                        f"[fatal] token id out of range: min={min_id} max={max_id} vocab_size={cfg.vocab_size}"
                    )
                    print("[fatal] first row head:", tmp[0, :32].tolist())
                    raise RuntimeError("Token id out of range. Fix vocab_size/tokenizer/shards.")

            window_sup_tokens_est += int(loss_mask_cpu.sum().item())

            input_ids = input_u16.to(device, dtype=torch.long, non_blocking=True)
            labels = labels_u16.to(device, dtype=torch.long, non_blocking=True)
            loss_mask = loss_mask_cpu.to(device, dtype=torch.float32, non_blocking=True)

            # Forward under autocast, CE in FP32 inside masked_weighted_ce_loss()
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

            # Debug (once per interval, micro==0)
            if (global_step % args.debug_every == 0) and (micro == 0):
                lm = float(logits.float().mean().item())
                ls = float(logits.float().std().item())

                same = ((input_ids == labels) & (loss_mask > 0)).float().mean().item()
                print("[dbg] frac(input_ids==labels) over supervised:", float(same))

                mce = masked_ce_128_debug(logits, labels, loss_mask, tt=128)
                print(
                    f"[dbg] step={global_step} logits_mean={lm:.4f} logits_std={ls:.4f} masked_ce_128={mce:.6f}"
                )

                m = loss_mask
                print("[dbg] mask_mean:", float(m.mean().item()), "mask_sum:", float(m.sum().item()))

                y = labels
                print("[dbg] labels min/max:", int(y.min().item()), int(y.max().item()))

                eos = int(args.eos_id)
                eos_frac = (
                    ((y == eos) & (m > 0)).float().sum()
                    / (m > 0).float().sum().clamp_min(1)
                ).item()
                print(f"[dbg] eos_frac_supervised: {float(eos_frac):.6f}")

                pred = logits.argmax(dim=-1)
                hit = ((pred == y) & (m > 0)).sum().item()
                tot = (m > 0).sum().item()
                print("[dbg] masked_top1_acc:", float(hit / max(1.0, tot)))

                B, T, V = logits.shape
                tt = min(T, 128)
                ce_nomask = F.cross_entropy(
                    logits[:, :tt, :].reshape(-1, V).float(),
                    y[:, :tt].reshape(-1),
                    reduction="mean",
                )
                print("[dbg] ce_nomask_128:", float(ce_nomask.item()))

                _ = causal_leak_check(
                    model=model,
                    input_ids=input_ids,
                    device=device,
                    vocab_size=cfg.vocab_size,
                    check_pos=128,
                )

        # Gradient clipping
        if args.grad_clip and args.grad_clip > 0:
            if use_fp16:
                scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))

        # Optim step
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
            tok_s = window_sup_tokens_est / max(dt, 1e-6)
            mean_loss_raw = accum_loss_raw / max(1, args.grad_accum)
            print(
                f"[train] step={global_step} loss={mean_loss_raw:.4f} "
                f"(eos_w={cur_eos_weight:g}) lr={lr:.2e} tok/s={tok_s:.0f}"
            )
            t_window = time.time()
            window_sup_tokens_est = 0

        # Eval + samples
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

        # Milestone (optional)
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
