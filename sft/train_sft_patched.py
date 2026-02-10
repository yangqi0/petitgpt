#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal, correct SFT trainer for canonical chat messages.

Input format (jsonl), one example per line:
{
  "messages": [
    {"role":"system","content":"..."},
    {"role":"user","content":"..."},
    {"role":"assistant","content":"..."},
    ...
  ],
  "meta": {...}  # optional
}

Key properties:
- DOES NOT manually add BOS/EOS (tokenizer may add via post-processor).
- Builds labels so that ONLY assistant content tokens are supervised; everything else is -100.
- Supports grad accumulation, fp16/bf16, checkpoint save/resume, eval, logging.

Model import is flexible:
- Tries to import from src.model:
  - GPT, GPTConfig (common)
  - or build_model(config)
  - or Model class named "Transformer"/"GPT" etc.
You may need to adjust the model construction part for your repo, but the SFT/data/masking logic is solid.

"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tokenizers import Tokenizer


# -------------------------
# Chat template (no BOS/EOS)
# -------------------------

SYS_OPEN = "[SYS]\n"
SYS_CLOSE = "\n[/SYS]\n"
USER_OPEN = "[USER]\n"
USER_CLOSE = "\n[/USER]\n"
ASSIST_OPEN = "[ASSISTANT]\n"
ASSIST_CLOSE = "\n[/ASSISTANT]\n"


def clean_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s.strip()


def render_segments_and_supervision(messages: List[Dict[str, str]]) -> List[Tuple[str, bool]]:
    """
    Return list of (text_segment, supervise) where supervise==True only for assistant *content*.
    Tags and system/user content are not supervised.
    """
    segs: List[Tuple[str, bool]] = []

    # system: take first system if present, else none
    i = 0
    if messages and messages[0].get("role") == "system":
        sys_txt = clean_text(messages[0].get("content", ""))
        segs.append((SYS_OPEN, False))
        segs.append((sys_txt, False))
        segs.append((SYS_CLOSE, False))
        i = 1

    # remaining turns
    for m in messages[i:]:
        role = (m.get("role") or "").strip().lower()
        txt = clean_text(m.get("content", ""))
        if not txt:
            continue
        if role == "user":
            segs.append((USER_OPEN, False))
            segs.append((txt, False))
            segs.append((USER_CLOSE, False))
        elif role == "assistant":
            segs.append((ASSIST_OPEN, False))
            # ONLY assistant content is supervised
            segs.append((txt, True))
            segs.append((ASSIST_CLOSE, False))
        else:
            # ignore unknown roles
            continue

    return segs


# ---------------------------------
# Tokenization helpers (BOS/EOS safe)
# ---------------------------------

@dataclass
class TokInfo:
    tok: Tokenizer
    bos_id: int
    eos_id: int
    pad_id: int


def load_tokenizer(path: str) -> TokInfo:
    tok = Tokenizer.from_file(path)
    # You told earlier bos_id=2 eos_id=3; but let's read if possible, fallback:
    # tokenizers doesn't always expose special ids; we take safe defaults.
    bos_id = 2
    eos_id = 3

    # pad: if you didn't add PAD token, use 0 (common)
    pad_id = 0
    return TokInfo(tok=tok, bos_id=bos_id, eos_id=eos_id, pad_id=pad_id)


def encode_piece(tok: Tokenizer, text: str) -> List[int]:
    """
    Encode a piece of text. We do NOT try to add/remove BOS/EOS here.
    We rely on single full-text encode below, so offsets are consistent.
    """
    return tok.encode(text).ids


def encode_full(tok: Tokenizer, text: str) -> List[int]:
    return tok.encode(text).ids


# -------------------------
# Canonical dataset (jsonl)
# -------------------------

class JsonlOffsetsDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        self.offsets: List[int] = []
        with open(path, "rb") as f:
            off = 0
            for line in f:
                self.offsets.append(off)
                off += len(line)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        off = self.offsets[idx]
        with open(self.path, "rb") as f:
            f.seek(off)
            line = f.readline().decode("utf-8")
        return json.loads(line)


# -------------------------
# Build (input_ids, labels)
# -------------------------

def build_example(
    ex: Dict[str, Any],
    tok_info: TokInfo,
    seq_len: int,
    default_system: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build one training example:
    - input_ids: (T,)
    - labels:    (T,) with -100 for non-supervised tokens
    - attn_mask: (T,)
    """
    messages = ex.get("messages") or []
    if not messages:
        raise ValueError("missing messages")

    # If no system present, inject one (not supervised)
    if messages[0].get("role") != "system" and default_system:
        messages = [{"role": "system", "content": default_system}] + messages

    segs = render_segments_and_supervision(messages)

    # Strategy for correct assistant-only mask without special-token double counting:
    # 1) Build FULL rendered text (no manual BOS/EOS).
    # 2) Encode full text once -> input_ids_full (tokenizer may add BOS/EOS once).
    # 3) To find supervised token positions, we re-encode prefix lengths incrementally,
    #    but only over segments (cheap: number of segments per chat is small).
    #
    # This avoids needing encode(add_special_tokens=False) which may vary across tokenizers.
    full_text = "".join(s for s, _ in segs)
    input_ids_full = encode_full(tok_info.tok, full_text)

    # labels initialized to -100
    labels_full = [-100] * len(input_ids_full)

    # For each supervised segment, compute its token span by prefix encoding.
    # We compute token index boundaries using "encode_full(prefix).ids" lengths.
    # NOTE: This is O(#segments) encodes per example; acceptable for SFT.
    prefix = ""
    prev_len = 0
    for seg_text, supervise in segs:
        prefix += seg_text
        cur_ids = encode_full(tok_info.tok, prefix)
        cur_len = len(cur_ids)

        # tokens contributed by this segment are [prev_len:cur_len)
        if supervise:
            for i in range(prev_len, cur_len):
                labels_full[i] = cur_ids[i]  # label is token itself (shift handled in loss)
        prev_len = cur_len

    # Truncate/pad to seq_len (keep the tail, so assistant targets are more likely preserved)
    if len(input_ids_full) > seq_len:
        input_ids_full = input_ids_full[-seq_len:]
        labels_full = labels_full[-seq_len:]
    else:
        pad_n = seq_len - len(input_ids_full)
        input_ids_full = input_ids_full + [tok_info.pad_id] * pad_n
        labels_full = labels_full + [-100] * pad_n

    attn_mask = [1 if tid != tok_info.pad_id else 0 for tid in input_ids_full]

    return (
        torch.tensor(input_ids_full, dtype=torch.long),
        torch.tensor(labels_full, dtype=torch.long),
        torch.tensor(attn_mask, dtype=torch.long),
    )


def collate_fn_builder(tok_info: TokInfo, seq_len: int, default_system: str, debug_first_batch: bool):
    printed = {"done": False}

    def collate(batch: List[Dict[str, Any]]):
        xs, ys, ms = [], [], []
        for ex in batch:
            x, y, m = build_example(ex, tok_info, seq_len, default_system)
            xs.append(x)
            ys.append(y)
            ms.append(m)

        input_ids = torch.stack(xs, dim=0)  # (B,T)
        labels = torch.stack(ys, dim=0)     # (B,T)
        attn_mask = torch.stack(ms, dim=0)  # (B,T)

        if debug_first_batch and (not printed["done"]):
            printed["done"] = True

            tok = tok_info.tok
            x0 = input_ids[0]   # (T,)
            y0 = labels[0]      # (T,)

            sup = (y0 != -100).nonzero(as_tuple=False).view(-1)
            sup_cnt = int(sup.numel())
            T = int(y0.numel())
            print(f"[dbg] supervised tokens: {sup_cnt}/{T} ({sup_cnt/float(T):.4f})")

            if sup_cnt == 0:
                print("[dbg][WARN] no supervised tokens found in first batch item! check your masking logic.")
            else:
                s0 = int(sup[0].item())
                s1 = int(sup[-1].item())
                mism = int((x0[sup] != y0[sup]).sum().item())
                print(f"[dbg] supervised span idx: [{s0}, {s1}]  label==input mismatches: {mism}")

                def _clip(a, b):
                    a = max(0, a)
                    b = min(T, b)
                    return a, b

                a, b = _clip(s0 - 32, s0 + 64)
                ctx_ids = x0[a:b].tolist()
                ctx_txt = tok.decode(ctx_ids)  # tokenizers Tokenizer.decode(list[int])
                print("=" * 80)
                print(f"[dbg] context around first supervised token (idx {s0}): window [{a}:{b}]")
                print(ctx_txt.replace("\n", "\\n"))
                print("=" * 80)

                a2, b2 = _clip(s0, s0 + 200)
                sup_txt = tok.decode(x0[a2:b2].tolist())
                print("[dbg] supervised decode head:")
                print(sup_txt.replace("\n", "\\n"))
                print("=" * 80)

                # shift sanity: effective supervised positions for logits are where labels[1:] != -100
                eff = (y0[1:] != -100).nonzero(as_tuple=False).view(-1) + 1
                eff_cnt = int(eff.numel())
                if eff_cnt == 0:
                    print("[dbg][WARN] after shift, no effective supervised positions! (supervised might start at idx 0 only?)")
                else:
                    e0 = int(eff[0].item())
                    prev_tok = tok.decode([int(x0[e0 - 1].item())]) if e0 - 1 >= 0 else "<BOS?>"
                    targ_tok = tok.decode([int(x0[e0].item())])
                    print(f"[dbg] effective supervised positions for logits: {eff_cnt} (first at idx {e0})")
                    print(f"[dbg] example boundary: at position {e0-1}, prev token={prev_tok!r} -> target token={targ_tok!r}")

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn_mask}

    return collate


# -------------------------
# Loss / training utilities
# -------------------------

def masked_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    logits: (B,T,V)
    labels: (B,T) with -100 mask
    We do next-token prediction: logits[:, :-1] vs labels[:, 1:].
    """
    B, T, V = logits.shape
    logits = logits[:, :-1, :].contiguous().view(-1, V)
    labels = labels[:, 1:].contiguous().view(-1)
    return F.cross_entropy(logits, labels, ignore_index=-100)


def save_checkpoint_atomic(path: str, obj: Dict[str, Any]):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def try_build_model(vocab_size: int, ckpt_path: Optional[str]):
    """
    Flexible model construction. You WILL likely need to adapt this to your src/model.py.
    """
    # 1) Try common nanoGPT-ish API
    try:
        from src.model import GPT, GPTConfig  # type: ignore
        # If ckpt exists, load config from it; else minimal config placeholder
        if ckpt_path and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            cfg_dict = ckpt.get("model_cfg") or ckpt.get("config") or ckpt.get("cfg")
            if cfg_dict is None:
                # fall back
                cfg = GPTConfig(vocab_size=vocab_size)
            else:
                cfg_dict = dict(cfg_dict)
                cfg_dict["vocab_size"] = vocab_size
                cfg = GPTConfig(**cfg_dict)
        else:
            cfg = GPTConfig(vocab_size=vocab_size)

        model = GPT(cfg)
        if ckpt_path and os.path.exists(ckpt_path):
            sd = ckpt.get("model") or ckpt.get("model_state_dict") or ckpt.get("state_dict")
            if sd:
                model.load_state_dict(sd, strict=False)
        return model
    except Exception:
        pass

    raise RuntimeError(
        "Could not construct model. Please edit try_build_model() to match your src/model.py API."
    )


# -------------------------
# Main training loop
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tokenizer_path", required=True)

    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--micro_bsz", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)

    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--max_steps", type=int, default=20000)
    ap.add_argument("--warmup_steps", type=int, default=500)

    ap.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="bf16")
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=2000)

    ap.add_argument("--pretrained_ckpt", default="", help="optional init ckpt from pretrain")
    ap.add_argument("--resume", default="", help="resume SFT training checkpoint")
    ap.add_argument("--default_system", default="You are a helpful assistant.")

    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--debug_first_batch", action="store_true")

    ap.add_argument("--overfit_steps", type=int, default=0,
                help="If >0, run an overfit test: repeatedly train on the first batch for N steps (4).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    tok_info = load_tokenizer(args.tokenizer_path)

    # Estimate vocab size from tokenizer (best-effort)
    try:
        vocab_size = tok_info.tok.get_vocab_size()
    except Exception:
        vocab_size = 32000

    # Build model (edit try_build_model if needed)
    init_ckpt = args.pretrained_ckpt if args.pretrained_ckpt else None
    model = try_build_model(vocab_size=vocab_size, ckpt_path=init_ckpt)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Optional resume
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        sd = ckpt.get("model") or ckpt.get("model_state_dict") or ckpt.get("state_dict")
        if sd:
            model.load_state_dict(sd, strict=False)
        opt = ckpt.get("optimizer")
        if opt:
            optimizer.load_state_dict(opt)
        start_step = int(ckpt.get("step", 0))
        print(f"[*] Resumed from {args.resume} at step={start_step}")

    # Data
    train_ds = JsonlOffsetsDataset(args.train_jsonl)
    val_ds = JsonlOffsetsDataset(args.val_jsonl)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.micro_bsz,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn_builder(tok_info, args.seq_len, args.default_system, args.debug_first_batch),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.micro_bsz,
        shuffle=False,
        num_workers=max(0, args.num_workers // 2),
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn_builder(tok_info, args.seq_len, args.default_system, False),
        drop_last=False,
    )

    # =========================
    # quick overfit test (debug)
    # =========================
    if args.overfit_steps > 0:
        print(f"[*] overfit test: training on the first batch for {args.overfit_steps} steps ...")
        model.train()

        batch = next(iter(train_loader))
        for k in batch:
            batch[k] = batch[k].to(device)

        for t in range(args.overfit_steps):
            optimizer.zero_grad(set_to_none=True)

            try:
                out = model(batch["input_ids"], attention_mask=batch["attention_mask"])
            except TypeError:
                out = model(batch["input_ids"])

            logits = out["logits"] if isinstance(out, dict) else (out.logits if hasattr(out, "logits") else out)
            loss = masked_ce_loss(logits, batch["labels"])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if (t + 1) % 20 == 0 or t == 0:
                print(f"[overfit] step {t+1}/{args.overfit_steps} loss={loss.item():.4f}")

    # Mixed precision
    use_fp16 = args.precision == "fp16" and device == "cuda"
    use_bf16 = args.precision == "bf16" and device == "cuda"
    autocast_dtype = torch.float16 if use_fp16 else (torch.bfloat16 if use_bf16 else None)
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    # Simple cosine schedule
    def get_lr(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        t = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        t = min(max(t, 0.0), 1.0)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * t))

    model.train()
    t0 = time.time()
    running_loss = 0.0
    steps_done = start_step

    train_iter = iter(train_loader)

    while steps_done < args.max_steps:
        optimizer.zero_grad(set_to_none=True)
        micro_loss = 0.0

        for _ in range(args.grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            attn_mask = batch["attention_mask"].to(device, non_blocking=True)

            lr = get_lr(steps_done)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(autocast_dtype is not None)):
                # Forward: adapt if your model signature differs
                try:
                    out = model(input_ids, attention_mask=attn_mask)
                except TypeError:
                    out = model(input_ids)
                logits = out["logits"] if isinstance(out, dict) else (out.logits if hasattr(out, "logits") else out)
                loss = masked_ce_loss(logits, labels) / args.grad_accum

            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            micro_loss += loss.item()

        # step
        if use_fp16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        steps_done += 1
        running_loss += micro_loss

        # log
        if steps_done % 50 == 0:
            dt = time.time() - t0
            print(f"[train] step={steps_done} loss={running_loss/50:.4f} lr={get_lr(steps_done):.2e} dt={dt:.1f}s")
            running_loss = 0.0
            t0 = time.time()

        # eval
        if args.eval_every > 0 and steps_done % args.eval_every == 0:
            model.eval()
            losses = []
            with torch.no_grad():
                for j, vb in enumerate(val_loader):
                    if j >= 50:  # cap eval cost
                        break
                    vi = vb["input_ids"].to(device)
                    vl = vb["labels"].to(device)
                    vm = vb["attention_mask"].to(device)
                    with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(autocast_dtype is not None)):
                        try:
                            out = model(vi, attention_mask=vm)
                        except TypeError:
                            out = model(vi)
                        logits = out["logits"] if isinstance(out, dict) else (out.logits if hasattr(out, "logits") else out)
                        vloss = masked_ce_loss(logits, vl)
                    losses.append(vloss.item())
            print(f"[eval] step={steps_done} val_loss={sum(losses)/max(1,len(losses)):.4f}")
            model.train()

        # save
        if args.save_every > 0 and steps_done % args.save_every == 0:
            ckpt_path = os.path.join(args.out_dir, f"step_{steps_done:06d}.pt")
            ckpt = {
                "step": steps_done,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "vocab_size": vocab_size,
            }
            save_checkpoint_atomic(ckpt_path, ckpt)
            # update latest
            save_checkpoint_atomic(os.path.join(args.out_dir, "latest.pt"), ckpt)
            print(f"[ckpt] saved {ckpt_path}")

    print("[done]")


if __name__ == "__main__":
    main()
