#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SFT trainer for canonical chat messages.

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
- Can init from your pretrain ckpt (keys: model/optim/scaler/global_step/config/...).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import GPT, GPTConfig


# -------------------------
# Chat template (NO BOS/EOS)
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

    i = 0
    if messages and messages[0].get("role") == "system":
        sys_txt = clean_text(messages[0].get("content", ""))
        if sys_txt:
            segs.append((SYS_OPEN, False))
            segs.append((sys_txt, False))
            segs.append((SYS_CLOSE, False))
        i = 1

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
            segs.append((txt, True))   # supervise ONLY assistant content
            segs.append((ASSIST_CLOSE, False))
        else:
            continue

    return segs


# -------------------------
# Dataset: jsonl offsets
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
# Build example: input_ids + labels (assistant-only)
# -------------------------
def build_example(
    ex: Dict[str, Any],
    tok: Tokenizer,
    seq_len: int,
    pad_id: int,
    default_system: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    messages = ex.get("messages") or []
    if not messages:
        raise ValueError("missing messages")

    # inject default system if absent
    if messages[0].get("role") != "system" and default_system:
        messages = [{"role": "system", "content": default_system}] + messages

    segs = render_segments_and_supervision(messages)

    # Render full text (NO manual BOS/EOS)
    full_text = "".join(s for s, _ in segs)
    full_ids = tok.encode(full_text).ids  # tokenizer may auto add BOS/EOS once
    labels = [-100] * len(full_ids)

    # Find supervised spans by prefix re-encode (segment count is small, OK for SFT)
    prefix = ""
    prev_len = 0
    for seg_text, supervise in segs:
        prefix += seg_text
        cur_ids = tok.encode(prefix).ids
        cur_len = len(cur_ids)
        if supervise:
            for i in range(prev_len, cur_len):
                labels[i] = cur_ids[i]
        prev_len = cur_len

    # Truncate/pad to seq_len (keep tail so assistant targets more likely preserved)
    if len(full_ids) > seq_len:
        full_ids = full_ids[-seq_len:]
        labels = labels[-seq_len:]
    else:
        pad_n = seq_len - len(full_ids)
        full_ids = full_ids + [pad_id] * pad_n
        labels = labels + [-100] * pad_n

    return torch.tensor(full_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def collate_fn_builder(tok: Tokenizer, seq_len: int, pad_id: int, default_system: str, debug_first_batch: bool):
    printed = {"done": False}

    def collate(batch: List[Dict[str, Any]]):
        xs, ys = [], []
        for ex in batch:
            x, y = build_example(ex, tok, seq_len, pad_id, default_system)
            xs.append(x)
            ys.append(y)
        input_ids = torch.stack(xs, dim=0)  # (B,T)
        labels = torch.stack(ys, dim=0)     # (B,T)

        if debug_first_batch and not printed["done"]:
            printed["done"] = True
            sup = (labels[0] != -100).sum().item()
            tot = labels[0].numel()
            print(f"[dbg] supervised tokens(sample0): {sup}/{tot} ({sup/tot:.3f})")
            idx = (labels[0] != -100).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() > 0:
                dec = tok.decode(input_ids[0, idx].tolist())
                print("[dbg] decoded supervised span(first 300 chars):")
                print(dec[:300])
            else:
                print("[dbg] WARNING: no supervised tokens in sample0")

        return {"input_ids": input_ids, "labels": labels}

    return collate


# -------------------------
# Loss: next-token with ignore_index=-100
# -------------------------
def masked_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # logits: [B,T,V], labels: [B,T]
    V = logits.size(-1)
    logits = logits[:, :-1, :].contiguous().view(-1, V)
    labels = labels[:, 1:].contiguous().view(-1)
    return F.cross_entropy(logits, labels, ignore_index=-100)


def save_checkpoint_atomic(path: str, obj: Dict[str, Any]):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_ckpt(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")


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

    # init/resume
    ap.add_argument("--init_from_pretrain", default="", help="init model weights from pretrain ckpt (your keys: model/optim/scaler/global_step/config)")
    ap.add_argument("--resume", default="", help="resume SFT checkpoint (latest.pt or step_*.pt)")
    ap.add_argument("--default_system", default="You are a helpful assistant.")

    # model override (only used if not init_from_pretrain)
    ap.add_argument("--n_layers", type=int, default=12)
    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--n_heads", type=int, default=12)
    ap.add_argument("--d_ff", type=int, default=3072)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--tie_embeddings", action="store_true")  # if omitted, will be False; pretrain config may set True

    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--debug_first_batch", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    tok = Tokenizer.from_file(args.tokenizer_path)
    vocab_size = tok.get_vocab_size()

    # PAD id: if you didn't add [PAD], use 0
    pad_id = 0

    # -------------------------
    # Build model config
    # -------------------------
    cfg: Optional[GPTConfig] = None

    if args.init_from_pretrain:
        ck = load_ckpt(args.init_from_pretrain)
        cfg_dict = ck.get("config")
        if not isinstance(cfg_dict, dict):
            raise RuntimeError("pretrain ckpt missing 'config' dict")
        # Force vocab_size/max_seq_len to match current args/tokenizer if needed
        cfg_dict = dict(cfg_dict)
        cfg_dict["vocab_size"] = vocab_size
        cfg_dict["max_seq_len"] = args.seq_len
        cfg = GPTConfig(**cfg_dict)
        model = GPT(cfg).to(device)

        sd = ck.get("model")
        if sd is None:
            raise RuntimeError("pretrain ckpt missing 'model' state_dict")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[*] initialized from pretrain: {args.init_from_pretrain}")
        print(f"    missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
        if (len(missing) or len(unexpected)):
            print("    missing (first 20):", missing[:20])
            print("    unexpected (first 20):", unexpected[:20])
    else:
        # build from CLI flags
        cfg = GPTConfig(
            vocab_size=vocab_size,
            n_layers=args.n_layers,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            max_seq_len=args.seq_len,
            dropout=args.dropout,
            tie_embeddings=args.tie_embeddings,
        )
        model = GPT(cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Mixed precision
    use_fp16 = args.precision == "fp16" and device == "cuda"
    use_bf16 = args.precision == "bf16" and device == "cuda"
    autocast_dtype = torch.float16 if use_fp16 else (torch.bfloat16 if use_bf16 else None)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    # Resume SFT checkpoint (note: SFT ckpt format differs from pretrain ckpt)
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ck = load_ckpt(args.resume)
        sd = ck.get("model")
        if sd is None:
            raise RuntimeError("SFT resume ckpt missing 'model'")
        model.load_state_dict(sd, strict=False)

        opt = ck.get("optimizer") or ck.get("optim")
        if opt is not None:
            optimizer.load_state_dict(opt)

        # if resuming fp16 SFT, try restore scaler
        sc = ck.get("scaler")
        if sc is not None and use_fp16:
            scaler.load_state_dict(sc)

        start_step = int(ck.get("step", ck.get("global_step", 0)))
        print(f"[*] resumed SFT: {args.resume} at step={start_step}")

    # Data
    train_ds = JsonlOffsetsDataset(args.train_jsonl)
    val_ds = JsonlOffsetsDataset(args.val_jsonl)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.micro_bsz,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn_builder(tok, args.seq_len, pad_id, args.default_system, args.debug_first_batch),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.micro_bsz,
        shuffle=False,
        num_workers=max(0, args.num_workers // 2),
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn_builder(tok, args.seq_len, pad_id, args.default_system, False),
        drop_last=False,
    )

    # Cosine schedule
    def get_lr(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        t = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        t = min(max(t, 0.0), 1.0)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * t))

    model.train()
    t0 = time.time()
    running_loss = 0.0
    step = start_step
    train_iter = iter(train_loader)

    while step < args.max_steps:
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

            lr = get_lr(step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(autocast_dtype is not None)):
                logits = model(input_ids)  # your model forward is input_ids-only
                loss = masked_ce_loss(logits, labels) / args.grad_accum

            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            micro_loss += loss.item()

        if use_fp16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        step += 1
        running_loss += micro_loss

        if step % 50 == 0:
            dt = time.time() - t0
            print(f"[train] step={step} loss={running_loss/50:.4f} lr={get_lr(step):.2e} dt={dt:.1f}s")
            running_loss = 0.0
            t0 = time.time()

        if args.eval_every > 0 and step % args.eval_every == 0:
            model.eval()
            losses = []
            with torch.no_grad():
                for j, vb in enumerate(val_loader):
                    if j >= 50:
                        break
                    vi = vb["input_ids"].to(device)
                    vl = vb["labels"].to(device)
                    with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(autocast_dtype is not None)):
                        v_logits = model(vi)
                        v_loss = masked_ce_loss(v_logits, vl)
                    losses.append(v_loss.item())
            print(f"[eval] step={step} val_loss={sum(losses)/max(1,len(losses)):.4f}")
            model.train()

        if args.save_every > 0 and step % args.save_every == 0:
            ckpt_path = os.path.join(args.out_dir, f"step_{step:06d}.pt")
            ckpt = {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if use_fp16 else None,
                "cfg": asdict(cfg) if cfg is not None else None,
                "args": vars(args),
            }
            save_checkpoint_atomic(ckpt_path, ckpt)
            save_checkpoint_atomic(os.path.join(args.out_dir, "latest.pt"), ckpt)
            print(f"[ckpt] saved {ckpt_path}")

    print("[done]")


if __name__ == "__main__":
    main()
