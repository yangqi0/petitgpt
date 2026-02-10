#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer

# Make imports work no matter where you run from
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import GPT, GPTConfig  # noqa: E402


# -------------------------
# Chat template (NO manual BOS/EOS text)
# -------------------------
SYS_OPEN = "[SYS]\n"
SYS_CLOSE = "\n[/SYS]\n"
USER_OPEN = "[USER]\n"
USER_CLOSE = "\n[/USER]\n"
ASSIST_OPEN = "[ASSISTANT]\n"
ASSIST_CLOSE = "\n[/ASSISTANT]\n"

BOS_ID = 2
EOS_ID = 3


def clean_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s.strip()


def render_segments_and_supervision(messages: List[Dict[str, str]]) -> List[Tuple[str, bool]]:
    """Return list of (text_segment, supervise) where supervise==True only for assistant content."""
    segs: List[Tuple[str, bool]] = []

    i = 0
    if messages and (messages[0].get("role") or "").strip().lower() == "system":
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
            segs.append((txt, True))
            segs.append((ASSIST_CLOSE, False))

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
# Encoding helpers (robust to tokenizer auto BOS/EOS)
# -------------------------
def encode_strip_special(tok: Tokenizer, text: str, bos_id: int, eos_id: int) -> List[int]:
    """Encode text and strip a leading BOS and trailing EOS if present."""
    ids = tok.encode(text).ids
    if ids and ids[0] == bos_id:
        ids = ids[1:]
    if ids and ids[-1] == eos_id:
        ids = ids[:-1]
    return ids


def encode_full_once(tok: Tokenizer, text: str, bos_id: int, eos_id: int) -> Tuple[bool, bool]:
    """Detect whether tokenizer auto-adds BOS/EOS for this text."""
    ids = tok.encode(text).ids
    has_bos = bool(ids) and ids[0] == bos_id
    has_eos = bool(ids) and ids[-1] == eos_id
    return has_bos, has_eos


# -------------------------
# Build example: input_ids + labels (assistant-only, exact spans)
# -------------------------
def build_example(
    ex: Dict[str, Any],
    tok: Tokenizer,
    seq_len: int,
    pad_id: int,
    default_system: str,
    bos_id: int,
    eos_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    messages = ex.get("messages") or []
    if not messages:
        raise ValueError("missing messages")

    if (messages[0].get("role") or "").strip().lower() != "system" and default_system:
        messages = [{"role": "system", "content": default_system}] + messages

    segs = render_segments_and_supervision(messages)

    # Figure out whether tokenizer auto-adds BOS/EOS (once)
    full_text = "".join(s for s, _ in segs)
    has_bos, has_eos = encode_full_once(tok, full_text, bos_id, eos_id)

    ids_all: List[int] = []
    labels_all: List[int] = []

    # Add exactly one BOS/EOS IF tokenizer would have added them for full text
    if has_bos:
        ids_all.append(bos_id)
        labels_all.append(-100)

    for seg_text, supervise in segs:
        seg_ids = encode_strip_special(tok, seg_text, bos_id, eos_id)
        ids_all.extend(seg_ids)
        if supervise:
            labels_all.extend(seg_ids)
        else:
            labels_all.extend([-100] * len(seg_ids))

    if has_eos:
        ids_all.append(eos_id)
        labels_all.append(-100)

    # Truncate/pad (keep tail so assistant targets more likely preserved)
    if len(ids_all) > seq_len:
        ids_all = ids_all[-seq_len:]
        labels_all = labels_all[-seq_len:]
    else:
        pad_n = seq_len - len(ids_all)
        ids_all = ids_all + [pad_id] * pad_n
        labels_all = labels_all + [-100] * pad_n

    return torch.tensor(ids_all, dtype=torch.long), torch.tensor(labels_all, dtype=torch.long)


def collate_fn_builder(tok: Tokenizer, seq_len: int, pad_id: int, default_system: str, debug_first_batch: bool):
    printed = {"done": False}

    def collate(batch: List[Dict[str, Any]]):
        xs, ys = [], []
        for ex in batch:
            x, y = build_example(ex, tok, seq_len, pad_id, default_system, BOS_ID, EOS_ID)
            xs.append(x)
            ys.append(y)
        input_ids = torch.stack(xs, dim=0)
        labels = torch.stack(ys, dim=0)

        if debug_first_batch and not printed["done"]:
            printed["done"] = True
            sup = int((labels[0] != -100).sum().item())
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


def masked_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
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

    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--max_steps", type=int, default=15000)
    ap.add_argument("--warmup_steps", type=int, default=300)

    ap.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="bf16")
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_batches", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=2000)

    ap.add_argument("--init_from_pretrain", default="")
    ap.add_argument("--resume", default="")
    ap.add_argument("--default_system", default="You are a helpful assistant.")

    ap.add_argument("--n_layers", type=int, default=12)
    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--n_heads", type=int, default=12)
    ap.add_argument("--d_ff", type=int, default=3072)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--tie_embeddings", action="store_true")

    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--debug_first_batch", action="store_true")

    # sampling
    ap.add_argument("--sample_every", type=int, default=1000)
    ap.add_argument("--samples_dir", type=str, default="")
    ap.add_argument("--sample_max_new_tokens", type=int, default=192)
    ap.add_argument("--sample_temperature", type=float, default=0.7)
    ap.add_argument("--sample_top_p", type=float, default=0.9)
    ap.add_argument("--sample_top_k", type=int, default=0)
    ap.add_argument("--sample_seed", type=int, default=1234)

    # in-domain sampling (val_jsonl)
    ap.add_argument("--sample_in_domain_n", type=int, default=10, help="sample N prompts from val_jsonl each sample_every")
    ap.add_argument("--sample_in_domain_seed", type=int, default=1234, help="rng seed for picking val examples")
    ap.add_argument("--sample_in_domain_show_ref", action="store_true", help="also print reference assistant for that val example")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    tok = Tokenizer.from_file(args.tokenizer_path)
    vocab_size = tok.get_vocab_size()
    pad_id = 0  # if no [PAD], use 0

    FIXED_PROMPTS = [
        "Explain Bayes' theorem in simple terms with a tiny example.",
        "Write a polite email to ask for an update on a job application.",
        "Summarize the following text in 3 bullet points: 'Deep learning has transformed many fields...'",
        "Give me a step-by-step plan to learn Python for data analysis in 4 weeks.",
        "What is gradient descent? Explain it like I'm 12.",
        "Solve: If P(A)=0.3, P(B)=0.5, and A and B are independent, what is P(A∩B)?",
        "Convert this to French: 'Thank you for your time and consideration.'",
        "Draft a short LinkedIn post announcing I built a small GPT model from scratch.",
        "What are 3 weekend trip ideas near Lille? Provide pros/cons.",
        "Given a list of numbers, explain how to compute mean and variance.",
    ]

    cfg: Optional[GPTConfig] = None

    if args.init_from_pretrain:
        ck = load_ckpt(args.init_from_pretrain)
        cfg_dict = ck.get("config")
        if not isinstance(cfg_dict, dict):
            raise RuntimeError("pretrain ckpt missing 'config' dict")
        cfg_dict = dict(cfg_dict)
        cfg_dict["vocab_size"] = vocab_size
        cfg_dict["max_seq_len"] = args.seq_len
        cfg = GPTConfig(**cfg_dict)
        model = GPT(cfg).to(device)

        sd = ck.get("model")
        if sd is None:
            raise RuntimeError("pretrain ckpt missing 'model'")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[*] initialized from pretrain: {args.init_from_pretrain}")
        print(f"    missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    else:
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

    use_fp16 = args.precision == "fp16" and device == "cuda"
    use_bf16 = args.precision == "bf16" and device == "cuda"
    autocast_dtype = torch.float16 if use_fp16 else (torch.bfloat16 if use_bf16 else None)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ck = load_ckpt(args.resume)
        sd = ck.get("model")
        if sd is None:
            raise RuntimeError("resume ckpt missing 'model'")
        model.load_state_dict(sd, strict=False)

        opt = ck.get("optimizer") or ck.get("optim")
        if opt is not None:
            optimizer.load_state_dict(opt)

        sc = ck.get("scaler")
        if sc is not None and use_fp16:
            scaler.load_state_dict(sc)

        start_step = int(ck.get("step", ck.get("global_step", 0)))
        print(f"[*] resumed: {args.resume} at step={start_step}")

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

    def get_lr(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        t = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        t = min(max(t, 0.0), 1.0)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * t))

    def build_chat_prompt(system: str, user: str) -> str:
        system = (system or "").strip()
        user = (user or "").strip()
        parts = []
        if system:
            parts += [SYS_OPEN, system, SYS_CLOSE]
        parts += [USER_OPEN, user, USER_CLOSE, ASSIST_OPEN]
        return "".join(parts)

    def extract_last_user_and_ref(messages: List[Dict[str, str]]) -> Tuple[str, str]:
        """Return (last_user_text, ref_assistant_text_after_it_if_any_else_last_assistant)."""
        last_user = ""
        ref = ""
        # find last user
        for m in reversed(messages):
            if (m.get("role") or "").strip().lower() == "user":
                last_user = clean_text(m.get("content", ""))
                break
        # find last assistant (best-effort)
        for m in reversed(messages):
            if (m.get("role") or "").strip().lower() == "assistant":
                ref = clean_text(m.get("content", ""))
                break
        return last_user, ref

    @torch.no_grad()
    def sample_once(user_text: str) -> str:
        model.eval()
        prompt = build_chat_prompt(args.default_system, user_text)
        prompt_ids = tok.encode(prompt).ids
        if prompt_ids and prompt_ids[-1] == EOS_ID:
            prompt_ids = prompt_ids[:-1]
        if not prompt_ids:
            return ""

        if len(prompt_ids) >= args.seq_len:
            prompt_ids = prompt_ids[-(args.seq_len - 1):]

        ids = torch.tensor(prompt_ids, device=device, dtype=torch.long)[None, :]
        g = torch.Generator(device=device)
        g.manual_seed(args.sample_seed)

        # stop when [/ASSISTANT] appears (token-level)
        stop_ids = encode_strip_special(tok, ASSIST_CLOSE, BOS_ID, EOS_ID)

        def top_k_top_p(logits_1d: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
            if top_k and top_k > 0:
                top_k = min(top_k, logits_1d.size(-1))
                v, _ = torch.topk(logits_1d, top_k)
                thresh = v[-1]
                logits_1d = torch.where(logits_1d < thresh, torch.full_like(logits_1d, -float("inf")), logits_1d)
            if top_p and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits_1d, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cum = torch.cumsum(probs, dim=-1)
                mask_sorted = cum > top_p
                mask_sorted[0] = False
                mask = torch.zeros_like(mask_sorted)
                mask.scatter_(0, sorted_idx, mask_sorted)
                logits_1d = torch.where(mask, torch.full_like(logits_1d, -float("inf")), logits_1d)
            return logits_1d

        max_new = args.sample_max_new_tokens
        temperature = args.sample_temperature
        top_p = args.sample_top_p
        top_k = args.sample_top_k

        gen: List[int] = []
        for _ in range(max_new):
            if ids.size(1) > args.seq_len:
                ids = ids[:, -args.seq_len:]
            logits = model(ids)
            next_logits = logits[0, -1, :].float()

            if temperature <= 0:
                nxt = int(torch.argmax(next_logits).item())
            else:
                next_logits = next_logits / temperature
                next_logits = top_k_top_p(next_logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(next_logits, dim=-1)
                if torch.isnan(probs).any() or float(probs.sum().item()) == 0.0:
                    nxt = int(torch.argmax(next_logits).item())
                else:
                    nxt = int(torch.multinomial(probs, num_samples=1, generator=g).item())

            gen.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=device, dtype=torch.long)], dim=1)

            if stop_ids and len(gen) >= len(stop_ids) and gen[-len(stop_ids):] == stop_ids:
                break
            if nxt == EOS_ID:
                break

        text = tok.decode(ids[0].tolist())
        pos = text.rfind(ASSIST_OPEN)
        if pos != -1:
            out = text[pos + len(ASSIST_OPEN):]
            close_pos = out.rfind(ASSIST_CLOSE)
            if close_pos != -1:
                out = out[:close_pos]
            return out.strip()
        return text.strip()

    model.train()
    t0 = time.time()
    running_loss = 0.0
    step = start_step
    train_iter = iter(train_loader)

    # rng for in-domain sampling (choose val items)
    in_rng = random.Random(args.sample_in_domain_seed)

    while step < args.max_steps:
        optimizer.zero_grad(set_to_none=True)
        micro_loss = 0.0

        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        for _ in range(args.grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(autocast_dtype is not None)):
                logits = model(input_ids)
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
            tokens_per_step = args.micro_bsz * args.grad_accum * args.seq_len
            tok_s = tokens_per_step * 50.0 / max(dt, 1e-9)
            print(f"[train] step={step} loss={running_loss/50:.4f} lr={get_lr(step):.2e} tok/s≈{tok_s:.0f} dt={dt:.1f}s")
            running_loss = 0.0
            t0 = time.time()

        if args.eval_every > 0 and step % args.eval_every == 0:
            model.eval()
            losses = []
            with torch.no_grad():
                for j, vb in enumerate(val_loader):
                    if j >= args.eval_batches:
                        break
                    vi = vb["input_ids"].to(device)
                    vl = vb["labels"].to(device)
                    with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(autocast_dtype is not None)):
                        v_logits = model(vi)
                        v_loss = masked_ce_loss(v_logits, vl)
                    losses.append(float(v_loss.item()))
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

        # ---- Sampling (fixed + in-domain) ----
        if args.sample_every and args.sample_every > 0 and step % args.sample_every == 0:
            sdir = args.samples_dir or os.path.join(args.out_dir, "samples")
            Path(sdir).mkdir(parents=True, exist_ok=True)
            out_path = os.path.join(sdir, f"step_{step:06d}.txt")

            lines: List[str] = []
            lines.append(f"step={step}\n")
            lines.append(
                f"sampling: temp={args.sample_temperature} top_p={args.sample_top_p} top_k={args.sample_top_k} max_new={args.sample_max_new_tokens}\n"
            )
            lines.append("=" * 80 + "\n")

            # A) fixed prompts (out-of-domain)
            lines.append("[Fixed prompts]\n")
            lines.append("-" * 80 + "\n")
            for i, q in enumerate(FIXED_PROMPTS):
                ans = sample_once(q)
                lines.append(f"[Q{i+1}] {q}\n")
                lines.append(f"[A{i+1}] {ans}\n")
                lines.append("-" * 80 + "\n")

            # B) in-domain prompts from val_jsonl
            if args.sample_in_domain_n > 0 and len(val_ds) > 0:
                lines.append("\n[In-domain prompts from val_jsonl]\n")
                lines.append("-" * 80 + "\n")
                n = min(args.sample_in_domain_n, len(val_ds))
                # sample indices with deterministic RNG
                idxs = [in_rng.randrange(len(val_ds)) for _ in range(n)]
                for k, idx in enumerate(idxs, start=1):
                    ex = val_ds[idx]
                    msgs = ex.get("messages") or []
                    user_q, ref_a = extract_last_user_and_ref(msgs)
                    if not user_q:
                        continue
                    ans = sample_once(user_q)
                    lines.append(f"[V{k}] idx={idx}\n")
                    lines.append(f"[User]\n{user_q}\n")
                    if args.sample_in_domain_show_ref and ref_a:
                        lines.append(f"[Ref assistant]\n{ref_a}\n")
                    lines.append(f"[Model]\n{ans}\n")
                    lines.append("-" * 80 + "\n")

            with open(out_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
            print(f"[sample] wrote {out_path}")

    print("[done]")


if __name__ == "__main__":
    main()
