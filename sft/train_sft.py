#!/usr/bin/env python3

"""Supervised fine-tuning on chat data (also the engine behind distill/train_distill.py).

Chat encoding, loss masking, and refusal weighting all live in
src/chat_template.py — a single source of truth shared with DPO/GRPO and with
the sampling code below, so training-time and inference-time token sequences
are identical by construction.

Contract: [BOS] <|system|> ... <|user|> ... <|assistant|> {answer} [EOS] ...,
supervised span = every assistant answer + its trailing EOS.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random

# Make imports work no matter where you run from
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.chat_template import (  # noqa: E402
    build_example,
    decode_completion,
    encode_prompt,
    extract_last_user_and_ref,
    load_chat_tokenizer,
)
from src.model import GPT, GPTConfig  # noqa: E402
from src.optim import build_optimizer  # noqa: E402
from src.special_tokens import EOS_ID, PAD_ID  # noqa: E402
from src.tracking import Tracker  # noqa: E402


# -------------------------
# Dataset: jsonl offsets
# -------------------------
class JsonlOffsetsDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        self.offsets: list[int] = []
        with open(path, "rb") as f:
            off = 0
            for line in f:
                self.offsets.append(off)
                off += len(line)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        off = self.offsets[idx]
        with open(self.path, "rb") as f:
            f.seek(off)
            line = f.readline().decode("utf-8")
        return json.loads(line)


# -------------------------
# Read train/val jsonl
# -------------------------
def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# -------------------------
# Fixed prompts
# -------------------------
FIXED_PROMPTS = [
    "[Code] Write a Python function running_sum(nums) that returns cumulative sums.",
    "[Code] Write a Python function lowercase_keys(d) that returns a new dictionary with lowercase string keys.",
    "[General] Write a short polite email asking for an update on a job application after an interview.",
    "[General] Write a short professional email asking to reschedule a meeting to next week.",
    "[General] Summarize this in 3 bullet points: 'Regular exercise can improve mood, support heart health, and help maintain energy levels.'",
    "[General] Rewrite this to be more formal: 'Thanks for the quick reply. I’ll send the file tomorrow.'",
    "[General] Rewrite this to be more concise: 'I am writing this email in order to ask whether it would be possible to move our meeting to Friday afternoon.'",
    "[General] Explain in at most 4 sentences what a budget is, in simple everyday language.",
]


def collate_fn_builder(
    tok,
    seq_len: int,
    default_system: str,
    debug_first_batch: bool,
    refusal_downweight: float,
    refusal_patterns: list[str],
    refusal_mode: str,
):
    printed = {"done": False}

    def collate(batch: list[dict[str, Any]]):
        xs, ys, ws = [], [], []
        for ex in batch:
            x, y, w = build_example(
                ex,
                tok,
                seq_len,
                default_system,
                refusal_downweight,
                refusal_patterns,
                refusal_mode,
                pad_id=PAD_ID,
            )
            xs.append(x)
            ys.append(y)
            ws.append(w)

        input_ids = torch.stack(xs, dim=0)
        labels = torch.stack(ys, dim=0)
        weights = torch.tensor(ws, dtype=torch.float32)

        if debug_first_batch and not printed["done"]:
            printed["done"] = True
            sup = int((labels[0] != -100).sum().item())
            tot = labels[0].numel()
            print(f"[dbg] supervised tokens(sample0): {sup}/{tot} ({sup / tot:.3f})")
            print(f"[dbg] example_weight(sample0): {float(weights[0].item()):.3f}")

            idx = (labels[0] != -100).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() > 0:
                dec = tok.decode(input_ids[0, idx].tolist())
                print("[dbg] decoded supervised span(first 300 chars):")
                print(dec[:300])
            else:
                print("[dbg] WARNING: no supervised tokens in sample0")

        return {"input_ids": input_ids, "labels": labels, "weights": weights}

    return collate


def masked_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None = None,
    reduction: str = "token_mean",
) -> torch.Tensor:
    """
    Next-token CE, ignoring -100 labels.

    reduction:
      - token_mean: average over all supervised tokens in batch (mainstream
        default: every supervised token gets equal weight)
      - example_mean: mean CE per-example (over supervised tokens), then average
        across batch (upweights tokens in short answers)

    If weights provided (shape [B]), they act as per-example scalars (e.g. refusal downweight).
    """
    B, T, V = logits.size()
    # next-token
    logits2 = logits[:, :-1, :].contiguous()  # [B, T-1, V]
    labels2 = labels[:, 1:].contiguous()  # [B, T-1]

    # per-token loss
    loss_tok = F.cross_entropy(
        logits2.view(-1, V),
        labels2.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(B, T - 1)  # [B, T-1]

    mask = (labels2 != -100).float()  # [B, T-1]
    if weights is None:
        weights = torch.ones((B,), device=logits.device, dtype=torch.float32)
    else:
        weights = weights.to(device=logits.device, dtype=torch.float32)

    if reduction == "token_mean":
        w = weights.view(B, 1)  # [B,1]
        loss_tok = loss_tok * mask * w
        denom = (mask * w).sum().clamp_min(1.0)
        return loss_tok.sum() / denom

    if reduction == "example_mean":
        tok_cnt = mask.sum(dim=1).clamp_min(1.0)  # [B]
        loss_ex = (loss_tok * mask).sum(dim=1) / tok_cnt  # [B]
        denom = weights.sum().clamp_min(1.0)
        return (loss_ex * weights).sum() / denom

    raise ValueError(f"unknown reduction: {reduction}")


def save_checkpoint_atomic(path: str, obj: dict[str, Any]):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_ckpt(path: str) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def build_arg_parser() -> argparse.ArgumentParser:
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
    ap.add_argument("--optimizer", choices=["muon", "adamw"], default="muon",
                    help="muon: Muon on hidden matrices + AdamW on embeddings/norms (default). adamw: AdamW everywhere.")
    ap.add_argument("--muon_lr", type=float, default=0.0,
                    help="LR for Muon matrix groups (<=0: reuse --lr; Muon update RMS is matched to AdamW's).")
    ap.add_argument("--muon_momentum", type=float, default=0.95)
    ap.add_argument("--max_steps", type=int, default=15000)
    ap.add_argument("--warmup_steps", type=int, default=300)
    ap.add_argument("--grad_clip", type=float, default=1.0, help="0 disables grad clipping")

    ap.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="bf16")
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_batches", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=2000)

    ap.add_argument("--init_from_pretrain", default="")
    ap.add_argument("--resume", default="")
    ap.add_argument("--default_system", default="You are a helpful assistant.")

    ap.add_argument("--n_layers", type=int, default=16)
    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--n_heads", type=int, default=12)
    ap.add_argument("--d_ff", type=int, default=1920)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--tie_embeddings", action="store_true")

    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--debug_first_batch", action="store_true")

    ap.add_argument(
        "--loss_reduction",
        choices=["token_mean", "example_mean"],
        default="token_mean",
        help="token_mean (mainstream default): every supervised token weighs the same. "
        "example_mean: every example weighs the same (upweights short answers).",
    )

    # ---- refusal downweight (TRAINING, not sampling) ----
    ap.add_argument(
        "--refusal_downweight",
        type=float,
        default=0.25,
        help="If an example contains refusal-like assistant text, multiply its loss by this factor (e.g. 0.1~0.5). 1.0=off",
    )
    ap.add_argument(
        "--refusal_mode",
        choices=["contains_any"],
        default="contains_any",
        help="How to detect refusal examples.",
    )
    ap.add_argument(
        "--refusal_patterns",
        type=str,
        default="i am not capable,i am not able,i do not have access,i can't,i cannot,as an ai,i'm unable,unable to,not allowed to,can't help with,do not have the capability",
        help="Comma-separated substrings used to detect refusals in assistant content.",
    )

    # sampling
    ap.add_argument("--sample_every", type=int, default=1000)
    ap.add_argument("--samples_dir", type=str, default="")
    ap.add_argument("--sample_max_new_tokens", type=int, default=192)
    ap.add_argument("--sample_temperature", type=float, default=0.7)
    ap.add_argument("--sample_top_p", type=float, default=0.9)
    ap.add_argument("--sample_top_k", type=int, default=50)
    ap.add_argument("--sample_seed", type=int, default=1234)
    ap.add_argument(
        "--sample_repetition_penalty",
        type=float,
        default=1.12,
        help=">1.0 discourages repeating tokens during sampling (e.g. 1.05~1.25)",
    )
    ap.add_argument(
        "--sample_repetition_window",
        type=int,
        default=256,
        help="how many recent tokens to apply repetition penalty over",
    )
    ap.add_argument(
        "--sample_no_repeat_ngram",
        type=int,
        default=3,
        help="disallow repeating n-grams of this size during sampling (0=off). e.g. 3",
    )

    # in-domain sampling (val_jsonl)
    ap.add_argument("--sample_in_domain_n", type=int, default=10)
    ap.add_argument("--sample_in_domain_seed", type=int, default=1234)
    ap.add_argument("--sample_in_domain_show_ref", action="store_true")
    ap.add_argument(
        "--sample_in_domain_mode",
        choices=["last_user", "full_context"],
        default="full_context",
    )
    ap.add_argument(
        "--sample_in_domain_dump_prompt",
        action="store_true",
        help="also dump the rendered prompt (decoded) for debugging inputs",
    )
    ap.add_argument(
        "--sample_eval_jsonl",
        default="",
        help="Optional curated eval jsonl for sampling. If set, this is used instead of random val_jsonl in-domain sampling.",
    )
    ap.add_argument(
        "--sample_only_ckpt",
        default="",
        help="Optional ckpt path. If set, load this checkpoint, write samples once, then exit without training.",
    )
    return ap


def main(argv: list[str] | None = None):
    args = build_arg_parser().parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    tok = load_chat_tokenizer(args.tokenizer_path)
    vocab_size = tok.get_vocab_size()

    tracker = Tracker(args.out_dir)
    tracker.log_run_start(vars(args), args.tokenizer_path)

    refusal_patterns = [
        p.strip() for p in args.refusal_patterns.split(",") if p.strip()
    ]

    cfg: GPTConfig | None = None

    if args.init_from_pretrain:
        ck = load_ckpt(args.init_from_pretrain)
        cfg_dict = ck.get("config") or ck.get("cfg")
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
        if any(k.startswith("_orig_mod.") for k in sd.keys()):
            sd = {k[len("_orig_mod.") :]: v for k, v in sd.items()}

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

    optimizer = build_optimizer(
        model,
        name=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        muon_lr=args.muon_lr,
        muon_momentum=args.muon_momentum,
    )

    use_fp16 = args.precision == "fp16" and device == "cuda"
    use_bf16 = args.precision == "bf16" and device == "cuda"
    autocast_dtype = (
        torch.float16 if use_fp16 else (torch.bfloat16 if use_bf16 else None)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ck = load_ckpt(args.resume)
        sd = ck.get("model")
        if sd is None:
            raise RuntimeError("resume ckpt missing 'model'")

        if any(k.startswith("_orig_mod.") for k in sd.keys()):
            sd = {k[len("_orig_mod.") :]: v for k, v in sd.items()}

        model.load_state_dict(sd, strict=False)

        opt = ck.get("optimizer") or ck.get("optim")
        if opt is not None:
            try:
                optimizer.load_state_dict(opt)
            except ValueError as e:
                print(f"[warn] optimizer state incompatible (ckpt saved with a different "
                      f"--optimizer?); starting with fresh optimizer state: {e}")

        sc = ck.get("scaler")
        if sc is not None and use_fp16:
            scaler.load_state_dict(sc)

        start_step = int(ck.get("step", ck.get("global_step", 0)))
        print(f"[*] resumed: {args.resume} at step={start_step}")

    train_ds = JsonlOffsetsDataset(args.train_jsonl)
    val_ds = JsonlOffsetsDataset(args.val_jsonl)
    sample_eval_ds: list[dict[str, Any]] = []
    if args.sample_eval_jsonl:
        sample_eval_ds = read_jsonl(args.sample_eval_jsonl)
        print(
            f"[*] sample_eval_jsonl: {args.sample_eval_jsonl} lines={len(sample_eval_ds)}"
        )

    print(f"[*] dataset: train_lines={len(train_ds)} val_lines={len(val_ds)}")
    print(
        f"[*] effective_tokens/step = micro_bsz({args.micro_bsz}) * grad_accum({args.grad_accum}) * seq_len({args.seq_len})"
        f" = {args.micro_bsz * args.grad_accum * args.seq_len}"
    )
    print(
        f"[*] refusal downweight: mode={args.refusal_mode} downweight={args.refusal_downweight} patterns={len(refusal_patterns)}"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.micro_bsz,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn_builder(
            tok,
            args.seq_len,
            args.default_system,
            args.debug_first_batch,
            args.refusal_downweight,
            refusal_patterns,
            args.refusal_mode,
        ),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.micro_bsz,
        shuffle=False,
        num_workers=max(0, args.num_workers // 2),
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn_builder(
            tok,
            args.seq_len,
            args.default_system,
            False,
            args.refusal_downweight,
            refusal_patterns,
            args.refusal_mode,
        ),
        drop_last=False,
    )

    def get_lr(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        t = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        t = min(max(t, 0.0), 1.0)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * t))

    @torch.no_grad()
    def sample_from_ids(prompt_ids: list[int]) -> str:
        """Generate a completion for an already-encoded prompt (which ends with
        the <|assistant|> cue) and decode ONLY the generated tokens."""
        model.eval()

        if not prompt_ids:
            return ""
        if len(prompt_ids) >= args.seq_len:
            prompt_ids = prompt_ids[-(args.seq_len - 1) :]

        ids = torch.tensor(prompt_ids, device=device, dtype=torch.long)[None, :]
        prompt_len = ids.size(1)
        g = torch.Generator(device=device)
        g.manual_seed(args.sample_seed)

        def top_k_top_p(
            logits_1d: torch.Tensor, top_k: int, top_p: float
        ) -> torch.Tensor:
            if top_k and top_k > 0:
                k = min(top_k, logits_1d.size(-1))
                v, _ = torch.topk(logits_1d, k)
                thresh = v[-1]
                logits_1d = torch.where(
                    logits_1d < thresh,
                    torch.full_like(logits_1d, -float("inf")),
                    logits_1d,
                )
            if top_p and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits_1d, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cum = torch.cumsum(probs, dim=-1)
                mask_sorted = cum > top_p
                mask_sorted[0] = False
                mask = torch.zeros_like(mask_sorted)
                mask.scatter_(0, sorted_idx, mask_sorted)
                logits_1d = torch.where(
                    mask, torch.full_like(logits_1d, -float("inf")), logits_1d
                )
            return logits_1d

        def apply_repetition_penalty(
            logits_1d: torch.Tensor, prev_ids: list[int], penalty: float
        ) -> torch.Tensor:
            if penalty is None or penalty <= 1.0:
                return logits_1d
            if not prev_ids:
                return logits_1d
            uniq = set(prev_ids)
            for tid in uniq:
                if tid < 0 or tid >= logits_1d.numel():
                    continue
                v = logits_1d[tid]
                logits_1d[tid] = v / penalty if v > 0 else v * penalty
            return logits_1d

        def ban_repeat_ngrams(
            logits_1d: torch.Tensor, generated: list[int], n: int
        ) -> torch.Tensor:
            if n is None or n <= 0:
                return logits_1d
            if len(generated) < n - 1:
                return logits_1d
            prefix = tuple(generated[-(n - 1) :])
            banned = set()
            for i in range(len(generated) - n + 1):
                if tuple(generated[i : i + n - 1]) == prefix:
                    banned.add(generated[i + n - 1])
            if banned:
                for t in banned:
                    if 0 <= t < logits_1d.numel():
                        logits_1d[t] = -1e10
            return logits_1d

        for _ in range(args.sample_max_new_tokens):
            if ids.size(1) > args.seq_len:
                ids = ids[:, -args.seq_len :]
            logits = model(ids)
            next_logits = logits[0, -1, :].float()

            if args.sample_temperature <= 0:
                nxt = int(torch.argmax(next_logits).item())
            else:
                next_logits = next_logits / args.sample_temperature
                recent = ids[0, -args.sample_repetition_window :].tolist()
                next_logits = apply_repetition_penalty(
                    next_logits, recent, args.sample_repetition_penalty
                )
                gen_so_far = ids[0].tolist()
                next_logits = ban_repeat_ngrams(
                    next_logits, gen_so_far, args.sample_no_repeat_ngram
                )
                next_logits = top_k_top_p(
                    next_logits, top_k=args.sample_top_k, top_p=args.sample_top_p
                )
                probs = torch.softmax(next_logits, dim=-1)
                if torch.isnan(probs).any() or float(probs.sum().item()) == 0.0:
                    nxt = int(torch.argmax(next_logits).item())
                else:
                    nxt = int(
                        torch.multinomial(probs, num_samples=1, generator=g).item()
                    )

            ids = torch.cat(
                [ids, torch.tensor([[nxt]], device=device, dtype=torch.long)], dim=1
            )
            if nxt == EOS_ID:
                break

        return decode_completion(tok, ids[0, prompt_len:].tolist())

    def build_fixed_prompt_ids(user_q: str) -> list[int]:
        return encode_prompt(
            tok,
            [{"role": "user", "content": user_q}],
            default_system=args.default_system,
            mode="last_user",
        )

    def build_sampling_examples() -> tuple[str, list[tuple[str, dict[str, Any]]]]:
        """
        Returns:
          eval_name: str
          rows: List[(tag, example_dict)]
        """
        rows: list[tuple[str, dict[str, Any]]] = []
        if sample_eval_ds:
            for i, ex in enumerate(sample_eval_ds, start=1):
                meta = ex.get("meta") or {}
                tag = str(meta.get("name", f"eval_{i:02d}"))
                rows.append((tag, ex))
            return "sample_eval_jsonl", rows

        if args.sample_in_domain_n > 0 and len(val_ds) > 0:
            n = min(args.sample_in_domain_n, len(val_ds))
            idxs = [in_rng.randrange(len(val_ds)) for _ in range(n)]
            for idx in idxs:
                ex = val_ds[idx]
                rows.append((f"idx={idx}", ex))
            return "val_jsonl", rows

        return "", rows

    model.train()
    t0 = time.time()
    running_loss = 0.0
    step = start_step
    train_iter = iter(train_loader)

    in_rng = random.Random(args.sample_in_domain_seed)

    @torch.no_grad()
    def emit_samples(step_tag: str) -> None:
        sdir = args.samples_dir or os.path.join(args.out_dir, "samples")
        Path(sdir).mkdir(parents=True, exist_ok=True)
        out_path = os.path.join(sdir, f"{step_tag}.txt")

        lines: List[str] = []
        lines.append(f"step={step_tag}\n")
        lines.append(
            f"sampling: temp={args.sample_temperature} top_p={args.sample_top_p} top_k={args.sample_top_k} max_new={args.sample_max_new_tokens}\n"
        )
        lines.append(
            f"in_domain: n={args.sample_in_domain_n} mode={args.sample_in_domain_mode} "
            f"show_ref={bool(args.sample_in_domain_show_ref)} dump_prompt={bool(args.sample_in_domain_dump_prompt)} "
            f"sample_eval_jsonl={bool(args.sample_eval_jsonl)}\n"
        )
        lines.append("=" * 80 + "\n")

        # A) fixed prompts
        lines.append("[Fixed prompts]\n")
        lines.append("-" * 80 + "\n")
        for i, q in enumerate(FIXED_PROMPTS):
            ans = sample_from_ids(build_fixed_prompt_ids(q))
            lines.append(f"[Q{i + 1}] {q}\n")
            lines.append(f"[A{i + 1}] {ans}\n")
            lines.append("-" * 80 + "\n")

        # B) curated eval prompts or fallback val_jsonl
        eval_name, eval_rows = build_sampling_examples()
        if eval_rows:
            lines.append(f"\n[In-domain prompts from {eval_name}]\n")
            lines.append("-" * 80 + "\n")
            tags = [tag for tag, _ in eval_rows]
            lines.append(f"[eval tags] {tags}\n")
            lines.append("-" * 80 + "\n")

            for k, (tag, ex) in enumerate(eval_rows, start=1):
                msgs = ex.get("messages") or []
                user_q, ref_a = extract_last_user_and_ref(msgs)
                if not user_q:
                    continue

                prompt_ids = encode_prompt(
                    tok, msgs, args.default_system, args.sample_in_domain_mode
                )
                ans = sample_from_ids(prompt_ids)

                meta = ex.get("meta") or {}
                bucket = str(meta.get("bucket", ""))
                lines.append(f"[V{k}] {tag} bucket={bucket}\n")
                lines.append("[User]\n")
                lines.append(f"{user_q}\n")

                if args.sample_in_domain_dump_prompt:
                    lines.append("[Prompt]\n")
                    p = tok.decode(prompt_ids)
                    if len(p) > 4000:
                        p = p[:4000] + "\n...[truncated]\n"
                    lines.append(p + "\n")

                if args.sample_in_domain_show_ref and ref_a:
                    lines.append("[Ref assistant]\n")
                    lines.append(ref_a + "\n")

                lines.append("[Model]\n")
                lines.append(ans + "\n")
                lines.append("-" * 80 + "\n")

        with open(out_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"[sample] wrote {out_path}")

    if args.sample_only_ckpt:
        ck = load_ckpt(args.sample_only_ckpt)
        sd = ck.get("model")
        if sd is None:
            raise RuntimeError("sample_only_ckpt missing 'model'")
        if any(k.startswith("_orig_mod.") for k in sd.keys()):
            sd = {k[len("_orig_mod.") :]: v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[*] sample-only loaded: {args.sample_only_ckpt}")
        print(f"    missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
        model.eval()
        emit_samples(Path(args.sample_only_ckpt).stem)
        print("[done]")
        return

    while step < args.max_steps:
        optimizer.zero_grad(set_to_none=True)
        micro_loss = 0.0

        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr * pg.get("lr_ratio", 1.0)

        for _ in range(args.grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            weights = batch["weights"].to(device, non_blocking=True)

            with torch.autocast(
                device_type="cuda",
                dtype=autocast_dtype,
                enabled=(autocast_dtype is not None),
            ):
                logits = model(input_ids)
                loss = (
                    masked_ce_loss(
                        logits, labels, weights=weights, reduction=args.loss_reduction
                    )
                    / args.grad_accum
                )

            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            micro_loss += float(loss.item())

        if args.grad_clip > 0:
            if use_fp16:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

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
            train_loss_avg = running_loss / 50
            tok_s = tokens_per_step * 50.0 / max(dt, 1e-9)
            print(
                f"[train] step={step} loss={train_loss_avg:.4f} lr={get_lr(step):.2e} tok/s≈{tok_s:.0f} dt={dt:.1f}s"
            )
            tracker.log("train", step, loss=train_loss_avg, lr=get_lr(step), tok_s=tok_s)
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
                    with torch.autocast(
                        device_type="cuda",
                        dtype=autocast_dtype,
                        enabled=(autocast_dtype is not None),
                    ):
                        v_logits = model(vi)
                        # Unweighted token-mean: val loss stays a clean metric,
                        # comparable across runs and refusal/weight settings.
                        v_loss = masked_ce_loss(
                            v_logits, vl, weights=None, reduction="token_mean"
                        )
                    losses.append(float(v_loss.item()))
            val_loss_avg = sum(losses) / max(1, len(losses))
            print(
                f"[eval] step={step} val_loss={val_loss_avg:.4f}"
            )
            tracker.log("val", step, val_loss=val_loss_avg)
            tracker.render()
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
        if (
            args.sample_every
            and args.sample_every > 0
            and step % args.sample_every == 0
        ):
            emit_samples(f"step_{step:06d}")

    tracker.render()
    print("[done]")


if __name__ == "__main__":
    main()
