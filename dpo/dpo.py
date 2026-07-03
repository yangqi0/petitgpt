#!/usr/bin/env python3

"""
Direct Preference Optimization (DPO) post-training for petitgpt.

Starts from an SFT or distill checkpoint (schema: keys "model" + "cfg"/"config"),
builds a frozen reference copy of that same checkpoint, and optimizes the policy
on preference pairs with the standard DPO loss (Rafailov et al., 2023):

    L = -log sigmoid( beta * [ (logpi_w - logref_w) - (logpi_l - logref_l) ] )

where w/l are the chosen/rejected completions and logpi_*/logref_* are the
summed log-probs of the completion tokens under the policy / reference model.

Expected data format (JSONL, one example per line), matching the plain chat
template used by sft/train_sft.py and distill/train_distill.py:

    {"messages": [{"role": "system", "content": "..."},
                   {"role": "user", "content": "..."}],
     "chosen": "the preferred assistant response",
     "rejected": "the dispreferred assistant response"}

`messages` is the shared prompt context (system/user turns, and any earlier
assistant turns for multi-turn prompts); `chosen`/`rejected` are plain
assistant-completion strings that get rendered with the same
"System: .../User: .../Assistant: ..." template and appended to that context.

Example:
    python dpo/dpo.py \\
      --train_jsonl datasets/dpo/train.jsonl --val_jsonl datasets/dpo/val.jsonl \\
      --out_dir outputs/dpo_run --tokenizer_path tokenizer/tokenizer.json \\
      --init_ckpt outputs/sft_v6_general_code/step_003500.pt \\
      --seq_len 1024 --micro_bsz 2 --grad_accum 8 --lr 5e-6 --beta 0.1
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import GPT, GPTConfig  # noqa: E402

# -------------------------
# Plain chat template (NO bracket tags; robust to tokenizer)
# Kept identical to sft/train_sft.py and distill/train_distill.py.
# -------------------------
BOS_ID = 2
EOS_ID = 3

SYS_PREFIX = "System: "
USER_PREFIX = "User: "
ASSIST_PREFIX = "Assistant: "
SEP = "\n\n"

FIXED_PROMPTS = [
    "[Code] Write a Python function running_sum(nums) that returns cumulative sums.",
    "[Code] Write a Python function lowercase_keys(d) that returns a new dictionary with lowercase string keys.",
    "[General] Write a short polite email asking for an update on a job application after an interview.",
    "[General] Rewrite this to be more concise: 'I am writing this email in order to ask whether it would be possible to move our meeting to Friday afternoon.'",
]


def norm_newlines(s: str) -> str:
    return (s or "").replace("\r\n", "\n").replace("\r", "\n")


def clean_text(s: str) -> str:
    return norm_newlines(s).strip()


def clean_text_assistant(s: str) -> str:
    # do not strip: keeps code indentation / markdown formatting in completions
    return norm_newlines(s)


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
# Encoding helpers (robust to tokenizer auto BOS/EOS)
# -------------------------
def encode_strip_special(tok: Tokenizer, text: str, bos_id: int, eos_id: int) -> list[int]:
    ids = tok.encode(text).ids
    if ids and ids[0] == bos_id:
        ids = ids[1:]
    if ids and ids[-1] == eos_id:
        ids = ids[:-1]
    return ids


def tokenizer_auto_bos_eos(tok: Tokenizer, bos_id: int, eos_id: int) -> tuple[bool, bool]:
    probe = tok.encode("x").ids
    has_bos = bool(probe) and probe[0] == bos_id
    has_eos = bool(probe) and probe[-1] == eos_id
    return has_bos, has_eos


# -------------------------
# Prompt rendering + preference-pair example building
# -------------------------
def render_prompt_segments(
    messages: list[dict[str, str]], default_system: str
) -> list[tuple[str, bool]]:
    """Render shared prompt context (system/user/earlier-assistant turns) as
    unsupervised segments. The trailing chosen/rejected completion is appended
    separately by build_completion_example()."""
    msgs = messages or []
    segs: list[tuple[str, bool]] = []
    if not msgs:
        return segs

    if (msgs[0].get("role") or "").strip().lower() != "system" and default_system:
        msgs = [{"role": "system", "content": default_system}] + msgs

    if msgs and (msgs[0].get("role") or "").strip().lower() == "system":
        sys_txt = clean_text(msgs[0].get("content", ""))
        if sys_txt:
            segs.append((SYS_PREFIX, False))
            segs.append((sys_txt, False))
            segs.append((SEP, False))
        start = 1
    else:
        start = 0

    for m in msgs[start:]:
        role = (m.get("role") or "").strip().lower()
        raw = m.get("content", "")
        txt = clean_text_assistant(raw) if role == "assistant" else clean_text(raw)
        if not txt:
            continue
        if role == "user":
            segs.append((USER_PREFIX, False))
            segs.append((txt, False))
            segs.append((SEP, False))
        elif role == "assistant":
            segs.append((ASSIST_PREFIX, False))
            segs.append((txt, False))
            segs.append((SEP, False))

    return segs


def build_completion_example(
    messages: list[dict[str, str]],
    completion: str,
    tok: Tokenizer,
    seq_len: int,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    default_system: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render (prompt + completion) and return (input_ids, labels), with
    labels=-100 everywhere except the completion tokens (EOS excluded from
    supervision, matching sft/train_sft.py's build_example)."""
    segs = render_prompt_segments(messages, default_system)
    segs.append((ASSIST_PREFIX, False))
    segs.append((clean_text_assistant(completion), True))

    has_bos, has_eos = tokenizer_auto_bos_eos(tok, bos_id, eos_id)

    ids_all: list[int] = []
    labels_all: list[int] = []

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

    # Truncate/pad (keep tail so the completion is more likely preserved)
    if len(ids_all) > seq_len:
        ids_all = ids_all[-seq_len:]
        labels_all = labels_all[-seq_len:]
    else:
        pad_n = seq_len - len(ids_all)
        ids_all = ids_all + [pad_id] * pad_n
        labels_all = labels_all + [-100] * pad_n

    return (
        torch.tensor(ids_all, dtype=torch.long),
        torch.tensor(labels_all, dtype=torch.long),
    )


def collate_fn_builder(
    tok: Tokenizer,
    seq_len: int,
    pad_id: int,
    default_system: str,
    debug_first_batch: bool,
):
    printed = {"done": False}

    def collate(batch: list[dict[str, Any]]):
        xs_c, ys_c, xs_r, ys_r = [], [], [], []
        for ex in batch:
            messages = ex.get("messages") or []
            chosen = ex.get("chosen", "")
            rejected = ex.get("rejected", "")
            if not messages:
                raise ValueError("missing messages")
            if not chosen or not rejected:
                raise ValueError("missing chosen/rejected completion")

            xc, yc = build_completion_example(
                messages, chosen, tok, seq_len, pad_id, BOS_ID, EOS_ID, default_system
            )
            xr, yr = build_completion_example(
                messages, rejected, tok, seq_len, pad_id, BOS_ID, EOS_ID, default_system
            )
            xs_c.append(xc)
            ys_c.append(yc)
            xs_r.append(xr)
            ys_r.append(yr)

        batch_out = {
            "input_ids_chosen": torch.stack(xs_c, dim=0),
            "labels_chosen": torch.stack(ys_c, dim=0),
            "input_ids_rejected": torch.stack(xs_r, dim=0),
            "labels_rejected": torch.stack(ys_r, dim=0),
        }

        if debug_first_batch and not printed["done"]:
            printed["done"] = True
            lc0 = batch_out["labels_chosen"][0]
            lr0 = batch_out["labels_rejected"][0]
            print(f"[dbg] chosen supervised tokens(sample0): {int((lc0 != -100).sum().item())}")
            print(f"[dbg] rejected supervised tokens(sample0): {int((lr0 != -100).sum().item())}")
            idx = (lc0 != -100).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() > 0:
                dec = tok.decode(batch_out["input_ids_chosen"][0, idx].tolist())
                print(f"[dbg] decoded chosen span(first 300 chars): {dec[:300]}")

        return batch_out

    return collate


# -------------------------
# DPO loss
# -------------------------
def sequence_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum of log-probs of the supervised (label != -100) next-token targets,
    per example. logits: [B,T,V], labels: [B,T]."""
    B, T, V = logits.size()
    logits2 = logits[:, :-1, :].float()
    labels2 = labels[:, 1:]
    # per-token NLL; ignore_index positions contribute exactly 0 to the sum
    nll = F.cross_entropy(
        logits2.reshape(-1, V),
        labels2.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(B, T - 1)
    return -nll.sum(dim=1)


def get_batch_logps(
    model: torch.nn.Module,
    input_ids_chosen: torch.Tensor,
    labels_chosen: torch.Tensor,
    input_ids_rejected: torch.Tensor,
    labels_rejected: torch.Tensor,
    autocast_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single fused forward pass over chosen+rejected (concatenated on the
    batch dim) for one model, returning (chosen_logps, rejected_logps)."""
    b = input_ids_chosen.size(0)
    input_ids = torch.cat([input_ids_chosen, input_ids_rejected], dim=0)
    labels = torch.cat([labels_chosen, labels_rejected], dim=0)
    with torch.autocast(
        device_type="cuda", dtype=autocast_dtype, enabled=(autocast_dtype is not None)
    ):
        logits = model(input_ids)
    logps = sequence_logps(logits, labels)
    return logps[:b], logps[b:]


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-example DPO loss + implicit rewards. Returns (losses[B], chosen_rewards[B], rejected_rewards[B])."""
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = (beta * (policy_chosen_logps - ref_chosen_logps)).detach()
    rejected_rewards = (beta * (policy_rejected_logps - ref_rejected_logps)).detach()
    return losses, chosen_rewards, rejected_rewards


# -------------------------
# Checkpoints
# -------------------------
def save_checkpoint_atomic(path: str, obj: dict[str, Any]) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_ckpt(path: str) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def build_model_from_ckpt(ckpt: dict[str, Any], vocab_size: int, seq_len: int, device: str) -> tuple[GPT, GPTConfig]:
    cfg_dict = ckpt.get("cfg") or ckpt.get("config")
    if not isinstance(cfg_dict, dict):
        raise RuntimeError("checkpoint missing 'cfg'/'config' dict")
    cfg_dict = dict(cfg_dict)
    cfg_dict["vocab_size"] = vocab_size
    cfg_dict["max_seq_len"] = seq_len
    cfg = GPTConfig(**cfg_dict)
    model = GPT(cfg).to(device)

    sd = ckpt.get("model")
    if sd is None:
        raise RuntimeError("checkpoint missing 'model'")
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k[len("_orig_mod.") :]: v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"    missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    return model, cfg


# -------------------------
# Sampling (qualitative check; policy only, no reference comparison)
# -------------------------
@torch.no_grad()
def sample_from_prompt(
    model: torch.nn.Module,
    tok: Tokenizer,
    prompt_text: str,
    device: str,
    seq_len: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    model.eval()
    prompt_ids = tok.encode(prompt_text).ids
    if prompt_ids and prompt_ids[-1] == EOS_ID:
        prompt_ids = prompt_ids[:-1]
    if not prompt_ids:
        return ""
    if len(prompt_ids) >= seq_len:
        prompt_ids = prompt_ids[-(seq_len - 1) :]

    ids = torch.tensor(prompt_ids, device=device, dtype=torch.long)[None, :]
    for _ in range(max_new_tokens):
        if ids.size(1) > seq_len:
            ids = ids[:, -seq_len:]
        logits = model(ids)[0, -1, :].float()

        if temperature <= 0:
            nxt = int(torch.argmax(logits).item())
        else:
            logits = logits / temperature
            if top_p and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cum = torch.cumsum(probs, dim=-1)
                mask_sorted = cum > top_p
                mask_sorted[0] = False
                mask = torch.zeros_like(mask_sorted)
                mask.scatter_(0, sorted_idx, mask_sorted)
                logits = torch.where(mask, torch.full_like(logits, -float("inf")), logits)
            probs = torch.softmax(logits, dim=-1)
            nxt = int(torch.multinomial(probs, num_samples=1).item())

        ids = torch.cat([ids, torch.tensor([[nxt]], device=device, dtype=torch.long)], dim=1)
        if nxt == EOS_ID:
            break

    text = tok.decode(ids[0].tolist())
    pos = text.rfind(ASSIST_PREFIX)
    out = text[pos + len(ASSIST_PREFIX) :] if pos != -1 else text
    model.train()
    return out.strip()


def emit_samples(
    policy: torch.nn.Module,
    tok: Tokenizer,
    samples_dir: str,
    step_tag: str,
    device: str,
    seq_len: int,
    default_system: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> None:
    Path(samples_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(samples_dir, f"{step_tag}.txt")
    lines: list[str] = [f"step={step_tag}\n", "=" * 80 + "\n"]
    for i, q in enumerate(FIXED_PROMPTS, start=1):
        prompt = SYS_PREFIX + default_system.strip() + SEP + USER_PREFIX + q.strip() + SEP + ASSIST_PREFIX
        ans = sample_from_prompt(policy, tok, prompt, device, seq_len, max_new_tokens, temperature, top_p)
        lines.append(f"[Q{i}] {q}\n[A{i}] {ans}\n" + "-" * 80 + "\n")
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"[sample] wrote {out_path}")


# -------------------------
# Eval
# -------------------------
@torch.no_grad()
def evaluate(
    policy: torch.nn.Module,
    reference: torch.nn.Module,
    val_loader: DataLoader,
    device: str,
    autocast_dtype: torch.dtype | None,
    beta: float,
    max_batches: int,
) -> dict[str, float]:
    policy.eval()
    losses, margins, accs = [], [], []
    for j, batch in enumerate(val_loader):
        if j >= max_batches:
            break
        in_c = batch["input_ids_chosen"].to(device, non_blocking=True)
        lb_c = batch["labels_chosen"].to(device, non_blocking=True)
        in_r = batch["input_ids_rejected"].to(device, non_blocking=True)
        lb_r = batch["labels_rejected"].to(device, non_blocking=True)

        pc, pr = get_batch_logps(policy, in_c, lb_c, in_r, lb_r, autocast_dtype)
        rc, rr = get_batch_logps(reference, in_c, lb_c, in_r, lb_r, autocast_dtype)
        loss_vec, chosen_rewards, rejected_rewards = dpo_loss(pc, pr, rc, rr, beta)

        losses.append(float(loss_vec.mean().item()))
        margins.append(float((chosen_rewards - rejected_rewards).mean().item()))
        accs.append(float((chosen_rewards > rejected_rewards).float().mean().item()))

    policy.train()
    n = max(1, len(losses))
    return {
        "val_loss": sum(losses) / n,
        "val_reward_margin": sum(margins) / n,
        "val_reward_acc": sum(accs) / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tokenizer_path", required=True)

    ap.add_argument("--init_ckpt", required=True, help="SFT or distill checkpoint to init policy+reference from")
    ap.add_argument("--ref_ckpt", default="", help="Optional separate reference checkpoint (default: deep-copy of the initialized policy)")

    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--micro_bsz", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)

    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_steps", type=int, default=150)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--grad_clip", type=float, default=1.0, help="0 disables grad clipping")
    ap.add_argument("--beta", type=float, default=0.1, help="DPO temperature; higher = closer to reference")

    ap.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="bf16")
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_batches", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=200)

    ap.add_argument("--default_system", default="You are a helpful assistant.")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--debug_first_batch", action="store_true")

    ap.add_argument("--sample_every", type=int, default=200)
    ap.add_argument("--samples_dir", type=str, default="")
    ap.add_argument("--sample_max_new_tokens", type=int, default=192)
    ap.add_argument("--sample_temperature", type=float, default=0.7)
    ap.add_argument("--sample_top_p", type=float, default=0.9)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    tok = Tokenizer.from_file(args.tokenizer_path)
    vocab_size = tok.get_vocab_size()
    pad_id = 0

    print(f"[*] loading policy init from: {args.init_ckpt}")
    init_ckpt = load_ckpt(args.init_ckpt)
    policy, cfg = build_model_from_ckpt(init_ckpt, vocab_size, args.seq_len, device)

    if args.ref_ckpt:
        print(f"[*] loading reference from: {args.ref_ckpt}")
        ref_ckpt = load_ckpt(args.ref_ckpt)
        reference, _ = build_model_from_ckpt(ref_ckpt, vocab_size, args.seq_len, device)
    else:
        print("[*] reference = frozen deep-copy of the initialized policy")
        reference = copy.deepcopy(policy)

    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    use_fp16 = args.precision == "fp16" and device == "cuda"
    use_bf16 = args.precision == "bf16" and device == "cuda"
    autocast_dtype = torch.float16 if use_fp16 else (torch.bfloat16 if use_bf16 else None)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    train_ds = JsonlOffsetsDataset(args.train_jsonl)
    val_ds = JsonlOffsetsDataset(args.val_jsonl)
    print(f"[*] dataset: train_lines={len(train_ds)} val_lines={len(val_ds)}")
    print(
        f"[*] effective_pairs/step = micro_bsz({args.micro_bsz}) * grad_accum({args.grad_accum})"
        f" = {args.micro_bsz * args.grad_accum}"
    )

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

    samples_dir = args.samples_dir or os.path.join(args.out_dir, "samples")

    policy.train()
    t0 = time.time()
    running_loss = 0.0
    running_margin = 0.0
    running_acc = 0.0
    step = 0
    train_iter = iter(train_loader)

    while step < args.max_steps:
        optimizer.zero_grad(set_to_none=True)
        micro_loss = 0.0
        micro_margin = 0.0
        micro_acc = 0.0

        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        for _ in range(args.grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            in_c = batch["input_ids_chosen"].to(device, non_blocking=True)
            lb_c = batch["labels_chosen"].to(device, non_blocking=True)
            in_r = batch["input_ids_rejected"].to(device, non_blocking=True)
            lb_r = batch["labels_rejected"].to(device, non_blocking=True)

            pc, pr = get_batch_logps(policy, in_c, lb_c, in_r, lb_r, autocast_dtype)
            with torch.no_grad():
                rc, rr = get_batch_logps(reference, in_c, lb_c, in_r, lb_r, autocast_dtype)

            loss_vec, chosen_rewards, rejected_rewards = dpo_loss(pc, pr, rc, rr, args.beta)
            loss = loss_vec.mean() / args.grad_accum

            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            micro_loss += float(loss.item())
            micro_margin += float((chosen_rewards - rejected_rewards).mean().item()) / args.grad_accum
            micro_acc += float((chosen_rewards > rejected_rewards).float().mean().item()) / args.grad_accum

        if args.grad_clip > 0:
            if use_fp16:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)

        if use_fp16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        step += 1
        running_loss += micro_loss
        running_margin += micro_margin
        running_acc += micro_acc

        if step % 20 == 0:
            dt = time.time() - t0
            print(
                f"[train] step={step} loss={running_loss / 20:.4f} "
                f"reward_margin={running_margin / 20:.4f} reward_acc={running_acc / 20:.4f} "
                f"lr={lr:.2e} dt={dt:.1f}s"
            )
            running_loss = 0.0
            running_margin = 0.0
            running_acc = 0.0
            t0 = time.time()

        if args.eval_every > 0 and step % args.eval_every == 0:
            metrics = evaluate(policy, reference, val_loader, device, autocast_dtype, args.beta, args.eval_batches)
            print(
                f"[eval] step={step} val_loss={metrics['val_loss']:.4f} "
                f"val_reward_margin={metrics['val_reward_margin']:.4f} "
                f"val_reward_acc={metrics['val_reward_acc']:.4f}"
            )

        if args.save_every > 0 and step % args.save_every == 0:
            ckpt = {
                "step": step,
                "model": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if use_fp16 else None,
                "cfg": asdict(cfg),
                "args": vars(args),
                "kind": "dpo",
            }
            save_checkpoint_atomic(os.path.join(args.out_dir, f"step_{step:06d}.pt"), ckpt)
            save_checkpoint_atomic(os.path.join(args.out_dir, "latest.pt"), ckpt)
            print(f"[ckpt] saved step_{step:06d}.pt")

        if args.sample_every and args.sample_every > 0 and step % args.sample_every == 0:
            emit_samples(
                policy,
                tok,
                samples_dir,
                f"step_{step:06d}",
                device,
                args.seq_len,
                args.default_system,
                args.sample_max_new_tokens,
                args.sample_temperature,
                args.sample_top_p,
            )

    print("[done]")


if __name__ == "__main__":
    main()
