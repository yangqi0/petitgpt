#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import random
import sys
import time
from typing import Any

from tokenizers import Tokenizer
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import GPT, GPTConfig


# ---------- data ----------
def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# @dataclass
# class ChatFormat:
#     # You can later swap to your repo's exact chat template.
#     # For now, we use simple tags to separate roles.
#     sys_open: str = "[SYS]\n"
#     sys_close: str = "\n[/SYS]\n"
#     user_open: str = "[USER]\n"
#     user_close: str = "\n[/USER]\n"
#     asst_open: str = "[ASSISTANT]\n"
#     asst_close: str = "\n[/ASSISTANT]\n"

#     def render(self, messages: List[Dict[str, str]]) -> Tuple[str, List[Tuple[int,int]]]:
#         """
#         Returns:
#           full_text
#           assistant_spans: list of (start_char, end_char) spans in full_text
#         """
#         parts: List[str] = []
#         spans: List[Tuple[int,int]] = []

#         for m in messages:
#             role = m["role"]
#             content = m["content"]
#             if role == "system":
#                 parts.append(self.sys_open + content + self.sys_close)
#             elif role == "user":
#                 parts.append(self.user_open + content + self.user_close)
#             elif role == "assistant":
#                 start = sum(len(p) for p in parts) + len(self.asst_open)
#                 parts.append(self.asst_open + content + self.asst_close)
#                 end = sum(len(p) for p in parts) - len(self.asst_close)
#                 spans.append((start, end))
#             else:
#                 raise ValueError(f"unknown role: {role}")

#         full = "".join(parts)
#         return full, spans


@dataclass
class ChatFormat:
    def render(self, messages: list[dict[str, str]]) -> tuple[str, list[tuple[int, int]]]:
        """
        Return:
        text: the full concatenated string fed to tokenizer
        asst_spans: list[(start_char, end_char)] spans of assistant CONTENT only
        """
        parts: list[str] = []
        spans: list[tuple[int, int]] = []
        cur = 0

        def add(s: str):
            nonlocal cur
            parts.append(s)
            cur += len(s)

        for m in messages:
            role = m["role"]
            content = m.get("content", "")

            if role == "system":
                add(content.rstrip() + "\n\n")
            elif role == "user":
                add(content.rstrip() + "\n\n")
            elif role == "assistant":
                # IMPORTANT: record span exactly where we append assistant content
                start = cur
                add(content)  # keep exact, DO NOT rstrip here
                end = cur
                spans.append((start, end))
                # ensure separation between turns (but do NOT include in span)
                if not content.endswith("\n\n"):
                    add("\n\n")
            else:
                # ignore unknown roles
                continue

        text = "".join(parts)
        return text, spans


class SFTJsonlDataset(Dataset):
    def __init__(
        self, path: str, tok: Tokenizer, max_len: int, add_bos: bool, bos_id: int, eos_id: int
    ):
        self.rows = read_jsonl(path)
        self.tok = tok
        self.max_len = max_len
        self.add_bos = add_bos
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.fmt = ChatFormat()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.rows[idx]
        text, asst_spans = self.fmt.render(ex["messages"])

        enc = self.tok.encode(text)
        ids = enc.ids
        offsets = enc.offsets  # list of (start,end) char offsets per token

        if self.add_bos:
            ids = [self.bos_id] + ids
            offsets = [(0, 0)] + offsets

        # truncate
        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
            offsets = offsets[: self.max_len]

        # labels: only supervise tokens whose char span overlaps any assistant span
        labels = [-100] * len(ids)  # next-token targets (written to position i-1)

        def token_is_in_asst(tok_span: tuple[int, int]) -> bool:
            ts, te = tok_span
            if ts == te == 0:
                return False
            for a0, a1 in asst_spans:
                # overlap
                if te > a0 and ts < a1:
                    return True
            return False

        # First build a boolean mask for which *tokens* belong to assistant spans.
        is_asst = [False] * len(ids)
        for i, off in enumerate(offsets):
            if token_is_in_asst(off):
                is_asst[i] = True

        # Convert to next-token targets:
        # if token i is assistant, we supervise the model prediction at position i-1 to output ids[i].
        for i in range(1, len(ids)):
            if is_asst[i]:
                labels[i - 1] = ids[i]

        # (optional) avoid supervising EOS/BOS if they appear
        for i, t in enumerate(ids):
            if t == self.bos_id or t == self.eos_id:
                labels[i] = -100

        return {"input_ids": ids, "labels": labels}


def collate(batch: list[dict[str, Any]], pad_id: int, max_len: int):
    bs = len(batch)
    seqlen = min(max(len(x["input_ids"]) for x in batch), max_len)
    input_ids = torch.full((bs, seqlen), pad_id, dtype=torch.long)
    labels = torch.full((bs, seqlen), -100, dtype=torch.long)
    attn = torch.zeros((bs, seqlen), dtype=torch.long)

    for i, ex in enumerate(batch):
        ids = ex["input_ids"][:seqlen]
        lab = ex["labels"][:seqlen]
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        labels[i, : len(lab)] = torch.tensor(lab, dtype=torch.long)
        attn[i, : len(ids)] = 1
    return input_ids, labels, attn


# ---------- train ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl", default="")
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--init_ckpt", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--micro_bsz", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")

    ap.add_argument("--add_bos", action="store_true")
    ap.add_argument("--bos_id", type=int, default=2)
    ap.add_argument("--eos_id", type=int, default=3)
    ap.add_argument("--pad_id", type=int, default=0)

    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = Tokenizer.from_file(args.tokenizer_path)

    # load init ckpt
    ckpt = torch.load(args.init_ckpt, map_location="cpu")
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    sd = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k[len("_orig_mod.") :]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.train()

    train_ds = SFTJsonlDataset(
        args.train_jsonl, tok, args.max_len, args.add_bos, args.bos_id, args.eos_id
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=args.micro_bsz,
        shuffle=True,
        collate_fn=lambda b: collate(b, args.pad_id, args.max_len),
        num_workers=0,
    )

    val_dl = None
    if args.val_jsonl:
        val_ds = SFTJsonlDataset(
            args.val_jsonl, tok, args.max_len, args.add_bos, args.bos_id, args.eos_id
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=args.micro_bsz,
            shuffle=False,
            collate_fn=lambda b: collate(b, args.pad_id, args.max_len),
            num_workers=0,
        )

    if args.precision == "bf16":
        autocast_dtype = torch.bfloat16
    elif args.precision == "fp16":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = None

    # opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9,0.95), weight_decay=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    os.makedirs(args.out_dir, exist_ok=True)

    def save(step: int):
        out = {
            "config": ckpt["config"],
            "model": model.state_dict(),
            "global_step": step,
            "meta": {"init_ckpt": args.init_ckpt},
        }
        path = os.path.join(args.out_dir, f"step_{step:06d}.pt")
        torch.save(out, path)
        torch.save(out, os.path.join(args.out_dir, "latest.pt"))
        print(f"[ckpt] saved {path}")

    def evaluate(step: int):
        if val_dl is None:
            return
        model.eval()
        losses = []
        with torch.no_grad():
            for input_ids, labels, attn in val_dl:
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                with torch.autocast(
                    "cuda",
                    dtype=autocast_dtype,
                    enabled=(autocast_dtype is not None and device.type == "cuda"),
                ):
                    out = model(input_ids)
                    logits = out["logits"] if isinstance(out, dict) else out
                    # shift for causal LM
                    logits = logits[:, :-1, :].contiguous()
                    # labels already store next-token targets at position i-1
                    tgt = labels[:, :-1].contiguous()
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        tgt.view(-1),
                        ignore_index=-100,
                    )
                losses.append(loss.item())
        print(f"[eval] step={step} val_loss={sum(losses) / max(1, len(losses)):.4f}")
        model.train()

    # train loop
    t0 = time.time()
    step = 0
    dl_it = iter(train_dl)
    opt.zero_grad(set_to_none=True)

    # ---- SANITY: verify supervision spans ----
    row = train_ds.rows[0]
    text, asst_spans = train_ds.fmt.render(row["messages"])

    enc = tok.encode(text)
    ids = enc.ids
    offsets = enc.offsets
    if args.add_bos:
        ids = [args.bos_id] + ids
        offsets = [(0, 0)] + offsets

    # truncate like dataset
    ids = ids[: args.max_len]
    offsets = offsets[: args.max_len]

    labels = [-100] * len(ids)

    def token_is_in_asst(tok_span):
        ts, te = tok_span
        if ts == te == 0:
            return False
        for a0, a1 in asst_spans:
            if te > a0 and ts < a1:
                return True
        return False

    for i, off in enumerate(offsets):
        if token_is_in_asst(off):
            labels[i] = ids[i]

    # mask BOS/EOS
    for i, t in enumerate(ids):
        if t == args.bos_id or t == args.eos_id:
            labels[i] = -100

    sup = [tid for tid, lab in zip(ids, labels) if lab != -100]

    print("TEXT_TAIL:", repr(text[-200:]))
    print("ASST_SPANS:", asst_spans[:3], "...")
    print("SUP_TOKENS:", len(sup))
    print("SUP_DECODED_HEAD:", repr(tok.decode(sup)[:200]))
    print("SUP_DECODED_TAIL:", repr(tok.decode(sup)[-200:]))

    # raise SystemExit

    while step < args.steps:
        try:
            input_ids, labels, attn = next(dl_it)
        except StopIteration:
            dl_it = iter(train_dl)
            input_ids, labels, attn = next(dl_it)

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        with torch.autocast(
            "cuda",
            dtype=autocast_dtype,
            enabled=(autocast_dtype is not None and device.type == "cuda"),
        ):
            out = model(input_ids)
            logits = out["logits"] if isinstance(out, dict) else out
            # shift for causal LM
            logits = logits[:, :-1, :].contiguous()
            # labels already store next-token targets at position i-1
            tgt = labels[:, :-1].contiguous()
            loss = (
                torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    tgt.view(-1),
                    ignore_index=-100,
                )
                / args.grad_accum
            )

        loss.backward()

        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        if step % 50 == 0:
            elapsed = time.time() - t0
            print(
                f"[train] step={step} loss={loss.item() * args.grad_accum:.4f} lr={args.lr} elapsed_s={elapsed:.1f}"
            )

        if args.eval_every and step > 0 and step % args.eval_every == 0:
            evaluate(step)

        if args.save_every and step > 0 and step % args.save_every == 0:
            save(step)

        step += 1

    save(step)
    evaluate(step)


if __name__ == "__main__":
    main()
