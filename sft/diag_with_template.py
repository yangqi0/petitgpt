#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import GPT, GPTConfig

BOS_ID = 2
EOS_ID = 3

MARKERS = [
    "[SYS]", "[/SYS]",
    "[USER]", "[/USER]",
    "[ASSISTANT]",
]

def infer_vocab_size(tok_json: str) -> int:
    obj = json.loads(Path(tok_json).read_text(encoding="utf-8"))
    vocab = obj["model"]["vocab"]
    return len(vocab) if isinstance(vocab, dict) else len(vocab)

def strip_bos_eos(ids: List[int]) -> List[int]:
    if ids and ids[0] == BOS_ID:
        ids = ids[1:]
    if ids and ids[-1] == EOS_ID:
        ids = ids[:-1]
    return ids

@torch.no_grad()
def sample_next(logits, temperature, top_p, top_k):
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / float(temperature)
    probs = F.softmax(logits, dim=-1)

    if top_k and top_k > 0:
        k = min(int(top_k), probs.numel())
        p2, idx = torch.topk(probs, k=k)
        probs = torch.zeros_like(probs).scatter_(0, idx, p2)
        probs = probs / probs.sum().clamp_min(1e-12)

    if top_p and top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cdf = torch.cumsum(sp, dim=-1)
        keep = cdf <= float(top_p)
        keep[0] = True
        kept = si[keep]
        probs2 = torch.zeros_like(probs)
        probs2[kept] = probs[kept]
        probs = probs2 / probs2.sum().clamp_min(1e-12)

    return int(torch.multinomial(probs, 1).item())

@torch.no_grad()
def generate(model, tok, prompt_text, max_seq_len, max_new, min_new, temperature, top_p, top_k):
    # remove trailing EOS from prompt for continuation
    prompt_ids = strip_bos_eos(tok.encode(prompt_text).ids)

    # if tokenizer auto-adds BOS, keep BOS for consistent behavior
    test = tok.encode("x").ids
    if test and test[0] == BOS_ID:
        prompt_ids = [BOS_ID] + prompt_ids

    x = torch.tensor(prompt_ids, dtype=torch.long, device=next(model.parameters()).device)[None, :]
    if x.shape[1] > max_seq_len:
        x = x[:, -max_seq_len:]

    out = x
    for t in range(max_new):
        if out.shape[1] >= max_seq_len:
            break
        logits = model(out)[:, -1, :][0]
        nxt = sample_next(logits, temperature, top_p, top_k)
        out = torch.cat([out, torch.tensor([[nxt]], device=out.device)], dim=1)
        if t + 1 >= min_new and nxt == EOS_ID:
            break

    out_ids = out[0].tolist()
    if out_ids and out_ids[0] == BOS_ID:
        out_ids = out_ids[1:]
    if out_ids and out_ids[-1] == EOS_ID:
        out_ids = out_ids[:-1]
    return out_ids, tok.decode(out_ids)

def build_model(ckpt_path, tok_json, seq_len, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k[len("_orig_mod.") :]: v for k, v in sd.items()}

    vocab = infer_vocab_size(tok_json)
    cfg = GPTConfig(
        vocab_size=vocab,
        n_layers=12,
        d_model=768,
        n_heads=12,
        d_ff=3072,
        max_seq_len=seq_len,
        dropout=0.0,
        tie_embeddings=True,
    )
    m = GPT(cfg).to(device)
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m

def check_marker(tok: Tokenizer, marker: str):
    enc = tok.encode(marker)
    ids = enc.ids
    pieces = enc.tokens
    ok_single = (len(ids) == 1)  # ideal if special token
    return ok_single, ids, pieces, tok.decode(ids)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--max_new", type=int, default=200)
    ap.add_argument("--min_new", type=int, default=32)
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer_path)

    print("=== Marker tokenization ===")
    any_bad = False
    for m in MARKERS:
        ok, ids, pieces, rt = check_marker(tok, m)
        if not ok:
            any_bad = True
        print(f"{m:12s} single_token={ok} ids={ids} pieces={pieces} roundtrip={rt}")

    if any_bad:
        print("\n!! Your chat markers are NOT single special tokens.")
        print("   Using [SYS]/[USER]/... as template will likely corrupt training/generation.\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.ckpt, args.tokenizer_path, args.seq_len, device)

    natural = "Write a short story about a small robot who learns to be kind."
    templ = "[SYS]\nYou are a helpful assistant.\n[/SYS]\n[USER]\nWrite a short story about a small robot who learns to be kind.\n[/USER]\n[ASSISTANT]\n"

    print("=== Tokenization preview ===")
    for name, p in [("NATURAL", natural), ("TEMPLATE", templ)]:
        enc = tok.encode(p)
        print(f"\n[{name}] len={len(enc.ids)} max_id={max(enc.ids) if enc.ids else None}")
        print("tokens(first 40):", enc.tokens[:40])
        print("ids(first 40):", enc.ids[:40])

    print("\n=== Generation ===")
    for name, p in [("NATURAL", natural), ("TEMPLATE", templ)]:
        out_ids, out_text = generate(model, tok, p, args.seq_len, args.max_new, args.min_new,
                                     args.temperature, args.top_p, args.top_k)
        print(f"\n[{name}] out_ids(first 60)={out_ids[:60]}")
        print(out_text)

if __name__ == "__main__":
    main()
