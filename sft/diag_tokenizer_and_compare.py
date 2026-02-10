#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import sys

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import GPT, GPTConfig


def infer_vocab_size_from_tokenizer_json(path: str) -> int:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    model = obj.get("model", {})
    vocab = model.get("vocab", None)
    if isinstance(vocab, dict):
        return len(vocab)
    if isinstance(vocab, list):
        return len(vocab)
    # fallback
    added = obj.get("added_tokens", [])
    if isinstance(added, list) and added:
        return max(int(t.get("id", -1)) for t in added) + 1
    raise ValueError(f"Cannot infer vocab_size from tokenizer.json: {path}")


def load_tokenizer(tok_path: str) -> Tokenizer:
    return Tokenizer.from_file(tok_path)


def strip_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        return {k[len("_orig_mod.") :]: v for k, v in sd.items()}
    return sd


def get_tok_emb_shape(sd: Dict[str, torch.Tensor]) -> Tuple[str, Tuple[int, int]]:
    for k in ["tok_emb.weight", "token_embedding.weight", "transformer.wte.weight", "wte.weight", "embed_tokens.weight"]:
        if k in sd:
            w = sd[k]
            return k, (int(w.shape[0]), int(w.shape[1]))
    for k, v in sd.items():
        if v.ndim == 2 and v.shape[0] > 1000:
            return k, (int(v.shape[0]), int(v.shape[1]))
    raise KeyError("No token embedding weight found in state_dict.")


def any_nan_inf(sd: Dict[str, torch.Tensor]) -> Tuple[bool, bool]:
    has_nan = False
    has_inf = False
    for _, v in sd.items():
        if not torch.is_floating_point(v):
            continue
        if torch.isnan(v).any().item():
            has_nan = True
        if torch.isinf(v).any().item():
            has_inf = True
    return has_nan, has_inf


@torch.no_grad()
def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float, top_k: int) -> int:
    # logits: [V]
    if temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits / float(temperature)
    probs = F.softmax(logits, dim=-1)

    # top_k
    if top_k and top_k > 0:
        k = min(int(top_k), probs.numel())
        topk_probs, topk_idx = torch.topk(probs, k=k)
        probs2 = torch.zeros_like(probs)
        probs2[topk_idx] = topk_probs
        probs = probs2 / probs2.sum().clamp_min(1e-12)

    # top_p
    if top_p and top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cdf = torch.cumsum(sorted_probs, dim=-1)
        keep = cdf <= float(top_p)
        keep[..., 0] = True
        kept_idx = sorted_idx[keep]
        probs2 = torch.zeros_like(probs)
        probs2[kept_idx] = probs[kept_idx]
        probs = probs2 / probs2.sum().clamp_min(1e-12)

    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate_ids(
    model: torch.nn.Module,
    prompt_ids: List[int],
    *,
    device: torch.device,
    max_seq_len: int,
    max_new_tokens: int,
    min_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    eos_id: int,
) -> Tuple[List[int], Dict[str, float]]:
    model.eval()
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, :]
    if x.shape[1] > max_seq_len:
        x = x[:, -max_seq_len:]

    out = x
    stats = {"logits_mean": 0.0, "logits_std": 0.0, "logits_nan": 0.0, "logits_inf": 0.0}
    # one-step logits stats on initial context
    logits0 = model(out)[:, -1, :]
    stats["logits_nan"] = float(torch.isnan(logits0).any().item())
    stats["logits_inf"] = float(torch.isinf(logits0).any().item())
    stats["logits_mean"] = float(logits0.mean().item())
    stats["logits_std"] = float(logits0.std().item())

    for step in range(max_new_tokens):
        if out.shape[1] >= max_seq_len:
            break
        logits = model(out)[:, -1, :]  # [1,V]
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            # stop early if numerically broken
            break
        nxt = sample_next_token(logits[0], temperature, top_p, top_k)
        out = torch.cat([out, torch.tensor([[nxt]], device=device)], dim=1)
        if step + 1 >= min_new_tokens and nxt == eos_id:
            break

    return out[0].tolist(), stats


def build_model_for_ckpt(tok_path: str, seq_len: int, ckpt_sd: Dict[str, torch.Tensor], device: torch.device) -> torch.nn.Module:
    vocab_size = infer_vocab_size_from_tokenizer_json(tok_path)
    emb_k, (V, D) = get_tok_emb_shape(ckpt_sd)

    # sanity: vocab in tokenizer should match ckpt embedding vocab
    if V != vocab_size:
        raise RuntimeError(f"Vocab mismatch: ckpt {emb_k} has V={V}, tokenizer vocab_size={vocab_size}")

    # NOTE: adapt if your architecture differs
    cfg = GPTConfig(
        vocab_size=vocab_size,
        n_layers=12,
        d_model=768,
        n_heads=12,
        d_ff=3072,
        max_seq_len=seq_len,
        dropout=0.0,
        tie_embeddings=True,
    )
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt_sd, strict=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--pretrain_ckpt", required=True)
    ap.add_argument("--sft_ckpt", required=True)
    ap.add_argument("--prompt", default="Explain gradient descent in simple terms.")
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--min_new_tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--bos_id", type=int, default=2)
    ap.add_argument("--eos_id", type=int, default=3)
    ap.add_argument("--add_bos", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # (C) tokenizer checks
    vocab_size = infer_vocab_size_from_tokenizer_json(args.tokenizer_path)
    tok = load_tokenizer(args.tokenizer_path)
    enc = tok.encode(args.prompt)
    ids = enc.ids

    print("=== (C) TOKENIZER CHECKS ===")
    print(f"tokenizer_path: {args.tokenizer_path}")
    print(f"vocab_size: {vocab_size}")
    print(f"bos_id/eos_id: {args.bos_id}/{args.eos_id}")
    print(f"bos/eos in range: {0 <= args.bos_id < vocab_size} / {0 <= args.eos_id < vocab_size}")
    print(f"prompt_len_tokens: {len(ids)}")
    print(f"prompt_ids_max: {max(ids) if ids else None}")
    if ids and max(ids) >= vocab_size:
        print("!! ERROR: prompt has token id >= vocab_size (tokenizer inconsistency)")
    print("prompt_ids (first 40):", ids[:40])
    print("prompt_decoded_roundtrip:", tok.decode(ids))
    print()

    # load ckpts
    pre_raw = torch.load(args.pretrain_ckpt, map_location="cpu")
    sft_raw = torch.load(args.sft_ckpt, map_location="cpu")
    pre_sd = strip_prefix(pre_raw["model"])
    sft_sd = strip_prefix(sft_raw["model"])

    print("=== CKPT HEALTH ===")
    pre_nan, pre_inf = any_nan_inf(pre_sd)
    sft_nan, sft_inf = any_nan_inf(sft_sd)
    print(f"pretrain has_nan={pre_nan} has_inf={pre_inf}")
    print(f"sft      has_nan={sft_nan} has_inf={sft_inf}")
    print()

    # build models
    pre = build_model_for_ckpt(args.tokenizer_path, args.seq_len, pre_sd, device)
    sft = build_model_for_ckpt(args.tokenizer_path, args.seq_len, sft_sd, device)

    # (A) generation compare
    prompt_ids = ([args.bos_id] + ids) if args.add_bos else ids

    pre_out_ids, pre_stats = generate_ids(
        pre,
        prompt_ids,
        device=device,
        max_seq_len=args.seq_len,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        eos_id=args.eos_id,
    )

    sft_out_ids, sft_stats = generate_ids(
        sft,
        prompt_ids,
        device=device,
        max_seq_len=args.seq_len,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        eos_id=args.eos_id,
    )

    # strip prompt for display
    pre_gen_ids = pre_out_ids[len(prompt_ids):]
    sft_gen_ids = sft_out_ids[len(prompt_ids):]

    print("=== (A) GENERATION COMPARE ===")
    print(f"sampling: temp={args.temperature} top_p={args.top_p} top_k={args.top_k} "
          f"min_new={args.min_new_tokens} max_new={args.max_new_tokens}")
    print("--- PRETRAIN logits stats ---", pre_stats)
    print("--- SFT      logits stats ---", sft_stats)
    print()

    print("[PRETRAIN] gen_ids (first 60):", pre_gen_ids[:60])
    print("[PRETRAIN] text:\n", tok.decode(pre_out_ids))
    print()
    print("[SFT] gen_ids (first 60):", sft_gen_ids[:60])
    print("[SFT] text:\n", tok.decode(sft_out_ids))


if __name__ == "__main__":
    main()
