# pretrain/sample.py
# A robust sampling script for petitgpt that works even when checkpoints don't store config.
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tokenizers import Tokenizer

# Ensure project root is on sys.path so `import src.model` works no matter where you run from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_PROMPTS: List[str] = [
    "Once upon a time, ",
    "In a distant future, humans and robots ",
    "The following is a news report:\n\n",
    "Neural networks are a class of machine learning models that ",
    "Here is a short Python snippet:\n\ndef fib(n):\n    ",
]


def load_tokenizer(tokenizer_path: str) -> Tokenizer:
    return Tokenizer.from_file(tokenizer_path)


def encode_prompt(tok: Tokenizer, prompt: str, add_bos: bool, bos_id: int) -> List[int]:
    ids = tok.encode(prompt).ids
    if add_bos:
        ids = [bos_id] + ids
    return ids


def decode_ids(tok: Tokenizer, ids: List[int]) -> str:
    return tok.decode(ids)


def _autocast_dtype(precision: str) -> Optional[torch.dtype]:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return None


def _top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    probs: [B, V]
    keep smallest set with cumulative prob <= top_p (nucleus), renormalize
    """
    if not (0.0 < top_p <= 1.0):
        return probs

    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    mask = cum_probs > top_p
    mask[..., 0] = False  # keep at least 1 token
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)

    denom = sorted_probs.sum(dim=-1, keepdim=True)
    sorted_probs = torch.where(denom > 0, sorted_probs / denom, sorted_probs)

    filtered = torch.zeros_like(probs)
    filtered.scatter_(dim=-1, index=sorted_idx, src=sorted_probs)
    return filtered


def _top_k_filter(probs: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k is None or top_k <= 0:
        return probs
    top_k = min(int(top_k), probs.size(-1))
    tk_probs, tk_ids = torch.topk(probs, k=top_k, dim=-1)
    filtered = torch.zeros_like(probs)
    filtered.scatter_(dim=-1, index=tk_ids, src=tk_probs)

    denom = filtered.sum(dim=-1, keepdim=True)
    filtered = torch.where(denom > 0, filtered / denom, filtered)
    return filtered


def _sample_next_id(
    probs: torch.Tensor,
    top_p: float,
    top_k: int,
    greedy: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    probs: [B, V] softmax'ed already
    returns (next_id [B,1], filtered_probs [B,V])
    """
    filtered = probs
    filtered = _top_k_filter(filtered, top_k)
    filtered = _top_p_filter(filtered, top_p)

    # Safety fallback
    row_sum = filtered.sum(dim=-1, keepdim=True)
    filtered = torch.where(row_sum > 0, filtered / row_sum, probs)

    if greedy:
        nxt = torch.argmax(filtered, dim=-1, keepdim=True)
        return nxt.long(), filtered

    nxt = torch.multinomial(filtered, num_samples=1)
    return nxt.long(), filtered


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tok: Tokenizer,
    prompt: str,
    device: torch.device,
    max_seq_len: int,
    precision: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    eos_id: int,
    add_bos: bool,
    bos_id: int,
    min_new_tokens: int,
    greedy: bool,
    debug: bool,
    debug_topk: int = 10,
) -> Tuple[str, str, List[int], int, Dict[str, Any]]:
    ac_dtype = _autocast_dtype(precision)

    prompt_ids = encode_prompt(tok, prompt, add_bos=add_bos, bos_id=bos_id)
    prompt_len = len(prompt_ids)

    ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1,T]
    new_ids: List[int] = []

    dbg: Dict[str, Any] = {}
    if debug:
        dbg["first_step"] = {}
        try:
            dbg["bos_token"] = tok.id_to_token(bos_id)
        except Exception:
            dbg["bos_token"] = None
        try:
            dbg["eos_token"] = tok.id_to_token(eos_id)
        except Exception:
            dbg["eos_token"] = None

    for t in range(int(max_new_tokens)):
        if ids.size(1) > max_seq_len:
            ids = ids[:, -max_seq_len:]

        if ac_dtype is not None:
            with torch.autocast("cuda", dtype=ac_dtype):
                logits = model(ids)  # [1,T,V]
        else:
            logits = model(ids)

        if debug and (not torch.isfinite(logits).all()):
            dbg["non_finite_logits"] = True

        next_logits = logits[:, -1, :].float()

        if greedy or temperature == 0.0:
            probs = torch.softmax(next_logits, dim=-1)
        else:
            next_logits = next_logits / float(temperature)
            probs = torch.softmax(next_logits, dim=-1)

        if debug and t == 0:
            eos_prob = float(probs[0, eos_id].item()) if 0 <= eos_id < probs.size(-1) else None
            k = min(int(debug_topk), probs.size(-1))
            tk_probs, tk_ids = torch.topk(probs, k=k, dim=-1)
            dbg["first_step"] = {
                "eos_prob": eos_prob,
                "topk_ids": tk_ids[0].tolist(),
                "topk_probs": [float(x) for x in tk_probs[0].tolist()],
            }

        next_id, _f = _sample_next_id(probs=probs, top_p=top_p, top_k=top_k, greedy=greedy)
        nxt = int(next_id.item())

        ids = torch.cat([ids, next_id.to(ids.device)], dim=1)
        new_ids.append(nxt)

        if (t + 1) >= int(min_new_tokens) and nxt == eos_id:
            break

    full_text = decode_ids(tok, ids.squeeze(0).tolist())
    new_text = decode_ids(tok, new_ids) if new_ids else ""
    return full_text, new_text, new_ids, prompt_len, dbg


def write_samples_file(
    out_path: Path,
    header: str,
    rows: List[Tuple[str, str, str, List[int], int, Dict[str, Any]]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [header.rstrip() + "\n"]

    for i, (prompt, full_text, new_text, new_ids, prompt_len, dbg) in enumerate(rows, 1):
        lines.append("=" * 80 + "\n")
        lines.append(f"[Prompt {i}] (prompt_tokens={prompt_len})\n{prompt}\n\n")

        if dbg:
            fs = dbg.get("first_step", {}) or {}
            lines.append("[Debug]\n")
            lines.append(f"bos_id={dbg.get('bos_token')} eos_id={dbg.get('eos_token')}\n")
            if fs:
                lines.append(f"first_step_eos_prob={fs.get('eos_prob')}\n")
                lines.append(f"first_step_topk_ids={fs.get('topk_ids')}\n")
                lines.append(f"first_step_topk_probs={fs.get('topk_probs')}\n")
            if dbg.get("non_finite_logits"):
                lines.append("WARNING: non-finite logits detected!\n")
            lines.append("\n")

        lines.append(f"[New tokens {i}] count={len(new_ids)} first30={new_ids[:30]}\n")
        lines.append(f"[New text {i}] repr={repr(new_text[:500])}\n\n")
        lines.append(f"[Full output {i}]\n{full_text}\n\n")

    out_path.write_text("".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="checkpoint .pt path")
    ap.add_argument("--tokenizer", type=str, required=True, help="tokenizer.json path")
    ap.add_argument("--out", type=str, required=True, help="output txt path")

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])

    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--min_new_tokens", type=int, default=0)
    ap.add_argument("--greedy", action="store_true")

    ap.add_argument("--eos_id", type=int, default=3)
    ap.add_argument("--bos_id", type=int, default=2)
    ap.add_argument("--no_bos", action="store_true")
    ap.add_argument("--no_debug", action="store_true")

    # Model hyperparams fallback (matches your train_pretrain.py defaults)
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--n_heads", type=int, default=12)
    ap.add_argument("--d_ff", type=int, default=3072)
    ap.add_argument("--dropout", type=float, default=0.0)

    ap.add_argument("--tie_embeddings", action="store_true", default=True, help="Tie token embedding and LM head weights")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    from src.model import GPT, GPTConfig  # noqa: E402

    ckpt = torch.load(args.ckpt, map_location="cpu")

    # Determine config: use ckpt["config"] if present, else use CLI fallback.
    cfg_dict = ckpt.get("config", None)
    if cfg_dict is not None:
        cfg = GPTConfig(**cfg_dict)
    else:
        cfg = GPTConfig(
            vocab_size=args.vocab_size,
            n_layers=args.layers,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            max_seq_len=args.seq_len,
            dropout=args.dropout,
            tie_embeddings=bool(args.tie_embeddings),
        )

    model = GPT(cfg)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    def strip_prefix_if_present(state_dict, prefix="_orig_mod."):
        if not isinstance(state_dict, dict):
            return state_dict
        # detect if any key startswith prefix
        has = any(k.startswith(prefix) for k in state_dict.keys())
        if not has:
            return state_dict
        return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}

    state = strip_prefix_if_present(state, prefix="_orig_mod.")
    model.load_state_dict(state, strict=True)
    model.eval()

    device = torch.device(args.device)
    model.to(device)

    tok = load_tokenizer(args.tokenizer)

    rows: List[Tuple[str, str, str, List[int], int, Dict[str, Any]]] = []
    for p in DEFAULT_PROMPTS:
        full_text, new_text, new_ids, prompt_len, dbg = generate(
            model=model,
            tok=tok,
            prompt=p,
            device=device,
            max_seq_len=args.seq_len,
            precision=args.precision,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            eos_id=args.eos_id,
            add_bos=not args.no_bos,
            bos_id=args.bos_id,
            min_new_tokens=args.min_new_tokens,
            greedy=args.greedy,
            debug=not args.no_debug,
        )
        rows.append((p, full_text, new_text, new_ids, prompt_len, dbg))

    header = (
        f"Samples generated with tokenizer={args.tokenizer}\n"
        f"precision={args.precision}, temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}, "
        f"max_new_tokens={args.max_new_tokens}, min_new_tokens={args.min_new_tokens}, greedy={args.greedy}\n"
        f"model: vocab_size={args.vocab_size}, seq_len={args.seq_len}, layers={args.layers}, d_model={args.d_model}, "
        f"n_heads={args.n_heads}, d_ff={args.d_ff}, dropout={args.dropout}, tie_embeddings={bool(args.tie_embeddings)}\n"
    )

    write_samples_file(Path(args.out), header, rows)


if __name__ == "__main__":
    main()
