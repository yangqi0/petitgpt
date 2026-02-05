# pretrain/sample.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from tokenizers import Tokenizer

DEFAULT_PROMPTS: List[str] = [
    "Write a short story about a small robot who learns to be kind.\n\nStory:\n",
    "Q: Why is the sky blue?\nA:",
    "Instruction: Explain gradient descent in simple terms.\nResponse:",
    "Write a Python function that checks if a string is a palindrome.\n\n```python\n",
    "Summarize the following text in 3 bullet points:\n\nDeep learning has transformed many fields by learning representations from data...\n\nSummary:\n",
]


def load_tokenizer(tokenizer_path: str) -> Tokenizer:
    """Load a HuggingFace tokenizers tokenizer.json."""
    return Tokenizer.from_file(tokenizer_path)


def encode_prompt(tok: Tokenizer, prompt: str, add_bos: bool, bos_id: int) -> List[int]:
    ids = tok.encode(prompt).ids
    if add_bos:
        ids = [bos_id] + ids
    return ids


def decode_ids(tok: Tokenizer, ids: List[int]) -> str:
    return tok.decode(ids)


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tok: Tokenizer,
    prompt: str,
    device: torch.device,
    max_seq_len: int,
    precision: str = "bf16",
    temperature: float = 0.9,
    top_p: float = 0.95,
    max_new_tokens: int = 200,
    eos_id: int = 3,
    add_bos: bool = True,
    bos_id: int = 2,
) -> Tuple[str, str, List[int], int]:
    """
    Generate text from a prompt using top-p sampling.
    Returns:
      full_text: decoded prompt + generated
      new_text: decoded ONLY generated tokens (may include weird whitespace)
      new_token_ids: list of generated token ids
      prompt_len: number of prompt tokens
    """
    if precision == "bf16":
        ac_dtype = torch.bfloat16
    elif precision == "fp16":
        ac_dtype = torch.float16
    else:
        ac_dtype = None

    prompt_ids = encode_prompt(tok, prompt, add_bos=add_bos, bos_id=bos_id)
    prompt_len = len(prompt_ids)

    ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]
    new_token_ids: List[int] = []

    for _ in range(max_new_tokens):
        # Always keep the last max_seq_len tokens as context (causal LM)
        if ids.size(1) > max_seq_len:
            ids = ids[:, -max_seq_len:]

        if ac_dtype is not None:
            with torch.autocast("cuda", dtype=ac_dtype):
                logits = model(ids)  # [1, T, V]
        else:
            logits = model(ids)

        next_logits = logits[:, -1, :] / temperature
        probs = torch.softmax(next_logits, dim=-1)

        # Top-p nucleus sampling
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)

        cutoff = cum_probs > top_p
        cutoff[..., 0] = False
        sorted_probs[cutoff] = 0.0
        sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-12)

        sampled = torch.multinomial(sorted_probs, num_samples=1)  # [1, 1]
        next_id = torch.gather(sorted_idx, dim=-1, index=sampled)  # [1, 1]
        next_int = int(next_id.item())

        ids = torch.cat([ids, next_id], dim=1)
        new_token_ids.append(next_int)

        if next_int == eos_id:
            break

    full_ids = ids.squeeze(0).tolist()

    # Decode full text and the "new part" separately.
    # This helps debug cases where decoding hides whitespace/control chars.
    full_text = decode_ids(tok, full_ids)

    # The new part is the tokens generated after the prompt.
    # NOTE: if prompt got truncated due to max_seq_len, this "new part" is still correct
    # because it's based on new_token_ids, not on slicing full_ids.
    new_text = decode_ids(tok, new_token_ids) if len(new_token_ids) > 0 else ""

    return full_text, new_text, new_token_ids, prompt_len


def write_samples_file(
    out_path: Path,
    header: str,
    generations: List[Tuple[str, str, str, List[int], int]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append(header.rstrip() + "\n")

    for i, (prompt, full_text, new_text, new_ids, prompt_len) in enumerate(generations, 1):
        lines.append("=" * 80 + "\n")
        lines.append(f"[Prompt {i}] (prompt_tokens={prompt_len})\n{prompt}\n\n")

        lines.append(f"[New tokens {i}] count={len(new_ids)} first30={new_ids[:30]}\n")
        # Use repr() so invisible whitespace/control chars become visible in the file
        lines.append(f"[New text {i}] repr={repr(new_text[:500])}\n\n")

        lines.append(f"[Full output {i}]\n{full_text}\n\n")

    out_path.write_text("".join(lines), encoding="utf-8")


@torch.no_grad()
def generate_default_samples(
    model: torch.nn.Module,
    tokenizer_path: str,
    device: torch.device,
    max_seq_len: int,
    precision: str,
    out_path: Path,
    temperature: float = 0.9,
    top_p: float = 0.95,
    max_new_tokens: int = 200,
    eos_id: int = 3,
    add_bos: bool = True,
    bos_id: int = 2,
) -> None:
    """Generate 5 default samples and write to out_path."""
    tok = load_tokenizer(tokenizer_path)
    generations: List[Tuple[str, str, str, List[int], int]] = []

    for p in DEFAULT_PROMPTS:
        full_text, new_text, new_ids, prompt_len = generate(
            model=model,
            tok=tok,
            prompt=p,
            device=device,
            max_seq_len=max_seq_len,
            precision=precision,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            eos_id=eos_id,
            add_bos=add_bos,
            bos_id=bos_id,
        )
        generations.append((p, full_text, new_text, new_ids, prompt_len))

    header = (
        f"Samples generated with tokenizer={tokenizer_path}\n"
        f"precision={precision}, temperature={temperature}, top_p={top_p}, max_new_tokens={max_new_tokens}\n"
    )
    write_samples_file(out_path, header, generations)
