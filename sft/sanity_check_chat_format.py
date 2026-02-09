#!/usr/bin/env python3

"""
Sanity check for chat data:
1) Render canonical messages into the fixed TAG-based chat template
2) Tokenize with the trained tokenizer
3) Inspect tokens, ids, and decoded text

This verifies what the model will actually see during training.
"""

import argparse
import json
from typing import Any


def load_tokenizer(tokenizer_path: str):
    """Load a HuggingFace `tokenizers` Tokenizer from tokenizer.json."""
    from tokenizers import Tokenizer

    return Tokenizer.from_file(tokenizer_path)


def render_chat(
    messages: list[dict[str, str]],
    bos: str = "[BOS]",
    eos: str = "[EOS]",
) -> str:
    """
    Render messages using the fixed TAG-based template:

    [BOS]
    [SYS]
    ...
    [/SYS]
    [USER]
    ...
    [/USER]
    [ASSISTANT]
    ...
    [/ASSISTANT]
    [EOS]
    """
    parts: list[str] = [bos]

    idx = 0

    # Optional system message (must be first if present)
    if messages and messages[0]["role"] == "system":
        parts.append("[SYS]\n" + messages[0]["content"] + "\n[/SYS]")
        idx = 1

    # Remaining turns
    for m in messages[idx:]:
        role = m["role"]
        content = m["content"]

        if role == "user":
            parts.append("[USER]\n" + content + "\n[/USER]")
        elif role == "assistant":
            parts.append("[ASSISTANT]\n" + content + "\n[/ASSISTANT]")
        elif role == "tool":
            parts.append("[TOOL]\n" + content + "\n[/TOOL]")
        else:
            # Fallback for unexpected roles
            parts.append(f"[{role.upper()}]\n{content}\n[/{role.upper()}]")

    parts.append(eos)
    return "\n".join(parts) + "\n"


def read_jsonl(path: str, n: int) -> list[dict[str, Any]]:
    """Read the first n JSON objects from a JSONL file."""
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if len(rows) >= n:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical_jsonl", required=True, help="Canonical JSONL file")
    ap.add_argument("--tokenizer", required=True, help="Path to tokenizer.json")
    ap.add_argument("--n", type=int, default=10, help="Number of samples to inspect")
    ap.add_argument("--bos", default="[BOS]", help="BOS marker string")
    ap.add_argument("--eos", default="[EOS]", help="EOS marker string")
    ap.add_argument("--max_tokens_print", type=int, default=120, help="Max tokens to print")
    ap.add_argument("--max_decoded_chars", type=int, default=600, help="Max decoded chars to print")
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    samples = read_jsonl(args.canonical_jsonl, args.n)

    for i, s in enumerate(samples):
        messages = s["messages"]
        text = render_chat(messages, bos=args.bos, eos=args.eos)

        enc = tok.encode(text)
        tokens = enc.tokens
        ids = enc.ids
        decoded = tok.decode(ids)

        k = args.max_tokens_print

        print("=" * 80)
        print(f"Sample {i}")
        print("- Rendered text:")
        print(text)
        print("- Tokens (truncated):")
        print(tokens[:k], "..." if len(tokens) > k else "")
        print("- Token IDs (truncated):")
        print(ids[:k], "..." if len(ids) > k else "")
        print(f"- Decoded (first {args.max_decoded_chars} chars):")
        print(decoded[: args.max_decoded_chars])


if __name__ == "__main__":
    main()
