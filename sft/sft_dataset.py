#!/usr/bin/env python3

"""
SFT Dataset for canonical chat JSONL.

Input JSONL format (one sample per line):
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    ...
  ]
}

This dataset:
- Renders chat with the tag-based template
- Tokenizes with offsets
- Masks labels so only assistant CONTENT tokens contribute to the loss
- Supports length truncation and padding via a collate_fn
"""

from __future__ import annotations

import json
from typing import Any

import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100


def render_chat_tag_template(
    messages: list[dict[str, str]],
    include_bos: bool = False,
    include_eos: bool = True,
    bos: str = "[BOS]",
    eos: str = "[EOS]",
) -> str:
    """
    Render messages using the tag-based template.

    NOTE: We default include_bos=False to avoid double-BOS if tokenizer adds BOS automatically.
    """
    parts: list[str] = []
    if include_bos:
        parts.append(bos)

    idx = 0
    if messages and messages[0]["role"] == "system":
        parts.append("[SYS]\n" + messages[0]["content"] + "\n[/SYS]")
        idx = 1

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
            parts.append(f"[{role.upper()}]\n{content}\n[/{role.upper()}]")

    if include_eos:
        parts.append(eos)

    return "\n".join(parts) + "\n"


def build_sft_tensors(
    tokenizer,
    messages: list[dict[str, str]],
    max_length: int | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """
    Build (input_ids, labels, attention_mask) for one example using char-span masking.

    Only tokens overlapping assistant CONTENT are supervised (labels = input_ids),
    all other tokens are masked with IGNORE_INDEX.

    If max_length is set, we truncate to that length.
    """
    text = render_chat_tag_template(messages, include_bos=False, include_eos=True)

    enc = tokenizer.encode(text)
    input_ids = enc.ids
    offsets = enc.offsets  # list of (start_char, end_char)

    labels = [IGNORE_INDEX] * len(input_ids)

    # Find assistant content spans in the rendered text
    cursor = 0
    for m in messages:
        if m["role"] != "assistant":
            continue

        content = m["content"]
        start = text.find(content, cursor)
        if start == -1:
            continue
        end = start + len(content)
        cursor = end

        # Mark tokens whose offsets overlap [start, end)
        for i, (s, e) in enumerate(offsets):
            if e <= start:
                continue
            if s >= end:
                break
            labels[i] = input_ids[i]

    attention_mask = [1] * len(input_ids)

    if max_length is not None and len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        attention_mask = attention_mask[:max_length]

    return input_ids, labels, attention_mask


class SFTJsonlDataset(Dataset):
    """
    A simple JSONL-backed dataset.

    For large files, you can switch to an index-based dataset (byte offsets) later.
    v1 keeps it simple: load lines into memory.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        max_length: int | None = 2048,
        limit: int | None = None,
    ):
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.samples: list[dict[str, Any]] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "messages" in obj and isinstance(obj["messages"], list):
                    self.samples.append(obj)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        messages = self.samples[idx]["messages"]

        input_ids, labels, attention_mask = build_sft_tensors(
            tokenizer=self.tokenizer,
            messages=messages,
            max_length=self.max_length,
        )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def collate_sft_batch(
    batch: list[dict[str, torch.Tensor]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """
    Collate function that pads input_ids/attention_mask to the longest sequence in batch,
    and pads labels with IGNORE_INDEX.
    """
    max_len = max(x["input_ids"].shape[0] for x in batch)

    input_ids_list = []
    labels_list = []
    attention_list = []

    for x in batch:
        input_ids = x["input_ids"]
        labels = x["labels"]
        attn = x["attention_mask"]

        pad_len = max_len - input_ids.shape[0]
        if pad_len > 0:
            input_ids = torch.cat(
                [input_ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)]
            )
            labels = torch.cat([labels, torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long)])
            attn = torch.cat([attn, torch.zeros((pad_len,), dtype=torch.long)])

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_list.append(attn)

    return {
        "input_ids": torch.stack(input_ids_list, dim=0),
        "labels": torch.stack(labels_list, dim=0),
        "attention_mask": torch.stack(attention_list, dim=0),
    }
