#!/usr/bin/env python3

"""
Build SFT training examples from canonical chat messages.

We render a full conversation into text, tokenize it, then create labels such that
only assistant spans contribute to the loss (others are masked with -100).

This is the core logic for chat SFT.
"""

from __future__ import annotations

from dataclasses import dataclass

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

    If include_bos=False, we do NOT insert the literal "[BOS]" into text.
    This avoids double-BOS when tokenizer already adds BOS automatically.
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


def find_subsequence_spans(haystack: list[int], needle: list[int]) -> list[tuple[int, int]]:
    """
    Find all (start, end) spans where `needle` occurs in `haystack`.
    end is exclusive.
    Naive O(n*m) search is OK for v1; optimize later if needed.
    """
    if not needle:
        return []
    spans: list[tuple[int, int]] = []
    m = len(needle)
    for i in range(0, len(haystack) - m + 1):
        if haystack[i : i + m] == needle:
            spans.append((i, i + m))
    return spans


@dataclass
class SFTExample:
    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]
    rendered_text: str | None = None


def build_sft_example(
    tokenizer,
    messages: list[dict[str, str]],
    keep_rendered_text: bool = False,
) -> SFTExample:
    """
    Build one SFT example using character-span based masking.

    Only tokens corresponding to assistant CONTENT (not tags) are supervised.
    This approach is robust to tokenizer boundary effects.
    """
    full_text = render_chat_tag_template(messages, include_bos=False, include_eos=True)

    # Tokenize with offsets (char-level alignment)
    enc = tokenizer.encode(full_text)
    input_ids = enc.ids
    offsets = enc.offsets  # List[(start_char, end_char)]

    labels = [IGNORE_INDEX] * len(input_ids)

    # Walk through rendered text to find assistant content spans
    cursor = 0
    for m in messages:
        if m["role"] != "assistant":
            continue

        # Find the assistant content in the rendered text
        # We search for the FIRST occurrence after cursor to avoid collisions
        content = m["content"]
        start = full_text.find(content, cursor)
        if start == -1:
            # Should not happen in normal cases; skip if it does
            continue
        end = start + len(content)
        cursor = end

        # Mark tokens whose char span overlaps [start, end)
        for i, (s, e) in enumerate(offsets):
            if e <= start:
                continue
            if s >= end:
                break
            # Overlap
            labels[i] = input_ids[i]

    attention_mask = [1] * len(input_ids)

    return SFTExample(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        rendered_text=full_text if keep_rendered_text else None,
    )
