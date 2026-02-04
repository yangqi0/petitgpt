#!/usr/bin/env python3

"""
Normalize heterogeneous chat datasets (UltraChat / OASST) into a canonical schema:

Default canonical output (minimal):
{
  "messages": [
    {"role": "system|user|assistant|tool", "content": "..."}, ...
  ]
}

Optionally keep metadata (id/source/lang) via --keep_meta.
"""

import argparse
from collections.abc import Iterable
import json
import re
from typing import Any


def read_jsonl(path: str) -> Iterable[dict[str, Any]]:
    """Yield JSON objects from a JSONL file (one JSON object per line)."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str, rows: Iterable[dict[str, Any]]) -> None:
    """Write JSON objects to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def clean_text(s: str) -> str:
    """Lightweight text cleanup: normalize newlines and strip surrounding whitespace."""
    s = s.replace("\u0000", "")
    s = re.sub(r"\r\n", "\n", s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def normalize_role(raw_role: str) -> str:
    """Map dataset-specific role names to canonical roles."""
    r = (raw_role or "").strip().lower()
    if r in ("human", "user"):
        return "user"
    if r in ("gpt", "assistant", "model"):
        return "assistant"
    if r in ("system",):
        return "system"
    if r in ("tool", "function"):
        return "tool"
    # Fallback: treat unknown as user to avoid dropping content
    return "user"


def parse_ultrachat(obj: dict[str, Any]) -> list[dict[str, str]]:
    """
    Robust UltraChat parser.

    This UltraChat variant stores the whole multi-turn dialogue in a single string field:
      obj["text"] = "user: ... assistant: ... user: ... assistant: ..."

    We parse that format first. If not present, we fall back to other common UltraChat formats
    (prompt/response, messages list, conversations list, etc.).
    """
    # -------- Case 0: single-string dialogue in `text` --------
    if isinstance(obj.get("text"), str) and obj["text"].strip():
        raw = obj["text"].strip()

        # Split by speaker tags like "user:" / "assistant:" / "system:" (case-insensitive)
        # Keep the tag in the result by capturing it.
        import re

        pattern = re.compile(r"\b(user|assistant|system)\s*:\s*", flags=re.IGNORECASE)
        parts = pattern.split(raw)

        # Example split result:
        # ["", "user", "hello ...", "assistant", "hi ...", "user", "...", ...]
        msgs: list[dict[str, str]] = []
        i = 1
        while i + 1 < len(parts):
            role_raw = parts[i]
            content = parts[i + 1]
            role = normalize_role(role_raw)
            content = clean_text(content)

            if content:
                msgs.append({"role": role, "content": content})
            i += 2

        # Basic sanity: we want at least user+assistant
        if len(msgs) >= 2:
            return msgs
        return []

    # -------- Case 1: single-turn instruction style --------
    for p_key, r_key in [
        ("prompt", "response"),
        ("instruction", "output"),
        ("input", "output"),
        ("question", "answer"),
    ]:
        if p_key in obj and r_key in obj:
            prompt = obj.get(p_key)
            resp = obj.get(r_key)
            if prompt and resp:
                return [
                    {"role": "user", "content": clean_text(str(prompt))},
                    {"role": "assistant", "content": clean_text(str(resp))},
                ]

    # -------- Case 2: messages list --------
    if isinstance(obj.get("messages"), list):
        msgs: list[dict[str, str]] = []
        for m in obj["messages"]:
            role = m.get("role")
            content = m.get("content")
            if role and content:
                msgs.append(
                    {"role": normalize_role(str(role)), "content": clean_text(str(content))}
                )
        return [m for m in msgs if m["content"]]

    # -------- Case 3: nested data.messages --------
    if isinstance(obj.get("data"), dict) and isinstance(obj["data"].get("messages"), list):
        msgs: list[dict[str, str]] = []
        for m in obj["data"]["messages"]:
            role = m.get("role")
            content = m.get("content")
            if role and content:
                msgs.append(
                    {"role": normalize_role(str(role)), "content": clean_text(str(content))}
                )
        return [m for m in msgs if m["content"]]

    # -------- Case 4: conversations / dialog / turns --------
    for key in ("conversations", "dialog", "turns"):
        if isinstance(obj.get(key), list):
            msgs: list[dict[str, str]] = []
            for m in obj[key]:
                raw_role = m.get("from") or m.get("role") or m.get("speaker")
                content = m.get("value") or m.get("content") or m.get("text")
                if raw_role and content:
                    msgs.append(
                        {"role": normalize_role(str(raw_role)), "content": clean_text(str(content))}
                    )
            return [m for m in msgs if m["content"]]

    # -------- Fallback: nothing matched --------
    return []


def parse_oasst(obj: dict[str, Any]) -> list[dict[str, str]]:
    """
    Parse OASST samples into canonical messages.

    OASST also has multiple export formats. Common patterns:
    - obj["messages"]: list of {role, content}
    - obj["conversation"]: list of {role/speaker, text/content}
    - fallback: prompt/response or instruction/output
    """
    msgs: list[dict[str, str]] = []

    if isinstance(obj.get("messages"), list):
        for m in obj["messages"]:
            role = m.get("role")
            content = m.get("content")
            if role and content:
                msgs.append(
                    {"role": normalize_role(str(role)), "content": clean_text(str(content))}
                )

    elif isinstance(obj.get("conversation"), list):
        for m in obj["conversation"]:
            raw_role = m.get("role") or m.get("speaker") or ""
            content = m.get("text") or m.get("content")
            if raw_role and content:
                msgs.append(
                    {"role": normalize_role(str(raw_role)), "content": clean_text(str(content))}
                )

    else:
        # Fallback for single-turn instruction format
        prompt = obj.get("prompt") or obj.get("instruction") or obj.get("input")
        resp = obj.get("response") or obj.get("output")
        if prompt and resp:
            msgs = [
                {"role": "user", "content": clean_text(str(prompt))},
                {"role": "assistant", "content": clean_text(str(resp))},
            ]

    msgs = [m for m in msgs if m["content"]]
    return msgs


def merge_consecutive_same_role(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Merge consecutive messages with the same role by concatenating contents.
    This makes downstream rendering/training more stable.
    """
    if not messages:
        return messages

    merged: list[dict[str, str]] = []
    cur_role = messages[0]["role"]
    cur_text = messages[0]["content"]

    for m in messages[1:]:
        role = m["role"]
        text = m["content"]
        if role == cur_role:
            # Concatenate with a blank line to preserve readability
            cur_text = (cur_text + "\n\n" + text).strip()
        else:
            merged.append({"role": cur_role, "content": cur_text})
            cur_role = role
            cur_text = text

    merged.append({"role": cur_role, "content": cur_text})
    return merged


def build_canonical(
    obj: dict[str, Any],
    source: str,
    lang: str | None,
    keep_meta: bool,
    default_system: str | None,
) -> dict[str, Any]:
    """Convert a raw sample into the canonical schema."""
    if source == "ultrachat":
        messages = parse_ultrachat(obj)
    elif source == "oasst":
        messages = parse_oasst(obj)
    else:
        raise ValueError(f"Unsupported source: {source}")

    # Minimal validation: need at least a user+assistant pair (or more)
    if len(messages) < 2:
        return {}

    # Ensure roles are canonical and content is clean
    for m in messages:
        m["role"] = normalize_role(m.get("role", "user"))
        m["content"] = clean_text(m.get("content", ""))

    # Merge consecutive messages with the same role
    messages = merge_consecutive_same_role(messages)

    # OPTIONAL: prepend a default system message if missing
    if default_system is not None:
        if not messages or messages[0]["role"] != "system":
            messages = [{"role": "system", "content": default_system}] + messages

    out: dict[str, Any] = {"messages": messages}

    if keep_meta:
        # Keep a few useful fields if present
        cid = obj.get("id") or obj.get("conversation_id") or obj.get("uuid")
        if cid is not None:
            out["id"] = cid
        out["source"] = source
        if lang is not None:
            out["lang"] = lang

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSONL file")
    ap.add_argument("--output", required=True, help="Output canonical JSONL file")
    ap.add_argument(
        "--source", required=True, choices=["ultrachat", "oasst"], help="Dataset source type"
    )
    ap.add_argument(
        "--lang", default=None, help="Optional language tag to attach when --keep_meta is enabled"
    )
    ap.add_argument("--max_rows", type=int, default=None, help="Limit number of rows processed")
    ap.add_argument("--keep_meta", action="store_true", help="Keep id/source/lang fields in output")
    ap.add_argument(
        "--default_system", default=None, help="If set, prepend a system message when missing"
    )
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    for i, obj in enumerate(read_jsonl(args.input)):
        if args.max_rows is not None and i >= args.max_rows:
            break
        try:
            can = build_canonical(obj, args.source, args.lang, args.keep_meta, args.default_system)
            if can:
                rows.append(can)
        except Exception:
            # v1: skip parsing failures (we can tighten this later)
            continue

    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
