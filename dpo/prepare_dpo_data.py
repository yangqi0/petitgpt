#!/usr/bin/env python3

"""
Build a DPO preference-pair dataset for dpo/dpo.py.

Pulls from a small registry of open HuggingFace preference datasets,
normalizes each row to this repo's DPO schema, filters by length/dedup,
and writes OUT_DIR/train.jsonl + OUT_DIR/val.jsonl.

Output schema (one example per line), matching what dpo/dpo.py expects:
    {"messages": [{"role": "system", "content": "..."},
                   {"role": "user", "content": "..."}],
     "chosen": "preferred assistant response",
     "rejected": "dispreferred assistant response",
     "meta": {"source": "...", "bucket": "A_general" | "D_safety"}}

`meta` is informational only (mirrors the sft/distill convention of tagging
examples with a source/bucket) -- dpo/dpo.py does not currently read it.

Sources (see build_registry()):
    ultrafeedback      HuggingFaceH4/ultrafeedback_binarized  general instruction-following
    orca_dpo           Intel/orca_dpo_pairs                   general reasoning/instructions
    hh_rlhf_helpful    Anthropic/hh-rlhf (helpful-base)       general helpfulness
    hh_rlhf_harmless   Anthropic/hh-rlhf (harmless-base)      safety refusals (bucket=D_safety)

A code-domain preference source is intentionally not included here: unlike
the general sources above, there is no ready-made open code-DPO dataset that
fits this project's simple-Python-function scope, and building one would
mean a rejection-sampling pipeline (sample from the local model + teacher,
verify with distill/code_verify_v1.py, pair by pass/fail) similar in shape
to distill/, not a plain data-prep script.

Example:
    python dpo/prepare_dpo_data.py \\
      --sources ultrafeedback,orca_dpo,hh_rlhf_helpful,hh_rlhf_harmless \\
      --tokenizer_path tokenizer/tokenizer.json --out_dir datasets/dpo \\
      --seq_len 1024 --max_per_source 6000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from datasets import load_dataset
from tokenizers import Tokenizer

DEFAULT_SYSTEM = "You are a helpful assistant."
BOS_ID = 2
EOS_ID = 3

SYS_PREFIX = "System: "
USER_PREFIX = "User: "
ASSIST_PREFIX = "Assistant: "
SEP = "\n\n"


def norm_newlines(s: str) -> str:
    return (s or "").replace("\r\n", "\n").replace("\r", "\n")


def norm_for_hash(s: str) -> str:
    s = norm_newlines(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def tokenizer_auto_bos_eos(tok: Tokenizer) -> tuple[bool, bool]:
    probe = tok.encode("x").ids
    return (bool(probe) and probe[0] == BOS_ID, bool(probe) and probe[-1] == EOS_ID)


def encode_strip_special(tok: Tokenizer, text: str) -> list[int]:
    ids = tok.encode(text).ids
    if ids and ids[0] == BOS_ID:
        ids = ids[1:]
    if ids and ids[-1] == EOS_ID:
        ids = ids[:-1]
    return ids


def render_plain_with_completion(
    messages: list[dict[str, str]], completion: str, default_system: str
) -> str:
    """Render System:/User:/Assistant: text ending in `completion`, for token counting.
    Mirrors the template used by dpo/dpo.py's build_completion_example()."""
    msgs = messages[:]
    if msgs and msgs[0].get("role") != "system":
        msgs = [{"role": "system", "content": default_system}] + msgs

    out: list[str] = []
    if msgs and msgs[0].get("role") == "system":
        sys_txt = norm_newlines(msgs[0].get("content", "")).strip()
        if sys_txt:
            out.append(SYS_PREFIX + sys_txt + SEP)
        start = 1
    else:
        start = 0

    for m in msgs[start:]:
        role = m.get("role")
        txt = norm_newlines(m.get("content", ""))
        if role == "user":
            out.append(USER_PREFIX + txt.strip() + SEP)
        elif role == "assistant":
            out.append(ASSIST_PREFIX + txt + SEP)

    out.append(ASSIST_PREFIX)
    if completion:
        out.append(norm_newlines(completion))
    return "".join(out)


def count_tokens(tok: Tokenizer, messages: list[dict[str, str]], completion: str, default_system: str) -> int:
    has_bos, has_eos = tokenizer_auto_bos_eos(tok)
    n = len(encode_strip_special(tok, render_plain_with_completion(messages, completion, default_system)))
    if has_bos:
        n += 1
    if has_eos:
        n += 1
    return n


@dataclass
class Pair:
    messages: list[dict[str, str]]
    chosen: str
    rejected: str
    meta: dict[str, Any] = field(default_factory=dict)


# -------------------------
# Per-source loaders -> Iterable[Pair]
# -------------------------
def iter_ultrafeedback(split: str) -> Iterable[Pair]:
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split)
    for row in ds:
        prompt = row.get("prompt") or ""
        chosen_msgs = row.get("chosen") or []
        rejected_msgs = row.get("rejected") or []
        if not prompt.strip() or not chosen_msgs or not rejected_msgs:
            continue

        c_last = chosen_msgs[-1] if isinstance(chosen_msgs[-1], dict) else {}
        r_last = rejected_msgs[-1] if isinstance(rejected_msgs[-1], dict) else {}
        if str(c_last.get("role", "")).lower() != "assistant":
            continue
        if str(r_last.get("role", "")).lower() != "assistant":
            continue

        chosen_text = str(c_last.get("content", "") or "")
        rejected_text = str(r_last.get("content", "") or "")
        if not chosen_text.strip() or not rejected_text.strip():
            continue

        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": norm_newlines(prompt).strip()},
        ]
        yield Pair(messages, norm_newlines(chosen_text), norm_newlines(rejected_text), {"source": "ultrafeedback"})


def iter_orca_dpo(split: str) -> Iterable[Pair]:
    ds = load_dataset("Intel/orca_dpo_pairs", split=split)
    for row in ds:
        system = (row.get("system") or "").strip()
        question = (row.get("question") or "").strip()
        chosen = row.get("chosen") or ""
        rejected = row.get("rejected") or ""
        if not question or not chosen.strip() or not rejected.strip():
            continue

        messages = [
            {"role": "system", "content": norm_newlines(system) or DEFAULT_SYSTEM},
            {"role": "user", "content": norm_newlines(question)},
        ]
        yield Pair(messages, norm_newlines(chosen), norm_newlines(rejected), {"source": "orca_dpo"})


_HH_TURN_RE = re.compile(r"\n\n(Human|Assistant): ")


def _split_hh_dialogue(text: str) -> list[tuple[str, str]]:
    """Split an Anthropic hh-rlhf dialogue string into [(role, content), ...]."""
    parts = _HH_TURN_RE.split(norm_newlines(text))
    turns: list[tuple[str, str]] = []
    for i in range(1, len(parts), 2):
        role = "user" if parts[i].strip().lower() == "human" else "assistant"
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        turns.append((role, content))
    return turns


def _hh_pair_from_row(row: dict[str, Any], bucket: str) -> Pair | None:
    chosen_turns = _split_hh_dialogue(row.get("chosen", ""))
    rejected_turns = _split_hh_dialogue(row.get("rejected", ""))
    if not chosen_turns or not rejected_turns:
        return None
    if chosen_turns[-1][0] != "assistant" or rejected_turns[-1][0] != "assistant":
        return None

    prefix_c, prefix_r = chosen_turns[:-1], rejected_turns[:-1]
    if not prefix_c or prefix_c != prefix_r:
        return None
    if any(not content for _, content in prefix_c):
        return None

    messages = [{"role": "system", "content": DEFAULT_SYSTEM}]
    messages += [{"role": role, "content": content} for role, content in prefix_c]
    return Pair(messages, chosen_turns[-1][1], rejected_turns[-1][1], {"source": "hh_rlhf", "bucket": bucket})


def iter_hh_rlhf(subset: str, split: str, bucket: str) -> Iterable[Pair]:
    ds = load_dataset("Anthropic/hh-rlhf", data_dir=subset, split=split)
    for row in ds:
        pair = _hh_pair_from_row(row, bucket)
        if pair is not None:
            yield pair


@dataclass
class SourceDef:
    loader: Callable[[], Iterable[Pair]]
    hint: str


def build_registry() -> dict[str, SourceDef]:
    return {
        "ultrafeedback": SourceDef(lambda: iter_ultrafeedback("train_prefs"), "general instruction-following"),
        "orca_dpo": SourceDef(lambda: iter_orca_dpo("train"), "general reasoning/instructions"),
        "hh_rlhf_helpful": SourceDef(lambda: iter_hh_rlhf("helpful-base", "train", "A_general"), "general helpfulness"),
        "hh_rlhf_harmless": SourceDef(lambda: iter_hh_rlhf("harmless-base", "train", "D_safety"), "safety / harmlessness"),
    }


# -------------------------
# Filtering
# -------------------------
def passes_filters(pair: Pair, tok: Tokenizer, args: argparse.Namespace) -> Pair | None:
    chosen = pair.chosen.strip()
    rejected = pair.rejected.strip()
    if not chosen or not rejected:
        return None
    if chosen == rejected:
        return None
    if len(chosen) < args.min_completion_chars or len(rejected) < args.min_completion_chars:
        return None

    prompt_toks = count_tokens(tok, pair.messages, "", DEFAULT_SYSTEM)
    if prompt_toks > args.max_prompt_tokens:
        return None

    n_chosen = count_tokens(tok, pair.messages, pair.chosen, DEFAULT_SYSTEM)
    n_rejected = count_tokens(tok, pair.messages, pair.rejected, DEFAULT_SYSTEM)
    if n_chosen > args.seq_len or n_rejected > args.seq_len:
        return None
    if (n_chosen - prompt_toks) < args.min_completion_tokens:
        return None
    if (n_rejected - prompt_toks) < args.min_completion_tokens:
        return None

    return pair


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sources",
        default="ultrafeedback,orca_dpo,hh_rlhf_helpful,hh_rlhf_harmless",
        help="comma-separated source names (see build_registry())",
    )
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--out_dir", default="datasets/dpo")

    ap.add_argument("--seq_len", type=int, default=1024, help="drop pairs where prompt+completion exceeds this many tokens (should match dpo/dpo.py --seq_len)")
    ap.add_argument("--max_prompt_tokens", type=int, default=512)
    ap.add_argument("--min_completion_tokens", type=int, default=4)
    ap.add_argument("--min_completion_chars", type=int, default=4)
    ap.add_argument("--max_per_source", type=int, default=6000, help="0 = no cap")

    ap.add_argument("--val_ratio", type=float, default=0.03)
    ap.add_argument("--min_val_per_source", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer_path)
    registry = build_registry()

    names = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = [n for n in names if n not in registry]
    if unknown:
        raise ValueError(f"unknown source(s): {unknown} (available: {sorted(registry)})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"

    seen: set[str] = set()
    n_train_total = 0
    n_val_total = 0

    with open(train_path, "w", encoding="utf-8") as ftr, open(val_path, "w", encoding="utf-8") as fval:
        for name in names:
            src = registry[name]
            print(f"[*] loading source: {name} ({src.hint})")

            rows: list[Pair] = []
            try:
                for raw_pair in src.loader():
                    pair = passes_filters(raw_pair, tok, args)
                    if pair is None:
                        continue
                    prompt_text = render_plain_with_completion(pair.messages, "", DEFAULT_SYSTEM)
                    key = sha1_hex(norm_for_hash(prompt_text[-2000:] + "\x1f" + pair.chosen[:2000]))
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(pair)
            except Exception as e:
                print(f"    WARN: failed to load/process {name}: {e}")
                continue

            if not rows:
                print(f"    WARN: 0 usable pairs from {name}, skipping")
                continue

            idx = list(range(len(rows)))
            random.Random(args.seed).shuffle(idx)
            if args.max_per_source and len(idx) > args.max_per_source:
                idx = idx[: args.max_per_source]

            n_val_src = max(args.min_val_per_source, int(len(idx) * args.val_ratio))
            n_val_src = min(n_val_src, len(idx) // 5 + 1)  # never take more than ~20% as val
            val_idx = set(idx[:n_val_src])

            src_train = src_val = 0
            for i in idx:
                pair = rows[i]
                rec = {
                    "messages": pair.messages,
                    "chosen": pair.chosen,
                    "rejected": pair.rejected,
                    "meta": {**pair.meta, "bucket": pair.meta.get("bucket", "A_general")},
                }
                line = json.dumps(rec, ensure_ascii=False) + "\n"
                if i in val_idx:
                    fval.write(line)
                    src_val += 1
                else:
                    ftr.write(line)
                    src_train += 1

            print(f"    {name}: kept={len(rows)} -> train={src_train} val={src_val}")
            n_train_total += src_train
            n_val_total += src_val

    print(f"Done: train={n_train_total} -> {train_path}")
    print(f"      val={n_val_total} -> {val_path}")


if __name__ == "__main__":
    main()
