#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_sft_mix_v5_split_local.py

Build SFT mix from a YAML config -> OUT_DIR/train.jsonl + OUT_DIR/val.jsonl

This version adds:
  - source-level max_examples / max_examples_train / max_examples_val
  - allow_if_prompt_contains / ban_if_prompt_contains
  - jsonl_split sources with explicit train_path / val_path

It is intended for small local code sources that must be manually split before mix building.
"""
from __future__ import annotations

import argparse
import ast as pyast
import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml
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


def ensure_messages(obj: Any) -> Optional[List[Dict[str, str]]]:
    if not isinstance(obj, list):
        return None
    out: List[Dict[str, str]] = []
    for m in obj:
        if not isinstance(m, dict):
            return None
        if "role" in m and "content" in m:
            role = str(m["role"]).strip().lower()
            if role not in ("system", "user", "assistant"):
                return None
            out.append({"role": role, "content": norm_newlines(str(m["content"]))})
            continue
        if "from" in m and "value" in m:
            fr = str(m["from"]).strip().lower()
            role = (
                "user" if fr in ("human", "user") else
                "assistant" if fr in ("gpt", "assistant", "bot") else
                "system" if fr == "system" else
                ""
            )
            if role == "":
                return None
            out.append({"role": role, "content": norm_newlines(str(m["value"]))})
            continue
        return None
    return out if out else None


def canon_from_instruction(instruction: str, inp: str, output: str) -> List[Dict[str, str]]:
    instruction = norm_newlines(instruction).strip()
    inp = norm_newlines(inp).strip()
    output = norm_newlines(output)
    user = instruction if not inp else (instruction + "\n\n" + inp)
    return [
        {"role": "system", "content": DEFAULT_SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": output},
    ]


def extract_first_code_block(text: str) -> Optional[str]:
    t = norm_newlines(text)
    m = re.search(r"```(?:python)?\s*\n(.*?)\n```", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"```(?:python)?\s*(.*?)```", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def passes_ast_parse(text: str) -> bool:
    code = extract_first_code_block(text)
    if code is None:
        code = text
    try:
        pyast.parse(code)
        return True
    except Exception:
        return False


def tokenizer_auto_bos_eos(tok: Tokenizer) -> Tuple[bool, bool]:
    probe = tok.encode("x").ids
    return (bool(probe) and probe[0] == BOS_ID, bool(probe) and probe[-1] == EOS_ID)


def encode_strip_special(tok: Tokenizer, text: str) -> List[int]:
    ids = tok.encode(text).ids
    if ids and ids[0] == BOS_ID:
        ids = ids[1:]
    if ids and ids[-1] == EOS_ID:
        ids = ids[:-1]
    return ids


def render_plain(messages: List[Dict[str, str]], default_system: str = DEFAULT_SYSTEM) -> str:
    msgs = messages[:]
    if msgs and msgs[0]["role"] != "system":
        msgs = [{"role": "system", "content": default_system}] + msgs
    out: List[str] = []
    if msgs and msgs[0]["role"] == "system":
        sys_txt = norm_newlines(msgs[0]["content"]).strip()
        if sys_txt:
            out.append(SYS_PREFIX + sys_txt + SEP)
        start = 1
    else:
        start = 0
    for m in msgs[start:]:
        role = m["role"]
        txt = norm_newlines(m["content"])
        if role == "user":
            out.append(USER_PREFIX + txt.strip() + SEP)
        elif role == "assistant":
            out.append(ASSIST_PREFIX + txt + SEP)
    return "".join(out)


def count_tokens(tok: Tokenizer, messages: List[Dict[str, str]]) -> int:
    has_bos, has_eos = tokenizer_auto_bos_eos(tok)
    n = len(encode_strip_special(tok, render_plain(messages)))
    if has_bos:
        n += 1
    if has_eos:
        n += 1
    return n


@dataclass
class SourceDef:
    dataset_id: str
    split: str
    subset: Optional[str] = None
    streaming: bool = False


REG: Dict[str, SourceDef] = {
    "smol_smoltalk": SourceDef("HuggingFaceTB/smol-smoltalk", "train"),
    "dolly_15k": SourceDef("databricks/databricks-dolly-15k", "train"),
    "codealpaca_20k": SourceDef("sahil2801/CodeAlpaca-20k", "train"),
    "viscode_200k": SourceDef("TIGER-Lab/VisCode-200K", "train"),
    "no_robots": SourceDef("HuggingFaceH4/no_robots", "train"),
    "alpaca_cleaned": SourceDef("yahma/alpaca-cleaned", "train"),
}


def convert_row(source: str, row: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    msgs = ensure_messages(row.get("messages"))
    if msgs is not None:
        return msgs

    if source == "codealpaca_20k":
        ins, inp, out = row.get("instruction", ""), row.get("input", ""), row.get("output", "")
        if isinstance(ins, str) and isinstance(out, str) and ins.strip() and out.strip():
            if isinstance(inp, str) and inp.strip().lower() in ("<noinput>", "noinput", "none"):
                inp = ""
            return canon_from_instruction(ins, inp if isinstance(inp, str) else "", out)

    if source == "dolly_15k":
        ins = row.get("instruction", "") or row.get("prompt", "")
        ctx = row.get("context", "") or row.get("input", "")
        out = row.get("response", "") or row.get("output", "")
        if isinstance(ins, str) and isinstance(out, str) and ins.strip() and out.strip():
            return canon_from_instruction(ins, str(ctx) if ctx is not None else "", out)

    if "instruction" in row and "output" in row:
        ins, out = row.get("instruction", ""), row.get("output", "")
        inp = row.get("input", "") if isinstance(row.get("input", ""), str) else ""
        if isinstance(ins, str) and isinstance(out, str) and ins.strip() and out.strip():
            return canon_from_instruction(ins, inp, out)

    if "prompt" in row and "response" in row:
        p, r = row.get("prompt", ""), row.get("response", "")
        if isinstance(p, str) and isinstance(r, str) and p.strip() and r.strip():
            return canon_from_instruction(p, "", r)

    return None


def get_user_prompt_text(msgs: List[Dict[str, str]]) -> str:
    parts = []
    for m in msgs:
        if m.get("role") == "user":
            parts.append(norm_newlines(str(m.get("content", ""))))
    return "\n\n".join(parts).strip().lower()


def contains_any(text: str, patterns: List[str]) -> bool:
    if not patterns:
        return False
    t = (text or "").lower()
    return any(p.lower() in t for p in patterns if p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    os.makedirs(args.out_dir, exist_ok=True)
    tok = Tokenizer.from_file(args.tokenizer_path)

    token_budget = cfg["token_budget"]
    target_train_tokens = int(token_budget.get("target_train_tokens", 200_000_000))
    val_ratio = float(token_budget.get("val_ratio", 0.002))
    min_val_tokens_per_source = int(token_budget.get("min_val_tokens_per_source", 200_000))

    mix = cfg["mixture_by_token"]
    buckets = cfg["buckets"]
    gf = cfg.get("global_filters", {})
    ban_tokens = set(gf.get("ban_tokens_in_assistant", []))
    max_total = int(gf.get("max_total_tokens_per_example", 2048))
    drop_short_chars = int(gf.get("drop_if_too_short_chars", 16))

    targets: List[Tuple[str, str, int, Dict[str, Any]]] = []
    for bucket, bw in mix.items():
        bucket_tokens = int(target_train_tokens * float(bw))
        for s in buckets[bucket]["sources"]:
            w = float(s.get("weight_in_bucket", 0.0))
            if w <= 0:
                continue
            up = float(s.get("upsample", 1.0) or 1.0)
            targets.append((bucket, s["name"], int(bucket_tokens * w * up), s))

    val_total = int(target_train_tokens * val_ratio)
    sum_train = max(1, sum(t[2] for t in targets))
    val_targets: List[Tuple[str, str, int, Dict[str, Any]]] = []
    for bucket, name, ttrain, spec in targets:
        vt = int(val_total * (ttrain / sum_train))
        vt = max(vt, min_val_tokens_per_source)
        val_targets.append((bucket, name, vt, spec))

    out_train = os.path.join(args.out_dir, "train.jsonl")
    out_val = os.path.join(args.out_dir, "val.jsonl")

    seen: set[str] = set()

    def trunc_rounds(msgs: List[Dict[str, str]], max_rounds: Optional[int]) -> List[Dict[str, str]]:
        if not max_rounds:
            return msgs
        sys_msg = msgs[0] if msgs and msgs[0]["role"] == "system" else None
        rest = msgs[1:] if sys_msg else msgs
        rest = rest[: 2 * int(max_rounds)]
        return ([sys_msg] + rest) if sys_msg else rest

    def prompt_passes_filters(msgs: List[Dict[str, str]], spec: Dict[str, Any]) -> bool:
        user_prompt = get_user_prompt_text(msgs)
        allow_list = list(spec.get("allow_if_prompt_contains", []) or [])
        ban_list = list(spec.get("ban_if_prompt_contains", []) or [])
        if allow_list and not contains_any(user_prompt, allow_list):
            return False
        if ban_list and contains_any(user_prompt, ban_list):
            return False
        return True

    def ok_example(msgs: List[Dict[str, str]], spec: Dict[str, Any]) -> Optional[Tuple[int, List[Dict[str, str]]]]:
        if not prompt_passes_filters(msgs, spec):
            return None

        out: List[Dict[str, str]] = []
        for m in msgs:
            role = str(m.get("role", "")).strip().lower()
            if role not in ("system", "user", "assistant"):
                return None
            content = norm_newlines(str(m.get("content", "")))
            if role == "assistant":
                for bad in ban_tokens:
                    if bad and bad in content:
                        return None
                if drop_short_chars and len(content.strip()) < drop_short_chars:
                    return None
                ma = spec.get("max_assistant_tokens", None)
                if ma is not None and len(encode_strip_special(tok, content)) > int(ma):
                    return None
                if spec.get("require_ast_parse", False) and not passes_ast_parse(content):
                    return None
            out.append({"role": role, "content": content})

        out = trunc_rounds(out, spec.get("max_rounds", None))
        while out and out[-1]["role"] == "user":
            out.pop()
        if not out or out[0]["role"] != "system":
            out = [{"role": "system", "content": DEFAULT_SYSTEM}] + out
        if not any(m["role"] == "user" for m in out):
            return None
        if not any(m["role"] == "assistant" for m in out):
            return None
        n = count_tokens(tok, out)
        if n > max_total:
            return None

        text = render_plain(out)
        tail = text[-3000:] if len(text) > 3000 else text
        h = sha1_hex(norm_for_hash(tail))
        if h in seen:
            return None
        seen.add(h)
        return n, out

    def iter_rows(name: str, spec: Dict[str, Any], split_name: str) -> Iterable[Dict[str, Any]]:
        stype = spec.get("type")
        if stype == "jsonl":
            path = spec.get("path", "")
            if not path or not os.path.exists(path):
                return []
            def gen_jsonl():
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            yield json.loads(line)
            return gen_jsonl()

        if stype == "jsonl_split":
            key = "train_path" if split_name == "train" else "val_path"
            path = spec.get(key, "")
            if not path or not os.path.exists(path):
                return []
            def gen_jsonl_split():
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            yield json.loads(line)
            return gen_jsonl_split()

        sd = REG[name]
        ds = load_dataset(sd.dataset_id, sd.subset, split=sd.split, streaming=sd.streaming) if sd.subset else load_dataset(sd.dataset_id, split=sd.split, streaming=sd.streaming)
        if sd.streaming:
            ds = ds.shuffle(seed=args.seed, buffer_size=10_000)
            return (dict(r) for r in ds)
        idxs = list(range(len(ds)))
        random.Random(args.seed).shuffle(idxs)
        return (dict(ds[i]) for i in idxs)

    def max_examples_for_split(spec: Dict[str, Any], split_name: str) -> int:
        key = "max_examples_train" if split_name == "train" else "max_examples_val"
        if key in spec:
            return int(spec[key] or 0)
        return int(spec.get("max_examples", 0) or 0)

    def fill(fp, split_name: str, bucket: str, name: str, target_tokens: int, spec: Dict[str, Any]) -> int:
        tokens = 0
        written = 0
        mx = max_examples_for_split(spec, split_name)
        for row in iter_rows(name, spec, split_name):
            msgs = row.get("messages") if isinstance(row, dict) else None
            if msgs is not None:
                canon = ensure_messages(msgs)
            else:
                canon = convert_row(name, row)
            if canon is None:
                continue
            res = ok_example(canon, spec)
            if res is None:
                continue
            n, out_msgs = res
            fp.write(json.dumps({"messages": out_msgs, "meta": {"bucket": bucket, "dataset": name, "split": split_name}}, ensure_ascii=False) + "\n")
            tokens += n
            written += 1
            if mx and written >= mx:
                break
            if tokens >= target_tokens:
                break
        print(f"[{split_name}] {bucket}/{name}: {written} ex, {tokens} tok (target {target_tokens})")
        return tokens

    with open(out_val, "w", encoding="utf-8") as fval:
        for bucket, name, vt, spec in val_targets:
            try:
                fill(fval, "val", bucket, name, vt, spec)
            except Exception as e:
                print(f"[val] WARN skip {bucket}/{name}: {e}")

    with open(out_train, "w", encoding="utf-8") as ftr:
        for bucket, name, tt, spec in targets:
            try:
                fill(ftr, "train", bucket, name, tt, spec)
            except Exception as e:
                print(f"[train] WARN skip {bucket}/{name}: {e}")

    print(f"Done:\n  {out_train}\n  {out_val}")


if __name__ == "__main__":
    main()
