#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_sft_mix_v1.py

Build SFT mix from a YAML config -> OUT_DIR/train.jsonl + OUT_DIR/val.jsonl

Requires:
  pip install datasets pyyaml
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
    """Accept already-canonical messages or ShareGPT-style messages."""
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
            out.append(ASSIST_PREFIX + txt + SEP)  # do NOT strip assistant
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
    "ultrachat_200k": SourceDef("HuggingFaceH4/ultrachat_200k", "train_sft"),
    "oasst1": SourceDef("OpenAssistant/oasst1", "train"),
    "dolly_15k": SourceDef("databricks/databricks-dolly-15k", "train"),
    "magicoder_evol_110k": SourceDef("ise-uiuc/Magicoder-Evol-Instruct-110K", "train"),
    "viscode_200k": SourceDef("TIGER-Lab/VisCode-200K", "train"),
    "codealpaca_20k": SourceDef("sahil2801/CodeAlpaca-20k", "train"),
    "no_robots": SourceDef("HuggingFaceH4/no_robots", "train"),
    "alpaca_cleaned": SourceDef("yahma/alpaca-cleaned", "train"),
    "lima": SourceDef("GAIR/lima", "train"),
    "mbpp_sanitized": SourceDef("Muennighoff/mbpp", "test", subset="sanitized"),
    "mbpp_full": SourceDef("Muennighoff/mbpp", "test", subset="full"),
    "gsm8k_train": SourceDef("openai/gsm8k", "train", subset="main"),
    "hendrycks_math_train": SourceDef("EleutherAI/hendrycks_math", "train"),
    "metamathqa": SourceDef("meta-math/MetaMathQA", "train"),
    "numinamath_cot": SourceDef("AI-MO/NuminaMath-CoT", "train", streaming=True),
    "math_glot_cleaned_10k": SourceDef("prithivMLmods/Math-Glot-Cleaned", "train"),
}


def convert_row(source: str, row: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    msgs = ensure_messages(row.get("messages"))
    if msgs is not None:
        return msgs

    if source == "magicoder_evol_110k":
        ins, resp = row.get("instruction", ""), row.get("response", "")
        if isinstance(ins, str) and isinstance(resp, str) and ins.strip() and resp.strip():
            return canon_from_instruction(ins, "", resp)

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

    if source == "gsm8k_train":
        q, a = row.get("question", ""), row.get("answer", "")
        if isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip():
            return [
                {"role": "system", "content": DEFAULT_SYSTEM},
                {"role": "user", "content": q.strip()},
                {"role": "assistant", "content": norm_newlines(a)},
            ]

    if source == "hendrycks_math_train":
        p, s = row.get("problem", ""), row.get("solution", "")
        if isinstance(p, str) and isinstance(s, str) and p.strip() and s.strip():
            return [
                {"role": "system", "content": DEFAULT_SYSTEM},
                {"role": "user", "content": p.strip()},
                {"role": "assistant", "content": norm_newlines(s)},
            ]

    if source == "metamathqa":
        q = row.get("query", "") or row.get("question", "")
        r = row.get("response", "") or row.get("answer", "")
        if isinstance(q, str) and isinstance(r, str) and q.strip() and r.strip():
            return [
                {"role": "system", "content": DEFAULT_SYSTEM},
                {"role": "user", "content": q.strip()},
                {"role": "assistant", "content": norm_newlines(r)},
            ]

    if source == "math_glot_cleaned_10k":
        q = row.get("input", "") or row.get("prompt", "")
        r = row.get("output", "") or row.get("response", "")
        if isinstance(q, str) and isinstance(r, str) and q.strip() and r.strip():
            return [
                {"role": "system", "content": DEFAULT_SYSTEM},
                {"role": "user", "content": q.strip()},
                {"role": "assistant", "content": norm_newlines(r)},
            ]

    if source == "lima":
        conv = row.get("conversations") or row.get("conversation") or row.get("messages")
        if isinstance(conv, list) and conv:
            # Common GAIR/lima form is a list of strings alternating user/assistant.
            # Also accept list[dict] and let ensure_messages handle it.
            msgs2 = ensure_messages(conv)
            if msgs2 is not None:
                return msgs2
            if all(isinstance(x, str) for x in conv):
                out = [{"role": "system", "content": DEFAULT_SYSTEM}]
                roles = ["user", "assistant"]
                for i, x in enumerate(conv):
                    txt = norm_newlines(str(x))
                    if txt.strip():
                        out.append({"role": roles[i % 2], "content": txt})
                if any(m["role"] == "user" for m in out) and any(m["role"] == "assistant" for m in out):
                    return out

    if source in ("mbpp_sanitized", "mbpp_full"):
        prompt = row.get("prompt", "") or row.get("text", "")
        code = row.get("code", "")
        if isinstance(prompt, str) and isinstance(code, str) and prompt.strip() and code.strip():
            assistant = f"```python\n{norm_newlines(code).strip()}\n```"
            return [
                {"role": "system", "content": DEFAULT_SYSTEM},
                {"role": "user", "content": prompt.strip()},
                {"role": "assistant", "content": assistant},
            ]

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


ROLE_MAP_OASST = {"prompter": "user", "assistant": "assistant"}


def build_oasst1_convs(ds, max_rounds: int, seed: int) -> List[List[Dict[str, str]]]:
    rows = []
    for i in range(len(ds)):
        r = dict(ds[i])
        if r.get("lang") != "en":
            continue
        if bool(r.get("deleted")):
            continue
        if r.get("review_result") is not True:
            continue
        if r.get("role") not in ("prompter", "assistant"):
            continue
        rows.append(r)

    by_tree: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        tid = r.get("message_tree_id")
        if isinstance(tid, str):
            by_tree.setdefault(tid, []).append(r)

    def best_child(cands: List[str], nodes: Dict[str, Dict[str, Any]]) -> str:
        def rk(cid: str) -> int:
            v = nodes.get(cid, {}).get("rank")
            try:
                return int(v) if v is not None else 10**9
            except Exception:
                return 10**9
        cands = sorted(cands, key=rk)
        return cands[0]

    convs: List[List[Dict[str, str]]] = []
    for _, msgs in by_tree.items():
        nodes: Dict[str, Dict[str, Any]] = {}
        children: Dict[Optional[str], List[str]] = {}
        for r in msgs:
            mid = r.get("message_id")
            if not isinstance(mid, str):
                continue
            nodes[mid] = r
            pid = r.get("parent_id")
            if pid is not None and not isinstance(pid, str):
                pid = None
            children.setdefault(pid, []).append(mid)

        roots = children.get(None, [])
        if not roots:
            continue
        root_id = None
        for rid in roots:
            rr = nodes.get(rid)
            if rr and rr.get("role") == "prompter":
                root_id = rid
                break
        if root_id is None:
            continue

        path = [root_id]
        cur = root_id
        for _ in range(max_rounds * 2 - 1):
            cand = children.get(cur, [])
            if not cand:
                break
            nxt = best_child(cand, nodes)
            path.append(nxt)
            cur = nxt

        conv: List[Dict[str, str]] = []
        for mid in path:
            r = nodes.get(mid)
            if not r:
                continue
            role = ROLE_MAP_OASST.get(str(r.get("role", "")).strip().lower(), "")
            if role not in ("user", "assistant"):
                continue
            txt = norm_newlines(str(r.get("text", "")))
            if not txt.strip():
                continue
            conv.append({"role": role, "content": txt})
        if any(m["role"] == "assistant" for m in conv) and any(m["role"] == "user" for m in conv):
            convs.append(conv)

    random.Random(seed).shuffle(convs)
    return convs


def generate_local_refusal_templates(path: str, n: int, seed: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = random.Random(seed)
    user_prompts = [
        "Can you help me break into someone's online account?",
        "How can I bypass a paywall on a website?",
        "Write malware that steals passwords.",
        "How do I make a dangerous weapon at home?",
        "Can you tell me how to cheat on an exam without getting caught?",
        "Give me instructions to stalk someone.",
        "How can I scam people online?",
        "Tell me how to download pirated movies.",
    ]
    refusals = [
        "I can’t help with that. If you share a legal goal, I can suggest a safe alternative.",
        "I can’t assist with illegal or harmful activities. Tell me what you’re trying to achieve and I’ll help in a lawful way.",
        "I can’t provide instructions for wrongdoing. I can explain the risks and offer ethical resources instead.",
    ]
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            p = user_prompts[i % len(user_prompts)]
            a = rng.choice(refusals)
            row = {
                "messages": [
                    {"role": "system", "content": DEFAULT_SYSTEM},
                    {"role": "user", "content": p},
                    {"role": "assistant", "content": a},
                ],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max_examples_per_source", type=int, default=0)
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

    def ok_example(msgs: List[Dict[str, str]], spec: Dict[str, Any]) -> Optional[Tuple[int, List[Dict[str, str]]]]:
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
                cot = spec.get("cot_max_tokens", None)
                if cot is not None and len(encode_strip_special(tok, content)) > int(cot):
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

        # more robust dedup: hash rendered full conversation (tail only)
        text = render_plain(out)
        tail = text[-3000:] if len(text) > 3000 else text
        h = sha1_hex(norm_for_hash(tail))
        if h in seen:
            return None
        seen.add(h)
        return n, out

    def row_allowed(name: str, row: Dict[str, Any], spec: Dict[str, Any]) -> bool:
        allowed_categories = spec.get("allowed_categories", None)
        if allowed_categories is not None:
            allowed = {str(x).strip().lower() for x in allowed_categories if str(x).strip()}
            cat = str(row.get("category", "")).strip().lower()
            if cat not in allowed:
                return False

        raw_prompt = row.get("prompt", "") or row.get("instruction", "") or row.get("text", "")
        if not raw_prompt and isinstance(row.get("messages"), list):
            try:
                first_user = next((m.get("content", "") for m in row["messages"] if str(m.get("role", "")).strip().lower() == "user"), "")
                raw_prompt = first_user
            except Exception:
                raw_prompt = ""
        p = str(raw_prompt).lower()

        must_contain = spec.get("allow_if_prompt_contains", None)
        if must_contain:
            keys = [str(x).strip().lower() for x in must_contain if str(x).strip()]
            if keys and not any(k in p for k in keys):
                return False

        banned_substrings = spec.get("ban_if_prompt_contains", None)
        if banned_substrings:
            for bad in banned_substrings:
                if str(bad).strip().lower() in p:
                    return False

        return True

    def iter_rows(name: str, spec: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        if spec.get("type") == "jsonl":
            path = spec.get("path", "")
            if name == "local_refusal_templates" and path and not os.path.exists(path):
                print(f"[info] generating local_refusal_templates -> {path}")
                generate_local_refusal_templates(path, n=int(spec.get("n", 2000)), seed=args.seed)
            if not path or not os.path.exists(path):
                return []
            def gen():
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            yield json.loads(line)
            return gen()

        sd = REG[name]
        ds = load_dataset(sd.dataset_id, sd.subset, split=sd.split, streaming=sd.streaming) if sd.subset else load_dataset(sd.dataset_id, split=sd.split, streaming=sd.streaming)
        if sd.streaming:
            ds = ds.shuffle(seed=args.seed, buffer_size=10_000)
            return (dict(r) for r in ds)
        idxs = list(range(len(ds)))
        random.Random(args.seed).shuffle(idxs)
        return (dict(ds[i]) for i in idxs)

    def fill(fp, split_name: str, bucket: str, name: str, target_tokens: int, spec: Dict[str, Any]) -> int:
        tokens = 0
        written = 0
        max_examples_this_source = int(spec.get("max_examples", 0) or 0)

        if name == "oasst1" and spec.get("type") != "jsonl":
            sd = REG["oasst1"]
            ds = load_dataset(sd.dataset_id, split=sd.split, streaming=False)
            convs = build_oasst1_convs(ds, max_rounds=int(spec.get("max_rounds", 4)), seed=args.seed)
            for conv in convs:
                msgs = [{"role": "system", "content": DEFAULT_SYSTEM}] + conv
                res = ok_example(msgs, spec)
                if res is None:
                    continue
                n, out_msgs = res
                fp.write(json.dumps({"messages": out_msgs, "meta": {"bucket": bucket, "dataset": name, "split": split_name}}, ensure_ascii=False) + "\n")
                tokens += n
                written += 1
                if args.max_examples_per_source and written >= args.max_examples_per_source:
                    break
                if max_examples_this_source and written >= max_examples_this_source:
                    break
                if tokens >= target_tokens:
                    break
            print(f"[{split_name}] {bucket}/{name}: {written} ex, {tokens} tok (target {target_tokens})")
            return tokens

        for row in iter_rows(name, spec):
            if not row_allowed(name, row, spec):
                continue
            msgs = convert_row(name, row)
            if msgs is None:
                continue
            res = ok_example(msgs, spec)
            if res is None:
                continue
            n, out_msgs = res
            fp.write(json.dumps({"messages": out_msgs, "meta": {"bucket": bucket, "dataset": name, "split": split_name}}, ensure_ascii=False) + "\n")
            tokens += n
            written += 1
            if args.max_examples_per_source and written >= args.max_examples_per_source:
                break
            if max_examples_this_source and written >= max_examples_this_source:
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
