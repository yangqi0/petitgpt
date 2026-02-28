#!/usr/bin/env python3
"""
Evaluate petitgpt checkpoints on a tiny jsonl bench.

- Uses pretrain/sample.py via subprocess (so it matches your sampling behavior).
- Supports tasks: arithmetic / syllogism / code
- For code: stops on EOC marker and runs python tests in a temp file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from typing import Any

EOC_LINE = "###EOC###"
EOC_BLOCK = f"\n{EOC_LINE}\n"

SYS_ARITH = (
    "You are a precise assistant.\n"
    "Rules for arithmetic questions:\n"
    "- Output ONLY the final numeric answer.\n"
    "- No explanation, no extra words, no punctuation.\n"
)

SYS_SYLL = (
    "You are a logic assistant.\n"
    "Rules:\n"
    "- Output ONLY one token: yes / no / unknown.\n"
    "- Do not explain.\n"
)

SYS_CODE = (
    "You are a coding assistant.\n"
    "Rules:\n"
    "- Output ONLY Python code.\n"
    "- Do NOT write any explanation.\n"
    "- Do NOT write extra functions.\n"
    "- Do NOT write a main block.\n"
    "- Implement exactly the requested function.\n"
    "- Output ONLY the function body (no def line).\n"
    "- Do NOT use 'yield'.\n"
    "- Do NOT use semicolons ';'.\n"
    "- Write exactly ONE statement per line (no two statements on one line).\n"
    f"- End your output with exactly: {EOC_LINE}\n"
)

_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_YNU_RE = re.compile(r"\b(yes|no|unknown)\b", re.IGNORECASE)


def extract_last_number(text: str) -> float | None:
    ms = list(_NUM_RE.finditer(text))
    if not ms:
        return None
    try:
        return float(ms[-1].group(0))
    except Exception:
        return None


def extract_ynu(text: str) -> str:
    m = _YNU_RE.search(text.strip())
    if not m:
        return "unknown"
    return m.group(1).lower()


def _build_prompt(task: str, item: dict[str, Any]) -> tuple[str, str]:
    """
    Returns (task, full_prompt) where full_prompt includes system-rules + user prompt.
    bench format:
      - arithmetic/syllogism: has "prompt" and "gold"
      - code: may be either:
          a) has "prompt" + tests
          b) has "entry"+"signature"+"tests"
    """
    if task == "arithmetic":
        user = item["prompt"]
        return task, SYS_ARITH + "\n" + user
    if task == "syllogism":
        user = item["prompt"]
        return task, SYS_SYLL + "\n" + user
    if task == "code":
        if "prompt" in item:
            user = item["prompt"]
        else:
            sig = item.get("signature") or item.get("entry")
            if not sig:
                raise KeyError("code item missing prompt/signature/entry")
            user = "Complete the following Python function:\n\n" + sig.rstrip() + "\n    "
        return task, SYS_CODE + "\n" + user
    raise ValueError(f"unknown task: {task}")


def generate_one(
    sample_py: str,
    ckpt: str,
    tokenizer_path: str,
    prompt: str,
    max_seq_len: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    min_new_tokens: int,
    greedy: bool,
    avoid_first_whitespace: bool,
    ban_first_steps: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    max_repeat_token: int,
    stop_on_newline: bool,
    stop_string: str | None,
    stop_regex: str | None,
    restrict: str | None,
    extra_ban_token_ids: str | None,
) -> str:
    cmd = [
        "python",
        sample_py,
        "--ckpt",
        ckpt,
        "--tokenizer_path",
        tokenizer_path,
        "--prompt",
        prompt,
        "--max_seq_len",
        str(max_seq_len),
        "--temperature",
        str(temperature),
        "--top_p",
        str(top_p),
        "--top_k",
        str(top_k),
        "--max_new_tokens",
        str(max_new_tokens),
        "--min_new_tokens",
        str(min_new_tokens),
        "--repetition_penalty",
        str(repetition_penalty),
        "--no_repeat_ngram_size",
        str(no_repeat_ngram_size),
        "--max_repeat_token",
        str(max_repeat_token),
        "--quiet",
    ]
    if greedy:
        cmd.append("--greedy")
    if avoid_first_whitespace:
        cmd.append("--avoid_first_whitespace")
        cmd += ["--ban_first_steps", str(ban_first_steps)]
    if stop_on_newline:
        cmd.append("--stop_on_newline")
    if stop_string is not None and stop_string != "":
        cmd += ["--stop_string", stop_string]

    if stop_regex is not None and stop_regex != "":
        cmd += ["--stop_regex", stop_regex]
    if restrict is not None and restrict != "" and restrict != "none":
        cmd += ["--restrict", restrict]
    if extra_ban_token_ids is not None and extra_ban_token_ids != "":
        cmd += ["--extra_ban_token_ids", extra_ban_token_ids]

    out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    return out


def try_extract_code_body(gen: str) -> str:
    """
    Extract code body from generated text:
    - strip fences
    - cut at EOC_LINE if present (preferred)
    - otherwise keep only the first "reasonable" block
    """
    g = gen.replace("\r\n", "\n")

    # strip markdown fences
    g = re.sub(r"^```[a-zA-Z0-9_+-]*\n", "", g.strip())
    g = re.sub(r"\n```$", "", g)

    # cut at EOC marker (ignore anything after)
    if EOC_LINE in g:
        g = g.split(EOC_LINE, 1)[0]

    g = g.strip("\n")

    # keep only first continuous code block before obvious junk
    kept: list[str] = []
    for ln in g.splitlines():
        if ln.strip() == "":
            kept.append("")
            continue
        # hard stop if it looks like a new function/class/main block
        if ln.lstrip().startswith(("def ", "class ", "if __name__")):
            break
        kept.append(ln.rstrip())

    body = "\n".join(kept).strip("\n")
    return body


def _normalize_leading_indent_noise(lines: list[str]) -> list[str]:
    """
    Fix common model artifact: every non-empty line begins with 1-3 leading spaces.
    If the *minimum* leading spaces among non-empty lines is 1-3, remove that from all lines.
    """
    indents = []
    for ln in lines:
        if ln.strip() == "":
            continue
        n = len(ln) - len(ln.lstrip(" "))
        indents.append(n)
    if not indents:
        return lines
    m = min(indents)
    if 1 <= m <= 3:
        return [ln[m:] if ln.strip() != "" and ln.startswith(" " * m) else ln for ln in lines]
    return lines


def run_python_tests(signature: str, body: str, tests: list[str]) -> tuple[bool, str]:
    """
    Wrap body into a function with 'signature' then run tests.
    IMPORTANT: indenting must preserve relative indentation.
    We therefore ALWAYS prefix 4 spaces to every non-empty line.
    """
    body_lines = body.splitlines()
    body_lines = _normalize_leading_indent_noise(body_lines)

    if not body_lines or all(ln.strip() == "" for ln in body_lines):
        body_lines = ["pass"]

    fixed: list[str] = []
    for ln in body_lines:
        if ln.strip() == "":
            fixed.append("")
        else:
            fixed.append("    " + ln.rstrip("\n"))

    code = signature.rstrip() + "\n" + "\n".join(fixed) + "\n\n" + "\n".join(tests) + "\n"

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "tmp_eval.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        p = subprocess.run(
            ["python", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if p.returncode == 0:
            return True, ""
        detail = p.stderr.strip()
        detail += "\n\n[CODE_SNIPPET]\n" + code
        return False, detail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--eos_id", type=int, default=3)

    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=192)
    ap.add_argument("--min_new_tokens", type=int, default=1)

    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--avoid_first_whitespace", action="store_true")
    ap.add_argument("--ban_first_steps", type=int, default=4)

    ap.add_argument("--repetition_penalty", type=float, default=1.2)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=2)
    ap.add_argument("--max_repeat_token", type=int, default=2)

    args = ap.parse_args()

    sample_py = os.path.join(os.path.dirname(__file__), "sample.py")

    items: list[dict[str, Any]] = []
    with open(args.bench, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))

    t0 = time.time()
    results: list[dict[str, Any]] = []

    cnt = {
        "arithmetic": {"n": 0, "correct": 0},
        "syllogism": {"n": 0, "correct": 0},
        "code": {"n": 0, "correct": 0},
    }

    for it in items:
        task = it.get("task")
        if task not in ("arithmetic", "syllogism", "code"):
            continue
        if task in ("arithmetic", "syllogism") and "gold" not in it:
            print(
                f"[WARN] missing gold: id={it.get('id')} task={task} keys={list(it.keys())}",
                flush=True,
            )

        _, prompt = _build_prompt(task, it)

        # task-specific decoding strategy (and output constraints)
        if task == "code":
            stop_on_newline = False
            stop_string = EOC_BLOCK
            stop_regex = None
            restrict = None
            extra_ban = None
            max_new = args.max_new_tokens
        elif task == "arithmetic":
            stop_on_newline = True
            stop_string = None
            stop_regex = r"[-+]?\d+(?:\.\d+)?"
            restrict = "digits"
            extra_ban = str(args.eos_id)
            max_new = min(args.max_new_tokens, 32)
        else:
            stop_on_newline = True
            stop_string = None
            stop_regex = r"\b(?:yes|no|unknown)\b"
            restrict = "ynu"
            extra_ban = str(args.eos_id)
            max_new = min(args.max_new_tokens, 16)

        gen = generate_one(
            sample_py=sample_py,
            ckpt=args.ckpt,
            tokenizer_path=args.tokenizer_path,
            prompt=prompt,
            max_seq_len=args.max_seq_len,
            temperature=(0.0 if args.greedy else args.temperature),
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=max_new,
            min_new_tokens=args.min_new_tokens,
            greedy=args.greedy,
            avoid_first_whitespace=args.avoid_first_whitespace,
            ban_first_steps=args.ban_first_steps,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            max_repeat_token=args.max_repeat_token,
            stop_on_newline=stop_on_newline,
            stop_string=stop_string,
            stop_regex=stop_regex,
            restrict=restrict,
            extra_ban_token_ids=extra_ban,
        )

        ok = False
        detail = ""

        # --- arithmetic ---
        if task == "arithmetic":
            cnt["arithmetic"]["n"] += 1
            gold_raw = it.get("gold", None)
            if gold_raw is None:
                ok = False
                detail = "missing gold"
                gold = None
            else:
                gold = str(gold_raw).strip()
                pred_num = extract_last_number(gen)
                # compare as int if gold is integer-like
                if re.fullmatch(r"[-+]?\d+", gold) and pred_num is not None:
                    ok = int(pred_num) == int(gold)
                else:
                    ok = pred_num is not None and str(pred_num) == gold
                detail = f"pred={pred_num} gold={gold}"

        # --- syllogism ---
        elif task == "syllogism":
            cnt["syllogism"]["n"] += 1
            gold_raw = it.get("gold", None)
            if gold_raw is None:
                ok = False
                detail = "missing gold"
                gold = None
            else:
                gold = str(gold_raw).strip().lower()
                pred = extract_ynu(gen)
                ok = pred == gold
                detail = f"pred={pred} gold={gold}"

        # --- code ---
        else:
            cnt["code"]["n"] += 1
            signature = it.get("signature") or it.get("entry")
            tests = it.get("tests") or []
            if not signature or not isinstance(tests, list):
                ok = False
                detail = "missing signature/tests"
            else:
                body = try_extract_code_body(gen)
                ok, detail = run_python_tests(signature, body, tests)
            gold = None

        if ok:
            cnt[task]["correct"] += 1

        results.append(
            {
                "id": it.get("id"),
                "task": task,
                "ok": ok,
                "prompt": prompt,
                "gold": it.get("gold", None),
                "gen": gen,
                "detail": detail,
            }
        )

    total = sum(v["n"] for v in cnt.values())
    out = {
        "ckpt": args.ckpt,
        "bench": args.bench,
        "time_s": round(time.time() - t0, 3),
        "acc_arithmetic": (cnt["arithmetic"]["correct"] / max(cnt["arithmetic"]["n"], 1)),
        "acc_syllogism": (cnt["syllogism"]["correct"] / max(cnt["syllogism"]["n"], 1)),
        "acc_code": (cnt["code"]["correct"] / max(cnt["code"]["n"], 1)),
        "counts": {"total": total, **cnt},
        "results": results,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {
                k: out[k]
                for k in [
                    "ckpt",
                    "bench",
                    "time_s",
                    "acc_arithmetic",
                    "acc_syllogism",
                    "acc_code",
                    "counts",
                ]
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
