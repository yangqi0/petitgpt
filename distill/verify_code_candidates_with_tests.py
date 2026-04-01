#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Verify code teacher candidates with:
1) exact one code block
2) AST parse
3) entrypoint exists
4) minimal unit tests

This script intentionally canonicalizes accepted outputs to:
```python
<code>
```

Security note:
Executing model-generated code is risky.
This verifier uses:
- a subprocess
- a timeout
- limited builtins
- a restricted importer
Even so, it is still best run in an isolated environment/container.

Comments are in English by design.
"""

from __future__ import annotations

import argparse
import ast
import json
import multiprocessing as mp
import os
import re
import traceback
from typing import Any


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_code_blocks(text: str) -> int:
    return len(re.findall(r"```", text)) // 2


def extract_code_block(text: str) -> str | None:
    m = re.search(r"```python\s*\n(.*?)\n```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"```\s*\n(.*?)\n```", text, flags=re.DOTALL)
    if m:
        return m.group(1)
    return None


def canonical_code_response(code: str) -> str:
    code = code.rstrip() + "\n"
    return f"```python\n{code}```"


def safe_import(name: str, globals=None, locals=None, fromlist=(), level=0):
    allowed = {
        "re",
        "math",
        "collections",
        "itertools",
        "functools",
        "statistics",
        "heapq",
        "bisect",
        "string",
    }
    if name in allowed:
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(f"Import not allowed: {name}")


def build_safe_builtins() -> dict[str, Any]:
    allowed_names = [
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
        "int", "len", "list", "map", "max", "min", "None", "pow", "print",
        "range", "reversed", "round", "set", "slice", "sorted", "str", "sum",
        "tuple", "zip",
        "Exception", "ValueError", "TypeError", "IndexError", "KeyError",
    ]
    builtins_obj = __builtins__
    if isinstance(builtins_obj, dict):
        src = builtins_obj
    else:
        src = builtins_obj.__dict__
    out = {k: src[k] for k in allowed_names if k in src}
    out["__import__"] = safe_import
    return out


def normalize_number(x: Any) -> Any:
    if isinstance(x, float):
        return round(x, 8)
    return x


def values_equal(a: Any, b: Any) -> bool:
    if type(a) != type(b):
        # Allow int/float comparison if numerically equal.
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return abs(float(a) - float(b)) < 1e-8
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(values_equal(a[k], b[k]) for k in a.keys())
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(values_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) < 1e-8
    return normalize_number(a) == normalize_number(b)


def worker(code: str, entrypoint: str, tests: list[dict[str, Any]], queue: mp.Queue) -> None:
    try:
        # Optional resource limits on Unix.
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
            # 512MB address space cap.
            mem = 512 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        except Exception:
            pass

        safe_globals: dict[str, Any] = {"__builtins__": build_safe_builtins()}
        exec(code, safe_globals, safe_globals)

        fn = safe_globals.get(entrypoint)
        if fn is None:
            queue.put({"ok": False, "reason": "missing_entrypoint"})
            return
        if not callable(fn):
            queue.put({"ok": False, "reason": "entrypoint_not_callable"})
            return

        for idx, test in enumerate(tests):
            args = test.get("args", [])
            kwargs = test.get("kwargs", {})
            expected = test.get("expected")
            raises = test.get("raises")

            try:
                got = fn(*args, **kwargs)
                if raises:
                    queue.put({"ok": False, "reason": f"test_{idx}_expected_raise_{raises}"})
                    return
                if not values_equal(got, expected):
                    queue.put({
                        "ok": False,
                        "reason": f"test_{idx}_wrong_answer",
                        "expected": expected,
                        "got": got,
                    })
                    return
            except Exception as e:
                if raises and type(e).__name__ == raises:
                    continue
                queue.put({
                    "ok": False,
                    "reason": f"test_{idx}_exception_{type(e).__name__}",
                    "traceback": traceback.format_exc(),
                })
                return

        queue.put({"ok": True})
    except Exception as e:
        queue.put({
            "ok": False,
            "reason": f"worker_exception_{type(e).__name__}",
            "traceback": traceback.format_exc(),
        })


def run_with_timeout(code: str, entrypoint: str, tests: list[dict[str, Any]], timeout_s: float) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    p = ctx.Process(target=worker, args=(code, entrypoint, tests, q))
    p.start()
    p.join(timeout_s)

    if p.is_alive():
        p.kill()
        p.join()
        return {"ok": False, "reason": "timeout"}

    if not q.empty():
        return q.get()
    return {"ok": False, "reason": "no_result"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--rejected_jsonl", required=True)
    ap.add_argument("--timeout_s", type=float, default=3.0)
    args = ap.parse_args()

    rows = read_jsonl(args.in_jsonl)
    accepted_best: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []

    for row in rows:
        prompt_id = row["prompt_id"]
        meta = row.get("meta", {}) or {}
        entrypoint = meta.get("entrypoint")
        tests = meta.get("tests", [])

        text = row.get("response", "")
        if count_code_blocks(text) != 1:
            rejected.append({**row, "reject_reason": "need_exactly_one_code_block"})
            continue

        code = extract_code_block(text)
        if not code or not code.strip():
            rejected.append({**row, "reject_reason": "empty_code"})
            continue

        if not entrypoint:
            rejected.append({**row, "reject_reason": "missing_entrypoint_meta"})
            continue

        try:
            ast.parse(code)
        except Exception as e:
            rejected.append({**row, "reject_reason": f"ast_error_{type(e).__name__}"})
            continue

        verdict = run_with_timeout(code, entrypoint, tests, args.timeout_s)
        if not verdict.get("ok", False):
            rejected.append({**row, "reject_reason": verdict.get("reason", "unknown")})
            continue

        canonical = canonical_code_response(code)
        kept = {
            **row,
            "response": canonical,
            "code": code,
        }

        # Keep the shortest accepted response per prompt_id.
        prev = accepted_best.get(prompt_id)
        if prev is None or len(canonical) < len(prev["response"]):
            accepted_best[prompt_id] = kept

    accepted = list(accepted_best.values())
    accepted.sort(key=lambda x: x["prompt_id"])
    rejected.sort(key=lambda x: (x["prompt_id"], x.get("candidate_id", 0)))

    write_jsonl(args.out_jsonl, accepted)
    write_jsonl(args.rejected_jsonl, rejected)

    print(f"Accepted: {len(accepted)}")
    print(f"Rejected: {len(rejected)}")
    print(f"Wrote accepted -> {args.out_jsonl}")
    print(f"Wrote rejected -> {args.rejected_jsonl}")


if __name__ == "__main__":
    main()
