from __future__ import annotations

import ast
import hashlib
import json
import multiprocessing as mp
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

BANNED_CALLS = {"eval", "exec", "open", "input", "compile", "__import__"}
BANNED_TOPLEVEL = (
    ast.Import,
    ast.ImportFrom,
    ast.ClassDef,
    ast.AsyncFunctionDef,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Raise,
    ast.Global,
    ast.Nonlocal,
)
BANNED_ANYWHERE = (
    ast.Lambda,
    ast.Yield,
    ast.YieldFrom,
    ast.Await,
)

def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def normalize_text(s: str) -> str:
    s = (s or "").replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\r\n?", "\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def normalize_prompt(s: str) -> str:
    s = normalize_text(s).lower()
    s = re.sub(r"[^\w\s\"'`-]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def stable_id(prefix: str, *parts: str, length: int = 10) -> str:
    text = "||".join([prefix] + [str(p) for p in parts])
    return f"{prefix}_{hashlib.md5(text.encode('utf-8')).hexdigest()[:length]}"

def extract_first_code_block(text: str) -> str:
    text = normalize_text(text)
    m = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()

def code_line_count(code: str) -> int:
    return len([x for x in code.splitlines() if x.strip()])

def ast_node_count(tree: ast.AST) -> int:
    return sum(1 for _ in ast.walk(tree))

def max_ast_depth(node: ast.AST, depth: int = 0) -> int:
    children = list(ast.iter_child_nodes(node))
    if not children:
        return depth
    return max(max_ast_depth(c, depth + 1) for c in children)

def infer_entry_point_from_code(code: str) -> Optional[str]:
    try:
        tree = ast.parse(code)
    except Exception:
        return None
    funcs = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(funcs) == 1:
        return funcs[0]
    return None

def infer_entry_point_from_tests(tests: Sequence[str]) -> Optional[str]:
    for t in tests:
        m = re.search(r"assert\s+([A-Za-z_]\w*)\s*\(", t)
        if m:
            return m.group(1)
    return None

def has_banned_call(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in BANNED_CALLS:
                return True
            if isinstance(fn, ast.Attribute) and fn.attr in BANNED_CALLS:
                return True
    return False

def recursion_detected(tree: ast.AST, entry_point: str) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == entry_point:
            return True
    return False

def verify_ast_structure(code: str, entry_point: str) -> List[str]:
    reasons: List[str] = []
    try:
        tree = ast.parse(code)
    except Exception:
        return ["syntax_error"]

    top_funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(top_funcs) != 1:
        reasons.append("top_level_function_count")
    if top_funcs and top_funcs[0].name != entry_point:
        reasons.append("wrong_function_name")

    for node in tree.body:
        if isinstance(node, BANNED_TOPLEVEL):
            reasons.append("banned_toplevel_node")
            break

    for node in ast.walk(tree):
        if isinstance(node, BANNED_ANYWHERE):
            reasons.append("banned_node")
            break

    if has_banned_call(tree):
        reasons.append("banned_call")
    if recursion_detected(tree, entry_point):
        reasons.append("recursion")
    if code_line_count(code) > 25:
        reasons.append("too_many_lines")
    if ast_node_count(tree) > 140:
        reasons.append("too_many_ast_nodes")
    if max_ast_depth(tree) > 12:
        reasons.append("too_deep_ast")
    return reasons

def _run_tests_worker(code: str, entry_point: str, tests: List[str], queue: mp.Queue) -> None:
    try:
        builtins = {
            "len": len, "range": range, "enumerate": enumerate, "sum": sum, "min": min, "max": max,
            "abs": abs, "all": all, "any": any, "sorted": sorted, "list": list, "dict": dict,
            "set": set, "tuple": tuple, "str": str, "int": int, "float": float, "bool": bool, "zip": zip
        }
        glb: Dict[str, Any] = {"__builtins__": builtins}
        loc: Dict[str, Any] = {}
        exec(code, glb, loc)
        fn = loc.get(entry_point) or glb.get(entry_point)
        if fn is None:
            queue.put(("fail", "missing_entry_point"))
            return
        env = dict(glb)
        env.update(loc)
        env[entry_point] = fn
        for t in tests:
            exec(t, env, env)
        queue.put(("pass", None))
    except Exception as e:
        queue.put(("fail", f"{type(e).__name__}: {e}"))

def run_tests_with_timeout(code: str, entry_point: str, tests: List[str], timeout: float = 0.5) -> Tuple[bool, Optional[str]]:
    q: mp.Queue = mp.Queue()
    p = mp.Process(target=_run_tests_worker, args=(code, entry_point, tests, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return False, "timeout"
    if q.empty():
        return False, "no_result"
    status, detail = q.get()
    return status == "pass", detail

def family_specs() -> Dict[str, Dict[str, Any]]:
    return {
        "safe_divide": {"entry_point": "safe_divide", "tests": [
            "assert safe_divide(6, 2) == 3",
            "assert safe_divide(1, 0) == 0.0",
            "assert safe_divide(1, 0, 7) == 7",
            "assert safe_divide(-9, 3) == -3",
            "assert abs(safe_divide(1.5, 0.5) - 3.0) < 1e-9",
        ]},
        "running_sum": {"entry_point": "running_sum", "tests": [
            "assert running_sum([]) == []",
            "assert running_sum([1]) == [1]",
            "assert running_sum([1, 2, 3]) == [1, 3, 6]",
            "assert running_sum([3, -1, 2]) == [3, 2, 4]",
            "assert running_sum([0, 0, 0]) == [0, 0, 0]",
        ]},
        "running_max": {"entry_point": "running_max", "tests": [
            "assert running_max([]) == []",
            "assert running_max([1]) == [1]",
            "assert running_max([1, 3, 2, 5]) == [1, 3, 3, 5]",
            "assert running_max([-2, -5, -1]) == [-2, -2, -1]",
        ]},
        "dedup_preserve_order": {"entry_point": "dedup_preserve_order", "tests": [
            "assert dedup_preserve_order([]) == []",
            "assert dedup_preserve_order([1, 2, 1, 3, 2]) == [1, 2, 3]",
            "assert dedup_preserve_order(['a', 'b', 'a']) == ['a', 'b']",
            "assert dedup_preserve_order([1, 1, 1]) == [1]",
        ]},
        "count_words": {"entry_point": "count_words", "tests": [
            "assert count_words('') == {}",
            "assert count_words('a') == {'a': 1}",
            "assert count_words('a a b') == {'a': 2, 'b': 1}",
            "assert count_words('One one TWO') == {'one': 2, 'two': 1}",
            "assert count_words('a   b   a') == {'a': 2, 'b': 1}",
        ]},
        "lowercase_keys": {"entry_point": "lowercase_keys", "tests": [
            "assert lowercase_keys({}) == {}",
            "assert lowercase_keys({'A': 1}) == {'a': 1}",
            "assert lowercase_keys({'A': 1, 'b': 2}) == {'a': 1, 'b': 2}",
            "assert lowercase_keys({'A': 1, 'a': 3}) == {'a': 3}",
        ]},
        "flatten_once": {"entry_point": "flatten_once", "tests": [
            "assert flatten_once([]) == []",
            "assert flatten_once([1, [2, 3], 4]) == [1, 2, 3, 4]",
            "assert flatten_once([[1], [2], [3]]) == [1, 2, 3]",
            "assert flatten_once([1, [], 2]) == [1, 2]",
            "assert flatten_once([1, [2, [3]], 4]) == [1, 2, [3], 4]",
        ]},
        "reverse_string": {"entry_point": "reverse_string", "tests": [
            "assert reverse_string('') == ''",
            "assert reverse_string('a') == 'a'",
            "assert reverse_string('abc') == 'cba'",
            "assert reverse_string('ab cd') == 'dc ba'",
            "assert reverse_string('你好') == '好你'",
        ]},
        "is_prime": {"entry_point": "is_prime", "tests": [
            "assert is_prime(-3) is False",
            "assert is_prime(0) is False",
            "assert is_prime(1) is False",
            "assert is_prime(2) is True",
            "assert is_prime(3) is True",
            "assert is_prime(4) is False",
            "assert is_prime(29) is True",
            "assert is_prime(49) is False",
        ]},
        "clamp": {"entry_point": "clamp", "tests": [
            "assert clamp(5, 0, 10) == 5",
            "assert clamp(-2, 0, 10) == 0",
            "assert clamp(15, 0, 10) == 10",
            "assert clamp(0, 0, 10) == 0",
        ]},
        "remove_none": {"entry_point": "remove_none", "tests": [
            "assert remove_none([]) == []",
            "assert remove_none([None, 1, None, 2]) == [1, 2]",
            "assert remove_none([None, None]) == []",
            "assert remove_none(['a', None, 'b']) == ['a', 'b']",
        ]},
        "merge_counts": {"entry_point": "merge_counts", "tests": [
            "assert merge_counts({}, {}) == {}",
            "assert merge_counts({'a': 1}, {}) == {'a': 1}",
            "assert merge_counts({'a': 1}, {'a': 2}) == {'a': 3}",
            "assert merge_counts({'a': 1, 'b': 2}, {'b': 5, 'c': 1}) == {'a': 1, 'b': 7, 'c': 1}",
        ]},
    }

def repr_py(x: Any) -> str:
    return repr(x)

def extract_assert_tests_from_apps_io(io_obj: Dict[str, Any], fn_name: str, max_tests: int = 8) -> List[str]:
    inputs = io_obj.get("inputs", [])
    outputs = io_obj.get("outputs", [])
    tests: List[str] = []
    for i, (inp, out) in enumerate(zip(inputs, outputs)):
        if i >= max_tests:
            break
        args = inp
        if not isinstance(args, list):
            args = [args]
        arg_str = ", ".join(repr_py(a) for a in args)
        tests.append(f"assert {fn_name}({arg_str}) == {repr_py(out)}")
    return tests

def json_load_maybe(x: Any) -> Any:
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x

def semantic_vectors(texts: List[str]) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
        model_name = os.environ.get("SEMANTIC_MODEL", "all-MiniLM-L6-v2")
        model = SentenceTransformer(model_name)
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    except Exception:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            return vec.fit_transform(texts)
        except Exception:
            return None

def cosine_sim(a: Any, i: int, j: int) -> float:
    try:
        import numpy as np
        vi = a[i]
        vj = a[j]
        if hasattr(vi, "toarray"):
            vi = vi.toarray()[0]
            vj = vj.toarray()[0]
        denom = (np.linalg.norm(vi) * np.linalg.norm(vj))
        if denom == 0:
            return 0.0
        return float(np.dot(vi, vj) / denom)
    except Exception:
        return 0.0

def openai_compatible_chat(
    api_base: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 120,
    api_key: Optional[str] = None,
    timeout: int = 120,
) -> str:
    import urllib.request
    api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("VLLM_API_KEY") or "EMPTY"
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    return out["choices"][0]["message"]["content"]
