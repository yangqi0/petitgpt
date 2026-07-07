"""Reward functions for GRPO (grpo/grpo.py).

GRPO turns a *scalar reward per sampled completion* into a learning signal, so
the reward function is the crux of the whole stage. This module keeps rewards
pluggable behind a tiny registry: each reward is `fn(completion, example) ->
float`, where `example` is the raw JSONL record for the prompt (so a reward can
read `example["messages"]`, `example["reference"]`, `example["tests"]`, etc.).

Built-in rewards fall into two groups:

- Self-contained rule-based rewards (`nonempty`, `length`, `no_repeat`,
  `reference_exact`, `reference_contains`) — no external process, CPU-only,
  cheap to test. Good for smoke-testing the RL loop and for tasks with a
  reference/final answer.
- `code` — a verifiable reward (RLVR) that reuses the distillation code
  verifier (`distill/code_utils.py`): extract the code block, check the AST
  structure, and run the example's unit tests in the restricted sandbox. This is
  the reward that makes GRPO meaningful for the project's coding target.

Combine rewards with a spec string: `"nonempty+no_repeat"` sums them equally,
`"code:1.0+no_repeat:0.1"` sums with weights. See `get_reward_fn`.
"""

from __future__ import annotations

from collections.abc import Callable
import math
import re

RewardFn = Callable[[str, dict], float]

_REGISTRY: dict[str, RewardFn] = {}


def register(name: str) -> Callable[[RewardFn], RewardFn]:
    def deco(fn: RewardFn) -> RewardFn:
        _REGISTRY[name] = fn
        return fn

    return deco


def available() -> list[str]:
    return sorted(_REGISTRY)


def _lookup(name: str) -> RewardFn:
    if name not in _REGISTRY:
        raise KeyError(f"unknown reward {name!r}; available: {available()}")
    return _REGISTRY[name]


def get_reward_fn(spec: str) -> RewardFn:
    """Parse a reward spec into a single callable.

    spec grammar: one or more `name` or `name:weight` joined by `+`, e.g.
    "code", "nonempty+no_repeat", "code:1.0+no_repeat:0.1". The result sums the
    (weighted) component rewards.
    """
    parts = [p.strip() for p in spec.split("+") if p.strip()]
    if not parts:
        raise ValueError(f"empty reward spec: {spec!r}")
    weighted: list[tuple[RewardFn, float]] = []
    for p in parts:
        if ":" in p:
            name, w = p.split(":", 1)
            weighted.append((_lookup(name.strip()), float(w)))
        else:
            weighted.append((_lookup(p), 1.0))

    def combined(completion: str, example: dict) -> float:
        return float(sum(w * fn(completion, example) for fn, w in weighted))

    return combined


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


# ---------------------------------------------------------------------------
# Self-contained rule-based rewards
# ---------------------------------------------------------------------------
@register("nonempty")
def reward_nonempty(completion: str, example: dict) -> float:
    return 1.0 if completion.strip() else 0.0


@register("length")
def reward_length(completion: str, example: dict) -> float:
    """Smooth bump peaking at `meta.target_words` (default 40), in [0, 1].

    A gentle shaping reward: useful to sanity-check the loop and to discourage
    empty or runaway generations. Target is per-example so different prompts can
    ask for different lengths.
    """
    target = float((example.get("meta") or {}).get("target_words", 40))
    n = len(completion.split())
    if n == 0:
        return 0.0
    return math.exp(-(((n - target) / max(target, 1.0)) ** 2))


@register("no_repeat")
def reward_no_repeat(completion: str, example: dict) -> float:
    """Fraction of distinct trigrams (1.0 = no repetition, lower = degenerate).

    Directly penalizes the repetition-loop failure mode that RL on a small model
    is prone to reward-hack into.
    """
    toks = completion.split()
    if len(toks) < 4:
        return 1.0
    grams = list(zip(toks, toks[1:], toks[2:], strict=False))
    return len(set(grams)) / len(grams)


@register("reference_exact")
def reward_reference_exact(completion: str, example: dict) -> float:
    """1.0 iff the completion equals `example["reference"]` (whitespace/case
    normalized). For tasks with a single canonical answer."""
    ref = example.get("reference")
    if ref is None:
        return 0.0
    return 1.0 if _norm(completion) == _norm(ref) else 0.0


@register("reference_contains")
def reward_reference_contains(completion: str, example: dict) -> float:
    """1.0 iff the reference/answer string appears in the completion. Handy for
    math-style tasks where only the final answer must match."""
    ref = example.get("reference") or example.get("answer")
    if not ref:
        return 0.0
    return 1.0 if _norm(str(ref)) in _norm(completion) else 0.0


# ---------------------------------------------------------------------------
# Verifiable code reward (RLVR) — reuses the distillation verifier
# ---------------------------------------------------------------------------
@register("code")
def reward_code(completion: str, example: dict) -> float:
    """Verifiable reward for Python-function tasks.

    Returns 1.0 if the extracted function passes the AST structure checks AND
    the example's unit tests; 0.5 if it is structurally valid but has no tests
    or fails them; 0.0 if no valid single-function block can be extracted.

    Expects `example["tests"]` (list of assert strings) and optionally
    `example["entry_point"]`. Imports `distill.code_utils` lazily so that merely
    importing this module never pulls in the sandbox machinery.
    """
    from distill import code_utils as cu

    code = cu.extract_first_code_block(completion) or completion
    entry = (
        example.get("entry_point")
        or cu.infer_entry_point_from_code(code)
        or cu.infer_entry_point_from_tests(example.get("tests") or [])
    )
    if not entry:
        return 0.0
    if cu.verify_ast_structure(code, entry):  # non-empty list of reasons => invalid
        return 0.0
    tests = example.get("tests") or []
    if not tests:
        return 0.5  # structurally valid but unverifiable
    timeout = float((example.get("meta") or {}).get("timeout", 0.5))
    ok, _ = cu.run_tests_with_timeout(code, entry, tests, timeout=timeout)
    return 1.0 if ok else 0.5
