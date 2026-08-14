"""Single source of truth for the chat format, shared by SFT / distill / DPO /
GRPO training AND their sampling code (previously seven duplicated copies).

Token-level template — role boundaries are special tokens, not plain text:

    [BOS] <|system|> {system} <|user|> {q1} <|assistant|> {a1} [EOS] <|user|> {q2} ...

Design rules:
- Role tokens delimit turns. BPE can never merge across a special token, so a
  conversation encodes to the same ids whether it is built turn-by-turn during
  training or as a generation prompt at inference (the old plain-text
  "User: ...\\n\\n" template drifted at every segment boundary).
- [EOS] appears ONLY after assistant turns: it means "assistant finished,
  stop generating" — the same stop semantics as document ends in pretraining.
  System/user turns need no terminator; the next role token is the boundary.
- The supervised span is exactly each assistant turn's content tokens plus its
  trailing [EOS] (so the model is explicitly taught to stop).
- Content is encoded with `tokenizer.encode_special_tokens = True`
  (see `load_chat_tokenizer`), so literal "[EOS]" / "<|user|>" strings inside
  user or corpus text are tokenized as plain text and can never inject real
  control tokens.
"""

from __future__ import annotations

from typing import Any

from tokenizers import Tokenizer

from src.special_tokens import (
    ASSISTANT_ID,
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SYSTEM_ID,
    USER_ID,
    assert_special_token_ids,
)

DEFAULT_SYSTEM = "You are a helpful assistant."

IGNORE_INDEX = -100


# -------------------------
# Tokenizer loading
# -------------------------
def configure_chat_tokenizer(tok: Tokenizer) -> Tokenizer:
    """Make `tok.encode` treat special-token strings in raw text as plain text.

    All special tokens in this pipeline are inserted by ID by the code below,
    never parsed out of content — this closes the prompt-injection hole where a
    document containing the literal string "[EOS]" would encode to the real
    EOS id.
    """
    tok.encode_special_tokens = True
    return tok


def load_chat_tokenizer(tokenizer_path: str) -> Tokenizer:
    """Load tokenizer.json, assert the hardcoded special-token IDs, and disable
    special-token matching in raw text. Every chat-stage script should load its
    tokenizer through this."""
    assert_special_token_ids(tokenizer_path)
    return configure_chat_tokenizer(Tokenizer.from_file(tokenizer_path))


# -------------------------
# Text cleaning
# -------------------------
def norm_newlines(s: str) -> str:
    return (s or "").replace("\r\n", "\n").replace("\r", "\n")


def clean_text(s: str) -> str:
    # for system/user text: strip leading/trailing whitespace
    return norm_newlines(s).strip()


def clean_text_assistant(s: str) -> str:
    # IMPORTANT: do not strip assistant text (keeps code indentation / markdown
    # formatting); only trailing whitespace goes (EOS follows immediately).
    return norm_newlines(s).rstrip()


def _normalized_messages(
    messages: list[dict[str, str]], default_system: str
) -> list[dict[str, str]]:
    """Prepend the default system turn when missing; drop malformed/empty turns."""
    msgs = list(messages or [])
    if msgs and (msgs[0].get("role") or "").strip().lower() != "system" and default_system:
        msgs = [{"role": "system", "content": default_system}] + msgs
    out: list[dict[str, str]] = []
    for m in msgs:
        role = (m.get("role") or "").strip().lower()
        raw = m.get("content", "")
        txt = clean_text_assistant(raw) if role == "assistant" else clean_text(raw)
        if role not in ("system", "user", "assistant") or not txt.strip():
            continue
        out.append({"role": role, "content": txt})
    return out


# -------------------------
# Core encoding
# -------------------------
_ROLE_TOKEN_ID = {"system": SYSTEM_ID, "user": USER_ID, "assistant": ASSISTANT_ID}


def encode_chat(
    tok: Tokenizer,
    messages: list[dict[str, str]],
    default_system: str = DEFAULT_SYSTEM,
) -> tuple[list[int], list[int]]:
    """Encode a full conversation for training.

    Returns (ids, labels), same length. labels[i] == ids[i] on supervised
    positions (every assistant turn's content + its trailing EOS) and
    IGNORE_INDEX everywhere else (BOS, role tokens, system/user content).
    """
    msgs = _normalized_messages(messages, default_system)
    if not msgs:
        raise ValueError("missing messages")

    ids: list[int] = [BOS_ID]
    labels: list[int] = [IGNORE_INDEX]
    for m in msgs:
        role = m["role"]
        content_ids = tok.encode(m["content"]).ids
        ids.append(_ROLE_TOKEN_ID[role])
        labels.append(IGNORE_INDEX)
        if role == "assistant":
            ids.extend(content_ids)
            labels.extend(content_ids)
            ids.append(EOS_ID)
            labels.append(EOS_ID)
        else:
            ids.extend(content_ids)
            labels.extend([IGNORE_INDEX] * len(content_ids))
    return ids, labels


def encode_prompt(
    tok: Tokenizer,
    messages: list[dict[str, str]],
    default_system: str = DEFAULT_SYSTEM,
    mode: str = "full_context",
) -> list[int]:
    """Encode a generation prompt: context ending in the `<|assistant|>` cue.

    mode:
      - "full_context": all turns up to and including the LAST user turn
        (earlier assistant turns keep their [EOS]); trailing assistant turns
        after the last user turn are dropped (we are generating the reply).
      - "last_user": system turn + last user turn only.

    The returned ids start with BOS and end with ASSISTANT_ID, exactly matching
    the training-time prefix for an assistant turn.
    """
    msgs = _normalized_messages(messages, default_system)

    last_user = -1
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i]["role"] == "user":
            last_user = i
            break

    if mode == "last_user":
        keep = [m for m in msgs if m["role"] == "system"][:1]
        if last_user != -1:
            keep.append(msgs[last_user])
        msgs = keep
    elif mode == "full_context":
        if last_user != -1:
            msgs = msgs[: last_user + 1]
    else:
        raise ValueError(f"unknown prompt mode: {mode}")

    ids, _ = encode_chat(tok, msgs, default_system="") if msgs else ([BOS_ID], [IGNORE_INDEX])
    ids.append(ASSISTANT_ID)
    return ids


def encode_completion(
    tok: Tokenizer,
    messages: list[dict[str, str]],
    completion: str,
    default_system: str = DEFAULT_SYSTEM,
) -> tuple[list[int], list[int]]:
    """Encode (prompt context + one assistant completion) for DPO-style scoring.

    `messages` is the shared context (must end with a user turn); `completion`
    is a plain assistant string. Returns (ids, labels) where the supervised
    span is the completion tokens + trailing EOS — logps therefore include the
    stop decision.
    """
    ids = encode_prompt(tok, messages, default_system, mode="full_context")
    labels = [IGNORE_INDEX] * len(ids)
    comp_ids = tok.encode(clean_text_assistant(completion)).ids
    ids.extend(comp_ids)
    labels.extend(comp_ids)
    ids.append(EOS_ID)
    labels.append(EOS_ID)
    return ids, labels


def pad_or_truncate(
    ids: list[int],
    labels: list[int],
    seq_len: int,
    pad_id: int = PAD_ID,
) -> tuple[list[int], list[int]]:
    """Fixed-length batch shaping. Truncation keeps the TAIL so the supervised
    assistant span (which sits at the end) survives; padding is appended with
    IGNORE_INDEX labels."""
    if len(ids) > seq_len:
        return ids[-seq_len:], labels[-seq_len:]
    pad_n = seq_len - len(ids)
    return ids + [pad_id] * pad_n, labels + [IGNORE_INDEX] * pad_n


def count_chat_tokens(
    tok: Tokenizer,
    messages: list[dict[str, str]],
    default_system: str = DEFAULT_SYSTEM,
) -> int:
    """Exact token count of the training encoding (for mix/token budgeting)."""
    ids, _ = encode_chat(tok, messages, default_system)
    return len(ids)


def extract_last_user_and_ref(messages: list[dict[str, str]]) -> tuple[str, str]:
    """Return (last_user_text, last_assistant_text_if_any) — for sample logs."""
    last_user = ""
    ref = ""
    for m in reversed(messages or []):
        if (m.get("role") or "").strip().lower() == "user":
            last_user = clean_text(m.get("content", ""))
            break
    for m in reversed(messages or []):
        if (m.get("role") or "").strip().lower() == "assistant":
            ref = clean_text_assistant(m.get("content", ""))
            break
    return last_user, ref


def decode_completion(tok: Tokenizer, ids: list[int]) -> str:
    """Decode generated completion ids (the tokens AFTER the prompt), dropping
    a trailing EOS if present. Replaces the old fragile rfind('Assistant: ')."""
    if ids and ids[-1] == EOS_ID:
        ids = ids[:-1]
    return tok.decode(ids).strip() if ids else ""


# -------------------------
# Refusal detection (shared by SFT/distill example weighting)
# -------------------------
def is_refusal_text(text: str, patterns: list[str]) -> bool:
    """If assistant content contains any refusal-ish substring, treat as refusal."""
    t = (text or "").strip().lower()
    if not t:
        return False
    for p in patterns:
        p2 = p.strip().lower()
        if p2 and p2 in t:
            return True
    return False


def compute_example_weight_from_messages(
    messages: list[dict[str, str]],
    refusal_downweight: float,
    refusal_patterns: list[str],
    refusal_mode: str,
) -> float:
    """Scalar loss weight for a training example; downweights refusal-looking
    assistant turns (refusal_mode="contains_any")."""
    if refusal_downweight >= 1.0:
        return 1.0
    if refusal_downweight <= 0.0:
        return 0.0
    if refusal_mode != "contains_any":
        raise ValueError(f"unknown refusal_mode: {refusal_mode}")
    for m in messages or []:
        if (m.get("role") or "").strip().lower() == "assistant":
            if is_refusal_text(m.get("content", ""), refusal_patterns):
                return refusal_downweight
    return 1.0


def build_example(
    ex: dict[str, Any],
    tok: Tokenizer,
    seq_len: int,
    default_system: str,
    refusal_downweight: float,
    refusal_patterns: list[str],
    refusal_mode: str,
    pad_id: int = PAD_ID,
):
    """One SFT/distill training example -> (input_ids, labels, example_weight).

    Tensors are torch.long of length seq_len; labels use IGNORE_INDEX outside
    the supervised assistant spans. Honors meta.bucket safety exemption and
    meta.weight multipliers exactly as before.
    """
    import torch

    messages = ex.get("messages") or []
    if not messages:
        raise ValueError("missing messages")

    meta = ex.get("meta") or {}
    bucket = str(meta.get("bucket", "")).strip()
    # Do NOT downweight refusals inside the safety bucket (otherwise safety
    # examples get muted).
    refusal_dw_eff = 1.0 if bucket in ("D_safety", "D") else refusal_downweight
    ex_weight = compute_example_weight_from_messages(
        messages, refusal_dw_eff, refusal_patterns, refusal_mode
    )
    w0 = meta.get("weight", None) if isinstance(meta, dict) else None
    if isinstance(w0, (int, float)):
        ex_weight *= float(w0)

    ids, labels = encode_chat(tok, messages, default_system)
    ids, labels = pad_or_truncate(ids, labels, seq_len, pad_id)
    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        float(ex_weight),
    )
