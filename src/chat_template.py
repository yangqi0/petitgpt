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
    assert_tokenizer_contract,
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
    assert_tokenizer_contract(tokenizer_path)
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
    messages: list[dict[str, str]],
    default_system: str,
    *,
    expected_end: str | None,
) -> list[dict[str, str]]:
    """Clean and validate the canonical chat state machine.

    A chat has exactly one initial system turn, followed by non-empty user and
    assistant turns in strict alternation. ``expected_end`` is ``"user"`` for
    a generation prompt and ``"assistant"`` for a complete training sample.
    No malformed turn is silently skipped.
    """
    out: list[dict[str, str]] = []
    for index, message in enumerate(messages or []):
        if not isinstance(message, dict):
            raise ValueError(f"message {index} must be an object")
        role_value = message.get("role")
        role = role_value.strip().lower() if isinstance(role_value, str) else ""
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"message {index} has invalid role {role_value!r}")
        raw = message.get("content")
        if not isinstance(raw, str):
            raise ValueError(f"message {index} content must be a string")
        text = clean_text_assistant(raw) if role == "assistant" else clean_text(raw)
        if not text.strip():
            if index == 0 and role == "system":
                fallback = clean_text(default_system)
                if not fallback:
                    raise ValueError(
                        "chat requires a non-empty initial system turn or non-empty default_system"
                    )
                text = fallback
            else:
                raise ValueError(f"message {index} ({role}) content must be non-empty")
        out.append({"role": role, "content": text})

    if not out or out[0]["role"] != "system":
        fallback = clean_text(default_system)
        if not fallback:
            raise ValueError(
                "chat requires a non-empty initial system turn or non-empty default_system"
            )
        out.insert(0, {"role": "system", "content": fallback})

    for index, message in enumerate(out):
        expected = "system" if index == 0 else ("user" if index % 2 else "assistant")
        if message["role"] != expected:
            raise ValueError(
                "invalid chat role order: initial system must be followed by "
                f"alternating user/assistant turns (index {index}: expected "
                f"{expected!r}, got {message['role']!r})"
            )
    if len(out) == 1:
        raise ValueError("chat requires at least one non-empty user turn")
    if expected_end is not None and out[-1]["role"] != expected_end:
        raise ValueError(
            f"chat must end with a non-empty {expected_end} turn; got {out[-1]['role']!r}"
        )
    return out


def prepare_prompt_messages(
    messages: list[dict[str, str]],
    default_system: str = DEFAULT_SYSTEM,
) -> list[dict[str, str]]:
    """Explicitly turn a valid conversation/example into a USER-ending prompt.

    Callers sampling from a complete SFT example must opt in to removing its
    final assistant answer. ``encode_prompt`` itself never drops turns.
    """
    normalized = _normalized_messages(messages, default_system, expected_end=None)
    if normalized[-1]["role"] == "assistant":
        normalized = normalized[:-1]
    if normalized[-1]["role"] != "user":
        raise ValueError("generation prompt context must end with a user turn")
    return normalized


# -------------------------
# Core encoding
# -------------------------
_ROLE_TOKEN_ID = {"system": SYSTEM_ID, "user": USER_ID, "assistant": ASSISTANT_ID}


def _encode_normalized_chat(
    tok: Tokenizer, messages: list[dict[str, str]]
) -> tuple[list[int], list[int]]:
    ids: list[int] = [BOS_ID]
    labels: list[int] = [IGNORE_INDEX]
    for index, message in enumerate(messages):
        role = message["role"]
        content_ids = tok.encode(message["content"]).ids
        if not content_ids:
            raise ValueError(f"message {index} ({role}) must encode to at least one token")
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
    msgs = _normalized_messages(messages, default_system, expected_end="assistant")
    return _encode_normalized_chat(tok, msgs)


def encode_prompt(
    tok: Tokenizer,
    messages: list[dict[str, str]],
    default_system: str = DEFAULT_SYSTEM,
    mode: str = "full_context",
) -> list[int]:
    """Encode a generation prompt: context ending in the ``<|assistant|>`` cue.

    mode:
      - "full_context": the entire validated USER-ending context (earlier
        assistant turns keep their [EOS]).
      - "last_user": system turn + last user turn only.

    The returned ids start with BOS and end with ASSISTANT_ID, exactly matching
    the training-time prefix for an assistant turn.
    """
    msgs = _normalized_messages(messages, default_system, expected_end="user")

    if mode == "last_user":
        msgs = [msgs[0], msgs[-1]]
    elif mode != "full_context":
        raise ValueError(f"unknown prompt mode: {mode}")

    ids, _ = _encode_normalized_chat(tok, msgs)
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
    if not isinstance(completion, str) or not completion.strip():
        raise ValueError("DPO completion must be a non-empty, non-whitespace string")
    comp_ids = tok.encode(clean_text_assistant(completion)).ids
    if not comp_ids:
        raise ValueError("DPO completion must encode to at least one token")
    ids.extend(comp_ids)
    labels.extend(comp_ids)
    ids.append(EOS_ID)
    labels.append(EOS_ID)
    return ids, labels


def truncate_chat_sequence(
    ids: list[int],
    labels: list[int] | None,
    max_len: int,
) -> tuple[list[int], list[int] | None]:
    """Validate and truncate only at complete user-turn boundaries.

    The mandatory prefix ``BOS + SYSTEM + non-empty system content`` is always
    retained. The remainder is the largest recent suffix that begins at a real
    ``USER_ID`` marker and runs through the original sequence end. A training
    sequence must end with an assistant EOS; a prompt must end with the final
    assistant cue. Literal special-token text cannot create these boundaries
    because chat tokenizers disable special-token matching for raw content.

    If the system prefix plus the latest complete user-led suffix cannot fit,
    this function raises instead of cutting content, role markers, assistant
    completions, or EOS targets.
    """
    if max_len <= 0:
        raise ValueError("max_len must be positive")
    if labels is not None and len(labels) != len(ids):
        raise ValueError("ids and labels must have identical lengths")
    if len(ids) < 3 or ids[0] != BOS_ID or ids[1] != SYSTEM_ID:
        raise ValueError("chat sequence must start with BOS_ID, SYSTEM_ID, and system content")

    forbidden_content_ids = {PAD_ID, BOS_ID, EOS_ID, SYSTEM_ID, USER_ID, ASSISTANT_ID}
    role_ids = {SYSTEM_ID, USER_ID, ASSISTANT_ID}
    first_role = next((index for index in range(2, len(ids)) if ids[index] in role_ids), len(ids))
    if first_role == 2:
        raise ValueError("initial system content must contain at least one token")
    if any(token_id in forbidden_content_ids for token_id in ids[2:first_role]):
        raise ValueError("system content contains a structural special-token ID")
    if first_role == len(ids) or ids[first_role] != USER_ID:
        raise ValueError("chat sequence must contain a user turn after the initial system turn")

    user_starts: list[int] = []
    cursor = first_role
    while cursor < len(ids):
        if ids[cursor] != USER_ID:
            raise ValueError("chat role sequence must alternate USER and ASSISTANT")
        user_starts.append(cursor)
        user_content_start = cursor + 1
        assistant_pos = next(
            (
                index
                for index in range(user_content_start, len(ids))
                if ids[index] in forbidden_content_ids
            ),
            len(ids),
        )
        if assistant_pos == user_content_start:
            raise ValueError("user content must contain at least one token")
        if assistant_pos == len(ids) or ids[assistant_pos] != ASSISTANT_ID:
            raise ValueError("each user turn must be followed by an assistant turn")

        assistant_content_start = assistant_pos + 1
        if assistant_content_start == len(ids):
            if labels is not None:
                raise ValueError("training chat must end with assistant content and EOS")
            cursor = len(ids)
            break

        eos_pos = next(
            (
                index
                for index in range(assistant_content_start, len(ids))
                if ids[index] in forbidden_content_ids
            ),
            len(ids),
        )
        if eos_pos == assistant_content_start:
            raise ValueError("assistant content must contain at least one token")
        if eos_pos == len(ids) or ids[eos_pos] != EOS_ID:
            raise ValueError("each assistant completion must end with EOS")
        cursor = eos_pos + 1
        if cursor == len(ids):
            if labels is None:
                raise ValueError("generation prompt must end with an assistant cue")
            break

    if labels is not None:
        if ids[-1] != EOS_ID:
            raise ValueError("training chat must end with a supervised assistant EOS")
        if labels[-1] != EOS_ID:
            raise ValueError("final assistant EOS must be supervised")

    prefix_end = first_role
    chosen_start = next(
        (start for start in user_starts if prefix_end + (len(ids) - start) <= max_len),
        None,
    )
    if chosen_start is None:
        minimum = prefix_end + (len(ids) - user_starts[-1])
        raise ValueError(
            "chat sequence does not fit without cutting the system prefix or latest "
            f"user-led suffix (requires at least {minimum} tokens, max_len={max_len})"
        )

    if chosen_start == prefix_end:
        return list(ids), list(labels) if labels is not None else None

    kept_ids = ids[:prefix_end] + ids[chosen_start:]
    if labels is None:
        return kept_ids, None
    return kept_ids, labels[:prefix_end] + labels[chosen_start:]


def pad_or_truncate(
    ids: list[int],
    labels: list[int],
    seq_len: int,
    pad_id: int = PAD_ID,
) -> tuple[list[int], list[int]]:
    """Structure-aware fixed-length shaping plus right padding."""
    ids, maybe_labels = truncate_chat_sequence(ids, labels, seq_len)
    assert maybe_labels is not None
    labels = maybe_labels
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
    if all(label == IGNORE_INDEX for label in labels):
        raise ValueError("SFT example requires a retained non-empty assistant turn")
    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        float(ex_weight),
    )
