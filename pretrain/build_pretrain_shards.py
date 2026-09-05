#!/usr/bin/env python3

"""Build auditable packed pretraining shards from JSONL sources.

Production defaults enforce the repository-wide contract:

* the tokenizer satisfies the full canonical 32k/seven-special-token contract;
* every document is [BOS] content [EOS];
* no textual document separator is inserted;
* a source that cannot satisfy its quota fails instead of silently replaying;
* validation membership is assigned by a stable content hash, not file order;
* token accounting separates content, structural boundaries, legacy separators,
  and emitted shard tokens.
* shards are built in a sibling staging directory and atomically published.
* every shard is bound by ordered path/size/token-count/SHA-256 metadata.

The two --legacy_* switches are deliberately conspicuous escape hatches for
old/debug runs. Their use is persisted in meta.json and is never implicit.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile
from typing import Any
import uuid

import numpy as np
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.special_tokens import (  # noqa: E402
    BOS_ID,
    CANONICAL_VOCAB_SIZE,
    EOS_ID,
    SPECIAL_TOKEN_IDS,
    assert_tokenizer_contract,
)


class SourceExhaustedError(RuntimeError):
    """A finite source ended before its serialized train/validation quotas."""


_HASH_SPACE = 1 << 64
_VALIDATION_HASH_ALGORITHM = "blake2b-64-clean-text-v1"
CLEANED_TEXT_HASH_ALGORITHM = "sha256-cleaned-text-utf8-v1"
EXCLUSION_MANIFEST_KIND = "petitgpt_reference_validation_exclusions"


def _write_json_atomic(path: Path, obj: dict[str, Any]) -> None:
    """Write JSON durably without exposing a half-written manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def cleaned_text_sha256(text: str) -> str:
    """Return the canonical cross-build identity for a cleaned document."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cleaning_contract(
    *,
    strip_leading_noise: bool,
    normalize_quotes: bool,
    underscores_policy: str,
    min_chars: int,
    min_ascii_ratio: float,
) -> dict[str, Any]:
    """Describe every transform/filter that affects exclusion hash identity."""
    return {
        "strip_leading_noise": bool(strip_leading_noise),
        "normalize_quotes": bool(normalize_quotes),
        "underscores_policy": str(underscores_policy),
        "min_chars": int(min_chars),
        "min_ascii_ratio": float(min_ascii_ratio),
    }


def load_exclusion_hash_manifest(
    path: Path,
    *,
    expected_cleaning: dict[str, Any] | None = None,
) -> tuple[frozenset[str], dict[str, Any]]:
    """Load and strictly validate a reference-validation exclusion manifest."""
    try:
        before = path.stat()
        raw_manifest = path.read_bytes()
        after = path.stat()
        before_identity = (
            before.st_size,
            before.st_mtime_ns,
            before.st_ino,
            before.st_dev,
        )
        after_identity = (
            after.st_size,
            after.st_mtime_ns,
            after.st_ino,
            after.st_dev,
        )
        if before_identity != after_identity:
            raise RuntimeError(f"exclusion hash manifest changed while reading: {path}")
        manifest = json.loads(raw_manifest)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read exclusion hash manifest {path}: {exc}") from exc

    if not isinstance(manifest, dict):
        raise ValueError("exclusion hash manifest must be a JSON object")
    if manifest.get("kind") != EXCLUSION_MANIFEST_KIND:
        raise ValueError(
            "unsupported exclusion manifest kind: "
            f"{manifest.get('kind')!r}; expected {EXCLUSION_MANIFEST_KIND!r}"
        )
    if manifest.get("hash_algorithm") != CLEANED_TEXT_HASH_ALGORITHM:
        raise ValueError(
            "unsupported exclusion hash algorithm: "
            f"{manifest.get('hash_algorithm')!r}; "
            f"expected {CLEANED_TEXT_HASH_ALGORITHM!r}"
        )
    manifest_cleaning = manifest.get("cleaning")
    if not isinstance(manifest_cleaning, dict):
        raise ValueError("exclusion hash manifest 'cleaning' must be an object")
    if expected_cleaning is not None and manifest_cleaning != expected_cleaning:
        raise ValueError(
            "reference-validation and train cleaning contracts differ; "
            f"reference={manifest_cleaning!r}, train={expected_cleaning!r}"
        )

    raw_hashes = manifest.get("hashes")
    if not isinstance(raw_hashes, list):
        raise ValueError("exclusion hash manifest 'hashes' must be a list")
    hashes: list[str] = []
    for value in raw_hashes:
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"invalid cleaned-text SHA-256 in exclusion manifest: {value!r}")
        hashes.append(value)
    if len(set(hashes)) != len(hashes):
        raise ValueError("exclusion hash manifest contains duplicate hashes")
    declared_count = manifest.get("hash_count")
    if declared_count is not None and int(declared_count) != len(hashes):
        raise ValueError(
            f"exclusion hash_count={declared_count}, but hashes contains {len(hashes)} entries"
        )

    return frozenset(hashes), {
        "enabled": True,
        "manifest_path": str(path),
        "manifest_resolved": str(path.resolve()),
        "manifest_size_bytes": len(raw_manifest),
        "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "kind": manifest["kind"],
        "hash_algorithm": manifest["hash_algorithm"],
        "hash_count": len(hashes),
        "cleaning": manifest_cleaning,
    }


def validate_build_contract(
    tokenizer_path: str,
    *,
    add_bos: bool,
    add_eos: bool,
    bos_id: int,
    eos_id: int,
    doc_sep: str,
    legacy_allow_noncanonical_contract: bool = False,
) -> dict[str, Any]:
    """Validate the production tokenizer/boundary contract before any writes."""
    issues: list[str] = []

    try:
        assert_tokenizer_contract(tokenizer_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        issues.append(f"canonical tokenizer contract failed: {exc}")

    try:
        with open(tokenizer_path, encoding="utf-8") as f:
            tokenizer_json = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read tokenizer contract from {tokenizer_path}: {exc}") from exc

    special_entries = [
        item for item in (tokenizer_json.get("added_tokens") or []) if item.get("special") is True
    ]
    registered_specials = {item.get("content"): item.get("id") for item in special_entries}
    model_vocab = (tokenizer_json.get("model") or {}).get("vocab") or {}
    runtime_ids = set(model_vocab.values())
    runtime_ids.update(
        item.get("id")
        for item in (tokenizer_json.get("added_tokens") or [])
        if isinstance(item, dict) and isinstance(item.get("id"), int)
    )
    actual_vocab_size = len(runtime_ids)
    if len(special_entries) != len(SPECIAL_TOKEN_IDS) or registered_specials != SPECIAL_TOKEN_IDS:
        issues.append(
            "tokenizer must register exactly the seven canonical special tokens; "
            f"got {registered_specials!r}"
        )
    if tokenizer_json.get("post_processor") not in (None, {}):
        issues.append("tokenizer post_processor must be absent/disabled for shard building")
    if not add_bos:
        issues.append("BOS insertion is disabled")
    if not add_eos:
        issues.append("EOS insertion is disabled")
    if int(bos_id) != BOS_ID:
        issues.append(f"bos_id={bos_id}, expected {BOS_ID}")
    if int(eos_id) != EOS_ID:
        issues.append(f"eos_id={eos_id}, expected {EOS_ID}")
    if doc_sep:
        issues.append("textual doc_sep is non-empty")

    if issues and not legacy_allow_noncanonical_contract:
        joined = "; ".join(issues)
        raise ValueError(
            f"non-canonical pretraining build refused: {joined}. "
            "Only legacy/debug runs may pass --legacy_allow_noncanonical_contract."
        )

    return {
        "mode": "legacy_noncanonical" if issues else "canonical",
        "canonical": not issues,
        "legacy_allow_noncanonical_contract": bool(legacy_allow_noncanonical_contract),
        "issues": issues,
        "expected_special_token_ids": dict(SPECIAL_TOKEN_IDS),
        "expected_vocab_size": CANONICAL_VOCAB_SIZE,
        "actual_vocab_size": actual_vocab_size,
        "add_bos": bool(add_bos),
        "add_eos": bool(add_eos),
        "bos_id": int(bos_id),
        "eos_id": int(eos_id),
        "doc_sep": doc_sep,
    }


def is_validation_holdout(
    text: str,
    *,
    train_target_tokens: int,
    val_target_tokens: int,
    seed: int,
) -> bool:
    """Assign a cleaned document to a stable validation partition.

    The comparison is integer-only, so membership is reproducible across Python
    versions and independent of source order.
    """
    train_target = max(0, int(train_target_tokens))
    val_target = max(0, int(val_target_tokens))
    if val_target == 0:
        return False
    if train_target == 0:
        return True

    h = hashlib.blake2b(digest_size=8, person=b"PetitGPT-val-v1")
    h.update(str(int(seed)).encode("ascii"))
    h.update(b"\0")
    h.update(text.encode("utf-8"))
    bucket = int.from_bytes(h.digest(), "big", signed=False)
    return bucket * (train_target + val_target) < val_target * _HASH_SPACE


# -------------------------
# IO utils
# -------------------------
def iter_jsonl_texts(path: Path, field: str = "text") -> Iterator[str]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            t = obj.get(field, None)
            if isinstance(t, str) and t:
                yield t


def file_fingerprint(p: Path) -> dict[str, Any]:
    before = p.stat()
    sha256 = _file_sha256(p)
    after = p.stat()
    before_identity = (
        int(before.st_size),
        int(before.st_mtime_ns),
        int(getattr(before, "st_ino", 0)),
        int(getattr(before, "st_dev", 0)),
    )
    after_identity = (
        int(after.st_size),
        int(after.st_mtime_ns),
        int(getattr(after, "st_ino", 0)),
        int(getattr(after, "st_dev", 0)),
    )
    if before_identity != after_identity:
        raise RuntimeError(f"source changed while hashing: {p}")
    return {
        "path": str(p),
        "resolved": str(p.resolve()),
        "size": int(after.st_size),
        "mtime": float(after.st_mtime),
        "mtime_ns": int(after.st_mtime_ns),
        "inode": int(getattr(after, "st_ino", 0)),
        "device": int(getattr(after, "st_dev", 0)),
        "sha256": sha256,
    }


def fast_count_lines(p: Path, max_lines: int) -> dict[str, Any]:
    n = 0
    with open(p, "rb") as f:
        for _ in f:
            n += 1
            if n >= max_lines:
                return {"lines": int(n), "at_least": True, "max_lines": int(max_lines)}
    return {"lines": int(n), "at_least": False, "max_lines": int(max_lines)}


# -------------------------
# Tokenizer helpers
# -------------------------
def load_tokenizer(tokenizer_path: str) -> Tokenizer:
    tok = Tokenizer.from_file(tokenizer_path)
    # Treat special-token strings occurring in corpus text as plain text.
    # BOS/EOS are inserted by ID below — a document containing the literal
    # string "[EOS]" must never inject a real EOS token into the shards.
    tok.encode_special_tokens = True
    return tok


def _tokenizer_vocab_size(tok: Tokenizer) -> int | None:
    # tokenizers Tokenizer doesn't expose vocab_size directly in all cases;
    # we can get it via get_vocab() when available.
    try:
        v = tok.get_vocab()
        return int(len(v))
    except Exception:
        return None


def _encode_base(tok: Tokenizer, text: str) -> list[int]:
    enc = tok.encode(text)
    return list(enc.ids)


def encode_with_accounting(
    tok: Tokenizer,
    text: str,
    *,
    add_bos: bool,
    add_eos: bool,
    bos_id: int,
    eos_id: int,
) -> tuple[list[int], int, int]:
    """Encode content and report content/boundary token counts separately."""
    content_ids = _encode_base(tok, text)
    ids: list[int] = []
    if add_bos:
        ids.append(int(bos_id))
    ids.extend(content_ids)
    if add_eos:
        ids.append(int(eos_id))
    boundary_tokens = int(bool(add_bos)) + int(bool(add_eos))
    return ids, len(content_ids), boundary_tokens


def encode(
    tok: Tokenizer,
    text: str,
    *,
    add_bos: bool,
    add_eos: bool,
    bos_id: int,
    eos_id: int,
) -> list[int]:
    ids, _, _ = encode_with_accounting(
        tok,
        text,
        add_bos=add_bos,
        add_eos=add_eos,
        bos_id=bos_id,
        eos_id=eos_id,
    )
    return ids


def assert_token_ids_ok(
    ids: list[int],
    *,
    vocab_size: int | None,
    dtype: np.dtype,
    src: str,
    text_preview: str,
) -> None:
    # type + sign checks
    for x in ids:
        if not isinstance(x, int):
            raise AssertionError(f"[token-id] non-int id: {type(x)} from src={src}")
        if x < 0:
            raise AssertionError(f"[token-id] negative id={x} from src={src}")

    # vocab range checks (strong)
    if vocab_size is not None:
        mx = max(ids)
        if mx >= vocab_size:
            raise AssertionError(
                f"[token-id] out-of-range id={mx} >= vocab_size={vocab_size} from src={src}\n"
                f"text_preview={text_preview!r}"
            )

    # dtype overflow checks
    if dtype == np.uint16:
        mx = max(ids)
        if mx > 65535:
            raise AssertionError(
                f"[token-id] uint16 overflow risk: max_id={mx} > 65535 from src={src}\n"
                f"text_preview={text_preview!r}"
            )


# -------------------------
# Cleaning / filters (kept minimal; you already cleaned upstream)
# -------------------------
_RE_LEADING_NOISE = re.compile(r"^\s*(?:\ufeff|<!--.*?-->|<\?xml.*?\?>)+", re.DOTALL)


def ascii_ratio(s: str) -> float:
    if not s:
        return 0.0
    ascii_cnt = sum(1 for ch in s if ord(ch) < 128)
    return ascii_cnt / max(1, len(s))


def clean_text(
    t: str,
    *,
    strip_leading_noise: bool,
    normalize_quotes: bool,
    underscores_policy: str,
    min_chars: int,
    min_ascii_ratio: float,
) -> str | None:
    if t is None:
        return None
    if not isinstance(t, str):
        return None
    t = t.strip("\n\r")
    if strip_leading_noise:
        t = _RE_LEADING_NOISE.sub("", t)
    if normalize_quotes:
        t = t.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    if underscores_policy == "space":
        t = t.replace("_", " ")
    elif underscores_policy == "remove":
        t = t.replace("_", "")
    if min_chars > 0 and len(t) < min_chars:
        return None
    if min_ascii_ratio > 0.0 and ascii_ratio(t) < min_ascii_ratio:
        return None
    return t if t else None


# -------------------------
# Source parsing
# -------------------------
def parse_sources(source_args: list[str]) -> list[tuple[Path, float]]:
    items: list[tuple[Path, float]] = []
    for s in source_args:
        if ":" not in s:
            raise ValueError(f"--source must be like path:weight, got {s!r}")
        p, w = s.rsplit(":", 1)
        p = p.strip()
        w = w.strip()
        wf = float(w)
        if wf <= 0:
            raise ValueError(f"--source weight must be > 0, got {wf} in {s!r}")
        items.append((Path(p), wf))
    tot = sum(w for _, w in items) if items else 1.0
    return [(p, w / tot) for p, w in items]


def topk_counter(c: Counter[int], k: int) -> list[dict[str, Any]]:
    return [{"id": int(tid), "count": int(cnt)} for tid, cnt in c.most_common(k)]


# -------------------------
# Per-source iterators + token-quota scheduler
# -------------------------
@dataclass
class SrcState:
    path: Path
    weight: float
    it: Iterator[str]
    name: str
    pass_kept_docs: int = 0
    exhaustion_count: int = 0
    replay_count: int = 0


def choose_src_by_remaining(
    rng: random.Random,
    states: list[SrcState],
    remaining: dict[str, int],
) -> SrcState:
    """Choose proportionally to outstanding serialized train+validation quota."""
    live = [(st, max(0, int(remaining[st.name]))) for st in states if remaining[st.name] > 0]
    if not live:
        raise RuntimeError("source scheduler called with no remaining quota")
    total = sum(value for _, value in live)
    draw = rng.randrange(total)
    acc = 0
    for st, value in live:
        acc += value
        if draw < acc:
            return st
    return live[-1][0]


def _allocate_weighted_tokens(total_tokens: int, states: list[SrcState]) -> dict[str, int]:
    """Largest-remainder allocation whose integer targets sum exactly."""
    raw = [(st.name, st.weight * int(total_tokens)) for st in states]
    result: dict[str, int] = {}
    floor_sum = 0
    fractions: list[tuple[float, str]] = []
    for name, value in raw:
        floor_value = int(value)
        result[name] = floor_value
        floor_sum += floor_value
        fractions.append((value - floor_value, name))
    fractions.sort(reverse=True)
    for i in range(max(0, int(total_tokens) - floor_sum)):
        result[fractions[i % len(fractions)][1]] += 1
    return result


def _empty_split_accounting() -> dict[str, int]:
    return {
        "documents": 0,
        "content_tokens": 0,
        "boundary_tokens": 0,
        "serialized_tokens": 0,
        "separator_tokens": 0,
        "emitted_shard_tokens": 0,
    }


# -------------------------
# Shard writer
# -------------------------
def _write_shards_to_directory(
    *,
    sources: list[tuple[Path, float]],
    out_dir: Path,
    tokenizer_path: str,
    shard_tokens: int,
    val_shard_tokens: int,
    val_ratio: float,
    min_val_tokens_per_source: int,
    seed: int,
    add_bos: bool,
    add_eos: bool,
    bos_id: int,
    eos_id: int,
    target_train_tokens: int,
    precheck_max_lines: int,
    doc_sep: str,
    first_token_topk: int,
    strip_leading_noise: bool,
    normalize_quotes: bool,
    underscores_policy: str,
    min_chars: int,
    min_ascii_ratio: float,
    validation_hash_seed: int = 1234,
    legacy_allow_noncanonical_contract: bool = False,
    legacy_replay_on_exhaustion: bool = False,
    exclude_hash_manifests: list[Path] | None = None,
) -> dict[str, Any]:
    """Build shards and return the exact manifest written to meta.json."""
    tokenizer_file = Path(tokenizer_path)
    tokenizer_fingerprint = file_fingerprint(tokenizer_file)
    contract = validate_build_contract(
        tokenizer_path,
        add_bos=add_bos,
        add_eos=add_eos,
        bos_id=bos_id,
        eos_id=eos_id,
        doc_sep=doc_sep,
        legacy_allow_noncanonical_contract=legacy_allow_noncanonical_contract,
    )
    if file_fingerprint(tokenizer_file) != tokenizer_fingerprint:
        raise RuntimeError("tokenizer changed during contract validation")
    tokenizer_sha256 = str(tokenizer_fingerprint["sha256"])
    if not sources:
        raise ValueError("at least one source is required")
    source_names = [str(path) for path, _ in sources]
    if len(set(source_names)) != len(source_names):
        raise ValueError("duplicate --source paths are not allowed")
    for path, _ in sources:
        if not path.is_file():
            raise FileNotFoundError(f"source does not exist or is not a file: {path}")

    active_cleaning_contract = cleaning_contract(
        strip_leading_noise=strip_leading_noise,
        normalize_quotes=normalize_quotes,
        underscores_policy=underscores_policy,
        min_chars=min_chars,
        min_ascii_ratio=min_ascii_ratio,
    )
    exclusion_hash_sets: list[frozenset[str]] = []
    exclusion_hashes_mutable: set[str] = set()
    exclusion_meta: dict[str, Any] = {
        "enabled": False,
        "hash_algorithm": CLEANED_TEXT_HASH_ALGORITHM,
        "manifest_count": 0,
        "union_hash_count": 0,
        "cleaning": active_cleaning_contract,
        "manifests": [],
    }
    for manifest_path in exclude_hash_manifests or []:
        hashes, loaded_meta = load_exclusion_hash_manifest(
            Path(manifest_path), expected_cleaning=active_cleaning_contract
        )
        exclusion_hash_sets.append(hashes)
        exclusion_hashes_mutable.update(hashes)
        exclusion_meta["manifests"].append({
            **loaded_meta,
            "matched_documents": 0,
            "matched_per_source": {},
        })
    exclusion_hashes = frozenset(exclusion_hashes_mutable)
    exclusion_meta["enabled"] = bool(exclusion_hash_sets)
    exclusion_meta["manifest_count"] = len(exclusion_hash_sets)
    exclusion_meta["union_hash_count"] = len(exclusion_hashes)

    out_train = out_dir / "train"
    out_val = out_dir / "val"
    out_val_by_src = out_dir / "val_by_source"
    out_train.mkdir(parents=True, exist_ok=True)
    out_val.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    tok = load_tokenizer(tokenizer_path)
    if file_fingerprint(tokenizer_file) != tokenizer_fingerprint:
        raise RuntimeError("tokenizer changed while loading")
    vocab_size = _tokenizer_vocab_size(tok)
    dtype = np.uint32 if (vocab_size is not None and vocab_size > 65535) else np.uint16
    doc_sep_ids = _encode_base(tok, doc_sep) if doc_sep else []

    states = [
        SrcState(path=path, weight=weight, it=iter_jsonl_texts(path), name=str(path))
        for path, weight in sources
    ]
    state_by_name = {st.name: st for st in states}

    train_target_per_src = _allocate_weighted_tokens(target_train_tokens, states)
    default_total_val = int(round(target_train_tokens * val_ratio))
    val_target_per_src = _allocate_weighted_tokens(default_total_val, states)
    for st in states:
        val_target_per_src[st.name] = max(
            val_target_per_src[st.name], int(min_val_tokens_per_source)
        )

    train_remaining = dict(train_target_per_src)
    val_remaining = dict(val_target_per_src)

    per_src: dict[str, Any] = {}
    source_fingerprints: dict[str, Any] = {}
    source_line_precheck: dict[str, Any] = {}
    first_tok_all: Counter[int] = Counter()
    first_tok_per_src: dict[str, Counter[int]] = {}

    src_dirname: dict[str, str] = {}
    used_names: set[str] = set()
    for st in states:
        base = re.sub(r"[^A-Za-z0-9_.-]", "_", st.path.stem) or "source"
        dirname = base
        suffix = 2
        while dirname in used_names:
            dirname = f"{base}_{suffix}"
            suffix += 1
        used_names.add(dirname)
        src_dirname[st.name] = dirname
        source_fingerprints[st.name] = file_fingerprint(st.path)
        source_line_precheck[st.name] = fast_count_lines(st.path, max_lines=precheck_max_lines)
        first_tok_per_src[st.name] = Counter()
        per_src[st.name] = {
            "weight": float(st.weight),
            "targets": {
                "train_serialized_tokens": int(train_target_per_src[st.name]),
                "val_serialized_tokens": int(val_target_per_src[st.name]),
            },
            "realized": {
                "train": _empty_split_accounting(),
                "val": _empty_split_accounting(),
            },
            "seen_docs": 0,
            "eligible_docs": 0,
            "kept_docs": 0,
            "excluded_reference": {
                "documents": 0,
                "cleaned_chars": 0,
                "content_tokens": 0,
                "boundary_tokens": 0,
                "serialized_tokens": 0,
            },
            "dropped_short": 0,
            "dropped_ascii": 0,
            "dropped_empty": 0,
            "holdout": {
                "algorithm": _VALIDATION_HASH_ALGORITHM,
                "seed": int(validation_hash_seed),
                "validation_candidate_docs": 0,
                "training_candidate_docs": 0,
                "skipped_validation_after_quota": 0,
                "skipped_training_after_quota": 0,
            },
            "exhaustion": {
                "count": 0,
                "replay_count": 0,
                "exhausted_before_quota": False,
                "remaining_train_serialized_tokens": 0,
                "remaining_val_serialized_tokens": 0,
            },
        }

    accounting = {
        "train": _empty_split_accounting(),
        "val": _empty_split_accounting(),
    }

    buf_train: list[int] = []
    buf_val: list[int] = []
    buf_val_src: dict[str, list[int]] = {st.name: [] for st in states}
    shard_idx_train = 0
    shard_idx_val = 0
    shard_idx_val_src: dict[str, int] = {st.name: 0 for st in states}
    total_train = 0
    total_val = 0
    total_val_src: dict[str, int] = {st.name: 0 for st in states}
    shard_files_train: list[dict[str, Any]] = []
    shard_files_val: list[dict[str, Any]] = []
    shard_files_val_src: dict[str, list[dict[str, Any]]] = {
        src_dirname[st.name]: [] for st in states
    }

    seen_docs = 0
    kept_docs = 0
    excluded_reference = {
        "documents": 0,
        "cleaned_chars": 0,
        "content_tokens": 0,
        "boundary_tokens": 0,
        "serialized_tokens": 0,
    }
    dropped_short = 0
    dropped_ascii = 0
    dropped_empty = 0

    def flush(buf: list[int], out_path: Path) -> dict[str, Any]:
        """Write one shard and fingerprint the exact in-memory byte image."""
        arr = np.asarray(buf, dtype=dtype)
        sha256 = hashlib.sha256(memoryview(arr).cast("B")).hexdigest()
        with open(out_path, "wb") as handle:
            arr.tofile(handle)
            handle.flush()
            os.fsync(handle.fileno())
        if out_path.stat().st_size != arr.nbytes:
            raise OSError(
                f"short shard write for {out_path}: "
                f"actual={out_path.stat().st_size}, expected={arr.nbytes}"
            )
        return {
            "path": out_path.relative_to(out_dir).as_posix(),
            "size_bytes": int(arr.nbytes),
            "token_count": int(arr.size),
            "sha256": sha256,
        }

    def maybe_flush_train() -> None:
        nonlocal buf_train, shard_idx_train, total_train
        while len(buf_train) >= shard_tokens:
            chunk = buf_train[:shard_tokens]
            del buf_train[:shard_tokens]
            path = out_train / f"shard_{shard_idx_train:05d}.bin"
            record = flush(chunk, path)
            shard_files_train.append(record)
            total_train += int(record["token_count"])
            shard_idx_train += 1

    def maybe_flush_val() -> None:
        nonlocal buf_val, shard_idx_val, total_val
        while len(buf_val) >= val_shard_tokens:
            chunk = buf_val[:val_shard_tokens]
            del buf_val[:val_shard_tokens]
            path = out_val / f"shard_{shard_idx_val:05d}.bin"
            record = flush(chunk, path)
            shard_files_val.append(record)
            total_val += int(record["token_count"])
            shard_idx_val += 1

    def flush_val_src(src_name: str, force: bool = False) -> None:
        buf = buf_val_src[src_name]
        directory = out_val_by_src / src_dirname[src_name]
        while len(buf) >= val_shard_tokens or (force and buf):
            n_take = min(val_shard_tokens, len(buf))
            chunk = buf[:n_take]
            del buf[:n_take]
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"shard_{shard_idx_val_src[src_name]:05d}.bin"
            record = flush(chunk, path)
            shard_files_val_src[src_dirname[src_name]].append(record)
            total_val_src[src_name] += int(record["token_count"])
            shard_idx_val_src[src_name] += 1

    def outstanding() -> dict[str, int]:
        return {
            st.name: max(0, train_remaining[st.name]) + max(0, val_remaining[st.name])
            for st in states
        }

    def failure_manifest(error: str) -> dict[str, Any]:
        return {
            "schema_version": 3,
            "status": "failed",
            "error": error,
            "tokenizer_path": tokenizer_path,
            "tokenizer_sha256": tokenizer_sha256,
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "contract": contract,
            "source_fingerprints": source_fingerprints,
            "source_exhaustion_policy": (
                "legacy_replay" if legacy_replay_on_exhaustion else "fail_fast"
            ),
            "validation_holdout": {
                "algorithm": _VALIDATION_HASH_ALGORITHM,
                "seed": int(validation_hash_seed),
            },
            "reference_validation_exclusion": {
                **exclusion_meta,
                "matched": excluded_reference,
            },
            "targets": {
                "train_serialized_tokens": int(target_train_tokens),
                "val_serialized_tokens": int(sum(val_target_per_src.values())),
                "train_per_source": train_target_per_src,
                "val_per_source": val_target_per_src,
            },
            "realized": accounting,
            "remaining": {
                "train_per_source": train_remaining,
                "val_per_source": val_remaining,
            },
            "per_source": per_src,
        }

    def handle_source_exhaustion(st: SrcState) -> None:
        st.exhaustion_count += 1
        info = per_src[st.name]["exhaustion"]
        info["count"] = st.exhaustion_count
        info["exhausted_before_quota"] = True
        info["remaining_train_serialized_tokens"] = int(max(0, train_remaining[st.name]))
        info["remaining_val_serialized_tokens"] = int(max(0, val_remaining[st.name]))
        message = (
            f"source exhausted before quota: {st.path}; "
            f"train_remaining={max(0, train_remaining[st.name])}; "
            f"val_remaining={max(0, val_remaining[st.name])}"
        )

        if not legacy_replay_on_exhaustion:
            _write_json_atomic(out_dir / "meta.failed.json", failure_manifest(message))
            raise SourceExhaustedError(message)
        if st.pass_kept_docs == 0:
            message += "; the completed pass emitted no documents, so replay cannot progress"
            _write_json_atomic(out_dir / "meta.failed.json", failure_manifest(message))
            raise SourceExhaustedError(message)

        st.replay_count += 1
        info["replay_count"] = st.replay_count
        st.pass_kept_docs = 0
        st.it = iter_jsonl_texts(st.path)

    def route_document(src_name: str, text: str) -> str | None:
        holdout = is_validation_holdout(
            text,
            train_target_tokens=train_target_per_src[src_name],
            val_target_tokens=val_target_per_src[src_name],
            seed=validation_hash_seed,
        )
        holdout_meta = per_src[src_name]["holdout"]
        if holdout:
            holdout_meta["validation_candidate_docs"] += 1
            if val_remaining[src_name] > 0:
                return "val"
            holdout_meta["skipped_validation_after_quota"] += 1
            return None

        holdout_meta["training_candidate_docs"] += 1
        if train_remaining[src_name] > 0:
            return "train"
        holdout_meta["skipped_training_after_quota"] += 1
        return None

    def add_doc(
        src_name: str,
        split: str,
        ids: list[int],
        *,
        content_tokens: int,
        boundary_tokens: int,
    ) -> None:
        nonlocal kept_docs, buf_train, buf_val
        source_realized = per_src[src_name]["realized"][split]
        split_accounting = accounting[split]
        separator_count = 0
        serialized_count = int(content_tokens) + int(boundary_tokens)

        if split == "train":
            if doc_sep_ids and split_accounting["documents"] > 0:
                buf_train.extend(doc_sep_ids)
                separator_count = len(doc_sep_ids)
            buf_train.extend(ids)
        else:
            if doc_sep_ids and split_accounting["documents"] > 0:
                buf_val.extend(doc_sep_ids)
                separator_count = len(doc_sep_ids)
            buf_val.extend(ids)
            source_separator_count = 0
            if doc_sep_ids and source_realized["documents"] > 0:
                buf_val_src[src_name].extend(doc_sep_ids)
                source_separator_count = len(doc_sep_ids)
            buf_val_src[src_name].extend(ids)
            source_realized["val_by_source_separator_tokens"] = (
                source_realized.get("val_by_source_separator_tokens", 0) + source_separator_count
            )

        for target in (source_realized, split_accounting):
            target["documents"] += 1
            target["content_tokens"] += int(content_tokens)
            target["boundary_tokens"] += int(boundary_tokens)
            target["serialized_tokens"] += serialized_count
            target["separator_tokens"] += separator_count

        per_src[src_name]["kept_docs"] += 1
        kept_docs += 1
        state_by_name[src_name].pass_kept_docs += 1
        first_tok_all[ids[0]] += 1
        first_tok_per_src[src_name][ids[0]] += 1

        if split == "train":
            train_remaining[src_name] = max(0, train_remaining[src_name] - serialized_count)
            maybe_flush_train()
        else:
            val_remaining[src_name] = max(0, val_remaining[src_name] - serialized_count)
            maybe_flush_val()
            flush_val_src(src_name)

    while any(value > 0 for value in outstanding().values()):
        st = choose_src_by_remaining(rng, states, outstanding())
        src_name = st.name
        try:
            text = next(st.it)
        except StopIteration:
            handle_source_exhaustion(st)
            try:
                text = next(st.it)
            except StopIteration as exc:
                message = f"source is empty and cannot be replayed: {st.path}"
                _write_json_atomic(out_dir / "meta.failed.json", failure_manifest(message))
                raise SourceExhaustedError(message) from exc

        seen_docs += 1
        per_src[src_name]["seen_docs"] += 1

        cleaned = clean_text(
            text,
            strip_leading_noise=strip_leading_noise,
            normalize_quotes=normalize_quotes,
            underscores_policy=underscores_policy,
            min_chars=0,
            min_ascii_ratio=0.0,
        )
        if cleaned is None:
            per_src[src_name]["dropped_empty"] += 1
            dropped_empty += 1
            continue
        if min_chars > 0 and len(cleaned) < min_chars:
            per_src[src_name]["dropped_short"] += 1
            dropped_short += 1
            continue
        if min_ascii_ratio > 0.0 and ascii_ratio(cleaned) < min_ascii_ratio:
            per_src[src_name]["dropped_ascii"] += 1
            dropped_ascii += 1
            continue

        per_src[src_name]["eligible_docs"] += 1
        clean_hash = cleaned_text_sha256(cleaned)
        if clean_hash in exclusion_hashes:
            for index, hashes in enumerate(exclusion_hash_sets):
                if clean_hash not in hashes:
                    continue
                manifest_meta = exclusion_meta["manifests"][index]
                manifest_meta["matched_documents"] += 1
                matched_per_source = manifest_meta["matched_per_source"]
                matched_per_source[src_name] = matched_per_source.get(src_name, 0) + 1
            ids, content_count, boundary_count = encode_with_accounting(
                tok,
                cleaned,
                add_bos=add_bos,
                add_eos=add_eos,
                bos_id=bos_id,
                eos_id=eos_id,
            )
            assert_token_ids_ok(
                ids,
                vocab_size=vocab_size,
                dtype=dtype,
                src=src_name,
                text_preview=cleaned[:200],
            )
            for target in (
                excluded_reference,
                per_src[src_name]["excluded_reference"],
            ):
                target["documents"] += 1
                target["cleaned_chars"] += len(cleaned)
                target["content_tokens"] += content_count
                target["boundary_tokens"] += boundary_count
                target["serialized_tokens"] += len(ids)
            continue

        destination = route_document(src_name, cleaned)
        if destination is None:
            continue

        ids, content_count, boundary_count = encode_with_accounting(
            tok,
            cleaned,
            add_bos=add_bos,
            add_eos=add_eos,
            bos_id=bos_id,
            eos_id=eos_id,
        )
        if not ids:
            per_src[src_name]["dropped_empty"] += 1
            dropped_empty += 1
            continue
        assert_token_ids_ok(
            ids,
            vocab_size=vocab_size,
            dtype=dtype,
            src=src_name,
            text_preview=cleaned[:200],
        )
        add_doc(
            src_name,
            destination,
            ids,
            content_tokens=content_count,
            boundary_tokens=boundary_count,
        )

    try:
        current_tokenizer_fingerprint = file_fingerprint(tokenizer_file)
    except (OSError, RuntimeError) as exc:
        message = f"tokenizer unavailable during post-scan verification: {exc}"
        _write_json_atomic(out_dir / "meta.failed.json", failure_manifest(message))
        raise RuntimeError(message) from exc
    if current_tokenizer_fingerprint != tokenizer_fingerprint:
        message = "tokenizer changed during shard construction"
        _write_json_atomic(out_dir / "meta.failed.json", failure_manifest(message))
        raise RuntimeError(message)
    for st in states:
        try:
            current_fingerprint = file_fingerprint(st.path)
        except (OSError, RuntimeError) as exc:
            message = f"source unavailable during post-scan verification: {st.path}: {exc}"
            _write_json_atomic(out_dir / "meta.failed.json", failure_manifest(message))
            raise RuntimeError(message) from exc
        if current_fingerprint != source_fingerprints[st.name]:
            message = f"source changed between the pre-scan and post-scan snapshots: {st.path}"
            _write_json_atomic(out_dir / "meta.failed.json", failure_manifest(message))
            raise RuntimeError(message)
    if buf_train:
        path = out_train / f"shard_{shard_idx_train:05d}.bin"
        record = flush(buf_train, path)
        shard_files_train.append(record)
        total_train += int(record["token_count"])
        shard_idx_train += 1
    if buf_val:
        path = out_val / f"shard_{shard_idx_val:05d}.bin"
        record = flush(buf_val, path)
        shard_files_val.append(record)
        total_val += int(record["token_count"])
        shard_idx_val += 1
    for st in states:
        flush_val_src(st.name, force=True)

    accounting["train"]["emitted_shard_tokens"] = int(total_train)
    accounting["val"]["emitted_shard_tokens"] = int(total_val)
    for split, emitted in (("train", total_train), ("val", total_val)):
        expected = accounting[split]["serialized_tokens"] + accounting[split]["separator_tokens"]
        if emitted != expected:
            raise AssertionError(
                f"{split} accounting mismatch: emitted={emitted}, expected={expected}"
            )

    for st in states:
        source_meta = per_src[st.name]
        for split in ("train", "val"):
            realized = source_meta["realized"][split]
            realized["emitted_shard_tokens"] = (
                realized["serialized_tokens"] + realized["separator_tokens"]
            )
        source_meta["train_tokens"] = source_meta["realized"]["train"]["serialized_tokens"]
        source_meta["val_tokens"] = source_meta["realized"]["val"]["serialized_tokens"]
        source_meta["train_content_tokens"] = source_meta["realized"]["train"]["content_tokens"]
        source_meta["val_content_tokens"] = source_meta["realized"]["val"]["content_tokens"]
        source_meta["target_vs_realized"] = {
            "train_serialized_overshoot": (
                source_meta["realized"]["train"]["serialized_tokens"]
                - source_meta["targets"]["train_serialized_tokens"]
            ),
            "val_serialized_overshoot": (
                source_meta["realized"]["val"]["serialized_tokens"]
                - source_meta["targets"]["val_serialized_tokens"]
            ),
        }

    first_tok_meta = {
        "topk_all": topk_counter(first_tok_all, first_token_topk),
        "topk_per_source": {
            key: topk_counter(value, first_token_topk) for key, value in first_tok_per_src.items()
        },
    }
    meta = {
        "schema_version": 3,
        "status": "complete",
        "tokenizer_path": tokenizer_path,
        "tokenizer_sha256": tokenizer_sha256,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "dtype": "uint32" if dtype == np.uint32 else "uint16",
        "vocab_size": vocab_size,
        "contract": contract,
        "legacy_flags": {
            "allow_noncanonical_contract": bool(legacy_allow_noncanonical_contract),
            "replay_on_exhaustion": bool(legacy_replay_on_exhaustion),
        },
        "source_exhaustion_policy": (
            "legacy_replay" if legacy_replay_on_exhaustion else "fail_fast"
        ),
        "validation_holdout": {
            "algorithm": _VALIDATION_HASH_ALGORITHM,
            "seed": int(validation_hash_seed),
            "membership_basis": "cleaned document text",
        },
        "reference_validation_exclusion": {
            **exclusion_meta,
            "matched": excluded_reference,
        },
        "sources": [{"path": str(path), "weight": float(weight)} for path, weight in sources],
        "shard_tokens": int(shard_tokens),
        "val_shard_tokens": int(val_shard_tokens),
        "val_ratio": float(val_ratio),
        "min_val_tokens_per_source": int(min_val_tokens_per_source),
        "seed": int(seed),
        "add_bos": bool(add_bos),
        "add_eos": bool(add_eos),
        "bos_id": int(bos_id),
        "eos_id": int(eos_id),
        "doc_sep": doc_sep,
        "filters": active_cleaning_contract,
        "seen_docs": int(seen_docs),
        "kept_docs": int(kept_docs),
        "dropped_short": int(dropped_short),
        "dropped_ascii": int(dropped_ascii),
        "dropped_empty": int(dropped_empty),
        "targets": {
            "train_serialized_tokens": int(target_train_tokens),
            "val_serialized_tokens": int(sum(val_target_per_src.values())),
            "train_per_source": train_target_per_src,
            "val_per_source": val_target_per_src,
        },
        "accounting": accounting,
        "train_tokens": int(total_train),
        "val_tokens": int(total_val),
        "train_shards": int(shard_idx_train),
        "val_shards": int(shard_idx_val),
        "train_content_tokens": int(accounting["train"]["content_tokens"]),
        "val_content_tokens": int(accounting["val"]["content_tokens"]),
        "train_boundary_tokens": int(accounting["train"]["boundary_tokens"]),
        "val_boundary_tokens": int(accounting["val"]["boundary_tokens"]),
        "train_serialized_tokens": int(accounting["train"]["serialized_tokens"]),
        "val_serialized_tokens": int(accounting["val"]["serialized_tokens"]),
        "train_separator_tokens": int(accounting["train"]["separator_tokens"]),
        "val_separator_tokens": int(accounting["val"]["separator_tokens"]),
        "train_target_tokens": int(target_train_tokens),
        "val_target_tokens_default": int(default_total_val),
        "train_target_per_source": train_target_per_src,
        "val_target_per_source": val_target_per_src,
        "source_fingerprints": source_fingerprints,
        "source_line_precheck": source_line_precheck,
        "first_token_topk": first_tok_meta,
        "per_source": per_src,
        "val_by_source": {
            src_dirname[st.name]: {
                "source": st.name,
                "dir": str(Path("val_by_source") / src_dirname[st.name]),
                "tokens": int(total_val_src[st.name]),
                "shards": int(shard_idx_val_src[st.name]),
            }
            for st in states
        },
        "shard_files": {
            "hash_algorithm": "sha256",
            "train": shard_files_train,
            "val": shard_files_val,
            "val_by_source": shard_files_val_src,
        },
    }

    _write_json_atomic(out_dir / "meta.json", meta)
    print("Done.")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


def write_shards(*, out_dir: Path, **build_kwargs: Any) -> dict[str, Any]:
    """Build in a sibling staging directory and atomically publish on success.

    The final output path is never created until every shard and ``meta.json``
    is complete. A failed build leaves only a sibling ``*.failed-*.json`` audit
    record, never a directory that a trainer could mistake for production data.
    """
    final_out_dir = Path(out_dir)
    if os.path.lexists(final_out_dir):
        raise FileExistsError(
            f"refusing to mix with an existing shard output path: {final_out_dir}"
        )

    final_out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{final_out_dir.name}.building-",
            dir=str(final_out_dir.parent),
        )
    )
    try:
        meta = _write_shards_to_directory(
            out_dir=staging_dir,
            **build_kwargs,
        )
        if not (staging_dir / "meta.json").is_file():
            raise AssertionError("completed staging build has no meta.json")
        if os.path.lexists(final_out_dir):
            raise FileExistsError(
                f"output path appeared during build; refusing publish: {final_out_dir}"
            )
        os.rename(staging_dir, final_out_dir)
        return meta
    except BaseException as exc:
        staged_failure = staging_dir / "meta.failed.json"
        failure: dict[str, Any]
        try:
            with open(staged_failure, encoding="utf-8") as f:
                loaded = json.load(f)
            failure = loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError):
            failure = {}
        failure.update({
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "requested_out_dir": str(final_out_dir),
            "publication": "sibling_staging_then_atomic_rename",
            "production_directory_published": False,
        })
        failure_path = final_out_dir.with_name(
            f"{final_out_dir.name}.failed-{uuid.uuid4().hex}.json"
        )
        try:
            _write_json_atomic(failure_path, failure)
        except OSError:
            # Preserve the original build exception; the final production path
            # still remains absent even if the auxiliary audit write fails.
            pass
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", default=[], help="path:weight (repeatable)")
    ap.add_argument(
        "--target_train_tokens",
        type=int,
        required=True,
        help="Target serialized non-validation document tokens (content + BOS/EOS).",
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tokenizer_path", required=True)

    ap.add_argument("--shard_tokens", type=int, default=10_000_000)
    ap.add_argument("--val_shard_tokens", type=int, default=2_000_000)
    ap.add_argument("--val_ratio", type=float, default=0.002)
    ap.add_argument("--min_val_tokens_per_source", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--validation_hash_seed",
        type=int,
        default=1234,
        help="Independent stable-holdout seed; changing source order does not change membership.",
    )

    ap.add_argument(
        "--add_bos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Canonical default ON. Disabling requires --legacy_allow_noncanonical_contract.",
    )
    ap.add_argument(
        "--add_eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Canonical default ON. Disabling requires --legacy_allow_noncanonical_contract.",
    )
    ap.add_argument("--bos_id", type=int, default=BOS_ID)
    ap.add_argument("--eos_id", type=int, default=EOS_ID)
    ap.add_argument(
        "--doc_sep",
        type=str,
        default="",
        help="Canonical default is empty. Non-empty separators require the legacy contract flag.",
    )
    ap.add_argument(
        "--legacy_allow_noncanonical_contract",
        action="store_true",
        help="LEGACY/DEBUG ONLY: allow noncanonical IDs, boundaries, tokenizer, or doc_sep; recorded in meta.json.",
    )
    ap.add_argument(
        "--legacy_replay_on_exhaustion",
        action="store_true",
        help="LEGACY/DEBUG ONLY: replay a finite source after exhaustion; recorded in meta.json.",
    )
    ap.add_argument(
        "--exclude_hash_manifest",
        type=Path,
        action="append",
        default=[],
        help=(
            "Repeatable canonical exclusion_hash_manifest.json. Matching cleaned "
            "documents are barred from train and ordinary validation shards."
        ),
    )

    ap.add_argument("--first_token_topk", type=int, default=50)
    ap.add_argument("--precheck_max_lines", type=int, default=300_000)

    ap.add_argument("--strip_leading_noise", action="store_true")
    ap.add_argument("--normalize_quotes", action="store_true")
    ap.add_argument(
        "--underscores_policy",
        type=str,
        default="keep",
        choices=["keep", "space", "remove"],
    )
    ap.add_argument("--min_chars", type=int, default=0)
    ap.add_argument("--min_ascii_ratio", type=float, default=0.0)

    args = ap.parse_args()
    sources = parse_sources(args.source)
    if not sources:
        raise SystemExit("Provide at least one --source path:weight")
    if args.val_shard_tokens <= 0 or args.shard_tokens <= 0:
        raise SystemExit("shard sizes must be > 0")
    if not (0.0 <= args.val_ratio <= 1.0):
        raise SystemExit("--val_ratio must be in [0,1]")
    if args.target_train_tokens <= 0:
        raise SystemExit("--target_train_tokens must be > 0")
    if args.min_val_tokens_per_source < 0:
        raise SystemExit("--min_val_tokens_per_source must be >= 0")
    if not (0.0 <= args.min_ascii_ratio <= 1.0):
        raise SystemExit("--min_ascii_ratio must be in [0,1]")

    write_shards(
        sources=sources,
        out_dir=Path(args.out_dir),
        tokenizer_path=args.tokenizer_path,
        shard_tokens=args.shard_tokens,
        val_shard_tokens=args.val_shard_tokens,
        val_ratio=args.val_ratio,
        min_val_tokens_per_source=args.min_val_tokens_per_source,
        seed=args.seed,
        validation_hash_seed=args.validation_hash_seed,
        add_bos=args.add_bos,
        add_eos=args.add_eos,
        bos_id=args.bos_id,
        eos_id=args.eos_id,
        target_train_tokens=args.target_train_tokens,
        precheck_max_lines=args.precheck_max_lines,
        doc_sep=args.doc_sep,
        first_token_topk=args.first_token_topk,
        strip_leading_noise=args.strip_leading_noise,
        normalize_quotes=args.normalize_quotes,
        underscores_policy=args.underscores_policy,
        min_chars=args.min_chars,
        min_ascii_ratio=args.min_ascii_ratio,
        legacy_allow_noncanonical_contract=args.legacy_allow_noncanonical_contract,
        legacy_replay_on_exhaustion=args.legacy_replay_on_exhaustion,
        exclude_hash_manifests=args.exclude_hash_manifest,
    )


if __name__ == "__main__":
    main()
