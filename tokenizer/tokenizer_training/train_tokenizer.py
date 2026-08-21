#!/usr/bin/env python3

"""
Train a Byte-Level BPE tokenizer for LLM pretraining/SFT.

Design goals
------------
1) Preserve whitespace AND unicode exactly (no normalizer, do NOT strip text
   fields), so decode(encode(x)) == x. Mainstream code tokenizers (GPT-4,
   Llama) do not normalize; NFKC silently rewrote full-width/compatibility
   characters, which corrupts string literals in code.
2) Avoid forced prefix-space behavior by default (add_prefix_space=False).
3) Keep BOS/EOS insertion out of the tokenizer (no post_processor by default).
   Add special tokens by ID exactly once in the data pipeline:
   - pretrain/build_pretrain_shards.py wraps each document as [BOS] doc [EOS];
     documents are separated by those special tokens ONLY (never "\\n\\n").
   - SFT/distill/DPO/GRPO use the token-level chat template in
     src/chat_template.py: [BOS] <|system|> ... <|user|> ... <|assistant|> ... [EOS]
     with a supervised EOS after every assistant answer.
4) Special tokens are single-source-of-truth in src/special_tokens.py:
   [PAD]=0 [UNK]=1 [BOS]=2 [EOS]=3 <|system|>=4 <|user|>=5 <|assistant|>=6.

Supported input JSONL formats
-----------------------------
Each line is a JSON object. This script can extract text from:
- Plain fields: {"text": "..."} or {"prompt": "...", "response": "..."}
- Chat messages: {"messages": [{"role":"user","content":"..."}, ...]}

Usage example
-------------
python train_tokenizer.py \
  --data datasets/tokenization/fineweb_sample.jsonl datasets/tokenization/tinystories_train.jsonl \
  --fields text \
  --vocab_size 32000 \
  --out_dir tokenizer \
  --min_freq 2
"""

from __future__ import annotations

import argparse
import atexit
from collections.abc import Iterator
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
from typing import Any

from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pretrain.build_pretrain_shards import (  # noqa: E402
    _write_json_atomic,
    clean_text,
    cleaned_text_sha256,
    load_exclusion_hash_manifest,
)
from src.special_tokens import (  # noqa: E402
    CANONICAL_VOCAB_SIZE,
    SPECIAL_TOKEN_IDS,
    SPECIAL_TOKENS,
    assert_tokenizer_contract,
)

PROJECT_ROOT = Path(ROOT)

# The canonical tokenizer-training corpus is the frozen Stage-F release. Its manifest schema is
# pinned so a differently shaped corpus manifest can never be accepted as canonical provenance.
CORPUS_MANIFEST_SCHEMA = "petitgpt-f-tokenizer-corpus-v1"

# Difficult round-trip fixtures. Byte-level BPE must reproduce every one of these exactly; they
# are the cheap bounded half of the OD-G3 validation contract.
ROUNDTRIP_FIXTURES: tuple[str, ...] = (
    "\n",
    "\n\n",
    "\r\n",
    "\r",
    " ",
    "  ",
    "\t",
    "\t\t",
    " leading",
    "trailing ",
    "  both  ",
    "a\nb",
    "\nHello",
    "Hello\n",
    "line1\r\nline2\r\n",
    "under_score_name",
    "__dunder__",
    "```python\nprint('hi')\n```",
    '{\n  "a": 1,\n  "b": [2, 3]\n}\n',
    "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
    "if x:\n\tif y:\n\t\treturn 1\n",
    "\uff46\uff55\uff4c\uff4c\uff0d\uff57\uff49\uff44\uff54\uff48\uff1a\uff11\uff12\uff13 and \u2460\u2461\u2462 caf\u00e9 \ufb01le",
    "\u4e2d\u6587\u6d4b\u8bd5 \u30c6\u30b9\u30c8 \ud55c\uad6d\uc5b4 \u0627\u0644\u0639\u0631\u0628\u064a\u0629 \u05e2\u05d1\u05e8\u05d9\u05ea",
    "emoji \U0001f680\U0001f525\u2713 ZWJ \U0001f468\u200d\U0001f469\u200d\U0001f467\u200d\U0001f466 flag \U0001f1ef\U0001f1f5",
    "combining: e\u0301 a\u0300 n\u0303  NFC/NFD: \u00e9 vs e\u0301",
    "math \u2211\u222b\u221a\u2260\u2264\u2265 arrows \u2192\u2190\u2194 punct \u2014\u2013\u2026\u00ab\u00bb\u201c\u201d\u2018\u2019",
    "tab\tsep, vt\x0b, ff\x0c",
    "astral: \U0001d518\U0001d52b\U0001d526\U0001d520\U0001d52c\U0001d521\U0001d522",
    "[EOS] <|assistant|> [BOS] literal control strings",
    "x",
    "\u00a0nbsp\u00a0",
    "\u200bzwsp\u200b",
)


def _json_loads(line: str) -> Any | None:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def _string_or_none(x: Any) -> str | None:
    return x if isinstance(x, str) and x != "" else None


def _render_messages(messages: list[dict[str, Any]]) -> str | None:
    """
    Convert chat-style messages into a single training string for BPE.

    Role special tokens are NOT written here on purpose — they are added
    tokens, invisible to BPE merge learning; this is just corpus text.
    """
    parts: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = _string_or_none(m.get("role"))
        content = _string_or_none(m.get("content"))
        if role is None or content is None:
            continue
        # Use a simple, stable delimiter. Do NOT strip content.
        parts.append(f"{role}:\n{content}")
    if not parts:
        return None
    return "\n\n".join(parts)


def iter_texts(
    paths: list[str],
    fields: list[str],
    messages_key: str = "messages",
    allow_messages: bool = True,
    exclusion_hash_sets: list[frozenset[str]] | None = None,
    exclusion_cleaning: dict[str, Any] | None = None,
    exclusion_stats: dict[str, Any] | None = None,
    per_file_stats: list[dict[str, Any]] | None = None,
) -> Iterator[str]:
    """
    Stream text samples from one or more JSONL files.

    Important: we preserve whitespace and do NOT call strip() on text fields.
    """
    hash_sets = exclusion_hash_sets or []
    if hash_sets and exclusion_cleaning is None:
        raise ValueError("exclusion_cleaning is required with exclusion hash sets")
    stats = exclusion_stats if exclusion_stats is not None else {}
    stats.setdefault("considered_samples", 0)
    stats.setdefault("excluded_samples", 0)
    stats.setdefault("yielded_samples", 0)
    stats.setdefault("matched_per_manifest", [0 for _ in hash_sets])

    def should_exclude(text: str) -> bool:
        stats["considered_samples"] += 1
        if not hash_sets:
            stats["yielded_samples"] += 1
            return False
        assert exclusion_cleaning is not None
        required = {
            "strip_leading_noise",
            "normalize_quotes",
            "underscores_policy",
            "min_chars",
            "min_ascii_ratio",
        }
        if set(exclusion_cleaning) != required:
            raise ValueError(
                "exclusion manifest cleaning contract has unexpected keys: "
                f"{sorted(exclusion_cleaning)}"
            )
        cleaned = clean_text(
            text,
            strip_leading_noise=bool(exclusion_cleaning["strip_leading_noise"]),
            normalize_quotes=bool(exclusion_cleaning["normalize_quotes"]),
            underscores_policy=str(exclusion_cleaning["underscores_policy"]),
            min_chars=int(exclusion_cleaning["min_chars"]),
            min_ascii_ratio=float(exclusion_cleaning["min_ascii_ratio"]),
        )
        if cleaned is not None:
            value = cleaned_text_sha256(cleaned)
            matched = [index for index, hashes in enumerate(hash_sets) if value in hashes]
            if matched:
                stats["excluded_samples"] += 1
                for index in matched:
                    stats["matched_per_manifest"][index] += 1
                return True
        stats["yielded_samples"] += 1
        return False

    for p in paths:
        counters = {"path": str(p), "physical_lines": 0, "yielded_samples": 0, "utf8_bytes": 0}
        if per_file_stats is not None:
            per_file_stats.append(counters)
        with open(p, encoding="utf-8") as f:
            for raw_line in f:
                counters["physical_lines"] += 1
                # Only remove the trailing newline(s) from the JSONL file, not leading spaces.
                line = raw_line.rstrip("\r\n")
                if not line:
                    continue

                obj = _json_loads(line)
                if obj is None or not isinstance(obj, dict):
                    continue

                # 1) Chat-style: {"messages":[...]}
                if allow_messages and messages_key in obj and isinstance(obj[messages_key], list):
                    rendered = _render_messages(obj[messages_key])
                    if rendered is not None:
                        if not should_exclude(rendered):
                            counters["yielded_samples"] += 1
                            counters["utf8_bytes"] += len(rendered.encode("utf-8"))
                            yield rendered
                        continue

                # 2) Plain fields: concatenate in the given order
                parts: list[str] = []
                for k in fields:
                    v = _string_or_none(obj.get(k))
                    if v is not None:
                        parts.append(v)

                if parts:
                    text = "\n".join(parts)
                    if not should_exclude(text):
                        counters["yielded_samples"] += 1
                        counters["utf8_bytes"] += len(text.encode("utf-8"))
                        yield text


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str | None:
    """Read HEAD without shelling out, mirroring the Stage-F builder."""
    head_path = PROJECT_ROOT / ".git/HEAD"
    if not head_path.is_file():
        return None
    value = head_path.read_text(encoding="ascii").strip()
    if value.startswith("ref: "):
        ref_path = PROJECT_ROOT / ".git" / value[5:]
        if not ref_path.is_file():
            return None
        value = ref_path.read_text(encoding="ascii").strip()
    return value if len(value) == 40 else None


def load_corpus_release_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Load and strictly validate the frozen canonical corpus release manifest.

    A path string is not provenance. This pins the manifest by its own SHA-256, checks the
    frozen schema, and returns the exact per-bucket output records that every training file
    must match. Anything unexpected raises instead of degrading to a weaker binding.
    """
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"corpus release manifest is not a regular file: {path}")
    actual = _sha256_path(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"corpus release manifest SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("corpus release manifest must be a JSON object")
    if manifest.get("schema_version") != CORPUS_MANIFEST_SCHEMA:
        raise ValueError(
            f"unexpected corpus manifest schema: {manifest.get('schema_version')!r}; "
            f"expected {CORPUS_MANIFEST_SCHEMA!r}"
        )
    if manifest.get("immutable_publication") is not True:
        raise ValueError("corpus release manifest is not marked immutable")
    for field in ("run_fingerprint_sha256",):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise ValueError(f"corpus release manifest is missing {field}")
    selection = manifest.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("corpus release manifest is missing its selection block")
    if not isinstance(selection.get("selected_occurrence_set_sha256"), str):
        raise ValueError("corpus release manifest is missing the selected-set fingerprint")
    buckets = selection.get("buckets")
    if not isinstance(buckets, list) or not buckets:
        raise ValueError("corpus release manifest declares no buckets")

    records: dict[str, dict[str, Any]] = {}
    for bucket in buckets:
        name = bucket.get("canonical_name")
        output = bucket.get("output") or {}
        member = output.get("path")
        if not isinstance(name, str) or not isinstance(member, str):
            raise ValueError("corpus release manifest bucket is missing a name or output path")
        resolved = (path.parent / member).resolve()
        if str(resolved) in records:
            raise ValueError(f"duplicate corpus output path in manifest: {resolved}")
        for field in ("sha256", "size_bytes", "rows"):
            if output.get(field) is None:
                raise ValueError(f"{name}: corpus manifest output is missing {field}")
        records[str(resolved)] = {
            "canonical_bucket": name,
            "path": str(resolved),
            "manifest_relative_path": member,
            "sha256": output["sha256"],
            "size_bytes": int(output["size_bytes"]),
            "expected_occurrences": int(output["rows"]),
        }
    return {
        "manifest_path": str(path),
        "manifest_sha256": actual,
        "schema_version": manifest["schema_version"],
        "run_fingerprint_sha256": manifest["run_fingerprint_sha256"],
        "selected_occurrence_set_sha256": selection["selected_occurrence_set_sha256"],
        "total_selected_documents": selection.get("total_selected_documents"),
        "total_realized_cleaned_utf8_bytes": selection.get("total_realized_cleaned_utf8_bytes"),
        "outputs": records,
    }


def verify_corpus_binding(binding: dict[str, Any], data_paths: list[str]) -> list[dict[str, Any]]:
    """Prove the training file set is exactly the frozen corpus, byte for byte.

    Every ``--data`` file must resolve to a manifest output, the set must match exactly with no
    missing, extra or duplicated member, and each file's size and SHA-256 must equal the frozen
    values. This runs before any training work begins.
    """
    outputs = binding["outputs"]
    resolved: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw in data_paths:
        candidate = Path(raw)
        if not candidate.is_file() or candidate.is_symlink():
            raise FileNotFoundError(f"training input is not a regular file: {raw}")
        key = str(candidate.resolve())
        if key in seen:
            raise ValueError(f"training input supplied twice: {raw}")
        seen.add(key)
        if key not in outputs:
            raise ValueError(f"training input is not a member of the bound corpus release: {raw}")
        resolved.append((key, raw))

    missing = sorted(set(outputs) - seen)
    if missing:
        raise ValueError(
            "training input set does not cover the bound corpus release; missing: "
            + ", ".join(outputs[item]["canonical_bucket"] for item in missing)
        )

    verified: list[dict[str, Any]] = []
    for key, raw in resolved:
        record = outputs[key]
        size = Path(key).stat().st_size
        if size != record["size_bytes"]:
            raise RuntimeError(
                f"{record['canonical_bucket']}: size mismatch: "
                f"expected {record['size_bytes']}, got {size}"
            )
        actual = _sha256_path(Path(key))
        if actual != record["sha256"]:
            raise RuntimeError(
                f"{record['canonical_bucket']}: SHA-256 mismatch: "
                f"expected {record['sha256']}, got {actual}"
            )
        verified.append({**record, "supplied_as": raw, "verified_sha256": actual})
    return verified


def environment_manifest(
    data_paths: list[str], trainer_path: Path, argv: list[str]
) -> dict[str, Any]:
    """Everything a later rebuild needs in order to be checkable rather than merely likely."""
    return {
        "schema_version": 1,
        "kind": "petitgpt_tokenizer_environment",
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "tokenizers_version": _tokenizers_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "trainer_path": str(trainer_path.relative_to(PROJECT_ROOT)),
        "trainer_sha256": _sha256_path(trainer_path),
        "git_head": _git_head(),
        "argv": list(argv),
        "input_file_order": list(data_paths),
        "parallelism_environment": {
            name: os.environ.get(name)
            for name in ("RAYON_NUM_THREADS", "TOKENIZERS_PARALLELISM", "OMP_NUM_THREADS")
        },
        "cpu_count_reported": os.cpu_count(),
    }


def _tokenizers_version() -> str:
    import tokenizers

    return str(getattr(tokenizers, "__version__", "unknown"))


def validate_tokenizer_release(
    tokenizer_path: Path,
    data_paths: list[str],
    corpus_records: list[dict[str, Any]] | None,
    *,
    expected_vocab_size: int,
    full_corpus: bool,
    exclusion_hash_sets: list[frozenset[str]],
    exclusion_cleaning: dict[str, Any] | None,
    fields: list[str],
    messages_key: str,
    allow_messages: bool,
) -> dict[str, Any]:
    """Validate the staged tokenizer before it is allowed to become an immutable release.

    Three layers, per the frozen acceptance contract: the structural contract and reload
    equivalence, bounded difficult fixtures, and — for canonical production — one streaming pass
    over every training occurrence proving ``decode(encode(text)) == text``.
    """
    tokenizer_json = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    saved = Tokenizer.from_file(str(tokenizer_path))
    # Reloading and re-serialising must reproduce the saved artifact exactly.
    reload_equivalent = json.loads(saved.to_str()) == tokenizer_json
    # Literal special-token strings must stay ordinary text; this is the pipeline-wide setting.
    saved.encode_special_tokens = True
    special_ids = {name: saved.token_to_id(name) for name in SPECIAL_TOKEN_IDS}
    special_id_values = set(SPECIAL_TOKEN_IDS.values())

    fixture_failures = [
        text for text in ROUNDTRIP_FIXTURES if saved.decode(saved.encode(text).ids) != text
    ]
    injection_probe = saved.encode("literal [EOS] and <|assistant|> and [BOS] in text").ids
    injection_leak = [item for item in injection_probe if item in special_id_values]

    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "petitgpt_tokenizer_validation",
        "tokenizer_sha256": _sha256_path(tokenizer_path),
        "contract": {
            "vocab_size": int(saved.get_vocab_size(with_added_tokens=True)),
            "expected_vocab_size": expected_vocab_size,
            "special_token_ids": special_ids,
            "special_token_ids_match": special_ids == SPECIAL_TOKEN_IDS,
            "normalizer": tokenizer_json.get("normalizer"),
            "post_processor": tokenizer_json.get("post_processor"),
            "pre_tokenizer": tokenizer_json.get("pre_tokenizer"),
            "decoder": tokenizer_json.get("decoder"),
            "bpe_unk_token": (tokenizer_json.get("model") or {}).get("unk_token"),
            "reload_equivalent": reload_equivalent,
        },
        "fixtures": {
            "count": len(ROUNDTRIP_FIXTURES),
            "failures": len(fixture_failures),
            "failed_repr": [repr(item) for item in fixture_failures[:10]],
            "injection_hardening_ok": not injection_leak,
        },
        "full_corpus_stream": None,
    }

    if full_corpus:
        by_path = {}
        if corpus_records:
            by_path = {record["path"]: record for record in corpus_records}
        per_file: list[dict[str, Any]] = []
        started = time.monotonic()
        totals = {
            "occurrences": 0,
            "utf8_bytes": 0,
            "tokens": 0,
            "roundtrip_failures": 0,
            "special_id_occurrences": 0,
            "unk_occurrences": 0,
        }
        for raw in data_paths:
            key = str(Path(raw).resolve())
            record = by_path.get(key, {})
            counters = {
                "path": raw,
                "canonical_bucket": record.get("canonical_bucket"),
                "sha256": record.get("sha256"),
                "expected_occurrences": record.get("expected_occurrences"),
                "occurrences": 0,
                "utf8_bytes": 0,
                "tokens": 0,
                "roundtrip_failures": 0,
                "special_id_occurrences": 0,
                "unk_occurrences": 0,
            }
            stream = iter_texts(
                paths=[raw],
                fields=fields,
                messages_key=messages_key,
                allow_messages=allow_messages,
                exclusion_hash_sets=exclusion_hash_sets,
                exclusion_cleaning=exclusion_cleaning,
                exclusion_stats={},
            )
            for text in stream:
                ids = saved.encode(text).ids
                counters["occurrences"] += 1
                counters["utf8_bytes"] += len(text.encode("utf-8"))
                counters["tokens"] += len(ids)
                if saved.decode(ids) != text:
                    counters["roundtrip_failures"] += 1
                for token_id in ids:
                    if token_id in special_id_values:
                        counters["special_id_occurrences"] += 1
                        if token_id == SPECIAL_TOKEN_IDS["[UNK]"]:
                            counters["unk_occurrences"] += 1
            expected = counters["expected_occurrences"]
            counters["occurrence_count_matches_manifest"] = (
                None if expected is None else counters["occurrences"] == expected
            )
            per_file.append(counters)
            for field in (
                "occurrences",
                "utf8_bytes",
                "tokens",
                "roundtrip_failures",
                "special_id_occurrences",
                "unk_occurrences",
            ):
                totals[field] += counters[field]
        report["full_corpus_stream"] = {
            "per_file": per_file,
            "totals": totals,
            "bytes_per_token": (
                totals["utf8_bytes"] / totals["tokens"] if totals["tokens"] else None
            ),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "status": (
                "PASS"
                if totals["roundtrip_failures"] == 0
                and totals["special_id_occurrences"] == 0
                and all(item["occurrence_count_matches_manifest"] is not False for item in per_file)
                else "FAIL"
            ),
        }

    contract = report["contract"]
    problems: list[str] = []
    if contract["vocab_size"] != expected_vocab_size:
        problems.append(f"vocab_size {contract['vocab_size']} != {expected_vocab_size}")
    if not contract["special_token_ids_match"]:
        problems.append("special token IDs do not match the frozen layout")
    if contract["normalizer"] is not None:
        problems.append("a normalizer is configured")
    if contract["post_processor"] is not None:
        problems.append("an automatic BOS/EOS post-processor is configured")
    if not contract["reload_equivalent"]:
        problems.append("the saved tokenizer does not reload equivalently")
    if fixture_failures:
        problems.append(f"{len(fixture_failures)} round-trip fixtures failed")
    if injection_leak:
        problems.append("literal special-token strings injected real ids")
    stream = report["full_corpus_stream"]
    if stream is not None and stream["status"] != "PASS":
        problems.append("full-corpus streaming validation failed")
    report["problems"] = problems
    report["status"] = "PASS" if not problems else "FAIL"
    return report


def make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def restore_writable(root: Path) -> None:
    """Undo read-only modes so a failed staging tree can still be cleaned up."""
    try:
        root.chmod(0o700)
    except OSError:
        pass
    for path in root.rglob("*"):
        try:
            path.chmod(0o700 if path.is_dir() else 0o600)
        except OSError:
            pass


def write_sha256sums(staging: Path, exclude: set[str]) -> None:
    lines = []
    for path in sorted(staging.rglob("*")):
        if path.is_file() and path.name not in exclude:
            lines.append(f"{_sha256_path(path)}  {path.relative_to(staging)}")
    target = staging / "SHA256SUMS"
    with open(target, "w", encoding="ascii") as handle:
        handle.write("\n".join(lines) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_tokenizer(add_prefix_space: bool) -> Tokenizer:
    """
    Build a Byte-Level BPE tokenizer.

    No normalizer: byte-level fallback already handles rare unicode, and any
    normalization (e.g. NFKC) breaks decode(encode(x)) == x for code.
    """
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=add_prefix_space)
    tok.decoder = ByteLevelDecoder()
    return tok


def maybe_add_post_processor(tok: Tokenizer, enabled: bool) -> None:
    """
    Optionally add BOS/EOS automatically.

    Recommended default: disabled. Add BOS/EOS exactly once in your data pipeline.
    """
    if not enabled:
        return
    bos_id = tok.token_to_id("[BOS]")
    eos_id = tok.token_to_id("[EOS]")
    assert bos_id is not None and eos_id is not None
    tok.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [EOS] $B:1 [EOS]:1",
        special_tokens=[("[BOS]", bos_id), ("[EOS]", eos_id)],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="Input JSONL files.")
    ap.add_argument(
        "--fields",
        nargs="+",
        default=["text"],
        help="JSON fields to concatenate when not using chat messages.",
    )
    ap.add_argument("--messages_key", type=str, default="messages", help="Chat messages key.")
    ap.add_argument("--no_messages", action="store_true", help="Disable chat messages parsing.")
    ap.add_argument("--vocab_size", type=int, default=32000, help="Target vocabulary size.")
    ap.add_argument("--min_freq", type=int, default=2, help="Minimum token frequency.")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory.")
    ap.add_argument(
        "--add_prefix_space",
        action="store_true",
        help="If set, use ByteLevel(add_prefix_space=True). Not recommended for strict round-trip.",
    )
    ap.add_argument(
        "--add_bos_eos_post_processor",
        action="store_true",
        help="If set, tokenizer will automatically add BOS/EOS via post_processor (usually avoid).",
    )
    ap.add_argument(
        "--strict_special_ids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Assert the src/special_tokens.py ID layout (default ON: all downstream "
        "scripts hardcode these IDs). --no-strict_special_ids to disable.",
    )
    ap.add_argument(
        "--model_max_length",
        type=int,
        default=2048,
        help="Write into tokenizer_config.json (informational).",
    )
    ap.add_argument(
        "--legacy_allow_noncanonical_contract",
        action="store_true",
        help=(
            "DEBUG/TEST ONLY: permit non-32k, prefix-space, post-processor, "
            "non-strict IDs, or missing reference exclusions. The manifest is "
            "marked noncanonical and production consumers reject it."
        ),
    )
    ap.add_argument(
        "--exclude_hash_manifest",
        type=Path,
        action="append",
        default=[],
        help=(
            "Repeatable pre-tokenizer reference-reserve exclusion manifest. "
            "Matching samples never reach BPE training."
        ),
    )
    ap.add_argument(
        "--corpus_release_manifest",
        type=Path,
        default=None,
        help=(
            "Frozen canonical tokenizer-training corpus release manifest. Required for a "
            "canonical release: every --data file is verified against it by SHA-256 before "
            "training, and its identity is bound into the published release."
        ),
    )
    ap.add_argument(
        "--corpus_release_manifest_sha256",
        type=str,
        default=None,
        help="Expected SHA-256 of --corpus_release_manifest. Required whenever it is supplied.",
    )
    ap.add_argument(
        "--full_corpus_validation",
        action="store_true",
        help=(
            "Stream every training occurrence through the trained tokenizer and require "
            "decode(encode(text)) == text. Required for a canonical release."
        ),
    )
    args = ap.parse_args()

    contract_issues: list[str] = []
    if int(args.vocab_size) != CANONICAL_VOCAB_SIZE:
        contract_issues.append(
            f"vocab_size={args.vocab_size}, expected {CANONICAL_VOCAB_SIZE}"
        )
    if bool(args.add_prefix_space):
        contract_issues.append("add_prefix_space=True")
    if bool(args.add_bos_eos_post_processor):
        contract_issues.append("automatic BOS/EOS post-processor enabled")
    if not bool(args.strict_special_ids):
        contract_issues.append("strict special-token IDs disabled")
    if not args.exclude_hash_manifest:
        contract_issues.append("reference-reserve exclusion manifest missing")
    if args.corpus_release_manifest is None:
        contract_issues.append("canonical corpus release manifest missing")
    if not bool(args.full_corpus_validation):
        contract_issues.append("full-corpus streaming validation disabled")
    if contract_issues and not args.legacy_allow_noncanonical_contract:
        raise SystemExit(
            "noncanonical tokenizer release refused: " + "; ".join(contract_issues)
        )
    if args.corpus_release_manifest is not None and not args.corpus_release_manifest_sha256:
        raise SystemExit(
            "--corpus_release_manifest requires --corpus_release_manifest_sha256; "
            "an unpinned manifest is not provenance"
        )

    final_out_dir = Path(args.out_dir)
    if os.path.lexists(final_out_dir):
        raise FileExistsError(
            f"refusing to replace existing tokenizer output path: {final_out_dir}"
        )
    final_out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{final_out_dir.name}.building-",
            dir=str(final_out_dir.parent),
        )
    )

    def cleanup_staging() -> None:
        if staging_dir.exists():
            restore_writable(staging_dir)
        shutil.rmtree(staging_dir, ignore_errors=True)

    atexit.register(cleanup_staging)

    # Staging must never outlive a failure. atexit covers the CLI, but an in-process
    # caller would otherwise leak a half-written release tree.
    try:
        exclusion_hash_sets: list[frozenset[str]] = []
        exclusion_manifests: list[dict[str, Any]] = []
        exclusion_cleaning: dict[str, Any] | None = None
        for path in args.exclude_hash_manifest:
            hashes, metadata = load_exclusion_hash_manifest(
                path,
                expected_cleaning=exclusion_cleaning,
            )
            if exclusion_cleaning is None:
                exclusion_cleaning = metadata["cleaning"]
            exclusion_hash_sets.append(hashes)
            exclusion_manifests.append(metadata)

        # Provenance is verified before training, not after: a mutated or unexpected corpus must
        # cost nothing but an immediate refusal.
        corpus_binding: dict[str, Any] | None = None
        corpus_records: list[dict[str, Any]] | None = None
        if args.corpus_release_manifest is not None:
            corpus_binding = load_corpus_release_manifest(
                args.corpus_release_manifest,
                str(args.corpus_release_manifest_sha256),
            )
            corpus_records = verify_corpus_binding(corpus_binding, list(args.data))
            print(
                f"[OK] bound {len(corpus_records)} training files to corpus release "
                f"{corpus_binding['manifest_sha256'][:16]}…",
                flush=True,
            )

        tok = build_tokenizer(add_prefix_space=args.add_prefix_space)

        trainer = BpeTrainer(
            vocab_size=args.vocab_size,
            min_frequency=args.min_freq,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )

        allow_messages = not args.no_messages
        exclusion_stats: dict[str, Any] = {}
        per_file_stats: list[dict[str, Any]] = []
        iterator = iter_texts(
            paths=args.data,
            fields=args.fields,
            messages_key=args.messages_key,
            allow_messages=allow_messages,
            exclusion_hash_sets=exclusion_hash_sets,
            exclusion_cleaning=exclusion_cleaning,
            exclusion_stats=exclusion_stats,
            per_file_stats=per_file_stats,
        )
        training_started = time.monotonic()
        tok.train_from_iterator(iterator, trainer=trainer)
        training_seconds = round(time.monotonic() - training_started, 3)

        # A bound corpus must deliver exactly the frozen occurrence count for every file; a silent
        # drop would otherwise be indistinguishable from a smaller corpus.
        if corpus_records is not None:
            by_path = {record["path"]: record for record in corpus_records}
            for counters in per_file_stats:
                record = by_path.get(str(Path(counters["path"]).resolve()))
                if record is None:
                    raise RuntimeError(f"unbound training file consumed: {counters['path']}")
                counters["canonical_bucket"] = record["canonical_bucket"]
                counters["expected_occurrences"] = record["expected_occurrences"]
                if counters["yielded_samples"] != record["expected_occurrences"]:
                    raise RuntimeError(
                        f"{record['canonical_bucket']}: consumed {counters['yielded_samples']} "
                        f"occurrences, corpus manifest declares {record['expected_occurrences']}"
                    )

        # --- Validate special tokens ---
        tok2id = {t: tok.token_to_id(t) for t in SPECIAL_TOKENS}
        for t, tid in tok2id.items():
            assert tid is not None, f"Missing special token in vocab: {t}"
        assert len(set(tok2id.values())) == len(tok2id), (
            f"Special token IDs are not unique: {tok2id}"
        )

        if args.strict_special_ids:
            for t, expected in SPECIAL_TOKEN_IDS.items():
                assert tok2id[t] == expected, f"Expected {t}={expected}, got {tok2id[t]}"
        else:
            print("Special token IDs:", tok2id)

        maybe_add_post_processor(tok, enabled=args.add_bos_eos_post_processor)

        # --- Save artifacts ---
        tokenizer_path = str(staging_dir / "tokenizer.json")
        tok.save(tokenizer_path)
        if not contract_issues:
            assert_tokenizer_contract(tokenizer_path)

        # vocab.json / merges.txt are an additional serialization of the same trained model.
        # model.save() does not mutate the tokenizer; tokenizer.json remains the canonical runtime
        # artifact and the identity every downstream stage binds.
        model_files = [Path(item).name for item in tok.model.save(str(staging_dir))]

        with open(staging_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "tokenizer_class": "PreTrainedTokenizerFast",
                    "model_max_length": args.model_max_length,
                    "padding_side": "right",
                    "truncation_side": "right",
                    "bos_token": "[BOS]",
                    "eos_token": "[EOS]",
                    "unk_token": "[UNK]",
                    "pad_token": "[PAD]",
                    "additional_special_tokens": ["<|system|>", "<|user|>", "<|assistant|>"],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        with open(staging_dir / "special_tokens_map.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "bos_token": "[BOS]",
                    "eos_token": "[EOS]",
                    "unk_token": "[UNK]",
                    "pad_token": "[PAD]",
                    "additional_special_tokens": ["<|system|>", "<|user|>", "<|assistant|>"],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        tokenizer_sha256 = hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest()
        for index, metadata in enumerate(exclusion_manifests):
            metadata["matched_samples"] = exclusion_stats["matched_per_manifest"][index]

        # Validate the staged tokenizer before it is allowed to become immutable.
        validation = validate_tokenizer_release(
            Path(tokenizer_path),
            list(args.data),
            corpus_records,
            expected_vocab_size=int(args.vocab_size),
            full_corpus=bool(args.full_corpus_validation),
            exclusion_hash_sets=exclusion_hash_sets,
            exclusion_cleaning=exclusion_cleaning,
            fields=list(args.fields),
            messages_key=args.messages_key,
            allow_messages=allow_messages,
        )
        _write_json_atomic(staging_dir / "validation.json", validation)
        if validation["status"] != "PASS":
            raise SystemExit(
                "tokenizer validation failed, refusing to publish: "
                + "; ".join(validation["problems"])
            )
        print(f"[OK] validation PASS ({validation['fixtures']['count']} fixtures)", flush=True)

        environment = environment_manifest(list(args.data), Path(__file__).resolve(), sys.argv)
        _write_json_atomic(staging_dir / "environment.json", environment)

        release_manifest = {
            "schema_version": 2,
            "kind": "petitgpt_tokenizer_release",
            "status": "complete",
            "contract": {
                "canonical": not contract_issues,
                "issues": contract_issues,
                "legacy_allow_noncanonical_contract": bool(args.legacy_allow_noncanonical_contract),
            },
            "publication": "sibling_staging_then_atomic_rename",
            "tokenizer_sha256": tokenizer_sha256,
            "vocab_size": tok.get_vocab_size(),
            "special_token_ids": tok2id,
            "training": {
                "data": list(args.data),
                "fields": list(args.fields),
                "messages_key": args.messages_key,
                "allow_messages": allow_messages,
                "vocab_size_target": args.vocab_size,
                "min_frequency": args.min_freq,
                "add_prefix_space": args.add_prefix_space,
                "post_processor_enabled": args.add_bos_eos_post_processor,
                "elapsed_seconds": training_seconds,
                "per_file": per_file_stats,
                "consumed_occurrences": sum(item["yielded_samples"] for item in per_file_stats),
                "consumed_utf8_bytes": sum(item["utf8_bytes"] for item in per_file_stats),
            },
            "corpus_binding": (
                None
                if corpus_binding is None
                else {
                    "verified_before_training": True,
                    "manifest_path": corpus_binding["manifest_path"],
                    "manifest_sha256": corpus_binding["manifest_sha256"],
                    "manifest_schema_version": corpus_binding["schema_version"],
                    "run_fingerprint_sha256": corpus_binding["run_fingerprint_sha256"],
                    "selected_occurrence_set_sha256": (
                        corpus_binding["selected_occurrence_set_sha256"]
                    ),
                    "total_selected_documents": corpus_binding["total_selected_documents"],
                    "total_realized_cleaned_utf8_bytes": (
                        corpus_binding["total_realized_cleaned_utf8_bytes"]
                    ),
                    "files": corpus_records,
                }
            ),
            "validation": {
                "status": validation["status"],
                "vocab_size": validation["contract"]["vocab_size"],
                "fixtures": validation["fixtures"],
                "full_corpus_stream": (
                    None
                    if validation["full_corpus_stream"] is None
                    else {
                        key: validation["full_corpus_stream"][key]
                        for key in ("totals", "status", "bytes_per_token", "elapsed_seconds")
                    }
                ),
                "report_path": "validation.json",
            },
            "environment": {
                "python_version": environment["python_version"],
                "tokenizers_version": environment["tokenizers_version"],
                "trainer_sha256": environment["trainer_sha256"],
                "git_head": environment["git_head"],
                "report_path": "environment.json",
            },
            "artifacts": {
                "canonical_runtime_artifact": "tokenizer.json",
                "model_serialization": model_files,
                "authoritative_manifest": "tokenizer_release_manifest.json",
                "byte_identical_manifest_copy": (
                    "manifest.json" if corpus_binding is not None else None
                ),
            },
            "reference_reserve_exclusion": {
                "enabled": bool(exclusion_hash_sets),
                "hash_algorithm": (
                    exclusion_manifests[0]["hash_algorithm"] if exclusion_manifests else None
                ),
                "cleaning": exclusion_cleaning,
                "manifest_count": len(exclusion_manifests),
                "union_hash_count": len(set().union(*exclusion_hash_sets))
                if exclusion_hash_sets
                else 0,
                "considered_samples": exclusion_stats["considered_samples"],
                "excluded_samples": exclusion_stats["excluded_samples"],
                "yielded_samples": exclusion_stats["yielded_samples"],
                "manifests": exclusion_manifests,
            },
        }
        # Checksums cover every payload file; the two manifest copies and SHA256SUMS itself are
        # hashed explicitly at closeout instead of self-referencing.
        write_sha256sums(
            staging_dir,
            exclude={"manifest.json", "tokenizer_release_manifest.json", "SHA256SUMS"},
        )

        # Manifest last. tokenizer_release_manifest.json is the historical authoritative name and
        # is always written. A provenance-bound release additionally gets a byte-identical
        # manifest.json under the standard immutable-release name, so both names resolve to one
        # manifest identity. An unbound legacy/development output keeps the historical layout and
        # must not emit a release-shaped manifest.json.
        _write_json_atomic(staging_dir / "tokenizer_release_manifest.json", release_manifest)
        if corpus_binding is not None:
            shutil.copyfile(
                staging_dir / "tokenizer_release_manifest.json", staging_dir / "manifest.json"
            )

        if not contract_issues:
            make_read_only(staging_dir)

        if os.path.lexists(final_out_dir):
            raise FileExistsError(
                f"output path appeared during tokenizer training: {final_out_dir}"
            )
        os.rename(staging_dir, final_out_dir)
        atexit.unregister(cleanup_staging)
    except BaseException:
        cleanup_staging()
        atexit.unregister(cleanup_staging)
        raise

    print("Saved tokenizer to:", final_out_dir)
    print("vocab_size =", tok.get_vocab_size())
    print("add_prefix_space =", args.add_prefix_space)
    print("post_processor(BOS/EOS) =", args.add_bos_eos_post_processor)


if __name__ == "__main__":
    main()
