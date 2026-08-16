#!/usr/bin/env python3

"""Minimal production Gate C candidate builder for the non-Python sources.

Gate C is the source-local stage between bounded Gate E inspection and the later
tokenizer/selection/shard stages.  It scans one exact pinned dataset revision
sequentially, applies source-local mechanical filters, and publishes an immutable
candidate release of raw accepted documents plus provenance.

Deliberately out of scope at this gate, and asserted as such in every manifest:
chat conversion, textual document separators, tokenizer counting, BOS/EOS
insertion, cross-source near-dedup, benchmark decontamination, and
reference-reserve exclusion.  Those belong to the tokenizer/selection/shard
stages, which own the canonical ``[BOS] document [EOS]`` framing.

The frozen bounded-inspection utility ``pretrain/corpus_gate_e.py`` is a
different tool with different semantics (stratified whole-sample inspection, no
resume); it is neither imported nor modified here.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TOOL_SCHEMA_VERSION = "petitgpt-corpus-gate-c-v1"
SPEC_VERSION = "nonpython-gate-c-source-spec-2026-08-16"

DATASET_SERVER_ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET_SERVER_PAGE_ROWS = 50
DATASET_SERVER_MAX_RESPONSE_BYTES = 32 * 1024 * 1024
DATASET_SERVER_TIMEOUT_SECONDS = 60.0
DATASET_SERVER_MAX_ATTEMPTS = 3

DEFAULT_MIN_BYTES = 200
DEFAULT_MAX_BYTES = 100 * 1024

MAX_ACCEPTED_DOCUMENTS = 4096
MAX_SCANNED_RECORDS = 200_000
MAX_RESPONSE_BYTES = 4 * 1024 * 1024 * 1024
MAX_WALL_SECONDS = 6 * 60 * 60
DEFAULT_CHECKPOINT_EVERY = 32

CHECKPOINT_NAME = "checkpoint.json"
DOCUMENTS_NAME = "documents.jsonl"
MANIFEST_NAME = "manifest.json"
CHECKSUMS_NAME = "MANIFEST.sha256"

_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_INLINE_LATEX_RE = re.compile(r"(?<!\$)\$(?!\$)[^$\n]{1,200}\$(?!\$)")
_NUMERIC_ENTITY_RE = re.compile(r"&#x?[0-9A-Fa-f]{1,6};")
_SUBSUP_TAG_RE = re.compile(r"</?su[bp]>", re.IGNORECASE)
_RM_CONSTRUCT_RE = re.compile(r"\$\{\s*\\(?:rm|it|bf|mathrm|mbox|text)\b")
_MOJIBAKE_RE = re.compile(r"(?:Ã[\x80-\xbf]|â€[\x9c\x9d\x99\x93\x94]|Â[\xa0-\xbf]|ï»¿)")
_ENUMERATED_LINE_RE = re.compile(r"^\s{0,3}(?:\d{1,2}[.)]|Step\s+\d{1,2}\b)", re.IGNORECASE)
_IMPERATIVE_LINE_RE = re.compile(
    r"^\s{0,3}(?:\d{1,2}[.)]|Step\s+\d{1,2}[:.)]?)\s+([A-Za-z][A-Za-z-]*)\b"
)
_HEADING_LINE_RE = re.compile(r"^\s{0,3}(?:#{1,6}\s+\S|[-*•]\s+\S|\d{1,2}[.)]\s+\S)")
_MATH_WORKSHEET_RE = re.compile(
    r"(?:answer key|show your work|worksheet|solve for [a-z]\b|problem set)", re.IGNORECASE
)
_TRANSFORMATION_FRAME_RE = re.compile(
    r"(?:^|\n)[ \t]*(?:here (?:is|'s) the (?:rewritten|revised|following)\b"
    r"|(?:the )?(?:rewritten|revised) (?:document|text|version)\b"
    r"|document[ \t]*:[ \t]*\S"
    r"|(?:sure|certainly)[,!]?[ \t]+here\b"
    r"|below is the (?:rewritten|revised)\b"
    r"|as (?:a|an) (?:step-by-step )?tutorial[ \t]*:)",
    re.IGNORECASE,
)
_TERMINAL_PUNCTUATION = (".", "!", "?", '."', ".'", '?"', '!"', ".)", ".]", "…")
_BIBLIOGRAPHIC_TITLE_RE = re.compile(
    r"(?:\bpp?\.\s*\d|\b\d{1,4}\s*pp\b|\bISBN\b|\(\s*(?:19|20)\d{2}\s*\)"
    r"|\b(?:19|20)\d{2},\s*\d{1,4}\s*(?:pages|pp)\b"
    r"|\b(?:edited by|reviewed by|book review)\b"
    r"|\b(?:Press|Publishers?|Verlag)\b[^.]{0,40}(?:19|20)\d{2})",
    re.IGNORECASE,
)
_STRUCTURED_ABSTRACT_RE = re.compile(
    r"\b(?:OBJECTIVES?|BACKGROUND|METHODS?|RESULTS?|CONCLUSIONS?|PURPOSE|DESIGN)\b\s*[:.]"
)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
_STOPWORDS = frozenset({
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "are",
    "was",
    "were",
    "have",
    "has",
    "been",
    "its",
    "into",
    "not",
    "but",
    "which",
    "their",
    "they",
    "his",
    "her",
    "our",
    "these",
    "those",
    "such",
    "than",
    "then",
    "there",
    "here",
    "when",
    "where",
    "what",
    "who",
    "how",
    "all",
    "any",
    "can",
    "will",
    "may",
    "one",
    "two",
    "new",
    "also",
    "more",
    "most",
    "some",
    "other",
    "over",
    "between",
    "during",
    "after",
    "before",
    "study",
    "studies",
    "using",
    "used",
})
_NON_IMPERATIVE_OPENERS = frozenset({
    "the",
    "a",
    "an",
    "this",
    "that",
    "these",
    "those",
    "it",
    "there",
    "many",
    "most",
    "some",
    "one",
    "two",
    "his",
    "her",
    "their",
    "its",
    "he",
    "she",
    "they",
    "we",
    "you",
})
_META_FORUM_RE = re.compile(r"(?:^|_)meta(?:_|$)")
# Derived from the complete 1,934,979-row / 361-forum census (2026-08-16), not from a guessed
# language-prefix pattern.  An open-ended prefix rule wrongly excluded cs_stackexchange_com
# (Computer Science, 3,460 rows) and vi_stackexchange_com (Vi/Vim, 2,119 rows), whose prefixes are
# topic abbreviations rather than language codes.  These are the localized Stack Overflow hosts
# that actually exist in the pinned population; their meta variants are caught by _META_FORUM_RE.
_LANGUAGE_LOCALIZED_FORUMS = frozenset({
    "es_stackoverflow_com",
    "ja_stackoverflow_com",
    "pt_stackoverflow_com",
    "ru_stackoverflow_com",
})
_CODE_INDENT_RE = re.compile(r"^(?: {4}|\t)")

COSMOPEDIA_PRIMARY_FORMATS = frozenset({
    "wikihow",
    "textbook",
    "textbook_unconditionned_topic",
})
COSMOPEDIA_STORY_FORMATS = frozenset({
    "textbook_narrative",
    "scientific_article",
    "story",
    "story_children",
    "story_life_lessons",
    "story_morality",
})
COSMOPEDIA_EXCLUDED_FORMATS = frozenset({"dialogue", "story_reddit", "story_forums"})
COSMOPEDIA_EXCLUDED_AUDIENCES = frozenset({"alien", "requires_details"})

FINEPHRASE_MIN_BYTES = 600
STACKEXCHANGE_MIN_QUESTION_SCORE = 3
STACKEXCHANGE_MIN_ANSWER_SCORE = 5
STACKEXCHANGE_INLINE_LATEX_LIMIT = 8
STACKEXCHANGE_CODE_DOMINANT_BYTES = 3 * 1024
STACKEXCHANGE_CODE_DOMINANT_RATIO = 0.60
DCLM_MIN_EDU_INT_SCORE = 3


class GateCError(RuntimeError):
    """A fail-closed Gate C contract error."""


class PlannedInterruption(RuntimeError):
    """Raised by tests to interrupt a build at an exact point."""


@dataclass(frozen=True)
class SourceSpec:
    """An exact, frozen, source-local Gate C binding.

    ``required_schema`` freezes the complete set of live leaf paths.  A ``None`` dtype means the
    path must exist but its dtype is not pinned, which is used only for list leaves whose inner
    dtype the Gate E evidence did not resolve.  Every load-bearing leaf carries an exact dtype.
    """

    key: str
    dataset: str
    dataset_config: str
    split: str
    revision: str
    body_path: str
    required_schema: tuple[tuple[str, str | None], ...]
    metadata_paths: tuple[str, ...]
    license: str
    natural_id_path: str | None = None
    min_bytes: int = DEFAULT_MIN_BYTES
    max_bytes: int = DEFAULT_MAX_BYTES

    @property
    def required_schema_map(self) -> dict[str, str | None]:
        return dict(self.required_schema)


def _schema(*items: tuple[str, str | None]) -> tuple[tuple[str, str | None], ...]:
    return tuple(sorted(items))


SOURCES: dict[str, SourceSpec] = {
    "fineweb_edu_dedup": SourceSpec(
        key="fineweb_edu_dedup",
        dataset="HuggingFaceTB/smollm-corpus",
        dataset_config="fineweb-edu-dedup",
        split="train",
        revision="3ba9d605774198c5868892d7a8deda78031a781f",
        body_path="text",
        natural_id_path="id",
        required_schema=_schema(
            ("id", "string"),
            ("metadata.date", "timestamp[s]"),
            ("metadata.dump", "string"),
            ("metadata.file_path", "string"),
            ("metadata.int_score", "int64"),
            ("metadata.language", "string"),
            ("metadata.language_score", "float64"),
            ("metadata.score", "float64"),
            ("metadata.token_count", "int64"),
            ("metadata.url", "string"),
            ("text", "string"),
        ),
        metadata_paths=(
            "id",
            "metadata.score",
            "metadata.int_score",
            "metadata.url",
            "metadata.dump",
            "metadata.language",
            "metadata.language_score",
            "metadata.token_count",
            "metadata.date",
        ),
        license="odc-by-1.0",
    ),
    "dclm_edu": SourceSpec(
        key="dclm_edu",
        dataset="HuggingFaceTB/dclm-edu",
        dataset_config="default",
        split="train",
        revision="dbad8ad71224482740cd9c9d353591adbf62fe04",
        body_path="text",
        natural_id_path="id",
        required_schema=_schema(
            ("edu_int_score", "int64"),
            ("edu_score", "float64"),
            ("fasttext_score", "float64"),
            ("id", "string"),
            ("language", "string"),
            ("language_score", "float64"),
            ("text", "string"),
            ("url", "string"),
        ),
        metadata_paths=(
            "id",
            "edu_score",
            "edu_int_score",
            "fasttext_score",
            "url",
            "language",
            "language_score",
        ),
        license="cc-by-4.0",
    ),
    "finewiki_en": SourceSpec(
        key="finewiki_en",
        dataset="HuggingFaceFW/finewiki",
        dataset_config="en",
        split="train",
        revision="8bd13e72e6a002407649b3e898535f42ceb1aeb9",
        body_path="text",
        natural_id_path="id",
        required_schema=_schema(
            ("bytes_html", "int64"),
            ("date_modified", "string"),
            ("has_math", "bool"),
            ("id", "string"),
            ("in_language", "string"),
            ("infoboxes", "string"),
            ("page_id", "int64"),
            ("text", "string"),
            ("title", "string"),
            ("url", "string"),
            ("version", "int64"),
            ("wikidata_id", "string"),
            ("wikiname", "string"),
            ("wikitext", "string"),
        ),
        metadata_paths=(
            "id",
            "title",
            "page_id",
            "url",
            "version",
            "wikidata_id",
            "wikiname",
            "in_language",
            "date_modified",
            "has_math",
            "bytes_html",
        ),
        license="cc-by-sa-4.0",
    ),
    "pes2o": SourceSpec(
        key="pes2o",
        dataset="allenai/dolmino-mix-1124",
        dataset_config="pes2o",
        split="train",
        revision="a319f19eef1e257417b11ea8c30da266ae175557",
        body_path="text",
        natural_id_path="id",
        required_schema=_schema(
            ("added", "string"),
            ("created", "string"),
            ("id", "string"),
            ("metadata.abstract", "string"),
            ("metadata.abstract_count", "int64"),
            ("metadata.abstract_language", "string"),
            ("metadata.abstract_perplexity", "float64"),
            ("metadata.extfieldsofstudy", "list[string]"),
            ("metadata.provenance", "string"),
            ("metadata.s2fieldsofstudy", "list[string]"),
            ("metadata.sha1", "string"),
            ("metadata.sources", "list[string]"),
            ("metadata.title", "string"),
            ("metadata.title_count", "int64"),
            ("metadata.title_language", "string"),
            ("metadata.title_perplexity", "float64"),
            # ``top_frequencies`` is a list of {token, count} structs, so it flattens to two leaves.
            ("metadata.top_frequencies.count", "list[int64]"),
            ("metadata.top_frequencies.token", "list[string]"),
            ("metadata.year", "int64"),
            ("source", "string"),
            ("text", "string"),
            ("version", "string"),
        ),
        metadata_paths=(
            "id",
            "source",
            "version",
            "added",
            "created",
            "metadata.title",
            "metadata.year",
            "metadata.provenance",
            "metadata.sources",
            "metadata.s2fieldsofstudy",
            "metadata.extfieldsofstudy",
            "metadata.abstract_language",
        ),
        license="odc-by-1.0",
    ),
    "stackexchange": SourceSpec(
        key="stackexchange",
        dataset="allenai/dolmino-mix-1124",
        dataset_config="stackexchange",
        split="train",
        revision="a319f19eef1e257417b11ea8c30da266ae175557",
        body_path="text",
        natural_id_path="id",
        required_schema=_schema(
            ("added", "string"),
            ("attributes.dedupe_para_ngrams_13_1", None),
            ("created", "string"),
            ("id", "string"),
            ("metadata.answer_comment_count", "int64"),
            ("metadata.answer_content_license", "string"),
            ("metadata.answer_id", "int64"),
            ("metadata.answer_last_activity_date", "string"),
            ("metadata.answer_last_edit_date", "string"),
            ("metadata.answer_last_editor_user_id", "int64"),
            ("metadata.answer_owner_user_id", "int64"),
            ("metadata.answer_score", "int64"),
            ("metadata.answer_view_count", "int64"),
            ("metadata.forum", "string"),
            ("metadata.provenance", "string"),
            ("metadata.question_comment_count", "int64"),
            ("metadata.question_content_license", "string"),
            ("metadata.question_id", "int64"),
            ("metadata.question_last_activity_date", "string"),
            ("metadata.question_last_edit_date", "string"),
            ("metadata.question_last_editor_user_id", "int64"),
            ("metadata.question_owner_user_id", "int64"),
            ("metadata.question_score", "int64"),
            ("metadata.question_view_count", "int64"),
            ("source", "string"),
            ("text", "string"),
            ("version", "string"),
        ),
        metadata_paths=(
            "id",
            "source",
            "version",
            "metadata.forum",
            "metadata.question_id",
            "metadata.answer_id",
            "metadata.question_score",
            "metadata.answer_score",
            "metadata.question_content_license",
            "metadata.answer_content_license",
            "metadata.provenance",
        ),
        license="cc-by-sa",
    ),
    "cosmopedia_v2": SourceSpec(
        key="cosmopedia_v2",
        dataset="HuggingFaceTB/smollm-corpus",
        dataset_config="cosmopedia-v2",
        split="train",
        revision="3ba9d605774198c5868892d7a8deda78031a781f",
        body_path="text",
        natural_id_path=None,
        required_schema=_schema(
            ("audience", "string"),
            ("format", "string"),
            ("prompt", "string"),
            ("seed_data", "string"),
            ("text", "string"),
            ("token_length", "int64"),
        ),
        metadata_paths=("format", "audience", "seed_data", "token_length"),
        license="odc-by-1.0",
    ),
    "finephrase_tutorial": SourceSpec(
        key="finephrase_tutorial",
        dataset="HuggingFaceFW/finephrase",
        dataset_config="tutorial",
        split="train",
        revision="78cf4a5ed0099214979c094c963e699c19163838",
        body_path="rollout_results.0.text",
        natural_id_path="id",
        required_schema=_schema(
            ("dataset", "string"),
            ("dump", "string"),
            ("file_path", "string"),
            ("id", "string"),
            ("int_score", "int64"),
            ("language", "string"),
            ("language_score", "float64"),
            ("rollout_results.finish_reason", "list[string]"),
            ("rollout_results.text", "list[string]"),
            ("rollout_results.usage.completion_tokens", "list[int64]"),
            ("rollout_results.usage.prompt_tokens", "list[int64]"),
            ("rollout_results.usage.prompt_tokens_details", None),
            ("rollout_results.usage.total_tokens", "list[int64]"),
            ("score", "float64"),
            ("text", "string"),
            ("token_count", "int64"),
            ("url", "string"),
        ),
        metadata_paths=(
            "id",
            "url",
            "dump",
            "dataset",
            "score",
            "int_score",
            "language",
            "language_score",
            "token_count",
        ),
        license="odc-by-1.0",
        min_bytes=FINEPHRASE_MIN_BYTES,
    ),
}

# Sources whose natural id is not a source-provided field.  Cosmopedia v2 exposes no id column, so
# its stable identity is derived from the split-local row index, which is deterministic under the
# pinned revision and the strictly ascending scan order.
_ROW_INDEX_IDENTITY = frozenset({"cosmopedia_v2"})


@dataclass(frozen=True)
class BuildConfig:
    """Bounded Gate C build parameters."""

    source: SourceSpec
    output_dir: Path
    work_dir: Path
    target_documents: int
    max_scanned: int
    max_response_bytes: int
    max_wall_seconds: float
    seed: int
    stop_after_documents: int | None = None
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY


@dataclass
class BuildCounters:
    """Scanned/accepted/rejected/byte accounting plus structured rejection counters."""

    scanned: int = 0
    accepted: int = 0
    rejected: int = 0
    accepted_text_bytes: int = 0
    response_bytes: int = 0
    request_count: int = 0
    rejections: Counter = field(default_factory=Counter)
    diagnostics: Counter = field(default_factory=Counter)

    def to_json(self) -> dict[str, Any]:
        return {
            "scanned": self.scanned,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "accepted_text_bytes": self.accepted_text_bytes,
            "response_bytes": self.response_bytes,
            "request_count": self.request_count,
            "rejections": dict(sorted(self.rejections.items())),
            "diagnostics": dict(sorted(self.diagnostics.items())),
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> BuildCounters:
        rejections = value.get("rejections") or {}
        diagnostics = value.get("diagnostics") or {}
        if not isinstance(rejections, Mapping) or not isinstance(diagnostics, Mapping):
            raise GateCError("checkpoint counter maps are invalid")
        return cls(
            scanned=_require_nonnegative_int(value.get("scanned"), "scanned"),
            accepted=_require_nonnegative_int(value.get("accepted"), "accepted"),
            rejected=_require_nonnegative_int(value.get("rejected"), "rejected"),
            accepted_text_bytes=_require_nonnegative_int(
                value.get("accepted_text_bytes"), "accepted_text_bytes"
            ),
            response_bytes=_require_nonnegative_int(value.get("response_bytes"), "response_bytes"),
            request_count=_require_nonnegative_int(value.get("request_count"), "request_count"),
            rejections=Counter(dict(rejections)),
            diagnostics=Counter(dict(diagnostics)),
        )


# --------------------------------------------------------------------------------------
# Small shared helpers
# --------------------------------------------------------------------------------------


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


# Correction E: physical JSONL framing. json.dumps(ensure_ascii=False) emits U+0085, U+2028 and
# U+2029 verbatim. They are legal inside a JSON string, but Python's str.splitlines() and many other
# line-oriented readers treat them as line terminators, so a record containing one is split in two
# by a naive parser. These three code points are escaped at the SERIALIZATION layer only: the parsed
# text is codepoint-identical to the original, and text_sha256/text_bytes stay based on the original
# UTF-8 text. No training body character is removed, replaced, or cleaned.
_JSONL_LINE_SEPARATOR_ESCAPES = (
    (b"\xc2\x85", b"\\u0085"),
    (b"\xe2\x80\xa8", b"\\u2028"),
    (b"\xe2\x80\xa9", b"\\u2029"),
)


def canonical_jsonl_record_bytes(record: Any) -> bytes:
    """Serialize one JSONL record, terminated by exactly one ASCII LF.

    The only physical record delimiter is ``\n``; every Unicode line separator inside a string is
    escaped so a bytes-based reader and a ``str.splitlines()`` reader agree on the record count.
    """
    payload = _canonical_json_bytes(record)
    for raw, escaped in _JSONL_LINE_SEPARATOR_ESCAPES:
        payload = payload.replace(raw, escaped)
    return payload + b"\n"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise GateCError(f"checkpoint field is invalid: {label}")
    return value


def _ensure_revision(value: str) -> str:
    if not isinstance(value, str) or not _REVISION_RE.fullmatch(value):
        raise GateCError("source revision must be an exact lowercase 40-hex commit")
    return value


def _ensure_under_workspace(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise GateCError("all Gate C paths must remain under the repository") from exc
    return resolved


def _ensure_git_ignored(path: Path) -> None:
    resolved = _ensure_under_workspace(path)
    relative = resolved.relative_to(PROJECT_ROOT)
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", "--", str(relative)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise GateCError(f"Gate C output/work path is not Git-ignored: {relative}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new_file(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _atomic_replace_bytes(path: Path, content: bytes) -> None:
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    _write_new_file(temporary, content)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _publish_directory(staging: Path, destination: Path) -> None:
    _fsync_directory(staging)
    if destination.exists():
        raise GateCError(f"refusing to overwrite published output: {destination}")
    os.rename(staging, destination)
    _fsync_directory(destination.parent)


def _staging_path(output_dir: Path) -> Path:
    return output_dir.parent / f".{output_dir.name}.partial"


def _get_path(row: Mapping[str, Any], path: str) -> Any:
    current: Any = row
    for part in path.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return None
            current = current[part]
        elif isinstance(current, (list, tuple)):
            if not part.isdigit():
                return None
            index = int(part)
            if index >= len(current):
                return None
            current = current[index]
        else:
            return None
    return current


# --------------------------------------------------------------------------------------
# Schema handling and bounded sequential transport
# --------------------------------------------------------------------------------------


def _server_feature_entries(feature: Any, prefix: str) -> dict[str, str]:
    """Flatten one Dataset Server feature into dotted leaf paths.

    A structured list flattens per child (``rollout_results.text -> list[string]``) so that a
    nested body field is an addressable path and can never collide with a same-named top-level
    column.  That collision is the exact FinePhrase hazard.
    """
    if not isinstance(feature, Mapping):
        raise GateCError(f"Dataset Server feature is not an object: {prefix}")
    dtype = feature.get("dtype")
    if isinstance(dtype, str):
        return {prefix: dtype}
    nested = feature.get("feature")
    if nested is not None:
        nested_entries = _server_feature_entries(nested, prefix)
        return {name: f"list[{kind}]" for name, kind in nested_entries.items()}
    ignored_keys = {"_type", "id", "length"}
    children = {str(key): value for key, value in feature.items() if key not in ignored_keys}
    if not children:
        raise GateCError(f"Dataset Server feature type is unsupported: {prefix}")
    flattened: dict[str, str] = {}
    for name, child in children.items():
        flattened.update(_server_feature_entries(child, f"{prefix}.{name}"))
    return flattened


def _server_schema(features: Any) -> dict[str, str]:
    if not isinstance(features, list) or not features:
        raise GateCError("Dataset Server response is missing features")
    flattened: dict[str, str] = {}
    for index, record in enumerate(features):
        if not isinstance(record, Mapping):
            raise GateCError("Dataset Server feature record is invalid")
        if record.get("feature_idx") != index:
            raise GateCError("Dataset Server feature indexes are not canonical")
        name = record.get("name")
        if not isinstance(name, str) or not name:
            raise GateCError("Dataset Server feature name is invalid")
        entries = _server_feature_entries(record.get("type"), name)
        if set(flattened).intersection(entries):
            raise GateCError("Dataset Server feature paths collide")
        flattened.update(entries)
    return dict(sorted(flattened.items()))


def assert_schema(source: SourceSpec, live_schema: Mapping[str, str]) -> None:
    """Fail closed on any live schema drift against the frozen binding.

    Every frozen leaf path must be present and no unfrozen leaf path may appear, so an added,
    removed or renamed column at any depth is a hard error.  Dtypes are compared exactly for every
    path whose frozen dtype is not ``None``.
    """
    required = source.required_schema_map
    observed = dict(live_schema)
    missing = sorted(set(required) - set(observed))
    added = sorted(set(observed) - set(required))
    changed = sorted(
        f"{name}:{required[name]}->{observed[name]}"
        for name in set(required) & set(observed)
        if required[name] is not None and required[name] != observed[name]
    )
    if missing or added or changed:
        raise GateCError(
            f"live schema drift for {source.key}: missing={missing} added={added} changed={changed}"
        )


def _dataset_server_request(
    source: SourceSpec,
    *,
    offset: int,
    length: int,
    opener: Callable[..., Any],
    sleeper: Callable[[float], None],
) -> tuple[dict[str, Any], int, int]:
    if type(offset) is not int or offset < 0:
        raise GateCError("Dataset Server offset is invalid")
    if type(length) is not int or not 1 <= length <= 100:
        raise GateCError("Dataset Server length exceeds the bounded API contract")
    query = urlencode({
        "dataset": source.dataset,
        "config": source.dataset_config,
        "split": source.split,
        "offset": offset,
        "length": length,
    })
    headers = {"User-Agent": "PetitGPT-Gate-C/1"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(f"{DATASET_SERVER_ROWS_URL}?{query}", headers=headers)
    last_error: BaseException | None = None
    for attempt in range(DATASET_SERVER_MAX_ATTEMPTS):
        try:
            with opener(request, timeout=DATASET_SERVER_TIMEOUT_SECONDS) as response:
                resolved_revision = response.headers.get("x-revision")
                if resolved_revision != source.revision:
                    raise GateCError("Dataset Server x-revision does not match the frozen revision")
                declared_length = response.headers.get("content-length")
                if declared_length is not None:
                    try:
                        declared_bytes = int(declared_length)
                    except ValueError as exc:
                        raise GateCError("Dataset Server Content-Length is invalid") from exc
                    if declared_bytes > DATASET_SERVER_MAX_RESPONSE_BYTES:
                        raise GateCError("Dataset Server response exceeds the per-call byte cap")
                raw = response.read(DATASET_SERVER_MAX_RESPONSE_BYTES + 1)
            if len(raw) > DATASET_SERVER_MAX_RESPONSE_BYTES:
                raise GateCError("Dataset Server response exceeds the per-call byte cap")
            try:
                payload = json.loads(raw.decode("utf-8", errors="strict"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise GateCError("Dataset Server response is not strict UTF-8 JSON") from exc
            if not isinstance(payload, dict):
                raise GateCError("Dataset Server response is not an object")
            return payload, len(raw), attempt + 1
        except GateCError:
            raise
        except HTTPError as exc:
            if exc.code < 500 and exc.code != 429:
                raise GateCError(f"Dataset Server returned HTTP {exc.code}") from exc
            last_error = exc
        except (URLError, TimeoutError, OSError) as exc:
            last_error = exc
        if attempt + 1 < DATASET_SERVER_MAX_ATTEMPTS:
            sleeper(float(2**attempt))
    raise GateCError("Dataset Server request failed after bounded retries") from last_error


def _dataset_server_page(
    payload: Mapping[str, Any], *, offset: int
) -> tuple[dict[str, str], int, list[tuple[int, Mapping[str, Any]]]]:
    schema = _server_schema(payload.get("features"))
    total_rows = payload.get("num_rows_total")
    rows = payload.get("rows")
    if type(total_rows) is not int or total_rows <= 0:
        raise GateCError("Dataset Server total row count is invalid")
    if not isinstance(rows, list):
        raise GateCError("Dataset Server rows payload is invalid")
    extracted: list[tuple[int, Mapping[str, Any]]] = []
    for position, record in enumerate(rows):
        if not isinstance(record, Mapping):
            raise GateCError("Dataset Server row record is invalid")
        row_index = record.get("row_idx")
        if type(row_index) is not int or row_index != offset + position:
            raise GateCError("Dataset Server row indexes drifted from the requested window")
        if record.get("truncated_cells") != []:
            raise GateCError("Dataset Server truncated at least one selected row")
        row = record.get("row")
        if not isinstance(row, Mapping):
            raise GateCError("Dataset Server returned a non-object row")
        extracted.append((row_index, row))
    return schema, total_rows, extracted


@dataclass(frozen=True)
class ScannedRow:
    """One sequentially scanned source row plus the transport cost attributed to it."""

    row_index: int
    row: Mapping[str, Any]
    response_bytes: int
    request_count: int
    live_schema: Mapping[str, str]
    total_rows: int


def dataset_server_scanner(
    source: SourceSpec,
    *,
    start_row: int,
    opener: Callable[..., Any] = urlopen,
    sleeper: Callable[[float], None] = time.sleep,
) -> Iterator[ScannedRow]:
    """Yield rows sequentially from ``start_row`` at the pinned revision.

    The scan is strictly ascending and page-bounded, so a checkpointed cursor resumes exactly where
    the previous invocation stopped.  Exhaustion ends the iterator; that is a normal stop, not an
    error.  Transport cost is attributed to the first row of each page so the caller's byte cap is
    charged once per request.
    """
    _ensure_revision(source.revision)
    offset = start_row
    schema: dict[str, str] | None = None
    total_rows: int | None = None
    while True:
        if total_rows is not None and offset >= total_rows:
            return
        length = DATASET_SERVER_PAGE_ROWS
        if total_rows is not None:
            length = min(length, total_rows - offset)
        payload, page_bytes, attempts = _dataset_server_request(
            source, offset=offset, length=length, opener=opener, sleeper=sleeper
        )
        page_schema, page_total, rows = _dataset_server_page(payload, offset=offset)
        if schema is None:
            assert_schema(source, page_schema)
            schema = page_schema
            total_rows = page_total
        elif page_schema != schema:
            raise GateCError("Dataset Server schema drifted mid-scan")
        elif page_total != total_rows:
            raise GateCError("Dataset Server population metadata drifted mid-scan")
        if not rows:
            return
        for position, (row_index, row) in enumerate(rows):
            yield ScannedRow(
                row_index=row_index,
                row=row,
                response_bytes=page_bytes if position == 0 else 0,
                request_count=attempts if position == 0 else 0,
                live_schema=schema,
                total_rows=page_total,
            )
        offset += len(rows)


# --------------------------------------------------------------------------------------
# Shared mechanical filters
# --------------------------------------------------------------------------------------


def _strict_utf8_bytes(text: Any) -> bytes | None:
    """Return the strict UTF-8 encoding, or ``None`` when the value is not strictly round-trippable."""
    if not isinstance(text, str):
        return None
    if "�" in text:
        return None
    try:
        encoded = text.encode("utf-8", errors="strict")
        if encoded.decode("utf-8", errors="strict") != text:
            return None
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None
    return encoded


def _pathological_repetition(text: str) -> bool:
    """Coarse mechanical garbage signal.  Deliberately not a quality metric.

    Gate E measured this signal as blind on generated text (0/250 on FinePhrase tutorial), so it
    must never be reported as quality evidence for Cosmopedia or FinePhrase.
    """
    stripped = text.strip()
    if not stripped:
        return True
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(lines) >= 8 and Counter(lines).most_common(1)[0][1] / len(lines) >= 0.5:
        return True
    words = stripped.split()
    if len(words) >= 40:
        if Counter(words).most_common(1)[0][1] / len(words) >= 0.4:
            return True
        shingles = Counter(tuple(words[index : index + 8]) for index in range(len(words) - 7))
        if shingles and max(shingles.values()) >= 5:
            return True
    return False


def _significant_words(text: str) -> set[str]:
    return {
        word.lower()
        for word in _WORD_RE.findall(text)
        if len(word) > 3 and word.lower() not in _STOPWORDS
    }


# --------------------------------------------------------------------------------------
# Source-local extraction and filtering
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    """A single record's Gate C outcome."""

    accepted: bool
    reason: str | None = None
    text: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: tuple[str, ...] = ()


def _reject(reason: str) -> Decision:
    return Decision(accepted=False, reason=reason)


def _collect_metadata(row: Mapping[str, Any], source: SourceSpec) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for path in source.metadata_paths:
        value = _get_path(row, path)
        if value is None or isinstance(value, (str, int, float, bool)):
            metadata[path] = value
        elif isinstance(value, (list, tuple)):
            metadata[path] = [item for item in value if isinstance(item, (str, int, float, bool))]
    return metadata


def _extract_body(row: Mapping[str, Any], source: SourceSpec) -> tuple[str | None, str | None]:
    """Return ``(body, rejection_reason)`` using the source's single frozen body path."""
    if source.key == "finephrase_tutorial":
        rollouts = row.get("rollout_results")
        if not isinstance(rollouts, (list, tuple)) or not rollouts:
            return None, "rollout_missing"
        first = rollouts[0]
        if not isinstance(first, Mapping):
            return None, "rollout_missing"
        body = first.get("text")
        if not isinstance(body, str):
            return None, "rollout_missing"
        # Structural guard: the emitted body must be the generated rollout, never the top-level
        # FineWeb-Edu source document.  The flattened layout carries two columns named ``text``.
        if body == row.get("text"):
            return None, "source_text_confusion"
        return body, None
    body = _get_path(row, source.body_path)
    if not isinstance(body, str):
        return None, "body_field_missing"
    return body, None


def _filter_fineweb(row: Mapping[str, Any], text: str) -> Decision:
    del row
    if _pathological_repetition(text):
        return _reject("pathological_repetition")
    return Decision(accepted=True, text=text)


def _filter_dclm(row: Mapping[str, Any], text: str) -> Decision:
    score = row.get("edu_int_score")
    if type(score) is not int:
        return _reject("quality_field_missing")
    if score < DCLM_MIN_EDU_INT_SCORE:
        return _reject("quality_below_minimum")
    if _pathological_repetition(text):
        return _reject("pathological_repetition")
    return Decision(accepted=True, text=text)


def _filter_finewiki(row: Mapping[str, Any], text: str) -> Decision:
    del row
    if _pathological_repetition(text):
        return _reject("pathological_repetition")
    return Decision(accepted=True, text=text)


def _clean_pes2o_markup(text: str) -> tuple[str, bool]:
    """Strip unambiguously safe markup residue; report whether structural residue remains."""
    cleaned = _SUBSUP_TAG_RE.sub("", text)
    cleaned = _NUMERIC_ENTITY_RE.sub(" ", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    structural = bool(_RM_CONSTRUCT_RE.search(cleaned)) or bool(_MOJIBAKE_RE.search(cleaned))
    return cleaned, structural


def _filter_pes2o(row: Mapping[str, Any], text: str) -> Decision:
    cleaned, structural_residue = _clean_pes2o_markup(text)
    if structural_residue:
        return _reject("markup_or_mojibake")
    stripped = cleaned.strip()
    if not stripped:
        return _reject("below_min_bytes")
    if not stripped.endswith(_TERMINAL_PUNCTUATION):
        return _reject("truncated_no_terminal_punctuation")
    if stripped[0].islower() or stripped[0] in ",;:)]":
        return _reject("mid_sentence_start")
    title = _get_path(row, "metadata.title")
    if isinstance(title, str) and _BIBLIOGRAPHIC_TITLE_RE.search(title):
        body = stripped[len(title) :] if stripped.startswith(title) else stripped
        title_words = _significant_words(title)
        if title_words and not (title_words & _significant_words(body)):
            return _reject("title_body_review_mismatch")
    if _pathological_repetition(stripped):
        return _reject("pathological_repetition")
    diagnostics = ("structured_abstract",) if _STRUCTURED_ABSTRACT_RE.search(stripped) else ()
    return Decision(accepted=True, text=cleaned, diagnostics=diagnostics)


def _filter_stackexchange(row: Mapping[str, Any], text: str) -> Decision:
    question_score = _get_path(row, "metadata.question_score")
    answer_score = _get_path(row, "metadata.answer_score")
    if type(question_score) is not int or type(answer_score) is not int:
        return _reject("score_field_missing")
    if question_score < STACKEXCHANGE_MIN_QUESTION_SCORE:
        raise GateCError(
            "StackExchange question_score below the asserted Dolmino floor: "
            f"{question_score} < {STACKEXCHANGE_MIN_QUESTION_SCORE}"
        )
    if answer_score < STACKEXCHANGE_MIN_ANSWER_SCORE:
        raise GateCError(
            "StackExchange answer_score below the asserted Dolmino floor: "
            f"{answer_score} < {STACKEXCHANGE_MIN_ANSWER_SCORE}"
        )
    forum = _get_path(row, "metadata.forum")
    if not isinstance(forum, str) or not forum:
        return _reject("forum_field_missing")
    if forum == "mathoverflow_net":
        return _reject("forum_mathoverflow")
    if _META_FORUM_RE.search(forum):
        return _reject("forum_meta")
    if forum in _LANGUAGE_LOCALIZED_FORUMS:
        return _reject("forum_language_localized")
    if "$$" in text or len(_INLINE_LATEX_RE.findall(text)) >= STACKEXCHANGE_INLINE_LATEX_LIMIT:
        return _reject("heavy_math")
    if len(text.encode("utf-8")) > STACKEXCHANGE_CODE_DOMINANT_BYTES:
        lines = [line for line in text.splitlines() if line.strip()]
        indented = sum(1 for line in lines if _CODE_INDENT_RE.match(line))
        if lines and indented / len(lines) >= STACKEXCHANGE_CODE_DOMINANT_RATIO:
            return _reject("code_dominant")
    if _pathological_repetition(text):
        return _reject("pathological_repetition")
    return Decision(accepted=True, text=text)


def _filter_cosmopedia(row: Mapping[str, Any], text: str) -> Decision:
    document_format = row.get("format")
    audience = row.get("audience")
    if not isinstance(document_format, str) or not document_format:
        return _reject("format_field_missing")
    if document_format in COSMOPEDIA_EXCLUDED_FORMATS:
        return _reject("format_excluded")
    if document_format not in COSMOPEDIA_PRIMARY_FORMATS | COSMOPEDIA_STORY_FORMATS:
        return _reject("format_unknown")
    if isinstance(audience, str) and audience in COSMOPEDIA_EXCLUDED_AUDIENCES:
        return _reject("audience_excluded")
    if row.get("seed_data") == "auto_math_text":
        return _reject("seed_data_auto_math_text")
    body = text.lstrip()
    if _pathological_repetition(body):
        return _reject("pathological_repetition")
    family = "story" if document_format in COSMOPEDIA_STORY_FORMATS else "explanatory"
    return Decision(
        accepted=True,
        text=body,
        metadata={"format_family": family},
        diagnostics=(f"format_family_{family}",),
    )


def _is_outline_only(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return False
    heading_lines = sum(1 for line in lines if _HEADING_LINE_RE.match(line))
    has_prose = any(len(line.strip()) >= 80 for line in lines)
    return heading_lines / len(lines) >= 0.8 and not has_prose


def _declarative_numbered_list(text: str) -> bool:
    """One simple diagnostic: enumerated items that read declarative rather than imperative."""
    lines = [line for line in text.splitlines() if line.strip()]
    enumerated = [line for line in lines if _ENUMERATED_LINE_RE.match(line)]
    if len(enumerated) < 3:
        return False
    imperative = 0
    for line in enumerated:
        match = _IMPERATIVE_LINE_RE.match(line)
        if match is None:
            continue
        word = match.group(1)
        if word[0].isupper() and word.lower() not in _NON_IMPERATIVE_OPENERS:
            imperative += 1
    return imperative / len(enumerated) < 0.5


def _filter_finephrase(row: Mapping[str, Any], text: str) -> Decision:
    rollouts = row.get("rollout_results")
    first = rollouts[0] if isinstance(rollouts, (list, tuple)) and rollouts else {}
    finish_reason = first.get("finish_reason") if isinstance(first, Mapping) else None
    if finish_reason != "stop":
        return _reject("finish_reason_not_stop")
    if _TRANSFORMATION_FRAME_RE.search(text):
        return _reject("transformation_frame")
    if _is_outline_only(text):
        return _reject("outline_only")
    if _MATH_WORKSHEET_RE.search(text):
        return _reject("math_worksheet")
    if _pathological_repetition(text):
        return _reject("pathological_repetition")
    metadata: dict[str, Any] = {"finish_reason": finish_reason}
    source_text = row.get("text")
    if isinstance(source_text, str):
        metadata["source_text_sha256"] = _sha256_bytes(source_text.encode("utf-8"))
        metadata["source_text_bytes"] = len(source_text.encode("utf-8"))
    declarative = _declarative_numbered_list(text)
    metadata["declarative_numbered_list"] = declarative
    return Decision(
        accepted=True,
        text=text,
        metadata=metadata,
        diagnostics=("declarative_numbered_list",) if declarative else (),
    )


_SOURCE_FILTERS: dict[str, Callable[[Mapping[str, Any], str], Decision]] = {
    "fineweb_edu_dedup": _filter_fineweb,
    "dclm_edu": _filter_dclm,
    "finewiki_en": _filter_finewiki,
    "pes2o": _filter_pes2o,
    "stackexchange": _filter_stackexchange,
    "cosmopedia_v2": _filter_cosmopedia,
    "finephrase_tutorial": _filter_finephrase,
}


def evaluate_row(row: Mapping[str, Any], source: SourceSpec) -> Decision:
    """Apply the universal contract, then the source-local filters, to one live row."""
    body, reason = _extract_body(row, source)
    if body is None:
        return _reject(reason or "body_field_missing")
    if _strict_utf8_bytes(body) is None:
        return _reject("strict_utf8_failure")
    decision = _SOURCE_FILTERS[source.key](row, body)
    if not decision.accepted:
        return decision
    emitted = decision.text if decision.text is not None else body
    encoded = _strict_utf8_bytes(emitted)
    if encoded is None:
        return _reject("strict_utf8_failure")
    if len(encoded) < source.min_bytes:
        return _reject("below_min_bytes")
    if len(encoded) > source.max_bytes:
        return _reject("above_max_bytes")
    metadata = _collect_metadata(row, source)
    metadata.update(decision.metadata)
    return Decision(
        accepted=True, text=emitted, metadata=metadata, diagnostics=decision.diagnostics
    )


def _natural_id(row: Mapping[str, Any], source: SourceSpec, row_index: int) -> str:
    if source.key in _ROW_INDEX_IDENTITY:
        return f"row:{row_index}"
    value = _get_path(row, source.natural_id_path or "")
    if isinstance(value, str) and value:
        return value
    if type(value) is int:
        return str(value)
    raise GateCError(f"source record is missing its natural id: {source.key}")


def _source_record_id(source_key: str, natural_id: str) -> str:
    return _sha256_bytes(f"{source_key}\x00{natural_id}".encode())


# --------------------------------------------------------------------------------------
# Checkpoint and configuration
# --------------------------------------------------------------------------------------


def _validate_config(config: BuildConfig) -> None:
    _ensure_revision(config.source.revision)
    if not 1 <= config.target_documents <= MAX_ACCEPTED_DOCUMENTS:
        raise GateCError(f"target_documents must be in [1, {MAX_ACCEPTED_DOCUMENTS}]")
    if not config.target_documents <= config.max_scanned <= MAX_SCANNED_RECORDS:
        raise GateCError(f"max_scanned must be in [target_documents, {MAX_SCANNED_RECORDS}]")
    if not 1 <= config.max_response_bytes <= MAX_RESPONSE_BYTES:
        raise GateCError(f"max_response_bytes must be in [1, {MAX_RESPONSE_BYTES}]")
    if (
        not math.isfinite(config.max_wall_seconds)
        or not 0 < config.max_wall_seconds <= MAX_WALL_SECONDS
    ):
        raise GateCError(f"max_wall_seconds must be in (0, {MAX_WALL_SECONDS}]")
    if type(config.seed) is not int or config.seed < 0:
        raise GateCError("seed must be a non-negative integer")
    if config.stop_after_documents is not None and not (
        1 <= config.stop_after_documents <= config.target_documents
    ):
        raise GateCError("stop_after_documents must be in [1, target_documents]")
    if not 1 <= config.checkpoint_every <= MAX_ACCEPTED_DOCUMENTS:
        raise GateCError(f"checkpoint_every must be in [1, {MAX_ACCEPTED_DOCUMENTS}]")
    for path in (config.output_dir, config.work_dir, _staging_path(config.output_dir)):
        _ensure_git_ignored(path)


def run_fingerprint(config: BuildConfig) -> str:
    """Bind every semantic build input; checkpoint cadence is operational and excluded."""
    source = config.source
    return _sha256_bytes(
        _canonical_json_bytes({
            "tool_schema_version": TOOL_SCHEMA_VERSION,
            "spec_version": SPEC_VERSION,
            "source_key": source.key,
            "dataset": source.dataset,
            "config": source.dataset_config,
            "split": source.split,
            "revision": source.revision,
            "body_path": source.body_path,
            "min_bytes": source.min_bytes,
            "max_bytes": source.max_bytes,
            "required_schema": source.required_schema_map,
            "target_documents": config.target_documents,
            "max_scanned": config.max_scanned,
            "seed": config.seed,
            "output_dir": str(_ensure_under_workspace(config.output_dir)),
        })
    )


def _new_checkpoint(fingerprint: str) -> dict[str, Any]:
    return {
        "tool_schema_version": TOOL_SCHEMA_VERSION,
        "run_fingerprint": fingerprint,
        "next_row_index": 0,
        "counters": BuildCounters().to_json(),
        "seen_record_ids": [],
        "seen_text_sha256": [],
        "documents_sha256": _sha256_bytes(b""),
        "documents_bytes": 0,
        "resume_count": 0,
        "exhausted": False,
        "live_schema": {},
        "forum_histogram": {},
        "updated_at": _utc_now(),
    }


def _checkpoint_payload(state: Mapping[str, Any]) -> bytes:
    return _canonical_json_bytes({k: v for k, v in state.items() if k != "checksum"})


def _write_checkpoint(path: Path, state: Mapping[str, Any]) -> None:
    body = {key: value for key, value in state.items() if key != "checksum"}
    body["checksum"] = _sha256_bytes(_checkpoint_payload(body))
    _atomic_replace_bytes(path, json.dumps(body, indent=2, sort_keys=True).encode() + b"\n")


def _read_checkpoint(path: Path, fingerprint: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise GateCError(f"checkpoint is unreadable: {path}") from exc
    try:
        state = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateCError("checkpoint is corrupted: not strict UTF-8 JSON") from exc
    if not isinstance(state, dict):
        raise GateCError("checkpoint is corrupted: not an object")
    checksum = state.get("checksum")
    if not isinstance(checksum, str) or checksum != _sha256_bytes(_checkpoint_payload(state)):
        raise GateCError("checkpoint checksum mismatch; refusing to resume")
    if state.get("tool_schema_version") != TOOL_SCHEMA_VERSION:
        raise GateCError("checkpoint tool schema version mismatch")
    if state.get("run_fingerprint") != fingerprint:
        raise GateCError("checkpoint run fingerprint mismatch; refusing to resume")
    _require_nonnegative_int(state.get("next_row_index"), "next_row_index")
    _require_nonnegative_int(state.get("documents_bytes"), "documents_bytes")
    _require_nonnegative_int(state.get("resume_count"), "resume_count")
    for name in ("seen_record_ids", "seen_text_sha256"):
        if not isinstance(state.get(name), list):
            raise GateCError(f"checkpoint field is invalid: {name}")
    if not isinstance(state.get("documents_sha256"), str):
        raise GateCError("checkpoint field is invalid: documents_sha256")
    if not isinstance(state.get("exhausted"), bool):
        raise GateCError("checkpoint field is invalid: exhausted")
    BuildCounters.from_json(state.get("counters") or {})
    return state


def _restore_documents_to_checkpoint(path: Path, state: Mapping[str, Any]) -> None:
    """Restore ``documents.jsonl`` to exactly the committed checkpoint prefix.

    Records written after the last checkpoint are uncommitted by definition; dropping them is what
    makes an interruption before a checkpoint safe and duplicate-free on resume.
    """
    expected_bytes = int(state["documents_bytes"])
    expected_sha = str(state["documents_sha256"])
    if not path.exists():
        if expected_bytes != 0:
            raise GateCError("checkpoint references accepted documents but the file is missing")
        _write_new_file(path, b"")
        return
    actual = path.read_bytes()
    if len(actual) < expected_bytes:
        raise GateCError("accepted-document file is shorter than its committed checkpoint prefix")
    if _sha256_bytes(actual[:expected_bytes]) != expected_sha:
        raise GateCError("accepted-document prefix does not match the committed checkpoint hash")
    if len(actual) != expected_bytes:
        with open(path, "r+b") as handle:
            handle.truncate(expected_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)


# --------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------


def _make_manifest(
    config: BuildConfig,
    state: Mapping[str, Any],
    counters: BuildCounters,
    *,
    fingerprint: str,
    live_schema: Mapping[str, str],
    stop_reason: str,
    wall_seconds: float,
    forum_histogram: Mapping[str, int],
    documents_sha256: str,
    documents_bytes: int,
) -> dict[str, Any]:
    source = config.source
    accepted = counters.accepted
    diagnostics = dict(counters.diagnostics)
    return {
        "tool_schema_version": TOOL_SCHEMA_VERSION,
        "spec_version": SPEC_VERSION,
        "generated_at": _utc_now(),
        # A cap-stopped release is otherwise byte-indistinguishable from a completed one, so every
        # release states its own promotability rather than leaving it to be inferred from
        # stop_reason.  Nothing this builder produces is ever promotable.
        "release_kind": "c0_diagnostic",
        "promotion_eligible": False,
        "promotion_eligible_rationale": (
            "production_candidate mode is not implemented; a promotable release would additionally "
            "require stop_reason == 'target_reached', or an owner-accepted 'source_exhausted'. A "
            "cap stop can never be promotable."
        ),
        "run_fingerprint": fingerprint,
        "source": {
            "key": source.key,
            "dataset": source.dataset,
            "config": source.dataset_config,
            "split": source.split,
            "revision": source.revision,
            "license": source.license,
            "body_path": source.body_path,
            "min_bytes": source.min_bytes,
            "max_bytes": source.max_bytes,
            "transport": "huggingface_dataset_server_rows",
        },
        "live_schema": dict(sorted(live_schema.items())),
        "required_schema": source.required_schema_map,
        "schema_verified": bool(live_schema),
        "caps": {
            "target_documents": config.target_documents,
            "max_scanned": config.max_scanned,
            "max_response_bytes": config.max_response_bytes,
            "max_wall_seconds": config.max_wall_seconds,
            "checkpoint_every": config.checkpoint_every,
        },
        "seed": config.seed,
        "accounting": counters.to_json(),
        "yield_rate": (accepted / counters.scanned) if counters.scanned else None,
        "diagnostic_shares": {
            "structured_abstract_share": (
                diagnostics.get("structured_abstract", 0) / accepted if accepted else None
            ),
            "story_family_share": (
                diagnostics.get("format_family_story", 0) / accepted if accepted else None
            ),
            "declarative_numbered_list_share": (
                diagnostics.get("declarative_numbered_list", 0) / accepted if accepted else None
            ),
        },
        "forum_histogram": dict(sorted(forum_histogram.items())),
        "stop_reason": stop_reason,
        "next_row_index": state["next_row_index"],
        "exhausted": bool(state["exhausted"]),
        "resume_count": _require_nonnegative_int(state.get("resume_count"), "resume_count"),
        "wall_seconds": round(wall_seconds, 3),
        "documents_file": DOCUMENTS_NAME,
        "documents_sha256": documents_sha256,
        "documents_bytes": documents_bytes,
        "gate_c_scope": {
            "chat_conversion": False,
            "textual_document_separator": False,
            "tokenizer_counting": False,
            "bos_eos_inserted": False,
            "cross_source_near_dedup": False,
            "benchmark_decontamination": False,
            "reference_reserve_exclusion": False,
        },
        "hard_stops": {
            "bulk_candidate_quota_started": False,
            "tokenizer_trained": False,
            "gate_r_started": False,
            "final_shards_built": False,
            "model_training_started": False,
        },
    }


# --------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------


def build_candidates(
    config: BuildConfig,
    *,
    scanner: Callable[..., Iterator[ScannedRow]] = dataset_server_scanner,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Scan, filter, checkpoint and atomically publish one source-local candidate release."""
    _validate_config(config)
    output_dir = _ensure_under_workspace(config.output_dir)
    if output_dir.exists():
        raise GateCError(f"refusing to overwrite published output: {output_dir}")
    work_dir = _ensure_under_workspace(config.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = run_fingerprint(config)
    checkpoint_path = work_dir / CHECKPOINT_NAME
    documents_path = work_dir / DOCUMENTS_NAME

    resumed = checkpoint_path.exists()
    if resumed:
        state = _read_checkpoint(checkpoint_path, fingerprint)
        state["resume_count"] = int(state["resume_count"]) + 1
    else:
        state = _new_checkpoint(fingerprint)
    _restore_documents_to_checkpoint(documents_path, state)

    counters = BuildCounters.from_json(state["counters"])
    seen_record_ids = set(state["seen_record_ids"])
    seen_text_hashes = set(state["seen_text_sha256"])
    documents_digest = hashlib.sha256(documents_path.read_bytes())
    documents_bytes = int(state["documents_bytes"])
    forum_histogram: Counter = Counter(state.get("forum_histogram") or {})
    live_schema: dict[str, str] = dict(state.get("live_schema") or {})
    next_row_index = int(state["next_row_index"])
    exhausted = bool(state["exhausted"])

    start = clock()
    handle = open(documents_path, "ab")

    def commit(is_exhausted: bool) -> None:
        handle.flush()
        os.fsync(handle.fileno())
        state["next_row_index"] = next_row_index
        state["counters"] = counters.to_json()
        state["seen_record_ids"] = sorted(seen_record_ids)
        state["seen_text_sha256"] = sorted(seen_text_hashes)
        state["documents_sha256"] = documents_digest.hexdigest()
        state["documents_bytes"] = documents_bytes
        state["exhausted"] = is_exhausted
        state["live_schema"] = live_schema
        state["forum_histogram"] = dict(forum_histogram)
        state["updated_at"] = _utc_now()
        _write_checkpoint(checkpoint_path, state)

    try:
        if counters.accepted >= config.target_documents:
            stop_reason = "target_reached"
        elif exhausted:
            stop_reason = "source_exhausted"
        else:
            stop_reason = "source_exhausted"
            since_checkpoint = 0
            for item in scanner(config.source, start_row=next_row_index):
                counters.response_bytes += item.response_bytes
                counters.request_count += item.request_count
                if not live_schema:
                    live_schema = dict(item.live_schema)
                if counters.response_bytes > config.max_response_bytes:
                    stop_reason = "byte_cap"
                    break
                if clock() - start > config.max_wall_seconds:
                    stop_reason = "time_cap"
                    break
                if item.row_index < next_row_index:
                    raise GateCError("scanner produced a row before the checkpointed cursor")
                next_row_index = item.row_index + 1
                counters.scanned += 1

                decision = evaluate_row(item.row, config.source)
                if not decision.accepted:
                    counters.rejected += 1
                    counters.rejections[decision.reason or "unspecified"] += 1
                else:
                    natural_id = _natural_id(item.row, config.source, item.row_index)
                    record_id = _source_record_id(config.source.key, natural_id)
                    text = decision.text or ""
                    text_bytes = text.encode("utf-8")
                    text_sha = _sha256_bytes(text_bytes)
                    if record_id in seen_record_ids:
                        counters.rejected += 1
                        counters.rejections["duplicate_source_record_id"] += 1
                    elif text_sha in seen_text_hashes:
                        counters.rejected += 1
                        counters.rejections["duplicate_text_sha256"] += 1
                    else:
                        seen_record_ids.add(record_id)
                        seen_text_hashes.add(text_sha)
                        line = canonical_jsonl_record_bytes({
                            "source_key": config.source.key,
                            "source_record_id": record_id,
                            "natural_id": natural_id,
                            "text": text,
                            "text_sha256": text_sha,
                            "text_bytes": len(text_bytes),
                            "row_index": item.row_index,
                            "metadata": dict(decision.metadata),
                            "provenance": {
                                "dataset": config.source.dataset,
                                "config": config.source.dataset_config,
                                "split": config.source.split,
                                "revision": config.source.revision,
                                "license": config.source.license,
                                "transport": "huggingface_dataset_server_rows",
                            },
                        })
                        handle.write(line)
                        documents_digest.update(line)
                        documents_bytes += len(line)
                        counters.accepted += 1
                        counters.accepted_text_bytes += len(text_bytes)
                        for name in decision.diagnostics:
                            counters.diagnostics[name] += 1
                        forum = decision.metadata.get("metadata.forum")
                        if isinstance(forum, str) and forum:
                            forum_histogram[forum] += 1
                        since_checkpoint += 1

                if counters.accepted >= config.target_documents:
                    stop_reason = "target_reached"
                    break
                if config.stop_after_documents is not None and (
                    counters.accepted >= config.stop_after_documents
                ):
                    stop_reason = "stop_after_documents"
                    break
                if counters.scanned >= config.max_scanned:
                    stop_reason = "scan_cap"
                    break
                if since_checkpoint >= config.checkpoint_every:
                    commit(False)
                    since_checkpoint = 0
            exhausted = stop_reason == "source_exhausted"
        # Committing only on a normal stop is deliberate: an interruption leaves the last committed
        # checkpoint authoritative, and the uncommitted suffix is truncated on the next resume.
        commit(exhausted)
    finally:
        handle.close()

    if stop_reason == "stop_after_documents":
        return {
            "published": False,
            "stop_reason": stop_reason,
            "work_dir": str(work_dir),
            "accepted": counters.accepted,
            "scanned": counters.scanned,
            "rejected": counters.rejected,
            "response_bytes": counters.response_bytes,
            "next_row_index": next_row_index,
            "resume_count": int(state["resume_count"]),
            "resumed": resumed,
            "documents_sha256": documents_digest.hexdigest(),
        }

    wall_seconds = clock() - start
    manifest = _make_manifest(
        config,
        state,
        counters,
        fingerprint=fingerprint,
        live_schema=live_schema,
        stop_reason=stop_reason,
        wall_seconds=wall_seconds,
        forum_histogram=forum_histogram,
        documents_sha256=documents_digest.hexdigest(),
        documents_bytes=documents_bytes,
    )

    staging = _staging_path(output_dir)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copyfile(documents_path, staging / DOCUMENTS_NAME)
    _write_new_file(
        staging / MANIFEST_NAME, json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    )
    _write_new_file(
        staging / CHECKSUMS_NAME,
        "".join(
            f"{_sha256_bytes((staging / name).read_bytes())}  {name}\n"
            for name in (DOCUMENTS_NAME, MANIFEST_NAME)
        ).encode(),
    )
    _publish_directory(staging, output_dir)

    published_sha = _sha256_bytes((output_dir / DOCUMENTS_NAME).read_bytes())
    if published_sha != manifest["documents_sha256"]:
        raise GateCError("published document file does not match its manifest checksum")

    return {
        "published": True,
        "output_dir": str(output_dir),
        "stop_reason": stop_reason,
        "accepted": counters.accepted,
        "scanned": counters.scanned,
        "rejected": counters.rejected,
        "accepted_text_bytes": counters.accepted_text_bytes,
        "response_bytes": counters.response_bytes,
        "request_count": counters.request_count,
        "wall_seconds": round(wall_seconds, 3),
        "resume_count": int(state["resume_count"]),
        "resumed": resumed,
        "next_row_index": next_row_index,
        "documents_sha256": published_sha,
        "manifest_sha256": _sha256_bytes((output_dir / MANIFEST_NAME).read_bytes()),
        "rejections": dict(sorted(counters.rejections.items())),
    }


def verify_release(output_dir: Path) -> dict[str, Any]:
    """Re-read a published release and verify every recorded checksum."""
    output_dir = _ensure_under_workspace(output_dir)
    manifest_bytes = (output_dir / MANIFEST_NAME).read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8", errors="strict"))
    documents_bytes = (output_dir / DOCUMENTS_NAME).read_bytes()
    documents_sha = _sha256_bytes(documents_bytes)
    if documents_sha != manifest["documents_sha256"]:
        raise GateCError("documents.jsonl does not match manifest documents_sha256")
    if len(documents_bytes) != manifest["documents_bytes"]:
        raise GateCError("documents.jsonl length does not match the manifest")
    record_ids: set[str] = set()
    text_hashes: set[str] = set()
    accepted = 0
    byte_lines = [line for line in documents_bytes.split(b"\n") if line]
    str_lines = [line for line in documents_bytes.decode("utf-8").splitlines() if line]
    if len(byte_lines) != len(str_lines):
        raise GateCError(
            f"physical JSONL framing is ambiguous: {len(byte_lines)} byte-delimited records vs "
            f"{len(str_lines)} str.splitlines() records"
        )
    for line in byte_lines:
        record = json.loads(line.decode("utf-8", errors="strict"))
        if _sha256_bytes(record["text"].encode("utf-8")) != record["text_sha256"]:
            raise GateCError("candidate record text_sha256 was tampered with")
        if record["source_record_id"] in record_ids:
            raise GateCError("published release contains a duplicate source record id")
        if record["text_sha256"] in text_hashes:
            raise GateCError("published release contains a duplicate text hash")
        record_ids.add(record["source_record_id"])
        text_hashes.add(record["text_sha256"])
        accepted += 1
    if accepted != manifest["accounting"]["accepted"]:
        raise GateCError("published record count does not match the manifest accounting")
    for entry in (output_dir / CHECKSUMS_NAME).read_text().splitlines():
        digest, name = entry.split("  ", 1)
        if _sha256_bytes((output_dir / name).read_bytes()) != digest:
            raise GateCError(f"MANIFEST.sha256 mismatch for {name}")
    return {
        "output_dir": str(output_dir),
        "accepted": accepted,
        "documents_sha256": documents_sha,
        "manifest_sha256": _sha256_bytes(manifest_bytes),
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PetitGPT non-Python Gate C candidate builder")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build one source-local candidate release")
    build.add_argument("--source", required=True, choices=sorted(SOURCES))
    build.add_argument("--output-dir", required=True, type=Path)
    build.add_argument("--work-dir", required=True, type=Path)
    build.add_argument("--target-documents", required=True, type=int)
    build.add_argument("--max-scanned", required=True, type=int)
    build.add_argument("--max-response-bytes", required=True, type=int)
    build.add_argument("--max-wall-seconds", required=True, type=float)
    build.add_argument("--seed", required=True, type=int)
    build.add_argument("--stop-after-documents", type=int, default=None)
    build.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)

    verify = subparsers.add_parser("verify", help="verify a published candidate release")
    verify.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "build":
            result = build_candidates(
                BuildConfig(
                    source=SOURCES[args.source],
                    output_dir=args.output_dir,
                    work_dir=args.work_dir,
                    target_documents=args.target_documents,
                    max_scanned=args.max_scanned,
                    max_response_bytes=args.max_response_bytes,
                    max_wall_seconds=args.max_wall_seconds,
                    seed=args.seed,
                    stop_after_documents=args.stop_after_documents,
                    checkpoint_every=args.checkpoint_every,
                )
            )
        else:
            result = verify_release(args.output_dir)
    except GateCError as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
