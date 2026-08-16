"""Pure synthetic contract tests for the non-Python Gate C candidate builder.

No network, no tokenizer, no GPU, no real corpus.  Every source row here is fabricated so the tests
exercise the fail-closed contracts rather than upstream data.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pretrain import corpus_gate_c as gc  # noqa: E402

RUNS_ROOT = PROJECT_ROOT / "runs"


# --------------------------------------------------------------------------------------
# Synthetic rows and scanners
# --------------------------------------------------------------------------------------


def _prose(index: int, sentences: int = 14) -> str:
    """Varied synthetic prose.

    Every 8-word window carries at least one unique number, so this text passes the coarse
    pathological-repetition gate.  Fixtures that need to trip that gate build repetition explicitly.
    """
    return " ".join(
        f"Observation {index}-{position} recorded a value of {index * 31 + position * 7} "
        f"during trial {position}."
        for position in range(sentences)
    )


def _finewiki_row(index: int, *, text: str | None = None) -> dict[str, object]:
    body = text if text is not None else f"Topic {index}. {_prose(index)}"
    return {
        "bytes_html": 4096,
        "date_modified": "2024-01-01",
        "has_math": False,
        "id": f"finewiki-{index}",
        "in_language": "en",
        "infoboxes": "",
        "page_id": 1000 + index,
        "text": body,
        "title": f"Title {index}",
        "url": f"https://en.wikipedia.org/wiki/Topic_{index}",
        "version": 1,
        "wikidata_id": f"Q{index}",
        "wikiname": "enwiki",
        "wikitext": "== Heading ==",
    }


FINEWIKI_SCHEMA = {
    "bytes_html": "int64",
    "date_modified": "string",
    "has_math": "bool",
    "id": "string",
    "in_language": "string",
    "infoboxes": "string",
    "page_id": "int64",
    "text": "string",
    "title": "string",
    "url": "string",
    "version": "int64",
    "wikidata_id": "string",
    "wikiname": "string",
    "wikitext": "string",
}


def make_scanner(
    rows: list[dict[str, object]],
    *,
    schema: dict[str, str] | None = None,
    page_bytes: int = 1024,
    page_rows: int = 5,
    interrupt_at: int | None = None,
):
    """Build a deterministic in-memory scanner over ``rows``."""
    live_schema = dict(schema if schema is not None else FINEWIKI_SCHEMA)

    def scanner(source: gc.SourceSpec, *, start_row: int):
        gc.assert_schema(source, live_schema)
        for offset, row in enumerate(rows[start_row:], start=start_row):
            if interrupt_at is not None and offset >= interrupt_at:
                raise gc.PlannedInterruption(f"planned interruption at row {offset}")
            yield gc.ScannedRow(
                row_index=offset,
                row=row,
                response_bytes=page_bytes if offset % page_rows == 0 else 0,
                request_count=1 if offset % page_rows == 0 else 0,
                live_schema=live_schema,
                total_rows=len(rows),
            )

    return scanner


def make_config(tmp_path: Path, **overrides) -> gc.BuildConfig:
    base = {
        "source": gc.SOURCES["finewiki_en"],
        "output_dir": tmp_path / "release",
        "work_dir": tmp_path / "work",
        "target_documents": 4,
        "max_scanned": 100,
        "max_response_bytes": 64 * 1024 * 1024,
        "max_wall_seconds": 60.0,
        "seed": 20260816,
        "checkpoint_every": 2,
    }
    base.update(overrides)
    return gc.BuildConfig(**base)


@pytest.fixture
def run_dir(tmp_path_factory) -> Path:
    """A Git-ignored working directory: Gate C refuses to write anywhere else."""
    root = RUNS_ROOT / "_pytest_gate_c"
    root.mkdir(parents=True, exist_ok=True)
    path = Path(tmp_path_factory.mktemp("case", numbered=True))
    target = root / path.name
    target.mkdir(parents=True, exist_ok=True)
    yield target
    shutil.rmtree(target, ignore_errors=True)


def read_documents(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


# --------------------------------------------------------------------------------------
# Source bindings
# --------------------------------------------------------------------------------------


def test_no_python_source_is_bound():
    """The Python track is governance-blocked; no Python binding may exist in this builder."""
    blob = json.dumps({
        key: [spec.dataset, spec.dataset_config] for key, spec in gc.SOURCES.items()
    }).lower()
    for forbidden in (
        "python-edu",
        "stack-edu",
        "stackv2",
        "the-stack",
        "common-pile",
        "software-heritage",
        "starcoder",
    ):
        assert forbidden not in blob
    assert not any("python" in key for key in gc.SOURCES)


def test_active_source_set_is_exactly_the_seven_non_python_sources():
    assert sorted(gc.SOURCES) == [
        "cosmopedia_v2",
        "dclm_edu",
        "finephrase_tutorial",
        "fineweb_edu_dedup",
        "finewiki_en",
        "pes2o",
        "stackexchange",
    ]
    # FinePhrase faq is excluded from the active 13B and must not be bound.
    assert all(spec.dataset_config != "faq" for spec in gc.SOURCES.values())


def test_every_binding_pins_an_exact_revision():
    for spec in gc.SOURCES.values():
        assert gc._ensure_revision(spec.revision) == spec.revision


# --------------------------------------------------------------------------------------
# Schema drift
# --------------------------------------------------------------------------------------


def test_schema_drift_added_column_fails_closed(run_dir):
    schema = dict(FINEWIKI_SCHEMA, unexpected_column="string")
    config = make_config(run_dir)
    with pytest.raises(gc.GateCError, match="live schema drift"):
        gc.build_candidates(
            config, scanner=make_scanner([_finewiki_row(i) for i in range(8)], schema=schema)
        )


def test_schema_drift_missing_column_fails_closed(run_dir):
    schema = {k: v for k, v in FINEWIKI_SCHEMA.items() if k != "page_id"}
    config = make_config(run_dir)
    with pytest.raises(gc.GateCError, match="live schema drift"):
        gc.build_candidates(
            config, scanner=make_scanner([_finewiki_row(i) for i in range(8)], schema=schema)
        )


def test_schema_drift_changed_dtype_fails_closed(run_dir):
    schema = dict(FINEWIKI_SCHEMA, page_id="string")
    config = make_config(run_dir)
    with pytest.raises(gc.GateCError, match="live schema drift"):
        gc.build_candidates(
            config, scanner=make_scanner([_finewiki_row(i) for i in range(8)], schema=schema)
        )


def test_revision_drift_is_rejected_by_the_transport():
    class _Response:
        headers = {"x-revision": "0" * 40, "content-length": "10"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, _size):
            return b"{}"

    with pytest.raises(gc.GateCError, match="x-revision"):
        gc._dataset_server_request(
            gc.SOURCES["finewiki_en"],
            offset=0,
            length=1,
            opener=lambda *a, **k: _Response(),
            sleeper=lambda _s: None,
        )


def test_unpinned_list_dtype_is_tolerated_but_the_path_is_not():
    """A ``None`` dtype leaves the dtype unpinned; the path itself is still mandatory."""
    source = gc.SOURCES["stackexchange"]
    unpinned = "attributes.dedupe_para_ngrams_13_1"
    assert source.required_schema_map[unpinned] is None
    live = {path: (dtype or "list[list[float64]]") for path, dtype in source.required_schema}
    gc.assert_schema(source, live)
    live[unpinned] = "list[list[int64]]"
    gc.assert_schema(source, live)
    del live[unpinned]
    with pytest.raises(gc.GateCError, match="missing="):
        gc.assert_schema(source, live)


def test_pes2o_struct_list_leaves_are_pinned_exactly():
    """PES2O ``top_frequencies`` is a {token, count} struct list, i.e. two addressable leaves."""
    schema = gc.SOURCES["pes2o"].required_schema_map
    assert schema["metadata.top_frequencies.token"] == "list[string]"
    assert schema["metadata.top_frequencies.count"] == "list[int64]"
    assert "metadata.top_frequencies" not in schema
    assert all(dtype is not None for dtype in schema.values())


def test_structured_list_flattens_to_addressable_child_paths():
    features = [
        {
            "feature_idx": 0,
            "name": "rollout_results",
            "type": {
                "feature": {
                    "finish_reason": {"dtype": "string", "_type": "Value"},
                    "text": {"dtype": "string", "_type": "Value"},
                },
                "_type": "List",
            },
        }
    ]
    assert gc._server_schema(features) == {
        "rollout_results.finish_reason": "list[string]",
        "rollout_results.text": "list[string]",
    }


# --------------------------------------------------------------------------------------
# Universal contract: strict UTF-8, dedup, byte window
# --------------------------------------------------------------------------------------


def test_strict_utf8_failure_is_counted_and_rejected(run_dir):
    rows = [_finewiki_row(i) for i in range(6)]
    rows[0]["text"] = "Broken � replacement character text. " * 20
    rows[1]["text"] = "Lone surrogate \ud800 in the body. " * 20
    config = make_config(run_dir, target_documents=3)
    result = gc.build_candidates(config, scanner=make_scanner(rows))
    assert result["rejections"]["strict_utf8_failure"] == 2
    assert result["accepted"] == 3


def test_duplicate_source_record_ids_are_rejected_once(run_dir):
    rows = [_finewiki_row(i) for i in range(6)]
    rows[1]["id"] = rows[0]["id"]
    rows[1]["text"] = f"Distinct body. {_prose(99)}"
    config = make_config(run_dir, target_documents=4)
    result = gc.build_candidates(config, scanner=make_scanner(rows))
    assert result["rejections"]["duplicate_source_record_id"] == 1
    documents = read_documents(Path(result["output_dir"]) / gc.DOCUMENTS_NAME)
    assert len({doc["source_record_id"] for doc in documents}) == len(documents)


def test_duplicate_text_hashes_are_rejected_once(run_dir):
    rows = [_finewiki_row(i) for i in range(6)]
    rows[1]["text"] = rows[0]["text"]
    config = make_config(run_dir, target_documents=4)
    result = gc.build_candidates(config, scanner=make_scanner(rows))
    assert result["rejections"]["duplicate_text_sha256"] == 1
    documents = read_documents(Path(result["output_dir"]) / gc.DOCUMENTS_NAME)
    assert len({doc["text_sha256"] for doc in documents}) == len(documents)


def test_byte_window_rejects_short_and_long_bodies(run_dir):
    rows = [_finewiki_row(i) for i in range(6)]
    rows[0]["text"] = "too short"
    rows[1]["text"] = _prose(1, sentences=3000)
    config = make_config(run_dir, target_documents=3)
    result = gc.build_candidates(config, scanner=make_scanner(rows))
    assert result["rejections"]["below_min_bytes"] == 1
    assert result["rejections"]["above_max_bytes"] == 1


def test_pathological_repetition_is_rejected(run_dir):
    rows = [_finewiki_row(i) for i in range(6)]
    rows[0]["text"] = "the same sentence repeated again and again.\n" * 40
    config = make_config(run_dir, target_documents=3)
    result = gc.build_candidates(config, scanner=make_scanner(rows))
    assert result["rejections"]["pathological_repetition"] == 1


# --------------------------------------------------------------------------------------
# Source-specific rejection counters
# --------------------------------------------------------------------------------------


def _dclm_row(index: int, score: int) -> dict[str, object]:
    return {
        "edu_int_score": score,
        "edu_score": float(score) + 0.2,
        "fasttext_score": 0.9,
        "id": f"dclm-{index}",
        "language": "en",
        "language_score": 0.99,
        "text": f"Educational passage {index}. {_prose(index)}",
        "url": f"https://example.org/{index}",
    }


DCLM_SCHEMA = {path: dtype for path, dtype in gc.SOURCES["dclm_edu"].required_schema}


def test_dclm_edu_int_score_floor(run_dir):
    rows = [_dclm_row(0, 2), _dclm_row(1, 1), _dclm_row(2, 3), _dclm_row(3, 4), _dclm_row(4, 3)]
    config = make_config(run_dir, source=gc.SOURCES["dclm_edu"], target_documents=3)
    result = gc.build_candidates(config, scanner=make_scanner(rows, schema=DCLM_SCHEMA))
    assert result["rejections"]["quality_below_minimum"] == 2
    assert result["accepted"] == 3
    # Stage B is deliberately not frozen at edu_int_score == 4; the score metadata is retained so a
    # real post-filter score curve can decide later.
    documents = read_documents(Path(result["output_dir"]) / gc.DOCUMENTS_NAME)
    assert {doc["metadata"]["edu_int_score"] for doc in documents} == {3, 4}


def _stackexchange_row(index: int, **overrides) -> dict[str, object]:
    row = {
        "added": "2024-01-01",
        "attributes": {"dedupe_para_ngrams_13_1": []},
        "created": "2020-01-01",
        "id": f"se-{index}",
        "metadata": {
            "answer_comment_count": 1,
            "answer_content_license": "CC BY-SA 4.0",
            "answer_id": 2000 + index,
            "answer_last_activity_date": "2020-01-02",
            "answer_last_edit_date": "2020-01-02",
            "answer_last_editor_user_id": 5,
            "answer_owner_user_id": 6,
            "answer_score": 10,
            "answer_view_count": 100,
            "forum": "stackoverflow_com",
            "provenance": "dolmino",
            "question_comment_count": 0,
            "question_content_license": "CC BY-SA 4.0",
            "question_id": 1000 + index,
            "question_last_activity_date": "2020-01-02",
            "question_last_edit_date": "2020-01-02",
            "question_last_editor_user_id": 7,
            "question_owner_user_id": 8,
            "question_score": 8,
            "question_view_count": 500,
        },
        "source": "stackexchange",
        "text": (
            f"How do I configure the widget {index}?\n\n\n"
            "I have tried the obvious approach and it does not work.\n\n"
            "Set the timeout before opening the connection; otherwise the handshake fails. "
            "The reason is that the library caches the value at construction time."
        ),
        "version": "v1",
    }
    metadata_overrides = overrides.pop("metadata", {})
    row["metadata"].update(metadata_overrides)
    row.update(overrides)
    return row


STACKEXCHANGE_SCHEMA = {
    path: (dtype or "list[list[float64]]")
    for path, dtype in gc.SOURCES["stackexchange"].required_schema
}


def test_stackexchange_forum_and_content_exclusions(run_dir):
    rows = [
        _stackexchange_row(0, metadata={"forum": "mathoverflow_net"}),
        _stackexchange_row(1, metadata={"forum": "ru_stackoverflow_com"}),
        _stackexchange_row(2, metadata={"forum": "stackoverflow_meta_com"}),
        _stackexchange_row(3),
        _stackexchange_row(4),
    ]
    rows[3]["text"] = rows[3]["text"] + "\n\n$$\\int_0^1 f(x)\\,dx$$"
    rows[4]["text"] = (
        "How do I read this file?\n\n\nI need help.\n\n" + "    line of source code here\n" * 400
    )
    rows.extend([_stackexchange_row(5), _stackexchange_row(6)])
    config = make_config(run_dir, source=gc.SOURCES["stackexchange"], target_documents=2)
    result = gc.build_candidates(config, scanner=make_scanner(rows, schema=STACKEXCHANGE_SCHEMA))
    assert result["rejections"]["forum_mathoverflow"] == 1
    assert result["rejections"]["forum_language_localized"] == 1
    assert result["rejections"]["forum_meta"] == 1
    assert result["rejections"]["heavy_math"] == 1
    assert result["rejections"]["code_dominant"] == 1
    assert result["accepted"] == 2


def test_stackexchange_topic_prefixed_forums_are_not_treated_as_localized(run_dir):
    """cs.stackexchange (Computer Science) and vi.stackexchange (Vi/Vim) are English and on-topic.

    An open-ended language-prefix rule excluded both; the deny list is now the explicit set of
    localized Stack Overflow hosts found in the complete 361-forum census.
    """
    rows = [
        _stackexchange_row(0, metadata={"forum": "cs_stackexchange_com"}),
        _stackexchange_row(1, metadata={"forum": "vi_stackexchange_com"}),
        _stackexchange_row(2, metadata={"forum": "ru_stackoverflow_com"}),
        _stackexchange_row(3, metadata={"forum": "pt_stackoverflow_com"}),
        _stackexchange_row(4, metadata={"forum": "es_stackoverflow_com"}),
        _stackexchange_row(5, metadata={"forum": "ja_stackoverflow_com"}),
    ]
    config = make_config(run_dir, source=gc.SOURCES["stackexchange"], target_documents=2)
    result = gc.build_candidates(config, scanner=make_scanner(rows, schema=STACKEXCHANGE_SCHEMA))
    assert result["accepted"] == 2
    assert result["rejections"].get("forum_language_localized", 0) == 0
    assert gc._LANGUAGE_LOCALIZED_FORUMS == {
        "es_stackoverflow_com",
        "ja_stackoverflow_com",
        "pt_stackoverflow_com",
        "ru_stackoverflow_com",
    }


def test_stackexchange_plain_html_text_is_not_removed(run_dir):
    rows = [_stackexchange_row(index) for index in range(3)]
    rows[0]["text"] = (
        "How do I escape a <div> tag inside my template?\n\n\n"
        "My renderer keeps eating the <span> elements.\n\n"
        "Escape the angle brackets as &lt; and &gt; before handing the string to the renderer. "
        "The template engine only unescapes once, so a single pass is enough."
    )
    config = make_config(run_dir, source=gc.SOURCES["stackexchange"], target_documents=3)
    result = gc.build_candidates(config, scanner=make_scanner(rows, schema=STACKEXCHANGE_SCHEMA))
    assert result["accepted"] == 3
    assert result["rejected"] == 0


def test_stackexchange_score_floor_is_a_fail_closed_assertion(run_dir):
    rows = [_stackexchange_row(0, metadata={"answer_score": 1})]
    config = make_config(run_dir, source=gc.SOURCES["stackexchange"], target_documents=1)
    with pytest.raises(gc.GateCError, match="answer_score below the asserted Dolmino floor"):
        gc.build_candidates(config, scanner=make_scanner(rows, schema=STACKEXCHANGE_SCHEMA))


def test_stackexchange_documents_stay_ordinary_cpt_text(run_dir):
    rows = [_stackexchange_row(index) for index in range(3)]
    config = make_config(run_dir, source=gc.SOURCES["stackexchange"], target_documents=3)
    result = gc.build_candidates(config, scanner=make_scanner(rows, schema=STACKEXCHANGE_SCHEMA))
    documents = read_documents(Path(result["output_dir"]) / gc.DOCUMENTS_NAME)
    for document, row in zip(documents, rows, strict=True):
        assert document["text"] == row["text"]
        for forbidden in ("<|user|>", "<|assistant|>", "<|system|>", "[BOS]", "[EOS]"):
            assert forbidden not in document["text"]
    manifest = json.loads((Path(result["output_dir"]) / gc.MANIFEST_NAME).read_text())
    assert manifest["gate_c_scope"]["chat_conversion"] is False
    assert manifest["gate_c_scope"]["textual_document_separator"] is False
    assert manifest["gate_c_scope"]["bos_eos_inserted"] is False
    assert manifest["source"]["license"] == "cc-by-sa"
    assert manifest["forum_histogram"] == {"stackoverflow_com": 3}


def _pes2o_row(index: int, *, title: str | None = None, text: str | None = None):
    resolved_title = title if title is not None else f"A study of subject {index}"
    body = (
        text
        if text is not None
        else (
            f"{resolved_title}\n\nWe investigate subject {index} in plain language. {_prose(index)}"
        )
    )
    return {
        "added": "2024-01-01",
        "created": "2019-01-01",
        "id": f"pes2o-{index}",
        "metadata": {
            "abstract": "abstract text",
            "abstract_count": 40,
            "abstract_language": "en",
            "abstract_perplexity": 20.0,
            "extfieldsofstudy": ["Medicine"],
            "provenance": "s2",
            "s2fieldsofstudy": ["Medicine"],
            "sha1": "0" * 40,
            "sources": ["s2"],
            "title": resolved_title,
            "title_count": 6,
            "title_language": "en",
            "title_perplexity": 30.0,
            "top_frequencies": ["the"],
            "year": 2019,
        },
        "source": "pes2o",
        "text": body,
        "version": "v1",
    }


PES2O_SCHEMA = {
    path: (dtype or "list[string]") for path, dtype in gc.SOURCES["pes2o"].required_schema
}


def test_pes2o_structural_rejections_and_markup_cleaning(run_dir):
    truncated = _pes2o_row(0)
    truncated["text"] = truncated["text"].rstrip(".") + " and the measured response was"
    mid_sentence = _pes2o_row(1)
    mid_sentence["text"] = "and therefore the observed effect persisted across every cohort tested."
    mojibake = _pes2o_row(2)
    mojibake["text"] = mojibake["text"] + " The value was ${\\rm 5}$ units."
    review = _pes2o_row(
        3,
        title="Quantum Cartography. Cambridge Press 2019, 240 pp.",
        text=(
            "Quantum Cartography. Cambridge Press 2019, 240 pp.\n\n"
            "Sunflower cultivation depends heavily on irrigation scheduling. "
            f"{_prose(88)}"
        ),
    )
    markup = _pes2o_row(4)
    markup["text"] = markup["text"].replace("subject", "H<sub>2</sub>O&#160;subject")
    structured = _pes2o_row(5)
    structured["text"] = (
        "Structured report on subject five\n\nOBJECTIVE: To measure the effect. "
        "METHODS: A randomised trial. RESULTS: The effect was small. "
        f"CONCLUSIONS: Further work is needed. {_prose(55)}"
    )
    rows = [truncated, mid_sentence, mojibake, review, markup, structured]
    config = make_config(run_dir, source=gc.SOURCES["pes2o"], target_documents=2)
    result = gc.build_candidates(config, scanner=make_scanner(rows, schema=PES2O_SCHEMA))
    assert result["rejections"]["truncated_no_terminal_punctuation"] == 1
    assert result["rejections"]["mid_sentence_start"] == 1
    assert result["rejections"]["markup_or_mojibake"] == 1
    assert result["rejections"]["title_body_review_mismatch"] == 1
    assert result["accepted"] == 2

    documents = read_documents(Path(result["output_dir"]) / gc.DOCUMENTS_NAME)
    cleaned = documents[0]["text"]
    assert "<sub>" not in cleaned and "&#160;" not in cleaned
    manifest = json.loads((Path(result["output_dir"]) / gc.MANIFEST_NAME).read_text())
    # Structured abstracts are measured, not filtered.
    assert manifest["diagnostic_shares"]["structured_abstract_share"] == 0.5


def test_pes2o_does_not_filter_on_field_of_study_or_readability(run_dir):
    rows = [_pes2o_row(index) for index in range(3)]
    rows[0]["metadata"]["s2fieldsofstudy"] = ["Medicine"]
    rows[1]["metadata"]["s2fieldsofstudy"] = ["Physics"]
    rows[2]["text"] = (
        "Heteroscedastic covariance estimation\n\nWe derive a consistent estimator of the "
        "heteroscedastic covariance operator under weak mixing assumptions. "
        f"{_prose(77)}"
    )
    config = make_config(run_dir, source=gc.SOURCES["pes2o"], target_documents=3)
    result = gc.build_candidates(config, scanner=make_scanner(rows, schema=PES2O_SCHEMA))
    assert result["accepted"] == 3


def _cosmopedia_row(
    index: int,
    *,
    document_format: str = "textbook",
    audience: str = "middle_school",
    seed_data: str = "fineweb",
    text: str | None = None,
) -> dict[str, object]:
    body = (
        text if text is not None else (f" Chapter {index}: the topic explained.\n\n{_prose(index)}")
    )
    return {
        "audience": audience,
        "format": document_format,
        "prompt": "PROMPT TEXT THAT MUST NEVER BE EMITTED AS A TRAINING CANDIDATE",
        "seed_data": seed_data,
        "text": body,
        "token_length": 512,
    }


COSMOPEDIA_SCHEMA = {path: dtype for path, dtype in gc.SOURCES["cosmopedia_v2"].required_schema}


def test_cosmopedia_never_emits_prompt_or_seed_data(run_dir):
    rows = [_cosmopedia_row(index) for index in range(3)]
    config = make_config(run_dir, source=gc.SOURCES["cosmopedia_v2"], target_documents=3)
    result = gc.build_candidates(config, scanner=make_scanner(rows, schema=COSMOPEDIA_SCHEMA))
    documents = read_documents(Path(result["output_dir"]) / gc.DOCUMENTS_NAME)
    assert len(documents) == 3
    for document, row in zip(documents, rows, strict=True):
        assert "PROMPT TEXT" not in document["text"]
        assert document["text"] == row["text"].lstrip()
        # lstrip() only: internal newlines survive verbatim.
        assert "\n\n" in document["text"]
    assert gc.SOURCES["cosmopedia_v2"].body_path == "text"


def test_cosmopedia_format_audience_and_seed_exclusions(run_dir):
    rows = [
        _cosmopedia_row(0, document_format="dialogue"),
        _cosmopedia_row(1, document_format="story_reddit"),
        _cosmopedia_row(2, document_format="story_forums"),
        _cosmopedia_row(3, audience="alien"),
        _cosmopedia_row(4, audience="requires_details"),
        _cosmopedia_row(5, seed_data="auto_math_text"),
        _cosmopedia_row(6, document_format="wikihow"),
        _cosmopedia_row(7, document_format="story_children"),
    ]
    config = make_config(run_dir, source=gc.SOURCES["cosmopedia_v2"], target_documents=2)
    result = gc.build_candidates(config, scanner=make_scanner(rows, schema=COSMOPEDIA_SCHEMA))
    assert result["rejections"]["format_excluded"] == 3
    assert result["rejections"]["audience_excluded"] == 2
    assert result["rejections"]["seed_data_auto_math_text"] == 1
    assert result["accepted"] == 2
    manifest = json.loads((Path(result["output_dir"]) / gc.MANIFEST_NAME).read_text())
    # The story family share is reported so a later selection stage can cap it at <= 5%.
    assert manifest["diagnostic_shares"]["story_family_share"] == 0.5


def _finephrase_row(
    index: int,
    *,
    finish_reason: str = "stop",
    generated: str | None = None,
    source_text: str | None = None,
) -> dict[str, object]:
    body = (
        generated
        if generated is not None
        else (f"Making sourdough bread, part {index}.\n\n{_prose(index, sentences=20)}")
    )
    return {
        "dataset": "fineweb-edu",
        "dump": "CC-MAIN-2024-10",
        "file_path": "s3://bucket/file.parquet",
        "id": f"fp-{index}",
        "int_score": 3,
        "language": "en",
        "language_score": 0.98,
        "rollout_results": [
            {
                "finish_reason": finish_reason,
                "text": body,
                "usage": {
                    "completion_tokens": 400,
                    "prompt_tokens": 800,
                    "prompt_tokens_details": None,
                    "total_tokens": 1200,
                },
            }
        ],
        "score": 3.1,
        "text": source_text
        if source_text is not None
        else (
            f"ORIGINAL FINEWEB SOURCE DOCUMENT {index} THAT MUST NEVER BE EMITTED. "
            f"{_prose(index + 500, sentences=20)}"
        ),
        "token_count": 900,
        "url": f"https://example.org/article/{index}",
    }


FINEPHRASE_SCHEMA = {
    path: (dtype or "list[null]")
    for path, dtype in gc.SOURCES["finephrase_tutorial"].required_schema
}


def test_finephrase_emits_the_rollout_body_not_the_top_level_source_text(run_dir):
    rows = [_finephrase_row(index) for index in range(3)]
    config = make_config(run_dir, source=gc.SOURCES["finephrase_tutorial"], target_documents=3)
    result = gc.build_candidates(config, scanner=make_scanner(rows, schema=FINEPHRASE_SCHEMA))
    documents = read_documents(Path(result["output_dir"]) / gc.DOCUMENTS_NAME)
    assert len(documents) == 3
    for document, row in zip(documents, rows, strict=True):
        assert document["text"] == row["rollout_results"][0]["text"]
        assert "ORIGINAL FINEWEB SOURCE DOCUMENT" not in document["text"]
        assert document["text"] != row["text"]
        # The source document survives only as a provenance digest.
        assert document["metadata"]["source_text_sha256"] != document["text_sha256"]
    assert gc.SOURCES["finephrase_tutorial"].body_path == "rollout_results.0.text"


def test_finephrase_source_text_confusion_is_rejected_structurally(run_dir):
    identical = f"This paragraph appears in both columns. {_prose(7, sentences=20)}"
    rows = [_finephrase_row(0, generated=identical, source_text=identical)]
    rows.extend(_finephrase_row(index) for index in range(1, 4))
    config = make_config(run_dir, source=gc.SOURCES["finephrase_tutorial"], target_documents=3)
    result = gc.build_candidates(config, scanner=make_scanner(rows, schema=FINEPHRASE_SCHEMA))
    assert result["rejections"]["source_text_confusion"] == 1


def test_finephrase_rejection_counters_and_declarative_diagnostic(run_dir):
    length_stop = _finephrase_row(0, finish_reason="length")
    short = _finephrase_row(1, generated="Too short to be a tutorial body.")
    frame = _finephrase_row(
        2,
        generated=(
            "Here is the rewritten document as a step-by-step tutorial:\n\n"
            f"{_prose(3, sentences=20)}"
        ),
    )
    outline = _finephrase_row(
        3,
        generated="# Introduction\n## Materials\n## Method\n## Results\n## Summary\n" * 12,
    )
    worksheet = _finephrase_row(
        4,
        generated=(
            "Fraction practice worksheet. Solve for x in each expression.\n\n"
            f"{_prose(4, sentences=20)}"
        ),
    )
    declarative = _finephrase_row(
        5,
        generated=(
            "Facts about the solar system.\n\n"
            "1. The planet Jupiter is the largest in the solar system by a wide margin.\n"
            "2. Their orbital periods vary considerably between the inner and outer planets.\n"
            "3. Many moons were discovered only after the invention of the modern telescope.\n"
            "4. These bodies are studied continuously by observatories around the world.\n"
            f"{_prose(5, sentences=20)}"
        ),
    )
    rows = [length_stop, short, frame, outline, worksheet, declarative]
    config = make_config(run_dir, source=gc.SOURCES["finephrase_tutorial"], target_documents=1)
    result = gc.build_candidates(config, scanner=make_scanner(rows, schema=FINEPHRASE_SCHEMA))
    assert result["rejections"]["finish_reason_not_stop"] == 1
    assert result["rejections"]["below_min_bytes"] == 1
    assert result["rejections"]["transformation_frame"] == 1
    assert result["rejections"]["outline_only"] == 1
    assert result["rejections"]["math_worksheet"] == 1
    assert result["accepted"] == 1
    documents = read_documents(Path(result["output_dir"]) / gc.DOCUMENTS_NAME)
    # A single diagnostic boolean, never a filter and never a new classifier.
    assert documents[0]["metadata"]["declarative_numbered_list"] is True


# --------------------------------------------------------------------------------------
# Bounded scanning: caps and exhaustion
# --------------------------------------------------------------------------------------


def test_source_exhaustion_stops_cleanly_below_target(run_dir):
    rows = [_finewiki_row(i) for i in range(3)]
    config = make_config(run_dir, target_documents=10, max_scanned=50)
    result = gc.build_candidates(config, scanner=make_scanner(rows))
    assert result["stop_reason"] == "source_exhausted"
    assert result["accepted"] == 3
    manifest = json.loads((Path(result["output_dir"]) / gc.MANIFEST_NAME).read_text())
    assert manifest["exhausted"] is True


def test_scan_cap_stops_the_build(run_dir):
    rows = [_finewiki_row(i, text="too short") for i in range(50)]
    config = make_config(run_dir, target_documents=5, max_scanned=12)
    result = gc.build_candidates(config, scanner=make_scanner(rows))
    assert result["stop_reason"] == "scan_cap"
    assert result["scanned"] == 12


def test_byte_cap_stops_the_build(run_dir):
    rows = [_finewiki_row(i) for i in range(50)]
    config = make_config(run_dir, target_documents=40, max_scanned=100, max_response_bytes=2048)
    result = gc.build_candidates(config, scanner=make_scanner(rows, page_bytes=1024, page_rows=5))
    assert result["stop_reason"] == "byte_cap"
    assert result["response_bytes"] > 2048


def test_time_cap_stops_the_build(run_dir):
    rows = [_finewiki_row(i) for i in range(50)]
    ticks = iter(range(0, 10_000))
    config = make_config(run_dir, target_documents=40, max_scanned=100, max_wall_seconds=3.0)
    result = gc.build_candidates(
        config, scanner=make_scanner(rows), clock=lambda: float(next(ticks))
    )
    assert result["stop_reason"] == "time_cap"


# --------------------------------------------------------------------------------------
# Checkpoint, interruption, resume
# --------------------------------------------------------------------------------------


def test_interruption_before_checkpoint_loses_no_committed_work(run_dir):
    rows = [_finewiki_row(i) for i in range(20)]
    config = make_config(run_dir, target_documents=10, checkpoint_every=4)
    with pytest.raises(gc.PlannedInterruption):
        gc.build_candidates(config, scanner=make_scanner(rows, interrupt_at=7))
    checkpoint = json.loads((Path(config.work_dir) / gc.CHECKPOINT_NAME).read_text())
    assert checkpoint["counters"]["accepted"] == 4
    assert checkpoint["next_row_index"] == 4

    result = gc.build_candidates(config, scanner=make_scanner(rows))
    assert result["published"] is True
    assert result["resume_count"] == 1
    documents = read_documents(Path(result["output_dir"]) / gc.DOCUMENTS_NAME)
    assert len(documents) == 10
    assert len({doc["source_record_id"] for doc in documents}) == 10
    assert [doc["row_index"] for doc in documents] == list(range(10))


def test_interruption_after_checkpoint_resumes_from_the_committed_cursor(run_dir):
    rows = [_finewiki_row(i) for i in range(20)]
    config = make_config(run_dir, target_documents=12, checkpoint_every=2)
    with pytest.raises(gc.PlannedInterruption):
        gc.build_candidates(config, scanner=make_scanner(rows, interrupt_at=6))
    checkpoint = json.loads((Path(config.work_dir) / gc.CHECKPOINT_NAME).read_text())
    assert checkpoint["counters"]["accepted"] == 6
    committed = (Path(config.work_dir) / gc.DOCUMENTS_NAME).read_bytes()
    assert len(committed) == checkpoint["documents_bytes"]

    result = gc.build_candidates(config, scanner=make_scanner(rows))
    documents = read_documents(Path(result["output_dir"]) / gc.DOCUMENTS_NAME)
    assert len(documents) == 12
    assert [doc["row_index"] for doc in documents] == list(range(12))


def test_resume_preserves_the_first_prefix_byte_for_byte(run_dir):
    """The Phase 4 pilot shape: stop at N accepted, resume to the full target."""
    rows = [_finewiki_row(i) for i in range(40)]
    partial = make_config(run_dir, target_documents=16, stop_after_documents=6)
    first = gc.build_candidates(partial, scanner=make_scanner(rows))
    assert first["published"] is False
    assert first["accepted"] == 6
    prefix = (Path(partial.work_dir) / gc.DOCUMENTS_NAME).read_bytes()

    full = make_config(run_dir, target_documents=16)
    second = gc.build_candidates(full, scanner=make_scanner(rows))
    assert second["published"] is True
    assert second["accepted"] == 16
    assert second["resume_count"] == 1
    published = (Path(second["output_dir"]) / gc.DOCUMENTS_NAME).read_bytes()
    assert published.startswith(prefix)
    documents = read_documents(Path(second["output_dir"]) / gc.DOCUMENTS_NAME)
    assert len({doc["source_record_id"] for doc in documents}) == 16


def test_corrupted_checkpoint_is_a_hard_error(run_dir):
    rows = [_finewiki_row(i) for i in range(20)]
    config = make_config(run_dir, target_documents=10, stop_after_documents=4)
    gc.build_candidates(config, scanner=make_scanner(rows))
    checkpoint_path = Path(config.work_dir) / gc.CHECKPOINT_NAME
    state = json.loads(checkpoint_path.read_text())
    state["counters"]["accepted"] = 999
    checkpoint_path.write_text(json.dumps(state))
    with pytest.raises(gc.GateCError, match="checksum mismatch"):
        gc.build_candidates(make_config(run_dir, target_documents=10), scanner=make_scanner(rows))


def test_truncated_checkpoint_json_is_a_hard_error(run_dir):
    rows = [_finewiki_row(i) for i in range(20)]
    config = make_config(run_dir, target_documents=10, stop_after_documents=4)
    gc.build_candidates(config, scanner=make_scanner(rows))
    checkpoint_path = Path(config.work_dir) / gc.CHECKPOINT_NAME
    checkpoint_path.write_bytes(checkpoint_path.read_bytes()[:40])
    with pytest.raises(gc.GateCError, match="corrupted"):
        gc.build_candidates(make_config(run_dir, target_documents=10), scanner=make_scanner(rows))


def test_fingerprint_mismatch_refuses_to_resume(run_dir):
    rows = [_finewiki_row(i) for i in range(20)]
    config = make_config(run_dir, target_documents=10, stop_after_documents=4)
    gc.build_candidates(config, scanner=make_scanner(rows))
    # A different semantic target is a different run and must not silently reuse the checkpoint.
    with pytest.raises(gc.GateCError, match="fingerprint mismatch"):
        gc.build_candidates(make_config(run_dir, target_documents=11), scanner=make_scanner(rows))


def test_tampered_document_prefix_is_detected_on_resume(run_dir):
    rows = [_finewiki_row(i) for i in range(20)]
    config = make_config(run_dir, target_documents=10, stop_after_documents=4)
    gc.build_candidates(config, scanner=make_scanner(rows))
    documents_path = Path(config.work_dir) / gc.DOCUMENTS_NAME
    raw = bytearray(documents_path.read_bytes())
    raw[10] = raw[10] ^ 0x20
    documents_path.write_bytes(bytes(raw))
    with pytest.raises(gc.GateCError, match="does not match the committed checkpoint hash"):
        gc.build_candidates(make_config(run_dir, target_documents=10), scanner=make_scanner(rows))


def test_checkpoint_cadence_is_not_part_of_the_fingerprint(run_dir):
    rows = [_finewiki_row(i) for i in range(20)]
    first = make_config(run_dir, target_documents=8, stop_after_documents=4, checkpoint_every=2)
    gc.build_candidates(first, scanner=make_scanner(rows))
    second = make_config(run_dir, target_documents=8, checkpoint_every=7)
    result = gc.build_candidates(second, scanner=make_scanner(rows))
    assert result["accepted"] == 8
    assert result["resume_count"] == 1


# --------------------------------------------------------------------------------------
# Publication
# --------------------------------------------------------------------------------------


def test_atomic_publication_leaves_no_staging_directory(run_dir):
    rows = [_finewiki_row(i) for i in range(10)]
    config = make_config(run_dir, target_documents=5)
    result = gc.build_candidates(config, scanner=make_scanner(rows))
    output_dir = Path(result["output_dir"])
    assert output_dir.is_dir()
    assert not gc._staging_path(output_dir).exists()
    assert sorted(item.name for item in output_dir.iterdir()) == [
        gc.CHECKSUMS_NAME,
        gc.DOCUMENTS_NAME,
        gc.MANIFEST_NAME,
    ]


def test_refuses_to_overwrite_an_existing_final_output(run_dir):
    rows = [_finewiki_row(i) for i in range(10)]
    config = make_config(run_dir, target_documents=5)
    gc.build_candidates(config, scanner=make_scanner(rows))
    with pytest.raises(gc.GateCError, match="refusing to overwrite published output"):
        gc.build_candidates(
            make_config(run_dir, target_documents=5, work_dir=run_dir / "work2"),
            scanner=make_scanner(rows),
        )


def test_refuses_to_write_outside_a_git_ignored_path(tmp_path):
    config = make_config(tmp_path, output_dir=PROJECT_ROOT / "pretrain" / "_gate_c_release")
    with pytest.raises(gc.GateCError, match="not Git-ignored"):
        gc.build_candidates(config, scanner=make_scanner([_finewiki_row(0)]))


def test_manifest_accounting_matches_the_published_records(run_dir):
    rows = [_finewiki_row(i) for i in range(12)]
    rows[2]["text"] = "too short"
    config = make_config(run_dir, target_documents=6)
    result = gc.build_candidates(config, scanner=make_scanner(rows))
    output_dir = Path(result["output_dir"])
    manifest = json.loads((output_dir / gc.MANIFEST_NAME).read_text())
    documents = read_documents(output_dir / gc.DOCUMENTS_NAME)
    assert manifest["accounting"]["accepted"] == len(documents) == 6
    assert manifest["accounting"]["rejected"] == 1
    assert manifest["accounting"]["scanned"] == 7
    assert manifest["accounting"]["accepted_text_bytes"] == sum(
        doc["text_bytes"] for doc in documents
    )
    assert manifest["yield_rate"] == pytest.approx(6 / 7)
    assert manifest["source"]["revision"] == gc.SOURCES["finewiki_en"].revision
    assert manifest["live_schema"] == FINEWIKI_SCHEMA
    assert manifest["hard_stops"] == {
        "bulk_candidate_quota_started": False,
        "tokenizer_trained": False,
        "gate_r_started": False,
        "final_shards_built": False,
        "model_training_started": False,
    }
    assert gc.verify_release(output_dir)["accepted"] == 6


def test_candidate_sha_tampering_is_detected_by_verify(run_dir):
    rows = [_finewiki_row(i) for i in range(10)]
    config = make_config(run_dir, target_documents=4)
    result = gc.build_candidates(config, scanner=make_scanner(rows))
    output_dir = Path(result["output_dir"])
    documents = read_documents(output_dir / gc.DOCUMENTS_NAME)
    documents[0]["text"] = documents[0]["text"] + " tampered"
    (output_dir / gc.DOCUMENTS_NAME).write_bytes(
        b"".join(gc._canonical_json_bytes(doc) + b"\n" for doc in documents)
    )
    with pytest.raises(gc.GateCError, match="documents.jsonl does not match"):
        gc.verify_release(output_dir)


def test_verify_detects_a_record_level_sha_forgery(run_dir):
    rows = [_finewiki_row(i) for i in range(10)]
    config = make_config(run_dir, target_documents=4)
    result = gc.build_candidates(config, scanner=make_scanner(rows))
    output_dir = Path(result["output_dir"])
    documents = read_documents(output_dir / gc.DOCUMENTS_NAME)
    documents[0]["text"] = documents[0]["text"] + " tampered"
    payload = b"".join(gc._canonical_json_bytes(doc) + b"\n" for doc in documents)
    (output_dir / gc.DOCUMENTS_NAME).write_bytes(payload)
    manifest = json.loads((output_dir / gc.MANIFEST_NAME).read_text())
    manifest["documents_sha256"] = gc._sha256_bytes(payload)
    manifest["documents_bytes"] = len(payload)
    (output_dir / gc.MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    with pytest.raises(gc.GateCError, match="text_sha256 was tampered with"):
        gc.verify_release(output_dir)


def test_records_carry_provenance_and_no_token_or_separator_artifacts(run_dir):
    rows = [_finewiki_row(i) for i in range(6)]
    config = make_config(run_dir, target_documents=3)
    result = gc.build_candidates(config, scanner=make_scanner(rows))
    for document in read_documents(Path(result["output_dir"]) / gc.DOCUMENTS_NAME):
        assert document["provenance"] == {
            "dataset": "HuggingFaceFW/finewiki",
            "config": "en",
            "split": "train",
            "revision": gc.SOURCES["finewiki_en"].revision,
            "license": "cc-by-sa-4.0",
            "transport": "huggingface_dataset_server_rows",
        }
        assert set(document) == {
            "source_key",
            "source_record_id",
            "natural_id",
            "text",
            "text_sha256",
            "text_bytes",
            "row_index",
            "metadata",
            "provenance",
        }
        assert "token_ids" not in document and "tokens" not in document
        assert not document["text"].startswith("[BOS]")
        assert not document["text"].endswith("[EOS]")


def test_finewiki_never_reads_wikitext_or_html(run_dir):
    rows = [_finewiki_row(i) for i in range(4)]
    for row in rows:
        row["wikitext"] = "== MUST NOT BE EMITTED ==\n'''bold''' [[link]]"
    config = make_config(run_dir, target_documents=3)
    result = gc.build_candidates(config, scanner=make_scanner(rows))
    for document in read_documents(Path(result["output_dir"]) / gc.DOCUMENTS_NAME):
        assert "MUST NOT BE EMITTED" not in document["text"]
    assert gc.SOURCES["finewiki_en"].body_path == "text"


def test_fineweb_retains_score_metadata_without_freezing_a_threshold(run_dir):
    fineweb_schema = {
        path: dtype for path, dtype in gc.SOURCES["fineweb_edu_dedup"].required_schema
    }
    rows = []
    for index, score in enumerate((2, 3, 4, 5)):
        rows.append({
            "id": f"fw-{index}",
            "metadata": {
                "date": "2024-01-01T00:00:00",
                "dump": "CC-MAIN-2024-10",
                "file_path": "s3://bucket/x.parquet",
                "int_score": score,
                "language": "en",
                "language_score": 0.97,
                "score": score + 0.3,
                "token_count": 700,
                "url": f"https://example.org/{index}",
            },
            "text": f"Article {index}. {_prose(index)}",
        })
    config = make_config(run_dir, source=gc.SOURCES["fineweb_edu_dedup"], target_documents=4)
    result = gc.build_candidates(config, scanner=make_scanner(rows, schema=fineweb_schema))
    # No educational-score threshold is frozen at C0: every score survives, metadata is retained.
    assert result["accepted"] == 4
    documents = read_documents(Path(result["output_dir"]) / gc.DOCUMENTS_NAME)
    assert sorted(doc["metadata"]["metadata.int_score"] for doc in documents) == [2, 3, 4, 5]


def test_deterministic_ordering_is_reproducible(run_dir):
    rows = [_finewiki_row(i) for i in range(20)]
    first = gc.build_candidates(
        make_config(run_dir, target_documents=8, output_dir=run_dir / "a", work_dir=run_dir / "wa"),
        scanner=make_scanner(copy.deepcopy(rows)),
    )
    second = gc.build_candidates(
        make_config(run_dir, target_documents=8, output_dir=run_dir / "b", work_dir=run_dir / "wb"),
        scanner=make_scanner(copy.deepcopy(rows)),
    )
    assert first["documents_sha256"] == second["documents_sha256"]


def test_scanner_may_not_rewind_before_the_checkpointed_cursor(run_dir):
    rows = [_finewiki_row(i) for i in range(20)]
    config = make_config(run_dir, target_documents=10, stop_after_documents=4)
    gc.build_candidates(config, scanner=make_scanner(rows))

    def rewinding_scanner(source, *, start_row):
        del start_row
        yield from make_scanner(rows)(source, start_row=0)

    with pytest.raises(gc.GateCError, match="before the checkpointed cursor"):
        gc.build_candidates(make_config(run_dir, target_documents=10), scanner=rewinding_scanner)
