from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

import pretrain.corpus_gate_e as corpus_gate_e
from pretrain.corpus_gate_e import (
    GateEError,
    InspectionConfig,
    LoadedSource,
    PlannedInterruption,
    SmokeConfig,
    SourceSpec,
    build_candidate_smoke,
    inspect_corpus,
)

REVISION = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(autouse=True)
def _stub_external_checks(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(corpus_gate_e, "_git_head", lambda: REVISION)
    monkeypatch.setattr(corpus_gate_e, "_ensure_git_ignored", lambda _path: None)


def _source(*, python_gate: bool = False) -> SourceSpec:
    required = [("text", "string"), ("score", "float64")]
    if python_gate:
        required.extend([
            ("metadata.is_generated", "bool"),
            ("metadata.is_vendor", "bool"),
            ("metadata.language", "string"),
            ("metadata.length_bytes", "int64"),
            ("metadata.path", "string"),
            ("metadata.src_encoding", "string"),
        ])
    return SourceSpec(
        key="synthetic_python" if python_gate else "synthetic_web",
        dataset="owner/synthetic",
        dataset_config="default",
        split="train",
        revision=REVISION,
        inspection_cap=40,
        required_schema=tuple(sorted(required)),
        quality_field="score",
        minimum_quality=3,
        language_field="metadata.language" if python_gate else None,
        language_value="Python" if python_gate else None,
        minimum_bytes=1,
        maximum_bytes=100_000,
        python_gate=python_gate,
        volume_rows=1_000,
        volume_rows_basis="synthetic exact",
        card_license_fact="synthetic-fact",
    )


def _schema(source: SourceSpec) -> dict[str, str]:
    return dict(source.required_schema)


def _loader(rows: list[dict], source: SourceSpec, *, revision: str = REVISION):
    def load(_source: SourceSpec, _cache_dir: Path, _documents: int) -> LoadedSource:
        assert _source == source
        return LoadedSource(
            rows=list(rows),
            resolved_revision=revision,
            live_schema=_schema(source),
            transport="synthetic_stream",
        )

    return load


class StepClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.25
        return self.value


def _inspection_config(tmp_path: Path, source: SourceSpec, documents: int) -> InspectionConfig:
    return InspectionConfig(
        source=source,
        output_dir=tmp_path / "inspection",
        cache_dir=tmp_path / "cache",
        documents=documents,
        accepted_snippets=20,
        rejected_snippets=10,
        snippet_characters=120,
    )


def test_bounded_inspection_publishes_compact_report_and_review_artifact(tmp_path: Path):
    source = _source()
    rows = [
        {"text": f"Useful educational document {index}.\nSecond line.", "score": 4.0}
        for index in range(28)
    ]
    rows.extend({"text": f"Low score document {index}.", "score": 2.0} for index in range(12))
    report = inspect_corpus(
        _inspection_config(tmp_path, source, len(rows)),
        loader=_loader(rows, source),
        clock=StepClock(),
    )

    output = tmp_path / "inspection"
    assert report["gate_e"]["status"] == "PASS"
    assert report["access"]["documents_inspected"] == 40
    assert report["basic_filter"]["accepted"] == 28
    assert report["basic_filter"]["rejected"] == 12
    assert report["basic_filter"]["common_rejection_causes"] == {"quality_below_minimum": 12}
    assert report["review_artifact"]["accepted_written"] == 20
    assert report["review_artifact"]["rejected_written"] == 10
    review_bytes = (output / "review_snippets.jsonl").read_bytes()
    assert hashlib.sha256(review_bytes).hexdigest() == report["review_artifact"]["sha256"]
    review_rows = [json.loads(line) for line in review_bytes.splitlines()]
    assert sum(row["bucket"] == "accepted" for row in review_rows) == 20
    assert sum(row["bucket"] == "rejected" for row in review_rows) == 10
    assert all(len(row["snippet"]) <= 120 for row in review_rows)
    persisted = json.loads((output / "report.json").read_text())
    assert persisted == report
    serialized = json.dumps(persisted).casefold()
    for forbidden in ("wilson", "sealed attestation", "blind review", "p1b"):
        assert forbidden not in serialized


def test_python_gate_counts_ast_generated_vendor_length_and_repetition(tmp_path: Path):
    source = _source(python_gate=True)

    def row(text: str, **metadata_overrides):
        metadata = {
            "language": "Python",
            "length_bytes": len(text.encode()),
            "is_generated": False,
            "is_vendor": False,
            "path": "src/module.py",
            "src_encoding": "UTF-8",
        }
        metadata.update(metadata_overrides)
        return {"text": text, "score": 4.0, "metadata": metadata}

    good = [row(f"def useful_{index}(value):\n    return value + {index}\n") for index in range(22)]
    rejected = [
        row("def broken(:\n    pass\n"),
        row("# generated by a tool\nvalue = 1\n"),
        row("value = 1\n", is_vendor=True),
        row("value = 1\n", language="Java"),
        row("value = 1\n", length_bytes=999),
        row("value = 1\n" * 25),
    ]
    low_quality = [row(f"value_{index} = 1\n") for index in range(5)]
    for item in low_quality:
        item["score"] = 2.0
    rows = good + rejected + low_quality
    report = inspect_corpus(
        _inspection_config(tmp_path, source, len(rows)),
        loader=_loader(rows, source),
        clock=StepClock(),
    )
    causes = report["basic_filter"]["common_rejection_causes"]
    assert report["gate_e"]["status"] == "PASS"
    assert causes["python_ast_parse_failed"] == 1
    assert causes["generated_marker"] == 1
    assert causes["vendor_metadata_or_path"] == 1
    assert causes["language_mismatch"] == 1
    assert "metadata_length_mismatch" not in causes
    assert report["basic_filter"]["diagnostic_counts_not_used_as_rejections"] == {
        "metadata_length_mismatch": 1
    }
    assert causes["pathological_repetition"] == 1


@pytest.mark.parametrize("failure", ["revision", "schema"])
def test_revision_or_schema_drift_fails_before_publication(tmp_path: Path, failure: str):
    source = _source()
    rows = [{"text": "healthy", "score": 4.0}] * 20
    if failure == "revision":
        loader = _loader(rows, source, revision="f" * 40)
    else:
        loader = lambda _source, _cache, _documents: LoadedSource(  # noqa: E731
            rows=rows,
            resolved_revision=REVISION,
            live_schema={"text": "string"},
            transport="synthetic_stream",
        )
    with pytest.raises(GateEError):
        inspect_corpus(
            _inspection_config(tmp_path, source, 20),
            loader=loader,
            clock=StepClock(),
        )
    assert not (tmp_path / "inspection").exists()


def _write_passing_report(path: Path, source: SourceSpec) -> None:
    rejected_count = 10
    rows = [
        {"text": f"Useful candidate inspection document {index}.", "score": 4.0}
        for index in range(source.inspection_cap - rejected_count)
    ]
    rows.extend(
        {"text": f"Rejected candidate inspection document {index}.", "score": 2.0}
        for index in range(rejected_count)
    )
    report = inspect_corpus(
        InspectionConfig(
            source=source,
            output_dir=path.parent,
            cache_dir=path.parent.parent / "inspection_cache",
            documents=len(rows),
            accepted_snippets=20,
            rejected_snippets=10,
            snippet_characters=120,
        ),
        loader=_loader(rows, source),
        clock=StepClock(),
    )
    assert report["gate_e"]["status"] == "PASS"
    assert path.is_file()


def _smoke_config(
    tmp_path: Path,
    source: SourceSpec,
    *,
    target: int = 8,
    stop_after: int | None = None,
) -> SmokeConfig:
    report = tmp_path / "inspection" / "report.json"
    if not report.exists():
        _write_passing_report(report, source)
    return SmokeConfig(
        source=source,
        inspection_report=report,
        output_dir=tmp_path / "candidate_smoke",
        cache_dir=tmp_path / "cache",
        target_documents=target,
        max_scanned=20,
        stop_after_documents=stop_after,
    )


def test_candidate_smoke_resumes_without_gap_or_duplicate_and_publishes_atomically(
    tmp_path: Path,
):
    source = _source()
    rows = [{"text": f"Candidate document {index}.", "score": 4.0} for index in range(20)]
    with pytest.raises(PlannedInterruption):
        build_candidate_smoke(
            _smoke_config(tmp_path, source, stop_after=3),
            loader=_loader(rows, source),
            clock=StepClock(),
        )
    output = tmp_path / "candidate_smoke"
    staging = tmp_path / ".candidate_smoke.building"
    assert not output.exists()
    assert staging.is_dir()
    assert (staging / "state.json").is_file()

    manifest = build_candidate_smoke(
        _smoke_config(tmp_path, source),
        loader=_loader(rows, source),
        clock=StepClock(),
    )
    assert output.is_dir()
    assert not staging.exists()
    assert manifest["status"] == "SMOKE_PASS"
    assert manifest["accounting"]["accepted"] == 8
    assert manifest["accounting"]["resume_count"] == 1
    records = [json.loads(line) for line in (output / "candidate.jsonl").read_bytes().splitlines()]
    assert [record["_petitgpt_candidate"]["source_row_index"] for record in records] == list(
        range(8)
    )
    assert len({record["_petitgpt_candidate"]["text_sha256"] for record in records}) == 8
    data = (output / "candidate.jsonl").read_bytes()
    assert hashlib.sha256(data).hexdigest() == manifest["output"]["sha256"]
    assert manifest["output"]["bytes"] == len(data)
    assert all(value is False for value in manifest["production_actions"].values())


def test_candidate_resume_rejects_config_drift_without_changing_checkpoint(tmp_path: Path):
    source = _source()
    rows = [{"text": f"Candidate {index}.", "score": 4.0} for index in range(20)]
    with pytest.raises(PlannedInterruption):
        build_candidate_smoke(
            _smoke_config(tmp_path, source, stop_after=2),
            loader=_loader(rows, source),
            clock=StepClock(),
        )
    staging = tmp_path / ".candidate_smoke.building"
    before = {path.name: path.read_bytes() for path in staging.iterdir() if path.is_file()}
    with pytest.raises(GateEError, match="fingerprint"):
        build_candidate_smoke(
            _smoke_config(tmp_path, source, target=9),
            loader=_loader(rows, source),
            clock=StepClock(),
        )
    after = {path.name: path.read_bytes() for path in staging.iterdir() if path.is_file()}
    assert before == after


@pytest.mark.parametrize(
    "damage",
    [
        "invalid_json",
        "failed_criterion",
        "review_shortfall",
        "false_snippet_sha256",
        "tampered_snippet_bytes",
    ],
)
def test_candidate_fail_closed_rejects_untrusted_inspection_artifact_before_loading(
    tmp_path: Path,
    damage: str,
):
    source = _source()
    config = _smoke_config(tmp_path, source)
    report_path = config.inspection_report
    snippets_path = report_path.parent / "review_snippets.jsonl"

    if damage == "invalid_json":
        report_path.write_bytes(b"{not-json\n")
    elif damage == "tampered_snippet_bytes":
        snippets_path.write_bytes(snippets_path.read_bytes() + b"{}\n")
    else:
        report = json.loads(report_path.read_text())
        if damage == "failed_criterion":
            report["gate_e"]["criteria"]["required_schema"] = False
        elif damage == "review_shortfall":
            report["review_artifact"]["rejected_written"] -= 1
            report["review_artifact"]["rejected_shortfall"] = 1
        elif damage == "false_snippet_sha256":
            report["review_artifact"]["sha256"] = "0" * 64
        else:  # pragma: no cover - the parameter list is exhaustive
            raise AssertionError(damage)
        report_path.write_text(json.dumps(report, sort_keys=True) + "\n")

    load_calls = 0

    def never_load(_source: SourceSpec, _cache: Path, _documents: int) -> LoadedSource:
        nonlocal load_calls
        load_calls += 1
        raise AssertionError("untrusted inspection artifacts must fail before source access")

    with pytest.raises(GateEError):
        build_candidate_smoke(config, loader=never_load, clock=StepClock())

    assert load_calls == 0
    assert not (tmp_path / "candidate_smoke").exists()
    assert not (tmp_path / ".candidate_smoke.building").exists()


def test_candidate_fail_closed_target_checkpoint_resume_does_not_overproduce(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = _source()
    target = 8
    rows = [{"text": f"Candidate target row {index}.", "score": 4.0} for index in range(20)]
    original_checkpoint = corpus_gate_e._checkpoint_smoke
    interrupted = False

    def interrupt_at_target_checkpoint(*, partial_path, state_path, state):
        nonlocal interrupted
        original_checkpoint(
            partial_path=partial_path,
            state_path=state_path,
            state=state,
        )
        if state["accepted"] == target and not interrupted:
            interrupted = True
            raise PlannedInterruption("simulated crash after the target checkpoint")

    monkeypatch.setattr(corpus_gate_e, "_checkpoint_smoke", interrupt_at_target_checkpoint)
    with pytest.raises(PlannedInterruption):
        build_candidate_smoke(
            _smoke_config(tmp_path, source, target=target),
            loader=_loader(rows, source),
            clock=StepClock(),
        )

    staging = tmp_path / ".candidate_smoke.building"
    checkpoint_bytes = (staging / "candidate.jsonl.partial").read_bytes()
    checkpoint_records = [json.loads(line) for line in checkpoint_bytes.splitlines()]
    assert len(checkpoint_records) == target

    monkeypatch.setattr(corpus_gate_e, "_checkpoint_smoke", original_checkpoint)
    manifest = build_candidate_smoke(
        _smoke_config(tmp_path, source, target=target),
        loader=_loader(rows, source),
        clock=StepClock(),
    )

    published_bytes = (tmp_path / "candidate_smoke" / "candidate.jsonl").read_bytes()
    assert published_bytes == checkpoint_bytes
    assert manifest["accounting"]["accepted"] == target
    assert manifest["output"]["documents"] == target
    assert [
        record["_petitgpt_candidate"]["source_row_index"] for record in checkpoint_records
    ] == list(range(target))


@pytest.mark.parametrize("failure", ["revision", "schema"])
def test_candidate_fail_closed_source_validation_preserves_checkpoint_bytes(
    tmp_path: Path,
    failure: str,
):
    source = _source()
    rows = [{"text": f"Candidate validation row {index}.", "score": 4.0} for index in range(20)]
    with pytest.raises(PlannedInterruption):
        build_candidate_smoke(
            _smoke_config(tmp_path, source, stop_after=2),
            loader=_loader(rows, source),
            clock=StepClock(),
        )

    staging = tmp_path / ".candidate_smoke.building"
    state_path = staging / "state.json"
    partial_path = staging / "candidate.jsonl.partial"
    before_state = state_path.read_bytes()
    before_partial = partial_path.read_bytes()
    if failure == "revision":
        bad_loader = _loader(rows, source, revision="f" * 40)
    else:
        bad_loader = lambda _source, _cache, _documents: LoadedSource(  # noqa: E731
            rows=rows,
            resolved_revision=REVISION,
            live_schema={"text": "string"},
            transport="synthetic_stream",
        )

    with pytest.raises(GateEError, match=failure):
        build_candidate_smoke(
            _smoke_config(tmp_path, source),
            loader=bad_loader,
            clock=StepClock(),
        )

    assert state_path.read_bytes() == before_state
    assert partial_path.read_bytes() == before_partial


def test_candidate_fail_closed_resumes_after_data_finalization_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = _source()
    target = 8
    rows = [{"text": f"Candidate finalization row {index}.", "score": 4.0} for index in range(20)]
    original_replace = corpus_gate_e.os.replace
    interrupted = False

    def replace_then_interrupt(source_path, destination_path):
        nonlocal interrupted
        original_replace(source_path, destination_path)
        if (
            Path(source_path).name == "candidate.jsonl.partial"
            and Path(destination_path).name == "candidate.jsonl"
            and not interrupted
        ):
            interrupted = True
            raise PlannedInterruption("simulated crash after final data rename")

    monkeypatch.setattr(corpus_gate_e.os, "replace", replace_then_interrupt)
    with pytest.raises(PlannedInterruption):
        build_candidate_smoke(
            _smoke_config(tmp_path, source, target=target),
            loader=_loader(rows, source),
            clock=StepClock(),
        )

    staging = tmp_path / ".candidate_smoke.building"
    finalized_bytes = (staging / "candidate.jsonl").read_bytes()
    assert not (staging / "candidate.jsonl.partial").exists()

    monkeypatch.setattr(corpus_gate_e.os, "replace", original_replace)
    manifest = build_candidate_smoke(
        _smoke_config(tmp_path, source, target=target),
        loader=_loader(rows, source),
        clock=StepClock(),
    )

    assert (tmp_path / "candidate_smoke" / "candidate.jsonl").read_bytes() == finalized_bytes
    assert manifest["accounting"]["accepted"] == target
    assert manifest["output"]["documents"] == target


def test_candidate_fail_closed_retry_after_published_interruption_is_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = _source()
    rows = [{"text": f"Candidate published row {index}.", "score": 4.0} for index in range(20)]
    config = _smoke_config(tmp_path, source)
    original_publish = corpus_gate_e._publish_directory

    def publish_then_interrupt(staging: Path, destination: Path) -> None:
        original_publish(staging, destination)
        raise PlannedInterruption("simulated crash after atomic publication")

    monkeypatch.setattr(corpus_gate_e, "_publish_directory", publish_then_interrupt)
    with pytest.raises(PlannedInterruption):
        build_candidate_smoke(
            config,
            loader=_loader(rows, source),
            clock=StepClock(),
        )

    output = tmp_path / "candidate_smoke"
    before = {
        path.relative_to(output): (
            path.read_bytes(),
            path.stat().st_ino,
            path.stat().st_mtime_ns,
        )
        for path in output.iterdir()
        if path.is_file()
    }

    monkeypatch.setattr(corpus_gate_e, "_publish_directory", original_publish)

    def never_load(_source: SourceSpec, _cache: Path, _documents: int) -> LoadedSource:
        raise AssertionError("a verified published release must be recovered without reloading")

    manifest = build_candidate_smoke(
        config,
        loader=never_load,
        clock=StepClock(),
    )
    after = {
        path.relative_to(output): (
            path.read_bytes(),
            path.stat().st_ino,
            path.stat().st_mtime_ns,
        )
        for path in output.iterdir()
        if path.is_file()
    }
    assert after == before
    assert manifest == json.loads((output / "manifest.json").read_text())


def test_output_and_cap_contracts_fail_before_loading(tmp_path: Path):
    source = _source()
    calls = 0

    def never_load(_source: SourceSpec, _cache: Path, _documents: int) -> LoadedSource:
        nonlocal calls
        calls += 1
        raise AssertionError("loader must not be called")

    config = _inspection_config(tmp_path, source, source.inspection_cap + 1)
    with pytest.raises(GateEError, match="documents"):
        inspect_corpus(config, loader=never_load)
    assert calls == 0

    existing = tmp_path / "candidate_smoke"
    existing.mkdir()
    with pytest.raises(GateEError, match="overwrite"):
        build_candidate_smoke(
            _smoke_config(tmp_path, source),
            loader=never_load,
        )
    assert calls == 0


def test_flatten_features_preserves_sequence_element_type_and_timestamp_unit():
    from datasets import Features, Sequence, Value

    from pretrain.corpus_gate_e import _flatten_features

    features = Features({
        "created": Value("timestamp[us]"),
        "metadata": {
            "detected_licenses": Sequence(Value("string")),
        },
    })

    assert _flatten_features(features) == {
        "created": "timestamp[us]",
        "metadata.detected_licenses": "list[string]",
    }


def test_review_snippet_shortfall_blocks_gate_e(tmp_path: Path):
    source = _source()
    rows = [{"text": f"Useful educational document {index}.", "score": 4.0} for index in range(30)]

    report = inspect_corpus(
        _inspection_config(tmp_path, source, len(rows)),
        loader=_loader(rows, source),
        clock=StepClock(),
    )

    assert report["basic_filter"]["accepted"] == 30
    assert report["basic_filter"]["rejected"] == 0
    assert report["review_artifact"]["accepted_written"] == 20
    assert report["review_artifact"]["rejected_written"] == 0
    assert report["review_artifact"]["rejected_shortfall"] == 10
    assert report["gate_e"]["status"] == "BLOCKED"


def test_transparent_review_frame_does_not_change_basic_filter_counts(tmp_path: Path):
    from dataclasses import replace

    source = replace(
        _source(),
        review_minimum_quality=4.0,
        review_minimum_bytes=20,
        review_maximum_bytes=100,
        review_scope="synthetic_quality_length_contrast",
    )
    rows = [{"text": f"{'x' * 48}{index:02d}", "score": 4.0} for index in range(24)]
    rows.extend({"text": f"{'q' * 48}{index:02d}", "score": 3.0} for index in range(4))
    rows.extend({"text": f"short-{index}", "score": 4.0} for index in range(4))
    rows.extend({"text": f"{'z' * 118}{index:02d}", "score": 4.0} for index in range(4))
    config = replace(
        _inspection_config(tmp_path, source, len(rows)),
        rejected_snippets=12,
    )

    report = inspect_corpus(
        config,
        loader=_loader(rows, source),
        clock=StepClock(),
    )

    assert report["basic_filter"]["accepted"] == 36
    assert report["basic_filter"]["rejected"] == 0
    assert report["review_artifact"]["accepted_written"] == 20
    assert report["review_artifact"]["rejected_written"] == 12
    assert report["review_artifact"]["accepted_shortfall"] == 0
    assert report["review_artifact"]["rejected_shortfall"] == 0
    assert report["gate_e"]["status"] == "PASS"

    serialized_report = json.dumps(report, sort_keys=True)
    assert "synthetic_quality_length_contrast" in serialized_report
    for field in (
        "review_minimum_quality",
        "review_minimum_bytes",
        "review_maximum_bytes",
    ):
        assert field in serialized_report

    snippets = [
        json.loads(line)
        for line in (tmp_path / "inspection" / "review_snippets.jsonl").read_bytes().splitlines()
    ]
    accepted = [record for record in snippets if record["bucket"] == "accepted"]
    rejected = [record for record in snippets if record["bucket"] == "rejected"]
    assert len(accepted) == 20
    assert len(rejected) == 12
    assert all(record["rejection_causes"] == [] for record in accepted)
    assert all(
        record["rejection_causes"]
        and all(cause.startswith("review_") for cause in record["rejection_causes"])
        for record in rejected
    )
    rejection_causes = {cause for record in rejected for cause in record["rejection_causes"]}
    assert any("quality" in cause for cause in rejection_causes)
    assert any("minimum_bytes" in cause for cause in rejection_causes)
    assert any("maximum_bytes" in cause for cause in rejection_causes)


DATASET_SERVER_FEATURES = [
    {
        "feature_idx": 0,
        "name": "text",
        "type": {"dtype": "string", "_type": "Value"},
    },
    {
        "feature_idx": 1,
        "name": "score",
        "type": {"dtype": "float64", "_type": "Value"},
    },
    {
        "feature_idx": 2,
        "name": "metadata",
        "type": {
            "language": {"dtype": "string", "_type": "Value"},
            "details": {
                "line_count": {"dtype": "int64", "_type": "Value"},
                "licenses": {
                    "feature": {"dtype": "string", "_type": "Value"},
                    "length": -1,
                    "_type": "List",
                },
            },
        },
    },
]


def _dataset_server_source(*, inspection_cap: int = 40) -> SourceSpec:
    return SourceSpec(
        key="synthetic_dataset_server",
        dataset="owner/synthetic-dataset-server",
        dataset_config="nested",
        split="train",
        revision=REVISION,
        inspection_cap=inspection_cap,
        required_schema=tuple(
            sorted((
                ("metadata.details.licenses", "list[string]"),
                ("metadata.details.line_count", "int64"),
                ("metadata.language", "string"),
                ("score", "float64"),
                ("text", "string"),
            ))
        ),
        quality_field="score",
        minimum_quality=3,
        minimum_bytes=1,
        maximum_bytes=100_000,
        volume_rows=20,
        volume_rows_basis="synthetic dataset-server report",
        card_license_fact="synthetic-fact",
    )


class FakeDatasetServerHeaders:
    def __init__(self, revision: str) -> None:
        self._headers = {"x-revision": revision}

    def get(self, name: str, default=None):
        return self._headers.get(name.casefold(), default)


class FakeDatasetServerResponse:
    def __init__(self, body: bytes, revision: str) -> None:
        self.body = body
        self.headers = FakeDatasetServerHeaders(revision)
        self.read_limits: list[int | None] = []

    def read(self, amount: int | None = None) -> bytes:
        self.read_limits.append(amount)
        if amount is None or amount < 0:
            return self.body
        return self.body[:amount]

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> bool:
        return False


class FakeDatasetServer:
    def __init__(
        self,
        *,
        total_rows: int = 20,
        partial: bool = True,
        drift: str | None = None,
    ) -> None:
        self.total_rows = total_rows
        self.partial = partial
        self.drift = drift
        self.calls: list[tuple[int, int]] = []
        self.queries: list[dict[str, list[str]]] = []
        self.response_bytes: list[int] = []

    @staticmethod
    def _row(row_idx: int) -> dict:
        return {
            "row_idx": row_idx,
            "row": {
                "text": f"Bounded synthetic educational row {row_idx}.",
                "score": 4.0,
                "metadata": {
                    "language": "en",
                    "details": {
                        "line_count": 1,
                        "licenses": ["synthetic"],
                    },
                },
            },
            "truncated_cells": [],
        }

    def __call__(self, request, timeout: float = 60.0) -> FakeDatasetServerResponse:
        assert timeout == 60.0
        url = request.full_url if hasattr(request, "full_url") else str(request)
        parsed = urlparse(url)
        assert parsed.path.endswith("/rows")
        query = parse_qs(parsed.query)
        assert query["dataset"] == ["owner/synthetic-dataset-server"]
        assert query["config"] == ["nested"]
        assert query["split"] == ["train"]
        offset = int(query["offset"][0])
        length = int(query["length"][0])
        call_index = len(self.calls)
        self.calls.append((offset, length))
        self.queries.append(query)

        rows = [
            self._row(row_idx) for row_idx in range(offset, min(offset + length, self.total_rows))
        ]
        features = json.loads(json.dumps(DATASET_SERVER_FEATURES))
        revision = REVISION
        if call_index == 1:
            if self.drift == "revision":
                revision = "f" * 40
            elif self.drift == "truncated_cells":
                rows[0]["truncated_cells"] = ["text"]
            elif self.drift == "row_idx":
                rows[0]["row_idx"] += 1
            elif self.drift == "schema":
                features[1]["type"]["dtype"] = "int64"

        body = json.dumps(
            {
                "features": features,
                "rows": rows,
                "num_rows_total": self.total_rows,
                "partial": self.partial,
            },
            sort_keys=True,
        ).encode()
        self.response_bytes.append(len(body))
        return FakeDatasetServerResponse(body, revision)


def _never_sleep(_seconds: float) -> None:
    raise AssertionError("the synthetic success path must not retry or sleep")


def test_dataset_server_loader_is_pinned_bounded_stratified_and_schema_exact(
    tmp_path: Path,
):
    server = FakeDatasetServer(total_rows=20, partial=True)
    source = _dataset_server_source(inspection_cap=5)

    loaded = corpus_gate_e.hf_dataset_server_loader(
        source,
        tmp_path / "cache",
        5,
        opener=server,
        sleeper=_never_sleep,
    )
    rows = list(loaded.rows)

    expected_windows = ((0, 1), (2, 1), (7, 1), (12, 1), (17, 1))
    assert loaded.resolved_revision == REVISION
    assert loaded.live_schema == dict(source.required_schema)
    assert [row["text"] for row in rows] == [
        f"Bounded synthetic educational row {index}." for index in (0, 2, 7, 12, 17)
    ]
    assert len(rows) == 5
    assert sum(length for _, length in server.calls) == 5
    assert tuple(server.calls) == expected_windows
    assert loaded.sample_windows == expected_windows
    assert loaded.request_count == 5
    assert loaded.response_bytes == sum(server.response_bytes)
    assert loaded.reported_rows == 20
    assert loaded.partial is True

    second_server = FakeDatasetServer(total_rows=20, partial=True)
    second = corpus_gate_e.hf_dataset_server_loader(
        source,
        tmp_path / "second-cache",
        5,
        opener=second_server,
        sleeper=_never_sleep,
    )
    assert second.sample_windows == expected_windows
    assert second_server.calls == server.calls


def test_dataset_server_transport_stats_are_exposed_in_inspection_report(tmp_path: Path):
    documents = 30
    server = FakeDatasetServer(total_rows=120, partial=True)
    source = _dataset_server_source(inspection_cap=documents)

    def loader(
        loader_source: SourceSpec,
        cache_dir: Path,
        loader_documents: int,
    ) -> LoadedSource:
        assert loader_documents == documents
        return corpus_gate_e.hf_dataset_server_loader(
            loader_source,
            cache_dir,
            loader_documents,
            opener=server,
            sleeper=_never_sleep,
        )

    report = inspect_corpus(
        InspectionConfig(
            source=source,
            output_dir=tmp_path / "inspection",
            cache_dir=tmp_path / "cache",
            documents=documents,
            accepted_snippets=20,
            rejected_snippets=10,
            snippet_characters=120,
        ),
        loader=loader,
        clock=StepClock(),
    )

    access = report["access"]
    assert access["documents_requested"] == documents
    assert access["documents_inspected"] == documents
    assert access["dataset_server_response_bytes"] == sum(server.response_bytes)
    assert access["dataset_server_request_count"] == len(server.calls)
    assert access["dataset_server_reported_rows"] == 120
    assert access["dataset_server_partial"] is True
    assert access["sample_windows"] == [
        {"offset": offset, "length": length} for offset, length in server.calls
    ]
    assert access["transport_bounds"] == {
        "max_windows": 10,
        "max_response_bytes": 32 * 1024**2,
        "max_total_bytes": 256 * 1024**2,
        "timeout_seconds": 60.0,
        "max_attempts_per_request": 3,
    }


@pytest.mark.parametrize("drift", ["revision", "truncated_cells", "row_idx", "schema"])
def test_dataset_server_loader_fails_closed_on_page_drift(
    tmp_path: Path,
    drift: str,
):
    server = FakeDatasetServer(total_rows=20, drift=drift)

    with pytest.raises(GateEError):
        corpus_gate_e.hf_dataset_server_loader(
            _dataset_server_source(inspection_cap=5),
            tmp_path / "cache",
            5,
            opener=server,
            sleeper=_never_sleep,
        )

    assert len(server.calls) == 2


def test_dataset_server_loader_enforces_response_byte_caps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    assert corpus_gate_e.DATASET_SERVER_MAX_RESPONSE_BYTES == 32 * 1024**2
    assert corpus_gate_e.DATASET_SERVER_MAX_TOTAL_BYTES == 256 * 1024**2

    monkeypatch.setattr(corpus_gate_e, "DATASET_SERVER_MAX_RESPONSE_BYTES", 64)

    def oversized_response(_request, timeout: float = 60.0) -> FakeDatasetServerResponse:
        assert timeout == 60.0
        return FakeDatasetServerResponse(b"x" * 65, REVISION)

    with pytest.raises(GateEError, match="per-call byte cap"):
        corpus_gate_e.hf_dataset_server_loader(
            _dataset_server_source(inspection_cap=1),
            tmp_path / "cache",
            1,
            opener=oversized_response,
            sleeper=_never_sleep,
        )


def test_dataset_server_loader_enforces_total_response_byte_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    server = FakeDatasetServer(total_rows=20)
    monkeypatch.setattr(corpus_gate_e, "DATASET_SERVER_MAX_TOTAL_BYTES", 1)

    with pytest.raises(GateEError, match="total byte cap"):
        corpus_gate_e.hf_dataset_server_loader(
            _dataset_server_source(inspection_cap=5),
            tmp_path / "cache",
            5,
            opener=server,
            sleeper=_never_sleep,
        )


STACKV2_PYTHON_FILTER_WHERE = "\"metadata.language\"='Python'"


def _dataset_server_filter_source(*, inspection_cap: int = 12) -> SourceSpec:
    from dataclasses import replace

    return replace(
        _dataset_server_source(inspection_cap=inspection_cap),
        dataset_server_where=STACKV2_PYTHON_FILTER_WHERE,
        language_field="metadata.language",
        language_value="Python",
    )


class FakeFilteredDatasetServer:
    def __init__(
        self,
        *,
        total_rows: int = 30,
        partial: bool = True,
        drift: str | None = None,
    ) -> None:
        self.total_rows = total_rows
        self.partial = partial
        self.drift = drift
        self.calls: list[tuple[int, int]] = []
        self.where_values: list[str] = []
        self.response_bytes: list[int] = []
        self.last_row_idx: int | None = None

    @staticmethod
    def _original_row_idx(filtered_position: int) -> int:
        return 1_000 + filtered_position * 3 + filtered_position // 4

    @classmethod
    def _row(cls, filtered_position: int) -> dict:
        original_row_idx = cls._original_row_idx(filtered_position)
        record = FakeDatasetServer._row(original_row_idx)
        record["row"]["metadata"]["language"] = "Python"
        return record

    def __call__(self, request, timeout: float = 60.0) -> FakeDatasetServerResponse:
        assert timeout == 60.0
        url = request.full_url if hasattr(request, "full_url") else str(request)
        parsed = urlparse(url)
        assert parsed.path.endswith("/filter")
        query = parse_qs(parsed.query)
        assert query["dataset"] == ["owner/synthetic-dataset-server"]
        assert query["config"] == ["nested"]
        assert query["split"] == ["train"]
        assert query["where"] == [STACKV2_PYTHON_FILTER_WHERE]
        offset = int(query["offset"][0])
        length = int(query["length"][0])
        call_index = len(self.calls)
        self.calls.append((offset, length))
        self.where_values.append(query["where"][0])

        rows = [self._row(position) for position in range(offset, offset + length)]
        features = json.loads(json.dumps(DATASET_SERVER_FEATURES))
        revision = REVISION
        reported_total = self.total_rows
        reported_partial = self.partial

        if call_index == 1:
            if self.drift == "revision":
                revision = "f" * 40
            elif self.drift == "truncated_cells":
                rows[0]["truncated_cells"] = ["text"]
            elif self.drift == "schema":
                features[1]["type"]["dtype"] = "int64"
            elif self.drift == "partial":
                reported_partial = not self.partial
            elif self.drift == "num_rows_total":
                reported_total += 1
            elif self.drift == "page_duplicate":
                rows[1]["row_idx"] = rows[0]["row_idx"]
            elif self.drift == "page_descending":
                rows[1]["row_idx"] = rows[0]["row_idx"] - 1
            elif self.drift == "negative":
                rows[0]["row_idx"] = -1
            elif self.drift == "non_integer":
                rows[0]["row_idx"] = str(rows[0]["row_idx"])
        elif call_index == 2 and self.last_row_idx is not None:
            if self.drift == "cross_window_duplicate":
                rows[0]["row_idx"] = self.last_row_idx
            elif self.drift == "cross_window_backtrack":
                rows[0]["row_idx"] = self.last_row_idx - 1

        body = json.dumps(
            {
                "features": features,
                "rows": rows,
                "num_rows_total": reported_total,
                "partial": reported_partial,
            },
            sort_keys=True,
        ).encode()
        self.response_bytes.append(len(body))
        if rows and type(rows[-1]["row_idx"]) is int:
            self.last_row_idx = rows[-1]["row_idx"]
        return FakeDatasetServerResponse(body, revision)


def test_dataset_server_filter_uses_exact_where_and_original_row_indexes(tmp_path: Path):
    assert (
        corpus_gate_e.SOURCE_SPECS["stackv2_python"].dataset_server_where
        == STACKV2_PYTHON_FILTER_WHERE
    )
    server = FakeFilteredDatasetServer(total_rows=30, partial=True)
    source = _dataset_server_filter_source(inspection_cap=12)

    loaded = corpus_gate_e.hf_dataset_server_loader(
        source,
        tmp_path / "cache",
        12,
        opener=server,
        sleeper=_never_sleep,
    )
    rows = list(loaded.rows)

    expected_windows = (
        (0, 1),
        (1, 2),
        (4, 1),
        (7, 1),
        (10, 1),
        (13, 1),
        (16, 1),
        (19, 1),
        (22, 1),
        (25, 1),
        (28, 1),
    )
    filtered_positions = [
        position
        for offset, length in expected_windows
        for position in range(offset, offset + length)
    ]
    expected_original_indexes = [
        server._original_row_idx(position) for position in filtered_positions
    ]

    assert tuple(server.calls) == expected_windows
    assert loaded.sample_windows == expected_windows
    assert len(rows) == 12
    assert sum(length for _, length in server.calls) == 12
    assert [row["text"] for row in rows] == [
        f"Bounded synthetic educational row {row_idx}." for row_idx in expected_original_indexes
    ]
    assert all(value == STACKV2_PYTHON_FILTER_WHERE for value in server.where_values)
    assert loaded.transport == "huggingface_dataset_server_filter"
    assert loaded.resolved_revision == REVISION
    assert loaded.live_schema == dict(source.required_schema)
    assert loaded.reported_rows == 30
    assert loaded.partial is True
    assert loaded.request_count == len(server.calls)
    assert loaded.response_bytes == sum(server.response_bytes)


@pytest.mark.parametrize(
    "drift",
    [
        "revision",
        "truncated_cells",
        "schema",
        "partial",
        "num_rows_total",
        "page_duplicate",
        "page_descending",
        "negative",
        "non_integer",
        "cross_window_duplicate",
        "cross_window_backtrack",
    ],
)
def test_dataset_server_filter_fails_closed_on_transport_or_original_index_drift(
    tmp_path: Path,
    drift: str,
):
    server = FakeFilteredDatasetServer(total_rows=30, drift=drift)

    with pytest.raises(GateEError):
        corpus_gate_e.hf_dataset_server_loader(
            _dataset_server_filter_source(inspection_cap=12),
            tmp_path / "cache",
            12,
            opener=server,
            sleeper=_never_sleep,
        )

    expected_calls = 3 if drift.startswith("cross_window_") else 2
    assert len(server.calls) == expected_calls


@pytest.mark.parametrize("cap_kind", ["per_call", "total"])
def test_dataset_server_filter_enforces_response_byte_caps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cap_kind: str,
):
    assert corpus_gate_e.DATASET_SERVER_MAX_RESPONSE_BYTES == 32 * 1024**2
    assert corpus_gate_e.DATASET_SERVER_MAX_TOTAL_BYTES == 256 * 1024**2
    source = _dataset_server_filter_source(inspection_cap=12)

    if cap_kind == "per_call":
        monkeypatch.setattr(corpus_gate_e, "DATASET_SERVER_MAX_RESPONSE_BYTES", 64)

        def opener(request, timeout: float = 60.0) -> FakeDatasetServerResponse:
            assert timeout == 60.0
            parsed = urlparse(request.full_url)
            assert parsed.path.endswith("/filter")
            assert parse_qs(parsed.query)["where"] == [STACKV2_PYTHON_FILTER_WHERE]
            return FakeDatasetServerResponse(b"x" * 65, REVISION)

        expected_match = "per-call byte cap"
    else:
        server = FakeFilteredDatasetServer(total_rows=30)
        monkeypatch.setattr(corpus_gate_e, "DATASET_SERVER_MAX_TOTAL_BYTES", 1)
        opener = server
        expected_match = "total byte cap"

    with pytest.raises(GateEError, match=expected_match):
        corpus_gate_e.hf_dataset_server_loader(
            source,
            tmp_path / "cache",
            12,
            opener=opener,
            sleeper=_never_sleep,
        )
