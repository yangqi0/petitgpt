from __future__ import annotations

import hashlib
import json
import math

import pytest

from pretrain.python_quality import (
    CHARS_PER_TOKEN_SENSITIVITY,
    HARD_GATE_ORDER,
    YIELD_TARGETS,
    analyze_python_source,
    deterministic_bootstrap_mean_interval,
    estimate_pretokenizer_yield,
    summarize_hard_gate_funnel,
    wilson_interval,
)


def _padded_source(prefix: str, *, minimum_bytes: int = 240) -> bytes:
    payload = prefix.encode("utf-8")
    if len(payload) < minimum_bytes:
        payload += b"#" + b"p" * (minimum_bytes - len(payload) - 1)
    return payload


def _analyze(raw: bytes, *, path: str = "source.py"):
    return analyze_python_source(
        raw,
        raw.decode("utf-8"),
        path=path,
        stdlib_modules={"json", "os", "sys"},
    )


def test_rich_python_metrics_are_source_free_and_structurally_complete():
    source = '''"""module documentation"""
# comment-secret-7843
import os
import external_package_secret
from .local_secret import helper
from package_secret import *

class BoxSecret:
    """class documentation"""
    def method_secret(self, values):
        """method documentation"""
        try:
            return [value for value in values if value]
        except ValueError:
            raise RuntimeError("literal-secret-2291")

async def async_secret(items):
    return (item async for item in items)

if __name__ == "__main__":
    BoxSecret().method_secret([])
'''
    raw = source.encode()
    metrics = analyze_python_source(
        raw,
        source,
        path="private-secret-path/tests/test_hidden_secret.py",
        stdlib_modules={"os", "sys"},
    )

    assert metrics["contains_source_text"] is False
    assert metrics["content_identity"]["raw_sha256"] == hashlib.sha256(raw).hexdigest()
    fingerprint = metrics["content_identity"]["ast_canonical_fingerprint"]
    assert fingerprint["available"] is True
    assert len(fingerprint["sha256"]) == 64
    assert fingerprint["detection_only"] is True
    assert fingerprint["training_text_modified"] is False
    assert metrics["syntax"]["python3_ast_parse_ok"] is True
    assert metrics["path_signals"]["test"] is True
    assert metrics["comments"]["count"] == 1
    assert metrics["comments"]["characters"] > 0
    assert metrics["docstrings"]["count"] == 3
    assert metrics["docstrings"]["character_share"] > 0

    counts = metrics["ast_constructs"]["counts"]
    groups = metrics["ast_constructs"]["groups"]
    assert counts["FunctionDef"] == 1
    assert counts["AsyncFunctionDef"] == 1
    assert counts["ClassDef"] == 1
    assert groups["import_statements"] == 4
    assert groups["exception_constructs"] == 3
    assert groups["comprehensions"] == 2

    imports = metrics["imports"]
    assert imports["statements"] == 4
    assert imports["imported_names"] == 4
    assert imports["absolute_stdlib_names"] == 1
    assert imports["absolute_nonstdlib_or_local_names"] == 2
    assert imports["relative_names"] == 1
    assert imports["wildcard_names"] == 1
    assert imports["proxies"]["has_relative_import"] is True
    assert imports["proxies"]["has_wildcard_import"] is True
    assert imports["proxies"]["has_nonstdlib_or_local_absolute_import"] is True
    assert metrics["boilerplate_descriptors"]["has_main_guard"] is True
    assert metrics["passes_all_hard_gates"] is True

    serialized = json.dumps(metrics, allow_nan=False, sort_keys=True)
    for private_value in (
        "comment-secret-7843",
        "external_package_secret",
        "literal-secret-2291",
        "private-secret-path",
        "BoxSecret",
    ):
        assert private_value not in serialized


def test_ast_canonical_fingerprint_is_detection_only_and_format_insensitive():
    first = 'alpha_secret = "same-literal"\n'
    second = 'alpha_secret="same-literal"  # a private comment\n'
    changed = 'alpha_secret = "different-literal"\n'
    first_metrics = analyze_python_source(first.encode(), first, stdlib_modules=set())
    second_metrics = analyze_python_source(second.encode(), second, stdlib_modules=set())
    changed_metrics = analyze_python_source(changed.encode(), changed, stdlib_modules=set())

    first_identity = first_metrics["content_identity"]
    second_identity = second_metrics["content_identity"]
    changed_identity = changed_metrics["content_identity"]
    assert first_identity["raw_sha256"] != second_identity["raw_sha256"]
    assert (
        first_identity["ast_canonical_fingerprint"]["sha256"]
        == second_identity["ast_canonical_fingerprint"]["sha256"]
    )
    assert (
        first_identity["ast_canonical_fingerprint"]["sha256"]
        != changed_identity["ast_canonical_fingerprint"]["sha256"]
    )
    assert "alpha_secret" not in json.dumps(first_metrics, sort_keys=True)


@pytest.mark.parametrize(
    ("path", "signal"),
    [
        ("project/tests/test_unit.py", "test"),
        ("project/config/settings.py", "config"),
        ("project/locks/file.py", "lock"),
        ("project/notebooks/demo.ipynb", "notebook"),
        ("project/third_party/module.py", "vendor"),
        ("project/generated/messages_pb2.py", "generated"),
    ],
)
def test_path_category_signals(path: str, signal: str):
    metrics = _analyze(_padded_source("value = 1\n"), path=path)
    assert metrics["path_signals"][signal] is True


def test_generated_vendor_repetition_minified_and_boilerplate_descriptors():
    generated_repeated = ("# AUTO-GENERATED - DO NOT EDIT\n" + "value = 1\n" * 25).encode()
    metrics = _analyze(
        generated_repeated,
        path="project/vendor/config/locks/.ipynb_checkpoints/test_file.ipynb",
    )
    assert metrics["generated"]["heuristic_flag"] is True
    assert metrics["path_signals"]["vendor"] is True
    assert metrics["path_signals"]["config"] is True
    assert metrics["path_signals"]["lock"] is True
    assert metrics["path_signals"]["notebook"] is True
    assert metrics["path_signals"]["test"] is True
    assert metrics["repetition"]["heuristic_flag"] is True
    assert metrics["hard_gates"]["not_generated"] is False
    assert metrics["hard_gates"]["not_vendor"] is False
    assert metrics["hard_gates"]["not_pathological_repetition"] is False

    minified = "long_integer = " + "1" * 600
    minified_metrics = analyze_python_source(minified.encode(), minified, stdlib_modules=set())
    assert minified_metrics["path_signals"]["minified"] is True
    assert minified_metrics["syntax"]["python3_ast_parse_ok"] is True

    boilerplate = '''"""documentation"""
# Copyright 2026
import os
__version__ = "1"
'''
    boilerplate_metrics = analyze_python_source(
        boilerplate.encode(), boilerplate, stdlib_modules={"os"}
    )["boilerplate_descriptors"]
    assert boilerplate_metrics["descriptive_only_not_a_hard_gate"] is True
    assert boilerplate_metrics["only_imports_assignments_and_docstring"] is True
    assert boilerplate_metrics["only_declarations_imports_assignments_and_docstring"] is True
    assert boilerplate_metrics["top_level_executable_statement_count"] == 0
    assert boilerplate_metrics["license_header_marker_count"] == 1


def test_empty_whitespace_control_size_and_invalid_ast_boundaries():
    empty = analyze_python_source(b"", "", stdlib_modules=set())
    whitespace_text = " " * 250
    whitespace = analyze_python_source(
        whitespace_text.encode(), whitespace_text, stdlib_modules=set()
    )
    assert empty["text_fidelity"]["empty"] is True
    assert whitespace["text_fidelity"]["whitespace_only"] is True
    assert empty["hard_gates"]["nonempty_nonwhitespace"] is False
    assert whitespace["hard_gates"]["nonempty_nonwhitespace"] is False

    exactly_one_percent = "x" * 99 + "\x01"
    above_one_percent = "x" * 98 + "\x01\x02"
    nul = "x" * 99 + "\x00"
    one_metrics = analyze_python_source(
        exactly_one_percent.encode(), exactly_one_percent, stdlib_modules=set()
    )
    above_metrics = analyze_python_source(
        above_one_percent.encode(), above_one_percent, stdlib_modules=set()
    )
    nul_metrics = analyze_python_source(nul.encode(), nul, stdlib_modules=set())
    assert one_metrics["text_fidelity"]["disallowed_c0_control_ratio"] == 0.01
    assert one_metrics["text_fidelity"]["binary_like"] is False
    assert above_metrics["text_fidelity"]["binary_like"] is True
    assert nul_metrics["text_fidelity"]["nul_present"] is True
    assert nul_metrics["text_fidelity"]["binary_like"] is True

    for length, expected in ((199, False), (200, True), (100_000, True), (100_001, False)):
        raw = b"#" + b"x" * (length - 2) + b"\n"
        metrics = _analyze(raw)
        assert len(raw) == length
        assert metrics["size"]["within_200_to_100000_bytes"] is expected

    invalid = "def secret_name(:\n    pass\n"
    invalid_metrics = analyze_python_source(
        invalid.encode(), invalid, path="secret-path.py", stdlib_modules=set()
    )
    assert invalid_metrics["syntax"]["python3_ast_parse_ok"] is False
    assert invalid_metrics["syntax"]["error_location_only_no_source"]["type"] == "SyntaxError"
    assert invalid_metrics["content_identity"]["ast_canonical_fingerprint"]["available"] is False
    serialized = json.dumps(invalid_metrics, allow_nan=False)
    assert "secret_name" not in serialized
    assert "secret-path" not in serialized


def test_gate_funnel_has_independent_and_ordered_document_and_byte_accounting():
    passing = _analyze(_padded_source("value = 1\n"))
    wrong_raw = analyze_python_source(b"wrong", "value = 1\n", stdlib_modules=set())
    too_small = _analyze(b"value = 1\n")
    whitespace = _analyze(b" " * 240)
    binary = _analyze(_padded_source("value = 1\n")[:-1] + b"\x00")
    syntax = _analyze(_padded_source("def broken(:\n"))
    generated = _analyze(_padded_source("# auto-generated\nvalue = 1\n"))
    vendor = _analyze(_padded_source("value = 1\n"), path="project/vendor/value.py")
    repetition = _analyze(("long_variable_name = 1\n" * 25).encode())
    analyses = [
        passing,
        wrong_raw,
        too_small,
        whitespace,
        binary,
        syntax,
        generated,
        vendor,
        repetition,
    ]

    funnel = summarize_hard_gate_funnel(analyses)
    assert funnel["hard_gate_order"] == list(HARD_GATE_ORDER)
    assert funnel["considered_documents"] == 9
    assert funnel["considered_raw_bytes"] == sum(
        metrics["size"]["raw_bytes"] for metrics in analyses
    )
    assert funnel["retained_documents"] == 1
    assert funnel["retained_raw_bytes"] == passing["size"]["raw_bytes"]
    assert [step["rejected_at_gate_documents"] for step in funnel["ordered_gate_funnel"]] == [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ]
    assert funnel["independent_gate_results"]["python3_ast_parse"]["failed_documents"] >= 2
    assert json.loads(json.dumps(funnel, allow_nan=False)) == funnel


def _yield_analysis(raw_bytes: int, text_characters: int, retained: bool):
    return {
        "size": {"raw_bytes": raw_bytes, "text_characters": text_characters},
        "passes_all_hard_gates": retained,
    }


def test_deterministic_confidence_intervals_and_pretokenizer_yield_math():
    constant = deterministic_bootstrap_mean_interval([4, 4, 4], seed=7, resamples=40)
    assert constant["estimate"] == constant["lower"] == constant["upper"] == 4.0
    first = deterministic_bootstrap_mean_interval([1, 2, 9, 10], seed=17, resamples=80)
    second = deterministic_bootstrap_mean_interval([1, 2, 9, 10], seed=17, resamples=80)
    assert first == second
    assert first["lower"] <= first["estimate"] <= first["upper"]

    analyses = [
        _yield_analysis(200, 400, True),
        _yield_analysis(300, 600, True),
        _yield_analysis(500, 800, False),
        _yield_analysis(700, 1_000, False),
    ]
    report = estimate_pretokenizer_yield(
        analyses,
        population_documents=100,
        seed=29,
        resamples=100,
    )
    replay = estimate_pretokenizer_yield(
        analyses,
        population_documents=100,
        seed=29,
        resamples=100,
    )
    assert report == replay
    assert report["scope"] == "PRE_TOKENIZER_SENSITIVITY_ONLY_NOT_A_CANONICAL_TOKEN_COUNT"
    assert report["canonical_tokenizer_used"] is False
    assert report["final_token_quota_supported"] is False
    assert report["sample"]["retained_documents"] == 2
    assert report["projected_retained_documents"]["estimate"] == 50
    assert report["projected_retained_raw_bytes"]["estimate"] == 12_500
    assert report["projected_retained_characters"]["estimate"] == 25_000

    sensitivity = report["serialized_token_sensitivity"]
    assert [item["assumed_characters_per_content_token"] for item in sensitivity] == list(
        CHARS_PER_TOKEN_SENSITIVITY
    )
    assert sensitivity[0]["projected_serialized_token_equivalent"]["estimate"] == 12_600
    assert all(item["includes_bos_and_eos_per_retained_document"] for item in sensitivity)

    break_even = report["break_even_characters_per_content_token"]
    assert [item["target_serialized_tokens"] for item in break_even] == list(YIELD_TARGETS)
    expected = 25_000 / (1_100_000_000 - 100)
    assert math.isclose(break_even[0]["estimate"], expected)
    assert "maximum assumed characters" in break_even[0]["meaning"]
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_wilson_interval_and_invalid_aggregate_inputs_fail_loudly():
    interval = wilson_interval(0, 300)
    assert interval["estimate"] == 0
    assert interval["lower"] == 0
    assert 0 < interval["upper"] < 0.02
    with pytest.raises(ValueError, match="total > 0"):
        wilson_interval(0, 0)
    with pytest.raises(ValueError, match="hard gate"):
        summarize_hard_gate_funnel([
            {"size": {"raw_bytes": 1}, "hard_gates": {"utf8_round_trip": True}}
        ])
    with pytest.raises(ValueError, match="not be empty"):
        estimate_pretokenizer_yield([], population_documents=10, seed=1)
    with pytest.raises(ValueError, match="finite"):
        deterministic_bootstrap_mean_interval([1.0, float("nan")], seed=1)
