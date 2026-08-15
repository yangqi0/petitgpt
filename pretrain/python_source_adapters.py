"""Schema adapters for revision-pinned Python metadata collection.

Adapters only rename fields that are present upstream.  Optional fields stay
absent when the source schema does not provide them; no provenance, quality, or
license value is synthesized.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any


class AdapterError(RuntimeError):
    """Raised when an upstream row cannot be normalized without guessing."""


@dataclass(frozen=True)
class PythonSourceAdapter:
    name: str
    aliases: Mapping[str, tuple[str, ...]]
    required_fields: frozenset[str] = frozenset({"blob_id", "length_bytes"})

    def resolve_schema(self, upstream_fields: set[str]) -> dict[str, str]:
        """Resolve one unambiguous upstream field for every available value."""
        resolved: dict[str, str] = {}
        for canonical, candidates in self.aliases.items():
            present = [candidate for candidate in candidates if candidate in upstream_fields]
            if len(present) > 1:
                raise AdapterError(
                    f"adapter {self.name!r} found ambiguous aliases for {canonical!r}: {present}"
                )
            if present:
                resolved[canonical] = present[0]
        missing = sorted(self.required_fields.difference(resolved))
        if missing:
            raise AdapterError(
                f"adapter {self.name!r} is missing required upstream fields: {missing}"
            )
        return resolved

    def normalize(
        self,
        row: Mapping[str, Any],
        *,
        field_map: Mapping[str, str],
    ) -> dict[str, Any]:
        normalized: dict[str, Any] = {"row_idx": _required_int(row, "row_idx")}
        for canonical, upstream in field_map.items():
            if upstream not in row:
                raise AdapterError(f"row {normalized['row_idx']} lost mapped field {upstream!r}")
            value = row[upstream]
            if value is None:
                if canonical in self.required_fields:
                    raise AdapterError(
                        f"row {normalized['row_idx']} has null required field {upstream!r}"
                    )
                continue
            normalized[canonical] = value

        blob_id = normalized.get("blob_id")
        if not isinstance(blob_id, str) or not blob_id:
            raise AdapterError(f"row {normalized['row_idx']} has invalid blob_id")
        normalized["length_bytes"] = _required_int(normalized, "length_bytes")
        if normalized["length_bytes"] < 0:
            raise AdapterError(f"row {normalized['row_idx']} has negative length_bytes")
        for name in ("score",):
            if name in normalized:
                try:
                    value = float(normalized[name])
                except (TypeError, ValueError) as exc:
                    raise AdapterError(f"row {normalized['row_idx']} has invalid {name}") from exc
                if not math.isfinite(value):
                    raise AdapterError(f"row {normalized['row_idx']} has non-finite {name}")
                normalized[name] = value
        if "int_score" in normalized:
            normalized["int_score"] = _required_int(normalized, "int_score")
        for name in (
            "repo_name",
            "path",
            "license",
            "license_type",
            "src_encoding",
            "language",
        ):
            if name in normalized and not isinstance(normalized[name], str):
                raise AdapterError(f"row {normalized['row_idx']} has non-string {name}")
        if "detected_licenses" in normalized:
            detected = normalized["detected_licenses"]
            if not isinstance(detected, list) or any(
                not isinstance(value, str) for value in detected
            ):
                raise AdapterError(f"row {normalized['row_idx']} has invalid detected_licenses")
            normalized["detected_licenses"] = list(detected)
        return normalized


def _required_int(row: Mapping[str, Any], name: str) -> int:
    value = row.get(name)
    if isinstance(value, bool):
        raise AdapterError(f"{name} cannot be boolean")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise AdapterError(f"invalid integer field {name!r}: {value!r}") from exc


_COMMON_FIELD_ALIASES = {
    "repo_name": ("repo_name", "repository_name"),
    "path": ("path", "file_path"),
    "score": ("score", "edu_score"),
    "int_score": ("int_score", "edu_int_score"),
    "license": ("license",),
    "src_encoding": ("src_encoding", "source_encoding"),
    "language": ("language",),
}

_PRIMARY_REQUIRED_FIELDS = frozenset({
    "blob_id",
    "length_bytes",
    "repo_name",
    "path",
    "score",
    "int_score",
})
_STACK_REQUIRED_FIELDS = _PRIMARY_REQUIRED_FIELDS | frozenset({
    "language",
    "src_encoding",
    "detected_licenses",
    "license_type",
})


ADAPTERS: dict[str, PythonSourceAdapter] = {
    "smollm_python_edu": PythonSourceAdapter(
        name="smollm_python_edu",
        required_fields=_PRIMARY_REQUIRED_FIELDS,
        aliases={
            "blob_id": ("blob_id",),
            "length_bytes": ("length_bytes",),
            **_COMMON_FIELD_ALIASES,
        },
    ),
    "stack_edu_python": PythonSourceAdapter(
        name="stack_edu_python",
        required_fields=_STACK_REQUIRED_FIELDS,
        aliases={
            "blob_id": ("blob_id", "content_id", "hexsha"),
            "length_bytes": ("length_bytes", "byte_size", "content_length"),
            **_COMMON_FIELD_ALIASES,
            "detected_licenses": ("detected_licenses",),
            "license_type": ("license_type",),
        },
    ),
}


def get_adapter(name: str) -> PythonSourceAdapter:
    try:
        return ADAPTERS[name]
    except KeyError as exc:
        raise AdapterError(
            f"unknown Python source adapter {name!r}; choose one of {sorted(ADAPTERS)}"
        ) from exc
