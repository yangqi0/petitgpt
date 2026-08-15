#!/usr/bin/env python3

"""Small, source-agnostic I/O primitives for private manual review.

This module deliberately contains no dataset, network, review-policy, or TTY
workflow code.  It centralizes the narrow filesystem and terminal-sanitizing
operations that a later blinded-review tool can compose.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import resource
import secrets
import stat
from typing import Any
import unicodedata


class ManualReviewIOError(RuntimeError):
    """A fail-closed manual-review I/O contract violation."""


_FULL_FILE_FLAGS = os.O_RDONLY | os.O_NONBLOCK
_CREATE_FILE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL
_DIRECTORY_FLAGS = os.O_RDONLY
for _flag_name in ("O_CLOEXEC", "O_NOFOLLOW"):
    if not hasattr(os, _flag_name):  # pragma: no cover - required on production Linux.
        raise RuntimeError(f"manual-review I/O requires os.{_flag_name}")
    _FULL_FILE_FLAGS |= getattr(os, _flag_name)
    _CREATE_FILE_FLAGS |= getattr(os, _flag_name)
    _DIRECTORY_FLAGS |= getattr(os, _flag_name)
if hasattr(os, "O_DIRECTORY"):
    _DIRECTORY_FLAGS |= os.O_DIRECTORY


def sanitize_terminal_text(text: str) -> str:
    """Escape terminal-control Unicode while preserving reviewable text.

    Tab and newline are the only Cc characters emitted verbatim.  Every other
    Cc or Cf code point becomes an ASCII ``U+XXXX`` token (with more than four
    hex digits when needed).  Printable Unicode is otherwise unchanged.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    rendered: list[str] = []
    for character in text:
        if character in {"\t", "\n"}:
            rendered.append(character)
            continue
        if (
            unicodedata.category(character).startswith("C")
            or not character.isprintable()
        ):
            rendered.append(f"U+{ord(character):04X}")
        else:
            rendered.append(character)
    return "".join(rendered)


def _safe_basename(name: str, *, label: str) -> str:
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or Path(name).name != name
    ):
        raise ManualReviewIOError(f"{label} must be a non-empty basename")
    return name


def _absolute_components(path: Path | str) -> tuple[str, ...]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    if not absolute.is_absolute():  # pragma: no cover - abspath guarantees this.
        raise ManualReviewIOError("path must resolve lexically to an absolute path")
    return absolute.parts[1:]


def _open_directory_nofollow(directory: Path | str) -> int:
    """Open a directory after refusing symlinks in every path component."""
    descriptor: int | None = None
    try:
        descriptor = os.open("/", _DIRECTORY_FLAGS)
        for component in _absolute_components(directory):
            next_descriptor = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise ManualReviewIOError("directory must be an existing non-symlink directory") from exc
    assert descriptor is not None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ManualReviewIOError("directory descriptor is not a directory")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


@contextmanager
def open_directory_nofollow(directory: Path | str) -> Iterator[int]:
    """Yield an anchored directory fd after refusing every component symlink."""
    descriptor = _open_directory_nofollow(directory)
    try:
        yield descriptor
    finally:
        os.close(descriptor)


@contextmanager
def open_regular_file_nofollow(path: Path | str) -> Iterator[int]:
    """Open a regular file without following any path-component symlink."""
    absolute = Path(os.path.abspath(os.fspath(path)))
    parent_descriptor = _open_directory_nofollow(absolute.parent)
    try:
        descriptor = os.open(absolute.name, _FULL_FILE_FLAGS, dir_fd=parent_descriptor)
    except OSError as exc:
        os.close(parent_descriptor)
        raise ManualReviewIOError("file must be an existing regular non-symlink file") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ManualReviewIOError("opened object is not a regular file")
        yield descriptor
    finally:
        os.close(descriptor)
        os.close(parent_descriptor)


def _read_all_from_descriptor(descriptor: int, *, chunk_bytes: int = 1024 * 1024) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, chunk_bytes)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def read_regular_file_at_nofollow(
    directory_fd: int,
    name: str,
    *,
    max_bytes: int | None = None,
) -> bytes:
    """Read one regular basename relative to an already-anchored directory fd."""
    name = _safe_basename(name, label="file name")
    try:
        descriptor = os.open(name, _FULL_FILE_FLAGS, dir_fd=directory_fd)
    except OSError as exc:
        raise ManualReviewIOError("file must be an existing regular non-symlink file") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ManualReviewIOError("opened object is not a regular file")
        if max_bytes is not None and before.st_size > max_bytes:
            raise ManualReviewIOError("regular file exceeds the configured byte bound")
        data = _read_all_from_descriptor(descriptor)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ManualReviewIOError("regular file changed while it was being read")
        if len(data) != after.st_size or (max_bytes is not None and len(data) > max_bytes):
            raise ManualReviewIOError("regular file size changed while it was being read")
        return data
    finally:
        os.close(descriptor)


def _verify_published_file_at(directory_fd: int, name: str, expected: bytes) -> None:
    try:
        descriptor = os.open(name, _FULL_FILE_FLAGS, dir_fd=directory_fd)
    except OSError as exc:
        raise ManualReviewIOError("cannot reopen published manual-review output") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
            or metadata.st_size != len(expected)
        ):
            raise ManualReviewIOError("published manual-review output metadata drifted")
        observed = _read_all_from_descriptor(descriptor)
        if observed != expected or hashlib.sha256(observed).digest() != hashlib.sha256(expected).digest():
            raise ManualReviewIOError("published manual-review output bytes drifted")
    finally:
        os.close(descriptor)


def read_regular_file_nofollow(
    path: Path | str,
    *,
    max_bytes: int | None = None,
    chunk_bytes: int = 1024 * 1024,
) -> bytes:
    """Read a regular non-symlink file through its already-validated fd."""
    if max_bytes is not None and (
        isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0
    ):
        raise ValueError("max_bytes must be a non-negative integer or None")
    if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be a positive integer")
    with open_regular_file_nofollow(path) as descriptor:
        before = os.fstat(descriptor)
        if max_bytes is not None and before.st_size > max_bytes:
            raise ManualReviewIOError("regular file exceeds the configured byte bound")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, chunk_bytes)
            if not chunk:
                break
            total += len(chunk)
            if max_bytes is not None and total > max_bytes:
                raise ManualReviewIOError("regular file exceeded the configured byte bound")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ManualReviewIOError("regular file changed while it was being read")
        data = b"".join(chunks)
        if len(data) != after.st_size:
            raise ManualReviewIOError("regular file size changed while it was being read")
        return data


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize the frozen manual-review canonical JSON representation."""
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:  # pragma: no cover - defensive against impossible regular-file behavior.
            raise ManualReviewIOError("exclusive file write made no progress")
        written += count


def _create_exclusive_file_at(
    directory_fd: int,
    name: str,
    data: bytes,
) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(name, _CREATE_FILE_FLAGS, 0o600, dir_fd=directory_fd)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):  # pragma: no cover - O_EXCL creation guarantees it.
            raise ManualReviewIOError("exclusive output is not a regular file")
        os.fchmod(descriptor, 0o600)
        _write_all(descriptor, data)
        os.fsync(descriptor)
        if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o600:
            raise ManualReviewIOError("exclusive output mode is not 0600")
    except FileExistsError as exc:
        raise ManualReviewIOError("refusing to overwrite an existing file") from exc
    except OSError as exc:
        raise ManualReviewIOError("cannot create exclusive manual-review file") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def publish_content_addressed_bytes(
    directory: Path | str,
    *,
    stem: str,
    data: bytes,
    suffix: str = ".json",
) -> Path:
    """Atomically publish bytes under a digest name without overwrite.

    A mode-0600 temporary regular file is created in the destination directory,
    fsynced, and atomically linked to the final name.  Hard-link publication is
    used because ordinary rename would silently replace an existing path.
    """
    stem = _safe_basename(stem, label="stem")
    suffix = _safe_basename(suffix, label="suffix")
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    digest = hashlib.sha256(data).hexdigest()
    final_name = f"{stem}.sha256-{digest}{suffix}"
    _safe_basename(final_name, label="content-addressed filename")
    temporary_name = f".{stem}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    _safe_basename(temporary_name, label="temporary filename")
    directory_path = Path(directory)
    directory_fd = _open_directory_nofollow(directory_path)
    temporary_exists = False
    final_linked = False
    try:
        _create_exclusive_file_at(directory_fd, temporary_name, data)
        temporary_exists = True
        try:
            os.link(
                temporary_name,
                final_name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise ManualReviewIOError(
                "refusing to overwrite existing content-addressed output"
            ) from exc
        except OSError as exc:
            raise ManualReviewIOError("cannot atomically publish content-addressed output") from exc
        final_linked = True
        os.fsync(directory_fd)
        os.unlink(temporary_name, dir_fd=directory_fd)
        temporary_exists = False
        os.fsync(directory_fd)
        _verify_published_file_at(directory_fd, final_name, data)
    finally:
        if temporary_exists:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
                os.fsync(directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)
    if not final_linked:  # pragma: no cover - every failure raises above.
        raise ManualReviewIOError("content-addressed output was not published")
    return directory_path / final_name


def publish_content_addressed_json(
    directory: Path | str,
    *,
    stem: str,
    value: Mapping[str, Any],
) -> Path:
    """Canonicalize and content-address one JSON object."""
    if not isinstance(value, Mapping):
        raise TypeError("value must be a mapping")
    return publish_content_addressed_bytes(
        directory,
        stem=stem,
        data=canonical_json_bytes(value),
        suffix=".json",
    )


class SessionLock:
    """An O_EXCL, mode-0600 lock scoped to one existing directory."""

    def __init__(self, directory: Path | str, name: str = ".manual-review.lock") -> None:
        self.directory = Path(directory)
        self.name = _safe_basename(name, label="lock name")
        self._directory_fd: int | None = None
        self._held = False

    def __enter__(self) -> SessionLock:
        if self._held:
            raise ManualReviewIOError("session lock is already held by this object")
        directory_fd = _open_directory_nofollow(self.directory)
        try:
            _create_exclusive_file_at(directory_fd, self.name, b"")
            os.fsync(directory_fd)
        except Exception:
            os.close(directory_fd)
            raise
        self._directory_fd = directory_fd
        self._held = True
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        directory_fd = self._directory_fd
        self._directory_fd = None
        self._held = False
        if directory_fd is None:
            return
        try:
            os.unlink(self.name, dir_fd=directory_fd)
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def create_session_marker(
    directory: Path | str,
    *,
    name: str,
    value: Mapping[str, Any],
) -> Path:
    """Create one durable fixed-name commit marker, exactly once."""
    name = _safe_basename(name, label="marker name")
    if not isinstance(value, Mapping):
        raise TypeError("value must be a mapping")
    directory_path = Path(directory)
    directory_fd = _open_directory_nofollow(directory_path)
    try:
        data = canonical_json_bytes(value)
        _create_exclusive_file_at(directory_fd, name, data)
        os.fsync(directory_fd)
        _verify_published_file_at(directory_fd, name, data)
    finally:
        os.close(directory_fd)
    return directory_path / name


def disable_core_dumps() -> None:
    """Irreversibly set this process's soft and hard core limits to zero."""
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        current = resource.getrlimit(resource.RLIMIT_CORE)
    except (OSError, ValueError) as exc:
        raise ManualReviewIOError("cannot disable process core dumps") from exc
    if current != (0, 0):  # pragma: no cover - defensive kernel contract check.
        raise ManualReviewIOError("process core-dump limits are not zero")
