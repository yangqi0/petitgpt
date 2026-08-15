from __future__ import annotations

import fcntl
import hashlib
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from pretrain.manual_review_io import (
    ManualReviewIOError,
    SessionLock,
    canonical_json_bytes,
    create_session_marker,
    open_regular_file_nofollow,
    publish_content_addressed_bytes,
    publish_content_addressed_json,
    read_regular_file_nofollow,
    sanitize_terminal_text,
)


def test_terminal_sanitizer_preserves_text_tab_newline_and_escapes_cc_cf():
    source = "print('héllo')\t# ok\n\x00\x1b\r\x7f\u200d\u202e\U000e0001\u2028\u2029\ue000終"
    sanitized = sanitize_terminal_text(source)
    assert sanitized == (
        "print('héllo')\t# ok\nU+0000U+001BU+000DU+007FU+200DU+202EU+E0001U+2028U+2029U+E000終"
    )
    assert "\x00" not in sanitized
    assert "\x1b" not in sanitized
    with pytest.raises(TypeError):
        sanitize_terminal_text(b"not text")  # type: ignore[arg-type]


def test_nofollow_regular_open_and_bounded_read(tmp_path: Path):
    path = tmp_path / "evidence.bin"
    payload = b"private synthetic bytes\x00"
    path.write_bytes(payload)

    with open_regular_file_nofollow(path) as descriptor:
        metadata = os.fstat(descriptor)
        assert stat.S_ISREG(metadata.st_mode)
        assert fcntl.fcntl(descriptor, fcntl.F_GETFD) & fcntl.FD_CLOEXEC
    assert read_regular_file_nofollow(path, max_bytes=len(payload)) == payload
    with pytest.raises(ManualReviewIOError, match="byte bound"):
        read_regular_file_nofollow(path, max_bytes=len(payload) - 1)
    with pytest.raises(ValueError):
        read_regular_file_nofollow(path, max_bytes=-1)
    with pytest.raises(ValueError):
        read_regular_file_nofollow(path, chunk_bytes=0)


def test_nofollow_open_rejects_symlink_directory_and_fifo(tmp_path: Path):
    target = tmp_path / "target"
    target.write_bytes(b"secret")
    symlink = tmp_path / "symlink"
    symlink.symlink_to(target)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)

    for path in (symlink, tmp_path, fifo):
        with pytest.raises(ManualReviewIOError):
            read_regular_file_nofollow(path)

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    nested = real_parent / "nested.bin"
    nested.write_bytes(b"synthetic")
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ManualReviewIOError, match="directory"):
        read_regular_file_nofollow(linked_parent / nested.name)


def test_canonical_json_bytes_is_utf8_sorted_finite_and_newline_terminated():
    value = {"z": "é", "a": {"n": 1}}
    assert canonical_json_bytes(value) == (b'{\n  "a": {\n    "n": 1\n  },\n  "z": "\xc3\xa9"\n}\n')
    with pytest.raises(ValueError):
        canonical_json_bytes({"bad": float("nan")})


def test_content_addressed_publish_is_0600_atomic_and_never_overwrites(tmp_path: Path):
    directory = tmp_path / "private-output"
    directory.mkdir(mode=0o700)
    value = {"schema_version": 1, "kind": "synthetic"}
    data = canonical_json_bytes(value)
    digest = hashlib.sha256(data).hexdigest()

    path = publish_content_addressed_json(directory, stem="review-queue", value=value)
    assert path.name == f"review-queue.sha256-{digest}.json"
    assert path.read_bytes() == data
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not [item for item in directory.iterdir() if item.name.startswith(".review-queue.")]

    with pytest.raises(ManualReviewIOError, match="overwrite"):
        publish_content_addressed_json(directory, stem="review-queue", value=value)
    assert path.read_bytes() == data


def test_content_addressed_publish_refuses_preexisting_final_and_symlink_directory(
    tmp_path: Path,
):
    directory = tmp_path / "private-output"
    directory.mkdir()
    data = b"sealed\n"
    digest = hashlib.sha256(data).hexdigest()
    final = directory / f"attestation.sha256-{digest}.json"
    final.write_bytes(b"do not replace")
    with pytest.raises(ManualReviewIOError, match="overwrite"):
        publish_content_addressed_bytes(
            directory,
            stem="attestation",
            data=data,
        )
    assert final.read_bytes() == b"do not replace"

    directory_link = tmp_path / "output-link"
    directory_link.symlink_to(directory, target_is_directory=True)
    with pytest.raises(ManualReviewIOError, match="directory"):
        publish_content_addressed_bytes(
            directory_link,
            stem="result",
            data=data,
        )


def test_session_lock_is_exclusive_durable_and_removed_on_exit(tmp_path: Path):
    lock_path = tmp_path / ".session.lock"
    with SessionLock(tmp_path, ".session.lock"):
        assert lock_path.is_file()
        assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600
        with pytest.raises(ManualReviewIOError, match="overwrite"):
            with SessionLock(tmp_path, ".session.lock"):
                raise AssertionError("unreachable")
    assert not lock_path.exists()


def test_session_marker_is_canonical_0600_and_single_use(tmp_path: Path):
    value = {"kind": "sealed", "complete": True}
    path = create_session_marker(
        tmp_path,
        name="SEALED.json",
        value=value,
    )
    assert path.read_bytes() == canonical_json_bytes(value)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(ManualReviewIOError, match="overwrite"):
        create_session_marker(tmp_path, name="SEALED.json", value=value)
    with pytest.raises(ManualReviewIOError, match="basename"):
        create_session_marker(tmp_path, name="../escape", value=value)


def test_disable_core_dumps_sets_soft_and_hard_limits_to_zero_in_subprocess():
    code = (
        "import resource\n"
        "from pretrain.manual_review_io import disable_core_dumps\n"
        "disable_core_dumps()\n"
        "assert resource.getrlimit(resource.RLIMIT_CORE) == (0, 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_publish_rejects_unsafe_names(tmp_path: Path):
    with pytest.raises(ManualReviewIOError, match="basename"):
        publish_content_addressed_bytes(tmp_path, stem="../escape", data=b"x")
    with pytest.raises(ManualReviewIOError, match="basename"):
        publish_content_addressed_bytes(tmp_path, stem="safe", suffix="../bad", data=b"x")
    with pytest.raises(TypeError):
        publish_content_addressed_bytes(tmp_path, stem="safe", data="x")  # type: ignore[arg-type]
