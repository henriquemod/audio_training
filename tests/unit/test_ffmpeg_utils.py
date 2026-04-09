"""Unit tests for src/ffmpeg_utils.py."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.ffmpeg_utils import FfmpegError, run_ffmpeg


def test_run_ffmpeg_success(tmp_path: Path, fake_ffmpeg):
    out = tmp_path / "out.wav"
    fake_ffmpeg.configure(exit_code=0, touch_output=True)
    run_ffmpeg(
        ["-i", "input.wav", str(out)],
        context="test stage",
        expected_output=out,
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_run_ffmpeg_nonzero_exit_raises(tmp_path: Path, fake_ffmpeg):
    out = tmp_path / "out.wav"
    fake_ffmpeg.configure(exit_code=1, stderr="synthetic ffmpeg failure", touch_output=False)
    with pytest.raises(FfmpegError) as excinfo:
        run_ffmpeg(
            ["-i", "input.wav", str(out)],
            context="loudnorm stage",
            expected_output=out,
        )
    msg = str(excinfo.value)
    assert "loudnorm stage" in msg
    assert "synthetic ffmpeg failure" in msg


def test_run_ffmpeg_missing_output_raises(tmp_path: Path, fake_ffmpeg):
    out = tmp_path / "out.wav"
    fake_ffmpeg.configure(exit_code=0, touch_output=False)
    with pytest.raises(FfmpegError) as excinfo:
        run_ffmpeg(
            ["-i", "input.wav", str(out)],
            context="slicing stage",
            expected_output=out,
        )
    assert "did not produce" in str(excinfo.value).lower() or "missing" in str(excinfo.value).lower()


def test_run_ffmpeg_empty_output_raises(tmp_path: Path, fake_ffmpeg):
    # Fake exits 0 but does NOT touch the output file. Pre-create it empty so
    # the existence check passes and only the size check trips.
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    fake_ffmpeg.configure(exit_code=0, touch_output=False)

    with pytest.raises(FfmpegError) as excinfo:
        run_ffmpeg(
            ["-i", "input.wav", str(empty)],
            context="convert stage",
            expected_output=empty,
        )
    assert "empty" in str(excinfo.value).lower()


def test_run_ffmpeg_never_uses_shell(tmp_path: Path, fake_ffmpeg):
    """Regression: args must be passed as a list, never via shell=True."""
    out = tmp_path / "out.wav"
    # An argument with a semicolon should NOT be interpreted by a shell.
    run_ffmpeg(
        ["-i", "weird;filename.wav", str(out)],
        context="shell safety",
        expected_output=out,
    )
    # If shell=True were used, the semicolon would split the command and
    # the fake wouldn't touch the output file. Since it exists, we're safe.
    assert out.exists()
