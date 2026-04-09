"""Unit tests for src/doctor.py dependency verification."""

from __future__ import annotations

from unittest.mock import patch

from src.doctor import (
    check_ffmpeg,
    check_ffmpeg_filters,
    check_mise,
    check_python_version,
    parse_ffmpeg_version,
    parse_ffmpeg_version_display,
)

# --- parse_ffmpeg_version ---


def test_parse_ffmpeg_version_standard():
    output = "ffmpeg version 6.1.1-3ubuntu5 Copyright (c) 2000-2023 the FFmpeg developers"
    assert parse_ffmpeg_version(output) == (6, 1, 1)


def test_parse_ffmpeg_version_short():
    output = "ffmpeg version 5.0 Copyright (c) 2000-2022"
    assert parse_ffmpeg_version(output) == (5, 0, 0)


def test_parse_ffmpeg_version_unparseable_returns_none():
    assert parse_ffmpeg_version("garbage output") is None


def test_parse_ffmpeg_version_nightly_btbn_build():
    # BtbN static builds used on training pods report a git tag, not a release.
    output = (
        "ffmpeg version N-123884-gd3d0b7a5ee-20260409 "
        "Copyright (c) 2000-2026 the FFmpeg developers"
    )
    result = parse_ffmpeg_version(output)
    assert result is not None
    # Sentinel major must be high enough to satisfy any realistic floor.
    assert result >= (5, 0, 0)
    assert result[0] >= 9999


# --- parse_ffmpeg_version_display ---


def test_parse_ffmpeg_version_display_stable():
    output = "ffmpeg version 6.1.1-3ubuntu5 Copyright (c) 2000-2023 the FFmpeg developers"
    assert parse_ffmpeg_version_display(output) == "6.1.1-3ubuntu5"


def test_parse_ffmpeg_version_display_nightly():
    output = (
        "ffmpeg version N-123884-gd3d0b7a5ee-20260409 "
        "Copyright (c) 2000-2026 the FFmpeg developers"
    )
    assert parse_ffmpeg_version_display(output) == "N-123884-gd3d0b7a5ee-20260409"


def test_parse_ffmpeg_version_display_unknown():
    assert parse_ffmpeg_version_display("no version info here") == "unknown"


# --- check_ffmpeg ---


def test_check_ffmpeg_present_and_recent():
    fake_output = "ffmpeg version 6.1.1 Copyright (c) 2000-2023"
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = fake_output
        result = check_ffmpeg()
    assert result.ok is True
    assert "6.1.1" in result.detail


def test_check_ffmpeg_too_old():
    fake_output = "ffmpeg version 4.2.0 Copyright (c) 2000-2020"
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = fake_output
        result = check_ffmpeg()
    assert result.ok is False
    assert "5.0" in result.fix_hint


def test_check_ffmpeg_missing():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        result = check_ffmpeg()
    assert result.ok is False
    assert "apt install ffmpeg" in result.fix_hint


def test_check_ffmpeg_nightly_btbn_build_accepted():
    # BtbN static ffmpeg (used by scripts/setup_pod.sh) reports a git-tag version.
    # The check must accept it and surface the raw build tag in `detail` rather
    # than a misleading `9999.0.0`.
    fake_output = (
        "ffmpeg version N-123884-gd3d0b7a5ee-20260409 "
        "Copyright (c) 2000-2026 the FFmpeg developers"
    )
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = fake_output
        result = check_ffmpeg()
    assert result.ok is True
    assert result.detail == "N-123884-gd3d0b7a5ee-20260409"


# --- check_ffmpeg_filters ---


def test_check_ffmpeg_filters_all_present():
    fake_filters = (
        " TSC afftdn            A->A       Denoise audio samples using FFT.\n"
        " ... loudnorm          A->A       EBU R128 loudness normalization\n"
        " ... silencedetect     A->A       Detect silence.\n"
    )
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = fake_filters
        result = check_ffmpeg_filters()
    assert result.ok is True


def test_check_ffmpeg_filters_missing_afftdn():
    fake_filters = (
        " ... loudnorm          A->A       EBU R128 loudness normalization\n"
        " ... silencedetect     A->A       Detect silence.\n"
    )
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = fake_filters
        result = check_ffmpeg_filters()
    assert result.ok is False
    assert "afftdn" in result.detail


# --- check_python_version ---


def test_check_python_version_310():
    with patch("sys.version_info", (3, 10, 14, "final", 0)):
        result = check_python_version()
    assert result.ok is True


def test_check_python_version_314_fails():
    with patch("sys.version_info", (3, 14, 3, "final", 0)):
        result = check_python_version()
    assert result.ok is False
    assert "mise" in result.fix_hint.lower()


# --- check_mise ---


def test_check_mise_present():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "2026.3.18 linux-x64"
        result = check_mise()
    assert result.ok is True


def test_check_mise_missing_is_soft_ok():
    # mise is a laptop-dev convenience; on pods it is not expected to exist.
    # A missing `mise` binary must NOT fail the system-checks pre-flight that
    # scripts/setup_rvc.sh runs, otherwise pod bootstrap aborts.
    with patch("subprocess.run", side_effect=FileNotFoundError):
        result = check_mise()
    assert result.ok is True
    assert "optional" in result.detail.lower()
    assert result.fix_hint == ""


def test_check_mise_broken_install_is_soft_fail():
    # A mise binary that exits non-zero (broken install) is still surfaced as
    # a failure so laptop users notice, but only when the binary is present.
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = "mise: config error"
        mock_run.return_value.stdout = ""
        result = check_mise()
    assert result.ok is False
    assert "mise.jdx.dev" in result.fix_hint
