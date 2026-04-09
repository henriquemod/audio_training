"""Unit tests for src/doctor.py dependency verification."""
from __future__ import annotations

from unittest.mock import patch

from src.doctor import (
    check_ffmpeg,
    check_ffmpeg_filters,
    check_mise,
    check_python_version,
    parse_ffmpeg_version,
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


def test_check_mise_missing():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        result = check_mise()
    assert result.ok is False
    assert "mise.jdx.dev" in result.fix_hint
