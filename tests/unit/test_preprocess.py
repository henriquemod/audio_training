"""Unit tests for src/preprocess.py. No real ffmpeg or audio-slicer calls."""

from __future__ import annotations

from pathlib import Path

from src.preprocess import (
    PreprocessError,  # noqa: F401 - imported for export check
    build_canonical_args,
    build_denoise_args,
    build_loudnorm_args,
)


def test_build_canonical_args_has_correct_sample_rate():
    args = build_canonical_args(Path("in.mp3"), Path("out.wav"))
    assert "-ar" in args
    assert "44100" in args
    assert "-ac" in args
    assert "1" in args
    assert args[-1] == "out.wav"
    assert "in.mp3" in args


def test_build_denoise_args_has_required_filters():
    args = build_denoise_args(Path("in.wav"), Path("out.wav"))
    joined = " ".join(args)
    assert "highpass=f=75" in joined
    assert "lowpass=f=15000" in joined
    assert "afftdn" in joined


def test_build_loudnorm_args_target_lufs():
    args = build_loudnorm_args(Path("in.wav"), Path("out.wav"), target_lufs=-20)
    joined = " ".join(args)
    assert "loudnorm=I=-20" in joined
    assert "TP=-1" in joined


def test_build_loudnorm_args_custom_lufs():
    args = build_loudnorm_args(Path("in.wav"), Path("out.wav"), target_lufs=-18)
    assert "loudnorm=I=-18" in " ".join(args)
