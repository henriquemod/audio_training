"""Integration test using real ffmpeg. Generates synthetic audio and runs
the full preprocess pipeline against it."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import soundfile as sf

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="ffmpeg not available",
)


def _synth_speech_like_wav(path: Path, num_segments: int = 4, segment_s: float = 5.0) -> None:
    """Generate a synthetic wav alternating tone-with-noise and silence, so
    slicer2 finds silence boundaries and produces multiple clips.

    Layout (default): [5s tone][1s silence][5s tone][1s silence]... repeated
    num_segments times. Total ~24s for defaults.
    """
    # Build a filter_complex that concatenates N [tone][silence] segment pairs.
    # Each tone segment is a 220 Hz sine mixed with low-level noise, mono 44.1k.
    tone_filter = (
        f"sine=frequency=220:duration={segment_s}:sample_rate=44100,"
        "volume=0.5"
    )
    silence_filter = "anullsrc=r=44100:cl=mono,atrim=duration=1.0,asetpts=N/SR/TB"

    parts = []
    for _ in range(num_segments):
        parts.append(tone_filter)
        parts.append(silence_filter)
    # Use lavfi for each part, then concat
    inputs: list[str] = []
    for p in parts:
        inputs += ["-f", "lavfi", "-i", p]

    concat_labels = "".join(f"[{i}:a]" for i in range(len(parts)))
    filter_complex = f"{concat_labels}concat=n={len(parts)}:v=0:a=1[out]"

    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-ac", "1",
            "-ar", "44100",
            str(path),
        ],
        check=True,
    )


def test_preprocess_produces_wav_clips(tmp_path: Path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    processed.mkdir()
    _synth_speech_like_wav(raw / "sample.wav")

    from src.preprocess import run_preprocess

    run_preprocess(
        input_dir=raw,
        output_dir=processed,
        min_len_s=3.0,
        max_len_s=15.0,
        target_lufs=-20,
    )

    wavs = sorted(processed.glob("*.wav"))
    assert len(wavs) >= 1, "preprocess should produce at least one output clip"

    for w in wavs:
        data, sr = sf.read(w)
        assert sr == 44100, f"expected 44.1kHz, got {sr} for {w.name}"
        assert data.ndim == 1, f"expected mono, got {data.ndim}-channel for {w.name}"
        duration = len(data) / sr
        assert 3.0 <= duration <= 15.0 + 0.1, f"clip {w.name} has duration {duration:.2f}s"
