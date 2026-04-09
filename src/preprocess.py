"""Raw voice recordings -> RVC-ready training clips.

Pipeline per input file:
  1. Convert to canonical 44.1kHz mono 16-bit WAV
  2. Light denoise + bandpass (highpass 75, lowpass 15000, afftdn)
  3. Loudness normalization (EBU R128, target -20 LUFS)
  4. Slice into 3-15s clips via vendored slicer2
  5. Drop clips failing RMS/peak checks

Idempotent: wipes dataset/processed/ on every run.
Fails loud on any ffmpeg error.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

# Allow running as a script: ensure project root is on sys.path so
# `from src.* import ...` works whether invoked as a module or a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import soundfile as sf
import typer
from rich.console import Console

from src.doctor import check_ffmpeg, check_ffmpeg_filters
from src.ffmpeg_utils import FfmpegError, run_ffmpeg

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}
CANONICAL_SR = 44100
SILENCE_RMS_THRESHOLD = 0.005  # drop clips quieter than this
PEAK_CLIPPING_THRESHOLD = 0.9999  # peak above this = clipping


class PreprocessError(RuntimeError):
    pass


# ---------- ffmpeg arg builders (pure, easy to unit-test) ----------

def build_canonical_args(input_path: Path, output_path: Path) -> list[str]:
    """Build ffmpeg args to convert any audio to 44.1kHz mono 16-bit WAV."""
    return [
        "-i", str(input_path),
        "-ar", str(CANONICAL_SR),
        "-ac", "1",
        "-sample_fmt", "s16",
        str(output_path),
    ]


def build_denoise_args(input_path: Path, output_path: Path) -> list[str]:
    """Highpass + lowpass + mild FFT denoise."""
    return [
        "-i", str(input_path),
        "-af", "highpass=f=75,lowpass=f=15000,afftdn=nr=12",
        "-ar", str(CANONICAL_SR),
        "-ac", "1",
        "-sample_fmt", "s16",
        str(output_path),
    ]


def build_loudnorm_args(input_path: Path, output_path: Path, target_lufs: int) -> list[str]:
    """EBU R128 loudness normalization to target LUFS."""
    return [
        "-i", str(input_path),
        "-af", f"loudnorm=I={target_lufs}:TP=-1:LRA=11",
        "-ar", str(CANONICAL_SR),
        "-ac", "1",
        "-sample_fmt", "s16",
        str(output_path),
    ]


# ---------- Slicing ----------

def _slice_with_slicer2(
    wav_path: Path,
    out_dir: Path,
    min_len_s: float,
    max_len_s: float,
) -> list[Path]:
    """Slice a normalized wav into clips using vendored slicer2.

    Returns list of written clip paths.
    """
    try:
        from src.slicer2 import Slicer
    except ImportError:
        from slicer2 import Slicer  # script-invocation fallback

    audio, sr = sf.read(str(wav_path))
    slicer = Slicer(
        sr=sr,
        threshold=-40,
        min_length=int(min_len_s * 1000),
        min_interval=300,
        hop_size=10,
        max_sil_kept=500,
    )
    chunks = slicer.slice(audio)

    written: list[Path] = []
    stem = wav_path.stem
    for i, chunk in enumerate(chunks):
        if chunk.ndim > 1:
            chunk = chunk.mean(axis=1)  # defensive mono
        duration = len(chunk) / sr
        if duration < min_len_s or duration > max_len_s:
            continue
        # Drop near-silent clips
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        if rms < SILENCE_RMS_THRESHOLD:
            continue
        # Drop clipped clips
        peak = float(np.max(np.abs(chunk)))
        if peak > PEAK_CLIPPING_THRESHOLD:
            continue
        out_path = out_dir / f"{stem}_{i:04d}.wav"
        sf.write(str(out_path), chunk, sr, subtype="PCM_16")
        written.append(out_path)
    return written


# ---------- Pipeline ----------

def run_preprocess(
    input_dir: Path,
    output_dir: Path,
    min_len_s: float,
    max_len_s: float,
    target_lufs: int,
    dry_run: bool = False,
) -> dict:
    """Run the full preprocess pipeline. Returns a summary dict."""
    if not input_dir.exists():
        raise PreprocessError(f"input dir does not exist: {input_dir}")

    inputs = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS)
    if not inputs:
        raise PreprocessError(f"no audio files found in {input_dir}")

    if not dry_run:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

    console = Console()
    total_clips = 0
    total_duration = 0.0

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        for idx, in_path in enumerate(inputs, start=1):
            console.print(f"[{idx}/{len(inputs)}] {in_path.name}")
            canonical = tmpdir / f"{in_path.stem}_canonical.wav"
            denoised = tmpdir / f"{in_path.stem}_denoised.wav"
            normalized = tmpdir / f"{in_path.stem}_normalized.wav"

            canonical_args = build_canonical_args(in_path, canonical)
            denoise_args = build_denoise_args(canonical, denoised)
            loudnorm_args = build_loudnorm_args(denoised, normalized, target_lufs)

            if dry_run:
                console.print(f"  ffmpeg {' '.join(canonical_args)}")
                console.print(f"  ffmpeg {' '.join(denoise_args)}")
                console.print(f"  ffmpeg {' '.join(loudnorm_args)}")
                continue

            run_ffmpeg(canonical_args, context="canonical", expected_output=canonical)
            run_ffmpeg(denoise_args, context="denoise", expected_output=denoised)
            run_ffmpeg(loudnorm_args, context="loudnorm", expected_output=normalized)

            clips = _slice_with_slicer2(normalized, output_dir, min_len_s, max_len_s)
            total_clips += len(clips)
            for c in clips:
                data, sr = sf.read(str(c))
                total_duration += len(data) / sr
            console.print(f"  -> {len(clips)} clips")

    summary = {
        "inputs": len(inputs),
        "clips": total_clips,
        "total_duration_s": round(total_duration, 2),
        "mean_clip_s": round(total_duration / total_clips, 2) if total_clips else 0.0,
    }
    console.print(f"\n[bold green]Preprocess complete[/bold green]: {summary}")
    return summary


# ---------- CLI ----------

app = typer.Typer(add_completion=False, help="Preprocess raw voice recordings into RVC training clips.")


@app.command()
def main(
    input_dir: Path = typer.Option(Path("dataset/raw"), "--input", help="Directory containing raw audio"),
    output_dir: Path = typer.Option(Path("dataset/processed"), "--output", help="Where to write clips"),
    min_len: float = typer.Option(3.0, "--min-len", help="Minimum clip length (seconds)"),
    max_len: float = typer.Option(15.0, "--max-len", help="Maximum clip length (seconds)"),
    target_lufs: int = typer.Option(-20, "--target-lufs", help="Loudness normalization target"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print commands without running"),
) -> None:
    """Preprocess raw voice recordings."""
    # Pre-flight
    ff = check_ffmpeg()
    if not ff.ok:
        typer.echo(f"ffmpeg check failed: {ff.detail}\n{ff.fix_hint}", err=True)
        raise typer.Exit(code=1)
    ff2 = check_ffmpeg_filters()
    if not ff2.ok:
        typer.echo(f"ffmpeg filter check failed: {ff2.detail}\n{ff2.fix_hint}", err=True)
        raise typer.Exit(code=1)

    try:
        run_preprocess(
            input_dir=input_dir,
            output_dir=output_dir,
            min_len_s=min_len,
            max_len_s=max_len,
            target_lufs=target_lufs,
            dry_run=dry_run,
        )
    except (PreprocessError, FfmpegError) as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=3) from exc


if __name__ == "__main__":
    app()
