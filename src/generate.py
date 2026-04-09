"""text -> your voice CLI.

Pipeline:
  1. Resolve text source (positional arg, file, or smoke-test canned text)
  2. Generate intermediate audio via Edge-TTS (Microsoft TTS, English voice)
  3. Convert Edge-TTS mp3 to 44.1kHz mono wav
  4. Run RVC voice conversion via subprocess into rvc/.venv
  5. Verify output exists and is readable
  6. Clean up intermediate files unless --keep-intermediate

Exit codes:
  0  success
  1  config / setup error (missing model, missing venv)
  2  user input error (empty text, mutually-exclusive flags, bad voice)
  3  runtime error (ffmpeg, edge-tts, rvc subprocess failure)
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

# Allow running as a script: `python src/generate.py ...` puts src/ on
# sys.path instead of the project root, breaking `from src.* import ...`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import soundfile as sf
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from src.doctor import (
    PROJECT_ROOT,
    RVC_DIR,
    RVC_VENV_PYTHON,
    check_edge_tts_importable,
    check_ffmpeg,
    check_model_file,
    check_rvc_cloned,
    check_rvc_venv,
)
from src.ffmpeg_utils import FfmpegError, run_ffmpeg

load_dotenv()

DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "myvoice_v1")
DEFAULT_EDGE_VOICE = os.environ.get("DEFAULT_EDGE_VOICE", "en-US-GuyNeural")
DEFAULT_DEVICE = os.environ.get("DEVICE", "cuda:0")
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

app = typer.Typer(
    add_completion=False,
    help="Generate audio from text using your trained RVC voice.",
)
console = Console()


# ---------- Helpers ----------


def _slugify(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip())
    s = s.strip("_").lower()
    return s[:max_len] or "untitled"


def _default_output_path(text: str) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return OUTPUT_DIR / f"{ts}_{_slugify(text)}.wav"


async def _list_english_voices() -> list[dict]:
    import edge_tts

    voices = await edge_tts.list_voices()
    return [v for v in voices if v.get("Locale", "").startswith("en-")]


async def _generate_edge_tts(text: str, voice: str, out_mp3: Path) -> None:
    import edge_tts

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(out_mp3))


def build_rvc_subprocess_cmd(
    *,
    rvc_python: Path,
    rvc_dir: Path,
    input_wav: Path,
    model_name: str,
    index_path: Path,
    output_wav: Path,
    pitch: int,
    index_rate: float,
    filter_radius: int,
    device: str,
) -> list[str]:
    """Build the argv for RVC's tools/infer_cli.py.

    tools/infer_cli.py reads --model_name as the filename stored under
    rvc/assets/weights/. We copy our models/<name>.pth there before running.
    """
    return [
        str(rvc_python),
        "tools/infer_cli.py",
        "--input_path",
        str(input_wav.resolve()),
        "--index_path",
        str(index_path.resolve()),
        "--f0method",
        "rmvpe",
        "--opt_path",
        str(output_wav.resolve()),
        "--model_name",
        f"{model_name}.pth",
        "--index_rate",
        str(index_rate),
        "--device",
        device,
        "--is_half",
        "True",
        "--filter_radius",
        str(filter_radius),
        "--resample_sr",
        "0",
        "--rms_mix_rate",
        "1",
        "--protect",
        "0.33",
        "--f0up_key",
        str(pitch),
    ]


def _ensure_rvc_weight_staged(model_name: str) -> Path:
    """RVC's infer_cli reads weights from rvc/assets/weights/<name>.pth.
    Make sure the file is there (copied from models/)."""
    src = MODELS_DIR / f"{model_name}.pth"
    dst_dir = RVC_DIR / "assets" / "weights"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{model_name}.pth"
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        shutil.copy2(src, dst)
    return dst


def _tail(s: str, n: int) -> str:
    lines = s.strip().split("\n")
    return "\n".join(lines[-n:])


# ---------- CLI ----------


@app.command()
def main(
    text: Optional[str] = typer.Argument(None, help="Text to speak"),
    text_file: Optional[Path] = typer.Option(None, "--text-file", help="Read text from a file"),
    out: Optional[Path] = typer.Option(None, "--out", help="Output wav path"),
    model: str = typer.Option(
        DEFAULT_MODEL, "--model", help="Model name (resolves to models/<name>.pth + .index)"
    ),
    tts_voice: str = typer.Option(
        DEFAULT_EDGE_VOICE, "--tts-voice", help="Edge-TTS voice for stage 1"
    ),
    pitch: int = typer.Option(0, "--pitch", help="RVC pitch shift in semitones"),
    index_rate: float = typer.Option(0.7, "--index-rate", help="RVC index blending (0.0-1.0)"),
    filter_radius: int = typer.Option(3, "--filter-radius", help="RVC median filter radius"),
    device: str = typer.Option(DEFAULT_DEVICE, "--device", help="GPU device, e.g. cuda:0"),
    keep_intermediate: bool = typer.Option(
        False, "--keep-intermediate", help="Preserve Edge-TTS intermediate wav"
    ),
    smoke_test: bool = typer.Option(False, "--smoke-test", help="Run canned end-to-end test"),
    list_voices: bool = typer.Option(False, "--list-voices", help="List English Edge-TTS voices"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print steps without running"),
    verbose: bool = typer.Option(False, "--verbose", help="Print full RVC stderr"),
) -> None:
    """Generate audio from text using your trained voice."""
    # --- Handle non-generation modes ---
    if list_voices:
        voices = asyncio.run(_list_english_voices())
        table = Table(title="English Edge-TTS voices")
        table.add_column("ShortName", style="cyan")
        table.add_column("Gender")
        table.add_column("Locale")
        for v in voices:
            table.add_row(v.get("ShortName", "?"), v.get("Gender", "?"), v.get("Locale", "?"))
        console.print(table)
        raise typer.Exit(code=0)

    # --- Resolve text source ---
    if smoke_test:
        if text or text_file:
            typer.echo(
                "[error] --smoke-test is mutually exclusive with text and --text-file", err=True
            )
            raise typer.Exit(code=2)
        resolved_text = "Testing the voice pipeline. One, two, three."
    elif text and text_file:
        typer.echo("[error] text argument and --text-file are mutually exclusive", err=True)
        raise typer.Exit(code=2)
    elif text_file:
        if not text_file.exists():
            typer.echo(f"[error] text file not found: {text_file}", err=True)
            raise typer.Exit(code=2)
        resolved_text = text_file.read_text().strip()
    elif text is not None:
        resolved_text = text.strip()
    else:
        typer.echo("[error] provide text as positional argument or --text-file", err=True)
        raise typer.Exit(code=2)

    if not resolved_text:
        typer.echo("[error] text input is empty", err=True)
        raise typer.Exit(code=2)

    # --- Pre-flight checks ---
    for check_fn in (check_ffmpeg, check_edge_tts_importable, check_rvc_cloned, check_rvc_venv):
        r = check_fn()
        if not r.ok:
            typer.echo(f"[error] {r.name}: {r.detail}\n  fix: {r.fix_hint}", err=True)
            raise typer.Exit(code=1)

    model_check = check_model_file(model)
    if not model_check.ok:
        typer.echo(
            f"[error] {model_check.name}: {model_check.detail}\n  fix: {model_check.fix_hint}",
            err=True,
        )
        raise typer.Exit(code=1)

    index_path = MODELS_DIR / f"{model}.index"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_out = out if out else _default_output_path(resolved_text)
    final_out.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        console.print(f"[yellow]DRY RUN[/yellow] — would generate: {final_out}")
        console.print(f"  text: {resolved_text[:80]}...")
        console.print(f"  voice: {tts_voice}")
        console.print(f"  model: {model}")
        raise typer.Exit(code=0)

    # --- Run the pipeline ---
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        mp3_path = tmpdir / "edge_tts.mp3"
        wav_path = tmpdir / "edge_tts.wav"

        # Stage 1: Edge-TTS
        console.print(f"[cyan][1/3][/cyan] Edge-TTS ({tts_voice}) ...")
        try:
            asyncio.run(_generate_edge_tts(resolved_text, tts_voice, mp3_path))
        except Exception as exc:
            typer.echo(
                f"[error] Edge-TTS failed: {exc}\n  hint: check network and voice name (--list-voices)",
                err=True,
            )
            raise typer.Exit(code=2) from exc
        if not mp3_path.exists() or mp3_path.stat().st_size == 0:
            typer.echo("[error] Edge-TTS produced empty output", err=True)
            raise typer.Exit(code=3)

        # Stage 2: ffmpeg convert mp3 -> canonical wav
        console.print("[cyan][2/3][/cyan] Converting to canonical wav ...")
        try:
            run_ffmpeg(
                [
                    "-i",
                    str(mp3_path),
                    "-ar",
                    "44100",
                    "-ac",
                    "1",
                    "-sample_fmt",
                    "s16",
                    str(wav_path),
                ],
                context="tts-to-canonical",
                expected_output=wav_path,
            )
        except FfmpegError as exc:
            typer.echo(f"[error] {exc}", err=True)
            raise typer.Exit(code=3) from exc

        # Stage 3: RVC inference
        console.print(f"[cyan][3/3][/cyan] RVC inference ({model}) ...")
        _ensure_rvc_weight_staged(model)
        cmd = build_rvc_subprocess_cmd(
            rvc_python=RVC_VENV_PYTHON,
            rvc_dir=RVC_DIR,
            input_wav=wav_path,
            model_name=model,
            index_path=index_path,
            output_wav=final_out,
            pitch=pitch,
            index_rate=index_rate,
            filter_radius=filter_radius,
            device=device,
        )
        proc = subprocess.run(cmd, cwd=RVC_DIR, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            typer.echo("[error] RVC inference failed", err=True)
            typer.echo(f"  command: {' '.join(cmd)}", err=True)
            typer.echo(f"  stderr (last lines):\n{_tail(proc.stderr, 20)}", err=True)
            if "CUDA out of memory" in proc.stderr:
                typer.echo("  hint: close other GPU apps or lower batch size", err=True)
            if keep_intermediate:
                kept = OUTPUT_DIR / "_last_intermediate.wav"
                shutil.copy(wav_path, kept)
                typer.echo(f"  intermediate kept at: {kept}", err=True)
            raise typer.Exit(code=3)

        if verbose and proc.stderr.strip():
            console.print(f"[dim]{proc.stderr.strip()}[/dim]")

        # Verify output
        if not final_out.exists() or final_out.stat().st_size < 1024:
            typer.echo(f"[error] RVC output missing or too small: {final_out}", err=True)
            raise typer.Exit(code=3)
        try:
            data, sr = sf.read(str(final_out))
            duration = len(data) / sr
        except Exception as exc:
            typer.echo(f"[error] output wav unreadable: {exc}", err=True)
            raise typer.Exit(code=3) from exc

        if keep_intermediate:
            kept = OUTPUT_DIR / f"_intermediate_{final_out.stem}.wav"
            shutil.copy(wav_path, kept)
            console.print(f"  intermediate kept at: {kept}")

    elapsed = time.time() - t0
    size_mb = final_out.stat().st_size / (1024 * 1024)
    console.print(
        f"[bold green]✓[/bold green] Generated {final_out} "
        f"({duration:.1f}s audio, {size_mb:.1f}MB) in {elapsed:.1f}s"
    )


if __name__ == "__main__":
    app()
