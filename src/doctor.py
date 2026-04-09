"""Dependency verification. Single source of truth for 'is this machine ready?'.

Every script calls the relevant subset of these checks before doing work.
Running `python src/doctor.py` directly prints a full report and exits
non-zero on any failure.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RVC_DIR = PROJECT_ROOT / "rvc"
RVC_VENV_PYTHON = RVC_DIR / ".venv" / "bin" / "python"
ROOT_VENV = PROJECT_ROOT / ".venv"
MODELS_DIR = PROJECT_ROOT / "models"

MIN_FFMPEG_VERSION = (5, 0, 0)
REQUIRED_PYTHON = (3, 10)
REQUIRED_FFMPEG_FILTERS = ("afftdn", "loudnorm", "silencedetect")


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    fix_hint: str = ""


Check = CheckResult  # alias for test imports


# ---------- Parsing helpers ----------


def parse_ffmpeg_version(output: str) -> tuple[int, int, int] | None:
    """Parse `ffmpeg -version` output and return (major, minor, patch), or None."""
    match = re.search(r"ffmpeg version (\d+)\.(\d+)(?:\.(\d+))?", output)
    if not match:
        return None
    major = int(match.group(1))
    minor = int(match.group(2))
    patch = int(match.group(3) or 0)
    return (major, minor, patch)


# ---------- System checks ----------


def check_mise() -> CheckResult:
    try:
        proc = subprocess.run(
            ["mise", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return CheckResult(
            name="mise",
            ok=False,
            fix_hint="Install mise: https://mise.jdx.dev/ (curl https://mise.run | sh)",
        )
    if proc.returncode != 0:
        return CheckResult(
            name="mise",
            ok=False,
            detail=proc.stderr.strip(),
            fix_hint="Install mise: https://mise.jdx.dev/",
        )
    return CheckResult(name="mise", ok=True, detail=proc.stdout.strip().split("\n")[0])


def check_python_version() -> CheckResult:
    actual = sys.version_info
    if (actual[0], actual[1]) != REQUIRED_PYTHON:
        return CheckResult(
            name="Python version",
            ok=False,
            detail=f"found {actual[0]}.{actual[1]}.{actual[2]}, need {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}.x",
            fix_hint="Run `mise install python@3.10 && mise use python@3.10` in the project directory",
        )
    return CheckResult(
        name="Python version",
        ok=True,
        detail=f"{actual[0]}.{actual[1]}.{actual[2]}",
    )


def check_ffmpeg() -> CheckResult:
    try:
        proc = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return CheckResult(
            name="ffmpeg",
            ok=False,
            fix_hint="Install ffmpeg: sudo apt install ffmpeg",
        )
    if proc.returncode != 0:
        return CheckResult(
            name="ffmpeg",
            ok=False,
            detail=proc.stderr.strip(),
            fix_hint="Reinstall ffmpeg: sudo apt install --reinstall ffmpeg",
        )
    version = parse_ffmpeg_version(proc.stdout)
    if version is None:
        return CheckResult(
            name="ffmpeg",
            ok=False,
            detail="could not parse ffmpeg version output",
            fix_hint="Reinstall ffmpeg: sudo apt install --reinstall ffmpeg",
        )
    if version < MIN_FFMPEG_VERSION:
        return CheckResult(
            name="ffmpeg",
            ok=False,
            detail=f"found {version[0]}.{version[1]}.{version[2]}",
            fix_hint="Need ffmpeg >= 5.0. Install a newer version (your distro may ship an older one).",
        )
    return CheckResult(
        name="ffmpeg",
        ok=True,
        detail=f"{version[0]}.{version[1]}.{version[2]}",
    )


def check_ffmpeg_filters() -> CheckResult:
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return CheckResult(
            name="ffmpeg filters",
            ok=False,
            fix_hint="Install ffmpeg: sudo apt install ffmpeg",
        )
    if proc.returncode != 0:
        return CheckResult(
            name="ffmpeg filters",
            ok=False,
            detail=proc.stderr.strip(),
        )
    missing = [f for f in REQUIRED_FFMPEG_FILTERS if f not in proc.stdout]
    if missing:
        return CheckResult(
            name="ffmpeg filters",
            ok=False,
            detail=f"missing filters: {', '.join(missing)}",
            fix_hint="Your ffmpeg build lacks required filters. Install the full 'ffmpeg' package.",
        )
    return CheckResult(
        name="ffmpeg filters",
        ok=True,
        detail=", ".join(REQUIRED_FFMPEG_FILTERS),
    )


def check_git() -> CheckResult:
    if shutil.which("git") is None:
        return CheckResult(
            name="git",
            ok=False,
            fix_hint="Install git: sudo apt install git",
        )
    return CheckResult(name="git", ok=True)


def check_nvidia_smi() -> CheckResult:
    if shutil.which("nvidia-smi") is None:
        return CheckResult(
            name="nvidia-smi",
            ok=False,
            fix_hint="Install NVIDIA drivers. See https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/",
        )
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return CheckResult(
            name="nvidia-smi",
            ok=False,
            detail=proc.stderr.strip(),
            fix_hint="nvidia-smi is installed but failed. Check driver installation.",
        )
    return CheckResult(name="nvidia-smi", ok=True, detail=proc.stdout.strip())


# ---------- Project state checks ----------


def check_rvc_cloned() -> CheckResult:
    if not RVC_DIR.exists():
        return CheckResult(
            name="rvc/ cloned",
            ok=False,
            fix_hint="Run ./scripts/setup_rvc.sh",
        )
    if not (RVC_DIR / ".git").exists():
        return CheckResult(
            name="rvc/ cloned",
            ok=False,
            detail="rvc/ exists but is not a git clone",
            fix_hint="Delete rvc/ and run ./scripts/setup_rvc.sh",
        )
    return CheckResult(name="rvc/ cloned", ok=True)


def check_rvc_venv() -> CheckResult:
    if not RVC_VENV_PYTHON.exists():
        return CheckResult(
            name="rvc/.venv",
            ok=False,
            fix_hint="Run ./scripts/setup_rvc.sh",
        )
    return CheckResult(name="rvc/.venv", ok=True)


def check_rvc_weights() -> CheckResult:
    hubert = RVC_DIR / "assets" / "hubert" / "hubert_base.pt"
    rmvpe = RVC_DIR / "assets" / "rmvpe" / "rmvpe.pt"
    missing = [p for p in (hubert, rmvpe) if not p.exists()]
    if missing:
        return CheckResult(
            name="rvc pretrained weights",
            ok=False,
            detail=f"missing: {', '.join(str(p.relative_to(PROJECT_ROOT)) for p in missing)}",
            fix_hint="Run ./scripts/setup_rvc.sh (it downloads the weights)",
        )
    return CheckResult(name="rvc pretrained weights", ok=True)


def check_model_file(model_name: str) -> CheckResult:
    pth = MODELS_DIR / f"{model_name}.pth"
    index = MODELS_DIR / f"{model_name}.index"
    if not pth.exists():
        return CheckResult(
            name=f"model {model_name}",
            ok=False,
            detail=f"{pth.relative_to(PROJECT_ROOT)} missing",
            fix_hint="Train a model via RVC WebUI, then run ./scripts/install_model.sh <name>",
        )
    if not index.exists():
        return CheckResult(
            name=f"model {model_name}",
            ok=False,
            detail=f"{index.relative_to(PROJECT_ROOT)} missing",
            fix_hint="Train the feature index in RVC WebUI, then run ./scripts/install_model.sh <name>",
        )
    return CheckResult(name=f"model {model_name}", ok=True)


# ---------- Runtime checks ----------


def check_edge_tts_importable() -> CheckResult:
    try:
        import edge_tts  # noqa: F401
    except ImportError as exc:
        return CheckResult(
            name="edge-tts importable",
            ok=False,
            detail=str(exc),
            fix_hint="Run: .venv/bin/pip install -e '.[dev]'",
        )
    return CheckResult(name="edge-tts importable", ok=True)


def check_slicer2_importable() -> CheckResult:
    """Verify the vendored slicer2 module is importable.

    Tries both package-qualified (`src.slicer2`) and bare (`slicer2`) imports
    so the check works whether doctor is run as a module or a script.
    """
    try:
        from src.slicer2 import Slicer  # noqa: F401

        return CheckResult(name="slicer2 importable", ok=True)
    except ImportError:
        pass
    try:
        import slicer2  # noqa: F401

        return CheckResult(name="slicer2 importable", ok=True)
    except ImportError as exc:
        return CheckResult(
            name="slicer2 importable",
            ok=False,
            detail=str(exc),
            fix_hint="src/slicer2.py is vendored. Check the file exists and numpy is installed.",
        )


def check_rvc_torch_cuda() -> CheckResult:
    if not RVC_VENV_PYTHON.exists():
        return CheckResult(
            name="rvc torch+cuda",
            ok=False,
            fix_hint="Run ./scripts/setup_rvc.sh first",
        )
    proc = subprocess.run(
        [
            str(RVC_VENV_PYTHON),
            "-c",
            "import torch; print(torch.__version__); print(torch.cuda.is_available())",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return CheckResult(
            name="rvc torch+cuda",
            ok=False,
            detail=proc.stderr.strip(),
            fix_hint="Reinstall torch in rvc/.venv with CUDA 12.1 wheels",
        )
    lines = proc.stdout.strip().split("\n")
    if len(lines) < 2 or lines[1].strip() != "True":
        return CheckResult(
            name="rvc torch+cuda",
            ok=False,
            detail=f"torch={lines[0] if lines else '?'}, cuda.is_available=False",
            fix_hint="Install PyTorch with CUDA: pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121",
        )
    return CheckResult(name="rvc torch+cuda", ok=True, detail=f"torch {lines[0]}")


# ---------- CLI ----------

app = typer.Typer(add_completion=False, help="Dependency verification for train_audio_model.")


def _run_checks(checks: list) -> bool:
    console = Console()
    table = Table(title="train_audio_model doctor", show_lines=False)
    table.add_column("Check", style="cyan")
    table.add_column("Status")
    table.add_column("Detail", style="dim")

    all_ok = True
    failures: list[CheckResult] = []
    for check_fn in checks:
        result = check_fn()
        mark = "[green]OK[/green]" if result.ok else "[red]FAIL[/red]"
        table.add_row(result.name, mark, result.detail)
        if not result.ok:
            all_ok = False
            failures.append(result)

    console.print(table)
    if failures:
        console.print("\n[bold red]Fix the following before proceeding:[/bold red]")
        for f in failures:
            console.print(f"  - [cyan]{f.name}[/cyan]: {f.fix_hint or '(see detail above)'}")
    return all_ok


@app.command()
def main(
    system_only: bool = typer.Option(False, "--system-only", help="Only check system-level deps"),
    rvc_only: bool = typer.Option(False, "--rvc-only", help="Only check RVC state"),
    runtime: bool = typer.Option(False, "--runtime", help="Only check Python runtime imports"),
    model: Optional[str] = typer.Option(
        None, "--model", help="Also verify this model exists in models/"
    ),
) -> None:
    """Run dependency checks. Exits non-zero if any check fails."""
    system_checks = [
        check_mise,
        check_python_version,
        check_ffmpeg,
        check_ffmpeg_filters,
        check_git,
        check_nvidia_smi,
    ]
    rvc_checks = [
        check_rvc_cloned,
        check_rvc_venv,
        check_rvc_weights,
        check_rvc_torch_cuda,
    ]
    runtime_checks = [
        check_edge_tts_importable,
        check_slicer2_importable,
    ]

    if system_only:
        selected = system_checks
    elif rvc_only:
        selected = rvc_checks
    elif runtime:
        selected = runtime_checks
    else:
        selected = system_checks + rvc_checks + runtime_checks

    if model is not None:
        selected = selected + [lambda m=model: check_model_file(m)]

    ok = _run_checks(selected)
    raise typer.Exit(code=0 if ok else 1)


if __name__ == "__main__":
    app()
