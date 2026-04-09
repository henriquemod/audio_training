# Technology Stack

**Analysis Date:** 2026-04-09

## Languages

**Primary:**
- Python 3.10 (exact) - All application code in `src/`, tests in `tests/`. Pinned by `pyproject.toml` (`requires-python = "==3.10.*"`) and `.mise.toml` (`python = "3.10"`).
- Bash - Setup and orchestration scripts in `scripts/` (`setup_rvc.sh`, `launch_rvc_webui.sh`, `install_model.sh`, `check.sh`).

**Secondary:**
- None. The `rvc/` directory is a vendored upstream clone (RVC-Project/Retrieval-based-Voice-Conversion-WebUI) pinned to commit `7ef19867780cf703841ebafb565a4e47d1ea86ff` (2024-11-24). It is excluded from lint/format and treated as a black-box subprocess.

## Runtime

**Environment:**
- CPython 3.10 on Linux (tested on Ubuntu/WSL2 per `README.md`).
- Two isolated virtualenvs by design:
  - `./.venv` - application venv (edge-tts, typer, rich, soundfile, numpy). Enforced by `scripts/setup_rvc.sh` via `ROOT_PYTHON="$PROJECT_ROOT/.venv/bin/python"`.
  - `./rvc/.venv` - RVC venv (PyTorch 2.1.2 + CUDA 12.1, fairseq, gradio 3.34.0). Created by `scripts/setup_rvc.sh`.
- Cross-venv communication is subprocess only. See `src/generate.py:build_rvc_subprocess_cmd`.

**Python Version Manager:**
- `mise` - declared in `.mise.toml`. The `doctor.check_mise()` function verifies availability.

**Package Manager:**
- `pip` (standard). Application venv installs via `pip install -e ".[dev]"`.
- RVC venv pins `pip<24.1` in `scripts/setup_rvc.sh` because fairseq 0.12.2 has legacy PEP 440 metadata pip 24.1+ rejects.
- Lockfile: `requirements.txt` (frozen reference, not the install source) and `pyproject.toml` (editable install source). No `poetry.lock` / `uv.lock` / `pip-tools` output for the app venv.

## Frameworks

**CLI / Application:**
- `typer==0.12.3` - CLI framework for `src/doctor.py`, `src/generate.py`, `src/preprocess.py`. Pinned; `pyproject.toml` explicitly preserves `Optional[X]` (UP007 ignored) because typer 0.12.3 lacks PEP 604 support.
- `click>=8.1,<8.2` - typer dependency, kept compatible.
- `rich==13.7.1` - console output tables (`doctor._run_checks`, `generate.main --list-voices`).
- `python-dotenv==1.0.1` - loads `.env` in `src/generate.py`.

**Audio:**
- `edge-tts>=7.0,<8` - Microsoft Edge TTS client (stage 1 of the pipeline in `src/generate.py`).
- `soundfile==0.12.1` - WAV I/O in `src/preprocess.py` and `src/generate.py`.
- `numpy>=1.24,<2.0` (installed as `1.26.4`) - RMS/peak clip filtering in `src/preprocess.py`.
- **Vendored:** `src/slicer2.py` - copied from `audio-slicer` 1.0.1 (MIT). Avoids a librosa/pydub dependency chain. Excluded from ruff via `extend-exclude`.
- External binary: `ffmpeg >= 5.0` with `afftdn`, `loudnorm`, `silencedetect` filters. Verified by `src.doctor.check_ffmpeg` and `check_ffmpeg_filters`.

**Testing:**
- `pytest==8.2.0`
- `pytest-mock==3.14.0`
- Custom pytest markers in `pyproject.toml`: `network`, `gpu`. Default `addopts = "-m 'not network and not gpu'"`.

**Lint / Format:**
- `ruff==0.4.4` - both linter and formatter. Config in `pyproject.toml` (`line-length = 100`, `target-version = "py310"`, selected rules `E, F, W, I, B, UP, N, SIM`). Excludes `rvc/` and `src/slicer2.py`.

**RVC venv (managed separately, not our direct dependency):**
- `torch==2.1.2` + `torchaudio==2.1.2` + `torchvision==0.16.2` from the CUDA 12.1 wheel index.
- `gradio==3.34.0` with `gradio_client==0.2.7` (pinned by `setup_rvc.sh` to prevent the `media_data` symbol mismatch).
- `matplotlib==3.7.3` (pinned by `setup_rvc.sh` because RVC calls the removed `FigureCanvasAgg.tostring_rgb()`).
- `fairseq 0.12.2` (indirect, via RVC's `requirements.txt`).

## Key Dependencies

**Critical (application venv):**
- `edge-tts` - stage 1 TTS. Network-dependent; failure mode is Edge-TTS 403 after Microsoft token rotation (documented in `README.md` troubleshooting).
- `soundfile` - every audio I/O path touches it. Requires libsndfile.
- `typer` - every entry point is a `typer.Typer()` app.

**Critical (RVC venv, shell-invoked):**
- `torch` with CUDA - mandatory. `doctor.check_rvc_torch_cuda` asserts `torch.cuda.is_available()` is True.
- RVC pretrained weights: `rvc/assets/hubert/hubert_base.pt` and `rvc/assets/rmvpe/rmvpe.pt`. Downloaded by `rvc/tools/download_models.py` during setup. Checked by `doctor.check_rvc_weights`.

**Infrastructure:**
- `ffmpeg` (system binary) - used for all audio transcoding and filter chains.
- `git` - required by `setup_rvc.sh` to clone and checkout the pinned RVC commit.
- `nvidia-smi` / NVIDIA drivers - required for GPU inference. Checked by `doctor.check_nvidia_smi`.

## Configuration

**Environment:**
- `.env` file (git-ignored) loaded by `src/generate.py` via `python-dotenv`. Template at `.env.example`.
- Keys (non-secret, user preferences):
  - `DEFAULT_MODEL` - base name of the trained model in `models/` (e.g. `myvoice_v1`, resolves to `models/myvoice_v1.pth` + `.index`).
  - `DEFAULT_EDGE_VOICE` - Edge-TTS voice short name (default `en-US-GuyNeural`).
  - `DEVICE` - CUDA device string (default `cuda:0`).
- `.env` has no secrets, but it is git-ignored per `.gitignore`.

**Build:**
- `pyproject.toml` - PEP 621 project metadata, ruff, pytest config. Only build file for the app package.
- `requirements.txt` - flat frozen list (not used for install; reference only).
- `.mise.toml` - pins Python 3.10.
- No `setup.py`, no `Makefile`. Orchestration lives in `scripts/check.sh`.

## Platform Requirements

**Development:**
- Linux (Ubuntu / WSL2 tested).
- mise installed.
- ffmpeg >= 5.0 with `afftdn`, `loudnorm`, `silencedetect` filters.
- git.
- NVIDIA GPU (RTX 4090 target) with CUDA 12.1+ drivers.
- ~5 GB disk for RVC weights and venv.

**Production:**
- Not deployed; this is a local developer tool. "Production" = the same developer box used for training, driven by `scripts/check.sh` and `src/doctor.py` as the readiness oracle.

---

*Stack analysis: 2026-04-09*
