# Architecture

**Analysis Date:** 2026-04-09

## Pattern Overview

**Overall:** Two-venv subprocess-orchestrated CLI pipeline.

The project is a thin Python CLI that orchestrates a three-stage audio pipeline:
`text -> Edge-TTS (mp3) -> ffmpeg (canonical wav) -> RVC (final wav)`.

Our code owns stages 1 and 2 and the glue. Stage 3 is delegated to the upstream RVC project, which lives in its own isolated virtualenv and is only ever invoked as a subprocess. There is no shared Python import path between the two halves.

**Key Characteristics:**
- **Hard venv boundary.** `./.venv` (our code) and `./rvc/.venv` (torch + fairseq) never share interpreter state. The only cross-boundary primitive is `subprocess.run(..., cwd=RVC_DIR)` in `src/generate.py:main`.
- **Doctor-first.** Every entry point calls the relevant subset of `src/doctor.py` checks before doing work. Misconfigured systems fail fast with actionable `fix_hint` strings instead of midway crashes.
- **Subprocess-wrapper discipline.** All ffmpeg calls go through `src/ffmpeg_utils.py:run_ffmpeg`. No `shell=True`, no bare `subprocess.run("ffmpeg ...")` anywhere. The RVC invocation has its own builder, `src/generate.py:build_rvc_subprocess_cmd`, that is a pure function (unit-tested).
- **Vendoring over dependency creep.** `src/slicer2.py` is vendored from `audio-slicer` 1.0.1 rather than pulling in librosa/pydub. `rvc/` is a pinned clone (commit `7ef19867780cf703841ebafb565a4e47d1ea86ff`), not a submodule, not a package.
- **No framework runtime.** No Flask/FastAPI, no database, no scheduler, no async server. Just `typer` CLIs and subprocess calls.

## Layers

**Entry layer - CLI commands (`src/*.py` as scripts):**
- Purpose: Parse arguments, validate inputs, call the check layer, then call the pipeline layer.
- Location: `src/doctor.py`, `src/generate.py`, `src/preprocess.py`.
- Pattern: Each file defines a `typer.Typer()` app with a single `@app.command() def main(...)`. Running `python src/<x>.py` invokes `app()`.
- Import fix-up: `src/generate.py` and `src/preprocess.py` both prepend `PROJECT_ROOT` to `sys.path` before importing `src.*`. This is why `pyproject.toml` lists those two files under `per-file-ignores = ["E402"]`.
- Depends on: check layer, pipeline-function layer, `typer`, `rich`, `python-dotenv`.
- Used by: shell scripts in `scripts/`, end users.

**Check layer (`src/doctor.py`):**
- Purpose: Single source of truth for "is this machine ready?". Every other entry point composes a subset of these checks.
- Location: `src/doctor.py`.
- Public surface: `CheckResult` dataclass (aliased `Check`), plus functions `check_mise`, `check_python_version`, `check_ffmpeg`, `check_ffmpeg_filters`, `check_git`, `check_nvidia_smi`, `check_rvc_cloned`, `check_rvc_venv`, `check_rvc_weights`, `check_model_file`, `check_edge_tts_importable`, `check_slicer2_importable`, `check_rvc_torch_cuda`.
- Constants exported for reuse: `PROJECT_ROOT`, `RVC_DIR`, `RVC_VENV_PYTHON`, `ROOT_VENV`, `MODELS_DIR`, `MIN_FFMPEG_VERSION`, `REQUIRED_PYTHON`, `REQUIRED_FFMPEG_FILTERS`.
- Depends on: `subprocess`, `shutil`, `typer`, `rich`.
- Used by: `src/generate.py` (pre-flight before the pipeline), `src/preprocess.py` (pre-flight), `scripts/setup_rvc.sh` (`--system-only` at start, `--rvc-only` at end), `scripts/check.sh`, unit tests.

**Pipeline-function layer (pure/unit-testable helpers):**
- Purpose: Encapsulate pipeline primitives as small functions with no side effects on module import.
- Key functions:
  - `src/preprocess.py:build_canonical_args`, `build_denoise_args`, `build_loudnorm_args` - pure ffmpeg arg builders.
  - `src/preprocess.py:_slice_with_slicer2` - imports `src.slicer2.Slicer` and drops low-RMS/clipped clips.
  - `src/preprocess.py:run_preprocess` - orchestrator for the whole preprocess stage (not a CLI).
  - `src/generate.py:build_rvc_subprocess_cmd` - pure argv builder for RVC's `tools/infer_cli.py`.
  - `src/generate.py:_ensure_rvc_weight_staged` - copies `models/<name>.pth` into `rvc/assets/weights/` on mtime miss.
  - `src/generate.py:_slugify`, `_default_output_path`, `_tail` - utility helpers.
- Depends on: `numpy`, `soundfile`, stdlib.
- Used by: CLI `main()` functions in the same file; unit tests.

**Subprocess-wrapper layer (`src/ffmpeg_utils.py`):**
- Purpose: The only place in the codebase that directly shells out to ffmpeg. Provides `FfmpegError` and the `run_ffmpeg(args, *, context, expected_output, binary="ffmpeg")` entry point.
- Contract: prepends `-hide_banner -loglevel error -y`, captures stdout+stderr, verifies `expected_output` exists and is non-empty, raises with pipeline-stage context on failure.
- Used by: `src/preprocess.py` (3 calls per input file), `src/generate.py` (1 call per generation).

**Vendored slicer layer (`src/slicer2.py`):**
- Purpose: Audio silence-based slicer. Copied from `audio-slicer` 1.0.1 (MIT). Excluded from ruff via `pyproject.toml:extend-exclude`.
- Used by: `src/preprocess.py:_slice_with_slicer2` (with a fallback `import slicer2` for direct script invocation).

**Shell-orchestration layer (`scripts/`):**
- Purpose: Operations that must span venvs or do large side-effectful work (clone a repo, download models, launch a webui).
- Files:
  - `scripts/setup_rvc.sh` - clones RVC at the pinned commit, creates `rvc/.venv` seeded from `./.venv/bin/python`, pins `pip<24.1`, installs torch 2.1.2 + CU121, installs RVC requirements, pins `gradio_client==0.2.7` and `matplotlib==3.7.3`, downloads weights, calls `doctor.py --rvc-only`. Tees to `scripts/setup_rvc.log` via a self-re-exec pattern that preserves `set -e`.
  - `scripts/launch_rvc_webui.sh` - `exec rvc/.venv/bin/python infer-web.py --pycmd <rvc_py> --port 7865`.
  - `scripts/install_model.sh <name>` - copies `rvc/assets/weights/<name>.pth` and `rvc/logs/<name>/added_*.index` into `./models/`.
  - `scripts/check.sh` - doctor + pytest unit + pytest integration (`-m "not network"`) + `ruff check` + `ruff format --check`.

## Data Flow

**Generate flow (`src/generate.py`):**

1. Load `.env` (`DEFAULT_MODEL`, `DEFAULT_EDGE_VOICE`, `DEVICE`).
2. Parse typer args. Handle early exits: `--list-voices`, `--smoke-test`, empty text, mutually exclusive flags.
3. Pre-flight: `check_ffmpeg`, `check_edge_tts_importable`, `check_rvc_cloned`, `check_rvc_venv`, `check_model_file(model)`. Non-ok -> exit 1.
4. Resolve output path (`out` or `output/<timestamp>_<slug>.wav`). Create `output/`.
5. `--dry-run` short-circuit.
6. Stage 1: `_generate_edge_tts` writes `tmpdir/edge_tts.mp3`.
7. Stage 2: `run_ffmpeg` converts mp3 -> `tmpdir/edge_tts.wav` (44.1 kHz mono s16).
8. Stage 3: `_ensure_rvc_weight_staged(model)` stages the weight, `build_rvc_subprocess_cmd` assembles argv, `subprocess.run(cmd, cwd=RVC_DIR, capture_output=True)` invokes RVC.
9. Failure path: last 20 lines of stderr printed via `_tail`; extra hint on `CUDA out of memory`; if `--keep-intermediate`, copy the tmp wav to `output/_last_intermediate.wav`.
10. Verify final output exists, size >= 1024 bytes, readable by `soundfile.read`. Compute duration, print summary.

**Preprocess flow (`src/preprocess.py:run_preprocess`):**

1. Validate `input_dir` exists and contains at least one file in `AUDIO_EXTS`.
2. Wipe and recreate `output_dir` (idempotent).
3. For each input, in a single shared `tempfile.TemporaryDirectory`:
   - `build_canonical_args` -> `run_ffmpeg` -> 44.1 kHz mono s16 WAV.
   - `build_denoise_args` -> `run_ffmpeg` -> `highpass=75,lowpass=15000,afftdn=nr=12`.
   - `build_loudnorm_args` -> `run_ffmpeg` -> `loudnorm=I=<target>:TP=-1:LRA=11`.
   - `_slice_with_slicer2` -> slice into 3-15s clips, drop clips with `rms < 0.005` or `peak > 0.9999`.
4. Accumulate total clips and seconds. Print summary via `rich.Console`.

**Setup flow (`scripts/setup_rvc.sh`):**

1. Self re-exec piped through `tee -a scripts/setup_rvc.log` (preserves exit codes).
2. `doctor.py --system-only` (ffmpeg, ffmpeg filters, git, nvidia-smi, mise, python 3.10).
3. Verify `./.venv/bin/python` is Python 3.10.
4. Clone RVC (or fetch if present); check out the pinned commit; assert HEAD matches.
5. Create `rvc/.venv` via `./.venv/bin/python -m venv`.
6. Install `pip<24.1` inside `rvc/.venv`.
7. Install torch 2.1.2 + torchaudio + torchvision from the CUDA 12.1 wheel index.
8. Install `-r rvc/requirements.txt`, then pin `gradio_client==0.2.7` and `matplotlib==3.7.3`.
9. Run `rvc/tools/download_models.py`.
10. Smoke-test: `python -c "import torch; assert torch.cuda.is_available()"`.
11. `doctor.py --rvc-only`.

**State management:**
- No in-process state. No singletons. No caches. Every run re-reads `.env`, re-runs the checks, re-stages the RVC weight (on mtime miss), and writes fresh output.
- The only persistent state is on disk: `models/`, `rvc/assets/weights/`, `rvc/logs/`, `output/`, `dataset/processed/`.

## Key Abstractions

**`CheckResult` (dataclass, `src/doctor.py`):**
- Purpose: uniform shape for every health check. Fields: `name`, `ok`, `detail`, `fix_hint`.
- Aliased as `Check` for tests.
- Every check function returns one instance and never raises.

**`FfmpegError` / `FfmpegResult` (`src/ffmpeg_utils.py`):**
- Purpose: a single exception class for all ffmpeg failure modes (non-zero exit, missing output, empty output, binary not found). Always includes pipeline-stage `context` and the full command in the message.

**Pure arg-builder functions:**
- `build_canonical_args`, `build_denoise_args`, `build_loudnorm_args` (preprocess), `build_rvc_subprocess_cmd` (generate).
- All return `list[str]` and have zero side effects. This is what makes the CLI layer thin and the test layer small.

**Pinned-commit RVC clone:**
- Not quite an abstraction, but an architectural choice: RVC is treated as an opaque binary dependency, pinned by commit hash in a shell script, invoked only as a subprocess. There is no Python-level abstraction over RVC - the argv is the interface.

## Entry Points

**`python src/doctor.py [--system-only|--rvc-only|--runtime|--model <name>]`:**
- Location: `src/doctor.py:main`.
- Triggers: developer invocation, `scripts/check.sh`, `scripts/setup_rvc.sh`.
- Responsibilities: run the selected group(s) of checks, print a `rich.Table`, exit 0/1.

**`python src/preprocess.py [--input DIR] [--output DIR] [--min-len S] [--max-len S] [--target-lufs N] [--dry-run]`:**
- Location: `src/preprocess.py:main`.
- Triggers: developer invocation after recording raw audio.
- Responsibilities: pre-flight ffmpeg checks, then delegate to `run_preprocess`.

**`python src/generate.py [TEXT] [--text-file F] [--out PATH] [--model NAME] [--tts-voice V] [--pitch N] [--index-rate F] [--filter-radius N] [--device D] [--keep-intermediate] [--smoke-test] [--list-voices] [--dry-run] [--verbose]`:**
- Location: `src/generate.py:main`.
- Triggers: end-user voice-cloning invocation.
- Responsibilities: pre-flight, resolve text source, run the 3-stage pipeline, verify output.

**`./scripts/setup_rvc.sh [--force]`:**
- Triggers: one-time setup (or re-run with `--force` to wipe and redo).
- Responsibilities: clone RVC, set up `rvc/.venv`, install pinned torch + RVC deps, download weights, verify CUDA.

**`./scripts/launch_rvc_webui.sh`:**
- Triggers: training the voice model (browser-based UI).

**`./scripts/install_model.sh <experiment_name>`:**
- Triggers: after training completes, to promote the model to `./models/`.

**`./scripts/check.sh`:**
- Triggers: developer sanity check. The project's "CI equivalent."

## Error Handling

**Strategy:** explicit exit codes + actionable stderr. No exception propagation to users.

**Exit code convention (enforced in `src/generate.py` and echoed in the README):**
- `0` success
- `1` configuration / setup error (missing venv, missing model, missing binary) - fixable by running doctor or setup scripts.
- `2` user input error (empty text, mutually exclusive flags, unknown voice, missing text file).
- `3` runtime error (ffmpeg, edge-tts, RVC subprocess failure) - usually the last 20 lines of stderr are included.

**Patterns:**
- **Check-before-run.** Every pipeline entry point calls the relevant doctor checks first and exits 1 with `fix_hint` on failure. Never crashes mid-pipeline with an ImportError.
- **Wrap-and-context.** `run_ffmpeg` wraps the subprocess and converts every failure mode into `FfmpegError` with a `context` string naming the pipeline stage (`canonical`, `denoise`, `loudnorm`, `tts-to-canonical`).
- **Tail on failure.** RVC failures print only the last 20 stderr lines by default (`_tail`); `--verbose` prints the whole thing. A `CUDA out of memory` substring triggers an extra hint.
- **No broad `except Exception`.** Catches are narrow (`FfmpegError`, `PreprocessError`, `ImportError`). The one broad `except Exception` is on `sf.read` verification of the final output, converted to exit 3.
- **Re-raise with `from exc`** is used throughout so the original traceback is preserved when `--verbose` or pytest surfaces it.

## Cross-Cutting Concerns

**Logging:**
- `rich.Console` for user-facing progress and summaries. No stdlib `logging` setup. The only persistent log is `scripts/setup_rvc.log`, populated by `tee`.

**Validation:**
- Argument validation is done in the CLI `main()` via typer option declarations and explicit `if` branches that echo to stderr and raise `typer.Exit(code=2)`.
- Data validation lives in the pipeline: `check_model_file`, the `AUDIO_EXTS` filter in preprocess, RMS and peak thresholds in the slicer loop, `expected_output` size checks in `run_ffmpeg`, a `>= 1024` byte floor on the final generated WAV.

**Authentication:**
- None. Not applicable.

**Configuration loading:**
- `python-dotenv` `load_dotenv()` is called exactly once, at module import of `src/generate.py`. Other modules read environment variables directly but do not load `.env` themselves, meaning `preprocess.py` and `doctor.py` run with whatever the shell provides.

**Path conventions:**
- `PROJECT_ROOT = Path(__file__).resolve().parent.parent` in both `src/doctor.py` and `src/generate.py`. These are the two files whose constants are imported elsewhere. `RVC_DIR`, `RVC_VENV_PYTHON`, `ROOT_VENV`, `MODELS_DIR` all derive from it in `doctor.py` and are re-imported by `generate.py`.

---

*Architecture analysis: 2026-04-09*
