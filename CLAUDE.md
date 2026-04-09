<!-- GSD:project-start source:PROJECT.md -->
## Project

**train_audio_model**

A self-contained voice-cloning toolkit built on RVC (Retrieval-based Voice Conversion). Today it ships an Edge-TTS → ffmpeg → RVC inference CLI for generating speech from a trained voice. The next milestone makes the *training* half of the workflow first-class so an entire model can be trained end-to-end on a rented GPU pod from a clean Ubuntu box.

**Core Value:** A single user — me — can rent a GPU pod, run two bash scripts, and walk away with a downloadable `.pth` + `.index` voice model trained from raw audio I provided.

### Constraints

- **Tech stack**: Python 3.10 only (pinned by `pyproject.toml` and `.mise.toml`). RVC venv stays on torch 2.1.2 + CUDA 12.1 + fairseq 0.12.2 + gradio 3.34.0 + `pip<24.1`. Reason: pinned upstream RVC commit only works with this combination; deviating breaks fairseq install.
- **Tech stack**: RVC venv pins `pip<24.1`. Reason: fairseq 0.12.2 has legacy PEP 440 metadata that pip 24.1+ rejects.
- **Tech stack**: `mise` is the Python version manager. Reason: user preference and existing `.mise.toml`.
- **Compatibility**: New training code must coexist with existing inference code without changing it. The `src/generate.py` pipeline must keep working byte-for-byte.
- **Compatibility**: All ffmpeg shell-outs must go through `src/ffmpeg_utils.py:run_ffmpeg`. All RVC subprocess invocations must follow the pattern in `src/generate.py:build_rvc_subprocess_cmd` (pure arg builder + thin runner).
- **Compatibility**: Subprocess discipline — no `shell=True`, no bare `subprocess.run("...")` strings, no shared Python imports across the venv boundary.
- **Performance**: Training script must be resumable — interrupted runs (`SIGTERM`, pod kill, network drop) must continue from the last checkpoint on next invocation, not restart.
- **Dependencies**: RVC clone stays pinned to commit `7ef19867780cf703841ebafb565a4e47d1ea86ff`. Vendoring over dependency creep — no new pip packages in the app venv unless absolutely necessary.
- **Dependencies**: No new framework runtime (no Flask, no FastAPI, no async server, no DB, no scheduler). Stays a thin typer CLI plus shell glue.
- **Operational**: No interactive prompts in any pod-side script — every input is a CLI flag, env var, or config file. Idle interactive prompts on a billing pod are unacceptable.
- **Operational**: No provider-specific code or SDKs. The contract is "Linux + NVIDIA driver + bash". Provider integrations are documented patterns only.
- **Security**: `.env` is never committed. `settings.local.json` is never committed.
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- Python 3.10 (exact) - All application code in `src/`, tests in `tests/`. Pinned by `pyproject.toml` (`requires-python = "==3.10.*"`) and `.mise.toml` (`python = "3.10"`).
- Bash - Setup and orchestration scripts in `scripts/` (`setup_rvc.sh`, `launch_rvc_webui.sh`, `install_model.sh`, `check.sh`).
- None. The `rvc/` directory is a vendored upstream clone (RVC-Project/Retrieval-based-Voice-Conversion-WebUI) pinned to commit `7ef19867780cf703841ebafb565a4e47d1ea86ff` (2024-11-24). It is excluded from lint/format and treated as a black-box subprocess.
## Runtime
- CPython 3.10 on Linux (tested on Ubuntu/WSL2 per `README.md`).
- Two isolated virtualenvs by design:
- Cross-venv communication is subprocess only. See `src/generate.py:build_rvc_subprocess_cmd`.
- `mise` - declared in `.mise.toml`. The `doctor.check_mise()` function verifies availability.
- `pip` (standard). Application venv installs via `pip install -e ".[dev]"`.
- RVC venv pins `pip<24.1` in `scripts/setup_rvc.sh` because fairseq 0.12.2 has legacy PEP 440 metadata pip 24.1+ rejects.
- Lockfile: `requirements.txt` (frozen reference, not the install source) and `pyproject.toml` (editable install source). No `poetry.lock` / `uv.lock` / `pip-tools` output for the app venv.
## Frameworks
- `typer==0.12.3` - CLI framework for `src/doctor.py`, `src/generate.py`, `src/preprocess.py`. Pinned; `pyproject.toml` explicitly preserves `Optional[X]` (UP007 ignored) because typer 0.12.3 lacks PEP 604 support.
- `click>=8.1,<8.2` - typer dependency, kept compatible.
- `rich==13.7.1` - console output tables (`doctor._run_checks`, `generate.main --list-voices`).
- `python-dotenv==1.0.1` - loads `.env` in `src/generate.py`.
- `edge-tts>=7.0,<8` - Microsoft Edge TTS client (stage 1 of the pipeline in `src/generate.py`).
- `soundfile==0.12.1` - WAV I/O in `src/preprocess.py` and `src/generate.py`.
- `numpy>=1.24,<2.0` (installed as `1.26.4`) - RMS/peak clip filtering in `src/preprocess.py`.
- **Vendored:** `src/slicer2.py` - copied from `audio-slicer` 1.0.1 (MIT). Avoids a librosa/pydub dependency chain. Excluded from ruff via `extend-exclude`.
- External binary: `ffmpeg >= 5.0` with `afftdn`, `loudnorm`, `silencedetect` filters. Verified by `src.doctor.check_ffmpeg` and `check_ffmpeg_filters`.
- `pytest==8.2.0`
- `pytest-mock==3.14.0`
- Custom pytest markers in `pyproject.toml`: `network`, `gpu`. Default `addopts = "-m 'not network and not gpu'"`.
- `ruff==0.4.4` - both linter and formatter. Config in `pyproject.toml` (`line-length = 100`, `target-version = "py310"`, selected rules `E, F, W, I, B, UP, N, SIM`). Excludes `rvc/` and `src/slicer2.py`.
- `torch==2.1.2` + `torchaudio==2.1.2` + `torchvision==0.16.2` from the CUDA 12.1 wheel index.
- `gradio==3.34.0` with `gradio_client==0.2.7` (pinned by `setup_rvc.sh` to prevent the `media_data` symbol mismatch).
- `matplotlib==3.7.3` (pinned by `setup_rvc.sh` because RVC calls the removed `FigureCanvasAgg.tostring_rgb()`).
- `fairseq 0.12.2` (indirect, via RVC's `requirements.txt`).
## Key Dependencies
- `edge-tts` - stage 1 TTS. Network-dependent; failure mode is Edge-TTS 403 after Microsoft token rotation (documented in `README.md` troubleshooting).
- `soundfile` - every audio I/O path touches it. Requires libsndfile.
- `typer` - every entry point is a `typer.Typer()` app.
- `torch` with CUDA - mandatory. `doctor.check_rvc_torch_cuda` asserts `torch.cuda.is_available()` is True.
- RVC pretrained weights: `rvc/assets/hubert/hubert_base.pt` and `rvc/assets/rmvpe/rmvpe.pt`. Downloaded by `rvc/tools/download_models.py` during setup. Checked by `doctor.check_rvc_weights`.
- `ffmpeg` (system binary) - used for all audio transcoding and filter chains.
- `git` - required by `setup_rvc.sh` to clone and checkout the pinned RVC commit.
- `nvidia-smi` / NVIDIA drivers - required for GPU inference. Checked by `doctor.check_nvidia_smi`.
## Configuration
- `.env` file (git-ignored) loaded by `src/generate.py` via `python-dotenv`. Template at `.env.example`.
- Keys (non-secret, user preferences):
- `.env` has no secrets, but it is git-ignored per `.gitignore`.
- `pyproject.toml` - PEP 621 project metadata, ruff, pytest config. Only build file for the app package.
- `requirements.txt` - flat frozen list (not used for install; reference only).
- `.mise.toml` - pins Python 3.10.
- No `setup.py`, no `Makefile`. Orchestration lives in `scripts/check.sh`.
## Platform Requirements
- Linux (Ubuntu / WSL2 tested).
- mise installed.
- ffmpeg >= 5.0 with `afftdn`, `loudnorm`, `silencedetect` filters.
- git.
- NVIDIA GPU (RTX 4090 target) with CUDA 12.1+ drivers.
- ~5 GB disk for RVC weights and venv.
- Not deployed; this is a local developer tool. "Production" = the same developer box used for training, driven by `scripts/check.sh` and `src/doctor.py` as the readiness oracle.
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- Lowercase with underscores: `ffmpeg_utils.py`, `preprocess.py`, `generate.py`
- Exception: Vendored third-party code kept with original names: `slicer2.py`
- snake_case: `run_ffmpeg()`, `check_ffmpeg()`, `build_canonical_args()`, `_slugify()`
- Private/internal functions prefixed with underscore: `_slice_with_slicer2()`, `_default_output_path()`, `_run_checks()`
- Helper functions prefixed with underscore when internal to module: `_tail()`, `_ensure_model_in_rvc_weights()`
- PascalCase: `FfmpegError`, `FfmpegResult`, `PreprocessError`, `CheckResult`
- Custom exceptions inherit from appropriate base: `FfmpegError(RuntimeError)`, `PreprocessError(RuntimeError)`
- snake_case for local/module variables: `resolved_text`, `final_out`, `written`
- UPPER_CASE for constants: `AUDIO_EXTS`, `CANONICAL_SR`, `SILENCE_RMS_THRESHOLD`, `PEAK_CLIPPING_THRESHOLD`, `DEFAULT_MODEL`, `DEFAULT_EDGE_VOICE`
- Module-level variables starting with underscore for internal state: `_PROJECT_ROOT`, `RVC_VENV_PYTHON`, `MODELS_DIR`
- PascalCase for type names: `Path`, `CheckResult`, `FfmpegResult`, `Optional`, `list[str]`
- Modern union syntax with `|` (requires `from __future__ import annotations`): `tuple[int, int, int] | None`
- Optional use `Optional[Type]` when required by typer 0.12.3 (which lacks PEP 604 support)
## Code Style
- Line length: 100 characters (configured in ruff)
- Tool: ruff (built-in formatter)
- Python version: 3.10.x (strict enforcement via `requires-python = "==3.10.*"`)
- Tool: ruff
- Config file: `pyproject.toml` under `[tool.ruff]` and `[tool.ruff.lint]`
- Key rules enabled: E (pycodestyle), F (pyflakes), W (warnings), I (isort), B (flake8-bugbear), UP (pyupgrade), N (pep8-naming), SIM (flake8-simplify)
- Line length rule (E501) ignored (handled by formatter)
- PEP 604 unions (UP007) ignored (typer 0.12.3 requires `Optional[X]`)
- Typer idiom (B008) ignored (typer.Option() in defaults is standard)
- `src/generate.py`: Allow E402 (module-level import not at top) for sys.path fixup
- `src/preprocess.py`: Allow E402 (module-level import not at top) for sys.path fixup
- `rvc/` directory: Completely excluded from ruff checks (third-party code)
- `src/slicer2.py`: Excluded from ruff (vendored upstream)
## Import Organization
- No path aliases configured
- Relative imports within src/ use full module path: `from src.doctor import ...` not `from .doctor import ...`
- This allows files to be run as scripts or modules
- Files that may run as scripts include sys.path fixup before internal imports:
- This pattern seen in `src/preprocess.py` and `src/generate.py` (marked with E402 exception)
- Fallback imports for module/script execution: try qualified import first, then bare import (see `src/slicer2.py` import in `src/preprocess.py` line 111-113)
## Module Docstrings
- Triple-quoted docstring at module top (before `from __future__`)
- Describes purpose and any important caveats
- Pipeline-oriented modules list steps: `1. Step\n2. Step\n3. Step`
- Example: `src/ffmpeg_utils.py` documents that all ffmpeg calls must go through `run_ffmpeg()`
- Example: `src/preprocess.py` documents the full pipeline and that it's idempotent
## Function Design
- Google-style docstrings with Args, Returns, Raises sections
- Example from `src/ffmpeg_utils.py`:
- Use keyword-only parameters (after `*`) for named arguments: `run_ffmpeg(args, *, context, expected_output, binary="ffmpeg")`
- Positional parameters only when essential (rare)
- Type hints always included: `args: list[str]`, `context: str`, `expected_output: Path`
- Optional parameters have defaults: `binary: str = "ffmpeg"`
- Single dataclass for multiple return values: `FfmpegResult` with fields `stdout: str`, `stderr: str`
- Exceptions raised for errors instead of returning None/error codes
- Functions either succeed and return data, or raise an exception with context
- Functions are concise, typically 10-40 lines
- Complex pipelines broken into helper functions with clear names
- Section comments delineate major function groups: `# ---------- System checks ----------`
## Error Handling
- Custom exception classes inherit from `RuntimeError` or `RuntimeError` subclass: `FfmpegError(RuntimeError)`, `PreprocessError(RuntimeError)`
- Exception messages include context tags in brackets: `[{context}] ffmpeg exited with code...`
- Full error info provided: command invoked, exit code, stderr output, file paths
- Example from `src/ffmpeg_utils.py`:
- 0: success
- 1: config/setup error (missing model, missing venv, missing ffmpeg)
- 2: user input error (empty text, mutually-exclusive flags, bad voice)
- 3: runtime error (ffmpeg, edge-tts, subprocess failure)
- Documented at module top: `src/generate.py` documents all exit codes
- Always use `check=False` and inspect `returncode` manually
- Capture both stdout and stderr: `capture_output=True, text=True`
- Provide full command and stderr in error messages for debugging
## Logging
- Create console at module top if needed: `console = Console()` in `src/generate.py`, `src/doctor.py`
- Use console.print() for output: `console.print(table)`, `console.print(f"[green]✓[/green] {detail}")`
- Rich markup for colors/styling: `[green]OK[/green]`, `[red]FAIL[/red]`, `[cyan]Check[/cyan]`
- CLI uses typer.echo() for simple output: `typer.echo("[error] message", err=True)`
- Error messages to stderr: `typer.echo(..., err=True)`
- Doctor checks: always output result table
- CLI validation: output error with exit code
- Success: minimal output (let file/dir existence confirm)
- Debug info: only with `--verbose` flag
## Comments
- Section headers: `# ---------- System checks ----------`
- Complex logic that isn't obvious from names
- Non-obvious regex patterns or ffmpeg filter logic
- Workarounds or vendor-specific quirks
- Obvious code: `name = "ffmpeg"  # set the name`
- Self-documenting patterns: clear function names, type hints eliminate need
- Python uses docstrings, not comments
- Docstrings are mandatory for: public functions, classes, modules
- Private functions may omit docstrings if name is clear
## Constants & Configuration
- Module-level constants are UPPER_CASE: `CANONICAL_SR = 44100`, `SILENCE_RMS_THRESHOLD = 0.005`
- Grouped with other module constants at module top (after class definitions, before functions)
- Magic numbers always become named constants
- Environment variables via python-dotenv: `from dotenv import load_dotenv; load_dotenv()`
- Environment reads: `DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "myvoice_v1")`
- Fallback values always provided: `os.environ.get("KEY", default_value)`
- Config sourced at module level and passed to functions/CLI via parameters (not global access)
## Type Hints
- Parameters always typed: `def run_ffmpeg(args: list[str], *, context: str) -> FfmpegResult:`
- Return types always specified
- Union types use modern syntax: `tuple[int, int, int] | None` (via `from __future__ import annotations`)
- Use Optional[] only when required by framework: `typer.Option(None, ...)` requires `Optional[str]` parameter
- Use dataclasses for structured returns: `@dataclass class FfmpegResult:`
- Avoid bare `dict` and `list`; use `dict[str, str]` and `list[Path]`
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern Overview
- **Hard venv boundary.** `./.venv` (our code) and `./rvc/.venv` (torch + fairseq) never share interpreter state. The only cross-boundary primitive is `subprocess.run(..., cwd=RVC_DIR)` in `src/generate.py:main`.
- **Doctor-first.** Every entry point calls the relevant subset of `src/doctor.py` checks before doing work. Misconfigured systems fail fast with actionable `fix_hint` strings instead of midway crashes.
- **Subprocess-wrapper discipline.** All ffmpeg calls go through `src/ffmpeg_utils.py:run_ffmpeg`. No `shell=True`, no bare `subprocess.run("ffmpeg ...")` anywhere. The RVC invocation has its own builder, `src/generate.py:build_rvc_subprocess_cmd`, that is a pure function (unit-tested).
- **Vendoring over dependency creep.** `src/slicer2.py` is vendored from `audio-slicer` 1.0.1 rather than pulling in librosa/pydub. `rvc/` is a pinned clone (commit `7ef19867780cf703841ebafb565a4e47d1ea86ff`), not a submodule, not a package.
- **No framework runtime.** No Flask/FastAPI, no database, no scheduler, no async server. Just `typer` CLIs and subprocess calls.
## Layers
- Purpose: Parse arguments, validate inputs, call the check layer, then call the pipeline layer.
- Location: `src/doctor.py`, `src/generate.py`, `src/preprocess.py`.
- Pattern: Each file defines a `typer.Typer()` app with a single `@app.command() def main(...)`. Running `python src/<x>.py` invokes `app()`.
- Import fix-up: `src/generate.py` and `src/preprocess.py` both prepend `PROJECT_ROOT` to `sys.path` before importing `src.*`. This is why `pyproject.toml` lists those two files under `per-file-ignores = ["E402"]`.
- Depends on: check layer, pipeline-function layer, `typer`, `rich`, `python-dotenv`.
- Used by: shell scripts in `scripts/`, end users.
- Purpose: Single source of truth for "is this machine ready?". Every other entry point composes a subset of these checks.
- Location: `src/doctor.py`.
- Public surface: `CheckResult` dataclass (aliased `Check`), plus functions `check_mise`, `check_python_version`, `check_ffmpeg`, `check_ffmpeg_filters`, `check_git`, `check_nvidia_smi`, `check_rvc_cloned`, `check_rvc_venv`, `check_rvc_weights`, `check_model_file`, `check_edge_tts_importable`, `check_slicer2_importable`, `check_rvc_torch_cuda`.
- Constants exported for reuse: `PROJECT_ROOT`, `RVC_DIR`, `RVC_VENV_PYTHON`, `ROOT_VENV`, `MODELS_DIR`, `MIN_FFMPEG_VERSION`, `REQUIRED_PYTHON`, `REQUIRED_FFMPEG_FILTERS`.
- Depends on: `subprocess`, `shutil`, `typer`, `rich`.
- Used by: `src/generate.py` (pre-flight before the pipeline), `src/preprocess.py` (pre-flight), `scripts/setup_rvc.sh` (`--system-only` at start, `--rvc-only` at end), `scripts/check.sh`, unit tests.
- Purpose: Encapsulate pipeline primitives as small functions with no side effects on module import.
- Key functions:
- Depends on: `numpy`, `soundfile`, stdlib.
- Used by: CLI `main()` functions in the same file; unit tests.
- Purpose: The only place in the codebase that directly shells out to ffmpeg. Provides `FfmpegError` and the `run_ffmpeg(args, *, context, expected_output, binary="ffmpeg")` entry point.
- Contract: prepends `-hide_banner -loglevel error -y`, captures stdout+stderr, verifies `expected_output` exists and is non-empty, raises with pipeline-stage context on failure.
- Used by: `src/preprocess.py` (3 calls per input file), `src/generate.py` (1 call per generation).
- Purpose: Audio silence-based slicer. Copied from `audio-slicer` 1.0.1 (MIT). Excluded from ruff via `pyproject.toml:extend-exclude`.
- Used by: `src/preprocess.py:_slice_with_slicer2` (with a fallback `import slicer2` for direct script invocation).
- Purpose: Operations that must span venvs or do large side-effectful work (clone a repo, download models, launch a webui).
- Files:
## Data Flow
- No in-process state. No singletons. No caches. Every run re-reads `.env`, re-runs the checks, re-stages the RVC weight (on mtime miss), and writes fresh output.
- The only persistent state is on disk: `models/`, `rvc/assets/weights/`, `rvc/logs/`, `output/`, `dataset/processed/`.
## Key Abstractions
- Purpose: uniform shape for every health check. Fields: `name`, `ok`, `detail`, `fix_hint`.
- Aliased as `Check` for tests.
- Every check function returns one instance and never raises.
- Purpose: a single exception class for all ffmpeg failure modes (non-zero exit, missing output, empty output, binary not found). Always includes pipeline-stage `context` and the full command in the message.
- `build_canonical_args`, `build_denoise_args`, `build_loudnorm_args` (preprocess), `build_rvc_subprocess_cmd` (generate).
- All return `list[str]` and have zero side effects. This is what makes the CLI layer thin and the test layer small.
- Not quite an abstraction, but an architectural choice: RVC is treated as an opaque binary dependency, pinned by commit hash in a shell script, invoked only as a subprocess. There is no Python-level abstraction over RVC - the argv is the interface.
## Entry Points
- Location: `src/doctor.py:main`.
- Triggers: developer invocation, `scripts/check.sh`, `scripts/setup_rvc.sh`.
- Responsibilities: run the selected group(s) of checks, print a `rich.Table`, exit 0/1.
- Location: `src/preprocess.py:main`.
- Triggers: developer invocation after recording raw audio.
- Responsibilities: pre-flight ffmpeg checks, then delegate to `run_preprocess`.
- Location: `src/generate.py:main`.
- Triggers: end-user voice-cloning invocation.
- Responsibilities: pre-flight, resolve text source, run the 3-stage pipeline, verify output.
- Triggers: one-time setup (or re-run with `--force` to wipe and redo).
- Responsibilities: clone RVC, set up `rvc/.venv`, install pinned torch + RVC deps, download weights, verify CUDA.
- Triggers: training the voice model (browser-based UI).
- Triggers: after training completes, to promote the model to `./models/`.
- Triggers: developer sanity check. The project's "CI equivalent."
## Error Handling
- `0` success
- `1` configuration / setup error (missing venv, missing model, missing binary) - fixable by running doctor or setup scripts.
- `2` user input error (empty text, mutually exclusive flags, unknown voice, missing text file).
- `3` runtime error (ffmpeg, edge-tts, RVC subprocess failure) - usually the last 20 lines of stderr are included.
- **Check-before-run.** Every pipeline entry point calls the relevant doctor checks first and exits 1 with `fix_hint` on failure. Never crashes mid-pipeline with an ImportError.
- **Wrap-and-context.** `run_ffmpeg` wraps the subprocess and converts every failure mode into `FfmpegError` with a `context` string naming the pipeline stage (`canonical`, `denoise`, `loudnorm`, `tts-to-canonical`).
- **Tail on failure.** RVC failures print only the last 20 stderr lines by default (`_tail`); `--verbose` prints the whole thing. A `CUDA out of memory` substring triggers an extra hint.
- **No broad `except Exception`.** Catches are narrow (`FfmpegError`, `PreprocessError`, `ImportError`). The one broad `except Exception` is on `sf.read` verification of the final output, converted to exit 3.
- **Re-raise with `from exc`** is used throughout so the original traceback is preserved when `--verbose` or pytest surfaces it.
## Cross-Cutting Concerns
- `rich.Console` for user-facing progress and summaries. No stdlib `logging` setup. The only persistent log is `scripts/setup_rvc.log`, populated by `tee`.
- Argument validation is done in the CLI `main()` via typer option declarations and explicit `if` branches that echo to stderr and raise `typer.Exit(code=2)`.
- Data validation lives in the pipeline: `check_model_file`, the `AUDIO_EXTS` filter in preprocess, RMS and peak thresholds in the slicer loop, `expected_output` size checks in `run_ffmpeg`, a `>= 1024` byte floor on the final generated WAV.
- None. Not applicable.
- `python-dotenv` `load_dotenv()` is called exactly once, at module import of `src/generate.py`. Other modules read environment variables directly but do not load `.env` themselves, meaning `preprocess.py` and `doctor.py` run with whatever the shell provides.
- `PROJECT_ROOT = Path(__file__).resolve().parent.parent` in both `src/doctor.py` and `src/generate.py`. These are the two files whose constants are imported elsewhere. `RVC_DIR`, `RVC_VENV_PYTHON`, `ROOT_VENV`, `MODELS_DIR` all derive from it in `doctor.py` and are re-imported by `generate.py`.
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, or `.github/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
