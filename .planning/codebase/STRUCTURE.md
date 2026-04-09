# Codebase Structure

**Analysis Date:** 2026-04-09

## Directory Layout

```
train_audio_model/
├── .env                         # Local user config (DEFAULT_MODEL, DEFAULT_EDGE_VOICE, DEVICE) - git-ignored
├── .env.example                 # Template for .env (committed)
├── .gitignore                   # Ignores venvs, rvc/, dataset/, models/*.pth, models/*.index, output/, .env, setup_rvc.log
├── .mise.toml                   # mise tool pin: python = "3.10"
├── .venv/                       # Application virtualenv (git-ignored)
├── README.md                    # User-facing quickstart, training walkthrough, troubleshooting
├── pyproject.toml               # PEP 621 metadata, ruff config, pytest config
├── requirements.txt             # Frozen reference (not used by pip install; pyproject.toml is source of truth)
├── .planning/                   # GSD planning and codebase docs
│   └── codebase/                # This directory: STACK.md, INTEGRATIONS.md, ARCHITECTURE.md, STRUCTURE.md
├── docs/                        # Developer docs (superpowers plans & specs subdirs)
│   └── superpowers/
│       ├── plans/
│       └── specs/
├── src/                         # Application package (editable-installed as `train-audio-model`)
│   ├── __init__.py              # Empty package marker
│   ├── doctor.py                # Health-check module + CLI (single source of truth for "is the system ready?")
│   ├── ffmpeg_utils.py          # The ONLY place ffmpeg is shelled out: run_ffmpeg(), FfmpegError, FfmpegResult
│   ├── generate.py              # End-user CLI: text -> Edge-TTS -> ffmpeg -> RVC subprocess -> final wav
│   ├── preprocess.py            # Preprocessing CLI: dataset/raw -> dataset/processed
│   └── slicer2.py               # Vendored from audio-slicer 1.0.1 (MIT) - excluded from ruff
├── scripts/                     # Bash orchestration scripts
│   ├── check.sh                 # doctor + pytest + ruff (project's "CI equivalent")
│   ├── setup_rvc.sh             # Clone RVC at pinned commit, build rvc/.venv, install deps, download weights
│   ├── launch_rvc_webui.sh      # exec rvc/.venv/bin/python infer-web.py --port 7865
│   ├── install_model.sh         # Copy rvc/assets/weights/<name>.pth + rvc/logs/<name>/added_*.index into models/
│   └── setup_rvc.log            # Git-ignored setup log (appended by setup_rvc.sh)
├── tests/                       # pytest test suite
│   ├── __init__.py
│   ├── conftest.py              # Shared fixtures (fake_ffmpeg)
│   ├── unit/                    # Fast, no network, no GPU
│   │   ├── test_doctor.py
│   │   ├── test_ffmpeg_utils.py
│   │   ├── test_generate_cli.py
│   │   └── test_preprocess.py
│   └── integration/             # May need ffmpeg / network (gated by pytest markers)
│       ├── test_edge_tts.py     # Marked `network`
│       └── test_preprocess_real.py
├── dataset/                     # User audio data (git-ignored)
│   ├── raw/                     # User-supplied recordings (wav/mp3/flac/m4a/ogg/aac)
│   └── processed/               # preprocess.py output: 44.1kHz mono s16 WAV clips, 3-15s each
├── models/                      # Trained RVC models installed by scripts/install_model.sh
│   ├── <name>.pth               # Generator weights (git-ignored)
│   └── <name>.index             # Feature index (git-ignored)
├── output/                      # Generated wavs, auto-named <YYYYMMDD_HHMMSS>_<slug>.wav (git-ignored)
└── rvc/                         # Pinned clone of RVC-Project/Retrieval-based-Voice-Conversion-WebUI
                                 # commit 7ef19867780cf703841ebafb565a4e47d1ea86ff (2024-11-24)
                                 # git-ignored; managed entirely by setup_rvc.sh; treated as opaque subprocess
```

## Directory Purposes

**`src/` - application package:**
- Purpose: All first-party Python. Installed editable via `pip install -e ".[dev]"`.
- Contains: typer CLI modules, shared helpers, vendored slicer.
- Key files: `doctor.py` (checks), `generate.py` (main user CLI), `preprocess.py` (training data prep), `ffmpeg_utils.py` (subprocess wrapper), `slicer2.py` (vendored).
- Conventions:
  - Every new entry-point module uses `typer.Typer(add_completion=False)` and ends with `if __name__ == "__main__": app()`.
  - Scripts that are expected to be run as `python src/<x>.py` must prepend `PROJECT_ROOT` to `sys.path` before their `from src.* import ...` lines, and add themselves to `[tool.ruff.lint.per-file-ignores]` with `E402`.
  - Use `from __future__ import annotations` at the top of every file.

**`scripts/` - shell glue:**
- Purpose: Cross-venv operations, large side-effectful jobs (clone / download), and the project's CI-equivalent aggregator.
- Contains: only Bash scripts with `set -euo pipefail`.
- Key files: `setup_rvc.sh` (once-per-box), `check.sh` (per-commit).
- Conventions:
  - Start with `set -euo pipefail`.
  - Compute `PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"` at the top.
  - Use `$PROJECT_ROOT/.venv/bin/python` as `ROOT_PYTHON`. Do not rely on `mise` shell activation inside scripts.
  - Long-running scripts that want tee'd logs must use the self-re-exec pattern from `setup_rvc.sh` (process substitution breaks `set -e`).

**`tests/` - pytest suite:**
- Purpose: unit tests (pure functions, mocked side effects) and integration tests (real ffmpeg, optional network/GPU).
- Key files: `conftest.py` (`fake_ffmpeg` fixture), `unit/test_*.py`, `integration/test_*.py`.
- Conventions:
  - Mirror source layout: `src/doctor.py` -> `tests/unit/test_doctor.py`, etc.
  - Integration tests requiring network use `@pytest.mark.network`; GPU tests use `@pytest.mark.gpu`. Default pytest config deselects both (`addopts = "-m 'not network and not gpu'"`).

**`dataset/` - user audio:**
- Purpose: Input and intermediate audio for training.
- Subdirectories: `raw/` (user's recordings, any supported format), `processed/` (canonicalized/denoised/normalized/sliced clips from `preprocess.py`).
- Generated: `processed/` is rewritten every run (idempotent wipe).
- Committed: No. `dataset/raw/` and `dataset/processed/` are in `.gitignore`.

**`models/` - installed trained models:**
- Purpose: Trained RVC models, promoted from `rvc/` by `scripts/install_model.sh`.
- Contains: `<name>.pth` (generator), `<name>.index` (feature index).
- Committed: No. `*.pth` and `*.index` are git-ignored.

**`output/` - generated audio:**
- Purpose: Final WAVs from `src/generate.py`.
- Naming: `<YYYYMMDD_HHMMSS>_<slugified_text>.wav` from `_default_output_path`.
- Contains occasional `_intermediate_*.wav` and `_last_intermediate.wav` when `--keep-intermediate` is set or when an RVC subprocess fails.
- Committed: No. Whole directory is git-ignored.

**`rvc/` - pinned external project:**
- Purpose: Upstream RVC (Retrieval-based Voice Conversion WebUI). Used for training (WebUI) and inference (`tools/infer_cli.py`).
- Generated: Yes - by `scripts/setup_rvc.sh`.
- Committed: No. Whole directory is git-ignored.
- Do not import from `rvc.*` in Python. Cross-venv communication is subprocess only. The pinned commit is in `scripts/setup_rvc.sh:RVC_COMMIT`.
- Excluded from ruff (`extend-exclude = ["rvc", "src/slicer2.py"]`).

**`docs/` - developer docs:**
- Purpose: Planning and spec documents for the superpowers workflow.
- Subdirectories: `docs/superpowers/plans/`, `docs/superpowers/specs/`.
- Committed: `docs` is listed in `.gitignore` currently, so treat this as workspace scratch, not published docs.

**`.planning/codebase/` - this directory:**
- Purpose: GSD codebase maps (STACK, INTEGRATIONS, ARCHITECTURE, STRUCTURE, etc.).
- Consumed by: `/gsd-plan-phase` and `/gsd-execute-phase`.

## Key File Locations

**Entry Points:**
- `src/doctor.py` - health checks (also a library for other modules).
- `src/preprocess.py` - prep raw recordings into training clips.
- `src/generate.py` - produce a wav from text in the trained voice.
- `scripts/setup_rvc.sh` - one-shot environment setup.
- `scripts/launch_rvc_webui.sh` - start training UI.
- `scripts/install_model.sh` - promote a trained model to `./models/`.
- `scripts/check.sh` - run all project checks.

**Configuration:**
- `pyproject.toml` - Python package, ruff, pytest.
- `.mise.toml` - Python version pin.
- `.env` / `.env.example` - runtime user preferences.
- `requirements.txt` - frozen dependency reference (not used by installer).

**Core Logic:**
- `src/generate.py:main` - the end-to-end pipeline orchestrator.
- `src/preprocess.py:run_preprocess` - training data prep orchestrator.
- `src/generate.py:build_rvc_subprocess_cmd` - RVC invocation contract.
- `src/ffmpeg_utils.py:run_ffmpeg` - the one ffmpeg call site.
- `src/doctor.py` - every `check_*` function and `CheckResult`.

**Testing:**
- `tests/conftest.py:fake_ffmpeg` - the shared fixture that installs a controllable fake `ffmpeg` on `PATH`. Use this for any test that would otherwise shell out to real ffmpeg.
- `tests/unit/` - fast tests, default selection.
- `tests/integration/` - real ffmpeg and optionally network; use markers to gate.

## Naming Conventions

**Files:**
- `snake_case.py` for Python modules.
- `snake_case.sh` for shell scripts.
- `test_<module>.py` for pytest files, one per `src/<module>.py`.
- Trained models: `<name>_v<N>.pth` + `<name>_v<N>.index` (README convention: `myvoice_v1`).
- Generated output: `<YYYYMMDD_HHMMSS>_<slug>.wav` produced by `_default_output_path`.

**Directories:**
- Lowercase single words where possible (`src`, `tests`, `scripts`, `models`, `output`, `dataset`, `docs`).
- Test subdirectories by type: `tests/unit/`, `tests/integration/`.

**Python identifiers (as enforced by ruff `N` rules):**
- Functions: `snake_case`.
- Classes: `PascalCase` (`CheckResult`, `FfmpegError`, `FfmpegResult`, `PreprocessError`).
- Constants: `UPPER_SNAKE_CASE` (`PROJECT_ROOT`, `CANONICAL_SR`, `MIN_FFMPEG_VERSION`, `AUDIO_EXTS`).
- Private helpers: leading underscore (`_slugify`, `_tail`, `_ensure_rvc_weight_staged`, `_slice_with_slicer2`).

## Where to Add New Code

**New CLI command (new pipeline stage or new tool):**
- Primary code: `src/<new_tool>.py`.
- Template:
  - `from __future__ import annotations`
  - sys.path fix-up if it will be run as `python src/<new_tool>.py` (and add the file to `pyproject.toml:[tool.ruff.lint.per-file-ignores]` under `E402`).
  - Define a `typer.Typer(add_completion=False, help="...")` app.
  - Call the relevant `src.doctor.check_*` functions before doing work.
  - Exit with the project's code convention: 1 for config, 2 for user input, 3 for runtime.
- Tests: `tests/unit/test_<new_tool>.py` (mirror name). Network/GPU gated via markers if needed.
- Shell entry (optional): `scripts/<new_tool>.sh` wrapping the Python invocation.

**New health check:**
- Primary code: add `check_<thing>()` to `src/doctor.py` returning `CheckResult`.
- Register it in the appropriate list inside `doctor.main` (`system_checks`, `rvc_checks`, or `runtime_checks`).
- Import and call it in the pre-flight block of any entry point that depends on it (`src/generate.py` and `src/preprocess.py` both currently have explicit pre-flight loops).
- Tests: add cases to `tests/unit/test_doctor.py`.

**New ffmpeg pipeline stage:**
- Add a pure `build_<stage>_args(input_path, output_path, ...) -> list[str]` helper next to the existing builders in `src/preprocess.py` (or in `src/ffmpeg_utils.py` if it is shared across pipelines).
- Call it through `run_ffmpeg(..., context="<stage>", expected_output=...)`. Do NOT shell out to ffmpeg directly.
- Tests: unit-test the arg builder as a pure function; integration-test the stage with the `fake_ffmpeg` fixture from `conftest.py`.

**New RVC interaction:**
- Add the argv builder as a pure function in `src/generate.py` (mirror `build_rvc_subprocess_cmd`).
- Invoke via `subprocess.run(cmd, cwd=RVC_DIR, capture_output=True, text=True, check=False)`.
- Use `RVC_VENV_PYTHON` and `RVC_DIR` constants imported from `src.doctor`. Do not hard-code paths.
- Never `import` anything from `rvc/` inside the application venv - it cannot resolve and that is by design.

**New utility shared across modules:**
- Put it in `src/ffmpeg_utils.py` (if ffmpeg-adjacent) or create a new `src/<name>_utils.py`. Keep modules single-purpose.
- Avoid creating a generic `src/utils.py` - the current layout prefers topical small modules.

**New test:**
- Unit: `tests/unit/test_<module>.py`, mirroring `src/<module>.py`. Use `pytest-mock` and the `fake_ffmpeg` fixture. Default addopts skip `network` and `gpu` markers.
- Integration: `tests/integration/test_<scenario>.py`. Mark network calls with `@pytest.mark.network`; mark GPU calls with `@pytest.mark.gpu`.

## Special Directories

**`rvc/`:**
- Purpose: upstream Retrieval-based Voice Conversion WebUI, pinned to a specific commit.
- Generated: Yes, by `scripts/setup_rvc.sh`.
- Committed: No.
- Rule: treat it as a read-only subprocess target. No Python-level imports. No modifications should be made under version control. If a fix is needed upstream, consider bumping the pinned commit hash in `scripts/setup_rvc.sh`.

**`rvc/.venv/`:**
- Purpose: isolated torch+fairseq+gradio virtualenv for RVC only.
- Generated: Yes, by `scripts/setup_rvc.sh`.
- Committed: No.
- Rule: never activate or install into this venv outside of `setup_rvc.sh`. Pin additions must be added to that script so the setup is reproducible.

**`src/slicer2.py`:**
- Purpose: vendored audio slicer from `audio-slicer` 1.0.1 (MIT). Copied, not imported via pip, to avoid a librosa/pydub dependency chain.
- Committed: Yes.
- Rule: do not hand-edit. It is excluded from ruff via `pyproject.toml:extend-exclude`. If the upstream fixes a bug, re-vendor the whole file.

**`.venv/`:**
- Purpose: application virtualenv (our code).
- Generated: `mise exec python@3.10 -- python -m venv .venv && .venv/bin/pip install -e ".[dev]"`.
- Committed: No.

**`scripts/setup_rvc.log`:**
- Purpose: append-only log of every `setup_rvc.sh` run. Useful for post-mortem when setup fails.
- Generated: Yes.
- Committed: No.

---

*Structure analysis: 2026-04-09*
