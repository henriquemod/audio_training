# Coding Conventions

**Analysis Date:** 2026-04-09

## Naming Patterns

**Files:**
- Lowercase with underscores: `ffmpeg_utils.py`, `preprocess.py`, `generate.py`
- Exception: Vendored third-party code kept with original names: `slicer2.py`

**Functions:**
- snake_case: `run_ffmpeg()`, `check_ffmpeg()`, `build_canonical_args()`, `_slugify()`
- Private/internal functions prefixed with underscore: `_slice_with_slicer2()`, `_default_output_path()`, `_run_checks()`
- Helper functions prefixed with underscore when internal to module: `_tail()`, `_ensure_model_in_rvc_weights()`

**Classes:**
- PascalCase: `FfmpegError`, `FfmpegResult`, `PreprocessError`, `CheckResult`
- Custom exceptions inherit from appropriate base: `FfmpegError(RuntimeError)`, `PreprocessError(RuntimeError)`

**Variables:**
- snake_case for local/module variables: `resolved_text`, `final_out`, `written`
- UPPER_CASE for constants: `AUDIO_EXTS`, `CANONICAL_SR`, `SILENCE_RMS_THRESHOLD`, `PEAK_CLIPPING_THRESHOLD`, `DEFAULT_MODEL`, `DEFAULT_EDGE_VOICE`
- Module-level variables starting with underscore for internal state: `_PROJECT_ROOT`, `RVC_VENV_PYTHON`, `MODELS_DIR`

**Type annotations:**
- PascalCase for type names: `Path`, `CheckResult`, `FfmpegResult`, `Optional`, `list[str]`
- Modern union syntax with `|` (requires `from __future__ import annotations`): `tuple[int, int, int] | None`
- Optional use `Optional[Type]` when required by typer 0.12.3 (which lacks PEP 604 support)

## Code Style

**Formatting:**
- Line length: 100 characters (configured in ruff)
- Tool: ruff (built-in formatter)
- Python version: 3.10.x (strict enforcement via `requires-python = "==3.10.*"`)

**Linting:**
- Tool: ruff
- Config file: `pyproject.toml` under `[tool.ruff]` and `[tool.ruff.lint]`
- Key rules enabled: E (pycodestyle), F (pyflakes), W (warnings), I (isort), B (flake8-bugbear), UP (pyupgrade), N (pep8-naming), SIM (flake8-simplify)
- Line length rule (E501) ignored (handled by formatter)
- PEP 604 unions (UP007) ignored (typer 0.12.3 requires `Optional[X]`)
- Typer idiom (B008) ignored (typer.Option() in defaults is standard)

**Per-file exceptions:**
- `src/generate.py`: Allow E402 (module-level import not at top) for sys.path fixup
- `src/preprocess.py`: Allow E402 (module-level import not at top) for sys.path fixup
- `rvc/` directory: Completely excluded from ruff checks (third-party code)
- `src/slicer2.py`: Excluded from ruff (vendored upstream)

## Import Organization

**Order:**
1. `from __future__ import annotations` (always first, enables PEP 563 type annotation strings)
2. Standard library imports (stdlib): `import subprocess`, `from pathlib import Path`, `from typing import Optional`
3. Third-party imports: `import typer`, `import numpy as np`, `from rich.console import Console`
4. Local imports: `from src.doctor import check_ffmpeg`, `from src.ffmpeg_utils import FfmpegError`
5. Module-level setup/constants/classes (after imports)

**Path aliases:**
- No path aliases configured
- Relative imports within src/ use full module path: `from src.doctor import ...` not `from .doctor import ...`
- This allows files to be run as scripts or modules

**Special import handling:**
- Files that may run as scripts include sys.path fixup before internal imports:
  ```python
  _PROJECT_ROOT = Path(__file__).resolve().parent.parent
  if str(_PROJECT_ROOT) not in sys.path:
      sys.path.insert(0, str(_PROJECT_ROOT))
  ```
- This pattern seen in `src/preprocess.py` and `src/generate.py` (marked with E402 exception)
- Fallback imports for module/script execution: try qualified import first, then bare import (see `src/slicer2.py` import in `src/preprocess.py` line 111-113)

## Module Docstrings

**Format:**
- Triple-quoted docstring at module top (before `from __future__`)
- Describes purpose and any important caveats
- Pipeline-oriented modules list steps: `1. Step\n2. Step\n3. Step`
- Example: `src/ffmpeg_utils.py` documents that all ffmpeg calls must go through `run_ffmpeg()`
- Example: `src/preprocess.py` documents the full pipeline and that it's idempotent

**Convention:** Docstring acts as design documentation and is the single source of truth for module behavior.

## Function Design

**Docstring style:**
- Google-style docstrings with Args, Returns, Raises sections
- Example from `src/ffmpeg_utils.py`:
  ```python
  def run_ffmpeg(
      args: list[str],
      *,
      context: str,
      expected_output: Path,
      binary: str = "ffmpeg",
  ) -> FfmpegResult:
      """Run ffmpeg with the given args and verify it produced the expected output.
      
      Args:
          args: ffmpeg arguments WITHOUT the leading "ffmpeg"...
          context: a short description of the pipeline stage...
          
      Returns:
          FfmpegResult with captured stdout and stderr.
          
      Raises:
          FfmpegError: on non-zero exit, or if expected_output is missing/empty.
      """
  ```

**Parameter design:**
- Use keyword-only parameters (after `*`) for named arguments: `run_ffmpeg(args, *, context, expected_output, binary="ffmpeg")`
- Positional parameters only when essential (rare)
- Type hints always included: `args: list[str]`, `context: str`, `expected_output: Path`
- Optional parameters have defaults: `binary: str = "ffmpeg"`

**Return values:**
- Single dataclass for multiple return values: `FfmpegResult` with fields `stdout: str`, `stderr: str`
- Exceptions raised for errors instead of returning None/error codes
- Functions either succeed and return data, or raise an exception with context

**Size:**
- Functions are concise, typically 10-40 lines
- Complex pipelines broken into helper functions with clear names
- Section comments delineate major function groups: `# ---------- System checks ----------`

## Error Handling

**Philosophy:** Fail loud with context.

**Patterns:**
- Custom exception classes inherit from `RuntimeError` or `RuntimeError` subclass: `FfmpegError(RuntimeError)`, `PreprocessError(RuntimeError)`
- Exception messages include context tags in brackets: `[{context}] ffmpeg exited with code...`
- Full error info provided: command invoked, exit code, stderr output, file paths
- Example from `src/ffmpeg_utils.py`:
  ```python
  raise FfmpegError(
      f"[{context}] ffmpeg exited with code {proc.returncode}\n"
      f"  command: {' '.join(full_cmd)}\n"
      f"  stderr: {proc.stderr.strip() or '(empty)'}"
  )
  ```

**CLI exit codes:**
- 0: success
- 1: config/setup error (missing model, missing venv, missing ffmpeg)
- 2: user input error (empty text, mutually-exclusive flags, bad voice)
- 3: runtime error (ffmpeg, edge-tts, subprocess failure)
- Documented at module top: `src/generate.py` documents all exit codes

**Subprocess error handling:**
- Always use `check=False` and inspect `returncode` manually
- Capture both stdout and stderr: `capture_output=True, text=True`
- Provide full command and stderr in error messages for debugging

## Logging

**Framework:** rich.console.Console (not logging module)

**Patterns:**
- Create console at module top if needed: `console = Console()` in `src/generate.py`, `src/doctor.py`
- Use console.print() for output: `console.print(table)`, `console.print(f"[green]✓[/green] {detail}")`
- Rich markup for colors/styling: `[green]OK[/green]`, `[red]FAIL[/red]`, `[cyan]Check[/cyan]`
- CLI uses typer.echo() for simple output: `typer.echo("[error] message", err=True)`
- Error messages to stderr: `typer.echo(..., err=True)`

**When to output:**
- Doctor checks: always output result table
- CLI validation: output error with exit code
- Success: minimal output (let file/dir existence confirm)
- Debug info: only with `--verbose` flag

## Comments

**When to comment:**
- Section headers: `# ---------- System checks ----------`
- Complex logic that isn't obvious from names
- Non-obvious regex patterns or ffmpeg filter logic
- Workarounds or vendor-specific quirks

**What NOT to comment:**
- Obvious code: `name = "ffmpeg"  # set the name`
- Self-documenting patterns: clear function names, type hints eliminate need

**JSDoc/TSDoc:**
- Python uses docstrings, not comments
- Docstrings are mandatory for: public functions, classes, modules
- Private functions may omit docstrings if name is clear

## Constants & Configuration

**Constants:**
- Module-level constants are UPPER_CASE: `CANONICAL_SR = 44100`, `SILENCE_RMS_THRESHOLD = 0.005`
- Grouped with other module constants at module top (after class definitions, before functions)
- Magic numbers always become named constants

**Configuration:**
- Environment variables via python-dotenv: `from dotenv import load_dotenv; load_dotenv()`
- Environment reads: `DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "myvoice_v1")`
- Fallback values always provided: `os.environ.get("KEY", default_value)`
- Config sourced at module level and passed to functions/CLI via parameters (not global access)

## Type Hints

**Coverage:** 100% of function signatures
- Parameters always typed: `def run_ffmpeg(args: list[str], *, context: str) -> FfmpegResult:`
- Return types always specified
- Union types use modern syntax: `tuple[int, int, int] | None` (via `from __future__ import annotations`)
- Use Optional[] only when required by framework: `typer.Option(None, ...)` requires `Optional[str]` parameter

**Complex types:**
- Use dataclasses for structured returns: `@dataclass class FfmpegResult:`
- Avoid bare `dict` and `list`; use `dict[str, str]` and `list[Path]`

---

*Convention analysis: 2026-04-09*
