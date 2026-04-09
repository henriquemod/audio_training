---
phase: 02-training-cli
plan: 03
type: execute
wave: 3
depends_on: [02-01, 02-02]
files_modified:
  - src/train.py
  - tests/unit/test_train_runner.py
autonomous: true
requirements: [TRAIN-02, TRAIN-07, TRAIN-08, TRAIN-09, TRAIN-10, TRAIN-11, TRAIN-12, TRAIN-14]
tags: [training, runner, sentinel, resume, error-handling]

must_haves:
  truths:
    - "src/train.py main() runs all four RVC stages in order via subprocess.Popen with cwd=RVC_DIR and shell=False"
    - "Each stage's stdout is streamed live to terminal AND tee'd to rvc/logs/<exp>/train.log"
    - "Re-running an experiment with completed sentinels skips already-done stages and prints 'Stage N: skipping' lines"
    - "RVC exit codes 0 AND 61 are both treated as success"
    - "Stage 4 cross-check verifies rvc/assets/weights/<exp>.pth exists and is >= 1024 bytes"
    - "On Stage N non-success exit, the last 30 lines of train.log are printed with stage context and exit 3 is raised"
    - "Subprocess env includes TRANSFORMERS_OFFLINE=1, HF_DATASETS_OFFLINE=1, LANG=C.UTF-8"
    - "src/train.py contains zero imports of torch, fairseq, faiss, or rvc (TRAIN-10 / D-24)"
    - "Smoke run on a pod (manual, non-autonomous): python src/train.py --experiment-name smoke --dataset-dir dataset/processed/ --epochs 1 --batch-size 1 exits 0 or 61, leaves rvc/assets/weights/smoke.pth on disk; second invocation skips all stages with the fast-path"
  artifacts:
    - path: "src/train.py"
      provides: "_run_stage_streamed, _tail_file, _build_subprocess_env, _is_train_success, run_pipeline orchestrator wired into main()"
      contains: "def _run_stage_streamed"
    - path: "tests/unit/test_train_runner.py"
      provides: "Tests for _tail_file, _build_subprocess_env, _is_train_success, mocked Popen orchestrator (TRAIN-09 resume), TRAIN-11 tail-on-failure"
      contains: "def test_is_train_success_61"
  key_links:
    - from: "src/train.py main()"
      to: "_run_stage_streamed"
      via: "called once per stage with the appropriate build_*_cmd output"
      pattern: "_run_stage_streamed\\("
    - from: "_run_stage_streamed"
      to: "subprocess.Popen"
      via: "stdout=PIPE, stderr=STDOUT, bufsize=1, text=True, cwd=RVC_DIR"
      pattern: "subprocess.Popen"
---

<objective>
Wire the stage runner: replace the "TODO Plan 03" stub in `src/train.py:main()` with the orchestrator that probes sentinels, runs each stage as a streamed subprocess, tees output to `rvc/logs/<exp>/train.log`, treats exit codes 0 and 61 as success, cross-checks the final weight file, and tails the log on failure with stage context.

Purpose: Closes Phase 2 by delivering an end-to-end runnable training pipeline that satisfies all five ROADMAP success criteria. After this plan, `python src/train.py --experiment-name smoke --dataset-dir dataset/processed/ --preset smoke` produces `rvc/assets/weights/smoke.pth` on a GPU pod.

Output: Extended `src/train.py` (+ ~250 LOC for the runner + main wiring), new `tests/unit/test_train_runner.py` (+ ~15 tests using mocked subprocess.Popen + tmp_path log files).
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/phases/02-training-cli/02-CONTEXT.md
@.planning/phases/02-training-cli/02-RESEARCH.md
@CLAUDE.md
@src/generate.py
@src/train.py

<interfaces>
<!-- All helpers below are pure or use only stdlib + the constants from Plan 01. -->

```python
# Existing Plan 01 imports already available:
# SR_STR_MAP, PRESETS, SUBPROCESS_EXTRA_ENV, TRAIN_SUCCESS_EXIT_CODES, WEIGHT_FILE_FLOOR_BYTES,
# STAGE_BANNER, build_rvc_*_cmd functions, _write_filelist, _write_exp_config,
# count_dataset_inputs, stage_1_is_done .. stage_4_is_done, resolve_pretrained_paths
# RVC_DIR, RVC_VENV_PYTHON imported from src.doctor

def _build_subprocess_env() -> dict[str, str]:
    """Return os.environ.copy() merged with SUBPROCESS_EXTRA_ENV (D-19, TRAIN-14)."""

def _is_train_success(returncode: int) -> bool:
    """True iff returncode in TRAIN_SUCCESS_EXIT_CODES (0 or 61)."""

def _tail_file(path: Path, n: int) -> str:
    """Return the last n lines of path. Returns '' if missing/unreadable.
    Defensive: catches FileNotFoundError, PermissionError, UnicodeDecodeError.
    For large logs, seek from the end (don't load the whole file)."""

def _run_stage_streamed(
    cmd: list[str],
    *,
    stage_num: int,
    stage_name: str,
    log_path: Path,
    env: dict[str, str],
) -> int:
    """Popen with stdout=PIPE, stderr=STDOUT, bufsize=1, text=True, cwd=RVC_DIR.
    Writes STAGE_BANNER to log_path and stdout, then iterates popen.stdout
    line-by-line, tee'ing each line to sys.stdout AND log_path.
    Returns the exit code; caller decides success/failure."""

def run_pipeline(
    *,
    experiment_name: str,
    dataset_dir: Path,
    sample_rate: int,
    rvc_version: str,
    f0_method: str,
    hp: dict[str, int],     # {"epochs", "batch_size", "save_every"}
    num_procs: int,
    gpus: str,
    pretrained_g: Path,
    pretrained_d: Path,
    if_f0: bool,
    verbose: bool,
) -> int:
    """Orchestrate stages 1-4 with sentinel skip + streamed runner.
    Returns 0 on success, 3 on stage failure / missing weight.
    Raises typer.Exit only at the top level (main()); this function returns codes."""
```

<!-- Sentinel decision tree (RESEARCH.md §5 + D-08/D-09): -->
```
exp_dir = RVC_DIR / "logs" / exp_name
weight_path = RVC_DIR / "assets" / "weights" / f"{exp_name}.pth"
N = count_dataset_inputs(dataset_dir)

# D-09 fast-path: fully complete already?
if stage_4_is_done(weight_path):
    print("Experiment already complete — nothing to do.")
    return 0

exp_dir.mkdir(parents=True, exist_ok=True)
log_path = exp_dir / "train.log"
env = _build_subprocess_env()

# Stage 1
if stage_1_is_done(exp_dir, N):
    print(f"Stage 1: skipping — already populated ({count} files)")
else:
    rc = _run_stage_streamed(build_rvc_preprocess_cmd(...), stage_num=1, stage_name="preprocess", ...)
    if rc not in TRAIN_SUCCESS_EXIT_CODES:
        _print_failure_tail(log_path, stage=1, name="preprocess", verbose=verbose)
        return 3
    if not stage_1_is_done(exp_dir, N):
        # Post-run sentinel check: even exit 0 may have produced incomplete output
        _print_failure_tail(log_path, stage=1, name="preprocess", verbose=verbose,
                             extra_hint="Stage 1 exited 0 but 0_gt_wavs/ count < expected")
        return 3

# Stage 2 — same pattern, build_rvc_extract_f0_cmd
# Stage 3 — same pattern, build_rvc_extract_feature_cmd
#           Note: STATE.md pitfall — extract_feature_print.py exits 0 when hubert is missing.
#           The post-run sentinel probe (stage_3_is_done) catches this.

# Pre-Stage 4: filelist + config (always regenerate, idempotent)
_write_filelist(exp_dir, version=rvc_version, sample_rate=sample_rate, if_f0=if_f0)
_write_exp_config(exp_dir, version=rvc_version, sample_rate=sample_rate)

# Stage 4
rc = _run_stage_streamed(build_rvc_train_cmd(...), stage_num=4, stage_name="train", ...)
if not _is_train_success(rc):
    _print_failure_tail(log_path, stage=4, name="train", verbose=verbose)
    return 3

# D-18: cross-check final weight
if not stage_4_is_done(weight_path):
    _print_failure_tail(log_path, stage=4, name="train", verbose=verbose,
        extra_hint="RVC reported success but no weight file was produced — check train.log for silent failures")
    return 3

print(f"[green]✓[/green] Training complete. Output: {weight_path}")
return 0
```

<!-- Failure tail helper (RESEARCH.md §6 + D-15): -->
```python
def _print_failure_tail(
    log_path: Path,
    *,
    stage: int,
    name: str,
    verbose: bool,
    extra_hint: str = "",
) -> None:
    n = 100 if verbose else 30
    tail = _tail_file(log_path, n)
    typer.echo(f"[error] Stage {stage} ({name}) failed", err=True)
    if tail:
        typer.echo(tail, err=True)
    else:
        typer.echo("(no output captured before crash)", err=True)
    if "CUDA out of memory" in tail:
        typer.echo("[hint] lower --batch-size or use --preset low", err=True)
    if extra_hint:
        typer.echo(f"[hint] {extra_hint}", err=True)
```

<!-- Subprocess streaming pattern (RESEARCH.md §10.1, locked by D-14, Risk R3/R4/R7) -->
```python
import subprocess
from datetime import datetime

def _run_stage_streamed(cmd, *, stage_num, stage_name, log_path, env):
    banner = STAGE_BANNER.format(
        n=stage_num, name=stage_name, ts=datetime.now().strftime("%H:%M:%S")
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", buffering=1, encoding="utf-8") as logf:
        logf.write("\n" + banner + "\n")
        sys.stdout.write(banner + "\n")
        sys.stdout.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            cwd=RVC_DIR,
            env=env,
        )
        assert proc.stdout is not None
        try:
            for line in iter(proc.stdout.readline, ""):
                sys.stdout.write(line)
                sys.stdout.flush()
                logf.write(line)
        finally:
            proc.stdout.close()
            proc.wait()
    return proc.returncode
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Add runner helpers (_build_subprocess_env, _is_train_success, _tail_file, _run_stage_streamed, _print_failure_tail) to src/train.py</name>
  <files>src/train.py, tests/unit/test_train_runner.py</files>
  <read_first>
    - src/train.py (the file as it stands after Plans 01 and 02 — extend it; do not rewrite)
    - src/generate.py (find the existing `_tail` helper — confirm signature; copy idiom for stderr-tail-on-failure; find the CUDA OOM hint pattern around line 320 per RESEARCH.md §6)
    - .planning/phases/02-training-cli/02-RESEARCH.md §6 (Error Handling & Tailing — exit-code mapping table)
    - .planning/phases/02-training-cli/02-RESEARCH.md §10.1 (Subprocess streaming pattern verbatim)
    - .planning/phases/02-training-cli/02-CONTEXT.md D-14, D-15, D-16, D-19
  </read_first>
  <behavior>
    - _is_train_success(0) is True; _is_train_success(61) is True; _is_train_success(1) is False; _is_train_success(-9) is False
    - _build_subprocess_env() returns a dict that contains every key in os.environ AND has TRANSFORMERS_OFFLINE=="1", HF_DATASETS_OFFLINE=="1", LANG=="C.UTF-8"
    - _build_subprocess_env() does NOT mutate os.environ (returns a fresh copy)
    - _tail_file(missing_path, 10) == ""
    - _tail_file(file_with_5_lines, 10) returns all 5 lines
    - _tail_file(file_with_100_lines, 10) returns last 10 lines (each ending with newline as appropriate)
    - _tail_file handles binary garbage gracefully (returns "" or partial; no crash)
    - _run_stage_streamed with a `python -c "print('hi')"` command writes "hi" to both stdout and log_path; returncode is 0; STAGE_BANNER appears as the first line in log
    - _run_stage_streamed with `python -c "import sys; sys.exit(7)"` returns 7
  </behavior>
  <action>
APPEND to `src/train.py` (after the typer command from Plan 02; before `if __name__ == "__main__"`).

Add these imports near the top of `src/train.py` if not already present:
```python
import subprocess
from datetime import datetime
```

Then add this section, commented `# ---------- Stage runner (subprocess streaming) ----------`:

```python
def _build_subprocess_env() -> dict[str, str]:
    """Build the env dict passed to every RVC subprocess (D-19, TRAIN-14).

    Returns a fresh copy of os.environ merged with the offline + locale flags.
    """
    env = os.environ.copy()
    env.update(SUBPROCESS_EXTRA_ENV)
    return env


def _is_train_success(returncode: int) -> bool:
    """Return True iff the RVC train.py exit code indicates success.

    Per TRAIN-07 / D-17, both 0 and 61 are success — RVC's os._exit(2333333)
    truncates to 61 on Linux.
    """
    return returncode in TRAIN_SUCCESS_EXIT_CODES


def _tail_file(path: Path, n: int) -> str:
    """Return the last n lines of path as a single string.

    Returns "" if the file is missing or unreadable. Defensive against
    multi-GB log files: reads from the end via seek() rather than loading
    the full file into memory.
    """
    try:
        size = path.stat().st_size
    except (FileNotFoundError, PermissionError, OSError):
        return ""
    if size == 0:
        return ""
    # Read last ~16 KB per requested line on average; cap at 1 MB for safety.
    block = min(max(n * 256, 4096), 1024 * 1024)
    try:
        with open(path, "rb") as f:
            if size <= block:
                f.seek(0)
            else:
                f.seek(size - block)
            data = f.read()
    except (FileNotFoundError, PermissionError, OSError):
        return ""
    try:
        text = data.decode("utf-8", errors="replace")
    except UnicodeDecodeError:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[-n:])


def _run_stage_streamed(
    cmd: list[str],
    *,
    stage_num: int,
    stage_name: str,
    log_path: Path,
    env: dict[str, str],
) -> int:
    """Run a stage subprocess, streaming stdout live to terminal AND log_path.

    Implements D-14: Popen with stdout=PIPE, stderr=STDOUT, bufsize=1, text=True,
    cwd=RVC_DIR. Iterates popen.stdout line-by-line (Risk R3 mitigation: prevents
    pipe-buffer deadlock). Writes STAGE_BANNER (D-16) to both streams before
    the first subprocess line.

    Args:
        cmd: argv list (already built by build_rvc_*_cmd).
        stage_num: 1-4.
        stage_name: human-readable name (preprocess/extract_f0/extract_feature/train).
        log_path: rvc/logs/<exp>/train.log — appended to.
        env: full subprocess environment (from _build_subprocess_env).

    Returns:
        The subprocess's exit code. Caller maps to success/failure via
        _is_train_success or per-stage 0-only logic.
    """
    banner = STAGE_BANNER.format(
        n=stage_num,
        name=stage_name,
        ts=datetime.now().strftime("%H:%M:%S"),
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", buffering=1, encoding="utf-8") as logf:
        logf.write("\n" + banner + "\n")
        sys.stdout.write(banner + "\n")
        sys.stdout.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            cwd=RVC_DIR,
            env=env,
        )
        assert proc.stdout is not None
        try:
            for line in iter(proc.stdout.readline, ""):
                sys.stdout.write(line)
                sys.stdout.flush()
                logf.write(line)
        finally:
            proc.stdout.close()
            proc.wait()
    return proc.returncode


def _print_failure_tail(
    log_path: Path,
    *,
    stage: int,
    name: str,
    verbose: bool,
    extra_hint: str = "",
) -> None:
    """Print stage failure context to stderr: tail of train.log + hints (D-15)."""
    n = 100 if verbose else 30
    tail = _tail_file(log_path, n)
    typer.echo(f"[error] Stage {stage} ({name}) failed", err=True)
    if tail:
        typer.echo(tail, err=True)
    else:
        typer.echo("(no output captured before crash)", err=True)
    if "CUDA out of memory" in tail:
        typer.echo("[hint] lower --batch-size or use --preset low", err=True)
    if extra_hint:
        typer.echo(f"[hint] {extra_hint}", err=True)
```

Then create `tests/unit/test_train_runner.py`:

```python
"""Tests for the stage-runner helpers in src/train.py."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from src.train import (
    SUBPROCESS_EXTRA_ENV,
    TRAIN_SUCCESS_EXIT_CODES,
    _build_subprocess_env,
    _is_train_success,
    _print_failure_tail,
    _run_stage_streamed,
    _tail_file,
)


# ---------- _is_train_success (TRAIN-07) ----------

def test_is_train_success_zero():
    assert _is_train_success(0) is True


def test_is_train_success_61():
    """RVC's os._exit(2333333) truncates to 61 on Linux — must be treated as success."""
    assert _is_train_success(61) is True


def test_is_train_success_other_failures():
    for rc in (1, 2, 3, 7, 137, -9, 255):
        assert _is_train_success(rc) is False


def test_train_success_codes_constant():
    assert TRAIN_SUCCESS_EXIT_CODES == (0, 61)


# ---------- _build_subprocess_env (D-19, TRAIN-14) ----------

def test_build_subprocess_env_includes_offline_flags():
    env = _build_subprocess_env()
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["HF_DATASETS_OFFLINE"] == "1"
    assert env["LANG"] == "C.UTF-8"


def test_build_subprocess_env_inherits_path():
    env = _build_subprocess_env()
    assert "PATH" in env
    assert env["PATH"] == os.environ["PATH"]


def test_build_subprocess_env_does_not_mutate_os_environ():
    before = os.environ.get("TRANSFORMERS_OFFLINE")
    _build_subprocess_env()
    after = os.environ.get("TRANSFORMERS_OFFLINE")
    assert before == after


def test_extra_env_constant():
    assert SUBPROCESS_EXTRA_ENV == {
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "LANG": "C.UTF-8",
    }


# ---------- _tail_file ----------

def test_tail_file_missing(tmp_path):
    assert _tail_file(tmp_path / "nope.log", 10) == ""


def test_tail_file_empty(tmp_path):
    p = tmp_path / "empty.log"
    p.touch()
    assert _tail_file(p, 10) == ""


def test_tail_file_fewer_lines(tmp_path):
    p = tmp_path / "five.log"
    p.write_text("a\nb\nc\nd\ne\n")
    out = _tail_file(p, 10)
    assert out.splitlines() == ["a", "b", "c", "d", "e"]


def test_tail_file_more_lines(tmp_path):
    p = tmp_path / "many.log"
    p.write_text("\n".join(f"line{i}" for i in range(100)) + "\n")
    out = _tail_file(p, 10)
    lines = out.splitlines()
    assert lines == [f"line{i}" for i in range(90, 100)]


def test_tail_file_handles_binary_garbage(tmp_path):
    p = tmp_path / "bin.log"
    p.write_bytes(b"\x00\xff\xfe text after\n" * 50)
    # Should not crash; returns some string (utf-8 errors=replace)
    out = _tail_file(p, 5)
    assert isinstance(out, str)


# ---------- _run_stage_streamed (uses real subprocess of cpython for stability) ----------

def test_run_stage_streamed_success(tmp_path):
    log = tmp_path / "train.log"
    rc = _run_stage_streamed(
        [sys.executable, "-c", "print('hello from stage')"],
        stage_num=1,
        stage_name="preprocess",
        log_path=log,
        env=os.environ.copy(),
    )
    assert rc == 0
    text = log.read_text()
    assert "Stage 1: preprocess" in text  # banner
    assert "hello from stage" in text


def test_run_stage_streamed_captures_nonzero_exit(tmp_path):
    log = tmp_path / "train.log"
    rc = _run_stage_streamed(
        [sys.executable, "-c", "import sys; print('boom'); sys.exit(7)"],
        stage_num=2,
        stage_name="extract_f0",
        log_path=log,
        env=os.environ.copy(),
    )
    assert rc == 7
    text = log.read_text()
    assert "boom" in text
    assert "Stage 2: extract_f0" in text


def test_run_stage_streamed_appends_not_overwrites(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("PRIOR CONTENT\n")
    rc = _run_stage_streamed(
        [sys.executable, "-c", "print('new line')"],
        stage_num=3,
        stage_name="extract_feature",
        log_path=log,
        env=os.environ.copy(),
    )
    assert rc == 0
    text = log.read_text()
    assert "PRIOR CONTENT" in text
    assert "new line" in text


# ---------- _print_failure_tail (TRAIN-11) ----------

def test_print_failure_tail_writes_30_lines(tmp_path, capsys):
    log = tmp_path / "train.log"
    log.write_text("\n".join(f"L{i}" for i in range(50)) + "\n")
    _print_failure_tail(log, stage=1, name="preprocess", verbose=False)
    err = capsys.readouterr().err
    assert "Stage 1 (preprocess) failed" in err
    assert "L49" in err
    assert "L20" in err
    # 30 lines means L20..L49 inclusive; L19 should NOT be there
    assert "L19" not in err


def test_print_failure_tail_verbose_writes_100_lines(tmp_path, capsys):
    log = tmp_path / "train.log"
    log.write_text("\n".join(f"L{i}" for i in range(150)) + "\n")
    _print_failure_tail(log, stage=4, name="train", verbose=True)
    err = capsys.readouterr().err
    assert "L50" in err  # 100 lines means L50..L149


def test_print_failure_tail_cuda_oom_hint(tmp_path, capsys):
    log = tmp_path / "train.log"
    log.write_text("some output\nCUDA out of memory\nmore output\n")
    _print_failure_tail(log, stage=4, name="train", verbose=False)
    err = capsys.readouterr().err
    assert "lower --batch-size" in err


def test_print_failure_tail_missing_log(tmp_path, capsys):
    _print_failure_tail(tmp_path / "nope.log", stage=1, name="preprocess", verbose=False)
    err = capsys.readouterr().err
    assert "no output captured" in err


def test_print_failure_tail_extra_hint(tmp_path, capsys):
    log = tmp_path / "train.log"
    log.write_text("done\n")
    _print_failure_tail(
        log, stage=4, name="train", verbose=False,
        extra_hint="RVC reported success but no weight file was produced",
    )
    err = capsys.readouterr().err
    assert "RVC reported success" in err
```
  </action>
  <verify>
    <automated>cd /home/henrique/Development/train_audio_model && .venv/bin/python -m pytest tests/unit/test_train_runner.py -x -q</automated>
  </verify>
  <acceptance_criteria>
    - `grep -q "def _build_subprocess_env" src/train.py`
    - `grep -q "def _is_train_success" src/train.py`
    - `grep -q "def _tail_file" src/train.py`
    - `grep -q "def _run_stage_streamed" src/train.py`
    - `grep -q "def _print_failure_tail" src/train.py`
    - `grep -q "subprocess.Popen" src/train.py`
    - `grep -q "stderr=subprocess.STDOUT" src/train.py`
    - `grep -q "bufsize=1" src/train.py`
    - `grep -q "cwd=RVC_DIR" src/train.py`
    - `grep -q 'iter(proc.stdout.readline, ""' src/train.py` (Risk R4 idiom)
    - `! grep -q "shell=True" src/train.py`
    - `.venv/bin/python -m pytest tests/unit/test_train_runner.py -x -q` exits 0 with at least 18 tests
    - `.venv/bin/ruff check src/train.py tests/unit/test_train_runner.py` exits 0
  </acceptance_criteria>
  <done>All five runner helpers exist with correct semantics; tests cover success/failure exit codes, env merging, file tailing edge cases, banner emission, and the CUDA OOM hint.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Wire run_pipeline orchestrator into main() with sentinel-skip resume and full stage flow</name>
  <files>src/train.py, tests/unit/test_train_runner.py</files>
  <read_first>
    - src/train.py (the file as it stands after Task 1 — find the "TODO Plan 03" stub in main() and the @app.command() main signature; you will replace the stub)
    - .planning/phases/02-training-cli/02-RESEARCH.md §5 (Sentinel & Resume Strategy — the decision tree pseudo-code)
    - .planning/phases/02-training-cli/02-RESEARCH.md §6 (Exit-code mapping table)
    - .planning/phases/02-training-cli/02-CONTEXT.md D-08, D-09, D-10, D-14, D-15, D-16, D-18, D-23
    - The `<interfaces>` block above (run_pipeline pseudo-code is canonical)
  </read_first>
  <behavior>
    - run_pipeline returns 0 on full success, 3 on any stage failure or post-run sentinel mismatch
    - Stage 1 is built with build_rvc_preprocess_cmd(rvc_python=RVC_VENV_PYTHON, dataset_dir=dataset_dir, sample_rate=sample_rate, num_procs=num_procs, exp_name=experiment_name)
    - Stage 2 is built with build_rvc_extract_f0_cmd(rvc_python=RVC_VENV_PYTHON, exp_name=experiment_name, num_procs=num_procs, f0_method=f0_method)
    - Stage 3 is built with build_rvc_extract_feature_cmd(rvc_python=RVC_VENV_PYTHON, exp_name=experiment_name, version=rvc_version)
    - Stage 4 is built with build_rvc_train_cmd(rvc_python=RVC_VENV_PYTHON, exp_name=experiment_name, sample_rate=sample_rate, version=rvc_version, epochs=hp["epochs"], batch_size=hp["batch_size"], save_every=hp["save_every"], f0_method=f0_method, pretrained_g=pretrained_g, pretrained_d=pretrained_d, if_f0=if_f0, gpus=gpus)
    - Each stage is preceded by its sentinel probe; if probe returns True, stage is SKIPPED with a "Stage N: skipping" log line and _run_stage_streamed is NOT called
    - The fast-path (stage_4_is_done(weight_path) at the very top) returns 0 immediately without touching any stage
    - Stages 1-3: a non-zero exit code (regardless of value) is treated as failure (these stages do NOT use the 0/61 success rule — only Stage 4 does); failure → tail + return 3
    - Stage 4: only failure if returncode not in TRAIN_SUCCESS_EXIT_CODES; on success, additional cross-check stage_4_is_done(weight_path) — if False, return 3 with "RVC reported success but no weight file was produced" hint
    - After every successful stage 1-3 run, a post-run sentinel probe verifies the expected output count; if mismatched (silent-failure case for Stage 3 when hubert is missing — STATE.md pitfall), tail + return 3
    - _write_filelist and _write_exp_config are called between Stage 3 success and Stage 4 invocation, every run (idempotent regeneration)
    - main() replaces the "TODO Plan 03" stub with `rc = run_pipeline(...); raise typer.Exit(code=rc)` after the doctor preflight passes
    - Mocked end-to-end tests verify: (a) sentinel skip on already-done stages, (b) failure tail printed on Stage 1 nonzero exit, (c) Stage 4 exit 61 treated as success, (d) missing weight after Stage 4 success returns 3, (e) the fast-path returns 0 without invoking _run_stage_streamed
  </behavior>
  <action>
APPEND `run_pipeline` to `src/train.py` (after `_print_failure_tail`, before the typer `app` definition — or after the typer app, doesn't matter). Then REPLACE the "TODO Plan 03" stub block in `main()` with a call to `run_pipeline(...)`.

```python
def run_pipeline(
    *,
    experiment_name: str,
    dataset_dir: Path,
    sample_rate: int,
    rvc_version: str,
    f0_method: str,
    hp: dict[str, int],
    num_procs: int,
    gpus: str,
    pretrained_g: Path,
    pretrained_d: Path,
    if_f0: bool,
    verbose: bool,
) -> int:
    """Orchestrate stages 1-4 with intrinsic probe-and-skip resume.

    Returns:
        0 on success.
        3 on stage failure / silent post-run sentinel mismatch / missing weight.

    See RESEARCH.md §5 for the decision tree and CONTEXT.md D-08/D-09/D-18.
    """
    exp_dir = (RVC_DIR / "logs" / experiment_name).resolve()
    weight_path = (RVC_DIR / "assets" / "weights" / f"{experiment_name}.pth").resolve()
    log_path = exp_dir / "train.log"

    # D-09 fast-path: experiment already complete?
    if stage_4_is_done(weight_path):
        sys.stdout.write(
            f"Experiment '{experiment_name}' already complete — final weight at {weight_path}\n"
        )
        return 0

    # Count expected outputs from the input dataset (D-08).
    n = count_dataset_inputs(dataset_dir)
    if n == 0:
        # Shouldn't happen — doctor's check_training_dataset_nonempty caught it — but be defensive.
        typer.echo(f"[error] dataset {dataset_dir} contains no audio files", err=True)
        return 3

    exp_dir.mkdir(parents=True, exist_ok=True)
    env = _build_subprocess_env()

    # ----- Stage 1: preprocess -----
    if stage_1_is_done(exp_dir, n):
        count = len(list((exp_dir / "0_gt_wavs").glob("*.wav")))
        sys.stdout.write(f"Stage 1: skipping — already populated ({count} files)\n")
    else:
        rc = _run_stage_streamed(
            build_rvc_preprocess_cmd(
                rvc_python=RVC_VENV_PYTHON,
                dataset_dir=dataset_dir,
                sample_rate=sample_rate,
                num_procs=num_procs,
                exp_name=experiment_name,
            ),
            stage_num=1,
            stage_name="preprocess",
            log_path=log_path,
            env=env,
        )
        if rc != 0:
            _print_failure_tail(log_path, stage=1, name="preprocess", verbose=verbose)
            return 3
        if not stage_1_is_done(exp_dir, n):
            _print_failure_tail(
                log_path, stage=1, name="preprocess", verbose=verbose,
                extra_hint=f"Stage 1 exited 0 but 0_gt_wavs/ has fewer than {n} files",
            )
            return 3

    # ----- Stage 2: extract_f0 -----
    if stage_2_is_done(exp_dir, n):
        sys.stdout.write("Stage 2: skipping — already populated\n")
    else:
        rc = _run_stage_streamed(
            build_rvc_extract_f0_cmd(
                rvc_python=RVC_VENV_PYTHON,
                exp_name=experiment_name,
                num_procs=num_procs,
                f0_method=f0_method,
            ),
            stage_num=2,
            stage_name="extract_f0",
            log_path=log_path,
            env=env,
        )
        if rc != 0:
            _print_failure_tail(log_path, stage=2, name="extract_f0", verbose=verbose)
            return 3
        if not stage_2_is_done(exp_dir, n):
            _print_failure_tail(
                log_path, stage=2, name="extract_f0", verbose=verbose,
                extra_hint=f"Stage 2 exited 0 but f0 output count is below expected {n}",
            )
            return 3

    # ----- Stage 3: extract_feature -----
    if stage_3_is_done(exp_dir, n, rvc_version):
        sys.stdout.write("Stage 3: skipping — already populated\n")
    else:
        rc = _run_stage_streamed(
            build_rvc_extract_feature_cmd(
                rvc_python=RVC_VENV_PYTHON,
                exp_name=experiment_name,
                version=rvc_version,
            ),
            stage_num=3,
            stage_name="extract_feature",
            log_path=log_path,
            env=env,
        )
        if rc != 0:
            _print_failure_tail(log_path, stage=3, name="extract_feature", verbose=verbose)
            return 3
        # STATE.md pitfall: extract_feature_print.py exits 0 when hubert is missing.
        # Post-run sentinel catches the silent failure.
        if not stage_3_is_done(exp_dir, n, rvc_version):
            _print_failure_tail(
                log_path, stage=3, name="extract_feature", verbose=verbose,
                extra_hint=(
                    "Stage 3 exited 0 but feature output is empty/short. "
                    "Likely hubert_base.pt is missing or corrupt — run scripts/setup_rvc.sh."
                ),
            )
            return 3

    # ----- Pre-Stage 4: filelist + config (always regenerate; idempotent) -----
    try:
        _write_filelist(exp_dir, version=rvc_version, sample_rate=sample_rate, if_f0=if_f0)
        _write_exp_config(exp_dir, version=rvc_version, sample_rate=sample_rate)
    except (RuntimeError, FileNotFoundError) as exc:
        typer.echo(f"[error] pre-stage 4 setup failed: {exc}", err=True)
        return 3

    # ----- Stage 4: train -----
    rc = _run_stage_streamed(
        build_rvc_train_cmd(
            rvc_python=RVC_VENV_PYTHON,
            exp_name=experiment_name,
            sample_rate=sample_rate,
            version=rvc_version,
            epochs=hp["epochs"],
            batch_size=hp["batch_size"],
            save_every=hp["save_every"],
            f0_method=f0_method,
            pretrained_g=pretrained_g,
            pretrained_d=pretrained_d,
            if_f0=if_f0,
            gpus=gpus,
        ),
        stage_num=4,
        stage_name="train",
        log_path=log_path,
        env=env,
    )
    if not _is_train_success(rc):
        _print_failure_tail(log_path, stage=4, name="train", verbose=verbose)
        return 3

    # D-18: cross-check the final weight file
    if not stage_4_is_done(weight_path):
        _print_failure_tail(
            log_path, stage=4, name="train", verbose=verbose,
            extra_hint=(
                "RVC reported success but no weight file was produced — "
                "check train.log for silent failures"
            ),
        )
        return 3

    sys.stdout.write(f"\nTraining complete. Output: {weight_path}\n")
    return 0
```

Then REPLACE the "TODO Plan 03" stub in `main()` with:

```python
    # Step 5: run the pipeline (Plan 03)
    rc = run_pipeline(
        experiment_name=experiment_name,
        dataset_dir=dataset_dir_abs,
        sample_rate=sample_rate,
        rvc_version=rvc_version,
        f0_method=f0_method,
        hp=hp,
        num_procs=num_procs,
        gpus=gpus,
        pretrained_g=pretrained_g,
        pretrained_d=pretrained_d,
        if_f0=if_f0,
        verbose=verbose,
    )
    raise typer.Exit(code=rc)
```

APPEND these end-to-end orchestrator tests to `tests/unit/test_train_runner.py`:

```python
# ---------- run_pipeline orchestrator (mocked subprocess + filesystem) ----------

import src.train as train_mod
from src.train import run_pipeline


def _make_dataset(tmp_path: Path, n: int = 3) -> Path:
    ds = tmp_path / "ds"
    ds.mkdir()
    for i in range(n):
        (ds / f"clip{i}.wav").touch()
    return ds


def _populate_stage_outputs(exp_dir: Path, n: int, version: str = "v2") -> None:
    """Pretend stages 1-3 ran successfully."""
    for sub in ("0_gt_wavs",):
        (exp_dir / sub).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (exp_dir / sub / f"clip{i}.wav").touch()
    for sub in ("2a_f0", "2b-f0nsf"):
        (exp_dir / sub).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (exp_dir / sub / f"clip{i}.wav.npy").touch()
    feat = "3_feature768" if version == "v2" else "3_feature256"
    (exp_dir / feat).mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (exp_dir / feat / f"clip{i}.npy").touch()


def _stub_rvc_root(tmp_path: Path, monkeypatch) -> Path:
    """Make RVC_DIR point at a fake tree under tmp_path. Also stub mute and configs."""
    fake_rvc = tmp_path / "rvc"
    # mute tree for _write_filelist
    mute_gt = fake_rvc / "logs" / "mute" / "0_gt_wavs"
    mute_gt.mkdir(parents=True)
    (mute_gt / "mute40k.wav").touch()
    # config templates for _write_exp_config (v2+40k uses v1 dir)
    cfg = fake_rvc / "configs" / "v1" / "40k.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text('{"stub": true}')
    monkeypatch.setattr(train_mod, "RVC_DIR", fake_rvc)
    return fake_rvc


def test_run_pipeline_fast_path_when_weight_exists(tmp_path, monkeypatch, capsys):
    fake_rvc = _stub_rvc_root(tmp_path, monkeypatch)
    weight = fake_rvc / "assets" / "weights" / "smoke.pth"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"\x00" * 2048)
    ds = _make_dataset(tmp_path)

    called = {"count": 0}
    def fake_runner(cmd, **kwargs):
        called["count"] += 1
        return 0
    monkeypatch.setattr(train_mod, "_run_stage_streamed", fake_runner)

    rc = run_pipeline(
        experiment_name="smoke", dataset_dir=ds, sample_rate=40000,
        rvc_version="v2", f0_method="rmvpe",
        hp={"epochs": 1, "batch_size": 1, "save_every": 1},
        num_procs=1, gpus="0",
        pretrained_g=Path("/g"), pretrained_d=Path("/d"),
        if_f0=True, verbose=False,
    )
    assert rc == 0
    assert called["count"] == 0  # fast-path: NO stages invoked


def test_run_pipeline_skips_done_stages_runs_train(tmp_path, monkeypatch, capsys):
    fake_rvc = _stub_rvc_root(tmp_path, monkeypatch)
    ds = _make_dataset(tmp_path, n=3)
    exp_dir = fake_rvc / "logs" / "smoke"
    _populate_stage_outputs(exp_dir, n=3, version="v2")

    weight = fake_rvc / "assets" / "weights" / "smoke.pth"
    weight.parent.mkdir(parents=True)

    invocations = []
    def fake_runner(cmd, *, stage_num, stage_name, log_path, env):
        invocations.append((stage_num, stage_name))
        if stage_num == 4:
            weight.write_bytes(b"\x00" * 2048)  # simulate train.py producing the weight
        return 0
    monkeypatch.setattr(train_mod, "_run_stage_streamed", fake_runner)

    rc = run_pipeline(
        experiment_name="smoke", dataset_dir=ds, sample_rate=40000,
        rvc_version="v2", f0_method="rmvpe",
        hp={"epochs": 1, "batch_size": 1, "save_every": 1},
        num_procs=1, gpus="0",
        pretrained_g=Path("/g"), pretrained_d=Path("/d"),
        if_f0=True, verbose=False,
    )
    assert rc == 0
    # Only Stage 4 should have been invoked; 1-3 were skipped via sentinels.
    assert [s[0] for s in invocations] == [4]
    out = capsys.readouterr().out
    assert "Stage 1: skipping" in out
    assert "Stage 2: skipping" in out
    assert "Stage 3: skipping" in out


def test_run_pipeline_stage_4_exit_61_is_success(tmp_path, monkeypatch):
    fake_rvc = _stub_rvc_root(tmp_path, monkeypatch)
    ds = _make_dataset(tmp_path, n=2)
    exp_dir = fake_rvc / "logs" / "smoke"
    _populate_stage_outputs(exp_dir, n=2, version="v2")
    weight = fake_rvc / "assets" / "weights" / "smoke.pth"
    weight.parent.mkdir(parents=True)

    def fake_runner(cmd, *, stage_num, stage_name, log_path, env):
        if stage_num == 4:
            weight.write_bytes(b"\x00" * 2048)
            return 61  # the os._exit(2333333) truncation
        return 0
    monkeypatch.setattr(train_mod, "_run_stage_streamed", fake_runner)

    rc = run_pipeline(
        experiment_name="smoke", dataset_dir=ds, sample_rate=40000,
        rvc_version="v2", f0_method="rmvpe",
        hp={"epochs": 1, "batch_size": 1, "save_every": 1},
        num_procs=1, gpus="0",
        pretrained_g=Path("/g"), pretrained_d=Path("/d"),
        if_f0=True, verbose=False,
    )
    assert rc == 0


def test_run_pipeline_stage_1_failure_returns_3(tmp_path, monkeypatch, capsys):
    fake_rvc = _stub_rvc_root(tmp_path, monkeypatch)
    ds = _make_dataset(tmp_path, n=2)

    def fake_runner(cmd, *, stage_num, stage_name, log_path, env):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("preprocess crash output\n")
        return 7
    monkeypatch.setattr(train_mod, "_run_stage_streamed", fake_runner)

    rc = run_pipeline(
        experiment_name="smoke", dataset_dir=ds, sample_rate=40000,
        rvc_version="v2", f0_method="rmvpe",
        hp={"epochs": 1, "batch_size": 1, "save_every": 1},
        num_procs=1, gpus="0",
        pretrained_g=Path("/g"), pretrained_d=Path("/d"),
        if_f0=True, verbose=False,
    )
    assert rc == 3
    err = capsys.readouterr().err
    assert "Stage 1 (preprocess) failed" in err
    assert "preprocess crash output" in err


def test_run_pipeline_stage_4_success_but_missing_weight_returns_3(tmp_path, monkeypatch, capsys):
    fake_rvc = _stub_rvc_root(tmp_path, monkeypatch)
    ds = _make_dataset(tmp_path, n=2)
    exp_dir = fake_rvc / "logs" / "smoke"
    _populate_stage_outputs(exp_dir, n=2, version="v2")

    def fake_runner(cmd, *, stage_num, stage_name, log_path, env):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("training done but no weight written\n")
        return 0  # exit 0 but no weight file written
    monkeypatch.setattr(train_mod, "_run_stage_streamed", fake_runner)

    rc = run_pipeline(
        experiment_name="smoke", dataset_dir=ds, sample_rate=40000,
        rvc_version="v2", f0_method="rmvpe",
        hp={"epochs": 1, "batch_size": 1, "save_every": 1},
        num_procs=1, gpus="0",
        pretrained_g=Path("/g"), pretrained_d=Path("/d"),
        if_f0=True, verbose=False,
    )
    assert rc == 3
    err = capsys.readouterr().err
    assert "no weight file was produced" in err


def test_run_pipeline_stage_3_silent_hubert_failure(tmp_path, monkeypatch, capsys):
    """STATE.md pitfall: extract_feature_print.py exits 0 when hubert is missing."""
    fake_rvc = _stub_rvc_root(tmp_path, monkeypatch)
    ds = _make_dataset(tmp_path, n=3)
    exp_dir = fake_rvc / "logs" / "smoke"
    # Only stages 1 and 2 are populated; stage 3 will be invoked.
    (exp_dir / "0_gt_wavs").mkdir(parents=True)
    for i in range(3):
        (exp_dir / "0_gt_wavs" / f"clip{i}.wav").touch()
    for sub in ("2a_f0", "2b-f0nsf"):
        (exp_dir / sub).mkdir()
        for i in range(3):
            (exp_dir / sub / f"clip{i}.wav.npy").touch()

    def fake_runner(cmd, *, stage_num, stage_name, log_path, env):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(f"stage {stage_num} ran\n")
        return 0  # silent success — but no output files written
    monkeypatch.setattr(train_mod, "_run_stage_streamed", fake_runner)

    rc = run_pipeline(
        experiment_name="smoke", dataset_dir=ds, sample_rate=40000,
        rvc_version="v2", f0_method="rmvpe",
        hp={"epochs": 1, "batch_size": 1, "save_every": 1},
        num_procs=1, gpus="0",
        pretrained_g=Path("/g"), pretrained_d=Path("/d"),
        if_f0=True, verbose=False,
    )
    assert rc == 3
    err = capsys.readouterr().err
    assert "Stage 3" in err
    assert "hubert" in err.lower()
```
  </action>
  <verify>
    <automated>cd /home/henrique/Development/train_audio_model && .venv/bin/python -m pytest tests/unit/test_train_runner.py tests/unit/test_train.py tests/unit/test_train_cli.py tests/unit/test_doctor.py -x -q && .venv/bin/python -m pytest -x -q && .venv/bin/python src/train.py --help</automated>
  </verify>
  <acceptance_criteria>
    - `grep -q "def run_pipeline" src/train.py`
    - `grep -q "stage_4_is_done(weight_path)" src/train.py` (fast-path D-09)
    - `grep -q "Stage 1: skipping" src/train.py`
    - `grep -q "Stage 2: skipping" src/train.py`
    - `grep -q "Stage 3: skipping" src/train.py`
    - `grep -q "_write_filelist" src/train.py` AND `grep -q "_write_exp_config" src/train.py` (called in pipeline)
    - `grep -q "build_rvc_preprocess_cmd" src/train.py` and similar for the other 3 builders (called in pipeline)
    - `grep -q "_is_train_success(rc)" src/train.py` (Stage 4 only — TRAIN-07)
    - `grep -q "no weight file was produced" src/train.py` (D-18)
    - `grep -q "hubert" src/train.py` (Stage 3 silent-failure hint)
    - `! grep -q "TODO Plan 03" src/train.py` (stub removed)
    - `! grep -q "shell=True" src/train.py`
    - `grep -c "import torch\|import fairseq\|import faiss" src/train.py` returns 0 (TRAIN-10 / D-24 still holds)
    - `.venv/bin/python -m pytest tests/unit/test_train_runner.py -x -q` exits 0 with at least 24 tests (18 helper + 6 orchestrator)
    - `.venv/bin/python -m pytest -x -q` exits 0 (full suite green; no Phase 1 / generate / preprocess regressions)
    - `.venv/bin/python src/train.py --help` exits 0
    - `.venv/bin/ruff check src/train.py tests/unit/test_train_runner.py` exits 0
  </acceptance_criteria>
  <done>run_pipeline orchestrator wired into main(), all six end-to-end orchestrator tests pass with mocked subprocess, full test suite green, two-venv boundary still intact.</done>
</task>

</tasks>

<verification>
- `.venv/bin/python -m pytest -x -q` exits 0 (full project suite)
- `.venv/bin/python src/train.py --help` exits 0
- `.venv/bin/python src/train.py --experiment-name smoke --dataset-dir /nonexistent 2>&1; echo $?` prints 1 (dataset check)
- `grep -E "^import (torch|fairseq|faiss)|^from (torch|fairseq|faiss)" src/train.py` returns nothing (TRAIN-10)
- `grep -E "shell=True" src/train.py` returns nothing
- `.venv/bin/ruff check src/train.py src/doctor.py tests/unit/` exits 0

### Pod-side smoke verification (manual, non-autonomous — required to close success criteria 1, 2, 3, 5)
On a provisioned RunPod / Vast / Lambda Labs box with `dataset/processed/` populated:

1. `python src/train.py --experiment-name smoke --dataset-dir dataset/processed/ --epochs 1 --batch-size 1`
   - Expected: exit 0, `rvc/assets/weights/smoke.pth` exists and is >= 1 KiB
2. `python src/train.py --experiment-name smoke --dataset-dir dataset/processed/ --epochs 1 --batch-size 1` (re-invoke)
   - Expected: prints "Experiment 'smoke' already complete" and exits 0 in <2s (D-09 fast-path)
3. Delete `rvc/assets/weights/smoke.pth`, then re-invoke
   - Expected: prints "Stage 1: skipping", "Stage 2: skipping", "Stage 3: skipping", runs Stage 4, exits 0
4. (Failure path) `python src/train.py --experiment-name nonexistentname --dataset-dir dataset/processed/ --rvc-version v2 --sample-rate 12345`
   - Expected: exit 2 with stderr error message
</verification>

<success_criteria>
- TRAIN-02 ✓ all four stages orchestrated via subprocess.Popen with cwd=RVC_DIR (verified by mocked-Popen test seeing all four stage_num values)
- TRAIN-07 ✓ exit codes 0 AND 61 both treated as success in Stage 4 (test_run_pipeline_stage_4_exit_61_is_success)
- TRAIN-08 ✓ sentinel-based skip-if-done resume works for stages 1-3 (test_run_pipeline_skips_done_stages_runs_train)
- TRAIN-09 ✓ aborted-mid-pipeline resume produces a valid `.pth` (mocked test asserts the resume path; pod-side smoke run validates the real flow per ROADMAP success criterion #3)
- TRAIN-10 ✓ no torch/fairseq/faiss imports anywhere in src/train.py (test_no_rvc_imports_in_train_module from Plan 01 still green)
- TRAIN-11 ✓ failure tail prints last 30 lines with stage context, exits 3 (test_print_failure_tail_writes_30_lines + test_run_pipeline_stage_1_failure_returns_3)
- TRAIN-12 ✓ exit code 3 wired for stage failures (Plan 02 wired 0/1/2; this plan completes the convention)
- TRAIN-14 ✓ subprocess env carries TRANSFORMERS_OFFLINE/HF_DATASETS_OFFLINE/LANG (test_build_subprocess_env_includes_offline_flags)
- ROADMAP success criterion #1: deliverable on a pod via the smoke CLI line above
- ROADMAP success criterion #2: covered by test_run_pipeline_skips_done_stages_runs_train (mock-level) + manual pod re-run (real-level)
- ROADMAP success criterion #3: covered by mocked test (logic) + manual pod kill-and-resume (smoke)
- ROADMAP success criterion #4: ALREADY satisfied by Plan 01 (test_train.py exact-argv tests)
- ROADMAP success criterion #5: covered by test_run_pipeline_stage_1_failure_returns_3 (30-line tail + stage context + exit 3)
</success_criteria>

<output>
After completion, create `.planning/phases/02-training-cli/02-03-SUMMARY.md` documenting:
- Final src/train.py shape (line count, public symbols, the run_pipeline orchestrator)
- Test count delta (test_train_runner.py: 24 tests)
- Each ROADMAP success criterion mapped to its proving test or pod-side smoke step
- The two-venv boundary verification (grep result)
- The pod-side smoke run as a manual verification step the user must perform after merging Phase 2 (this is the Phase Gate)
</output>
