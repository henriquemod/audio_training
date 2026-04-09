---
phase: 02-training-cli
plan: 02
type: execute
wave: 2
depends_on: [02-01]
files_modified:
  - src/doctor.py
  - src/train.py
  - tests/unit/test_doctor.py
  - tests/unit/test_train_cli.py
autonomous: true
requirements: [TRAIN-01, TRAIN-06, TRAIN-12, TRAIN-13]
tags: [training, doctor, cli, validation]

must_haves:
  truths:
    - "src/doctor.py exposes check_pretrained_v2_weights, check_training_dataset_nonempty, run_training_checks"
    - "doctor.py --training composition includes the two new TRAIN-06 checks"
    - "src/train.py exposes a typer CLI with all flags from TRAIN-01 (minus --resume per D-05) and the --preset/--num-procs/--gpus/--verbose flags from D-01"
    - "Invalid CLI flag combinations exit 2 with stderr error and no subprocess started"
    - "--rvc-version v1 --sample-rate 32000 is rejected with exit 2"
    - "Doctor pre-flight failure exits 1 with rich.Table summary"
    - "src/train.py threat surface validated: experiment-name regex blocks ../, dataset-dir resolved to absolute and existence-checked"
  artifacts:
    - path: "src/doctor.py"
      provides: "Two new check functions + run_training_checks composition helper"
      contains: "def check_pretrained_v2_weights"
    - path: "src/train.py"
      provides: "typer CLI main() with validation, preset resolution, doctor pre-flight call"
      contains: "app = typer.Typer"
    - path: "tests/unit/test_doctor.py"
      provides: "Tests for check_pretrained_v2_weights and check_training_dataset_nonempty"
      contains: "def test_check_pretrained_v2_weights_missing"
    - path: "tests/unit/test_train_cli.py"
      provides: "Typer CliRunner tests for flag validation, exit codes, preset resolution"
      contains: "def test_cli_rejects_v1_with_32k"
  key_links:
    - from: "src/train.py main()"
      to: "src/doctor.py run_training_checks"
      via: "import + call before any subprocess"
      pattern: "from src.doctor import.*run_training_checks"
---

<objective>
Wire the CLI surface and pre-flight: extend `src/doctor.py` with the two new TRAIN-06 checks plus a `run_training_checks()` composition helper, and add the typer `main()` to `src/train.py` that validates flags (rejecting bad combinations with exit 2), resolves the preset, calls `run_training_checks()` (exit 1 on fail), and wires every D-01..D-05 + D-13 + D-23 decision into a runnable command surface.

Purpose: After this plan, `python src/train.py --help` works, every invalid input is rejected before any billable subprocess starts, and all TRAIN-06 doctor checks fire in the right order. Plan 03 will plug in the actual stage runner.

Output: Extended `src/doctor.py` (+ ~120 LOC), extended `src/train.py` (+ ~250 LOC for CLI/main), `tests/unit/test_doctor.py` (+ 6 tests), new `tests/unit/test_train_cli.py` (+ 12 tests using `typer.testing.CliRunner`).
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
@src/doctor.py
@src/train.py
@tests/unit/test_doctor.py
@tests/unit/test_doctor_training.py
@tests/unit/test_generate_cli.py

<interfaces>
<!-- New doctor functions to add (RESEARCH.md §7 + §4.9). -->

```python
# src/doctor.py — APPEND to existing module
PRETRAINED_MIN_BYTES = 30_000_000  # matches Phase 1's f0G40k.pth/f0D40k.pth floor

def check_pretrained_v2_weights(
    sample_rate: int, version: str, if_f0: bool
) -> CheckResult:
    """Verify the two pretrained files train.py will load as -pg and -pd
    exist and are non-empty (truncation guard)."""

def check_training_dataset_nonempty(dataset_dir: Path) -> CheckResult:
    """Verify dataset_dir exists, is a directory, has >= 1 file matching
    AUDIO_EXTS imported from src.preprocess."""

def run_training_checks(
    *, dataset_dir: Path, sample_rate: int, version: str, if_f0: bool
) -> list[CheckResult]:
    """Compose the full Phase 2 training pre-flight set."""
```

<!-- The Phase 1 training set composition is currently inline in src/doctor.py's typer command (per RESEARCH.md §1: "no run_training_checks() helper yet"). The new helper should mirror the Phase 1 list and APPEND the two new Phase 2 checks. The doctor.py typer command's `--training` branch should be refactored to call the new helper so there's a single source of truth. -->

<!-- typer CLI for src/train.py (RESEARCH.md §3 table) -->

```python
# Flag validation rules (exit 2 on any failure):
# - --experiment-name: validate_experiment_name() must return True (regex ^[a-zA-Z0-9_-]{1,64}$)
# - --sample-rate: must be in VALID_SAMPLE_RATES = (32000, 40000, 48000)
# - --rvc-version: must be in VALID_VERSIONS = ("v1", "v2")
# - --f0-method: must be in VALID_F0_METHODS = ("pm","harvest","rmvpe","rmvpe_gpu")
# - --preset: must be in VALID_PRESETS = ("smoke","low","balanced","high")
# - --rvc-version v1 + --sample-rate 32000: REJECT with exit 2 (Open Q4 — research recommendation; webui silently corrects, we don't)
# - --dataset-dir: must exist, must be a directory (resolve to absolute first)
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Add check_pretrained_v2_weights, check_training_dataset_nonempty, run_training_checks to src/doctor.py</name>
  <files>src/doctor.py, tests/unit/test_doctor.py</files>
  <read_first>
    - src/doctor.py (the WHOLE file — find the existing `--training` composition in the typer main; understand the inline list around lines 628-644 per RESEARCH.md §7; understand existing `check_hubert_base`, `check_disk_space_floor`, `check_gpu_vram_floor`, `check_rvc_mute_refs` so the new helper composes them correctly)
    - src/preprocess.py (find AUDIO_EXTS — confirm exact tuple)
    - tests/unit/test_doctor.py (whole file — match the existing test style)
    - tests/unit/test_doctor_training.py (Phase 1's training-flag tests — for monkeypatch RVC_DIR pattern)
    - .planning/phases/02-training-cli/02-RESEARCH.md §7 (Doctor Pre-flight — full check_pretrained_v2_weights body sketch)
  </read_first>
  <behavior>
    - check_pretrained_v2_weights(40000, "v2", True) returns ok=True when both rvc/assets/pretrained_v2/f0G40k.pth and f0D40k.pth exist and are >= 30 MB
    - check_pretrained_v2_weights(40000, "v2", True) returns ok=False with detail mentioning "f0G40k.pth" when the file is missing
    - check_pretrained_v2_weights(40000, "v2", True) returns ok=False with detail mentioning "bytes" when the file is truncated (< 30 MB)
    - check_pretrained_v2_weights(40000, "v1", False) probes rvc/assets/pretrained/G40k.pth (no f0 prefix, no _v2 suffix on dir)
    - check_training_dataset_nonempty(missing_path) returns ok=False
    - check_training_dataset_nonempty(file_not_dir) returns ok=False
    - check_training_dataset_nonempty(empty_dir) returns ok=False
    - check_training_dataset_nonempty(dir_with_wav_file) returns ok=True
    - check_training_dataset_nonempty(dir_with_only_txt_files) returns ok=False (only AUDIO_EXTS files count)
    - run_training_checks(dataset_dir, 40000, "v2", True) returns a list[CheckResult] including the Phase 1 base set + the two new Phase 2 checks
    - The doctor.py --training typer subcommand still works and includes the new checks (no behavioral regression vs Phase 1)
  </behavior>
  <action>
APPEND to `src/doctor.py` (do not rewrite existing functions). Place new code in a section commented `# ---------- Training pre-flight (Phase 2 additions) ----------` near the existing training-related checks.

```python
from src.preprocess import AUDIO_EXTS  # noqa: E402  # add to imports if not already present

PRETRAINED_MIN_BYTES = 30_000_000


def check_pretrained_v2_weights(
    sample_rate: int, version: str, if_f0: bool
) -> CheckResult:
    """Verify the two pretrained files RVC train.py will load as -pg and -pd
    exist on disk and are non-empty.

    Closes the STATE.md pitfall: "Missing pretrained weights cause silent
    random-init training." D-21 ensures we always pass -pg/-pd, but if the
    files don't exist on disk, training proceeds anyway with random init.
    Catching here is the only defense.

    Args:
        sample_rate: One of 32000, 40000, 48000.
        version: "v1" or "v2".
        if_f0: True for f0-aware models (default), False otherwise.

    Returns:
        CheckResult with ok=True if both files present and >= PRETRAINED_MIN_BYTES.
    """
    sr_str_map = {32000: "32k", 40000: "40k", 48000: "48k"}
    sr_str = sr_str_map.get(sample_rate)
    if sr_str is None:
        return CheckResult(
            name="rvc pretrained weights",
            ok=False,
            detail=f"unsupported sample_rate={sample_rate}",
            fix_hint="Use --sample-rate 32000, 40000, or 48000",
        )
    sub = "pretrained_v2" if version == "v2" else "pretrained"
    prefix = "f0" if if_f0 else ""
    g = RVC_DIR / "assets" / sub / f"{prefix}G{sr_str}.pth"
    d = RVC_DIR / "assets" / sub / f"{prefix}D{sr_str}.pth"
    missing = [p for p in (g, d) if not p.exists()]
    if missing:
        names = ", ".join(p.name for p in missing)
        return CheckResult(
            name="rvc pretrained weights",
            ok=False,
            detail=f"missing: {names}",
            fix_hint=f"Run scripts/setup_rvc.sh to download {names}",
        )
    for p in (g, d):
        size = p.stat().st_size
        if size < PRETRAINED_MIN_BYTES:
            return CheckResult(
                name="rvc pretrained weights",
                ok=False,
                detail=f"{p.name} is {size} bytes (expected >= {PRETRAINED_MIN_BYTES})",
                fix_hint="Truncated download. Run ./scripts/setup_rvc.sh --force.",
            )
    return CheckResult(
        name="rvc pretrained weights",
        ok=True,
        detail=f"{sub}/{prefix}{{G,D}}{sr_str}.pth",
    )


def check_training_dataset_nonempty(dataset_dir: Path) -> CheckResult:
    """Verify dataset_dir exists, is a directory, contains >= 1 audio file.

    Audio files are those whose suffix.lower() is in src.preprocess.AUDIO_EXTS.
    """
    if not dataset_dir.exists():
        return CheckResult(
            name="training dataset",
            ok=False,
            detail=f"not found: {dataset_dir}",
            fix_hint=f"Provide --dataset-dir pointing at an existing directory of audio files",
        )
    if not dataset_dir.is_dir():
        return CheckResult(
            name="training dataset",
            ok=False,
            detail=f"not a directory: {dataset_dir}",
            fix_hint="Pass a directory, not a file",
        )
    count = sum(
        1 for p in dataset_dir.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )
    if count == 0:
        return CheckResult(
            name="training dataset",
            ok=False,
            detail=f"no audio files in {dataset_dir} (looked for {AUDIO_EXTS})",
            fix_hint="Run src/preprocess.py first, or point --dataset-dir at the right place",
        )
    return CheckResult(
        name="training dataset",
        ok=True,
        detail=f"{count} audio file(s) in {dataset_dir}",
    )


def run_training_checks(
    *,
    dataset_dir: Path,
    sample_rate: int,
    version: str,
    if_f0: bool,
) -> list[CheckResult]:
    """Compose the full Phase 2 training pre-flight set.

    Includes Phase 1's base training checks plus the two new Phase 2 checks
    (check_pretrained_v2_weights, check_training_dataset_nonempty). Returns
    all results in order; does not raise.
    """
    checks = [
        check_python_version,
        check_ffmpeg,
        check_nvidia_smi,
        check_rvc_cloned,
        check_rvc_venv,
        check_rvc_weights,
        check_rvc_torch_cuda,
        lambda: check_disk_space_floor(PROJECT_ROOT, 20),
        lambda: check_gpu_vram_floor(12),
        check_rvc_mute_refs,
        check_hubert_base,
        lambda: check_pretrained_v2_weights(sample_rate, version, if_f0),
        lambda: check_training_dataset_nonempty(dataset_dir),
    ]
    return [c() for c in checks]
```

Then refactor the existing `--training` branch in `doctor.py`'s typer `main()` to call `run_training_checks` instead of duplicating the inline list. Use sensible defaults for the Phase 1-style invocation when called from the CLI without flags: `dataset_dir = PROJECT_ROOT / "dataset" / "processed"`, `sample_rate=40000`, `version="v2"`, `if_f0=True`. (These match the project defaults; the standalone `doctor.py --training` invocation does not need user-specified values — `src/train.py` will pass real values.)

If the existing `check_python_version` / `check_ffmpeg` / etc. function names differ in the actual Phase 1 codebase (READ src/doctor.py first to confirm), use the actual names. The list above is from RESEARCH.md §7 — verify against the file. If `check_disk_space_floor` / `check_gpu_vram_floor` lambda wrappers in Phase 1's --training set don't match this exactly, mirror Phase 1's pattern.

APPEND to `tests/unit/test_doctor.py`:

```python
# At top of file, add imports if not present:
from src.doctor import (
    check_pretrained_v2_weights,
    check_training_dataset_nonempty,
    run_training_checks,
    PRETRAINED_MIN_BYTES,
)

def test_check_pretrained_v2_weights_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("src.doctor.RVC_DIR", tmp_path)
    result = check_pretrained_v2_weights(40000, "v2", if_f0=True)
    assert not result.ok
    assert "f0G40k.pth" in result.detail or "f0D40k.pth" in result.detail
    assert result.fix_hint  # non-empty


def test_check_pretrained_v2_weights_truncated(tmp_path, monkeypatch):
    monkeypatch.setattr("src.doctor.RVC_DIR", tmp_path)
    sub = tmp_path / "assets" / "pretrained_v2"
    sub.mkdir(parents=True)
    (sub / "f0G40k.pth").write_bytes(b"\x00" * 100)
    (sub / "f0D40k.pth").write_bytes(b"\x00" * 100)
    result = check_pretrained_v2_weights(40000, "v2", if_f0=True)
    assert not result.ok
    assert "bytes" in result.detail
    assert "Truncated" in result.fix_hint or "truncated" in result.fix_hint.lower()


def test_check_pretrained_v2_weights_ok(tmp_path, monkeypatch):
    monkeypatch.setattr("src.doctor.RVC_DIR", tmp_path)
    sub = tmp_path / "assets" / "pretrained_v2"
    sub.mkdir(parents=True)
    big = b"\x00" * (PRETRAINED_MIN_BYTES + 1)
    (sub / "f0G40k.pth").write_bytes(big)
    (sub / "f0D40k.pth").write_bytes(big)
    result = check_pretrained_v2_weights(40000, "v2", if_f0=True)
    assert result.ok, f"expected ok, got: {result.detail}"


def test_check_pretrained_v1_no_f0_uses_pretrained_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("src.doctor.RVC_DIR", tmp_path)
    sub = tmp_path / "assets" / "pretrained"
    sub.mkdir(parents=True)
    big = b"\x00" * (PRETRAINED_MIN_BYTES + 1)
    (sub / "G40k.pth").write_bytes(big)
    (sub / "D40k.pth").write_bytes(big)
    result = check_pretrained_v2_weights(40000, "v1", if_f0=False)
    assert result.ok


def test_check_training_dataset_nonempty_missing(tmp_path):
    result = check_training_dataset_nonempty(tmp_path / "nope")
    assert not result.ok
    assert "not found" in result.detail


def test_check_training_dataset_nonempty_not_dir(tmp_path):
    f = tmp_path / "file.wav"
    f.touch()
    result = check_training_dataset_nonempty(f)
    assert not result.ok
    assert "not a directory" in result.detail


def test_check_training_dataset_nonempty_empty(tmp_path):
    result = check_training_dataset_nonempty(tmp_path)
    assert not result.ok
    assert "no audio files" in result.detail


def test_check_training_dataset_nonempty_only_txt(tmp_path):
    (tmp_path / "readme.txt").touch()
    result = check_training_dataset_nonempty(tmp_path)
    assert not result.ok


def test_check_training_dataset_nonempty_ok(tmp_path):
    (tmp_path / "a.wav").touch()
    (tmp_path / "b.flac").touch()
    result = check_training_dataset_nonempty(tmp_path)
    assert result.ok
    assert "2 audio file" in result.detail
```

Do NOT rewrite existing tests. Append only.
  </action>
  <verify>
    <automated>cd /home/henrique/Development/train_audio_model && .venv/bin/python -m pytest tests/unit/test_doctor.py -x -q -k "pretrained or training_dataset"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -q "def check_pretrained_v2_weights" src/doctor.py`
    - `grep -q "def check_training_dataset_nonempty" src/doctor.py`
    - `grep -q "def run_training_checks" src/doctor.py`
    - `grep -q "PRETRAINED_MIN_BYTES = 30_000_000" src/doctor.py`
    - `grep -q "from src.preprocess import AUDIO_EXTS" src/doctor.py`
    - `.venv/bin/python -m pytest tests/unit/test_doctor.py -x -q -k "pretrained or training_dataset"` exits 0 with at least 9 tests collected
    - `.venv/bin/python -m pytest tests/unit/test_doctor.py -x -q` exits 0 (no Phase 1 regressions)
    - `.venv/bin/python -m pytest tests/unit/test_doctor_training.py -x -q` exits 0 (Phase 1 --training composition still works)
    - `.venv/bin/ruff check src/doctor.py tests/unit/test_doctor.py` exits 0
  </acceptance_criteria>
  <done>Two new check functions plus the run_training_checks helper exist and pass tests; the doctor.py --training subcommand uses run_training_checks under the hood with no Phase 1 regressions.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Add typer CLI main() with flag validation, preset resolution, and doctor pre-flight to src/train.py</name>
  <files>src/train.py, tests/unit/test_train_cli.py</files>
  <read_first>
    - src/train.py (the file from Plan 01 — extend, do not rewrite)
    - src/generate.py (lines after _tail to end — copy the typer.Typer + Console pattern, the validation-and-exit-2 idiom, the doctor pre-flight call shape, the rich.Table usage for check failures)
    - src/doctor.py (the new run_training_checks signature from Task 1)
    - tests/unit/test_generate_cli.py (the existing typer.testing.CliRunner pattern for the project)
    - .planning/phases/02-training-cli/02-RESEARCH.md §3 (CLI Surface table with all flags)
    - .planning/phases/02-training-cli/02-RESEARCH.md §4.8 (main() outline)
    - .planning/phases/02-training-cli/02-RESEARCH.md §15 (security: experiment-name regex; threat model)
    - .planning/phases/02-training-cli/02-CONTEXT.md D-01 through D-13, D-17, D-23
  </read_first>
  <behavior>
    - `python src/train.py --help` exits 0 and lists all flags from RESEARCH.md §3 (--experiment-name, --dataset-dir, --sample-rate, --rvc-version, --f0-method, --preset, --epochs, --batch-size, --save-every, --num-procs, --gpus, --verbose); --resume is NOT listed (D-05)
    - Empty/invalid --experiment-name (e.g. "../etc", "exp/01", "x"*65) exits 2 with a clear error message to stderr
    - --sample-rate 44100 exits 2
    - --rvc-version v3 exits 2
    - --f0-method dio exits 2 (D-04 / Open Q5: dio excluded)
    - --f0-method crepe exits 2
    - --preset extreme exits 2
    - --rvc-version v1 --sample-rate 32000 exits 2 with message naming the invalid combo (Open Q4: research recommendation, do NOT silently correct)
    - --dataset-dir /nonexistent exits 2 (or 1 — see note below) with message "not found"
    - When all flag validation passes but doctor.run_training_checks returns any ok=False: exit 1, print rich.Table summary, print first fix_hint
    - When all flags valid AND doctor passes: main() should NOT crash even though Plan 03 has not added the stage runner — for now, print "TODO: stage runner (Plan 03)" and exit 0. (Plan 03 will replace this stub.)
    - Every test in tests/unit/test_train_cli.py uses typer.testing.CliRunner with monkeypatched run_training_checks to avoid touching the real RVC pretrained weights (tests run on dev box, not pod)

    Note on dataset-dir error code: per D-17, "missing dataset" is a config/setup error → exit 1. But the dataset-dir nonexistence is also caught later by check_training_dataset_nonempty (which returns ok=False → exit 1 from doctor preflight). To keep the validation path clean, do the directory existence check INSIDE doctor.run_training_checks (already does this via check_training_dataset_nonempty) — `main()` only enforces typer-level "is this a Path" + the experiment-name regex + the v1+32k combo at the exit-2 layer. So `--dataset-dir /nope` ends up exit 1 via doctor, NOT exit 2. Document this in the test names.
  </behavior>
  <action>
APPEND to `src/train.py` (after the helpers and builders from Plan 01):

```python
# ---------- Imports for CLI (extend the existing imports section above) ----------
# Add: import typer; from rich.console import Console; from rich.table import Table
# Add: from src.doctor import run_training_checks, CheckResult


# ---------- Validation helpers ----------

def _validate_cli_flags(
    *,
    experiment_name: str,
    sample_rate: int,
    rvc_version: str,
    f0_method: str,
    preset: str,
) -> Optional[str]:
    """Return None if all flags valid, else an error message string for stderr.

    Pure function — no I/O, no side effects. Caller handles exit.
    """
    if not validate_experiment_name(experiment_name):
        return (
            f"invalid --experiment-name {experiment_name!r}: "
            f"must match {EXPERIMENT_NAME_RE} (1-64 chars, alnum + _ + -)"
        )
    if sample_rate not in VALID_SAMPLE_RATES:
        return f"invalid --sample-rate {sample_rate}: must be one of {VALID_SAMPLE_RATES}"
    if rvc_version not in VALID_VERSIONS:
        return f"invalid --rvc-version {rvc_version!r}: must be one of {VALID_VERSIONS}"
    if f0_method not in VALID_F0_METHODS:
        return f"invalid --f0-method {f0_method!r}: must be one of {VALID_F0_METHODS}"
    if preset not in VALID_PRESETS:
        return f"invalid --preset {preset!r}: must be one of {VALID_PRESETS}"
    # Open Q4: reject the v1+32k combo explicitly (webui silently corrects; we don't).
    if rvc_version == "v1" and sample_rate == 32000:
        return (
            "invalid combination: --rvc-version v1 with --sample-rate 32000 is unsupported. "
            "Use --sample-rate 40000 or --sample-rate 48000 with v1."
        )
    return None


# ---------- typer app ----------

app = typer.Typer(
    add_completion=False,
    help="Train an RVC voice model end-to-end via the four-stage pipeline.",
)
console = Console()


@app.command()
def main(
    experiment_name: str = typer.Option(..., "--experiment-name", help="Experiment name (alnum + _ + -, 1-64 chars). Used as exp dir and output weight filename."),
    dataset_dir: Path = typer.Option(..., "--dataset-dir", help="Directory of preprocessed audio clips."),
    sample_rate: int = typer.Option(40000, "--sample-rate", help="Target sample rate: 32000, 40000, or 48000."),
    rvc_version: str = typer.Option("v2", "--rvc-version", help="RVC model version: v1 or v2."),
    f0_method: str = typer.Option("rmvpe", "--f0-method", help="F0 extraction: pm, harvest, rmvpe, or rmvpe_gpu."),
    preset: str = typer.Option("balanced", "--preset", help="Hyperparameter preset: smoke, low, balanced, high."),
    epochs: Optional[int] = typer.Option(None, "--epochs", help="Override preset epochs."),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Override preset batch size."),
    save_every: Optional[int] = typer.Option(None, "--save-every", help="Override preset save_every."),
    num_procs: int = typer.Option(DEFAULT_NUM_PROCS, "--num-procs", help="Worker processes for stages 1 and 2."),
    gpus: str = typer.Option("0", "--gpus", help="GPU ids for stage 4, dash-separated (e.g. '0' or '0-1')."),
    verbose: bool = typer.Option(False, "--verbose", help="Print full failure tail on error."),
) -> None:
    """Run the RVC training pipeline end-to-end (stages 1-4)."""
    # Step 1: validate CLI flag combinations (exit 2 on any failure)
    err = _validate_cli_flags(
        experiment_name=experiment_name,
        sample_rate=sample_rate,
        rvc_version=rvc_version,
        f0_method=f0_method,
        preset=preset,
    )
    if err:
        typer.echo(f"[error] {err}", err=True)
        raise typer.Exit(code=2)

    # Step 2: resolve preset + overrides (D-01/D-02/D-03)
    hp = resolve_preset(preset, epochs=epochs, batch_size=batch_size, save_every=save_every)

    # Step 3: resolve pretrained paths (D-21 — passed to builder later)
    if_f0 = f0_method != "none"  # always True for valid methods; placeholder for future "none"
    pretrained_g, pretrained_d = resolve_pretrained_paths(
        sample_rate=sample_rate, version=rvc_version, if_f0=if_f0
    )

    # Step 4: doctor pre-flight (D-13 — exit 1 on any fail)
    dataset_dir_abs = dataset_dir.resolve()
    results = run_training_checks(
        dataset_dir=dataset_dir_abs,
        sample_rate=sample_rate,
        version=rvc_version,
        if_f0=if_f0,
    )
    failed = [r for r in results if not r.ok]
    if failed:
        table = Table(title="Training pre-flight checks")
        table.add_column("Check")
        table.add_column("Status")
        table.add_column("Detail")
        for r in results:
            status = "[green]OK[/green]" if r.ok else "[red]FAIL[/red]"
            table.add_row(r.name, status, r.detail)
        console.print(table)
        first = failed[0]
        typer.echo(f"[error] {first.name}: {first.detail}", err=True)
        if first.fix_hint:
            typer.echo(f"[hint] {first.fix_hint}", err=True)
        raise typer.Exit(code=1)

    # Step 5: stage runner (Plan 03 will fill this in)
    console.print(
        f"[yellow]TODO Plan 03: stage runner not yet wired. "
        f"Resolved hp={hp}, dataset={dataset_dir_abs}, pretrained_g={pretrained_g}[/yellow]"
    )
    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
```

Make sure to add `import typer`, `from rich.console import Console`, `from rich.table import Table`, and `from src.doctor import run_training_checks` to the imports at the top of `src/train.py`. Keep the `noqa: E402` comments on `src.*` imports because of the sys.path fixup pattern.

Then create `tests/unit/test_train_cli.py`:

```python
"""CLI / main() tests for src/train.py using typer.testing.CliRunner."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from src.doctor import CheckResult
import src.train as train_mod
from src.train import app

runner = CliRunner(mix_stderr=False)


def _all_ok_stub(*, dataset_dir, sample_rate, version, if_f0):
    return [CheckResult(name="stub", ok=True, detail="stub")]


def _one_fail_stub(*, dataset_dir, sample_rate, version, if_f0):
    return [
        CheckResult(name="stub-ok", ok=True, detail="ok"),
        CheckResult(name="stub-fail", ok=False, detail="bad", fix_hint="fix it"),
    ]


def _make_dataset(tmp_path: Path) -> Path:
    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "a.wav").touch()
    return ds


def test_help_lists_all_flags():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    out = result.stdout
    for flag in ("--experiment-name", "--dataset-dir", "--sample-rate", "--rvc-version",
                 "--f0-method", "--preset", "--epochs", "--batch-size", "--save-every",
                 "--num-procs", "--gpus", "--verbose"):
        assert flag in out
    assert "--resume" not in out  # D-05


def test_rejects_invalid_experiment_name(tmp_path, monkeypatch):
    monkeypatch.setattr("src.train.run_training_checks", _all_ok_stub)
    ds = _make_dataset(tmp_path)
    result = runner.invoke(app, ["--experiment-name", "../etc", "--dataset-dir", str(ds)])
    assert result.exit_code == 2
    assert "invalid --experiment-name" in result.stderr


def test_rejects_long_experiment_name(tmp_path, monkeypatch):
    monkeypatch.setattr("src.train.run_training_checks", _all_ok_stub)
    ds = _make_dataset(tmp_path)
    result = runner.invoke(app, ["--experiment-name", "x" * 65, "--dataset-dir", str(ds)])
    assert result.exit_code == 2


def test_rejects_invalid_sample_rate(tmp_path, monkeypatch):
    monkeypatch.setattr("src.train.run_training_checks", _all_ok_stub)
    ds = _make_dataset(tmp_path)
    result = runner.invoke(app, [
        "--experiment-name", "smoke", "--dataset-dir", str(ds), "--sample-rate", "44100"
    ])
    assert result.exit_code == 2
    assert "--sample-rate" in result.stderr


def test_rejects_invalid_rvc_version(tmp_path, monkeypatch):
    monkeypatch.setattr("src.train.run_training_checks", _all_ok_stub)
    ds = _make_dataset(tmp_path)
    result = runner.invoke(app, [
        "--experiment-name", "smoke", "--dataset-dir", str(ds), "--rvc-version", "v3"
    ])
    assert result.exit_code == 2


def test_rejects_invalid_f0_method_dio(tmp_path, monkeypatch):
    """dio is excluded per Open Q5 / D-04."""
    monkeypatch.setattr("src.train.run_training_checks", _all_ok_stub)
    ds = _make_dataset(tmp_path)
    result = runner.invoke(app, [
        "--experiment-name", "smoke", "--dataset-dir", str(ds), "--f0-method", "dio"
    ])
    assert result.exit_code == 2
    assert "--f0-method" in result.stderr


def test_rejects_invalid_preset(tmp_path, monkeypatch):
    monkeypatch.setattr("src.train.run_training_checks", _all_ok_stub)
    ds = _make_dataset(tmp_path)
    result = runner.invoke(app, [
        "--experiment-name", "smoke", "--dataset-dir", str(ds), "--preset", "extreme"
    ])
    assert result.exit_code == 2


def test_rejects_v1_with_32k_combination(tmp_path, monkeypatch):
    """Open Q4 / Risk: webui silently corrects this; we reject it explicitly."""
    monkeypatch.setattr("src.train.run_training_checks", _all_ok_stub)
    ds = _make_dataset(tmp_path)
    result = runner.invoke(app, [
        "--experiment-name", "smoke", "--dataset-dir", str(ds),
        "--rvc-version", "v1", "--sample-rate", "32000",
    ])
    assert result.exit_code == 2
    assert "v1" in result.stderr and "32000" in result.stderr


def test_doctor_failure_exits_1(tmp_path, monkeypatch):
    monkeypatch.setattr("src.train.run_training_checks", _one_fail_stub)
    ds = _make_dataset(tmp_path)
    result = runner.invoke(app, [
        "--experiment-name", "smoke", "--dataset-dir", str(ds),
    ])
    assert result.exit_code == 1
    assert "stub-fail" in result.stderr or "stub-fail" in result.stdout
    assert "fix it" in result.stderr or "fix it" in result.stdout


def test_all_valid_reaches_runner_stub(tmp_path, monkeypatch):
    """All flags valid + doctor pass → exits 0 with the Plan-03 TODO marker."""
    monkeypatch.setattr("src.train.run_training_checks", _all_ok_stub)
    ds = _make_dataset(tmp_path)
    result = runner.invoke(app, [
        "--experiment-name", "smoke", "--dataset-dir", str(ds),
        "--preset", "smoke",
    ])
    assert result.exit_code == 0
    assert "TODO Plan 03" in result.stdout or "TODO Plan 03" in result.stderr


def test_preset_override_reaches_main(tmp_path, monkeypatch):
    """Smoke verifies preset + override resolution does not crash."""
    monkeypatch.setattr("src.train.run_training_checks", _all_ok_stub)
    ds = _make_dataset(tmp_path)
    result = runner.invoke(app, [
        "--experiment-name", "smoke", "--dataset-dir", str(ds),
        "--preset", "high", "--epochs", "800",
    ])
    assert result.exit_code == 0


def test_dataset_dir_missing_reaches_doctor_exit_1(tmp_path, monkeypatch):
    """Missing dataset dir is caught by check_training_dataset_nonempty -> exit 1, NOT exit 2."""
    # Use the real run_training_checks for this case so dataset check fires.
    # Stub everything except check_training_dataset_nonempty by importing the real function.
    from src.doctor import check_training_dataset_nonempty
    def stub(*, dataset_dir, sample_rate, version, if_f0):
        return [check_training_dataset_nonempty(dataset_dir)]
    monkeypatch.setattr("src.train.run_training_checks", stub)
    result = runner.invoke(app, [
        "--experiment-name", "smoke", "--dataset-dir", str(tmp_path / "nope"),
    ])
    assert result.exit_code == 1
```

Use the existing `tests/unit/test_generate_cli.py` for the precise CliRunner import style and `mix_stderr=False` pattern (the project may use a slightly different idiom — match it).
  </action>
  <verify>
    <automated>cd /home/henrique/Development/train_audio_model && .venv/bin/python -m pytest tests/unit/test_train_cli.py -x -q && .venv/bin/python src/train.py --help</automated>
  </verify>
  <acceptance_criteria>
    - `grep -q "app = typer.Typer" src/train.py`
    - `grep -q "@app.command" src/train.py`
    - `grep -q "def _validate_cli_flags" src/train.py`
    - `grep -q "EXPERIMENT_NAME_RE" src/train.py`
    - `grep -q "from src.doctor import.*run_training_checks" src/train.py`
    - `grep -q '"--experiment-name"' src/train.py` and similar for all 12 flags
    - `grep -q "raise typer.Exit(code=2)" src/train.py` (validation exit)
    - `grep -q "raise typer.Exit(code=1)" src/train.py` (doctor failure exit)
    - `grep -q "rvc_version == \"v1\" and sample_rate == 32000" src/train.py` (Open Q4)
    - `! grep -q "\"--resume\"" src/train.py` (D-05)
    - `.venv/bin/python src/train.py --help` exits 0 and stdout contains "--experiment-name"
    - `.venv/bin/python -m pytest tests/unit/test_train_cli.py -x -q` exits 0 with at least 11 tests passing
    - `.venv/bin/python -m pytest tests/unit/test_train.py tests/unit/test_train_cli.py tests/unit/test_doctor.py -x -q` exits 0 (no regressions)
    - `.venv/bin/ruff check src/train.py tests/unit/test_train_cli.py` exits 0
    - `grep -c "import torch\|import fairseq\|import faiss" src/train.py` returns 0 (D-24 still holds)
  </acceptance_criteria>
  <done>typer CLI is wired with all flags from D-01..D-05 + RESEARCH §3, validation rejects every bad input with exit 2, doctor preflight failures exit 1 with rich.Table, and a Plan-03 TODO stub is in place where the stage runner will go.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| user CLI → src/train.py | User-supplied --experiment-name, --dataset-dir, --gpus crossing into Path / subprocess argv construction |
| src/train.py → rvc/.venv subprocess | Constructed argv crossing the venv boundary; argv list never includes shell metacharacters because shell=False |
| src/train.py → filesystem (rvc/logs/<name>, rvc/assets/weights/<name>.pth) | Path construction from --experiment-name; traversal could write outside rvc/logs and rvc/assets/weights |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-02-01 | Tampering | `src/train.py` `--experiment-name` flag | mitigate | `validate_experiment_name()` enforces regex `^[a-zA-Z0-9_-]{1,64}$`; `_validate_cli_flags` rejects with exit 2 BEFORE any path construction or subprocess. Test: `test_rejects_invalid_experiment_name` asserts `"../etc"` is rejected. |
| T-02-02 | Tampering | `src/train.py` `--dataset-dir` flag | mitigate | `Path.resolve()` normalizes the path; `check_training_dataset_nonempty` asserts existence + directory + content type. Symlinks are accepted as user-owned risk (single-user workstation/pod, no privilege escalation surface). |
| T-02-03 | Tampering / Injection | RVC subprocess argv construction | mitigate | All `subprocess.Popen` calls (Plan 03) MUST use list argv with `shell=False` (project-wide constraint, enforced by `grep -E "shell=True" src/train.py` returning nothing). Argv values from validated, regex-restricted experiment_name and resolved Path objects only. |
| T-02-04 | Information Disclosure | Doctor check error messages | accept | Pretrained file paths in `fix_hint` reveal repo layout. Single-user local tool, no untrusted readers. |
| T-02-05 | DoS | Doctor check_pretrained_v2_weights `stat().st_size` | accept | One stat() per file, two files. Bounded cost. |

All STRIDE categories applicable to a CLI with no network and no auth surface are addressed (V5 Input Validation, V10 Malicious Code, V12 Files and Resources per RESEARCH.md §15).
</threat_model>

<verification>
- `.venv/bin/python -m pytest tests/unit/test_train.py tests/unit/test_train_cli.py tests/unit/test_doctor.py tests/unit/test_doctor_training.py -x -q` exits 0
- `.venv/bin/python src/train.py --help` exits 0 with all 12 flags listed
- `.venv/bin/python src/train.py --experiment-name "../bad" --dataset-dir /tmp 2>&1; echo $?` prints 2
- `.venv/bin/python src/train.py --experiment-name x --dataset-dir /tmp --rvc-version v1 --sample-rate 32000 2>&1; echo $?` prints 2
- `.venv/bin/ruff check src/train.py src/doctor.py tests/unit/test_train_cli.py tests/unit/test_doctor.py` exits 0
- `grep -c "shell=True" src/train.py src/doctor.py` returns 0
</verification>

<success_criteria>
- TRAIN-06 satisfied: doctor exposes check_pretrained_v2_weights + check_training_dataset_nonempty + run_training_checks (composes Phase 1's training checks + the two new ones)
- TRAIN-01 partially satisfied: every CLI flag from the requirement (minus --resume, per documented D-05 deviation) is present; --resume removal is noted in src/train.py module docstring
- TRAIN-12 partially satisfied: exit codes 0/1/2 wired (3 added in Plan 03 with the stage runner)
- All open questions from RESEARCH.md §9 resolved in code:
  - Q1: `_write_exp_config` is a separate sibling helper (not folded into _write_filelist) — locked by Plan 01
  - Q3: DEFAULT_NUM_PROCS = min(os.cpu_count() or 1, 8) — locked by Plan 01
  - Q4: --rvc-version v1 + --sample-rate 32000 explicitly rejected with exit 2 (this plan)
  - Q5: --f0-method valid set is exactly (pm, harvest, rmvpe, rmvpe_gpu) — dio excluded (this plan)
  - Experiment-name regex `^[a-zA-Z0-9_-]{1,64}$` enforced (this plan + Plan 01)
- Plan 03 has a stable typer surface to plug the stage runner into (replaces the "TODO Plan 03" stub)
</success_criteria>

<output>
After completion, create `.planning/phases/02-training-cli/02-02-SUMMARY.md` documenting:
- New doctor functions added (signatures + LOC)
- src/train.py CLI surface (final flag list, with --resume noted as removed per D-05)
- Test count delta (test_doctor.py +9, test_train_cli.py +12)
- Threat-model verification: all five T-02-XX threats mitigated or accepted with rationale
- Resolved open questions and the test that locks each one
</output>
