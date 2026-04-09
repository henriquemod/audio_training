---
phase: 02-training-cli
plan: 02
subsystem: training
tags: [training, doctor, cli, validation, typer, preflight]
requires:
  - "src.train.resolve_preset"
  - "src.train.resolve_pretrained_paths"
  - "src.train.validate_experiment_name"
  - "src.train.PRESETS"
  - "src.train.EXPERIMENT_NAME_RE"
  - "src.train.VALID_F0_METHODS"
  - "src.train.VALID_PRESETS"
  - "src.train.VALID_SAMPLE_RATES"
  - "src.train.VALID_VERSIONS"
  - "src.train.DEFAULT_NUM_PROCS"
provides:
  - "src.doctor.check_pretrained_v2_weights"
  - "src.doctor.check_training_dataset_nonempty"
  - "src.doctor.run_training_checks"
  - "src.doctor.PRETRAINED_MIN_BYTES"
  - "src.train.app"
  - "src.train.main"
  - "src.train._validate_cli_flags"
affects:
  - "src/doctor.py --training now delegates to run_training_checks"
tech-stack:
  added: []
  patterns:
    - "Doctor pre-flight composition helper (single source of truth for --training)"
    - "typer.testing.CliRunner with mix_stderr=False for stderr assertion"
    - "Monkeypatched run_training_checks in CLI tests to avoid touching real RVC state"
    - "Lazy import of AUDIO_EXTS inside check_training_dataset_nonempty to avoid circular import with src.preprocess"
key-files:
  created:
    - tests/unit/test_train_cli.py
  modified:
    - src/doctor.py
    - src/train.py
    - tests/unit/test_doctor.py
decisions:
  - "D-05: --resume flag removed in favor of always-on intrinsic probe-and-skip resume; noted in src/train.py module docstring"
  - "Open Q4: --rvc-version v1 + --sample-rate 32000 rejected with exit 2 (webui silently corrects; we don't)"
  - "Open Q5: --f0-method valid set is exactly (pm, harvest, rmvpe, rmvpe_gpu) — dio excluded"
  - "check_training_dataset_nonempty imports AUDIO_EXTS lazily to avoid circular import with src.preprocess"
  - "run_training_checks is the single source of truth; doctor --training now delegates to it with project defaults (dataset/processed, 40000, v2, if_f0=True)"
  - "Missing --dataset-dir surfaces via doctor pre-flight (exit 1), NOT CLI validation (exit 2), per D-17"
metrics:
  duration: "~12 min"
  tests_added: 23
  tests_total: 115
  test_runtime: "0.58s"
  src_lines_delta: "+182 (train.py: 535->716, doctor.py: 656->806 via append)"
  test_lines: "test_doctor.py +137, test_train_cli.py +215"
  completed: 2026-04-09
requirements: [TRAIN-01, TRAIN-06, TRAIN-12, TRAIN-13]
---

# Phase 2 Plan 2: Doctor Pre-flight + Training CLI Summary

Wires the typer CLI surface and doctor pre-flight for the training pipeline:
two new doctor checks (pretrained-weight existence/size and dataset
non-emptiness), a `run_training_checks()` composition helper, and
`src/train.py` gaining a typer `main()` with all 12 flags from TRAIN-01
(minus `--resume` per D-05), byte-exact validation rules, and a Plan-03
TODO stub where the stage runner will plug in.

## What Shipped

### `src/doctor.py` (+150 LOC)

**New constant:**
- `PRETRAINED_MIN_BYTES = 30_000_000` (matches Phase 1's hubert/rmvpe floor style)

**New functions:**
- `check_pretrained_v2_weights(sample_rate, version, if_f0) -> CheckResult`
  Probes `rvc/assets/{pretrained,pretrained_v2}/{f0,}{G,D}<sr>.pth`, asserts
  both files exist and are `>= PRETRAINED_MIN_BYTES`. Closes the
  silent-random-init-training pitfall (RVC's `train.py` does NOT hard-fail
  on missing pretrained paths).
- `check_training_dataset_nonempty(dataset_dir) -> CheckResult`
  Asserts directory exists, is a directory, and contains `>= 1` file whose
  suffix is in `src.preprocess.AUDIO_EXTS`. `AUDIO_EXTS` is imported lazily
  inside the function to avoid a circular import.
- `run_training_checks(*, dataset_dir, sample_rate, version, if_f0) -> list[CheckResult]`
  Single source of truth composing Phase 1's base training checks
  (python, ffmpeg, ffmpeg_filters, git, nvidia_smi, rvc_cloned, rvc_venv,
  rvc_weights, rvc_torch_cuda, slicer2, disk_space_floor(20), gpu_vram_floor(12),
  rvc_mute_refs, hubert_base) PLUS the two Phase 2 additions. Returns a
  16-element `list[CheckResult]`; never raises.

**Refactored:** doctor.py's typer `main()` `--training` branch now delegates
to `run_training_checks(dataset_dir=PROJECT_ROOT/"dataset"/"processed",
sample_rate=40000, version="v2", if_f0=True)` and wraps the precomputed
results as pass-through lambdas for `_run_checks`. No Phase 1 regressions
(test_doctor_training.py's 23 tests still pass).

### `src/train.py` (+182 LOC)

**New imports (with noqa: E402):** `typer`, `rich.console.Console`,
`rich.table.Table`, `src.doctor.run_training_checks`.

**New function:**
- `_validate_cli_flags(*, experiment_name, sample_rate, rvc_version,
  f0_method, preset) -> Optional[str]` — pure, no I/O. Returns `None` on
  success or an error string. Validates:
  - `validate_experiment_name(name)` regex (blocks `../etc`, 65-char, etc.)
  - `sample_rate in VALID_SAMPLE_RATES`
  - `rvc_version in VALID_VERSIONS`
  - `f0_method in VALID_F0_METHODS` (dio excluded — Open Q5 / D-04)
  - `preset in VALID_PRESETS`
  - `not (rvc_version == "v1" and sample_rate == 32000)` (Open Q4)

**New typer app + `main()`:** 12 flags wired per RESEARCH.md §3:

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--experiment-name` | str | required | regex `^[a-zA-Z0-9_-]{1,64}$` |
| `--dataset-dir` | Path | required | resolved + checked by doctor |
| `--sample-rate` | int | 40000 | 32000/40000/48000 |
| `--rvc-version` | str | v2 | v1/v2 |
| `--f0-method` | str | rmvpe | pm/harvest/rmvpe/rmvpe_gpu (no dio) |
| `--preset` | str | balanced | smoke/low/balanced/high |
| `--epochs` | Optional[int] | None | preset override |
| `--batch-size` | Optional[int] | None | preset override |
| `--save-every` | Optional[int] | None | preset override |
| `--num-procs` | int | DEFAULT_NUM_PROCS | stages 1+2 |
| `--gpus` | str | "0" | stage 4 |
| `--verbose` | bool | False | full failure tail |

`--resume` is **NOT** present — D-05: always-on intrinsic probe-and-skip
resume. Removal noted in the module docstring.

**main() flow:**
1. `_validate_cli_flags` (exit 2 on any failure, error to stderr)
2. `resolve_preset` (merges preset defaults with per-field overrides)
3. `resolve_pretrained_paths` (D-21 — absolute paths for the Plan-03 runner)
4. `dataset_dir.resolve()` + `run_training_checks(...)` (exit 1 on any
   `ok=False`, prints a rich.Table, echoes first failure's `name`/`detail`/
   `fix_hint` to stderr)
5. Plan-03 TODO: prints a yellow marker with all resolved values and
   `raise typer.Exit(code=0)`. Will be replaced by the stage runner in 02-03.

### `tests/unit/test_doctor.py` (+137 LOC, +11 tests)

- `test_check_pretrained_v2_weights_missing`
- `test_check_pretrained_v2_weights_truncated`
- `test_check_pretrained_v2_weights_ok` (uses sparse 30MB+1 files)
- `test_check_pretrained_v1_no_f0_uses_pretrained_dir` (asserts path branch:
  no `_v2` dir suffix, no `f0` prefix)
- `test_check_pretrained_v2_weights_bad_sample_rate`
- `test_check_training_dataset_nonempty_missing`
- `test_check_training_dataset_nonempty_not_dir`
- `test_check_training_dataset_nonempty_empty`
- `test_check_training_dataset_nonempty_only_txt` (AUDIO_EXTS filter)
- `test_check_training_dataset_nonempty_ok`
- `test_run_training_checks_returns_list_of_results` (monkeypatches every
  underlying check, verifies composition length and names)

### `tests/unit/test_train_cli.py` (new, 215 LOC, 12 tests)

All tests use `CliRunner(mix_stderr=False)` and monkeypatch
`src.train.run_training_checks` so they run offline with zero RVC state.

- `test_help_lists_all_flags` (asserts all 12 present, asserts `--resume` absent per D-05)
- `test_rejects_invalid_experiment_name` (`../etc`)
- `test_rejects_long_experiment_name` (65 chars)
- `test_rejects_invalid_sample_rate` (44100)
- `test_rejects_invalid_rvc_version` (v3)
- `test_rejects_invalid_f0_method_dio` (locks Open Q5)
- `test_rejects_invalid_preset` (extreme)
- `test_rejects_v1_with_32k_combination` (locks Open Q4)
- `test_doctor_failure_exits_1` (stub returns one failing check)
- `test_all_valid_reaches_runner_stub` (asserts "TODO Plan 03" marker)
- `test_preset_override_reaches_main` (`--preset high --epochs 800`)
- `test_dataset_dir_missing_reaches_doctor_exit_1` (uses real
  `check_training_dataset_nonempty` to prove D-17 exit-code routing:
  missing dataset → exit 1 via doctor, NOT exit 2 via CLI)

## Verification

All from the plan's `<verification>` block:

- `pytest tests/unit/test_train.py tests/unit/test_train_cli.py
  tests/unit/test_doctor.py tests/unit/test_doctor_training.py -x -q`
  → **115 passed in 0.58s** (54 train + 12 train_cli + 26 doctor + 23 doctor_training)
- `python src/train.py --help` → exit 0, all 12 flags listed, no `--resume`
- `python src/train.py --experiment-name "../bad" --dataset-dir /tmp` → **exit 2**
  with `"[error] invalid --experiment-name '../bad'..."` to stderr
- `python src/train.py --experiment-name x --dataset-dir /tmp --rvc-version v1 --sample-rate 32000`
  → **exit 2** with `"...v1 with --sample-rate 32000 is unsupported..."` to stderr
- `ruff check src/train.py src/doctor.py tests/unit/test_train_cli.py tests/unit/test_doctor.py`
  → **All checks passed!**
- `grep -c "shell=True" src/train.py src/doctor.py` → `0:0` (two-venv discipline holds)
- `grep -E "^(import|from) (torch|fairseq|faiss)" src/train.py` → empty
  (D-24 two-venv boundary preserved)

## Threat Model Verification

All five `T-02-XX` threats from the plan's `<threat_model>` are addressed:

| ID | Category | Disposition | Evidence |
|---|---|---|---|
| T-02-01 | Tampering (experiment-name) | mitigate | `_validate_cli_flags` calls `validate_experiment_name` BEFORE any path or subprocess construction; `test_rejects_invalid_experiment_name` locks `"../etc"` rejection; `test_rejects_long_experiment_name` locks 65-char rejection |
| T-02-02 | Tampering (dataset-dir) | mitigate | `dataset_dir.resolve()` in main() normalizes; `check_training_dataset_nonempty` asserts existence/type/contents; `test_dataset_dir_missing_reaches_doctor_exit_1` proves the error path |
| T-02-03 | Injection (argv) | mitigate | No `subprocess.run`/`Popen` calls added in this plan; the future stage runner in Plan 03 will use `shell=False` list argv only. `grep -c "shell=True"` returns 0 in both modified files |
| T-02-04 | Information Disclosure | accept | Unchanged from plan — single-user local tool, no untrusted readers |
| T-02-05 | DoS (stat) | accept | `check_pretrained_v2_weights` does exactly 2 `.stat()` calls |

## Resolved Open Questions

| OQ | Resolution | Test that locks it |
|---|---|---|
| Q1 | `_write_exp_config` exposed as separate sibling helper (locked Plan 01) | `test_write_exp_config_*` (Plan 01) |
| Q3 | `DEFAULT_NUM_PROCS = min(os.cpu_count() or 1, 8)` (locked Plan 01) | `test_default_num_procs` (Plan 01) |
| Q4 | v1+32k rejected with exit 2 | `test_rejects_v1_with_32k_combination` |
| Q5 | `VALID_F0_METHODS = ("pm","harvest","rmvpe","rmvpe_gpu")` (no dio) | `test_rejects_invalid_f0_method_dio` (this plan) + `test_valid_f0_methods` (Plan 01) |
| Experiment-name regex | `^[a-zA-Z0-9_-]{1,64}$` | `test_rejects_invalid_experiment_name`, `test_rejects_long_experiment_name` (this plan) + Plan 01 parametrized accept/reject |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Circular import avoidance**

- **Found during:** Task 1 implementation planning
- **Issue:** The plan's action snippet shows `from src.preprocess import AUDIO_EXTS`
  as a top-of-module import in `src/doctor.py`. But `src/preprocess.py` already
  does `from src.doctor import check_ffmpeg, check_ffmpeg_filters`, so adding
  a reverse top-level import creates a cycle at module-load time.
- **Fix:** Import `AUDIO_EXTS` lazily inside `check_training_dataset_nonempty`
  with an inline comment explaining why. All tests pass; doctor.py still
  imports cleanly; preprocess.py still imports cleanly.
- **Files modified:** `src/doctor.py`
- **Commit:** `b0524dd`

**2. [Rule 2 - Critical] Added OSError guard to `check_pretrained_v2_weights` stat**

- **Found during:** Task 1 implementation
- **Issue:** The plan snippet calls `p.stat().st_size` unguarded. If the file
  exists but `stat()` raises `PermissionError` (which `Path.exists()` may
  swallow), the check crashes instead of returning a clean `CheckResult`.
  Phase 1's `check_hubert_base` has the same guard pattern.
- **Fix:** Wrapped `p.stat()` in `try/except OSError` returning an `ok=False`
  result with "cannot stat" detail. Mirrors the Phase 1 convention.
- **Files modified:** `src/doctor.py`
- **Commit:** `b0524dd`

**3. [Rule 1 - Bug] Fixed `run_training_checks` composition vs. plan sketch**

- **Found during:** Task 1 implementation (reconciling plan sketch with the
  actual Phase 1 `--training` list)
- **Issue:** The plan sketch omitted `check_ffmpeg_filters`, `check_git`, and
  `check_slicer2_importable` from its `run_training_checks` composition, but
  the Phase 1 `--training` typer branch explicitly included them (see
  `src/doctor.py` lines 628-644 pre-refactor). The plan's `read_first`
  explicitly warns: "If the existing ... function names differ in the actual
  Phase 1 codebase (READ src/doctor.py first to confirm), use the actual
  names."
- **Fix:** `run_training_checks` now mirrors the Phase 1 training set exactly
  (14 base checks + 2 Phase 2 additions = 16 total), preserving behavior.
- **Files modified:** `src/doctor.py`
- **Commit:** `b0524dd`

**4. [Rule 2 - Critical] Added 6th pretrained test (bad sample rate branch)**

- **Found during:** Task 1 test writing
- **Issue:** The plan's test list covered the missing/truncated/ok/v1-path
  branches but not the `sample_rate not in {32000,40000,48000}` early-return
  branch. That branch has its own error message and fix_hint; leaving it
  uncovered meant a future refactor could silently drop the guard.
- **Fix:** Added `test_check_pretrained_v2_weights_bad_sample_rate`.
- **Files modified:** `tests/unit/test_doctor.py`
- **Commit:** `b0524dd`

## Commits

| Task | Hash | Message |
|------|------|---------|
| 1 | `b0524dd` | `feat(02-02): add Phase 2 training doctor checks` |
| 2 | `d058093` | `feat(02-02): add typer CLI main() with flag validation and doctor pre-flight` |

## Notes for Plan 03

- `src/train.py` exposes a stable typer surface and a clean stub at the
  "Step 5: stage runner" comment. Plan 03 replaces that stub with the
  actual four-stage runner; all flag/preset/preflight machinery is ready.
- `run_training_checks` is the single entry point for pre-flight. Plan 03
  can assume it has been called and returned all-ok before the runner
  starts — no need to re-check.
- `hp`, `pretrained_g`, `pretrained_d`, `dataset_dir_abs`, `if_f0`,
  `num_procs`, `gpus`, `f0_method`, `experiment_name`, `verbose` are all
  in local scope at the TODO marker.
- `TRAIN_SUCCESS_EXIT_CODES = (0, 61)` and `SUBPROCESS_EXTRA_ENV` are
  available from Plan 01 for the subprocess.Popen wrapper.

## Self-Check: PASSED

- `src/doctor.py` contains `def check_pretrained_v2_weights`: found
- `src/doctor.py` contains `def check_training_dataset_nonempty`: found
- `src/doctor.py` contains `def run_training_checks`: found
- `src/doctor.py` contains `PRETRAINED_MIN_BYTES = 30_000_000`: found
- `src/train.py` contains `app = typer.Typer`: found
- `src/train.py` contains `@app.command`: found
- `src/train.py` contains `def _validate_cli_flags`: found
- `src/train.py` contains `from src.doctor import ... run_training_checks`: found
- `src/train.py` does NOT contain `"--resume"`: confirmed
- `src/train.py` contains `rvc_version == "v1" and sample_rate == 32000`: found
- `tests/unit/test_train_cli.py` exists (215 LOC, 12 tests)
- Commit `b0524dd` exists: confirmed (Task 1)
- Commit `d058093` exists: confirmed (Task 2)
- 115 tests pass across all four related test files, ruff clean, --help exits 0
