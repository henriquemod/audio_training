---
phase: 02-training-cli
plan: 01
subsystem: training
tags: [training, rvc, cli, builders, unit-tests]
requires: []
provides:
  - "src.train.build_rvc_preprocess_cmd"
  - "src.train.build_rvc_extract_f0_cmd"
  - "src.train.build_rvc_extract_feature_cmd"
  - "src.train.build_rvc_train_cmd"
  - "src.train._write_filelist"
  - "src.train._write_exp_config"
  - "src.train.resolve_preset"
  - "src.train.resolve_pretrained_paths"
  - "src.train.count_dataset_inputs"
  - "src.train.stage_1_is_done"
  - "src.train.stage_2_is_done"
  - "src.train.stage_3_is_done"
  - "src.train.stage_4_is_done"
  - "src.train.validate_experiment_name"
  - "src.train.PRESETS"
  - "src.train.SR_STR_MAP"
  - "src.train.TRAIN_SUCCESS_EXIT_CODES"
  - "src.train.SUBPROCESS_EXTRA_ENV"
affects: []
tech-stack:
  added: []
  patterns:
    - "Pure arg-builder pattern (mirrors src/generate.py:build_rvc_subprocess_cmd)"
    - "Two-venv boundary enforced via test_no_rvc_imports_in_train_module"
key-files:
  created:
    - src/train.py
    - tests/unit/test_train.py
  modified: []
decisions:
  - "Config helper exposed as separate `_write_exp_config` (Open Question 1)"
  - "`--f0-method` VALID set excludes `dio` (Open Question 4)"
  - "Experiment-name regex `^[a-zA-Z0-9_-]{1,64}$` (Open Question 5)"
metrics:
  duration: "~15 min"
  tests: 54
  test_runtime: "0.20s"
  src_lines: 534
  test_lines: 554
  completed: 2026-04-09
requirements: [TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04, TRAIN-05, TRAIN-13, TRAIN-14]
---

# Phase 2 Plan 1: RVC Arg Builders Summary

Pure-function backbone of `src/train.py` — module skeleton, constants, four
byte-exact RVC arg-builders, filelist/config helpers, preset resolver, pretrained
resolver, and sentinel probes — plus 54 unit tests asserting byte-exact argv
against the pinned RVC commit with zero GPU and zero `rvc/.venv` access.

## What Shipped

### `src/train.py` (534 LOC)

**Constants** (all verbatim from PLAN interfaces):
- `SR_STR_MAP = {32000: "32k", 40000: "40k", 48000: "48k"}`
- `VALID_F0_METHODS = ("pm", "harvest", "rmvpe", "rmvpe_gpu")` (dio excluded)
- `VALID_VERSIONS`, `VALID_SAMPLE_RATES`, `VALID_PRESETS`
- `PRESETS` (smoke / low / balanced / high)
- `SUBPROCESS_EXTRA_ENV` with TRANSFORMERS_OFFLINE / HF_DATASETS_OFFLINE / LANG
- `TRAIN_SUCCESS_EXIT_CODES = (0, 61)` — treats RVC's `os._exit(2333333) -> 61` as success
- `WEIGHT_FILE_FLOOR_BYTES = 1024`
- `PRETRAINED_MIN_BYTES = 30_000_000`
- `EXPERIMENT_NAME_RE = r"^[a-zA-Z0-9_-]{1,64}$"`
- `DEFAULT_NUM_PROCS = min(os.cpu_count() or 1, 8)`

**Pure helpers:**
- `resolve_preset(name, *, epochs, batch_size, save_every) -> dict[str, int]`
- `resolve_pretrained_paths(*, sample_rate, version, if_f0) -> tuple[Path, Path]`
- `count_dataset_inputs(dataset_dir: Path) -> int` (non-recursive, AUDIO_EXTS filter)
- `stage_1_is_done`, `stage_2_is_done`, `stage_3_is_done`, `stage_4_is_done`
- `validate_experiment_name(name) -> bool`

**Four byte-exact arg-builders:**
- `build_rvc_preprocess_cmd` — Stage 1 (preprocess.py)
- `build_rvc_extract_f0_cmd` — Stage 2 with two branches (rmvpe_gpu vs others)
- `build_rvc_extract_feature_cmd` — Stage 3 (single-GPU path)
- `build_rvc_train_cmd` — Stage 4 with D-21 enforcement (`-pg`/`-pd` always absolute)

**Filelist + config helpers:**
- `_write_filelist` — writes `exp_dir/filelist.txt` with 5-field rows (f0 mode),
  stem-intersection across the 4 stage dirs, plus D-07 mute-reference rows
- `_write_exp_config` — copies `rvc/configs/{v1,v2}/<sr>.json` with the v2+40k
  quirk (reads from `configs/v1/40k.json` per RVC webui lines 555-569)

### `tests/unit/test_train.py` (554 LOC, 54 tests, 0.20s)

- Constant-shape tests (exact dict/tuple equality)
- Preset resolver override-mix tests (D-03 example cases)
- `validate_experiment_name` parametrized accepts/rejects including `../../etc/passwd`
- Pretrained resolver paths for v1/v2 × with/without f0 × 32k/40k/48k
- Byte-exact argv assertions for all four stage builders
- `test_sample_rate_flows_to_both_preprocess_and_train` proves TRAIN-13 chain
- `test_no_rvc_imports_in_train_module` proves TRAIN-10 two-venv boundary
- `test_subprocess_env_has_offline_flags` proves TRAIN-14
- Sentinel probe coverage for all four stages including size-floor edge cases
- `_write_filelist` coverage: v2+f0 5-field rows, v1 uses `3_feature256`, empty → RuntimeError
- `_write_exp_config` coverage: v2+48k, v2+40k quirk, missing source → FileNotFoundError

## Verification

- `pytest tests/unit/test_train.py -q` → **54 passed in 0.20s**
- `ruff check src/train.py tests/unit/test_train.py` → **All checks passed**
- `grep "^import torch\|^from torch\|^import fairseq\|^from fairseq\|^import faiss\|^from faiss" src/train.py` → **empty**
- `import src.train` → **clean**

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed `_write_filelist` double-stripping `.wav` on stage-2 stems**

- **Found during:** Task 3 (first test run)
- **Issue:** The PLAN's `_write_filelist` implementation built
  `f0_stems = {n[:-len(".wav")] for n in _stems(f0_dir, ".wav.npy")}`. But
  `_stems(f0_dir, ".wav.npy")` already returns stems with the full `.wav.npy`
  suffix stripped (e.g. `"clip1"`), so the outer comprehension then chopped
  the trailing 4 characters a second time, yielding garbage like `""` or `"c"`.
  Result: the stem intersection was always empty and `_write_filelist` only
  wrote the mute row, failing `test_write_filelist_v2_with_f0` which asserts
  `len(lines) >= 4` (3 clip rows + 1 mute row).
- **Fix:** Use `_stems(...)` results directly for `f0_stems` / `f0nsf_stems` with
  an explanatory comment. All 54 tests pass.
- **Files modified:** `src/train.py`
- **Commit:** `19caeca`

**2. [Rule 1 - Bug] Fixed docstring triggering the import-ban substring match**

- **Found during:** Task 3 (second test run)
- **Issue:** `test_no_rvc_imports_in_train_module` does a literal substring
  search for `"import torch"` in the module source. The original module
  docstring contained the phrase `MUST NOT import torch, fairseq, faiss, or rvc`
  which matched and failed the test.
- **Fix:** Rephrased the docstring to `MUST NOT depend on torch / fairseq /
  faiss / rvc at the Python level` — same meaning, no substring collision.
- **Files modified:** `src/train.py`
- **Commit:** `19caeca`

**3. [Rule 3 - Blocking] Ruff `SIM108` (if/else → ternary) in `_write_filelist`**

- **Found during:** Task 2 (ruff check)
- **Issue:** Two-branch `if if_f0: common = ...` block flagged by `SIM108`.
- **Fix:** Inlined as `common = (... if if_f0 else ...)` ternary.
- **Commit:** `08cb047`

**4. [Rule 3 - Blocking] Ruff `SIM300` (yoda conditions) in test file**

- **Found during:** Task 3 (ruff check)
- **Issue:** Two test assertions written `assert EXPECTED == ACTUAL` triggered
  `SIM300`. Auto-fixed by `ruff --fix`, resulting in assertions in the form
  `assert {expected_dict} == ACTUAL_CONST`.
- **Commit:** `19caeca`

## Commits

| Task | Hash | Message |
|------|------|---------|
| 1 | `794f5b6` | `feat(02-01): add src/train.py constants and pure helpers` |
| 2 | `08cb047` | `feat(02-01): add RVC arg-builders + filelist/config helpers` |
| 3 | `19caeca` | `test(02-01): add unit tests for src/train.py builders` |

## Resolved Open Questions

- **OQ 1 (config helper shape):** Exposed as a separate `_write_exp_config`
  function (not folded into `_write_filelist`), matching the PLAN's interface
  block and allowing independent unit testing of the v2+40k source-dir quirk.
- **OQ 4 (`--f0-method` valid set):** `dio` excluded; `VALID_F0_METHODS =
  ("pm", "harvest", "rmvpe", "rmvpe_gpu")`.
- **OQ 5 (experiment-name regex):** `^[a-zA-Z0-9_-]{1,64}$`. Covered by
  parametrized accept/reject tests including path-traversal attempts.

## Notes for Plan 02 / Plan 03

- The argv contract is locked. Plan 02 (doctor + CLI) can call these builders
  without rediscovering the RVC invocation shape.
- Plan 03 (stage runner) can rely on `stage_N_is_done` + `count_dataset_inputs`
  for resume detection, `SUBPROCESS_EXTRA_ENV` for offline subprocess env,
  and `TRAIN_SUCCESS_EXIT_CODES` for the post-train exit-code check.
- `_write_filelist` currently writes unsorted-except-by-stem-name lines
  (sorted for determinism). If training requires shuffle, Plan 03 can
  post-process or the helper can be extended.

## Self-Check: PASSED

- `src/train.py` exists (534 LOC)
- `tests/unit/test_train.py` exists (554 LOC)
- Commit `794f5b6` exists (Task 1)
- Commit `08cb047` exists (Task 2)
- Commit `19caeca` exists (Task 3)
- 54 tests pass, ruff clean, module imports cleanly
