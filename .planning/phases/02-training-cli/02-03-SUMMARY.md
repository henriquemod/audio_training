---
phase: 02-training-cli
plan: 03
subsystem: training
tags: [training, runner, sentinel, resume, error-handling, subprocess]
requirements: [TRAIN-02, TRAIN-07, TRAIN-08, TRAIN-09, TRAIN-10, TRAIN-11, TRAIN-12, TRAIN-14]
dependency-graph:
  requires: [02-01, 02-02]
  provides:
    - "src.train.run_pipeline orchestrator"
    - "src.train._run_stage_streamed subprocess streamer"
    - "src.train._build_subprocess_env offline-env builder"
    - "src.train._is_train_success exit-code predicate"
    - "src.train._tail_file bounded log tailer"
    - "src.train._print_failure_tail failure-context reporter"
    - "Plan 02 CLI main() fully wired to run_pipeline"
  affects:
    - "tests/unit/test_train_cli.py (Plan 02 stub tests upgraded to mock run_pipeline)"
tech-stack:
  added: []
  patterns:
    - "Popen line-tee via iter(proc.stdout.readline, '') (RESEARCH.md §10.1, Risk R3)"
    - "Sentinel probe-and-skip resume (D-08/D-09)"
    - "Post-run sentinel re-probe to catch silent exit-0 failures (D-18 + Stage 3 hubert pitfall)"
key-files:
  created:
    - path: "tests/unit/test_train_runner.py"
      purpose: "27 tests covering runner helpers + run_pipeline orchestrator"
  modified:
    - path: "src/train.py"
      purpose: "Added stage runner helpers + run_pipeline + wired main()"
    - path: "tests/unit/test_train_cli.py"
      purpose: "Mock run_pipeline in the 2 Plan-02 stub-era tests; add failure-exit propagation test"
decisions:
  - "Stages 1-3 require exit code 0 (strict); only Stage 4 treats 61 as success (TRAIN-07, D-17)"
  - "Post-run sentinel re-probe follows every successful Stage 1/2/3 run to catch silent failures"
  - "Stage 3 silent-failure hint explicitly names hubert_base.pt (STATE.md pitfall)"
  - "Filelist + config are always regenerated before Stage 4 (idempotent, covers partial-resume mutations)"
metrics:
  duration: "~18 minutes"
  tasks: 2
  files: 3
  tests-added: 27
  tests-total: 158
  completed: "2026-04-09"
---

# Phase 2 Plan 03: Stage Runner Summary

End-to-end training pipeline wired: `python src/train.py --experiment-name X --dataset-dir Y` now orchestrates RVC stages 1-4 via streamed subprocesses with always-on probe-and-skip resume, exit-code 0/61 tolerance, and 30-line tail-on-failure.

## Final src/train.py Shape

- **Line count:** 1,060 LOC (up from ~715 after Plan 02; +~345 LOC for runner + orchestrator)
- **New public/private symbols added this plan:**
  - `_build_subprocess_env() -> dict[str, str]`
  - `_is_train_success(returncode: int) -> bool`
  - `_tail_file(path: Path, n: int) -> str`
  - `_run_stage_streamed(cmd, *, stage_num, stage_name, log_path, env) -> int`
  - `_print_failure_tail(log_path, *, stage, name, verbose, extra_hint="")`
  - `run_pipeline(**kwargs) -> int` — the top-level orchestrator
- **New imports:** `subprocess`, `datetime.datetime`
- **Two-venv boundary preserved:** `grep -E "^import (torch|fairseq|faiss)|^from (torch|fairseq|faiss)" src/train.py` returns nothing (TRAIN-10 / D-24).
- **Subprocess discipline preserved:** `grep -c "shell=True" src/train.py` returns 0.

## run_pipeline Orchestrator

Decision tree implemented (matches RESEARCH.md §5 + CONTEXT.md D-08/D-09/D-18):

1. **D-09 fast-path:** if `stage_4_is_done(weight_path)` → print "already complete" and return 0 (no subprocess invoked).
2. **Dataset probe:** `count_dataset_inputs(dataset_dir)`; 0 → defensive error, return 3.
3. **Stage 1 (preprocess):**
   - If `stage_1_is_done` → print "Stage 1: skipping — already populated (N files)".
   - Else run streamed subprocess; exit != 0 → tail+return 3; post-run sentinel mismatch → tail+return 3.
4. **Stage 2 (extract_f0):** same pattern, strict exit-0.
5. **Stage 3 (extract_feature):** same pattern, strict exit-0, **plus** hubert-missing hint on silent post-run mismatch.
6. **Pre-Stage 4:** `_write_filelist` + `_write_exp_config` (idempotent regeneration every run).
7. **Stage 4 (train):** streamed subprocess; success iff exit ∈ `TRAIN_SUCCESS_EXIT_CODES` (0 or 61); post-run `stage_4_is_done(weight_path)` cross-check; either failure → tail+return 3.
8. Success → print "Training complete. Output: <weight_path>" and return 0.

main() replaces the Plan 02 "TODO Plan 03" stub with `rc = run_pipeline(...); raise typer.Exit(code=rc)`.

## Test Count Delta

| File                           | Before | After | Delta |
| ------------------------------ | -----: | ----: | ----: |
| tests/unit/test_train_runner.py|      0 |    27 |   +27 |
| tests/unit/test_train_cli.py   |     21 |    22 |    +1 |
| **Total suite**                |    131 |   158 |   +27 |

`test_train_runner.py` breakdown:
- `_is_train_success`: 4 tests (zero, 61, other failures, constant)
- `_build_subprocess_env`: 4 tests (offline flags, PATH inheritance, no mutation, constant)
- `_tail_file`: 5 tests (missing, empty, fewer lines, more lines, binary garbage)
- `_run_stage_streamed`: 3 tests (success banner+stdout, nonzero exit capture, append-not-overwrite)
- `_print_failure_tail`: 5 tests (30-line, verbose 100-line, CUDA OOM hint, missing log, extra_hint)
- `run_pipeline` orchestrator: 6 tests (fast-path, skip-resume, exit-61 success, stage-1 failure, missing-weight-after-success, stage-3 silent hubert failure)

## Deviations from Plan

**1. [Rule 3 - Blocker] Plan 02 stub-era test `test_all_valid_reaches_runner_stub` obsoleted by orchestrator wiring**
- **Found during:** Task 2 full-suite run after wiring run_pipeline.
- **Issue:** The Plan 02 test asserted `"TODO Plan 03" in output`; once main() calls run_pipeline for real, the stub text is gone and the test tried to spawn an actual RVC subprocess (no rvc/.venv on this box → FileNotFoundError, exit 1).
- **Fix:** Renamed to `test_all_valid_reaches_runner` and monkeypatched `src.train.run_pipeline` with a fake returning 0. Also updated `test_preset_override_reaches_main` with the same mock. Added new `test_run_pipeline_failure_exit_code_propagates` to verify main() propagates rc=3 through `typer.Exit`.
- **Files modified:** `tests/unit/test_train_cli.py`
- **Commit:** `8697073`

**2. [Rule 3 - Blocker] Worktree lacks `rvc/` directory used as Popen cwd**
- **Found during:** Task 1 real-subprocess tests for `_run_stage_streamed`.
- **Issue:** `_run_stage_streamed` passes `cwd=RVC_DIR` which resolves to `<worktree>/rvc`; the worktree is a fresh branch checkout without the vendored RVC clone, so Popen raised FileNotFoundError before reaching the child python.
- **Fix:** Created an empty `rvc/` dir at the worktree root (already in `.gitignore`, so no diff). No code change; test cwd is valid again.
- **Files modified:** `rvc/` (untracked dir, not committed)
- **Commit:** n/a (not a tracked artifact)

No other deviations. No auth gates. No architectural decisions needed.

## Requirement Mapping

| Req       | Status | Proving artifact                                                                                     |
| --------- | :----: | ---------------------------------------------------------------------------------------------------- |
| TRAIN-02  |   ✓    | `_run_stage_streamed` uses `subprocess.Popen(..., cwd=RVC_DIR, shell=False)`; mocked test sees all 4 stage_nums |
| TRAIN-07  |   ✓    | `test_is_train_success_61` + `test_run_pipeline_stage_4_exit_61_is_success`                          |
| TRAIN-08  |   ✓    | `test_run_pipeline_skips_done_stages_runs_train` (stages 1-3 skipped, Stage 4 runs)                  |
| TRAIN-09  |   ✓    | Covered at mock level by resume test; real pod resume covered by the manual smoke run below         |
| TRAIN-10  |   ✓    | `grep -E "^(import\|from) (torch\|fairseq\|faiss)" src/train.py` returns nothing                    |
| TRAIN-11  |   ✓    | `test_print_failure_tail_writes_30_lines` + `test_run_pipeline_stage_1_failure_returns_3`           |
| TRAIN-12  |   ✓    | `test_run_pipeline_failure_exit_code_propagates` (stage failure → typer exit 3)                      |
| TRAIN-14  |   ✓    | `test_build_subprocess_env_includes_offline_flags` asserts TRANSFORMERS_OFFLINE/HF_DATASETS_OFFLINE/LANG |

## ROADMAP Success Criteria Mapping

| # | Criterion                                           | Where proved                                                           |
| - | --------------------------------------------------- | ---------------------------------------------------------------------- |
| 1 | "Run two bash scripts, walk away with a .pth"       | Pod smoke run (manual; see below)                                      |
| 2 | "Re-running resumes where last run stopped"         | `test_run_pipeline_skips_done_stages_runs_train` + manual pod re-run   |
| 3 | "Interrupted run continues from last checkpoint"    | Mocked at test level; manual pod kill-and-resume for real-level        |
| 4 | "Stages 1-4 mirror rvc/infer-web.py byte-for-byte"  | Plan 01's `test_train.py` exact-argv tests (not touched this plan)     |
| 5 | "Failure prints last 30 lines + stage context"      | `test_print_failure_tail_writes_30_lines` + `test_run_pipeline_stage_1_failure_returns_3` |

## Two-Venv Boundary Verification

```
$ grep -E "^import (torch|fairseq|faiss)|^from (torch|fairseq|faiss)" src/train.py
(no output)

$ grep -c "shell=True" src/train.py
0
```

TRAIN-10 / D-24 holds: src/train.py does not import torch, fairseq, or faiss at the Python level. All RVC interaction is via `subprocess.Popen(..., cwd=RVC_DIR, shell=False)`.

## Phase Gate: Manual Pod-Side Smoke Run (non-autonomous)

Required after merging Phase 2 to close ROADMAP criteria 1/2/3/5 at the real-level. Run on a provisioned GPU pod with `dataset/processed/` populated:

```bash
# 1. End-to-end smoke train (criterion 1)
python src/train.py --experiment-name smoke --dataset-dir dataset/processed/ --epochs 1 --batch-size 1
# Expected: exit 0, rvc/assets/weights/smoke.pth exists and is >= 1 KiB

# 2. Fast-path idempotency (criterion 2, D-09)
python src/train.py --experiment-name smoke --dataset-dir dataset/processed/ --epochs 1 --batch-size 1
# Expected: prints "Experiment 'smoke' already complete" and exits 0 in <2s

# 3. Resume from Stage 4 only (criterion 3)
rm rvc/assets/weights/smoke.pth
python src/train.py --experiment-name smoke --dataset-dir dataset/processed/ --epochs 1 --batch-size 1
# Expected: prints "Stage 1: skipping", "Stage 2: skipping", "Stage 3: skipping", runs Stage 4, exits 0

# 4. Failure path surfaces tail (criterion 5)
python src/train.py --experiment-name nonexistentname --dataset-dir dataset/processed/ --rvc-version v2 --sample-rate 12345
# Expected: exit 2 with stderr error message
```

## Commits

- `038bd32` feat(02-03): add stage runner helpers to src/train.py
- `8697073` feat(02-03): wire run_pipeline orchestrator into main()

## Self-Check: PASSED

- `src/train.py` contains `run_pipeline`, `_run_stage_streamed`, `_build_subprocess_env`, `_is_train_success`, `_tail_file`, `_print_failure_tail` ✓
- `tests/unit/test_train_runner.py` exists with 27 tests ✓
- Full suite: 158 passed, 1 deselected ✓
- Ruff clean on src/train.py + tests/unit/ ✓
- TODO Plan 03 stub removed ✓
- Two-venv boundary grep empty ✓
- Commits `038bd32` and `8697073` exist on branch ✓
