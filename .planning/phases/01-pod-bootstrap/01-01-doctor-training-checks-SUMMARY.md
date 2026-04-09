---
phase: 01-pod-bootstrap
plan: 01
name: doctor-training-checks
subsystem: doctor
tags: [bootstrap, doctor, gpu, disk, testing]
requirements: [BOOT-09, BOOT-10]
dependency_graph:
  requires: []
  provides:
    - "src.doctor.check_disk_space_floor"
    - "src.doctor.check_gpu_vram_floor"
    - "src.doctor.check_rvc_mute_refs"
    - "src.doctor.check_hubert_base"
    - "src.doctor.HUBERT_MIN_BYTES"
    - "python src/doctor.py --training"
  affects:
    - "src/doctor.py"
tech_stack:
  added: []
  patterns:
    - "CheckResult-returning, never-raising doctor check functions"
    - "unittest.mock.patch on global subprocess.run / shutil.* (mirrors tests/unit/test_doctor.py)"
    - "sparse-file trick (seek+write single byte) to test 100 MB size floor without allocating blocks"
key_files:
  created:
    - "tests/unit/test_doctor_training.py"
  modified:
    - "src/doctor.py"
decisions:
  - "check_mise deliberately excluded from --training set (pod does not need mise; D-09)"
  - "nvidia-smi (not torch) drives check_gpu_vram_floor to preserve the two-venv boundary"
  - "Disk/VRAM wrapped in zero-arg lambdas in the selected-checks list to match _run_checks() call convention"
  - "Hubert integrity is byte-count only — SHA256 deferred to V2 (T-01-01 accepted)"
metrics:
  duration: "~5 min"
  tasks: 2
  files_touched: 2
  tests_added: 17
  completed: "2026-04-09"
---

# Phase 1 Plan 01: doctor-training-checks Summary

One-liner: Adds four training-phase doctor checks (disk floor, VRAM floor, rvc mute refs, hubert_base.pt integrity) plus a composing `--training` CLI flag so Plan 02's `scripts/setup_pod.sh` can gate pod readiness with `python src/doctor.py --training`.

## What Was Built

### src/doctor.py
- `HUBERT_MIN_BYTES = 100_000_000` — module constant, exported for Phase 2 and tests.
- `check_disk_space_floor(path: Path, min_gb: int) -> CheckResult` — uses `shutil.disk_usage`, comparison is `>=` (equality is pass), converts `FileNotFoundError` and `PermissionError` into clean `ok=False` results.
- `check_gpu_vram_floor(min_gb: int) -> CheckResult` — shells to `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits`, guards with `shutil.which`, picks `max()` across GPUs, handles non-zero exits and unparseable output without raising. No torch import.
- `check_rvc_mute_refs() -> CheckResult` — minimal Phase 2 plumbing: checks `rvc/logs/mute/` exists and is non-empty.
- `check_hubert_base() -> CheckResult` — existence + size floor (`>= HUBERT_MIN_BYTES`).
- `main()` gains a `--training` typer option which composes: `python_version, ffmpeg, ffmpeg_filters, git, nvidia-smi, rvc_cloned, rvc_venv, rvc_weights, rvc_torch_cuda, slicer2_importable, disk_space_floor(PROJECT_ROOT, 20), gpu_vram_floor(12), rvc_mute_refs, hubert_base`. `check_mise` is deliberately excluded (D-09).

### tests/unit/test_doctor_training.py (new, 17 tests)
- disk: above / at-threshold / below / missing-path / permission-error (5)
- vram: single-above / multi-max-picks / below / no-driver / nvidia-smi-nonzero / unparseable (6)
- rvc mute refs: present / missing / empty-dir (3)
- hubert_base: present (sparse file) / truncated / missing (3)

All tests patch `src.doctor.RVC_DIR` AND `src.doctor.PROJECT_ROOT` to `tmp_path` so `.relative_to()` calls in failure branches resolve correctly. No GPU, no network, no real `nvidia-smi`.

## Verification

```
$ .venv/bin/ruff check src/doctor.py tests/unit/test_doctor_training.py
All checks passed!

$ .venv/bin/pytest tests/unit/test_doctor_training.py -x -q
.................                                                        [100%]
17 passed in 0.15s

$ .venv/bin/pytest tests/unit/ -q
...........................................                              [100%]
43 passed in 0.31s

$ .venv/bin/python src/doctor.py --help | grep -- "--training"
│ --training                 Run full training pre-flight set                  │
```

Acceptance criteria from the plan all hold:
- Four new `def check_*` lines present, one each.
- `HUBERT_MIN_BYTES = 100_000_000` present once.
- `grep -c "import torch" src/doctor.py` → 0 (two-venv boundary preserved).
- `--training` typer option present; `elif training:` branch present.
- `check_disk_space_floor(PROJECT_ROOT, 20)` and `check_gpu_vram_floor(12)` each appear exactly once inside the training set.
- `check_mise` not referenced inside the training branch.
- `monkeypatch.setattr('src.doctor.PROJECT_ROOT'` present in the test file.
- `pytest -k vram` collects 6, `pytest -k disk` collects 5 (plan floor: 5 and 4).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Robustness] Added `test_check_disk_space_floor_permission_error`**
- Found during: Task 2
- Issue: Plan lists PermissionError as a behavior the function must handle (`check_disk_space_floor on PermissionError → ok=False, no exception raised`) but the numbered test list in the plan only enumerated 14 tests and omitted a PermissionError case.
- Fix: Added one extra test so the documented behavior is covered. Brings the test count to 17 (plan minimum was 14, criterion was "14+").
- Files modified: tests/unit/test_doctor_training.py
- Commit: 5a5c727

**2. [Rule 2 - Robustness] Added `test_check_gpu_vram_floor_nvidia_smi_nonzero` and `test_check_rvc_mute_refs_empty_dir`**
- Found during: Task 2
- Issue: Same as above — the behavior block listed "nvidia-smi exits non-zero → ok=False, detail contains stderr" and "mute dir present but empty" as required behaviors but the numbered test list didn't include them.
- Fix: Added one test each to match the behavior spec.
- Files modified: tests/unit/test_doctor_training.py
- Commit: 5a5c727

No architectural changes. No auth gates. No checkpoints.

## Claude's Discretion Notes

- Detail-string wording for disk/vram checks is paraphrased (e.g. `"only 5.0 GiB free"`, `"largest GPU has 8.0 GiB VRAM"`) — the plan only specified substring invariants, which the tests assert.
- Sparse-file helper is implemented as `seek(size-1); write(b"\x00")` in a standalone `_make_sparse_file(path, size)` module helper inside the test file (T-01-03 mitigation). Creates a 100 MB apparent file with one block of real storage.
- VRAM floor detail picks the largest GPU via `max(mibs)` (not sum or min), matching the D-09 "largest visible GPU" wording and the `test_check_gpu_vram_floor_multi_gpu_picks_max` test.
- `check_gpu_vram_floor` short-circuits with a "no GPUs" result if `nvidia-smi` succeeds but returns an empty list — behavior not in the plan, but the alternative (IndexError on `max([])`) would violate the never-raise contract. Not documented as a separate test because it's defensive; standard ops path is either "driver missing", "driver broken", or "at least one GPU".

## Requirement Coverage

- BOOT-09 (doctor --training pre-flight set) — ✓ `--training` flag composes all required checks; disk floor 20 GiB and VRAM floor 12 GiB both wired with PROJECT_ROOT and 12 args respectively.
- BOOT-10 (unit tests for training-phase doctor checks) — ✓ 17 tests pass without a GPU, all mocked via patched `subprocess.run` / `shutil.*`.

## Threat Flags

None. The plan's threat model enumerated T-01-01..05; all were either `accept` (in-plan rationale) or `mitigate` via the sparse-file test helper (T-01-03), which was applied.

## Known Stubs

None. `check_rvc_mute_refs` and `check_hubert_base` are flagged as "minimal Phase 2 plumbing" in the plan, but they are fully functional for their stated scope (existence + non-empty for mute refs, existence + size floor for hubert). Phase 2 may tighten them (e.g. per-file name checks on mute refs) but no stubbed data or placeholder values flow anywhere.

## Self-Check: PASSED

- [x] `src/doctor.py` exists and contains the four new functions (verified via imports in verification block).
- [x] `tests/unit/test_doctor_training.py` exists (17 tests).
- [x] Commit `0cd2120` exists: `feat(01-01): add training-phase doctor checks`.
- [x] Commit `5a5c727` exists: `feat(01-01): add --training doctor flag and unit tests`.
- [x] `.venv/bin/pytest tests/unit/ -q` → 43 passed.
- [x] `.venv/bin/ruff check src/doctor.py tests/unit/test_doctor_training.py` → All checks passed.
- [x] `.venv/bin/python src/doctor.py --help | grep -- "--training"` → match.
