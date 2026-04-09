---
status: complete
phase: 01-pod-bootstrap
source:
  - 01-01-doctor-training-checks-SUMMARY.md
  - 01-02-setup-pod-script-SUMMARY.md
started: 2026-04-09T16:40:14-03:00
updated: 2026-04-09T16:50:00-03:00
---

## Current Test

[testing complete]

## Tests

### 1. Cold Start Smoke Test
expected: On a fresh Ubuntu 22.04 pod with NVIDIA driver only, `bash scripts/setup_pod.sh` completes non-interactively with exit 0. Installs ffmpeg >=5, CUDA 12.1, Python 3.10, app venv, RVC venv, and pretrained weights. Final `doctor --training` step prints 14 OK checks.
result: pass

### 2. Warm Re-run Under 30s
expected: Re-running `bash scripts/setup_pod.sh` on an already-provisioned pod exits 0 in well under 30 seconds. Output shows skip messages for apt prereqs, ffmpeg, CUDA, app venv, and RVC delegation. No network I/O beyond the dpkg probe.
result: pass

### 3. doctor --training Flag
expected: Running `.venv/bin/python src/doctor.py --training` on a provisioned pod prints a rich table with 14 checks (python_version, ffmpeg, ffmpeg_filters, git, nvidia-smi, rvc_cloned, rvc_venv, rvc_weights, rvc_torch_cuda, slicer2_importable, disk_space_floor, gpu_vram_floor, rvc_mute_refs, hubert_base), all OK, exit 0. `check_mise` is NOT in the list.
result: pass

### 4. doctor --training Disk Floor Check
expected: `check_disk_space_floor(PROJECT_ROOT, 20)` reports OK when ≥20 GiB free at project root. If disk is below 20 GiB, check reports ok=False with a clear "only X GiB free" detail. Missing-path and permission errors are handled gracefully without raising.
result: pass

### 5. doctor --training GPU VRAM Floor Check
expected: `check_gpu_vram_floor(12)` reports OK when the largest visible GPU has ≥12 GiB VRAM. Multi-GPU hosts pick the largest via max(). No torch import — nvidia-smi drives the check. Missing driver or nonzero nvidia-smi returns ok=False without raising.
result: pass

### 6. hubert_base.pt Integrity Check
expected: `check_hubert_base` reports OK when `rvc/assets/hubert/hubert_base.pt` exists and is ≥100 MB (HUBERT_MIN_BYTES). A truncated or missing file reports ok=False with a clear detail.
result: pass

### 7. ffmpeg Nightly Build Parse
expected: `parse_ffmpeg_version` accepts the BtbN nightly tag (e.g. `N-123884-gd3d0b7a5ee-20260409`) and returns sentinel `(9999, 0, 0)`. `check_ffmpeg` reports OK with the raw nightly version string in the detail field, not a misleading `9999.0.0`. `doctor --training` passes with the BtbN static build installed.
result: pass

### 8. check_mise Soft-OK on Pod
expected: On a pod without mise installed, `check_mise` returns ok=True with detail "not installed (optional — only needed on dev laptops)". Doctor table shows OK, no "Fix the following" banner, exit 0. If mise binary is present but broken (nonzero exit), check still reports failure so laptop users notice.
result: pass

### 9. setup_pod.sh Non-root Fail Fast
expected: Running `bash scripts/setup_pod.sh` as a non-root user (EUID != 0) exits immediately with a clear "POD-ONLY" error message. No apt calls are attempted.
result: pass

### 10. setup_rvc.sh Unchanged
expected: `scripts/setup_rvc.sh` is byte-for-byte identical to the version before phase 1 (BOOT-05 contract). `git diff` against the phase-1 base shows zero changes to this file.
result: pass
verified: git log --all -- scripts/setup_rvc.sh shows no phase-1 commits

### 11. Unit Tests Pass
expected: `.venv/bin/pytest tests/unit/ -q` runs and all 43+ tests pass (17 from test_doctor_training.py plus 8 new tests in test_doctor.py, plus pre-existing tests). No network, no GPU, no real nvidia-smi required.
result: pass
verified: .venv/bin/pytest tests/unit/ -q → 52 passed in 0.42s

## Summary

total: 11
passed: 11
issues: 0
pending: 0
skipped: 0

## Gaps

[none yet]
