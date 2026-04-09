# Roadmap: train_audio_model — Pod-Ready Training

**Milestone:** Pod-Ready Training
**Granularity:** Coarse (3-5 phases)
**Created:** 2026-04-09
**Coverage:** 41/41 v1 requirements mapped

## Phases

- [ ] **Phase 1: Pod Bootstrap** - `scripts/setup_pod.sh` takes a bare Ubuntu + NVIDIA-driver pod to fully provisioned, idempotently
- [ ] **Phase 2: Training CLI** - `python src/train.py` runs the full RVC training pipeline end-to-end with sentinel-based resume and doctor pre-flight
- [ ] **Phase 3: Index Training + Auto-Export** - FAISS index build (Stage 5) and deterministic export to `models/<name>/` complete the pipeline
- [ ] **Phase 4: Shell Orchestration + Remote Source** - `scripts/train.sh` one-liner and `--dataset-url` remote pull make the pod workflow self-contained
- [ ] **Phase 5: Pod Shutdown Documentation** - README section documents end-to-end pod workflow and per-provider auto-stop patterns

## Phase Details

### Phase 1: Pod Bootstrap
**Goal**: A single non-interactive invocation of `scripts/setup_pod.sh` provisions any bare Ubuntu + NVIDIA-driver pod — installing CUDA toolkit, mise, Python 3.10, app venv, and RVC venv — and re-runs in ~10 seconds on an already-provisioned pod.
**Depends on**: Nothing (first phase)
**Requirements**: BOOT-01, BOOT-02, BOOT-03, BOOT-04, BOOT-05, BOOT-06, BOOT-07, BOOT-08, BOOT-09, BOOT-10
**Complexity**: Large
**Success Criteria** (what must be TRUE):
  1. Running `bash scripts/setup_pod.sh` on a clean Ubuntu 22.04 + NVIDIA-driver image completes without any interactive prompt and exits 0, leaving `.venv/bin/python`, `rvc/.venv/bin/python`, and all pretrained weight files present on disk
  2. Re-running `bash scripts/setup_pod.sh` on an already-provisioned pod completes in under 30 seconds (all probe-and-skip sentinels fire correctly) and exits 0 without modifying any venv
  3. Running `python src/doctor.py --training` passes all checks on a provisioned pod: CUDA available, disk space floor met, GPU VRAM floor met, pretrained v2 weights present, hubert base present
  4. Unit tests for `check_disk_space_floor` and `check_gpu_vram_floor` pass in CI without a real GPU (mocked `shutil.disk_usage` and mocked `rvc/.venv` torch call)
**Plans**: TBD
**UI hint**: no

### Phase 2: Training CLI
**Goal**: `python src/train.py --experiment-name <name> --dataset-dir <path>` orchestrates all four RVC training stages end-to-end with hyperparameter flags, sentinel-based skip-if-done resume, doctor pre-flight checks, and structured error output.
**Depends on**: Phase 1
**Requirements**: TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04, TRAIN-05, TRAIN-06, TRAIN-07, TRAIN-08, TRAIN-09, TRAIN-10, TRAIN-11, TRAIN-12, TRAIN-13, TRAIN-14
**Complexity**: Large
**Success Criteria** (what must be TRUE):
  1. `python src/train.py --experiment-name smoke --dataset-dir dataset/processed/ --epochs 1 --batch-size 1` runs all four pipeline stages and exits 0 (or 61) on a pod with a GPU, leaving `rvc/assets/weights/smoke.pth` on disk
  2. Re-invoking the same command for a completed experiment skips already-done stages (verified by log output showing "stage N: skipping — sentinel present") without re-processing any audio
  3. A run aborted after Stage 2 (SIGKILL simulation) re-invoked completes from Stage 3 onward, not from Stage 1, and produces the same final weight as an uninterrupted run would
  4. `python -m pytest tests/unit/test_train.py` passes without a GPU: all four builder functions (`build_rvc_preprocess_cmd`, `build_rvc_extract_f0_cmd`, `build_rvc_extract_feature_cmd`, `build_rvc_train_cmd`) assert exact argv contents for representative inputs
  5. On a deliberate training failure (bad experiment name), exit code is 3 and the last 30 lines of the failed subprocess stderr are printed to the terminal with stage context
**Plans**: TBD
**UI hint**: no

### Phase 3: Index Training + Auto-Export
**Goal**: Stage 5 (FAISS index build) is wired into the pipeline via a vendored shim committed under `scripts/rvc_patches/` and copied to `rvc/tools/` at bootstrap; after successful training + indexing, the pipeline auto-exports `.pth` and newest-by-mtime `added_*.index` to `models/<name>/` with a `manifest.json`.
**Depends on**: Phase 2
**Requirements**: INDEX-01, INDEX-02, INDEX-03, INDEX-04, INDEX-05, INDEX-06, INDEX-07
**Complexity**: Medium
**Success Criteria** (what must be TRUE):
  1. After a complete training run, `models/<experiment-name>/` contains `<name>.pth`, an `added_*.index` file, and `manifest.json` — matching the layout `scripts/install_model.sh` produces — without any manual intervention
  2. `manifest.json` contains experiment name, sample rate, RVC version, epoch count, batch size, f0 method, dataset directory, training start/end timestamps, and the RVC pinned commit hash
  3. Re-running `python src/train.py` for an already-exported experiment overwrites `models/<name>/` in place, emits "already exported — overwriting" to stdout, and exits 0 without leaving a corrupt partial export
  4. `python -m pytest tests/unit/test_train.py` covers `build_rvc_train_index_cmd`, the newest-by-mtime `added_*.index` picker, and the manifest writer — all passing without a GPU
**Plans**: TBD
**UI hint**: no

### Phase 4: Shell Orchestration + Remote Source
**Goal**: `scripts/train.sh <experiment-name> <dataset-source>` is the single user-facing command for a full pod run; `--dataset-url <URL>` in `src/train.py` pulls a remote dataset archive via `curl -fL` before training begins; sentinel files (`DONE`/`FAILED`) are written on exit so external watchdogs can detect completion.
**Depends on**: Phase 3
**Requirements**: ORCH-01, ORCH-02, ORCH-03, ORCH-04, ORCH-05, ORCH-06
**Complexity**: Small
**Success Criteria** (what must be TRUE):
  1. `bash scripts/train.sh myvoice dataset/processed/` runs doctor pre-flight, then `src/train.py`, and writes `models/myvoice/DONE` on success or `models/myvoice/FAILED` on failure — verified by inspecting the file after each outcome
  2. `python src/train.py --dataset-url https://example.com/audio.zip` downloads the archive, extracts it to a scratch directory, and proceeds identically to `--dataset-dir` for the rest of the run
  3. Specifying both `--dataset-dir` and `--dataset-url` exits immediately with code 2 and a clear mutual-exclusion error before any download or training work begins
  4. A `--dataset-url` pointing to a non-existent resource (404) exits with code 1 before any billable training step starts, printing the HTTP error to stderr
**Plans**: TBD
**UI hint**: no

### Phase 5: Pod Shutdown Documentation
**Goal**: `README.md` contains a complete "Training on a GPU pod" section that an unfamiliar user can follow from renting a bare pod to downloading an exported model, including copy-pasteable auto-stop patterns for RunPod, Vast.ai, Lambda Labs, and generic systemd keyed on the `DONE`/`FAILED` sentinel files.
**Depends on**: Phase 4
**Requirements**: DOCS-01, DOCS-02, DOCS-03, DOCS-04
**Complexity**: Small
**Success Criteria** (what must be TRUE):
  1. README contains a "Training on a GPU pod" section with step-by-step instructions covering: pod provisioning, running `setup_pod.sh`, uploading or pulling training audio, running `train.sh`, and downloading `models/<name>/`
  2. README explicitly states that Ubuntu 22.04 is recommended and that Ubuntu 24.04 CUDA 12.1 apt install is unsupported/best-effort
  3. README contains copy-pasteable auto-shutdown snippets for RunPod (`runpodctl stop pod`), Vast.ai (`vastctl destroy`), Lambda Labs, and a generic `systemd` timer, all wired to the `DONE`/`FAILED` sentinel files
  4. README explicitly lists what is NOT supported on the pod path (web UI, multi-GPU, provider SDKs, auto-shutdown-in-code, smoke inference on the pod) so future contributors can see the scope boundary at a glance
**Plans**: TBD
**UI hint**: no

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Pod Bootstrap | 0/? | Not started | - |
| 2. Training CLI | 0/? | Not started | - |
| 3. Index Training + Auto-Export | 0/? | Not started | - |
| 4. Shell Orchestration + Remote Source | 0/? | Not started | - |
| 5. Pod Shutdown Documentation | 0/? | Not started | - |

---
*Roadmap created: 2026-04-09*
*Last updated: 2026-04-09 after initial creation*
