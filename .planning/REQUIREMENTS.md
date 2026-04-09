# Requirements: train_audio_model — Pod-Ready Training

**Defined:** 2026-04-09
**Core Value:** A single user — me — can rent a GPU pod, run two bash scripts, and walk away with a downloadable `.pth` + `.index` voice model trained from raw audio I provided.

## v1 Requirements

Requirements for the Pod-Ready Training milestone. Each maps to exactly one roadmap phase.

### Bootstrap (BOOT)

- [ ] **BOOT-01**: `scripts/setup_pod.sh` takes a bare Ubuntu 22.04 + NVIDIA-driver pod to fully provisioned in one non-interactive invocation (no TTY prompts, no `sudo` surprises)
- [ ] **BOOT-02**: Bootstrap script uses a detect-and-adapt probe pattern — each layer (CUDA toolkit 12.1 → mise → Python 3.10 → app venv → RVC venv → RVC pinned clone → RVC weights) is installed only if missing, so re-running on a healthy pod completes in ~10 seconds
- [ ] **BOOT-03**: Bootstrap installs CUDA toolkit 12.1 via the NVIDIA apt keyring on Ubuntu 22.04 without interactive prompts (`DEBIAN_FRONTEND=noninteractive`, `TZ=UTC`)
- [ ] **BOOT-04**: Bootstrap installs Python 3.10 via `mise` using non-interactive patterns (`mise exec` or `$(mise where python)/bin/python3`), not `mise activate bash`
- [ ] **BOOT-05**: Bootstrap creates `./.venv` as Python 3.10 with the project installed editable (`pip install -e ".[dev]"`), then delegates to `scripts/setup_rvc.sh` for `rvc/.venv` creation (does NOT modify `setup_rvc.sh`)
- [ ] **BOOT-06**: Bootstrap verifies `torch.cuda.is_available()` returns True in `rvc/.venv` as a post-install sanity check, surfacing a diagnostic hint if CUDA + driver + torch don't agree
- [ ] **BOOT-07**: Bootstrap preserves the existing `pip<24.1` pin in `rvc/.venv` — no bootstrap step upgrades pip in that venv
- [ ] **BOOT-08**: Bootstrap ensures `rvc/assets/pretrained_v2/`, `rvc/assets/hubert/hubert_base.pt`, `rvc/assets/rmvpe.pt`, and `rvc/logs/mute/` are populated before finishing, by calling `rvc/tools/download_models.py` and asserting file sizes
- [ ] **BOOT-09**: `src/doctor.py` gains `check_disk_space_floor(path, min_gb)` and `check_gpu_vram_floor(min_gb)` checks, plus a `--training` flag that runs the full training pre-flight check set
- [ ] **BOOT-10**: Bootstrap is fully covered by unit tests for the new doctor check functions (no integration test required on a real GPU)

### Training CLI (TRAIN)

- [ ] **TRAIN-01**: `python src/train.py` accepts `--experiment-name`, `--dataset-dir`, `--sample-rate` (32000/40000/48000), `--epochs`, `--batch-size`, `--rvc-version` (v1/v2), `--f0-method` (pm/harvest/rmvpe/rmvpe_gpu), `--save-every`, and `--resume` flags with sensible defaults
- [ ] **TRAIN-02**: `src/train.py` orchestrates the full RVC training pipeline stages 1-4 end-to-end: RVC preprocess (Stage 1) → F0 extraction (Stage 2) → feature extraction (Stage 3) → model train (Stage 4), each via `subprocess.run(cwd=RVC_DIR)`
- [ ] **TRAIN-03**: Four pure arg-builder functions (`build_rvc_preprocess_cmd`, `build_rvc_extract_f0_cmd`, `build_rvc_extract_feature_cmd`, `build_rvc_train_cmd`) return `list[str]` with zero side effects, following the `build_rvc_subprocess_cmd` pattern from `src/generate.py`
- [ ] **TRAIN-04**: Every builder function is covered by unit tests in `tests/unit/test_train.py` that assert exact argv contents for representative inputs — no GPU required
- [ ] **TRAIN-05**: A `_write_filelist(exp_dir, version, sample_rate)` helper generates `rvc/logs/<exp>/filelist.txt` in the exact format `infer-web.py:click_train` produces (lines 500-546 of the pinned commit), including mute-reference rows from `rvc/logs/mute/`, and asserts the resulting file is non-empty before training starts
- [ ] **TRAIN-06**: `src/train.py` calls new doctor pre-flight checks before any billable work: `check_pretrained_v2_weights(sample_rate, version, if_f0)`, `check_training_dataset_nonempty(dataset_dir)`, `check_rvc_mute_refs()`, `check_hubert_base()` (existence AND minimum size)
- [ ] **TRAIN-07**: Training invocation treats `train.py` exit codes **0 AND 61** as success (RVC's `os._exit(2333333)` truncates to 61 on Linux), and cross-checks by verifying `rvc/assets/weights/<name>.pth` exists after the process returns
- [ ] **TRAIN-08**: Training is resumable via sentinel-based skip-if-done guards: re-invoking `src/train.py` for the same experiment skips Stage 1 if `0_gt_wavs/` is populated, Stage 2 if both `2a_f0/` and `2b-f0nsf/` are populated, Stage 3 if `3_feature768/` (v2) / `3_feature256/` (v1) is populated, and lets RVC's built-in checkpoint resume handle Stage 4
- [ ] **TRAIN-09**: A training run aborted mid-pipeline (SIGKILL, pod reboot, network drop) resumes from the last completed stage on the next invocation without re-doing prior work, and without producing a corrupt model
- [ ] **TRAIN-10**: `src/train.py` enforces the two-venv boundary — no `import torch`, `import fairseq`, or `import faiss` anywhere in the file; every subprocess is invoked with `cwd=RVC_DIR, shell=False`
- [ ] **TRAIN-11**: On training failure, `src/train.py` prints the last 30 lines of the failed subprocess stderr with pipeline-stage context (matching the `_tail` pattern from `src/generate.py`), and exits with code 3
- [ ] **TRAIN-12**: Exit codes follow the existing convention: 0 on success, 1 on config/setup error (missing weights, missing dataset, missing hubert), 2 on user-input error (bad CLI flags), 3 on training subprocess failure
- [ ] **TRAIN-13**: `--sample-rate` passed to `src/train.py` drives *both* the RVC preprocess resample target AND the `train.py -sr` flag, so there's no possible chain mismatch between stages
- [ ] **TRAIN-14**: `src/train.py` sets `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, and `LANG=C.UTF-8` in the subprocess environment to prevent network probes and locale hangs on minimal pod images

### Index Training + Auto-Export (INDEX)

- [ ] **INDEX-01**: A vendored shim `scripts/rvc_patches/train_index_cli.py` (committed in the app repo under `scripts/rvc_patches/`, copied to `rvc/tools/` by `scripts/setup_pod.sh` at bootstrap time) runs the FAISS index build logic from `rvc/infer-web.py:train_index` (lines 616-700) as a standalone CLI, accepting experiment name and RVC version as arguments
- [ ] **INDEX-02**: `src/train.py` gains `build_rvc_train_index_cmd` (pure arg builder, unit-tested) and invokes the FAISS shim as Stage 5 of the pipeline via `rvc/.venv` subprocess
- [ ] **INDEX-03**: After successful training + indexing, `src/train.py` auto-exports the trained model to `models/<experiment-name>/`: copies `rvc/assets/weights/<name>.pth` and the **newest by mtime** `rvc/logs/<name>/added_*.index` into `models/<name>/`, matching the existing `scripts/install_model.sh` layout
- [ ] **INDEX-04**: Auto-export writes a `models/<name>/manifest.json` summarizing: experiment name, sample rate, RVC version, epoch count, batch size, f0 method, dataset directory, training start/end timestamps, RVC pinned commit hash, final checkpoint filename
- [ ] **INDEX-05**: Auto-export is idempotent — re-running on a completed experiment overwrites in place without corrupting a previously exported model, and emits a clear "already exported" status line
- [ ] **INDEX-06**: Auto-export verifies each exported artifact is readable (file exists, non-empty, `.pth` loadable as a torch state dict via the existing `rvc/.venv` if verification is cheap) before declaring success
- [ ] **INDEX-07**: Stage 5 + auto-export are unit-tested via `tests/unit/test_train.py` fixtures that exercise `build_rvc_train_index_cmd`, the newest-by-mtime picker, and the manifest writer — no GPU required

### Shell Orchestration + Remote Source (ORCH)

- [ ] **ORCH-01**: `scripts/train.sh <experiment-name> <dataset-source> [additional flags...]` is a thin one-liner that validates inputs, calls `src/doctor.py --training`, then invokes `src/train.py` with forwarded arguments
- [ ] **ORCH-02**: `src/train.py --dataset-url <URL>` pulls a remote dataset archive via `curl -fL` (no boto3, no AWS SDK, no rclone), extracts it to a local scratch directory, and treats the result as `--dataset-dir` for the rest of the run
- [ ] **ORCH-03**: `--dataset-url` supports plain HTTP/HTTPS URLs and pre-signed S3 URLs (which are just HTTPS); presigning is the user's responsibility
- [ ] **ORCH-04**: `--dataset-dir` and `--dataset-url` are mutually exclusive; specifying both exits with code 2 and a clear error
- [ ] **ORCH-05**: Remote pull failures (non-200 status, empty archive, unsupported archive format) exit with code 1 before any billable work starts
- [ ] **ORCH-06**: `scripts/train.sh` writes a `DONE` sentinel file in `models/<experiment-name>/` on success and a `FAILED` sentinel on failure, so an external watchdog can detect completion without polling the Python process

### Documentation (DOCS)

- [ ] **DOCS-01**: `README.md` gains a "Training on a GPU pod" section that documents the end-to-end flow: provisioning a bare Ubuntu 22.04 pod with an NVIDIA GPU, running `scripts/setup_pod.sh`, uploading or pulling training audio, running `scripts/train.sh`, and downloading the exported model
- [ ] **DOCS-02**: README documents the "Ubuntu 22.04 recommended" stance and explicitly calls out that 24.04 CUDA install is best-effort
- [ ] **DOCS-03**: README's pod section includes copy-pasteable auto-shutdown patterns for RunPod (`runpodctl stop pod`), Vast.ai (`vastctl destroy`), Lambda Labs, and a generic `systemd-timer` approach — all wired to the `DONE`/`FAILED` sentinel files written by `scripts/train.sh`
- [ ] **DOCS-04**: README explicitly lists what is NOT supported on the pod path (web UI, multi-GPU, provider SDKs, auto-shutdown-in-code, smoke inference on the pod) so the scope is clear to future contributors

## v2 Requirements

Deferred beyond this milestone. Acknowledged but not in the current roadmap.

### Training Ergonomics

- **V2-TRAIN-01**: Smart batch-size default based on detected GPU VRAM
- **V2-TRAIN-02**: Structured progress output (tqdm + `rich.Progress`) with ETA during long training runs
- **V2-TRAIN-03**: Training hang detection with diagnostic timeouts on `MASTER_PORT` binding (may be required if real pod providers block localhost ports)

### Dataset Tooling

- **V2-DATA-01**: Dataset quality report (total duration, clip count distribution, RMS histogram, clipping warnings)
- **V2-DATA-02**: Support additional archive formats beyond the initial HTTPS-zip/tar path

### Fast Setup (explicitly deferred)

- **V2-FAST-01**: Aggressive caching of pretrained weights across pod reboots via persistent volume
- **V2-FAST-02**: Prebuilt wheel cache for `rvc/.venv` installs

## Out of Scope

Explicitly excluded. Documented to prevent scope creep and accidental re-addition.

| Feature | Reason |
|---------|--------|
| Web UI (RVC `infer-web.py` for training) | Incompatible with headless pod use — requires browser + port forwarding + manual clicks per run |
| Provider-specific SDKs (RunPod, Vast.ai, Lambda Labs APIs) | Too much surface area; "generic Linux + NVIDIA driver" is the explicit contract |
| Auto-shutdown implementation (`shutdown -h now` or API-based) | Risk of terminating a billable resource before export is verified downloaded; documented only |
| Smoke inference on the pod after training | Adds complexity and cost; user runs inference elsewhere |
| Multi-GPU / distributed training | Single-GPU pods only; simplicity matches actual usage |
| Cloud storage SDKs (boto3, awscli-v2, rclone as hard deps) | `curl -fL` against a presigned URL is enough for v1 |
| YAML/TOML config files | CLI flags match existing `src/generate.py` pattern and remove an upload-to-pod step |
| Automatic checkpoint cleanup / retention policy | Pod storage is ephemeral; cleanup is the user's problem |
| Tensorboard dashboard | Adds port forwarding and daemon complexity incompatible with headless use |
| Inference improvements (batching, longer text, error recovery) | Explicitly frozen this milestone — focus |
| Modifying `scripts/setup_rvc.sh` | Script is correct and tested; bootstrap wraps it rather than changes it |
| New app-venv runtime dependencies | Vendoring over dependency creep; no new pip packages unless absolutely necessary |

## Traceability

Empty initially. Populated by `gsd-roadmapper` during Step 8.

| Requirement | Phase | Status |
|-------------|-------|--------|
| BOOT-01 | — | Pending |
| BOOT-02 | — | Pending |
| BOOT-03 | — | Pending |
| BOOT-04 | — | Pending |
| BOOT-05 | — | Pending |
| BOOT-06 | — | Pending |
| BOOT-07 | — | Pending |
| BOOT-08 | — | Pending |
| BOOT-09 | — | Pending |
| BOOT-10 | — | Pending |
| TRAIN-01 | — | Pending |
| TRAIN-02 | — | Pending |
| TRAIN-03 | — | Pending |
| TRAIN-04 | — | Pending |
| TRAIN-05 | — | Pending |
| TRAIN-06 | — | Pending |
| TRAIN-07 | — | Pending |
| TRAIN-08 | — | Pending |
| TRAIN-09 | — | Pending |
| TRAIN-10 | — | Pending |
| TRAIN-11 | — | Pending |
| TRAIN-12 | — | Pending |
| TRAIN-13 | — | Pending |
| TRAIN-14 | — | Pending |
| INDEX-01 | — | Pending |
| INDEX-02 | — | Pending |
| INDEX-03 | — | Pending |
| INDEX-04 | — | Pending |
| INDEX-05 | — | Pending |
| INDEX-06 | — | Pending |
| INDEX-07 | — | Pending |
| ORCH-01 | — | Pending |
| ORCH-02 | — | Pending |
| ORCH-03 | — | Pending |
| ORCH-04 | — | Pending |
| ORCH-05 | — | Pending |
| ORCH-06 | — | Pending |
| DOCS-01 | — | Pending |
| DOCS-02 | — | Pending |
| DOCS-03 | — | Pending |
| DOCS-04 | — | Pending |

**Coverage:**
- v1 requirements: 41 total
- Mapped to phases: 0 (roadmap not yet created)
- Unmapped: 41 ⚠️ (expected — traceability fills in at Step 8)

---
*Requirements defined: 2026-04-09*
*Last updated: 2026-04-09 after initial definition*
