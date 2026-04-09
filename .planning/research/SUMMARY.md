# Project Research Summary

**Project:** train_audio_model — Pod-Ready Training milestone
**Domain:** Headless RVC voice-model training on rented GPU pods (brownfield extension)
**Researched:** 2026-04-09
**Confidence:** HIGH (all findings sourced from the vendored RVC clone at the pinned commit)

## Executive Summary

Brownfield CLI extension to an existing Edge-TTS → ffmpeg → RVC inference pipeline. Goal: one bash script provisions a bare Ubuntu + NVIDIA-driver pod, and one Python command trains a voice model end-to-end. The existing codebase already provides the right patterns — doctor-first architecture, two-venv isolation, pure arg-builder functions, subprocess-only RVC invocations — and every new component slots into those patterns without deviation. The approach is strictly additive: `scripts/setup_pod.sh` wraps `setup_rvc.sh`, and `src/train.py` wraps RVC's four training subprocess scripts, mirroring how `src/generate.py` wraps RVC's inference.

RVC's training pipeline has several hard gotchas confirmed directly in the vendored source. The dominant risk throughout is **silent failure on a billing pod**: processes that exit 0 while having produced nothing useful, training that completes with plausible loss curves but produces an unusable model, and bash scripts that hang on interactive prompts. The mitigation is the same for all: pre-flight doctor checks before any billable step, post-step assertions on sentinel artifacts, and strict non-interactive conventions in all scripts.

---

## Stack Recommendations

Almost entirely fixed by upstream RVC constraints. No new runtime dependencies in either venv.

**Fixed upstream constraints (cannot change):**
- Python 3.10 — pinned in `pyproject.toml` and `.mise.toml`
- torch 2.1.2 + torchaudio 2.1.2, CUDA 12.1 wheels — in `rvc/.venv`
- fairseq 0.12.2 — requires `pip<24.1` (legacy PEP 440 metadata)
- RVC pinned at commit `7ef19867780cf703841ebafb565a4e47d1ea86ff`
- Two-venv split: `.venv` (app) and `rvc/.venv` (torch/fairseq) — load-bearing

**New infrastructure needed:**
- `cuda-toolkit-12-1` via NVIDIA apt repo (Ubuntu 22.04 only; 24.04 requires runfile installer with `--toolkit --silent --override`)
- `mise` — use `$(mise where python)/bin/python3` or `mise exec` in scripts; never `mise activate bash` in non-interactive contexts
- `curl -fL` for remote dataset pull — no SDK, already present on every Ubuntu server image
- Progress reporting: pipe-and-tee subprocess stdout; `rich.Console` for wrapper-level status only

**Base image recommendation:** Ubuntu 22.04. Ubuntu 24.04 CUDA 12.1 apt install is unsupported upstream; runfile-install path exists but is MEDIUM-confidence until validated.

---

## Feature Categories

### Table stakes (must have)
- `scripts/setup_pod.sh` — detect-and-adapt bootstrap; probes each layer; idempotent; ~10 seconds re-run on healthy pod
- `python src/train.py` with `--experiment-name`, `--dataset-dir`, `--sample-rate`, `--epochs`, `--batch-size`, `--rvc-version`, `--f0-method`, `--save-every`
- Four pure arg-builder functions (`build_rvc_preprocess_cmd`, `build_rvc_extract_f0_cmd`, `build_rvc_extract_feature_cmd`, `build_rvc_train_cmd`) — unit-testable without GPU
- `_write_filelist` — pure Python, replicates `infer-web.py:click_train` lines 500-546 including mute-reference rows
- Skip-if-done guards on all pipeline stages (sentinel-based resume)
- Pre-flight doctor checks: pretrained weights (existence + size), GPU VRAM floor, disk space floor, dataset non-empty
- Auto-export: `models/<name>/<name>.pth` + newest `added_*.index` by mtime
- Exit codes: 0 = success, 1 = setup/config error, 3 = training subprocess failure
- Remote dataset pull via `--dataset-url` (curl only)

### Differentiators
- Experiment manifest `models/<name>/manifest.json` (hyperparams + dataset hash + base weight commit + training duration)
- Smart batch-size default from VRAM probe
- Structured log tail on failure (last 30 lines to stderr, with full log retained on disk)
- Pod shutdown documentation in README (per-provider patterns, no code)

### Anti-features (must not build this milestone)
- Web UI of any kind (RVC webui included)
- Provider-specific SDKs (RunPod, Vast.ai, Lambda Labs APIs)
- Auto-shutdown implementation (docs only)
- Smoke inference on the pod after training
- Multi-GPU / distributed training
- boto3 / AWS SDK / rclone as hard deps
- YAML/TOML config files (CLI flags only)
- Automatic checkpoint cleanup / retention policy
- Tensorboard dashboard
- Inference improvements (batching, longer text, better error recovery)

---

## Architecture Decisions

Strictly additive. No existing files change behavior except `src/doctor.py` (adds check functions + `--training` flag) and `tests/unit/test_doctor.py`.

**New files:**
- `scripts/setup_pod.sh` (Phase 1)
- `src/train.py` (Phase 2)
- `tests/unit/test_train.py` (Phase 2)
- `scripts/rvc_patches/train_index_cli.py` (Phase 3 — copied to `rvc/tools/` at bootstrap since `rvc/` is gitignored)
- `scripts/train.sh` (Phase 4)

**Critical architectural constraints:**
- `src/train.py` must never `import torch`, `import faiss`, `import fairseq` — two-venv boundary is absolute
- All subprocess builders return `list[str]`; invoked with `cwd=RVC_DIR`, `shell=False`
- `_write_filelist` and `build_rvc_train_index_cmd` are entirely new responsibilities with no existing RVC CLI equivalent
- `src/train.py` must follow `src/generate.py` exactly: pure helpers + doctor pre-flight + thin orchestrator + `run_ffmpeg` or subprocess wrapper discipline

**Two-preprocess chain (not alternatives):**
- Step A: `src/preprocess.py` (in `.venv`, 44.1 kHz canonical output) — user runs locally before pod upload to produce `dataset/processed/`
- Step B: `rvc/infer/modules/train/preprocess.py` (in `rvc/.venv`, resamples to 32/40/48k) — run by `src/train.py` on the pod, consumes Step A output

They chain. Neither replaces the other. Step A is optional on the pod itself if the user pre-processed locally.

**Bootstrap layering:** `scripts/setup_pod.sh` handles system-level install (CUDA toolkit, mise, Python 3.10, app venv) and then `exec`s into `setup_rvc.sh`. Do NOT modify `setup_rvc.sh`; it is correct and tested.

---

## Critical Pitfalls

1. **`train.py` exits code 61 on success** (CRIT-5) — `os._exit(2333333)` at line 635 truncates to 61 on Linux. Treat exit codes 0 AND 61 as success. Cross-check by verifying `rvc/assets/weights/<name>.pth` exists afterward.

2. **Missing pretrained weights → silent random-init** (CRIT-1) — bare `except:` at lines 225-254 in `train.py` falls through to random init with no error. `check_pretrained_v2_weights(sr, version, if_f0)` must block before training starts. Always pass `-pg` / `-pd` explicitly.

3. **`filelist.txt` is our responsibility** — no RVC subprocess script creates it. Replicate `infer-web.py:click_train` lines 500-546 in `_write_filelist`. Include mute-reference rows from `rvc/logs/mute/`. Assert non-empty before invoking `train.py`.

4. **FAISS index build has no standalone CLI** — `infer-web.py:train_index` lines 616-700 is in-process only. Implement `scripts/rvc_patches/train_index_cli.py` (replicate the ~30-line faiss block). Commit under `scripts/rvc_patches/`; copy to `rvc/tools/` at bootstrap.

5. **`extract_feature_print.py` exits 0 on missing hubert model** (CRIT-4) — `exit(0)` if `assets/hubert/hubert_base.pt` is absent. Doctor pre-flight must check file existence AND size (~360 MB).

6. **`added_*.index` correct picking rule** — IVF cluster count in filename changes with dataset size; multiple files may exist. Always: `sorted(glob("added_*.index"), key=os.path.getmtime)[-1]`.

7. **`mise activate bash` fails in non-interactive scripts** — use `$(mise where python)/bin/python3` or `mise exec`.

8. **`DEBIAN_FRONTEND=noninteractive TZ=UTC` required on all apt calls** — `tzdata` prompts interactively; hangs cost money on billing pods.

9. **`pip<24.1` in `rvc/.venv` must never be upgraded** — fairseq 0.12.2 has legacy PEP 440 metadata that pip 24.1+ rejects.

10. **Version/weights pairing is strict** — v1 → `assets/pretrained/`, v2 → `assets/pretrained_v2/`. Mixing causes silent shape-mismatch via `load_state_dict(strict=False)`.

11. **Sample-rate chain mismatch** (CRIT-2) — `src/preprocess.py` outputs 44.1 kHz; RVC preprocess must resample to `--sample-rate`. Passing the wrong `--sample-rate` flag silently trains on downsampled audio with degraded quality.

12. **Checkpoint pair rule** — RVC resume silently falls back to random init if only G or only D checkpoint exists. Pre-flight for resume: assert both `G_*.pth` and `D_*.pth` exist, or assert neither does.

13. **`-l 1` (save-latest) writes fixed filenames** `G_2333333.pth` / `D_2333333.pth`. If a previous run used `-l 0`, both numbered files and `2333333` may coexist; `latest_checkpoint_path` always picks `2333333` by digit sort. Always use a consistent `-l` value per experiment.

---

## Key Facts the Roadmap Must Not Forget

**RVC training script locations** (PROJECT.md initial draft used outdated names):
Scripts live in `rvc/infer/modules/train/`, NOT `rvc/tools/`:
- `rvc/infer/modules/train/preprocess.py` — Stage 1 (positional argv, no flags)
- `rvc/infer/modules/train/extract/extract_f0_print.py` — Stage 2 CPU/harvest
- `rvc/infer/modules/train/extract/extract_f0_rmvpe.py` — Stage 2 GPU (recommended f0 method)
- `rvc/infer/modules/train/extract_feature_print.py` — Stage 3 (7-arg positional form)
- `rvc/infer/modules/train/train.py` — Stage 4 (named flags: `-e -sr -f0 -bs -g -te -se -pg -pd -l -c -sw -v`)

**`-e` flag is a NAME, not a path:** `train.py` resolves `./logs/<name>/` from `cwd`. Pass the bare experiment name (e.g., `"myvoice_v1"`), not a path.

**`extract_feature_print.py` reads `1_16k_wavs/`, not `0_gt_wavs/`:** Stage 3 input is the 16k copies. v2 output dir is `3_feature768/`; v1 is `3_feature256/`.

**`rvc/logs/mute/` must be populated before training:** `setup_rvc.sh` / `download_models.py` handles this. Pre-flight check required; `_write_filelist` depends on these reference rows.

**`rmvpe.pt` is required for the default `f0-method`:** `rvc/assets/rmvpe.pt`. Downloaded by `rvc/tools/download_models.py`. Doctor must verify presence.

**FAISS and sklearn live only in `rvc/.venv`:** any index-building code must be a subprocess, never an in-process call from `src/train.py`.

**Exit codes interpretation:**
- Stages 1-3 (`preprocess`, `extract_f0`, `extract_feature`): success = exit 0; verify output directory is non-empty
- Stage 4 (`train`): success = exit 0 **OR** exit 61; verify `rvc/assets/weights/<name>.pth` exists
- Stage 5 (our FAISS shim): success = exit 0; verify newest `added_*.index` exists

**Sentinel-based resume semantics:**
- Stage 1 done: `rvc/logs/<exp>/0_gt_wavs/*.wav` non-empty
- Stage 2 done: `rvc/logs/<exp>/2a_f0/*.wav.npy` non-empty AND `2b-f0nsf/*.wav.npy` non-empty
- Stage 3 done: `rvc/logs/<exp>/3_feature768/*.npy` non-empty (v2) / `3_feature256/` (v1)
- Stage 4 done: `rvc/assets/weights/<exp>.pth` exists AND (if resumable) `rvc/logs/<exp>/G_*.pth` + `D_*.pth` exist
- Stage 5 done: `rvc/logs/<exp>/added_*.index` exists

---

## Phase Ordering Implications (5 phases, strictly linear)

### Phase 1: Pod Bootstrap
**Goal:** `scripts/setup_pod.sh` takes a bare Ubuntu + NVIDIA-driver pod to fully provisioned, idempotently.
**New files:** `scripts/setup_pod.sh`
**Modified:** `src/doctor.py` (add `check_disk_space_floor`, `check_gpu_vram_floor`, `--training` flag), `tests/unit/test_doctor.py`
**Must avoid:** MOD-2 (`pip<24.1`), MOD-4 (`LD_LIBRARY_PATH`), MOD-5 (`DEBIAN_FRONTEND=noninteractive`), MOD-1 (partial venv state on reused pod)
**Dependency:** none (first phase)

### Phase 2: Training CLI
**Goal:** `python src/train.py --experiment-name ... --dataset-dir ...` runs stages 1-4 of the pipeline end-to-end, resumable, with pre-flight doctor checks.
**New files:** `src/train.py`, `tests/unit/test_train.py`
**Modified:** `src/doctor.py` (add `check_pretrained_v2_weights`, `check_training_dataset_nonempty`, `check_rvc_mute_refs`, `check_hubert_base`)
**Must avoid:** CRIT-1 (random init), CRIT-2 (sr mismatch), CRIT-3 (empty filelist), CRIT-4 (missing hubert silent exit), CRIT-5 (exit 61), MOD-6 (cross-venv import), MOD-7 (wrong cwd)
**Dependency:** Phase 1 (bootstrap must work to test this on a pod)

### Phase 3: Index Training (FAISS shim)
**Goal:** Stage 5 of the pipeline. Adds the vendored FAISS CLI shim and auto-exports `.pth` + newest `added_*.index` to `models/<name>/`.
**New files:** `scripts/rvc_patches/train_index_cli.py`
**Modified:** `src/train.py` (add `build_rvc_train_index_cmd` and `_auto_export`), `scripts/setup_pod.sh` (copy shim into `rvc/tools/` at bootstrap time — creates cross-phase dependency that must be handled in Phase 3, not Phase 1)
**Must avoid:** MIN-1 (wrong index by alphabetic sort), degenerate FAISS on tiny dataset
**Dependency:** Phase 2

### Phase 4: Shell Orchestration + Remote Source
**Goal:** `./scripts/train.sh <experiment-name> <dataset-source>` one-liner; `--dataset-url` remote pull support in `src/train.py`.
**New files:** `scripts/train.sh`
**Modified:** `src/train.py` (add `_pull_remote_source` via `curl -fL` subprocess)
**Dependency:** Phase 3 (export path must be final before user-facing one-liner is written)

### Phase 5: Pod Shutdown Documentation
**Goal:** README section with RunPod / Vast / Lambda / systemd auto-stop patterns. No code.
**Modified:** `README.md`
**Dependency:** Phase 4 (depends on finalized sentinel/exit convention)

---

## Gaps and Open Questions

- **Ubuntu 24.04 CUDA 12.1 runfile install:** `--toolkit --silent --override` flags are community-sourced. Mitigate by recommending Ubuntu 22.04 base; document 24.04 as unsupported until tested.
- **`MASTER_PORT` TCP binding on pod firewalls:** `train.py` binds a random localhost port even for single-GPU. Add a 120-second hang timeout with diagnostic message in Phase 2.
- **fairseq cache warm-up:** First feature extraction may attempt network calls. Set `TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1` in subprocess environment in Phase 2 as a precaution.
- **`torch.distributed` localhost port binding:** Needs validation on real RunPod / Vast.ai pods; add diagnostic timeout rather than blocking the phase.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack (RVC upstream) | HIGH | Scripts verified from vendored clone at pinned commit |
| Stack (CUDA apt 22.04) | HIGH | NVIDIA keyring repo is current; `cuda-toolkit-12-1` meta-package confirmed |
| Stack (CUDA runfile 24.04) | MEDIUM | Community-sourced flags; treat as best-effort |
| Features | HIGH | All features sourced from `infer-web.py` + `utils.py` + PROJECT.md |
| Architecture | HIGH | All paths, signatures, and conventions derived from existing code |
| Pitfalls | HIGH | All critical pitfalls backed by specific line numbers in vendored clone |

**Overall confidence:** HIGH

---

## Sources

**Primary (HIGH confidence — verified from vendored codebase):**
- `rvc/infer/modules/train/train.py` — exit code 61, resume behavior, checkpoint naming, pretrained weight loading
- `rvc/infer/modules/train/preprocess.py` — Stage 1 argv signature
- `rvc/infer/modules/train/extract/extract_f0_print.py` — Stage 2 argv signature
- `rvc/infer/modules/train/extract_feature_print.py` — Stage 3 argv signature (7-arg form)
- `rvc/infer/lib/train/utils.py` — `get_hparams`, `latest_checkpoint_path`, `load_checkpoint`
- `rvc/infer-web.py` lines 218-778 — full training workflow; filelist.txt generation (500-546); FAISS index build (616-700)
- `rvc/assets/pretrained_v2/` — 12 pretrained weight files confirmed present
- `src/generate.py`, `src/doctor.py`, `src/preprocess.py`, `scripts/setup_rvc.sh` — existing patterns to mirror

**Secondary (MEDIUM confidence):**
- NVIDIA apt repository — CUDA 12.1 toolkit install on Ubuntu 22.04
- CUDA 12.1 runfile installer for Ubuntu 24.04 — community-sourced flags
- mise documentation — `mise where python` / `mise exec` non-interactive usage

---
*Research completed: 2026-04-09*
*Ready for roadmap: yes*
