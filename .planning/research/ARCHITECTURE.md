# Architecture Patterns: Pod-Ready RVC Training

**Domain:** Headless GPU pod — audio training CLI extension
**Researched:** 2026-04-09
**Confidence:** HIGH (all findings derived directly from existing code and pinned RVC sources)

---

## Standard Architecture

### System Overview

```
scripts/setup_pod.sh          ← NEW: bare-pod bootstrap (installs system layer)
    └── scripts/setup_rvc.sh  ← EXISTING (unchanged): RVC venv + deps
            └── src/doctor.py --system-only / --rvc-only

scripts/train.sh              ← NEW: end-to-end shell orchestrator
    └── src/train.py          ← NEW: headless training CLI (analogous to generate.py)
            ├── src/doctor.py (pre-flight checks, including new training checks)
            └── subprocess.run([RVC_VENV_PYTHON, ...], cwd=RVC_DIR)
                    ├── infer/modules/train/preprocess.py        (RVC resample step)
                    ├── infer/modules/train/extract/extract_f0_print.py
                    ├── infer/modules/train/extract_feature_print.py
                    └── infer/modules/train/train.py             (with filelist.txt written by src/train.py)

scripts/install_model.sh      ← EXISTING: promotes rvc/assets/weights + rvc/logs → models/
```

Every new component extends into an existing layer slot. No new layers are added.

---

## Question 1: Bootstrap Script Layering

### Recommendation: Option (a) — new `scripts/setup_pod.sh` that calls `setup_rvc.sh`

Do not add a `--bootstrap` flag to `setup_rvc.sh`. The two scripts have different scopes:

| Script | Scope | Re-run semantics |
|--------|-------|------------------|
| `scripts/setup_rvc.sh` | RVC venv layer (already tested, already idempotent) | `--force` wipes rvc/ |
| `scripts/setup_pod.sh` | System layer + app venv + call `setup_rvc.sh` | Probe-and-skip each layer |

**Rationale for separation:**

1. `setup_rvc.sh` is already the idempotent, tested unit for the RVC venv step. Mixing system-level concerns into it risks breaking the working path for non-pod users.
2. A `--bootstrap` flag creates conditional branching inside an already-complex script. The self-re-exec/tee pattern makes conditional flags harder to reason about.
3. `setup_pod.sh` exec-ing into `setup_rvc.sh` at the end is the same "layers call down" pattern already in the codebase (`setup_rvc.sh` calls `doctor.py --system-only` before doing its own work).
4. `doctor.py --system-only` is still the canonical pre-flight oracle — `setup_pod.sh` calls it after the system layer is provisioned, then delegates RVC layer to `setup_rvc.sh` which also calls it.

**New file:** `scripts/setup_pod.sh`

```
scripts/setup_pod.sh [--force]
```

Step sequence (each step probes before acting — idempotent):

```
1. Detect CUDA toolkit (nvcc --version or /usr/local/cuda/bin/nvcc)
   → missing: apt-get install cuda-toolkit-12-1 from NVIDIA repo
   → present: skip

2. Detect mise (mise --version)
   → missing: curl https://mise.run | sh && mise activate (to ~/.profile)
   → present: skip

3. Detect Python 3.10 in mise (mise list python | grep "3.10")
   → missing: mise install python@3.10 && mise use --global python@3.10
   → present: skip

4. Detect app venv (.venv/bin/python)
   → missing: mise exec python@3.10 -- python -m venv .venv && .venv/bin/pip install -e ".[dev]"
   → present: skip (re-install if --force)

5. Run doctor.py --system-only (validates ffmpeg, git, nvidia-smi, mise, python 3.10)
   → any failure: print fix_hint and exit 1

6. exec scripts/setup_rvc.sh "$@"   (passes --force through if given)
```

Key script conventions:
- `set -euo pipefail` at top
- Same self-re-exec/tee pattern as `setup_rvc.sh` (log to `scripts/setup_pod.log`)
- `PROJECT_ROOT` computed from `${BASH_SOURCE[0]}`
- `ROOT_PYTHON="$PROJECT_ROOT/.venv/bin/python"` only used after step 4
- No interactive prompts — all NVIDIA apt steps use `-y`
- `exec` at step 6 preserves exit code and log continuity

**Re-run on a re-used pod (e.g., pod restarted but not wiped):**
Every layer probes its sentinel before acting. On a fully provisioned pod that was restarted:
- Steps 1-4 all skip (sentinels present)
- Step 5 runs doctor (instant, confirms health)
- Step 6 delegates to `setup_rvc.sh`, which also probes (rvc/.git present, rvc/.venv present) and skips the expensive parts

Total re-run time on a healthy pod: ~10 seconds.

**Failure modes:**

| Failure point | Behaviour |
|---------------|-----------|
| CUDA toolkit apt-get fails | `set -e` exits; `setup_pod.log` shows apt error; user retries or uses a different base image |
| mise install fails | exits; fix_hint guides to `mise install python@3.10` |
| app venv creation fails | exits; disk full is the most likely cause — doctor will report disk space if we add that check |
| `setup_rvc.sh` fails | exit code propagates; `setup_rvc.log` already exists with detail |

---

## Question 2: Headless Training Entry Point

### New file: `src/train.py`

Mirrors `src/generate.py` in every structural respect.

**Layering inside `src/train.py`:**

```
CLI (typer @app.command main)
  ├── Pre-flight: check_rvc_cloned, check_rvc_venv, check_rvc_torch_cuda,
  │               check_pretrained_v2_weights, check_gpu_vram_floor,
  │               check_disk_space_floor, check_training_dataset_nonempty
  ├── Optional: _pull_remote_source(url, dest)   ← download if --source-url given
  ├── _write_filelist(exp_dir, version, sample_rate, if_f0) ← pure, generates filelist.txt
  ├── _run_rvc_preprocess(...)   ← calls build_rvc_preprocess_cmd + subprocess.run
  ├── _run_rvc_extract_f0(...)   ← calls build_rvc_extract_f0_cmd + subprocess.run
  ├── _run_rvc_extract_feature(...)  ← calls build_rvc_extract_feature_cmd + subprocess.run
  ├── _run_rvc_train(...)        ← calls build_rvc_train_cmd + subprocess.run
  └── _export_model(exp_name)    ← calls existing scripts/install_model.sh pattern (copies files)
```

### Pure arg-builder functions (one per RVC training sub-process)

All live in `src/train.py` alongside the CLI, mirroring how `build_rvc_subprocess_cmd` lives in `generate.py`.

**`build_rvc_preprocess_cmd`**
```python
def build_rvc_preprocess_cmd(
    *,
    rvc_python: Path,
    trainset_dir: Path,
    exp_dir: Path,          # absolute path to rvc/logs/<experiment>/
    sample_rate: int,       # 32000, 40000, or 48000
    n_processes: int = 1,   # RVC's n_p arg
    no_parallel: bool = True,
    per: float = 3.7,       # RVC default
) -> list[str]:
    # Invokes: infer/modules/train/preprocess.py <trainset_dir> <sr> <n_p> <exp_dir> <noparallel> <per>
```

**`build_rvc_extract_f0_cmd`**
```python
def build_rvc_extract_f0_cmd(
    *,
    rvc_python: Path,
    exp_dir: Path,          # absolute path to rvc/logs/<experiment>/
    n_processes: int = 1,
    f0_method: str = "rmvpe",  # "rmvpe" is the right default for single-GPU
) -> list[str]:
    # Invokes: infer/modules/train/extract/extract_f0_print.py <exp_dir> <n_p> <f0method>
    # OR for rmvpe on GPU: extract_f0_rmvpe.py (handled via method dispatch in _run_rvc_extract_f0)
```

**`build_rvc_extract_feature_cmd`**
```python
def build_rvc_extract_feature_cmd(
    *,
    rvc_python: Path,
    exp_dir: Path,
    gpu_id: str = "0",
    version: str = "v2",
    is_half: bool = True,
) -> list[str]:
    # Invokes: infer/modules/train/extract_feature_print.py
    #   <device> <n_part> <i_part> <i_gpu> <exp_dir> <version> <is_half>
    # For single GPU: n_part=1, i_part=0, i_gpu=0
```

**`build_rvc_train_cmd`**
```python
def build_rvc_train_cmd(
    *,
    rvc_python: Path,
    exp_name: str,
    sample_rate: str,        # "40k" | "48k" | "32k"
    if_f0: int = 1,          # 1=use pitch, matches f0 pretrained weights
    batch_size: int = 4,
    gpu_ids: str = "0",
    total_epochs: int = 100,
    save_every_epochs: int = 10,
    pretrained_g: str = "",  # e.g. "assets/pretrained_v2/f0G40k.pth"
    pretrained_d: str = "",  # e.g. "assets/pretrained_v2/f0D40k.pth"
    save_latest_only: int = 1,
    cache_gpu: int = 0,
    save_every_weights: int = 0,
    version: str = "v2",
) -> list[str]:
    # Invokes: infer/modules/train/train.py
    #   -e <exp_name> -sr <sr> -f0 <if_f0> -bs <bs> -g <gpus>
    #   -te <total_epoch> -se <save_every> -pg <pg> -pd <pd>
    #   -l <save_latest> -c <cache> -sw <save_every_weights> -v <version>
```

**`_write_filelist`** (not a subprocess cmd builder — pure file I/O, but side-effect-free on input data):
```python
def _write_filelist(
    *,
    exp_dir: Path,
    version: str,
    sample_rate: str,
    if_f0: bool,
    speaker_id: int = 0,
) -> Path:
    # Replicates the filelist.txt construction from infer-web.py:click_train
    # Reads 0_gt_wavs/, 3_feature768/ (or 3_feature256/), 2a_f0/, 2b-f0nsf/
    # Intersects stems, shuffles, writes exp_dir/filelist.txt
    # Returns the path of the written file
```

Note: `_write_filelist` must also append mute reference rows (the `logs/mute/` entries that RVC requires to exist). `setup_rvc.sh` or `scripts/train.sh` must ensure `rvc/logs/mute/` is present — it is downloaded by `tools/download_models.py`.

### Relationship between `src/preprocess.py` and the RVC training preprocess step

These are two **entirely different** pipelines that run in sequence, not as alternatives:

| Step | Script | Venv | Input | Output | Purpose |
|------|--------|------|-------|--------|---------|
| 1. ffmpeg pipeline | `src/preprocess.py` | `.venv` | `dataset/raw/` | `dataset/processed/` | Canonicalize, denoise, normalize, slice to 3-15s clips using slicer2 |
| 2. RVC resample | `rvc/infer/modules/train/preprocess.py` | `rvc/.venv` | `dataset/processed/` | `rvc/logs/<exp>/0_gt_wavs/` | Resample to target SR (32/40/48k), apply RVC's internal slicer and normalization |

The user runs `src/preprocess.py` first (existing CLI, unchanged). Then `src/train.py` feeds `dataset/processed/` as the `trainset_dir` to the RVC preprocess step. These steps chain; neither replaces the other. The existing `src/preprocess.py` output is the *input* to the RVC preprocess step.

### One orchestrator vs four separate CLI commands

**Recommendation: one orchestrator `run_train` function called by a single `python src/train.py` invocation.**

Rationale:
- The existing pattern (`run_preprocess` in `preprocess.py`) is a single orchestrator function called by a single CLI. Follow it.
- Training sub-steps have a strict linear dependency (preprocess → extract_f0 → extract_feature → train). Exposing them as four separate commands invites accidental wrong-order invocation.
- Resumability is simpler with one entry point: the orchestrator checks which sentinel files are present and skips already-completed sub-steps without requiring the user to remember what ran last.
- The single-command model matches the "no interactive prompts" operational constraint exactly.

The four builder functions exist as separate pure functions (unit-testable in isolation) but are wired together by `run_train` in the same file.

---

## Question 3: Doctor Checks for the Training Path

### Table stakes (block training if missing)

| Check | Function name | What it verifies | Why blocking |
|-------|--------------|------------------|--------------|
| Pretrained base weights | `check_pretrained_v2_weights` | `rvc/assets/pretrained_v2/f0G{sr}.pth` and `f0D{sr}.pth` exist for the target sample rate | Training fails immediately if these are absent; the error from RVC is cryptic |
| Training dataset non-empty | `check_training_dataset_nonempty` | given `trainset_dir` exists and contains at least one `.wav` | Fails at RVC preprocess step with a confusing error |
| Disk space floor | `check_disk_space_floor` | free space >= configurable floor (default 20 GB) | Training writes ~1-2 GB of intermediate features + checkpoints; running out mid-training loses all progress |
| GPU VRAM floor | `check_gpu_vram_floor` | VRAM >= configurable floor (default 6 GB) | Below 6 GB, even batch_size=1 at 40k OOMs; fail early with a clear message |

Implementation notes:

```python
def check_pretrained_v2_weights(sample_rate: str = "40k") -> CheckResult:
    # sample_rate: "32k" | "40k" | "48k"
    # Checks rvc/assets/pretrained_v2/f0G{sr}.pth and f0D{sr}.pth
    # fix_hint: "Run ./scripts/setup_rvc.sh (downloads pretrained_v2 weights)"

def check_training_dataset_nonempty(trainset_dir: Path) -> CheckResult:
    # Checks trainset_dir exists and glob("*.wav") is non-empty
    # fix_hint: "Run python src/preprocess.py first, then check dataset/processed/"

def check_disk_space_floor(required_gb: float = 20.0) -> CheckResult:
    # Uses shutil.disk_usage(PROJECT_ROOT) — free / (1024**3) >= required_gb
    # fix_hint: "Free at least {required_gb}GB on the pod's volume"

def check_gpu_vram_floor(required_gb: float = 6.0) -> CheckResult:
    # Runs: rvc/.venv/bin/python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory)"
    # fix_hint: "Need ≥{required_gb}GB VRAM. Choose a GPU with more memory."
```

Register in `doctor.py`:
- `check_pretrained_v2_weights`, `check_disk_space_floor`, `check_gpu_vram_floor` → add to `rvc_checks` list
- `check_training_dataset_nonempty` → NOT added to the default `rvc_checks` list (it takes a path argument); called explicitly from `src/train.py` pre-flight with the user-supplied `trainset_dir`

Add a `--training` flag to `doctor.py main` that runs `rvc_checks + [check_pretrained_v2_weights, check_disk_space_floor, check_gpu_vram_floor]`.

### Nice-to-have (warn but do not block)

- Sample rate mismatch: if clips in `trainset_dir` are not already at target SR, warn (they will be resampled by RVC's preprocess step, so this is noise not a blocker)
- CUDA compute capability: warn if GPU is below compute 7.0 (Volta) since torch 2.1.2 performance degrades, but this is not a hard blocker

---

## Question 4: Data Flow for a Training Run

### Inputs

```
python src/train.py \
  --experiment  <name>          # e.g. "myvoice_v1"
  --trainset    dataset/processed/    # OR --source-url <url>
  --sample-rate 40k             # 32k | 40k | 48k
  --epochs      100
  --batch-size  4
  --save-every  10
  --version     v2
  --device      cuda:0
```

### End-to-End Data Flow

```
INPUTS
  dataset/processed/*.wav           (from src/preprocess.py — user's cleaned clips)
  OR remote URL → curl → dataset/processed/   (if --source-url given)

STAGE 0: Pre-flight
  src/doctor.py checks (7 checks)
  exit 1 on any failure

STAGE 1: RVC preprocess (src/train.py → build_rvc_preprocess_cmd → subprocess)
  INPUT:  dataset/processed/*.wav
  OUTPUT: rvc/logs/<experiment>/0_gt_wavs/*.wav   (resampled to target SR)
          rvc/logs/<experiment>/preprocess.log
  SENTINEL: rvc/logs/<experiment>/0_gt_wavs/ non-empty
  RESUME: if sentinel present and not --force-preprocess, skip

STAGE 2: F0 extraction (src/train.py → build_rvc_extract_f0_cmd → subprocess)
  INPUT:  rvc/logs/<experiment>/0_gt_wavs/*.wav
  OUTPUT: rvc/logs/<experiment>/2a_f0/*.wav.npy
          rvc/logs/<experiment>/2b-f0nsf/*.wav.npy
          rvc/logs/<experiment>/extract_f0_feature.log
  SENTINEL: rvc/logs/<experiment>/2a_f0/ non-empty
  RESUME: if sentinel present, skip

STAGE 3: Feature extraction (src/train.py → build_rvc_extract_feature_cmd → subprocess)
  INPUT:  rvc/logs/<experiment>/0_gt_wavs/*.wav  +  rvc/assets/hubert/hubert_base.pt
  OUTPUT: rvc/logs/<experiment>/3_feature768/*.npy   (for v2)
          rvc/logs/<experiment>/extract_f0_feature.log  (appended)
  SENTINEL: rvc/logs/<experiment>/3_feature768/ non-empty
  RESUME: if sentinel present, skip

STAGE 4: Filelist generation (src/train.py → _write_filelist → pure Python)
  INPUT:  rvc/logs/<experiment>/0_gt_wavs/, 3_feature768/, 2a_f0/, 2b-f0nsf/
  OUTPUT: rvc/logs/<experiment>/filelist.txt   (always regenerated, fast)
          rvc/logs/<experiment>/config.json    (written from configs/v2/{sr}.json)

STAGE 5: Model training (src/train.py → build_rvc_train_cmd → subprocess)
  INPUT:  rvc/logs/<experiment>/filelist.txt
          rvc/logs/<experiment>/config.json
          rvc/assets/pretrained_v2/f0G40k.pth (or matching SR)
          rvc/assets/pretrained_v2/f0D40k.pth
  OUTPUT: rvc/logs/<experiment>/G_<step>.pth  (generator checkpoint every N epochs)
          rvc/logs/<experiment>/D_<step>.pth  (discriminator checkpoint)
          rvc/logs/<experiment>/train.log
          rvc/assets/weights/<experiment>.pth (latest extracted weight, when --save-every-weights=1)
  RESUME: RVC's train.py detects G_<step>.pth / D_<step>.pth and resumes from highest step
  ERROR PATHS:
    - CUDA OOM: prints "CUDA out of memory" in train.log; next invocation retries from last checkpoint
    - SIGTERM / pod kill: training stops; G_<step>.pth / D_<step>.pth survive as resume point
    - Epoch 47 failure: last clean checkpoint is G_<last_saved_step>.pth; resume skips stages 1-4 (sentinels present), re-runs train.py which picks up from G_<last_saved_step>.pth

STAGE 6: Index training (in-process Python in src/train.py, NOT a subprocess)
  The train_index logic from infer-web.py uses faiss and numpy — both available in rvc/.venv.
  This MUST be invoked as a subprocess:
    build_rvc_train_index_cmd → subprocess.run([RVC_VENV_PYTHON, "tools/train_index.py", ...], cwd=RVC_DIR)
  OR: expose train_index as a standalone script in rvc/tools/ (it does not exist there currently;
  we write a thin wrapper script at rvc/tools/train_index_cli.py that the builder invokes).
  
  INPUT:  rvc/logs/<experiment>/3_feature768/*.npy
  OUTPUT: rvc/logs/<experiment>/added_IVF*_Flat_nprobe_*_<experiment>_v2.index
          rvc/logs/<experiment>/trained_IVF*_Flat_nprobe_*_<experiment>_v2.index
  SENTINEL: rvc/logs/<experiment>/added_*.index exists
  NOTE: faiss and sklearn live in rvc/.venv; this step MUST be a subprocess to respect the two-venv boundary.

STAGE 7: Export (src/train.py → _export_model, pure Python in app venv)
  INPUT:  rvc/assets/weights/<experiment>.pth   (if --save-every-weights=1 used during training)
          OR: rvc/logs/<experiment>/G_<highest_step>.pth (if not)
  OUTPUT: models/<experiment>/<experiment>.pth
          models/<experiment>/<experiment>.index
  Implementation: replicate scripts/install_model.sh logic in Python (shutil.copy2)
  SENTINEL: models/<experiment>/<experiment>.pth exists and size > 1 MB
```

### Resume Semantics (exact state required)

A re-invocation resumes rather than restarts when:

| Stage | Required state for resume | How checked |
|-------|---------------------------|-------------|
| Skip stage 1 (RVC preprocess) | `rvc/logs/<exp>/0_gt_wavs/` is non-empty | `any(exp_dir.joinpath("0_gt_wavs").glob("*.wav"))` |
| Skip stage 2 (F0 extract) | `rvc/logs/<exp>/2a_f0/` is non-empty | `any(exp_dir.joinpath("2a_f0").glob("*.npy"))` |
| Skip stage 3 (feature extract) | `rvc/logs/<exp>/3_feature768/` is non-empty | `any(exp_dir.joinpath("3_feature768").glob("*.npy"))` |
| Resume training from checkpoint | `rvc/logs/<exp>/G_*.pth` exists (RVC handles automatically) | RVC train.py scans for highest step automatically |
| Skip index training | `rvc/logs/<exp>/added_*.index` exists | `any(exp_dir.glob("added_*.index"))` |
| Skip export | `models/<exp>/<exp>.pth` exists and size > 1 MB | standard Path.stat() check |

All resume decisions are made before the corresponding subprocess is launched. This means a `--force` flag (per-stage or global) must clear the relevant sentinel to force a re-run.

### Intermediate Directory Layout

```
rvc/
└── logs/
    └── <experiment>/
        ├── 0_gt_wavs/          ← stage 1 output: resampled training wavs
        ├── 2a_f0/              ← stage 2 output: F0 curves (.wav.npy)
        ├── 2b-f0nsf/           ← stage 2 output: F0 NSF frames (.wav.npy)
        ├── 3_feature768/       ← stage 3 output: HuBERT features (v2; v1 is 3_feature256)
        ├── config.json         ← stage 4 output: training config
        ├── filelist.txt        ← stage 4 output: shuffled training manifest
        ├── preprocess.log
        ├── extract_f0_feature.log
        ├── train.log
        ├── G_<step>.pth        ← stage 5 output: generator checkpoint(s)
        ├── D_<step>.pth        ← stage 5 output: discriminator checkpoint(s)
        └── added_*.index       ← stage 6 output: FAISS feature index
rvc/assets/weights/
    └── <experiment>.pth        ← stage 5 output (if --save-every-weights=1)
models/
    └── <experiment>/
        ├── <experiment>.pth    ← stage 7 output: final promoted weight
        └── <experiment>.index  ← stage 7 output: final promoted index
```

---

## Question 5: Build Order (3–5 Phases)

### Phase Breakdown

**Phase 1: Bootstrap — Pod provisioning**

Goal: a single script takes a bare Ubuntu+NVIDIA-driver pod to fully provisioned, idempotently.

New files:
- `scripts/setup_pod.sh`
- `scripts/setup_pod.log` (generated)

Depends on: nothing new (builds on existing `setup_rvc.sh`)

Doctor additions:
- `check_disk_space_floor` (add to rvc_checks)
- `check_gpu_vram_floor` (add to rvc_checks)
- New `--training` flag on `doctor.py main`

Tests:
- `tests/unit/test_doctor.py`: add cases for the two new check functions (mock `shutil.disk_usage`, mock the `rvc/.venv/bin/python` torch call)

**Phase 2: Training CLI — Headless subprocess wrappers**

Goal: `python src/train.py` orchestrates RVC's four training scripts end-to-end with all hyperparameter flags, sentinel-based resume, and auto-export.

New files:
- `src/train.py` (CLI + four builder functions + `_write_filelist` + `_export_model`)
- `tests/unit/test_train.py` (unit tests for all builder functions as pure functions)

Depends on: Phase 1 (doctor checks must exist for pre-flight)

Key additions to `src/doctor.py`:
- `check_pretrained_v2_weights(sample_rate)` — add to rvc_checks
- `check_training_dataset_nonempty(trainset_dir)` — called explicitly in `src/train.py`, not added to doctor's named groups (takes a Path argument)

**Phase 3: Index training — FAISS step**

Goal: complete the training pipeline by adding the index build step as a subprocess.

New files:
- `rvc/tools/train_index_cli.py` — thin script that runs the faiss index-build logic from `infer-web.py:train_index`, accepts `<exp_dir> <version>` positional args (runs inside `rvc/.venv`, so faiss/sklearn/numpy are available)
- `build_rvc_train_index_cmd` added to `src/train.py`

Depends on: Phase 2 (stages 1-5 must work before index build is wired in)

Note: `rvc/tools/train_index_cli.py` is the only new file that lives inside `rvc/`. Because `rvc/` is git-ignored and regenerated by `setup_rvc.sh`, the file must either be (a) committed under a different path and copied in by `setup_pod.sh`/`train.sh`, or (b) written by `src/train.py` at runtime before the subprocess call. Option (a) is cleaner: commit `scripts/rvc_patches/train_index_cli.py` and have `scripts/setup_pod.sh` copy it to `rvc/tools/` after the RVC clone step.

**Phase 4: Shell orchestration + remote source**

Goal: `./scripts/train.sh` is the single user-facing command for a full pod run; `--source-url` support for pulling audio from a URL before training.

New files:
- `scripts/train.sh` — wraps `python src/train.py` with sensible defaults, logging, and a completion sentinel (`models/<exp>/<exp>.pth` exists) suitable for provider shutdown hooks
- `_pull_remote_source(url, dest)` added to `src/train.py` (uses `subprocess.run(["curl", ...])` or `["wget", ...]`, no Python HTTP)

Depends on: Phases 1-3

This phase is intentionally thin — `scripts/train.sh` is mostly a convenience wrapper and documentation target. The real logic is in `src/train.py`.

**Phase 5: Documentation — Pod shutdown guide**

Goal: README section documenting how to wire provider auto-stop hooks to the training script's exit code / sentinel file.

No new code. Depends on Phase 4 (sentinel file convention must be finalized).

### Phase Dependency Graph

```
Phase 1 (bootstrap) → Phase 2 (train CLI) → Phase 3 (index) → Phase 4 (shell + remote) → Phase 5 (docs)
```

All phases are strictly linear. No phases can be built in parallel because each gate-keeps the next.

---

## Component Boundaries

| Component | File(s) | Layer | Communicates With |
|-----------|---------|-------|-------------------|
| Pod bootstrap | `scripts/setup_pod.sh` | Shell-orchestration | System (apt), mise, `.venv`, `scripts/setup_rvc.sh`, `src/doctor.py` |
| Training CLI | `src/train.py` | Entry + pipeline-function | `src/doctor.py` (pre-flight), `subprocess.run(cwd=RVC_DIR)` via four builders |
| Training preprocess builder | `build_rvc_preprocess_cmd` in `src/train.py` | Pipeline-function | Pure, returns `list[str]` |
| F0 extraction builder | `build_rvc_extract_f0_cmd` in `src/train.py` | Pipeline-function | Pure, returns `list[str]` |
| Feature extraction builder | `build_rvc_extract_feature_cmd` in `src/train.py` | Pipeline-function | Pure, returns `list[str]` |
| Training builder | `build_rvc_train_cmd` in `src/train.py` | Pipeline-function | Pure, returns `list[str]` |
| Index training builder | `build_rvc_train_index_cmd` in `src/train.py` | Pipeline-function | Pure, returns `list[str]` |
| Filelist writer | `_write_filelist` in `src/train.py` | Pipeline-function | Reads `rvc/logs/<exp>/` directories, writes `filelist.txt` |
| Index CLI shim | `scripts/rvc_patches/train_index_cli.py` | Vendored shim (in `rvc/.venv` subprocess context) | `faiss`, `numpy`, `sklearn` — all inside `rvc/.venv` |
| New doctor checks | `check_pretrained_v2_weights`, `check_gpu_vram_floor`, `check_disk_space_floor`, `check_training_dataset_nonempty` in `src/doctor.py` | Check layer | `subprocess.run` (for VRAM), `shutil.disk_usage`, `Path.glob` |
| Shell orchestrator | `scripts/train.sh` | Shell-orchestration | `src/train.py`, `models/` |

### Two-Venv Boundary Preservation

All faiss, torch, fairseq, sklearn, librosa, numpy (RVC's copy) operations happen inside `rvc/.venv` via subprocess. The app venv (`src/train.py`) only:
- Builds argv lists (pure Python, no GPU imports)
- Calls `subprocess.run([RVC_VENV_PYTHON, ...], cwd=RVC_DIR, ...)`
- Reads/writes file paths (sentinels, filelist.txt, export copies)
- Calls `shutil.copy2` for export (stdlib only)

No `import faiss`, `import torch`, `import fairseq` anywhere in `src/`.

---

## Concrete New File Paths

```
scripts/setup_pod.sh                       ← new (Phase 1)
scripts/setup_pod.log                      ← generated (Phase 1)
scripts/rvc_patches/train_index_cli.py     ← new (Phase 3, copied to rvc/tools/ at bootstrap)
scripts/train.sh                           ← new (Phase 4)
src/train.py                               ← new (Phase 2)
tests/unit/test_train.py                   ← new (Phase 2)
```

Modifications to existing files:
```
src/doctor.py   ← add 4 check functions + --training flag to main
tests/unit/test_doctor.py  ← add test cases for new checks
```

No other existing files are modified.

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: shell=True in subprocess builders
The existing `infer-web.py` uses `Popen(cmd, shell=True)` throughout. Do NOT copy this pattern. All builders in `src/train.py` return `list[str]` and are invoked with `subprocess.run(cmd, cwd=RVC_DIR, shell=False)`. This is already enforced by convention in the codebase and directly mirrors `build_rvc_subprocess_cmd`.

### Anti-Pattern 2: Importing from rvc/ in app venv
`faiss`, `sklearn`, `torch`, `librosa` are not installed in `.venv`. Any attempt to import them in `src/train.py` will fail. The index training step (which needs faiss) must be a subprocess into `rvc/.venv`, not an in-process call.

### Anti-Pattern 3: One monolithic builder for all four training steps
The webui uses a single `click_train` function that conflates filelist generation, config writing, and subprocess launch. Split these. Pure builders make the steps unit-testable without a GPU.

### Anti-Pattern 4: Hardcoding `rvc/logs/` relative paths
RVC's internal scripts use `os.getcwd()` and construct paths relative to it. Our builders must pass absolute paths where the scripts accept them, and always invoke with `cwd=RVC_DIR` so `os.getcwd()` resolves to `rvc/` as RVC expects.

### Anti-Pattern 5: Interactive prompts in training or bootstrap
Both `setup_pod.sh` and `src/train.py` must never pause for user input. Every decision point must have a flag (`--force`, `--sample-rate`, etc.) or a sane default that is correct for a pod context.

---

## Sources

- Existing codebase read directly: `src/doctor.py`, `src/generate.py`, `src/preprocess.py`, `scripts/setup_rvc.sh`
- RVC training orchestration read directly: `rvc/infer-web.py` (functions `preprocess_dataset`, `extract_f0_feature`, `click_train`, `train_index`, `train1key`)
- RVC script signatures read directly: `rvc/infer/modules/train/preprocess.py`, `rvc/infer/modules/train/extract/extract_f0_print.py`, `rvc/infer/modules/train/extract_feature_print.py`, `rvc/infer/lib/train/utils.py:get_hparams`
- RVC assets layout confirmed by directory listing: `rvc/assets/pretrained_v2/` (12 .pth files for 32k/40k/48k, v1/v2, G/D, f0/non-f0)
- RVC tools directory listing confirms available scripts
