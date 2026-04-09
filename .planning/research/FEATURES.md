# Feature Research

**Domain:** Headless RVC voice-model training on rented GPU pods
**Researched:** 2026-04-09
**Confidence:** HIGH (sourced directly from rvc/infer-web.py, rvc/infer/lib/train/utils.py,
rvc/infer/modules/train/preprocess.py, and .planning/PROJECT.md)

---

## RVC Webui Training Workflow (Actual Order, Verified from infer-web.py)

The webui training tab exposes these steps in strict order. Any headless CLI must replicate
the same sequence or training silently produces nothing.

**Step 1 — Configure experiment** (global settings shared across all steps)
- Experiment name (`exp_dir1`) — becomes `rvc/logs/<name>/`
- Target sample rate (`sr2`): `40k` | `48k` | `32k` (v2 only)
- RVC version (`version19`): `v1` | `v2` (v2 is default, uses 768-dim features)
- Use pitch guidance (`if_f0_3`): True for speech/singing, False speeds up extraction
- CPU process count (`np7`): ceil(n_cpu / 1.5)

**Step 2a — Preprocess dataset** (button: "处理数据")
- Invokes: `infer/modules/train/preprocess.py <trainset_dir> <sr_int> <n_p> <exp_dir> <noparallel> <per>`
- Reads raw audio from `trainset_dir`, writes sliced WAVs to `rvc/logs/<name>/0_gt_wavs/`
- Writes progress to `rvc/logs/<name>/preprocess.log`

**Step 2b — Extract F0 + features** (button: "特征提取")
- F0 extraction: `infer/modules/train/extract/extract_f0_print.py <exp_dir> <n_p> <f0method>`
  - f0method choices: `pm`, `harvest`, `dio`, `rmvpe`, `rmvpe_gpu` (default: `rmvpe_gpu`)
  - Only runs if `if_f0 = True`
  - Writes to `rvc/logs/<name>/2a_f0/` and `rvc/logs/<name>/2b-f0nsf/`
- Feature extraction: `infer/modules/train/extract_feature_print.py <device> <n_part> <i_part> <i_gpu> <exp_dir> <version> <is_half>`
  - Writes to `rvc/logs/<name>/3_feature256/` (v1) or `rvc/logs/<name>/3_feature768/` (v2)
  - Writes to `rvc/logs/<name>/extract_f0_feature.log`

**Step 3a — Build filelist + train model** (button: "训练模型")
- Builds `rvc/logs/<name>/filelist.txt` by intersecting gt_wavs / feature / f0 dirs
- Copies `configs/v2/<sr>.json` into `rvc/logs/<name>/config.json` if not present (resume-safe)
- Invokes: `infer/modules/train/train.py` with flags:
  `-e <exp_dir> -sr <sr> -f0 <0|1> -bs <batch_size> -g <gpus> -te <total_epoch> -se <save_epoch>`
  `-pg <pretrained_G> -pd <pretrained_D> -l <if_latest> -c <if_cache_gpu> -sw <save_every_weights> -v <version>`
- Required pretrained base models: `assets/pretrained_v2/f0G<sr>.pth` + `f0D<sr>.pth`
  (or non-f0 variants: `G<sr>.pth` + `D<sr>.pth`)
- Checkpoints saved to `rvc/logs/<name>/` as `G_<step>.pth` / `D_<step>.pth`
- Small "weights" checkpoints saved to `rvc/assets/weights/<name>_e<N>_s<step>.pth` when `-sw 1`

**Step 3b — Train FAISS index** (button: "训练特征索引")
- Loads all `.npy` files from `3_feature768/`, optionally kmeans-reduces to 10k centers
- Writes `rvc/logs/<name>/added_IVF<n>_Flat_nprobe_1_<name>_v2.index`
- This is in-process Python (no subprocess), using faiss + sklearn

**"One-click train"** (button: "一键训练") runs 2a → 2b → 3a → 3b in sequence.

---

## Feature Landscape

### Table Stakes (Pod-Training Flow Breaks Without These)

| Feature | Why a Pod User Cares | Complexity | Notes |
|---------|----------------------|------------|-------|
| Single entrypoint bash script: zero to provisioned | Pod bills from first second; manual multi-step setup on a bare image is error-prone and wastes money | MEDIUM | Builds on existing `setup_rvc.sh`; adds system-layer (CUDA toolkit, Python 3.10 via mise) |
| Detect-and-adapt installer (probe each layer before installing) | Same script must work on bare CUDA image, PyTorch base image, and minimal Ubuntu; guessing wrong crashes the install half-way through a billed session | MEDIUM | Probe: nvidia-smi → nvcc/CUDA toolkit → python3.10 → `.venv` → `rvc/.venv` → torch CUDA → RVC weights |
| Headless training entrypoint `python src/train.py` | Without this, training requires browser + port forwarding + 5 manual button clicks — impossible to automate | MEDIUM | Wraps the 4-step sequence: preprocess → extract_f0 → extract_feature → train_model → train_index |
| CLI accepts `--experiment-name`, `--dataset-dir`, `--sample-rate`, `--epochs`, `--batch-size` | Every training run needs these five inputs; hardcoding any one means you can't reuse the script across experiments without editing it | SMALL | Directly maps to RVC's train.py `-e -sr -te -bs`; defaults: sr=40k, epochs=200, batch=auto |
| CLI accepts `--rvc-version` (v1/v2) and `--f0-method` | v2 is the current default; f0method `rmvpe` vs `harvest` meaningfully changes quality and speed; wrong choice silently produces a worse model | SMALL | Default: version=v2, f0method=rmvpe_gpu (falls back to rmvpe on CPU-only probing) |
| `--save-every` flag (checkpoint interval) | On a preemptible pod, the right save interval is the difference between losing 2 hours and losing 10 minutes of training | SMALL | Maps to `-se` in train.py; sane default: 10 epochs |
| Pretrained base models auto-verified before training starts | `rvc/tools/download_models.py` already handles download; if pretrained G/D are missing, train.py silently trains from scratch and produces a bad model in far more epochs | SMALL | Doctor check: assert `assets/pretrained_v2/f0G<sr>.pth` and `f0D<sr>.pth` exist |
| Resumable training (pick up from last checkpoint) | Pods are preemptible; losing 3 hours of a 6-hour run is the single biggest cost risk | MEDIUM | RVC's train.py already resumes if `G_<step>.pth` / `D_<step>.pth` exist in `logs/<name>/`; the wrapper must NOT wipe `logs/<name>/` on re-invocation, and must skip preprocessing/extraction if artifacts already exist |
| Auto-export on training completion | Without this, the user must manually run `scripts/install_model.sh` — easy to forget, and the pod may be shut down before they remember | SMALL | Copy latest `G_<step>.pth` → `models/<name>/<name>.pth`; copy `added_*.index` → `models/<name>/`; matches existing `install_model.sh` layout |
| Non-interactive: zero prompts | Every prompt on a billing pod is money burned waiting. Any `[y/N]` in a script is a pod-killing bug | SMALL | All inputs via CLI flags or env vars; no `read`, no `confirm` |
| Exit with a deterministic exit code | The user's auto-shutdown hook (whatever it is) needs a reliable signal that training succeeded vs. failed | SMALL | Exit 0 = success + export complete; exit 1 = setup/config error; exit 3 = training subprocess failure |
| Training subprocess run inside `rvc/.venv` via `build_rvc_train_subprocess_cmd` (pure function, unit-tested) | Preserves the two-venv discipline; torch/fairseq must not bleed into the app venv; also makes the arg builder testable without GPU | SMALL | Analogous to existing `build_rvc_subprocess_cmd` in `src/generate.py` |

### Differentiators (Genuinely Valuable for a Minute-Billed Pod User)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Pre-flight doctor checks for the training path | Surfaces problems (missing VRAM, wrong sr, no pretrained weights, insufficient disk) before a billable training run starts, not 20 minutes into one | SMALL | New checks: `check_training_pretrained_weights`, `check_disk_space(min_gb=10)`, `check_gpu_vram(min_gb=4)`, `check_training_dataset_nonempty`. Fits the existing `src/doctor.py` pattern |
| Skip-if-done guards on each pipeline stage | If the pod restarts mid-extraction, re-running the script should skip already-completed stages (preprocess, F0 extraction, feature extraction) rather than reprocessing from scratch — saves real money | MEDIUM | Check sentinel: `rvc/logs/<name>/preprocess.log` exists and non-empty → skip preprocess; `3_feature768/` has files → skip extraction. Must be overridable with `--force` |
| Remote audio source pull before training (`--dataset-url`) | Decouples dataset upload from pod provisioning; user can store audio on R2/S3/HTTPS and the script fetches it with curl/wget/rclone before training starts | SMALL | No S3 SDK; just `curl -L` or `rclone copy` via subprocess. Mutually exclusive with `--dataset-dir` |
| Experiment manifest written on export | After export, write `models/<name>/manifest.json` recording: experiment name, sample rate, rvc version, f0method, epochs trained, batch size, training date, source audio path/URL. Lets the user know exactly what produced a model weeks later | SMALL | Pure Python dict → JSON write; no new deps |
| Structured training log tail on failure | If training subprocess fails, print the last 30 lines of `rvc/logs/<name>/train.log` to stderr with context ("training failed at epoch X") rather than a silent non-zero exit | SMALL | Same `_tail` pattern as existing `src/generate.py` |
| Smart batch-size default based on detected VRAM | Rather than picking 4 (RVC's webui default) which may OOM on a 4 GB card or waste capacity on a 24 GB card, probe `torch.cuda.get_device_properties(0).total_memory` and suggest a sensible default (4 GB → bs=4, 8 GB → bs=8, 16+ GB → bs=16) | SMALL | A pure Python function; unit-testable without GPU using a mock. No user intervention needed; still overridable with `--batch-size` |
| Pod shutdown documentation in README | User explicitly stated this is the right scope boundary — docs over code. Concrete examples for RunPod, Vast, Lambda, and generic systemd oneshot. The script provides clean exit codes; the README explains how to wire them | SMALL | Markdown only; no code |

### Anti-Features (Do Not Build This Milestone)

| Anti-Feature | Why Requested | Why Wrong Here | Alternative |
|--------------|---------------|----------------|-------------|
| Web UI (infer-web.py) for training | RVC ships it; familiar to existing RVC users | Requires browser, port forwarding, manual clicks per run — fundamentally incompatible with one-shot pod automation | Headless `src/train.py` CLI replaces it entirely |
| Provider-specific SDKs (RunPod API, Vast CLI, Lambda API) | Would enable auto-shutdown, cost tracking, and pod lifecycle management from within the script | Creates lock-in to a single provider; the "generic Linux + NVIDIA driver" contract is the design rule; provider integrations explode the maintenance surface | README documents provider-specific patterns; script exits cleanly with known codes |
| Auto-shutdown implementation (`shutdown -h now` or provider API calls) | Saves money when the user forgets | Script must not be responsible for terminating a billable resource before the user has verified the exported model is downloadable. A bug here deletes GPU time that can't be recovered | README section describes how to wire provider auto-stop to the script's exit code / sentinel file |
| Smoke inference on the pod after training | Gives confidence the trained model works before pod shutdown | Out of scope (inference improvements excluded); adds RVC inference dependencies and complexity to the training script; inference is the user's responsibility after download | User runs `src/generate.py` locally after downloading the model |
| Multi-GPU / distributed training | Faster training | Single-GPU pods cover the actual use case; multi-GPU adds torch.distributed setup complexity and makes CUDA_VISIBLE_DEVICES management much harder | Single-GPU only; `--gpus` flag defaults to `"0"` |
| Dataset management beyond existing preprocess pipeline | RVC webui includes vocal separation (UVR5), resampling, and normalization | The existing `src/preprocess.py` pipeline (canonicalize → denoise → loudnorm → slice) is sufficient input for training; adding more here is scope creep | Run `src/preprocess.py` locally before uploading dataset to pod |
| Cloud storage integrations (boto3, AWS SDK, Google Cloud SDK) | S3/R2/GCS are common dataset stores | Heavy deps, auth complexity, SDK version drift; `curl`/`rclone` covers 100% of the use cases with zero new Python packages | `--dataset-url` flag invokes `curl -L` or `rclone copy` via subprocess |
| Fast-setup optimization (parallel installs, prebuilt wheels, apt caching) | Reduces setup time on a billed pod | Correctness over speed this milestone; user explicitly deprioritized it; premature optimization here has historically broken the fairseq/pip<24.1 constraint | Accept 5-10 minute setup time in exchange for a script that works on first run |
| Training multiple experiments in one invocation | Batch training sessions across experiments | Adds orchestration complexity; the pod is rented for one training run; running multiple experiments is a separate workflow | Invoke the script once per experiment |
| YAML/TOML config file for hyperparameters | Matches "modern" CLI patterns; easier to version control | Adds a config file format to maintain; conflicts with the existing pattern of CLI flags + env vars in `src/generate.py`; one fewer thing to upload to a pod | CLI flags with sane defaults; env vars for secrets |
| Tensorboard / training metrics dashboard | RVC's train.py writes tensorboard events; useful for monitoring loss curves | Requires a browser or port forwarding on the pod; adds a blocking service; on a billing pod you want to train, not watch charts | After download, user can run `tensorboard --logdir rvc/logs/<name>` locally |
| Checkpoint cleanup / pruning old checkpoints | Keeps disk usage bounded | Pod disk is ephemeral; cleanup on a pod saves nothing meaningful; local disk management after download is the user's concern | `--save-latest` flag (maps to `-l 1` in train.py) keeps only the latest G/D if desired |

---

## Feature Dependencies

```
[Detect-and-adapt installer]
    └──required-by──> [Headless training entrypoint]
                          └──required-by──> [Resumable training]
                          └──required-by──> [Auto-export on completion]

[Pretrained base model verification]
    └──required-by──> [Training pre-flight doctor checks]
                          └──must-run-before──> [Headless training entrypoint]

[build_rvc_train_subprocess_cmd (pure fn)]
    └──required-by──> [Headless training entrypoint]
    └──enables──>     [Unit tests without GPU]

[Skip-if-done guards]
    └──enhances──> [Resumable training]
    └──requires──> [Headless training entrypoint produces deterministic artifact paths]

[Experiment manifest]
    └──requires──> [Auto-export on completion]

[Smart batch-size default]
    └──enhances──> [CLI flags]
    └──requires──> [GPU VRAM doctor check]
```

### Dependency Notes

- **Detect-and-adapt installer required before training entrypoint:** The training entrypoint
  assumes `rvc/.venv` and pretrained weights exist; the installer is what creates them on a bare pod.
- **Skip-if-done guards require deterministic artifact paths:** The guard for "skip preprocess"
  checks for `rvc/logs/<name>/preprocess.log`; the guard for "skip extraction" checks for
  `rvc/logs/<name>/3_feature768/*.npy`. Both paths are fixed by RVC's own convention.
- **`build_rvc_train_subprocess_cmd` enables unit tests:** Because it is a pure function returning
  `list[str]`, tests can assert correct argv construction without a GPU or RVC venv.

---

## MVP Definition

### Launch With (this milestone)

- [x] Detect-and-adapt bootstrap script (zero to provisioned on bare CUDA pod)
- [x] `python src/train.py` headless entrypoint with `--experiment-name`, `--dataset-dir`, `--sample-rate`, `--epochs`, `--batch-size`, `--rvc-version`, `--f0-method`, `--save-every`
- [x] `build_rvc_train_subprocess_cmd` pure function + unit tests
- [x] Skip-if-done guards on preprocess, F0-extraction, feature-extraction stages
- [x] Resumable training (do not wipe `rvc/logs/<name>/` on re-invocation)
- [x] Pre-flight doctor checks: pretrained weights, GPU VRAM, disk space, dataset non-empty
- [x] Auto-export: `models/<name>/<name>.pth` + `added_*.index` on completion
- [x] Non-zero exit codes (0 / 1 / 3)
- [x] Remote dataset pull via `--dataset-url` (curl/wget/rclone, no SDK)
- [x] Experiment manifest `models/<name>/manifest.json`

### Add After Validation (v1.x)

- [ ] Smart batch-size default from VRAM probe — useful but requires testing across GPU SKUs
- [ ] Structured training log tail on failure — polish; currently the subprocess stderr leaks through

### Future Consideration (v2+)

- [ ] Fast-setup optimization (parallel installs, prebuilt wheels) — user explicitly deprioritized
- [ ] Tensorboard metrics export in manifest — only useful once multiple training runs accumulate

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Detect-and-adapt bootstrap | HIGH | MEDIUM | P1 |
| Headless training entrypoint (5 core flags) | HIGH | MEDIUM | P1 |
| `build_rvc_train_subprocess_cmd` (pure fn + tests) | HIGH | SMALL | P1 |
| Resumable training / skip-if-done guards | HIGH | MEDIUM | P1 |
| Pre-flight doctor checks (training path) | HIGH | SMALL | P1 |
| Auto-export on completion | HIGH | SMALL | P1 |
| Non-interactive / deterministic exit codes | HIGH | SMALL | P1 |
| Remote dataset pull (`--dataset-url`) | MEDIUM | SMALL | P1 |
| Experiment manifest JSON | MEDIUM | SMALL | P2 |
| Smart batch-size default from VRAM | MEDIUM | SMALL | P2 |
| Structured log tail on failure | LOW | SMALL | P2 |
| Pod shutdown documentation (README) | MEDIUM | SMALL | P2 |

---

## Competitor / Reference Analysis

| Feature | RVC infer-web.py | rvc-easy / typical Colab notebooks | Our headless CLI |
|---------|------------------|-------------------------------------|------------------|
| Training entry | 5 manual button clicks in Gradio UI | Notebook cells; semi-manual | Single `python src/train.py` invocation |
| Resumable | Only if you don't clear logs manually | Usually restarts from epoch 0 | Skip-if-done guards + RVC's native checkpoint resume |
| Auto-export | Manual "install_model.sh" step | Manual cell | Automatic on completion |
| Pre-flight checks | None (fails mid-run) | None | doctor checks before billing starts |
| Remote dataset | Not supported | `gdown` or manual upload | `--dataset-url` via curl/rclone |
| Non-interactive | No (requires browser) | Partially (still manual steps) | Fully non-interactive |
| Provider lock-in | None | Google Colab specific | None (generic Linux + NVIDIA driver) |

---

## Sources

- `rvc/infer-web.py` lines 218–778: all training functions (`preprocess_dataset`,
  `extract_f0_feature`, `click_train`, `train_index`, `train1key`) and the training tab UI
  (lines 1171–1422). Verified step order, all parameters, and subprocess command construction.
- `rvc/infer/lib/train/utils.py` lines 291–366: `get_hparams()` — authoritative list of
  train.py CLI flags (`-se`, `-te`, `-pg`, `-pd`, `-g`, `-bs`, `-e`, `-sr`, `-sw`, `-v`, `-f0`,
  `-l`, `-c`).
- `rvc/infer/modules/train/preprocess.py` lines 1–16: preprocess script argv convention.
- `rvc/infer/modules/train/extract/extract_f0_print.py` lines 19–30: extract_f0 argv convention.
- `rvc/assets/pretrained_v2/`: 12 pretrained weight files confirmed present.
- `.planning/PROJECT.md`: requirements (Active + Out of Scope sections), constraints, and
  key decisions. All anti-features cross-checked against PROJECT.md Out of Scope list.
- `.planning/codebase/ARCHITECTURE.md`: existing CLI patterns, doctor-first architecture,
  subprocess-wrapper discipline, two-venv boundary.

---

*Feature research for: headless RVC voice-model training on rented GPU pods*
*Researched: 2026-04-09*
