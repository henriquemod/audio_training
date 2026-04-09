# train_audio_model

## What This Is

A self-contained voice-cloning toolkit built on RVC (Retrieval-based Voice Conversion). Today it ships an Edge-TTS → ffmpeg → RVC inference CLI for generating speech from a trained voice. The next milestone makes the *training* half of the workflow first-class so an entire model can be trained end-to-end on a rented GPU pod from a clean Ubuntu box.

## Core Value

A single user — me — can rent a GPU pod, run two bash scripts, and walk away with a downloadable `.pth` + `.index` voice model trained from raw audio I provided.

## Requirements

### Validated

<!-- Inferred from existing code (see .planning/codebase/) -->

- ✓ Doctor-first CLI architecture — `src/doctor.py` is single source of truth for environment health checks — existing
- ✓ Two-venv isolation — `./.venv` (app) and `./rvc/.venv` (torch + fairseq) never share interpreter state — existing
- ✓ RVC pinned by commit hash (`7ef19867780cf703841ebafb565a4e47d1ea86ff`), invoked only as a subprocess via `build_rvc_subprocess_cmd` — existing
- ✓ Audio preprocess pipeline — `src/preprocess.py:run_preprocess` handles canonicalize → denoise → loudnorm → slice (3–15s clips with RMS/peak filters) — existing
- ✓ Edge-TTS → ffmpeg → RVC inference pipeline — `src/generate.py` produces a final WAV from input text and a model name — existing
- ✓ `scripts/setup_rvc.sh` clones the pinned RVC commit, builds `rvc/.venv` with torch 2.1.2 + CU121, installs RVC requirements, downloads weights — existing
- ✓ `scripts/install_model.sh` promotes a trained model from `rvc/assets/weights/` + `rvc/logs/` into `./models/` — existing
- ✓ Subprocess-wrapper discipline — all ffmpeg calls go through `src/ffmpeg_utils.py:run_ffmpeg`; no `shell=True` anywhere — existing
- ✓ `scripts/check.sh` runs doctor + pytest unit + pytest integration (`-m "not network"`) + ruff — existing

### Active

<!-- Pod-Ready Training milestone -->

- [ ] One-shot bootstrap: a single bash script takes a clean Ubuntu + NVIDIA-driver pod from zero to fully provisioned (CUDA toolkit if missing, Python 3.10 via mise if missing, app venv, RVC venv, torch 2.1.2 + CU121, RVC clone + pinned commit, RVC weights)
- [ ] Detect-and-adapt provisioning — installer probes each layer (driver → CUDA toolkit → Python 3.10 → torch → RVC) and installs only what is missing, so the same script works on a bare CUDA image, a PyTorch base image, or anything in between
- [ ] Headless training CLI — wraps RVC's underlying training scripts (`trainset_preprocess_pipeline_print.py` / `extract_f0_print.py` / `extract_feature_print.py` / `train_nsf_sim_cache_sid_load_pretrain.py`) into a single `python src/train.py` entrypoint, with no browser, no port forwarding, no manual webui clicks
- [ ] End-to-end training script — one bash command runs preprocess → feature extraction → model training → export, given a directory of raw audio and an experiment name
- [ ] CLI flags for hyperparameters — `--epochs`, `--batch-size`, `--sample-rate`, `--rvc-version`, `--save-every`, etc., with sane defaults that work for a typical training run
- [ ] Configurable audio source — training script accepts either a pre-uploaded local directory OR a remote URL (S3 / R2 / HTTP) and pulls before training
- [ ] Resumable training — checkpoints persist across pod restarts; a re-invoked training run picks up where it left off instead of restarting from epoch 0
- [ ] Auto-export on completion — when training finishes, the final `.pth` weight and `added_*.index` file are bundled into `./models/<experiment>/` (matching the existing layout `install_model.sh` produces) ready for download
- [ ] Doctor coverage for the training path — new doctor checks for the training-only prerequisites (pretrained base models, sample rate compatibility, disk space, GPU VRAM headroom) so failures surface before a billable training run starts
- [ ] Pod-shutdown documentation — README section describing how to wire common provider auto-stop hooks (RunPod, Vast, Lambda, generic systemd) to the training script's exit / sentinel file

### Out of Scope

- **Provider-specific integrations (RunPod SDK, Vast CLI, Lambda API)** — too much surface area; "generic Linux + NVIDIA driver" is the contract instead. Providers are documented, not coded against.
- **Auto-shutdown implementation** — no `shutdown -h now`, no provider API calls. Script exits cleanly with a known status; user wires their own auto-stop. Reason: we don't want the script to be responsible for terminating a billable resource if export hasn't been verified downloaded.
- **Fast-setup optimization (parallel installs, aggressive caching, prebuilt wheels)** — explicitly deprioritized this milestone. Correctness over speed; the user will trade 5 minutes of setup for a script that works on the first try.
- **Web UI (RVC `infer-web.py`) for training** — the headless CLI replaces it. The webui binary still exists in the vendored RVC clone for ad-hoc debugging but is not a supported workflow path.
- **Inference improvements (batching, longer text, better error recovery)** — generation stays exactly as it is this milestone. Not regressed, not extended. Reason: focus.
- **Inference on the pod** — no smoke generation step inside the training script. The pod produces a model file; what happens to it after download is the user's call.
- **Cloud storage integrations beyond a generic URL pull** — no S3 SDK, no boto3. `curl`/`wget`/`rclone` invoked via shell is enough for the foreseeable future.
- **Multi-GPU / distributed training** — single-GPU pods only. Reason: simplicity and matches actual usage.
- **A separate dataset-management feature beyond the existing preprocess pipeline** — the existing slicer/loudnorm pipeline is sufficient input for training.

## Context

**Technical environment:**
- Existing codebase is a Python 3.10 typer CLI with a doctor-first architecture (`src/doctor.py`), pure ffmpeg arg builders (`src/preprocess.py`), and a pinned RVC clone invoked via `subprocess.run(..., cwd=RVC_DIR)`.
- The two-venv split (`./.venv` and `./rvc/.venv`) is load-bearing — RVC needs torch 2.1.2 + CU121 + fairseq + an ancient gradio, none of which can coexist with the app venv. This must hold in pod environments too.
- `scripts/setup_rvc.sh` already does about 60% of the bootstrap work for the RVC venv. The pod-ready setup script will build on it rather than replace it — it adds the system-layer install (CUDA toolkit, Python 3.10) and orchestrates `setup_rvc.sh` underneath.
- Existing tests (`tests/unit/`, `tests/integration/`) cover ffmpeg arg builders, RVC subprocess builder, slicer behavior, and doctor checks. The new training CLI must follow the same pattern: pure helpers + a thin CLI + unit tests on the helpers + integration smoke that doesn't require GPU.
- The project has a `.mise.toml` pinning Python 3.10. **The user prefers `mise` for Python version management** (private memory).

**Operational context:**
- Pods bill by the minute. Time is literally money during a training run. The bootstrap and training scripts must not require any interactive prompts — every input is a CLI flag or env var.
- "Generic Linux + NVIDIA driver" was the explicit choice over provider-specific code — the same script must work whether the base image is bare CUDA, a PyTorch image, or a minimal Ubuntu with only the driver. "Detect and adapt" is the design rule.
- Cost-saving priorities (in user's stated order): **resumable training**, **auto-export**, **shutdown documentation**. Notably, "fast setup" was *not* selected — user is willing to wait for setup if it works on the first run.

**Known issues to address:**
- `scripts/setup_rvc.sh` currently assumes `./.venv/bin/python` already exists and is Python 3.10. On a bare pod, that's not true. The bootstrap script must create the app venv first.
- RVC's training step uses several `tools/*.py` scripts inside the cloned repo. These must be invoked via `rvc/.venv/bin/python` from `cwd=RVC_DIR` — the same subprocess discipline as inference. We need a `build_rvc_train_subprocess_cmd` (analogous to `build_rvc_subprocess_cmd`) that is a pure function and unit-tested.
- Pretrained base models (`assets/pretrained_v2/*.pth`) must be downloaded for training to start. `rvc/tools/download_models.py` already handles this — bootstrap must verify it ran successfully.
- RVC's training pipeline writes to `rvc/logs/<experiment>/` and `rvc/assets/weights/<experiment>.pth`. The auto-export step must locate the latest checkpoint and `added_*.index` and copy them to `./models/<experiment>/` deterministically.

## Constraints

- **Tech stack**: Python 3.10 only (pinned by `pyproject.toml` and `.mise.toml`). RVC venv stays on torch 2.1.2 + CUDA 12.1 + fairseq 0.12.2 + gradio 3.34.0 + `pip<24.1`. Reason: pinned upstream RVC commit only works with this combination; deviating breaks fairseq install.
- **Tech stack**: RVC venv pins `pip<24.1`. Reason: fairseq 0.12.2 has legacy PEP 440 metadata that pip 24.1+ rejects.
- **Tech stack**: `mise` is the Python version manager. Reason: user preference and existing `.mise.toml`.
- **Compatibility**: New training code must coexist with existing inference code without changing it. The `src/generate.py` pipeline must keep working byte-for-byte.
- **Compatibility**: All ffmpeg shell-outs must go through `src/ffmpeg_utils.py:run_ffmpeg`. All RVC subprocess invocations must follow the pattern in `src/generate.py:build_rvc_subprocess_cmd` (pure arg builder + thin runner).
- **Compatibility**: Subprocess discipline — no `shell=True`, no bare `subprocess.run("...")` strings, no shared Python imports across the venv boundary.
- **Performance**: Training script must be resumable — interrupted runs (`SIGTERM`, pod kill, network drop) must continue from the last checkpoint on next invocation, not restart.
- **Dependencies**: RVC clone stays pinned to commit `7ef19867780cf703841ebafb565a4e47d1ea86ff`. Vendoring over dependency creep — no new pip packages in the app venv unless absolutely necessary.
- **Dependencies**: No new framework runtime (no Flask, no FastAPI, no async server, no DB, no scheduler). Stays a thin typer CLI plus shell glue.
- **Operational**: No interactive prompts in any pod-side script — every input is a CLI flag, env var, or config file. Idle interactive prompts on a billing pod are unacceptable.
- **Operational**: No provider-specific code or SDKs. The contract is "Linux + NVIDIA driver + bash". Provider integrations are documented patterns only.
- **Security**: `.env` is never committed. `settings.local.json` is never committed.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Headless CLI wrapper around RVC's training scripts (no webui) | Webui requires a browser, port forwarding, and manual clicks per run — incompatible with one-shot pod usage | — Pending |
| CLI flags with sane defaults (no YAML config file) | Matches existing `src/generate.py` pattern; simpler; one fewer thing to upload to a pod | — Pending |
| "Detect and adapt" install layers (driver → CUDA → Python → torch → RVC) | Same script works across bare CUDA images, PyTorch base images, and minimal Ubuntu pods, with no provider lock-in | — Pending |
| Build pod bootstrap *on top of* `scripts/setup_rvc.sh` instead of replacing it | Existing script already handles the RVC venv correctly; reusing it minimizes risk to inference path | — Pending |
| Audio source: support both pre-uploaded directory and remote URL pull | Decoupling source from training lets the user choose scp/rsync or s3/curl per run without code changes | — Pending |
| Resumable training is mandatory; auto-export is mandatory; auto-shutdown is documented only | Pods bill by the minute; loss on restart is the biggest cost risk; auto-shutdown is provider-specific so it stays in the README | — Pending |
| Inference (`src/generate.py`) is frozen for this milestone | Scope discipline — training is the focus, inference must not regress but must not be extended either | — Pending |
| Two-venv split is preserved on the pod | RVC's torch/fairseq combo cannot coexist with the app venv; this is non-negotiable upstream | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-09 after initialization*
