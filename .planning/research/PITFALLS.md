# Domain Pitfalls

**Domain:** One-shot pod bootstrap + headless RVC voice-model training
**Researched:** 2026-04-09
**Codebase inspected:** rvc/infer/modules/train/preprocess.py, train.py, extract_f0_print.py, extract_feature_print.py, infer/lib/train/utils.py, infer/lib/train/process_ckpt.py, infer-web.py, scripts/setup_rvc.sh

---

## Critical Pitfalls

Mistakes that cause rewrites, silent bad models, or irreversible pod-hour waste.

---

### CRIT-1: Missing pretrained base weights → silent random-init training

**What goes wrong:** If `assets/pretrained_v2/f0G40k.pth` (or whichever sr/version combo is being used) is absent, `train.py` lines 225-254 catch the `load_checkpoint` exception with a bare `except:` and fall through to a random-weight initialization — no error, no abort, no log line that says "pretrained weights not loaded". The model trains from scratch. This produces a model that converges much more slowly or not at all within a normal epoch budget. On a pod, the user pays for the full run, downloads the result, and discovers it is unusable at inference time.

**Evidence:** `infer/modules/train/train.py` lines 208-254: the `try: ... except:` block that loads G/D checkpoints also doubles as the pretrained-weights loader. `pretrainG` defaults to `""` if the `-pg` flag is empty (utils.py line 320). `infer-web.py` line 551-554 logs "No pretrained Generator" as an INFO only — not an abort.

**Symptom:** Training completes with plausible-looking loss numbers but the exported model produces robotic or unintelligible output at inference.

**Detection:** Doctor pre-flight check: assert `assets/pretrained_v2/f0G{sr}.pth` and `assets/pretrained_v2/f0D{sr}.pth` exist and are non-empty before invoking train.py. Also assert that `assets/pretrained/` equivalents exist for v1 paths.

**Prevention:**
- `src/train.py` must always pass `-pg` and `-pd` with the correct resolved paths.
- Doctor check `check_rvc_pretrained_weights(sr, version, if_f0)` added before training starts.
- Bootstrap must verify `download_models.py` completed (check file existence + size, not just exit code — the script uses `requests.get` with `r.raise_for_status()` but has no retry).

**Phase mapping:** Bootstrap (download verification) + Doctor (pre-flight check before train) + Training-CLI (mandatory `-pg`/`-pd` arg resolution, not optional).

---

### CRIT-2: Sample-rate mismatch between `--sr` flag and actual preprocessed audio

**What goes wrong:** RVC's `preprocess.py` (line 83) calls `load_audio(path, self.sr)` where `self.sr` is the sr integer passed as `sys.argv[2]`. It resamples input to whatever sr you pass. The resulting `0_gt_wavs/` files are written at that sr. Then `train.py` reads `hps.data.sampling_rate` from `config.json` — which is determined by the `--sample_rate` flag (utils.py line 385). If you preprocess at 40000 but pass `--sample_rate 48k` to train, the DataLoader reads WAVs that say 40 kHz but the model expects 48 kHz. The mismatch is silent — PyTorch loads the raw PCM without checking the WAV header's declared rate.

The more common variant: the user runs the existing `src/preprocess.py` (which hardcodes `CANONICAL_SR = 44100`), then passes `--sample_rate 40k` (40000 Hz) to training. 44.1 kHz ≠ 40 kHz. Training runs to completion; the output model has the wrong internal sample rate assumption baked in.

**Evidence:** `src/preprocess.py` line 36 `CANONICAL_SR = 44100`. RVC's own preprocess (`infer/modules/train/preprocess.py` line 83) resamples to whatever sr it receives. `config.json` controls the model's internal sr; it is written from `configs/v2/40k.json` or `48k.json` etc., chosen by the sr flag.

**Symptom:** Model trains to apparent convergence but produced audio is pitched wrong or has aliasing artifacts at inference time. No error at any stage.

**Detection:** Before training, assert that a sample from `logs/<exp>/0_gt_wavs/*.wav` has a sampling rate equal to the integer value of the `--sample_rate` flag. `scipy.io.wavfile.read` or `soundfile.info` gives the declared rate in one line.

**Prevention:**
- The headless training CLI must run RVC's own `preprocess.py` with the same sr that training will use, not assume the app's preprocess.py output is compatible.
- Doctor check: sample 3 files from `0_gt_wavs/`, assert all have sr matching the training sr argument.
- Document clearly that `src/preprocess.py` is for inference-pipeline preprocessing; RVC training preprocessing is a separate step (RVC's `infer/modules/train/preprocess.py`).

**Phase mapping:** Training-CLI (must own both RVC preprocess invocation and training invocation with the same sr) + Doctor (pre-flight audio sr sanity check).

---

### CRIT-3: `if_f0` / f0-method mismatch silently trains the wrong model architecture

**What goes wrong:** `train.py` line 136 branches on `hps.if_f0 == 1` to choose between `RVC_Model_f0` (NSF-conditioned on pitch) and `RVC_Model_nof0`. If `if_f0=1` is passed to train but `extract_f0_print.py` was never run (so `2a_f0/` and `2b-f0nsf/` are empty), `click_train` in `infer-web.py` lines 490-498 takes the f0 path when building `filelist.txt` and the intersection with an empty f0 dir yields an empty filelist. Training starts with zero samples and exits immediately — or more precisely, `os._exit(2333333)` is called (train.py line 635) with a non-standard exit code that looks like success to a naive `$?` check.

The reverse is also dangerous: passing `if_f0=0` when f0 features were extracted means the model is trained without pitch guidance but the user expects pitch-shifting capability at inference.

**Evidence:** `train.py` line 136-155 (DataLoader branch on `if_f0`). `infer-web.py` lines 490-502 (filelist intersection logic: f0 dirs included only when `if_f0_3` is True). `train.py` line 635 `os._exit(2333333)`.

**Symptom:** Training exits immediately (empty filelist) with `os._exit(2333333)` — exit code is 2333333 % 256 = some non-zero byte on Linux, but may appear as success in some shell checks. Or: model trains without pitch control and pitch-shift at inference does nothing useful.

**Detection:**
- After filelist generation, assert `len(filelist.txt lines) > 0` before invoking train.py.
- Check that if `if_f0=1`, `2a_f0/` and `2b-f0nsf/` are non-empty.
- Check that the f0 method used for extraction is compatible: `rmvpe` requires `assets/rmvpe/rmvpe.pt`; `harvest`/`dio` require `pyworld`.

**Prevention:**
- Training-CLI must enforce: if `if_f0=1`, f0 extraction step must have run first (check directory non-empty).
- Wrap the `train.py` subprocess and treat exit code != 0 and exit code != 2333333 as failure. For exit code 2333333, treat as "training completed normally" (this is RVC's intentional normal-exit sentinel).
- Add a `check_rvc_filelist(exp_dir, if_f0)` doctor check.

**Phase mapping:** Training-CLI (pipeline step ordering enforcement, filelist pre-check) + Doctor (filelist non-empty check before train subprocess).

---

### CRIT-4: `assets/hubert/hubert_base.pt` not on disk when `extract_feature_print.py` runs

**What goes wrong:** `extract_feature_print.py` line 55 hardcodes `model_path = "assets/hubert/hubert_base.pt"` as a relative path. It checks `os.access(model_path, os.F_OK)` (line 83) and, if missing, prints an error and calls `exit(0)` — exit code zero. The subprocess returns 0 (apparent success). All downstream `.npy` feature files are missing. `filelist.txt` will have zero entries because the intersection with an empty `3_feature768/` dir is empty.

**Evidence:** `extract_feature_print.py` lines 55, 83-88: `exit(0)` on missing hubert model.

**Symptom:** Feature extraction appears to succeed (exit 0), but `3_feature768/` is empty. Filelist is empty. Train exits immediately. Pod hour wasted.

**Detection:** After feature extraction subprocess, assert that `3_feature768/` (or `3_feature256/` for v1) contains at least one `.npy` file. Doctor pre-flight: assert `assets/hubert/hubert_base.pt` exists, readable, and non-empty (size > 100 MB).

**Prevention:**
- Bootstrap must verify `download_models.py` success by checking file sizes, not just exit code.
- `check_rvc_weights` in doctor.py already checks for hubert and rmvpe — extend it to verify file size thresholds (hubert_base.pt should be ~360 MB).
- Training-CLI must check `3_feature768/` non-empty after feature extraction before proceeding.

**Phase mapping:** Bootstrap (download verification with size checks) + Doctor (pre-flight weight existence + size) + Training-CLI (post-step assertion after extraction subprocess).

---

### CRIT-5: `os._exit(2333333)` exit code confusion

**What goes wrong:** `train.py` line 635 uses `os._exit(2333333)` as its "training done" signal. On Linux, `os._exit(n)` with n > 255 truncates to `n % 256 = 2333333 % 256 = 61`. The subprocess returns exit code 61. If the training orchestrator treats only 0 as success, it misidentifies a completed training run as an error. If it treats any non-zero as error, it will always think training failed.

**Evidence:** `train.py` line 635: `os._exit(2333333)`.

**Symptom:** Post-training automation reports "training failed" even when training completed successfully. Export step is skipped. Pod is not marked done.

**Prevention:**
- `build_rvc_train_subprocess_cmd` or the runner wrapping it must document this and treat exit code 61 (and 0) as success. Add a comment explaining the magic number.
- Cross-check: after subprocess returns with code 61, verify `rvc/assets/weights/<name>.pth` exists as the final confirmation.

**Phase mapping:** Training-CLI (subprocess exit code handling).

---

### CRIT-6: Version/architecture mismatch between pretrained weights and `--version` flag

**What goes wrong:** `get_pretrained_models()` in `infer-web.py` (line 398) constructs the pretrained path as `assets/pretrained{path_str}/{f0_str}G{sr2}.pth` where `path_str = "" if version19 == "v1" else "_v2"`. Using `assets/pretrained_v2/f0G40k.pth` with `--version v1` loads a v2 architecture into a v1 model class. PyTorch's `load_state_dict(strict=False)` (utils.py line 126) silently ignores shape mismatches — keys that don't match fall back to random init (line 124).

The symptom is identical to CRIT-1: training runs to completion, bad model output.

**Evidence:** `infer/lib/train/utils.py` lines 110-124: shape mismatch raises KeyError, caught, missing key gets random init value, `strict=False` suppresses any final error.

**Prevention:**
- The training-CLI must enforce: `v1 → assets/pretrained/`, `v2 → assets/pretrained_v2/`. No mixing.
- Default recommendation: always use `v2` with `40k` or `48k`. `v1` only for specific backward-compat needs (document this).
- Doctor: assert pretrained file exists at the exact resolved path before passing `-pg`/`-pd` to train.py.

**Phase mapping:** Training-CLI (argument validation before subprocess invocation) + Doctor (resolved path existence check).

---

## Moderate Pitfalls

Mistakes that cause failures caught at runtime but waste setup time or pod hours.

---

### MOD-1: `setup_rvc.sh` re-run on a re-used pod breaks if `.venv` was wiped but `rvc/` was not (or vice versa)

**What goes wrong:** `setup_rvc.sh` line 39 calls `doctor.py --system-only` using `$ROOT_PYTHON = $PROJECT_ROOT/.venv/bin/python`. If the app venv was wiped but `rvc/` still exists from a previous pod snapshot, the script fails immediately because `$ROOT_PYTHON` doesn't exist. The fix path (re-create `.venv`) is not in the script.

Conversely: if `rvc/.venv` exists but was created from a different Python 3.10 binary (e.g., system Python vs mise Python), the venv may have a broken `python` symlink.

**Evidence:** `setup_rvc.sh` line 15: `ROOT_PYTHON="$PROJECT_ROOT/.venv/bin/python"`. Lines 42-45: hard fail if not Python 3.10. No check that the venv's Python symlink is valid.

**Symptom:** `setup_rvc.sh` aborts at line 39 with "No such file or directory". Requires manual intervention on a billing pod.

**Prevention:**
- The pod bootstrap script must create `.venv` before calling `setup_rvc.sh`. This is already noted in PROJECT.md constraints.
- Add a pre-check to the bootstrap: if `rvc/.venv` exists but `rvc/.venv/bin/python --version` fails, wipe `rvc/.venv` and re-run.
- `setup_rvc.sh`'s `--force` flag wipes `rvc/` entirely. The bootstrap script should offer a `--force-rvc-venv` that wipes only `rvc/.venv` without wiping the cloned source or pretrained weights.

**Phase mapping:** Bootstrap (layer detection + re-run safety).

---

### MOD-2: `pip<24.1` pin in `rvc/.venv` cannot survive a `pip install --upgrade pip` elsewhere

**What goes wrong:** If any setup step calls `pip install --upgrade pip` inside `rvc/.venv` (e.g., a naive bootstrap step that upgrades pip everywhere before installing packages), the `pip<24.1` pin is overridden. The next `pip install -r requirements.txt` fails with `ResolutionImpossible` on fairseq 0.12.2's `PyYAML>=5.1.*` legacy metadata.

The error message from pip 24.1+ is: `ERROR: Cannot install fairseq==0.12.2 because these package versions have conflicting dependencies.` This is confusing because the real cause is pip's version, not a genuine dependency conflict.

**Evidence:** `setup_rvc.sh` lines 85-91: pip is pinned immediately after venv creation, before any requirements install. But if the bootstrap script runs `pip install --upgrade pip` globally before calling `setup_rvc.sh`, the pin is moot.

**Symptom:** RVC venv setup fails with pip ResolutionImpossible. Error message misleadingly points to fairseq/PyYAML conflict, not pip version.

**Prevention:**
- Bootstrap script must never upgrade pip in `rvc/.venv`. Document: "do not upgrade pip in the RVC venv".
- Bootstrap may upgrade pip in `.venv` (app venv) freely.
- If bootstrap must upgrade pip system-wide, do it before creating `rvc/.venv`, so the venv is seeded from the upgraded pip but immediately downgraded by `setup_rvc.sh` line 91.
- Add a doctor check: `rvc/.venv/bin/pip --version` output should show `< 24.1`.

**Phase mapping:** Bootstrap (ordering of pip operations) + Doctor (pip version check in rvc venv).

---

### MOD-3: `fairseq` model cache not populated; `fairseq.checkpoint_utils.load_model_ensemble_and_task` silently downloads on first call

**What goes wrong:** `extract_feature_print.py` line 89 calls `fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])` where `model_path = "assets/hubert/hubert_base.pt"`. This is a local file path, not a download trigger — but fairseq internally may look for cached config files in `~/.cache/torch/hub/` or `TORCH_HUB_DIR`. On a pod without internet access (or with slow HuggingFace access), this may cause a hang or a cryptic error.

More directly: the hubert model uses a `task` object that references config. If fairseq's internal cache is stale or the home directory is on a network mount, this can add 30-120 seconds of silent waiting to each feature extraction subprocess invocation.

**Symptom:** Feature extraction subprocess hangs for minutes before producing output. May time out if the bootstrap added any timeout wrapper.

**Detection:** After model load (line 95 `printt("move model to %s")`), the log should appear quickly. If it takes > 60 seconds, suspect network or cache issue.

**Prevention:**
- Run feature extraction once during setup (warm-up invocation) to populate any fairseq caches before training starts.
- Set `TRANSFORMERS_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` env vars if operating in an air-gapped or restricted-network pod. These prevent fairseq/HuggingFace from attempting network calls.
- Ensure `TORCH_HOME` points to a local fast disk, not a network mount.

**Phase mapping:** Bootstrap (cache warm-up step) + Training-CLI (env var passthrough).

---

### MOD-4: Pod image has `nvidia-smi` working but `torch.cuda.is_available()` returns False

**What goes wrong:** Several pod providers (Vast.ai especially) ship images where the CUDA toolkit is present in a non-standard path, or the CUDA driver version is incompatible with the torch wheel's expected libcuda version. `nvidia-smi` talks to the kernel driver module directly and works fine. `torch.cuda.is_available()` requires the CUDA runtime library (`libcuda.so`) to be discoverable by the dynamic linker, which it may not be if `LD_LIBRARY_PATH` is not set or the toolkit is in `/usr/local/cuda-12.x/` rather than `/usr/local/cuda/`.

**Evidence:** `setup_rvc.sh` line 133 asserts `torch.cuda.is_available()` inside `rvc/.venv`. This is the right check. But if it fails, the error is just "CUDA not available in rvc/.venv" with no diagnosis of why.

**Symptom:** `setup_rvc.sh` fails at the CUDA verification step. `nvidia-smi` shows a GPU. Pod hour charged during debugging.

**Prevention:**
- Bootstrap must set `LD_LIBRARY_PATH` to include CUDA lib dirs before creating or using `rvc/.venv`. Probe common paths: `/usr/local/cuda/lib64`, `/usr/local/cuda-12.1/lib64`, `/usr/lib/x86_64-linux-gnu`.
- Add diagnostic: if `torch.cuda.is_available()` returns False, print `nvidia-smi` output, `nvcc --version`, and `ldconfig -p | grep libcuda` to help diagnose.
- Doctor check `check_rvc_torch_cuda` already exists — extend its `fix_hint` with `LD_LIBRARY_PATH` diagnostic steps.
- For the torch 2.1.2 + CU121 wheel, the minimum driver version is 525.x. Add a driver version check before installing torch.

**Phase mapping:** Bootstrap (LD_LIBRARY_PATH setup, driver version check) + Doctor (enhanced fix_hint for cuda false).

---

### MOD-5: Interactive prompts from apt/pip/tzdata on a non-interactive pod shell

**What goes wrong:** The bootstrap script will likely call `apt-get install` for CUDA toolkit, Python build deps, etc. Several packages prompt interactively: `tzdata` asks for timezone region, `debconf` for keyboard layout, some CUDA packages for license acceptance. On a non-interactive shell (pod SSH session with no tty), these prompts either hang indefinitely or crash with `debconf: unable to initialize frontend: Dialog`.

**Evidence:** Common knowledge, but specifically: `apt-get install -y cuda-toolkit-12-1` on a bare Ubuntu 22.04 triggers `tzdata` interactive prompt unless `DEBIAN_FRONTEND=noninteractive` and `TZ` are pre-set. CUDA installer scripts (`.run` files) are even worse — they need `--silent --override` flags.

**Symptom:** Bootstrap script hangs on `apt-get` for unlimited time. Pod billed for the hang duration.

**Prevention:**
- Every `apt-get` call in the bootstrap must be prefixed with `DEBIAN_FRONTEND=noninteractive`.
- Export `TZ=UTC` before any apt install.
- Pre-seed debconf: `echo 'tzdata tzdata/Areas select Etc' | debconf-set-selections && echo 'tzdata tzdata/Zones/Etc select UTC' | debconf-set-selections`.
- Never use `.run` CUDA installers; use the `cuda-toolkit-12-1` apt package with the NVIDIA apt repo.
- All `pip install` calls already non-interactive (pip is non-interactive by default), but add `--no-input` flag explicitly.

**Phase mapping:** Bootstrap (all apt-get calls must use DEBIAN_FRONTEND=noninteractive).

---

### MOD-6: App venv accidentally imports torch (cross-venv contamination)

**What goes wrong:** `src/train.py` is in the app venv. If it `import torch` at module level (e.g., to check VRAM), the app venv will fail to start because torch is not installed there. Worse: if a developer adds `torch` to `pyproject.toml` as a dependency (to "fix" the ImportError), pip would resolve a CPU torch wheel into the app venv. Then `src/doctor.py` might accidentally start using app-venv torch for checks, and the venv boundary breaks.

The existing `src/generate.py` demonstrates the correct pattern: it never imports torch — it delegates everything to `rvc/.venv` via subprocess. `src/train.py` must follow the same discipline.

**Evidence:** `src/generate.py` has no torch import. `src/doctor.py:check_rvc_torch_cuda` (line 34 of ARCHITECTURE.md) runs the CUDA check via subprocess into `rvc/.venv`, not by importing torch directly.

**Symptom:** `ImportError: No module named 'torch'` when running `src/train.py` in the app venv, OR accidentally polluted app venv with a CPU-only torch.

**Prevention:**
- Lint rule: add a ruff rule or pre-commit check that forbids `import torch`, `import fairseq`, `import gradio` in `src/*.py`.
- Code review checklist: any helper in `src/train.py` that needs to know about VRAM must delegate to a subprocess call into `rvc/.venv`, not a direct import.
- Unit tests for `src/train.py` must run without torch installed.

**Phase mapping:** Training-CLI (code architecture discipline enforced at review time) + Doctor (should remain subprocess-based for all RVC checks).

---

### MOD-7: `sys.path` manipulation in RVC scripts reaches into unexpected locations

**What goes wrong:** `preprocess.py` line 7-8, `extract_f0_print.py` line 8: both do `sys.path.append(now_dir)` where `now_dir = os.getcwd()`. These scripts must be invoked with `cwd=RVC_DIR`. If the subprocess is launched with the wrong `cwd`, `now_dir` will be the project root or some other directory, and `from infer.lib.audio import load_audio` (line 23 of preprocess.py) will fail with ModuleNotFoundError.

This is a subprocess-discipline issue specific to training scripts, not just inference (inference was already handled by `build_rvc_subprocess_cmd` with `cwd=RVC_DIR`).

**Evidence:** `preprocess.py` line 7-8; `extract_feature_print.py` line 8. All training scripts use `sys.path.append(now_dir)` for relative imports.

**Symptom:** `ModuleNotFoundError: No module named 'infer'` when running training subprocess. Fast fail but confusing error.

**Prevention:**
- `build_rvc_train_subprocess_cmd` (the new pure arg-builder) must always target invocation with `cwd=RVC_DIR`.
- Unit test: assert that `build_rvc_train_subprocess_cmd` output, when run with `subprocess.run(..., cwd=RVC_DIR)`, resolves imports correctly (dry-run mode or `--help` flag check).

**Phase mapping:** Training-CLI (subprocess builder must enforce cwd=RVC_DIR).

---

### MOD-8: Resume fails when only G or only D checkpoint is present

**What goes wrong:** `train.py` lines 208-224: the resume path tries to load `D_*.pth` first, then `G_*.pth`. Both use `utils.latest_checkpoint_path()` (utils.py line 210-215). If D exists but G was deleted (or vice versa — e.g., pod ran out of disk mid-write and only one was flushed), `latest_checkpoint_path()` raises an exception (it calls `f_list[-1]` on an empty list if no match). The outer `except:` (line 221) catches this and falls through to random init + pretrained load — silently resetting all training progress.

**Evidence:** `train.py` lines 208-221: `try/except` on loading both D and G. `latest_checkpoint_path()` line 210-215 in utils.py: `f_list[-1]` on potentially empty list raises `IndexError`.

**Symptom:** Pod restarts, training "resumes" but actually starts from epoch 1 with pretrained weights. All previous checkpoint progress lost. No error message.

**Prevention:**
- Before starting the training subprocess on a resume run, pre-flight check: assert both `G_*.pth` and `D_*.pth` exist in `logs/<exp>/`. If only one is present, abort with an actionable message: "Checkpoint pair incomplete. Delete both and re-run to start fresh."
- Add a `check_rvc_checkpoint_pair(exp_dir)` doctor check.
- Use `--save_every_weights 1` (or the equivalent `hps.save_every_weights`) to write a named checkpoint to `assets/weights/` at each save interval as a backup, independent of the G/D pair.

**Phase mapping:** Doctor (checkpoint pair pre-flight) + Training-CLI (pre-run check on resume) + Bootstrap (disk space check before training, since low disk can cause partial writes).

---

### MOD-9: Locale / `LC_ALL` issues cause subprocess hangs or garbled output

**What goes wrong:** Some minimal Ubuntu pod images have `LC_ALL=C` or no locale set. Python's subprocess output may contain Chinese characters (RVC log messages are partially in Chinese — e.g., `infer-web.py` line 687: `"成功构建索引..."`). If the subprocess stdout is piped and the parent process's locale cannot decode UTF-8, the pipe read raises `UnicodeDecodeError`.

Also: `apt-get` on a C-locale system may produce garbled multi-byte output in error messages.

**Evidence:** `infer-web.py` lines 687-688: Chinese characters in log output. `rvc/infer/lib/train/utils.py` line 286: `load_filepaths_and_text` tries UTF-8 first then falls back to system default encoding (line 284-287) — which on C-locale is ASCII.

**Prevention:**
- Bootstrap must set `export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8` before any apt or pip call.
- All subprocess calls to RVC training scripts must pass `encoding="utf-8"` or read stdout as bytes and decode manually with `errors="replace"`.
- `build_rvc_train_subprocess_cmd` runner should use `text=False` (bytes) and decode with `errors="replace"` to prevent crashes on Chinese log lines.

**Phase mapping:** Bootstrap (locale export) + Training-CLI (subprocess output handling).

---

## Minor Pitfalls

---

### MIN-1: `added_*.index` glob — correct picking rule

**What goes wrong:** After `train_index` runs, the index file is named:
```
added_IVF{n_ivf}_Flat_nprobe_{nprobe}_{exp_dir1}_{version19}.index
```
(see `infer-web.py` lines 682-684). The IVF cluster count `n_ivf` is derived from `min(int(16 * sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)` — it changes with the size of the feature dataset. If `train_index` is re-run (e.g., after adding more training data), a second `added_*.index` file is created with a different IVF count in the filename. Both files exist.

The existing `scripts/install_model.sh` uses `rvc/logs/<name>/added_*.index` as a glob — ambiguous if multiple files exist.

**Correct picking rule:** Sort `added_*.index` files by mtime (most recently modified) and take the last one. This is correct because re-running `train_index` overwrites with a fresh build. Do not sort by filename alphabetically — IVF count is not monotonically related to training quality.

**Evidence:** `infer-web.py` lines 682-685 for the naming pattern. `infer-web.py` line 661 for the n_ivf calculation (data-size-dependent).

**Symptom:** `install_model.sh` glob matches multiple files, `cp` copies one arbitrarily (bash glob order is filesystem-dependent). May copy the stale index from a smaller dataset.

**Prevention:**
- Auto-export logic: `sorted(glob("logs/<exp>/added_*.index"), key=os.path.getmtime)[-1]`
- If zero matches: abort with actionable message ("run train_index step first").
- If multiple matches: log a warning naming all found files, pick the newest by mtime, continue.

**Phase mapping:** Export (auto-export must use mtime-sorted glob, not plain glob).

---

### MIN-2: `if_latest=1` mode creates a single `G_2333333.pth` / `D_2333333.pth` overwritten each save

**What goes wrong:** `train.py` lines 580-593: when `if_latest=1`, checkpoints are saved as `G_2333333.pth` and `D_2333333.pth` — the same filename every save. The resume path (utils.py line 210-215) uses `latest_checkpoint_path` with `"G_*.pth"` glob and sorts by digit extraction: `int("".join(filter(str.isdigit, f)))`. For `G_2333333.pth`, the digit extraction gives `2333333`. This is fine for resume if only this file exists.

But if `if_latest=0` was used in a previous run, both `G_500.pth` and `G_2333333.pth` may be present. `latest_checkpoint_path` would pick `G_2333333.pth` even if `G_500.pth` is actually the most recent epoch.

**Prevention:**
- Always use a single consistent `if_latest` setting for a given experiment. Default recommendation: `if_latest=1` (saves disk space; safe for single-run experiments).
- Doctor: if both `G_2333333.pth` and epoch-numbered `G_*.pth` files coexist in the same experiment dir, warn user and ask them to confirm which to use for resume.

**Phase mapping:** Training-CLI (default `if_latest=1`, document it) + Doctor (checkpoint file consistency check).

---

### MIN-3: Non-WAV input formats cause silent skip in RVC preprocess

**What goes wrong:** RVC's `preprocess.py` (line 111-131) calls `pipeline_mp_inp_dir` which calls `os.listdir(inp_root)` with no extension filter. `load_audio` (imported from `infer.lib.audio`) likely handles only WAV/MP3 via librosa — if given a `.DS_Store` or `.m4a` or `.ogg`, it raises an exception caught by the bare `except:` on line 104: `println("%s -> %s")`. The file is skipped, no exit code change. If all input files are in an unsupported format, `0_gt_wavs/` is empty, and training fails later.

**Evidence:** `preprocess.py` line 104-105: `except: println(traceback)` — per-file failure is logged but not fatal.

**Symptom:** Preprocessing "completes" but `0_gt_wavs/` has zero files. Filelist is empty. Training exits with the 2333333 problem (CRIT-3).

**Prevention:**
- The app's `src/preprocess.py` already canonicalizes audio to WAV before the training pipeline touches it. The workflow should always pipe `src/preprocess.py` output (WAV files) into RVC preprocessing, never raw mixed-format files.
- Doctor: after RVC preprocessing step, assert `0_gt_wavs/` has at least N files.
- Training-CLI: only pass directories that contain `.wav` files to RVC's preprocess script.

**Phase mapping:** Training-CLI (input validation before invoking RVC preprocess).

---

### MIN-4: Unicode filenames in dataset cause filelist parsing failure

**What goes wrong:** `utils.py` line 280-288 `load_filepaths_and_text` opens `filelist.txt` with `encoding="utf-8"` first, then falls back to default encoding on `UnicodeDecodeError`. But if a WAV filename in `0_gt_wavs/` contains non-ASCII characters (e.g., a Japanese voice actor's name), `click_train` writes those paths into `filelist.txt`. On a C-locale pod, the fallback encoding is ASCII, and `filelist.txt` reading fails with `UnicodeDecodeError` during training data loading.

**Prevention:**
- Experiment names and all intermediate paths must be ASCII-only. The training-CLI should validate `experiment_name` matches `^[a-zA-Z0-9_-]+$` and reject anything else.
- Source audio filenames should be sanitized (slugified) before preprocessing. The app's `_slugify` helper in `src/generate.py` can be extracted and reused.

**Phase mapping:** Training-CLI (experiment name validation, input file sanitization).

---

### MIN-5: Very small datasets (< 10 minutes) produce a degenerate FAISS index

**What goes wrong:** `train_index` in `infer-web.py` line 661: `n_ivf = min(int(16 * sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)`. For a tiny dataset, `big_npy.shape[0] // 39` may be very small (e.g., 5-10 clusters). FAISS with IVF < 16 clusters will either fail or produce an unusable index. The `index.train(big_npy)` call raises a FAISS assertion error if `n_train < n_ivf * 39`.

**Evidence:** `infer-web.py` line 639: if `big_npy.shape[0] > 2e5`, KMeans reduction is applied. For small datasets (below 2e5 vectors), this path is skipped, but the small n_ivf issue still applies.

**Symptom:** `train_index` crashes with a FAISS assertion error. No index file written. Auto-export finds no `added_*.index`.

**Prevention:**
- Doctor pre-flight: count total clips and estimated feature vectors from `1_16k_wavs/`. Minimum recommendation is 10 minutes of clean audio (yields ~hundreds of clips and tens of thousands of feature vectors).
- Warn if estimated vector count < 2000 (below which n_ivf < 50 and index quality degrades).

**Phase mapping:** Doctor (dataset size pre-flight warning) + Training-CLI (minimum dataset size validation).

---

### MIN-6: `download_models.py` has no retry and no partial-download detection

**What goes wrong:** `download_models.py` streams each model file via `requests.get` with no retry logic (line 10-16). On a pod with intermittent HuggingFace connectivity, a partial download writes a truncated file. On re-run of `setup_rvc.sh`, the check is only `doctor.check_rvc_weights` which verifies file existence, not file size or integrity. Subsequent `extract_feature_print.py` loads a corrupted hubert model and fails with a cryptic PyTorch deserialization error.

**Evidence:** `download_models.py` lines 10-16: no retry, no checksum. `setup_rvc.sh` line 127 runs `download_models.py` once without validation beyond the doctor check.

**Symptom:** `torch.load(hubert_base.pt)` raises `RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central directory` or similar corruption error. Cryptic, no obvious link to the download.

**Prevention:**
- After `download_models.py`, verify file sizes: `hubert_base.pt` should be ~360 MB, `rmvpe.pt` ~180 MB, each pretrained `.pth` ~100-200 MB.
- Doctor `check_rvc_weights` should include minimum file size thresholds.
- Bootstrap can wrap `download_models.py` in a retry loop (3 attempts).

**Phase mapping:** Bootstrap (download retry + size verification) + Doctor (file size thresholds in weight checks).

---

### MIN-7: `torch.distributed` requires a free port; pod firewalls may block localhost binding

**What goes wrong:** `train.py` line 105: `os.environ["MASTER_PORT"] = str(randint(20000, 55555))`. PyTorch's `dist.init_process_group` binds to this port on localhost. Some pod environments run with network namespaces or restrictive `iptables` rules that block even localhost TCP on high ports. The `dist.init_process_group(backend="gloo", ...)` call hangs indefinitely.

**Symptom:** Training subprocess hangs immediately after start, before any epoch logging.

**Detection:** If training subprocess produces no log output within 60 seconds of start, suspect distributed init hang.

**Prevention:**
- For single-GPU training (`n_gpus=1`), the DDP setup still runs but only process 0 exists. The gloo backend on localhost should work on most pods, but document the requirement.
- Add a timeout to the training subprocess invocation (e.g., 120-second window before first log line before declaring a hang).
- If hang detected, emit diagnostic: "torch.distributed localhost port bind may be blocked. Check iptables or try a different MASTER_PORT."

**Phase mapping:** Training-CLI (subprocess hang detection with timeout + diagnostic message).

---

## Phase-Specific Warnings

| Phase | Likely Pitfall | Mitigation |
|-------|---------------|------------|
| Bootstrap | Pod image has `nvidia-smi` but `torch.cuda.is_available()` False (MOD-4) | Set `LD_LIBRARY_PATH` before creating rvc/.venv; add driver version pre-check |
| Bootstrap | Interactive prompts from apt halt script (MOD-5) | `DEBIAN_FRONTEND=noninteractive TZ=UTC` on all apt calls |
| Bootstrap | Partial download of pretrained weights (MIN-6) | Post-download file size verification; retry wrapper |
| Bootstrap | `pip<24.1` pin destroyed by an upstream `pip upgrade` (MOD-2) | Never upgrade pip in rvc/.venv; do global pip upgrade before venv creation only |
| Bootstrap | Re-run on reused pod with mismatched venv state (MOD-1) | Layer detection; `--force-rvc-venv` option |
| Doctor | Missing pretrained weights not caught before training (CRIT-1) | `check_rvc_pretrained_weights(sr, version, if_f0)` |
| Doctor | hubert model partially downloaded (CRIT-4 / MIN-6) | File size thresholds in `check_rvc_weights` |
| Doctor | Checkpoint pair incomplete before resume (MOD-8) | `check_rvc_checkpoint_pair(exp_dir)` |
| Doctor | pip version in rvc/.venv > 24.0 (MOD-2) | `check_rvc_pip_version()` |
| Doctor | Dataset too small for FAISS index (MIN-5) | Dataset size warning check |
| Training-CLI | Sample rate mismatch between preprocessing and training (CRIT-2) | Enforce same sr through entire pipeline; post-preprocess WAV sr assertion |
| Training-CLI | `if_f0=1` but empty f0 dirs produces empty filelist (CRIT-3) | Step ordering enforcement; filelist non-empty assertion before train subprocess |
| Training-CLI | Wrong version/weights combination (CRIT-6) | Argument validation; resolved path existence check |
| Training-CLI | `os._exit(2333333)` misread as failure (CRIT-5) | Treat exit code 61 as success; verify `assets/weights/<name>.pth` exists |
| Training-CLI | Cross-venv torch import in src/train.py (MOD-6) | Lint rule; subprocess discipline |
| Training-CLI | Wrong `cwd` for RVC subprocess (MOD-7) | `build_rvc_train_subprocess_cmd` enforces `cwd=RVC_DIR` |
| Training-CLI | Distributed init hang (MIN-7) | Subprocess timeout + hang diagnostic |
| Training-CLI | Unicode filenames in dataset (MIN-4) | Experiment name and path sanitization |
| Export | `added_*.index` glob matches multiple files (MIN-1) | mtime-sorted glob; newest wins |
| Export | `if_latest=1` checkpoint naming collision (MIN-2) | Consistent flag; document default |

## Sources

- `rvc/infer/modules/train/train.py` (inspected in full)
- `rvc/infer/modules/train/preprocess.py` (inspected in full)
- `rvc/infer/modules/train/extract_feature_print.py` (inspected in full)
- `rvc/infer/modules/train/extract/extract_f0_print.py` (inspected in full)
- `rvc/infer/lib/train/utils.py` (inspected: `latest_checkpoint_path`, `load_checkpoint`, `get_hparams`, `plot_spectrogram_to_numpy`)
- `rvc/infer/lib/train/process_ckpt.py` (inspected: `savee`, `extract_small_model`)
- `rvc/infer-web.py` (inspected: `click_train`, `train_index`, `get_pretrained_models`, `change_sr2`, `change_version19`)
- `rvc/tools/download_models.py` (inspected in full)
- `scripts/setup_rvc.sh` (inspected in full)
- `.planning/codebase/CONCERNS.md` (existing tech debt)
- `.planning/PROJECT.md` (constraints, active requirements)
