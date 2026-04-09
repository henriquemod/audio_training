# External Integrations

**Analysis Date:** 2026-04-09

## APIs & External Services

**Text-to-Speech (stage 1 of the generation pipeline):**
- Microsoft Edge TTS (via reverse-engineered public endpoint) - produces the intermediate voice used as RVC's input source.
  - SDK/Client: `edge-tts>=7.0,<8` (pinned to 7.x band in `pyproject.toml`).
  - Auth: none (anonymous public endpoint). The client handles a rotating token internally; when Microsoft rotates, calls return 403 until the client is upgraded. Documented as a known failure mode in `README.md`.
  - Entry points:
    - `src/generate.py:_generate_edge_tts` - `edge_tts.Communicate(text, voice).save(mp3_path)`.
    - `src/generate.py:_list_english_voices` - `edge_tts.list_voices()` filtered to `en-*` locales, rendered via `rich.Table` for `--list-voices`.
  - Network required. Integration-tested in `tests/integration/test_edge_tts.py` under the `network` pytest marker.

## External Subprocess Integrations

**RVC (Retrieval-based Voice Conversion WebUI):**
- Upstream: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
- Pinned commit: `7ef19867780cf703841ebafb565a4e47d1ea86ff` (2024-11-24), declared in `scripts/setup_rvc.sh` as `RVC_COMMIT`.
- Not a git submodule. Cloned by `scripts/setup_rvc.sh` into `./rvc/` and git-ignored.
- Integration surface (shell-invoked only; no Python imports across venvs):
  - **Inference**: `src/generate.py:build_rvc_subprocess_cmd` builds `["$RVC_VENV_PYTHON", "tools/infer_cli.py", ...]` and `subprocess.run(cmd, cwd=RVC_DIR, ...)`. Args passed: `--input_path`, `--index_path`, `--f0method rmvpe`, `--opt_path`, `--model_name`, `--index_rate`, `--device`, `--is_half True`, `--filter_radius`, `--resample_sr 0`, `--rms_mix_rate 1`, `--protect 0.33`, `--f0up_key`.
  - **Weight staging**: `src/generate.py:_ensure_rvc_weight_staged` copies `models/<name>.pth` to `rvc/assets/weights/<name>.pth` (`infer_cli.py` reads from there).
  - **WebUI launch**: `scripts/launch_rvc_webui.sh` execs `rvc/.venv/bin/python infer-web.py --pycmd <rvc_py> --port 7865` for training. Opens at http://localhost:7865.
  - **Weight download**: `scripts/setup_rvc.sh` calls `rvc/.venv/bin/python tools/download_models.py` to fetch `hubert_base.pt` and `rmvpe.pt`.
  - **Model installation**: `scripts/install_model.sh <name>` copies `rvc/assets/weights/<name>.pth` and `rvc/logs/<name>/added_*.index` into `./models/`.

**ffmpeg (system binary):**
- All invocations routed through `src/ffmpeg_utils.py:run_ffmpeg` - never called directly, never with `shell=True`.
- Flags always prepended: `-hide_banner -loglevel error -y`.
- Callers:
  - `src/preprocess.py` - three-stage chain via `build_canonical_args`, `build_denoise_args`, `build_loudnorm_args`. Filter expressions: `highpass=f=75,lowpass=f=15000,afftdn=nr=12` and `loudnorm=I={target_lufs}:TP=-1:LRA=11`.
  - `src/generate.py` - mp3-to-canonical wav conversion in stage 2 (`-ar 44100 -ac 1 -sample_fmt s16`).
- Version and filter presence verified by `src/doctor.py:check_ffmpeg` and `check_ffmpeg_filters`.

## Data Storage

**Databases:**
- None. No SQL, no KV store, no schema.

**File Storage (all local):**
- `dataset/raw/` - user's source recordings (`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.aac`). Git-ignored.
- `dataset/processed/` - preprocessed 3-15s clips, 44.1 kHz mono 16-bit PCM WAV. Wiped and rebuilt on every `preprocess.py` run. Git-ignored.
- `models/<name>.pth` + `models/<name>.index` - installed RVC weights and feature indexes. Git-ignored.
- `output/` - generated WAVs, auto-named `<YYYYMMDD_HHMMSS>_<slug>.wav`. Git-ignored.
- `rvc/assets/weights/`, `rvc/assets/hubert/`, `rvc/assets/rmvpe/`, `rvc/logs/<name>/` - RVC-owned storage. Only touched via the install and generate helpers.

**Caching:**
- None beyond Python bytecode (`__pycache__`) and tool caches (`.pytest_cache`, `.ruff_cache`).

## Authentication & Identity

- No user accounts. No auth. This is a local CLI.
- Edge-TTS reaches out to Microsoft with no caller-owned credentials.

## Monitoring & Observability

**Error Tracking:**
- None. Failures surface as non-zero exit codes with rich-formatted stderr messages.

**Logs:**
- `rich.Console` to stdout for progress.
- `scripts/setup_rvc.log` - appended by `scripts/setup_rvc.sh` via a re-exec + `tee -a` pattern that preserves `set -e`. Git-ignored.
- Exit code convention defined in `src/generate.py` module docstring and echoed in `README.md`:
  - `0` success
  - `1` config/setup error
  - `2` user input error
  - `3` runtime error (subprocess failure)

## CI/CD & Deployment

**Hosting:** None - local developer tool only.

**CI Pipeline:** None. `scripts/check.sh` is the local equivalent: doctor, unit tests, integration tests (skip `network`), `ruff check`, `ruff format --check`.

## Environment Configuration

**Required env vars:** All optional; defaults exist in `src/generate.py`.
- `DEFAULT_MODEL` (default `myvoice_v1`)
- `DEFAULT_EDGE_VOICE` (default `en-US-GuyNeural`)
- `DEVICE` (default `cuda:0`)

**Secrets location:** There are no secrets. `.env` is git-ignored defensively but contains only user preferences. `.env.example` is committed as the template.

## Webhooks & Callbacks

- None. No incoming HTTP surface. Only outbound egress is Edge-TTS (from `edge-tts` client) and https fetches during `setup_rvc.sh` (pip, GitHub, PyTorch wheel index, RVC's `download_models.py`).

---

*Integration audit: 2026-04-09*
