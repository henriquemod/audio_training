# train_audio_model

Local voice cloning on RTX 4090. Trains an RVC v2 model of your English-speaking voice and generates audio from text prompts via a simple CLI.

## What this does

Pipeline: `text → Edge-TTS (English voice) → RVC v2 (your trained voice) → .wav`

- **Training** happens once via RVC's official WebUI.
- **Daily use** is a CLI: `python src/generate.py "Hello world"` produces a wav in your voice.
- **Reliability** is a design goal: every dependency is verified before anything runs, every error is actionable.

## Requirements

- Linux (tested on Ubuntu/WSL2)
- RTX 4090 or equivalent NVIDIA GPU with CUDA 12.1+ drivers
- [mise](https://mise.jdx.dev) for Python version management
- `ffmpeg` >= 5.0 with `afftdn`, `loudnorm`, `silencedetect` filters
- `git`
- ~5 GB disk for RVC weights and dependencies

Verify everything at once:
```bash
.venv/bin/python src/doctor.py --system-only
```

## Quickstart

```bash
# 1. Install Python 3.10 via mise (reads .mise.toml)
cd train_audio_model
mise trust .mise.toml
mise install

# 2. Create the root venv and install deps
mise exec python@3.10 -- python -m venv .venv
.venv/bin/pip install -e ".[dev]"

# 3. Verify system dependencies
.venv/bin/python src/doctor.py --system-only

# 4. Install RVC (clones, creates its own venv, downloads weights)
./scripts/setup_rvc.sh

# 5. Put your raw recordings in dataset/raw/
#    30-60 minutes of clean speech in a quiet room with a decent mic.

# 6. Preprocess
.venv/bin/python src/preprocess.py

# 7. Train via RVC WebUI (see "Train your voice model" below)
./scripts/launch_rvc_webui.sh
#    ... train in the browser, then:
./scripts/install_model.sh myvoice_v1

# 8. Configure default model
cp .env.example .env
# edit .env so DEFAULT_MODEL=myvoice_v1

# 9. Generate audio
.venv/bin/python src/generate.py "Hello, this is my cloned voice."
```

## Train your voice model

### Record your audio

- 30-60 minutes total. More is better *if* quality stays high.
- Read naturally, varied prosody. A book chapter or a long article is ideal.
- Quiet room, consistent mic distance, no background music.
- Save as WAV, FLAC, or MP3 in `dataset/raw/`.

### Run preprocessing

```bash
.venv/bin/python src/preprocess.py
```

This converts everything to 44.1 kHz mono, applies mild denoising, normalizes loudness to -20 LUFS, and slices into 3-15 second clips into `dataset/processed/`. Idempotent — safe to re-run with different settings.

### Train in RVC WebUI

```bash
./scripts/launch_rvc_webui.sh
```

Open http://localhost:7865 and in the **Train** tab:

| Field | Value |
|---|---|
| Experiment name | `myvoice_v1` (convention: `<name>_v<N>`) |
| Dataset path | absolute path to `dataset/processed/` |
| Target sample rate | `48k` |
| f0 extraction method | `rmvpe` |
| Total training epochs | `500` |
| Batch size per GPU | `12` (drop to `8` if VRAM spikes) |
| Save frequency | `50` |

Click **Process data → Feature extraction → Train model → Train feature index**, in that order. Expected total time on a 4090: 30-90 minutes for ~30 min of audio. Stop when loss plateaus around 0.3-0.4.

### Install the trained model

```bash
./scripts/install_model.sh myvoice_v1
```

Copies `rvc/assets/weights/myvoice_v1.pth` and the feature index into `models/`.

## Generate audio

Basic:
```bash
.venv/bin/python src/generate.py "Hello, world."
```

With options:
```bash
.venv/bin/python src/generate.py "Long text here" \
  --out output/greeting.wav \
  --model myvoice_v1 \
  --tts-voice en-US-GuyNeural \
  --pitch 0 \
  --index-rate 0.7
```

From a file:
```bash
.venv/bin/python src/generate.py --text-file my_script.txt --out output/script.wav
```

Smoke test (verifies the whole pipeline works):
```bash
.venv/bin/python src/generate.py --smoke-test
```

List available Edge-TTS voices:
```bash
.venv/bin/python src/generate.py --list-voices
```

## Troubleshooting

All scripts exit with specific codes:

- **Exit 1 — config/setup error.** Something isn't installed or a path is wrong. Run `.venv/bin/python src/doctor.py` for the full picture.
- **Exit 2 — user input error.** Empty text, bad flag combination, voice name doesn't exist. The error message tells you what to fix.
- **Exit 3 — runtime error.** ffmpeg, Edge-TTS, or RVC subprocess failed. The last 20 lines of subprocess stderr are always printed.

### Common failures

| Symptom | Cause | Fix |
|---|---|---|
| `doctor: Python version FAIL` | Wrong Python active | `mise install python@3.10 && mise use python@3.10` |
| `doctor: ffmpeg filters FAIL (missing afftdn)` | Minimal ffmpeg build | `sudo apt install ffmpeg` (full package) |
| `setup_rvc.sh: CUDA not available in rvc/.venv` | Wrong torch wheel | Rerun with `./scripts/setup_rvc.sh --force` |
| `generate.py: model myvoice_v1 not found` | Trained model not installed | `./scripts/install_model.sh myvoice_v1` |
| `CUDA out of memory` during inference | Other GPU apps running | Close them, or try `--pitch 0` (default already) |
| `Edge-TTS failed: ... 403` | Microsoft rotated their token; edge-tts 6.1.12 pin is stale | Upgrade: `.venv/bin/pip install -U edge-tts` |

## Architecture

- **Two Python environments**: `./.venv` for our CLI (edge-tts, typer), `./rvc/.venv` for RVC (torch, fairseq). They never share imports. The only cross-venv call is `src/generate.py` shelling out to `rvc/.venv/bin/python tools/infer_cli.py`.
- **RVC is pinned** to commit `7ef19867780c` (2024-11-24) inside `scripts/setup_rvc.sh`. Not a submodule — just a pinned clone for simplicity.
- **slicer2.py is vendored** at `src/slicer2.py` (MIT, from `audio-slicer` 1.0.1) to avoid pulling in librosa/pydub and to eliminate a top-level `src` package collision on install.
- **Pipeline diagram:**
  ```
  text -> Edge-TTS -> mp3 -> ffmpeg -> wav -> RVC (subprocess) -> final.wav
  ```

## Development

```bash
# Run the full check (doctor + tests + lint)
./scripts/check.sh

# Unit tests only
.venv/bin/pytest tests/unit -v

# Integration tests (need ffmpeg, skip network by default)
.venv/bin/pytest tests/integration -v

# Include network tests
.venv/bin/pytest -m network

# Auto-format
.venv/bin/ruff format src/ tests/
```

## License

MIT for our code. RVC is licensed separately by its upstream. `src/slicer2.py` is MIT, vendored from [audio-slicer](https://github.com/ai-forks/audio-slicer).
