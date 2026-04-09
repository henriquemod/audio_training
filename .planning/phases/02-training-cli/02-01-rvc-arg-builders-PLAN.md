---
phase: 02-training-cli
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/train.py
  - tests/unit/test_train.py
autonomous: true
requirements: [TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04, TRAIN-05, TRAIN-13, TRAIN-14]
tags: [training, rvc, cli, builders]

must_haves:
  truths:
    - "src/train.py exposes four pure arg-builder functions for the four RVC training stages"
    - "Every builder returns list[str] with byte-exact argv matching the pinned RVC commit"
    - "_write_filelist produces a format-equivalent filelist accepted by RVC's train.py"
    - "_write_exp_config copies rvc/configs/{v1,v2}/<sr>.json to rvc/logs/<exp>/config.json"
    - "Same --sample-rate value drives BOTH preprocess (Hz int) and train (-sr 40k string) builders"
    - "All builder unit tests pass under the default pytest filter without GPU or rvc/.venv"
  artifacts:
    - path: "src/train.py"
      provides: "Module skeleton, constants, four pure arg-builders, filelist/config helpers, preset resolver, pretrained resolver"
      contains: "def build_rvc_preprocess_cmd"
    - path: "tests/unit/test_train.py"
      provides: "Argv-assertion tests, filelist tests, sentinel probe tests, two-venv import-ban test, sample-rate flow test"
      contains: "def test_build_rvc_preprocess_cmd"
  key_links:
    - from: "tests/unit/test_train.py"
      to: "src/train.py"
      via: "from src.train import build_rvc_preprocess_cmd, build_rvc_extract_f0_cmd, build_rvc_extract_feature_cmd, build_rvc_train_cmd, _write_filelist, _write_exp_config, SR_STR_MAP, PRESETS, resolve_preset, resolve_pretrained_paths, count_dataset_inputs, stage_1_is_done, stage_2_is_done, stage_3_is_done, stage_4_is_done"
      pattern: "from src.train import"
---

<objective>
Create the pure-function backbone of `src/train.py`: module skeleton, constants, four arg-builders, filelist + config.json helpers, preset resolver, pretrained resolver, and sentinel probes — plus a complete `tests/unit/test_train.py` asserting byte-exact argv against the pinned RVC commit. Zero subprocess execution. Zero GPU. Zero `rvc/.venv` access.

Purpose: Lock the RVC contract in code via tests so Plans 02 and 03 can wire the orchestrator and doctor pre-flight against a verified surface. Closes the STATE.md pitfalls "missing -pg/-pd → silent random init" (D-21) and "sample-rate chain mismatch" (D-23) at the unit-test layer.

Output: `src/train.py` (~400 LOC, no `import torch`/`fairseq`/`faiss`/`rvc`), `tests/unit/test_train.py` (~25 tests, all green under `python -m pytest tests/unit/test_train.py -x`).
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/phases/02-training-cli/02-CONTEXT.md
@.planning/phases/02-training-cli/02-RESEARCH.md
@CLAUDE.md
@src/generate.py
@src/doctor.py
@src/preprocess.py

<interfaces>
<!-- Source-of-truth contracts the executor MUST replicate. All argv strings below are verbatim from RESEARCH.md §2 / §10 (verified against pinned RVC commit 7ef19867780cf703841ebafb565a4e47d1ea86ff). -->

From src/doctor.py (already exists, import these):
```python
PROJECT_ROOT: Path  # = Path(__file__).resolve().parent.parent
RVC_DIR: Path       # = PROJECT_ROOT / "rvc"
RVC_VENV_PYTHON: Path  # = RVC_DIR / ".venv" / "bin" / "python"
@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    fix_hint: str = ""
```

From src/preprocess.py (already exists, import this):
```python
AUDIO_EXTS: tuple[str, ...]  # tuple of valid audio file extensions, e.g. (".wav", ".mp3", ".flac", ...)
```

Module-level constants to declare in src/train.py:
```python
SR_STR_MAP: dict[int, str] = {32000: "32k", 40000: "40k", 48000: "48k"}
VALID_F0_METHODS: tuple[str, ...] = ("pm", "harvest", "rmvpe", "rmvpe_gpu")
VALID_VERSIONS: tuple[str, ...] = ("v1", "v2")
VALID_SAMPLE_RATES: tuple[int, ...] = (32000, 40000, 48000)
VALID_PRESETS: tuple[str, ...] = ("smoke", "low", "balanced", "high")
PRESETS: dict[str, dict[str, int]] = {
    "smoke":    {"epochs": 1,   "batch_size": 1,  "save_every": 1},
    "low":      {"epochs": 200, "batch_size": 6,  "save_every": 50},
    "balanced": {"epochs": 300, "batch_size": 12, "save_every": 50},
    "high":     {"epochs": 500, "batch_size": 40, "save_every": 50},
}
SUBPROCESS_EXTRA_ENV: dict[str, str] = {
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "LANG": "C.UTF-8",
}
TRAIN_SUCCESS_EXIT_CODES: tuple[int, ...] = (0, 61)
WEIGHT_FILE_FLOOR_BYTES: int = 1024
PRETRAINED_MIN_BYTES: int = 30_000_000
EXPERIMENT_NAME_RE: str = r"^[a-zA-Z0-9_-]{1,64}$"
DEFAULT_NUM_PROCS: int  # = min(os.cpu_count() or 1, 8)
STAGE_BANNER: str = "===== Stage {n}: {name} (started {ts}) ====="
```

EXACT argv for `build_rvc_preprocess_cmd` (RESEARCH.md §2.1):
```python
[
    str(rvc_python),
    "infer/modules/train/preprocess.py",
    str(dataset_dir.resolve()),
    str(sample_rate),                                  # "40000" not "40k"
    str(num_procs),
    str((RVC_DIR / "logs" / exp_name).resolve()),
    "False",                                            # noparallel
    "3.7",                                              # preprocess_per
]
```

EXACT argv for `build_rvc_extract_f0_cmd` Branch A (f0_method in {"pm","harvest","rmvpe"}):
```python
[
    str(rvc_python),
    "infer/modules/train/extract/extract_f0_print.py",
    str((RVC_DIR / "logs" / exp_name).resolve()),
    str(num_procs),
    f0_method,
]
```

EXACT argv for `build_rvc_extract_f0_cmd` Branch B (f0_method == "rmvpe_gpu"):
```python
[
    str(rvc_python),
    "infer/modules/train/extract/extract_f0_rmvpe.py",
    "1",                                                # leng
    "0",                                                # idx
    gpu_id,                                             # n_g, default "0"
    str((RVC_DIR / "logs" / exp_name).resolve()),
    "True" if is_half else "False",
]
```

EXACT argv for `build_rvc_extract_feature_cmd` (single-GPU):
```python
[
    str(rvc_python),
    "infer/modules/train/extract_feature_print.py",
    device,                                             # default "cuda:0"
    "1",                                                # leng
    "0",                                                # idx
    gpu_id,                                             # n_g, default "0"
    str((RVC_DIR / "logs" / exp_name).resolve()),
    version,                                            # "v1" or "v2"
    "True" if is_half else "False",
]
```

EXACT argv for `build_rvc_train_cmd` (RESEARCH.md §10.2, verified against rvc/infer-web.py:571-589):
```python
[
    str(rvc_python),
    "infer/modules/train/train.py",
    "-e", exp_name,                                     # bare name, NOT a path
    "-sr", SR_STR_MAP[sample_rate],                     # "40k" not "40000"
    "-f0", "1" if if_f0 else "0",
    "-bs", str(batch_size),
    "-g", gpus,                                         # default "0"
    "-te", str(epochs),
    "-se", str(save_every),
    "-pg", str(pretrained_g.resolve()),                 # D-21: ALWAYS absolute, ALWAYS passed
    "-pd", str(pretrained_d.resolve()),                 # D-21: ALWAYS absolute, ALWAYS passed
    "-l", "0",
    "-c", "0",
    "-sw", "0",
    "-v", version,
]
```

resolve_pretrained_paths semantics (RESEARCH.md §4.7):
```python
# version="v2" → subdir "pretrained_v2"; version="v1" → subdir "pretrained"
# prefix = "f0" if if_f0 else ""
# G path: RVC_DIR / "assets" / sub / f"{prefix}G{sr_str}.pth"
# D path: RVC_DIR / "assets" / sub / f"{prefix}D{sr_str}.pth"
# Examples:
#   v2 + 40000 + if_f0=True  → rvc/assets/pretrained_v2/f0G40k.pth, .../f0D40k.pth
#   v1 + 48000 + if_f0=False → rvc/assets/pretrained/G48k.pth,    .../D48k.pth
```

resolve_preset semantics (D-01/D-02/D-03):
```python
def resolve_preset(name: str, *, epochs: Optional[int], batch_size: Optional[int], save_every: Optional[int]) -> dict[str, int]:
    resolved = dict(PRESETS[name])
    if epochs is not None:     resolved["epochs"] = epochs
    if batch_size is not None: resolved["batch_size"] = batch_size
    if save_every is not None: resolved["save_every"] = save_every
    return resolved
```

count_dataset_inputs / sentinel probe semantics (D-08, D-09, RESEARCH.md §5):
```python
def count_dataset_inputs(dataset_dir: Path) -> int:
    """Return count of files in dataset_dir (non-recursive) whose suffix.lower() is in AUDIO_EXTS."""

def stage_1_is_done(exp_dir: Path, expected: int) -> bool:
    """True iff len(list((exp_dir / '0_gt_wavs').glob('*.wav'))) >= expected (and expected > 0)."""

def stage_2_is_done(exp_dir: Path, expected: int) -> bool:
    """True iff count(2a_f0/*.npy) >= expected AND count(2b-f0nsf/*.npy) >= expected."""

def stage_3_is_done(exp_dir: Path, expected: int, version: str) -> bool:
    """True iff count(3_feature768/*.npy) >= expected for v2, or 3_feature256/*.npy for v1."""

def stage_4_is_done(weight_path: Path) -> bool:
    """True iff weight_path exists and stat().st_size >= WEIGHT_FILE_FLOOR_BYTES (1024)."""
```

_write_filelist semantics (D-06, D-07, RESEARCH.md §2.5.1):
```python
# Path: exp_dir / "filelist.txt"
# Per-clip row format (with f0, 5 fields, pipe-separated):
#   <gt_wavs_abs>/<stem>.wav|<feature_dir_abs>/<stem>.npy|<f0_dir_abs>/<stem>.wav.npy|<f0nsf_dir_abs>/<stem>.wav.npy|0
# gt_wavs_abs   = (exp_dir / "0_gt_wavs").resolve()
# feature_dir   = "3_feature768" (v2) or "3_feature256" (v1) under exp_dir, resolved
# f0_dir        = (exp_dir / "2a_f0").resolve()
# f0nsf_dir     = (exp_dir / "2b-f0nsf").resolve()
# Stems = intersection of stems present in all 4 dirs (use set intersection by os.path.splitext)
# Mute-reference rows (D-07 deviates from webui's hardcoded range(2)):
#   For each file glob `(RVC_DIR / "logs" / "mute" / "0_gt_wavs").glob(f"*{sr_str}*.wav")` (or *.wav if none match)
#   Append one row per file using template:
#     {RVC_DIR}/logs/mute/0_gt_wavs/mute{sr2}.wav|{RVC_DIR}/logs/mute/3_feature{fea_dim}/mute.npy|{RVC_DIR}/logs/mute/2a_f0/mute.wav.npy|{RVC_DIR}/logs/mute/2b-f0nsf/mute.wav.npy|0
#   where sr2 = SR_STR_MAP[sample_rate], fea_dim = 768 (v2) or 256 (v1)
# Shuffle lines before writing (use random.Random(seed=0).shuffle for deterministic tests, or just write unsorted)
# Assert resulting file is non-empty (line count >= 1) before return; raise RuntimeError if empty.
# Returns: the Path written.
```

_write_exp_config semantics (RESEARCH.md §2.5.2, Open Question 1 — RESOLVED):
```python
# Source: RVC_DIR / "configs" / ("v1" if version == "v1" or sr_str == "40k" else "v2") / f"{sr_str}.json"
# Wait — re-read RESEARCH.md §2.5.2 carefully: "version == 'v1' OR sr == '40k' → configs/v1/<sr>.json; else → configs/v2/<sr>.json"
# Use shutil.copy2 to write to: exp_dir / "config.json"
# Returns: the destination Path.
# Raises FileNotFoundError (with clear context) if source missing.
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Scaffold src/train.py with constants, resolvers, and pure helpers</name>
  <files>src/train.py</files>
  <read_first>
    - src/generate.py (the WHOLE file — copy module-docstring shape, sys.path fixup pattern, _PROJECT_ROOT idiom, exit-code docstring style, typer.Typer + Console + Table imports, the _tail helper signature for reference)
    - src/doctor.py (lines 1-50 — confirm exact import names: PROJECT_ROOT, RVC_DIR, RVC_VENV_PYTHON, CheckResult)
    - src/preprocess.py (find AUDIO_EXTS — confirm it is a tuple of lowercase extensions including the leading dot)
    - .planning/phases/02-training-cli/02-RESEARCH.md §2 (Stage Reference table, all argv blocks)
    - .planning/phases/02-training-cli/02-RESEARCH.md §4 (Module Layout — verbatim function signatures)
    - .planning/phases/02-training-cli/02-RESEARCH.md §10.2 (verified train_cmd template)
    - .planning/phases/02-training-cli/02-CONTEXT.md (D-01..D-24)
  </read_first>
  <behavior>
    - SR_STR_MAP[40000] == "40k"; SR_STR_MAP[32000] == "32k"; SR_STR_MAP[48000] == "48k"
    - VALID_F0_METHODS == ("pm", "harvest", "rmvpe", "rmvpe_gpu") — exactly four, dio EXCLUDED (Open Question 5 — research recommendation)
    - VALID_VERSIONS == ("v1", "v2"); VALID_PRESETS == ("smoke","low","balanced","high"); VALID_SAMPLE_RATES == (32000,40000,48000)
    - PRESETS["smoke"] == {"epochs":1,"batch_size":1,"save_every":1}
    - PRESETS["low"] == {"epochs":200,"batch_size":6,"save_every":50}
    - PRESETS["balanced"] == {"epochs":300,"batch_size":12,"save_every":50}
    - PRESETS["high"] == {"epochs":500,"batch_size":40,"save_every":50}
    - resolve_preset("high", epochs=800, batch_size=None, save_every=None) == {"epochs":800,"batch_size":40,"save_every":50}
    - resolve_pretrained_paths(sample_rate=40000, version="v2", if_f0=True) returns (RVC_DIR/"assets/pretrained_v2/f0G40k.pth", .../f0D40k.pth) — exact paths
    - resolve_pretrained_paths(sample_rate=48000, version="v1", if_f0=False) returns (RVC_DIR/"assets/pretrained/G48k.pth", .../D48k.pth)
    - count_dataset_inputs(empty_dir) == 0; with 5 .wav files returns 5; ignores subdirs; ignores files whose suffix.lower() is not in AUDIO_EXTS
    - stage_1_is_done(exp, 5) is True when 0_gt_wavs/ has 5 .wav files; False when 4
    - stage_2_is_done returns True only when BOTH 2a_f0/ and 2b-f0nsf/ have >= expected .npy files
    - stage_3_is_done with version="v2" probes 3_feature768/; with version="v1" probes 3_feature256/
    - stage_4_is_done(missing_path) is False; with file of 1023 bytes False; with file of 1024 bytes True
    - SUBPROCESS_EXTRA_ENV == {"TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "LANG": "C.UTF-8"}
    - TRAIN_SUCCESS_EXIT_CODES == (0, 61)
    - DEFAULT_NUM_PROCS == min(os.cpu_count() or 1, 8)
  </behavior>
  <action>
Create `src/train.py` with this exact top-of-file structure (copy idiom from src/generate.py lines 1-55):

```python
"""RVC training pipeline CLI: preprocess -> extract_f0 -> extract_feature -> train.

Orchestrates all four RVC training stages end-to-end as subprocesses into rvc/.venv,
with always-on intrinsic probe-and-skip resume.

Two-venv boundary: this module MUST NOT import torch, fairseq, faiss, or rvc.
All RVC interaction is via subprocess.Popen(..., cwd=RVC_DIR, shell=False).

Exit codes:
  0  success (incl. raw exit 61 from RVC's os._exit(2333333))
  1  config / setup error (missing weights, missing dataset, failing doctor pre-flight)
  2  user input error (bad CLI flag combinations, bad preset name, invalid experiment name)
  3  runtime error (Stage N subprocess non-{0,61} exit, missing output file)
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

# Allow `python src/train.py ...` invocation: put project root on sys.path
# before importing src.* (mirrors src/generate.py).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.doctor import (  # noqa: E402
    PROJECT_ROOT,
    RVC_DIR,
    RVC_VENV_PYTHON,
)
from src.preprocess import AUDIO_EXTS  # noqa: E402
```

Then declare ALL constants from the `<interfaces>` block above (verbatim values). DEFAULT_NUM_PROCS computed at module load: `DEFAULT_NUM_PROCS = min(os.cpu_count() or 1, 8)`.

Then implement these PURE helpers (no I/O except where explicitly noted):

1. `def resolve_preset(name: str, *, epochs: Optional[int], batch_size: Optional[int], save_every: Optional[int]) -> dict[str, int]`
   - Body: `resolved = dict(PRESETS[name]); if epochs is not None: resolved["epochs"] = epochs; ...; return resolved`
   - Raises KeyError if name not in PRESETS (callers validate first).

2. `def resolve_pretrained_paths(*, sample_rate: int, version: str, if_f0: bool) -> tuple[Path, Path]`
   - `sr_str = SR_STR_MAP[sample_rate]`
   - `sub = "pretrained_v2" if version == "v2" else "pretrained"`
   - `prefix = "f0" if if_f0 else ""`
   - `g = RVC_DIR / "assets" / sub / f"{prefix}G{sr_str}.pth"`
   - `d = RVC_DIR / "assets" / sub / f"{prefix}D{sr_str}.pth"`
   - `return (g, d)`

3. `def count_dataset_inputs(dataset_dir: Path) -> int`
   - `return sum(1 for p in dataset_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS)`

4. `def stage_1_is_done(exp_dir: Path, expected: int) -> bool`
   - `if expected <= 0: return False`
   - `return len(list((exp_dir / "0_gt_wavs").glob("*.wav"))) >= expected`
   - Wrap directory access in try/except FileNotFoundError → return False

5. `def stage_2_is_done(exp_dir: Path, expected: int) -> bool`
   - Both `2a_f0/*.npy` and `2b-f0nsf/*.npy` must have >= expected; missing dir → False

6. `def stage_3_is_done(exp_dir: Path, expected: int, version: str) -> bool`
   - `feat_dir = exp_dir / ("3_feature768" if version == "v2" else "3_feature256")`
   - count `*.npy` >= expected; missing → False

7. `def stage_4_is_done(weight_path: Path) -> bool`
   - `try: return weight_path.exists() and weight_path.stat().st_size >= WEIGHT_FILE_FLOOR_BYTES`
   - `except OSError: return False`

8. `def validate_experiment_name(name: str) -> bool`
   - `return bool(re.match(EXPERIMENT_NAME_RE, name))`

Add Google-style docstrings on every function. Use `from __future__ import annotations`. Line length 100. Use `Optional[X]` (typer 0.12.3 quirk). DO NOT add any `import torch`, `import fairseq`, `import faiss`, `import rvc`.

DO NOT define the four `build_*_cmd` functions in this task — Task 2 adds them.
DO NOT define `_write_filelist` or `_write_exp_config` — Task 2 adds them.
DO NOT add typer CLI / `main()` — Plan 02 adds it.
DO NOT add subprocess runner — Plan 03 adds it.
  </action>
  <verify>
    <automated>cd /home/henrique/Development/train_audio_model && .venv/bin/python -c "from src.train import SR_STR_MAP, PRESETS, VALID_F0_METHODS, VALID_VERSIONS, VALID_SAMPLE_RATES, VALID_PRESETS, SUBPROCESS_EXTRA_ENV, TRAIN_SUCCESS_EXIT_CODES, WEIGHT_FILE_FLOOR_BYTES, DEFAULT_NUM_PROCS, EXPERIMENT_NAME_RE, resolve_preset, resolve_pretrained_paths, count_dataset_inputs, stage_1_is_done, stage_2_is_done, stage_3_is_done, stage_4_is_done, validate_experiment_name; assert SR_STR_MAP == {32000:'40k'.replace('40','32'),40000:'40k',48000:'48k'} or SR_STR_MAP == {32000:'32k',40000:'40k',48000:'48k'}; assert PRESETS['balanced']['epochs'] == 300; assert TRAIN_SUCCESS_EXIT_CODES == (0,61); assert resolve_preset('high', epochs=800, batch_size=None, save_every=None) == {'epochs':800,'batch_size':40,'save_every':50}; print('OK')"</automated>
  </verify>
  <acceptance_criteria>
    - File `src/train.py` exists
    - `grep -c "import torch\|import fairseq\|import faiss\|^import rvc\|from torch\|from fairseq\|from faiss\|from rvc" src/train.py` returns 0
    - `grep -q "SR_STR_MAP: dict\[int, str\] = {32000: \"32k\", 40000: \"40k\", 48000: \"48k\"}" src/train.py` succeeds
    - `grep -q "PRESETS" src/train.py` and `grep -q '"smoke":' src/train.py` and `grep -q '"balanced"' src/train.py`
    - `grep -q "TRAIN_SUCCESS_EXIT_CODES" src/train.py` succeeds
    - `grep -q "def resolve_preset" src/train.py` and `grep -q "def resolve_pretrained_paths" src/train.py`
    - `grep -q "def count_dataset_inputs\|def stage_1_is_done\|def stage_4_is_done" src/train.py` (all four sentinel probes present)
    - `grep -q "def validate_experiment_name" src/train.py`
    - `grep -q 'EXPERIMENT_NAME_RE.*\^\[a-zA-Z0-9_-\]{1,64}\$' src/train.py`
    - `.venv/bin/python -c "import src.train"` exits 0
    - `.venv/bin/ruff check src/train.py` exits 0
  </acceptance_criteria>
  <done>Module imports cleanly, all constants and pure helpers exist with correct values, no rvc-stack imports.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Add four pure arg-builders + filelist + config helpers to src/train.py</name>
  <files>src/train.py</files>
  <read_first>
    - src/train.py (the file you just created in Task 1 — extend it, do not rewrite)
    - .planning/phases/02-training-cli/02-RESEARCH.md §2 (every Stage subsection — Stage 1 at lines 155-192, Stage 2 at 193-233, Stage 3 at 234-265, Pre-Stage 4 at 266-303, Stage 4 at 317-384)
    - .planning/phases/02-training-cli/02-RESEARCH.md §10.2 (build_rvc_train_cmd verbatim)
    - .planning/phases/02-training-cli/02-RESEARCH.md §4.4 (filelist + config helper signatures)
    - The `<interfaces>` block in this PLAN.md (every argv block is canonical — copy them verbatim)
  </read_first>
  <behavior>
    - build_rvc_preprocess_cmd produces argv exactly: [str(rvc_python), "infer/modules/train/preprocess.py", str(dataset_dir.resolve()), str(sample_rate), str(num_procs), str((RVC_DIR/"logs"/exp_name).resolve()), "False", "3.7"]
    - build_rvc_extract_f0_cmd with f0_method="rmvpe" produces argv exactly: [str(rvc_python), "infer/modules/train/extract/extract_f0_print.py", str((RVC_DIR/"logs"/exp_name).resolve()), str(num_procs), "rmvpe"]
    - build_rvc_extract_f0_cmd with f0_method="rmvpe_gpu" produces argv exactly: [str(rvc_python), "infer/modules/train/extract/extract_f0_rmvpe.py", "1", "0", "0", str((RVC_DIR/"logs"/exp_name).resolve()), "True"]
    - build_rvc_extract_feature_cmd produces 9 args: [str(rvc_python), "infer/modules/train/extract_feature_print.py", "cuda:0", "1", "0", "0", str((RVC_DIR/"logs"/exp_name).resolve()), version, "True"]
    - build_rvc_train_cmd produces argv with -e bare exp_name, -sr SR_STR_MAP[sample_rate], -pg/-pd ABSOLUTE pretrained paths, -l "0", -c "0", -sw "0", -v version (full template in <interfaces>)
    - All builders are pure (no I/O, no global mutation); calling twice with same args returns equal lists
    - _write_filelist creates exp_dir/filelist.txt with at least one row per intersected stem and at least one mute-reference row; 5-field rows when if_f0=True (sid="0" as last field); 3-field rows when if_f0=False; raises RuntimeError if zero clip stems intersect AND zero mute files found
    - _write_exp_config copies rvc/configs/v1/40k.json (v2+40k case) or rvc/configs/v2/<sr_str>.json to exp_dir/config.json; raises FileNotFoundError with explicit source path if missing
  </behavior>
  <action>
Append to `src/train.py` (after the helpers from Task 1, before any future CLI code).

### 1. The four arg-builders (use the EXACT argv blocks from `<interfaces>` above)

```python
def build_rvc_preprocess_cmd(
    *,
    rvc_python: Path,
    dataset_dir: Path,
    sample_rate: int,
    num_procs: int,
    exp_name: str,
) -> list[str]:
    """Build argv for RVC Stage 1 (preprocess.py).

    Mirrors rvc/infer-web.py:218-254 (preprocess_dataset). Pure function.
    sample_rate is the integer Hz value (e.g. 40000), NOT the "40k" string.
    """
    return [
        str(rvc_python),
        "infer/modules/train/preprocess.py",
        str(dataset_dir.resolve()),
        str(sample_rate),
        str(num_procs),
        str((RVC_DIR / "logs" / exp_name).resolve()),
        "False",
        "3.7",
    ]


def build_rvc_extract_f0_cmd(
    *,
    rvc_python: Path,
    exp_name: str,
    num_procs: int,
    f0_method: str,
    gpu_id: str = "0",
    is_half: bool = True,
) -> list[str]:
    """Build argv for RVC Stage 2 (F0 extraction).

    Two branches per rvc/infer-web.py:258-346:
      - f0_method in {"pm","harvest","rmvpe"} -> extract_f0_print.py
      - f0_method == "rmvpe_gpu"              -> extract_f0_rmvpe.py
    """
    if f0_method == "rmvpe_gpu":
        return [
            str(rvc_python),
            "infer/modules/train/extract/extract_f0_rmvpe.py",
            "1",
            "0",
            gpu_id,
            str((RVC_DIR / "logs" / exp_name).resolve()),
            "True" if is_half else "False",
        ]
    return [
        str(rvc_python),
        "infer/modules/train/extract/extract_f0_print.py",
        str((RVC_DIR / "logs" / exp_name).resolve()),
        str(num_procs),
        f0_method,
    ]


def build_rvc_extract_feature_cmd(
    *,
    rvc_python: Path,
    exp_name: str,
    version: str,
    num_procs_per_gpu: int = 1,  # signature parity; single-GPU path uses leng=1
    device: str = "cuda:0",
    gpu_id: str = "0",
    is_half: bool = True,
) -> list[str]:
    """Build argv for RVC Stage 3 (feature extraction).

    Mirrors rvc/infer-web.py:355-395 (single-GPU branch).
    """
    return [
        str(rvc_python),
        "infer/modules/train/extract_feature_print.py",
        device,
        "1",
        "0",
        gpu_id,
        str((RVC_DIR / "logs" / exp_name).resolve()),
        version,
        "True" if is_half else "False",
    ]


def build_rvc_train_cmd(
    *,
    rvc_python: Path,
    exp_name: str,
    sample_rate: int,
    version: str,
    epochs: int,
    batch_size: int,
    save_every: int,
    f0_method: str,
    pretrained_g: Path,
    pretrained_d: Path,
    if_f0: bool = True,
    gpus: str = "0",
) -> list[str]:
    """Build argv for RVC Stage 4 (train.py).

    Mirrors rvc/infer-web.py:571-589 (click_train, gpus-present branch). D-21:
    -pg and -pd are ALWAYS passed with absolute paths to prevent silent
    random-init training when pretrained files are missing from disk.
    """
    sr_str = SR_STR_MAP[sample_rate]
    return [
        str(rvc_python),
        "infer/modules/train/train.py",
        "-e", exp_name,
        "-sr", sr_str,
        "-f0", "1" if if_f0 else "0",
        "-bs", str(batch_size),
        "-g", gpus,
        "-te", str(epochs),
        "-se", str(save_every),
        "-pg", str(pretrained_g.resolve()),
        "-pd", str(pretrained_d.resolve()),
        "-l", "0",
        "-c", "0",
        "-sw", "0",
        "-v", version,
    ]
```

### 2. `_write_filelist` and `_write_exp_config` helpers

```python
def _write_filelist(exp_dir: Path, *, version: str, sample_rate: int, if_f0: bool = True) -> Path:
    """Write rvc/logs/<exp>/filelist.txt with format-equivalent rows for RVC train.py.

    Format equivalent to (NOT byte-exact with) rvc/infer-web.py:click_train lines 500-546.
    Per-clip row (with f0): 5 pipe-separated fields, sid=0.
    Mute rows: one per file found under rvc/logs/mute/0_gt_wavs/ matching the
    requested sample rate (D-07 — deviates from webui's hardcoded range(2)).
    Raises RuntimeError if the resulting file would be empty.
    """
    sr_str = SR_STR_MAP[sample_rate]
    fea_dim = 768 if version == "v2" else 256
    feat_subdir = f"3_feature{fea_dim}"

    gt_dir = (exp_dir / "0_gt_wavs").resolve()
    feat_dir = (exp_dir / feat_subdir).resolve()
    f0_dir = (exp_dir / "2a_f0").resolve()
    f0nsf_dir = (exp_dir / "2b-f0nsf").resolve()

    def _stems(d: Path, suffix: str) -> set[str]:
        try:
            return {p.name[: -len(suffix)] for p in d.iterdir() if p.name.endswith(suffix)}
        except FileNotFoundError:
            return set()

    gt_stems = _stems(gt_dir, ".wav")
    feat_stems = _stems(feat_dir, ".npy")
    f0_stems = {n[: -len(".wav")] for n in _stems(f0_dir, ".wav.npy")}
    f0nsf_stems = {n[: -len(".wav")] for n in _stems(f0nsf_dir, ".wav.npy")}

    if if_f0:
        common = gt_stems & feat_stems & f0_stems & f0nsf_stems
    else:
        common = gt_stems & feat_stems

    lines: list[str] = []
    for stem in sorted(common):
        if if_f0:
            lines.append(
                f"{gt_dir}/{stem}.wav|{feat_dir}/{stem}.npy|"
                f"{f0_dir}/{stem}.wav.npy|{f0nsf_dir}/{stem}.wav.npy|0"
            )
        else:
            lines.append(f"{gt_dir}/{stem}.wav|{feat_dir}/{stem}.npy|0")

    # Mute rows (D-07): one per file found under rvc/logs/mute/0_gt_wavs/
    mute_root = RVC_DIR / "logs" / "mute"
    mute_gt = mute_root / "0_gt_wavs"
    mute_files: list[Path] = []
    if mute_gt.is_dir():
        # Prefer files matching the requested sample rate, otherwise any *.wav
        sr_matches = sorted(mute_gt.glob(f"*{sr_str}*.wav"))
        mute_files = sr_matches if sr_matches else sorted(mute_gt.glob("*.wav"))

    for _ in mute_files:
        if if_f0:
            lines.append(
                f"{mute_root}/0_gt_wavs/mute{sr_str}.wav|"
                f"{mute_root}/3_feature{fea_dim}/mute.npy|"
                f"{mute_root}/2a_f0/mute.wav.npy|"
                f"{mute_root}/2b-f0nsf/mute.wav.npy|0"
            )
        else:
            lines.append(
                f"{mute_root}/0_gt_wavs/mute{sr_str}.wav|"
                f"{mute_root}/3_feature{fea_dim}/mute.npy|0"
            )

    if not lines:
        raise RuntimeError(
            f"[_write_filelist] empty filelist for {exp_dir.name}: "
            f"no clip stems intersected and no mute files found at {mute_gt}"
        )

    out = exp_dir / "filelist.txt"
    out.write_text("\n".join(lines) + "\n")
    return out


def _write_exp_config(exp_dir: Path, *, version: str, sample_rate: int) -> Path:
    """Copy rvc/configs/{v1,v2}/<sr>.json to exp_dir/config.json.

    train.py opens this at startup and crashes if missing. Path resolution
    follows rvc/infer-web.py:click_train lines 555-569: v1 OR (v2+40k) -> v1 dir;
    else v2 dir. Reads from the tracked source files (not configs/inuse/, which
    is lazily populated by the webui and may not exist on a cold pod).
    """
    sr_str = SR_STR_MAP[sample_rate]
    sub = "v1" if (version == "v1" or sr_str == "40k") else "v2"
    src = RVC_DIR / "configs" / sub / f"{sr_str}.json"
    if not src.is_file():
        raise FileNotFoundError(
            f"[_write_exp_config] RVC config template missing: {src}"
        )
    dst = exp_dir / "config.json"
    exp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst
```

Use Google-style docstrings. Keep all four builders pure (zero side effects). Ruff-clean (line length 100, no UP007 violations because we already use `Optional` only where needed).
  </action>
  <verify>
    <automated>cd /home/henrique/Development/train_audio_model && .venv/bin/python -c "
from pathlib import Path
from src.train import build_rvc_preprocess_cmd, build_rvc_extract_f0_cmd, build_rvc_extract_feature_cmd, build_rvc_train_cmd
from src.doctor import RVC_DIR
cmd = build_rvc_train_cmd(rvc_python=Path('/x/python'), exp_name='smoke', sample_rate=40000, version='v2', epochs=1, batch_size=1, save_every=1, f0_method='rmvpe', pretrained_g=Path('/abs/g.pth'), pretrained_d=Path('/abs/d.pth'), if_f0=True, gpus='0')
assert cmd[0] == '/x/python'
assert cmd[1] == 'infer/modules/train/train.py'
assert cmd[2:6] == ['-e', 'smoke', '-sr', '40k']
assert '-pg' in cmd and '-pd' in cmd
assert cmd[-2:] == ['-v', 'v2']
print('OK')
" && .venv/bin/ruff check src/train.py</automated>
  </verify>
  <acceptance_criteria>
    - `grep -q "def build_rvc_preprocess_cmd" src/train.py`
    - `grep -q "def build_rvc_extract_f0_cmd" src/train.py`
    - `grep -q "def build_rvc_extract_feature_cmd" src/train.py`
    - `grep -q "def build_rvc_train_cmd" src/train.py`
    - `grep -q "def _write_filelist" src/train.py`
    - `grep -q "def _write_exp_config" src/train.py`
    - `grep -q '"infer/modules/train/preprocess.py"' src/train.py`
    - `grep -q '"infer/modules/train/extract/extract_f0_print.py"' src/train.py`
    - `grep -q '"infer/modules/train/extract/extract_f0_rmvpe.py"' src/train.py`
    - `grep -q '"infer/modules/train/extract_feature_print.py"' src/train.py`
    - `grep -q '"infer/modules/train/train.py"' src/train.py`
    - `grep -q '"-pg"' src/train.py` AND `grep -q '"-pd"' src/train.py` AND `grep -q '"-sr"' src/train.py`
    - `grep -q "pretrained_g.resolve()" src/train.py` (D-21: absolute path enforcement)
    - `grep -c "import torch\|import fairseq\|import faiss" src/train.py` returns 0
    - `.venv/bin/ruff check src/train.py` exits 0
  </acceptance_criteria>
  <done>All four builders + filelist + config helpers exist, with the EXACT argv from RESEARCH.md §2 / §10.2.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: Create tests/unit/test_train.py with byte-exact argv assertions and helper coverage</name>
  <files>tests/unit/test_train.py</files>
  <read_first>
    - src/train.py (the completed module from Tasks 1 + 2)
    - tests/unit/test_doctor.py (for pytest style: imports, monkeypatch usage, tmp_path patterns)
    - tests/unit/test_generate_cli.py (for typer + monkeypatch test patterns)
    - .planning/phases/02-training-cli/02-RESEARCH.md §8 (Testability section, all 8 subsections)
    - The `<interfaces>` block in this PLAN.md (every expected argv list)
  </read_first>
  <behavior>
    - All tests pass under `python -m pytest tests/unit/test_train.py -x` with no markers needed
    - No GPU required, no rvc/.venv required, no network
    - Every test asserts a concrete value (an exact list, an exact string, an exact int) — no "is not None" smoke checks
    - Two-venv import-ban test scans src/train.py source text for forbidden import strings
    - Sample-rate flow test asserts the SAME int input produces "40000" in preprocess argv AND "40k" in train argv
  </behavior>
  <action>
Create `tests/unit/test_train.py` with these exact tests. Use the project's existing test style (look at tests/unit/test_doctor.py for the pattern).

```python
"""Unit tests for src/train.py — pure functions only, no GPU, no rvc/.venv."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.doctor import RVC_DIR
from src.train import (
    DEFAULT_NUM_PROCS,
    EXPERIMENT_NAME_RE,
    PRESETS,
    SR_STR_MAP,
    SUBPROCESS_EXTRA_ENV,
    TRAIN_SUCCESS_EXIT_CODES,
    VALID_F0_METHODS,
    VALID_PRESETS,
    VALID_SAMPLE_RATES,
    VALID_VERSIONS,
    WEIGHT_FILE_FLOOR_BYTES,
    _write_exp_config,
    _write_filelist,
    build_rvc_extract_f0_cmd,
    build_rvc_extract_feature_cmd,
    build_rvc_preprocess_cmd,
    build_rvc_train_cmd,
    count_dataset_inputs,
    resolve_preset,
    resolve_pretrained_paths,
    stage_1_is_done,
    stage_2_is_done,
    stage_3_is_done,
    stage_4_is_done,
    validate_experiment_name,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------- Constants ----------

def test_sr_str_map_exact():
    assert SR_STR_MAP == {32000: "32k", 40000: "40k", 48000: "48k"}


def test_valid_sets_exact():
    assert VALID_F0_METHODS == ("pm", "harvest", "rmvpe", "rmvpe_gpu")
    assert VALID_VERSIONS == ("v1", "v2")
    assert VALID_SAMPLE_RATES == (32000, 40000, 48000)
    assert VALID_PRESETS == ("smoke", "low", "balanced", "high")


def test_presets_exact_values():
    assert PRESETS["smoke"] == {"epochs": 1, "batch_size": 1, "save_every": 1}
    assert PRESETS["low"] == {"epochs": 200, "batch_size": 6, "save_every": 50}
    assert PRESETS["balanced"] == {"epochs": 300, "batch_size": 12, "save_every": 50}
    assert PRESETS["high"] == {"epochs": 500, "batch_size": 40, "save_every": 50}


def test_treats_61_as_success():
    assert TRAIN_SUCCESS_EXIT_CODES == (0, 61)


def test_subprocess_env_has_offline_flags():
    assert SUBPROCESS_EXTRA_ENV == {
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "LANG": "C.UTF-8",
    }


def test_default_num_procs_is_capped_at_8():
    assert DEFAULT_NUM_PROCS >= 1
    assert DEFAULT_NUM_PROCS <= 8


def test_weight_floor_bytes():
    assert WEIGHT_FILE_FLOOR_BYTES == 1024


# ---------- Preset resolution (D-03) ----------

def test_preset_default_no_overrides():
    assert resolve_preset("balanced", epochs=None, batch_size=None, save_every=None) == \
        {"epochs": 300, "batch_size": 12, "save_every": 50}


def test_preset_override_mix_high_plus_epochs():
    # D-03 example: --preset high --epochs 800
    assert resolve_preset("high", epochs=800, batch_size=None, save_every=None) == \
        {"epochs": 800, "batch_size": 40, "save_every": 50}


def test_preset_smoke_matches_roadmap_smoke_test():
    assert resolve_preset("smoke", epochs=None, batch_size=None, save_every=None) == \
        {"epochs": 1, "batch_size": 1, "save_every": 1}


# ---------- Experiment name validator ----------

@pytest.mark.parametrize("name", ["smoke", "myvoice_v1", "exp-01", "a", "A_B-C_123"])
def test_validate_experiment_name_accepts(name):
    assert validate_experiment_name(name) is True


@pytest.mark.parametrize("name", ["", "../../etc/passwd", "exp/01", "exp.01", "exp 01", "x" * 65, "exp!"])
def test_validate_experiment_name_rejects(name):
    assert validate_experiment_name(name) is False


# ---------- Pretrained resolver ----------

def test_resolve_pretrained_v2_40k_f0():
    g, d = resolve_pretrained_paths(sample_rate=40000, version="v2", if_f0=True)
    assert g == RVC_DIR / "assets" / "pretrained_v2" / "f0G40k.pth"
    assert d == RVC_DIR / "assets" / "pretrained_v2" / "f0D40k.pth"


def test_resolve_pretrained_v1_48k_no_f0():
    g, d = resolve_pretrained_paths(sample_rate=48000, version="v1", if_f0=False)
    assert g == RVC_DIR / "assets" / "pretrained" / "G48k.pth"
    assert d == RVC_DIR / "assets" / "pretrained" / "D48k.pth"


def test_resolve_pretrained_v2_32k_f0():
    g, d = resolve_pretrained_paths(sample_rate=32000, version="v2", if_f0=True)
    assert g.name == "f0G32k.pth"
    assert d.name == "f0D32k.pth"


# ---------- build_rvc_preprocess_cmd (Stage 1) ----------

def test_build_rvc_preprocess_cmd_balanced_v2(tmp_path):
    rvc_py = Path("/abs/rvc/.venv/bin/python")
    ds = tmp_path / "dataset"
    ds.mkdir()
    cmd = build_rvc_preprocess_cmd(
        rvc_python=rvc_py, dataset_dir=ds, sample_rate=40000, num_procs=8, exp_name="myvoice"
    )
    assert cmd == [
        str(rvc_py),
        "infer/modules/train/preprocess.py",
        str(ds.resolve()),
        "40000",
        "8",
        str((RVC_DIR / "logs" / "myvoice").resolve()),
        "False",
        "3.7",
    ]


def test_build_rvc_preprocess_cmd_48k(tmp_path):
    cmd = build_rvc_preprocess_cmd(
        rvc_python=Path("/p"), dataset_dir=tmp_path, sample_rate=48000, num_procs=4, exp_name="x"
    )
    assert "48000" in cmd
    assert "40000" not in cmd


# ---------- build_rvc_extract_f0_cmd (Stage 2) ----------

def test_build_rvc_extract_f0_cmd_rmvpe_branch_a():
    cmd = build_rvc_extract_f0_cmd(
        rvc_python=Path("/p"), exp_name="myvoice", num_procs=8, f0_method="rmvpe"
    )
    assert cmd == [
        "/p",
        "infer/modules/train/extract/extract_f0_print.py",
        str((RVC_DIR / "logs" / "myvoice").resolve()),
        "8",
        "rmvpe",
    ]


def test_build_rvc_extract_f0_cmd_rmvpe_gpu_branch_b():
    cmd = build_rvc_extract_f0_cmd(
        rvc_python=Path("/p"), exp_name="myvoice", num_procs=8, f0_method="rmvpe_gpu"
    )
    assert cmd == [
        "/p",
        "infer/modules/train/extract/extract_f0_rmvpe.py",
        "1",
        "0",
        "0",
        str((RVC_DIR / "logs" / "myvoice").resolve()),
        "True",
    ]


def test_build_rvc_extract_f0_cmd_pm_and_harvest_use_branch_a():
    for method in ("pm", "harvest"):
        cmd = build_rvc_extract_f0_cmd(
            rvc_python=Path("/p"), exp_name="x", num_procs=4, f0_method=method
        )
        assert cmd[1] == "infer/modules/train/extract/extract_f0_print.py"
        assert cmd[-1] == method


# ---------- build_rvc_extract_feature_cmd (Stage 3) ----------

def test_build_rvc_extract_feature_cmd_v2():
    cmd = build_rvc_extract_feature_cmd(
        rvc_python=Path("/p"), exp_name="myvoice", version="v2"
    )
    assert cmd == [
        "/p",
        "infer/modules/train/extract_feature_print.py",
        "cuda:0",
        "1",
        "0",
        "0",
        str((RVC_DIR / "logs" / "myvoice").resolve()),
        "v2",
        "True",
    ]


def test_build_rvc_extract_feature_cmd_v1():
    cmd = build_rvc_extract_feature_cmd(
        rvc_python=Path("/p"), exp_name="x", version="v1"
    )
    assert cmd[-2] == "v1"


# ---------- build_rvc_train_cmd (Stage 4) ----------

def test_build_rvc_train_cmd_smoke_v2_40k():
    cmd = build_rvc_train_cmd(
        rvc_python=Path("/abs/rvc/.venv/bin/python"),
        exp_name="smoke",
        sample_rate=40000,
        version="v2",
        epochs=1,
        batch_size=1,
        save_every=1,
        f0_method="rmvpe",
        pretrained_g=Path("/abs/rvc/assets/pretrained_v2/f0G40k.pth"),
        pretrained_d=Path("/abs/rvc/assets/pretrained_v2/f0D40k.pth"),
        if_f0=True,
        gpus="0",
    )
    assert cmd == [
        "/abs/rvc/.venv/bin/python",
        "infer/modules/train/train.py",
        "-e", "smoke",
        "-sr", "40k",
        "-f0", "1",
        "-bs", "1",
        "-g", "0",
        "-te", "1",
        "-se", "1",
        "-pg", "/abs/rvc/assets/pretrained_v2/f0G40k.pth",
        "-pd", "/abs/rvc/assets/pretrained_v2/f0D40k.pth",
        "-l", "0",
        "-c", "0",
        "-sw", "0",
        "-v", "v2",
    ]


def test_build_rvc_train_cmd_e_is_bare_name_not_path():
    cmd = build_rvc_train_cmd(
        rvc_python=Path("/p"), exp_name="myvoice", sample_rate=40000, version="v2",
        epochs=300, batch_size=12, save_every=50, f0_method="rmvpe",
        pretrained_g=Path("/g"), pretrained_d=Path("/d"),
    )
    e_idx = cmd.index("-e")
    assert cmd[e_idx + 1] == "myvoice"
    assert "/" not in cmd[e_idx + 1]


def test_build_rvc_train_cmd_pg_pd_always_passed_d21():
    cmd = build_rvc_train_cmd(
        rvc_python=Path("/p"), exp_name="x", sample_rate=40000, version="v2",
        epochs=1, batch_size=1, save_every=1, f0_method="rmvpe",
        pretrained_g=Path("/g.pth"), pretrained_d=Path("/d.pth"),
    )
    assert "-pg" in cmd
    assert "-pd" in cmd
    pg_idx = cmd.index("-pg")
    pd_idx = cmd.index("-pd")
    assert cmd[pg_idx + 1] != ""
    assert cmd[pd_idx + 1] != ""


def test_build_rvc_train_cmd_no_f0():
    cmd = build_rvc_train_cmd(
        rvc_python=Path("/p"), exp_name="x", sample_rate=40000, version="v2",
        epochs=1, batch_size=1, save_every=1, f0_method="pm",
        pretrained_g=Path("/g"), pretrained_d=Path("/d"), if_f0=False,
    )
    f0_idx = cmd.index("-f0")
    assert cmd[f0_idx + 1] == "0"


# ---------- TRAIN-13 sample-rate flow ----------

def test_sample_rate_flows_to_both_preprocess_and_train(tmp_path):
    sr = 48000
    ds = tmp_path / "ds"
    ds.mkdir()
    pre = build_rvc_preprocess_cmd(
        rvc_python=Path("/p"), dataset_dir=ds, sample_rate=sr, num_procs=4, exp_name="x"
    )
    trn = build_rvc_train_cmd(
        rvc_python=Path("/p"), exp_name="x", sample_rate=sr, version="v2",
        epochs=1, batch_size=1, save_every=1, f0_method="rmvpe",
        pretrained_g=Path("/g"), pretrained_d=Path("/d"),
    )
    assert "48000" in pre
    sr_idx = trn.index("-sr")
    assert trn[sr_idx + 1] == "48k"


# ---------- count_dataset_inputs ----------

def test_count_dataset_inputs_empty(tmp_path):
    assert count_dataset_inputs(tmp_path) == 0


def test_count_dataset_inputs_mixed(tmp_path):
    (tmp_path / "a.wav").touch()
    (tmp_path / "b.wav").touch()
    (tmp_path / "c.flac").touch()
    (tmp_path / "readme.txt").touch()
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "d.wav").touch()  # not counted (non-recursive)
    n = count_dataset_inputs(tmp_path)
    assert n == 3


# ---------- Sentinel probes (TRAIN-08, D-08, D-09) ----------

def test_stage_1_is_done(tmp_path):
    exp = tmp_path / "logs" / "x"
    (exp / "0_gt_wavs").mkdir(parents=True)
    for i in range(5):
        (exp / "0_gt_wavs" / f"clip{i}.wav").touch()
    assert stage_1_is_done(exp, 5) is True
    assert stage_1_is_done(exp, 6) is False


def test_stage_1_is_done_missing_dir(tmp_path):
    assert stage_1_is_done(tmp_path / "nope", 5) is False


def test_stage_1_is_done_zero_expected_returns_false(tmp_path):
    (tmp_path / "0_gt_wavs").mkdir()
    assert stage_1_is_done(tmp_path, 0) is False


def test_stage_2_is_done_requires_both_dirs(tmp_path):
    (tmp_path / "2a_f0").mkdir()
    (tmp_path / "2b-f0nsf").mkdir()
    for i in range(3):
        (tmp_path / "2a_f0" / f"c{i}.npy").touch()
    assert stage_2_is_done(tmp_path, 3) is False  # 2b-f0nsf empty
    for i in range(3):
        (tmp_path / "2b-f0nsf" / f"c{i}.npy").touch()
    assert stage_2_is_done(tmp_path, 3) is True


def test_stage_3_v2_uses_feature768(tmp_path):
    (tmp_path / "3_feature768").mkdir()
    for i in range(2):
        (tmp_path / "3_feature768" / f"c{i}.npy").touch()
    assert stage_3_is_done(tmp_path, 2, "v2") is True
    assert stage_3_is_done(tmp_path, 2, "v1") is False  # wrong dir for v1


def test_stage_3_v1_uses_feature256(tmp_path):
    (tmp_path / "3_feature256").mkdir()
    (tmp_path / "3_feature256" / "c.npy").touch()
    assert stage_3_is_done(tmp_path, 1, "v1") is True


def test_stage_4_is_done_size_floor(tmp_path):
    p = tmp_path / "x.pth"
    assert stage_4_is_done(p) is False
    p.write_bytes(b"\x00" * 1023)
    assert stage_4_is_done(p) is False
    p.write_bytes(b"\x00" * 1024)
    assert stage_4_is_done(p) is True


# ---------- _write_filelist (TRAIN-05) ----------

def test_write_filelist_v2_with_f0(tmp_path, monkeypatch):
    # Stub RVC_DIR to point at a fake mute tree
    fake_rvc = tmp_path / "rvc"
    mute_gt = fake_rvc / "logs" / "mute" / "0_gt_wavs"
    mute_gt.mkdir(parents=True)
    (mute_gt / "mute40k.wav").touch()
    monkeypatch.setattr("src.train.RVC_DIR", fake_rvc)

    exp_dir = tmp_path / "logs" / "test"
    for sub in ("0_gt_wavs", "2a_f0", "2b-f0nsf", "3_feature768"):
        (exp_dir / sub).mkdir(parents=True)
    for name in ("clip1", "clip2", "clip3"):
        (exp_dir / "0_gt_wavs" / f"{name}.wav").touch()
        (exp_dir / "2a_f0" / f"{name}.wav.npy").touch()
        (exp_dir / "2b-f0nsf" / f"{name}.wav.npy").touch()
        (exp_dir / "3_feature768" / f"{name}.npy").touch()

    out = _write_filelist(exp_dir, version="v2", sample_rate=40000, if_f0=True)
    assert out == exp_dir / "filelist.txt"
    lines = [ln for ln in out.read_text().splitlines() if ln]
    # 3 clip rows + at least 1 mute row
    assert len(lines) >= 4
    for line in lines:
        fields = line.split("|")
        assert len(fields) == 5
        assert fields[4] == "0"  # sid


def test_write_filelist_v1_uses_feature256(tmp_path, monkeypatch):
    fake_rvc = tmp_path / "rvc"
    (fake_rvc / "logs" / "mute" / "0_gt_wavs").mkdir(parents=True)
    (fake_rvc / "logs" / "mute" / "0_gt_wavs" / "mute40k.wav").touch()
    monkeypatch.setattr("src.train.RVC_DIR", fake_rvc)

    exp_dir = tmp_path / "logs" / "v1exp"
    for sub in ("0_gt_wavs", "2a_f0", "2b-f0nsf", "3_feature256"):
        (exp_dir / sub).mkdir(parents=True)
    (exp_dir / "0_gt_wavs" / "c.wav").touch()
    (exp_dir / "2a_f0" / "c.wav.npy").touch()
    (exp_dir / "2b-f0nsf" / "c.wav.npy").touch()
    (exp_dir / "3_feature256" / "c.npy").touch()

    out = _write_filelist(exp_dir, version="v1", sample_rate=40000, if_f0=True)
    text = out.read_text()
    assert "3_feature256" in text
    assert "3_feature768" not in text


def test_write_filelist_raises_on_empty(tmp_path, monkeypatch):
    fake_rvc = tmp_path / "rvc"  # no mute dir
    monkeypatch.setattr("src.train.RVC_DIR", fake_rvc)
    exp_dir = tmp_path / "logs" / "empty"
    for sub in ("0_gt_wavs", "2a_f0", "2b-f0nsf", "3_feature768"):
        (exp_dir / sub).mkdir(parents=True)
    with pytest.raises(RuntimeError, match="empty filelist"):
        _write_filelist(exp_dir, version="v2", sample_rate=40000, if_f0=True)


# ---------- _write_exp_config ----------

def test_write_exp_config_copies_v2_48k(tmp_path, monkeypatch):
    fake_rvc = tmp_path / "rvc"
    cfg_src = fake_rvc / "configs" / "v2" / "48k.json"
    cfg_src.parent.mkdir(parents=True)
    cfg_src.write_text('{"x": 1}')
    monkeypatch.setattr("src.train.RVC_DIR", fake_rvc)

    exp_dir = tmp_path / "logs" / "exp"
    out = _write_exp_config(exp_dir, version="v2", sample_rate=48000)
    assert out == exp_dir / "config.json"
    assert out.read_text() == '{"x": 1}'


def test_write_exp_config_v2_40k_uses_v1_dir_quirk(tmp_path, monkeypatch):
    """v2+40k reads from configs/v1/40k.json per click_train lines 555-569."""
    fake_rvc = tmp_path / "rvc"
    cfg_src = fake_rvc / "configs" / "v1" / "40k.json"
    cfg_src.parent.mkdir(parents=True)
    cfg_src.write_text('{"v1quirk": true}')
    monkeypatch.setattr("src.train.RVC_DIR", fake_rvc)
    exp_dir = tmp_path / "logs" / "exp"
    out = _write_exp_config(exp_dir, version="v2", sample_rate=40000)
    assert out.read_text() == '{"v1quirk": true}'


def test_write_exp_config_missing_source_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("src.train.RVC_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="RVC config template missing"):
        _write_exp_config(tmp_path / "logs" / "exp", version="v2", sample_rate=48000)


# ---------- TRAIN-10 / D-24 two-venv import ban ----------

def test_no_rvc_imports_in_train_module():
    src = (PROJECT_ROOT / "src" / "train.py").read_text()
    forbidden = (
        "import torch",
        "import fairseq",
        "import faiss",
        "from torch",
        "from fairseq",
        "from faiss",
    )
    for bad in forbidden:
        assert bad not in src, f"src/train.py must not contain: {bad!r}"


def test_experiment_name_re_blocks_traversal():
    import re as _re
    pat = _re.compile(EXPERIMENT_NAME_RE)
    assert pat.match("../../etc/passwd") is None
    assert pat.match("good_name") is not None
```

DO NOT skip any test. DO NOT mark any with `@pytest.mark.gpu` or `@pytest.mark.network` — these are all pure unit tests.
  </action>
  <verify>
    <automated>cd /home/henrique/Development/train_audio_model && .venv/bin/python -m pytest tests/unit/test_train.py -x -q</automated>
  </verify>
  <acceptance_criteria>
    - File `tests/unit/test_train.py` exists
    - `.venv/bin/python -m pytest tests/unit/test_train.py -x -q` exits 0
    - `.venv/bin/python -m pytest tests/unit/test_train.py --collect-only -q | grep -c "::test_"` returns at least 30
    - `grep -q "def test_build_rvc_preprocess_cmd_balanced_v2" tests/unit/test_train.py`
    - `grep -q "def test_build_rvc_train_cmd_smoke_v2_40k" tests/unit/test_train.py`
    - `grep -q "def test_sample_rate_flows_to_both_preprocess_and_train" tests/unit/test_train.py`
    - `grep -q "def test_no_rvc_imports_in_train_module" tests/unit/test_train.py`
    - `grep -q "def test_write_filelist_v2_with_f0" tests/unit/test_train.py`
    - `grep -q "def test_treats_61_as_success" tests/unit/test_train.py`
    - `grep -q "def test_write_exp_config_v2_40k_uses_v1_dir_quirk" tests/unit/test_train.py`
    - `.venv/bin/ruff check tests/unit/test_train.py` exits 0
  </acceptance_criteria>
  <done>All ~30 unit tests pass under default pytest filter; argv assertions are byte-exact against the pinned RVC commit.</done>
</task>

</tasks>

<verification>
- `cd /home/henrique/Development/train_audio_model && .venv/bin/python -m pytest tests/unit/test_train.py -x -q` exits 0
- `.venv/bin/ruff check src/train.py tests/unit/test_train.py` exits 0
- `grep -E "^import (torch|fairseq|faiss)|^from (torch|fairseq|faiss)" src/train.py` returns nothing
- `.venv/bin/python -c "import src.train; print(src.train.SR_STR_MAP)"` prints `{32000: '32k', 40000: '40k', 48000: '48k'}`
</verification>

<success_criteria>
- Four pure arg-builders, filelist helper, config helper, preset resolver, pretrained resolver, sentinel probes, experiment-name validator all exist and are tested
- Every argv list in the test suite is byte-exact against the pinned RVC commit per RESEARCH.md §2 / §10.2
- TRAIN-13 sample-rate consistency proven by `test_sample_rate_flows_to_both_preprocess_and_train`
- TRAIN-10 two-venv boundary proven by `test_no_rvc_imports_in_train_module`
- TRAIN-14 offline env proven by `test_subprocess_env_has_offline_flags`
- TRAIN-04 satisfied by ~30 passing unit tests with no GPU
- TRAIN-05 satisfied by `_write_filelist` + 3 dedicated tests
- Plan 02 can build the typer CLI on top of these helpers without rediscovering the argv contract
</success_criteria>

<output>
After completion, create `.planning/phases/02-training-cli/02-01-SUMMARY.md` documenting:
- Final shape of `src/train.py` (line count, public symbols)
- Test count and runtime
- Any deviation from RESEARCH.md §2 argv tables (should be zero)
- Resolved Open Questions: 1 (config helper as separate `_write_exp_config`), 4 (--f0-method valid set excludes dio), 5 (experiment-name regex)
</output>
