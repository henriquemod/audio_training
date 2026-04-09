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

# ---------- Constants ----------

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
DEFAULT_NUM_PROCS: int = min(os.cpu_count() or 1, 8)
STAGE_BANNER: str = "===== Stage {n}: {name} (started {ts}) ====="

# Silence unused-import warnings for re-exports used by other modules/tests.
_ = (PROJECT_ROOT, RVC_VENV_PYTHON)


# ---------- Pure helpers ----------


def resolve_preset(
    name: str,
    *,
    epochs: Optional[int],
    batch_size: Optional[int],
    save_every: Optional[int],
) -> dict[str, int]:
    """Resolve a named preset with optional per-field overrides.

    Args:
        name: Preset name, must be one of VALID_PRESETS.
        epochs: Override value for ``epochs`` (or ``None`` to keep preset default).
        batch_size: Override value for ``batch_size`` (or ``None``).
        save_every: Override value for ``save_every`` (or ``None``).

    Returns:
        A new dict with keys ``epochs``, ``batch_size``, ``save_every``.

    Raises:
        KeyError: If ``name`` is not in ``PRESETS``.
    """
    resolved = dict(PRESETS[name])
    if epochs is not None:
        resolved["epochs"] = epochs
    if batch_size is not None:
        resolved["batch_size"] = batch_size
    if save_every is not None:
        resolved["save_every"] = save_every
    return resolved


def resolve_pretrained_paths(
    *,
    sample_rate: int,
    version: str,
    if_f0: bool,
) -> tuple[Path, Path]:
    """Return the expected ``(G_path, D_path)`` pretrained weight paths.

    Args:
        sample_rate: Integer Hz (32000, 40000, or 48000).
        version: ``"v1"`` or ``"v2"``.
        if_f0: Whether pitch-guided (``f0``) pretrained weights are needed.

    Returns:
        Tuple of (generator path, discriminator path) under ``rvc/assets/``.

    Raises:
        KeyError: If ``sample_rate`` is not in ``SR_STR_MAP``.
    """
    sr_str = SR_STR_MAP[sample_rate]
    sub = "pretrained_v2" if version == "v2" else "pretrained"
    prefix = "f0" if if_f0 else ""
    g = RVC_DIR / "assets" / sub / f"{prefix}G{sr_str}.pth"
    d = RVC_DIR / "assets" / sub / f"{prefix}D{sr_str}.pth"
    return (g, d)


def count_dataset_inputs(dataset_dir: Path) -> int:
    """Count audio files (non-recursive) in ``dataset_dir`` matching ``AUDIO_EXTS``.

    Args:
        dataset_dir: Directory to scan.

    Returns:
        Number of files whose lowercase suffix is in :data:`AUDIO_EXTS`. Returns
        0 if the directory does not exist.
    """
    try:
        return sum(
            1
            for p in dataset_dir.iterdir()
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS
        )
    except FileNotFoundError:
        return 0


def stage_1_is_done(exp_dir: Path, expected: int) -> bool:
    """Return True iff Stage 1 (preprocess) looks complete.

    Checks that ``exp_dir/0_gt_wavs/*.wav`` contains at least ``expected`` files.
    Returns False if ``expected <= 0`` or the directory is missing.
    """
    if expected <= 0:
        return False
    try:
        return len(list((exp_dir / "0_gt_wavs").glob("*.wav"))) >= expected
    except FileNotFoundError:
        return False


def stage_2_is_done(exp_dir: Path, expected: int) -> bool:
    """Return True iff Stage 2 (F0 extraction) looks complete.

    Requires BOTH ``2a_f0/*.npy`` and ``2b-f0nsf/*.npy`` to have at least
    ``expected`` files. Returns False if ``expected <= 0`` or either dir missing.
    """
    if expected <= 0:
        return False
    try:
        a = len(list((exp_dir / "2a_f0").glob("*.npy")))
        b = len(list((exp_dir / "2b-f0nsf").glob("*.npy")))
    except FileNotFoundError:
        return False
    return a >= expected and b >= expected


def stage_3_is_done(exp_dir: Path, expected: int, version: str) -> bool:
    """Return True iff Stage 3 (feature extraction) looks complete.

    Probes ``3_feature768/`` for v2 or ``3_feature256/`` for v1. Returns False
    if ``expected <= 0`` or the feature directory is missing.
    """
    if expected <= 0:
        return False
    feat_subdir = "3_feature768" if version == "v2" else "3_feature256"
    try:
        return len(list((exp_dir / feat_subdir).glob("*.npy"))) >= expected
    except FileNotFoundError:
        return False


def stage_4_is_done(weight_path: Path) -> bool:
    """Return True iff the trained weight file exists and is non-trivial.

    Uses ``WEIGHT_FILE_FLOOR_BYTES`` (1024) as the minimum-size floor to guard
    against partially-written checkpoints.
    """
    try:
        return weight_path.exists() and weight_path.stat().st_size >= WEIGHT_FILE_FLOOR_BYTES
    except OSError:
        return False


def validate_experiment_name(name: str) -> bool:
    """Return True iff ``name`` matches :data:`EXPERIMENT_NAME_RE`.

    Blocks path traversal (``..``, ``/``) and shell metacharacters by restricting
    the experiment name to ``[a-zA-Z0-9_-]{1,64}``.
    """
    return bool(re.match(EXPERIMENT_NAME_RE, name))
