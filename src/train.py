"""RVC training pipeline CLI: preprocess -> extract_f0 -> extract_feature -> train.

Orchestrates all four RVC training stages end-to-end as subprocesses into rvc/.venv,
with always-on intrinsic probe-and-skip resume.

Two-venv boundary: this module MUST NOT depend on torch / fairseq / faiss / rvc
at the Python level. All RVC interaction is via subprocess.Popen(..., cwd=RVC_DIR,
shell=False) — no in-process imports of the RVC stack are permitted.

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


# ---------- Pure arg-builders ----------


def build_rvc_preprocess_cmd(
    *,
    rvc_python: Path,
    dataset_dir: Path,
    sample_rate: int,
    num_procs: int,
    exp_name: str,
) -> list[str]:
    """Build argv for RVC Stage 1 (preprocess.py).

    Mirrors ``rvc/infer-web.py:218-254`` (``preprocess_dataset``). Pure function.
    ``sample_rate`` is the integer Hz value (e.g. 40000), NOT the ``"40k"`` string.

    Args:
        rvc_python: Absolute path to ``rvc/.venv/bin/python``.
        dataset_dir: Directory of input audio clips.
        sample_rate: Integer Hz sample rate.
        num_procs: Parallel worker count.
        exp_name: Experiment name (bare, not a path).

    Returns:
        Argv list ready to pass to ``subprocess.Popen(..., cwd=RVC_DIR)``.
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

    Two branches per ``rvc/infer-web.py:258-346``:
      - ``f0_method in {"pm", "harvest", "rmvpe"}`` -> ``extract_f0_print.py``
      - ``f0_method == "rmvpe_gpu"``               -> ``extract_f0_rmvpe.py``

    Args:
        rvc_python: Absolute path to ``rvc/.venv/bin/python``.
        exp_name: Experiment name (bare).
        num_procs: Worker count (ignored on the ``rmvpe_gpu`` branch).
        f0_method: One of :data:`VALID_F0_METHODS`.
        gpu_id: GPU index string (``rmvpe_gpu`` branch only).
        is_half: Whether to run in fp16 (``rmvpe_gpu`` branch only).

    Returns:
        Argv list ready to pass to ``subprocess.Popen(..., cwd=RVC_DIR)``.
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

    Mirrors ``rvc/infer-web.py:355-395`` (single-GPU branch).

    Args:
        rvc_python: Absolute path to ``rvc/.venv/bin/python``.
        exp_name: Experiment name (bare).
        version: ``"v1"`` or ``"v2"``.
        num_procs_per_gpu: Signature parity; single-GPU path always uses ``leng=1``.
        device: Torch device string (default ``"cuda:0"``).
        gpu_id: GPU index string.
        is_half: Whether to run in fp16.

    Returns:
        Argv list ready to pass to ``subprocess.Popen(..., cwd=RVC_DIR)``.
    """
    del num_procs_per_gpu  # kept for signature parity; single-GPU branch hardcodes leng=1
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

    Mirrors ``rvc/infer-web.py:571-589`` (``click_train``, gpus-present branch).
    D-21: ``-pg`` and ``-pd`` are ALWAYS passed with absolute paths to prevent
    silent random-init training when pretrained files are missing from disk.

    Args:
        rvc_python: Absolute path to ``rvc/.venv/bin/python``.
        exp_name: Experiment name (bare; passed to ``-e`` as-is, NOT a path).
        sample_rate: Integer Hz; converted to ``"40k"``-style string for ``-sr``.
        version: ``"v1"`` or ``"v2"``.
        epochs: Total epochs (``-te``).
        batch_size: Per-GPU batch size (``-bs``).
        save_every: Checkpoint-save period in epochs (``-se``).
        f0_method: Retained for signature parity with the other builders.
        pretrained_g: Absolute path to the G (generator) pretrained weight.
        pretrained_d: Absolute path to the D (discriminator) pretrained weight.
        if_f0: True -> ``-f0 1``, False -> ``-f0 0``.
        gpus: GPU index string.

    Returns:
        Argv list ready to pass to ``subprocess.Popen(..., cwd=RVC_DIR)``.
    """
    del f0_method  # signature parity with the stage-2 builder; train.py needs no method arg
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


# ---------- Filelist + config helpers ----------


def _write_filelist(
    exp_dir: Path,
    *,
    version: str,
    sample_rate: int,
    if_f0: bool = True,
) -> Path:
    """Write ``exp_dir/filelist.txt`` with format-equivalent rows for RVC train.py.

    Format equivalent to (NOT byte-exact with) ``rvc/infer-web.py:click_train``
    lines 500-546. Per-clip row (with f0): 5 pipe-separated fields, ``sid=0``.
    Mute rows: one per file found under ``rvc/logs/mute/0_gt_wavs/`` matching
    the requested sample rate (D-07 — deviates from the webui's hardcoded
    ``range(2)``).

    Args:
        exp_dir: ``rvc/logs/<exp>/`` experiment directory.
        version: ``"v1"`` (uses ``3_feature256``) or ``"v2"`` (uses ``3_feature768``).
        sample_rate: Integer Hz; picks mute-reference files by ``sr_str``.
        if_f0: True -> 5-field rows; False -> 3-field rows.

    Returns:
        The path to the written filelist.

    Raises:
        RuntimeError: If the resulting file would be empty (zero clip stems
            intersected and zero mute files found).
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
    # Stage 2 files are named like "<stem>.wav.npy"; _stems strips the full
    # ".wav.npy" suffix, so no secondary strip is needed.
    f0_stems = _stems(f0_dir, ".wav.npy")
    f0nsf_stems = _stems(f0nsf_dir, ".wav.npy")

    common = (
        gt_stems & feat_stems & f0_stems & f0nsf_stems if if_f0 else gt_stems & feat_stems
    )

    lines: list[str] = []
    for stem in sorted(common):
        if if_f0:
            lines.append(
                f"{gt_dir}/{stem}.wav|{feat_dir}/{stem}.npy|"
                f"{f0_dir}/{stem}.wav.npy|{f0nsf_dir}/{stem}.wav.npy|0"
            )
        else:
            lines.append(f"{gt_dir}/{stem}.wav|{feat_dir}/{stem}.npy|0")

    # Mute rows (D-07): one per file found under rvc/logs/mute/0_gt_wavs/.
    # Prefer files matching the requested sample rate, otherwise any *.wav.
    mute_root = RVC_DIR / "logs" / "mute"
    mute_gt = mute_root / "0_gt_wavs"
    mute_files: list[Path] = []
    if mute_gt.is_dir():
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

    exp_dir.mkdir(parents=True, exist_ok=True)
    out = exp_dir / "filelist.txt"
    out.write_text("\n".join(lines) + "\n")
    return out


def _write_exp_config(
    exp_dir: Path,
    *,
    version: str,
    sample_rate: int,
) -> Path:
    """Copy ``rvc/configs/{v1,v2}/<sr>.json`` to ``exp_dir/config.json``.

    ``train.py`` opens this at startup and crashes if missing. Path resolution
    follows ``rvc/infer-web.py:click_train`` lines 555-569: ``v1`` OR
    (``v2`` AND ``sr == 40k``) -> ``v1`` directory; otherwise -> ``v2`` directory.
    Reads from the tracked source files (not ``configs/inuse/``, which is
    lazily populated by the webui and may not exist on a cold pod).

    Args:
        exp_dir: ``rvc/logs/<exp>/`` experiment directory (will be created).
        version: ``"v1"`` or ``"v2"``.
        sample_rate: Integer Hz; must be in :data:`SR_STR_MAP`.

    Returns:
        The destination path (``exp_dir/config.json``).

    Raises:
        FileNotFoundError: If the source template file is missing.
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
