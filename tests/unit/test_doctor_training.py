"""Unit tests for the training pre-flight doctor checks added in Phase 1."""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from unittest.mock import patch

from src.doctor import (
    HUBERT_MIN_BYTES,
    check_disk_space_floor,
    check_gpu_vram_floor,
    check_hubert_base,
    check_rvc_mute_refs,
)

_DiskUsage = namedtuple("_DiskUsage", ["total", "used", "free"])


def _make_sparse_file(path: Path, size: int) -> None:
    """Create a sparse file of `size` bytes without allocating real blocks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        if size > 0:
            f.seek(size - 1)
            f.write(b"\x00")


# --- check_disk_space_floor ---


def test_check_disk_space_floor_above():
    usage = _DiskUsage(total=100 * 1024**3, used=50 * 1024**3, free=50 * 1024**3)
    with patch("shutil.disk_usage", return_value=usage):
        result = check_disk_space_floor(Path("/tmp"), min_gb=20)
    assert result.ok is True
    assert "GiB free" in result.detail


def test_check_disk_space_floor_at_threshold():
    usage = _DiskUsage(total=100 * 1024**3, used=80 * 1024**3, free=20 * 1024**3)
    with patch("shutil.disk_usage", return_value=usage):
        result = check_disk_space_floor(Path("/tmp"), min_gb=20)
    assert result.ok is True


def test_check_disk_space_floor_below():
    usage = _DiskUsage(total=100 * 1024**3, used=95 * 1024**3, free=5 * 1024**3)
    with patch("shutil.disk_usage", return_value=usage):
        result = check_disk_space_floor(Path("/tmp"), min_gb=20)
    assert result.ok is False
    assert "5" in result.detail
    assert "20" in result.fix_hint


def test_check_disk_space_floor_missing_path():
    with patch("shutil.disk_usage", side_effect=FileNotFoundError):
        result = check_disk_space_floor(Path("/nonexistent/path"), min_gb=20)
    assert result.ok is False
    assert result.detail  # non-empty


def test_check_disk_space_floor_permission_error():
    with patch("shutil.disk_usage", side_effect=PermissionError):
        result = check_disk_space_floor(Path("/root/forbidden"), min_gb=20)
    assert result.ok is False
    assert "permission" in result.detail.lower()


# --- check_gpu_vram_floor ---


def test_check_gpu_vram_floor_single_gpu_above():
    with patch("shutil.which", return_value="/usr/bin/nvidia-smi"), patch(
        "subprocess.run"
    ) as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "24564\n"
        mock_run.return_value.stderr = ""
        result = check_gpu_vram_floor(12)
    assert result.ok is True
    assert "24" in result.detail


def test_check_gpu_vram_floor_multi_gpu_picks_max():
    with patch("shutil.which", return_value="/usr/bin/nvidia-smi"), patch(
        "subprocess.run"
    ) as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "8192\n24564\n"
        mock_run.return_value.stderr = ""
        result = check_gpu_vram_floor(12)
    assert result.ok is True
    assert "24" in result.detail


def test_check_gpu_vram_floor_below():
    with patch("shutil.which", return_value="/usr/bin/nvidia-smi"), patch(
        "subprocess.run"
    ) as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "8192\n"
        mock_run.return_value.stderr = ""
        result = check_gpu_vram_floor(12)
    assert result.ok is False
    assert "12" in result.fix_hint


def test_check_gpu_vram_floor_no_driver():
    with patch("shutil.which", return_value=None):
        result = check_gpu_vram_floor(12)
    assert result.ok is False
    assert "nvidia-smi" in (result.fix_hint + result.detail).lower()


def test_check_gpu_vram_floor_nvidia_smi_nonzero():
    with patch("shutil.which", return_value="/usr/bin/nvidia-smi"), patch(
        "subprocess.run"
    ) as mock_run:
        mock_run.return_value.returncode = 9
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = "driver mismatch"
        result = check_gpu_vram_floor(12)
    assert result.ok is False
    assert "driver mismatch" in result.detail


def test_check_gpu_vram_floor_unparseable_output():
    with patch("shutil.which", return_value="/usr/bin/nvidia-smi"), patch(
        "subprocess.run"
    ) as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "not a number\n"
        mock_run.return_value.stderr = ""
        result = check_gpu_vram_floor(12)
    assert result.ok is False


# --- check_rvc_mute_refs ---


def test_check_rvc_mute_refs_present(tmp_path, monkeypatch):
    monkeypatch.setattr("src.doctor.RVC_DIR", tmp_path)
    monkeypatch.setattr("src.doctor.PROJECT_ROOT", tmp_path)
    mute = tmp_path / "logs" / "mute"
    mute.mkdir(parents=True)
    (mute / "dummy.wav").write_bytes(b"\x00")
    result = check_rvc_mute_refs()
    assert result.ok is True


def test_check_rvc_mute_refs_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("src.doctor.RVC_DIR", tmp_path)
    monkeypatch.setattr("src.doctor.PROJECT_ROOT", tmp_path)
    result = check_rvc_mute_refs()
    assert result.ok is False
    assert "setup_rvc.sh" in result.fix_hint


def test_check_rvc_mute_refs_empty_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("src.doctor.RVC_DIR", tmp_path)
    monkeypatch.setattr("src.doctor.PROJECT_ROOT", tmp_path)
    (tmp_path / "logs" / "mute").mkdir(parents=True)
    result = check_rvc_mute_refs()
    assert result.ok is False
    assert "empty" in result.detail


def test_check_rvc_mute_refs_permission_error(tmp_path, monkeypatch):
    monkeypatch.setattr("src.doctor.RVC_DIR", tmp_path)
    monkeypatch.setattr("src.doctor.PROJECT_ROOT", tmp_path)
    (tmp_path / "logs" / "mute").mkdir(parents=True)
    with patch("pathlib.Path.iterdir", side_effect=PermissionError("EACCES")):
        result = check_rvc_mute_refs()
    assert result.ok is False
    assert "cannot read" in result.detail or "EACCES" in result.detail


# --- check_hubert_base ---


def test_check_hubert_base_present(tmp_path, monkeypatch):
    monkeypatch.setattr("src.doctor.RVC_DIR", tmp_path)
    monkeypatch.setattr("src.doctor.PROJECT_ROOT", tmp_path)
    hubert = tmp_path / "assets" / "hubert" / "hubert_base.pt"
    _make_sparse_file(hubert, HUBERT_MIN_BYTES)
    result = check_hubert_base()
    assert result.ok is True


def test_check_hubert_base_truncated(tmp_path, monkeypatch):
    monkeypatch.setattr("src.doctor.RVC_DIR", tmp_path)
    monkeypatch.setattr("src.doctor.PROJECT_ROOT", tmp_path)
    hubert = tmp_path / "assets" / "hubert" / "hubert_base.pt"
    hubert.parent.mkdir(parents=True)
    hubert.write_bytes(b"\x00" * 1000)
    result = check_hubert_base()
    assert result.ok is False
    assert "1000" in result.detail


def test_check_hubert_base_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("src.doctor.RVC_DIR", tmp_path)
    monkeypatch.setattr("src.doctor.PROJECT_ROOT", tmp_path)
    result = check_hubert_base()
    assert result.ok is False
    assert "setup_rvc.sh" in result.fix_hint


def test_check_hubert_base_permission_error(tmp_path, monkeypatch):
    monkeypatch.setattr("src.doctor.RVC_DIR", tmp_path)
    monkeypatch.setattr("src.doctor.PROJECT_ROOT", tmp_path)
    hubert = tmp_path / "assets" / "hubert" / "hubert_base.pt"
    hubert.parent.mkdir(parents=True)
    hubert.write_bytes(b"\x00")
    # Path.exists() swallows OSError and returns False, so patch it to True
    # and let the subsequent stat() raise — exercises the try/except guard.
    with patch("pathlib.Path.exists", return_value=True), patch(
        "pathlib.Path.stat", side_effect=PermissionError("EACCES")
    ):
        result = check_hubert_base()
    assert result.ok is False
    assert "cannot stat" in result.detail or "EACCES" in result.detail
