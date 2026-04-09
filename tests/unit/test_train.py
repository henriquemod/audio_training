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
    assert {32000: "32k", 40000: "40k", 48000: "48k"} == SR_STR_MAP


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
    assert {
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "LANG": "C.UTF-8",
    } == SUBPROCESS_EXTRA_ENV


def test_default_num_procs_is_capped_at_8():
    assert DEFAULT_NUM_PROCS >= 1
    assert DEFAULT_NUM_PROCS <= 8


def test_weight_floor_bytes():
    assert WEIGHT_FILE_FLOOR_BYTES == 1024


# ---------- Preset resolution (D-03) ----------


def test_preset_default_no_overrides():
    assert resolve_preset("balanced", epochs=None, batch_size=None, save_every=None) == {
        "epochs": 300,
        "batch_size": 12,
        "save_every": 50,
    }


def test_preset_override_mix_high_plus_epochs():
    # D-03 example: --preset high --epochs 800
    assert resolve_preset("high", epochs=800, batch_size=None, save_every=None) == {
        "epochs": 800,
        "batch_size": 40,
        "save_every": 50,
    }


def test_preset_smoke_matches_roadmap_smoke_test():
    assert resolve_preset("smoke", epochs=None, batch_size=None, save_every=None) == {
        "epochs": 1,
        "batch_size": 1,
        "save_every": 1,
    }


# ---------- Experiment name validator ----------


@pytest.mark.parametrize("name", ["smoke", "myvoice_v1", "exp-01", "a", "A_B-C_123"])
def test_validate_experiment_name_accepts(name):
    assert validate_experiment_name(name) is True


@pytest.mark.parametrize(
    "name",
    ["", "../../etc/passwd", "exp/01", "exp.01", "exp 01", "x" * 65, "exp!"],
)
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
    cmd = build_rvc_extract_feature_cmd(rvc_python=Path("/p"), exp_name="x", version="v1")
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
        rvc_python=Path("/p"),
        exp_name="myvoice",
        sample_rate=40000,
        version="v2",
        epochs=300,
        batch_size=12,
        save_every=50,
        f0_method="rmvpe",
        pretrained_g=Path("/g"),
        pretrained_d=Path("/d"),
    )
    e_idx = cmd.index("-e")
    assert cmd[e_idx + 1] == "myvoice"
    assert "/" not in cmd[e_idx + 1]


def test_build_rvc_train_cmd_pg_pd_always_passed_d21():
    cmd = build_rvc_train_cmd(
        rvc_python=Path("/p"),
        exp_name="x",
        sample_rate=40000,
        version="v2",
        epochs=1,
        batch_size=1,
        save_every=1,
        f0_method="rmvpe",
        pretrained_g=Path("/g.pth"),
        pretrained_d=Path("/d.pth"),
    )
    assert "-pg" in cmd
    assert "-pd" in cmd
    pg_idx = cmd.index("-pg")
    pd_idx = cmd.index("-pd")
    assert cmd[pg_idx + 1] != ""
    assert cmd[pd_idx + 1] != ""


def test_build_rvc_train_cmd_no_f0():
    cmd = build_rvc_train_cmd(
        rvc_python=Path("/p"),
        exp_name="x",
        sample_rate=40000,
        version="v2",
        epochs=1,
        batch_size=1,
        save_every=1,
        f0_method="pm",
        pretrained_g=Path("/g"),
        pretrained_d=Path("/d"),
        if_f0=False,
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
        rvc_python=Path("/p"),
        exp_name="x",
        sample_rate=sr,
        version="v2",
        epochs=1,
        batch_size=1,
        save_every=1,
        f0_method="rmvpe",
        pretrained_g=Path("/g"),
        pretrained_d=Path("/d"),
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
