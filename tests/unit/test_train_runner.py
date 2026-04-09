"""Tests for the stage-runner helpers in src/train.py."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import src.train as train_mod
from src.train import (
    SUBPROCESS_EXTRA_ENV,
    TRAIN_SUCCESS_EXIT_CODES,
    _build_subprocess_env,
    _is_train_success,
    _print_failure_tail,
    _run_stage_streamed,
    _tail_file,
    run_pipeline,
)

# ---------- _is_train_success (TRAIN-07) ----------


def test_is_train_success_zero():
    assert _is_train_success(0) is True


def test_is_train_success_61():
    """RVC's os._exit(2333333) truncates to 61 on Linux — must be treated as success."""
    assert _is_train_success(61) is True


def test_is_train_success_other_failures():
    for rc in (1, 2, 3, 7, 137, -9, 255):
        assert _is_train_success(rc) is False


def test_train_success_codes_constant():
    assert TRAIN_SUCCESS_EXIT_CODES == (0, 61)


# ---------- _build_subprocess_env (D-19, TRAIN-14) ----------


def test_build_subprocess_env_includes_offline_flags():
    env = _build_subprocess_env()
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["HF_DATASETS_OFFLINE"] == "1"
    assert env["LANG"] == "C.UTF-8"


def test_build_subprocess_env_inherits_path():
    env = _build_subprocess_env()
    assert "PATH" in env
    assert env["PATH"] == os.environ["PATH"]


def test_build_subprocess_env_does_not_mutate_os_environ():
    before = os.environ.get("TRANSFORMERS_OFFLINE")
    _build_subprocess_env()
    after = os.environ.get("TRANSFORMERS_OFFLINE")
    assert before == after


def test_extra_env_constant():
    expected = {
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "LANG": "C.UTF-8",
    }
    assert expected == SUBPROCESS_EXTRA_ENV


# ---------- _tail_file ----------


def test_tail_file_missing(tmp_path):
    assert _tail_file(tmp_path / "nope.log", 10) == ""


def test_tail_file_empty(tmp_path):
    p = tmp_path / "empty.log"
    p.touch()
    assert _tail_file(p, 10) == ""


def test_tail_file_fewer_lines(tmp_path):
    p = tmp_path / "five.log"
    p.write_text("a\nb\nc\nd\ne\n")
    out = _tail_file(p, 10)
    assert out.splitlines() == ["a", "b", "c", "d", "e"]


def test_tail_file_more_lines(tmp_path):
    p = tmp_path / "many.log"
    p.write_text("\n".join(f"line{i}" for i in range(100)) + "\n")
    out = _tail_file(p, 10)
    lines = out.splitlines()
    assert lines == [f"line{i}" for i in range(90, 100)]


def test_tail_file_handles_binary_garbage(tmp_path):
    p = tmp_path / "bin.log"
    p.write_bytes(b"\x00\xff\xfe text after\n" * 50)
    # Should not crash; returns some string (utf-8 errors=replace)
    out = _tail_file(p, 5)
    assert isinstance(out, str)


# ---------- _run_stage_streamed (uses real subprocess of cpython for stability) ----------


def test_run_stage_streamed_success(tmp_path):
    log = tmp_path / "train.log"
    rc = _run_stage_streamed(
        [sys.executable, "-c", "print('hello from stage')"],
        stage_num=1,
        stage_name="preprocess",
        log_path=log,
        env=os.environ.copy(),
    )
    assert rc == 0
    text = log.read_text()
    assert "Stage 1: preprocess" in text  # banner
    assert "hello from stage" in text


def test_run_stage_streamed_captures_nonzero_exit(tmp_path):
    log = tmp_path / "train.log"
    rc = _run_stage_streamed(
        [sys.executable, "-c", "import sys; print('boom'); sys.exit(7)"],
        stage_num=2,
        stage_name="extract_f0",
        log_path=log,
        env=os.environ.copy(),
    )
    assert rc == 7
    text = log.read_text()
    assert "boom" in text
    assert "Stage 2: extract_f0" in text


def test_run_stage_streamed_appends_not_overwrites(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("PRIOR CONTENT\n")
    rc = _run_stage_streamed(
        [sys.executable, "-c", "print('new line')"],
        stage_num=3,
        stage_name="extract_feature",
        log_path=log,
        env=os.environ.copy(),
    )
    assert rc == 0
    text = log.read_text()
    assert "PRIOR CONTENT" in text
    assert "new line" in text


# ---------- _print_failure_tail (TRAIN-11) ----------


def test_print_failure_tail_writes_30_lines(tmp_path, capsys):
    log = tmp_path / "train.log"
    log.write_text("\n".join(f"L{i}" for i in range(50)) + "\n")
    _print_failure_tail(log, stage=1, name="preprocess", verbose=False)
    err = capsys.readouterr().err
    assert "Stage 1 (preprocess) failed" in err
    assert "L49" in err
    assert "L20" in err
    # 30 lines means L20..L49 inclusive; L19 should NOT be there
    assert "L19" not in err


def test_print_failure_tail_verbose_writes_100_lines(tmp_path, capsys):
    log = tmp_path / "train.log"
    log.write_text("\n".join(f"L{i}" for i in range(150)) + "\n")
    _print_failure_tail(log, stage=4, name="train", verbose=True)
    err = capsys.readouterr().err
    assert "L50" in err  # 100 lines means L50..L149


def test_print_failure_tail_cuda_oom_hint(tmp_path, capsys):
    log = tmp_path / "train.log"
    log.write_text("some output\nCUDA out of memory\nmore output\n")
    _print_failure_tail(log, stage=4, name="train", verbose=False)
    err = capsys.readouterr().err
    assert "lower --batch-size" in err


def test_print_failure_tail_missing_log(tmp_path, capsys):
    _print_failure_tail(tmp_path / "nope.log", stage=1, name="preprocess", verbose=False)
    err = capsys.readouterr().err
    assert "no output captured" in err


def test_print_failure_tail_extra_hint(tmp_path, capsys):
    log = tmp_path / "train.log"
    log.write_text("done\n")
    _print_failure_tail(
        log, stage=4, name="train", verbose=False,
        extra_hint="RVC reported success but no weight file was produced",
    )
    err = capsys.readouterr().err
    assert "RVC reported success" in err


# ---------- run_pipeline orchestrator (mocked subprocess + filesystem) ----------


def _make_dataset(tmp_path: Path, n: int = 3) -> Path:
    ds = tmp_path / "ds"
    ds.mkdir()
    for i in range(n):
        (ds / f"clip{i}.wav").touch()
    return ds


def _populate_stage_outputs(exp_dir: Path, n: int, version: str = "v2") -> None:
    """Pretend stages 1-3 ran successfully."""
    (exp_dir / "0_gt_wavs").mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (exp_dir / "0_gt_wavs" / f"clip{i}.wav").touch()
    for sub in ("2a_f0", "2b-f0nsf"):
        (exp_dir / sub).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (exp_dir / sub / f"clip{i}.wav.npy").touch()
    feat = "3_feature768" if version == "v2" else "3_feature256"
    (exp_dir / feat).mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (exp_dir / feat / f"clip{i}.npy").touch()


def _stub_rvc_root(tmp_path: Path, monkeypatch) -> Path:
    """Make RVC_DIR point at a fake tree under tmp_path.

    Also stubs the mute reference audio tree and the config template used by
    ``_write_filelist`` and ``_write_exp_config``.
    """
    fake_rvc = tmp_path / "rvc"
    mute_gt = fake_rvc / "logs" / "mute" / "0_gt_wavs"
    mute_gt.mkdir(parents=True)
    (mute_gt / "mute40k.wav").touch()
    # v2 + 40k routes to configs/v1/40k.json per _write_exp_config.
    cfg = fake_rvc / "configs" / "v1" / "40k.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text('{"stub": true}')
    monkeypatch.setattr(train_mod, "RVC_DIR", fake_rvc)
    return fake_rvc


def test_run_pipeline_fast_path_when_weight_exists(tmp_path, monkeypatch, capsys):
    fake_rvc = _stub_rvc_root(tmp_path, monkeypatch)
    weight = fake_rvc / "assets" / "weights" / "smoke.pth"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"\x00" * 2048)
    ds = _make_dataset(tmp_path)

    called = {"count": 0}

    def fake_runner(cmd, **kwargs):
        called["count"] += 1
        return 0

    monkeypatch.setattr(train_mod, "_run_stage_streamed", fake_runner)

    rc = run_pipeline(
        experiment_name="smoke",
        dataset_dir=ds,
        sample_rate=40000,
        rvc_version="v2",
        f0_method="rmvpe",
        hp={"epochs": 1, "batch_size": 1, "save_every": 1},
        num_procs=1,
        gpus="0",
        pretrained_g=Path("/g"),
        pretrained_d=Path("/d"),
        if_f0=True,
        verbose=False,
    )
    assert rc == 0
    assert called["count"] == 0  # fast-path: NO stages invoked


def test_run_pipeline_skips_done_stages_runs_train(tmp_path, monkeypatch, capsys):
    fake_rvc = _stub_rvc_root(tmp_path, monkeypatch)
    ds = _make_dataset(tmp_path, n=3)
    exp_dir = fake_rvc / "logs" / "smoke"
    _populate_stage_outputs(exp_dir, n=3, version="v2")

    weight = fake_rvc / "assets" / "weights" / "smoke.pth"
    weight.parent.mkdir(parents=True)

    invocations = []

    def fake_runner(cmd, *, stage_num, stage_name, log_path, env):
        invocations.append((stage_num, stage_name))
        if stage_num == 4:
            weight.write_bytes(b"\x00" * 2048)  # simulate train.py producing the weight
        return 0

    monkeypatch.setattr(train_mod, "_run_stage_streamed", fake_runner)

    rc = run_pipeline(
        experiment_name="smoke",
        dataset_dir=ds,
        sample_rate=40000,
        rvc_version="v2",
        f0_method="rmvpe",
        hp={"epochs": 1, "batch_size": 1, "save_every": 1},
        num_procs=1,
        gpus="0",
        pretrained_g=Path("/g"),
        pretrained_d=Path("/d"),
        if_f0=True,
        verbose=False,
    )
    assert rc == 0
    # Only Stage 4 should have been invoked; 1-3 were skipped via sentinels.
    assert [s[0] for s in invocations] == [4]
    out = capsys.readouterr().out
    assert "Stage 1: skipping" in out
    assert "Stage 2: skipping" in out
    assert "Stage 3: skipping" in out


def test_run_pipeline_stage_4_exit_61_is_success(tmp_path, monkeypatch):
    fake_rvc = _stub_rvc_root(tmp_path, monkeypatch)
    ds = _make_dataset(tmp_path, n=2)
    exp_dir = fake_rvc / "logs" / "smoke"
    _populate_stage_outputs(exp_dir, n=2, version="v2")
    weight = fake_rvc / "assets" / "weights" / "smoke.pth"
    weight.parent.mkdir(parents=True)

    def fake_runner(cmd, *, stage_num, stage_name, log_path, env):
        if stage_num == 4:
            weight.write_bytes(b"\x00" * 2048)
            return 61  # the os._exit(2333333) truncation
        return 0

    monkeypatch.setattr(train_mod, "_run_stage_streamed", fake_runner)

    rc = run_pipeline(
        experiment_name="smoke",
        dataset_dir=ds,
        sample_rate=40000,
        rvc_version="v2",
        f0_method="rmvpe",
        hp={"epochs": 1, "batch_size": 1, "save_every": 1},
        num_procs=1,
        gpus="0",
        pretrained_g=Path("/g"),
        pretrained_d=Path("/d"),
        if_f0=True,
        verbose=False,
    )
    assert rc == 0


def test_run_pipeline_stage_1_failure_returns_3(tmp_path, monkeypatch, capsys):
    _stub_rvc_root(tmp_path, monkeypatch)
    ds = _make_dataset(tmp_path, n=2)

    def fake_runner(cmd, *, stage_num, stage_name, log_path, env):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("preprocess crash output\n")
        return 7

    monkeypatch.setattr(train_mod, "_run_stage_streamed", fake_runner)

    rc = run_pipeline(
        experiment_name="smoke",
        dataset_dir=ds,
        sample_rate=40000,
        rvc_version="v2",
        f0_method="rmvpe",
        hp={"epochs": 1, "batch_size": 1, "save_every": 1},
        num_procs=1,
        gpus="0",
        pretrained_g=Path("/g"),
        pretrained_d=Path("/d"),
        if_f0=True,
        verbose=False,
    )
    assert rc == 3
    err = capsys.readouterr().err
    assert "Stage 1 (preprocess) failed" in err
    assert "preprocess crash output" in err


def test_run_pipeline_stage_4_success_but_missing_weight_returns_3(
    tmp_path, monkeypatch, capsys
):
    fake_rvc = _stub_rvc_root(tmp_path, monkeypatch)
    ds = _make_dataset(tmp_path, n=2)
    exp_dir = fake_rvc / "logs" / "smoke"
    _populate_stage_outputs(exp_dir, n=2, version="v2")

    def fake_runner(cmd, *, stage_num, stage_name, log_path, env):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("training done but no weight written\n")
        return 0  # exit 0 but no weight file written

    monkeypatch.setattr(train_mod, "_run_stage_streamed", fake_runner)

    rc = run_pipeline(
        experiment_name="smoke",
        dataset_dir=ds,
        sample_rate=40000,
        rvc_version="v2",
        f0_method="rmvpe",
        hp={"epochs": 1, "batch_size": 1, "save_every": 1},
        num_procs=1,
        gpus="0",
        pretrained_g=Path("/g"),
        pretrained_d=Path("/d"),
        if_f0=True,
        verbose=False,
    )
    assert rc == 3
    err = capsys.readouterr().err
    assert "no weight file was produced" in err


def test_run_pipeline_stage_3_silent_hubert_failure(tmp_path, monkeypatch, capsys):
    """STATE.md pitfall: extract_feature_print.py exits 0 when hubert is missing."""
    fake_rvc = _stub_rvc_root(tmp_path, monkeypatch)
    ds = _make_dataset(tmp_path, n=3)
    exp_dir = fake_rvc / "logs" / "smoke"
    # Only stages 1 and 2 are populated; stage 3 will be invoked.
    (exp_dir / "0_gt_wavs").mkdir(parents=True)
    for i in range(3):
        (exp_dir / "0_gt_wavs" / f"clip{i}.wav").touch()
    for sub in ("2a_f0", "2b-f0nsf"):
        (exp_dir / sub).mkdir()
        for i in range(3):
            (exp_dir / sub / f"clip{i}.wav.npy").touch()

    def fake_runner(cmd, *, stage_num, stage_name, log_path, env):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(f"stage {stage_num} ran\n")
        return 0  # silent success — but no output files written

    monkeypatch.setattr(train_mod, "_run_stage_streamed", fake_runner)

    rc = run_pipeline(
        experiment_name="smoke",
        dataset_dir=ds,
        sample_rate=40000,
        rvc_version="v2",
        f0_method="rmvpe",
        hp={"epochs": 1, "batch_size": 1, "save_every": 1},
        num_procs=1,
        gpus="0",
        pretrained_g=Path("/g"),
        pretrained_d=Path("/d"),
        if_f0=True,
        verbose=False,
    )
    assert rc == 3
    err = capsys.readouterr().err
    assert "Stage 3" in err
    assert "hubert" in err.lower()
