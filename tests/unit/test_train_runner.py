"""Tests for the stage-runner helpers in src/train.py."""

from __future__ import annotations

import os
import sys

from src.train import (
    SUBPROCESS_EXTRA_ENV,
    TRAIN_SUCCESS_EXIT_CODES,
    _build_subprocess_env,
    _is_train_success,
    _print_failure_tail,
    _run_stage_streamed,
    _tail_file,
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
