"""Safe ffmpeg invocation wrapper.

All ffmpeg calls across this project go through `run_ffmpeg()`. The wrapper:
- Passes args as a list (never shell=True)
- Captures stderr fully
- Verifies the expected output file exists and is non-empty
- Raises FfmpegError with full context on any failure
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


class FfmpegError(RuntimeError):
    """Raised when an ffmpeg invocation fails or produces no output."""


@dataclass
class FfmpegResult:
    stdout: str
    stderr: str


def run_ffmpeg(
    args: list[str],
    *,
    context: str,
    expected_output: Path,
    binary: str = "ffmpeg",
) -> FfmpegResult:
    """Run ffmpeg with the given args and verify it produced the expected output.

    Args:
        args: ffmpeg arguments WITHOUT the leading "ffmpeg". Example:
            ["-i", "in.wav", "-ar", "44100", "out.wav"]
        context: a short description of the pipeline stage. Included in error
            messages to make debugging easier.
        expected_output: the file ffmpeg should produce. Verified to exist and
            be non-empty after ffmpeg exits zero.
        binary: override the ffmpeg binary path. Default: "ffmpeg" from PATH.

    Returns:
        FfmpegResult with captured stdout and stderr.

    Raises:
        FfmpegError: on non-zero exit, or if expected_output is missing/empty.
    """
    full_cmd = [binary, "-hide_banner", "-loglevel", "error", "-y", *args]
    try:
        proc = subprocess.run(
            full_cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise FfmpegError(
            f"[{context}] ffmpeg binary not found (looked for {binary!r}). "
            f"Install ffmpeg: sudo apt install ffmpeg"
        ) from exc

    if proc.returncode != 0:
        raise FfmpegError(
            f"[{context}] ffmpeg exited with code {proc.returncode}\n"
            f"  command: {' '.join(full_cmd)}\n"
            f"  stderr: {proc.stderr.strip() or '(empty)'}"
        )

    if not expected_output.exists():
        raise FfmpegError(
            f"[{context}] ffmpeg did not produce the expected output file: {expected_output}\n"
            f"  command: {' '.join(full_cmd)}\n"
            f"  stderr: {proc.stderr.strip() or '(empty)'}"
        )

    if expected_output.stat().st_size == 0:
        raise FfmpegError(
            f"[{context}] ffmpeg produced an empty output file: {expected_output}\n"
            f"  command: {' '.join(full_cmd)}\n"
            f"  stderr: {proc.stderr.strip() or '(empty)'}"
        )

    return FfmpegResult(stdout=proc.stdout, stderr=proc.stderr)
