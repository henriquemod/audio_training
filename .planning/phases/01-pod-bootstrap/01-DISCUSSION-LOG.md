# Phase 1: Pod Bootstrap - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-09
**Phase:** 01-pod-bootstrap
**Areas discussed:** CUDA install strategy, mise + Python 3.10 acquisition, Doctor --training composition, Idempotency + error surfacing

---

## Gray Area Selection

| Option | Description | Selected |
|--------|-------------|----------|
| CUDA install strategy | Probe logic, apt vs runfile, 24.04 behavior | ✓ |
| mise install + Python 3.10 acquisition | How mise and Python 3.10 are obtained on a bare pod | ✓ |
| Doctor --training check composition | Check set + disk/VRAM floor values + probe method | ✓ |
| Idempotency sentinels + error surfacing | Probe style, failure mode, logging | ✓ |

**User's choice:** All four areas selected.

---

## CUDA Install Strategy

### Q1: How should setup_pod.sh probe for an existing CUDA 12.1 toolkit?

| Option | Description | Selected |
|--------|-------------|----------|
| `nvcc --version` grep 12.1 | Strict 12.1 match, skip if present | ✓ |
| Any 12.x accepted | Forgiving on PyTorch base images | |
| `torch.cuda.is_available()` in rvc/.venv | End-state check | |

**User's choice:** Strict `nvcc --version` grep `release 12.1`.
**Notes:** Avoids silent drift against RVC's `torch 2.1.2+cu121` pin.

### Q2: On Ubuntu 22.04, how should CUDA 12.1 be installed when missing?

| Option | Description | Selected |
|--------|-------------|----------|
| NVIDIA apt keyring 22.04 | Matches BOOT-03 verbatim, deterministic | ✓ |
| Minimal: only what torch wheel needs | Skip apt toolkit entirely | |
| Runfile `--toolkit --silent --override` | Cross-distro, medium-confidence recipe | |

**User's choice:** NVIDIA apt keyring 22.04 method.
**Notes:** BOOT-03 is explicit; matches DEBIAN_FRONTEND=noninteractive TZ=UTC rule.

### Q3: What should setup_pod.sh do on Ubuntu 24.04?

| Option | Description | Selected |
|--------|-------------|----------|
| Detect, warn, attempt apt | Best-effort per DOCS-02 | ✓ |
| Hard refuse on 24.04 | Safest, loses best-effort promise | |
| Fall back to runfile on 24.04 | Two code paths | |

**User's choice:** Detect via `/etc/os-release`, warn loudly, attempt apt, exit 1 with clear hint on failure.

### Q4: PATH handling after CUDA install

| Option | Description | Selected |
|--------|-------------|----------|
| Export in-script only | Minimal system pollution | ✓ |
| Also write `/etc/profile.d/cuda.sh` | Persisted for interactive shells | |
| Rely on absolute paths only | Zero env pollution, verbose | |

**User's choice:** In-script `export PATH=/usr/local/cuda-12.1/bin:$PATH` only.

---

## mise + Python 3.10 Acquisition

### Q5: How should setup_pod.sh install mise on a bare Ubuntu pod?

| Option | Description | Selected |
|--------|-------------|----------|
| Official curl installer to `/root/.local/bin` | Matches project mise-first preference | |
| Install to `/usr/local/bin/mise` | Avoids home-dir PATH dance | |
| `apt install mise` | Rejected — not in default repos | |

**User's choice:** _Pivoted the framing._
**Notes:** "The `mise` usage was done for my local environment, it is convenient, but when running in a disposable pod, I don't really care about mise, I only care about having the job done, can use/detect mise, use it only if it's present, if not, go for a more traditional approach."

### Q6 (revised): How should setup_pod.sh acquire Python 3.10 on a pod?

| Option | Description | Selected |
|--------|-------------|----------|
| Detect mise → else deadsnakes PPA | Use mise if present, otherwise traditional path | partial |
| Deadsnakes always | Single code path | |
| Detect python3.10 first, then deadsnakes | Most minimal install | |

**User's choice:** _Asked for a fuller probe ladder_ — pods often already have Python installed; probe-and-skip should happen before any install.
**Notes:** "These pods also most likely already have python installed and configured, how do we deal with it? Maybe detect the python, if version meets requirement then could we skip some steps?"

### Q7: Full Python 3.10 probe ladder — first match wins

| Option | Description | Selected |
|--------|-------------|----------|
| `venv → system → mise → deadsnakes` | Four layers, each one probe line | ✓ |
| `venv → system → deadsnakes` (skip mise) | Three layers, simpler | |

**User's choice:** Four-layer ladder: `.venv/bin/python` → `python3.10` in PATH → `mise install python@3.10` → deadsnakes PPA.

### Q8: App venv creation idempotency

| Option | Description | Selected |
|--------|-------------|----------|
| Skip if egg-info present | Keeps 10s re-run target | ✓ |
| Always run `pip install -e` | Simpler, adds ~2-5s | |

**User's choice:** Probe for `src/train_audio_model.egg-info`; skip `pip install -e ".[dev]"` if present.

---

## Doctor --training Check Composition

### Q9: What exact set of checks should `doctor.py --training` run?

| Option | Description | Selected |
|--------|-------------|----------|
| System + RVC + training-specific | Full pod-readiness composition | ✓ |
| Training-specific only | Composable but more typing | |
| All doctor checks (no subset) | Includes irrelevant inference checks | |

**User's choice:** Full composition — system + RVC + new training checks. One command = "is this pod ready to train?".

### Q10: What should `check_disk_space_floor` default `min_gb` be?

| Option | Description | Selected |
|--------|-------------|----------|
| 20 GB on dataset/rvc root | Covers weights + dataset + intermediates + export | ✓ |
| 10 GB | Tighter, risky for longer runs | |
| 50 GB | Generous, locks out small ephemeral disks | |

**User's choice:** 20 GB default, parameter overridable.

### Q11: What should `check_gpu_vram_floor` default `min_gb` be?

| Option | Description | Selected |
|--------|-------------|----------|
| 8 GB | Minimum viable RVC v2 training | |
| 6 GB | Permits 1060/1660 6 GB | |
| 12 GB | Matches RTX 3060 12 GB tier | ✓ |

**User's choice:** 12 GB. Locks out 8 GB cards, admits RTX 3060 and above.

### Q12: Where should `check_disk_space_floor` measure from by default?

| Option | Description | Selected |
|--------|-------------|----------|
| `PROJECT_ROOT` | Single partition model, caller can override | ✓ |
| Both `PROJECT_ROOT` and `/tmp` | Two checks for split-mount pods | |

**User's choice:** `PROJECT_ROOT` with override capability.

### Q13: How should `check_gpu_vram_floor` measure VRAM?

| Option | Description | Selected |
|--------|-------------|----------|
| `nvidia-smi --query-gpu=memory.total` | Matches `check_nvidia_smi` pattern, no torch | ✓ |
| `rvc/.venv` torch subprocess | More authoritative, slower, venv hop | |

**User's choice:** `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits`, parse MiB → GB, take max across GPUs.

---

## Idempotency + Error Surfacing

### Q14: How should each install layer mark "done"?

| Option | Description | Selected |
|--------|-------------|----------|
| Intrinsic probes only | Probe the real artifact, no markers | ✓ |
| Sentinel files under `.planning/pod/` | Faster re-probe, stale-marker risk | |
| Hybrid: probe + marker for expensive steps | Optimization, adds complexity | |

**User's choice:** Intrinsic-probe-only. Matches existing `setup_rvc.sh` pattern, zero stale-marker risk.

### Q15: Failure mode on mid-layer install failure

| Option | Description | Selected |
|--------|-------------|----------|
| Fail fast, print diagnostic, exit non-zero | `set -euo pipefail` + `_tail`-style stderr | ✓ |
| Fail fast + rollback partial layer | Cleaner state, fragile rollback code | |
| Continue past failures, summarize | Violates BOOT-01 "exits 0" requirement | |

**User's choice:** Fail fast, no rollback. Print layer name + failing command + last ~20 lines of stderr. Exit 1.

### Q16: Logging approach

| Option | Description | Selected |
|--------|-------------|----------|
| `tee` to `scripts/setup_pod.log` + re-exec pattern | Mirrors existing `setup_rvc.sh` | ✓ |
| Stream only, no log file | Simpler, loses post-mortem | |

**User's choice:** Re-exec + `tee -a scripts/setup_pod.log` guarded by `_SETUP_POD_REEXEC`, with `PIPESTATUS[0]` propagation.

---

## Claude's Discretion

- Exact error message wording and `rich.Table` formatting details in `src/doctor.py`.
- Internal function naming within `src/doctor.py` and bash helper extraction in `setup_pod.sh`.
- Layer ordering past the hard dependency chain.
- Optional debugging flags like `--skip-cuda` (not required, low-cost to add).

## Deferred Ideas

- Aggressive caching of pretrained weights (V2-FAST-01).
- Prebuilt wheel cache for `rvc/.venv` (V2-FAST-02).
- Parallel installs (project-level deprioritization).
- Provider-specific pod image detection (out of scope).
- Smart batch-size from VRAM detection (V2-TRAIN-01).
