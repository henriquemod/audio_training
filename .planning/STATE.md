---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: 3
current_plan: Not started
status: planning
last_updated: "2026-04-09T23:55:06.087Z"
progress:
  total_phases: 5
  completed_phases: 2
  total_plans: 5
  completed_plans: 5
  percent: 100
---

# Project State: train_audio_model

**Milestone:** Pod-Ready Training
**Last updated:** 2026-04-09

---

## Project Reference

**Core value:** A single user can rent a GPU pod, run two bash scripts, and walk away with a downloadable `.pth` + `.index` voice model trained from raw audio they provided.

**Project file:** `.planning/PROJECT.md`
**Requirements file:** `.planning/REQUIREMENTS.md`
**Roadmap file:** `.planning/ROADMAP.md`

---

## Current Position

Phase: 01 (pod-bootstrap) — EXECUTING
Plan: 1 of 2
**Current phase:** 3
**Current plan:** Not started
**Status:** Ready to plan

**Progress:**

```
[Phase 1] Pod Bootstrap            [ ] Not started
[Phase 2] Training CLI             [ ] Not started
[Phase 3] Index Training + Export  [ ] Not started
[Phase 4] Shell Orchestration      [ ] Not started
[Phase 5] Documentation            [ ] Not started

Overall: 0/5 phases complete
```

---

## Performance Metrics

- Plans completed this milestone: 0
- Phases completed this milestone: 0
- Requirements delivered: 0/41

---

## Accumulated Context

### Key Decisions Locked In

- `scripts/setup_pod.sh` wraps `scripts/setup_rvc.sh` — does NOT modify `setup_rvc.sh`
- `src/train.py` mirrors `src/generate.py` exactly: pure arg-builders + doctor pre-flight + thin orchestrator
- Two-venv boundary is absolute: no `import torch/faiss/fairseq` in any `src/` file
- FAISS index build committed as `scripts/rvc_patches/train_index_cli.py`, copied to `rvc/tools/` at bootstrap (not stored in `rvc/` directly — that dir is gitignored)
- Exit codes 0 AND 61 both indicate training success (`os._exit(2333333)` truncates to 61 on Linux)
- `curl -fL` is the only allowed remote pull mechanism — no boto3, no AWS SDK
- Auto-shutdown is documentation only — no `shutdown -h now` in any script

### Critical Pitfalls to Remember (from research)

- Missing pretrained weights cause silent random-init training — always pass `-pg`/`-pd` explicitly
- `extract_feature_print.py` exits 0 when hubert is missing — assert output dir non-empty after the subprocess
- `added_*.index` must be picked by mtime, not alphabetically — IVF cluster count in filename changes with dataset size
- `mise activate bash` fails in non-interactive scripts — use `$(mise where python)/bin/python3` or `mise exec`
- `DEBIAN_FRONTEND=noninteractive TZ=UTC` required on all apt calls — tzdata prompt hangs on billing pods
- Sample-rate chain: `--sample-rate` flag must drive BOTH the RVC preprocess resample AND the `train.py -sr` flag

### Open Questions

- Ubuntu 24.04 CUDA 12.1 runfile install: `--toolkit --silent --override` flags are community-sourced (medium confidence)
- `MASTER_PORT` TCP binding behavior on real pod provider firewalls — needs validation; add 120s hang timeout diagnostic

### Todos

- (none yet — roadmap just created)

### Blockers

- (none)

---

## Session Continuity

**To resume work:** Read `.planning/ROADMAP.md` to see phase structure, then run `/gsd-plan-phase 1` to begin planning Phase 1.

**Next action:** `/gsd-plan-phase 1`

---
*State initialized: 2026-04-09*
