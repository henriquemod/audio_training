# Phase 2: Training CLI - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-09
**Phase:** 02-training-cli
**Areas discussed:** Flag defaults & resume, Filelist exactness, Stage-skip probe rules, Log streaming strategy, Doctor check placement

---

## Flag defaults & resume

### Q1: Default --sample-rate for a first training run?

| Option | Description | Selected |
|--------|-------------|----------|
| 40000 (Recommended) | RVC community default; middle quality/speed tier; most pretrained_v2 weights ship at 40k | ✓ |
| 48000 | Highest quality; larger intermediate files; less battle-tested | |
| 32000 | Fastest/smallest; lower fidelity; smoke tests only | |

**User's choice:** 40000
**Notes:** Matches RVC legacy habit; planner will codify as the default for `--sample-rate`.

### Q2: Default --epochs, --batch-size, --save-every?

| Option | Description | Selected |
|--------|-------------|----------|
| epochs=200, batch=8, save_every=50 (Recommended) | Standard RVC v2 run; fits 12GB VRAM floor | |
| epochs=100, batch=8, save_every=25 | Shorter smoke-friendly | |
| epochs=500, batch=8, save_every=50 | Conservative upper bound | |

**User's choice:** Other (free text)
**User's free-text response:** "lets have some presets? because i want to allow a granulated and refined options like your suggested but i also want presets for convinience, and sits this option switable for a 12gb vram gpu for the lower preset, for the hevier presset lets go big, something ready to use a high end nvidia gpu like a H200 oe H100"
**Notes:** Pivoted to a `--preset` flag with preset override semantics. Follow-up questions designed the preset set.

### Q2a (follow-up): Preset naming and VRAM targets?

| Option | Description | Selected |
|--------|-------------|----------|
| smoke / low / balanced / high (Recommended, default=balanced, 24GB) | Four tiers covering 1-file smoke, 12GB, 24GB, 80GB+ | ✓ |
| smoke / low / balanced / high with default=low | Same tiers, safer default | |
| smoke / default / max (3 tiers) | Fewer knobs | |

**User's choice:** smoke / low / balanced / high (Recommended)
**Notes:** Preset values locked as D-02 in CONTEXT.md.

### Q2b (follow-up): Should --preset and explicit --batch-size/--epochs be mixable?

| Option | Description | Selected |
|--------|-------------|----------|
| Mixable — explicit flags override preset values (Recommended) | Presets are starting points | ✓ |
| Mutually exclusive — pick preset OR all explicit flags | Cleaner invariants, more friction | |

**User's choice:** Mixable
**Notes:** `preset_dict.update(explicit_overrides)` after typer parsing.

### Q3: Default --rvc-version and --f0-method?

| Option | Description | Selected |
|--------|-------------|----------|
| v2 + rmvpe (Recommended) | Current arch; most accurate f0; no Stage 2 GPU contention | ✓ |
| v2 + rmvpe_gpu | Faster f0 but GPU pressure | |
| v2 + harvest | CPU-only, reproducible, slower | |

**User's choice:** v2 + rmvpe

### Q4: How should resume work?

| Option | Description | Selected |
|--------|-------------|----------|
| Always-on probe-and-skip; --resume removed (Recommended) | Mirrors Phase 1 intrinsic-probe philosophy | ✓ |
| --resume flag opt-in; without it, refuses if logs exist | Explicit, more friction | |
| --resume flag opt-in; without it, wipes logs first | Destructive-by-default | |

**User's choice:** Always-on probe-and-skip; --resume removed
**Notes:** Deliberate deviation from TRAIN-01's literal flag list; planner must note in PLAN.md so verification accepts the missing flag.

---

## Filelist exactness (TRAIN-05)

### Q5: How strictly should _write_filelist match click_train's output?

| Option | Description | Selected |
|--------|-------------|----------|
| Format-equivalent; unit-tested against a fixture (Recommended) | Line format matches; tests assert structure not bytes | ✓ |
| Byte-exact reproduction; diff test against captured golden | Brittle | |
| Call RVC's click_train helper directly via subprocess | Violates two-venv boundary — rejected | |

**User's choice:** Format-equivalent
**Notes:** Tests assert pipe-field count, path validity, sid values, mute-ref injection count, non-empty.

### Q6: Mute-reference row handling?

| Option | Description | Selected |
|--------|-------------|----------|
| Mirror click_train: one row per mute file in rvc/logs/mute/ (Recommended) | Glob-driven; count matches what download_models.py shipped | ✓ |
| Hard-code 3 mute-ref rows | Fragile | |

**User's choice:** Mirror click_train
**Notes:** check_rvc_mute_refs (Phase 1) validates presence as pre-flight.

---

## Stage-skip probe rules (TRAIN-08)

### Q7: What counts as 'populated' for a stage-skip probe?

| Option | Description | Selected |
|--------|-------------|----------|
| Expected file count match against dataset; mismatch = re-run (Recommended) | Catches partial-crash state; no stale markers | ✓ |
| Non-empty directory check; any file present = skip | Simple but unsafe for partial crashes (violates TRAIN-09) | |
| Per-stage sentinel file (STAGE1_DONE etc.) written after stage returns 0 | Stale-marker risk; conflicts with intrinsic-probe philosophy | |

**User's choice:** Expected file count match
**Notes:** N = count of audio files in --dataset-dir matching AUDIO_EXTS. RVC's per-file overwrite behavior makes stage re-runs safe.

### Q8: Stage 4 (train) resume — let RVC handle it, or add our own guard?

| Option | Description | Selected |
|--------|-------------|----------|
| Let RVC's built-in checkpoint resume handle it (Recommended) | TRAIN-08 explicitly says this; G_*/D_* auto-load | ✓ |
| Skip Stage 4 entirely if <name>.pth exists | Loses partial checkpoint progress | |

**User's choice:** Let RVC handle it
**Notes:** Fast-path: if <name>.pth already exists and non-empty, skip Stage 4 entirely (D-09 in CONTEXT.md).

---

## Log streaming strategy

### Q9: How should train.py surface subprocess output from multi-hour RVC stages?

| Option | Description | Selected |
|--------|-------------|----------|
| Stream live to terminal + tee to rvc/logs/<name>/train.log (Recommended) | Real-time visibility + post-mortem log | ✓ |
| Silent; buffer stderr; print last 30 lines on failure | Matches generate.py but bad UX for multi-hour runs | |
| Live stream to terminal only; no log file | Simpler; loses post-mortem data after SSH drops | |

**User's choice:** Stream live + tee
**Notes:** subprocess.Popen with line iteration; _tail reads from the log file at failure time.

### Q10: How should existing experiment state be handled on re-invocation?

| Option | Description | Selected |
|--------|-------------|----------|
| Probe-and-continue (Recommended) | Same name + same dataset = resume; no force flag | ✓ |
| Refuse unless --force; --force wipes logs | Friction on pods | |
| Timestamp each run | Defeats resume entirely | |

**User's choice:** Probe-and-continue
**Notes:** User picks a distinct --experiment-name for a clean slate.

---

## Doctor check placement

### Q11: Where should TRAIN-06's new pre-flight checks live?

| Option | Description | Selected |
|--------|-------------|----------|
| In src/doctor.py; --training composition includes them (Recommended) | Single source of training-readiness truth; aligned with Phase 1 D-09 | ✓ |
| Inline in train.py; doctor stays generic | Contradicts Phase 1 D-09 | |

**User's choice:** In src/doctor.py
**Notes:** New functions check_pretrained_v2_weights, check_training_dataset_nonempty. check_hubert_base and check_rvc_mute_refs already plumbed in Phase 1.

---

## Claude's Discretion

Captured in CONTEXT.md §Decisions → Claude's Discretion. Highlights:
- Exact typer Option(...) help text and rich.Table cosmetic formatting
- Preset dict container shape (dict vs dataclass)
- Per-stage num_procs defaults (will match RVC's historical values)
- Log banner wording
- Whether _write_filelist lives inline or in a new helper module

## Deferred Ideas

Captured in CONTEXT.md §Deferred. Highlights: V2-TRAIN-01 (smart batch size), V2-TRAIN-02 (tqdm/rich progress), V2-TRAIN-03 (hang detection), Stage 5 + auto-export (Phase 3), shell orchestration (Phase 4), README (Phase 5).
