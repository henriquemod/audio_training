---
status: partial
phase: 02-training-cli
source: [02-VERIFICATION.md]
started: 2026-04-09T00:00:00Z
updated: 2026-04-09T00:00:00Z
---

## Current Test

[awaiting human testing on a GPU pod]

## Tests

### 1. End-to-end training run on a fresh GPU pod
expected: `python src/train.py --experiment-name myvoice --dataset-dir dataset/processed/` runs all 4 stages and produces `rvc/assets/weights/myvoice.pth` (>= 1 KiB), exits 0
result: [pending]

### 2. Sentinel-based skip-if-done resume
expected: Re-invoking the same command after success prints `Experiment 'myvoice' already complete` and exits 0 in <2s; or with stage-by-stage skip messages if only weight was deleted
result: [pending]

### 3. Mid-pipeline abort and resume
expected: After aborting after Stage 2 completes (e.g. SIGTERM during Stage 3), re-invoking the same command resumes from Stage 3 and produces an identical final weight file
result: [pending]

### 4. Deliberate training failure produces tail + exit 3 with stage context
expected: Triggering a real RVC failure (e.g. corrupted dataset, missing hubert) prints last 30 lines of stage stderr with stage context banner and exits 3
result: [pending]

## Summary

total: 4
passed: 0
issues: 0
pending: 4
skipped: 0
blocked: 0

## Gaps
