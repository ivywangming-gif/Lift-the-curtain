# Round 4070-E Git Delivery Report

## Scope

- The E implementation is limited to `scripts/round4070_fixed_probe_benchmark/`, `configs/round4070_fixed_probe_benchmark/`, and `research/round4070_fixed_probe_benchmark/`.
- The pre-existing dirty file `lift_curtain/perception/online_occupancy.py` and unrelated historical files were left untouched.
- Raw trial outputs, resource telemetry, checkpoints and caches remain under ignored `outputs/` paths and are not part of the scoped delivery.

## Checkpoint and Push

- Pre-E local checkpoint: `bf8b54e65f7cf495a558d11d5f713029d8ec75b3` (`research: checkpoint Round 4070 evidence and runners`).
- The E delivery is intended to be committed as a second scoped local checkpoint after final validation; the exact resulting SHA is recorded in the final task response with `git log`.
- `git push origin main` was attempted once during this task and was rejected by the Codex safety policy because the proposed checkpoint was treated as a potentially private large change to an unverified external remote.
- No remote update is claimed. No force-push, history rewrite, or destructive cleanup was performed.

## Reproduction Evidence

- The manifest, frozen rule file, physics-gate status, trial metadata, command JSON and generated research summaries provide the configuration and provenance chain.
- P1 results are explicitly marked physics-only and cannot invalidate P0.
- A summary directory is counted as an experiment only when the corresponding `summary.json` exists; registered rows are not substituted for observed trials.
