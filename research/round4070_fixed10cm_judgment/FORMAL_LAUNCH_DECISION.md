# Formal Launch Decision

Date: 2026-11-05 (Asia/Hong_Kong)

Decision: **WITHHELD AFTER SAFETY VALIDATION**.

The physics-only gates passed, and the required bilateral safety checks for `EMPTY_CONTROL`, `RIGID_CONTROL` and `FABRIC_NOMINAL` produced complete runtime summaries with safe recovery. The six protocol-matched safety rows nevertheless stopped as `PROBE_INCOMPLETE`; their signed actual FK displacements were approximately `-0.014 m` to `0.088 m`, rather than a completed `0.10 m` target.

An independent FK sweep of the pinned G1 URDF also found that the current symmetric shoulder-pitch/elbow reference has a maximum forward projection of about `0.0889 m` over its allowed `[0, 1.20] rad` step interval. A different probe pose or trajectory may be possible, but it would be a new calibration contract and was not silently substituted into this preregistered campaign.

An isolated full-arm FK calibration recheck was then run outside the manifest, with no navigation or benchmark labeling. The right-side candidate reached `0.100378 m` with `TARGET_REACHED`; the left-side candidate reached `0.093505 m` and remained `PROBE_INCOMPLETE`. Both sides had runtime-finalize validity, no safety stop, and successful retract/neutralize/stand recovery. This is useful evidence that a different target can work on one side, but it does not establish a bilateral reproducible calibration and therefore does not clear the formal launch gate.

The protocol says the target is not a forced actual displacement, so no extra torque, retry, or longer corrective push was added. Under the frozen evaluation rule, an incomplete probe is `UNKNOWN`/invalid for passability labeling. Launching the remaining 220 core rows would therefore create more execution histories but no valid PASSABLE/BLOCKED denominator, and would not be a statistically meaningful Round 4070-D benchmark.

Calibration and validation outputs remain preserved. The `.70 rad` historical output and the `.1.20 rad` calibration output are explicitly excluded from the current `.85 rad` formal protocol by the evaluator's protocol-mismatch audit.

This is a protocol-feasibility stop, not a classifier-error stop. The next experiment must first establish a reproducible bilateral probe trajectory, or define a preregistered lower-depth evaluation with an appropriate physical ground truth. The calibration recheck is documented separately in `PROBE_CALIBRATION_RECHECK.md`; it does not amend the current manifest.
