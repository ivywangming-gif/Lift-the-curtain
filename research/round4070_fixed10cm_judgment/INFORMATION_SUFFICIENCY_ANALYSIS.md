# Information Sufficiency Analysis

The fixed 10 cm history is evaluated separately from the offline label.

- Type A: legal history features repeatably differ, but E1 does not use them.
- Type B: matched wall/no-wall histories remain close, so the probe itself is information-limited.
- Type C: local hand response exists without whole-body clearance.
- Type D: frozen confirmation disagrees with discovery.
- Type E: physics, safety, planner or runtime invalidity; excluded from method error rates.

| Family | Rows | Labeled | Unknown/invalid | E0 determinate | E1 determinate | E2 determinate |
|---|---:|---:|---:|---:|---:|---:|
| basic_fabric | 60 | 0 | 60 | 2 | 2 | 1 |
| control | 20 | 0 | 20 | 6 | 4 | 3 |
| heavy | 40 | 0 | 40 | 0 | 0 | 0 |
| hidden_wall | 60 | 0 | 60 | 0 | 0 | 0 |
| matched_no_wall | 30 | 0 | 30 | 0 | 0 | 0 |
| strong_constraint | 40 | 0 | 40 | 0 | 0 | 0 |

No UNKNOWN row is converted into BLOCKED. Matched-pair evidence is only claimable when both sides have valid deployment histories and a physics gate.

This run is `INSUFFICIENT_POSITIVE_COVERAGE`: the required safety checks were runtime-valid and safely recovered, but the fixed action produced `PROBE_INCOMPLETE` actual FK motion. No Type A-D classifier conclusion is claimed before a valid benchmark row exists.

Raw features and label reasons are in `LABEL_AUDIT.csv` and `stats/evaluator_rows.json`.
