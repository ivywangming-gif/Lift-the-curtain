# Self-Supervised Data Audit

Status: EXPLORATORY_NOT_FORMAL_BENCHMARK.

Snapshot identity: 5e3c3fba3ad4e55a2ebc083fabcc0ae0033ffcaa90275eaa1d802e046d116b1b; manifest SHA-256: f4d2e517da27eaed22e44aa3115ea8d8866abe32f03c4ad87e550e87563a2255.
Both pause flags were present; formal training window allow_training was False.
No manifest, raw episode, pause flag, or formal queue was modified.

## Data Counts

- Admitted histories: 18; all legal histories including pending collision audits: 21.
- Known labels including EMPTY controls: 14; EMPTY controls: 2.
- Core known labels after excluding EMPTY: 12 (PASSABLE 8, BLOCKED 4).
- Legal LABEL_UNKNOWN histories: 7 (admitted 4, pending audit 3).
- Pending collision-audit histories: G_B10_left_coverage_00_nominal_s71700_rng1_act2__attempt01 [B10/left; ACTION_TUBE_VIOLATION ]; G_D01_left_coverage_00_nominal_s72700_rng1_act2__attempt01 [D01/left; FINITE_WALL_NO_CLOSED_LOOP_ACTION_IMPOSSIBILITY_PROOF ]; G_D01_right_coverage_00_nominal_s72700_rng1_act2__attempt01 [D01/right; FINITE_WALL_NO_CLOSED_LOOP_ACTION_IMPOSSIBILITY_PROOF ].
- Admitted physical configurations: 8; all legal configuration IDs including pending: 10.
- Left/right legal history coverage: 11 / 10.

## Inclusion Rules

- Supervised BCE uses only admitted, non-control, reliable PASSABLE/BLOCKED rows in the split train role.
- EMPTY controls are never core supervised labels; their histories may be present in self-supervised pretraining as unlabeled histories.
- Admitted UNKNOWN rows are allowed only for self-supervised pretraining when they are in the split train role.
- B10 and D01 pending collision-audit histories remain UNKNOWN and are excluded from supervision and self-supervised training.
- Calibration and test histories are never used for self-supervised fitting or normalization.

## Augmentation Contract

Each view applies whole-timestamp contiguous masking and small sensor-only noise. Time order is preserved; q_cmd, q, and qdot are never independently permuted or scaled. No arbitrary crop is used, so the legal contact phase cannot be removed by augmentation.

## Evaluation Limitation

Only the two cross-hand splits have both core classes in the test endpoint. They share physical groups across hands and therefore are hand-transfer diagnostics, not independent-configuration OOD evidence. No formal OOD ranking or deployment promotion is made.
