# Physics Validation

This file is the D-round physics gate record. A gate is a scene-validity gate, not a passability label.

## Contract

- Hidden-wall gaps are measured from the initial curtain front surface (`x=0.85 m` in the formal scene) to the hidden rigid front face (`center_x - 0.02 m`). The registered gaps are 0.02, 0.05 and 0.08 m, all inside the 0.10 m probe target.
- No hidden wall is exposed to the deployment classifier. Wall geometry is used only by the physics/evaluation process.
- A no-probe matched control is required before a probe gate is accepted.
- Any front-face penetration, solver instability, non-finite state or uncontrolled stretch fails the condition and prevents G1 launches for that condition.
- PVC and multi-panel/opening branches remain excluded unless their independent physics gate passes. The Round 4070-C articulated-strip surrogate did not pass and is not copied into D labels.

## Gate Ledger

| Gate | Scene change | D status | G1 eligibility |
|---|---|---|---|
| `A0_TOP_EDGE_ONLY` | nominal top edge | PASS, D no-probe + probe | eligible after the separate safety gate |
| `A1_TOP_EDGE_BOTTOM_TWO_CORNERS` | two bottom corners | PASS, D no-probe + probe | eligible after the separate safety gate |
| `A2_FOUR_CORNERS_ONLY` | four-corner topology | PASS, D no-probe + probe | eligible after the separate safety gate |
| `A3_FULL_PERIMETER_FIXED` | full perimeter | PASS, D no-probe + probe | eligible after the separate safety gate |
| `B_D225_NEW` | density multiplier 2.25 | PASS, D no-probe + probe | eligible after the separate safety gate |
| `B_D225_A1` | density 2.25 plus A1 constraint | PASS, D no-probe + probe | eligible after the separate safety gate |
| `C_GAP02` | hidden wall gap 2 cm | PASS, D no-probe + probe | eligible after the separate safety gate |
| `C_GAP05` | hidden wall gap 5 cm | PASS, D no-probe + probe | eligible after the separate safety gate |
| `C_GAP08` | hidden wall gap 8 cm | PASS, D no-probe + probe | eligible after the separate safety gate |
| `CONTROL_ONLY` | empty control | control | safety validation only |
| `INHERITED_RIGID_CONTROL` | rigid panel control | PASS_CONTROL | safety validation only |

The machine-readable results are written to `outputs/round4070_fixed10cm_judgment/physics/` and summarized by `scripts/round4070_fixed10cm_judgment/run_physics_matrix.ps1`. The frozen launch status is in `configs/round4070_fixed10cm_judgment/PHYSICS_GATE_STATUS.json`; all nine core scene gates are currently `PASS` and both controls are `PASS_CONTROL`.

The physics gates pass, but they do not certify that the robot can realize the 0.10 m actual FK displacement. The separate probe safety audit records that distinction; a safe `PROBE_INCOMPLETE` is never counted as a physical BLOCKED label.
