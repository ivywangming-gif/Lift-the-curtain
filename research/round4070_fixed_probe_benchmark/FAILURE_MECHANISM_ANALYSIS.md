# Failure Mechanism Analysis

Counts use observed trial directories only. The required F1-F7 research mechanisms are not mutually exclusive; an execution record may appear in more than one evidence category. Low displacement is not counted as BLOCKED without independent contact/clearance evidence.

## Required Research Taxonomy

| Mechanism | Count | Evidence trial/run IDs | Interpretation |
|---|---:|---|---|
| F1 information underuse | 0 proven | closest E1 false-pass cases: E_PHASE_1_FABRIC_WALL_GAP_08_R_01_S408311, E_PHASE_1_FABRIC_WALL_GAP_08_R_02_S408318, E_PHASE_2_FABRIC_WALL_GAP_08_R_03_S409325, E_PHASE_2_FABRIC_WALL_GAP_08_R_04_S409332 | No proven E2-correct rescue exists; current data cannot isolate an E1 threshold/use failure from insufficient decisive observation. |
| F2 probe information insufficient | 60 | E_PHASE_0_FABRIC_NOMINAL_L_01_S407000, E_PHASE_0_FABRIC_NOMINAL_R_01_S407001, E_PHASE_1_FABRIC_NOMINAL_L_01_S408000, E_PHASE_1_FABRIC_NOMINAL_R_01_S408001, E_PHASE_1_FABRIC_NOMINAL_L_02_S408007, E_PHASE_1_FABRIC_NOMINAL_R_02_S408008 | Valid one-shot probes were contact-limited but did not independently establish whole-body clearance; these remain UNKNOWN. |
| F3 local contact versus whole-body traversal mismatch | 0 confirmed; 60 candidates | E_PHASE_0_FABRIC_NOMINAL_L_01_S407000, E_PHASE_0_FABRIC_NOMINAL_R_01_S407001, E_PHASE_1_FABRIC_NOMINAL_L_01_S408000, E_PHASE_1_FABRIC_NOMINAL_R_01_S408001, E_PHASE_1_FABRIC_NOMINAL_L_02_S408007, E_PHASE_1_FABRIC_NOMINAL_R_02_S408008 | Candidate pattern overlaps F2, but no independent whole-body evidence is sufficient to call these confirmed local-contact/global-traversal mismatches. |
| F4 heavy-curtain over-conservatism | 0 proven | heavy UNKNOWN examples: E_PHASE_1_FABRIC_HEAVY_FREE_L_01_S408186, E_PHASE_1_FABRIC_HEAVY_FREE_R_01_S408187, E_PHASE_1_FABRIC_HEAVY_FREE_L_02_S408193, E_PHASE_1_FABRIC_HEAVY_FREE_R_02_S408194, E_PHASE_1_FABRIC_HEAVY_CONSTRAINED_L_01_S408217, E_PHASE_1_FABRIC_HEAVY_CONSTRAINED_R_01_S408218 | No valid heavy-curtain PASSABLE label exists in this run, so a false-reject claim is not supported. |
| F5 constrained-curtain over-release | 26 unique blocked trials (E0=26, E1=4, E2=1) | E_PHASE_1_FABRIC_WALL_GAP_02_L_01_S408248, E_PHASE_1_FABRIC_WALL_GAP_02_R_01_S408249, E_PHASE_1_FABRIC_WALL_GAP_02_L_02_S408255, E_PHASE_1_FABRIC_WALL_GAP_02_R_02_S408256, E_PHASE_1_FABRIC_WALL_GAP_05_L_01_S408279, E_PHASE_1_FABRIC_WALL_GAP_05_R_01_S408280 | Hidden-wall BLOCKED rows are independent negative evidence; local responsiveness causes false PASSABLE decisions, reduced but not eliminated by E2. |
| F6 robot control/planning | 3 | E_PHASE_1_FABRIC_NOMINAL_L_01_S408000, E_PHASE_1_FABRIC_NOMINAL_R_01_S408001, E_PHASE_1_FABRIC_NOMINAL_L_02_S408007 | Full task-branch PointGoal failures without independent label evidence; these are not relabeled BLOCKED. |
| F7 physical-model invalidity | 8 P1 physics-only runs | P1_SPLIT_CENTER_CURTAIN_no_probe, P1_SPLIT_CENTER_CURTAIN_probe, P1_LEFT_OPEN_RIGHT_CONSTRAINED_no_probe, P1_LEFT_OPEN_RIGHT_CONSTRAINED_probe, P1_RIGHT_OPEN_LEFT_CONSTRAINED_no_probe, P1_RIGHT_OPEN_LEFT_CONSTRAINED_probe | P1 runs failed the frozen physics sanity bound; they are excluded from G1 metrics, and real PVC is not claimed. |

F1 and F4 are reported as unproven rather than inferred from UNKNOWN rows. F2/F3 candidate counts intentionally expose the evidence boundary instead of converting local contact into a whole-body truth label.

## Execution Diagnostics

| Diagnostic | Count | Example trial IDs |
|---|---:|---|
| RUNTIME_INVALID | 0 | - |
| PROBE_INVALID_OR_MISSING_HISTORY | 0 | - |
| CONTACT_LIMITED_MOTION | 90 | E_PHASE_0_RIGID_CONTROL_L_01_S407372, E_PHASE_0_RIGID_CONTROL_R_01_S407373, E_PHASE_0_FABRIC_NOMINAL_L_01_S407000, E_PHASE_0_FABRIC_NOMINAL_R_01_S407001, E_PHASE_1_FABRIC_NOMINAL_L_01_S408000 |
| FREE_SPACE_UNDERTRACKING | 14 | E_PHASE_0_EMPTY_CONTROL_L_01_S407341, E_PHASE_0_EMPTY_CONTROL_R_01_S407342, E_MATCHED_CONTROL_WALL_GAP_02_MATCHED_NO_WALL_L_01_S411403, E_MATCHED_CONTROL_WALL_GAP_02_MATCHED_NO_WALL_R_01_S411404, E_MATCHED_CONTROL_WALL_GAP_02_MATCHED_NO_WALL_L_02_S411410 |
| SAFETY_STOP | 0 | - |
| TASK_BRANCH_POINTGOAL_FAILURE_WITHOUT_LABEL_EVIDENCE | 3 | E_PHASE_1_FABRIC_NOMINAL_L_01_S408000, E_PHASE_1_FABRIC_NOMINAL_R_01_S408001, E_PHASE_1_FABRIC_NOMINAL_L_02_S408007 |
| UNKNOWN_WITHOUT_INDEPENDENT_EVIDENCE | 60 | E_PHASE_0_FABRIC_NOMINAL_L_01_S407000, E_PHASE_0_FABRIC_NOMINAL_R_01_S407001, E_PHASE_1_FABRIC_NOMINAL_L_01_S408000, E_PHASE_1_FABRIC_NOMINAL_R_01_S408001, E_PHASE_1_FABRIC_NOMINAL_L_02_S408007 |

P1 physics-only failures are listed under F7 and cannot invalidate P0.
