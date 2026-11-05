# Round 4070-E Contract Validation

Status: **PASS**

The nominal 0.10 m target is fixed in the command contract. Actual FK displacement is recorded as an outcome and is not a runtime/probe validity gate.

| Check | Result | Observed |
|---|---|---|
| manifest_count_within_budget | PASS | `112` |
| manifest_ids_unique | PASS | `112` |
| all_registered_probe_modes | PASS | `["PROBE_10CM"]` |
| all_registered_targets | PASS | `[0.1]` |
| fixed_command_contract | PASS | `"FIXED_ONE_SHOT_NOMINAL_0P10M"` |
| p0_condition_coverage | PASS | `["EMPTY_CONTROL", "FABRIC_BOTH_BOTTOM_CORNERS", "FABRIC_BOTTOM_LEFT_ANCHOR", "FABRIC_BOTTOM_RIGHT_ANCHOR", "FABRIC_FOUR_CORNERS_ONLY", "FABRIC_FULL_PERIMETER_FIXED", "FABRIC_HEAVY_CONSTRAINED", "FABRIC_HEAVY_FREE", "FABRIC_NOMINAL", "FABRIC_WALL_GAP_02", "FABRIC_WALL_GAP_05", "FABRIC_WALL_GAP_08", "RIGID_CONTROL"]` |
| phase0_minimum_six | PASS | `6` |
| phase1_two_seeds_per_side | PASS | `2` |
| hidden_wall_gap_set | PASS | `[0.02, 0.05, 0.08]` |
| old_depth_schedule_absent | PASS | `false` |
| actual_motion_not_validity_gate_in_runner | PASS | `true` |
| phase0_recorded_and_valid | PASS | `{"summary": 6, "runtime": 6, "probe": 6}` |
| phase1_formal_initial_screen_recorded | PASS | `{"registered": 44, "summary": 44, "runtime": 44, "probe": 44}` |
| deployment_header_leakage | PASS | `{}` |
| p0_physics_gates_pass | PASS | `{"A0_TOP_EDGE_ONLY": "PASS_E_PHYSICS", "A1_BOTTOM_LEFT_ANCHOR": "PASS_E_PHYSICS", "A1_BOTTOM_RIGHT_ANCHOR": "PASS_E_PHYSICS", "A1_TOP_EDGE_BOTTOM_TWO_CORNERS": "PASS_E_PHYSICS", "A2_FOUR_CORNERS_ONLY": "PASS_E_PHYSICS", "A3_FULL_PERIMETER_FIXED": "PASS_E_PHYSICS", "B_D225_A1": "PASS_E_PHYSICS", "B_D225_NEW": "PASS_E_PHYSICS", "C_GAP02": "PASS_E_PHYSICS", "C_GAP05": "PASS_E_PHYSICS", "C_GAP08": "PASS_E_PHYSICS"}` |
