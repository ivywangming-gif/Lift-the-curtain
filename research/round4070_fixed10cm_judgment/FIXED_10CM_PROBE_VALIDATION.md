# Fixed 10 cm Probe Validation

Status: **PASS**

The formal contract is one continuous `PROBE_10CM` action with a 0.10 m target from the verified precontact/preprobe pose. Actual FK displacement, post-contact displacement and stop reason are distinct fields.

| Check | Pass | Observed |
|---|---|---|
| manifest_count | PASS | `250` |
| all_probe_modes | PASS | `["PROBE_10CM"]` |
| all_target_displacements | PASS | `[0.1]` |
| core_condition_count | PASS | `10` |
| core_side_repeat_count | PASS | `[10]` |
| runner_rejects_old_depth_schedule | PASS | `True` |
| hidden_wall_gaps_inside_probe | PASS | `[0.02, 0.05, 0.08]` |
| bilateral_safety_validation | PASS | `[{"trial_id": "D_EMPTY_CONTROL_L_04_S410502", "summary_exists": true, "runtime_valid": true, "target_m": 0.1, "actual_m": 0.08235049247741699, "post_contact_m": null, "stop_reason": "PROBE_INCOMPLETE", "safety_stop": null, "safe_recovery": true, "joint_step_rad": 0.85, "protocol_match": true, "protocol_detail": "", "condition_id": "EMPTY_CONTROL", "probe_side": "left"}, {"trial_id": "D_EMPTY_CONTROL_R_04_S410520", "summary_exists": true, "runtime_valid": true, "target_m": 0.1, "actual_m": 0.08738696575164795, "post_contact_m": null, "stop_reason": "PROBE_INCOMPLETE", "safety_stop": null, "safe_recovery": true, "joint_step_rad": 0.85, "protocol_match": true, "protocol_detail": "", "condition_id": "EMPTY_CONTROL", "probe_side": "right"}, {"trial_id": "D_RIGID_CONTROL_L_04_S410672", "summary_exists": true, "runtime_valid": true, "target_m": 0.1, "actual_m": -0.0136910080909729, "post_contact_m": 0.004373013973236084, "stop_reason": "PROBE_INCOMPLETE", "safety_stop": null, "safe_recovery": true, "joint_step_rad": 0.85, "protocol_match": true, "protocol_detail": "", "condition_id": "RIGID_CONTROL", "probe_side": "left"}, {"trial_id": "D_RIGID_CONTROL_R_04_S410690", "summary_exists": true, "runtime_valid": true, "target_m": 0.1, "actual_m": 0.01232755184173584, "post_contact_m": 0.012447595596313477, "stop_reason": "PROBE_INCOMPLETE", "safety_stop": null, "safe_recovery": true, "joint_step_rad": 0.85, "protocol_match": true, "protocol_detail": "", "condition_id": "RIGID_CONTROL", "probe_side": "right"}, {"trial_id": "D_FABRIC_NOMINAL_L_04_S407102", "summary_exists": true, "runtime_valid": true, "target_m": 0.1, "actual_m": 0.07563197612762451, "post_contact_m": 0.075717031955719, "stop_reason": "PROBE_INCOMPLETE", "safety_stop": null, "safe_recovery": true, "joint_step_rad": 0.85, "protocol_match": true, "protocol_detail": "", "condition_id": "FABRIC_NOMINAL", "probe_side": "left"}, {"trial_id": "D_FABRIC_NOMINAL_R_04_S407120", "summary_exists": true, "runtime_valid": true, "target_m": 0.1, "actual_m": 0.08794575929641724, "post_contact_m": 0.08827453851699829, "stop_reason": "PROBE_INCOMPLETE", "safety_stop": null, "safe_recovery": true, "joint_step_rad": 0.85, "protocol_match": true, "protocol_detail": "", "condition_id": "FABRIC_NOMINAL", "probe_side": "right"}, {"empty_target_reached_bilateral": false, "empty_probe_fields_bilateral": true, "note": "The protocol treats 0.10 m as a target, not a forced actual displacement; PROBE_INCOMPLETE remains recorded and is excluded from passability labels."}]` |
| deployment_header_leakage | PASS | `{"D_FABRIC_NOMINAL_L_04_S407102": [], "D_FABRIC_NOMINAL_R_04_S407120": [], "D_EMPTY_CONTROL_L_02_S410434": [], "D_EMPTY_CONTROL_L_04_S410502": [], "D_EMPTY_CONTROL_R_04_S410520": [], "D_EMPTY_CONTROL_L_05_S410536": [], "D_RIGID_CONTROL_L_04_S410672": [], "D_RIGID_CONTROL_R_04_S410690": []}` |

Safety abort or incomplete target movement is not mapped to BLOCKED. Only the offline evaluation stage can assign a physical label, and it must retain the execution/safety certificate.

The old D0-D3 schedule is not accepted by the D runner.

The current symmetric two-joint reference is also bounded by the robot geometry: an independent FK sweep of the pinned G1 URDF gives a maximum forward projection of about 0.0889 m over the allowed step interval [0, 1.20] rad. Reaching 0.10 m would therefore require a separately preregistered probe-pose/trajectory calibration, not an unbounded push or a post hoc benchmark relabel.

A separate full-arm FK target was tested only as a calibration recheck outside the manifest: left actual=0.093505 m stop=PROBE_INCOMPLETE safe=True; right actual=0.100378 m stop=TARGET_REACHED safe=True. The right side reached the target, but the left side did not; this does not pass the bilateral formal-launch gate and is not a benchmark label.
