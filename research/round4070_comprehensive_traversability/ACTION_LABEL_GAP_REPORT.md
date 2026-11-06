# Finite-wall action-label evidence gap

Revision: `G_FINITE_WALL_STATIC_B_GAP_V1`. The action remains `G_HOLD_PROBE_POSE_FORWARD24_V1`; no action parameters, original raw outcomes or physical scenes were changed by this correction.

The former geometric rule treated a fixed wall covering y ±0.35 m, z 0.05–1.60 m and the forward path as proof of BLOCKED. Actual D01–D04 walls are finite: y −0.60–0.60 m and z 0–2 m. Their x intervals are D01 [0.83,0.87], D02 [0.87,0.91], D03 [0.90,0.94], D04 [0.93,0.97] m. There are no side walls closing those routes.

The actual closed-loop action has no enforced world-y corridor or fixed global-yaw tube. It commands up to ±0.15 m/s body-frame lateral speed and ±0.35 rad/s yaw rate. Each bounded segment starts at the actual measured position and heads toward (1.3,0), retaining the actual yaw at that segment's start. Speed/rate limits and a return-to-center reference do not establish a positively invariant spatial tube under robot/contact dynamics. The precontact y ±0.10 m condition applies before probing, not throughout followup. The progress monitor constrains insufficient progress and roll/pitch, not cumulative lateral displacement. CandidateRegion's y bounds describe a map rectangle; the clearance predicate only checks x back extents plus stability.

R1 does not deliberately plan a detour. This does not prove that all realizable closed-loop motion stays within the central wall span. D01 additionally has an x-only rigid safety veto, which ignores y; that is a supervisory refusal rather than independent geometric impossibility. D02–D04 do not use that rigid-branch veto.

All D01–D04 frozen plans withhold the forward action via `--probe-validation-only`. The two actual old D01 pilot records have zero logged followup-advance steps. Their source BLOCKED results therefore are not observations of complete action failure. The physical fact established is obstruction of the nominal central passage, not impossibility of the full specified closed-loop action.

The importer now maps every existing V1/V4 static-wall BLOCKED outcome to **LABEL_UNKNOWN**, reason `FINITE_WALL_NO_CLOSED_LOOP_ACTION_IMPOSSIBILITY_PROOF`. Original labels, reasons, source hashes and source files are retained. Collision-verified UNKNOWN can only enter response fitting when all other primary admission requirements hold. Missing/unsupported collision evidence remains inadmissible even as UNKNOWN.

No negative class is fabricated to satisfy coverage goals. If this leaves no reliable BLOCKED supervision, the affected classification splits remain NOT_EVALUABLE. Old V3 pilots remain excluded from all primary learning independently of this correction. A future explicitly constrained action would require a new contract, new trial IDs, new admission/label versions and separate dataset statistics; its geometry proof cannot be applied retrospectively to V1.

Source references: `g1_recording.py:geometry_blocked_for_action`; `g1_trial.py:frozen_args`; frozen runner `run_walk_to_waypoint` and bounded-advance loop; `se2_feedback.py:PushPathTracker`; `progress_monitor.py`; `round22_state_machine.py:verify_candidate_clearance`; actual `physics_scene_builder.py` rigid-barrier construction. This was a source/metadata audit only, not a simulation or learned-performance experiment.
