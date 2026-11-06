# Round4070-G actual dataset audit

Current importer view: **2** accepted real G histories, **0** excluded/unfinalized source directories. This is a partial dataset status, not benchmark completion.

Raw G files remain read-only. The model array contains only robot q/qdot/actually sent q_cmd/applied torque, actual previous control action, current policy action, body-local dual-hand FK and causal displacement, robot IMU/state, known commanded action and true observation intervals. All geometry, material, constraint node indices, contact truth, scene IDs, labels and followup outcomes remain in separate metadata. No absolute world position, map or goal channel is exported. Missing acceleration is explicitly zero-plus-availability-mask, never presented as an observed zero.

Physics runs at 200 Hz, policy at 50 Hz, geometry/progress safety at 10 Hz. New robot records use existing 50 Hz control ticks plus actual phase-first ticks; no interpolated observations. Actual per-episode timestamps determine Hz and dt. Only physical_probe/physical_probe_hold before the recorded cutoff and at most 2 s enter seq. All methods must select the same cutoff; .5/1 s prefixes cannot normalize by future final duration.

Official joint order comes from falcon_contract.py and is recorded in manifest; left/right arm columns are selected by joint name. q_cmd is the target already sent before the measured state, and previous_control_action is copied immediately before the next policy computation. Consecutive recorded control updates verify previous==preceding current. Arm commanded reference equals the actual q_cmd upper subset.

UNKNOWN is y=-1 and never a third supervised class. Physics-invalid or probe-invalid files do not enter valid arrays. Runtime/probe/label/task flags remain independent. Source F117 histories are listed separately with all old labels audit-only; their 10 Hz schema cannot silently fill unavailable G channels.

If the probe was independently valid but later followup execution failed, the finite physics-valid probe can remain response-only UNKNOWN; this does not assert runtime validity for the whole episode. Separate runtime/probe counts are recorded in dataset_status.json. Repeated actual attempts have separate IDs, retain the same physical group and frozen split role, and are never selected by best label. Existing exported NPZ arrays are compared and reused, preserving their hashes across partial imports.

Splits are fixed metadata rules, not optimized from labels. Every episode is atomic. Group holdouts collapse identical physical hashes across aliases/seeds/hands/poses. Only training UNKNOWN may enter response fitting or normalization. Material OOD remains unavailable unless actual material fidelity has independently been validated.

G E0/E1/E2 retain the historical frozen thresholds, but compute progress along the initial robot-body forward axis using the legal FK channels. This is an explicit G coordinate adaptation, not a bit-for-bit replay of the old world-x online rule. Empty-response references are derived separately from each split's training same-hand controls only; stall uses real timestamp left integration. Acceleration candidates remain audit-only throughout the frozen v3 model protocol because gravity/frame calibration has not been independently established.

Current source and validity inventory: `outputs/round4070_comprehensive_traversability/dataset/import_audit.json`. Feature schema and actual splits: `outputs/round4070_comprehensive_traversability/dataset/manifest.json`, `split_manifest.json`.
