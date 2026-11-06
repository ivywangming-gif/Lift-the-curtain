# V4 recording and validity contract

The primary protocol is `G1_RNG_V1_BODY_COLLISION_AUDIT_V1`; the independent label-validity revision is `G_BODY_COLLISION_VALIDITY_V1`. The frozen action remains `G_HOLD_PROBE_POSE_FORWARD24_V1`. This revision repairs missing reset variation and missing actual robot collision evidence. It does not change FALCON weights, action parameters, safety limits, formal split roles or select configurations from learned predictions.

The importer has actually isolated 10 completed V3 pilots: EMPTY original P2, C01 original P2, D01 original B2, A01/A02 original UNKNOWN4. Their raw files and online labels are unchanged. C01's former P only established its original envelope/recovery condition; the independent supervised interpretation is UNKNOWN pending body collision evidence. All ten pilots, including controls, are excluded from primary supervision, response fitting, normalization, calibration and tests. The current primary manifest contains zero episodes. This is a validity result, not a measured learning result.

## Reset evidence

Before launch, the wrapper verifies the frozen randomization sidecar SHA and recreates the two PCG64 draws with `numpy.random.default_rng(plan.seed).uniform(-.001,.001,size=2)`. They must exactly match the frozen additive y/yaw values. The corresponding actual root pose is read from the PhysX articulation view, converting XYZW to WXYZ, and verified against the written reset within float32 precision.

After `sim.reset()` and before the first physics step, free-node xyz velocities are sampled in ascending global vertex order from the same generator, bounded by ±0.001 m/s. Every attachment vertex is exactly zero. Both the applied full float32 array and its immediate simulator readback are saved with fixed/free vertex IDs, array SHA, NPZ SHA and actual simulation time. Same seed, geometry and node ordering give the same perturbations across hands and paired robot variants. No seed, node ID, world pose or perturbation metadata enters model features.

Files: `g_reset_randomization.json` and `g_reset_randomization_arrays_audit_only.npz`. A plan declaring a seed is insufficient; the importer replays the actual saved draws and checks readback evidence.

## Timing and observations

The original loop remains 200 Hz physics, 50 Hz control and 10 Hz safety/geometry decisions. The legal robot JSONL and its 252 exported feature columns are unchanged. Previous action is copied before the next policy call; q_cmd is the target actually sent before the measured state. The single probe window ends before followup, uses real timestamps, and is capped at two seconds. No interpolation creates observations. Unvalidated acceleration remains a zero-plus-unavailable mask for every model.

The new evaluation-only recorder requests actual synchronized body-link poses and all cloth nodes/velocities at existing 50 Hz control ticks, phase-first ticks and exact final hold/recovery boundaries. Non-safety ticks perform fresh geometry reads; they do not repeat a cached 10 Hz node sample. Each row carries the legacy episode clock, actual SimulationContext clock, physics-step index, phase and boundary flag. A coincident periodic/final sample is retained once with its boundary marked. The standard 825-node scene requires approximately 60 MB per 60 simulated seconds for uncompressed float32 node positions plus velocities; actual runtime overhead awaits the revised G1 smoke.

`g_body_collision_history_audit_only.npz` contains `time_s`, `actual_simulation_time_s`, `physics_step`, `phase`, `boundary`, `body_names`, `body_positions_world`, `body_quaternions_wxyz`, `nodes_world` and `node_velocities_world`. These are audit truth, never deployment features. `g_body_collision_capture.json` records cadence, source, finalization and hashes.

## Actual collision geometry

`g_robot_collision_geometry.json` traverses actual CollisionAPI prims including instance proxies, enabled state, body ownership, relative transforms, primitive dimensions, mesh/cooking output, material/filter attributes and global CollisionGroup collections. The local-to-body matrix uses column vectors and is the transposed USD relative transform to the actual rigid-body ancestor; it is not inferred by subtracting a current simulation pose from a static USD transform. Body quaternions use WXYZ. Stage units, PhysX binding identity and the loaded exporter source SHA are recorded.

No visual bounding box, arbitrary sphere, guessed hull or authored mesh is substituted for unavailable actual collision geometry. The independent extractor diagnostics currently identify a real representation issue with CollisionAPI Xform prims; their geometry remains NOT_VERIFIED while physics investigates the native representation. A successful API call or varying signal alone is insufficient to validate its coordinate frame. Revised G1 collection remains paused pending this check.

## Independent admission and labels

The physics auditor writes a derived `g_collision_audit.json` with schema `round4070-g.body_collision_audit.v1`, source file hashes and one of three statuses:

- `VERIFIED`: supported actual geometry, complete required sampled phases, finite data and no interior penetration under the documented audit scope. An inconclusive action may then be a valid UNKNOWN response history.
- `NOT_VERIFIED`: missing, unsupported or insufficient evidence. Neither supervision nor response fitting is allowed.
- `PHYSICS_INVALID`: actual penetration or invalid physics. The episode is excluded.

The source filenames/hashes include the geometry JSON, synchronized history NPZ and original outcome. The importer preserves byte-exact SHA-named copies of derived audit/RNG JSON evidence; `source_files_base` identifies the original directory for relative references inside those copies. This permits reproducible frozen snapshots without rewriting raw outcomes. Action-evidence revision `G_FINITE_WALL_STATIC_B_GAP_V1` separately withdraws finite-wall static B conclusions: V1 has no invariant lateral/height tube, so the nominal central corridor obstruction does not prove complete closed-loop action impossibility. See ACTION_LABEL_GAP_REPORT.md.

P additionally requires actual recorded bounded advance, exact final `responsive_traverse_stand` collider clearance, original hold stability (speed gate, speed ≤0.06 m/s, finite/no fall or external safety failure) and subsequent safe release/recovery. The frozen candidate interval is center x ±0.06 m, so the clearance threshold is its back boundary plus 0.04 m: **center x +0.10 m**, or **0.950 m** for the standard 0.850 m plane. The independent collider minimum world x must meet this threshold; the old visual-envelope flag is diagnostic only. Finite-wall static B is formally UNKNOWN under the current V1 action, preserving its original raw label only for audit. D07 remains outside this near-field contract.

Discrete sampling is reported explicitly and is not described as a continuous-time nonpenetration proof. Contract/source tests and a V4 plan dry run have passed; they do not substitute for a revised actual G1 instrumented run or its independent physics audit.
