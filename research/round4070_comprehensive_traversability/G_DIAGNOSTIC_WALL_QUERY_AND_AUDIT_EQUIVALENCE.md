# G diagnostic: native static wall queries and equivalent collision audit

Both wall runs below are isolated physics-only apparatus checks. They instantiate the frozen D01 Cube and ground, with **no robot, no FALCON, no G1 action and zero G1 budget consumption**. Their output directories are unique and both original attempts remain intact under `outputs/round4070_comprehensive_traversability/physics_diagnostics/`.

| Run | Actual result | Guard wall time | Cleanup timeout |
|---|---|---:|---|
| `actual_static_wall_v1_attempt01` | NOT_VERIFIED: all native queries returned no hits | 31.7376883 s | None |
| `actual_static_wall_v1_attempt02` | VERIFIED: actual static Cube, native overlap and six boundary faces agree | 32.8789162 s | None |

The first run read the actual enabled static Cube correctly but could not establish native query presence. Local primary code explains why: `IsaacLab/source/isaaclab/isaaclab/sim/simulation_cfg.py:377` defaults `enable_scene_query_support=False`, documents that the scene-query manager is not created when false, and describes raycasts/sweeps/overlaps as queries outside the physics engine. `simulation_context.py:230` automatically enables it only for GUI operation. This was an instrumentation precondition failure, not missing wall geometry or material instability.

The second diagnostic explicitly set only `SimulationCfg(enable_scene_query_support=True)` and read back `sim.get_physics_context().get_enable_scene_query_support() == True`. The local `isaacsim.core.api/physics_context/physics_context.py:695` setter only authors `PhysxSceneAPI.EnableSceneQuerySupport`; contact/solver/material/action parameters were unchanged. It observed `/World/GScene/RigidBarrier/geometry/mesh`, center x `0.8500000238418579`, size `0.04 × 1.2 × 2.0 m`, enabled CollisionAPI, no trigger, no dynamic ancestor and no animated transform. Native overlap reported the exact path once. All six native entry faces matched the analytic Cube within 10 micrometres, including normal and distance. Ray callbacks numbered `[1, 1, 1, 1, 2, 2]`, with the extra ground hits preserved and filtered by exact collision path.

The ray API returns a **boolean hit indicator**, whereas overlap returns a hit count. Production evidence stores the ray boolean separately from callback count. The multi-hit diagnostic and a CPU counterexample validate this distinction. Query presence alone is not relabeled as a native simulation flag: enabled CollisionAPI, no trigger, static ancestry, actual transforms, and native face agreement are separate prerequisites.

Production G1 enables this query facility only for V5 scenes with a wall or rigid board. No model observes the native query results. Per-trial source archives and query-configuration before/after/readback evidence distinguish this instrumentation revision. The native robot binder has no dependency on query support being false; its geometry/pose/volume/asset checks remain unchanged and it requires simulation shape membership without forbidding an additional query flag. Existing no-wall raw remains compatible. Frozen native libraries and proofs were not rewritten.

## Exact artifact hashes

| Artifact | SHA256 |
|---|---|
| attempt01 `summary.json` | `231e4bdee55bfc071e56671283ab7c381c6852991a847d3002ed532411ab596f` |
| attempt01 wall JSON | `bdd20b5b24e489a9a7e39cb01dfeefec33a0f4820336d9d0c05c62039f3f7a46` |
| attempt01 diagnostic source | `012561023a764f806621e6181a7d8131e9141965b9bed1291936db53a75f2f47` |
| attempt02 `summary.json` | `9b2b7b39b340fa17e807dc32a2823060daa8bb77cdd68f6abaa440da7aa8562d` |
| attempt02 wall JSON | `8de59e81cb2a344588e67d89c33980939be91ce32bfd6b0b93f862be3e95202f` |
| attempt02 diagnostic source | `ebf78a4e4a8b35b530cbcf1bf3b9ef3cb6951f107889bd8cf93d56cad70a20c0` |
| wall helper in both actual diagnostics | `dc7065c6694eb8f1e86f1891ff76b47edf1da76a0b0dcf55af997664d9a83247` |

Subsequent production snapshots additionally record the PhysxScene query-enable attribute directly. The two original diagnostic JSONs and archived helper versions remain unchanged; their actual flag readback is in attempt02's summary.

## Same-data audit equivalence

`protocol_repairs/auditor_closed_shortcircuit_v1/` preserves prior source bytes, hashes, a new derived audit, and `comparison.json`. The optimization first evaluates closed-solid intersection at zero tolerance. If that intersection is empty, stricter inward-offset intersections are also empty. Conservative envelopes need only the closed test. The collision geometry implementation, thresholds, initial-state requirement, sample coverage and label rules were not changed.

The completed real C01-left episode (`G_C01_left_coverage_00_nominal_s71900_rng1_act2/attempt01`) was re-audited through all 9110 poststep states and step0, including 248149 narrow-phase candidate tests. After excluding only `auditor_source_sha256`, **every result field was exactly equal**, with no changed non-source keys. The measured re-audit time was **149.7718261 seconds**. This is one measured run, not a claimed universal speedup. The original audit SHA is `ecb8a5d308581beb24998b35885cb0fcc6db6c410b2a97b960c503862b10f31e`; original raw and original audit were not modified.

The CPU suite after the short-circuit and ray-boolean fixes passed 44 tests. The later explicit query-manager-readback predicate passed all 7 wall evidence tests, including the disabled-manager rejection. Neither CPU tests nor these apparatus diagnostics substitute for an actual D01 G1 result.
