# Round4070-G V5 actual discrete-state collision audit

This protocol supersedes the admission semantics of the V3 pilots and V4 proposal. Original raw artifacts and frozen manifests remain unchanged. V5 is not a claim of continuous-time collision freedom or real material calibration.

## Native geometry and frame evidence

The original robot asset is unchanged. The actual PhysX native recording contains 39 simulation-enabled shapes: 27 convex meshes, 8 spheres, and 4 zero-margin native core cylinders. Of the convex meshes, 15 have consistent native faces and 12 have inconsistent face halfspaces. The latter retain their original native vertices/faces and use a full-native-vertex convex hull only as a conservative outer envelope. Thus 27 shapes are exact and 12 are separation-only envelopes. Envelope overlap is NOT_VERIFIED, never a penetration witness.

`g1_native_collision_library_v3.json` and `g1_native_geometry_validation_v2.json` bind the native inventory to four independently recorded actual link-pose states. Raw native actor and shape poses are used; USD display rotations are diagnostics. Known compound shapes independently test child translation, rotation, and nonuniform scale. Native core cylinders use the source-confirmed native X axis before the actual shape transform. The 27 mesh scales are isotropic, so the native writer's omitted mesh-scale rotation does not change their geometry.

Every trial checks original asset-layer hashes, PhysX binary identity, the IsaacLab link-frame implementation, exact body-name order, all enabled collider paths, static collision/material attributes and relationships, local transforms, and all native-query shape fingerprints without path deduplication. The fixed action witness also requires the exact frozen body index, native shape path, part hash, point, and interior radius. The independent offline audit repeats these checks from immutable raw evidence and the archived query source.

## Recorded-state completeness

The measured reset/RNG state is recorded separately at step 0. Every subsequent actual 200 Hz physics state is recorded in immutable chunks of at most 500 states. Model observations and the controller remain at 50 Hz; safety callbacks retain their existing rate. There is no interpolation or synthesized sample.

The auditor checks every chunk hash, contiguous absolute steps 1 through N, independent execution anchors, actual simulation-clock consistency, body/node coverage, finite values, and real phase execution counts. A zero-step phase cannot supply a final sample. All chunks must be consumed before history is VERIFIED. Geometry must be audited at step 0 and every step 1 through N: `geometry_checked_state_count = sample_count + 1` and `initial_state_collision_audited = true` are mandatory admission conditions.

## Collision evidence and tolerances

The main strict penetration threshold is 5 mm; surface/body sensitivity counts at 0 and 1 mm are also retained. Distinct sufficient witnesses are named explicitly:

- An actual cloth or wall surface point more than 5 mm inside an exact body shape is a body-depth witness.
- An exact native solid surface point, or a primitive's true interior center, inside the wall's open 5 mm-eroded interior is a one-sided wall-depth witness. This proves positive-volume solid overlap. It does not claim a common 5 mm ball and includes a small sphere fully buried in a wall.
- Exact convex versus wall common-interior-ball optimization supplements boundary tests, including coincident and containment cases. Its verified radius must strictly exceed 5 mm.
- The fixed action witness has an independently verified exact native inner ball of radius 51.8855 mm, exceeding the required 6 mm. When this actual point lies in a wall's 5 mm-eroded interior, it supplies an additional solid-overlap witness.

Numerical uncertainty, unsupported geometry, or conservative-envelope overlap yields NOT_VERIFIED. Exact penetration yields PHYSICS_INVALID. Neither is admitted as supervised UNKNOWN. Complete verified geometry with an inconclusive action can remain a valid UNKNOWN response episode.

## Fixed action and completion

V5 uses `G_HOLD_PROBE_POSE_FORWARD24_TUBE_V2`. The independent replay checks the actual final probe-hold entry, every subsequent actual step until first release/retract, the fixed witness y/z bounds, and maximum 20 mm x displacement per 5 ms step. Runtime first-violation reason, step, and point must match independent replay. A latched guard violation is UNKNOWN, never a blocking proof. Recovery cannot be followed by new advance/hold phases.

PASSABLE still requires the existing stable final hold and safe recovery, plus all actual collision shapes' rear support beyond candidate center + 0.10 m. For conservative shapes, minimum x is a lower bound, sufficient for clearance but not an exact surface witness. The actual final hold sample is used, never an earlier transient crossing.

A fixed-wall BLOCKED certificate additionally needs the wall to cover the entire frozen witness tube with strict margins, the eroded wall width to exceed the maximum x step, actual entry before the wall, and unchanged completion to require reaching beyond it. Far/out-of-probe-range scenes do not obtain this certificate. Original V1/V4 finite straight-corridor obstruction is not this V5 proof.

## Immutable results and resource handling

Each audit result is preserved as `g_collision_audit_<sha256>.json`; the current entry is derived and may point to a newer audit. Raw states are never rewritten. Re-audit precedes any bounded implementation recapture. Genuine physical invalidity blocks that scene, and an implementation-wide admission gap pauses only G1 while independent physics gates continue.

The campaign reserves at least 20 GiB free output-disk space before each launch, uses the shared physics/training resource lock, and counts historical as well as new actual G1 launches against the limit of 1000. No raw files are deleted to make space.
