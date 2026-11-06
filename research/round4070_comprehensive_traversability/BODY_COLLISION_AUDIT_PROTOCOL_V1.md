# Independent G1 body collision evidence

This protocol is frozen before any V4 primary G1 admission. Original V3 pilots and raw online labels remain unchanged and excluded from primary learning. Environment physics gates verify their probe apparatus; they do not establish robot body or hand nonpenetration.

## Required evidence

The evaluation-only recorder saves actual `body_link_pose_w` (WXYZ), all deformable nodes and velocities, physics steps and both clocks at the existing 50 Hz control ticks, plus phase-first and exact phase-final observations. This does not add safety or control decisions. A complete finalized history must have finite values, matching body order, strictly increasing time/steps, maximum interval 20 ms, and consistent 200 Hz physics timing. Probe and final release/recovery phases must be present.

The enabled USD CollisionAPI inventory, body mapping, authored local transforms, collision settings, groups, binding binary SHA and geometry source hashes are retained. A visual mesh, fallback sphere, body subtree bounding box, or unverified cooked coordinate frame cannot establish validity. Native OmniPVD recording is being tested as a read-only source of actual PhysX convex vertices, shape poses and mesh scales. Its extractor deliberately reports `EXTRACTED_PENDING_NATIVE_INVENTORY_VALIDATION`; an extraction alone is not an admission certificate. Every native shape, body mapping, units, affine frame, source asset and runtime identity still require verification.

## Geometric tests

All real surface triangles are transformed using synchronized link poses into each actual solid collider's body frame. Convex planes are recovered from complete native closed faces and verified against actual native vertices. Sphere/cylinder tests use their actual analytic dimensions only when the native representation and transform are verified. Unsupported shapes cause `NOT_VERIFIED`.

The main geometric invalidity condition is strict surface intersection beyond a 5 mm inward offset of an actual collider, with a 1 nm numerical strictness margin. Closed intersections and 0/1 mm inward sensitivity counts are retained separately. These thresholds are fixed before revised G1 data, not adjusted to obtain a positive label. Small primitives whose 5 mm inner region is empty retain their closed and finer sensitivity results. Wall faces are tested against actual colliders, and actual solid points wholly contained in the wall are checked to prevent a containment false negative. Every witness records sample, part, body, surface and location.

This is a discrete 50 Hz poststep audit plus exact phase boundaries. It is not continuous collision detection and does not prove absence of intersample penetration. It also does not establish real material contact fidelity or real robot safety. Ground contact and robot self-contact are outside this surface-versus-robot test; existing solver/safety constraints remain in force.

For candidate clearance, the actual minimum world X over every represented native collision part is evaluated at the final `responsive_traverse_stand` observation. Convex vertices and analytic primitive support are used; bounding boxes are not substituted. The fixed threshold is scene center X + 0.06 m + 0.04 m (nominal 0.95 m). Existing stand stability (finite, no fall, speed <= 0.06 m/s), actual advance and safe recovery requirements also remain necessary for PASSABLE. Missing clearance can leave an otherwise verified episode UNKNOWN; missing body collision evidence cannot enter primary response or supervised learning.

## Admission and recovery

`VERIFIED` means the complete recorded evidence passed this protocol. A witnessed strict intersection is `PHYSICS_INVALID` and independently blocks further G1 launches for that scene pending scientific review. Missing or unsupported evidence is `NOT_VERIFIED`, creates `PAUSE_G1.flag`, and allows independent environment physics gates to continue. A complete raw episode is re-audited before considering recapture; an explicit source-linked repair and reason why raw cannot be repaired are required for at most attempt 02. No result selection seeks a better label.

Every obstacle G1 condition also requires an actual V4 EMPTY episode with the same hand, posture variant and robot parameters. That EMPTY must pass runtime/probe/recovery, actual seeded reset readback, this independent collision audit and final actual-collider clearance. EMPTY itself is exempt from this dependency. Failed or pending hand/variant controls withhold only the matching obstacle branch. The first revised EMPTY pair remains a bounded smoke stage for independent review.

Derived audit versions are stored as `g_collision_audit_<SHA256>.json` and never replaced. `g_collision_audit.json` is only the current pointer content. Dataset snapshots bind immutable audit versions and exact raw source hashes. Campaign reports distinguish raw online outcome labels from audited primary labels.

## Actual implementation checks

Eight admission tests and six audit tests passed on 2026-11-06. These are contract/math tests, not research episodes. The native extraction diagnostic and full V4 G1 validation remain pending; no primary episode has been admitted by these tests.
