# BIMANUAL_V1 candidate assets

The JSON files in scene_specs/ and OBJ files in static_obj_candidates/ are
offline, inspectable candidate geometry and configuration descriptions for
Round 4070-G. They are not native USD assets, do not prove the runtime
PxShape/PxConvexMesh binding, and do not count as a physics gate.

Every candidate is fail-closed until all of the following are independently
verified: three physics seeds, rest/contact/release/recovery behavior,
collision and penetration evidence, synchronized bimanual action safety, and
the current B10/V6 safety approval. Real-material rows intentionally remain
marked as missing external assets; proxy rows must not be reported as real
cloth, mesh, or PVC validation.

The generator is scripts/round4070_g/audit_bimanual_scene_matrix.py.
Re-running it is an offline operation and preserves the frozen V5/V6 campaign
and all historical trials.
