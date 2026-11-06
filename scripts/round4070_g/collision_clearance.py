"""World-X support or certified lower bounds for actual collision parts.

Inputs are actual baked metre geometry in the corresponding rigid-body frame.
The upstream collider dump must certify cooked-frame identity and completeness;
this module cannot infer either from a plausible array. No visual/proxy bounds
or surrogate spheres are used. Unknown/omitted geometry is NOT_VERIFIED.
"""
from __future__ import annotations

import numpy as np
from collision_geometry import GeometryNotVerified,prepare_collider


def _rotations(quaternions_wxyz, tolerance):
    q = np.asarray(quaternions_wxyz,dtype=float)
    if q.ndim != 2 or q.shape[1] != 4 or not np.isfinite(q).all():
        raise GeometryNotVerified("Expected finite body quaternions [B,4] in wxyz order")
    norms = np.linalg.norm(q,axis=1)
    if np.any(norms <= 0.) or np.any(np.abs(norms-1.) > tolerance):
        raise GeometryNotVerified("Non-unit body quaternion; do not silently repair a pose/frame error")
    q = q/norms[:,None]  # only accepted floating-point drift, recorded below
    w,x,y,z = q.T
    rotation = np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                         2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                         2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),axis=1).reshape(-1,3,3)
    return rotation,norms


def actual_collider_min_world_x(parts, body_positions_world, body_quaternions_wxyz, *, expected_body_indices=None, quaternion_norm_tolerance=1e-4):
    """Return the actual minimum world X, a witness and every part's result.

Each part has body_index and geometry (a prepare_* product). Convex parts also
provide vertices_body_m: the COMPLETE actual cooked convex vertex list after
the verified affine mesh-to-body transform. Repeated parts on one body remain
separate. expected_body_indices, when supplied, enforces exact body coverage;
without it the result certifies only the supplied list, not whole-robot coverage.
"""
    if not isinstance(parts,(list,tuple)) or not parts:
        raise GeometryNotVerified("No actual collision parts; clearance is not verified")
    positions = np.asarray(body_positions_world,dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions) or not np.isfinite(positions).all():
        raise GeometryNotVerified("Expected finite body positions [B,3]")
    tolerance = float(quaternion_norm_tolerance)
    if not np.isfinite(tolerance) or not 0 < tolerance <= 1e-3:
        raise GeometryNotVerified("Invalid quaternion norm tolerance")
    rotations,norms = _rotations(body_quaternions_wxyz,tolerance)
    if len(rotations) != len(positions):
        raise GeometryNotVerified("Pose arrays have different body counts")
    indices = []
    for part in parts:
        if not isinstance(part,dict):
            raise GeometryNotVerified("Invalid collision part description")
        index = part.get("body_index")
        if isinstance(index,bool) or not isinstance(index,(int,np.integer)) or not 0 <= int(index) < len(positions):
            raise GeometryNotVerified("Missing/out-of-range actual collider body index")
        indices.append(int(index))
    covered = sorted(set(indices))
    expected = None
    if expected_body_indices is not None:
        if any(isinstance(i,bool) or not isinstance(i,(int,np.integer)) or not 0 <= int(i) < len(positions) for i in expected_body_indices):
            raise GeometryNotVerified("Invalid expected collision-body indices")
        expected = sorted(set(int(i) for i in expected_body_indices))
        if covered != expected:
            raise GeometryNotVerified(f"Collision-body coverage differs: expected {expected}, received {covered}")
    per_part = []
    for part_index,(part,body_index) in enumerate(zip(parts,indices)):
        if "geometry" not in part:
            raise GeometryNotVerified("Actual collider geometry is missing")
        source_geometry = part["geometry"]
        conservative = (source_geometry.get("exact_intersection_allowed") is False
            or "CONSERVATIVE" in str(source_geometry.get("evidence_mode", part.get("evidence_mode", "")))
            or part.get("exact_intersection_allowed") is False)
        geometry = prepare_collider(source_geometry)
        rotation,position = rotations[body_index],positions[body_index]
        direction = rotation[0]  # world +X expressed in the rigid body's frame
        kind = geometry["type"]
        vertex_index = None
        if kind == "convex":
            if "vertices_body_m" not in part:
                raise GeometryNotVerified("Convex clearance requires actual complete cooked vertices; AABB is not a substitute")
            vertices = np.asarray(part["vertices_body_m"],dtype=float)
            if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4 or not np.isfinite(vertices).all():
                raise GeometryNotVerified("Invalid actual cooked convex vertex list")
            planes = geometry["planes"]
            if np.max(vertices @ planes[:,:3].T+planes[:,3]) > geometry.get("validation_tolerance_m",1e-6):
                raise GeometryNotVerified("Cooked vertices and actual collider planes disagree in body frame")
            vertex_index = int(np.argmin(vertices @ direction))
            witness_body = vertices[vertex_index]
        elif kind == "sphere":
            witness_body = geometry["center"]-geometry["radius"]*direction
        elif kind == "cylinder":
            axis = np.asarray(geometry["axis"],dtype=float)
            if axis.shape != (3,) or not np.isfinite(axis).all() or abs(np.linalg.norm(axis)-1.) > 1e-8:
                raise GeometryNotVerified("Cylinder axis must be a verified unit body-frame vector")
            axial = float(np.clip(direction @ axis,-1.,1.))
            radial = direction-axial*axis
            radial_length = float(np.linalg.norm(radial))
            witness_body = geometry["center"]-geometry["half_height"]*np.sign(axial)*axis
            if radial_length > 1e-12:
                witness_body = witness_body-geometry["radius"]*radial/radial_length
        else:
            raise GeometryNotVerified(f"Unsupported actual collider type {kind!r}")
        witness_world = rotation @ witness_body+position
        if not np.isfinite(witness_world).all():
            raise GeometryNotVerified("Nonfinite actual-collider support witness")
        per_part.append(dict(part_index=part_index,body_index=body_index,geometry_type=kind,
            collider_path=part.get("collider_path",part.get("path")),min_world_x_m=float(witness_world[0]),
            witness_world_m=witness_world.tolist(),witness_body_m=witness_body.tolist(),cooked_vertex_index=vertex_index,
            min_world_x_evidence="CONSERVATIVE_LOWER_BOUND" if conservative else "EXACT_SUPPORT",
            witness_is_actual_surface_certified=not conservative))
    minimum = min(per_part,key=lambda result:result["min_world_x_m"])
    all_exact = all(p["min_world_x_evidence"] == "EXACT_SUPPORT" for p in per_part)
    return dict(min_world_x_m=minimum["min_world_x_m"],witness_world_m=minimum["witness_world_m"],min_part_index=minimum["part_index"],
        min_world_x_evidence="EXACT_SUPPORT" if all_exact else "CONSERVATIVE_LOWER_BOUND",
        all_parts_exact_support=all_exact,
        per_part=per_part,part_count=len(parts),body_indices_covered=covered,expected_body_indices=expected,
        all_supplied_parts_evaluated=len(per_part)==len(parts),all_expected_bodies_covered=True if expected is not None else None,
        coverage_scope="exact expected collision-body set" if expected is not None else "ALL_SUPPLIED_PARTS_ONLY; whole-robot completeness must be established upstream",
        max_input_quaternion_norm_error=float(np.max(np.abs(norms-1.))),quaternion_norm_tolerance=tolerance,
        units="metres",coordinate_contract="verified baked body-frame actual geometry; body pose world position and wxyz quaternion",
        source_scope="complete native convex vertices / analytic primitives; flagged outer envelopes give only conservative lower bounds; no continuous-time collision claim")
