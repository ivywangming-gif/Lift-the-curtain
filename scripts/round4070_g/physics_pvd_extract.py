"""Read native OmniPVD recording USD; never load or modify a simulation.

Use the output of IPhysXPvd.ovd_to_usd, NOT ConvertOmniPvdToPhysXUSD.
The latter creates a new physics scene and is not the recorded geometry.
Imports of USD are deferred so this file's numerical contracts run without Kit.
An extraction is deliberately pending: the caller must verify the original
asset, native shape inventory, actor/link mapping and an independent frame test.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import itertools
import json
import math
import numpy as np


SCHEMA = "round4070-g.native_omnipvd_geometry.v1"
PARSER_REVISION = 5
MATCHING_RELEASE_SOURCE = "https://github.com/NVIDIA-Omniverse/PhysX/blob/107.3-omni-and-physx-5.6.1/physx/source/physx/src/omnipvd/OmniPvdPxSampler.cpp#L525-L591"
ACTOR_CLASSES = {"PxActor", "PxRigidDynamic", "PxRigidStatic", "PxArticulationLink"}
INVARIANCE_TOLERANCE_M = 1e-6


class ExtractionNotVerified(ValueError):
    pass


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _serial(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _serial(v) for k, v in value.items()}
    if hasattr(value, "tolist"):
        return value.tolist()
    try:
        return [_serial(v) for v in value]
    except TypeError:
        return str(value)


def _get(prim, name, time=None):
    attr = prim.GetAttribute(name)
    return attr.Get(time) if attr and time is not None else attr.Get() if attr else None


def _class(prim):
    return str(_get(prim, "omni:pvdi:class") or "")


def _native_name(prim):
    attr = prim.GetAttribute("omni:pvd:name")
    if not attr:
        return None
    values = []
    default = attr.Get()
    if default is not None:
        values.append(str(default))
    values.extend(str(attr.Get(t)) for t in attr.GetTimeSamples())
    names = sorted(set(values))
    if len(names) != 1:
        raise ExtractionNotVerified(f"Missing/changing native actor name at {prim.GetPath()}: {names}")
    return names[0]


def _finite_matrix(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ExtractionNotVerified("Nonfinite affine transform")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-10, rtol=0):
        raise ExtractionNotVerified("Non-affine geometry transform")
    if abs(np.linalg.det(matrix[:3, :3])) < 1e-12:
        raise ExtractionNotVerified("Singular geometry transform")
    return matrix


def _native_pose_matrix(raw):
    """Native PxTransform arrays are [qx,qy,qz,qw,px,py,pz].

Normalize only float32-sized unit drift; never use the converted display
quaternion, whose near-identity angle conversion can magnify that drift.
"""
    values = np.asarray(raw, dtype=float)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ExtractionNotVerified("Missing/nonfinite raw native PxTransform")
    norm = float(np.linalg.norm(values[:4]))
    if abs(norm-1.) > 1e-4:
        raise ExtractionNotVerified("Raw native PxTransform quaternion is not unit")
    x,y,z,w = values[:4]/norm
    result = np.eye(4)
    result[:3,:3] = [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    result[:3,3] = values[4:]
    return result, norm


def _transform_points(points, matrix, meters_per_unit):
    matrix = _finite_matrix(matrix)
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ExtractionNotVerified("Invalid actual native points")
    return (points @ matrix[:3, :3].T + matrix[:3, 3]) * meters_per_unit


def _convex_from_faces(points_body_m, face_counts, face_indices):
    """Recover planes from all recorded native polygon faces, not a proxy hull.

Reject missing/open/nonconvex geometry. Exact duplicate vertex records are
merged for the closed-manifold check; no sampled or visual vertices are added.
"""
    points = np.asarray(points_body_m, dtype=float)
    counts = np.asarray(face_counts, dtype=np.int64)
    indices = np.asarray(face_indices, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4 or not np.isfinite(points).all():
        raise ExtractionNotVerified("Invalid native convex points")
    if counts.ndim != 1 or len(counts) < 4 or np.any(counts < 3) or indices.ndim != 1 or counts.sum() != len(indices):
        raise ExtractionNotVerified("Native face counts/indices incomplete")
    if np.any(indices < 0) or np.any(indices >= len(points)):
        raise ExtractionNotVerified("Native face index outside point array")
    unique, first, inverse = np.unique(points, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)
    vertices = unique[order]
    remap = np.empty(len(order), dtype=int)
    remap[order] = np.arange(len(order))
    inverse = remap[inverse]
    if np.linalg.matrix_rank(vertices - vertices.mean(axis=0), tol=1e-10) < 3:
        raise ExtractionNotVerified("Native convex is not three-dimensional")
    if set(inverse[indices].tolist()) != set(range(len(vertices))):
        raise ExtractionNotVerified("Native convex has unreferenced vertices")
    center = vertices.mean(axis=0)
    edges = Counter()
    triangles = []
    planes = []
    offset = 0
    signed_volume = 0.0
    for count in counts:
        polygon = inverse[indices[offset:offset + count]]
        offset += int(count)
        if len(set(polygon.tolist())) != len(polygon):
            raise ExtractionNotVerified("Repeated vertex within a native polygon")
        for a, b in zip(polygon, np.roll(polygon, -1)):
            edges[tuple(sorted((int(a), int(b))))] += 1
        face_plane = None
        for j in range(1, len(polygon) - 1):
            tri = [int(polygon[0]), int(polygon[j]), int(polygon[j + 1])]
            a, b, c = vertices[tri]
            normal = np.cross(b - a, c - a)
            length = float(np.linalg.norm(normal))
            if length <= 1e-14:
                raise ExtractionNotVerified("Degenerate native triangle")
            normal /= length
            distance = -float(normal @ a)
            if normal @ center + distance > 0:
                normal = -normal
                distance = -distance
                tri[1], tri[2] = tri[2], tri[1]
            plane = np.r_[normal, distance]
            if np.max(vertices @ normal + distance) > INVARIANCE_TOLERANCE_M:
                raise ExtractionNotVerified("Recorded native mesh is nonconvex or incomplete")
            if face_plane is not None and np.max(np.abs(vertices[polygon] @ face_plane[:3] + face_plane[3])) > INVARIANCE_TOLERANCE_M:
                raise ExtractionNotVerified("Nonplanar recorded polygon")
            face_plane = plane
            if not any(np.allclose(plane[:3], p[:3], atol=1e-8, rtol=0) and abs(plane[3] - p[3]) <= 1e-8 for p in planes):
                planes.append(plane)
            triangles.append(tri)
            a, b, c = vertices[tri] - center
            signed_volume += float(a @ np.cross(b, c)) / 6
    if not edges or any(v != 2 for v in edges.values()):
        raise ExtractionNotVerified("Native convex polygon mesh is open or nonmanifold")
    if len(planes) < 4 or not math.isfinite(signed_volume) or signed_volume <= 0:
        raise ExtractionNotVerified("Native convex has invalid closed volume")
    return dict(type="convex", vertices_body_m=vertices.tolist(), planes=np.asarray(planes).tolist(),
                triangles_unique_vertices=triangles, volume_m3=signed_volume,
                aabb_body_m=[vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
                closed_manifold_edges_verified=True, original_point_count=len(points),
                unique_point_count=len(vertices), original_face_count=len(counts))


def _uniform_scale(matrix):
    linear = _finite_matrix(matrix)[:3, :3]
    scales = np.linalg.norm(linear, axis=0)
    if not np.allclose(scales, scales[0], atol=1e-8, rtol=1e-7) or not np.allclose(linear.T @ linear, np.eye(3) * scales[0] ** 2, atol=1e-8, rtol=1e-7):
        raise ExtractionNotVerified("Native primitive nonuniform scale/shear unsupported")
    return float(scales[0])


def _native_vertex_outer_envelope(points_body_m, counts, indices, exact_failure):
    """Conservative hull of COMPLETE recorded native points; separation only.

Native OVD triangles can disagree with convexity. This branch does not repair
or certify those triangles as an exact solid. Its convex hull contains every
recorded triangle, so disjointness is sufficient; overlap remains UNKNOWN.
"""
    from scipy.spatial import ConvexHull
    points = np.asarray(points_body_m, dtype=float)
    counts = np.asarray(counts, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all() or counts.ndim != 1 or np.any(counts < 3) or indices.ndim != 1 or counts.sum() != len(indices) or not len(counts):
        raise ExtractionNotVerified("Cannot bound incomplete native mesh arrays")
    if np.any(indices < 0) or np.any(indices >= len(points)):
        raise ExtractionNotVerified("Cannot bound invalid native mesh indices")
    unique, first, inverse = np.unique(points, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)
    vertices = unique[order]
    remap = np.empty(len(order), dtype=int); remap[order] = np.arange(len(order))
    inverse = remap[inverse]
    if set(inverse[indices].tolist()) != set(range(len(vertices))):
        raise ExtractionNotVerified("Native outer envelope has unreferenced points")
    edges = Counter(); original_triangles = []; violation = 0.; min_area = math.inf; worst_area = None
    center = vertices.mean(axis=0); offset = 0
    for count in counts:
        polygon = inverse[indices[offset:offset + count]]; offset += int(count)
        if len(set(polygon.tolist())) != len(polygon):
            raise ExtractionNotVerified("Repeated native polygon vertex")
        for a, b in zip(polygon, np.roll(polygon, -1)):
            edges[tuple(sorted((int(a), int(b))))] += 1
        for j in range(1, len(polygon) - 1):
            tri = [int(polygon[0]), int(polygon[j]), int(polygon[j + 1])]
            a, b, c = vertices[tri]; normal = np.cross(b - a, c - a); area2 = float(np.linalg.norm(normal))
            if area2 <= 1e-14:
                raise ExtractionNotVerified("Degenerate native face cannot establish a complete outer envelope source")
            normal /= area2; distance = -float(normal @ a)
            if normal @ center + distance > 0: normal = -normal; distance = -distance
            excess = float(np.max(vertices @ normal + distance))
            if excess > violation: violation = excess; worst_area = area2 / 2
            min_area = min(min_area, area2 / 2); original_triangles.append(tri)
    if not edges or any(v != 2 for v in edges.values()):
        raise ExtractionNotVerified("Native triangle source is open/nonmanifold; completeness not established")
    hull = ConvexHull(vertices)  # No QJ perturbation, no PhysX recooking, no visual vertices.
    planes = np.asarray(hull.equations, dtype=float)
    excess = float(np.max(vertices @ planes[:, :3].T + planes[:, 3]))
    if excess > 1e-9:
        raise ExtractionNotVerified("Native-vertex outer hull does not contain all source points")
    return dict(type="convex", vertices_body_m=vertices.tolist(), planes=planes.tolist(),
                triangles_unique_vertices=hull.simplices.tolist(), original_native_triangles_unique_vertices=original_triangles,
                volume_m3=float(hull.volume), aabb_body_m=[vertices.min(0).tolist(), vertices.max(0).tolist()],
                closed_manifold_edges_verified=True, original_point_count=len(points), unique_point_count=len(vertices), original_face_count=len(counts),
                evidence_mode="CONSERVATIVE_NATIVE_VERTEX_OUTER_ENVELOPE_SEPARATION_ONLY", exact_intersection_allowed=False,
                inside_witness_allowed=False, min_world_x_evidence="CONSERVATIVE_LOWER_BOUND",
                outer_envelope_derivation="ConvexHull of all exact-deduplicated recorded native PxConvexMesh points; original topology retained; no point perturbation",
                native_face_inconsistency=dict(exact_rejection=exact_failure, maximum_vertex_outside_face_plane_m=violation,
                    minimum_native_triangle_area_m2=min_area, worst_violating_face_area_m2=worst_area,
                    envelope_point_containment_max_error_m=excess),
                overlap_semantics="NOT_VERIFIED: an outer-envelope intersection is not evidence of actual penetration")


def _descendants_of_shape(shape, Usd):
    return [p for p in Usd.PrimRange(shape, Usd.TraverseInstanceProxies()) if p != shape]


def _native_geometry_prim(geom):
    current = geom
    while current and _class(current.GetParent()) != "PxShape":
        current = current.GetParent()
    if not current:
        raise ExtractionNotVerified("Native geometry has no owning PxShape")
    return current


def _verify_native_handles(shape, geom, prims, time, UsdGeom):
    """Check recorded PxShape -> geometry -> PxConvexMesh handle provenance."""
    native_geometry = _native_geometry_prim(geom)
    shape_geometry = _get(shape, "omni:pvdh:geom", time)
    geometry_handle = _get(native_geometry, "omni:pvdi:handle")
    if shape_geometry is None or geometry_handle is None or int(shape_geometry) != int(geometry_handle):
        raise ExtractionNotVerified("PxShape native geometry handle chain is broken")
    result = dict(shape_geometry_handle=int(shape_geometry), native_geometry_handle=int(geometry_handle))
    if geom.IsA(UsdGeom.Mesh):
        if _class(native_geometry) not in {"PxGeomConvexMesh", "PxConvexMeshGeometry"}:
            raise ExtractionNotVerified("Mesh parent is not a native PxGeomConvexMesh")
        handle = _get(native_geometry, "omni:pvdh:convexMesh", time)
        matches = [p for p in prims if _class(p) == "PxConvexMesh" and _get(p, "omni:pvdi:handle") == handle]
        if handle is None or len(matches) != 1:
            raise ExtractionNotVerified("Native convexMesh handle does not uniquely identify source PxConvexMesh")
        original = matches[0]
        for name in ("points", "faceVertexCounts", "faceVertexIndices"):
            composed_value, original_value = _get(geom, name, time), _get(original, name, time)
            if composed_value is None or original_value is None or not np.array_equal(np.asarray(composed_value), np.asarray(original_value)):
                raise ExtractionNotVerified(f"Referenced native mesh {name} differs from original recorded PxConvexMesh")
        result.update(native_convex_mesh_handle=int(handle), native_convex_mesh_source_path=str(original.GetPath()))
        scale = _get(native_geometry, "omni:pvd:scale", time)
        if scale is not None:
            scale = np.asarray(scale, dtype=float)
            if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ExtractionNotVerified("Unsupported native mesh scale")
            isotropic = bool(np.allclose(scale, scale[0], atol=1e-12, rtol=0))
            result.update(native_recorded_mesh_scale=scale.tolist(), native_mesh_scale_isotropic=isotropic,
                mesh_scale_rotation_evidence="IRRELEVANT_FOR_ISOTROPIC_SCALE" if isotropic else "NOT_RECORDED_BY_MATCHING_RELEASE_WRITER_REQUIRES_INDEPENDENT_TRANSFORM_TEST")
        else:
            result["mesh_scale_rotation_evidence"] = "NO_NATIVE_SCALE_ARRAY_LEGACY_RECORDING_REQUIRES_INDEPENDENT_TRANSFORM_TEST"
    elif _class(native_geometry) == "PxConvexCoreGeometry":
        margin = _get(native_geometry, "omni:pvd:margin", time)
        if margin is None or float(margin) != 0.0:
            raise ExtractionNotVerified("Rounded convex core margin is unsupported; cannot substitute an ordinary cylinder")
        handle = _get(native_geometry, "omni:pvdh:core", time)
        matches = [p for p in prims if _class(p) == "PxConvexCoreCylinder" and _get(p, "omni:pvdi:handle") == handle]
        if handle is None or len(matches) != 1 or not geom.IsA(UsdGeom.Cylinder):
            raise ExtractionNotVerified("Native convex core is not a uniquely identified cylinder")
        original = matches[0]
        for name in ("radius", "height", "axis", "omni:pvd:radius", "omni:pvd:height"):
            a, b = _get(geom, name, time), _get(original, name, time)
            if a is None or b is None or a != b:
                raise ExtractionNotVerified(f"Referenced native cylinder {name} differs from source core")
        result.update(native_core_handle=int(handle), native_core_class="PxConvexCoreCylinder", native_core_source_path=str(original.GetPath()), native_core_margin=0.0)
    return result


def _geometry_at(geom, actor, time, meters_per_unit, UsdGeom):
    cache = UsdGeom.XformCache(time)
    actor_world = _finite_matrix(np.asarray(cache.GetLocalToWorldTransform(actor), dtype=float).T)
    if not np.allclose(actor_world[:3, :3].T @ actor_world[:3, :3], np.eye(3), atol=1e-5, rtol=0) or np.linalg.det(actor_world[:3, :3]) <= 0:
        raise ExtractionNotVerified("Native actor frame is not a rigid link frame")
    geom_world = _finite_matrix(np.asarray(cache.GetLocalToWorldTransform(geom), dtype=float).T)
    display_matrix = _finite_matrix(np.linalg.inv(actor_world) @ geom_world)
    shape = _native_geometry_prim(geom).GetParent()
    raw_local_pose = _get(shape, "omni:pvd:localPose", time)
    native_local, local_norm = _native_pose_matrix(raw_local_pose)
    # Actor display errors cancel in relative geometry, but local shape display
    # rotations can independently have the same issue. Use the recorded pose.
    geometry_to_shape = _finite_matrix(np.asarray(cache.ComputeRelativeTransform(geom, shape)[0], dtype=float).T)
    matrix = _finite_matrix(native_local @ geometry_to_shape)
    native_actor_world, actor_norm = _native_pose_matrix(_get(actor, "omni:pvd:globalPose", time))
    native_class = _class(geom)
    if geom.IsA(UsdGeom.Mesh):
        if native_class not in {"PxConvexMesh", "convexmesh_ref"}:
            raise ExtractionNotVerified(f"Mesh is not native PxConvexMesh: {native_class}")
        mesh = UsdGeom.Mesh(geom)
        points = mesh.GetPointsAttr().Get(time)
        counts = mesh.GetFaceVertexCountsAttr().Get(time)
        indices = mesh.GetFaceVertexIndicesAttr().Get(time)
        if points is None or counts is None or indices is None:
            raise ExtractionNotVerified("Native PxConvexMesh points or topology absent")
        baked = _transform_points(points, matrix, meters_per_unit)
        try:
            result = _convex_from_faces(baked, counts, indices)
        except ExtractionNotVerified as exc:
            result = _native_vertex_outer_envelope(baked, counts, indices, str(exc))
        result["recorded_mesh_local"] = dict(points=_serial(points), face_vertex_counts=_serial(counts), face_vertex_indices=_serial(indices))
    elif native_class in {"PxGeomBox", "PxBoxGeometry"} and geom.IsA(UsdGeom.Cube):
        size = float(UsdGeom.Cube(geom).GetSizeAttr().Get(time))
        if not math.isfinite(size) or size <= 0:
            raise ExtractionNotVerified("Invalid native box size")
        vertices = np.asarray(list(itertools.product((-size / 2, size / 2), repeat=3)))
        faces = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
        result = _convex_from_faces(_transform_points(vertices, matrix, meters_per_unit), [4] * 6, np.asarray(faces).ravel())
        result["exact_box_to_convex"] = True
    elif native_class in {"PxGeomSphere", "PxSphereGeometry"} and geom.IsA(UsdGeom.Sphere):
        radius = float(UsdGeom.Sphere(geom).GetRadiusAttr().Get(time)) * _uniform_scale(matrix) * meters_per_unit
        if not math.isfinite(radius) or radius <= 0:
            raise ExtractionNotVerified("Invalid native sphere radius")
        result = dict(type="sphere", radius=radius, center=(matrix[:3, 3] * meters_per_unit).tolist())
    elif native_class == "object_ref" and geom.IsA(UsdGeom.Cylinder) and _class(_native_geometry_prim(geom)) == "PxConvexCoreGeometry":
        # _verify_native_handles has established the exact zero-margin native
        # cylinder core and identical recorded radius/height/axis attributes.
        cylinder = UsdGeom.Cylinder(geom)
        scale = _uniform_scale(matrix)
        radius = float(cylinder.GetRadiusAttr().Get(time)) * scale * meters_per_unit
        half_height = .5 * float(cylinder.GetHeightAttr().Get(time)) * scale * meters_per_unit
        axis_name = str(cylinder.GetAxisAttr().Get(time)).upper()
        if axis_name != "X" or not np.allclose(geometry_to_shape, np.eye(4), atol=1e-10, rtol=0):
            raise ExtractionNotVerified("Native PxConvexCoreCylinder must retain native X axis without display transform")
        if axis_name not in {"X", "Y", "Z"} or not np.isfinite([radius, half_height]).all() or min(radius, half_height) <= 0:
            raise ExtractionNotVerified("Invalid native cylinder dimensions or axis")
        axis = matrix[:3, :3][:, {"X": 0, "Y": 1, "Z": 2}[axis_name]] / scale
        result = dict(type="cylinder", radius=radius, half_height=half_height, axis=axis.tolist(), center=(matrix[:3, 3] * meters_per_unit).tolist(), native_core_class="PxConvexCoreCylinder", native_core_margin_m=0.0, native_recorded_axis=axis_name)
    else:
        # A tessellated sphere/capsule/cylinder is never silently treated as an
        # exact analytic shape. The native class and USD schema must agree.
        raise ExtractionNotVerified(f"Unsupported native geometry {native_class!r}/{geom.GetTypeName()}")
    result["native_geometry_class"] = native_class
    result.setdefault("evidence_mode", "EXACT_NATIVE_FACE_POLYTOPE" if result["type"] == "convex" else "EXACT_NATIVE_ANALYTIC_PRIMITIVE")
    result.setdefault("exact_intersection_allowed", True)
    result.setdefault("inside_witness_allowed", True)
    result.setdefault("min_world_x_evidence", "EXACT_SUPPORT")
    result["geometry_prim_path"] = str(geom.GetPath())
    result["geometry_to_actor_matrix_column_stage_units"] = matrix.tolist()
    result["raw_native_shape_local_pose_xyzw_xyz"] = _serial(raw_local_pose)
    result["raw_native_shape_quaternion_norm_error"] = abs(local_norm-1.)
    result["native_local_pose_source"] = "RAW_NATIVE_PXTRANSFORM_NORMALIZED_WITHIN_1E-4_UNIT_TOLERANCE"
    result["display_geometry_to_actor_max_difference_diagnostic"] = float(np.max(np.abs(matrix-display_matrix)))
    result["raw_native_actor_quaternion_norm_error"] = abs(actor_norm-1.)
    actor_world_m = native_actor_world.copy()
    actor_world_m[:3, 3] *= meters_per_unit
    return result, actor_world_m


def _difference(a, b):
    if a["type"] != b["type"]:
        raise ExtractionNotVerified("Native geometry type changes between frames")
    fields = ("vertices_body_m",) if a["type"] == "convex" else ("center", "radius", "half_height", "axis") if a["type"] == "cylinder" else ("center", "radius")
    difference = 0.0
    for key in fields:
        x, y = np.asarray(a[key]), np.asarray(b[key])
        if x.shape != y.shape:
            raise ExtractionNotVerified("Native geometry topology changes between frames")
        difference = max(difference, float(np.max(np.abs(x - y))))
    if a["type"] == "convex" and a.get("original_native_triangles_unique_vertices", a["triangles_unique_vertices"]) != b.get("original_native_triangles_unique_vertices", b["triangles_unique_vertices"]):
        raise ExtractionNotVerified("Native geometry connectivity changes between frames")
    return difference


def extract_native_geometry(stage_usda_path, output_json_path, robot_prefix="/World/envs/env_0/Robot", *, source_stage_meters_per_unit=None):
    """Extract native recording geometry; no simulation or USD writes occur.

source_stage_meters_per_unit must come from the ORIGINAL simulated USD when the
raw OVD conversion lacks authored units. USD's default 0.01 is not evidence.
Output is immutable and never claims whole-robot validation on its own.
"""
    output = Path(output_json_path)
    if output.exists():
        raise FileExistsError(f"Never replace native geometry evidence: {output}")
    source = Path(stage_usda_path).resolve()
    report = dict(schema=SCHEMA, parser_revision=PARSER_REVISION, status="NOT_VERIFIED", geometry_complete=False,
                  evaluation_only=True, model_input=False,
                  created_utc=datetime.now(timezone.utc).isoformat(), source_stage=str(source),
                  extractor_source_sha256=_sha(__file__), robot_prefix=robot_prefix,
                  matching_release_writer_source=MATCHING_RELEASE_SOURCE,
                  writer_source_scope="Matching published 107.3/PhysX5.6.1 source supports native vertex provenance; not proof of binary/source identity. Writer stores scale vector but omits PxMeshScale.rotation.",
                  source_layers=[], actors=[], shapes=[], parts=[], errors=[],
                  invariance_tolerance_m=INVARIANCE_TOLERANCE_M,
                  pending_validations=["RAW_OVD_AND_OFFICIAL_CONVERSION_PROVENANCE", "ORIGINAL_ASSET_AND_RUNTIME_CONFIGURATION_IDENTITY", "NATIVE_SHAPE_INVENTORY_AND_ALL_BODY_LINKS", "INDEPENDENT_NATIVE_FRAME_AND_SCALE_VALIDATION"])
    try:
        from pxr import Usd, UsdGeom
        stage = Usd.Stage.Open(str(source))
        if not stage:
            raise ExtractionNotVerified("Cannot open native OVD USD stage")
        # This is a native recorded stage, not a freshly converted physics asset.
        prims = list(stage.TraverseAll())
        if not any(_class(p) == "PxShape" for p in prims):
            raise ExtractionNotVerified("Native PxShape provenance is absent")
        for layer in sorted(stage.GetUsedLayers(), key=lambda x: x.identifier):
            if layer.anonymous:
                if not layer.empty:
                    raise ExtractionNotVerified("Nonempty anonymous USD layer cannot be source-hashed")
                continue
            path = Path(layer.realPath or layer.identifier)
            if not path.is_file():
                raise ExtractionNotVerified(f"Unresolved source layer {layer.identifier}")
            report["source_layers"].append(dict(path=str(path.resolve()), sha256=_sha(path), bytes=path.stat().st_size))
        authored_units = bool(stage.HasAuthoredMetadata("metersPerUnit"))
        stage_units = float(UsdGeom.GetStageMetersPerUnit(stage)) if authored_units else None
        if source_stage_meters_per_unit is None:
            if not authored_units:
                raise ExtractionNotVerified("METERS_PER_UNIT_NOT_VERIFIED: pass original simulated stage units explicitly")
            meters = stage_units
        else:
            meters = float(source_stage_meters_per_unit)
            if authored_units and not math.isclose(meters, stage_units, rel_tol=1e-8, abs_tol=1e-12):
                raise ExtractionNotVerified("Original simulated stage units disagree with native recording metadata")
        if not math.isfinite(meters) or meters <= 0:
            raise ExtractionNotVerified("Invalid stage units")
        report.update(meters_per_unit=meters, unit_source="ORIGINAL_SIMULATED_STAGE_ARGUMENT" if source_stage_meters_per_unit is not None else "AUTHORED_NATIVE_RECORDING_STAGE_METADATA", converted_stage_authored_meters_per_unit=stage_units, up_axis=str(UsdGeom.GetStageUpAxis(stage)))
        all_times = {float(stage.GetStartTimeCode()), float(stage.GetEndTimeCode())}
        for prim in prims:
            for attr in prim.GetAttributes():
                if attr.GetName().startswith("xformOp:") or attr.GetName() in {"visibility", "omni:pvdi:viz", "points", "faceVertexCounts", "faceVertexIndices", "radius", "size", "height", "omni:pvd:shapeFlags", "omni:pvd:margin"}:
                    all_times.update(float(t) for t in attr.GetTimeSamples())
        times = sorted(t for t in all_times if stage.GetStartTimeCode() <= t <= stage.GetEndTimeCode())
        if not times:
            raise ExtractionNotVerified("Native recording has no observation times")
        report["recording_time_codes"] = times
        native_names = set()
        shape_counts = Counter()
        native_shape_classes = Counter()
        for actor in prims:
            if _class(actor) not in ACTOR_CLASSES:
                continue
            name = _native_name(actor)
            if not (name == robot_prefix or name.startswith(robot_prefix.rstrip("/") + "/")):
                continue
            if name in native_names:
                raise ExtractionNotVerified(f"Duplicate native actor name {name}")
            native_names.add(name)
            owned = []
            for shape in Usd.PrimRange(actor, Usd.TraverseInstanceProxies()):
                if _class(shape) != "PxShape":
                    continue
                parent = shape.GetParent()
                while parent and _class(parent) not in ACTOR_CLASSES:
                    parent = parent.GetParent()
                if parent == actor:
                    owned.append(shape)
            actor_entry = dict(native_name=name, body_name=name.rsplit("/", 1)[-1], prim_path=str(actor.GetPath()), native_class=_class(actor), handle=_serial(_get(actor, "omni:pvdi:handle")), shape_count=len(owned), shape_paths=[str(s.GetPath()) for s in owned])
            actor_entry["actor_poses"] = []
            for t in times:
                display = np.asarray(UsdGeom.XformCache(Usd.TimeCode(t)).GetLocalToWorldTransform(actor), dtype=float).T
                display[:3, 3] *= meters
                raw = _get(actor, "omni:pvd:globalPose", Usd.TimeCode(t))
                matrix, norm = _native_pose_matrix(raw)
                matrix[:3, 3] *= meters
                actor_entry["actor_poses"].append(dict(time_code=t, actor_to_world_m=matrix.tolist(), native_global_pose_source_stage_units=_serial(raw),
                    raw_native_quaternion_norm_error=abs(norm-1.), display_actor_to_world_m_diagnostic=display.tolist(),
                    transform_source="RAW_NATIVE_PXTRANSFORM_NORMALIZED_WITHIN_1E-4_UNIT_TOLERANCE"))
            report["actors"].append(actor_entry)
            for shape in owned:
                entry = dict(native_actor_name=name, body_name=actor_entry["body_name"], actor_prim_path=str(actor.GetPath()), shape_prim_path=str(shape.GetPath()), shape_handle=_serial(_get(shape, "omni:pvdi:handle")), status="NOT_VERIFIED")
                report["shapes"].append(entry)
                try:
                    active = [t for t in times if UsdGeom.Imageable(shape).ComputeVisibility(Usd.TimeCode(t)) != UsdGeom.Tokens.invisible
                              and _get(shape, "omni:pvdi:viz", Usd.TimeCode(t)) is not False]
                    if not active:
                        raise ExtractionNotVerified("Native shape is never present in recorded observation frames")
                    flags = [_serial(_get(shape, "omni:pvd:shapeFlags", Usd.TimeCode(t))) for t in active]
                    entry.update(active_time_codes=active, shape_flags_by_frame=flags)
                    if any(not isinstance(f, list) for f in flags):
                        raise ExtractionNotVerified("Native shape flags absent")
                    if any(f != flags[0] for f in flags[1:]):
                        raise ExtractionNotVerified("Native shape flags change between frames")
                    descendants = _descendants_of_shape(shape, Usd)
                    geometries = [p for p in descendants if any(p.IsA(schema) for schema in (UsdGeom.Mesh, UsdGeom.Cube, UsdGeom.Sphere, UsdGeom.Capsule, UsdGeom.Cylinder))]
                    entry["geometry_candidates"] = [{"path": str(p.GetPath()), "native_class": _class(p), "usd_type": p.GetTypeName()} for p in geometries]
                    if len(geometries) != 1:
                        raise ExtractionNotVerified(f"Expected exactly one actual geometry per PxShape, got {len(geometries)}")
                    geom = geometries[0]
                    shape_counts[_class(geom)] += 1
                    native_shape_classes[_class(_native_geometry_prim(geom))] += 1
                    baseline = None
                    max_difference = 0.0
                    actor_poses = []
                    for t in active:
                        handles = _verify_native_handles(shape, geom, prims, Usd.TimeCode(t), UsdGeom)
                        if "native_handle_chain" not in entry:
                            entry["native_handle_chain"] = handles
                        elif entry["native_handle_chain"] != handles:
                            raise ExtractionNotVerified("Native shape/mesh handle changes between frames")
                        geometry, actor_pose = _geometry_at(geom, actor, Usd.TimeCode(t), meters, UsdGeom)
                        bounds = _get(actor, "omni:pvd:worldBounds", Usd.TimeCode(t))
                        if bounds is not None and not np.isfinite(np.asarray(bounds, dtype=float)).all():
                            raise ExtractionNotVerified("Nonfinite native actor bounds")
                        actor_poses.append(dict(time_code=t, actor_to_world_m=actor_pose.tolist(), native_world_bounds_source_stage_units=_serial(bounds)))
                        if baseline is None:
                            baseline = geometry
                        else:
                            max_difference = max(max_difference, _difference(baseline, geometry))
                    if max_difference > INVARIANCE_TOLERANCE_M:
                        raise ExtractionNotVerified(f"Body-local geometry varies by {max_difference} m")
                    entry.update(status="EXTRACTED", evidence_mode=baseline["evidence_mode"], exact_intersection_allowed=baseline["exact_intersection_allowed"], native_geometry_class=_class(geom), geometry_type=baseline["type"], shape_flags=flags[0], body_frame_invariance=dict(sample_count=len(active), max_difference_m=max_difference, all_recorded_transform_time_samples_checked=True, multiple_samples=len(active) >= 2), actor_poses=actor_poses)
                    report["parts"].append(dict(body_name=actor_entry["body_name"], body_native_path=name, shape_prim_path=str(shape.GetPath()), shape_handle=entry["shape_handle"], shape_flags=flags[0], native_handle_chain=entry["native_handle_chain"], geometry=baseline, body_frame_invariance=entry["body_frame_invariance"]))
                except Exception as exc:
                    entry["error"] = f"{type(exc).__name__}: {exc}"
                    report["errors"].append(f"{shape.GetPath()}: {entry['error']}")
        report.update(actor_count=len(report["actors"]), body_names=[a["body_name"] for a in report["actors"]], shape_count=len(report["shapes"]), extracted_shape_count=len(report["parts"]), native_geometry_class_counts=dict(shape_counts), native_shape_geometry_class_counts=dict(native_shape_classes), complete_prim_traversal=True,
                      transform_convention="Column-vector matrices; native geometry-to-actor full affine applied once, then source-stage units converted to metres; actor is native link frame, not COM")
        if not report["actors"] or not report["shapes"]:
            report["errors"].append("NO_NATIVE_ROBOT_ACTORS_OR_SHAPES")
        report["geometry_complete"] = bool(report["shapes"]) and not report["errors"] and len(report["shapes"]) == len(report["parts"])
        report["evidence_mode_counts"] = dict(Counter(p["geometry"]["evidence_mode"] for p in report["parts"]))
        report["all_shapes_exact_intersection_eligible"] = report["geometry_complete"] and all(p["geometry"]["exact_intersection_allowed"] for p in report["parts"])
        if report["geometry_complete"]:
            report["status"] = "EXTRACTED_PENDING_NATIVE_INVENTORY_VALIDATION"
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return report
