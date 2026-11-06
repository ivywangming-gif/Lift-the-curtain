"""Triangle versus actual collider geometry in a common metre coordinate frame.

Convex planes use n.x+d <= 0. No mesh is replaced by a sphere. All scale must
already be baked into geometry; transforms belong to the caller. Closed contact,
strict interior intersection and penetration beyond an inward distance tolerance
are separate results. These are surface-triangle tests, not a temporal or solver
validity certificate. Unsupported/invalid geometry raises GeometryNotVerified.
"""
from __future__ import annotations

import itertools
import numpy as np


class GeometryNotVerified(ValueError):
    """No collision-validity conclusion may be inferred for this input."""


NotVerifiedGeometry = GeometryNotVerified
NUMERICAL_TOLERANCE_M = 1e-9
VALIDATION_TOLERANCE_M = 1e-6


def _array(value, shape=None, name="geometry"):
    result = np.asarray(value, dtype=np.float64)
    if (shape is not None and result.shape != shape) or not np.isfinite(result).all():
        raise GeometryNotVerified(f"Invalid/nonfinite {name}")
    return result


def _scale(scale):
    if scale is None:
        return
    value = np.asarray(scale, dtype=float)
    if value.ndim == 0:
        value = np.repeat(value, 3)
    if value.shape != (3,) or not np.isfinite(value).all() or not np.allclose(value, 1., rtol=0., atol=1e-12):
        raise GeometryNotVerified("UNBAKED_SCALE: bake actual affine convex geometry, or uniform primitive dimensions, into metres first; nonuniform primitives unsupported")


def _aabb(value):
    bounds = _array(value, (2,3), "AABB")
    if np.any(bounds[0] > bounds[1]):
        raise GeometryNotVerified("Inverted AABB")
    return bounds


def aabb_overlap(triangle, bounds, epsilon=NUMERICAL_TOLERANCE_M):
    triangle, bounds = _array(triangle, (3,3), "triangle"), _aabb(bounds)
    return bool(np.all(triangle.max(0) >= bounds[0]-epsilon) and np.all(triangle.min(0) <= bounds[1]+epsilon))


def _halfspace_vertices(planes, tolerance):
    """Find feasible edge endpoints, also proving finite bounds (3D only).

For each independent pair of boundary planes, intersect its line with every
halfspace. A bounded full-dimensional polytope has all vertices on such lines.
This derives bounds from the planes, so an incomplete optional vertex list can
never produce a falsely tight broad-phase box. Work is batched O(faces^3).
"""
    normals, offset = planes[:,:3], planes[:,3]
    pairs = np.asarray(list(itertools.combinations(range(len(planes)), 2)))
    endpoints = []
    for start in range(0, len(pairs), 512):
        indices = pairs[start:start+512]
        na, nb = normals[indices[:,0]], normals[indices[:,1]]
        direction = np.cross(na, nb)
        squared = np.einsum("ij,ij->i", direction, direction)
        valid = squared > 1e-24
        if not np.any(valid):
            continue
        indices, na, nb, direction, squared = indices[valid], na[valid], nb[valid], direction[valid], squared[valid]
        origin = (-offset[indices[:,0],None]*np.cross(nb,direction)-offset[indices[:,1],None]*np.cross(direction,na))/squared[:,None]
        unit = direction/np.sqrt(squared[:,None])
        coefficient = unit @ normals.T
        rhs = -(origin @ normals.T+offset)
        parallel = np.abs(coefficient) <= 1e-12
        feasible = ~np.any(parallel & (rhs < -tolerance), axis=1)
        ratio = np.divide(rhs, coefficient, out=np.zeros_like(rhs), where=~parallel)
        low = np.max(np.where(coefficient < -1e-12, ratio, -np.inf), axis=1)
        high = np.min(np.where(coefficient > 1e-12, ratio, np.inf), axis=1)
        feasible &= low <= high+tolerance
        if np.any(feasible & (~np.isfinite(low) | ~np.isfinite(high))):
            raise GeometryNotVerified("Convex halfspaces are unbounded; not a closed collider")
        if np.any(feasible):
            endpoints.extend((origin[feasible]+low[feasible,None]*unit[feasible], origin[feasible]+high[feasible,None]*unit[feasible]))
    if not endpoints:
        raise GeometryNotVerified("Convex halfspaces have no verified feasible vertices")
    vertices = np.concatenate(endpoints)
    if np.linalg.matrix_rank(vertices-vertices.mean(0), tol=tolerance) < 3:
        raise GeometryNotVerified("Convex collider is empty or lower dimensional")
    return vertices


def prepare_convex(planes, vertices=None, aabb=None, scale=None, validation_tolerance=VALIDATION_TOLERANCE_M):
    _scale(scale)
    validation_tolerance = float(validation_tolerance)
    if not np.isfinite(validation_tolerance) or validation_tolerance <= 0:
        raise GeometryNotVerified("Invalid convex validation tolerance")
    p = _array(planes, name="convex planes")
    if p.ndim != 2 or p.shape[1] != 4 or not 4 <= len(p) <= 512:
        raise GeometryNotVerified("Expected 4..512 actual cooked polygon planes [nx,ny,nz,d]")
    lengths = np.linalg.norm(p[:,:3], axis=1)
    if np.any(lengths < 1e-12) or np.linalg.matrix_rank(p[:,:3]) < 3:
        raise GeometryNotVerified("Degenerate convex plane normals")
    p = p/lengths[:,None]
    derived = _halfspace_vertices(p, validation_tolerance)
    bounds = np.stack((derived.min(0)-validation_tolerance, derived.max(0)+validation_tolerance))
    if vertices is not None:
        supplied = _array(vertices, name="actual cooked vertices")
        if supplied.ndim != 2 or supplied.shape[1] != 3 or len(supplied) < 4:
            raise GeometryNotVerified("Invalid cooked vertex array")
        distances = supplied @ p[:,:3].T+p[:,3]
        if np.max(distances) > validation_tolerance:
            raise GeometryNotVerified("Cooked vertices violate outward halfspace convention n.x+d<=0")
        if np.any(np.min(np.abs(distances), axis=0) > validation_tolerance):
            raise GeometryNotVerified("A cooked polygon plane is unsupported by the supplied vertices")
    if aabb is not None:
        proposed = _aabb(aabb)
        if np.any(proposed[0] > derived.min(0)+validation_tolerance) or np.any(proposed[1] < derived.max(0)-validation_tolerance):
            raise GeometryNotVerified("Supplied AABB excludes actual halfspace geometry")
        bounds = np.stack((np.minimum(proposed[0],bounds[0]), np.maximum(proposed[1],bounds[1])))
    return dict(prepared_geometry=True, type="convex", planes=p, aabb=bounds,
                validation_tolerance_m=float(validation_tolerance), plane_convention="n.x+d<=0; normalized metre distances")


def prepare_sphere(radius, center=None, scale=None):
    _scale(scale)
    radius = float(radius)
    center = np.zeros(3) if center is None else _array(center,(3,),"sphere center")
    if not np.isfinite(radius) or radius <= 0:
        raise GeometryNotVerified("Sphere radius must be positive")
    return dict(prepared_geometry=True,type="sphere",radius=radius,center=center,aabb=np.stack((center-radius,center+radius)))


def prepare_cylinder(radius, half_height, axis="z", center=None, scale=None):
    _scale(scale)
    radius, half_height = float(radius), float(half_height)
    if not np.isfinite([radius,half_height]).all() or min(radius,half_height) <= 0:
        raise GeometryNotVerified("Cylinder radius/half-height must be positive")
    center = np.zeros(3) if center is None else _array(center,(3,),"cylinder center")
    if isinstance(axis,str):
        if axis.lower() not in ("x","y","z"):
            raise GeometryNotVerified("Unsupported cylinder axis")
        axis = np.eye(3)[("x","y","z").index(axis.lower())]
    axis = _array(axis,(3,),"cylinder axis")
    if np.linalg.norm(axis) < 1e-12:
        raise GeometryNotVerified("Degenerate cylinder axis")
    axis = axis/np.linalg.norm(axis)
    reference = np.eye(3)[np.argmin(np.abs(axis))]
    first = np.cross(reference,axis); first /= np.linalg.norm(first)
    second = np.cross(axis,first)
    basis = np.column_stack((first,second,axis))
    extent = half_height*np.abs(axis)+radius*np.sqrt(np.maximum(0.,1-axis*axis))
    return dict(prepared_geometry=True,type="cylinder",radius=radius,half_height=half_height,
                center=center,axis=axis,basis=basis,aabb=np.stack((center-extent,center+extent)))


def prepare_collider(collider):
    if not isinstance(collider,dict):
        raise GeometryNotVerified("Collider must be a supported geometry description")
    kind = str(collider.get("type", "")).lower()
    if kind not in ("convex","sphere","cylinder"):
        raise GeometryNotVerified(f"Unsupported actual collider type: {kind!r}; no surrogate substitution")
    required = {"convex":("planes",),"sphere":("radius",),"cylinder":("radius","half_height")}[kind]
    if any(key not in collider for key in required):
        raise GeometryNotVerified(f"Incomplete {kind} collider description")
    _scale(collider.get("scale"))
    if collider.get("prepared_geometry") is True:
        return collider
    if kind == "convex":
        return prepare_convex(collider["planes"],collider.get("vertices"),collider.get("aabb"),collider.get("scale"))
    if kind == "sphere":
        return prepare_sphere(collider["radius"],collider.get("center"),collider.get("scale"))
    if kind == "cylinder":
        return prepare_cylinder(collider["radius"],collider["half_height"],collider.get("axis","z"),collider.get("center"),collider.get("scale"))
    raise GeometryNotVerified(f"Unsupported actual collider type: {kind!r}")


def _clip_polygon(polygon, planes):
    polygon = np.asarray(polygon,dtype=float).reshape(-1,3)
    for plane in planes:
        if not len(polygon):
            break
        output = []
        previous = polygon[-1]; previous_distance = float(previous @ plane[:3]+plane[3])
        for current in polygon:
            current_distance = float(current @ plane[:3]+plane[3])
            previous_inside, current_inside = previous_distance <= 0., current_distance <= 0.
            if previous_inside != current_inside:
                fraction = previous_distance/(previous_distance-current_distance)
                output.append(previous+fraction*(current-previous))
            if current_inside:
                output.append(current)
            previous, previous_distance = current, current_distance
        polygon = np.asarray(output,dtype=float).reshape(-1,3)
    return polygon


def _closest_origin_triangle(triangle):
    # Boundary candidates also handle line/point-degenerate cloth triangles.
    candidates = []
    for a,b in ((0,1),(1,2),(2,0)):
        edge = triangle[b]-triangle[a]; squared = float(edge @ edge)
        t = float(np.clip(-(triangle[a] @ edge)/squared,0.,1.)) if squared else 0.
        weights = np.zeros(3); weights[a],weights[b] = 1-t,t
        point = weights @ triangle
        candidates.append((float(point @ point),point,weights))
    normal = np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0])
    squared = float(normal @ normal)
    edge_scale = max(np.linalg.norm(triangle[1]-triangle[0]),np.linalg.norm(triangle[2]-triangle[0]),1e-15)
    if squared > (NUMERICAL_TOLERANCE_M*edge_scale)**2:
        point = normal*(float(normal @ triangle[0])/squared)
        weights = np.asarray([np.dot(np.cross(triangle[(i+1)%3]-point,triangle[(i+2)%3]-point),normal)/squared for i in range(3)])
        if np.min(weights) >= -NUMERICAL_TOLERANCE_M/edge_scale:
            weights = np.maximum(weights,0.); weights /= weights.sum()
            point = weights @ triangle
            candidates.append((float(point @ point),point,weights))
    return min(candidates,key=lambda value:value[0])


def _radial_witness(polygon):
    if not len(polygon):
        return None,None
    if len(polygon) == 1:
        return float(np.linalg.norm(polygon[0,:2])),polygon[0]
    best = None
    for i in range(1,max(2,len(polygon)-1)):
        indices = [0,i,min(i+1,len(polygon)-1)]
        projected = polygon[indices].copy(); projected[:,2] = 0.
        squared,_,weights = _closest_origin_triangle(projected)
        witness = weights @ polygon[indices]
        if best is None or squared < best[0]:
            best = squared,witness
    return float(np.sqrt(max(0.,best[0]))),best[1]


def _status(closed,strict,inner):
    return "INNER_PENETRATION" if inner else "INTERSECTION_WITHIN_TOLERANCE" if strict else "TOUCH_ONLY" if closed else "DISJOINT"


def triangle_collider_relation(triangle, collider, penetration_tolerance=.005, numerical_tolerance=NUMERICAL_TOLERANCE_M):
    tri = _array(triangle,(3,3),"triangle")
    tolerance,epsilon = float(penetration_tolerance),float(numerical_tolerance)
    if not np.isfinite([tolerance,epsilon]).all() or tolerance < 0 or epsilon < NUMERICAL_TOLERANCE_M:
        raise GeometryNotVerified("Invalid penetration/numerical tolerance")
    shape = prepare_collider(collider)
    kind = shape["type"]
    empty = shape["radius"] <= tolerance+epsilon if kind == "sphere" else min(shape["radius"],shape["half_height"]) <= tolerance+epsilon if kind == "cylinder" else None
    result = dict(geometry_type=kind,penetration_tolerance_m=tolerance,numerical_tolerance_m=epsilon,
                  intersects_closed=False,has_strict_interior_intersection=False,inner_penetration=False,
                  inner_region_empty=bool(empty) if empty is not None else None,contact_status="DISJOINT",aabb_rejected=False,
                  witness_local=None,inner_witness_local=None,inner_witness_clearance_m=None,
                  definition="inner_penetration requires a triangle point deeper than tolerance inside every collider boundary; local coordinates are actual metres")
    if not aabb_overlap(tri,shape["aabb"],epsilon):
        result["aabb_rejected"] = True
        return result
    if kind == "convex":
        planes = shape["planes"]
        def clip(inset):
            shifted = planes.copy(); shifted[:,3] += inset
            return _clip_polygon(tri,shifted)
        outer,strict_polygon,inner = clip(-epsilon),clip(epsilon),clip(tolerance+epsilon)
        closed,strict,penetrates = bool(len(outer)),bool(len(strict_polygon)),bool(len(inner))
        witness = outer.mean(0) if closed else None
        inner_witness = inner.mean(0) if penetrates else None
        clearance = float(np.min(-(planes[:,:3] @ inner_witness+planes[:,3]))) if penetrates else None
        result.update(intersection_polygon_local=outer.tolist(),inner_clipped_polygon_local=inner.tolist())
    elif kind == "sphere":
        squared,point,_ = _closest_origin_triangle(tri-shape["center"])
        distance = float(np.sqrt(max(0.,squared)))
        clearance = shape["radius"]-distance
        closed = distance <= shape["radius"]+epsilon
        strict = clearance > epsilon
        penetrates = not empty and clearance > tolerance+epsilon
        witness = point+shape["center"] if closed else None
        inner_witness = witness if penetrates else None
        result["maximum_sphere_interior_depth_m"] = clearance if closed else None
        clearance = clearance if penetrates else None
    elif kind == "cylinder":
        local = (tri-shape["center"]) @ shape["basis"]
        def clipped(inset):
            height = shape["half_height"]-inset
            if height <= 0:
                return np.empty((0,3)),None,None
            polygon = _clip_polygon(local,np.asarray([[0,0,1,-height],[0,0,-1,-height]],dtype=float))
            radial,point = _radial_witness(polygon)
            return polygon,radial,point
        outer,radial,point = clipped(-epsilon)
        _,strict_radial,_ = clipped(epsilon)
        inner,inner_radial,inner_point = clipped(tolerance+epsilon)
        closed = radial is not None and radial <= shape["radius"]+epsilon
        strict = strict_radial is not None and strict_radial < shape["radius"]-epsilon
        penetrates = not empty and inner_radial is not None and inner_radial < shape["radius"]-tolerance-epsilon
        to_frame = lambda p: p @ shape["basis"].T+shape["center"]
        witness = to_frame(point) if closed else None
        inner_witness = to_frame(inner_point) if penetrates else None
        clearance = min(shape["radius"]-inner_radial,shape["half_height"]-abs(float(inner_point[2]))) if penetrates else None
        result.update(axial_clipped_polygon_local=to_frame(outer).tolist(),inner_axial_clipped_polygon_local=to_frame(inner).tolist(),
                      cylinder_polygon_note="Clipped by axial caps only; curved-side intersection is decided analytically by radial minimum, not a polygonal approximation")
    else:
        raise GeometryNotVerified(f"Unsupported prepared collider type: {kind}")
    result.update(intersects_closed=bool(closed),has_strict_interior_intersection=bool(strict),inner_penetration=bool(penetrates),
                  contact_status=_status(closed,strict,penetrates),witness_local=None if witness is None else witness.tolist(),
                  inner_witness_local=None if inner_witness is None else inner_witness.tolist(),
                  inner_witness_clearance_m=None if clearance is None else float(clearance))
    return result


def triangle_convex_relation(triangle_local, planes, penetration_tolerance=.005, vertices=None, aabb=None):
    return triangle_collider_relation(triangle_local,prepare_convex(planes,vertices,aabb),penetration_tolerance)


def triangle_sphere_relation(triangle_local, radius, penetration_tolerance=.005, center=None, scale=None):
    return triangle_collider_relation(triangle_local,prepare_sphere(radius,center,scale),penetration_tolerance)


def triangle_cylinder_relation(triangle_local, radius, half_height, penetration_tolerance=.005, axis="z", center=None, scale=None):
    return triangle_collider_relation(triangle_local,prepare_cylinder(radius,half_height,axis,center,scale),penetration_tolerance)


def convex_box_solid_relation(body_geometry, world_box_bounds, body_rotation, body_position,
                              penetration_tolerance=.005, numerical_tolerance=1e-8):
    """Common strict interior of a convex solid and a WORLD-aligned solid box.

Maximize a common inscribed-ball radius under both normalized halfspace sets.
This detects containment and coincident solids missed by boundary triangles.
Only mathematically certified separation/overlap is returned. Solver failure
or a threshold-sized numerical ambiguity remains NUMERICAL_UNVERIFIED.
For a conservative native envelope, any overlap is NOT_VERIFIED rather than
actual penetration. The caller must retain the source evidence_mode/flag in
body_geometry; never drop that flag while preparing an envelope.
"""
    from scipy.optimize import linprog
    conservative = (body_geometry.get("exact_intersection_allowed") is False or
                    "CONSERVATIVE" in str(body_geometry.get("evidence_mode", "")))
    shape = prepare_collider(body_geometry)
    if shape["type"] != "convex":
        raise GeometryNotVerified("Convex-box solid LP requires actual convex halfspaces")
    bounds = _aabb(world_box_bounds)
    if np.any(bounds[1] <= bounds[0]):
        raise GeometryNotVerified("Wall must be a full-dimensional solid box")
    rotation = _array(body_rotation, (3,3), "body rotation")
    position = _array(body_position, (3,), "body position")
    if not np.allclose(rotation.T@rotation,np.eye(3),rtol=0,atol=1e-8) or np.linalg.det(rotation) < 0:
        raise GeometryNotVerified("Body rotation must be proper orthonormal; bake scale first")
    tolerance, epsilon = float(penetration_tolerance),float(numerical_tolerance)
    if not np.isfinite([tolerance,epsilon]).all() or tolerance < 0 or not 1e-10 <= epsilon <= 1e-6:
        raise GeometryNotVerified("Invalid common-interior tolerance")
    result=dict(verified=False,status="NUMERICAL_UNVERIFIED",disjoint=False,inner_penetration=False,
                intersects_closed=None,common_interior_radius_m=None,witness_world=None,
                penetration_tolerance_m=tolerance,numerical_tolerance_m=epsilon,
                conservative_envelope=conservative,actual_penetration_certified=False,aabb_rejected=False)
    corners=np.asarray(list(itertools.product(*zip(shape["aabb"][0],shape["aabb"][1]))))
    world=corners@rotation.T+position
    if np.any(world.max(0)<bounds[0]-epsilon) or np.any(world.min(0)>bounds[1]+epsilon):
        result.update(verified=True,status="DISJOINT",disjoint=True,intersects_closed=False,aabb_rejected=True)
        return result
    local=shape["planes"]
    normals=local[:,:3]@rotation.T
    offsets=local[:,3]-normals@position
    normals=np.vstack((normals,np.eye(3),-np.eye(3)))
    offsets=np.r_[offsets,-bounds[1],bounds[0]]
    constraint=np.c_[normals,np.ones(len(normals))]
    rhs=-offsets
    try:
        optimum=linprog([0.,0.,0.,-1.],A_ub=constraint,b_ub=rhs,bounds=[(None,None)]*4,
                        method="highs",options={"primal_feasibility_tolerance":1e-9,"dual_feasibility_tolerance":1e-9})
    except Exception as exc:
        result["solver_reason"]=type(exc).__name__+": "+str(exc)
        return result
    result.update(solver_status=int(optimum.status),solver_message=str(optimum.message))
    if not optimum.success or optimum.x is None or not np.isfinite(optimum.x).all():
        return result
    witness=optimum.x[:3]
    radius=float(optimum.x[3])
    actual_clearance=float(np.min(-(normals@witness+offsets)))
    primal_error=float(np.max(constraint@optimum.x-rhs))
    # The dual weights prove an upper bound on radius. They are nonnegative,
    # sum to one and balance the normals; do not use a bare optimizer success.
    dual=-np.asarray(optimum.ineqlin.marginals,dtype=float)
    dual_error=max(float(max(0.,-dual.min())),abs(float(dual.sum())-1.),float(np.max(np.abs(dual@normals))))
    dual_bound=float(dual@rhs)
    gap=dual_bound-actual_clearance
    result.update(common_interior_radius_m=radius,witness_world=witness.tolist(),
                  witness_common_clearance_m=actual_clearance,dual_radius_upper_bound_m=dual_bound,
                  primal_violation_m=primal_error,dual_certificate_residual=dual_error,primal_dual_gap_m=gap)
    if primal_error>epsilon or dual_error>epsilon or gap < -epsilon or gap > 10*epsilon:
        return result
    if dual_bound < -epsilon:
        result.update(verified=True,status="DISJOINT",disjoint=True,intersects_closed=False)
        return result
    if conservative:
        result.update(status="CONSERVATIVE_ENVELOPE_OVERLAP_NOT_VERIFIED",intersects_closed=actual_clearance>=-epsilon)
        return result
    if actual_clearance > tolerance+epsilon:
        result.update(verified=True,status="INNER_PENETRATION",inner_penetration=True,
                      intersects_closed=True,actual_penetration_certified=True)
    elif dual_bound <= tolerance+epsilon and actual_clearance >= -epsilon:
        result.update(verified=True,status="INTERSECTION_WITHIN_TOLERANCE",intersects_closed=True)
    return result
