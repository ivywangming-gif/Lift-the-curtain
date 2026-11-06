"""Read-only actual static Cube and native scene-query identity evidence.

No stage edits, simulation updates, rigid-body creation, or control writes.
Native overlap establishes shape presence; CollisionAPI/trigger/static readback
separately establishes the authored simulation-collider contract.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from collision_geometry import GeometryNotVerified

SCHEMA='round4070-g.actual_static_wall.v1'
SOURCE_SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cube_bounds(size,matrix):
    matrix=np.asarray(matrix,float)
    if matrix.shape!=(4,4) or not np.isfinite(matrix).all() or not np.isfinite(size) or size<=0 or not np.allclose(matrix[3],[0,0,0,1],rtol=0,atol=1e-12):
        raise GeometryNotVerified('INVALID_ACTUAL_CUBE_TRANSFORM')
    linear=matrix[:3,:3];norm=np.linalg.norm(linear,axis=0)
    if np.any(norm<=0):raise GeometryNotVerified('DEGENERATE_ACTUAL_WALL_SCALE')
    axes=np.abs(linear/norm)
    if not np.allclose(axes.sum(0),1,rtol=0,atol=1e-8) or not np.allclose(axes.sum(1),1,rtol=0,atol=1e-8) or not np.allclose(axes,np.rint(axes),rtol=0,atol=1e-8):
        raise GeometryNotVerified('ACTUAL_WALL_NOT_AXIS_ALIGNED')
    corners=np.array([[x,y,z,1.] for x in [-size/2,size/2] for y in [-size/2,size/2] for z in [-size/2,size/2]])
    world=(corners@matrix.T)[:,:3]
    return np.stack([world.min(0),world.max(0)])


def validate_wall_snapshot(value,expected):
    """Pure replay of the captured analytic/static/native-presence evidence."""
    if value.get('schema')!=SCHEMA:raise GeometryNotVerified('WALL_EVIDENCE_SCHEMA_MISMATCH')
    if expected is None:
        if value.get('wall_requested') is not False or value.get('barrier_root_exists') is not False:
            raise GeometryNotVerified('UNEXPECTED_ACTUAL_BARRIER_WITHOUT_WALL_PLAN')
        return None
    if value.get('wall_requested') is not True or value.get('barrier_root_exists') is not True or value.get('stage_meters_per_unit')!=1.:
        raise GeometryNotVerified('ACTUAL_WALL_MISSING_OR_UNITS_UNVERIFIED')
    colliders=value.get('collision_prims',[])
    if len(colliders)!=1:raise GeometryNotVerified('ACTUAL_WALL_COLLIDER_INVENTORY_NOT_ONE')
    wall=colliders[0]
    if wall.get('type')!='Cube' or wall.get('has_collision_api') is not True or wall.get('collision_enabled') is not True:
        raise GeometryNotVerified('ACTUAL_WALL_COLLISION_DISABLED_OR_NOT_ANALYTIC_CUBE')
    if wall.get('trigger_api') is not False or wall.get('enabled_rigid_body_ancestors') or wall.get('time_varying_transform_attributes'):
        raise GeometryNotVerified('ACTUAL_WALL_NOT_STATIC_SIMULATION_COLLIDER')
    bounds=_cube_bounds(wall['cube_size'],wall['local_to_world_matrix4'])
    center=np.asarray(expected['center_xyz'],float);size=np.asarray(expected['size_xyz'],float)
    planned=np.stack([center-size/2,center+size/2])
    if expected.get('fixed') is not True or not np.allclose(bounds,planned,rtol=0,atol=1e-6):
        raise GeometryNotVerified('ACTUAL_WALL_BOUNDS_DIFFER_FROM_FROZEN_SCENE')
    query=value.get('native_presence_query',{})
    if query.get('api')!='IPhysxSceneQuery.overlap_box' or query.get('completed') is not True or query.get('any_hit') is not False:
        raise GeometryNotVerified('ACTUAL_WALL_NATIVE_PRESENCE_QUERY_INCOMPLETE')
    if not any(row.get('collision')==wall['prim_path'] for row in query.get('hits',[])):
        raise GeometryNotVerified('ACTUAL_WALL_NATIVE_COLLISION_PATH_NOT_FOUND')
    if query.get('returned_count')!=len(query['hits']):raise GeometryNotVerified('NATIVE_WALL_QUERY_CALLBACK_COUNT_MISMATCH')
    point=np.asarray(query.get('center_world_m'),float);half=np.asarray(query.get('half_extent_m'),float)
    if point.shape!=(3,) or half.shape!=(3,) or not np.isfinite(np.r_[point,half]).all() or np.any(half<=0) or not np.all(point-half>bounds[0]) or not np.all(point+half<bounds[1]):
        raise GeometryNotVerified('NATIVE_QUERY_BOX_NOT_STRICTLY_INSIDE_ACTUAL_WALL')
    rays=value.get('native_face_rays',[])
    if len(rays)!=6:raise GeometryNotVerified('SIX_NATIVE_WALL_FACES_NOT_RECORDED')
    seen=set()
    for ray in rays:
        axis=ray.get('axis');side=ray.get('side')
        if axis not in [0,1,2] or side not in [-1,1] or (axis,side) in seen:raise GeometryNotVerified('INVALID_OR_DUPLICATE_NATIVE_FACE_RAY')
        seen.add((axis,side));origin=np.asarray(ray['origin_world_m'],float);direction=np.asarray(ray['unit_direction'],float)
        expected_origin=bounds.mean(0);expected_origin[axis]=bounds[0 if side<0 else 1,axis]+side*.01
        expected_direction=np.zeros(3);expected_direction[axis]=-side
        if not np.allclose(origin,expected_origin,rtol=0,atol=1e-9) or not np.array_equal(direction,expected_direction):
            raise GeometryNotVerified('NATIVE_FACE_RAY_ORIGIN_OR_DIRECTION_MISMATCH')
        if (ray.get('api')!='IPhysxSceneQuery.raycast_all' or ray.get('completed') is not True
                or ray.get('callback_count')!=len(ray.get('hits',[])) or not isinstance(ray.get('returned_hit'),bool)
                or ray['returned_hit']!=bool(ray.get('hits'))):
            raise GeometryNotVerified('NATIVE_FACE_QUERY_INCOMPLETE')
        hits=[hit for hit in ray['hits'] if hit.get('collision')==wall['prim_path']]
        if not hits:raise GeometryNotVerified('ACTUAL_WALL_NATIVE_FACE_NOT_HIT')
        hit=min(hits,key=lambda x:x['distance_m']);position=np.asarray(hit['position_world_m'],float);normal=np.asarray(hit['normal_world'],float)
        expected_position=bounds.mean(0);expected_position[axis]=bounds[0 if side<0 else 1,axis]
        expected_normal=-expected_direction
        if not np.isfinite(np.r_[position,normal,hit['distance_m']]).all() or not np.allclose(position,expected_position,rtol=0,atol=1e-5) or not np.allclose(normal,expected_normal,rtol=0,atol=1e-5) or abs(hit['distance_m']-.01)>1e-5:
            raise GeometryNotVerified('NATIVE_WALL_FACE_DIFFERS_FROM_ACTUAL_USD_BOX')
    return dict(center_xyz=((bounds[0]+bounds[1])/2).tolist(),size_xyz=(bounds[1]-bounds[0]).tolist(),fixed=True,
        initial_gap_m=expected.get('initial_gap_m'),candidate_plane_x=expected.get('candidate_plane_x'),
        actual_collision_prim_path=wall['prim_path'],actual_static_analytic_geometry_verified=True,
        native_presence_evidence='Exact collider path in native overlap plus six face raycasts matching actual analytic Cube bounds; no visual substitute')


def capture_wall_evidence(stage,expected,root):
    from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema
    import carb
    from omni.physx import get_physx_scene_query_interface
    from omni.physx.bindings import _physx
    root=Path(root);barrier=stage.GetPrimAtPath('/World/GScene/RigidBarrier')
    value=dict(schema=SCHEMA,status='NOT_VERIFIED',evaluation_only=True,model_input=False,wall_requested=expected is not None,
        barrier_root_exists=bool(barrier),barrier_root_path='/World/GScene/RigidBarrier',stage_meters_per_unit=float(UsdGeom.GetStageMetersPerUnit(stage)),
        source_sha256=SOURCE_SHA,native_binding_sha256=sha(_physx.__file__),expected_wall_evaluation_only=expected,collision_prims=[])
    try:
        if barrier:
            cache=UsdGeom.XformCache(Usd.TimeCode.Default())
            for prim in Usd.PrimRange(barrier,Usd.TraverseInstanceProxies()):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
                enabled=UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                ancestors=[];animated=[];node=prim
                while node and node.GetPath().pathString!='/':
                    if node.HasAPI(UsdPhysics.RigidBodyAPI):
                        active=UsdPhysics.RigidBodyAPI(node).GetRigidBodyEnabledAttr().Get()
                        if active is None or bool(active):ancestors.append(str(node.GetPath()))
                    for attr in node.GetAttributes():
                        if attr.GetName().startswith('xformOp:') and attr.ValueMightBeTimeVarying():animated.append(str(attr.GetPath()))
                    node=node.GetParent()
                is_cube=prim.IsA(UsdGeom.Cube)
                value['collision_prims'].append(dict(prim_path=str(prim.GetPath()),type=prim.GetTypeName(),has_collision_api=True,
                    collision_enabled=True if enabled is None else bool(enabled),trigger_api=prim.HasAPI(PhysxSchema.PhysxTriggerAPI),
                    enabled_rigid_body_ancestors=ancestors,time_varying_transform_attributes=animated,
                    cube_size=float(UsdGeom.Cube(prim).GetSizeAttr().Get()) if is_cube else None,
                    local_to_world_matrix4=np.asarray(cache.GetLocalToWorldTransform(prim),float).T.tolist(),
                    physics_attributes={a.GetName():str(a.Get()) for a in prim.GetAttributes() if any(k in a.GetName().lower() for k in ['physics','collision','material'])},
                    relationships={r.GetName():[str(t) for t in r.GetTargets()] for r in prim.GetRelationships()}))
        if expected is not None and len(value['collision_prims'])==1 and value['collision_prims'][0]['type']=='Cube':
            cube=value['collision_prims'][0];bounds=_cube_bounds(cube['cube_size'],cube['local_to_world_matrix4']);center=bounds.mean(0)
            half=np.minimum(np.full(3,.001),(bounds[1]-bounds[0])/4);hits=[]
            def report(hit):
                hits.append(dict(collision=str(hit.collision),rigid_body=str(hit.rigid_body),prototype_index=int(hit.protoIndex)));return True
            count=get_physx_scene_query_interface().overlap_box(carb.Float3(*half.tolist()),carb.Float3(*center.tolist()),carb.Float4(0.,0.,0.,1.),report,False)
            value['native_presence_query']=dict(api='IPhysxSceneQuery.overlap_box',completed=True,any_hit=False,
                center_world_m=center.tolist(),half_extent_m=half.tolist(),quaternion_xyzw=[0,0,0,1],returned_count=int(count),hits=hits,
                scope='Native scene-query collider identity/presence; simulation CollisionAPI and static/trigger properties are separately read back')
            rays=[]
            for axis in range(3):
                for side in [-1,1]:
                    origin=center.copy();origin[axis]=bounds[0 if side<0 else 1,axis]+side*.01
                    direction=np.zeros(3);direction[axis]=-side;ray_hits=[]
                    def report_ray(hit):
                        ray_hits.append(dict(collision=str(hit.collision),rigid_body=str(hit.rigid_body),prototype_index=int(hit.protoIndex),
                            position_world_m=list(hit.position),normal_world=list(hit.normal),distance_m=float(hit.distance)));return True
                    distance=float(bounds[1,axis]-bounds[0,axis]+.02)
                    hit_found=get_physx_scene_query_interface().raycast_all(carb.Float3(*origin.tolist()),carb.Float3(*direction.tolist()),distance,report_ray,False)
                    rays.append(dict(axis=axis,side=side,api='IPhysxSceneQuery.raycast_all',completed=True,origin_world_m=origin.tolist(),
                        unit_direction=direction.tolist(),maximum_distance_m=distance,both_sides=False,returned_hit=bool(hit_found),callback_count=len(ray_hits),hits=ray_hits))
            value['native_face_rays']=rays
        value['actual_wall']=validate_wall_snapshot(value,expected)
        value['status']='VERIFIED' if expected is not None else 'VERIFIED_NO_WALL'
    except Exception as exc:value['reason']=type(exc).__name__+': '+str(exc)
    path=root/'g_wall_collision_evidence.json'
    if path.exists():raise FileExistsError('Never overwrite actual wall evidence')
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')
    return dict(path=path.name,sha256=sha(path),status=value['status'])


def verified_wall_for_audit(run,geom,sources):
    expected=geom.get('wall');binding=geom.get('wall_evidence')
    if expected is None and binding is None:return None  # Existing no-wall raw remains valid.
    if not isinstance(binding,dict):raise GeometryNotVerified('ACTUAL_WALL_ENTITY_EVIDENCE_MISSING')
    run=Path(run).resolve();path=(run/binding['path']).resolve()
    if not path.is_relative_to(run) or not path.is_file() or sha(path)!=binding.get('sha256'):
        raise GeometryNotVerified('ACTUAL_WALL_ENTITY_EVIDENCE_SHA_MISMATCH')
    sources[path.relative_to(run).as_posix()]=sha(path);value=json.loads(path.read_text(encoding='utf-8-sig'))
    runner=json.loads((run/'g_runner_contract.json').read_text(encoding='utf-8-sig'))
    archived=runner.get('runtime_source_archive',{}).get('physics_wall_evidence.py',{})
    archive_path=Path(archived.get('path','MISSING'))
    if value.get('source_sha256')!=archived.get('sha256') or not archive_path.is_file() or sha(archive_path)!=value.get('source_sha256'):
        raise GeometryNotVerified('WALL_QUERY_SOURCE_ARCHIVE_MISMATCH')
    if value.get('native_binding_sha256')!=geom.get('runtime_physx_binding',{}).get('module_sha256'):
        raise GeometryNotVerified('WALL_NATIVE_QUERY_BINARY_MISMATCH')
    return validate_wall_snapshot(value,expected)
