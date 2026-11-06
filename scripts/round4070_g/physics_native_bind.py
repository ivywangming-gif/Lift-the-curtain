"""Read-only native shape inventory and per-episode verified library binding.

No launcher, simulation step, actuator write, or USD mutation belongs here.
Missing callbacks are evidence gaps; authored visual geometry is never a fallback.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from collision_clearance import _rotations

ROOT=Path(__file__).resolve().parents[2]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write_new(path,value):
    path=Path(path)
    if path.exists():raise FileExistsError('Native evidence is append-only: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def _query_template(summary):
    rows=[]
    for q in summary['native_property_queries']:
        if not q['body_path'].startswith('/World/envs/env_0/Robot/'):continue
        rows.append(dict(body_name=q['body_path'].rsplit('/',1)[-1],body_path=q['body_path'],
            colliders=[dict(ordinal=i,path=c['path'],volume=c['volume'],aabb_local_min=c['aabb_local_min'],
                aabb_local_max=c['aabb_local_max'],local_pos=c['local_pos'],local_rot_xyzw=c['local_rot']) for i,c in enumerate(q['colliders'])]))
    return rows


def verify_native_mesh_scale(source):
    """The native handle chain, not a display-class alias, owns mesh scale."""
    chain=source.get('native_handle_chain',{})
    if chain.get('native_convex_mesh_handle') is not None:
        scale=np.asarray(chain.get('native_recorded_mesh_scale'),float)
        if (chain.get('native_mesh_scale_isotropic') is not True or chain.get('mesh_scale_rotation_evidence')!='IRRELEVANT_FOR_ISOTROPIC_SCALE'
                or scale.shape!=(3,) or not np.isfinite(scale).all() or np.any(scale<=0) or not np.allclose(scale,scale[0],rtol=0,atol=1e-12)):
            raise ValueError('Native mesh scale rotation is unverified')


def validate_and_build_library(run,native_path,compound_path,output):
    """Bind complete actual native shape evidence to measured link frames.

    No scientific G1 result is read here. Conservative native vertex envelopes
    remain explicitly different from exact native intersection geometry.
    """
    run=Path(run);native_path=Path(native_path);compound_path=Path(compound_path)
    summary=read(run/'summary.json');native=read(native_path);link=read(run/'native_link_pose_samples.json')
    geom=read(run/'g_robot_collision_geometry.json');compound=read(compound_path)
    if not native.get('geometry_complete') or native.get('errors'):raise ValueError('Incomplete native extraction')
    if native.get('shape_count')!=39 or native.get('extracted_shape_count')!=39:raise ValueError('Native inventory expected exactly actual 39 shapes')
    if native.get('native_shape_geometry_class_counts')!={'PxConvexMeshGeometry':27,'PxSphereGeometry':8,'PxConvexCoreGeometry':4}:raise ValueError('Unexpected actual native class inventory')
    names=link['body_names'];actors={a['body_name']:a for a in native['actors']}
    if len(names)!=39 or set(names)!=set(actors):raise ValueError('Native actor/link body coverage mismatch')
    times=sorted(set(p['time_code'] for actor in actors.values() for p in actor['actor_poses']))
    matches=[]
    for sample in link['samples']:
        pose=np.asarray(sample['body_link_pose_w']);com=np.asarray(sample['body_com_pose_w']);rotation,_=_rotations(pose[:,3:],1e-4)
        candidates=[]
        for time in times:
            matrices=[];display_matrices=[]
            for name in names:
                matching=[p for p in actors[name]['actor_poses'] if p['time_code']==time]
                if len(matching)!=1:break
                evidence=matching[0];raw=np.asarray(evidence['native_global_pose_source_stage_units'],float)
                if raw.shape!=(7,) or not np.isfinite(raw).all():raise ValueError('Missing raw native actor global pose')
                native_rotation,_=_rotations(raw[[3,0,1,2]][None,:],1e-4)
                matrix=np.eye(4);matrix[:3,:3]=native_rotation[0];matrix[:3,3]=raw[4:]*native['meters_per_unit']
                matrices.append(matrix);display_matrices.append(evidence['display_actor_to_world_m_diagnostic'])
            if len(matrices)!=len(names):continue
            matrices=np.asarray(matrices);pe=float(np.max(np.abs(matrices[:,:3,3]-pose[:,:3])));re=float(np.max(np.abs(matrices[:,:3,:3]-rotation)))
            display_error=float(np.max(np.abs(np.asarray(display_matrices)[:,:3,:3]-matrices[:,:3,:3])))
            candidates.append((max(pe,re),time,pe,re,float(np.max(np.abs(matrices[:,:3,3]-com[:,:3]))),display_error))
        candidates.sort()
        if not candidates or candidates[0][0]>1e-6 or (len(candidates)>1 and candidates[1][0]<5e-6):raise ValueError('Native actor frame lacks a unique same-tick link-pose match')
        _,time,pe,re,ce,display_error=candidates[0]
        matches.append(dict(simulation_time_s=sample['simulation_time_s'],actual_physics_step_from_clock=int(round(sample['simulation_time_s']/.005)),
            diagnostic_explicit_capture_counter=sample['physics_step'],native_time_code=time,max_position_error_m=pe,max_rotation_matrix_error=re,
            max_native_vs_com_position_difference_m=ce,display_usd_rotation_error=display_error,
            rotation_source='Raw native globalPose XYZW, norm drift <=1e-4 accepted and normalized exactly as IsaacLab quaternion audit; USD display transform not used as pose truth'))
    if len(matches)<3:raise ValueError('Insufficient independent frame samples')
    clock=np.asarray([m['actual_physics_step_from_clock'] for m in matches]);codes=np.asarray([m['native_time_code'] for m in matches])
    if not np.all(np.diff(clock)==1) or not np.allclose(codes,clock*2,rtol=0,atol=1e-9):raise ValueError('Native sample timing does not match actual physics clock')
    if max(m['max_native_vs_com_position_difference_m'] for m in matches)<1e-4:raise ValueError('Diagnostic does not distinguish actor and COM frames')
    expected_compound=summary['known_compound_children'];compound_errors=[]
    if len(compound.get('parts',[]))!=len(expected_compound):raise ValueError('Known compound shape count mismatch')
    available=list(range(len(compound['parts'])))
    for expected in expected_compound:
        target=np.asarray(expected['actual_body_local_vertices']);distances=[]
        for index in available:
            actual=np.asarray(compound['parts'][index]['geometry']['vertices_body_m']);distance=np.linalg.norm(actual[:,None]-target[None],axis=-1)
            distances.append((float(max(distance.min(0).max(),distance.min(1).max())),index))
        error,index=min(distances)
        if error>1e-6:raise ValueError('Independent compound rotation/scale/body-frame validation failed')
        available.remove(index);compound_errors.append(dict(source_child=expected['path'],native_part_index=index,bidirectional_hausdorff_error_m=error))
    parts=[];mode_counts={}
    for source in native['parts']:
        if 'eSIMULATION_SHAPE' not in source.get('shape_flags',[]):raise ValueError('Native part is not a simulation-enabled collider')
        geometry=source['geometry'];mode=geometry.get('evidence_mode','')
        conservative='CONSERVATIVE' in mode
        # The writer omits PxMeshScale.rotation. Only actual isotropic mesh
        # scale makes that omitted rotation irrelevant. Parser class aliases
        # are not a reliable way to identify native convex mesh handles.
        verify_native_mesh_scale(source)
        standardized='NATIVE_VERTEX_CONSERVATIVE_ENVELOPE' if conservative else 'EXACT_NATIVE'
        mode_counts[standardized]=mode_counts.get(standardized,0)+1
        part=dict(body_name=source['body_name'],body_index=names.index(source['body_name']),shape_handle=source['shape_handle'],
            shape_prim_path=source['shape_prim_path'],shape_flags=source['shape_flags'],native_handle_chain=source['native_handle_chain'],
            source_evidence_mode=mode,evidence_mode=standardized,exact_intersection_allowed=not conservative,
            inside_witness_allowed=bool(geometry.get('inside_witness_allowed')) and not conservative,
            native_geometry_source=geometry,type=geometry['type'])
        if part['type']=='convex':part.update(vertices_body_m=geometry['vertices_body_m'],planes_body_m=geometry['planes'])
        elif part['type']=='sphere':part.update(radius_m=geometry['radius'],center_body_m=geometry['center'])
        elif part['type']=='cylinder':part.update(radius_m=geometry['radius'],half_height_m=geometry['half_height'],center_body_m=geometry['center'],axis_body=geometry['axis'])
        else:raise ValueError('Unsupported native geometry '+part['type'])
        parts.append(part)
    if mode_counts!={'EXACT_NATIVE':27,'NATIVE_VERTEX_CONSERVATIVE_ENVELOPE':12}:raise ValueError('Unexpected exact/conservative split')
    # Freeze a witness only inside the original-face-consistent pelvis solid.
    from scipy.optimize import linprog
    eligible=[(i,p) for i,p in enumerate(parts) if p['body_name']=='pelvis_contour_link' and p['type']=='convex' and p['inside_witness_allowed']]
    if len(eligible)!=1:raise ValueError('Unambiguous exact pelvis witness solid is missing')
    index,part=eligible[0];planes=np.asarray(part['planes_body_m']);normal=np.linalg.norm(planes[:,:3],axis=1);planes=planes/normal[:,None]
    solution=linprog([0,0,0,-1],A_ub=np.c_[planes[:,:3],np.ones(len(planes))],b_ub=-planes[:,3],bounds=[(None,None)]*3+[(0,None)],method='highs')
    if not solution.success:raise ValueError('Exact native interior witness optimization failed')
    point=solution.x[:3];clearance=float(np.min(-(planes[:,:3]@point+planes[:,3])))
    if clearance<.006+1e-9:raise ValueError('Actual native witness lacks a verified 6mm inner ball')
    witness=dict(body_name=part['body_name'],body_index=part['body_index'],part_index=index,point_body_m=point.tolist(),
        actual_inner_ball_radius_m=clearance,required_minimum_radius_m=.006,exact_native=True,
        native_part_geometry_sha256=hashlib.sha256(json.dumps(part,sort_keys=True,separators=(',',':')).encode()).hexdigest())
    sources=[run/'summary.json',run/'native_link_pose_samples.json',run/'g_robot_collision_geometry.json',native_path,compound_path]
    for conversion in summary['native_omnipvd_capture']:
        if conversion.get('conversion_success') and conversion.get('cache')==str(native_path.parent.relative_to(run)):
            sources.append(run/conversion['ovd']);sources.extend(run/name for name in conversion['files'])
    library=dict(schema='round4070-g.verified_native_robot_library.v1',status='VERIFIED_MIXED_EXACT_AND_CONSERVATIVE',geometry_complete=True,frame_verified=True,
        body_names=names,native_collision_shape_count=len(parts),expected_collision_body_indices=sorted({p['body_index'] for p in parts}),parts=parts,
        evidence_mode_counts=mode_counts,action_witness=witness,frame_matches=matches,known_compound_validation=compound_errors,
        counter_note='Diagnostic physics_step was its explicit capture-call index. Actual native tick is matched using recorded SimulationContext clock; reset performed two warmup steps.',
        source_files={str(p.resolve().relative_to(ROOT)).replace('\\','/'):sha(p) for p in sources},source_asset_layers=summary['source_asset_layers'],
        runtime_physx_binding=geom['runtime_physx_binding'],articulation_data_source_sha256=link['class_source_sha256'],
        usd_inventory_template=geom['colliders'],native_query_template=_query_template(summary),
        extractor_source_sha256=native['extractor_source_sha256'],verifier_source_sha256=sha(__file__))
    write_new(output,library);return library


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def query_native_inventory(stage,robot):
    from pxr import Usd,UsdPhysics,UsdUtils,PhysicsSchemaTools
    from omni.physx import get_physx_property_query_interface
    root=stage.GetPrimAtPath('/World/envs/env_0/Robot');names=list(robot.body_names)
    paths={p.GetName():str(p.GetPath()) for p in Usd.PrimRange(root,Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    rows=[];errors=[];stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    for index,name in enumerate(names):
        row=dict(body_index=index,body_name=name,body_path=paths.get(name),finished=False,rigid_body_results=[],colliders=[])
        if name not in paths:errors.append('MISSING_BODY:'+name);rows.append(row);continue
        def finished(record=row):record['finished']=True
        def rigid(response,record=row):record['rigid_body_results'].append(str(response.result))
        def collider(response,record=row):
            record['colliders'].append(dict(ordinal=len(record['colliders']),path=str(PhysicsSchemaTools.intToSdfPath(response.path_id)),
                result=str(response.result),volume=float(response.volume),aabb_local_min=list(response.aabb_local_min),
                aabb_local_max=list(response.aabb_local_max),local_pos=list(response.local_pos),local_rot_xyzw=list(response.local_rot)))
        try:
            get_physx_property_query_interface().query_prim(stage_id=stage_id,
                prim_id=PhysicsSchemaTools.sdfPathToInt(paths[name]),timeout_ms=5000,
                finished_fn=finished,rigid_body_fn=rigid,collider_fn=collider)
        except Exception as exc:errors.append(name+':'+type(exc).__name__+':'+str(exc))
        if not row['finished']:errors.append('CALLBACK_NOT_FINISHED_SYNCHRONOUSLY:'+name)
        if not row['rigid_body_results'] or any(x!='PhysxPropertyQueryResult.VALID' for x in row['rigid_body_results']):errors.append('RIGID_BODY_QUERY_INVALID:'+name)
        if any(x['result']!='PhysxPropertyQueryResult.VALID' for x in row['colliders']):errors.append('COLLIDER_QUERY_INVALID:'+name)
        rows.append(row)
    return dict(schema='round4070-g.native_property_inventory.v1',status='CAPTURED' if not errors else 'NOT_VERIFIED',
        evaluation_only=True,api='IPhysxPropertyQuery.query_prim QUERY_RIGID_BODY_WITH_COLLIDERS',
        source_sha256=sha(__file__),body_count=len(names),shape_count=sum(len(r['colliders']) for r in rows),
        body_names=names,bodies=rows,errors=errors,
        scope='Native cooking-derived identity/volume/bounds/pose fingerprint only; bounds are not substituted as actual collision geometry')


def bind_trial_geometry(run_root,contract,*,write_output=True,archived_query_source_sha256=None):
    """Verify an episode's actual scene against the frozen native library.

    contract is the frozen action sidecar, including exact library/proof SHAs.
    Produces a new per-trial audit artifact; never changes raw geometry/history.
    """
    from g1_action_contract_v2 import validate_contract
    validate_contract(contract)
    run=Path(run_root);geometry_path=run/'g_robot_collision_geometry.json';geometry=read(geometry_path)
    witness_contract=contract['witness'];library_path=(ROOT/witness_contract['native_library_path']).resolve()
    proof_path=(ROOT/witness_contract['source_evidence_path']).resolve()
    if not library_path.is_relative_to(ROOT) or not proof_path.is_relative_to(ROOT):raise ValueError('Native frozen evidence outside repository')
    if sha(library_path)!=witness_contract['native_library_sha256'] or sha(proof_path)!=witness_contract['source_evidence_sha256']:raise ValueError('Frozen native library/proof SHA mismatch')
    library=read(library_path);proof=read(proof_path)
    if library.get('status')!='VERIFIED_MIXED_EXACT_AND_CONSERVATIVE' or proof.get('status')!='VERIFIED':raise ValueError('Native calibration is not verified')
    if proof.get('native_library_sha256')!=sha(library_path):raise ValueError('Native calibration proof/library binding mismatch')
    if geometry.get('stage_meters_per_unit')!=1. or geometry.get('complete_robot_prim_traversal_with_instance_proxies') is not True:raise ValueError('Runtime units/inventory incomplete')
    if geometry['runtime_physx_binding']['module_sha256']!=library['runtime_physx_binding']['module_sha256']:raise ValueError('Runtime native PhysX binding changed')
    data_source=geometry.get('runtime_articulation_data',{})
    if data_source.get('status')!='CAPTURED' or data_source.get('source_sha256')!=library['articulation_data_source_sha256']:raise ValueError('Runtime link-frame implementation changed/unavailable')
    runtime_layers={str(Path(p).resolve()).casefold():digest for p,digest in geometry.get('source_asset_layers',{}).items()}
    for name,digest in library['source_asset_layers'].items():
        path=Path(name)
        if runtime_layers.get(str(path.resolve()).casefold())!=digest or not path.is_file() or sha(path)!=digest:raise ValueError('Runtime asset layer changed: '+name)
    names=geometry['body_names']
    if len(names)!=len(set(names)) or names!=library['body_names']:raise ValueError('Runtime body order/coverage differs from frozen action indices')
    template={c['prim_path']:c for c in library['usd_inventory_template'] if c['enabled']}
    actual={c['prim_path']:c for c in geometry['colliders'] if c['enabled']}
    if set(actual)!=set(template):raise ValueError('Runtime enabled collision prim paths changed')
    for path,old in template.items():
        now=actual[path]
        if now['body_name']!=old['body_name'] or now['body_index']!=names.index(now['body_name']) or now['type']!=old['type']:raise ValueError('Runtime collider/body mapping changed: '+path)
        if not np.allclose(now['local_to_body_matrix4'],old['local_to_body_matrix4'],rtol=0,atol=1e-6):raise ValueError('Runtime collider scale/pose changed: '+path)
        # Collision prim attributes and relationships are all static identity.
        # Ancestor velocity/mass state is deliberately not a collision flag;
        # ancestor collision/material/filter/owner flags must still match.
        for field in ['physics_attributes','relationships']:
            if now.get(field,{})!=old.get(field,{}):raise ValueError('Runtime collision attributes/material relationships changed: '+path+':'+field)
        def collision_properties(value):
            return {k:v for k,v in value.items() if any(word in k.lower() for word in ['collision','material','filter','simulationowner']) or k=='physics:rigidBodyEnabled'}
        for field in ['ancestor_physics_attributes','ancestor_relationships']:
            if collision_properties(now.get(field,{}))!=collision_properties(old.get(field,{})):raise ValueError('Runtime ancestor collision/filter state changed: '+path+':'+field)
    native=geometry.get('native_property_query_inventory',{})
    query_source=archived_query_source_sha256 or sha(__file__)
    if archived_query_source_sha256 is not None:
        runner=read(run/'g_runner_contract.json');source=runner['runtime_source_archive']['physics_native_bind.py'];archive=Path(source['path'])
        if source['sha256']!=query_source or not archive.is_file() or sha(archive)!=query_source:raise ValueError('Archived native query source unavailable or changed')
    if native.get('status')!='CAPTURED' or native.get('shape_count')!=library['native_collision_shape_count'] or native.get('body_names')!=names or native.get('source_sha256')!=query_source:raise ValueError('Runtime native query not complete or query source differs')
    queried={r['body_name']:r for r in native['bodies']}
    if len(native['bodies'])!=len(names) or set(queried)!=set(names):raise ValueError('Runtime native body query coverage/duplicates changed')
    for old in library['native_query_template']:
        now=queried.get(old['body_name'],{})
        if now.get('finished') is not True or not now.get('rigid_body_results') or any(x!='PhysxPropertyQueryResult.VALID' for x in now['rigid_body_results']):raise ValueError('Native body query incomplete or invalid')
        if now.get('body_path')!=old['body_path'] or len(now.get('colliders',[]))!=len(old['colliders']):raise ValueError('Native body/shape counts changed')
        # Keep duplicate same-path native shape records; each ordinal matters.
        for expected,actual_query in zip(old['colliders'],now['colliders']):
            if expected['ordinal']!=actual_query.get('ordinal') or expected['path']!=actual_query['path'] or actual_query.get('result')!='PhysxPropertyQueryResult.VALID':raise ValueError('Native collider ordinal/path/result changed')
            for field in ['aabb_local_min','aabb_local_max','local_pos','local_rot_xyzw']:
                if not np.allclose(expected[field],actual_query[field],rtol=0,atol=1e-6):raise ValueError('Native collider fingerprint changed: '+field)
            if not np.isclose(expected['volume'],actual_query['volume'],rtol=1e-6,atol=1e-10):raise ValueError('Native collider volume changed')
    parts=[]
    for source in library['parts']:
        part=dict(source,body_index=names.index(source['body_name']))
        if 'eSIMULATION_SHAPE' not in part.get('shape_flags',[]):raise ValueError('Native nonsimulation shape cannot be a collider')
        parts.append(part)
    action_witness=dict(library['action_witness']);action_witness['body_index']=names.index(action_witness['body_name'])
    if action_witness['body_name']!=witness_contract['body_name'] or not np.allclose(action_witness['point_body_m'],witness_contract['point_body_m'],rtol=0,atol=1e-12):raise ValueError('Action witness differs from frozen native proof')
    witness_part=parts[action_witness['part_index']]
    witness_sha=hashlib.sha256(json.dumps(witness_part,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if action_witness['body_index']!=witness_contract['body_index'] or witness_part['shape_prim_path']!=witness_contract['native_shape_id']:raise ValueError('Frozen action witness body index/native shape mismatch')
    if witness_sha!=witness_contract['native_geometry_sha256'] or action_witness['native_part_geometry_sha256']!=witness_sha:raise ValueError('Frozen action witness native part SHA mismatch')
    if abs(action_witness['actual_inner_ball_radius_m']-witness_contract['verified_interior_radius_m'])>1e-12:raise ValueError('Frozen action witness interior radius mismatch')
    result=dict(schema='round4070-g.verified_native_robot_geometry.v1',status='VERIFIED',geometry_complete=True,frame_verified=True,
        all_geometry_exact=False,evidence_mode_counts=library['evidence_mode_counts'],body_names=names,parts=parts,
        native_collision_shape_count=len(parts),expected_collision_body_indices=sorted({p['body_index'] for p in parts}),
        action_witness=action_witness,runtime_geometry_sha256=sha(geometry_path),
        native_library_path=str(library_path.relative_to(ROOT)).replace('\\','/'),native_library_sha256=sha(library_path),
        source_evidence_path=str(proof_path.relative_to(ROOT)).replace('\\','/'),source_evidence_sha256=sha(proof_path),
        binder_source_sha256=query_source,checked=['ASSET_LAYER_BYTES','PHYSX_BINARY','ISAACLAB_LINK_FRAME_SOURCE','ALL_BODY_NAMES','ENABLED_COLLISION_PRIM_PATHS',
            'COLLIDER_BODY_MAPPING','COLLIDER_LOCAL_TRANSFORMS_AND_SCALE','COLLIDER_ATTRIBUTES_AND_MATERIAL_RELATIONSHIPS','ANCESTOR_COLLISION_FILTER_FLAGS','ALL_NATIVE_QUERY_SHAPES_AND_FINGERPRINTS','SIMULATION_SHAPE_FLAGS','FROZEN_EXACT_INTERIOR_WITNESS'])
    if not write_output:return result
    output=run/'g_native_geometry_verified.json';write_new(output,result)
    return dict(path=output.name,sha256=sha(output),status=result['status'],evidence_mode_counts=result['evidence_mode_counts'])
