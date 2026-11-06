"""Independent evaluation of synchronized actual robot/surface collision evidence.

V5 audits every actual 200 Hz step and the measured initial state in bounded
chunks. Legacy 50 Hz evidence remains explicitly separate. Neither mode claims
continuous collision detection or uses visual meshes as collision geometry.
Unsupported native geometry or an unverified frame is never admitted as UNKNOWN.
"""
from pathlib import Path
import argparse,hashlib,json,os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
from collision_geometry import GeometryNotVerified,prepare_convex,prepare_sphere,prepare_cylinder,triangle_collider_relation,convex_box_solid_relation
from collision_clearance import actual_collider_min_world_x,_rotations
from physics_admission import auditor_source_hashes
from physics_history import iterate_verified_chunks
from g1_action_contract_v2 import load_plan_contract,witness_world,separation_certificate
from physics_wall_evidence import verified_wall_for_audit


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def actual_parts(native,names):
    if native.get('schema')!='round4070-g.verified_native_robot_geometry.v1' or native.get('status')!='VERIFIED':
        raise GeometryNotVerified('NATIVE_ROBOT_GEOMETRY_NOT_INDEPENDENTLY_VERIFIED')
    if not native.get('geometry_complete') or not native.get('frame_verified'):
        raise GeometryNotVerified('NATIVE_ROBOT_GEOMETRY_OR_FRAME_INCOMPLETE')
    if native.get('body_names')!=list(names):raise GeometryNotVerified('NATIVE_BODY_ORDER_MISMATCH')
    parts=[]
    for source in native['parts']:
        part=dict(source);kind=part['type']
        if kind=='convex':part['geometry']=prepare_convex(part['planes_body_m'],vertices=part['vertices_body_m'])
        elif kind=='sphere':part['geometry']=prepare_sphere(part['radius_m'],center=part['center_body_m'])
        elif kind=='cylinder':part['geometry']=prepare_cylinder(part['radius_m'],part['half_height_m'],axis=part['axis_body'],center=part['center_body_m'])
        else:raise GeometryNotVerified('UNSUPPORTED_ACTUAL_NATIVE_GEOMETRY:'+str(kind))
        part['geometry']['evidence_mode']=part.get('evidence_mode','EXACT_NATIVE')
        part['geometry']['exact_intersection_allowed']=part.get('exact_intersection_allowed',True)
        if part.get('body_name')!=names[part['body_index']]:raise GeometryNotVerified('NATIVE_PART_BODY_MAPPING_MISMATCH')
        parts.append(part)
    expected=sorted(native['expected_collision_body_indices'])
    if sorted({p['body_index'] for p in parts})!=expected:raise GeometryNotVerified('NATIVE_PART_COVERAGE_INCOMPLETE')
    if len(parts)!=native.get('native_collision_shape_count'):raise GeometryNotVerified('NATIVE_SHAPE_COUNT_MISMATCH')
    return parts,expected


def _wall_triangles(wall):
    if wall is None:return np.zeros((0,3,3)),None
    center=np.asarray(wall['center_xyz'],float);size=np.asarray(wall['size_xyz'],float)
    if center.shape!=(3,) or size.shape!=(3,) or not np.isfinite(center).all() or not np.all(size>0):raise GeometryNotVerified('INVALID_ACTUAL_WALL_GEOMETRY')
    bounds=np.stack([center-size/2,center+size/2])
    vertices=np.array([[x,y,z] for x in bounds[:,0] for y in bounds[:,1] for z in bounds[:,2]])
    faces=np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]])
    return vertices[faces],bounds


def audit_intersections(parts,positions,quats,nodes,faces,wall=None):
    """Actual surface triangles versus every actual solid, with conservative broad phase."""
    tolerances=[0.,.001,.005];counts={str(x):0 for x in tolerances};closed=0;witnesses=[];tested=0;uncertain=[]
    wall_tris,wall_bounds=_wall_triangles(wall)
    by_body={}
    for index,part in enumerate(parts):by_body.setdefault(part['body_index'],[]).append((index,part))
    for sample in range(len(positions)):
        rotations,_=_rotations(quats[sample],1e-4)
        for body,body_parts in by_body.items():
            rotation=rotations[body];position=positions[sample,body]
            local_nodes=(nodes[sample]-position)@rotation
            cloth=local_nodes[faces]
            wall_local=(wall_tris-position)@rotation if len(wall_tris) else wall_tris
            for index,part in body_parts:
                geometry=part['geometry'];bounds=np.asarray(geometry['aabb'])
                exact=geometry.get('exact_intersection_allowed') is not False
                for surface,triangles in [('cloth',cloth),('wall',wall_local)]:
                    if not len(triangles):continue
                    mask=np.all(triangles.max(1)>=bounds[0]-1e-6,axis=1)&np.all(triangles.min(1)<=bounds[1]+1e-6,axis=1)
                    for triangle_index in np.flatnonzero(mask):
                        tri=triangles[triangle_index];tested+=1
                        zero=triangle_collider_relation(tri,geometry,penetration_tolerance=0.)
                        closed+=int(zero['intersects_closed'])
                        # All inward-offset interiors are subsets of the
                        # closed solid. A disjoint triangle cannot penetrate
                        # at any stricter tolerance; avoid duplicate clips.
                        if not zero['intersects_closed']:continue
                        if not exact:
                            if len(uncertain)<50:uncertain.append(dict(sample_index=sample,part_index=index,surface=surface,reason='NATIVE_VERTEX_CONSERVATIVE_ENVELOPE_OVERLAP_NOT_VERIFIED'))
                            continue
                        relations={0.:zero,**{tol:triangle_collider_relation(tri,geometry,penetration_tolerance=tol) for tol in tolerances[1:]}}
                        for tol in tolerances:counts[str(tol)]+=int(relations[tol]['inner_penetration'])
                        if relations[.005]['inner_penetration'] and len(witnesses)<50:
                            witness=relations[.005].get('inner_witness_local')
                            witnesses.append(dict(sample_index=sample,part_index=index,body_index=body,surface=surface,triangle_index=int(triangle_index),
                                witness_body_m=witness,witness_world_m=None if witness is None else rotation@np.asarray(witness)+position,
                                criterion='ACTUAL_SURFACE_POINT_MORE_THAN_5MM_INSIDE_EXACT_BODY',tolerance_applies_to='BODY_INTERIOR_DISTANCE',
                                strict_body_interior_clearance_m=relations[.005].get('inner_witness_clearance_m')))
                # Boundary-triangle tests alone miss a solid fully inside a wall.
                if wall_bounds is not None:
                    if part['type']=='convex':
                        solid=convex_box_solid_relation(geometry,wall_bounds,rotation,position,penetration_tolerance=.005)
                        if not solid['verified']:
                            if len(uncertain)<50:uncertain.append(dict(sample_index=sample,part_index=index,surface='wall',reason=solid['status']))
                        elif exact and solid['inner_penetration']:
                            counts['0.005']+=1
                            if len(witnesses)<50:witnesses.append(dict(sample_index=sample,part_index=index,body_index=body,surface='wall',kind='EXACT_NATIVE_COMMON_INTERIOR_BALL',criterion='COMMON_INTERIOR_BALL_RADIUS_EXCEEDS_5MM',tolerance_applies_to='BOTH_SOLIDS_COMMON_BALL',evidence=solid))
                    if not exact:continue
                    # This is a separate SINGLE-SIDED wall-depth criterion.
                    # An actual full-dimensional solid point in the open,
                    # eroded wall gives positive-volume overlap even for a
                    # small sphere whose own inradius is <=5mm. It is not a
                    # claim that both solids contain a common 5mm ball.
                    points=np.asarray(part['vertices_body_m']) if part['type']=='convex' else np.asarray([geometry['center']])
                    world=points@rotation.T+position
                    inside=np.all(world>wall_bounds[0]+.005+1e-9,axis=1)&np.all(world<wall_bounds[1]-.005-1e-9,axis=1)
                    if inside.any():
                        counts['0.005']+=1
                        if len(witnesses)<50:witnesses.append(dict(sample_index=sample,part_index=index,body_index=body,surface='wall',kind='ACTUAL_SOLID_POINT_STRICTLY_INSIDE_WALL_5MM',criterion='EXACT_SOLID_POINT_IN_5MM_ERODED_WALL_POSITIVE_VOLUME_OVERLAP',tolerance_applies_to='WALL_INTERIOR_DISTANCE_ONLY',witness_world_m=world[np.flatnonzero(inside)[0]]))
    return dict(closed_triangle_intersections=closed,strict_inner_triangle_intersections_by_tolerance_m=counts,
                narrow_phase_triangle_tests=tested,penetration_witnesses=witnesses,unverified_geometry_interactions=uncertain,
                penetration_free=False if counts['0.005'] else None if uncertain else True)


def audit_run_legacy(run):
    run=Path(run);source_names=['g_robot_collision_geometry.json','g_body_collision_history_audit_only.npz','g_outcome.json','g_body_collision_capture.json','g_trial_metadata.json']
    result=dict(schema='round4070-g.body_collision_audit.v1',status='NOT_VERIFIED',penetration_free=None,
        all_robot_collision_bodies=False,probe_followup_release_complete=False,finite=False,sample_count=0,
        scope='Discrete actual 50 Hz poststep geometry plus exact phase-final samples; no guarantee about unrecorded intersample motion',
        main_strict_penetration_tolerance_m=.005,sensitivity_tolerances_m=[0.,.001,.005],
        source_files={n:sha(run/n) for n in source_names if (run/n).is_file()},
        auditor_source_sha256=auditor_source_hashes(),
        candidate_clearance=dict(status='NOT_VERIFIED',actual_collision_geometry=False))
    try:
        missing=[n for n in source_names if not (run/n).is_file()]
        if missing:raise GeometryNotVerified('MISSING_SYNCHRONIZED_EVIDENCE:'+','.join(missing))
        geom=read(run/source_names[0]);outcome=read(run/'g_outcome.json');capture=read(run/'g_body_collision_capture.json')
        if capture.get('history_sha256')!=result['source_files']['g_body_collision_history_audit_only.npz'] or capture.get('geometry',{}).get('sha256')!=result['source_files']['g_robot_collision_geometry.json']:
            raise GeometryNotVerified('CAPTURE_SOURCE_SHA_MISMATCH')
        with np.load(run/source_names[1],allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
        time=arrays['time_s'];steps=arrays['physics_step'];positions=arrays['body_positions_world'];quats=arrays['body_quaternions_wxyz'];nodes=arrays['nodes_world'];names=arrays['body_names'].tolist();phases=arrays['phase'].astype(str);boundary=arrays['boundary'].astype(bool)
        result['sample_count']=len(time)
        if names!=geom['body_names'] or positions.shape!=(len(time),len(names),3) or quats.shape!=(len(time),len(names),4):raise GeometryNotVerified('SYNCHRONIZED_BODY_MAPPING_OR_SHAPE_MISMATCH')
        finite=all(np.isfinite(arrays[k]).all() for k in ['time_s','actual_simulation_time_s','body_positions_world','body_quaternions_wxyz','nodes_world','node_velocities_world'])
        result['finite']=finite
        if not finite:raise GeometryNotVerified('NONFINITE_RECORDED_GEOMETRY')
        if len(time)<2 or not np.all(np.diff(time)>0) or not np.all(np.diff(steps)>0):raise GeometryNotVerified('NONMONOTONE_OR_EMPTY_SYNCHRONIZED_HISTORY')
        gaps=np.diff(time);result['maximum_sample_gap_s']=float(gaps.max())
        if gaps.max()>.0200001 or np.max(np.diff(steps))>4:raise GeometryNotVerified('HISTORY_DOES_NOT_MEET_ACTUAL_50HZ_COVERAGE')
        if not np.allclose(np.diff(arrays['actual_simulation_time_s']),np.diff(steps)*.005,rtol=0,atol=1e-6):raise GeometryNotVerified('SIMULATION_CLOCK_DISAGREEMENT')
        if capture.get('episode_finalized') is not True or capture.get('synchronized_poststep') is not True or capture.get('nominal_hz')!=50:raise GeometryNotVerified('SYNCHRONIZED_EPISODE_NOT_FINALIZED')
        phase_names=set(phases);probe=any(p.startswith('physical_probe') for p in phase_names)
        recovery=any(p in phase_names for p in ['responsive_release_stand','physical_post_retract_stand'])
        final_recovery=any(boundary[i] and phases[i] in ['responsive_release_stand','physical_post_retract_stand'] for i in range(len(phases)))
        complete=bool(probe and recovery and final_recovery and outcome.get('status')=='G_FIXED_ACTION_FINISHED')
        result['probe_followup_release_complete']=complete;result['recorded_phase_names']=sorted(phase_names)
        if not complete:raise GeometryNotVerified('PROBE_FOLLOWUP_RECOVERY_HISTORY_INCOMPLETE')
        native_path=run/'g_native_geometry_verified.json'
        if not native_path.is_file():raise GeometryNotVerified('ACTUAL_NATIVE_GEOMETRY_EXTRACTION_PENDING_NO_VISUAL_SUBSTITUTE')
        native=read(native_path);result['source_files'][native_path.name]=sha(native_path)
        if native.get('runtime_geometry_sha256')!=result['source_files']['g_robot_collision_geometry.json']:raise GeometryNotVerified('NATIVE_GEOMETRY_NOT_BOUND_TO_THIS_EPISODE')
        parts,expected=actual_parts(native,names);result['all_robot_collision_bodies']=True
        faces=[];expected_nodes=0
        for sheet in geom['sheets']:
            offset=sheet['global_node_offset'];count=sheet['vertex_count'];local=np.asarray(sheet['faces'],dtype=np.int64)
            if local.ndim!=2 or local.shape[1]!=3 or local.min()<0 or local.max()>=count:raise GeometryNotVerified('INVALID_ACTUAL_SHEET_TRIANGLES')
            faces.extend((local+offset).tolist());expected_nodes+=count
        if nodes.shape!=(len(time),expected_nodes,3):raise GeometryNotVerified('SHEET_GLOBAL_NODE_COVERAGE_MISMATCH')
        faces=np.asarray(faces,dtype=np.int64).reshape(-1,3)
        intersections=audit_intersections(parts,positions,quats,nodes,faces,geom.get('wall'));result.update(intersections)
        result['status']='VERIFIED' if intersections['penetration_free'] is True else 'PHYSICS_INVALID' if intersections['penetration_free'] is False else 'NOT_VERIFIED'
        result['reason']='ALL_RECORDED_ACTUAL_COLLIDERS_AUDITED' if intersections['penetration_free'] is True else 'ACTUAL_SURFACE_SOLID_INTERIOR_INTERSECTION_BEYOND_5MM' if intersections['penetration_free'] is False else 'CONSERVATIVE_ENVELOPE_OR_NUMERICAL_INTERSECTION_UNVERIFIED'
        indices=np.flatnonzero((phases=='responsive_traverse_stand')&boundary)
        if len(indices):
            index=int(indices[-1]);clear=actual_collider_min_world_x(parts,positions[index],quats[index],expected_body_indices=expected)
            metadata=read(run/'g_trial_metadata.json');scene=metadata.get('scene_config_evaluation_only',{})
            center=scene.get('geometry',{}).get('center_x_m')
            if center is None:center=outcome.get('scene_evaluation_only',{}).get('candidate_plane_x')
            if center is None:result['candidate_clearance']['reason']='FIXED_CANDIDATE_PLANE_METADATA_MISSING'
            else:
                result['candidate_clearance']=dict(status='VERIFIED',actual_collision_geometry=True,threshold_x_m=float(center)+.10,
                    min_world_x_m=clear['min_world_x_m'],sample_index=index,time_s=float(time[index]),phase='responsive_traverse_stand',
                    phase_final_sample=True,comparison='>=',cleared=clear['min_world_x_m']>=float(center)+.10,actual_support_evidence=clear)
    except (GeometryNotVerified,KeyError,ValueError,TypeError,IndexError) as exc:
        if result['status']=='PHYSICS_INVALID':result['additional_evidence_limitations']=type(exc).__name__+': '+str(exc)
        else:result.update(status='NOT_VERIFIED',penetration_free=None,reason=type(exc).__name__+': '+str(exc))
    return result


def sheet_faces(geom):
    faces=[];expected=0
    for sheet in geom['sheets']:
        count=sheet['vertex_count'];offset=sheet['global_node_offset'];local=np.asarray(sheet['faces'],dtype=np.int64)
        if offset!=expected or count<0 or local.ndim!=2 or local.shape[1]!=3 or (local.size and (local.min()<0 or local.max()>=count)):
            raise GeometryNotVerified('INVALID_ACTUAL_SHEET_TRIANGLES_OR_OFFSETS')
        faces.extend((local+offset).tolist());expected+=count
    return np.asarray(faces,dtype=np.int64).reshape(-1,3)


class ActionReplay:
    """Independently recompute the frozen point's entire guarded state sequence."""
    def __init__(self,contract,native,runtime):
        self.contract=contract;self.runtime=runtime;self.entry=None;self.previous=None
        self.violation=None;self.violation_step=None;self.violation_point=None;self.count=0;self.recovery=False;self.maximum_dx=0.
        if runtime.get('schema')!='round4070-g.action_tube_runtime.v2' or runtime.get('enabled') is not True:
            raise GeometryNotVerified('ACTION_RUNTIME_CONTRACT_MISSING')
        expected=contract['witness'];actual=native['action_witness'];part=native['parts'][actual['part_index']]
        if actual['body_index']!=expected['body_index'] or actual['body_name']!=expected['body_name'] or not np.array_equal(actual['point_body_m'],expected['point_body_m']):
            raise GeometryNotVerified('NATIVE_ACTION_WITNESS_MAPPING_MISMATCH')
        if part.get('inside_witness_allowed') is not True or part.get('evidence_mode')!='EXACT_NATIVE' or 'eSIMULATION_SHAPE' not in part.get('shape_flags',[]):
            raise GeometryNotVerified('ACTION_WITNESS_NOT_ACTUAL_ENABLED_EXACT_SOLID')
        digest=hashlib.sha256(json.dumps(part,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if digest!=expected['native_geometry_sha256'] or part.get('shape_prim_path')!=expected['native_shape_id']:
            raise GeometryNotVerified('ACTION_WITNESS_NATIVE_GEOMETRY_HASH_OR_SHAPE_MISMATCH')
        planes=np.asarray(part['planes_body_m'],float);distance=-(planes[:,:3]@np.asarray(expected['point_body_m'])+planes[:,3])/np.linalg.norm(planes[:,:3],axis=1)
        if not np.isfinite(distance).all() or distance.min()<.006 or abs(float(distance.min())-expected['verified_interior_radius_m'])>1e-9:
            raise GeometryNotVerified('EXACT_WITNESS_INNER_BALL_NOT_VERIFIED')
        self.entry_claim=runtime.get('entry')
        if not isinstance(self.entry_claim,dict) or not isinstance(self.entry_claim.get('physics_step'),int):
            raise GeometryNotVerified('ACTION_ENTRY_ACTUAL_SAMPLE_MISSING')

    def observe(self,sample):
        step=int(sample['physics_step']);point=witness_world(sample,self.contract['witness'])
        if step<self.entry_claim['physics_step']:return point
        if self.entry is None:
            if step!=self.entry_claim['physics_step'] or sample['phase']!='physical_probe_hold' or not sample['boundary']:
                raise GeometryNotVerified('ACTION_ENTRY_NOT_ACTUAL_FINAL_PROBE_HOLD')
            if abs(float(sample['time_s'])-self.entry_claim['time_s'])>1e-9 or not np.allclose(point,self.entry_claim['witness_world_m'],rtol=0,atol=1e-9):
                raise GeometryNotVerified('ACTION_ENTRY_RUNTIME_POSE_OR_CLOCK_DISAGREEMENT')
            self.entry=dict(physics_step=step,time_s=float(sample['time_s']),witness_world_m=point.tolist())
        elif sample['phase'] in self.contract['phase_scope']['recovery_start_phases']:
            self.recovery=True
        if self.recovery:
            if str(sample['phase']) not in {'physical_retract','physical_neutralize','physical_post_retract_stand','responsive_release','responsive_neutralize','responsive_release_stand'}:
                raise GeometryNotVerified('NONRECOVERY_PHASE_AFTER_FIRST_RELEASE')
            return point
        self.count+=1;bounds=self.contract['tube_bounds_m'];reason=None
        if not np.isfinite(point).all():reason='ACTION_TUBE_NONFINITE'
        elif not(bounds['y'][0]<=point[1]<=bounds['y'][1] and bounds['z'][0]<=point[2]<=bounds['z'][1]):reason='ACTION_TUBE_BOUNDS_VIOLATION'
        if self.previous is not None:
            previous_step,old=self.previous;dx=float(abs(point[0]-old[0]));self.maximum_dx=max(self.maximum_dx,dx)
            if reason is None and step!=previous_step+1:reason='ACTION_TUBE_PHYSICS_STEP_GAP'
            if reason is None and dx>self.contract['max_step_x_m']:reason='ACTION_TUBE_STEP_X_VIOLATION'
        if self.violation is None and reason is not None:self.violation=reason;self.violation_step=step;self.violation_point=point.tolist()
        self.previous=(step,point.copy())
        return point

    def finish(self,wall,center,outcome,physics_status,scope):
        if self.entry is None or not self.recovery:raise GeometryNotVerified('ACTION_ENTRY_OR_ACTUAL_RELEASE_INTERVAL_INCOMPLETE')
        if self.runtime.get('violation')!=self.violation:raise GeometryNotVerified('ONLINE_GUARD_AND_INDEPENDENT_ACTION_REPLAY_DISAGREE')
        records=self.runtime.get('violations',[])
        if self.violation is None:
            if records:raise GeometryNotVerified('SPURIOUS_RUNTIME_GUARD_VIOLATION_RECORD')
        elif (len(records)!=1 or records[0].get('reason')!=self.violation or records[0].get('physics_step')!=self.violation_step
              or not np.allclose(records[0].get('witness_world_m'),self.violation_point,rtol=0,atol=1e-9)):
            raise GeometryNotVerified('RUNTIME_FIRST_GUARD_VIOLATION_EVIDENCE_DISAGREES')
        certificate=dict(status='NOT_APPLICABLE',reason='NO_FIXED_WALL_CERTIFICATE')
        if wall is not None and scope not in {'out_of_probe_range','DEFER_VISUAL_REOBSERVATION'}:
            candidate=separation_certificate(self.contract,wall,entry_x=self.entry['witness_world_m'][0],candidate_center_x=center)
            sufficient=(candidate['mathematically_sufficient'] and self.violation is None and physics_status=='VERIFIED'
                        and outcome.get('runtime_valid') is True and outcome.get('probe_valid') is True and outcome.get('safe_recovery') is True)
            certificate=dict(candidate,status='VERIFIED' if sufficient else 'NOT_VERIFIED',physical_evidence_verified=sufficient,
                exact_simulation_enabled_witness=True,all_guarded_actual_steps_verified=self.violation is None,
                scope='FROZEN_DISCRETE_200HZ_ACTION_SOLID_SEPARATION_CERTIFICATE')
        return dict(status='VERIFIED',action_id=self.contract['action_id'],entry_witness_world_m=self.entry['witness_world_m'],
            entry=self.entry,guard_violation=self.violation,guard_violation_physics_step=self.violation_step,
            all_required_steps_valid=self.violation is None,guarded_sample_count=self.count,maximum_actual_step_x_m=self.maximum_dx,
            release_or_retract_observed=True,static_blocking_certificate=certificate,
            scope='Every actual discrete 200Hz step after entry until first release/retract; initial entry checked; no continuous-time claim')


def audit_run_v5(run):
    run=Path(run);required=['g_robot_collision_geometry.json','g_outcome.json','g_body_collision_capture.json','g_trial_metadata.json','g_native_geometry_verified.json','g_action_tube_runtime.json','g_runner_contract.json']
    result=dict(schema='round4070-g.body_collision_audit.v1',status='NOT_VERIFIED',penetration_free=None,
        all_robot_collision_bodies=False,probe_followup_release_complete=False,finite=False,sample_count=0,
        initial_state_collision_audited=False,initial_state_collision_status='NOT_VERIFIED',geometry_checked_state_count=0,
        scope='All actual discrete 200Hz poststep states plus measured reset/RNG step0; conservative native envelopes only prove separation; no continuous-time guarantee',
        main_strict_penetration_tolerance_m=.005,sensitivity_tolerances_m=[0.,.001,.005],
        source_files={name:sha(run/name) for name in required if (run/name).is_file()},auditor_source_sha256=auditor_source_hashes(),
        candidate_clearance=dict(status='NOT_VERIFIED',actual_collision_geometry=False),
        action_contract_audit=dict(status='NOT_VERIFIED',all_required_steps_valid=False,guard_violation=None,static_blocking_certificate=dict(status='NOT_VERIFIED')))
    collision_status=True;counts={'0.0':0,'0.001':0,'0.005':0};witnesses=[];uncertain=[];closed=tested=0;history=None
    try:
        if not all((run/name).is_file() for name in required):raise GeometryNotVerified('V5_REQUIRED_RAW_EVIDENCE_MISSING')
        geom=read(run/required[0]);outcome=read(run/'g_outcome.json');capture=read(run/'g_body_collision_capture.json');metadata=read(run/'g_trial_metadata.json')
        if outcome.get('status')!='G_FIXED_ACTION_FINISHED':raise GeometryNotVerified('V5_EPISODE_NOT_FINALIZED')
        plan=metadata['plan'];contract=load_plan_contract(plan)
        native=read(run/'g_native_geometry_verified.json')
        if capture.get('native_geometry')!={'path':'g_native_geometry_verified.json','sha256':sha(run/'g_native_geometry_verified.json')}:
            raise GeometryNotVerified('CAPTURE_NATIVE_GEOMETRY_BINDING_MISMATCH')
        if native.get('runtime_geometry_sha256')!=sha(run/'g_robot_collision_geometry.json'):raise GeometryNotVerified('NATIVE_GEOMETRY_NOT_BOUND_TO_THIS_EPISODE')
        # Recheck the frozen native bytes and compare every supplied part, not
        # merely a caller-provided VERIFIED flag. Startup binder checks raw USD
        # and native property fingerprints; its exact source is launch-archived.
        root=Path(__file__).resolve().parents[2];library=read(root/contract['witness']['native_library_path'])
        for field in ['native_library_path','native_library_sha256','source_evidence_path','source_evidence_sha256']:
            if native.get(field)!=contract['witness'][field]:raise GeometryNotVerified('NATIVE_FROZEN_BINDING_MISMATCH:'+field)
        if native['parts']!=library['parts'] or native['body_names']!=library['body_names'] or native['action_witness']!=library['action_witness']:
            raise GeometryNotVerified('PER_TRIAL_NATIVE_PARTS_DIFFER_FROM_FROZEN_LIBRARY')
        from physics_native_bind import bind_trial_geometry
        rebound=bind_trial_geometry(run,contract,write_output=False,archived_query_source_sha256=native['binder_source_sha256'])
        if rebound!=native:raise GeometryNotVerified('NATIVE_BINDER_INDEPENDENT_REPLAY_DISAGREEMENT')
        wall=verified_wall_for_audit(run,geom,result['source_files'])
        parts,expected=actual_parts(native,geom['body_names']);faces=sheet_faces(geom);history=iterate_verified_chunks(run,capture,geom)
        result['all_robot_collision_bodies']=True;result['native_evidence_mode_counts']=native.get('evidence_mode_counts')
        replay=ActionReplay(contract,native,read(run/'g_action_tube_runtime.json'))
        center=float(metadata['scene_config_evaluation_only']['geometry']['center_x_m']);hold=None
        def consume_math(positions,quaternions,nodes,steps):
            nonlocal collision_status,closed,tested
            evidence=audit_intersections(parts,positions,quaternions,nodes,faces,wall)
            state=evidence['penetration_free']
            if state is False:collision_status=False
            elif state is None and collision_status is not False:collision_status=None
            closed+=evidence['closed_triangle_intersections'];tested+=evidence['narrow_phase_triangle_tests']
            for key,value in evidence['strict_inner_triangle_intersections_by_tolerance_m'].items():counts[key]+=value
            for key,target in [('penetration_witnesses',witnesses),('unverified_geometry_interactions',uncertain)]:
                for row in evidence[key][:max(0,50-len(target))]:
                    local=row['sample_index'];target.append(dict(row,physics_step=int(steps[local]),sample_index=int(steps[local])-1))
            result['geometry_checked_state_count']+=len(positions)
            return state
        initial=history.initial_arrays;pose=initial['body_link_pose_w']
        first=consume_math(pose[None,:,:3],pose[None,:,3:],initial['nodes_world'][None,:,:],np.array([0]))
        result['initial_state_collision_audited']=True
        result['initial_state_collision_status']='VERIFIED' if first is True else 'PHYSICS_INVALID' if first is False else 'NOT_VERIFIED'
        _,wall_bounds=_wall_triangles(wall)
        for arrays in history:
            steps=arrays['physics_step'];consume_math(arrays['body_positions_world'],arrays['body_quaternions_wxyz'],arrays['nodes_world'],steps)
            for index,step in enumerate(steps):
                sample={key:arrays[key][index] for key in ['physics_step','time_s','phase','boundary','body_positions_world','body_quaternions_wxyz']}
                point=replay.observe(sample)
                # An exact >=6mm interior point strictly >5mm inside the
                # actual fixed wall proves solid-solid overlap, including
                # complete containment with no boundary triangle crossing.
                if wall_bounds is not None and np.all(point>wall_bounds[0]+.005+1e-9) and np.all(point<wall_bounds[1]-.005-1e-9):
                    collision_status=False;counts['0.005']+=1
                    if len(witnesses)<50:witnesses.append(dict(physics_step=int(step),sample_index=int(step)-1,kind='FROZEN_EXACT_INTERIOR_WITNESS_INSIDE_WALL_5MM',witness_world_m=point.tolist()))
                if str(sample['phase'])=='responsive_traverse_stand' and sample['boundary']:
                    hold=dict(sample);hold['physics_step']=int(step)
            result['sample_count']+=len(steps)
        result['source_files'].update(history.summary['source_files']);result['history_verification']=history.summary
        if history.summary['status']!='VERIFIED_HISTORY' or result['geometry_checked_state_count']!=result['sample_count']+1:
            raise GeometryNotVerified('FULL_STATE_GEOMETRY_AUDIT_INCOMPLETE')
        result.update(finite=True,probe_followup_release_complete=True,maximum_sample_gap_s=.005,nominal_hz=200,
            closed_triangle_intersections=closed,narrow_phase_triangle_tests=tested,strict_inner_triangle_intersections_by_tolerance_m=counts,
            penetration_witnesses=witnesses,unverified_geometry_interactions=uncertain,penetration_free=collision_status)
        result['status']='VERIFIED' if collision_status is True else 'PHYSICS_INVALID' if collision_status is False else 'NOT_VERIFIED'
        result['reason']='ALL_ACTUAL_200HZ_STATES_AND_INITIAL_STATE_AUDITED' if collision_status is True else 'ACTUAL_SOLID_PENETRATION_BEYOND_5MM' if collision_status is False else 'CONSERVATIVE_ENVELOPE_OR_NUMERICAL_OVERLAP_NOT_VERIFIED'
        result['action_contract_audit']=replay.finish(wall,center,outcome,result['status'],metadata['scene_config_evaluation_only'].get('scope'))
        result['actual_wall_entity_audit']=dict(status='VERIFIED' if wall is not None else 'NOT_APPLICABLE',actual_geometry=wall)
        if hold is not None:
            clear=actual_collider_min_world_x(parts,hold['body_positions_world'],hold['body_quaternions_wxyz'],expected_body_indices=expected)
            result['candidate_clearance']=dict(status='VERIFIED',actual_collision_geometry=True,threshold_x_m=center+.10,
                min_world_x_m=clear['min_world_x_m'],sample_index=hold['physics_step']-1,physics_step=hold['physics_step'],time_s=float(hold['time_s']),
                phase='responsive_traverse_stand',phase_final_sample=True,comparison='>=',cleared=clear['min_world_x_m']>=center+.10,actual_support_evidence=clear)
    except (GeometryNotVerified,KeyError,ValueError,TypeError,IndexError,OSError) as exc:
        if history is not None:result['source_files'].update(history.summary['source_files'])
        result.update(status='PHYSICS_INVALID' if collision_status is False else 'NOT_VERIFIED',penetration_free=False if collision_status is False else None,
            reason=type(exc).__name__+': '+str(exc))
        result['penetration_witnesses']=witnesses;result['unverified_geometry_interactions']=uncertain
    return result


def audit_run(run):
    run=Path(run);capture=read(run/'g_body_collision_capture.json') if (run/'g_body_collision_capture.json').is_file() else {}
    metadata=read(run/'g_trial_metadata.json') if (run/'g_trial_metadata.json').is_file() else {}
    if capture.get('history_layout')=='CHUNKED_NPZ_V2' or '200HZ' in metadata.get('plan',{}).get('primary_protocol',''):
        return audit_run_v5(run)
    return audit_run_legacy(run)


def write_audit(run,result):
    run=Path(run);payload=json.dumps(result,indent=2,default=serial).encode('utf-8');digest=hashlib.sha256(payload).hexdigest()
    archive=run/f'g_collision_audit_{digest}.json'
    if not archive.exists():archive.write_bytes(payload)
    current=run/'g_collision_audit.json';temporary=run/f'g_collision_audit.{os.getpid()}.tmp';temporary.write_bytes(payload);os.replace(temporary,current)
    return archive


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,required=True);args=parser.parse_args()
    result=audit_run(args.run_root);archive=write_audit(args.run_root,result)
    print(json.dumps(dict(status=result['status'],reason=result.get('reason'),sample_count=result['sample_count'],immutable_audit=str(archive))))
    return 0 if result['status']=='VERIFIED' else 2


if __name__=='__main__':raise SystemExit(main())
