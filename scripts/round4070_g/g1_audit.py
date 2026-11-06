"""Evaluation-only actual collider and synchronized body/cloth evidence.

Nothing in this module is imported by the deployment feature exporter. USD and
PhysX imports are deferred until the existing Isaac app is already initialized.
No physics step, controller action or geometry mutation is performed here.
"""
from pathlib import Path
import functools
import inspect
import hashlib
import numpy as np
from g1_recording import write_json
EXPORTER_SOURCE_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def serial(value):
    if value is None or isinstance(value,(str,int,float,bool)):return value
    try:return [serial(x) for x in value]
    except TypeError:return str(value)


def _attributes(prim):
    return {a.GetName():serial(a.Get()) for a in prim.GetAttributes()
            if any(x in a.GetName().lower() for x in ('physics','collision','material'))}


def _relationships(prim):
    return {r.GetName():[str(x) for x in r.GetTargets()] for r in prim.GetRelationships()}


def _cooked_convexes(stage,prim):
    from pxr import UsdUtils,PhysicsSchemaTools
    from omni.physx import get_physx_cooking_interface
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
    received={}
    def callback(result,convexes):
        received['result']=str(result)
        received['valid']=result==PhysxCollisionRepresentationResult.RESULT_VALID
        received['convexes']=[dict(vertices=[list(v) for v in c.vertices],indices=list(c.indices),
            polygons=[dict(plane=list(p.plane),index_base=int(p.index_base),num_vertices=int(p.num_vertices)) for p in c.polygons]) for c in convexes]
    try:
        get_physx_cooking_interface().request_convex_collision_representation(
            stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt(),
            collision_prim_id=PhysicsSchemaTools.sdfPathToInt(str(prim.GetPath())),run_asynchronously=False,on_result=callback)
    except Exception as exc:
        received.update(valid=False,error=type(exc).__name__+': '+str(exc))
    received.setdefault('valid',False)
    received.setdefault('result','NO_SYNCHRONOUS_CALLBACK')
    received['api']='PhysXCooking.request_convex_collision_representation(run_asynchronously=False)'
    received['frame']='API_RAW_COLLIDER_LOCAL_PENDING_INDEPENDENT_FRAME_VERIFICATION'
    return received


def dump_collision_geometry(stage,robot,bundle,root):
    from pxr import Usd,UsdGeom,UsdPhysics
    root=Path(root);cache=UsdGeom.XformCache(Usd.TimeCode.Default())
    robot_root=stage.GetPrimAtPath('/World/envs/env_0/Robot')
    if not robot_root:raise RuntimeError('Actual robot USD root is missing')
    names=list(robot.body_names);colliders=[];body_paths={};errors=[]
    prims=list(Usd.PrimRange(robot_root,Usd.TraverseInstanceProxies()))
    for prim in prims:
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.GetName() in names:
            body_paths[prim.GetName()]=str(prim.GetPath())
    for prim in prims:
        if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
        api=UsdPhysics.CollisionAPI(prim);enabled=api.GetCollisionEnabledAttr().Get()
        enabled=True if enabled is None else bool(enabled)
        ancestor=prim
        while ancestor and ancestor!=robot_root and not ancestor.HasAPI(UsdPhysics.RigidBodyAPI):ancestor=ancestor.GetParent()
        body_name=ancestor.GetName() if ancestor and ancestor.HasAPI(UsdPhysics.RigidBodyAPI) else None
        body_index=names.index(body_name) if body_name in names else None
        entry=dict(prim_path=str(prim.GetPath()),body_name=body_name,body_index=body_index,
            body_prim_path=str(ancestor.GetPath()) if body_index is not None else None,
            enabled=enabled,type=prim.GetTypeName(),physics_attributes=_attributes(prim),relationships=_relationships(prim),
            ancestor_physics_attributes=_attributes(ancestor) if ancestor else {},
            ancestor_relationships=_relationships(ancestor) if ancestor else {})
        if body_index is not None:
            matrix,_=cache.ComputeRelativeTransform(prim,ancestor)
            entry['local_to_body_matrix4']=np.asarray(matrix,float).T.tolist()
        else:errors.append('COLLIDER_WITHOUT_KNOWN_BODY:'+str(prim.GetPath()))
        geometry={};kind=prim.GetTypeName()
        if kind=='Mesh':
            mesh=UsdGeom.Mesh(prim);approx=UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() or 'none'
            entry['approximation']=str(approx)
            geometry=dict(kind='mesh',authored_points=serial(mesh.GetPointsAttr().Get()),
                authored_face_vertex_indices=serial(mesh.GetFaceVertexIndicesAttr().Get()),
                authored_face_vertex_counts=serial(mesh.GetFaceVertexCountsAttr().Get()))
            if enabled:geometry['cooked_representation']=_cooked_convexes(stage,prim)
            entry['geometry_status']='COOKED_API_CAPTURED_PENDING_FRAME_VERIFICATION' if geometry.get('cooked_representation',{}).get('valid') else 'UNSUPPORTED_OR_UNVERIFIED_COOKED_GEOMETRY'
        elif kind=='Cube':
            geometry=dict(kind='cube',size=float(UsdGeom.Cube(prim).GetSizeAttr().Get()));entry['geometry_status']='AUTHORED_ANALYTIC_PRIMITIVE'
        elif kind=='Sphere':
            geometry=dict(kind='sphere',radius=float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()));entry['geometry_status']='AUTHORED_ANALYTIC_PRIMITIVE'
        elif kind in ('Capsule','Cylinder','Cone'):
            shape=getattr(UsdGeom,kind)(prim)
            geometry=dict(kind=kind.lower(),radius=float(shape.GetRadiusAttr().Get()),height=float(shape.GetHeightAttr().Get()),axis=str(shape.GetAxisAttr().Get()))
            entry['geometry_status']='AUTHORED_ANALYTIC_PRIMITIVE'
        else:geometry=dict(kind=kind);entry['geometry_status']='UNSUPPORTED_COLLIDER_TYPE'
        entry['geometry']=geometry;colliders.append(entry)
    coverage=[]
    for i,name in enumerate(names):
        owned=[c for c in colliders if c['body_index']==i]
        coverage.append(dict(body_index=i,body_name=name,body_prim_path=body_paths.get(name),
            enabled_colliders=sum(c['enabled'] for c in owned),disabled_colliders=sum(not c['enabled'] for c in owned),
            no_enabled_collider_reason=None if any(c['enabled'] for c in owned) else 'ACTUAL_USD_BODY_HAS_NO_ENABLED_COLLISION_PRIM' if name in body_paths else 'RIGID_BODY_PRIM_NOT_FOUND'))
    sheets=[]
    for info,offset,faces in zip(bundle.infos,bundle.offsets,bundle.faces):
        prim=stage.GetPrimAtPath(info.prim_path)
        sheets.append(dict(prim_path=info.prim_path,global_node_offset=int(offset),vertex_count=len(info.rest_points),
            faces=np.asarray(faces,dtype=np.int64).tolist(),fixed_vertex_indices_local=list(info.requested_fixed_vertex_indices),
            physics_attributes=_attributes(prim),relationships=_relationships(prim)))
    groups=[]
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.CollisionGroup):
            groups.append(dict(prim_path=str(prim.GetPath()),attributes={a.GetName():serial(a.Get()) for a in prim.GetAttributes()},
                               relationships=_relationships(prim)))
    from omni.physx.bindings import _physx
    binding_path=Path(_physx.__file__)
    binding_identity=dict(module_path=str(binding_path),module_sha256=hashlib.sha256(binding_path.read_bytes()).hexdigest(),
        extension_path=str(binding_path.parent.parent.parent),api='PhysXCooking.request_convex_collision_representation',
        module_version=getattr(_physx,'__version__',None))
    layers={};unhashed=[]
    for layer in stage.GetUsedLayers():
        path=Path(str(layer.realPath)) if layer.realPath else None
        if path is not None and path.is_file():layers[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
        else:unhashed.append(str(layer.identifier))
    data=getattr(robot,'data',None)
    data_type=type(data) if data is not None else None
    data_source=inspect.getsourcefile(data_type) if data_type is not None else None
    data_identity=dict(type_name=data_type.__module__+'.'+data_type.__qualname__ if data_type else None,source_path=data_source,
        source_sha256=hashlib.sha256(Path(data_source).read_bytes()).hexdigest() if data_source and Path(data_source).is_file() else None,
        status='CAPTURED' if data_source and Path(data_source).is_file() else 'NOT_VERIFIED_MISSING_ARTICULATION_DATA_SOURCE')
    try:
        from physics_native_bind import query_native_inventory
        native_inventory=query_native_inventory(stage,robot)
    except ImportError:
        native_inventory=dict(status='NOT_VERIFIED',reason='NATIVE_INVENTORY_HELPER_NOT_AVAILABLE',rows=[])
    from physics_wall_evidence import capture_wall_evidence
    expected_wall=bundle.evaluation_metadata.get('wall')
    wall_evidence=capture_wall_evidence(stage,expected_wall,root)
    result=dict(schema='round4070-g.robot_collision_geometry.v1',evaluation_only=True,model_input=False,
        exporter_source_sha256=EXPORTER_SOURCE_SHA256,runtime_physx_binding=binding_identity,
        source_asset_layers=layers,unhashed_stage_layers=unhashed,runtime_articulation_data=data_identity,
        native_property_query_inventory=native_inventory,
        stage_meters_per_unit=float(UsdGeom.GetStageMetersPerUnit(stage)),body_names=names,
        pose_source='IsaacLab body_link_pose_w; WXYZ quaternion; actual poststep state',
        transform_convention='column vector [x,y,z,1]; matrix is USD collider-local to RigidBody-ancestor local; Gf row-vector matrix transposed; coordinates in stage units',
        mesh_cooked_frame_status='PENDING_INDEPENDENT_API_FRAME_VALIDATION_NO_GUESSED_HULL_OR_VISUAL_SUBSTITUTE',
        colliders=colliders,body_coverage=coverage,sheets=sheets,wall=expected_wall,wall_evidence=wall_evidence,collision_groups=groups,
        unsupported_or_mapping_errors=errors,complete_robot_prim_traversal_with_instance_proxies=True,
        simulation_time_s=None,stage='AFTER_RESET_BEFORE_FIRST_PHYSICS_STEP')
    path=root/'g_robot_collision_geometry.json'
    if path.exists():raise FileExistsError('Never replace actual collider evidence')
    write_json(path,result)
    if expected_wall is not None and wall_evidence['status']!='VERIFIED':
        raise RuntimeError('ACTUAL_WALL_ENTITY_NOT_VERIFIED_BEFORE_FIRST_STEP: '+wall_evidence['path'])
    return dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),collider_count=len(colliders),
                enabled_collider_count=sum(c['enabled'] for c in colliders),body_count=len(names))


class BodyCollisionRecorder:
    def __init__(self,root,*,nominal_hz=50,chunk_steps=500,action_contract=None):
        if nominal_hz not in (50,200):raise ValueError('Only actual 50Hz or every-physics-step 200Hz audit is supported')
        self.root=Path(root);self.samples=[];self.names=None;self.saved=False;self.geometry=None
        self.nominal_hz=nominal_hz;self.decimation=round(200/nominal_hz);self.chunk_steps=int(chunk_steps)
        if self.chunk_steps!=500:raise ValueError('Frozen V5 chunk size is 500 actual steps')
        self.chunked=nominal_hz==200;self.chunks=[];self.total_samples=0;self.node_count=0;self.phase_names=set()
        self.phase_execution_ledger=[];self._phase_stack=[];self._last_closed_phase=None
        self.last_executed_step=None;self.last_executed_phase=None;self.last_executed_phase_id=None
        self.last_sample=None;self.boundary_requests=[]
        self.initial_state=None;self.executed_step_count=0;self.first_executed_step=None;self.first_executed_time_s=None
        self.action_contract=action_contract;self.native_geometry=None

    def should_sample(self,sim_step,local_step):
        return local_step==0 or sim_step%self.decimation==0

    def track_phase(self,fn):
        """Record an actual call even if it executes zero physics steps."""
        @functools.wraps(fn)
        def tracked(phase,*args,**kwargs):
            identity=len(self.phase_execution_ledger)
            entry=dict(phase_instance_id=identity,phase=str(phase),start_physics_step=None,end_physics_step=None,
                executed_step_count=0,final_snapshot_physics_step=None,start_time_s=None,end_time_s=None,call_status='RUNNING')
            self.phase_execution_ledger.append(entry);self._phase_stack.append(identity)
            failed=True
            try:
                result=fn(phase,*args,**kwargs);failed=False;return result
            finally:
                popped=self._phase_stack.pop()
                if popped!=identity:raise RuntimeError('Nested phase ledger mismatch')
                entry['call_status']='EXCEPTION' if failed else 'COMPLETED' if entry['executed_step_count'] else 'ZERO_STEPS'
                if self.last_sample is not None and entry['executed_step_count'] and self.last_sample['phase_instance_id']==identity and self.last_sample['physics_step']==entry['end_physics_step']:
                    entry['final_snapshot_physics_step']=entry['end_physics_step']
                self._last_closed_phase=identity
        return tracked

    def note_executed_step(self,phase,physics_step,time_s):
        if not self._phase_stack:raise RuntimeError('An actual physics step has no registered phase call')
        identity=self._phase_stack[-1];entry=self.phase_execution_ledger[identity]
        if entry['phase']!=phase:raise RuntimeError('Actual physics phase differs from active phase call')
        if self.last_executed_step is not None and physics_step!=self.last_executed_step+1:
            raise RuntimeError('Skipped or duplicated actual physics-step ledger')
        if not entry['executed_step_count']:
            entry['start_physics_step']=int(physics_step);entry['start_time_s']=float(time_s)
        entry['end_physics_step']=int(physics_step);entry['end_time_s']=float(time_s);entry['executed_step_count']+=1
        if entry['end_physics_step']-entry['start_physics_step']+1!=entry['executed_step_count']:
            raise RuntimeError('Noncontiguous actual phase execution')
        self.last_executed_step=int(physics_step);self.last_executed_phase=phase;self.last_executed_phase_id=identity
        if self.first_executed_step is None:self.first_executed_step=int(physics_step);self.first_executed_time_s=float(time_s)
        self.executed_step_count+=1

    def initialize(self,stage,robot,bundle):
        self.names=list(robot.body_names);self.geometry=dump_collision_geometry(stage,robot,bundle,self.root)
        self.node_count=sum(len(info.rest_points) for info in bundle.infos)
        if self.action_contract is not None:
            if self.nominal_hz!=200:raise RuntimeError('V5 native binding requires every actual physics step audit')
            witness=self.action_contract['witness'];index=witness['body_index']
            if index>=len(self.names) or self.names[index]!=witness['body_name']:
                raise RuntimeError('Actual body order does not match frozen action witness index')
            from physics_native_bind import bind_trial_geometry
            bind_trial_geometry(self.root,self.action_contract)
            path=self.root/'g_native_geometry_verified.json'
            self.native_geometry=dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def capture_initial_state(self,*,robot,nodes,node_velocities,actual_simulation_time_s):
        if self.initial_state is not None:raise RuntimeError('Initial reset state already captured')
        source=robot.data.body_link_pose_w
        value=source[0].detach().cpu().numpy() if hasattr(source,'detach') else np.asarray(source)[0]
        path=self.root/'g_body_collision_initial_state.npz'
        if path.exists():raise FileExistsError('Never replace actual reset state')
        np.savez_compressed(path,body_link_pose_w=np.asarray(value,dtype=np.float64),nodes_world=np.asarray(nodes,dtype=np.float32),
            node_velocities_world=np.asarray(node_velocities,dtype=np.float32),actual_simulation_time_s=float(actual_simulation_time_s),physics_step=0)
        self.initial_state=dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),physics_step=0,
            actual_simulation_time_s=float(actual_simulation_time_s),stage='AFTER_RESET_AND_RNG_BEFORE_FIRST_ACTUAL_STEP')

    def observe(self,*,robot,poses,nodes,node_velocities,phase,physics_step,time_s,actual_simulation_time_s,boundary=False):
        if self.names is None:raise RuntimeError('Body collision recorder was not initialized')
        identity=self.last_executed_phase_id
        if boundary:
            identity=self._last_closed_phase
            entry=None if identity is None else self.phase_execution_ledger[identity]
            valid=bool(entry and entry['phase']==phase and entry['executed_step_count']>0
                and entry['end_physics_step']==physics_step and identity==self.last_executed_phase_id)
            self.boundary_requests.append(dict(phase=phase,phase_instance_id=identity,requested_physics_step=int(physics_step),
                recorded=valid,reason='ACTUAL_FINAL_STEP' if valid else 'ZERO_STEPS_OR_NOT_LAST_ACTUAL_PHASE'))
            if not valid:return False
        if phase!=self.last_executed_phase or physics_step!=self.last_executed_step:
            raise RuntimeError('Snapshot is not the actual latest physics step/phase')
        # Require link poses explicitly. COM pose fallback is unacceptable for
        # transforming geometry attached to the USD rigid-body/link frame.
        source=getattr(robot.data,'body_link_pose_w',None)
        if source is None:raise RuntimeError('Actual body_link_pose_w is unavailable')
        a=source[0].detach().cpu().numpy() if hasattr(source,'detach') else np.asarray(source)[0]
        a=np.asarray(a,dtype=np.float64)
        if a.shape!=(len(self.names),7):raise RuntimeError('Unexpected body link pose shape')
        positions=np.stack([poses[n][0] for n in self.names]);quats=np.stack([poses[n][1] for n in self.names])
        if not np.allclose(a[:,:3],positions,rtol=0,atol=1e-8):raise RuntimeError('Geometry and collision pose reads do not match')
        norms=np.linalg.norm(a[:,3:7],axis=1)
        if not np.isfinite(a).all() or not np.allclose(norms,1,atol=1e-4):raise RuntimeError('Invalid body pose evidence')
        if not np.allclose(np.abs(np.sum(a[:,3:7]*quats,axis=1)),1,rtol=0,atol=2e-4):
            raise RuntimeError('Geometry and collision quaternion reads do not match')
        positions=a[:,:3];quats=a[:,3:7]
        sample=dict(time_s=float(time_s),actual_simulation_time_s=float(actual_simulation_time_s),boundary=bool(boundary),phase_instance_id=int(identity),
            physics_step=int(physics_step),phase=phase,body_positions_world=positions.copy(),body_quaternions_wxyz=quats.copy(),
            nodes_world=np.asarray(nodes,dtype=np.float32).reshape(-1,3).copy(),
            node_velocities_world=np.asarray(node_velocities,dtype=np.float32).reshape(-1,3).copy())
        if self.last_sample is not None and sample['time_s']==self.last_sample['time_s']:
            # A phase-final capture can coincide with its last periodic sample.
            # Keep one actual observation at that time, marking its boundary.
            if sample['phase_instance_id']!=self.last_sample['phase_instance_id'] or sample['phase']!=self.last_sample['phase']:
                raise RuntimeError('A boundary cannot relabel an earlier phase sample')
            sample['boundary']=bool(boundary or self.last_sample['boundary']);self.samples[-1]=sample
        else:
            # Retain the last full chunk until the next actual sample arrives,
            # so an exact phase-final marker never rewrites an immutable chunk.
            if self.chunked and len(self.samples)>=self.chunk_steps:self._flush_chunk()
            self.samples.append(sample);self.total_samples+=1
        self.last_sample=sample;self.phase_names.add(phase)
        if boundary:self.phase_execution_ledger[identity]['final_snapshot_physics_step']=int(physics_step)
        return True

    def _flush_chunk(self):
        if not self.samples:return
        payload={k:np.asarray([s[k] for s in self.samples]) for k in self.samples[0]}
        directory=self.root/'g_body_collision_history_chunks';directory.mkdir(exist_ok=True)
        path=directory/f'chunk_{len(self.chunks):06d}.npz';temporary=path.with_suffix('.tmp.npz')
        if path.exists():raise FileExistsError('Never replace a completed raw audit chunk')
        np.savez_compressed(temporary,**payload);temporary.replace(path)
        self.chunks.append(dict(path=path.relative_to(self.root).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            sample_count=len(self.samples),first_physics_step=int(payload['physics_step'][0]),last_physics_step=int(payload['physics_step'][-1]),
            first_time_s=float(payload['time_s'][0]),last_time_s=float(payload['time_s'][-1])))
        self.samples=[]

    def _write_chunk_index(self,complete):
        path=self.root/'g_body_collision_history_index.json'
        if path.exists():raise FileExistsError('Never replace a finalized chunk index')
        write_json(path,dict(schema='round4070-g.body_collision_history_chunks.v2',nominal_hz=self.nominal_hz,
            sample_count=self.total_samples,body_names=self.names,node_count=self.node_count,chunks=self.chunks,
            episode_finalized=bool(complete),model_input=False))
        return dict(history_layout='CHUNKED_NPZ_V2',history_index_path=path.name,
                    history_index_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def finish(self,complete,*,expected_total_steps=None,final_actual_simulation_time_s=None):
        if self.saved:return
        if expected_total_steps is not None and int(expected_total_steps)!=self.executed_step_count:
            raise RuntimeError('Execution counter differs from the complete phase ledger')
        if final_actual_simulation_time_s is not None and self.last_sample is not None and abs(float(final_actual_simulation_time_s)-self.last_sample['actual_simulation_time_s'])>1e-8:
            raise RuntimeError('Final simulator clock is not the final actually sampled step')
        if self.total_samples:
            if self.chunked:
                self._flush_chunk();history=self._write_chunk_index(complete)
            else:
                payload={k:np.asarray([s[k] for s in self.samples]) for k in self.samples[0]};payload['body_names']=np.asarray(self.names)
                path=self.root/'g_body_collision_history_audit_only.npz'
                if path.exists():raise FileExistsError('Never replace actual synchronized collision history')
                np.savez_compressed(path,**payload)
                history=dict(history_layout='SINGLE_NPZ_V1',history_path=path.name,history_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            query_path=self.root/'g_scene_query_configuration.json'
            query_binding={} if not query_path.exists() else dict(scene_query_configuration=dict(
                path=query_path.name,sha256=hashlib.sha256(query_path.read_bytes()).hexdigest()))
            write_json(self.root/'g_body_collision_capture.json',dict(schema='round4070-g.body_collision_capture.v1',
                evaluation_only=True,model_input=False,nominal_hz=self.nominal_hz,additional_phase_first_samples=True,additional_phase_final_samples=True,
                synchronized_poststep=True,pose_source='body_link_pose_w',quaternion_order='WXYZ',
                recorder_time_origin='absolute executed physics step from reset; first initialization step=1; original E model clock unchanged',
                sample_count=self.total_samples,body_count=len(self.names),phase_names=sorted(self.phase_names),
                episode_finalized=bool(complete),geometry=self.geometry,native_geometry=self.native_geometry,**history,**query_binding,
                phase_execution_ledger=self.phase_execution_ledger,phase_boundary_requests=self.boundary_requests,
                initial_state=self.initial_state,
                execution_anchors=dict(first_physics_step=self.first_executed_step,last_physics_step=self.last_executed_step,
                    executed_step_count=self.executed_step_count,first_time_s=self.first_executed_time_s,
                    final_time_s=None if self.last_sample is None else self.last_sample['time_s'],
                    initial_actual_simulation_time_s=None if self.initial_state is None else self.initial_state['actual_simulation_time_s'],
                    final_actual_simulation_time_s=final_actual_simulation_time_s,
                    independently_expected_total_steps=expected_total_steps),
                status='CAPTURED_REQUIRES_INDEPENDENT_COLLISION_AUDIT'))
        self.saved=True
