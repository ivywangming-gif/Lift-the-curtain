"""Physics-only verification of cooked collider frame and actual robot inventory.

No FALCON policy or G1 action episode is executed. This does not validate any
cloth interaction; it tests the evaluation apparatus before admitting G1 data.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,json,os,sys,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from lift_curtain.experiments.stage_log import StageLogger
pre=argparse.ArgumentParser(add_help=False);pre.add_argument('--run-root',type=Path,required=True);pa,_=pre.parse_known_args()
LOGGER=StageLogger(pa.run_root);LOGGER.mark('BEFORE_APP',details={'pid':os.getpid(),'scope':'physics-only actual collider extraction, no G1 policy'})
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);AppLauncher.add_app_launcher_args(p);ARGS=p.parse_args();APP=AppLauncher(ARGS).app;LOGGER.mark('AFTER_APP')


def main():
    from pxr import Usd,UsdGeom,UsdPhysics,UsdUtils,PhysicsSchemaTools,Gf
    import isaaclab.sim as sim_utils
    from isaacsim.core.utils.stage import get_current_stage
    from g1_audit import _cooked_convexes,dump_collision_geometry
    run=ARGS.run_root;run.mkdir(parents=True,exist_ok=True)
    report=dict(schema='round4070-g.cooked_frame_verification.v1',status='NOT_VERIFIED',scope='API coordinate frame and actual collision inventory; no robot action or material validity claim',physics_only=True,source_sha256={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in ['g1_audit.py','physics_collision_smoke.py']})
    import carb.settings
    settings=carb.settings.get_settings();pvd_key='/persistent/physics/omniPvdOvdRecordingDirectory';old_pvd_dir=settings.get(pvd_key)
    old_pvd_enabled=settings.get('/physics/omniPvdOutputEnabled')
    try:
        pvd_directory=run/'native_ovd';pvd_directory.mkdir()
        settings.set(pvd_key,str(pvd_directory)+'/');settings.set('/physics/omniPvdOutputEnabled',True)
        sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005,render_interval=4,device='cuda:0'));stage=get_current_stage();UsdGeom.SetStageMetersPerUnit(stage,1.)
        asset=ROOT/'assets/generated/g1_29dof_falcon_rubberhand.usd'
        robot_root=UsdGeom.Xform.Define(stage,'/World/envs/env_0/Robot').GetPrim();robot_root.GetReferences().AddReference(str(asset))
        body_names=[x.GetName() for x in Usd.PrimRange(robot_root,Usd.TraverseInstanceProxies()) if x.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(body_names)!=len(set(body_names)) or not body_names:raise RuntimeError('Actual rigid body inventory ambiguous or empty')
        raw=np.array([[x,y,z] for x in [-.11,.11] for y in [-.17,.17] for z in [-.23,.23]])
        faces=[[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]]
        cases=[]
        for name,scale,translation,rotation in [('Identity',[1,1,1],[0,0,3],[0,0,0]),('TranslatedRotatedScaled',[2,3,4],[2.1,-1.3,4.2],[15,30,45])]:
            path='/World/FrameTest'+name;mesh=UsdGeom.Mesh.Define(stage,path);mesh.CreatePointsAttr([Gf.Vec3f(*x) for x in raw]);mesh.CreateFaceVertexCountsAttr([3]*12);mesh.CreateFaceVertexIndicesAttr(np.array(faces).ravel().tolist());mesh.CreateSubdivisionSchemeAttr('none')
            xf=UsdGeom.XformCommonAPI(mesh);xf.SetTranslate(Gf.Vec3d(*translation));xf.SetRotate(Gf.Vec3f(*rotation));xf.SetScale(Gf.Vec3f(*scale))
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim());UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set('convexHull')
            cases.append((path,np.array(scale),np.array(translation),mesh.GetPrim()))
        compound=UsdGeom.Xform.Define(stage,'/World/FrameTestCompoundBody')
        UsdPhysics.RigidBodyAPI.Apply(compound.GetPrim()).CreateKinematicEnabledAttr(True)
        UsdGeom.XformCommonAPI(compound).SetTranslate(Gf.Vec3d(0,0,7))
        compound_owner=UsdGeom.Xform.Define(stage,str(compound.GetPath())+'/CollisionOwner')
        UsdPhysics.CollisionAPI.Apply(compound_owner.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(compound_owner.GetPrim()).CreateApproximationAttr().Set('convexHull')
        compound_cases=[]
        for index,(translation,rotation,scale) in enumerate([([.4,.2,.3],[0,0,0],[1,1,1]),([-.4,-.2,-.3],[15,30,45],[2,3,4])]):
            mesh=UsdGeom.Mesh.Define(stage,str(compound_owner.GetPath())+f'/Mesh{index}')
            mesh.CreatePointsAttr([Gf.Vec3f(*x) for x in raw]);mesh.CreateFaceVertexCountsAttr([3]*12)
            mesh.CreateFaceVertexIndicesAttr(np.array(faces).ravel().tolist());mesh.CreateSubdivisionSchemeAttr('none')
            xf=UsdGeom.XformCommonAPI(mesh);xf.SetTranslate(Gf.Vec3d(*translation));xf.SetRotate(Gf.Vec3f(*rotation));xf.SetScale(Gf.Vec3f(*scale))
            transform=np.asarray(UsdGeom.XformCache().ComputeRelativeTransform(mesh.GetPrim(),compound.GetPrim())[0],float).T
            compound_cases.append(dict(path=str(mesh.GetPath()),source_vertices=raw.tolist(),local_to_body_matrix4=transform.tolist(),
                actual_body_local_vertices=(np.c_[raw,np.ones(len(raw))]@transform.T)[:,:3].tolist()))
        report['known_compound_children']=compound_cases
        LOGGER.mark('AFTER_SCENE',details={'actual_rigid_bodies':len(body_names),'known_mesh_cases':len(cases)})
        sim.reset();LOGGER.mark('AFTER_RESET');sim.step(render=False);LOGGER.mark('FIRST_PHYSICS_STEP')
        # Use the very same IsaacLab data property as production G1, without
        # creating actuators or writing a single robot state/control value.
        from isaacsim.core.simulation_manager import SimulationManager
        from isaaclab.assets.articulation.articulation_data import ArticulationData
        articulation_roots=[str(x.GetPath()) for x in Usd.PrimRange(robot_root,Usd.TraverseInstanceProxies()) if x.HasAPI(UsdPhysics.ArticulationRootAPI)]
        if len(articulation_roots)!=1:raise RuntimeError('Actual articulation root is ambiguous: '+str(articulation_roots))
        native_view=SimulationManager.get_physics_sim_view().create_articulation_view(articulation_roots[0])
        if native_view.count!=1:raise RuntimeError('Expected exactly one actual articulation view')
        link_data=ArticulationData(native_view,device='cuda:0');link_names=list(native_view.shared_metatype.link_names)
        import inspect
        link_source=Path(inspect.getfile(ArticulationData))
        link_evidence=dict(schema='round4070-g.native_link_pose_samples.v1',body_names=link_names,
            pose_source='IsaacLab ArticulationData.body_link_pose_w on actual SimulationManager ArticulationView; no actuator instance or state writes',
            quaternion_order='WXYZ',class_source_path=str(link_source),class_source_sha256=hashlib.sha256(link_source.read_bytes()).hexdigest(),samples=[])
        def capture_links(step):
            link_data.update(.005)
            link_evidence['samples'].append(dict(physics_step=step,simulation_time_s=float(sim.current_time),
                body_link_pose_w=link_data.body_link_pose_w[0].detach().cpu().numpy().tolist(),
                body_com_pose_w=link_data.body_com_pose_w[0].detach().cpu().numpy().tolist()))
        capture_links(1)
        from omni.physx import get_physxunittests_interface
        report['native_physics_shape_statistics_before_queries']=get_physxunittests_interface().get_physics_stats()
        # Query actual native collider identities before asking the cooking API.
        # USD CollisionAPI inventory alone does not prove a native shape exists.
        from omni.physx import get_physx_property_query_interface
        native_queries=[]
        for body in [x for x in Usd.PrimRange(robot_root,Usd.TraverseInstanceProxies()) if x.HasAPI(UsdPhysics.RigidBodyAPI)]+[x[3] for x in cases]+[compound.GetPrim()]:
            query=dict(body_path=str(body.GetPath()),finished=False,colliders=[],rigid_body=[])
            def finished():query['finished']=True
            def collider(response):
                path=str(PhysicsSchemaTools.intToSdfPath(response.path_id))
                query['colliders'].append(dict(path=path,result=str(response.result),volume=float(response.volume),
                    aabb_local_min=list(response.aabb_local_min),aabb_local_max=list(response.aabb_local_max),
                    local_pos=list(response.local_pos),local_rot=list(response.local_rot)))
            def rigid(response):query['rigid_body'].append(dict(result=str(response.result)))
            get_physx_property_query_interface().query_prim(stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt(),
                prim_id=PhysicsSchemaTools.sdfPathToInt(str(body.GetPath())),timeout_ms=5000,
                finished_fn=finished,rigid_body_fn=rigid,collider_fn=collider)
            for _ in range(10):
                if query['finished']:break
                APP.update()
            native_queries.append(query)
        report['native_property_queries']=native_queries
        native_cooking={}
        for query in native_queries:
            for collider in query['colliders']:
                path=collider['path']
                if path not in native_cooking:
                    native_cooking[path]=_cooked_convexes(stage,stage.GetPrimAtPath(path))
        report['native_collider_path_cooking']=native_cooking
        outputs=[];conventions=[]
        for path,scale,translation,prim in cases:
            captured=_cooked_convexes(stage,prim)
            if not captured.get('valid') or not captured.get('convexes'):raise RuntimeError('Actual cooking callback failed: '+json.dumps(captured))
            vertices=np.concatenate([np.asarray(c['vertices'],float) for c in captured['convexes']])
            matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim),float).T
            world=(np.c_[raw,np.ones(len(raw))]@matrix.T)[:,:3]
            def error(points):return float(np.linalg.norm(vertices[:,None,:]-points[None,:,:],axis=-1).min(axis=1).max())
            errors=dict(collider_local_unscaled=error(raw),collider_local_scaled=error(raw*scale),authored_world=error(world))
            choices=[key for key,value in errors.items() if value<=2e-6]
            convention='COLLIDER_LOCAL_UNSCALED' if 'collider_local_unscaled' in choices else 'COLLIDER_LOCAL_SCALED' if 'collider_local_scaled' in choices else 'AUTHORED_WORLD' if 'authored_world' in choices else 'UNRESOLVED'
            conventions.append(convention);outputs.append(dict(prim_path=path,source_vertices=raw.tolist(),local_to_world_matrix4=matrix.tolist(),cooked=captured,nearest_source_vertex_error_m=errors,detected_convention=convention))
        frame='COLLIDER_LOCAL_UNSCALED' if all(x=='COLLIDER_LOCAL_UNSCALED' for x in conventions) else 'UNRESOLVED_OR_UNSUPPORTED'
        bundle=SimpleNamespace(infos=[],offsets=[],faces=[],evaluation_metadata={'wall':None})
        inventory=dump_collision_geometry(stage,SimpleNamespace(body_names=body_names),bundle,run)
        geom=json.loads((run/inventory['path']).read_text(encoding='utf-8'));bounds=[]
        xform_inspection=[]
        for c in geom['colliders']:
            if c['enabled'] and c['type']=='Xform':
                prim=stage.GetPrimAtPath(c['prim_path'])
                descendants=[]
                for child in Usd.PrimRange(prim,Usd.TraverseInstanceProxies()):
                    points=child.GetAttribute('points')
                    row=dict(path=str(child.GetPath()),type=child.GetTypeName(),points_count=len(points.Get() or []) if points else 0,has_collision_api=child.HasAPI(UsdPhysics.CollisionAPI),has_mesh_collision_api=child.HasAPI(UsdPhysics.MeshCollisionAPI),attributes=[a.GetName() for a in child.GetAttributes()],prim_stack=[str(s.layer.identifier)+':'+str(s.path) for s in child.GetPrimStack()])
                    if child.IsA(UsdGeom.Mesh):
                        row['direct_child_cooking_result']=_cooked_convexes(stage,child)
                        points_array=np.asarray(points.Get(),float)
                        row['authored_bounds']=[points_array.min(0).tolist(),points_array.max(0).tolist()]
                        row['relative_child_to_owner_matrix4']=np.asarray(UsdGeom.XformCache().ComputeRelativeTransform(child,prim)[0],float).T.tolist()
                    descendants.append(row)
                xform_inspection.append(dict(prim_path=c['prim_path'],descendants=descendants,direct_cooking_result=_cooked_convexes(stage,prim)))
        report['xform_collision_inspection']=xform_inspection
        report['native_physics_shape_statistics_after_queries']=get_physxunittests_interface().get_physics_stats()
        for c in geom['colliders']:
            if not c['enabled'] or c['geometry']['kind']!='mesh':continue
            authored=np.asarray(c['geometry']['authored_points'],float);cooked=c['geometry'].get('cooked_representation',{})
            valid=bool(cooked.get('valid') and cooked.get('convexes'))
            cooked_vertices=np.concatenate([np.asarray(x['vertices'],float) for x in cooked['convexes']]) if valid else np.zeros((0,3))
            error=None if not valid else float(np.max(np.abs(np.stack([authored.min(0),authored.max(0)])-np.stack([cooked_vertices.min(0),cooked_vertices.max(0)]))))
            bounds.append(dict(prim_path=c['prim_path'],cooking_valid=valid,authored_bounds=([authored.min(0).tolist(),authored.max(0).tolist()]),cooked_bounds=None if not valid else [cooked_vertices.min(0).tolist(),cooked_vertices.max(0).tolist()],maximum_bounds_difference_m=error))
        mapping_complete=not geom['unsupported_or_mapping_errors'] and all(x['body_prim_path'] for x in geom['body_coverage'])
        report.update(frame_convention=frame,known_cases=outputs,actual_robot_inventory=inventory,actual_robot_mesh_bounds=bounds,mapping_complete=mapping_complete,actual_robot_body_count=len(body_names),enabled_collision_count=inventory['enabled_collider_count'],robot_asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),stage_meters_per_unit=1.)
        report['source_asset_layers']={str(Path(layer.realPath).resolve()):hashlib.sha256(Path(layer.realPath).read_bytes()).hexdigest()
            for layer in stage.GetUsedLayers() if layer.realPath and Path(layer.realPath).is_file()}
        frame_valid=frame=='COLLIDER_LOCAL_UNSCALED'
        supported=all(c['geometry_status']!='UNSUPPORTED_COLLIDER_TYPE' for c in geom['colliders'] if c['enabled'])
        report.update(frame_validation_status='VERIFIED' if frame_valid else 'NOT_VERIFIED',actual_geometry_complete=supported)
        if frame_valid and mapping_complete and supported and all(x['cooking_valid'] for x in bounds):report['status']='VERIFIED'
        elif frame_valid:report['status']='FRAME_VERIFIED_GEOMETRY_UNRESOLVED'
        LOGGER.mark('ROLLOUT_PROGRESS',details={'diagnostic_phase':'BEFORE_EXTRA_NATIVE_CAPTURE_STEPS'})
        for step in range(3):
            sim.step(render=False);capture_links(step+2);LOGGER.mark('ROLLOUT_PROGRESS',details={'diagnostic_phase':'NATIVE_CAPTURE_STEP','index':step})
        (run/'native_link_pose_samples.json').write_text(json.dumps(link_evidence,indent=2),encoding='utf-8')
        report['native_link_pose_samples']=dict(path='native_link_pose_samples.json',sha256=hashlib.sha256((run/'native_link_pose_samples.json').read_bytes()).hexdigest())
        LOGGER.mark('ROLLOUT_PROGRESS',details={'diagnostic_phase':'BEFORE_SIM_STOP'})
        # IsaacLab's standalone STOP callback intentionally renders forever
        # while stopped. Its own reset() uses this flag to bypass that callback.
        # This diagnostic is finished simulating and needs an actual stop to
        # flush native OVD; no simulation/controller step is altered or resumed.
        assert hasattr(sim,'_disable_app_control_on_stop_handle')
        sim._disable_app_control_on_stop_handle=True
        import inspect
        sim_source=Path(inspect.getfile(type(sim)))
        report['recording_stop_apparatus_fix']=dict(reason='Bypass standalone infinite-render STOP callback after all diagnostic steps',
            source_path=str(sim_source),source_sha256=hashlib.sha256(sim_source.read_bytes()).hexdigest(),
            mechanism='Same _disable_app_control_on_stop_handle flag used by SimulationContext.reset; no resumption after stop')
        sim.stop();LOGGER.mark('ROLLOUT_PROGRESS',details={'diagnostic_phase':'AFTER_SIM_STOP'})
        APP.update();settings.set('/physics/omniPvdOutputEnabled',False);APP.update()
        LOGGER.mark('ROLLOUT_PROGRESS',details={'diagnostic_phase':'BEFORE_PVD_EXTENSION'})
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension('omni.physx.pvd');APP.update()
        LOGGER.mark('ROLLOUT_PROGRESS',details={'diagnostic_phase':'AFTER_PVD_EXTENSION'})
        from omni.physxpvd.bindings import _physxPvd
        pvd_api=_physxPvd.acquire_physx_pvd_interface();conversions=[]
        for index,ovd in enumerate(sorted(pvd_directory.rglob('*.ovd'))):
            cache=run/f'native_pvd_usd_{index:02d}';cache.mkdir()
            LOGGER.mark('ROLLOUT_PROGRESS',details={'diagnostic_phase':'BEFORE_OVD_TO_USD','input_size':ovd.stat().st_size})
            ok=bool(pvd_api.ovd_to_usd(str(ovd),str(cache)+'/',1,1))
            LOGGER.mark('ROLLOUT_PROGRESS',details={'diagnostic_phase':'AFTER_OVD_TO_USD','success':ok})
            row=dict(ovd=str(ovd.relative_to(run)),ovd_sha256=hashlib.sha256(ovd.read_bytes()).hexdigest(),
                conversion_success=ok,up_axis='Z',format='USDA',cache=str(cache.relative_to(run)),
                files={str(x.relative_to(run)):hashlib.sha256(x.read_bytes()).hexdigest() for x in cache.rglob('*') if x.is_file()})
            helper=run/'runtime_sources/physics_pvd_extract.py'
            if ok and helper.exists():
                import importlib.util
                module_spec=importlib.util.spec_from_file_location('frozen_pvd_extract',helper)
                helper_module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(helper_module)
                extract_native_geometry=helper_module.extract_native_geometry
                row['extractor_source_sha256']=hashlib.sha256(helper.read_bytes()).hexdigest()
                try:row['extraction']=extract_native_geometry(cache/'stage.usda',cache/'native_geometry.json',source_stage_meters_per_unit=float(UsdGeom.GetStageMetersPerUnit(stage)))
                except Exception as exc:row['extraction_error']=type(exc).__name__+': '+str(exc)
            conversions.append(row)
        report['native_omnipvd_capture']=conversions
    except Exception as exc:report.update(exception=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc())
    finally:
        settings.set('/physics/omniPvdOutputEnabled',bool(old_pvd_enabled))
        if old_pvd_dir is None:settings.destroy_item(pvd_key)
        else:settings.set(pvd_key,old_pvd_dir)
    (run/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8');LOGGER.finalize(status='PASS' if report['status']=='VERIFIED' else 'FAIL',details={'status':report['status'],'frame':report.get('frame_convention'),'reason':report.get('reason')});print('COLLIDER_FRAME_SMOKE_FINALIZED',report['status'],report.get('frame_convention'),report.get('reason'),flush=True);APP.close()


if __name__=='__main__':main()
