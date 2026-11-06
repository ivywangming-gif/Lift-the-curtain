"""One-step F05 readback diagnostic; preserves the frozen scene and thresholds.

This records authored state, reset/warmup state, the existing velocity-zero
operation and the first explicit step. It is not a new passing physics gate.
No robot, policy, alternative collision filtering, or parameter tuning is used.
"""
from pathlib import Path
import argparse,datetime as dt,hashlib,json,os,sys,traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from lift_curtain.experiments.stage_log import StageLogger
pre=argparse.ArgumentParser(add_help=False)
pre.add_argument('--run-root',type=Path,required=True)
pre_args,_=pre.parse_known_args()
if (pre_args.run_root/'summary.json').exists():raise FileExistsError('Never overwrite a diagnostic')
LOGGER=StageLogger(pre_args.run_root)
LOGGER.mark('BEFORE_APP',details={'pid':os.getpid(),'scope':'F05 exact frozen initialization readback; one explicit step; no G1'})
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-root',type=Path,required=True)
parser.add_argument('--scene-config-json',type=Path,required=True)
parser.add_argument('--seed',type=int,default=40701)
AppLauncher.add_app_launcher_args(parser)
ARGS=parser.parse_args();APP=AppLauncher(ARGS).app
LOGGER.mark('AFTER_APP')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path,value):
    with Path(path).open('x',encoding='utf-8') as handle:
        json.dump(value,handle,indent=2,allow_nan=False)


def main():
    import torch
    import omni.kit.app
    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext,SimulationCfg
    from isaaclab.assets import RigidObject,RigidObjectCfg
    from isaacsim.core.utils.stage import get_current_stage
    from isaacsim.core.simulation_manager import SimulationManager
    from pxr import Gf,UsdGeom,UsdPhysics,PhysxSchema
    from physics_scene_builder import create_scene,array
    from physics_scenes import physical_hash
    run=ARGS.run_root
    summary=dict(schema='round4070-g.f05_initialization_readback.v1',status='NOT_COMPLETED',
        physics_only=True,g1_budget_consumed=0,robot_instantiated=False,FALCON_loaded=False,
        scientific_gate_admission=False,seed=ARGS.seed,explicit_step_budget=1,
        frozen_parameters_or_thresholds_changed=False,
        scope='Read-only checkpoints around the existing reset, velocity-zero and first explicit physics step; not a 24-second gate')
    stages=[]
    try:
        manifest_path=ROOT/'outputs/round4070_comprehensive_traversability/campaign_manifest_v5.json'
        manifest_sha=json.loads(manifest_path.with_suffix('.sha256.json').read_text())['sha256']
        if sha(manifest_path)!=manifest_sha:raise ValueError('Frozen V5 manifest SHA mismatch')
        spec=json.loads(ARGS.scene_config_json.read_text(encoding='utf-8'))
        frozen=json.loads(manifest_path.read_text())['scene_specs']['F05']
        if spec!=frozen or spec['scene_id']!='F05' or physical_hash(spec)!=spec['physical_config_id']:
            raise ValueError('Only the exact frozen F05 scene is permitted')
        if ARGS.seed not in json.loads(manifest_path.read_text())['physics_seeds']:
            raise ValueError('Seed is not a frozen physics seed')
        source_paths=[Path(__file__),ARGS.scene_config_json,
            ROOT/'scripts/round4070_g/physics_scene_builder.py',ROOT/'scripts/round4070_g/physics_scenes.py',
            ROOT/'scripts/round4070_g/physics_gate.py',ROOT/'scripts/round4070_fixed_probe_benchmark/structural_curtain_asset.py',
            Path(sys.modules[SimulationContext.__module__].__file__),Path(sys.modules[SimulationManager.__module__].__file__)]
        archive=run/'diagnostic_sources';archive.mkdir(exist_ok=False)
        source_rows=[]
        for index,path in enumerate(source_paths):
            content=path.read_bytes();target=archive/f'{index:02d}_{path.name}'
            target.write_bytes(content)
            source_rows.append(dict(original_path=str(path.resolve()),path=target.relative_to(run).as_posix(),sha256=sha(target)))
        summary.update(manifest_sha256=manifest_sha,physical_config_id=spec['physical_config_id'],source_files=source_rows)
        np.random.seed(ARGS.seed);torch.manual_seed(ARGS.seed)
        sim=SimulationContext(SimulationCfg(dt=.005,render_interval=4,device='cuda:0'))
        ground=sim_utils.CuboidCfg(size=(8.,8.,.10),collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.005,rest_offset=0.))
        ground.func('/World/defaultGroundPlane',ground,translation=(0.,0.,-.05))
        stage=get_current_stage()
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                scene=UsdPhysics.Scene(prim);scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0,0,-1));scene.CreateGravityMagnitudeAttr().Set(9.81)
        bundle=create_scene(stage,spec)
        rng=np.random.default_rng(ARGS.seed)
        start=np.array([spec['geometry']['center_x_m']-.30,
            spec['geometry']['center_y_m']+float(rng.uniform(-.008,.008)),.95+float(rng.uniform(-.005,.005))])
        probe=RigidObject(RigidObjectCfg(prim_path='/World/GPhysicsProbe',spawn=sim_utils.CuboidCfg(
            size=(.12,.12,.12),mass_props=sim_utils.MassPropertiesCfg(mass=.15),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True,max_depenetration_velocity=.5),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.005,rest_offset=0.)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(start))))

        def clock():
            return dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                simulation_context_time_s=float(sim.current_time),
                simulation_context_step_index=int(sim.current_time_step_index),
                native_simulation_manager_time_s=float(SimulationManager.get_simulation_time()),
                native_simulation_manager_step_count=int(SimulationManager.get_num_physics_steps()),
                physics_dt_s=float(sim.get_physics_dt()))

        def save_arrays(name,positions,velocities=None,rest=None,extra=None):
            path=run/(name+'.npz')
            if path.exists():raise FileExistsError(path)
            values={'positions_world':np.asarray(positions,dtype=np.float64)}
            if velocities is not None:values['velocities_world']=np.asarray(velocities,dtype=np.float64)
            if rest is not None:values['native_rest_positions_world']=np.asarray(rest,dtype=np.float64)
            np.savez_compressed(path,**values)
            row=dict(stage=name,clock=clock(),path=path.name,sha256=sha(path),node_count=len(positions),
                position_finite=bool(np.isfinite(positions).all()),extra=extra or {})
            if velocities is not None:
                row.update(velocity_finite=bool(np.isfinite(velocities).all()),maximum_node_speed_mps=float(np.linalg.norm(velocities,axis=1).max()))
            stages.append(row);json_write(run/(name+'.json'),row)
            print('DIAGNOSTIC_CHECKPOINT',name,json.dumps(row['clock']),flush=True)
            return row

        cache=UsdGeom.XformCache();authored=[];prims=[]
        for prim in stage.Traverse():
            if str(prim.GetPath()).startswith('/World/GScene'):
                prims.append(dict(path=str(prim.GetPath()),type=prim.GetTypeName(),apis=list(prim.GetAppliedSchemas()),
                    attributes={a.GetName():str(a.Get()) for a in prim.GetAttributes()},
                    relationships={r.GetName():[str(t) for t in r.GetTargets()] for r in prim.GetRelationships()}))
        for info in bundle.infos:
            mesh=UsdGeom.Mesh(stage.GetPrimAtPath(info.prim_path));points=np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
            matrix=np.asarray(cache.GetLocalToWorldTransform(mesh.GetPrim()),dtype=float).T
            authored.append((np.c_[points,np.ones(len(points))]@matrix.T)[:,:3])
        authored=np.concatenate(authored)
        json_write(run/'authored_scene_attributes.json',dict(prims=prims,actual_constraints=bundle.constraints,
            mesh_nodes=[len(i.rest_points) for i in bundle.infos],material_or_contact_parameters_modified=False))
        save_arrays('01_authored_before_reset',authored)
        LOGGER.mark('AFTER_SCENE',details={'meshes':len(bundle.infos),'constraints':len(bundle.constraints)})
        before_reset=clock();sim.reset();after_reset=clock()
        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate('isaacsim.core.experimental.prims',True)
        deform=bundle.make_deformable();probe.reset();probe.update(.005)
        data=deform.get_nodal_positions();initial=array(data[0])[0].astype(float);rest=array(data[2])[0].astype(float)
        velocity_before=array(deform.get_nodal_velocities())[0].astype(float)
        save_arrays('02_post_reset_before_zero',initial,velocity_before,rest,dict(
            clock_immediately_before_reset=before_reset,clock_immediately_after_reset=after_reset,
            snapshot_after_experimental_view_creation=True,
            maximum_displacement_from_authored_m=float(np.linalg.norm(initial-authored,axis=1).max())))
        deform.set_nodal_velocities(np.zeros((1,len(initial),3),np.float32))
        positions_zero=array(deform.get_nodal_positions()[0])[0].astype(float)
        velocity_zero=array(deform.get_nodal_velocities())[0].astype(float)
        save_arrays('03_post_zero_readback',positions_zero,velocity_zero,rest,dict(
            applied_velocity_array_all_zero=True,maximum_velocity_abs_readback=float(np.abs(velocity_zero).max()),
            positions_unchanged_by_velocity_set=bool(np.array_equal(positions_zero,initial))))
        LOGGER.mark('AFTER_RESET',details={'runtime_nodes':len(initial)})
        sim.step(render=False);probe.update(.005);LOGGER.mark('FIRST_PHYSICS_STEP')
        positions=array(deform.get_nodal_positions()[0])[0].astype(float);velocity=array(deform.get_nodal_velocities())[0].astype(float)
        row=save_arrays('04_after_first_explicit_step',positions,velocity,rest,dict(
            maximum_displacement_from_post_reset_m=float(np.linalg.norm(positions-initial,axis=1).max()),
            maximum_displacement_from_authored_m=float(np.linalg.norm(positions-authored,axis=1).max())))
        finite=bool(np.isfinite(positions).all() and np.isfinite(velocity).all())
        top,anchor=bundle.constraint_drifts(positions,initial)
        stretch=bundle.max_neighbor_stretch(positions,rest);below=int((positions[:,2]<-.005).sum())
        summary.update(status='DIAGNOSTIC_CAPTURE_COMPLETE',checkpoints=stages,explicit_steps_executed=1,
            final_probe_position_world=array(probe.data.root_pos_w)[0].tolist(),
            same_first_step_gate_observation=dict(finite=finite,node_speed_mps=row['maximum_node_speed_mps'],
                original_speed_limit_mps=5.,neighbor_stretch=stretch,original_stretch_limit=4.,
                top_drift_m=top,anchor_drift_m=anchor,original_constraint_limit_m=.01,nodes_below_ground=below,
                original_gate_would_stop=bool(not finite or row['maximum_node_speed_mps']>5 or stretch>4 or top>.01 or anchor>.01 or below>0)),
            initialization_clock_caveat='Native warmup may propagate states before explicit step1; both independent API clocks are recorded without subtracting or guessing warmup counts',
            solver_computed_collision_offsets='NOT_OBSERVED_UNLESS_PRESENT_IN_CAPTURED_NATIVE_OR_USD_ATTRIBUTES; no default guessed')
    except Exception as exc:
        summary.update(reason=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc(),checkpoints=stages)
    finally:
        json_write(run/'summary.json',summary)
        LOGGER.finalize(status='PASS' if summary['status']=='DIAGNOSTIC_CAPTURE_COMPLETE' else 'FAIL',
            details={'diagnostic_status':summary['status'],'scientific_gate_admission':False})
        if 'sim' in locals():sim._disable_app_control_on_stop_handle=True
        APP.close()
    return 0 if summary['status']=='DIAGNOSTIC_CAPTURE_COMPLETE' else 2


if __name__=='__main__':raise SystemExit(main())
