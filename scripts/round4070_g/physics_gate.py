"""Independent 24 s simulation gate: settle, perturb, contact, release, tail.

Discrete 50 Hz mesh checks are geometric evidence, not continuous collision
certification or calibration of cloth/PVC/net material parameters.
"""
from pathlib import Path
import argparse
import csv
import json
import os
import sys
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts/round4070_f'))
from lift_curtain.experiments.stage_log import StageLogger
from physics_scenes import physical_hash
from physics_scene_builder import create_scene,array,rotation
from physics_p1_audit import triangle_box_overlap

pre=argparse.ArgumentParser(add_help=False);pre.add_argument('--run-root',type=Path,required=True)
pa,_=pre.parse_known_args();LOGGER=StageLogger(pa.run_root)
LOGGER.mark('BEFORE_APP',details={'pid':os.getpid(),'argv':sys.argv})
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-root',type=Path,required=True)
parser.add_argument('--scene-config-json',type=Path,required=True)
parser.add_argument('--seed',type=int,required=True)
parser.add_argument('--seconds',type=float,default=24.)
AppLauncher.add_app_launcher_args(parser);ARGS=parser.parse_args()
APP=AppLauncher(ARGS,multi_gpu=False).app;LOGGER.mark('AFTER_APP')
DT=.005


def write_json(path,payload):
    path.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')


def oriented_box_overlap(center,quat,half,wall,margin=0.):
    """Full 15-axis OBB/AABB SAT, including rotated probe faces and edges."""
    axes_a=rotation(quat);axes_b=np.eye(3)
    ha=np.full(3,half);hb=np.asarray(wall['size_xyz'])/2+margin
    delta=np.asarray(wall['center_xyz'])-np.asarray(center)
    axes=[axes_a[:,i] for i in range(3)]+[axes_b[:,i] for i in range(3)]
    axes += [np.cross(axes_a[:,i],axes_b[:,j]) for i in range(3) for j in range(3)]
    for axis in axes:
        if np.linalg.norm(axis)<1e-9:continue
        radius=np.sum(ha*np.abs(axes_a.T@axis))+np.sum(hb*np.abs(axes_b.T@axis))
        if abs(np.dot(delta,axis))>=radius:return False
    return True


def main():
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext,SimulationCfg
    from isaaclab.assets import RigidObject,RigidObjectCfg
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import Gf,UsdPhysics,PhysxSchema
    import omni.kit.app
    spec=json.loads(ARGS.scene_config_json.read_text(encoding='utf-8'))
    assert physical_hash(spec)==spec['physical_config_id'],'Frozen physical configuration mismatch'
    run=ARGS.run_root;report=dict(schema='round4070-g.physics-gate.v1',scene_id=spec['scene_id'],physical_config_id=spec['physical_config_id'],seed=ARGS.seed,physics_valid=False,real_material_validated=False,material_fidelity=spec['material_fidelity'],duration_requested_s=ARGS.seconds,sampling_hz=50,physics_hz=200,probe_force_cap_n=3.,discrete_collision_scope='50 Hz sampled triangle/interior box checks with 10 mm probe and 5 mm wall penetration tolerance; no continuous collision guarantee',simulation_host_configuration=dict(multi_gpu=False,active_gpu='cuda:0',physics_gpu='cuda:0',scope='Host startup stability only; no physics, controller, action, scene, or FALCON input change',reason='Single RTX 4070 host; current hybrid-GPU Kit multi-GPU startup path is unstable'))
    launch_file=run/'campaign_launch.json'
    if launch_file.exists():report['source_sha256']=json.loads(launch_file.read_text(encoding='utf-8')).get('source_sha256',{})
    rows=[];snapshots=[];velocities=[];probes=[];support_rows=[];checks={};failure=None
    try:
        np.random.seed(ARGS.seed);torch.manual_seed(ARGS.seed)
        sim=SimulationContext(SimulationCfg(dt=DT,render_interval=4,device='cuda:0'))
        ground=sim_utils.CuboidCfg(size=(8.,8.,.10),collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.005,rest_offset=0.))
        ground.func('/World/defaultGroundPlane',ground,translation=(0.,0.,-.05))
        stage=get_current_stage()
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                scene=UsdPhysics.Scene(prim);scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0,0,-1));scene.CreateGravityMagnitudeAttr().Set(9.81)
        bundle=create_scene(stage,spec)
        collision_offsets=[]
        for info in bundle.infos:
            prim=stage.GetPrimAtPath(info.prim_path)
            api=PhysxSchema.PhysxCollisionAPI(prim)
            collision_offsets.append(dict(path=info.prim_path,contact_offset_attr=api.GetContactOffsetAttr().Get(),rest_offset_attr=api.GetRestOffsetAttr().Get(),collision_attributes={a.GetName():str(a.Get()) for a in prim.GetAttributes() if any(k in a.GetName().lower() for k in ['offset','collision','thickness'])}))
        x=float(spec['geometry']['center_x_m']);y=float(spec['geometry']['center_y_m'])
        rng=np.random.default_rng(ARGS.seed);probe_y=y+float(rng.uniform(-.008,.008));probe_z=.95+float(rng.uniform(-.005,.005))
        start=np.array([x-.30,probe_y,probe_z])
        probe=RigidObject(RigidObjectCfg(prim_path='/World/GPhysicsProbe',spawn=sim_utils.CuboidCfg(size=(.12,.12,.12),mass_props=sim_utils.MassPropertiesCfg(mass=.15),rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True,max_depenetration_velocity=.5),collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.005,rest_offset=0.)),init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(start))))
        report.update(evaluation_metadata=bundle.evaluation_metadata,actual_constraints=bundle.constraints,mesh_collision_offsets=collision_offsets,mesh_nodes=[len(i.rest_points) for i in bundle.infos],mesh_triangles=[len(f) for f in bundle.faces],seed_perturbation=dict(probe_y_m=probe_y,probe_z_m=probe_z,free_node_velocity_mps=.002),apparatus_revision='v2 local-frame force conversion; finite torque damping; contact shell reported separately from strict interior penetration')
        write_json(run/'scene_actual_topology.json',report)
        LOGGER.mark('AFTER_SCENE',details={'meshes':len(bundle.infos),'constraints':len(bundle.constraints)})
        sim.reset();omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate('isaacsim.core.experimental.prims',True)
        deform=bundle.make_deformable();probe.reset();probe.update(DT)
        if deform:
            initial=array(deform.get_nodal_positions()[0])[0].astype(float)
            deform.set_nodal_velocities(np.zeros((1,len(initial),3),np.float32))
            rest=array(deform.get_nodal_positions()[2])[0].astype(float)
        else:initial=rest=np.zeros((0,3))
        LOGGER.mark('AFTER_RESET',details={'runtime_nodes':len(initial)})
        sim.step(render=False);probe.update(DT);LOGGER.mark('FIRST_PHYSICS_STEP')
        count=round(ARGS.seconds/DT);contact_seen=False;released=False
        for step in range(count+1):
            t=step*DT
            progress=np.clip((t-6)/4,0,1) if t<11 else 1-np.clip((t-11)/3,0,1)
            target=start.copy();target[0]+=.40*progress
            if step==800 and deform:
                v=rng.uniform(-.002,.002,size=(1,len(initial),3)).astype(np.float32)
                constrained={i for c in bundle.constraints for i in c['global_vertex_indices']}
                if constrained:v[:,list(constrained)]=0
                deform.set_nodal_velocities(v)
            p=array(probe.data.root_pos_w)[0];q=array(probe.data.root_quat_w)[0];vprobe=array(probe.data.root_lin_vel_w)[0]
            force=80*(target-p)-8*vprobe;norm=np.linalg.norm(force)
            if norm>3:force*=3/norm
            if step%4==0 or step==count:
                if deform:
                    current=array(deform.get_nodal_positions()[0])[0].astype(float);v=array(deform.get_nodal_velocities())[0].astype(float)
                    finite=bool(np.isfinite(current).all() and np.isfinite(v).all())
                    top,anchor=bundle.constraint_drifts(current,initial) if finite else (999.,999.)
                    speed=float(np.linalg.norm(v,axis=1).max()) if finite else 999.
                    stretch=bundle.max_neighbor_stretch(current,rest) if finite else 999.
                    below=int((current[:,2]<-.005).sum()) if finite else 999
                    disp=float(np.linalg.norm(current-initial,axis=1).max()) if finite else 999.
                else:current=v=np.zeros((0,3));finite=True;top=anchor=speed=below=disp=0.;stretch=1.
                penetration=touch=wallpenetration=0
                for offset,face in zip(bundle.offsets,bundle.faces):
                    tri=current[face+offset]
                    local=(tri-p)@rotation(q)
                    penetration+=int(triangle_box_overlap(local,np.zeros(3),half=.05).sum())
                    # The deformable collision shell can separate rendered
                    # triangles before their geometric surfaces meet. Record
                    # shell proximity together with induced deformation and
                    # bounded force response; this is not contact-force truth.
                    touch+=int(triangle_box_overlap(local,np.zeros(3),half=.085).sum())
                    wall=bundle.evaluation_metadata['wall']
                    if wall:
                        half=np.asarray(wall['size_xyz'])/2-.005
                        scaled=(tri-np.asarray(wall['center_xyz']))/half
                        wallpenetration+=int(triangle_box_overlap(scaled,np.zeros(3),half=1.).sum())
                wall=bundle.evaluation_metadata['wall']
                if wall:
                    wall_touch=oriented_box_overlap(p,q,.07,wall)
                    wall_probe_penetration=oriented_box_overlap(p,q,.05,wall,margin=-.005)
                else:wall_touch=False;wall_probe_penetration=False
                contact_seen |= bool(6<=t<=14 and ((touch>0 and disp>.02 and np.linalg.norm(force)>.05) or wall_touch))
                released=bool(t>=18 and touch==0 and not wall_touch and np.linalg.norm(p-start)<.02)
                support=bundle.support_metrics();support_rows.append(dict(time_s=t,supports=support))
                rails=[s for s in support if s['mechanism']=='PRISMATIC_RAIL']
                rail_valid=all(abs(s['dy_m'])<=.305 and s['transverse_drift_m']<=.01 and s['rotation_error_rad']<=.01 for s in rails)
                row=dict(step=step,time_s=t,finite=finite,node_speed_mps=speed,neighbor_stretch=stretch,top_relative_drift_m=top,anchor_relative_drift_m=anchor,nodes_below_ground=below,displacement_m=disp,probe_interior_triangles=penetration,wall_interior_triangles=wallpenetration,probe_wall_penetration=bool(wall_probe_penetration),probe_touch_triangles=touch,probe_x_m=float(p[0]),probe_target_x_m=float(target[0]),probe_force_n=float(np.linalg.norm(force)),rail_valid=rail_valid,release_clear=released)
                rows.append(row)
                if step%20==0:snapshots.append(current.astype(np.float32));velocities.append(v.astype(np.float32));probes.append(np.r_[t,p,q])
                if not finite or speed>5 or stretch>4 or top>.01 or anchor>.01 or below>0 or not rail_valid:
                    failure='NUMERICAL_OR_CONSTRAINT_GATE';break
            if step<count:
                # WrenchComposer caches link quaternion until reset. Explicit
                # fresh world->body conversion avoids a stale initial pose.
                angular=array(probe.data.root_ang_vel_w)[0]
                torque=-.02*angular-.03*(q[1:]*np.sign(q[0]));tn=np.linalg.norm(torque)
                if tn>.03:torque*=.03/tn
                f=torch.as_tensor(force@rotation(q),dtype=torch.float32,device=sim.device).reshape(1,1,3)
                tau=torch.as_tensor(torque@rotation(q),dtype=torch.float32,device=sim.device).reshape(1,1,3)
                probe.set_external_force_and_torque(f,tau,is_global=False);probe.write_data_to_sim()
                sim.step(render=False);probe.update(DT)
            if step%100==0:LOGGER.heartbeat(progress=step/count,details={'step':step,'sim_time_s':t})
        tail=[r for r in rows if r['time_s']>=ARGS.seconds-4];settled=[r for r in rows if 3<=r['time_s']<=4]
        checks=dict(duration_completed=bool(rows and rows[-1]['time_s']>=ARGS.seconds),finite=all(r['finite'] for r in rows),node_speed_bounded=all(r['node_speed_mps']<=5 for r in rows),stretch_bounded=all(r['neighbor_stretch']<=4 for r in rows),anchors_relative_fixed=all(max(r['top_relative_drift_m'],r['anchor_relative_drift_m'])<=.01 for r in rows),above_ground=all(r['nodes_below_ground']==0 for r in rows),initial_hanging_stable=bool(settled and max(r['node_speed_mps'] for r in settled)<.25),contact_demonstrated=contact_seen if spec['geometry']['kind']!='empty' else True,probe_nonpenetration=all(r['probe_interior_triangles']==0 and not r['probe_wall_penetration'] for r in rows),wall_nonpenetration=all(r['wall_interior_triangles']==0 for r in rows),rail_bounds_and_rotation=all(r['rail_valid'] for r in rows),release_clear=bool(tail and all(r['release_clear'] for r in tail)),late_stability=bool(tail and max(r['node_speed_mps'] for r in tail)<.25))
        report.update(checks=checks,execution_valid=True,physics_valid=bool(checks and all(checks.values())),duration_actual_s=rows[-1]['time_s'],failure_reason=failure,failure_type=None if all(checks.values()) else 'PHYSICS_GATE_FAILED',contact_demonstrated=contact_seen,summary_metrics={k:max(r[k] for r in rows) for k in ['node_speed_mps','neighbor_stretch','top_relative_drift_m','anchor_relative_drift_m','probe_interior_triangles','wall_interior_triangles']})
        np.savez_compressed(run/'physics_snapshots.npz',positions=np.asarray(snapshots),velocities=np.asarray(velocities),probe_time_pose=np.asarray(probes),rest=rest)
        write_json(run/'support_trajectory.json',support_rows)
    except Exception as exc:
        report.update(failure_reason=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc(),physics_valid=False,execution_valid=False,failure_type='IMPLEMENTATION_OR_RUNTIME_EXCEPTION')
    finally:
        if rows:
            with (run/'telemetry.csv').open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        write_json(run/'summary.json',report)
        LOGGER.finalize(status='PASS' if report['physics_valid'] else 'FAIL',details={'physics_valid':report['physics_valid'],'failure_reason':report.get('failure_reason')})
        print('G_PHYSICS_FINALIZED',report['scene_id'],report['physics_valid'],report.get('failure_reason'),flush=True)
        # Post-rollout cleanup only. IsaacLab's STOP callback otherwise enters
        # its interactive render loop while App.close is shutting down. The
        # same official flag is used by SimulationContext.reset; independent
        # native-collider diagnostic v7 verified normal stop/flush with it.
        if 'sim' in locals():
            sim._disable_app_control_on_stop_handle=True
            write_json(run/'post_finalize_cleanup.json',dict(protocol='G_POST_FINALIZE_DISABLE_INTERACTIVE_STOP_RENDER_V1',
                applied_after_summary_and_finalize=True,rollout_or_thresholds_changed=False))
        APP.close()


if __name__=='__main__':main()
