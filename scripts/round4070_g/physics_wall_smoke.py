"""Bounded physics-only actual D01 Cube/native wall-evidence diagnostic.

No robot asset, FALCON, robot action, or scientific G1 episode is instantiated.
"""
from pathlib import Path
import argparse,hashlib,json,os,sys,traceback
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from lift_curtain.experiments.stage_log import StageLogger
pre=argparse.ArgumentParser(add_help=False);pre.add_argument('--run-root',type=Path,required=True);parsed,_=pre.parse_known_args()
LOGGER=StageLogger(parsed.run_root);LOGGER.mark('BEFORE_APP',details={'pid':os.getpid(),'scope':'physics-only native static wall proof; no robot or FALCON'})
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,required=True);AppLauncher.add_app_launcher_args(parser)
ARGS=parser.parse_args();APP=AppLauncher(ARGS).app;LOGGER.mark('AFTER_APP')


def main():
    import isaaclab.sim as sim_utils
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import UsdGeom
    from physics_scene_builder import create_scene
    from physics_scenes import physical_hash
    from physics_wall_evidence import capture_wall_evidence,validate_wall_snapshot
    run=ARGS.run_root;summary=dict(schema='round4070-g.static_wall_native_diagnostic.v1',status='NOT_VERIFIED',physics_only=True,g1_budget_consumed=0,
        robot_instantiated=False,FALCON_loaded=False,scope='Actual D01 analytic static wall and native query apparatus, not a G1 label')
    try:
        manifest_path=ROOT/'outputs/round4070_comprehensive_traversability/campaign_manifest_v5.json'
        expected=json.loads(manifest_path.with_suffix('.sha256.json').read_text())['sha256']
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=expected:raise ValueError('Frozen V5 manifest SHA mismatch')
        spec=json.loads(manifest_path.read_text())['scene_specs']['D01']
        if physical_hash(spec)!=spec['physical_config_id']:raise ValueError('D01 physical configuration mismatch')
        sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005,render_interval=4,device='cuda:0',enable_scene_query_support=True))
        summary['scene_query_support_requested']=True
        summary['scene_query_support_readback']=bool(sim.get_physics_context().get_enable_scene_query_support())
        stage=get_current_stage();UsdGeom.SetStageMetersPerUnit(stage,1.)
        ground=sim_utils.CuboidCfg(size=(8.,8.,.1),collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.005,rest_offset=0.))
        ground.func('/World/defaultGroundPlane',ground,translation=(0.,0.,-.05))
        bundle=create_scene(stage,spec);LOGGER.mark('AFTER_SCENE',details={'scene_id':'D01','robot_count':0})
        sim.reset();LOGGER.mark('AFTER_RESET');sim.step(render=False);LOGGER.mark('FIRST_PHYSICS_STEP');LOGGER.mark('ROLLOUT_PROGRESS')
        binding=capture_wall_evidence(stage,bundle.evaluation_metadata['wall'],run)
        evidence=json.loads((run/binding['path']).read_text());actual=validate_wall_snapshot(evidence,bundle.evaluation_metadata['wall'])
        summary.update(status='VERIFIED',physical_config_id=spec['physical_config_id'],manifest_sha256=expected,
            wall_evidence=binding,actual_wall=actual,native_overlap_hit_count=evidence['native_presence_query']['returned_count'],
            native_ray_callback_counts=[r['callback_count'] for r in evidence['native_face_rays']],native_ray_returned_hits=[r['returned_hit'] for r in evidence['native_face_rays']])
    except Exception as exc:summary.update(reason=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        (run/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        LOGGER.finalize(status='PASS' if summary['status']=='VERIFIED' else 'FAIL',details={'wall_evidence_status':summary['status']})
        if 'sim' in locals():sim._disable_app_control_on_stop_handle=True
        APP.close()
    return 0 if summary['status']=='VERIFIED' else 2


if __name__=='__main__':raise SystemExit(main())
