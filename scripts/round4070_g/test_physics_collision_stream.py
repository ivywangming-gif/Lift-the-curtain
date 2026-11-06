"""End-to-end synthetic 200Hz byte/history/geometry/action contracts.

No simulator is launched and none of these fixtures are campaign data.
"""
from pathlib import Path
import json,tempfile,unittest
import numpy as np
from test_physics_native_bind import fixture,TEMP
from physics_native_bind import bind_trial_geometry,sha,read,ROOT
from physics_g1_collision_audit import audit_run,write_audit
from physics_admission import audit_state
from g1_action_contract_v2 import witness_world


def write(path,value):path.write_text(json.dumps(value))


def stream_fixture(run,total=503,initial_penetration=False):
    contract,geom,library=fixture(run);names=geom['body_names'];b=len(names)
    nodes=np.array([[10,10,10],[10,11,10],[10,10,11.]],float)
    geom['sheets']=[dict(global_node_offset=0,vertex_count=3,faces=[[0,1,2]])]
    write(run/'g_robot_collision_geometry.json',geom)
    native=bind_trial_geometry(run,contract)
    position=np.tile([1.3,0,.8],(b,1));quaternion=np.tile([1.,0,0,0],(b,1))
    initial_nodes=nodes.copy()
    if initial_penetration:
        point=np.array(contract['witness']['point_body_m'])+position[contract['witness']['body_index']]
        initial_nodes=point+np.array([[0,-.002,0],[0,.002,0],[0,0,.002]])
    initial=run/'initial.npz'
    np.savez(initial,body_link_pose_w=np.c_[position,quaternion],nodes_world=initial_nodes,node_velocities_world=np.zeros_like(nodes),actual_simulation_time_s=.01,physics_step=0)
    ledger=[]
    ranges=[('initialization_physics_step',1,1),('physical_probe',2,10),('physical_probe_hold',11,12),
            ('responsive_traverse_stand',13,15),('responsive_release',16,18),('responsive_release_stand',19,total)]
    for identity,(phase,start,end) in enumerate(ranges):
        ledger.append(dict(phase_instance_id=identity,phase=phase,start_physics_step=start,end_physics_step=end,
            executed_step_count=end-start+1,final_snapshot_physics_step=end,call_status='COMPLETED',start_time_s=start*.005,end_time_s=end*.005))
    chunks=[]
    for start in range(1,total+1,500):
        steps=np.arange(start,min(total+1,start+500));n=len(steps)
        ids=np.array([next(i for i,e in enumerate(ledger) if e['start_physics_step']<=s<=e['end_physics_step']) for s in steps])
        arrays=dict(time_s=steps*.005,actual_simulation_time_s=.01+steps*.005,physics_step=steps,phase_instance_id=ids,
            phase=np.array([ledger[i]['phase'] for i in ids]),boundary=np.array([s==ledger[i]['end_physics_step'] for s,i in zip(steps,ids)]),
            body_positions_world=np.tile(position,(n,1,1)),body_quaternions_wxyz=np.tile(quaternion,(n,1,1)),
            nodes_world=np.tile(nodes,(n,1,1)).astype(np.float32),node_velocities_world=np.zeros((n,3,3),np.float32))
        path=run/f'chunk_{len(chunks)}.npz';np.savez(path,**arrays)
        chunks.append(dict(path=path.name,sha256=sha(path),sample_count=n,first_physics_step=int(steps[0]),last_physics_step=int(steps[-1]),first_time_s=float(steps[0]*.005),last_time_s=float(steps[-1]*.005)))
    index=dict(schema='round4070-g.body_collision_history_chunks.v2',nominal_hz=200,sample_count=total,body_names=names,node_count=3,chunks=chunks,episode_finalized=True,model_input=False)
    index_path=run/'g_body_collision_history_index.json';write(index_path,index)
    capture=dict(history_layout='CHUNKED_NPZ_V2',nominal_hz=200,episode_finalized=True,synchronized_poststep=True,
        pose_source='body_link_pose_w',quaternion_order='WXYZ',history_index_path=index_path.name,history_index_sha256=sha(index_path),
        geometry={'path':'g_robot_collision_geometry.json','sha256':sha(run/'g_robot_collision_geometry.json')},native_geometry={'path':native['path'],'sha256':native['sha256']},
        sample_count=total,body_count=b,phase_names=[x[0] for x in ranges],phase_execution_ledger=ledger,
        initial_state=dict(path=initial.name,sha256=sha(initial),physics_step=0,actual_simulation_time_s=.01,stage='AFTER_RESET_AND_RNG_BEFORE_FIRST_ACTUAL_STEP'),
        execution_anchors=dict(first_physics_step=1,last_physics_step=total,executed_step_count=total,independently_expected_total_steps=total,
            first_time_s=.005,final_time_s=total*.005,initial_actual_simulation_time_s=.01,final_actual_simulation_time_s=.01+total*.005),phase_boundary_requests=[])
    write(run/'g_body_collision_capture.json',capture)
    plan=dict(primary_protocol=contract['primary_protocol'],action_contract_v2=contract,
        action_contract_v2_sha256=sha(ROOT/'outputs/round4070_comprehensive_traversability/g1_action_contract_v2.json'),
        body_collision_audit_hz=200,followup_action_id=contract['action_id'],label_contract=contract['label_contract'])
    write(run/'g_trial_metadata.json',dict(plan=plan,scene_config_evaluation_only={'geometry':{'center_x_m':.85},'scope':'CORE'}))
    write(run/'g_outcome.json',dict(status='G_FIXED_ACTION_FINISHED',runtime_valid=True,probe_valid=True,safe_recovery=True))
    point=witness_world({'body_positions_world':position,'body_quaternions_wxyz':quaternion},contract['witness'])
    write(run/'g_action_tube_runtime.json',dict(schema='round4070-g.action_tube_runtime.v2',enabled=True,
        entry={'physics_step':12,'time_s':.06,'witness_world_m':point.tolist()},violation=None,violations=[]))
    return capture,index


class StreamingAuditTests(unittest.TestCase):
    def test_all_actual_steps_plus_step_zero_and_native_binding(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as directory:
            run=Path(directory);stream_fixture(run);result=audit_run(run)
            self.assertEqual(result['status'],'VERIFIED',result.get('reason'))
            self.assertTrue(result['initial_state_collision_audited']);self.assertEqual(result['geometry_checked_state_count'],504)
            self.assertEqual(result['sample_count'],503);self.assertEqual(result['action_contract_audit']['status'],'VERIFIED')
            self.assertTrue(result['action_contract_audit']['all_required_steps_valid'])
            self.assertEqual(result['candidate_clearance']['physics_step'],15)
            self.assertEqual(result['history_verification']['status'],'VERIFIED_HISTORY')
            self.assertIn('chunk_1.npz',result['source_files']);self.assertIn('initial.npz',result['source_files'])
            write_audit(run,result)
            self.assertEqual(audit_state(run)['status'],'VERIFIED')
            result['initial_state_collision_audited']=False;write_audit(run,result)
            self.assertEqual(audit_state(run)['reason'],'V5_INITIAL_OR_COMPLETE_ACTUAL_STEP_AUDIT_MISSING')
    def test_initial_only_penetration_cannot_hide_behind_clear_poststeps(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as directory:
            run=Path(directory);stream_fixture(run,initial_penetration=True);result=audit_run(run)
            self.assertEqual(result['status'],'PHYSICS_INVALID',result.get('reason'))
            self.assertEqual(result['initial_state_collision_status'],'PHYSICS_INVALID')
            self.assertTrue(any(x['physics_step']==0 for x in result['penetration_witnesses']))
    def test_changed_chunk_cannot_be_admitted_by_completion_flags(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as directory:
            run=Path(directory);stream_fixture(run)
            with (run/'chunk_1.npz').open('ab') as output:output.write(b'changed')
            result=audit_run(run);self.assertEqual(result['status'],'NOT_VERIFIED')
            self.assertIn('SHA256_MISMATCH',result['reason'])
    def test_runtime_collider_flag_change_rejected_by_offline_replay(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as directory:
            run=Path(directory);stream_fixture(run);geom=read(run/'g_robot_collision_geometry.json')
            geom['colliders'][0]['physics_attributes']['physics:collisionEnabled']=False;write(run/'g_robot_collision_geometry.json',geom)
            native=read(run/'g_native_geometry_verified.json');native['runtime_geometry_sha256']=sha(run/'g_robot_collision_geometry.json');write(run/'g_native_geometry_verified.json',native)
            cap=read(run/'g_body_collision_capture.json');cap['geometry']['sha256']=sha(run/'g_robot_collision_geometry.json');cap['native_geometry']['sha256']=sha(run/'g_native_geometry_verified.json');write(run/'g_body_collision_capture.json',cap)
            result=audit_run(run);self.assertEqual(result['status'],'NOT_VERIFIED');self.assertIn('attributes',result['reason'])


if __name__=='__main__':unittest.main(verbosity=2)
