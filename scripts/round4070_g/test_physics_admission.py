import copy,json,tempfile,unittest
from pathlib import Path
from physics_admission import audit_state,empty_dependency,retry_authorized,sha,trial_state


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        workspace=Path(__file__).resolve().parents[2]
        self.tmp=tempfile.TemporaryDirectory(dir=workspace/'outputs/round4070_comprehensive_traversability/implementation_smoke')
        self.out=Path(self.tmp.name).resolve();assert self.out.is_relative_to(workspace)
        self.plan=dict(trial_id='EMPTY_left_rng1',scene_id='EMPTY',probe_side='left',posture_variant_id='nominal',
            primary_protocol='G1_RNG_V1_BODY_COLLISION_AUDIT_V1',seed=9,g1_reset_randomization={'spec_sha256':'frozen'},robot={'reset_root_y':-.004})
        self.manifest={'trial_plans':[self.plan],'scene_specs':{'EMPTY':{'geometry':{'center_x_m':.85}}}}
        self.run=self.out/'trials'/self.plan['trial_id']/'attempt01';self.run.mkdir(parents=True)
        self.outcome=dict(runtime_valid=True,probe_valid=True,safe_recovery=True,physical_probe_result={'responsive_traverse':{'stand':
            {'speed_gate_reached':True,'finite_no_fall':True,'final_planar_speed_mps':.01}}})
        self.write('g_outcome.json',self.outcome)
        (self.run/'g_robot_collision_geometry.json').write_text('{}')
        (self.run/'g_body_collision_history_audit_only.npz').write_bytes(b'historyfixture')
        (self.run/'rng.npz').write_bytes(b'actualdrawfixture')
        self.write('g_reset_randomization.json',dict(schema='G1_RNG_V1',readback_verified=True,robot_pose_readback_verified=True,
            seed=9,spec_sha256='frozen',arrays_file='rng.npz',arrays_sha256=sha(self.run/'rng.npz')))
        self.audit=dict(schema='round4070-g.body_collision_audit.v1',status='VERIFIED',sample_count=102,
            penetration_free=True,all_robot_collision_bodies=True,probe_followup_release_complete=True,finite=True,
            candidate_clearance={'status':'VERIFIED','actual_collision_geometry':True,'phase_final_sample':True,
                'phase':'responsive_traverse_stand','min_world_x_m':.96,'threshold_x_m':.95,'comparison':'>='})
        self.refresh()
        self.obstacle=dict(self.plan,trial_id='A01_left_rng1',scene_id='A01')

    def tearDown(self):self.tmp.cleanup()
    def write(self,name,value):(self.run/name).write_text(json.dumps(value))
    def refresh(self):
        self.audit['source_files']={n:sha(self.run/n) for n in ['g_robot_collision_geometry.json','g_body_collision_history_audit_only.npz','g_outcome.json']}
        self.write('g_collision_audit.json',self.audit)
    def test_complete_matching_empty(self):
        self.assertEqual(trial_state(self.run,self.plan)['status'],'COMPLETE')
        self.assertTrue(empty_dependency(self.obstacle,self.manifest,self.out)['ready'])
    def test_wrong_side_and_robot_variant(self):
        for field,value in [('probe_side','right'),('posture_variant_id','yaw'),('robot',{'reset_root_y':.006})]:
            plan=dict(self.obstacle,**{field:value});self.assertFalse(empty_dependency(plan,self.manifest,self.out)['ready'])
    def test_runtime_alone_never_admits(self):
        (self.run/'g_collision_audit.json').unlink()
        self.assertEqual(trial_state(self.run,self.plan)['status'],'NOT_VERIFIED')
        self.assertFalse(empty_dependency(self.obstacle,self.manifest,self.out)['ready'])
    def test_mutated_geometry_rejects(self):
        (self.run/'g_robot_collision_geometry.json').write_text('{"changed":true}')
        self.assertEqual(audit_state(self.run)['status'],'NOT_VERIFIED')
    def test_false_threshold_and_unsafe_hold_reject(self):
        self.audit['candidate_clearance']['threshold_x_m']=.5;self.refresh()
        self.assertFalse(empty_dependency(self.obstacle,self.manifest,self.out)['ready'])
        self.audit['candidate_clearance']['threshold_x_m']=.95
        self.outcome['physical_probe_result']['responsive_traverse']['stand']['final_planar_speed_mps']=.061
        self.write('g_outcome.json',self.outcome);self.refresh()
        self.assertFalse(empty_dependency(self.obstacle,self.manifest,self.out)['ready'])
    def test_penetration_cannot_retry(self):
        self.audit['status']='PHYSICS_INVALID';self.audit['penetration_free']=False;self.refresh()
        self.assertEqual(trial_state(self.run,self.plan)['status'],'PHYSICS_INVALID')
        self.assertFalse(retry_authorized(self.run,self.plan))
    def test_missing_rng_never_complete(self):
        (self.run/'rng.npz').unlink();self.assertEqual(trial_state(self.run,self.plan)['status'],'NOT_VERIFIED')
    def test_empty_itself_exempt(self):self.assertTrue(empty_dependency(self.plan,self.manifest,self.out)['ready'])


if __name__=='__main__':unittest.main()
