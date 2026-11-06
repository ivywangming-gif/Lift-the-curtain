"""Pure CPU checks for the high-impact G logging/label/split boundaries."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np

from g1_recording import Recorder, action_spec, assign_label, online_physics_invalid, ONLINE_LABEL_REVISION, local_forward_corridor_obstructed
from g1_trial import patch_source, frozen_args, SOURCE,verify_gate_evidence,verify_empty_evidence,assert_runtime_plan_matches,post_finalize_cleanup,scene_query_required,configure_scene_query,record_scene_query_configuration
from build_dataset import legal_sequence, make_splits, schema, collision_admission, actual_collider_clearance, conservative_action_label, original_online_label,matched_pairs,frozen_evidence,actual_launch_inventory,independently_verify_wall_evidence
from g1_randomization import ResetRandomization, ROOT
from g1_audit import BodyCollisionRecorder
from physics_scenes import scene_specs
from g1_action_contract_v2 import draft_contract,validate_contract,separation_certificate,ActionTubeGuard,OUTPUT,SIDECAR_NAME


class Contracts(unittest.TestCase):
    def test_actual_launches_do_not_require_episode_metadata_or_invent_outcomes(self):
        test_root=OUTPUT/'contract_tests';test_root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_root) as folder:
            folder=Path(folder)
            for trial in ['failed','open','finished','malformed']:
                run=folder/trial/'attempt01';run.mkdir(parents=True)
                (run/'campaign_launch.json').write_text(json.dumps(dict(kind='G1',trial_id=trial,utc='SYNTHETIC_TEST_ONLY')))
            failed=folder/'failed/attempt01'
            (failed/'process_stderr.log').write_text('Traceback\nRuntimeError: frozen plan mismatch\n')
            (failed/'watchdog_summary.json').write_text(json.dumps(dict(exit_code=None,timeout_reason=None,last_valid_stage='STARTUP')))
            finished=folder/'finished/attempt01'
            (finished/'g_trial_metadata.json').write_text('{}')
            (finished/'g_outcome.json').write_text(json.dumps(dict(status='G_FIXED_ACTION_FINISHED')))
            (finished/'process_stderr.log').write_text('nonfatal shutdown diagnostic\n')
            (folder/'malformed/attempt01/g_outcome.json').write_text('[]')
            before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob('*') if p.is_file()}
            result=actual_launch_inventory(folder);rows={r['trial_id']:r for r in result['actual_launches']}
            self.assertEqual(result['actual_launch_count'],4);self.assertEqual(result['finalized_outcome_count'],1)
            self.assertEqual(result['no_metadata_count'],3);self.assertEqual(result['no_final_outcome_count'],3)
            self.assertEqual(rows['failed']['status'],'STOPPED_WITHOUT_FINAL_OUTCOME')
            self.assertEqual(rows['failed']['error'],'RuntimeError: frozen plan mismatch');self.assertIsNone(rows['failed']['watchdog_exit_code'])
            self.assertEqual(rows['open']['status'],'OPEN_OR_UNFINALIZED');self.assertIsNone(rows['open']['error'])
            self.assertEqual(rows['finished']['status'],'FINALIZED_OUTCOME_PRESENT');self.assertIsNone(rows['finished']['error'])
            self.assertTrue(rows['malformed']['read_errors'])
            self.assertEqual(before,{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob('*') if p.is_file()})

    def test_actual_failed_preapp_attempt_remains_in_launch_inventory(self):
        result=actual_launch_inventory(OUTPUT/'trials')
        suffix='G_EMPTY_left_empty_control_00_nominal_s74800_rng1_act2/attempt01'
        row=next(r for r in result['actual_launches'] if r['attempt_path'].endswith(suffix))
        self.assertFalse(row['metadata_present']);self.assertFalse(row['final_outcome_present'])
        self.assertEqual(row['status'],'STOPPED_WITHOUT_FINAL_OUTCOME');self.assertIn('frozen campaign',row['error'])

    def test_wall_entity_import_replays_sources_and_independent_audit(self):
        from test_physics_wall_evidence import fixture
        from physics_wall_evidence import validate_wall_snapshot
        from collision_geometry import GeometryNotVerified
        test_root=OUTPUT/'contract_tests';test_root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_root) as folder:
            folder=Path(folder);wall,value=fixture()
            source=Path(__file__).with_name('physics_wall_evidence.py');digest=hashlib.sha256(source.read_bytes()).hexdigest()
            value.update(source_sha256=digest,native_binding_sha256='SYNTHETIC_BINARY_SHA')
            wall_path=folder/'g_wall_collision_evidence.json';wall_path.write_text(json.dumps(value))
            wall_sha=hashlib.sha256(wall_path.read_bytes()).hexdigest()
            runner=dict(runtime_source_archive={'physics_wall_evidence.py':dict(path=str(source),sha256=digest)})
            (folder/'g_runner_contract.json').write_text(json.dumps(runner))
            geometry=dict(wall=wall,wall_evidence=dict(path=wall_path.name,sha256=wall_sha,status='VERIFIED'),
                          runtime_physx_binding=dict(module_sha256='SYNTHETIC_BINARY_SHA'))
            actual=validate_wall_snapshot(value,wall)
            audit=dict(source_files={wall_path.name:wall_sha},actual_wall_entity_audit=dict(status='VERIFIED',actual_geometry=actual))
            independently_verify_wall_evidence(folder,geometry,audit)
            with self.assertRaisesRegex(ValueError,'SOURCE_HASHES_MISSING'):independently_verify_wall_evidence(folder,geometry,{**audit,'source_files':{}})
            with self.assertRaisesRegex(ValueError,'AUDIT_REPLAY_MISMATCH'):independently_verify_wall_evidence(folder,geometry,{**audit,'actual_wall_entity_audit':{}})
            with self.assertRaises(GeometryNotVerified):independently_verify_wall_evidence(folder,dict(wall=wall),audit)
            bad=copy.deepcopy(geometry);bad['runtime_physx_binding']['module_sha256']='OTHER_BINARY'
            with self.assertRaisesRegex(GeometryNotVerified,'BINARY_MISMATCH'):independently_verify_wall_evidence(folder,bad,audit)
            runner['runtime_source_archive']['physics_wall_evidence.py']['sha256']='bad'
            (folder/'g_runner_contract.json').write_text(json.dumps(runner))
            with self.assertRaisesRegex(GeometryNotVerified,'SOURCE_ARCHIVE_MISMATCH'):independently_verify_wall_evidence(folder,geometry,audit)
            independently_verify_wall_evidence(folder,dict(wall=None),{})

    def test_checked_source_and_safety(self):
        p=patch_source(SOURCE.read_text(encoding='utf-8-sig'))
        self.assertIn('and not conservative_stop and fall_reason is None and external_safety_reason is None',p)
        self.assertIn('or external_safety_reason is not None\n            or fall_reason is not None',p)
        self.assertIn('_G_SCENE.max_neighbor_stretch(current_nodes, node_rest)',p)
        self.assertIn('active_history, baseline=baseline_history, thresholds=thresholds, sample_period_s=DT',p)
        self.assertLess(p.index('deployment_samples.setdefault(phase, []).append(deployment_sample)'),p.index('            if telemetry_sample:\n                observation = decode_last_observation'))
        self.assertIn('retain_accelerations=True',p)
        self.assertLess(p.index('_G_CONFIGURE_SCENE_QUERY(_g_sim_cfg)'),p.index('sim = SimulationContext(_g_sim_cfg)'))
        self.assertLess(p.index('_G_RECORD_SCENE_QUERY(sim, _g_sim_cfg)'),p.index('        ground = sim_utils.CuboidCfg('))
        self.assertLess(p.index('_G_RANDOMIZATION.apply_nodes('),p.index('LOGGER.mark("FIRST_PHYSICS_STEP")'))
        self.assertIn('if _G_AUDIT.should_sample(sim_step,local_step):\n                if not telemetry_sample:',p)
        self.assertIn('_g_capture_boundary("responsive_traverse_stand")',p)
        self.assertLess(p.index('_g_capture_boundary("physical_probe_hold")'),p.index('_G_ACTION_GUARD.begin(_G_AUDIT.last_sample)'))
        self.assertLess(p.index('_g_finished_result = _G_RECORDER.finish('),p.index('_G_POST_FINALIZE_CLEANUP(sim)'))
        self.assertLess(p.index('_G_POST_FINALIZE_CLEANUP(sim)'),p.index('        return _g_finished_result'))
        self.assertNotIn('zero_runtime_velocity(curtain_deformable, curtain_positions.shape[0])',p)
        with self.assertRaises(RuntimeError):patch_source('unexpected version')

    def test_scene_query_facility_only_for_v5_wall_without_physics_changes(self):
        from g1_action_contract_v2 import PRIMARY_PROTOCOL
        specs=scene_specs();plan=dict(primary_protocol=PRIMARY_PROTOCOL)
        self.assertTrue(scene_query_required(plan,specs['D01']))
        self.assertTrue(scene_query_required(plan,specs['D02']))
        self.assertFalse(scene_query_required(plan,specs['C01']))
        self.assertFalse(scene_query_required(plan,dict(geometry=dict(kind='empty'),wall=None)))
        self.assertFalse(scene_query_required(dict(primary_protocol='legacy'),specs['D01']))
        class Config:
            def __init__(self):self.enable_scene_query_support=False
            def to_dict(self):return dict(dt=.005,render_interval=4,device='cuda:0',enable_scene_query_support=self.enable_scene_query_support,physx=dict(solver_type=1,contact_offset=.005))
        for required in [False,True]:
            cfg=Config();record=configure_scene_query(cfg,required)
            self.assertEqual(record['changed_configuration_fields'],['enable_scene_query_support'] if required else [])
            before=record['configuration_before'];after=record['configuration_after']
            self.assertEqual(before['physx'],after['physx']);self.assertEqual(before['dt'],after['dt'])
            with tempfile.TemporaryDirectory(dir=OUTPUT/'contract_tests') as folder:
                sim=SimpleNamespace(get_physics_context=lambda:SimpleNamespace(get_enable_scene_query_support=lambda:required))
                proof=dict(runtime_source_archive={'g1_trial.py':dict(sha256='SYNTHETIC_TEST_ONLY')})
                evidence=record_scene_query_configuration(sim,cfg,record,proof,folder)
                self.assertEqual(evidence['actual_native_readback_enabled'],required)
                path=Path(folder)/'g_scene_query_configuration.json'
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),proof['scene_query_configuration_evidence']['sha256'])
                with self.assertRaises(FileExistsError):record_scene_query_configuration(sim,cfg,record,proof,folder)
                bad=SimpleNamespace(get_physics_context=lambda:SimpleNamespace(get_enable_scene_query_support=lambda:not required))
                with self.assertRaisesRegex(RuntimeError,'differs from the requested'):record_scene_query_configuration(bad,cfg,record,proof,folder)

    def test_label_causality(self):
        action=action_spec({'probe_side':'left'});wall=dict(wall=dict(fixed=True,center_xyz=[.92,0,1],size_xyz=[.04,1.2,2]),candidate_plane_x=.85,initial_gap_m=.05)
        args=dict(physics_valid=True,runtime_valid=True,probe_valid=True,recovered=True,clearance=False,advance_steps=0,metadata=wall,action=action)
        self.assertTrue(local_forward_corridor_obstructed(wall,action))
        self.assertEqual(assign_label(**args)[0],'LABEL_UNKNOWN')
        self.assertEqual(assign_label(**{**args,'metadata':{}})[0],'LABEL_UNKNOWN')
        self.assertEqual(assign_label(**{**args,'physics_valid':False})[0],'LABEL_UNKNOWN')
        self.assertEqual(assign_label(**{**args,'clearance':True,'advance_steps':30})[0],'LABEL_UNKNOWN')
        self.assertEqual(assign_label(**{**args,'clearance':True,'advance_steps':30,'metadata':{}})[0],'PASSABLE_FOR_ACTION')
        partial={**wall,'wall':{**wall['wall'],'size_xyz':[.04,.45,2]}}
        self.assertEqual(assign_label(**{**args,'metadata':partial})[0],'LABEL_UNKNOWN')
        self.assertEqual(assign_label(**{**args,'clearance':True,'advance_steps':30,'metadata':{**wall,'out_of_probe_range':True}})[0],'LABEL_UNKNOWN')
        self.assertEqual(assign_label(**{**args,'recovered':False})[0],'LABEL_UNKNOWN')
        self.assertTrue(online_physics_invalid([{'evaluation_curtain_anchor_drift_m':.03}],None))
        self.assertIsNone(online_physics_invalid([], 'UPPER_TRACKING_RMS_LIMIT'))

    def test_no_protocol_escape(self):
        spec=scene_specs()['A01'];plan=dict(seed=1,probe_side='left',robot={})
        with self.assertRaises(ValueError):frozen_args(plan,spec,Path('unused'),['--probe-validation-only'])
        with self.assertRaises(ValueError):action_spec({**plan,'followup_hand':'right'})
        argv,_=frozen_args(plan,spec,Path('unused'),['--headless'])
        self.assertIn('--headless',argv);self.assertEqual(argv[argv.index('--max-bounded-advance-segments')+1],'24')

    def test_real_control_action_alignment_and_cutoff(self):
        test_root=Path(__file__).resolve().parents[2]/'outputs/round4070_comprehensive_traversability/contract_tests'
        test_root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_root) as folder:
            assert Path(folder).resolve().is_relative_to(test_root.resolve())
            r=Recorder(folder,{'trial_id':'unit_test_only','probe_side':'right'},scene_specs()['A01'],'test')
            adapter=SimpleNamespace(previous_action=np.zeros(29));after=SimpleNamespace(effort_source='applied_torque',base_ang_vel_b=np.zeros(3),projected_gravity_b=np.array([0,0,-1]),base_lin_vel_b=np.zeros(3),base_pos_w=np.zeros(3))
            robot=SimpleNamespace(body_names=[])
            for step in (0,4,8):
                r.before_control(adapter,step);adapter.previous_action=np.full(29,step+.5)
                r.observe(phase='physical_probe',local_step=step,sim_step=step,q=np.zeros(29),qdot=np.zeros(29),q_cmd=np.zeros(29),effort=np.zeros(29),after=after,
                    hands={'left':np.array([.2,0,1]),'right':np.array([.2+step*.001,0,1])},velocities={'left':np.zeros(3),'right':np.array([.2,0,0])},rpy=(0,0,0),
                    adapter=adapter,robot=robot,upper=np.zeros(14),preprobe_upper=np.zeros(14),target_upper=np.ones(14),command=np.zeros(3),walk_mode=False)
            rows=copy.deepcopy(r.rows);r.close_incomplete()
            seq,t,phase,raw,audit=legal_sequence(rows,r.action,.025)
            self.assertEqual(seq.shape,(2,len(schema()[0])));self.assertEqual(audit['true_previous_action_adjacent_updates_verified'],1)
            self.assertAlmostEqual(t[-1],.025);self.assertEqual(audit['imu_acceleration_missing_rows'],2)
            self.assertTrue(np.all(raw['previous_control_action'][1]==.5))
            rows[1]['previous_control_action']=[99.]*29
            with self.assertRaisesRegex(ValueError,'PREVIOUS_ACTION_TEMPORAL_MISMATCH'):legal_sequence(rows,r.action,.025)

    def test_split_roles_independent_of_labels(self):
        episodes=[]
        for i,spec in enumerate(scene_specs().values()):
            for rep in range(3):
                episodes.append(dict(trial_id=f't{i}_{rep}',scene_config_evaluation_only=spec,robot_conditions_audit_only={},side='left' if rep%2 else 'right',
                    group_id=spec['physical_config_id'],seed=rep,repetition_index=rep,is_control=False,y=rep%2))
        before=make_splits(episodes)
        after=make_splits([{**e,'y':-1} for e in episodes])
        for a,b in zip(before['splits'],after['splits']):
            for role in ('train','calibration','test'):self.assertEqual(a[role],b[role])

    def test_confirmation_never_enters_any_fit_or_main_test(self):
        specs=scene_specs();episodes=[]
        for sid in ('A01','B05','D04'):
            spec=specs[sid]
            for side in ('left','right'):
                for role,rep in [('train',0),('calibration',1),('test',2),('confirmation',4)]:
                    episodes.append(dict(trial_id=f'{sid}_{side}_{role}',scene_config_evaluation_only=spec,robot_conditions_audit_only={},
                        side=side,group_id=spec['physical_config_id'],seed=rep,repetition_index=rep,experiment_role=role,is_control=False,y=rep%2))
        confirmations={e['trial_id'] for e in episodes if e['experiment_role']=='confirmation'}
        splits=make_splits(episodes)['splits']
        for split in splits:
            for role in ('train','calibration','test'):self.assertFalse(confirmations&set(split[role]))
            self.assertTrue(set(split['confirmation'])<=confirmations)
        self.assertEqual(set(splits[0]['confirmation']),confirmations)
        anchor=next(s for s in splits if s['split_type']=='anchor')
        self.assertEqual(set(anchor['confirmation']),{'B05_left_confirmation','B05_right_confirmation'})

    def test_frozen_rng_replay_and_anchor_exclusion(self):
        path=next((ROOT/'outputs/round4070_comprehensive_traversability/plans_v4').glob('G_EMPTY_left*'))
        plan=json.loads(path.read_text(encoding='utf-8-sig'))
        test_root=ROOT/'outputs/round4070_comprehensive_traversability/contract_tests'
        test_root.mkdir(parents=True,exist_ok=True)
        class Deformable:
            def set_nodal_velocities(self,value):self.value=np.asarray(value).copy()
            def get_nodal_velocities(self):return self.value
        arrays=[]
        for side in ('left','right'):
            with tempfile.TemporaryDirectory(dir=test_root) as folder:
                r=ResetRandomization({**plan,'probe_side':side,'followup_hand':side},folder);d=Deformable()
                evidence=r.apply_nodes(d,8,[0,3,7],simulation_time_s=0.)
                arrays.append(d.value.copy());self.assertTrue(np.all(d.value[0,[0,3,7]]==0))
                self.assertTrue(np.any(d.value));self.assertLessEqual(float(np.abs(d.value).max()),.001)
                self.assertTrue(evidence['readback_verified'])
                with self.assertRaises(RuntimeError):r.apply_nodes(d,8,[0,3,7],simulation_time_s=0.)
        self.assertTrue(np.array_equal(*arrays))
        bad=copy.deepcopy(plan);bad['g1_reset_randomization']['frozen_robot_delta_y_m']+=.0001
        with self.assertRaisesRegex(ValueError,'RNG draws mismatch'):ResetRandomization(bad,test_root)

    def test_missing_body_collision_audit_is_not_admitted(self):
        test_root=ROOT/'outputs/round4070_comprehensive_traversability/contract_tests'
        with tempfile.TemporaryDirectory(dir=test_root) as folder:
            self.assertEqual(collision_admission(Path(folder))['status'],'NOT_VERIFIED')

    def test_finite_wall_static_fact_cannot_label_v1_closed_loop_blocked(self):
        self.assertEqual(conservative_action_label('BLOCKED','INDEPENDENT_FIXED_WALL_COVERS_FORWARD_ACTION_CORRIDOR'),
            ('LABEL_UNKNOWN','FINITE_WALL_NO_CLOSED_LOOP_ACTION_IMPOSSIBILITY_PROOF'))
        self.assertEqual(conservative_action_label('PASSABLE_FOR_ACTION','actual evidence'),('PASSABLE_FOR_ACTION','actual evidence'))

    def test_new_online_labels_are_unknown_old_raw_semantics_remain_auditable(self):
        wall=dict(wall=dict(fixed=True,center_xyz=[.92,0,1],size_xyz=[.04,1.2,2]),candidate_plane_x=.85,initial_gap_m=.05)
        source=dict(physics_valid=True,runtime_valid=True,probe_valid=True,safe_recovery=True,
            whole_body_candidate_clearance=False,followup_advance_logged_steps=0,scene_evaluation_only=wall)
        action=action_spec({'probe_side':'right'})
        self.assertEqual(original_online_label(source,action)[0],'BLOCKED')
        self.assertEqual(original_online_label({**source,'online_label_revision':ONLINE_LABEL_REVISION},action)[0],'LABEL_UNKNOWN')

    def test_actual_collider_clearance_uses_candidate_back_boundary(self):
        scene=scene_specs()['C01'];action=action_spec({'probe_side':'left'})
        self.assertFalse(actual_collider_clearance({'status':'VERIFIED'},scene,action))
        evidence=dict(status='VERIFIED',actual_collision_geometry=True,phase_final_sample=True,
            threshold_x_m=.95,min_world_x_m=.951,time_s=30.,phase='responsive_traverse_stand',comparison='>=',sample_index=1500)
        self.assertTrue(actual_collider_clearance({'candidate_clearance':evidence},scene,action))
        self.assertFalse(actual_collider_clearance({'candidate_clearance':{**evidence,'min_world_x_m':.60}},scene,action))
        with self.assertRaisesRegex(ValueError,'THRESHOLD_MISMATCH'):
            actual_collider_clearance({'candidate_clearance':{**evidence,'threshold_x_m':.89}},scene,action)

    def test_body_snapshot_real_clock_and_final_boundary(self):
        test_root=ROOT/'outputs/round4070_comprehensive_traversability/contract_tests'
        with tempfile.TemporaryDirectory(dir=test_root) as folder:
            recorder=BodyCollisionRecorder(folder);recorder.names=['pelvis']
            pose=np.array([[[.1,.2,.8,1.,0.,0.,0.]]]);robot=SimpleNamespace(data=SimpleNamespace(body_link_pose_w=pose))
            args=dict(robot=robot,poses={'pelvis':(pose[0,0,:3],pose[0,0,3:])},nodes=np.zeros((2,3)),
                node_velocities=np.zeros((2,3)),phase='responsive_traverse_stand',physics_step=4,time_s=.025,actual_simulation_time_s=.030)
            @recorder.track_phase
            def run_phase(phase):
                recorder.note_executed_step(phase,4,.025);recorder.observe(**args)
            run_phase('responsive_traverse_stand');recorder.observe(**args,boundary=True);recorder.finish(True)
            with np.load(Path(folder)/'g_body_collision_history_audit_only.npz') as data:
                self.assertEqual(len(data['time_s']),1);self.assertTrue(data['boundary'][0])
                self.assertAlmostEqual(data['actual_simulation_time_s'][0]-data['time_s'][0],.005)
                self.assertEqual(data['nodes_world'].dtype,np.float32)

    def test_zero_step_phase_never_relabels_previous_sample(self):
        test_root=ROOT/'outputs/round4070_comprehensive_traversability/contract_tests'
        with tempfile.TemporaryDirectory(dir=test_root) as folder:
            recorder=BodyCollisionRecorder(folder);recorder.names=['pelvis']
            pose=np.array([[[.1,.2,.8,1.,0.,0.,0.]]]);robot=SimpleNamespace(data=SimpleNamespace(body_link_pose_w=pose))
            args=dict(robot=robot,poses={'pelvis':(pose[0,0,:3],pose[0,0,3:])},nodes=np.zeros((2,3)),
                node_velocities=np.zeros((2,3)),physics_step=0,time_s=.005,actual_simulation_time_s=.010)
            @recorder.track_phase
            def run_phase(phase,steps):
                if steps:recorder.note_executed_step(phase,0,.005);recorder.observe(**args,phase=phase)
            run_phase('physical_probe',1);run_phase('physical_post_retract_stand',0)
            self.assertFalse(recorder.observe(**args,phase='physical_post_retract_stand',boundary=True))
            self.assertEqual(recorder.samples[-1]['phase'],'physical_probe')
            self.assertEqual(recorder.phase_execution_ledger[-1]['executed_step_count'],0)
            self.assertIsNone(recorder.phase_execution_ledger[-1]['final_snapshot_physics_step'])

    def test_200hz_chunk_boundaries_and_final_marker_are_real(self):
        test_root=ROOT/'outputs/round4070_comprehensive_traversability/contract_tests'
        with tempfile.TemporaryDirectory(dir=test_root) as folder:
            recorder=BodyCollisionRecorder(folder,nominal_hz=200);recorder.names=['pelvis'];recorder.node_count=2
            pose=np.array([[[.1,.2,.8,1.,0.,0.,0.]]]);robot=SimpleNamespace(data=SimpleNamespace(body_link_pose_w=pose))
            @recorder.track_phase
            def run_phase(phase):
                for step in range(501):
                    recorder.note_executed_step(phase,step,(step+1)*.005)
                    recorder.observe(robot=robot,poses={'pelvis':(pose[0,0,:3],pose[0,0,3:])},nodes=np.full((2,3),step*.001),
                        node_velocities=np.zeros((2,3)),physics_step=step,time_s=(step+1)*.005,actual_simulation_time_s=(step+2)*.005,phase=phase)
            run_phase('responsive_traverse_stand')
            recorder.observe(robot=robot,poses={'pelvis':(pose[0,0,:3],pose[0,0,3:])},nodes=np.full((2,3),.5),node_velocities=np.zeros((2,3)),
                physics_step=500,time_s=501*.005,actual_simulation_time_s=502*.005,phase='responsive_traverse_stand',boundary=True)
            recorder.finish(True);index=json.loads((Path(folder)/'g_body_collision_history_index.json').read_text())
            self.assertEqual([c['sample_count'] for c in index['chunks']],[500,1]);self.assertEqual(index['sample_count'],501)
            with np.load(Path(folder)/index['chunks'][1]['path']) as data:self.assertTrue(data['boundary'][0]);self.assertEqual(data['physics_step'][0],500)
            self.assertEqual(recorder.phase_execution_ledger[0]['final_snapshot_physics_step'],500)

    def test_v5_draft_cannot_launch_and_static_proof_strict_bounds(self):
        contract=draft_contract()
        with self.assertRaisesRegex(ValueError,'independently verified'):validate_contract(contract)
        wall=dict(fixed=True,center_xyz=[.95,0,1],size_xyz=[.04,1.2,2])
        certificate=separation_certificate(contract,wall,entry_x=.8,candidate_center_x=.85)
        self.assertTrue(certificate['mathematically_sufficient'])
        self.assertFalse(certificate['physical_evidence_verified'])
        self.assertAlmostEqual(certificate['completion_witness_x_lower_bound_m'],.956)
        # A later entry cannot claim the left-to-interior separation argument.
        self.assertFalse(separation_certificate(contract,wall,entry_x=.94,candidate_center_x=.85)['mathematically_sufficient'])
        thin={**wall,'size_xyz':[.029,1.2,2]}
        self.assertFalse(separation_certificate(contract,thin,entry_x=.8,candidate_center_x=.85)['mathematically_sufficient'])
        narrow={**wall,'size_xyz':[.04,.6,2]}
        self.assertFalse(separation_certificate(contract,narrow,entry_x=.8,candidate_center_x=.85)['mathematically_sufficient'])

    def test_v5_guard_covers_every_intermediate_step_and_allows_recovery(self):
        contract=draft_contract();contract['witness'].update(body_index=0,point_body_m=[0,0,0])
        guard=ActionTubeGuard(contract)
        def sample(step,phase,x=.4,y=0.):
            return dict(physics_step=step,time_s=step*.005,phase=phase,
                body_positions_world=np.array([[x,y,.8]]),body_quaternions_wxyz=np.array([[1.,0,0,0]]))
        self.assertIsNone(guard.begin(sample(10,'physical_probe_hold')))
        self.assertIsNone(guard.observe(sample(11,'unexpected_intermediate_phase',x=.41)))
        self.assertIsNone(guard.observe(sample(12,'round26_cautious_advance_00',x=.42)))
        self.assertEqual(guard.observe(sample(13,'responsive_traverse_stand',x=.45)),'ACTION_TUBE_STEP_X_VIOLATION')
        self.assertIsNone(guard.observe(sample(14,'physical_retract',y=2.)))
        self.assertIsNone(guard.observe(sample(15,'physical_neutralize',y=2.)))
        self.assertEqual(guard.evidence()['violation'],'ACTION_TUBE_STEP_X_VIOLATION')

    def test_actual_native_witness_proof_and_geometry_hash(self):
        contract=json.loads((OUTPUT/SIDECAR_NAME).read_text())
        if contract['status']!='VALIDATED_READY_TO_FREEZE':self.skipTest('Actual native witness remains pending')
        validate_contract(contract)
        changed=copy.deepcopy(contract);changed['witness']['point_body_m'][0]+=.001
        with self.assertRaisesRegex(ValueError,'differs from native proof'):validate_contract(changed)
        changed=copy.deepcopy(contract);changed['witness']['native_geometry_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'part hash mismatch'):validate_contract(changed)

    def test_actual_runtime_admission_plan_roundtrip_and_forbidden_changes(self):
        path=OUTPUT/'runtime_plans/G_EMPTY_left_empty_control_00_nominal_s74800_rng1_act2_attempt01.json'
        if not path.exists():self.skipTest('Actual V5 first launch has not occurred')
        plan=json.loads(path.read_text());manifest=json.loads((OUTPUT/'campaign_manifest_v5.json').read_text())
        frozen=next(p for p in manifest['trial_plans'] if p['trial_id']==plan['trial_id'])
        self.assertIs(frozen['physics_valid'],False);self.assertIs(plan['physics_valid'],True)
        gates=verify_gate_evidence(plan,manifest['scene_specs'][plan['scene_id']])
        self.assertGreaterEqual(len({g['seed'] for g in gates}),3)
        self.assertEqual(verify_empty_evidence(plan)['reason'],'EMPTY_SELF_VALIDATION')
        assert_runtime_plan_matches(frozen,plan,verified_gate_seeds=[g['seed'] for g in gates])
        with self.assertRaisesRegex(RuntimeError,'three independently verified'):
            assert_runtime_plan_matches(frozen,plan,verified_gate_seeds=[])
        no_evidence=copy.deepcopy(plan);no_evidence['physics_gate_evidence']=[]
        with self.assertRaisesRegex(RuntimeError,'three actual distinct-seed'):verify_empty_evidence(no_evidence)
        changed=copy.deepcopy(plan);changed['robot']['reset_root_y']+=.001
        with self.assertRaisesRegex(RuntimeError,'differs from frozen campaign: robot'):verify_empty_evidence(changed)
        changed=copy.deepcopy(plan);changed['seed']+=1
        with self.assertRaisesRegex(RuntimeError,'differs from frozen campaign: seed'):verify_empty_evidence(changed)
        changed=copy.deepcopy(plan);changed['followup_action_id']='G_HOLD_PROBE_POSE_FORWARD24_V1'
        with self.assertRaisesRegex(RuntimeError,'exact frozen G action'):verify_empty_evidence(changed)

    def test_pairs_use_actual_action_version_and_independent_label_evidence(self):
        from g1_action_contract_v2 import PRIMARY_PROTOCOL,ACTION_ID,LABEL_CONTRACT
        specs=scene_specs();action=action_spec(dict(primary_protocol=PRIMARY_PROTOCOL,probe_side='left'))
        episodes=[]
        for identity,sid,y in [('new_p','A01',1),('new_b','B02',0)]:
            episodes.append(dict(trial_id=identity,y=y,label='PASSABLE_FOR_ACTION' if y else 'BLOCKED',is_control=False,
                group_id=specs[sid]['physical_config_id'],followup_action=action,action_contract=ACTION_ID,followup_action_id=ACTION_ID,
                label_contract=LABEL_CONTRACT,side='left',scene_config_evaluation_only=specs[sid],source_path='SYNTHETIC_TEST_ONLY',
                body_collision_audit=dict(source_evidence_path='synthetic/'+identity+'.json',source_evidence_sha256='f'*64)))
        legacy=copy.deepcopy(episodes[1]);legacy['trial_id']='legacy_b';legacy['followup_action']=action_spec(dict(probe_side='left'))
        legacy['action_contract']=legacy['followup_action_id']=legacy['followup_action']['action_id'];legacy['label_contract']=legacy['followup_action']['label_contract']
        episodes.append(legacy);splits=dict(splits=[dict(split_id='synthetic',train=[e['trial_id'] for e in episodes])])
        pairs=matched_pairs(episodes,splits)['by_split']['synthetic']
        self.assertEqual(len(pairs),1);self.assertEqual(pairs[0]['evidence']['action_contract'],ACTION_ID)
        self.assertEqual(pairs[0]['evidence']['label_contract'],LABEL_CONTRACT)
        self.assertNotIn('legacy_b',[pairs[0]['a'],pairs[0]['b']])
        self.assertEqual(pairs[0]['evidence']['source_label_evidence'],['synthetic/new_p.json','synthetic/new_b.json'])
        bad=copy.deepcopy(episodes);bad[0]['action_contract']=legacy['action_contract']
        with self.assertRaisesRegex(ValueError,'ACTION_METADATA_MISMATCH'):matched_pairs(bad,splits)

    def test_actual_completed_episode_campaign_field_and_frozen_evidence(self):
        run=OUTPUT/'trials/G_EMPTY_left_empty_control_00_nominal_s74800_rng1_act2/attempt02'
        if not (run/'g_outcome.json').exists():self.skipTest('Actual V5 completed smoke not yet available')
        meta=json.loads((run/'g_trial_metadata.json').read_text());manifest=json.loads((OUTPUT/'campaign_manifest_v5.json').read_text())
        self.assertEqual(manifest['g1_primary_protocol'],meta['plan']['primary_protocol'])
        result=frozen_evidence(run,meta,meta['plan'])
        self.assertEqual(result['physics_gate_seeds'],[40701,40702,40703])

    def test_cleanup_flag_only_after_all_recorders_and_artifacts_finalize(self):
        test_root=ROOT/'outputs/round4070_comprehensive_traversability/contract_tests'
        with tempfile.TemporaryDirectory(dir=test_root) as folder:
            folder=Path(folder);sim=SimpleNamespace(_disable_app_control_on_stop_handle=False)
            recorder=SimpleNamespace(completed=False,stream=SimpleNamespace(closed=False));body=SimpleNamespace(saved=False)
            with self.assertRaisesRegex(RuntimeError,'cannot precede'):post_finalize_cleanup(sim,recorder,body,folder)
            self.assertFalse(sim._disable_app_control_on_stop_handle)
            recorder.completed=True;recorder.stream.closed=True;body.saved=True
            with self.assertRaisesRegex(RuntimeError,'missing completed artifact'):post_finalize_cleanup(sim,recorder,body,folder)
            for name,value in [('summary.json',{}),('g_outcome.json',dict(status='G_FIXED_ACTION_FINISHED')),
                ('g_body_collision_capture.json',dict(episode_finalized=True,execution_anchors=dict(last_physics_step=503))),('g_action_tube_runtime.json',{})]:
                (folder/name).write_text(json.dumps(value))
            post_finalize_cleanup(sim,recorder,body,folder)
            self.assertTrue(sim._disable_app_control_on_stop_handle)
            receipt=json.loads((folder/'post_finalize_cleanup.json').read_text())
            self.assertTrue(receipt['recorders_closed']);self.assertEqual(receipt['final_physics_step'],503)

    def test_v5_absolute_initial_and_final_execution_anchors(self):
        test_root=ROOT/'outputs/round4070_comprehensive_traversability/contract_tests'
        with tempfile.TemporaryDirectory(dir=test_root) as folder:
            recorder=BodyCollisionRecorder(folder,nominal_hz=200);recorder.names=['pelvis'];recorder.node_count=0
            pose=np.array([[[.1,.2,.8,1.,0.,0.,0.]]]);robot=SimpleNamespace(data=SimpleNamespace(body_link_pose_w=pose))
            recorder.capture_initial_state(robot=robot,nodes=np.zeros((0,3)),node_velocities=np.zeros((0,3)),actual_simulation_time_s=7.)
            @recorder.track_phase
            def run_phase(phase,step):
                recorder.note_executed_step(phase,step,step*.005)
                recorder.observe(robot=robot,poses={'pelvis':(pose[0,0,:3],pose[0,0,3:])},nodes=np.zeros((0,3)),node_velocities=np.zeros((0,3)),
                    physics_step=step,time_s=step*.005,actual_simulation_time_s=7.+step*.005,phase=phase)
            run_phase('initialization_physics_step',1);run_phase('settle',2)
            with self.assertRaisesRegex(RuntimeError,'counter differs'):recorder.finish(True,expected_total_steps=1)
            with self.assertRaisesRegex(RuntimeError,'Final simulator clock'):recorder.finish(True,expected_total_steps=2,final_actual_simulation_time_s=7.02)
            recorder.finish(True,expected_total_steps=2,final_actual_simulation_time_s=7.01)
            capture=json.loads((Path(folder)/'g_body_collision_capture.json').read_text())
            self.assertEqual(capture['execution_anchors']['first_physics_step'],1)
            self.assertEqual(capture['execution_anchors']['independently_expected_total_steps'],2)
            self.assertEqual(capture['initial_state']['actual_simulation_time_s'],7.)

    def test_actual_recorder_output_matches_streaming_history_contract(self):
        from physics_history import verify_history
        test_root=ROOT/'outputs/round4070_comprehensive_traversability/contract_tests'
        with tempfile.TemporaryDirectory(dir=test_root) as folder:
            folder=Path(folder);recorder=BodyCollisionRecorder(folder,nominal_hz=200)
            recorder.names=['pelvis'];recorder.node_count=0
            query_path=folder/'g_scene_query_configuration.json';query_path.write_text('{"synthetic_test_only":true}')
            geom=dict(synthetic_fixture=True,body_names=['pelvis'],sheets=[])
            path=folder/'g_robot_collision_geometry.json';path.write_text(json.dumps(geom))
            recorder.geometry=dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            pose=np.array([[[.1,.2,.8,1.,0.,0.,0.]]]);robot=SimpleNamespace(data=SimpleNamespace(body_link_pose_w=pose))
            common=dict(robot=robot,poses={'pelvis':(pose[0,0,:3],pose[0,0,3:])},nodes=np.zeros((0,3)),node_velocities=np.zeros((0,3)))
            recorder.capture_initial_state(robot=robot,nodes=common['nodes'],node_velocities=common['node_velocities'],actual_simulation_time_s=7.)
            @recorder.track_phase
            def run_phase(phase,start,end):
                for step in range(start,end+1):
                    recorder.note_executed_step(phase,step,step*.005)
                    recorder.observe(**common,physics_step=step,time_s=step*.005,actual_simulation_time_s=7.+step*.005,phase=phase)
            for phase,start,end in [('initialization_physics_step',1,1),('physical_probe',2,101),('physical_probe_hold',102,201),
                                    ('round26_cautious_advance_00',202,301),('responsive_traverse_stand',302,401),
                                    ('responsive_release',402,451),('responsive_neutralize',452,477),('responsive_release_stand',478,503)]:
                run_phase(phase,start,end)
                if phase in ('physical_probe_hold','responsive_traverse_stand','responsive_release_stand'):
                    recorder.observe(**common,physics_step=end,time_s=end*.005,actual_simulation_time_s=7.+end*.005,phase=phase,boundary=True)
            recorder.finish(True,expected_total_steps=503,final_actual_simulation_time_s=7.+503*.005)
            capture=json.loads((folder/'g_body_collision_capture.json').read_text())
            self.assertEqual(capture['scene_query_configuration'],dict(path=query_path.name,sha256=hashlib.sha256(query_path.read_bytes()).hexdigest()))
            result=verify_history(folder,capture,geom)
            self.assertEqual(result['status'],'VERIFIED_HISTORY');self.assertEqual(result['sample_count'],503)


if __name__=='__main__':unittest.main()
