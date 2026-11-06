"""Pure CPU checks for the high-impact G logging/label/split boundaries."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np

from g1_recording import Recorder, action_spec, assign_label, online_physics_invalid
from g1_trial import patch_source, frozen_args, SOURCE
from build_dataset import legal_sequence, make_splits, schema
from physics_scenes import scene_specs


class Contracts(unittest.TestCase):
    def test_checked_source_and_safety(self):
        p=patch_source(SOURCE.read_text(encoding='utf-8-sig'))
        self.assertIn('and not conservative_stop and fall_reason is None and external_safety_reason is None',p)
        self.assertIn('or external_safety_reason is not None\n            or fall_reason is not None',p)
        self.assertIn('_G_SCENE.max_neighbor_stretch(current_nodes, node_rest)',p)
        self.assertIn('active_history, baseline=baseline_history, thresholds=thresholds, sample_period_s=DT',p)
        self.assertLess(p.index('deployment_samples.setdefault(phase, []).append(deployment_sample)'),p.index('            if telemetry_sample:\n                observation = decode_last_observation'))
        self.assertIn('retain_accelerations=True',p)
        with self.assertRaises(RuntimeError):patch_source('unexpected version')

    def test_label_causality(self):
        action=action_spec({'probe_side':'left'});wall=dict(wall=dict(fixed=True,center_xyz=[.92,0,1],size_xyz=[.04,1.2,2]),candidate_plane_x=.85,initial_gap_m=.05)
        args=dict(physics_valid=True,runtime_valid=True,probe_valid=True,recovered=True,clearance=False,advance_steps=0,metadata=wall,action=action)
        self.assertEqual(assign_label(**args)[0],'BLOCKED')
        self.assertEqual(assign_label(**{**args,'metadata':{}})[0],'LABEL_UNKNOWN')
        self.assertEqual(assign_label(**{**args,'physics_valid':False})[0],'LABEL_UNKNOWN')
        self.assertEqual(assign_label(**{**args,'clearance':True,'advance_steps':30})[0],'PASSABLE_FOR_ACTION')
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


if __name__=='__main__':unittest.main()
