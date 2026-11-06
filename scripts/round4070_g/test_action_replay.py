"""CPU adversarial checks of the independent discrete action replay.

Poses below are explicitly synthetic mathematical fixtures; they are never
exported as trials or admitted to a learning manifest.
"""
import copy
import json
import unittest
import numpy as np
from g1_action_contract_v2 import ActionTubeGuard,OUTPUT,SIDECAR_NAME,ROOT
from physics_g1_collision_audit import ActionReplay
from collision_geometry import GeometryNotVerified


class ActionReplayContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract=json.loads((OUTPUT/SIDECAR_NAME).read_text())
        cls.library=json.loads((ROOT/cls.contract['witness']['native_library_path']).read_text())

    def sample(self,step,phase,x=.4,y=0.,z=.8,boundary=False):
        body=self.contract['witness']['body_index'];point=np.asarray(self.contract['witness']['point_body_m'])
        positions=np.zeros((len(self.library['body_names']),3));positions[body]=np.array([x,y,z])-point
        quats=np.zeros((len(positions),4));quats[:,0]=1.
        return dict(physics_step=step,time_s=step*.005,phase=phase,boundary=boundary,
                    body_positions_world=positions,body_quaternions_wxyz=quats)

    def rows(self,*,violation=None):
        middle=self.sample(11,'an_intermediate_actual_phase',x=.41)
        if violation=='tube':middle=self.sample(11,'an_intermediate_actual_phase',x=.41,y=.351)
        if violation=='step':middle=self.sample(11,'an_intermediate_actual_phase',x=.431)
        return [self.sample(10,'physical_probe_hold',boundary=True),middle,
                self.sample(12,'responsive_traverse_stand',x=.42,boundary=True),
                self.sample(13,'physical_retract',x=.43,y=2.),
                self.sample(14,'physical_neutralize',x=.44,y=2.),
                self.sample(15,'physical_post_retract_stand',x=.45,y=2.,boundary=True)]

    def runtime(self,rows):
        guard=ActionTubeGuard(self.contract);guard.begin(rows[0])
        for row in rows[1:]:guard.observe(row)
        return guard.evidence()

    def replay(self,rows,runtime=None):
        replay=ActionReplay(self.contract,self.library,runtime or self.runtime(rows))
        for row in rows:replay.observe(row)
        return replay

    def finish(self,replay,wall=None,scope=None,physics_status='VERIFIED'):
        return replay.finish(wall,.85,dict(runtime_valid=True,probe_valid=True,safe_recovery=True),physics_status,scope)

    def test_all_intermediate_steps_are_checked_and_recovery_exempt(self):
        result=self.finish(self.replay(self.rows()))
        self.assertEqual(result['status'],'VERIFIED');self.assertTrue(result['all_required_steps_valid'])
        self.assertEqual(result['guarded_sample_count'],3)
        self.assertAlmostEqual(result['maximum_actual_step_x_m'],.01)
        self.assertEqual(result['static_blocking_certificate']['status'],'NOT_APPLICABLE')

    def test_tube_or_jump_violation_is_audited_but_not_valid_or_blocked(self):
        wall=dict(fixed=True,center_xyz=[.95,0,1],size_xyz=[.04,1.2,2])
        for kind,reason in [('tube','ACTION_TUBE_BOUNDS_VIOLATION'),('step','ACTION_TUBE_STEP_X_VIOLATION')]:
            with self.subTest(kind=kind):
                result=self.finish(self.replay(self.rows(violation=kind)),wall)
                self.assertEqual(result['status'],'VERIFIED');self.assertFalse(result['all_required_steps_valid'])
                self.assertEqual(result['guard_violation'],reason);self.assertEqual(result['guard_violation_physics_step'],11)
                self.assertNotEqual(result['static_blocking_certificate']['status'],'VERIFIED')

    def test_last_real_probe_hold_boundary_and_clock_required(self):
        rows=self.rows();runtime=self.runtime(rows)
        rows[0]['boundary']=False
        with self.assertRaisesRegex(GeometryNotVerified,'FINAL_PROBE_HOLD'):self.replay(rows,runtime)
        rows=self.rows();runtime=self.runtime(rows);runtime['entry']['time_s']+=.005
        with self.assertRaisesRegex(GeometryNotVerified,'CLOCK_DISAGREEMENT'):self.replay(rows,runtime)
        rows=self.rows();runtime=self.runtime(rows);runtime['entry']['witness_world_m'][0]+=.001
        with self.assertRaisesRegex(GeometryNotVerified,'POSE_OR_CLOCK'):self.replay(rows,runtime)

    def test_online_reason_and_first_violation_trace_must_match(self):
        rows=self.rows(violation='tube');runtime=self.runtime(rows)
        runtime['violation']=None
        with self.assertRaisesRegex(GeometryNotVerified,'DISAGREE'):self.finish(self.replay(rows,runtime))
        runtime=self.runtime(rows);runtime['violations'][0]['physics_step']+=1
        with self.assertRaises(GeometryNotVerified):self.finish(self.replay(rows,runtime))
        runtime=self.runtime(rows);runtime['violations'][0]['witness_world_m'][1]+=.001
        with self.assertRaises(GeometryNotVerified):self.finish(self.replay(rows,runtime))

    def test_no_forward_or_completion_phase_can_resume_after_release(self):
        rows=self.rows();runtime=self.runtime(rows)
        replay=self.replay(rows,runtime)
        with self.assertRaises(GeometryNotVerified):replay.observe(self.sample(16,'responsive_traverse_stand',x=1.2,boundary=True))

    def test_discrete_wall_certificate_and_scope_and_physics_gates(self):
        wall=dict(fixed=True,center_xyz=[.95,0,1],size_xyz=[.04,1.2,2])
        self.assertEqual(self.finish(self.replay(self.rows()),wall)['static_blocking_certificate']['status'],'VERIFIED')
        self.assertNotEqual(self.finish(self.replay(self.rows()),wall,physics_status='NOT_VERIFIED')['static_blocking_certificate']['status'],'VERIFIED')
        for scope in ('out_of_probe_range','DEFER_VISUAL_REOBSERVATION'):
            self.assertEqual(self.finish(self.replay(self.rows()),wall,scope)['static_blocking_certificate']['status'],'NOT_APPLICABLE')
        for changed in ({**wall,'fixed':False},{**wall,'size_xyz':[.02,1.2,2]},{**wall,'size_xyz':[.04,.6,2]}):
            self.assertNotEqual(self.finish(self.replay(self.rows()),changed)['static_blocking_certificate']['status'],'VERIFIED')

    def test_native_witness_cannot_be_conservative_or_remapped(self):
        native=copy.deepcopy(self.library);native['parts'][native['action_witness']['part_index']]['evidence_mode']='NATIVE_VERTEX_CONSERVATIVE_ENVELOPE'
        with self.assertRaisesRegex(GeometryNotVerified,'ENABLED_EXACT_SOLID'):ActionReplay(self.contract,native,self.runtime(self.rows()))
        native=copy.deepcopy(self.library);native['action_witness']['body_index']+=1
        with self.assertRaisesRegex(GeometryNotVerified,'MAPPING_MISMATCH'):ActionReplay(self.contract,native,self.runtime(self.rows()))


if __name__=='__main__':unittest.main()
