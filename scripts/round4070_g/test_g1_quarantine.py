"""Scheduling quarantine contracts; no simulation, relabeling, or recapture."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import g1_quarantine as q

RUN=q.OUT/'trials/G_D01_left_coverage_00_nominal_s72700_rng1_act2/attempt01'


class QuarantineContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit=q.read(RUN/'g_collision_audit.json');cls.native=q.read(RUN/'g_native_geometry_verified.json')
        manifest=q.read(q.OUT/'campaign_manifest_v5.json')
        cls.plan=next(p for p in manifest['trial_plans'] if p['trial_id']==RUN.parent.name)
        cls.evidence=q.inspect_evidence(RUN,cls.plan)
        cls.test_root=q.OUT/'contract_tests';cls.test_root.mkdir(exist_ok=True)

    def record(self):
        return dict(schema=q.SCHEMA,status='ACTIVE_WITHHOLD_ONLY',scope='EXACT_PLAN',
            authorization=dict(authorized_by='/root',scope='EXACT_PLAN',trial_id=self.plan['trial_id'],instruction='SYNTHETIC_TEST_ONLY'),
            evidence=copy.deepcopy(self.evidence),automatic_retry_permitted=False,changes_labels=False,
            changes_campaign_or_budget=False,supervision_or_response_admission=False,formal_status_retained='NOT_VERIFIED')

    def write(self,folder,record):
        p=Path(folder)/(self.plan['trial_id']+'.json');p.write_text(json.dumps(record),encoding='utf-8')
        p.with_suffix('.sha256').write_text(q.sha(p),encoding='ascii');return p

    def test_actual_evidence_is_only_local_scheduling_uncertainty(self):
        evidence=self.evidence
        self.assertEqual(evidence['classification_evidence']['affected_parts'],[dict(part_index=28,body_name='left_rubber_hand',evidence_mode='NATIVE_VERTEX_CONSERVATIVE_ENVELOPE')])
        self.assertEqual(evidence['classification_evidence']['exact_5mm_penetration_count'],0)
        self.assertEqual(evidence['classification_evidence']['history_sample_count'],2633)
        self.assertTrue(evidence['rng_readback_verified'])
        self.assertEqual(evidence['immutable_audit_sha256'],'494e6ea74ea63418c136aa01cd65907d678cf8876f96ee8a3dafb7fb54667dea')

    def test_unknown_bug_rng_history_and_real_penetration_never_local(self):
        changes=[lambda a:a.__setitem__('reason','RuntimeError: unknown bug'),
            lambda a:a.__setitem__('status','PHYSICS_INVALID'),
            lambda a:a['strict_inner_triangle_intersections_by_tolerance_m'].__setitem__('0.005',1),
            lambda a:a['history_verification'].__setitem__('phase_ledger_verified',False),
            lambda a:a.__setitem__('initial_state_collision_status','NOT_VERIFIED'),
            lambda a:a['scene_query_configuration_audit'].__setitem__('actual_native_readback_enabled',False),
            lambda a:a['actual_wall_entity_audit'].__setitem__('status','NOT_VERIFIED'),
            lambda a:a['unverified_geometry_interactions'][0].__setitem__('reason','NUMERICAL_UNVERIFIED'),
            lambda a:a['unverified_geometry_interactions'][0].__setitem__('surface','cloth'),
            lambda a:a['action_contract_audit'].__setitem__('guard_violation','OUT_OF_TUBE')]
        for change in changes:
            audit=copy.deepcopy(self.audit);change(audit)
            with self.subTest(change=change),self.assertRaises(ValueError):q.classify_local_uncertainty(audit,self.native)
        native=copy.deepcopy(self.native);native['parts'][28]['exact_intersection_allowed']=True
        with self.assertRaises(ValueError):q.classify_local_uncertainty(self.audit,native)

    def test_exact_plan_only_and_no_record_does_not_grant_admission(self):
        with tempfile.TemporaryDirectory(dir=self.test_root) as folder:
            self.write(folder,self.record())
            with patch.object(q,'inspect_evidence',return_value=self.evidence):
                self.assertEqual(q.quarantine_state(RUN,self.plan,folder)['status'],'VALIDATED_WITHHOLD')
            right=copy.deepcopy(self.plan);right['trial_id']=right['trial_id'].replace('_left_','_right_')
            result=q.quarantine_state(RUN.parent.parent/right['trial_id']/'attempt01',right,folder)
            self.assertEqual(result['status'],'NOT_QUARANTINED');self.assertFalse(result['withhold'])

    def test_changed_record_source_audit_attempt_or_identity_requires_global_pause(self):
        with tempfile.TemporaryDirectory(dir=self.test_root) as folder:
            path=self.write(folder,self.record());path.write_text(path.read_text()+' ',encoding='utf-8')
            self.assertEqual(q.quarantine_state(RUN,self.plan,folder)['status'],'GLOBAL_PAUSE_REQUIRED')
            for field,value in [('run_path','other/attempt02'),('immutable_audit_sha256','f'*64),('existing_attempt_directories',['attempt01','attempt02']),('raw_source_sha256',{})]:
                self.write(folder,self.record());changed=copy.deepcopy(self.evidence);changed[field]=value
                with patch.object(q,'inspect_evidence',return_value=changed):
                    self.assertEqual(q.quarantine_state(RUN,self.plan,folder)['status'],'GLOBAL_PAUSE_REQUIRED')
            for reason in ['RNG_READBACK_MISMATCH','RAW_SOURCE_HASH_CHANGED','NATIVE_BINDING_REPLAY_DISAGREEMENT','UNKNOWN_IMPLEMENTATION_BUG']:
                with patch.object(q,'inspect_evidence',side_effect=ValueError(reason)):
                    result=q.quarantine_state(RUN,self.plan,folder);self.assertEqual(result['status'],'GLOBAL_PAUSE_REQUIRED');self.assertIn(reason,result['reason'])

    def test_authorization_or_scope_cannot_silently_expand(self):
        with tempfile.TemporaryDirectory(dir=self.test_root) as folder:
            for change in [lambda r:r.__setitem__('scope','ALL_WALL_SCENES'),lambda r:r.__setitem__('automatic_retry_permitted',True),
                lambda r:r.__setitem__('changes_labels',True),lambda r:r['authorization'].__setitem__('authorized_by','AUTO'),
                lambda r:r['authorization'].__setitem__('trial_id','ANOTHER_PLAN')]:
                record=self.record();change(record);self.write(folder,record)
                self.assertEqual(q.quarantine_state(RUN,self.plan,folder)['status'],'GLOBAL_PAUSE_REQUIRED')
            with self.assertRaisesRegex(ValueError,'AUTHORIZATION_REQUIRED'):
                q.create_authorized_quarantine(RUN,self.plan,{},folder)

    def test_actual_frozen_plan_parameter_change_is_rejected(self):
        changed=copy.deepcopy(self.plan);changed['seed']+=1
        with self.assertRaisesRegex(RuntimeError,'differs from frozen campaign: seed'):q.inspect_evidence(RUN,changed)


if __name__=='__main__':unittest.main()
