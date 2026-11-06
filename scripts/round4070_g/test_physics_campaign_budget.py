"""Collector integration only; no simulator/guard or real ledger mutation."""
from contextlib import ExitStack,nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import hashlib,json,tempfile,unittest
import physics_campaign as campaign
from budget_ledger import BudgetLedgerError


class CampaignBudgetTests(unittest.TestCase):
    def test_exact_plan_quarantine_precedes_reaudit_and_does_not_block_other_hand(self):
        for changed in [False,True]:
            with tempfile.TemporaryDirectory(dir=campaign.ROOT/'outputs') as temp,ExitStack() as stack:
                root=Path(temp);left='G_TEST_left';right='G_TEST_right'
                (root/'trials'/left/'attempt01').mkdir(parents=True)
                plans=[dict(trial_id=name,scene_id='D01',repetition_index=0,phase='coverage',physical_config_id='test') for name in [left,right]]
                manifest=dict(max_g1_launches=1000,scene_specs={'D01':{'physical_config_id':'test'}},representative_scene_ids=['D01'],trial_plans=plans)
                frozen=root/'campaign_manifest.json';frozen.write_text(json.dumps(manifest),encoding='utf-8')
                frozen.with_suffix('.sha256.json').write_text(json.dumps({'sha256':hashlib.sha256(frozen.read_bytes()).hexdigest()}),encoding='utf-8')
                for seed in [1,2,3]:
                    directory=root/'physics/D01'/f'seed{seed}_attempt01';directory.mkdir(parents=True)
                    (directory/'summary.json').write_text(json.dumps({'physics_valid':True,'seed':seed,'physical_config_id':'test'}),encoding='utf-8')
                stack.enter_context(patch.object(campaign,'OUT',root))
                stack.enter_context(patch.object(campaign,'resource_lock',return_value=nullcontext()))
                stack.enter_context(patch.object(campaign,'launch_ledger_summary',return_value={'count':16}))
                stack.enter_context(patch.object(campaign,'trial_budget_state',return_value={'withhold':False,'minimum_next_attempt_number':1}))
                stack.enter_context(patch.object(campaign,'empty_dependency',return_value={'ready':True}))
                stack.enter_context(patch.object(campaign,'summarize',return_value={'g1_launches':16}))
                events=stack.enter_context(patch.object(campaign,'event'))
                def quarantine(run,plan):
                    status=('GLOBAL_PAUSE_REQUIRED' if changed else 'VALIDATED_WITHHOLD') if plan['trial_id']==left else 'NOT_QUARANTINED'
                    return {'status':status,'reason':'test exact evidence','withhold':plan['trial_id']==left}
                stack.enter_context(patch.object(campaign,'quarantine_state',side_effect=quarantine))
                guard=stack.enter_context(patch.object(campaign,'launch',return_value=None))
                replay=stack.enter_context(patch.object(campaign.subprocess,'run'))
                stack.enter_context(patch('sys.argv',['physics_campaign.py','--manifest',str(frozen),'--mode','g1','--scene','D01','--max-jobs','1']))
                campaign.main()
                replay.assert_not_called()
                if changed:
                    guard.assert_not_called();self.assertTrue((root/'PAUSE_G1.flag').exists())
                    self.assertIn('G1_QUARANTINE_EVIDENCE_GLOBAL_PAUSE',[call.args[0] for call in events.call_args_list])
                else:
                    guard.assert_called_once();self.assertEqual(guard.call_args.args[-1],right)
                    self.assertIn('G1_EXACT_PLAN_QUARANTINE_WITHHELD',[call.args[0] for call in events.call_args_list])

    def test_known_missing_raw_and_existing_attempt_identity_are_withheld(self):
        summary=dict(count=16,budget_exhausted=False)
        missing=dict(withhold=True,reason='KNOWN_ATTEMPT_RAW_MISSING_RESTORE_OR_REVIEW_REQUIRED',minimum_next_attempt_number=2)
        with patch.object(campaign,'launch_ledger_summary',return_value=summary),patch.object(campaign,'trial_budget_state',return_value=missing):
            self.assertFalse(campaign.g1_budget_admission({},'trial',Path('attempt02'))['ready'])
            missing['withhold']=False
            self.assertEqual(campaign.g1_budget_admission({},'trial',Path('attempt01'))['reason'],'KNOWN_ATTEMPT_PATH_REUSE_FORBIDDEN')
            self.assertTrue(campaign.g1_budget_admission({},'trial',Path('attempt02'))['ready'])

    def test_guard_never_starts_before_durable_append_and_append_failure_aborts(self):
        for failure in [False,True]:
            with tempfile.TemporaryDirectory(dir=campaign.ROOT/'outputs') as temp,ExitStack() as stack:
                root=Path(temp);run=root/'attempt01';policy=root/'policy';policy.write_bytes(b'unchanged policy')
                calls=[]
                stack.enter_context(patch.object(campaign,'OUT',root))
                stack.enter_context(patch.object(campaign,'resource_lock',return_value=nullcontext()))
                stack.enter_context(patch.object(campaign,'training_active',return_value=False))
                stack.enter_context(patch.object(campaign,'resource_ready',return_value=(True,{})))
                stack.enter_context(patch.object(campaign,'g1_budget_admission',return_value={'ready':True}))
                stack.enter_context(patch.object(campaign,'FROZEN_FALCON',policy))
                stack.enter_context(patch.object(campaign,'FROZEN_FALCON_SHA256',hashlib.sha256(policy.read_bytes()).hexdigest()))
                stack.enter_context(patch.object(campaign,'archive_source_files',return_value=({},{})))
                stack.enter_context(patch.object(campaign,'event'))
                def append(path,**kwargs):
                    self.assertTrue(path.is_file());calls.append('append')
                    if failure:raise BudgetLedgerError('synthetic durable write failure')
                    return {'count':17}
                def guard(*args,**kwargs):
                    self.assertEqual(calls,['append']);calls.append('guard');return SimpleNamespace(returncode=0)
                stack.enter_context(patch.object(campaign,'append_actual_launch',side_effect=append))
                mocked=stack.enter_context(patch.object(campaign.subprocess,'run',side_effect=guard))
                if failure:
                    with self.assertRaises(BudgetLedgerError):campaign.launch(root/'script.py',run,[],'G1','trial',manifest={})
                    mocked.assert_not_called();self.assertEqual(calls,['append'])
                else:
                    self.assertEqual(campaign.launch(root/'script.py',run,[],'G1','trial',manifest={}),0)
                    self.assertEqual(calls,['append','guard'])


if __name__=='__main__':unittest.main(verbosity=2)
