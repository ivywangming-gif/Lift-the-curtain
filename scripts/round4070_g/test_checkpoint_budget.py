"""A restored checkout must retain spent launches and isolate missing raw."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import checkpoint
from budget_ledger import sync_launch_ledger


class CheckpointBudget(unittest.TestCase):
    def test_postfreeze_missing_primary_is_counted_and_not_called_a_pilot(self):
        fixture_root = checkpoint.OUT / 'implementation_smoke'
        fixture_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='checkpoint_budget_', dir=fixture_root) as temporary:
            root = Path(temporary).resolve()
            self.assertTrue(root.is_relative_to(fixture_root.resolve()))
            out = root / 'outputs/round4070_comprehensive_traversability'
            doc = root / 'research/round4070_comprehensive_traversability'
            doc.mkdir(parents=True)
            primary = dict(trial_id='G_CURRENT_left', primary_protocol='CURRENT',
                           scene_id='A01', physical_config_id='cfg', seed=1,
                           probe_side='left', experiment_role='train')
            launch_files = []
            for name in ('G_OLD_left', primary['trial_id']):
                path = out / 'trials' / name / 'attempt01/campaign_launch.json'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(dict(kind='G1', trial_id=name, utc='2026-11-06T00:00:00+00:00')))
                launch_files.append(path)
            manifest = dict(g1_primary_protocol='CURRENT', max_g1_launches=1000,
                            trial_plans=[primary], prior_actual_g1_launches=[dict(
                                launch_file=launch_files[0].relative_to(root).as_posix(),
                                sha256=hashlib.sha256(launch_files[0].read_bytes()).hexdigest())])
            sync_launch_ledger(manifest, repo_root=root,
                               ledger_path=doc / 'G1_LAUNCH_LEDGER.json', initialize=True)
            for path in launch_files:
                path.unlink()
            campaign = out / 'campaign_manifest_v5.json'
            campaign.write_text(json.dumps(manifest))
            digest = hashlib.sha256(campaign.read_bytes()).hexdigest()
            campaign.with_suffix('.sha256.json').write_text(json.dumps(dict(sha256=digest)))
            (doc / 'CURRENT_CAMPAIGN.json').write_text(json.dumps(dict(manifest=campaign.name, sha256=digest)))
            with patch.object(checkpoint, 'ROOT', root), patch.object(checkpoint, 'OUT', out), \
                    patch.object(checkpoint, 'DOC', doc), redirect_stdout(io.StringIO()):
                checkpoint.main()
            result = json.loads((doc / 'PROGRESS_CHECKPOINT.json').read_text())
            self.assertEqual(result['g1_attempts_launched'], 2)
            self.assertEqual(result['primary_g1_attempts_launched'], 1)
            self.assertEqual(result['superseded_pilot_attempts_launched'], 1)
            self.assertEqual(result['historical_launches_not_present_locally'], 2)
            self.assertEqual(result['remaining_actual_g1_launch_budget'], 998)
            self.assertEqual(result['g1_launch_files_locally_available'], 0)
            import csv
            rows = list(csv.DictReader(io.StringIO((doc / 'TRIAL_PROGRESS_LEDGER.csv').read_text())))
            self.assertEqual(rows[0]['stage'], 'BLOCKED_WITH_REASON')
            self.assertEqual(rows[0]['attempts_launched'], '1')
            self.assertEqual(rows[0]['missing_raw_budget_withhold'], 'True')
            self.assertFalse(any(out.glob('trials/*/*/campaign_launch.json')))


if __name__ == '__main__':
    unittest.main()
