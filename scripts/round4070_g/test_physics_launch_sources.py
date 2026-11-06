"""No subprocess or App startup: provenance survives a hypothetical early exit."""
from pathlib import Path
import hashlib,tempfile,unittest
from physics_campaign import archive_source_files,launch_source_paths,ROOT,GUARD


class LaunchSourceTests(unittest.TestCase):
    def test_exact_bytes_deduplicated_and_preserved_after_live_source_changes(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs/round4070_comprehensive_traversability/implementation_smoke') as directory:
            run=Path(directory);source=run/'example.py';original=b'first\r\nsecond\n';source.write_bytes(original)
            digests,archive=archive_source_files([source,source],run)
            self.assertEqual(len(archive),1);entry=next(iter(archive.values()));self.assertEqual(entry['sha256'],hashlib.sha256(original).hexdigest())
            source.write_bytes(b'changed')
            self.assertEqual((run/entry['path']).read_bytes(),original)
            with self.assertRaises(FileExistsError):archive_source_files([source],run)
    def test_required_runtime_and_guard_sources_included(self):
        invocation=ROOT/'outputs/round4070_comprehensive_traversability/launch_guard.ps1'
        paths=launch_source_paths(ROOT/'scripts/round4070_g/g1_trial.py',invocation);names={p.name for p in paths}
        for name in ['g1_trial.py','g1_recording.py','g1_audit.py','g1_randomization.py','g1_action_contract_v2.py','physics_native_bind.py',
                     'physics_history.py','physics_scenes.py','physics_scene_builder.py','physics_admission.py','physics_campaign.py',
                     'collision_geometry.py','collision_clearance.py','resource_lock.py','budget_ledger.py','g1_quarantine.py','run_episode.py','launch_guard.ps1',GUARD.name]:
            self.assertIn(name,names)


if __name__=='__main__':unittest.main(verbosity=2)
