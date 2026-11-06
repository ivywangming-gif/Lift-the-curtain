"""CPU checks for immutable metadata and no-data training admission."""
import json
import hashlib
from pathlib import Path
import unittest
import uuid
from unittest.mock import patch

import workflow
import checkpoint


class WorkflowContracts(unittest.TestCase):
    def setUp(self):
        # Keep tiny synthetic fixtures inside the authorized G output root.
        self.fixture = workflow.OUT / 'implementation_smoke' / ('workflow_' + uuid.uuid4().hex)
        (self.fixture / 'dataset').mkdir(parents=True)

    def test_snapshot_immutability_and_tamper_rejection(self):
        with patch.object(workflow, 'OUT', self.fixture):
            for name in workflow.DATA_FILES:
                (self.fixture / 'dataset' / name).write_text(json.dumps(dict(version=1, name=name)))
            first, identity = workflow.snapshot_dataset()
            second, identity2 = workflow.snapshot_dataset()
            self.assertEqual((first, identity), (second, identity2))
            (self.fixture / 'dataset/manifest.json').write_text(json.dumps(dict(version=2)))
            third, identity3 = workflow.snapshot_dataset()
            self.assertNotEqual(identity, identity3)
            self.assertEqual(json.loads((first / 'manifest.json').read_text())['version'], 1)
            (third / 'manifest.json').write_text('tampered')
            with self.assertRaisesRegex(RuntimeError, 'Immutable snapshot changed'):
                workflow.snapshot_dataset()

    def test_blocked_queue_never_starts_training_or_creates_window(self):
        run = self.fixture / 'learning'
        run.mkdir()
        (run / 'task_queue.json').write_text(json.dumps(dict(tasks=[dict(status='BLOCKED_WITH_REASON')])))
        with patch.object(workflow, 'OUT', self.fixture), patch.object(workflow, 'call') as call:
            workflow.train(run)
            call.assert_not_called()
        self.assertFalse((self.fixture / 'training_window.json').exists())

    def test_g1_admission_pause_prevents_even_planned_training(self):
        (self.fixture / 'PAUSE_G1.flag').write_text('actual collision validation pending')
        run = self.fixture / 'learning'
        run.mkdir()
        (run / 'task_queue.json').write_text(json.dumps(dict(tasks=[dict(status='PLANNED')])))
        with patch.object(workflow, 'OUT', self.fixture), patch.object(workflow, 'call') as call:
            workflow.train(run)
            call.assert_not_called()
        self.assertFalse((self.fixture / 'training_window.json').exists())

    def test_current_manifest_hash_is_mandatory(self):
        document = self.fixture / 'bundled'
        document.mkdir()
        source = self.fixture / 'campaign_manifest_v4.json'
        source.write_bytes(b'{"frozen": true}')
        (document / 'CURRENT_CAMPAIGN.json').write_text(json.dumps(dict(manifest=source.name,
            sha256=hashlib.sha256(source.read_bytes()).hexdigest())))
        with patch.object(workflow, 'OUT', self.fixture), patch.object(workflow, 'DOC', document):
            self.assertEqual(workflow.current_manifest(), source)
            source.write_bytes(b'{"frozen": false}')
            with self.assertRaisesRegex(RuntimeError, 'SHA256 mismatch'):
                workflow.current_manifest()

    def test_checkpoint_uses_frozen_pointer_not_highest_draft_version(self):
        document=self.fixture/'bundled';document.mkdir()
        source=self.fixture/'campaign_manifest_v4.json'
        source.write_bytes(b'{"version":4}')
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        source.with_suffix('.sha256.json').write_text(json.dumps(dict(sha256=digest)))
        (self.fixture/'campaign_manifest_v99.json').write_bytes(b'{"unfrozen_draft":true}')
        (document/'CURRENT_CAMPAIGN.json').write_text(json.dumps(dict(manifest=source.name,sha256=digest)))
        with patch.object(checkpoint,'OUT',self.fixture),patch.object(checkpoint,'DOC',document):
            self.assertEqual(checkpoint.current_campaign(),(source,digest))
            source.write_bytes(b'{"changed":true}')
            with self.assertRaisesRegex(RuntimeError,'SHA256 mismatch'):
                checkpoint.current_campaign()


if __name__ == '__main__':
    unittest.main()
