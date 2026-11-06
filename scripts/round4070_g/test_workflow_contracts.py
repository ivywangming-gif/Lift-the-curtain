"""CPU checks for immutable metadata and no-data training admission."""
import json
from pathlib import Path
import unittest
import uuid
from unittest.mock import patch

import workflow


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


if __name__ == '__main__':
    unittest.main()
