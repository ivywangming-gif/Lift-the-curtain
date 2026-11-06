"""No-fit orchestration identity contracts; temporary synthetic files only."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import workflow
from train_benchmark import sources


class WorkflowLearningIdentityContracts(unittest.TestCase):
    def test_real_registry_exactly_matches_current_nine_trainer_sources(self):
        self.assertEqual(len(workflow.LEARNING_FILES),9)
        self.assertEqual(workflow.learning_source_hashes(),sources())
        self.assertTrue({"physics_history.py","collision_geometry.py","g1_action_contract_v2.py"} <= set(workflow.LEARNING_FILES))

    def test_each_new_dependency_changes_identity_without_overwriting_old_protocol(self):
        parent=workflow.ROOT/"outputs/round4070_comprehensive_traversability/contract_tests"
        parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            root=Path(directory);scripts=root/"scripts";scripts.mkdir();snapshot=root/"snapshot";snapshot.mkdir()
            for name in workflow.LEARNING_FILES:(scripts/name).write_text("synthetic initial "+name)
            (snapshot/"manifest.json").write_text('{}')
            def current_sources():return {n:workflow.sha(scripts/n) for n in workflow.LEARNING_FILES}
            calls=[]
            def nofit_freeze(script,*arguments):
                self.assertEqual(script,"train_benchmark.py");self.assertEqual(arguments[0],"freeze")
                run=Path(arguments[arguments.index("--output")+1]);run.mkdir(parents=True)
                calls.append(run)
                for name,value in (("frozen_protocol.json",dict(scope="SYNTHETIC_NO_FIT_IDENTITY_TEST",manifest_sha256=workflow.sha(snapshot/"manifest.json"),code_sha256=current_sources())),
                                   ("task_queue.json",dict(status="NO_FIT_SYNTHETIC_TEST",tasks=[]))):
                    with (run/name).open("x") as output:json.dump(value,output)
            with patch.object(workflow,"SCRIPTS",scripts),patch.object(workflow,"OUT",root/"output"),patch("train_benchmark.sources",side_effect=current_sources),patch.object(workflow,"call",side_effect=nofit_freeze):
                first=workflow.freeze_learning(snapshot,"frozen-dataset-id")
                original=(first/"frozen_protocol.json").read_bytes()
                self.assertEqual(workflow.freeze_learning(snapshot,"frozen-dataset-id"),first)
                self.assertEqual(len(calls),1)
                runs={first}
                for name in ("physics_history.py","collision_geometry.py","g1_action_contract_v2.py"):
                    (scripts/name).write_text("synthetic revised "+name)
                    runs.add(workflow.freeze_learning(snapshot,"frozen-dataset-id"))
                self.assertEqual(len(runs),4)
                self.assertEqual(len(calls),4)
                self.assertEqual((first/"frozen_protocol.json").read_bytes(),original)
                self.assertEqual(hashlib.sha256(original).hexdigest(),workflow.sha(first/"frozen_protocol.json"))

    def test_registry_drift_is_rejected_before_freeze(self):
        changed=sources();changed["new_unregistered_dependency.py"]="0"*64
        with patch("train_benchmark.sources",return_value=changed),patch.object(workflow,"call",side_effect=AssertionError("freeze must not start")):
            with self.assertRaises(RuntimeError):workflow.learning_source_hashes()


if __name__ == "__main__":unittest.main(verbosity=2)
