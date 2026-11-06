"""Synthetic durable-budget counterexamples; no simulator or model fitting."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest

from budget_ledger import (BudgetLedgerError, LAUNCH_PREFIX, LEGACY_PROTOCOL,
    append_actual_launch, launch_ledger_summary, load_launch_ledger,
    sync_launch_ledger, trial_budget_state)


TEMP_ROOT = Path(__file__).resolve().parents[2] / "outputs/round4070_comprehensive_traversability/implementation_smoke/budget_ledger_contracts"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)


class BudgetLedgerContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEMP_ROOT)
        self.root = Path(self.temp.name)
        self.ledger = self.root / "research/G1_LAUNCH_LEDGER.json"
        self.manifest = dict(g1_primary_protocol="SYNTHETIC_PROTOCOL_NOT_REAL_G_DATA", max_g1_launches=20,
            trial_plans=[dict(trial_id=t, primary_protocol="SYNTHETIC_PROTOCOL_NOT_REAL_G_DATA") for t in ["G_NEW", "G_OTHER"]],
            prior_actual_g1_launches=[])

    def tearDown(self):
        self.temp.cleanup()

    def launch(self, trial, attempt=1, extra=None):
        path = self.root / LAUNCH_PREFIX / trial / f"attempt{attempt:02d}" / "campaign_launch.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = dict(kind="G1", trial_id=trial, utc="2026-11-06T16:00:00+00:00")
        data.update(extra or {})
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def frozen(self, path):
        self.manifest["prior_actual_g1_launches"].append(dict(launch_file=path.relative_to(self.root).as_posix(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

    def summary(self, **kw):
        return launch_ledger_summary(self.manifest, repo_root=self.root, ledger_path=self.ledger, **kw)

    def sync(self, **kw):
        return sync_launch_ledger(self.manifest, repo_root=self.root, ledger_path=self.ledger, **kw)

    def test_fresh_checkout_retains_postfreeze_budget_and_withholds_plan(self):
        old = self.launch("G_OLD"); self.frozen(old)
        first = self.launch("G_NEW"); second = self.launch("G_NEW", 2)
        initial = self.sync(initialize=True)
        self.assertEqual(initial["count"], 3)
        old.unlink(); first.unlink(); second.unlink()
        restored = self.summary()
        self.assertEqual(restored["count"], 3)
        self.assertEqual(restored["local_launch_file_count"], 0)
        self.assertEqual(set(restored["withheld_trial_ids"]), {"G_OLD", "G_NEW"})
        state = trial_budget_state("G_NEW", restored)
        self.assertTrue(state["withhold"])
        self.assertEqual(state["minimum_next_attempt_number"], 3)

    def test_same_path_recreated_with_changed_bytes_is_hard_conflict(self):
        path = self.launch("G_NEW"); self.sync(initialize=True)
        previous = self.ledger.read_bytes()
        path.unlink(); self.launch("G_NEW", extra={"queue_pid": 777})
        with self.assertRaisesRegex(BudgetLedgerError, "DIFFERENT_LAUNCH_SHA256"):
            self.summary()
        with self.assertRaises(BudgetLedgerError):
            self.sync()
        self.assertEqual(self.ledger.read_bytes(), previous)

    def test_windows_path_aliases_deduplicate_and_original_entry_never_changes(self):
        path = self.launch("G_OLD"); self.frozen(path)
        self.manifest["prior_actual_g1_launches"].append(dict(
            launch_file=path.relative_to(self.root).as_posix().upper().replace("/", "\\"),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        result = self.sync(initialize=True)
        self.assertEqual(result["count"], 1)
        old_bytes = self.ledger.read_bytes(); old_entry = result["entries"][0]
        self.sync()
        self.assertEqual(self.ledger.read_bytes(), old_bytes)
        self.assertEqual(result["entries"][0]["primary_protocol"], LEGACY_PROTOCOL)
        self.launch("G_NEW"); updated = self.sync()
        self.assertEqual(updated["entries"][0], old_entry)

    def test_missing_historical_raw_counts_without_inventing_protocol_or_utc(self):
        old = self.launch("G_OLD"); self.frozen(old); old.unlink()
        result = self.sync(initialize=True)
        entry = result["entries"][0]
        self.assertEqual(result["count"], 1)
        self.assertEqual(entry["primary_protocol"], LEGACY_PROTOCOL)
        self.assertIsNone(entry["launch_utc"])
        self.assertEqual(entry["evidence_origin"], "FROZEN_HISTORY_WITHOUT_LOCAL_RAW")
        # Restoring exact raw never rewrites the original durable metadata.
        self.launch("G_OLD"); restored = self.sync()
        self.assertEqual(restored["entries"][0], entry)
        self.assertFalse(trial_budget_state("G_OLD", restored)["withhold"])

    def test_append_idempotent_new_attempt_costs_one_and_preserves_prior_entries(self):
        self.sync(initialize=True)
        first = self.launch("G_NEW")
        one = append_actual_launch(first, manifest=self.manifest, repo_root=self.root, ledger_path=self.ledger)
        before = self.ledger.read_bytes()
        again = append_actual_launch(first, manifest=self.manifest, repo_root=self.root, ledger_path=self.ledger)
        self.assertEqual((one["count"], again["count"]), (1, 1))
        self.assertEqual(before, self.ledger.read_bytes())
        second = self.launch("G_NEW", 2)
        two = append_actual_launch(second, manifest=self.manifest, repo_root=self.root, ledger_path=self.ledger)
        self.assertEqual(two["count"], 2)
        self.assertEqual(one["entries"], two["entries"][:1])

    def test_missing_corrupt_or_busy_ledger_does_not_self_reinitialize(self):
        with self.assertRaisesRegex(BudgetLedgerError, "LEDGER_MISSING"):
            self.summary()
        with self.assertRaisesRegex(BudgetLedgerError, "LEDGER_MISSING"):
            self.sync()
        self.sync(initialize=True)
        content = json.loads(self.ledger.read_text(encoding="utf-8")); content["entry_count"] = 1
        self.ledger.write_text(json.dumps(content), encoding="utf-8")
        with self.assertRaisesRegex(BudgetLedgerError, "COUNT_OR_DIGEST"):
            self.summary()
        lock = Path(str(self.ledger)+".lock"); lock.write_text('{"pid":987654,"token":"other"}')
        with self.assertRaisesRegex(BudgetLedgerError, "LOCK_BUSY"):
            self.sync()
        self.assertTrue(lock.exists())

    def test_path_traversal_payload_identity_and_protocol_rejected(self):
        self.sync(initialize=True)
        self.manifest["prior_actual_g1_launches"] = [dict(launch_file="../external/campaign_launch.json", sha256="0"*64)]
        with self.assertRaisesRegex(BudgetLedgerError, "INVALID_REPOSITORY_RELATIVE"):
            self.summary()
        self.manifest["prior_actual_g1_launches"] = []
        path = self.launch("G_NEW", extra={"trial_id": "G_OTHER"})
        with self.assertRaisesRegex(BudgetLedgerError, "TRIAL_PATH_MISMATCH"):
            self.summary()
        self.launch("G_NEW", extra={"primary_protocol": "WRONG_PROTOCOL"})
        with self.assertRaisesRegex(BudgetLedgerError, "PROTOCOL_CONFLICT"):
            self.summary()
        path.unlink()

    def test_overbudget_attempt_still_recorded_but_guard_must_not_start(self):
        self.manifest["max_g1_launches"] = 1
        self.sync(initialize=True)
        first = self.launch("G_NEW")
        append_actual_launch(first, manifest=self.manifest, repo_root=self.root, ledger_path=self.ledger)
        second = self.launch("G_OTHER")
        with self.assertRaisesRegex(BudgetLedgerError, "OVERRUN_RECORDED"):
            append_actual_launch(second, manifest=self.manifest, repo_root=self.root, ledger_path=self.ledger)
        summary = self.summary()
        self.assertEqual(summary["count"], 2)
        self.assertTrue(summary["budget_overrun"])


if __name__ == "__main__":
    unittest.main()
