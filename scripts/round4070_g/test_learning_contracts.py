"""Small synthetic contract checks; these are not G benchmark experiments."""
import sys
from pathlib import Path
import tempfile
import unittest
import uuid

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from learning_core import (approved_pairs, causal_rows, conformal_decisions, conformal_fit,
    metrics, prepare_features, remap_active_hand, select_thresholds, split_indices)
from learning_models import METHODS, SmallModel, matched_pair_loss, supcon_loss
from resource_lock import resource_lock
from observation_randomization import fit_scale, transform, episode_seed

torch.set_num_threads(2)


def fixture():
    names = ["q/left_shoulder_pitch_joint", "q/right_shoulder_pitch_joint",
             "qdot/left_shoulder_pitch_joint", "qdot/right_shoulder_pitch_joint",
             "q_cmd/left_shoulder_pitch_joint", "q_cmd/right_shoulder_pitch_joint",
             "effort/left_shoulder_pitch_joint", "effort/right_shoulder_pitch_joint",
             "left_hand_fk_b/x", "right_hand_fk_b/x", "left_hand_delta_initial_b/x", "right_hand_delta_initial_b/x",
             "imu_rpy/roll", "probe_side/right", "elapsed_s", "dt_s", "command_progress/fraction", "cumulative_command/path", "stall/s"]
    t = np.arange(0., 2.001, .02)
    rng = np.random.default_rng(41)
    raw = [rng.normal(size=(len(t), len(names))).astype(np.float32) for _ in range(6)]
    episodes = [dict(trial_id=f"t{i}", y=i % 2, group_id=f"g{i}", side="right" if i % 2 else "left", is_control=i < 2,
                     action_contract="contract", followup_action_id="action") for i in range(6)]
    return names, raw, [t.copy() for _ in raw], episodes


class LearningContracts(unittest.TestCase):
    def test_fixed_prefix_never_reads_future(self):
        x = np.arange(90).reshape(30, 3)
        t = np.arange(30)/10
        a, ta = causal_rows(x, t, .5)
        altered = x.copy()
        altered[t > .5] = -999
        b, tb = causal_rows(altered, t, .5)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.all(ta <= .5))
        self.assertEqual(len(a), 6)

    def test_train_only_normalization_and_strict_instant(self):
        names, raw, times, episodes = fixture()
        _, _, _, audit = prepare_features(raw, times, names, episodes, [0, 1], "GRU")
        changed = [x.copy() for x in raw]
        changed[5] *= 10000
        _, _, _, second = prepare_features(changed, times, names, episodes, [0, 1], "GRU")
        np.testing.assert_array_equal(audit["mean"], second["mean"])
        for illegal in ("elapsed_s", "dt_s", "command_progress/fraction"):
            self.assertNotIn(illegal, audit["feature_names"])
        x, lengths, _, instant = prepare_features(raw, times, names, episodes, [0, 1], "Instantaneous")
        self.assertTrue(np.all(lengths == 1))
        self.assertFalse(any(any(s in n for s in ("delta_initial", "elapsed", "dt_s", "progress", "stall", "cumulative")) for n in instant["feature_names"]))
        _, _, _, command = prepare_features(raw, times, names, episodes, [0, 1], "GRUCommand")
        self.assertIn("command_progress/fraction", command["feature_names"])
        self.assertNotIn("derived/dt_s", command["feature_names"])
        _, _, _, time_progress = prepare_features(raw, times, names, episodes, [0, 1], "GRUTimeProgress")
        self.assertIn("derived/dt_s", time_progress["feature_names"])

    def test_real_rate_ablation_without_interpolation(self):
        names, raw, times, episodes = fixture()
        _, lengths50, _, _ = prepare_features(raw, times, names, episodes, [0, 1], "GRU")
        _, lengths20, t20, _ = prepare_features(raw, times, names, episodes, [0, 1], "GRU20Hz")
        _, lengths10, t10, _ = prepare_features(raw, times, names, episodes, [0, 1], "GRU10Hz")
        self.assertTrue(np.all(lengths50 == 101))
        self.assertTrue(np.all(lengths20 == 41))
        self.assertTrue(np.all(lengths10 == 21))
        self.assertTrue(all(np.min(np.abs(times[0]-float(value))) <= 1e-7 for value in t10[0, :21]))

    def test_group_exclusion_and_pair_truth(self):
        names, raw, times, episodes = fixture()
        episodes[1]["group_id"] = episodes[0]["group_id"]
        with self.assertRaises(ValueError):
            split_indices(dict(train=["t0"], calibration=[], test=["t1"], split_type="anchor"), episodes)
        episodes[1]["group_id"] = "g1"
        pair = dict(a="t0", b="t1", kind="hard_negative", evidence={"measured_geometry_matched": True})
        approved = approved_pairs({"by_split": {"a": [pair]}}, "a", [0, 1], episodes)
        self.assertEqual(approved, [(0, 1, "hard_negative")])
        with self.assertRaises(ValueError):
            approved_pairs({"by_split": {"a": [pair]}}, "a", [0], episodes)
        episodes[1]["y"] = -1
        with self.assertRaises(ValueError):
            approved_pairs({"by_split": {"a": [pair]}}, "a", [0, 1], episodes)

    def test_rejection_and_single_class_metrics(self):
        result = metrics([0, 0, 1, 1], [.1, .5, .9, .5], .2, .8)
        self.assertEqual((result["true_block_n"], result["true_pass_n"], result["uncertain_n"]), (1, 1, 2))
        self.assertEqual(result["balanced_accuracy"], .5)
        self.assertEqual(result["coverage"], .5)
        self.assertIsNone(metrics([0], [.1])["balanced_accuracy"])
        with self.assertRaises(ValueError):
            metrics([-1], [.5])
        fit = conformal_fit([0, 1], [.1, .9])
        np.testing.assert_array_equal(conformal_decisions([.01, .99], fit), [-1, -1])
        self.assertIsNone(select_thresholds([0, 0], [.1, .3])["low"])

    def test_hand_mapping_is_bijective_not_label_mirroring(self):
        names, raw, times, episodes = fixture()
        mirrored = remap_active_hand(raw[0], names, "left")
        np.testing.assert_array_equal(mirrored[:, 0], raw[0][:, 1])
        np.testing.assert_array_equal(remap_active_hand(mirrored, names, "left"), raw[0])
        self.assertEqual(episodes[0]["y"], 0)

    def test_every_neural_family_forward_backward_and_padding(self):
        torch.manual_seed(17)
        x = torch.randn(4, 12, 8)
        lengths = torch.tensor([12, 7, 10, 4])
        for i, n in enumerate(lengths):
            x[i, n:] = 0
        for method in METHODS:
            if method in ("Logistic", "DurationOnlyLogistic", "GradientBoosting", "GaussianAR1"):
                continue
            with self.subTest(method=method):
                model = SmallModel(method, 8, max_steps=12, hidden=16, arm=[0,1,2,3], body=[4,5,6,7], response_indices=[0,1])
                model.eval()
                logits, z, response = model(x, lengths, return_aux=True)
                self.assertEqual(tuple(logits.shape), (4,))
                self.assertTrue(torch.isfinite(logits).all())
                logits.square().mean().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                if z is not None:
                    self.assertTrue(torch.isfinite(supcon_loss(z, torch.tensor([0,0,1,1]))[0]))
                    self.assertTrue(torch.isfinite(matched_pair_loss(z, [(0,1,"cross_configuration_positive"),(0,2,"hard_negative")], method)[0]))

    def test_atomic_lock_excludes_other_owner(self):
        path = Path(__file__).resolve().parents[2]/"outputs/round4070_comprehensive_traversability/contract_tests"/(uuid.uuid4().hex+".json")
        with resource_lock("test_learning", "synthetic", path):
            with self.assertRaises(InterruptedError):
                with resource_lock("test_physics", "synthetic", path):
                    self.fail("Concurrent owners acquired the same resource")
        self.assertFalse(path.exists())

    def test_observation_dr_train_scale_causality_and_reproducibility(self):
        names, raw, times, episodes = fixture()
        ids = [e["trial_id"] for e in episodes]
        fitted = fit_scale(raw, names, [0,1], ids)
        changed = [x.copy() for x in raw]
        changed[-1] *= 1e6
        self.assertEqual(fitted, fit_scale(changed, names, [0,1], ids))
        seed = episode_seed(40700, "synthetic")
        a, ta, audit = transform(raw[0], times[0], names, fitted, seed, cutoff_s=.5)
        b, tb, second = transform(raw[0], times[0], names, fitted, seed, cutoff_s=.5)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(ta, tb)
        self.assertEqual(audit, second)
        self.assertTrue(np.all(np.diff(ta) > 0))
        self.assertTrue(all(t <= .5+1e-9 for t in audit["source_acquisition_times"]))
        self.assertTrue(all(source <= receipt for source, receipt in zip(audit["source_acquisition_times"], audit["receipt_true_times"])))
        self.assertLessEqual(audit["max_actual_delay_s"], .0200001)
        command = names.index("q_cmd/left_shoulder_pitch_joint")
        np.testing.assert_array_equal(a[:, command], raw[0][audit["receipt_indices"], command])
        self.assertEqual([e["y"] for e in episodes], [0,1,0,1,0,1])

    def test_conformal_scores_do_not_refit_temperature(self):
        from train_benchmark import calibrate_and_evaluate
        from learning_core import read_json
        p = np.linspace(.03, .97, 22)
        episodes = [dict(trial_id=f"c{i}", y=int(i >= 10), group_id=f"cg{i}", side="left", is_control=False) for i in range(22)]
        roles = dict(train=[], calibration=list(range(20)), test=[20,21])
        folder = Path(__file__).resolve().parents[2]/"outputs/round4070_comprehensive_traversability/contract_tests"/uuid.uuid4().hex
        folder.mkdir(parents=True)
        protocol = dict(risk_grid=[0.], conformal_alpha=.1)
        task = dict(split_id="synthetic_contract", method="GRU", seed=0)
        rows, predictions, bins = calibrate_and_evaluate(p, roles, episodes, protocol, folder, task)
        saved = read_json(folder/"calibration.json")
        expected = conformal_fit([e["y"] for e in episodes[:20]], p[:20])
        self.assertEqual(saved["conformal"]["q"], expected["q"])
        self.assertTrue(saved["conformal"]["score_source"].startswith("frozen_model_raw_probability"))
        np.testing.assert_array_equal([r["p"] for r in predictions if r["mode"] == "label_conditional_conformal"], p[20:])
        roles["calibration"] = [0,1]
        _, _, bins = calibrate_and_evaluate(p, roles, episodes, protocol, folder, task)
        self.assertEqual({r["probability_kind"] for r in bins}, {"raw"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
