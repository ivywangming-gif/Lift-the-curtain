"""Small synthetic contract checks; these are not G benchmark experiments."""
import sys
from pathlib import Path
import tempfile
import unittest
import uuid

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from learning_core import (ACTION_CONTRACT,PRIMARY_PROTOCOL,LABEL_VALIDITY_VERSION,validate_primary_manifest,
    approved_pairs, apply_label_budget, frozen_label_budget, causal_rows, conformal_decisions, conformal_fit,
    metrics, prepare_features, remap_active_hand, select_thresholds, split_indices)
from learning_models import METHODS, SmallModel, matched_pair_loss, supcon_loss
from resource_lock import resource_lock
from observation_randomization import fit_scale, transform, episode_seed

torch.set_num_threads(2)


def fixture():
    from g1_action_contract_v2 import draft_contract
    contract=draft_contract()
    contract.update(status="VALIDATED_READY_TO_FREEZE")
    contract["witness"].update(status="VERIFIED",evidence_mode="EXACT_NATIVE",point_body_m=[0.,0.,0.],verified_interior_radius_m=.01,
        body_name="SYNTHETIC_BODY",body_index=0,native_shape_id="SYNTHETIC_SHAPE",native_geometry_sha256="0"*64,
        native_library_sha256="0"*64,source_evidence_sha256="0"*64,native_library_path="SYNTHETIC_NO_LIBRARY",source_evidence_path="SYNTHETIC_NO_PROOF")
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
                     action_contract=ACTION_CONTRACT, followup_action_id="SYNTHETIC_ACTION",synthetic=True,seed=40700+i,
                     primary_protocol=PRIMARY_PROTOCOL,label_validity_version=LABEL_VALIDITY_VERSION,
                     action_contract_v2=contract,action_contract_v2_sha256="0"*64,
                     primary_admission=True,admission_status="PRIMARY_VERIFIED",response_history_eligible=True,supervision_eligible=True,
                     body_collision_audit=dict(status="VERIFIED",sample_count=101,scope="SYNTHETIC_FIXTURE_NOT_PHYSICS_EVIDENCE",source_evidence_path="SYNTHETIC_COLLISION_NO_REAL_FILE",source_evidence_sha256="0"*64,
                        action_contract_audit=dict(status="VERIFIED",all_required_steps_valid=True,guard_violation=None,static_blocking_certificate=dict(status="VERIFIED"))),
                     reset_randomization_evidence=dict(readback_verified=True,seed=40700+i,source_evidence_path="SYNTHETIC_RNG_NO_REAL_FILE",source_evidence_sha256="0"*64)) for i in range(6)]
    return names, raw, [t.copy() for _ in raw], episodes


class LearningContracts(unittest.TestCase):
    def test_primary_admission_rejects_pilots_pending_and_synthetic_benchmarks(self):
        import copy
        _,_,_,episodes = fixture()
        manifest = dict(schema="round4070-g.causal_episodes.v3",primary_protocol=PRIMARY_PROTOCOL,label_validity_version=LABEL_VALIDITY_VERSION,
                        data_scope="SYNTHETIC_CONTRACT_TEST",episodes=episodes)
        validate_primary_manifest(manifest,verify_sources=False,synthetic_test=True)
        with self.assertRaises(ValueError):
            validate_primary_manifest(manifest,verify_sources=False)
        changed = copy.deepcopy(manifest); changed["primary_protocol"] = "V3_PILOT"
        with self.assertRaises(ValueError):
            validate_primary_manifest(changed,verify_sources=False,synthetic_test=True)
        for key,value in (("primary_admission",False),("admission_status","PENDING_BODY_COLLISION_AUDIT"),("response_history_eligible",False)):
            changed = copy.deepcopy(manifest); changed["episodes"][0][key] = value
            with self.assertRaises(ValueError):
                validate_primary_manifest(changed,verify_sources=False,synthetic_test=True)
        changed = copy.deepcopy(manifest)
        changed["episodes"][0].update(y=-1,supervision_eligible=False)
        changed["episodes"][0]["body_collision_audit"]["status"] = "NOT_VERIFIED"
        with self.assertRaises(ValueError):
            validate_primary_manifest(changed,verify_sources=False,synthetic_test=True)
        # Empty primary is valid for a no-training/all-blocked freeze.
        validate_primary_manifest(dict(schema="round4070-g.causal_episodes.v3",primary_protocol=PRIMARY_PROTOCOL,
                                       label_validity_version=LABEL_VALIDITY_VERSION,episodes=[]))

    def test_v5_action_violation_is_unknown_only_and_draft_rejected(self):
        import copy
        _,_,_,episodes=fixture()
        manifest=dict(schema="round4070-g.causal_episodes.v3",primary_protocol=PRIMARY_PROTOCOL,label_validity_version=LABEL_VALIDITY_VERSION,
                      data_scope="SYNTHETIC_CONTRACT_TEST",episodes=episodes)
        changed=copy.deepcopy(manifest)
        changed["episodes"][0]["body_collision_audit"]["action_contract_audit"].update(guard_violation="TUBE_EXIT",all_required_steps_valid=False)
        with self.assertRaises(ValueError):validate_primary_manifest(changed,verify_sources=False,synthetic_test=True)
        changed["episodes"][0].update(y=-1,supervision_eligible=False)
        validate_primary_manifest(changed,verify_sources=False,synthetic_test=True)
        changed["episodes"][0]["action_contract_v2"]["status"]="DRAFT_PENDING_NATIVE_WITNESS"
        with self.assertRaises(ValueError):validate_primary_manifest(changed,verify_sources=False,synthetic_test=True)

    def test_partial_heldout_scope_retains_full_split_and_method_gates(self):
        from train_benchmark import partial_heldout_tasks
        _,_,_,episodes=fixture()
        split=dict(split_id="s",split_type="configuration",train=["t2","t3"],calibration=["t1"],test=["t4"],status="NOT_EVALUABLE")
        base=dict(task_id="s__GRU__40700",split_id="s",method="GRU",seed=40700,label_fraction=1.,
                  status="BLOCKED_WITH_REASON",reasons=["TEST_CORE_BOTH_LABELS_NOT_SUPPORTED","TEST_BOTH_CLASSES_NOT_SUPPORTED"])
        tasks=partial_heldout_tasks(split,episodes,[base,{**base,"task_id":"residual","method":"ResidualGRU","reasons":base["reasons"]+["NO_TRAIN_EMPTY_CONTROL_FOR_NORMAL_RESPONSE"]}])
        self.assertEqual(tasks[0]["status"],"PLANNED")
        self.assertEqual(tasks[0]["evaluation_scope"],"PARTIAL_HELDOUT_SINGLE_CLASS")
        self.assertEqual(tasks[0]["heldout_supported_class"],0)
        self.assertEqual(tasks[1]["status"],"BLOCKED_WITH_REASON")
        self.assertEqual(base["status"],"BLOCKED_WITH_REASON")
        self.assertEqual(split["status"],"NOT_EVALUABLE")
        split["test"]=["t4","t5"]
        self.assertFalse(partial_heldout_tasks(split,episodes,[base]))
        split["test"]=[]
        self.assertFalse(partial_heldout_tasks(split,episodes,[base]))

    def test_partial_freeze_and_uncalibrated_prediction_contract(self):
        import argparse,json
        from unittest.mock import patch
        from train_benchmark import freeze,calibrate_and_evaluate
        from learning_report import generate
        from learning_core import write_csv,write_json,sha,read_json
        names,raw,times,episodes=fixture()
        parent=Path(__file__).resolve().parents[2]/"outputs/round4070_comprehensive_traversability/contract_tests"
        parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            folder=Path(directory)
            manifest=folder/"manifest.json";manifest.write_text('{}')
            split=dict(split_id="synthetic_single_class",split_type="configuration",train=["t2","t3"],calibration=["t1"],test=["t4"],
                       status="NOT_EVALUABLE",reasons=["TEST_CORE_BOTH_LABELS_NOT_SUPPORTED"])
            splits=folder/"splits.json";splits.write_text(json.dumps(dict(splits=[split])))
            args=argparse.Namespace(methods="GRU",seeds=list(range(5)),output=folder/"freeze",manifest=manifest,splits=splits,pairs=None,legacy=False)
            with patch("train_benchmark.load_dataset",return_value=({},episodes,names,raw,times,{})),patch("builtins.print"):
                freeze(args)
                generate(args.output,folder/"report")
            queue=json.loads((args.output/"task_queue.json").read_text())
            self.assertEqual(sum(t["status"]=="PLANNED" for t in queue["tasks"]),5)
            self.assertTrue(all(t["evaluation_scope"]=="PARTIAL_HELDOUT_SINGLE_CLASS" for t in queue["tasks"] if t["status"]=="PLANNED"))
            # A synthetic completed partial artifact must never increment the
            # full benchmark or enter its main method/confirmation tables.
            task=next(t for t in queue["tasks"] if t["status"]=="PLANNED")
            task["status"]="EVALUATED"
            fit=args.output/"fits"/task["task_id"];fit.mkdir(parents=True)
            row=dict(split_id=task["split_id"],method="GRU",seed=task["seed"],mode="forced_raw",subset="all",subset_value="all",
                     evaluation_scope="PARTIAL_HELDOUT_SINGLE_CLASS",label_fraction=1.,**metrics([0],[.9]))
            write_csv(fit/"metrics.csv",[row])
            write_csv(fit/"predictions.csv",[dict(split_id=task["split_id"],method="GRU",seed=task["seed"],mode="forced_raw",
                evaluation_scope="PARTIAL_HELDOUT_SINGLE_CLASS",trial_id="t4",group_id="g4",side="left",y=0,p=.9,decision=1,is_control=False,label_fraction=1.)])
            write_json(fit/"completion.json",dict(predictions_sha256=sha(fit/"predictions.csv"),checkpoint_sha256=None))
            write_json(args.output/"task_queue.json",queue)
            with patch("builtins.print"):generate(args.output,folder/"report")
            state=read_json(folder/"report/LEARNING_DELIVERY_STATUS.json")
            self.assertEqual(state["evaluated_fit_n"],0)
            self.assertEqual(state["partial_single_class_evaluated_fit_n"],1)
            self.assertEqual(len((folder/"report/STANDARD_LEARNING_BASELINES.csv").read_text().splitlines()),1)
            self.assertEqual(len((folder/"report/PARTIAL_HELDOUT_SINGLE_CLASS_RESULTS.csv").read_text().splitlines()),2)
            calibrated=folder/"uncalibrated";calibrated.mkdir()
            with patch("torch.optim.LBFGS",side_effect=AssertionError("Single-class calibration must not fit temperature")):
                rows,predictions,_=calibrate_and_evaluate(np.full(6,.7),dict(train=[2,3],calibration=[5],test=[4]),episodes,
                    dict(risk_grid=[.05],conformal_alpha=.1),calibrated,dict(split_id="synthetic",method="GRU",seed=0))
            self.assertEqual({r["mode"] for r in rows},{"forced_raw","fixed_reject_raw"})
            self.assertTrue(all(r["balanced_accuracy"] is None and r["auroc"] is None for r in rows))
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

    def test_label_budgets_nested_holdout_untouched_and_reports_separate(self):
        from learning_report import average_rows
        episodes = [dict(trial_id=f"b{i}", y=i % 2, label="PASSABLE" if i % 2 else "BLOCKED", group_id=f"bg{i}", is_control=i < 2) for i in range(30)]
        train = list(range(24))
        quarter = frozen_label_budget(episodes, train, "synthetic_budget", .25)
        half = frozen_label_budget(episodes, train, "synthetic_budget", .5)
        full = frozen_label_budget(episodes, train, "synthetic_budget", 1.)
        self.assertEqual(quarter["selected_class_counts"], {"0":3, "1":3})
        self.assertEqual(half["selected_labelled_train_n"], 12)
        self.assertTrue(set(quarter["selected_train_label_ids"]).issubset(half["selected_train_label_ids"]))
        self.assertTrue(set(half["selected_train_label_ids"]).issubset(full["selected_train_label_ids"]))
        self.assertEqual(quarter, frozen_label_budget(episodes, train[::-1], "synthetic_budget", .25))
        altered = [dict(e) for e in episodes]
        for e in altered[24:]:
            e["y"] = -1
        self.assertEqual(quarter, frozen_label_budget(altered, train, "synthetic_budget", .25))
        reduced = apply_label_budget(episodes, train, quarter)
        self.assertEqual(reduced[24:], episodes[24:])
        self.assertEqual(sum(e["y"] >= 0 for e in reduced[:24]), 6)
        self.assertEqual(sum(e["y"] == -1 for e in reduced[:24]), 18)
        self.assertEqual([e["y"] for e in episodes], [i % 2 for i in range(30)])
        rows = [dict(split_id="s", method=display, base_method="GRU", label_fraction=fraction, supervised_train_n=count,
                     mode="forced_raw", seed=seed, n=2, passable_n=1, blocked_n=1, false_pass_n=0)
                for display, fraction, count in (("GRU", 1., 24), ("GRU__labels050", .5, 12), ("GRU__labels025", .25, 6)) for seed in range(5)]
        summary = average_rows(rows)
        self.assertEqual(len(summary), 3)
        self.assertTrue(all(r["actual_fits"] == 5 for r in summary))

    def test_confirmation_role_exclusion_and_normalization(self):
        from train_benchmark import evaluation_subsets
        names, raw, times, episodes = fixture()
        episodes[4]["experiment_role"] = "confirmation"
        split = dict(split_type="anchor", train=["t0","t1"], calibration=["t2"], test=["t3"], confirmation=["t4"])
        roles = split_indices(split, episodes)
        self.assertEqual(roles["confirmation"], [4])
        with self.assertRaises(ValueError):
            split_indices({**split, "train":["t0","t1","t4"], "confirmation":[]}, episodes)
        with self.assertRaises(ValueError):
            split_indices({**split, "test":["t3","t4"]}, episodes)
        episodes[4]["group_id"] = "g0"
        with self.assertRaises(ValueError):
            split_indices(split, episodes)
        episodes[4]["group_id"] = "g3"
        split_indices(split, episodes)  # confirmation may share the heldout test domain
        _, _, _, before = prepare_features(raw, times, names, episodes, roles["train"], "GRU")
        changed = [x.copy() for x in raw]
        changed[4] *= 1e6
        _, _, _, after = prepare_features(changed, times, names, episodes, roles["train"], "GRU")
        self.assertEqual(before["mean"], after["mean"])
        self.assertEqual(before["scale"], after["scale"])
        subsets = evaluation_subsets([0,1,2,3], episodes)
        self.assertEqual({r[0] for r in subsets}, {"all","side","group_id","condition_family","robot_variant"})

    def test_confirmation_reuses_calibrator_without_any_refit(self):
        from unittest.mock import patch
        from train_benchmark import calibrate_and_evaluate
        from learning_core import read_json
        p = np.linspace(.03,.97,24)
        episodes = [dict(trial_id=f"cf{i}", y=i % 2, group_id=f"cfg{i}", side="left", is_control=False) for i in range(24)]
        roles = dict(train=[], calibration=list(range(20)), test=[20,21], confirmation=[22,23])
        folder = Path(__file__).resolve().parents[2]/"outputs/round4070_comprehensive_traversability/contract_tests"/uuid.uuid4().hex
        folder.mkdir(parents=True)
        protocol = dict(risk_grid=[0.], conformal_alpha=.1)
        task = dict(split_id="synthetic_confirmation", method="GRU", seed=0)
        calibrate_and_evaluate(p, roles, episodes, protocol, folder, task)
        frozen = read_json(folder/"calibration.json")
        confirm_folder = folder/"confirmation"
        confirm_folder.mkdir()
        altered = p.copy(); altered[22:] = [1.,0.]
        with patch("torch.optim.LBFGS", side_effect=AssertionError("Unexpected temperature refit")), patch("train_benchmark.select_thresholds", side_effect=AssertionError("Unexpected threshold refit")), patch("train_benchmark.conformal_fit", side_effect=AssertionError("Unexpected conformal refit")):
            _, predictions, _ = calibrate_and_evaluate(altered, {**roles,"test":roles["confirmation"]}, episodes, protocol, confirm_folder, task, fitted_calibration=frozen)
        reused = read_json(confirm_folder/"calibration.json")
        for key in ("temperature","thresholds","conformal","calibration_ids"):
            self.assertEqual(frozen[key], reused[key])
        self.assertEqual({r["trial_id"] for r in predictions}, {"cf22","cf23"})
        self.assertEqual(reused["calibration_operation"], "REUSED_FROZEN_MAIN_CALIBRATION")

    def test_cluster_bootstrap_seed_unit_risk_modes_and_confirmation(self):
        from learning_report import paired_group_bootstrap
        rows = []
        for mode in ("forced_raw", "calibration_risk_0.05", "confirmation/calibration_risk_0.05"):
            for method in ("GRU","GRUResponseAux"):
                for i in range(6):
                    truth = i % 2
                    decision = (0 if truth == 0 else -1) if method == "GRU" else (int(i == 0) if truth == 0 else 1)
                    for seed in range(5):
                        rows.append(dict(split_id="s",mode=mode,method=method,trial_id=f"e{i}",seed=str(seed),y=str(truth),group_id=f"g{i//2}",decision=str(decision),is_control="False"))
        five = paired_group_bootstrap(rows, list(range(5)))
        one = paired_group_bootstrap(rows, [0])
        self.assertEqual(len(five), 12)
        self.assertEqual({r["evaluation_endpoint"] for r in five}, {"main_test","confirmation"})
        for a,b in zip(five,one):
            self.assertEqual(a["physical_group_n"], 3)
            self.assertEqual(a["paired_episode_n"], 6 if a["metric"] == "coverage" else 3)
            self.assertEqual((a["ci95_low"],a["ci95_high"]),(b["ci95_low"],b["ci95_high"]))
        cov = next(r for r in five if r["metric"] == "coverage")
        recall = next(r for r in five if r["metric"] == "passable_recall")
        self.assertEqual(cov["mean_difference"], .5)
        self.assertEqual(recall["mean_difference"], 1.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
