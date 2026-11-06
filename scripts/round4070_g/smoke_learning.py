"""Tiny synthetic fit/serialization/resume test; NEVER a research result.

Requires the same resource lock as simulation. Uses CPU two threads by default.
Writes only under a unique outputs/.../implementation_smoke/ directory.
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import uuid

import numpy as np
import torch

from learning_core import ROOT, prepare_features, write_json
from learning_models import METHODS
from resource_lock import resource_lock
from test_learning_contracts import fixture
from train_benchmark import calibrate_and_evaluate, checkpoint_predict, require_window, train_fit
from observation_randomization import fit_scale, transform_indices


def main(device="cpu"):
    torch.set_num_threads(2)
    root = ROOT/"outputs/round4070_comprehensive_traversability/implementation_smoke"/uuid.uuid4().hex
    root.mkdir(parents=True)
    permit = root/"synthetic_only_window.json"
    write_json(permit, dict(allow_training=True, isaac_stopped_confirmed=True,
                           expires_utc=(dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=5)).isoformat(),
                           scope="EXPLICITLY_AUTHORIZED_TINY_SYNTHETIC_IMPLEMENTATION_TEST_ONLY"))
    with resource_lock("learning_synthetic_smoke", str(root)):
        require_window(permit)
        names, raw, times, episodes = fixture()
        rng = np.random.default_rng(80)
        for i in range(6, 10):
            raw.append(rng.normal(size=raw[0].shape).astype(np.float32))
            times.append(times[0].copy())
            episodes.append(dict(trial_id=f"t{i}", y=i % 2, group_id=f"g{i}", side="right" if i % 2 else "left", is_control=False, action_contract="contract", followup_action_id="action"))
        episodes[4]["y"] = -1
        roles = dict(train=[0,1,2,3,4], calibration=[5,6], test=[7,8,9])
        protocol = dict(epochs=2, pretrain_epochs=1, batch_size=4, hidden=16, max_steps=12, cutoff_s=2.,
                        lr=.001, weight_decay=.01, logistic_max_iter=200, logistic_tolerance_grad=1e-8,
                        gradient_clip=5., response_weight=.1, contrastive_weight=.1, supcon_temperature=.1,
                        risk_grid=[0., .05, .2], conformal_alpha=.1, scope="SYNTHETIC_IMPLEMENTATION_ONLY")
        pairs = [(0,2,"cross_configuration_positive"),(1,3,"cross_configuration_positive"),(0,1,"hard_negative"),(2,3,"hard_negative")]
        ids = [e["trial_id"] for e in episodes]
        noise_scale = fit_scale(raw, names, roles["train"], ids)
        results = []
        for method in METHODS:
            folder = root/method
            folder.mkdir()
            x, lengths, _, audit = prepare_features(raw, times, names, episodes, roles["train"], method, max_steps=12)
            if method == "GRUObsDR":
                augmented, stamps, noise_audit = transform_indices(raw, times, names, noise_scale, roles["train"], ids, 40700)
                x, lengths, _, audit = prepare_features(augmented, stamps, names, episodes, roles["train"], method, max_steps=12, normalization_override=audit)
                write_json(folder/"training_observation_transform.json", noise_audit)
            probabilities, curve, meta = train_fit(method, 40700, x, lengths, audit, roles, episodes, pairs, protocol, folder, device, permit)
            assert np.isfinite(probabilities).all() and len(probabilities) == 10
            assert (folder/"checkpoint.pt").exists()
            restored = torch.load(folder/"checkpoint.pt", map_location="cpu", weights_only=False)
            assert restored["method"] == method
            replay = checkpoint_predict(folder/"checkpoint.pt", x, lengths, audit, device)
            np.testing.assert_allclose(probabilities, replay, atol=1e-6, rtol=0)
            noisy_raw, noisy_t, noise_audit = transform_indices(raw, times, names, noise_scale, roles["test"], ids, 40706000)
            nx, nl, _, na = prepare_features(noisy_raw, noisy_t, names, episodes, roles["train"], method, max_steps=12, normalization_override=audit)
            noisy_p = checkpoint_predict(folder/"checkpoint.pt", nx, nl, na, device)
            assert np.isfinite(noisy_p).all()
            write_json(folder/"test_observation_transform.json", noise_audit)
            rows, predictions, bins = calibrate_and_evaluate(probabilities, roles, episodes, protocol, folder, dict(split_id="synthetic", method=method, seed=40700))
            assert rows and predictions and bins
            assert all(r["n"] == 3 for r in rows if r["subset"] == "all")
            if method == "GRU":
                resumed, _, resumed_meta = train_fit(method, 40700, x, lengths, audit, roles, episodes, pairs, protocol, folder, device, permit)
                np.testing.assert_allclose(probabilities, resumed, atol=1e-7, rtol=0)
                assert resumed_meta["resumed_from_epoch"] == 2
            results.append(dict(method=method, status="SYNTHETIC_FIT_CHECKPOINT_CALIBRATION_PASS", parameters=meta["parameters"]))
            print(method, "PASS", flush=True)
        write_json(root/"implementation_test_results.json", dict(status="PASSED", scope="SYNTHETIC_ONLY_NOT_G_DATA_NOT_BENCHMARK", device=device, results=results))
    window = dict(allow_training=False, isaac_stopped_confirmed=True, expires_utc=dt.datetime.now(dt.timezone.utc).isoformat(), scope="SYNTHETIC_TEST_ENDED")
    write_json(permit, window)
    print("SYNTHETIC_IMPLEMENTATION_TEST_COMPLETE", root, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    main(parser.parse_args().device)
