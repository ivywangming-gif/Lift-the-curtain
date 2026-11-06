"""Freeze, resume and evaluate Round G learning without touching Round F.

freeze reads only protocol, input data and split/pair eligibility; it does not fit
or inspect model test predictions. run requires an explicit expiring simulation-
free training window. Every completed fit is durable; interruptions are resumable.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Limit CPU BLAS work as well as Torch; GPU training still uses CPU two threads.
for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "2"

import numpy as np

from learning_core import (ROOT, ACTION_CONTRACT, approved_pairs, apply_label_budget, frozen_label_budget, conformal_decisions,
    conformal_fit, load_dataset, metrics, prepare_features, read_json, select_thresholds,
    sha, split_indices, statistics_features, write_csv, write_json)

OUT = ROOT / "outputs/round4070_comprehensive_traversability"
DEFAULT_SEEDS = [40700, 40701, 40702, 40703, 40704]


def sources():
    return {p.name: sha(p) for p in [Path(__file__), *[Path(__file__).with_name(n) for n in
        ("learning_core.py","learning_models.py","resource_lock.py","engineering_rules.py","observation_randomization.py",
         "physics_history.py","collision_geometry.py","g1_action_contract_v2.py")]]}


def method_names():
    # Importing definitions creates no network or training job.
    from learning_models import METHODS, ENGINEERING
    return METHODS, ENGINEERING


def partial_heldout_tasks(split, episodes, full_tasks):
    """Preregistered one-class diagnostic; never changes full split eligibility.

Eligibility uses label support only, before any model predictions. Train core
must contain both labels. Calibration may be unsupported: fixed final-epoch
models still report raw forced-.5 outcomes, without fitted risk/calibration.
"""
    roles = split_indices(split, episodes)
    test_classes = {int(episodes[i]["y"]) for i in roles["test"] if int(episodes[i]["y"]) >= 0 and not episodes[i].get("is_control")}
    if len(test_classes) != 1:
        return []
    train_classes = {int(episodes[i]["y"]) for i in roles["train"] if int(episodes[i]["y"]) >= 0 and not episodes[i].get("is_control")}
    tasks = []
    for task in full_tasks:
        if task["split_id"] != split["split_id"] or task.get("label_fraction", 1.) != 1.:
            continue
        reasons = [r for r in task["reasons"] if r not in {"TEST_CORE_BOTH_LABELS_NOT_SUPPORTED", "TEST_BOTH_CLASSES_NOT_SUPPORTED"}]
        if train_classes != {0, 1}:
            reasons.append("PARTIAL_DIAGNOSTIC_REQUIRES_BOTH_CORE_TRAIN_CLASSES")
        tasks.append({**task, "task_id":task["task_id"]+"__partial_single_class",
                      "evaluation_scope":"PARTIAL_HELDOUT_SINGLE_CLASS", "heldout_supported_class":next(iter(test_classes)),
                      "full_split_status":split.get("status"), "reasons":sorted(set(reasons)),
                      "status":"BLOCKED_WITH_REASON" if reasons else "PLANNED"})
    return tasks


def window_status(path):
    if not path or not Path(path).exists():
        return False, "MISSING_EXPLICIT_TRAINING_WINDOW"
    window = read_json(path)
    if not window.get("allow_training") or not window.get("isaac_stopped_confirmed"):
        return False, "SIMULATION_FREE_WINDOW_NOT_CONFIRMED"
    expiry = dt.datetime.fromisoformat(window["expires_utc"].replace("Z", "+00:00"))
    if expiry.tzinfo is None or dt.datetime.now(dt.timezone.utc) >= expiry:
        return False, "TRAINING_WINDOW_EXPIRED"
    # Only process arguments identify simulation; the Python environment path may
    # legitimately contain 'isaaclab' for both learning and simulation.
    if importlib.util.find_spec("psutil"):
        import psutil
        simulation_scripts = {"run_episode.py", "physics_p1_asset.py", "run_physics_scene.py", "physics_scene.py", "physics_gate.py", "g1_trial.py"}
        for process in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                command = process.info["cmdline"] or []
                if process.info["pid"] == os.getpid():
                    continue
                if (process.info["name"] or "").lower() in ("kit.exe", "isaac-sim.exe") or any(Path(x).name.lower() in simulation_scripts for x in command[1:]):
                    return False, f"SIMULATION_PROCESS_ACTIVE:{process.info['pid']}"
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    return True, "AUTHORIZED_WINDOW"


def require_window(path):
    okay, reason = window_status(path)
    if not okay:
        raise InterruptedError(reason)


def freeze(args):
    from observation_randomization import CONFIG as OBSERVATION_DR
    methods, engineering = method_names()
    if args.methods:
        requested = args.methods.split(",")
        if set(requested)-set(methods+engineering):
            raise ValueError("Unknown method")
        methods = [m for m in requested if m in methods]
        engineering = [m for m in requested if m in engineering]
    if len(args.seeds) < 5 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("At least five distinct fitted seeds required")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Freeze into a new directory; use run to resume existing protocol")
    manifest, episodes, names, raw, times, hashes = load_dataset(args.manifest, args.legacy)
    split_doc = read_json(args.splits)
    pair_doc = read_json(args.pairs) if args.pairs and args.pairs.exists() else {"by_split": {}}
    split_list = split_doc["splits"]
    tasks, label_budgets = [], {}
    for split in split_list:
        sid = split["split_id"]
        roles = split_indices(split, episodes)
        train = roles["train"]
        labelled = [i for i in train if int(episodes[i]["y"]) >= 0]
        labels = {int(episodes[i]["y"]) for i in labelled}
        core_test = [i for i in roles["test"] if int(episodes[i]["y"]) >= 0 and not episodes[i].get("is_control")]
        test_labels = {int(episodes[i]["y"]) for i in core_test}
        split_reason = []
        if labels != {0, 1}:
            split_reason.append("TRAIN_REQUIRES_RELIABLE_BOTH_CLASSES")
        if not core_test:
            split_reason.append("NO_RELIABLE_CORE_TEST_LABELS")
        if test_labels != {0, 1}:
            split_reason.append("TEST_BOTH_CLASSES_NOT_SUPPORTED")
        if str(split.get("status", "")).upper().startswith(("BLOCKED", "UNAVAILABLE", "NOT_EVALUABLE", "PENDING")):
            split_reason.extend(split.get("reasons", ["SPLIT_DECLARED_UNAVAILABLE"]))
        valid_pairs = approved_pairs(pair_doc, sid, train, episodes)
        kinds = {p[2] for p in valid_pairs}
        label_budgets[sid] = {str(f): frozen_label_budget(episodes, train, sid, f) for f in (.25, .5, 1.)}
        for method in engineering+methods:
            reason = list(split_reason)
            if method in engineering:
                reason = [r for r in reason if r != "TRAIN_REQUIRES_RELIABLE_BOTH_CLASSES"]
            if method in ("ResidualGRU", "OldM2") and not any(episodes[i].get("is_control") and int(episodes[i]["y"]) == 1 for i in train):
                reason.append("NO_TRAIN_EMPTY_CONTROL_FOR_NORMAL_RESPONSE")
            if method in ("GRUMatched", "MatchedNoHardNegative", "MatchedNoCrossPositive"):
                # All three compare on the same admissible pair dataset. An
                # ablation never hides that the complete mechanism is unevaluable.
                if kinds != {"hard_negative", "cross_configuration_positive"}:
                    reason.append("MATCHED_MECHANISM_REQUIRES_REAL_HARD_NEGATIVE_AND_CROSS_CONFIG_POSITIVE")
            if method == "GRUSupCon" and not all(sum(int(episodes[i]["y"]) == c for i in labelled) >= 2 for c in (0, 1)):
                reason.append("SUPCON_REQUIRES_TWO_REAL_TRAIN_EXAMPLES_PER_CLASS")
            if method in ("GRUResponseAux", "GRUPretrainFinetune") and not any(int(episodes[i]["y"]) < 0 for i in train):
                reason.append("NO_TRAIN_UNKNOWN_FOR_UNLABELLED_RESPONSE_COMPARISON")
            for seed in ([None] if method in engineering else args.seeds):
                task_id = f"{sid}__{method}__{seed if seed is not None else 'deterministic'}"
                tasks.append(dict(task_id=task_id, split_id=sid, method=method, seed=seed,
                                  display_method=method, label_fraction=1.,
                                  status="BLOCKED_WITH_REASON" if reason else "PLANNED", reasons=reason,
                                  train_n=len(train), labelled_train_n=len(labelled), core_test_n=len(core_test)))
        for fraction in (.25, .5):
            budget = label_budgets[sid][str(fraction)]
            selected_labels = set(budget["original_train_labels_audit_only"][tid] for tid in budget["selected_train_label_ids"])
            reduced = apply_label_budget(episodes, train, budget)
            for method in ("GRU", "GRUResponseAux", "GRUPretrainFinetune"):
                if method not in methods:
                    continue
                reason = list(split_reason)
                if selected_labels != {0, 1}:
                    reason.append("LABEL_BUDGET_REQUIRES_RELIABLE_BOTH_TRAIN_CLASSES")
                if not all(budget["selected_core_class_counts"][str(c)] for c in (0, 1)):
                    reason.append("LABEL_BUDGET_CORE_BOTH_CLASSES_NOT_SUPPORTED")
                if method != "GRU" and not any(int(reduced[i]["y"]) < 0 for i in train):
                    reason.append("NO_TRAIN_UNKNOWN_FOR_UNLABELLED_RESPONSE_COMPARISON")
                display = f"{method}__labels{int(fraction*100):03d}"
                for seed in args.seeds:
                    tasks.append(dict(task_id=f"{sid}__{display}__{seed}", split_id=sid, method=method,
                                      display_method=display, label_fraction=fraction, seed=seed,
                                      status="BLOCKED_WITH_REASON" if reason else "PLANNED", reasons=reason,
                                      train_n=len(train), labelled_train_n=budget["selected_labelled_train_n"], core_test_n=len(core_test)))
    for task in tasks:
        task["evaluation_scope"] = "FULL_SPLIT_BENCHMARK"
    full_tasks = list(tasks)
    if not args.legacy:
        for split in split_list:
            tasks.extend(partial_heldout_tasks(split, episodes, full_tasks))
    protocol = dict(schema="round4070_g.learning.freeze.v1", created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    scope="LEGACY_F_CONTRACT_DIAGNOSTIC_NOT_G_BENCHMARK" if args.legacy else "G_ACTION_CONTRACT_BENCHMARK",
                    legacy=args.legacy, action_contract=ACTION_CONTRACT, seeds=args.seeds,
                    primary_protocol=manifest.get("primary_protocol"),label_validity_version=manifest.get("label_validity_version"),
                    methods=engineering+methods, epochs=80, pretrain_epochs=40, batch_size=32,
                    hidden=64, max_steps=128, cutoff_s=2., optimizer="AdamW", lr=.001, weight_decay=.01,
                    gradient_clip=5., contrastive_weight=.1, response_weight=.1, supcon_temperature=.1,
                    matched_negative_cosine_margin=.2, logistic_C=1., logistic_max_iter=200,
                    logistic_tolerance_grad=1e-8, selection="fixed final epoch; no test-driven tuning",
                    normalization="training valid rows only, including train UNKNOWN; never calibration/test histories",
                    response_pretraining="only training split; target next observed q/qdot; no hidden physical targets",
                    calibration="independent calibration split only; fixed temperature objective and empirical target risk grid",
                    partial_heldout_policy=dict(scope="PARTIAL_HELDOUT_SINGLE_CLASS",label_fraction=1.,
                        trigger="exactly one reliable core heldout class; class support only, no fitted predictions",
                        training="both core train classes; same frozen final epoch; retain method-specific validity requirements",
                        calibration="unsupported calibration disables fitted temperature/risk/conformal only; raw forced-.5 remains evaluable",
                        interpretation="class-conditional counts/errors only; BA/AUROC/two-class risk NA; original full split stays NOT_EVALUABLE; no cross-split checkpoint reuse"),
                    confirmation_policy="reserved fourth role; separate endpoint using each frozen model and main calibration exactly as fitted; no selection or refitting",
                    risk_grid=[0., .01, .025, .05, .1, .2], conformal_alpha=.1,
                    observation_dr=OBSERVATION_DR, observation_test_seed=40706000,
                    observation_protocol="all standard models clean training/clean test; GRUObsDR only train raw observations corrupted; all fitted models additionally clean-calibrated/noisy-test diagnostic; same original labels and split IDs",
                    label_efficiency=dict(fractions=[.25, .5, 1.], methods=["GRU", "GRUResponseAux", "GRUPretrainFinetune"],
                                          subsets=label_budgets, full_budget="reuse original 100% tasks; never refit duplicate tasks",
                                          note="five seeds and three methods share one frozen subset per split/fraction; withheld training labels become UNKNOWN; test/calibration unchanged"),
                    manifest=str(args.manifest.resolve()), manifest_sha256=sha(args.manifest), source_npz_sha256=hashes,
                    splits=str(args.splits.resolve()), splits_sha256=sha(args.splits), split_definitions=split_list,
                    pairs=str(args.pairs.resolve()) if args.pairs and args.pairs.exists() else None,
                    pairs_sha256=sha(args.pairs) if args.pairs and args.pairs.exists() else None,
                    approved_pairs=pair_doc, code_sha256=sources(), feature_names=names,
                    original_F_untouched=True, statistical_unit="physical episode/configuration, not training seed")
    write_json(args.output / "frozen_protocol.json", protocol)
    write_json(args.output / "task_queue.json", dict(status="PENDING", tasks=tasks))
    write_csv(args.output / "TASK_STATUS.csv", [{**t, "reasons": "|".join(t["reasons"])} for t in tasks],
              fields=["task_id", "split_id", "method", "display_method", "label_fraction", "seed", "status", "reasons", "train_n", "labelled_train_n", "core_test_n",
                      "evaluation_scope", "heldout_supported_class", "full_split_status"])
    print(json.dumps(dict(status="FROZEN_NO_TRAINING", tasks=len(tasks), planned=sum(t["status"] == "PLANNED" for t in tasks), blocked=sum(t["status"] != "PLANNED" for t in tasks), output=str(args.output))), flush=True)


def engineer_probability(episode, method):
    if method == "AlwaysPassable":
        return 1.
    if method == "AlwaysBlocked":
        return 0.
    # Dataset exporter computes the frozen engineering rules from the exact same
    # legal observation window, with real-dt stall integration. No truth fallback.
    decisions = episode.get("engineering_decisions_time_corrected", {})
    key = {"E0": "E0", "E1TimeCorrected": "E1", "E2TimeCorrected": "E2"}[method]
    value = decisions.get(key)
    if value is None:
        decisions = episode.get("engineering_decisions", {})
        value = decisions.get(key+"_time_corrected")
    if value not in ("PASSABLE", "BLOCKED", "UNCERTAIN"):
        raise ValueError(f"Missing time-corrected {key} decision; do not substitute the legacy erroneous rule")
    return {"PASSABLE": 1., "BLOCKED": 0., "UNCERTAIN": .5}[value]


def train_fit(method, seed, x, lengths, audit, roles, episodes, pairs, protocol, folder, device, window):
    import torch
    from torch import nn
    from learning_models import (CONTRASTIVE, MATCHED, RESPONSE, GaussianAR1, SmallModel,
                                 VectorStumpBoost, mask_for, matched_pair_loss, supcon_loss)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    train_all = roles["train"]
    y = np.asarray([int(e["y"]) for e in episodes], dtype=np.float32)
    train = [i for i in train_all if y[i] >= 0]
    start = time.perf_counter()
    curve, convergence = [], {}
    if method in ("Logistic", "DurationOnlyLogistic", "GradientBoosting", "GaussianAR1"):
        if method == "GaussianAR1":
            model = GaussianAR1().fit(x[train], lengths[train], y[train])
            probabilities = model.predict(x, lengths)
            payload = model.export()
            parameters = sum(np.asarray(v).size for m in model.models for v in m)
        else:
            features = statistics_features(x, lengths).astype(np.float64)
            if method == "DurationOnlyLogistic":
                features = np.column_stack((audit["acquired_sample_counts"], audit["observed_durations_s"])).astype(np.float64)
                timing_mean = features[train_all].mean(0)
                timing_scale = np.maximum(features[train_all].std(0), 1e-3)
                features = np.clip((features-timing_mean)/timing_scale, -10, 10)
                audit = dict(audit, timing_only_input=["acquired_sample_count", "observed_duration_s"], timing_mean=timing_mean.tolist(), timing_scale=timing_scale.tolist())
            if method == "GradientBoosting":
                model = VectorStumpBoost().fit(features[train], y[train])
                probabilities = model.predict(features)
                payload, curve = model.export(), model.curve
                parameters = len(model.trees)*4+1
            else:
                model = nn.Linear(features.shape[1], 1, dtype=torch.float64)
                z = torch.from_numpy(features)
                yy = torch.from_numpy(y[train].astype(np.float64))
                optimizer = torch.optim.LBFGS(model.parameters(), max_iter=protocol["logistic_max_iter"], max_eval=250, tolerance_grad=protocol["logistic_tolerance_grad"], tolerance_change=1e-15, line_search_fn="strong_wolfe")
                def closure():
                    optimizer.zero_grad()
                    loss = nn.functional.binary_cross_entropy_with_logits(model(z[train]).squeeze(-1), yy)+model.weight.square().sum()/(2*len(train))
                    loss.backward()
                    curve.append(dict(epoch=len(curve), train_loss=float(loss.detach()), point_kind="LBFGS_LINE_SEARCH_EVALUATION"))
                    return loss
                optimizer.step(closure)
                final = closure()
                grad = float(torch.cat([p.grad.flatten() for p in model.parameters()]).abs().max())
                state = optimizer.state[model.weight]
                convergence = dict(objective=float(final.detach()), gradient_inf=grad, gradient_converged=grad <= protocol["logistic_tolerance_grad"], iterations=state.get("n_iter", 0), function_evaluations=state.get("func_evals", 0))
                with torch.no_grad():
                    probabilities = torch.sigmoid(model(z).squeeze(-1)).numpy()
                payload = dict(state_dict=model.state_dict(), convergence=convergence)
                parameters = sum(v.numel() for v in model.parameters())
        training_seconds = time.perf_counter()-start
        # Timing reports actual single-input inference, not acquisition latency.
        sample = train[:1]
        began = time.perf_counter()
        for _ in range(20):
            if method == "GaussianAR1":
                model.predict(x[sample], lengths[sample])
            elif method == "GradientBoosting":
                model.predict(features[sample])
            else:
                with torch.no_grad():
                    model(z[sample])
        inference_ms = (time.perf_counter()-began)*1000/20
        torch.save(dict(method=method, seed=seed, preprocessing=audit, model=payload, train_ids=[episodes[i]["trial_id"] for i in train], protocol=protocol), folder/"checkpoint.pt")
        return probabilities, curve, dict(parameters=parameters, training_seconds=training_seconds, inference_ms=inference_ms, deterministic_algorithm=method in ("GradientBoosting", "GaussianAR1"), **convergence)

    xx, ll = torch.from_numpy(x).to(device), torch.from_numpy(lengths).to(device)
    yy = torch.from_numpy(y).to(device)
    # The param-matched target follows the exact input augmentation formula.
    if method == "GRUParamMatched":
        from learning_core import temporal_extras
        dummy = np.zeros((2, len(audit["feature_names"])), dtype=np.float32)
        extra, _ = temporal_extras(dummy, np.asarray([0., .02]), audit["feature_names"])
        target = 3*64*(x.shape[2]+extra.shape[1]+audit["removed_command_progress_channels_count"]+64+2)+65
    else:
        target = None
    response_indices = [i for i, n in enumerate(audit["feature_names"]) if n.startswith(("q/", "qdot/"))]
    model = SmallModel(method, x.shape[2], protocol["max_steps"], protocol["hidden"], audit["arm"], audit["body"], response_indices, target).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["lr"], weight_decay=protocol["weight_decay"])
    total_epochs = protocol["epochs"] + (protocol["pretrain_epochs"] if method == "GRUPretrainFinetune" else 0)
    beginning = 0
    prior_training_seconds = 0.
    partial = folder/"partial.pt"
    if partial.exists():
        previous = torch.load(partial, map_location=device, weights_only=False)
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        beginning, curve = previous["next_epoch"], previous["curve"]
        prior_training_seconds = previous.get("cumulative_training_seconds", 0.)
        torch.set_rng_state(previous["cpu_rng"].cpu())
        if device.startswith("cuda") and previous.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(previous["cuda_rng"])
    for epoch in range(beginning, total_epochs):
        require_window(window)
        pretrain = method == "GRUPretrainFinetune" and epoch < protocol["pretrain_epochs"]
        use_response = method == "GRUResponseAux" or pretrain
        if method == "GRUPretrainFinetune" and epoch == protocol["pretrain_epochs"]:
            # Explicit staged optimization reset; not a test-selected checkpoint.
            optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["lr"], weight_decay=protocol["weight_decay"])
        rng = np.random.default_rng(seed + epoch*7919)
        if method in MATCHED or method == "GRUSupCon":
            batch_indices = [list(train)]  # all real pairs present; no synthetic pair duplication
        else:
            order = rng.permutation(train_all if use_response else train)
            batch_indices = [order[k:k+protocol["batch_size"]].tolist() for k in range(0, len(order), protocol["batch_size"])]
        model.train()
        sums = dict(total_loss=0., supervised_loss=0., auxiliary_loss=0., contrastive_loss=0., batches=0, contrastive_anchor_n=0, matched_positive_n=0, matched_negative_n=0)
        for indices in batch_indices:
            optimizer.zero_grad()
            bx = xx[indices]
            if method == "GRUShuffled":
                bx = bx.clone()
                # One frozen permutation per episode/seed, also used at inference.
                for j, i in enumerate(indices):
                    perm = np.random.default_rng(seed+i*101).permutation(lengths[i])
                    bx[j, :lengths[i]] = bx[j, perm]
            logits, projected, response = model(bx, ll[indices], return_aux=True)
            labelled = yy[indices] >= 0
            zero = logits.sum()*0
            supervised = nn.functional.binary_cross_entropy_with_logits(logits[labelled], yy[indices][labelled]) if labelled.any() and not pretrain else zero
            auxiliary, contrastive = zero, zero
            if use_response:
                valid = mask_for(ll[indices]-1, bx.shape[1]-1)
                squared = (response[:, :-1]-bx[:, 1:, response_indices]).square().mean(-1)
                auxiliary = squared[valid].mean() if valid.any() else zero
            if method == "GRUSupCon":
                contrastive, anchors = supcon_loss(projected, yy[indices], protocol["supcon_temperature"])
                sums["contrastive_anchor_n"] += anchors
            elif method in MATCHED:
                local = {global_i: local_i for local_i, global_i in enumerate(indices)}
                local_pairs = [(local[a], local[b], kind) for a, b, kind in pairs]
                contrastive, positive_n, negative_n = matched_pair_loss(projected, local_pairs, method)
                sums["matched_positive_n"] += positive_n
                sums["matched_negative_n"] += negative_n
            loss = supervised + (1. if pretrain else protocol["response_weight"])*auxiliary + protocol["contrastive_weight"]*contrastive
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), protocol["gradient_clip"])
            optimizer.step()
            for key, value in (("total_loss", loss), ("supervised_loss", supervised), ("auxiliary_loss", auxiliary), ("contrastive_loss", contrastive)):
                sums[key] += float(value.detach())
            sums["batches"] += 1
        curve.append(dict(epoch=epoch, phase="response_pretrain" if pretrain else "supervised_or_joint", **{k: v/sums["batches"] if k.endswith("loss") else v for k, v in sums.items()}))
        temporary = folder/"partial.tmp.pt"
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), next_epoch=epoch+1, curve=curve,
                        cumulative_training_seconds=prior_training_seconds+time.perf_counter()-start,
                        cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.startswith("cuda") else None), temporary)
        temporary.replace(partial)
    training_seconds = prior_training_seconds+time.perf_counter()-start
    model.eval()
    prediction_x = xx.clone() if method == "GRUShuffled" else xx
    if method == "GRUShuffled":
        for i in range(len(x)):
            permutation = np.random.default_rng(seed+i*101).permutation(lengths[i])
            prediction_x[i, :lengths[i]] = xx[i, permutation]
    with torch.no_grad():
        probabilities = np.concatenate([torch.sigmoid(model(prediction_x[k:k+32], ll[k:k+32])).cpu().numpy() for k in range(0, len(x), 32)])
        for _ in range(5):
            model(prediction_x[:1], ll[:1])
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        began = time.perf_counter()
        for _ in range(30):
            model(prediction_x[:1], ll[:1])
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        inference_ms = (time.perf_counter()-began)*1000/30
    torch.save(dict(method=method, seed=seed, state_dict=model.state_dict(), preprocessing=audit, protocol=protocol,
                    hidden=model.hidden, train_ids=[episodes[i]["trial_id"] for i in train],
                    auxiliary_train_ids=[episodes[i]["trial_id"] for i in train_all] if method in RESPONSE else [],
                    response_indices=response_indices, parameter_match_target=target), folder/"checkpoint.pt")
    return probabilities, curve, dict(parameters=sum(v.numel() for v in model.parameters()), training_seconds=training_seconds,
                                     inference_ms=inference_ms, resumed_from_epoch=beginning, epochs=total_epochs,
                                     parameter_match_target=target, deterministic_algorithm=False)


def evaluation_subsets(indices, episodes):
    subsets = [("all", "all", indices)]
    for key in ("side", "group_id", "condition_family", "robot_variant"):
        def value(i):
            return str(episodes[i].get(key, episodes[i].get("probe_side", "unspecified") if key == "side" else "unspecified"))
        for category in sorted({value(i) for i in indices}):
            subsets.append((key, category, [i for i in indices if value(i) == category]))
    return subsets


def confirmation_status(roles, episodes):
    indices = roles.get("confirmation", [])
    known = [i for i in indices if int(episodes[i]["y"]) >= 0 and not episodes[i].get("is_control")]
    classes = {int(episodes[i]["y"]) for i in known}
    status = ("NOT_APPLICABLE_OR_NOT_YET_COLLECTED" if not indices else
              "NO_RELIABLE_CORE_CONFIRMATION_LABELS" if not known else
              "PARTIAL_CONFIRMATION_SINGLE_CLASS" if classes != {0, 1} else "EVALUATED_INDEPENDENT_CONFIRMATION")
    return dict(status=status, confirmation_ids=[episodes[i]["trial_id"] for i in indices],
                core_labelled_n=len(known), passable_n=sum(episodes[i]["y"] == 1 for i in known),
                blocked_n=sum(episodes[i]["y"] == 0 for i in known),
                scope="separate endpoint; frozen model and main calibration only; never model/threshold selection",
                observation_diagnostic="clean only; no confirmation observation-DR claim")


def calibrate_and_evaluate(probabilities, roles, episodes, protocol, folder, task, fitted_calibration=None):
    import torch
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray([int(e["y"]) for e in episodes])
    calibration = [i for i in roles["calibration"] if y[i] >= 0 and not episodes[i].get("is_control")]
    temperature = 1.
    temperature_status = "UNAVAILABLE_CALIBRATION_REQUIRES_BOTH_CLASSES"
    if fitted_calibration is not None:
        if fitted_calibration["calibration_ids"] != [episodes[i]["trial_id"] for i in calibration]:
            raise ValueError("Frozen calibrator IDs changed for independent evaluation")
        temperature = fitted_calibration["temperature"]
        temperature_status = fitted_calibration["temperature_status"]
    elif set(y[calibration].tolist()) == {0, 1}:
        logits = torch.tensor(np.log(np.clip(p[calibration], 1e-7, 1-1e-7)/(1-np.clip(p[calibration], 1e-7, 1-1e-7))), dtype=torch.float64)
        target = torch.tensor(y[calibration], dtype=torch.float64)
        log_temperature = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([log_temperature], max_iter=50, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits/torch.exp(log_temperature.clamp(-3, 3)), target)+.01*log_temperature.square().sum()
            loss.backward()
            return loss
        opt.step(closure)
        temperature = float(torch.exp(log_temperature.detach().clamp(-3, 3)))
        temperature_status = "CALIBRATION_ONLY_FIXED_REGULARIZED_TEMPERATURE"
    pp = 1/(1+np.exp(-np.log(np.clip(p, 1e-7, 1-1e-7)/(1-np.clip(p, 1e-7, 1-1e-7)))/temperature))
    threshold_fits = fitted_calibration["thresholds"] if fitted_calibration is not None else [select_thresholds(y[calibration], pp[calibration], risk) for risk in protocol["risk_grid"]]
    # The conformal scoring function must be frozen before calibration labels.
    # Temperature is fitted on this calibration set, so using pp here would
    # silently reuse these labels inside the score function. Use raw p instead.
    conformal = dict(fitted_calibration["conformal"]) if fitted_calibration is not None else conformal_fit(y[calibration], p[calibration], protocol["conformal_alpha"])
    conformal["score_source"] = "frozen_model_raw_probability; independent of calibration-fitted temperature"
    write_json(folder/"calibration.json", dict(calibration_ids=[episodes[i]["trial_id"] for i in calibration], temperature=temperature,
               temperature_status=temperature_status, thresholds=threshold_fits, conformal=conformal,
               calibration_operation="REUSED_FROZEN_MAIN_CALIBRATION" if fitted_calibration is not None else "FIT_ON_CALIBRATION_ONLY",
               guarantee="NONE: small selected physical configurations; exchangeability unverified; test never fits thresholds"))
    metrics_rows, prediction_rows, calibration_bins = [], [], []
    modes = [("forced_raw", p, .5, .5, None), ("fixed_reject_raw", p, .2, .8, None)]
    if temperature_status.startswith("CALIBRATION"):
        modes.append(("temperature_forced", pp, .5, .5, None))
    for risk, fit in zip(protocol["risk_grid"], threshold_fits):
        if fit["low"] is not None:
            modes.append((f"calibration_risk_{risk:g}", pp, fit["low"], fit["high"], None))
    if "q" in conformal:
        modes.append(("label_conditional_conformal", p, None, None, conformal_decisions(p, conformal)))
    test = [i for i in roles["test"] if y[i] >= 0 and not episodes[i].get("is_control")]
    subsets = evaluation_subsets(test, episodes)
    for mode, probabilities_used, low, high, decisions in modes:
        d = np.where(probabilities_used >= high, 1, np.where(probabilities_used < low, 0, -1)) if decisions is None else decisions
        for subset, value, indices in subsets:
            metrics_rows.append(dict(split_id=task["split_id"], method=task["method"], seed=task["seed"], mode=mode,
                                     subset=subset, subset_value=value, low=low, high=high,
                                     **metrics(y[indices], probabilities_used[indices], decisions=d[indices])))
        for i in roles["test"]:
            prediction_rows.append(dict(split_id=task["split_id"], method=task["method"], seed=task["seed"], mode=mode,
                                        trial_id=episodes[i]["trial_id"], group_id=episodes[i]["group_id"],
                                        side=episodes[i].get("probe_side", episodes[i].get("side")), y=int(y[i]),
                                        is_control=episodes[i].get("is_control", False), p=float(probabilities_used[i]), decision=int(d[i])))
    bin_sources = [("raw", p)]
    if temperature_status.startswith("CALIBRATION"):
        bin_sources.append(("temperature", pp))
    for which, probabilities_used in bin_sources:
        for b in range(10):
            ix = [i for i in test if min(int(probabilities_used[i]*10), 9) == b]
            calibration_bins.append(dict(split_id=task["split_id"], method=task["method"], seed=task["seed"], probability_kind=which,
                                         bin=b, n=len(ix), predicted_mean=float(probabilities_used[ix].mean()) if ix else None,
                                         observed_passable_rate=float(y[ix].mean()) if ix else None))
    return metrics_rows, prediction_rows, calibration_bins


def checkpoint_predict(path, x, lengths, current_audit, device):
    """Frozen inference on transformed inputs; no parameter or threshold fitting."""
    import torch
    from torch import nn
    from learning_models import GaussianAR1, SmallModel, VectorStumpBoost
    saved = torch.load(path, map_location=device, weights_only=False)
    method = saved["method"]
    pre = saved["preprocessing"]
    if method in ("Logistic", "DurationOnlyLogistic", "GradientBoosting"):
        features = statistics_features(x, lengths).astype(np.float64)
        if method == "DurationOnlyLogistic":
            features = np.column_stack((current_audit["acquired_sample_counts"], current_audit["observed_durations_s"])).astype(np.float64)
            features = np.clip((features-np.asarray(pre["timing_mean"]))/np.asarray(pre["timing_scale"]), -10, 10)
        if method == "GradientBoosting":
            model = VectorStumpBoost()
            model.bias, model.trees = saved["model"]["bias"], saved["model"]["trees"]
            return model.predict(features)
        model = nn.Linear(features.shape[1], 1, dtype=torch.float64)
        model.load_state_dict(saved["model"]["state_dict"])
        with torch.no_grad():
            return torch.sigmoid(model(torch.from_numpy(features)).squeeze(-1)).numpy()
    if method == "GaussianAR1":
        model = GaussianAR1()
        model.projection = np.asarray(saved["model"]["projection"])
        model.models = [[np.asarray(v) for v in m] for m in saved["model"]["class_models"]]
        return model.predict(x, lengths)
    model = SmallModel(method, x.shape[2], saved["protocol"]["max_steps"], saved["hidden"], pre["arm"], pre["body"], saved["response_indices"]).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    values = x.copy()
    if method == "GRUShuffled":
        for i in range(len(values)):
            values[i, :lengths[i]] = x[i, np.random.default_rng(saved["seed"]+i*101).permutation(lengths[i])]
    with torch.no_grad():
        return np.concatenate([torch.sigmoid(model(torch.from_numpy(values[k:k+32]).to(device), torch.from_numpy(lengths[k:k+32]).to(device))).cpu().numpy() for k in range(0, len(values), 32)])


def run(args):
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    protocol = read_json(args.output/"frozen_protocol.json")
    queue = read_json(args.output/"task_queue.json")
    if protocol["code_sha256"] != sources():
        raise ValueError("Code changed after protocol freeze; preserve this run and create a new audited snapshot")
    if sha(protocol["manifest"]) != protocol["manifest_sha256"] or sha(protocol["splits"]) != protocol["splits_sha256"]:
        raise ValueError("Dataset/splits changed after freeze")
    require_window(args.training_window)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested GPU unavailable; choose a new explicit --device cpu invocation")
    manifest, episodes, names, raw, times, hashes = load_dataset(protocol["manifest"], protocol["legacy"])
    if hashes != protocol["source_npz_sha256"]:
        raise ValueError("Source observations changed after freeze")
    split_map = {s["split_id"]: s for s in protocol["split_definitions"]}
    _, engineering = method_names()
    completed = 0
    cached_key, cached = None, None
    rule_cache = {}
    noise_scale_cache = {}
    for task in queue["tasks"]:
        if task["status"] in ("EVALUATED", "BLOCKED_WITH_REASON"):
            continue
        if args.max_fits is not None and completed >= args.max_fits:
            break
        okay, reason = window_status(args.training_window)
        if not okay:
            queue["last_pause_reason"] = reason
            break
        folder = args.output/"fits"/task["task_id"]
        folder.mkdir(parents=True, exist_ok=True)
        roles = split_indices(split_map[task["split_id"]], episodes)
        fraction = task.get("label_fraction", 1.)
        budget = protocol.get("label_efficiency", {}).get("subsets", {}).get(task["split_id"], {}).get(str(fraction))
        fit_episodes = apply_label_budget(episodes, roles["train"], budget) if budget else episodes
        evaluation_task = {**task, "method": task.get("display_method", task["method"])}
        task["status"] = "TRAINING"
        write_json(args.output/"task_queue.json", queue)
        try:
            if task["method"] in engineering:
                p = np.full(len(episodes), .5)
                evaluation_indices = roles["test"]+roles["confirmation"]
                if task["method"] in ("AlwaysPassable", "AlwaysBlocked"):
                    for i in evaluation_indices:
                        p[i] = engineer_probability(episodes[i], task["method"])
                elif protocol["legacy"]:
                    for i in evaluation_indices:
                        p[i] = engineer_probability(episodes[i], task["method"])
                else:
                    if task["split_id"] not in rule_cache:
                        from engineering_rules import evaluate_rules
                        rule_cache[task["split_id"]] = evaluate_rules(episodes, raw, times, names, roles["train"])
                    rules, reference_audit = rule_cache[task["split_id"]]
                    write_json(folder/"engineering_reference_audit.json", reference_audit)
                    key = {"E0": "E0", "E1TimeCorrected": "E1", "E2TimeCorrected": "E2"}[task["method"]]
                    for i in evaluation_indices:
                        p[i] = {"PASSABLE": 1., "BLOCKED": 0., "UNCERTAIN": .5}[rules[i][key]]
                decisions = np.where(p == 1., 1, np.where(p == 0., 0, -1))
                rows, predictions = [], []
                for endpoint in ("test", "confirmation"):
                    if endpoint == "confirmation" and not roles[endpoint]:
                        continue
                    mode = ("confirmation/" if endpoint == "confirmation" else "")+"native_engineering_three_state"
                    indices = [i for i in roles[endpoint] if int(episodes[i]["y"]) >= 0 and not episodes[i].get("is_control")]
                    for subset, value, selected in evaluation_subsets(indices, episodes):
                        rows.append(dict(split_id=task["split_id"], method=task["method"], seed=None, mode=mode, subset=subset, subset_value=value,
                                         **metrics([episodes[i]["y"] for i in selected], p[selected], decisions=decisions[selected])))
                    predictions.extend(dict(split_id=task["split_id"], method=task["method"], seed=None, mode=mode, trial_id=episodes[i]["trial_id"], group_id=episodes[i]["group_id"], side=episodes[i].get("probe_side", episodes[i].get("side")), y=int(episodes[i]["y"]), p=float(p[i]), decision=int(decisions[i]), is_control=episodes[i].get("is_control", False)) for i in roles[endpoint])
                for row in rows:
                    # Rule scores encode decisions, not calibrated probabilities.
                    row.update(brier=None, ece=None, auroc=None, auprc=None)
                curve, bins, fit_metadata = [], [], dict(parameters=0, training_seconds=0., inference_ms=None)
            else:
                from observation_randomization import fit_scale, transform_indices
                trial_ids = [e["trial_id"] for e in episodes]
                if task["split_id"] not in noise_scale_cache:
                    noise_scale_cache[task["split_id"]] = fit_scale(raw, names, roles["train"], trial_ids)
                noise_scale = noise_scale_cache[task["split_id"]]
                key = (task["split_id"], task["method"], task["seed"] if task["method"] == "GRUObsDR" else None)
                if key != cached_key:
                    cached = prepare_features(raw, times, names, episodes, roles["train"], task["method"], protocol["max_steps"], protocol["cutoff_s"])
                    if task["method"] == "GRUObsDR":
                        _, _, _, clean_audit = cached
                        augmented, augmented_time, transform_audit = transform_indices(raw, times, names, noise_scale, roles["train"], trial_ids, task["seed"]+40708000)
                        cached = prepare_features(augmented, augmented_time, names, episodes, roles["train"], task["method"], protocol["max_steps"], protocol["cutoff_s"], normalization_override=clean_audit)
                        cached[3]["observation_training_augmentation"] = transform_audit
                        cached[3]["normalization_source"] = "same clean training rows as baseline GRU; augmentation never fits a second scaler"
                    cached_key = key
                x, lengths, real_time, audit = cached
                pairs = approved_pairs(protocol["approved_pairs"], task["split_id"], roles["train"], episodes)
                p, curve, fit_metadata = train_fit(task["method"], task["seed"], x, lengths, audit, roles, fit_episodes, pairs, protocol, folder, args.device, args.training_window)
                task["status"] = "TRAINED"
                write_json(args.output/"task_queue.json", queue)
                rows, predictions, bins = calibrate_and_evaluate(p, roles, fit_episodes, protocol, folder, evaluation_task)
                if roles["confirmation"]:
                    confirmation_folder = folder/"confirmation"
                    confirmation_folder.mkdir(exist_ok=True)
                    confirmation_roles = {**roles, "test": roles["confirmation"]}
                    cr, cp, cb = calibrate_and_evaluate(p, confirmation_roles, fit_episodes, protocol, confirmation_folder,
                                                      evaluation_task, fitted_calibration=read_json(folder/"calibration.json"))
                    for record in cr+cp:
                        record["mode"] = "confirmation/"+record["mode"]
                    for record in cb:
                        record["probability_kind"] = "confirmation/"+record["probability_kind"]
                    rows.extend(cr); predictions.extend(cp); bins.extend(cb)
                if budget:
                    write_json(folder/"label_budget_audit.json", budget)
                fit_metadata["inference_scope"] = "model_forward_only_excludes_acquisition_preprocessing_residual_and_IO"
                # A shared heldout sensor realization tests robustness without
                # retraining, changing labels, or fitting noisy-test thresholds.
                noisy_folder = folder/"noisy_test_diagnostic"
                noisy_folder.mkdir(exist_ok=True)
                try:
                    noisy_raw, noisy_time, noise_audit = transform_indices(raw, times, names, noise_scale, roles["test"], trial_ids, protocol["observation_test_seed"])
                    if task["method"] in ("GRUPrefix05", "GRUPrefix10"):
                        duration = .5 if task["method"] == "GRUPrefix05" else 1.
                        for record, i in zip(noise_audit["episodes"], roles["test"]):
                            valid = np.asarray(record["receipt_true_times"]) <= duration+1e-9
                            noisy_raw[i], noisy_time[i] = noisy_raw[i][valid], noisy_time[i][valid]
                        noise_audit["prefix_filter"] = "true receipt cutoff before reported jitter cutoff; common full-window realization"
                    nx, nl, _, noisy_pre = prepare_features(noisy_raw, noisy_time, names, episodes, roles["train"], task["method"], protocol["max_steps"], protocol["cutoff_s"], normalization_override=audit)
                    noisy_p = checkpoint_predict(folder/"checkpoint.pt", nx, nl, noisy_pre, args.device)
                    hybrid_p = np.asarray(p).copy()
                    hybrid_p[roles["test"]] = noisy_p[roles["test"]]
                    nr, npred, nb = calibrate_and_evaluate(hybrid_p, roles, fit_episodes, protocol, noisy_folder, evaluation_task,
                                                         fitted_calibration=read_json(folder/"calibration.json"))
                    for r in nr+npred:
                        r["mode"] = "noisy_test/"+r["mode"]
                    for r in nb:
                        r["probability_kind"] = "noisy_test/"+r["probability_kind"]
                    rows.extend(nr); predictions.extend(npred); bins.extend(nb)
                    write_json(noisy_folder/"transformation_audit.json", noise_audit)
                    write_json(noisy_folder/"status.json", dict(status="EVALUATED_SENSOR_SENSITIVITY", labels_unchanged=True, split_ids_unchanged=True,
                               calibration="identical clean calibration probabilities; no noisy holdout calibration", independent_new_episodes=0))
                except ValueError as noise_error:
                    write_json(noisy_folder/"status.json", dict(status="UNAVAILABLE_WITH_REASON", reason=str(noise_error), clean_evaluation_preserved=True))
            for record in rows+predictions+bins:
                endpoint = "confirmation" if record.get("mode", record.get("probability_kind", "")).startswith("confirmation/") else "main_test"
                record.update(base_method=task["method"], label_fraction=fraction, supervised_train_n=task["labelled_train_n"], evaluation_endpoint=endpoint,
                              evaluation_scope=task.get("evaluation_scope", "FULL_SPLIT_BENCHMARK"),
                              heldout_supported_class=task.get("heldout_supported_class"))
            if task.get("evaluation_scope") == "PARTIAL_HELDOUT_SINGLE_CLASS":
                for row in rows:
                    row.update(class_conditional_selective_error=row.get("selective_risk"),selective_risk=None,
                               balanced_accuracy=None,auroc=None,auprc=None,
                               metric_scope="single-class heldout descriptive counts and supported class-conditional rates only; no complete OOD comparison")
            write_json(folder/"confirmation_status.json", confirmation_status(roles, episodes))
            write_csv(folder/"metrics.csv", [{**r, **fit_metadata, "scope": protocol["scope"]} for r in rows])
            write_csv(folder/"predictions.csv", predictions)
            if curve:
                write_csv(folder/"training_curves.csv", curve)
            if bins:
                write_csv(folder/"calibration_bins.csv", bins)
            write_json(folder/"completion.json", dict(status="EVALUATED", task=task["task_id"], metadata=fit_metadata,
                       predictions_sha256=sha(folder/"predictions.csv"), checkpoint_sha256=sha(folder/"checkpoint.pt") if (folder/"checkpoint.pt").exists() else None))
            task.update(status="EVALUATED", reasons=[], completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
            completed += 1
            print(json.dumps(dict(task=task["task_id"], status="EVALUATED", **fit_metadata)), flush=True)
        except InterruptedError as error:
            task.update(status="PLANNED", reasons=[str(error)])
            queue["last_pause_reason"] = str(error)
            write_json(args.output/"task_queue.json", queue)
            break
        except Exception as error:
            task.update(status="FAILED_WITH_REASON", reasons=[f"{type(error).__name__}: {error}"])
            write_json(folder/"failure.json", dict(reason=task["reasons"], actual_exception=True))
            print(json.dumps(dict(task=task["task_id"], status=task["status"], reasons=task["reasons"])), flush=True)
        write_json(args.output/"task_queue.json", queue)
    counts = {s: sum(t["status"] == s for t in queue["tasks"]) for s in sorted({t["status"] for t in queue["tasks"]})}
    queue["status"] = "ALL_PLANNED_FITS_EVALUATED" if all(t["status"] == "EVALUATED" for t in queue["tasks"]) else "INCOMPLETE"
    queue["counts"] = counts
    write_json(args.output/"task_queue.json", queue)
    write_csv(args.output/"TASK_STATUS.csv", [{**t, "reasons": "|".join(t["reasons"])} for t in queue["tasks"]])
    print(json.dumps(dict(status=queue["status"], counts=counts, scope=protocol["scope"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--manifest", type=Path, default=OUT/"dataset/manifest.json")
    f.add_argument("--splits", type=Path, default=OUT/"dataset/split_manifest.json")
    f.add_argument("--pairs", type=Path, default=OUT/"dataset/matched_pairs.json")
    f.add_argument("--output", type=Path, required=True)
    f.add_argument("--methods", default="")
    f.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    f.add_argument("--legacy", action="store_true", help="Explicit F-contract diagnostic; never G benchmark completion")
    r = sub.add_parser("run")
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--training-window", type=Path, required=True)
    r.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    r.add_argument("--max-fits", type=int)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args)
    else:
        from resource_lock import resource_lock
        require_window(args.training_window)
        with resource_lock("learning", str(args.output.resolve())):
            require_window(args.training_window)
            run(args)


if __name__ == "__main__":
    main()
