"""Posthoc convex-logistic optimization/regularization sensitivity audit.

This is a separate exploratory audit, not a replacement for the original
17-method Phase A comparison. Run only after all Isaac processes have ended:

  python scripts/round4070_f/converged_logistic.py

The 165 channels, 48-point past-hold resampling, train-group normalization,
clipping and mean/std/min/max features match the original Logistic inputs.
The objective is fixed before this audit: mean BCE + ||w||^2 / (2*n_train),
equivalent to standard L2 logistic C=1; the intercept is unregularized.
This changes both optimizer and regularization relative to 60-step AdamW,
so any performance change cannot be attributed to optimization alone.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from train_models import metrics, resample


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/round4070_learning_traversability"
REPORT = ROOT / "research/round4070_learning_traversability"
METHOD = "Logistic_LBFGS_C1_posthoc"
SEEDS = (4070, 4071, 4072)
FOLD_NAMES = ("family_fold_0", "family_fold_1")
MAX_ITER = 200
MAX_EVAL = 250
TOL_GRAD = 1e-8
TOL_CHANGE = 1e-15
AUDIT_NOTE = "POSTHOC_OPTIMIZATION_AND_REGULARIZATION_SENSITIVITY_NOT_REPLACEMENT"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty results table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader()
        writer.writerows(rows)


def prepare(manifest_path: Path, original_run: Path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    frozen_path = original_run / "frozen_config.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8-sig"))
    if sha(manifest_path) != frozen["manifest_sha256"]:
        raise ValueError("Manifest differs from phase_a_valid; do not silently change the audit dataset")
    if frozen["sequence_length"] != 48 or frozen["seeds"] != list(SEEDS):
        raise ValueError("Unexpected original protocol")
    episodes = manifest["episodes"]
    by_id = {e["trial_id"]: i for i, e in enumerate(episodes)}
    if len(by_id) != len(episodes) or len(episodes) != 104:
        raise ValueError("This audit requires the original 104 unique E episodes")
    raw, names, data_hashes = [], None, {}
    for e in episodes:
        path = Path(e["npz_path"])
        path = path if path.is_absolute() else ROOT / path
        with np.load(path, allow_pickle=False) as data:
            these_names = data["feature_names"].astype(str).tolist()
            if names is None:
                names = these_names
            elif names != these_names:
                raise ValueError("Episode sensor schema mismatch")
            seq = data["seq"].astype(np.float32)
            timestamps = data["time"]
            if not np.isfinite(seq).all() or not np.all(np.diff(timestamps) > 0):
                raise ValueError("Nonfinite sequence or nonmonotone timestamps")
            raw.append(resample(seq, n=48, times=timestamps))
        data_hashes[e["trial_id"]] = sha(path)
    keep = [i for i, name in enumerate(names) if name != "elapsed/s"]
    names = [names[i] for i in keep]
    if len(names) != 165:
        raise ValueError("Expected the original 165 legal channels after elapsed removal")
    sequences = np.stack(raw)[:, :, keep]
    labels = np.asarray([int(e["y"]) for e in episodes], dtype=np.float64)
    folds = []
    tested = []
    for name in FOLD_NAMES:
        fold = next(f for f in frozen["folds"] if f["name"] == name)
        indices = {role: [by_id[trial] for trial in fold[role]] for role in ("train", "validation", "test")}
        train_all, validation, test = (indices[k] for k in ("train", "validation", "test"))
        train = [i for i in train_all if labels[i] >= 0]
        groups = {role: {episodes[i]["group_id"] for i in values} for role, values in indices.items()}
        if groups["train"] & (groups["validation"] | groups["test"]) or groups["validation"] & groups["test"]:
            raise ValueError("Physical groups overlap across split roles")
        if set(labels[train]) != {0.0, 1.0}:
            raise ValueError("Convex binary fit requires both training labels")
        if any(labels[i] < 0 or episodes[i]["is_control"] for i in test):
            raise ValueError("Test must contain only reliable non-control labels")
        tested.extend(test)
        # Keep NumPy float32 normalization identical to train_models.py.
        mean = sequences[train_all].mean((0, 1))
        scale = np.maximum(sequences[train_all].std((0, 1)), 1e-3)
        x = np.clip((sequences - mean) / scale, -10, 10).astype(np.float32)
        # Match the original Torch Logistic's population std and feature order.
        xt = torch.from_numpy(x)
        features = torch.cat((xt.mean(1), xt.std(1, unbiased=False), xt.min(1).values, xt.max(1).values), 1).double()
        folds.append(dict(name=name, train_all=train_all, train=train, validation=validation,
                          test=test, features=features, mean=mean, scale=scale))
    expected = {i for i, e in enumerate(episodes) if labels[i] >= 0 and not e["is_control"]}
    if len(tested) != 28 or len(set(tested)) != 28 or set(tested) != expected:
        raise ValueError("Frozen test folds do not form exactly 28 unique labelled core episodes")
    return episodes, labels, names, folds, data_hashes


def fit_one(features, labels, train, seed, fold_name):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = nn.Linear(features.shape[1], 1, dtype=torch.float64)
    xtrain = features[train]
    ytrain = torch.from_numpy(labels[train])
    ntrain = len(train)
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=MAX_ITER, max_eval=MAX_EVAL,
        tolerance_grad=TOL_GRAD, tolerance_change=TOL_CHANGE,
        history_size=100, line_search_fn="strong_wolfe",
    )
    curve = []

    def objective():
        bce = nn.functional.binary_cross_entropy_with_logits(model(xtrain).squeeze(-1), ytrain)
        penalty = model.weight.square().sum() / (2.0 * ntrain)
        return bce + penalty, bce, penalty

    def closure():
        optimizer.zero_grad()
        loss, bce, penalty = objective()
        loss.backward()
        gradient = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
        curve.append(dict(
            fold=fold_name, method=METHOD, seed=seed, closure_evaluation=len(curve) + 1,
            optimizer_iteration=int(optimizer.state[model.weight].get("n_iter", 0)),
            objective=float(loss.detach()), train_bce=float(bce.detach()),
            l2_penalty=float(penalty.detach()), gradient_inf=float(gradient.abs().max()),
            point_kind="LINE_SEARCH_EVALUATION_NOT_NECESSARILY_ACCEPTED_ITERATE",
        ))
        return loss

    started = time.perf_counter()
    optimizer.step(closure)
    train_seconds = time.perf_counter() - started
    optimizer.zero_grad()
    final_loss, final_bce, final_penalty = objective()
    final_loss.backward()
    gradient = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
    grad_inf, grad_l2 = float(gradient.abs().max()), float(gradient.norm())
    if not np.isfinite([float(final_loss.detach()), grad_inf, grad_l2]).all():
        raise FloatingPointError("Nonfinite optimization result")
    state = optimizer.state[model.weight]
    iterations, evaluations = int(state.get("n_iter", 0)), int(state.get("func_evals", 0))
    gradient_converged = grad_inf <= TOL_GRAD
    stop = ("GRADIENT_TOLERANCE_SATISFIED" if gradient_converged else
            "ITERATION_LIMIT" if iterations >= MAX_ITER else
            "FUNCTION_EVALUATION_LIMIT" if evaluations >= MAX_EVAL else
            "SOLVER_STOPPED_BEFORE_GRADIENT_TOLERANCE")
    convergence = dict(
        fold=fold_name, method=METHOD, seed=seed, train_n=ntrain,
        iterations=iterations, function_evaluations=evaluations,
        gradient_inf=grad_inf, gradient_l2=grad_l2,
        gradient_converged=gradient_converged, stop_status=stop,
        objective=float(final_loss.detach()), train_bce=float(final_bce.detach()),
        l2_penalty=float(final_penalty.detach()), train_seconds=train_seconds,
    )
    curve.append({**convergence, "closure_evaluation": len(curve) + 1,
                  "optimizer_iteration": iterations, "point_kind": "FINAL_ACCEPTED_PARAMETERS"})
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(features).squeeze(-1)).numpy()
    return model, probabilities, convergence, curve


def pooled_metrics(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["seed"], row["mode"]].append(row)
    pooled = []
    for (seed, mode), values in sorted(groups.items()):
        if len(values) != 2 or {r["fold"] for r in values} != set(FOLD_NAMES):
            raise ValueError("Incomplete pooled fold result")
        totals = {key: sum(r[key] for r in values) for key in (
            "n", "passable_n", "blocked_n", "true_pass_n", "true_block_n",
            "false_pass_n", "false_reject_n", "uncertain_n")}
        n, positives, negatives = (totals[k] for k in ("n", "passable_n", "blocked_n"))
        if (n, positives, negatives) != (28, 2, 26):
            raise ValueError("Unexpected pooled denominators")
        pr, br = totals["true_pass_n"] / positives, totals["true_block_n"] / negatives
        pooled.append(dict(
            method=METHOD, audit_status=AUDIT_NOTE, seed=seed, mode=mode, **totals,
            false_pass=totals["false_pass_n"] / negatives,
            false_reject=totals["false_reject_n"] / positives,
            passable_recall=pr, blocked_recall=br, balanced_accuracy=(pr + br) / 2,
            coverage=1 - totals["uncertain_n"] / n, uncertain_rate=totals["uncertain_n"] / n,
            brier=sum(r["brier"] * r["n"] for r in values) / n,
            train_n_min=min(r["train_n"] for r in values), train_n_max=max(r["train_n"] for r in values),
            train_seconds=sum(r["train_seconds"] for r in values),
            iterations_max=max(r["iterations"] for r in values),
            gradient_inf_max=max(r["gradient_inf"] for r in values),
            all_folds_gradient_converged=all(r["gradient_converged"] for r in values),
        ))
    return pooled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=OUT / "dataset/manifest.json")
    parser.add_argument("--original-run", type=Path, default=OUT / "phase_a_valid")
    parser.add_argument("--output", type=Path, default=OUT / "converged_logistic")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Preserving existing audit output; choose a new --output directory")
    report_path = REPORT / "CONVERGED_LOGISTIC_RESULTS.csv"
    if report_path.exists():
        raise FileExistsError("Preserving the existing dedicated audit report")
    episodes, labels, names, folds, data_hashes = prepare(args.manifest, args.original_run)
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        method=METHOD, audit_status=AUDIT_NOTE, seeds=list(SEEDS), C=1.0,
        objective="mean BCE + 0.5/n_train * sum(weight**2)", intercept_penalized=False,
        optimizer="LBFGS", line_search="strong_wolfe", max_iter=MAX_ITER, max_eval=MAX_EVAL,
        tolerance_grad=TOL_GRAD, tolerance_change=TOL_CHANGE, dtype="float64 optimizer; original float32 preprocessing",
        sequence_length=48, feature_count=165, statistic_count=660,
        statistics=["mean", "population_std", "min", "max"], clip=[-10, 10], normalization_scale_floor=.001,
        normalization="only frozen train groups, including their UNKNOWN; BCE only reliable labelled train rows",
        selection="fixed objective/tolerances, no test-driven tuning or checkpoint selection",
        threshold_modes={"forced": [.5, .5], "selective": [.2, .8],
                         "validation_zero_fp": "low=.2; high=max(.8,max(validation_negative_p)+1e-6); no risk guarantee"},
        interpretation="Changes both optimization and penalty versus original AdamW; no pure-optimizer causal claim",
        manifest_sha256=sha(args.manifest), original_config_sha256=sha(args.original_run / "frozen_config.json"),
        script_sha256=sha(Path(__file__)), helper_source_sha256=sha(Path(__file__).with_name("train_models.py")),
        source_npz_sha256=data_hashes,
        folds=[dict(name=f["name"], **{role: [episodes[i]["trial_id"] for i in f[role]]
                    for role in ("train_all", "train", "validation", "test")}) for f in folds],
    )
    dump(args.output / "frozen_audit_config.json", protocol)
    all_metrics, all_predictions, all_curves, convergences = [], [], [], []
    statistic_names = [f"{kind}/{name}" for kind in protocol["statistics"] for name in names]
    for fold in folds:
        for seed in SEEDS:
            model, probabilities, conv, curve = fit_one(fold["features"], labels, fold["train"], seed, fold["name"])
            convergences.append(conv)
            all_curves.extend(curve)
            negative_validation = [i for i in fold["validation"] if labels[i] == 0]
            high = max(.8, float(probabilities[negative_validation].max()) + 1e-6) if negative_validation else 1.000001
            modes = (("forced", .5, .5), ("selective", .2, .8), ("validation_zero_fp", .2, high))
            for mode, low, upper in modes:
                all_metrics.append({**conv, "audit_status": AUDIT_NOTE, "mode": mode,
                                    "low": low, "high": upper, "validation_n": len(fold["validation"]),
                                    **metrics(labels[fold["test"]], probabilities[fold["test"]], low, upper)})
            for role in ("train", "validation", "test"):
                for i in fold[role]:
                    probability = float(probabilities[i])
                    entry = dict(fold=fold["name"], method=METHOD, seed=seed, role=role,
                                 trial_id=episodes[i]["trial_id"], group_id=episodes[i]["group_id"],
                                 side=episodes[i]["side"], is_control=episodes[i]["is_control"],
                                 y=int(labels[i]), p=probability)
                    for mode, low, upper in modes:
                        entry[mode + "_decision"] = "PASSABLE" if probability >= upper else "BLOCKED" if probability < low else "UNCERTAIN"
                    all_predictions.append(entry)
            torch.save(dict(
                state_dict=model.state_dict(), convergence=conv, seed=seed, method=METHOD,
                feature_names=names, statistic_names=statistic_names, mean=fold["mean"], scale=fold["scale"],
                train_ids=[episodes[i]["trial_id"] for i in fold["train"]],
                normalization_train_ids=[episodes[i]["trial_id"] for i in fold["train_all"]],
                audit_protocol=protocol, input="population statistics of normalized 48-point 165-channel sequence",
            ), args.output / f"{fold['name']}_{METHOD}_{seed}.pt")
            # Preserve completed fits even if a later fit is interrupted.
            write_csv(args.output / "metrics.csv", all_metrics)
            write_csv(args.output / "predictions.csv", all_predictions)
            write_csv(args.output / "training_curves.csv", all_curves)
            write_csv(args.output / "convergence.csv", convergences)
            print(json.dumps(conv), flush=True)
    pooled = pooled_metrics(all_metrics)
    write_csv(args.output / "pooled_results.csv", pooled)
    write_csv(report_path, pooled)
    stats = []
    for mode in ("forced", "selective", "validation_zero_fp"):
        selected = [r for r in pooled if r["mode"] == mode]
        row = dict(method=METHOD, mode=mode, seeds=len(selected), audit_status=AUDIT_NOTE)
        for key in ("false_pass_n", "false_pass", "false_reject", "passable_recall", "blocked_recall", "balanced_accuracy", "coverage", "uncertain_rate", "brier"):
            values = [r[key] for r in selected]
            row[key + "_mean"] = statistics.mean(values)
            row[key + "_std"] = statistics.stdev(values)
        stats.append(row)
    write_csv(args.output / "mean_std.csv", stats)
    dump(args.output / "completion.json", dict(
        status="COMPLETED", fits=len(convergences), report=str(report_path),
        all_fits_gradient_converged=all(r["gradient_converged"] for r in convergences),
        interpretation=AUDIT_NOTE, original_17_method_reports_untouched=True,
    ))
    print("CONVERGED_LOGISTIC_AUDIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
