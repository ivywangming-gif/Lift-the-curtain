"""Evaluate small deployment-only baselines for the Round 4070 campaign.

The evaluator deliberately separates deployment inputs from evaluation labels.
Only T0/T1 telemetry-derived values are used as features.  Whole-body
clearance is read from the evidence ledger after feature construction and is
never exposed to a model during fitting or prediction.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
RESEARCH = REPO / "research" / "round4070_problem_discovery"
FEATURE_NAMES = (
    "hand_forward_delta_m",
    "base_delta_m",
    "upper_tracking_rms_mean_rad",
    "effort_rms_mean",
    "joint_velocity_upper_norm_mean",
    "q_cmd_minus_q_upper_rms_mean_rad",
    "measured_planar_speed_mean_mps",
    "command_vx_mean_mps",
)
RESPONSIVE_PROGRESS_M = 0.0076811246573925025


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    names = sorted({key for row in rows for key in row}) or ["status"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def fit_logistic(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if len(x) == 0 or len(np.unique(y)) < 2:
        return None
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1.0e-8] = 1.0
    normalized = (x - mean) / std
    design = np.concatenate([np.ones((len(normalized), 1)), normalized], axis=1)
    weights = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(800):
        probability = sigmoid(design @ weights)
        gradient = design.T @ (probability - y) / len(y)
        gradient[1:] += 1.0e-3 * weights[1:]
        weights -= 0.03 * gradient
    return weights, mean, std


def predict_logistic(x: np.ndarray, model: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    weights, mean, std = model
    normalized = (np.asarray(x, dtype=np.float64) - mean) / std
    return sigmoid(np.concatenate([np.ones((len(normalized), 1)), normalized], axis=1) @ weights)


def auroc(y: np.ndarray, score: np.ndarray) -> float | None:
    positives = score[y == 1]
    negatives = score[y == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return None
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in positives for n in negatives]))


def average_precision(y: np.ndarray, score: np.ndarray) -> float | None:
    if int(y.sum()) == 0:
        return None
    order = np.argsort(-score, kind="stable")
    sorted_y = y[order]
    precision = np.cumsum(sorted_y) / np.arange(1, len(y) + 1)
    return float(np.sum(precision[sorted_y == 1]) / int(y.sum()))


def metrics(y: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    prediction = (score >= threshold).astype(np.int64)
    tp = int(((y == 1) & (prediction == 1)).sum())
    tn = int(((y == 0) & (prediction == 0)).sum())
    fp = int(((y == 0) & (prediction == 1)).sum())
    fn = int(((y == 1) & (prediction == 0)).sum())
    positive_recall = tp / (tp + fn) if tp + fn else None
    negative_recall = tn / (tn + fp) if tn + fp else None
    ece = 0.0
    for bin_id in range(10):
        low = bin_id / 10.0
        high = (bin_id + 1) / 10.0
        selected = (score >= low) & (score < high if bin_id < 9 else score <= high)
        if selected.any():
            ece += float(selected.mean()) * abs(float(score[selected].mean()) - float(y[selected].mean()))
    return {
        "status": "COMPLETED",
        "n": int(len(y)),
        "positive": int(y.sum()),
        "negative": int((1 - y).sum()),
        "accuracy": float((prediction == y).mean()) if len(y) else None,
        "balanced_accuracy": ((positive_recall + negative_recall) / 2.0 if positive_recall is not None and negative_recall is not None else None),
        "positive_recall": positive_recall,
        "negative_recall": negative_recall,
        "false_pass_rate": fp / (fp + tn) if fp + tn else None,
        "false_reject_rate": fn / (fn + tp) if fn + tp else None,
        "brier": float(np.mean((score - y) ** 2)) if len(y) else None,
        "auroc": auroc(y, score),
        "average_precision": average_precision(y, score),
        "ece_10_equal_width": float(ece),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def build_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evidence = {row["trial_id"]: row for row in read_csv(RESEARCH / "EVIDENCE_MATRIX.csv")}
    features = read_csv(RESEARCH / "deployment_cutoff_features.csv")
    by_trial_cutoff = {(row["trial_id"], row["cutoff"]): row for row in features}
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for trial_id, evidence_row in sorted(evidence.items()):
        if not as_bool(evidence_row.get("runtime_valid")):
            exclusions.append({"trial_id": trial_id, "reason": "runtime_invalid_or_incomplete"})
            continue
        if evidence_row.get("scenario") == "empty":
            exclusions.append({"trial_id": trial_id, "reason": "empty_control_not_a_candidate"})
            continue
        if not as_bool(evidence_row.get("effort_source_valid_for_probe")):
            exclusions.append({"trial_id": trial_id, "reason": "effort_source_unavailable"})
            continue
        t0 = by_trial_cutoff.get((trial_id, "T0"))
        t1 = by_trial_cutoff.get((trial_id, "T1"))
        t2 = by_trial_cutoff.get((trial_id, "T2"))
        if t0 is None or t1 is None:
            exclusions.append({"trial_id": trial_id, "reason": "missing_T0_or_T1_features"})
            continue
        values: dict[str, float] = {}
        residuals: dict[str, float] = {}
        t2_values: dict[str, float] = {}
        t2_residuals: dict[str, float] = {}
        missing = []
        for name in FEATURE_NAMES:
            before = as_float(t0.get(name))
            after = as_float(t1.get(name))
            if before is None or after is None:
                missing.append(name)
                continue
            values[f"t1_{name}"] = after
            values[f"residual_{name}"] = after - before
            residuals[name] = after - before
            if t2 is not None:
                later = as_float(t2.get(name))
                if later is not None:
                    t2_values[f"t2_{name}"] = later
                    t2_residuals[f"residual_t2_{name}"] = later - before
        if missing:
            exclusions.append({"trial_id": trial_id, "reason": "missing_numeric_feature", "fields": "|".join(missing)})
            continue
        rows.append({
            "trial_id": trial_id,
            "stage": evidence_row.get("stage"),
            "scenario": evidence_row.get("scenario"),
            "probe_side": evidence_row.get("probe_side"),
            "seed": evidence_row.get("seed"),
            "label_clearance": int(as_bool(evidence_row.get("whole_body_clearance"))),
            "decision_responsive": int(evidence_row.get("decision") == "RESPONSIVE"),
            "t1": [values[f"t1_{name}"] for name in FEATURE_NAMES],
            "residual": [residuals[name] for name in FEATURE_NAMES],
            "t2": [t2_values.get(f"t2_{name}") for name in FEATURE_NAMES] if len(t2_values) == len(FEATURE_NAMES) else None,
            "residual_t2": [t2_residuals.get(f"residual_t2_{name}") for name in FEATURE_NAMES] if len(t2_residuals) == len(FEATURE_NAMES) else None,
            "t0_effort_rms_mean": as_float(t0.get("effort_rms_mean")),
            "t1_low_speed_duration_s": as_float(t1.get("hand_low_speed_duration_s")),
            "t1_duration_s": as_float(t1.get("duration_s")),
            **values,
        })
    return rows, exclusions


def main() -> None:
    RESEARCH.mkdir(parents=True, exist_ok=True)
    rows, exclusions = build_rows()
    train_rows = [row for row in rows if row["stage"] in {"baseline", "exploration"}]
    test_rows = [row for row in rows if row["stage"] == "confirmatory"]
    train_y = np.asarray([row["label_clearance"] for row in train_rows], dtype=np.int64)
    test_y = np.asarray([row["label_clearance"] for row in test_rows], dtype=np.int64)
    train_t1 = np.asarray([row["t1"] for row in train_rows], dtype=np.float64)
    test_t1 = np.asarray([row["t1"] for row in test_rows], dtype=np.float64)
    train_residual = np.asarray([row["residual"] for row in train_rows], dtype=np.float64)
    test_residual = np.asarray([row["residual"] for row in test_rows], dtype=np.float64)

    predictions: list[dict[str, Any]] = []
    model_results: dict[str, dict[str, Any]] = {}

    def record_model(name: str, description: str, train_score: np.ndarray | None, test_score: np.ndarray | None, status: str = "COMPLETED", reason: str | None = None) -> None:
        result: dict[str, Any] = {
            "name": name,
            "description": description,
            "status": status,
            "train_n": int(len(train_rows)),
            "test_n": int(len(test_rows)),
            "train_positive": int(train_y.sum()),
            "test_positive": int(test_y.sum()),
        }
        if reason:
            result["reason"] = reason
        if train_score is not None:
            result["train_metrics"] = metrics(train_y, train_score)
        if test_score is not None:
            result["test_metrics"] = metrics(test_y, test_score)
            for row, score in zip(test_rows, test_score):
                predictions.append({
                    "trial_id": row["trial_id"],
                    "stage": row["stage"],
                    "model": name,
                    "score": float(score),
                    "predicted_clearance_at_0_5": bool(score >= 0.5),
                    "evaluation_label_clearance": bool(row["label_clearance"]),
                })
        model_results[name] = result

    # A: current frozen response decision.  This is a lawful deployment proxy,
    # but it was designed for local response, not whole-body clearance.
    score_a_train = np.asarray([row["decision_responsive"] for row in train_rows], dtype=np.float64)
    score_a_test = np.asarray([row["decision_responsive"] for row in test_rows], dtype=np.float64)
    record_model("A_frozen_response_rule", "Current proprioceptive RESPONSIVE versus BLOCKED decision.", score_a_train, score_a_test)

    # B: a transparent stability-gated engineering rule. This was specified
    # after the physical runs, so it is explicitly exploratory, not a
    # confirmatory controller comparison.
    def engineering_score(row: dict[str, Any]) -> float:
        t1 = row["t1"]
        progress_ok = t1[0] >= RESPONSIVE_PROGRESS_M
        base_ok = t1[1] <= 0.003
        tracking_ok = t1[2] <= 0.05
        effort_delta = t1[3] - row["t0_effort_rms_mean"]
        effort_ok = abs(effort_delta) <= 1.0
        low_speed_s = row["t1_low_speed_duration_s"]
        stall_ok = low_speed_s is not None and low_speed_s <= 0.15
        return float(row["decision_responsive"] and progress_ok and base_ok and tracking_ok and effort_ok and stall_ok)

    score_b_train = np.asarray([engineering_score(row) for row in train_rows], dtype=np.float64)
    score_b_test = np.asarray([engineering_score(row) for row in test_rows], dtype=np.float64)
    record_model("B_response_stability_rule", "RESPONSIVE plus T1 progress, base motion, q_cmd-q tracking, T1-minus-T0 effort RMS, low-speed duration, and phase duration; post-hoc exploratory engineering rule.", score_b_train, score_b_test)

    # C: residual logistic model.  The residual is T1 minus the same episode's
    # pre-probe T0 telemetry, which is a deployment-only history operation.
    residual_model = fit_logistic(train_residual, train_y)
    if residual_model is None:
        record_model("C_t1_t0_residual_logistic", "Logistic model on deployment-only T1-minus-T0 residuals.", None, None, "NOT_ASSESSABLE", "training set lacks two clearance classes")
    else:
        record_model(
            "C_t1_t0_residual_logistic",
            "Logistic model on deployment-only T1-minus-T0 residuals.",
            predict_logistic(train_residual, residual_model),
            predict_logistic(test_residual, residual_model),
        )

    # D: simple T1 logistic model.  It is intentionally small and not a deep
    # temporal learner; the confirmation set remains untouched during fitting.
    t1_model = fit_logistic(train_t1, train_y)
    if t1_model is None:
        record_model("D_t1_logistic", "Small logistic model on deployment-only T1 telemetry features.", None, None, "NOT_ASSESSABLE", "training set lacks two clearance classes")
    else:
        record_model(
            "D_t1_logistic",
            "Small logistic model on deployment-only T1 telemetry features.",
            predict_logistic(train_t1, t1_model),
            predict_logistic(test_t1, t1_model),
        )

    # E: the requested history MLP/GRU is deliberately excluded.  There are
    # only 19 discovery candidates and 16 confirmation candidates, with no
    # independent scene family or repeated action branches sufficient for a
    # defensible temporal learner.
    record_model(
        "E_history_mlp_or_gru",
        "History MLP/GRU candidate; not trained.",
        None,
        None,
        "EXCLUDED_INSUFFICIENT_DATA",
        "insufficient independent conditions and repeated action branches; fitting would be model-selection overfit",
    )

    # T2 is reported as an exploratory, post-advance analysis.  It is not
    # evidence for a T1 decision because the cautious-advance phase can itself
    # change the physical state and is close to the eventual outcome.
    t2_rows = [row for row in rows if row["t2"] is not None and row["residual_t2"] is not None]
    t2_train_rows = [row for row in t2_rows if row["stage"] in {"baseline", "exploration"}]
    t2_test_rows = [row for row in t2_rows if row["stage"] == "confirmatory"]
    t2_train_y = np.asarray([row["label_clearance"] for row in t2_train_rows], dtype=np.int64)
    t2_test_y = np.asarray([row["label_clearance"] for row in t2_test_rows], dtype=np.int64)
    t2_train = np.asarray([row["t2"] for row in t2_train_rows], dtype=np.float64)
    t2_test = np.asarray([row["t2"] for row in t2_test_rows], dtype=np.float64)
    t2_train_residual = np.asarray([row["residual_t2"] for row in t2_train_rows], dtype=np.float64)
    t2_test_residual = np.asarray([row["residual_t2"] for row in t2_test_rows], dtype=np.float64)
    t2_results: dict[str, dict[str, Any]] = {}

    def record_t2(name: str, description: str, train_score: np.ndarray | None, test_score: np.ndarray | None, status: str = "COMPLETED", reason: str | None = None) -> None:
        result: dict[str, Any] = {
            "name": name,
            "description": description,
            "status": status,
            "cutoff": "T2_exploratory",
            "train_n": int(len(t2_train_rows)),
            "test_n": int(len(t2_test_rows)),
            "train_positive": int(t2_train_y.sum()),
            "test_positive": int(t2_test_y.sum()),
        }
        if reason:
            result["reason"] = reason
        if train_score is not None:
            result["train_metrics"] = metrics(t2_train_y, train_score)
        if test_score is not None:
            result["test_metrics"] = metrics(t2_test_y, test_score)
            for row, score in zip(t2_test_rows, test_score):
                predictions.append({
                    "trial_id": row["trial_id"],
                    "stage": row["stage"],
                    "model": name,
                    "score": float(score),
                    "predicted_clearance_at_0_5": bool(score >= 0.5),
                    "evaluation_label_clearance": bool(row["label_clearance"]),
                })
        t2_results[name] = result

    t2_residual_model = fit_logistic(t2_train_residual, t2_train_y)
    if t2_residual_model is None:
        record_t2("C_t2_t0_residual_logistic", "Exploratory T2-minus-T0 residual logistic model.", None, None, "NOT_ASSESSABLE", "T2 training set lacks two clearance classes")
    else:
        record_t2(
            "C_t2_t0_residual_logistic",
            "Exploratory T2-minus-T0 residual logistic model.",
            predict_logistic(t2_train_residual, t2_residual_model),
            predict_logistic(t2_test_residual, t2_residual_model),
        )
    t2_model = fit_logistic(t2_train, t2_train_y)
    if t2_model is None:
        record_t2("D_t2_logistic", "Exploratory T2 telemetry logistic model.", None, None, "NOT_ASSESSABLE", "T2 training set lacks two clearance classes")
    else:
        record_t2(
            "D_t2_logistic",
            "Exploratory T2 telemetry logistic model.",
            predict_logistic(t2_train, t2_model),
            predict_logistic(t2_test, t2_model),
        )

    feature_rows: list[dict[str, Any]] = []
    for row in rows:
        feature_rows.append({
            "trial_id": row["trial_id"],
            "stage": row["stage"],
            "scenario": row["scenario"],
            "probe_side": row["probe_side"],
            "seed": row["seed"],
            **{f"t1_{name}": value for name, value in zip(FEATURE_NAMES, row["t1"])},
            **{f"residual_{name}": value for name, value in zip(FEATURE_NAMES, row["residual"])},
            "t0_effort_rms_mean": row["t0_effort_rms_mean"],
            "t1_low_speed_duration_s": row["t1_low_speed_duration_s"],
            "t1_duration_s": row["t1_duration_s"],
            **({f"t2_{name}": value for name, value in zip(FEATURE_NAMES, row["t2"])} if row["t2"] is not None else {}),
            **({f"residual_t2_{name}": value for name, value in zip(FEATURE_NAMES, row["residual_t2"])} if row["residual_t2"] is not None else {}),
        })
    write_csv(RESEARCH / "deployment_model_features.csv", feature_rows)
    write_csv(RESEARCH / "BASELINE_PREDICTIONS.csv", predictions)

    evaluation = {
        "schema": "lift-the-curtain.round4070-baseline-evaluation.v1",
        "status": "OFFLINE_CONFIRMATORY_EVALUATION",
        "feature_source": "research/round4070_problem_discovery/deployment_model_features.csv",
        "label_source": "research/round4070_problem_discovery/EVIDENCE_MATRIX.csv",
        "feature_contract": {
            "deployment_only": True,
            "feature_names": list(FEATURE_NAMES),
            "residual_definition": "T1 minus same-trial T0 numeric telemetry",
            "forbidden_inputs": ["scenario", "object identity", "simulator contact", "cloth nodes", "true clearance", "future outcome"],
        },
        "split": {
            "train": "stage in {baseline, exploration}",
            "test": "stage == confirmatory",
            "interpretation": "seed-held-out confirmation within the preregistered scenario family, not an OOD benchmark",
            "train_n": len(train_rows),
            "test_n": len(test_rows),
            "train_positive": int(train_y.sum()),
            "test_positive": int(test_y.sum()),
        },
        "models": model_results,
        "t2_exploratory_models": t2_results,
        "t2_rows_available": len(t2_rows),
        "t2_rows_missing": len(rows) - len(t2_rows),
        "excluded_rows": exclusions,
        "caveats": [
            "The confirmation set contains no valid rigid-panel examples.",
            "Whole-body clearance is an evaluation label and was not available to the deployment branch.",
            "Metrics are descriptive at small n; no algorithm-readiness claim is made.",
        ],
    }
    write_json(RESEARCH / "BASELINE_EVALUATION.json", evaluation)

    # Preserve the descriptive historical rows and append a machine-readable
    # view of the offline baseline results in the required comparison file.
    comparison_path = RESEARCH / "BASELINE_COMPARISON.csv"
    existing = read_csv(comparison_path) if comparison_path.is_file() else []
    for name, result in model_results.items():
        test_metrics = result.get("test_metrics", {})
        existing.append({
            "comparison": "offline_deployment_only_baseline",
            "policy": name,
            "n_nonempty": result.get("test_n"),
            "n_valid": result.get("test_n"),
            "n_clearance": result.get("test_positive"),
            "n_pointgoal": "NOT_USED",
            "n_rigid": "NOT_IN_TEST",
            "n_flexible": result.get("test_n"),
            "n_safe_retract": "NOT_USED",
            "train_n": result.get("train_n"),
            "test_auroc": test_metrics.get("auroc"),
            "test_average_precision": test_metrics.get("average_precision"),
            "test_brier": test_metrics.get("brier"),
            "test_false_pass_rate": test_metrics.get("false_pass_rate"),
            "test_false_reject_rate": test_metrics.get("false_reject_rate"),
            "status": result.get("status"),
            "interpretation": result.get("reason", result.get("description")),
        })
    for name, result in t2_results.items():
        test_metrics = result.get("test_metrics", {})
        existing.append({
            "comparison": "offline_deployment_only_baseline_t2_exploratory",
            "policy": name,
            "n_nonempty": result.get("test_n"),
            "n_valid": result.get("test_n"),
            "n_clearance": result.get("test_positive"),
            "n_pointgoal": "NOT_USED",
            "n_rigid": "NOT_IN_TEST",
            "n_flexible": result.get("test_n"),
            "n_safe_retract": "NOT_USED",
            "train_n": result.get("train_n"),
            "test_auroc": test_metrics.get("auroc"),
            "test_average_precision": test_metrics.get("average_precision"),
            "test_brier": test_metrics.get("brier"),
            "test_false_pass_rate": test_metrics.get("false_pass_rate"),
            "test_false_reject_rate": test_metrics.get("false_reject_rate"),
            "status": result.get("status"),
            "interpretation": result.get("reason", result.get("description")),
        })
    write_csv(comparison_path, existing)
    print(json.dumps({"train_n": len(train_rows), "test_n": len(test_rows), "models": list(model_results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
