"""Offline, leakage-audited evaluator for Round 4070-D.

E0/E1/E2 read only deployment_observation.csv. Simulator truth is used only
afterwards to make an offline label; invalid or incomplete trials remain
UNKNOWN and are excluded from classifier error rates.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RESEARCH = REPO / "research" / "round4070_fixed10cm_judgment"
CONFIG = REPO / "configs" / "round4070_fixed10cm_judgment"
OUTPUT = REPO / "outputs" / "round4070_fixed10cm_judgment"
TRIALS = OUTPUT / "trials"
CALIBRATION_RECHECK = OUTPUT / "calibration_target_fk10cm"
TARGET_M = 0.10
E0_Q_ERROR_THRESHOLD = 0.12
E2_PASS_PROGRESS_M = 0.075
E2_BLOCK_PROGRESS_M = 0.035
E2_BLOCK_RATIO = 0.40
E2_STALL_S = 0.20
E2_Q_RESIDUAL_RAD = 0.10


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return default


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "pass"}:
        return True
    if text in {"false", "0", "no", "fail"}:
        return False
    return None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_vector(row: dict[str, str], field: str) -> np.ndarray:
    value = np.asarray(json.loads(row[field]), dtype=np.float64).reshape(-1)
    if value.size == 0 or not np.isfinite(value).all():
        raise ValueError(f"{field} is not a finite vector")
    return value


def load_history(path: Path):
    from lift_curtain.control.deployment_observation import DeploymentObservation

    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            phase = str(row.get("phase", ""))
            if phase.startswith("physical_probe") or phase in {"probe", "contact_hold"}:
                rows.append(row)
    if not rows:
        raise ValueError("no fixed-probe rows")
    observations = []
    for row in rows:
        source = str(row.get("deployment_effort_source", ""))
        if source in {"", "unavailable", "zero_filled"}:
            raise ValueError("effort source unavailable")
        observations.append(
            DeploymentObservation(
                q=parse_vector(row, "deployment_q_actual_json"),
                qdot=parse_vector(row, "deployment_qdot_json"),
                q_cmd=parse_vector(row, "deployment_q_cmd_json"),
                motor_effort=parse_vector(row, "deployment_motor_effort_json"),
                previous_action=parse_vector(row, "deployment_previous_action_json"),
                hand_fk_position_w=parse_vector(row, "deployment_hand_fk_position_w_json"),
                hand_fk_velocity_w=parse_vector(row, "deployment_hand_fk_velocity_w_json"),
                base_position_w=parse_vector(row, "deployment_base_position_w_json"),
                base_velocity_w=parse_vector(row, "deployment_base_velocity_b_json"),
                imu_orientation_rpy=np.zeros(3, dtype=np.float64),
                imu_angular_velocity=np.zeros(3, dtype=np.float64),
                localization_assumption=row.get("deployment_localization_assumption", ""),
            )
        )
    return observations, rows


def feature_dict(history, baseline):
    from lift_curtain.control.deployment_observation import extract_proprioceptive_features

    result = extract_proprioceptive_features(
        history, baseline=baseline, sample_period_s=0.005, forward_axis_w=(1.0, 0.0, 0.0)
    )
    return {key: (None if isinstance(value, float) and not math.isfinite(value) else value) for key, value in result.items()}


def frozen_e1(history, baseline):
    from lift_curtain.control.deployment_observation import ProprioceptiveThresholds, classify_proprioceptive_response

    threshold_path = REPO / "configs" / "proprioceptive_thresholds.json"
    thresholds = ProprioceptiveThresholds(**read_json(threshold_path, {}))
    decision, features = classify_proprioceptive_response(
        history, baseline=baseline, thresholds=thresholds, sample_period_s=0.005
    )
    return decision.value, {key: (None if isinstance(value, float) and not math.isfinite(value) else value) for key, value in features.items()}


def e0_decision(features: dict[str, Any]) -> str:
    value = finite(features.get("upper_tracking_rms"))
    if value is None:
        return "UNCERTAIN"
    return "PASSABLE" if value < E0_Q_ERROR_THRESHOLD else "BLOCKED"


def e2_decision(features: dict[str, Any], baseline_features: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    progress = finite(features.get("hand_forward_progress_m"))
    effort = finite(features.get("effort_deviation"))
    tracking = finite(features.get("upper_tracking_rms"))
    stall = finite(features.get("hand_low_speed_duration_s"))
    normal_tracking = finite(baseline_features.get("upper_tracking_rms")) or 0.0
    progress_ratio = None if progress is None else progress / TARGET_M
    q_residual = None if tracking is None else tracking - normal_tracking
    blocked = bool(
        progress is not None and effort is not None and stall is not None
        and (progress <= E2_BLOCK_PROGRESS_M or (progress_ratio is not None and progress_ratio <= E2_BLOCK_RATIO))
        and stall >= E2_STALL_S
        and (effort >= 2.2690875035237013 or (q_residual is not None and q_residual >= E2_Q_RESIDUAL_RAD))
    )
    passable = bool(
        progress is not None and tracking is not None and stall is not None
        and progress >= E2_PASS_PROGRESS_M
        and q_residual is not None and q_residual < E2_Q_RESIDUAL_RAD
        and stall < 0.80
    )
    decision = "BLOCKED" if blocked else "PASSABLE" if passable else "UNCERTAIN"
    return decision, {
        "target_progress_m": TARGET_M,
        "progress_ratio_to_target": progress_ratio,
        "normal_tracking_rms": normal_tracking,
        "tracking_residual_rms": q_residual,
        "pass_progress_threshold_m": E2_PASS_PROGRESS_M,
        "blocked_progress_threshold_m": E2_BLOCK_PROGRESS_M,
        "blocked_stall_threshold_s": E2_STALL_S,
        "blocked_effort_threshold": 2.2690875035237013,
        "rule_version": "E2_UNIFIED_FIXED_THRESHOLDS_V1",
    }


def gate_status(gate_id: str) -> str:
    if gate_id in {"CONTROL_ONLY", "INHERITED_RIGID_CONTROL"}:
        return "PASS_CONTROL"
    summary = OUTPUT / "physics" / gate_id / "probe" / "summary.json"
    value = read_json(summary, {}) or {}
    status = str(value.get("status", "MISSING"))
    if status in {"PASS_PHYSICS_SANITY", "CONTACT_PASS"}:
        return "PASS"
    inherited = {
        "A0_TOP_EDGE_ONLY": "PASS_INHERITED_C",
        "A1_TOP_EDGE_BOTTOM_TWO_CORNERS": "PASS_INHERITED_C",
        "A2_FOUR_CORNERS_ONLY": "PASS_INHERITED_C",
        "A3_FULL_PERIMETER_FIXED_V2": "PASS_INHERITED_C",
        "B_D225_NEW": "PASS_INHERITED_C",
        "C_GAP05": "PASS_INHERITED_C",
    }
    return inherited.get(gate_id, "PENDING") if status == "MISSING" else f"FAIL_{status}"


def classify_truth(
    plan: dict[str, Any],
    summary: dict[str, Any],
    physics: str,
    *,
    protocol_match: bool = True,
    protocol_detail: str = "",
    probe_validation_only: bool = False,
) -> tuple[str, str, str]:
    family = plan.get("condition_family")
    if family == "control":
        if plan.get("condition_id") == "EMPTY_CONTROL":
            return "CONTROL_PASSABLE", "control", "empty control is not a flexible benchmark positive"
        return "CONTROL_BLOCKED", "control", "rigid control is not a flexible benchmark negative"
    if not protocol_match:
        return "UNKNOWN", "PROTOCOL_MISMATCH", protocol_detail
    if probe_validation_only:
        return "UNKNOWN", "SAFETY_VALIDATION_ONLY", "probe-validation-only output is not a benchmark trial"
    if physics.startswith("FAIL") or physics.startswith("PENDING") or physics == "MISSING":
        return "UNKNOWN", "PHYSICS_INVALID_OR_UNVALIDATED", physics
    if str(summary.get("runtime_status", "")) != "RUNTIME_PASS_FINALIZE_REACHED":
        return "UNKNOWN", "EXECUTION_INVALID", str(summary.get("runtime_status", "missing"))
    if not boolean(summary.get("safe_recovery")):
        return "UNKNOWN", "SAFETY_OR_RECOVERY_INVALID", "safe_recovery=false"
    stop_reason = str(summary.get("probe_stop_reason", ""))
    actual = finite(summary.get("probe_actual_displacement_m", summary.get("probe_actual_hand_forward_progress_m")))
    if stop_reason.startswith("SAFETY_ABORT"):
        return "UNKNOWN", "SAFETY_ABORT", stop_reason
    if actual is None or actual < 0.095:
        return "UNKNOWN", "PROBE_INCOMPLETE", f"actual={actual}; stop={stop_reason}"
    clearance = boolean(summary.get("whole_body_candidate_clearance"))
    physical = boolean(summary.get("physical_branch_pass"))
    probe = summary.get("physical_probe_result") or {}
    contact = boolean(probe.get("evaluation_hand_contact_confirmed"))
    local = str(probe.get("decision", "UNKNOWN"))
    if clearance is True and physical is True:
        return "PASSABLE", "OFFLINE_WHOLE_BODY_CLEARANCE", "candidate clearance verified after fixed continuation"
    if clearance is False and physical is True and boolean(summary.get("safe_recovery")) and contact:
        if local == "BLOCKED" or str(summary.get("probe_safety_stop_reason", "")) or str(summary.get("external_safety_reason", "")):
            return "BLOCKED", "OFFLINE_PHYSICAL_CONSTRAINT", "contact plus no whole-body clearance"
    return "UNKNOWN", "PLANNER_OR_OBSERVATION_HORIZON", "no independent physical pass/block certificate"


def wilson(success: int, total: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    p = success / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return max(0.0, center - half), min(1.0, center + half)


def metrics(rows: list[dict[str, Any]], method: str, split: str, scope: str) -> dict[str, Any]:
    selected = [
        row for row in rows
        if row.get("benchmark_eligible", True)
        and row["label"] in {"PASSABLE", "BLOCKED"}
        and row["split"] == split
        and row["scope"] == scope
    ]
    decisions = [row["decisions"][method] for row in selected]
    positives = sum(row["label"] == "PASSABLE" for row in selected)
    negatives = sum(row["label"] == "BLOCKED" for row in selected)
    false_pass = sum(row["label"] == "BLOCKED" and row["decisions"][method] == "PASSABLE" for row in selected)
    false_reject = sum(row["label"] == "PASSABLE" and row["decisions"][method] == "BLOCKED" for row in selected)
    passable_correct = sum(row["label"] == "PASSABLE" and row["decisions"][method] == "PASSABLE" for row in selected)
    blocked_correct = sum(row["label"] == "BLOCKED" and row["decisions"][method] == "BLOCKED" for row in selected)
    uncertain = sum(decision == "UNCERTAIN" for decision in decisions)
    coverage = None if not selected else (len(selected) - uncertain) / len(selected)
    p_recall = None if positives == 0 else passable_correct / positives
    b_recall = None if negatives == 0 else blocked_correct / negatives
    balanced = None if p_recall is None or b_recall is None else 0.5 * (p_recall + b_recall)
    fp_lo, fp_hi = wilson(false_pass, negatives)
    fr_lo, fr_hi = wilson(false_reject, positives)
    cm = {truth: {prediction: 0 for prediction in ("PASSABLE", "BLOCKED", "UNCERTAIN")} for truth in ("PASSABLE", "BLOCKED")}
    for row in selected:
        cm[row["label"]][row["decisions"][method]] += 1
    return {
        "method": method, "split": split, "scope": scope,
        "eligible_n": len(selected), "passable_n": positives, "blocked_n": negatives,
        "false_pass_n": false_pass, "false_pass_rate": None if negatives == 0 else false_pass / negatives,
        "false_pass_ci95_low": fp_lo, "false_pass_ci95_high": fp_hi,
        "false_reject_n": false_reject, "false_reject_rate": None if positives == 0 else false_reject / positives,
        "false_reject_ci95_low": fr_lo, "false_reject_ci95_high": fr_hi,
        "passable_recall": p_recall, "blocked_recall": b_recall,
        "uncertain_n": uncertain, "uncertain_rate": None if not selected else uncertain / len(selected),
        "decision_coverage": coverage, "balanced_accuracy": balanced,
        "confusion_matrix_json": json.dumps(cm, sort_keys=True),
    }


def load_manifest() -> list[dict[str, Any]]:
    payload = read_json(CONFIG / "PREREGISTERED_MANIFEST.json", {}) or {}
    return list(payload.get("trials", []))


def protocol_audit(plan: dict[str, Any], summary: dict[str, Any], metadata: dict[str, Any]) -> tuple[bool, str]:
    expected_step = finite(plan.get("probe_10cm_joint_step_rad"))
    observed_step = finite(((metadata.get("overrides") or {}).get("probe_10cm_joint_step_rad")))
    expected_target = finite(plan.get("target_displacement_m"))
    observed_target = finite(summary.get("probe_target_displacement_m"))
    observed_mode = summary.get("probe_depth_level")
    problems: list[str] = []
    if expected_step is None or observed_step is None or abs(expected_step - observed_step) > 1.0e-9:
        problems.append(f"joint_step expected={expected_step}; observed={observed_step}")
    if expected_target is None or observed_target is None or abs(expected_target - observed_target) > 1.0e-9:
        problems.append(f"target expected={expected_target}; observed={observed_target}")
    if observed_mode not in {None, "PROBE_10CM"}:
        problems.append(f"mode observed={observed_mode}")
    return not problems, "; ".join(problems)


def collect_trials(manifest: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[Any]]]:
    baselines: dict[str, list[Any]] = {}
    result: list[dict[str, Any]] = []
    for plan in manifest:
        path = TRIALS / str(plan["trial_id"])
        summary = read_json(path / "summary.json", {}) or {}
        metadata = read_json(path / "trial_metadata.json", {}) or {}
        history = None
        feature_error = None
        if (path / "deployment_observation.csv").is_file():
            try:
                history, _ = load_history(path / "deployment_observation.csv")
            except Exception as exc:
                feature_error = f"{type(exc).__name__}: {exc}"
        protocol_match, protocol_detail = protocol_audit(plan, summary, metadata)
        probe_validation_only = boolean(summary.get("probe_validation_only")) is True
        if (
            plan.get("condition_id") == "EMPTY_CONTROL"
            and plan.get("stage") == "discovery"
            and history is not None
            and protocol_match
        ):
            baselines.setdefault(str(plan["probe_side"]), history)
        physics = gate_status(str(plan.get("physics_gate_id", "")))
        label, label_reason, label_detail = classify_truth(
            plan,
            summary,
            physics,
            protocol_match=protocol_match,
            protocol_detail=protocol_detail,
            probe_validation_only=probe_validation_only,
        )
        result.append({
            "plan": plan, "path": path, "summary": summary, "history": history,
            "feature_error": feature_error, "physics_status": physics,
            "runtime_valid": str(summary.get("runtime_status", "")) == "RUNTIME_PASS_FINALIZE_REACHED",
            "protocol_match": protocol_match, "protocol_detail": protocol_detail,
            "probe_validation_only": probe_validation_only,
            "label": label, "label_reason": label_reason, "label_detail": label_detail,
        })
    return result, baselines


def evaluate_rows(raw: list[dict[str, Any]], baselines: dict[str, list[Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_features = {side: feature_dict(history, history) for side, history in baselines.items()}
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    forbidden = {
        "condition_id", "scenario", "material", "hidden_rigid_x", "hidden_rigid_gap_m", "structural_topology",
        "density_scale_multiplier", "simulator_contact_flag", "true_passability", "future_clearance", "evaluation_hand_contact_force_norm_N",
    }
    for item in raw:
        plan, path, summary = item["plan"], item["path"], item["summary"]
        side = str(plan.get("probe_side"))
        history, features, details = item["history"], {}, {}
        decisions = {"E0": "UNCERTAIN", "E1": "UNCERTAIN", "E2": "UNCERTAIN", "ALWAYS_PASSABLE": "PASSABLE", "ALWAYS_BLOCKED": "BLOCKED"}
        if history is not None and side in baselines:
            try:
                features = feature_dict(history, baselines[side])
                e1, e1_features = frozen_e1(history, baselines[side])
                decisions["E0"] = e0_decision(features)
                decisions["E1"] = {"RESPONSIVE": "PASSABLE", "BLOCKED": "BLOCKED", "UNCERTAIN": "UNCERTAIN"}.get(e1, "UNCERTAIN")
                decisions["E2"], details["E2"] = e2_decision(features, baseline_features[side])
                details["E1"] = {"raw_local_response": e1, "frozen_features": e1_features}
            except Exception as exc:
                item["feature_error"] = f"{type(exc).__name__}: {exc}"
        headers: set[str] = set()
        deployment_path = path / "deployment_observation.csv"
        if deployment_path.is_file():
            with deployment_path.open(encoding="utf-8", newline="") as handle:
                headers = set(csv.DictReader(handle).fieldnames or [])
        leaked = sorted(headers & forbidden)
        row = {
            "trial_id": plan["trial_id"], "condition_id": plan.get("condition_id"),
            "condition_family": plan.get("condition_family"), "probe_side": side,
            "split": str(plan.get("stage", "unknown")), "scope": "core" if plan.get("formal_core") else "noncore",
            "seed": plan.get("seed"), "summary_exists": (path / "summary.json").is_file(),
            "runtime_valid": item["runtime_valid"], "physics_status": item["physics_status"],
            "protocol_match": item["protocol_match"], "protocol_detail": item["protocol_detail"],
            "probe_validation_only": item["probe_validation_only"],
            "benchmark_eligible": bool(plan.get("formal_core")) and item["protocol_match"] and not item["probe_validation_only"],
            "target_displacement_m": TARGET_M,
            "actual_displacement_m": finite(summary.get("probe_actual_displacement_m", summary.get("probe_actual_hand_forward_progress_m"))),
            "post_contact_displacement_m": finite(summary.get("probe_post_contact_displacement_m_evaluation_only", summary.get("probe_actual_continued_forward_after_contact_m_evaluation_only"))),
            "stop_reason": summary.get("probe_stop_reason"), "label": item["label"],
            "label_reason": item["label_reason"], "label_detail": item["label_detail"],
            "features": features, "decisions": decisions, "method_details": details,
        }
        rows.append(row)
        audits.append({
            "trial_id": plan["trial_id"], "condition_id_offline": plan.get("condition_id"), "probe_side": side,
            "stage": row["split"], "runtime_valid": item["runtime_valid"], "physics_status": item["physics_status"],
            "summary_exists": row["summary_exists"],
            "target_displacement_m": TARGET_M, "actual_displacement_m": row["actual_displacement_m"],
            "post_contact_displacement_m": row["post_contact_displacement_m"], "stop_reason": row["stop_reason"],
            "label": item["label"], "label_reason": item["label_reason"], "label_detail": item["label_detail"],
            "protocol_match": item["protocol_match"], "protocol_mismatch_detail": item["protocol_detail"],
            "probe_validation_only": item["probe_validation_only"],
            "benchmark_eligible": row["benchmark_eligible"],
            "feature_error": item["feature_error"], "deployment_headers_contain_forbidden_fields": bool(leaked),
            "forbidden_fields_found": ";".join(leaked), "truth_used_online": False,
            "online_classifier_source": "deployment_observation.csv only",
        })
    return rows, audits


def case_report(path: Path, title: str, rows: list[dict[str, Any]], definition: str) -> None:
    lines = [f"# {title}", "", definition + ".", "", "UNKNOWN, runtime-invalid and physics-invalid trials are excluded from the denominator.", ""]
    if not rows:
        lines.append("No qualifying cases were observed in the available valid benchmark rows. This does not prove that the true rate is zero; use the confidence intervals in `ENGINEERING_BASELINE_COMPARISON.csv`.")
    else:
        lines.append("| Trial | Condition | Side | E0 | E1 | E2 | Features |")
        lines.append("|---|---|---|---|---|---|---|")
        for row in rows:
            lines.append(f"| {row['trial_id']} | {row['condition_id']} | {row['probe_side']} | {row['decisions']['E0']} | {row['decisions']['E1']} | {row['decisions']['E2']} | `{json.dumps(row['features'], sort_keys=True)}` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_calibration_recheck() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for side in ("left", "right"):
        summary_path = CALIBRATION_RECHECK / side / "summary.json"
        summary = read_json(summary_path, {}) or {}
        if not summary:
            continue
        probe = summary.get("physical_probe_result") or {}
        contract = summary.get("probe_depth_contract") or {}
        actual = finite(summary.get("probe_actual_displacement_m"))
        stop_reason = str(summary.get("probe_stop_reason", ""))
        rows.append({
            "side": side,
            "summary_path": str(summary_path),
            "profile": contract.get("target_reference_profile") or probe.get("probe_target_reference_profile"),
            "target_m": finite(summary.get("probe_target_displacement_m")),
            "actual_m": actual,
            "stop_reason": stop_reason,
            "runtime_valid": str(summary.get("runtime_status", "")) == "RUNTIME_PASS_FINALIZE_REACHED",
            "safe_recovery": boolean(summary.get("safe_recovery")),
            "safety_stop": probe.get("probe_safety_stop_reason"),
            "retract_ok": boolean((probe.get("retract") or {}).get("speed_gate_reached")),
            "neutralize_ok": boolean((probe.get("neutralize") or {}).get("speed_gate_reached")),
            "stand_ok": boolean((probe.get("post_retract_stand") or {}).get("actor_stand_flag_ok")),
            "completion": actual is not None and actual >= 0.095 and stop_reason == "TARGET_REACHED",
        })
    return rows


def write_calibration_recheck_report(rows: list[dict[str, Any]]) -> None:
    bilateral_pass = bool(rows) and all(
        row["completion"] and row["runtime_valid"] and row["safe_recovery"]
        and row["safety_stop"] in (None, "")
        and row["retract_ok"] and row["neutralize_ok"] and row["stand_ok"]
        for row in rows
    ) and {row["side"] for row in rows} == {"left", "right"}
    lines = [
        "# Probe Calibration Recheck", "",
        f"Status: **{'BILATERAL_CALIBRATION_PASS' if bilateral_pass else 'BILATERAL_CALIBRATION_NOT_PASSED'}**", "",
        "This is an isolated calibration-only recheck. It is outside the preregistered 250-trial manifest, does not alter the frozen `.85 rad` formal contract, and produces no passability label.", "",
        "Completion criterion: actual signed hand FK displacement >= `0.095 m`, `TARGET_REACHED`, runtime-finalize valid, no safety stop, and successful retract/neutralize/stand recovery.", "",
        "| Side | Target m | Actual m | Stop | Runtime | Safe recovery | Safety stop | Recovery gates | Profile |",
        "|---|---:|---:|---|---|---|---|---|---|",
    ]
    if not rows:
        lines.append("| - | - | - | NOT_RUN | - | - | - | - | - |")
    else:
        for row in rows:
            recovery = "retract={} neutralize={} stand={}".format(row["retract_ok"], row["neutralize_ok"], row["stand_ok"])
            lines.append(
                f"| {row['side']} | {row['target_m']} | {row['actual_m']} | {row['stop_reason']} | "
                f"{row['runtime_valid']} | {row['safe_recovery']} | {row['safety_stop']} | `{recovery}` | `{row['profile']}` |"
            )
    lines.extend([
        "",
        "Interpretation: a right-side candidate reached the target in this run, while the left-side candidate remained incomplete. The bilateral calibration gate therefore remains unmet; neither calibration output is included in formal benchmark metrics.",
        "",
        "Raw summaries are preserved under `outputs/round4070_fixed10cm_judgment/calibration_target_fk10cm/`.",
    ])
    (RESEARCH / "PROBE_CALIBRATION_RECHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_supporting_reports(rows: list[dict[str, Any]], metric_rows: list[dict[str, Any]]) -> None:
    calibration_rows = load_calibration_recheck()
    write_calibration_recheck_report(calibration_rows)
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(str(row["condition_family"]), []).append(row)
    info = [
        "# Information Sufficiency Analysis", "",
        "The fixed 10 cm history is evaluated separately from the offline label.", "",
        "- Type A: legal history features repeatably differ, but E1 does not use them.",
        "- Type B: matched wall/no-wall histories remain close, so the probe itself is information-limited.",
        "- Type C: local hand response exists without whole-body clearance.",
        "- Type D: frozen confirmation disagrees with discovery.",
        "- Type E: physics, safety, planner or runtime invalidity; excluded from method error rates.", "",
        "| Family | Rows | Labeled | Unknown/invalid | E0 determinate | E1 determinate | E2 determinate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family in sorted(by_family):
        family_rows = by_family[family]
        labeled = [row for row in family_rows if row["label"] in {"PASSABLE", "BLOCKED"}]
        info.append(f"| {family} | {len(family_rows)} | {len(labeled)} | {len(family_rows)-len(labeled)} | {sum(row['decisions']['E0'] != 'UNCERTAIN' for row in family_rows)} | {sum(row['decisions']['E1'] != 'UNCERTAIN' for row in family_rows)} | {sum(row['decisions']['E2'] != 'UNCERTAIN' for row in family_rows)} |")
    info.extend([
        "",
        "No UNKNOWN row is converted into BLOCKED. Matched-pair evidence is only claimable when both sides have valid deployment histories and a physics gate.",
        "",
        "This run is `INSUFFICIENT_POSITIVE_COVERAGE`: the required safety checks were runtime-valid and safely recovered, but the fixed action produced `PROBE_INCOMPLETE` actual FK motion. No Type A-D classifier conclusion is claimed before a valid benchmark row exists.",
        "",
        "Raw features and label reasons are in `LABEL_AUDIT.csv` and `stats/evaluator_rows.json`.",
    ])
    (RESEARCH / "INFORMATION_SUFFICIENCY_ANALYSIS.md").write_text("\n".join(info) + "\n", encoding="utf-8")

    general = [
        "# Generalization Results", "",
        "Discovery is the first six repeats per core condition; frozen confirmation is repeats seven through ten. The rule thresholds are frozen before confirmation.", "",
        "| Split | Core rows | Labeled | PASSABLE | BLOCKED | Unknown/invalid |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("discovery", "frozen_confirmation"):
        selected = [row for row in rows if row["scope"] == "core" and row["split"] == split]
        general.append(f"| {split} | {len(selected)} | {sum(row['label'] in {'PASSABLE','BLOCKED'} for row in selected)} | {sum(row['label']=='PASSABLE' for row in selected)} | {sum(row['label']=='BLOCKED' for row in selected)} | {sum(row['label']=='UNKNOWN' for row in selected)} |")
    general.extend([
        "",
        "Same-condition new-seed generalization and unseen physical-condition generalization are separate claims. An unseen condition is not declared successful until its complete frozen rows are valid and it was not used to set thresholds.",
        "",
        "Generalization is not established in this run because the core benchmark has zero eligible PASSABLE/BLOCKED labels.",
        "",
        "Per-family and per-hand results are in `ENGINEERING_BASELINE_COMPARISON.csv`.",
    ])
    (RESEARCH / "GENERALIZATION_RESULTS.md").write_text("\n".join(general) + "\n", encoding="utf-8")

    core = [row for row in rows if row["scope"] == "core"]
    e2_fp = [row for row in core if row["label"] == "BLOCKED" and row["decisions"]["E2"] == "PASSABLE"]
    e2_fr = [row for row in core if row["label"] == "PASSABLE" and row["decisions"]["E2"] == "BLOCKED"]
    case_report(RESEARCH / "FALSE_PASS_CASES.md", "False-pass cases", e2_fp, "E2 predicted PASSABLE while the offline label was BLOCKED")
    case_report(RESEARCH / "FALSE_REJECT_CASES.md", "False-reject cases", e2_fr, "E2 predicted BLOCKED while the offline label was PASSABLE")

    def metric(method: str) -> dict[str, Any] | None:
        return next((item for item in metric_rows if item["method"] == method and item["split"] == "all" and item["scope"] == "core"), None)

    def metric_text(item: dict[str, Any] | None) -> str:
        if item is None:
            return "尚无可用核心正负标签"
        return f"eligible={item['eligible_n']}, false-pass={item['false_pass_n']}/{item['blocked_n']}, false-reject={item['false_reject_n']}/{item['passable_n']}, coverage={item['decision_coverage']}, balanced_accuracy={item['balanced_accuracy']}"

    condition_order = [
        "FABRIC_NOMINAL", "FABRIC_BOTTOM_RIGHT_ANCHOR", "FABRIC_BOTTOM_LEFT_ANCHOR",
        "FABRIC_FOUR_CORNERS_ONLY", "FABRIC_FULL_PERIMETER_FIXED", "FABRIC_WALL_GAP_02",
        "FABRIC_WALL_GAP_05", "FABRIC_WALL_GAP_08", "FABRIC_HEAVY_FREE", "FABRIC_HEAVY_CONSTRAINED",
        "EMPTY_CONTROL", "RIGID_CONTROL", "WALL_GAP_02_MATCHED_NO_WALL", "WALL_GAP_05_MATCHED_NO_WALL",
        "WALL_GAP_08_MATCHED_NO_WALL",
    ]
    execution_table = [
        "| Condition | Registered | Output | Runtime-valid | Protocol-match | Safety-only | Incomplete | Labels P/B/U | Actual FK m (min..max) | L/R runtime-valid |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for condition_id in condition_order:
        condition_rows = [row for row in rows if row["condition_id"] == condition_id]
        observed = [row for row in condition_rows if row["summary_exists"]]
        actuals = [float(row["actual_displacement_m"]) for row in observed if row["actual_displacement_m"] is not None]
        label_text = f"{sum(row['label']=='PASSABLE' for row in condition_rows)}/{sum(row['label']=='BLOCKED' for row in condition_rows)}/{sum(row['label']=='UNKNOWN' for row in condition_rows)}"
        actual_text = "-" if not actuals else f"{min(actuals):.3f}..{max(actuals):.3f}"
        side_text = ", ".join(
            f"{side[0].upper()}:{sum(row['runtime_valid'] for row in condition_rows if row['probe_side']==side)}"
            for side in ("left", "right")
        )
        execution_table.append(
            f"| {condition_id} | {len(condition_rows)} | {len(observed)} | {sum(row['runtime_valid'] for row in condition_rows)} | "
            f"{sum(row['protocol_match'] for row in condition_rows)} | {sum(row['probe_validation_only'] for row in condition_rows)} | "
            f"{sum(row['stop_reason']=='PROBE_INCOMPLETE' for row in condition_rows)} | {label_text} | {actual_text} | {side_text} |"
        )

    runtime_valid_n = sum(bool(row["runtime_valid"]) for row in rows)
    core_runtime_valid_n = sum(bool(row["runtime_valid"]) for row in core)
    core_labeled_n = sum(row["label"] in {"PASSABLE", "BLOCKED"} for row in core)
    core_incomplete_n = sum(row["scope"] == "core" and row["stop_reason"] == "PROBE_INCOMPLETE" for row in rows)
    protocol_valid_core_n = sum(row["scope"] == "core" and row["protocol_match"] for row in rows)
    safety_only_n = sum(bool(row["probe_validation_only"]) and bool(row["runtime_valid"]) for row in rows)
    if calibration_rows:
        calibration_summary = "; ".join(
            f"{row['side']}={row['actual_m']:.6f} m/{row['stop_reason']}/safe={row['safe_recovery']}"
            for row in calibration_rows
        )
        calibration_line = (
            f"- 独立全臂 FK 校准复核（不属于正式 manifest）：{calibration_summary}；右侧达到目标、左侧未达到，双侧校准门槛仍未通过，正式发射保持 withheld。"
        )
    else:
        calibration_line = "- 独立全臂 FK 校准复核：没有发现校准输出，不能据此解除正式发射门槛。"
    status = "COMPLETE" if core_labeled_n >= 200 else "INCOMPLETE_PROTOCOL_FEASIBILITY"
    lines = [
        "# Round 4070-D：固定 10 cm G1 本体通行性判断报告", "", "## 执行状态", "",
        f"- 研究状态：`{status}`；预注册 trial：{len(rows)}；核心目标：200 条有效重复。",
        "- 覆盖状态：`INSUFFICIENT_POSITIVE_COVERAGE`；没有用空场或刚板控制替代柔性障碍正负样本。",
        f"- 实际输出目录：{sum(bool(row['summary_exists']) for row in rows)}；runtime-valid：{runtime_valid_n}（核心 {core_runtime_valid_n}）；安全验证输出：{safety_only_n}。",
        f"- 核心协议匹配行：{protocol_valid_core_n}；核心有效物理标签：{core_labeled_n}（PASSABLE={sum(row['label']=='PASSABLE' for row in core)}，BLOCKED={sum(row['label']=='BLOCKED' for row in core)}）。",
        f"- 核心 `PROBE_INCOMPLETE`：{core_incomplete_n}；由于固定动作的实际 FK 未达到目标，按预注册规则保持 `UNKNOWN`，不自动判为 BLOCKED。",
        "- 固定协议：`PROBE_10CM`，目标 0.10 m，固定世界方向，单次连续轨迹，零重试；实际 FK、post-contact 位移和 stop reason 分开保存。",
        "- 安全验证结论：空场、刚板、正常布帘左右两侧均能安全撤回，但有符号实际 FK 位移约 -0.014 m 至 0.088 m（刚板受阻时可为负），未形成可用于 PASSABLE/BLOCKED 的完整 10 cm benchmark 行。",
        "- 独立几何复核：对当前对称肩 pitch/肘关节参考及固定 G1 URDF 做 FK 扫描，在允许的 step 区间 [0, 1.20] rad 内最大前向投影约 0.0889 m；要实现 0.10 m 必须重新预注册探测姿态/轨迹，不能靠追加力矩或重试补足。",
        calibration_line,
        "- 因此本轮没有启动其余核心正式重复；这是探测协议可执行性门禁触发的停止，不是因为发现少量误判而提前停止。", "",
        "## 实际运行分组", "",
        *execution_table, "",
        "标签顺序为 `PASSABLE/BLOCKED/UNKNOWN`；`Output` 包括安全校准/验证输出，不把它们当作正式 benchmark。", "",
        "## 方法对比", "", f"- E0：{metric_text(metric('E0'))}", f"- E1：{metric_text(metric('E1'))}", f"- E2：{metric_text(metric('E2'))}", "",
        "当前没有核心有效正负标签，故不报告准确率、false-pass、false-reject 或 recall；安全验证行上的 E0/E1/E2 仅作特征可计算性记录，不进入分母。", "",
        "## Q1-Q10", "",
        "- **Q1**：`NOT_ESTABLISHED`。没有足够的核心 PASSABLE/BLOCKED 标签，不能声称 E1 可靠或失败。",
        "- **Q2**：`NOT_ESTABLISHED`。E0 与 E1 在少量安全验证历史上可计算，但没有公平核心标签分母可比较。",
        "- **Q3/Q4**：`NOT_ESTABLISHED`。E2 尚未在有效正负样本上证明能同时降低两类错误。",
        "- **Q5**：本轮没有证据支持按材料分别调参；但也没有足够跨条件数据证明统一规则可泛化。",
        "- **Q6**：`NOT_ESTABLISHED`。没有完成 discovery/confirmation 的核心 seed 覆盖。",
        "- **Q7**：当前最直接的限制属于 Type E/协议执行边界：固定关节参考在安全运行下没有产生目标 0.10 m 的实际 FK 行程；不能把该缺口归因于分类器特征。",
        "- **Q8**：没有依据判断无需学习模型；在探测本身未形成有效标签前，学习模型结论也不成立。",
        "- **Q9**：没有合法匹配的 false-pass/false-reject 案例；`FALSE_PASS_CASES.md` 与 `FALSE_REJECT_CASES.md` 保持空案例审计。",
        "- **Q10**：当前证据不足以直接进入本体历史学习。下一步应先完成可重复的 0.10 m 探测标定，或正式接受 `PROBE_INCOMPLETE -> UNKNOWN` 并重新定义可评价问题。", "",
        "PVC 和多片中开帘没有独立 physics gate，因此不进入 benchmark。",
    ]
    (RESEARCH / "FINAL_REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def integrity() -> None:
    names = [
        "FIXED_10CM_PROBE_VALIDATION.md", "SCENE_MATRIX.csv", "PHYSICS_VALIDATION.md", "TRIAL_MANIFEST.csv",
        "LABEL_AUDIT.csv", "ENGINEERING_BASELINE_COMPARISON.csv", "FALSE_PASS_CASES.md", "FALSE_REJECT_CASES.md",
        "INFORMATION_SUFFICIENCY_ANALYSIS.md", "GENERALIZATION_RESULTS.md", "FINAL_REPORT_ZH.md", "RUN_INSTRUCTIONS.md",
        "FORMAL_LAUNCH_DECISION.md", "PROBE_CALIBRATION_RECHECK.md",
    ]
    rows, hashes = [], []
    for name in names:
        path = RESEARCH / name
        digest, size = None, None
        if path.is_file():
            data = path.read_bytes()
            digest, size = hashlib.sha256(data).hexdigest(), len(data)
            hashes.append({"path": str(path), "sha256": digest, "size_bytes": size})
        rows.append({"artifact": name, "exists": path.is_file(), "size_bytes": size, "sha256": digest})
    write_csv(RESEARCH / "INTEGRITY.csv", rows)
    write_csv(RESEARCH / "ARTIFACT_HASHES.csv", hashes)
    write_json(RESEARCH / "integrity_report.json", {
        "schema": "lift-the-curtain.round4070-fixed10cm-integrity.v1",
        "required_artifacts": len(names), "present_artifacts": sum(bool(row["exists"]) for row in rows),
        "deployment_truth_leakage_audit": "LABEL_AUDIT.csv; truth_used_online=false",
        "status": "PASS" if all(row["exists"] for row in rows) else "INCOMPLETE",
    })


def main() -> None:
    manifest = load_manifest()
    raw, baselines = collect_trials(manifest)
    rows, audits = evaluate_rows(raw, baselines)
    baseline_features = {side: feature_dict(history, history) for side, history in baselines.items()}
    metric_rows: list[dict[str, Any]] = []
    methods = ["ALWAYS_PASSABLE", "ALWAYS_BLOCKED", "E0", "E1", "E2"]
    for split in ("all", "discovery", "frozen_confirmation"):
        for scope in ("core", "all"):
            selected = []
            for row in rows:
                copy = dict(row)
                copy["split"] = "all" if split == "all" else row["split"]
                copy["scope"] = "all" if scope == "all" else row["scope"]
                selected.append(copy)
            for method in methods:
                metric_rows.append(metrics(selected, method, split, scope))
    for family in sorted({str(row["condition_family"]) for row in rows if row["scope"] == "core"}):
        for side in ("left", "right"):
            selected = [dict(row, split="all", scope="all") for row in rows if row["condition_family"] == family and row["probe_side"] == side and row["scope"] == "core"]
            for method in methods:
                item = metrics(selected, method, "all", "all")
                item.update({"condition_family": family, "probe_side": side})
                metric_rows.append(item)
    write_csv(RESEARCH / "LABEL_AUDIT.csv", audits)
    write_csv(RESEARCH / "ENGINEERING_BASELINE_COMPARISON.csv", metric_rows)
    merged = []
    by_id = {row["trial_id"]: row for row in rows}
    for plan in manifest:
        row = by_id.get(plan["trial_id"], {})
        merged.append({
            **plan,
            "runtime_valid": row.get("runtime_valid"),
            "physics_status": row.get("physics_status"),
            "actual_displacement_m": row.get("actual_displacement_m"),
            "post_contact_displacement_m": row.get("post_contact_displacement_m"),
            "stop_reason": row.get("stop_reason"),
            "label": row.get("label"),
            "label_reason": row.get("label_reason"),
            "protocol_match": row.get("protocol_match"),
            "protocol_mismatch_detail": row.get("protocol_detail"),
            "probe_validation_only": row.get("probe_validation_only"),
            "benchmark_eligible": row.get("benchmark_eligible"),
        })
    write_csv(RESEARCH / "TRIAL_MANIFEST.csv", merged)
    write_json(RESEARCH / "stats" / "evaluator_rows.json", rows)
    write_json(RESEARCH / "stats" / "e2_calibration.json", {
        "schema": "round4070-d-e2-calibration.v1", "target_displacement_m": TARGET_M,
        "threshold_source": "pre-registered fixed thresholds; only empty residual baseline is discovery-derived",
        "thresholds": {"pass_progress_m": E2_PASS_PROGRESS_M, "blocked_progress_m": E2_BLOCK_PROGRESS_M, "blocked_ratio": E2_BLOCK_RATIO, "blocked_stall_s": E2_STALL_S, "blocked_q_residual_rms": E2_Q_RESIDUAL_RAD, "e0_q_error_threshold": E0_Q_ERROR_THRESHOLD},
        "normal_baseline_sides": sorted(baseline_features), "frozen_before_confirmation": True,
    })
    write_supporting_reports(rows, metric_rows)
    integrity()
    print(json.dumps({"manifest_trials": len(manifest), "trial_rows": len(rows), "runtime_valid": sum(bool(row["runtime_valid"]) for row in rows), "labels": {label: sum(row["label"] == label for row in rows) for label in ("PASSABLE", "BLOCKED", "UNKNOWN", "CONTROL_PASSABLE", "CONTROL_BLOCKED")}, "baseline_sides": sorted(baselines)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
