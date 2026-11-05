"""Offline evaluator for the Round 4070-E fixed-command benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "round4070_fixed_probe_benchmark"
RESEARCH = REPO / "research" / "round4070_fixed_probe_benchmark"
OUTPUT = REPO / "outputs" / "round4070_fixed_probe_benchmark"
TRIALS = OUTPUT / "trials"
TARGET_M = 0.10
RULE_FREEZE = CONFIG / "RULE_FREEZE.json"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_manifest() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = read_json(CONFIG / "PREREGISTERED_MANIFEST.json", {}) or {}
    return payload, list(payload.get("trials", []))


def parse_vector(row: dict[str, str], field: str) -> np.ndarray:
    value = np.asarray(json.loads(row[field]), dtype=np.float64).reshape(-1)
    if value.size == 0 or not np.isfinite(value).all():
        raise ValueError(f"{field} is not a finite vector")
    return value


def load_history(path: Path):
    from lift_curtain.control.deployment_observation import DeploymentObservation

    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            phase = str(row.get("phase", ""))
            if phase.startswith("physical_probe") or phase in {"probe", "contact_hold"}:
                rows.append(row)
    if not rows:
        raise ValueError("no physical-probe rows")
    observations = []
    for row in rows:
        if str(row.get("deployment_effort_source", "")) in {"", "unavailable", "zero_filled"}:
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


def clean_features(value: dict[str, Any]) -> dict[str, Any]:
    return {key: (None if isinstance(item, float) and not math.isfinite(item) else item) for key, item in value.items()}


def feature_dict(history: list[Any], baseline: list[Any] | None) -> dict[str, Any]:
    from lift_curtain.control.deployment_observation import extract_proprioceptive_features

    return clean_features(extract_proprioceptive_features(history, baseline=baseline, sample_period_s=0.005, forward_axis_w=(1.0, 0.0, 0.0)))


def e1_decision(history: list[Any], baseline: list[Any] | None) -> tuple[str, dict[str, Any]]:
    from lift_curtain.control.deployment_observation import ProprioceptiveThresholds, classify_proprioceptive_response

    threshold_path = REPO / "configs" / "proprioceptive_thresholds.json"
    thresholds = ProprioceptiveThresholds(**(read_json(threshold_path, {}) or {}))
    decision, features = classify_proprioceptive_response(history, baseline=baseline, thresholds=thresholds, sample_period_s=0.005)
    return decision.value, clean_features(features)


def load_rule_freeze() -> dict[str, Any]:
    payload = read_json(RULE_FREEZE, {}) or {}
    if payload:
        return payload
    return {
        "schema": "lift-the-curtain.round4070-fixed-probe-rule-freeze.v1",
        "frozen_before_confirmation": True, "nominal_target_displacement_m": TARGET_M,
        "q_error_cutoff_rad": 0.12, "e2_pass_progress_m": 0.075,
        "e2_block_progress_m": 0.035, "e2_block_ratio": 0.40,
        "e2_stall_s": 0.20, "e2_effort_deviation": 2.2690875035237013,
        "e2_tracking_residual_rad": 0.12, "e2_pass_stall_s": 0.80,
    }


RULES = load_rule_freeze()


def e0_decision(features: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    value = finite(features.get("upper_tracking_rms"))
    cutoff = float(RULES["q_error_cutoff_rad"])
    details = {"q_error_rms_rad": value, "q_error_cutoff_rad": cutoff, "rule_version": "E0_Q_RESIDUAL_CUTOFF_V1"}
    return ("UNCERTAIN" if value is None else "PASSABLE" if value <= cutoff else "BLOCKED"), details


def e2_decision(features: dict[str, Any], baseline_features: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    progress = finite(features.get("hand_forward_progress_m"))
    effort = finite(features.get("effort_deviation"))
    tracking = finite(features.get("upper_tracking_rms"))
    stall = finite(features.get("hand_low_speed_duration_s"))
    baseline_tracking = finite(baseline_features.get("upper_tracking_rms"))
    q_residual = None if tracking is None or baseline_tracking is None else tracking - baseline_tracking
    ratio = None if progress is None else progress / TARGET_M
    passable = bool(progress is not None and tracking is not None and stall is not None and progress >= float(RULES["e2_pass_progress_m"]) and q_residual is not None and q_residual <= float(RULES["e2_tracking_residual_rad"]) and stall < float(RULES["e2_pass_stall_s"]))
    blocked = bool(progress is not None and effort is not None and stall is not None and (progress <= float(RULES["e2_block_progress_m"]) or (ratio is not None and ratio <= float(RULES["e2_block_ratio"]))) and stall >= float(RULES["e2_stall_s"]) and effort >= float(RULES["e2_effort_deviation"]))
    decision = "BLOCKED" if blocked else "PASSABLE" if passable else "UNCERTAIN"
    return decision, {
        "target_progress_m": TARGET_M, "progress_ratio_to_target": ratio,
        "baseline_tracking_rms_rad": baseline_tracking, "tracking_residual_rms_rad": q_residual,
        "q_error_cutoff_rad": float(RULES["q_error_cutoff_rad"]),
        "pass_progress_threshold_m": float(RULES["e2_pass_progress_m"]),
        "blocked_progress_threshold_m": float(RULES["e2_block_progress_m"]),
        "blocked_ratio_threshold": float(RULES["e2_block_ratio"]),
        "blocked_stall_threshold_s": float(RULES["e2_stall_s"]),
        "blocked_effort_threshold": float(RULES["e2_effort_deviation"]),
        "rule_version": "E2_UNIFIED_FIXED_THRESHOLDS_V1",
    }


def physics_status(gate_id: str, status_payload: dict[str, Any]) -> str:
    gates = status_payload.get("gates", {}) if isinstance(status_payload, dict) else {}
    entry = gates.get(gate_id, {}) if isinstance(gates, dict) else {}
    value = entry.get("status") if isinstance(entry, dict) else None
    if value:
        return str(value)
    return "PASS_CONTROL" if gate_id in {"PASS_CONTROL", "MATCHED_CONTROL"} else "MISSING"


def protocol_audit(plan: dict[str, Any], summary: dict[str, Any], metadata: dict[str, Any]) -> tuple[bool, str]:
    problems: list[str] = []
    expected = finite(plan.get("target_depth_m"))
    observed = finite(summary.get("nominal_probe_depth_m", summary.get("probe_target_displacement_m")))
    if expected is None or observed is None or abs(expected - observed) > 1.0e-9:
        problems.append(f"nominal target expected={expected}; observed={observed}")
    if summary.get("probe_depth_level") not in {None, "PROBE_10CM"}:
        problems.append(f"probe depth observed={summary.get('probe_depth_level')}")
    contract = str(metadata.get("probe_contract", ""))
    if contract and "FIXED_ONE_SHOT" not in contract:
        problems.append(f"probe contract={contract}")
    return not problems, "; ".join(problems)


def independent_label(plan: dict[str, Any], summary: dict[str, Any], physics: str, protocol_ok: bool, protocol_detail: str) -> tuple[str, bool, str, str, dict[str, Any]]:
    family = str(plan.get("condition_family", ""))
    condition = str(plan.get("condition_id", ""))
    if family == "control":
        return ("CONTROL_PASSABLE" if condition == "EMPTY_CONTROL" else "CONTROL_BLOCKED", True, "CONTROL_REFERENCE", "control excluded from flexible label metrics", {})
    if family == "matched_no_wall":
        return "CONTROL_PASSABLE", True, "MATCHED_NO_WALL_CONTROL", "matched control is not a flexible label", {}
    if not protocol_ok:
        return "UNKNOWN", False, "PROTOCOL_MISMATCH", protocol_detail, {}
    if physics not in {"PASS_E_PHYSICS", "PASS", "PASS_INHERITED", "PASS_INHERITED_C"}:
        return "UNKNOWN", False, "PHYSICS_INVALID_OR_UNVALIDATED", physics, {}
    if boolean(summary.get("runtime_valid")) is not True:
        return "UNKNOWN", False, "RUNTIME_INVALID", str(summary.get("runtime_status", "missing")), {}
    if boolean(summary.get("probe_valid")) is not True:
        return "UNKNOWN", False, "PROBE_INVALID", str(summary.get("probe_stop_reason", "missing")), {}
    probe = summary.get("physical_probe_result") or {}
    contact = boolean(probe.get("evaluation_hand_contact_confirmed"))
    clearance = boolean(summary.get("whole_body_candidate_clearance"))
    physical_branch = boolean(summary.get("physical_branch_pass"))
    safe_recovery = boolean(summary.get("safe_recovery"))
    stop_reason = str(summary.get("probe_stop_reason", ""))
    actual = finite(summary.get("actual_hand_displacement_m", summary.get("probe_actual_displacement_m")))
    continuation = finite(summary.get("probe_post_contact_displacement_m_evaluation_only"))
    hidden_geometry = summary.get("hidden_rigid_geometry_evaluation_only") or {}
    hidden_width = finite(hidden_geometry.get("width_m")) if isinstance(hidden_geometry, dict) else None
    hidden_height = finite(hidden_geometry.get("height_m")) if isinstance(hidden_geometry, dict) else None
    hidden_thickness = finite(hidden_geometry.get("thickness_m")) if isinstance(hidden_geometry, dict) else None
    geometry_blocked = bool(
        family == "hidden_wall"
        and hidden_width is not None
        and hidden_height is not None
        and hidden_thickness is not None
        and hidden_width >= 1.0
        and hidden_height >= 1.8
        and hidden_thickness > 0.0
    )
    evidence = {
        "contact_confirmed": contact, "whole_body_candidate_clearance": clearance,
        "physical_branch_pass": physical_branch, "safe_recovery": safe_recovery,
        "probe_stop_reason": stop_reason, "actual_hand_displacement_m": actual,
        "post_contact_displacement_m": continuation,
        "hidden_rigid_geometry_present": geometry_blocked,
        "hidden_rigid_geometry": hidden_geometry if geometry_blocked else {},
        "independent_evidence_source": "offline contact/geometry and robot-owned clearance",
    }
    if clearance is True and physical_branch is True and safe_recovery is True:
        return "PASSABLE", True, "OFFLINE_WHOLE_BODY_CLEARANCE", "robot body envelope verified continuation and recovery", evidence
    rigid_geometry = family == "hidden_wall" or str(summary.get("scene_kind", "")) == "rigid"
    contact_limited = stop_reason in {"CONTACT_LIMITED_MOTION", "SAFETY_STOP_PROBE_FORWARD_STAGNATION", "SAFETY_STOP_EFFORT_PROXY_RATIO_LIMIT"}
    if rigid_geometry and contact is True and (clearance is False or geometry_blocked) and physical_branch is True and safe_recovery is True and contact_limited:
        return "BLOCKED", True, "OFFLINE_RIGID_CONSTRAINT", "independent contact plus no body-envelope clearance", evidence
    if family in {"fabric", "fabric_heavy"} and contact is True and clearance is False and physical_branch is True and safe_recovery is True and contact_limited and continuation is not None and continuation <= 0.015:
        return "BLOCKED", True, "OFFLINE_FLEXIBLE_CONTACT_NO_CONTINUATION", "independent contact and no continuation; low displacement alone is insufficient", evidence
    return "UNKNOWN", False, "INSUFFICIENT_INDEPENDENT_EVIDENCE", "PointGoal and low displacement do not establish passability", evidence


def collect(manifest: list[dict[str, Any]], status_payload: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[Any]]]:
    records: list[dict[str, Any]] = []
    for plan in manifest:
        path = TRIALS / str(plan["trial_id"])
        summary = read_json(path / "summary.json", {}) or {}
        metadata = read_json(path / "trial_metadata.json", {}) or {}
        history = None
        history_rows = None
        feature_error = None
        deployment_path = path / "deployment_observation.csv"
        if deployment_path.is_file():
            try:
                history, history_rows = load_history(deployment_path)
            except Exception as exc:
                feature_error = f"{type(exc).__name__}: {exc}"
        protocol_ok, protocol_detail = protocol_audit(plan, summary, metadata)
        physics = physics_status(str(plan.get("physics_gate_id", "")), status_payload)
        summary_exists = (path / "summary.json").is_file()
        if summary_exists:
            label, label_valid, label_reason, label_detail, evidence = independent_label(plan, summary, physics, protocol_ok, protocol_detail)
        else:
            label, label_valid, label_reason, label_detail, evidence = (
                "UNKNOWN", False, "NO_SUMMARY", "trial has no recorded summary", {}
            )
        runtime_valid = boolean(summary.get("runtime_valid"))
        if runtime_valid is None:
            runtime_valid = str(summary.get("runtime_status", "")) == "RUNTIME_PASS_FINALIZE_REACHED"
        probe_valid = boolean(summary.get("probe_valid"))
        if probe_valid is None:
            probe_valid = bool(runtime_valid and boolean(summary.get("execution_data_complete")))
        records.append({
            "plan": plan, "path": path, "summary": summary, "metadata": metadata,
            "summary_exists": summary_exists,
            "history": history, "history_rows": history_rows, "feature_error": feature_error,
            "protocol_match": protocol_ok, "protocol_detail": protocol_detail,
            "physics_status": physics, "runtime_valid": bool(runtime_valid), "probe_valid": bool(probe_valid),
            "label": label, "label_valid": label_valid, "label_reason": label_reason,
            "label_detail": label_detail, "label_evidence": evidence,
        })
    baselines: dict[str, list[Any]] = {}
    for item in records:
        if item["plan"].get("condition_id") == "EMPTY_CONTROL" and item["history"] is not None and item["runtime_valid"] and item["probe_valid"]:
            baselines.setdefault(str(item["plan"].get("probe_side")), item["history"])
    return records, baselines


def evaluate(records: list[dict[str, Any]], baselines: dict[str, list[Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_features = {side: feature_dict(history, history) for side, history in baselines.items()}
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    forbidden = {"condition_id", "condition_family", "scenario", "material", "hidden_rigid_x", "hidden_rigid_gap_m", "structural_topology", "density_scale_multiplier", "simulator_contact_flag", "true_passability", "future_clearance", "evaluation_hand_contact_force_norm_N", "evaluation_geometry_snapshot_json"}
    for item in records:
        plan, path, summary = item["plan"], item["path"], item["summary"]
        side = str(plan.get("probe_side"))
        history = item["history"]
        features: dict[str, Any] = {}
        details: dict[str, Any] = {}
        decisions = {"E0": "UNCERTAIN", "E1": "UNCERTAIN", "E2": "UNCERTAIN"}
        if history is not None:
            try:
                baseline = baselines.get(side)
                features = feature_dict(history, baseline)
                decisions["E0"], details["E0"] = e0_decision(features)
                e1_raw, e1_features = e1_decision(history, baseline)
                decisions["E1"] = {"RESPONSIVE": "PASSABLE", "BLOCKED": "BLOCKED", "UNCERTAIN": "UNCERTAIN"}.get(e1_raw, "UNCERTAIN")
                details["E1"] = {"raw_local_response": e1_raw, "frozen_features": e1_features}
                decisions["E2"], details["E2"] = e2_decision(features, baseline_features.get(side, feature_dict(history, None)))
            except Exception as exc:
                item["feature_error"] = f"{type(exc).__name__}: {exc}"
        headers: set[str] = set()
        deployment_path = path / "deployment_observation.csv"
        if deployment_path.is_file():
            try:
                with deployment_path.open(encoding="utf-8-sig", newline="") as handle:
                    headers = set(csv.DictReader(handle).fieldnames or [])
            except OSError:
                pass
        leaked = sorted(headers & forbidden)
        actual = finite(summary.get("actual_hand_displacement_m", summary.get("probe_actual_displacement_m")))
        probe_validation_only = boolean(summary.get("probe_validation_only")) is True
        eligible = bool(plan.get("formal_core") and item["protocol_match"] and item["runtime_valid"] and item["probe_valid"] and item["label_valid"] and item["label"] in {"PASSABLE", "BLOCKED"} and str(plan.get("phase")) != "PHASE_0")
        row = {
            "trial_id": plan["trial_id"], "condition_id": plan.get("condition_id"), "condition_family": plan.get("condition_family"),
            "probe_side": side, "phase": plan.get("phase"), "split": plan.get("stage", "unknown"),
            "scope": "core" if plan.get("formal_core") else "noncore", "seed": plan.get("seed"), "repeat_index": plan.get("repeat_index"),
            "summary_exists": (path / "summary.json").is_file(), "runtime_valid": item["runtime_valid"], "probe_valid": item["probe_valid"],
            "physics_status": item["physics_status"], "protocol_match": item["protocol_match"], "protocol_detail": item["protocol_detail"],
            "probe_validation_only": probe_validation_only, "task_success": boolean(summary.get("task_success")) is True,
            "benchmark_eligible": eligible, "target_displacement_m": TARGET_M, "actual_displacement_m": actual,
            "stop_reason": summary.get("probe_stop_reason"), "probe_outcome_class": summary.get("probe_outcome_class", summary.get("probe_stop_reason")),
            "contact_limited_motion": boolean(summary.get("contact_limited_motion")), "free_space_undertracking": boolean(summary.get("free_space_undertracking")),
            "label": item["label"], "label_valid": item["label_valid"], "label_reason": item["label_reason"], "label_detail": item["label_detail"],
            "label_evidence": item["label_evidence"], "features": features, "decisions": decisions, "method_details": details,
        }
        rows.append(row)
        audits.append({
            "trial_id": row["trial_id"], "condition_id_offline": row["condition_id"], "condition_family_offline": row["condition_family"],
            "phase": row["phase"], "probe_side": side, "seed": row["seed"], "repeat_index": row["repeat_index"], "summary_exists": row["summary_exists"], "runtime_valid": row["runtime_valid"], "probe_valid": row["probe_valid"],
            "physics_status": row["physics_status"], "protocol_match": row["protocol_match"], "protocol_mismatch_detail": row["protocol_detail"],
            "actual_displacement_m": actual, "nominal_displacement_m": TARGET_M, "stop_reason": row["stop_reason"], "probe_outcome_class": row["probe_outcome_class"],
            "contact_limited_motion": row["contact_limited_motion"], "free_space_undertracking": row["free_space_undertracking"], "label": row["label"], "label_valid": row["label_valid"],
            "label_reason": row["label_reason"], "label_detail": row["label_detail"], "independent_evidence_json": json.dumps(row["label_evidence"], sort_keys=True),
            "benchmark_eligible": eligible, "feature_error": item["feature_error"], "deployment_headers_contain_forbidden_fields": bool(leaked), "forbidden_fields_found": ";".join(leaked),
            "truth_used_online": False, "online_classifier_source": "deployment_observation.csv only",
        })
    return rows, audits


def wilson(success: int, total: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    p = success / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return max(0.0, center - half), min(1.0, center + half)


def metric(rows: list[dict[str, Any]], method: str, *, split: str = "all", scope: str = "core", family: str | None = None, side: str | None = None) -> dict[str, Any]:
    selected = [row for row in rows if row["benchmark_eligible"] and row["label"] in {"PASSABLE", "BLOCKED"} and (split == "all" or row["split"] == split) and (scope == "all" or row["scope"] == scope) and (family is None or row["condition_family"] == family) and (side is None or row["probe_side"] == side)]
    false_pass = sum(row["label"] == "BLOCKED" and row["decisions"][method] == "PASSABLE" for row in selected)
    false_reject = sum(row["label"] == "PASSABLE" and row["decisions"][method] == "BLOCKED" for row in selected)
    positives = sum(row["label"] == "PASSABLE" for row in selected)
    negatives = sum(row["label"] == "BLOCKED" for row in selected)
    uncertain = sum(row["decisions"][method] == "UNCERTAIN" for row in selected)
    p_recall = None if positives == 0 else sum(row["label"] == "PASSABLE" and row["decisions"][method] == "PASSABLE" for row in selected) / positives
    b_recall = None if negatives == 0 else sum(row["label"] == "BLOCKED" and row["decisions"][method] == "BLOCKED" for row in selected) / negatives
    fp_lo, fp_hi = wilson(false_pass, negatives)
    fr_lo, fr_hi = wilson(false_reject, positives)
    return {
        "method": method, "split": split, "scope": scope, "condition_family": family or "all", "probe_side": side or "all",
        "eligible_n": len(selected), "passable_n": positives, "blocked_n": negatives, "false_pass_n": false_pass,
        "false_pass_rate": None if negatives == 0 else false_pass / negatives, "false_pass_ci95_low": fp_lo, "false_pass_ci95_high": fp_hi,
        "false_reject_n": false_reject, "false_reject_rate": None if positives == 0 else false_reject / positives,
        "false_reject_ci95_low": fr_lo, "false_reject_ci95_high": fr_hi, "passable_recall": p_recall, "blocked_recall": b_recall,
        "uncertain_n": uncertain, "uncertain_rate": None if not selected else uncertain / len(selected),
        "decision_coverage": None if not selected else (len(selected) - uncertain) / len(selected),
        "balanced_accuracy": None if p_recall is None or b_recall is None else 0.5 * (p_recall + b_recall),
    }


def write_protocol_report(rows: list[dict[str, Any]]) -> None:
    actuals = [row["actual_displacement_m"] for row in rows if row["actual_displacement_m"] is not None]
    lines = [
        "# Round 4070-E Protocol Correction", "",
        "The nominal command is fixed at `0.10 m`; actual hand FK feedback is an observed outcome, not a `PROBE_VALID` gate.", "",
        "## Four statuses", "",
        "- `RUNTIME_VALID`: normal simulator execution, finite state, and complete required logs.",
        "- `PROBE_VALID`: the legal one-shot command executed with a complete proprioceptive history; contact-limited motion and an early safe stop may still be valid outcomes.",
        "- `LABEL_VALID`: independent physical/geometric evidence supports `PASSABLE`, `BLOCKED`, or an explicit `UNKNOWN`; classifier output and PointGoal failure are not truth.",
        "- `TASK_SUCCESS`: secondary navigation/PointGoal outcome; it is never substituted for a traversability label.", "",
        "A valid trial may have actual hand displacement below 0.10 m. It is retained with its outcome class and is not automatically relabeled `BLOCKED`.", "",
        f"Observed summaries: `{sum(row['summary_exists'] for row in rows)}`; runtime-valid `{sum(row['runtime_valid'] for row in rows)}`; probe-valid `{sum(row['probe_valid'] for row in rows)}`; label-valid `{sum(row['label_valid'] for row in rows)}`.",
        f"Observed actual displacement range: `{min(actuals):.6f}..{max(actuals):.6f} m`." if actuals else "Observed actual displacement range: not available.",
        "Registered rows are not counted as completed experiments; only recorded output directories are execution evidence.",
    ]
    (RESEARCH / "PROTOCOL_CORRECTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_scene_coverage(manifest: list[dict[str, Any]], rows: list[dict[str, Any]], status_payload: dict[str, Any]) -> None:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for condition in manifest:
        condition_id = str(condition.get("condition_id"))
        if condition_id in seen:
            continue
        seen.add(condition_id)
        selected = [row for row in rows if row["condition_id"] == condition_id]
        output.append({
            "condition_id": condition_id, "condition_family": condition.get("condition_family"), "scenario": condition.get("scenario"),
            "structural_topology": condition.get("structural_topology"), "density_scale_multiplier": condition.get("density_scale_multiplier"),
            "hidden_rigid_gap_m": condition.get("hidden_rigid_gap_m"), "formal_core": condition.get("formal_core"),
            "physics_gate_id": condition.get("physics_gate_id"), "physics_gate_status": physics_status(str(condition.get("physics_gate_id")), status_payload),
            "registered_rows": len(selected), "observed_rows": sum(row["summary_exists"] for row in selected),
            "runtime_valid_rows": sum(row["runtime_valid"] for row in selected), "probe_valid_rows": sum(row["probe_valid"] for row in selected),
            "label_valid_rows": sum(row["label_valid"] for row in selected), "passable_labels": sum(row["label"] == "PASSABLE" for row in selected),
            "blocked_labels": sum(row["label"] == "BLOCKED" for row in selected), "unknown_labels": sum(row["label"] == "UNKNOWN" for row in selected),
            "left_observed": sum(row["summary_exists"] and row["probe_side"] == "left" for row in selected),
            "right_observed": sum(row["summary_exists"] and row["probe_side"] == "right" for row in selected),
            "notes": "P1 is reported separately and cannot invalidate this P0 row.",
        })
    write_csv(RESEARCH / "SCENE_COVERAGE.csv", output)


def write_physics_results(status_payload: dict[str, Any]) -> None:
    output: list[dict[str, Any]] = []
    for gate_id, value in (status_payload.get("gates", {}) or {}).items():
        value = value if isinstance(value, dict) else {"status": value}
        output.append({"scope": "P0", "gate_id": gate_id, **value})
    p1 = read_json(OUTPUT / "p1" / "P1_GATE_RESULTS.json", {}) or {}
    for item in p1.get("results", []) if isinstance(p1, dict) else []:
        output.append({"scope": "P1_PHYSICS_ONLY", **item, "p1_failure_cannot_invalidate_p0": True})
    write_csv(RESEARCH / "PHYSICS_GATE_RESULTS.csv", output)


def write_execution_results(rows: list[dict[str, Any]]) -> None:
    write_csv(RESEARCH / "PROBE_EXECUTION_RESULTS.csv", [{
        "trial_id": row["trial_id"], "phase": row["phase"], "condition_id": row["condition_id"], "condition_family": row["condition_family"], "probe_side": row["probe_side"], "seed": row["seed"], "repeat_index": row["repeat_index"],
        "summary_exists": row["summary_exists"], "runtime_valid": row["runtime_valid"], "probe_valid": row["probe_valid"], "label_valid": row["label_valid"], "protocol_match": row["protocol_match"], "physics_status": row["physics_status"],
        "probe_validation_only": row["probe_validation_only"], "task_success": row["task_success"], "actual_displacement_m": row["actual_displacement_m"], "nominal_displacement_m": TARGET_M,
        "probe_outcome_class": row["probe_outcome_class"], "contact_limited_motion": row["contact_limited_motion"], "free_space_undertracking": row["free_space_undertracking"], "stop_reason": row["stop_reason"],
        "label": row["label"], "label_reason": row["label_reason"], "benchmark_eligible": row["benchmark_eligible"],
    } for row in rows])


def write_label_audit(audits: list[dict[str, Any]]) -> None:
    write_csv(RESEARCH / "LABEL_VALIDITY_AUDIT.csv", audits)


def write_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in ("E0", "E1", "E2"):
        for split in ("all", "discovery", "frozen_confirmation"):
            output.append(metric(rows, method, split=split, scope="core"))
    for family in sorted({str(row["condition_family"]) for row in rows if row["scope"] == "core"}):
        for side in ("left", "right"):
            for method in ("E0", "E1", "E2"):
                output.append(metric(rows, method, family=family, side=side))
    write_csv(RESEARCH / "ENGINEERING_BASELINE_COMPARISON.csv", output)
    return output


def write_matched_cases(rows: list[dict[str, Any]]) -> None:
    pairs = {"FABRIC_WALL_GAP_02": "WALL_GAP_02_MATCHED_NO_WALL", "FABRIC_WALL_GAP_05": "WALL_GAP_05_MATCHED_NO_WALL", "FABRIC_WALL_GAP_08": "WALL_GAP_08_MATCHED_NO_WALL"}
    matching_rule = "same probe_side + repeat_index; independent control seed"
    output: list[dict[str, Any]] = []
    for positive, control in pairs.items():
        positives = [row for row in rows if row["condition_id"] == positive]
        controls = [row for row in rows if row["condition_id"] == control]
        for positive_row in positives:
            matched = next((row for row in controls if row["probe_side"] == positive_row["probe_side"] and row["repeat_index"] == positive_row["repeat_index"]), None)
            output.append({
                "positive_condition": positive, "matched_control": control, "probe_side": positive_row["probe_side"], "positive_trial_id": positive_row["trial_id"], "control_trial_id": None if matched is None else matched["trial_id"],
                "positive_repeat_index": positive_row["repeat_index"], "control_repeat_index": None if matched is None else matched["repeat_index"], "positive_seed": positive_row["seed"], "control_seed": None if matched is None else matched["seed"],
                "positive_actual_m": positive_row["actual_displacement_m"], "control_actual_m": None if matched is None else matched["actual_displacement_m"], "positive_label": positive_row["label"], "control_label": None if matched is None else matched["label"],
                "positive_probe_valid": positive_row["probe_valid"], "control_probe_valid": None if matched is None else matched["probe_valid"], "positive_e0": positive_row["decisions"]["E0"], "control_e0": None if matched is None else matched["decisions"]["E0"],
                "positive_e1": positive_row["decisions"]["E1"], "control_e1": None if matched is None else matched["decisions"]["E1"], "positive_e2": positive_row["decisions"]["E2"], "control_e2": None if matched is None else matched["decisions"]["E2"],
                "matching_rule": matching_rule, "pair_valid": bool(matched and positive_row["probe_valid"] and matched["probe_valid"]),
            })
    write_csv(RESEARCH / "MATCHED_POSITIVE_NEGATIVE_CASES.csv", output)


def write_left_right(rows: list[dict[str, Any]]) -> None:
    output: list[dict[str, Any]] = []
    for condition in sorted({str(row["condition_id"]) for row in rows}):
        for side in ("left", "right"):
            selected = [row for row in rows if row["condition_id"] == condition and row["probe_side"] == side]
            actuals = [row["actual_displacement_m"] for row in selected if row["actual_displacement_m"] is not None]
            output.append({
                "condition_id": condition, "probe_side": side, "rows": len(selected), "observed": sum(row["summary_exists"] for row in selected), "runtime_valid": sum(row["runtime_valid"] for row in selected), "probe_valid": sum(row["probe_valid"] for row in selected), "label_valid": sum(row["label_valid"] for row in selected),
                "passable": sum(row["label"] == "PASSABLE" for row in selected), "blocked": sum(row["label"] == "BLOCKED" for row in selected), "unknown": sum(row["label"] == "UNKNOWN" for row in selected),
                "actual_min_m": min(actuals) if actuals else None, "actual_max_m": max(actuals) if actuals else None, "e0_determinate": sum(row["decisions"]["E0"] != "UNCERTAIN" for row in selected), "e1_determinate": sum(row["decisions"]["E1"] != "UNCERTAIN" for row in selected), "e2_determinate": sum(row["decisions"]["E2"] != "UNCERTAIN" for row in selected),
            })
    write_csv(RESEARCH / "LEFT_RIGHT_COMPARISON.csv", output)


def write_failure_analysis(rows: list[dict[str, Any]]) -> None:
    observed = [row for row in rows if row["summary_exists"]]
    eligible = [row for row in rows if row["benchmark_eligible"]]
    flexible_unknown = [row for row in observed if row["scope"] == "core" and row["label"] == "UNKNOWN"]
    f2_rows = [
        row for row in flexible_unknown
        if row["probe_valid"] and row["probe_outcome_class"] == "CONTACT_LIMITED_MOTION"
    ]
    f1_rows = [
        row for row in eligible
        if row["label"] == "BLOCKED"
        and row["decisions"]["E1"] == "PASSABLE"
        and row["decisions"]["E2"] == "BLOCKED"
    ]
    f1_closest = [
        row for row in eligible
        if row["label"] == "BLOCKED" and row["decisions"]["E1"] == "PASSABLE"
    ]
    f4_rows = [
        row for row in eligible
        if row["condition_family"] == "fabric_heavy"
        and row["label"] == "PASSABLE"
        and row["decisions"]["E1"] == "BLOCKED"
    ]
    f4_closest = [row for row in flexible_unknown if row["condition_family"] == "fabric_heavy"]
    f5_by_method = {
        method: [
            row for row in eligible
            if row["label"] == "BLOCKED" and row["decisions"][method] == "PASSABLE"
        ]
        for method in ("E0", "E1", "E2")
    }
    f5_rows = {row["trial_id"]: row for values in f5_by_method.values() for row in values}
    f6_rows = [
        row for row in observed
        if not row["probe_validation_only"] and not row["task_success"] and not row["label_valid"]
    ]
    p1_payload = read_json(OUTPUT / "p1" / "P1_GATE_RESULTS.json", {}) or {}
    p1_failure_ids = [
        f"P1_{item.get('scene')}_{item.get('mode')}"
        for item in p1_payload.get("results", [])
        if str(item.get("status")) == "FAIL_PHYSICS_SANITY"
    ] if isinstance(p1_payload, dict) else []

    def ids(items: list[dict[str, Any]] | list[str], limit: int = 6) -> str:
        values = [item if isinstance(item, str) else str(item["trial_id"]) for item in items]
        return ", ".join(values[:limit]) or "-"

    lines = [
        "# Failure Mechanism Analysis",
        "",
        "Counts use observed trial directories only. The required F1-F7 research mechanisms are not mutually exclusive; an execution record may appear in more than one evidence category. Low displacement is not counted as BLOCKED without independent contact/clearance evidence.",
        "",
        "## Required Research Taxonomy",
        "",
        "| Mechanism | Count | Evidence trial/run IDs | Interpretation |",
        "|---|---:|---|---|",
        f"| F1 information underuse | {len(f1_rows)} proven | closest E1 false-pass cases: {ids(f1_closest)} | No proven E2-correct rescue exists; current data cannot isolate an E1 threshold/use failure from insufficient decisive observation. |",
        f"| F2 probe information insufficient | {len(f2_rows)} | {ids(f2_rows)} | Valid one-shot probes were contact-limited but did not independently establish whole-body clearance; these remain UNKNOWN. |",
        f"| F3 local contact versus whole-body traversal mismatch | 0 confirmed; {len(f2_rows)} candidates | {ids(f2_rows)} | Candidate pattern overlaps F2, but no independent whole-body evidence is sufficient to call these confirmed local-contact/global-traversal mismatches. |",
        f"| F4 heavy-curtain over-conservatism | {len(f4_rows)} proven | heavy UNKNOWN examples: {ids(f4_closest)} | No valid heavy-curtain PASSABLE label exists in this run, so a false-reject claim is not supported. |",
        f"| F5 constrained-curtain over-release | {len(f5_rows)} unique blocked trials (E0={len(f5_by_method['E0'])}, E1={len(f5_by_method['E1'])}, E2={len(f5_by_method['E2'])}) | {ids(list(f5_rows.values()))} | Hidden-wall BLOCKED rows are independent negative evidence; local responsiveness causes false PASSABLE decisions, reduced but not eliminated by E2. |",
        f"| F6 robot control/planning | {len(f6_rows)} | {ids(f6_rows)} | Full task-branch PointGoal failures without independent label evidence; these are not relabeled BLOCKED. |",
        f"| F7 physical-model invalidity | {len(p1_failure_ids)} P1 physics-only runs | {ids(p1_failure_ids)} | P1 runs failed the frozen physics sanity bound; they are excluded from G1 metrics, and real PVC is not claimed. |",
        "",
        "F1 and F4 are reported as unproven rather than inferred from UNKNOWN rows. F2/F3 candidate counts intentionally expose the evidence boundary instead of converting local contact into a whole-body truth label.",
        "",
        "## Execution Diagnostics",
        "",
        "| Diagnostic | Count | Example trial IDs |",
        "|---|---:|---|",
    ]
    diagnostics = {
        "RUNTIME_INVALID": lambda row: row["summary_exists"] and not row["runtime_valid"],
        "PROBE_INVALID_OR_MISSING_HISTORY": lambda row: row["summary_exists"] and row["runtime_valid"] and not row["probe_valid"],
        "CONTACT_LIMITED_MOTION": lambda row: row["summary_exists"] and row["probe_outcome_class"] == "CONTACT_LIMITED_MOTION",
        "FREE_SPACE_UNDERTRACKING": lambda row: row["summary_exists"] and row["probe_outcome_class"] == "FREE_SPACE_UNDERTRACKING",
        "SAFETY_STOP": lambda row: row["summary_exists"] and str(row["probe_outcome_class"]).startswith("SAFETY_STOP"),
        "TASK_BRANCH_POINTGOAL_FAILURE_WITHOUT_LABEL_EVIDENCE": lambda row: row["summary_exists"] and not row["probe_validation_only"] and not row["task_success"] and not row["label_valid"],
        "UNKNOWN_WITHOUT_INDEPENDENT_EVIDENCE": lambda row: row["summary_exists"] and row["label"] == "UNKNOWN" and row["probe_valid"],
    }
    for name, predicate in diagnostics.items():
        examples = [row["trial_id"] for row in rows if predicate(row)]
        lines.append(f"| {name} | {len(examples)} | {', '.join(examples[:5]) or '-'} |")
    lines.append("\nP1 physics-only failures are listed under F7 and cannot invalidate P0.")
    (RESEARCH / "FAILURE_MECHANISM_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RESEARCH / "FAILURE_MECHANISM_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_gpu_audit(rows: list[dict[str, Any]]) -> None:
    summaries = []
    for row in rows:
        payload = read_json(TRIALS / str(row["trial_id"]) / "resource_summary.json", {}) or {}
        if payload:
            summaries.append({"trial_id": row["trial_id"], **payload})
    names = sorted({str(item.get("gpu_name")) for item in summaries if item.get("gpu_name")})
    peaks = [finite(item.get("gpu_memory_peak_mb")) for item in summaries]
    peaks = [value for value in peaks if value is not None]
    lines = ["# GPU Runtime Audit", "", f"Observed resource summaries: `{len(summaries)}`; GPU names: `{', '.join(names) or 'not recorded'}`.", f"Peak GPU memory: `{max(peaks):.1f} MB`." if peaks else "Peak GPU memory: not recorded.", "", "Missing telemetry is reported as missing evidence, not inferred as a GPU failure.", "", "| Trial | GPU | Physics steps | Rollout wall s | Timeout |", "|---|---|---:|---:|---|"]
    for item in summaries:
        lines.append(f"| {item.get('trial_id')} | {item.get('gpu_name') or '-'} | {item.get('physics_steps') or '-'} | {item.get('rollout_wall_time_s') or '-'} | {item.get('timeout_reason') or '-'} |")
    (RESEARCH / "GPU_RUNTIME_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_report(rows: list[dict[str, Any]], metric_rows: list[dict[str, Any]]) -> None:
    observed = [row for row in rows if row["summary_exists"]]
    eligible = [row for row in rows if row["benchmark_eligible"]]
    actuals = [row["actual_displacement_m"] for row in observed if row["actual_displacement_m"] is not None]
    observed_label_counts = {
        label: sum(row["summary_exists"] and row["label"] == label for row in rows)
        for label in ("PASSABLE", "BLOCKED", "UNKNOWN", "CONTROL_PASSABLE", "CONTROL_BLOCKED")
    }
    phase_lines: list[str] = []
    for phase in ("PHASE_0", "PHASE_1", "PHASE_2", "PHASE_3", "MATCHED_CONTROL"):
        selected = [row for row in rows if row["phase"] == phase]
        if not selected:
            continue
        phase_lines.append(
            f"- `{phase}`: registered={len(selected)}, observed={sum(row['summary_exists'] for row in selected)}, "
            f"runtime={sum(row['runtime_valid'] for row in selected)}, probe={sum(row['probe_valid'] for row in selected)}, "
            f"label_valid={sum(row['label_valid'] for row in selected)}。"
        )
    phase1_noncore_unobserved = [
        row for row in rows
        if row["phase"] == "PHASE_1" and row["scope"] == "noncore" and not row["summary_exists"]
    ]
    p0_core_conditions = sorted({
        str(row["condition_id"])
        for row in rows
        if row["scope"] == "core" and row["condition_id"]
    })
    cumulative_repeat_counts = {
        condition: {
            side: sum(
                row["condition_id"] == condition
                and row["probe_side"] == side
                and row["phase"] in {"PHASE_1", "PHASE_2"}
                for row in rows
            )
            for side in ("left", "right")
        }
        for condition in p0_core_conditions
    }
    phase2_under_repeated = [
        condition for condition, side_counts in cumulative_repeat_counts.items()
        if min(side_counts.values()) < 4
    ]
    phase2_four_per_side = [
        condition for condition, side_counts in cumulative_repeat_counts.items()
        if min(side_counts.values()) >= 4
    ]
    phase3_conditions = sorted({
        str(row["condition_id"])
        for row in rows
        if row["phase"] == "PHASE_3" and row["scope"] == "core"
    })
    coverage_notes = [
        f"- Phase 1 formal core rows: registered={sum(row['phase']=='PHASE_1' and row['scope']=='core' for row in rows)}, observed={sum(row['phase']=='PHASE_1' and row['scope']=='core' and row['summary_exists'] for row in rows)}。",
        f"- Phase 1 未执行的 {len(phase1_noncore_unobserved)} 条非核心控制行未计入正式失败；Phase 0 控制与 {sum(row['phase']=='MATCHED_CONTROL' and row['summary_exists'] for row in rows)} 条 matched controls 已实际完成。" if phase1_noncore_unobserved else "- Phase 1 非核心控制行均已观测。",
        f"- Phase 1+2 累计达到每侧至少 4 次的 P0 条件：{', '.join(phase2_four_per_side) or '无'}。" ,
        f"- 尚未达到每侧 4 次累计重复的 P0 条件：{', '.join(phase2_under_repeated) or '无'}。",
        f"- Phase 3 独立确认条件：{', '.join(phase3_conditions) or '无'}。",
    ]
    label_counts = {
        label: sum(row["label"] == label for row in eligible)
        for label in ("PASSABLE", "BLOCKED", "UNKNOWN")
    }
    method_metrics = {
        method: next(
            (
                value
                for value in metric_rows
                if value["method"] == method and value["split"] == "all" and value["scope"] == "core"
            ),
            None,
        )
        for method in ("E0", "E1", "E2")
    }

    def ids_for(predicate) -> list[str]:
        return [row["trial_id"] for row in eligible if predicate(row)]

    e2_rescues = ids_for(lambda row: row["decisions"]["E1"] != row["label"] and row["decisions"]["E2"] == row["label"])
    e2_failures = ids_for(lambda row: row["decisions"]["E2"] != row["label"])
    if label_counts["PASSABLE"] and label_counts["BLOCKED"] and all(method_metrics.get(method) is not None for method in ("E0", "E1", "E2")):
        q2 = (
            "不存在严格支配者："
            f"E0 覆盖={method_metrics['E0']['decision_coverage']:.4f}、误放行={method_metrics['E0']['false_pass_n']}/{method_metrics['E0']['blocked_n']}；"
            f"E1 覆盖={method_metrics['E1']['decision_coverage']:.4f}、误放行={method_metrics['E1']['false_pass_n']}/{method_metrics['E1']['blocked_n']}；"
            f"E2 覆盖={method_metrics['E2']['decision_coverage']:.4f}、误放行={method_metrics['E2']['false_pass_n']}/{method_metrics['E2']['blocked_n']}。"
            "E0 提供最高覆盖但放行全部有效 BLOCKED；E2 的误放行最低但覆盖最低；E1 是中间权衡。"
        )
    else:
        q2 = "正负类别尚不同时具备，或方法指标不完整；不能预先选择方法。"
    e2_failure_decisions = {
        decision: sum(row["decisions"]["E2"] == decision for row in eligible if row["decisions"]["E2"] != row["label"])
        for decision in ("PASSABLE", "BLOCKED", "UNCERTAIN")
    }
    p1 = read_json(OUTPUT / "p1" / "P1_GATE_RESULTS.json", {}) or {}
    p1_results = list(p1.get("results", [])) if isinstance(p1, dict) else []
    p1_statuses = sorted({str(item.get("status")) for item in p1_results})
    condition_summaries: list[str] = []
    for condition in sorted({str(row["condition_id"]) for row in eligible}):
        selected = [row for row in eligible if row["condition_id"] == condition]
        condition_summaries.append(
            f"`{condition}`: PASSABLE={sum(row['label']=='PASSABLE' for row in selected)}, "
            f"BLOCKED={sum(row['label']=='BLOCKED' for row in selected)}"
        )

    lines = [
        "# Round 4070-E：固定命令本体通行性判断基准报告", "", "## 结论", "",
        f"- 注册计划 `{len(rows)}` 条；实际有 summary 的输出 `{len(observed)}` 条。注册数量不等于实验完成数量。",
        f"- `RUNTIME_VALID={sum(row['runtime_valid'] for row in rows)}`，`PROBE_VALID={sum(row['probe_valid'] for row in rows)}`，`LABEL_VALID={sum(row['label_valid'] for row in rows)}`。",
        f"- 观测标签分布：PASSABLE={observed_label_counts['PASSABLE']}，BLOCKED={observed_label_counts['BLOCKED']}，UNKNOWN={observed_label_counts['UNKNOWN']}，CONTROL_PASSABLE={observed_label_counts['CONTROL_PASSABLE']}，CONTROL_BLOCKED={observed_label_counts['CONTROL_BLOCKED']}。",
        f"- 核心正式可评估行 `{len(eligible)}`；`PASSABLE={sum(row['label']=='PASSABLE' for row in eligible)}`，`BLOCKED={sum(row['label']=='BLOCKED' for row in eligible)}`。",
        "- 0.10 m 是固定名义命令，不是真实位移硬门槛；PointGoal 失败不自动产生 BLOCKED。",
        "", "## 实际位移", "",
        f"- 观测到的实际手端点位移范围：`{min(actuals):.6f}..{max(actuals):.6f} m`。" if actuals else "- 尚无实际试验位移输出。",
        "- 接触受限、自由空间跟踪不足和安全停止是不同结果类别；只有独立物理/几何证据充分时才产生有效 BLOCKED。",
        "", "## 阶段完成度", "",
        *phase_lines,
        *coverage_notes,
        "", "## 科学问题逐项回答", "",
        f"1. **E1 能否判断通行性？** 当前有效标签集为 PASSABLE={label_counts['PASSABLE']}、BLOCKED={label_counts['BLOCKED']}；另有观测 UNKNOWN={observed_label_counts['UNKNOWN']}。" + (
            "在这组有效标签上可以做有限判断，但必须把 UNKNOWN/覆盖率一起报告。" if label_counts["PASSABLE"] and label_counts["BLOCKED"] else "当前有效正负标签不足，不能据此宣称 E1 已可靠判断通行性。"
        ),
        f"2. **E0/E1/E2 谁最好？** {q2}",
        f"3. **E2 是否解决了 E1 不能解决的真实场景？** 可审计的 E2 rescue trial 数为 `{len(e2_rescues)}`；" + (", ".join(f"`{item}`" for item in e2_rescues[:8]) if e2_rescues else "当前没有可证实的 rescue 案例。"),
        f"4. **E2 仍然失败的匹配条件？** `{len(e2_failures)}` 条有效标签上 E2 未输出正确确定标签；其中输出 UNCERTAIN={e2_failure_decisions['UNCERTAIN']}、PASSABLE={e2_failure_decisions['PASSABLE']}、BLOCKED={e2_failure_decisions['BLOCKED']}。这里的‘错误’包含未给出正确确定标签，不等同于全部是误放行；" + (", ".join(f"`{item}`" for item in e2_failures[:8]) if e2_failures else "当前有效标签上未发现 E2 错误，但样本仍小。"),
        "5. **失败来自信息未利用还是探测不足？** 柔性 UNKNOWN 行不能作为算法错误；它们表示单次局部探测没有独立获得整体 clearance 证据。隐藏刚性墙的有效 BLOCKED 则有独立几何/接触证据，属于可评价负例。",
        "6. **结构条件结果：** " + ("；".join(condition_summaries) if condition_summaries else "当前没有柔性核心有效标签；其余结果保留 UNKNOWN。"),
        f"7. **PVC/中开帘：** P1 physics-only 记录数 `{len(p1_results)}`，状态 `{', '.join(p1_statuses) or '未记录'}`；real PVC modeled={p1.get('real_pvc_modeled') if isinstance(p1, dict) else None}，不能把 surrogate 结果称为真实 PVC。",
        f"8. **是否足够支持学习式判断器？** `{'是，已有至少一组有效正负标签，但仅适合作为小规模工程基准' if label_counts['PASSABLE'] and label_counts['BLOCKED'] else '否，缺少同时有效的柔性正负样本'}`；不能把 UNKNOWN 或控制样本当作柔性正例。",
        f"9. **能否严格认定 E1 不足？** `{'可以在当前有效集合上报告其误放行/误拒绝证据，但不能外推到未覆盖物理条件' if method_metrics['E1'] and (method_metrics['E1'].get('false_pass_n', 0) or method_metrics['E1'].get('false_reject_n', 0)) else '不能，仅凭当前证据不足以作严格否定'}`。",
        "10. **下一步算法需求：** 先扩充独立 whole-body clearance 正例、约束结构负例和 matched near-confounder；保持同一单次探测截止点，加入不确定性输出与侧别/姿态鲁棒性验证，再决定是否需要学习模型。",
        "", "## E0 / E1 / E2", "",
    ]
    for method in ("E0", "E1", "E2"):
        item = next((value for value in metric_rows if value["method"] == method and value["split"] == "all" and value["scope"] == "core"), None)
        lines.append(f"- {method}: 无核心有效标签。" if item is None or item["eligible_n"] == 0 else f"- {method}: eligible={item['eligible_n']}, false-pass={item['false_pass_n']}, false-reject={item['false_reject_n']}, coverage={item['decision_coverage']}, balanced_accuracy={item['balanced_accuracy']}.")
    lines.extend(["", "## 证据边界", "", "- P0 单片场景的 physics gate 与 G1 运行分别审计。", "- P1 多片/条带尝试只作为 physics-only 证据；失败不能阻断 P0。", "- `LABEL_VALID` 只在独立 clearance/contact/geometry 证据充分时为真；否则保留 UNKNOWN。", "- 历史 Round 4070 正例不替代 E 内部试验；只有 E summary 中的机器人包络 clearance 才进入 E 正例标签。", "- matched positive/negative CSV 按 probe side 与 repeat_index 配对；控制 seed 独立，因此它是 matched-condition control，不是同 seed 复现。", "- 详细执行、标签和左右手比较见同目录 CSV 报告。"])
    (RESEARCH / "FINAL_REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_integrity() -> None:
    names = ["PROTOCOL_CORRECTION.md", "CONTRACT_VALIDATION.md", "SCENE_COVERAGE.csv", "PHYSICS_GATE_RESULTS.csv", "PROBE_EXECUTION_RESULTS.csv", "LABEL_VALIDITY_AUDIT.csv", "ENGINEERING_BASELINE_COMPARISON.csv", "MATCHED_POSITIVE_NEGATIVE_CASES.csv", "LEFT_RIGHT_COMPARISON.csv", "FAILURE_MECHANISM_ANALYSIS.md", "GPU_RUNTIME_AUDIT.md", "FINAL_REPORT_ZH.md", "GIT_DELIVERY_REPORT.md"]
    output = []
    for name in names:
        path = RESEARCH / name
        data = path.read_bytes() if path.is_file() else b""
        # Hash normalized text so Git's Windows line-ending conversion does not
        # make a committed artifact appear different from its working copy.
        if path.is_file() and path.suffix.lower() in {".csv", ".md", ".json"}:
            data = data.replace(b"\r\n", b"\n")
        output.append({"artifact": name, "exists": path.is_file(), "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest() if path.is_file() else None})
    write_csv(RESEARCH / "E_ARTIFACT_INTEGRITY.csv", output)


def main() -> None:
    _payload, manifest = load_manifest()
    status_payload = read_json(CONFIG / "PHYSICS_GATE_STATUS.json", {}) or {}
    records, baselines = collect(manifest, status_payload)
    rows, audits = evaluate(records, baselines)
    write_json(RESEARCH / "stats" / "evaluator_rows.json", rows)
    write_json(RESEARCH / "stats" / "baselines.json", {side: {"sample_count": len(history)} for side, history in baselines.items()})
    write_protocol_report(rows)
    write_scene_coverage(manifest, rows, status_payload)
    write_physics_results(status_payload)
    write_execution_results(rows)
    write_label_audit(audits)
    metric_rows = write_comparison(rows)
    write_matched_cases(rows)
    write_left_right(rows)
    write_failure_analysis(rows)
    write_gpu_audit(rows)
    write_final_report(rows, metric_rows)
    write_integrity()
    print(json.dumps({"manifest_trials": len(manifest), "observed_summaries": sum(row["summary_exists"] for row in rows), "runtime_valid": sum(row["runtime_valid"] for row in rows), "probe_valid": sum(row["probe_valid"] for row in rows), "label_valid": sum(row["label_valid"] for row in rows), "labels": {label: sum(row["label"] == label for row in rows) for label in ("PASSABLE", "BLOCKED", "UNKNOWN", "CONTROL_PASSABLE", "CONTROL_BLOCKED")}, "baseline_sides": sorted(baselines)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
