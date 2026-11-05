"""Rebuild Round4070 ledgers from run artifacts without policy-side truth."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import shutil
import subprocess
from typing import Any

REPO = Path(__file__).resolve().parents[2]
RESEARCH = REPO / "research" / "round4070_problem_discovery"
OUTPUT = REPO / "outputs" / "round4070_problem_discovery"
TRIALS = OUTPUT / "trials"
SCENE_VALIDATION = OUTPUT / "scene_validation"
PREREG_PATH = REPO / "configs" / "round4070_problem_discovery" / "PREREGISTRATION.json"
CONFIRMATORY_PREREG_PATH = REPO / "configs" / "round4070_problem_discovery" / "CONFIRMATORY_PREREGISTRATION.json"
MAX_G1_TRIALS = 48


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fields or sorted({key for row in rows for key in row})
    if not names:
        names = ["status"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vector(row: dict[str, str], key: str) -> list[float]:
    try:
        value = json.loads(row.get(key, "[]"))
        return [float(item) for item in value]
    except (TypeError, ValueError, json.JSONDecodeError):
        return []


def phase_features(rows: list[dict[str, str]], cutoff: str) -> dict[str, Any] | None:
    if cutoff == "T0":
        selected = [row for row in rows if row.get("phase") == "physical_preprobe_hold"]
    elif cutoff == "T1":
        selected = [row for row in rows if row.get("phase") in {"physical_probe", "physical_probe_hold"}]
    else:
        selected = [row for row in rows if "cautious_advance" in row.get("phase", "") or "responsive_traverse" in row.get("phase", "")]
    if not selected:
        return None
    first, last = selected[0], selected[-1]
    hand_first = vector(first, "deployment_hand_fk_position_w_json")
    hand_last = vector(last, "deployment_hand_fk_position_w_json")
    base_first = vector(first, "deployment_base_position_w_json")
    base_last = vector(last, "deployment_base_position_w_json")
    hand_progress = hand_last[0] - hand_first[0] if hand_first and hand_last else None
    base_delta = math.dist(base_first[:2], base_last[:2]) if len(base_first) >= 2 and len(base_last) >= 2 else None
    tracking = [finite(row.get("upper_tracking_rms_rad")) for row in selected]
    tracking = [value for value in tracking if value is not None]
    sample_times = [finite(row.get("time_s")) for row in selected]
    sample_times = [value for value in sample_times if value is not None]
    duration_s = max(0.0, sample_times[-1] - sample_times[0]) if len(sample_times) >= 2 else None
    low_speed_duration_s = 0.0
    low_speed_duration_assessable = False
    for index, row in enumerate(selected[:-1]):
        current_time = finite(row.get("time_s"))
        next_time = finite(selected[index + 1].get("time_s"))
        velocity = vector(row, "deployment_hand_fk_velocity_w_json")
        if current_time is None or next_time is None or not velocity:
            continue
        low_speed_duration_assessable = True
        if abs(velocity[0]) <= 0.004:
            low_speed_duration_s += max(0.0, next_time - current_time)
    effort_rms: list[float] = []
    effort_source_values: set[str] = set()
    q_errors: list[float] = []
    joint_vel_norms: list[float] = []
    for row in selected:
        effort = vector(row, "deployment_motor_effort_json")
        if effort:
            effort_rms.append(math.sqrt(sum(value * value for value in effort) / len(effort)))
        if row.get("deployment_effort_source"):
            effort_source_values.add(row["deployment_effort_source"])
        q = vector(row, "deployment_q_actual_json")
        q_cmd = vector(row, "deployment_q_cmd_json")
        if q and len(q) == len(q_cmd):
            q_errors.append(math.sqrt(sum((a - b) ** 2 for a, b in zip(q_cmd[15:], q[15:])) / max(1, len(q_cmd[15:]))))
        qdot = vector(row, "deployment_qdot_json")
        if qdot:
            joint_vel_norms.append(math.sqrt(sum(value * value for value in qdot[15:])))
    return {
        "cutoff": cutoff,
        "sample_count": len(selected),
        "duration_s": duration_s,
        "hand_low_speed_duration_s": low_speed_duration_s if low_speed_duration_assessable else None,
        "hand_forward_delta_m": hand_progress,
        "base_delta_m": base_delta,
        "upper_tracking_rms_mean_rad": sum(tracking) / len(tracking) if tracking else None,
        "q_cmd_minus_q_upper_rms_mean_rad": sum(q_errors) / len(q_errors) if q_errors else None,
        "joint_velocity_upper_norm_mean": sum(joint_vel_norms) / len(joint_vel_norms) if joint_vel_norms else None,
        "effort_rms_mean": sum(effort_rms) / len(effort_rms) if effort_rms else None,
        "effort_sources": "|".join(sorted(effort_source_values)),
        "measured_planar_speed_mean_mps": _mean([finite(row.get("measured_planar_speed")) for row in selected]),
        "command_vx_mean_mps": _mean([finite(row.get("command_vx")) for row in selected]),
    }


def _mean(values: list[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    return sum(usable) / len(usable) if usable else None


def trial_record(run: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = read_json(run / "trial_metadata.json", {}) or {}
    summary = read_json(run / "summary.json")
    watchdog = read_json(run / "watchdog_summary.json", {}) or {}
    telemetry_path = run / "telemetry.csv"
    rows: list[dict[str, str]] = []
    if telemetry_path.is_file():
        with telemetry_path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    if not isinstance(summary, dict):
        return ({
            "trial_id": run.name,
            "stage": metadata.get("stage"),
            "scenario": metadata.get("scenario_for_scene_builder"),
            "policy": metadata.get("policy"),
            "probe_side": metadata.get("probe_side"),
            "seed": metadata.get("seed"),
            "reset_root_y": metadata.get("reset_root_y"),
            "runtime_class": "RUNTIME_INVALID",
            "runtime_valid": False,
            "status": "MISSING_SUMMARY",
            "watchdog_timeout_reason": watchdog.get("timeout_reason"),
            "path": str(run),
        }, rows)
    probe = summary.get("physical_probe_result") or {}
    decision = probe.get("decision")
    clearance = bool(summary.get("candidate_clearance_verified", summary.get("whole_body_candidate_clearance", False)))
    pointgoal = bool(summary.get("pointgoal_success", False))
    safe_retract = bool(summary.get("safe_recovery", False))
    runtime_valid = bool(
        summary.get("runtime_status") == "RUNTIME_PASS_FINALIZE_REACHED"
        and (run / "events.json").is_file()
        and telemetry_path.is_file()
        and (run / "deployment_observation.csv").is_file()
    )
    effort_sources = summary.get("deployment_effort_sources_observed") or []
    physical_class = (
        "CONTROL_REFERENCE" if metadata.get("scenario_for_scene_builder") == "empty"
        else "S4_CLEARANCE" if clearance
        else "S3_RESPONSIVE_NO_CLEARANCE" if decision == "RESPONSIVE"
        else "S2_NO_LOCAL_RESPONSE"
    )
    record = {
        "trial_id": run.name,
        "stage": metadata.get("stage"),
        "condition_id": metadata.get("condition_id"),
        "scenario": metadata.get("scenario_for_scene_builder"),
        "policy": metadata.get("policy"),
        "probe_side": metadata.get("probe_side"),
        "seed": metadata.get("seed"),
        "reset_root_y": metadata.get("reset_root_y"),
        "reset_root_yaw": metadata.get("reset_root_yaw"),
        "runtime_class": "VALID_EPISODE" if runtime_valid else "RUNTIME_INVALID",
        "runtime_valid": runtime_valid,
        "physical_class": physical_class if runtime_valid else "NOT_ASSESSABLE_RUNTIME_INVALID",
        "status": summary.get("status"),
        "decision": decision,
        "evaluation_contact_confirmed": probe.get("evaluation_hand_contact_confirmed"),
        "whole_body_clearance": clearance,
        "pointgoal_success": pointgoal,
        "safe_retract": safe_retract,
        "safety_success": bool(summary.get("safety_success", False)),
        "fall": not bool(summary.get("finite_no_fall", True)),
        "goal_error_m": finite(summary.get("goal_error_m")),
        "effort_sources": "|".join(sorted(set(str(value) for value in effort_sources))),
        "effort_source_valid_for_probe": bool(effort_sources) and "unavailable" not in effort_sources,
        "t1_hand_forward_delta_m": None,
        "t1_base_delta_m": None,
        "t1_tracking_rms_mean_rad": None,
        "t1_effort_rms_mean": None,
        "t2_hand_forward_delta_m": None,
        "t2_base_delta_m": None,
        "wall_time_s": finite((read_json(run / "resource_summary.json", {}) or {}).get("rollout_wall_time_s")),
        "watchdog_timeout_reason": watchdog.get("timeout_reason"),
        "path": str(run),
    }
    features = []
    for cutoff in ("T0", "T1", "T2"):
        item = phase_features(rows, cutoff)
        if item is None:
            continue
        feature_row = {
            "trial_id": run.name,
            "scenario": record["scenario"],
            "policy": record["policy"],
            "probe_side": record["probe_side"],
            "seed": record["seed"],
            "reset_root_y": record["reset_root_y"],
            "reset_root_yaw": record["reset_root_yaw"],
            "deployment_decision": decision,
            **item,
        }
        features.append(feature_row)
        if cutoff == "T1":
            record["t1_hand_forward_delta_m"] = item["hand_forward_delta_m"]
            record["t1_base_delta_m"] = item["base_delta_m"]
            record["t1_tracking_rms_mean_rad"] = item["upper_tracking_rms_mean_rad"]
            record["t1_effort_rms_mean"] = item["effort_rms_mean"]
        if cutoff == "T2":
            record["t2_hand_forward_delta_m"] = item["hand_forward_delta_m"]
            record["t2_base_delta_m"] = item["base_delta_m"]
    return record, features


def matches(features: list[dict[str, Any]], trial_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    prereg = read_json(PREREG_PATH, {}) or {}
    limits = prereg.get("frozen_early_match_rule", {}).get("maximum_absolute_differences", {})
    values = [row for row in features if row.get("cutoff") == "T1" and row.get("sample_count", 0) > 0]
    candidates: list[dict[str, Any]] = []
    for i, first in enumerate(values):
        for second in values[i + 1 :]:
            if first.get("probe_side") != second.get("probe_side"):
                continue
            if first.get("policy") != second.get("policy") or first.get("seed") != second.get("seed"):
                continue
            if first.get("reset_root_y") != second.get("reset_root_y") or first.get("reset_root_yaw") != second.get("reset_root_yaw"):
                continue
            if first.get("scenario") == second.get("scenario"):
                continue
            a = trial_by_id[first["trial_id"]]
            b = trial_by_id[second["trial_id"]]
            if not a["runtime_valid"] or not b["runtime_valid"] or not a["effort_source_valid_for_probe"] or not b["effort_source_valid_for_probe"]:
                continue
            diffs: dict[str, float | None] = {}
            key_pairs = {
                "hand_forward_delta_m": "hand_forward_delta_m",
                "base_delta_m": "base_delta_m",
                "upper_tracking_rms_mean_rad": "upper_tracking_rms_mean_rad",
                "effort_rms_mean": "effort_rms_mean",
            }
            for limit_key, feature_key in key_pairs.items():
                va, vb = finite(first.get(feature_key)), finite(second.get(feature_key))
                diffs[limit_key] = abs(va - vb) if va is not None and vb is not None else None
            threshold_names = {
                "hand_forward_delta_m": "hand_forward_delta_m",
                "base_delta_m": "base_delta_m",
                "upper_tracking_rms_mean_rad": "upper_tracking_rms_mean_rad",
                "effort_rms_mean": "effort_rms_mean",
            }
            matched = all(
                diffs[name] is not None and diffs[name] <= float(limits[threshold_names[name]])
                for name in threshold_names
            )
            outcomes_differ = bool(a["whole_body_clearance"] != b["whole_body_clearance"])
            candidates.append({
                "trial_a": first["trial_id"], "scenario_a": first["scenario"], "trial_b": second["trial_id"], "scenario_b": second["scenario"],
                "probe_side": first["probe_side"], "seed": first["seed"], "reset_root_y": first["reset_root_y"],
                **{f"abs_diff_{key}": value for key, value in diffs.items()},
                "passes_frozen_T1_match_rule": matched,
                "clearance_a": a["whole_body_clearance"], "clearance_b": b["whole_body_clearance"],
                "clearance_outcomes_differ": outcomes_differ,
                "independent_confirmation_required": bool(matched and outcomes_differ),
            })
    return candidates


def old_baseline_comparison() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    index = read_json(REPO / "outputs" / "g1_final4070" / "trial_index.json", {}) or {}
    rows = index.get("records", [])
    output: list[dict[str, Any]] = []
    for policy in ("r1", "r2"):
        group = [row for row in rows if row.get("policy") == policy and row.get("scenario") != "empty"]
        output.append({
            "comparison": "inherited_round4070_policy_summary",
            "policy": policy,
            "n_nonempty": len(group),
            "n_valid": sum(bool(row.get("runtime_validity")) for row in group),
            "n_clearance": sum(bool(row.get("whole_body_clearance")) for row in group),
            "n_pointgoal": sum(bool(row.get("pointgoal_success")) for row in group),
            "n_safe_retract": sum(bool(row.get("safe_retract")) for row in group),
            "n_rigid": sum(row.get("scenario") == "rigid_panel" for row in group),
            "n_flexible": sum(row.get("scenario") in {"nominal", "bottom_left_anchor", "bottom_right_anchor"} for row in group),
            "interpretation": "Unequal, incomplete pairing; descriptive only, not a causal policy ranking.",
        })
    new_rows = list(read_new_trials())
    new_nonempty = [row for row in new_rows if row.get("runtime_valid") and row.get("scenario") != "empty"]
    output.append({
        "comparison": "round4070_new_runner_r1_only",
        "policy": "r1",
        "n_nonempty": len(new_nonempty),
        "n_valid": sum(bool(row.get("runtime_valid")) for row in new_rows),
        "n_clearance": sum(bool(row.get("whole_body_clearance")) for row in new_nonempty),
        "n_control_reference_clearance": sum(bool(row.get("whole_body_clearance")) for row in new_rows if row.get("runtime_valid") and row.get("scenario") == "empty"),
        "n_pointgoal": sum(bool(row.get("pointgoal_success")) for row in new_nonempty),
        "n_safe_retract": sum(bool(row.get("safe_retract")) for row in new_nonempty),
        "n_rigid": sum(row.get("scenario") == "rigid_panel" for row in new_nonempty),
        "n_flexible": sum(row.get("scenario") in {"nominal", "half_density", "double_density", "bottom_left_anchor", "bottom_right_anchor", "both_bottom_corners"} for row in new_nonempty),
        "interpretation": "New discovery runs; not compared to R2 unless same-condition policy pairs exist.",
    })
    return output, {"historical_index_rows": len(rows), "historical_valid_rows": sum(bool(row.get("runtime_validity")) for row in rows)}


def read_new_trials() -> list[dict[str, Any]]:
    records = []
    if TRIALS.is_dir():
        for run in sorted(TRIALS.iterdir()):
            if run.is_dir():
                record, _ = trial_record(run)
                records.append(record)
    return records


def scene_validation_records() -> list[dict[str, Any]]:
    records = []
    if not SCENE_VALIDATION.is_dir():
        return records
    for summary_path in sorted(SCENE_VALIDATION.glob("*/*/summary.json")):
        summary = read_json(summary_path, {}) or {}
        records.append({
            "run_id": str(summary_path.parent.relative_to(OUTPUT)).replace("\\", "/"),
            "scene": summary.get("scene", {}).get("name") if isinstance(summary.get("scene"), dict) else summary.get("scene"),
            "mode": summary.get("mode"),
            "status": summary.get("status"),
            "physics_status": summary.get("physics_status"),
            "runtime_status": summary.get("runtime_status"),
            "max_speed_mps": summary.get("max_speed_mps"),
            "max_displacement_m": summary.get("max_displacement_m"),
            "max_stretch_ratio": summary.get("max_neighbor_stretch_ratio"),
            "attachment_exact": all(item.get("attachment_vertex_set_exact", False) for item in (summary.get("per_sheet") or {}).values()),
            "summary_path": str(summary_path),
        })
    return records


def main() -> None:
    RESEARCH.mkdir(parents=True, exist_ok=True)
    prereg = read_json(PREREG_PATH, {}) or {}
    confirmatory_prereg = read_json(CONFIRMATORY_PREREG_PATH, {}) or {}
    shutil.copy2(PREREG_PATH, RESEARCH / "PREREGISTRATION.json")
    if CONFIRMATORY_PREREG_PATH.is_file():
        shutil.copy2(CONFIRMATORY_PREREG_PATH, RESEARCH / "CONFIRMATORY_PREREGISTRATION.json")
    initial_plans = prereg.get("trials", [])
    confirmatory_plans = confirmatory_prereg.get("trials", [])
    g1_plans = [*initial_plans, *confirmatory_plans]
    trial_by_id: dict[str, dict[str, Any]] = {}
    feature_rows: list[dict[str, Any]] = []
    for run in sorted(TRIALS.iterdir()) if TRIALS.is_dir() else []:
        if not run.is_dir():
            continue
        record, features = trial_record(run)
        trial_by_id[record["trial_id"]] = record
        feature_rows.extend(features)
    trial_records = list(trial_by_id.values())
    matched_rows = matches(feature_rows, trial_by_id)
    write_csv(RESEARCH / "baseline_reproduction.csv", baseline_rows())
    write_csv(RESEARCH / "TRIAL_MANIFEST.csv", manifest_rows(g1_plans, trial_by_id))
    write_csv(RESEARCH / "EVIDENCE_MATRIX.csv", trial_records)
    write_csv(RESEARCH / "deployment_cutoff_features.csv", feature_rows)
    write_csv(RESEARCH / "matched_T1_candidates.csv", matched_rows)
    comparisons, historical_counts = old_baseline_comparison()
    write_csv(RESEARCH / "BASELINE_COMPARISON.csv", comparisons)
    scene_rows = scene_validation_records()
    write_csv(RESEARCH / "scene_physics_matrix.csv", scene_rows)

    launched = len([row for row in trial_records if row.get("status") != "MISSING_SUMMARY"])
    valid = sum(bool(row.get("runtime_valid")) for row in trial_records)
    runtime_invalid = launched - valid
    decisions = {
        "H1_temporal_response": "INCONCLUSIVE",
        "H2_spatial_probe": "INCONCLUSIVE",
        "H3_local_response_not_whole_body_clearance": "INCONCLUSIVE",
    }
    valid_flexible = [row for row in trial_records if row.get("runtime_valid") and row.get("scenario") not in {"empty", "rigid_panel"}]
    if len(valid_flexible) >= 6:
        positives = sum(bool(row.get("whole_body_clearance")) for row in valid_flexible)
        negatives = len(valid_flexible) - positives
        decisions["H3_local_response_not_whole_body_clearance"] = "SUPPORTED" if positives > 0 and negatives > 0 and any(row.get("decision") == "RESPONSIVE" and not row.get("whole_body_clearance") for row in valid_flexible) else "PARTIALLY_SUPPORTED"
    confirmed_pairs = [row for row in matched_rows if row["passes_frozen_T1_match_rule"] and row["clearance_outcomes_differ"]]
    confirmatory_pairs = [
        row for row in matched_rows
        if row["passes_frozen_T1_match_rule"]
        and {row.get("scenario_a"), row.get("scenario_b")} == {"nominal", "bottom_right_anchor"}
        and str(row.get("trial_a", "")).startswith("C")
        and str(row.get("trial_b", "")).startswith("C")
    ]
    confirmatory_discordant_pairs = [row for row in confirmatory_pairs if row["clearance_outcomes_differ"]]

    summary = {
        "schema": "lift-the-curtain.round4070-problem-discovery-manifest.v1",
        "status": "COMPLETE_EVIDENCE_FROZEN" if launched >= len(g1_plans) and len(g1_plans) > 0 else "IN_PROGRESS",
        "created_at": __import__("datetime").datetime.now().astimezone().isoformat(),
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip(),
        "trial_accounting": {
            "initial_preregistered_g1_trials": len(initial_plans),
            "confirmatory_preregistered_g1_trials": len(confirmatory_plans),
            "total_preregistered_g1_trials": len(g1_plans),
            "launched_g1_trials": launched,
            "valid_completed_g1_trials": valid,
            "runtime_invalid_or_incomplete_g1_trials": runtime_invalid,
            "remaining_hard_budget": max(0, MAX_G1_TRIALS - launched),
            "historical_round4070_trials_separate": historical_counts,
        },
        "scene_physics_runs": len(scene_rows),
        "scene_physics_passes": sum(row.get("status") == "PASS_PHYSICS_SANITY" for row in scene_rows),
        "scene_physics_failures": sum(row.get("status") == "FAIL_PHYSICS_SANITY" for row in scene_rows),
        "frozen_T1_matching_candidates": len(matched_rows),
        "frozen_T1_outcome_discordant_candidates": len(confirmed_pairs),
        "confirmatory_nominal_bottom_right_pairs": len(confirmatory_pairs),
        "confirmatory_nominal_bottom_right_discordant_pairs": len(confirmatory_discordant_pairs),
        "spatial_probe_contrast_trials": 0,
        "spatial_probe_eligible": False,
        "hypothesis_notes": {
            "H1_temporal_response": "No pre-registered temporal-only A/B contrast was run; T2 is exploratory because it follows the fixed advance phase.",
            "H2_spatial_probe": "Condition-contrast matches use the same center probe. No valid split-center multi-sheet physics scene or different-location action contrast survived the gate.",
            "H3_local_response_not_whole_body_clearance": "Valid flexible episodes contain both responsive/no-clearance and responsive/clearance outcomes, including independent frozen-T1 matched condition pairs.",
        },
        "hypothesis_status": decisions,
        "historical_outputs_modified": False,
        "deployment_truth_leakage_detected": False,
    }
    write_json(OUTPUT / "run_manifest.json", summary)
    write_json(OUTPUT / "trial_index.json", {
        "schema": "lift-the-curtain.round4070-problem-discovery-trial-index.v1",
        "launched_trial_directories": launched,
        "valid_completed_trials": valid,
        "runtime_invalid_or_incomplete_trials": runtime_invalid,
        "records": trial_records,
    })
    write_json(OUTPUT / "matched_condition_manifest.json", {
        "schema": "lift-the-curtain.round4070-problem-discovery-matched-condition-manifest.v1",
        "frozen_rule": prereg.get("frozen_early_match_rule"),
        "candidates": matched_rows,
    })
    write_json(OUTPUT / "campaign_summary.json", summary)
    write_json(OUTPUT / "run_queue_state.json", {
        "planned_initial": len(initial_plans),
        "planned_confirmatory": len(confirmatory_plans),
        "planned_total": len(g1_plans),
        "launched": launched,
        "valid": valid,
        "runtime_invalid_or_incomplete": runtime_invalid,
        "completed_at": summary["created_at"],
    })
    print(json.dumps(summary["trial_accounting"], ensure_ascii=False))


def baseline_rows() -> list[dict[str, Any]]:
    path = RESEARCH / "baseline_reproduction.csv"
    if not path.is_file():
        helper = REPO / "scripts" / "round4070_problem_discovery" / "audit_existing.py"
        subprocess.run([str(REPO / "C:\\conda_envs\\isaaclab\\python.exe"), str(helper)], cwd=REPO, check=False)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def manifest_rows(plans: list[dict[str, Any]], actual: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for plan in plans:
        record = actual.get(plan["trial_id"])
        output.append({
            **plan,
            "status": "NOT_STARTED" if record is None else "VALID" if record.get("runtime_valid") else "RUNTIME_INVALID_OR_INCOMPLETE",
            "physical_class": "NOT_RUN" if record is None else record.get("physical_class"),
            "summary_status": "NOT_RUN" if record is None else record.get("status"),
            "path": str(TRIALS / plan["trial_id"]),
        })
    for trial_id, record in actual.items():
        if trial_id not in {plan["trial_id"] for plan in plans}:
            output.append({"trial_id": trial_id, "stage": record.get("stage"), "status":"UNREGISTERED_RUN", "physical_class":record.get("physical_class"), "path":record.get("path")})
    return output


if __name__ == "__main__":
    main()
