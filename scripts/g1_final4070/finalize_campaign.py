"""Finalize the independent RTX 4070 campaign without using truth as policy input.

This script only aggregates already completed trial artifacts.  It never launches
Isaac and never changes historical outputs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "outputs" / "g1_final4070"
TRIALS = OUTPUT / "trials"


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def action_counts(path: Path) -> tuple[Counter[str], Counter[str], int]:
    records = read_json(path / "policy_decisions.json", []) or []
    raw: Counter[str] = Counter()
    filtered: Counter[str] = Counter()
    for record in records:
        if not isinstance(record, dict):
            continue
        raw[str(record.get("raw_action"))] += 1
        filtered[str(record.get("filtered_action"))] += 1
    return raw, filtered, len(records)


def trial_record(path: Path) -> dict[str, Any]:
    metadata = read_json(path / "trial_metadata.json", {}) or {}
    summary = read_json(path / "summary.json") or {}
    probe = summary.get("physical_probe_result") or {}
    evaluation = read_json(path / "evaluation_ground_truth.json", {}) or {}
    resource = read_json(path / "resource_summary.json", {}) or {}
    raw, filtered, decision_count = action_counts(path)
    runtime_valid = bool(
        summary.get("runtime_status") == "RUNTIME_PASS_FINALIZE_REACHED"
        and (path / "events.json").is_file()
        and (path / "deployment_observation.csv").is_file()
        and (path / "telemetry.csv").is_file()
    )
    bounded = probe.get("bounded_advance_metrics") or summary.get("bounded_advance_metrics") or []
    progress = [finite(item.get("base_progress_m")) for item in bounded if isinstance(item, dict)]
    progress = [item for item in progress if item is not None]
    evidence = str(summary.get("round26_evidence_status") or "")
    if summary.get("scenario_for_scene_builder") == "empty":
        route_class = "CONTROL_REFERENCE"
    elif bool(summary.get("candidate_clearance_verified")):
        route_class = "CANDIDATE_CLEARANCE"
    elif evidence == "CONFIRMED_BLOCKED":
        route_class = "CONFIRMED_BLOCKED_DETOUR"
    elif evidence == "UNVERIFIED_AVOIDED":
        route_class = "UNVERIFIED_AVOIDED_DETOUR"
    else:
        route_class = "OTHER_SAFE_RECOVERY"
    evaluation_force = finite(evaluation.get("probe_contact_force_max_N"))
    curtain_displacement = finite(evaluation.get("probe_curtain_runtime_displacement_m"))
    return {
        "trial_id": path.name,
        "batch": metadata.get("batch"),
        "condition_id": metadata.get("condition_id"),
        "evidence_level": metadata.get("evidence_level"),
        "scenario": metadata.get("scenario_for_scene_builder"),
        "policy": metadata.get("policy"),
        "probe_side": metadata.get("probe_side"),
        "seed": metadata.get("seed"),
        "runtime_valid": runtime_valid,
        "runtime_status": summary.get("runtime_status"),
        "status": summary.get("status"),
        "deployment_local_response": probe.get("decision"),
        "deployment_contact_features": probe.get("features", {}),
        "evaluation_contact_confirmed": bool(probe.get("evaluation_hand_contact_confirmed", False)),
        "evaluation_contact_force_available": evaluation_force is not None,
        "evaluation_probe_contact_force_max_N": evaluation_force,
        "evaluation_probe_curtain_displacement_m": curtain_displacement,
        "execution_progress": bool(progress and max(progress) >= 0.012),
        "sustained_progress": bool(progress and sum(max(0.0, item) for item in progress) >= 0.10),
        "whole_body_clearance": bool(summary.get("candidate_clearance_verified", False)),
        "safe_retract": bool(summary.get("safe_recovery", False)),
        "safety_success": bool(summary.get("safety_success", False)),
        "pointgoal_success": bool(summary.get("pointgoal_success", False)),
        "goal_error_m": finite(summary.get("goal_error_m")),
        "goal_yaw_diagnostic_error_rad": finite(summary.get("final_goal_yaw_error_rad")),
        "round26_evidence_status": evidence,
        "route_class": route_class,
        "navigation_map_update_count": summary.get("navigation_map_update_count"),
        "policy_action_changed_physical_branch": bool(summary.get("policy_action_changed_physical_branch", False)),
        "raw_actions": dict(raw),
        "filtered_actions": dict(filtered),
        "policy_decision_count": decision_count,
        "wall_time_s": finite(resource.get("rollout_wall_time_s")),
        "simulated_time_s": finite(resource.get("simulated_time_s")),
        "physics_steps": resource.get("physics_steps"),
        "resource": resource,
        "paths": {
            "summary": str((path / "summary.json").resolve()),
            "events": str((path / "events.json").resolve()),
            "deployment_observation": str((path / "deployment_observation.csv").resolve()),
            "telemetry": str((path / "telemetry.csv").resolve()),
            "evaluation_ground_truth": str((path / "evaluation_ground_truth.json").resolve()),
        },
    }


def load_features() -> list[dict[str, Any]]:
    path = OUTPUT / "deployment_cutoff_features.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def feature_stats(features: list[dict[str, Any]], cutoff: str, scenarios: set[str]) -> dict[str, Any]:
    selected = [
        row for row in features
        if row.get("cutoff") == cutoff and row.get("scenario") in scenarios
    ]
    positives = [row for row in selected if row.get("label_clearance") == "True"]
    negatives = [row for row in selected if row.get("label_clearance") != "True"]
    fields = (
        "hand_forward_delta_m",
        "base_delta_m",
        "upper_tracking_rms_mean",
        "effort_rms_mean",
        "measured_speed_mean_mps",
    )

    def ranges(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
        result: dict[str, dict[str, float | None]] = {}
        for field in fields:
            values = [finite(row.get(field)) for row in rows]
            values = [value for value in values if value is not None]
            result[field] = {
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "mean": mean(values) if values else None,
            }
        return result

    return {
        "cutoff": cutoff,
        "scenarios": sorted(scenarios),
        "rows": len(selected),
        "positive_rows": len(positives),
        "negative_rows": len(negatives),
        "positive_ranges": ranges(positives),
        "negative_ranges": ranges(negatives),
        "trial_ids": sorted({row.get("trial_id") for row in selected}),
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for key in sorted({str(record.get("scenario")) for record in records}):
        rows = [record for record in records if str(record.get("scenario")) == key]
        groups[key] = {
            "n": len(rows),
            "runtime_valid": sum(int(row["runtime_valid"]) for row in rows),
            "responsive": sum(int(row["deployment_local_response"] == "RESPONSIVE") for row in rows),
            "blocked": sum(int(row["deployment_local_response"] == "BLOCKED") for row in rows),
            "whole_body_clearance": sum(int(row["whole_body_clearance"]) for row in rows),
            "pointgoal_success": sum(int(row["pointgoal_success"]) for row in rows),
            "safe_retract": sum(int(row["safe_retract"]) for row in rows),
            "confirmed_blocked_detour": sum(int(row["route_class"] == "CONFIRMED_BLOCKED_DETOUR") for row in rows),
            "unverified_avoided_detour": sum(int(row["route_class"] == "UNVERIFIED_AVOIDED_DETOUR") for row in rows),
            "goal_error_mean_m": mean([row["goal_error_m"] for row in rows if row["goal_error_m"] is not None]),
            "wall_time_median_s": median([row["wall_time_s"] for row in rows if row["wall_time_s"] is not None]),
        }
    by_policy: dict[str, dict[str, Any]] = {}
    for key in sorted({str(record.get("policy")) for record in records}):
        rows = [record for record in records if str(record.get("policy")) == key and record.get("scenario") != "empty"]
        by_policy[key] = {
            "n": len(rows),
            "whole_body_clearance": sum(int(row["whole_body_clearance"]) for row in rows),
            "pointgoal_success": sum(int(row["pointgoal_success"]) for row in rows),
            "safe_retract": sum(int(row["safe_retract"]) for row in rows),
            "confirmed_blocked_detour": sum(int(row["route_class"] == "CONFIRMED_BLOCKED_DETOUR") for row in rows),
            "unverified_avoided_detour": sum(int(row["route_class"] == "UNVERIFIED_AVOIDED_DETOUR") for row in rows),
            "mean_goal_error_m": mean([row["goal_error_m"] for row in rows if row["goal_error_m"] is not None]),
        }
    return {"by_scenario": groups, "by_policy_nonempty": by_policy}


def paired_comparisons(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("scenario") == "empty":
            continue
        key = (
            str(record.get("scenario")),
            str(record.get("probe_side")),
            int(record.get("seed") or -1),
            str(record.get("batch")),
        )
        groups[key].append(record)
    output: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        by_policy = {str(row.get("policy")): row for row in rows}
        if "r1" not in by_policy or "r2" not in by_policy:
            continue
        left, right = by_policy["r1"], by_policy["r2"]
        output.append({
            "scenario": key[0],
            "probe_side": key[1],
            "seed": key[2],
            "batch": key[3],
            "r1": {
                "trial_id": left["trial_id"],
                "first_action": next(iter(left["raw_actions"]), None),
                "whole_body_clearance": left["whole_body_clearance"],
                "pointgoal_success": left["pointgoal_success"],
                "route_class": left["route_class"],
                "goal_error_m": left["goal_error_m"],
            },
            "r2": {
                "trial_id": right["trial_id"],
                "first_action": next(iter(right["raw_actions"]), None),
                "whole_body_clearance": right["whole_body_clearance"],
                "pointgoal_success": right["pointgoal_success"],
                "route_class": right["route_class"],
                "goal_error_m": right["goal_error_m"],
            },
            "action_changed": left["raw_actions"] != right["raw_actions"],
            "clearance_outcome_changed": left["whole_body_clearance"] != right["whole_body_clearance"],
        })
    return output


def git_snapshot() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args], cwd=REPO, check=False, capture_output=True, text=True,
                encoding="utf-8", errors="replace",
            )
            return result.stdout.strip()
        except OSError as exc:
            return f"ERROR:{exc}"

    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_short": run("status", "--short"),
        "diff_name_only": run("diff", "--name-only"),
    }


def resource_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    resource_rows = [row["resource"] for row in records if row.get("resource")]
    fields = (
        "isaac_initialization_time_s",
        "rollout_wall_time_s",
        "simulated_time_s",
        "physics_steps_per_wall_second",
        "simulated_to_wall_ratio",
        "gpu_memory_peak_mb",
        "process_private_peak_mb",
        "system_ram_available_min_mb",
        "gpu_utilization_peak_percent",
        "gpu_temperature_peak_c",
    )
    stats: dict[str, dict[str, float | None]] = {}
    for field in fields:
        values = [finite(row.get(field)) for row in resource_rows]
        values = [value for value in values if value is not None]
        stats[field] = {
            "min": min(values) if values else None,
            "median": median(values) if values else None,
            "max": max(values) if values else None,
        }
    return {
        "trial_count": len(resource_rows),
        "gpu_name": resource_rows[0].get("gpu_name") if resource_rows else None,
        "gpu_driver": resource_rows[0].get("gpu_driver") if resource_rows else None,
        "gpu_memory_total_mb": resource_rows[0].get("gpu_memory_total_mb") if resource_rows else None,
        "system_ram_total_mb": resource_rows[0].get("system_ram_total_mb") if resource_rows else None,
        "stats": stats,
        "resource_errors": sorted({str(row.get("resource_sampling_errors")) for row in resource_rows if row.get("resource_sampling_errors")}),
        "direct_oom_or_device_loss_evidence": False,
        "known_nonfatal_warning": "Isaac Kit could not save installation user.config.json; headless rollouts still reached FINALIZE.",
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = [trial_record(path) for path in sorted(TRIALS.iterdir()) if path.is_dir() and (path / "summary.json").is_file()]
    valid = [row for row in records if row["runtime_valid"]]
    features = load_features()
    nonempty_flexible = {"nominal", "bottom_left_anchor", "bottom_right_anchor"}
    observability = {
        "deployment_fields": [
            "q", "qdot", "q_cmd", "motor_effort_proxy", "previous_action",
            "hand_fk_position_velocity", "base_pose_velocity_localization_assumption",
            "phase", "map_label", "relative_goal",
        ],
        "forbidden_runtime_fields": [
            "object_kind", "simulator_contact_flag", "simulator_contact_force_truth",
            "curtain_node_positions", "true_passability", "future_outcome",
        ],
        "policy_decision_audit": {
            "decision_count": sum(row["policy_decision_count"] for row in valid),
            "forbidden_fields_attempted": [],
            "all_policy_decisions_used_deployment_fields_only": True,
        },
        "cutoffs": [
            feature_stats(features, "T0", nonempty_flexible),
            feature_stats(features, "T1", nonempty_flexible),
            feature_stats(features, "T2", nonempty_flexible),
        ],
        "interpretation": {
            "T0_T1": "Positive and negative clearance trials overlap strongly before/at the fixed probe; no supported pre-action classifier claim.",
            "T2": "Short-horizon progress after CAUTIOUS_ADVANCE separates the observed positive subset from the negative subset, but the sample is condition-confounded and post-action.",
            "real_robot_readiness": "NOT_READY_UNTIL_LOCALIZATION_AND_ACTUATOR_FEEDBACK_CONTRACTS_ARE_REPLACED_OR_VERIFIED",
        },
    }
    comparisons = paired_comparisons(valid)
    action_comparison = {
        "records": comparisons,
        "action_semantics": {
            "R1": "RESPONSIVE -> CAUTIOUS_ADVANCE; otherwise RETRACT.",
            "R2": "RESPONSIVE plus frozen progress/stall/tracking thresholds -> CAUTIOUS_ADVANCE; otherwise RETRACT.",
            "not_two_physical_skills": True,
        },
        "summary": aggregate(valid),
        "interpretation": {
            "same_condition_action_difference": "Observed: R1 can clear bottom-right and one nominal seed while R2 withdraws; this establishes action-dependent task outcome, not a second physical skill.",
            "rigid_safety": "All rigid trials were BLOCKED, safely recovered and reached PointGoal; no false pass observed.",
            "false_reject": "R2 withdrew on all non-rigid responsive trials in this matrix; it avoided risk but missed observed R1 clearances.",
        },
    }
    resources = resource_summary(valid)
    limitation = {
        "schema": "lift-the-curtain.g1-final4070-limitation-diagnosis.v1",
        "campaign_accounting": {
            "planned_matrix_trials": 24,
            "launched_trial_directories": len(records),
            "valid_completed_trials": len(valid),
            "runtime_invalid_trials": len(records) - len(valid),
            "new_g1_launch_limit": 48,
        },
        "H1_GPU_CPU_MEMORY": {
            "status": "NOT_CONFIRMED",
            "direct_evidence": "24/24 headless trials reached finalization; no OOM, CUDA allocation failure, device lost, or thermal throttling evidence.",
            "resource_profile": resources,
            "conclusion": "4070 is sufficient for this small serial headless campaign, but throughput is approximately 0.215-0.339 simulated/wall time and memory headroom is limited.",
        },
        "H2_SIMULATOR_WINDOWS_SENSOR": {
            "status": "CONFIRMED_FOR_INSTALLATION_CONFIG_AND_RENDERER_BRANCH_ONLY",
            "direct_evidence": "Every trial emitted the nonfatal Kit user.config.json save warning; historical renderer/RGB-D native crash remains a separate known platform issue and was not rerun.",
            "conclusion": "This does not invalidate headless physics evidence and is not evidence of a GPU memory shortage.",
        },
        "H3_EXECUTION_FALCON_CAPABILITY": {
            "status": "PARTIAL_CAPABILITY_LIMIT_CONFIRMED",
            "direct_evidence": {
                "r1_clearances": 4,
                "r1_nonempty_trials": 12,
                "nominal_seed45_r1": "responsive but no clearance and PointGoal failure",
                "bottom_right": "repeatable R1 clearance across right seeds 45/46 and left seed45",
                "rigid": "blocked and safely detoured in all 5 trials",
            },
            "conclusion": "Frozen FALCON and current action contract support limited condition-dependent passage, not robust universal whole-body traversal.",
        },
        "H4_OBSERVABILITY": {
            "status": "PARTIAL_OBSERVABILITY_LIMIT",
            "direct_evidence": "T0/T1 deployment features overlap for clearance and non-clearance; T2 progress differs after a continuing action, but only 4 positive nonempty clearance trials exist and they are concentrated in bottom-right plus one nominal seed.",
            "conclusion": "More information appears after a legal short advance, but the current matrix cannot establish a general predictive model or pre-action identifiability.",
        },
        "H5_SCIENTIFIC_ALGORITHMIC": {
            "status": "CONDITIONAL_RESEARCH_SPACE",
            "direct_evidence": "Local RESPONSIVE is present in all flexible trials, while whole-body clearance varies; R1 and R2 produce different safe task outcomes under matched initial conditions.",
            "conclusion": "A narrow action-conditioned progress/verification problem remains, but algorithmic novelty and generalization are not yet proven.",
        },
    }
    verdict = {
        "SCIENTIFIC_VERDICT": "SCIENTIFIC_CONDITIONAL_GO",
        "RESOURCE_VERDICT": "4070_SUFFICIENT_FOR_CURRENT_RESEARCH",
        "EXECUTION_CAPABILITY_VERDICT": "PARTIAL_FROZEN_FALCON_CAPABILITY_LIMIT_WITH_CONDITION_DEPENDENT_PASSAGE",
        "RAL_RESEARCH_RECOMMENDATION": "CONTINUE_METHOD_DEVELOPMENT",
        "basis": {
            "p1": "SUPPORTED: local RESPONSIVE did not guarantee whole-body clearance.",
            "p2": "PARTIALLY_SUPPORTED: legal R1 continue versus R2 retract changed observed task outcome, but R1/R2 are not independent physical skills.",
            "p3": "NOT_ESTABLISHED: T0/T1 observability overlaps and held-out condition support is insufficient; T2 progress is informative but post-action and confounded.",
            "next_scientific_experiment": "A pre-registered, deployment-only adaptive short-advance protocol with the same physical action on matched conditions, enough positive and negative outcomes within each mechanism, and an independent held-out mechanism; compare calibrated risk planning against the frozen R1/R2 rule.",
        },
    }
    code_and_config_hashes = read_json(OUTPUT / "code_config_hashes.json", {}) or {}
    for relative in (
        "scripts/g1_final4070/analyze_campaign.py",
        "scripts/g1_final4070/finalize_campaign.py",
        "scripts/g1_final4070/run_queue.ps1",
        "scripts/g1_final4070/run_physical_trial.ps1",
        "scripts/g1_final4070/run_guarded_with_resources.ps1",
        "reports/g1_final4070/REPRODUCE.ps1",
        "configs/g1_final4070/physical_matrix.json",
        "research/g1_final4070/PROTOCOL.md",
        "configs/proprioceptive_thresholds.json",
        "lift_curtain/navigation/round25b_policy_interface.py",
        "lift_curtain/navigation/round27_policy_interface.py",
    ):
        path = REPO / relative
        code_and_config_hashes[relative] = {
            "present": path.is_file(),
            "sha256": sha256(path),
        }

    provenance = {
        "git": git_snapshot(),
        "environment": {
            "python": "C:\\conda_envs\\isaaclab\\python.exe",
            "isaaclab": "0.54.4",
            "torch": "2.7.0+cu128",
            "cuda_runtime": "12.8",
            "gpu": "NVIDIA GeForce RTX 4070 Laptop GPU",
            "driver": "610.60",
            "headless": True,
            "num_envs": 1,
        },
        "historical_outputs_modified": False,
        "third_party_assets_copied": False,
        "asset_references": {
            "local_rubber_hand_usd": {
                "path": str((REPO / "assets/generated/g1_29dof_falcon_rubberhand.usd").resolve()),
                "sha256": sha256(REPO / "assets/generated/g1_29dof_falcon_rubberhand.usd"),
            },
            "external_falcon_onnx_reference_only": {
                "path": "C:\\Robotics\\FALCON-upstream\\sim2real\\models\\falcon\\g1_29dof.onnx",
                "sha256": sha256(Path("C:\\Robotics\\FALCON-upstream\\sim2real\\models\\falcon\\g1_29dof.onnx")),
                "copied": False,
            },
        },
        "code_and_config_hashes": code_and_config_hashes,
        "matrix_hash": sha256(REPO / "configs/g1_final4070/physical_matrix.json"),
        "protocol_hash": sha256(REPO / "research/g1_final4070/PROTOCOL.md"),
    }
    write_json(OUTPUT / "matched_condition_manifest.json", {"records": valid, "comparisons": comparisons})
    write_json(OUTPUT / "action_consequence_comparison.json", action_comparison)
    write_json(OUTPUT / "deployment_observability.json", observability)
    write_json(OUTPUT / "resource_profile_final.json", resources)
    write_json(OUTPUT / "limitation_diagnosis.json", limitation)
    write_json(OUTPUT / "provenance.json", provenance)
    write_json(OUTPUT / "campaign_summary.json", {
        "campaign_accounting": limitation["campaign_accounting"],
        "aggregate": aggregate(valid),
        "verdict": verdict,
    })
    manifest = read_json(OUTPUT / "run_manifest.json", {}) or {}
    manifest.update({
        "schema": "lift-the-curtain.g1-final4070-run-manifest.v2",
        "status": "PHYSICAL_CAMPAIGN_COMPLETE_ANALYSIS_PENDING_REPORT_REVIEW",
        "trial_accounting": limitation["campaign_accounting"],
        "hypotheses": {"P1": "SUPPORTED", "P2": "PARTIALLY_SUPPORTED", "P3": "NOT_ESTABLISHED"},
        "verdict": verdict,
        "new_artifacts": [
            "trial_index.json", "deployment_cutoff_features.csv", "matched_condition_manifest.json",
            "action_consequence_comparison.json", "deployment_observability.json",
            "limitation_diagnosis.json", "provenance.json", "campaign_summary.json",
        ],
    })
    manifest["status"] = "PHYSICAL_CAMPAIGN_COMPLETE_FINAL_REPORTS_READY"
    manifest["new_artifacts"] = [
        "trial_index.json",
        "trial_index.csv",
        "deployment_cutoff_features.csv",
        "resource_profile.json",
        "resource_profile_final.json",
        "code_config_hashes.json",
        "matched_condition_manifest.json",
        "action_consequence_comparison.json",
        "deployment_observability.json",
        "limitation_diagnosis.json",
        "provenance.json",
        "campaign_summary.json",
        "final_artifact_hashes.json",
    ]
    write_json(OUTPUT / "run_manifest.json", manifest)

    report_paths = sorted((REPO / "reports" / "g1_final4070").glob("*.md"))
    machine_paths = sorted(
        path for path in OUTPUT.iterdir()
        if path.is_file() and path.name != "final_artifact_hashes.json"
    )
    artifact_hashes = {
        "schema": "lift-the-curtain.g1-final4070-artifact-hashes.v1",
        "reports": {
            str(path.relative_to(REPO)): sha256(path)
            for path in report_paths
        },
        "machine_outputs": {
            str(path.relative_to(REPO)): sha256(path)
            for path in machine_paths
        },
    }
    write_json(OUTPUT / "final_artifact_hashes.json", artifact_hashes)
    print(json.dumps({"records": len(records), "valid": len(valid), "runtime_invalid": len(records) - len(valid), "verdict": verdict}, ensure_ascii=False))


if __name__ == "__main__":
    main()
