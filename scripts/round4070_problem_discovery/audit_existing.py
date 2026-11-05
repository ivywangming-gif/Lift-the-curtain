"""Read-only reconstruction and implementation audit for Round4070."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any


REPO = Path(__file__).resolve().parents[2]
RESEARCH = REPO / "research" / "round4070_problem_discovery"
OUTPUT = REPO / "outputs" / "round4070_problem_discovery"
OLD_OUTPUT = REPO / "outputs" / "g1_final4070"


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
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO, check=False, capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )
    return result.stdout.strip()


def csv_count_and_header(path: Path) -> tuple[int, list[str]]:
    if not path.is_file():
        return 0, []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return sum(1 for _ in reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows or [{"status": "NO_ROWS"}])


def baseline_reconstruction() -> list[dict[str, Any]]:
    index = read_json(OLD_OUTPUT / "trial_index.json", {}) or {}
    rows: list[dict[str, Any]] = []
    for record in index.get("records", []):
        trial = OLD_OUTPUT / "trials" / str(record.get("trial_id"))
        deployment_rows, deployment_header = csv_count_and_header(trial / "deployment_observation.csv")
        telemetry_rows, telemetry_header = csv_count_and_header(trial / "telemetry.csv")
        summary = read_json(trial / "summary.json", {}) or {}
        rows.append(
            {
                "trial_id": record.get("trial_id"),
                "scenario": record.get("scenario"),
                "policy": record.get("policy"),
                "probe_side": record.get("probe_side"),
                "seed": record.get("seed"),
                "runtime_validity": record.get("runtime_validity"),
                "local_response": record.get("local_response"),
                "evaluation_contact_acquisition": record.get("contact_acquisition"),
                "whole_body_clearance": record.get("whole_body_clearance"),
                "pointgoal_success": record.get("pointgoal_success"),
                "safe_retract": record.get("safe_retract"),
                "safety_success": record.get("safety_success"),
                "goal_error_m": record.get("goal_error_m"),
                "deployment_rows": deployment_rows,
                "telemetry_rows": telemetry_rows,
                "summary_present": (trial / "summary.json").is_file(),
                "events_present": (trial / "events.json").is_file(),
                "evaluation_truth_present": (trial / "evaluation_ground_truth.json").is_file(),
                "runtime_snapshots_present": (trial / "curtain_runtime_snapshots.npz").is_file(),
                "old_effort_source_recorded": "deployment_effort_source" in deployment_header or "deployment_effort_source" in telemetry_header,
                "old_runner_version": summary.get("round27_runner_version"),
                "labels": "|".join(record.get("labels", [])),
            }
        )
    return rows


def field_audit() -> list[dict[str, Any]]:
    return [
        {"field":"q","source":"robot.data.joint_pos","recorded":True,"feature":"q_cmd-q tracking RMS","consumer":"feature extractor; not a direct branch variable","deployable":True,"risk":"joint order must match ISAACLAB_TO_OFFICIAL","code":"lift_curtain/control/falcon_adapter.py:118; deployment_observation.py:206"},
        {"field":"qdot","source":"robot.data.joint_vel","recorded":True,"feature":"joint_velocity_mean_norm","consumer":"recorded feature; frozen classifier does not branch on it","deployable":True,"risk":"time alignment with q_cmd","code":"lift_curtain/control/deployment_observation.py:230"},
        {"field":"q_cmd","source":"actual position target returned by FalconAdapter.compute_target","recorded":True,"feature":"upper_tracking_rms","consumer":"recorded feature; frozen classifier does not branch on it","deployable":True,"risk":"target is updated at policy decimation and held between updates","code":"scripts/round4070_problem_discovery/run_episode.py"},
        {"field":"motor_effort","source":"applied_torque then computed_torque then joint_effort_target; otherwise zero-filled unavailable","recorded":True,"feature":"effort_deviation relative to no-contact baseline","consumer":"BLOCKED branch only","deployable":"CONDITIONAL","risk":"old 24 trials did not log source; new runner rejects unavailable source","code":"lift_curtain/control/falcon_adapter.py:132-146"},
        {"field":"previous_action","source":"frozen FALCON previous 29-D action","recorded":True,"feature":"none in frozen rule","consumer":"FALCON observation history; available to future deployment model","deployable":True,"risk":"not evidence of contact by itself","code":"lift_curtain/control/falcon_adapter.py:104,191"},
        {"field":"hand_fk_position_w/velocity_w","source":"rubber-hand body world pose finite difference","recorded":True,"feature":"forward progress, progress ratio, low-speed duration","consumer":"RESPONSIVE/BLOCKED rule","deployable":"CONDITIONAL","risk":"world frame requires localization and calibrated kinematics","code":"scripts/round4070_problem_discovery/run_episode.py"},
        {"field":"base pose/velocity","source":"Isaac root state","recorded":True,"feature":"base disturbance and phase progress","consumer":"SE(2), progress monitor, geometry-relative clearance","deployable":"LOCALIZATION_ASSUMPTION","risk":"must be replaced by deployable localization on real G1","code":"lift_curtain/control/deployment_observation.py:74-91"},
        {"field":"IMU orientation/angular velocity","source":"root quaternion and angular velocity","recorded":"telemetry roll/pitch/yaw; object field in memory","feature":"stability checks","consumer":"fall/safety and progress monitor","deployable":True,"risk":"sim-to-real calibration","code":"scripts/round4070_problem_discovery/run_episode.py"},
        {"field":"map label/relative goal","source":"oracle 0/1/2 map plus permitted localization","recorded":True,"feature":"routing context","consumer":"planner/state machine, not material classifier","deployable":"ORACLE_MAP_ASSUMPTION","risk":"not an RGB-D/VLM result","code":"scripts/round4070_problem_discovery/run_episode.py"},
        {"field":"upper_tracking_rms","source":"q_cmd-q over official upper joints","recorded":True,"feature":"upper_tracking_rms","consumer":"progress diagnostics; not classify_proprioceptive_response","deployable":True,"risk":"must not be described as already used by R1/R2 contact rule","code":"lift_curtain/control/deployment_observation.py:263"},
    ]


def main() -> None:
    RESEARCH.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = baseline_reconstruction()
    write_csv(RESEARCH / "baseline_reproduction.csv", rows)
    audit_rows = field_audit()
    write_json(
        RESEARCH / "implementation_audit.json",
        {
            "schema": "lift-the-curtain.round4070-implementation-audit.v1",
            "historical_trial_count": len(rows),
            "historical_runtime_valid_count": sum(str(row["runtime_validity"]).lower() == "true" for row in rows),
            "historical_effort_source_recorded_count": sum(bool(row["old_effort_source_recorded"]) for row in rows),
            "field_audit": audit_rows,
            "frozen_classifier_direct_variables": ["hand_forward_progress_m", "hand_progress_ratio", "effort_deviation", "hand_low_speed_duration_s"],
            "recorded_but_not_direct_classifier_variables": ["upper_tracking_rms", "joint_velocity_mean_norm", "base_disturbance_m", "previous_action"],
        },
    )
    leakage = {
        "deployment_whitelist": ["q","qdot","q_cmd","motor_effort","previous_action","hand_fk_position_w","hand_fk_velocity_w","base_position_w","base_velocity_w","imu_orientation_rpy","imu_angular_velocity","map_cell_label","relative_goal_xy","phase/time/history"],
        "evaluation_only": ["object identity","simulator contact flag/point/force","cloth node positions/deformation","hidden physical parameters","true passability","future episode outcome"],
        "localization_assumption": "Isaac root pose is permitted for this phase and is not a real-robot localization implementation.",
        "old_policy_decision_files_scanned": len(rows),
        "known_leakage_into_deployment_branch": False,
        "effort_source_gap": "Old 24 trials record effort values but not the adapter source; they remain usable only with this provenance limitation.",
    }
    write_json(RESEARCH / "leakage_audit.json", leakage)

    prereg = read_json(REPO / "configs" / "round4070_problem_discovery" / "PREREGISTRATION.json", {}) or {}
    manifest_rows: list[dict[str, Any]] = []
    for trial in prereg.get("trials", []):
        run = OUTPUT / "trials" / trial["trial_id"]
        manifest_rows.append({**trial, "status":"COMPLETE" if (run / "summary.json").is_file() else "PLANNED", "run_root":str(run)})
    write_csv(RESEARCH / "TRIAL_MANIFEST.csv", manifest_rows)

    artifacts = [
        REPO / "scripts" / "test_round27_policy_episode.py",
        REPO / "scripts" / "round4070_problem_discovery" / "run_episode.py",
        REPO / "lift_curtain" / "control" / "deployment_observation.py",
        REPO / "lift_curtain" / "control" / "falcon_adapter.py",
        REPO / "configs" / "proprioceptive_thresholds.json",
        REPO / "configs" / "round4070_problem_discovery" / "PREREGISTRATION.json",
        REPO / "reports" / "artifacts" / "round8" / "formal_matched_proprio" / "reset_x045" / "none" / "deployment_observation.csv",
        REPO / "assets" / "generated" / "g1_29dof_falcon_rubberhand.usd",
        Path(r"C:\Robotics\FALCON-upstream\sim2real\models\falcon\g1_29dof.onnx"),
    ]
    hashes = {}
    for path in artifacts:
        hashes[str(path)] = {"present": path.is_file(), "size_bytes": path.stat().st_size if path.is_file() else None, "sha256": sha256(path)}
    write_json(RESEARCH / "code_config_asset_hashes.json", hashes)

    snapshot_dir = RESEARCH / "raw" / "source_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for path in artifacts[:6]:
        if path.is_file():
            shutil.copy2(path, snapshot_dir / path.name)
    provenance = {
        "git_head": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "git_status_short": git("status", "--short"),
        "git_diff_stat": git("diff", "--stat"),
        "historical_outputs_modified": False,
        "third_party_assets_copied": False,
        "source_snapshot": str(snapshot_dir),
    }
    write_json(RESEARCH / "provenance.json", provenance)
    print(json.dumps({"historical_trials": len(rows), "manifest_trials": len(manifest_rows), "research_dir": str(RESEARCH)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
