"""Aggregate Final RTX 4070 trials without using evaluation truth as features."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return default


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
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_trial(run: Path) -> dict[str, Any]:
    metadata = read_json(run / "trial_metadata.json", {}) or {}
    summary = read_json(run / "summary.json")
    watchdog = read_json(run / "watchdog_summary.json", {}) or {}
    resource = read_json(run / "resource_summary.json", {}) or {}
    record: dict[str, Any] = {
        "trial_id": run.name,
        "batch": metadata.get("batch"), "condition_id": metadata.get("condition_id"),
        "evidence_level": metadata.get("evidence_level"), "scenario": metadata.get("scenario_for_scene_builder"),
        "policy": metadata.get("policy"), "probe_side": metadata.get("probe_side"), "seed": metadata.get("seed"),
        "reset_root_y": metadata.get("reset_root_y"), "reset_root_yaw": metadata.get("reset_root_yaw"),
        "watchdog_timeout_reason": watchdog.get("timeout_reason"),
        "process_exit_code": metadata.get("exit_code", watchdog.get("exit_code")), "runtime_validity": False,
    }
    if not isinstance(summary, dict):
        record.update({"runtime_class": "RUNTIME_INVALID", "error": "missing_or_invalid_summary"})
        return record
    runtime_status = summary.get("runtime_status")
    runtime_valid = bool(runtime_status == "RUNTIME_PASS_FINALIZE_REACHED" and (run / "events.json").is_file())
    probe = summary.get("physical_probe_result") or {}
    decision = probe.get("decision")
    clearance = bool(summary.get("candidate_clearance_verified", summary.get("whole_body_candidate_clearance", False)))
    pointgoal = bool(summary.get("pointgoal_success", False))
    safe_recovery = bool(summary.get("safe_recovery", False))
    detour = bool(summary.get("detour_waypoints_reached", False))
    contact = bool(probe.get("evaluation_hand_contact_confirmed", False))
    bounded = probe.get("bounded_advance_metrics") or summary.get("bounded_advance_metrics") or []
    progress = [finite(item.get("base_progress_m")) for item in bounded if isinstance(item, dict)]
    progress = [value for value in progress if value is not None]
    execution_progress = bool(progress and max(progress) >= 0.012)
    sustained_progress = bool(progress and sum(max(0.0, value) for value in progress) >= 0.10)
    labels: list[str] = []
    if not runtime_valid: labels.append("RUNTIME_INVALID")
    elif metadata.get("scenario_for_scene_builder") == "empty": labels.append("CONTROL_REFERENCE")
    elif not contact: labels.append("S1_NO_CONTACT")
    elif decision == "RESPONSIVE" and clearance: labels.append("S4_LOCAL_EFFECT_WHOLE_BODY_PASS")
    elif decision == "RESPONSIVE": labels.append("S3_LOCAL_EFFECT_NO_WHOLE_BODY_PASS")
    else: labels.append("S2_LOCAL_EFFECT_NOT_ESTABLISHED")
    if safe_recovery: labels.append("SAFE_RETRACT")
    if detour and pointgoal and not clearance: labels.append("DETOUR_SUCCESS")
    if clearance and not pointgoal: labels.append("POST_CLEARANCE_GOAL_FAILED")
    policy_decisions = read_json(run / "policy_decisions.json", []) or []
    record.update({
        "runtime_validity": runtime_valid, "runtime_class": "VALID_EPISODE" if runtime_valid else "RUNTIME_INVALID",
        "status": summary.get("status"), "contact_acquisition": contact, "local_response": decision,
        "execution_progress": execution_progress, "sustained_progress": sustained_progress,
        "whole_body_clearance": clearance, "safe_retract": safe_recovery,
        "detour_success": bool(detour and pointgoal and not clearance), "pointgoal_success": pointgoal,
        "goal_error_m": finite(summary.get("goal_error_m")), "safety_success": bool(summary.get("safety_success", False)),
        "fall": not bool(summary.get("finite_no_fall", True)), "navigation_map_update_count": summary.get("navigation_map_update_count"),
        "raw_actions": [item.get("raw_action") for item in policy_decisions if isinstance(item, dict)],
        "filtered_actions": [item.get("filtered_action") for item in policy_decisions if isinstance(item, dict)],
        "policy_decision_count": len(policy_decisions), "labels": labels, "resource": resource,
        "paths": {name: str((run / filename).as_posix()) for name, filename in {
            "summary": "summary.json", "events": "events.json", "deployment": "deployment_observation.csv",
            "telemetry": "telemetry.csv", "resource": "resource_telemetry.csv"}.items()},
    })
    return record


def deployment_cutoff_features(run: Path, record: dict[str, Any]) -> list[dict[str, Any]]:
    telemetry = run / "telemetry.csv"
    if not telemetry.is_file() or not record.get("runtime_validity"): return []
    with telemetry.open(encoding="utf-8-sig", newline="") as handle: rows = list(csv.DictReader(handle))
    groups = {
        "T0": [row for row in rows if "physical_preprobe_hold" in row.get("phase", "")],
        "T1": [row for row in rows if "physical_probe" in row.get("phase", "")],
        "T2": [row for row in rows if "cautious_advance" in row.get("phase", "") or "responsive_traverse" in row.get("phase", "")],
    }
    def vector(row: dict[str, str], key: str) -> list[float]:
        try: return [float(value) for value in json.loads(row.get(key, "[]"))]
        except (TypeError, ValueError, json.JSONDecodeError): return []
    output = []
    for cutoff, selected in groups.items():
        if not selected: continue
        first, last = selected[0], selected[-1]
        base0, base1 = vector(first, "deployment_base_position_w_json"), vector(last, "deployment_base_position_w_json")
        hand0, hand1 = vector(first, "deployment_hand_fk_position_w_json"), vector(last, "deployment_hand_fk_position_w_json")
        tracking = [finite(row.get("upper_tracking_rms_rad")) for row in selected]
        tracking = [value for value in tracking if value is not None]
        efforts, speeds, commands = [], [], []
        for row in selected:
            values = vector(row, "deployment_motor_effort_json")
            if values: efforts.append(math.sqrt(sum(value * value for value in values) / len(values)))
            speed = finite(row.get("measured_planar_speed")); command = finite(row.get("command_vx"))
            if speed is not None: speeds.append(speed)
            if command is not None: commands.append(command)
        output.append({
            "trial_id": record["trial_id"], "scenario": record.get("scenario"), "policy": record.get("policy"),
            "probe_side": record.get("probe_side"), "seed": record.get("seed"), "cutoff": cutoff, "sample_count": len(selected),
            "base_delta_m": math.dist(base0[:2], base1[:2]) if len(base0)>=2 and len(base1)>=2 else None,
            "hand_forward_delta_m": hand1[0]-hand0[0] if hand0 and hand1 else None,
            "hand_lateral_delta_m": hand1[1]-hand0[1] if len(hand0)>=2 and len(hand1)>=2 else None,
            "upper_tracking_rms_mean": sum(tracking)/len(tracking) if tracking else None,
            "effort_rms_mean": sum(efforts)/len(efforts) if efforts else None,
            "measured_speed_mean_mps": sum(speeds)/len(speeds) if speeds else None,
            "command_vx_mean_mps": sum(commands)/len(commands) if commands else None,
            "label_clearance": record.get("whole_body_clearance"), "label_pointgoal": record.get("pointgoal_success"),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2]); args = parser.parse_args()
    repo = args.repo.resolve(); output = repo / "outputs" / "g1_final4070"; trials = output / "trials"; output.mkdir(parents=True, exist_ok=True)
    records = [classify_trial(path) for path in sorted(trials.iterdir()) if path.is_dir()] if trials.is_dir() else []
    valid = [record for record in records if record.get("runtime_validity")]
    features = [feature for record in valid for feature in deployment_cutoff_features(trials / record["trial_id"], record)]
    labels = Counter(label for record in records for label in record.get("labels", []))
    by_scenario: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in valid:
        group = by_scenario[str(record.get("scenario"))]; group["n"] += 1
        group["clearance"] += int(bool(record.get("whole_body_clearance"))); group["pointgoal"] += int(bool(record.get("pointgoal_success")))
        group["responsive"] += int(record.get("local_response") == "RESPONSIVE"); group["blocked"] += int(record.get("local_response") == "BLOCKED")
    index = {"schema":"lift-the-curtain.g1-final4070-trial-index.v1", "launched_trial_directories":len(records),
             "valid_completed_trials":len(valid), "runtime_invalid_trials":len(records)-len(valid), "label_counts":dict(labels),
             "scenario_counts":{key:dict(value) for key,value in by_scenario.items()}, "records":records}
    (output/"trial_index.json").write_text(json.dumps(index,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    if records:
        flat = []
        for record in records:
            row = {key:value for key,value in record.items() if key not in {"resource","paths","raw_actions","filtered_actions","labels"}}
            row["labels"]="|".join(record.get("labels",[])); row["raw_actions"]="|".join(str(item) for item in record.get("raw_actions",[])); row["filtered_actions"]="|".join(str(item) for item in record.get("filtered_actions",[]))
            for key,value in (record.get("resource") or {}).items():
                if not isinstance(value,(dict,list)): row[f"resource_{key}"]=value
            flat.append(row)
        fields=sorted({key for row in flat for key in row})
        with (output/"trial_index.csv").open("w",encoding="utf-8",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(flat)
    if features:
        fields=sorted({key for row in features for key in row})
        with (output/"deployment_cutoff_features.csv").open("w",encoding="utf-8",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(features)
    resources=[record for record in valid if record.get("resource")]
    resource_profile={"schema":"lift-the-curtain.g1-final4070-resource-profile.v1","trials_with_resource_telemetry":len(resources),
                      "records":[{"trial_id":record["trial_id"],"scenario":record.get("scenario"),**record["resource"]} for record in resources],
                      "direct_resource_failure_evidence":any(any(token in json.dumps(record,ensure_ascii=False).lower() for token in ("out of memory","cuda allocation","device lost")) for record in records)}
    (output/"resource_profile.json").write_text(json.dumps(resource_profile,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    hashes={}
    for relative in ("scripts/test_round27_policy_episode.py","scripts/g1_final4070/run_physical_trial.ps1","scripts/g1_final4070/run_guarded_with_resources.ps1","scripts/g1_final4070/analyze_campaign.py","lift_curtain/navigation/round25b_policy_interface.py","lift_curtain/navigation/round27_policy_interface.py","configs/proprioceptive_thresholds.json"):
        path=repo/relative; hashes[relative]={"present":path.is_file(),"sha256":sha256(path)}
    (output/"code_config_hashes.json").write_text(json.dumps(hashes,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    manifest={"schema":"lift-the-curtain.g1-final4070-run-manifest.v1","status":"PHYSICAL_CAMPAIGN_IN_PROGRESS","launch_limit":48,"target_valid_trials":[18,24],
              "trial_accounting":{"launched":len(records),"valid":len(valid),"runtime_invalid":len(records)-len(valid)},
              "hypotheses":{"P1":"PENDING","P2":"PENDING","P3":"PENDING"},"historical_outputs_modified":False,"third_party_assets_copied":False}
    (output/"run_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(manifest["trial_accounting"],ensure_ascii=False))


if __name__ == "__main__": main()

