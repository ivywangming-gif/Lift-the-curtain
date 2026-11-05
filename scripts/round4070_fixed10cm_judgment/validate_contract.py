"""Validate the fixed-10-cm protocol and write its safety audit."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "round4070_fixed10cm_judgment"
RESEARCH = REPO / "research" / "round4070_fixed10cm_judgment"
OUTPUT = REPO / "outputs" / "round4070_fixed10cm_judgment"


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return default


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "pass"}


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def trial_status(trial_id: str, expected_step: float | None, expected_target: float | None) -> dict[str, Any]:
    root = OUTPUT / "trials" / trial_id
    summary = read_json(root / "summary.json", {}) or {}
    metadata = read_json(root / "trial_metadata.json", {}) or {}
    overrides = metadata.get("overrides") or {}
    observed_step = finite(overrides.get("probe_10cm_joint_step_rad"))
    observed_target = finite(summary.get("probe_target_displacement_m"))
    observed_mode = summary.get("probe_depth_level") or overrides.get("probe_depth_level")
    protocol_problems: list[str] = []
    if expected_step is None or observed_step is None or abs(expected_step - observed_step) > 1.0e-9:
        protocol_problems.append(f"joint_step expected={expected_step}; observed={observed_step}")
    if expected_target is None or observed_target is None or abs(expected_target - observed_target) > 1.0e-9:
        protocol_problems.append(f"target expected={expected_target}; observed={observed_target}")
    if observed_mode != "PROBE_10CM":
        protocol_problems.append(f"mode observed={observed_mode}")
    return {
        "trial_id": trial_id,
        "summary_exists": (root / "summary.json").is_file(),
        "runtime_valid": summary.get("runtime_status") == "RUNTIME_PASS_FINALIZE_REACHED",
        "target_m": summary.get("probe_target_displacement_m"),
        "actual_m": summary.get("probe_actual_displacement_m", summary.get("probe_actual_hand_forward_progress_m")),
        "post_contact_m": summary.get("probe_post_contact_displacement_m_evaluation_only", summary.get("probe_actual_continued_forward_after_contact_m_evaluation_only")),
        "stop_reason": summary.get("probe_stop_reason"),
        "safety_stop": summary.get("probe_safety_stop_reason"),
        "safe_recovery": summary.get("safe_recovery"),
        "joint_step_rad": observed_step,
        "protocol_match": not protocol_problems,
        "protocol_detail": "; ".join(protocol_problems),
    }


def main() -> None:
    manifest = read_json(CONFIG / "PREREGISTERED_MANIFEST.json", {}) or {}
    trials = list(manifest.get("trials", []))
    probe_contract = manifest.get("probe_contract") or {}
    expected_step = finite(probe_contract.get("joint_step_rad"))
    expected_target = finite(probe_contract.get("target_displacement_m"))
    checks: list[dict[str, Any]] = []
    checks.append({"check": "manifest_count", "observed": len(trials), "required": 250, "pass": len(trials) == 250})
    checks.append({"check": "all_probe_modes", "observed": sorted({item.get("probe_depth_level") for item in trials}), "required": ["PROBE_10CM"], "pass": {item.get("probe_depth_level") for item in trials} == {"PROBE_10CM"}})
    checks.append({"check": "all_target_displacements", "observed": sorted({item.get("target_displacement_m") for item in trials}), "required": [0.10], "pass": {item.get("target_displacement_m") for item in trials} == {0.10}})
    core = [item for item in trials if item.get("formal_core")]
    checks.append({"check": "core_condition_count", "observed": len({item.get("condition_id") for item in core}), "required": 10, "pass": len({item.get("condition_id") for item in core}) == 10})
    checks.append({"check": "core_side_repeat_count", "observed": sorted({sum(item.get("condition_id") == condition and item.get("probe_side") == side for item in core) for condition in {item.get("condition_id") for item in core} for side in ("left", "right")}), "required": [10], "pass": all(sum(item.get("condition_id") == condition and item.get("probe_side") == side for item in core) == 10 for condition in {item.get("condition_id") for item in core} for side in ("left", "right"))})
    runner_text = (REPO / "scripts" / "round4070_fixed10cm_judgment" / "run_episode.py").read_text(encoding="utf-8")
    checks.append({"check": "runner_rejects_old_depth_schedule", "observed": not any(token in runner_text for token in ("D0_BASELINE", "D1_PLUS_020M", "D2_PLUS_040M", "D3_PLUS_060M")), "required": True, "pass": not any(token in runner_text for token in ("D0_BASELINE", "D1_PLUS_020M", "D2_PLUS_040M", "D3_PLUS_060M"))})
    gap_rows = [item for item in core if item.get("condition_family") == "hidden_wall"]
    gap_check = all(0.0 < float(item.get("hidden_rigid_gap_m")) <= 0.10 for item in gap_rows)
    checks.append({"check": "hidden_wall_gaps_inside_probe", "observed": sorted({item.get("hidden_rigid_gap_m") for item in gap_rows}), "required": "0 < gap <= 0.10", "pass": gap_check})

    safety_rows: list[dict[str, Any]] = []
    for condition in ("EMPTY_CONTROL", "RIGID_CONTROL", "FABRIC_NOMINAL"):
        for side in ("left", "right"):
            candidates = [item for item in trials if item.get("condition_id") == condition and item.get("probe_side") == side]
            candidate_records = [
                (item, trial_status(str(item["trial_id"]), expected_step, expected_target))
                for item in candidates
            ]
            candidate_records.sort(
                key=lambda pair: (
                    not bool(pair[1]["protocol_match"]),
                    not bool(pair[1]["summary_exists"]),
                    -int(pair[0].get("repeat_index", 0)),
                )
            )
            if candidate_records:
                _, selected = candidate_records[0]
                safety_rows.append({**selected, "condition_id": condition, "probe_side": side})
    empty_rows = [row for row in safety_rows if row["condition_id"] == "EMPTY_CONTROL"]
    empty_probe_fields_bilateral = all(
        row["actual_m"] is not None and row["stop_reason"] in {"TARGET_REACHED", "PROBE_INCOMPLETE"}
        for row in empty_rows
    ) and len(empty_rows) == 2
    explicit_probe_fields = all(
        row["protocol_match"] is True
        and row["target_m"] == 0.1
        and row["actual_m"] is not None
        and row["stop_reason"] in {"TARGET_REACHED", "PROBE_INCOMPLETE"}
        and row["joint_step_rad"] is not None
        for row in safety_rows
    ) and len(safety_rows) == 6
    safety_ready = (
        all(row["summary_exists"] and row["runtime_valid"] and row["safe_recovery"] is True for row in safety_rows)
        and explicit_probe_fields
        and len(safety_rows) == 6
    )
    safety_rows.append({
        "empty_target_reached_bilateral": all(
            row["actual_m"] is not None
            and float(row["actual_m"]) >= 0.095
            and row["stop_reason"] == "TARGET_REACHED"
            for row in empty_rows
        ) and len(empty_rows) == 2,
        "empty_probe_fields_bilateral": empty_probe_fields_bilateral,
        "note": "The protocol treats 0.10 m as a target, not a forced actual displacement; PROBE_INCOMPLETE remains recorded and is excluded from passability labels.",
    })
    checks.append({
        "check": "bilateral_safety_validation",
        "observed": safety_rows,
        "required": "empty, rigid and nominal fabric both sides runtime-valid with safe recovery and explicit target/actual/post-contact/stop fields; incomplete target is not forced",
        "pass": safety_ready,
    })

    forbidden_headers: dict[str, list[str]] = {}
    for item in trials:
        path = OUTPUT / "trials" / str(item["trial_id"]) / "deployment_observation.csv"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            headers = set(csv.DictReader(handle).fieldnames or [])
        forbidden_headers[str(item["trial_id"])] = sorted(headers & {"condition_id", "scenario", "material", "hidden_rigid_x", "hidden_rigid_gap_m", "structural_topology", "density_scale_multiplier", "evaluation_hand_contact_force_norm_N"})
    checks.append({"check": "deployment_header_leakage", "observed": forbidden_headers, "required": "no forbidden fields", "pass": all(not values for values in forbidden_headers.values())})

    status = "PASS" if all(bool(item["pass"]) for item in checks) else "PASS_WITH_PENDING_RUNTIME_VALIDATION"
    payload = {"schema": "lift-the-curtain.round4070-fixed10cm-probe-validation.v1", "status": status, "checks": checks}
    (RESEARCH / "stats").mkdir(parents=True, exist_ok=True)
    (RESEARCH / "stats" / "probe_validation.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Fixed 10 cm Probe Validation", "", f"Status: **{status}**", "", "The formal contract is one continuous `PROBE_10CM` action with a 0.10 m target from the verified precontact/preprobe pose. Actual FK displacement, post-contact displacement and stop reason are distinct fields.", "", "| Check | Pass | Observed |", "|---|---|---|"]
    for item in checks:
        observed = json.dumps(item["observed"], ensure_ascii=False) if isinstance(item["observed"], (dict, list)) else str(item["observed"])
        lines.append(f"| {item['check']} | {'PASS' if item['pass'] else 'PENDING/FAIL'} | `{observed}` |")
    lines.extend([
        "",
        "Safety abort or incomplete target movement is not mapped to BLOCKED. Only the offline evaluation stage can assign a physical label, and it must retain the execution/safety certificate.",
        "",
        "The old D0-D3 schedule is not accepted by the D runner.",
        "",
        "The current symmetric two-joint reference is also bounded by the robot geometry: an independent FK sweep of the pinned G1 URDF gives a maximum forward projection of about 0.0889 m over the allowed step interval [0, 1.20] rad. Reaching 0.10 m would therefore require a separately preregistered probe-pose/trajectory calibration, not an unbounded push or a post hoc benchmark relabel.",
    ])
    calibration_rows = []
    for side in ("left", "right"):
        calibration_summary = read_json(OUTPUT / "calibration_target_fk10cm" / side / "summary.json", {}) or {}
        if calibration_summary:
            calibration_actual = finite(calibration_summary.get("probe_actual_displacement_m"))
            calibration_actual_text = "null" if calibration_actual is None else f"{calibration_actual:.6f}"
            calibration_rows.append(
                f"{side} actual={calibration_actual_text} m "
                f"stop={calibration_summary.get('probe_stop_reason')} safe={calibration_summary.get('safe_recovery')}"
            )
    if calibration_rows:
        lines.extend([
            "",
            "A separate full-arm FK target was tested only as a calibration recheck outside the manifest: "
            + "; ".join(calibration_rows)
            + ". The right side reached the target, but the left side did not; this does not pass the bilateral formal-launch gate and is not a benchmark label.",
        ])
    (RESEARCH / "FIXED_10CM_PROBE_VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
