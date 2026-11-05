"""Validate the frozen Round 4070-E contract and observed execution evidence."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "round4070_fixed_probe_benchmark"
RESEARCH = REPO / "research" / "round4070_fixed_probe_benchmark"
OUTPUT = REPO / "outputs" / "round4070_fixed_probe_benchmark"

P0_CONDITIONS = {
    "FABRIC_NOMINAL",
    "FABRIC_BOTTOM_LEFT_ANCHOR",
    "FABRIC_BOTTOM_RIGHT_ANCHOR",
    "FABRIC_BOTH_BOTTOM_CORNERS",
    "FABRIC_FOUR_CORNERS_ONLY",
    "FABRIC_FULL_PERIMETER_FIXED",
    "FABRIC_HEAVY_FREE",
    "FABRIC_HEAVY_CONSTRAINED",
    "FABRIC_WALL_GAP_02",
    "FABRIC_WALL_GAP_05",
    "FABRIC_WALL_GAP_08",
    "EMPTY_CONTROL",
    "RIGID_CONTROL",
}
FORBIDDEN_DEPLOYMENT_FIELDS = {
    "condition_id",
    "condition_family",
    "scenario",
    "material",
    "hidden_rigid_x",
    "hidden_rigid_gap_m",
    "structural_topology",
    "density_scale_multiplier",
    "simulator_contact_flag",
    "true_passability",
    "future_clearance",
    "task_success",
    "evaluation_hand_contact_force_norm_N",
    "evaluation_geometry_snapshot_json",
}


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


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "pass"}


def command_arg(command: dict[str, Any], name: str) -> str | None:
    args = list(command.get("args") or [])
    for index, value in enumerate(args[:-1]):
        if str(value) == name:
            return str(args[index + 1])
    return None


def trial_status(plan: dict[str, Any]) -> dict[str, Any]:
    trial_id = str(plan["trial_id"])
    root = OUTPUT / "trials" / trial_id
    summary_path = root / "summary.json"
    metadata = read_json(root / "trial_metadata.json", {}) or {}
    command = read_json(root / "command.json", {}) or {}
    summary = read_json(summary_path, {}) or {}
    observed_step = finite(command_arg(command, "--probe-10cm-joint-step-rad"))
    expected_step = finite(plan.get("probe_10cm_joint_step_rad"))
    observed_target = finite(summary.get("probe_target_displacement_m"))
    expected_target = finite(plan.get("target_depth_m"))
    problems: list[str] = []
    if observed_step is None or expected_step is None or abs(observed_step - expected_step) > 1.0e-9:
        problems.append(f"joint_step expected={expected_step}; observed={observed_step}")
    if observed_target is None or expected_target is None or abs(observed_target - expected_target) > 1.0e-9:
        problems.append(f"target expected={expected_target}; observed={observed_target}")
    if summary and summary.get("probe_depth_level") != "PROBE_10CM":
        problems.append(f"mode observed={summary.get('probe_depth_level')}")
    if command and "FIXED_ONE_SHOT" not in str(metadata.get("probe_contract", "")):
        problems.append(f"probe contract={metadata.get('probe_contract')}")
    return {
        "trial_id": trial_id,
        "summary_exists": summary_path.is_file(),
        "runtime_valid": bool_value(summary.get("runtime_valid")) or summary.get("runtime_status") == "RUNTIME_PASS_FINALIZE_REACHED",
        "probe_valid": bool_value(summary.get("probe_valid")),
        "actual_m": finite(summary.get("actual_hand_displacement_m", summary.get("probe_actual_displacement_m"))),
        "target_m": observed_target,
        "stop_reason": summary.get("probe_stop_reason"),
        "safe_recovery": summary.get("safe_recovery"),
        "protocol_match": not problems,
        "protocol_detail": "; ".join(problems),
    }


def check_deployment_headers(trial_id: str) -> list[str]:
    path = OUTPUT / "trials" / trial_id / "deployment_observation.csv"
    if not path.is_file():
        return []
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            headers = set(csv.DictReader(handle).fieldnames or [])
    except (OSError, csv.Error):
        return ["<unreadable_csv>"]
    return sorted(headers & FORBIDDEN_DEPLOYMENT_FIELDS)


def main() -> None:
    manifest = read_json(CONFIG / "PREREGISTERED_MANIFEST.json", {}) or {}
    trials = list(manifest.get("trials", []))
    probe_contract = manifest.get("probe_contract") or {}
    checks: list[dict[str, Any]] = []

    ids = [str(item.get("trial_id")) for item in trials]
    checks.append({"check": "manifest_count_within_budget", "observed": len(trials), "required": "<= 200", "pass": 0 < len(trials) <= 200})
    checks.append({"check": "manifest_ids_unique", "observed": len(ids), "required": len(set(ids)), "pass": len(ids) == len(set(ids))})
    checks.append({"check": "all_registered_probe_modes", "observed": sorted({item.get("probe_depth_level") for item in trials}), "required": ["PROBE_10CM"], "pass": {item.get("probe_depth_level") for item in trials} == {"PROBE_10CM"}})
    checks.append({"check": "all_registered_targets", "observed": sorted({item.get("target_depth_m") for item in trials}), "required": [0.1], "pass": {item.get("target_depth_m") for item in trials} == {0.1}})
    checks.append({"check": "fixed_command_contract", "observed": probe_contract.get("command_mode"), "required": "FIXED_ONE_SHOT_NOMINAL_0P10M", "pass": probe_contract.get("command_mode") == "FIXED_ONE_SHOT_NOMINAL_0P10M" and probe_contract.get("actual_feedback_is_not_probe_validity_gate") is True})
    observed_conditions = {str(item.get("condition_id")) for item in trials}
    checks.append({"check": "p0_condition_coverage", "observed": sorted(observed_conditions & P0_CONDITIONS), "required": sorted(P0_CONDITIONS), "pass": P0_CONDITIONS <= observed_conditions})

    phase0 = [item for item in trials if item.get("phase") == "PHASE_0"]
    phase1 = [item for item in trials if item.get("phase") == "PHASE_1"]
    checks.append({"check": "phase0_minimum_six", "observed": len(phase0), "required": 6, "pass": len(phase0) == 6})
    side_repeat_counts = {
        (str(item.get("condition_id")), str(item.get("probe_side"))): sum(
            candidate.get("condition_id") == item.get("condition_id") and candidate.get("probe_side") == item.get("probe_side")
            for candidate in phase1
        )
        for item in phase1
    }
    required_phase1 = {(condition, side) for condition in P0_CONDITIONS for side in ("left", "right")}
    checks.append({"check": "phase1_two_seeds_per_side", "observed_min": min(side_repeat_counts.get(key, 0) for key in required_phase1), "required": 2, "pass": all(side_repeat_counts.get(key, 0) >= 2 for key in required_phase1)})
    gap_values = sorted({float(item["hidden_rigid_gap_m"]) for item in trials if item.get("condition_family") == "hidden_wall" and item.get("hidden_rigid_gap_m") is not None})
    checks.append({"check": "hidden_wall_gap_set", "observed": gap_values, "required": [0.02, 0.05, 0.08], "pass": gap_values == [0.02, 0.05, 0.08]})

    runner_text = (REPO / "scripts" / "round4070_fixed_probe_benchmark" / "run_episode.py").read_text(encoding="utf-8")
    old_schedule_tokens = ("D0_BASELINE", "D1_PLUS_020M", "D2_PLUS_040M", "D3_PLUS_060M")
    checks.append({"check": "old_depth_schedule_absent", "observed": any(token in runner_text for token in old_schedule_tokens), "required": False, "pass": not any(token in runner_text for token in old_schedule_tokens)})
    checks.append({"check": "actual_motion_not_validity_gate_in_runner", "observed": "probe_valid" in runner_text and "actual hand" in runner_text.lower(), "required": True, "pass": "probe_valid" in runner_text and "actual hand" in runner_text.lower()})

    statuses = [trial_status(item) for item in trials]
    phase0_status = [status for status in statuses if next((item for item in phase0 if str(item.get("trial_id")) == status["trial_id"]), None)]
    phase1_formal = [item for item in trials if item.get("phase") == "PHASE_1" and item.get("formal_core")]
    phase1_status = [status for status in statuses if status["trial_id"] in {str(item["trial_id"]) for item in phase1_formal}]
    checks.append({"check": "phase0_recorded_and_valid", "observed": {"summary": sum(s["summary_exists"] for s in phase0_status), "runtime": sum(s["runtime_valid"] for s in phase0_status), "probe": sum(s["probe_valid"] for s in phase0_status)}, "required": "6 summaries, runtime-valid and probe-valid", "pass": len(phase0_status) == 6 and all(s["summary_exists"] and s["runtime_valid"] and s["probe_valid"] and s["protocol_match"] and s["safe_recovery"] is True for s in phase0_status)})
    checks.append({"check": "phase1_formal_initial_screen_recorded", "observed": {"registered": len(phase1_formal), "summary": sum(s["summary_exists"] for s in phase1_status), "runtime": sum(s["runtime_valid"] for s in phase1_status), "probe": sum(s["probe_valid"] for s in phase1_status)}, "required": "all registered Phase 1 formal rows", "pass": len(phase1_status) == len(phase1_formal) and all(s["summary_exists"] and s["runtime_valid"] and s["probe_valid"] and s["protocol_match"] for s in phase1_status)})
    forbidden: dict[str, list[str]] = {}
    for item in trials:
        leaked = check_deployment_headers(str(item["trial_id"]))
        if leaked:
            forbidden[str(item["trial_id"])] = leaked
    checks.append({"check": "deployment_header_leakage", "observed": forbidden, "required": "no forbidden fields", "pass": not forbidden})

    gate_status = read_json(CONFIG / "PHYSICS_GATE_STATUS.json", {}) or {}
    p0_gate_names = sorted({str(item.get("physics_gate_id")) for item in trials if item.get("condition_family") not in {"control", "matched_no_wall"}})
    gate_values = {name: (gate_status.get("gates", {}).get(name) or {}).get("status") for name in p0_gate_names}
    checks.append({"check": "p0_physics_gates_pass", "observed": gate_values, "required": "all PASS_E_PHYSICS", "pass": all(value in {"PASS_E_PHYSICS", "PASS"} for value in gate_values.values())})

    passed = all(bool(item["pass"]) for item in checks)
    status = "PASS" if passed else "FAIL_CONTRACT_OR_REQUIRED_INITIAL_EVIDENCE"
    payload = {
        "schema": "lift-the-curtain.round4070-fixed-probe-contract-validation.v1",
        "status": status,
        "checks": checks,
        "observed": {
            "registered_trials": len(trials),
            "summaries": sum(item["summary_exists"] for item in statuses),
            "runtime_valid": sum(item["runtime_valid"] for item in statuses),
            "probe_valid": sum(item["probe_valid"] for item in statuses),
        },
    }
    (RESEARCH / "stats").mkdir(parents=True, exist_ok=True)
    (RESEARCH / "stats" / "contract_validation.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Round 4070-E Contract Validation",
        "",
        f"Status: **{status}**",
        "",
        "The nominal 0.10 m target is fixed in the command contract. Actual FK displacement is recorded as an outcome and is not a runtime/probe validity gate.",
        "",
        "| Check | Result | Observed |",
        "|---|---|---|",
    ]
    for item in checks:
        observed = json.dumps(item.get("observed", item.get("observed_min", "")), ensure_ascii=False)
        lines.append(f"| {item['check']} | {'PASS' if item['pass'] else 'FAIL'} | `{observed}` |")
    (RESEARCH / "CONTRACT_VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
