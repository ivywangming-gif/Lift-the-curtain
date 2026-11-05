"""Audit Round 4070 trial artifacts without changing raw outputs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "outputs" / "round4070_problem_discovery"
TRIALS = OUTPUT / "trials"
RESEARCH = REPO / "research" / "round4070_problem_discovery"
REQUIRED_FILES = (
    "trial_metadata.json",
    "command.json",
    "summary.json",
    "events.json",
    "telemetry.csv",
    "deployment_observation.csv",
    "resolved_config.json",
    "resource_summary.json",
    "watchdog_summary.json",
)
FORBIDDEN_DEPLOYMENT_HEADERS = (
    "evaluation_",
    "object_kind",
    "scene_name",
    "scenario",
    "true_clearance",
    "whole_body_clearance",
    "cloth_",
    "contact_truth",
)
JSON_VECTOR_COLUMNS = (
    "deployment_q_actual_json",
    "deployment_qdot_json",
    "deployment_q_cmd_json",
    "deployment_motor_effort_json",
    "deployment_previous_action_json",
    "deployment_hand_fk_position_w_json",
    "deployment_hand_fk_velocity_w_json",
    "deployment_left_hand_fk_position_w_json",
    "deployment_left_hand_fk_velocity_w_json",
    "deployment_right_hand_fk_position_w_json",
    "deployment_right_hand_fk_velocity_w_json",
    "deployment_base_position_w_json",
    "deployment_base_velocity_b_json",
)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def finite_json_value(value: Any) -> bool:
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(finite_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(finite_json_value(item) for item in value.values())
    return value is None or isinstance(value, (str, bool))


def bool_text(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def planned_ids() -> set[str]:
    ids: set[str] = set()
    for path in (
        REPO / "configs" / "round4070_problem_discovery" / "PREREGISTRATION.json",
        REPO / "configs" / "round4070_problem_discovery" / "CONFIRMATORY_PREREGISTRATION.json",
    ):
        payload = read_json(path, {}) or {}
        ids.update(str(item.get("trial_id")) for item in payload.get("trials", []))
    return ids


def audit_trial(run: Path, evidence: dict[str, dict[str, str]]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    for name in REQUIRED_FILES:
        if not (run / name).is_file():
            errors.append(f"missing:{name}")
    metadata = read_json(run / "trial_metadata.json", {}) or {}
    summary = read_json(run / "summary.json")
    runtime_valid = bool(isinstance(summary, dict) and summary.get("runtime_status") == "RUNTIME_PASS_FINALIZE_REACHED")
    telemetry: list[dict[str, str]] = []
    deployment: list[dict[str, str]] = []
    if (run / "telemetry.csv").is_file():
        try:
            telemetry = read_csv(run / "telemetry.csv")
        except (OSError, csv.Error) as exc:
            errors.append(f"telemetry_parse:{exc}")
    if (run / "deployment_observation.csv").is_file():
        try:
            deployment = read_csv(run / "deployment_observation.csv")
        except (OSError, csv.Error) as exc:
            errors.append(f"deployment_parse:{exc}")
    deployment_headers = set(deployment[0]) if deployment else set()
    forbidden_headers = sorted(
        header for header in deployment_headers
        if any(token in header.lower() for token in FORBIDDEN_DEPLOYMENT_HEADERS)
    )
    if forbidden_headers:
        errors.append("deployment_forbidden_headers:" + "|".join(forbidden_headers))
    if not deployment:
        errors.append("deployment_observation_empty")
    if not telemetry:
        errors.append("telemetry_empty")

    vector_parse_errors = 0
    nonfinite_vectors = 0
    for row in telemetry:
        for column in JSON_VECTOR_COLUMNS:
            raw = row.get(column, "")
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                vector_parse_errors += 1
                continue
            if not finite_json_value(parsed):
                nonfinite_vectors += 1
    if vector_parse_errors:
        errors.append(f"telemetry_vector_parse_errors:{vector_parse_errors}")
    if nonfinite_vectors:
        errors.append(f"telemetry_nonfinite_vectors:{nonfinite_vectors}")

    probe_rows = [row for row in telemetry if row.get("phase") in {"physical_probe", "physical_probe_hold"}]
    probe_sources = sorted({row.get("deployment_effort_source", "") for row in probe_rows})
    if runtime_valid and (not probe_rows or not probe_sources or "unavailable" in probe_sources):
        errors.append("valid_trial_probe_effort_source_invalid")
    if not runtime_valid and summary is not None:
        warnings.append("runtime_invalid_not_assessable")
    if summary is None:
        warnings.append("missing_summary_not_assessable")

    expected = evidence.get(run.name)
    if expected is None:
        warnings.append("unregistered_trial_directory")
    else:
        if bool_text(expected.get("runtime_valid")) != runtime_valid:
            errors.append("ledger_runtime_valid_mismatch")
        if str(metadata.get("trial_id")) != run.name:
            errors.append("metadata_trial_id_mismatch")
        if str(metadata.get("probe_side")) != str(expected.get("probe_side")):
            errors.append("metadata_probe_side_mismatch")
        if str(metadata.get("scenario_for_scene_builder")) != str(expected.get("scenario")):
            errors.append("metadata_scenario_mismatch")

    runtime_artifact_errors = {
        "missing:events.json",
        "missing:telemetry.csv",
        "missing:deployment_observation.csv",
        "deployment_observation_empty",
        "telemetry_empty",
    }
    hard_errors = [error for error in errors if error not in runtime_artifact_errors]
    status = "PASS" if not errors and runtime_valid else "RUNTIME_INVALID_NOT_ASSESSABLE" if not runtime_valid and not hard_errors else "FAIL"
    return {
        "trial_id": run.name,
        "stage": metadata.get("stage"),
        "scenario": metadata.get("scenario_for_scene_builder"),
        "runtime_valid": runtime_valid,
        "artifact_count": sum((run / name).is_file() for name in REQUIRED_FILES),
        "telemetry_rows": len(telemetry),
        "deployment_rows": len(deployment),
        "probe_effort_sources": "|".join(probe_sources),
        "forbidden_deployment_headers": "|".join(forbidden_headers),
        "status": status,
        "errors": "|".join(errors),
        "warnings": "|".join(warnings),
        "path": str(run),
    }


def main() -> None:
    RESEARCH.mkdir(parents=True, exist_ok=True)
    evidence_rows = read_csv(RESEARCH / "EVIDENCE_MATRIX.csv")
    evidence = {row["trial_id"]: row for row in evidence_rows}
    runs = sorted(path for path in TRIALS.iterdir() if path.is_dir()) if TRIALS.is_dir() else []
    rows = [audit_trial(run, evidence) for run in runs]
    expected_ids = planned_ids()
    actual_ids = {row["trial_id"] for row in rows}
    missing_planned = sorted(expected_ids - actual_ids)
    unregistered = sorted(actual_ids - expected_ids)
    global_errors = []
    if missing_planned:
        global_errors.append("missing_planned_trials:" + "|".join(missing_planned))
    if unregistered:
        global_errors.append("unregistered_trial_directories:" + "|".join(unregistered))
    if any(row["status"] == "FAIL" for row in rows):
        global_errors.append("one_or_more_trial_integrity_failures")
    summary = {
        "schema": "lift-the-curtain.round4070-trial-integrity-audit.v1",
        "status": "PASS" if not global_errors else "FAIL",
        "planned_trial_count": len(expected_ids),
        "observed_trial_directory_count": len(rows),
        "valid_trial_count": sum(row["runtime_valid"] for row in rows),
        "runtime_invalid_trial_count": sum(not row["runtime_valid"] for row in rows),
        "pass_count": sum(row["status"] == "PASS" for row in rows),
        "runtime_invalid_not_assessable_count": sum(row["status"] == "RUNTIME_INVALID_NOT_ASSESSABLE" for row in rows),
        "fail_count": sum(row["status"] == "FAIL" for row in rows),
        "deployment_truth_leakage_detected": any(bool(row["forbidden_deployment_headers"]) for row in rows),
        "global_errors": global_errors,
        "rows": rows,
    }
    (RESEARCH / "TRIAL_INTEGRITY_AUDIT.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (RESEARCH / "TRIAL_INTEGRITY_AUDIT.csv").open("w", encoding="utf-8", newline="") as stream:
        names = sorted({key for row in rows for key in row}) or ["status"]
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: summary[key] for key in ("status", "planned_trial_count", "observed_trial_directory_count", "valid_trial_count", "runtime_invalid_trial_count", "fail_count")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
