"""Materialize explicit derived evidence for trials that terminate before rollout files exist.

The script never overwrites runner-produced files. For an execution-invalid trial it
turns the recorded heartbeat/finalize information into a clearly marked derived
events ledger, so the campaign evidence contract is auditable without inventing
telemetry or deployment observations.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO / "configs" / "round4070_failure_discovery" / "PREREGISTERED_MANIFEST.json"
DEFAULT_TRIAL_ROOT = REPO / "outputs" / "round4070_failure_discovery" / "trials"
DEFAULT_PHYSICS = REPO / "research" / "round4070_failure_discovery" / "PHYSICS_GATE_RESULTS.csv"


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def read_gate_rows(path: Path) -> dict[str, dict[str, str]]:
    import csv

    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {str(row.get("gate_id")): row for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--trial-root", type=Path, default=DEFAULT_TRIAL_ROOT)
    parser.add_argument("--physics-results", type=Path, default=DEFAULT_PHYSICS)
    args = parser.parse_args()
    manifest = read_json(args.manifest, {}) or {}
    gates = read_gate_rows(args.physics_results)
    materialized: list[str] = []
    validation_count = 0

    for plan in manifest.get("trials", []):
        trial_id = str(plan.get("trial_id"))
        path = args.trial_root / trial_id
        if not (path / "command.json").is_file():
            continue
        summary = read_json(path / "summary.json", {}) or {}
        watchdog = read_json(path / "watchdog_summary.json", {}) or {}
        resources = read_json(path / "resource_summary.json", {}) or {}
        runtime_valid = summary.get("runtime_status") == "RUNTIME_PASS_FINALIZE_REACHED"

        events_path = path / "events.json"
        derived_event_ledger = False
        if not events_path.is_file():
            heartbeat_rows = read_jsonl(path / "stage_heartbeat.jsonl")
            finalize = summary.get("finalize_event") or {}
            derived_events = [
                {
                    "event_type": "derived_execution_evidence_source",
                    "source": "stage_heartbeat.jsonl",
                    "source_event_count": len(heartbeat_rows),
                    "raw_sensor_events_present": False,
                }
            ]
            derived_events.extend(
                {"event_type": "stage_heartbeat", "payload": row} for row in heartbeat_rows
            )
            derived_events.append(
                {
                    "event_type": "execution_invalid_or_early_finalize",
                    "runtime_valid": False,
                    "summary_status": summary.get("status"),
                    "runtime_status": summary.get("runtime_status"),
                    "finalize_event": finalize,
                    "watchdog_timeout_reason": watchdog.get("timeout_reason"),
                }
            )
            events_path.write_text(
                json.dumps(
                    {
                        "schema": "lift-the-curtain.round4070-derived-execution-events.v1",
                        "trial_id": trial_id,
                        "derived": True,
                        "warning": "No rollout telemetry or deployment-observation samples were fabricated; this ledger records only execution evidence.",
                        "events": derived_events,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            materialized.append(trial_id)
            derived_event_ledger = True

        validation_path = path / "validation_result.json"
        gate = gates.get(str(plan.get("gate_id")), {})
        validation = {
            "schema": "lift-the-curtain.round4070-trial-validation-result.v1",
            "trial_id": trial_id,
            "generated_at": datetime.now().astimezone().isoformat(),
            "runtime_class": "VALID_EPISODE" if runtime_valid else "RUNTIME_INVALID_OR_EXECUTION_INVALID",
            "runtime_valid": runtime_valid,
            "summary_status": summary.get("status"),
            "runtime_status": summary.get("runtime_status"),
            "physics_gate_id": plan.get("gate_id"),
            "physics_gate_status": gate.get("gate_status"),
            "resource_status": {
                "gpu_memory_peak_mb": resources.get("gpu_memory_peak_mb"),
                "process_private_peak_mb": resources.get("process_private_peak_mb"),
                "watchdog_timeout_reason": watchdog.get("timeout_reason"),
            },
            "evidence_presence": {
                "command": (path / "command.json").is_file(),
                "trial_metadata": (path / "trial_metadata.json").is_file(),
                "raw_stdout": (path / "process_stdout.log").is_file(),
                "raw_stderr": (path / "process_stderr.log").is_file(),
                "heartbeat": (path / "stage_heartbeat.jsonl").is_file(),
                "events": (path / "events.json").is_file(),
                "summary": (path / "summary.json").is_file(),
                "gpu_resource_summary": (path / "resource_summary.json").is_file(),
                "telemetry": (path / "telemetry.csv").is_file(),
                "deployment_observation": (path / "deployment_observation.csv").is_file(),
            },
            "invalid_reason": None if runtime_valid else summary.get("error") or (summary.get("finalize_event") or {}).get("details"),
            "derived_event_ledger": derived_event_ledger,
        }
        validation_path.write_text(json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        validation_count += 1

    print(json.dumps({"derived_event_ledgers": materialized, "validation_results": validation_count}, ensure_ascii=False))


if __name__ == "__main__":
    main()
