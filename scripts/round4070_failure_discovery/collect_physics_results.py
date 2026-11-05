"""Collect the frozen Round 4070-B physics-gate outputs without changing them."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO / "configs" / "round4070_failure_discovery" / "PREREGISTERED_MANIFEST.json"
DEFAULT_PHYSICS_ROOT = REPO / "outputs" / "round4070_failure_discovery" / "physics"
DEFAULT_OUTPUT = REPO / "research" / "round4070_failure_discovery" / "PHYSICS_GATE_RESULTS.csv"


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def acceptance_failures(summary: dict[str, Any] | None) -> list[str]:
    if not summary:
        return []
    acceptance = summary.get("acceptance")
    if not isinstance(acceptance, dict):
        return []
    return [str(key) for key, value in acceptance.items() if value is False]


def metric(summary: dict[str, Any] | None, key: str) -> Any:
    if not summary:
        return None
    return summary.get(key)


def collect_gate(gate_id: str, physics_root: Path) -> dict[str, Any]:
    modes: dict[str, dict[str, Any] | None] = {}
    watchdogs: dict[str, dict[str, Any] | None] = {}
    resources: dict[str, dict[str, Any] | None] = {}
    for mode in ("no_probe", "probe"):
        run_root = physics_root / gate_id / mode
        modes[mode] = read_json(run_root / "summary.json")
        watchdogs[mode] = read_json(run_root / "watchdog_summary.json")
        resources[mode] = read_json(run_root / "resource_summary.json")

    no_probe = modes["no_probe"]
    probe = modes["probe"]
    no_probe_status = str(no_probe.get("status")) if no_probe else "MISSING"
    probe_status = str(probe.get("status")) if probe else "MISSING"
    no_probe_physics = str(no_probe.get("physics_status")) if no_probe else "MISSING"
    probe_physics = str(probe.get("physics_status")) if probe else "MISSING"
    no_probe_runtime = str(no_probe.get("runtime_status")) if no_probe else "MISSING"
    probe_runtime = str(probe.get("runtime_status")) if probe else "MISSING"
    probe_contact = str(probe.get("contact_status")) if probe else "MISSING"

    if not no_probe:
        gate_status = "RUNTIME_INVALID_NO_PROBE"
    elif no_probe_status != "PASS_PHYSICS_SANITY":
        gate_status = "PHYSICS_GATE_FAIL_NO_PROBE"
    elif not probe:
        gate_status = "RUNTIME_INVALID_PROBE_MISSING"
    elif probe_status != "PASS_PHYSICS_SANITY" or probe_contact != "CONTACT_PASS":
        gate_status = "PHYSICS_GATE_FAIL_PROBE"
    else:
        gate_status = "PASS_PHYSICS_GATE"

    def resource_value(mode: str, key: str) -> Any:
        item = resources[mode]
        return item.get(key) if item else None

    return {
        "gate_id": gate_id,
        "gate_status": gate_status,
        "no_probe_status": no_probe_status,
        "no_probe_physics_status": no_probe_physics,
        "no_probe_runtime_status": no_probe_runtime,
        "no_probe_acceptance_failures": ";".join(acceptance_failures(no_probe)),
        "probe_status": probe_status,
        "probe_physics_status": probe_physics,
        "probe_runtime_status": probe_runtime,
        "probe_contact_status": probe_contact,
        "probe_acceptance_failures": ";".join(acceptance_failures(probe)),
        "no_probe_max_speed_mps": metric(no_probe, "max_speed_mps"),
        "no_probe_max_displacement_m": metric(no_probe, "max_displacement_m"),
        "no_probe_max_neighbor_stretch_ratio": metric(no_probe, "max_neighbor_stretch_ratio"),
        "no_probe_min_bottom_z_m": metric(no_probe, "minimum_bottom_z_m"),
        "probe_max_speed_mps": metric(probe, "max_speed_mps"),
        "probe_max_displacement_m": metric(probe, "max_displacement_m"),
        "probe_max_neighbor_stretch_ratio": metric(probe, "max_neighbor_stretch_ratio"),
        "probe_min_bottom_z_m": metric(probe, "minimum_bottom_z_m"),
        "probe_min_hidden_wall_face_gap_m": metric(probe, "min_hidden_wall_face_gap_m"),
        "probe_max_nodes_at_or_beyond_hidden_wall_front": metric(
            probe, "max_nodes_at_or_beyond_hidden_wall_front"
        ),
        "no_probe_gpu_peak_mb": resource_value("no_probe", "gpu_memory_peak_mb"),
        "probe_gpu_peak_mb": resource_value("probe", "gpu_memory_peak_mb"),
        "no_probe_process_private_peak_mb": resource_value("no_probe", "process_private_peak_mb"),
        "probe_process_private_peak_mb": resource_value("probe", "process_private_peak_mb"),
        "no_probe_watchdog_timeout": (watchdogs["no_probe"] or {}).get("timeout_reason"),
        "probe_watchdog_timeout": (watchdogs["probe"] or {}).get("timeout_reason"),
        "no_probe_summary_path": str(physics_root / gate_id / "no_probe" / "summary.json"),
        "probe_summary_path": str(physics_root / gate_id / "probe" / "summary.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--physics-root", type=Path, default=DEFAULT_PHYSICS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    gate_ids = list(dict.fromkeys(str(item["gate_id"]) for item in manifest["trials"]))
    rows = [collect_gate(gate_id, args.physics_root) for gate_id in gate_ids]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["gate_id", "gate_status"]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_output = args.output.with_suffix(".json")
    json_output.write_text(
        json.dumps(
            {
                "schema": "lift-the-curtain.round4070-failure-discovery-physics-results.v1",
                "manifest": str(args.manifest),
                "physics_root": str(args.physics_root),
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "gates": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
