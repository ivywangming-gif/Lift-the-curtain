"""Materialize Round 4070-E P0 physics-gate status from recorded summaries."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "outputs" / "round4070_fixed_probe_benchmark" / "physics"
CONFIG = REPO / "configs" / "round4070_fixed_probe_benchmark"

GATES = {
    "A0_TOP_EDGE_ONLY": "FABRIC_NOMINAL",
    "A1_BOTTOM_LEFT_ANCHOR": "FABRIC_BOTTOM_LEFT_ANCHOR",
    "A1_BOTTOM_RIGHT_ANCHOR": "FABRIC_BOTTOM_RIGHT_ANCHOR",
    "A1_TOP_EDGE_BOTTOM_TWO_CORNERS": "FABRIC_BOTH_BOTTOM_CORNERS",
    "A2_FOUR_CORNERS_ONLY": "FABRIC_FOUR_CORNERS_ONLY",
    "A3_FULL_PERIMETER_FIXED": "FABRIC_FULL_PERIMETER_FIXED",
    "B_D225_NEW": "FABRIC_HEAVY_FREE",
    "B_D225_A1": "FABRIC_HEAVY_CONSTRAINED",
    "C_GAP02": "FABRIC_WALL_GAP_02",
    "C_GAP05": "FABRIC_WALL_GAP_05",
    "C_GAP08": "FABRIC_WALL_GAP_08",
}


def read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {}
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}


def main() -> None:
    gates: dict[str, Any] = {}
    for gate_id, condition_id in GATES.items():
        root = OUTPUT / gate_id
        no_probe_path = root / "no_probe" / "summary.json"
        probe_path = root / "probe" / "summary.json"
        no_probe = read(no_probe_path)
        probe = read(probe_path)
        no_status = str(no_probe.get("status", "MISSING"))
        probe_status = str(probe.get("status", "MISSING"))
        passed = no_status == "PASS_PHYSICS_SANITY" and probe_status in {"PASS_PHYSICS_SANITY", "CONTACT_PASS"}
        gates[gate_id] = {
            "status": "PASS_E_PHYSICS" if passed else "BLOCKED",
            "condition_id": condition_id,
            "no_probe_status": no_status,
            "probe_status": probe_status,
            "evidence_root": str(root.resolve()),
            "model_contract": "SINGLE_SURFACE_PHYSICS_GATE",
        }
    gates["PASS_CONTROL"] = {
        "status": "PASS_CONTROL",
        "condition_id": "EMPTY_CONTROL_OR_RIGID_CONTROL",
        "evidence_root": "G1_PHASE_0",
        "model_contract": "G1_RUNTIME_CONTROL",
    }
    payload = {
        "schema": "lift-the-curtain.round4070-fixed-probe-physics-gate-status.v1",
        "campaign": "Round 4070-E",
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "gates": gates,
        "p0_claim_scope": "Only P0 single-sheet scenes are covered here; P1 multi-sheet attempts are reported separately.",
    }
    CONFIG.mkdir(parents=True, exist_ok=True)
    (CONFIG / "PHYSICS_GATE_STATUS.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if all(value["status"] == "PASS_E_PHYSICS" for key, value in gates.items() if key != "PASS_CONTROL") else 2)


if __name__ == "__main__":
    main()
