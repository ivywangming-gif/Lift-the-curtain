"""Materialize D physics gate status used by the G1 launch wrapper."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "outputs" / "round4070_fixed10cm_judgment" / "physics"
CONFIG = REPO / "configs" / "round4070_fixed10cm_judgment"


def read(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def main() -> None:
    gate_ids = {
        "A0_TOP_EDGE_ONLY", "A1_TOP_EDGE_BOTTOM_TWO_CORNERS", "A2_FOUR_CORNERS_ONLY",
        "A3_FULL_PERIMETER_FIXED", "B_D225_NEW", "B_D225_A1", "C_GAP02", "C_GAP05", "C_GAP08",
        "CONTROL_ONLY", "INHERITED_RIGID_CONTROL",
    }
    inherited = {"A0_TOP_EDGE_ONLY", "A1_TOP_EDGE_BOTTOM_TWO_CORNERS", "A2_FOUR_CORNERS_ONLY", "A3_FULL_PERIMETER_FIXED", "B_D225_NEW", "C_GAP05"}
    gates = {}
    for gate in sorted(gate_ids):
        if gate in {"CONTROL_ONLY", "INHERITED_RIGID_CONTROL"}:
            gates[gate] = {"status": "PASS_CONTROL", "source": "control contract"}
            continue
        summary = read(OUTPUT / gate / "probe" / "summary.json") or {}
        no_probe = read(OUTPUT / gate / "no_probe" / "summary.json") or {}
        status = str(summary.get("status", "PENDING"))
        no_status = str(no_probe.get("status", "PENDING"))
        if status in {"PASS_PHYSICS_SANITY", "CONTACT_PASS"} and no_status == "PASS_PHYSICS_SANITY":
            gates[gate] = {"status": "PASS", "source": "D physics-only gate", "probe_status": status, "no_probe_status": no_status}
        elif gate in inherited and not summary:
            gates[gate] = {"status": "PASS_INHERITED_C", "source": "unchanged C physics gate", "probe_status": "not_rerun", "no_probe_status": "not_rerun"}
        else:
            gates[gate] = {"status": "PENDING" if status == "PENDING" else f"FAIL_{status}", "source": "D physics-only gate", "probe_status": status, "no_probe_status": no_status}
    payload = {"schema": "lift-the-curtain.round4070-fixed10cm-physics-gate-status.v1", "gates": gates}
    CONFIG.mkdir(parents=True, exist_ok=True)
    (CONFIG / "PHYSICS_GATE_STATUS.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
