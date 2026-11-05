"""Create the immutable Round 4070-D trial manifest and scene matrix."""

from __future__ import annotations

import csv
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "round4070_fixed10cm_judgment"
RESEARCH = REPO / "research" / "round4070_fixed10cm_judgment"
TARGET_STEP_RAD = 0.85
TARGET_M = 0.10


CORE = [
    ("FABRIC_NOMINAL", "basic_fabric", "nominal", "A0_TOP_EDGE_ONLY", 1.0, None, "A0_TOP_EDGE_ONLY"),
    ("FABRIC_BOTTOM_RIGHT_ANCHOR", "basic_fabric", "bottom_right_anchor", "A0_TOP_EDGE_ONLY", 1.0, None, "A0_TOP_EDGE_ONLY"),
    ("FABRIC_BOTTOM_LEFT_ANCHOR", "basic_fabric", "bottom_left_anchor", "A0_TOP_EDGE_ONLY", 1.0, None, "A0_TOP_EDGE_ONLY"),
    ("FABRIC_FOUR_CORNERS_ONLY", "strong_constraint", "nominal", "A2_FOUR_CORNERS_ONLY", 1.0, None, "A2_FOUR_CORNERS_ONLY"),
    ("FABRIC_FULL_PERIMETER_FIXED", "strong_constraint", "nominal", "A3_FULL_PERIMETER_FIXED", 1.0, None, "A3_FULL_PERIMETER_FIXED_V2"),
    ("FABRIC_WALL_GAP_02", "hidden_wall", "curtain_hidden_rigid", "A0_TOP_EDGE_ONLY", 1.0, 0.89, "C_GAP02"),
    ("FABRIC_WALL_GAP_05", "hidden_wall", "curtain_hidden_rigid", "A0_TOP_EDGE_ONLY", 1.0, 0.92, "C_GAP05"),
    ("FABRIC_WALL_GAP_08", "hidden_wall", "curtain_hidden_rigid", "A0_TOP_EDGE_ONLY", 1.0, 0.95, "C_GAP08"),
    ("FABRIC_HEAVY_FREE", "heavy", "nominal", "A0_TOP_EDGE_ONLY", 2.25, None, "B_D225_NEW"),
    ("FABRIC_HEAVY_CONSTRAINED", "heavy", "nominal", "A1_TOP_EDGE_BOTTOM_TWO_CORNERS", 2.25, None, "B_D225_A1"),
]

CONTROL = [
    ("EMPTY_CONTROL", "control", "nominal", "A0_TOP_EDGE_ONLY", 1.0, None, "CONTROL_ONLY"),
    ("RIGID_CONTROL", "control", "rigid_panel", "A0_TOP_EDGE_ONLY", 1.0, None, "INHERITED_RIGID_CONTROL"),
]

MATCHED = [
    ("WALL_GAP_02_MATCHED_NO_WALL", "matched_no_wall", "nominal", "A0_TOP_EDGE_ONLY", 1.0, None, "A0_TOP_EDGE_ONLY", "FABRIC_WALL_GAP_02"),
    ("WALL_GAP_05_MATCHED_NO_WALL", "matched_no_wall", "nominal", "A0_TOP_EDGE_ONLY", 1.0, None, "A0_TOP_EDGE_ONLY", "FABRIC_WALL_GAP_05"),
    ("WALL_GAP_08_MATCHED_NO_WALL", "matched_no_wall", "nominal", "A0_TOP_EDGE_ONLY", 1.0, None, "A0_TOP_EDGE_ONLY", "FABRIC_WALL_GAP_08"),
]


def make_trial(
    condition: tuple,
    repeat: int,
    side: str,
    ordinal: int,
    *,
    matched_for: str | None = None,
) -> dict[str, object]:
    condition_id, family, scenario, topology, density, hidden_x, gate_id = condition[:7]
    seed = 407000 + ordinal * 17 + (0 if side == "left" else 1)
    signed = -1.0 if ordinal % 2 else 1.0
    reset_y = signed * (0.006 + 0.001 * (repeat % 4))
    reset_yaw = (-1.0 if (ordinal + repeat) % 2 else 1.0) * 0.006 * (repeat % 3)
    return {
        "trial_id": f"D_{condition_id}_{side[0].upper()}_{repeat + 1:02d}_S{seed}",
        "stage": "discovery" if repeat < 6 else "frozen_confirmation",
        "group": family.upper(),
        "condition_id": condition_id,
        "condition_family": family,
        "scenario": scenario,
        "probe_side": side,
        "seed": seed,
        "repeat_index": repeat + 1,
        "reset_root_y": round(reset_y, 6),
        "reset_root_yaw": round(reset_yaw, 6),
        "probe_depth_level": "PROBE_10CM",
        "probe_10cm_joint_step_rad": TARGET_STEP_RAD,
        "target_displacement_m": TARGET_M,
        "structural_topology": topology,
        "density_scale_multiplier": density,
        "hidden_rigid_x": hidden_x,
        "hidden_rigid_gap_m": None if hidden_x is None else round(float(hidden_x) - 0.02 - 0.85, 6),
        "physics_gate_id": gate_id,
        "matched_control_for": matched_for,
        "formal_core": family not in {"control", "matched_no_wall"},
        "deployment_label_forbidden": [
            "condition_id", "scenario", "material", "hidden_rigid_x", "hidden_rigid_gap_m",
            "structural_topology", "density_scale_multiplier", "simulator_contact_flag",
            "true_passability", "future_clearance",
        ],
    }


def build() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    trials: list[dict[str, object]] = []
    scene_rows: list[dict[str, object]] = []
    ordinal = 0
    for condition in CORE:
        condition_id, family, scenario, topology, density, hidden_x, gate_id = condition
        scene_rows.append({
            "condition_id": condition_id,
            "condition_family": family,
            "scenario_builder": scenario,
            "structural_topology": topology,
            "density_scale_multiplier": density,
            "hidden_rigid_center_x_m": hidden_x,
            "hidden_rigid_front_gap_m": None if hidden_x is None else round(float(hidden_x) - 0.02 - 0.85, 6),
            "target_displacement_m": TARGET_M,
            "physics_gate_id": gate_id,
            "formal_core": True,
            "physics_status_at_registration": "PENDING_D_VALIDATION",
            "online_scene_identity_available": False,
            "notes": "One unified PROBE_10CM rule; no condition-specific classifier.",
        })
        for repeat in range(10):
            for side in ("left", "right"):
                trials.append(make_trial(condition, repeat, side, ordinal))
                ordinal += 1
    for condition in CONTROL:
        condition_id, family, scenario, topology, density, hidden_x, gate_id = condition
        scene_rows.append({
            "condition_id": condition_id,
            "condition_family": family,
            "scenario_builder": scenario,
            "structural_topology": topology,
            "density_scale_multiplier": density,
            "hidden_rigid_center_x_m": hidden_x,
            "hidden_rigid_front_gap_m": None,
            "target_displacement_m": TARGET_M,
            "physics_gate_id": gate_id,
            "formal_core": False,
            "physics_status_at_registration": "CONTROL_OR_INHERITED",
            "online_scene_identity_available": False,
            "notes": "Safety/control only; never substitutes for flexible positive examples.",
        })
        for repeat in range(5):
            for side in ("left", "right"):
                trial = make_trial(condition, repeat, side, ordinal)
                trial["diagnostic_no_obstacle"] = condition_id == "EMPTY_CONTROL"
                trials.append(trial)
                ordinal += 1
    for condition in MATCHED:
        condition_id, family, scenario, topology, density, hidden_x, gate_id, matched_for = condition
        scene_rows.append({
            "condition_id": condition_id,
            "condition_family": family,
            "scenario_builder": scenario,
            "structural_topology": topology,
            "density_scale_multiplier": density,
            "hidden_rigid_center_x_m": hidden_x,
            "hidden_rigid_front_gap_m": None,
            "target_displacement_m": TARGET_M,
            "physics_gate_id": gate_id,
            "formal_core": False,
            "physics_status_at_registration": "MATCHED_NO_WALL_CONTROL",
            "online_scene_identity_available": False,
            "notes": f"Matched no-wall control for {matched_for}; paired by seed and side.",
        })
        for repeat in range(5):
            for side in ("left", "right"):
                trial = make_trial(condition, repeat, side, ordinal, matched_for=matched_for)
                trials.append(trial)
                ordinal += 1
    return trials, scene_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    trials, scene_rows = build()
    manifest = {
        "schema": "lift-the-curtain.round4070-fixed10cm-preregistered-manifest.v1",
        "campaign": "Round 4070-D: Fixed 10 cm G1 proprioceptive passability judgment",
        "created_at": "2026-11-05T00:00:00+08:00",
        "probe_contract": {
            "mode": "PROBE_10CM",
            "target_displacement_m": TARGET_M,
            "direction_world": [1.0, 0.0, 0.0],
            "continuous_single_probe": True,
            "no_force_push": True,
            "no_retry": True,
            "joint_step_rad": TARGET_STEP_RAD,
            "actual_fk_is_authoritative": True,
        },
        "sample_plan": {
            "core_conditions": 10,
            "core_valid_target": 200,
            "empty_and_rigid_controls": 20,
            "matched_no_wall_controls": 30,
            "registered_trial_count": len(trials),
            "recommended_launch_budget": 260,
            "discovery_repeats_per_core_side": 6,
            "confirmation_repeats_per_core_side": 4,
        },
        "method_contract": {
            "e0": "global q_cmd_minus_q threshold",
            "e1": "frozen deployment_observation.classify_proprioceptive_response",
            "e2": "single interpretable rule over the same deployment history and empty residual baseline",
            "confirmation_is_frozen": True,
            "material_specific_rules": False,
            "truth_in_classifier": False,
        },
        "stopping_rules": [
            "Never launch a non-control condition before its physics gate is PASS.",
            "A safety abort or incomplete 10 cm movement is UNKNOWN/invalid, never an automatic BLOCKED label.",
            "Do not overwrite an existing trial directory or summary.",
            "Stop a condition after two same-cause runtime-invalid attempts.",
            "No PVC or multi-panel condition enters the benchmark without a separate physics gate.",
        ],
        "trials": trials,
    }
    CONFIG.mkdir(parents=True, exist_ok=True)
    (CONFIG / "PREREGISTERED_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(RESEARCH / "SCENE_MATRIX.csv", scene_rows)
    write_csv(RESEARCH / "TRIAL_MANIFEST.csv", trials)
    print(json.dumps({"trial_count": len(trials), "scene_count": len(scene_rows)}, indent=2))


if __name__ == "__main__":
    main()
