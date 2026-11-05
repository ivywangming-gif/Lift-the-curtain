"""Generate the Round 4070-E fixed-command benchmark manifest."""

from __future__ import annotations

import csv
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "round4070_fixed_probe_benchmark"
RESEARCH = REPO / "research" / "round4070_fixed_probe_benchmark"
TARGET_M = 0.10
JOINT_STEP_RAD = 0.85


P0 = [
    {"condition_id": "FABRIC_NOMINAL", "family": "fabric", "scenario": "nominal", "topology": "A0_TOP_EDGE_ONLY", "density": 1.0, "gate": "A0_TOP_EDGE_ONLY"},
    {"condition_id": "FABRIC_BOTTOM_LEFT_ANCHOR", "family": "fabric", "scenario": "bottom_left_anchor", "topology": "A0_TOP_EDGE_ONLY", "density": 1.0, "gate": "A1_BOTTOM_LEFT_ANCHOR"},
    {"condition_id": "FABRIC_BOTTOM_RIGHT_ANCHOR", "family": "fabric", "scenario": "bottom_right_anchor", "topology": "A0_TOP_EDGE_ONLY", "density": 1.0, "gate": "A1_BOTTOM_RIGHT_ANCHOR"},
    {"condition_id": "FABRIC_BOTH_BOTTOM_CORNERS", "family": "fabric", "scenario": "both_bottom_corners", "topology": "A1_TOP_EDGE_BOTTOM_TWO_CORNERS", "density": 1.0, "gate": "A1_TOP_EDGE_BOTTOM_TWO_CORNERS"},
    {"condition_id": "FABRIC_FOUR_CORNERS_ONLY", "family": "fabric", "scenario": "nominal", "topology": "A2_FOUR_CORNERS_ONLY", "density": 1.0, "gate": "A2_FOUR_CORNERS_ONLY"},
    {"condition_id": "FABRIC_FULL_PERIMETER_FIXED", "family": "fabric", "scenario": "nominal", "topology": "A3_FULL_PERIMETER_FIXED", "density": 1.0, "gate": "A3_FULL_PERIMETER_FIXED"},
    {"condition_id": "FABRIC_HEAVY_FREE", "family": "fabric_heavy", "scenario": "nominal", "topology": "A0_TOP_EDGE_ONLY", "density": 2.25, "gate": "B_D225_NEW"},
    {"condition_id": "FABRIC_HEAVY_CONSTRAINED", "family": "fabric_heavy", "scenario": "nominal", "topology": "A1_TOP_EDGE_BOTTOM_TWO_CORNERS", "density": 2.25, "gate": "B_D225_A1"},
    {"condition_id": "FABRIC_WALL_GAP_02", "family": "hidden_wall", "scenario": "curtain_hidden_rigid", "topology": "A0_TOP_EDGE_ONLY", "density": 1.0, "hidden_x": 0.89, "gap_m": 0.02, "gate": "C_GAP02"},
    {"condition_id": "FABRIC_WALL_GAP_05", "family": "hidden_wall", "scenario": "curtain_hidden_rigid", "topology": "A0_TOP_EDGE_ONLY", "density": 1.0, "hidden_x": 0.92, "gap_m": 0.05, "gate": "C_GAP05"},
    {"condition_id": "FABRIC_WALL_GAP_08", "family": "hidden_wall", "scenario": "curtain_hidden_rigid", "topology": "A0_TOP_EDGE_ONLY", "density": 1.0, "hidden_x": 0.95, "gap_m": 0.08, "gate": "C_GAP08"},
    {"condition_id": "EMPTY_CONTROL", "family": "control", "scenario": "nominal", "topology": "A0_TOP_EDGE_ONLY", "density": 1.0, "gate": "PASS_CONTROL", "diagnostic_no_obstacle": True},
    {"condition_id": "RIGID_CONTROL", "family": "control", "scenario": "rigid_panel", "topology": "A0_TOP_EDGE_ONLY", "density": 1.0, "gate": "PASS_CONTROL"},
]

MATCHED = [
    {"condition_id": "WALL_GAP_02_MATCHED_NO_WALL", "family": "matched_no_wall", "scenario": "nominal", "topology": "A0_TOP_EDGE_ONLY", "density": 1.0, "gate": "MATCHED_CONTROL", "matched_for": "FABRIC_WALL_GAP_02", "diagnostic_no_obstacle": True},
    {"condition_id": "WALL_GAP_05_MATCHED_NO_WALL", "family": "matched_no_wall", "scenario": "nominal", "topology": "A0_TOP_EDGE_ONLY", "density": 1.0, "gate": "MATCHED_CONTROL", "matched_for": "FABRIC_WALL_GAP_05", "diagnostic_no_obstacle": True},
    {"condition_id": "WALL_GAP_08_MATCHED_NO_WALL", "family": "matched_no_wall", "scenario": "nominal", "topology": "A0_TOP_EDGE_ONLY", "density": 1.0, "gate": "MATCHED_CONTROL", "matched_for": "FABRIC_WALL_GAP_08", "diagnostic_no_obstacle": True},
]


def seed_for(condition_index: int, side: str, repeat: int, phase: str) -> int:
    phase_offset = {"PHASE_0": 0, "PHASE_1": 1000, "PHASE_2": 2000, "PHASE_3": 3000, "MATCHED_CONTROL": 4000}[phase]
    side_offset = 0 if side == "left" else 1
    return 407000 + phase_offset + condition_index * 31 + repeat * 7 + side_offset


def make_trial(condition: dict[str, object], condition_index: int, side: str, repeat: int, phase: str, ordinal: int) -> dict[str, object]:
    condition_id = str(condition["condition_id"])
    reset_sign = -1.0 if (ordinal + repeat) % 2 else 1.0
    reset_y = reset_sign * (0.004 + 0.001 * (repeat % 4))
    reset_yaw = (-1.0 if (ordinal + repeat) % 2 else 1.0) * 0.004 * (repeat % 3)
    seed = seed_for(condition_index, side, repeat, phase)
    return {
        "trial_id": f"E_{phase}_{condition_id}_{side[0].upper()}_{repeat + 1:02d}_S{seed}",
        "phase": phase,
        "stage": "discovery" if phase in {"PHASE_0", "PHASE_1", "PHASE_2"} else "frozen_confirmation",
        "condition_id": condition_id,
        "condition_family": condition["family"],
        "scenario": condition["scenario"],
        "probe_side": side,
        "seed": seed,
        "repeat_index": repeat + 1,
        "reset_root_y": round(reset_y, 6),
        "reset_root_yaw": round(reset_yaw, 6),
        "probe_depth_level": "PROBE_10CM",
        "probe_command_mode": "FIXED_ONE_SHOT_NOMINAL_0P10M",
        "probe_10cm_joint_step_rad": JOINT_STEP_RAD,
        "target_depth_m": TARGET_M,
        "nominal_target_displacement_m": TARGET_M,
        "structural_topology": condition["topology"],
        "density_scale_multiplier": condition["density"],
        "hidden_rigid_x": condition.get("hidden_x"),
        "hidden_rigid_gap_m": condition.get("gap_m"),
        "physics_gate_id": condition["gate"],
        "matched_control_for": condition.get("matched_for"),
        "formal_core": condition["family"] not in {"control", "matched_no_wall"},
        "diagnostic_no_obstacle": bool(condition.get("diagnostic_no_obstacle", False)),
        "deployment_label_forbidden": [
            "condition_id", "condition_family", "scenario", "material", "hidden_rigid_x",
            "hidden_rigid_gap_m", "structural_topology", "density_scale_multiplier",
            "simulator_contact_flag", "true_passability", "future_clearance", "task_success",
        ],
    }


def build() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    trials: list[dict[str, object]] = []
    ordinal = 0

    phase0 = ["EMPTY_CONTROL", "RIGID_CONTROL", "FABRIC_NOMINAL"]
    for condition_id in phase0:
        condition_index = next(index for index, item in enumerate(P0) if item["condition_id"] == condition_id)
        condition = P0[condition_index]
        for side in ("left", "right"):
            trials.append(make_trial(condition, condition_index, side, 0, "PHASE_0", ordinal))
            ordinal += 1

    for condition_index, condition in enumerate(P0):
        for repeat in (0, 1):
            for side in ("left", "right"):
                trials.append(make_trial(condition, condition_index, side, repeat, "PHASE_1", ordinal))
                ordinal += 1

    phase2_ids = {
        "FABRIC_NOMINAL", "FABRIC_BOTTOM_LEFT_ANCHOR", "FABRIC_BOTTOM_RIGHT_ANCHOR",
        "FABRIC_BOTH_BOTTOM_CORNERS", "FABRIC_HEAVY_CONSTRAINED",
        "FABRIC_WALL_GAP_02", "FABRIC_WALL_GAP_05", "FABRIC_WALL_GAP_08",
    }
    for condition_index, condition in enumerate(P0):
        if condition["condition_id"] not in phase2_ids:
            continue
        for repeat in (2, 3):
            for side in ("left", "right"):
                trials.append(make_trial(condition, condition_index, side, repeat, "PHASE_2", ordinal))
                ordinal += 1

    phase3_ids = {"FABRIC_NOMINAL", "FABRIC_BOTTOM_LEFT_ANCHOR", "FABRIC_BOTTOM_RIGHT_ANCHOR", "FABRIC_HEAVY_CONSTRAINED", "FABRIC_WALL_GAP_05"}
    for condition_index, condition in enumerate(P0):
        if condition["condition_id"] not in phase3_ids:
            continue
        for side in ("left", "right"):
            trials.append(make_trial(condition, condition_index, side, 4, "PHASE_3", ordinal))
            ordinal += 1

    for condition_index, condition in enumerate(MATCHED, len(P0)):
        for repeat in (0, 1):
            for side in ("left", "right"):
                trials.append(make_trial(condition, condition_index, side, repeat, "MATCHED_CONTROL", ordinal))
                ordinal += 1

    scenes = []
    for condition in [*P0, *MATCHED]:
        scenes.append({
            "condition_id": condition["condition_id"],
            "condition_family": condition["family"],
            "scenario_builder": condition["scenario"],
            "structural_topology": condition["topology"],
            "density_scale_multiplier": condition["density"],
            "hidden_rigid_center_x_m": condition.get("hidden_x"),
            "hidden_rigid_gap_m": condition.get("gap_m"),
            "target_depth_m": TARGET_M,
            "physics_gate_id": condition["gate"],
            "formal_core": condition["family"] not in {"control", "matched_no_wall"},
            "physics_status_at_registration": "PENDING_E_PHYSICS_GATE",
            "online_scene_identity_available": False,
            "matched_control_for": condition.get("matched_for"),
            "notes": "Nominal command is fixed; actual feedback is recorded, not used as a validity gate.",
        })
    return trials, scenes


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    trials, scenes = build()
    manifest = {
        "schema": "lift-the-curtain.round4070-fixed-probe-benchmark-manifest.v1",
        "campaign": "Round 4070-E: Fixed-command proprioceptive traversability judgment benchmark",
        "created_at": "2026-11-06T00:00:00+08:00",
        "probe_contract": {
            "command_mode": "FIXED_ONE_SHOT_NOMINAL_0P10M",
            "nominal_target_displacement_m": TARGET_M,
            "direction_world": [1.0, 0.0, 0.0],
            "continuous_single_probe": True,
            "no_force_push": True,
            "retry_count_per_trial": 0,
            "actual_feedback_authoritative_for_outcome": True,
            "actual_feedback_is_not_probe_validity_gate": True,
        },
        "validity_contract": {
            "runtime_valid": "normal simulator execution, finite state, complete required logs",
            "probe_valid": "legal one-shot command executed with complete proprioceptive history; contact-limited or early safe stop is allowed",
            "label_valid": "independent physical/geometric evidence supports PASSABLE/BLOCKED/UNKNOWN; classifier output is not truth",
            "task_success": "secondary PointGoal/navigation outcome only",
        },
        "comparison_contract": {
            "e0": "q_cmd_minus_q threshold on the same probe history",
            "e1": "frozen deployment_observation.classify_proprioceptive_response",
            "e2": "unified interpretable rule using the same history and preregistered cutoff",
            "same_cutoff_required": True,
            "material_specific_rules": False,
            "truth_in_classifier": False,
        },
        "sample_plan": {
            "p0_condition_count": len(P0),
            "registered_trial_count": len(trials),
            "maximum_new_g1_trials": 200,
            "phase0_trial_count": 6,
            "phase1_requirement": "every P0 condition has at least two independent seeds per side",
            "phase2_requirement": "stable P0 and key conditions receive additional repeats as feasible",
            "phase3_requirement": "independent confirmation after E0/E1/E2 freeze",
        },
        "stopping_rules": [
            "Do not launch formal physics trials before the matching P0 physics gate is PASS.",
            "Actual displacement below nominal 0.10 m is an outcome category, not automatic invalidity or BLOCKED.",
            "Never infer flexible-scene BLOCKED from PointGoal failure or low hand displacement alone.",
            "Stop a branch after two same-cause runtime-invalid attempts and record the branch stop.",
            "Do not overwrite a non-empty incomplete trial directory.",
            "P1 physics-only failures cannot invalidate P0 evidence.",
        ],
        "trials": trials,
    }
    CONFIG.mkdir(parents=True, exist_ok=True)
    RESEARCH.mkdir(parents=True, exist_ok=True)
    (CONFIG / "PREREGISTERED_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_csv(RESEARCH / "SCENE_COVERAGE.csv", scenes)
    write_csv(RESEARCH / "TRIAL_MANIFEST.csv", trials)
    print(json.dumps({"trial_count": len(trials), "scene_count": len(scenes), "max_new_g1_trials": 200}, indent=2))


if __name__ == "__main__":
    main()
