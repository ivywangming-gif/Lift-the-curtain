"""Freeze the Round 4070-B scene/g1 manifest before any new launch."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
import json
import subprocess
from zoneinfo import ZoneInfo


REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "configs" / "round4070_failure_discovery"
RESEARCH_DIR = REPO / "research" / "round4070_failure_discovery"


def git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False
    )
    return result.stdout.strip()


def trial(
    trial_id: str,
    stage: str,
    group: str,
    condition_id: str,
    scenario: str,
    probe_side: str,
    seed: int,
    gate_id: str,
    **overrides: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "trial_id": trial_id,
        "stage": stage,
        "group": group,
        "condition_id": condition_id,
        "scenario": scenario,
        "policy": "r1",
        "probe_side": probe_side,
        "seed": seed,
        "reset_root_y": 0.0,
        "reset_root_yaw": 0.0,
        "gate_id": gate_id,
        "physics_required": True,
    }
    row.update(overrides)
    return row


def build_trials() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    # Existing-scene bilateral/runtime controls.
    rows += [
        trial("FD_B0_FABRIC_L_S61", "baseline", "BASELINE", "FABRIC_ONLY", "nominal", "left", 61, "FABRIC_ONLY"),
        trial("FD_B0_FABRIC_R_S61", "baseline", "BASELINE", "FABRIC_ONLY", "nominal", "right", 61, "FABRIC_ONLY"),
        trial("FD_B0_RIGID_L_S61", "baseline", "BASELINE", "RIGID_CONTROL", "rigid_panel", "left", 61, "RIGID_CONTROL"),
        trial("FD_B0_RIGID_R_S61", "baseline", "BASELINE", "RIGID_CONTROL", "rigid_panel", "right", 61, "RIGID_CONTROL"),
    ]

    # S1 hidden backing: same front curtain, only backing placement/coverage changes.
    rows += [
        trial("FD_S1A_FABRIC_L_S62", "discovery", "S1_HIDDEN_RIGID", "S1A_FABRIC_ONLY", "nominal", "left", 62, "FABRIC_ONLY"),
        trial("FD_S1A_FABRIC_R_S62", "discovery", "S1_HIDDEN_RIGID", "S1A_FABRIC_ONLY", "nominal", "right", 62, "FABRIC_ONLY"),
        trial("FD_S1B_NEAR_L_S62", "discovery", "S1_HIDDEN_RIGID", "S1B_NEAR", "curtain_hidden_rigid", "left", 62, "S1B_NEAR", hidden_rigid_x=0.916),
        trial("FD_S1B_NEAR_R_S62", "discovery", "S1_HIDDEN_RIGID", "S1B_NEAR", "curtain_hidden_rigid", "right", 62, "S1B_NEAR", hidden_rigid_x=0.916),
        trial("FD_S1B_NEAR_L_S63", "discovery", "S1_HIDDEN_RIGID", "S1B_NEAR", "curtain_hidden_rigid", "left", 63, "S1B_NEAR", hidden_rigid_x=0.916),
        trial("FD_S1B_NEAR_R_S63", "discovery", "S1_HIDDEN_RIGID", "S1B_NEAR", "curtain_hidden_rigid", "right", 63, "S1B_NEAR", hidden_rigid_x=0.916),
        trial("FD_S1C_FAR_L_S62", "discovery", "S1_HIDDEN_RIGID", "S1C_FAR", "curtain_hidden_rigid", "left", 62, "S1C_FAR", hidden_rigid_x=1.240),
        trial("FD_S1C_FAR_R_S62", "discovery", "S1_HIDDEN_RIGID", "S1C_FAR", "curtain_hidden_rigid", "right", 62, "S1C_FAR", hidden_rigid_x=1.240),
        trial("FD_S1C_FAR_L_S63", "discovery", "S1_HIDDEN_RIGID", "S1C_FAR", "curtain_hidden_rigid", "left", 63, "S1C_FAR", hidden_rigid_x=1.240),
        trial("FD_S1C_FAR_R_S63", "discovery", "S1_HIDDEN_RIGID", "S1C_FAR", "curtain_hidden_rigid", "right", 63, "S1C_FAR", hidden_rigid_x=1.240),
        trial("FD_S1D_PARTIAL_L_S62", "discovery", "S1_HIDDEN_RIGID", "S1D_PARTIAL", "curtain_hidden_rigid", "left", 62, "S1D_PARTIAL", hidden_rigid_x=0.916, hidden_rigid_y=0.420, hidden_rigid_width=0.500),
        trial("FD_S1D_PARTIAL_R_S62", "discovery", "S1_HIDDEN_RIGID", "S1D_PARTIAL", "curtain_hidden_rigid", "right", 62, "S1D_PARTIAL", hidden_rigid_x=0.916, hidden_rigid_y=0.420, hidden_rigid_width=0.500),
    ]

    # S2 heavy/high-resistance single curtains. Each is a candidate only after a gate pass.
    rows += [
        trial("FD_S2_D125_L_S64", "discovery", "S2_HEAVY_CURTAIN", "S2_D125", "nominal", "left", 64, "S2_D125", density_scale_multiplier=1.25),
        trial("FD_S2_D125_R_S64", "discovery", "S2_HEAVY_CURTAIN", "S2_D125", "nominal", "right", 64, "S2_D125", density_scale_multiplier=1.25),
        trial("FD_S2_D150_L_S64", "discovery", "S2_HEAVY_CURTAIN", "S2_D150", "nominal", "left", 64, "S2_D150", density_scale_multiplier=1.50),
        trial("FD_S2_D150_R_S64", "discovery", "S2_HEAVY_CURTAIN", "S2_D150", "nominal", "right", 64, "S2_D150", density_scale_multiplier=1.50),
        trial("FD_S2_D200_L_S64", "discovery", "S2_HEAVY_CURTAIN", "S2_D200", "nominal", "left", 64, "S2_D200", density_scale_multiplier=2.00),
        trial("FD_S2_D200_R_S64", "discovery", "S2_HEAVY_CURTAIN", "S2_D200", "nominal", "right", 64, "S2_D200", density_scale_multiplier=2.00),
        trial("FD_S2_D150_L_S65", "discovery", "S2_HEAVY_CURTAIN", "S2_D150", "nominal", "left", 65, "S2_D150", density_scale_multiplier=1.50),
        trial("FD_S2_D150_R_S65", "discovery", "S2_HEAVY_CURTAIN", "S2_D150", "nominal", "right", 65, "S2_D150", density_scale_multiplier=1.50),
        trial("FD_S2_DAMP_L_S64", "discovery", "S2_HEAVY_CURTAIN", "S2_DAMP", "nominal", "left", 64, "S2_DAMP", curtain_damping=0.20),
        trial("FD_S2_DAMP_R_S64", "discovery", "S2_HEAVY_CURTAIN", "S2_DAMP", "nominal", "right", 64, "S2_DAMP", curtain_damping=0.20),
        trial("FD_S2_BEND_L_S64", "discovery", "S2_HEAVY_CURTAIN", "S2_BEND", "nominal", "left", 64, "S2_BEND", curtain_bend_stiffness=0.08),
        trial("FD_S2_BEND_R_S64", "discovery", "S2_HEAVY_CURTAIN", "S2_BEND", "nominal", "right", 64, "S2_BEND", curtain_bend_stiffness=0.08),
    ]

    # S3 bilateral anchor controls and S4 length controls.
    rows += [
        trial("FD_S3_BL_L_S65", "discovery", "S3_S4", "S3_BOTTOM_LEFT", "bottom_left_anchor", "left", 65, "S3_BOTTOM_LEFT"),
        trial("FD_S3_BL_R_S65", "discovery", "S3_S4", "S3_BOTTOM_LEFT", "bottom_left_anchor", "right", 65, "S3_BOTTOM_LEFT"),
        trial("FD_S3_BR_L_S65", "discovery", "S3_S4", "S3_BOTTOM_RIGHT", "bottom_right_anchor", "left", 65, "S3_BOTTOM_RIGHT"),
        trial("FD_S3_BR_R_S65", "discovery", "S3_S4", "S3_BOTTOM_RIGHT", "bottom_right_anchor", "right", 65, "S3_BOTTOM_RIGHT"),
        trial("FD_S4_LEN180_L_S66", "discovery", "S3_S4", "S4_LEN180", "nominal", "left", 66, "S4_LEN180", curtain_length=1.80, curtain_top_z=1.82),
        trial("FD_S4_LEN180_R_S66", "discovery", "S3_S4", "S4_LEN180", "nominal", "right", 66, "S4_LEN180", curtain_length=1.80, curtain_top_z=1.82),
        trial("FD_S4_LEN160_L_S66", "discovery", "S3_S4", "S4_LEN160", "nominal", "left", 66, "S4_LEN160", curtain_length=1.60, curtain_top_z=1.62),
        trial("FD_S4_LEN160_R_S66", "discovery", "S3_S4", "S4_LEN160", "nominal", "right", 66, "S4_LEN160", curtain_length=1.60, curtain_top_z=1.62),
    ]

    # Independent confirmation slots are frozen structurally; a condition may be ruled out by its gate.
    rows += [
        trial("FD_C_S1_NEAR_L_S67", "confirmation", "CONFIRMATION", "S1B_NEAR", "curtain_hidden_rigid", "left", 67, "S1B_NEAR", hidden_rigid_x=0.916),
        trial("FD_C_S1_NEAR_R_S67", "confirmation", "CONFIRMATION", "S1B_NEAR", "curtain_hidden_rigid", "right", 67, "S1B_NEAR", hidden_rigid_x=0.916),
        trial("FD_C_S1_FAR_L_S67", "confirmation", "CONFIRMATION", "S1C_FAR", "curtain_hidden_rigid", "left", 67, "S1C_FAR", hidden_rigid_x=1.240),
        trial("FD_C_S1_FAR_R_S67", "confirmation", "CONFIRMATION", "S1C_FAR", "curtain_hidden_rigid", "right", 67, "S1C_FAR", hidden_rigid_x=1.240),
        trial("FD_C_S2_D150_L_S67", "confirmation", "CONFIRMATION", "S2_D150", "nominal", "left", 67, "S2_D150", density_scale_multiplier=1.50),
        trial("FD_C_S2_D150_R_S67", "confirmation", "CONFIRMATION", "S2_D150", "nominal", "right", 67, "S2_D150", density_scale_multiplier=1.50),
        trial("FD_C_S4_LEN180_L_S67", "confirmation", "CONFIRMATION", "S4_LEN180", "nominal", "left", 67, "S4_LEN180", curtain_length=1.80, curtain_top_z=1.82),
        trial("FD_C_S4_LEN180_R_S67", "confirmation", "CONFIRMATION", "S4_LEN180", "nominal", "right", 67, "S4_LEN180", curtain_length=1.80, curtain_top_z=1.82),
    ]

    # Four explicitly reserved slots; they are never launched merely to fill the budget.
    rows += [
        trial("FD_R_S1_NEAR_L_S68", "reserved", "RESERVED", "S1B_NEAR", "curtain_hidden_rigid", "left", 68, "S1B_NEAR", hidden_rigid_x=0.916),
        trial("FD_R_S1_NEAR_R_S68", "reserved", "RESERVED", "S1B_NEAR", "curtain_hidden_rigid", "right", 68, "S1B_NEAR", hidden_rigid_x=0.916),
        trial("FD_R_S2_D150_L_S68", "reserved", "RESERVED", "S2_D150", "nominal", "left", 68, "S2_D150", density_scale_multiplier=1.50),
        trial("FD_R_S2_D150_R_S68", "reserved", "RESERVED", "S2_D150", "nominal", "right", 68, "S2_D150", density_scale_multiplier=1.50),
    ]
    assert len(rows) == 48, len(rows)
    return rows


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    trials = build_trials()
    frozen_at = datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat()
    gate_thresholds = {
        "initial_velocity_max_mps": 0.001,
        "top_drift_max_m": 0.010,
        "anchor_drift_max_m": 0.010,
        "max_speed_mps_exclusive": 5.0,
        "max_displacement_m_exclusive": 1.5,
        "max_neighbor_stretch_ratio_exclusive": 4.0,
        "ground_penetration_tolerance_m": -0.005,
        "g1_gpu_memory_peak_max_mb": 6000.0,
        "g1_process_private_peak_max_mb": 12000.0,
    }
    manifest = {
        "schema": "lift-the-curtain.round4070-failure-discovery-preregistered-manifest.v1",
        "campaign": "Round 4070-B: G1 Proprioceptive Traversability Failure Discovery",
        "frozen_at_start": frozen_at,
        "git_head_at_freeze": git_head(),
        "new_g1_trial_hard_limit": 48,
        "trial_count_frozen": len(trials),
        "runner": "scripts/round4070_failure_discovery/run_episode.py",
        "physics_runner": "scripts/round4070_failure_discovery/validate_single_panel.py",
        "threshold_file": "configs/proprioceptive_thresholds.json",
        "frozen_falcon": True,
        "headless": True,
        "num_envs": 1,
        "serial_execution_required": True,
        "physics_gate_thresholds": gate_thresholds,
        "forbidden_deployment_inputs": [
            "obstacle identity", "material parameters", "hidden rigid truth", "deformable mesh nodes",
            "future clearance", "simulator contact ground truth",
        ],
        "label_contract": {
            "local_response": ["RESPONSIVE", "BLOCKED", "UNCERTAIN"],
            "upper_decision": ["PASSABLE_FOR_ACTION", "BLOCKED_FOR_ACTION", "UNKNOWN"],
            "runtime_invalid_is_not_a_physical_negative": True,
        },
        "stopping_rules": [
            "No G1 launch before the matching physics gate passes.",
            "Do not launch more than 48 new G1 trial directories.",
            "Stop a condition after two same-cause runtime-invalid launches.",
            "Do not alter frozen FALCON, detector thresholds, probe timing, or safety limits.",
            "Do not use a failed single action to claim all actions are blocked.",
        ],
        "trials": trials,
    }
    (CONFIG_DIR / "PREREGISTERED_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (RESEARCH_DIR / "PREREGISTERED_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (RESEARCH_DIR / "TRIAL_MANIFEST.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = sorted({key for row in trials for key in row})
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(trials)
    print(json.dumps({"trial_count": len(trials), "frozen_at": frozen_at, "git_head": manifest["git_head_at_freeze"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
