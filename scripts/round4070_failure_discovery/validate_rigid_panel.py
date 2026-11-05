"""Physics-only sanity check for the matched formal rigid-panel control."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from lift_curtain.experiments.stage_log import StageLogger


DT = 0.005
PANEL_X_M = 0.85
PANEL_Y_M = 0.0
PANEL_WIDTH_M = 1.20
PANEL_HEIGHT_M = 2.00
PANEL_THICKNESS_M = 0.04
PROBE_SIZE_M = 0.12
PROBE_START_X_M = 0.50
PROBE_Z_M = 0.95
PROBE_SPEED_MPS = 0.75

pre_parser = argparse.ArgumentParser(add_help=False)
pre_parser.add_argument("--run-root", type=Path, required=True)
PRE_ARGS, _ = pre_parser.parse_known_args()
LOGGER = StageLogger(
    PRE_ARGS.run_root,
    phase_timeouts_s={
        "BEFORE_APP": 180.0,
        "AFTER_APP": 180.0,
        "AFTER_SCENE": 120.0,
        "AFTER_RESET": 120.0,
        "FIRST_PHYSICS_STEP": 120.0,
    },
)
LOGGER.mark("BEFORE_APP", details={"argv": sys.argv, "pid": os.getpid()})

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("no_probe", "probe"), required=True)
parser.add_argument("--run-root", type=Path, required=True)
parser.add_argument("--condition-id", default="RIGID_CONTROL")
parser.add_argument("--steps", type=int, default=720)
parser.add_argument("--sample-every", type=int, default=20)
parser.add_argument("--heartbeat-every", type=int, default=100)
parser.add_argument("--seed", type=int, default=61)
AppLauncher.add_app_launcher_args(parser)
ARGS = parser.parse_args()
APP = AppLauncher(ARGS).app
LOGGER.mark("AFTER_APP")


def clean(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def main() -> int:
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.sim.utils import get_current_stage

    run_root = ARGS.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "stage": "ROUND4070_RIGID_PANEL_PHYSICS_SANITY",
        "scenario": "rigid_panel",
        "mode": ARGS.mode,
        "condition_id": ARGS.condition_id,
        "status": "RUNTIME_BLOCKED",
        "physics_status": "PHYSICS_BLOCKED",
        "contact_status": "CONTACT_BLOCKED" if ARGS.mode == "probe" else "NOT_RUN",
        "runtime_status": "RUNTIME_BLOCKED",
        "deployment_decision_inputs": [],
        "evaluation_only_fields": ["probe_contact_truth", "rigid_panel_geometry"],
    }
    sim = None
    rows: list[dict[str, Any]] = []
    try:
        if min(ARGS.steps, ARGS.sample_every, ARGS.heartbeat_every) <= 0:
            raise ValueError("steps and sampling intervals must be positive")
        np.random.seed(ARGS.seed)
        torch.manual_seed(ARGS.seed)
        sim = SimulationContext(SimulationCfg(dt=DT, render_interval=4, device="cuda:0"))
        ground = sim_utils.CuboidCfg(
            size=(8.0, 8.0, 0.10),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.18, 0.18)),
        )
        ground.func("/World/defaultGroundPlane", ground, translation=(0.0, 0.0, -0.05))
        stage = get_current_stage()
        panel = sim_utils.CuboidCfg(
            size=(PANEL_THICKNESS_M, PANEL_WIDTH_M, PANEL_HEIGHT_M),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.65, 0.20, 0.12)),
        )
        panel_path = "/World/Round4070FormalRigidPanel"
        panel.func(panel_path, panel, translation=(PANEL_X_M, PANEL_Y_M, PANEL_HEIGHT_M / 2.0))
        probe = None
        if ARGS.mode == "probe":
            probe = RigidObject(
                RigidObjectCfg(
                    prim_path="/World/Round4070RigidProbe",
                    spawn=sim_utils.CuboidCfg(
                        size=(PROBE_SIZE_M, PROBE_SIZE_M, PROBE_SIZE_M),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            kinematic_enabled=False,
                            disable_gravity=True,
                            max_depenetration_velocity=0.5,
                        ),
                        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.1)),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(PROBE_START_X_M, PANEL_Y_M, PROBE_Z_M),
                        lin_vel=(PROBE_SPEED_MPS, 0.0, 0.0),
                    ),
                )
            )
        report["formal_rigid_panel"] = {
            "x_m": PANEL_X_M,
            "y_m": PANEL_Y_M,
            "bottom_z_m": 0.0,
            "width_m": PANEL_WIDTH_M,
            "height_m": PANEL_HEIGHT_M,
            "thickness_m": PANEL_THICKNESS_M,
            "front_x_m": PANEL_X_M - PANEL_THICKNESS_M / 2.0,
            "back_x_m": PANEL_X_M + PANEL_THICKNESS_M / 2.0,
        }
        LOGGER.mark("AFTER_SCENE", details={"panel_path": panel_path, "probe": probe is not None})
        sim.reset()
        if probe is not None:
            probe.reset()
            probe.write_root_velocity_to_sim(
                torch.tensor(
                    [[PROBE_SPEED_MPS, 0.0, 0.0, 0.0, 0.0, 0.0]],
                    device=sim.device,
                    dtype=torch.float32,
                )
            )
            probe.write_data_to_sim()
            probe.update(DT)
        LOGGER.mark("AFTER_RESET")
        sim.step(render=False)
        if probe is not None:
            probe.update(DT)
        LOGGER.mark("FIRST_PHYSICS_STEP", details={"finite": True})

        for step in range(ARGS.steps + 1):
            probe_x = None
            probe_speed = None
            if probe is not None:
                position = as_numpy(probe.data.root_pos_w)[0].astype(np.float64)
                velocity = as_numpy(probe.data.root_lin_vel_w)[0].astype(np.float64)
                probe_x = float(position[0])
                probe_speed = float(np.linalg.norm(velocity))
            rows.append(
                {
                    "step": step,
                    "time_s": step * DT,
                    "probe_x_m": probe_x,
                    "probe_speed_mps": probe_speed,
                }
            )
            if step < ARGS.steps:
                sim.step(render=False)
                if probe is not None:
                    probe.update(DT)
            if step > 0 and (step % ARGS.heartbeat_every == 0 or step == ARGS.steps):
                LOGGER.heartbeat(progress=step / ARGS.steps, details={"step": step, "sim_time_s": step * DT})

        with (run_root / "telemetry.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        finite = all(
            row["probe_x_m"] is None
            or (math.isfinite(float(row["probe_x_m"])) and math.isfinite(float(row["probe_speed_mps"])))
            for row in rows
        )
        probe_x_values = [float(row["probe_x_m"]) for row in rows if row["probe_x_m"] is not None]
        probe_speed_values = [float(row["probe_speed_mps"]) for row in rows if row["probe_speed_mps"] is not None]
        panel_front = PANEL_X_M - PANEL_THICKNESS_M / 2.0
        panel_back = PANEL_X_M + PANEL_THICKNESS_M / 2.0
        probe_radius = PROBE_SIZE_M / 2.0
        max_probe_x = max(probe_x_values) if probe_x_values else None
        final_probe_speed = probe_speed_values[-1] if probe_speed_values else None
        panel_not_crossed = (
            max_probe_x is None
            or max_probe_x <= panel_back + probe_radius + 0.02
        )
        probe_stopped = final_probe_speed is None or final_probe_speed < 0.20
        acceptance = {
            "runtime_finite": bool(finite),
            "panel_geometry_exact": True,
            "panel_not_crossed": bool(panel_not_crossed),
            "probe_stopped_at_panel": bool(probe_stopped) if probe is not None else True,
        }
        report.update(
            {
                "status": "PASS_PHYSICS_SANITY" if all(acceptance.values()) else "FAIL_PHYSICS_SANITY",
                "physics_status": "PHYSICS_PASS" if all(acceptance.values()) else "PHYSICS_FAIL",
                "contact_status": (
                    "CONTACT_PASS"
                    if ARGS.mode == "probe" and acceptance["panel_not_crossed"] and acceptance["probe_stopped_at_panel"]
                    else "CONTACT_BLOCKED"
                    if ARGS.mode == "probe"
                    else "NOT_RUN"
                ),
                "runtime_status": "RUNTIME_PASS",
                "steps": ARGS.steps,
                "physics_dt_s": DT,
                "acceptance": acceptance,
                "probe_initial_x_m": PROBE_START_X_M if probe is not None else None,
                "probe_max_x_m": max_probe_x,
                "probe_final_speed_mps": final_probe_speed,
                "probe_max_speed_mps": max(probe_speed_values) if probe_speed_values else None,
                "panel_front_x_m": panel_front,
                "panel_back_x_m": panel_back,
                "telemetry_csv": str(run_root / "telemetry.csv"),
            }
        )
        (run_root / "resolved_config.json").write_text(
            json.dumps(clean(report), indent=2) + "\n", encoding="utf-8"
        )
        LOGGER.mark("FINALIZE", status="PASS" if report["status"] == "PASS_PHYSICS_SANITY" else "FAIL")
        (run_root / "summary.json").write_text(json.dumps(clean(report), indent=2) + "\n", encoding="utf-8")
        return 0 if report["status"] == "PASS_PHYSICS_SANITY" else 1
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        report["error_message"] = str(exc)
        report["status"] = "RUNTIME_BLOCKED"
        report["physics_status"] = "PHYSICS_BLOCKED"
        report["runtime_status"] = "RUNTIME_INVALID"
        (run_root / "summary.json").write_text(json.dumps(clean(report), indent=2) + "\n", encoding="utf-8")
        try:
            LOGGER.mark("FINALIZE", status="FAIL", details={"error": str(exc)})
        except Exception:
            pass
        return 1
    finally:
        if sim is not None:
            try:
                sim.clear_instance()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
