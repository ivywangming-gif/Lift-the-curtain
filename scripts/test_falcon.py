"""Headless one-environment M0 test for the converted FALCON G1 asset."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from isaaclab.app import AppLauncher


DEFAULT_ASSET = REPO / "assets" / "generated" / "g1_29dof_falcon_rubberhand.usd"
DEFAULT_ONNX = Path(r"C:\Robotics\FALCON-upstream\sim2real\models\falcon\g1_29dof.onnx")
DEFAULT_RUN_ROOT = REPO / "reports" / "artifacts" / "m0" / "falcon_adapter_run"
DT = 0.005
DECIMATION = 4

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
parser.add_argument("--case-duration", type=float, default=2.0)
parser.add_argument("--forward-speed", type=float, default=0.20)
parser.add_argument("--turn-target", type=float, default=0.35)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
ARGS = parser.parse_args()
APP = AppLauncher(ARGS).app


def clean(value: Any) -> Any:
    if hasattr(value, "item"):
        return clean(value.item())
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def yaw_from_quat(q) -> float:
    w, x, y, z = [float(value) for value in q]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def roll_pitch_from_quat(q) -> tuple[float, float]:
    w, x, y, z = [float(value) for value in q]
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    argument = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    return roll, math.asin(argument)


def main() -> int:
    args = ARGS
    asset = args.asset.resolve()
    onnx = args.onnx.resolve()
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    if not asset.is_file():
        raise FileNotFoundError(asset)
    if not onnx.is_file():
        raise FileNotFoundError(onnx)
    if args.case_duration <= 0.0 or args.forward_speed < 0.0:
        raise ValueError("case duration must be positive and forward speed non-negative")

    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from lift_curtain.control.falcon_adapter import FalconAdapter, FalconAdapterConfig, VelocityCommand
    from lift_curtain.control.falcon_contract import (
        DEFAULT_JOINT_POS, ISAACLAB_TO_OFFICIAL, JOINT_EFFORT_LIMIT, JOINT_KD,
        JOINT_KP, JOINT_VELOCITY_LIMIT, OFFICIAL_POLICY_JOINT_ORDER,
    )
    from lift_curtain.control.se2_feedback import PushPathTracker, PushPathTrackerConfig, SE2Reference, straight_reference

    config = {
        "asset": str(asset), "onnx": str(onnx), "run_root": str(run_root),
        "seed": args.seed, "num_envs": 1, "headless": True, "physics_dt_s": DT,
        "decimation": DECIMATION, "control_dt_s": DT * DECIMATION,
        "case_duration_s": args.case_duration, "forward_speed_mps": args.forward_speed,
        "turn_target_rad": args.turn_target,
    }
    (run_root / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.reset_peak_memory_stats()

    sim = None
    try:
        sim = SimulationContext(SimulationCfg(dt=DT, render_interval=DECIMATION, device="cuda:0"))
        ground = sim_utils.GroundPlaneCfg()
        ground.func("/World/defaultGroundPlane", ground)
        actuators = {
            name: ImplicitActuatorCfg(
                joint_names_expr=[name], effort_limit_sim=float(JOINT_EFFORT_LIMIT[index]),
                velocity_limit_sim=float(JOINT_VELOCITY_LIMIT[index]), stiffness=float(JOINT_KP[index]),
                damping=float(JOINT_KD[index]),
            )
            for index, name in enumerate(OFFICIAL_POLICY_JOINT_ORDER)
        }
        initial_joint_pos = {name: float(DEFAULT_JOINT_POS[index]) for index, name in enumerate(OFFICIAL_POLICY_JOINT_ORDER)}
        robot = Articulation(ArticulationCfg(
            prim_path="/World/envs/env_0/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(asset), activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False, retain_accelerations=False,
                    max_linear_velocity=1000.0, max_angular_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    articulation_enabled=True, enabled_self_collisions=True, fix_root_link=False,
                    solver_position_iteration_count=8, solver_velocity_iteration_count=4,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.80), rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos=initial_joint_pos, joint_vel={".*": 0.0},
            ),
            actuators=actuators,
        ))
        sim.reset()
        robot.reset()
        robot.update(DT)
        contract = {
            "joint_names": list(robot.joint_names), "body_names": list(robot.body_names),
            "is_fixed_base": bool(robot.is_fixed_base), "num_joints": len(robot.joint_names),
            "num_bodies": len(robot.body_names),
        }
        (run_root / "articulation_contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
        FalconAdapter.validate_articulation(robot)
        adapter = FalconAdapter(FalconAdapterConfig(onnx_path=onnx))

        initial_root_pose = robot.data.root_pose_w[0].detach().cpu().numpy().copy()
        initial_joint = robot.data.joint_pos[0].detach().clone()
        initial_velocity = robot.data.joint_vel[0].detach().clone()
        initial_pose = robot.data.root_pos_w[0].detach().cpu().numpy().copy()
        initial_yaw = yaw_from_quat(initial_root_pose[3:7])
        tracker = PushPathTracker(PushPathTrackerConfig(vx_max=0.30, vy_limit=0.10, wz_limit=0.30))
        rows: list[dict[str, Any]] = []
        cases = (
            ("stand", "stand"), ("forward", "forward"),
            ("yaw_left", "yaw_left"), ("yaw_right", "yaw_right"),
        )
        case_summaries = []
        for case_name, case_kind in cases:
            robot.write_root_pose_to_sim(torch.as_tensor(initial_root_pose, device=sim.device).unsqueeze(0))
            robot.write_joint_state_to_sim(initial_joint.unsqueeze(0), initial_velocity.unsqueeze(0))
            robot.set_joint_position_target(initial_joint.unsqueeze(0))
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(DT)
            adapter.reset()
            tracker.reset()
            origin = robot.data.root_pos_w[0].detach().cpu().numpy()[:2].astype(float)
            yaw0 = yaw_from_quat(robot.data.root_quat_w[0].detach().cpu().numpy())
            total_steps = int(round(args.case_duration / DT))
            fall_reason = None
            case_rows: list[dict[str, Any]] = []
            for step in range(total_steps):
                state = adapter.make_state(robot)
                current_yaw = yaw_from_quat(state.base_quat_w)
                if case_kind == "forward":
                    reference = straight_reference(step * DT, tuple(origin), yaw=yaw0, speed_mps=args.forward_speed)
                elif case_kind == "yaw_left":
                    reference = SE2Reference(tuple(origin), yaw0 + args.turn_target)
                elif case_kind == "yaw_right":
                    reference = SE2Reference(tuple(origin), yaw0 - args.turn_target)
                else:
                    reference = SE2Reference(tuple(origin), yaw0)
                command = tracker((float(state.base_pos_w[0]), float(state.base_pos_w[1]), current_yaw), reference)
                if step % DECIMATION == 0:
                    target = adapter.compute_target(state, VelocityCommand(
                        vx=float(command[0]), vy=float(command[1]), wz=float(command[2]),
                    ))
                    target_tensor = torch.as_tensor(target, device=sim.device, dtype=robot.data.joint_pos.dtype).unsqueeze(0)
                    robot.set_joint_position_target(target_tensor)
                robot.write_data_to_sim()
                sim.step(render=False)
                robot.update(DT)
                after = adapter.make_state(robot)
                roll, pitch = roll_pitch_from_quat(after.base_quat_w)
                finite = bool(np.isfinite(after.base_pos_w).all() and np.isfinite(after.joint_pos).all())
                if not finite and fall_reason is None:
                    fall_reason = "NONFINITE_STATE"
                elif after.base_pos_w[2] < 0.55 and fall_reason is None:
                    fall_reason = "ROOT_HEIGHT_BELOW_0P55"
                elif (abs(roll) > 0.6 or abs(pitch) > 0.6) and fall_reason is None:
                    fall_reason = "ROOT_ROLL_PITCH_EXCEEDED_0P6"
                row = {
                    "case": case_name, "step": step, "time_s": (step + 1) * DT,
                    "command_vx": float(command[0]), "command_vy": float(command[1]), "command_wz": float(command[2]),
                    "measured_vx": float(after.base_lin_vel_b[0]), "measured_vy": float(after.base_lin_vel_b[1]),
                    "measured_wz": float(after.base_ang_vel_b[2]), "base_x": float(after.base_pos_w[0]),
                    "base_y": float(after.base_pos_w[1]), "base_z": float(after.base_pos_w[2]),
                    "yaw": float(yaw_from_quat(after.base_quat_w)), "roll": float(roll), "pitch": float(pitch),
                    "effort_max_abs": float(np.max(np.abs(after.joint_effort))), "effort_source": after.effort_source,
                    "joint_tracking_rms": float(np.sqrt(np.mean(np.square(after.joint_pos -
                        robot.data.default_joint_pos[0].detach().cpu().numpy())))),
                    "gpu_allocated_mb": float(torch.cuda.memory_allocated() / (1024 ** 2)),
                    "gpu_reserved_mb": float(torch.cuda.memory_reserved() / (1024 ** 2)),
                    "finite": finite, "fall": fall_reason is not None, "fall_reason": fall_reason or "",
                }
                rows.append(row)
                case_rows.append(row)
            forward_displacement = float(case_rows[-1]["base_x"] - case_rows[0]["base_x"])
            yaw_change = float((case_rows[-1]["yaw"] - case_rows[0]["yaw"] + math.pi) % (2 * math.pi) - math.pi)
            expected_direction = 1.0 if case_kind == "forward" else 0.0
            direction_ok = expected_direction == 0.0 or forward_displacement > 0.02
            yaw_ok = case_kind not in ("yaw_left", "yaw_right") or (
                yaw_change * (1.0 if case_kind == "yaw_left" else -1.0) > 0.02
            )
            summary = {
                "case": case_name, "status": "PASS" if fall_reason is None and direction_ok and yaw_ok else "FAIL",
                "steps_completed": len(case_rows), "steps_requested": total_steps,
                "fall": fall_reason is not None, "fall_reason": fall_reason,
                "base_displacement_x_m": forward_displacement, "yaw_change_rad": yaw_change,
                "max_gpu_allocated_mb": max(row["gpu_allocated_mb"] for row in case_rows),
                "max_gpu_reserved_mb": max(row["gpu_reserved_mb"] for row in case_rows),
                "effort_signal": case_rows[-1]["effort_source"],
            }
            case_summaries.append(summary)

        csv_path = run_root / "telemetry.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        overall = "PASS" if all(item["status"] == "PASS" for item in case_summaries) else "FAIL"
        report = {
            **config, "status": overall, "cases": case_summaries,
            "telemetry_csv": str(csv_path), "peak_gpu_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
            "peak_gpu_reserved_mb": float(torch.cuda.max_memory_reserved() / (1024 ** 2)),
            "torch_cuda_device": torch.cuda.get_device_name(0),
        }
        (run_root / "summary.json").write_text(json.dumps(clean(report), indent=2) + "\n", encoding="utf-8")
        (REPO / "reports" / "artifacts" / "m0" / "m0_summary.json").write_text(
            json.dumps(clean(report), indent=2) + "\n", encoding="utf-8")
        print(json.dumps(clean(report), indent=2))
        return 0 if overall == "PASS" else 2
    except Exception as exc:
        failure = {**config, "status": "BLOCKED", "error_type": type(exc).__name__, "error": str(exc)}
        (run_root / "failure.json").write_text(json.dumps(clean(failure), indent=2) + "\n", encoding="utf-8")
        print(json.dumps(clean(failure), indent=2))
        return 2
    finally:
        if sim is not None:
            try:
                sim.stop()
                sim.clear_instance()
            except Exception:
                pass
        try:
            APP.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
