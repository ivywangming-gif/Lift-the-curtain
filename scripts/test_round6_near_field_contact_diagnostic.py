"""Round 6 diagnostic-only near-field G1 rubber-hand contact experiment.

The root pose is changed once while constructing the episode reset.  After
that write, the root is controlled only by a frozen SE(2) stance reference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from isaaclab.app import AppLauncher

from lift_curtain.control.deployment_observation import (
    DeploymentObservation,
    ProprioceptiveThresholds,
    classify_proprioceptive_response,
)
from lift_curtain.paths import falcon_onnx_path

DEFAULT_ASSET = REPO / "assets" / "generated" / "g1_29dof_falcon_rubberhand.usd"
DEFAULT_ONNX = falcon_onnx_path()
DEFAULT_RUN_ROOT = REPO / "reports" / "artifacts" / "round6" / "near_field_contact_diagnostic"
DEFAULT_GEOMETRY_SUMMARY = REPO / "reports" / "artifacts" / "round4" / "hand_first_feasibility_oriented_bbox" / "summary.json"
DT = 0.005
DECIMATION = 4
CONTROL_DT = DT * DECIMATION
TOP_DRIFT_TOLERANCE_M = 0.005

# Fixed before launching any condition. These are the finite geometry-gated
# posture and one small bounded continuation; there is no online search.
PREPROBE_UPPER = np.zeros(14, dtype=np.float32)
PREPROBE_UPPER[7] = -0.45
PREPROBE_UPPER[10] = 0.45
PROBE_TARGET_UPPER = PREPROBE_UPPER.copy()
PROBE_TARGET_UPPER[7] = -0.55
PROBE_TARGET_UPPER[10] = 0.55
PALM_FRONT_OFFSET_M = 0.14248830493846448

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--condition", choices=("none", "rigid_patch", "curtain"), required=True)
parser.add_argument("--scene-profile", choices=("legacy_155m", "formal_2m"), default="legacy_155m", help="Keep legacy defaults unchanged; formal_2m selects the independent Round 8 task-scale geometry.")
parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
parser.add_argument("--wall-x", type=float, default=0.39)
parser.add_argument("--geometry-summary", type=Path, default=DEFAULT_GEOMETRY_SUMMARY)
parser.add_argument("--reset-root-x", type=float, default=0.0)
parser.add_argument("--reset-root-y", type=float, default=0.0)
parser.add_argument("--reset-root-z", type=float, default=None)
parser.add_argument("--palm-front-offset", type=float, default=PALM_FRONT_OFFSET_M)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--settle-duration", type=float, default=0.50)
parser.add_argument("--precontact-extension-duration", type=float, default=1.50)
parser.add_argument("--precontact-hold-duration", type=float, default=0.40)
parser.add_argument("--probe-duration", type=float, default=1.50)
parser.add_argument("--contact-hold-duration", type=float, default=0.50)
parser.add_argument("--retract-duration", type=float, default=1.50)
parser.add_argument("--post-retract-hold-duration", type=float, default=0.40)
parser.add_argument("--contact-event-gap", type=float, default=0.018)
parser.add_argument("--contact-event-displacement", type=float, default=0.002)
parser.add_argument("--minimum-nonhand-clearance", type=float, default=0.08)
parser.add_argument("--sample-every", type=int, default=4)
parser.add_argument("--base-mode", choices=("zero", "se2_stance"), default="zero")
parser.add_argument("--preserve-curtain-velocity", action="store_true", help="Do not write nodal velocities after reset; preserve the runtime deformable state.")
parser.add_argument("--skip-probe", action="store_true", help="Run the same settle/extension/hold window but omit probe, contact-hold, and retract.")
parser.add_argument("--do-sweep", action="store_true", help="After contact hold, run one bounded lateral shoulder-roll sweep and return to the probe posture.")
parser.add_argument("--sweep-duration", type=float, default=2.0)
parser.add_argument("--sweep-amplitude-rad", type=float, default=0.10)
parser.add_argument("--do-walkthrough", action="store_true", help="After contact/sweep, run a bounded SE(2) feedback passage diagnostic.")
parser.add_argument("--walkthrough-duration", type=float, default=8.0)
parser.add_argument("--walkthrough-distance", type=float, default=0.55)
parser.add_argument("--walkthrough-hand-y-offset-rad", type=float, default=0.10)
parser.add_argument(
    "--walkthrough-upper-policy",
    choices=("extended", "safe_retract"),
    default="extended",
    help="Walkthrough-only upper schedule; safe_retract returns smoothly to the verified pre-probe posture.",
)
parser.add_argument("--walkthrough-vx-max", type=float, default=None, help="Optional bounded forward limit for the walkthrough-only tracker.")
parser.add_argument("--walkthrough-feedforward-speed", type=float, default=None, help="Optional world-frame trajectory feedforward speed; when omitted the legacy static endpoint reference is retained.")
parser.add_argument("--walkthrough-terminal-speed", type=float, default=0.15, help="Minimum effective feedforward while closing the final localization error.")
parser.add_argument("--walkthrough-position-tolerance", type=float, default=0.08)
parser.add_argument("--walkthrough-stop-duration", type=float, default=1.50)
parser.add_argument("--walkthrough-stop-speed-threshold", type=float, default=0.06)
parser.add_argument("--walkthrough-stop-dwell-duration", type=float, default=0.30)
parser.add_argument("--walkthrough-safety-clearance", type=float, default=None, help="External robot-geometry safety stop margin in front of the known plane.")
parser.add_argument("--gate-walkthrough-on-proprio", action="store_true", help="Continue to sweep/walk only when the deployment-only probe rule returns RESPONSIVE.")
parser.add_argument("--proprio-baseline-csv", type=Path, default=None, help="Deployment-only matched baseline CSV used by the runtime probe gate.")
parser.add_argument("--proprio-thresholds-json", type=Path, default=None, help="Deployment-only deterministic threshold configuration for the runtime probe gate.")
AppLauncher.add_app_launcher_args(parser)
ARGS = parser.parse_args()
APP = AppLauncher(ARGS).app


def clean(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if hasattr(value, "item"):
        return clean(value.item())
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _deployment_vector(row: dict[str, str], field: str) -> np.ndarray:
    value = np.asarray(json.loads(row[field]), dtype=np.float64).reshape(-1)
    if value.size == 0 or not np.isfinite(value).all():
        raise ValueError(f"{field} must contain a finite non-empty vector")
    return value


def deployment_observation_from_row(row: dict[str, str]) -> DeploymentObservation:
    """Build a classifier input from the explicitly separated runtime fields."""
    return DeploymentObservation(
        q=_deployment_vector(row, "deployment_q_actual_json"),
        qdot=_deployment_vector(row, "deployment_qdot_json"),
        q_cmd=_deployment_vector(row, "deployment_q_cmd_json"),
        motor_effort=_deployment_vector(row, "deployment_motor_effort_json"),
        previous_action=_deployment_vector(row, "deployment_previous_action_json"),
        hand_fk_position_w=_deployment_vector(row, "deployment_hand_fk_position_w_json"),
        hand_fk_velocity_w=_deployment_vector(row, "deployment_hand_fk_velocity_w_json"),
        base_position_w=_deployment_vector(row, "deployment_base_position_w_json"),
        base_velocity_w=_deployment_vector(row, "deployment_base_velocity_b_json"),
        localization_assumption=row.get("deployment_localization_assumption", ""),
    )


def load_deployment_history_csv(path: Path) -> list[DeploymentObservation]:
    """Load only the probe/contact phases from a separated deployment CSV."""
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "phase",
            "deployment_q_actual_json",
            "deployment_qdot_json",
            "deployment_q_cmd_json",
            "deployment_motor_effort_json",
            "deployment_previous_action_json",
            "deployment_hand_fk_position_w_json",
            "deployment_hand_fk_velocity_w_json",
            "deployment_base_position_w_json",
            "deployment_base_velocity_b_json",
            "deployment_localization_assumption",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing deployment fields: {sorted(missing)}")
        rows = [row for row in reader if row.get("phase") in ("probe", "contact_hold")]
    if not rows:
        raise ValueError(f"{path} has no probe/contact deployment rows")
    return [deployment_observation_from_row(row) for row in rows]


def yaw_from_quat(q: np.ndarray) -> float:
    w, x, y, z = [float(value) for value in q]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def roll_pitch_from_quat(q: np.ndarray) -> tuple[float, float]:
    w, x, y, z = [float(value) for value in q]
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    argument = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    return roll, math.asin(argument)


def decode_last_observation(adapter: Any) -> dict[str, float]:
    from lift_curtain.control.falcon_contract import (
        OBSERVATION_DIMS,
        OBSERVATION_ORDER,
        OBSERVATION_SCALES,
        SINGLE_FRAME_DIM,
    )

    frame = np.asarray(adapter.last_observation, dtype=np.float32).reshape(-1)[-SINGLE_FRAME_DIM:]
    decoded: dict[str, float] = {}
    offset = 0
    for name in OBSERVATION_ORDER:
        size = OBSERVATION_DIMS[name]
        field = frame[offset : offset + size] / float(OBSERVATION_SCALES[name])
        offset += size
        if name == "command_lin_vel":
            decoded["actor_obs_command_vx"] = float(field[0])
            decoded["actor_obs_command_vy"] = float(field[1])
        elif name == "command_ang_vel":
            decoded["actor_obs_command_wz"] = float(field[0])
        elif name == "command_stand":
            decoded["actor_obs_command_stand"] = float(field[0])
    return decoded


def set_gravity(stage, magnitude_mps2: float) -> list[str]:
    from pxr import Gf, UsdPhysics

    scenes = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
    if not scenes:
        scenes = [UsdPhysics.Scene.Define(stage, "/physicsScene").GetPrim()]
    paths = []
    for prim in scenes:
        scene = UsdPhysics.Scene(prim)
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(float(magnitude_mps2))
        paths.append(str(prim.GetPath()))
    return paths


def body_position(robot: Any, body_name: str) -> np.ndarray:
    names = tuple(robot.body_names)
    if body_name not in names:
        raise RuntimeError(f"missing body: {body_name}")
    index = names.index(body_name)
    for field_name in ("body_link_pose_w", "body_pose_w", "body_state_w"):
        value = getattr(robot.data, field_name, None)
        if value is None:
            continue
        array = value[0, index].detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)[0, index]
        array = np.asarray(array, dtype=np.float64)
        if array.shape[-1] >= 3:
            return array[:3].copy()
    raise RuntimeError("no public body pose field is available")


def normalize_quat(value: np.ndarray) -> np.ndarray:
    quat = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if not math.isfinite(norm) or norm < 1.0e-8:
        return np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
    return quat / norm


def quat_conjugate(value: np.ndarray) -> np.ndarray:
    quat = normalize_quat(value)
    return np.asarray((quat[0], -quat[1], -quat[2], -quat[3]), dtype=np.float64)


def quat_rotate(quat_value: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quat = normalize_quat(quat_value)
    vec = np.asarray(vector, dtype=np.float64)
    qvec = quat[1:]
    twice_cross = 2.0 * np.cross(qvec, vec)
    return vec + quat[0] * twice_cross + np.cross(qvec, twice_cross)


def body_pose(robot: Any, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    names = tuple(robot.body_names)
    if body_name not in names:
        raise RuntimeError(f"missing body: {body_name}")
    index = names.index(body_name)
    for field_name in ("body_link_pose_w", "body_pose_w", "body_state_w"):
        value = getattr(robot.data, field_name, None)
        if value is None:
            continue
        array = value[0, index].detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)[0, index]
        array = np.asarray(array, dtype=np.float64)
        if array.shape[-1] >= 7:
            return array[:3].copy(), normalize_quat(array[3:7])
    raise RuntimeError("no public body pose field is available")


def body_poses(robot: Any) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {name: body_pose(robot, name) for name in robot.body_names}


def contact_reading(sensor: Any) -> dict[str, Any]:
    """Return evaluation-only foot contact telemetry when the reporter exists.

    This function is deliberately called only while constructing telemetry.  No
    contact reporter value is passed to FALCON, the SE(2) tracker, or the
    deployment-only probe gate.
    """
    if sensor is None:
        return {
            "foot_contact_reporter_available": False,
            "left_foot_force_norm_N": None,
            "right_foot_force_norm_N": None,
            "left_foot_contact": None,
            "right_foot_contact": None,
        }
    try:
        raw_force = sensor.data.net_forces_w
        if hasattr(raw_force, "detach"):
            raw_force = raw_force.detach().cpu().numpy()
        force = np.asarray(raw_force)[0]
        norms = np.linalg.norm(force, axis=-1)
        result: dict[str, Any] = {
            "foot_contact_reporter_available": True,
            "left_foot_force_norm_N": None,
            "right_foot_force_norm_N": None,
            "left_foot_contact": None,
            "right_foot_contact": None,
        }
        for name, value in zip(sensor.body_names, norms):
            if "left" in name:
                result["left_foot_force_norm_N"] = float(value)
                result["left_foot_contact"] = bool(value > 1.0)
            elif "right" in name:
                result["right_foot_force_norm_N"] = float(value)
                result["right_foot_contact"] = bool(value > 1.0)
        return result
    except Exception as exc:
        return {
            "foot_contact_reporter_available": False,
            "left_foot_force_norm_N": None,
            "right_foot_force_norm_N": None,
            "left_foot_contact": None,
            "right_foot_contact": None,
            "foot_contact_reporter_error": f"{type(exc).__name__}: {exc}",
        }


def pose_record(pose: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
    position, orientation = pose
    return {
        "position_m": position.tolist(),
        "orientation_wxyz": orientation.tolist(),
        "local_x_world": quat_rotate(orientation, np.asarray((1.0, 0.0, 0.0))).tolist(),
        "local_y_world": quat_rotate(orientation, np.asarray((0.0, 1.0, 0.0))).tolist(),
        "local_z_world": quat_rotate(orientation, np.asarray((0.0, 0.0, 1.0))).tolist(),
    }


def discover_body_paths(stage, body_names: tuple[str, ...], robot_root: str) -> dict[str, list[str]]:
    paths = {name: [] for name in body_names}
    prefix = robot_root.rstrip("/") + "/"
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith(prefix) and str(prim.GetName()) in paths:
            paths[str(prim.GetName())].append(path)
    return paths


def fallback_radius(name: str) -> float:
    if "rubber_hand" in name:
        return 0.085
    if name in {"pelvis", "pelvis_contour_link", "waist_yaw_link", "waist_roll_link", "torso_link", "waist_support_link", "head_link", "d435_link", "mid360_link"}:
        return 0.24
    if any(token in name for token in ("hip", "knee", "ankle")):
        return 0.13
    if any(token in name for token in ("shoulder", "elbow", "wrist")):
        return 0.10
    return 0.12


def build_bbox_template(stage, robot: Any, reset_poses: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    from pxr import Usd, UsdGeom

    names = tuple(robot.body_names)
    paths = discover_body_paths(stage, names, "/World/envs/env_0/Robot")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    template: dict[str, dict[str, Any]] = {}
    for name in names:
        center, orientation = reset_poses[name]
        selected_path = None
        local_corners = None
        bound_min = None
        bound_max = None
        radius = fallback_radius(name)
        for path in sorted(paths.get(name, ()), key=lambda item: (item.count("/"), len(item))):
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                continue
            try:
                value = cache.ComputeWorldBound(prim).GetRange()
                if value.IsEmpty():
                    continue
                lower = np.asarray(value.GetMin(), dtype=np.float64)
                upper = np.asarray(value.GetMax(), dtype=np.float64)
                corners = np.asarray([[x, y, z] for x in (lower[0], upper[0]) for y in (lower[1], upper[1]) for z in (lower[2], upper[2])], dtype=np.float64)
                candidate_radius = float(np.max(np.linalg.norm(corners - center[None, :], axis=1)))
                if not math.isfinite(candidate_radius) or candidate_radius <= 1.0e-5 or candidate_radius > 2.0:
                    continue
                selected_path = path
                bound_min = lower
                bound_max = upper
                radius = candidate_radius
                local_corners = np.asarray([quat_rotate(quat_conjugate(orientation), corner - center) for corner in corners], dtype=np.float64)
                break
            except Exception:
                continue
        template[name] = {
            "body_name": name,
            "prim_path": selected_path,
            "source": "runtime_usd_world_bbox_reset_body_local_box" if local_corners is not None else "category_fallback_radius",
            "radius_m": float(radius),
            "reset_center_m": center.tolist(),
            "reset_orientation_wxyz": normalize_quat(orientation).tolist(),
            "reset_bbox_min_m": None if bound_min is None else bound_min.tolist(),
            "reset_bbox_max_m": None if bound_max is None else bound_max.tolist(),
            "reset_local_bbox_corners_m": None if local_corners is None else local_corners.tolist(),
        }
    return template, paths


def geometry_snapshot(poses: dict[str, tuple[np.ndarray, np.ndarray]], template: dict[str, dict[str, Any]], root_x: float, wall_x: float) -> dict[str, Any]:
    relative_front: dict[str, float] = {}
    absolute_min_x: dict[str, float] = {}
    for name, (center, orientation) in poses.items():
        corners = template[name].get("reset_local_bbox_corners_m")
        if corners is None:
            front = float(center[0] + template[name]["radius_m"])
            back = float(center[0] - template[name]["radius_m"])
        else:
            world_corners = np.asarray([quat_rotate(orientation, corner) for corner in np.asarray(corners, dtype=np.float64)], dtype=np.float64)
            front = float(center[0] + np.max(world_corners[:, 0]))
            back = float(center[0] + np.min(world_corners[:, 0]))
        relative_front[name] = front - float(root_x)
        absolute_min_x[name] = back
    nonhand = {name: value for name, value in relative_front.items() if "rubber_hand" not in name}
    nonhand_min_x = {name: value for name, value in absolute_min_x.items() if "rubber_hand" not in name}
    hand_front = relative_front.get("right_rubber_hand", float("nan"))
    nonhand_front = max(nonhand.values()) if nonhand else float("nan")
    return {
        "relative_front_extent_by_body_m": relative_front,
        "absolute_back_extent_by_body_m": absolute_min_x,
        "right_hand_front_relative_x_m": hand_front,
        "max_nonhand_front_relative_x_m": nonhand_front,
        "min_robot_back_x_m": min(absolute_min_x.values()) if absolute_min_x else float("nan"),
        "min_nonhand_back_x_m": min(nonhand_min_x.values()) if nonhand_min_x else float("nan"),
        "all_robot_bodies_past_wall": bool(absolute_min_x) and min(absolute_min_x.values()) > float(wall_x),
        "all_nonhand_bodies_past_wall": bool(nonhand_min_x) and min(nonhand_min_x.values()) > float(wall_x),
        "right_hand_bbox_gap_m": float(wall_x - (root_x + hand_front)),
        "nonhand_clearance_m": float(wall_x - (root_x + nonhand_front)),
        "hand_first_clearance_margin_m": float(hand_front - nonhand_front),
        "root_x_m": float(root_x),
        "wall_x_m": float(wall_x),
    }


def read_runtime(deformable) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    simulation, _, rest = deformable.get_nodal_positions()
    positions = as_numpy(simulation)[0].astype(np.float64)
    rest_positions = as_numpy(rest)[0].astype(np.float64)
    velocities = as_numpy(deformable.get_nodal_velocities())[0].astype(np.float64)
    return positions, velocities, rest_positions


def zero_runtime_velocity(deformable, node_count: int) -> None:
    deformable.set_nodal_velocities(np.zeros((1, node_count, 3), dtype=np.float32))


def write_robot_reset(robot, sim, root_pose, joint_pos, joint_vel) -> None:
    robot.write_root_pose_to_sim(root_pose.unsqueeze(0))
    robot.write_joint_state_to_sim(joint_pos.unsqueeze(0), joint_vel.unsqueeze(0))
    robot.set_joint_position_target(joint_pos.unsqueeze(0))
    robot.write_data_to_sim()
    sim.step(render=False)
    robot.update(DT)


def smoothstep(alpha: float) -> float:
    value = max(0.0, min(1.0, float(alpha)))
    return value * value * (3.0 - 2.0 * value)


def interpolate_upper(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    value = smoothstep(alpha)
    return ((1.0 - value) * start + value * end).astype(np.float32)


def save_plot(run_root: Path, rows: list[dict[str, Any]], wall_x: float) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = np.asarray([row["time_s"] for row in rows])
    hand_x = np.asarray([row["hand_x"] for row in rows])
    base_x = np.asarray([row["base_x"] for row in rows])
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(times, hand_x, label="right hand x")
    axes[0].axhline(wall_x, color="black", linestyle="--", linewidth=0.8, label="obstacle x")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].set_ylabel("x (m)")
    axes[1].plot(times, base_x, label="base x")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("x (m)")
    figure.tight_layout()
    path = run_root / "matched_trajectory.png"
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return str(path)


def main() -> int:
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.sim.utils import get_current_stage
    from lift_curtain.control.falcon_adapter import FalconAdapter, FalconAdapterConfig, VelocityCommand
    from lift_curtain.control.falcon_contract import (
        DEFAULT_JOINT_POS,
        OFFICIAL_TO_ISAACLAB,
        ISAACLAB_TO_OFFICIAL,
        JOINT_EFFORT_LIMIT,
        JOINT_KD,
        JOINT_KP,
        JOINT_POS_LOWER,
        JOINT_POS_UPPER,
        JOINT_VELOCITY_LIMIT,
        OFFICIAL_POLICY_JOINT_ORDER,
    )
    from lift_curtain.control.se2_feedback import PushPathTracker, PushPathTrackerConfig, SE2Reference

    asset = ARGS.asset.resolve()
    onnx = ARGS.onnx.resolve()
    run_root = (ARGS.run_root / ARGS.condition).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {
        "stage": "ROUND6_DIAGNOSTIC_ONLY_NEAR_FIELD_CONTACT_REPLAY",
        "condition": ARGS.condition,
        "scene_profile": ARGS.scene_profile,
        "asset": str(asset),
        "onnx": str(onnx),
        "num_envs": 1,
        "headless": True,
        "physics_dt_s": DT,
        "decimation": DECIMATION,
        "control_dt_s": DT * DECIMATION,
        "wall_x_m": ARGS.wall_x,
        "reset_only_root_pose_request_m": [ARGS.reset_root_x, ARGS.reset_root_y, ARGS.reset_root_z],
        "geometry_summary": str(ARGS.geometry_summary.resolve()),
        "palm_front_offset_m": ARGS.palm_front_offset,
        "durations_s": {
            "initial_stance_settle": ARGS.settle_duration,
            "post_reset_reanchor_window": ARGS.settle_duration,
            "precontact_extension": ARGS.precontact_extension_duration,
            "precontact_hold": ARGS.precontact_hold_duration,
            "probe": ARGS.probe_duration,
            "contact_hold": ARGS.contact_hold_duration,
            "retract": ARGS.retract_duration,
        },
        "hypothesis": "The previous M2.1 blocker is base approach reachability. A reset-only root pose inside the measured hand-first interval should let the frozen FALCON upper reference reach a real curtain contact without runtime teleport.",
        "exact_question": "With only the episode-reset root pose changed, do actual FK, collision geometry, upper tracking, and runtime curtain nodes show a valid hand-first contact and safe retract?",
        "changed_variable": "reset-time root x/y only; no checkpoint, actuator, physics schema, upper posture, or runtime control law change",
        "fixed_upper_reference_pair": {
            "preprobe": PREPROBE_UPPER.tolist(),
            "probe_target": PROBE_TARGET_UPPER.tolist(),
        },
        "diagnostic_scope": "Reset-only near-field contact; this is not a traversal result.",
        "classification_policy": "The deployment rule uses only separated proprioceptive fields. Scene labels and runtime deformable state remain evaluation-only.",
        "diagnostic_only": True,
        "base_mode": ARGS.base_mode,
        "preserve_curtain_velocity": bool(ARGS.preserve_curtain_velocity),
        "skip_probe": bool(ARGS.skip_probe),
        "do_sweep": bool(ARGS.do_sweep),
        "sweep_duration_s": ARGS.sweep_duration,
        "sweep_joint": "right_shoulder_yaw_joint",
        "sweep_joint_official_upper_index": 9,
        "sweep_amplitude_rad": ARGS.sweep_amplitude_rad,
        "do_walkthrough": bool(ARGS.do_walkthrough),
        "walkthrough_duration_s": ARGS.walkthrough_duration,
        "walkthrough_distance_m": ARGS.walkthrough_distance,
        "walkthrough_hand_y_offset_rad": ARGS.walkthrough_hand_y_offset_rad,
        "walkthrough_upper_policy": ARGS.walkthrough_upper_policy,
        "walkthrough_vx_max_mps": ARGS.walkthrough_vx_max,
        "walkthrough_feedforward_speed_mps": ARGS.walkthrough_feedforward_speed,
        "walkthrough_terminal_speed_mps": ARGS.walkthrough_terminal_speed,
        "walkthrough_position_tolerance_m": ARGS.walkthrough_position_tolerance,
        "walkthrough_stop_duration_s": ARGS.walkthrough_stop_duration,
        "walkthrough_stop_speed_threshold_mps": ARGS.walkthrough_stop_speed_threshold,
        "walkthrough_stop_dwell_duration_s": ARGS.walkthrough_stop_dwell_duration,
        "walkthrough_safety_clearance_m": ARGS.walkthrough_safety_clearance,
        "gate_walkthrough_on_proprio": bool(ARGS.gate_walkthrough_on_proprio),
        "proprio_baseline_csv": None if ARGS.proprio_baseline_csv is None else str(ARGS.proprio_baseline_csv.resolve()),
        "proprio_thresholds_json": None if ARGS.proprio_thresholds_json is None else str(ARGS.proprio_thresholds_json.resolve()),
        "walkthrough_controller": "PushPathTracker world-frame moving feedforward reference plus measured-pose SE(2) feedback, explicit Walk then Stand",
        "initialization_policy": "Explicit DEFAULT_JOINT_POS reordered to IsaacLab joint order; the pre-write articulation buffer is not used.",
        "contact_gate": "CONTACT PASS requires legal hand-first geometry, a physical curtain runtime response, and finite/no-fall retract; the gap proxy alone is insufficient.",
        "runtime_decision_isolation": {
            "probe_gate_inputs": "DeploymentObservation only",
            "forbidden_fields_not_used_for_gate": [
                "object_kind",
                "simulator_contact_flag",
                "true_hand_obstacle_gap_m",
                "curtain_node_positions_w",
                "curtain_node_displacement_m",
                "true_passability",
            ],
            "robot_geometry_safety_stop_role": "external_safety_termination_only",
        },
    }
    (run_root / "resolved_config.json").write_text(json.dumps(clean(config), indent=2) + "\n", encoding="utf-8")
    report: dict[str, Any] = {**config, "status": "BLOCKED"}
    sim = None
    contact_sensor = None
    try:
        if not asset.is_file() or not onnx.is_file():
            raise FileNotFoundError(f"missing asset or checkpoint: {asset}, {onnx}")
        durations = (
            ARGS.settle_duration,
            ARGS.probe_duration,
            ARGS.contact_hold_duration,
            ARGS.retract_duration,
        )
        if any(value <= 0.0 for value in durations) or ARGS.sample_every < 1:
            raise ValueError("durations must be positive and sample-every must be positive")
        if ARGS.do_walkthrough:
            if min(
                ARGS.walkthrough_position_tolerance,
                ARGS.walkthrough_stop_duration,
                ARGS.walkthrough_stop_speed_threshold,
                ARGS.walkthrough_stop_dwell_duration,
            ) <= 0.0:
                raise ValueError("walkthrough position tolerance and stop duration must be positive")
            if ARGS.walkthrough_stop_dwell_duration > ARGS.walkthrough_stop_duration:
                raise ValueError("walkthrough stop dwell duration cannot exceed stop duration")
            if ARGS.walkthrough_feedforward_speed is not None:
                if not math.isfinite(ARGS.walkthrough_feedforward_speed) or ARGS.walkthrough_feedforward_speed <= 0.0:
                    raise ValueError("walkthrough feedforward speed must be finite and positive")
                if not 0.0 < ARGS.walkthrough_terminal_speed <= ARGS.walkthrough_feedforward_speed:
                    raise ValueError("walkthrough terminal speed must be positive and no larger than feedforward speed")
                if ARGS.walkthrough_vx_max is None or ARGS.walkthrough_feedforward_speed > ARGS.walkthrough_vx_max:
                    raise ValueError("moving feedforward requires walkthrough-vx-max >= feedforward speed")
        np.random.seed(ARGS.seed)
        torch.manual_seed(ARGS.seed)
        sim = SimulationContext(SimulationCfg(dt=DT, render_interval=DECIMATION, device="cuda:0"))
        ground = sim_utils.CuboidCfg(
            size=(6.0, 6.0, 0.10),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.18, 0.18)),
        )
        ground.func("/World/defaultGroundPlane", ground, translation=(0.0, 0.0, -0.05))
        actuators = {
            name: ImplicitActuatorCfg(
                joint_names_expr=[name],
                effort_limit_sim=float(JOINT_EFFORT_LIMIT[index]),
                velocity_limit_sim=float(JOINT_VELOCITY_LIMIT[index]),
                stiffness=float(JOINT_KP[index]),
                damping=float(JOINT_KD[index]),
            )
            for index, name in enumerate(OFFICIAL_POLICY_JOINT_ORDER)
        }
        initial_joint_pos = {
            name: float(DEFAULT_JOINT_POS[index])
            for index, name in enumerate(OFFICIAL_POLICY_JOINT_ORDER)
        }
        robot = Articulation(ArticulationCfg(
            prim_path="/World/envs/env_0/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(asset),
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    retain_accelerations=False,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    articulation_enabled=True,
                    enabled_self_collisions=True,
                    fix_root_link=False,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=4,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.80),
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos=initial_joint_pos,
                joint_vel={".*": 0.0},
            ),
            actuators=actuators,
        ))
        contact_sensor_creation_error = None
        try:
            contact_sensor = ContactSensor(ContactSensorCfg(
                prim_path="/World/envs/env_0/Robot/.*ankle_roll_link",
                update_period=0.0,
                history_length=1,
                track_air_time=True,
                force_threshold=1.0,
            ))
        except Exception as exc:
            contact_sensor = None
            contact_sensor_creation_error = f"{type(exc).__name__}: {exc}"
        stage = get_current_stage()
        report["physics_scene_paths"] = set_gravity(stage, 9.81)
        curtain_info = None
        curtain_deformable = None
        runtime_rest = None
        top_indices: list[int] = []
        if ARGS.condition == "curtain":
            from lift_curtain.envs.curtain_asset import CurtainConfig, create_surface_sheet
            from pxr import UsdPhysics
            if ARGS.scene_profile == "formal_2m":
                from lift_curtain.envs.task_scale_scene import FormalTaskSceneConfig

                formal_scene = FormalTaskSceneConfig(
                    curtain_x_m=float(ARGS.wall_x),
                    rigid_panel=FormalTaskSceneConfig().rigid_panel,
                )
                curtain_config = formal_scene.curtain_config()
                report["formal_scene_config"] = formal_scene.to_dict()
            else:
                curtain_config = CurtainConfig(
                    width_m=1.20,
                    height_m=1.55,
                    columns=25,
                    rows=25,
                    density_kg_m3=250.0,
                    surface_thickness_m=0.002,
                    stretch_stiffness=500.0,
                    shear_stiffness=250.0,
                    bend_stiffness=0.02,
                    damping=0.05,
                    solver_position_iterations=8,
                    support_z_m=1.85,
                    support_thickness_m=0.04,
                    attachment_overlap_offset_m=0.010,
                    attachment_enable_rigid_surface_attachments=True,
                    translation_m=(ARGS.wall_x, 0.0, 0.0),
                    yaw_rad=math.pi / 2.0,
                )
            curtain_info = create_surface_sheet(stage, "/World/Curtain/TaskScaleSheet", curtain_config, create_attachment=True)
            sheet_prim = stage.GetPrimAtPath(curtain_info.prim_path)
            report.update({
                "curtain_config": curtain_config.__dict__,
                "curtain_mesh_path": curtain_info.prim_path,
                "curtain_support_path": curtain_info.support_path,
                "curtain_attachment_path": curtain_info.attachment_path,
                "curtain_collision_api_applied": bool(sheet_prim.HasAPI(UsdPhysics.CollisionAPI)),
                "curtain_applied_schemas": list(sheet_prim.GetAppliedSchemas()),
            })
        elif ARGS.condition == "rigid_patch":
            if ARGS.scene_profile == "formal_2m":
                from lift_curtain.envs.task_scale_scene import FormalTaskSceneConfig

                panel_spec = FormalTaskSceneConfig().rigid_panel
                panel = sim_utils.CuboidCfg(
                    size=(panel_spec.thickness_m, panel_spec.width_m, panel_spec.height_m),
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.65, 0.20, 0.12)),
                )
                center = (float(ARGS.wall_x), panel_spec.y_m, panel_spec.bottom_z_m + panel_spec.height_m * 0.5)
                panel.func("/World/Diagnostic/FormalRigidPanel", panel, translation=center)
                report["rigid_patch"] = {
                    "path": "/World/Diagnostic/FormalRigidPanel",
                    "classification": "FORMAL_2M_FULL_HEIGHT_RIGID_PANEL_FOR_MATCHED_DIAGNOSTIC",
                    "center_m": list(center),
                    "size_m": [panel_spec.thickness_m, panel_spec.width_m, panel_spec.height_m],
                    "full_height_substitute": True,
                    "bottom_z_m": panel_spec.bottom_z_m,
                }
            else:
                panel = sim_utils.CuboidCfg(
                    # Legacy hand-height local patch retained for old replay artifacts.
                    size=(0.04, 0.50, 0.42),
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.65, 0.20, 0.12)),
                )
                panel.func("/World/Diagnostic/RigidPatch", panel, translation=(float(ARGS.wall_x), -0.17, 0.83))
                report["rigid_patch"] = {
                    "path": "/World/Diagnostic/RigidPatch",
                    "classification": "DIAGNOSTIC_ONLY_RIGID_PATCH",
                    "center_m": [float(ARGS.wall_x), -0.17, 0.83],
                    "size_m": [0.04, 0.50, 0.42],
                    "full_height_substitute": False,
                }

        sim.reset()
        robot.reset()
        robot.update(DT)
        try:
            if contact_sensor is None:
                raise RuntimeError(contact_sensor_creation_error or "contact sensor was not created")
            contact_sensor.update(DT, force_recompute=True)
            report["foot_contact_reporter"] = {
                "status": "INITIALIZED",
                "body_names": list(contact_sensor.body_names),
                "prim_path": contact_sensor.cfg.prim_path,
                "threshold_N": 1.0,
                "role": "evaluation-only telemetry; never a deployment decision input",
            }
        except Exception as exc:
            contact_sensor = None
            report["foot_contact_reporter"] = {
                "status": "UNAVAILABLE",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "proxy_policy": "ankle world pose/velocity only",
                "role": "evaluation-only telemetry; never a deployment decision input",
            }
        contract = {
            "joint_names": list(robot.joint_names),
            "body_names": list(robot.body_names),
            "is_fixed_base": bool(robot.is_fixed_base),
            "num_joints": len(robot.joint_names),
            "num_bodies": len(robot.body_names),
        }
        (run_root / "articulation_contract.json").write_text(json.dumps(clean(contract), indent=2) + "\n", encoding="utf-8")
        report["articulation_contract"] = contract
        FalconAdapter.validate_articulation(robot)
        adapter = FalconAdapter(FalconAdapterConfig(onnx_path=onnx))
        default_root_pose = robot.data.default_root_state[0, :7].detach().clone()
        joint_pos = torch.as_tensor(
            np.asarray(DEFAULT_JOINT_POS, dtype=np.float32)[list(OFFICIAL_TO_ISAACLAB)],
            device=sim.device,
            dtype=robot.data.joint_pos.dtype,
        )
        joint_vel = torch.zeros_like(joint_pos)
        report["configured_initial_joint_pos_official"] = np.asarray(DEFAULT_JOINT_POS, dtype=np.float32).tolist()
        report["configured_initial_joint_pos_isaaclab"] = joint_pos.detach().cpu().numpy().tolist()
        report["pre_write_articulation_joint_pos_isaaclab"] = robot.data.joint_pos[0].detach().cpu().numpy().tolist()
        # Build body-local collision geometry while the asset is still at its
        # authored default pose.  Isaac's USD BBoxCache does not reliably
        # reflect a later floating-base reset write, but these local corners
        # remain valid when reprojected with runtime FK poses.
        default_body_poses = body_poses(robot)
        bbox_template, body_paths = build_bbox_template(get_current_stage(), robot, default_body_poses)
        root_pose = default_root_pose.clone()
        root_pose[0] = float(ARGS.reset_root_x)
        root_pose[1] = float(ARGS.reset_root_y)
        if ARGS.reset_root_z is not None:
            root_pose[2] = float(ARGS.reset_root_z)
        # The only root pose write in this episode is this reset-time write.
        write_robot_reset(robot, sim, root_pose, joint_pos, joint_vel)
        adapter.reset()
        reset_state = adapter.make_state(robot)
        initial_root_position = reset_state.base_pos_w.copy().astype(np.float64)
        initial_yaw = yaw_from_quat(reset_state.base_quat_w)
        reset_poses = body_poses(robot)
        initial_geometry = geometry_snapshot(reset_poses, bbox_template, float(initial_root_position[0]), ARGS.wall_x)
        report.update({
            "reset_pose_write_count": 1,
            "no_runtime_root_pose_write_after_reset": True,
            "default_root_position_m": default_root_pose[:3].detach().cpu().numpy().tolist(),
            "actual_reset_root_position_m": initial_root_position.tolist(),
            "initial_root_position": initial_root_position.tolist(),
            "initial_yaw_rad": initial_yaw,
            "body_prim_path_candidates": body_paths,
            "body_geometry_template": bbox_template,
            "initial_geometry_audit": initial_geometry,
            "initial_selected_body_poses": {
                name: pose_record(reset_poses[name]) for name in (
                    "right_shoulder_pitch_link", "right_elbow_link",
                    "right_wrist_pitch_link", "right_rubber_hand",
                )
            },
        })
        (run_root / "geometry_audit.json").write_text(
            json.dumps(clean({"initial": initial_geometry, "body_geometry_template": bbox_template}), indent=2) + "\n",
            encoding="utf-8",
        )
        tracker_config = PushPathTrackerConfig(
            position_gain_xy=(0.45, 0.65),
            heading_gain=0.70,
            vx_min=0.0,
            vx_max=0.18,
            vy_limit=0.08,
            wz_limit=0.20,
            max_command_step=(0.035, 0.025, 0.035),
        )
        calibration_reference = SE2Reference((float(initial_root_position[0]), float(initial_root_position[1])), initial_yaw)
        tracker = PushPathTracker(tracker_config)
        last_target = joint_pos.unsqueeze(0).clone()
        for step in range(int(round(ARGS.settle_duration / DT))):
            state = adapter.make_state(robot)
            current_yaw = yaw_from_quat(state.base_quat_w)
            command = (
                np.zeros(3, dtype=np.float64)
                if ARGS.base_mode == "zero"
                else tracker((float(state.base_pos_w[0]), float(state.base_pos_w[1]), current_yaw), calibration_reference)
            )
            if step % DECIMATION == 0:
                target = adapter.compute_target(
                    state,
                    VelocityCommand(
                        vx=float(command[0]),
                        vy=float(command[1]),
                        wz=float(command[2]),
                        walk_mode=bool(np.linalg.norm(command) > 1.0e-8),
                    ),
                    upper_reference=np.zeros(14, dtype=np.float32),
                )
                last_target = torch.as_tensor(target, device=sim.device, dtype=robot.data.joint_pos.dtype).unsqueeze(0)
            robot.set_joint_position_target(last_target)
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(DT)

        # The reset transient can move a floating base even with zero command.
        # Re-anchor the stance reference to the measured post-settle pose before
        # starting upper-body extension, as required for blocked/probe phases.
        settled_state = adapter.make_state(robot)
        settled_yaw = yaw_from_quat(settled_state.base_quat_w)
        calibration_reference = SE2Reference(
            (float(settled_state.base_pos_w[0]), float(settled_state.base_pos_w[1])),
            settled_yaw,
        )
        report["post_settle_root_position_m"] = settled_state.base_pos_w.tolist()
        report["post_settle_root_yaw_rad"] = settled_yaw
        report["reset_settle_root_drift_m"] = float(np.linalg.norm(settled_state.base_pos_w[:2] - initial_root_position[:2]))
        tracker.reset()

        raw_initial_velocity_max = 0.0
        curtain_initial = None
        if ARGS.condition == "curtain":
            import omni.kit.app
            manager = omni.kit.app.get_app().get_extension_manager()
            manager.set_extension_enabled_immediate("isaacsim.core.experimental.prims", True)
            from isaacsim.core.experimental.prims import DeformablePrim

            curtain_deformable = DeformablePrim(curtain_info.prim_path, deformable_type="surface")
            curtain_positions, raw_velocities, runtime_rest = read_runtime(curtain_deformable)
            raw_initial_velocity_max = float(np.max(np.linalg.norm(raw_velocities, axis=1)))
            if not ARGS.preserve_curtain_velocity:
                zero_runtime_velocity(curtain_deformable, curtain_positions.shape[0])
            curtain_positions, curtain_velocities, runtime_rest = read_runtime(curtain_deformable)
            curtain_initial = curtain_positions.copy()
            top_indices = np.flatnonzero(np.abs(runtime_rest[:, 2] - np.max(runtime_rest[:, 2])) <= 1.0e-5).astype(int).tolist()
            report.update({
                "runtime_node_count": int(curtain_positions.shape[0]),
                "runtime_initial_velocity_max_mps": float(np.max(np.linalg.norm(curtain_velocities, axis=1))),
                "runtime_raw_initial_velocity_max_mps": raw_initial_velocity_max,
                "top_node_indices": top_indices,
            })

        # Freeze the measured post-settle pose as the stance reference; no
        # runtime root teleport or open-loop forward command is used afterward.
        approach_reference = SE2Reference(
            (float(settled_state.base_pos_w[0]), float(settled_state.base_pos_w[1])),
            settled_yaw,
        )
        rows: list[dict[str, Any]] = []
        hand_positions: list[np.ndarray] = []
        upper_arrays: list[np.ndarray] = []
        effort_arrays: list[np.ndarray] = []
        curtain_snapshots: list[np.ndarray] = []
        snapshot_times: list[float] = []
        phase_boundaries: dict[str, int] = {}
        policy_latencies: list[float] = []
        command_saturated_count = 0
        sim_step = 0
        previous_foot_positions: dict[str, np.ndarray] = {}
        fall_reason: str | None = None
        safety_stop_reason: str | None = None
        walkthrough_safety_stop_reason: str | None = None
        pre_probe_curtain = None
        pre_probe_hand = None
        pre_probe_rest = runtime_rest.copy() if runtime_rest is not None else None
        contact_event = False
        contact_event_time_s: float | None = None
        contact_event_reason: str | None = None
        command_rate_limited_count = 0
        target_official = joint_pos.detach().cpu().numpy().astype(np.float64)[list(ISAACLAB_TO_OFFICIAL)]

        def run_segment(
            name: str,
            duration_s: float,
            upper_fn,
            base_reference,
            *,
            stop_target_xy: tuple[float, float] | None = None,
        ) -> dict[str, Any]:
            nonlocal last_target, target_official, sim_step, fall_reason
            nonlocal safety_stop_reason, walkthrough_safety_stop_reason, contact_event, contact_event_time_s, contact_event_reason
            nonlocal command_saturated_count, command_rate_limited_count
            nonlocal contact_sensor
            phase_boundaries[name] = len(rows)
            command = np.zeros(3, dtype=np.float64)
            active_reference = base_reference(0.0, duration_s) if callable(base_reference) else base_reference
            saturated = False
            command_rate_limited = False
            target_reached = False
            final_target_error_m: float | None = None
            for local_step in range(int(round(duration_s / DT))):
                if fall_reason is not None or safety_stop_reason is not None:
                    break
                state = adapter.make_state(robot)
                current_yaw = yaw_from_quat(state.base_quat_w)
                feedback_active = bool(ARGS.base_mode != "zero" or name == "walkthrough")
                if local_step == 0 or sim_step % DECIMATION == 0:
                    active_reference = (
                        base_reference(local_step * DT, duration_s)
                        if callable(base_reference)
                        else base_reference
                    )
                    previous_command = (
                        tracker._previous_command.copy()
                        if tracker._previous_command is not None
                        else None
                    )
                    command = (
                        tracker((float(state.base_pos_w[0]), float(state.base_pos_w[1]), current_yaw), active_reference)
                        if feedback_active
                        else np.zeros(3, dtype=np.float64)
                    )
                    saturated = bool(
                        abs(float(command[0]) - tracker.config.vx_max) < 1.0e-7
                        or abs(abs(float(command[1])) - tracker.config.vy_limit) < 1.0e-7
                        or abs(abs(float(command[2])) - tracker.config.wz_limit) < 1.0e-7
                    )
                    command_rate_limited = bool(
                        previous_command is not None
                        and np.any(np.abs(command - previous_command) >= np.asarray(tracker.config.max_command_step) - 1.0e-8)
                    )
                command_saturated_count += int(saturated)
                command_rate_limited_count += int(command_rate_limited)
                upper = np.asarray(upper_fn(local_step * DT, duration_s), dtype=np.float32)
                requested_walk_mode = bool(name == "walkthrough" or np.linalg.norm(command) > 1.0e-8)
                if local_step == 0 or sim_step % DECIMATION == 0:
                    started = time.perf_counter()
                    target = adapter.compute_target(
                        state,
                        VelocityCommand(
                            vx=float(command[0]),
                            vy=float(command[1]),
                            wz=float(command[2]),
                            walk_mode=requested_walk_mode,
                        ),
                        upper_reference=upper,
                    )
                    policy_latencies.append(time.perf_counter() - started)
                    last_target = torch.as_tensor(target, device=sim.device, dtype=robot.data.joint_pos.dtype).unsqueeze(0)
                    target_official = np.asarray(target, dtype=np.float64)[list(ISAACLAB_TO_OFFICIAL)]
                robot.set_joint_position_target(last_target)
                robot.write_data_to_sim()
                sim.step(render=False)
                robot.update(DT)
                if contact_sensor is not None:
                    try:
                        contact_sensor.update(DT, force_recompute=True)
                    except Exception:
                        contact_sensor = None
                after = adapter.make_state(robot)
                poses = body_poses(robot)
                hand = poses["right_rubber_hand"][0]
                left_foot = poses["left_ankle_roll_link"][0]
                right_foot = poses["right_ankle_roll_link"][0]
                left_foot_velocity = (
                    (left_foot - previous_foot_positions["left_ankle_roll_link"]) / DT
                    if "left_ankle_roll_link" in previous_foot_positions
                    else np.zeros(3, dtype=np.float64)
                )
                right_foot_velocity = (
                    (right_foot - previous_foot_positions["right_ankle_roll_link"]) / DT
                    if "right_ankle_roll_link" in previous_foot_positions
                    else np.zeros(3, dtype=np.float64)
                )
                previous_foot_positions["left_ankle_roll_link"] = left_foot.copy()
                previous_foot_positions["right_ankle_roll_link"] = right_foot.copy()
                foot_contact = contact_reading(contact_sensor)
                geometry = geometry_snapshot(poses, bbox_template, float(after.base_pos_w[0]), ARGS.wall_x)
                hand_positions.append(hand.copy())
                hand_velocity = (
                    (hand - hand_positions[-2]) / DT
                    if len(hand_positions) >= 2
                    else np.zeros(3, dtype=np.float64)
                )
                roll, pitch = roll_pitch_from_quat(after.base_quat_w)
                finite = bool(np.isfinite(after.base_pos_w).all() and np.isfinite(after.joint_pos).all() and np.isfinite(hand).all())
                if not finite and fall_reason is None:
                    fall_reason = "NONFINITE_STATE"
                elif after.base_pos_w[2] < 0.55 and fall_reason is None:
                    fall_reason = "ROOT_HEIGHT_BELOW_0P55"
                elif (abs(roll) > 0.60 or abs(pitch) > 0.60) and fall_reason is None:
                    fall_reason = "ROOT_ROLL_PITCH_EXCEEDED_0P60"
                q_official = after.joint_pos[list(ISAACLAB_TO_OFFICIAL)]
                dq_official = after.joint_vel[list(ISAACLAB_TO_OFFICIAL)]
                effort_official = after.joint_effort[list(ISAACLAB_TO_OFFICIAL)]
                effort_ratio = float(np.max(np.abs(effort_official[15:]) / np.maximum(np.asarray([25.0, 25.0, 25.0, 25.0, 5.0, 5.0, 5.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0, 5.0]), 1.0e-6)))
                obstacle_gap = float(ARGS.wall_x - hand[0])
                palm_front_gap = float(ARGS.wall_x - (hand[0] + ARGS.palm_front_offset))
                geometry_contact = bool(abs(palm_front_gap) <= 0.05 and abs(float(hand[1])) <= 0.70 and 0.30 <= float(hand[2]) <= 1.80)
                penetration_proxy = max(0.0, float(-palm_front_gap))
                curtain_top_drift = 0.0
                curtain_contact_displacement = 0.0
                curtain_contact_y_displacement = 0.0
                curtain_speed = 0.0
                if curtain_deformable is not None and (len(rows) % max(1, ARGS.sample_every) == 0 or local_step == 0):
                    current, velocity, _ = read_runtime(curtain_deformable)
                    curtain_speed = float(np.max(np.linalg.norm(velocity, axis=1)))
                    curtain_snapshots.append(current.copy())
                    snapshot_times.append(sim_step * DT)
                    if pre_probe_curtain is not None and name in ("probe", "contact_hold", "sweep", "walkthrough", "walkthrough_stop", "retract"):
                        curtain_top_drift = float(np.max(np.linalg.norm(current[top_indices] - pre_probe_curtain[top_indices], axis=1)))
                        if pre_probe_rest is not None and pre_probe_hand is not None:
                            mask = (
                                (np.abs(pre_probe_rest[:, 0] - ARGS.wall_x) <= 0.04)
                                & (np.abs(pre_probe_rest[:, 1] - pre_probe_hand[1]) <= 0.25)
                                & (np.abs(pre_probe_rest[:, 2] - pre_probe_hand[2]) <= 0.30)
                            )
                            if np.any(mask):
                                delta = current[mask] - pre_probe_curtain[mask]
                                curtain_contact_displacement = float(np.max(np.linalg.norm(delta, axis=1)))
                                curtain_contact_y_displacement = float(np.max(np.abs(delta[:, 1])))
                if name in ("probe", "contact_hold", "sweep") and not contact_event:
                    if palm_front_gap <= ARGS.contact_event_gap:
                        contact_event = True
                        contact_event_time_s = sim_step * DT
                        contact_event_reason = "HAND_GAP_THRESHOLD"
                    elif curtain_contact_displacement >= ARGS.contact_event_displacement:
                        contact_event = True
                        contact_event_time_s = sim_step * DT
                        contact_event_reason = "CURTAIN_RUNTIME_DISPLACEMENT"
                if geometry["nonhand_clearance_m"] < 0.0 and palm_front_gap > 0.0 and safety_stop_reason is None:
                    safety_stop_reason = "NONHAND_GEOMETRY_CROSSED_WALL_BEFORE_PALM"
                if (
                    name == "walkthrough"
                    and ARGS.walkthrough_safety_clearance is not None
                    and geometry["nonhand_clearance_m"] < float(ARGS.walkthrough_safety_clearance)
                    and safety_stop_reason is None
                ):
                    # This is an external safety termination based on the
                    # robot's own collision envelope and known plane geometry.
                    # It is not a contact classifier or a passability signal.
                    walkthrough_safety_stop_reason = "EXTERNAL_SAFETY_BODY_CLEARANCE"
                    safety_stop_reason = walkthrough_safety_stop_reason
                right_shoulder = pose_record(poses["right_shoulder_pitch_link"])
                right_elbow = pose_record(poses["right_elbow_link"])
                right_wrist = pose_record(poses["right_wrist_pitch_link"])
                right_hand = pose_record(poses["right_rubber_hand"])
                actor_observation = decode_last_observation(adapter)
                rows.append({
                    "phase": name,
                    "step": sim_step,
                    "time_s": sim_step * DT,
                    "command_vx": float(command[0]),
                    "command_vy": float(command[1]),
                    "command_wz": float(command[2]),
                    "feedback_active": feedback_active,
                    "requested_walk_mode": requested_walk_mode,
                    **actor_observation,
                    "reference_x": float(active_reference.position_world[0]),
                    "reference_y": float(active_reference.position_world[1]),
                    "reference_yaw": float(active_reference.yaw),
                    "reference_vx_world": float(active_reference.velocity_world[0]),
                    "reference_vy_world": float(active_reference.velocity_world[1]),
                    "measured_vx": float(after.base_lin_vel_b[0]),
                    "measured_vy": float(after.base_lin_vel_b[1]),
                    "measured_wz": float(after.base_ang_vel_b[2]),
                    "measured_planar_speed": float(np.linalg.norm(after.base_lin_vel_b[:2].astype(np.float64))),
                    "base_x": float(after.base_pos_w[0]),
                    "base_y": float(after.base_pos_w[1]),
                    "base_z": float(after.base_pos_w[2]),
                    "yaw": float(current_yaw),
                    "roll": float(roll),
                    "pitch": float(pitch),
                    "hand_x": float(hand[0]),
                    "hand_y": float(hand[1]),
                    "hand_z": float(hand[2]),
                    # Foot pose/velocity and contact reporter values are
                    # evaluation telemetry only.  They are never included in
                    # the deployment observation CSV or controller inputs.
                    "left_ankle_roll_link_x_m": float(left_foot[0]),
                    "left_ankle_roll_link_y_m": float(left_foot[1]),
                    "left_ankle_roll_link_z_m": float(left_foot[2]),
                    "left_ankle_roll_link_vx_world_mps": float(left_foot_velocity[0]),
                    "left_ankle_roll_link_vy_world_mps": float(left_foot_velocity[1]),
                    "left_ankle_roll_link_vz_world_mps": float(left_foot_velocity[2]),
                    "right_ankle_roll_link_x_m": float(right_foot[0]),
                    "right_ankle_roll_link_y_m": float(right_foot[1]),
                    "right_ankle_roll_link_z_m": float(right_foot[2]),
                    "right_ankle_roll_link_vx_world_mps": float(right_foot_velocity[0]),
                    "right_ankle_roll_link_vy_world_mps": float(right_foot_velocity[1]),
                    "right_ankle_roll_link_vz_world_mps": float(right_foot_velocity[2]),
                    **foot_contact,
                    "right_shoulder_position_json": json.dumps(clean(right_shoulder["position_m"]), separators=(",", ":")),
                    "right_elbow_position_json": json.dumps(clean(right_elbow["position_m"]), separators=(",", ":")),
                    "right_wrist_position_json": json.dumps(clean(right_wrist["position_m"]), separators=(",", ":")),
                    "hand_quat_wxyz_json": json.dumps(clean(right_hand["orientation_wxyz"]), separators=(",", ":")),
                    "hand_local_x_world_json": json.dumps(clean(right_hand["local_x_world"]), separators=(",", ":")),
                    "hand_local_y_world_json": json.dumps(clean(right_hand["local_y_world"]), separators=(",", ":")),
                    "hand_local_z_world_json": json.dumps(clean(right_hand["local_z_world"]), separators=(",", ":")),
                    "hand_progress_from_pre_probe_m": float(hand[0] - (pre_probe_hand[0] if pre_probe_hand is not None else hand[0])),
                    "obstacle_gap_m": obstacle_gap,
                    "palm_front_gap_m": palm_front_gap,
                    "hand_bbox_gap_m": geometry["right_hand_bbox_gap_m"],
                    "nonhand_clearance_m": geometry["nonhand_clearance_m"],
                    "hand_first_clearance_margin_m": geometry["hand_first_clearance_margin_m"],
                    "min_robot_back_x_m": geometry["min_robot_back_x_m"],
                    "min_nonhand_back_x_m": geometry["min_nonhand_back_x_m"],
                    "all_robot_bodies_past_wall": geometry["all_robot_bodies_past_wall"],
                    "all_nonhand_bodies_past_wall": geometry["all_nonhand_bodies_past_wall"],
                    "geometry_contact_window": geometry_contact,
                    "hand_penetration_proxy_m": penetration_proxy,
                    "upper_command_json": json.dumps(clean(target_official[15:].tolist()), separators=(",", ":")),
                    "upper_actual_json": json.dumps(clean(q_official[15:].tolist()), separators=(",", ":")),
                    # Round 7 deployment-observation snapshot.  The formal
                    # proprioceptive replay consumes only these fields.  The
                    # geometry/deformable fields below remain evaluation-only.
                    "deployment_q_actual_json": json.dumps(clean(q_official.tolist()), separators=(",", ":")),
                    "deployment_qdot_json": json.dumps(clean(dq_official.tolist()), separators=(",", ":")),
                    "deployment_q_cmd_json": json.dumps(clean(target_official.tolist()), separators=(",", ":")),
                    "deployment_motor_effort_json": json.dumps(clean(effort_official.tolist()), separators=(",", ":")),
                    "deployment_previous_action_json": json.dumps(clean(adapter.previous_action.tolist()), separators=(",", ":")),
                    "deployment_hand_fk_position_w_json": json.dumps(clean(hand.tolist()), separators=(",", ":")),
                    "deployment_hand_fk_velocity_w_json": json.dumps(clean(hand_velocity.tolist()), separators=(",", ":")),
                    "deployment_base_position_w_json": json.dumps(clean(after.base_pos_w.tolist()), separators=(",", ":")),
                    "deployment_base_velocity_b_json": json.dumps(clean(after.base_lin_vel_b.tolist()), separators=(",", ":")),
                    "deployment_localization_assumption": "Isaac articulation pose/FK proxy; replace with deployable localization on real G1",
                    "upper_tracking_rms_rad": float(np.sqrt(np.mean(np.square(q_official[15:] - upper)))),
                    "upper_limit_margin_min_rad": float(np.min(np.minimum(q_official[15:] - JOINT_POS_LOWER[15:], JOINT_POS_UPPER[15:] - q_official[15:]))),
                    "right_shoulder_pitch_cmd_rad": float(target_official[22]),
                    "right_shoulder_pitch_actual_rad": float(q_official[22]),
                    "right_elbow_cmd_rad": float(target_official[25]),
                    "right_elbow_actual_rad": float(q_official[25]),
                    "right_wrist_pitch_cmd_rad": float(target_official[27]),
                    "right_wrist_pitch_actual_rad": float(q_official[27]),
                    "effort_max_abs": float(np.max(np.abs(effort_official))),
                    "effort_upper_max_abs": float(np.max(np.abs(effort_official[15:]))),
                    "effort_ratio": effort_ratio,
                    "effort_source": after.effort_source,
                    "base_drift_from_reference_m": float(np.linalg.norm(after.base_pos_w[:2] - np.asarray(active_reference.position_world))),
                    "curtain_top_drift_m": curtain_top_drift,
                    "curtain_contact_region_displacement_m": curtain_contact_displacement,
                    "curtain_contact_region_abs_y_displacement_m": curtain_contact_y_displacement,
                    "curtain_max_node_speed_mps": curtain_speed,
                    "command_saturated": saturated,
                    "command_rate_limited": command_rate_limited,
                    "contact_event": contact_event,
                    "safety_stop_reason": safety_stop_reason or "",
                    "finite": finite,
                    "fall": fall_reason is not None,
                    "fall_reason": fall_reason or "",
                })
                upper_arrays.append(upper.copy())
                effort_arrays.append(effort_official.copy())
                sim_step += 1
                if stop_target_xy is not None:
                    final_target_error_m = float(
                        np.linalg.norm(after.base_pos_w[:2].astype(np.float64) - np.asarray(stop_target_xy, dtype=np.float64))
                    )
                    if local_step >= int(round(0.50 / DT)) and final_target_error_m <= ARGS.walkthrough_position_tolerance:
                        target_reached = True
                        break
            return {
                "target_reached": target_reached,
                "final_target_error_m": final_target_error_m,
                "executed_steps": len(rows) - phase_boundaries[name],
                "fall_reason": fall_reason,
                "safety_stop_reason": safety_stop_reason,
            }

        run_segment(
            "precontact_extension",
            ARGS.precontact_extension_duration,
            lambda t, d: interpolate_upper(np.zeros(14, dtype=np.float32), PREPROBE_UPPER, t / max(d, 1.0e-6)),
            approach_reference,
        )
        run_segment("precontact_hold", ARGS.precontact_hold_duration, lambda _t, _d: PREPROBE_UPPER, approach_reference)
        approach_end = adapter.make_state(robot)
        approach_poses = body_poses(robot)
        approach_hand = approach_poses["right_rubber_hand"][0]
        approach_geometry = geometry_snapshot(approach_poses, bbox_template, float(approach_end.base_pos_w[0]), ARGS.wall_x)
        stance_reference = SE2Reference(
            (float(approach_end.base_pos_w[0]), float(approach_end.base_pos_w[1])),
            yaw_from_quat(approach_end.base_quat_w),
        )
        tracker.reset()
        if curtain_deformable is not None:
            pre_probe_curtain, _, _ = read_runtime(curtain_deformable)
        pre_probe_hand = approach_hand.copy()
        if runtime_rest is not None:
            pre_probe_rest = runtime_rest.copy()
            contact_mask = (
                (np.abs(pre_probe_rest[:, 0] - ARGS.wall_x) <= 0.04)
                & (np.abs(pre_probe_rest[:, 1] - pre_probe_hand[1]) <= 0.25)
                & (np.abs(pre_probe_rest[:, 2] - pre_probe_hand[2]) <= 0.30)
            )
            report["pre_probe_contact_region_node_count"] = int(np.sum(contact_mask))
        if approach_geometry["nonhand_clearance_m"] < 0.0 and safety_stop_reason is None:
            safety_stop_reason = "RESET_NEAR_FIELD_GEOMETRY_NOT_SAFE"
        report.update({
            "pre_probe_base_position_m": approach_end.base_pos_w.tolist(),
            "pre_probe_hand_position_m": approach_hand.tolist(),
            "pre_probe_geometry_audit": approach_geometry,
            "pre_probe_finite": fall_reason is None,
        })
        if safety_stop_reason is None and not ARGS.skip_probe:
            run_segment(
                "probe",
                ARGS.probe_duration,
                lambda t, d: interpolate_upper(PREPROBE_UPPER, PROBE_TARGET_UPPER, t / max(d, 1.0e-6)),
                stance_reference,
            )
            run_segment("contact_hold", ARGS.contact_hold_duration, lambda _t, _d: PROBE_TARGET_UPPER, stance_reference)
            deployment_probe_decision = None
            deployment_probe_features: dict[str, Any] | None = None
            walkthrough_allowed = True
            if ARGS.gate_walkthrough_on_proprio:
                if ARGS.proprio_baseline_csv is None or ARGS.proprio_thresholds_json is None:
                    raise ValueError(
                        "--gate-walkthrough-on-proprio requires --proprio-baseline-csv and --proprio-thresholds-json"
                    )
                baseline_path = ARGS.proprio_baseline_csv.resolve()
                thresholds_path = ARGS.proprio_thresholds_json.resolve()
                if not baseline_path.is_file() or not thresholds_path.is_file():
                    raise FileNotFoundError(f"missing deployment gate input: {baseline_path}, {thresholds_path}")
                baseline_history = load_deployment_history_csv(baseline_path)
                threshold_values = json.loads(thresholds_path.read_text(encoding="utf-8"))
                thresholds = ProprioceptiveThresholds(**threshold_values)
                active_history = [
                    deployment_observation_from_row(row)
                    for row in rows
                    if row["phase"] in ("probe", "contact_hold")
                ]
                if not active_history:
                    raise RuntimeError("deployment probe gate has no active history")
                decision, features = classify_proprioceptive_response(
                    active_history,
                    baseline=baseline_history,
                    thresholds=thresholds,
                    sample_period_s=DT,
                )
                deployment_probe_decision = decision.value
                deployment_probe_features = features
                walkthrough_allowed = deployment_probe_decision == "RESPONSIVE"
                report.update({
                    "deployment_probe_decision": deployment_probe_decision,
                    "deployment_probe_features": features,
                    "deployment_probe_gate": "CONTINUE" if walkthrough_allowed else "STOP_AND_RETRACT",
                    "deployment_probe_baseline_csv": str(baseline_path),
                    "deployment_probe_thresholds_json": str(thresholds_path),
                })
            walkthrough_result: dict[str, Any] | None = None
            walkthrough_stop_result: dict[str, Any] | None = None
            if ARGS.do_sweep and walkthrough_allowed:
                def sweep_upper(_t: float, duration: float) -> np.ndarray:
                    phase = max(0.0, min(1.0, _t / max(duration, 1.0e-6)))
                    # One bounded lateral sweep and return: neutral -> +amp -> neutral.
                    offset = float(ARGS.sweep_amplitude_rad) * math.sin(math.pi * phase)
                    target = PROBE_TARGET_UPPER.copy()
                    target[9] = offset
                    return target

                run_segment("sweep", ARGS.sweep_duration, sweep_upper, stance_reference)
            if ARGS.do_walkthrough and walkthrough_allowed:
                if ARGS.walkthrough_vx_max is not None:
                    if not math.isfinite(float(ARGS.walkthrough_vx_max)) or float(ARGS.walkthrough_vx_max) <= 0.0:
                        raise ValueError("walkthrough-vx-max must be finite and positive")
                    # Only the walkthrough tracker authority changes.  The
                    # contact/probe tracker and FALCON checkpoint remain fixed.
                    tracker = PushPathTracker(PushPathTrackerConfig(
                        position_gain_xy=(0.55, 0.90),
                        heading_gain=0.85,
                        vx_min=0.0,
                        vx_max=float(ARGS.walkthrough_vx_max),
                        vy_limit=0.15,
                        wz_limit=0.35,
                        max_command_step=(0.020, 0.020, 0.030),
                    ))
                    report["walkthrough_tracker_override"] = {
                        "position_gain_xy": [0.55, 0.90],
                        "heading_gain": 0.85,
                        "vx_max_mps": float(ARGS.walkthrough_vx_max),
                        "vy_limit_mps": 0.15,
                        "wz_limit_rps": 0.35,
                        "max_command_step": [0.020, 0.020, 0.030],
                        "reason": "Round 10 no-obstacle speed workspace and SE(2) feedforward/feedback diagnostic",
                    }
                walk_start_state = adapter.make_state(robot)
                walk_origin = walk_start_state.base_pos_w[:2].astype(np.float64).copy()
                walk_yaw = yaw_from_quat(walk_start_state.base_quat_w)
                walk_target_xy = (
                    float(walk_origin[0] + ARGS.walkthrough_distance),
                    float(walk_origin[1]),
                )
                if ARGS.walkthrough_feedforward_speed is None:
                    walk_reference = SE2Reference(walk_target_xy, walk_yaw)
                else:
                    cruise_speed = float(ARGS.walkthrough_feedforward_speed)
                    terminal_speed = float(ARGS.walkthrough_terminal_speed)
                    accel_duration = 1.0
                    accel_distance = 0.5 * cruise_speed * accel_duration
                    decel_distance = 0.20
                    if ARGS.walkthrough_distance <= accel_distance + decel_distance:
                        raise ValueError("walkthrough distance is too short for the configured bounded feedforward profile")
                    cruise_distance = ARGS.walkthrough_distance - accel_distance - decel_distance
                    cruise_duration = cruise_distance / cruise_speed
                    decel_duration = 2.0 * decel_distance / (cruise_speed + terminal_speed)
                    decel_start_time = accel_duration + cruise_duration
                    target_time = decel_start_time + decel_duration

                    def walk_reference(t: float, _duration: float) -> SE2Reference:
                        if t < accel_duration:
                            speed = cruise_speed * t / accel_duration
                            progress = 0.5 * cruise_speed * t * t / accel_duration
                        elif t < decel_start_time:
                            speed = cruise_speed
                            progress = accel_distance + cruise_speed * (t - accel_duration)
                        elif t < target_time:
                            tau = t - decel_start_time
                            acceleration = (terminal_speed - cruise_speed) / decel_duration
                            speed = cruise_speed + acceleration * tau
                            progress = accel_distance + cruise_distance + cruise_speed * tau + 0.5 * acceleration * tau * tau
                        else:
                            speed = terminal_speed
                            progress = ARGS.walkthrough_distance
                        progress = min(float(ARGS.walkthrough_distance), float(progress))
                        return SE2Reference(
                            (float(walk_origin[0] + progress), float(walk_origin[1])),
                            walk_yaw,
                            velocity_world=(float(speed), 0.0),
                        )

                    report["walkthrough_feedforward_profile"] = {
                        "cruise_speed_mps": cruise_speed,
                        "terminal_speed_mps": terminal_speed,
                        "accel_duration_s": accel_duration,
                        "cruise_duration_s": cruise_duration,
                        "decel_duration_s": decel_duration,
                        "reference_target_time_s": target_time,
                        "target_xy_m": list(walk_target_xy),
                        "stop_condition": "deployment localization target error only; simulator contact/deformation not used",
                    }
                tracker.reset()

                def walkthrough_upper(_t: float, duration: float) -> np.ndarray:
                    if ARGS.walkthrough_upper_policy == "safe_retract":
                        # Remove the additional forward-push posture after the
                        # probe while retaining the known stable pre-probe pose.
                        retract_alpha = smoothstep(min(1.0, _t / min(0.75, max(duration, 1.0e-6))))
                        return interpolate_upper(PROBE_TARGET_UPPER, PREPROBE_UPPER, retract_alpha)
                    target = PROBE_TARGET_UPPER.copy()
                    ramp = smoothstep(min(1.0, _t / min(0.75, max(duration, 1.0e-6))))
                    target[9] = float(ARGS.walkthrough_hand_y_offset_rad) * ramp
                    return target

                walkthrough_result = run_segment(
                    "walkthrough",
                    ARGS.walkthrough_duration,
                    walkthrough_upper,
                    walk_reference,
                    stop_target_xy=walk_target_xy,
                )
                if walkthrough_safety_stop_reason is not None:
                    report["walkthrough_external_safety_stop_reason"] = walkthrough_safety_stop_reason
                    # The external stop applies to forward passage only; the
                    # pre-authorized retract remains available and is not a
                    # second attempt or a simulator-truth decision.
                    safety_stop_reason = None
                walk_stop_state = adapter.make_state(robot)
                walk_stop_reference = SE2Reference(
                    (float(walk_stop_state.base_pos_w[0]), float(walk_stop_state.base_pos_w[1])),
                    yaw_from_quat(walk_stop_state.base_quat_w),
                )
                tracker.reset()
                final_walk_upper = walkthrough_upper(ARGS.walkthrough_duration, ARGS.walkthrough_duration)
                walkthrough_stop_result = run_segment(
                    "walkthrough_stop",
                    ARGS.walkthrough_stop_duration,
                    lambda _t, _d: final_walk_upper,
                    walk_stop_reference,
                )
                stance_reference = walk_stop_reference
                report["walkthrough_motion_result"] = walkthrough_result
                report["walkthrough_stop_result"] = walkthrough_stop_result
            elif ARGS.do_walkthrough and not walkthrough_allowed:
                report["walkthrough_skipped_by_deployment_gate"] = True
            run_segment(
                "retract",
                ARGS.retract_duration,
                lambda t, d: interpolate_upper(PROBE_TARGET_UPPER, PREPROBE_UPPER, t / max(d, 1.0e-6)),
                stance_reference,
            )
            run_segment("post_retract_hold", ARGS.post_retract_hold_duration, lambda _t, _d: PREPROBE_UPPER, stance_reference)

        telemetry_path = run_root / "telemetry.csv"
        with telemetry_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        deployment_fields = [
            "phase", "step", "time_s", "command_vx", "command_vy", "command_wz",
            "requested_walk_mode", "actor_obs_command_vx", "actor_obs_command_vy",
            "actor_obs_command_wz", "actor_obs_command_stand", "measured_vx", "measured_vy",
            "measured_wz", "measured_planar_speed", "reference_x", "reference_y",
            "reference_yaw", "reference_vx_world", "reference_vy_world",
            "deployment_q_actual_json", "deployment_qdot_json",
            "deployment_q_cmd_json", "deployment_motor_effort_json",
            "deployment_previous_action_json", "deployment_hand_fk_position_w_json",
            "deployment_hand_fk_velocity_w_json", "deployment_base_position_w_json",
            "deployment_base_velocity_b_json", "deployment_localization_assumption",
            "upper_tracking_rms_rad", "base_drift_from_reference_m", "finite", "fall",
        ]
        deployment_path = run_root / "deployment_observation.csv"
        with deployment_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=deployment_fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in deployment_fields})
        evaluation_truth = {
            "condition_label_for_offline_evaluation_only": ARGS.condition,
            "simulator_geometry_contact_window": [row["geometry_contact_window"] for row in rows],
            "simulator_palm_front_gap_m": [row["palm_front_gap_m"] for row in rows],
            "simulator_curtain_contact_region_displacement_m": [row["curtain_contact_region_displacement_m"] for row in rows],
            "simulator_curtain_top_drift_m": [row["curtain_top_drift_m"] for row in rows],
            "simulator_hand_penetration_proxy_m": [row["hand_penetration_proxy_m"] for row in rows],
            "simulator_nonhand_clearance_m": [row["nonhand_clearance_m"] for row in rows],
            "role": "offline evaluation and external safety diagnostics only; never a Round 7 deployment decision input",
        }
        (run_root / "evaluation_ground_truth.json").write_text(
            json.dumps(clean(evaluation_truth), indent=2) + "\n", encoding="utf-8"
        )
        np.savez_compressed(
            run_root / "matched_arrays.npz",
            upper_commands=np.asarray(upper_arrays),
            effort_official=np.asarray(effort_arrays),
            hand_positions=np.asarray(hand_positions),
            curtain_snapshot_times=np.asarray(snapshot_times),
            curtain_snapshots=np.asarray(curtain_snapshots),
            top_indices=np.asarray(top_indices, dtype=np.int64),
            runtime_rest_positions=np.asarray(runtime_rest) if runtime_rest is not None else np.empty((0, 3)),
            pre_probe_rest_positions=np.asarray(pre_probe_rest) if pre_probe_rest is not None else np.empty((0, 3)),
        )
        plot_path = save_plot(run_root, rows, ARGS.wall_x)
        probe_rows = [row for row in rows if row["phase"] in ("probe", "contact_hold")]
        sweep_rows = [row for row in rows if row["phase"] == "sweep"]
        walkthrough_motion_rows = [row for row in rows if row["phase"] == "walkthrough"]
        walkthrough_stop_rows = [row for row in rows if row["phase"] == "walkthrough_stop"]
        walkthrough_rows = walkthrough_motion_rows + walkthrough_stop_rows
        stationary_rows = [row for row in rows if row["phase"] in ("probe", "contact_hold", "sweep", "walkthrough_stop", "retract", "post_retract_hold")]
        finite = bool(rows) and all(bool(row["finite"]) for row in rows)
        nominal_hand_window_count = sum(1 for row in probe_rows if row["geometry_contact_window"])
        geometry_contact_count = nominal_hand_window_count if ARGS.condition != "none" else 0
        min_palm_front_gap = min(float(row["palm_front_gap_m"]) for row in rows) if rows else None
        max_contact_deformation = max((float(row["curtain_contact_region_displacement_m"]) for row in probe_rows), default=0.0)
        max_top_drift = max((float(row["curtain_top_drift_m"]) for row in stationary_rows), default=0.0)
        max_nonhand_overlap = max((max(0.0, -float(row["nonhand_clearance_m"])) for row in rows), default=0.0)
        # Contact validity is a near-field probe gate.  During a permitted
        # flexible-curtain walkthrough the body may cross the curtain plane;
        # that later evaluation proxy must not retroactively invalidate the
        # hand-first contact that was established during probe/contact_hold.
        max_probe_nonhand_overlap = max(
            (max(0.0, -float(row["nonhand_clearance_m"])) for row in probe_rows),
            default=0.0,
        )
        minimum_contact_deformation_m = 0.002
        valid_curtain_contact = bool(
            ARGS.condition == "curtain"
            and contact_event
            and nominal_hand_window_count > 0
            and max_contact_deformation >= minimum_contact_deformation_m
            and max_top_drift <= TOP_DRIFT_TOLERANCE_M
            and max_probe_nonhand_overlap <= 0.0
        )
        if ARGS.condition == "rigid_patch":
            if ARGS.scene_profile == "formal_2m":
                diagnostic_classification = (
                    "FORMAL_2M_FULL_HEIGHT_RIGID_PANEL_HAND_FIRST_RESPONSE_WINDOW"
                    if geometry_contact_count > 0 else
                    "FORMAL_2M_FULL_HEIGHT_RIGID_PANEL_NO_HAND_WINDOW"
                )
            else:
                diagnostic_classification = (
                    "DIAGNOSTIC_ONLY_RIGID_PATCH_HAND_FIRST_RESPONSE_WINDOW"
                    if geometry_contact_count > 0 else
                    "DIAGNOSTIC_ONLY_RIGID_PATCH_NO_HAND_WINDOW"
                )
        elif ARGS.condition == "curtain":
            diagnostic_classification = "CURTAIN_HAND_FIRST_DIAGNOSTIC_RUN"
        else:
            diagnostic_classification = "NO_OBSTACLE_CONTROL"
        if ARGS.condition == "curtain":
            stage_status = "PASS" if finite and fall_reason is None and valid_curtain_contact else "BLOCKED" if finite and fall_reason is None else "FAIL"
            contact_result = "VALID_HAND_CURTAIN_CONTACT" if valid_curtain_contact else "NO_VALID_HAND_CURTAIN_CONTACT"
        elif ARGS.condition == "none":
            stage_status = "PASS" if finite and fall_reason is None else "FAIL"
            contact_result = "NO_OBSTACLE_CONTROL"
        else:
            stage_status = "PASS" if finite and fall_reason is None else "FAIL"
            contact_result = "RIGID_PATCH_HAND_WINDOW" if contact_event else "NO_VALID_RIGID_HAND_CONTACT"
        walkthrough_finite_no_fall = bool(walkthrough_rows) and all(
            str(row["finite"]).lower() == "true" and str(row["fall"]).lower() == "false"
            for row in walkthrough_rows
        )
        stop_dwell_steps = int(math.ceil(ARGS.walkthrough_stop_dwell_duration / DT))
        walkthrough_stop_speed_dwell_pass = bool(
            len(walkthrough_stop_rows) >= stop_dwell_steps
            and all(
                float(row["measured_planar_speed"]) <= ARGS.walkthrough_stop_speed_threshold
                for row in walkthrough_stop_rows[-stop_dwell_steps:]
            )
        )
        walkthrough_actor_walk_flag_ok = bool(walkthrough_motion_rows) and all(
            abs(float(row["actor_obs_command_stand"]) - 1.0) <= 1.0e-6
            for row in walkthrough_motion_rows
        )
        walkthrough_actor_stand_flag_ok = bool(walkthrough_stop_rows) and all(
            abs(float(row["actor_obs_command_stand"])) <= 1.0e-6
            for row in walkthrough_stop_rows
        )
        walkthrough_target_reached = bool(report.get("walkthrough_motion_result", {}).get("target_reached", False))
        walkthrough_pass = bool(
            ARGS.do_walkthrough
            and walkthrough_finite_no_fall
            and walkthrough_target_reached
            and walkthrough_stop_speed_dwell_pass
            and walkthrough_actor_walk_flag_ok
            and walkthrough_actor_stand_flag_ok
            and str(walkthrough_rows[-1]["all_robot_bodies_past_wall"]).lower() == "true"
            and max((float(row["curtain_top_drift_m"]) for row in walkthrough_rows), default=0.0) <= TOP_DRIFT_TOLERANCE_M
        )
        report.update({
            "status": stage_status,
            "simulation_status": "COMPLETED",
            "finite_no_fall": bool(finite and fall_reason is None),
            "contact_result": contact_result,
            "m2_1_gate": "PASS" if valid_curtain_contact else "BLOCKED" if ARGS.condition == "curtain" else "NOT_APPLICABLE",
            "contact_event": contact_event,
            "contact_event_time_s": contact_event_time_s,
            "contact_event_reason": contact_event_reason,
            "safety_stop_reason": safety_stop_reason,
            "phase_boundaries": phase_boundaries,
            "nominal_hand_window_count": nominal_hand_window_count,
            "geometry_contact_window_count": geometry_contact_count,
            "max_probe_effort_abs": max(float(row["effort_max_abs"]) for row in probe_rows) if probe_rows else 0.0,
            "max_probe_upper_effort_abs": max(float(row["effort_upper_max_abs"]) for row in probe_rows) if probe_rows else 0.0,
            "max_probe_effort_ratio": max(float(row["effort_ratio"]) for row in probe_rows) if probe_rows else 0.0,
            "max_hand_x_m": max(float(row["hand_x"]) for row in rows) if rows else None,
            "min_obstacle_gap_m": min(float(row["obstacle_gap_m"]) for row in rows) if rows else None,
            "min_palm_front_gap_m": min_palm_front_gap,
            "max_hand_penetration_proxy_m": max(float(row["hand_penetration_proxy_m"]) for row in rows) if rows else None,
            "max_probe_nonhand_overlap_m": max_probe_nonhand_overlap,
            "max_curtain_top_drift_m": max(float(row["curtain_top_drift_m"]) for row in stationary_rows) if stationary_rows else 0.0,
            "max_curtain_contact_region_displacement_m": max(float(row["curtain_contact_region_displacement_m"]) for row in probe_rows) if probe_rows else 0.0,
            "max_curtain_contact_region_abs_y_displacement_m": max(float(row["curtain_contact_region_abs_y_displacement_m"]) for row in probe_rows) if probe_rows else 0.0,
            "max_contact_region_displacement_m": max_contact_deformation,
            "sweep_hand_y_range_m": (max(float(row["hand_y"]) for row in sweep_rows) - min(float(row["hand_y"]) for row in sweep_rows)) if sweep_rows else 0.0,
            "sweep_hand_x_range_m": (max(float(row["hand_x"]) for row in sweep_rows) - min(float(row["hand_x"]) for row in sweep_rows)) if sweep_rows else 0.0,
            "sweep_curtain_contact_region_displacement_m": max((float(row["curtain_contact_region_displacement_m"]) for row in sweep_rows), default=0.0),
            "sweep_base_drift_m": max((float(row["base_drift_from_reference_m"]) for row in sweep_rows), default=0.0),
            "sweep_upper_effort_max_abs": max((float(row["effort_upper_max_abs"]) for row in sweep_rows), default=0.0),
            "walkthrough_base_x_start_m": float(walkthrough_rows[0]["base_x"]) if walkthrough_rows else None,
            "walkthrough_base_x_end_m": float(walkthrough_rows[-1]["base_x"]) if walkthrough_rows else None,
            "walkthrough_base_x_progress_m": (float(walkthrough_rows[-1]["base_x"]) - float(walkthrough_rows[0]["base_x"])) if walkthrough_rows else 0.0,
            "walkthrough_command_vx_mean_mps": float(np.mean([float(row["command_vx"]) for row in walkthrough_motion_rows])) if walkthrough_motion_rows else 0.0,
            "walkthrough_measured_vx_mean_mps": float(np.mean([float(row["measured_vx"]) for row in walkthrough_motion_rows])) if walkthrough_motion_rows else 0.0,
            "walkthrough_max_base_drift_from_target_m": max((float(row["base_drift_from_reference_m"]) for row in walkthrough_rows), default=0.0),
            "walkthrough_max_hand_y_range_m": (max(float(row["hand_y"]) for row in walkthrough_rows) - min(float(row["hand_y"]) for row in walkthrough_rows)) if walkthrough_rows else 0.0,
            "walkthrough_max_curtain_contact_region_displacement_m": max((float(row["curtain_contact_region_displacement_m"]) for row in walkthrough_rows), default=0.0),
            "walkthrough_all_nonhand_bodies_past_wall": any(str(row["all_nonhand_bodies_past_wall"]).lower() == "true" for row in walkthrough_rows),
            "walkthrough_all_robot_bodies_past_wall": any(str(row["all_robot_bodies_past_wall"]).lower() == "true" for row in walkthrough_rows),
            "walkthrough_final_all_nonhand_bodies_past_wall": bool(walkthrough_rows) and str(walkthrough_rows[-1]["all_nonhand_bodies_past_wall"]).lower() == "true",
            "walkthrough_final_all_robot_bodies_past_wall": bool(walkthrough_rows) and str(walkthrough_rows[-1]["all_robot_bodies_past_wall"]).lower() == "true",
            "walkthrough_finite_no_fall": walkthrough_finite_no_fall,
            "walkthrough_target_reached": walkthrough_target_reached,
            "walkthrough_final_target_error_m": report.get("walkthrough_motion_result", {}).get("final_target_error_m"),
            "walkthrough_actor_walk_flag_ok": walkthrough_actor_walk_flag_ok,
            "walkthrough_actor_stand_flag_ok": walkthrough_actor_stand_flag_ok,
            "walkthrough_stop_speed_dwell_pass": walkthrough_stop_speed_dwell_pass,
            "walkthrough_stop_final_speed_mps": float(walkthrough_stop_rows[-1]["measured_planar_speed"]) if walkthrough_stop_rows else None,
            "walkthrough_stop_drift_m": (
                float(np.linalg.norm(
                    np.asarray([walkthrough_stop_rows[-1]["base_x"], walkthrough_stop_rows[-1]["base_y"]], dtype=np.float64)
                    - np.asarray([walkthrough_stop_rows[0]["base_x"], walkthrough_stop_rows[0]["base_y"]], dtype=np.float64)
                )) if walkthrough_stop_rows else None
            ),
            "walkthrough_status": "PASS_DIAGNOSTIC" if walkthrough_pass else "FAIL_DIAGNOSTIC" if ARGS.do_walkthrough else "NOT_RUN",
            "max_top_drift_m": max_top_drift,
            "max_nonhand_overlap_m": max_nonhand_overlap,
            "max_stationary_base_drift_m": max(float(row["base_drift_from_reference_m"]) for row in stationary_rows) if stationary_rows else None,
            "max_upper_tracking_rms_rad": max(float(row["upper_tracking_rms_rad"]) for row in rows) if rows else None,
            "max_hand_speed_mps": float(np.max(np.linalg.norm(np.diff(np.asarray(hand_positions), axis=0), axis=1) / DT)) if len(hand_positions) > 1 else 0.0,
            "command_saturation_ratio": command_saturated_count / max(1, len(rows)),
            "command_rate_limited_ratio": command_rate_limited_count / max(1, len(rows)),
            "policy_update_count": len(policy_latencies),
            "policy_update_frequency_sim_hz": len(policy_latencies) / max(len(rows) * DT, 1.0e-9),
            "policy_latency_p50_ms": float(np.percentile(np.asarray(policy_latencies) * 1000.0, 50)) if policy_latencies else None,
            "policy_latency_p95_ms": float(np.percentile(np.asarray(policy_latencies) * 1000.0, 95)) if policy_latencies else None,
            "telemetry_csv": str(telemetry_path),
            "array_artifact": str(run_root / "matched_arrays.npz"),
            "visualization_evidence": [plot_path],
            "diagnostic_classification": diagnostic_classification,
            "m2_1_status_current": "PASS_DIAGNOSTIC" if valid_curtain_contact else "BLOCKED" if ARGS.condition == "curtain" else "NOT_APPLICABLE",
            "legacy_round6_m2_1_status_before_this_run": "BLOCKED",
            "notes": [
                (
                    f"This is a near-field controlled-passage diagnostic with scene_profile={ARGS.scene_profile}; "
                    "it is not a remote approach or full map-integrated episode."
                    if ARGS.do_walkthrough
                    else f"This is a diagnostic-only near-field contact scene with scene_profile={ARGS.scene_profile}; it is not a traversal result."
                ),
                "The root pose is written once at reset; runtime base and curtain node teleport are prohibited.",
                "base_mode=zero isolates hand physics from the Round 6 C feedback failure; it is not an approach controller result.",
                "The legacy rigid condition remains a hand-height local patch; the formal_2m profile uses a full-height 2.0 m panel for a matched task-scale diagnostic.",
                "No camera, RGB-D, global map, tactile sensor, learning, or node-position teleport is used.",
                "For curtain, M2.1 PASS requires a hand-window event plus nonzero runtime curtain deformation and stable top nodes; finite/no-fall alone is insufficient.",
                "This run records whether runtime deformable velocities were preserved and whether the active probe was omitted; these are diagnostic controls, not task behavior changes.",
            ],
        })
        (run_root / "summary.json").write_text(json.dumps(clean(report), indent=2) + "\n", encoding="utf-8")
        print(json.dumps(clean(report), indent=2))
        return 0 if report["status"] == "PASS" else 2
    except Exception as exc:
        report.update({"status": "BLOCKED", "simulation_status": "NOT COMPLETED", "error_type": type(exc).__name__, "error": str(exc)})
        (run_root / "summary.json").write_text(json.dumps(clean(report), indent=2) + "\n", encoding="utf-8")
        print(json.dumps(clean(report), indent=2))
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
