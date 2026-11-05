"""Round4070 problem-discovery PointGoal physical episode executor.

The high-level state machine is shared by the formal fixed-top curtain and the
matched fixed rigid panel. Scene kind is used only by the scene builder and
offline evaluation writer. Probe classification, memory, planning and
locomotion consume deployment observations, the known 0/1/2 map and the
permitted pose-localization assumption; simulator truth is isolated in an
evaluation-only artifact and an external safety monitor.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, is_dataclass, replace
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

DEFAULT_ASSET = REPO / "assets" / "generated" / "g1_29dof_falcon_rubberhand.usd"
DEFAULT_ONNX = Path(r"C:\Robotics\FALCON-upstream\sim2real\models\falcon\g1_29dof.onnx")
DEFAULT_RUN_ROOT = REPO / "outputs" / "round4070_fixed10cm_judgment" / "trials"
TARGET_DISPLACEMENT_M = 0.10
PROBE_DIRECTION_WORLD = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
DEFAULT_PROBE_10CM_JOINT_STEP_RAD = 0.70
DT = 0.005
DECIMATION = 4
CONTROL_DT = DT * DECIMATION
EVALUATION_DECIMATION = 20
EVALUATION_DT = DT * EVALUATION_DECIMATION
PREPROBE_UPPER = np.asarray(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.45, 0.0, 0.0, 0.45, 0.0, 0.0, 0.0],
    dtype=np.float32,
)
PROBE_TARGET_UPPER = np.asarray(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.55, 0.0, 0.0, 0.55, 0.0, 0.0, 0.0],
    dtype=np.float32,
)
DEFAULT_PROPRIO_BASELINE = REPO / "reports" / "artifacts" / "round8" / "formal_matched_proprio" / "reset_x045" / "none" / "deployment_observation.csv"
DEFAULT_PROPRIO_THRESHOLDS = REPO / "configs" / "proprioceptive_thresholds.json"

from lift_curtain.experiments.stage_log import StageLogger

pre_parser = argparse.ArgumentParser(add_help=False)
pre_parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
PRE_ARGS, _ = pre_parser.parse_known_args()
LOGGER = StageLogger(
    PRE_ARGS.run_root,
    phase_timeouts_s={
        "BEFORE_APP": 180.0,
        "AFTER_APP": 240.0,
        "AFTER_SCENE": 180.0,
        "AFTER_RESET": 180.0,
        "FIRST_PHYSICS_STEP": 180.0,
    },
)
LOGGER.mark("BEFORE_APP", details={"argv": sys.argv})

from isaaclab.app import AppLauncher
from lift_curtain.envs.round13_scenarios import scenario_by_name

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--scenario",
    choices=(
        "nominal",
        "half_density",
        "double_density",
        "bottom_left_anchor",
        "bottom_right_anchor",
        "both_bottom_corners",
        "curtain_hidden_rigid",
        "rigid_panel",
    ),
    default="rigid_panel",
)
parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
parser.add_argument("--settle-duration", type=float, default=0.50)
parser.add_argument("--freeze-duration", type=float, default=0.60)
parser.add_argument("--segment-timeout", type=float, default=7.0)
parser.add_argument(
    "--max-bounded-advance-segments",
    type=int,
    default=24,
    help="Finite number of deployment-monitored cautious-advance segments before clearance evaluation.",
)
parser.add_argument("--position-tolerance", type=float, default=0.08)
parser.add_argument("--cruise-speed", type=float, default=0.25)
parser.add_argument("--vx-max", type=float, default=0.30)
parser.add_argument("--stand-duration", type=float, default=1.50)
parser.add_argument("--stop-dwell-duration", type=float, default=0.30)
parser.add_argument("--reset-root-x", type=float, default=0.0, help="Episode-reset root x; runtime root writes remain forbidden after reset.")
parser.add_argument("--reset-root-y", type=float, default=0.0, help="Episode-reset root y; runtime root writes remain forbidden after reset.")
parser.add_argument("--reset-root-yaw", type=float, default=0.0, help="Episode-reset root yaw in radians; runtime root writes remain forbidden after reset.")
parser.add_argument("--precontact-offset", type=float, default=0.18, help="Map planner offset from candidate; use only a geometry-derived value.")
parser.add_argument("--precontact-region-x-min", type=float, default=None, help="Optional geometry-derived root x lower bound for region acceptance.")
parser.add_argument("--precontact-region-x-max", type=float, default=None, help="Optional geometry-derived root x upper bound for region acceptance.")
parser.add_argument("--precontact-region-y-half-width", type=float, default=None, help="Optional geometry-derived root y half-width for region acceptance.")
parser.add_argument("--precontact-region-speed-max", type=float, default=0.08, help="Maximum measured planar base speed for region acceptance.")
parser.add_argument("--reset-policy-at-precontact", action="store_true", help="Reset FALCON history once when entering the pre-contact stop/re-anchor mode.")
parser.add_argument("--physical-rigid", action="store_true", help=argparse.SUPPRESS)
parser.add_argument("--proprio-baseline-csv", type=Path, default=DEFAULT_PROPRIO_BASELINE)
parser.add_argument("--proprio-thresholds-json", type=Path, default=DEFAULT_PROPRIO_THRESHOLDS)
parser.add_argument("--preprobe-duration", type=float, default=1.50)
parser.add_argument("--preprobe-hold-duration", type=float, default=0.40)
parser.add_argument("--probe-duration", type=float, default=1.50)
parser.add_argument("--probe-hold-duration", type=float, default=0.50)
parser.add_argument("--retract-duration", type=float, default=1.50)
parser.add_argument("--neutralize-duration", type=float, default=1.50)
parser.add_argument("--traverse-exit-x", type=float, default=1.30, help="Independent candidate-clear exit target; not the final goal.")
parser.add_argument("--traverse-exit-y", type=float, default=0.0, help="Independent candidate-clear exit target; not the final goal.")
parser.add_argument("--candidate-clear-margin", type=float, default=0.04, help="Robot-envelope margin beyond the candidate normal extent.")
parser.add_argument("--goal-yaw-tolerance", type=float, default=0.15, help="Diagnostic-only final yaw tolerance; never gates PointGoal success.")
parser.add_argument(
    "--hidden-rigid-x",
    type=float,
    default=None,
    help="Round 14 H2 diagnostic only: override hidden rigid panel center x; default preserves Round 13 geometry.",
)
parser.add_argument("--hidden-rigid-y", type=float, default=None)
parser.add_argument("--hidden-rigid-width", type=float, default=None)
parser.add_argument("--hidden-rigid-height", type=float, default=None)
parser.add_argument("--hidden-rigid-thickness", type=float, default=None)
parser.add_argument("--hidden-rigid-bottom-z", type=float, default=None)
parser.add_argument(
    "--curtain-length",
    type=float,
    default=2.00,
    help="Failure-discovery override; the bottom remains at 0.02 m when the top is derived.",
)
parser.add_argument(
    "--curtain-top-z",
    type=float,
    default=None,
    help="Optional explicit top z; must keep the bottom at 0.02 m.",
)
parser.add_argument("--density-scale-multiplier", type=float, default=1.0)
parser.add_argument("--curtain-damping", type=float, default=None)
parser.add_argument("--curtain-bend-stiffness", type=float, default=None)
parser.add_argument("--curtain-stretch-stiffness", type=float, default=None)
parser.add_argument("--curtain-shear-stiffness", type=float, default=None)
parser.add_argument("--condition-id", type=str, default=None, help="Offline manifest label; never read by deployment logic.")
parser.add_argument(
    "--engineering-fix-accept-goal-reached",
    action="store_true",
    help="ENGINEERING_FIX_EXPERIMENT only: accept planner GOAL_REACHED after responsive release; frozen baseline keeps this off.",
)
parser.add_argument(
    "--research-feasibility-post-clearance-replan",
    action="store_true",
    help="Independent research branch: after deployment-owned candidate clearance, permit a guarded goal-side replan without changing the frozen baseline.",
)
parser.add_argument(
    "--round15-progress-monitor",
    action="store_true",
    help="Enable the Round15 phase-aware deployment-only progress monitor during responsive traverse.",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--probe-side", choices=("left", "right"), default="right", help="Round21 mirrored active hand; deployment branch remains side-agnostic.")
parser.add_argument(
    "--probe-depth-level",
    choices=("PROBE_10CM",),
    default="PROBE_10CM",
    help="Round 4070-D formal probe contract. D0-D3 schedules are rejected.",
)
parser.add_argument("--probe-depth-joint-step-rad", type=float, default=0.15)
parser.add_argument(
    "--probe-10cm-joint-step-rad",
    type=float,
    default=DEFAULT_PROBE_10CM_JOINT_STEP_RAD,
    help="Frozen joint-reference increment calibrated for the 0.10 m target; FK is authoritative for actual displacement.",
)
parser.add_argument(
    "--probe-target-upper-json",
    type=Path,
    default=None,
    help="Calibration-only 14-vector upper-body target override; requires --probe-validation-only.",
)
parser.add_argument(
    "--structural-topology",
    choices=("A0_TOP_EDGE_ONLY", "A1_TOP_EDGE_BOTTOM_TWO_CORNERS", "A2_FOUR_CORNERS_ONLY", "A3_FULL_PERIMETER_FIXED"),
    default="A0_TOP_EDGE_ONLY",
)
parser.add_argument("--surface-thickness", type=float, default=0.002)
parser.add_argument("--opening-diagnostic", action="store_true", help="Compatibility alias for --opening-mode hold.")
parser.add_argument("--opening-mode", choices=("none", "hold"), default="none", help="Round22 matched action: none or bounded lateral hold.")
parser.add_argument("--diagnostic-no-obstacle", action="store_true", help="Skip physical candidate construction for the matched empty-scene action control.")
parser.add_argument("--probe-validation-only", action="store_true", help="Run the fixed probe, safety monitors and retract only; do not enter post-probe navigation or benchmark labeling.")
parser.add_argument("--opening-lateral-amplitude", type=float, default=0.15, help="Bounded shoulder-yaw reference amplitude for the opening diagnostic.")
parser.add_argument("--opening-direction-sign", type=float, choices=(-1.0, 1.0), default=1.0, help="Fixed action direction used by the diagnostic; never selected from simulator truth.")
parser.add_argument("--opening-sweep-duration", type=float, default=1.0)
parser.add_argument("--opening-hold-duration", type=float, default=0.8)
parser.add_argument(
    "--high-level-policy",
    choices=("r1", "r2", "a0", "a1", "b0", "b1", "b2", "d1"),
    default="r1",
    help="Round27 option policy; all actions pass through the independent safety supervisor.",
)
parser.add_argument(
    "--policy-checkpoint",
    type=Path,
    default=None,
    help="Optional trusted local Round25 A/B checkpoint; omitted uses the frozen seed-0 artifact.",
)
parser.add_argument("--policy-candidate-route-cost", type=float, default=1.0)
parser.add_argument("--policy-zero-route-cost", type=float, default=2.0)
AppLauncher.add_app_launcher_args(parser)
ARGS = parser.parse_args()
OPENING_ENABLED = bool(ARGS.opening_diagnostic or ARGS.opening_mode != "none")

def make_side_upper(probe_side: str, shoulder_pitch: float, elbow: float) -> np.ndarray:
    result = np.zeros(14, dtype=np.float32)
    offset = 0 if probe_side == "left" else 7
    result[offset] = float(shoulder_pitch)
    result[offset + 3] = float(elbow)
    return result

ACTIVE_HAND_NAME = f"{ARGS.probe_side}_rubber_hand"
PREPROBE_UPPER = make_side_upper(ARGS.probe_side, -0.45, 0.45)
_depth_step = float(ARGS.probe_10cm_joint_step_rad)
PROBE_TARGET_UPPER = make_side_upper(ARGS.probe_side, -0.55 - _depth_step, 0.55 + _depth_step)
PROBE_TARGET_REFERENCE_PROFILE = "SYMMETRIC_SHOULDER_ELBOW"
PROBE_TARGET_REFERENCE_SOURCE = None
if ARGS.probe_target_upper_json is not None:
    if not ARGS.probe_validation_only:
        raise ValueError("--probe-target-upper-json is calibration-only and requires --probe-validation-only")
    _target_path = ARGS.probe_target_upper_json.expanduser().resolve()
    if not _target_path.is_file():
        raise FileNotFoundError(f"probe target JSON does not exist: {_target_path}")
    _target_payload = json.loads(_target_path.read_text(encoding="utf-8"))
    if isinstance(_target_payload, dict):
        _target_values = _target_payload.get("upper_target")
        _target_profile = _target_payload.get("profile", _target_path.stem)
        _target_side = _target_payload.get("probe_side")
    elif isinstance(_target_payload, list):
        _target_values = _target_payload
        _target_profile = _target_path.stem
        _target_side = None
    else:
        raise ValueError("probe target JSON must contain an object or a 14-element list")
    if _target_values is None:
        raise ValueError("probe target JSON must contain upper_target")
    if _target_side is not None and _target_side != ARGS.probe_side:
        raise ValueError(f"probe target side {_target_side!r} does not match --probe-side {ARGS.probe_side!r}")
    if not isinstance(_target_profile, str) or not _target_profile.strip():
        raise ValueError("probe target profile must be a non-empty string")
    _target_array = np.asarray(_target_values, dtype=np.float32)
    if _target_array.shape != (14,) or not np.all(np.isfinite(_target_array)):
        raise ValueError("probe target upper must be a finite 14-element vector")
    _inactive_slice = slice(7, 14) if ARGS.probe_side == "left" else slice(0, 7)
    if np.max(np.abs(_target_array[_inactive_slice])) > 1.0e-6:
        raise ValueError("probe target upper must keep the inactive arm at zero")
    PROBE_TARGET_UPPER = _target_array
    PROBE_TARGET_REFERENCE_PROFILE = _target_profile.strip()
    PROBE_TARGET_REFERENCE_SOURCE = str(_target_path)
PROBE_TARGET_POSITION_ROLE = (
    "calibration-only custom full-arm FK target; hand FK is authoritative for actual displacement"
    if ARGS.probe_target_upper_json is not None
    else "fixed calibrated joint reference; hand FK is authoritative for actual displacement"
)
PROBE_TARGET_POSITION_MEASUREMENT_METHOD = (
    "fixed 0.10 m contract with custom full-arm FK target; actual hand FK is authoritative"
    if ARGS.probe_target_upper_json is not None
    else "fixed 0.10 m contract with calibrated joint reference; actual hand FK is authoritative"
)
if not math.isfinite(float(ARGS.opening_lateral_amplitude)) or not 0.0 < float(ARGS.opening_lateral_amplitude) <= 0.30:
    raise ValueError("opening-lateral-amplitude must be finite and in (0, 0.30]")
if int(ARGS.max_bounded_advance_segments) <= 0:
    raise ValueError("max-bounded-advance-segments must be positive")
if not math.isfinite(float(ARGS.opening_sweep_duration)) or not math.isfinite(float(ARGS.opening_hold_duration)) or ARGS.opening_sweep_duration <= 0.0 or ARGS.opening_hold_duration <= 0.0:
    raise ValueError("opening durations must be finite and positive")
if not math.isfinite(float(ARGS.curtain_length)) or ARGS.curtain_length <= 0.0:
    raise ValueError("curtain-length must be finite and positive")
CURTAIN_TOP_Z_M = float(ARGS.curtain_top_z) if ARGS.curtain_top_z is not None else float(ARGS.curtain_length) + 0.02
if not math.isfinite(CURTAIN_TOP_Z_M) or abs(CURTAIN_TOP_Z_M - float(ARGS.curtain_length) - 0.02) > 1.0e-6:
    raise ValueError("curtain-top-z must keep the curtain bottom at 0.02 m")
if not math.isfinite(float(ARGS.density_scale_multiplier)) or ARGS.density_scale_multiplier <= 0.0:
    raise ValueError("density-scale-multiplier must be finite and positive")
if not math.isfinite(float(ARGS.probe_10cm_joint_step_rad)) or not 0.0 < float(ARGS.probe_10cm_joint_step_rad) <= 1.20:
    raise ValueError("probe-10cm-joint-step-rad must be finite and in (0, 1.20]")
if not math.isfinite(float(ARGS.surface_thickness)) or not 0.0 < float(ARGS.surface_thickness) <= 0.010:
    raise ValueError("surface-thickness must be finite and in (0, 0.010]")
for _name in (
    "hidden_rigid_y",
    "hidden_rigid_width",
    "hidden_rigid_height",
    "hidden_rigid_thickness",
    "hidden_rigid_bottom_z",
    "curtain_damping",
    "curtain_bend_stiffness",
    "curtain_stretch_stiffness",
    "curtain_shear_stiffness",
):
    _value = getattr(ARGS, _name)
    if _value is not None and (
        not math.isfinite(float(_value))
        or (_name not in {"hidden_rigid_bottom_z"} and float(_value) <= 0.0)
    ):
        raise ValueError(f"{_name} must be finite and positive when provided")
OPENING_TARGET_UPPER = PROBE_TARGET_UPPER.copy()
_opening_offset = 0 if ARGS.probe_side == "left" else 7
if OPENING_ENABLED:
    OPENING_TARGET_UPPER[_opening_offset + 2] = float(ARGS.opening_direction_sign * ARGS.opening_lateral_amplitude)
SCENARIO = scenario_by_name(ARGS.scenario)
if ARGS.hidden_rigid_x is not None:
    if SCENARIO.hidden_rigid is None:
        raise ValueError("--hidden-rigid-x requires --scenario curtain_hidden_rigid")
    if not math.isfinite(float(ARGS.hidden_rigid_x)):
        raise ValueError("--hidden-rigid-x must be finite")
    SCENARIO = replace(
        SCENARIO,
        hidden_rigid=replace(SCENARIO.hidden_rigid, x_m=float(ARGS.hidden_rigid_x)),
    )
if SCENARIO.hidden_rigid is not None:
    hidden_updates = {}
    for _field, _arg in (
        ("y_m", ARGS.hidden_rigid_y),
        ("width_m", ARGS.hidden_rigid_width),
        ("height_m", ARGS.hidden_rigid_height),
        ("thickness_m", ARGS.hidden_rigid_thickness),
        ("bottom_z_m", ARGS.hidden_rigid_bottom_z),
    ):
        if _arg is not None:
            hidden_updates[_field] = float(_arg)
    if hidden_updates:
        SCENARIO = replace(SCENARIO, hidden_rigid=replace(SCENARIO.hidden_rigid, **hidden_updates))
if ARGS.diagnostic_no_obstacle:
    ARGS.scene_kind = "none"
    ARGS.physical_rigid = False
else:
    ARGS.scene_kind = SCENARIO.scene_object
    ARGS.physical_rigid = SCENARIO.scene_object == "rigid"
APP = AppLauncher(ARGS).app
LOGGER.mark("AFTER_APP")


def clean(value: Any) -> Any:
    if is_dataclass(value):
        return clean(asdict(value))
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if hasattr(value, "item"):
        return clean(value.item())
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def yaw_from_quat(q: np.ndarray) -> float:
    w, x, y, z = [float(item) for item in q]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    """Return an Isaac/Usd-compatible wxyz quaternion for reset-only yaw."""
    half = 0.5 * float(yaw)
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def roll_pitch_from_quat(q: np.ndarray) -> tuple[float, float]:
    w, x, y, z = [float(item) for item in q]
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    argument = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    return roll, math.asin(argument)


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def set_gravity(stage: Any, magnitude_mps2: float) -> list[str]:
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


def normalize_quat(value: np.ndarray) -> np.ndarray:
    quat = np.asarray(value, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(quat))
    return quat / norm if math.isfinite(norm) and norm > 1.0e-8 else np.asarray((1.0, 0.0, 0.0, 0.0))


def quat_rotate(quat_value: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quat = normalize_quat(quat_value)
    vec = np.asarray(vector, dtype=np.float64)
    qvec = quat[1:]
    twice_cross = 2.0 * np.cross(qvec, vec)
    return vec + quat[0] * twice_cross + np.cross(qvec, twice_cross)


def quat_conjugate(value: np.ndarray) -> np.ndarray:
    quat = normalize_quat(value)
    return np.asarray((quat[0], -quat[1], -quat[2], -quat[3]), dtype=np.float64)


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
    for field_name in ("body_link_pose_w", "body_pose_w", "body_state_w"):
        value = getattr(robot.data, field_name, None)
        if value is None:
            continue
        array = value[0].detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)[0]
        array = np.asarray(array, dtype=np.float64)
        if array.ndim == 2 and array.shape[0] == len(robot.body_names) and array.shape[1] >= 7:
            return {
                name: (array[index, :3].copy(), normalize_quat(array[index, 3:7]))
                for index, name in enumerate(robot.body_names)
            }
    raise RuntimeError("no public batched body pose field is available")


def pose_record(pose: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
    position, orientation = pose
    return {
        "position_m": position.tolist(),
        "orientation_wxyz": orientation.tolist(),
        "local_x_world": quat_rotate(orientation, np.asarray((1.0, 0.0, 0.0))).tolist(),
        "local_y_world": quat_rotate(orientation, np.asarray((0.0, 1.0, 0.0))).tolist(),
        "local_z_world": quat_rotate(orientation, np.asarray((0.0, 0.0, 1.0))).tolist(),
    }


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


def build_bbox_template(stage: Any, robot: Any, reset_poses: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, dict[str, Any]]:
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    template: dict[str, dict[str, Any]] = {}
    for name in robot.body_names:
        center, orientation = reset_poses[name]
        radius = fallback_radius(name)
        local_corners = None
        prefix = "/World/envs/env_0/Robot/"
        for prim in stage.Traverse():
            if not str(prim.GetPath()).startswith(prefix) or str(prim.GetName()) != name:
                continue
            try:
                bounds = cache.ComputeWorldBound(prim).GetRange()
                if bounds.IsEmpty():
                    continue
                lower = np.asarray(bounds.GetMin(), dtype=np.float64)
                upper = np.asarray(bounds.GetMax(), dtype=np.float64)
                corners = np.asarray([[x, y, z] for x in (lower[0], upper[0]) for y in (lower[1], upper[1]) for z in (lower[2], upper[2])], dtype=np.float64)
                candidate = float(np.max(np.linalg.norm(corners - center[None, :], axis=1)))
                if math.isfinite(candidate) and 1.0e-5 < candidate < 2.0:
                    radius = candidate
                    local_corners = np.asarray([quat_rotate(quat_conjugate(orientation), corner - center) for corner in corners])
                    break
            except Exception:
                continue
        template[name] = {"radius_m": float(radius), "local_corners_m": local_corners}
    return template


def geometry_snapshot(poses: dict[str, tuple[np.ndarray, np.ndarray]], template: dict[str, dict[str, Any]], plane_x: float, *, hand_name: str = "right_rubber_hand") -> dict[str, Any]:
    back_x: dict[str, float] = {}
    front_x: dict[str, float] = {}
    for name, (center, orientation) in poses.items():
        corners = template[name]["local_corners_m"]
        if corners is None:
            local_x = np.asarray((-template[name]["radius_m"], template[name]["radius_m"]))
            back_x[name] = float(center[0] + local_x[0])
            front_x[name] = float(center[0] + local_x[1])
        else:
            world = np.asarray([quat_rotate(orientation, corner) for corner in corners])
            back_x[name] = float(center[0] + np.min(world[:, 0]))
            front_x[name] = float(center[0] + np.max(world[:, 0]))
    nonhand = {name: value for name, value in front_x.items() if "rubber_hand" not in name}
    nonhand_back = {name: value for name, value in back_x.items() if "rubber_hand" not in name}
    return {
        "all_robot_bodies_past_plane": bool(back_x) and min(back_x.values()) > float(plane_x),
        "all_nonhand_bodies_past_plane": bool(nonhand_back) and min(nonhand_back.values()) > float(plane_x),
        "back_x_by_body_m": back_x,
        "nonhand_back_x_by_body_m": nonhand_back,
        "hand_front_x_m": front_x.get(hand_name),
        "max_nonhand_front_x_m": max(nonhand.values()) if nonhand else None,
        "min_robot_back_x_m": min(back_x.values()) if back_x else None,
        "min_nonhand_back_x_m": min(nonhand_back.values()) if nonhand_back else None,
        "nonhand_clearance_m": float(plane_x - max(nonhand.values())) if nonhand else None,
        "hand_plane_gap_m": float(plane_x - front_x[hand_name]) if hand_name in front_x else None,
    }


def read_runtime(deformable: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    simulation, _, rest = deformable.get_nodal_positions()
    positions = as_numpy(simulation)[0].astype(np.float64)
    rest_positions = as_numpy(rest)[0].astype(np.float64)
    velocities = as_numpy(deformable.get_nodal_velocities())[0].astype(np.float64)
    return positions, velocities, rest_positions


def zero_runtime_velocity(deformable: Any, node_count: int) -> None:
    deformable.set_nodal_velocities(np.zeros((1, node_count, 3), dtype=np.float32))


def max_neighbor_stretch(positions: np.ndarray, rest: np.ndarray, *, columns: int = 25, rows: int = 33) -> float:
    ratios: list[np.ndarray] = []
    grid_pos = np.asarray(positions, dtype=np.float64).reshape(rows, columns, 3)
    grid_rest = np.asarray(rest, dtype=np.float64).reshape(rows, columns, 3)
    for axis in (0, 1):
        current = np.linalg.norm(np.diff(grid_pos, axis=axis), axis=2)
        reference = np.linalg.norm(np.diff(grid_rest, axis=axis), axis=2)
        ratios.append(current / np.maximum(reference, 1.0e-12))
    return float(max(np.max(value) for value in ratios))


def write_robot_reset(robot: Any, sim: Any, root_pose: Any, joint_pos: Any, joint_vel: Any) -> None:
    robot.write_root_pose_to_sim(root_pose.unsqueeze(0))
    robot.write_joint_state_to_sim(joint_pos.unsqueeze(0), joint_vel.unsqueeze(0))
    robot.set_joint_position_target(joint_pos.unsqueeze(0))
    robot.write_data_to_sim()
    sim.step(render=False)
    robot.update(DT)


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


def smoothstep(alpha: float) -> float:
    value = max(0.0, min(1.0, float(alpha)))
    return value * value * (3.0 - 2.0 * value)


def interpolate_upper(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    value = smoothstep(alpha)
    return ((1.0 - value) * start + value * end).astype(np.float32)


def _deployment_vector(row: dict[str, str], field: str) -> np.ndarray:
    value = np.asarray(json.loads(row[field]), dtype=np.float64).reshape(-1)
    if value.size == 0 or not np.isfinite(value).all():
        raise ValueError(f"{field} must contain a finite non-empty vector")
    return value


def load_deployment_history_csv(path: Path):
    from lift_curtain.control.deployment_observation import DeploymentObservation

    with path.open(encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("phase") in ("probe", "contact_hold")]
    if not rows:
        raise ValueError(f"{path} has no probe/contact deployment rows")
    return [
        DeploymentObservation(
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
        for row in rows
    ]


def make_oracle_map(*, physical_rigid: bool = False):
    from lift_curtain.navigation.oracle_grid_map import OracleCellLabel, OracleGridMap

    grid = OracleGridMap.filled_free(
        origin_xy=(-0.80, -1.50),
        resolution_m=0.10,
        size_xy=(30, 30),
    )
    # The candidate blocks the direct route but has free space around both
    # ends.  It has no curtain/rigid subtype in the map.  Physical mode uses
    # a configuration-space inflation around the 1.2 m panel for body safety.
    candidate_y_half_width = 1.00
    grid.mark_rectangle((0.79, 0.91), (-candidate_y_half_width, candidate_y_half_width), OracleCellLabel.INTERACTION_CANDIDATE)
    # Ordinary label-1 structure shapes the upper route; the lower zero-cell
    # route remains available for a conservative detour.
    if physical_rigid:
        grid.mark_rectangle((0.94, 1.18), (1.10, 1.40), OracleCellLabel.FIXED_OBSTACLE)
    else:
        grid.mark_rectangle((0.94, 1.18), (1.10, 1.40), OracleCellLabel.FIXED_OBSTACLE)
    return grid


def simplify_free_path(grid, path: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(path) <= 2:
        return list(path)
    simplified = [path[0]]
    index = 0
    while index < len(path) - 1:
        chosen = index + 1
        for candidate in range(len(path) - 1, index, -1):
            if grid.segment_is_free(path[index], path[candidate]):
                chosen = candidate
                break
        simplified.append(path[chosen])
        index = chosen
    return simplified


def axis_aligned_zero_route(
    grid,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
) -> list[tuple[float, float]]:
    """Choose a short axis-aligned route from the explicitly free map cells.

    The grid BFS path is retained as evidence, but the locomotion controller
    receives axis-aligned receding-horizon waypoints.  This avoids asking the
    tested low-level tracker to make a long diagonal correction through a
    narrow detour while still requiring every segment to be label 0.
    """
    candidates = [
        grid.grid_to_world(ix, iy)[1]
        for iy in range(grid.size_xy[1])
        for ix in range(grid.size_xy[0])
    ]
    candidates.sort(key=lambda value: abs(float(value) - float(start_xy[1])))
    for y_value in candidates:
        points = [
            (float(start_xy[0]), float(start_xy[1])),
            (float(start_xy[0]), float(y_value)),
            (float(goal_xy[0]), float(y_value)),
            (float(goal_xy[0]), float(goal_xy[1])),
        ]
        compact: list[tuple[float, float]] = [points[0]]
        for point in points[1:]:
            if float(np.linalg.norm(np.asarray(point) - np.asarray(compact[-1]))) > 1.0e-6:
                compact.append(point)
        if all(grid.segment_is_free(a, b) for a, b in zip(compact, compact[1:])):
            return compact
    return simplify_free_path(grid, [tuple(point) for point in grid.free_path(start_xy, goal_xy) or ()])


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
    from lift_curtain.control.deployment_observation import (
        ContactDecision,
        DeploymentObservation,
        ProprioceptiveThresholds,
        classify_proprioceptive_response,
    )
    from lift_curtain.control.progress_monitor import (
        PhaseAwareProgressMonitor,
        ProgressMonitorConfig,
        ProgressPhase,
        ProgressSample,
    )
    from lift_curtain.control.se2_feedback import PushPathTracker, PushPathTrackerConfig, SE2Reference
    from lift_curtain.control.point_goal import PointGoalConfig, evaluate_goal
    from lift_curtain.navigation.interaction_memory import InteractionMemory, InteractionStatus
    from lift_curtain.navigation.oracle_grid_map import KnownMapPlan, KnownMapPlanStatus, KnownMapPlanner, OracleCellLabel
    from lift_curtain.navigation.round22_state_machine import (
        CandidateRegion,
        ProbeDecision,
        Round22State,
    )
    from lift_curtain.navigation.round25b_policy_interface import (
        HighLevelAction,
        PolicyContext,
        DEPLOYMENT_FIELDS_USED,
    )
    from lift_curtain.navigation.round27_policy_interface import (
        Round27PolicyAdapter,
        round27_privileged_information_leakage_audit as policy_privileged_information_leakage_audit,
    )
    from lift_curtain.navigation.round26_semantics import (
        Round26EvidenceStatus,
        make_evidence_record,
    )
    from lift_curtain.navigation.round26_state_machine import Round26NavigationWorkflow

    asset = ARGS.asset.resolve()
    onnx = ARGS.onnx.resolve()
    run_root = ARGS.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    initial_oracle_map = make_oracle_map(physical_rigid=True)
    candidate_region = CandidateRegion(
        (0.79, 0.91), (-1.00, 1.00), clear_margin_m=float(ARGS.candidate_clear_margin)
    )
    workflow = Round26NavigationWorkflow(initial_oracle_map, candidate_region)
    grid = workflow.navigation_map
    candidate_xy = (0.85, 0.0)
    goal_xy = (1.35, 0.0)
    traverse_exit_xy = (float(ARGS.traverse_exit_x), float(ARGS.traverse_exit_y))
    planner = KnownMapPlanner(goal_tolerance_m=float(ARGS.position_tolerance), precontact_offset_m=ARGS.precontact_offset)
    memory = InteractionMemory()
    config: dict[str, Any] = {
        "stage": "ROUND4070_FIXED10CM_JUDGMENT_PHYSICAL_EPISODE",
        "algorithm_version": "ROUND4070_FIXED10CM_JUDGMENT_RUNNER_V1",
        "task_contract": "POINTGOAL_XY_STOP_SAFETY_V1",
        "pointgoal_position_tolerance_m": ARGS.position_tolerance,
        "pointgoal_yaw_is_diagnostic_only": True,
        "baseline_snapshot": "reports/artifacts/round13/baseline_snapshot.json",
        "scenario_for_scene_builder_only": SCENARIO.to_dict(),
        "scene_kind_for_scene_builder_only": ARGS.scene_kind,
        "asset": str(asset),
        "onnx": str(onnx),
        "headless": True,
        "num_envs": 1,
        "physics_dt_s": DT,
        "decimation": DECIMATION,
        "control_dt_s": CONTROL_DT,
        "evaluation_telemetry_dt_s": EVALUATION_DT,
        "deployment_observation_dt_s": DT,
        "settle_duration_s": ARGS.settle_duration,
        "freeze_duration_s": ARGS.freeze_duration,
        "segment_timeout_s": ARGS.segment_timeout,
        "position_tolerance_m": ARGS.position_tolerance,
        "cruise_speed_mps": ARGS.cruise_speed,
        "vx_max_mps": ARGS.vx_max,
        "stand_duration_s": ARGS.stand_duration,
        "stop_dwell_duration_s": ARGS.stop_dwell_duration,
        "reset_root_xy_m": [ARGS.reset_root_x, ARGS.reset_root_y],
        "reset_root_yaw_rad": ARGS.reset_root_yaw,
        "precontact_offset_m": ARGS.precontact_offset,
        "traverse_exit_xy_m": list(traverse_exit_xy),
        "candidate_clear_margin_m": ARGS.candidate_clear_margin,
        "goal_yaw_tolerance_rad": ARGS.goal_yaw_tolerance,
            "goal_yaw_role": "diagnostic_only_never_a_pointgoal_gate",
        "precontact_region": {
            "enabled": ARGS.precontact_region_x_min is not None or ARGS.precontact_region_x_max is not None or ARGS.precontact_region_y_half_width is not None,
            "x_min_m": ARGS.precontact_region_x_min,
            "x_max_m": ARGS.precontact_region_x_max,
            "y_half_width_m": ARGS.precontact_region_y_half_width,
            "speed_max_mps": ARGS.precontact_region_speed_max,
        },
        "physical_candidate": True,
        "scene_geometry": {
            "kind": ARGS.scene_kind,
            "center_m": [candidate_xy[0], candidate_xy[1], 1.0],
            "size_m": [0.04, 1.20, 2.00],
            "map_candidate_y_half_width_m": 1.00,
            "map_inflation_role": "known configuration-space safety margin; no object subtype or passability information",
            "curtain_top_z_m": CURTAIN_TOP_Z_M,
            "curtain_bottom_z_m": 0.02,
            "curtain_width_m": 1.20,
            "curtain_length_m": float(ARGS.curtain_length),
        },
        "probe_schedule_s": {
            "preprobe": ARGS.preprobe_duration,
            "preprobe_hold": ARGS.preprobe_hold_duration,
            "probe": ARGS.probe_duration,
            "probe_hold": ARGS.probe_hold_duration,
            "retract": ARGS.retract_duration,
            "neutralize": ARGS.neutralize_duration,
        },
        "probe_depth_contract": {
            "level": "PROBE_10CM",
            "target_displacement_m": TARGET_DISPLACEMENT_M,
            "direction_world": PROBE_DIRECTION_WORLD.tolist(),
            "command_start": "verified_precontact_preprobe_pose",
            "continuous_single_probe": True,
            "retry_count": 0,
            "baseline_target_upper": make_side_upper(ARGS.probe_side, -0.55, 0.55).tolist(),
            "target_upper": PROBE_TARGET_UPPER.tolist(),
            "joint_step_rad": float(ARGS.probe_10cm_joint_step_rad),
            "target_reference_profile": PROBE_TARGET_REFERENCE_PROFILE,
            "target_reference_source": PROBE_TARGET_REFERENCE_SOURCE,
            "target_position_role": PROBE_TARGET_POSITION_ROLE,
            "old_depth_schedule_forbidden": True,
        },
        "probe_validation_only": bool(ARGS.probe_validation_only),
        "structural_topology": ARGS.structural_topology,
        "surface_thickness_m": float(ARGS.surface_thickness),
        "opening_action_contract": {
            "mode": "hold" if OPENING_ENABLED else "none",
                "enabled": bool(OPENING_ENABLED),
            "empty_scene_control": bool(ARGS.diagnostic_no_obstacle),
            "active_side": ARGS.probe_side,
            "joint": "shoulder_yaw",
            "target_upper": OPENING_TARGET_UPPER.tolist(),
            "amplitude_rad": float(ARGS.opening_lateral_amplitude),
            "direction_sign": float(ARGS.opening_direction_sign),
            "sweep_duration_s": float(ARGS.opening_sweep_duration),
            "hold_duration_s": float(ARGS.opening_hold_duration),
            "deployment_trigger": "RESPONSIVE only; no simulator contact/object identity input",
        },
        "proprio_baseline_csv": str(ARGS.proprio_baseline_csv.resolve()),
        "proprio_thresholds_json": str(ARGS.proprio_thresholds_json.resolve()),
        "reset_policy_at_precontact": False,
        "deprecated_reset_policy_flag_requested": bool(ARGS.reset_policy_at_precontact),
        "engineering_fix_experiment": {
            "accept_planner_goal_reached_after_responsive_release": bool(ARGS.engineering_fix_accept_goal_reached),
            "included_in_frozen_baseline_success_rate": False,
        },
        "research_feasibility_post_clearance_replan": {
            "enabled": bool(ARGS.research_feasibility_post_clearance_replan),
            "deployment_gate": "candidate_clearance_verified plus no fixed-label-1 cell on measured post-release-to-goal segment",
            "uses_simulator_truth": False,
            "historical_baseline_modified": False,
        },
        "round22_progress_monitor": {
            "enabled": bool(ARGS.round15_progress_monitor),
            "thresholds_frozen": True,
            "deployment_only": True,
            "active_phase": "CAUTIOUS_TRAVERSE",
            "sample_period_s": EVALUATION_DT,
            "clearance_source": "robot_owned_collision_envelope_and_known_candidate_geometry",
        },
        "candidate_xy_m": list(candidate_xy),
        "goal_xy_m": list(goal_xy),
        "initial_oracle_map": initial_oracle_map.to_dict(),
        "oracle_map": grid.to_dict(),
        "state_machine_states": [state.value for state in Round22State],
        "evaluation_roles": {"deployment_observation": True, "evaluation_ground_truth": "offline_only"},
        "probe_decision_source": {"mode": "actual_bounded_probe", "replayed_decision": None, "not_object_truth": True},
        "failure_discovery_condition_id": ARGS.condition_id,
        "failure_discovery_overrides": {
            "curtain_length_m": float(ARGS.curtain_length),
            "curtain_top_z_m": CURTAIN_TOP_Z_M,
            "density_scale_multiplier": float(ARGS.density_scale_multiplier),
            "curtain_damping": ARGS.curtain_damping,
            "curtain_bend_stiffness": ARGS.curtain_bend_stiffness,
            "curtain_stretch_stiffness": ARGS.curtain_stretch_stiffness,
            "curtain_shear_stiffness": ARGS.curtain_shear_stiffness,
            "structural_topology": ARGS.structural_topology,
            "surface_thickness_m": float(ARGS.surface_thickness),
            "hidden_rigid_x": ARGS.hidden_rigid_x,
            "hidden_rigid_y": ARGS.hidden_rigid_y,
            "hidden_rigid_width": ARGS.hidden_rigid_width,
            "hidden_rigid_height": ARGS.hidden_rigid_height,
            "hidden_rigid_thickness": ARGS.hidden_rigid_thickness,
            "hidden_rigid_bottom_z": ARGS.hidden_rigid_bottom_z,
        },
        "high_level_policy": ARGS.high_level_policy,
        "dynamic_decision_contract": {
            "decision_boundaries": ["post_probe", "post_bounded_advance", "progress_monitor_stop"],
            "advance_segment_length_m": 0.12,
            "max_advance_segments": 10,
            "all_policies_share_action_mask": True,
        },
        "policy_checkpoint_requested": None if ARGS.policy_checkpoint is None else str(ARGS.policy_checkpoint.resolve()),
        "policy_action_contract": [action.value for action in HighLevelAction],
        "policy_safety_supervisor": "independent deployment-safe action mask and stability/progress filter",
        "policy_privileged_information_audit": policy_privileged_information_leakage_audit(),
        "hypothesis": "A bounded multi-decision high-level interface can use deployment-only progress evidence without converting conservative avoidance into physical blocking evidence.",
        "exact_question": "Can the same safe action contract compare rule and learned policies on a real short-cut decision?",
        "changed_variable": "Round 13 scene construction only; FALCON checkpoint, actuator values, proprio thresholds, SE(2) gains, command limits and state-machine gates remain frozen from Round 12.",
        "privileged_information_policy": {
            "controller_inputs": [
                "oracle_grid_0_1_2",
                "simulator_pose_localization_assumption",
                "relative_goal",
                "q_qdot_q_cmd_motor_effort_previous_action",
                "robot_FK_hand_history",
            ],
            "forbidden_inputs": ["object_class", "contact_flag", "contact_force", "curtain_nodes", "true_passability", "episode_outcome_oracle"],
        },
    }
    (run_root / "resolved_config.json").write_text(json.dumps(clean(config), indent=2) + "\n", encoding="utf-8")
    report: dict[str, Any] = {**config, "status": "BLOCKED"}
    sim = None
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    policy_decisions: list[dict[str, Any]] = []
    policy_adapter = None
    try:
        if not asset.is_file() or not onnx.is_file():
            raise FileNotFoundError(f"missing asset or checkpoint: {asset}, {onnx}")
        policy_adapter = Round27PolicyAdapter(
            ARGS.high_level_policy,
            repo=REPO,
            checkpoint=None if ARGS.policy_checkpoint is None else ARGS.policy_checkpoint.resolve(),
        )
        events.append({
            "event": "high_level_policy_loaded",
            "policy": ARGS.high_level_policy.upper(),
            "checkpoint": None if policy_adapter.checkpoint is None else str(policy_adapter.checkpoint),
            "model_status": policy_adapter.model_status,
        })
        if min(
            ARGS.settle_duration,
            ARGS.freeze_duration,
            ARGS.segment_timeout,
            ARGS.position_tolerance,
            ARGS.stand_duration,
            ARGS.stop_dwell_duration,
        ) <= 0.0:
            raise ValueError("durations and position tolerance must be positive")
        if min(
            ARGS.preprobe_duration,
            ARGS.preprobe_hold_duration,
            ARGS.probe_duration,
            ARGS.probe_hold_duration,
            ARGS.retract_duration,
            ARGS.neutralize_duration,
        ) <= 0.0:
            raise ValueError("physical rigid probe durations must be positive")
        if (
            not ARGS.proprio_baseline_csv.resolve().is_file()
            or not ARGS.proprio_thresholds_json.resolve().is_file()
        ):
            raise FileNotFoundError("physical rigid mode requires the frozen proprio baseline and thresholds")
        if not 0.0 < ARGS.cruise_speed <= ARGS.vx_max:
            raise ValueError("cruise speed must be positive and no larger than vx-max")
        if not all(math.isfinite(float(value)) for value in (*traverse_exit_xy, ARGS.candidate_clear_margin)):
            raise ValueError("traverse exit and candidate clear margin must be finite")
        if ARGS.candidate_clear_margin < 0.0:
            raise ValueError("candidate clear margin cannot be negative")
        if not math.isfinite(ARGS.goal_yaw_tolerance) or ARGS.goal_yaw_tolerance <= 0.0:
            raise ValueError("goal yaw tolerance must be finite and positive")
        if not math.isfinite(ARGS.reset_root_yaw):
            raise ValueError("reset root yaw must be finite")
        region_enabled = (
            ARGS.precontact_region_x_min is not None
            or ARGS.precontact_region_x_max is not None
            or ARGS.precontact_region_y_half_width is not None
        )
        if region_enabled:
            if (
                ARGS.precontact_region_x_min is None
                or ARGS.precontact_region_x_max is None
                or ARGS.precontact_region_y_half_width is None
            ):
                raise ValueError("precontact region requires x-min, x-max and y-half-width")
            if not all(math.isfinite(value) for value in (
                ARGS.precontact_region_x_min,
                ARGS.precontact_region_x_max,
                ARGS.precontact_region_y_half_width,
            )):
                raise ValueError("precontact region bounds must be finite")
            if ARGS.precontact_region_x_min >= ARGS.precontact_region_x_max:
                raise ValueError("precontact region x-min must be smaller than x-max")
            if ARGS.precontact_region_y_half_width <= 0.0 or ARGS.precontact_region_speed_max <= 0.0:
                raise ValueError("precontact region width and speed limit must be positive")
        if ARGS.reset_policy_at_precontact:
            raise ValueError("Round 10 forbids policy-history reset at pre-contact; use explicit Stand instead")
        np.random.seed(ARGS.seed)
        torch.manual_seed(ARGS.seed)
        sim = SimulationContext(SimulationCfg(dt=DT, render_interval=DECIMATION, device="cuda:0"))
        ground = sim_utils.CuboidCfg(
            size=(8.0, 8.0, 0.10),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.18, 0.18)),
        )
        ground.func("/World/defaultGroundPlane", ground, translation=(0.0, 0.0, -0.05))
        stage = get_current_stage()
        report["physics_scene_paths"] = set_gravity(stage, 9.81)
        curtain_info = None
        curtain_deformable = None
        runtime_rest = None
        top_indices: list[int] = []
        anchor_indices: list[int] = []
        if ARGS.scene_kind == "curtain":
            from lift_curtain.envs.task_scale_scene import FormalTaskSceneConfig
            from structural_curtain_asset import CurtainConfig as StructuralCurtainConfig, create_surface_sheet

            formal_scene = FormalTaskSceneConfig(
                curtain_x_m=float(candidate_xy[0]),
                curtain_top_z_m=CURTAIN_TOP_Z_M,
                curtain_length_m=float(ARGS.curtain_length),
            )
            scenario_curtain_config = SCENARIO.curtain_config(formal_scene)
            if scenario_curtain_config is None:
                raise RuntimeError("curtain scenario did not produce a curtain config")
            curtain_values = dict(scenario_curtain_config.__dict__)
            curtain_values.update(
                {
                    "density_kg_m3": float(scenario_curtain_config.density_kg_m3) * float(ARGS.density_scale_multiplier),
                    "damping": float(ARGS.curtain_damping) if ARGS.curtain_damping is not None else scenario_curtain_config.damping,
                    "bend_stiffness": float(ARGS.curtain_bend_stiffness) if ARGS.curtain_bend_stiffness is not None else scenario_curtain_config.bend_stiffness,
                    "stretch_stiffness": float(ARGS.curtain_stretch_stiffness) if ARGS.curtain_stretch_stiffness is not None else scenario_curtain_config.stretch_stiffness,
                    "shear_stiffness": float(ARGS.curtain_shear_stiffness) if ARGS.curtain_shear_stiffness is not None else scenario_curtain_config.shear_stiffness,
                    "surface_thickness_m": float(ARGS.surface_thickness),
                    "structural_topology": str(ARGS.structural_topology),
                }
            )
            scenario_curtain_config = StructuralCurtainConfig(**curtain_values)
            scenario_curtain_config.validate()
            curtain_info = create_surface_sheet(
                stage,
                "/World/Curtain/Round13FormalSheet",
                scenario_curtain_config,
                create_attachment=True,
            )
            report["formal_scene_config"] = formal_scene.to_dict()
            report["scenario_curtain_config"] = clean(scenario_curtain_config)
            report["curtain_runtime_paths"] = {
                "mesh": curtain_info.prim_path,
                "support": curtain_info.support_path,
                "attachment": curtain_info.attachment_path,
                "attachment_paths": list(curtain_info.attachment_paths),
                "bottom_anchor_support_paths": list(curtain_info.bottom_anchor_support_paths),
                "bottom_anchor_attachment_paths": list(curtain_info.bottom_anchor_attachment_paths),
                "structural_topology": curtain_info.structural_topology,
                "requested_fixed_vertex_indices": list(curtain_info.requested_fixed_vertex_indices),
                "constraint_objects": list(curtain_info.constraint_objects),
            }
            anchor_indices = list(curtain_info.requested_bottom_anchor_indices)
            if SCENARIO.hidden_rigid is not None:
                hidden = SCENARIO.hidden_rigid
                hidden_wall = sim_utils.CuboidCfg(
                    size=(hidden.thickness_m, hidden.width_m, hidden.height_m),
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.16, 0.12)),
                )
                hidden_wall.func(
                    "/World/Diagnostic/Round13HiddenRigid",
                    hidden_wall,
                    translation=hidden.center_xyz,
                )
                report["hidden_rigid_geometry_evaluation_only"] = clean(SCENARIO.hidden_rigid)
        if ARGS.physical_rigid:
            panel = sim_utils.CuboidCfg(
                size=(0.04, 1.20, 2.00),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.65, 0.20, 0.12)),
            )
            panel.func(
                "/World/Diagnostic/FormalRigidPanel",
                panel,
                translation=(float(candidate_xy[0]), float(candidate_xy[1]), 1.0),
            )
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
                pos=(float(ARGS.reset_root_x), float(ARGS.reset_root_y), 0.80),
                rot=quat_from_yaw(ARGS.reset_root_yaw),
                joint_pos={name: float(DEFAULT_JOINT_POS[index]) for index, name in enumerate(OFFICIAL_POLICY_JOINT_ORDER)},
                joint_vel={".*": 0.0},
            ),
            actuators=actuators,
        ))
        hand_contact_sensor = None
        hand_contact_sensor_error = None
        try:
            hand_contact_sensor = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/envs/env_0/Robot/{ACTIVE_HAND_NAME}",
                update_period=0.0,
                history_length=1,
                track_air_time=False,
                force_threshold=1.0,
            ))
        except Exception as exc:
            hand_contact_sensor_error = f"{type(exc).__name__}: {exc}"
        LOGGER.mark(
            "AFTER_SCENE",
            details={
                "scenario": ARGS.scenario,
                "scene_kind": ARGS.scene_kind,
                "curtain_attachment_paths": [] if curtain_info is None else list(curtain_info.attachment_paths),
            },
        )
        sim.reset()
        robot.reset()
        robot.update(DT)
        LOGGER.mark("AFTER_RESET", details={"robot_bodies": len(robot.body_names), "robot_joints": len(robot.joint_names)})
        if hand_contact_sensor is not None:
            try:
                hand_contact_sensor.update(DT, force_recompute=True)
                report["hand_contact_reporter"] = {
                    "status": "INITIALIZED",
                    "body_names": list(hand_contact_sensor.body_names),
                    "prim_path": hand_contact_sensor.cfg.prim_path,
                    "threshold_N": 1.0,
                    "role": "evaluation-only; never read by classifier, memory, planner or locomotion controller",
                }
            except Exception as exc:
                hand_contact_sensor_error = f"{type(exc).__name__}: {exc}"
                hand_contact_sensor = None
        if hand_contact_sensor is None:
            report["hand_contact_reporter"] = {
                "status": "UNAVAILABLE",
                "error": hand_contact_sensor_error,
                "role": "evaluation-only",
            }
        FalconAdapter.validate_articulation(robot)
        adapter = FalconAdapter(FalconAdapterConfig(onnx_path=onnx))
        initial_root_pose = robot.data.default_root_state[0, :7].detach().clone()
        initial_joint = torch.as_tensor(
            np.asarray(DEFAULT_JOINT_POS, dtype=np.float32)[list(OFFICIAL_TO_ISAACLAB)],
            device=sim.device,
            dtype=robot.data.joint_pos.dtype,
        )
        initial_velocity = torch.zeros_like(initial_joint)
        robot.write_root_pose_to_sim(initial_root_pose.unsqueeze(0))
        if hasattr(robot, "write_root_velocity_to_sim"):
            robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device, dtype=initial_root_pose.dtype))
        robot.write_joint_state_to_sim(initial_joint.unsqueeze(0), initial_velocity.unsqueeze(0))
        robot.set_joint_position_target(initial_joint.unsqueeze(0))
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(DT)
        LOGGER.mark("FIRST_PHYSICS_STEP")
        adapter.reset()
        hand_body_index = robot.body_names.index(ACTIVE_HAND_NAME)
        left_hand_body_index = robot.body_names.index("left_rubber_hand")
        right_hand_body_index = robot.body_names.index("right_rubber_hand")
        initial_body_poses = body_poses(robot)
        bbox_template = build_bbox_template(stage, robot, initial_body_poses)
        last_geometry = geometry_snapshot(initial_body_poses, bbox_template, float(candidate_xy[0]), hand_name=ACTIVE_HAND_NAME)
        curtain_initial = None
        if curtain_info is not None:
            import omni.kit.app

            manager = omni.kit.app.get_app().get_extension_manager()
            manager.set_extension_enabled_immediate("isaacsim.core.experimental.prims", True)
            from isaacsim.core.experimental.prims import DeformablePrim

            curtain_deformable = DeformablePrim(curtain_info.prim_path, deformable_type="surface")
            curtain_positions, raw_velocities, runtime_rest = read_runtime(curtain_deformable)
            zero_runtime_velocity(curtain_deformable, curtain_positions.shape[0])
            curtain_positions, curtain_velocities, runtime_rest = read_runtime(curtain_deformable)
            curtain_initial = curtain_positions.copy()
            top_indices = np.flatnonzero(
                np.abs(runtime_rest[:, 2] - np.max(runtime_rest[:, 2])) <= 1.0e-5
            ).astype(int).tolist()
            report["curtain_runtime_initialization"] = {
                "node_count": int(curtain_positions.shape[0]),
                "raw_velocity_max_mps": float(np.max(np.linalg.norm(raw_velocities, axis=1))),
                "zeroed_velocity_max_mps": float(np.max(np.linalg.norm(curtain_velocities, axis=1))),
                "top_indices": top_indices,
                "requested_bottom_anchor_indices": anchor_indices,
            }

        tracker_config = PushPathTrackerConfig(
            position_gain_xy=(0.55, 0.90),
            heading_gain=0.85,
            vx_min=0.0,
            vx_max=ARGS.vx_max,
            vy_limit=0.15,
            wz_limit=0.35,
            max_command_step=(0.020, 0.020, 0.030),
        )
        tracker = PushPathTracker(tracker_config)
        last_target = initial_joint.unsqueeze(0).clone()
        target_official = np.asarray(DEFAULT_JOINT_POS, dtype=np.float64).copy()
        sim_step = 0
        fall_reason: str | None = None
        latencies: list[float] = []
        deployment_samples: dict[str, list[DeploymentObservation]] = {}
        deployment_effort_sources: dict[str, list[str]] = {}
        previous_hand_positions: dict[str, np.ndarray | None] = {"left": None, "right": None}
        curtain_snapshots: list[np.ndarray] = []
        curtain_snapshot_times: list[float] = []
        pre_probe_curtain: np.ndarray | None = None
        pre_probe_hand: np.ndarray | None = None
        external_safety_reason: str | None = None
        probe_safety_stop_reason: str | None = None
        probe_safety_stop_time_s: float | None = None
        probe_safety_stall_steps = 0
        probe_first_contact_time_s: float | None = None
        probe_first_contact_hand: np.ndarray | None = None
        probe_first_contact_curtain_displacement_m: float | None = None
        progress_monitor = (
            PhaseAwareProgressMonitor(ProgressMonitorConfig())
            if ARGS.round15_progress_monitor
            else None
        )
        progress_monitor_stop_requested = False
        progress_monitor_axis: np.ndarray | None = None
        progress_monitor_segment_id: str | None = None
        progress_monitor_last_reading: Any | None = None
        progress_monitor_stop_reason: str | None = None
        last_hand_contact_force_norm: float | None = None
        last_curtain_top_drift = 0.0
        last_curtain_anchor_drift = 0.0
        last_curtain_region_displacement = 0.0
        last_curtain_max_speed = 0.0
        last_curtain_stretch = 1.0
        last_rollout_heartbeat_wall = time.monotonic()

        def execute_step(
            phase: str,
            local_step: int,
            reference: SE2Reference,
            target_xy: tuple[float, float],
            command: np.ndarray,
            *,
            walk_mode: bool,
            upper_reference: np.ndarray | None = None,
            command_saturated: bool = False,
            command_rate_limited: bool = False,
        ) -> tuple[Any, float]:
            nonlocal last_target, target_official, sim_step, fall_reason, previous_hand_positions
            nonlocal external_safety_reason, last_geometry, last_hand_contact_force_norm
            nonlocal probe_safety_stop_reason, probe_safety_stop_time_s, probe_safety_stall_steps
            nonlocal probe_first_contact_time_s, probe_first_contact_hand, probe_first_contact_curtain_displacement_m
            nonlocal last_curtain_top_drift, last_curtain_anchor_drift
            nonlocal last_curtain_region_displacement, last_curtain_max_speed, last_curtain_stretch
            nonlocal last_rollout_heartbeat_wall
            nonlocal progress_monitor_stop_requested, progress_monitor_last_reading
            nonlocal progress_monitor_stop_reason
            state = adapter.make_state(robot)
            requested_upper = np.zeros(14, dtype=np.float32) if upper_reference is None else np.asarray(upper_reference, dtype=np.float32)
            if requested_upper.shape != (14,) or not np.isfinite(requested_upper).all():
                raise ValueError("upper_reference must be a finite 14-vector")
            if local_step == 0 or sim_step % DECIMATION == 0:
                started = time.perf_counter()
                target = adapter.compute_target(
                    state,
                    VelocityCommand(
                        vx=float(command[0]),
                        vy=float(command[1]),
                        wz=float(command[2]),
                        walk_mode=walk_mode,
                    ),
                    upper_reference=requested_upper,
                )
                latencies.append(time.perf_counter() - started)
                last_target = torch.as_tensor(target, device=sim.device, dtype=robot.data.joint_pos.dtype).unsqueeze(0)
                target_official = np.asarray(target, dtype=np.float64)[list(ISAACLAB_TO_OFFICIAL)]
            robot.set_joint_position_target(last_target)
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(DT)
            telemetry_sample = bool(local_step == 0 or sim_step % EVALUATION_DECIMATION == 0)
            if hand_contact_sensor is not None and telemetry_sample:
                try:
                    hand_contact_sensor.update(DT, force_recompute=True)
                    raw_force = hand_contact_sensor.data.net_forces_w
                    if hasattr(raw_force, "detach"):
                        raw_force = raw_force.detach().cpu().numpy()
                    last_hand_contact_force_norm = float(np.max(np.linalg.norm(np.asarray(raw_force)[0], axis=-1)))
                except Exception:
                    last_hand_contact_force_norm = None
            hand_contact_force_norm = last_hand_contact_force_norm
            after = adapter.make_state(robot)
            roll, pitch = roll_pitch_from_quat(after.base_quat_w)
            yaw = yaw_from_quat(after.base_quat_w)
            finite = bool(np.isfinite(after.base_pos_w).all() and np.isfinite(after.joint_pos).all())
            if not finite and fall_reason is None:
                fall_reason = "NONFINITE_STATE"
            elif after.base_pos_w[2] < 0.55 and fall_reason is None:
                fall_reason = "ROOT_HEIGHT_BELOW_0P55"
            elif (abs(roll) > 0.60 or abs(pitch) > 0.60) and fall_reason is None:
                fall_reason = "ROOT_ROLL_PITCH_EXCEEDED_0P60"
            reference_error = np.asarray(reference.position_world, dtype=np.float64) - after.base_pos_w[:2].astype(np.float64)
            target_error = np.asarray(target_xy, dtype=np.float64) - after.base_pos_w[:2].astype(np.float64)
            cross_track = -math.sin(reference.yaw) * reference_error[0] + math.cos(reference.yaw) * reference_error[1]
            longitudinal = math.cos(reference.yaw) * reference_error[0] + math.sin(reference.yaw) * reference_error[1]
            target_distance = float(np.linalg.norm(target_error))
            q_official = after.joint_pos[list(ISAACLAB_TO_OFFICIAL)].astype(np.float64)
            qdot_official = after.joint_vel[list(ISAACLAB_TO_OFFICIAL)].astype(np.float64)
            effort_official = after.joint_effort[list(ISAACLAB_TO_OFFICIAL)].astype(np.float64)
            hand_positions = {
                "left": robot.data.body_pos_w[0, left_hand_body_index].detach().cpu().numpy().astype(np.float64),
                "right": robot.data.body_pos_w[0, right_hand_body_index].detach().cpu().numpy().astype(np.float64),
            }
            hand_velocities: dict[str, np.ndarray] = {}
            for side, position in hand_positions.items():
                previous = previous_hand_positions[side]
                hand_velocities[side] = np.zeros(3, dtype=np.float64) if previous is None else (position - previous) / DT
                previous_hand_positions[side] = position.copy()
            hand_position = hand_positions[ARGS.probe_side]
            hand_velocity = hand_velocities[ARGS.probe_side]
            if telemetry_sample:
                last_geometry = geometry_snapshot(body_poses(robot), bbox_template, float(candidate_xy[0]), hand_name=ACTIVE_HAND_NAME)
            geometry = last_geometry
            if curtain_deformable is not None and telemetry_sample:
                current_nodes, node_velocities, node_rest = read_runtime(curtain_deformable)
                curtain_snapshots.append(current_nodes.copy())
                curtain_snapshot_times.append(sim_step * DT)
                last_curtain_max_speed = float(np.max(np.linalg.norm(node_velocities, axis=1)))
                last_curtain_stretch = max_neighbor_stretch(current_nodes, node_rest)
                if top_indices:
                    last_curtain_top_drift = float(
                        np.max(np.linalg.norm(current_nodes[top_indices] - curtain_initial[top_indices], axis=1))
                    )
                if anchor_indices:
                    last_curtain_anchor_drift = float(
                        np.max(
                            np.linalg.norm(
                                current_nodes[anchor_indices] - curtain_initial[anchor_indices],
                                axis=1,
                            )
                        )
                    )
                if pre_probe_curtain is not None:
                    distances = np.linalg.norm(current_nodes - pre_probe_curtain, axis=1)
                    last_curtain_region_displacement = float(np.max(distances))
            curtain_top_drift = last_curtain_top_drift
            curtain_anchor_drift = last_curtain_anchor_drift
            curtain_region_displacement = last_curtain_region_displacement
            curtain_max_speed = last_curtain_max_speed
            curtain_stretch = last_curtain_stretch
            if (
                probe_first_contact_time_s is None
                and phase.startswith("physical_probe")
                and (
                    (hand_contact_force_norm is not None and hand_contact_force_norm > 1.0)
                    or curtain_region_displacement >= 0.002
                )
            ):
                probe_first_contact_time_s = (sim_step + 1) * DT
                probe_first_contact_hand = hand_position.copy()
                probe_first_contact_curtain_displacement_m = float(curtain_region_displacement)

            if phase == "physical_probe" and probe_safety_stop_reason is None:
                tracking_rms = float(np.sqrt(np.mean(np.square(target_official[15:] - q_official[15:]))))
                effort_ratio = float(np.max(np.abs(effort_official) / np.maximum(JOINT_EFFORT_LIMIT, 1.0e-6)))
                position_limit_violation = bool(
                    np.any(q_official < JOINT_POS_LOWER - 0.05)
                    or np.any(q_official > JOINT_POS_UPPER + 0.05)
                )
                velocity_limit_violation = bool(
                    np.any(np.abs(qdot_official) > 0.95 * JOINT_VELOCITY_LIMIT)
                )
                if local_step >= int(round(0.30 / DT)) and abs(float(hand_velocity[0])) < 0.001:
                    probe_safety_stall_steps += 1
                else:
                    probe_safety_stall_steps = 0
                safety_reason = None
                if position_limit_violation:
                    safety_reason = "JOINT_POSITION_LIMIT_MARGIN"
                elif velocity_limit_violation:
                    safety_reason = "JOINT_VELOCITY_LIMIT_MARGIN"
                elif abs(roll) > 0.45 or abs(pitch) > 0.45:
                    safety_reason = "BALANCE_ROLL_PITCH_MARGIN"
                elif tracking_rms > 0.20:
                    safety_reason = "UPPER_TRACKING_RMS_LIMIT"
                elif effort_ratio > 0.85:
                    safety_reason = "EFFORT_PROXY_RATIO_LIMIT"
                elif not (-0.20 <= float(hand_position[0]) <= 1.10 and abs(float(hand_position[1])) <= 0.85 and 0.30 <= float(hand_position[2]) <= 1.40):
                    safety_reason = "HAND_WORKSPACE_LIMIT"
                elif probe_safety_stall_steps >= int(round(0.40 / DT)):
                    safety_reason = "PROBE_FORWARD_STAGNATION"
                if safety_reason is not None:
                    probe_safety_stop_reason = safety_reason
                    probe_safety_stop_time_s = (sim_step + 1) * DT
                    events.append({
                        "event": "probe_safety_stop",
                        "phase": phase,
                        "reason": safety_reason,
                        "time_s": probe_safety_stop_time_s,
                        "depth_level": ARGS.probe_depth_level,
                        "effort_proxy_ratio": effort_ratio,
                        "upper_tracking_rms_rad": tracking_rms,
                        "base_roll_rad": float(roll),
                        "base_pitch_rad": float(pitch),
                    })
            # The responsive branch is the actual curtain traverse phase in
            # the shared runner. Keep the evaluation-only curtain limits
            # active there as well as for the historical phase name.
            if phase.startswith(("curtain_traverse", "responsive_traverse")) and curtain_deformable is not None:
                if curtain_top_drift > 0.01:
                    external_safety_reason = "CURTAIN_TOP_DRIFT_LIMIT"
                elif curtain_stretch > 4.0:
                    external_safety_reason = "CURTAIN_STRETCH_LIMIT"
                elif curtain_region_displacement > 1.50:
                    external_safety_reason = "CURTAIN_DISPLACEMENT_LIMIT"
            if phase.startswith("responsive_traverse") and ARGS.scene_kind == "rigid":
                # Known rigid geometry is an external safety monitor only. It
                # cannot select the probe branch or classify the object.
                if geometry["nonhand_clearance_m"] is not None and geometry["nonhand_clearance_m"] < -0.01:
                    external_safety_reason = "RIGID_BODY_GEOMETRY_SAFETY_LIMIT"
            if progress_monitor is not None:
                # The frozen monitor contract is evaluated at the existing
                # 0.1 s deployment telemetry cadence. Updating it at every
                # 0.005 s physics step would reinterpret "3 windows" as
                # three physics samples and cause an integration-only stop.
                if (
                    phase == "responsive_traverse" or phase.startswith("round26_cautious_advance")
                ) and telemetry_sample:
                    if progress_monitor_axis is None or progress_monitor_segment_id is None:
                        raise RuntimeError("Round15 progress monitor segment was not initialized")
                    command_world = np.asarray(
                        (
                            math.cos(yaw) * float(command[0]) - math.sin(yaw) * float(command[1]),
                            math.sin(yaw) * float(command[0]) + math.cos(yaw) * float(command[1]),
                        ),
                        dtype=np.float64,
                    )
                    minimum_nonhand_back = geometry.get("min_nonhand_back_x_m")
                    candidate_clearance = bool(
                        minimum_nonhand_back is not None
                        and float(minimum_nonhand_back)
                        >= float(candidate_region.x_limits_m[1] + ARGS.candidate_clear_margin)
                    )
                    progress_monitor_last_reading = progress_monitor.update(
                        ProgressSample(
                            time_s=(sim_step + 1) * DT,
                            base_xy_m=(float(after.base_pos_w[0]), float(after.base_pos_w[1])),
                            command_world_xy_mps=(float(command_world[0]), float(command_world[1])),
                            roll_rad=float(roll),
                            pitch_rad=float(pitch),
                            hand_xy_m=(float(hand_position[0]), float(hand_position[1])),
                        ),
                        phase=ProgressPhase.CAUTIOUS_TRAVERSE,
                        requires_forward_progress=not candidate_clearance,
                        candidate_clearance_verified=candidate_clearance,
                        progress_axis_world=progress_monitor_axis,
                        segment_id=progress_monitor_segment_id,
                    )
                    if progress_monitor_last_reading.stop_requested:
                        progress_monitor_stop_requested = True
                        progress_monitor_stop_reason = progress_monitor_last_reading.reason
                elif progress_monitor.active:
                    progress_monitor_last_reading = progress_monitor.update(
                        ProgressSample(
                            time_s=(sim_step + 1) * DT,
                            base_xy_m=(float(after.base_pos_w[0]), float(after.base_pos_w[1])),
                            command_world_xy_mps=(0.0, 0.0),
                            roll_rad=float(roll),
                            pitch_rad=float(pitch),
                            hand_xy_m=(float(hand_position[0]), float(hand_position[1])),
                        ),
                        phase=ProgressPhase.INACTIVE,
                        requires_forward_progress=False,
                    )
            deployment_sample = DeploymentObservation(
                q=q_official,
                qdot=qdot_official,
                q_cmd=target_official,
                motor_effort=effort_official,
                previous_action=np.asarray(adapter.previous_action, dtype=np.float64),
                hand_fk_position_w=hand_position,
                hand_fk_velocity_w=hand_velocity,
                base_position_w=after.base_pos_w.astype(np.float64),
                base_velocity_w=after.base_lin_vel_b.astype(np.float64),
                imu_orientation_rpy=np.asarray((roll, pitch, yaw), dtype=np.float64),
                imu_angular_velocity=after.base_ang_vel_b.astype(np.float64),
                map_cell_label=int(grid.label_at(float(after.base_pos_w[0]), float(after.base_pos_w[1]))),
                relative_goal_xy=np.asarray(goal_xy, dtype=np.float64) - after.base_pos_w[:2].astype(np.float64),
                localization_assumption="Isaac root pose used only as the permitted phase-1 localization assumption",
            )
            deployment_samples.setdefault(phase, []).append(deployment_sample)
            deployment_effort_sources.setdefault(phase, []).append(str(after.effort_source))
            if telemetry_sample:
                observation = decode_last_observation(adapter)
                rows.append({
                "phase": phase,
                "phase_step": local_step,
                "step": sim_step,
                "time_s": (sim_step + 1) * DT,
                "reference_x": float(reference.position_world[0]),
                "reference_y": float(reference.position_world[1]),
                "reference_yaw": float(reference.yaw),
                "reference_vx_world": float(reference.velocity_world[0]),
                "reference_vy_world": float(reference.velocity_world[1]),
                "target_x": float(target_xy[0]),
                "target_y": float(target_xy[1]),
                "base_x": float(after.base_pos_w[0]),
                "base_y": float(after.base_pos_w[1]),
                "base_z": float(after.base_pos_w[2]),
                "roll": float(roll),
                "pitch": float(pitch),
                "yaw": float(yaw),
                "command_vx": float(command[0]),
                "command_vy": float(command[1]),
                "command_wz": float(command[2]),
                "requested_walk_mode": walk_mode,
                **observation,
                "measured_vx": float(after.base_lin_vel_b[0]),
                "measured_vy": float(after.base_lin_vel_b[1]),
                "measured_wz": float(after.base_ang_vel_b[2]),
                "measured_planar_speed": float(np.linalg.norm(after.base_lin_vel_b[:2].astype(np.float64))),
                "target_distance_m": target_distance,
                "cross_track_error_m": float(cross_track),
                "longitudinal_error_m": float(longitudinal),
                "yaw_error_rad": float(wrap_angle(reference.yaw - yaw)),
                "map_label_at_base": int(grid.label_at(float(after.base_pos_w[0]), float(after.base_pos_w[1]))),
                "hand_x": float(hand_position[0]),
                "hand_y": float(hand_position[1]),
                "hand_z": float(hand_position[2]),
                "evaluation_hand_contact_force_norm_N": hand_contact_force_norm,
                "evaluation_geometry_snapshot_json": json.dumps(clean(geometry), separators=(",", ":")),
                "evaluation_curtain_top_drift_m": float(curtain_top_drift),
                "evaluation_curtain_anchor_drift_m": float(curtain_anchor_drift),
                "evaluation_curtain_region_displacement_m": float(curtain_region_displacement),
                "evaluation_curtain_max_speed_mps": float(curtain_max_speed),
                "evaluation_curtain_max_neighbor_stretch_ratio": float(curtain_stretch),
                "external_safety_reason": external_safety_reason or "",
                "probe_safety_stop_reason": probe_safety_stop_reason or "",
                "deployment_q_actual_json": json.dumps(q_official.tolist(), separators=(",", ":")),
                "deployment_qdot_json": json.dumps(qdot_official.tolist(), separators=(",", ":")),
                "deployment_q_cmd_json": json.dumps(target_official.tolist(), separators=(",", ":")),
                "deployment_motor_effort_json": json.dumps(effort_official.tolist(), separators=(",", ":")),
                "deployment_effort_source": str(after.effort_source),
                "deployment_previous_action_json": json.dumps(np.asarray(adapter.previous_action, dtype=np.float64).tolist(), separators=(",", ":")),
                "deployment_probe_side": ARGS.probe_side,
                "deployment_hand_fk_position_w_json": json.dumps(hand_position.tolist(), separators=(",", ":")),
                "deployment_hand_fk_velocity_w_json": json.dumps(hand_velocity.tolist(), separators=(",", ":")),
                "deployment_left_hand_fk_position_w_json": json.dumps(hand_positions["left"].tolist(), separators=(",", ":")),
                "deployment_left_hand_fk_velocity_w_json": json.dumps(hand_velocities["left"].tolist(), separators=(",", ":")),
                "deployment_right_hand_fk_position_w_json": json.dumps(hand_positions["right"].tolist(), separators=(",", ":")),
                "deployment_right_hand_fk_velocity_w_json": json.dumps(hand_velocities["right"].tolist(), separators=(",", ":")),
                "deployment_base_position_w_json": json.dumps(after.base_pos_w.astype(np.float64).tolist(), separators=(",", ":")),
                "deployment_base_velocity_b_json": json.dumps(after.base_lin_vel_b.astype(np.float64).tolist(), separators=(",", ":")),
                "deployment_localization_assumption": "Isaac root pose used only as the permitted phase-1 localization assumption",
                "deployment_progress_monitor_status": "" if progress_monitor_last_reading is None else progress_monitor_last_reading.status.value,
                "deployment_progress_monitor_stop_requested": False if progress_monitor_last_reading is None else bool(progress_monitor_last_reading.stop_requested),
                "deployment_progress_monitor_reason": "" if progress_monitor_last_reading is None else progress_monitor_last_reading.reason,
                "deployment_progress_monitor_stagnant_windows": 0 if progress_monitor_last_reading is None else int(progress_monitor_last_reading.stagnant_windows),
                "upper_tracking_rms_rad": float(np.sqrt(np.mean(np.square(target_official[15:] - q_official[15:])))),
                "effort_ratio": float(np.max(np.abs(effort_official) / np.maximum(JOINT_EFFORT_LIMIT, 1.0e-6))),
                "command_saturated": command_saturated,
                "command_rate_limited": command_rate_limited,
                "finite": finite,
                "fall": fall_reason is not None,
                "fall_reason": fall_reason or "",
                "localization_assumption": "Isaac root pose used only as the permitted phase-1 localization assumption",
                })
            sim_step += 1
            heartbeat_now = time.monotonic()
            if sim_step == 1 or heartbeat_now - last_rollout_heartbeat_wall >= 10.0:
                LOGGER.heartbeat(
                    details={
                        "scenario": ARGS.scenario,
                        "phase": phase,
                        "sim_step": sim_step,
                        "telemetry_rows": len(rows),
                    }
                )
                last_rollout_heartbeat_wall = heartbeat_now
            return after, target_distance

        def safety_blocks_phase(phase: str) -> bool:
            recovery_phase = phase.startswith(("physical_retract", "physical_neutralize", "physical_post_retract"))
            return external_safety_reason is not None and not recovery_phase

        def run_walk_to_waypoint(
            phase: str,
            target_xy: tuple[float, float],
            target_yaw: float,
            duration_s: float,
            upper_fn=None,
        ) -> tuple[bool, np.ndarray]:
            nonlocal progress_monitor_stop_requested, progress_monitor_axis
            nonlocal progress_monitor_segment_id, progress_monitor_stop_reason
            start = adapter.make_state(robot)
            start_xy_array = start.base_pos_w[:2].astype(np.float64).copy()
            target_array = np.asarray(target_xy, dtype=np.float64)
            displacement = target_array - start_xy_array
            distance = float(np.linalg.norm(displacement))
            if distance <= ARGS.position_tolerance:
                return True, start_xy_array
            direction = displacement / distance
            tracker.reset()
            if progress_monitor is not None and (
                phase == "responsive_traverse" or phase.startswith("round26_cautious_advance")
            ):
                progress_monitor_stop_requested = False
                progress_monitor_stop_reason = None
                progress_monitor_axis = direction.copy()
                progress_monitor_segment_id = f"{phase}-{sim_step}"
                progress_monitor.begin_traverse(
                    progress_axis_world=progress_monitor_axis,
                    segment_id=progress_monitor_segment_id,
                )
            reference_progress = 0.0
            profile_speed = 0.0
            command = np.zeros(3, dtype=np.float64)
            reference = SE2Reference(tuple(start_xy_array), target_yaw)
            saturated = False
            rate_limited = False
            reached = False
            final_xy = start_xy_array.copy()
            for local_step in range(int(round(duration_s / DT))):
                if fall_reason is not None or safety_blocks_phase(phase) or (
                    probe_safety_stop_reason is not None and phase.startswith("physical_probe")
                ):
                    break
                state = adapter.make_state(robot)
                target_distance_before = float(np.linalg.norm(target_array - state.base_pos_w[:2].astype(np.float64)))
                if local_step == 0 or sim_step % DECIMATION == 0:
                    remaining_reference = max(0.0, distance - reference_progress)
                    if remaining_reference > 1.0e-9:
                        terminal_limit = max(0.15, ARGS.cruise_speed * min(1.0, remaining_reference / 0.20))
                        desired_speed = min(ARGS.cruise_speed, terminal_limit)
                        if profile_speed < desired_speed:
                            profile_speed = min(desired_speed, profile_speed + 0.006)
                        else:
                            profile_speed = max(desired_speed, profile_speed - 0.010)
                        reference_progress = min(distance, reference_progress + profile_speed * CONTROL_DT)
                    else:
                        profile_speed = 0.15 if target_distance_before > ARGS.position_tolerance else 0.0
                    reference_xy = start_xy_array + direction * reference_progress
                    reference = SE2Reference(
                        (float(reference_xy[0]), float(reference_xy[1])),
                        target_yaw,
                        velocity_world=(float(direction[0] * profile_speed), float(direction[1] * profile_speed)),
                    )
                    previous = tracker._previous_command.copy() if tracker._previous_command is not None else None
                    command = tracker(
                        (float(state.base_pos_w[0]), float(state.base_pos_w[1]), yaw_from_quat(state.base_quat_w)),
                        reference,
                    )
                    saturated = bool(
                        abs(float(command[0]) - tracker_config.vx_max) < 1.0e-7
                        or abs(abs(float(command[1])) - tracker_config.vy_limit) < 1.0e-7
                        or abs(abs(float(command[2])) - tracker_config.wz_limit) < 1.0e-7
                    )
                    rate_limited = bool(
                        previous is not None
                        and np.any(np.abs(command - previous) >= np.asarray(tracker_config.max_command_step) - 1.0e-8)
                    )
                after, target_distance = execute_step(
                    phase,
                    local_step,
                    reference,
                    target_xy,
                    command,
                    walk_mode=True,
                    upper_reference=(
                        np.asarray(upper_fn(local_step * DT, duration_s), dtype=np.float32)
                        if upper_fn is not None else None
                    ),
                    command_saturated=saturated,
                    command_rate_limited=rate_limited,
                )
                final_xy = after.base_pos_w[:2].astype(np.float64).copy()
                if progress_monitor is not None and (
                    phase == "responsive_traverse" or phase.startswith("round26_cautious_advance")
                ) and progress_monitor_stop_requested:
                    events.append({
                        "event": "deployment_progress_monitor_stop",
                        "phase": phase,
                        "reason": progress_monitor_stop_reason,
                        "segment_id": progress_monitor_segment_id,
                        "time_s": sim_step * DT,
                    })
                    break
                if local_step >= int(round(0.50 / DT)) and target_distance <= ARGS.position_tolerance:
                    reached = True
                    break
            return reached, final_xy

        def run_stand(phase: str, duration_s: float) -> dict[str, Any]:
            start = adapter.make_state(robot)
            start_xy_array = start.base_pos_w[:2].astype(np.float64).copy()
            target_xy = (float(start_xy_array[0]), float(start_xy_array[1]))
            reference = SE2Reference(target_xy, yaw_from_quat(start.base_quat_w))
            dwell_required = int(math.ceil(ARGS.stop_dwell_duration / DT))
            dwell_steps = 0
            speed_gate_reached = False
            gate_time_s: float | None = None
            final_state = start
            for local_step in range(int(round(duration_s / DT))):
                if fall_reason is not None or safety_blocks_phase(phase) or (
                    probe_safety_stop_reason is not None and phase.startswith("physical_probe")
                ):
                    break
                final_state, _ = execute_step(
                    phase,
                    local_step,
                    reference,
                    target_xy,
                    np.zeros(3, dtype=np.float64),
                    walk_mode=False,
                )
                speed = float(np.linalg.norm(final_state.base_lin_vel_b[:2].astype(np.float64)))
                dwell_steps = dwell_steps + 1 if speed <= ARGS.precontact_region_speed_max else 0
                if not speed_gate_reached and dwell_steps >= dwell_required:
                    speed_gate_reached = True
                    gate_time_s = (local_step + 1) * DT
            final_xy = final_state.base_pos_w[:2].astype(np.float64).copy()
            phase_rows = [row for row in rows if row["phase"] == phase]
            actor_stand_flag_ok = bool(phase_rows) and all(
                abs(float(row["actor_obs_command_stand"])) <= 1.0e-6 for row in phase_rows
            )
            return {
                "speed_gate_reached": speed_gate_reached,
                "speed_gate_time_s": gate_time_s,
                "final_xy": final_xy,
                "final_planar_speed_mps": float(np.linalg.norm(final_state.base_lin_vel_b[:2].astype(np.float64))),
                "drift_m": float(np.linalg.norm(final_xy - start_xy_array)),
                "actor_stand_flag_ok": actor_stand_flag_ok,
            }

        def run_upper_stand(phase: str, duration_s: float, upper_fn) -> dict[str, Any]:
            start = adapter.make_state(robot)
            start_xy_array = start.base_pos_w[:2].astype(np.float64).copy()
            target_xy = (float(start_xy_array[0]), float(start_xy_array[1]))
            reference = SE2Reference(target_xy, yaw_from_quat(start.base_quat_w))
            start_hand = robot.data.body_pos_w[0, hand_body_index].detach().cpu().numpy().astype(np.float64)
            dwell_required = int(math.ceil(ARGS.stop_dwell_duration / DT))
            dwell_steps = 0
            speed_gate_reached = False
            speed_gate_time_s: float | None = None
            final_state = start
            for local_step in range(int(round(duration_s / DT))):
                if fall_reason is not None or safety_blocks_phase(phase) or (
                    probe_safety_stop_reason is not None and phase.startswith("physical_probe")
                ):
                    break
                local_time = local_step * DT
                upper_reference = np.asarray(upper_fn(local_time, duration_s), dtype=np.float32)
                final_state, _ = execute_step(
                    phase,
                    local_step,
                    reference,
                    target_xy,
                    np.zeros(3, dtype=np.float64),
                    walk_mode=False,
                    upper_reference=upper_reference,
                )
                speed = float(np.linalg.norm(final_state.base_lin_vel_b[:2].astype(np.float64)))
                dwell_steps = dwell_steps + 1 if speed <= ARGS.precontact_region_speed_max else 0
                if not speed_gate_reached and dwell_steps >= dwell_required:
                    speed_gate_reached = True
                    speed_gate_time_s = (local_step + 1) * DT
            final_xy = final_state.base_pos_w[:2].astype(np.float64).copy()
            final_hand = robot.data.body_pos_w[0, hand_body_index].detach().cpu().numpy().astype(np.float64)
            phase_rows = [row for row in rows if row["phase"] == phase]
            return {
                "final_xy": final_xy,
                "final_planar_speed_mps": float(np.linalg.norm(final_state.base_lin_vel_b[:2].astype(np.float64))),
                "speed_gate_reached": speed_gate_reached,
                "speed_gate_time_s": speed_gate_time_s,
                "base_drift_m": float(np.linalg.norm(final_xy - start_xy_array)),
                "hand_start_m": start_hand,
                "hand_end_m": final_hand,
                "hand_forward_progress_m": float(final_hand[0] - start_hand[0]),
                "hand_lateral_progress_m": float(final_hand[1] - start_hand[1]),
                "actor_stand_flag_ok": bool(phase_rows) and all(
                    abs(float(row["actor_obs_command_stand"])) <= 1.0e-6 for row in phase_rows
                ),
                "finite_no_fall": fall_reason is None and bool(phase_rows) and all(bool(row["finite"]) for row in phase_rows),
            }

        # Initial neutral settle uses the official stance flag; policy history is
        # reset only at episode reset and is preserved across every later mode switch.
        settle_result = run_stand("settle", ARGS.settle_duration)
        settled = adapter.make_state(robot)
        start_xy = (float(settled.base_pos_w[0]), float(settled.base_pos_w[1]))
        start_yaw = yaw_from_quat(settled.base_quat_w)
        if grid.label_at(*start_xy) != OracleCellLabel.FREE:
            raise RuntimeError(f"settled start is not in a zero cell: {start_xy}")
        route_comparison = workflow.compare_routes(
            start_xy=start_xy,
            goal_xy=goal_xy,
            candidate_penalty_m=0.30,
        )
        events.append({
            "event": "initial_route_comparison",
            "candidate_route": clean(route_comparison),
            "selected_route": route_comparison.selected_route,
            "candidate_route_is_not_free": route_comparison.candidate_contains_label_2,
        })
        if route_comparison.selected_route != "candidate_2":
            raise RuntimeError("initial map did not select the expected shorter candidate-2 route")

        approach_plan = planner.plan(
            grid,
            current_xy=start_xy,
            goal_xy=goal_xy,
            candidate_xy=candidate_xy,
            interaction_memory=memory,
        )
        if approach_plan.status != KnownMapPlanStatus.APPROACH_CANDIDATE or approach_plan.waypoint_xy is None:
            raise RuntimeError(f"candidate approach was not selected: {approach_plan}")
        approach_heading = math.atan2(
            approach_plan.waypoint_xy[1] - start_xy[1],
            approach_plan.waypoint_xy[0] - start_xy[0],
        )
        fixed_waypoint_reached, arrival_xy = run_walk_to_waypoint(
            "approach_candidate_2",
            tuple(map(float, approach_plan.waypoint_xy)),
            approach_heading,
            ARGS.segment_timeout,
        )
        precontact_before_region_settle = np.asarray(arrival_xy, dtype=np.float64).copy()
        precontact_stand = run_stand("precontact_explicit_stand", ARGS.stand_duration)
        precontact_actual = np.asarray(precontact_stand["final_xy"], dtype=np.float64)
        precontact_speed_mps = float(precontact_stand["final_planar_speed_mps"])
        freeze_end = precontact_actual.copy()
        freeze_drift = float(precontact_stand["drift_m"])
        precontact_region_settle_used = True
        region_pose_ok = True
        if region_enabled:
            region_pose_ok = bool(
                float(ARGS.precontact_region_x_min) <= float(precontact_actual[0]) <= float(ARGS.precontact_region_x_max)
                and abs(float(precontact_actual[1])) <= float(ARGS.precontact_region_y_half_width)
                and fall_reason is None
            )
        precontact_region_acceptance = bool(
            region_pose_ok
            and precontact_stand["speed_gate_reached"]
            and precontact_stand["actor_stand_flag_ok"]
            and precontact_speed_mps <= float(ARGS.precontact_region_speed_max)
            and fall_reason is None
        )
        reached_precontact = bool(
            precontact_region_acceptance
            and (fixed_waypoint_reached or region_enabled)
        )
        events.append({
            "event": "approach_candidate",
            "plan": clean(approach_plan),
            "reached": reached_precontact,
            "fixed_waypoint_reached": fixed_waypoint_reached,
            "actual_xy": clean(precontact_actual),
            "actual_xy_before_region_settle": clean(precontact_before_region_settle),
            "actual_planar_speed_mps": precontact_speed_mps,
            "accepted_by_geometry_region": precontact_region_acceptance,
            "region_settle_used": precontact_region_settle_used,
            "region_settle_drift_m": freeze_drift,
            "explicit_stand": clean(precontact_stand),
            "policy_history_reset": False,
        })
        if not reached_precontact or fall_reason is not None:
            raise RuntimeError("FALCON did not reach the map-generated pre-contact waypoint")
        workflow.reach_precontact()
        events.append({
            "event": "state_transition",
            "state": workflow.state.value,
            "reason": "PRECONTACT_STABLE_AND_IN_FREE_REGION",
        })

        if not memory.reserve_probe(candidate_xy, time_s=sim_step * DT, candidate_id="oracle_candidate_2"):
            raise RuntimeError("single probe reservation unexpectedly failed")
        # The following is the single physical probe path for both scenes.
        # The scene kind is not consulted until the offline evaluation/safety
        # layer; the branch below is selected only by DeploymentObservation.
        preprobe = run_upper_stand(
            "physical_preprobe",
            ARGS.preprobe_duration,
            lambda t, duration: interpolate_upper(
                np.zeros(14, dtype=np.float32), PREPROBE_UPPER, t / max(duration, 1.0e-9)
            ),
        )
        preprobe_hold = run_upper_stand(
            "physical_preprobe_hold", ARGS.preprobe_hold_duration,
            lambda _t, _duration: PREPROBE_UPPER,
        )
        pre_probe_hand = robot.data.body_pos_w[0, hand_body_index].detach().cpu().numpy().astype(np.float64)
        if curtain_deformable is not None:
            pre_probe_curtain, _, _ = read_runtime(curtain_deformable)
        probe = run_upper_stand(
            "physical_probe",
            ARGS.probe_duration,
            lambda t, duration: interpolate_upper(
                PREPROBE_UPPER, PROBE_TARGET_UPPER, t / max(duration, 1.0e-9)
            ),
        )
        probe_hold = run_upper_stand(
            "physical_probe_hold", ARGS.probe_hold_duration,
            lambda _t, _duration: PROBE_TARGET_UPPER,
        )
        if probe_safety_stop_reason is not None:
            external_safety_reason = f"PROBE_SAFE_STOP_{probe_safety_stop_reason}"
        physical_probe_rows = [
            row for row in rows if row["phase"] in ("physical_probe", "physical_probe_hold")
        ]
        probe_actual_hand_start = np.asarray(probe["hand_start_m"], dtype=np.float64)
        probe_actual_hand_end = np.asarray(probe["hand_end_m"], dtype=np.float64)
        probe_actual_forward_m = float(probe_actual_hand_end[0] - probe_actual_hand_start[0])
        depth_increment_m = TARGET_DISPLACEMENT_M
        target_hand_forward_estimate_m = TARGET_DISPLACEMENT_M
        if probe_safety_stop_reason is not None:
            probe_stop_reason = f"SAFETY_ABORT_{probe_safety_stop_reason}"
        elif probe_actual_forward_m >= 0.095:
            probe_stop_reason = "TARGET_REACHED"
        else:
            probe_stop_reason = "PROBE_INCOMPLETE"
        continued_after_contact_m = None
        contact_to_retract_s = None
        if probe_first_contact_time_s is not None and probe_first_contact_hand is not None:
            after_contact_rows = [
                row for row in physical_probe_rows
                if float(row["time_s"]) >= float(probe_first_contact_time_s)
            ]
            if after_contact_rows:
                continued_after_contact_m = max(
                    0.0,
                    max(float(row["hand_x"]) for row in after_contact_rows)
                    - float(probe_first_contact_hand[0]),
                )
            retract_rows = [row for row in rows if row["phase"] == "physical_retract"]
            if retract_rows:
                contact_to_retract_s = max(
                    0.0,
                    float(retract_rows[0]["time_s"]) - float(probe_first_contact_time_s),
                )
        curtain_deformation_m = max(
            (float(row.get("evaluation_curtain_region_displacement_m", 0.0)) for row in physical_probe_rows),
            default=0.0,
        )
        deformation_to_hand_ratio = (
            None if abs(probe_actual_forward_m) < 1.0e-6
            else float(curtain_deformation_m / abs(probe_actual_forward_m))
        )
        baseline_path = ARGS.proprio_baseline_csv.resolve()
        thresholds_path = ARGS.proprio_thresholds_json.resolve()
        baseline_history = load_deployment_history_csv(baseline_path)
        threshold_values = json.loads(thresholds_path.read_text(encoding="utf-8"))
        thresholds = ProprioceptiveThresholds(**threshold_values)
        active_history = deployment_samples.get("physical_probe", []) + deployment_samples.get("physical_probe_hold", [])
        active_effort_sources = (
            deployment_effort_sources.get("physical_probe", [])
            + deployment_effort_sources.get("physical_probe_hold", [])
        )
        if not active_effort_sources or any(source == "unavailable" for source in active_effort_sources):
            raise RuntimeError(
                "deployment effort source is unavailable during the probe; zero-filled effort must not enter a decision"
            )
        decision, features = classify_proprioceptive_response(
            active_history, baseline=baseline_history, thresholds=thresholds, sample_period_s=DT
        )
        workflow.record_probe(ProbeDecision(decision.value))
        events.append({
            "event": "state_transition",
            "state": workflow.state.value,
            "probe_decision": decision.value,
            "reason": "DEPLOYMENT_PROPRIO_DECISION",
        })
        if policy_adapter is None:
            raise RuntimeError("Round25B high-level policy adapter was not initialized")
        policy_context = PolicyContext(
            history_rows=tuple(
                {key: row.get(key, "") for key in DEPLOYMENT_FIELDS_USED}
                for row in rows
            ),
            phase="post_probe_decision",
            contact_decision=decision,
            stable=bool(
                fall_reason is None
                and precontact_stand["speed_gate_reached"]
                and precontact_stand["actor_stand_flag_ok"]
            ),
            finite=bool(fall_reason is None and all(bool(row["finite"]) for row in rows)),
            candidate_map_label=int(grid.label_at(*candidate_xy)),
            candidate_route_cost_m=float(ARGS.policy_candidate_route_cost),
            zero_only_route_cost_m=float(ARGS.policy_zero_route_cost),
            probe_budget_remaining=0,
            already_probed=True,
            progress_monitor_stop_requested=False,
            contact_features=features,
        )
        policy_decision = policy_adapter.decide(policy_context)
        policy_decisions.append(policy_decision.to_dict())
        events.append({
            "event": "high_level_policy_decision",
            "phase": "post_probe_decision",
            "policy": policy_decision.policy,
            "version": policy_decision.version,
            "valid_actions": list(policy_decision.valid_actions),
            "raw_action": policy_decision.raw_action,
            "filtered_action": policy_decision.filtered_action,
            "reason": policy_decision.reason,
            "raw_scores": policy_decision.raw_scores,
            "belief": policy_decision.belief,
            "deployment_fields_used": list(policy_decision.deployment_fields_used),
            "model_checkpoint": policy_decision.model_checkpoint,
            "model_status": policy_decision.model_status,
        })
        contact_forces = [
            float(row["evaluation_hand_contact_force_norm_N"])
            for row in physical_probe_rows
            if row.get("evaluation_hand_contact_force_norm_N") is not None
        ]
        evaluation_contact_force_max = max(contact_forces, default=None)
        evaluation_curtain_displacement = max(
            (float(row.get("evaluation_curtain_region_displacement_m", 0.0)) for row in physical_probe_rows),
            default=0.0,
        )
        evaluation_contact_confirmed = bool(
            (evaluation_contact_force_max is not None and evaluation_contact_force_max > 1.0)
            or (ARGS.scene_kind == "curtain" and evaluation_curtain_displacement >= 0.002)
        )

        responsive_traverse = None
        opening_sweep = None
        opening_hold = None
        opening_verified = False
        candidate_clearance_verified = False
        retract = None
        neutralize = None
        post_retract_stand = None
        advance_metrics_history: list[dict[str, Any]] = []
        policy_rejected_responsive = False
        round26_evidence = None
        conservative_stop = (
            ARGS.probe_validation_only
            or decision in (ContactDecision.BLOCKED, ContactDecision.UNCERTAIN)
            or probe_safety_stop_reason is not None
        )

        # Only explicit deployment blocking evidence is allowed to mutate the
        # persistent 2->1 navigation map.  Uncertain evidence remains a
        # temporary episode-local avoidance until a later physical test.
        if decision == ContactDecision.BLOCKED:
            round26_evidence = make_evidence_record(
                contact_decision=decision.value,
                policy_action=policy_decision.filtered_action,
                reason="DEPLOYMENT_PROPRIOCEPTIVE_BLOCKED_WITH_STALL_AND_EFFORT",
                progress_confirmed_blocked=True,
            )
            workflow.mark_confirmed_blocked(round26_evidence)
        elif decision == ContactDecision.UNCERTAIN:
            round26_evidence = make_evidence_record(
                contact_decision=decision.value,
                policy_action=policy_decision.filtered_action,
                reason="DEPLOYMENT_PROPRIOCEPTIVE_EVIDENCE_INSUFFICIENT",
            )
            workflow.mark_unverified_avoided(round26_evidence)
        if decision == ContactDecision.RESPONSIVE and not ARGS.probe_validation_only:
            if OPENING_ENABLED:
                opening_sweep = run_upper_stand(
                    "opening_sweep",
                    ARGS.opening_sweep_duration,
                    lambda t, duration: interpolate_upper(
                        PROBE_TARGET_UPPER,
                        OPENING_TARGET_UPPER,
                        t / max(duration, 1.0e-9),
                    ),
                )
                opening_hold = run_upper_stand(
                    "opening_hold",
                    ARGS.opening_hold_duration,
                    lambda _t, _duration: OPENING_TARGET_UPPER,
                )

            def traverse_upper(_local_time: float, _duration: float) -> np.ndarray:
                # Keep the opening reference active through every bounded
                # advance. It is never replaced by the old probe posture.
                return (OPENING_TARGET_UPPER if OPENING_ENABLED else PROBE_TARGET_UPPER).copy()

            segment_results: list[dict[str, Any]] = []
            traverse_xy = adapter.make_state(robot).base_pos_w[:2].astype(np.float64).copy()
            traverse_reached = False
            for segment_index in range(int(ARGS.max_bounded_advance_segments)):
                current_state = adapter.make_state(robot)
                current_xy = current_state.base_pos_w[:2].astype(np.float64).copy()
                remaining = np.asarray(traverse_exit_xy, dtype=np.float64) - current_xy
                remaining_distance = float(np.linalg.norm(remaining))
                if remaining_distance <= float(ARGS.position_tolerance):
                    traverse_reached = True
                    break

                if segment_index > 0:
                    advance_context = PolicyContext(
                        history_rows=tuple(
                            {key: row.get(key, "") for key in DEPLOYMENT_FIELDS_USED}
                            for row in rows
                        ),
                        phase="post_advance_decision",
                        contact_decision=decision,
                        stable=bool(fall_reason is None),
                        finite=bool(fall_reason is None and all(bool(row["finite"]) for row in rows)),
                        candidate_map_label=int(grid.label_at(*candidate_xy)),
                        candidate_route_cost_m=float(ARGS.policy_candidate_route_cost),
                        zero_only_route_cost_m=float(ARGS.policy_zero_route_cost),
                        probe_budget_remaining=0,
                        already_probed=True,
                        progress_monitor_stop_requested=bool(progress_monitor_stop_requested),
                        contact_features={**features, **advance_metrics_history[-1]},
                    )
                    policy_decision = policy_adapter.decide(
                        advance_context,
                        stage="post_advance",
                        advance_metrics=advance_metrics_history[-1],
                    )
                    policy_decisions.append(policy_decision.to_dict())
                    events.append({
                        "event": "high_level_policy_decision",
                        "phase": "post_bounded_advance",
                        "segment_index": segment_index,
                        "policy": policy_decision.policy,
                        "version": policy_decision.version,
                        "valid_actions": list(policy_decision.valid_actions),
                        "raw_action": policy_decision.raw_action,
                        "filtered_action": policy_decision.filtered_action,
                        "reason": policy_decision.reason,
                        "raw_scores": policy_decision.raw_scores,
                        "belief": policy_decision.belief,
                        "deployment_fields_used": list(policy_decision.deployment_fields_used),
                    })

                if policy_decision.filtered_action != HighLevelAction.CAUTIOUS_ADVANCE.value:
                    policy_rejected_responsive = True
                    conservative_stop = True
                    round26_evidence = make_evidence_record(
                        contact_decision=decision.value,
                        policy_action=policy_decision.filtered_action,
                        reason="POLICY_AVOIDED_WITHOUT_PHYSICAL_BLOCKING_PROOF",
                    )
                    workflow.mark_unverified_avoided(round26_evidence)
                    break

                direction = remaining / max(remaining_distance, 1.0e-9)
                segment_target = current_xy + direction * min(0.12, remaining_distance)
                segment_phase = f"round26_cautious_advance_{segment_index:02d}"
                segment_start_row = len(rows)
                segment_motion_reached, segment_xy = run_walk_to_waypoint(
                    segment_phase,
                    (float(segment_target[0]), float(segment_target[1])),
                    yaw_from_quat(current_state.base_quat_w),
                    max(ARGS.segment_timeout, 5.0),
                    upper_fn=traverse_upper,
                )
                segment_rows = rows[segment_start_row:]
                base_delta = np.asarray(segment_xy, dtype=np.float64) - current_xy
                base_progress = float(np.dot(base_delta, direction))
                commanded_progress = float(
                    sum(
                        max(
                            float(row.get("reference_vx_world", 0.0)) * float(direction[0])
                            + float(row.get("reference_vy_world", 0.0)) * float(direction[1]),
                            0.0,
                        )
                        * EVALUATION_DT
                        for row in segment_rows
                    )
                )
                tracking = max(
                    [float(row["upper_tracking_rms_rad"]) for row in segment_rows] or [999.0]
                )
                stagnant_windows = max(
                    [int(row.get("deployment_progress_monitor_stagnant_windows", 0)) for row in segment_rows] or [0]
                )
                segment_metrics = {
                    "segment_index": segment_index,
                    "base_progress_m": base_progress,
                    "commanded_progress_m": commanded_progress,
                    "advance_stagnant_windows": stagnant_windows,
                    "advance_upper_tracking_rms": tracking,
                    "segment_motion_reached": bool(segment_motion_reached),
                    "segment_target_xy": segment_target.tolist(),
                    "segment_arrival_xy": np.asarray(segment_xy).tolist(),
                    "segment_stand_speed_gate": None,
                    "segment_settle_mode": "CONTINUOUS_WALK_NO_INTERMEDIATE_STAND",
                }
                advance_metrics_history.append(segment_metrics)
                segment_results.append(segment_metrics)
                traverse_xy = np.asarray(segment_xy, dtype=np.float64)
                insufficient_partial_progress = bool(
                    not segment_motion_reached and base_progress < 0.012
                )
                if (
                    progress_monitor_stop_requested
                    or insufficient_partial_progress
                    or stagnant_windows > 0
                    or fall_reason is not None
                    or external_safety_reason is not None
                ):
                    conservative_stop = True
                    round26_evidence = make_evidence_record(
                        contact_decision=decision.value,
                        policy_action=HighLevelAction.RETRACT.value,
                        reason=(
                            "BOUNDED_ADVANCE_STOPPED_WITHOUT_BLOCKING_PROOF"
                            if not insufficient_partial_progress
                            else "BOUNDED_ADVANCE_NO_MEANINGFUL_PROGRESS"
                        ),
                    )
                    workflow.mark_unverified_avoided(round26_evidence)
                    break
                if float(np.linalg.norm(np.asarray(traverse_exit_xy) - traverse_xy)) <= float(ARGS.position_tolerance):
                    traverse_reached = True
                    break

            if not conservative_stop:
                # The independent traverse exit waypoint is a diagnostic
                # motion target, not the candidate-clearance gate.  A finite
                # bounded budget may stop before that waypoint while the
                # robot-owned body envelope has already cleared the candidate.
                # Clearance is evaluated below in CAUTIOUS_TRAVERSE; if it is
                # not verified, the normal conservative recovery path runs.
                if not conservative_stop:
                    traverse_stand = run_upper_stand(
                        "responsive_traverse_stand", ARGS.stand_duration,
                        lambda t, d: traverse_upper(d, d),
                    )
                    final_geometry = geometry_snapshot(
                        body_poses(robot), bbox_template, float(candidate_xy[0]), hand_name=ACTIVE_HAND_NAME
                    )
                    final_state = adapter.make_state(robot)
                    traverse_exit_error = float(
                        np.linalg.norm(final_state.base_pos_w[:2] - np.asarray(traverse_exit_xy))
                    )
                    candidate_clearance_verified = workflow.verify_candidate_clearance(
                        body_back_extents_x_m=final_geometry["back_x_by_body_m"].values(),
                        stable=bool(
                            traverse_stand["speed_gate_reached"]
                            and float(traverse_stand["final_planar_speed_mps"]) <= ARGS.precontact_region_speed_max
                        ),
                        finite=bool(
                            fall_reason is None
                            and external_safety_reason is None
                            and traverse_stand["finite_no_fall"]
                        ),
                    )
                    opening_verified = candidate_clearance_verified
                    if candidate_clearance_verified:
                        round26_evidence = make_evidence_record(
                            contact_decision=decision.value,
                            policy_action=HighLevelAction.CAUTIOUS_ADVANCE.value,
                            reason="ROBOT_ENVELOPE_CLEARANCE_CONFIRMED",
                        )
                        workflow.record_evidence(round26_evidence)
                    else:
                        conservative_stop = True
                        round26_evidence = make_evidence_record(
                            contact_decision=decision.value,
                            policy_action=HighLevelAction.RETRACT.value,
                            reason="TRAVERSE_CLEARANCE_NOT_VERIFIED",
                        )
                        workflow.mark_unverified_avoided(round26_evidence)
                    events.append({
                        "event": "state_transition",
                        "state": workflow.state.value,
                        "candidate_clearance_verified": candidate_clearance_verified,
                        "traverse_motion_reached": traverse_reached,
                        "reason": "ROBOT_ENVELOPE_CLEAR_OF_CANDIDATE" if candidate_clearance_verified else "TRAVERSE_CLEARANCE_NOT_VERIFIED",
                    })
                    responsive_traverse = {
                        "motion_reached": traverse_reached,
                        "arrival_xy": clean(traverse_xy),
                        "traverse_exit_xy": clean(traverse_exit_xy),
                        "traverse_exit_error_m": traverse_exit_error,
                        "stand": clean(traverse_stand),
                        "final_geometry": clean(final_geometry),
                        "candidate_clearance_verified": candidate_clearance_verified,
                        "candidate_clear_margin_m": ARGS.candidate_clear_margin,
                        "opening_verified_from_robot_geometry": opening_verified,
                        "external_safety_reason": external_safety_reason,
                        "progress_monitor_stop_requested": progress_monitor_stop_requested,
                        "progress_monitor_stop_reason": progress_monitor_stop_reason,
                        "bounded_advance_segments": segment_results,
                    }

        if conservative_stop:
            retract = run_upper_stand(
                "physical_retract", ARGS.retract_duration,
                lambda t, duration: interpolate_upper(OPENING_TARGET_UPPER if OPENING_ENABLED else PROBE_TARGET_UPPER, PREPROBE_UPPER, t / max(duration, 1.0e-9)),
            )
            neutralize = run_upper_stand(
                "physical_neutralize", ARGS.neutralize_duration,
                lambda t, duration: interpolate_upper(PREPROBE_UPPER, np.zeros(14, dtype=np.float32), t / max(duration, 1.0e-9)),
            )
            post_retract_stand = run_stand("physical_post_retract_stand", ARGS.stand_duration)
            memory_status = (
                InteractionStatus.BLOCKED
                if round26_evidence is not None
                and round26_evidence.status == Round26EvidenceStatus.CONFIRMED_BLOCKED
                else InteractionStatus.DEFERRED_THIS_EPISODE
            )
            memory.record_response(
                candidate_xy, memory_status, time_s=sim_step * DT,
                failure_reason=(
                    f"round26_{round26_evidence.status.value.lower()}"
                    if round26_evidence is not None
                    else f"deployment_proprio_rule_{decision.value}_stop_and_retract"
                ),
            )
            post_probe_xy = tuple(map(float, post_retract_stand["final_xy"]))
            if workflow.state == Round22State.RELEASE_AND_RECOVER:
                workflow.complete_release_and_recover(
                    stable=bool(post_retract_stand["speed_gate_reached"]),
                    finite=bool(post_retract_stand["actor_stand_flag_ok"] and fall_reason is None),
                )
        else:
            memory.record_response(
                candidate_xy, InteractionStatus.PASSABLE, time_s=sim_step * DT,
                failure_reason="deployment_RESPONSIVE_then_robot_geometry_opening_verified",
                opening_verified=True,
            )
            memory.mark_currently_clear(candidate_xy, time_s=sim_step * DT, opening_verified=True)
            # Physical release is its own stage.  Re-anchor only after the
            # measured retract/neutralize/stand sequence; do not reuse the
            # rejected generic goal-side correction as a substitute.
            retract = run_upper_stand(
                "responsive_release", ARGS.retract_duration,
                lambda t, duration: interpolate_upper(OPENING_TARGET_UPPER if OPENING_ENABLED else PROBE_TARGET_UPPER, PREPROBE_UPPER, t / max(duration, 1.0e-9)),
            )
            neutralize = run_upper_stand(
                "responsive_neutralize", ARGS.neutralize_duration,
                lambda t, duration: interpolate_upper(PREPROBE_UPPER, np.zeros(14, dtype=np.float32), t / max(duration, 1.0e-9)),
            )
            post_retract_stand = run_stand("responsive_release_stand", ARGS.stand_duration)
            post_probe_xy = tuple(map(float, post_retract_stand["final_xy"]))
            workflow.complete_release_and_recover(
                stable=bool(post_retract_stand["speed_gate_reached"] and fall_reason is None),
                finite=bool(post_retract_stand["actor_stand_flag_ok"] and fall_reason is None),
            )
        repeated_probe_allowed = memory.can_probe(candidate_xy, candidate_id="new_component_id")
        goal_navigation_yaw = yaw_from_quat(adapter.make_state(robot).base_quat_w)
        physical_probe_result = {
            "decision": decision.value,
            "high_level_policy": policy_decision.to_dict(),
            "policy_rejected_responsive": policy_rejected_responsive,
            "policy_action_changed_physical_branch": bool(
                policy_decision.filtered_action != (
                    HighLevelAction.CAUTIOUS_ADVANCE.value
                    if decision == ContactDecision.RESPONSIVE
                    else HighLevelAction.RETRACT.value
                )
            ),
            "features": features,
            "conservative_stop_and_retract": conservative_stop,
            "preprobe": preprobe,
            "preprobe_hold": preprobe_hold,
            "probe": probe,
            "probe_hold": probe_hold,
            "probe_depth_level": ARGS.probe_depth_level,
            "probe_depth_candidate_increment_m": depth_increment_m,
            "probe_target_hand_forward_distance_m_estimate": target_hand_forward_estimate_m,
            "probe_target_position_measurement_method": PROBE_TARGET_POSITION_MEASUREMENT_METHOD,
            "probe_target_reference_profile": PROBE_TARGET_REFERENCE_PROFILE,
            "probe_target_reference_source": PROBE_TARGET_REFERENCE_SOURCE,
            "probe_target_displacement_m": TARGET_DISPLACEMENT_M,
            "probe_direction_world": PROBE_DIRECTION_WORLD,
            "probe_actual_displacement_m": probe_actual_forward_m,
            "probe_post_contact_displacement_m_evaluation_only": continued_after_contact_m,
            "probe_stop_reason": probe_stop_reason,
            "probe_retry_count": 0,
            "probe_target_upper": PROBE_TARGET_UPPER,
            "probe_actual_hand_forward_progress_m": probe_actual_forward_m,
            "probe_actual_hand_start_m": probe_actual_hand_start,
            "probe_actual_hand_end_m": probe_actual_hand_end,
            "probe_first_contact_time_s_evaluation_only": probe_first_contact_time_s,
            "probe_first_contact_hand_m_evaluation_only": probe_first_contact_hand,
            "probe_actual_continued_forward_after_contact_m_evaluation_only": continued_after_contact_m,
            "probe_contact_to_retract_s_evaluation_only": contact_to_retract_s,
            "probe_curtain_deformation_m_evaluation_only": curtain_deformation_m,
            "probe_curtain_deformation_to_hand_forward_ratio_evaluation_only": deformation_to_hand_ratio,
            "probe_safety_stop_reason": probe_safety_stop_reason,
            "probe_safety_stop_time_s": probe_safety_stop_time_s,
            "probe_safety_limits": {
                "joint_position_margin_rad": 0.05,
                "joint_velocity_fraction": 0.95,
                "balance_roll_pitch_abs_rad": 0.45,
                "upper_tracking_rms_rad": 0.20,
                "effort_proxy_ratio": 0.85,
                "forward_stagnation_s": 0.40,
                "hand_workspace_x_m": [-0.20, 1.10],
                "hand_workspace_y_abs_m": 0.85,
                "hand_workspace_z_m": [0.30, 1.40],
                "effort_proxy_role": "normalized actuator effort diagnostic, not absolute contact force",
            },
            "responsive_traverse": responsive_traverse,
            "bounded_advance_metrics": advance_metrics_history,
            "round26_evidence": None if round26_evidence is None else round26_evidence.to_dict(),
            "round26_evidence_status": None if round26_evidence is None else round26_evidence.status.value,
            "persistent_map_update_allowed": bool(
                round26_evidence is not None and round26_evidence.map_update_allowed
            ),
            "persistent_map_update_reason": None
            if round26_evidence is None or round26_evidence.map_update_reason is None
            else round26_evidence.map_update_reason.value,
            "opening_action_contract": {
                "mode": "hold" if OPENING_ENABLED else "none",
                "enabled": bool(OPENING_ENABLED),
                "empty_scene_control": bool(ARGS.diagnostic_no_obstacle),
                "target_upper": OPENING_TARGET_UPPER,
                "sweep": opening_sweep,
                "hold": opening_hold,
                "hand_lateral_progress_m": None if opening_sweep is None else opening_sweep.get("hand_lateral_progress_m"),
                "hold_hand_lateral_progress_m": None if opening_hold is None else opening_hold.get("hand_lateral_progress_m"),
                "evaluation_contact_force_is_not_deployment_input": True,
            },
            "candidate_clearance_verified": candidate_clearance_verified,
            "retract": retract,
            "neutralize": neutralize,
            "post_retract_stand": post_retract_stand,
            "probe_side": ARGS.probe_side,
            "active_hand_name": ACTIVE_HAND_NAME,
            "bilateral_fk_logging": True,
            "baseline_csv": str(baseline_path),
            "thresholds_json": str(thresholds_path),
            "evaluation_hand_contact_force_max_N": evaluation_contact_force_max,
            "evaluation_curtain_probe_displacement_m": evaluation_curtain_displacement,
            "evaluation_hand_contact_confirmed": evaluation_contact_confirmed,
            "evaluation_contact_reporter_used_for_decision": False,
            "evaluation_condition": f"round13 {ARGS.scenario}; never passed to classifier or planner",
            "navigation_map_update_count": workflow.map_update_count,
            "navigation_map_candidate_label_after_probe": int(grid.label_at(*candidate_xy)),
            "goal_navigation_yaw_reference_rad": goal_navigation_yaw,
            "progress_monitor_enabled": bool(progress_monitor is not None),
            "progress_monitor_stop_requested": progress_monitor_stop_requested,
            "progress_monitor_stop_reason": progress_monitor_stop_reason,
        }
        report["physical_probe_result"] = clean(physical_probe_result)
        events.append({
            "event": "deployment_physical_probe",
            "decision": decision.value,
            "policy": policy_decision.to_dict(),
            "features": features,
            "responsive_traverse": clean(responsive_traverse),
            "retract_completed": bool(conservative_stop and retract is not None and fall_reason is None),
            "memory": memory.snapshot(),
            "repeated_probe_allowed": repeated_probe_allowed,
        })

        if ARGS.probe_validation_only:
            telemetry_path = run_root / "telemetry.csv"
            with telemetry_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            deployment_fields = [
                "phase", "time_s", "deployment_q_actual_json", "deployment_qdot_json", "deployment_q_cmd_json",
                "deployment_motor_effort_json", "deployment_effort_source", "deployment_previous_action_json",
                "deployment_probe_side", "deployment_hand_fk_position_w_json", "deployment_hand_fk_velocity_w_json",
                "deployment_left_hand_fk_position_w_json", "deployment_left_hand_fk_velocity_w_json",
                "deployment_right_hand_fk_position_w_json", "deployment_right_hand_fk_velocity_w_json",
                "deployment_base_position_w_json", "deployment_base_velocity_b_json", "map_label_at_base",
                "target_x", "target_y", "deployment_localization_assumption",
            ]
            deployment_path = run_root / "deployment_observation.csv"
            with deployment_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=deployment_fields)
                writer.writeheader()
                for row in rows:
                    writer.writerow({field: row.get(field, "") for field in deployment_fields})
            evaluation_path = run_root / "evaluation_ground_truth.json"
            evaluation_path.write_text(json.dumps(clean({
                "condition_label_for_offline_evaluation_only": f"round4070_d_{ARGS.scenario}",
                "scene_geometry_for_evaluation_only": config.get("scene_geometry"),
                "external_safety_reason": external_safety_reason,
                "probe_contact_force_max_N": evaluation_contact_force_max,
                "probe_curtain_runtime_displacement_m": evaluation_curtain_displacement,
                "probe_first_contact_time_s": probe_first_contact_time_s,
                "probe_actual_continued_forward_after_contact_m": continued_after_contact_m,
                "probe_contact_to_retract_s": contact_to_retract_s,
                "structural_topology": ARGS.structural_topology,
                "probe_depth_level": "PROBE_10CM",
                "role": "offline evaluation only; not passed to the probe classifier",
            }), indent=2) + "\n", encoding="utf-8")
            (run_root / "initial_oracle_map.json").write_text(json.dumps(clean(initial_oracle_map.to_dict()), indent=2) + "\n", encoding="utf-8")
            (run_root / "navigation_map.json").write_text(json.dumps(clean(grid.to_dict()), indent=2) + "\n", encoding="utf-8")
            (run_root / "oracle_map.json").write_text(json.dumps(clean(grid.to_dict()), indent=2) + "\n", encoding="utf-8")
            (run_root / "events.json").write_text(json.dumps(clean(events), indent=2) + "\n", encoding="utf-8")
            (run_root / "policy_decisions.json").write_text(json.dumps(clean(policy_decisions), indent=2) + "\n", encoding="utf-8")
            validation_safe_recovery = bool(
                retract is not None
                and neutralize is not None
                and post_retract_stand is not None
                and post_retract_stand["speed_gate_reached"]
                and post_retract_stand["actor_stand_flag_ok"]
                and fall_reason is None
            )
            validation_pass = bool(
                reached_precontact
                and fall_reason is None
                and external_safety_reason is None
                and validation_safe_recovery
                and probe_stop_reason in {"TARGET_REACHED", "PROBE_INCOMPLETE"}
            )
            report.update({
                "status": "PASS_PROBE_VALIDATION" if validation_pass else "FAIL_PROBE_VALIDATION",
                "runtime_status": "RUNTIME_PASS_FINALIZE_REACHED",
                "probe_validation_only": True,
                "scenario": ARGS.scenario,
                "probe_side": ARGS.probe_side,
                "probe_depth_level": "PROBE_10CM",
                "precontact_reached": reached_precontact,
                "precontact_actual_xy_m": clean(precontact_actual),
                "precontact_region_acceptance": precontact_region_acceptance,
                "probe_target_displacement_m": TARGET_DISPLACEMENT_M,
                "probe_actual_displacement_m": probe_actual_forward_m,
                "probe_post_contact_displacement_m_evaluation_only": continued_after_contact_m,
                "probe_stop_reason": probe_stop_reason,
                "probe_retry_count": 0,
                "probe_safety_stop_reason": probe_safety_stop_reason,
                "external_safety_reason": external_safety_reason,
                "safe_recovery": validation_safe_recovery,
                "physical_branch_pass": validation_safe_recovery,
                "candidate_clearance_verified": None,
                "whole_body_candidate_clearance": None,
                "task_success": False,
                "pointgoal_success": False,
                "physical_probe_result": clean(physical_probe_result),
                "telemetry_csv": str(telemetry_path),
                "deployment_observation_csv": str(deployment_path),
                "evaluation_ground_truth_json": str(evaluation_path),
                "events_json": str(run_root / "events.json"),
                "policy_decisions_json": str(run_root / "policy_decisions.json"),
                "round4070_runner_version": "ROUND4070_FIXED10CM_JUDGMENT_RUNNER_V1",
                "notes": ["Probe-validation-only output is a safety gate and is excluded from formal passability metrics.", "No post-probe navigation or whole-body passability label is produced."],
            })
            (run_root / "summary.json").write_text(json.dumps(clean(report), indent=2) + "\n", encoding="utf-8")
            final_event = LOGGER.finalize(status="PASS" if validation_pass else "FAIL", details={"scenario": ARGS.scenario, "probe_validation_only": True})
            report["stage_log"] = LOGGER.summary()
            report["finalize_event"] = clean(final_event)
            (run_root / "summary.json").write_text(json.dumps(clean(report), indent=2) + "\n", encoding="utf-8")
            print(json.dumps(clean(report), indent=2))
            return 0 if validation_pass else 2

        if conservative_stop:
            detour_plan = planner.plan(
                grid,
                current_xy=post_probe_xy,
                goal_xy=goal_xy,
                candidate_xy=candidate_xy,
                interaction_memory=memory,
                probe_decision="BLOCKED",
                opening_verified=False,
            )
            if detour_plan.status in (KnownMapPlanStatus.DIRECT, KnownMapPlanStatus.GOAL_REACHED):
                # A conservative stop can occur after an external safety
                # monitor has already halted the robot on the goal-side of
                # the candidate.  No artificial detour is required there.
                detour_path = list(detour_plan.path_xy)
                simplified_path = []
            else:
                if detour_plan.status != KnownMapPlanStatus.DETOUR or not detour_plan.path_xy:
                    raise RuntimeError(f"zero-cell detour was not produced: {detour_plan}")
                detour_path = list(detour_plan.path_xy)
                simplified_path = axis_aligned_zero_route(grid, post_probe_xy, goal_xy)
                if len(simplified_path) < 2:
                    simplified_path = simplify_free_path(grid, detour_path)
                if not all(grid.segment_is_free(a, b) for a, b in zip(simplified_path, simplified_path[1:])):
                    raise RuntimeError("simplified detour contains a non-zero segment")
        else:
            # The candidate-clear gate is independent from the final goal
            # gate.  Replan from the measured post-release pose; this direct
            # segment is allowed only after the robot envelope has cleared
            # the candidate normal extent.
            detour_plan = planner.plan(
                grid,
                current_xy=post_probe_xy,
                goal_xy=goal_xy,
                candidate_xy=candidate_xy,
                interaction_memory=memory,
                probe_decision=decision.value,
                opening_verified=False,
            )
            if detour_plan.status == KnownMapPlanStatus.GOAL_REACHED and ARGS.engineering_fix_accept_goal_reached:
                detour_path = list(detour_plan.path_xy)
                simplified_path = []
            elif (
                detour_plan.status == KnownMapPlanStatus.GOAL_REACHED
                and ARGS.research_feasibility_post_clearance_replan
                and decision == ContactDecision.RESPONSIVE
                and candidate_clearance_verified
            ):
                # The planner's tolerance is a waypoint-selection shortcut,
                # not the final PointGoal success gate.  After deployment-owned
                # candidate clearance, continue with an explicit measured-pose
                # to goal segment so the final XY/stability checks still run.
                post_goal_labels = grid.segment_labels(post_probe_xy, goal_xy)
                fixed_obstacle_on_segment = any(
                    label == OracleCellLabel.FIXED_OBSTACLE for label in post_goal_labels
                )
                goal_side_pose = bool(
                    float(post_probe_xy[0]) >= float(candidate_region.x_limits_m[1] + candidate_region.clear_margin_m)
                )
                if fixed_obstacle_on_segment or not goal_side_pose:
                    raise RuntimeError("goal-side path was not available after candidate clearance")
                detour_path = [tuple(map(float, post_probe_xy)), tuple(map(float, goal_xy))]
                simplified_path = list(detour_path)
                detour_plan = KnownMapPlan(
                    status=KnownMapPlanStatus.DIRECT,
                    waypoint_xy=tuple(map(float, goal_xy)),
                    path_xy=tuple(detour_path),
                    reason="RESEARCH_FEASIBILITY_POST_CLEARANCE_GOAL_REACHED_REPLAN",
                    encountered_labels=tuple(post_goal_labels),
                    probe_allowed=False,
                )
            elif detour_plan.status == KnownMapPlanStatus.DIRECT:
                detour_path = list(detour_plan.path_xy)
                simplified_path = [tuple(map(float, post_probe_xy)), tuple(map(float, goal_xy))]
            elif detour_plan.status == KnownMapPlanStatus.DETOUR and detour_plan.path_xy:
                detour_path = list(detour_plan.path_xy)
                simplified_path = axis_aligned_zero_route(grid, post_probe_xy, goal_xy)
            elif (
                detour_plan.status == KnownMapPlanStatus.UNRESOLVED
                and ARGS.research_feasibility_post_clearance_replan
                and decision == ContactDecision.RESPONSIVE
                and candidate_clearance_verified
            ):
                # The legacy planner can retain a candidate-2 sample at the
                # boundary after the body envelope has already cleared it.
                # This branch is independent and guarded by deployment-owned
                # geometry plus the absence of a fixed label-1 cell; it never
                # reads object identity, cloth nodes, contact truth or outcome.
                post_goal_labels = grid.segment_labels(post_probe_xy, goal_xy)
                fixed_obstacle_on_segment = any(
                    label == OracleCellLabel.FIXED_OBSTACLE for label in post_goal_labels
                )
                goal_side_pose = bool(
                    float(post_probe_xy[0]) >= float(candidate_region.x_limits_m[1] + candidate_region.clear_margin_m)
                )
                if fixed_obstacle_on_segment or not goal_side_pose:
                    raise RuntimeError("goal-side path was not available after candidate clearance")
                detour_path = [tuple(map(float, post_probe_xy)), tuple(map(float, goal_xy))]
                simplified_path = list(detour_path)
                detour_plan = KnownMapPlan(
                    status=KnownMapPlanStatus.DIRECT,
                    waypoint_xy=tuple(map(float, goal_xy)),
                    path_xy=tuple(detour_path),
                    reason="RESEARCH_FEASIBILITY_POST_CLEARANCE_DIRECT_REPLAN",
                    encountered_labels=tuple(post_goal_labels),
                    probe_allowed=False,
                )
            else:
                raise RuntimeError("goal-side path was not available after candidate clearance")
            if not all(grid.segment_is_free(a, b) for a, b in zip(simplified_path, simplified_path[1:])):
                raise RuntimeError("goal-side path contains a non-zero or unresolved segment")
        events.append({
            "event": "detour_plan",
            "plan": clean(detour_plan),
            "simplified_path_xy": clean(simplified_path),
            "navigation_map_label_counts": {
                str(int(label)): count for label, count in grid.label_counts().items()
            },
        })
        events.append({
            "event": "state_transition",
            "state": workflow.state.value,
            "reason": "RELEASED_AND_REFERENCE_REANCHORED",
        })

        # The conservative branch starts with a planned detour, so its
        # completion flag must be optimistic and be cleared only when a
        # waypoint attempt fails.  Initializing it to False made an entirely
        # successful zero-cell detour report a false negative.
        reached_detour = True
        current_xy = post_probe_xy
        executed_waypoints: list[tuple[float, float]] = []
        detour_stops: list[dict[str, Any]] = []
        route_prefix = "detour_waypoint" if conservative_stop else "navigate_to_goal"
        attempt_limit = 2 if conservative_stop else 1
        # Intermediate detour waypoints use the tighter local acceptance gate.
        # The final goal has an independent planner-level tolerance; reusing
        # the intermediate gate here can reject a stable, valid goal arrival.
        waypoint_acceptance_tolerance = (
            ARGS.position_tolerance
            if conservative_stop
            else max(ARGS.position_tolerance, planner.goal_tolerance_m)
        )
        for index, waypoint in enumerate(simplified_path[1:], start=1):
            if float(np.linalg.norm(np.asarray(waypoint) - np.asarray(current_xy))) <= 0.05:
                continue
            executed_waypoints.append(tuple(map(float, waypoint)))
            attempts: list[dict[str, Any]] = []
            reached = False
            for attempt in range(attempt_limit):
                if attempt > 0 and not grid.segment_is_free(current_xy, waypoint):
                    break
                if conservative_stop:
                    heading = math.atan2(waypoint[1] - current_xy[1], waypoint[0] - current_xy[0])
                else:
                    # The goal-side segment is almost longitudinal.  Preserve
                    # the measured post-release yaw and let SE(2) lateral
                    # feedback correct the small y error; do not turn the
                    # whole robot toward a short diagonal waypoint.
                    heading = goal_navigation_yaw
                phase = f"{route_prefix}_{index:02d}" if attempt == 0 else f"{route_prefix}_{index:02d}_bounded_retry"
                motion_reached, arrival_xy = run_walk_to_waypoint(
                    phase,
                    tuple(map(float, waypoint)),
                    heading,
                    ARGS.segment_timeout,
                )
                stop = run_stand(f"{phase}_stand", ARGS.stand_duration)
                actual_xy = np.asarray(stop["final_xy"], dtype=np.float64)
                post_stand_error = float(np.linalg.norm(actual_xy - np.asarray(waypoint, dtype=np.float64)))
                reached = bool(
                    post_stand_error <= waypoint_acceptance_tolerance
                    and stop["speed_gate_reached"]
                    and stop["actor_stand_flag_ok"]
                    and float(stop["final_planar_speed_mps"]) <= ARGS.precontact_region_speed_max
                )
                attempts.append({
                    "attempt": attempt + 1,
                    "motion_reached_before_stand": motion_reached,
                    "arrival_xy": clean(arrival_xy),
                    "post_stand_error_m": post_stand_error,
                    "stand": clean(stop),
                    "accepted": reached,
                })
                current_xy = (float(actual_xy[0]), float(actual_xy[1]))
                if reached or fall_reason is not None or external_safety_reason is not None:
                    break
            detour_stops.append({
                "waypoint_xy": clean(waypoint),
                "attempts": clean(attempts),
                "accepted": reached,
            })
            if not reached or fall_reason is not None or external_safety_reason is not None:
                reached_detour = False
                break

        if workflow.state != Round22State.NAVIGATE_TO_GOAL:
            raise RuntimeError(f"goal navigation started from invalid state: {workflow.state.value}")
        final_state = adapter.make_state(robot)
        final_xy = final_state.base_pos_w[:2].astype(np.float64)
        goal_error = float(np.linalg.norm(final_xy - np.asarray(goal_xy, dtype=np.float64)))
        final_yaw = yaw_from_quat(final_state.base_quat_w)
        final_goal_yaw_error = abs(wrap_angle(goal_navigation_yaw - final_yaw))
        finite_no_fall = (
            fall_reason is None
            and external_safety_reason is None
            and bool(rows)
            and all(bool(row["finite"]) for row in rows)
        )
        path_labels = [
            int(grid.label_at(float(row["base_x"]), float(row["base_y"])))
            for row in rows
            if row["phase"].startswith(("detour_waypoint", "navigate_to_goal"))
        ]
        actual_route_never_entered_candidate = all(label == int(OracleCellLabel.FREE) for label in path_labels)
        final_stop_phases = [
            str(row["phase"]) for row in rows if str(row["phase"]).endswith("_stand")
        ]
        final_stop_phase = final_stop_phases[-1] if final_stop_phases else None
        goal_speed_samples = [
            float(row["measured_planar_speed"])
            for row in rows
            if final_stop_phase is not None and row["phase"] == final_stop_phase
        ]
        if not goal_speed_samples:
            goal_speed_samples = [float(np.linalg.norm(final_state.base_lin_vel_b[:2].astype(np.float64)))]
        pointgoal_eval = evaluate_goal(
            final_xy=final_xy,
            goal_xy=goal_xy,
            speed_samples_mps=goal_speed_samples,
            speed_sample_period_s=EVALUATION_DT,
            finite_no_fall=finite_no_fall,
            external_safety_reason=external_safety_reason,
            final_yaw_rad=final_yaw,
            goal_yaw_rad=goal_navigation_yaw,
            config=PointGoalConfig(
                position_tolerance_m=float(ARGS.position_tolerance),
                stop_speed_threshold_mps=float(ARGS.precontact_region_speed_max),
                stop_dwell_duration_s=float(ARGS.stop_dwell_duration),
                yaw_tolerance_rad=float(ARGS.goal_yaw_tolerance),
            ),
        )
        pointgoal_success = bool(reached_detour and pointgoal_eval.point_goal_pass)
        safety_success = bool(pointgoal_eval.finite_no_fall_pass and pointgoal_eval.safety_pass)
        responsive_continue_branch = bool(
            physical_probe_result["decision"] == ContactDecision.RESPONSIVE.value
            and not conservative_stop
            and not policy_rejected_responsive
        )
        if responsive_continue_branch:
            physical_branch_pass = bool(
                physical_probe_result["evaluation_hand_contact_confirmed"]
                and physical_probe_result["responsive_traverse"] is not None
                and physical_probe_result["responsive_traverse"]["opening_verified_from_robot_geometry"]
                and physical_probe_result["retract"] is not None
                and physical_probe_result["neutralize"] is not None
                and physical_probe_result["post_retract_stand"] is not None
                and physical_probe_result["post_retract_stand"]["speed_gate_reached"]
            )
        else:
            physical_branch_pass = bool(
                (
                    physical_probe_result["decision"] in (ContactDecision.BLOCKED.value, ContactDecision.UNCERTAIN.value)
                    or policy_rejected_responsive
                    or conservative_stop
                )
                and physical_probe_result["conservative_stop_and_retract"]
                and physical_probe_result["retract"] is not None
                and physical_probe_result["neutralize"] is not None
                and physical_probe_result["post_retract_stand"] is not None
                and physical_probe_result["retract"]["finite_no_fall"]
                and physical_probe_result["neutralize"]["finite_no_fall"]
                and physical_probe_result["post_retract_stand"]["speed_gate_reached"]
                and physical_probe_result["post_retract_stand"]["actor_stand_flag_ok"]
                and (
                    physical_probe_result["evaluation_hand_contact_confirmed"]
                    if physical_probe_result["decision"] != ContactDecision.RESPONSIVE.value
                    else True
                )
            )
        safe_recovery = bool(
            physical_probe_result["retract"] is not None
            and physical_probe_result["neutralize"] is not None
            and physical_probe_result["post_retract_stand"] is not None
            and physical_probe_result["post_retract_stand"]["speed_gate_reached"]
            and physical_probe_result["post_retract_stand"]["actor_stand_flag_ok"]
            and fall_reason is None
        )
        whole_body_candidate_clearance = bool(candidate_clearance_verified)
        goal_stable = bool(pointgoal_success)
        if goal_stable:
            workflow.reach_point_goal(pointgoal_eval)
        control_reference_success = bool(
            ARGS.diagnostic_no_obstacle
            and pointgoal_success
            and safety_success
            and reached_precontact
            and freeze_drift <= 0.08
            and actual_route_never_entered_candidate
            and not repeated_probe_allowed
            and workflow.state == Round22State.GOAL_REACHED
        )
        task_success = bool(
            control_reference_success
            or (
                not ARGS.diagnostic_no_obstacle
                and pointgoal_success
                and safety_success
                and reached_precontact
                and freeze_drift <= 0.08
                and actual_route_never_entered_candidate
                and not repeated_probe_allowed
                and physical_branch_pass
                and (whole_body_candidate_clearance if responsive_continue_branch else safe_recovery)
                and workflow.state == Round22State.GOAL_REACHED
            )
        )
        passed = task_success

        telemetry_path = run_root / "telemetry.csv"
        with telemetry_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        deployment_fields = [
            "phase",
            "time_s",
            "deployment_q_actual_json",
            "deployment_qdot_json",
            "deployment_q_cmd_json",
            "deployment_motor_effort_json",
            "deployment_effort_source",
            "deployment_previous_action_json",
            "deployment_probe_side",
            "deployment_hand_fk_position_w_json",
            "deployment_hand_fk_velocity_w_json",
            "deployment_left_hand_fk_position_w_json",
            "deployment_left_hand_fk_velocity_w_json",
            "deployment_right_hand_fk_position_w_json",
            "deployment_right_hand_fk_velocity_w_json",
            "deployment_base_position_w_json",
            "deployment_base_velocity_b_json",
            "map_label_at_base",
            "target_x",
            "target_y",
            "deployment_localization_assumption",
        ]
        deployment_path = run_root / "deployment_observation.csv"
        with deployment_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=deployment_fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in deployment_fields})
        evaluation_truth = {
            "condition_label_for_offline_evaluation_only": f"round22_{ARGS.scenario}",
            "scene_geometry_for_evaluation_only": config.get("scene_geometry"),
            "external_safety_reason": external_safety_reason,
            "probe_contact_force_max_N": evaluation_contact_force_max,
            "probe_curtain_runtime_displacement_m": evaluation_curtain_displacement,
            "probe_first_contact_time_s": probe_first_contact_time_s,
            "probe_actual_continued_forward_after_contact_m": continued_after_contact_m,
            "probe_contact_to_retract_s": contact_to_retract_s,
            "structural_topology": ARGS.structural_topology,
            "probe_depth_level": ARGS.probe_depth_level,
            "role": "scene setup and offline evaluation only; never passed to the proprio classifier, memory, planner or locomotion controller",
        }
        evaluation_path = run_root / "evaluation_ground_truth.json"
        evaluation_path.write_text(json.dumps(clean(evaluation_truth), indent=2) + "\n", encoding="utf-8")
        if curtain_snapshots:
            np.savez_compressed(
                run_root / "curtain_runtime_snapshots.npz",
                curtain_snapshots=np.asarray(curtain_snapshots),
                curtain_snapshot_times=np.asarray(curtain_snapshot_times),
                runtime_rest_positions=np.asarray(runtime_rest),
                top_indices=np.asarray(top_indices, dtype=np.int64),
                requested_anchor_indices=np.asarray(anchor_indices, dtype=np.int64),
                requested_fixed_indices=np.asarray(
                    [] if curtain_info is None else curtain_info.requested_fixed_vertex_indices,
                    dtype=np.int64,
                ),
            )
        (run_root / "initial_oracle_map.json").write_text(
            json.dumps(clean(initial_oracle_map.to_dict()), indent=2) + "\n", encoding="utf-8"
        )
        (run_root / "navigation_map.json").write_text(
            json.dumps(clean(grid.to_dict()), indent=2) + "\n", encoding="utf-8"
        )
        # Keep the historical filename as an alias for downstream readers;
        # its contents are the episode's updated navigation map.
        (run_root / "oracle_map.json").write_text(json.dumps(clean(grid.to_dict()), indent=2) + "\n", encoding="utf-8")
        (run_root / "events.json").write_text(json.dumps(clean(events), indent=2) + "\n", encoding="utf-8")
        (run_root / "policy_decisions.json").write_text(json.dumps(clean(policy_decisions), indent=2) + "\n", encoding="utf-8")
        report.update({
            "status": (
                "PASS_ROUND27_CONTROL_REFERENCE_EMPTY"
                if control_reference_success
                else f"PASS_ROUND27_ENGINEERING_FIX_{ARGS.scenario.upper()}_EPISODE"
                if passed and ARGS.engineering_fix_accept_goal_reached
                else f"FAIL_ROUND27_ENGINEERING_FIX_{ARGS.scenario.upper()}_EPISODE"
                if ARGS.engineering_fix_accept_goal_reached
                else f"PASS_ROUND27_PROTOCOL_{ARGS.scenario.upper()}_EPISODE"
                if passed
                else f"FAIL_ROUND27_PROTOCOL_{ARGS.scenario.upper()}_EPISODE"
            ),
            "formal_baseline_status": (
                "ENGINEERING_FIX_EXPERIMENT_NOT_IN_FROZEN_RATE"
                if ARGS.engineering_fix_accept_goal_reached
                else "ROUND25B_POLICY_POINTGOAL_PHYSICAL_CASE"
            ),
            "frozen_baseline_parameters": not ARGS.engineering_fix_accept_goal_reached,
            "engineering_fix_experiment": bool(ARGS.engineering_fix_accept_goal_reached),
            "scenario": ARGS.scenario,
            "round25b_policy": ARGS.high_level_policy,
            "round27_policy": ARGS.high_level_policy,
            "policy_checkpoint": None if policy_adapter is None or policy_adapter.checkpoint is None else str(policy_adapter.checkpoint),
            "policy_decisions_json": str(run_root / "policy_decisions.json"),
            "policy_decision_count": len(policy_decisions),
            "policy_action_changed_physical_branch": bool(
                policy_decisions
                and report.get("physical_probe_result", {}).get("policy_action_changed_physical_branch", False)
            ),
            "round26_evidence": None if round26_evidence is None else round26_evidence.to_dict(),
            "round26_evidence_status": None if round26_evidence is None else round26_evidence.status.value,
            "round27_runner_version": "ROUND4070_FIXED10CM_JUDGMENT_RUNNER_V1",
            "round4070_runner_version": "ROUND4070_FIXED10CM_JUDGMENT_RUNNER_V1",
            "deployment_effort_sources_observed": sorted(
                {source for sources in deployment_effort_sources.values() for source in sources}
            ),
            "round26_state_machine_reused": "ROUND26_DYNAMIC_POLICY_IN_THE_LOOP_V1",
            "navigation_map_update_count": int(workflow.map_update_count),
            "bounded_advance_metrics": advance_metrics_history,
            "probe_side": ARGS.probe_side,
            "active_hand_name": ACTIVE_HAND_NAME,
            "probe_depth_level": ARGS.probe_depth_level,
            "probe_target_upper": PROBE_TARGET_UPPER.tolist(),
            "probe_target_reference_profile": PROBE_TARGET_REFERENCE_PROFILE,
            "probe_target_reference_source": PROBE_TARGET_REFERENCE_SOURCE,
            "structural_topology": ARGS.structural_topology,
            "surface_thickness_m": float(ARGS.surface_thickness),
            "opening_action_contract": {
                "mode": "hold" if OPENING_ENABLED else "none",
                "enabled": bool(OPENING_ENABLED),
                "empty_scene_control": bool(ARGS.diagnostic_no_obstacle),
                "target_upper": OPENING_TARGET_UPPER.tolist(),
                "sweep_duration_s": float(ARGS.opening_sweep_duration),
                "hold_duration_s": float(ARGS.opening_hold_duration),
                "direction_sign": float(ARGS.opening_direction_sign),
            },
            "runtime_status": "RUNTIME_PASS_FINALIZE_REACHED",
            "start_xy_m": list(start_xy),
            "start_yaw_rad": start_yaw,
            "approach_plan": clean(approach_plan),
            "precontact_reached": reached_precontact,
            "precontact_actual_xy_m": clean(precontact_actual),
            "precontact_fixed_waypoint_reached": fixed_waypoint_reached,
            "precontact_region_acceptance": precontact_region_acceptance,
            "precontact_actual_planar_speed_mps": precontact_speed_mps,
            "precontact_region_settle_used": precontact_region_settle_used,
            "freeze_drift_m": freeze_drift,
            "probe_memory": memory.snapshot(),
            "repeated_probe_allowed": repeated_probe_allowed,
            "detour_plan": clean(detour_plan),
            "simplified_detour_path_xy_m": clean(simplified_path),
            "executed_waypoints_xy_m": clean(executed_waypoints),
            "detour_stops": clean(detour_stops),
            "detour_waypoints_reached": reached_detour,
            "actual_detour_never_entered_candidate": actual_route_never_entered_candidate,
            "actual_route_never_entered_candidate": actual_route_never_entered_candidate,
            "physical_branch_pass": physical_branch_pass,
            "control_reference_success": control_reference_success,
            "task_success": task_success,
            "pointgoal_success": pointgoal_success,
            "pointgoal_evaluation": clean(pointgoal_eval),
            "posegoal_diagnostic_pass": bool(pointgoal_eval.pose_goal_pass),
            "safety_success": safety_success,
            "safe_recovery": safe_recovery,
            "whole_body_candidate_clearance": whole_body_candidate_clearance,
            "motion_reached": None if responsive_traverse is None else bool(responsive_traverse["motion_reached"]),
            "opening_mode": "hold" if OPENING_ENABLED else "none",
            "physical_probe_result": clean(physical_probe_result),
            "final_xy_m": clean(final_xy),
            "goal_error_m": goal_error,
            "finite_no_fall": finite_no_fall,
            "fall_reason": fall_reason,
            "external_safety_reason": external_safety_reason,
            "probe_safety_stop_reason": probe_safety_stop_reason,
            "probe_safety_stop_time_s": probe_safety_stop_time_s,
            "probe_actual_hand_forward_progress_m": probe_actual_forward_m,
            "probe_target_hand_forward_distance_m_estimate": target_hand_forward_estimate_m,
            "probe_target_displacement_m": TARGET_DISPLACEMENT_M,
            "probe_actual_displacement_m": probe_actual_forward_m,
            "probe_post_contact_displacement_m_evaluation_only": continued_after_contact_m,
            "probe_stop_reason": probe_stop_reason,
            "probe_retry_count": 0,
            "probe_actual_continued_forward_after_contact_m_evaluation_only": continued_after_contact_m,
            "probe_contact_to_retract_s_evaluation_only": contact_to_retract_s,
            "probe_curtain_deformation_m_evaluation_only": curtain_deformation_m,
            "probe_curtain_deformation_to_hand_forward_ratio_evaluation_only": deformation_to_hand_ratio,
            "scene_kind": ARGS.scene_kind,
            "opening_verified_from_robot_geometry": opening_verified,
            "candidate_clearance_verified": candidate_clearance_verified,
            "traverse_exit_xy_m": list(traverse_exit_xy),
            "traverse_exit_is_final_goal": False,
            "goal_stable_gate": goal_stable,
            "final_yaw_rad": final_yaw,
            "goal_yaw_reference_rad": goal_navigation_yaw,
            "final_goal_yaw_error_rad": final_goal_yaw_error,
            "goal_yaw_tolerance_rad": ARGS.goal_yaw_tolerance,
            "goal_yaw_role": "diagnostic_only_never_a_pointgoal_gate",
            "round22_state_machine": workflow.snapshot(),
            "round22_state_machine_snapshot": workflow.snapshot(),
            "round26_state_machine": workflow.snapshot(),
            "navigation_map_update_count": int(workflow.map_update_count),
            "initial_oracle_map_json": str(run_root / "initial_oracle_map.json"),
            "navigation_map_json": str(run_root / "navigation_map.json"),
            "command_saturation_ratio": sum(bool(row["command_saturated"]) for row in rows) / max(1, len(rows)),
            "max_effort_ratio": max(float(row["effort_ratio"]) for row in rows),
            "max_upper_tracking_rms_rad": max(float(row["upper_tracking_rms_rad"]) for row in rows),
            "max_curtain_top_drift_m": max(float(row["evaluation_curtain_top_drift_m"]) for row in rows),
            "max_curtain_anchor_drift_m": max(float(row["evaluation_curtain_anchor_drift_m"]) for row in rows),
            "max_curtain_region_displacement_m": max(float(row["evaluation_curtain_region_displacement_m"]) for row in rows),
            "max_curtain_speed_mps": max(float(row["evaluation_curtain_max_speed_mps"]) for row in rows),
            "max_curtain_neighbor_stretch_ratio": max(float(row["evaluation_curtain_max_neighbor_stretch_ratio"]) for row in rows),
            "policy_update_count": len(latencies),
            "policy_latency_p50_ms": float(np.percentile(np.asarray(latencies) * 1000.0, 50)) if latencies else None,
            "policy_latency_p95_ms": float(np.percentile(np.asarray(latencies) * 1000.0, 95)) if latencies else None,
            "telemetry_csv": str(telemetry_path),
            "deployment_observation_csv": str(deployment_path),
            "evaluation_ground_truth_json": str(evaluation_path),
            "events_json": str(run_root / "events.json"),
            "oracle_map_json": str(run_root / "oracle_map.json"),
            "notes": [
                "Round 13 scenario identity is used only by scene construction and offline evaluation; deployment branch selection uses the proprioceptive history.",
                "Round 21 opening diagnostic adds only a bounded shoulder-yaw reference after a deployment RESPONSIVE decision; no simulator contact, object label, cloth node or passability truth enters the action path.",
                "The opening sweep/hold is a mechanism diagnostic and does not establish whole-body passability or a learned side-selection policy.",
                "BLOCKED and UNCERTAIN trigger bounded retract and consume the one-probe episode budget.",
                "RESPONSIVE is not treated as PASSABLE until the robot's own body envelope verifies a complete opening and goal-side passage.",
                "The persistent candidate map region is updated from 2 to 1 only after CONFIRMED_BLOCKED; UNVERIFIED_AVOIDED uses a temporary route exclusion and remains 2 in the episode map.",
                "Candidate clearance and final goal arrival are independent gates; the traverse exit waypoint is not the final goal.",
                "Simulator root pose is explicitly the permitted phase-1 localization assumption.",
                "Round25B high-level actions are option-level only; FALCON, SE(2), the phase monitor and external safety monitors remain independent.",
                "A/B checkpoints are surrogate-trained artifacts and their physical decisions are reported separately from offline metrics.",
                "Round4070 records the actuator effort source explicitly and refuses to classify a probe if the adapter reports unavailable zero-filled effort.",
            ],
        })
        (run_root / "summary.json").write_text(json.dumps(clean(report), indent=2) + "\n", encoding="utf-8")
        final_event = LOGGER.finalize(
            status="PASS" if passed else "FAIL",
            details={"scenario": ARGS.scenario, "episode_status": report["status"]},
        )
        report["stage_log"] = LOGGER.summary()
        report["finalize_event"] = clean(final_event)
        (run_root / "summary.json").write_text(json.dumps(clean(report), indent=2) + "\n", encoding="utf-8")
        print(json.dumps(clean(report), indent=2))
        return 0 if passed else 2
    except Exception as exc:
        report.update({
            "status": "BLOCKED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "events": clean(events),
        })
        if rows:
            telemetry_path = run_root / "telemetry_partial.csv"
            with telemetry_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            report["partial_telemetry_csv"] = str(telemetry_path)
        (run_root / "initial_oracle_map.json").write_text(
            json.dumps(clean(initial_oracle_map.to_dict()), indent=2) + "\n", encoding="utf-8"
        )
        (run_root / "navigation_map.json").write_text(
            json.dumps(clean(grid.to_dict()), indent=2) + "\n", encoding="utf-8"
        )
        (run_root / "oracle_map.json").write_text(json.dumps(clean(grid.to_dict()), indent=2) + "\n", encoding="utf-8")
        report["round22_state_machine"] = workflow.snapshot()
        (run_root / "policy_decisions.json").write_text(json.dumps(clean(policy_decisions), indent=2) + "\n", encoding="utf-8")
        report["policy_decisions_json"] = str(run_root / "policy_decisions.json")
        (run_root / "summary.json").write_text(json.dumps(clean(report), indent=2) + "\n", encoding="utf-8")
        final_event = LOGGER.finalize(
            status="FAIL",
            details={
                "scenario": ARGS.scenario,
                "episode_status": report["status"],
                "error_type": type(exc).__name__,
            },
        )
        report["stage_log"] = LOGGER.summary()
        report["finalize_event"] = clean(final_event)
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

