"""Explicit separation between deployable observations and simulator truth.

The contact decision in this module intentionally consumes only the signals a
real robot controller can obtain from joint/actuator state, its own kinematics,
and an explicitly documented localization assumption.  Simulator contact,
object identity, cloth nodes, and passability are represented by a separate
dataclass for offline evaluation only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Iterable, Sequence

import numpy as np


class ContactDecision(str, Enum):
    """Conservative decision available to the deployment state machine."""

    RESPONSIVE = "RESPONSIVE"
    BLOCKED = "BLOCKED"
    UNCERTAIN = "UNCERTAIN"


DEPLOYMENT_OBSERVATION_FIELDS = (
    "q",
    "qdot",
    "q_cmd",
    "motor_effort",
    "previous_action",
    "hand_fk_position_w",
    "hand_fk_velocity_w",
    "base_position_w",
    "base_velocity_w",
    "imu_orientation_rpy",
    "imu_angular_velocity",
    "map_cell_label",
    "relative_goal_xy",
    "localization_assumption",
)

EVALUATION_GROUND_TRUTH_FIELDS = (
    "object_kind",
    "simulator_contact_flag",
    "simulator_contact_point_w",
    "true_hand_obstacle_gap_m",
    "curtain_node_positions_w",
    "curtain_node_displacement_m",
    "true_passability",
    "episode_outcome_oracle",
)


def _array(value: Iterable[float] | np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a non-empty finite vector")
    return result.copy()


def _vector_or_none(value: Iterable[float] | np.ndarray | None, *, name: str) -> np.ndarray | None:
    if value is None:
        return None
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be None or a non-empty finite vector")
    return result.copy()


@dataclass(frozen=True)
class DeploymentObservation:
    """The only observation structure accepted by the contact classifier.

    ``base_position_w`` is allowed in this phase because the project permits a
    simulator pose localization assumption.  The assumption is explicit and
    must be replaced by a deployable localization source before a real-robot
    claim is made.
    """

    q: np.ndarray
    qdot: np.ndarray
    q_cmd: np.ndarray
    motor_effort: np.ndarray
    previous_action: np.ndarray
    hand_fk_position_w: np.ndarray
    hand_fk_velocity_w: np.ndarray
    base_position_w: np.ndarray | None = None
    base_velocity_w: np.ndarray | None = None
    imu_orientation_rpy: np.ndarray | None = None
    imu_angular_velocity: np.ndarray | None = None
    map_cell_label: int | None = None
    relative_goal_xy: np.ndarray | None = None
    localization_assumption: str = ""

    def __post_init__(self) -> None:
        for name in ("q", "qdot", "q_cmd", "motor_effort", "previous_action"):
            object.__setattr__(self, name, _array(getattr(self, name), name=name))
        for name in ("hand_fk_position_w", "hand_fk_velocity_w"):
            value = _array(getattr(self, name), name=name)
            if value.size != 3:
                raise ValueError(f"{name} must contain exactly three values")
            object.__setattr__(self, name, value)
        for name in ("base_position_w", "base_velocity_w", "imu_orientation_rpy", "imu_angular_velocity"):
            object.__setattr__(self, name, _vector_or_none(getattr(self, name), name=name))
        object.__setattr__(self, "relative_goal_xy", _vector_or_none(self.relative_goal_xy, name="relative_goal_xy"))
        if self.q.shape != self.qdot.shape or self.q.shape != self.q_cmd.shape or self.q.shape != self.motor_effort.shape:
            raise ValueError("q, qdot, q_cmd, and motor_effort must have matching shapes")
        if self.map_cell_label is not None and int(self.map_cell_label) not in (0, 1, 2):
            raise ValueError("map_cell_label must be one of the known 0/1/2 labels")
        if self.localization_assumption is None:
            raise ValueError("localization_assumption must be an explicit string")

    def to_runtime_dict(self) -> dict[str, object]:
        """Serialize only fields that may enter a deployable controller."""
        result: dict[str, object] = {}
        for field_name in DEPLOYMENT_OBSERVATION_FIELDS:
            value = getattr(self, field_name)
            result[field_name] = None if value is None else (
                value.tolist() if isinstance(value, np.ndarray) else value
            )
        return result


@dataclass(frozen=True)
class EvaluationGroundTruth:
    """Simulator-only evidence; never pass this object to a controller."""

    object_kind: str | None = None
    simulator_contact_flag: bool | None = None
    simulator_contact_point_w: np.ndarray | None = None
    true_hand_obstacle_gap_m: float | None = None
    curtain_node_positions_w: np.ndarray | None = None
    curtain_node_displacement_m: float | None = None
    true_passability: bool | None = None
    episode_outcome_oracle: str | None = None

    def __post_init__(self) -> None:
        if self.simulator_contact_point_w is not None:
            point = np.asarray(self.simulator_contact_point_w, dtype=np.float64).reshape(-1)
            if point.size != 3 or not np.isfinite(point).all():
                raise ValueError("simulator_contact_point_w must be a finite 3-vector")
            object.__setattr__(self, "simulator_contact_point_w", point.copy())
        if self.curtain_node_positions_w is not None:
            nodes = np.asarray(self.curtain_node_positions_w, dtype=np.float64)
            if nodes.ndim != 2 or nodes.shape[1] != 3 or not np.isfinite(nodes).all():
                raise ValueError("curtain_node_positions_w must be a finite Nx3 array")
            object.__setattr__(self, "curtain_node_positions_w", nodes.copy())
        for name in ("true_hand_obstacle_gap_m", "curtain_node_displacement_m"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite when provided")

    def to_evaluation_dict(self) -> dict[str, object]:
        result = asdict(self)
        for key, value in list(result.items()):
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
        return result


@dataclass(frozen=True)
class ProprioceptiveThresholds:
    """Small, explicit engineering thresholds for a deterministic rule."""

    minimum_samples: int = 4
    responsive_progress_m: float = 0.008
    blocked_progress_max_m: float = 0.004
    responsive_progress_ratio_min: float = 0.75
    blocked_progress_ratio_max: float = 0.60
    blocked_effort_deviation_min: float = 0.10
    blocked_stall_duration_s: float = 0.15
    low_hand_speed_mps: float = 0.004

    def validate(self) -> None:
        if self.minimum_samples < 2:
            raise ValueError("minimum_samples must be at least two")
        for name in (
            "responsive_progress_m",
            "blocked_progress_max_m",
            "blocked_effort_deviation_min",
            "blocked_stall_duration_s",
            "low_hand_speed_mps",
            "responsive_progress_ratio_min",
            "blocked_progress_ratio_max",
        ):
            if float(getattr(self, name)) < 0.0 or not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite and non-negative")


def _validate_history(history: Sequence[DeploymentObservation]) -> None:
    if not history:
        raise ValueError("proprioceptive history cannot be empty")
    shape = history[0].q.shape
    for sample in history:
        if sample.q.shape != shape:
            raise ValueError("all history samples must have the same joint dimension")


def _effort_deviation(
    history: Sequence[DeploymentObservation],
    baseline: Sequence[DeploymentObservation] | None,
) -> float:
    effort = np.vstack([sample.motor_effort for sample in history])
    if baseline:
        _validate_history(baseline)
        reference = np.mean(np.vstack([sample.motor_effort for sample in baseline]), axis=0)
    else:
        reference = effort[0]
    return float(np.mean(np.linalg.norm(effort - reference[None, :], axis=1)))


def extract_proprioceptive_features(
    history: Sequence[DeploymentObservation],
    *,
    baseline: Sequence[DeploymentObservation] | None = None,
    forward_axis_w: Iterable[float] = (1.0, 0.0, 0.0),
    sample_period_s: float = 0.05,
) -> dict[str, float | int | bool]:
    """Compute contact features from deployment observations only."""
    _validate_history(history)
    if sample_period_s <= 0.0 or not math.isfinite(sample_period_s):
        raise ValueError("sample_period_s must be finite and positive")
    axis = np.asarray(tuple(forward_axis_w), dtype=np.float64).reshape(-1)
    if axis.size != 3 or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1.0e-9:
        raise ValueError("forward_axis_w must be a non-zero finite 3-vector")
    axis /= np.linalg.norm(axis)

    hand_positions = np.vstack([sample.hand_fk_position_w for sample in history])
    hand_velocities = np.vstack([sample.hand_fk_velocity_w for sample in history])
    projected = hand_positions @ axis
    progress = float(projected[-1] - projected[0])
    projected_speed = hand_velocities @ axis
    low_speed = np.abs(projected_speed) <= 0.004
    stall_duration = float(np.count_nonzero(low_speed) * sample_period_s)
    q_error = np.vstack([sample.q_cmd - sample.q for sample in history])
    q_error_rms = float(np.sqrt(np.mean(q_error * q_error)))
    qdot_norm = float(np.mean(np.linalg.norm(np.vstack([sample.qdot for sample in history]), axis=1)))
    base_positions = [sample.base_position_w for sample in history if sample.base_position_w is not None]
    if len(base_positions) >= 2:
        base_disturbance = float(np.linalg.norm(base_positions[-1] - base_positions[0]))
    else:
        base_disturbance = float("nan")
    baseline_progress = float("nan")
    if baseline:
        _validate_history(baseline)
        baseline_positions = np.vstack([sample.hand_fk_position_w for sample in baseline])
        baseline_progress = float((baseline_positions @ axis)[-1] - (baseline_positions @ axis)[0])
    progress_ratio = (
        float(progress / baseline_progress)
        if math.isfinite(baseline_progress) and abs(baseline_progress) > 1.0e-6
        else float("nan")
    )
    return {
        "sample_count": int(len(history)),
        "hand_forward_progress_m": progress,
        "baseline_hand_forward_progress_m": baseline_progress,
        "hand_progress_ratio": progress_ratio,
        "hand_forward_speed_mean_mps": float(np.mean(projected_speed)),
        "hand_low_speed_duration_s": stall_duration,
        "upper_tracking_rms": q_error_rms,
        "joint_velocity_mean_norm": qdot_norm,
        "effort_deviation": _effort_deviation(history, baseline),
        "base_disturbance_m": base_disturbance,
        "has_base_localization": bool(base_positions),
    }


def classify_proprioceptive_response(
    history: Sequence[DeploymentObservation],
    *,
    baseline: Sequence[DeploymentObservation] | None = None,
    thresholds: ProprioceptiveThresholds | None = None,
    forward_axis_w: Iterable[float] = (1.0, 0.0, 0.0),
    sample_period_s: float = 0.05,
) -> tuple[ContactDecision, dict[str, float | int | bool]]:
    """Return a conservative decision without consulting evaluation truth.

    The rule is intentionally modest: clear forward progress is responsive;
    sustained low progress together with elevated effort deviation is blocked;
    all ambiguous cases remain uncertain.
    """
    config = thresholds or ProprioceptiveThresholds()
    config.validate()
    features = extract_proprioceptive_features(
        history,
        baseline=baseline,
        forward_axis_w=forward_axis_w,
        sample_period_s=sample_period_s,
    )
    if int(features["sample_count"]) < config.minimum_samples:
        return ContactDecision.UNCERTAIN, features
    progress = float(features["hand_forward_progress_m"])
    progress_ratio = float(features["hand_progress_ratio"])
    effort_deviation = float(features["effort_deviation"])
    stall_duration = float(features["hand_low_speed_duration_s"])
    if progress >= config.responsive_progress_m and (
        not math.isfinite(progress_ratio) or progress_ratio >= config.responsive_progress_ratio_min
    ):
        return ContactDecision.RESPONSIVE, features
    if (
        progress <= config.blocked_progress_max_m
        or (
            math.isfinite(progress_ratio)
            and progress_ratio <= config.blocked_progress_ratio_max
        )
    ) and (
        effort_deviation >= config.blocked_effort_deviation_min
        and stall_duration >= config.blocked_stall_duration_s
    ):
        return ContactDecision.BLOCKED, features
    return ContactDecision.UNCERTAIN, features


def privileged_information_leakage_audit() -> dict[str, object]:
    """Return a machine-readable audit used by the Round 7 report."""
    return {
        "deployment_observation_fields": list(DEPLOYMENT_OBSERVATION_FIELDS),
        "evaluation_ground_truth_fields": list(EVALUATION_GROUND_TRUTH_FIELDS),
        "classifier_input": "DeploymentObservation history plus optional deployment baseline",
        "classifier_forbidden_inputs": list(EVALUATION_GROUND_TRUTH_FIELDS),
        "localization_assumption_required": True,
        "formal_real_robot_status": "NOT_READY_WHEN_LOCALIZATION_OR_ANY_FIELD_IS_UNAVAILABLE",
    }
