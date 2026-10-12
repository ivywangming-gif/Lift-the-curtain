"""Independent PointGoal and PoseGoal evaluation.

PointGoal is the formal Round 14 completion contract: planar position, a
measured stop-speed dwell, finite/no-fall state, and no external safety stop.
Yaw is retained as an audit value and is only required by the legacy PoseGoal
comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PointGoalConfig:
    position_tolerance_m: float = 0.08
    stop_speed_threshold_mps: float = 0.06
    stop_dwell_duration_s: float = 0.30
    yaw_tolerance_rad: float = 0.15

    def validate(self) -> None:
        values = (
            self.position_tolerance_m,
            self.stop_speed_threshold_mps,
            self.stop_dwell_duration_s,
            self.yaw_tolerance_rad,
        )
        if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in values):
            raise ValueError("PointGoal tolerances must be finite and positive")


@dataclass(frozen=True)
class GoalEvaluation:
    point_goal_pass: bool
    pose_goal_pass: bool
    xy_error_m: float
    final_yaw_error_rad: float | None
    stop_speed_max_mps: float | None
    stop_sample_count: int
    stop_dwell_duration_s: float
    position_pass: bool
    speed_pass: bool
    finite_no_fall_pass: bool
    safety_pass: bool
    yaw_pass: bool | None
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "point_goal_pass": self.point_goal_pass,
            "pose_goal_pass": self.pose_goal_pass,
            "xy_error_m": self.xy_error_m,
            "final_yaw_error_rad": self.final_yaw_error_rad,
            "stop_speed_max_mps": self.stop_speed_max_mps,
            "stop_sample_count": self.stop_sample_count,
            "stop_dwell_duration_s": self.stop_dwell_duration_s,
            "position_pass": self.position_pass,
            "speed_pass": self.speed_pass,
            "finite_no_fall_pass": self.finite_no_fall_pass,
            "safety_pass": self.safety_pass,
            "yaw_pass": self.yaw_pass,
            "reason": self.reason,
        }


def _wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def evaluate_goal(
    *,
    final_xy: Iterable[float],
    goal_xy: Iterable[float],
    speed_samples_mps: Iterable[float],
    speed_sample_period_s: float,
    finite_no_fall: bool,
    external_safety_reason: str | None = None,
    final_yaw_rad: float | None = None,
    goal_yaw_rad: float | None = None,
    config: PointGoalConfig | None = None,
) -> GoalEvaluation:
    """Evaluate both contracts without reading simulator object truth."""
    cfg = config or PointGoalConfig()
    cfg.validate()
    final = np.asarray(tuple(final_xy), dtype=np.float64).reshape(-1)
    goal = np.asarray(tuple(goal_xy), dtype=np.float64).reshape(-1)
    if final.size != 2 or goal.size != 2 or not np.isfinite(final).all() or not np.isfinite(goal).all():
        raise ValueError("final_xy and goal_xy must be finite 2-vectors")
    period = float(speed_sample_period_s)
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError("speed_sample_period_s must be finite and positive")
    speeds = np.asarray(tuple(speed_samples_mps), dtype=np.float64).reshape(-1)
    if speeds.size and (not np.isfinite(speeds).all() or np.any(speeds < 0.0)):
        raise ValueError("speed samples must be finite and non-negative")

    needed = max(1, int(math.ceil(cfg.stop_dwell_duration_s / period)))
    stop_samples = speeds[-needed:] if speeds.size >= needed else speeds
    dwell = float(stop_samples.size * period)
    speed_max = float(np.max(stop_samples)) if stop_samples.size else None
    position_error = float(np.linalg.norm(final - goal))
    position_pass = position_error <= cfg.position_tolerance_m
    speed_pass = bool(stop_samples.size >= needed and speed_max is not None and speed_max <= cfg.stop_speed_threshold_mps)
    finite_pass = bool(finite_no_fall)
    safety_pass = not bool(external_safety_reason)
    point_pass = bool(position_pass and speed_pass and finite_pass and safety_pass)

    yaw_error: float | None = None
    yaw_pass: bool | None = None
    pose_pass = False
    if final_yaw_rad is not None and goal_yaw_rad is not None:
        if not math.isfinite(float(final_yaw_rad)) or not math.isfinite(float(goal_yaw_rad)):
            raise ValueError("yaw values must be finite when provided")
        yaw_error = abs(_wrap_angle(float(final_yaw_rad) - float(goal_yaw_rad)))
        yaw_pass = yaw_error <= cfg.yaw_tolerance_rad
        pose_pass = bool(point_pass and yaw_pass)

    failures = []
    if not position_pass:
        failures.append("XY_ERROR")
    if not speed_pass:
        failures.append("STOP_SPEED_OR_DWELL")
    if not finite_pass:
        failures.append("FALL_OR_NONFINITE")
    if not safety_pass:
        failures.append("EXTERNAL_SAFETY_STOP")
    if yaw_pass is False:
        failures.append("YAW_ONLY_POSE_GOAL")
    reason = "PASS_POINT_GOAL" if point_pass else "+".join(failures)
    return GoalEvaluation(
        point_goal_pass=point_pass,
        pose_goal_pass=pose_pass,
        xy_error_m=position_error,
        final_yaw_error_rad=yaw_error,
        stop_speed_max_mps=speed_max,
        stop_sample_count=int(stop_samples.size),
        stop_dwell_duration_s=dwell,
        position_pass=position_pass,
        speed_pass=speed_pass,
        finite_no_fall_pass=finite_pass,
        safety_pass=safety_pass,
        yaw_pass=yaw_pass,
        reason=reason,
    )
