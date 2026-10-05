"""Bounded SE(2) feedback used for locomotion commands."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class SE2Reference:
    position_world: tuple[float, float]
    yaw: float
    velocity_world: tuple[float, float] = (0.0, 0.0)
    yaw_rate: float = 0.0


@dataclass(frozen=True)
class PushPathTrackerConfig:
    """Conservative command authority and rate limits."""

    position_gain_xy: tuple[float, float] = (0.55, 0.85)
    heading_gain: float = 0.80
    vx_min: float = 0.0
    vx_max: float = 0.30
    vy_limit: float = 0.10
    wz_limit: float = 0.30
    max_command_step: tuple[float, float, float] = (0.06, 0.04, 0.06)

    def __post_init__(self) -> None:
        if self.vx_min < 0.0 or self.vx_max < self.vx_min:
            raise ValueError("forward limits must satisfy 0 <= vx_min <= vx_max")
        if self.vy_limit <= 0.0 or self.wz_limit <= 0.0:
            raise ValueError("authority limits must be positive")
        if len(self.position_gain_xy) != 2 or len(self.max_command_step) != 3:
            raise ValueError("gains and rate limits have fixed dimensions")
        if any(value <= 0.0 for value in self.max_command_step):
            raise ValueError("rate limits must be positive")


class PushPathTracker:
    """World-frame P plus feed-forward tracker with mild rate limiting."""

    def __init__(self, config: PushPathTrackerConfig | None = None):
        self.config = config or PushPathTrackerConfig()
        self._previous_command: np.ndarray | None = None

    def reset(self) -> None:
        self._previous_command = None

    def __call__(self, current_pose_world: tuple[float, float, float] | np.ndarray,
                 reference: SE2Reference) -> np.ndarray:
        current = np.asarray(current_pose_world, dtype=np.float64)
        if current.shape != (3,) or not np.isfinite(current).all():
            raise ValueError("current_pose_world must be a finite three-vector")
        target = np.asarray(reference.position_world, dtype=np.float64)
        feedforward = np.asarray(reference.velocity_world, dtype=np.float64)
        if target.shape != (2,) or feedforward.shape != (2,):
            raise ValueError("reference position and velocity must be two-vectors")
        error_world = target - current[:2]
        world_velocity = feedforward + np.asarray(self.config.position_gain_xy) * error_world
        cosine, sine = math.cos(float(current[2])), math.sin(float(current[2]))
        world_to_body = np.asarray(((cosine, sine), (-sine, cosine)))
        body_velocity = world_to_body @ world_velocity
        heading_error = wrap_angle(float(reference.yaw) - float(current[2]))
        command = np.asarray((
            np.clip(body_velocity[0], self.config.vx_min, self.config.vx_max),
            np.clip(body_velocity[1], -self.config.vy_limit, self.config.vy_limit),
            np.clip(float(reference.yaw_rate) + self.config.heading_gain * heading_error,
                    -self.config.wz_limit, self.config.wz_limit),
        ), dtype=np.float64)
        if self._previous_command is not None:
            step = np.asarray(self.config.max_command_step, dtype=np.float64)
            command = np.clip(command, self._previous_command - step, self._previous_command + step)
            command[0] = np.clip(command[0], self.config.vx_min, self.config.vx_max)
            command[1] = np.clip(command[1], -self.config.vy_limit, self.config.vy_limit)
            command[2] = np.clip(command[2], -self.config.wz_limit, self.config.wz_limit)
        self._previous_command = command.copy()
        return command


def straight_reference(time_s: float, origin_world: tuple[float, float], yaw: float = 0.0,
                      speed_mps: float = 0.30) -> SE2Reference:
    if not math.isfinite(time_s) or time_s < 0.0 or speed_mps < 0.0:
        raise ValueError("time_s must be finite/non-negative and speed_mps must be non-negative")
    return SE2Reference(
        position_world=(float(origin_world[0]) + speed_mps * float(time_s), float(origin_world[1])),
        yaw=float(yaw), velocity_world=(float(speed_mps), 0.0), yaw_rate=0.0,
    )
