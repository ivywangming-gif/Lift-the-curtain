"""Bounded deployment-only progress/stuck monitoring.

The monitor consumes estimated robot state and commands only.  It has no
access to object identity, simulator contact, cloth nodes, or passability.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable

import numpy as np


class ProgressStatus(str, Enum):
    MOVING = "MOVING"
    STAGNANT = "STAGNANT"
    SAFETY_STOP = "SAFETY_STOP"


class ProgressPhase(str, Enum):
    """Phases in which a forward-progress monitor may be active."""

    INACTIVE = "INACTIVE"
    CAUTIOUS_TRAVERSE = "CAUTIOUS_TRAVERSE"
    NAVIGATE_TO_GOAL = "NAVIGATE_TO_GOAL"


@dataclass(frozen=True)
class ProgressMonitorConfig:
    window_duration_s: float = 0.80
    command_threshold_mps: float = 0.08
    minimum_progress_m: float = 0.015
    recovery_progress_m: float = 0.025
    required_stagnant_windows: int = 3
    max_abs_roll_pitch_rad: float = 0.60

    def validate(self) -> None:
        if self.window_duration_s <= 0.0 or self.command_threshold_mps < 0.0:
            raise ValueError("monitor durations and command threshold are invalid")
        if self.minimum_progress_m < 0.0 or self.recovery_progress_m < self.minimum_progress_m:
            raise ValueError("progress thresholds are invalid")
        if self.required_stagnant_windows < 1 or self.max_abs_roll_pitch_rad <= 0.0:
            raise ValueError("monitor limits are invalid")


@dataclass(frozen=True)
class ProgressSample:
    time_s: float
    base_xy_m: tuple[float, float]
    command_world_xy_mps: tuple[float, float]
    roll_rad: float | None = None
    pitch_rad: float | None = None
    hand_xy_m: tuple[float, float] | None = None


@dataclass(frozen=True)
class ProgressReading:
    status: ProgressStatus
    window_progress_m: float
    commanded_progress_m: float
    stagnant_windows: int
    stop_requested: bool
    reason: str


class PhaseAwareProgressMonitor:
    """Own the progress-window lifecycle at deployment state-machine edges.

    The wrapped monitor keeps its original frozen thresholds.  A new traverse
    segment resets its history and direction.  Stand, probe, retract, release,
    goal and verified candidate-clearance phases explicitly deactivate it, so
    commands from an earlier phase cannot create a later false STUCK event.
    """

    def __init__(self, config: ProgressMonitorConfig | None = None):
        self.monitor = DeploymentProgressMonitor(config)
        self.phase = ProgressPhase.INACTIVE
        self._axis_world = np.asarray((1.0, 0.0), dtype=np.float64)
        self._segment_id: str | None = None

    @property
    def active(self) -> bool:
        return self.phase == ProgressPhase.CAUTIOUS_TRAVERSE

    @property
    def segment_id(self) -> str | None:
        return self._segment_id

    def begin_traverse(
        self,
        *,
        progress_axis_world: Iterable[float],
        segment_id: str,
    ) -> None:
        axis = np.asarray(
            DeploymentProgressMonitor._finite_vector(
                progress_axis_world, 2, "progress_axis_world"
            ),
            dtype=np.float64,
        )
        if float(np.linalg.norm(axis)) <= 1.0e-9:
            raise ValueError("progress axis must be non-zero")
        if not str(segment_id):
            raise ValueError("segment_id must be non-empty")
        self.monitor.reset()
        self._axis_world = axis
        self._segment_id = str(segment_id)
        self.phase = ProgressPhase.CAUTIOUS_TRAVERSE

    def deactivate(self, *, phase: ProgressPhase = ProgressPhase.INACTIVE) -> None:
        if phase == ProgressPhase.CAUTIOUS_TRAVERSE:
            raise ValueError("deactivate phase cannot be CAUTIOUS_TRAVERSE")
        self.monitor.reset()
        self.phase = ProgressPhase(phase)
        self._segment_id = None

    def update(
        self,
        sample: ProgressSample,
        *,
        phase: ProgressPhase,
        requires_forward_progress: bool,
        candidate_clearance_verified: bool = False,
        progress_axis_world: Iterable[float] | None = None,
        segment_id: str | None = None,
    ) -> ProgressReading:
        current_phase = ProgressPhase(phase)
        if current_phase != ProgressPhase.CAUTIOUS_TRAVERSE:
            self.deactivate(phase=current_phase)
            return ProgressReading(
                ProgressStatus.MOVING,
                0.0,
                0.0,
                0,
                False,
                f"PHASE_INACTIVE:{current_phase.value}",
            )
        if candidate_clearance_verified or not requires_forward_progress:
            self.deactivate(phase=ProgressPhase.INACTIVE)
            reason = "CANDIDATE_CLEARANCE_VERIFIED" if candidate_clearance_verified else "FORWARD_PROGRESS_NOT_REQUIRED"
            return ProgressReading(ProgressStatus.MOVING, 0.0, 0.0, 0, False, reason)
        if (
            not self.active
            or (segment_id is not None and self._segment_id != str(segment_id))
        ):
            self.begin_traverse(
                progress_axis_world=(
                    progress_axis_world
                    if progress_axis_world is not None
                    else (1.0, 0.0)
                ),
                segment_id=segment_id or "traverse",
            )
        elif progress_axis_world is not None:
            axis = np.asarray(tuple(float(value) for value in progress_axis_world), dtype=np.float64)
            if axis.shape != (2,) or not np.isfinite(axis).all() or float(np.linalg.norm(axis)) <= 1.0e-9:
                raise ValueError("progress_axis_world must be a finite non-zero 2-vector")
            # A changed path direction starts a new finite window rather than
            # mixing progress measured along two different segments.
            if not np.allclose(axis, self._axis_world):
                self.begin_traverse(
                    progress_axis_world=axis,
                    segment_id=segment_id or "traverse",
                )
        return self.monitor.update(sample, progress_axis_world=self._axis_world)


class DeploymentProgressMonitor:
    """Hysteretic finite-window monitor for cautious interaction phases."""

    def __init__(self, config: ProgressMonitorConfig | None = None):
        self.config = config or ProgressMonitorConfig()
        self.config.validate()
        self._samples: deque[ProgressSample] = deque()
        self._stagnant_windows = 0
        self._status = ProgressStatus.MOVING

    def reset(self) -> None:
        self._samples.clear()
        self._stagnant_windows = 0
        self._status = ProgressStatus.MOVING

    @staticmethod
    def _finite_vector(value: Iterable[float], size: int, name: str) -> tuple[float, ...]:
        result = tuple(float(item) for item in value)
        if len(result) != size or not np.isfinite(result).all():
            raise ValueError(f"{name} must be a finite vector of size {size}")
        return result

    def update(self, sample: ProgressSample, *, progress_axis_world: Iterable[float] = (1.0, 0.0)) -> ProgressReading:
        axis = np.asarray(self._finite_vector(progress_axis_world, 2, "progress_axis_world"), dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if norm <= 1.0e-9:
            raise ValueError("progress axis must be non-zero")
        axis /= norm
        if not math.isfinite(float(sample.time_s)):
            raise ValueError("sample time must be finite")
        base = np.asarray(self._finite_vector(sample.base_xy_m, 2, "base_xy_m"), dtype=np.float64)
        command = np.asarray(self._finite_vector(sample.command_world_xy_mps, 2, "command_world_xy_mps"), dtype=np.float64)
        if sample.roll_rad is not None and (not math.isfinite(sample.roll_rad) or abs(sample.roll_rad) > self.config.max_abs_roll_pitch_rad):
            self._status = ProgressStatus.SAFETY_STOP
            return ProgressReading(self._status, 0.0, 0.0, self._stagnant_windows, True, "IMU_ROLL_LIMIT")
        if sample.pitch_rad is not None and (not math.isfinite(sample.pitch_rad) or abs(sample.pitch_rad) > self.config.max_abs_roll_pitch_rad):
            self._status = ProgressStatus.SAFETY_STOP
            return ProgressReading(self._status, 0.0, 0.0, self._stagnant_windows, True, "IMU_PITCH_LIMIT")
        clean = ProgressSample(float(sample.time_s), (float(base[0]), float(base[1])), (float(command[0]), float(command[1])), sample.roll_rad, sample.pitch_rad, sample.hand_xy_m)
        self._samples.append(clean)
        cutoff = clean.time_s - self.config.window_duration_s
        while len(self._samples) > 1 and self._samples[0].time_s < cutoff:
            self._samples.popleft()
        if len(self._samples) < 2:
            return ProgressReading(self._status, 0.0, 0.0, self._stagnant_windows, False, "WARMUP")
        first = self._samples[0]
        elapsed = max(clean.time_s - first.time_s, 0.0)
        displacement = np.asarray(clean.base_xy_m) - np.asarray(first.base_xy_m)
        progress = float(np.dot(displacement, axis))
        commanded = 0.0
        previous = first
        for current in list(self._samples)[1:]:
            dt = max(current.time_s - previous.time_s, 0.0)
            commanded += max(float(np.dot(np.asarray(previous.command_world_xy_mps), axis)), 0.0) * dt
            previous = current
        commanded += max(float(np.dot(np.asarray(clean.command_world_xy_mps), axis)), 0.0) * max(clean.time_s - previous.time_s, 0.0)
        commanded_motion = commanded >= self.config.command_threshold_mps * max(elapsed, 1.0e-6)
        if commanded_motion and progress < self.config.minimum_progress_m:
            self._stagnant_windows += 1
        elif progress >= self.config.recovery_progress_m:
            self._stagnant_windows = 0
            self._status = ProgressStatus.MOVING
        if self._stagnant_windows >= self.config.required_stagnant_windows:
            self._status = ProgressStatus.STAGNANT
        stop = self._status in (ProgressStatus.STAGNANT, ProgressStatus.SAFETY_STOP)
        reason = "STAGNANT_HYSTERESIS" if self._status == ProgressStatus.STAGNANT else ("SAFETY_LIMIT" if stop else "PROGRESS_OK")
        return ProgressReading(self._status, progress, commanded, self._stagnant_windows, stop, reason)
