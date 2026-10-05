"""Interpretable probe heuristic and engineering-baseline state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BaselineState(str, Enum):
    GO_TO_GOAL = "GO_TO_GOAL"
    BARRIER_DETECTED = "BARRIER_DETECTED"
    APPROACH = "APPROACH"
    PROBE = "PROBE"
    SWEEP_AND_ENTER = "SWEEP_AND_ENTER"
    RETRACT_AND_DETOUR = "RETRACT_AND_DETOUR"
    GOAL_REACHED = "GOAL_REACHED"


class BarrierResponse(str, Enum):
    PASSABLE = "PASSABLE"
    NONPASSABLE = "NONPASSABLE"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class ProbeEvaluatorConfig:
    displacement_threshold_m: float = 0.015
    effort_threshold_nm: float = 18.0
    tracking_error_threshold_rad: float = 0.30
    required_samples: int = 3


class ProbeEvaluator:
    """Decide from barrier displacement plus arm load/tracking response."""

    def __init__(self, config: ProbeEvaluatorConfig | None = None):
        self.config = config or ProbeEvaluatorConfig()

    def evaluate(self, barrier_displacements_m: list[float], arm_efforts_nm: list[float],
                 tracking_errors_rad: list[float]) -> BarrierResponse:
        n = self.config.required_samples
        if min(len(barrier_displacements_m), len(arm_efforts_nm), len(tracking_errors_rad)) < n:
            return BarrierResponse.INCONCLUSIVE
        displacement = max(abs(value) for value in barrier_displacements_m[-n:])
        effort = max(abs(value) for value in arm_efforts_nm[-n:])
        tracking = max(abs(value) for value in tracking_errors_rad[-n:])
        if (displacement >= self.config.displacement_threshold_m
                and effort < self.config.effort_threshold_nm
                and tracking < self.config.tracking_error_threshold_rad):
            return BarrierResponse.PASSABLE
        if (displacement < self.config.displacement_threshold_m
                and (effort >= self.config.effort_threshold_nm
                     or tracking >= self.config.tracking_error_threshold_rad)):
            return BarrierResponse.NONPASSABLE
        return BarrierResponse.INCONCLUSIVE


class EngineeringBaseline:
    """Explicit transitions for logging and later replacement."""

    def __init__(self):
        self.state = BaselineState.GO_TO_GOAL
        self.response = BarrierResponse.INCONCLUSIVE

    def barrier_seen(self) -> None:
        if self.state == BaselineState.GO_TO_GOAL:
            self.state = BaselineState.BARRIER_DETECTED

    def begin_approach(self) -> None:
        if self.state == BaselineState.BARRIER_DETECTED:
            self.state = BaselineState.APPROACH

    def begin_probe(self) -> None:
        if self.state == BaselineState.APPROACH:
            self.state = BaselineState.PROBE

    def apply_probe_response(self, response: BarrierResponse) -> None:
        if self.state != BaselineState.PROBE:
            raise RuntimeError(f"probe response received in {self.state}")
        self.response = response
        if response == BarrierResponse.PASSABLE:
            self.state = BaselineState.SWEEP_AND_ENTER
        elif response == BarrierResponse.NONPASSABLE:
            self.state = BaselineState.RETRACT_AND_DETOUR

    def goal_reached(self) -> None:
        self.state = BaselineState.GOAL_REACHED
