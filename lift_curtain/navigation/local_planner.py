"""Receding-horizon planner that only follows observed FREE cells."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from lift_curtain.perception.online_occupancy import CellState, LocalOccupancyGrid


class LocalPlanStatus(str, Enum):
    DIRECT = "DIRECT"
    DETOUR = "DETOUR"
    OBSERVE = "OBSERVE"
    GOAL_REACHED = "GOAL_REACHED"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class LocalPlannerConfig:
    goal_tolerance_m: float = 0.25
    local_forward_distances_m: tuple[float, ...] = (0.6, 0.9, 1.2)
    lateral_offsets_m: tuple[float, ...] = (-1.2, -0.9, -0.6, 0.6, 0.9, 1.2)
    max_candidate_count: int = 32

    def validate(self) -> None:
        if self.goal_tolerance_m <= 0.0:
            raise ValueError("goal_tolerance_m must be positive")
        if not self.local_forward_distances_m or not self.lateral_offsets_m:
            raise ValueError("local planner needs candidate distances and offsets")
        if self.max_candidate_count < 1:
            raise ValueError("max_candidate_count must be positive")


@dataclass(frozen=True)
class LocalPlan:
    status: LocalPlanStatus
    waypoint_xy: tuple[float, float] | None
    reason: str
    direct_state: CellState
    candidates_considered: int


class LocalPlanner:
    """Plan one short segment, then require a fresh observation and replan."""

    def __init__(self, config: LocalPlannerConfig | None = None):
        self.config = config or LocalPlannerConfig()
        self.config.validate()

    @staticmethod
    def _goal_frame(current_xy: tuple[float, float], goal_xy: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, float]:
        delta = np.asarray(goal_xy, dtype=np.float64) - np.asarray(current_xy, dtype=np.float64)
        distance = float(np.linalg.norm(delta))
        if distance < 1.0e-9:
            return np.asarray((1.0, 0.0)), np.asarray((0.0, 1.0)), distance
        forward = delta / distance
        lateral = np.asarray((-forward[1], forward[0]))
        return forward, lateral, distance

    @staticmethod
    def _segment_is_free(
        grid: LocalOccupancyGrid,
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
        height_values_m: tuple[float, ...],
    ) -> bool:
        return grid.segment_state(start_xy, end_xy, height_values_m) == CellState.FREE

    def plan(
        self,
        grid: LocalOccupancyGrid,
        *,
        current_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        height_values_m: tuple[float, ...],
    ) -> LocalPlan:
        forward, lateral, distance = self._goal_frame(current_xy, goal_xy)
        direct_state = grid.segment_state(current_xy, goal_xy, height_values_m)
        if distance <= self.config.goal_tolerance_m:
            return LocalPlan(LocalPlanStatus.GOAL_REACHED, goal_xy, "GOAL_WITHIN_TOLERANCE", direct_state, 0)
        if direct_state == CellState.FREE:
            return LocalPlan(LocalPlanStatus.DIRECT, goal_xy, "DIRECT_SEGMENT_VERIFIED_FREE", direct_state, 1)

        candidates: list[tuple[int, float, tuple[float, float]]] = []
        for forward_distance in self.config.local_forward_distances_m:
            for lateral_offset in self.config.lateral_offsets_m:
                point = np.asarray(current_xy, dtype=np.float64) + forward * float(forward_distance) + lateral * float(lateral_offset)
                waypoint = (float(point[0]), float(point[1]))
                if not self._segment_is_free(grid, current_xy, waypoint, height_values_m):
                    continue
                future_state = grid.segment_state(waypoint, goal_xy, height_values_m)
                future_rank = {
                    CellState.FREE: 0,
                    CellState.UNKNOWN: 1,
                    CellState.OCCUPIED: 2,
                }[future_state]
                goal_cost = float(np.linalg.norm(np.asarray(goal_xy) - point))
                lateral_cost = abs(float(lateral_offset)) * 0.15
                candidates.append((future_rank, goal_cost + lateral_cost, waypoint))
                if len(candidates) >= self.config.max_candidate_count:
                    break
            if len(candidates) >= self.config.max_candidate_count:
                break
        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1]))
            return LocalPlan(
                LocalPlanStatus.DETOUR,
                candidates[0][2],
                "DIRECT_NOT_VERIFIED_LOCAL_FREE_WAYPOINT_SELECTED",
                direct_state,
                len(candidates),
            )
        reason = (
            "DIRECT_OCCUPIED_NO_OBSERVED_DETOUR"
            if direct_state == CellState.OCCUPIED
            else "DIRECT_UNKNOWN_NO_OBSERVED_DETOUR"
        )
        return LocalPlan(LocalPlanStatus.UNRESOLVED, None, reason, direct_state, 0)
