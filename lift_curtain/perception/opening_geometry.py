"""Conservative geometry checks for a verified multi-height opening."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .online_occupancy import CellState, LocalOccupancyGrid


@dataclass(frozen=True)
class CorridorCheck:
    passed: bool
    required_width_m: float
    common_width_m: float
    lateral_interval_m: tuple[float, float] | None
    checked_height_slices: tuple[int, ...]
    unknown_samples: int
    occupied_samples: int
    reason: str


def _contiguous_runs(values: list[float], spacing: float) -> list[tuple[float, float]]:
    if not values:
        return []
    runs: list[tuple[float, float]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value - previous > spacing * 1.5:
            runs.append((start - spacing * 0.5, previous + spacing * 0.5))
            start = value
        previous = value
    runs.append((start - spacing * 0.5, previous + spacing * 0.5))
    return runs


def check_common_free_corridor(
    grid: LocalOccupancyGrid,
    *,
    center_xy: tuple[float, float],
    forward_yaw_rad: float,
    lateral_limits_m: tuple[float, float],
    forward_limits_m: tuple[float, float],
    required_width_m: float,
    height_slices: tuple[int, ...] | list[int] | None = None,
    forward_step_m: float | None = None,
) -> CorridorCheck:
    """Find one lateral interval free through all requested height slices.

    The interval is expressed as signed offsets from ``center_xy`` along the
    left-facing lateral axis.  Every sampled forward point and every requested
    height slice must be explicitly FREE.  A separate best gap at each height
    is never combined.
    """
    lateral_min, lateral_max = map(float, lateral_limits_m)
    forward_min, forward_max = map(float, forward_limits_m)
    if lateral_min >= lateral_max or forward_min > forward_max:
        raise ValueError("limits must be ordered and non-empty")
    if required_width_m <= 0.0 or not math.isfinite(required_width_m):
        raise ValueError("required_width_m must be finite and positive")
    if forward_step_m is not None and forward_step_m <= 0.0:
        raise ValueError("forward_step_m must be positive")
    requested = tuple(int(value) for value in (height_slices or range(grid.num_slices)))
    if not requested or any(value < 0 or value >= grid.num_slices for value in requested):
        raise ValueError("height_slices contains an invalid slice")

    lateral_step = grid.resolution_m
    lateral_offsets = np.arange(lateral_min, lateral_max + lateral_step * 0.25, lateral_step)
    forward_step = float(forward_step_m or grid.resolution_m)
    forward_offsets = np.arange(forward_min, forward_max + forward_step * 0.25, forward_step)
    forward_axis = np.asarray((math.cos(forward_yaw_rad), math.sin(forward_yaw_rad)))
    lateral_axis = np.asarray((-math.sin(forward_yaw_rad), math.cos(forward_yaw_rad)))
    center = np.asarray(center_xy, dtype=np.float64)

    valid_lateral: list[float] = []
    unknown_samples = 0
    occupied_samples = 0
    height_centers = [
        float((grid.height_edges[index] + grid.height_edges[index + 1]) * 0.5)
        for index in requested
    ]
    for lateral_offset in lateral_offsets:
        lateral_ok = True
        for forward_offset in forward_offsets:
            point = center + forward_axis * float(forward_offset) + lateral_axis * float(lateral_offset)
            for z_m in height_centers:
                state = grid.state_at(float(point[0]), float(point[1]), z_m)
                if state == CellState.UNKNOWN:
                    unknown_samples += 1
                    lateral_ok = False
                elif state == CellState.OCCUPIED:
                    occupied_samples += 1
                    lateral_ok = False
        if lateral_ok:
            valid_lateral.append(float(lateral_offset))

    runs = _contiguous_runs(valid_lateral, lateral_step)
    if not runs:
        reason = "NO_COMMON_FREE_INTERVAL_UNKNOWN_OR_OCCUPIED"
        return CorridorCheck(
            False, float(required_width_m), 0.0, None, requested,
            unknown_samples, occupied_samples, reason,
        )
    best = max(runs, key=lambda interval: interval[1] - interval[0])
    common_width = max(0.0, best[1] - best[0])
    passed = common_width + 1.0e-9 >= required_width_m
    return CorridorCheck(
        passed,
        float(required_width_m),
        float(common_width),
        (float(best[0]), float(best[1])),
        requested,
        unknown_samples,
        occupied_samples,
        "COMMON_FREE_CORRIDOR_VERIFIED" if passed else "COMMON_INTERVAL_TOO_NARROW",
    )
