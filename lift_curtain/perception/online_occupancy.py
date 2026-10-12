"""Conservative RGB-D driven local occupancy grid.

The grid starts UNKNOWN and is updated only by camera rays.  It intentionally
does not expose a simulator scene query or a global obstacle representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math

import numpy as np


class CellState(IntEnum):
    UNKNOWN = -1
    FREE = 0
    OCCUPIED = 1


@dataclass(frozen=True)
class CameraIntrinsics:
    width_px: int
    height_px: int
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float

    def validate(self) -> None:
        if self.width_px < 1 or self.height_px < 1:
            raise ValueError("camera dimensions must be positive")
        if min(self.fx_px, self.fy_px) <= 0.0:
            raise ValueError("camera focal lengths must be positive")
        if not np.isfinite((self.fx_px, self.fy_px, self.cx_px, self.cy_px)).all():
            raise ValueError("camera intrinsics must be finite")


def _bresenham(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    """Return integer grid cells on a 2-D ray, including both endpoints."""
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    cells: list[tuple[int, int]] = []
    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        double_error = 2 * error
        if double_error >= dy:
            error += dy
            x0 += sx
        if double_error <= dx:
            error += dx
            y0 += sy
    return cells


@dataclass
class LocalOccupancyGrid:
    """A finite local 2.5-D map with explicit UNKNOWN cells."""

    origin_xy: tuple[float, float]
    resolution_m: float
    size_xy: tuple[int, int]
    height_edges_m: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.resolution_m <= 0.0 or not math.isfinite(self.resolution_m):
            raise ValueError("resolution_m must be finite and positive")
        if len(self.size_xy) != 2 or min(self.size_xy) < 1:
            raise ValueError("size_xy must contain two positive dimensions")
        if len(self.height_edges_m) < 2 or any(
            not math.isfinite(value) for value in self.height_edges_m
        ) or any(a >= b for a, b in zip(self.height_edges_m, self.height_edges_m[1:])):
            raise ValueError("height_edges_m must be finite and strictly increasing")
        if not np.isfinite(self.origin_xy).all():
            raise ValueError("origin_xy must be finite")
        self.height_edges = np.asarray(self.height_edges_m, dtype=np.float64)
        self.states = np.full(
            (len(self.height_edges) - 1, int(self.size_xy[1]), int(self.size_xy[0])),
            CellState.UNKNOWN,
            dtype=np.int8,
        )
        self.observation_count = np.zeros_like(self.states, dtype=np.uint32)

    @property
    def num_slices(self) -> int:
        return int(self.states.shape[0])

    @property
    def width_m(self) -> float:
        return float(self.size_xy[0] * self.resolution_m)

    @property
    def depth_m(self) -> float:
        return float(self.size_xy[1] * self.resolution_m)

    def reset(self) -> None:
        self.states.fill(CellState.UNKNOWN)
        self.observation_count.fill(0)

    def height_slice_index(self, z_m: float) -> int | None:
        if not math.isfinite(float(z_m)):
            return None
        index = int(np.searchsorted(self.height_edges, float(z_m), side="right") - 1)
        return index if 0 <= index < self.num_slices else None

    def world_to_cell(self, x_m: float, y_m: float) -> tuple[int, int] | None:
        if not np.isfinite((x_m, y_m)).all():
            return None
        ix = math.floor((float(x_m) - float(self.origin_xy[0])) / self.resolution_m)
        iy = math.floor((float(y_m) - float(self.origin_xy[1])) / self.resolution_m)
        if 0 <= ix < self.size_xy[0] and 0 <= iy < self.size_xy[1]:
            return int(ix), int(iy)
        return None

    def cell_to_world(self, ix: int, iy: int) -> tuple[float, float]:
        if not (0 <= int(ix) < self.size_xy[0] and 0 <= int(iy) < self.size_xy[1]):
            raise IndexError("cell index outside map")
        return (
            float(self.origin_xy[0]) + (int(ix) + 0.5) * self.resolution_m,
            float(self.origin_xy[1]) + (int(iy) + 0.5) * self.resolution_m,
        )

    def set_cell(self, slice_index: int, ix: int, iy: int, state: CellState) -> None:
        if not (0 <= int(slice_index) < self.num_slices):
            raise IndexError("height slice outside map")
        if not (0 <= int(ix) < self.size_xy[0] and 0 <= int(iy) < self.size_xy[1]):
            return
        state = CellState(int(state))
        current = CellState(int(self.states[slice_index, iy, ix]))
        # Occupied wins over free observations.  The map is deliberately
        # conservative and does not erase an obstacle without an explicit reset.
        if state == CellState.OCCUPIED or current != CellState.OCCUPIED:
            self.states[slice_index, iy, ix] = int(state)
        self.observation_count[slice_index, iy, ix] += 1

    def state_at(self, x_m: float, y_m: float, z_m: float) -> CellState:
        cell = self.world_to_cell(x_m, y_m)
        slice_index = self.height_slice_index(z_m)
        if cell is None or slice_index is None:
            return CellState.UNKNOWN
        ix, iy = cell
        return CellState(int(self.states[slice_index, iy, ix]))

    def mark_world(self, x_m: float, y_m: float, z_m: float, state: CellState) -> None:
        cell = self.world_to_cell(x_m, y_m)
        slice_index = self.height_slice_index(z_m)
        if cell is None or slice_index is None:
            return
        self.set_cell(slice_index, cell[0], cell[1], state)

    def _mark_ray(
        self,
        sensor_xy: tuple[float, float],
        endpoint_xy: tuple[float, float],
        slice_index: int,
        hit_obstacle: bool,
    ) -> None:
        start = self.world_to_cell(*sensor_xy)
        if start is None:
            return
        start_point = np.asarray(sensor_xy, dtype=np.float64)
        end_point = np.asarray(endpoint_xy, dtype=np.float64)
        distance = float(np.linalg.norm(end_point - start_point))
        if distance < 1.0e-9:
            cells = [start]
        else:
            sample_count = max(1, int(math.ceil(distance / (self.resolution_m * 0.5))))
            cells = []
            for alpha in np.linspace(0.0, 1.0, sample_count + 1):
                point = start_point * (1.0 - alpha) + end_point * alpha
                cell = self.world_to_cell(float(point[0]), float(point[1]))
                if cell is not None and (not cells or cell != cells[-1]):
                    cells.append(cell)
        if not cells:
            return
        endpoint_cell = self.world_to_cell(*endpoint_xy)
        free_cells = cells[:-1] if hit_obstacle and endpoint_cell is not None else cells
        for ix, iy in free_cells:
            self.set_cell(slice_index, ix, iy, CellState.FREE)
        if hit_obstacle and endpoint_cell is not None:
            self.set_cell(slice_index, endpoint_cell[0], endpoint_cell[1], CellState.OCCUPIED)

    def update_rgbd(
        self,
        depth_m: np.ndarray,
        intrinsics: CameraIntrinsics,
        camera_to_world: np.ndarray,
        *,
        rgb: np.ndarray | None = None,
        stride_px: int = 1,
        min_depth_m: float = 0.15,
        max_depth_m: float = 8.0,
        max_range_epsilon_m: float = 0.03,
    ) -> dict[str, int]:
        """Integrate one RGB-D frame using ray clearing.

        A finite return strictly shorter than ``max_depth_m`` marks its endpoint
        OCCUPIED.  A finite value at the configured max range only clears the
        ray; invalid/absent depth is ignored and therefore remains UNKNOWN.
        The transform uses the conventional camera frame (x right, y down, z
        forward) and maps homogeneous camera points into world coordinates.
        """
        intrinsics.validate()
        depth = np.asarray(depth_m, dtype=np.float32)
        if depth.shape != (intrinsics.height_px, intrinsics.width_px):
            raise ValueError("depth_m shape does not match camera intrinsics")
        if rgb is not None:
            rgb_array = np.asarray(rgb)
            if rgb_array.shape[:2] != depth.shape or rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
                raise ValueError("rgb must have shape HxWx3 matching depth_m")
        transform = np.asarray(camera_to_world, dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("camera_to_world must be a finite 4x4 matrix")
        stride_px = int(stride_px)
        if stride_px < 1 or min_depth_m <= 0.0 or max_depth_m <= min_depth_m:
            raise ValueError("invalid depth integration limits")
        if max_range_epsilon_m < 0.0:
            raise ValueError("max_range_epsilon_m cannot be negative")

        sensor_world = transform[:3, 3]
        counts = {"valid_depth": 0, "rays_cleared": 0, "occupied_endpoints": 0}
        for v in range(0, intrinsics.height_px, stride_px):
            for u in range(0, intrinsics.width_px, stride_px):
                depth_value = float(depth[v, u])
                if not math.isfinite(depth_value) or not (min_depth_m <= depth_value <= max_depth_m):
                    continue
                camera_point = np.asarray(
                    [
                        (float(u) - intrinsics.cx_px) * depth_value / intrinsics.fx_px,
                        (float(v) - intrinsics.cy_px) * depth_value / intrinsics.fy_px,
                        depth_value,
                        1.0,
                    ],
                    dtype=np.float64,
                )
                world_point = (transform @ camera_point)[:3]
                slice_index = self.height_slice_index(float(world_point[2]))
                if slice_index is None or not np.isfinite(world_point).all():
                    continue
                hit = depth_value < max_depth_m - max_range_epsilon_m
                self._mark_ray(
                    (float(sensor_world[0]), float(sensor_world[1])),
                    (float(world_point[0]), float(world_point[1])),
                    slice_index,
                    hit,
                )
                counts["valid_depth"] += 1
                counts["rays_cleared"] += 1
                counts["occupied_endpoints"] += int(hit)
        return counts

    def segment_state(
        self,
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
        z_values_m: tuple[float, ...] | list[float] | None = None,
        *,
        sample_step_m: float | None = None,
    ) -> CellState:
        """Classify a segment conservatively: OCCUPIED wins, then UNKNOWN."""
        heights = tuple(z_values_m) if z_values_m is not None else tuple(
            float((self.height_edges[index] + self.height_edges[index + 1]) * 0.5)
            for index in range(self.num_slices)
        )
        if not heights:
            return CellState.UNKNOWN
        start = np.asarray(start_xy, dtype=np.float64)
        end = np.asarray(end_xy, dtype=np.float64)
        distance = float(np.linalg.norm(end - start))
        step = float(sample_step_m or self.resolution_m * 0.5)
        if step <= 0.0:
            raise ValueError("sample_step_m must be positive")
        sample_count = max(1, int(math.ceil(distance / step)))
        saw_unknown = False
        for alpha in np.linspace(0.0, 1.0, sample_count + 1):
            point = start * (1.0 - alpha) + end * alpha
            for z_m in heights:
                state = self.state_at(float(point[0]), float(point[1]), float(z_m))
                if state == CellState.OCCUPIED:
                    return CellState.OCCUPIED
                if state == CellState.UNKNOWN:
                    saw_unknown = True
        return CellState.UNKNOWN if saw_unknown else CellState.FREE
