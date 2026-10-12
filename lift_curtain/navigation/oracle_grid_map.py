"""Known 0/1/2 grid-map planning without object-class access.

The map is an engineering abstraction for the current phase.  Label ``2``
means only that an area may be probed; it is deliberately not a free-space
label and carries no curtain/rigid identity.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, IntEnum
import heapq
import math
from typing import Iterable, Sequence

import numpy as np

from .interaction_memory import InteractionMemory, InteractionStatus


class OracleCellLabel(IntEnum):
    FREE = 0
    FIXED_OBSTACLE = 1
    INTERACTION_CANDIDATE = 2


class KnownMapPlanStatus(str, Enum):
    GOAL_REACHED = "GOAL_REACHED"
    DIRECT = "DIRECT"
    APPROACH_CANDIDATE = "APPROACH_CANDIDATE"
    DETOUR = "DETOUR"
    UNRESOLVED = "UNRESOLVED"


def _finite_xy(value: Iterable[float], *, name: str) -> tuple[float, float]:
    result = np.asarray(tuple(value), dtype=np.float64).reshape(-1)
    if result.size != 2 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 2-vector")
    return float(result[0]), float(result[1])


@dataclass
class OracleGridMap:
    """A fully specified planar map with labels 0, 1, and 2."""

    origin_xy: tuple[float, float]
    resolution_m: float
    size_xy: tuple[int, int]
    labels: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.origin_xy = _finite_xy(self.origin_xy, name="origin_xy")
        self.resolution_m = float(self.resolution_m)
        if self.resolution_m <= 0.0 or not math.isfinite(self.resolution_m):
            raise ValueError("resolution_m must be finite and positive")
        if len(self.size_xy) != 2 or min(int(value) for value in self.size_xy) < 1:
            raise ValueError("size_xy must contain two positive dimensions")
        self.size_xy = (int(self.size_xy[0]), int(self.size_xy[1]))
        if self.labels is None:
            # An explicit constructor is still required to populate the map;
            # defaulting to fixed obstacles avoids silently treating unknown
            # cells as free when a scenario forgets to label them.
            self.labels = np.full(
                (self.size_xy[1], self.size_xy[0]),
                int(OracleCellLabel.FIXED_OBSTACLE),
                dtype=np.int8,
            )
        else:
            labels = np.asarray(self.labels, dtype=np.int8)
            if labels.shape != (self.size_xy[1], self.size_xy[0]):
                raise ValueError("labels shape must be (size_y, size_x)")
            if not np.isin(labels, [0, 1, 2]).all():
                raise ValueError("labels must contain only 0, 1, or 2")
            self.labels = labels.copy()

    @classmethod
    def filled_free(
        cls,
        *,
        origin_xy: tuple[float, float],
        resolution_m: float,
        size_xy: tuple[int, int],
    ) -> "OracleGridMap":
        labels = np.full((int(size_xy[1]), int(size_xy[0])), int(OracleCellLabel.FREE), dtype=np.int8)
        return cls(origin_xy, resolution_m, size_xy, labels)

    @property
    def width_m(self) -> float:
        return float(self.size_xy[0] * self.resolution_m)

    @property
    def depth_m(self) -> float:
        return float(self.size_xy[1] * self.resolution_m)

    def world_to_grid(self, x_m: float, y_m: float) -> tuple[int, int] | None:
        if not np.isfinite((x_m, y_m)).all():
            return None
        ix = math.floor((float(x_m) - self.origin_xy[0]) / self.resolution_m)
        iy = math.floor((float(y_m) - self.origin_xy[1]) / self.resolution_m)
        if 0 <= ix < self.size_xy[0] and 0 <= iy < self.size_xy[1]:
            return int(ix), int(iy)
        return None

    def grid_to_world(self, ix: int, iy: int) -> tuple[float, float]:
        if not (0 <= int(ix) < self.size_xy[0] and 0 <= int(iy) < self.size_xy[1]):
            raise IndexError("grid index outside map")
        return (
            self.origin_xy[0] + (int(ix) + 0.5) * self.resolution_m,
            self.origin_xy[1] + (int(iy) + 0.5) * self.resolution_m,
        )

    def label_at(self, x_m: float, y_m: float) -> OracleCellLabel:
        cell = self.world_to_grid(x_m, y_m)
        if cell is None:
            return OracleCellLabel.FIXED_OBSTACLE
        return OracleCellLabel(int(self.labels[cell[1], cell[0]]))

    def set_label(self, ix: int, iy: int, label: OracleCellLabel) -> None:
        if not (0 <= int(ix) < self.size_xy[0] and 0 <= int(iy) < self.size_xy[1]):
            return
        self.labels[int(iy), int(ix)] = int(OracleCellLabel(label))

    def set_world_label(self, x_m: float, y_m: float, label: OracleCellLabel) -> None:
        cell = self.world_to_grid(x_m, y_m)
        if cell is not None:
            self.set_label(cell[0], cell[1], label)

    def clone(self) -> "OracleGridMap":
        """Return an independent map snapshot for episode-level auditing."""
        return OracleGridMap(
            origin_xy=self.origin_xy,
            resolution_m=self.resolution_m,
            size_xy=self.size_xy,
            labels=self.labels.copy(),
        )

    def label_counts(self) -> dict[OracleCellLabel, int]:
        return {
            label: int(np.count_nonzero(self.labels == int(label)))
            for label in OracleCellLabel
        }

    def transition_cells(
        self,
        cells: Iterable[tuple[int, int]],
        *,
        expected: OracleCellLabel,
        replacement: OracleCellLabel,
    ) -> int:
        """Apply a guarded label transition and return the changed-cell count.

        The guard prevents a failed probe from silently overwriting fixed
        obstacles or already-updated cells.  It is deliberately independent
        of scene kind and physical outcome.
        """
        changed = 0
        expected_value = int(OracleCellLabel(expected))
        replacement_value = int(OracleCellLabel(replacement))
        for ix, iy in cells:
            if not (0 <= int(ix) < self.size_xy[0] and 0 <= int(iy) < self.size_xy[1]):
                continue
            if int(self.labels[int(iy), int(ix)]) != expected_value:
                continue
            self.labels[int(iy), int(ix)] = replacement_value
            changed += 1
        return changed

    def transition_rectangle(
        self,
        x_limits_m: tuple[float, float],
        y_limits_m: tuple[float, float],
        *,
        expected: OracleCellLabel,
        replacement: OracleCellLabel,
    ) -> int:
        """Guardedly update a world-aligned rectangular map region."""
        x0, x1 = sorted(map(float, x_limits_m))
        y0, y1 = sorted(map(float, y_limits_m))
        cells: list[tuple[int, int]] = []
        for iy in range(self.size_xy[1]):
            for ix in range(self.size_xy[0]):
                x, y = self.grid_to_world(ix, iy)
                if x0 <= x <= x1 and y0 <= y <= y1:
                    cells.append((ix, iy))
        return self.transition_cells(cells, expected=expected, replacement=replacement)

    def mark_verified_opening_free(self, center_xy: tuple[float, float], radius_m: float) -> int:
        """Promote observed candidate cells only after a geometric opening check.

        This is intentionally driven by an external deployment-observable
        opening measurement.  It does not inspect object identity or simulator
        passability, and it never changes fixed label-1 cells.
        """
        center = np.asarray(_finite_xy(center_xy, name="center_xy"), dtype=np.float64)
        radius = float(radius_m)
        if radius < 0.0 or not math.isfinite(radius):
            raise ValueError("radius_m must be finite and non-negative")
        changed = 0
        for iy in range(self.size_xy[1]):
            for ix in range(self.size_xy[0]):
                if self.labels[iy, ix] != int(OracleCellLabel.INTERACTION_CANDIDATE):
                    continue
                point = np.asarray(self.grid_to_world(ix, iy), dtype=np.float64)
                if float(np.linalg.norm(point - center)) <= radius + 1.0e-9:
                    self.labels[iy, ix] = int(OracleCellLabel.FREE)
                    changed += 1
        return changed

    def mark_rectangle(
        self,
        x_limits_m: tuple[float, float],
        y_limits_m: tuple[float, float],
        label: OracleCellLabel,
    ) -> None:
        x0, x1 = sorted(map(float, x_limits_m))
        y0, y1 = sorted(map(float, y_limits_m))
        for iy in range(self.size_xy[1]):
            for ix in range(self.size_xy[0]):
                x, y = self.grid_to_world(ix, iy)
                if x0 <= x <= x1 and y0 <= y <= y1:
                    self.set_label(ix, iy, label)

    def weighted_path(
        self,
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        *,
        candidate_penalty_m: float = 0.30,
        require_candidate: bool = False,
        allow_diagonal: bool = False,
    ) -> list[tuple[float, float]] | None:
        """Find a lowest-cost path over FREE and candidate cells.

        Label 1 is always impassable.  Label 2 is traversable only for route
        comparison; an executor must still stop at its pre-contact pose and
        obtain a deployment-only probe decision before crossing it.
        """
        start = self.world_to_grid(*start_xy)
        goal = self.world_to_grid(*goal_xy)
        penalty = float(candidate_penalty_m)
        if start is None or goal is None or penalty < 0.0 or not math.isfinite(penalty):
            return None
        allowed = {int(OracleCellLabel.FREE), int(OracleCellLabel.INTERACTION_CANDIDATE)}
        if int(self.labels[start[1], start[0]]) not in allowed:
            return None
        if int(self.labels[goal[1], goal[0]]) not in allowed:
            return None
        if allow_diagonal:
            neighbors = (
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1), (0, 1),
                (1, -1), (1, 0), (1, 1),
            )
        else:
            neighbors = ((-1, 0), (0, -1), (0, 1), (1, 0))

        # The boolean state keeps the candidate requirement explicit instead
        # of inferring it from the final path after the search.
        start_used = int(self.labels[start[1], start[0]]) == int(OracleCellLabel.INTERACTION_CANDIDATE)
        start_state = (start[0], start[1], start_used)
        distances: dict[tuple[int, int, bool], float] = {start_state: 0.0}
        parent: dict[tuple[int, int, bool], tuple[int, int, bool] | None] = {start_state: None}
        queue: list[tuple[float, int, int, bool]] = [(0.0, start[0], start[1], start_used)]
        target_state: tuple[int, int, bool] | None = None
        while queue:
            distance, ix, iy, used_candidate = heapq.heappop(queue)
            state = (ix, iy, used_candidate)
            if distance > distances.get(state, math.inf) + 1.0e-12:
                continue
            if (ix, iy) == goal and (used_candidate or not require_candidate):
                target_state = state
                break
            for dx, dy in neighbors:
                nx, ny = ix + dx, iy + dy
                if not (0 <= nx < self.size_xy[0] and 0 <= ny < self.size_xy[1]):
                    continue
                if dx and dy and not allow_diagonal:
                    continue
                label = int(self.labels[ny, nx])
                if label not in allowed:
                    continue
                # Prevent diagonal corner cutting when diagonal search is
                # explicitly requested.
                if dx and dy and allow_diagonal:
                    side_a = int(self.labels[iy, nx])
                    side_b = int(self.labels[ny, ix])
                    if side_a not in allowed or side_b not in allowed:
                        continue
                step = self.resolution_m * (math.sqrt(2.0) if dx and dy else 1.0)
                if label == int(OracleCellLabel.INTERACTION_CANDIDATE):
                    step += penalty
                next_used = used_candidate or label == int(OracleCellLabel.INTERACTION_CANDIDATE)
                next_state = (nx, ny, next_used)
                next_distance = distance + step
                if next_distance + 1.0e-12 >= distances.get(next_state, math.inf):
                    continue
                distances[next_state] = next_distance
                parent[next_state] = state
                heapq.heappush(queue, (next_distance, nx, ny, next_used))
        if target_state is None:
            return None
        cells: list[tuple[int, int]] = []
        current: tuple[int, int, bool] | None = target_state
        while current is not None:
            cells.append((current[0], current[1]))
            current = parent[current]
        cells.reverse()
        return [self.grid_to_world(ix, iy) for ix, iy in cells]

    def path_cost(
        self,
        path_xy: Sequence[tuple[float, float]],
        *,
        candidate_penalty_m: float = 0.30,
    ) -> float:
        """Return the same metric used by :meth:`weighted_path`."""
        if len(path_xy) < 2:
            return 0.0
        penalty = float(candidate_penalty_m)
        if penalty < 0.0 or not math.isfinite(penalty):
            raise ValueError("candidate_penalty_m must be finite and non-negative")
        total = 0.0
        for start, end in zip(path_xy, path_xy[1:]):
            start_cell = self.world_to_grid(*start)
            end_cell = self.world_to_grid(*end)
            if start_cell is None or end_cell is None:
                return math.inf
            total += float(np.linalg.norm(np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)))
            if self.labels[end_cell[1], end_cell[0]] == int(OracleCellLabel.INTERACTION_CANDIDATE):
                total += penalty
        return total

    def sample_segment(
        self,
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
        *,
        sample_step_m: float | None = None,
    ) -> list[OracleCellLabel]:
        start = np.asarray(_finite_xy(start_xy, name="start_xy"), dtype=np.float64)
        end = np.asarray(_finite_xy(end_xy, name="end_xy"), dtype=np.float64)
        step = float(sample_step_m or self.resolution_m * 0.5)
        if step <= 0.0 or not math.isfinite(step):
            raise ValueError("sample_step_m must be finite and positive")
        distance = float(np.linalg.norm(end - start))
        count = max(1, int(math.ceil(distance / step)))
        labels: list[OracleCellLabel] = []
        for alpha in np.linspace(0.0, 1.0, count + 1):
            point = start * (1.0 - alpha) + end * alpha
            label = self.label_at(float(point[0]), float(point[1]))
            if not labels or label != labels[-1]:
                labels.append(label)
        return labels

    def segment_labels(
        self,
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
        *,
        sample_step_m: float | None = None,
    ) -> tuple[OracleCellLabel, ...]:
        return tuple(self.sample_segment(start_xy, end_xy, sample_step_m=sample_step_m))

    def segment_is_free(self, start_xy: tuple[float, float], end_xy: tuple[float, float]) -> bool:
        return all(label == OracleCellLabel.FREE for label in self.sample_segment(start_xy, end_xy))

    def free_path(
        self,
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        *,
        allow_diagonal: bool = True,
    ) -> list[tuple[float, float]] | None:
        """Find a path whose cells are explicitly 0; 1 and 2 are blocked."""
        start = self.world_to_grid(*start_xy)
        goal = self.world_to_grid(*goal_xy)
        if start is None or goal is None:
            return None
        if self.labels[start[1], start[0]] != int(OracleCellLabel.FREE):
            return None
        if self.labels[goal[1], goal[0]] != int(OracleCellLabel.FREE):
            return None
        if allow_diagonal:
            neighbors = (
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1), (0, 1),
                (1, -1), (1, 0), (1, 1),
            )
        else:
            neighbors = ((-1, 0), (0, -1), (0, 1), (1, 0))
        queue: deque[tuple[int, int]] = deque([start])
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        while queue:
            current = queue.popleft()
            if current == goal:
                break
            for dx, dy in neighbors:
                candidate = (current[0] + dx, current[1] + dy)
                if candidate in parent:
                    continue
                if not (0 <= candidate[0] < self.size_xy[0] and 0 <= candidate[1] < self.size_xy[1]):
                    continue
                if self.labels[candidate[1], candidate[0]] != int(OracleCellLabel.FREE):
                    continue
                if dx and dy and allow_diagonal:
                    side_a = int(self.labels[current[1], candidate[0]])
                    side_b = int(self.labels[candidate[1], current[0]])
                    if side_a != int(OracleCellLabel.FREE) or side_b != int(OracleCellLabel.FREE):
                        continue
                parent[candidate] = current
                queue.append(candidate)
        if goal not in parent:
            return None
        cells: list[tuple[int, int]] = []
        current: tuple[int, int] | None = goal
        while current is not None:
            cells.append(current)
            current = parent[current]
        cells.reverse()
        return [self.grid_to_world(ix, iy) for ix, iy in cells]

    def to_dict(self) -> dict[str, object]:
        return {
            "origin_xy_m": [float(self.origin_xy[0]), float(self.origin_xy[1])],
            "resolution_m": float(self.resolution_m),
            "size_xy": [int(self.size_xy[0]), int(self.size_xy[1])],
            "labels_yx": self.labels.astype(int).tolist(),
            "label_semantics": {
                "0": "FREE",
                "1": "FIXED_OBSTACLE",
                "2": "INTERACTION_CANDIDATE_UNKNOWN_RESULT",
            },
        }


@dataclass(frozen=True)
class KnownMapPlan:
    status: KnownMapPlanStatus
    waypoint_xy: tuple[float, float] | None
    path_xy: tuple[tuple[float, float], ...]
    reason: str
    encountered_labels: tuple[OracleCellLabel, ...]
    probe_allowed: bool


class KnownMapPlanner:
    """Closed-loop waypoint selector for a known 0/1/2 map.

    The planner accepts a proprioceptive decision string only after a probe.
    It never accepts an object category or a simulator outcome oracle.
    """

    def __init__(self, *, goal_tolerance_m: float = 0.20, precontact_offset_m: float = 0.18):
        if goal_tolerance_m <= 0.0 or precontact_offset_m <= 0.0:
            raise ValueError("planner distances must be positive")
        self.goal_tolerance_m = float(goal_tolerance_m)
        self.precontact_offset_m = float(precontact_offset_m)

    @staticmethod
    def precontact_waypoint(
        candidate_xy: tuple[float, float],
        *,
        approach_yaw_rad: float,
        offset_m: float,
    ) -> tuple[float, float]:
        if offset_m <= 0.0 or not math.isfinite(offset_m):
            raise ValueError("offset_m must be finite and positive")
        return (
            float(candidate_xy[0] - math.cos(approach_yaw_rad) * offset_m),
            float(candidate_xy[1] - math.sin(approach_yaw_rad) * offset_m),
        )

    @staticmethod
    def _path_tuple(path: Sequence[tuple[float, float]] | None) -> tuple[tuple[float, float], ...]:
        return tuple(path or ())

    def plan(
        self,
        grid: OracleGridMap,
        *,
        current_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        candidate_xy: tuple[float, float],
        candidate_normal_yaw_rad: float = 0.0,
        interaction_memory: InteractionMemory,
        probe_decision: str | None = None,
        opening_verified: bool = False,
    ) -> KnownMapPlan:
        current = _finite_xy(current_xy, name="current_xy")
        goal = _finite_xy(goal_xy, name="goal_xy")
        candidate = _finite_xy(candidate_xy, name="candidate_xy")
        if float(np.linalg.norm(np.asarray(goal) - np.asarray(current))) <= self.goal_tolerance_m:
            return KnownMapPlan(KnownMapPlanStatus.GOAL_REACHED, goal, (goal,), "GOAL_WITHIN_TOLERANCE", (), False)

        labels = grid.segment_labels(current, goal)
        if all(label == OracleCellLabel.FREE for label in labels):
            return KnownMapPlan(KnownMapPlanStatus.DIRECT, goal, (current, goal), "DIRECT_ALL_ZERO", labels, False)

        candidate_status = interaction_memory.status_at(candidate, candidate_normal_yaw_rad)
        has_candidate = OracleCellLabel.INTERACTION_CANDIDATE in labels
        if has_candidate and candidate_status == InteractionStatus.UNKNOWN and interaction_memory.can_probe(
            candidate, candidate_normal_yaw_rad
        ):
            waypoint = self.precontact_waypoint(
                candidate,
                approach_yaw_rad=candidate_normal_yaw_rad,
                offset_m=self.precontact_offset_m,
            )
            return KnownMapPlan(
                KnownMapPlanStatus.APPROACH_CANDIDATE,
                waypoint,
                (current, waypoint),
                "CANDIDATE_2_REQUIRES_SINGLE_PROBE",
                labels,
                True,
            )

        # A responsive hand still does not imply passability.  Only an
        # externally measured, multi-height opening can authorize updating the
        # map; this function does not infer it from the response label alone.
        if has_candidate and opening_verified and probe_decision == "RESPONSIVE":
            grid.mark_verified_opening_free(candidate, self.precontact_offset_m)
            labels = grid.segment_labels(current, goal)
            if all(label == OracleCellLabel.FREE for label in labels):
                return KnownMapPlan(
                    KnownMapPlanStatus.DIRECT,
                    goal,
                    (current, goal),
                    "OPENING_VERIFIED_FROM_DEPLOYMENT_GEOMETRY",
                    labels,
                    False,
                )

        path = grid.free_path(current, goal)
        if path:
            return KnownMapPlan(
                KnownMapPlanStatus.DETOUR,
                path[min(1, len(path) - 1)],
                self._path_tuple(path),
                "CANDIDATE_OR_FIXED_OBSTACLE_NOT_FREE_DETOUR_ON_ZERO_CELLS",
                labels,
                False,
            )
        return KnownMapPlan(
            KnownMapPlanStatus.UNRESOLVED,
            None,
            (),
            "NO_PATH_THROUGH_EXPLICIT_ZERO_CELLS",
            labels,
            False,
        )


@dataclass(frozen=True)
class OracleRouteComparison:
    """Evidence comparing the candidate route against a zero-only detour."""

    candidate_path: tuple[tuple[float, float], ...]
    candidate_cost_m: float | None
    zero_detour_path: tuple[tuple[float, float], ...]
    zero_detour_cost_m: float | None
    selected_route: str
    candidate_contains_label_2: bool


def compare_candidate_and_zero_routes(
    grid: OracleGridMap,
    *,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    candidate_penalty_m: float = 0.30,
) -> OracleRouteComparison:
    """Compare a potentially interactive route with a conservative detour."""
    candidate = grid.weighted_path(
        start_xy,
        goal_xy,
        candidate_penalty_m=candidate_penalty_m,
        require_candidate=True,
    )
    zero = grid.free_path(start_xy, goal_xy, allow_diagonal=False)
    candidate_tuple = tuple(candidate or ())
    zero_tuple = tuple(zero or ())
    candidate_cost = grid.path_cost(candidate_tuple, candidate_penalty_m=candidate_penalty_m) if candidate else None
    zero_cost = grid.path_cost(zero_tuple, candidate_penalty_m=candidate_penalty_m) if zero else None
    if candidate_cost is not None and (zero_cost is None or candidate_cost < zero_cost):
        selected = "candidate_2"
    elif zero_cost is not None:
        selected = "zero_detour"
    else:
        selected = "unresolved"
    return OracleRouteComparison(
        candidate_path=candidate_tuple,
        candidate_cost_m=candidate_cost,
        zero_detour_path=zero_tuple,
        zero_detour_cost_m=zero_cost,
        selected_route=selected,
        candidate_contains_label_2=bool(candidate and any(
            grid.label_at(*point) == OracleCellLabel.INTERACTION_CANDIDATE for point in candidate
        )),
    )
