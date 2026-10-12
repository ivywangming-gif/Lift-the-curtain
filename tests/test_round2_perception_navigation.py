from __future__ import annotations

import math

import numpy as np
import pytest

from lift_curtain.navigation.interaction_memory import (
    InteractionMemory,
    InteractionStatus,
)
from lift_curtain.navigation.local_planner import LocalPlanStatus, LocalPlanner
from lift_curtain.perception.opening_geometry import check_common_free_corridor
from lift_curtain.perception.online_occupancy import (
    CameraIntrinsics,
    CellState,
    LocalOccupancyGrid,
)


def make_grid() -> LocalOccupancyGrid:
    return LocalOccupancyGrid(
        origin_xy=(-2.0, -2.0),
        resolution_m=0.1,
        size_xy=(80, 80),
        height_edges_m=(0.0, 0.8, 1.2, 2.0),
    )


def fill_box(grid: LocalOccupancyGrid, slice_index: int, x_limits: tuple[float, float], y_limits: tuple[float, float], state: CellState) -> None:
    for ix in range(grid.size_xy[0]):
        for iy in range(grid.size_xy[1]):
            x, y = grid.cell_to_world(ix, iy)
            if x_limits[0] <= x <= x_limits[1] and y_limits[0] <= y <= y_limits[1]:
                grid.set_cell(slice_index, ix, iy, state)


def test_unknown_goal_without_global_map_is_unresolved() -> None:
    grid = make_grid()
    plan = LocalPlanner().plan(
        grid,
        current_xy=(0.0, 0.0),
        goal_xy=(1.5, 0.0),
        height_values_m=(0.4, 1.0, 1.6),
    )
    assert plan.status == LocalPlanStatus.UNRESOLVED
    assert plan.waypoint_xy is None
    assert "UNKNOWN" in plan.reason


def test_invalid_depth_and_occluded_space_remain_unknown() -> None:
    grid = make_grid()
    intrinsics = CameraIntrinsics(5, 3, 50.0, 50.0, 2.0, 1.0)
    depth = np.full((3, 5), np.nan, dtype=np.float32)
    depth[1, 2] = 1.0
    grid.update_rgbd(depth, intrinsics, np.eye(4), stride_px=1, max_depth_m=4.0)
    assert grid.state_at(0.0, 0.0, 1.0) == CellState.OCCUPIED
    assert grid.state_at(0.2, 0.0, 1.0) == CellState.UNKNOWN
    assert grid.state_at(0.0, 0.0, 0.4) == CellState.UNKNOWN


def test_max_range_clears_ray_but_does_not_create_occupied_endpoint() -> None:
    grid = make_grid()
    intrinsics = CameraIntrinsics(3, 1, 10.0, 10.0, 1.0, 0.0)
    depth = np.asarray([[4.0, 4.0, 4.0]], dtype=np.float32)
    camera_to_world = np.asarray(
        [[0.0, 0.0, 1.0, 0.0],
         [1.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 1.0],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    grid.update_rgbd(depth, intrinsics, camera_to_world, max_depth_m=4.0)
    assert grid.state_at(0.0, 0.0, 1.0) == CellState.FREE
    assert grid.state_at(0.3, 0.0, 1.0) != CellState.OCCUPIED


def test_color_change_cannot_change_depth_map_conclusion() -> None:
    intrinsics = CameraIntrinsics(3, 1, 10.0, 10.0, 1.0, 0.0)
    depth = np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32)
    grid_a = make_grid()
    grid_b = make_grid()
    grid_a.update_rgbd(depth, intrinsics, np.eye(4), rgb=np.zeros((1, 3, 3), dtype=np.uint8))
    grid_b.update_rgbd(depth, intrinsics, np.eye(4), rgb=np.full((1, 3, 3), 255, dtype=np.uint8))
    np.testing.assert_array_equal(grid_a.states, grid_b.states)


def test_common_corridor_requires_same_lateral_interval_at_all_heights() -> None:
    grid = make_grid()
    fill_box(grid, 0, (-0.05, 0.45), (-0.8, -0.2), CellState.FREE)
    fill_box(grid, 1, (-0.05, 0.45), (0.2, 0.8), CellState.FREE)
    result = check_common_free_corridor(
        grid,
        center_xy=(0.0, 0.0),
        forward_yaw_rad=0.0,
        lateral_limits_m=(-0.9, 0.9),
        forward_limits_m=(0.0, 0.4),
        required_width_m=0.5,
        height_slices=(0, 1),
    )
    assert not result.passed
    assert result.common_width_m == 0.0
    assert result.unknown_samples > 0


def test_common_corridor_passes_only_for_verified_shared_width() -> None:
    grid = make_grid()
    for slice_index in (0, 1):
        fill_box(grid, slice_index, (-0.05, 0.45), (-0.2, 0.75), CellState.FREE)
    result = check_common_free_corridor(
        grid,
        center_xy=(0.0, 0.0),
        forward_yaw_rad=0.0,
        lateral_limits_m=(-0.4, 0.9),
        forward_limits_m=(0.0, 0.4),
        required_width_m=0.6,
        height_slices=(0, 1),
    )
    assert result.passed
    assert result.common_width_m >= 0.6


def test_failed_probe_is_not_reenabled_by_detector_id_or_viewpoint() -> None:
    memory = InteractionMemory()
    assert memory.reserve_probe((1.0, 0.0), candidate_id="component-17")
    memory.record_response((1.0, 0.0), InteractionStatus.BLOCKED, failure_reason="rigid_response")
    assert memory.status_at((1.05, 0.02)) == InteractionStatus.DEFERRED_THIS_EPISODE
    assert not memory.can_probe((1.08, -0.03), candidate_id="component-99")
    assert memory.probes_remaining == 0


def test_region_b_is_not_marked_physically_blocked_by_region_a_failure() -> None:
    memory = InteractionMemory()
    assert memory.reserve_probe((1.0, 0.0))
    memory.record_response((1.0, 0.0), InteractionStatus.BLOCKED)
    assert memory.status_at((1.8, 0.0)) == InteractionStatus.UNKNOWN
    assert not memory.can_probe((1.8, 0.0))  # episode budget, not a global wall label


def test_movable_does_not_become_passable_without_current_opening() -> None:
    memory = InteractionMemory()
    assert memory.reserve_probe((1.0, 0.0))
    memory.record_response((1.0, 0.0), InteractionStatus.MOVABLE)
    with pytest.raises(ValueError):
        memory.mark_currently_clear((1.0, 0.0), opening_verified=False)
    record = memory.record_response((1.0, 0.0), InteractionStatus.PASSABLE, opening_verified=False)
    assert record.status == InteractionStatus.MOVABLE
    record = memory.record_response((1.0, 0.0), InteractionStatus.PASSABLE, opening_verified=True)
    assert record.status == InteractionStatus.PASSABLE


def test_currently_closed_barrier_cannot_be_entered_from_memory_alone() -> None:
    memory = InteractionMemory()
    assert memory.reserve_probe((1.0, 0.0))
    memory.record_response((1.0, 0.0), InteractionStatus.MOVABLE)
    grid = make_grid()
    fill_box(grid, 0, (0.9, 1.1), (-0.4, 0.4), CellState.OCCUPIED)
    fill_box(grid, 1, (0.9, 1.1), (-0.4, 0.4), CellState.OCCUPIED)
    direct = grid.segment_state((0.0, 0.0), (1.5, 0.0), (0.4, 1.0))
    assert direct == CellState.OCCUPIED
    assert memory.status_at((1.0, 0.0)) == InteractionStatus.MOVABLE


def test_probe_budget_has_a_hard_episode_cap() -> None:
    memory = InteractionMemory()
    assert memory.reserve_probe((1.0, 0.0), candidate_id="a")
    memory.record_response((1.0, 0.0), InteractionStatus.BLOCKED)
    assert not memory.reserve_probe((2.0, 0.0), candidate_id="b")
    memory.begin_episode()
    assert memory.reserve_probe((2.0, 0.0), candidate_id="c")
