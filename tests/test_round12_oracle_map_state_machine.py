from __future__ import annotations

import numpy as np
import pytest

from lift_curtain.navigation.oracle_grid_map import (
    OracleCellLabel,
    OracleGridMap,
)
from lift_curtain.navigation.round12_state_machine import (
    CandidateRegion,
    ProbeDecision,
    Round12NavigationWorkflow,
    Round12State,
)


def make_round12_map() -> tuple[OracleGridMap, CandidateRegion]:
    grid = OracleGridMap.filled_free(origin_xy=(-1.0, -1.5), resolution_m=0.1, size_xy=(32, 30))
    region = CandidateRegion((0.75, 0.95), (-1.0, 1.0), clear_margin_m=0.04)
    grid.mark_rectangle(region.x_limits_m, region.y_limits_m, OracleCellLabel.INTERACTION_CANDIDATE)
    grid.mark_rectangle((1.05, 1.25), (1.1, 1.4), OracleCellLabel.FIXED_OBSTACLE)
    return grid, region


def test_candidate_route_is_compared_and_preferred_without_treating_two_as_free() -> None:
    grid, region = make_round12_map()
    workflow = Round12NavigationWorkflow(grid, region)
    comparison = workflow.compare_routes(start_xy=(-0.6, 0.0), goal_xy=(1.4, 0.0))
    assert comparison.selected_route == "candidate_2"
    assert comparison.candidate_contains_label_2
    assert comparison.candidate_cost_m is not None
    assert comparison.zero_detour_cost_m is not None
    assert comparison.candidate_cost_m < comparison.zero_detour_cost_m
    assert grid.label_at(0.8, 0.0) == OracleCellLabel.INTERACTION_CANDIDATE
    assert grid.segment_is_free((-0.6, 0.0), (1.4, 0.0)) is False


def test_blocked_probe_updates_navigation_map_2_to_1_and_replans_zero_only() -> None:
    grid, region = make_round12_map()
    workflow = Round12NavigationWorkflow(grid, region)
    initial = workflow.initial_oracle_map.labels.copy()
    workflow.reach_precontact()
    workflow.record_probe(ProbeDecision.BLOCKED)
    assert workflow.state == Round12State.RELEASE_AND_RECOVER
    workflow.complete_release_and_recover(stable=True, finite=True)
    assert workflow.state == Round12State.NAVIGATE_TO_GOAL
    assert workflow.map_update_count > 0
    assert workflow.navigation_map.label_at(0.8, 0.0) == OracleCellLabel.FIXED_OBSTACLE
    assert workflow.initial_oracle_map.label_at(0.8, 0.0) == OracleCellLabel.INTERACTION_CANDIDATE
    np.testing.assert_array_equal(workflow.initial_oracle_map.labels, initial)
    assert not workflow.repeated_probe_allowed()
    zero_path = workflow.navigation_map.free_path((-0.6, 0.0), (1.4, 0.0), allow_diagonal=False)
    assert zero_path is not None
    assert all(workflow.navigation_map.label_at(*point) == OracleCellLabel.FREE for point in zero_path)
    with pytest.raises(RuntimeError):
        workflow.record_probe(ProbeDecision.BLOCKED)


def test_responsive_branch_requires_robot_envelope_clearance_before_release() -> None:
    grid, region = make_round12_map()
    workflow = Round12NavigationWorkflow(grid, region)
    workflow.reach_precontact()
    workflow.record_probe(ProbeDecision.RESPONSIVE)
    assert workflow.state == Round12State.CAUTIOUS_TRAVERSE
    assert not workflow.verify_candidate_clearance(body_back_extents_x_m=(0.80, 0.91), stable=True, finite=True)
    assert workflow.state == Round12State.CAUTIOUS_TRAVERSE
    assert workflow.verify_candidate_clearance(body_back_extents_x_m=(1.00, 1.05), stable=True, finite=True)
    assert workflow.state == Round12State.CLEAR_CANDIDATE_REGION
    workflow.complete_release_and_recover(stable=True, finite=True)
    assert workflow.state == Round12State.NAVIGATE_TO_GOAL
    workflow.reach_goal(stable=True, finite=True)
    assert workflow.state == Round12State.GOAL_REACHED
    assert workflow.navigation_map.label_at(0.8, 0.0) == OracleCellLabel.INTERACTION_CANDIDATE


def test_illegal_direct_probe_to_traverse_and_goal_gate_are_rejected() -> None:
    grid, region = make_round12_map()
    workflow = Round12NavigationWorkflow(grid, region)
    with pytest.raises(RuntimeError):
        workflow.record_probe(ProbeDecision.RESPONSIVE)
    with pytest.raises(RuntimeError):
        workflow.reach_goal(stable=True, finite=True)
