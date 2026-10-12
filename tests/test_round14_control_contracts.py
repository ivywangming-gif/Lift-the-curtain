from __future__ import annotations

import numpy as np
import pytest

from lift_curtain.control.point_goal import PointGoalConfig, evaluate_goal
from lift_curtain.control.progress_monitor import (
    DeploymentProgressMonitor,
    PhaseAwareProgressMonitor,
    ProgressPhase,
    ProgressMonitorConfig,
    ProgressSample,
    ProgressStatus,
)
from lift_curtain.navigation.oracle_grid_map import OracleGridMap
from lift_curtain.navigation.round12_state_machine import CandidateRegion, ProbeDecision
from lift_curtain.navigation.round14_state_machine import Round14NavigationWorkflow, Round14State
from lift_curtain.navigation.spatial_probe import propose_probe_sites, select_next_site
from lift_curtain.control.se2_feedback import PushPathTracker, PushPathTrackerConfig, SE2Reference


def _map() -> tuple[OracleGridMap, CandidateRegion]:
    grid = OracleGridMap.filled_free(origin_xy=(-1.0, -1.5), resolution_m=0.1, size_xy=(32, 30))
    region = CandidateRegion((0.75, 0.95), (-0.6, 0.6), clear_margin_m=0.04)
    grid.mark_rectangle(region.x_limits_m, region.y_limits_m, 2)
    return grid, region


def test_point_goal_can_pass_when_pose_goal_fails_only_yaw() -> None:
    result = evaluate_goal(
        final_xy=(1.35, 0.03), goal_xy=(1.35, 0.0),
        speed_samples_mps=[0.01] * 20, speed_sample_period_s=0.02,
        finite_no_fall=True, final_yaw_rad=2.6, goal_yaw_rad=0.0,
    )
    assert result.point_goal_pass
    assert not result.pose_goal_pass
    assert result.reason == "PASS_POINT_GOAL"


def test_round14_state_machine_keeps_clearance_and_point_goal_separate() -> None:
    grid, region = _map()
    workflow = Round14NavigationWorkflow(grid, region)
    workflow.reach_precontact()
    workflow.record_probe(ProbeDecision.RESPONSIVE)
    assert workflow.state == Round14State.CAUTIOUS_TRAVERSE
    assert workflow.verify_candidate_clearance(body_back_extents_x_m=(1.1, 1.2), stable=True, finite=True)
    workflow.complete_release_and_recover(stable=True, finite=True)
    evaluation = evaluate_goal(
        final_xy=(1.35, 0.0), goal_xy=(1.35, 0.0), speed_samples_mps=[0.0] * 20,
        speed_sample_period_s=0.02, finite_no_fall=True,
        final_yaw_rad=2.5, goal_yaw_rad=0.0,
    )
    workflow.reach_point_goal(evaluation)
    assert workflow.state == Round14State.POINTGOAL_REACHED
    assert workflow.candidate_clearance_verified


def test_blocked_round14_updates_map_and_keeps_original_snapshot() -> None:
    grid, region = _map()
    workflow = Round14NavigationWorkflow(grid, region)
    original = workflow.initial_oracle_map.labels.copy()
    workflow.reach_precontact()
    workflow.record_probe(ProbeDecision.BLOCKED)
    changed_before = workflow.map_update_count
    workflow.complete_release_and_recover(stable=True, finite=True)
    assert workflow.map_update_count > changed_before
    assert workflow.navigation_map.label_at(0.85, 0.0) == 1
    np.testing.assert_array_equal(workflow.initial_oracle_map.labels, original)


def test_progress_monitor_requires_hysteresis_before_stop_and_recovers() -> None:
    monitor = DeploymentProgressMonitor(ProgressMonitorConfig(required_stagnant_windows=3, window_duration_s=0.2))
    reading = None
    for index in range(20):
        reading = monitor.update(ProgressSample(index * 0.1, (0.0, 0.0), (0.2, 0.0)))
    assert reading is not None and reading.status == ProgressStatus.STAGNANT
    assert reading.stop_requested
    recovered = monitor.update(ProgressSample(2.1, (0.04, 0.0), (0.2, 0.0)))
    assert recovered.status == ProgressStatus.MOVING
    assert not recovered.stop_requested


def test_phase_aware_monitor_does_not_carry_traverse_history_into_stand_or_goal() -> None:
    monitor = PhaseAwareProgressMonitor(
        ProgressMonitorConfig(
            required_stagnant_windows=2,
            window_duration_s=0.2,
            minimum_progress_m=0.015,
        )
    )
    # A commanded but immobile traverse is allowed to become stagnant.
    for index in range(12):
        reading = monitor.update(
            ProgressSample(index * 0.1, (0.0, 0.0), (0.2, 0.0)),
            phase=ProgressPhase.CAUTIOUS_TRAVERSE,
            requires_forward_progress=True,
            segment_id="candidate-0",
        )
    assert reading.stop_requested
    # Phase changes explicitly clear the finite window and cannot inherit STUCK.
    stand = monitor.update(
        ProgressSample(2.0, (0.0, 0.0), (0.0, 0.0)),
        phase=ProgressPhase.INACTIVE,
        requires_forward_progress=False,
    )
    assert stand.status == ProgressStatus.MOVING
    assert not stand.stop_requested
    resumed = monitor.update(
        ProgressSample(2.1, (0.0, 0.0), (0.2, 0.0)),
        phase=ProgressPhase.CAUTIOUS_TRAVERSE,
        requires_forward_progress=True,
        segment_id="candidate-1",
    )
    assert resumed.reason == "WARMUP"
    assert not resumed.stop_requested


def test_phase_aware_monitor_deactivates_after_robot_owned_clearance() -> None:
    monitor = PhaseAwareProgressMonitor(
        ProgressMonitorConfig(required_stagnant_windows=2, window_duration_s=0.2)
    )
    for index in range(5):
        monitor.update(
            ProgressSample(index * 0.1, (0.05 * index, 0.0), (0.2, 0.0)),
            phase=ProgressPhase.CAUTIOUS_TRAVERSE,
            requires_forward_progress=True,
            segment_id="candidate-0",
        )
    cleared = monitor.update(
        ProgressSample(0.6, (0.25, 0.0), (0.2, 0.0)),
        phase=ProgressPhase.CAUTIOUS_TRAVERSE,
        requires_forward_progress=False,
        candidate_clearance_verified=True,
        segment_id="candidate-0",
    )
    assert cleared.reason == "CANDIDATE_CLEARANCE_VERIFIED"
    assert not cleared.stop_requested
    assert not monitor.active


def test_spatial_probe_sites_are_workspace_bounded_and_distinct() -> None:
    sites = propose_probe_sites(
        candidate_x_limits_m=(0.79, 0.91), candidate_y_limits_m=(-0.6, 0.6),
        precontact_offset_m=0.18, lateral_offset_m=0.15,
        workspace_y_limits_m=(-0.4, 0.4), body_clearance_y_limits_m=(-0.3, 0.3),
    )
    assert len(sites) == 2
    assert sites[0].name != sites[1].name
    assert select_next_site(sites, {sites[0].name}) == sites[1]


def test_se2_feedback_uses_world_xy_yaw_and_rate_limits_commands() -> None:
    tracker = PushPathTracker(PushPathTrackerConfig(
        position_gain_xy=(1.0, 1.0), heading_gain=1.0,
        vx_max=0.30, vy_limit=0.20, wz_limit=0.35,
        max_command_step=(0.05, 0.05, 0.05),
    ))
    # A 90-degree base yaw makes the world-x error appear as body -y.  The
    # nonzero target yaw must independently produce a bounded yaw command.
    command = tracker(
        (0.0, 0.0, np.pi / 2.0),
        SE2Reference((0.4, 0.0), yaw=0.0),
    )
    assert command[0] == pytest.approx(0.0, abs=1.0e-8)
    assert command[1] == pytest.approx(-0.20, abs=1.0e-8)
    assert command[2] == pytest.approx(-0.35, abs=1.0e-8)
    second = tracker(
        (0.0, 0.0, np.pi / 2.0),
        SE2Reference((0.4, 0.2), yaw=-np.pi / 2.0),
    )
    assert abs(float(second[0])) <= 0.30
    assert abs(float(second[1])) <= 0.20
    assert abs(float(second[2])) <= 0.35
    assert np.all(np.abs(second - command) <= np.asarray((0.05, 0.05, 0.05)) + 1.0e-8)
