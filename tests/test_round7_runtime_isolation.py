from __future__ import annotations

import numpy as np

from lift_curtain.control.deployment_observation import (
    ContactDecision,
    DeploymentObservation,
    EvaluationGroundTruth,
    classify_proprioceptive_response,
    privileged_information_leakage_audit,
)
from lift_curtain.navigation.interaction_memory import InteractionMemory, InteractionStatus
from lift_curtain.navigation.oracle_grid_map import (
    KnownMapPlanStatus,
    KnownMapPlanner,
    OracleCellLabel,
    OracleGridMap,
)


def make_observation(
    *,
    hand_x: float,
    hand_vx: float,
    effort: float,
    q_cmd: float = 0.0,
) -> DeploymentObservation:
    return DeploymentObservation(
        q=np.zeros(2),
        qdot=np.zeros(2),
        q_cmd=np.full(2, q_cmd),
        motor_effort=np.full(2, effort),
        previous_action=np.zeros(2),
        hand_fk_position_w=np.asarray((hand_x, 0.0, 0.9)),
        hand_fk_velocity_w=np.asarray((hand_vx, 0.0, 0.0)),
        base_position_w=np.asarray((0.0, 0.0, 0.8)),
        localization_assumption="test_localization",
    )


def make_map() -> OracleGridMap:
    grid = OracleGridMap.filled_free(origin_xy=(-1.0, -1.0), resolution_m=0.1, size_xy=(32, 24))
    grid.mark_rectangle((0.65, 0.75), (-0.15, 0.15), OracleCellLabel.INTERACTION_CANDIDATE)
    grid.mark_rectangle((0.90, 1.00), (-0.60, 0.60), OracleCellLabel.FIXED_OBSTACLE)
    return grid


def test_runtime_and_evaluation_fields_are_disjoint() -> None:
    observation = make_observation(hand_x=0.0, hand_vx=0.01, effort=0.0)
    truth = EvaluationGroundTruth(
        object_kind="curtain",
        simulator_contact_flag=True,
        true_hand_obstacle_gap_m=0.0,
        curtain_node_displacement_m=0.01,
        true_passability=True,
    )
    runtime = observation.to_runtime_dict()
    evaluation = truth.to_evaluation_dict()
    assert set(runtime).isdisjoint(evaluation)
    assert "object_kind" not in runtime
    assert "simulator_contact_flag" not in runtime


def test_contact_rule_uses_proprio_history_only() -> None:
    baseline = [make_observation(hand_x=0.01 * i, hand_vx=0.01, effort=0.0) for i in range(5)]
    responsive = [make_observation(hand_x=0.01 + 0.012 * i, hand_vx=0.012, effort=0.0) for i in range(5)]
    blocked = [make_observation(hand_x=0.01, hand_vx=0.0, effort=0.25, q_cmd=0.12) for _ in range(5)]
    assert classify_proprioceptive_response(responsive, baseline=baseline)[0] == ContactDecision.RESPONSIVE
    assert classify_proprioceptive_response(blocked, baseline=baseline)[0] == ContactDecision.BLOCKED


def test_leakage_audit_lists_forbidden_fields_and_localization_assumption() -> None:
    audit = privileged_information_leakage_audit()
    assert "object_kind" in audit["evaluation_ground_truth_fields"]
    assert "simulator_contact_flag" in audit["classifier_forbidden_inputs"]
    assert audit["localization_assumption_required"] is True


def test_candidate_label_is_identical_for_curtain_and_rigid_scenarios() -> None:
    curtain_map = make_map()
    rigid_map = make_map()
    np.testing.assert_array_equal(curtain_map.labels, rigid_map.labels)
    assert curtain_map.label_at(0.70, 0.0) == OracleCellLabel.INTERACTION_CANDIDATE
    assert rigid_map.label_at(0.70, 0.0) == OracleCellLabel.INTERACTION_CANDIDATE


def test_candidate_two_is_not_direct_free_and_requires_one_probe() -> None:
    grid = make_map()
    planner = KnownMapPlanner(precontact_offset_m=0.15)
    memory = InteractionMemory()
    first = planner.plan(
        grid,
        current_xy=(0.0, 0.0),
        goal_xy=(1.5, 0.0),
        candidate_xy=(0.70, 0.0),
        interaction_memory=memory,
    )
    assert first.status == KnownMapPlanStatus.APPROACH_CANDIDATE
    assert first.probe_allowed
    assert not grid.segment_is_free((0.0, 0.0), (1.5, 0.0))
    assert memory.reserve_probe((0.70, 0.0))
    memory.record_response((0.70, 0.0), InteractionStatus.BLOCKED)
    second = planner.plan(
        grid,
        current_xy=(0.0, 0.0),
        goal_xy=(1.5, 0.0),
        candidate_xy=(0.70, 0.0),
        interaction_memory=memory,
        probe_decision=ContactDecision.BLOCKED.value,
    )
    assert second.status == KnownMapPlanStatus.DETOUR
    assert second.path_xy
    assert all(grid.label_at(*point) == OracleCellLabel.FREE for point in second.path_xy)
    assert memory.status_at((0.72, 0.02)) == InteractionStatus.DEFERRED_THIS_EPISODE
    assert not memory.can_probe((0.72, 0.02))
