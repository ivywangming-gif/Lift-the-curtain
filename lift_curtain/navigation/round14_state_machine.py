"""Round 14 state-machine contract with an independent PointGoal gate."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from lift_curtain.control.point_goal import GoalEvaluation
from .oracle_grid_map import OracleGridMap
from .round12_state_machine import CandidateRegion, ProbeDecision, Round12State


class Round14State(str, Enum):
    APPROACH_2 = Round12State.APPROACH_2.value
    STAND_AND_PROBE = Round12State.STAND_AND_PROBE.value
    CAUTIOUS_TRAVERSE = Round12State.CAUTIOUS_TRAVERSE.value
    CLEAR_CANDIDATE_REGION = Round12State.CLEAR_CANDIDATE_REGION.value
    RELEASE_AND_RECOVER = Round12State.RELEASE_AND_RECOVER.value
    NAVIGATE_TO_GOAL = Round12State.NAVIGATE_TO_GOAL.value
    POINTGOAL_REACHED = "POINTGOAL_REACHED"


@dataclass(frozen=True)
class Round14Transition:
    source: Round14State
    target: Round14State
    reason: str


@dataclass
class Round14NavigationWorkflow:
    initial_oracle_map: OracleGridMap
    candidate_region: CandidateRegion
    navigation_map: OracleGridMap = field(init=False)
    state: Round14State = field(default=Round14State.APPROACH_2, init=False)
    probe_decision: ProbeDecision | None = field(default=None, init=False)
    probe_count: int = field(default=0, init=False)
    map_update_count: int = field(default=0, init=False)
    candidate_clearance_verified: bool = field(default=False, init=False)
    transitions: list[Round14Transition] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.navigation_map = self.initial_oracle_map.clone()

    def _transition(self, target: Round14State, reason: str) -> None:
        allowed = {
            Round14State.APPROACH_2: {Round14State.STAND_AND_PROBE},
            Round14State.STAND_AND_PROBE: {Round14State.CAUTIOUS_TRAVERSE, Round14State.RELEASE_AND_RECOVER},
            Round14State.CAUTIOUS_TRAVERSE: {Round14State.CLEAR_CANDIDATE_REGION, Round14State.RELEASE_AND_RECOVER},
            Round14State.CLEAR_CANDIDATE_REGION: {Round14State.RELEASE_AND_RECOVER},
            Round14State.RELEASE_AND_RECOVER: {Round14State.NAVIGATE_TO_GOAL},
            Round14State.NAVIGATE_TO_GOAL: {Round14State.POINTGOAL_REACHED},
            Round14State.POINTGOAL_REACHED: set(),
        }
        if target not in allowed[self.state]:
            raise RuntimeError(f"illegal Round 14 transition {self.state.value}->{target.value}")
        self.transitions.append(Round14Transition(self.state, target, reason))
        self.state = target

    def reach_precontact(self) -> None:
        if self.state != Round14State.APPROACH_2:
            raise RuntimeError("precontact requires APPROACH_2")
        self._transition(Round14State.STAND_AND_PROBE, "PRECONTACT_STABLE")

    def record_probe(self, decision: ProbeDecision) -> None:
        if self.state != Round14State.STAND_AND_PROBE or self.probe_count >= 1:
            raise RuntimeError("probe budget or state does not permit probe")
        self.probe_count += 1
        self.probe_decision = ProbeDecision(decision)
        target = Round14State.CAUTIOUS_TRAVERSE if self.probe_decision == ProbeDecision.RESPONSIVE else Round14State.RELEASE_AND_RECOVER
        self._transition(target, "DEPLOYMENT_PROPRIO_DECISION")

    def verify_candidate_clearance(self, *, body_back_extents_x_m: Iterable[float], stable: bool, finite: bool) -> bool:
        if self.state != Round14State.CAUTIOUS_TRAVERSE:
            raise RuntimeError("candidate clearance requires CAUTIOUS_TRAVERSE")
        extents = [float(value) for value in body_back_extents_x_m]
        clear_x = self.candidate_region.x_limits_m[1] + self.candidate_region.clear_margin_m
        cleared = bool(extents) and min(extents) >= clear_x and bool(stable) and bool(finite)
        if cleared:
            self.candidate_clearance_verified = True
            self._transition(Round14State.CLEAR_CANDIDATE_REGION, "ROBOT_ENVELOPE_CLEAR")
        return cleared

    def mark_candidate_blocked(self) -> int:
        if self.state != Round14State.RELEASE_AND_RECOVER:
            raise RuntimeError("map update requires RELEASE_AND_RECOVER")
        changed = self.navigation_map.transition_rectangle(
            self.candidate_region.x_limits_m,
            self.candidate_region.y_limits_m,
            expected=2,
            replacement=1,
        )
        self.map_update_count += changed
        return changed

    def complete_release_and_recover(self, *, stable: bool, finite: bool) -> None:
        if self.state == Round14State.CLEAR_CANDIDATE_REGION:
            self._transition(Round14State.RELEASE_AND_RECOVER, "CANDIDATE_CLEARANCE_SEPARATE_FROM_GOAL")
        if self.state != Round14State.RELEASE_AND_RECOVER or not stable or not finite:
            raise RuntimeError("release/recover requires stable finite state")
        if self.probe_decision in (ProbeDecision.BLOCKED, ProbeDecision.UNCERTAIN) and self.map_update_count == 0:
            self.mark_candidate_blocked()
        self._transition(Round14State.NAVIGATE_TO_GOAL, "REFERENCE_REANCHORED")

    def reach_point_goal(self, evaluation: GoalEvaluation) -> None:
        if self.state != Round14State.NAVIGATE_TO_GOAL:
            raise RuntimeError("PointGoal gate requires NAVIGATE_TO_GOAL")
        if not evaluation.point_goal_pass:
            raise RuntimeError(f"PointGoal gate failed: {evaluation.reason}")
        self._transition(Round14State.POINTGOAL_REACHED, "POINTGOAL_XY_SPEED_SAFETY_GATE")

    def snapshot(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "probe_decision": None if self.probe_decision is None else self.probe_decision.value,
            "probe_count": self.probe_count,
            "map_update_count": self.map_update_count,
            "candidate_clearance_verified": self.candidate_clearance_verified,
            "initial_map": self.initial_oracle_map.to_dict(),
            "navigation_map": self.navigation_map.to_dict(),
            "transitions": [
                {"source": item.source.value, "target": item.target.value, "reason": item.reason}
                for item in self.transitions
            ],
        }
