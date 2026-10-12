"""Deployment-safe Round 12 interaction/navigation state machine.

The state machine receives only deployment observations and map geometry.  It
does not accept object identity, simulator contact truth, cloth nodes, or true
passability as inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from .oracle_grid_map import (
    OracleCellLabel,
    OracleGridMap,
    OracleRouteComparison,
    compare_candidate_and_zero_routes,
)


class Round12State(str, Enum):
    APPROACH_2 = "APPROACH_2"
    STAND_AND_PROBE = "STAND_AND_PROBE"
    CAUTIOUS_TRAVERSE = "CAUTIOUS_TRAVERSE"
    CLEAR_CANDIDATE_REGION = "CLEAR_CANDIDATE_REGION"
    RELEASE_AND_RECOVER = "RELEASE_AND_RECOVER"
    NAVIGATE_TO_GOAL = "NAVIGATE_TO_GOAL"
    GOAL_REACHED = "GOAL_REACHED"


class ProbeDecision(str, Enum):
    RESPONSIVE = "RESPONSIVE"
    BLOCKED = "BLOCKED"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class CandidateRegion:
    x_limits_m: tuple[float, float]
    y_limits_m: tuple[float, float]
    clear_margin_m: float = 0.04

    def __post_init__(self) -> None:
        if self.x_limits_m[0] > self.x_limits_m[1] or self.y_limits_m[0] > self.y_limits_m[1]:
            raise ValueError("candidate region limits must be ordered")
        if self.clear_margin_m < 0.0:
            raise ValueError("clear margin must be non-negative")


@dataclass(frozen=True)
class StateTransition:
    source: Round12State
    target: Round12State
    reason: str


@dataclass
class Round12NavigationWorkflow:
    """Small auditable contract shared by curtain and rigid episodes."""

    initial_oracle_map: OracleGridMap
    candidate_region: CandidateRegion
    navigation_map: OracleGridMap = field(init=False)
    state: Round12State = field(default=Round12State.APPROACH_2, init=False)
    probe_decision: ProbeDecision | None = field(default=None, init=False)
    probe_count: int = field(default=0, init=False)
    map_update_count: int = field(default=0, init=False)
    transitions: list[StateTransition] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.navigation_map = self.initial_oracle_map.clone()

    def compare_routes(
        self,
        *,
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        candidate_penalty_m: float = 0.30,
    ) -> OracleRouteComparison:
        return compare_candidate_and_zero_routes(
            self.navigation_map,
            start_xy=start_xy,
            goal_xy=goal_xy,
            candidate_penalty_m=candidate_penalty_m,
        )

    def _transition(self, target: Round12State, reason: str) -> None:
        allowed = {
            Round12State.APPROACH_2: {Round12State.STAND_AND_PROBE},
            Round12State.STAND_AND_PROBE: {
                Round12State.CAUTIOUS_TRAVERSE,
                Round12State.RELEASE_AND_RECOVER,
            },
            Round12State.CAUTIOUS_TRAVERSE: {
                Round12State.CLEAR_CANDIDATE_REGION,
                Round12State.RELEASE_AND_RECOVER,
            },
            Round12State.CLEAR_CANDIDATE_REGION: {Round12State.RELEASE_AND_RECOVER},
            Round12State.RELEASE_AND_RECOVER: {Round12State.NAVIGATE_TO_GOAL},
            Round12State.NAVIGATE_TO_GOAL: {Round12State.GOAL_REACHED},
            Round12State.GOAL_REACHED: set(),
        }
        if target not in allowed[self.state]:
            raise RuntimeError(f"illegal Round 12 transition {self.state.value}->{target.value}")
        self.transitions.append(StateTransition(self.state, target, reason))
        self.state = target

    def reach_precontact(self) -> None:
        if self.state != Round12State.APPROACH_2:
            raise RuntimeError("precontact can only be accepted from APPROACH_2")
        self._transition(Round12State.STAND_AND_PROBE, "PRECONTACT_STABLE_AND_IN_FREE_REGION")

    def record_probe(self, decision: ProbeDecision) -> None:
        if self.state != Round12State.STAND_AND_PROBE:
            raise RuntimeError("probe decision requires STAND_AND_PROBE")
        if self.probe_count >= 1:
            raise RuntimeError("Round 12 episode probe budget exhausted")
        self.probe_count += 1
        self.probe_decision = ProbeDecision(decision)
        if self.probe_decision == ProbeDecision.RESPONSIVE:
            self._transition(Round12State.CAUTIOUS_TRAVERSE, "DEPLOYMENT_PROPRIO_RESPONSIVE")
        else:
            self._transition(Round12State.RELEASE_AND_RECOVER, "DEPLOYMENT_PROPRIO_STOP_AND_RETRACT")

    def verify_candidate_clearance(
        self,
        *,
        body_back_extents_x_m: Iterable[float],
        stable: bool,
        finite: bool,
    ) -> bool:
        """Use robot-owned envelope values to authorize candidate clearance."""
        if self.state != Round12State.CAUTIOUS_TRAVERSE:
            raise RuntimeError("candidate clearance requires CAUTIOUS_TRAVERSE")
        extents = [float(value) for value in body_back_extents_x_m]
        clear_x = self.candidate_region.x_limits_m[1] + self.candidate_region.clear_margin_m
        cleared = bool(extents) and min(extents) >= clear_x and bool(stable) and bool(finite)
        if cleared:
            self._transition(Round12State.CLEAR_CANDIDATE_REGION, "ROBOT_ENVELOPE_CLEAR_OF_CANDIDATE")
        return cleared

    def mark_candidate_blocked(self) -> int:
        """Update only candidate label 2 cells to label 1 after a failed probe."""
        if self.state != Round12State.RELEASE_AND_RECOVER:
            raise RuntimeError("candidate map update requires RELEASE_AND_RECOVER")
        changed = self.navigation_map.transition_rectangle(
            self.candidate_region.x_limits_m,
            self.candidate_region.y_limits_m,
            expected=OracleCellLabel.INTERACTION_CANDIDATE,
            replacement=OracleCellLabel.FIXED_OBSTACLE,
        )
        self.map_update_count += changed
        return changed

    def complete_release_and_recover(self, *, stable: bool, finite: bool, blocked: bool = False) -> None:
        if self.state == Round12State.CLEAR_CANDIDATE_REGION:
            if blocked:
                raise RuntimeError("a cleared responsive candidate cannot be marked blocked without a new failed probe")
            self._transition(Round12State.RELEASE_AND_RECOVER, "CANDIDATE_CLEARANCE_CONFIRMED")
        elif self.state != Round12State.RELEASE_AND_RECOVER:
            raise RuntimeError("release/recover requires RELEASE_AND_RECOVER or CLEAR_CANDIDATE_REGION")
        if not stable or not finite:
            raise RuntimeError("release/recover requires finite stable robot state")
        if self.state == Round12State.RELEASE_AND_RECOVER and self.probe_decision in (
            ProbeDecision.BLOCKED,
            ProbeDecision.UNCERTAIN,
        ):
            if self.map_update_count == 0:
                self.mark_candidate_blocked()
        self._transition(Round12State.NAVIGATE_TO_GOAL, "RELEASED_AND_REFERENCE_REANCHORED")

    def mark_responsive_traverse_failed(self) -> None:
        if self.state not in (Round12State.CAUTIOUS_TRAVERSE, Round12State.CLEAR_CANDIDATE_REGION):
            raise RuntimeError("responsive traverse failure is only valid during interaction traversal")
        self._transition(Round12State.RELEASE_AND_RECOVER, "TRAVERSE_SAFETY_OR_PROGRESS_FAILURE")
        self.probe_decision = ProbeDecision.UNCERTAIN
        self.mark_candidate_blocked()

    def reach_goal(self, *, stable: bool, finite: bool) -> None:
        if self.state != Round12State.NAVIGATE_TO_GOAL:
            raise RuntimeError("goal gate requires NAVIGATE_TO_GOAL")
        if not stable or not finite:
            raise RuntimeError("goal gate requires finite stable robot state")
        self._transition(Round12State.GOAL_REACHED, "GOAL_POSITION_YAW_SPEED_GATE")

    def repeated_probe_allowed(self) -> bool:
        return self.probe_count < 1

    def snapshot(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "probe_decision": None if self.probe_decision is None else self.probe_decision.value,
            "probe_count": self.probe_count,
            "repeated_probe_allowed": self.repeated_probe_allowed(),
            "map_update_count": self.map_update_count,
            "initial_map": self.initial_oracle_map.to_dict(),
            "navigation_map": self.navigation_map.to_dict(),
            "transitions": [
                {"source": item.source.value, "target": item.target.value, "reason": item.reason}
                for item in self.transitions
            ],
        }
