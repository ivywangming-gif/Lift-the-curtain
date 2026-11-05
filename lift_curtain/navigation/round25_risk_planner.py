"""Deployment-only Round25 risk-aware high-level planner.

The planner consumes a belief produced from deployable proprioception/history
and public map geometry.  It never accepts object identity, simulator contact,
cloth nodes, or a true passability flag.  It is intentionally option-level;
low-level FALCON/SE(2) control remains unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class Round25Action(str, Enum):
    AVOID = "AVOID"
    PROBE = "PROBE"
    CAUTIOUS_ADVANCE = "CAUTIOUS_ADVANCE"
    RETRACT = "RETRACT"
    DETOUR = "DETOUR"


@dataclass(frozen=True)
class TraversabilityBelief:
    p_clearance: float
    p_stall: float
    p_safety_abort: float
    uncertainty: float

    def __post_init__(self) -> None:
        for name in ("p_clearance", "p_stall", "p_safety_abort", "uncertainty"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class RiskPlannerConfig:
    min_clearance_probability: float = 0.62
    max_stall_probability: float = 0.38
    max_safety_abort_probability: float = 0.20
    max_uncertainty: float = 0.35
    probe_cost_m: float = 0.20
    detour_cost_m: float = 0.80
    value_of_information_margin_m: float = 0.10


@dataclass(frozen=True)
class PlannerInput:
    belief: TraversabilityBelief
    candidate_route_cost_m: float
    zero_only_route_cost_m: float
    probe_budget_remaining: int
    already_probed: bool
    candidate_map_label: int
    stable: bool
    phase: str


def choose_action(data: PlannerInput, config: RiskPlannerConfig | None = None) -> Round25Action:
    """Choose one safe high-level option using no simulator truth."""
    cfg = config or RiskPlannerConfig()
    if data.candidate_map_label != 2:
        return Round25Action.AVOID
    if not data.stable or data.phase in {"fall", "unstable", "safety_abort"}:
        return Round25Action.RETRACT
    if data.already_probed or data.probe_budget_remaining <= 0:
        return Round25Action.DETOUR
    belief = data.belief
    unsafe = (
        belief.p_safety_abort > cfg.max_safety_abort_probability
        or belief.p_stall > cfg.max_stall_probability
        or belief.uncertainty > cfg.max_uncertainty
    )
    if unsafe:
        return Round25Action.DETOUR
    route_advantage = data.zero_only_route_cost_m - data.candidate_route_cost_m
    if route_advantage < cfg.probe_cost_m:
        return Round25Action.DETOUR
    if belief.p_clearance >= cfg.min_clearance_probability:
        return Round25Action.CAUTIOUS_ADVANCE
    # A probe is worthwhile only when the route advantage can pay for it and
    # uncertainty is bounded. It is not a command to push indefinitely.
    if route_advantage >= cfg.value_of_information_margin_m:
        return Round25Action.PROBE
    return Round25Action.DETOUR


def privileged_information_leakage_audit() -> dict[str, object]:
    return {
        "inputs": [
            "proprioceptive belief from q/qdot/q_cmd/motor_effort/previous_action/FK history",
            "public Oracle 0/1/2 map label and route costs",
            "probe budget, stability and current phase",
        ],
        "forbidden": [
            "object identity", "simulator contact flag/force", "curtain node state",
            "true passability", "episode outcome oracle", "hidden material/anchor label",
        ],
        "low_level_controller": "frozen FALCON plus existing SE(2) controller",
    }
