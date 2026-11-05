"""Round26 evidence semantics for learned high-level shortcut decisions.

This module keeps three facts separate:

* a deployment observation can support a physical ``BLOCKED`` decision;
* a policy can conservatively avoid a responsive candidate without proving it
  is blocked;
* a temporary planning exclusion is not the same thing as changing the
  episode's navigation map from 2 to 1.

No simulator truth is accepted by the public helpers.  Evaluation code may
attach the resulting ledger to a separate ground-truth report afterwards.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Iterable

from .oracle_grid_map import OracleCellLabel, OracleGridMap


class Round26EvidenceStatus(str, Enum):
    CONFIRMED_BLOCKED = "CONFIRMED_BLOCKED"
    UNVERIFIED_AVOIDED = "UNVERIFIED_AVOIDED"
    RESPONSIVE_CONTINUE = "RESPONSIVE_CONTINUE"
    UNCERTAIN_AVOIDED = "UNCERTAIN_AVOIDED"


class Round26MapUpdateReason(str, Enum):
    PHYSICAL_BLOCKED = "PHYSICAL_BLOCKED"
    SAFETY_ABORT_WITH_BLOCKING_EVIDENCE = "SAFETY_ABORT_WITH_BLOCKING_EVIDENCE"
    POLICY_AVOIDED_NO_PROOF = "POLICY_AVOIDED_NO_PROOF"
    RESPONSIVE_CLEARANCE = "RESPONSIVE_CLEARANCE"


@dataclass(frozen=True)
class Round26EvidenceRecord:
    status: Round26EvidenceStatus
    reason: str
    contact_decision: str
    policy_action: str
    progress_confirmed_blocked: bool = False
    safety_abort: bool = False
    map_update_allowed: bool = False
    map_update_reason: Round26MapUpdateReason | None = None

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["status"] = self.status.value
        if self.map_update_reason is not None:
            result["map_update_reason"] = self.map_update_reason.value
        return result


@dataclass
class Round26PlanningLedger:
    """Episode-local distinction between map truth and temporary avoidance."""

    navigation_map: OracleGridMap
    candidate_x_limits_m: tuple[float, float]
    candidate_y_limits_m: tuple[float, float]
    records: list[Round26EvidenceRecord] = field(default_factory=list)
    temporary_exclusion_count: int = 0

    def record(self, record: Round26EvidenceRecord) -> None:
        self.records.append(record)
        if record.status in (
            Round26EvidenceStatus.UNVERIFIED_AVOIDED,
            Round26EvidenceStatus.UNCERTAIN_AVOIDED,
        ):
            self.temporary_exclusion_count += 1

    def apply_confirmed_blocked(self, *, reason: Round26MapUpdateReason) -> int:
        if reason not in (
            Round26MapUpdateReason.PHYSICAL_BLOCKED,
            Round26MapUpdateReason.SAFETY_ABORT_WITH_BLOCKING_EVIDENCE,
        ):
            raise ValueError("only physical blocking evidence may update 2 to 1")
        return self.navigation_map.transition_rectangle(
            self.candidate_x_limits_m,
            self.candidate_y_limits_m,
            expected=OracleCellLabel.INTERACTION_CANDIDATE,
            replacement=OracleCellLabel.FIXED_OBSTACLE,
        )

    def planning_map(self) -> OracleGridMap:
        """Return a temporary map for a conservative avoided route.

        The returned clone is intentionally not the navigation map.  This
        lets a policy avoid a candidate for the current episode without
        claiming that the candidate has been physically proven blocked.
        """
        result = self.navigation_map.clone()
        if self.records and self.records[-1].status in (
            Round26EvidenceStatus.UNVERIFIED_AVOIDED,
            Round26EvidenceStatus.UNCERTAIN_AVOIDED,
        ):
            result.transition_rectangle(
                self.candidate_x_limits_m,
                self.candidate_y_limits_m,
                expected=OracleCellLabel.INTERACTION_CANDIDATE,
                replacement=OracleCellLabel.FIXED_OBSTACLE,
            )
        return result

    def snapshot(self) -> dict[str, object]:
        return {
            "temporary_exclusion_count": self.temporary_exclusion_count,
            "records": [record.to_dict() for record in self.records],
            "navigation_map": self.navigation_map.to_dict(),
            "planning_map_last": self.planning_map().to_dict(),
        }


def make_evidence_record(
    *,
    contact_decision: str,
    policy_action: str,
    reason: str,
    progress_confirmed_blocked: bool = False,
    safety_abort: bool = False,
) -> Round26EvidenceRecord:
    """Convert deployment evidence and action outcome into a ledger record."""

    decision = str(contact_decision).upper()
    action = str(policy_action).upper()
    if decision == "BLOCKED" and progress_confirmed_blocked:
        return Round26EvidenceRecord(
            Round26EvidenceStatus.CONFIRMED_BLOCKED,
            reason,
            decision,
            action,
            progress_confirmed_blocked=True,
            safety_abort=safety_abort,
            map_update_allowed=True,
            map_update_reason=Round26MapUpdateReason.PHYSICAL_BLOCKED,
        )
    if decision == "RESPONSIVE" and action in {"CAUTIOUS_ADVANCE", "CONTINUE"}:
        return Round26EvidenceRecord(
            Round26EvidenceStatus.RESPONSIVE_CONTINUE,
            reason,
            decision,
            action,
        )
    if decision == "RESPONSIVE":
        return Round26EvidenceRecord(
            Round26EvidenceStatus.UNVERIFIED_AVOIDED,
            reason,
            decision,
            action,
            safety_abort=safety_abort,
            map_update_reason=Round26MapUpdateReason.POLICY_AVOIDED_NO_PROOF,
        )
    return Round26EvidenceRecord(
        Round26EvidenceStatus.UNCERTAIN_AVOIDED,
        reason,
        decision,
        action,
        safety_abort=safety_abort,
        map_update_reason=Round26MapUpdateReason.POLICY_AVOIDED_NO_PROOF,
    )


def can_update_candidate_to_blocked(record: Round26EvidenceRecord) -> bool:
    """Return whether an episode may make the persistent 2->1 transition."""

    return bool(
        record.map_update_allowed
        and record.status == Round26EvidenceStatus.CONFIRMED_BLOCKED
        and record.map_update_reason
        in (
            Round26MapUpdateReason.PHYSICAL_BLOCKED,
            Round26MapUpdateReason.SAFETY_ABORT_WITH_BLOCKING_EVIDENCE,
        )
    )


def summarize_labels(grid: OracleGridMap) -> dict[str, int]:
    return {str(int(label)): count for label, count in grid.label_counts().items()}

