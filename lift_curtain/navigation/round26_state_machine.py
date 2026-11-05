"""Round26 state-machine extension with explicit avoided-vs-blocked evidence."""

from __future__ import annotations

from .round22_state_machine import (
    CandidateRegion,
    ProbeDecision,
    Round22NavigationWorkflow,
    Round22State,
)
from .round26_semantics import (
    Round26EvidenceRecord,
    Round26EvidenceStatus,
    Round26MapUpdateReason,
    can_update_candidate_to_blocked,
)


class Round26NavigationWorkflow(Round22NavigationWorkflow):
    """Reuse the proven transitions while preserving evidence semantics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evidence_records: list[Round26EvidenceRecord] = []
        self.last_evidence_status: Round26EvidenceStatus | None = None
        self.persistent_map_update_reason: Round26MapUpdateReason | None = None

    def record_evidence(self, record: Round26EvidenceRecord) -> None:
        self.evidence_records.append(record)
        self.last_evidence_status = record.status

    def mark_unverified_avoided(self, record: Round26EvidenceRecord) -> None:
        if record.status not in (
            Round26EvidenceStatus.UNVERIFIED_AVOIDED,
            Round26EvidenceStatus.UNCERTAIN_AVOIDED,
        ):
            raise ValueError("mark_unverified_avoided requires an avoidance record")
        if self.state == Round22State.CAUTIOUS_TRAVERSE:
            self._transition(Round22State.RELEASE_AND_RECOVER, "POLICY_AVOIDED_WITHOUT_BLOCKING_PROOF")
        elif self.state != Round22State.RELEASE_AND_RECOVER:
            raise RuntimeError("unverified avoidance requires an interaction state")
        self.record_evidence(record)

    def mark_confirmed_blocked(self, record: Round26EvidenceRecord) -> int:
        if not can_update_candidate_to_blocked(record):
            raise ValueError("record does not contain sufficient blocking evidence")
        if self.state == Round22State.CAUTIOUS_TRAVERSE:
            self._transition(Round22State.RELEASE_AND_RECOVER, "DEPLOYMENT_BLOCKING_EVIDENCE")
        elif self.state != Round22State.RELEASE_AND_RECOVER:
            raise RuntimeError("confirmed blocking requires an interaction state")
        self.record_evidence(record)
        changed = self.mark_candidate_blocked()
        self.persistent_map_update_reason = record.map_update_reason
        return changed

    def complete_release_and_recover(self, *, stable: bool, finite: bool, blocked: bool = False) -> None:
        # The parent method automatically updates 2->1 for BLOCKED/UNCERTAIN.
        # Round26 only permits that transition after explicit evidence has been
        # recorded, so handle the unverified branch separately.
        if self.last_evidence_status in (
            Round26EvidenceStatus.UNVERIFIED_AVOIDED,
            Round26EvidenceStatus.UNCERTAIN_AVOIDED,
        ):
            if self.state != Round22State.RELEASE_AND_RECOVER:
                raise RuntimeError("unverified recovery must start in RELEASE_AND_RECOVER")
            if blocked:
                raise RuntimeError("unverified avoidance cannot be marked blocked")
            if not stable or not finite:
                raise RuntimeError("release/recover requires finite stable robot state")
            self._transition(Round22State.NAVIGATE_TO_GOAL, "RELEASED_AFTER_UNVERIFIED_AVOIDANCE")
            return
        super().complete_release_and_recover(stable=stable, finite=finite, blocked=blocked)

    def snapshot(self) -> dict[str, object]:
        result = super().snapshot()
        result.update(
            {
                "round26_evidence_records": [item.to_dict() for item in self.evidence_records],
                "last_evidence_status": None if self.last_evidence_status is None else self.last_evidence_status.value,
                "persistent_map_update_reason": None
                if self.persistent_map_update_reason is None
                else self.persistent_map_update_reason.value,
            }
        )
        return result


__all__ = [
    "CandidateRegion",
    "ProbeDecision",
    "Round22State",
    "Round26NavigationWorkflow",
]

