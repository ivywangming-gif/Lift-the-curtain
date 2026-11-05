"""Round 27 deployment-only protocol adapter.

The new ``d1`` option deliberately removes the Round 25 learned belief from
the bounded-forward-validation decision. It reuses the frozen Round 26 action
contract and safety supervisor, but makes the only changed variable the
additional deployment history after a short advance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from lift_curtain.control.deployment_observation import ContactDecision
from lift_curtain.navigation.round25b_policy_interface import (
    DEPLOYMENT_FIELDS_USED,
    HighLevelAction,
    PolicyContext,
    PolicyDecision,
    SafetySupervisor,
    valid_actions_for_context,
)
from lift_curtain.navigation.round26_policy_interface import Round26PolicyAdapter


class Round27PolicyAdapter(Round26PolicyAdapter):
    """Round 26 policies plus a pure deployment-history D1 rule."""

    SUPPORTED = tuple(Round26PolicyAdapter.SUPPORTED) + ("d1",)

    def __init__(
        self,
        name: str,
        *,
        repo: Path,
        checkpoint: Path | None = None,
    ) -> None:
        normalized = str(name).lower()
        if normalized == "d1":
            # Initialize inherited bookkeeping with a non-learning rule.
            super().__init__("r1", repo=repo, checkpoint=None)
            self.name = "d1"
            self.checkpoint = None
            self.model_status = "round27_deployment_only_rule"
        else:
            super().__init__(normalized, repo=repo, checkpoint=checkpoint)

    def _d1_decision(
        self,
        context: PolicyContext,
        *,
        stage: str,
        advance_metrics: Mapping[str, Any] | None,
    ) -> PolicyDecision:
        valid = valid_actions_for_context(context)
        metrics = dict(advance_metrics or {})
        if context.progress_monitor_stop_requested or not context.stable or not context.finite:
            raw = HighLevelAction.RETRACT.value
            reason = "D1_SAFETY_OR_PROGRESS_STOP"
        elif context.contact_decision != ContactDecision.RESPONSIVE:
            raw = HighLevelAction.RETRACT.value
            reason = "D1_NONRESPONSIVE_REQUIRES_RETRACT"
        elif stage == "post_probe":
            raw = HighLevelAction.CAUTIOUS_ADVANCE.value
            reason = "D1_FIXED_BOUNDED_VALIDATION"
        elif stage == "post_advance":
            progress = float(metrics.get("advance_base_progress_m", 0.0))
            commanded = float(metrics.get("advance_commanded_progress_m", 0.0))
            stagnant = int(metrics.get("advance_stagnant_windows", 0))
            tracking = float(metrics.get("advance_upper_tracking_rms", 999.0))
            if stagnant > 0 or (commanded >= 0.04 and progress < 0.012):
                raw = HighLevelAction.RETRACT.value
                reason = "D1_PROGRESS_EVIDENCE_SUPPORTS_RETRACT"
            elif progress >= 0.015 and tracking <= 0.30:
                raw = HighLevelAction.CAUTIOUS_ADVANCE.value
                reason = "D1_PROGRESS_EVIDENCE_SUPPORTS_CONTINUE"
            else:
                raw = HighLevelAction.RETRACT.value
                reason = "D1_PROGRESS_EVIDENCE_UNCERTAIN"
        else:
            raw = HighLevelAction.RETRACT.value
            reason = "D1_ILLEGAL_STAGE_REQUIRES_RETRACT"
        filtered, filter_reason = SafetySupervisor.filter_action(raw, context, valid)
        return PolicyDecision(
            policy="D1",
            version="ROUND27_D1_RULE_V1",
            raw_action=raw,
            filtered_action=filtered,
            valid_actions=tuple(valid),
            reason=f"{reason};{filter_reason}",
            raw_scores={
                "rule_progress_m": float(metrics.get("advance_base_progress_m", 0.0)),
                "rule_commanded_progress_m": float(metrics.get("advance_commanded_progress_m", 0.0)),
                "round27_stage": stage,
            },
            belief=None,
            deployment_fields_used=DEPLOYMENT_FIELDS_USED,
            model_checkpoint=None,
            model_status=self.model_status,
        )

    def decide(
        self,
        context: PolicyContext,
        *,
        stage: str = "post_probe",
        advance_metrics: Mapping[str, Any] | None = None,
    ) -> PolicyDecision:
        if self.name == "d1":
            return self._d1_decision(context, stage=stage, advance_metrics=advance_metrics)
        return super().decide(context, stage=stage, advance_metrics=advance_metrics)


def round27_privileged_information_leakage_audit() -> dict[str, object]:
    return {
        "deployment_fields_used": list(DEPLOYMENT_FIELDS_USED),
        "forbidden_inputs": [
            "object identity",
            "simulator contact flag/force",
            "curtain nodes",
            "true passability",
            "episode outcome oracle",
        ],
        "d1_policy": "deployment-only bounded progress, stall and tracking rule; no learned checkpoint",
        "low_level": "frozen FALCON and existing SE(2) controller",
    }


__all__ = ["Round27PolicyAdapter", "round27_privileged_information_leakage_audit"]
