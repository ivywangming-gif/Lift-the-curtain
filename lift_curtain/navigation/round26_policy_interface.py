"""Round26 multi-stage high-level policy adapter.

The adapter reuses the Round25B deployment contract and adds only a stage
argument plus deployment-only progress evidence after a bounded advance. It
does not expose simulator labels, cloth nodes, contact flags or passability.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from lift_curtain.control.deployment_observation import ContactDecision

from .round25b_policy_interface import (
    DEPLOYMENT_FIELDS_USED,
    HighLevelAction,
    HighLevelPolicyAdapter,
    PolicyContext,
    PolicyDecision,
    SafetySupervisor,
    valid_actions_for_context,
)


class Round26PolicyAdapter:
    """R1/R2/A0/A1/B0/B1/B2 under one dynamic option contract."""

    SUPPORTED = ("r1", "r2", "a0", "a1", "b0", "b1", "b2")

    def __init__(
        self,
        name: str,
        *,
        repo: Path,
        checkpoint: Path | None = None,
    ) -> None:
        normalized = str(name).lower()
        if normalized not in self.SUPPORTED:
            raise ValueError(f"policy must be one of {', '.join(self.SUPPORTED)}")
        self.name = normalized
        self.repo = repo.resolve()
        self.checkpoint = checkpoint.resolve() if checkpoint is not None else None
        if self.checkpoint is None and normalized == "a0":
            self.checkpoint = self.repo / "outputs" / "round25_4070" / "method_a" / "a2_gru_seed0.pt"
        if self.checkpoint is None and normalized == "a1":
            # A1 intentionally reuses the frozen A0 predictor and changes only
            # the option-level evidence update, which keeps the ablation fair.
            self.checkpoint = self.repo / "outputs" / "round25_4070" / "method_a" / "a2_gru_seed0.pt"
        if self.checkpoint is None and normalized == "b0":
            self.checkpoint = self.repo / "outputs" / "round25_4070" / "method_b" / "b0_standard_ppo_seed0.pt"
        if self.checkpoint is None and normalized == "b1":
            self.checkpoint = self.repo / "outputs" / "round26_4070" / "models" / "b1_bc_seed0.pt"
        if self.checkpoint is None and normalized == "b2":
            self.checkpoint = self.repo / "outputs" / "round26_4070" / "models" / "b2_bc_ppo_seed0.pt"
        if normalized.startswith("a"):
            base = "a"
        elif normalized.startswith("b"):
            base = "b"
        else:
            base = normalized
        self._base = HighLevelPolicyAdapter(
            base,
            repo=self.repo,
            checkpoint=self.checkpoint,
        )
        self.checkpoint = self._base.checkpoint
        self.model_status = self._base.model_status

    @property
    def policy_label(self) -> str:
        return self.name.upper()

    @staticmethod
    def _features(context: PolicyContext, advance_metrics: Mapping[str, Any] | None) -> dict[str, Any]:
        merged = dict(context.contact_features or {})
        if advance_metrics:
            merged.update({f"advance_{key}": value for key, value in advance_metrics.items()})
        return merged

    def _a1_decision(
        self,
        context: PolicyContext,
        *,
        stage: str,
        advance_metrics: Mapping[str, Any] | None,
    ) -> PolicyDecision:
        base = self._base.decide(context)
        valid = valid_actions_for_context(context)
        features = self._features(context, advance_metrics)
        if context.progress_monitor_stop_requested or not context.stable or not context.finite:
            raw = HighLevelAction.RETRACT.value
            reason = "A1_SAFETY_OR_PROGRESS_STOP"
        elif context.contact_decision != ContactDecision.RESPONSIVE:
            raw = HighLevelAction.RETRACT.value
            reason = "A1_NONRESPONSIVE_REQUIRES_RETRACT"
        elif stage == "post_probe":
            probability = float((base.belief or {}).get("p_clearance", 0.0))
            route_advantage = float(context.zero_only_route_cost_m - context.candidate_route_cost_m)
            # This is a bounded information-gathering segment, not a
            # passability assertion. The runner requests another decision
            # after the segment and can force a retract.
            if probability >= 0.62:
                raw = HighLevelAction.CAUTIOUS_ADVANCE.value
                reason = "A1_PROBABILITY_SUPPORTS_BOUNDED_ADVANCE"
            elif probability >= 0.35 and route_advantage >= 0.40:
                raw = HighLevelAction.CAUTIOUS_ADVANCE.value
                reason = "A1_VALUE_OF_INFORMATION_BOUNDED_ADVANCE"
            else:
                raw = HighLevelAction.DETOUR.value
                reason = "A1_EVIDENCE_TOO_WEAK_FOR_BOUNDED_ADVANCE"
        elif stage == "post_advance":
            progress = float(features.get("advance_base_progress_m", 0.0))
            commanded = float(features.get("advance_commanded_progress_m", 0.0))
            stagnant = int(features.get("advance_stagnant_windows", 0))
            tracking = float(features.get("advance_upper_tracking_rms", 999.0))
            if stagnant > 0 or (commanded >= 0.04 and progress < 0.012):
                raw = HighLevelAction.RETRACT.value
                reason = "A1_PROGRESS_EVIDENCE_SUPPORTS_RETRACT"
            elif progress >= 0.015 and tracking <= 0.30:
                raw = HighLevelAction.CAUTIOUS_ADVANCE.value
                reason = "A1_NEW_PROGRESS_EVIDENCE_SUPPORTS_CONTINUE"
            else:
                raw = HighLevelAction.RETRACT.value
                reason = "A1_POST_ADVANCE_EVIDENCE_UNCERTAIN"
        else:
            raw = base.raw_action
            reason = "A1_REUSED_BASE_DECISION"
        filtered, filter_reason = SafetySupervisor.filter_action(raw, context, valid)
        return replace(
            base,
            policy="A1",
            version="ROUND26_POLICY_INTERFACE_V1",
            raw_action=raw,
            filtered_action=filtered,
            valid_actions=tuple(valid),
            reason=f"{reason};{filter_reason}",
            raw_scores={**dict(base.raw_scores), "round26_stage": stage},
            deployment_fields_used=DEPLOYMENT_FIELDS_USED,
        )

    def decide(
        self,
        context: PolicyContext,
        *,
        stage: str = "post_probe",
        advance_metrics: Mapping[str, Any] | None = None,
    ) -> PolicyDecision:
        stage = str(stage)
        if self.name == "a1":
            return self._a1_decision(context, stage=stage, advance_metrics=advance_metrics)
        base = self._base.decide(context)
        return replace(
            base,
            policy=self.policy_label,
            version="ROUND26_POLICY_INTERFACE_V1",
            reason=f"ROUND26_STAGE={stage};{base.reason}",
            deployment_fields_used=DEPLOYMENT_FIELDS_USED,
        )


def round26_privileged_information_leakage_audit() -> dict[str, object]:
    return {
        "deployment_fields_used": list(DEPLOYMENT_FIELDS_USED),
        "forbidden_inputs": [
            "object identity",
            "simulator contact flag/force",
            "curtain nodes",
            "true passability",
            "episode outcome oracle",
        ],
        "new_round26_input": "deployment-only bounded-advance progress/tracking/stall history",
        "low_level": "frozen FALCON and existing SE(2) controller",
    }


__all__ = [
    "Round26PolicyAdapter",
    "round26_privileged_information_leakage_audit",
]

