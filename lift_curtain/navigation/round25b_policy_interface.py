"""Round25B high-level policy interface and deployment safety filter.

This module is intentionally independent from Isaac.  It accepts only the
deployment observation history already emitted by the runner and public map
state.  FALCON, SE(2) tracking, and all simulator safety monitors remain
outside this option-level interface.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from lift_curtain.control.deployment_observation import ContactDecision
from lift_curtain.navigation.round25_risk_planner import (
    PlannerInput,
    RiskPlannerConfig,
    TraversabilityBelief,
    choose_action,
)


class HighLevelAction(str, Enum):
    """Actions that can be handed to the existing physical state machine."""

    PROBE = "PROBE"
    CAUTIOUS_ADVANCE = "CAUTIOUS_ADVANCE"
    RETRACT = "RETRACT"
    DETOUR = "DETOUR"
    STOP = "STOP"


MODEL_ACTIONS = ("AVOID", "PROBE", "CAUTIOUS_ADVANCE", "RETRACT", "DETOUR")
VALID_POST_PROBE_ACTIONS = (
    HighLevelAction.CAUTIOUS_ADVANCE.value,
    HighLevelAction.RETRACT.value,
    HighLevelAction.DETOUR.value,
    HighLevelAction.STOP.value,
)

# This is the same causal feature contract used by Round25 dataset creation.
PHASE_GROUPS = (
    "settle", "approach", "precontact", "probe", "opening", "near",
    "traverse", "navigate", "retract", "stand", "detour", "goal", "other",
)
FEATURE_NAMES = (
    "base_dx", "base_dy", "base_speed", "base_vx", "base_vy",
    "left_hand_dx", "right_hand_dx", "left_hand_dy", "right_hand_dy",
    "left_hand_speed", "right_hand_speed", "left_hand_vx", "right_hand_vx",
    "q_tracking_rms", "qdot_rms", "effort_rms", "previous_action_rms",
    "target_dx", "target_dy", "map_label", "time_s",
    *(f"phase_{phase}" for phase in PHASE_GROUPS),
)

FORBIDDEN_DEPLOYMENT_TOKENS = (
    "object_kind", "simulator_contact", "true_hand_obstacle_gap",
    "curtain_node", "true_passability", "episode_outcome_oracle",
    "evaluation_", "scene_kind", "hidden_rigid",
)

DEPLOYMENT_ROW_KEYS = frozenset({
    "deployment_q_actual_json", "deployment_qdot_json", "deployment_q_cmd_json",
    "deployment_motor_effort_json", "deployment_previous_action_json",
    "deployment_hand_fk_position_w_json", "deployment_hand_fk_velocity_w_json",
    "deployment_left_hand_fk_position_w_json", "deployment_right_hand_fk_position_w_json",
    "deployment_left_hand_fk_velocity_w_json", "deployment_right_hand_fk_velocity_w_json",
    "deployment_base_position_w_json", "deployment_base_velocity_b_json",
    "deployment_probe_side", "map_label_at_base", "target_x", "target_y",
    "phase", "time_s",
})


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _vec(row: Mapping[str, Any], key: str, size: int) -> np.ndarray:
    raw = row.get(key, "[]")
    try:
        value = np.asarray(json.loads(raw) if isinstance(raw, str) else raw, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError, json.JSONDecodeError):
        value = np.zeros(0, dtype=np.float32)
    result = np.zeros(size, dtype=np.float32)
    result[: min(size, value.size)] = value[:size]
    return result


def _phase_group(value: Any) -> str:
    text = str(value or "").lower()
    for phase in PHASE_GROUPS[:-1]:
        if phase in text:
            return phase
    return "other"


def _resample(matrix: np.ndarray, length: int = 96) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError("deployment feature matrix has an unexpected shape")
    if matrix.shape[0] == 0:
        return np.zeros((length, matrix.shape[1]), dtype=np.float32)
    if matrix.shape[0] == length:
        return matrix.astype(np.float32)
    source = np.linspace(0.0, 1.0, matrix.shape[0])
    target = np.linspace(0.0, 1.0, length)
    return np.stack(
        [np.interp(target, source, matrix[:, index]) for index in range(matrix.shape[1])],
        axis=1,
    ).astype(np.float32)


def build_deployment_feature_sequence(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Build the Round25 96x34 sequence without reading evaluation fields."""
    if not rows:
        raise ValueError("policy history cannot be empty")
    first = rows[0]
    base0 = _vec(first, "deployment_base_position_w_json", 3)
    hand0 = {
        side: _vec(first, f"deployment_{side}_hand_fk_position_w_json", 3)
        for side in ("left", "right")
    }
    active_side = str(first.get("deployment_probe_side", "right")).lower()
    active0 = _vec(first, "deployment_hand_fk_position_w_json", 3)
    if np.linalg.norm(hand0["left"]) < 1.0e-8 and "left" in active_side:
        hand0["left"] = active0
    if np.linalg.norm(hand0["right"]) < 1.0e-8 and "right" in active_side:
        hand0["right"] = active0

    values: list[np.ndarray] = []
    for row in rows:
        base = _vec(row, "deployment_base_position_w_json", 3)
        base_vel = _vec(row, "deployment_base_velocity_b_json", 3)
        q = _vec(row, "deployment_q_actual_json", 29)
        qcmd = _vec(row, "deployment_q_cmd_json", 29)
        qdot = _vec(row, "deployment_qdot_json", 29)
        effort = _vec(row, "deployment_motor_effort_json", 29)
        action = _vec(row, "deployment_previous_action_json", 29)
        hands = {
            side: _vec(row, f"deployment_{side}_hand_fk_position_w_json", 3)
            for side in ("left", "right")
        }
        hand_vel = {
            side: _vec(row, f"deployment_{side}_hand_fk_velocity_w_json", 3)
            for side in ("left", "right")
        }
        active = _vec(row, "deployment_hand_fk_position_w_json", 3)
        active_vel = _vec(row, "deployment_hand_fk_velocity_w_json", 3)
        side = str(row.get("deployment_probe_side", "right")).lower()
        if np.linalg.norm(hands["left"]) < 1.0e-8 and "left" in side:
            hands["left"], hand_vel["left"] = active, active_vel
        if np.linalg.norm(hands["right"]) < 1.0e-8 and "right" in side:
            hands["right"], hand_vel["right"] = active, active_vel
        phase = _phase_group(row.get("phase", ""))
        feature_values = [
            float(base[0] - base0[0]), float(base[1] - base0[1]),
            float(np.linalg.norm(base_vel[:2])), float(base_vel[0]), float(base_vel[1]),
            float(hands["left"][0] - hand0["left"][0]),
            float(hands["right"][0] - hand0["right"][0]),
            float(hands["left"][1] - hand0["left"][1]),
            float(hands["right"][1] - hand0["right"][1]),
            float(np.linalg.norm(hand_vel["left"])),
            float(np.linalg.norm(hand_vel["right"])),
            float(hand_vel["left"][0]), float(hand_vel["right"][0]),
            float(np.sqrt(np.mean((qcmd - q) ** 2))),
            float(np.linalg.norm(qdot) / math.sqrt(max(1, qdot.size))),
            float(np.linalg.norm(effort) / math.sqrt(max(1, effort.size))),
            float(np.linalg.norm(action) / math.sqrt(max(1, action.size))),
            _finite(row.get("target_x")) - float(base[0]),
            _finite(row.get("target_y")) - float(base[1]),
            _finite(row.get("map_label_at_base")), _finite(row.get("time_s")),
        ]
        feature_values.extend(1.0 if phase == item else 0.0 for item in PHASE_GROUPS)
        values.append(np.asarray(feature_values, dtype=np.float32))
    return _resample(np.stack(values, axis=0), 96)


def _dataset_normalization(repo: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """Reproduce the frozen Round25 PPO preprocessing contract."""
    data_dir = repo / "outputs" / "round25_4070" / "dataset"
    archive = np.load(data_dir / "candidate_clearance.npz", allow_pickle=False)
    sequences = np.asarray(archive["pre_traverse"], dtype=np.float32)
    ids = [str(item) for item in archive["episode_ids"].tolist()]
    rows = list(csv.DictReader((data_dir / "episodes.csv").open(encoding="utf-8", newline="")))
    by_id = {row["episode_id"]: row for row in rows}
    train_mask = np.asarray([by_id[item].get("candidate_split") == "train" for item in ids])
    if not np.any(train_mask):
        raise RuntimeError("Round25 PPO normalization has no training sequences")
    mean = sequences[train_mask].mean(axis=(0, 1), keepdims=True)
    std = sequences[train_mask].std(axis=(0, 1), keepdims=True)
    std[std < 1.0e-5] = 1.0
    return mean.astype(np.float32), std.astype(np.float32), str(data_dir / "candidate_clearance.npz")


@dataclass(frozen=True)
class PolicyContext:
    """Deployment-only context at a legal high-level decision boundary."""

    history_rows: tuple[Mapping[str, Any], ...]
    phase: str
    contact_decision: ContactDecision | None
    stable: bool
    finite: bool
    candidate_map_label: int
    candidate_route_cost_m: float = 1.0
    zero_only_route_cost_m: float = 2.0
    probe_budget_remaining: int = 0
    already_probed: bool = True
    progress_monitor_stop_requested: bool = False
    contact_features: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        for row in self.history_rows:
            unexpected = set(row).difference(DEPLOYMENT_ROW_KEYS)
            if unexpected:
                raise ValueError(
                    "PolicyContext received non-deployment fields: "
                    + ",".join(sorted(str(key) for key in unexpected))
                )
        if int(self.candidate_map_label) not in (0, 1, 2):
            raise ValueError("candidate_map_label must be 0, 1, or 2")


@dataclass(frozen=True)
class PolicyDecision:
    policy: str
    version: str
    raw_action: str
    filtered_action: str
    valid_actions: tuple[str, ...]
    reason: str
    raw_scores: Mapping[str, float]
    belief: Mapping[str, float] | None
    deployment_fields_used: tuple[str, ...]
    model_checkpoint: str | None = None
    model_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEPLOYMENT_FIELDS_USED = tuple(sorted(DEPLOYMENT_ROW_KEYS))


def valid_actions_for_context(context: PolicyContext) -> tuple[str, ...]:
    if context.progress_monitor_stop_requested or not context.stable or not context.finite:
        return (HighLevelAction.STOP.value, HighLevelAction.RETRACT.value)
    if context.contact_decision in (ContactDecision.BLOCKED, ContactDecision.UNCERTAIN):
        return (HighLevelAction.RETRACT.value, HighLevelAction.STOP.value)
    if context.phase in {"pre_probe", "probe", "stand_and_probe"} and not context.already_probed:
        return (HighLevelAction.PROBE.value, HighLevelAction.STOP.value)
    return VALID_POST_PROBE_ACTIONS


class SafetySupervisor:
    """Independent safety layer; policy outputs never bypass this filter."""

    @staticmethod
    def filter_action(raw_action: str, context: PolicyContext, valid_actions: Sequence[str]) -> tuple[str, str]:
        allowed = tuple(str(item) for item in valid_actions)
        if context.progress_monitor_stop_requested:
            return HighLevelAction.STOP.value, "PROGRESS_MONITOR_STOP_REQUESTED"
        if not context.stable or not context.finite:
            return HighLevelAction.STOP.value if HighLevelAction.STOP.value in allowed else HighLevelAction.RETRACT.value, "UNSTABLE_OR_NONFINITE"
        if context.contact_decision in (ContactDecision.BLOCKED, ContactDecision.UNCERTAIN):
            return HighLevelAction.RETRACT.value, "NONRESPONSIVE_CONTACT_REQUIRES_RETRACT"
        normalized = str(raw_action)
        if normalized == "AVOID":
            normalized = HighLevelAction.DETOUR.value
        if normalized == "PROBE" and context.already_probed:
            return HighLevelAction.RETRACT.value, "PROBE_BUDGET_ALREADY_CONSUMED"
        if normalized not in allowed:
            return HighLevelAction.RETRACT.value, "ACTION_MASK_REJECTED"
        if normalized == HighLevelAction.CAUTIOUS_ADVANCE.value and context.contact_decision != ContactDecision.RESPONSIVE:
            return HighLevelAction.RETRACT.value, "ADVANCE_REQUIRES_RESPONSIVE_CONTACT"
        return normalized, "ACTION_ACCEPTED"


class _PolicyValueModel:
    """Small local loader matching the Round25 Method-B checkpoint contract."""

    def __init__(self, input_dim: int, recurrent: bool) -> None:
        import torch
        from torch import nn

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.recurrent = recurrent
                if recurrent:
                    self.rnn = nn.GRU(input_dim, 64, batch_first=True)
                    self.body = nn.Sequential(nn.Linear(64, 32), nn.Tanh())
                else:
                    self.body = nn.Sequential(nn.Linear(input_dim, 64), nn.Tanh(), nn.Linear(64, 32), nn.Tanh())
                self.policy = nn.Linear(32, 5)
                self.value = nn.Linear(32, 1)

            def forward(self, obs, hidden=None):
                if self.recurrent:
                    latent, hidden_out = self.rnn(obs.unsqueeze(1), hidden)
                    body = self.body(latent[:, -1])
                else:
                    body = self.body(obs)
                    hidden_out = None
                return self.policy(body), self.value(body).squeeze(-1), hidden_out

        self.torch = torch
        self.model = Model()

    def load(self, path: Path) -> dict[str, Any]:
        payload = self.torch.load(path, map_location="cpu", weights_only=False)
        state = payload["state_dict"]
        self.model.load_state_dict(state)
        self.model.eval()
        return payload

    def predict(self, vector: np.ndarray) -> tuple[str, dict[str, float]]:
        with self.torch.no_grad():
            logits, _, _ = self.model(self.torch.as_tensor(vector[None], dtype=self.torch.float32))
            probabilities = self.torch.softmax(logits, dim=-1).cpu().numpy()[0]
        scores = {name: float(value) for name, value in zip(MODEL_ACTIONS, probabilities)}
        return MODEL_ACTIONS[int(np.argmax(probabilities))], scores


class HighLevelPolicyAdapter:
    """R1/R2 and Round25 A/B policies behind one action contract."""

    def __init__(
        self,
        name: str,
        *,
        repo: Path,
        checkpoint: Path | None = None,
        risk_config: RiskPlannerConfig | None = None,
    ) -> None:
        self.name = str(name).lower()
        if self.name not in {"r1", "r2", "a", "b"}:
            raise ValueError("policy must be one of r1, r2, a, b")
        self.repo = repo.resolve()
        self.risk_config = risk_config or RiskPlannerConfig()
        self.checkpoint = checkpoint.resolve() if checkpoint is not None else None
        self.model_status: str | None = None
        self._a_model = None
        self._a_mean = None
        self._a_std = None
        self._b_model: _PolicyValueModel | None = None
        self._b_mean = None
        self._b_std = None
        if self.name == "a":
            self._load_a()
        elif self.name == "b":
            self._load_b()

    def _load_a(self) -> None:
        import torch
        from torch import nn

        path = self.checkpoint or self.repo / "outputs" / "round25_4070" / "method_a" / "a2_gru_seed0.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        class SequenceGRU(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gru = nn.GRU(34, 32, batch_first=True)
                self.head = nn.Sequential(nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 1))
            def forward(self, x):
                _, hidden = self.gru(x)
                return self.head(hidden[-1]).squeeze(-1)
        self._a_model = SequenceGRU()
        self._a_model.load_state_dict(payload["state_dict"])
        self._a_model.eval()
        self._a_mean = np.asarray(payload["mean"], dtype=np.float32).reshape(1, -1)
        self._a_std = np.asarray(payload["std"], dtype=np.float32).reshape(1, -1)
        self.checkpoint = path.resolve()
        self.model_status = str(payload.get("feature_contract", "UNKNOWN"))

    def _load_b(self) -> None:
        path = self.checkpoint or self.repo / "outputs" / "round25_4070" / "method_b" / "b0_standard_ppo_seed0.pt"
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=False)
        input_dim = int(payload.get("observation_dim", 145))
        recurrent = any(str(key).startswith("rnn.") for key in payload["state_dict"])
        self._b_model = _PolicyValueModel(input_dim, recurrent)
        self._b_model.load(path)
        self._b_mean, self._b_std, _ = _dataset_normalization(self.repo)
        self.checkpoint = path.resolve()
        self.model_status = str(payload.get("status", "UNKNOWN"))

    def _a_predict(self, context: PolicyContext) -> tuple[str, dict[str, float], TraversabilityBelief]:
        import torch
        sequence = build_deployment_feature_sequence(context.history_rows)
        normalized = (sequence - self._a_mean) / self._a_std
        with torch.no_grad():
            probability = float(torch.sigmoid(self._a_model(torch.as_tensor(normalized[None], dtype=torch.float32))).item())
        belief = TraversabilityBelief(
            p_clearance=probability,
            p_stall=1.0 - probability,
            p_safety_abort=0.05,
            uncertainty=min(1.0, abs(probability - 0.5) * 2.0),
        )
        planner_action = choose_action(PlannerInput(
            belief=belief,
            candidate_route_cost_m=float(context.candidate_route_cost_m),
            zero_only_route_cost_m=float(context.zero_only_route_cost_m),
            probe_budget_remaining=int(context.probe_budget_remaining),
            already_probed=bool(context.already_probed),
            candidate_map_label=int(context.candidate_map_label),
            stable=bool(context.stable and context.finite),
            phase=context.phase,
        ))
        return planner_action.value, {"p_clearance": probability}, belief

    def _b_predict(self, context: PolicyContext) -> tuple[str, dict[str, float]]:
        sequence = build_deployment_feature_sequence(context.history_rows)
        normalized = (sequence - self._b_mean.reshape(1, -1)) / self._b_std.reshape(1, -1)
        summary = np.concatenate((normalized[0], normalized[-1], normalized.mean(axis=0), normalized.std(axis=0)))
        extra = np.asarray([1.0, 2.0, 2.0, 1.0, 0.0], dtype=np.float32)
        # At this decision boundary the single probe has already been used,
        # no recovery has happened, and exactly one option step is recorded.
        suffix = np.asarray([0.0, 0.0, 0.25, 1.0], dtype=np.float32)
        vector = np.concatenate((summary.astype(np.float32), extra, suffix))
        if vector.size != 145:
            raise RuntimeError(f"Method B observation contract mismatch: {vector.size} != 145")
        return self._b_model.predict(vector)

    def decide(self, context: PolicyContext) -> PolicyDecision:
        valid = valid_actions_for_context(context)
        belief: TraversabilityBelief | None = None
        if self.name == "r1":
            raw = HighLevelAction.CAUTIOUS_ADVANCE.value if context.contact_decision == ContactDecision.RESPONSIVE else HighLevelAction.RETRACT.value
            scores = {raw: 1.0}
            reason = "FROZEN_R1_CONTACT_BRANCH"
        elif self.name == "r2":
            features = dict(context.contact_features or {})
            progress = _finite(features.get("hand_forward_progress_m"), 0.0)
            stall = _finite(features.get("hand_low_speed_duration_s"), float("inf"))
            tracking = _finite(features.get("upper_tracking_rms"), float("inf"))
            if context.contact_decision == ContactDecision.RESPONSIVE and progress >= 0.008 and stall <= 0.15 and tracking <= 0.30:
                raw = HighLevelAction.CAUTIOUS_ADVANCE.value
                reason = "R2_PROGRESS_TRACKING_SAFE"
            else:
                raw = HighLevelAction.RETRACT.value
                reason = "R2_PROGRESS_MONITOR_CONSERVATIVE_REJECT"
            scores = {raw: 1.0, "progress_m": progress, "stall_s": stall, "tracking_rms": tracking}
        elif self.name == "a":
            raw, scores, belief = self._a_predict(context)
            reason = "A_RISK_PLANNER_FROM_DEPLOYMENT_HISTORY"
        else:
            raw, scores = self._b_predict(context)
            reason = "B_SURROGATE_PPO_ACTION_FROM_DEPLOYMENT_HISTORY"
        filtered, filter_reason = SafetySupervisor.filter_action(raw, context, valid)
        return PolicyDecision(
            policy=self.name.upper(),
            version="ROUND25B_POLICY_INTERFACE_V1",
            raw_action=str(raw),
            filtered_action=str(filtered),
            valid_actions=tuple(valid),
            reason=f"{reason};{filter_reason}",
            raw_scores=scores,
            belief=None if belief is None else {
                "p_clearance": belief.p_clearance,
                "p_stall": belief.p_stall,
                "p_safety_abort": belief.p_safety_abort,
                "uncertainty": belief.uncertainty,
            },
            deployment_fields_used=DEPLOYMENT_FIELDS_USED,
            model_checkpoint=None if self.checkpoint is None else str(self.checkpoint),
            model_status=self.model_status,
        )


def privileged_information_leakage_audit() -> dict[str, Any]:
    return {
        "deployment_fields_used": list(DEPLOYMENT_FIELDS_USED),
        "forbidden_fields_not_read": list(FORBIDDEN_DEPLOYMENT_TOKENS),
        "policy_contract": "history_rows + public map/goal/phase + valid action mask",
        "safety_supervisor": "independent; filters every raw policy action",
        "localization_assumption": "Isaac root pose is explicitly phase-1 deployment assumption",
    }

