"""Frozen Round 4070-G E1_FIXED engineering baseline.

The rule is deliberately small and deployment-only.  It keeps the historical
E1 decision semantics (RESPONSIVE/BLOCKED/UNCERTAIN) while correcting the old
offline stall-duration unit error by integrating the actual observation
timestamps.  This module is the single source used by the offline baseline
report; it does not read scene truth, labels, or follow-up outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


E1_FIXED_VERSION = "E1_FIXED_G_V1_REAL_DT"
DECISIONS = ("RESPONSIVE", "BLOCKED", "UNCERTAIN")
DECISION_TO_ACTION = {
    "RESPONSIVE": "CAUTIOUS_ADVANCE",
    "BLOCKED": "RETRACT",
    "UNCERTAIN": "RETRACT",
}


@dataclass(frozen=True)
class Reference:
    """Frozen same-hand EMPTY response reference for one evaluation batch."""

    side: str
    episode_ids: tuple[str, ...]
    progress_m: float
    effort_mean: np.ndarray
    tracking_rms: float


def load_thresholds(path: Path) -> dict[str, Any]:
    """Load and validate the historical, condition-independent thresholds."""
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    required = {
        "minimum_samples",
        "responsive_progress_m",
        "blocked_progress_max_m",
        "responsive_progress_ratio_min",
        "blocked_progress_ratio_max",
        "blocked_effort_deviation_min",
        "blocked_stall_duration_s",
        "low_hand_speed_mps",
    }
    if set(payload) != required:
        raise ValueError(f"E1 threshold schema changed: expected {sorted(required)}, got {sorted(payload)}")
    if int(payload["minimum_samples"]) < 2:
        raise ValueError("E1 minimum_samples must be at least two")
    for key in required - {"minimum_samples"}:
        value = float(payload[key])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"E1 threshold {key} must be finite and non-negative")
    return payload


def feature_indices(names: Sequence[str]) -> dict[str, Any]:
    """Resolve the official named channels without signed hand mirroring."""
    index = {str(name): i for i, name in enumerate(names)}
    q = [str(name) for name in names if str(name).startswith("q/")]
    if len(q) != 29:
        raise ValueError(f"E1 requires the official 29-joint q block, got {len(q)}")
    q_cmd_names = ["q_cmd/" + name[2:] for name in q]
    effort_names = ["effort/" + name[2:] for name in q]
    if any(name not in index for name in (*q_cmd_names, *effort_names)):
        raise ValueError("E1 q/q_cmd/effort named-channel bijection is incomplete")
    required = {
        "left_delta": "left_hand_delta_initial_b/x",
        "right_delta": "right_hand_delta_initial_b/x",
        "left_velocity": "left_hand_velocity_initial_b/x",
        "right_velocity": "right_hand_velocity_initial_b/x",
    }
    if any(name not in index for name in required.values()):
        raise ValueError("E1 hand body-forward channels are incomplete")
    return {
        "q": [index[name] for name in q],
        "q_cmd": [index[name] for name in q_cmd_names],
        "effort": [index[name] for name in effort_names],
        **{key: index[name] for key, name in required.items()},
    }


def _validate_sequence(sequence: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(sequence, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or len(t) != len(x) or len(t) < 2:
        raise ValueError("E1 requires a two-dimensional sequence and at least two timestamps")
    if not np.isfinite(x).all() or not np.isfinite(t).all():
        raise ValueError("E1 sequence/timestamps must be finite")
    t = t - t[0]
    if not np.all(np.diff(t) > 0.0):
        raise ValueError("E1 timestamps must be strictly increasing")
    if t[-1] > 2.0 + 1.0e-9:
        raise ValueError(f"E1 history exceeds the frozen two-second cutoff: {t[-1]}")
    return x, t


def _single_features(
    sequence: np.ndarray,
    times: np.ndarray,
    side: str,
    indices: dict[str, Any],
    thresholds: dict[str, Any],
    reference: Reference,
) -> dict[str, Any]:
    x, t = _validate_sequence(sequence, times)
    if side not in ("left", "right"):
        raise ValueError(f"Unsupported E1 side: {side}")
    delta_index = indices[f"{side}_delta"]
    velocity_index = indices[f"{side}_velocity"]
    progress = float(x[-1, delta_index] - x[0, delta_index])
    ratio = None if abs(reference.progress_m) <= 1.0e-6 else float(progress / reference.progress_m)
    q_error = x[:, indices["q_cmd"]] - x[:, indices["q"]]
    effort = x[:, indices["effort"]]
    low_speed = np.abs(x[:-1, velocity_index]) <= float(thresholds["low_hand_speed_mps"])
    stall = float(np.sum(np.diff(t) * low_speed))
    return {
        "sample_count": int(len(t)),
        "cutoff_duration_s": float(t[-1]),
        "hand_forward_progress_m": progress,
        "baseline_hand_forward_progress_m": float(reference.progress_m),
        "hand_progress_ratio": ratio,
        "effort_deviation": float(np.mean(np.linalg.norm(effort - reference.effort_mean[None, :], axis=1))),
        "hand_low_speed_duration_s": stall,
        "upper_tracking_rms": float(np.sqrt(np.mean(q_error * q_error))),
        "engineering_coordinate": "initial robot body forward x",
        "stall_integration": "sum(diff(actual_time) * low_speed_at_left_endpoint); no terminal extrapolation",
        "baseline_source": "same-hand EMPTY control(s), frozen before evaluation",
        "reference_side": side,
    }


def build_references(
    episodes: Sequence[dict[str, Any]],
    raw: Sequence[np.ndarray],
    times: Sequence[np.ndarray],
    names: Sequence[str],
    threshold_path: Path,
) -> tuple[dict[str, Reference], dict[str, Any]]:
    """Build same-hand EMPTY references without reading any scene metadata."""
    thresholds = load_thresholds(threshold_path)
    indices = feature_indices(names)
    references: dict[str, Reference] = {}
    audit: dict[str, Any] = {
        "reference_policy": "same-hand EMPTY controls only; no test labels or scene metadata",
        "sides": {},
    }
    for side in ("left", "right"):
        selected = [
            i
            for i, episode in enumerate(episodes)
            if bool(episode.get("is_control"))
            and int(episode.get("y", -1)) == 1
            and str(episode.get("side")) == side
            and bool(episode.get("runtime_valid"))
            and bool(episode.get("probe_valid"))
        ]
        if not selected:
            raise ValueError(f"Missing legal EMPTY reference for {side}")
        effort_blocks = []
        progresses = []
        tracking = []
        ids = []
        for i in selected:
            x, t = _validate_sequence(raw[i], times[i])
            ids.append(str(episodes[i]["trial_id"]))
            effort_blocks.append(x[:, indices["effort"]])
            progresses.append(float(x[-1, indices[f"{side}_delta"]] - x[0, indices[f"{side}_delta"]]))
            q_error = x[:, indices["q_cmd"]] - x[:, indices["q"]]
            tracking.append(float(np.sqrt(np.mean(q_error * q_error))))
        reference = Reference(
            side=side,
            episode_ids=tuple(ids),
            progress_m=float(np.mean(progresses)),
            effort_mean=np.mean(np.concatenate(effort_blocks, axis=0), axis=0),
            tracking_rms=float(np.mean(tracking)),
        )
        references[side] = reference
        audit["sides"][side] = {
            "episode_ids": ids,
            "progress_m": reference.progress_m,
            "tracking_rms_rad": reference.tracking_rms,
            "reference_sample_counts": [int(len(raw[i])) for i in selected],
        }
    return references, audit


def classify(features: dict[str, Any], thresholds: dict[str, Any]) -> str:
    """Apply the frozen E1 if-else ordering and return native semantics."""
    if int(features["sample_count"]) < int(thresholds["minimum_samples"]):
        return "UNCERTAIN"
    progress = float(features["hand_forward_progress_m"])
    ratio = features["hand_progress_ratio"]
    responsive = progress >= float(thresholds["responsive_progress_m"]) and (
        ratio is None or float(ratio) >= float(thresholds["responsive_progress_ratio_min"])
    )
    if responsive:
        return "RESPONSIVE"
    blocked = (
        progress <= float(thresholds["blocked_progress_max_m"])
        or (ratio is not None and float(ratio) <= float(thresholds["blocked_progress_ratio_max"]))
    ) and (
        float(features["effort_deviation"]) >= float(thresholds["blocked_effort_deviation_min"])
        and float(features["hand_low_speed_duration_s"]) >= float(thresholds["blocked_stall_duration_s"])
    )
    return "BLOCKED" if blocked else "UNCERTAIN"


def evaluate(
    episodes: Sequence[dict[str, Any]],
    raw: Sequence[np.ndarray],
    times: Sequence[np.ndarray],
    names: Sequence[str],
    threshold_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate one frozen batch and return auditable rows plus reference audit."""
    if not (len(episodes) == len(raw) == len(times)):
        raise ValueError("E1 episode/raw/time lengths differ")
    thresholds = load_thresholds(threshold_path)
    indices = feature_indices(names)
    references, reference_audit = build_references(episodes, raw, times, names, threshold_path)
    rows: list[dict[str, Any]] = []
    for episode, sequence, timestamp in zip(episodes, raw, times):
        side = str(episode.get("side"))
        features = _single_features(sequence, timestamp, side, indices, thresholds, references[side])
        decision = classify(features, thresholds)
        label = str(episode.get("label", "LABEL_UNKNOWN"))
        if bool(episode.get("is_control")):
            actual = "CONTROL_PASSABLE" if int(episode.get("y", -1)) == 1 else "CONTROL_UNKNOWN"
            scope = "CONTROL"
        else:
            actual = {
                "PASSABLE_FOR_ACTION": "PASSABLE",
                "BLOCKED": "BLOCKED",
                "LABEL_UNKNOWN": "LABEL_UNKNOWN",
            }.get(label, "LABEL_UNKNOWN")
            scope = "CORE"
        operational = {"RESPONSIVE": "PASSABLE", "BLOCKED": "BLOCKED", "UNCERTAIN": "UNCERTAIN"}[decision]
        valid = scope == "CORE" and actual in {"PASSABLE", "BLOCKED"}
        determinate = operational in {"PASSABLE", "BLOCKED"}
        correct = bool(valid and determinate and operational == actual)
        false_pass = bool(valid and actual == "BLOCKED" and operational == "PASSABLE")
        false_reject = bool(valid and actual == "PASSABLE" and operational == "BLOCKED")
        rows.append(
            {
                "episode_id": str(episode["trial_id"]),
                "planned_trial_id": str(episode.get("planned_trial_id", "")),
                "scene_id": str(episode.get("scene_id", "")),
                "scene_family": str(episode.get("condition_family", "")),
                "physical_config_id": str(episode.get("physical_config_id", episode.get("group_id", ""))),
                "side": side,
                "seed": episode.get("seed"),
                "robot_variant": str(episode.get("robot_variant", "")),
                "experiment_role": str(episode.get("experiment_role", "")),
                "source_stage": str(episode.get("source_stage", "")),
                "is_control": bool(episode.get("is_control")),
                "admission_status": str(episode.get("admission_status", "")),
                "physics_valid": bool(episode.get("physics_valid")),
                "runtime_valid": bool(episode.get("runtime_valid")),
                "probe_valid": bool(episode.get("probe_valid")),
                "label_valid": bool(episode.get("label_valid")),
                "actual_label": actual,
                "label_evaluation_scope": scope,
                "e1_decision": decision,
                "upper_contract_action": DECISION_TO_ACTION[decision],
                "operational_prediction": operational,
                "determinate": determinate,
                "correct_determinate": correct,
                "false_pass": false_pass,
                "false_reject": false_reject,
                "passable_uncertain": bool(valid and actual == "PASSABLE" and not determinate),
                "blocked_uncertain": bool(valid and actual == "BLOCKED" and not determinate),
                "reference_episode_ids": "|".join(references[side].episode_ids),
                "e1_fixed_version": E1_FIXED_VERSION,
                "responsive_action_contract": "RESPONSIVE -> CAUTIOUS_ADVANCE; not a PASSABLE truth label",
                **features,
            }
        )
    reference_audit.update(
        {
            "e1_fixed_version": E1_FIXED_VERSION,
            "decision_semantics": list(DECISIONS),
            "decision_to_upper_action": DECISION_TO_ACTION,
            "thresholds": thresholds,
            "threshold_path": str(Path(threshold_path)),
            "thresholds_are_condition_independent": True,
            "uses_scene_metadata_for_decision": False,
            "uses_labels_for_decision": False,
            "uses_followup_or_future_observations": False,
        }
    )
    return rows, reference_audit


def compare_compatibility(
    rows: Iterable[dict[str, Any]],
    compatibility_rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Compare the raw E1 output with the existing G learning adapter."""
    expected = {row["episode_id"]: row["e1_decision"] for row in rows}
    observed = {
        row["episode_id"]: {
            "PASSABLE": "RESPONSIVE",
            "BLOCKED": "BLOCKED",
            "UNCERTAIN": "UNCERTAIN",
        }.get(str(row.get("E1")), "INVALID")
        for row in compatibility_rows
    }
    mismatches = [episode_id for episode_id, decision in expected.items() if observed.get(episode_id) != decision]
    return {
        "status": "PASS" if not mismatches else "MISMATCH",
        "compared": len(expected),
        "mismatches": mismatches,
        "adapter_semantics": "engineering_rules.py encodes RESPONSIVE as PASSABLE only for numeric offline scoring",
    }
