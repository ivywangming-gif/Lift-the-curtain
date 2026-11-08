"""Contract and regression tests for the frozen Round 4070-G E1 baseline."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from if_else_baseline import (  # noqa: E402
    DECISION_TO_ACTION,
    _single_features,
    classify,
    evaluate,
    feature_indices,
    load_thresholds,
    Reference,
)
from engineering_rules import evaluate_rules  # noqa: E402
from run_if_else_baseline import load_current_batch  # noqa: E402


class IfElseBaselineContractTests(unittest.TestCase):
    @staticmethod
    def synthetic_names() -> list[str]:
        joints = [f"joint_{i}" for i in range(29)]
        return (
            [f"q/{name}" for name in joints]
            + [f"q_cmd/{name}" for name in joints]
            + [f"effort/{name}" for name in joints]
            + [
                "left_hand_delta_initial_b/x",
                "right_hand_delta_initial_b/x",
                "left_hand_velocity_initial_b/x",
                "right_hand_velocity_initial_b/x",
            ]
        )

    def test_real_timestamp_stall_integration_is_not_fixed_sample_period(self) -> None:
        names = self.synthetic_names()
        indices = feature_indices(names)
        sequence = np.zeros((4, len(names)), dtype=np.float32)
        sequence[:, indices["effort"]] = 3.0
        sequence[-1, indices["left_delta"]] = 0.002
        sequence[:, indices["left_velocity"]] = [0.0, 0.01, 0.0, 0.01]
        reference = Reference(
            side="left",
            episode_ids=("empty-left",),
            progress_m=1.0,
            effort_mean=np.zeros(29, dtype=np.float64),
            tracking_rms=0.0,
        )
        thresholds = load_thresholds(ROOT / "configs/proprioceptive_thresholds.json")
        features = _single_features(
            sequence,
            np.asarray([0.0, 0.1, 0.9, 1.0]),
            "left",
            indices,
            thresholds,
            reference,
        )
        self.assertAlmostEqual(features["hand_low_speed_duration_s"], 0.2, places=12)
        self.assertNotAlmostEqual(features["hand_low_speed_duration_s"], 0.01, places=6)

    def test_frozen_three_state_semantics_and_upper_mapping(self) -> None:
        thresholds = load_thresholds(ROOT / "configs/proprioceptive_thresholds.json")
        responsive = {
            "sample_count": 10,
            "hand_forward_progress_m": thresholds["responsive_progress_m"],
            "hand_progress_ratio": thresholds["responsive_progress_ratio_min"],
            "effort_deviation": 0.0,
            "hand_low_speed_duration_s": 0.0,
        }
        blocked = {
            "sample_count": 10,
            "hand_forward_progress_m": thresholds["blocked_progress_max_m"],
            "hand_progress_ratio": thresholds["blocked_progress_ratio_max"],
            "effort_deviation": thresholds["blocked_effort_deviation_min"],
            "hand_low_speed_duration_s": thresholds["blocked_stall_duration_s"],
        }
        uncertain = dict(blocked, hand_low_speed_duration_s=0.0)
        self.assertEqual(classify(responsive, thresholds), "RESPONSIVE")
        self.assertEqual(classify(blocked, thresholds), "BLOCKED")
        self.assertEqual(classify(uncertain, thresholds), "UNCERTAIN")
        self.assertEqual(DECISION_TO_ACTION, {
            "RESPONSIVE": "CAUTIOUS_ADVANCE",
            "BLOCKED": "RETRACT",
            "UNCERTAIN": "RETRACT",
        })

    def test_named_bilateral_mapping_and_scene_metadata_independence(self) -> None:
        names = self.synthetic_names()
        shuffled = list(reversed(names))
        resolved = feature_indices(shuffled)
        self.assertEqual(shuffled[resolved["left_delta"]], "left_hand_delta_initial_b/x")
        self.assertEqual(shuffled[resolved["right_delta"]], "right_hand_delta_initial_b/x")
        self.assertEqual(shuffled[resolved["left_velocity"]], "left_hand_velocity_initial_b/x")
        self.assertEqual(shuffled[resolved["right_velocity"]], "right_hand_velocity_initial_b/x")

        sequence = np.zeros((4, len(names)), dtype=np.float32)
        idx = feature_indices(names)
        sequence[-1, idx["left_delta"]] = 0.02
        sequence[-1, idx["right_delta"]] = 0.02
        sequence[:, idx["left_velocity"]] = 0.01
        sequence[:, idx["right_velocity"]] = 0.01
        episodes = [
            {"trial_id": "empty", "side": "left", "is_control": True, "y": 1, "label": "PASSABLE_FOR_ACTION",
             "runtime_valid": True, "probe_valid": True, "scene_id": "EMPTY", "condition_family": "E"},
            {"trial_id": "probe", "side": "left", "is_control": False, "y": -1, "label": "LABEL_UNKNOWN",
             "runtime_valid": True, "probe_valid": True, "scene_id": "A04", "condition_family": "A",
             "material_fidelity": "not_used"},
            {"trial_id": "empty-right", "side": "right", "is_control": True, "y": 1, "label": "PASSABLE_FOR_ACTION",
             "runtime_valid": True, "probe_valid": True, "scene_id": "EMPTY", "condition_family": "E"},
            {"trial_id": "probe-right", "side": "right", "is_control": False, "y": -1, "label": "LABEL_UNKNOWN",
             "runtime_valid": True, "probe_valid": True, "scene_id": "A04", "condition_family": "A",
             "material_fidelity": "not_used"},
        ]
        rows_a, _ = evaluate(
            episodes,
            [sequence] * 4,
            [np.asarray([0.0, 0.1, 0.2, 0.3])] * 4,
            names,
            ROOT / "configs/proprioceptive_thresholds.json",
        )
        altered = [dict(item, scene_id="D01", condition_family="D", material_fidelity="changed") for item in episodes]
        rows_b, _ = evaluate(
            altered,
            [sequence] * 4,
            [np.asarray([0.0, 0.1, 0.2, 0.3])] * 4,
            names,
            ROOT / "configs/proprioceptive_thresholds.json",
        )
        self.assertEqual([row["e1_decision"] for row in rows_a], [row["e1_decision"] for row in rows_b])
        self.assertEqual(rows_a[1]["hand_forward_progress_m"], rows_b[1]["hand_forward_progress_m"])

    def test_current_primary_histories_regression(self) -> None:
        batch = load_current_batch()
        self.assertEqual(len(batch["episodes"]), 21)
        self.assertEqual(len(batch["accepted_manifest"]["episodes"]), 18)
        self.assertEqual(len(batch["pending_manifest"]["episodes"]), 3)
        self.assertEqual(len(batch["names"]), 252)
        rows, _ = evaluate(
            batch["episodes"],
            batch["raw"],
            batch["times"],
            batch["names"],
            ROOT / "configs/proprioceptive_thresholds.json",
        )
        self.assertEqual(Counter(row["e1_decision"] for row in rows), {
            "RESPONSIVE": 15,
            "BLOCKED": 4,
            "UNCERTAIN": 2,
        })
        train_indices = [
            i for i, episode in enumerate(batch["episodes"])
            if episode.get("is_control") and int(episode.get("y", -1)) == 1
        ]
        compatibility, _ = evaluate_rules(
            batch["episodes"], batch["raw"], batch["times"], batch["names"], train_indices
        )
        compatibility = [dict(item, episode_id=episode["trial_id"]) for item, episode in zip(compatibility, batch["episodes"])]
        canonical = {row["episode_id"]: row["e1_decision"] for row in rows}
        adapter = {
            row["episode_id"]: {"PASSABLE": "RESPONSIVE", "BLOCKED": "BLOCKED", "UNCERTAIN": "UNCERTAIN"}[row["E1"]]
            for row in compatibility
        }
        self.assertEqual(canonical, adapter)
        self.assertEqual(
            {row["episode_id"] for row in rows if row["actual_label"] == "LABEL_UNKNOWN"},
            set(batch["pending_ids"]) | {
                episode["trial_id"] for episode in batch["accepted_manifest"]["episodes"] if episode["label"] == "LABEL_UNKNOWN"
            },
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
