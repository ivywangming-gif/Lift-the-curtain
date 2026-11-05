"""Freeze the pre-registered early-feature matching rule once."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
from zoneinfo import ZoneInfo


REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "research" / "round4070_failure_discovery" / "ANALYSIS_RULES.json"


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"Refusing to overwrite frozen rules: {OUTPUT}")
    payload = {
        "schema": "lift-the-curtain.round4070-failure-discovery-analysis-rules.v1",
        "frozen_at": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
        "cutoff": "T1 physical_probe plus physical_probe_hold; no cautious-advance samples",
        "same_required_fields": [
            "probe_side",
            "probe command",
            "probe duration",
            "policy r1",
            "effort_source",
            "seed",
            "reset_root_y",
            "reset_root_yaw",
        ],
        "maximum_absolute_differences": {
            "hand_forward_delta_m": 0.003,
            "base_delta_m": 0.003,
            "upper_tracking_rms_mean_rad": 0.005,
            "effort_rms_mean": 1.0,
            "initial_base_xy_m": 0.005,
            "initial_yaw_rad": 0.005,
        },
        "interpretation": "A pair satisfying every limit is an early-feature matched candidate; a different clearance outcome still requires independent confirmation before a causal claim.",
        "deployment_truth_excluded": [
            "obstacle identity",
            "material parameters",
            "hidden rigid truth",
            "deformable mesh nodes",
            "future clearance",
            "simulator contact ground truth",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "frozen_at": payload["frozen_at"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
