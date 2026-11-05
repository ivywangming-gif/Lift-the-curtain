"""Create a read-only audit of the frozen Round 4070 implementation/data."""

from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
RESEARCH = REPO / "research" / "round4070_failure_discovery"
OLD = REPO / "outputs" / "round4070_problem_discovery"


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return default


def parse_vector(row: dict[str, str], name: str) -> np.ndarray:
    return np.asarray(json.loads(row[name]), dtype=np.float64)


def phase_rows(trial: Path, phases: set[str]) -> list[dict[str, str]]:
    with (trial / "telemetry.csv").open(encoding="utf-8-sig", newline="") as stream:
        return [row for row in csv.DictReader(stream) if row.get("phase") in phases]


def features(rows: list[dict[str, str]]) -> dict[str, float | int | None]:
    if not rows:
        return {"sample_count": 0}
    hand = np.vstack([parse_vector(row, "deployment_hand_fk_position_w_json") for row in rows])
    hand_v = np.vstack([parse_vector(row, "deployment_hand_fk_velocity_w_json") for row in rows])
    q = np.vstack([parse_vector(row, "deployment_q_actual_json") for row in rows])
    q_cmd = np.vstack([parse_vector(row, "deployment_q_cmd_json") for row in rows])
    effort = np.vstack([parse_vector(row, "deployment_motor_effort_json") for row in rows])
    base = np.vstack([parse_vector(row, "deployment_base_position_w_json") for row in rows])
    projected_speed = hand_v[:, 0]
    return {
        "sample_count": int(len(rows)),
        "hand_forward_progress_m": float(hand[-1, 0] - hand[0, 0]),
        "hand_forward_speed_mean_mps": float(np.mean(projected_speed)),
        "hand_low_speed_duration_s": float(np.count_nonzero(np.abs(projected_speed) <= 0.004) * 0.05),
        "upper_tracking_rms": float(np.sqrt(np.mean((q_cmd - q) ** 2))),
        "effort_rms": float(np.sqrt(np.mean(effort**2))),
        "effort_deviation_from_first": float(np.mean(np.linalg.norm(effort - effort[0][None, :], axis=1))),
        "base_disturbance_m": float(np.linalg.norm(base[-1] - base[0])),
        "hand_speed_peak_mps": float(np.max(np.abs(projected_speed))),
    }


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)
    return result.stdout.strip()


def audit_trials() -> list[dict[str, Any]]:
    index = read_json(OLD / "trial_index.json", {}) or {}
    rows: list[dict[str, Any]] = []
    for item in index.get("records", []):
        trial_id = str(item.get("trial_id"))
        trial = OLD / "trials" / trial_id
        summary = read_json(trial / "summary.json", {}) or {}
        record: dict[str, Any] = {
            "trial_id": trial_id,
            "scenario": item.get("scenario"),
            "probe_side": item.get("probe_side"),
            "seed": item.get("seed"),
            "runtime_valid": item.get("runtime_valid"),
            "local_response": item.get("decision"),
            "whole_body_clearance": item.get("whole_body_clearance"),
            "pointgoal_success": item.get("pointgoal_success"),
            "summary_status": summary.get("runtime_status", summary.get("status")),
        }
        if (trial / "telemetry.csv").is_file():
            record["T0_preprobe"] = features(phase_rows(trial, {"physical_preprobe", "physical_preprobe_hold"}))
            record["T1_probe"] = features(phase_rows(trial, {"physical_probe", "physical_probe_hold"}))
            record["T2_advance"] = features(
                phase_rows(trial, {name for name in set(row.get("phase", "") for row in csv.DictReader((trial / "telemetry.csv").open(encoding="utf-8-sig", newline=""))) if "cautious_advance" in name})
            )
        rows.append(record)
    return rows


def source_contract() -> dict[str, Any]:
    path = REPO / "lift_curtain" / "control" / "deployment_observation.py"
    text = path.read_text(encoding="utf-8")
    direct = re.search(r"def classify_proprioceptive_response[\s\S]*?return ContactDecision\.UNCERTAIN", text)
    return {
        "source": str(path),
        "classifier_function_present": direct is not None,
        "direct_branch_variables": [
            "hand_forward_progress_m",
            "hand_progress_ratio",
            "effort_deviation",
            "hand_low_speed_duration_s",
        ],
        "recorded_but_not_direct_classifier_variables": [
            "q",
            "qdot",
            "q_cmd",
            "upper_tracking_rms",
            "joint_velocity_mean_norm",
            "base_disturbance_m",
            "previous_action",
        ],
        "local_response_labels": ["RESPONSIVE", "BLOCKED", "UNCERTAIN"],
        "upper_level_labels": ["PASSABLE_FOR_ACTION", "BLOCKED_FOR_ACTION", "UNKNOWN"],
        "effort_source_chain": ["applied_torque", "computed_torque", "joint_effort_target", "unavailable"],
        "deployment_forbidden_truth": [
            "object identity", "material parameters", "hidden rigid truth", "deformable nodes",
            "future clearance", "simulator contact ground truth",
        ],
    }


def main() -> None:
    RESEARCH.mkdir(parents=True, exist_ok=True)
    records = audit_trials()
    with (RESEARCH / "BASELINE_FEATURE_AUDIT.json").open("w", encoding="utf-8") as stream:
        json.dump(records, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    with (RESEARCH / "BASELINE_FEATURE_AUDIT.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "trial_id", "scenario", "probe_side", "seed", "runtime_valid", "local_response",
            "whole_body_clearance", "pointgoal_success", "summary_status",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({field: row.get(field) for field in fields})

    refs = []
    for path in (
        REPO / "reports" / "ROUND4070_FINAL_REPORT_ZH.md",
        REPO / "research" / "round4070_problem_discovery" / "IMPLEMENTATION_AUDIT.md",
        REPO / "research" / "round4070_problem_discovery" / "PHYSICAL_SCENE_VALIDATION.md",
        REPO / "configs" / "proprioceptive_thresholds.json",
        REPO / "scripts" / "round4070_problem_discovery" / "run_episode.py",
        REPO / "lift_curtain" / "control" / "deployment_observation.py",
    ):
        refs.append({"path": str(path), "present": path.is_file(), "sha256": sha256(path)})
    provenance = {
        "schema": "lift-the-curtain.round4070-failure-discovery-baseline-audit.v1",
        "audited_at": datetime.now().astimezone().isoformat(),
        "git_head": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "working_tree_status": git("status", "--short"),
        "historical_outputs_modified": False,
        "source_references": refs,
        "implementation_contract": source_contract(),
        "historical_trial_count": len(records),
        "historical_runtime_valid_count": sum(str(row.get("runtime_valid")).lower() == "true" for row in records),
    }
    (RESEARCH / "IMPLEMENTATION_AUDIT.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = read_json(OLD / "campaign_summary.json", {}) or {}
    report = [
        "# Round 4070-B 实现与基线审计",
        "",
        f"审计时间：{provenance['audited_at']}。本审计只读取上一轮报告和原始 artifact；没有修改 `outputs/round4070_problem_discovery/`。",
        "",
        "## 冻结实现",
        "",
        "- FALCON、G1 资产、Isaac Sim/PhysX 版本、DT=0.005 s、decimation=4、single-env/headless/serial 均沿用上一轮。",
        "- 当前局部响应分类器直接使用手部前向进展、相对空场进展比、effort deviation、低速持续时间。",
        "- `q`、`qdot`、`q_cmd`、upper tracking RMS、joint velocity、base disturbance、previous action 被记录或进入控制历史，但不应被描述为已进入旧分类器的直接 if-else 分支。",
        "- 左右手是独立 episode 的 active hand；本轮部署决策不能读取场景身份、刚板真值、布节点、未来 clearance 或 simulator contact truth。",
        "",
        "## 时间边界",
        "",
        "- T0：`physical_preprobe` + `physical_preprobe_hold`。",
        "- T1：`physical_probe` + `physical_probe_hold`，这是当前工程判断的观察截止点。",
        "- T2：`round26_cautious_advance_*`，包含额外机械动作和状态改变，只能作为后果分析，不能当作纯增加历史长度的因果证据。",
        "",
        "## 上一轮核查",
        "",
        f"上一轮 campaign summary 记录的新增 trial：{summary.get('trial_accounting', {}).get('launched_g1_trials', 'unknown')}；有效：{summary.get('trial_accounting', {}).get('valid_completed_g1_trials', 'unknown')}；runtime-invalid：{summary.get('trial_accounting', {}).get('runtime_invalid_or_incomplete_g1_trials', 'unknown')}。",
        "nominal 与 bottom_right_anchor 的差异不能直接归因为材料；需要同时比较早期 T1 特征、帘幕局部形变、身体包络和后续 continuation。",
        "",
        "详细机器结果：`IMPLEMENTATION_AUDIT.json`、`BASELINE_FEATURE_AUDIT.json`、`BASELINE_FEATURE_AUDIT.csv`。",
    ]
    (RESEARCH / "IMPLEMENTATION_AND_BASELINE_AUDIT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"historical_trials": len(records), "valid": provenance["historical_runtime_valid_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
