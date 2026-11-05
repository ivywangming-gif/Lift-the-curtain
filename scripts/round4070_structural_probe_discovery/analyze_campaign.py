"""Read-only aggregation for Round 4070-C structural probe discovery."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
RESEARCH = REPO / "research" / "round4070_structural_probe_discovery"
OUTPUT = REPO / "outputs" / "round4070_structural_probe_discovery"
TRIALS = OUTPUT / "trials"
PHYSICS = OUTPUT / "physics"
MANIFEST = REPO / "configs" / "round4070_structural_probe_discovery" / "PREREGISTERED_MANIFEST.json"
MANIFEST_NAMES = ("PREREGISTERED_MANIFEST.json", "CONFIRMATORY_MANIFEST.json", "INDEPENDENT_CONFIRMATORY_MANIFEST.json")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return default


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def boolean(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "pass"}:
        return True
    if text in {"false", "0", "no", "fail"}:
        return False
    return None


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def gate_id(plan: dict[str, Any]) -> str:
    condition = str(plan.get("condition_id", ""))
    topology = str(plan.get("structural_topology", "A0_TOP_EDGE_ONLY"))
    if condition == "EMPTY_CONTROL":
        return "CONTROL_ONLY"
    if condition == "RIGID_CONTROL":
        return "INHERITED_RIGID_CONTROL"
    if condition == "A3_FULL_PERIMETER_FIXED_V2":
        return "A3_FULL_PERIMETER_FIXED_V2"
    if condition in {"A0_TOP_EDGE_ONLY", "A1_TOP_EDGE_BOTTOM_TWO_CORNERS", "A2_FOUR_CORNERS_ONLY", "B_D225_NEW", "B_T003_NEW", "C_GAP05", "C_GAP10", "C_GAP20"}:
        return condition
    return topology


def gate_status(gate: str) -> str:
    if gate == "CONTROL_ONLY":
        return "CONTROL_ONLY"
    if gate == "INHERITED_RIGID_CONTROL":
        return "INHERITED_PASS_FROM_ROUND4070_B"
    root = PHYSICS / gate
    no_probe = read_json(root / "no_probe" / "summary.json", {}) or {}
    probe = read_json(root / "probe" / "summary.json", {}) or {}
    no_status = str(no_probe.get("status", "MISSING"))
    probe_status = str(probe.get("status", "NOT_RUN"))
    return f"no_probe={no_status};probe={probe_status}"


def limitation(row: dict[str, Any]) -> str:
    if not row["launched"]:
        return "INSUFFICIENT_EVIDENCE"
    if not row["runtime_valid"]:
        return "EXECUTION_INVALID"
    if row["group"] == "EMPTY_CONTROL":
        return "NONE_CONTROL"
    if row["gate_status"].find("FAIL_PHYSICS_SANITY") >= 0:
        return "PHYSICS_INVALID"
    if row["external_safety_reason"] or row["probe_safety_stop_reason"]:
        return "HIGH_RESISTANCE_AMBIGUITY"
    if row["local_response"] == "RESPONSIVE" and not row["clearance"]:
        return "OBSERVATION_HORIZON_LIMITATION"
    if row["local_response"] in {"BLOCKED", "UNCERTAIN"} and not row["clearance"]:
        if row["condition_id"] in {"RIGID_CONTROL", "A3_FULL_PERIMETER_FIXED_V2"}:
            return "SPATIAL_CONSTRAINT_LIMITATION"
        return "HIGH_RESISTANCE_AMBIGUITY"
    if row["clearance"] and not row["pointgoal"]:
        return "TEMPORAL_RESPONSE_AMBIGUITY"
    if row["local_response"] == "RESPONSIVE" and row["curtain_deformation_m"] is not None and row["curtain_deformation_m"] < 0.002:
        return "FEATURE_UTILIZATION_LIMITATION"
    return "INSUFFICIENT_EVIDENCE"


def trial_row(plan: dict[str, Any]) -> dict[str, Any]:
    trial_id = str(plan["trial_id"])
    path = TRIALS / trial_id
    summary = read_json(path / "summary.json", {}) or {}
    probe = summary.get("physical_probe_result") or {}
    resources = read_json(path / "resource_summary.json", {}) or {}
    launched = (path / "command.json").is_file()
    runtime_valid = str(summary.get("runtime_status", "")) == "RUNTIME_PASS_FINALIZE_REACHED"
    timeout_reason = resources.get("timeout_reason")
    recorded_status = summary.get("status")
    if not recorded_status and timeout_reason:
        recorded_status = f"RUNTIME_INVALID_{timeout_reason}"
    if not recorded_status and launched and not runtime_valid:
        recorded_status = "RUNTIME_INVALID_NO_SUMMARY"
    clearance = boolean(summary.get("candidate_clearance_verified"))
    pointgoal = boolean(summary.get("pointgoal_success"))
    row: dict[str, Any] = {
        "trial_id": trial_id,
        "stage": plan.get("stage"),
        "group": plan.get("group"),
        "condition_id": plan.get("condition_id"),
        "scenario": plan.get("scenario"),
        "probe_side": plan.get("probe_side"),
        "seed": plan.get("seed"),
        "probe_depth_level": plan.get("probe_depth_level"),
        "depth_increment_m": (plan.get("probe_depth_level") and {"D0_BASELINE": 0.0, "D1_PLUS_020M": 0.02, "D2_PLUS_040M": 0.04, "D3_PLUS_060M": 0.06}.get(str(plan.get("probe_depth_level")))),
        "structural_topology": plan.get("structural_topology"),
        "density_scale_multiplier": plan.get("density_scale_multiplier", 1.0),
        "hidden_rigid_x": plan.get("hidden_rigid_x"),
        "gate_id": gate_id(plan),
        "gate_status": gate_status(gate_id(plan)),
        "launched": launched,
        "runtime_valid": runtime_valid,
        "runtime_status": summary.get("runtime_status") or (f"RUNTIME_INVALID_{timeout_reason}" if timeout_reason else None),
        "status": recorded_status,
        "local_response": probe.get("decision", "UNKNOWN"),
        "upper_action": (probe.get("high_level_policy") or {}).get("filtered_action", "UNKNOWN"),
        "upper_evidence_status": probe.get("round26_evidence_status"),
        "contact_confirmed": probe.get("evaluation_hand_contact_confirmed"),
        "contact_force_max_N": finite(probe.get("evaluation_hand_contact_force_max_N")),
        "target_hand_forward_estimate_m": finite(summary.get("probe_target_hand_forward_distance_m_estimate")),
        "actual_hand_forward_m": finite(summary.get("probe_actual_hand_forward_progress_m")),
        "continued_forward_after_contact_m": finite(summary.get("probe_actual_continued_forward_after_contact_m_evaluation_only")),
        "contact_to_retract_s": finite(summary.get("probe_contact_to_retract_s_evaluation_only")),
        "curtain_deformation_m": finite(summary.get("probe_curtain_deformation_m_evaluation_only")),
        "deformation_to_hand_ratio": finite(summary.get("probe_curtain_deformation_to_hand_forward_ratio_evaluation_only")),
        "q_safety_stop": summary.get("probe_safety_stop_reason"),
        "external_safety_reason": summary.get("external_safety_reason"),
        "probe_safety_stop_reason": summary.get("probe_safety_stop_reason"),
        "clearance": clearance,
        "pointgoal": pointgoal,
        "task_success": boolean(summary.get("task_success")),
        "physical_branch_pass": boolean(summary.get("physical_branch_pass")),
        "safe_recovery": boolean(summary.get("safe_recovery")),
        "max_effort_ratio": finite(summary.get("max_effort_ratio")),
        "max_upper_tracking_rms_rad": finite(summary.get("max_upper_tracking_rms_rad")),
        "max_curtain_region_displacement_m": finite(summary.get("max_curtain_region_displacement_m")),
        "max_curtain_speed_mps": finite(summary.get("max_curtain_speed_mps")),
        "gpu_peak_mb": finite(resources.get("gpu_memory_peak_mb")),
        "process_private_peak_mb": finite(resources.get("process_private_peak_mb")),
        "summary_path": str(path / "summary.json"),
    }
    row["judgment_vs_clearance"] = (
        "CONTROL" if row["group"] == "EMPTY_CONTROL" else
        "RESPONSIVE_CLEARANCE_CONFIRMED" if row["local_response"] == "RESPONSIVE" and row["clearance"] else
        "RESPONSIVE_CLEARANCE_UNCONFIRMED" if row["local_response"] == "RESPONSIVE" else
        "BLOCKED_OR_UNCERTAIN_CLEARANCE_NOT_CONFIRMED" if not row["clearance"] else
        "CLEARANCE_WITHOUT_FINAL_POINTGOAL"
    )
    row["limitation_category"] = limitation(row)
    return row


def physics_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(PHYSICS.rglob("summary.json")):
        if "flexible" in summary_path.parts:
            continue
        summary = read_json(summary_path, {}) or {}
        rows.append({
            "gate_id": summary.get("condition_id") or summary_path.parent.parent.name,
            "mode": summary.get("mode"),
            "scenario": (summary.get("scenario") or {}).get("name") if isinstance(summary.get("scenario"), dict) else summary.get("scenario"),
            "status": summary.get("status"),
            "physics_status": summary.get("physics_status"),
            "contact_status": summary.get("contact_status"),
            "structural_topology": summary.get("structural_topology"),
            "requested_fixed_count": len(summary.get("requested_fixed_vertex_indices") or []),
            "actual_fixed_count": len(summary.get("actual_fixed_vertex_indices_low_level") or []),
            "attachment_vertex_set_exact": (summary.get("acceptance") or {}).get("attachment_vertex_set_exact"),
            "max_fixed_drift_m": finite(summary.get("max_fixed_drift_m")),
            "max_displacement_m": finite(summary.get("max_displacement_m")),
            "max_speed_mps": finite(summary.get("max_speed_mps")),
            "max_neighbor_stretch_ratio": finite(summary.get("max_neighbor_stretch_ratio")),
            "min_hidden_wall_face_gap_m": finite(summary.get("min_hidden_wall_face_gap_m")),
            "max_nodes_at_or_beyond_hidden_wall_front": summary.get("max_nodes_at_or_beyond_hidden_wall_front"),
            "summary_path": str(summary_path),
        })
    for summary_path in sorted((PHYSICS / "flexible").rglob("summary.json")):
        summary = read_json(summary_path, {}) or {}
        rows.append({
            "gate_id": summary_path.parent.parent.name,
            "mode": summary.get("mode"),
            "scenario": (summary.get("scene") or {}).get("name") if isinstance(summary.get("scene"), dict) else summary.get("scene"),
            "status": summary.get("status"),
            "physics_status": summary.get("physics_status"),
            "contact_status": summary.get("contact_status"),
            "structural_topology": "A0_TOP_EDGE_ONLY",
            "requested_fixed_count": None,
            "actual_fixed_count": None,
            "attachment_vertex_set_exact": (summary.get("acceptance") or {}).get("attachment_vertex_sets_exact"),
            "max_fixed_drift_m": None,
            "max_displacement_m": finite(summary.get("max_displacement_m")),
            "max_speed_mps": finite(summary.get("max_speed_mps")),
            "max_neighbor_stretch_ratio": finite(summary.get("max_neighbor_stretch_ratio")),
            "min_hidden_wall_face_gap_m": None,
            "max_nodes_at_or_beyond_hidden_wall_front": None,
            "model_contract": summary.get("model_contract", "ARTICULATED_STRIP_SURROGATE"),
            "real_pvc_claim": summary.get("real_pvc_claim", False),
            "summary_path": str(summary_path),
        })
    return rows


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join("---" for _ in fields) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def ratio_text(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "N/A"
    return f"{numerator}/{denominator} ({100.0 * numerator / denominator:.1f}%)"


def scene_summary(rows: list[dict[str, Any]], physics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    physics_by_gate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in physics:
        physics_by_gate[str(item.get("gate_id"))].append(item)

    specs = [
        ("EMPTY_CONTROL", "空场渐进深度控制", "EMPTY_CONTROL", "CONTROL_ONLY", "D0/D1/D3 左右均启动；D3 独立 seed 复核已登记。"),
        ("RIGID_CONTROL", "已有固定刚板控制", "RIGID_CONTROL", "INHERITED_RIGID_CONTROL", "沿用 Round 4070-B 已验证刚板控制 gate。"),
        ("A0_TOP_EDGE_ONLY", "普通单片布帘 A0", "A0_TOP_EDGE_ONLY", "A0_TOP_EDGE_ONLY", "D0/D1 左右。"),
        ("A1_TOP_EDGE_BOTTOM_TWO_CORNERS", "A1 顶边加底部两角", "A1_TOP_EDGE_BOTTOM_TWO_CORNERS", "A1_TOP_EDGE_BOTTOM_TWO_CORNERS", "D0 同 seed 加深前置；D2 同 seed 与 seed 4071 独立确认。"),
        ("A2_FOUR_CORNERS_ONLY", "A2 仅四角", "A2_FOUR_CORNERS_ONLY", "A2_FOUR_CORNERS_ONLY", "D0 左右。"),
        ("A3_FULL_PERIMETER_FIXED_V2", "A3 全周边固定修复版", "A3_FULL_PERIMETER_FIXED_V2", "A3_FULL_PERIMETER_FIXED_V2", "D0 左右；旧 A3 请求集合错误保留为审计证据。"),
        ("B_D225_NEW", "B 密度 x2.25", "B_D225_NEW", "B_D225_NEW", "D0 左右；左侧 execution invalid。"),
        ("C_GAP05", "隐藏墙安装间隙约 5 cm", "C_GAP05", "C_GAP05", "仅左侧 D0，物理 gate 通过。"),
        ("C_GAP10", "隐藏墙安装间隙约 10 cm", "C_GAP10", "C_GAP10", "仅右侧 D0，物理 gate 通过。"),
        ("B_T003_NEW", "B 厚度 0.003 m", None, "B_T003_NEW", "物理 gate 通过，但未进入冻结 22 条 G1 候选；预算优先保留 D225、拓扑和隐藏墙条件。"),
        ("C_NEAR_REFERENCE", "隐藏墙近距参考 S1B/S1C", None, None, "只完成旧 Round 4070-B 几何复核；本轮不重复启动 G1。"),
        ("C_GAP20", "隐藏墙安装间隙约 20 cm", None, "C_GAP20", "probe physics gate 前表面穿越，阻止 G1。"),
        ("D_STRIPS_SURROGATE", "多条带 PVC 替代结构", None, "strips_3_overlap_layered40", "ARTICULATED_STRIP_SURROGATE 无探测 physics gate 失败，未启动 G1。"),
        ("E_MULTI_PANEL_OPENING", "中开/左开/右开多片场景", None, None, "P1 分支未获得稳定且物理有效的多片候选，未进入 G1。"),
        ("F_LENGTH_RECOVERY", "长短与恢复行为", None, None, "P2 分支未占用核心 P0 探索预算，未启动新 G1。"),
    ]

    output: list[dict[str, Any]] = []
    for scene_id, label, condition_id, gate_id, note in specs:
        condition_rows = [row for row in rows if condition_id and row["condition_id"] == condition_id]
        launched = [row for row in condition_rows if row["launched"]]
        valid = [row for row in launched if row["runtime_valid"]]
        evaluable = [row for row in valid if row["clearance"] is not None]
        correct = sum((row["local_response"] == "RESPONSIVE") == bool(row["clearance"]) for row in evaluable)
        independent = [row for row in condition_rows if str(row.get("stage", "")).startswith("independent_confirmatory")]
        independent_valid = [row for row in independent if row["runtime_valid"]]
        gate_rows = physics_by_gate.get(str(gate_id), []) if gate_id else []
        probe_rows = [item for item in gate_rows if item.get("mode") == "probe"]
        probe_pass = sum(item.get("status") == "PASS_PHYSICS_SANITY" for item in probe_rows)
        if gate_id == "CONTROL_ONLY":
            physics_rate = "N/A (control)"
        elif gate_id == "INHERITED_RIGID_CONTROL":
            physics_rate = "inherited pass"
        else:
            physics_rate = ratio_text(probe_pass, len(probe_rows)) if probe_rows else "N/A"
        output.append({
            "scene_id": scene_id,
            "label": label,
            "g1_planned_or_launched": len(condition_rows),
            "g1_launched": len(launched),
            "runtime_valid": len(valid),
            "clearance_evaluable": len(evaluable),
            "judgment_agreement": "N/A (control)" if scene_id == "EMPTY_CONTROL" else ratio_text(correct, len(evaluable)),
            "physics_probe_pass_rate": physics_rate,
            "independent_confirmation_count": len(independent),
            "independent_confirmation_valid_count": len(independent_valid),
            "independent_confirmation_ids": ",".join(str(row["trial_id"]) for row in independent),
            "execution_note": note,
        })
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    plans: list[dict[str, Any]] = []
    for manifest_name in MANIFEST_NAMES:
        manifest = read_json(MANIFEST.parent / manifest_name, {}) or {}
        plans.extend(manifest.get("trials", []))
    rows = [trial_row(plan) for plan in plans]
    fields = list(rows[0].keys()) if rows else ["trial_id"]
    write_csv(RESEARCH / "TRIAL_MANIFEST.csv", rows, fields)
    write_csv(RESEARCH / "JUDGMENT_VS_CLEARANCE.csv", rows, [
        "trial_id", "condition_id", "probe_side", "probe_depth_level", "structural_topology", "gate_status", "runtime_valid", "local_response", "upper_action", "contact_confirmed", "actual_hand_forward_m", "target_hand_forward_estimate_m", "continued_forward_after_contact_m", "contact_to_retract_s", "curtain_deformation_m", "deformation_to_hand_ratio", "clearance", "pointgoal", "task_success", "physical_branch_pass", "safe_recovery", "max_effort_ratio", "max_upper_tracking_rms_rad", "judgment_vs_clearance", "limitation_category"
    ])
    write_csv(RESEARCH / "LEFT_RIGHT_RESULTS.csv", sorted(rows, key=lambda item: (str(item["condition_id"]), str(item["probe_depth_level"]), str(item["probe_side"]))), fields)

    pair_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row["condition_id"], row["probe_depth_level"], row["seed"])
        grouped[key][str(row["probe_side"])] = row
    for key, sides in sorted(grouped.items(), key=lambda item: str(item[0])):
        left = sides.get("left", {})
        right = sides.get("right", {})
        pair_rows.append({
            "condition_id": key[0], "probe_depth_level": key[1], "seed": key[2],
            "left_trial_id": left.get("trial_id"), "right_trial_id": right.get("trial_id"),
            "left_runtime_valid": left.get("runtime_valid"), "right_runtime_valid": right.get("runtime_valid"),
            "left_local_response": left.get("local_response"), "right_local_response": right.get("local_response"),
            "left_actual_hand_forward_m": left.get("actual_hand_forward_m"), "right_actual_hand_forward_m": right.get("actual_hand_forward_m"),
            "left_curtain_deformation_m": left.get("curtain_deformation_m"), "right_curtain_deformation_m": right.get("curtain_deformation_m"),
            "left_clearance": left.get("clearance"), "right_clearance": right.get("clearance"),
            "left_pointgoal": left.get("pointgoal"), "right_pointgoal": right.get("pointgoal"),
            "matched_outcome": left.get("judgment_vs_clearance") == right.get("judgment_vs_clearance") if left and right else False,
        })
    write_csv(RESEARCH / "MATCHED_FAILURE_PAIRS.csv", pair_rows, list(pair_rows[0].keys()) if pair_rows else ["condition_id"])

    physics = physics_rows()
    physics_fields = [
        "gate_id", "mode", "scenario", "status", "physics_status", "contact_status",
        "structural_topology", "requested_fixed_count", "actual_fixed_count",
        "attachment_vertex_set_exact", "max_fixed_drift_m", "max_displacement_m",
        "max_speed_mps", "max_neighbor_stretch_ratio", "min_hidden_wall_face_gap_m",
        "max_nodes_at_or_beyond_hidden_wall_front", "model_contract", "real_pvc_claim",
        "summary_path",
    ]
    write_csv(RESEARCH / "PHYSICS_GATE_REPORT.csv", physics, physics_fields)
    scene_rows = scene_summary(rows, physics)
    write_csv(RESEARCH / "SCENE_RESULT_SUMMARY.csv", scene_rows, list(scene_rows[0].keys()) if scene_rows else ["scene_id"])
    formal_gate_ids = {
        "A0_TOP_EDGE_ONLY", "A1_TOP_EDGE_BOTTOM_TWO_CORNERS", "A2_FOUR_CORNERS_ONLY",
        "A3_FULL_PERIMETER_FIXED_V2", "B_D225_NEW", "B_T003_NEW", "C_GAP05", "C_GAP10", "C_GAP20",
    }
    formal_probe_rows = [item for item in physics if item.get("gate_id") in formal_gate_ids and item.get("mode") == "probe"]
    formal_probe_pass = sum(item.get("status") == "PASS_PHYSICS_SANITY" for item in formal_probe_rows)
    formal_probe_rate = ratio_text(formal_probe_pass, len(formal_probe_rows))
    write_json(OUTPUT / "stats.json", {
        "schema": "lift-the-curtain.round4070-structural-probe-discovery-stats.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "planned_count": len(rows),
        "launched_count": sum(bool(item["launched"]) for item in rows),
        "runtime_valid_count": sum(bool(item["runtime_valid"]) for item in rows),
        "physics_summary_count": len(physics),
        "by_limitation": {key: sum(item["limitation_category"] == key for item in rows) for key in sorted({item["limitation_category"] for item in rows})},
        "by_depth": {key: sum(item["probe_depth_level"] == key for item in rows) for key in sorted({str(item["probe_depth_level"]) for item in rows})},
    })

    launched = [row for row in rows if row["launched"]]
    integrity_rows = []
    required = ("trial_metadata.json", "command.json", "summary.json", "events.json", "telemetry.csv", "deployment_observation.csv", "resource_summary.json")
    for row in rows:
        path = TRIALS / str(row["trial_id"])
        missing = [name for name in required if not (path / name).is_file()] if row["launched"] else []
        deployment = []
        deployment_path = path / "deployment_observation.csv"
        if deployment_path.is_file():
            with deployment_path.open(encoding="utf-8-sig", newline="") as handle:
                deployment = list(csv.DictReader(handle))
        forbidden = sorted(header for header in (set(deployment[0]) if deployment else set()) if any(token in header.lower() for token in ("evaluation_", "contact_truth", "scene_name", "object_kind")))
        status = "PASS" if not missing and not forbidden else "RUNTIME_INVALID_NOT_ASSESSABLE" if not row["runtime_valid"] and not forbidden else "FAIL"
        integrity_rows.append({"trial_id": row["trial_id"], "launched": row["launched"], "runtime_valid": row["runtime_valid"], "missing_required": "|".join(missing), "deployment_truth_leakage_headers": "|".join(forbidden), "status": status})
    write_csv(RESEARCH / "INTEGRITY.csv", integrity_rows, list(integrity_rows[0].keys()) if integrity_rows else ["trial_id"])
    write_json(RESEARCH / "integrity_report.json", {
        "schema": "lift-the-curtain.round4070-structural-probe-discovery-integrity.v1",
        "status": "FAIL" if any(row["status"] == "FAIL" for row in integrity_rows) else "PASS_WITH_EXECUTION_INVALID" if any(row["status"] == "RUNTIME_INVALID_NOT_ASSESSABLE" for row in integrity_rows) else "PASS",
        "runtime_invalid_trial_count": sum(row["status"] == "RUNTIME_INVALID_NOT_ASSESSABLE" for row in integrity_rows),
        "deployment_truth_leakage_detected": any(bool(row["deployment_truth_leakage_headers"]) for row in integrity_rows),
        "rows": integrity_rows,
    })

    hash_rows = []
    for path in sorted(OUTPUT.rglob("*")):
        if not path.is_file() or path.name in {"ARTIFACT_HASHES.csv"}:
            continue
        if path.suffix.lower() not in {".json", ".csv", ".npz", ".png", ".md"}:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hash_rows.append({"path": str(path.relative_to(REPO)), "size_bytes": path.stat().st_size, "sha256": digest})
    write_csv(RESEARCH / "ARTIFACT_HASHES.csv", hash_rows, ["path", "size_bytes", "sha256"])

    depth_rows = [row for row in rows if row["group"] in {"EMPTY_CONTROL", "RIGID_CONTROL", "FABRIC_BASELINE"}]
    depth_md = ["# 探测深度验证", "", "实际手部 FK 位移是执行量；名义 D0/D1/D2/D3 增量不是力测量。", "", markdown_table(depth_rows, ["trial_id", "group", "probe_side", "probe_depth_level", "actual_hand_forward_m", "target_hand_forward_estimate_m", "max_effort_ratio", "max_upper_tracking_rms_rad", "runtime_valid", "status", "limitation_category"])]
    (RESEARCH / "PROBE_DEPTH_VALIDATION.md").write_text("\n".join(depth_md) + "\n", encoding="utf-8")

    physics_md = [
        "# 物理门报告", "",
        "物理门只决定场景是否可用于后续 G1，不是可通行结论。",
        "旧 A3 与 smoke 条目保留用于审计和修复链路；它们不应被解读为新的 G1 条件。",
        "",
        markdown_table(physics, ["gate_id", "mode", "scenario", "status", "physics_status", "contact_status", "structural_topology", "max_displacement_m", "max_speed_mps", "max_neighbor_stretch_ratio", "min_hidden_wall_face_gap_m", "model_contract", "real_pvc_claim"]),
    ]
    (RESEARCH / "PHYSICS_GATE_REPORT.md").write_text("\n".join(physics_md) + "\n", encoding="utf-8")

    resource_paths = list(OUTPUT.rglob("resource_summary.json"))
    resource_rows = [read_json(path, {}) or {} for path in resource_paths]
    gpu_peak = max((finite(row.get("gpu_memory_peak_mb")) or 0.0 for row in resource_rows), default=0.0)
    private_peak = max((finite(row.get("process_private_peak_mb")) or 0.0 for row in resource_rows), default=0.0)
    (RESEARCH / "GPU_AND_SOLVER_AUDIT.md").write_text(
        "# GPU 与求解器审计\n\n"
        f"- 资源摘要数：{len(resource_rows)}\n"
        f"- GPU 峰值：{gpu_peak:.1f} MB（硬上限 6000 MB）\n"
        f"- 进程 private 峰值：{private_peak:.1f} MB（硬上限 12000 MB）\n"
        "- G1 physics dt：0.005 s；控制 decimation：4；deployment telemetry：0.005 s。\n"
        "- 单片 physics gate 使用 720 steps、0.005 s dt；surface solver position iterations 沿用 formal config（8）。\n"
        "- effort 字段是归一化 actuator effort proxy，不是绝对接触力；物理节点、接触力和隐藏墙几何只进入 evaluation artifact。\n"
        "- watchdog 在 FINALIZE 后常见 CLEANUP_TIMEOUT_AFTER_FINALIZE_30s；若 summary 已写入且 `stage_log.last_stage_status=PASS`，按完成的 artifact 处理。\n",
        encoding="utf-8",
    )

    category_counts = defaultdict(int)
    for row in rows:
        category_counts[str(row["limitation_category"])] += 1
    limitations = [
        "# 科学限制", "",
        "本轮不把 runtime invalid 当作物理负例，也不把 physics-invalid 条带当作 PVC 结论。", "",
        *[f"- `{key}`：{value} 条。" for key, value in sorted(category_counts.items())],
        "", "具体试验级别的分类见 `JUDGMENT_VS_CLEARANCE.csv`。",
        "- `ARTICULATED_STRIP_SURROGATE` 不是真实 PVC；当前多片 surface-deformable 构造的无探测门失败，只能归为 `PHYSICS_INVALID`。",
        "- D0/D1 的深度值是关节参考调度，实际前向位移由 FK 读取；不能将其解释为绝对力、刚度或穿越距离。",
        "- 响应后未完成 bounded advance 的结果受观察时域和导航终点双重影响，属于未证实，不直接下 BLOCKED 结论。",
        "- seed 4071 的 A1 D2 左右独立确认各在 1200 s 冻结上限内超时，属于 EXECUTION_INVALID；没有把半程 rollout 当成 A1 的物理标签。",
    ]
    (RESEARCH / "SCIENTIFIC_LIMITATIONS.md").write_text("\n".join(limitations) + "\n", encoding="utf-8")

    valid = [row for row in rows if row["runtime_valid"]]
    non_control_evaluable = [row for row in valid if row["group"] != "EMPTY_CONTROL" and row["clearance"] is not None]
    non_control_correct = sum((row["local_response"] == "RESPONSIVE") == bool(row["clearance"]) for row in non_control_evaluable)
    overall_agreement = ratio_text(non_control_correct, len(non_control_evaluable))
    non_control_valid = [row for row in valid if row["group"] != "EMPTY_CONTROL"]
    independent_rows = [row for row in rows if str(row.get("stage", "")).startswith("independent_confirmatory")]
    scene_rows = scene_summary(rows, physics)
    final = [
        "# Round 4070-C 最终报告", "",
        f"生成时间：{datetime.now().astimezone().isoformat()}", "",
        "## 结论摘要", "",
        f"- 冻结候选、同 seed 确认附录和 seed 4071 独立确认共 {len(rows)} 条 G1；已启动：{sum(bool(row['launched']) for row in rows)} 条；runtime-valid：{sum(bool(row['runtime_valid']) for row in rows)} 条；低于 48 条硬上限。",
        f"- 物理 summary：{len(physics)} 条（含 smoke、旧 A3 修复前审计和 surrogate 条目）；physics-invalid/blocked 条件保留在 `PHYSICS_GATE_REPORT.csv`，未进入可通行负样本。",
        f"- 正式 P0 physics probe gate（A0/A1/A2/A3-V2、D225、T003、GAP05/10/20）通过率：{formal_probe_rate}；C_GAP20 是唯一失败的正式 probe gate。",
        f"- valid 试验中 RESPONSIVE：{sum(row['local_response'] == 'RESPONSIVE' for row in valid)} 条；全部 valid 的 candidate clearance=true：{sum(row['clearance'] is True for row in valid)} 条；排除空场控制后，带帘面条件 candidate clearance=true：{sum(row['clearance'] is True for row in non_control_valid)} 条；pointgoal=true：{sum(row['pointgoal'] is True for row in valid)} 条。",
        f"- 非控制、runtime-valid 且 clearance 可评估样本的 local-response 描述性 agreement：{overall_agreement}；这不是跨场景泛化准确率。",
        "- 空场 D0/D1 左右控制均通过，证明深度调度能改变实际 FK 前向位移并保持安全监控。",
        "- episode PASS 不等于穿过候选面：必须分别读取 `candidate_clearance_verified`、`pointgoal_success` 与 `task_success`。部分 episode 经安全撤回/绕行抵达 pointgoal，candidate clearance 仍为 false。",
        "- 普通布料 D0/D1 左右均出现响应和可测帘面形变；D1 右侧 clearance=true 但 pointgoal=false，显示动作窗口和任务结论并不等价。",
        f"- 排除空场控制后仅 {sum(row['clearance'] is True for row in non_control_valid)}/{len(non_control_valid)} 条 runtime-valid G1 确认 candidate clearance，且该条未完成 pointgoal/task success；本轮没有端到端帘面穿越成功证据。",
        "- A1/A2/A3 两侧均完成 protocol 运行；其中 episode PASS 多数为 safe recovery/绕行到 goal，不可据此宣称跨越帘面。",
        "- D225 左侧因 `NO_PATH_THROUGH_EXPLICIT_ZERO_CELLS` 进入 BLOCKED，缺少完整运行产物，归 `EXECUTION_INVALID`；右侧有效但未确认 clearance。",
        "- 空场 D3 左侧 episode 未完成、右侧通过；仅有一侧差异，不能把 D3 认定为安全有效的推荐深度。A1 D2 左右的任务级 protocol 均通过，但 candidate clearance 仍需单独看列值。",
        "- C_GAP20 物理门在探测时出现前表面穿越，已阻止 G1；安装间隙与运行时最小间距分别记录。",
        "- 多条带 surrogate 无探测 physics gate 失败，未启动 probe/G1，不宣称真实 PVC。",
        f"- seed 4071 独立确认实际启动 {len(independent_rows)} 条，runtime-valid {sum(bool(row['runtime_valid']) for row in independent_rows)} 条；其中 A1 D2 左右各 1 条在 1200 s 冻结上限内未完成，归 `EXECUTION_INVALID`，结果见下表。",
        "",
        "## 场景级计数与判断 agreement",
        "`judgment_agreement` 只在 runtime-valid 且 `candidate_clearance_verified` 已有布尔值的样本上计算：RESPONSIVE 视为正判断，BLOCKED/UNCERTAIN 视为负判断，与 candidate clearance 做描述性一致率；空场控制不计入。该指标不是跨场景泛化准确率，也不把 execution-invalid 当作负例。",
        "",
        markdown_table(scene_rows, ["scene_id", "g1_launched", "runtime_valid", "physics_probe_pass_rate", "judgment_agreement", "independent_confirmation_count", "independent_confirmation_valid_count", "execution_note"]),
        "",
        "## 独立 seed 确认",
        "本节仅指 seed=4071 的新 seed 运行；此前 seed=4070 的 4 条确认附录单独保留，不冒充独立确认。",
        "",
        markdown_table(independent_rows, ["trial_id", "condition_id", "probe_side", "probe_depth_level", "runtime_valid", "status", "local_response", "clearance", "pointgoal", "task_success", "limitation_category"]),
        "", "## 结果文件", "",
        "- `JUDGMENT_VS_CLEARANCE.csv`：局部响应、上层动作、clearance、pointgoal 和限制类别。",
        "- `MATCHED_FAILURE_PAIRS.csv`：同条件同 seed 左右配对。",
        "- `LEFT_RIGHT_RESULTS.csv`：按侧展开的完整结果。",
        "- `SCENE_RESULT_SUMMARY.csv`：逐场启动数、有效数、物理 probe 通过率、描述性判断 agreement、独立确认与未执行原因。",
        "- `INTEGRITY.csv` / `integrity_report.json`：必需 artifact 与 deployment truth leakage 检查。",
        "- `ARTIFACT_HASHES.csv`：C 原始 JSON/CSV/NPZ/PNG/Markdown 的 SHA-256 与大小。",
        "", "## 分类原则", "",
        "本轮使用 `OBSERVATION_HORIZON_LIMITATION`、`FEATURE_UTILIZATION_LIMITATION`、`SPATIAL_CONSTRAINT_LIMITATION`、`HIGH_RESISTANCE_AMBIGUITY`、`TEMPORAL_RESPONSE_AMBIGUITY`、`PHYSICS_INVALID`、`EXECUTION_INVALID` 和 `INSUFFICIENT_EVIDENCE`；分类只基于已落盘的运行证据，不把单次失败泛化到所有动作。",
    ]
    (RESEARCH / "FINAL_REPORT_ZH.md").write_text("\n".join(final) + "\n", encoding="utf-8")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            usable = [row for row in rows if row["actual_hand_forward_m"] is not None]
            if usable:
                figure, axis = plt.subplots(figsize=(8, 4.5))
                groups = sorted({str(row["probe_depth_level"]) for row in usable})
                for group in groups:
                    subset = [row for row in usable if str(row["probe_depth_level"]) == group]
                    axis.scatter([row["actual_hand_forward_m"] for row in subset], [row["curtain_deformation_m"] or 0.0 for row in subset], label=group)
                axis.set_xlabel("actual hand forward displacement (m)")
                axis.set_ylabel("curtain deformation (m)")
                axis.legend()
                axis.grid(True, alpha=0.25)
                figure.tight_layout()
                (RESEARCH / "plots").mkdir(parents=True, exist_ok=True)
                figure.savefig(RESEARCH / "plots" / "probe_depth_vs_deformation.png", dpi=140)
                plt.close(figure)
        except Exception as exc:
            write_json(RESEARCH / "plots" / "plot_error.json", {"error": f"{type(exc).__name__}: {exc}"})

    print(json.dumps({"planned": len(rows), "launched": len(launched), "valid": len(valid), "physics": len(physics), "research": str(RESEARCH)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
