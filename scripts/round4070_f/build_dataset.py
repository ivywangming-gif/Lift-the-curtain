"""Read-only E audit, causal episode export, and frozen-rule reproduction.

No simulator initialization; never changes historical artifacts. Labels and
physical configuration identifiers live in the manifest, never in seq.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs/round4070_learning_traversability/dataset"
REPORT = ROOT / "research/round4070_learning_traversability"
from lift_curtain.control.falcon_contract import OFFICIAL_POLICY_JOINT_ORDER as JOINTS
from lift_curtain.control.falcon_contract import ISAACLAB_TO_OFFICIAL, OFFICIAL_TO_ISAACLAB
from lift_curtain.control.deployment_observation import ProprioceptiveThresholds


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def dump(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rotation(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr], [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])


def vector(row, key, n):
    a = np.asarray(json.loads(row[key]), dtype=np.float64)
    assert a.shape == (n,) and np.isfinite(a).all(), key
    return a


def export_episode(item):
    directory = item["path"]
    rows = csv_rows(directory / "deployment_observation.csv")
    telemetry = csv_rows(directory / "telemetry.csv")
    # Exact timestamp + phase join; never nearest/future-match an observation.
    lookup = {(r["phase"], r["time_s"]): r for r in telemetry}
    assert len(lookup) == len(telemetry), "duplicate telemetry key"
    selected = [r for r in rows if r["phase"].startswith("physical_probe") or r["phase"] in {"probe", "contact_hold"}]
    t = np.array([float(r["time_s"]) for r in selected])
    assert len(t) > 1 and np.all(np.diff(t) > 0)
    names, arrays, blocks = [], [], {}
    raw = {}

    def add(name, value, suffixes):
        start = len(names)
        arrays.append(value)
        names.extend(name + "/" + str(s) for s in suffixes)
        blocks[name] = list(range(start, len(names)))

    for name, key, unit in (("q", "q_actual", "rad"), ("qdot", "qdot", "rad_s"), ("q_cmd", "q_cmd", "rad"), ("effort", "motor_effort", "Nm"), ("policy_action_logged", "previous_action", "normalized")):
        a = np.stack([vector(r, "deployment_"+key+"_json", 29) for r in selected])
        add(name, a, JOINTS)
        raw[name] = a.astype(np.float32)
    joined = [lookup[(r["phase"], r["time_s"])] for r in selected]
    rpy = np.array([[float(r[k]) for k in ("roll", "pitch", "yaw")] for r in joined])
    rot = np.stack([rotation(r) for r in rpy])
    base = np.stack([vector(r, "deployment_base_position_w_json", 3) for r in selected])
    raw["base_position_w_audit_only"] = base
    for side in ("left", "right"):
        hand = np.stack([vector(r, "deployment_"+side+"_hand_fk_position_w_json", 3) for r in selected])
        # World position is retained for audit only. Learning gets robot-relative pose.
        body = np.einsum("tji,tj->ti", rot, hand-base)
        progress = (hand-hand[0]) @ rot[0]
        raw[side+"_hand_position_w_audit_only"] = hand
        add(side+"_hand_fk_b", body, "xyz")
        add(side+"_hand_delta_initial_b", progress, "xyz")
    basev = np.stack([vector(r, "deployment_base_velocity_b_json", 3) for r in selected])
    add("base_velocity_b", basev, "xyz")
    add("imu_rpy", rpy, ("roll", "pitch", "yaw"))
    add("gyro_z_b", np.array([[float(r["measured_wz"])] for r in joined]), ("z",))
    add("elapsed", (t-t[0])[:, None], ("s",))
    add("probe_side", np.full((len(t), 1), int(item["plan"]["probe_side"] == "right")), ("right",))
    seq = np.concatenate(arrays, axis=1).astype(np.float32)
    assert np.isfinite(seq).all()
    # NaNs are missing fields, not model inputs or synthetic zero IMU.
    raw["imu_angular_velocity_b"] = np.column_stack([np.full((len(t), 2), np.nan), [float(r["measured_wz"]) for r in joined]])
    raw["imu_linear_acceleration_b"] = np.full((len(t), 3), np.nan)
    raw["previous_logged_action"] = np.vstack([np.full((1, 29), np.nan), raw["policy_action_logged"][:-1]])
    raw["previous_logged_action_time"] = np.r_[np.nan, t[:-1]]
    phases = np.array([r["phase"] for r in selected])
    effort_sources = sorted({r["deployment_effort_source"] for r in selected})
    assert effort_sources == ["applied_torque"], effort_sources
    trial = item["plan"]["trial_id"]
    path = OUT / "episodes" / (trial + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, seq=seq, time=t, phase=phases, feature_names=np.array(names), effort_source=np.array(effort_sources), **raw)
    return {"npz_path": path.relative_to(ROOT).as_posix(), "sample_count": len(t), "first_time_s": float(t[0]), "last_time_s": float(t[-1]), "duration_s": float(t[-1]-t[0]), "dt_median_s": float(np.median(np.diff(t))), "dt_min_s": float(np.diff(t).min()), "dt_max_s": float(np.diff(t).max()), "effort_sources": effort_sources, "alignment_valid": True, "source_sha256": {p.name: sha(p) for p in (directory / "deployment_observation.csv", directory / "telemetry.csv", directory / "summary.json")}}, names, blocks


def corrected_decisions(evaluator, item, baseline, baseline_features):
    h = item["history"]
    features = evaluator.feature_dict(h, baseline)
    t = np.array([float(r["time_s"]) for r in item["history_rows"]])
    # Left integration uses only recorded low-speed values and elapsed intervals.
    # No artificial extrapolation before first/after last sample.
    low = np.abs(np.array([o.hand_fk_velocity_w[0] for o in h])) <= 0.004
    features["hand_low_speed_duration_s"] = float(np.sum(np.diff(t)*low[:-1]))
    config = ProprioceptiveThresholds(**load(ROOT / "configs/proprioceptive_thresholds.json"))
    p, ratio, effort, stall = (features[k] for k in ("hand_forward_progress_m", "hand_progress_ratio", "effort_deviation", "hand_low_speed_duration_s"))
    decision = "UNCERTAIN"
    if len(h) >= config.minimum_samples:
        if p >= config.responsive_progress_m and (ratio is None or ratio >= config.responsive_progress_ratio_min):
            decision = "PASSABLE"
        elif (p <= config.blocked_progress_max_m or (ratio is not None and ratio <= config.blocked_progress_ratio_max)) and effort >= config.blocked_effort_deviation_min and stall >= config.blocked_stall_duration_s:
            decision = "BLOCKED"
    return {"E0_time_corrected": evaluator.e0_decision(features)[0], "E1_time_corrected": decision, "E2_time_corrected": evaluator.e2_decision(features, baseline_features)[0]}, features


def audit_older():
    output = []
    for campaign in ("g1_final4070", "round4070_problem_discovery", "round4070_failure_discovery", "round4070_structural_probe_discovery", "round4070_fixed10cm_judgment"):
        root = ROOT / "outputs" / campaign
        summaries = sorted(root.rglob("summary.json"))
        histories = sorted(root.rglob("deployment_observation.csv"))
        phases, dims, sources = collections.Counter(), collections.Counter(), collections.Counter()
        history_records = []
        for path in histories:
            rows = csv_rows(path)
            if not rows:
                continue
            counts = collections.Counter(r.get("phase", "") for r in rows)
            phases.update(counts)
            for r in rows:
                sources[r.get("deployment_effort_source", "missing")] += 1
            for k in ("deployment_q_actual_json", "deployment_qdot_json", "deployment_q_cmd_json"):
                if rows[0].get(k):
                    dims[k+":"+str(len(json.loads(rows[0][k])))] += 1
            history_records.append({"path": path.relative_to(ROOT).as_posix(), "rows": len(rows), "phases": dict(counts), "sha256": sha(path)})
        summary_records = []
        for path in summaries:
            s = load(path)
            summary_records.append({"path": path.relative_to(ROOT).as_posix(), "status": s.get("status"), "runtime_status": s.get("runtime_status"), "candidate_clearance_verified": s.get("candidate_clearance_verified"), "whole_body_candidate_clearance": s.get("whole_body_candidate_clearance"), "probe_depth_level": s.get("probe_depth_level"), "safe_recovery": s.get("safe_recovery"), "sha256": sha(path)})
        output.append({"campaign": campaign, "summary_files": len(summaries), "history_files": len(histories), "phase_row_counts": dict(phases), "vector_dims": dict(dims), "effort_sources": dict(sources), "histories": history_records, "summaries": summary_records, "learning_status": "AUDIT_ONLY_DIFFERENT_PROTOCOL_NOT_POOLED"})
    dump(OUT / "historical_inventory.json", output)
    return output


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("e_evaluator", ROOT / "scripts/round4070_fixed_probe_benchmark/evaluate_benchmark.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    _, plans = evaluator.load_manifest()
    records, baselines = evaluator.collect(plans, load(ROOT / "configs/round4070_fixed_probe_benchmark/PHYSICS_GATE_STATUS.json"))
    rows, audits = evaluator.evaluate(records, baselines)
    indexed = {r["trial_id"]: r for r in rows}
    baseline_features = {side: evaluator.feature_dict(h, h) for side, h in baselines.items()}
    episodes, timing = [], []
    for item in records:
        if not item["summary_exists"]:
            continue
        plan, trial = item["plan"], item["plan"]["trial_id"]
        r = indexed[trial]
        fields, names, blocks = export_episode(item)
        corrected, features = corrected_decisions(evaluator, item, baselines[plan["probe_side"]], baseline_features[plan["probe_side"]])
        r["decisions"].update(corrected)
        r["decisions"]["always_PASSABLE"] = "PASSABLE"
        r["decisions"]["always_BLOCKED"] = "BLOCKED"
        label = item["label"]
        control = label.startswith("CONTROL_")
        group = "EMPTY_CONTROL" if plan.get("diagnostic_no_obstacle") else str(plan["condition_id"])
        # Collapse semantically identical no-obstacle controls across gap aliases.
        if plan.get("condition_family") == "matched_no_wall":
            group = "EMPTY_CONTROL"
        entry = {"trial_id": trial, "campaign": "4070-E", "label": label, "y": 1 if label in {"PASSABLE", "CONTROL_PASSABLE"} else 0 if label in {"BLOCKED", "CONTROL_BLOCKED"} else -1, "is_control": control, "group_id": group, "condition_id": plan["condition_id"], "condition_family": plan["condition_family"], "side": plan["probe_side"], "seed": plan["seed"], "source_stage": plan["stage"], "runtime_valid": item["runtime_valid"], "probe_valid": item["probe_valid"], "label_valid": item["label_valid"], "label_source": item["label_reason"], "label_evidence": item["label_evidence"], "engineering_decisions": r["decisions"], "engineering_features_time_corrected": features, "physics_configuration_audit_only": plan, **fields}
        episodes.append(entry)
        timing.append({"trial_id": trial, **{k: fields[k] for k in ("sample_count", "duration_s", "dt_median_s", "dt_min_s", "dt_max_s")}, "old_stall_s": r["features"]["hand_low_speed_duration_s"], "corrected_stall_s": features["hand_low_speed_duration_s"]})
    assert len(episodes) == 104
    labels = collections.Counter(e["label"] for e in episodes)
    # Name-order contract is a bijection and exported q/qdot/q_cmd have 29 finite values.
    assert len(JOINTS) == 29 and len(set(JOINTS)) == 29
    assert np.array_equal(np.arange(29)[list(OFFICIAL_TO_ISAACLAB)][list(ISAACLAB_TO_OFFICIAL)], np.arange(29))
    arm = [i for name in ("q", "qdot", "q_cmd", "effort", "policy_action_logged") for i in blocks[name][15:]]
    arm += [i for name in blocks if "hand_" in name or name in {"elapsed", "probe_side"} for i in blocks[name]]
    body = sorted(set(range(len(names)))-set(arm))
    units = {"q": "rad", "qdot": "rad/s", "q_cmd": "rad", "effort": "N.m (simulator applied_torque proxy)", "policy_action_logged": "normalized policy action", "left_hand_fk_b": "m", "right_hand_fk_b": "m", "left_hand_delta_initial_b": "m", "right_hand_delta_initial_b": "m", "base_velocity_b": "m/s", "imu_rpy": "rad", "gyro_z_b": "rad/s", "elapsed": "s", "probe_side": "binary right=1"}
    manifest = {"schema": "round4070_f_causal_episodes.v1", "path_base": "repository_root", "feature_names": names, "feature_blocks": blocks, "feature_units_by_block": units, "arm_indices": arm, "body_indices": body, "joint_order": list(JOINTS), "labels": dict(labels), "episodes": episodes, "forbidden_learning_inputs": ["condition_id", "group_id", "material", "anchor", "hidden_wall", "label_evidence", "physics_configuration_audit_only", "map", "goal", "contact_truth", "future_feedback", "*_audit_only"], "missing_channels": ["gyro_x_b", "gyro_y_b", "imu_acceleration_xyz", "true_previous_control_action"], "same_legal_window": "physical_probe and physical_probe_hold only; no retract/advance/navigation", "self_supervision": "train-group episodes only, including UNKNOWN; never holdout group histories", "split_policy": "group_id immutable across seed, side, pose and source stage; controls separate; no window random split"}
    dump(OUT / "manifest.json", manifest)
    dump(OUT / "evaluator_reproduction.json", rows)
    dump(OUT / "label_evidence.json", audits)
    save_csv(OUT / "timing_audit.csv", timing)
    methods = ("always_PASSABLE", "always_BLOCKED", "E0", "E1", "E2", "E0_time_corrected", "E1_time_corrected", "E2_time_corrected")
    metrics = [evaluator.metric(rows, method) for method in methods]
    save_csv(REPORT / "BASELINE_RESULTS.csv", metrics)
    old = audit_older()
    scene_rows = []
    for group in sorted({e["group_id"] for e in episodes}):
        selected = [e for e in episodes if e["group_id"] == group]
        scene_rows.append({"group_id": group, "n": len(selected), **{label: sum(e["label"] == label for e in selected) for label in ("PASSABLE", "BLOCKED", "UNKNOWN", "CONTROL_PASSABLE", "CONTROL_BLOCKED")}, "left": sum(e["side"] == "left" for e in selected), "right": sum(e["side"] == "right" for e in selected)})
    save_csv(OUT / "scene_counts.csv", scene_rows)
    print(json.dumps({"episodes": len(episodes), "labels": dict(labels), "features": len(names), "samples": [min(e["sample_count"] for e in episodes), max(e["sample_count"] for e in episodes)], "dt_medians": sorted({round(e["dt_median_s"], 6) for e in episodes}), "metrics": metrics, "older": [{k:v for k,v in r.items() if k in {"campaign", "history_files", "summary_files"}} for r in old]}, indent=2))


def integrate_supplementary(label_path):
    """Build a separate E+F view; failed runs remain audit entries, never negatives."""
    global OUT
    base_manifest = load(OUT / "manifest.json")
    OUT = ROOT / "outputs/round4070_learning_traversability/dataset_phase_c"
    OUT.mkdir(parents=True, exist_ok=True)
    label_path = Path(label_path)
    payload = load(label_path)
    if isinstance(payload, list):
        supplements = payload
    else:
        key = next((k for k in ("trials", "rows", "labels", "records", "results") if isinstance(payload.get(k), list)), None)
        if key is None:
            raise ValueError("Supplementary labels must be a list or contain trials/rows/labels/records/results list")
        supplements = payload[key]
    spec = importlib.util.spec_from_file_location("e_evaluator", ROOT / "scripts/round4070_fixed_probe_benchmark/evaluate_benchmark.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    _, plans = evaluator.load_manifest()
    _, baselines = evaluator.collect(plans, load(ROOT / "configs/round4070_fixed_probe_benchmark/PHYSICS_GATE_STATUS.json"))
    bf = {side: evaluator.feature_dict(h, h) for side, h in baselines.items()}
    episodes = list(base_manifest["episodes"])
    audits = []
    for record in supplements:
        trial = record["trial_id"]
        directory = Path(record.get("path", ROOT / "outputs/round4070_learning_traversability/physics/trials" / trial))
        if not directory.is_absolute():
            directory = ROOT / directory
        audit = {"trial_id": trial, "source_record": record, "included": False}
        if not all((directory / name).is_file() for name in ("summary.json", "deployment_observation.csv", "telemetry.csv", "trial_metadata.json")):
            audit["exclusion"] = "MISSING_COMPLETE_RAW_LOGS"
            audits.append(audit)
            continue
        if not evaluator.boolean(record.get("runtime_valid")) or not evaluator.boolean(record.get("probe_valid")):
            audit["exclusion"] = "RUNTIME_OR_PROBE_INVALID"
            audits.append(audit)
            continue
        metadata, summary = load(directory / "trial_metadata.json"), load(directory / "summary.json")
        plan = metadata.get("plan", metadata)
        assert plan["trial_id"] == trial and plan["probe_side"] == record["probe_side"] and int(plan["seed"]) == int(record["seed"]), "F label/metadata identity mismatch"
        assert "equivalent_e_condition_id" in plan and "curtain_length" in plan, "F grouping requires explicit, verified metadata (no inferred condition aliases)"
        assert plan["equivalent_e_condition_id"] in {e["condition_id"] for e in base_manifest["episodes"]}, "F equivalent condition not in audited E configurations"
        plan = {**plan, "trial_id": trial, "probe_side": record["probe_side"]}
        protocol_ok, protocol_detail = evaluator.protocol_audit(plan, summary, metadata)
        checked_label, checked_valid, checked_reason, _, checked_evidence = evaluator.independent_label(plan, summary, metadata.get("physics_gate_status", "MISSING"), protocol_ok, protocol_detail)
        assert checked_label == record["label"] and checked_valid == evaluator.boolean(record["label_valid"]), "F label changed on independent source-data reproduction"
        h, hr = evaluator.load_history(directory / "deployment_observation.csv")
        item = {"path": directory, "plan": plan, "history": h, "history_rows": hr}
        fields, names, blocks = export_episode(item)
        assert names == base_manifest["feature_names"], "F sensor schema mismatch"
        side = record["probe_side"]
        features = evaluator.feature_dict(h, baselines[side])
        e1, _ = evaluator.e1_decision(h, baselines[side])
        decisions = {"E0": evaluator.e0_decision(features)[0], "E1": {"RESPONSIVE": "PASSABLE", "BLOCKED": "BLOCKED"}.get(e1, "UNCERTAIN"), "E2": evaluator.e2_decision(features, bf[side])[0]}
        corrected, features = corrected_decisions(evaluator, item, baselines[side], bf[side])
        decisions.update(corrected)
        decisions.update(always_PASSABLE="PASSABLE", always_BLOCKED="BLOCKED")
        label = str(record["label"])
        assert label in {"PASSABLE", "BLOCKED", "UNKNOWN", "CONTROL_PASSABLE", "CONTROL_BLOCKED"}
        if label != "UNKNOWN":
            assert evaluator.boolean(record["label_valid"]), "non-UNKNOWN label must be independently valid"
            assert evaluator.boolean(record.get("protocol_match", True)), "supervised F label protocol mismatch"
        condition = plan["equivalent_e_condition_id"]
        length = float(plan["curtain_length"])
        group = condition if abs(length-2.0) < 1e-8 else f"{condition}|length={length:.4g}"
        if record.get("condition_family") == "matched_no_wall" or plan.get("diagnostic_no_obstacle"):
            group = "EMPTY_CONTROL"
        entry = {"trial_id": trial, "campaign": "4070-F", "label": label, "y": 1 if label in {"PASSABLE", "CONTROL_PASSABLE"} else 0 if label in {"BLOCKED", "CONTROL_BLOCKED"} else -1, "is_control": label.startswith("CONTROL_"), "group_id": group, "condition_family_group": condition, "condition_id": condition, "condition_family": record.get("condition_family", plan.get("condition_family")), "side": side, "seed": record["seed"], "source_stage": plan.get("stage", plan.get("phase", "F_SCREENING")), "runtime_valid": True, "probe_valid": True, "label_valid": checked_valid, "label_source": checked_reason, "label_evidence": checked_evidence, "grouping_source": "explicit equivalent_e_condition_id and curtain_length in trial_metadata.json", "trial_metadata_sha256": sha(directory / "trial_metadata.json"), "engineering_decisions": decisions, "engineering_features_time_corrected": features, "physics_configuration_audit_only": plan, **fields}
        assert trial not in {e["trial_id"] for e in episodes}
        episodes.append(entry)
        audit.update(included=True, label=label, group_id=group)
        audits.append(audit)
    base_manifest.update(episodes=episodes, labels=dict(collections.Counter(e["label"] for e in episodes)), source_e_manifest_sha256=sha(ROOT / "outputs/round4070_learning_traversability/dataset/manifest.json"), supplementary_labels_sha256=sha(label_path), supplementary_source=str(label_path), phase_c_note="E input files referenced unchanged; F raw logs independently exported. F screening is not independent confirmation. Group merges equivalent E/F configurations; new length has distinct full-config group but same structural family.")
    dump(OUT / "manifest.json", base_manifest)
    dump(OUT / "supplementary_audit.json", audits)
    save_csv(OUT / "episode_counts.csv", [{"campaign": campaign, "label": label, "n": sum(e["campaign"] == campaign and e["label"] == label for e in episodes)} for campaign in ("4070-E", "4070-F") for label in ("PASSABLE", "BLOCKED", "UNKNOWN", "CONTROL_PASSABLE", "CONTROL_BLOCKED")])
    print(json.dumps({"phase_c_episodes": len(episodes), "supplementary_records": len(audits), "supplementary_included": sum(a["included"] for a in audits), "labels": base_manifest["labels"], "manifest": str(OUT / "manifest.json")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--supplementary-labels", type=Path)
    args = parser.parse_args()
    if args.supplementary_labels:
        integrate_supplementary(args.supplementary_labels)
    else:
        main()
