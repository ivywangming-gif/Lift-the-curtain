"""Analyze Round 4070-B outputs and write new research artifacts only."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO / "configs" / "round4070_failure_discovery" / "PREREGISTERED_MANIFEST.json"
DEFAULT_RESEARCH = REPO / "research" / "round4070_failure_discovery"
DEFAULT_TRIAL_ROOT = REPO / "outputs" / "round4070_failure_discovery" / "trials"
DEFAULT_PHYSICS_ROOT = REPO / "outputs" / "round4070_failure_discovery" / "physics"


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or sorted({key for row in rows for key in row}) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def vector(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return [float(item) for item in parsed]
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
    return []


def mean(values: Iterable[float | None]) -> float | None:
    usable = [item for item in values if item is not None and math.isfinite(item)]
    return sum(usable) / len(usable) if usable else None


def rms(values: Iterable[float]) -> float | None:
    usable = [float(item) for item in values if math.isfinite(float(item))]
    return math.sqrt(sum(item * item for item in usable) / len(usable)) if usable else None


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def phase_rows(rows: list[dict[str, str]], phases: set[str]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("phase") in phases]


def phase_features(rows: list[dict[str, str]], cutoff: str) -> dict[str, Any] | None:
    if cutoff == "T0":
        selected = phase_rows(rows, {"physical_preprobe_hold"})
    elif cutoff == "T1":
        selected = phase_rows(rows, {"physical_probe", "physical_probe_hold"})
    else:
        selected = [row for row in rows if "cautious_advance" in row.get("phase", "") or "responsive_traverse" in row.get("phase", "")]
    if not selected:
        return None
    times = [finite(row.get("time_s")) for row in selected]
    times = [value for value in times if value is not None]
    hand_positions = [vector(row.get("deployment_hand_fk_position_w_json")) for row in selected]
    base_positions = [vector(row.get("deployment_base_position_w_json")) for row in selected]
    hand_start = next((value for value in hand_positions if len(value) >= 3), [])
    hand_end = next((value for value in reversed(hand_positions) if len(value) >= 3), [])
    base_start = next((value for value in base_positions if len(value) >= 2), [])
    base_end = next((value for value in reversed(base_positions) if len(value) >= 2), [])
    tracking = [finite(row.get("upper_tracking_rms_rad")) for row in selected]
    efforts = [rms(vector(row.get("deployment_motor_effort_json"))) for row in selected]
    q_errors: list[float] = []
    velocities: list[float] = []
    for row in selected:
        actual = vector(row.get("deployment_q_actual_json"))
        command = vector(row.get("deployment_q_cmd_json"))
        if len(actual) == len(command) and len(actual) >= 29:
            error = rms([a - b for a, b in zip(command[15:], actual[15:])])
            if error is not None:
                q_errors.append(error)
        qdot = vector(row.get("deployment_qdot_json"))
        if len(qdot) >= 29:
            value = rms(qdot[15:])
            if value is not None:
                velocities.append(value)
    low_speed = 0.0
    low_speed_assessable = False
    for index, row in enumerate(selected[:-1]):
        now = finite(row.get("time_s"))
        later = finite(selected[index + 1].get("time_s"))
        velocity = vector(row.get("deployment_hand_fk_velocity_w_json"))
        if now is None or later is None or not velocity:
            continue
        low_speed_assessable = True
        if abs(velocity[0]) <= 0.004:
            low_speed += max(0.0, later - now)
    return {
        "cutoff": cutoff,
        "sample_count": len(selected),
        "duration_s": max(0.0, times[-1] - times[0]) if len(times) >= 2 else None,
        "hand_forward_delta_m": hand_end[0] - hand_start[0] if hand_start and hand_end else None,
        "base_delta_m": math.dist(base_start[:2], base_end[:2]) if len(base_start) >= 2 and len(base_end) >= 2 else None,
        "upper_tracking_rms_mean_rad": mean(tracking),
        "q_cmd_minus_q_upper_rms_mean_rad": mean(q_errors),
        "joint_velocity_upper_norm_mean": mean(velocities),
        "effort_rms_mean": mean(efforts),
        "hand_low_speed_duration_s": low_speed if low_speed_assessable else None,
        "command_vx_mean_mps": mean([finite(row.get("command_vx")) for row in selected]),
        "measured_planar_speed_mean_mps": mean([finite(row.get("measured_planar_speed")) for row in selected]),
    }


def series_features(rows: list[dict[str, str]], phases: set[str]) -> dict[str, list[float]]:
    selected = phase_rows(rows, phases)
    result = {"time_s": [], "tracking": [], "effort_deviation": [], "hand_progress": [], "base_disturbance": []}
    if not selected:
        return result
    first_hand = vector(selected[0].get("deployment_hand_fk_position_w_json"))
    first_base = vector(selected[0].get("deployment_base_position_w_json"))
    baseline_effort: list[float] = []
    for row in selected[: min(10, len(selected))]:
        baseline_effort.extend(vector(row.get("deployment_motor_effort_json")))
    baseline = mean(baseline_effort) or 0.0
    for row in selected:
        hand = vector(row.get("deployment_hand_fk_position_w_json"))
        base = vector(row.get("deployment_base_position_w_json"))
        effort = vector(row.get("deployment_motor_effort_json"))
        if not hand or not base:
            continue
        result["time_s"].append(finite(row.get("time_s")) or 0.0)
        result["tracking"].append(finite(row.get("upper_tracking_rms_rad")) or 0.0)
        result["effort_deviation"].append(rms([value - baseline for value in effort]) or 0.0)
        result["hand_progress"].append(hand[0] - first_hand[0] if first_hand else 0.0)
        result["base_disturbance"].append(math.dist(base[:2], first_base[:2]) if len(first_base) >= 2 else 0.0)
    return result


class Trial:
    def __init__(self, plan: dict[str, Any], path: Path) -> None:
        self.plan = plan
        self.path = path
        self.summary = read_json(path / "summary.json")
        self.metadata = read_json(path / "trial_metadata.json", {}) or {}
        self.watchdog = read_json(path / "watchdog_summary.json", {}) or {}
        self.resources = read_json(path / "resource_summary.json", {}) or {}
        self.rows = load_csv(path / "telemetry.csv")

    @property
    def launched(self) -> bool:
        """A manifest row is launched only after its command ledger is written."""
        return (self.path / "command.json").is_file()

    @property
    def trial_id(self) -> str:
        return str(self.plan.get("trial_id"))

    @property
    def runtime_valid(self) -> bool:
        return bool(
            self.summary
            and self.summary.get("runtime_status") == "RUNTIME_PASS_FINALIZE_REACHED"
            and (self.path / "events.json").is_file()
            and (self.path / "deployment_observation.csv").is_file()
            and (self.path / "telemetry.csv").is_file()
        )

    @property
    def local_decision(self) -> str:
        return str((self.summary or {}).get("physical_probe_result", {}).get("decision") or "UNKNOWN")

    @property
    def upper_action(self) -> str:
        return str((self.summary or {}).get("physical_probe_result", {}).get("high_level_policy", {}).get("filtered_action") or "UNKNOWN")

    @property
    def clearance(self) -> bool | None:
        if not self.runtime_valid:
            return None
        summary = self.summary or {}
        return bool(summary.get("candidate_clearance_verified", summary.get("whole_body_candidate_clearance", False)))

    @property
    def pointgoal(self) -> bool | None:
        return bool((self.summary or {}).get("pointgoal_success", False)) if self.runtime_valid else None

    @property
    def error_code(self) -> str:
        if not self.launched:
            return "NOT_STARTED"
        if not self.runtime_valid:
            if self.summary and "NO_PATH_THROUGH_EXPLICIT_ZERO_CELLS" in json.dumps(self.summary, ensure_ascii=False):
                return "E6_EXECUTION_PLANNER_INVALID"
            return "E6_RUNTIME_OR_EXECUTION_INVALID"
        if self.local_decision == "RESPONSIVE" and self.clearance is False:
            return "E1_CANDIDATE_FALSE_PASS"
        if self.local_decision in {"BLOCKED", "UNCERTAIN"} and self.clearance is True:
            return "E2_FALSE_REJECT"
        return "NONE"

    def record(self) -> dict[str, Any]:
        summary = self.summary or {}
        probe = summary.get("physical_probe_result") or {}
        features = probe.get("features") or {}
        t0, t1, t2 = (phase_features(self.rows, cutoff) for cutoff in ("T0", "T1", "T2"))
        return {
            "trial_id": self.trial_id,
            "stage": self.plan.get("stage"),
            "group": self.plan.get("group"),
            "condition_id": self.plan.get("condition_id"),
            "gate_id": self.plan.get("gate_id"),
            "scenario": self.plan.get("scenario"),
            "probe_side": self.plan.get("probe_side"),
            "seed": self.plan.get("seed"),
            "launched": self.launched,
            "runtime_valid": self.runtime_valid,
            "runtime_class": (
                "NOT_STARTED"
                if not self.launched
                else "VALID_EPISODE"
                if self.runtime_valid
                else "RUNTIME_INVALID_OR_EXECUTION_INVALID"
            ),
            "summary_status": summary.get("status", "MISSING_SUMMARY"),
            "runtime_status": summary.get("runtime_status"),
            "local_response": self.local_decision,
            "upper_action": self.upper_action,
            "upper_mapping": "RESPONSIVE_TO_CAUTIOUS_ADVANCE" if self.local_decision == "RESPONSIVE" else self.upper_action,
            "whole_body_clearance": self.clearance,
            "pointgoal_success": self.pointgoal,
            "safe_recovery": summary.get("safe_recovery") if self.runtime_valid else None,
            "safety_success": summary.get("safety_success") if self.runtime_valid else None,
            "goal_error_m": finite(summary.get("goal_error_m")),
            "error_code": self.error_code,
            "error_note": str(summary.get("error", "")) if summary else "missing summary",
            "hand_forward_progress_m": features.get("hand_forward_progress_m"),
            "hand_progress_ratio": features.get("hand_progress_ratio"),
            "effort_deviation": features.get("effort_deviation"),
            "upper_tracking_rms_rad": features.get("upper_tracking_rms"),
            "joint_velocity_mean_norm": features.get("joint_velocity_mean_norm"),
            "base_disturbance_m": features.get("base_disturbance_m"),
            "t0_hand_forward_delta_m": None if t0 is None else t0.get("hand_forward_delta_m"),
            "t1_hand_forward_delta_m": None if t1 is None else t1.get("hand_forward_delta_m"),
            "t1_base_delta_m": None if t1 is None else t1.get("base_delta_m"),
            "t1_tracking_rms_mean_rad": None if t1 is None else t1.get("upper_tracking_rms_mean_rad"),
            "t1_effort_rms_mean": None if t1 is None else t1.get("effort_rms_mean"),
            "t2_hand_forward_delta_m": None if t2 is None else t2.get("hand_forward_delta_m"),
            "t2_base_delta_m": None if t2 is None else t2.get("base_delta_m"),
            "wall_time_s": finite(self.resources.get("rollout_wall_time_s")),
            "gpu_peak_mb": finite(self.resources.get("gpu_memory_peak_mb")),
            "process_private_peak_mb": finite(self.resources.get("process_private_peak_mb")),
            "watchdog_timeout_reason": self.watchdog.get("timeout_reason"),
            "summary_path": str(self.path / "summary.json"),
        }


def load_trials(manifest: dict[str, Any], trial_root: Path) -> list[Trial]:
    return [Trial(plan, trial_root / str(plan["trial_id"])) for plan in manifest.get("trials", [])]


def frozen_match_rows(trials: list[Trial], rules: dict[str, Any]) -> list[dict[str, Any]]:
    limits = rules.get("maximum_absolute_differences", {})
    usable = [trial for trial in trials if trial.runtime_valid and phase_features(trial.rows, "T1")]
    output: list[dict[str, Any]] = []
    for index, first in enumerate(usable):
        for second in usable[index + 1 :]:
            if first.plan.get("probe_side") != second.plan.get("probe_side"):
                continue
            if first.plan.get("seed") != second.plan.get("seed"):
                continue
            if first.plan.get("condition_id") == second.plan.get("condition_id"):
                continue
            a, b = phase_features(first.rows, "T1") or {}, phase_features(second.rows, "T1") or {}
            diffs = {
                "hand_forward_delta_m": abs((a.get("hand_forward_delta_m") or math.nan) - (b.get("hand_forward_delta_m") or math.nan)),
                "base_delta_m": abs((a.get("base_delta_m") or math.nan) - (b.get("base_delta_m") or math.nan)),
                "upper_tracking_rms_mean_rad": abs((a.get("upper_tracking_rms_mean_rad") or math.nan) - (b.get("upper_tracking_rms_mean_rad") or math.nan)),
                "effort_rms_mean": abs((a.get("effort_rms_mean") or math.nan) - (b.get("effort_rms_mean") or math.nan)),
            }
            matched = all(math.isfinite(value) and value <= float(limits.get(key, math.inf)) for key, value in diffs.items())
            series_a = series_features(first.rows, {"physical_probe", "physical_probe_hold"})
            series_b = series_features(second.rows, {"physical_probe", "physical_probe_hold"})
            paired = min(len(series_a["time_s"]), len(series_b["time_s"]))
            distances: list[float] = []
            for key in ("tracking", "effort_deviation", "hand_progress", "base_disturbance"):
                if paired:
                    value = rms([x - y for x, y in zip(series_a[key][:paired], series_b[key][:paired])])
                    if value is not None:
                        distances.append(value)
            full_series_distance = mean(distances)
            output.append({
                "trial_a": first.trial_id,
                "condition_a": first.plan.get("condition_id"),
                "trial_b": second.trial_id,
                "condition_b": second.plan.get("condition_id"),
                "probe_side": first.plan.get("probe_side"),
                "seed": first.plan.get("seed"),
                **{f"abs_diff_{key}": value for key, value in diffs.items()},
                "passes_frozen_T1_match_rule": matched,
                "clearance_a": first.clearance,
                "clearance_b": second.clearance,
                "clearance_outcomes_differ": first.clearance != second.clearance,
                "full_series_distance_candidate": full_series_distance,
                "observation_insufficient_candidate": bool(matched and first.clearance != second.clearance and first.local_decision == second.local_decision),
                "feature_compression_candidate": bool(matched and full_series_distance is not None and full_series_distance > 0.01),
                "independent_confirmation_required": bool(matched and first.clearance != second.clearance),
            })
    return output


def left_right_rows(trials: list[Trial]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Trial]] = {}
    for trial in trials:
        key = (trial.plan.get("condition_id"), trial.plan.get("seed"), trial.plan.get("stage"))
        grouped.setdefault(key, {})[str(trial.plan.get("probe_side"))] = trial
    output: list[dict[str, Any]] = []
    for key, pair in sorted(grouped.items(), key=str):
        if "left" not in pair or "right" not in pair:
            continue
        left, right = pair["left"], pair["right"]
        l, r = left.record(), right.record()
        output.append({
            "condition_id": key[0],
            "seed": key[1],
            "stage": key[2],
            "left_trial_id": left.trial_id,
            "right_trial_id": right.trial_id,
            "left_runtime_valid": left.runtime_valid,
            "right_runtime_valid": right.runtime_valid,
            "left_local_response": left.local_decision,
            "right_local_response": right.local_decision,
            "left_upper_action": left.upper_action,
            "right_upper_action": right.upper_action,
            "left_clearance": left.clearance,
            "right_clearance": right.clearance,
            "left_pointgoal": left.pointgoal,
            "right_pointgoal": right.pointgoal,
            "abs_t1_hand_progress_diff_m": abs((l.get("t1_hand_forward_delta_m") or math.nan) - (r.get("t1_hand_forward_delta_m") or math.nan)),
            "abs_t1_base_delta_diff_m": abs((l.get("t1_base_delta_m") or math.nan) - (r.get("t1_base_delta_m") or math.nan)),
            "abs_t1_tracking_diff_rad": abs((l.get("t1_tracking_rms_mean_rad") or math.nan) - (r.get("t1_tracking_rms_mean_rad") or math.nan)),
            "abs_t1_effort_diff": abs((l.get("t1_effort_rms_mean") or math.nan) - (r.get("t1_effort_rms_mean") or math.nan)),
            "outcome_differs": left.clearance != right.clearance if left.runtime_valid and right.runtime_valid else None,
            "interpretation": "left/right matched physical pair; do not pool across conditions",
        })
    return output


def plot_trial(trial: Trial, output_dir: Path) -> str | None:
    series = series_features(trial.rows, {"physical_probe", "physical_probe_hold", "cautious_advance", "responsive_traverse"})
    if not series["time_s"]:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    plots = [("tracking", "Joint tracking RMS (rad)"), ("effort_deviation", "Effort deviation"), ("hand_progress", "Hand forward progress (m)"), ("base_disturbance", "Base disturbance (m)")]
    for axis, (key, label) in zip(axes.flat, plots):
        axis.plot(series["time_s"], series[key], linewidth=1.2)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[1, 0].set_xlabel("time (s)")
    axes[1, 1].set_xlabel("time (s)")
    figure.suptitle(f"{trial.trial_id}: {trial.plan.get('condition_id')} {trial.plan.get('probe_side')}")
    figure.tight_layout()
    path = output_dir / f"{trial.trial_id}_tracking_effort_progress_base.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return str(path)


def resource_audit(trials: list[Trial], physics_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    launched_trials = [trial for trial in trials if trial.launched]
    for trial in launched_trials:
        rows.append({"run_id": trial.trial_id, "kind": "g1", **trial.resources})
    for path in sorted(physics_root.glob("*/resource_summary.json")):
        rows.append({"run_id": str(path.parent.relative_to(physics_root)), "kind": "physics", **(read_json(path, {}) or {})})
    gpu = [finite(row.get("gpu_memory_peak_mb")) for row in rows]
    private = [finite(row.get("process_private_peak_mb")) for row in rows]
    diagnostics: list[str] = []
    for trial in launched_trials:
        for log_name in ("process_stdout.log", "process_stderr.log"):
            path = trial.path / log_name
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for match in re.finditer(r"(?i).{0,80}(cuda|out of memory|solver|nan|inf).{0,160}", text):
                diagnostics.append(f"{trial.trial_id}/{log_name}: {match.group(0).strip()}")
    return {
        "run_count": len(rows),
        "g1_trial_count": len(launched_trials),
        "gpu_peak_max_mb": max((value for value in gpu if value is not None), default=None),
        "process_private_peak_max_mb": max((value for value in private if value is not None), default=None),
        "gpu_budget_mb": 6000.0,
        "process_private_budget_mb": 12000.0,
        "gpu_budget_exceeded": any(value is not None and value > 6000.0 for value in gpu),
        "process_private_budget_exceeded": any(value is not None and value > 12000.0 for value in private),
        "selected_log_diagnostics": diagnostics[:80],
        "resource_rows": rows,
    }


def historical_f2f3() -> dict[str, Any]:
    return {
        "known_invalid_history": "F2/F3 historical multi-sheet runs remain excluded; prior initial speeds were approximately 7.39-11.19 m/s.",
        "interpretation": "The campaign did not revalidate those structures; this is not evidence of a GPU memory limit being solved.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--research", type=Path, default=DEFAULT_RESEARCH)
    parser.add_argument("--trial-root", type=Path, default=DEFAULT_TRIAL_ROOT)
    parser.add_argument("--physics-root", type=Path, default=DEFAULT_PHYSICS_ROOT)
    args = parser.parse_args()
    manifest = read_json(args.manifest, {}) or {}
    rules = read_json(args.research / "ANALYSIS_RULES.json", {}) or {}
    trials = load_trials(manifest, args.trial_root)
    records = [trial.record() for trial in trials]
    launched = [trial for trial in trials if trial.launched]
    valid = [trial for trial in trials if trial.runtime_valid]
    runtime_invalid = [trial for trial in launched if not trial.runtime_valid]
    physics_rows = load_csv(args.research / "PHYSICS_GATE_RESULTS.csv")
    gate_by_id = {row.get("gate_id"): row for row in physics_rows}
    trial_manifest_rows = [{**trial.plan, **record, "gate_status": gate_by_id.get(str(trial.plan.get("gate_id")), {}).get("gate_status", "MISSING")} for trial, record in zip(trials, records)]
    write_csv(args.research / "TRIAL_MANIFEST.csv", trial_manifest_rows)
    write_csv(args.research / "EVIDENCE_MATRIX.csv", records)
    matches = frozen_match_rows(trials, rules)
    write_csv(args.research / "MATCHED_COUNTEREXAMPLES.csv", matches)
    write_csv(args.research / "LEFT_RIGHT_COMPARISON.csv", left_right_rows(trials))
    write_csv(
        args.research / "JUDGMENT_ERROR_CASES.csv",
        [record for record in records if record.get("launched") and record.get("error_code") != "NONE"],
    )
    plot_paths: list[str] = []
    for trial in valid:
        if str(trial.plan.get("group")) in {"S2_HEAVY_CURTAIN", "S1_HIDDEN_RIGID", "S3_S4"}:
            path = plot_trial(trial, args.research / "plots")
            if path:
                plot_paths.append(path)
    resources = resource_audit(trials, args.physics_root)
    write_json(args.research / "RESOURCE_AUDIT.json", resources)
    write_json(args.research / "TRIAL_RECORDS.json", {"records": records, "matches": matches})
    valid_flexible = [trial for trial in valid if trial.plan.get("scenario") != "rigid_panel"]
    e1 = [trial for trial in valid_flexible if trial.error_code == "E1_CANDIDATE_FALSE_PASS"]
    e2 = [trial for trial in valid_flexible if trial.error_code == "E2_FALSE_REJECT"]
    e3 = [row for row in matches if row.get("observation_insufficient_candidate")]
    e4 = [row for row in matches if row.get("feature_compression_candidate")]
    physics_failures = [row for row in physics_rows if row.get("gate_status") != "PASS_PHYSICS_GATE"]
    by_condition: dict[str, dict[str, int]] = {}
    for trial in trials:
        condition = str(trial.plan.get("condition_id"))
        bucket = by_condition.setdefault(condition, {"valid": 0, "invalid": 0, "responsive": 0, "blocked": 0, "clearance": 0, "no_clearance": 0, "e1": 0, "e2": 0})
        if not trial.launched:
            continue
        if trial.runtime_valid:
            bucket["valid"] += 1
            bucket["responsive"] += trial.local_decision == "RESPONSIVE"
            bucket["blocked"] += trial.local_decision == "BLOCKED"
            bucket["clearance"] += bool(trial.clearance)
            bucket["no_clearance"] += not bool(trial.clearance)
        else:
            bucket["invalid"] += 1
        bucket["e1"] += trial.error_code == "E1_CANDIDATE_FALSE_PASS"
        bucket["e2"] += trial.error_code == "E2_FALSE_REJECT"
    summary = {
        "schema": "lift-the-curtain.round4070-failure-discovery-analysis.v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "analysis_rules_sha256": sha256(args.research / "ANALYSIS_RULES.json"),
        "trial_accounting": {
            "planned": len(trials),
            "launched": len(launched),
            "runtime_valid": len(valid),
            "runtime_invalid_or_execution_invalid": len(runtime_invalid),
            "not_started": len(trials) - len(launched),
            "hard_budget": manifest.get("new_g1_trial_hard_limit"),
        },
        "physics_gate_accounting": {"gate_rows": len(physics_rows), "passed": sum(row.get("gate_status") == "PASS_PHYSICS_GATE" for row in physics_rows), "failed_or_invalid": len(physics_failures), "failed_gate_ids": [row.get("gate_id") for row in physics_failures]},
        "error_counts": {"E1": len(e1), "E2": len(e2), "E3_candidates": len(e3), "E4_candidates": len(e4), "E5": len(physics_failures), "E6": len(runtime_invalid)},
        "condition_summary": by_condition,
        "matched_counterexample_count": sum(bool(row.get("passes_frozen_T1_match_rule")) for row in matches),
        "matched_discordant_count": sum(bool(row.get("passes_frozen_T1_match_rule") and row.get("clearance_outcomes_differ")) for row in matches),
        "independent_confirmation_launched": sum(1 for trial in trials if trial.plan.get("stage") == "confirmation" and (trial.path / "summary.json").is_file()),
        "plot_paths": plot_paths,
        "resource_audit": {key: value for key, value in resources.items() if key != "resource_rows"},
        "historical_f2_f3": historical_f2f3(),
        "interpretation_limits": [
            "RESPONSIVE is an online local response label; it is not silently renamed PASSABLE.",
            "E1 maps the frozen RESPONSIVE-to-CAUTIOUS_ADVANCE branch to a valid no-clearance episode; it is not a claim that the controller emitted a final PASSABLE label.",
            "E5 and E6 records are excluded from judgment-error denominators.",
            "S1-D is excluded from G1 because its physics probe crossed the hidden wall front face.",
        ],
    }
    write_json(args.research / "CAMPAIGN_ANALYSIS.json", summary)
    print(json.dumps({"valid": len(valid), "e1": len(e1), "e2": len(e2), "e3": len(e3), "e4": len(e4), "plots": len(plot_paths)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

