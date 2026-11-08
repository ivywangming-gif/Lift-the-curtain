"""Replay the frozen Round 4070-G E1_FIXED rule on every legal current history.

This runner is intentionally offline and read-only with respect to experiment
state.  It consumes the admitted primary manifest plus the separate pending
collision-audit manifest, never promotes an UNKNOWN label, and writes only
auditable baseline artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
OUT = ROOT / "outputs/round4070_comprehensive_traversability"
RESEARCH = ROOT / "research/round4070_comprehensive_traversability"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from engineering_rules import evaluate_rules  # noqa: E402
from if_else_baseline import E1_FIXED_VERSION, evaluate, compare_compatibility  # noqa: E402
from learning_core import load_dataset, metrics  # noqa: E402


PRIMARY_PROTOCOL = "G1_RNG_V1_BODY_COLLISION_200HZ_ACTION_TUBE_V2"
LABEL_VALIDITY_VERSION = "G_BODY_COLLISION_VALIDITY_V1"
ACTION_CONTRACT = "G_HOLD_PROBE_POSE_FORWARD24_TUBE_V2"
EXPECTED_CURRENT_HISTORY_COUNT = 21


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def write_text_atomic(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_json_atomic(path: Path, value: Any) -> None:
    write_text_atomic(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n",
    )


def write_csv_atomic(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
            ).strip()
        except (OSError, subprocess.CalledProcessError) as error:
            return f"UNAVAILABLE:{error}"

    status = run("status", "--porcelain")
    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "worktree_dirty": bool(status),
        "worktree_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def run_contract_tests() -> dict[str, Any]:
    """Run the narrow E1 contract/regression suite before publishing reports."""
    command = [sys.executable, "-m", "unittest", "scripts/round4070_g/test_if_else_baseline.py", "-q"]
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=180,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"status": "ERROR", "returncode": None, "command": command, "error": repr(error)}
    output = (result.stdout + "\n" + result.stderr).strip()
    return {
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "returncode": result.returncode,
        "command": command,
        "output_tail": output[-2000:],
    }


def verify_current_campaign() -> dict[str, Any]:
    pointer_path = RESEARCH / "CURRENT_CAMPAIGN.json"
    pointer = read_json(pointer_path)
    manifest_name = str(pointer.get("manifest", ""))
    if not manifest_name or Path(manifest_name).name != manifest_name:
        raise RuntimeError("Current campaign pointer is missing or escapes its output directory")
    manifest_path = OUT / manifest_name
    manifest_hash = sha256(manifest_path)
    sidecar = read_json(manifest_path.with_suffix(".sha256.json"))
    if manifest_hash != pointer.get("sha256") or manifest_hash != sidecar.get("sha256"):
        raise RuntimeError("Current frozen campaign SHA256 mismatch")
    if pointer.get("primary_protocol") != PRIMARY_PROTOCOL:
        raise RuntimeError("Current campaign pointer has an unexpected primary protocol")
    return {
        "pointer_path": relative(pointer_path),
        "manifest_path": relative(manifest_path),
        "manifest_sha256": manifest_hash,
        "primary_protocol": PRIMARY_PROTOCOL,
    }


def _load_pending_episode(
    episode: dict[str, Any], names: list[str]
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, str]:
    if episode.get("primary_protocol") != PRIMARY_PROTOCOL:
        raise ValueError(f"Pending episode is not current primary protocol: {episode['trial_id']}")
    if episode.get("label_valid") is not False or int(episode.get("y", -1)) != -1:
        raise ValueError(f"Pending episode must remain UNKNOWN: {episode['trial_id']}")
    if episode.get("label") != "LABEL_UNKNOWN" or episode.get("supervision_eligible") is not False:
        raise ValueError(f"Pending episode has an invalid supervision state: {episode['trial_id']}")
    if episode.get("response_history_eligible") is not False:
        raise ValueError(f"Pending episode must not enter response history: {episode['trial_id']}")
    if episode.get("is_control") or not episode.get("runtime_valid") or not episode.get("probe_valid"):
        raise ValueError(f"Pending episode is not a legal replay history: {episode['trial_id']}")
    if episode.get("action_contract") != ACTION_CONTRACT:
        raise ValueError(f"Pending episode action contract drifted: {episode['trial_id']}")
    path = resolve_repo_path(episode["npz_path"])
    actual_hash = sha256(path)
    expected_hash = episode.get("npz_sha256")
    if expected_hash and actual_hash != expected_hash:
        raise ValueError(f"Pending NPZ hash mismatch: {episode['trial_id']}")
    with np.load(path, allow_pickle=False) as data:
        observed_names = data["feature_names"].astype(str).tolist()
        sequence = data["seq"].astype(np.float32)
        timestamp = data["time"].astype(np.float64)
    if observed_names != names:
        raise ValueError(f"Pending feature schema differs: {episode['trial_id']}")
    if sequence.ndim != 2 or sequence.shape != (len(timestamp), len(names)) or len(timestamp) < 2:
        raise ValueError(f"Pending sequence shape is invalid: {episode['trial_id']}")
    if not np.isfinite(sequence).all() or not np.isfinite(timestamp).all() or not np.all(np.diff(timestamp) > 0):
        raise ValueError(f"Pending sequence timestamps are invalid: {episode['trial_id']}")
    return dict(episode), sequence, timestamp - timestamp[0], actual_hash


def load_current_batch() -> dict[str, Any]:
    accepted_path = OUT / "dataset/manifest.json"
    pending_path = OUT / "dataset/pending_collision_audit_manifest.json"
    accepted_manifest, accepted, names, raw, times, source_hashes = load_dataset(accepted_path)
    pending_manifest = read_json(pending_path)
    if pending_manifest.get("schema") != "round4070-g.pending_collision_audit.v1":
        raise RuntimeError("Unexpected pending collision-audit manifest schema")

    episodes = [dict(item) for item in accepted]
    source_manifest_by_id = {item["trial_id"]: "dataset/manifest.json" for item in accepted}
    source_npz_hashes = dict(source_hashes)
    pending_ids = set()
    for item in pending_manifest.get("episodes", []):
        episode, sequence, timestamp, digest = _load_pending_episode(item, names)
        if episode["trial_id"] in {e["trial_id"] for e in episodes}:
            raise RuntimeError(f"Duplicate current episode: {episode['trial_id']}")
        pending_ids.add(episode["trial_id"])
        episodes.append(episode)
        raw.append(sequence)
        times.append(timestamp)
        source_npz_hashes[episode["trial_id"]] = digest
        source_manifest_by_id[episode["trial_id"]] = "dataset/pending_collision_audit_manifest.json"

    if len(episodes) != EXPECTED_CURRENT_HISTORY_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_CURRENT_HISTORY_COUNT} current histories, got {len(episodes)}")
    if len({episode["trial_id"] for episode in episodes}) != len(episodes):
        raise RuntimeError("Current E1 input contains duplicate episode IDs")
    if not pending_ids:
        raise RuntimeError("Pending collision-audit histories disappeared from the E1 input")
    if any(episode.get("primary_protocol") != PRIMARY_PROTOCOL for episode in episodes):
        raise RuntimeError("A non-current protocol entered the E1 input")
    if any(episode.get("label_valid") is False and int(episode.get("y", -1)) >= 0 for episode in episodes):
        raise RuntimeError("An invalid labelled episode entered the E1 input")
    if names != accepted_manifest["feature_names"]:
        raise RuntimeError("Accepted feature schema changed during loading")

    input_files = {
        "current_campaign": verify_current_campaign(),
        "accepted_manifest": {"path": relative(accepted_path), "sha256": sha256(accepted_path)},
        "pending_manifest": {"path": relative(pending_path), "sha256": sha256(pending_path)},
    }
    return {
        "accepted_manifest": accepted_manifest,
        "pending_manifest": pending_manifest,
        "episodes": episodes,
        "names": names,
        "raw": raw,
        "times": times,
        "source_npz_hashes": source_npz_hashes,
        "source_manifest_by_id": source_manifest_by_id,
        "pending_ids": sorted(pending_ids),
        "input_files": input_files,
    }


def _decision_probability(decision: str) -> float:
    return {"RESPONSIVE": 1.0, "BLOCKED": 0.0, "UNCERTAIN": 0.5}[decision]


def _decision_numeric(decision: str) -> int:
    return {"RESPONSIVE": 1, "BLOCKED": 0, "UNCERTAIN": -1}[decision]


def _label_is_valid(row: dict[str, Any]) -> bool:
    return row.get("label_evaluation_scope") == "CORE" and row.get("actual_label") in {"PASSABLE", "BLOCKED"}


def summarize_subset(name: str, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    subset = list(rows)
    core = [row for row in subset if row.get("label_evaluation_scope") == "CORE"]
    valid = [row for row in core if _label_is_valid(row)]
    y = np.asarray([1 if row["actual_label"] == "PASSABLE" else 0 for row in valid], dtype=int)
    p = np.asarray([_decision_probability(row["e1_decision"]) for row in valid], dtype=float)
    d = np.asarray([_decision_numeric(row["e1_decision"]) for row in valid], dtype=int)
    return {
        "group": name,
        "rows_total": len(subset),
        "core_rows": len(core),
        "control_rows": sum(bool(row.get("is_control")) for row in subset),
        "unknown_core_rows": sum(row.get("actual_label") == "LABEL_UNKNOWN" for row in core),
        "valid_core_rows": len(valid),
        "known_label_rows_all_scopes": sum(
            row.get("actual_label") in {"PASSABLE", "BLOCKED", "CONTROL_PASSABLE"} for row in subset
        ),
        "decision_counts_all_rows": dict(Counter(row["e1_decision"] for row in subset)),
        "label_counts_all_rows": dict(Counter(row["actual_label"] for row in subset)),
        "dataset_label_counts_all_rows": dict(Counter(row.get("dataset_label", "") for row in subset)),
        "actual_label_counts_core": dict(Counter(row["actual_label"] for row in core)),
        "metrics": metrics(y, p, decisions=d),
        "episode_ids": [row["episode_id"] for row in subset],
    }


def build_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    families = sorted({str(row["scene_family"]) for row in rows})
    scenes = sorted({str(row["scene_id"]) for row in rows})
    physical_configs = sorted({str(row["physical_config_id"]) for row in rows})
    roles = sorted({str(row["experiment_role"]) for row in rows})
    sides = sorted({str(row["side"]) for row in rows})
    return {
        "overall": summarize_subset("ALL_CURRENT_ROWS", rows),
        "overall_core_valid": summarize_subset("OVERALL_CORE_VALID_SCOPE", rows),
        "by_side": {side: summarize_subset(f"SIDE_{side}", [r for r in rows if r["side"] == side]) for side in sides},
        "by_scene_family": {
            family: summarize_subset(f"FAMILY_{family}", [r for r in rows if r["scene_family"] == family])
            for family in families
        },
        "by_scene": {
            scene: summarize_subset(f"SCENE_{scene}", [r for r in rows if r["scene_id"] == scene])
            for scene in scenes
        },
        "by_physical_config": {
            config: summarize_subset(
                f"PHYSICAL_CONFIG_{config}",
                [r for r in rows if r["physical_config_id"] == config],
            )
            for config in physical_configs
        },
        "by_experiment_role": {
            role: summarize_subset(f"ROLE_{role}", [r for r in rows if r["experiment_role"] == role])
            for role in roles
        },
        "ood": {
            "status": "UNAVAILABLE_NO_ELIGIBLE_OOD_HISTORY",
            "eligible_roles": {role: sum(_label_is_valid(r) for r in rows if r["experiment_role"] == role) for role in roles},
            "required_roles": ["test", "confirmation", "heldout_posture"],
        },
        "scope_note": "Unknown labels and EMPTY controls are excluded from overall_core_valid metrics; they remain in the row audit.",
    }


def compare_adapter_features(rows: list[dict[str, Any]], compatibility_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["episode_id"]: row for row in rows}
    feature_keys = (
        "sample_count",
        "hand_forward_progress_m",
        "hand_progress_ratio",
        "effort_deviation",
        "upper_tracking_rms",
        "hand_low_speed_duration_s",
    )
    tolerance = 1.0e-5
    mismatches = []
    within_tolerance = []
    max_abs = 0.0
    for item, compatibility in zip(rows, compatibility_rows):
        if item["episode_id"] not in by_id:
            mismatches.append(item["episode_id"])
            continue
        adapter_features = compatibility.get("features", {})
        for key in feature_keys:
            left, right = item.get(key), adapter_features.get(key)
            if left is None or right is None:
                if left != right:
                    mismatches.append(f"{item['episode_id']}:{key}")
                continue
            delta = abs(float(left) - float(right))
            max_abs = max(max_abs, delta)
            if delta > 1.0e-10:
                if delta <= tolerance:
                    within_tolerance.append(f"{item['episode_id']}:{key}")
                else:
                    mismatches.append(f"{item['episode_id']}:{key}")
    return {
        "numeric_feature_status": (
            "PASS_EXACT"
            if not mismatches and not within_tolerance
            else "PASS_FLOAT_PRECISION_TOLERANCE"
            if not mismatches
            else "MISMATCH"
        ),
        "numeric_feature_tolerance": tolerance,
        "numeric_feature_max_abs_delta": max_abs,
        "numeric_feature_mismatches": mismatches,
        "numeric_feature_within_tolerance": within_tolerance,
        "compared_keys": list(feature_keys),
    }


def source_hashes() -> dict[str, str]:
    paths = {
        "canonical_e1_fixed": SCRIPT_DIR / "if_else_baseline.py",
        "g_compatibility_adapter": SCRIPT_DIR / "engineering_rules.py",
        "historical_e1_reference": ROOT / "scripts/round4070_fixed_probe_benchmark/evaluate_benchmark.py",
        "semantic_observation_reference": ROOT / "lift_curtain/control/deployment_observation.py",
        "proprioceptive_thresholds": ROOT / "configs/proprioceptive_thresholds.json",
        "contract_test_source": SCRIPT_DIR / "test_if_else_baseline.py",
    }
    optional = ROOT / "configs/round4070_fixed_probe_benchmark/RULE_FREEZE.json"
    if optional.exists():
        paths["secondary_rule_freeze"] = optional
    return {key: sha256(path) for key, path in paths.items()}


def render_frozen_doc(
    *,
    generated_utc: str,
    git: dict[str, Any],
    hashes: dict[str, str],
    reference_audit: dict[str, Any],
    compatibility: dict[str, Any],
    input_scope: dict[str, Any],
    run_dir: Path,
    metrics_payload: dict[str, Any],
    contract_tests: dict[str, Any],
) -> str:
    thresholds = reference_audit["thresholds"]
    refs = reference_audit["sides"]
    overall = metrics_payload["overall_core_valid"]["metrics"]
    all_scope = metrics_payload["overall"]
    return f"""# Frozen Round 4070-G If–Else Baseline

## Freeze record

- Baseline identifier: `{E1_FIXED_VERSION}`
- Freeze/replay time (UTC): `{generated_utc}`
- Current Git HEAD at replay: `{git['head']}`
- Branch: `{git['branch']}`
- Worktree was dirty at replay: `{git['worktree_dirty']}`
- Status: `IN_PROGRESS_NOT_COMPLETE`
- This artifact is a deployment-rule baseline, not a completed benchmark or a claim of method advantage.

## Scope and inputs

The replay covers **{input_scope['history_count']}** legal current-primary G1 histories: **{input_scope['accepted_count']}** admitted histories from the primary manifest plus **{input_scope['pending_count']}** histories retained in the pending collision-audit manifest. Superseded pilot histories are excluded. The pending B10/D01 rows remain `LABEL_UNKNOWN` and are never promoted by this evaluator. Two EMPTY controls are used only as same-hand response references.

The evaluator consumes the frozen 252-channel sensor schema, actual observation timestamps, and the legal physical probe/probe-hold window. Time is made relative to the first recorded sample. The cutoff is at most 2.0 seconds; only observed samples are used, with no interpolation and no terminal extrapolation.

Named hand channels are resolved directly:

- left: `left_hand_delta_initial_b/x`, `left_hand_velocity_initial_b/x`
- right: `right_hand_delta_initial_b/x`, `right_hand_velocity_initial_b/x`

The engineering coordinate is the initial robot-body forward x axis. No scene ID, physical material, topology, contact truth, label, follow-up result, or future observation enters the decision.

## Frozen thresholds and rule order

The exact threshold source is `configs/proprioceptive_thresholds.json`:

```json
{json.dumps(thresholds, indent=2, ensure_ascii=False, sort_keys=True)}
```

The if–else order is fixed:

1. Fewer than `minimum_samples` observations -> `UNCERTAIN`.
2. Progress and same-hand reference ratio satisfy the responsive thresholds -> `RESPONSIVE`.
3. Otherwise, low progress/ratio together with effort deviation and real-time low-speed duration satisfy the blocked thresholds -> `BLOCKED`.
4. Otherwise -> `UNCERTAIN`.

The stall term is `sum(diff(actual_time) * low_speed_at_left_endpoint)`. The corrected unit is seconds; the former fixed-sample-period assumption is not used.

## Native outputs and upper-layer mapping

The raw E1 output is exactly one of `RESPONSIVE`, `BLOCKED`, or `UNCERTAIN`.

| Raw E1 output | Upper action | Offline operational encoding |
| --- | --- | --- |
| `RESPONSIVE` | `CAUTIOUS_ADVANCE` | `PASSABLE` only for metric compatibility |
| `BLOCKED` | `RETRACT` | `BLOCKED` |
| `UNCERTAIN` | `RETRACT` | `UNCERTAIN` |

`RESPONSIVE` is not a ground-truth `PASSABLE` label. The `PASSABLE` spelling exists only as an explicit numeric/metric compatibility encoding in the existing learning adapter.

Same-hand EMPTY references are frozen from the following legal controls:

- left: `{', '.join(refs['left']['episode_ids'])}`; progress `{refs['left']['progress_m']:.12g}` m
- right: `{', '.join(refs['right']['episode_ids'])}`; progress `{refs['right']['progress_m']:.12g}` m

## Replay and compatibility audit

- Run directory: `{relative(run_dir)}`
- Current-primary histories replayed: `{input_scope['history_count']}`
- Raw decision counts: `{json.dumps(metrics_payload['overall']['decision_counts_all_rows'], ensure_ascii=False, sort_keys=True)}`
- Known labels across all scopes: `{all_scope['known_label_rows_all_scopes']}` (including `{all_scope['control_rows']}` EMPTY controls)
- Non-control core metric rows: `{overall['n']}`
- Core decision coverage: `{overall['coverage']}`
- Existing `engineering_rules.py` compatibility: `{compatibility['status']}`
- Numeric feature compatibility: `{compatibility['numeric_feature_status']}` (maximum absolute delta `{compatibility['numeric_feature_max_abs_delta']:.6g}`, tolerance `{compatibility['numeric_feature_tolerance']:.6g}`)
- Contract/regression test suite: `{contract_tests['status']}`

The historical E1 implementation, the G compatibility adapter, the semantic observation module, and the threshold file were hashed at freeze time:

```text
{chr(10).join(f"{key}: {value}" for key, value in sorted(hashes.items()))}
```

The replay command is:

```powershell
& C:\\conda_envs\\isaaclab\\python.exe scripts/round4070_g/run_if_else_baseline.py
```

The runner executes `python -m unittest scripts/round4070_g/test_if_else_baseline.py -q` before publishing this report; the recorded result is `{contract_tests['status']}`.

No G1 process was started by this replay, and the global G1 pause files remain outside the evaluator's write set.
"""


def render_zh_report(
    *,
    generated_utc: str,
    input_scope: dict[str, Any],
    metrics_payload: dict[str, Any],
    compatibility: dict[str, Any],
    run_dir: Path,
    contract_tests: dict[str, Any],
) -> str:
    overall = metrics_payload["overall_core_valid"]
    m = overall["metrics"]
    all_scope = metrics_payload["overall"]
    dataset_labels = all_scope["dataset_label_counts_all_rows"]

    def pct(value: Any) -> str:
        return "NA" if value is None else f"{float(value) * 100:.2f}%"

    def line(group: str, value: dict[str, Any]) -> str:
        mm = value["metrics"]
        return (
            f"| {group} | {value['rows_total']} | {value['valid_core_rows']} | "
            f"{value['unknown_core_rows']} | {mm['uncertain_n']} | {pct(mm['coverage'])} | "
            f"{pct(mm['passable_recall'])} | {pct(mm['blocked_recall'])} |"
        )

    family_lines = "\n".join(line(f"family {key}", value) for key, value in metrics_payload["by_scene_family"].items())
    config_lines = "\n".join(
        line(f"config {key}", value) for key, value in metrics_payload["by_physical_config"].items()
    )
    side_lines = "\n".join(line(f"side {key}", value) for key, value in metrics_payload["by_side"].items())
    return f"""# Round 4070-G：E1_FIXED If–Else 基线回放报告

回放时间（UTC）：`{generated_utc}`
基线版本：`{E1_FIXED_VERSION}`
状态：`IN_PROGRESS_NOT_COMPLETE`

## 范围

本次只读回放覆盖当前主协议的 **{input_scope['history_count']}** 条合法 G1 历史：主 dataset manifest 中已接纳的 **{input_scope['accepted_count']}** 条，加上仍处于碰撞/action-tube 审计挂起状态的 **{input_scope['pending_count']}** 条。10 条 superseded pilot 不进入基线。B10 左侧与 D01 左右侧继续保持 `LABEL_UNKNOWN`，没有被 E1 输出改标。

两条 EMPTY 控制（左右手各一条）只用于冻结同手参考响应，单独审计，不进入核心真假标签分母。所有决策使用同一套 252 通道、同一观察窗口、真实记录时间；stall 时长采用 `diff(actual_time)` 积分，不使用固定采样周期替代。

## 核心结果

当前数据中的已知标签共 **{all_scope['known_label_rows_all_scopes']}** 条（原始计数 PASSABLE={dataset_labels.get('PASSABLE_FOR_ACTION', 0)}、BLOCKED={dataset_labels.get('BLOCKED', 0)}），其中包含左右两条 EMPTY 控制。排除控制后，非控制核心有效标签为 **{m['n']}** 条，其中 PASSABLE **{m['passable_n']}**、BLOCKED **{m['blocked_n']}**；另有核心 `LABEL_UNKNOWN` **{overall['unknown_core_rows']}** 条。控制不进入真假判断分母。

| 指标 | 数值 |
| --- | ---: |
| E1 原生 `RESPONSIVE` | {metrics_payload['overall']['decision_counts_all_rows'].get('RESPONSIVE', 0)} |
| E1 原生 `BLOCKED` | {metrics_payload['overall']['decision_counts_all_rows'].get('BLOCKED', 0)} |
| E1 原生 `UNCERTAIN` | {metrics_payload['overall']['decision_counts_all_rows'].get('UNCERTAIN', 0)} |
| 核心决策覆盖率（排除 UNKNOWN/控制） | {pct(m['coverage'])} |
| PASSABLE recall | {pct(m['passable_recall'])} |
| BLOCKED recall | {pct(m['blocked_recall'])} |
| balanced accuracy | {pct(m['balanced_accuracy'])} |
| false-pass | {pct(m['false_pass'])} |
| false-reject | {pct(m['false_reject'])} |
| UNCERTAIN | {m['uncertain_n']} |

## 按手侧

| 分组 | 总行数 | 核心有效 | UNKNOWN | UNCERTAIN | 覆盖率 | PASSABLE recall | BLOCKED recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{side_lines}

## 按场景族

| 分组 | 总行数 | 核心有效 | UNKNOWN | UNCERTAIN | 覆盖率 | PASSABLE recall | BLOCKED recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{family_lines}

同一物理配置 D03 的不同 seed/手侧记录保留为独立历史；其 E1 输出存在 `BLOCKED` 与 `UNCERTAIN` 的混合，未进行阈值调参或事后修正。D01 的两条 `BLOCKED` 输出仍只是对 UNKNOWN 历史的规则响应，不能当作有效 BLOCKED 真值。

## 按物理配置聚合

下表保留每个 episode 的独立记录，同时给出同一 `physical_config_id` 内的有效行数、UNKNOWN 数量与 UNCERTAIN 数量；它不是把训练 seed 当成独立物理配置的显著性检验。

| 分组 | 总行数 | 核心有效 | UNKNOWN | UNCERTAIN | 覆盖率 | PASSABLE recall | BLOCKED recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{config_lines}

## OOD 与兼容性

当前输入没有可用的 `test`、`confirmation` 或 `heldout_posture` 标签历史，因此 OOD 结果为不可评估，不报告泛化结论。
与现有 `scripts/round4070_g/engineering_rules.py` 的 E1 决策回归：**{compatibility['status']}**；其 `PASSABLE` 仅是数值指标兼容编码，不是 E1 原生语义。
数字特征逐项兼容性：**{compatibility['numeric_feature_status']}**（最大绝对差 `{compatibility['numeric_feature_max_abs_delta']:.6g}`，容差 `{compatibility['numeric_feature_tolerance']:.6g}`；差异来自旧适配器的 `float32` 与冻结回放的 `float64` 计算精度，不改变任何 E1 决策）。
契约与回归测试：**{contract_tests['status']}**。

## 研究状态

本次产物只冻结并回放官方工程 if–else 基线；没有启动新的 G1，没有删除或修改 `PAUSE_G1.flag`/`PAUSE.flag`，也没有宣称完整 benchmark、学习方法优势或研究完成。后续学习方法仍须在同一冻结基线上、同一历史和标签合同下进行。

产物目录：`{relative(run_dir)}`
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thresholds", type=Path, default=ROOT / "configs/proprioceptive_thresholds.json")
    args = parser.parse_args()

    campaign = verify_current_campaign()
    batch = load_current_batch()
    git = git_state()
    hashes = source_hashes()
    threshold_path = args.thresholds if args.thresholds.is_absolute() else ROOT / args.thresholds

    input_payload = {
        "e1_fixed_version": E1_FIXED_VERSION,
        "campaign": campaign,
        "input_files": batch["input_files"],
        "source_npz_hashes": batch["source_npz_hashes"],
        "canonical_source_sha256": hashes["canonical_e1_fixed"],
        "threshold_sha256": sha256(threshold_path),
    }
    input_digest = hashlib.sha256(
        json.dumps(input_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    run_dir = OUT / f"if_else_baseline_{input_digest[:20]}"

    rows, reference_audit = evaluate(
        batch["episodes"],
        batch["raw"],
        batch["times"],
        batch["names"],
        threshold_path,
    )
    for row, episode in zip(rows, batch["episodes"]):
        row.update(
            {
                "dataset_label": str(episode.get("label", "")),
                "dataset_y": int(episode.get("y", -1)),
                "dataset_evaluation_label": str(episode.get("evaluation_label", "")),
                "npz_path": str(episode["npz_path"]),
                "npz_sha256": batch["source_npz_hashes"][episode["trial_id"]],
                "source_manifest": batch["source_manifest_by_id"][episode["trial_id"]],
                "pending_collision_audit": episode["trial_id"] in batch["pending_ids"],
            }
        )

    train_indices = [
        i for i, episode in enumerate(batch["episodes"])
        if bool(episode.get("is_control")) and int(episode.get("y", -1)) == 1
    ]
    compatibility_rows, compatibility_reference = evaluate_rules(
        batch["episodes"], batch["raw"], batch["times"], batch["names"], train_indices
    )
    compatibility_rows = [
        dict(item, episode_id=episode["trial_id"])
        for item, episode in zip(compatibility_rows, batch["episodes"])
    ]
    compatibility = compare_compatibility(rows, compatibility_rows)
    compatibility.update(compare_adapter_features(rows, compatibility_rows))
    compatibility["status"] = (
        "PASS"
        if compatibility["status"] == "PASS"
        and compatibility["numeric_feature_status"] in {"PASS_EXACT", "PASS_FLOAT_PRECISION_TOLERANCE"}
        else "MISMATCH"
    )
    compatibility["existing_adapter_reference_audit"] = compatibility_reference

    contract_tests = run_contract_tests()
    if contract_tests["status"] != "PASS":
        raise RuntimeError("E1 contract/regression tests did not pass: " + json.dumps(contract_tests, ensure_ascii=False))

    metrics_payload = build_metrics(rows)
    generated_utc = datetime.now(timezone.utc).isoformat()
    input_scope = {
        "history_count": len(rows),
        "accepted_count": len(batch["accepted_manifest"].get("episodes", [])),
        "pending_count": len(batch["pending_manifest"].get("episodes", [])),
        "pending_ids": batch["pending_ids"],
        "superseded_pilots_excluded": True,
        "control_count": sum(bool(row["is_control"]) for row in rows),
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(run_dir / "rows.json", {"schema": "round4070-g.if_else.rows.v1", "rows": rows})
    write_json_atomic(run_dir / "reference_audit.json", reference_audit)
    write_json_atomic(run_dir / "metrics.json", metrics_payload)
    write_json_atomic(run_dir / "compatibility_audit.json", compatibility)

    csv_fields = [
        "episode_id",
        "planned_trial_id",
        "scene_id",
        "scene_family",
        "physical_config_id",
        "side",
        "seed",
        "robot_variant",
        "experiment_role",
        "source_stage",
        "is_control",
        "pending_collision_audit",
        "admission_status",
        "physics_valid",
        "runtime_valid",
        "probe_valid",
        "label_valid",
        "dataset_label",
        "dataset_y",
        "dataset_evaluation_label",
        "actual_label",
        "label_evaluation_scope",
        "e1_decision",
        "upper_contract_action",
        "operational_prediction",
        "determinate",
        "correct_determinate",
        "false_pass",
        "false_reject",
        "passable_uncertain",
        "blocked_uncertain",
        "sample_count",
        "cutoff_duration_s",
        "hand_forward_progress_m",
        "baseline_hand_forward_progress_m",
        "hand_progress_ratio",
        "effort_deviation",
        "hand_low_speed_duration_s",
        "upper_tracking_rms",
        "engineering_coordinate",
        "stall_integration",
        "baseline_source",
        "reference_side",
        "reference_episode_ids",
        "e1_fixed_version",
        "responsive_action_contract",
        "npz_path",
        "npz_sha256",
        "source_manifest",
    ]
    write_csv_atomic(RESEARCH / "IF_ELSE_BASELINE_ALL_SCENES.csv", rows, csv_fields)
    frozen_doc = render_frozen_doc(
        generated_utc=generated_utc,
        git=git,
        hashes=hashes,
        reference_audit=reference_audit,
        compatibility=compatibility,
        input_scope=input_scope,
        run_dir=run_dir,
        metrics_payload=metrics_payload,
        contract_tests=contract_tests,
    )
    zh_doc = render_zh_report(
        generated_utc=generated_utc,
        input_scope=input_scope,
        metrics_payload=metrics_payload,
        compatibility=compatibility,
        run_dir=run_dir,
        contract_tests=contract_tests,
    )
    write_text_atomic(RESEARCH / "FROZEN_IF_ELSE_BASELINE.md", frozen_doc)
    write_text_atomic(RESEARCH / "IF_ELSE_BASELINE_REPORT_ZH.md", zh_doc)

    status = {
        "schema": "round4070-g.if_else_baseline_status.v1",
        "status": "IN_PROGRESS_NOT_COMPLETE",
        "e1_fixed_version": E1_FIXED_VERSION,
        "generated_utc": generated_utc,
        "input_digest": input_digest,
        "run_directory": relative(run_dir),
        "input_scope": input_scope,
        "decision_counts": metrics_payload["overall"]["decision_counts_all_rows"],
        "dataset_label_counts": metrics_payload["overall"]["dataset_label_counts_all_rows"],
        "core_metrics": metrics_payload["overall_core_valid"]["metrics"],
        "compatibility_status": compatibility["status"],
        "contract_tests": contract_tests,
        "g1_admission_paused": (OUT / "PAUSE_G1.flag").exists(),
        "global_pause_present": (OUT / "PAUSE.flag").exists(),
        "outputs": {
            "frozen_doc": relative(RESEARCH / "FROZEN_IF_ELSE_BASELINE.md"),
            "zh_report": relative(RESEARCH / "IF_ELSE_BASELINE_REPORT_ZH.md"),
            "all_scenes_csv": relative(RESEARCH / "IF_ELSE_BASELINE_ALL_SCENES.csv"),
            "run_manifest": relative(run_dir / "run_manifest.json"),
        },
        "source_hashes": hashes,
        "git": git,
    }
    write_json_atomic(RESEARCH / "IF_ELSE_BASELINE_STATUS.json", status)

    artifact_paths = [
        run_dir / "rows.json",
        run_dir / "reference_audit.json",
        run_dir / "metrics.json",
        run_dir / "compatibility_audit.json",
        RESEARCH / "IF_ELSE_BASELINE_ALL_SCENES.csv",
        RESEARCH / "FROZEN_IF_ELSE_BASELINE.md",
        RESEARCH / "IF_ELSE_BASELINE_REPORT_ZH.md",
        RESEARCH / "IF_ELSE_BASELINE_STATUS.json",
    ]
    run_manifest = {
        "schema": "round4070-g.if_else_baseline_run.v1",
        "status": "IN_PROGRESS_NOT_COMPLETE",
        "e1_fixed_version": E1_FIXED_VERSION,
        "generated_utc": generated_utc,
        "input_digest": input_digest,
        "input_payload": input_payload,
        "input_scope": input_scope,
        "git": git,
        "source_hashes": hashes,
        "threshold_path": relative(threshold_path),
        "threshold_sha256": sha256(threshold_path),
        "compatibility": compatibility,
        "contract_tests": contract_tests,
        "artifact_hashes": {relative(path): sha256(path) for path in artifact_paths},
        "g1_pause_files_observed": {
            "PAUSE_G1.flag": (OUT / "PAUSE_G1.flag").exists(),
            "PAUSE.flag": (OUT / "PAUSE.flag").exists(),
        },
    }
    write_json_atomic(run_dir / "run_manifest.json", run_manifest)

    print(
        json.dumps(
            {
                "status": status["status"],
                "e1_fixed_version": E1_FIXED_VERSION,
                "run_directory": relative(run_dir),
                "history_count": len(rows),
                "decision_counts": metrics_payload["overall"]["decision_counts_all_rows"],
                "core_metrics": metrics_payload["overall_core_valid"]["metrics"],
                "compatibility_status": compatibility["status"],
                "pause_g1_present": status["g1_admission_paused"],
                "pause_present": status["global_pause_present"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
