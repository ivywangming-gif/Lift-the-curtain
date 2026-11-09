"""Fail-closed authorization and execution supervisor for BIMANUAL_V1 Stage 2A.

The default operation is read-only preflight.  No Kit import or simulator
process is permitted unless the exact plan, runner, worker, approval evidence,
protected hashes, and resource scope have all been independently bound.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "research/round4070_comprehensive_traversability"
OUTPUTS = ROOT / "outputs/round4070_comprehensive_traversability"
PLAN_PATH = REPORT / "BIMANUAL_STAGE2_KIT_PHYSICS_PLAN_20261109.json"
APPROVAL_PATH = REPORT / "BIMANUAL_STAGE2_KIT_PHYSICS_APPROVAL_TEMPLATE_20261109.json"
WORKER_REL = Path("scripts/round4070_g/bimanual_stage2_kit_physics_worker.py")
DOOR_CONTRACT_REL = Path("research/round4070_comprehensive_traversability/BIMANUAL_STAGE2_DOOR_ACCEPTANCE_CONTRACT_20261109.json")
LOCK_PATH = OUTPUTS / "resource_lock.json"

SOURCE_PATHS = {
    "scene_matrix": "research/round4070_comprehensive_traversability/BIMANUAL_V1_SCENE_MATRIX.csv",
    "readiness_csv": "research/round4070_comprehensive_traversability/BIMANUAL_DYNAMIC_SCENE_READINESS.csv",
    "environment_report": "research/round4070_comprehensive_traversability/BIMANUAL_PHYSICS_ENVIRONMENT_COMPATIBILITY.json",
    "dynamic_scene_builder": "scripts/round4070_g/bimanual_dynamic_scene_builder.py",
    "runtime_scene_builder": "scripts/round4070_g/physics_scene_builder.py",
    "physics_gate": "scripts/round4070_g/physics_gate.py",
    "door_acceptance_contract": "research/round4070_comprehensive_traversability/BIMANUAL_STAGE2_DOOR_ACCEPTANCE_CONTRACT_20261109.json",
    "v6_executor": "scripts/round4070_g/g1_action_executor_v6.py",
    "v6_adapter": "scripts/round4070_g/g1_trial_v6.py",
    "v6_protocol": "research/round4070_comprehensive_traversability/g1_execution_protocol_v6.json",
    "e1_fixed": "scripts/round4070_g/if_else_baseline.py",
    "falcon_contract": "lift_curtain/control/falcon_contract.py",
    "offline_supervisor": "scripts/round4070_g/run_bimanual_kit_physics_only.py",
    "native_worker": "scripts/round4070_g/bimanual_stage2_kit_physics_worker.py",
    "offline_supervisor_tests": "scripts/round4070_g/test_bimanual_stage2_runner.py",
    "native_worker_tests": "scripts/round4070_g/test_bimanual_stage2_worker.py",
}

REQUIRED_REVIEWER_FLAGS = (
    "explicitly_authorizes_kit_while_pause_flags_remain",
    "approves_scoped_addendum_to_global_hold",
    "approves_stage_2a_only_no_g1_usd_or_actions",
    "approves_24_runs_and_resource_scope",
    "door_thresholds_preregistered",
    "acknowledges_physx_seed_unverified_runs_remain_unresolved",
)
REQUIRED_DOOR_CHECKS = (
    "joint_limits_match_scene_plan",
    "drive_parameters_match_scene_plan",
    "contact_report_api_authored",
    "contact_reported_between_probe_and_door",
    "opens_in_declared_direction",
    "no_reverse_backdrive_beyond_tolerance",
    "no_joint_limit_overshoot",
    "release_angular_speed_bounded",
    "reset_angle_reproducible",
)
EXPECTED_SCENES = (
    "BI_A01_BASIC_CLOTH",
    "BI_C_THREE_MISSING_TOP_LEFT",
    "BI_C_FOUR_CORNERS",
    "BI_B_WALL_04CM",
    "BI_G_DOOR_ORDINARY",
    "BI_E_OPEN_MESH_PROXY",
    "BI_E_PVC_INDEPENDENT_PROXY",
    "BI_E_PVC_OVERLAP_PROXY",
)


class PreflightError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _inside_root(root: Path, relative: str | Path) -> Path:
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes repository: {relative}")
    return candidate


def _approval_time(value: str) -> bool:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return False
    return parsed.tzinfo is not None


def _verify_evidence(root: Path, approval: dict, role: str, blockers: list[str]) -> None:
    row = approval.get("approvals", {}).get(role, {})
    evidence_path = row.get("approval_evidence_path", "")
    evidence_hash = row.get("approval_evidence_sha256", "")
    if not evidence_path or not evidence_hash:
        blockers.append(f"{role.upper()}_SIGNED_EVIDENCE_MISSING")
        return
    try:
        path = _inside_root(root, evidence_path)
        if not path.is_file() or sha256(path) != evidence_hash:
            blockers.append(f"{role.upper()}_SIGNED_EVIDENCE_HASH_MISMATCH")
    except (OSError, ValueError):
        blockers.append(f"{role.upper()}_SIGNED_EVIDENCE_INVALID_PATH")


def _validate_approvals(root: Path, approval: dict, plan_hash: str, runner_hash: str,
                        blockers: list[str]) -> None:
    plan = _read_json(_inside_root(root, PLAN_PATH.relative_to(ROOT)))
    if approval.get("status") != "APPROVED_FOR_STAGE_2A_ONLY":
        blockers.append("STAGE_2A_NOT_APPROVED")
    if approval.get("plan_path") != PLAN_PATH.relative_to(ROOT).as_posix():
        blockers.append("APPROVAL_PLAN_PATH_MISMATCH")
    if approval.get("plan_sha256") != plan_hash:
        blockers.append("APPROVAL_PLAN_HASH_MISMATCH")
    if approval.get("runner_sha256") != runner_hash:
        blockers.append("APPROVAL_RUNNER_HASH_MISMATCH")
    if approval.get("worker_path") != WORKER_REL.as_posix():
        blockers.append("APPROVED_WORKER_PATH_MISMATCH")
    expected_hashes = {
        "pause_flag": plan.get("protected_hashes", {}).get("pause_flag"),
        "pause_g1_flag": plan.get("protected_hashes", {}).get("pause_g1_flag"),
        "v5_manifest": plan.get("protected_hashes", {}).get("v5_manifest"),
    }
    if approval.get("required_hashes") != expected_hashes:
        blockers.append("APPROVAL_PROTECTED_HASH_BINDING_MISMATCH")
    limits = approval.get("limits", {})
    scope = plan.get("scope", {})
    resources = plan.get("resource_limits", {})
    expected_limits = {
        "max_physics_runs": scope.get("max_physics_runs"),
        "max_concurrent_kit_processes": resources.get("concurrent_kit_processes"),
        "max_loaded_scenes": scope.get("max_loaded_scenes"),
        "headless": True,
        "single_gpu": scope.get("gpu"),
        "max_wall_seconds_per_run": resources.get("max_wall_seconds_per_run"),
        "max_total_wall_seconds": resources.get("max_total_wall_seconds"),
        "max_output_gib": resources.get("max_output_gib"),
        "stop_if_gpu_memory_used_mib_reaches": resources.get("stop_if_gpu_memory_used_mib_reaches"),
        "automatic_retries": resources.get("automatic_retries"),
        "output_root": scope.get("output_root"),
        "pause_flag_removal_or_mutation": False,
        "manifest_or_g1_queue_access": False,
        "g1_usd_or_controller": False,
        "falcon_or_e1_execution": False,
        "training_or_episode_creation": False,
        "formal_g1_collection": False,
    }
    if limits != expected_limits:
        blockers.append("APPROVAL_RESOURCE_OR_EXCLUSION_SCOPE_MISMATCH")
    stage_2b = approval.get("stage_2b_g1_native_hand_and_stop_retract", {})
    if stage_2b.get("included") is not False or stage_2b.get("requires_separate_approval") is not True:
        blockers.append("STAGE_2B_SCOPE_NOT_EXCLUDED")
    worker_hash = approval.get("worker_sha256", "")
    worker_path = _inside_root(root, WORKER_REL)
    if not worker_path.is_file():
        blockers.append("NATIVE_USD_PHYSICS_WORKER_MISSING")
    elif not worker_hash or sha256(worker_path) != worker_hash:
        blockers.append("APPROVED_WORKER_HASH_MISMATCH")

    owner = approval.get("approvals", {}).get("project_owner", {})
    if owner.get("status") != "APPROVED" or not owner.get("name_or_id"):
        blockers.append("PROJECT_OWNER_APPROVAL_MISSING")
    if owner.get("approved_scope") != "STAGE_2A_ONLY_PHYSICS_ONLY_NO_G1":
        blockers.append("PROJECT_OWNER_SCOPE_NOT_EXPLICITLY_LIMITED_TO_STAGE_2A")
    if owner.get("approved_plan_sha256") != plan_hash or owner.get("approved_worker_sha256") != worker_hash:
        blockers.append("PROJECT_OWNER_PLAN_OR_WORKER_BINDING_MISMATCH")
    if owner.get("approved_runner_sha256") != runner_hash:
        blockers.append("PROJECT_OWNER_DID_NOT_APPROVE_EXACT_RUNNER")
    if not owner.get("approval_reference") or not _approval_time(owner.get("signed_at", "")):
        blockers.append("PROJECT_OWNER_APPROVAL_REFERENCE_OR_TIME_INVALID")
    _verify_evidence(root, approval, "project_owner", blockers)

    reviewer = approval.get("approvals", {}).get("independent_safety_reviewer", {})
    if reviewer.get("status") != "APPROVED" or not reviewer.get("name_or_id"):
        blockers.append("INDEPENDENT_REVIEWER_APPROVAL_MISSING")
    implementation_author = approval.get("implementation_author_id", "")
    if not implementation_author or reviewer.get("name_or_id") == implementation_author:
        blockers.append("REVIEWER_MUST_BE_INDEPENDENT_OF_IMPLEMENTATION_AUTHOR")
    if reviewer.get("independent_of_implementation") is not True:
        blockers.append("REVIEWER_INDEPENDENCE_NOT_ATTESTED")
    if reviewer.get("name_or_id") == owner.get("name_or_id"):
        blockers.append("REVIEWER_MUST_BE_DIFFERENT_FROM_PROJECT_OWNER")
    if reviewer.get("reviewed_plan_sha256") != plan_hash:
        blockers.append("REVIEWER_PLAN_HASH_MISMATCH")
    if reviewer.get("reviewed_runner_sha256") != runner_hash:
        blockers.append("REVIEWER_RUNNER_HASH_MISMATCH")
    if reviewer.get("reviewed_worker_sha256") != worker_hash:
        blockers.append("REVIEWER_WORKER_HASH_MISMATCH")
    for name in REQUIRED_REVIEWER_FLAGS:
        if reviewer.get(name) is not True:
            blockers.append("REVIEWER_SCOPE_NOT_APPROVED:" + name)
    if not reviewer.get("approval_reference") or not _approval_time(reviewer.get("signed_at", "")):
        blockers.append("REVIEWER_APPROVAL_REFERENCE_OR_TIME_INVALID")
    _verify_evidence(root, approval, "independent_safety_reviewer", blockers)

    contract_path = approval.get("door_acceptance_contract_path", "")
    contract_hash = approval.get("door_acceptance_contract_sha256", "")
    if contract_path != DOOR_CONTRACT_REL.as_posix() or not contract_hash:
        blockers.append("DOOR_ACCEPTANCE_CONTRACT_NOT_PREREGISTERED")
    else:
        try:
            path = _inside_root(root, contract_path)
            if not path.is_file() or sha256(path) != contract_hash:
                blockers.append("DOOR_ACCEPTANCE_CONTRACT_HASH_MISMATCH")
            else:
                door_contract = _read_json(path)
                if (door_contract.get("schema") != "round4070-g.bimanual.door-acceptance-contract.v1"
                        or door_contract.get("status") != "PREREGISTERED"
                        or tuple(door_contract.get("required_checks", [])) != REQUIRED_DOOR_CHECKS
                        or not isinstance(door_contract.get("thresholds"), dict)
                        or door_contract.get("scene_id") != "BI_G_DOOR_ORDINARY"):
                    blockers.append("DOOR_ACCEPTANCE_CONTRACT_NOT_PREREGISTERED")
        except (OSError, ValueError):
            blockers.append("DOOR_ACCEPTANCE_CONTRACT_PATH_INVALID")


def preflight(root: Path = ROOT, *, approval_path: Path | None = None) -> dict:
    root = root.resolve()
    approval_path = approval_path or root / APPROVAL_PATH.relative_to(ROOT)
    blockers: list[str] = []
    checks: dict[str, object] = {}
    try:
        plan_path = _inside_root(root, PLAN_PATH.relative_to(ROOT))
        approval_file = _inside_root(root, approval_path.relative_to(root))
        runner_path = _inside_root(root, Path(__file__).resolve().relative_to(root))
        plan = _read_json(plan_path)
        approval = _read_json(approval_file)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"ready": False, "blockers": ["PREFLIGHT_INPUT_INVALID:" + str(exc)], "checks": checks}

    plan_hash = sha256(plan_path)
    runner_hash = sha256(runner_path)
    checks["plan_sha256"] = plan_hash
    checks["runner_sha256"] = runner_hash
    if plan.get("authorization", {}).get("runner_source_hash") != runner_hash:
        blockers.append("PLAN_RUNNER_HASH_MISMATCH")
    worker_path = _inside_root(root, WORKER_REL)
    worker_hash = sha256(worker_path) if worker_path.is_file() else "MISSING"
    if plan.get("authorization", {}).get("worker_source_hash") != worker_hash:
        blockers.append("PLAN_WORKER_HASH_MISMATCH")
    if plan.get("schema") != "round4070-g.bimanual.kit-physics-only-plan.v1":
        blockers.append("UNSUPPORTED_PLAN_SCHEMA")
    if plan.get("status") != "PENDING_INDEPENDENT_AUTHORIZATION":
        blockers.append("PLAN_STATUS_CHANGED_REVIEW_REQUIRED")
    if plan.get("expected_run_count") != 24 or len(plan.get("run_matrix", [])) != 8:
        blockers.append("RUN_MATRIX_SCOPE_MISMATCH")
    if tuple(row.get("scene_id") for row in plan.get("run_matrix", [])) != EXPECTED_SCENES:
        blockers.append("APPROVED_SCENE_SET_MISMATCH")
    if sorted(plan.get("seeds_per_scene", [])) != [4070701, 4070702, 4070703]:
        blockers.append("TEST_SEED_SCOPE_MISMATCH")
    scope = plan.get("scope", {})
    if (scope.get("mode") != "HEADLESS_KIT_PHYSICS_ONLY"
            or scope.get("max_concurrent_kit_processes") != 1
            or scope.get("max_loaded_scenes") != 1
            or scope.get("gpu") != "cuda:0"
            or scope.get("multi_gpu") is not False
            or scope.get("formal_g1_collection_authorized") is not False
            or scope.get("output_root") != "outputs/round4070_comprehensive_traversability/physics_diagnostics/bimanual_v1_kit_physics_stage2_20261109"):
        blockers.append("PLAN_EXECUTION_SCOPE_UNSAFE")
    expected_limits = {
        "max_wall_seconds_per_run": 600,
        "max_total_wall_seconds": 14400,
        "max_output_gib": 10,
        "stop_if_gpu_memory_used_mib_reaches": 7168,
        "concurrent_kit_processes": 1,
        "automatic_retries": 0,
    }
    limits = plan.get("resource_limits", {})
    for key, expected in expected_limits.items():
        if limits.get(key) != expected:
            blockers.append("RESOURCE_LIMIT_SCOPE_MISMATCH:" + key)
    contract = plan.get("acceptance_contract", {})
    if (contract.get("preserve_thresholds") is not True or contract.get("dt_s") != 0.005
            or contract.get("duration_s") != 24 or contract.get("probe_force_cap_n") != 3.0):
        blockers.append("FROZEN_PHYSICS_ACCEPTANCE_CONTRACT_MISMATCH")
    if plan.get("scene_matrix_recreated") is not False or plan.get("manifest_reset") is not False:
        blockers.append("PLAN_MUTATES_FROZEN_RESEARCH_STATE")

    configured_plan_hash = approval.get("plan_sha256")
    if configured_plan_hash != plan_hash:
        blockers.append("APPROVAL_PLAN_HASH_MISMATCH")

    expected_sources = plan.get("source_hashes", {})
    actual_sources: dict[str, str] = {}
    for name, relative in SOURCE_PATHS.items():
        expected = (plan.get("acceptance_contract", {}).get("baseline_sha256")
                    if name == "physics_gate" else expected_sources.get(name))
        path = _inside_root(root, relative)
        try:
            actual = sha256(path)
        except OSError:
            actual = "MISSING"
        actual_sources[name] = actual
        if not expected or actual != expected:
            blockers.append(f"PINNED_SOURCE_MISMATCH:{name}")

    protected = plan.get("protected_hashes", {})
    protected_paths = {
        "pause_flag": "outputs/round4070_comprehensive_traversability/PAUSE.flag",
        "pause_g1_flag": "outputs/round4070_comprehensive_traversability/PAUSE_G1.flag",
        "v5_manifest": "outputs/round4070_comprehensive_traversability/campaign_manifest_v5.json",
    }
    protected_actual = {}
    for key, relative in protected_paths.items():
        path = _inside_root(root, relative)
        try:
            actual = sha256(path)
        except OSError:
            actual = "MISSING"
        protected_actual[key] = actual
        if actual != protected.get(key):
            blockers.append("PROTECTED_HASH_MISMATCH:" + key)
    if any(not _inside_root(root, p).is_file() for p in protected_paths.values()):
        blockers.append("PAUSE_OR_FROZEN_MANIFEST_MISSING")

    run_ids = [f"BIV1A_{row['scene_id']}_seed{seed}_attempt01"
               for row in plan.get("run_matrix", []) for seed in plan.get("seeds_per_scene", [])]
    if len(run_ids) != 24 or len(set(run_ids)) != 24:
        blockers.append("RUN_IDS_NOT_UNIQUE")
    checks["source_sha256"] = actual_sources
    checks["protected_sha256"] = protected_actual
    checks["planned_runs"] = len(run_ids)
    checks["scene_ids"] = [row.get("scene_id") for row in plan.get("run_matrix", [])]
    checks["kit_launch_performed"] = False
    checks["g1_or_queue_access_performed"] = False

    try:
        _validate_approvals(root, approval, plan_hash, runner_hash, blockers)
    except (OSError, ValueError):
        blockers.append("APPROVAL_OR_WORKER_PATH_INVALID")

    # The existing physics gate does not serialize/reopen the tested USD and
    # does not evaluate the door joint/contact contract. It cannot be promoted
    # as the Stage 2A worker by a wrapper or by a zero-penetration count.
    checks["registered_worker"] = WORKER_REL.as_posix()
    checks["existing_gate_capability"] = {
        "native_usd_save_reopen": False,
        "hinged_door_contact_and_joint_acceptance": False,
    }
    checks["kit_launch_authorized"] = bool(not blockers)
    checks["stage_2b_or_g1_authorized"] = False
    unique = list(dict.fromkeys(blockers))
    return {"ready": not unique, "blockers": unique, "checks": checks}


def _tree_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for base, _, names in os.walk(path):
        for name in names:
            try:
                total += (Path(base) / name).stat().st_size
            except OSError:
                pass
    return total


def _assert_live_invariants(root: Path, plan: dict) -> None:
    expected = plan.get("protected_hashes", {})
    paths = {
        "pause_flag": "outputs/round4070_comprehensive_traversability/PAUSE.flag",
        "pause_g1_flag": "outputs/round4070_comprehensive_traversability/PAUSE_G1.flag",
        "v5_manifest": "outputs/round4070_comprehensive_traversability/campaign_manifest_v5.json",
    }
    for name, relative in paths.items():
        path = _inside_root(root, relative)
        try:
            actual = sha256(path)
        except OSError as exc:
            raise PreflightError("PROTECTED_FILE_MISSING_DURING_RUN:" + name) from exc
        if actual != expected.get(name):
            raise PreflightError("PROTECTED_HASH_CHANGED_DURING_RUN:" + name)


def _validate_worker_summary(summary: dict, *, run_id: str, scene_id: str,
                             physical_config_id: str, seed: int, usd_path: Path,
                             plan: dict) -> None:
    identity = (summary.get("schema") == "round4070-g.bimanual.stage2a.run.v1"
                and summary.get("run_id") == run_id
                and summary.get("scene_id") == scene_id
                and summary.get("physical_config_id") == physical_config_id
                and summary.get("seed") == seed)
    if not identity:
        raise PreflightError("WORKER_SUMMARY_IDENTITY_MISMATCH:" + run_id)
    if summary.get("acceptance_contract_sha256") != _canonical_sha256(plan.get("acceptance_contract", {})):
        raise PreflightError("WORKER_ACCEPTANCE_CONTRACT_MISMATCH:" + run_id)
    if summary.get("door_acceptance_contract_sha256") != plan.get("source_hashes", {}).get("door_acceptance_contract"):
        raise PreflightError("WORKER_DOOR_CONTRACT_MISMATCH:" + run_id)
    runtime_valid = summary.get("runtime_valid") is True
    physics_valid = summary.get("physics_valid") is True
    if not runtime_valid:
        if physics_valid or summary.get("failure_category") not in {
            "RESOURCE_BLOCKED", "SOURCE_OR_SCHEMA_ERROR", "RUNTIME_OR_IMPLEMENTATION_ERROR",
        }:
            raise PreflightError("UNCLASSIFIED_WORKER_RUNTIME_FAILURE:" + run_id)
        if usd_path.is_file() and summary.get("usd_sha256") not in (None, sha256(usd_path)):
            raise PreflightError("WORKER_USD_BINDING_MISMATCH:" + run_id)
        return
    if summary.get("usd_sha256") != sha256(usd_path):
        raise PreflightError("WORKER_USD_BINDING_MISMATCH:" + run_id)
    if summary.get("usd_saved_and_reopened") is not True:
        raise PreflightError("USD_SAVE_REOPEN_NOT_VERIFIED:" + run_id)
    if summary.get("usd_schema_bindings_verified") is not True:
        raise PreflightError("USD_SCHEMA_OR_BINDING_NOT_VERIFIED:" + run_id)
    if summary.get("resource_telemetry_complete") is not True:
        raise PreflightError("RESOURCE_TELEMETRY_INCOMPLETE:" + run_id)
    required_checks = (
        "finite", "node_speed_bounded", "neighbor_stretch", "fixed_or_support_drift", "nodes_below_ground",
        "sampled_interior_penetrations", "contact_demonstrated", "release_clear",
        "initial_hanging_stable", "rail_bounds_and_rotation", "late_node_speed", "reset_reproducible",
        "duration_completed",
    )
    checks = summary.get("physics_checks", {})
    missing = [name for name in required_checks if not isinstance(checks.get(name), bool)]
    if missing:
        raise PreflightError("FROZEN_PHYSICS_CHECKS_MISSING:" + run_id + ":" + ",".join(missing))
    seed_record = summary.get("seeded_initial_conditions", {})
    initial_conditions_verified = bool(
        summary.get("seeded_initial_conditions_verified") is True
        and seed_record.get("verified") is True
        and seed_record.get("initial_condition_signature_sha256") == _canonical_sha256({
            "probe_draw": seed_record.get("probe_draw"),
            "velocity_draw_sha256": seed_record.get("velocity_draw_sha256"),
        })
        and seed_record.get("python_seed") == seed
        and seed_record.get("numpy_pcg64_seed") == seed
        and seed_record.get("torch_seed") == seed
    )
    physx_seed_verified = bool(
        seed_record.get("physx_internal_rng_seed_controlled") is True
        and seed_record.get("physx_rng_seed_verified") is True
    )
    if summary.get("native_runtime_bindings_verified") is not True:
        raise PreflightError("NATIVE_RUNTIME_BINDING_NOT_VERIFIED:" + run_id)
    if summary.get("usd_file_immutable_during_physics") is not True:
        raise PreflightError("USD_CHANGED_DURING_PHYSICS:" + run_id)
    if not isinstance(summary.get("contact_event_count"), int) or summary["contact_event_count"] < 0:
        raise PreflightError("INVALID_NATIVE_CONTACT_EVENT_COUNT:" + run_id)
    has_door = scene_id == "BI_G_DOOR_ORDINARY"
    if has_door:
        door_checks = summary.get("door_checks", {})
        missing_door = [name for name in REQUIRED_DOOR_CHECKS if not isinstance(door_checks.get(name), bool)]
        if missing_door:
            raise PreflightError("DOOR_ACCEPTANCE_CHECKS_MISSING:" + run_id + ":" + ",".join(missing_door))
        failed_door = [name for name in REQUIRED_DOOR_CHECKS if door_checks[name] is not True]
        if failed_door and (summary.get("physics_valid") is not False
                            or summary.get("failure_category") != "PHYSICS_CONTRACT_VIOLATION"):
            raise PreflightError("UNCLASSIFIED_DOOR_CHECK_FAILURE:" + run_id + ":" + ",".join(failed_door))
    failed = [name for name in required_checks if checks[name] is not True]
    if failed:
        if (summary.get("physics_valid") is not False
                or summary.get("failure_category") != "PHYSICS_CONTRACT_VIOLATION"):
            raise PreflightError("UNCLASSIFIED_PHYSICS_CHECK_FAILURE:" + run_id + ":" + ",".join(failed))
        return
    if not initial_conditions_verified:
        if (summary.get("physics_valid") is not False
                or summary.get("failure_category") != "INITIAL_CONDITION_SEED_UNVERIFIED"):
            raise PreflightError("UNVERIFIED_INITIAL_CONDITION_SEED_CANNOT_PASS_GATE:" + run_id)
        return
    if not physx_seed_verified:
        if (summary.get("physics_valid") is not False
                or summary.get("failure_category") != "SEED_CONTROL_UNVERIFIED"):
            raise PreflightError("UNVERIFIED_PHYSX_SEED_CANNOT_PASS_GATE:" + run_id)
        return
    if summary.get("physics_valid") is not True:
        raise PreflightError("PHYSICS_GATE_STATUS_CONTRADICTS_CHECKS:" + run_id)


def _gpu_memory_mib() -> int:
    executable = shutil.which("nvidia-smi")
    if not executable:
        raise PreflightError("NVIDIA_SMI_UNAVAILABLE")
    result = subprocess.run(
        [executable, "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=10,
    )
    rows = [line.strip().split(",") for line in result.stdout.splitlines() if line.strip()]
    if not rows or int(rows[0][0].strip()) != 0:
        raise PreflightError("CUDA_DEVICE_0_TELEMETRY_UNAVAILABLE")
    return int(rows[0][1].strip())


def _active_simulator_processes() -> list[dict[str, object]]:
    try:
        import psutil
    except ImportError as exc:
        raise PreflightError("PROCESS_TELEMETRY_UNAVAILABLE") from exc
    watched_scripts = {
        "bimanual_stage2_kit_physics_worker.py", "physics_gate.py",
        "physics_collision_smoke.py", "physics_wall_smoke.py",
        "g1_trial.py", "g1_trial_v6.py",
    }
    active: list[dict[str, object]] = []
    for process in psutil.process_iter(("pid", "name", "exe", "cmdline")):
        try:
            info = process.info
            if info.get("pid") == os.getpid():
                continue
            name = str(info.get("name") or "").casefold()
            argv = info.get("cmdline") or []
            entry = Path(str(argv[1])).name.casefold() if len(argv) > 1 else ""
            native = name in {"kit.exe", "isaac-sim.exe", "kit", "isaac-sim"}
            python_worker = name.startswith("python") and entry in watched_scripts
            if native or python_worker:
                active.append({"pid": info.get("pid"), "name": name, "entrypoint": entry})
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return active


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def execute(root: Path = ROOT, *, approval_path: Path | None = None) -> int:
    """Run only a fully approved worker bundle; never falls back to physics_gate."""
    state = preflight(root, approval_path=approval_path)
    if not state["ready"]:
        raise PreflightError("STAGE_2A_EXECUTION_REFUSED:" + ",".join(state["blockers"]))
    root = root.resolve()
    plan = _read_json(_inside_root(root, PLAN_PATH.relative_to(ROOT)))
    approval = _read_json(_inside_root(root, (approval_path or root / APPROVAL_PATH.relative_to(ROOT)).relative_to(root)))
    worker = _inside_root(root, WORKER_REL)
    output = _inside_root(root, plan["scope"]["output_root"])
    if LOCK_PATH.exists():
        raise PreflightError("RESOURCE_LOCK_BUSY; do not remove or override it")
    if output.exists() and not (output / "runner_state.json").is_file():
        raise PreflightError("OUTPUT_ROOT_EXISTS_WITHOUT_RUNNER_STATE; append-only recovery review required")

    token = uuid.uuid4().hex
    lock_record = {"owner": "bimanual_stage2a", "pid": os.getpid(), "token": token,
                   "created_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise PreflightError("RESOURCE_LOCK_BUSY; do not remove or override it") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(lock_record, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())

    try:
        active = _active_simulator_processes()
        if active:
            raise PreflightError("SIMULATOR_PROCESS_ALREADY_ACTIVE:" + json.dumps(active))
        if _gpu_memory_mib() >= 7168:
            raise PreflightError("GPU_MEMORY_LIMIT_BEFORE_LAUNCH")
        output.mkdir(parents=True, exist_ok=True)
        state_path = output / "runner_state.json"
        if state_path.exists():
            saved = _read_json(state_path)
            if saved.get("plan_sha256") != sha256(PLAN_PATH) or saved.get("runner_sha256") != sha256(Path(__file__)):
                raise PreflightError("RESUME_HASH_MISMATCH; append-only output retained")
            accumulated_wall_seconds = float(saved.get("elapsed_wall_seconds", 0.0))
        else:
            accumulated_wall_seconds = 0.0
            state_path.write_text(json.dumps({
                "schema": "round4070-g.bimanual.stage2a-runner-state.v1",
                "plan_sha256": sha256(PLAN_PATH), "runner_sha256": sha256(Path(__file__)),
                "approval_sha256": sha256(_inside_root(root, (approval_path or root / APPROVAL_PATH.relative_to(ROOT)).relative_to(root))),
                "runs": [], "scene_results": {}, "elapsed_wall_seconds": 0.0, "status": "RUNNING",
            }, indent=2) + "\n", encoding="utf-8")

        if accumulated_wall_seconds >= 14400:
            raise PreflightError("TOTAL_WALL_TIME_LIMIT_ALREADY_CONSUMED")
        total_started = time.monotonic()
        execution_complete = False

        def persist_state(*, status: str, active_run_id: str | None = None) -> None:
            current = _read_json(state_path)
            current["status"] = status
            current["active_run_id"] = active_run_id
            current["elapsed_wall_seconds"] = accumulated_wall_seconds + time.monotonic() - total_started
            current["updated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            temporary = state_path.with_name(state_path.name + ".tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(current, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, state_path)

        for row in plan["run_matrix"]:
            scene_id = row["scene_id"]
            scene_runs: list[dict] = []
            scene_plan_rel = row["plan_path"]
            scene_plan = _inside_root(root, scene_plan_rel)
            if sha256(scene_plan) != row["plan_sha256"]:
                raise PreflightError("SCENE_PLAN_HASH_MISMATCH:" + scene_id)
            scene_asset = _inside_root(root, _read_json(scene_plan)["source_scene_asset"])
            scene_usd = output / "usd" / f"{scene_id}_{row['plan_sha256'][:12]}.usd"
            scene_failed = False
            for seed in plan["seeds_per_scene"]:
                run_id = f"BIV1A_{scene_id}_seed{seed}_attempt01"
                run_dir = output / "runs" / run_id
                if run_dir.exists():
                    summary = run_dir / "summary.json"
                    if not summary.is_file():
                        raise PreflightError("EXISTING_INCOMPLETE_RUN_REQUIRES_REVIEW:" + run_id)
                    result = _read_json(summary)
                    _validate_worker_summary(result, run_id=run_id, scene_id=scene_id,
                                             physical_config_id=row["physical_config_id"], seed=seed,
                                             usd_path=scene_usd, plan=plan)
                    scene_runs.append(result)
                    if (result.get("physics_valid") is not True
                            and result.get("failure_category") != "SEED_CONTROL_UNVERIFIED"):
                        scene_failed = True
                        break
                    continue
                if accumulated_wall_seconds + time.monotonic() - total_started >= 14400:
                    raise PreflightError("TOTAL_WALL_TIME_LIMIT")
                active = _active_simulator_processes()
                if active:
                    raise PreflightError("SIMULATOR_PROCESS_ALREADY_ACTIVE:" + json.dumps(active))
                if _gpu_memory_mib() >= 7168:
                    raise PreflightError("GPU_MEMORY_LIMIT_BEFORE_RUN:" + run_id)
                run_dir.mkdir(parents=True, exist_ok=False)
                runlog = (run_dir / "worker_console.log").open("w", encoding="utf-8")
                command = [sys.executable, str(worker), "--plan", str(scene_plan),
                           "--scene-asset", str(scene_asset), "--scene-id", scene_id,
                           "--seed", str(seed), "--run-id", run_id, "--run-root", str(run_dir),
                           "--usd-path", str(scene_usd), "--headless"]
                if not scene_usd.exists():
                    command.append("--author-usd")
                else:
                    command.extend(["--open-existing-usd", "--expected-usd-sha256", sha256(scene_usd)])
                started = time.monotonic()
                persist_state(status="RUNNING", active_run_id=run_id)
                process = subprocess.Popen(command, cwd=root, stdout=runlog, stderr=subprocess.STDOUT)
                try:
                    while process.poll() is None:
                        if time.monotonic() - started > 600:
                            _stop_process(process)
                            raise PreflightError("PER_RUN_WALL_TIME_LIMIT:" + run_id)
                        if _gpu_memory_mib() >= 7168:
                            _stop_process(process)
                            raise PreflightError("GPU_MEMORY_LIMIT:" + run_id)
                        if _tree_size(output) >= 10 * 1024**3:
                            _stop_process(process)
                            raise PreflightError("OUTPUT_SIZE_LIMIT:" + run_id)
                        try:
                            _assert_live_invariants(root, plan)
                        except PreflightError:
                            _stop_process(process)
                            raise
                        time.sleep(1)
                finally:
                    runlog.close()
                if process.returncode != 0:
                    raise PreflightError("WORKER_RUNTIME_ERROR:" + run_id)
                summary_path = run_dir / "summary.json"
                if not summary_path.is_file():
                    raise PreflightError("WORKER_SUMMARY_MISSING:" + run_id)
                summary = _read_json(summary_path)
                _validate_worker_summary(summary, run_id=run_id, scene_id=scene_id,
                                         physical_config_id=row["physical_config_id"], seed=seed,
                                         usd_path=scene_usd, plan=plan)
                scene_runs.append(summary)
                with state_path.open("r+", encoding="utf-8") as handle:
                    current = json.load(handle)
                    current["runs"].append({"run_id": run_id, "scene_id": scene_id, "seed": seed,
                                            "physics_valid": summary.get("physics_valid"),
                                            "summary_sha256": sha256(summary_path), "usd_sha256": sha256(scene_usd)})
                    handle.seek(0)
                    json.dump(current, handle, indent=2)
                    handle.write("\n")
                    handle.truncate()
                    handle.flush()
                    os.fsync(handle.fileno())
                persist_state(status="RUNNING", active_run_id=None)
                if (summary.get("physics_valid") is not True
                        and summary.get("failure_category") != "SEED_CONTROL_UNVERIFIED"):
                    scene_failed = True
                    break
            if scene_failed:
                last = scene_runs[-1] if scene_runs else {}
                scene_status = ("PHYSICS_INVALID" if last.get("failure_category") == "PHYSICS_CONTRACT_VIOLATION"
                                else "UNRESOLVED_RUNTIME_OR_SEED_FAILURE")
            elif len(scene_runs) == len(plan["seeds_per_scene"]):
                signatures = [run.get("seeded_initial_conditions", {}).get("initial_condition_signature_sha256")
                              for run in scene_runs]
                distinct_seeded_conditions = len(set(signatures)) == len(plan["seeds_per_scene"])
                scene_status = (
                    "PHYSICS_VALID" if all(run.get("physics_valid") is True for run in scene_runs)
                    and distinct_seeded_conditions else "UNRESOLVED_SEED_OR_GATE"
                )
            else:
                scene_status = "UNRESOLVED_INCOMPLETE_SEEDS"
            current = _read_json(state_path)
            current.setdefault("scene_results", {})[scene_id] = {
                "physical_config_id": row["physical_config_id"],
                "status": scene_status,
                "verified_seed_count": len(scene_runs),
                "run_ids": [run.get("run_id") for run in scene_runs],
                "usd_sha256": sha256(scene_usd) if scene_usd.is_file() else None,
            }
            temporary = state_path.with_name(state_path.name + ".tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(current, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, state_path)
        execution_complete = True
        persist_state(status="COMPLETED_OR_SCENE_FAILURES_RECORDED", active_run_id=None)
        return 0
    finally:
        if "state_path" in locals() and state_path.is_file() and "total_started" in locals():
            try:
                persist_state(
                    status=("COMPLETED_OR_SCENE_FAILURES_RECORDED" if locals().get("execution_complete")
                            else "INTERRUPTED_REQUIRES_RESUME_REVIEW"),
                    active_run_id=None if locals().get("execution_complete") else _read_json(state_path).get("active_run_id"),
                )
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        try:
            current = _read_json(LOCK_PATH)
            if current.get("token") == token and current.get("pid") == os.getpid():
                LOCK_PATH.unlink()
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run only after all exact-hash approvals pass")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--approval-file", type=Path)
    args = parser.parse_args(argv)
    state = preflight(args.root, approval_path=args.approval_file)
    if not args.execute:
        print(json.dumps(state, indent=2, ensure_ascii=False))
        return 0 if state["ready"] else 2
    if not state["ready"]:
        print(json.dumps(state, indent=2, ensure_ascii=False), file=sys.stderr)
        print("STAGE_2A_EXECUTION_REFUSED; no output directory or Kit process created", file=sys.stderr)
        return 3
    try:
        return execute(args.root, approval_path=args.approval_file)
    except (PreflightError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
