"""Native USD and Physics-only worker for one approved BIMANUAL_V1 run.

Importing this module is offline-safe. Isaac Sim, Isaac Lab, Omni and USD are
loaded only after the supervisor has validated the exact approved invocation.
The worker never loads G1, FALCON, E1, a controller, or the V5 queue.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "research/round4070_comprehensive_traversability"
MASTER_PLAN = REPORT / "BIMANUAL_STAGE2_KIT_PHYSICS_PLAN_20261109.json"
DOOR_CONTRACT = REPORT / "BIMANUAL_STAGE2_DOOR_ACCEPTANCE_CONTRACT_20261109.json"
DT = 0.005
METADATA_KEY = "round4070BundleMetadataJson"
SCHEMA = "round4070-g.bimanual.stage2a.run.v1"
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def inside(root: Path, path: Path) -> Path:
    candidate = path.resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes repository: {path}")
    return candidate


def base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--scene-asset", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--usd-path", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--author-usd", action="store_true")
    mode.add_argument("--open-existing-usd", action="store_true")
    parser.add_argument("--expected-usd-sha256", default="")
    return parser


def validate_invocation(args: argparse.Namespace) -> dict[str, Any]:
    plan_path = inside(ROOT, args.plan)
    asset_path = inside(ROOT, args.scene_asset)
    run_root = inside(ROOT, args.run_root)
    usd_path = inside(ROOT, args.usd_path)
    master = read_json(MASTER_PLAN)
    plan = read_json(plan_path)
    asset = read_json(asset_path)
    row = next((item for item in master.get("run_matrix", []) if item.get("scene_id") == args.scene_id), None)
    if row is None:
        raise ValueError("scene is outside the approved Stage 2A matrix")
    if sha256(plan_path) != row.get("plan_sha256"):
        raise ValueError("scene plan hash does not match the approved matrix")
    if plan.get("source_scene_asset") != str(asset_path.relative_to(ROOT)) and Path(
            str(plan.get("source_scene_asset", ""))).as_posix() != asset_path.relative_to(ROOT).as_posix():
        raise ValueError("scene asset path does not match the frozen dynamic plan")
    scene = asset.get("scene_config", {})
    if (plan.get("scene_id") != args.scene_id or asset.get("scene_id") != args.scene_id
            or scene.get("scene_id") != args.scene_id):
        raise ValueError("scene identity mismatch")
    if scene.get("physical_config_id") != row.get("physical_config_id"):
        raise ValueError("physical_config_id mismatch")
    if args.seed not in master.get("seeds_per_scene", []):
        raise ValueError("seed is outside the approved test-only namespace")
    expected_run_id = f"BIV1A_{args.scene_id}_seed{args.seed}_attempt01"
    if args.run_id != expected_run_id:
        raise ValueError("run_id is not the approved append-only identifier")
    output_root = inside(ROOT, ROOT / master["scope"]["output_root"])
    if not run_root.is_relative_to(output_root) or not usd_path.is_relative_to(output_root):
        raise ValueError("run or USD path is outside the approved output root")
    if args.author_usd and usd_path.exists():
        raise ValueError("author-usd refuses to overwrite an existing USD")
    if args.open_existing_usd:
        if not usd_path.is_file():
            raise ValueError("open-existing-usd requires an existing file")
        if not args.expected_usd_sha256 or sha256(usd_path) != args.expected_usd_sha256:
            raise ValueError("existing USD hash mismatch")
    protected_paths = {
        "pause_flag": ROOT / "outputs/round4070_comprehensive_traversability/PAUSE.flag",
        "pause_g1_flag": ROOT / "outputs/round4070_comprehensive_traversability/PAUSE_G1.flag",
    }
    for name, path in protected_paths.items():
        if not path.is_file() or sha256(path) != master["protected_hashes"][name]:
            raise ValueError("protected input mismatch: " + name)
    if not DOOR_CONTRACT.is_file():
        raise ValueError("door acceptance contract is missing")
    door_contract = read_json(DOOR_CONTRACT)
    door_asset_path = inside(ROOT, ROOT / door_contract.get("scene_asset_path", ""))
    door_plan_path = inside(ROOT, ROOT / door_contract.get("scene_plan_path", ""))
    if (door_contract.get("schema") != "round4070-g.bimanual.door-acceptance-contract.v1"
            or door_contract.get("status") != "PREREGISTERED"
            or door_contract.get("scene_id") != "BI_G_DOOR_ORDINARY"
            or tuple(door_contract.get("required_checks", [])) != REQUIRED_DOOR_CHECKS
            or not isinstance(door_contract.get("thresholds"), dict)
            or not door_asset_path.is_file() or not door_plan_path.is_file()
            or sha256(door_asset_path) != door_contract.get("scene_asset_sha256")
            or sha256(door_plan_path) != door_contract.get("scene_plan_sha256")):
        raise ValueError("unsupported door acceptance contract")
    if (args.scene_id == door_contract["scene_id"]
            and (door_contract.get("scene_asset_sha256") != sha256(asset_path)
                 or door_contract.get("scene_plan_sha256") != sha256(plan_path))):
        raise ValueError("door contract does not bind the frozen door scene asset and plan")
    if master.get("source_hashes", {}).get("door_acceptance_contract") != sha256(DOOR_CONTRACT):
        raise ValueError("door acceptance contract hash is not pinned in the approved plan")
    return {
        "master": master,
        "scene_plan": plan,
        "asset": asset,
        "scene": scene,
        "row": row,
        "run_root": run_root,
        "usd_path": usd_path,
        "door_contract": door_contract,
        "door_contract_sha256": sha256(DOOR_CONTRACT),
    }


def require_supervisor_authority(args: argparse.Namespace) -> None:
    """Reject direct worker launches that bypass the approved campaign runner."""
    sys.path.insert(0, str(ROOT / "scripts/round4070_g"))
    import run_bimanual_kit_physics_only as supervisor

    state = supervisor.preflight(ROOT)
    if not state.get("ready"):
        raise RuntimeError("supervisor preflight rejected worker launch: " + ",".join(state.get("blockers", [])))
    master = read_json(MASTER_PLAN)
    row = next(item for item in master["run_matrix"] if item["scene_id"] == args.scene_id)
    run_id = f"BIV1A_{args.scene_id}_seed{args.seed}_attempt01"
    if args.run_id != run_id:
        raise RuntimeError("worker run_id is outside the approved namespace")
    output_root = inside(ROOT, ROOT / master["scope"]["output_root"])
    expected_run = output_root / "runs" / run_id
    expected_usd = output_root / "usd" / f"{args.scene_id}_{row['plan_sha256'][:12]}.usd"
    expected_scene_plan = inside(ROOT, ROOT / row["plan_path"])
    scene_plan = read_json(expected_scene_plan)
    expected_scene_asset = inside(ROOT, ROOT / scene_plan["source_scene_asset"])
    if inside(ROOT, args.run_root) != expected_run.resolve():
        raise RuntimeError("worker run root was not assigned by the campaign runner")
    if inside(ROOT, args.usd_path) != expected_usd.resolve():
        raise RuntimeError("worker USD path was not assigned by the campaign runner")
    if inside(ROOT, args.plan) != expected_scene_plan.resolve():
        raise RuntimeError("worker scene plan is outside the approved matrix")
    if inside(ROOT, args.scene_asset) != expected_scene_asset.resolve():
        raise RuntimeError("worker scene asset is outside the approved matrix")
    lock_path = ROOT / "outputs/round4070_comprehensive_traversability/resource_lock.json"
    state_path = output_root / "runner_state.json"
    if not lock_path.is_file() or not state_path.is_file():
        raise RuntimeError("supervisor lock/state is missing")
    lock = read_json(lock_path)
    runner_state = read_json(state_path)
    parent_pid = os.getppid()
    if lock.get("owner") != "bimanual_stage2a" or lock.get("pid") != parent_pid:
        raise RuntimeError("worker parent does not own the Stage 2A resource lock")
    if runner_state.get("status") != "RUNNING" or runner_state.get("active_run_id") != run_id:
        raise RuntimeError("runner state does not authorize this active run")
    if runner_state.get("plan_sha256") != sha256(MASTER_PLAN):
        raise RuntimeError("runner state plan hash mismatch")
    if runner_state.get("runner_sha256") != sha256(ROOT / "scripts/round4070_g/run_bimanual_kit_physics_only.py"):
        raise RuntimeError("runner state source hash mismatch")
    try:
        import psutil

        parent = psutil.Process(parent_pid)
        command_line = parent.cmdline()
        if not any(Path(argument.strip('"')).name.casefold() == "run_bimanual_kit_physics_only.py"
                   for argument in command_line):
            raise RuntimeError("worker was not launched by the approved runner process")
    except ImportError as exc:
        raise RuntimeError("process ancestry cannot be verified without psutil") from exc


def quat_yaw(quaternion_wxyz: np.ndarray) -> float:
    w, x, y, z = np.asarray(quaternion_wxyz, dtype=float)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def oriented_boxes_overlap(center_a: np.ndarray, quaternion_a_wxyz: np.ndarray, half_a: np.ndarray,
                           center_b: np.ndarray, quaternion_b_wxyz: np.ndarray, half_b: np.ndarray,
                           rotation_fn, margin_a: float = 0.0, margin_b: float = 0.0) -> bool:
    axes_a = rotation_fn(quaternion_a_wxyz)
    axes_b = rotation_fn(quaternion_b_wxyz)
    half_a = np.maximum(np.asarray(half_a, dtype=float) + margin_a, 0.0)
    half_b = np.maximum(np.asarray(half_b, dtype=float) + margin_b, 0.0)
    delta = np.asarray(center_b, dtype=float) - np.asarray(center_a, dtype=float)
    axes = [axes_a[:, index] for index in range(3)] + [axes_b[:, index] for index in range(3)]
    axes += [np.cross(axes_a[:, i], axes_b[:, j]) for i in range(3) for j in range(3)]
    for axis in axes:
        if np.linalg.norm(axis) < 1e-9:
            continue
        radius = np.sum(half_a * np.abs(axes_a.T @ axis)) + np.sum(half_b * np.abs(axes_b.T @ axis))
        if abs(np.dot(delta, axis)) >= radius:
            return False
    return True


def oriented_box_overlap(center, quaternion_wxyz, half, wall, rotation_fn, margin=0.0) -> bool:
    return oriented_boxes_overlap(
        center, quaternion_wxyz, np.full(3, half), wall["center_xyz"], np.array([1.0, 0.0, 0.0, 0.0]),
        np.asarray(wall["size_xyz"], dtype=float) / 2.0, rotation_fn, margin_b=margin)


def resource_sample() -> dict[str, Any]:
    row: dict[str, Any] = {"wall_time_utc": time.time()}
    try:
        import psutil

        process = psutil.Process()
        memory = process.memory_info()
        row.update(process_rss_mib=memory.rss / 1024**2, process_vms_mib=memory.vms / 1024**2,
                   process_cpu_percent=process.cpu_percent(interval=None), system_ram_percent=psutil.virtual_memory().percent)
    except Exception as exc:  # telemetry failure must be visible, not fatal to evidence capture
        row["process_telemetry_error"] = type(exc).__name__ + ": " + str(exc)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=10,
        )
        values = [value.strip() for value in result.stdout.splitlines()[0].split(",")]
        row.update(gpu_memory_used_mib=float(values[0]), gpu_utilization_percent=float(values[1]),
                   gpu_temperature_c=float(values[2]), gpu_power_w=float(values[3]))
    except Exception as exc:
        row["gpu_telemetry_error"] = type(exc).__name__ + ": " + str(exc)
    return row


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_usd_stage(stage, context: dict[str, Any]) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    from pxr import PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    metadata = context["binding_metadata"]
    checks: dict[str, Any] = {
        "stage_units_m": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "stage_up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "physics_scene_count": sum(1 for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)),
        "mesh_checks": [],
        "attachment_checks": [],
        "door_checks": {},
        "collision_prim_offsets": [],
    }
    valid = checks["stage_units_m"] == 1.0 and checks["stage_up_axis"].upper() == "Z" and checks["physics_scene_count"] == 1
    ground = stage.GetPrimAtPath("/World/defaultGroundPlane")
    checks["ground_collision"] = bool(ground and ground.HasAPI(UsdPhysics.CollisionAPI))
    valid &= checks["ground_collision"]
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            attributes = {}
            for name in ("physxCollision:contactOffset", "physxCollision:restOffset"):
                attribute = prim.GetAttribute(name)
                if attribute:
                    attributes[name] = attribute.Get()
            checks["collision_prim_offsets"].append({"path": str(prim.GetPath()), "attributes": attributes})
    for info in metadata.get("infos", []):
        prim = stage.GetPrimAtPath(info["prim_path"])
        applied_schemas = list(prim.GetAppliedSchemas()) if prim else []
        try:
            bound_material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics") if prim else (None, None)
        except Exception:
            bound_material = None
        mesh_check = {
            "path": info["prim_path"],
            "exists": bool(prim),
            "is_mesh": bool(prim and prim.IsA(UsdGeom.Mesh)),
            "collision_api": bool(prim and prim.HasAPI(UsdPhysics.CollisionAPI)),
            "solver_iterations_authored": bool(prim and prim.GetAttribute("omniphysics:solverPositionIterationCount")),
            "material_exists": bool(stage.GetPrimAtPath(info["material_path"])),
            "deformable_schema_applied": any("deformable" in str(schema).casefold() for schema in applied_schemas),
            "deformable_schema_names": [str(schema) for schema in applied_schemas if "deformable" in str(schema).casefold()],
            "physics_material_bound": bool(bound_material and str(bound_material.GetPath()) == info["material_path"]),
        }
        mesh_check["pass"] = all(mesh_check[key] for key in (
            "exists", "is_mesh", "collision_api", "solver_iterations_authored", "material_exists",
            "deformable_schema_applied", "physics_material_bound"))
        checks["mesh_checks"].append(mesh_check)
        valid &= mesh_check["pass"]
        for path in info.get("attachment_paths", []):
            attachment = stage.GetPrimAtPath(path)
            attachment_check = {
                "path": path,
                "exists": bool(attachment),
                "vertex_indices": bool(attachment and attachment.GetAttribute("omniphysics:vtxIndicesSrc0")),
                "overlap_offset": bool(attachment and attachment.GetAttribute(
                    "physxAutoDeformableAttachment:deformableVertexOverlapOffset")),
            }
            expected = next((constraint.get("local_vertex_indices", []) for constraint in metadata.get("constraints", [])
                             if constraint.get("attachment_path") == path), None)
            actual = attachment.GetAttribute("omniphysics:vtxIndicesSrc0").Get() if attachment_check["vertex_indices"] else None
            attachment_check["vertex_indices_match_metadata"] = bool(
                expected is not None and actual is not None and sorted(map(int, actual)) == sorted(map(int, expected)))
            attachment_check["pass"] = all(attachment_check[key] for key in (
                "exists", "vertex_indices", "overlap_offset", "vertex_indices_match_metadata"))
            checks["attachment_checks"].append(attachment_check)
            valid &= attachment_check["pass"]
    door = metadata.get("door")
    if door:
        body = stage.GetPrimAtPath(door["door_path"])
        panel = stage.GetPrimAtPath(door["panel_path"])
        joint_prim = stage.GetPrimAtPath(door["joint_path"])
        joint = UsdPhysics.RevoluteJoint(joint_prim) if joint_prim else None
        door_checks = {
            "body_rigid": bool(body and body.HasAPI(UsdPhysics.RigidBodyAPI)),
            "panel_collision": bool(panel and panel.HasAPI(UsdPhysics.CollisionAPI)),
            "joint_is_revolute": bool(joint_prim and joint_prim.IsA(UsdPhysics.RevoluteJoint)),
            "axis_z": bool(joint and str(joint.GetAxisAttr().Get()).upper() == "Z"),
            "lower_limit_deg": None if not joint else float(joint.GetLowerLimitAttr().Get()),
            "upper_limit_deg": None if not joint else float(joint.GetUpperLimitAttr().Get()),
            "drive_max_force": None if not joint_prim else UsdPhysics.DriveAPI(joint_prim, "angular").GetMaxForceAttr().Get(),
            "drive_damping": None if not joint_prim else UsdPhysics.DriveAPI(joint_prim, "angular").GetDampingAttr().Get(),
        }
        plan_component = next(item["plan"] for item in context["scene_plan"]["components"]
                              if item["component_type"] == "HingedDoorBuilder")
        limit_deg = math.degrees(float(plan_component["hinge"]["nominal_open_angle_rad"]))
        if plan_component["hinge"]["opening_direction"] == "POSITIVE_Y":
            expected_limits = (0.0, limit_deg)
        else:
            expected_limits = (-limit_deg, 0.0)
        door_checks["joint_limits_match_scene_plan"] = bool(
            door_checks["lower_limit_deg"] is not None and door_checks["upper_limit_deg"] is not None
            and abs(door_checks["lower_limit_deg"] - expected_limits[0]) <= 1e-6
            and abs(door_checks["upper_limit_deg"] - expected_limits[1]) <= 1e-6)
        door_checks["drive_parameters_match_scene_plan"] = bool(
            door_checks["drive_max_force"] is not None and door_checks["drive_damping"] is not None
            and abs(float(door_checks["drive_max_force"]) - float(plan_component["resistance_model"]["max_force_nm"])) <= 1e-6
            and abs(float(door_checks["drive_damping"]) - float(plan_component["resistance_model"]["damping_nm_s_per_deg"])) <= 1e-9)
        panel_prim = stage.GetPrimAtPath(door["panel_path"])
        door_checks["contact_report_api_authored"] = bool(
            panel_prim and panel_prim.HasAPI(PhysxSchema.PhysxContactReportAPI))
        door_checks["pass"] = all(bool(door_checks[key]) for key in (
            "body_rigid", "panel_collision", "joint_is_revolute", "axis_z", "joint_limits_match_scene_plan",
            "drive_parameters_match_scene_plan", "contact_report_api_authored"))
        checks["door_checks"] = door_checks
        valid &= door_checks["pass"]
    solver = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene) or str(prim.GetPath()).startswith("/World/Bimanual"):
            values = {}
            for attribute in prim.GetAttributes():
                name = attribute.GetName()
                if any(token in name.lower() for token in ("solver", "iteration", "gpu", "broadphase", "contactoffset")):
                    try:
                        values[name] = str(attribute.Get())
                    except Exception:
                        values[name] = "UNREADABLE"
            if values:
                solver[str(prim.GetPath())] = values
    return bool(valid), checks, solver


def classify_failure(exc: BaseException) -> str:
    text = (type(exc).__name__ + ": " + str(exc)).casefold()
    if any(token in text for token in ("out of memory", "cuda error", "cuda out", "allocation")):
        return "RESOURCE_BLOCKED"
    if any(token in text for token in ("schema", "usd", "attachment", "binding", "scene identity", "hash")):
        return "SOURCE_OR_SCHEMA_ERROR"
    return "RUNTIME_OR_IMPLEMENTATION_ERROR"


def run_worker(args: argparse.Namespace, context: dict[str, Any]) -> dict[str, Any]:
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaacsim.core.utils.stage import create_new_stage, get_current_stage, open_stage, save_stage
    import omni.kit.app
    from omni.physx import get_physx_simulation_interface
    from pxr import Gf, PhysicsSchemaTools, PhysxSchema, UsdGeom, UsdPhysics

    sys.path.insert(0, str(ROOT / "scripts/round4070_g"))
    sys.path.insert(0, str(ROOT / "scripts/round4070_f"))
    from bimanual_dynamic_scene_builder import BimanualSceneComposer
    from physics_p1_audit import triangle_box_overlap
    from physics_scene_builder import array, bind_existing_scene, rotation, serialize_bundle

    scene = context["scene"]
    master = context["master"]
    usd_path = context["usd_path"]
    run_root = context["run_root"]
    run_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "run_id": args.run_id,
        "scene_id": args.scene_id,
        "physical_config_id": context["row"]["physical_config_id"],
        "seed": args.seed,
        "runtime_valid": False,
        "physics_valid": False,
        "usd_saved_and_reopened": False,
        "usd_schema_bindings_verified": False,
        "native_runtime_bindings_verified": False,
        "seeded_initial_conditions_verified": False,
        "usd_file_immutable_during_physics": False,
        "acceptance_contract_sha256": canonical_sha256(master["acceptance_contract"]),
        "door_acceptance_contract_sha256": context["door_contract_sha256"],
        "material_fidelity": scene["material_fidelity"],
        "real_material_validated": False,
        "g1_loaded": False,
        "controller_or_action_created": False,
        "formal_episode_created": False,
    }
    sim = None
    rows: list[dict[str, Any]] = []
    contacts: list[dict[str, Any]] = []
    door_rows: list[dict[str, Any]] = []
    resources: list[dict[str, Any]] = []
    snapshots: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    probe_poses: list[np.ndarray] = []
    support_rows: list[dict[str, Any]] = []
    subscription = None
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        if args.author_usd:
            create_new_stage()
            stage = get_current_stage()
            world = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(world.GetPrim())
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
            physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
            physics_scene.CreateGravityMagnitudeAttr().Set(9.81)
            PhysxSchema.PhysxSceneAPI.Apply(physics_scene.GetPrim())
            ground = sim_utils.CuboidCfg(
                size=(8.0, 8.0, 0.10),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            )
            ground.func("/World/defaultGroundPlane", ground, translation=(0.0, 0.0, -0.05))
            composed = BimanualSceneComposer().build_native(stage, context["asset"], prim_root="/World/Bimanual")
            binding = serialize_bundle(composed["fabric_bundle"], door=composed["door"], prim_root="/World/Bimanual")
            if composed["door"]:
                door_panel = stage.GetPrimAtPath(composed["door"]["panel_path"])
                door_contact_api = PhysxSchema.PhysxContactReportAPI.Apply(door_panel)
                door_contact_api.CreateThresholdAttr().Set(0.0)
            binding.update(
                master_plan_sha256=sha256(MASTER_PLAN),
                scene_plan_sha256=sha256(args.plan),
                scene_asset_sha256=sha256(args.scene_asset),
                door_contract_sha256=context["door_contract_sha256"],
            )
            root_prim = stage.GetPrimAtPath("/World/Bimanual")
            if not root_prim:
                raise RuntimeError("USD schema error: /World/Bimanual was not authored")
            root_prim.SetCustomDataByKey(METADATA_KEY, json.dumps(binding, ensure_ascii=True, separators=(",", ":")))
            usd_path.parent.mkdir(parents=True, exist_ok=True)
            if not save_stage(str(usd_path), save_and_reload_in_place=True):
                raise RuntimeError("USD save failed")
            if not open_stage(str(usd_path)):
                raise RuntimeError("USD explicit reopen failed")
        else:
            if not open_stage(str(usd_path)):
                raise RuntimeError("existing USD reopen failed")
        stage = get_current_stage()
        root_prim = stage.GetPrimAtPath("/World/Bimanual")
        raw_metadata = root_prim.GetCustomDataByKey(METADATA_KEY) if root_prim else None
        if not isinstance(raw_metadata, str):
            raise RuntimeError("USD binding metadata is missing after reopen")
        binding = json.loads(raw_metadata)
        if (binding.get("master_plan_sha256") != sha256(MASTER_PLAN)
                or binding.get("scene_plan_sha256") != sha256(args.plan)
                or binding.get("scene_asset_sha256") != sha256(args.scene_asset)
                or binding.get("door_contract_sha256") != context["door_contract_sha256"]):
            raise RuntimeError("USD binding metadata hash mismatch")
        context["binding_metadata"] = binding
        summary["usd_sha256"] = sha256(usd_path)
        if args.expected_usd_sha256 and summary["usd_sha256"] != args.expected_usd_sha256:
            raise RuntimeError("USD changed before physics")
        summary["usd_saved_and_reopened"] = True
        summary["usd_root_identifier"] = str(stage.GetRootLayer().realPath or stage.GetRootLayer().identifier)
        summary["usd_binding_metadata_sha256"] = canonical_sha256(binding)
        schema_ok, schema_checks, solver_attributes = verify_usd_stage(stage, context)
        summary["usd_schema_bindings_verified"] = schema_ok
        summary["usd_schema_checks"] = schema_checks
        summary["authored_solver_attributes"] = solver_attributes
        if not schema_ok:
            raise RuntimeError("USD schema or native binding verification failed")
        sim = SimulationContext(SimulationCfg(dt=DT, render_interval=4, device="cuda:0"))
        bundle = bind_existing_scene(stage, scene, binding)
        door_info = binding.get("door")
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                physics_scene = UsdPhysics.Scene(prim)
                physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
                physics_scene.CreateGravityMagnitudeAttr().Set(9.81)
        geometry = scene["geometry"]
        center_x = float(geometry["center_x_m"])
        center_y = float(geometry["center_y_m"])
        rng = np.random.default_rng(args.seed)
        reference_rng = np.random.default_rng(args.seed)
        probe_draw = rng.uniform([-0.008, -0.005], [0.008, 0.005])
        reference_probe_draw = reference_rng.uniform([-0.008, -0.005], [0.008, 0.005])
        probe_y = center_y + float(probe_draw[0])
        probe_z = 0.95 + float(probe_draw[1])
        probe_starts_on_door_front = bool(binding.get("door"))
        start = np.array([center_x + (0.30 if probe_starts_on_door_front else -0.30), probe_y, probe_z], dtype=float)
        probe = RigidObject(RigidObjectCfg(
            prim_path="/World/GPhysicsProbe",
            spawn=sim_utils.CuboidCfg(
                size=(0.12, 0.12, 0.12),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=0.5),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(start)),
        ))
        contact_api = PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath("/World/GPhysicsProbe"))
        contact_api.CreateThresholdAttr().Set(0.0)
        if door_info:
            door_contact_api = PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(door_info["door_path"]))
            door_contact_api.CreateThresholdAttr().Set(0.0)
            door_body = RigidObject(RigidObjectCfg(prim_path=door_info["door_path"], spawn=None))
        else:
            door_body = None
        contact_clock = {"time_s": 0.0}

        def on_contact(headers, data):
            for header in headers:
                actor0 = str(PhysicsSchemaTools.intToSdfPath(header.actor0))
                actor1 = str(PhysicsSchemaTools.intToSdfPath(header.actor1))
                collider0 = str(PhysicsSchemaTools.intToSdfPath(header.collider0))
                collider1 = str(PhysicsSchemaTools.intToSdfPath(header.collider1))
                impulse = 0.0
                minimum_separation = None
                for index in range(header.contact_data_offset, header.contact_data_offset + header.num_contact_data):
                    impulse += float(np.linalg.norm(np.asarray(data[index].impulse, dtype=float)))
                    separation = float(data[index].separation)
                    minimum_separation = separation if minimum_separation is None else min(minimum_separation, separation)
                contacts.append({
                    "time_s": float(contact_clock["time_s"]),
                    "actor0": actor0, "actor1": actor1,
                    "collider0": collider0, "collider1": collider1,
                    "contact_count": int(header.num_contact_data),
                    "impulse_sum_ns": impulse,
                    "minimum_separation_m": minimum_separation,
                })

        subscription = get_physx_simulation_interface().subscribe_full_contact_report_events(on_contact)
        resources.append(resource_sample())
        sim.reset()
        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
            "isaacsim.core.experimental.prims", True)
        deform = bundle.make_deformable()
        probe.reset()
        probe.update(DT)
        if door_body:
            door_body.reset()
            door_body.update(DT)
        if deform:
            initial = array(deform.get_nodal_positions()[0])[0].astype(float)
            rest = array(deform.get_nodal_positions()[2])[0].astype(float)
            deform.set_nodal_velocities(np.zeros((1, len(initial), 3), np.float32))
            velocity_draw = rng.uniform(-0.002, 0.002, size=(1, len(initial), 3)).astype(np.float32)
            reference_velocity_draw = reference_rng.uniform(-0.002, 0.002, size=(1, len(initial), 3)).astype(np.float32)
        else:
            initial = rest = np.zeros((0, 3), dtype=float)
            velocity_draw = reference_velocity_draw = np.zeros((1, 0, 3), dtype=np.float32)
        initial_condition_seed_verified = bool(
            np.array_equal(probe_draw, reference_probe_draw)
            and np.array_equal(velocity_draw, reference_velocity_draw))
        seed_signature = canonical_sha256({
            "probe_draw": probe_draw.tolist(),
            "velocity_draw_sha256": hashlib.sha256(velocity_draw.tobytes()).hexdigest(),
        })
        summary["seeded_initial_conditions"] = {
            "verified": initial_condition_seed_verified,
            "python_seed": args.seed,
            "numpy_pcg64_seed": args.seed,
            "torch_seed": args.seed,
            "physx_internal_rng_seed_controlled": False,
            "physx_rng_seed_verified": False,
            "seed_scope": "deterministic initial-condition perturbations only; no PhysX internal RNG API is used",
            "independence_basis": "fresh process/stage plus distinct hash-recorded probe offset and free-node velocity perturbations; not independent PhysX-internal randomness",
            "initial_condition_signature_sha256": seed_signature,
            "probe_draw": probe_draw.tolist(),
            "velocity_draw_sha256": hashlib.sha256(velocity_draw.tobytes()).hexdigest(),
        }
        summary["seeded_initial_conditions_verified"] = initial_condition_seed_verified
        summary["seeded_initial_conditions"]["perturbation_is_nonzero"] = bool(
            np.any(np.abs(probe_draw) > 0.0)
            and (not velocity_draw.size or np.any(np.abs(velocity_draw) > 0.0)))
        summary["seeded_initial_conditions_verified"] &= summary["seeded_initial_conditions"]["perturbation_is_nonzero"]
        summary["seeded_initial_conditions"]["verified"] = summary["seeded_initial_conditions_verified"]
        summary["native_runtime_bindings_verified"] = bool(
            probe.data is not None and (not binding.get("infos") or deform is not None)
            and (door_body is not None if door_info else True))
        if not summary["native_runtime_bindings_verified"]:
            raise RuntimeError("native runtime binding initialization failed")
        physics_context = sim.get_physics_context()
        physics_configuration = {
            "physics_dt_s": float(physics_context.get_physics_dt()),
            "solver_type": physics_context.get_solver_type(),
            "broadphase_type": physics_context.get_broadphase_type(),
            "gpu_dynamics_enabled": bool(physics_context.is_gpu_dynamics_enabled()),
            "ccd_enabled": bool(physics_context.is_ccd_enabled()),
            "stabilization_enabled": bool(physics_context.is_stablization_enabled()),
            "simulation_device": str(sim.device),
        }
        summary["solver_configuration"] = physics_configuration
        if abs(physics_configuration["physics_dt_s"] - DT) > 1e-9:
            raise RuntimeError("configured PhysX dt differs from the frozen 200 Hz contract")
        if binding.get("infos") and not physics_configuration["gpu_dynamics_enabled"]:
            raise RuntimeError("surface deformable stage initialized without GPU dynamics")
        sim.step(render=False)
        probe.update(DT)
        if door_body:
            door_body.update(DT)
        door_initial_yaw = None
        if door_body:
            door_initial_yaw = quat_yaw(array(door_body.data.root_quat_w)[0])
        count = round(float(master["acceptance_contract"]["duration_s"]) / DT)
        failure_reason = None
        for step in range(count + 1):
            t = step * DT
            contact_clock["time_s"] = t
            progress = np.clip((t - 6.0) / 4.0, 0.0, 1.0) if t < 11.0 else 1.0 - np.clip((t - 11.0) / 3.0, 0.0, 1.0)
            target = start.copy()
            target[0] += (-0.40 if probe_starts_on_door_front else 0.40) * progress
            if door_info:
                lateral_start = float(context["door_contract"]["lateral_push_start_s"])
                lateral_end = float(context["door_contract"]["lateral_push_end_s"])
                lateral_distance = float(context["door_contract"]["probe_lateral_displacement_m"])
                if t < lateral_start:
                    lateral_fraction = 0.0
                elif t < lateral_end:
                    lateral_fraction = (t - lateral_start) / (lateral_end - lateral_start)
                else:
                    lateral_fraction = 1.0 - np.clip((t - lateral_end) / 3.0, 0.0, 1.0)
                target[1] += lateral_distance * float(np.clip(lateral_fraction, 0.0, 1.0))
            if step == 800 and deform:
                constrained = {index for constraint in bundle.constraints for index in constraint["global_vertex_indices"]}
                if constrained:
                    velocity_draw[:, list(constrained)] = 0.0
                deform.set_nodal_velocities(velocity_draw)
            probe_position = array(probe.data.root_pos_w)[0]
            probe_quaternion = array(probe.data.root_quat_w)[0]
            probe_velocity = array(probe.data.root_lin_vel_w)[0]
            force = 80.0 * (target - probe_position) - 8.0 * probe_velocity
            force_norm = np.linalg.norm(force)
            if force_norm > float(master["acceptance_contract"]["probe_force_cap_n"]):
                force *= float(master["acceptance_contract"]["probe_force_cap_n"]) / force_norm
            if step % 4 == 0 or step == count:
                if deform:
                    current = array(deform.get_nodal_positions()[0])[0].astype(float)
                    node_velocity = array(deform.get_nodal_velocities())[0].astype(float)
                    finite = bool(np.isfinite(current).all() and np.isfinite(node_velocity).all())
                    top_drift, anchor_drift = bundle.constraint_drifts(current, initial) if finite else (999.0, 999.0)
                    node_speed = float(np.linalg.norm(node_velocity, axis=1).max()) if finite else 999.0
                    stretch = bundle.max_neighbor_stretch(current, rest) if finite else 999.0
                    below = int((current[:, 2] < -0.005).sum()) if finite else 999
                    displacement = float(np.linalg.norm(current - initial, axis=1).max()) if finite else 999.0
                else:
                    current = node_velocity = np.zeros((0, 3), dtype=float)
                    finite = True
                    top_drift = anchor_drift = node_speed = below = displacement = 0.0
                    stretch = 1.0
                penetration = touch = wall_penetration = 0
                for offset, face in zip(bundle.offsets, bundle.faces):
                    triangles = current[face + offset]
                    local = (triangles - probe_position) @ rotation(probe_quaternion)
                    penetration += int(triangle_box_overlap(local, np.zeros(3), half=0.05).sum())
                    touch += int(triangle_box_overlap(local, np.zeros(3), half=0.085).sum())
                    wall = bundle.evaluation_metadata["wall"]
                    if wall:
                        half = np.asarray(wall["size_xyz"], dtype=float) / 2.0 - 0.005
                        scaled = (triangles - np.asarray(wall["center_xyz"], dtype=float)) / half
                        wall_penetration += int(triangle_box_overlap(scaled, np.zeros(3), half=1.0).sum())
                wall = bundle.evaluation_metadata["wall"]
                wall_contact_shell_overlap = bool(
                    wall and oriented_box_overlap(probe_position, probe_quaternion, 0.07, wall, rotation, margin=0.0))
                wall_probe_penetration = bool(
                    wall and oriented_box_overlap(probe_position, probe_quaternion, 0.05, wall, rotation, margin=-0.005))
                probe_door_penetration = False
                if door_body and door_info:
                    door_position = array(door_body.data.root_pos_w)[0]
                    door_quaternion = array(door_body.data.root_quat_w)[0]
                    door_plan_component = next(item["plan"] for item in context["scene_plan"]["components"]
                                               if item["component_type"] == "HingedDoorBuilder")
                    offset_sign = 1.0 if door_plan_component["hinge"]["opening_direction"] == "POSITIVE_Y" else -1.0
                    panel_offset = np.array([0.0, offset_sign * float(door_plan_component["size_xyz_m"][1]) / 2.0,
                                             float(door_plan_component["size_xyz_m"][2]) / 2.0])
                    panel_center = door_position + rotation(door_quaternion) @ panel_offset
                    probe_door_penetration = oriented_boxes_overlap(
                    probe_position, probe_quaternion, np.full(3, 0.06), panel_center, door_quaternion,
                        np.asarray(door_plan_component["size_xyz_m"], dtype=float) / 2.0, rotation, margin_a=-0.005)
                support = bundle.support_metrics()
                support_rows.append({"time_s": t, "supports": support})
                rails = [item for item in support if item["mechanism"] == "PRISMATIC_RAIL"]
                rail_valid = all(abs(item["dy_m"]) <= 0.305 and item["transverse_drift_m"] <= 0.01
                                 and item["rotation_error_rad"] <= 0.01 for item in rails)
                probe_contacts = [event for event in contacts if 6.0 <= event["time_s"] <= 14.0
                                  and "/World/GPhysicsProbe" in {event["actor0"], event["actor1"],
                                                                 event["collider0"], event["collider1"]}]
                last_probe_contact = max((event["time_s"] for event in contacts
                                          if "/World/GPhysicsProbe" in {event["actor0"], event["actor1"],
                                                                         event["collider0"], event["collider1"]}),
                                         default=-1e9)
                release_clear = bool(t >= 18.0 and touch == 0 and not wall_contact_shell_overlap
                                     and np.linalg.norm(probe_position - start) < 0.02)
                row = {
                    "step": step, "time_s": t, "finite": finite,
                    "node_speed_mps": node_speed, "neighbor_stretch": stretch,
                    "top_relative_drift_m": top_drift, "anchor_relative_drift_m": anchor_drift,
                    "nodes_below_ground": below, "displacement_m": displacement,
                    "probe_interior_triangles": penetration, "wall_interior_triangles": wall_penetration,
                    "probe_wall_penetration": wall_probe_penetration,
                    "wall_contact_shell_overlap": wall_contact_shell_overlap,
                    "probe_door_penetration": probe_door_penetration,
                    "probe_touch_triangles": touch, "probe_contact_events_in_push": len(probe_contacts),
                    "probe_x_m": float(probe_position[0]), "probe_target_x_m": float(target[0]),
                    "probe_force_n": float(np.linalg.norm(force)), "rail_valid": rail_valid,
                    "release_clear": release_clear,
                }
                rows.append(row)
                if door_body:
                    door_body.update(DT)
                    door_quaternion = array(door_body.data.root_quat_w)[0]
                    door_angular_velocity = array(door_body.data.root_ang_vel_w)[0]
                    direction = 1.0 if context["door_contract"]["opening_direction"] == "POSITIVE_Y" else -1.0
                    signed_angle = direction * wrap_angle(quat_yaw(door_quaternion) - float(door_initial_yaw))
                    door_rows.append({
                        "step": step, "time_s": t, "signed_open_angle_rad": signed_angle,
                        "angular_speed_radps": float(np.linalg.norm(door_angular_velocity)),
                    })
                if step % 20 == 0:
                    snapshots.append(current.astype(np.float32))
                    velocities.append(node_velocity.astype(np.float32))
                    probe_poses.append(np.r_[t, probe_position, probe_quaternion])
                if not finite or node_speed > 5.0 or stretch > 4.0 or top_drift > 0.01 or anchor_drift > 0.01 or below > 0 or not rail_valid:
                    failure_reason = "NUMERICAL_OR_CONSTRAINT_GATE"
                    break
            if step < count:
                angular_velocity = array(probe.data.root_ang_vel_w)[0]
                torque = -0.02 * angular_velocity - 0.03 * (probe_quaternion[1:] * np.sign(probe_quaternion[0]))
                torque_norm = np.linalg.norm(torque)
                if torque_norm > 0.03:
                    torque *= 0.03 / torque_norm
                body_force = torch.as_tensor(force @ rotation(probe_quaternion), dtype=torch.float32,
                                             device=sim.device).reshape(1, 1, 3)
                body_torque = torch.as_tensor(torque @ rotation(probe_quaternion), dtype=torch.float32,
                                              device=sim.device).reshape(1, 1, 3)
                probe.set_external_force_and_torque(body_force, body_torque, is_global=False)
                probe.write_data_to_sim()
                sim.step(render=False)
                probe.update(DT)
                if door_body:
                    door_body.update(DT)
            if step % 200 == 0:
                sample = resource_sample()
                sample.update(step=step, simulation_time_s=t)
                resources.append(sample)
        duration_completed = bool(rows and rows[-1]["time_s"] >= float(master["acceptance_contract"]["duration_s"]))
        tail = [row for row in rows if row["time_s"] >= float(master["acceptance_contract"]["duration_s"]) - 4.0]
        probe_contact_seen = any(6.0 <= event["time_s"] <= 14.0 and "/World/GPhysicsProbe" in {
            event["actor0"], event["actor1"], event["collider0"], event["collider1"]} for event in contacts)
        geometric_contact_rows = [row for row in rows if 6.0 <= row["time_s"] <= 14.0
                                  and ((row["probe_touch_triangles"] > 0 and row["displacement_m"] > 0.02
                                        and row["probe_force_n"] > 0.05)
                                       or row["wall_contact_shell_overlap"])]
        geometric_contact_seen = bool(geometric_contact_rows)
        sim.reset()
        probe.reset()
        probe.update(DT)
        if door_body:
            door_body.reset()
            door_body.update(DT)
        if deform:
            reset_nodes = array(deform.get_nodal_positions()[0])[0].astype(float)
            reset_node_error = float(np.linalg.norm(reset_nodes - initial, axis=1).max())
        else:
            reset_node_error = 0.0
        reset_probe_error = float(np.linalg.norm(array(probe.data.root_pos_w)[0] - start))
        reset_door_error = 0.0
        if door_body:
            reset_door_error = abs(wrap_angle(quat_yaw(array(door_body.data.root_quat_w)[0]) - float(door_initial_yaw)))
        reset_contract = master["acceptance_contract"]["reset"]
        reset_reproducible = bool(
            reset_node_error <= float(reset_contract["node_max_error_m"])
            and reset_probe_error <= float(reset_contract["probe_max_error_m"])
            and reset_door_error <= float(reset_contract["door_angle_max_error_rad"]))
        checks = {
            "duration_completed": duration_completed,
            "finite": bool(rows and all(row["finite"] for row in rows)),
            "node_speed_bounded": bool(rows and all(row["node_speed_mps"] <= 5.0 for row in rows)),
            "neighbor_stretch": bool(rows and all(row["neighbor_stretch"] <= 4.0 for row in rows)),
            "fixed_or_support_drift": bool(rows and all(max(row["top_relative_drift_m"], row["anchor_relative_drift_m"]) <= 0.01 for row in rows)),
            "nodes_below_ground": bool(rows and all(row["nodes_below_ground"] == 0 for row in rows)),
            "sampled_interior_penetrations": bool(rows and all(row["probe_interior_triangles"] == 0
                                                               and row["wall_interior_triangles"] == 0
                                                               and not row["probe_wall_penetration"]
                                                               and not row["probe_door_penetration"] for row in rows)),
            "contact_demonstrated": bool(probe_contact_seen or geometric_contact_seen),
            "initial_hanging_stable": bool(
                (settled := [row for row in rows if 3.0 <= row["time_s"] <= 4.0])
                and max(row["node_speed_mps"] for row in settled) < 0.25),
            "rail_bounds_and_rotation": bool(rows and all(row["rail_valid"] for row in rows)),
            "release_clear": bool(tail and all(row["release_clear"] for row in tail)),
            "late_node_speed": bool(tail and max(row["node_speed_mps"] for row in tail) < 0.25),
            "reset_reproducible": reset_reproducible,
        }
        door_checks: dict[str, bool] = {}
        if door_info:
            contract = context["door_contract"]
            push = [row for row in door_rows if contract["push_window_s"][0] <= row["time_s"] <= contract["push_window_s"][1]]
            door_tail = [row for row in door_rows if row["time_s"] >= contract["release_tail_start_s"]]
            door_contact_seen = any(contract["push_window_s"][0] <= event["time_s"] <= contract["push_window_s"][1]
                                    and door_info["door_path"] in {event["actor0"], event["actor1"],
                                                                  event["collider0"], event["collider1"]}
                                    and "/World/GPhysicsProbe" in {event["actor0"], event["actor1"],
                                                                   event["collider0"], event["collider1"]}
                                    for event in contacts)
            peak_angle = max((row["signed_open_angle_rad"] for row in push), default=-math.inf)
            minimum_angle = min((row["signed_open_angle_rad"] for row in door_rows), default=math.inf)
            maximum_angle = max((row["signed_open_angle_rad"] for row in door_rows), default=-math.inf)
            door_checks = {
                "joint_limits_match_scene_plan": bool(schema_checks["door_checks"].get("joint_limits_match_scene_plan")),
                "drive_parameters_match_scene_plan": bool(schema_checks["door_checks"].get("drive_parameters_match_scene_plan")),
                "contact_report_api_authored": bool(schema_checks["door_checks"].get("contact_report_api_authored")),
                "contact_reported_between_probe_and_door": door_contact_seen,
                "opens_in_declared_direction": peak_angle >= float(contract["minimum_open_response_rad"]),
                "no_reverse_backdrive_beyond_tolerance": minimum_angle >= -float(contract["reverse_angle_tolerance_rad"]),
                "no_joint_limit_overshoot": maximum_angle <= float(contract["nominal_open_angle_rad"]) + float(contract["limit_overshoot_tolerance_rad"]),
                "release_angular_speed_bounded": bool(door_tail and max(row["angular_speed_radps"] for row in door_tail)
                                                       <= float(contract["late_angular_speed_max_radps"])),
                "reset_angle_reproducible": reset_door_error <= float(reset_contract["door_angle_max_error_rad"]),
            }
        resource_telemetry_complete = bool(resources and all(
            all(key in sample for key in ("gpu_memory_used_mib", "gpu_utilization_percent", "gpu_temperature_c",
                                          "process_rss_mib", "process_cpu_percent"))
            and "gpu_telemetry_error" not in sample and "process_telemetry_error" not in sample
            for sample in resources))
        runtime_valid = bool(schema_ok and summary["native_runtime_bindings_verified"] and rows
                             and resource_telemetry_complete)
        physics_checks_pass = bool(runtime_valid and duration_completed and all(checks.values())
                                   and all(door_checks.values() if door_checks else [True]))
        physx_seed_verified = summary["seeded_initial_conditions"].get("physx_rng_seed_verified") is True
        physics_valid = bool(physics_checks_pass and initial_condition_seed_verified and physx_seed_verified)
        if physics_valid:
            failure_category = None
        elif not resource_telemetry_complete:
            failure_category = "RESOURCE_BLOCKED"
        elif runtime_valid and not initial_condition_seed_verified:
            failure_category = "INITIAL_CONDITION_SEED_UNVERIFIED"
        elif physics_checks_pass and not physx_seed_verified:
            failure_category = "SEED_CONTROL_UNVERIFIED"
        elif runtime_valid:
            failure_category = "PHYSICS_CONTRACT_VIOLATION"
        else:
            failure_category = "RUNTIME_OR_IMPLEMENTATION_ERROR"
        summary.update(
            runtime_valid=runtime_valid,
            physics_valid=physics_valid,
            failure_category=failure_category,
            failure_reason=failure_reason,
            duration_completed=duration_completed,
            duration_actual_s=rows[-1]["time_s"] if rows else 0.0,
            physics_checks=checks,
            door_checks=door_checks,
            reset_evidence={
                "node_max_error_m": reset_node_error,
                "probe_position_error_m": reset_probe_error,
                "door_angle_error_rad": reset_door_error,
            },
            contact_report_source="PhysxContactReportAPI full contact events",
            contact_event_count=len(contacts),
            contact_evidence={
                "native_physx_contact_event_seen": bool(probe_contact_seen),
                "frozen_gate_geometric_contact_predicate_seen": bool(geometric_contact_seen),
                "geometric_predicate_max_displacement_m": max((row["displacement_m"] for row in geometric_contact_rows), default=0.0),
                "geometric_predicate_max_force_n": max((row["probe_force_n"] for row in geometric_contact_rows), default=0.0),
                "geometric_predicate_max_shell_triangles": max((row["probe_touch_triangles"] for row in geometric_contact_rows), default=0),
                "scope": "Original frozen physics_gate predicate preserved; native event telemetry is supplementary and door contact requires the actual PhysX event.",
            },
            solver_configuration=physics_configuration,
            resource_telemetry_complete=resource_telemetry_complete,
            sampled_collision_scope=master["acceptance_contract"]["sampled_collision_scope"],
            summary_metrics={
                key: max((row[key] for row in rows), default=None)
                for key in ("node_speed_mps", "neighbor_stretch", "top_relative_drift_m",
                            "anchor_relative_drift_m", "probe_interior_triangles", "wall_interior_triangles")
            },
        )
        summary["usd_file_immutable_during_physics"] = sha256(usd_path) == summary["usd_sha256"]
        if not summary["usd_file_immutable_during_physics"]:
            summary["runtime_valid"] = False
            summary["physics_valid"] = False
            summary["failure_category"] = "SOURCE_OR_SCHEMA_ERROR"
            summary["failure_reason"] = "USD_FILE_CHANGED_DURING_PHYSICS"
        if snapshots:
            np.savez_compressed(
                run_root / "physics_snapshots.npz",
                positions=np.asarray(snapshots), velocities=np.asarray(velocities),
                probe_time_pose=np.asarray(probe_poses), rest=rest,
            )
        write_json(run_root / "support_trajectory.json", support_rows)
        write_rows(run_root / "telemetry.csv", rows)
        write_rows(run_root / "door_telemetry.csv", door_rows)
        write_json(run_root / "contact_events.json", contacts)
        write_json(run_root / "resource_telemetry.json", resources)
    except Exception as exc:
        summary.update(
            runtime_valid=False,
            physics_valid=False,
            failure_category=classify_failure(exc),
            failure_reason=type(exc).__name__ + ": " + str(exc),
            traceback=traceback.format_exc(),
        )
        if usd_path.is_file():
            summary.setdefault("usd_sha256", sha256(usd_path))
        write_rows(run_root / "telemetry.csv", rows)
        write_rows(run_root / "door_telemetry.csv", door_rows)
        write_json(run_root / "contact_events.json", contacts)
        write_json(run_root / "resource_telemetry.json", resources)
    finally:
        if sim is not None:
            sim._disable_app_control_on_stop_handle = True
        summary["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(run_root / "summary.json", summary)
        subscription = None
    return summary


def main(argv: list[str] | None = None) -> int:
    offline_parser = base_parser()
    offline_args, _ = offline_parser.parse_known_args(argv)
    try:
        require_supervisor_authority(offline_args)
        validate_invocation(offline_args)
    except Exception as exc:
        print("STAGE2A_WORKER_PREFLIGHT_REFUSED_BEFORE_KIT: " + str(exc), file=sys.stderr)
        return 3
    from isaaclab.app import AppLauncher

    parser = base_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    if not getattr(args, "headless", False):
        print("STAGE2A_WORKER_REQUIRES_HEADLESS", file=sys.stderr)
        return 2
    context = validate_invocation(args)
    try:
        require_supervisor_authority(args)
        context = validate_invocation(args)
    except Exception as exc:
        print("STAGE2A_WORKER_PREFLIGHT_REFUSED_BEFORE_KIT: " + str(exc), file=sys.stderr)
        return 3
    app = AppLauncher(args, multi_gpu=False).app
    try:
        summary = run_worker(args, context)
        print(json.dumps({
            "run_id": args.run_id,
            "runtime_valid": summary.get("runtime_valid"),
            "physics_valid": summary.get("physics_valid"),
            "failure_category": summary.get("failure_category"),
        }, ensure_ascii=False), flush=True)
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
