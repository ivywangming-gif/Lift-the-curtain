"""Physics-only validation for independent split sheets and flexible strips.

All deformable state in this runner is evaluation-only.  No robot or
deployment policy is present, so a PASS is only a prerequisite for later G1
experiments and never a navigation or traversability result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from lift_curtain.experiments.stage_log import StageLogger
from research.round4070_problem_discovery.code.scene_specs import scene_by_name, scene_specs


DT = 0.005
TOP_DRIFT_TOLERANCE_M = 0.010
ANCHOR_DRIFT_TOLERANCE_M = 0.010
INITIAL_VELOCITY_TOLERANCE_MPS = 1.0e-3
MAX_SPEED_TOLERANCE_MPS = 5.0
MAX_DISPLACEMENT_TOLERANCE_M = 1.5
MAX_NEIGHBOR_STRETCH_RATIO = 4.0

pre_parser = argparse.ArgumentParser(add_help=False)
pre_parser.add_argument("--run-root", type=Path, required=True)
PRE_ARGS, _ = pre_parser.parse_known_args()
LOGGER = StageLogger(
    PRE_ARGS.run_root,
    phase_timeouts_s={
        "BEFORE_APP": 180.0,
        "AFTER_APP": 180.0,
        "AFTER_SCENE": 120.0,
        "AFTER_RESET": 120.0,
        "FIRST_PHYSICS_STEP": 120.0,
    },
)
LOGGER.mark("BEFORE_APP", details={"argv": sys.argv, "pid": os.getpid()})

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scene", choices=tuple(scene_specs()), required=True)
parser.add_argument("--mode", choices=("no_probe", "probe"), required=True)
parser.add_argument("--run-root", type=Path, required=True)
parser.add_argument("--steps", type=int, default=720)
parser.add_argument("--sample-every", type=int, default=20)
parser.add_argument("--heartbeat-every", type=int, default=100)
parser.add_argument("--matched-no-probe-summary", type=Path, default=None)
parser.add_argument("--seed", type=int, default=47)
AppLauncher.add_app_launcher_args(parser)
ARGS = parser.parse_args()
APP = AppLauncher(ARGS).app
LOGGER.mark("AFTER_APP")


def clean(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return clean(value.item())
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def process_resources(torch_module: Any | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        import psutil

        process = psutil.Process(os.getpid())
        memory = psutil.virtual_memory()
        result.update(
            process_rss_bytes=int(process.memory_info().rss),
            system_available_bytes=int(memory.available),
            system_total_bytes=int(memory.total),
        )
    except Exception as exc:
        result["psutil_error"] = f"{type(exc).__name__}: {exc}"
    if torch_module is not None and torch_module.cuda.is_available():
        result.update(
            cuda_allocated_bytes=int(torch_module.cuda.memory_allocated()),
            cuda_reserved_bytes=int(torch_module.cuda.memory_reserved()),
            cuda_max_allocated_bytes=int(torch_module.cuda.max_memory_allocated()),
        )
    return result


def set_gravity(stage: Any, magnitude_mps2: float) -> list[str]:
    from pxr import Gf, UsdPhysics

    scenes = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
    if not scenes:
        scenes = [UsdPhysics.Scene.Define(stage, "/physicsScene").GetPrim()]
    paths: list[str] = []
    for prim in scenes:
        scene = UsdPhysics.Scene(prim)
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(float(magnitude_mps2))
        paths.append(str(prim.GetPath()))
    return paths


def read_runtime(deformable: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    simulation, _, rest = deformable.get_nodal_positions()
    positions = as_numpy(simulation)[0].astype(np.float64)
    rest_positions = as_numpy(rest)[0].astype(np.float64)
    velocities = as_numpy(deformable.get_nodal_velocities())[0].astype(np.float64)
    return positions, velocities, rest_positions


def select_row(points: np.ndarray, high: bool) -> list[int]:
    value = float(np.max(points[:, 2]) if high else np.min(points[:, 2]))
    return np.flatnonzero(np.abs(points[:, 2] - value) <= 1.0e-5).astype(int).tolist()


def attachment_metadata(stage: Any, path: str) -> dict[str, Any]:
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return {"path": path, "exists": False, "low_level_vertex_indices": [], "descendant_count": 0}
    low_level_indices: list[int] = []
    descendants: list[str] = []
    prefix = path + "/"
    for item in stage.Traverse():
        item_path = str(item.GetPath())
        if not item_path.startswith(prefix):
            continue
        indices_attr = item.GetAttribute("omniphysics:vtxIndicesSrc0")
        indices = list(indices_attr.Get() or []) if indices_attr else []
        low_level_indices.extend(int(index) for index in indices)
        descendants.append(item_path)
    return {
        "path": path,
        "exists": True,
        "low_level_vertex_indices": sorted(set(low_level_indices)),
        "descendant_count": len(descendants),
        "descendants": descendants,
    }


def neighbor_stretch(positions: np.ndarray, rest: np.ndarray, columns: int, rows: int) -> float:
    ratios: list[float] = []
    for row in range(rows):
        for column in range(columns):
            index = row * columns + column
            if column + 1 < columns:
                other = index + 1
                ratios.append(float(np.linalg.norm(positions[other] - positions[index]) / max(np.linalg.norm(rest[other] - rest[index]), 1.0e-9)))
            if row + 1 < rows:
                other = index + columns
                ratios.append(float(np.linalg.norm(positions[other] - positions[index]) / max(np.linalg.norm(rest[other] - rest[index]), 1.0e-9)))
    return float(np.max(np.asarray(ratios, dtype=np.float64)))


def probe_pose(step: int, total_steps: int) -> np.ndarray:
    approach = max(1, int(round(total_steps * 0.20)))
    push = max(1, int(round(total_steps * 0.60)))
    retract = max(1, total_steps - approach - push)
    start = np.asarray((0.50, 0.0, 0.95), dtype=np.float64)
    contact_start = np.asarray((0.77, 0.0, 0.95), dtype=np.float64)
    contact_end = np.asarray((0.98, 0.0, 0.95), dtype=np.float64)
    if step < approach:
        alpha = step / max(1, approach - 1)
        return (1.0 - alpha) * start + alpha * contact_start
    if step < approach + push:
        alpha = (step - approach) / max(1, push - 1)
        return (1.0 - alpha) * contact_start + alpha * contact_end
    alpha = min(1.0, (step - approach - push) / max(1, retract - 1))
    return (1.0 - alpha) * contact_end + alpha * start


def save_visualization(run_root: Path, snapshots: dict[str, list[np.ndarray]], times_s: list[float], top_indices: dict[str, list[int]], anchor_indices: dict[str, list[int]]) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    picks = [0, len(times_s) // 2, len(times_s) - 1]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
    for axis, pick, label in zip(axes, picks, ("before", "during", "after")):
        for color, (name, values) in zip(colors, snapshots.items()):
            points = values[pick]
            axis.scatter(points[:, 1], points[:, 2], s=3, color=color, label=name)
            axis.scatter(points[top_indices[name], 1], points[top_indices[name], 2], s=8, color=color)
            if anchor_indices[name]:
                axis.scatter(points[anchor_indices[name], 1], points[anchor_indices[name], 2], s=18, marker="x", color=color)
        axis.set_title(f"{label}, t={times_s[pick]:.2f}s")
        axis.set_xlabel("world y (m)")
    axes[0].set_ylabel("world z (m)")
    axes[-1].legend(loc="best", fontsize=7)
    figure.tight_layout()
    path = run_root / "runtime_geometry.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return str(path)


def main() -> int:
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.sim.utils import get_current_stage

    from structural_curtain_asset import CurtainConfig as StructuralCurtainConfig, create_surface_sheet
    from lift_curtain.envs.task_scale_scene import FormalTaskSceneConfig

    run_root = ARGS.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema": "lift-the-curtain.round4070-flexible-scene-validation.v1",
        "stage": "ROUND4070_MULTI_SHEET_PHYSICS_SANITY",
        "scene": ARGS.scene,
        "mode": ARGS.mode,
        "status": "RUNTIME_BLOCKED",
        "physics_status": "PHYSICS_BLOCKED",
        "deployment_decision_inputs": [],
        "evaluation_only_fields": ["runtime_node_positions", "runtime_node_velocities", "attachment_vertex_indices"],
        "g1_navigation_claim": "NOT_APPLICABLE",
    }
    sim = None
    rows: list[dict[str, Any]] = []
    try:
        if min(ARGS.steps, ARGS.sample_every, ARGS.heartbeat_every) <= 0:
            raise ValueError("steps and sampling intervals must be positive")
        spec = scene_by_name(ARGS.scene)
        formal = FormalTaskSceneConfig()
        base_sheet_configs = {sheet.name: sheet.curtain_config(formal) for sheet in spec.sheets}
        sheet_configs = {}
        for name, value in base_sheet_configs.items():
            values = dict(vars(value))
            values["structural_topology"] = "A0_TOP_EDGE_ONLY"
            sheet_configs[name] = StructuralCurtainConfig(**values)
        config = {
            "scene": spec.to_dict(),
            "formal_scene": formal.to_dict(),
            "sheet_configs": {name: vars(value) for name, value in sheet_configs.items()},
            "physics_dt_s": DT,
            "steps": ARGS.steps,
            "sample_every": ARGS.sample_every,
            "mode": ARGS.mode,
            "hypothesis": "Independent sheets remain finite and top-fixed, and a kinematic probe produces a response distinguishable from matched gravity-only motion.",
            "changed_variable": "Only the declared number, width, lateral placement, overlap and explicit bottom anchor of independent sheets.",
            "acceptance_thresholds": {
                "top_drift_m": TOP_DRIFT_TOLERANCE_M,
                "anchor_drift_m": ANCHOR_DRIFT_TOLERANCE_M,
                "initial_velocity_mps": INITIAL_VELOCITY_TOLERANCE_MPS,
                "max_speed_mps": MAX_SPEED_TOLERANCE_MPS,
                "max_displacement_m": MAX_DISPLACEMENT_TOLERANCE_M,
                "max_neighbor_stretch_ratio": MAX_NEIGHBOR_STRETCH_RATIO,
            },
        }
        (run_root / "resolved_config.json").write_text(json.dumps(clean(config), indent=2) + "\n", encoding="utf-8")
        report.update(config)
        report["model_contract"] = "ARTICULATED_STRIP_SURROGATE"
        report["real_pvc_claim"] = False
        np.random.seed(ARGS.seed)
        torch.manual_seed(ARGS.seed)
        sim = SimulationContext(SimulationCfg(dt=DT, render_interval=4, device="cuda:0"))
        ground = sim_utils.CuboidCfg(
            size=(8.0, 8.0, 0.10),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.18, 0.18)),
        )
        ground.func("/World/defaultGroundPlane", ground, translation=(0.0, 0.0, -0.05))
        stage = get_current_stage()
        report["physics_scene_paths"] = set_gravity(stage, 9.81)
        infos: dict[str, Any] = {}
        for name, sheet_config in sheet_configs.items():
            infos[name] = create_surface_sheet(stage, f"/World/Round4070/{name}/Sheet", sheet_config, create_attachment=True)
        probe = None
        if ARGS.mode == "probe":
            probe = RigidObject(
                RigidObjectCfg(
                    prim_path="/World/Round4070Probe",
                    spawn=sim_utils.CuboidCfg(
                        size=(0.12, 0.12, 0.12),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True, max_depenetration_velocity=0.5),
                        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.1)),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.50, 0.0, 0.95)),
                )
            )
        pre_attachment = {name: {path: attachment_metadata(stage, path) for path in info.attachment_paths} for name, info in infos.items()}
        LOGGER.mark("AFTER_SCENE", details={"sheet_count": len(infos), "attachment_paths": {name: list(info.attachment_paths) for name, info in infos.items()}, **process_resources(torch)})

        sim.reset()
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate("isaacsim.core.experimental.prims", True)
        from isaacsim.core.experimental.prims import DeformablePrim

        deformables = {name: DeformablePrim(info.prim_path, deformable_type="surface") for name, info in infos.items()}
        initial_positions: dict[str, np.ndarray] = {}
        initial_velocities: dict[str, np.ndarray] = {}
        runtime_rest: dict[str, np.ndarray] = {}
        top_indices: dict[str, list[int]] = {}
        bottom_indices: dict[str, list[int]] = {}
        anchor_indices: dict[str, list[int]] = {}
        runtime_attachment: dict[str, dict[str, Any]] = {}
        for name, deformable in deformables.items():
            positions, _, _ = read_runtime(deformable)
            deformable.set_nodal_velocities(np.zeros((1, positions.shape[0], 3), dtype=np.float32))
            positions, velocities, rest = read_runtime(deformable)
            initial_positions[name] = positions
            initial_velocities[name] = velocities
            runtime_rest[name] = rest
            top_indices[name] = select_row(rest, True)
            bottom_indices[name] = select_row(rest, False)
            anchor_indices[name] = list(infos[name].requested_bottom_anchor_indices)
            runtime_attachment[name] = {path: attachment_metadata(stage, path) for path in infos[name].attachment_paths}
        LOGGER.mark("AFTER_RESET", details={"runtime_nodes": {name: int(value.shape[0]) for name, value in runtime_rest.items()}, **process_resources(torch)})
        if probe is not None:
            probe.reset(); probe.update(DT)
        sim.step(render=False)
        if probe is not None:
            probe.update(DT)
        first_finite = True
        for deformable in deformables.values():
            positions, velocities, _ = read_runtime(deformable)
            first_finite = first_finite and bool(np.isfinite(positions).all() and np.isfinite(velocities).all())
        LOGGER.mark("FIRST_PHYSICS_STEP", details={"finite": first_finite, **process_resources(torch)})

        snapshots: dict[str, list[np.ndarray]] = {name: [] for name in deformables}
        velocity_snapshots: dict[str, list[np.ndarray]] = {name: [] for name in deformables}
        times_s: list[float] = []
        probe_positions: list[np.ndarray] = []
        for step in range(ARGS.steps + 1):
            target = probe_pose(step, ARGS.steps) if probe is not None else None
            if step == 0 or step % ARGS.sample_every == 0 or step == ARGS.steps:
                aggregate = {
                    "step": step,
                    "time_s": step * DT,
                    "runtime_finite": True,
                    "velocity_finite": True,
                    "max_displacement_m": 0.0,
                    "free_region_max_displacement_m": 0.0,
                    "top_drift_m": 0.0,
                    "anchor_drift_m": 0.0,
                    "bottom_min_z_m": float("inf"),
                    "nodes_below_ground_tolerance": 0,
                    "max_speed_mps": 0.0,
                    "max_neighbor_stretch_ratio": 0.0,
                    "probe_x_m": None if target is None else float(target[0]),
                }
                sheet_metrics: dict[str, Any] = {}
                for name, deformable in deformables.items():
                    current, velocity, _ = read_runtime(deformable)
                    displacement = np.linalg.norm(current - initial_positions[name], axis=1)
                    excluded = set(top_indices[name] + anchor_indices[name])
                    free = [index for index in range(len(displacement)) if index not in excluded]
                    metrics = {
                        "runtime_finite": bool(np.isfinite(current).all()),
                        "velocity_finite": bool(np.isfinite(velocity).all()),
                        "max_displacement_m": float(np.max(displacement)),
                        "free_region_max_displacement_m": float(np.max(displacement[free])),
                        "top_drift_m": float(np.max(np.linalg.norm(current[top_indices[name]] - initial_positions[name][top_indices[name]], axis=1))),
                        "anchor_drift_m": 0.0 if not anchor_indices[name] else float(np.max(np.linalg.norm(current[anchor_indices[name]] - initial_positions[name][anchor_indices[name]], axis=1))),
                        "bottom_min_z_m": float(np.min(current[bottom_indices[name], 2])),
                        "nodes_below_ground_tolerance": int(np.count_nonzero(current[:, 2] < -0.005)),
                        "max_speed_mps": float(np.max(np.linalg.norm(velocity, axis=1))),
                        "max_neighbor_stretch_ratio": neighbor_stretch(current, runtime_rest[name], sheet_configs[name].columns, sheet_configs[name].rows),
                    }
                    sheet_metrics[name] = metrics
                    snapshots[name].append(current.copy())
                    velocity_snapshots[name].append(velocity.copy())
                    aggregate["runtime_finite"] = aggregate["runtime_finite"] and metrics["runtime_finite"]
                    aggregate["velocity_finite"] = aggregate["velocity_finite"] and metrics["velocity_finite"]
                    for key in ("max_displacement_m", "free_region_max_displacement_m", "top_drift_m", "anchor_drift_m", "nodes_below_ground_tolerance", "max_speed_mps", "max_neighbor_stretch_ratio"):
                        aggregate[key] = max(aggregate[key], metrics[key])
                    aggregate["bottom_min_z_m"] = min(aggregate["bottom_min_z_m"], metrics["bottom_min_z_m"])
                aggregate["sheet_metrics_json"] = json.dumps(clean(sheet_metrics), separators=(",", ":"))
                rows.append(aggregate)
                times_s.append(step * DT)
                if target is not None:
                    probe_positions.append(target.copy())
            if step < ARGS.steps:
                if probe is not None and target is not None:
                    pose = torch.tensor([target[0], target[1], target[2], 1.0, 0.0, 0.0, 0.0], device=sim.device, dtype=torch.float32).unsqueeze(0)
                    probe.write_root_pose_to_sim(pose)
                    probe.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device))
                    probe.write_data_to_sim()
                sim.step(render=False)
                if probe is not None:
                    probe.update(DT)
            if step > 0 and (step % ARGS.heartbeat_every == 0 or step == ARGS.steps):
                LOGGER.heartbeat(progress=step / ARGS.steps, details={"step": step, "sim_time_s": step * DT, **process_resources(torch)})

        npz_payload: dict[str, Any] = {"times_s": np.asarray(times_s), "probe_positions": np.asarray(probe_positions)}
        for name in deformables:
            npz_payload[f"runtime_positions__{name}"] = np.stack(snapshots[name])
            npz_payload[f"runtime_velocities__{name}"] = np.stack(velocity_snapshots[name])
            npz_payload[f"runtime_rest_positions__{name}"] = runtime_rest[name]
            npz_payload[f"top_indices__{name}"] = np.asarray(top_indices[name], dtype=np.int64)
            npz_payload[f"bottom_indices__{name}"] = np.asarray(bottom_indices[name], dtype=np.int64)
            npz_payload[f"anchor_indices__{name}"] = np.asarray(anchor_indices[name], dtype=np.int64)
        np.savez_compressed(run_root / "runtime_node_snapshots.npz", **npz_payload)
        telemetry_path = run_root / "telemetry.csv"
        with telemetry_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

        per_sheet: dict[str, Any] = {}
        for name in deformables:
            actual_anchor_indices: list[int] = []
            for path in infos[name].bottom_anchor_attachment_paths:
                actual_anchor_indices.extend(runtime_attachment[name][path]["low_level_vertex_indices"])
            actual_anchor_indices = sorted(set(actual_anchor_indices))
            top_low_level = runtime_attachment[name][infos[name].attachment_path]["low_level_vertex_indices"] if infos[name].attachment_path else []
            attachment_exact = bool(
                infos[name].attachment_path
                and set(top_low_level) == set(top_indices[name])
                and set(actual_anchor_indices) == set(anchor_indices[name])
                and all(meta["descendant_count"] > 0 for meta in runtime_attachment[name].values())
            )
            per_sheet[name] = {
                "attachment_vertex_set_exact": attachment_exact,
                "top_indices_geometric": top_indices[name],
                "top_indices_low_level": top_low_level,
                "requested_anchor_indices": anchor_indices[name],
                "actual_anchor_indices_low_level": actual_anchor_indices,
                "initial_max_speed_mps": float(np.max(np.linalg.norm(initial_velocities[name], axis=1))),
                "max_top_drift_m": max(json.loads(row["sheet_metrics_json"])[name]["top_drift_m"] for row in rows),
                "max_anchor_drift_m": max(json.loads(row["sheet_metrics_json"])[name]["anchor_drift_m"] for row in rows),
                "max_free_region_displacement_m": max(json.loads(row["sheet_metrics_json"])[name]["free_region_max_displacement_m"] for row in rows),
                "max_speed_mps": max(json.loads(row["sheet_metrics_json"])[name]["max_speed_mps"] for row in rows),
                "max_neighbor_stretch_ratio": max(json.loads(row["sheet_metrics_json"])[name]["max_neighbor_stretch_ratio"] for row in rows),
                "minimum_bottom_z_m": min(json.loads(row["sheet_metrics_json"])[name]["bottom_min_z_m"] for row in rows),
            }
        max_free_response = max(float(row["free_region_max_displacement_m"]) for row in rows)
        matched_control = None
        probe_exceeds_no_probe = ARGS.mode == "no_probe"
        control_path = None
        if ARGS.mode == "probe":
            control_path = ARGS.matched_no_probe_summary.resolve() if ARGS.matched_no_probe_summary else None
            if control_path is not None and control_path.is_file():
                matched_control = json.loads(control_path.read_text(encoding="utf-8"))
                probe_exceeds_no_probe = max_free_response > float(matched_control.get("max_free_region_displacement_m", 0.0)) + 1.0e-4
            else:
                probe_exceeds_no_probe = False
        acceptance = {
            "runtime_finite": all(bool(row["runtime_finite"] and row["velocity_finite"]) for row in rows),
            "initial_velocity_bounded": all(value["initial_max_speed_mps"] <= INITIAL_VELOCITY_TOLERANCE_MPS for value in per_sheet.values()),
            "attachment_vertex_sets_exact": all(value["attachment_vertex_set_exact"] for value in per_sheet.values()),
            "top_drift_bounded": float(max(row["top_drift_m"] for row in rows)) <= TOP_DRIFT_TOLERANCE_M,
            "anchor_drift_bounded": float(max(row["anchor_drift_m"] for row in rows)) <= ANCHOR_DRIFT_TOLERANCE_M,
            "speed_bounded": float(max(row["max_speed_mps"] for row in rows)) < MAX_SPEED_TOLERANCE_MPS,
            "displacement_bounded": float(max(row["max_displacement_m"] for row in rows)) < MAX_DISPLACEMENT_TOLERANCE_M,
            "neighbor_stretch_bounded": float(max(row["max_neighbor_stretch_ratio"] for row in rows)) < MAX_NEIGHBOR_STRETCH_RATIO,
            "no_gross_ground_penetration": max(int(row["nodes_below_ground_tolerance"]) for row in rows) == 0,
            "probe_exceeds_no_probe": bool(probe_exceeds_no_probe),
        }
        passed = all(acceptance.values())
        visualization = save_visualization(run_root, snapshots, times_s, top_indices, anchor_indices)
        report.update(
            status="PASS_PHYSICS_SANITY" if passed else "FAIL_PHYSICS_SANITY",
            physics_status="PHYSICS_PASS" if passed else "PHYSICS_BLOCKED",
            contact_status="CONTACT_PASS" if ARGS.mode == "probe" and probe_exceeds_no_probe else "CONTACT_BLOCKED" if ARGS.mode == "probe" else "NOT_RUN",
            runtime_status="RUNTIME_PASS",
            acceptance=acceptance,
            per_sheet=per_sheet,
            attachment_metadata_before_reset=pre_attachment,
            attachment_metadata_after_reset=runtime_attachment,
            matched_no_probe_summary=None if matched_control is None else str(control_path),
            max_free_region_displacement_m=max_free_response,
            max_displacement_m=max(float(row["max_displacement_m"]) for row in rows),
            max_speed_mps=max(float(row["max_speed_mps"]) for row in rows),
            max_neighbor_stretch_ratio=max(float(row["max_neighbor_stretch_ratio"]) for row in rows),
            telemetry_csv=str(telemetry_path),
            runtime_node_snapshots=str(run_root / "runtime_node_snapshots.npz"),
            runtime_geometry_visualization=visualization,
            resources_final=process_resources(torch),
        )
        (run_root / "summary.json").write_text(json.dumps(clean(report), indent=2) + "\n", encoding="utf-8")
        LOGGER.finalize(status="PASS" if passed else "FAIL", details={"summary": str(run_root / "summary.json")})
        report["stage_log"] = LOGGER.summary()
        (run_root / "summary.json").write_text(json.dumps(clean(report), indent=2) + "\n", encoding="utf-8")
        print(json.dumps(clean(report), indent=2))
        return 0 if passed else 2
    except Exception as exc:
        report.update(
            status="PHYSICS_SETUP_BLOCKED" if not rows else "RUNTIME_BLOCKED",
            physics_status="PHYSICS_BLOCKED",
            runtime_status="RUNTIME_BLOCKED",
            error_type=type(exc).__name__,
            error=str(exc),
            last_stage=LOGGER.last_stage,
            stage_log=LOGGER.summary(),
        )
        (run_root / "summary.json").write_text(json.dumps(clean(report), indent=2) + "\n", encoding="utf-8")
        LOGGER.finalize(status="BLOCKED", details={"error_type": type(exc).__name__, "error": str(exc)})
        print(json.dumps(clean(report), indent=2))
        return 2
    finally:
        if sim is not None:
            try:
                sim.stop(); sim.clear_instance()
            except Exception:
                pass
        try:
            APP.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
