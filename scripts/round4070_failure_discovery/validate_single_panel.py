"""Round 13 parameterized curtain physics sanity runner.

Runtime nodes and attachment internals are evaluation-only evidence. This
runner has no deployment controller and cannot contribute a navigation PASS.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from dataclasses import replace
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from lift_curtain.experiments.stage_log import StageLogger


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
parser.add_argument("--scenario", required=True)
parser.add_argument("--mode", choices=("no_probe", "probe"), required=True)
parser.add_argument("--run-root", type=Path, required=True)
parser.add_argument("--hidden-rigid-x", type=float, default=None, help="Round 14 H2 diagnostic center x override.")
parser.add_argument("--hidden-rigid-y", type=float, default=None)
parser.add_argument("--hidden-rigid-width", type=float, default=None)
parser.add_argument("--hidden-rigid-height", type=float, default=None)
parser.add_argument("--hidden-rigid-thickness", type=float, default=None)
parser.add_argument("--hidden-rigid-bottom-z", type=float, default=None)
parser.add_argument("--curtain-length", type=float, default=2.00)
parser.add_argument("--curtain-top-z", type=float, default=None)
parser.add_argument("--density-scale-multiplier", type=float, default=1.0)
parser.add_argument("--curtain-damping", type=float, default=None)
parser.add_argument("--curtain-bend-stiffness", type=float, default=None)
parser.add_argument("--curtain-stretch-stiffness", type=float, default=None)
parser.add_argument("--curtain-shear-stiffness", type=float, default=None)
parser.add_argument("--condition-id", type=str, default=None)
parser.add_argument("--steps", type=int, default=720)
parser.add_argument("--sample-every", type=int, default=20)
parser.add_argument("--heartbeat-every", type=int, default=100)
parser.add_argument("--matched-no-probe-summary", type=Path, default=None)
parser.add_argument("--seed", type=int, default=42)
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
            {
                "process_rss_bytes": int(process.memory_info().rss),
                "system_available_bytes": int(memory.available),
                "system_total_bytes": int(memory.total),
            }
        )
    except Exception as exc:
        result["psutil_error"] = f"{type(exc).__name__}: {exc}"
    if torch_module is not None and torch_module.cuda.is_available():
        result.update(
            {
                "cuda_allocated_bytes": int(torch_module.cuda.memory_allocated()),
                "cuda_reserved_bytes": int(torch_module.cuda.memory_reserved()),
                "cuda_max_allocated_bytes": int(torch_module.cuda.max_memory_allocated()),
            }
        )
    return result


def set_gravity(stage: Any, magnitude_mps2: float) -> list[str]:
    from pxr import Gf, UsdPhysics

    scenes = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
    if not scenes:
        scenes = [UsdPhysics.Scene.Define(stage, "/physicsScene").GetPrim()]
    paths = []
    for prim in scenes:
        scene = UsdPhysics.Scene(prim)
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(float(magnitude_mps2))
        paths.append(str(prim.GetPath()))
    return paths


def world_points(stage: Any, prim_path: str) -> np.ndarray:
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    points = UsdGeom.PointBased(prim).GetPointsAttr().Get(Usd.TimeCode.Default()) or []
    transform = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    return np.asarray(
        [[float(v[0]), float(v[1]), float(v[2])] for v in (transform.Transform(point) for point in points)],
        dtype=np.float64,
    )


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
        return {"path": path, "exists": False, "low_level_vertex_count": 0}
    descendants = []
    low_level_indices: list[int] = []
    prefix = path + "/"
    for item in stage.Traverse():
        item_path = str(item.GetPath())
        if not item_path.startswith(prefix):
            continue
        indices_attr = item.GetAttribute("omniphysics:vtxIndicesSrc0")
        indices = list(indices_attr.Get() or []) if indices_attr else []
        if indices:
            low_level_indices.extend(int(index) for index in indices)
        descendants.append(
            {
                "path": item_path,
                "type_name": item.GetTypeName(),
                "property_names": [str(name) for name in item.GetPropertyNames()],
                "vtx_indices_src0": indices,
                "src0": [str(value) for value in (item.GetRelationship("omniphysics:src0").GetTargets() or [])],
                "src1": [str(value) for value in (item.GetRelationship("omniphysics:src1").GetTargets() or [])],
            }
        )
    return {
        "path": path,
        "exists": True,
        "descendants": descendants,
        "descendant_count": len(descendants),
        "low_level_vertex_indices": sorted(set(low_level_indices)),
        "low_level_vertex_count": len(set(low_level_indices)),
    }


def neighbor_stretch(positions: np.ndarray, rest: np.ndarray, columns: int, rows: int) -> float:
    ratios: list[float] = []
    for row in range(rows):
        for column in range(columns):
            index = row * columns + column
            if column + 1 < columns:
                other = index + 1
                ratios.append(
                    float(
                        np.linalg.norm(positions[other] - positions[index])
                        / max(np.linalg.norm(rest[other] - rest[index]), 1.0e-9)
                    )
                )
            if row + 1 < rows:
                other = index + columns
                ratios.append(
                    float(
                        np.linalg.norm(positions[other] - positions[index])
                        / max(np.linalg.norm(rest[other] - rest[index]), 1.0e-9)
                    )
                )
    return float(np.max(np.asarray(ratios, dtype=np.float64)))


def probe_pose(step: int, total_steps: int) -> np.ndarray:
    approach = max(1, int(round(total_steps * 0.20)))
    push = max(1, int(round(total_steps * 0.60)))
    retract = max(1, total_steps - approach - push)
    start = np.asarray((0.50, 0.0, 0.95), dtype=np.float64)
    contact_start = np.asarray((0.77, 0.0, 0.95), dtype=np.float64)
    contact_end = np.asarray((0.98, 0.0, 0.95), dtype=np.float64)
    finish = np.asarray((0.50, 0.0, 0.95), dtype=np.float64)
    if step < approach:
        alpha = step / max(1, approach - 1)
        return (1.0 - alpha) * start + alpha * contact_start
    if step < approach + push:
        alpha = (step - approach) / max(1, push - 1)
        return (1.0 - alpha) * contact_start + alpha * contact_end
    alpha = min(1.0, (step - approach - push) / max(1, retract - 1))
    return (1.0 - alpha) * contact_end + alpha * finish


def save_visualization(
    run_root: Path,
    snapshots: np.ndarray,
    times_s: np.ndarray,
    top_indices: list[int],
    anchor_indices: list[int],
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    picks = [0, len(snapshots) // 2, len(snapshots) - 1]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    for axis, pick, label in zip(axes, picks, ("before", "during", "after")):
        points = snapshots[pick]
        axis.scatter(points[:, 0], points[:, 2], s=4, label="nodes")
        axis.scatter(points[top_indices, 0], points[top_indices, 2], s=9, label="top")
        if anchor_indices:
            axis.scatter(points[anchor_indices, 0], points[anchor_indices, 2], s=20, marker="x", label="corner")
        axis.set_title(f"{label}, t={times_s[pick]:.2f}s")
        axis.set_xlabel("world x (m)")
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

    from lift_curtain.envs.curtain_asset import create_surface_sheet
    from lift_curtain.envs.round13_scenarios import scenario_by_name
    from lift_curtain.envs.task_scale_scene import FormalTaskSceneConfig

    run_root = ARGS.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "stage": "ROUND13_PARAMETERIZED_CURTAIN_PHYSICS_SANITY",
        "scenario": ARGS.scenario,
        "mode": ARGS.mode,
        "status": "RUNTIME_BLOCKED",
        "physics_status": "PHYSICS_BLOCKED",
        "contact_status": "CONTACT_BLOCKED" if ARGS.mode == "probe" else "NOT_RUN",
        "runtime_status": "RUNTIME_BLOCKED",
        "deployment_decision_inputs": [],
        "evaluation_only_fields": [
            "runtime_node_positions",
            "runtime_node_velocities",
            "attachment_vertex_indices",
            "hidden_wall_geometry",
        ],
    }
    sim = None
    rows: list[dict[str, Any]] = []
    try:
        if min(ARGS.steps, ARGS.sample_every, ARGS.heartbeat_every) <= 0:
            raise ValueError("steps and sampling intervals must be positive")
        if not math.isfinite(float(ARGS.curtain_length)) or ARGS.curtain_length <= 0.0:
            raise ValueError("curtain-length must be finite and positive")
        curtain_top_z = float(ARGS.curtain_top_z) if ARGS.curtain_top_z is not None else float(ARGS.curtain_length) + 0.02
        if not math.isfinite(curtain_top_z) or abs(curtain_top_z - float(ARGS.curtain_length) - 0.02) > 1.0e-6:
            raise ValueError("curtain-top-z must keep the curtain bottom at 0.02 m")
        if not math.isfinite(float(ARGS.density_scale_multiplier)) or ARGS.density_scale_multiplier <= 0.0:
            raise ValueError("density-scale-multiplier must be finite and positive")
        scenario = scenario_by_name(ARGS.scenario)
        if scenario.scene_object != "curtain":
            raise ValueError("this runner accepts curtain scenarios only")
        if ARGS.hidden_rigid_x is not None:
            if scenario.hidden_rigid is None:
                raise ValueError("--hidden-rigid-x requires a hidden-rigid curtain scenario")
            if not math.isfinite(float(ARGS.hidden_rigid_x)):
                raise ValueError("--hidden-rigid-x must be finite")
            scenario = replace(
                scenario,
                hidden_rigid=replace(scenario.hidden_rigid, x_m=float(ARGS.hidden_rigid_x)),
            )
        if scenario.hidden_rigid is not None:
            hidden_updates = {}
            for field_name, value in (
                ("y_m", ARGS.hidden_rigid_y),
                ("width_m", ARGS.hidden_rigid_width),
                ("height_m", ARGS.hidden_rigid_height),
                ("thickness_m", ARGS.hidden_rigid_thickness),
                ("bottom_z_m", ARGS.hidden_rigid_bottom_z),
            ):
                if value is not None:
                    hidden_updates[field_name] = float(value)
            if hidden_updates:
                scenario = replace(scenario, hidden_rigid=replace(scenario.hidden_rigid, **hidden_updates))
        formal = FormalTaskSceneConfig(curtain_top_z_m=curtain_top_z, curtain_length_m=float(ARGS.curtain_length))
        curtain_config = scenario.curtain_config(formal)
        assert curtain_config is not None
        curtain_config = replace(
            curtain_config,
            density_kg_m3=float(curtain_config.density_kg_m3) * float(ARGS.density_scale_multiplier),
            damping=float(ARGS.curtain_damping) if ARGS.curtain_damping is not None else curtain_config.damping,
            bend_stiffness=float(ARGS.curtain_bend_stiffness) if ARGS.curtain_bend_stiffness is not None else curtain_config.bend_stiffness,
            stretch_stiffness=float(ARGS.curtain_stretch_stiffness) if ARGS.curtain_stretch_stiffness is not None else curtain_config.stretch_stiffness,
            shear_stiffness=float(ARGS.curtain_shear_stiffness) if ARGS.curtain_shear_stiffness is not None else curtain_config.shear_stiffness,
        )
        curtain_config.validate()
        config = {
            "scenario": scenario.to_dict(),
            "formal_scene": formal.to_dict(),
            "curtain_config": curtain_config.__dict__,
            "physics_dt_s": DT,
            "steps": ARGS.steps,
            "sample_every": ARGS.sample_every,
            "heartbeat_every": ARGS.heartbeat_every,
            "mode": ARGS.mode,
            "condition_id": ARGS.condition_id,
            "hypothesis": "The declared density/anchor change remains finite and physically initialized before any G1 decision experiment.",
            "changed_variable": "Only scenario density, bottom-corner anchor mode, or hidden rigid scene construction.",
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
        info = create_surface_sheet(
            stage,
            "/World/Round13Curtain/Sheet",
            curtain_config,
            create_attachment=True,
        )
        hidden_wall_path = None
        hidden_wall_front_x = None
        hidden_wall_back_x = None
        hidden_wall_initial_overlap = False
        if scenario.hidden_rigid is not None:
            hidden = scenario.hidden_rigid
            wall = sim_utils.CuboidCfg(
                size=(hidden.thickness_m, hidden.width_m, hidden.height_m),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.16, 0.12)),
            )
            hidden_wall_path = "/World/Round13HiddenRigid"
            wall.func(hidden_wall_path, wall, translation=hidden.center_xyz)
            hidden_wall_front_x = float(hidden.x_m - 0.5 * hidden.thickness_m)
            hidden_wall_back_x = float(hidden.x_m + 0.5 * hidden.thickness_m)
        probe = None
        if ARGS.mode == "probe":
            probe = RigidObject(
                RigidObjectCfg(
                    prim_path="/World/Round13Probe",
                    spawn=sim_utils.CuboidCfg(
                        size=(0.12, 0.12, 0.12),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            kinematic_enabled=True,
                            disable_gravity=True,
                            max_depenetration_velocity=0.5,
                        ),
                        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.1)),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.50, 0.0, 0.95)),
                )
            )
        pre_attachment = {path: attachment_metadata(stage, path) for path in info.attachment_paths}
        LOGGER.mark(
            "AFTER_SCENE",
            details={
                "attachment_paths": list(info.attachment_paths),
                "hidden_wall_path": hidden_wall_path,
                **process_resources(torch),
            },
        )

        sim.reset()
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate("isaacsim.core.experimental.prims", True)
        from isaacsim.core.experimental.prims import DeformablePrim

        deformable = DeformablePrim(info.prim_path, deformable_type="surface")
        initial_positions, raw_velocities, runtime_rest = read_runtime(deformable)
        deformable.set_nodal_velocities(np.zeros((1, initial_positions.shape[0], 3), dtype=np.float32))
        initial_positions, initial_velocities, runtime_rest = read_runtime(deformable)
        if hidden_wall_front_x is not None:
            hidden_wall_initial_overlap = bool(np.any(initial_positions[:, 0] >= hidden_wall_front_x - 1.0e-6))
        top_indices = select_row(runtime_rest, True)
        bottom_indices = select_row(runtime_rest, False)
        anchor_indices = list(info.requested_bottom_anchor_indices)
        runtime_attachment = {path: attachment_metadata(stage, path) for path in info.attachment_paths}
        LOGGER.mark(
            "AFTER_RESET",
            details={
                "runtime_nodes": int(runtime_rest.shape[0]),
                "top_indices": top_indices,
                "anchor_indices": anchor_indices,
                **process_resources(torch),
            },
        )

        if probe is not None:
            probe.reset()
            probe.update(DT)
        sim.step(render=False)
        if probe is not None:
            probe.update(DT)
        first_positions, first_velocities, _ = read_runtime(deformable)
        LOGGER.mark(
            "FIRST_PHYSICS_STEP",
            details={
                "finite": bool(np.isfinite(first_positions).all() and np.isfinite(first_velocities).all()),
                **process_resources(torch),
            },
        )

        snapshots: list[np.ndarray] = []
        velocities: list[np.ndarray] = []
        times_s: list[float] = []
        probe_positions: list[np.ndarray] = []
        for step in range(ARGS.steps + 1):
            target = probe_pose(step, ARGS.steps) if probe is not None else None
            if step == 0 or step % ARGS.sample_every == 0 or step == ARGS.steps:
                current, velocity, _ = read_runtime(deformable)
                displacement = np.linalg.norm(current - initial_positions, axis=1)
                snapshots.append(current.copy())
                velocities.append(velocity.copy())
                times_s.append(step * DT)
                if target is not None:
                    probe_positions.append(target.copy())
                rows.append(
                    {
                        "step": step,
                        "time_s": step * DT,
                        "runtime_finite": bool(np.isfinite(current).all()),
                        "velocity_finite": bool(np.isfinite(velocity).all()),
                        "max_displacement_m": float(np.max(displacement)),
                        "free_region_max_displacement_m": float(
                            np.max(displacement[[index for index in range(len(displacement)) if index not in set(top_indices + anchor_indices)]])
                        ),
                        "top_drift_m": float(
                            np.max(np.linalg.norm(current[top_indices] - initial_positions[top_indices], axis=1))
                        ),
                        "anchor_drift_m": 0.0 if not anchor_indices else float(
                            np.max(np.linalg.norm(current[anchor_indices] - initial_positions[anchor_indices], axis=1))
                        ),
                        "bottom_min_z_m": float(np.min(current[bottom_indices, 2])),
                        "nodes_below_ground_tolerance": int(np.count_nonzero(current[:, 2] < -0.005)),
                        "max_speed_mps": float(np.max(np.linalg.norm(velocity, axis=1))),
                        "max_neighbor_stretch_ratio": neighbor_stretch(
                            current, runtime_rest, curtain_config.columns, curtain_config.rows
                        ),
                        "probe_x_m": None if target is None else float(target[0]),
                        "min_hidden_wall_face_gap_m": (
                            None
                            if hidden_wall_front_x is None
                            else float(hidden_wall_front_x - np.max(current[:, 0]))
                        ),
                        "nodes_at_or_beyond_hidden_wall_front": (
                            0
                            if hidden_wall_front_x is None
                            else int(np.count_nonzero(current[:, 0] >= hidden_wall_front_x - 1.0e-4))
                        ),
                    }
                )
            if step < ARGS.steps:
                if probe is not None and target is not None:
                    pose = torch.tensor(
                        [target[0], target[1], target[2], 1.0, 0.0, 0.0, 0.0],
                        device=sim.device,
                        dtype=torch.float32,
                    ).unsqueeze(0)
                    probe.write_root_pose_to_sim(pose)
                    probe.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device))
                    probe.write_data_to_sim()
                sim.step(render=False)
                if probe is not None:
                    probe.update(DT)
            if step > 0 and (step % ARGS.heartbeat_every == 0 or step == ARGS.steps):
                LOGGER.heartbeat(
                    progress=step / ARGS.steps,
                    details={"step": step, "sim_time_s": step * DT, **process_resources(torch)},
                )

        snapshot_array = np.stack(snapshots)
        velocity_array = np.stack(velocities)
        np.savez_compressed(
            run_root / "runtime_node_snapshots.npz",
            times_s=np.asarray(times_s),
            runtime_positions=snapshot_array,
            runtime_velocities=velocity_array,
            runtime_rest_positions=runtime_rest,
            stage_rest_positions=world_points(stage, info.prim_path),
            top_indices=np.asarray(top_indices, dtype=np.int64),
            bottom_indices=np.asarray(bottom_indices, dtype=np.int64),
            requested_anchor_indices=np.asarray(anchor_indices, dtype=np.int64),
            probe_positions=np.asarray(probe_positions),
        )
        telemetry_path = run_root / "telemetry.csv"
        with telemetry_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        max_top_drift = max(float(row["top_drift_m"]) for row in rows)
        max_anchor_drift = max(float(row["anchor_drift_m"]) for row in rows)
        max_displacement = max(float(row["max_displacement_m"]) for row in rows)
        max_free_response = max(float(row["free_region_max_displacement_m"]) for row in rows)
        max_speed = max(float(row["max_speed_mps"]) for row in rows)
        max_stretch = max(float(row["max_neighbor_stretch_ratio"]) for row in rows)
        min_hidden_wall_face_gap = (
            None
            if hidden_wall_front_x is None
            else min(float(row["min_hidden_wall_face_gap_m"]) for row in rows)
        )
        max_nodes_at_or_beyond_hidden_wall_front = (
            0
            if hidden_wall_front_x is None
            else max(int(row["nodes_at_or_beyond_hidden_wall_front"]) for row in rows)
        )
        actual_anchor_indices: list[int] = []
        for path in info.bottom_anchor_attachment_paths:
            actual_anchor_indices.extend(runtime_attachment[path]["low_level_vertex_indices"])
        actual_anchor_indices = sorted(set(actual_anchor_indices))
        top_low_level_indices = runtime_attachment[info.attachment_path]["low_level_vertex_indices"] if info.attachment_path else []
        attachment_gate = bool(
            info.attachment_path
            and set(top_low_level_indices) == set(top_indices)
            and set(actual_anchor_indices) == set(anchor_indices)
            and all(meta["descendant_count"] > 0 for meta in runtime_attachment.values())
        )
        acceptance = {
            "runtime_finite": all(row["runtime_finite"] and row["velocity_finite"] for row in rows),
            "initial_velocity_bounded": float(np.max(np.linalg.norm(initial_velocities, axis=1))) <= INITIAL_VELOCITY_TOLERANCE_MPS,
            "attachment_vertex_set_exact": attachment_gate,
            "top_drift_bounded": max_top_drift <= TOP_DRIFT_TOLERANCE_M,
            "anchor_drift_bounded": max_anchor_drift <= ANCHOR_DRIFT_TOLERANCE_M,
            "speed_bounded": max_speed < MAX_SPEED_TOLERANCE_MPS,
            "displacement_bounded": max_displacement < MAX_DISPLACEMENT_TOLERANCE_M,
            "neighbor_stretch_bounded": max_stretch < MAX_NEIGHBOR_STRETCH_RATIO,
            "no_gross_ground_penetration": max(int(row["nodes_below_ground_tolerance"]) for row in rows) == 0,
        }
        if hidden_wall_front_x is not None:
            acceptance["hidden_wall_initial_nonoverlap"] = not hidden_wall_initial_overlap
            acceptance["hidden_wall_no_front_face_crossing"] = max_nodes_at_or_beyond_hidden_wall_front == 0
        matched_control = None
        if ARGS.mode == "probe":
            control_path = ARGS.matched_no_probe_summary.resolve() if ARGS.matched_no_probe_summary else None
            if control_path is not None and control_path.is_file():
                matched_control = json.loads(control_path.read_text(encoding="utf-8"))
                acceptance["probe_exceeds_no_probe"] = max_free_response > float(
                    matched_control.get("max_free_region_displacement_m", 0.0)
                ) + 1.0e-4
            else:
                acceptance["probe_exceeds_no_probe"] = False
        passed = all(acceptance.values())
        visualization = save_visualization(
            run_root,
            snapshot_array,
            np.asarray(times_s),
            top_indices,
            anchor_indices,
        )
        report.update(
            {
                "status": "PASS_PHYSICS_SANITY" if passed else "FAIL_PHYSICS_SANITY",
                "physics_status": "PHYSICS_PASS" if passed else "PHYSICS_BLOCKED",
                "contact_status": (
                    "CONTACT_PASS" if ARGS.mode == "probe" and acceptance.get("probe_exceeds_no_probe")
                    else "CONTACT_BLOCKED" if ARGS.mode == "probe"
                    else "NOT_RUN"
                ),
                "runtime_status": "RUNTIME_PASS",
                "acceptance": acceptance,
                "attachment_metadata_before_reset": pre_attachment,
                "attachment_metadata_after_reset": runtime_attachment,
                "top_vertex_indices_geometric": top_indices,
                "top_vertex_indices_low_level": top_low_level_indices,
                "bottom_left_vertex_index": info.bottom_left_index,
                "bottom_right_vertex_index": info.bottom_right_index,
                "requested_bottom_anchor_indices": anchor_indices,
                "actual_bottom_anchor_indices_low_level": actual_anchor_indices,
                "bottom_anchor_targets": list(info.bottom_anchor_support_paths),
                "matched_no_probe_summary": None if matched_control is None else str(control_path),
                "max_top_drift_m": max_top_drift,
                "max_anchor_drift_m": max_anchor_drift,
                "max_displacement_m": max_displacement,
                "max_free_region_displacement_m": max_free_response,
                "max_speed_mps": max_speed,
                "max_neighbor_stretch_ratio": max_stretch,
                "hidden_wall_front_x_m": hidden_wall_front_x,
                "hidden_wall_back_x_m": hidden_wall_back_x,
                "hidden_wall_initial_overlap": hidden_wall_initial_overlap,
                "min_hidden_wall_face_gap_m": min_hidden_wall_face_gap,
                "max_nodes_at_or_beyond_hidden_wall_front": max_nodes_at_or_beyond_hidden_wall_front,
                "minimum_bottom_z_m": min(float(row["bottom_min_z_m"]) for row in rows),
                "telemetry_csv": str(telemetry_path),
                "runtime_node_snapshots": str(run_root / "runtime_node_snapshots.npz"),
                "runtime_geometry_visualization": visualization,
                "stage_log": LOGGER.summary(),
                "resources_final": process_resources(torch),
            }
        )
        (run_root / "summary.json").write_text(json.dumps(clean(report), indent=2) + "\n", encoding="utf-8")
        LOGGER.finalize(status="PASS" if passed else "FAIL", details={"summary": str(run_root / "summary.json")})
        print(json.dumps(clean(report), indent=2))
        return 0 if passed else 2
    except Exception as exc:
        report.update(
            {
                "status": "PHYSICS_SETUP_BLOCKED" if not rows else "RUNTIME_BLOCKED",
                "physics_status": "PHYSICS_BLOCKED",
                "runtime_status": "RUNTIME_BLOCKED",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "last_stage": LOGGER.last_stage,
                "stage_log": LOGGER.summary(),
            }
        )
        (run_root / "summary.json").write_text(json.dumps(clean(report), indent=2) + "\n", encoding="utf-8")
        LOGGER.finalize(status="BLOCKED", details={"error_type": type(exc).__name__, "error": str(exc)})
        print(json.dumps(clean(report), indent=2))
        return 2
    finally:
        if sim is not None:
            try:
                sim.stop()
                sim.clear_instance()
            except Exception:
                pass
        try:
            APP.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
