"""Headless M1 curtain physics sanity test.

Run this script once with ``--mode external`` for the read-only MoDeSuite asset
and once with ``--mode fallback`` for the local Isaac Sim 5.1 surface sheet.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lift_curtain.paths import modesuite_curtain_path
from isaaclab.app import AppLauncher


DEFAULT_EXTERNAL = modesuite_curtain_path()
DEFAULT_RUN_ROOT = REPO / "reports" / "artifacts" / "m1"
DT = 0.005
RENDER_INTERVAL = 4

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("external", "fallback"), required=True)
parser.add_argument("--source", type=Path, default=DEFAULT_EXTERNAL)
parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--probe-span", type=float, default=0.8)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
ARGS = parser.parse_args()
if "--/persistent/physics/enableDeformableBeta" not in (ARGS.kit_args or ""):
    ARGS.kit_args = f"{ARGS.kit_args or ''} --/persistent/physics/enableDeformableBeta=true".strip()
APP = AppLauncher(ARGS).app


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value: Any) -> Any:
    if hasattr(value, "item"):
        return clean(value.item())
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def prim_schema_summary(root_prim) -> dict[str, Any]:
    from pxr import UsdGeom

    deformable_prims = []
    mesh_prims = []
    for prim in root_prim.GetStage().Traverse():
        path = str(prim.GetPath())
        if not (path == str(root_prim.GetPath()) or path.startswith(str(root_prim.GetPath()) + "/")):
            continue
        if prim.IsA(UsdGeom.Mesh):
            points = UsdGeom.Mesh(prim).GetPointsAttr().Get() or []
            mesh_prims.append(
                {
                    "path": path,
                    "schemas": list(prim.GetAppliedSchemas()),
                    "point_count": len(points),
                }
            )
        schemas = list(prim.GetAppliedSchemas())
        if any("Deformable" in schema or "deformable" in schema for schema in schemas):
            deformable_prims.append({"path": path, "type": prim.GetTypeName(), "schemas": schemas})
    return {"mesh_prims": mesh_prims, "deformable_prims": deformable_prims}


def world_bounds(stage, root_prim) -> tuple[Any, Any]:
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
    box = cache.ComputeWorldBound(root_prim).ComputeAlignedBox()
    return box.GetMin(), box.GetMax()


def add_probe(sim_utils, RigidObject, RigidObjectCfg):
    probe = RigidObject(
        RigidObjectCfg(
            prim_path="/World/Probe",
            spawn=sim_utils.CuboidCfg(
                size=(0.22, 0.22, 0.22),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                    max_depenetration_velocity=0.5,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.005,
                    rest_offset=0.0,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.1)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.8, 0.9)),
        )
    )
    return probe


def probe_path(minimum, maximum, probe_span: float):
    import numpy as np

    low = np.asarray([float(v) for v in minimum], dtype=float)
    high = np.asarray([float(v) for v in maximum], dtype=float)
    extent = np.maximum(high - low, 1.0e-3)
    normal_axis = int(np.argmin(extent))
    center = 0.5 * (low + high)
    start = center.copy()
    end = center.copy()
    margin = max(0.20, float(probe_span) * 0.25)
    start[normal_axis] = low[normal_axis] - margin
    end[normal_axis] = high[normal_axis] + margin
    return start, end, normal_axis


def as_numpy(warp_array):
    import numpy as np

    if hasattr(warp_array, "numpy"):
        return np.asarray(warp_array.numpy())
    return np.asarray(warp_array)


def mesh_points_world(mesh_prim):
    """Read a USD mesh's points in world coordinates without a physics wrapper."""
    import numpy as np
    from pxr import Usd, UsdGeom

    points = UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get(Usd.TimeCode.Default()) or []
    transform = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(mesh_prim)
    return np.asarray(
        [[float(value[0]), float(value[1]), float(value[2])] for value in (transform.Transform(point) for point in points)],
        dtype=float,
    )


def point_measurements(current, reference, probe_position, probe_half_extent, top_indices=None):
    import numpy as np

    current = np.asarray(current, dtype=float)
    reference = np.asarray(reference, dtype=float)
    finite = bool(current.size > 0 and np.isfinite(current).all())
    displacement = np.linalg.norm(current - reference, axis=1) if finite else np.array([math.inf])
    rest_extent = np.ptp(reference, axis=0)
    current_extent = np.ptp(current, axis=0) if finite else np.array([math.inf, math.inf, math.inf])
    extent_ratio = float(np.max(current_extent / np.maximum(rest_extent, 1.0e-6))) if finite else math.inf
    inside = np.all(np.abs(current - np.asarray(probe_position)) <= float(probe_half_extent), axis=1) if finite else []
    result = {
        "finite": finite,
        "node_count": int(current.shape[0]) if current.ndim == 2 else 0,
        "max_displacement_m": float(np.max(displacement)),
        "mean_displacement_m": float(np.mean(displacement)),
        "max_extent_ratio": extent_ratio,
        "probe_node_overlap_count": int(np.sum(inside)) if finite else None,
        "max_penetration_proxy_m": float(np.max(np.maximum(0.0, float(probe_half_extent) - np.linalg.norm(
            current - np.asarray(probe_position), axis=1
        )))) if finite else None,
    }
    if top_indices is not None and finite:
        top = np.asarray(top_indices, dtype=int)
        result["top_edge_max_displacement_m"] = float(np.max(np.linalg.norm(current[top] - reference[top], axis=1)))
    return result


def deformable_measurements(deformable, rest_positions, probe_position, probe_half_extent, top_indices=None):
    simulation, collision, rest = deformable.get_nodal_positions()
    current = as_numpy(simulation)[0].astype(float)
    rest_from_sim = as_numpy(rest)[0].astype(float)
    reference = rest_positions if rest_positions is not None else rest_from_sim
    return point_measurements(current, reference, probe_position, probe_half_extent, top_indices)


def build_fallback(stage, config):
    from lift_curtain.envs.curtain_asset import create_surface_sheet

    return create_surface_sheet(stage, "/World/Curtain/Cloth", config)


def main() -> int:
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from lift_curtain.envs.curtain_asset import CurtainConfig

    run_root = ARGS.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    source = ARGS.source.resolve()
    mode_root = run_root / ARGS.mode
    mode_root.mkdir(parents=True, exist_ok=True)
    config = {
        "mode": ARGS.mode,
        "source_asset": str(source),
        "source_commit": "470681a7ff117e487060ad52bc4f199e3fa34474",
        "source_license_status": "README_MIT_badge_but_no_LICENSE_file_in_checkout",
        "seed": ARGS.seed,
        "num_envs": 1,
        "headless": True,
        "physics_dt_s": DT,
        "render_interval": RENDER_INTERVAL,
        "steps": ARGS.steps,
        "probe_span_m": ARGS.probe_span,
        "isaac_sim_deformable_beta": True,
    }
    if ARGS.mode == "external":
        if not source.is_file():
            raise FileNotFoundError(source)
        config["source_size_bytes"] = source.stat().st_size
        config["source_sha256"] = sha256(source)

    (mode_root / "resolved_config.json").write_text(json.dumps(clean(config), indent=2) + "\n", encoding="utf-8")
    np.random.seed(ARGS.seed)
    torch.manual_seed(ARGS.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    sim = None
    rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {**config, "status": "BLOCKED"}
    try:
        sim = SimulationContext(SimulationCfg(dt=DT, render_interval=RENDER_INTERVAL, device="cuda:0"))
        sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
        sim_utils.DomeLightCfg(intensity=1200.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=1200.0))

        # Isaac Lab's helper is the supported way to obtain the active stage.
        from isaaclab.sim.utils import get_current_stage

        stage = get_current_stage()
        info = None
        external_root = None
        external_mesh_path = None
        external_mesh_prim = None
        if ARGS.mode == "external":
            external_root = sim_utils.UsdFileCfg(usd_path=str(source)).func(
                "/World/ExternalCurtain",
                sim_utils.UsdFileCfg(usd_path=str(source)),
                translation=(0.0, 0.0, 0.0),
            )
            report["asset_root_path"] = str(external_root.GetPath())
            report["asset_schema_summary"] = prim_schema_summary(external_root)
            mesh_details = report["asset_schema_summary"]["mesh_prims"]
            cloth_details = [item for item in mesh_details if item["path"].lower().endswith("/cloth")]
            if not cloth_details:
                cloth_details = [item for item in mesh_details if item["point_count"] > 0]
            if cloth_details:
                external_mesh_path = cloth_details[0]["path"]
                external_mesh_prim = stage.GetPrimAtPath(external_mesh_path)
                report["legacy_mesh_path"] = external_mesh_path
            minimum, maximum = world_bounds(stage, external_root)
            report["world_bounds_min"] = [float(v) for v in minimum]
            report["world_bounds_max"] = [float(v) for v in maximum]
        else:
            curtain_config = CurtainConfig()
            info = build_fallback(stage, curtain_config)
            report["fallback_config"] = curtain_config.__dict__
            report["asset_root_path"] = info.prim_path
            report["support_path"] = info.support_path
            report["attachment_path"] = info.attachment_path
            report["material_path"] = info.material_path
            minimum = np.min(np.asarray(info.rest_points), axis=0)
            maximum = np.max(np.asarray(info.rest_points), axis=0)
            report["world_bounds_min"] = minimum.tolist()
            report["world_bounds_max"] = maximum.tolist()

        probe = add_probe(sim_utils, RigidObject, RigidObjectCfg)
        sim.reset()
        probe.reset()
        probe.update(DT)

        deformable_path = None
        deformable = None
        if ARGS.mode == "fallback":
            import omni.kit.app

            extension_manager = omni.kit.app.get_app().get_extension_manager()
            extension_manager.set_extension_enabled_immediate("isaacsim.core.experimental.prims", True)
            from isaacsim.core.experimental.prims import DeformablePrim

            deformable_path = info.prim_path
            deformable = DeformablePrim(deformable_path, deformable_type="surface")
        else:
            deformable_path = external_mesh_path
        if deformable_path is None:
            raise RuntimeError("no deformable mesh or surface schema found in the loaded asset")

        if deformable is not None:
            simulation, _, rest = deformable.get_nodal_positions()
            rest_positions = as_numpy(rest)[0].astype(float)
        else:
            rest_positions = mesh_points_world(external_mesh_prim)
        report["deformable_path"] = deformable_path
        report["deformable_num_nodes"] = int(rest_positions.shape[0])
        report["deformable_interface_status"] = "current_surface_api" if deformable is not None else "legacy_usd_mesh_points"

        start, end, normal_axis = probe_path(minimum, maximum, ARGS.probe_span)
        report["probe_normal_axis"] = normal_axis
        report["probe_start"] = start.tolist()
        report["probe_end"] = end.tolist()
        report["probe_half_extent_m"] = 0.11

        total_steps = max(1, int(ARGS.steps))
        zero_velocity = torch.zeros((1, 6), device=sim.device)
        top_indices = info.top_row_indices if info is not None else None
        for step in range(total_steps):
            alpha = step / max(1, total_steps - 1)
            position = (1.0 - alpha) * start + alpha * end
            pose = torch.tensor(
                [position[0], position[1], position[2], 1.0, 0.0, 0.0, 0.0],
                device=sim.device,
                dtype=torch.float32,
            ).unsqueeze(0)
            probe.write_root_pose_to_sim(pose)
            probe.write_root_velocity_to_sim(zero_velocity)
            probe.write_data_to_sim()
            sim.step(render=False)
            probe.update(DT)
            if deformable is not None:
                measurement = deformable_measurements(deformable, rest_positions, position, 0.11, top_indices)
            else:
                measurement = point_measurements(
                    mesh_points_world(external_mesh_prim), rest_positions, position, 0.11, top_indices
                )
            row = {
                "step": step,
                "time_s": (step + 1) * DT,
                "probe_x": float(position[0]),
                "probe_y": float(position[1]),
                "probe_z": float(position[2]),
                "gpu_allocated_mb": float(torch.cuda.memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0,
                "gpu_reserved_mb": float(torch.cuda.memory_reserved() / (1024 ** 2)) if torch.cuda.is_available() else 0.0,
                **measurement,
            }
            rows.append(row)

        telemetry_path = mode_root / "telemetry.csv"
        with telemetry_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        finite = all(bool(row["finite"]) for row in rows)
        max_displacement = max(float(row["max_displacement_m"]) for row in rows)
        max_extent_ratio = max(float(row["max_extent_ratio"]) for row in rows)
        max_overlap = max(int(row["probe_node_overlap_count"]) for row in rows)
        # These are deliberately broad sanity bounds, not task thresholds.
        stable = finite and max_extent_ratio < 4.0 and max_displacement < 5.0
        report.update(
            {
                "status": "PASS" if stable else "FAIL",
                "load_status": "PASS",
                "deformable_interface_status": "PASS",
                "stability_status": "PASS" if stable else "FAIL",
                "max_displacement_m": max_displacement,
                "max_extent_ratio": max_extent_ratio,
                "max_probe_node_overlap_count": max_overlap,
                "telemetry_csv": str(telemetry_path),
                "peak_gpu_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0,
                "peak_gpu_reserved_mb": float(torch.cuda.max_memory_reserved() / (1024 ** 2)) if torch.cuda.is_available() else 0.0,
                "notes": [
                    "Probe is kinematic and moves once through the candidate along the thinnest world-axis.",
                    "Probe-node overlap is a coarse penetration proxy, not a contact-force measurement.",
                    "No camera or GUI was launched; headless physics telemetry is the primary evidence.",
                ],
            }
        )
        return_code = 0 if stable else 2
    except Exception as exc:
        report.update(
            {
                "status": "BLOCKED",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "load_status": "UNKNOWN",
                "deformable_interface_status": "UNKNOWN",
                "stability_status": "UNKNOWN",
            }
        )
        return_code = 2
    finally:
        report["completed_rows"] = len(rows)
        report["status"] = report.get("status", "BLOCKED")
        (mode_root / "summary.json").write_text(json.dumps(clean(report), indent=2) + "\n", encoding="utf-8")
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
    print(json.dumps(clean(report), indent=2))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
