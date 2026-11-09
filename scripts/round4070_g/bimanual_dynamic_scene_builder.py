"""Offline-first dynamic scene plans for the BIMANUAL_V1 candidate matrix.

This module separates reusable builder readiness, native asset authoring, and
physics validation. Importing it is safe while the campaign is paused: Isaac
Sim and Isaac Lab are imported only from explicit build_native methods.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "round4070-g.bimanual.dynamic_scene_plan.v1"
BUILDER_READY = "BUILDER_READY"
ASSET_READY = "ASSET_READY"
NOT_BUILT = "NOT_BUILT"
REQUIRES_PHYSICS_VALIDATION = "REQUIRES_PHYSICS_VALIDATION"
ALLOWED_SCENE_STATUSES = {ASSET_READY, BUILDER_READY, NOT_BUILT, REQUIRES_PHYSICS_VALIDATION}

REPO = Path(__file__).resolve().parents[2]
REPORT = REPO / "research" / "round4070_comprehensive_traversability"
ASSET_DIR = REPORT / "bimanual_v1_assets"
SCENE_DIR = ASSET_DIR / "scene_specs"
PLAN_DIR = ASSET_DIR / "dynamic_scene_plans"


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _finite_float(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _int(value: Any, name: str, *, minimum: int = 0) -> int:
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _deepcopy(value: Any) -> Any:
    return copy.deepcopy(value)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _scene_config(asset: dict[str, Any]) -> dict[str, Any]:
    value = asset.get("scene_config", asset)
    if not isinstance(value, dict):
        raise ValueError("scene asset does not contain a scene_config object")
    return value


def _expand_sheets(scene: dict[str, Any]) -> list[dict[str, Any]]:
    """Apply the same sheet override semantics as the existing runtime builder."""
    base_geometry = scene.get("geometry", {})
    base_physics = scene.get("physics", {})
    base_topology = scene.get("topology", {})
    overrides = scene.get("sheets") or [{}]
    result: list[dict[str, Any]] = []
    for index, override in enumerate(overrides):
        geometry = _deepcopy(base_geometry)
        physics = _deepcopy(base_physics)
        topology = _deepcopy(base_topology)
        if not isinstance(override, dict):
            raise ValueError(f"sheet override {index} is not an object")
        for key, value in override.items():
            if key in geometry:
                geometry[key] = _deepcopy(value)
            elif key in physics:
                physics[key] = _deepcopy(value)
            elif key == "topology":
                topology["kind"] = _deepcopy(value)
            elif key in topology:
                topology[key] = _deepcopy(value)
        geometry["center_x_m"] = float(geometry.get("center_x_m", 0.0)) + float(
            override.get("center_x_offset_m", 0.0)
        )
        result.append({"index": index, "geometry": geometry, "physics": physics, "topology": topology})
    return result


def _fixed_indices(topology: dict[str, Any], rows: int, columns: int) -> list[int]:
    """Return the audit's exact index convention, including its naming."""
    total = rows * columns
    explicit = topology.get("local_fixed_vertex_indices")
    if explicit is not None:
        indices = sorted({int(value) for value in explicit})
        if any(index < 0 or index >= total for index in indices):
            raise ValueError("explicit topology index is outside the sheet grid")
        return indices
    top = list(range(columns))
    bottom = list(range((rows - 1) * columns, rows * columns))
    right = [row * columns for row in range(rows)]
    left = [row * columns + columns - 1 for row in range(rows)]
    corners = [0, columns - 1, (rows - 1) * columns, rows * columns - 1]
    options = {
        "TOP_EDGE": top,
        "TOP_PLUS_LEFT": top + [bottom[-1]],
        "TOP_PLUS_RIGHT": top + [bottom[0]],
        "TOP_PLUS_BOTH": top + [bottom[0], bottom[-1]],
        "LEFT_EDGE": left,
        "RIGHT_EDGE": right,
        "BOTTOM_LEFT_CORNER": [corners[3]],
        "BOTTOM_RIGHT_CORNER": [corners[2]],
        "BOTTOM_BOTH_CORNERS": [corners[2], corners[3]],
        "FOUR_CORNERS": corners,
        "FOUR_CORNERS_ONLY": corners,
        "TOP_PLUS_BOTH_BOTTOM": top + [corners[2], corners[3]],
        "FULL_PERIMETER": top + bottom + left + right,
        "THREE_FREE_LEFT": corners[:-1],
        "THREE_FREE_RIGHT": [corners[0], corners[1], corners[3]],
        "PARTIAL_TOP": top[columns // 3 : 2 * columns // 3 + 1],
    }
    kind = str(topology.get("semantic_kind", topology.get("kind", "TOP_EDGE")))
    if kind.startswith("THREE_CORNERS_MISSING_"):
        missing = kind.removeprefix("THREE_CORNERS_MISSING_")
        names = ["TOP_RIGHT", "TOP_LEFT", "BOTTOM_RIGHT", "BOTTOM_LEFT"]
        return [index for name, index in zip(names, corners) if name != missing]
    if str(topology.get("suspension", "STATIC")) == "MULTI_RINGS":
        stride = max(1, int(topology.get("ring_stride", 3)))
        return sorted(set(top[::stride] + [top[-1]]))
    if kind not in options:
        raise ValueError(f"unsupported topology kind: {kind}")
    return sorted(set(options[kind]))


def _kept_cell(row: int, column: int, rows: int, columns: int, period: int) -> bool:
    if period == 0:
        return True
    if row in (0, rows - 2) or column in (0, columns - 2):
        return True
    return row % period != 0 or column % period != 0


def _sheet_mesh_plan(sheet: dict[str, Any], scene: dict[str, Any]) -> dict[str, Any]:
    geometry = sheet["geometry"]
    physics = sheet["physics"]
    topology = sheet["topology"]
    width = _finite_float(geometry.get("width_m"), "width_m", positive=True)
    height = _finite_float(geometry.get("height_m"), "height_m", positive=True)
    bottom_z = _finite_float(geometry.get("bottom_z_m", 0.0), "bottom_z_m")
    center_x = _finite_float(geometry.get("center_x_m", 0.0), "center_x_m")
    center_y = _finite_float(geometry.get("center_y_m", 0.0), "center_y_m")
    columns = _int(geometry.get("columns"), "columns", minimum=2)
    rows = _int(geometry.get("rows"), "rows", minimum=2)
    pleat_amplitude = _finite_float(geometry.get("pleat_amplitude_m", 0.0), "pleat_amplitude_m", nonnegative=True)
    pleat_waves = _finite_float(geometry.get("pleat_waves", 0.0), "pleat_waves", nonnegative=True)
    period = _int(geometry.get("net_hole_period", 0), "net_hole_period", minimum=0)
    thickness = _finite_float(physics.get("thickness_m"), "thickness_m", positive=True)
    fixed = _fixed_indices(topology, rows, columns)
    cells = (rows - 1) * (columns - 1)
    holes = sum(1 for row in range(rows - 1) for column in range(columns - 1)
                if not _kept_cell(row, column, rows, columns, period))
    triangles = 2 * (cells - holes)
    recipe = {
        "local_plane": "x_z",
        "native_yaw_rad": math.pi / 2.0,
        "world_axes": {"width": "+y_after_yaw", "normal": "+x_after_yaw", "height": "+z"},
        "width_m": width,
        "height_m": height,
        "bottom_z_m": bottom_z,
        "center_x_m": center_x,
        "center_y_m": center_y,
        "columns": columns,
        "rows": rows,
        "pleat_amplitude_m": pleat_amplitude,
        "pleat_waves": pleat_waves,
        "net_hole_period": period,
        "net_hole_size": int(geometry.get("net_hole_size", 1)),
    }
    mesh_hash = sha256_value({"recipe": recipe, "triangle_count": triangles, "fixed_indices": fixed})
    minimum_spacing = min(width / (columns - 1), height / (rows - 1))
    support_type = str(topology.get("suspension", "STATIC"))
    attachment_mode = {"STATIC": "WORLD_FIXED", "WHOLE_RAIL": "PRISMATIC_RAIL", "MULTI_RINGS": "RING_SUPPORTS"}.get(
        support_type, "EXPLICIT_VERTEX_SUPPORTS"
    )
    return {
        "sheet_index": int(sheet["index"]),
        "component_type": "surface_deformable_sheet",
        "units": "m_kg_s",
        "mesh_recipe": recipe,
        "mesh_hash": mesh_hash,
        "vertex_count": rows * columns,
        "cell_count": cells,
        "removed_hole_cell_count": holes,
        "triangle_count": triangles,
        "rest_bounds_m": {
            "min": [center_x - pleat_amplitude, center_y - width / 2.0, bottom_z],
            "max": [center_x + pleat_amplitude, center_y + width / 2.0, bottom_z + height],
        },
        "minimum_grid_spacing_m": minimum_spacing,
        "fixed_vertex_indices": fixed,
        "fixed_vertex_count": len(fixed),
        "attachment": {
            "mechanism": attachment_mode,
            "suspension": support_type,
            "rail_limit_m": float(topology.get("rail_limit_m", 0.3)),
            "ring_stride": int(topology.get("ring_stride", 3)),
            "overlap_offset_m": 0.005 if minimum_spacing <= 0.04 else 0.01,
        },
        "physics_material": {
            "density_kg_m3": _finite_float(physics.get("density_kg_m3"), "density_kg_m3", positive=True),
            "thickness_m": thickness,
            "static_friction": _finite_float(physics.get("static_friction", 0.5), "static_friction", nonnegative=True),
            "dynamic_friction": _finite_float(physics.get("dynamic_friction", 0.4), "dynamic_friction", nonnegative=True),
            "stretch_stiffness": _finite_float(physics.get("stretch_stiffness"), "stretch_stiffness", nonnegative=True),
            "shear_stiffness": _finite_float(physics.get("shear_stiffness"), "shear_stiffness", nonnegative=True),
            "bend_stiffness": _finite_float(physics.get("bend_stiffness"), "bend_stiffness", nonnegative=True),
            "damping": _finite_float(physics.get("damping"), "damping", nonnegative=True),
            "hem_mass_kg": _finite_float(physics.get("hem_mass_kg", 0.0), "hem_mass_kg", nonnegative=True),
        },
        "native_schema": {
            "mesh": "UsdGeom.Mesh",
            "deformable_body": "omni.physx.scripts.deformableUtils.set_physics_surface_deformable_body",
            "collision": "UsdPhysics.CollisionAPI",
            "material": "omni.physx.scripts.deformableUtils.add_surface_deformable_material",
            "attachment": "omni.physx.scripts.deformableUtils.create_auto_deformable_attachment",
            "attachment_refresh": "omni.physx.get_physx_attachment_private_interface.update_auto_deformable_attachment",
        },
    }


@dataclass(frozen=True)
class ComponentPlan:
    component_type: str
    status: str
    plan: dict[str, Any]
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"component_type": self.component_type, "status": self.status, "plan": self.plan,
                "errors": list(self.errors), "warnings": list(self.warnings)}


class FabricCurtainBuilder:
    """Build deterministic surface-deformable recipes for one or more sheets."""

    name = "FabricCurtainBuilder"
    native_api = "Isaac Sim 5.1 surface-deformable API"

    def plan(self, scene: dict[str, Any]) -> ComponentPlan:
        kind = str(scene.get("geometry", {}).get("kind", "surface"))
        if kind in {"empty", "rigid_board"}:
            return ComponentPlan(self.name, BUILDER_READY, {"enabled": False, "reason": f"geometry_kind={kind}"})
        errors: list[str] = []
        sheets: list[dict[str, Any]] = []
        try:
            for sheet in _expand_sheets(scene):
                sheets.append(_sheet_mesh_plan(sheet, scene))
        except (TypeError, ValueError, KeyError) as exc:
            errors.append(f"FABRIC_PLAN_INVALID:{exc}")
        status = BUILDER_READY if not errors else NOT_BUILT
        return ComponentPlan(
            self.name,
            status,
            {
                "enabled": True,
                "native_api": self.native_api,
                "surface_deformable": True,
                "sheet_count": len(sheets),
                "sheets": sheets,
                "runtime_builder_source": "scripts/round4070_g/physics_scene_builder.py + structural_curtain_asset.py",
            },
            tuple(errors),
        )

    def build_native(self, stage: Any, scene: dict[str, Any], prim_root: str = "/World/Bimanual") -> Any:
        """Explicit Kit-only entry point; never called by offline preparation."""
        try:
            from .physics_scene_builder import create_scene  # type: ignore
        except ImportError:
            from physics_scene_builder import create_scene  # type: ignore

        return create_scene(stage, scene, prim_root=prim_root)


class RigidObstacleBuilder:
    name = "RigidObstacleBuilder"

    def plan(self, scene: dict[str, Any]) -> ComponentPlan:
        geometry = scene.get("geometry", {})
        obstacles: list[dict[str, Any]] = []
        errors: list[str] = []
        wall = scene.get("wall")
        if wall is not None:
            try:
                gap = _finite_float(wall.get("initial_gap_m"), "wall.initial_gap_m", positive=True)
                width = _finite_float(wall.get("width_m"), "wall.width_m", positive=True)
                height = _finite_float(wall.get("height_m"), "wall.height_m", positive=True)
                thickness = _finite_float(wall.get("thickness_m"), "wall.thickness_m", positive=True)
                center_x = _finite_float(geometry.get("center_x_m"), "geometry.center_x_m") + gap + thickness / 2.0
                obstacles.append({
                    "kind": "hidden_rigid_wall",
                    "center_xyz_m": [center_x, float(wall.get("center_y_m", 0.0)), float(wall.get("bottom_z_m", 0.0)) + height / 2.0],
                    "size_xyz_m": [thickness, width, height],
                    "initial_gap_m": gap,
                    "fixed": bool(wall.get("fixed", True)),
                })
            except (TypeError, ValueError, KeyError) as exc:
                errors.append(f"WALL_PLAN_INVALID:{exc}")
        if str(geometry.get("kind")) == "rigid_board":
            obstacles.append({
                "kind": "rigid_board",
                "center_xyz_m": [float(geometry.get("center_x_m", 0.0)) + 0.02, 0.0, 1.0],
                "size_xyz_m": [0.04, 1.2, 2.0],
                "fixed": True,
                "material_fidelity": scene.get("material_fidelity"),
            })
        return ComponentPlan(
            self.name,
            BUILDER_READY if not errors else NOT_BUILT,
            {
                "enabled": bool(obstacles),
                "obstacles": obstacles,
                "native_schema": {
                    "geometry": "UsdGeom.Cube / Isaac Lab sim_utils.CuboidCfg",
                    "collision": "UsdPhysics.CollisionAPI",
                    "fixed_body": "no dynamic RigidBodyAPI for fixed barriers",
                },
            },
            tuple(errors),
        )

    def build_native(self, stage: Any, obstacles: Iterable[dict[str, Any]], prim_root: str = "/World/Bimanual") -> list[str]:
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics  # type: ignore

        paths: list[str] = []
        for index, obstacle in enumerate(obstacles):
            path = f"{prim_root}/RigidObstacle_{index:02d}"
            cube = UsdGeom.Cube.Define(stage, Sdf.Path(path))
            cube.CreateSizeAttr(1.0)
            xform = UsdGeom.XformCommonAPI(cube)
            xform.SetTranslate(Gf.Vec3d(*obstacle["center_xyz_m"]))
            xform.SetScale(Gf.Vec3f(*obstacle["size_xyz_m"]))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            paths.append(path)
        return paths


class PvcMeshBuilder:
    name = "PvcMeshBuilder"

    def plan(self, scene: dict[str, Any], fabric: ComponentPlan) -> ComponentPlan:
        fidelity = str(scene.get("material_fidelity", ""))
        geometry = scene.get("geometry", {})
        is_pvc_or_mesh = (
            "PVC" in fidelity.upper()
            or "MESH" in fidelity.upper()
            or int(geometry.get("net_hole_period", 0) or 0) > 0
            or len(scene.get("sheets") or []) > 1
        )
        if not is_pvc_or_mesh:
            return ComponentPlan(self.name, BUILDER_READY, {"enabled": False, "reason": "ordinary_surface_or_rigid_control"})
        real_required = fidelity.startswith("REAL_") or "REQUIRED_NOT_PRESENT" in fidelity
        warnings = ["PVC_OR_NET_PROXY_IS_NOT_MATERIAL_CALIBRATION"]
        errors: list[str] = []
        if real_required:
            errors.append("MISSING_EXTERNAL_MATERIAL_ASSET")
        representation = "pvc_independent_surface_strips" if len(scene.get("sheets") or []) > 1 else "perforated_surface_proxy"
        plan = {
            "enabled": True,
            "representation": representation,
            "material_fidelity": fidelity,
            "proxy_of": "PVC_or_open_mesh",
            "external_asset_required": real_required,
            "calibration_required_before_formal_use": True,
            "fabric_sheet_plan_hash": sha256_value(fabric.plan.get("sheets", [])),
            "native_surface_api": "same_surface_deformable_backend_as_fabric",
        }
        return ComponentPlan(self.name, NOT_BUILT if errors else BUILDER_READY, plan, tuple(errors), tuple(warnings))


class HingedDoorBuilder:
    name = "HingedDoorBuilder"

    def plan(self, scene: dict[str, Any]) -> ComponentPlan:
        door = scene.get("door")
        if door is None:
            return ComponentPlan(self.name, BUILDER_READY, {"enabled": False, "reason": "no_door_component"})
        errors: list[str] = []
        try:
            width = _finite_float(door.get("width_m"), "door.width_m", positive=True)
            height = _finite_float(door.get("height_m"), "door.height_m", positive=True)
            thickness = _finite_float(door.get("thickness_m"), "door.thickness_m", positive=True)
            gap = _finite_float(door.get("curtain_to_door_gap_m"), "door.curtain_to_door_gap_m", positive=True)
            resistance = _finite_float(door.get("hinge_friction_torque_nm"), "door.hinge_friction_torque_nm", nonnegative=True)
            initial_angle = _finite_float(door.get("initial_angle_rad", 0.0), "door.initial_angle_rad")
            nominal_angle = _finite_float(door.get("nominal_open_angle_rad"), "door.nominal_open_angle_rad", nonnegative=True)
            direction = str(door.get("opening_direction", "POSITIVE_Y"))
            if direction not in {"POSITIVE_Y", "NEGATIVE_Y"}:
                errors.append("UNSUPPORTED_DOOR_OPENING_DIRECTION")
            scene_x = _finite_float(scene.get("geometry", {}).get("center_x_m"), "geometry.center_x_m")
            scene_y = _finite_float(scene.get("geometry", {}).get("center_y_m", 0.0), "geometry.center_y_m")
            hinge_y = scene_y - width / 2.0 if direction == "POSITIVE_Y" else scene_y + width / 2.0
            door_x = scene_x + gap + thickness / 2.0
            reference_speed_deg_s = 30.0
            plan = {
                "enabled": True,
                "kind": "hinged_rigid_door",
                "center_xyz_m": [door_x, scene_y, height / 2.0],
                "size_xyz_m": [thickness, width, height],
                "mass_kg": 20.0,
                "mass_fidelity": "UNCALIBRATED_RIGID_DOOR_PROXY",
                "hinge": {
                    "axis": "Z",
                    "world_position_m": [door_x, hinge_y, 0.0],
                    "initial_angle_rad": initial_angle,
                    "nominal_open_angle_rad": nominal_angle,
                    "limit_units": "radians_in_plan; authoring adapter must convert to USD joint limit units",
                    "opening_direction": direction,
                    "friction_torque_nm": resistance,
                },
                "resistance_model": {
                    "kind": "VISCOUS_ANGULAR_DRIVE_PROXY",
                    "target_velocity_deg_s": 0.0,
                    "reference_speed_deg_s": reference_speed_deg_s,
                    "damping_nm_s_per_deg": resistance / reference_speed_deg_s,
                    "max_force_nm": resistance,
                    "requested_peak_resistance_nm": resistance,
                    "calibration_status": "REQUIRES_PHYSICS_VALIDATION",
                    "reason": "PhysxJointAPI jointFriction is dimensionless and cannot represent the requested Nm value directly.",
                },
                "pushable": bool(door.get("pushable", True)),
                "curtain_to_door_gap_m": gap,
                "label_scope": {
                    "first_stage": door.get("curtain_layer_label_scope", "CURTAIN_ONLY_FIRST"),
                    "second_stage": door.get("composite_route_label_scope", "SECOND_STAGE_ONLY"),
                },
                "native_schema": {
                    "geometry": "UsdGeom.Cube",
                    "collision": "UsdPhysics.CollisionAPI",
                    "rigid_body": "UsdPhysics.RigidBodyAPI",
                    "joint": "UsdPhysics.RevoluteJoint",
                    "joint_axis": "Z",
                    "joint_limit": "UsdPhysics.RevoluteJoint lower/upper limit",
                    "resistance": "UsdPhysics.DriveAPI angular force drive with zero target velocity",
                },
            }
        except (TypeError, ValueError, KeyError) as exc:
            errors.append(f"DOOR_PLAN_INVALID:{exc}")
            plan = {"enabled": True, "kind": "hinged_rigid_door"}
        return ComponentPlan(
            self.name,
            BUILDER_READY if not errors else NOT_BUILT,
            plan,
            tuple(errors),
            (
                "DOOR_REQUIRES_RUNTIME_JOINT_AND_CONTACT_VALIDATION",
                "DOOR_MASS_AND_RESISTANCE_ARE_UNCALIBRATED_PROXIES",
            ),
        )

    def build_native(self, stage: Any, plan: dict[str, Any], prim_root: str = "/World/Bimanual") -> dict[str, str]:
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics  # type: ignore

        door_path = f"{prim_root}/HingedDoorBody"
        body = UsdGeom.Xform.Define(stage, Sdf.Path(door_path))
        body_xform = UsdGeom.XformCommonAPI(body)
        hinge = plan["hinge"]["world_position_m"]
        body_xform.SetTranslate(Gf.Vec3d(*hinge))
        body_xform.SetRotate(Gf.Vec3f(0.0, 0.0, math.degrees(float(plan["hinge"]["initial_angle_rad"]))))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(float(plan["mass_kg"]))

        panel_path = f"{door_path}/Panel"
        cube = UsdGeom.Cube.Define(stage, Sdf.Path(panel_path))
        cube.CreateSizeAttr(1.0)
        panel_xform = UsdGeom.XformCommonAPI(cube)
        width = float(plan["size_xyz_m"][1])
        height = float(plan["size_xyz_m"][2])
        offset_sign = 1.0 if plan["hinge"]["opening_direction"] == "POSITIVE_Y" else -1.0
        panel_xform.SetTranslate(Gf.Vec3d(0.0, offset_sign * width / 2.0, height / 2.0))
        panel_xform.SetScale(Gf.Vec3f(*plan["size_xyz_m"]))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

        joint = UsdPhysics.RevoluteJoint.Define(stage, Sdf.Path(f"{prim_root}/HingedDoorJoint"))
        joint.CreateBody1Rel().SetTargets([body.GetPath()])
        joint.CreateAxisAttr("Z")
        joint.CreateLocalPos0Attr(Gf.Vec3f(*hinge))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        limit_deg = math.degrees(float(plan["hinge"]["nominal_open_angle_rad"]))
        if plan["hinge"]["opening_direction"] == "POSITIVE_Y":
            lower_deg, upper_deg = 0.0, limit_deg
        else:
            lower_deg, upper_deg = -limit_deg, 0.0
        joint.CreateLowerLimitAttr(lower_deg)
        joint.CreateUpperLimitAttr(upper_deg)

        resistance = plan["resistance_model"]
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(0.0)
        drive.CreateDampingAttr(float(resistance["damping_nm_s_per_deg"]))
        drive.CreateMaxForceAttr(float(resistance["max_force_nm"]))
        drive.CreateTargetVelocityAttr(float(resistance["target_velocity_deg_s"]))
        return {"door_path": str(door_path), "panel_path": str(panel_path), "joint_path": str(joint.GetPath())}


class BimanualSceneComposer:
    """Compose all reusable component plans for one candidate scene."""

    def __init__(self) -> None:
        self.fabric = FabricCurtainBuilder()
        self.rigid = RigidObstacleBuilder()
        self.pvc = PvcMeshBuilder()
        self.door = HingedDoorBuilder()

    def build_native(self, stage: Any, asset: dict[str, Any], prim_root: str = "/World/Bimanual") -> dict[str, Any]:
        """Author one complete scene into an already-created Kit stage.

        This method is deliberately explicit: callers must own the Kit stage
        and the safety admission that permits it. The offline workflow never
        calls it, so merely importing or planning scenes cannot start Isaac.
        """
        scene = _scene_config(asset)
        fabric_bundle = self.fabric.build_native(stage, scene, prim_root=prim_root)
        door_plan = self.door.plan(scene)
        door_result = None
        if door_plan.plan.get("enabled"):
            door_result = self.door.build_native(stage, door_plan.plan, prim_root=prim_root)
        return {
            "scene_id": scene.get("scene_id", asset.get("scene_id")),
            "fabric_bundle": fabric_bundle,
            "door": door_result,
            "physics_validation_required": True,
            "native_usd_asset_status": NOT_BUILT,
            "safety_note": "Native authoring does not constitute a physics gate or G1 admission.",
        }

    def compose(self, asset: dict[str, Any], *, source_path: str = "") -> dict[str, Any]:
        scene = _scene_config(asset)
        scene_id = str(scene.get("scene_id", asset.get("scene_id", "")))
        physical_config_id = str(scene.get("physical_config_id", asset.get("physical_config_id", "")))
        components = [self.fabric.plan(scene)]
        components.append(self.rigid.plan(scene))
        components.append(self.pvc.plan(scene, components[0]))
        components.append(self.door.plan(scene))
        fidelity = str(scene.get("material_fidelity", ""))
        if fidelity.startswith("REAL_") or "REQUIRED_NOT_PRESENT" in fidelity:
            components.append(ComponentPlan(
                "MaterialAssetDependency",
                NOT_BUILT,
                {
                    "enabled": True,
                    "material_fidelity": fidelity,
                    "external_asset_required": True,
                    "reason": "real material source is not present in the repository and cannot be substituted silently",
                },
                ("MISSING_EXTERNAL_MATERIAL_ASSET",),
            ))
        errors = [error for component in components for error in component.errors]
        warnings = [warning for component in components for warning in component.warnings]
        checks = self._offline_checks(scene, components, errors)
        real_material_missing = "MISSING_EXTERNAL_MATERIAL_ASSET" in errors
        builder_status = BUILDER_READY if not errors else NOT_BUILT
        # No Kit/Usd stage is allowed while PAUSE and PAUSE_G1 are present.
        dynamic_asset_status = NOT_BUILT
        physics_status = "NOT_RUN_PAUSED"
        overall_status = NOT_BUILT if real_material_missing or errors else REQUIRES_PHYSICS_VALIDATION
        return {
            "schema": SCHEMA,
            "plan_version": "BIMANUAL_DYNAMIC_BUILDER_V1",
            "scene_id": scene_id,
            "physical_config_id": physical_config_id,
            "source_scene_asset": source_path,
            "safety": {
                "kit_launch_performed": False,
                "physics_gate_performed": False,
                "g1_launch_performed": False,
                "pause_flags_required": True,
                "v6_status": "NO_GO_UNCHANGED",
            },
            "status": overall_status,
            "dynamic_builder_status": builder_status,
            "dynamic_asset_status": dynamic_asset_status,
            "physics_validation_status": physics_status,
            "missing_or_blocking_reasons": sorted(set(errors)),
            "warnings": sorted(set(warnings)),
            "components": [component.as_dict() for component in components],
            "offline_checks": checks,
            "hand_collider": {
                "status": "NOT_FORMALLY_CERTIFIED",
                "source": "research/round4070_comprehensive_traversability/NATIVE_HAND_COLLIDER_CERTIFICATION_V6.json",
                "exact_native_collider_admission": False,
                "reason": "B10/V6 native binding and contact-plane certification remains pending; no scene plan can upgrade it.",
            },
            "formal_use": "PROHIBITED_UNTIL_NATIVE_USD_PHYSICS_VALIDATION_AND_BIMANUAL_SAFETY_CERTIFICATION",
            "plan_sha256": sha256_value({"scene": scene, "components": [component.as_dict() for component in components]}),
        }

    @staticmethod
    def _offline_checks(scene: dict[str, Any], components: list[ComponentPlan], errors: list[str]) -> dict[str, Any]:
        fabric = components[0].plan
        sheets = fabric.get("sheets", [])
        mesh_ok = all(sheet.get("vertex_count", 0) > 0 and sheet.get("triangle_count", 0) > 0 for sheet in sheets) if sheets else True
        fixed_ok = all(
            all(0 <= int(index) < int(sheet["vertex_count"]) for index in sheet.get("fixed_vertex_indices", []))
            for sheet in sheets
        )
        spacing_ok = all(float(sheet.get("minimum_grid_spacing_m", 0.0)) > 0.0 for sheet in sheets)
        initial_gap_ok = True
        wall = scene.get("wall")
        door = scene.get("door")
        if wall is not None:
            initial_gap_ok = initial_gap_ok and float(wall.get("initial_gap_m", 0.0)) > 0.0
        if door is not None:
            initial_gap_ok = initial_gap_ok and float(door.get("curtain_to_door_gap_m", 0.0)) > 0.0
        return {
            "mesh": "PASS_OFFLINE_ANALYTIC" if mesh_ok else "FAIL_OFFLINE_ANALYTIC",
            "units": "PASS_METRIC_M_KG_S",
            "fixed_nodes": "PASS_INDEX_RANGE" if fixed_ok else "FAIL_INDEX_RANGE",
            "colliders": "DECLARED_NATIVE_SCHEMA_NOT_RUNTIME_BOUND",
            "physics_material": "DECLARED_NATIVE_MATERIAL_PARAMETERS",
            "usd_schema": "DECLARED_NATIVE_API_MAPPING",
            "initial_spacing": "PASS_POSITIVE_GAP_AND_GRID" if initial_gap_ok and spacing_ok else "FAIL_CLEARANCE_OR_GRID",
            "hand_collider": "NOT_FORMALLY_CERTIFIED",
            "composition": "PASS_OFFLINE_COMPONENT_COMPOSITION" if not errors else "FAIL_OFFLINE_COMPONENT_COMPOSITION",
            "physics_gate": "NOT_RUN_PAUSED",
        }

    def compose_all(self, scene_dir: Path = SCENE_DIR) -> list[dict[str, Any]]:
        plans: list[dict[str, Any]] = []
        for path in sorted(scene_dir.glob("BI_*.json")):
            plans.append(self.compose(_read_json(path), source_path=str(path.relative_to(REPO))))
        return plans


def write_plan_files(plans: Iterable[dict[str, Any]], plan_dir: Path = PLAN_DIR) -> list[str]:
    plan_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for plan in plans:
        path = plan_dir / f"{plan['scene_id']}.json"
        _write_json(path, plan)
        paths.append(str(path.relative_to(REPO)))
    return paths


def load_all_scene_assets(scene_dir: Path = SCENE_DIR) -> list[dict[str, Any]]:
    return [_read_json(path) for path in sorted(scene_dir.glob("BI_*.json"))]


def build_all_offline(scene_dir: Path = SCENE_DIR, plan_dir: Path = PLAN_DIR) -> list[dict[str, Any]]:
    plans = BimanualSceneComposer().compose_all(scene_dir)
    write_plan_files(plans, plan_dir)
    return plans


if __name__ == "__main__":
    plans = build_all_offline()
    print(json.dumps({
        "schema": SCHEMA,
        "scene_count": len(plans),
        "status_counts": {status: sum(plan["status"] == status for plan in plans) for status in sorted(ALLOWED_SCENE_STATUSES)},
        "builder_status_counts": {status: sum(plan["dynamic_builder_status"] == status for plan in plans) for status in (BUILDER_READY, NOT_BUILT)},
        "physics_gate": "NOT_RUN_PAUSED",
    }, ensure_ascii=False, indent=2))
