"""Small current-API surface-deformable curtain builder.

The MoDeSuite USD is kept outside this repository.  This module only builds the
minimal fallback used when that external asset cannot be loaded reliably.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CurtainConfig:
    width_m: float = 1.4
    height_m: float = 1.8
    columns: int = 16
    rows: int = 20
    density_kg_m3: float = 250.0
    surface_thickness_m: float = 0.002
    stretch_stiffness: float = 500.0
    shear_stiffness: float = 250.0
    bend_stiffness: float = 0.02
    damping: float = 0.05
    solver_position_iterations: int = 8
    support_z_m: float = 1.8
    support_thickness_m: float = 0.04
    attachment_overlap_offset_m: float = 0.01
    attachment_enable_rigid_surface_attachments: bool = True
    bottom_anchor_mode: str = "none"
    bottom_anchor_size_m: float = 0.020
    structural_topology: str = "A0_TOP_EDGE_ONLY"
    # The mesh is authored in a local x-z plane. These pose fields allow the
    # same verified attachment construction to be placed in a task scene.
    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_rad: float = 0.0

    def validate(self) -> None:
        if self.width_m <= 0.0 or self.height_m <= 0.0:
            raise ValueError("curtain dimensions must be positive")
        if self.columns < 2 or self.rows < 2:
            raise ValueError("curtain mesh resolution must be at least 2x2")
        if min(
            self.density_kg_m3,
            self.surface_thickness_m,
            self.stretch_stiffness,
            self.shear_stiffness,
            self.bend_stiffness,
            self.damping,
            self.attachment_overlap_offset_m,
        ) < 0.0:
            raise ValueError("curtain physical parameters cannot be negative")
        if self.solver_position_iterations < 1:
            raise ValueError("solver_position_iterations must be positive")
        if not 0.005 <= self.attachment_overlap_offset_m <= 0.020:
            raise ValueError("attachment_overlap_offset_m must be within 5--20 mm")
        if self.bottom_anchor_mode not in ("none", "left", "right", "both"):
            raise ValueError("bottom_anchor_mode must be none, left, right, or both")
        if self.structural_topology not in (
            "A0_TOP_EDGE_ONLY",
            "A1_TOP_EDGE_BOTTOM_TWO_CORNERS",
            "A2_FOUR_CORNERS_ONLY",
            "A3_FULL_PERIMETER_FIXED",
        ):
            raise ValueError("unsupported structural_topology")
        if not math.isfinite(float(self.bottom_anchor_size_m)) or self.bottom_anchor_size_m <= 0.0:
            raise ValueError("bottom_anchor_size_m must be finite and positive")
        minimum_spacing = min(
            self.width_m / (self.columns - 1),
            self.height_m / (self.rows - 1),
        )
        if self.bottom_anchor_size_m + 2.0 * self.attachment_overlap_offset_m >= minimum_spacing:
            raise ValueError("bottom anchor selection could overlap an adjacent mesh vertex")
        if len(self.translation_m) != 3 or not all(math.isfinite(float(value)) for value in self.translation_m):
            raise ValueError("translation_m must contain three finite values")
        if not math.isfinite(float(self.yaw_rad)):
            raise ValueError("yaw_rad must be finite")

    @property
    def top_row_indices(self) -> tuple[int, ...]:
        return tuple(range(self.columns))

    @property
    def bottom_left_index(self) -> int:
        return (self.rows - 1) * self.columns

    @property
    def bottom_right_index(self) -> int:
        return self.rows * self.columns - 1

    @property
    def requested_bottom_anchor_indices(self) -> tuple[int, ...]:
        if self.structural_topology == "A1_TOP_EDGE_BOTTOM_TWO_CORNERS":
            return (self.bottom_left_index, self.bottom_right_index)
        if self.structural_topology == "A2_FOUR_CORNERS_ONLY":
            return (self.bottom_left_index, self.bottom_right_index)
        if self.bottom_anchor_mode == "left":
            return (self.bottom_left_index,)
        if self.bottom_anchor_mode == "right":
            return (self.bottom_right_index,)
        if self.bottom_anchor_mode == "both":
            return (self.bottom_left_index, self.bottom_right_index)
        return ()

    @property
    def requested_fixed_vertex_indices(self) -> tuple[int, ...]:
        top = tuple(self.top_row_indices)
        bottom = (self.bottom_left_index, self.bottom_right_index)
        if self.structural_topology == "A0_TOP_EDGE_ONLY":
            # A0 normally fixes only the top edge, but the E benchmark also
            # registers single-corner anchor variants.  Their declared anchor
            # vertices are part of the fixed set and must be audited as such.
            return tuple(sorted(set(top + self.requested_bottom_anchor_indices)))
        if self.structural_topology == "A1_TOP_EDGE_BOTTOM_TWO_CORNERS":
            return top + bottom
        if self.structural_topology == "A2_FOUR_CORNERS_ONLY":
            return (0, self.columns - 1, self.bottom_left_index, self.bottom_right_index)
        bottom_edge = tuple(range((self.rows - 1) * self.columns, self.rows * self.columns))
        perimeter = set(top + bottom_edge)
        perimeter.update(row * self.columns for row in range(self.rows))
        perimeter.update(row * self.columns + self.columns - 1 for row in range(self.rows))
        return tuple(sorted(perimeter))


@dataclass(frozen=True)
class SurfaceSheetInfo:
    """Paths and rest-state data for a generated sheet."""

    prim_path: str
    support_path: str | None
    attachment_path: str | None
    material_path: str
    rest_points: tuple[tuple[float, float, float], ...]
    top_row_indices: tuple[int, ...]
    support_aabb_min: tuple[float, float, float]
    support_aabb_max: tuple[float, float, float]
    support_center_z_m: float
    attachment_overlap_offset_m: float
    attachment_paths: tuple[str, ...]
    bottom_anchor_support_paths: tuple[str, ...]
    bottom_anchor_attachment_paths: tuple[str, ...]
    bottom_left_index: int
    bottom_right_index: int
    requested_bottom_anchor_indices: tuple[int, ...]
    structural_topology: str
    requested_fixed_vertex_indices: tuple[int, ...]
    constraint_objects: tuple[dict[str, object], ...]


def create_surface_sheet(
    stage,
    prim_path: str,
    config: CurtainConfig,
    *,
    create_attachment: bool = True,
) -> SurfaceSheetInfo:
    """Create a triangular sheet using Isaac Sim 5.1's surface API.

    Surface deformables do not expose partial kinematic targets in the current
    experimental wrapper, so the top row is fixed through an auto attachment to
    a static support bar.
    """
    config.validate()
    try:
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade
        from omni.physx.scripts import deformableUtils
    except ImportError as exc:
        raise RuntimeError(f"Isaac Sim 5.1 surface-deformable API is unavailable: {exc}") from exc

    mesh = UsdGeom.Mesh.Define(stage, Sdf.Path(prim_path))
    origin = tuple(float(value) for value in config.translation_m)
    yaw = float(config.yaw_rad)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)

    def transform_local_point(point: tuple[float, float, float]) -> tuple[float, float, float]:
        x, y, z = point
        return (
            origin[0] + cosine * x - sine * y,
            origin[1] + sine * x + cosine * y,
            origin[2] + z,
        )

    vertices: list[Gf.Vec3f] = []
    rest_points: list[tuple[float, float, float]] = []
    top_row_indices: list[int] = []
    for row in range(config.rows):
        z = config.support_z_m - config.height_m * row / (config.rows - 1)
        for col in range(config.columns):
            x = config.width_m * (col / (config.columns - 1) - 0.5)
            point = (float(x), 0.0, float(z))
            vertices.append(Gf.Vec3f(*point))
            rest_points.append(transform_local_point(point))
            if row == 0:
                top_row_indices.append(row * config.columns + col)
    faces = []
    for row in range(config.rows - 1):
        for col in range(config.columns - 1):
            i = row * config.columns + col
            j = i + 1
            k = i + config.columns
            l = k + 1
            faces.extend(((i, j, l), (i, l, k)))
    mesh.CreatePointsAttr(vertices)
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
    mesh.CreateSubdivisionSchemeAttr().Set("none")
    mesh_xform = UsdGeom.XformCommonAPI(mesh)
    mesh_xform.SetTranslate(Gf.Vec3d(*origin))
    mesh_xform.SetRotate(Gf.Vec3f(0.0, 0.0, math.degrees(yaw)))

    if not deformableUtils.set_physics_surface_deformable_body(stage, prim_path):
        raise RuntimeError(f"failed to apply the Isaac Sim 5.1 surface schema to {prim_path}")
    # A deformable body can initialize in a curtain-only scene without this
    # schema, but PhysX cannot parse it as a collision actor when a rigid G1
    # body is present. Apply the public collision schema before attachments.
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())

    # The helper's current signature only applies the body/collision schemas.
    # Material values are authored through the matching current helper.
    material_path = f"{prim_path}_PhysicsMaterial"
    material = UsdShade.Material.Define(stage, Sdf.Path(material_path))
    if not deformableUtils.add_surface_deformable_material(
        stage,
        material_path,
        density=config.density_kg_m3,
        static_friction=0.5,
        dynamic_friction=0.4,
        surface_thickness=config.surface_thickness_m,
        surface_stretch_stiffness=config.stretch_stiffness,
        surface_shear_stiffness=config.shear_stiffness,
        surface_bend_stiffness=config.bend_stiffness,
    ):
        raise RuntimeError(f"failed to create the surface material at {material_path}")
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )

    # These attributes belong to the current OmniPhysics deformable schema.
    body = mesh.GetPrim()
    body.CreateAttribute("omniphysics:solverPositionIterationCount", Sdf.ValueTypeNames.UInt).Set(
        config.solver_position_iterations
    )
    body.CreateAttribute("omniphysics:vertexVelocityDamping", Sdf.ValueTypeNames.Float).Set(config.damping)

    attachment_paths: list[str] = []
    support_paths: list[str] = []
    constraint_objects: list[dict[str, object]] = []

    def define_support(
        name: str,
        local_center: tuple[float, float, float],
        local_size: tuple[float, float, float],
    ) -> str:
        path = f"{prim_path}_{name}"
        support = UsdGeom.Cube.Define(stage, Sdf.Path(path))
        support.CreateSizeAttr(1.0)
        world_center = transform_local_point(local_center)
        xform = UsdGeom.XformCommonAPI(support)
        xform.SetTranslate(Gf.Vec3d(*world_center))
        xform.SetRotate(Gf.Vec3f(0.0, 0.0, math.degrees(yaw)))
        xform.SetScale(Gf.Vec3f(*local_size))
        UsdPhysics.CollisionAPI.Apply(support.GetPrim())
        support_paths.append(path)
        return path

    def create_attachment_to_support(path: str, target_support_path: str) -> str:
        if not deformableUtils.create_auto_deformable_attachment(
            stage,
            Sdf.Path(path),
            Sdf.Path(prim_path),
            Sdf.Path(target_support_path),
        ):
            raise RuntimeError(f"failed to attach {prim_path} to {target_support_path}")

        attachment_prim = stage.GetPrimAtPath(path)
        attachment_attributes = {
            "physxAutoDeformableAttachment:enableDeformableVertexAttachments": True,
            "physxAutoDeformableAttachment:deformableVertexOverlapOffset": config.attachment_overlap_offset_m,
            "physxAutoDeformableAttachment:enableRigidSurfaceAttachments": (
                config.attachment_enable_rigid_surface_attachments
            ),
        }
        for name, value in attachment_attributes.items():
            attribute = attachment_prim.GetAttribute(name)
            if not attribute:
                raise RuntimeError(f"missing attachment attribute: {name}")
            attribute.Set(value)

        # The helper creates low-level attachment/filter prims before custom
        # overlap values are authored. Refresh them once after final metadata.
        from omni.physx import get_physx_attachment_private_interface

        private_interface = get_physx_attachment_private_interface()
        if not private_interface.update_auto_deformable_attachment(path):
            raise RuntimeError(f"failed to update attachment metadata at {path}")
        attachment_paths.append(path)
        constraint_objects.append({"attachment_path": path, "support_path": target_support_path})
        return path

    support_path: str | None = None
    attachment_path: str | None = None
    bottom_anchor_support_paths: list[str] = []
    bottom_anchor_attachment_paths: list[str] = []
    top_z = config.support_z_m
    bottom_z = config.support_z_m - config.height_m
    half_width = 0.5 * config.width_m
    support_width_m = config.width_m + 2.0 * config.support_thickness_m
    support_depth_m = config.support_thickness_m
    support_height_m = config.support_thickness_m
    support_center_z = top_z - config.attachment_overlap_offset_m + support_height_m * 0.5

    def define_edge_support(name: str, z: float) -> str:
        return define_support(
            name,
            (0.0, 0.0, z - config.attachment_overlap_offset_m + support_height_m * 0.5),
            (support_width_m, support_depth_m, support_height_m),
        )

    def define_corner_support(name: str, local_x: float, z: float) -> str:
        return define_support(
            name,
            (local_x, 0.0, z),
            (config.bottom_anchor_size_m, config.bottom_anchor_size_m, config.bottom_anchor_size_m),
        )

    if create_attachment:
        if config.structural_topology in ("A0_TOP_EDGE_ONLY", "A1_TOP_EDGE_BOTTOM_TWO_CORNERS"):
            support_path = define_edge_support("Support", top_z)
            attachment_path = create_attachment_to_support(f"{prim_path}_Attachment", support_path)
        elif config.structural_topology == "A3_FULL_PERIMETER_FIXED":
            support_path = define_edge_support("TopSupport", top_z)
            attachment_path = create_attachment_to_support(f"{prim_path}_TopAttachment", support_path)
            bottom_support = define_edge_support("BottomSupport", bottom_z)
            bottom_anchor_support_paths.append(bottom_support)
            bottom_anchor_attachment_paths.append(
                create_attachment_to_support(f"{prim_path}_BottomAttachment", bottom_support)
            )
            side_size = (config.support_thickness_m, support_depth_m, config.height_m + 2.0 * config.support_thickness_m)
            for name, local_x in (("Left", -half_width), ("Right", half_width)):
                side_support = define_support(
                    name + "SideSupport",
                    (local_x, 0.0, 0.5 * (top_z + bottom_z)),
                    side_size,
                )
                bottom_anchor_support_paths.append(side_support)
                bottom_anchor_attachment_paths.append(
                    create_attachment_to_support(f"{prim_path}_{name}SideAttachment", side_support)
                )
        elif config.structural_topology == "A2_FOUR_CORNERS_ONLY":
            for name, local_x, z in (
                ("TopLeft", -half_width, top_z),
                ("TopRight", half_width, top_z),
                ("BottomLeft", -half_width, bottom_z),
                ("BottomRight", half_width, bottom_z),
            ):
                support = define_corner_support(name + "CornerSupport", local_x, z)
                attachment = create_attachment_to_support(f"{prim_path}_{name}CornerAttachment", support)
                if z == bottom_z:
                    bottom_anchor_support_paths.append(support)
                    bottom_anchor_attachment_paths.append(attachment)
                elif attachment_path is None:
                    attachment_path = attachment
        else:
            raise ValueError(f"unsupported structural topology: {config.structural_topology}")

        if config.structural_topology == "A1_TOP_EDGE_BOTTOM_TWO_CORNERS":
            for name, local_x in (("BottomLeft", -half_width), ("BottomRight", half_width)):
                anchor_support_path = define_corner_support(name + "AnchorSupport", local_x, bottom_z)
                bottom_anchor_support_paths.append(anchor_support_path)
                bottom_anchor_attachment_paths.append(
                    create_attachment_to_support(f"{prim_path}_{name}Attachment", anchor_support_path)
                )
        elif config.structural_topology == "A0_TOP_EDGE_ONLY" and config.requested_bottom_anchor_indices:
            for name, local_x in (("BottomLeft", -half_width), ("BottomRight", half_width)):
                if (name == "BottomLeft" and config.bottom_anchor_mode not in ("left", "both")) or (
                    name == "BottomRight" and config.bottom_anchor_mode not in ("right", "both")
                ):
                    continue
                anchor_support_path = define_corner_support(name + "AnchorSupport", local_x, bottom_z)
                bottom_anchor_support_paths.append(anchor_support_path)
                bottom_anchor_attachment_paths.append(
                    create_attachment_to_support(f"{prim_path}_{name}Attachment", anchor_support_path)
                )

    support_half_extents = (
        support_width_m * 0.5,
        support_depth_m * 0.5,
        support_height_m * 0.5,
    )
    support_corners = [
        transform_local_point((sx, sy, support_center_z + sz))
        for sx in (-support_half_extents[0], support_half_extents[0])
        for sy in (-support_half_extents[1], support_half_extents[1])
        for sz in (-support_half_extents[2], support_half_extents[2])
    ]
    support_aabb_min = tuple(min(point[index] for point in support_corners) for index in range(3))
    support_aabb_max = tuple(max(point[index] for point in support_corners) for index in range(3))

    return SurfaceSheetInfo(
        prim_path=prim_path,
        support_path=support_path,
        attachment_path=attachment_path,
        material_path=material_path,
        rest_points=tuple(rest_points),
        top_row_indices=tuple(top_row_indices),
        support_aabb_min=tuple(float(value) for value in support_aabb_min),
        support_aabb_max=tuple(float(value) for value in support_aabb_max),
        support_center_z_m=float(origin[2] + support_center_z),
        attachment_overlap_offset_m=float(config.attachment_overlap_offset_m),
        attachment_paths=tuple(attachment_paths),
        bottom_anchor_support_paths=tuple(bottom_anchor_support_paths),
        bottom_anchor_attachment_paths=tuple(bottom_anchor_attachment_paths),
        bottom_left_index=config.bottom_left_index,
        bottom_right_index=config.bottom_right_index,
        requested_bottom_anchor_indices=config.requested_bottom_anchor_indices,
        structural_topology=config.structural_topology,
        requested_fixed_vertex_indices=config.requested_fixed_vertex_indices,
        constraint_objects=tuple(constraint_objects),
    )
