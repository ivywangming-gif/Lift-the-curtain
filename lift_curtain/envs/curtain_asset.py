"""Small current-API surface-deformable curtain builder.

The MoDeSuite USD is kept outside this repository.  This module only builds the
minimal fallback used when that external asset cannot be loaded reliably.
"""

from __future__ import annotations

from dataclasses import dataclass


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
        ) < 0.0:
            raise ValueError("curtain physical parameters cannot be negative")
        if self.solver_position_iterations < 1:
            raise ValueError("solver_position_iterations must be positive")


@dataclass(frozen=True)
class SurfaceSheetInfo:
    """Paths and rest-state data for a generated sheet."""

    prim_path: str
    support_path: str
    attachment_path: str
    material_path: str
    rest_points: tuple[tuple[float, float, float], ...]
    top_row_indices: tuple[int, ...]


def create_surface_sheet(stage, prim_path: str, config: CurtainConfig) -> SurfaceSheetInfo:
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
    vertices: list[Gf.Vec3f] = []
    rest_points: list[tuple[float, float, float]] = []
    top_row_indices: list[int] = []
    for row in range(config.rows):
        z = config.support_z_m - config.height_m * row / (config.rows - 1)
        for col in range(config.columns):
            x = config.width_m * (col / (config.columns - 1) - 0.5)
            point = (float(x), 0.0, float(z))
            vertices.append(Gf.Vec3f(*point))
            rest_points.append(point)
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

    if not deformableUtils.set_physics_surface_deformable_body(stage, prim_path):
        raise RuntimeError(f"failed to apply the Isaac Sim 5.1 surface schema to {prim_path}")

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

    support_path = f"{prim_path}_Support"
    support = UsdGeom.Cube.Define(stage, Sdf.Path(support_path))
    support.CreateSizeAttr(1.0)
    support_xform = UsdGeom.XformCommonAPI(support)
    support_xform.SetTranslate(Gf.Vec3d(0.0, 0.0, config.support_z_m + config.support_thickness_m * 0.5))
    support_xform.SetScale(
        Gf.Vec3f(
            config.width_m * 0.5 + config.support_thickness_m,
            config.support_thickness_m,
            config.support_thickness_m * 0.5,
        )
    )
    UsdPhysics.CollisionAPI.Apply(support.GetPrim())

    attachment_path = f"{prim_path}_Attachment"
    if not deformableUtils.create_auto_deformable_attachment(
        stage,
        Sdf.Path(attachment_path),
        Sdf.Path(prim_path),
        Sdf.Path(support_path),
    ):
        raise RuntimeError(f"failed to attach {prim_path} to {support_path}")

    return SurfaceSheetInfo(
        prim_path=prim_path,
        support_path=support_path,
        attachment_path=attachment_path,
        material_path=material_path,
        rest_points=tuple(rest_points),
        top_row_indices=tuple(top_row_indices),
    )
