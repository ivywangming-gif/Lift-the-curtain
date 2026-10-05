"""Current Isaac Sim surface-deformable curtain configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CurtainConfig:
    width_m: float = 1.4
    height_m: float = 1.8
    columns: int = 16
    rows: int = 20
    density_kg_m2: float = 0.25
    stretch_stiffness: float = 500.0
    bend_stiffness: float = 0.02
    damping: float = 0.05
    support_z_m: float = 1.8

    def validate(self) -> None:
        if self.width_m <= 0.0 or self.height_m <= 0.0:
            raise ValueError("curtain dimensions must be positive")
        if self.columns < 2 or self.rows < 2:
            raise ValueError("curtain mesh resolution must be at least 2x2")
        if min(self.density_kg_m2, self.stretch_stiffness, self.bend_stiffness, self.damping) < 0.0:
            raise ValueError("curtain physical parameters cannot be negative")


def create_surface_sheet(stage, prim_path: str, config: CurtainConfig):
    """Create a triangular sheet through Isaac Sim 5.1's surface API."""
    config.validate()
    try:
        from pxr import Gf, Sdf, UsdGeom
        from isaacsim.core.experimental.prims import DeformablePrim
        from isaacsim.core.utils import deformableUtils
    except ImportError as exc:
        raise RuntimeError("Isaac Sim 5.1 surface-deformable API is unavailable") from exc

    mesh = UsdGeom.Mesh.Define(stage, Sdf.Path(prim_path))
    vertices = []
    for row in range(config.rows):
        z = config.support_z_m - config.height_m * row / (config.rows - 1)
        for col in range(config.columns):
            x = config.width_m * (col / (config.columns - 1) - 0.5)
            vertices.append(Gf.Vec3f(x, 0.0, z))
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
    deformableUtils.set_physics_surface_deformable_body(
        stage, prim_path, simulation_hexahedral_resolution=0,
        self_collision=False, solver_position_iteration_count=8,
    )
    return DeformablePrim(prim_paths_expr=[prim_path])
