"""Round 8 formal task-scale scene descriptors.

This module contains geometry/configuration only.  It does not expose object
identity or simulator state to deployment decision code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


@dataclass(frozen=True)
class RigidPanelSpec:
    """A fixed panel used as the matched interaction candidate."""

    x_m: float = 0.85
    y_m: float = 0.0
    bottom_z_m: float = 0.0
    width_m: float = 1.20
    height_m: float = 2.00
    thickness_m: float = 0.04

    @property
    def center_xyz(self) -> tuple[float, float, float]:
        return (self.x_m, self.y_m, self.bottom_z_m + self.height_m * 0.5)

    @property
    def top_z_m(self) -> float:
        return self.bottom_z_m + self.height_m

    def validate(self) -> None:
        values = (self.x_m, self.y_m, self.bottom_z_m, self.width_m, self.height_m, self.thickness_m)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("rigid panel geometry must be finite")
        if min(self.width_m, self.height_m, self.thickness_m) <= 0.0:
            raise ValueError("rigid panel dimensions must be positive")
        if self.bottom_z_m < 0.0:
            raise ValueError("rigid panel cannot start below the ground plane")


@dataclass(frozen=True)
class DoorwaySpec:
    """Static side boundaries used when auditing the rigid detour geometry."""

    x_m: float = 1.45
    clearance_m: float = 0.70
    post_thickness_m: float = 0.08
    height_m: float = 2.05
    outer_width_m: float = 2.00

    @property
    def inner_y_limits_m(self) -> tuple[float, float]:
        half = self.clearance_m * 0.5
        return (-half, half)

    @property
    def post_centers_y_m(self) -> tuple[float, float]:
        half = self.clearance_m * 0.5 + self.post_thickness_m * 0.5
        return (-half, half)

    @property
    def top_beam_width_m(self) -> float:
        return self.clearance_m + 2.0 * self.post_thickness_m

    def validate(self) -> None:
        values = (
            self.x_m,
            self.clearance_m,
            self.post_thickness_m,
            self.height_m,
            self.outer_width_m,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("doorway geometry must be finite")
        if self.clearance_m < 0.70:
            raise ValueError("doorway clearance must be at least 0.70 m for Round 8")
        if min(self.post_thickness_m, self.height_m, self.outer_width_m) <= 0.0:
            raise ValueError("doorway dimensions must be positive")
        if self.outer_width_m < self.clearance_m + 2.0 * self.post_thickness_m:
            raise ValueError("outer doorway width is smaller than the opening plus posts")


@dataclass(frozen=True)
class FormalTaskSceneConfig:
    """Formal 2 m single-curtain scene used by Round 8 experiments."""

    scene_name: str = "round8_formal_2m_single_curtain"
    curtain_x_m: float = 0.85
    curtain_y_m: float = 0.0
    curtain_top_z_m: float = 2.02
    curtain_length_m: float = 2.00
    curtain_width_m: float = 1.20
    curtain_yaw_rad: float = math.pi / 2.0
    curtain_columns: int = 25
    curtain_rows: int = 33
    support_thickness_m: float = 0.04
    attachment_overlap_offset_m: float = 0.01
    rigid_panel: RigidPanelSpec = RigidPanelSpec()
    doorway: DoorwaySpec = DoorwaySpec()

    @property
    def curtain_bottom_z_m(self) -> float:
        return self.curtain_top_z_m - self.curtain_length_m

    @property
    def curtain_top_world_xyz(self) -> tuple[float, float, float]:
        return (self.curtain_x_m, self.curtain_y_m, self.curtain_top_z_m)

    @property
    def curtain_bottom_world_xyz(self) -> tuple[float, float, float]:
        return (self.curtain_x_m, self.curtain_y_m, self.curtain_bottom_z_m)

    @property
    def curtain_y_limits_m(self) -> tuple[float, float]:
        half = self.curtain_width_m * 0.5
        return (self.curtain_y_m - half, self.curtain_y_m + half)

    def validate(self) -> None:
        values = (
            self.curtain_x_m,
            self.curtain_y_m,
            self.curtain_top_z_m,
            self.curtain_length_m,
            self.curtain_width_m,
            self.curtain_yaw_rad,
            self.support_thickness_m,
            self.attachment_overlap_offset_m,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("formal curtain geometry must be finite")
        if min(self.curtain_length_m, self.curtain_width_m, self.support_thickness_m) <= 0.0:
            raise ValueError("formal curtain dimensions must be positive")
        if self.curtain_bottom_z_m < 0.0:
            raise ValueError("formal curtain bottom must not be below the ground plane")
        if abs(self.curtain_bottom_z_m - 0.02) > 1.0e-6:
            raise ValueError("Round 8 formal curtain bottom clearance must be 0.02 m")
        if self.curtain_columns < 2 or self.curtain_rows < 2:
            raise ValueError("curtain mesh resolution must be at least 2x2")
        self.rigid_panel.validate()
        self.doorway.validate()

    def curtain_config(self) -> Any:
        """Return the existing verified surface-sheet config for this scene."""
        from .curtain_asset import CurtainConfig

        self.validate()
        return CurtainConfig(
            width_m=self.curtain_width_m,
            height_m=self.curtain_length_m,
            columns=self.curtain_columns,
            rows=self.curtain_rows,
            support_z_m=self.curtain_top_z_m,
            support_thickness_m=self.support_thickness_m,
            attachment_overlap_offset_m=self.attachment_overlap_offset_m,
            translation_m=(self.curtain_x_m, self.curtain_y_m, 0.0),
            yaw_rad=self.curtain_yaw_rad,
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result = asdict(self)
        result["curtain_bottom_z_m"] = self.curtain_bottom_z_m
        result["curtain_y_limits_m"] = list(self.curtain_y_limits_m)
        result["curtain_top_world_xyz"] = list(self.curtain_top_world_xyz)
        result["curtain_bottom_world_xyz"] = list(self.curtain_bottom_world_xyz)
        result["rigid_panel_center_xyz"] = list(self.rigid_panel.center_xyz)
        result["rigid_panel_top_z_m"] = self.rigid_panel.top_z_m
        result["doorway_inner_y_limits_m"] = list(self.doorway.inner_y_limits_m)
        result["doorway_post_centers_y_m"] = list(self.doorway.post_centers_y_m)
        return result


def old_155m_regression_config(*, wall_x_m: float = 0.85) -> Any:
    """Return the old Round 7 1.55 m curtain without changing its fixture."""
    from .curtain_asset import CurtainConfig

    return CurtainConfig(
        width_m=1.20,
        height_m=1.55,
        columns=25,
        rows=25,
        support_z_m=1.85,
        support_thickness_m=0.04,
        attachment_overlap_offset_m=0.01,
        translation_m=(float(wall_x_m), 0.0, 0.0),
        yaw_rad=math.pi / 2.0,
    )


def old_155m_regression_summary(*, wall_x_m: float = 0.85) -> dict[str, Any]:
    config = old_155m_regression_config(wall_x_m=wall_x_m)
    config.validate()
    return {
        "fixture_name": "round7_legacy_1p55m_curtain",
        "width_m": config.width_m,
        "height_m": config.height_m,
        "support_z_m": config.support_z_m,
        "bottom_z_m": config.support_z_m - config.height_m,
        "translation_m": list(config.translation_m),
        "yaw_rad": config.yaw_rad,
        "columns": config.columns,
        "rows": config.rows,
        "preserved": True,
    }
