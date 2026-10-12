"""Round 13 scene variants kept separate from deployment decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any

from .curtain_asset import CurtainConfig
from .task_scale_scene import FormalTaskSceneConfig, RigidPanelSpec


class BottomAnchorMode(str, Enum):
    NONE = "none"
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"


class Round13ScenarioName(str, Enum):
    NOMINAL = "nominal"
    HALF_DENSITY = "half_density"
    DOUBLE_DENSITY = "double_density"
    BOTTOM_LEFT = "bottom_left_anchor"
    BOTTOM_RIGHT = "bottom_right_anchor"
    BOTH_BOTTOM = "both_bottom_corners"
    HIDDEN_RIGID = "curtain_hidden_rigid"
    RIGID = "rigid_panel"


@dataclass(frozen=True)
class HiddenRigidSpec:
    x_m: float = 1.08
    y_m: float = 0.0
    bottom_z_m: float = 0.0
    width_m: float = 1.20
    height_m: float = 2.00
    thickness_m: float = 0.04

    @property
    def center_xyz(self) -> tuple[float, float, float]:
        return (self.x_m, self.y_m, self.bottom_z_m + 0.5 * self.height_m)


@dataclass(frozen=True)
class Round13ScenarioSpec:
    name: Round13ScenarioName
    scene_object: str
    density_scale: float = 1.0
    bottom_anchor_mode: BottomAnchorMode = BottomAnchorMode.NONE
    hidden_rigid: HiddenRigidSpec | None = None
    map_assumption: str = "candidate_2_with_zero_only_detour"

    def validate(self) -> None:
        if self.scene_object not in ("curtain", "rigid"):
            raise ValueError("scene_object must be curtain or rigid")
        if self.density_scale <= 0.0:
            raise ValueError("density_scale must be positive")
        if self.scene_object == "rigid" and self.bottom_anchor_mode != BottomAnchorMode.NONE:
            raise ValueError("rigid scenario cannot declare cloth anchors")
        if self.hidden_rigid is not None and self.scene_object != "curtain":
            raise ValueError("hidden rigid obstruction requires a front curtain")

    def curtain_config(self, formal: FormalTaskSceneConfig | None = None) -> CurtainConfig | None:
        self.validate()
        if self.scene_object != "curtain":
            return None
        base = (formal or FormalTaskSceneConfig()).curtain_config()
        return replace(
            base,
            density_kg_m3=base.density_kg_m3 * self.density_scale,
            bottom_anchor_mode=self.bottom_anchor_mode.value,
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result = asdict(self)
        result["name"] = self.name.value
        result["bottom_anchor_mode"] = self.bottom_anchor_mode.value
        if self.hidden_rigid is not None:
            result["hidden_rigid_center_xyz"] = list(self.hidden_rigid.center_xyz)
        return result


def round13_scenarios() -> dict[Round13ScenarioName, Round13ScenarioSpec]:
    specs = {
        Round13ScenarioName.NOMINAL: Round13ScenarioSpec(Round13ScenarioName.NOMINAL, "curtain"),
        Round13ScenarioName.HALF_DENSITY: Round13ScenarioSpec(
            Round13ScenarioName.HALF_DENSITY, "curtain", density_scale=0.5
        ),
        Round13ScenarioName.DOUBLE_DENSITY: Round13ScenarioSpec(
            Round13ScenarioName.DOUBLE_DENSITY, "curtain", density_scale=2.0
        ),
        Round13ScenarioName.BOTTOM_LEFT: Round13ScenarioSpec(
            Round13ScenarioName.BOTTOM_LEFT, "curtain", bottom_anchor_mode=BottomAnchorMode.LEFT
        ),
        Round13ScenarioName.BOTTOM_RIGHT: Round13ScenarioSpec(
            Round13ScenarioName.BOTTOM_RIGHT, "curtain", bottom_anchor_mode=BottomAnchorMode.RIGHT
        ),
        Round13ScenarioName.BOTH_BOTTOM: Round13ScenarioSpec(
            Round13ScenarioName.BOTH_BOTTOM, "curtain", bottom_anchor_mode=BottomAnchorMode.BOTH
        ),
        Round13ScenarioName.HIDDEN_RIGID: Round13ScenarioSpec(
            Round13ScenarioName.HIDDEN_RIGID,
            "curtain",
            hidden_rigid=HiddenRigidSpec(),
            map_assumption="H1_full_oracle_or_H2_occlusion_limited_must_be_declared_per_episode",
        ),
        Round13ScenarioName.RIGID: Round13ScenarioSpec(Round13ScenarioName.RIGID, "rigid"),
    }
    for spec in specs.values():
        spec.validate()
    return specs


def scenario_by_name(name: str) -> Round13ScenarioSpec:
    try:
        key = Round13ScenarioName(str(name))
    except ValueError as exc:
        choices = ", ".join(item.value for item in Round13ScenarioName)
        raise ValueError(f"unknown Round 13 scenario {name!r}; choices: {choices}") from exc
    return round13_scenarios()[key]


def matched_rigid_panel() -> RigidPanelSpec:
    """Return the Round 12 panel geometry without inferring passability."""
    panel = FormalTaskSceneConfig().rigid_panel
    panel.validate()
    return panel
