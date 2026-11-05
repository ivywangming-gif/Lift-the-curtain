"""Mechanically explicit multi-sheet scenes for physics-only validation.

The scene labels in this module are construction metadata.  They are never
available to a deployment policy.  Each sheet uses the already validated
surface-deformable builder; the new question is whether multiple independent
sheets remain numerically and mechanically valid together.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from lift_curtain.envs.curtain_asset import CurtainConfig
from lift_curtain.envs.task_scale_scene import FormalTaskSceneConfig


@dataclass(frozen=True)
class SheetSpec:
    name: str
    center_y_m: float
    width_m: float
    columns: int
    bottom_anchor_mode: str = "none"
    center_x_offset_m: float = 0.0

    def curtain_config(self, formal: FormalTaskSceneConfig) -> CurtainConfig:
        base = formal.curtain_config()
        config = replace(
            base,
            width_m=float(self.width_m),
            columns=int(self.columns),
            bottom_anchor_mode=self.bottom_anchor_mode,
            translation_m=(formal.curtain_x_m + float(self.center_x_offset_m), float(self.center_y_m), 0.0),
        )
        config.validate()
        return config


@dataclass(frozen=True)
class FlexibleSceneSpec:
    name: str
    family: str
    sheets: tuple[SheetSpec, ...]
    nominal_span_y_m: tuple[float, float] = (-0.60, 0.60)
    intended_overlap_m: float = 0.0
    purpose: str = ""

    def validate(self) -> None:
        if self.family not in {"F2_SPLIT_CENTER", "F3_OVERLAPPING_STRIPS"}:
            raise ValueError(f"unsupported scene family: {self.family}")
        if len(self.sheets) < 2:
            raise ValueError("a multi-sheet scene requires at least two sheets")
        names = [sheet.name for sheet in self.sheets]
        if len(names) != len(set(names)):
            raise ValueError("sheet names must be unique")
        formal = FormalTaskSceneConfig()
        for sheet in self.sheets:
            sheet.curtain_config(formal)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def scene_specs() -> dict[str, FlexibleSceneSpec]:
    # Facing +x, robot-left is +y and robot-right is -y.
    split_free = FlexibleSceneSpec(
        name="split_free",
        family="F2_SPLIT_CENTER",
        sheets=(
            SheetSpec("robot_right_sheet", -0.28, 0.64, 15),
            SheetSpec("robot_left_sheet", 0.28, 0.64, 15),
        ),
        intended_overlap_m=0.08,
        purpose="Two physically independent top-fixed sheets with free center edges.",
    )
    split_left_limited = FlexibleSceneSpec(
        name="split_left_limited",
        family="F2_SPLIT_CENTER",
        sheets=(
            SheetSpec("robot_right_sheet", -0.28, 0.64, 15),
            SheetSpec("robot_left_sheet", 0.28, 0.64, 15, "right"),
        ),
        intended_overlap_m=0.08,
        purpose="Robot-left (+y) sheet has an outer bottom anchor; right-contact geometry is unchanged.",
    )
    split_right_limited = FlexibleSceneSpec(
        name="split_right_limited",
        family="F2_SPLIT_CENTER",
        sheets=(
            SheetSpec("robot_right_sheet", -0.28, 0.64, 15, "left"),
            SheetSpec("robot_left_sheet", 0.28, 0.64, 15),
        ),
        intended_overlap_m=0.08,
        purpose="Robot-right (-y) sheet has an outer bottom anchor; mirrored mechanical control.",
    )
    strips = FlexibleSceneSpec(
        name="strips_3_overlap",
        family="F3_OVERLAPPING_STRIPS",
        sheets=(
            SheetSpec("strip_right", -0.36, 0.48, 11),
            SheetSpec("strip_center", 0.00, 0.48, 11),
            SheetSpec("strip_left", 0.36, 0.48, 11),
        ),
        intended_overlap_m=0.12,
        purpose="Three independent wide flexible strips; no visual transparency shortcut.",
    )
    split_free_layered = FlexibleSceneSpec(
        name="split_free_layered",
        family="F2_SPLIT_CENTER",
        sheets=(
            SheetSpec("robot_right_sheet", -0.28, 0.64, 15, "none", -0.004),
            SheetSpec("robot_left_sheet", 0.28, 0.64, 15, "none", 0.004),
        ),
        intended_overlap_m=0.08,
        purpose="Split-free scene with explicit 8 mm front/back layer separation in the overlap.",
    )
    split_left_limited_layered = FlexibleSceneSpec(
        name="split_left_limited_layered",
        family="F2_SPLIT_CENTER",
        sheets=(
            SheetSpec("robot_right_sheet", -0.28, 0.64, 15, "none", -0.004),
            SheetSpec("robot_left_sheet", 0.28, 0.64, 15, "right", 0.004),
        ),
        intended_overlap_m=0.08,
        purpose="Layered split scene with robot-left outer bottom anchor.",
    )
    split_right_limited_layered = FlexibleSceneSpec(
        name="split_right_limited_layered",
        family="F2_SPLIT_CENTER",
        sheets=(
            SheetSpec("robot_right_sheet", -0.28, 0.64, 15, "left", -0.004),
            SheetSpec("robot_left_sheet", 0.28, 0.64, 15, "none", 0.004),
        ),
        intended_overlap_m=0.08,
        purpose="Layered split scene with robot-right outer bottom anchor.",
    )
    strips_layered = FlexibleSceneSpec(
        name="strips_3_overlap_layered",
        family="F3_OVERLAPPING_STRIPS",
        sheets=(
            SheetSpec("strip_right", -0.36, 0.48, 11, "none", -0.006),
            SheetSpec("strip_center", 0.00, 0.48, 11, "none", 0.000),
            SheetSpec("strip_left", 0.36, 0.48, 11, "none", 0.006),
        ),
        intended_overlap_m=0.12,
        purpose="Alternating-depth three-strip construction with 6 mm layer spacing.",
    )
    split_free_layered24 = FlexibleSceneSpec(
        name="split_free_layered24",
        family="F2_SPLIT_CENTER",
        sheets=(
            SheetSpec("robot_right_sheet", -0.28, 0.64, 15, "none", -0.012),
            SheetSpec("robot_left_sheet", 0.28, 0.64, 15, "none", 0.012),
        ),
        intended_overlap_m=0.08,
        purpose="Split-free scene with 24 mm front/back separation exceeding the authored contact-offset envelope.",
    )
    split_left_limited_layered24 = FlexibleSceneSpec(
        name="split_left_limited_layered24",
        family="F2_SPLIT_CENTER",
        sheets=(
            SheetSpec("robot_right_sheet", -0.28, 0.64, 15, "none", -0.012),
            SheetSpec("robot_left_sheet", 0.28, 0.64, 15, "right", 0.012),
        ),
        intended_overlap_m=0.08,
        purpose="24 mm layered split scene with robot-left outer bottom anchor.",
    )
    split_right_limited_layered24 = FlexibleSceneSpec(
        name="split_right_limited_layered24",
        family="F2_SPLIT_CENTER",
        sheets=(
            SheetSpec("robot_right_sheet", -0.28, 0.64, 15, "left", -0.012),
            SheetSpec("robot_left_sheet", 0.28, 0.64, 15, "none", 0.012),
        ),
        intended_overlap_m=0.08,
        purpose="24 mm layered split scene with robot-right outer bottom anchor.",
    )
    strips_layered40 = FlexibleSceneSpec(
        name="strips_3_overlap_layered40",
        family="F3_OVERLAPPING_STRIPS",
        sheets=(
            SheetSpec("strip_right", -0.36, 0.48, 11, "none", -0.020),
            SheetSpec("strip_center", 0.00, 0.48, 11, "none", 0.000),
            SheetSpec("strip_left", 0.36, 0.48, 11, "none", 0.020),
        ),
        intended_overlap_m=0.12,
        purpose="Three alternating strips with 20 mm adjacent depth spacing, above the contact-offset envelope.",
    )
    result = {
        scene.name: scene
        for scene in (
            split_free,
            split_left_limited,
            split_right_limited,
            strips,
            split_free_layered,
            split_left_limited_layered,
            split_right_limited_layered,
            strips_layered,
            split_free_layered24,
            split_left_limited_layered24,
            split_right_limited_layered24,
            strips_layered40,
        )
    }
    # E names are explicit about the benchmark role.  These aliases preserve
    # the already validated construction while making the PVC/surrogate
    # distinction visible in the P1 manifest and reports.
    result["SPLIT_CENTER_CURTAIN"] = replace(result["split_free"], name="SPLIT_CENTER_CURTAIN")
    result["LEFT_OPEN_RIGHT_CONSTRAINED"] = replace(result["split_left_limited"], name="LEFT_OPEN_RIGHT_CONSTRAINED")
    result["RIGHT_OPEN_LEFT_CONSTRAINED"] = replace(result["split_right_limited"], name="RIGHT_OPEN_LEFT_CONSTRAINED")
    result["PVC_INDEPENDENT_STRIPS"] = replace(result["strips_3_overlap"], name="PVC_INDEPENDENT_STRIPS")
    for scene in result.values():
        scene.validate()
    return result


def scene_by_name(name: str) -> FlexibleSceneSpec:
    try:
        return scene_specs()[name]
    except KeyError as exc:
        raise ValueError(f"unknown problem-discovery scene {name!r}") from exc
