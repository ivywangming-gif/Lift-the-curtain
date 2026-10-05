"""Small scene descriptors shared by the M2/M3 scripts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BarrierKind(str, Enum):
    CURTAIN = "curtain"
    RIGID_PANEL = "rigid_panel"


@dataclass(frozen=True)
class BarrierSpec:
    kind: BarrierKind
    center_xyz: tuple[float, float, float] = (1.5, 0.0, 1.0)
    width_m: float = 1.5
    height_m: float = 1.8
    thickness_m: float = 0.04
    appearance_rgb: tuple[float, float, float] = (0.55, 0.22, 0.08)


@dataclass(frozen=True)
class GoalSpec:
    position_xyz: tuple[float, float, float] = (3.0, 0.0, 0.0)
    radius_m: float = 0.35
