"""Geometry-only bounded probe-site selection for a known candidate region."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ProbeSite:
    name: str
    xy_m: tuple[float, float]
    approach_yaw_rad: float
    lateral_offset_m: float
    reason: str


def propose_probe_sites(
    *,
    candidate_x_limits_m: tuple[float, float],
    candidate_y_limits_m: tuple[float, float],
    precontact_offset_m: float,
    lateral_offset_m: float,
    workspace_y_limits_m: tuple[float, float],
    body_clearance_y_limits_m: tuple[float, float] | None = None,
) -> tuple[ProbeSite, ...]:
    """Return at most two sites, clamped to declared map/workspace geometry.

    The function does not know whether the candidate is a curtain or a rigid
    object.  It only uses map footprint, approach geometry, and robot-owned
    workspace limits.
    """
    x0, x1 = map(float, candidate_x_limits_m)
    y0, y1 = map(float, candidate_y_limits_m)
    wy0, wy1 = map(float, workspace_y_limits_m)
    if not all(math.isfinite(v) for v in (x0, x1, y0, y1, wy0, wy1, precontact_offset_m, lateral_offset_m)):
        raise ValueError("probe geometry must be finite")
    if x0 > x1 or y0 > y1 or wy0 > wy1 or precontact_offset_m <= 0.0 or lateral_offset_m <= 0.0:
        raise ValueError("probe geometry limits are invalid")
    safe0, safe1 = wy0, wy1
    if body_clearance_y_limits_m is not None:
        safe0 = max(safe0, float(body_clearance_y_limits_m[0]))
        safe1 = min(safe1, float(body_clearance_y_limits_m[1]))
    center = 0.5 * (y0 + y1)
    candidates = (("left", center + lateral_offset_m), ("right", center - lateral_offset_m))
    sites: list[ProbeSite] = []
    for name, y in candidates:
        if safe0 <= y <= safe1 and y0 <= y <= y1:
            sites.append(ProbeSite(
                name=f"{name}_offset",
                xy_m=(x0 - precontact_offset_m, y),
                approach_yaw_rad=0.0,
                lateral_offset_m=float(y - center),
                reason="candidate_footprint_and_robot_workspace_intersection",
            ))
    return tuple(sites)


def select_next_site(sites: tuple[ProbeSite, ...], attempted_names: set[str]) -> ProbeSite | None:
    for site in sites:
        if site.name not in attempted_names:
            return site
    return None
