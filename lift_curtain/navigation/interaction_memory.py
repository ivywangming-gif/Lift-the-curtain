"""Episode-level spatial memory for limited physical probing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math

import numpy as np


class InteractionStatus(str, Enum):
    UNKNOWN = "UNKNOWN"
    MOVABLE = "MOVABLE"
    PASSABLE = "PASSABLE"
    BLOCKED = "BLOCKED"
    DEFERRED_THIS_EPISODE = "DEFERRED_THIS_EPISODE"
    CURRENTLY_CLEAR = "CURRENTLY_CLEAR"


@dataclass(frozen=True)
class InteractionMemoryConfig:
    max_probes_per_episode: int = 1
    spatial_merge_radius_m: float = 0.35
    normal_merge_angle_rad: float = 1.0471975512

    def validate(self) -> None:
        if self.max_probes_per_episode < 0:
            raise ValueError("max_probes_per_episode cannot be negative")
        if self.spatial_merge_radius_m <= 0.0 or self.normal_merge_angle_rad <= 0.0:
            raise ValueError("memory spatial thresholds must be positive")


@dataclass
class InteractionRecord:
    center_xy: tuple[float, float]
    normal_yaw_rad: float
    status: InteractionStatus = InteractionStatus.UNKNOWN
    attempts: int = 0
    last_time_s: float | None = None
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["status"] = self.status.value
        return result


def _wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


class InteractionMemory:
    """Persist probe decisions by spatial region, not detector component ID."""

    def __init__(self, config: InteractionMemoryConfig | None = None):
        self.config = config or InteractionMemoryConfig()
        self.config.validate()
        self.begin_episode()

    def begin_episode(self) -> None:
        self.records: list[InteractionRecord] = []
        self.probes_used = 0

    @property
    def probes_remaining(self) -> int:
        return max(0, self.config.max_probes_per_episode - self.probes_used)

    def _find_record(self, center_xy: tuple[float, float], normal_yaw_rad: float) -> InteractionRecord | None:
        center = np.asarray(center_xy, dtype=np.float64)
        for record in self.records:
            distance = float(np.linalg.norm(center - np.asarray(record.center_xy, dtype=np.float64)))
            normal_error = abs(_wrap_angle(float(normal_yaw_rad) - record.normal_yaw_rad))
            if distance <= self.config.spatial_merge_radius_m and normal_error <= self.config.normal_merge_angle_rad:
                return record
        return None

    def status_at(self, center_xy: tuple[float, float], normal_yaw_rad: float = 0.0) -> InteractionStatus:
        record = self._find_record(center_xy, normal_yaw_rad)
        return record.status if record is not None else InteractionStatus.UNKNOWN

    def can_probe(
        self,
        center_xy: tuple[float, float],
        normal_yaw_rad: float = 0.0,
        *,
        candidate_id: str | None = None,
    ) -> bool:
        """Return whether this spatial entrance is eligible for one probe.

        ``candidate_id`` is intentionally accepted only for logging at the call
        site; it never resets or bypasses the spatial memory.
        """
        del candidate_id
        if self.probes_used >= self.config.max_probes_per_episode:
            return False
        return self._find_record(center_xy, normal_yaw_rad) is None

    def reserve_probe(
        self,
        center_xy: tuple[float, float],
        normal_yaw_rad: float = 0.0,
        *,
        time_s: float | None = None,
        candidate_id: str | None = None,
    ) -> bool:
        if not self.can_probe(center_xy, normal_yaw_rad, candidate_id=candidate_id):
            return False
        self.records.append(
            InteractionRecord(
                center_xy=(float(center_xy[0]), float(center_xy[1])),
                normal_yaw_rad=float(normal_yaw_rad),
                attempts=1,
                last_time_s=None if time_s is None else float(time_s),
            )
        )
        self.probes_used += 1
        return True

    def record_response(
        self,
        center_xy: tuple[float, float],
        status: InteractionStatus,
        *,
        normal_yaw_rad: float = 0.0,
        time_s: float | None = None,
        failure_reason: str | None = None,
        opening_verified: bool = False,
    ) -> InteractionRecord:
        record = self._find_record(center_xy, normal_yaw_rad)
        if record is None:
            raise KeyError("response has no reserved spatial probe")
        status = InteractionStatus(status)
        if status == InteractionStatus.PASSABLE and not opening_verified:
            # Movement alone is only MOVABLE; passability needs a current
            # multi-height free-corridor check.
            status = InteractionStatus.MOVABLE
        if status == InteractionStatus.BLOCKED:
            status = InteractionStatus.DEFERRED_THIS_EPISODE
        record.status = status
        record.last_time_s = None if time_s is None else float(time_s)
        record.failure_reason = failure_reason
        return record

    def mark_currently_clear(
        self,
        center_xy: tuple[float, float],
        *,
        normal_yaw_rad: float = 0.0,
        time_s: float | None = None,
        opening_verified: bool,
    ) -> InteractionRecord:
        if not opening_verified:
            raise ValueError("CURRENTLY_CLEAR requires a verified opening")
        record = self._find_record(center_xy, normal_yaw_rad)
        if record is None:
            raise KeyError("clear status has no reserved spatial probe")
        record.status = InteractionStatus.CURRENTLY_CLEAR
        record.last_time_s = None if time_s is None else float(time_s)
        return record

    def snapshot(self) -> dict[str, object]:
        return {
            "max_probes_per_episode": self.config.max_probes_per_episode,
            "probes_used": self.probes_used,
            "probes_remaining": self.probes_remaining,
            "records": [record.to_dict() for record in self.records],
        }
