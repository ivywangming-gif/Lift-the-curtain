"""Simulator-independent recording request and deformable-visual audit helpers.

This module does not open a renderer or create a video.  It gives replay
launchers a stable request schema and makes the distinction between a
requested recording and a verified MP4 explicit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True)
class RecordingSpec:
    scenario: str
    output_dir: Path
    camera_view: str = "front"
    width: int = 320
    height: int = 240
    fps: int = 30

    def validate(self) -> None:
        if not self.scenario:
            raise ValueError("scenario must be non-empty")
        if self.width < 16 or self.height < 16:
            raise ValueError("recording resolution must be at least 16x16")
        if self.fps < 1 or self.fps > 240:
            raise ValueError("fps must be between 1 and 240")
        if not self.camera_view.strip():
            raise ValueError("camera_view must be non-empty")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        result = asdict(self)
        result["output_dir"] = str(self.output_dir.expanduser().resolve())
        result["resolution"] = f"{self.width}x{self.height}"
        return result


def parse_resolution(value: str) -> tuple[int, int]:
    normalized = value.lower().replace(" ", "").replace(",", "x")
    parts = normalized.split("x")
    if len(parts) != 2:
        raise ValueError("resolution must use WIDTHxHEIGHT, for example 320x240")
    try:
        width, height = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("resolution must use integer WIDTHxHEIGHT") from exc
    return width, height


def write_recording_request(path: Path, spec: RecordingSpec, *, status: str = "REQUESTED_NOT_STARTED") -> None:
    spec.validate()
    payload = {
        "schema": "lift-the-curtain.recording-request.v1",
        "status": status,
        "verified_video": False,
        "spec": spec.to_dict(),
        "notes": "A renderer backend must consume this request; its existence is not a video artifact.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_state_sidecar(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    """Write a stable CSV sidecar that can be synchronized with video frames."""

    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("state sidecar requires at least one row")
    fieldnames = list(dict.fromkeys(key for row in materialized for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def classify_deformable_visual_sync(
    runtime_displacement_m: float | None,
    visual_displacement_m: float | None,
    *,
    tolerance_m: float = 0.005,
) -> str:
    """Flag a runtime/visual mismatch without using it for control decisions."""

    values = (runtime_displacement_m, visual_displacement_m, tolerance_m)
    if any(value is not None and not math.isfinite(float(value)) for value in values):
        return "NOT_ASSESSABLE"
    if runtime_displacement_m is None or visual_displacement_m is None:
        return "NOT_ASSESSABLE"
    if runtime_displacement_m > tolerance_m and visual_displacement_m <= tolerance_m:
        return "VISUAL_DEFORMABLE_MISMATCH"
    return "CONSISTENT_WITHIN_TOLERANCE"
