"""Simple color/depth candidate-barrier detector; no learned perception."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BarrierDetectorConfig:
    hsv_lower: tuple[int, int, int] = (5, 80, 40)
    hsv_upper: tuple[int, int, int] = (35, 255, 255)
    min_depth_m: float = 0.15
    max_depth_m: float = 8.0
    min_area_px: int = 100


@dataclass(frozen=True)
class BarrierDetection:
    detected: bool
    centroid_px: tuple[float, float] | None
    bbox_px: tuple[int, int, int, int] | None
    depth_m: float | None
    lateral_extent_px: tuple[int, int] | None
    mask_area_px: int


class BarrierDetector:
    """HSV threshold plus connected-component and depth filtering."""

    def __init__(self, config: BarrierDetectorConfig | None = None):
        self.config = config or BarrierDetectorConfig()

    def detect(self, rgb: np.ndarray, depth_m: np.ndarray) -> BarrierDetection:
        rgb = np.asarray(rgb)
        depth = np.asarray(depth_m, dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or depth.shape != rgb.shape[:2]:
            raise ValueError("rgb must be HxWx3 and depth_m must be HxW")
        if not np.isfinite(depth).any():
            return BarrierDetection(False, None, None, None, None, 0)
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("BarrierDetector requires OpenCV at runtime") from exc
        hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(
            hsv, np.asarray(self.config.hsv_lower, dtype=np.uint8),
            np.asarray(self.config.hsv_upper, dtype=np.uint8),
        ).astype(bool)
        mask &= np.isfinite(depth)
        mask &= depth >= self.config.min_depth_m
        mask &= depth <= self.config.max_depth_m
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        best = None
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= self.config.min_area_px and (best is None or area > best[0]):
                best = (area, label)
        if best is None:
            return BarrierDetection(False, None, None, None, None, 0)
        area, label = best
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        valid_depth = depth[labels == label]
        return BarrierDetection(
            True, (float(centroids[label, 0]), float(centroids[label, 1])),
            (x, y, width, height),
            float(np.median(valid_depth)) if valid_depth.size else None,
            (x, x + width - 1), area,
        )
