"""Green-object detector. Detection only: no windows, no state, no tracking logic."""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

# OpenCV hue runs 0-179 (green is ~35-85).
DEFAULT_LOWER = (25, 80, 50)
DEFAULT_UPPER = (85, 255, 255)


@dataclass
class DetectorConfig:
    lower: np.ndarray = field(default_factory=lambda: np.array(DEFAULT_LOWER, dtype=np.uint8))
    upper: np.ndarray = field(default_factory=lambda: np.array(DEFAULT_UPPER, dtype=np.uint8))
    min_area_frac: float = 0.003      # fraction of the frame area


@dataclass
class Detection:
    bbox: Optional[Tuple[int, int, int, int]]   # (x, y, w, h) in original-frame pixels
    mask: np.ndarray                            # processed-resolution mask, for debugging

    @property
    def found(self) -> bool:
        return self.bbox is not None

    @property
    def center(self) -> Optional[Tuple[float, float]]:
        if self.bbox is None:
            return None
        x, y, w, h = self.bbox
        return x + w / 2.0, y + h / 2.0


class ObjectDetector:
    def __init__(self, config: Optional[DetectorConfig] = None, detect_width: int = 480):
        self.config = config or DetectorConfig()
        self.detect_width = detect_width
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def detect(self, frame_bgr: np.ndarray) -> Detection:
        """Find the largest green blob. Returns bbox in original-frame coordinates."""
        height, width = frame_bgr.shape[:2]

        if width > self.detect_width:
            scale = self.detect_width / width
            small = cv2.resize(frame_bgr, (self.detect_width, int(height * scale)),
                               interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            small = frame_bgr

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.config.lower, self.config.upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return Detection(None, mask)

        best = max(contours, key=cv2.contourArea)
        min_area = self.config.min_area_frac * height * width * scale * scale
        if cv2.contourArea(best) < min_area:
            return Detection(None, mask)

        x, y, w, h = cv2.boundingRect(best)
        if scale != 1.0:
            x, y, w, h = (round(v / scale) for v in (x, y, w, h))
        return Detection((x, y, w, h), mask)