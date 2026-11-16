# mjpython python/tracking/main.py

"""
Green-object "look" detector, used by main.py (and runnable standalone).

    1. Threshold the frame in HSV to find green pixels.
    2. Clean the mask and take the largest blob as the object.
    3. "Looking" = the blob's center is inside the circle at the middle of the
       frame and has stayed there for the dwell time.

Standalone:
    python3 obj_detection.py [--camera N | --source clip.mp4]
Keys: q/ESC quit, p print HSV range, m toggle mask window.
"""

import argparse
import time

import cv2
import numpy as np

CONFIG_WINDOW = "Configuration"
VIEW_WINDOW = "Look Detector"
MASK_WINDOW = "Mask"

# OpenCV hue runs 0-179 (green is ~35-85).
DEFAULT_HSV = {
    "H low": 25, "H high": 85,
    "S low": 80, "S high": 255,
    "V low": 50, "V high": 255,
}
GRACE_SECONDS = 0.25          # tolerate brief detection dropouts


def nothing(_value):
    pass


def create_config_window():
    cv2.namedWindow(CONFIG_WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow(CONFIG_WINDOW, 420, 360)
    for name in ("H low", "H high"):
        cv2.createTrackbar(name, CONFIG_WINDOW, DEFAULT_HSV[name], 179, nothing)
    for name in ("S low", "S high", "V low", "V high"):
        cv2.createTrackbar(name, CONFIG_WINDOW, DEFAULT_HSV[name], 255, nothing)
    cv2.createTrackbar("Min size 0.1%", CONFIG_WINDOW, 5, 200, nothing)
    cv2.createTrackbar("Dwell x0.1s", CONFIG_WINDOW, 0, 50, nothing)
    # macOS: a trackbar-only window can misbehave without an image.
    cv2.imshow(CONFIG_WINDOW, np.zeros((1, 420, 3), dtype=np.uint8))


def get_config():
    pos = lambda name: cv2.getTrackbarPos(name, CONFIG_WINDOW)
    return {
        "lower": np.array([pos("H low"), pos("S low"), pos("V low")], dtype=np.uint8),
        "upper": np.array([pos("H high"), pos("S high"), pos("V high")], dtype=np.uint8),
        "min_area_frac": pos("Min size 0.1%") / 1000.0,
        "dwell_s": max(1, pos("Dwell x0.1s")) / 10.0,
    }


DETECT_WIDTH = 480

# Create this once instead of allocating a kernel every frame.
MORPH_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (3, 3)
)

def find_object(frame_bgr, config):
    """Return the largest green object's bbox in original-frame coordinates."""
    height, width = frame_bgr.shape[:2]

    # Downsample to reduce the number of pixels processed.
    if width > DETECT_WIDTH:
        scale = DETECT_WIDTH / width
        small = cv2.resize(
            frame_bgr,
            (DETECT_WIDTH, int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        scale = 1.0
        small = frame_bgr

    # Threshold directly; skip GaussianBlur.
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv, config["lower"], config["upper"]
    )

    # One lightweight cleanup operation.
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, MORPH_KERNEL
    )

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, mask

    best = max(contours, key=cv2.contourArea)

    # Scale the minimum area threshold to the processed image.
    min_area = (
        config["min_area_frac"] * height * width * scale * scale
    )
    if cv2.contourArea(best) < min_area:
        return None, mask

    x, y, w, h = cv2.boundingRect(best)

    # Return coordinates that match the original frame.
    if scale != 1.0:
        x = round(x / scale)
        y = round(y / scale)
        w = round(w / scale)
        h = round(h / scale)

    return (x, y, w, h), mask

def object_zone(bbox):
    """Circular zone centered on the detected object: (cx, cy, radius)."""
    x, y, w, h = bbox
    return x + w // 2, y + h // 2, int(min(w, h) / 2)


def crosshair_point(frame_shape):
    """Center of the frame (the crosshair)."""
    h, w = frame_shape[:2]
    return w // 2, h // 2


def in_zone(point, zone):
    """True if the point lies inside the circular zone."""
    px, py = point
    cx, cy, r = zone
    return (px - cx) ** 2 + (py - cy) ** 2 <= r * r

def dist(bbox):
    x, y, w, h = bbox
    pass


class DwellTracker:
    """Turns per-frame hits into a stable 'looking' state."""

    def __init__(self):
        self.start = None
        self.last_hit = 0.0
        self.looking = False

    def update(self, hit, now, dwell_s):
        if hit:
            if self.start is None:
                self.start = now
            self.last_hit = now
        elif self.start is not None and now - self.last_hit > GRACE_SECONDS:
            self.start = None

        elapsed = 0.0 if self.start is None else now - self.start
        was_looking = self.looking
        self.looking = elapsed >= dwell_s
        progress = min(1.0, elapsed / dwell_s)
        return progress, (self.looking and not was_looking), (was_looking and not self.looking)

def draw_overlay(frame_bgr, bbox, crosshair, zone, progress, looking):
    out = frame_bgr.copy()
    zone_color = (0, 255, 0) if looking else (255, 255, 255)

    # Crosshair at the frame center
    kx, ky = crosshair
    cv2.drawMarker(out, (kx, ky), zone_color, cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)

    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 165, 255), 2)

        # Detection circle on the object
        cx, cy, r = zone
        cv2.circle(out, (cx, cy), r, zone_color, 2, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 4, (0, 165, 255), -1, cv2.LINE_AA)

    height, width = out.shape[:2]
    bar_w = int(width * progress)
    cv2.rectangle(out, (0, height - 12), (width, height), (40, 40, 40), -1)
    cv2.rectangle(out, (0, height - 12), (bar_w, height),
                  (0, 255, 0) if looking else (0, 200, 255), -1)

    label = "LOOKING AT TARGET" if looking else "searching..."
    cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 0) if looking else (255, 255, 255), 2)
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--source", help="Video file path (overrides --camera)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source if args.source else args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the video source.")

    # Request a minimal capture buffer where supported.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    create_config_window()
    cv2.namedWindow(VIEW_WINDOW, cv2.WINDOW_NORMAL)
    tracker = DwellTracker()
    show_mask = True

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        config = get_config()

        t0 = time.perf_counter()
        bbox, mask = find_object(frame, config)
        detect_ms = (time.perf_counter() - t0) * 1000

        crosshair = crosshair_point(frame.shape)
        zone = object_zone(bbox) if bbox is not None else None
        hit = zone is not None and in_zone(crosshair, zone)

        progress, started, stopped = tracker.update(hit, time.monotonic(), config["dwell_s"])

        if started:
            print("Now looking at the target")
        if stopped:
            print("Looked away")

        cv2.putText(
            frame,
            f"Detection: {detect_ms:.1f} ms",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

        cv2.imshow(VIEW_WINDOW, draw_overlay(frame, bbox, crosshair, zone, progress, tracker.looking))
        if show_mask:
            cv2.imshow(MASK_WINDOW, mask)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("m"):
            show_mask = not show_mask
            if not show_mask:
                cv2.destroyWindow(MASK_WINDOW)
        if key == ord("p"):
            print(f"HSV lower={config['lower'].tolist()} upper={config['upper'].tolist()}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()