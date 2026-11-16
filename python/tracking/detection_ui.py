"""OpenCV UI for the detector: trackbars, overlay, and a standalone demo.

Standalone:
    python3 detection_ui.py [--camera N | --source clip.mp4]
Keys: q/ESC quit, p print HSV range, m toggle mask window.
"""

import argparse
import time

import cv2
import numpy as np

from obj_detection import DetectorConfig, ObjectDetector, DEFAULT_LOWER, DEFAULT_UPPER

CONFIG_WINDOW = "Configuration"
VIEW_WINDOW = "Look Detector"
MASK_WINDOW = "Mask"


def _nothing(_v):
    pass


def create_config_window():
    cv2.namedWindow(CONFIG_WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow(CONFIG_WINDOW, 420, 360)
    for i, ax in enumerate(("H", "S", "V")):
        top = 179 if ax == "H" else 255
        cv2.createTrackbar(f"{ax} low", CONFIG_WINDOW, DEFAULT_LOWER[i], top, _nothing)
        cv2.createTrackbar(f"{ax} high", CONFIG_WINDOW, DEFAULT_UPPER[i], top, _nothing)
    cv2.createTrackbar("Min size 0.1%", CONFIG_WINDOW, 3, 200, _nothing)
    # macOS: a trackbar-only window can misbehave without an image.
    cv2.imshow(CONFIG_WINDOW, np.zeros((1, 420, 3), dtype=np.uint8))


def read_config(config: DetectorConfig) -> None:
    """Update the detector config in place from the trackbars."""
    pos = lambda n: cv2.getTrackbarPos(n, CONFIG_WINDOW)
    config.lower[:] = [pos("H low"), pos("S low"), pos("V low")]
    config.upper[:] = [pos("H high"), pos("S high"), pos("V high")]
    config.min_area_frac = pos("Min size 0.1%") / 1000.0


def draw_overlay(frame_bgr, detection, locked=False, hover=False, zone=None):
    """zone is an optional (cx, cy, radius) circle from acquisition.object_zone.

    Colors: white = idle, yellow = crosshair hovering the target, green = locked.
    """
    out = frame_bgr.copy()
    if locked:
        color = (0, 255, 0)
    elif hover:
        color = (0, 255, 255)      # BGR yellow
    else:
        color = (255, 255, 255)
    h, w = out.shape[:2]
    cv2.drawMarker(out, (w // 2, h // 2), color, cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
    if detection.found:
        x, y, bw, bh = detection.bbox
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 165, 255), 2)
        cx, cy = detection.center
        cv2.circle(out, (int(cx), int(cy)), 4, (0, 165, 255), -1, cv2.LINE_AA)
        if zone is not None:
            zx, zy, zr = zone
            cv2.circle(out, (zx, zy), zr, color, 2, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--source", help="Video file path (overrides --camera)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source if args.source else args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the video source.")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    detector = ObjectDetector()
    create_config_window()
    cv2.namedWindow(VIEW_WINDOW, cv2.WINDOW_NORMAL)
    show_mask = True

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        read_config(detector.config)

        t0 = time.perf_counter()
        det = detector.detect(frame)
        ms = (time.perf_counter() - t0) * 1000

        view = draw_overlay(frame, det)
        cv2.putText(view, f"Detection: {ms:.1f} ms", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.imshow(VIEW_WINDOW, view)
        if show_mask:
            cv2.imshow(MASK_WINDOW, det.mask)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("m"):
            show_mask = not show_mask
            if not show_mask:
                cv2.destroyWindow(MASK_WINDOW)
        if key == ord("p"):
            c = detector.config
            print(f"HSV lower={c.lower.tolist()} upper={c.upper.tolist()}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()