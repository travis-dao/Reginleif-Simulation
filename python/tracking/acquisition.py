"""Target acquisition: hover the crosshair over the object, press space to lock/unlock."""

GRACE_SECONDS = 0.25          # auto-unlock if the target is lost this long (same clock as `now`)


def object_zone(bbox):
    """Circular zone centered on the detected object: (cx, cy, radius)."""
    x, y, w, h = bbox
    return x + w // 2, y + h // 2, int(min(w, h) / 0.5)


def crosshair_point(frame_shape):
    """Center of the frame (the crosshair)."""
    h, w = frame_shape[:2]
    return w // 2, h // 2


def in_zone(point, zone):
    """True if the point lies inside the circular zone."""
    px, py = point
    cx, cy, r = zone
    return (px - cx) ** 2 + (py - cy) ** 2 <= r * r


class TargetLock:
    """Toggle lock: space locks while hovering, space again unlocks.

    Also unlocks automatically if the target stays undetected longer than GRACE_SECONDS.
    """

    def __init__(self):
        self.locked = False
        self.last_detection = 0.0

    def unlock(self):
        self.locked = False

    def update(self, hover, detected, now, toggle):
        """Returns (started, stopped)."""
        was_locked = self.locked

        if toggle:
            if self.locked:
                self.locked = False
            elif hover and detected:
                self.locked = True
                self.last_detection = now
        elif self.locked:
            if detected:
                self.last_detection = now
            elif now - self.last_detection > GRACE_SECONDS:
                self.locked = False

        return (self.locked and not was_locked), (was_locked and not self.locked)