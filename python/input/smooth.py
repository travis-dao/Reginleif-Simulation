import math


# -------------------------- Smoothing configuration --------------------------

# (x, y) breakpoints, ordered by increasing x
BREAKPOINTS = [
    (0.00, 1.00),  # Start
    (0.25, 0.75),
    (0.75, 0.4),
    (1.00, 0.10),  # End
]


def smoothing(x: float) -> float:
    sign = math.copysign(1.0, x)
    x = abs(x)

    for (x_start, y_start), (x_end, y_end) in zip(
        BREAKPOINTS, BREAKPOINTS[1:]
    ):
        if x < x_end:
            t = (x - x_start) / (x_end - x_start)
            y = y_start + t * (y_end - y_start)
            return sign * y

    # Clamp to the final breakpoint
    return sign * BREAKPOINTS[-1][1]