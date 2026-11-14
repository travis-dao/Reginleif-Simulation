import math


# -------------------------- Smoothing configuration --------------------------

X0, Y0 = 0.00, 1.00  # Start
X1, Y1 = 0.25, 0.50  # First breakpoint
X2, Y2 = 0.75, 0.25  # Second breakpoint
X3, Y3 = 1.00, 0.10  # End


def smoothing(x: float) -> float:
    sign = math.copysign(1.0, x)
    x = abs(x)

    if x < X1:
        x_start, y_start = X0, Y0
        x_end, y_end = X1, Y1
    elif x < X2:
        x_start, y_start = X1, Y1
        x_end, y_end = X2, Y2
    elif x < X3:
        x_start, y_start = X2, Y2
        x_end, y_end = X3, Y3
    else:
        return sign * Y3

    # Linear interpolation between the selected breakpoints
    t = (x - x_start) / (x_end - x_start)
    y = y_start + t * (y_end - y_start)

    return sign * y