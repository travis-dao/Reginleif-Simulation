import math

# cycle input cw at max
def cycle(time_s: float, cycle_s: float = 5.0):
    phase = time_s % cycle_s
    angle_r = math.radians(-(phase / cycle_s * 360.0))

    return math.cos(angle_r), math.sin(angle_r)