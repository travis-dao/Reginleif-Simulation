# python python/sim/util/input_mapper.py

import sys
from pathlib import Path

# Add the python/ directory to the module search path.
PYTHON_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PYTHON_DIR))

import math

from input.smooth import smoothing

V_MAX_MM_S = 66.667  # Set to your robot's actual maximum speed (mm/s)

TURN_STEPS = 5
MAX_ANGLE_RAD = math.radians(45.0)

def generate_walk_commands():
    commands = []

    for direction, sign_y in (("forward", 1.0), ("backward", -1.0)):
        for i in range(-TURN_STEPS, TURN_STEPS + 1):
            angle = (i / TURN_STEPS) * MAX_ANGLE_RAD

            input_x = math.sin(angle)
            input_y = sign_y * math.cos(angle)

            if i == 0:
                name = f"walk_{direction}"
            else:
                side = "right" if input_x > 0 else "left"
                name = f"walk_{direction}_{side}_{abs(math.degrees(angle)):.0f}deg"

            commands.append((name, input_x, input_y))

    return commands


def map_input_to_command(input_x, input_y, v_max_mm_s):
    V = input_y * v_max_mm_s * 1e-3     # m/s   
    R = smoothing(-input_x)

    v_cmd_mps = V
    if abs(input_y) < 1e-3 and abs(input_x) > 1e-3:
        yaw_rate_cmd = v_max_mm_s * 1e-3 / R
    elif abs(input_x) >= 1e-3:
        yaw_rate_cmd = V / R
    else:
        yaw_rate_cmd = 0
    return v_cmd_mps, yaw_rate_cmd

def main():
    results = []

    for name, input_x, input_y in generate_walk_commands():
        v_cmd_mps, yaw_rate_cmd = map_input_to_command(input_x, input_y, V_MAX_MM_S)
        results.append((name, input_x, input_y, v_cmd_mps, yaw_rate_cmd))

    print(f"{'Command':<30} {'Input X':>9} {'Input Y':>9} {'Cmd V (m/s)':>13} {'Yaw (rad/s)':>13}")
    print("-" * 100)
    for name, x, y, v_cmd, yaw_rate in results:
        print(f"{name:<30} {x:9.4f} {y:9.4f}  {v_cmd:13.4f} {yaw_rate:13.4f}")


if __name__ == "__main__":
    main()