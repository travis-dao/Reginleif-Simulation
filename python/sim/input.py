# python python/sim/input.py

import sys
from pathlib import Path

# Add the python/ directory to the module search path.
PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

import mujoco
import time
import csv

from input.input import InputController
from input.visualizer import InputVisualizer
from input.smooth import smoothing
from input.cycle import cycle
import sim.config as cfg
from movement.leg.leg_controller import LegController

input_controller = InputController()
input_visualizer = InputVisualizer()

model = mujoco.MjModel.from_xml_path(cfg.DST)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

lc = LegController(model, data)

log = []

import math

# name, input.x, input.y
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


print(generate_walk_commands())

try:
    while input_visualizer.running:
        step_start = time.time()
        dt = model.opt.timestep

        input_controller.update(dt, cycle(data.time))
        input = input_controller.get_input()

        V = input.y * lc.v_max_mm_s * 1e-3     # m/s
        R = smoothing(-input.x)                # path radius, m; 0.1 = turning in place
        if cfg.REST_S <= data.time:
            v_cmd_mps = V
            if abs(input.y) < 1e-3 and abs(input.x) > 1e-3:
                yaw_rate_cmd = lc.v_max_mm_s * 1e-3 / R
            elif abs(input.x) >= 1e-3:
                yaw_rate_cmd = V / R
            else:
                yaw_rate_cmd = 0
        else:
            v_cmd_mps = yaw_rate_cmd = 0.0

        input_visualizer.update(input)

        row = {
            "time_s" : data.time,
            "input_x" : input.x,
            "input_y" : input.y,
            "V": V,
            "R": R,
            "v_cmd": v_cmd_mps,
            "yaw_rate_cmd": yaw_rate_cmd
        }
        log.append(row)

        # Update your MuJoCo simulation here.
        mujoco.mj_step(model, data)
        sleep = dt - (time.time() - step_start)
        if sleep > 0:
            time.sleep(sleep)
finally:
    input_controller.stop()
    input_visualizer.close()

    with open("python/sim/log/input_command_mapping.csv", mode="w", newline="", encoding="utf-8") as file:
        fieldnames = ["time_s", "input_x", "input_y", "V", "R", "v_cmd", "yaw_rate_cmd"]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log)