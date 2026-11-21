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
from sim.util.input_map import map_input_to_command, generate_walk_commands
from movement.leg.leg_controller import LegController

input_controller = InputController()
input_visualizer = InputVisualizer()

model = mujoco.MjModel.from_xml_path(cfg.DST)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

lc = LegController(model, data)

log = []

try:
    while input_visualizer.running:
        step_start = time.time()
        dt = model.opt.timestep

        input_controller.update(dt, cycle(data.time))
        input = input_controller.get_input()

        if cfg.REST_S <= data.time:
                v_cmd_mps, yaw_rate_cmd = map_input_to_command(input.x, input.y, V, R, lc.v_max_mm_s)
        else:
            v_cmd_mps = yaw_rate_cmd = 0.0

        input_visualizer.update(input)

        row = {
            "time_s" : data.time,
            "input_x" : input.x,
            "input_y" : input.y,
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