# mjpython python/main.py

import time
import numpy as np
import mujoco
import mujoco.viewer

from leg.leg_controller import LegController
from mpc.mpc import State, Input, Model, DiscreteModel, MPCProblem, MPCResult, MPCConfig, MPC
from mpc.read_state import read_state
from build_sim_xml import build_sim_xml

SRC = "assembly/xml/reginleif.xml"        # your original file
DST = "assembly/xml/reginleif_sim.xml"     # must be saved in the same folder as SRC

KP = 30.0                   # servo stiffness (tune)
FORCE = 9.4                 # max joint torque in N*m (set to your servo spec)
START_HEIGHT = 0.5         # drop height of the base, adjust to your geometry

build_sim_xml(SRC, DST, START_HEIGHT, KP, FORCE)

model = mujoco.MjModel.from_xml_path(DST)
data = mujoco.MjData(model)

controller = LegController(model, data)
controller.init()

body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "base"
)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # update gaits

        # read current model state
        read_state(model, data, body_id, debug=False)

        # mpc.solve

        # update foot position
        controller.update(0.0005, None)

        mujoco.mj_step(model, data)
        viewer.sync()
        sleep = model.opt.timestep - (time.time() - step_start)
        if sleep > 0:
            time.sleep(sleep)