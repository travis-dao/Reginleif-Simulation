# mjpython python/sim/model.py

import time
import numpy as np
import mujoco
import mujoco.viewer

import sys
from pathlib import Path

# Add the python/ directory to the module search path.
PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

from movement.rl.policy_controller import PolicyController
from movement.leg.movement import LegDriver
from movement.motion.imu import Imu
from sim.util.build_sim_xml import build_sim_xml
import sim.config as cfg
from sim.util.input_map import map_input_to_command
from input.input import InputController
from input.visualizer import InputVisualizer, StateVisualizer

build_sim_xml(cfg.SRC, cfg.DST, cfg.START_HEIGHT, cfg.KP, cfg.FORCE)

model = mujoco.MjModel.from_xml_path(cfg.DST)
data = mujoco.MjData(model)

body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")

policy = PolicyController(model, data, "python/rl/policy/bc_policy.pt")
imu = Imu(model, data, "base")

def reset_to_neutral(model, data, policy, body_id, imu, height_m=cfg.STAND_HEIGHT):
    mujoco.mj_resetData(model, data)                 # clears time, velocities, warm-start, contacts; qpos = XML spawn pose

    # joint qpos addresses, resolved through the same actuator names LegDriver / PolicyController use
    q_adr = []
    for leg in LegDriver.LEG_NAMES:
        for j in LegDriver.JOINTS:
            jid = int(model.actuator_trnid[policy.driver.act_id[f"{leg}_{j}"], 0])
            q_adr.append(int(model.jnt_qposadr[jid]))
    data.qpos[q_adr] = policy.q_default              # q_default is stored in the checkpoint

    b = int(model.jnt_qposadr[model.body_jntadr[body_id]])
    data.qpos[b + 2] = height_m                      # keep the spawn x, y; set the stand height
    data.qpos[b + 3:b + 7] = [1.0, 0.0, 0.0, 0.0]    # level, yaw 0
    data.qvel[:] = 0.0

    policy.step(0.0, imu.read())                     # not engaged yet -> writes q_default to the servo targets
    mujoco.mj_forward(model, data)

reset_to_neutral(model, data, policy, body_id, imu)

# start the estimator at the known spawn pose (the estimate lives in this world frame from here on)
_jnt = model.body_jntadr[body_id]
_qadr = model.jnt_qposadr[_jnt]

input_controller = InputController()
# input_visualizer = InputVisualizer()
state_visualizer = StateVisualizer()

def sim():
    mpc_active = False

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            dt = model.opt.timestep

            input_controller.update(dt)
            command = input_controller.get_input()
            # input_visualizer.update(command)

            imu.update(dt)

            input = input_controller.get_input()
            if cfg.REST_S <= data.time:
                v_cmd_mps, yaw_rate_cmd = map_input_to_command(input.x, input.y, 66.667)
            else:
                v_cmd_mps = yaw_rate_cmd = 0.0

            # Engage MPC once the robot has settled.
            if not policy.engaged and data.time > cfg.SETTLE_TIME_S:
                    policy.engage()

            if policy.engaged:
                policy.step(dt, imu.read())
                # print(f"v_cmd={v_cmd_mps}, yaw_cmd={yaw_rate_cmd}")
                policy.set_command(v_cmd_mps, yaw_rate_cmd)


            mujoco.mj_step(model, data)
            viewer.sync()
            sleep = dt - (time.time() - step_start)
            if sleep > 0:
                time.sleep(sleep)

if __name__ == "__main__":
    sim()

    input_controller.stop()
    # input_visualizer.close()
    state_visualizer.close()