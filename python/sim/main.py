# mjpython python/sim/main.py

import time
import numpy as np
import mujoco
import mujoco.viewer

import sys
from pathlib import Path

# Add the python/ directory to the module search path.
PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

from movement.leg.leg_controller import LegController
from movement.leg.leg_config import Movement
from movement.motion.mpc import MPC, MPCConfig
from movement.motion.core import Core
from movement.motion.imu import Imu
from sim.util.build_sim_xml import build_sim_xml
import sim.config as cfg
from sim.util.input_map import map_input_to_command
from input.input import InputController
from input.visualizer import InputVisualizer, StateVisualizer

build_sim_xml(cfg.SRC, cfg.DST, cfg.START_HEIGHT, cfg.KP, cfg.FORCE)

model = mujoco.MjModel.from_xml_path(cfg.DST)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")


def subtree_inertia(model, data, root: int) -> np.ndarray:
    """Inertia (3x3, world axes) of the whole robot about its CoM."""
    com = data.subtree_com[root]
    I = np.zeros((3, 3))
    for b in range(model.nbody):
        if model.body_rootid[b] != model.body_rootid[root] or b == 0:
            continue
        R = data.ximat[b].reshape(3, 3)
        Ib = R @ np.diag(model.body_inertia[b]) @ R.T
        d = data.xipos[b] - com
        I += Ib + model.body_mass[b] * (d @ d * np.eye(3) - np.outer(d, d))
    return I


mpc_cfg = MPCConfig(
    horizon=cfg.HORIZON,
    dt=cfg.MPC_DT,
    mass=float(model.body_subtreemass[body_id]),   # whole robot, not just the trunk
    I_body=subtree_inertia(model, data, body_id),  # base is level at t=0, so world axes == body axes
    R_weight=1e-3,                                 # larger = smoother forces, less chatter
    mu=0.4,                                        # keep <= foot friction in the XML
    # [roll pitch yaw | x y z | wx wy wz | vx vy vz | g]
    Q_diag=np.array([30, 30, 10, 
                     1, 1, 10, 
                     1, 1, 1, 
                     1, 1, 20, 
                     0.0]),
)
mpc = MPC(mpc_cfg)
lc = LegController(model, data)
imu = Imu(model, data, "base")

# z_des is latched from the measured standing height once the robot has settled
core = Core(mpc, lc, z_des_m=0.0, solve_period_s=cfg.MPC_DT,
            gain_mm_per_n=cfg.MPC_GAIN, gravity_z_mps2=float(model.opt.gravity[2]), ramp_s=0.1)

# start the estimator at the known spawn pose (the estimate lives in this world frame from here on)
_jnt = model.body_jntadr[body_id]
_qadr = model.jnt_qposadr[_jnt]
core.reset(data.qpos[_qadr:_qadr + 3].copy(), np.zeros(3), imu.read())

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
            s = core.estimate(dt, imu.read())

            input = input_controller.get_input()
            if cfg.REST_S <= data.time:
                core.v_cmd_mps, core.yaw_rate_cmd = map_input_to_command(input.x, input.y, lc.v_max_mm_s)
            else:
                core.v_cmd_mps = core.yaw_rate_cmd = 0.0

            
            state_visualizer.update(
                swing_leg=Movement.index_name(lc.swing_leg),
                gait=Movement.order_name(lc._gait),
                vel_cmd=core.v_cmd_mps * 1e3,
                vel_real=lc._v,
                vel_est=core.estimator.state.linear_vel[0] * 1e3,
                yaw_cmd=core.yaw_rate_cmd,
                yaw_real=lc._w,
                leg_duty=lc.duty
            )

            # Engage MPC once the robot has settled.
            if not mpc_active:
                if (data.time > cfg.SETTLE_TIME_S and np.linalg.norm(s.linear_vel) < cfg.SETTLE_VEL):
                    core.z_des = float(s.pos[2])
                    core.yaw_des = s.yaw
                    mpc_active = True
                    core._t_engaged = 0.0

                    print(f"MPC engaged at t={data.time:.2f}s, z_des={core.z_des:.3f} m")
                    print(f"Base height: {data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_rot")][2]:.3f} m")

            if mpc_active:
                core.step(dt)

            lc.update(dt, core.estimator.state)

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