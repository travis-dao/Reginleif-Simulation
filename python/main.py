# mjpython python/main.py

import time
import numpy as np
import mujoco
import mujoco.viewer
import argparse

from leg.leg_controller import LegController
from motion.mpc import MPC, MPCConfig
from motion.core import Core
from motion.state_estimator import make_imu   # sim only: MuJoCo stands in for the IMU
from motion.stabilizer import Stabilizer, Config
from build_sim_xml import build_sim_xml
from input.input import InputController
from input.visualizer import InputVisualizer

# print(mujoco.__version__)

SRC = "assembly/xml/reginleif.xml"
DST = "assembly/xml/reginleif_sim.xml"

KP = 30.0
FORCE = 9.4
START_HEIGHT = 0.5

HORIZON = 10
MPC_DT = 0.03
SETTLE_TIME_S = 0.0       # let the robot drop and stand on the kinematic gait before MPC engages
MPC_GAIN = 1.3            # mm of foot offset per N. Set 0.0 to run the pure gait (isolates MPC from gait problems)
SETTLE_VEL = 0.1          # ...and require the base to be (nearly) at rest

REST_S = 2.0              # stand still
WALK_S = 5.0              # then walk

build_sim_xml(SRC, DST, START_HEIGHT, KP, FORCE)

model = mujoco.MjModel.from_xml_path(DST)
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


cfg = MPCConfig(
    horizon=HORIZON,
    dt=MPC_DT,
    mass=float(model.body_subtreemass[body_id]),   # whole robot, not just the trunk
    I_body=subtree_inertia(model, data, body_id),  # base is level at t=0, so world axes == body axes
    R_weight=1e-3,                                 # larger = smoother forces, less chatter
    mu=0.4,                                        # keep <= foot friction in the XML
    # [roll pitch yaw | x y z | wx wy wz | vx vy vz | g]
    # Q_diag=np.array([5, 5, 30, 
    #                  1, 1, 50, 
    #                  1, 1, 10, 
    #                  1, 1, 1, 
    #                  0.0]),
    Q_diag=np.array([5, 5, 10, 
                     1, 1, 50, 
                     1, 1, 1, 
                     1, 1, 1, 
                     0.0]),
)
mpc_solver = MPC(cfg)
print(f"MPC mass={cfg.mass:.2f} kg, I=\n{np.round(cfg.I_body, 4)}")

controller = LegController(model, data)
controller.init()

stabilizer = Stabilizer(Config(
    K_p=0.0,
    K_i=0.0,
    K_d=0.0
), model, data, "base")

# z_des is latched from the measured standing height once the robot has settled
core = Core(mpc_solver, controller, z_des_m=0.0, solve_period_s=MPC_DT,
            gain_mm_per_n=MPC_GAIN, gravity_z_mps2=float(model.opt.gravity[2]))

# start the estimator at the known spawn pose (the estimate lives in this world frame from here on)
_jnt = model.body_jntadr[body_id]
_qadr = model.jnt_qposadr[_jnt]
_vadr = model.jnt_dofadr[_jnt]
core.reset(data.qpos[_qadr:_qadr + 3].copy(), np.zeros(3), make_imu(model, data, body_id))

input_controller = InputController()
visualizer = InputVisualizer()

import math
def smoothing(x: float, p: float = 2.0) -> float:
    if abs(x) <= 1e-3:
        return math.copysign(float("inf"), x)
    return math.copysign(0.1 / (abs(x) ** p), x)

def sim():
    # ground truth is read ONLY for this printout (sim diagnostic); the MPC never sees it
    EST_DEBUG_PERIOD_S = 1.0   # print estimate vs ground truth this often (0 = off)
    last_est_print = 0.0
    mpc_active = False
    last_print = 0.0
    pid_out = None


    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            dt = model.opt.timestep

            input_controller.update(dt)
            command = input_controller.get_input()

            visualizer.update(command)

            # Update state estimate every control step.
            imu = make_imu(model, data, body_id)
            s = core.estimate(dt, imu)

            if data.time - last_est_print >= 0.1:
                last_est_print = data.time

                # print(f"\rt={data.time:6.2f}s | rpy={np.array2string(imu.rpy_rad * 180.0 / math.pi, precision=3)}",
                #     flush=True,
                # )

            # if EST_DEBUG_PERIOD_S > 0 and data.time - last_est_print > EST_DEBUG_PERIOD_S:
            #     last_est_print = data.time
            #     pos_err = s.pos - data.qpos[_qadr:_qadr + 3]
            #     vel_err = s.linear_vel - data.qvel[_vadr:_vadr + 3]
            #     print(
            #         f"t={data.time:.2f}: "
            #         f"est pos err={np.round(pos_err, 4)}  "
            #         f"vel err={np.round(vel_err, 4)}"
            #     )

            # Command schedule: rest -> walk -> rest.
            input = input_controller.get_input()
            V = input.y * controller.v_max_mm_s * 1e-3     # m/s
            R = smoothing(-input.x)                                  # path radius, m; use float("inf") for straight; 0.1 = turning in place
            print(R)
            if REST_S <= data.time:
                core.v_cmd_mps = V
                core.yaw_rate_cmd = controller.v_max_mm_s * 1e-3 / R if abs(input.y) < 1e-3 else V / R
            else:
                core.v_cmd_mps = core.yaw_rate_cmd = 0.0

            # Engage MPC once the robot has settled.
            if not mpc_active:
                if (data.time > SETTLE_TIME_S and np.linalg.norm(s.linear_vel) < SETTLE_VEL):
                    core.z_des = float(s.pos[2])
                    core.yaw_des = s.yaw
                    mpc_active = True
                    core._t_engaged = 0.0

                    print(f"MPC engaged at t={data.time:.2f}s, z_des={core.z_des:.3f} m")

            # Solve MPC at most once per simulation step.
            if mpc_active:
                res = core.step(dt, data.time)

                if (res is not None and not res.success and data.time - last_print > 1.0 ):
                    print(f"t={data.time:.2f}: MPC solve failed, holding last offsets")
                    last_print = data.time

            # Update the PID from the latest completed physics state.
            # Stabilizer.update() returns the rotation matrix for the legs.
            pid_out = stabilizer.update(dt_s=dt)

            # Update each leg, apply PID correction, and command MuJoCo joints.
            controller.update(dt, pid_out)

            # Advance physics using the newly commanded joint angles.
            mujoco.mj_step(model, data)

            viewer.sync()

            sleep = dt - (time.time() - step_start)
            if sleep > 0:
                time.sleep(sleep)

if __name__ == "__main__":
    sim()

    input_controller.stop()
    visualizer.close()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--export",
        nargs="+",
        choices=["mpc", "pos"],
    )

    args = parser.parse_args()

    if args.export:
        if "mpc" in args.export:
            core.export_log()

        if "pos" in args.export:
            controller.export_pos_log([0])