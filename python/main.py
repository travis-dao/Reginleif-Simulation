# mjpython python/main.py

import time
import numpy as np
import mujoco
import mujoco.viewer
import argparse

from leg.leg_controller import LegController
from mpc.mpc import MPC, MPCConfig
from core import Core
from state_estimator import make_imu   # sim only: MuJoCo stands in for the IMU
from build_sim_xml import build_sim_xml

SRC = "assembly/xml/reginleif.xml"
DST = "assembly/xml/reginleif_sim.xml"

KP = 30.0
FORCE = 9.4
START_HEIGHT = 0.5

HORIZON = 10
MPC_DT = 0.03
SETTLE_TIME_S = 0.25       # let the robot drop and stand on the kinematic gait before MPC engages
MPC_GAIN = 1.3            # mm of foot offset per N. Set 0.0 to run the pure gait (isolates MPC from gait problems)
SETTLE_VEL = 0.1          # ...and require the base to be (nearly) at rest

REST_S = 2.0              # stand still
WALK_S = 8.0              # then walk
TAIL_S = 4.0              # keep running after the stop so the final rest is visible
END_T = REST_S + WALK_S + TAIL_S

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

# z_des is latched from the measured standing height once the robot has settled
core = Core(mpc_solver, controller, z_des_m=0.0, solve_period_s=MPC_DT,
            gain_mm_per_n=MPC_GAIN, gravity_z_mps2=float(model.opt.gravity[2]))

# start the estimator at the known spawn pose (the estimate lives in this world frame from here on)
_jnt = model.body_jntadr[body_id]
_qadr = model.jnt_qposadr[_jnt]
_vadr = model.jnt_dofadr[_jnt]
core.reset(data.qpos[_qadr:_qadr + 3].copy(), np.zeros(3), make_imu(model, data, body_id))


def sim():
    # ground truth is read ONLY for this printout (sim diagnostic); the MPC never sees it
    EST_DEBUG_PERIOD_S = 1.0   # print estimate vs ground truth this often (0 = off)
    last_est_print = 0.0
    mpc_active = False
    last_print = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            dt = model.opt.timestep

            # Update state estimate every control step.
            s = core.estimate(dt, make_imu(model, data, body_id))

            if EST_DEBUG_PERIOD_S > 0 and data.time - last_est_print > EST_DEBUG_PERIOD_S:
                last_est_print = data.time
                pos_err = s.pos - data.qpos[_qadr:_qadr + 3]
                vel_err = s.linear_vel - data.qvel[_vadr:_vadr + 3]
                print(
                    f"t={data.time:.2f}: "
                    f"est pos err={np.round(pos_err, 4)}  "
                    f"vel err={np.round(vel_err, 4)}"
                )

            # Command schedule: rest -> walk -> rest.
            core.vx_cmd = (
                1.0
                if REST_S <= data.time < REST_S + WALK_S
                else 0.0
            )

            # Engage MPC once the robot has settled.
            if not mpc_active:
                if (
                    data.time > SETTLE_TIME_S
                    and np.linalg.norm(s.linear_vel) < SETTLE_VEL
                ):
                    core.z_des = float(s.pos[2])
                    core.yaw_des = s.yaw
                    mpc_active = True
                    core._t_engaged = 0.0

                    print(
                        f"MPC engaged at t={data.time:.2f}s, "
                        f"z_des={core.z_des:.3f} m"
                    )

            # Solve MPC at most once per simulation step.
            if mpc_active:
                res = core.step(dt, data.time)

                if (
                    res is not None
                    and not res.success
                    and data.time - last_print > 1.0
                ):
                    print(
                        f"t={data.time:.2f}: "
                        "MPC solve failed, holding last offsets"
                    )
                    last_print = data.time

            # Gait + MPC offsets -> servo angles.
            controller.update(dt, None)

            mujoco.mj_step(model, data)
            viewer.sync()

            if data.time > END_T:
                break

            sleep = dt - (time.time() - step_start)
            if sleep > 0:
                time.sleep(sleep)

if __name__ == "__main__":
    sim()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--export",
        nargs="+",
        choices=["force", "pos"],
    )

    args = parser.parse_args()

    if args.export:
        if "force" in args.export:
            core.export_force_log()

        if "pos" in args.export:
            controller.export_pos_log([0])