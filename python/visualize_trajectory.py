
# mjpython python/visualize_trajectory.py

import time
import mujoco
import mujoco.viewer

from leg.leg_controller import LegController
from build_sim_xml import build_sim_xml

SRC = "assembly/xml/reginleif.xml"
DST = "assembly/xml/reginleif_sim.xml"

KP = 30.0
FORCE = 9.4
START_HEIGHT = 0.2

build_sim_xml(SRC, DST, START_HEIGHT, KP, FORCE)

model = mujoco.MjModel.from_xml_path(DST)
data = mujoco.MjData(model)

# Disable gravity.
model.opt.gravity[:] = 0.0

# Initialize simulation.
mujoco.mj_forward(model, data)

# Find the robot base.
body_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "base"
)

# Lock the base in place while allowing the legs to move.
# This assumes the base is a free body with 6 DOF.
jnt_id = model.body_jntadr[body_id]
if jnt_id >= 0 and model.jnt_type[jnt_id] == mujoco.mjtJoint.mjJNT_FREE:
    qpos_adr = model.jnt_qposadr[jnt_id]
    dof_adr = model.jnt_dofadr[jnt_id]

    # Preserve the initial base position and orientation.
    base_qpos = data.qpos[qpos_adr:qpos_adr + 7].copy()

    # Disable the base's translational and rotational motion.
    model.body_gravcomp[body_id] = 1.0

    # Lock the base using MuJoCo equality constraints is preferable
    # for a physically simulated robot. For trajectory visualization,
    # reset its pose and velocity every frame instead.

controller = LegController(model, data)
controller.init()

# Run the gait, but never run the MPC.
dt = model.opt.timestep

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # Update the gait and command the leg joints.
        controller.update(dt, None)

        # Advance physics.
        mujoco.mj_step(model, data)

        # Hold the base pose fixed.
        data.qpos[qpos_adr:qpos_adr + 7] = base_qpos
        data.qvel[dof_adr:dof_adr + 6] = 0.0
        mujoco.mj_forward(model, data)

        viewer.sync()

        sleep = dt - (time.time() - step_start)
        if sleep > 0:
            time.sleep(sleep)
