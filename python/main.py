import time
import mujoco
import mujoco.viewer
import math

model = mujoco.MjModel.from_xml_path("assembly/xml/reginleif.xml")
data = mujoco.MjData(model)

# Initialize simulation state
mujoco.mj_resetData(model, data)

# Set all joint angles to 0 radians
for i in range(model.njnt):
    joint_type = model.jnt_type[i]
    qpos_addr = model.jnt_qposadr[i]

    if joint_type == mujoco.mjtJoint.mjJNT_HINGE:
        data.qpos[qpos_addr] = 90.0 * (math.pi / 180.0)

# Update derived quantities
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # data.ctrl[:] = 0.5  # Your controller goes here
        # mujoco.mj_step(model, data)

        viewer.sync()

        sleep = model.opt.timestep - (time.time() - step_start)
        if sleep > 0:
            time.sleep(sleep)