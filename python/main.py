# mjpython python/main.py

import time
import mujoco
import mujoco.viewer
import numpy as np

'''
foot_neutral_offset_mag = { 120.0f, 120.0f, -40.0f }
Leg 0 | coxa=135.00, femur=37.03, tibia=112.71
Leg 1 | coxa=45.00, femur=142.97, tibia=67.29
Leg 2 | coxa=135.00, femur=142.97, tibia=67.29
Leg 3 | coxa=45.00, femur=37.03, tibia=112.71
'''

joint_angles = {
    "leg0_coxa":  0.0,
    "leg0_femur": 0.0,
    "leg0_tibia": 0.0,

    "leg1_coxa":  0.0,
    "leg1_femur": 0.0,
    "leg1_tibia": 0.0,

    "leg2_coxa":  0.0,
    "leg2_femur": 0.0,
    "leg2_tibia": 0.0,

    "leg3_coxa":  0.0,
    "leg3_femur": 0.0,
    "leg3_tibia": 0.0,
}

model = mujoco.MjModel.from_xml_path("assembly/xml/reginleif.xml")
data = mujoco.MjData(model)

# Update derived quantities
mujoco.mj_forward(model, data)

def init_joint_angles():
    joint_angles["leg0_coxa"] = np.deg2rad(135)
    joint_angles["leg1_coxa"] = np.deg2rad(45) #
    joint_angles["leg2_coxa"] = np.deg2rad(135) #
    joint_angles["leg3_coxa"] = np.deg2rad(45)

    joint_angles["leg0_femur"] = np.deg2rad(37)
    joint_angles["leg1_femur"] = np.deg2rad(143) #
    joint_angles["leg2_femur"] = np.deg2rad(143) #
    joint_angles["leg3_femur"] = np.deg2rad(37)

    joint_angles["leg0_tibia"] = np.deg2rad(113)
    joint_angles["leg1_tibia"] = np.deg2rad(67) #
    joint_angles["leg2_tibia"] = np.deg2rad(67) #
    joint_angles["leg3_tibia"] = np.deg2rad(113)

def set_joint_angles(model, data, angles, forward=True):
    """Set joint positions by name from a dict {joint_name: angle_rad}."""
    for name, angle in angles.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid == -1:
            raise KeyError(f"Joint '{name}' not found in model")
        qpos_adr = model.jnt_qposadr[jid]
        data.qpos[qpos_adr] = angle
    if forward:
        mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        init_joint_angles()
        set_joint_angles(model, data, joint_angles)

        step_start = time.time()

        # data.ctrl[:] = 0.5  # Your controller goes here
        # mujoco.mj_step(model, data)

        viewer.sync()

        sleep = model.opt.timestep - (time.time() - step_start)
        if sleep > 0:
            time.sleep(sleep)