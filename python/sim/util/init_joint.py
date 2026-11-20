import mujoco

# Desired initial joint angles (radians).
INITIAL_JOINT_ANGLES = {
    "leg2_coxa":  2.3562,
    "leg2_femur": 2.7066,
    "leg2_tibia": 0.9200,

    "leg1_coxa":  0.7854,
    "leg1_femur": 2.7066,
    "leg1_tibia": 0.9200,

    "leg3_coxa":  0.7854,
    "leg3_femur": 0.4349,
    "leg3_tibia": 2.2216,

    "leg0_coxa":  2.3562,
    "leg0_femur": 0.4349,
    "leg0_tibia": 2.2216,
}

def init_joint(model, data):
    # Apply the desired angles to the initial MuJoCo state.
    for joint_name, angle in INITIAL_JOINT_ANGLES.items():
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )

        if joint_id == -1:
            raise ValueError(f"Joint not found: {joint_name}")

        # Set the actual joint position.
        qpos_addr = model.jnt_qposadr[joint_id]
        data.qpos[qpos_addr] = angle

        # Set the position actuator's target to the same angle.
        actuator_name = f"{joint_name}_act"
        actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
        )

        if actuator_id == -1:
            raise ValueError(f"Actuator not found: {actuator_name}")

        data.ctrl[actuator_id] = angle