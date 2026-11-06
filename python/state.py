import numpy as np
import mujoco
from dataclasses import dataclass, field
from mpc import State

def read_state(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> State:
    s = State()

    # Free joint of this body
    jnt = model.body_jntadr[body_id]
    qadr = model.jnt_qposadr[jnt]
    vadr = model.jnt_dofadr[jnt]

    # Position (world frame): qpos[0:3]
    s.pos = data.qpos[qadr:qadr + 3].copy()

    # Orientation: qpos[3:7] is quaternion in (w, x, y, z) order
    w, x, y, z = data.qpos[qadr + 3:qadr + 7]

    # ZYX (yaw-pitch-roll) Euler angles
    s.roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    s.pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    s.yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    # Linear velocity (world frame): qvel[0:3]
    s.linear_vel = data.qvel[vadr:vadr + 3].copy()

    # Angular velocity: qvel[3:6], expressed in the BODY frame
    s.angular_vel = data.qvel[vadr + 3:vadr + 6].copy()

    # Gravity from the model rather than hardcoding
    s.g = float(model.opt.gravity[2])

    return s