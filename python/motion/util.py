import math
import numpy as np
from dataclasses import dataclass, field
from typing import List
import mujoco

from .mpc import State
from leg.leg_config import Vec3, Offset

def foot_forces_to_positions(
    foot_forces: List[Vec3],
    gain_mm_per_n: float = 0.5,
    max_correction_mm: float = 30.0,
) -> List[Vec3]:
    """
    Map world-frame ground reaction forces to heuristic foot corrections.

    Args:
        foot_forces: Ground reaction forces in Newtons, world frame.
        gain_mm_per_n: Correction gain in mm/N.
        max_correction_mm: Maximum correction magnitude per foot, in mm.

    Returns:
        World-frame displacement vectors in millimeters.
    """
    corrections: List[Vec3] = []

    for force in foot_forces:
        # Position-controlled leg = stiff spring: to PUSH on the body with force f
        # the foot target must go the opposite way (deeper into the ground / back
        # for forward thrust). Hence the minus sign.
        dx = -force.x * gain_mm_per_n
        dy = -force.y * gain_mm_per_n
        dz = -force.z * gain_mm_per_n

        magnitude = (dx * dx + dy * dy + dz * dz) ** 0.5

        if magnitude > max_correction_mm:
            scale = max_correction_mm / magnitude
            dx *= scale
            dy *= scale
            dz *= scale

        corrections.append(Vec3(dx, dy, dz))

    return corrections

def read_state(model: mujoco.MjModel, data: mujoco.MjData, body_id: int, debug: bool = False) -> State:
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

    # Angular velocity: MuJoCo free-joint qvel[3:6] is in the BODY frame, but the
    # MPC model (theta_dot = Rz^T omega) expects the WORLD frame -> rotate it.
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.ascontiguousarray(data.qpos[qadr + 3:qadr + 7]))
    s.angular_vel = R.reshape(3, 3) @ data.qvel[vadr + 3:vadr + 6]

    # Gravity from the model rather than hardcoding
    s.g = float(model.opt.gravity[2])

    if debug:
        print("\n========== MuJoCo State ==========")
        print(f"Position (world)    : {np.array2string(s.pos, precision=3)}")
        print(f"Orientation (deg)   : roll={np.degrees(s.roll):7.2f}, "
            f"pitch={np.degrees(s.pitch):7.2f}, yaw={np.degrees(s.yaw):7.2f}")
        print(f"Linear velocity    : {np.array2string(s.linear_vel, precision=3)}")
        print(f"Angular velocity   : {np.array2string(s.angular_vel, precision=3)}")
        print(f"Gravity Z          : {s.g:.3f} m/s^2")
        print("==================================\n")

    return s

def rotation_matrix_yaw(yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)

    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])

def legToBodyFrame(leg_idx: int, foot_pos_lf: Vec3) -> Vec3:
    return Offset.leg_base_pos_bf[leg_idx] + foot_pos_lf

def bodyToLegFrame(leg_idx: int, foot_pos_bf: Vec3) -> Vec3:
    return foot_pos_bf - Offset.leg_base_pos_bf[leg_idx]

def legToWorldFrame(leg_idx: int, foot_pos_lf: Vec3, body_pos_wf: Vec3, yaw: float) -> Vec3:
    foot_pos_bf = legToBodyFrame(leg_idx, foot_pos_lf)
    R_wb = rotation_matrix_yaw(yaw)

    p_b = np.array([foot_pos_bf.x, foot_pos_bf.y, foot_pos_bf.z])
    p_w = np.array([body_pos_wf.x, body_pos_wf.y, body_pos_wf.z]) + R_wb @ p_b

    return Vec3(float(p_w[0]), float(p_w[1]), float(p_w[2]))

def world_vector_to_leg_frame(
    vector_wf: Vec3,
    yaw: float,
) -> Vec3:
    """Rotate a vector from world axes into body-aligned leg axes.

    This transforms a displacement vector, so no translations are applied.
    Assumes leg axes are aligned with the body axes and ignores roll/pitch.
    """
    R_wb = rotation_matrix_yaw(yaw)

    v_w = np.array([vector_wf.x, vector_wf.y, vector_wf.z])
    v_b = R_wb.T @ v_w

    return Vec3(float(v_b[0]), float(v_b[1]), float(v_b[2]))