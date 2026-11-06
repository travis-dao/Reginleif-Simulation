# movement.py
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import mujoco

EPSILON = 1e-6
RAD_TO_DEG = 180.0 / math.pi
DEG_TO_RAD = math.pi / 180.0

'''
foot_neutral_offset_mag = { 120.0f, 120.0f, -40.0f }
Leg 0 | coxa=135.00, femur=37.03, tibia=112.71
Leg 1 | coxa=45.00, femur=142.97, tibia=67.29
Leg 2 | coxa=135.00, femur=142.97, tibia=67.29
Leg 3 | coxa=45.00, femur=37.03, tibia=112.71
'''

@dataclass
class Vec3:
    x: float
    y: float
    z: float

    def __mul__(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.x * other.x,
            self.y * other.y,
            self.z * other.z,
        )

quadrant_sign: list[Vec3] = [
    Vec3(+1, +1, +1),  # Front Left
    Vec3(+1, -1, +1),  # Front Right
    Vec3(-1, -1, +1),  # Back Right
    Vec3(-1, +1, +1),  # Back Left
]

neutral = Vec3(120.0, 120.0, -40.0)   # mm

neutral_foot_pos: list[Vec3] = [
    neutral * sign for sign in quadrant_sign
]


@dataclass
class Theta3:
    coxa: float
    femur: float
    tibia: float


@dataclass
class IKConfig:
    body_coxa_x_mm: float
    femur_length_mm: float
    tibia_length_mm: float
    c_max: float  # max reach (femur servo to foot tip)
    c_min: float  # min reach


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))

def is_right_leg(idx: int) -> bool:
    return (idx == 1) or (idx == 2)


def ik(cfg: IKConfig, target: Vec3) -> Optional[Theta3]:
    """Leg-frame foot position (mm) -> joint angles in degrees, or None if unreachable."""
    x, y, z = target.x, target.y, target.z

    d = math.hypot(x, y)
    r = d - cfg.body_coxa_x_mm
    femur_to_tip = math.hypot(z, r)

    if (femur_to_tip > cfg.c_max + EPSILON
            or femur_to_tip < cfg.c_min - EPSILON
            or femur_to_tip < EPSILON):
        return None

    a, b, c = cfg.femur_length_mm, cfg.tibia_length_mm, femur_to_tip
    cos2 = clamp((a * a + c * c - b * b) / (2 * a * c), -1.0, 1.0)
    cos3 = clamp((a * a + b * b - c * c) / (2 * a * b), -1.0, 1.0)

    return Theta3(
        coxa=math.atan2(y, x) * RAD_TO_DEG,
        femur=math.atan2(r, -z) * RAD_TO_DEG + math.acos(cos2) * RAD_TO_DEG,
        tibia=math.acos(cos3) * RAD_TO_DEG,
    )


def invert_angles(angles_deg: Theta3) -> Theta3:
    return Theta3(
        coxa=180.0 - angles_deg.coxa,
        femur=180.0 - angles_deg.femur,
        tibia=180.0 - angles_deg.tibia,
    )


class LegController:
    LEG_NAMES = ("leg0", "leg1", "leg2", "leg3")

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, cfg: IKConfig, debug=False):
        self.model = model
        self.data = data
        self.cfg = cfg
        self.debug = debug
        self.act_id = {}
        for leg in self.LEG_NAMES:
            for j in ("coxa", "femur", "tibia"):
                name = f"{leg}_{j}_act"
                aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                if aid == -1:
                    raise KeyError(f"Actuator '{name}' not found in model")
                self.act_id[f"{leg}_{j}"] = aid

    def set_foot(self, leg_idx: int, foot_xyz: Union[Vec3, Sequence[float]]) -> bool:
        """
        foot_xyz: leg-frame position in mm (hip joint = origin), as Vec3 or (x, y, z).
        Returns True if reachable. If not, the previous target is held.
        """
        if not isinstance(foot_xyz, Vec3):
            foot_xyz = Vec3(*foot_xyz)

        if (is_right_leg(leg_idx)):
            foot_xyz.y *= -1.0

        angles = ik(self.cfg, foot_xyz)
        if angles is None:
            return False

        if (not is_right_leg(leg_idx)):
            angles = invert_angles(angles)

        if self.debug:
            print(f"leg{leg_idx} target=({foot_xyz.x:.1f}, {foot_xyz.y:.1f}, {foot_xyz.z:.1f}) | "
                  f"coxa={angles.coxa:7.2f} femur={angles.femur:7.2f} tibia={angles.tibia:7.2f}")

        leg = self.LEG_NAMES[leg_idx]
        self.data.ctrl[self.act_id[f"{leg}_coxa"]] = angles.coxa * DEG_TO_RAD
        self.data.ctrl[self.act_id[f"{leg}_femur"]] = angles.femur * DEG_TO_RAD
        self.data.ctrl[self.act_id[f"{leg}_tibia"]] = angles.tibia * DEG_TO_RAD
        return True