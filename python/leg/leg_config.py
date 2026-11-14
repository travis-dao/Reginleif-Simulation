"""Python port of leg_config.h and the parts of types.h the leg code relies on.

types.h was not provided, so Vec3 / Theta3 / ImuData are reasonable
reconstructions based on how they are used. Adjust if yours differ.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple


'''
foot_neutral_offset_mag = { 120.0f, 120.0f, -40.0f }
Leg 0 | coxa=135.00, femur=37.03, tibia=112.71
Leg 1 | coxa=45.00, femur=142.97, tibia=67.29
Leg 2 | coxa=135.00, femur=142.97, tibia=67.29
Leg 3 | coxa=45.00, femur=37.03, tibia=112.71
'''

# --------------------------------------------------------------------------
# types (reconstructed from usage)
# --------------------------------------------------------------------------
@dataclass
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def copy(self) -> "Vec3":
        return Vec3(self.x, self.y, self.z)

    def __add__(self, o: "Vec3") -> "Vec3":
        return Vec3(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o: "Vec3") -> "Vec3":
        return Vec3(self.x - o.x, self.y - o.y, self.z - o.z)

    def __mul__(self, o):
        """Scalar multiply, or element-wise multiply when o is a Vec3."""
        if isinstance(o, Vec3):
            return Vec3(self.x * o.x, self.y * o.y, self.z * o.z)
        return Vec3(self.x * o, self.y * o, self.z * o)

    __rmul__ = __mul__

    def rotate(self, m: Sequence[Sequence[float]]) -> None:
        """In-place rotation by a 3x3 matrix (row-major), like Vec3::rotate."""
        x, y, z = self.x, self.y, self.z
        self.x = m[0][0] * x + m[0][1] * y + m[0][2] * z
        self.y = m[1][0] * x + m[1][1] * y + m[1][2] * z
        self.z = m[2][0] * x + m[2][1] * y + m[2][2] * z


@dataclass
class Theta3:
    coxa: float = 0.0
    femur: float = 0.0
    tibia: float = 0.0


@dataclass
class ImuCorrection:
    valid: bool = False
    rot_mat: List[List[float]] = field(
        default_factory=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )


# --------------------------------------------------------------------------
# leg geometry
# --------------------------------------------------------------------------
FEMUR_LENGTH_MM = 123.0
TIBIA_LENGTH_MM = 160.0
BODY_COXA_X_MM = 15.0

c_max = FEMUR_LENGTH_MM + TIBIA_LENGTH_MM
c_min = abs(FEMUR_LENGTH_MM - TIBIA_LENGTH_MM)


class Movement:
    STEP_LENGTH_MM = 100.0
    STEP_HEIGHT_MM = 60.0
    SWING_DURATION_S = 0.4
    GAP_DURATION_S = 0.1 # 1/4 of swing duration seems to work well

    # indexed by (current swing leg + 1); index 0 is "no leg yet" (-1)
    ORDER_WALK: Tuple[int, ...] = (0, 2, 1, 3)
    ORDER_TURN_R: Tuple[int, ...] = (0, 1, 2, 3)
    ORDER_TURN_L: Tuple[int, ...] = (0, 3, 2, 1)


_FOOT_NEUTRAL_MAG = Vec3(120.0, 100.0, -40.0)
_LEG_BASE_MAG = Vec3(34.0, 43.5, 0.0)
_QUADRANT_SIGN = (
    Vec3(+1, +1, +1),  # Front Left
    Vec3(+1, -1, +1),  # Front Right
    Vec3(-1, -1, +1),  # Back Right
    Vec3(-1, +1, +1),  # Back Left
)


class Offset:
    # foot frame, (0,0,0) = hip of leg
    foot_neutral_offset_mag = _FOOT_NEUTRAL_MAG
    # body frame, (0,0,0) = center of body
    leg_base_offset_mag = _LEG_BASE_MAG

    # Per-leg sign for each axis (FL, FR, BR, BL)
    quadrant_sign: Tuple[Vec3, ...] = _QUADRANT_SIGN

    # Leg base position, body frame
    leg_base_pos_bf: Tuple[Vec3, ...] = tuple(s * _LEG_BASE_MAG for s in _QUADRANT_SIGN)

    # Neutral foot position, leg frame
    neutral_foot_pos_lf: Tuple[Vec3, ...] = tuple(s * _FOOT_NEUTRAL_MAG for s in _QUADRANT_SIGN)

    # Neutral foot position, body frame
    neutral_foot_pos_bf: Tuple[Vec3, ...] = tuple(
        b + n for b, n in zip(leg_base_pos_bf, neutral_foot_pos_lf)
    )