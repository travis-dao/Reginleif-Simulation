from dataclasses import dataclass, field
from typing import Sequence, Tuple, Optional


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
    STEP_HEIGHT_MM = 40.0
    SWING_DURATION_S = 0.3
    GAP_DURATION_S = SWING_DURATION_S / 4 # 1/4 of swing duration seems to work well

    # indexed by (current swing leg + 1); index 0 is "no leg yet" (-1)
    ORDER_WALK_F: Tuple[int, ...] = (0, 2, 1, 3)
    ORDER_WALK_B: Tuple[int, ...] = (2, 0, 3, 1)
    ORDER_TURN_R: Tuple[int, ...] = (1, 2, 3, 0)
    ORDER_TURN_L: Tuple[int, ...] = (0, 3, 2, 1)

    @classmethod
    def order_name(cls, order: Tuple[int, ...]) -> str:
        for name, value in vars(cls).items():
            if name.startswith("ORDER_") and value == order:
                return name
        return str(order)
    
    @classmethod
    def index_name(cls, index: int) -> str:
        return {
            -1: "None",
            0: "FL",
            1: "FR",
            2: "BR",
            3: "BL",
        }.get(index, "Unknown")

@dataclass(frozen=True)
class GaitTiming:
    swing_s: float
    gap_s: float
    duty_zheta: float = 4.0
    step_length_mm: float = 100.0
    step_height_mm: float = 40.0
    duty_override: Optional[float] = None

    @property
    def duty(self) -> float:
        if self.duty_override is not None:
            return self.duty_override
        return 1.0 - self.swing_s / (self.duty_zheta * (self.swing_s + self.gap_s))

    @property
    def period_s(self) -> float:
        return self.swing_s / (1.0 - self.duty)

    @property
    def t_stance_s(self) -> float:
        return self.duty * self.period_s

    @property
    def v_max_mm_s(self) -> float:
        return self.step_length_mm / self.t_stance_s

@dataclass(frozen=True)
class Gait:
    name: str
    order: Tuple[int, ...]
    timing: GaitTiming
    # phase offset (fraction of period) of the k-th leg in `order`
    phase_offsets: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75)


_T, _G = Movement.SWING_DURATION_S, Movement.GAP_DURATION_S

def run_gait(name, swing_s, duty, step_len=100.0, step_h=40.0):
    duty = max(duty, 0.3)                    # keep >= 1 foot on the ground
    t = GaitTiming(swing_s=swing_s, gap_s=0.0,
                   step_length_mm=step_len, step_height_mm=step_h,
                   duty_override=duty)
    # period = swing_s / (1 - duty); phase spacing stays 0.25 -> same sequential order
    return Gait(name, Movement.ORDER_WALK_F, t, phase_offsets=(0.0, 0.25, 0.5, 0.75))

GAIT_RUN_F = run_gait("RUN_F", swing_s=0.3, duty=0.4, step_len=100, step_h=50)

GAIT_WALK_F = Gait("WALK_F", Movement.ORDER_WALK_F, GaitTiming(_T,       _G,     duty_zheta=3.8))
GAIT_WALK_B = Gait("WALK_B", Movement.ORDER_WALK_B, GaitTiming(_T * 1.3,       _G * 1.3,     duty_zheta=4.0))
GAIT_TURN_R = Gait("TURN_R", Movement.ORDER_TURN_R, GaitTiming(_T * 1.5, _G * 4, duty_zheta=4.0))
GAIT_TURN_L = Gait("TURN_L", Movement.ORDER_TURN_L, GaitTiming(_T * 1.5, _G * 4, duty_zheta=4.0))

_FOOT_NEUTRAL_MAG = Vec3(90.0, 90.0, -30.0)
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