import math
from dataclasses import dataclass
from enum import Enum, auto

from .leg_config import (BODY_COXA_X_MM, FEMUR_LENGTH_MM, TIBIA_LENGTH_MM, Offset, Theta3, Vec3, Gait, GAIT_WALK_F)
from .movement import EPSILON

from .movement import IK
from .limits import ServoConfig

class LegState(Enum):
    NEUTRAL = auto()     # initial pose: foot pinned to neutral
    SWING = auto()
    STANCE = auto()
    PAUSE = auto()    # NEW: foot frozen wherever it is (gait idle)


@dataclass
class FootPositionState:
    curr: Vec3
    target: Vec3

    swing_start: Vec3
    swing_target: Vec3

    stance_start: Vec3
    stance_target: Vec3


def clamp_to_limits(angles_deg: Theta3, leg_idx: int) -> Theta3:
    lim = ServoConfig.limits[leg_idx]
    return Theta3(
        lim.coxa.clamp(angles_deg.coxa),
        lim.femur.clamp(angles_deg.femur),
        lim.tibia.clamp(angles_deg.tibia),
    )
def _is_right_leg(leg_id: int) -> bool: return leg_id == 1 or leg_id == 2


class Leg:
    MPC_OFFSET_TAU_S = 0.04  # low-pass time constant of the MPC foot offset

    def __init__(self, leg_id: int):
        neutral = Offset.neutral_foot_pos_lf[leg_id]

        self.id = leg_id

        self.foot_state = FootPositionState(
            curr=neutral.copy(),
            target=neutral.copy(),
            swing_start=neutral.copy(),
            swing_target=neutral.copy(),
            stance_start=neutral.copy(),
            stance_target=neutral.copy(),
        )

        self.swing_phase = 0.0
        self.stance_phase = 0.0
        self.leg_state = LegState.NEUTRAL
        self.servo_angles_deg = Theta3()

        self.mpc_offset_lf = Vec3(0.0, 0.0, 0.0)
        self._mpc_offset_cmd = Vec3(0.0, 0.0, 0.0)

        self._gait: Gait = GAIT_WALK_F
        self.swing_v0 = Vec3(0.0, 0.0, 0.0)

        self._move_leg()

    # -------------------------- update --------------------------
    def update(self, dt_s: float, gait: Gait) -> Theta3:
        self._gait = gait

        if self.leg_state == LegState.SWING:
            self._update_swing(dt_s)
        elif self.leg_state == LegState.STANCE:
            self._update_stance(dt_s)
        elif self.leg_state == LegState.PAUSE:
            pass                                   # keep target as is
        else:                                      # NEUTRAL
            self.foot_state.target = Offset.neutral_foot_pos_lf[self.id].copy()

        # low-pass the MPC offset; swing legs carry no ground force -> decay to 0
        zero = Vec3(0.0, 0.0, 0.0)
        cmd = zero if self.leg_state == LegState.SWING else self._mpc_offset_cmd
        alpha = 1.0 - math.exp(-dt_s / self.MPC_OFFSET_TAU_S)
        self.mpc_offset_lf = self.mpc_offset_lf + (cmd - self.mpc_offset_lf) * alpha

        # nominal gait position: never modified by PID or MPC
        p_ik = self.foot_state.target.copy()

        # curr = where the foot is commanded (trajectory + PID + MPC offset)
        self.foot_state.curr = p_ik + self.mpc_offset_lf

        self._move_leg()

        return self.servo_angles_deg
    
    # -------------------------- commands --------------------------
    def start_swing(self) -> None:
        self.swing_v0 = self.stance_vel

        self.foot_state.swing_start = self.foot_state.target.copy()
        self.foot_state.swing_target = self.nom_land_pos()
        self.swing_phase = 0.0
        self._set_state(LegState.SWING)

    def start_stance(self, target_pos_lf: Vec3) -> None:
        self.foot_state.stance_start = self.foot_state.target.copy()
        self.foot_state.stance_target = (
            target_pos_lf.copy() if target_pos_lf is not None else self.rear_pos()
        )
        self.stance_phase = 0.0
        self._set_state(LegState.STANCE)

    def hold(self) -> None:
        """Freeze the trajectory at the current foot target (gait idle)."""
        if self.leg_state == LegState.SWING:
            return                                 # never freeze mid-swing
        self._set_state(LegState.PAUSE)

    def resume_stance(self) -> None:
        """Continue the stance sweep from the frozen position toward the rear limit."""
        self.start_stance(self.rear_pos())

    # -------------------------- setters --------------------------
    def set_mpc_offset(self, offset_lf: Vec3) -> None:
        self._mpc_offset_cmd = offset_lf.copy()

    # --------------------------  queries --------------------------
    def grounded(self) -> bool: return self.leg_state != LegState.SWING

    def cur_pos(self) -> Vec3: return self.foot_state.curr

    def swing_time_remaining_s(self) -> float:
        if self.leg_state != LegState.SWING:
            return 0.0
        return (1.0 - self.swing_phase) * self._gait.timing.swing_s

    def rear_pos(self) -> Vec3:
        """Rear endpoint relative to the direction of body travel."""
        p = Offset.neutral_foot_pos_lf[self.id].copy()
        dx, dy = self._travel_dir()
        half_step = self._half_step()

        p.x -= dx * half_step
        p.y -= dy * half_step
        return p

    def nom_land_pos(self) -> Vec3:
        """Front endpoint, opposite the rear endpoint."""
        p = Offset.neutral_foot_pos_lf[self.id].copy()
        dx, dy = self._travel_dir()
        half_step = self._half_step()
        p.x += dx * half_step
        p.y += dy * half_step
        return p

    # -------------------------- internals --------------------------
    def _set_state(self, state: LegState) -> None:
        self.leg_state = state

    def _move_leg(self) -> None:
        angles = None

        target = self.foot_state.curr.copy()
        target.y *= -1.0 if _is_right_leg(self.id) else 1.0  # flip y for right-side legs
        angles = IK.solve(target, BODY_COXA_X_MM, FEMUR_LENGTH_MM, TIBIA_LENGTH_MM)

        if angles is not None:
            if not _is_right_leg(self.id): # flip angles for reverse-mounted servos (left side)
                angles = IK.invert_angles(angles)
        else:
            print("ik failed")

        self.servo_angles_deg = angles

    def _update_stance(self, dt_s: float) -> None:
        foot = self.foot_state.target
        rear = self.rear_pos()

        # sweep at the gait's fixed rate toward the rear endpoint
        rate = self._gait.timing.step_length_mm / max(self._gait.timing.t_stance_s, EPSILON)
        ex, ey = rear.x - foot.x, rear.y - foot.y
        dist = math.hypot(ex, ey)

        if dist > EPSILON:
            travel = min(rate * dt_s, dist)
            foot.x += ex / dist * travel
            foot.y += ey / dist * travel
        else:
            foot.x, foot.y = rear.x, rear.y

        start = self.foot_state.stance_start
        span = math.hypot(rear.x - start.x, rear.y - start.y)
        done = math.hypot(foot.x - start.x, foot.y - start.y)
        self.stance_phase = 1.0 if span <= EPSILON else min(done / span, 1.0)

    def _update_swing(self, dt_s: float) -> None:
        self.swing_phase += dt_s / self._gait.timing.swing_s
        self.swing_phase = min(self.swing_phase, 1.0)

        t = self.swing_phase
        start = self.foot_state.swing_start
        target = self.foot_state.swing_target
        foot = self.foot_state.target

        T = self._gait.timing.swing_s
        cap = 0.6 * math.hypot(target.x - start.x, target.y - start.y)   # mm

        def lim(v):
            m = math.hypot(v.x, v.y) * T
            return v * (cap / m) if m > cap else v

        v0 = lim(self.swing_v0 * 0.7)
        v1 = lim(self.stance_vel)

        h00 = 2*t**3 - 3*t**2 + 1; h10 = t**3 - 2*t**2 + t
        h01 = -2*t**3 + 3*t**2;    h11 = t**3 - t**2
        foot.x = h00*start.x + h10*T*v0.x + h01*target.x + h11*T*v1.x
        foot.y = h00*start.y + h10*T*v0.y + h01*target.y + h11*T*v1.y

        u = t ** 0.7
        foot.z = start.z + self._gait.timing.step_height_mm * math.sin(math.pi * u) ** 2

        if t >= 1.0:
            self.foot_state.target = target.copy()
            self.start_stance(self.rear_pos())
    
    def _travel_dir(self) -> tuple[float, float]:
        """Unit vector of travel, fixed by the gait (not by velocity)."""
        return self._gait.direction, 0.0

    def _half_step(self) -> float:
        return 0.5 * self._gait.timing.step_length_mm

    @property
    def stance_vel(self) -> Vec3:
        """Foot velocity relative to the body during stance (mm/s): step_length / t_stance."""
        dx, dy = self._travel_dir()
        rate = self._gait.timing.step_length_mm / max(self._gait.timing.t_stance_s, EPSILON)
        return Vec3(-dx * rate, -dy * rate, 0.0)