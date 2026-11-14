"""Python port of leg_controller.h / leg_controller.cpp."""
from __future__ import annotations

from typing import List, Optional
import math

from .leg import Leg
from .leg_config import ImuCorrection, Movement, Offset, Theta3, Vec3
from .movement import LegDriver
from enum import Enum, auto

NUM_SERVOS = 12


class GaitState(Enum):
    IDLE = auto()       # all feet planted, stance sweep frozen, no swings scheduled
    WALKING = auto()    # normal crawl: swing, gap, swing, ...
    STOPPING = auto()   # stop requested: finish any swing in flight, then IDLE at the next gap


def leg_to_body_frame(leg: int, foot_pos_lf: Vec3) -> Vec3:
    """Leg-frame foot position -> body-frame position."""
    return Offset.leg_base_pos_bf[leg] + foot_pos_lf


def body_to_leg_frame(leg: int, foot_pos_bf: Vec3) -> Vec3:
    return foot_pos_bf - Offset.leg_base_pos_bf[leg]


class LegController:
    def __init__(self, model, data):
        self.legs: List[Optional[Leg]] = [None] * 4
        self.driver: LegDriver = LegDriver(model, data, debug=False)

        self.target_angles: List[Theta3] = [Theta3() for _ in range(4)]
        self.target_angles_buffer: List[float] = [0.0] * NUM_SERVOS

        self.grounded = True
        self.swing_leg = -1
        self._last_swing_leg = -1
        self.gap_remaining_s = 0.0

        self.state = GaitState.IDLE

        self._nom_bf = [leg_to_body_frame(i, Offset.neutral_foot_pos_lf[i]) for i in range(4)]
        self.t_stance_s = 3.0 * Movement.SWING_DURATION_S + 4.0 * Movement.GAP_DURATION_S
        self.v_max_mm_s = Movement.STEP_LENGTH_MM / self.t_stance_s
        self.max_step_y_mm = 100.0            # tune: sideways reach of the coxa
        self.a_v_mm_s2 = 150.0               # twist slew limits, tune
        self.a_w_rad_s2 = 0.5
        self._cmd = (0.0, 0.0)               # requested (v mm/s, w rad/s)
        self._v = 0.0                        # achieved sweep twist
        self._w = 0.0
        self._geom = (0.0, 0.0)              # twist that defines step geometry

        self.turn_v_eps = 5.0 # mm/s, below this, cound yaw cmd as "turn in place"
        self.turn_w_eps = 0.02 #rad/s
        self._order = Movement.ORDER_WALK

    def init(self) -> None:
        for i in range(4):
            self.legs[i] = Leg(i)

    # commands
    def set_body_command(self, v_mm_s: float, w_rad_s: float) -> None:
        self._cmd = (v_mm_s, w_rad_s)

    def body_twist(self):
        """Achieved (v mm/s, w rad/s) the stance feet are producing now."""
        return self._v, self._w

    def request_start(self) -> None:
        if self.state in (GaitState.IDLE, GaitState.STOPPING):
            self._set_state(GaitState.WALKING)

    def request_stop(self) -> None:
        if self.state is GaitState.WALKING:
            self._set_state(GaitState.STOPPING)

    def is_idle(self) -> bool:
        return self.state is GaitState.IDLE

    def next_swing_leg(self) -> int:
        return self._after(self._last_swing_leg)

    def _select_order(self):
        v, w = self._cmd                      # use the *command*, not the ramping achieved twist
        if abs(v) < self.turn_v_eps and abs(w) > self.turn_w_eps:
            return Movement.ORDER_TURN_L if w > 0 else Movement.ORDER_TURN_R
        return Movement.ORDER_WALK

    def _after(self, last: int) -> int:
        o = self._order
        return o[0] if last < 0 else o[(o.index(last) + 1) % 4]

    # transitions
    def _set_state(self, new: GaitState) -> None:
        old, self.state = self.state, new

        if old is GaitState.IDLE and new is GaitState.WALKING:
            for leg in self.legs:
                leg.resume_stance()
            self.gap_remaining_s = 0.0               # first swing starts immediately

        elif new is GaitState.IDLE:
            self.swing_leg = -1
            for leg in self.legs:
                leg.hold()                           # NEW in Leg: freeze at current pos

    def is_grounded(self) -> bool:
        return self.grounded

    def get_swing_leg(self) -> int:
        return self.swing_leg

    def swing_time_remaining_s(self) -> float:
        if self.swing_leg < 0:
            return 0.0
        return self.legs[self.swing_leg].swing_time_remaining_s()

    def update(self, dt_s: float, imu_data: Optional[ImuCorrection] = None) -> None:
        self._update_twist(dt_s)

        grounded = all(leg.is_grounded() for leg in self.legs)
        if grounded and not self.grounded:
            self.swing_leg = -1
            self.gap_remaining_s = Movement.GAP_DURATION_S
            self._order = self._select_order()
        self.grounded = grounded

        if self.grounded and self.state is not GaitState.IDLE:
            self.gap_remaining_s -= dt_s
            if self.gap_remaining_s <= 0.0:
                if self.state is GaitState.WALKING:
                    next = self.next_swing_leg()
                    self._last_swing_leg = next
                    self.swing_leg = next
                    self.legs[next].start_swing(self.legs[next].get_nominal_landing_pos())
                else:                                # STOPPING, gap closed
                    self._set_state(GaitState.IDLE)

        self.update_driver(dt_s, imu_data)

    def update_driver(self, dt_s: float, imu_data: ImuCorrection):
        for i in range(4):
            self.target_angles[i] = self.legs[i].update(dt_s, imu_data)
            self.driver.set_angles(i, self.target_angles[i])
    
    def swing_schedule(self, horizon_s: float):
        if self.state is GaitState.IDLE:
            return []
        T, G = Movement.SWING_DURATION_S, Movement.GAP_DURATION_S
        out = []
        if self.swing_leg >= 0:
            t_end = self.swing_time_remaining_s()
            out.append((self.swing_leg, 0.0, t_end))
            t_next = t_end + G
        else:
            t_next = max(self.gap_remaining_s, 0.0)
        if self.state is GaitState.STOPPING:
            return out                               # only the swing already in flight
        last = self._last_swing_leg
        while t_next < horizon_s:
            last = self._after(last)
            out.append((last, t_next, t_next + T))
            t_next += T + G
        return out

    def apply_mpc_corrections(self, offsets_lf: List[Vec3]) -> None:
        """Hand each leg its MPC foot OFFSET (leg frame, mm).

        Offsets are layered on top of the gait trajectory inside Leg.update()
        (low-pass filtered, zeroed for the swing leg), so they are not
        overwritten by the next trajectory tick and do not accumulate.
        """
        for leg, off in zip(self.legs, offsets_lf):
            # if leg.id == 0:
                # print(f"swing={self.swing_leg}, offset={off}")
            leg.set_mpc_offset(off)

    def foot_positions_body(self) -> List[Vec3]:
        """Current foot positions (trajectory + MPC offset), body frame, mm."""
        return [leg_to_body_frame(i, self.legs[i].get_current_leg_pos()) for i in range(4)]

    def landing_targets_body(self) -> List[Vec3]:
        """Nominal touchdown position of every foot, body frame, mm.

        The MPC horizon can span several swings (the gait is one leg at a time),
        so any leg may land inside it -- not only the currently airborne one."""
        return [leg_to_body_frame(i, self.legs[i].get_nominal_landing_pos()) for i in range(4)]

    def export_angle_log(self, ids: List[int]=[0, 1, 2, 3]) -> None:
        for i in ids:
            self.legs[i].export_angle_log()

    def export_pos_log(self, ids: List[int]=[0, 1, 2, 3]) -> None:
        for i in ids:
            self.legs[i].export_pos_log()

    def _limit(self, v, w):
        """Scale (v, w) together, which keeps R = v/w, so every step stays in the workspace."""
        T = self.t_stance_s
        k = 1.0
        for p in self._nom_bf:
            sx = abs(v - w * p.y) * T
            sy = abs(w * p.x) * T
            k = min(k,
                    Movement.STEP_LENGTH_MM / sx if sx > 1e-9 else 1.0,
                    self.max_step_y_mm / sy if sy > 1e-9 else 1.0)
        return v * k, w * k

    def _update_twist(self, dt_s: float) -> None:
        if self.state is GaitState.IDLE:
            self._v = self._w = 0.0                    # nothing sweeps
            self._geom = self._limit(*self._cmd)       # so the first swing lands correctly
        elif self.state is GaitState.WALKING:
            vc, wc = self._limit(*self._cmd)
            self._geom = (vc, wc)
            self._v += max(-self.a_v_mm_s2 * dt_s, min(self.a_v_mm_s2 * dt_s, vc - self._v))
            self._w += max(-self.a_w_rad_s2 * dt_s, min(self.a_w_rad_s2 * dt_s, wc - self._w))
        # STOPPING: hold the last twist until the gait goes IDLE

        vg, wg = self._geom
        T = self.t_stance_s
        for leg, p in zip(self.legs, self._nom_bf):
            vel = Vec3(-(self._v - self._w * p.y), -(self._w * p.x), 0.0)
            step = Vec3(-(vg - wg * p.y) * T, -(wg * p.x) * T, 0.0)
            leg.set_stance(vel, step)