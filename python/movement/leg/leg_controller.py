"""Python port of leg_controller.h / leg_controller.cpp."""
from __future__ import annotations

import sys
from pathlib import Path
import argparse

# Add the python/ directory to the module search path.
PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))


from typing import List, Optional
import math

from .leg import Leg
from .leg_config import ImuCorrection, Movement, Offset, Theta3, Vec3, rot_rp, transpose3
from .movement import LegDriver
from enum import Enum, auto
from motion.mpc import State

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

        self.grounded = False
        self.swing_leg = -1
        self._last_swing_leg = -1

        self.state = GaitState.IDLE

        T, G = Movement.SWING_DURATION_S, Movement.GAP_DURATION_S
        self.duty_zheta = 2.7 # 5.0
        self.duty = 1.0 - T / (self.duty_zheta * (T + G))      # >=0.75 = no overlap, <0.75 = overlap
        self.period_s = T / (1.0 - self.duty)
        self.t_stance_s = self.duty * self.period_s
        self.v_max_mm_s = Movement.STEP_LENGTH_MM / self.t_stance_s

        self._psi = [0.0] * 4          # per-leg clock; the leg swings when it wraps past 1
        self._grounded_s = 0.0         # time since the last swing ended (replaces gap_remaining_s)
        self.start_push_s = 0.0

        self._nom_bf = [leg_to_body_frame(i, Offset.neutral_foot_pos_lf[i]) for i in range(4)]
        self.v_max_mm_s = Movement.STEP_LENGTH_MM / self.t_stance_s
        self.max_step_y_mm = 80.0            # tune: sideways reach of the coxa
        self.a_v_mm_s2 = 150.0               # twist slew limits, tune
        self.a_w_rad_s2 = 0.5
        self._cmd = (0.0, 0.0)               # requested (v mm/s, w rad/s)
        self._v_cmd_xy: Vec3 = Vec3(0.0, 0.0, 0.0)
        self._v = 0.0                        # achieved sweep twist
        self._w = 0.0
        self._geom = (0.0, 0.0)              # twist that defines step geometry

        self.turn_v_eps = 5.0 # mm/s, below this, cound yaw cmd as "turn in place"
        self.turn_w_eps = 0.02 #rad/s

        self._order = Movement.ORDER_WALK_F

        self._state = State()
        self.body_roll = 0.0
        self.body_pitch = 0.0

    def init(self) -> None:
        for i in range(4):
            self.legs[i] = Leg(i)

    # commands
    def set_body_command(self, v_mm_s: float, w_rad_s: float) -> None:
        self._cmd = (v_mm_s, w_rad_s)

    def set_body_tilt(self, roll_rad: float, pitch_rad: float) -> None:
        """Commanded body attitude. Applied to every leg in every state."""
        self.body_roll, self.body_pitch = roll_rad, pitch_rad
        R_T = transpose3(rot_rp(roll_rad, pitch_rad))     # build once, share with all legs
        for leg in self.legs:
            leg.set_body_tilt(R_T)

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
    
    def _most_overdue_leg(self) -> int:
        best, best_d = -1, -1.0
        for i, leg in enumerate(self.legs):
            cur = leg.get_current_leg_pos()
            d = math.hypot(cur.x, cur.y)
            if d > best_d:
                best, best_d = i, d
        return best

    def _select_order(self):
        v, w = self._cmd                      # use the *command*, not the ramping achieved twist
        if abs(v) < self.turn_v_eps and abs(w) > self.turn_w_eps:
            return Movement.ORDER_TURN_L if w > 0 else Movement.ORDER_TURN_R
        return Movement.ORDER_WALK_B if v < 0 else Movement.ORDER_WALK_F

    def _after(self, last: int) -> int:
        o = self._order

        if last < 0:
            return o[0]

        return o[(o.index(last) + 1) % len(o)]

    # transitions
    def _set_state(self, new: GaitState) -> None:
        old, self.state = self.state, new
        G = Movement.GAP_DURATION_S

        if new is GaitState.WALKING and old is GaitState.IDLE:
            for leg in self.legs:
                leg.resume_stance()
            order = self._select_order()
            changed = order is not self._order
            self._order = order
            first = (self._most_overdue_leg() if changed and self._last_swing_leg >= 0 else self._after(self._last_swing_leg))
            push = self.start_push_s if (order is Movement.ORDER_WALK_F or order is Movement.ORDER_WALK_B) else 0.0
            self._schedule_from(first, push)

        elif new is GaitState.WALKING and old is GaitState.STOPPING:
            # resume the clock where the gap left off (or after the in-flight swing + gap)
            if self.swing_leg >= 0:
                delay = self.swing_time_remaining_s() + G
            else:
                delay = max(G - self._grounded_s, 0.0)
            self._schedule_from(self._after(self._last_swing_leg), delay)

        elif new is GaitState.IDLE:
            self.swing_leg = -1
            for leg in self.legs:
                leg.hold()

    def is_grounded(self) -> bool:
        return self.grounded

    def get_swing_leg(self) -> int:
        return self.swing_leg

    def swing_time_remaining_s(self) -> float:
        if self.swing_leg < 0:
            return 0.0
        return self.legs[self.swing_leg].swing_time_remaining_s()

    def update(self, dt_s: float, state: State, imu_data:ImuCorrection=None):
        self._update_twist(dt_s)
        self._state = state

        yaw = state.yaw
        v_cmd, _ = self._cmd
        self._v_cmd_xy = Vec3(v_cmd * math.cos(yaw), v_cmd * math.sin(yaw), 0.0)

        grounded = all(leg.is_grounded() for leg in self.legs)
        self._grounded_s = self._grounded_s + dt_s if grounded else 0.0

        if grounded and not self.grounded:               # a swing just finished
            self.swing_leg = -1
            new_order = self._select_order()
            if new_order is not self._order:
                self._order = new_order
                if self._last_swing_leg >= 0:
                    # keep the time to the next swing, but re-sequence from the most overdue leg
                    remaining = min((1.0 - p) for p in self._psi) * self.period_s
                    self._schedule_from(self._most_overdue_leg(), remaining)
        self.grounded = grounded

        if self.state is GaitState.WALKING:
            self._advance_clock(dt_s)
        elif self.state is GaitState.STOPPING:
            # clock frozen: no new swings, finish the one in flight, then idle after a gap
            if self.grounded and self._grounded_s >= Movement.GAP_DURATION_S:
                self._set_state(GaitState.IDLE)

        for i in range(4):
            self.target_angles[i] = self.legs[i].update(dt_s, imu_data)
            self.driver.set_angles(i, self.target_angles[i])

    def swing_schedule(self, horizon_s: float):
        if self.state is GaitState.IDLE:
            return []
        T, P = Movement.SWING_DURATION_S, self.period_s
        out = []
        for i, leg in enumerate(self.legs):
            if not leg.is_grounded():
                out.append((i, 0.0, leg.swing_time_remaining_s()))
            if self.state is GaitState.STOPPING:
                continue                                  # only swings already in flight
            t = (1.0 - self._psi[i]) * P                  # time until this leg's next swing
            while t < horizon_s:
                out.append((i, t, t + T))
                t += P
        out.sort(key=lambda s: s[1])
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
        return [leg_to_body_frame(i, self.legs[i].get_nominal_landing_pos(
            v_act=Vec3(self._state.linear_vel[0] * 1e3, self._state.linear_vel[1] * 1e3, 0.0),
            v_cmd=self._v_cmd_xy,
            h=max(float(self._state.pos[2]) * 1e3, 1.0)
        )) for i in range(4)]

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

    def _schedule_from(self, first: int, delay_s: float) -> None:
        """Make `first` swing in delay_s, then the rest in order, a quarter period apart."""
        o = self._order
        i0 = o.index(first)
        tau0 = delay_s / self.period_s
        for j in range(4):
            # psi may go negative: that just means "fires later than one period from now"
            self._psi[o[(i0 + j) % 4]] = 1.0 - (tau0 + j / 4.0)

    def _advance_clock(self, dt_s: float) -> None:
        dphase = dt_s / self.period_s
        for i, leg in enumerate(self.legs):
            self._psi[i] += dphase

            if self._psi[i] >= 1.0:
                self._psi[i] -= 1.0

                if leg.is_grounded():
                    self._start_swing(i)

    def _start_swing(self, i: int) -> None:
        self._last_swing_leg = i
        self.swing_leg = i
        self.legs[i].start_swing(self.legs[i].get_nominal_landing_pos(
            v_act=Vec3(self._state.linear_vel[0] * 1e3, self._state.linear_vel[1] * 1e3, 0.0),
            v_cmd=self._v_cmd_xy,
            h=max(float(self._state.pos[2]) * 1e3, 1.0),
        ))