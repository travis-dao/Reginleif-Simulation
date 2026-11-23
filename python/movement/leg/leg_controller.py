import sys
from pathlib import Path

# Add the python/ directory to the module search path.
PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))


from typing import List, Optional
import math

from .leg import Leg
from .leg_config import (Movement, Offset, Theta3, Vec3,
                         Gait, GAIT_WALK_F, GAIT_WALK_B, GAIT_TURN_L, GAIT_TURN_R)
from .movement import LegDriver
from enum import Enum, auto
from motion.mpc import State

NUM_SERVOS = 12


class MoveState(Enum):
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

        self.move_state = MoveState.IDLE

        self._psi = [0.0] * 4          # per-leg clock; the leg swings when it wraps past 1
        self._grounded_s = 0.0         # time since the last swing ended (replaces gap_remaining_s)
        self.start_push_s = 0.2

        self._nom_bf = [leg_to_body_frame(i, Offset.neutral_foot_pos_lf[i]) for i in range(4)]
        self.max_step_y_mm = 100.0            # tune: sideways reach of the coxa
        self.a_v_mm_s2 = 300.0               # twist slew limits, tune
        self.a_w_rad_s2 = 0.5
        self._cmd = (0.0, 0.0)               # requested (v mm/s, w rad/s)
        self._v_cmd_xy: Vec3 = Vec3(0.0, 0.0, 0.0)
        self._v = 0.0                        # achieved sweep twist
        self._w = 0.0
        self._geom = (0.0, 0.0)              # twist that defines step geometry

        self.turn_v_eps = 5.0 # mm/s, below this, cound yaw cmd as "turn in place"
        self.turn_w_eps = 0.02 #rad/s

        self._gait: Gait = GAIT_WALK_F

        self._state = State()
        self.body_roll = 0.0
        self.body_pitch = 0.0
        
        for i in range(4):
            self.legs[i] = Leg(i)

    # commands
    def set_body_command(self, v_mm_s: float, w_rad_s: float) -> None:
        self._cmd = (v_mm_s, w_rad_s)

    def body_twist(self):
        """Achieved (v mm/s, w rad/s) the stance feet are producing now."""
        return self._v, self._w

    def request_start(self) -> None:
        if self.move_state in (MoveState.IDLE, MoveState.STOPPING):
            self._set_state(MoveState.WALKING)

    def request_stop(self) -> None:
        if self.move_state is MoveState.WALKING:
            self._set_state(MoveState.STOPPING)

    def is_idle(self) -> bool:
        return self.move_state is MoveState.IDLE
    
    def _most_overdue_leg(self) -> int:
        best, best_d = -1, -1.0
        for i, leg in enumerate(self.legs):
            cur = leg.get_current_leg_pos()
            d = math.hypot(cur.x, cur.y)
            if d > best_d:
                best, best_d = i, d
        return best

    def _select_gait(self) -> Gait:
        v, w = self._cmd
        if abs(v) < self.turn_v_eps and abs(w) > self.turn_w_eps:
            return GAIT_TURN_L if w > 0 else GAIT_TURN_R
        return GAIT_WALK_B if v < 0 else GAIT_WALK_F

    def _after(self, last: int) -> int:
        o = self._gait.order
        if last < 0:
            return o[0]
        return o[(o.index(last) + 1) % len(o)]

    # transitions
    def _set_state(self, new: MoveState) -> None:
        old, self.move_state = self.move_state, new
        G = self.timing.gap_s

        if new is MoveState.WALKING and old is MoveState.IDLE:
            for leg in self.legs:
                leg.resume_stance()
            gait = self._select_gait()
            push = self.start_push_s if gait in (GAIT_WALK_F, GAIT_WALK_B) else 0.0
            if not self._set_gait(gait, push):
                self._schedule_from(self._after(self._last_swing_leg), push)

        elif new is MoveState.WALKING and old is MoveState.STOPPING:
            # resume the clock where the gap left off (or after the in-flight swing + gap)
            if self.swing_leg >= 0:
                delay = self.swing_time_remaining_s() + G
            else:
                delay = max(G - self._grounded_s, 0.0)
            self._schedule_from(self._after(self._last_swing_leg), delay)

        elif new is MoveState.IDLE:
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

    def update(self, dt_s: float, state: State | None):
        if state is not None:
            self._update_twist(dt_s)
            self._state = state

            yaw = state.yaw
            v_cmd, _ = self._cmd
            self._v_cmd_xy = Vec3(v_cmd * math.cos(yaw), v_cmd * math.sin(yaw), 0.0)

            grounded = all(leg.is_grounded() for leg in self.legs)
            self._grounded_s = self._grounded_s + dt_s if grounded else 0.0

            if grounded and not self.grounded:               # a swing just finished
                self.swing_leg = -1
                self._set_gait(self._select_gait())
            self.grounded = grounded

            if self.move_state is MoveState.WALKING:
                self._advance_clock(dt_s)
            elif self.move_state is MoveState.STOPPING:
                # clock frozen: no new swings, finish the one in flight, then idle after a gap
                if self.grounded and self._grounded_s >= self.timing.gap_s:
                    self._set_state(MoveState.IDLE)

        for i in range(4):
            self.target_angles[i] = self.legs[i].update(dt_s, self._gait)
            self.driver.set_angles(i, self.target_angles[i])

    def swing_schedule(self, horizon_s: float):
        if self.move_state is MoveState.IDLE:
            return []
        T, P = Movement.SWING_DURATION_S, self.period_s
        out = []
        for i, leg in enumerate(self.legs):
            if not leg.is_grounded():
                out.append((i, 0.0, leg.swing_time_remaining_s()))
            if self.move_state is MoveState.STOPPING:
                continue                                  # only swings already in flight
            t = (1.0 - self._psi[i]) * P                  # time until this leg's next swing
            while t < horizon_s:
                out.append((i, t, t + T))
                t += P
        out.sort(key=lambda s: s[1])
        return out
    
    def apply_mpc_corrections(self, offsets_lf: List[Vec3]) -> None:
        for leg, off in zip(self.legs, offsets_lf): leg.set_mpc_offset(off)

    def foot_positions_body(self) -> List[Vec3]:
        return [leg_to_body_frame(i, self.legs[i].get_current_leg_pos()) for i in range(4)]

    def landing_targets_body(self) -> List[Vec3]: 
        return [leg_to_body_frame(i, self.legs[i].get_nominal_landing_pos()) for i in range(4)]
    
    """Scale (v, w) together, which keeps R = v/w"""
    def _limit(self, v, w):
        T = self.t_stance_s
        step_len = self.timing.step_length_mm
        k = 1.0
        for p in self._nom_bf:
            sx = abs(v - w * p.y) * T
            sy = abs(w * p.x) * T
            k = min(k, step_len / sx if sx > 1e-9 else 1.0, self.max_step_y_mm / sy if sy > 1e-9 else 1.0)
        return v * k, w * k

    def _update_twist(self, dt_s: float) -> None:
        if self.move_state is MoveState.IDLE:
            self._v = self._w = 0.0                    # nothing sweeps
            self._geom = self._limit(*self._cmd)       # so the first swing lands correctly
        elif self.move_state is MoveState.WALKING:
            vc, wc = self._limit(*self._cmd)
            self._geom = (vc, wc)
            self._v += max(-self.a_v_mm_s2 * dt_s, min(self.a_v_mm_s2 * dt_s, vc - self._v))
            self._w += max(-self.a_w_rad_s2 * dt_s, min(self.a_w_rad_s2 * dt_s, wc - self._w))
        # STOPPING: hold the last twist until the gait goes IDLE

        vg, wg = self._geom
        T = self.t_stance_s
        for leg, p in zip(self.legs, self._nom_bf):
            vel = Vec3(-(self._v - self._w * p.y), -(self._w * p.x), 0.0)
            step = Vec3((vg - wg * p.y) * T, (wg * p.x) * T, 0.0)
            leg.set_stance(vel, step)

    def _schedule_from(self, first: int, delay_s: float) -> None:
        o, offs = self._gait.order, self._gait.phase_offsets
        i0 = o.index(first)
        tau0 = delay_s / self.period_s
        for j in range(4):
            self._psi[o[(i0 + j) % 4]] = 1.0 - (tau0 + offs[j])

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
        self.legs[i].start_swing(self.legs[i].get_nominal_landing_pos());

    def _set_gait(self, gait: Gait, delay_s: Optional[float] = None) -> bool:
        if gait is self._gait:
            return False
        if delay_s is None:
            delay_s = max(min(1.0 - p for p in self._psi), 0.0) * self.period_s
        self._gait = gait
        # first = self._most_overdue_leg() if self._last_swing_leg >= 0 else gait.order[0]
        first = self._last_swing_leg if self._last_swing_leg >= 0 else gait.order[0]
        self._schedule_from(first, delay_s)
        return True

    @property
    def timing(self):      return self._gait.timing
    @property
    def period_s(self):    return self._gait.timing.period_s
    @property
    def t_stance_s(self):  return self._gait.timing.t_stance_s
    @property
    def v_max_mm_s(self):  return self._gait.timing.v_max_mm_s