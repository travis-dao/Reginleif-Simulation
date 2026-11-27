import sys
from pathlib import Path

# Add the python/ directory to the module search path.
PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))


from typing import List, Optional
import math

from .leg import Leg
from .leg_config import Movement, Offset, Theta3, Vec3, Gait, GAIT_WALK_F, GAIT_WALK_B
from .movement import LegDriver
from enum import Enum, auto
from motion.mpc import State

NUM_SERVOS = 12


class MoveState(Enum):
    IDLE = auto()       # all feet planted, stance sweep frozen, no swings scheduled
    WALKING = auto()    # swings are driven purely by the duty-based gait clock
    STOPPING = auto()   # stop requested: clock frozen, finish any swing in flight, then IDLE


def leg_to_body_frame(leg: int, foot_pos_lf: Vec3) -> Vec3: return Offset.leg_base_pos_bf[leg] + foot_pos_lf
def body_to_leg_frame(leg: int, foot_pos_bf: Vec3) -> Vec3: return foot_pos_bf - Offset.leg_base_pos_bf[leg]


class LegController:
    def __init__(self, model, data):
        self.legs: List[Optional[Leg]] = [None] * 4
        self.driver: LegDriver = LegDriver(model, data, debug=False)
        self.target_angles: List[Theta3] = [Theta3() for _ in range(4)]
        self.target_angles_buffer: List[float] = [0.0] * NUM_SERVOS

        self._grounded = False
        self.swing_leg = -1
        self._last_swing_leg = -1

        self.move_state = MoveState.IDLE

        self._psi = [0.0] * 4          # per-leg clock; the leg swings when it wraps past 1
        self.start_push_s = 0.2

        self._cmd = 0.0                      # requested forward speed (mm/s)

        self._gait: Gait = GAIT_WALK_F

        self._state = State()
        self.body_roll = 0.0
        self.body_pitch = 0.0

        self.run_enabled = False

        for i in range(4):
            self.legs[i] = Leg(i)

    # -------------------------- update --------------------------
    def update(self, dt_s: float, state: State | None):
        if state is not None:
            self._state = state

            grounded = all(leg.grounded() for leg in self.legs)

            if grounded and not self._grounded: # a swing just finished
                self.swing_leg = -1
                self._set_gait(self._select_gait())
            self._grounded = grounded

            if self.move_state is MoveState.WALKING:
                self._advance_clock(dt_s)
            elif self.move_state is MoveState.STOPPING: # clock frozen: no new swings; idle as soon as the swing in flight lands
                if self._grounded:
                    self._set_state(MoveState.IDLE)

        for i in range(4):
            self.target_angles[i] = self.legs[i].update(dt_s, self._gait)
            self.driver.set_angles(i, self.target_angles[i])

    # -------------------------- commands / setters --------------------------
    def set_body_command(self, v_mm_s: float) -> None: self._cmd = v_mm_s

    def request_start(self) -> None:
        if self.move_state in (MoveState.IDLE, MoveState.STOPPING):
            self._set_state(MoveState.WALKING)

    def request_stop(self) -> None:
        if self.move_state is MoveState.WALKING:
            self._set_state(MoveState.STOPPING)

    def apply_mpc_corrections(self, offsets_lf: List[Vec3]) -> None:
        for leg, off in zip(self.legs, offsets_lf): leg.set_mpc_offset(off)

    # -------------------------- queries --------------------------
    def grounded(self) -> bool:
        return self._grounded

    def idle(self) -> bool:
        return self.move_state is MoveState.IDLE

    def swing_legs(self):
        return [i for i, leg in enumerate(self.legs) if not leg.grounded()]

    def swing_time_remaining_s(self) -> float:
        if self.swing_leg < 0:
            return 0.0
        return self.legs[self.swing_leg].swing_time_remaining_s()

    def swing_schedule(self, horizon_s: float):
        if self.move_state is MoveState.IDLE:
            return []
        T, P = self.swing_s, self.period_s
        out = []
        for i, leg in enumerate(self.legs):
            if not leg.grounded():
                out.append((i, 0.0, leg.swing_time_remaining_s()))
            if self.move_state is MoveState.STOPPING:
                continue                                  # only swings already in flight
            t = (1.0 - self._psi[i]) * P                  # time until this leg's next swing
            while t < horizon_s:
                out.append((i, t, t + T))
                t += P
        out.sort(key=lambda s: s[1])
        return out

    def foot_positions_body(self) -> List[Vec3]:
        return [leg_to_body_frame(i, self.legs[i].cur_pos()) for i in range(4)]
    def foot_positions_body_nominal(self) -> List[Vec3]:
        return [leg_to_body_frame(i, self.legs[i].foot_state.target) for i in range(4)]
    def landing_targets_body(self) -> List[Vec3]:
        return [leg_to_body_frame(i, self.legs[i].nom_land_pos()) for i in range(4)]

    # -------------------------- internals --------------------------
    def _set_state(self, new: MoveState) -> None:
        old, self.move_state = self.move_state, new

        if new is MoveState.WALKING and old is MoveState.IDLE:
            for leg in self.legs:
                leg.resume_stance()
            gait = self._select_gait()
            if not self._set_gait(gait, self.start_push_s):
                self._schedule_from(self._after(self._last_swing_leg), self.start_push_s)

        elif new is MoveState.WALKING and old is MoveState.STOPPING:
            pass    # the clock was frozen, not reset: just let it run on from where it stopped

        elif new is MoveState.IDLE:
            self.swing_leg = -1
            for leg in self.legs:
                leg.hold()

    def _most_overdue_leg(self) -> int:
        best, best_d = -1, -1.0
        for i, leg in enumerate(self.legs):
            cur = leg.cur_pos()
            d = math.hypot(cur.x, cur.y)
            if d > best_d:
                best, best_d = i, d
        return best

    def _select_gait(self) -> Gait:
        return GAIT_WALK_B if self._cmd < 0 else GAIT_WALK_F

    def _after(self, last: int) -> int:
        o = self._gait.order
        if last < 0:
            return o[0]
        return o[(o.index(last) + 1) % len(o)]

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
                if leg.grounded():
                    self._start_swing(i)

    def _start_swing(self, i: int) -> None:
        self._last_swing_leg = i
        self.swing_leg = i
        self.legs[i].start_swing()

    def _set_gait(self, gait: Gait, delay_s: Optional[float] = None) -> bool:
        if gait is self._gait:
            return False
        if delay_s is None:
            delay_s = max(min(1.0 - p for p in self._psi), 0.0) * self.period_s
        self._gait = gait
        uniform = all(abs(o - 0.25 * j) < 1e-9 for j, o in enumerate(gait.phase_offsets))
        first = (self._last_swing_leg if uniform and self._last_swing_leg >= 0
                else gait.order[0])
        self._schedule_from(first, delay_s)
        return True

    @property
    def timing(self):      return self._gait.timing
    @property
    def period_s(self):    return self._gait.timing.period_s
    @property
    def t_stance_s(self):  return self._gait.timing.t_stance_s
    @property
    def duty(self):        return self.t_stance_s / self.period_s          # fraction of the cycle a foot is on the ground
    @property
    def swing_s(self):     return (1.0 - self.duty) * self.period_s        # swing length follows from the duty
    @property
    def v_max_mm_s(self):  return self._gait.timing.v_max_mm_s
    