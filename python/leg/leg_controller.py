"""Python port of leg_controller.h / leg_controller.cpp."""
from __future__ import annotations

from typing import List, Optional

from .leg import Leg
from .leg_config import ImuData, Movement, Offset, Theta3, Vec3
from .movement import LegDriver

NUM_SERVOS = 12


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

        self.walking = False

    def init(self) -> None:
        for i in range(4):
            self.legs[i] = Leg(i)

    def is_grounded(self) -> bool:
        return self.grounded

    def get_swing_leg(self) -> int:
        return self.swing_leg

    def swing_time_remaining_s(self) -> float:
        if self.swing_leg < 0:
            return 0.0
        return self.legs[self.swing_leg].swing_time_remaining_s()

    def start_walk(self) -> None:
        """Move every leg out of HOLD and into STANCE, then let the gait pick the swing leg."""
        self.swing_leg = -1
        self.walking = True
        for leg in self.legs:
            leg.start_stance(leg.get_rear_pos())

    def update(self, dt_s: float, imu_data: ImuData) -> None:
        grounded = all(leg.is_grounded() for leg in self.legs)

        # a swing just finished: open the all-stance gap
        if grounded and not self.grounded:
            self.swing_leg = -1
            self.gap_remaining_s = Leg.GAP_DURATION_S
        self.grounded = grounded

        if self.grounded:
            if not self.walking:
                self.start_walk()
                self.gap_remaining_s = 0.0      # first swing starts immediately
            self.gap_remaining_s -= dt_s
            if self.gap_remaining_s <= 0.0:
                nxt = Movement.next_leg_gait[self._last_swing_leg + 1]
                self._last_swing_leg = nxt
                self.swing_leg = nxt
                self.legs[nxt].start_swing(self.legs[nxt].get_nominal_landing_pos())

        for i in range(4):
            self.target_angles[i] = self.legs[i].update(dt_s, imu_data)
            self.driver.set_angles(i, self.target_angles[i])
    
    def swing_schedule(self, horizon_s: float):
        """[(leg, t_start, t_end)] of predicted swings, seconds from now, including gaps."""
        if not self.walking:
            return []
        T, G = Movement.SWING_DURATION_S, Leg.GAP_DURATION_S
        out = []
        if self.swing_leg >= 0:
            t_end = self.swing_time_remaining_s()
            out.append((self.swing_leg, 0.0, t_end))
            t_next = t_end + G
        else:
            t_next = max(self.gap_remaining_s, 0.0)
        last = self._last_swing_leg
        while t_next < horizon_s:
            last = Movement.next_leg_gait[last + 1]
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