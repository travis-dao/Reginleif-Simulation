"""Python port of leg_controller.h / leg_controller.cpp."""
from __future__ import annotations

from typing import List, Optional
from dataclasses import dataclass
import mujoco

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

    def init(self) -> None:
        for i in range(4):
            self.legs[i] = Leg(i)

    def is_grounded(self) -> bool:
        return self.grounded

    def get_swing_leg(self) -> int:
        return self.swing_leg

    def update(self, dt_s: float, imu_data: ImuData) -> None:
        # gait sequencing
        self.grounded = all(leg.is_grounded() for leg in self.legs)

        # only start next swing sequence if all legs are grounded
        if self.grounded:
            nxt = Movement.next_leg_gait[self.swing_leg + 1]
            self.swing_leg = nxt

            nominal_landing_bf = leg_to_body_frame(nxt, self.legs[nxt].get_nominal_landing_pos())
            self.legs[nxt].start_swing(body_to_leg_frame(nxt, nominal_landing_bf))

        # leg movement calculations
        for i in range(4):
            self.target_angles[i] = self.legs[i].update(dt_s, imu_data)
            self.driver.set_angles(i, self.target_angles[i])

    def foot_positions_body(self) -> List[Vec3]:
        """Current foot positions, body frame."""
        return [leg_to_body_frame(i, self.legs[i].get_current_leg_pos()) for i in range(4)]

    def landing_targets_body(self) -> List[Vec3]:
        """Where each foot touches down. Only the airborne leg differs from its
        current position; the next leg to lift lands after the MPC horizon
        (one swing) ends, so it doesn't need one."""
        out = []
        for i in range(4):
            foot_lf = (
                self.legs[i].get_nominal_landing_pos()
                if i == self.swing_leg
                else self.legs[i].get_current_leg_pos()
            )
            out.append(leg_to_body_frame(i, foot_lf))
        return out