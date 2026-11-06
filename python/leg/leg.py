"""Python port of leg.h / leg.cpp."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple

from .leg_config import (
    BODY_COXA_X_MM,
    FEMUR_LENGTH_MM,
    TIBIA_LENGTH_MM,
    ImuData,
    Movement,
    Offset,
    Theta3,
    Vec3,
    c_max,
    c_min,
)

# External modules from your project (ik.h / servo.h equivalents).
# Expected signatures:
#   ik(params: dict, target: Vec3) -> (reachable: bool, angles: Theta3)
#   invert_angles(angles: Theta3) -> Theta3
#   ServoConfig.limits[leg_idx].{coxa,femur,tibia}.{min_deg,max_deg,clamp(deg)}
from .movement import IK
# from servo import ServoConfig


class LegState(Enum):
    HOLD = auto()
    SWING = auto()
    STANCE = auto()


@dataclass
class FootPositionState:
    foot_curr_pos_lf: Vec3
    foot_target_pos_lf: Vec3

    swing_start_pos_lf: Vec3
    swing_target_pos_lf: Vec3

    stance_start_pos_lf: Vec3
    stance_target_pos_lf: Vec3


# def clamp_to_limits(angles_deg: Theta3, leg_idx: int) -> Theta3:
#     lim = ServoConfig.limits[leg_idx]
#     return Theta3(
#         lim.coxa.clamp(angles_deg.coxa),
#         lim.femur.clamp(angles_deg.femur),
#         lim.tibia.clamp(angles_deg.tibia),
#     )


# def outside_servo_limits(angles_deg: Theta3, leg_idx: int) -> bool:
#     # NOTE: the C++ original returns False when a limit is exceeded and True
#     # when everything is in range -- the opposite of its name. Ported
#     # faithfully; it is unused. Flip the booleans if you intend to use it.
#     lim = ServoConfig.limits[leg_idx]

#     if angles_deg.coxa > lim.coxa.max_deg or angles_deg.coxa < lim.coxa.min_deg:
#         return False
#     if angles_deg.femur > lim.femur.max_deg or angles_deg.femur < lim.femur.min_deg:
#         return False
#     if angles_deg.tibia > lim.tibia.max_deg or angles_deg.tibia < lim.tibia.min_deg:
#         return False
#     return True


def _is_right_leg(leg_id: int) -> bool:
    return leg_id == 1 or leg_id == 2


def _is_front_leg(leg_id: int) -> bool:
    return leg_id == 0 or leg_id == 1


def _smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


class Leg:
    def __init__(self, leg_id: int):
        neutral = Offset.neutral_foot_pos_lf[leg_id]

        self.id = leg_id

        self.pos_state = FootPositionState(
            foot_curr_pos_lf=neutral.copy(),
            foot_target_pos_lf=neutral.copy(),
            swing_start_pos_lf=neutral.copy(),
            swing_target_pos_lf=neutral.copy(),
            stance_start_pos_lf=neutral.copy(),
            stance_target_pos_lf=neutral.copy(),
        )

        self.swing_phase = 0.0
        self.stance_phase = 0.0
        self.leg_state = LegState.HOLD
        self.servo_angles_deg = Theta3()

        self._move_leg()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def is_grounded(self) -> bool:
        return self.leg_state != LegState.SWING

    def get_current_leg_pos(self) -> Vec3:
        return self.pos_state.foot_curr_pos_lf

    def get_nominal_landing_pos(self) -> Vec3:
        end = self.pos_state.swing_start_pos_lf.copy()
        end.x = Offset.neutral_foot_pos_lf[self.id].x + Movement.STEP_LENGTH_MM / 2.0
        return end

    def start_swing(self, target_pos_lf: Vec3) -> None:
        self.pos_state.swing_start_pos_lf = self.pos_state.foot_target_pos_lf.copy()
        self.pos_state.swing_target_pos_lf = target_pos_lf.copy()
        self.swing_phase = 0.0
        self._set_state(LegState.SWING)

    def start_stance(self, target_pos_lf: Vec3) -> None:
        self.pos_state.stance_start_pos_lf = self.pos_state.foot_target_pos_lf.copy()
        self.pos_state.stance_target_pos_lf = target_pos_lf.copy()
        self.stance_phase = 0.0
        self._set_state(LegState.STANCE)

    def update(self, dt_s: float, imu_data: ImuData) -> Theta3:
        """Advance the leg by dt_s and return the target servo angles (deg).

        (The C++ version used an out-parameter; Python just returns it.)
        """
        if self.leg_state == LegState.SWING:
            self._update_swing(dt_s)
        elif self.leg_state == LegState.STANCE:
            self._update_stance(dt_s)
            # self.pid_correction(imu_data)
        else:
            # TODO: CHANGE LATER (from original)
            self.pos_state.foot_target_pos_lf = Offset.neutral_foot_pos_lf[self.id].copy()

        self._move_leg()

        self.pos_state.foot_curr_pos_lf = self.pos_state.foot_target_pos_lf.copy()
        return self.servo_angles_deg

    def pid_correction(self, imu_data: ImuData) -> None:
        if imu_data.valid:
            self.pos_state.foot_target_pos_lf.rotate(imu_data.rot_mat)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _set_state(self, state: LegState) -> None:
        self.leg_state = state

    def _move_leg(self) -> None:
        # flip y for right-side legs
        target = self.pos_state.foot_target_pos_lf.copy()
        target.y *= -1.0 if _is_right_leg(self.id) else 1.0

        angles = IK.solve(
            target=target,
            body_coxa_x_mm=BODY_COXA_X_MM,
            femur_length_mm=FEMUR_LENGTH_MM,
            tibia_length_mm=TIBIA_LENGTH_MM,
            c_min=c_min,
            c_max=c_max,
        )

        if angles is not None:
            # flip angles for reverse-mounted servos (left side)
            if not _is_right_leg(self.id):
                angles = IK.invert_angles(angles)

            # clamp to calibrated limits
            # self.servo_angles_deg = clamp_to_limits(angles, self.id)

        # print(f"leg {self.id}, coxa={angles.coxa}, femur={angles.femur}, tibia={angles.tibia}")
        self.servo_angles_deg = angles

    def _update_stance(self, dt_s: float) -> None:
        self.stance_phase += dt_s / (Movement.SWING_DURATION_S * 3.0)
        self.stance_phase = min(self.stance_phase, 1.0)

        t = self.stance_phase
        start = self.pos_state.stance_start_pos_lf
        target = self.pos_state.stance_target_pos_lf

        alpha = _smoothstep(t)
        self.pos_state.foot_target_pos_lf = start + (target - start) * alpha

        if t >= 1.0:
            self.pos_state.foot_target_pos_lf = target.copy()

    def _update_swing(self, dt_s: float) -> None:
        self.swing_phase += dt_s / Movement.SWING_DURATION_S
        self.swing_phase = min(self.swing_phase, 1.0)

        t = self.swing_phase
        start = self.pos_state.swing_start_pos_lf
        target = self.pos_state.swing_target_pos_lf

        alpha = _smoothstep(t)
        foot = self.pos_state.foot_target_pos_lf
        foot.x = start.x + alpha * (target.x - start.x)
        foot.y = start.y + alpha * (target.y - start.y)

        # lift the foot above the start-to-target line
        clearance = Movement.STEP_HEIGHT_MM * math.sin(math.pi * t)
        foot.z = (start.z + t * (target.z - start.z)) + clearance

        if t >= 1.0:
            self.pos_state.foot_target_pos_lf = target.copy()
            end = self.pos_state.swing_start_pos_lf.copy()
            end.x = Offset.neutral_foot_pos_lf[self.id].x - Movement.STEP_LENGTH_MM / 2.0
            self.start_stance(end)