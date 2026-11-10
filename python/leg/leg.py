"""Python port of leg.h / leg.cpp."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto

import csv
from pathlib import Path

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
from .limits import ServoLimit, ServoConfig, LegServoLimits

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


def clamp_to_limits(angles_deg: Theta3, leg_idx: int) -> Theta3:
    lim = ServoConfig.limits[leg_idx]
    return Theta3(
        lim.coxa.clamp(angles_deg.coxa),
        lim.femur.clamp(angles_deg.femur),
        lim.tibia.clamp(angles_deg.tibia),
    )


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
    MPC_OFFSET_TAU_S = 0.05  # low-pass time constant of the MPC foot offset
    GAP_DURATION_S = 0.1


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

        # MPC foot offset (leg frame, mm). Kept OUTSIDE the swing/stance
        # trajectory so it never accumulates into it.
        self.mpc_offset_lf = Vec3(0.0, 0.0, 0.0)
        self._mpc_offset_cmd = Vec3(0.0, 0.0, 0.0)

        self.angle_log = []
        self.foot_pos_log = []
        self.time_s = 0.0

        self._move_leg()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def is_grounded(self) -> bool:
        return self.leg_state != LegState.SWING

    def get_current_leg_pos(self) -> Vec3:
        return self.pos_state.foot_curr_pos_lf

    def get_nominal_landing_pos(self) -> Vec3:
        # Built from the neutral pose (not swing_start_pos_lf, which is stale
        # when this is called just before start_swing()).
        end = Offset.neutral_foot_pos_lf[self.id].copy()
        end.x += Movement.STEP_LENGTH_MM / 2.0
        return end

    def swing_time_remaining_s(self) -> float:
        if self.leg_state != LegState.SWING:
            return 0.0
        return (1.0 - self.swing_phase) * Movement.SWING_DURATION_S

    def set_mpc_offset(self, offset_lf: Vec3) -> None:
        """Set the desired MPC foot offset (leg frame, mm). Applied smoothly in update()."""
        self._mpc_offset_cmd = offset_lf.copy()

    def start_swing(self, target_pos_lf: Vec3) -> None:
        self.pos_state.swing_start_pos_lf = self.pos_state.foot_target_pos_lf.copy()
        self.pos_state.swing_target_pos_lf = target_pos_lf.copy()
        self.swing_phase = 0.0
        self._set_state(LegState.SWING)

    def start_stance(self, target_pos_lf: Vec3) -> None:
        self.pos_state.stance_start_pos_lf = self.pos_state.foot_target_pos_lf.copy()
        self.pos_state.stance_target_pos_lf = (
            target_pos_lf.copy() if target_pos_lf is not None else self.get_rear_pos()
        )
        self.stance_phase = 0.0
        self._set_state(LegState.STANCE)

    def get_state(self) -> LegState:
        return self.leg_state

    @staticmethod
    def _stance_speed_mm_s() -> float:
        # a foot is in stance for 3 swing periods (the other 3 legs' swings)
        return Movement.STEP_LENGTH_MM / (3.0 * Movement.SWING_DURATION_S + 4.0 * Leg.GAP_DURATION_S)

    def get_rear_pos(self) -> Vec3:
        """End-of-stance (rear) foot position, leg frame."""
        p = Offset.neutral_foot_pos_lf[self.id].copy()
        p.x -= Movement.STEP_LENGTH_MM / 2.0
        return p

    def update(self, dt_s: float, imu_data: ImuData) -> Theta3:
        self.time_s += dt_s

        if self.leg_state == LegState.SWING:
            self._update_swing(dt_s)
        elif self.leg_state == LegState.STANCE:
            self._update_stance(dt_s)
            # self.pid_correction(imu_data)
        else:
            # TODO: CHANGE LATER (from original)
            self.pos_state.foot_target_pos_lf = Offset.neutral_foot_pos_lf[self.id].copy()

        # low-pass the MPC offset; swing legs carry no ground force -> decay to 0
        zero = Vec3(0.0, 0.0, 0.0)
        cmd = zero if self.leg_state == LegState.SWING else self._mpc_offset_cmd
        alpha = 1.0 - math.exp(-dt_s / self.MPC_OFFSET_TAU_S)
        self.mpc_offset_lf = self.mpc_offset_lf + (cmd - self.mpc_offset_lf) * alpha
        # print(self.mpc_offset_lf)

        self._move_leg()

        # curr = where the foot really is (trajectory + MPC offset)
        self.pos_state.foot_curr_pos_lf = self.pos_state.foot_target_pos_lf + self.mpc_offset_lf
        self._log_foot_pos()
        return self.servo_angles_deg

    def pid_correction(self, imu_data: ImuData) -> None:
        if imu_data.valid:
            self.pos_state.foot_target_pos_lf.rotate(imu_data.rot_mat)

    def export_angle_log(self) -> None:
        path = Path(f"python/logs/leg{self.id}_trajectory_angles.csv")
        path.parent.mkdir(parents=True, exist_ok=True)

        if not self.angle_log:
            print("No data samples recorded; CSV not written.")
            return

        fieldnames = list(self.angle_log[0].keys())

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.angle_log)

        print(f"Recorded samples: {len(self.angle_log)}")
        print(f"CSV saved to: {path.resolve()}")

    def export_pos_log(self) -> None:
        path = Path(f"python/logs/leg{self.id}_foot_pos.csv")
        path.parent.mkdir(parents=True, exist_ok=True)

        if not self.foot_pos_log:
            print("No data samples recorded; CSV not written.")
            return

        fieldnames = list(self.foot_pos_log[0].keys())

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.foot_pos_log)

        print(f"Recorded samples: {len(self.foot_pos_log)}")
        print(f"CSV saved to: {path.resolve()}")

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _set_state(self, state: LegState) -> None:
        self.leg_state = state

    def _move_leg(self) -> None:
        # try the full MPC offset first; if IK says unreachable (which would leave the
        # servos frozen at their old angles) retry with less offset, then none.
        angles = None

        
        target = (self.pos_state.foot_target_pos_lf + self.mpc_offset_lf).copy()
        target.y *= -1.0 if _is_right_leg(self.id) else 1.0  # flip y for right-side legs
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

            row = {}
            row[f"time_s"] = self.time_s
            row[f"target_coxa"] = angles.coxa
            row[f"target_femur"] = angles.femur
            row[f"target_tibia"] = angles.tibia
            # clamp to calibrated limits
            self.servo_angles_deg = clamp_to_limits(angles, self.id)
            row[f"real_coxa"] = self.servo_angles_deg.coxa
            row[f"real_femur"] = self.servo_angles_deg.femur
            row[f"real_tibia"] = self.servo_angles_deg.tibia
            row[f"swinging"] = 1 if self.leg_state == LegState.SWING else 0
            self.angle_log.append(row)
        else:
            print("ik failed")

        # print(f"leg {self.id}, coxa={angles.coxa}, femur={angles.femur}, tibia={angles.tibia}")
        # self.servo_angles_deg = angles

    def _log_foot_pos(self):
        row = {}
        row[f"time_s"] = self.time_s
        row[f"target_x"] = self.pos_state.foot_target_pos_lf.x
        row[f"offset_x"] = self.mpc_offset_lf.x
        row[f"final_x"] = self.pos_state.foot_curr_pos_lf.x

        row[f"target_y"] = self.pos_state.foot_target_pos_lf.y
        row[f"offset_y"] = self.mpc_offset_lf.y
        row[f"final_y"] = self.pos_state.foot_curr_pos_lf.y

        row[f"target_z"] = self.pos_state.foot_target_pos_lf.z
        row[f"offset_z"] = self.mpc_offset_lf.z
        row[f"final_z"] = self.pos_state.foot_curr_pos_lf.z
        
        row[f"swinging"] = 1 if self.leg_state == LegState.SWING else 0

        self.foot_pos_log.append(row)
        return

    def _update_stance(self, dt_s: float) -> None:
        foot = self.pos_state.foot_target_pos_lf
        rear_x = self.pos_state.stance_target_pos_lf.x

        # constant-velocity slide toward the rear; stop (don't overshoot) at the limit
        foot.x = max(foot.x - self._stance_speed_mm_s() * dt_s, rear_x)

        # progress, for logging/debug only
        start_x = self.pos_state.stance_start_pos_lf.x
        span = start_x - rear_x
        self.stance_phase = 1.0 if span <= 1e-6 else min((start_x - foot.x) / span, 1.0)

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
        clearance = Movement.STEP_HEIGHT_MM * math.sin(math.pi * t) ** 0.5
        foot.z = (start.z + t * (target.z - start.z)) + clearance

        if t >= 1.0:
            self.pos_state.foot_target_pos_lf = target.copy()
            self.start_stance(self.get_rear_pos())