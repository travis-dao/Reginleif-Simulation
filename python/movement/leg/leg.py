"""Python port of leg.h / leg.cpp."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import csv
from pathlib import Path

from .leg_config import (
    BODY_COXA_X_MM,
    FEMUR_LENGTH_MM,
    TIBIA_LENGTH_MM,
    ImuCorrection,
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
    HOLD = auto()     # initial pose: foot pinned to neutral
    SWING = auto()
    STANCE = auto()
    PAUSE = auto()    # NEW: foot frozen wherever it is (gait idle)


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


def _is_right_leg(leg_id: int) -> bool:
    return leg_id == 1 or leg_id == 2


def _is_front_leg(leg_id: int) -> bool:
    return leg_id == 0 or leg_id == 1

# cubic Hermite in xy whose end velocities equal the stance sweep
def _hermite(p0, p1, v0, v1, T, t):          # t in [0,1], velocities in mm/s
    t2, t3 = t*t, t*t*t
    h00, h10 = 2*t3 - 3*t2 + 1, t3 - 2*t2 + t
    h01, h11 = -2*t3 + 3*t2,    t3 - t2
    return h00*p0 + h10*T*v0 + h01*p1 + h11*T*v1


class Leg:
    MPC_OFFSET_TAU_S = 0.025  # low-pass time constant of the MPC foot offset

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

        self.stance_vel = Vec3(0.0, 0.0, 0.0)                       # mm/s: live sweep velocity
        self.step_vec = Vec3(-Movement.STEP_LENGTH_MM, 0.0, 0.0)    # mm: foot displacement over one stance

        self.angle_log = []
        self.foot_pos_log = []
        self.time_s = 0.0

        self._tilt_RT = None      # None = no tilt

        self._move_leg()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def is_grounded(self) -> bool:
        return self.leg_state != LegState.SWING

    def get_current_leg_pos(self) -> Vec3:
        return self.pos_state.foot_curr_pos_lf

    def set_stance(self, vel: Vec3, step: Vec3) -> None:
        self.stance_vel = vel.copy()
        self.step_vec = step.copy()
        # print(f"{self.id} | vec={self.step_vec}, vel={self.stance_vel}")

    def set_body_tilt(self, R_T) -> None:
        self._tilt_RT = R_T

    def get_rear_pos(self) -> Vec3:
        p = Offset.neutral_foot_pos_lf[self.id].copy()
        p.x += self.step_vec.x / 2.0
        p.y += self.step_vec.y / 2.0
        return p

    def get_nominal_landing_pos(self, v_act, v_cmd, h: float, k: float = 1.0) -> Vec3:
        p = Offset.neutral_foot_pos_lf[self.id].copy()

        p.x -= self.step_vec.x / 2.0
        p.y -= self.step_vec.y / 2.0
        omega = math.sqrt(9.81 * 1e3 / h)

        p.x += k * (v_act.x - v_cmd.x) / omega       # k ~ 0.5..1
        p.y += k * (v_act.y - v_cmd.y) / omega

        return p
    
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

    def hold(self) -> None:
        """Freeze the trajectory at the current foot target (gait idle)."""
        if self.leg_state == LegState.SWING:
            return                                 # never freeze mid-swing
        self._set_state(LegState.PAUSE)

    def resume_stance(self) -> None:
        """Continue the stance sweep from the frozen position toward the rear limit."""
        self.start_stance(self.get_rear_pos())

    def get_state(self) -> LegState:
        return self.leg_state

    def update(self, dt_s: float, imu_data: Optional[ImuCorrection] = None) -> Theta3:
        self.time_s += dt_s

        if self.leg_state == LegState.SWING:
            self._update_swing(dt_s)
        elif self.leg_state == LegState.STANCE:
            self._update_stance(dt_s)
            # self.pid_correction(imu_data)
        elif self.leg_state == LegState.PAUSE:
            pass                                   # keep foot_target_pos_lf as is
        else:                                      # HOLD
            self.pos_state.foot_target_pos_lf = Offset.neutral_foot_pos_lf[self.id].copy()

        # low-pass the MPC offset; swing legs carry no ground force -> decay to 0
        zero = Vec3(0.0, 0.0, 0.0)
        cmd = zero if self.leg_state == LegState.SWING else self._mpc_offset_cmd
        alpha = 1.0 - math.exp(-dt_s / self.MPC_OFFSET_TAU_S)
        self.mpc_offset_lf = self.mpc_offset_lf + (cmd - self.mpc_offset_lf) * alpha
        # print(f"{self.id} | offset={self.mpc_offset_lf}")

        # nominal gait position: never modified by PID or MPC
        p_ik = self.pos_state.foot_target_pos_lf.copy()

        # if self.leg_state == LegState.STANCE and imu_data and imu_data.valid:
        #     p_ik.rotate(imu_data.rot_mat)          # rotate a throwaway copy only

        # body tilt: rotate about the body centre, not the hip
        if self._tilt_RT is not None:
            # print(f"before={p_ik}")
            base = Offset.leg_base_pos_bf[self.id]
            p_bf = base + p_ik          # Vec3 + Vec3 returns a new object
            p_bf.rotate(self._tilt_RT)
            # print(f"R_T={self._tilt_RT}")
            p_ik = p_bf - base
            # print(f"after={p_ik}")

        # curr = where the foot is commanded (trajectory + PID + MPC offset)
        self.pos_state.foot_curr_pos_lf = p_ik + self.mpc_offset_lf
        self._log_foot_pos()

        self._move_leg()

        return self.servo_angles_deg

    def export_angle_log(self) -> None:
        path = Path(f"python/movement/logs/leg{self.id}_trajectory_angles.csv")
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
        path = Path(f"python/movement/logs/leg{self.id}_foot_pos.csv")
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

        target = self.pos_state.foot_curr_pos_lf.copy()
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
        rear = self.get_rear_pos()
        sv = self.stance_vel
        # keep sweeping only while the rear limit is still ahead of the foot
        if (rear.x - foot.x) * sv.x + (rear.y - foot.y) * sv.y > 0.0:
            foot.x += sv.x * dt_s
            foot.y += sv.y * dt_s
        start = self.pos_state.stance_start_pos_lf
        span = math.hypot(rear.x - start.x, rear.y - start.y)
        done = math.hypot(foot.x - start.x, foot.y - start.y)
        self.stance_phase = 1.0 if span <= 1e-6 else min(done / span, 1.0)   # debug only

    def _update_swing(self, dt_s: float) -> None:
        self.swing_phase += dt_s / Movement.SWING_DURATION_S
        self.swing_phase = min(self.swing_phase, 1.0)

        t = self.swing_phase
        start = self.pos_state.swing_start_pos_lf
        target = self.pos_state.swing_target_pos_lf

        foot = self.pos_state.foot_target_pos_lf
        sv = self.stance_vel
        foot.x = _hermite(start.x, target.x, sv.x, sv.x, Movement.SWING_DURATION_S, t)
        foot.y = _hermite(start.y, target.y, sv.y, sv.y, Movement.SWING_DURATION_S, t)

        # lift the foot above the start-to-target line
        # Asymmetric lift: fast ascent, gentle descent
        LIFT_FRACTION = 0.30

        if t < LIFT_FRACTION:
            # Fast lift: ease out toward maximum height
            u = t / LIFT_FRACTION
            clearance = Movement.STEP_HEIGHT_MM * (1.0 - (1.0 - u) ** 2)
        else:
            # Gentle descent: ease in toward ground level
            u = (t - LIFT_FRACTION) / (1.0 - LIFT_FRACTION)
            clearance = Movement.STEP_HEIGHT_MM * (1.0 - u ** 2)

        foot.z = (start.z + t * (target.z - start.z)) + clearance

        if t >= 1.0:
            self.pos_state.foot_target_pos_lf = target.copy()
            self.start_stance(self.get_rear_pos())