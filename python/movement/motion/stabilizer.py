"""Python port of the ESP32 Stabilizer (PID attitude compensator),
using MuJoCo body orientation and angular velocity.
"""

import math
from dataclasses import dataclass
from typing import Optional

import mujoco
import numpy as np

from movement.leg.leg_config import ImuCorrection

MAX_COMP_DEG = 15.0
INTEGRAL_MAX = 10.0
DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi


@dataclass
class Config:
    K_p: float
    K_i: float
    K_d: float


@dataclass
class EulerAngles:
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


class Stabilizer:
    def __init__(
        self,
        config: Config,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        body_name: str,
        gyro_sensor: Optional[str] = None,
    ):
        self.model = model
        self.data = data

        self.body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        if self.body_id < 0:
            raise ValueError(f"Body '{body_name}' not found in model")

        self.gyro_adr = None
        if gyro_sensor is not None:
            sid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SENSOR, gyro_sensor
            )
            if sid < 0:
                raise ValueError(f"Sensor '{gyro_sensor}' not found in model")

            if model.sensor_dim[sid] != 3:
                raise ValueError(
                    f"Gyro sensor '{gyro_sensor}' must have dimension 3"
                )

            self.gyro_adr = model.sensor_adr[sid]

        self.target_angles_deg = np.zeros(3)
        self.gyro_dps = np.zeros(3)
        self.angles_deg = EulerAngles()

        self.K_p = config.K_p
        self.K_i = config.K_i
        self.K_d = config.K_d

        self.integral_sum = np.zeros(3)
        self.rot_mat = np.eye(3)

    # ------------------------------------------------------------ accessors

    def angles(self) -> np.ndarray:
        """Standard [roll, pitch, yaw] Euler angles in degrees."""
        return np.array([
            self.angles_deg.roll,
            self.angles_deg.pitch,
            self.angles_deg.yaw,
        ])

    def gyro(self) -> np.ndarray:
        """Angular velocity in the body frame, degrees/s."""
        return self.gyro_dps.copy()

    # --------------------------------------------------------- angle helpers

    @staticmethod
    def _euler_zyx_deg(R: np.ndarray) -> np.ndarray:
        """Extract standard ZYX Euler angles from a body-to-world rotation.
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
        Returns [roll, pitch, yaw] in degrees.
        """
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.asin(float(np.clip(-R[2, 0], -1.0, 1.0)))
        yaw = math.atan2(R[1, 0], R[0, 0])
        return np.array([roll, pitch, yaw]) * RAD_TO_DEG

    # ------------------------------------------------------------- IMU update

    def _update_imu(self):
        # MuJoCo xmat is the body's rotation relative to the world.
        R_body = self.data.xmat[self.body_id].reshape(3, 3)

        e = self._euler_zyx_deg(R_body)
        self.angles_deg = EulerAngles(*e)

        if self.gyro_adr is not None:
            # Assumes the gyro sensor axes align with the body axes.
            # MuJoCo gyro sensors report angular velocity in the sensor frame.
            g_rad = np.asarray(
                self.data.sensordata[
                    self.gyro_adr:self.gyro_adr + 3
                ],
                dtype=float,
            )
        else:
            # Request local-frame angular velocity.
            # MuJoCo spatial velocity layout: [angular, linear].
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(
                self.model,
                self.data,
                mujoco.mjtObj.mjOBJ_BODY,
                self.body_id,
                vel,
                1,
            )
            g_rad = vel[:3]

        self.gyro_dps = g_rad * RAD_TO_DEG

    # --------------------------------------------------------- PID controller

    def _update_integral_axis(
        self,
        error_axis: float,
        d_axis: float,
        kp: float,
        ki: float,
        integral_axis: float,
        dt_s: float,
    ) -> float:
        if ki == 0.0:
            return integral_axis

        proposed_integral = float(np.clip(
            integral_axis + error_axis * dt_s,
            -INTEGRAL_MAX,
            INTEGRAL_MAX,
        ))

        proposed_output = (
            kp * error_axis
            + ki * proposed_integral
            + d_axis
        )

        drives_further_into_saturation = (
            (
                proposed_output > MAX_COMP_DEG
                and ki * error_axis > 0.0
            )
            or (
                proposed_output < -MAX_COMP_DEG
                and ki * error_axis < 0.0
            )
        )

        return (
            integral_axis
            if drives_further_into_saturation
            else proposed_integral
        )

    def _update_pid(
        self,
        dt_s: float,
        angles_in_deg: np.ndarray,
        gyro_dps: np.ndarray,
    ):
        if dt_s <= 0.0:
            return

        angles_in_deg = np.asarray(angles_in_deg, dtype=float)
        gyro_dps = np.asarray(gyro_dps, dtype=float)

        # Shortest signed yaw error; roll/pitch are normally kept near zero.
        error = self.target_angles_deg - angles_in_deg
        error[2] = (error[2] + 180.0) % 360.0 - 180.0

        # Derivative-on-measurement avoids derivative kick.
        d_term = -gyro_dps * self.K_d

        for i in range(3):
            self.integral_sum[i] = self._update_integral_axis(
                error[i],
                d_term[i],
                self.K_p,
                self.K_i,
                self.integral_sum[i],
                dt_s,
            )

        comp_deg = (
            error * self.K_p
            + self.integral_sum * self.K_i
            + d_term
        )

        comp_deg = np.clip(
            comp_deg,
            -MAX_COMP_DEG,
            MAX_COMP_DEG,
        )

        self.rot_mat = self.get_rotation_matrix(comp_deg * DEG_TO_RAD)

    # --------------------------------------------------------------- update

    def update(
        self,
        dt_s: Optional[float] = None,
        out_data: Optional[ImuCorrection] = None,
    ) -> ImuCorrection:
        """Call after mj_step/mj_forward."""
        if dt_s is None:
            dt_s = self.model.opt.timestep

        self._update_imu()
        self._update_pid(dt_s, self.angles(), self.gyro())

        if out_data is None:
            out_data = ImuCorrection()

        out_data.rot_mat = self.rot_mat.copy()
        out_data.valid = True

        return out_data

    # ------------------------------------------------------ rotation matrices

    @staticmethod
    def get_rotation_matrix(a: np.ndarray) -> np.ndarray:
        """Construct Rz(yaw) @ Ry(pitch) @ Rx(roll).

        Input angles are radians in [roll, pitch, yaw] order.
        """
        roll, pitch, yaw = a

        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)

        Rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ])

        Ry = np.array([
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ])

        Rz = np.array([
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ])

        return Rz @ Ry @ Rx
