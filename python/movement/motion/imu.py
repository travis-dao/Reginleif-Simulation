import math
from dataclasses import dataclass, field
from typing import Optional

import mujoco
import numpy as np

MAX_COMP_DEG = 15.0
INTEGRAL_MAX = 10.0
DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi

@dataclass
class EulerAngles:
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    
@dataclass
class ImuData:
    valid: bool = True
    rpy_rad: np.ndarray = field(default_factory=lambda: np.zeros(3))      # ZYX roll, pitch, yaw
    gyro_rad_s: np.ndarray = field(default_factory=lambda: np.zeros(3))   # BODY frame


class Imu:
    def __init__(
        self,
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

    # --------------------------------------------------------------- update

    def update(self, dt_s: Optional[float] = None) -> None:
        """Call after mj_step/mj_forward."""
        if dt_s is None:
            dt_s = self.model.opt.timestep

        self._update_imu()

    def read(self) -> ImuData:
        vadr = self.model.jnt_dofadr[self.model.body_jntadr[self.body_id]]

        R_body = self.data.xmat[self.body_id].reshape(3, 3)          # body -> world
        gyro_body = self.data.qvel[vadr + 3:vadr + 6].copy()    # free-joint angular vel is body-frame

        return ImuData(valid=True, rpy_rad=_rpy_zyx(R_body), gyro_rad_s=gyro_body)


def _rpy_zyx(R: np.ndarray) -> np.ndarray:
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])