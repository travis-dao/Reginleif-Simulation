"""
Python port of the Cheetah-style kinematic Kalman filter (state_estimator.cpp/.h).

    x = [ p_body(3), v_body(3), p_foot_FL(3), p_foot_FR(3), p_foot_BR(3), p_foot_BL(3) ]  (world, m)

predict : p += v dt, v constant (no accelerometer), feet stationary
measure : foot position relative to body (leg FK), body velocity from stance-foot
          kinematics (-R (w x r + dr)), and foot height = 0 on flat ground

Orientation + gyro play the role of the IMU and come from MuJoCo (see make_imu).
Position + linear velocity come from the filter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import mujoco
import numpy as np

from mpc.mpc import State

NL = 4
NS = 6 + 3 * NL   # 18
NY = 7 * NL       # 28: 12 pos + 12 vel + 4 height


# ----------------------------------------------------------------------
# Data containers
# ----------------------------------------------------------------------
@dataclass
class ImuData:
    valid: bool = True
    rpy_rad: np.ndarray = field(default_factory=lambda: np.zeros(3))      # ZYX roll, pitch, yaw
    gyro_rad_s: np.ndarray = field(default_factory=lambda: np.zeros(3))   # BODY frame


def rot_from_rpy(rpy: np.ndarray) -> np.ndarray:
    """rpy = (roll, pitch, yaw) [rad], ZYX convention -> body-to-world rotation."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


# ----------------------------------------------------------------------
# Estimator
# ----------------------------------------------------------------------
@dataclass
class EstimatorConfig:
    # Replace the C++ LegController dependencies with callables:
    foot_rel_body_fn: Callable[[], np.ndarray]   # -> (NL, 3) foot positions in body frame [m]
    swing_leg_fn: Callable[[], int]              # -> index of swing leg, -1 = all feet planted

    gravity_z: float = -9.81   # only used to fill State.g

    # Process noise (scaled by dt each tick)
    process_pos: float = 1.0 / 20.0
    process_vel: float = 9.8 / 20.0
    process_foot: float = 1.0

    # Measurement noise
    meas_foot_pos: float = 0.001
    meas_foot_vel: float = 0.1
    meas_foot_height: float = 0.001

    high_suspect: float = 100.0        # noise multiplier (1 + (1-trust)*k) for non-stance feet
    touchdown_trust_s: float = 0.05    # trust ramps 0 -> 1 over this time after touchdown
    foot_vel_alpha: float = 0.5        # low-pass weight of newest finite-difference foot velocity

    init_var_pos: float = 1e-2
    init_var_vel: float = 1e-2
    init_var_foot: float = 1e-2


class StateEstimator:
    def __init__(self, cfg: EstimatorConfig):
        self.cfg = cfg
        self.x = np.zeros(NS)
        self.P = np.eye(NS)
        self.state = State()
        self.state.g = cfg.gravity_z
        self.initialized = False
        self.have_prev = False

        self.t_stance = np.full(NL, 1e9)
        self.trust = np.ones(NL)
        self.r_prev = np.zeros((NL, 3))
        self.dr = np.zeros((NL, 3))

    # ------------------------------------------------------------------
    def reset(self, p0: np.ndarray, v0: np.ndarray, imu: ImuData) -> None:
        """Init at a known pose."""
        if not imu.valid:
            return
        rpy = np.asarray(imu.rpy_rad, dtype=float)
        gyro = np.asarray(imu.gyro_rad_s, dtype=float)
        R = rot_from_rpy(rpy)
        r = self._foot_rel_body()

        self.x = np.zeros(NS)
        self.x[0:3] = p0
        self.x[3:6] = v0
        for i in range(NL):
            self.x[6 + 3 * i: 9 + 3 * i] = p0 + R @ r[i]

        c = self.cfg
        diag = np.zeros(NS)
        diag[0:3] = c.init_var_pos
        diag[3:6] = c.init_var_vel
        diag[6:] = c.init_var_foot
        self.P = np.diag(diag)

        self.t_stance[:] = 1e9
        self.trust[:] = 1.0
        self.r_prev = r.copy()
        self.dr[:] = 0.0
        self.have_prev = True
        self.initialized = True
        self._publish(rpy, R, gyro)

    # ------------------------------------------------------------------
    def update(self, dt: float, imu: ImuData) -> State:
        """One filter tick. Call every control tick."""
        if not imu.valid or dt <= 0.0:
            return self.state                      # hold last estimate
        if not self.initialized:
            self.reset(np.zeros(3), np.zeros(3), imu)

        c = self.cfg
        rpy = np.asarray(imu.rpy_rad, dtype=float)
        gyro = np.asarray(imu.gyro_rad_s, dtype=float)   # body frame
        R = rot_from_rpy(rpy)

        self._update_contact(dt)

        # Foot kinematics in body frame + filtered finite-difference foot velocity
        r = self._foot_rel_body()
        for i in range(NL):
            dr_raw = (r[i] - self.r_prev[i]) / dt if self.have_prev else np.zeros(3)
            self.dr[i] = c.foot_vel_alpha * dr_raw + (1.0 - c.foot_vel_alpha) * self.dr[i]
            self.r_prev[i] = r[i]
        self.have_prev = True

        # ---------------- predict (constant velocity) ----------------
        A = np.eye(NS)
        A[0:3, 3:6] = dt * np.eye(3)
        self.x = A @ self.x

        q_diag = np.zeros(NS)
        q_diag[0:3] = dt * c.process_pos
        q_diag[3:6] = dt * c.process_vel
        for i in range(NL):
            s = 1.0 + (1.0 - self.trust[i]) * c.high_suspect
            q_diag[6 + 3 * i: 9 + 3 * i] = dt * c.process_foot * s
        self.P = A @ self.P @ A.T + np.diag(q_diag)

        # ---------------- measurement ----------------
        y = np.zeros(NY)
        C = np.zeros((NY, NS))
        Rd = np.zeros(NY)

        v_pred = self.x[3:6]
        pz_pred = self.x[2]

        for i in range(NL):
            ip, iv, ih, ifoot = 3 * i, 12 + 3 * i, 24 + i, 6 + 3 * i
            trust = self.trust[i]
            scale = 1.0 + (1.0 - trust) * c.high_suspect

            p_f = R @ r[i]   # body -> foot, world axes

            # (1) foot position relative to body: p_body - p_foot = -R r
            y[ip:ip + 3] = -p_f
            C[ip:ip + 3, 0:3] = np.eye(3)
            C[ip:ip + 3, ifoot:ifoot + 3] = -np.eye(3)

            # (2) body velocity from a planted foot: v_body = -R (w x r + dr)
            dp_f = R @ (np.cross(gyro, r[i]) + self.dr[i])
            y[iv:iv + 3] = (1.0 - trust) * v_pred + trust * (-dp_f)
            C[iv:iv + 3, 3:6] = np.eye(3)

            # (3) planted foot height = 0 (flat ground)
            y[ih] = (1.0 - trust) * (pz_pred + p_f[2])
            C[ih, ifoot + 2] = 1.0

            Rd[ip:ip + 3] = c.meas_foot_pos * scale
            Rd[iv:iv + 3] = c.meas_foot_vel * scale
            Rd[ih] = c.meas_foot_height * scale

        # ---------------- update ----------------
        innov = y - C @ self.x
        CP = C @ self.P
        S = CP @ C.T + np.diag(Rd)
        K = np.linalg.solve(S, CP).T          # P and S are symmetric

        self.x = self.x + K @ innov
        self.P = (np.eye(NS) - K @ C) @ self.P
        self.P = 0.5 * (self.P + self.P.T)

        self._publish(rpy, R, gyro)
        return self.state

    # ------------------------------------------------------------------
    def _foot_rel_body(self) -> np.ndarray:
        return np.asarray(self.cfg.foot_rel_body_fn(), dtype=float).reshape(NL, 3)

    def _update_contact(self, dt: float) -> None:
        """Crawl gait: exactly one swing leg at a time (-1 = all feet planted)."""
        c = self.cfg
        swing = int(c.swing_leg_fn())
        for i in range(NL):
            if i == swing:
                self.t_stance[i] = 0.0
                self.trust[i] = 0.0
            else:
                self.t_stance[i] += dt
                self.trust[i] = (min(1.0, self.t_stance[i] / c.touchdown_trust_s)
                                 if c.touchdown_trust_s > 0.0 else 1.0)

    def _publish(self, rpy: np.ndarray, R: np.ndarray, gyro: np.ndarray) -> None:
        s = self.state
        s.pos = self.x[0:3].copy()
        s.linear_vel = self.x[3:6].copy()
        s.roll, s.pitch, s.yaw = map(float, rpy)
        s.angular_vel = R @ gyro      # world frame, as the MPC expects
        s.g = self.cfg.gravity_z


# ----------------------------------------------------------------------
# MuJoCo adapters
# ----------------------------------------------------------------------
class MujocoFootKinematics:
    """
    Stand-in for LegController::footPositionsBody().

    Returns foot positions in the body frame [m]. `foot_ids` are MuJoCo site ids (default)
    or geom ids (kind="geom"), ordered to match the estimator's leg order (FL, FR, BR, BL).

    The body-frame origin is taken as the body's CoM (data.xipos), matching the estimator's
    assumption. Use origin="frame" to use the body frame origin (data.xpos) instead.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, body_id: int,
                 foot_ids: Sequence[int], kind: str = "site", origin: str = "com"):
        assert len(foot_ids) == NL
        self.model, self.data, self.body_id = model, data, body_id
        self.foot_ids = list(foot_ids)
        self.kind = kind
        self.origin = origin

    def __call__(self) -> np.ndarray:
        d = self.data
        R = d.xmat[self.body_id].reshape(3, 3)                  # body -> world
        origin = d.xipos[self.body_id] if self.origin == "com" else d.xpos[self.body_id]
        world = d.site_xpos if self.kind == "site" else d.geom_xpos
        return np.array([R.T @ (world[i] - origin) for i in self.foot_ids])


def make_imu(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> ImuData:
    """MuJoCo -> ImuData: rpy (ZYX) from the free-joint quaternion, gyro in the BODY frame."""
    jnt = model.body_jntadr[body_id]
    qadr = model.jnt_qposadr[jnt]
    vadr = model.jnt_dofadr[jnt]

    w, x, y, z = data.qpos[qadr + 3:qadr + 7]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    # MuJoCo free-joint qvel[3:6] is already the body-frame angular velocity = a gyro
    gyro = data.qvel[vadr + 3:vadr + 6].copy()
    return ImuData(valid=True, rpy_rad=np.array([roll, pitch, yaw]), gyro_rad_s=gyro)


# ----------------------------------------------------------------------
# Example wiring (the estimator is owned by Core; see core.py / main.py)
# ----------------------------------------------------------------------
# core = Core(mpc_solver, controller, z_des_m=0.0, gravity_z_mps2=float(model.opt.gravity[2]))
# core.reset(p0, v0, make_imu(model, data, body_id))      # once, at a known pose
#
# every control tick (dt = control period):
#   imu = make_imu(model, data, body_id)                  # sim only; on hardware: your IMU driver
#   s = core.estimate(dt, imu)                            # pos / linear_vel estimated; rpy + gyro from IMU
#   core.step(dt, time_s)                                 # MPC uses core.state