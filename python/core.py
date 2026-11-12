"""Glue between LegController (mm, body frame, leg-by-leg gait) and MPC (m, N, world frame).

Per control tick:
    imu = make_imu(model, data, body_id)    # sim only: MuJoCo stands in for the IMU
    s = core.estimate(dt, imu)              # state estimator (Kalman filter), EVERY tick
    core.step(dt, data.time)                # solve MPC (every solve_period) -> push foot offsets
    leg_ctrl.update(dt, imu)                # gait + offsets -> servo angles
    leg_ctrl.update_model_angles()
"""
from __future__ import annotations

from typing import List, Optional
import csv
from pathlib import Path
from enum import Enum, auto

import numpy as np

from leg.leg_config import Movement, Vec3
from leg.leg_controller import LegController
from leg.leg import Leg
from mpc.mpc import MPC, MPCProblem, MPCResult, State, NX, NUM_LEGS
from mpc.util import (
    foot_forces_to_positions,
    rotation_matrix_yaw,
    world_vector_to_leg_frame,
)

from state_estimator import StateEstimator, EstimatorConfig, ImuData

MM = 1e-3
def _v_m(v: Vec3) -> np.ndarray:
    """Vec3 in mm -> ndarray in m."""
    return np.array([v.x, v.y, v.z]) * MM

def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi

class Mode(Enum):
    STAND = auto()   # lc IDLE, holding position
    WALK = auto()    # lc WALKING or STOPPING

class Core:
    def __init__(
        self,
        mpc: MPC,
        leg_ctrl: LegController,
        z_des_m: float,
        vx_des_mps: Optional[float] = None,
        yaw_des: Optional[float] = None,
        solve_period_s: float = 0.03,
        gain_mm_per_n: float = 1.0,
        max_corr_mm: float = 12.0,
        ramp_s: float = 2.0,
        gravity_z_mps2: float = -9.81,
    ):
        self.mpc = mpc
        self.lc = leg_ctrl
        self.z_des = z_des_m
        self.yaw_des = yaw_des
        self.period = solve_period_s
        self.gain = gain_mm_per_n
        self.max_corr = max_corr_mm
        self.ramp_s = ramp_s            # gain fades 0 -> full after engage
        self._t_engaged = 0.0

        self.mode = Mode.STAND
        self._hold_xy = None

        self.v_cmd_mps = 0.0
        self.yaw_rate_cmd = 0.0          # rad/s, + = left
        self.yaw_leash_rad = 0.3         # how far yaw_des may run ahead of the measured yaw

        self._offsets: List[Vec3] = [Vec3(0.0, 0.0, 0.0) for _ in range(NUM_LEGS)]
        self._t_since = 1e9
        self.last_result: Optional[MPCResult] = None

        self.force_log = []
        self.force_csv_path = Path("python/logs/mpc_ground_forces.csv")

        # State estimation lives in the motion subsystem: pos / linear_vel are estimated from
        # leg kinematics + the gait's swing leg; orientation / gyro come from the IMU.
        self.estimator = StateEstimator(EstimatorConfig(
            foot_rel_body_fn=self._foot_rel_body_m,
            swing_leg_fn=self.lc.get_swing_leg,
            gravity_z=gravity_z_mps2,
        ))
        self.state: State = self.estimator.state

    # ------------------------------------------------------------------
    def reset(self, p0: np.ndarray, v0: np.ndarray, imu: ImuData) -> None:
        """Init the estimator at a known pose (sim: spawn pose; hardware: zeros / neutral)."""
        self.estimator.reset(p0, v0, imu)

    def estimate(self, dt_s: float, imu: ImuData) -> State:
        """Estimator tick. Call EVERY control tick (also before the MPC engages), not just on
        MPC solves, so the foot velocities and contact trust stay current."""
        self.state = self.estimator.update(dt_s, imu)
        return self.state

    def _foot_rel_body_m(self) -> np.ndarray:
        """Foot positions in the body frame, (4, 3) in metres (the estimator's FK input)."""
        return np.array([_v_m(p) for p in self.lc.foot_positions_body()])

    # ------------------------------------------------------------------
    def step(self, dt_s: float, time_s: float) -> Optional[MPCResult]:
        """MPC tick; solves from the latest estimate() every solve_period."""
        self._t_since += dt_s
        self._t_engaged += dt_s
        if self._t_since < self.period:
            return self.last_result

        elapsed = self._t_since
        self._t_since = 0.0
        s0 = self.state
        feet_world = self._foot_positions_world(s0)
        self.lc.set_body_command(self.v_cmd_mps / MM, self.yaw_rate_cmd)
        self._update_mode(s0)

        # unwrap yaw_des against the measured yaw and leash it
        self.yaw_des = s0.yaw + float(np.clip(_wrap(self.yaw_des - s0.yaw), -self.yaw_leash_rad, self.yaw_leash_rad))
        x_ref = self._reference(s0)
        if self.mode is Mode.WALK:
            self.yaw_des += self.lc.body_twist()[1] * elapsed
        contact = self._contact_schedule()
        feet = self._foot_rel_pos(s0, x_ref, contact)

        res = self.mpc.solve(MPCProblem(x0=s0, x_ref=x_ref, foot_rel_pos=feet, contact=contact))
        self.last_result = res
        if res.success:  # on failure keep the last good offsets
            swing_leg = self.lc.get_swing_leg()
            forces = np.asarray(
                res.u0.foot_ground_force, dtype=float
            ).reshape(4, 3)

            # Print swing leg and ground reaction forces
            leg_names = ["FL", "FR", "BR", "BL"]
            swing_name = (
                leg_names[swing_leg]
                if 0 <= swing_leg < 4
                else "NONE"
            )

            # One row per successful MPC solve.
            row = {
                "time_s": float(time_s),
                "swing_leg": swing_name,
                "swing_leg_id": int(swing_leg),
                "yaw_rad": float(s0.yaw),
            }

            row["com_x_m"] = float(s0.pos[0])
            row["com_y_m"] = float(s0.pos[1])
            row["com_z_m"] = float(s0.pos[2])

            for i, name in enumerate(["FL", "FR", "BR", "BL"]):
                # World-frame foot position, metres.
                row[f"{name}_x_m"] = float(feet_world[i, 0])
                row[f"{name}_y_m"] = float(feet_world[i, 1])
                row[f"{name}_z_m"] = float(feet_world[i, 2])

                # Optimized ground reaction force, newtons.
                row[f"{name}_Fx_N"] = float(forces[i, 0])
                row[f"{name}_Fy_N"] = float(forces[i, 1])
                row[f"{name}_Fz_N"] = float(forces[i, 2])

            self.force_log.append(row)

            # Convert forces to foot position corrections
            self._offsets = self._to_offsets(res.u0.foot_ground_force, contact[0], s0.yaw)
            # print(self._offsets)
        self.lc.apply_mpc_corrections(self._offsets)
        return res

    def export_force_log(self, filepath="python/logs/mpc_ground_forces.csv"):
        # print(self.force_log)
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not self.force_log:
            print("No force samples recorded; CSV not written.")
            return

        fieldnames = list(self.force_log[0].keys())

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.force_log)

        print(f"Recorded samples: {len(self.force_log)}")
        print(f"CSV saved to: {path.resolve()}")

    # ------------------------------------------------------------------
    def _reference(self, s0: State) -> List[np.ndarray]:
        dt, N = self.mpc.cfg.dt, self.mpc.N
        if self.mode is Mode.WALK:
            v_mm, w = self.lc.body_twist()
            v = v_mm * MM
        else:
            v, w = 0.0, 0.0

        p = s0.pos.copy()
        if self.mode is Mode.STAND and self._hold_xy is not None:
            p[:2] = self._hold_xy

        out = []
        for k in range(N):
            yaw_mid = self.yaw_des + w * (k + 0.5) * dt     # midpoint integration of the arc
            yaw_end = self.yaw_des + w * (k + 1) * dt
            p[:2] += rotation_matrix_yaw(yaw_mid)[:2, :2] @ np.array([v, 0.0]) * dt
            p[2] = self.z_des

            x = np.zeros(NX)
            x[2] = yaw_end
            x[3:6] = p
            x[8] = w                                        # yaw rate
            x[9:12] = rotation_matrix_yaw(yaw_end) @ np.array([v, 0.0, 0.0])
            x[12] = s0.g
            out.append(x)
        return out

    def _contact_schedule(self) -> List[List[bool]]:
        N, dt = self.mpc.N, self.mpc.cfg.dt
        contact = [[True] * NUM_LEGS for _ in range(N)]
        for leg, t0, t1 in self.lc.swing_schedule(N * dt):
            for k in range(N):
                if t0 <= (k + 0.5) * dt < t1:
                    contact[k][leg] = False
        return contact

    def _foot_rel_pos(self, s0: State, x_ref, contact) -> List[np.ndarray]:
        """Foot minus CoM, world frame, metres, per step.
        Planted feet are fixed in the WORLD (so r shrinks/grows as the CoM advances);
        a foot that lands inside the horizon is planted at its nominal touchdown point."""
        R0 = rotation_matrix_yaw(s0.yaw)
        world = [s0.pos + R0 @ _v_m(p) for p in self.lc.foot_positions_body()]
        land_bf = [_v_m(p) for p in self.lc.landing_targets_body()]
        airborne = [not contact[0][i] for i in range(NUM_LEGS)]

        out = []
        for k in range(self.mpc.N):
            com = s0.pos if k == 0 else x_ref[k - 1][3:6]   # CoM at the START of step k
            R = rotation_matrix_yaw(x_ref[k][2])
            r = np.zeros((NUM_LEGS, 3))
            for i in range(NUM_LEGS):
                if contact[k][i]:
                    if airborne[i]:                          # touchdown
                        world[i] = com + R @ land_bf[i]
                        airborne[i] = False
                else:
                    airborne[i] = True
                r[i] = world[i] - com
            out.append(r)
        return out

    def _to_offsets(self, forces: np.ndarray, contact0, yaw: float) -> List[Vec3]:
        """MPC forces -> leg-frame foot offsets (mm).
        The kinematic standing pose already carries the static load, so only the
        deviation from an even weight share is turned into an offset."""
        cfg = self.mpc.cfg
        n_st = max(sum(contact0), 1)
        share = cfg.mass * 9.81 / n_st
        fw = []
        for i in range(NUM_LEGS):
            if contact0[i]:
                f = forces[i].copy()
                f[2] -= share
            else:
                f = np.zeros(3)
            fw.append(Vec3(float(f[0]), float(f[1]), float(f[2])))
        ramp = min(1.0, self._t_engaged / self.ramp_s) if self.ramp_s > 0 else 1.0
        corr_w = foot_forces_to_positions(fw, self.gain * ramp, self.max_corr)
        # print(corr_w) # prints 0
        # print(
        #     f"leg={i} "
        #     f"force={forces[i]} "
        #     f"nominal_share={share:.2f} "
        #     f"force_deviation={f} "
        #     f"offset={corr_w[i]}"
        # )
        return [world_vector_to_leg_frame(c, yaw) for c in corr_w]

    
    def _foot_positions_world(self, s0: State) -> np.ndarray:
        """Return FL, FR, BR, BL foot positions in world coordinates (m)."""
        R = rotation_matrix_yaw(s0.yaw)
        feet_body = self.lc.foot_positions_body()

        return np.array([
            s0.pos + R @ _v_m(foot)
            for foot in feet_body
        ], dtype=float)

    def _update_mode(self, s0):
        moving = abs(self.v_cmd_mps) > 1e-3 or abs(self.yaw_rate_cmd) > 1e-3
        if moving:
            self.lc.request_start()
            self.mode = Mode.WALK
        else:
            self.lc.request_stop()
            if self.mode is Mode.WALK and self.lc.is_idle():
                self._hold_xy = s0.pos[:2].copy()
                self.yaw_des = s0.yaw            # latch heading too
                self.mode = Mode.STAND