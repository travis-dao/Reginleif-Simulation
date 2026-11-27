from typing import List, Optional
from enum import Enum, auto

import numpy as np

from movement.leg.leg_config import Vec3
from movement.leg.leg_controller import LegController
from movement.motion.mpc import MPC, MPCProblem, MPCResult, State, NX, NUM_LEGS
from movement.motion.util import foot_forces_to_positions, rotation_matrix_yaw, world_vector_to_leg_frame
from movement.motion.state_estimator import StateEstimator, EstimatorConfig, ImuData

MM = 1e-3
def _v_m(v: Vec3) -> np.ndarray: return np.array([v.x, v.y, v.z]) * MM

class Mode(Enum):
    STAND = auto()   # lc IDLE, holding position
    WALK = auto()    # lc WALKING or STOPPING

class Core:
    def __init__(
        self,
        mpc: MPC,
        leg_ctrl: LegController,
        z_des_m: float,
        yaw_des: Optional[float] = None,
        solve_period_s: float = 0.02,
        gain_mm_per_n: float = 1.0,
        max_corr_mm: float = 10.0,
        ramp_s: float = 1.0,
        gravity_z_mps2: float = -9.81,
    ):
        self.mpc = mpc
        self.lc = leg_ctrl
        self.z_des = z_des_m
        self.yaw_des = yaw_des           # fixed heading to hold; latched from the estimate if None
        self.period = solve_period_s
        self.gain = gain_mm_per_n
        self.max_corr = max_corr_mm
        self.ramp_s = ramp_s            # gain fades 0 -> full after engage
        self._t_engaged = 0.0

        self.mode = Mode.STAND
        self._hold_xy = None

        self.v_cmd_mps = 0.0

        self._offsets: List[Vec3] = [Vec3(0.0, 0.0, 0.0) for _ in range(NUM_LEGS)]
        self._t_since = 1e9
        self.last_result: Optional[MPCResult] = None

        # State estimation lives in the motion subsystem: pos / linear_vel are estimated from
        # leg kinematics + the gait's swing leg; orientation / gyro come from the IMU.
        self.estimator = StateEstimator(EstimatorConfig(
            foot_rel_body_fn=self._foot_rel_body_m,
            swing_leg_fn=self.lc.swing_legs,
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
        return np.array([_v_m(p) for p in self.lc.foot_positions_body_nominal()])

    # ------------------------------------------------------------------
    def step(self, dt_s: float) -> Optional[MPCResult]:
        """MPC tick; solves from the latest estimate() every solve_period."""
        self._t_since += dt_s
        self._t_engaged += dt_s
        if self._t_since < self.period:
            return self.last_result

        self._t_since = 0.0
        s0 = self.state
        if self.yaw_des is None:
            self.yaw_des = s0.yaw
        self.lc.set_body_command(self.v_cmd_mps / MM)
        self._update_mode(s0)

        x_ref = self._reference(s0)
        contact = self._contact_schedule()
        feet = self._foot_rel_pos(s0, x_ref, contact)

        res = self.mpc.solve(MPCProblem(x0=s0, x_ref=x_ref, foot_rel_pos=feet, contact=contact))
        self.last_result = res
        if res.success:  # on failure keep the last good offsets
            self._offsets = self._to_offsets(s0, res.u0.foot_ground_force, contact[0], s0.yaw) # force -> pos offset

        if self.mode is not Mode.STAND:
            self.lc.apply_mpc_corrections(self._offsets)
        return res

    # ------------------------------------------------------------------
    def _reference(self, s0: State) -> List[np.ndarray]:
        dts, N = self.mpc.cfg.dts, self.mpc.N
        if self.mode is Mode.WALK:
            v = self.v_cmd_mps * MM
        else:
            v = 0.0

        p = s0.pos.copy()
        if self.mode is Mode.STAND and self._hold_xy is not None:
            p[:2] = self._hold_xy

        R_des = rotation_matrix_yaw(self.yaw_des)
        v_world = R_des @ np.array([v, 0.0, 0.0])

        v_now = float(R_des[:, 0] @ s0.linear_vel)        # speed along heading
        a_max = 0.5 * self.mpc.cfg.mu * -s0.g             # stay under the friction limit
        out, vk = [], v_now
        for k in range(N):
            vk += np.clip(v - vk, -a_max * dts[k], a_max * dts[k])
            v_world = R_des @ np.array([vk, 0.0, 0.0])
            p[:2] += v_world[:2] * dts[k]
            p[2] = self.z_des
            x = np.zeros(NX)
            x[2] = self.yaw_des; x[3:6] = p; x[9:12] = v_world; x[12] = s0.g
            out.append(x)
        return out

    def _contact_schedule(self) -> List[List[bool]]:
        T, N = self.mpc.T, self.mpc.N
        tmid = 0.5 * (T[:-1] + T[1:])
        contact = [[True] * NUM_LEGS for _ in range(N)]
        for leg, t0, t1 in self.lc.swing_schedule(T[-1]):
            for i in range(N):
                if t0 <= tmid[i] < t1:
                    contact[i][leg] = False
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

    def _to_offsets(self, s0: State, forces: np.ndarray, contact0, yaw: float) -> List[Vec3]:
        """MPC forces -> leg-frame foot offsets (mm).
        The kinematic standing pose already carries the static load, so only the
        deviation from an even weight share is turned into an offset."""
        cfg = self.mpc.cfg
        n_st = max(sum(contact0), 1)
        share = cfg.mass * -s0.g / n_st
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
        if abs(self.v_cmd_mps) > 1e-3:
            self.lc.request_start()
            self.mode = Mode.WALK
        else:
            self.lc.request_stop()
            if self.mode is Mode.WALK and self.lc.idle():
                self._hold_xy = s0.pos[:2].copy()
                self.mode = Mode.STAND