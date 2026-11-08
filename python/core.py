"""Glue between LegController (mm, body frame, leg-by-leg gait) and MPC (m, N, world frame).

Per control tick:
    bridge.step(dt, model, data, body_id)   # read state -> solve MPC -> push foot offsets
    leg_ctrl.update(dt, imu)                # gait + offsets -> servo angles
    leg_ctrl.update_model_angles()
"""
from __future__ import annotations

from typing import List, Optional
import csv
from pathlib import Path

import numpy as np

from leg.leg_config import Movement, Vec3
from leg.leg_controller import LegController
from mpc.mpc import MPC, MPCProblem, MPCResult, State, NX, NUM_LEGS
from mpc.util import (
    foot_forces_to_positions,
    read_state,
    rotation_matrix_yaw,
    world_vector_to_leg_frame,
)

MM = 1e-3


def _v_m(v: Vec3) -> np.ndarray:
    """Vec3 in mm -> ndarray in m."""
    return np.array([v.x, v.y, v.z]) * MM


class Core:
    def __init__(
        self,
        mpc: MPC,
        leg_ctrl: LegController,
        z_des_m: float,
        vx_des_mps: Optional[float] = None,
        yaw_des: Optional[float] = None,
        solve_period_s: float = 0.03,
        gain_mm_per_n: float = 0.15,
        max_corr_mm: float = 12.0,
        ramp_s: float = 2.0,
    ):
        self.mpc = mpc
        self.lc = leg_ctrl
        self.z_des = z_des_m
        # crawl gait: a foot travels STEP_LENGTH back in the body frame over 3 swing durations
        self.vx_des = (
            vx_des_mps if vx_des_mps is not None
            else Movement.STEP_LENGTH_MM * MM / (3.0 * Movement.SWING_DURATION_S)
        )
        self.yaw_des = yaw_des          # None -> latch yaw on first step
        self.period = solve_period_s
        self.gain = gain_mm_per_n
        self.max_corr = max_corr_mm
        self.ramp_s = ramp_s            # gain fades 0 -> full after engage
        self._t_engaged = 0.0

        self._offsets: List[Vec3] = [Vec3(0.0, 0.0, 0.0) for _ in range(NUM_LEGS)]
        self._t_since = 1e9
        self.last_result: Optional[MPCResult] = None

        self.force_log = []
        self.force_csv_path = Path("python/logs/mpc_ground_forces.csv")

    # ------------------------------------------------------------------
    def step(self, dt_s: float, model, data, body_id: int) -> Optional[MPCResult]:
        self._t_since += dt_s
        self._t_engaged += dt_s
        if self._t_since < self.period:
            return self.last_result
        self._t_since = 0.0

        s0 = read_state(model, data, body_id)
        feet_world = self._foot_positions_world(s0)

        if self.yaw_des is None:
            self.yaw_des = s0.yaw

        x_ref = self._reference(s0)
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
                "time_s": float(data.time),
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
        """Constant-velocity reference at steps 1..N: level body, fixed height, forward speed."""
        dt = self.mpc.cfg.dt
        yaw = self.yaw_des
        v_w = rotation_matrix_yaw(yaw) @ np.array([self.vx_des, 0.0, 0.0])
        out = []
        for k in range(self.mpc.N):
            x = np.zeros(NX)
            x[2] = yaw
            pos = s0.pos + v_w * (k + 1) * dt
            pos[2] = self.z_des # TODO ??
            x[3:6] = pos
            x[9:12] = v_w
            x[12] = s0.g          # gravity slot must equal State.g
            out.append(x)
        return out

    def _contact_schedule(self) -> List[List[bool]]:
        """Predicted stance/swing per step. Gait is one swing at a time, back to back:
        current swing leg (for its remaining time), then the next leg in the gait table, ..."""
        N, dt = self.mpc.N, self.mpc.cfg.dt
        T = Movement.SWING_DURATION_S
        contact = [[True] * NUM_LEGS for _ in range(N)]
        leg = self.lc.get_swing_leg()
        if leg < 0:
            return contact
        t0, t1 = 0.0, self.lc.swing_time_remaining_s()
        while t0 < N * dt:
            for k in range(N):
                if t0 <= (k + 0.5) * dt < t1:
                    contact[k][leg] = False
            leg = Movement.next_leg_gait[leg + 1]
            t0, t1 = t1, t1 + T
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