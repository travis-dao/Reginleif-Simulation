#!/usr/bin/env python3
"""Collect (observation, action) pairs from the MPC-driven quadruped for training the MLP policy.

    mjpython python/sim/collect_data.py                      # drive it yourself with the live input
    mjpython python/sim/collect_data.py --random             # viewer, random command schedule
    python   python/sim/collect_data.py --headless --duration 600   # fast, no viewer (implies --random)

Policy format (must match policy_weights.h):
    obs  = 3 frames (oldest first) x 45 dims = 135
      frame: [0:3]   gyro (body)            * GYRO_SCALE
             [3:6]   gravity dir (body)     unit vector
             [6:9]   cmd vx, vy * CMD_SCALE_LIN, yaw rate * CMD_SCALE_ANG
             [9:21]  joint pos - default    rad
             [21:33] joint vel              * QD_SCALE
             [33:45] previous action        clamped to [-1, 1]
    action (12) = clip((q_target - q_default) / ACTION_SCALE, -1, 1)     <- tanh output range

q_target is what the MPC + gait pipeline commands to the position actuators (data.ctrl) on that
tick, in RADIANS (LegDriver converts the gait's servo degrees to rad). It is already post-IK,
post side-inversion (left legs) and post servo-limit clamp, so it is the final servo angle.
Joint order: leg0..leg3 (FL, FR, BR, BL) x (coxa, femur, tibia), via the leg{i}_{joint}_act names.
The observation is sampled BEFORE that command is applied, so (obs_t -> action_t) is exactly
what the policy has to reproduce.

Deployment: q_target_rad = q_default + ACTION_SCALE * action. Servo degrees = q_target_rad * 180/pi
(no further inversion or clamping-order changes: the values are already in servo space).

Output .npz keys:
    obs        (N, 135)  stacked, scaled observation (history padded with the first frame)
    actions    (N, 12)   normalized, clipped teacher action
    q_target   (N, 12)   raw teacher joint targets [rad]  (re-normalize from this if you change scales)
    q_default  (12,)
    cmd        (N, 3)    unscaled [vx m/s, vy m/s, yaw_rate rad/s]
    mpc_forces (N, 12)   MPC ground forces (world frame, N) -- optional auxiliary target
    mpc_ok     (N,)      False where the QP failed and the last offsets were held (filter these out)
    time, episode, meta (json string: scales, joint names, policy rate, ...)

Load:
    d = np.load("dataset.npz"); X, Y = d["obs"], d["actions"]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# Add the python/ directory to the module search path (same as main.py).
PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

from movement.leg.leg_controller import LegController
from movement.motion.mpc import MPC, MPCConfig
from movement.motion.core import Core
from movement.motion.state_estimator import make_imu
from movement.motion.stabilizer import Stabilizer, Config
from sim.build_sim_xml import build_sim_xml
from input.smooth import smoothing

# ----------------------------------------------------------------------------
# Policy-side constants. KEEP IN SYNC with policy_weights.h (these are placeholders).
# ----------------------------------------------------------------------------
OBS_HISTORY = 3
OBS_FRAME_DIM = 45
POLICY_ACT_DIM = 12

GYRO_SCALE = 0.25
CMD_SCALE_LIN = 2.0
CMD_SCALE_ANG = 0.25
QD_SCALE = 0.05
ACTION_SCALE_RAD = 1.0     # action = 1.0  <->  1 rad away from the default pose (see --action-scale)

# Policy joint order = LegDriver's actuators: leg0..leg3 (FL, FR, BR, BL) x (coxa, femur, tibia).
LEG_NAMES = ("leg0", "leg1", "leg2", "leg3")
JOINTS = ("coxa", "femur", "tibia")

# ----------------------------------------------------------------------------
# Sim setup (mirrors main.py)
# ----------------------------------------------------------------------------
SRC = "models/body/xml/reginleif.xml"
DST = "models/body/xml/reginleif_sim.xml"
KP = 30.0
FORCE = 9.4
START_HEIGHT = 0.5
HORIZON = 10
MPC_DT = 0.03
SETTLE_TIME_S = 0.0
SETTLE_VEL = 0.1
MPC_GAIN = 1.3
REST_S = 2.0               # stand still before commands start
SKIP_AFTER_ENGAGE_S = 0.5  # don't record while the MPC gain is still ramping in
FALL_UP_Z = 0.5            # body z-axis . world z below this = fallen
FALL_TRIM_S = 1.0          # drop this much data before a fall


def subtree_inertia(model, data, root: int) -> np.ndarray:
    com = data.subtree_com[root]
    I = np.zeros((3, 3))
    for b in range(model.nbody):
        if model.body_rootid[b] != model.body_rootid[root] or b == 0:
            continue
        R = data.ximat[b].reshape(3, 3)
        Ib = R @ np.diag(model.body_inertia[b]) @ R.T
        d = data.xipos[b] - com
        I += Ib + model.body_mass[b] * (d @ d * np.eye(3) - np.outer(d, d))
    return I


def resolve_joints(model):
    """-> (actuator ids, joint ids, joint names) in policy order."""
    act_ids, jids, names = [], [], []
    for leg in LEG_NAMES:
        for j in JOINTS:
            act_name = f"{leg}_{j}_act"
            a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            if a == -1:
                raise KeyError(f"Actuator '{act_name}' not found in model")
            assert model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT, f"{act_name} is not a joint actuator"
            jid = int(model.actuator_trnid[a, 0])
            act_ids.append(a)
            jids.append(jid)
            names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or act_name)
    assert len(act_ids) == POLICY_ACT_DIM
    return act_ids, jids, names


class Sim:
    """Everything main.py builds at module level, in one object."""

    def __init__(self):
        build_sim_xml(SRC, DST, START_HEIGHT, KP, FORCE)
        self.model = model = mujoco.MjModel.from_xml_path(DST)
        self.data = data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.body_id = body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")

        cfg = MPCConfig(
            horizon=HORIZON, dt=MPC_DT,
            mass=float(model.body_subtreemass[body_id]),
            I_body=subtree_inertia(model, data, body_id),
            R_weight=1e-3, mu=0.4,
            Q_diag=np.array([10, 10, 20, 1, 1, 50, 1, 1, 1, 1, 1, 1, 0.0]),
        )
        self.controller = LegController(model, data)
        self.controller.init()
        self.stabilizer = Stabilizer(Config(K_p=0.0, K_i=0.0, K_d=0.0), model, data, "base")
        self.core = Core(MPC(cfg), self.controller, z_des_m=0.0, solve_period_s=MPC_DT,
                         gain_mm_per_n=MPC_GAIN, gravity_z_mps2=float(model.opt.gravity[2]), ramp_s=0.1)

        jnt = model.body_jntadr[body_id]
        qadr = model.jnt_qposadr[jnt]
        self.core.reset(data.qpos[qadr:qadr + 3].copy(), np.zeros(3), make_imu(model, data, body_id))

        self.act_ids, self.jids, self.joint_names = resolve_joints(model)
        self.q_adr = np.array([model.jnt_qposadr[j] for j in self.jids])
        self.v_adr = np.array([model.jnt_dofadr[j] for j in self.jids])


# ----------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------
def command_from_input(x: float, y: float, v_max_mm_s: float):
    """Same stick -> (v m/s, yaw rate rad/s) mapping as main.py."""
    V = y * v_max_mm_s * 1e-3
    R = smoothing(-x)
    if abs(y) < 1e-3 and abs(x) > 1e-3:
        w = v_max_mm_s * 1e-3 / R
    elif abs(x) >= 1e-3:
        w = V / R
    else:
        w = 0.0
    return V, w


class RandomCommands:
    """Behavior-based random sticks (x = steer, y = forward/back, both in [-1, 1]).

    Each behavior is held for several seconds before the next one is drawn:
        STAND      x = 0, y = 0
        WALK       straight, forward (70%) or backward (30%), speed 0.3..1
        WALK_TURN  as WALK, plus steering whose amount drifts linearly during the segment
        TURN       y = 0, x = +-(0.3..1): turn in place
    Between forward and backward walking a short STAND pause is inserted so the direction never
    flips instantly.
    """
    STAND, WALK, WALK_TURN, TURN = range(4)
    NAMES = ("stand", "walk", "walk+turn", "turn in place")
    WEIGHTS = (0.15, 0.30, 0.35, 0.20)
    DURATION_S = {STAND: (2.0, 4.0), WALK: (4.0, 8.0), WALK_TURN: (5.0, 10.0), TURN: (3.0, 6.0)}
    PAUSE_S = (1.0, 1.5)

    def __init__(self, seed: int, first_stand_s: float = REST_S + 1.0):
        self.rng = np.random.default_rng(seed)
        self.b = self.STAND
        self.t_start, self.t_end = 0.0, first_stand_s
        self.x0 = self.x1 = self.y = 0.0
        self.last_dir = 0          # +1 / -1 while walking, 0 after a stand or turn
        self.pending = None        # (behavior, direction) waiting behind a pause

    def _next_segment(self, t0: float) -> None:
        r = self.rng
        pause = False
        if self.pending is not None:
            b, d = self.pending
            self.pending = None
        else:
            b = int(r.choice(4, p=self.WEIGHTS))
            if b == self.STAND and self.b == self.STAND:
                b = self.WALK                                   # no back-to-back stands
            d = (-1 if r.random() < 0.3 else 1) if b in (self.WALK, self.WALK_TURN) else 0
            if d != 0 and self.last_dir != 0 and d != self.last_dir:
                self.pending = (b, d)                           # reversing: stop first
                b, d, pause = self.STAND, 0, True

        if b == self.STAND:
            self.y, self.x0, self.x1 = 0.0, 0.0, 0.0
        elif b == self.WALK:
            self.y, self.x0, self.x1 = d * r.uniform(0.3, 1.0), 0.0, 0.0
        elif b == self.WALK_TURN:
            s = r.choice([-1, 1])
            self.y = d * r.uniform(0.3, 1.0)
            self.x0, self.x1 = s * r.uniform(0.15, 1.0), s * r.uniform(0.15, 1.0)
        else:  # TURN
            self.y = 0.0
            self.x0 = self.x1 = r.choice([-1, 1]) * r.uniform(0.3, 1.0)

        lo, hi = self.PAUSE_S if pause else self.DURATION_S[b]
        self.b, self.last_dir = b, d
        self.t_start, self.t_end = t0, t0 + r.uniform(lo, hi)

    def get(self, t: float):
        while t >= self.t_end:
            self._next_segment(self.t_end)
        a = (t - self.t_start) / max(self.t_end - self.t_start, 1e-9)
        return self.x0 + (self.x1 - self.x0) * min(max(a, 0.0), 1.0), self.y


class _NoViewer:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def is_running(self): return True
    def sync(self): pass


# ----------------------------------------------------------------------------
# Observation building
# ----------------------------------------------------------------------------
def build_frame(sim: Sim, imu, cmd, q_default, prev_action) -> np.ndarray:
    d = sim.data
    R = d.xmat[sim.body_id].reshape(3, 3)               # body -> world
    g_body = R.T @ np.array([0.0, 0.0, -1.0])           # gravity direction in the body frame
    q = d.qpos[sim.q_adr]
    qd = d.qvel[sim.v_adr]
    frame = np.concatenate([
        np.asarray(imu.gyro_rad_s) * GYRO_SCALE,
        g_body,
        [cmd[0] * CMD_SCALE_LIN, cmd[1] * CMD_SCALE_LIN, cmd[2] * CMD_SCALE_ANG],
        q - q_default,
        qd * QD_SCALE,
        prev_action,
    ])
    assert frame.shape[0] == OBS_FRAME_DIM
    return frame


class Recorder:
    def __init__(self):
        self.hist: deque = deque(maxlen=OBS_HISTORY)
        self.obs, self.act, self.q_tgt, self.cmd = [], [], [], []
        self.forces, self.ok, self.t, self.ep = [], [], [], []

    def stacked(self, frame: np.ndarray) -> np.ndarray:
        if not self.hist:
            self.hist.extend([frame] * OBS_HISTORY)      # pad the start with the first frame
        else:
            self.hist.append(frame)
        return np.concatenate(list(self.hist))           # oldest first

    def add(self, obs, act, q_tgt, cmd, forces, ok, t, ep):
        self.obs.append(obs); self.act.append(act); self.q_tgt.append(q_tgt)
        self.cmd.append(cmd); self.forces.append(forces); self.ok.append(ok)
        self.t.append(t); self.ep.append(ep)

    def trim_after(self, t_cut: float):
        keep = int(np.searchsorted(np.asarray(self.t), t_cut))
        for name in ("obs", "act", "q_tgt", "cmd", "forces", "ok", "t", "ep"):
            setattr(self, name, getattr(self, name)[:keep])


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def run(args) -> None:
    sim = Sim()
    model, data, core, controller = sim.model, sim.data, sim.core, sim.controller
    dt = model.opt.timestep
    decim = max(1, int(round(1.0 / (args.policy_hz * dt))))
    policy_hz = 1.0 / (decim * dt)

    if args.random or args.headless:
        cmds, input_controller = RandomCommands(args.seed), None
    else:
        from input.input import InputController
        input_controller = InputController()
        cmds = None

    rec = Recorder()
    q_default = None
    prev_action = np.zeros(POLICY_ACT_DIM)
    mpc_active, t_engaged, rec_tick, fell = False, 0.0, 0, False
    t_fall = None
    t_start = time.time()

    print(f"policy rate {policy_hz:.1f} Hz (every {decim} sim steps of {dt*1e3:.2f} ms)")
    viewer_ctx = _NoViewer() if args.headless else mujoco.viewer.launch_passive(model, data)

    try:
        with viewer_ctx as viewer:
            while viewer.is_running() and (args.duration <= 0 or data.time < args.duration):
                step_start = time.time()

                # ---- commands ----
                if cmds is not None:
                    x, y = cmds.get(data.time)
                else:
                    input_controller.update(dt)
                    inp = input_controller.get_input()
                    x, y = inp.x, inp.y
                if data.time >= REST_S:
                    V, W = command_from_input(x, y, controller.v_max_mm_s)
                else:
                    V, W = 0.0, 0.0
                core.v_cmd_mps, core.yaw_rate_cmd = V, W

                # ---- estimate / MPC (identical order to main.py) ----
                imu = make_imu(model, data, sim.body_id)
                s = core.estimate(dt, imu)

                if not mpc_active and data.time > SETTLE_TIME_S and np.linalg.norm(s.linear_vel) < SETTLE_VEL:
                    core.z_des = float(s.pos[2])
                    core.yaw_des = s.yaw
                    core._t_engaged = 0.0
                    mpc_active, t_engaged = True, data.time
                    print(f"MPC engaged at t={data.time:.2f}s, z_des={core.z_des:.3f} m")

                res = core.step(dt, data.time) if mpc_active else None
                pid_out = sim.stabilizer.update(dt_s=dt)

                # ---- fall check ----
                up_z = data.xmat[sim.body_id].reshape(3, 3)[2, 2]
                if mpc_active and up_z < FALL_UP_Z:
                    fell, t_fall = True, data.time
                    print(f"Robot fell at t={data.time:.2f}s (up_z={up_z:.2f}); stopping.")
                    break

                # ---- observation: sampled BEFORE this tick's command is applied ----
                record = (mpc_active and data.time >= t_engaged + SKIP_AFTER_ENGAGE_S
                          and q_default is not None and rec_tick % decim == 0)
                if record:
                    cmd = np.array([V, 0.0, W])
                    obs = rec.stacked(build_frame(sim, imu, cmd, q_default, prev_action))
                if mpc_active and data.time >= t_engaged + SKIP_AFTER_ENGAGE_S:
                    rec_tick += 1

                # ---- teacher action: servo targets the pipeline commands this tick ----
                controller.update(dt_s=dt, state=core.estimator.state, imu_data=pid_out)

                if q_default is None:
                    # first tick: legs are in HOLD at the neutral pose -> default joint targets
                    q_default = data.ctrl[sim.act_ids].copy()
                    print("joint mapping (policy idx | joint | default target rad | qpos rad):")
                    for i, n in enumerate(sim.joint_names):
                        print(f"  {i:2d} | {n} | {q_default[i]:+.3f} | {data.qpos[sim.q_adr[i]]:+.3f}")

                if record:
                    q_tgt = data.ctrl[sim.act_ids].copy()
                    action = np.clip((q_tgt - q_default) / args.action_scale, -1.0, 1.0)
                    forces = (np.asarray(res.u0.foot_ground_force).reshape(-1)
                              if res is not None else core.last_result.u0.foot_ground_force.reshape(-1)
                              if core.last_result is not None else np.zeros(12))
                    ok = bool(core.last_result.success) if core.last_result is not None else False
                    ep = int((data.time - t_engaged) // args.episode_s)
                    rec.add(obs, action, q_tgt, cmd, forces, ok, data.time, ep)
                    prev_action = action

                mujoco.mj_step(model, data)
                viewer.sync()

                if not args.headless:
                    sleep = dt - (time.time() - step_start)
                    if sleep > 0:
                        time.sleep(sleep)
    except KeyboardInterrupt:
        print("Interrupted; saving what was collected.")
    finally:
        if input_controller is not None:
            input_controller.stop()

    if fell and t_fall is not None:
        rec.trim_after(t_fall - FALL_TRIM_S)

    save(rec, q_default, sim, args, policy_hz, time.time() - t_start)


def save(rec: Recorder, q_default, sim: Sim, args, policy_hz: float, wall_s: float) -> None:
    if not rec.obs:
        print("No samples recorded; nothing written.")
        return
    out = Path(args.out) if args.out else Path(
        f"python/movement/rl/datasets/dataset_{datetime.now():%Y%m%d_%H%M%S}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)

    q_tgt = np.array(rec.q_tgt)
    meta = dict(
        obs_history=OBS_HISTORY, obs_frame_dim=OBS_FRAME_DIM, act_dim=POLICY_ACT_DIM,
        gyro_scale=GYRO_SCALE, cmd_scale_lin=CMD_SCALE_LIN, cmd_scale_ang=CMD_SCALE_ANG,
        qd_scale=QD_SCALE, action_scale_rad=args.action_scale, policy_hz=policy_hz,
        joint_names=sim.joint_names, mpc_dt=MPC_DT, horizon=HORIZON,
    )
    np.savez_compressed(
        out,
        obs=np.array(rec.obs, dtype=np.float32),
        actions=np.array(rec.act, dtype=np.float32),
        q_target=q_tgt.astype(np.float32),
        q_default=q_default.astype(np.float32),
        cmd=np.array(rec.cmd, dtype=np.float32),
        mpc_forces=np.array(rec.forces, dtype=np.float32),
        mpc_ok=np.array(rec.ok, dtype=bool),
        time=np.array(rec.t, dtype=np.float64),
        episode=np.array(rec.ep, dtype=np.int32),
        meta=np.array(json.dumps(meta)),
    )

    raw = (q_tgt - q_default) / args.action_scale
    clipped = float(np.mean(np.abs(raw) > 1.0)) * 100
    print(f"\nSaved {len(rec.obs)} samples ({len(rec.obs) / policy_hz:.1f} s of data, "
          f"{wall_s:.0f} s wall) -> {out.resolve()}")
    print(f"MPC failures: {int(np.sum(~np.array(rec.ok)))} samples")
    print(f"|action| 99th pct per joint: {np.round(np.percentile(np.abs(raw), 99, axis=0), 2)}")
    print(f"clipped by action-scale {args.action_scale} rad: {clipped:.2f}% of entries"
          + ("   <-- raise --action-scale" if clipped > 1.0 else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="output .npz path (default: python/movement/rl/datasets/dataset_<time>.npz)")
    ap.add_argument("--duration", type=float, default=0.0, help="sim seconds to run (0 = until the viewer is closed)")
    ap.add_argument("--random", action="store_true", help="random command schedule instead of live input")
    ap.add_argument("--headless", action="store_true", help="no viewer, no real-time sleep (implies --random)")
    ap.add_argument("--policy-hz", type=float, default=50.0, help="rate at which samples are recorded")
    ap.add_argument("--action-scale", type=float, default=ACTION_SCALE_RAD, help="rad per unit of normalized action")
    ap.add_argument("--episode-s", type=float, default=20.0, help="chunk length for the episode id (train/val splits)")
    ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args())