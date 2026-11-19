#!/usr/bin/env python3
"""Closed-loop evaluation of the trained policy in MuJoCo (the real test of the behavior clone).

Put this next to collect_data.py (python/sim/) and run:

    python python/sim/eval_policy.py                       # latest run in python/movement/rl/runs
    python python/sim/eval_policy.py --policy path/to/policy.pt --episodes 10 --duration 90
    mjpython python/sim/eval_policy.py --viewer            # watch the first policy episode in real time

For each episode (fresh sim, same seeded command schedule for both modes):
    policy  : obs (built exactly like collect_data.py) -> MLP -> q = q_default + scale * action -> data.ctrl
              The MPC, gait and state estimator are NOT used; only IMU + joint encoders feed the policy.
    teacher : the original MPC + gait pipeline, as the baseline to compare against.

Timeline per episode: hold the default pose for --engage-s (robot lands and settles), then the policy
takes over with an empty history, stands for --stand-s, and the random command schedule runs
(stand / walk / walk+turn / turn in place, the same generator used for data collection).

Reported (policy vs teacher): falls, survival time, forward-velocity and yaw-rate tracking (error and
least-squares gain), standing drift, body tilt, height variation and action jitter. Compare the policy
to the teacher, not to the command: the gait itself ramps and saturates the commanded twist.
"""

import argparse
import importlib.util
import json
import math
import sys
import time
import warnings
from collections import deque
from datetime import datetime
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PYTHON_DIR = HERE.parent
sys.path.insert(0, str(PYTHON_DIR))
sys.path.insert(0, str(HERE))

import collect_data as cd                                   # Sim, RandomCommands, build_frame, ...
from movement.leg.leg import clamp_to_limits
from movement.leg.leg_config import Theta3
from movement.motion.state_estimator import make_imu

RUNS_DIR = PYTHON_DIR / "movement" / "rl" / "runs"
TRAIN_PY = PYTHON_DIR / "movement" / "rl" / "train.py"


# ----------------------------------------------------------------------------
# Policy loading
# ----------------------------------------------------------------------------
def find_latest_policy() -> Path:
    cands = sorted(RUNS_DIR.glob("*/policy.pt"))
    if not cands:
        raise SystemExit(f"No policy.pt found under {RUNS_DIR}; pass --policy.")
    return cands[-1]


def load_policy(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)   # ckpt holds numpy + dicts
    spec = importlib.util.spec_from_file_location("bc_train", TRAIN_PY)
    bc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bc)                                       # reuse the exact Policy class

    sd = ckpt["state_dict"]
    model = bc.Policy(sd["mean"].numpy(), sd["std"].numpy(), ckpt["clip"], ckpt["act"])
    model.load_state_dict(sd)
    model.eval()
    meta = ckpt["meta"]

    # build_frame() uses collect_data's scale constants: they must equal the ones the data was made with
    for key, const in [("gyro_scale", cd.GYRO_SCALE), ("cmd_scale_lin", cd.CMD_SCALE_LIN),
                       ("cmd_scale_ang", cd.CMD_SCALE_ANG), ("qd_scale", cd.QD_SCALE)]:
        if not math.isclose(meta[key], const, rel_tol=1e-6):
            raise SystemExit(f"Scale mismatch: checkpoint {key}={meta[key]} but collect_data.py has {const}. "
                             "Set the constant in collect_data.py to the checkpoint value.")
    return model, meta, np.asarray(ckpt["q_default"], dtype=np.float64)


def clamp_servo(q: np.ndarray) -> np.ndarray:
    """Clamp joint targets [rad] to the calibrated servo limits (what the real leg code does)."""
    out = q.copy()
    for leg in range(4):
        a = Theta3(coxa=math.degrees(q[3 * leg]), femur=math.degrees(q[3 * leg + 1]),
                   tibia=math.degrees(q[3 * leg + 2]))
        c = clamp_to_limits(a, leg)
        out[3 * leg:3 * leg + 3] = np.radians([c.coxa, c.femur, c.tibia])
    return out


# ----------------------------------------------------------------------------
# One episode
# ----------------------------------------------------------------------------
def run_episode(args, mode: str, seed: int, bundle, show: bool = False) -> dict:
    net, meta, q_default = bundle
    scale = meta["action_scale_rad"]

    sim = cd.Sim()
    m, d, core = sim.model, sim.data, sim.core
    if list(sim.joint_names) != list(meta["joint_names"]):
        raise SystemExit("Joint order in the sim differs from the checkpoint's meta['joint_names'].")
    dt = m.opt.timestep
    decim = max(1, int(round(1.0 / (meta["policy_hz"] * dt))))
    v_max = sim.controller.v_max_mm_s
    v_adr0 = int(m.jnt_dofadr[m.body_jntadr[sim.body_id]])

    t_cmd0 = args.engage_s
    cmds = cd.RandomCommands(seed, first_stand_s=args.stand_s)
    total_steps = int(round((args.engage_s + args.duration) / dt))

    hist: deque = deque(maxlen=cd.OBS_HISTORY)
    prev_action = np.zeros(cd.POLICY_ACT_DIM)
    q_goal = q_default.copy()
    mpc_active = False
    ltick, fell, t_fall = 0, False, None
    L = {k: [] for k in ("V", "W", "vx", "vy", "wz", "roll", "pitch", "z", "act")}

    viewer_ctx = mujoco.viewer.launch_passive(m, d) if show else cd._NoViewer()
    with viewer_ctx as viewer:
        for _ in range(total_steps):
            if not viewer.is_running():
                break
            wall0 = time.time()
            t = d.time

            if t >= t_cmd0:
                x, y = cmds.get(t - t_cmd0)
                V, W = cd.command_from_input(x, y, v_max)
            else:
                V, W = 0.0, 0.0

            imu = make_imu(m, d, sim.body_id)
            R = d.xmat[sim.body_id].reshape(3, 3)
            if R[2, 2] < cd.FALL_UP_Z:
                fell, t_fall = True, t
                break

            record = t >= t_cmd0 and ltick % decim == 0
            act = None

            if mode == "policy":
                if t < t_cmd0:
                    d.ctrl[sim.act_ids] = q_default                 # land and settle in the default pose
                else:
                    if ltick % decim == 0:                           # policy tick
                        frame = cd.build_frame(sim, imu, np.array([V, 0.0, W]), q_default, prev_action)
                        if not hist:
                            hist.extend([frame] * cd.OBS_HISTORY)    # empty history -> repeat first frame
                        else:
                            hist.append(frame)
                        obs = np.concatenate(list(hist)).astype(np.float32)
                        with torch.no_grad():
                            act = net(torch.from_numpy(obs)[None])[0].numpy().astype(np.float64)
                        act = np.clip(act, -1.0, 1.0)
                        prev_action = act                            # un-ramped, as in the dataset
                        q_goal = q_default + scale * act
                        if not args.no_clamp:
                            q_goal = clamp_servo(q_goal)
                    cur = d.ctrl[sim.act_ids].copy()
                    if args.slew > 0:                                # optional output slew limiter
                        step = args.slew * dt
                        cur = cur + np.clip(q_goal - cur, -step, step)
                    else:
                        cur = q_goal
                    d.ctrl[sim.act_ids] = cur
            else:                                                    # teacher: same order as collect_data.py
                core.v_cmd_mps, core.yaw_rate_cmd = V, W
                s = core.estimate(dt, imu)
                if not mpc_active and t > cd.SETTLE_TIME_S and np.linalg.norm(s.linear_vel) < cd.SETTLE_VEL:
                    core.z_des, core.yaw_des, core._t_engaged = float(s.pos[2]), s.yaw, 0.0
                    mpc_active = True
                if mpc_active:
                    core.step(dt, t)
                pid_out = sim.imu.update(dt_s=dt)
                sim.controller.update(dt_s=dt, state=core.estimator.state, imu_data=pid_out)
                if record:
                    act = np.clip((d.ctrl[sim.act_ids] - q_default) / scale, -1.0, 1.0)

            if record:
                vb = R.T @ d.qvel[v_adr0:v_adr0 + 3]                 # body-frame linear velocity
                L["V"].append(V); L["W"].append(W)
                L["vx"].append(vb[0]); L["vy"].append(vb[1])
                L["wz"].append((R @ np.asarray(imu.gyro_rad_s))[2])  # yaw rate about world z
                L["roll"].append(imu.rpy_rad[0]); L["pitch"].append(imu.rpy_rad[1])
                L["z"].append(d.xpos[sim.body_id][2])
                L["act"].append(act)
            if t >= t_cmd0:
                ltick += 1

            mujoco.mj_step(m, d)
            if show:
                viewer.sync()
                sleep = dt - (time.time() - wall0)
                if sleep > 0:
                    time.sleep(sleep)

    end_t = t_fall if fell else d.time
    return summarize(L, max(end_t - t_cmd0, 0.0), fell)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def summarize(L: dict, survived_s: float, fell: bool) -> dict:
    nan = float("nan")
    r = dict(fell=float(fell), survived_s=survived_s, vx_mae=nan, vx_gain=nan, wz_mae=nan, wz_gain=nan,
             stand_speed=nan, stand_wz=nan, roll_rms=nan, pitch_rms=nan, z_std=nan, act_jitter=nan, n=len(L["V"]))
    if len(L["V"]) < 5:
        return r
    V, W, vx, vy, wz = (np.asarray(L[k]) for k in ("V", "W", "vx", "vy", "wz"))
    walk, turn = np.abs(V) > 1e-3, np.abs(W) > 1e-3
    stand = ~walk & ~turn

    def mae(a, b, mk):
        return float(np.mean(np.abs(a[mk] - b[mk]))) if mk.any() else nan

    def gain(a, b, mk):                                              # least-squares slope of achieved vs command
        den = float(np.sum(b[mk] ** 2))
        return float(np.sum(a[mk] * b[mk]) / den) if mk.any() and den > 0 else nan

    r["vx_mae"] = mae(vx, V, walk) * 1e3                             # mm/s
    r["vx_gain"] = gain(vx, V, walk)
    r["wz_mae"] = mae(wz, W, turn)                                   # rad/s
    r["wz_gain"] = gain(wz, W, turn)
    if stand.any():
        r["stand_speed"] = float(np.mean(np.hypot(vx, vy)[stand])) * 1e3   # mm/s
        r["stand_wz"] = float(np.mean(np.abs(wz[stand])))
    r["roll_rms"] = float(np.degrees(np.sqrt(np.mean(np.square(L["roll"])))))
    r["pitch_rms"] = float(np.degrees(np.sqrt(np.mean(np.square(L["pitch"])))))
    r["z_std"] = float(np.std(L["z"])) * 1e3                         # mm
    A = np.asarray(L["act"])
    r["act_jitter"] = float(np.mean(np.abs(np.diff(A, axis=0))))     # per policy tick, normalized units
    return r


ROWS = [  # key, label, format
    ("fell", "falls (fraction of episodes)", "{:.2f}"),
    ("survived_s", "survival time [s]", "{:.1f}"),
    ("vx_mae", "forward speed MAE, walking [mm/s]", "{:.1f}"),
    ("vx_gain", "forward speed gain (1 = tracks cmd)", "{:.2f}"),
    ("wz_mae", "yaw rate MAE, turning [rad/s]", "{:.3f}"),
    ("wz_gain", "yaw rate gain (1 = tracks cmd)", "{:.2f}"),
    ("stand_speed", "drift speed when standing [mm/s]", "{:.1f}"),
    ("stand_wz", "yaw drift when standing [rad/s]", "{:.3f}"),
    ("roll_rms", "roll RMS [deg]", "{:.2f}"),
    ("pitch_rms", "pitch RMS [deg]", "{:.2f}"),
    ("z_std", "height std [mm]", "{:.2f}"),
    ("act_jitter", "action jitter (mean |da| per tick)", "{:.4f}"),
]


def aggregate(results: list[dict]) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return {k: float(np.nanmean([r[k] for r in results])) for k, _, _ in ROWS}


def print_table(agg: dict[str, dict]) -> None:
    names = list(agg)
    print("\n" + f"{'metric':42s}" + "".join(f"{n:>12s}" for n in names))
    for key, label, fmt in ROWS:
        print(f"{label:42s}" + "".join(f"{fmt.format(agg[n][key]):>12s}" for n in names))


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", type=Path, default=None, help="path to policy.pt (default: latest run)")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--duration", type=float, default=60.0, help="seconds per episode AFTER engage")
    ap.add_argument("--engage-s", type=float, default=cd.REST_S, help="hold default pose this long, then start")
    ap.add_argument("--stand-s", type=float, default=1.0, help="standing time before the first command")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--slew", type=float, default=0.0, help="optional ctrl slew limit [rad/s] (0 = off)")
    ap.add_argument("--no-clamp", action="store_true", help="do not clamp targets to servo limits")
    ap.add_argument("--no-teacher", action="store_true", help="skip the MPC baseline")
    ap.add_argument("--viewer", action="store_true", help="show the first policy episode in real time")
    args = ap.parse_args()

    path = args.policy or find_latest_policy()
    print(f"policy: {path}")
    bundle = load_policy(path)
    print(f"policy rate {bundle[1]['policy_hz']:.1f} Hz, action scale {bundle[1]['action_scale_rad']} rad, "
          f"{args.episodes} episodes x {args.duration:.0f} s")

    modes = ["policy"] + ([] if args.no_teacher else ["teacher"])
    per_mode: dict[str, list[dict]] = {m: [] for m in modes}
    for i in range(args.episodes):
        seed = args.seed + i
        for mode in modes:
            t0 = time.time()
            r = run_episode(args, mode, seed, bundle, show=(args.viewer and i == 0 and mode == "policy"))
            per_mode[mode].append(r)
            status = f"FELL after {r['survived_s']:.1f}s" if r["fell"] else "ok"
            print(f"  ep {i} seed {seed} {mode:8s}: {status:18s} vx MAE {r['vx_mae']:6.1f} mm/s  "
                  f"wz MAE {r['wz_mae']:.3f} rad/s  ({time.time() - t0:.0f}s wall)")

    agg = {m: aggregate(per_mode[m]) for m in modes}
    print_table(agg)

    out = path.parent / f"eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({"policy": str(path), "args": {k: str(v) for k, v in vars(args).items()},
                               "episodes": per_mode, "mean": agg}, indent=2))
    print(f"\nsaved -> {out.resolve()}")


if __name__ == "__main__":
    main()