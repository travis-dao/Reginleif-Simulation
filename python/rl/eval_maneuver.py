# Scripted maneuver eval in plain CPU MuJoCo (no DR, no pushes): stand -> run -> stop -> turn -> run.
#   mjpython python/rl/eval_maneuver.py --ckpt python/rl/data/bc_actor.pt logs/rsl_rl/stage_1/5/model_500.pt --vx 0.08 --turn_rate 1.0
#
# Per scenario (turn angle x run speed) it prints:
#   surv          did the robot stay up for the whole sequence
#   stop_t (s)    time after the stop command until body speed < max(0.1 * vx, 0.01 m/s)   (nan = never within stop_s)
#   turn_deg      heading change achieved during the turn phase vs the commanded angle (error = open-loop angle error)
#   resume_t (s)  time after the second run command until forward body speed >= 0.9 * vx   (nan = never)
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.config import CFG, OBS, NUM_JOINTS, load_config
from rl.obs import build_obs, roll_history
from rl.eval import load_policy
from rl.play import Sim
import sim.config as cfg


def run_maneuver(sim, policy, vx, rate, angle_deg, warm_s, run_s, stop_s, resume_s):
    m, mp, base = sim.model, sim.mp, sim.base
    hz = OBS.policy_hz
    d = sim.fresh(sim.z_stand + 0.005)
    decim = max(1, int(round(1.0 / (hz * m.opt.timestep))))
    hist = np.zeros((OBS.history_len, NUM_JOINTS), np.float32)
    down = np.array([0.0, 0.0, -1.0])
    turn_s = np.deg2rad(abs(angle_deg)) / abs(rate)
    signed_rate = float(np.sign(angle_deg) * abs(rate))
    phases = [("warm", [0, 0, 0], warm_s), ("run1", [vx, 0, 0], run_s), ("stop", [0, 0, 0], stop_s),
              ("turn", [0, 0, signed_rate], turn_s), ("run2", [vx, 0, 0], resume_s)]
    bounds, t0 = {}, 0
    for name, c, dur in phases:
        n = int(round(dur * hz)); bounds[name] = (t0, t0 + n, np.asarray(c, np.float32)); t0 += n

    alive, yaw_acc, last_yaw = True, 0.0, None
    yaw_at = {}                       # accumulated heading at the first step of each phase
    speed, vfwd = [], []
    for step in range(t0):
        name = next(k for k, (a, b, _) in bounds.items() if a <= step < b)
        cmd = bounds[name][2]
        R = d.xmat[base].reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        if last_yaw is not None:
            yaw_acc += (yaw - last_yaw + np.pi) % (2 * np.pi) - np.pi      # unwrapped heading
        last_yaw = yaw
        if step == bounds[name][0]:
            yaw_at[name] = yaw_acc
        obs = build_obs(np, OBS, d.qvel[sim.dof + 3:sim.dof + 6], R.T @ down, cmd, d.qpos[mp.q_adr], d.qvel[mp.v_adr],
                        sim.q_default, hist)
        with torch.no_grad():
            act = np.clip(policy(torch.from_numpy(obs)[None]).cpu().numpy()[0], -1.0, 1.0).astype(np.float32)
        q_t = sim.q_default + OBS.action_scale * act
        d.ctrl[mp.aids] = np.where(mp.limited, np.clip(q_t, mp.range[:, 0], mp.range[:, 1]), q_t)
        hist = roll_history(np, hist, act)
        for _ in range(decim):
            mujoco.mj_step(m, d)
        R = d.xmat[base].reshape(3, 3)
        if R[2, 2] < cfg.FALL_UP_Z or d.qpos[sim.qa + 2] < sim.min_z:
            alive = False
            break
        v_b = R.T @ d.qvel[sim.dof:sim.dof + 3]
        speed.append(np.hypot(v_b[0], v_b[1])); vfwd.append(v_b[0])
    speed, vfwd = np.asarray(speed), np.asarray(vfwd)

    def first(sl, cond):
        a, b = sl
        idx = np.where(cond[a:min(b, len(cond))])[0]
        return float(idx[0] / hz) if len(idx) else float("nan")

    stop_t = first(bounds["stop"][:2], speed < max(0.1 * vx, 0.01)) if len(speed) >= bounds["stop"][1] else float("nan")
    turn_deg = float("nan")
    if "run2" in yaw_at:
        turn_deg = float(np.rad2deg(yaw_at["run2"] - yaw_at["turn"]))
    resume_t = first(bounds["run2"][:2], vfwd >= 0.9 * vx) if len(vfwd) > bounds["run2"][0] else float("nan")
    return alive, stop_t, turn_deg, resume_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--vx", type=float, nargs="+", default=[0.08])
    ap.add_argument("--turn_rate", type=float, default=1.0, help="rad/s during the turn phase")
    ap.add_argument("--angles", type=float, nargs="+", default=[180, -180, 15, -15, 30, -30], help="deg, + = left")
    ap.add_argument("--warm_s", type=float, default=1.0)
    ap.add_argument("--run_s", type=float, default=3.0)
    ap.add_argument("--stop_s", type=float, default=1.0)
    ap.add_argument("--resume_s", type=float, default=3.0)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    conf = load_config(args.config) if args.config else CFG
    sim = Sim(conf)
    for ck in args.ckpt:
        policy = load_policy(ck, conf, "cpu")
        print(f"\n=== {ck}   (turn rate {args.turn_rate} rad/s)")
        print(f"{'vx':>6} {'angle':>7} {'surv':>5} {'stop_t':>7} {'turn_deg':>9} {'err_deg':>8} {'resume_t':>9}")
        errs = []
        for vx in args.vx:
            for ang in args.angles:
                alive, st, td, rt = run_maneuver(sim, policy, vx, args.turn_rate, ang, args.warm_s, args.run_s, args.stop_s, args.resume_s)
                err = abs(td - ang)
                errs.append(err)
                print(f"{vx:6.3f} {ang:7.1f} {int(alive):5d} {st:7.2f} {td:9.1f} {err:8.1f} {rt:9.2f}")
        print(f"mean |angle error| {np.nanmean(errs):.1f} deg")


if __name__ == "__main__":
    main()