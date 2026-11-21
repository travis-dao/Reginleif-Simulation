# Closed-loop eval of the BC student on the MPC scenario files: does it adapt its FOOTHOLDS like the teacher?
#   python python/rl/bc_eval.py --data python/rl/data/mpc --ckpt bc_policy.pt --accels 3.0 --max 60
#
# For each scenario file (command x push accel x direction x gait phase) the student is started from the teacher's
# recorded standing state with the same command, pushed with the same force at the same step, and run closed-loop
# (no MPC).  We then compare, for every foot touchdown in the 3 s after the push:
#     teacher shift = teacher foothold (pushed)  - teacher foothold (unpushed run, same command)
#     student shift = student foothold (pushed)  - student foothold (unpushed run, same command)
# in the body frame.  Differencing against each policy's own unpushed run removes plain gait mismatch.
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import mujoco


def _find_python_dir():
    for p in Path(__file__).resolve().parents:
        if (p / "sim").is_dir() and (p / "rl").is_dir():
            return p
    raise RuntimeError("could not find the python/ directory (a folder containing sim/ and rl/)")


PYTHON_DIR = _find_python_dir()
sys.path.insert(0, str(PYTHON_DIR))

import sim.config as cfg
from rl.environments.env import Env
from rl.models.actor_critic import ActorCritic
MATCH_TOL = 15            # policy steps (0.3 s): max timing difference when pairing touchdowns across runs
WINDOW_S = 3.0            # score touchdowns in this window after the push
KEYS = ("obs", "act", "cmd", "contacts", "foot_pos_bf", "base_pos", "base_yaw")


# ------------------------------------------------------------------ data
def load(path):
    d = np.load(path)
    ep = {k: d[k] for k in KEYS}
    for k in ("push_index", "push_dur", "push_accel", "push_angle_deg", "policy_hz", "z_stand", "action_scale"):
        ep[k] = float(d[k])
    ep["push_force_bf"] = d["push_force_bf"].astype(np.float64)
    ep["fell"] = bool(d["fell"])
    ep["q_default"] = d["q_default"].astype(np.float32)
    ep["contacts"] = ep["contacts"].astype(bool)
    ep["name"] = Path(path).stem
    return ep


def load_manifest(data_dir):
    m = Path(data_dir) / "rl/data/mpc/manifest.json"
    if m.exists():
        return json.loads(m.read_text())
    out = []                                              # fallback: scan the files
    for p in sorted(Path(data_dir).glob("*.npz")):
        d = np.load(p)
        out.append(dict(file=p.name, command=str(d["command_name"]), accel=float(d["push_accel"]),
                        angle_deg=float(d["push_angle_deg"])))
    return out


# ------------------------------------------------------------------ student rollout
def foot_contacts(env):
    """Correct foot-ground contact test. (Env._get_foot_contacts has an operator-precedence bug, see notes.)"""
    c = np.zeros(4, bool)
    gid, dat = env.ground_geom_id, env.data
    for j in range(dat.ncon):
        g1, g2 = int(dat.contact[j].geom1), int(dat.contact[j].geom2)
        for i, fg in enumerate(env.foot_geom_ids):
            if (g1 == gid and g2 in fg) or (g2 == gid and g1 in fg):
                c[i] = True
    return c


def rollout(env, model, ep, push):
    """Start from the teacher's recorded standing state, run the student closed-loop for as long as the teacher ran."""
    n, hz, q_def = len(ep["obs"]), ep["policy_hz"], ep["q_default"]
    env.reset()
    d = env.data
    d.qpos[env.q_adr] = q_def + ep["obs"][0][9:21]                 # obs layout: gyro3 grav3 cmd3 q12 qd12 prev_act12
    b = env.base_qadr
    d.qpos[b:b + 3] = ep["base_pos"][0]
    yaw0 = float(ep["base_yaw"][0])
    d.qpos[b + 3:b + 7] = [np.cos(yaw0 / 2), 0.0, 0.0, np.sin(yaw0 / 2)]
    d.qvel[:] = 0.0
    d.ctrl[env.actuator_ids] = q_def + env.action_scale * ep["act"][0]
    mujoco.mj_forward(env.model, d)
    env.imu.update()
    env.command[:] = ep["cmd"]
    env.command_timer = 1e9                                         # fixed command: never resample
    env.prev_action[:] = 0.0
    obs = env.get_observation()

    z_floor = 0.6 * (ep["z_stand"] if ep["z_stand"] > 0 else float(ep["base_pos"][0][2]))   # same fall rule as the generator
    p0, n_push = int(ep["push_index"]), max(1, int(round(ep["push_dur"] * hz)))
    do_push = push and ep["push_accel"] > 0 and p0 >= 0
    F_w, h = None, None
    foot, con, pos, fell_i = [], [], [], -1

    for i in range(n):
        o = torch.as_tensor(obs, dtype=torch.float32).view(1, 1, -1)
        with torch.no_grad():
            a, _, _, h = model.actor.sample_action(o, h, deterministic=True)
        if do_push and i == p0:
            R = d.xmat[env.body_id].reshape(3, 3)
            y = np.arctan2(R[1, 0], R[0, 0])
            Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
            F_w = Rz @ ep["push_force_bf"]
        if do_push and p0 <= i < p0 + n_push:
            d.xfrc_applied[env.body_id, :3] = F_w
        else:
            d.xfrc_applied[env.body_id, :] = 0.0

        obs, *_ = env.step(a.view(-1).numpy())

        R = d.xmat[env.body_id].reshape(3, 3)
        foot.append((d.xpos[env.foot_body_ids] - d.xpos[env.body_id]) @ R)
        con.append(foot_contacts(env))
        pos.append(d.xpos[env.body_id].copy())
        if R[2, 2] < cfg.FALL_UP_Z or d.xpos[env.body_id, 2] < z_floor:
            fell_i = i
            break
    d.xfrc_applied[:] = 0.0
    return dict(foot=np.array(foot), con=np.array(con), pos=np.array(pos), fell=fell_i >= 0, fell_i=fell_i)


def teacher_run(ep):
    return dict(foot=ep["foot_pos_bf"], con=ep["contacts"], pos=ep["base_pos"], fell=ep["fell"], fell_i=-1)


# ------------------------------------------------------------------ footholds
def touchdowns(con, lo, hi):
    out = []
    for f in range(4):
        c = con[:, f]
        out.append([t for t in range(max(lo, 1), min(hi, len(c))) if c[t] and not c[t - 1]])
    return out


def key_events(td, ref):
    """Label each touchdown with the index of the nearest reference touchdown (same foot, within MATCH_TOL)."""
    keyed = {}
    for f in range(4):
        used = set()
        for t in td[f]:
            best = None
            for r, rt in enumerate(ref[f]):
                if r in used or abs(rt - t) > MATCH_TOL:
                    continue
                if best is None or abs(rt - t) < abs(ref[f][best] - t):
                    best = r
            if best is not None:
                used.add(best)
                keyed[(f, best)] = t
    return keyed


def shifts(push_run, base_run, ref, lo, hi):
    kp = key_events(touchdowns(push_run["con"], lo, hi), ref)
    kb = key_events(touchdowns(base_run["con"], lo, hi), ref)
    return {k: (push_run["foot"][kp[k], k[0], :2] - base_run["foot"][kb[k], k[0], :2]) * 1000.0   # mm, body-frame xy
            for k in kp.keys() & kb.keys()}


def pooled(dT, dS):
    if len(dT) == 0:
        return None
    dT, dS = np.asarray(dT), np.asarray(dS)
    tt, ss, ts = (dT * dT).sum(), (dS * dS).sum(), (dT * dS).sum()
    return dict(n=len(dT), mag_T=np.linalg.norm(dT, axis=1).mean(), mag_S=np.linalg.norm(dS, axis=1).mean(),
                slope=ts / tt if tt > 0 else np.nan, cos=ts / np.sqrt(tt * ss) if tt > 0 and ss > 0 else np.nan)


def drift(run, base):
    L = min(len(run["pos"]), len(base["pos"]))
    return float(np.linalg.norm(run["pos"][L - 1, :2] - base["pos"][L - 1, :2]))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="python/rl/data/mpc")
    ap.add_argument("--ckpt", default="python/rl/policy/bc_policy.pt")
    ap.add_argument("--accels", type=float, nargs="+", default=[3.0], help="push strengths to evaluate (use ones the student never trained on)")
    ap.add_argument("--commands", nargs="+", default=None, help="restrict to these command names")
    ap.add_argument("--max", type=int, default=60, help="evaluate at most this many pushed scenarios (evenly spread)")
    ap.add_argument("--out", default="python/rl/bc_eval_results.json")
    args = ap.parse_args()

    man = load_manifest(args.data)
    base_entries = {m["command"]: m for m in man if m["accel"] == 0}
    cand = [m for m in man if m["accel"] in args.accels and m["command"] in base_entries
            and (args.commands is None or m["command"] in args.commands)]
    if not cand:
        sys.exit("no scenarios match --accels/--commands (and have an unpushed baseline file for the same command)")
    if len(cand) > args.max:
        cand = [cand[i] for i in np.linspace(0, len(cand) - 1, args.max).astype(int)]
    print(f"{len(cand)} pushed scenarios, {len({m['command'] for m in cand})} commands")

    first = load(Path(args.data) / cand[0]["file"])
    env = Env(q_default=first["q_default"], action_scale_rad=first["action_scale"],
              foot_body_names=cfg.FOOT_BODIES, ground_geom_name=cfg.GROUND_GEOM)
    model = ActorCritic(q_default=first["q_default"], action_scale=first["action_scale"])
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()

    base_cache = {}                                                     # command -> (teacher baseline ep, student baseline run)
    def baseline(cmd):
        if cmd not in base_cache:
            ep_b = load(Path(args.data) / base_entries[cmd]["file"])
            base_cache[cmd] = (ep_b, rollout(env, model, ep_b, push=False))
        return base_cache[cmd]

    rows, all_T, all_S, by_angle = [], [], [], {}
    t0 = time.time()
    for k, m in enumerate(cand):
        ep = load(Path(args.data) / m["file"])
        ep_b, run_sb = baseline(m["command"])
        run_t, run_tb = teacher_run(ep), teacher_run(ep_b)
        run_s = rollout(env, model, ep, push=True)

        p0 = int(ep["push_index"])
        hi = p0 + int(WINDOW_S * ep["policy_hz"])
        ref = touchdowns(run_tb["con"], p0, hi)
        sT, sS = shifts(run_t, run_tb, ref, p0, hi), shifts(run_s, run_sb, ref, p0, hi)
        keys = sorted(sT.keys() & sS.keys())
        dT = [sT[q] for q in keys]; dS = [sS[q] for q in keys]
        all_T += dT; all_S += dS
        by_angle.setdefault(ep["push_angle_deg"], ([], []))
        by_angle[ep["push_angle_deg"]][0].extend(dT); by_angle[ep["push_angle_deg"]][1].extend(dS)

        ok = not run_s["fell"] and not run_t["fell"]
        rows.append(dict(name=ep["name"], command=m["command"], accel=ep["push_accel"], angle=ep["push_angle_deg"],
                         teacher_fell=run_t["fell"], student_fell=run_s["fell"], student_fell_step=run_s["fell_i"],
                         events=len(keys), drift_T=drift(run_t, run_tb) if ok else None,
                         drift_S=drift(run_s, run_sb) if ok else None, **(pooled(dT, dS) or {})))
        print(f"[{k + 1}/{len(cand)}] {ep['name']}: student {'FELL' if run_s['fell'] else 'ok'}, teacher {'FELL' if run_t['fell'] else 'ok'}, "
              f"{len(keys)} matched touchdowns  ({time.time() - t0:.0f}s)")

    # ---------------- report
    print("\n================ RESULTS ================")
    for acc in sorted({r["accel"] for r in rows}):
        rr = [r for r in rows if r["accel"] == acc]
        print(f"push {acc:.1f} m/s^2: student fell {sum(r['student_fell'] for r in rr)}/{len(rr)}, "
              f"teacher fell {sum(r['teacher_fell'] for r in rr)}/{len(rr)}")
    g = pooled(all_T, all_S)
    if g is None:
        print("\nno matched touchdowns after pushes: nothing to score (did the teacher step? is the gait running?)")
    else:
        print(f"\nfoothold shift after a push, pooled over {g['n']} touchdowns (body frame, xy):")
        print(f"  teacher mean |shift| {g['mag_T']:.2f} mm   student mean |shift| {g['mag_S']:.2f} mm")
        print(f"  slope  (student on teacher, 1.0 = same size & direction, 0 = no adaptation): {g['slope']:.2f}")
        print(f"  cosine (direction agreement, 1.0 = same direction): {g['cos']:.2f}")
        print("\nby push direction (0 = forward, 90 = left, body frame):")
        print("  angle |    n | teacher mm | student mm | slope | cosine")
        for ang in sorted(by_angle):
            s = pooled(*by_angle[ang])
            if s:
                print(f"  {ang:5.0f} | {s['n']:4d} | {s['mag_T']:10.2f} | {s['mag_S']:10.2f} | {s['slope']:5.2f} | {s['cos']:6.2f}")
        if g["mag_T"] < 1.0:
            print("\nNOTE: the teacher's own foothold shift is under 1 mm, so there is almost no adaptation signal to clone at this push size.")
    dT_, dS_ = [r["drift_T"] for r in rows if r["drift_T"] is not None], [r["drift_S"] for r in rows if r["drift_S"] is not None]
    if dT_:
        print(f"\nbase xy displacement vs unpushed run at episode end: teacher {np.mean(dT_) * 1000:.0f} mm, student {np.mean(dS_) * 1000:.0f} mm (episodes where neither fell)")
    Path(args.out).write_text(json.dumps(rows, indent=1, default=float))
    print(f"\nper-scenario results -> {args.out}")


if __name__ == "__main__":
    main()