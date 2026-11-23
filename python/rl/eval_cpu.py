# Run the STUDENT policy in plain CPU MuJoCo, the same plant the MPC teacher ran in (no MPC, no MJX), and compare its
# yaw/lin tracking with the MPC's. Separates "MJX physics gap" from "imitation error":
#   yaw std on CPU ~ MPC  -> the MJX env is the cause (policy is fine)
#   yaw std on CPU ~ 2x   -> the policy itself wobbles (imitation error)
#
#   mjpython python/rl/eval_cpu.py --ckpt python/rl/data/bc_actor.pt --manifest python/rl/data/mpc/manifest.json
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.config import CFG, OBS, NUM_JOINTS, Q_DEFAULT_PATH, load_config
from rl.obs import build_obs, roll_history
from rl.eval import load_policy, pick_cmds, teacher_stats
from sim.util.mujoco_robot_map import MuJuCoMap
import sim.config as cfg
from sim.util.build_sim_xml import build_sim_xml


def run(model, mp, policy, q_default, cmd, seconds, z_stand, qa, dof, base, min_z):
    """Same start as the MJX env (standing pose, q_default targets, zero velocity) and same obs/action definitions."""
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[mp.q_adr] = q_default
    data.qpos[qa + 2] = z_stand + 0.005
    mujoco.mj_forward(model, data)

    decim = max(1, int(round(1.0 / (OBS.policy_hz * model.opt.timestep))))
    hist = np.zeros((OBS.history_len, NUM_JOINTS), np.float32)
    cmd = np.asarray(cmd, np.float32)
    lin, ys, alive = [], [], True
    down = np.array([0.0, 0.0, -1.0])
    for _ in range(int(seconds * OBS.policy_hz)):
        R = data.xmat[base].reshape(3, 3)
        obs = build_obs(np, OBS, data.qvel[dof + 3:dof + 6], R.T @ down, cmd,
                        data.qpos[mp.q_adr], data.qvel[mp.v_adr], q_default, hist)
        with torch.no_grad():
            act = policy(torch.from_numpy(obs)[None]).cpu().numpy()[0]
        act = np.clip(act, -1.0, 1.0).astype(np.float32)
        q_t = q_default + OBS.action_scale * act
        q_t = np.where(mp.limited, np.clip(q_t, mp.range[:, 0], mp.range[:, 1]), q_t)
        data.ctrl[mp.aids] = q_t
        hist = roll_history(np, hist, act)
        for _ in range(decim):
            mujoco.mj_step(model, data)
        R = data.xmat[base].reshape(3, 3)
        if R[2, 2] < cfg.FALL_UP_Z or data.qpos[qa + 2] < min_z:
            alive = False
            break
        v_b = R.T @ data.qvel[dof:dof + 3]
        lin.append(np.hypot(v_b[0] - cmd[0], v_b[1] - cmd[1]))
        ys.append(data.qvel[dof + 5] - cmd[2])
    ys = np.asarray(ys)
    return alive, float(np.mean(lin)), float(np.abs(ys).mean()), float(ys.mean()), float(ys.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True, help="MPC test manifest.json (supplies the commands + teacher numbers)")
    ap.add_argument("--n_cmds", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    conf = load_config(args.config) if args.config else CFG
    policy = load_policy(args.ckpt, conf, "cpu")

    build_sim_xml(cfg.SRC, cfg.DST, cfg.START_HEIGHT, cfg.KP, cfg.FORCE)
    model = mujoco.MjModel.from_xml_path(cfg.DST)
    mp = MuJuCoMap(model)
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    qa, dof = int(model.jnt_qposadr[model.body_jntadr[base]]), int(model.jnt_dofadr[model.body_jntadr[base]])
    q_default = np.load(Q_DEFAULT_PATH).astype(np.float32)

    # measure the standing height exactly like QuadrupedEnv does
    d = mujoco.MjData(model)
    d.qpos[:] = model.qpos0; d.qpos[mp.q_adr] = q_default; d.qpos[qa + 2] = cfg.START_HEIGHT
    d.ctrl[mp.aids] = q_default
    for _ in range(int(1.0 / model.opt.timestep)):
        mujoco.mj_step(model, d)
    z_stand = float(d.qpos[qa + 2]); min_z = conf["env"]["min_height_frac"] * z_stand
    print(f"standing height {z_stand:.3f} m, terminate below {min_z:.3f} m")

    teacher = [t for t in teacher_stats(args.manifest) if t["accel"] == 0.0]
    cmds = pick_cmds(sorted({t["cmd"] for t in teacher}), args.n_cmds)

    print(f"\n{'cmd':>26} {'surv':>5} {'lin cpu':>8} {'lin mpc':>8} {'yaw cpu':>8} {'yaw mpc':>8} {'std cpu':>8} {'std mpc':>8} {'bias cpu':>9} {'bias mpc':>9}")
    ratios = []
    for c in cmds:
        alive, lin, yaw, bias, std = run(model, mp, policy, q_default, c, args.seconds, z_stand, qa, dof, base, min_z)
        t = next(x for x in teacher if x["cmd"] == c)
        ratios.append(std / max(t["yaw_std"], 1e-9))
        print(f"{str(list(c)):>26} {int(alive):5d} {lin:8.4f} {t['lin']:8.4f} {yaw:8.4f} {t['yaw']:8.4f} "
              f"{std:8.4f} {t['yaw_std']:8.4f} {bias:9.4f} {t['yaw_bias']:9.4f}")
    r = float(np.mean(ratios))
    print(f"\nmean yaw-std ratio (CPU student / MPC): {r:.2f}   (MJX student / MPC was ~2.0)")
    print("-> ~1.0: the MJX env explains the wobble.  ~2.0: the policy itself wobbles (imitation error).")


if __name__ == "__main__":
    main()