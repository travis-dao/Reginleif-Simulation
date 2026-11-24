# Watch a policy in CPU MuJoCo and print what the robot actually does with a given command.
#
#   mjpython python/rl/play.py --ckpt python/rl/data/bc_actor.pt logs/rsl_rl/<exp>/<run>/model_99.pt \
#            --cmd 0.08333 0 0 --view
#   mjpython python/rl/play.py --ckpt python/rl/data/bc_actor.pt logs/rsl_rl/stage_1/model_100.pt --cmd 0.08333 0 0 --view
#
# --view opens a viewer per checkpoint, in order (close the window to continue; needs mjpython on macOS).
# Without --view it runs headless and only prints the numbers, so BC and PPO can be compared side by side.
#
# Printed per checkpoint:
#   translation / turning as a fraction of the commanded value (from the measured body-frame velocity)
#   foot touchdowns per second  (0 -> feet never lift;  >0 but no translation -> stepping in place)
#   joint travel (rad/s of total joint motion), mean base height
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.config import CFG, OBS, NUM_JOINTS, Q_DEFAULT_PATH, load_config
from rl.obs import build_obs, roll_history
from rl.eval import load_policy
from sim.util.mujoco_robot_map import MuJuCoMap
import sim.config as cfg
from sim.util.build_sim_xml import build_sim_xml


class Sim:
    def __init__(self, conf):
        build_sim_xml(cfg.SRC, cfg.DST, cfg.START_HEIGHT, cfg.KP, cfg.FORCE)
        self.model = m = mujoco.MjModel.from_xml_path(cfg.DST)
        self.mp = mp = MuJuCoMap(m)
        self.base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
        j = m.body_jntadr[self.base]
        self.qa, self.dof = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
        self.q_default = np.load(Q_DEFAULT_PATH).astype(np.float32)
        self.foot_geoms = [np.where((m.geom_bodyid == b) & (m.geom_contype > 0))[0] for b in mp.foot_bodies]
        d = self.fresh(cfg.START_HEIGHT)                       # measure the standing height like QuadrupedEnv does
        d.ctrl[mp.aids] = self.q_default
        for _ in range(int(1.0 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        self.z_stand = float(d.qpos[self.qa + 2])
        self.min_z = conf["env"]["min_height_frac"] * self.z_stand

    def fresh(self, z):
        d = mujoco.MjData(self.model)
        d.qpos[:] = self.model.qpos0
        d.qpos[self.mp.q_adr] = self.q_default
        d.qpos[self.qa + 2] = z
        mujoco.mj_forward(self.model, d)
        return d

    def contacts(self, d):
        c = np.zeros(len(self.foot_geoms), bool)
        for i in range(d.ncon):
            g1, g2 = d.contact[i].geom1, d.contact[i].geom2
            for f, gs in enumerate(self.foot_geoms):
                if g1 in gs or g2 in gs:
                    c[f] = True
        return c


def run(sim, policy, cmd, seconds, view):
    m, mp, base = sim.model, sim.mp, sim.base
    d = sim.fresh(sim.z_stand + 0.005)
    decim = max(1, int(round(1.0 / (OBS.policy_hz * m.opt.timestep))))
    hist = np.zeros((OBS.history_len, NUM_JOINTS), np.float32)
    cmd = np.asarray(cmd, np.float32)
    down = np.array([0.0, 0.0, -1.0])
    viewer = None
    if view:
        from mujoco import viewer as mj_viewer
        viewer = viewer.launch_passive(m, d)

    vb, wz, zs, td, travel = [], [], [], 0, 0.0
    prev_c, prev_q, alive, t_alive = sim.contacts(d), d.qpos[mp.q_adr].copy(), True, 0.0
    for step in range(int(seconds * OBS.policy_hz)):
        if viewer is not None and not viewer.is_running():
            break
        t0 = time.time()
        R = d.xmat[base].reshape(3, 3)
        obs = build_obs(np, OBS, d.qvel[sim.dof + 3:sim.dof + 6], R.T @ down, cmd,
                        d.qpos[mp.q_adr], d.qvel[mp.v_adr], sim.q_default, hist)
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
        t_alive += 1.0 / OBS.policy_hz
        vb.append(R.T @ d.qvel[sim.dof:sim.dof + 3]); wz.append(d.qvel[sim.dof + 5]); zs.append(d.qpos[sim.qa + 2])
        c, q = sim.contacts(d), d.qpos[mp.q_adr].copy()
        td += int((c & ~prev_c).sum())                        # touchdown = foot goes from air to ground
        travel += float(np.abs(q - prev_q).sum())
        prev_c, prev_q = c, q
        if viewer is not None:
            viewer.sync()
            time.sleep(max(0.0, 1.0 / OBS.policy_hz - (time.time() - t0)))
    if viewer is not None:
        viewer.close()

    T = max(t_alive, 1e-6)
    vb = np.asarray(vb).reshape(-1, 3)
    # translation along the commanded direction, as a fraction of the commanded speed
    speed_cmd = float(np.hypot(cmd[0], cmd[1]))
    along = float(vb[:, :2].mean(0) @ (cmd[:2] / speed_cmd)) if speed_cmd > 1e-6 and len(vb) else float("nan")
    return dict(alive=alive, T=T, vx=float(vb[:, 0].mean()) if len(vb) else 0.0, vy=float(vb[:, 1].mean()) if len(vb) else 0.0,
                speed_frac=along / speed_cmd if speed_cmd > 1e-6 else float("nan"),
                yaw=float(np.mean(wz)) if wz else 0.0, yaw_frac=float(np.mean(wz)) / cmd[2] if abs(cmd[2]) > 1e-6 and wz else float("nan"),
                td_per_s=td / T, travel_per_s=travel / T, z=float(np.mean(zs)) if zs else 0.0)


def verdict(r, cmd):
    moving_cmd = abs(cmd[0]) + abs(cmd[1]) + abs(cmd[2]) > 1e-6
    if not r["alive"]:
        return f"fell after {r['T']:.1f} s"
    if not moving_cmd:
        return "standing (zero command)"
    fracs = [f for f in (r["speed_frac"], r["yaw_frac"]) if not np.isnan(f)]
    tracking = max(fracs) if fracs else 0.0
    if tracking >= 0.5:
        return "WALKING / TRACKING the command"
    if r["td_per_s"] < 0.2:
        return "NOT MOVING: feet do not lift (frozen / standing)"
    return (f"UNDER-TRACKING ({100 * tracking:.0f}% of the command): feet lift ({r['td_per_s']:.1f} touchdowns/s) "
            "but the body barely follows. Stepping in place, a weak response, or wrong direction")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True, help="one or more: bc_actor.pt and/or rsl_rl model_*.pt")
    ap.add_argument("--cmd", nargs=3, type=float, default=[0.0667, 0.0, 0.0], metavar=("VX", "VY", "YAW"))
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    conf = load_config(args.config) if args.config else CFG
    sim = Sim(conf)
    print(f"standing height {sim.z_stand:.3f} m | command [vx, vy, yaw] = {args.cmd} | {args.seconds:.0f} s")
    rows = []
    for ck in args.ckpt:
        policy = load_policy(ck, conf, "cpu")
        r = run(sim, policy, args.cmd, args.seconds, args.view)
        rows.append((ck, r))
    print(f"\n{'checkpoint':<44} {'alive':>5} {'vx':>7} {'vy':>7} {'speed%':>7} {'yaw':>7} {'yaw%':>6} {'td/s':>6} {'travel':>7} {'z':>6}")
    for ck, r in rows:
        pct = lambda f: f"{100 * f:6.0f}%" if not np.isnan(f) else f"{'-':>7}"
        print(f"{Path(ck).parent.name + '/' + Path(ck).name:<44} {int(r['alive']):5d} {r['vx']:7.4f} {r['vy']:7.4f} {pct(r['speed_frac'])} "
              f"{r['yaw']:7.4f} {pct(r['yaw_frac'])} {r['td_per_s']:6.2f} {r['travel_per_s']:7.2f} {r['z']:6.3f}")
    print("\n  speed% / yaw% = measured body velocity as % of the command; td/s = foot touchdowns per second (all feet);"
          "\n  travel = total joint motion in rad/s")
    for ck, r in rows:
        print(f"  {Path(ck).name}: {verdict(r, args.cmd)}")


if __name__ == "__main__":
    main()
