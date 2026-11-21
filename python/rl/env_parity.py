# Env parity test: does replaying the teacher's recorded actions open-loop through Env reproduce the teacher?
#   python python/rl/env_parity.py --file python/rl/data/mpc/walk_forward_a0.0_ang000_t06.00.npz
# Put this next to bc_eval.py (it imports load/foot_contacts/touchdowns from it).
import argparse
import sys

import numpy as np
import mujoco

from bc_eval import load, foot_contacts, touchdowns, PYTHON_DIR   # noqa: F401  (also puts python/ on sys.path)
import sim.config as cfg
from rl.environments.env import Env
from sim.util.mujoco_robot_map import MuJuCoMap

GROUPS = dict(gyro=slice(0, 3), grav=slice(3, 6), cmd=slice(6, 9), q=slice(9, 21), qd=slice(21, 33), prev_act=slice(33, 45))
CHECK_AT = (0, 1, 2, 5, 10, 25, 50, 100, 200, 400)


def check(name, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="an UNPUSHED walking scenario, e.g. walk_forward_a0.0_*.npz")
    args = ap.parse_args()

    ep = load(args.file)
    env = Env(q_default=ep["q_default"], action_scale_rad=ep["action_scale"],
              foot_body_names=cfg.FOOT_BODIES, ground_geom_name=cfg.GROUND_GEOM)
    mm = MuJuCoMap(env.model)

    print("\n--- static checks (generator's MuJuCoMap vs Env)")
    ok = True
    ok &= check("actuator order  (MuJuCoMap.aids == Env.actuator_ids)", np.array_equal(np.asarray(mm.aids), env.actuator_ids))
    ok &= check("joint qpos addr (MuJuCoMap.q_adr == Env.q_adr)", np.array_equal(np.asarray(mm.q_adr), env.q_adr))
    ok &= check("joint qvel addr (MuJuCoMap.v_adr == Env.v_adr)", np.array_equal(np.asarray(mm.v_adr), env.v_adr))
    ok &= check("foot bodies", np.array_equal(np.asarray(mm.foot_bodies), env.foot_body_ids))
    ok &= check("ctrl limits", np.array_equal(np.asarray(mm.limited), env.ctrl_limited)
                and np.allclose(np.asarray(mm.range), env.ctrl_range))
    ok &= check(f"policy rate (Env {env.actual_policy_hz:.3f} Hz vs file {ep['policy_hz']:.3f} Hz)",
                abs(env.actual_policy_hz - ep["policy_hz"]) < 1e-6)
    ok &= check("action_scale", np.allclose(env.action_scale, ep["action_scale"]))
    if not ok:
        print("\n  A static check failed: fix that before anything else. A joint-order mismatch alone explains 'tiny wrong movements'.")

    # ---- replay (same initialisation as bc_eval.rollout) ----
    n = len(ep["obs"])
    env.reset()
    d, b = env.data, env.base_qadr
    q_def = ep["q_default"]
    d.qpos[env.q_adr] = q_def + ep["obs"][0][9:21]
    d.qpos[b:b + 3] = ep["base_pos"][0]
    yaw0 = float(ep["base_yaw"][0])
    d.qpos[b + 3:b + 7] = [np.cos(yaw0 / 2), 0.0, 0.0, np.sin(yaw0 / 2)]
    d.qvel[:] = 0.0
    d.ctrl[env.actuator_ids] = q_def + env.action_scale * ep["act"][0]
    mujoco.mj_forward(env.model, d)
    env.imu.update()
    env.command[:] = ep["cmd"]
    env.command_timer = 1e9
    env.prev_action[:] = 0.0

    z_floor = 0.6 * (ep["z_stand"] if ep["z_stand"] > 0 else float(ep["base_pos"][0][2]))
    errs, con, pos, fell_i = [], [], [], -1
    for i in range(n):
        o = env.get_observation()
        errs.append({g: float(np.abs(o[s] - ep["obs"][i][s]).max()) for g, s in GROUPS.items()})
        env.step(ep["act"][i])
        R = d.xmat[env.body_id].reshape(3, 3)
        con.append(foot_contacts(env))
        pos.append(d.xpos[env.body_id].copy())
        if R[2, 2] < cfg.FALL_UP_Z or d.xpos[env.body_id, 2] < z_floor:
            fell_i = i
            break
    con, pos = np.array(con), np.array(pos)
    L = len(pos)

    print(f"\n--- obs error: max |Env obs - recorded obs| per group (replay of {args.file})")
    print("  step | " + " ".join(f"{g:>9}" for g in GROUPS))
    for t in CHECK_AT:
        if t < len(errs):
            print(f"  {t:4d} | " + " ".join(f"{errs[t][g]:9.2e}" for g in GROUPS))

    tc = ep["contacts"]
    td_T = sum(len(x) for x in touchdowns(tc, 1, L))
    td_R = sum(len(x) for x in touchdowns(con, 1, L))
    disp_T = ep["base_pos"][L - 1, :2] - ep["base_pos"][0, :2]
    disp_R = pos[L - 1, :2] - ep["base_pos"][0, :2]
    print("\n--- open-loop outcome")
    print(f"  replay fell:            {'YES at step %d' % fell_i if fell_i >= 0 else 'no'}   (ran {L}/{n} steps)")
    print(f"  base xy displacement:   teacher {np.round(disp_T, 3)} m   replay {np.round(disp_R, 3)} m")
    print(f"  base height at end:     teacher {ep['base_pos'][L - 1, 2]:.3f} m   replay {pos[L - 1, 2]:.3f} m")
    print(f"  foot touchdowns:        teacher {td_T}   replay {td_R}")

    print("""
How to read this:
  * Steps 0-5 should be near zero for every group. If gyro is off at step 0, suspect Imu.update(dt) vs update().
    If q/qd are off at step 0, suspect the init or joint order. If prev_act is off from step 1, suspect the action mapping.
  * Errors growing slowly after ~50 steps is normal (open loop + contact chaos). A jump in q/qd within the first few
    steps, or displacement near zero with touchdowns far below the teacher's, means the labels do not drive Env like the teacher.
  * Replay walks (displacement and touchdowns close to teacher, no fall): obs/action plumbing is fine. The 'little movements'
    problem is then compounding error or the copy shortcut, so go to the prev_action ablation and DART noise.
  * Replay barely moves or falls: fix the Env/generator mismatch first. Retraining will not help.""")


if __name__ == "__main__":
    main()