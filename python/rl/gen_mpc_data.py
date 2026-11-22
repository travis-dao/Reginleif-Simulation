# Generate MPC (teacher) datasets, one .npz per episode.
#   mjpython python/rl/gen_mpc_data.py                          # fixed, reproducible TEST grid (~150 episodes)
#   mjpython python/rl/gen_mpc_data.py --quick                  # small grid for smoke tests
#   mjpython python/rl/gen_mpc_data.py --mode train --episodes 2000
#
# test  : command x push accel x push direction (body frame) x push time (gait phase); fixed seed.
# train : random command / push / DART-style action noise (noise is added AFTER the label is recorded,
#         so labels stay clean while the visited states get diverse). Episodes where the teacher falls are dropped.
#
# Observation (see rl/obs.py + rl/configs/model.yaml):
#   gyro | gravity(body) | cmd [v_x mps, v_y mps = 0, yaw rad/s] | q - q_default | qd | past N actions (N = action_history_s * policy_hz)
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import mujoco

PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

from movement.leg.leg_controller import LegController
from movement.motion.mpc import MPC, MPCConfig
from movement.motion.core import Core
from movement.motion.imu import Imu
from rl.config import OBS, NUM_JOINTS, Q_DEFAULT_PATH
from rl.obs import build_obs, roll_history
from sim.util.mujoco_robot_map import MuJuCoMap
import sim.config as cfg
from sim.util.build_sim_xml import build_sim_xml, subtree_inertia
from sim.util.input_map import map_input_to_command, generate_walk_commands

POLICY_HZ = OBS.policy_hz

PUSH_T = 6.0          # s, gait is well established by then
PUSH_DUR = 0.1        # s
GAIT_SLOT_S = 0.45    # a swing starts every 0.45 s (period 1.8 s / 4 legs); offsets below cycle which leg is mid-swing
YAW_CMD = 0.30        # rad/s for the turn scenario (limit is ~0.4)

# name, input(x, y)
COMMANDS = (*generate_walk_commands(), ("turn_right", 1.0, 0.0), ("turn_left", -1.0, 0.0), ("stand", 0.0, 0.0))


def run_scenario(q_default, cmd_name, input_x, input_y, accel, angle_deg, push_t, seconds, noise_sigma, noise_tau, seed):
    model = mujoco.MjModel.from_xml_path(cfg.DST)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    mujoco_map = MuJuCoMap(model)
    mass = float(model.body_subtreemass[body])
    dof = model.jnt_dofadr[model.body_jntadr[body]]

    mpc_cfg = MPCConfig(horizon=cfg.HORIZON, dt=cfg.MPC_DT, mass=mass, I_body=subtree_inertia(model, data, body),
                        R_weight=1e-3, mu=0.4, Q_diag=np.array([30, 30, 10, 1, 1, 10, 1, 1, 1, 1, 1, 20, 0.0]))
    mpc, lc, imu = MPC(mpc_cfg), LegController(model, data), Imu(model, data, "base")
    core = Core(mpc, lc, z_des_m=0.0, solve_period_s=cfg.MPC_DT, gain_mm_per_n=cfg.MPC_GAIN,
                gravity_z_mps2=float(model.opt.gravity[2]), ramp_s=0.1)
    qa = model.jnt_qposadr[model.body_jntadr[body]]
    core.reset(data.qpos[qa:qa + 3].copy(), np.zeros(3), imu.read())

    v_cmd, yaw_cmd = map_input_to_command(input_x, input_y, lc.v_max_mm_s)
    cmd3 = np.array([v_cmd, 0.0, yaw_cmd], np.float32)       # what the student is told: [vx m/s, vy m/s (MPC has no lateral), yaw rad/s]
    dt = model.opt.timestep
    decim = max(1, int(round(1.0 / (POLICY_HZ * dt))))

    mpc_active, forced, z_stand = False, False, None
    act_hist = np.zeros((OBS.history_len, NUM_JOINTS), np.float32)   # oldest -> newest, zeros at episode start
    push_F_bf = np.zeros(3)
    push_start_idx, pushing, push_end = -1, False, -1.0
    fell, fell_t = False, -1.0
    log = {k: [] for k in ("obs", "act", "contacts", "foot_pos_bf", "foot_cmd_bf", "landing_bf",
                           "base_pos", "base_vel", "base_yaw", "time", "push_active")}

    rng = np.random.default_rng(seed)
    ou = np.zeros(NUM_JOINTS)

    for k in range(int(seconds / dt)):
        imu.update(dt)
        s = core.estimate(dt, imu.read())
        on = data.time >= cfg.REST_S
        core.v_cmd_mps, core.yaw_rate_cmd = (v_cmd, yaw_cmd) if on else (0.0, 0.0)

        if not mpc_active and np.linalg.norm(s.linear_vel) < cfg.SETTLE_VEL:
            core.z_des, core.yaw_des = float(s.pos[2]), s.yaw
            mpc_active, core._t_engaged = True, 0.0
        if mpc_active:
            core.step(dt)
        if on and lc.is_idle() and (abs(v_cmd) > 1e-3 or abs(yaw_cmd) > 1e-3):
            forced = True
            lc.set_body_command(v_cmd * 1e3, yaw_cmd)
            lc.request_start()
        lc.update(dt, core.estimator.state)

        R = data.xmat[body].reshape(3, 3)
        z = data.xpos[body, 2]
        if on and z_stand is None:
            z_stand = z
        if R[2, 2] < cfg.FALL_UP_Z or (z_stand is not None and z < 0.6 * z_stand):
            fell, fell_t = True, data.time
            break

        # ---- push: fixed direction in the BODY frame (so "forward push" means the same at any heading) ----
        if accel > 0 and not pushing and data.time >= push_t and push_start_idx < 0:
            yaw = np.arctan2(R[1, 0], R[0, 0])
            a = np.deg2rad(angle_deg)
            d_bf = np.array([np.cos(a), np.sin(a), 0.0])
            Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
            push_F_bf = accel * mass * d_bf
            data.xfrc_applied[body, :3] = Rz @ push_F_bf
            pushing, push_start_idx, push_end = True, len(log["obs"]), data.time + PUSH_DUR
        if pushing and data.time >= push_end:
            data.xfrc_applied[body, :] = 0.0
            pushing = False

        # ---- record at policy rate (state the teacher saw, command the teacher issued) ----
        if on and k % decim == 0:
            q_t = data.ctrl[mujoco_map.aids].copy()
            if mujoco_map.limited.any():
                q_t = np.where(mujoco_map.limited, np.clip(q_t, mujoco_map.range[:, 0], mujoco_map.range[:, 1]), q_t)
            act = np.clip((q_t - q_default) / OBS.action_scale, -1.0, 1.0).astype(np.float32)
            # gyro = body-frame angular velocity of the free joint: the exact quantity the MJX env reads
            obs = build_obs(np, OBS, data.qvel[dof + 3:dof + 6], R.T @ np.array([0.0, 0.0, -1.0]), cmd3,
                            data.qpos[mujoco_map.q_adr], data.qvel[mujoco_map.v_adr], q_default, act_hist)
            assert obs.shape == (OBS.obs_dim,)
            foot_w = np.array([data.xpos[b] for b in mujoco_map.foot_bodies])
            log["obs"].append(obs); log["act"].append(act)
            log["contacts"].append(mujoco_map.contacts(model, data))
            log["foot_pos_bf"].append((foot_w - data.xpos[body]) @ R)                      # R.T @ p, row-vector form; metres
            log["foot_cmd_bf"].append([[v.x, v.y, v.z] for v in lc.foot_positions_body()])  # mm, controller target
            log["landing_bf"].append([[v.x, v.y, v.z] for v in lc.landing_targets_body()])  # mm
            log["base_pos"].append(data.xpos[body].copy())
            log["base_vel"].append(data.qvel[dof:dof + 3].copy())
            log["base_yaw"].append(np.arctan2(R[1, 0], R[0, 0]))
            log["time"].append(data.time)
            log["push_active"].append(pushing)
            act_hist = roll_history(np, act_hist, act)      # obs at the next step contains this action as the newest entry

        if noise_sigma > 0 and on:
            ou += (-ou / noise_tau) * dt + noise_sigma * np.sqrt(2 * dt / noise_tau) * rng.standard_normal(NUM_JOINTS)
            c = data.ctrl[mujoco_map.aids] + ou
            data.ctrl[mujoco_map.aids] = np.where(mujoco_map.limited, np.clip(c, mujoco_map.range[:, 0], mujoco_map.range[:, 1]), c)

        mujoco.mj_step(model, data)

    arrays = {k: np.asarray(v) for k, v in log.items()}
    meta = dict(command_name=cmd_name, v_cmd=v_cmd, yaw_cmd=yaw_cmd, cmd=cmd3,
                push_accel=accel, push_angle_deg=angle_deg, push_t=push_t, push_dur=PUSH_DUR,
                push_force_bf=push_F_bf, push_index=push_start_idx, fell=fell, fell_t=fell_t,
                gait_started_manually=forced, z_stand=z_stand if z_stand is not None else -1.0,
                policy_hz=POLICY_HZ, q_default=q_default, action_scale=OBS.action_scale,
                obs_dim=OBS.obs_dim, history_len=OBS.history_len, noise_sigma=noise_sigma)
    return arrays, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["test", "train"], default="test")
    ap.add_argument("--out", default=None, help="default: python/rl/data/mpc (test) or python/rl/data/mpc_train (train)")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--accels", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0])
    ap.add_argument("--angles", type=float, nargs="+", default=[0, 90, 180, 270], help="deg, body frame; 0 = forward, 90 = left")
    ap.add_argument("--offsets", type=float, nargs="+", default=[0.0, 1 * GAIT_SLOT_S, 2 * GAIT_SLOT_S, 3 * GAIT_SLOT_S],
                    help="seconds added to push time, cycles which leg is mid-swing")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--episodes", type=int, default=500, help="train mode: number of random episodes")
    ap.add_argument("--push_prob", type=float, default=0.7, help="train mode")
    ap.add_argument("--seed", type=int, default=0, help="fixed default so the test set is reproducible")
    ap.add_argument("--noise_sigma", type=float, default=None, help="rad; default 0 (test) / 0.03 (train)")
    ap.add_argument("--noise_tau", type=float, default=0.3)
    args = ap.parse_args()
    if args.quick:
        args.accels, args.angles, args.offsets = [3.0], [0, 90, 180, 270], [0.0, 2 * GAIT_SLOT_S]
    if args.noise_sigma is None:
        args.noise_sigma = 0.0 if args.mode == "test" else 0.03
    out = Path(args.out or (PYTHON_DIR / "rl" / "data" / ("mpc" if args.mode == "test" else "mpc_train")))
    out.mkdir(parents=True, exist_ok=True)
    print(f"mode={args.mode} seed={args.seed} obs_dim={OBS.obs_dim} (history {OBS.history_len} x {NUM_JOINTS}) -> {out}")

    q_default = np.load(Q_DEFAULT_PATH).astype(np.float32)
    assert q_default.shape == (NUM_JOINTS,), q_default.shape
    build_sim_xml(cfg.SRC, cfg.DST, cfg.START_HEIGHT, cfg.KP, cfg.FORCE)       # once, not per episode
    vs = [map_input_to_command(x, y, 1000.0)[0] for _, x, y in COMMANDS]       # only the sign of v matters here
    n_back, n_fwd = sum(v < -1e-6 for v in vs), sum(v > 1e-6 for v in vs)
    print(f"{len(COMMANDS)} commands: {n_fwd} forward, {n_back} backward, {len(COMMANDS) - n_fwd - n_back} zero-velocity (turn/stand)")
    if n_back == 0:
        print("WARNING: no backward command in COMMANDS -> the student will never see v_x < 0. "
              "Check generate_walk_commands() / map_input_to_command() or add backward entries.")

    jobs = []
    if args.mode == "test":
        for name, input_x, input_y in COMMANDS:
            jobs.append((name, input_x, input_y, 0.0, 0.0, PUSH_T))                           # unpushed baseline
            offsets = args.offsets if name != "stand" else args.offsets[:1]          # gait phase is irrelevant when standing
            for acc, ang, off in itertools.product(args.accels, args.angles, offsets):
                jobs.append((name, input_x, input_y, acc, ang, PUSH_T + off))
    else:
        r = np.random.default_rng(args.seed)
        for _ in range(args.episodes):
            name, input_x, input_y = COMMANDS[r.integers(len(COMMANDS))]
            if r.random() < args.push_prob:
                jobs.append((name, input_x, input_y, float(r.uniform(0.5, 4.5)), float(r.uniform(0, 360)), float(r.uniform(4.0, 10.0))))
            else:
                jobs.append((name, input_x, input_y, 0.0, 0.0, PUSH_T))

    manifest, dropped = [], 0
    for i, (name, input_x, input_y, acc, ang, pt) in enumerate(jobs):
        arrays, meta = run_scenario(q_default, name, input_x, input_y, acc, ang, pt, args.seconds,
                                    args.noise_sigma, args.noise_tau, args.seed + i)
        if args.mode == "train" and meta["fell"]:
            dropped += 1                                    # teacher failure = bad labels
            print(f"[{i + 1}/{len(jobs)}] teacher fell, dropped")
            continue
        fname = (f"{name}_a{acc:.1f}_ang{int(ang):03d}_t{pt:05.2f}.npz" if args.mode == "test"
                 else f"ep{i:05d}_{name}_a{acc:.1f}_ang{int(ang):03d}_t{pt:05.2f}.npz")
        np.savez_compressed(out / fname, **arrays, **meta)
        print(f"[{i + 1}/{len(jobs)}] {fname}: {len(arrays['obs'])} steps, fell={meta['fell']}"
              + (" (gait started manually)" if meta["gait_started_manually"] else ""))
        manifest.append(dict(file=fname, command=name, accel=acc, angle_deg=ang, push_t=pt,
                             steps=int(len(arrays["obs"])), fell=bool(meta["fell"])))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "spec.json").write_text(json.dumps(dict(obs_dim=OBS.obs_dim, history_len=OBS.history_len, policy_hz=OBS.policy_hz,
                                                   seed=args.seed, mode=args.mode, noise_sigma=args.noise_sigma), indent=1))
    n_fell = sum(m["fell"] for m in manifest)
    print(f"\nwrote {len(manifest)} files to {out}; teacher fell in {n_fell}; dropped {dropped}")


if __name__ == "__main__":
    main()