# Headless MPC teacher rollouts -> BC dataset.   run: mjpython python/rl/collect_data.py
# Records exactly what the student will see (Env.get_observation layout) and what the teacher commanded.
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
from sim.util.build_sim_xml import build_sim_xml, subtree_inertia
from rl.models.actor_critic import ModelConfig
from rl.environments.env import (
    GYRO_SCALE, CMD_SCALE_LIN, CMD_SCALE_ANG, QD_SCALE, LEG_NAMES, JOINTS,
    SRC, DST, KP, FORCE, START_HEIGHT, FALL_UP_Z, FALL_HEIGHT)

HORIZON, MPC_DT, MPC_GAIN = 10, 0.03, 1.3
REST_S, SETTLE_VEL = 2.0, 0.1
POLICY_HZ = 50.0
FOOT_BODIES = ("l0t", "l1t", "l2t", "l3t")
GROUND_GEOM = "floor"


class Resolver:
    """Same joint/actuator ordering as Env (leg0_coxa, leg0_femur, ...)."""
    def __init__(self, model):
        aids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{l}_{j}_act") for l in LEG_NAMES for j in JOINTS]
        jids = [int(model.actuator_trnid[a, 0]) for a in aids]
        self.aids = np.array(aids)
        self.q_adr = np.array([model.jnt_qposadr[j] for j in jids])
        self.v_adr = np.array([model.jnt_dofadr[j] for j in jids])
        self.limited = model.actuator_ctrllimited[self.aids].astype(bool)
        self.range = model.actuator_ctrlrange[self.aids].copy()
        self.foot_bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in FOOT_BODIES]
        self.floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, GROUND_GEOM)

    def contacts(self, model, data):
        c = np.zeros(4, dtype=bool)
        for i in range(data.ncon):
            g1, g2 = data.contact[i].geom1, data.contact[i].geom2
            if self.floor not in (g1, g2):
                continue
            other = g2 if g1 == self.floor else g1
            b = model.geom_bodyid[other]
            if b in self.foot_bodies:
                c[self.foot_bodies.index(b)] = True
        return c


class ScriptedCommands:
    """Random (v, yaw_rate) command, resampled every 3-6 s. Matches Env command ranges."""
    def __init__(self, rng, v=(-0.3, 0.3), yaw=(-0.8, 0.8), p_stand=0.15):
        self.rng, self.v, self.yaw, self.p_stand = rng, v, yaw, p_stand
        self.t_left, self.cmd = 0.0, (0.0, 0.0)

    def step(self, dt):
        self.t_left -= dt
        if self.t_left <= 0:
            self.t_left = self.rng.uniform(3, 6)
            if self.rng.random() < self.p_stand:
                self.cmd = (0.0, 0.0)
            else:
                self.cmd = (self.rng.uniform(*self.v), self.rng.uniform(*self.yaw))
        return self.cmd


class Pusher:
    """Random horizontal shoves on the base: accel (m/s^2) * robot mass, for ~0.1 s."""
    def __init__(self, rng, mass, acc=(1.0, 4.0), every=(2.0, 5.0), dur=0.1):
        self.rng, self.mass, self.acc, self.every, self.dur = rng, mass, acc, every, dur
        self.next_t, self.end_t = rng.uniform(*every), -1.0
        self.last_push_t = -1.0

    def apply(self, data, body_id):
        t = data.time
        if t >= self.next_t:
            ang = self.rng.uniform(0, 2 * np.pi)
            F = self.rng.uniform(*self.acc) * self.mass
            data.xfrc_applied[body_id, :3] = [F * np.cos(ang), F * np.sin(ang), 0.0]
            self.end_t, self.last_push_t = t + self.dur, t
            self.next_t = t + self.rng.uniform(*self.every)
        if t >= self.end_t:
            data.xfrc_applied[body_id, :] = 0.0


def run_episode(q_default, action_scale, seconds, seed, push, verbose=False):
    rng = np.random.default_rng(seed)
    build_sim_xml(str(SRC), str(DST), START_HEIGHT, KP, FORCE)
    model = mujoco.MjModel.from_xml_path(str(DST))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    res = Resolver(model)

    cfg = MPCConfig(horizon=HORIZON, dt=MPC_DT, mass=float(model.body_subtreemass[body]),
                    I_body=subtree_inertia(model, data, body), R_weight=1e-3, mu=0.4,
                    Q_diag=np.array([30, 30, 10, 1, 1, 10, 1, 1, 1, 1, 1, 20, 0.0]))
    mpc = MPC(cfg)
    ctrl = LegController(model, data)
    imu = Imu(model, data, "base")
    core = Core(mpc, ctrl, z_des_m=0.0, solve_period_s=MPC_DT, gain_mm_per_n=MPC_GAIN,
                gravity_z_mps2=float(model.opt.gravity[2]), ramp_s=0.1)
    qa = model.jnt_qposadr[model.body_jntadr[body]]
    core.reset(data.qpos[qa:qa + 3].copy(), np.zeros(3), imu.read())

    vmax = ctrl.v_max_mm_s * 1e-3                      # m/s, ~0.067 with the current gait settings
    cmds = ScriptedCommands(rng, v=(-vmax, vmax), yaw=(-0.35, 0.35))
    forced = False
    pusher = Pusher(rng, float(model.body_subtreemass[body])) if push else None

    dt = model.opt.timestep
    decim = max(1, int(round(1.0 / (POLICY_HZ * dt))))
    n_steps = int(seconds / dt)
    mpc_active, prev_action = False, np.zeros(12, np.float32)
    why = "ran full length"
    z_stand = None      # standing height, measured once the drop has settled
    next_log = 0.0
    O, A, C, K, P = [], [], [], [], []   # obs, action, command, contacts, pushed-flag

    for k in range(n_steps):
        imu.update(dt)
        s = core.estimate(dt, imu.read())
        v, yaw = cmds.step(dt) if data.time >= REST_S else (0.0, 0.0)
        core.v_cmd_mps, core.yaw_rate_cmd = v, yaw

        if not mpc_active and np.linalg.norm(s.linear_vel) < SETTLE_VEL:
            core.z_des, core.yaw_des = float(s.pos[2]), s.yaw
            mpc_active, core._t_engaged = True, 0.0
        if mpc_active:
            core.step(dt)
        if data.time >= REST_S and ctrl.is_idle() and (abs(v) > 1e-3 or abs(yaw) > 1e-3):
            if not forced:
                print("  NOTE: gait still IDLE with a non-zero command -> starting it manually (Core did not)")
                forced = True
            ctrl.set_body_command(v * 1e3, yaw)
            ctrl.request_start()
        ctrl.update(dt, core.estimator.state)

        R = data.xmat[body].reshape(3, 3)
        if verbose and data.time >= next_log:
            print(f"  t={data.time:5.2f} z={data.xpos[body, 2]:.3f} up_z={R[2, 2]:.2f} mpc={mpc_active} gait={ctrl.move_state.name} swing={ctrl.swing_leg} v={ctrl._v:.1f}mm/s cmd=({v:.3f},{yaw:.2f}) recorded={len(O)}")
            next_log += 0.5
        z = data.xpos[body, 2]
        if data.time >= REST_S and z_stand is None:
            z_stand = z
            print(f"  standing height at t={data.time:.2f}s: z={z_stand:.3f} (Env FALL_HEIGHT={FALL_HEIGHT})")
        # the robot is dropped from START_HEIGHT, so an absolute height test is meaningless during the drop
        tipped = R[2, 2] < FALL_UP_Z
        collapsed = z_stand is not None and z < 0.6 * z_stand
        if tipped or collapsed:
            why = f"FELL at t={data.time:.2f}s (z={z:.3f}, z_stand={z_stand}, up_z={R[2, 2]:.2f}, mpc_active={mpc_active})"
            break
        if not np.all(np.isfinite(data.qpos)):
            why = f"NaN state at t={data.time:.2f}s"
            break

        # ---- record (state the teacher saw, command the teacher issued) ----
        if data.time >= REST_S and k % decim == 0:
            q_t = data.ctrl[res.aids].copy()
            if res.limited.any():
                q_t = np.where(res.limited, np.clip(q_t, res.range[:, 0], res.range[:, 1]), q_t)
            act = np.clip((q_t - q_default) / action_scale, -1.0, 1.0).astype(np.float32)

            cmd3 = np.array([v, 0.0, yaw], np.float32)
            obs = np.concatenate([
                np.asarray(imu.read().gyro_rad_s) * GYRO_SCALE,
                R.T @ np.array([0.0, 0.0, -1.0]),
                cmd3 * np.array([CMD_SCALE_LIN, CMD_SCALE_LIN, CMD_SCALE_ANG]),
                data.qpos[res.q_adr] - q_default,
                data.qvel[res.v_adr] * QD_SCALE,
                prev_action,
            ]).astype(np.float32)
            assert obs.shape == (ModelConfig.OBS_DIM,)

            O.append(obs); A.append(act); C.append(cmd3); K.append(res.contacts(model, data))
            P.append(pusher is not None and 0.0 <= data.time - pusher.last_push_t < 1.0)
            prev_action = act

        if pusher:
            pusher.apply(data, body)
        mujoco.mj_step(model, data)

    print(f"  ended: {why}; recorded {len(O)} steps")
    # if the teacher fell, drop the last second (it was already failing)
    n = len(O) - (int(POLICY_HZ) if why.startswith("FELL") else 0)
    if n < 50:
        return None
    sl = slice(0, n)
    return {k: np.asarray(v[sl]) for k, v in dict(obs=O, act=A, cmd=C, contacts=K, pushed=P).items()}


def debug():
    """One episode, no pushes, prints state every 0.5 s.  python collect_data.py debug"""
    q_default = np.load("python/rl/data/run1.npz")["q_default"].astype(np.float32)
    run_episode(q_default, 1.0, 16.0, seed=0, push=False, verbose=True)


def main(out="python/rl/data/bc_run1.npz", n_episodes=200, seconds=25.0, push_fraction=0.5):
    q_default = np.load("python/rl/data/run1.npz")["q_default"].astype(np.float32)
    action_scale = 1.0            # keep identical to Env(action_scale_rad=...)
    eps = []
    for i in range(n_episodes):
        ep = run_episode(q_default, action_scale, seconds, seed=i, push=(i / n_episodes) < push_fraction)
        if ep is None:
            print(f"ep {i}: too short, skipped"); continue
        eps.append(ep)
        sat = (np.abs(ep["act"]) > 0.999).mean()
        print(f"ep {i}: {len(ep['obs'])} steps, saturated actions {sat:.1%}")
    lens = np.array([len(e["obs"]) for e in eps])
    np.savez_compressed(out, ep_len=lens, q_default=q_default, action_scale=action_scale,
                        **{k: np.concatenate([e[k] for e in eps]) for k in eps[0]})
    print("saved", out, "total steps", lens.sum())


if __name__ == "__main__":
    debug() if sys.argv[1:] == ["debug"] else main()