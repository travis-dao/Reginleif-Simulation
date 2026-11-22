"""MuJoCo Playground (MJX) env for the quadruped. Same obs/action definition as the MPC data (rl/obs.py).

  action in [-1, 1]^12  ->  q_target = q_default + action_scale * action  -> position actuators (kp from build_sim_xml)
  policy obs  = gyro | gravity_b | [v_x, v_y, yaw] cmd | q-q_default | qd | past N actions     (what the student gets)
  privileged  = body-frame lin vel | base height                                           (critic only)

Notes
  * The MJCF must be MJX-compatible (primitive collision geoms on the feet; mesh-mesh collisions are very limited).
  * No domain randomisation is included yet besides random commands + pushes.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

PYTHON_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PYTHON_DIR))

from rl.config import CFG, OBS, NUM_JOINTS, Q_DEFAULT_PATH
from rl.obs import build_obs, roll_history
from sim.util.mujoco_robot_map import MuJuCoMap
import sim.config as cfg
from sim.util.build_sim_xml import build_sim_xml

_NEVER = 1 << 30


def _load_mj_model() -> mujoco.MjModel:
    build_sim_xml(cfg.SRC, cfg.DST, cfg.START_HEIGHT, cfg.KP, cfg.FORCE)
    return mujoco.MjModel.from_xml_path(cfg.DST)


def default_config() -> config_dict.ConfigDict:
    c = config_dict.ConfigDict(copy.deepcopy(CFG["env"]))
    c.ctrl_dt = 1.0 / OBS.policy_hz
    c.sim_dt = float(_load_mj_model().opt.timestep)
    return c


class QuadrupedEnv(mjx_env.MjxEnv):
    def __init__(self, config: config_dict.ConfigDict | None = None, config_overrides: dict | None = None):
        super().__init__(config or default_config(), config_overrides)
        m = _load_mj_model()
        m.opt.timestep = self.sim_dt
        self._mj_model, self._mjx_model, self._xml_path = m, mjx.put_model(m), cfg.DST

        mp = MuJuCoMap(m)
        self._base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
        jadr = m.body_jntadr[self._base]
        self._qa, self._dof = int(m.jnt_qposadr[jadr]), int(m.jnt_dofadr[jadr])   # assumes "base" owns the free joint
        self._q_adr, self._v_adr = jnp.asarray(np.asarray(mp.q_adr)), jnp.asarray(np.asarray(mp.v_adr))
        self._aids = jnp.asarray(np.asarray(mp.aids))
        self._limited = jnp.asarray(np.asarray(mp.limited))
        self._lo, self._hi = jnp.asarray(mp.range[:, 0]), jnp.asarray(mp.range[:, 1])
        self._mass = float(m.body_subtreemass[self._base])
        self._q_default = jnp.asarray(np.load(Q_DEFAULT_PATH).astype(np.float32))
        self._n_sub = max(1, int(round(self._ctrl_dt / self.sim_dt)))
        self._push_steps = int(round(self._config.push.dur_s * OBS.policy_hz))
        self._down = jnp.asarray([0.0, 0.0, -1.0])
        self._scales = dict(self._config.reward_scales)

        qpos0 = np.array(m.qpos0)
        qpos0[np.asarray(mp.q_adr)] = np.asarray(self._q_default)
        qpos0[self._qa + 2] = cfg.START_HEIGHT
        # START_HEIGHT is a DROP height (~0.5 m); the robot actually stands much lower. Measure the real standing
        # height once on CPU (hold q_default), start the env there, and terminate relative to it.
        d = mujoco.MjData(m)
        d.qpos[:] = qpos0
        d.ctrl[np.asarray(mp.aids)] = np.asarray(self._q_default)
        for _ in range(int(1.0 / self.sim_dt)):
            mujoco.mj_step(m, d)
        self._z_stand = float(d.qpos[self._qa + 2])
        self._min_z = float(self._config.min_height_frac * self._z_stand)
        qpos0[self._qa + 2] = self._z_stand + 0.005
        print(f"[QuadrupedEnv] standing height {self._z_stand:.3f} m, terminate below {self._min_z:.3f} m")
        self._qpos0 = jnp.asarray(qpos0)
        self._metric_keys = ("track_lin", "track_yaw", "upright", "action_rate", "pose", "alive", "lin_err", "yaw_err", "yaw_signed")

    # ---------------------------------------------------------------- helpers
    def _ctrl_from_action(self, a):
        q_t = self._q_default + OBS.action_scale * a
        q_t = jnp.where(self._limited, jnp.clip(q_t, self._lo, self._hi), q_t)
        return jnp.zeros(self._mj_model.nu).at[self._aids].set(q_t)

    def get_obs(self, data, info):
        R = data.xmat[self._base].reshape(3, 3)
        policy = build_obs(jnp, OBS, data.qvel[self._dof + 3:self._dof + 6], R.T @ self._down, info["cmd"],
                           data.qpos[self._q_adr], data.qvel[self._v_adr], self._q_default, info["hist"])
        priv = jnp.concatenate([R.T @ data.qvel[self._dof:self._dof + 3], data.xpos[self._base, 2:3]]).astype(jnp.float32)
        return {"policy": policy, "privileged": priv}

    def _sample_cmd(self, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        c = self._config
        vx = jax.random.uniform(k1, minval=c.cmd_vx_range[0], maxval=c.cmd_vx_range[1])
        vy = jax.random.uniform(k2, minval=c.cmd_vy_range[0], maxval=c.cmd_vy_range[1])
        w = jax.random.uniform(k3, minval=c.cmd_yaw_range[0], maxval=c.cmd_yaw_range[1])
        stand = jax.random.uniform(k4) < c.stand_prob
        return jnp.where(stand, 0.0, jnp.stack([vx, vy, w])).astype(jnp.float32)

    # ---------------------------------------------------------------- API
    def reset(self, rng):
        k = jax.random.split(rng, 6)
        p = self._config.push
        data = mjx.make_data(self._mjx_model)
        data = data.replace(qpos=self._qpos0, qvel=jnp.zeros(self._mj_model.nv),
                            ctrl=self._ctrl_from_action(jnp.zeros(NUM_JOINTS)))
        data = mjx.forward(self._mjx_model, data)
        push_on = jax.random.uniform(k[1]) < p.prob
        t_push = jax.random.randint(k[2], (), int(p.t_range_s[0] * OBS.policy_hz), int(p.t_range_s[1] * OBS.policy_hz))
        info = {
            "step": jnp.int32(0),
            "cmd": self._sample_cmd(k[0]),
            "hist": jnp.zeros((OBS.history_len, NUM_JOINTS), jnp.float32),
            "last_act": jnp.zeros(NUM_JOINTS, jnp.float32),
            "next_push": jnp.where(push_on, t_push, _NEVER).astype(jnp.int32),
            "push_left": jnp.int32(0),
            "push_acc": jax.random.uniform(k[3], minval=p.accel_range[0], maxval=p.accel_range[1]),
            "push_ang": jax.random.uniform(k[4], minval=0.0, maxval=2 * jnp.pi),   # body frame, 0 = forward
            "push_force_w": jnp.zeros(3),
            "truncated": jnp.float32(0.0),
        }
        return mjx_env.State(data=data, obs=self.get_obs(data, info), reward=jnp.float32(0.0), done=jnp.float32(0.0),
                             metrics={k_: jnp.float32(0.0) for k_ in self._metric_keys}, info=info)

    def step(self, state, action):
        d, info = state.data, dict(state.info)
        a = jnp.clip(action, -1.0, 1.0)

        # ---- push (direction fixed in the body frame, same as gen_mpc_data.py) ----
        R = d.xmat[self._base].reshape(3, 3)
        yaw = jnp.arctan2(R[1, 0], R[0, 0])
        start = info["step"] == info["next_push"]
        f_bf = info["push_acc"] * self._mass * jnp.stack([jnp.cos(info["push_ang"]), jnp.sin(info["push_ang"])])
        f_w = jnp.stack([jnp.cos(yaw) * f_bf[0] - jnp.sin(yaw) * f_bf[1], jnp.sin(yaw) * f_bf[0] + jnp.cos(yaw) * f_bf[1], 0.0])
        info["push_force_w"] = jnp.where(start, f_w, info["push_force_w"])
        left = jnp.where(start, self._push_steps, info["push_left"])
        force = jnp.where(left > 0, info["push_force_w"], 0.0)
        info["push_left"] = jnp.maximum(left - 1, 0)

        d = d.replace(ctrl=self._ctrl_from_action(a), xfrc_applied=d.xfrc_applied.at[self._base, :3].set(force))
        d = jax.lax.fori_loop(0, self._n_sub, lambda _, x: mjx.step(self._mjx_model, x), d)

        prev_a = info["last_act"]
        info["hist"] = roll_history(jnp, info["hist"], a)      # newest entry = action just applied
        info["last_act"] = a
        info["step"] = info["step"] + 1

        # ---- reward / termination ----
        R = d.xmat[self._base].reshape(3, 3)
        grav = R.T @ self._down
        v_b = R.T @ d.qvel[self._dof:self._dof + 3]
        wz = d.qvel[self._dof + 5]
        cmd, c = info["cmd"], self._config
        lin_err = (v_b[0] - cmd[0]) ** 2 + (v_b[1] - cmd[1]) ** 2
        yaw_err = (wz - cmd[2]) ** 2
        terms = {
            "track_lin": jnp.exp(-lin_err / c.track_sigma_lin),
            "track_yaw": jnp.exp(-yaw_err / c.track_sigma_yaw),
            "upright": jnp.sum(grav[:2] ** 2),
            "action_rate": jnp.sum((a - prev_a) ** 2),
            "pose": jnp.sum((d.qpos[self._q_adr] - self._q_default) ** 2),
            "alive": jnp.float32(1.0),
        }
        bad = jnp.isnan(d.qpos).any() | jnp.isnan(d.qvel).any()
        terminated = bad | (-grav[2] < cfg.FALL_UP_Z) | (d.qpos[self._qa + 2] < self._min_z)
        truncated = (info["step"] >= c.episode_length) & ~terminated
        info["truncated"] = truncated.astype(jnp.float32)
        reward = sum(self._scales[k] * v for k, v in terms.items()) + self._scales["termination"] * terminated
        reward = jnp.nan_to_num(reward).astype(jnp.float32)
        metrics = {**{k: jnp.nan_to_num(v).astype(jnp.float32) for k, v in terms.items()},
                   "lin_err": jnp.nan_to_num(jnp.sqrt(lin_err)), "yaw_err": jnp.nan_to_num(jnp.sqrt(yaw_err)),
                   "yaw_signed": jnp.nan_to_num(wz - cmd[2]).astype(jnp.float32)}
        return state.replace(data=d, obs=self.get_obs(d, info), reward=reward,
                             done=(terminated | truncated).astype(jnp.float32), metrics=metrics, info=info)

    # ---------------------------------------------------------------- MjxEnv abstract properties
    @property
    def xml_path(self): return self._xml_path
    @property
    def action_size(self): return NUM_JOINTS
    @property
    def mj_model(self): return self._mj_model
    @property
    def mjx_model(self): return self._mjx_model