"""MuJoCo Playground (MJX) env for the quadruped. Same obs/action definition as the MPC data (rl/obs.py).

  action in [-1, 1]^12  ->  q_target = q_default + action_scale * action  -> position actuators (kp from build_sim_xml)
  policy obs  = gyro | gravity_b | [v_x, v_y, yaw] cmd | q-q_default | qd | past N actions     (what the student gets)
  privileged  = body-frame lin vel | base height                                               (critic only)

Maneuvers (env.maneuver): scripted run -> stop -> turn (180 deg or small +-angle) -> run sequences, expressed purely as a
  time-varying [v_x, v_y, yaw] command, so the observation layout is unchanged. Episodes with maneuver.prob use them;
  the rest keep the random command resampling below.

Agility additions
  * commands are resampled mid-episode (new random command / brake to zero / reverse), so braking and direction
    changes are trained, not just constant-velocity walking
  * domain randomisation: friction, motor strength, payload (base mass), action latency, pushes
  * rewards: feet air time, foot slip, torque, joint limits (+ the original tracking / upright / smoothness terms)

Notes
  * The MJCF must be MJX-compatible (primitive or simple mesh colliders on the feet).
  * Motor-strength DR scales the position error (q_target - q), which is exactly a kp scale for a P-only servo.
  * Payload DR adds a constant vertical force on the base (changes weight, not inertia).
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

from rl.config import CFG, OBS, NUM_JOINTS, Q_DEFAULT_PATH, load_baseline_cmds
from rl.obs import build_obs, roll_history
from sim.util.mujoco_robot_map import MuJuCoMap
import sim.config as cfg
from sim.util.build_sim_xml import build_sim_xml

_NEVER = 1 << 30
_TERMS = ("track_lin", "track_yaw", "upright", "action_rate", "pose", "alive",
          "feet_air_time", "foot_slip", "torque", "joint_limit", "stand_still")
_METRICS = _TERMS + ("lin_err", "yaw_err", "yaw_signed", "term_tilt", "term_height", "term_nan")


def _load_mj_model() -> mujoco.MjModel:
    build_sim_xml(cfg.SRC, cfg.DST, cfg.START_HEIGHT, cfg.KP, cfg.FORCE)
    return mujoco.MjModel.from_xml_path(cfg.DST)


def default_config(cfg_dict: dict | None = None) -> config_dict.ConfigDict:
    c = config_dict.ConfigDict(copy.deepcopy((cfg_dict or CFG)["env"]))
    c.cmd_bank = load_baseline_cmds(cfg_dict or CFG).tolist()      # MPC's own commands, sampled with prob env.cmd_bank_prob
    c.ctrl_dt = 1.0 / OBS.policy_hz
    c.sim_dt = float(_load_mj_model().opt.timestep)
    return c


def _u(key, lohi):
    return jax.random.uniform(key, minval=lohi[0], maxval=lohi[1])


class QuadrupedEnv(mjx_env.MjxEnv):
    def __init__(self, config: config_dict.ConfigDict | None = None, config_overrides: dict | None = None):
        super().__init__(config or default_config(), config_overrides)
        m = _load_mj_model()
        m.opt.timestep = self._sim_dt
        self._mj_model, self._mjx_model, self._xml_path = m, mjx.put_model(m), cfg.DST
        c = self._config

        mp = MuJuCoMap(m)
        self._base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
        jadr = m.body_jntadr[self._base]
        self._qa, self._dof = int(m.jnt_qposadr[jadr]), int(m.jnt_dofadr[jadr])   # assumes "base" owns the free joint
        aids_np = np.asarray(mp.aids)
        self._q_adr, self._v_adr = jnp.asarray(np.asarray(mp.q_adr)), jnp.asarray(np.asarray(mp.v_adr))
        self._aids = jnp.asarray(aids_np)
        self._limited = jnp.asarray(np.asarray(mp.limited))
        self._lo, self._hi = jnp.asarray(mp.range[:, 0]), jnp.asarray(mp.range[:, 1])
        self._mass = float(m.body_subtreemass[self._base])
        self._gz = float(m.opt.gravity[2])
        self._q_default = jnp.asarray(np.load(Q_DEFAULT_PATH).astype(np.float32))
        self._n_sub = max(1, int(round(self._ctrl_dt / self._sim_dt)))
        self._push_steps = int(round(c.push.dur_s * OBS.policy_hz))
        self._down = jnp.asarray([0.0, 0.0, -1.0])
        self._scales = dict(c.reward_scales)
        self._dr = bool(c.dr.enabled)
        self._lat_max = int(c.dr.latency_steps_max)
        self._resample = float(c.cmd_resample_s[1]) > 0
        bank = np.asarray(c.get("cmd_bank", []), np.float32).reshape(-1, 3)
        self._bank = jnp.asarray(bank)
        self._bank_p = float(c.get("cmd_bank_prob", 0.0)) if len(bank) else 0.0

        # maneuver curriculum (run -> stop -> turn -> run); prob 0 or enabled=false -> legacy random commands only
        self._man = c.get("maneuver", None)
        self._man_p = float(self._man.prob) if (self._man is not None and bool(self._man.get("enabled", False))) else 0.0
        # optional speed-relative tracking: widens the reward sigma at high commanded speed (0 = off = old behaviour)
        self._lin_ref = float(c.get("track_lin_rel_ref", 0.0))
        self._yaw_ref = float(c.get("track_yaw_rel_ref", 0.0))

        # feet: bodies, colliding geoms (for contact detection), friction table (for friction DR)
        self._foot_bodies = np.asarray(mp.foot_bodies)
        self._foot_geoms = [np.where((m.geom_bodyid == b) & (m.geom_contype > 0))[0] for b in self._foot_bodies]
        assert all(len(g) for g in self._foot_geoms), "no colliding geom found on a foot body"
        self._geom_friction = jnp.asarray(m.geom_friction)

        # torque normalisation (dimensionless penalty): actuator force range if limited, else 1
        lim = np.asarray(m.actuator_forcelimited, bool)[aids_np]
        fr = np.abs(m.actuator_forcerange[aids_np]).max(axis=1)
        self._f_scale = jnp.asarray(np.where(lim & (fr > 0), fr, 1.0), jnp.float32)

        qpos0 = np.array(m.qpos0)
        qpos0[np.asarray(mp.q_adr)] = np.asarray(self._q_default)
        qpos0[self._qa + 2] = cfg.START_HEIGHT
        # START_HEIGHT is a DROP height; measure the real standing height once on CPU (hold q_default)
        d = mujoco.MjData(m)
        d.qpos[:] = qpos0
        d.ctrl[aids_np] = np.asarray(self._q_default)
        for _ in range(int(1.0 / self.sim_dt)):
            mujoco.mj_step(m, d)
        self._z_stand = float(d.qpos[self._qa + 2])
        self._min_z = float(c.min_height_frac * self._z_stand)
        qpos0[self._qa + 2] = self._z_stand + 0.005
        print(f"[QuadrupedEnv] standing height {self._z_stand:.3f} m, terminate below {self._min_z:.3f} m, DR={self._dr}")
        self._qpos0 = jnp.asarray(qpos0)

    # ---------------------------------------------------------------- helpers
    def _target(self, a):
        q_t = self._q_default + OBS.action_scale * a
        return jnp.where(self._limited, jnp.clip(q_t, self._lo, self._hi), q_t)

    def _ctrl(self, q_t):
        return jnp.zeros(self._mj_model.nu).at[self._aids].set(q_t)

    def _foot_contacts(self, data):
        con = getattr(data, "_impl", data).contact      # newer MJX moved Data.contact under _impl
        hits = []
        for gs in self._foot_geoms:
            hit = jnp.zeros((), bool)
            for g in gs:
                hit = hit | jnp.any(((con.geom1 == g) | (con.geom2 == g)) & (con.dist < 0))
            hits.append(hit)
        return jnp.stack(hits)

    def _model_with_friction(self, fric):
        fr = self._geom_friction.at[:, 0].multiply(fric)
        try:
            return self._mjx_model.tree_replace({"geom_friction": fr})
        except AttributeError:
            return self._mjx_model.replace(geom_friction=fr)

    def get_obs(self, data, info):
        R = data.xmat[self._base].reshape(3, 3)
        policy = build_obs(jnp, OBS, data.qvel[self._dof + 3:self._dof + 6], R.T @ self._down, info["cmd"],
                           data.qpos[self._q_adr], data.qvel[self._v_adr], self._q_default, info["hist"])
        priv = jnp.concatenate([R.T @ data.qvel[self._dof:self._dof + 3], data.xpos[self._base, 2:3]]).astype(jnp.float32)
        return {"policy": policy, "privileged": priv}

    def _sample_cmd(self, key):
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        c = self._config
        vx, vy, w = _u(k1, c.cmd_vx_range), _u(k2, c.cmd_vy_range), _u(k3, c.cmd_yaw_range)
        stand = jax.random.uniform(k4) < c.stand_prob
        uniform_cmd = jnp.where(stand, 0.0, jnp.stack([vx, vy, w]))
        if self._bank_p > 0:                         # with prob cmd_bank_prob: one of the MPC's own commands
            pick = self._bank[jax.random.randint(k5, (), 0, self._bank.shape[0])]
            uniform_cmd = jnp.where(jax.random.uniform(k6) < self._bank_p, pick, uniform_cmd)
        return uniform_cmd.astype(jnp.float32)

    # ---------------------------------------------------------------- maneuvers
    def _man_plan(self, key):
        """One maneuver: run at vx for run_n steps, stand for stop_n steps, turn at `rate` for turn_n steps.
        Returns vx, signed yaw rate, int32 [run_n, stop_n, turn_n]. Turn length = angle / |rate| (open-loop)."""
        m, hz = self._man, OBS.policy_hz
        k = jax.random.split(key, 7)
        vx = _u(k[0], m.vx_range)
        big = jax.random.uniform(k[1]) < m.p_turn180                     # 180 deg turn vs small angle
        rate = _u(k[2], m.turn_rate_range)
        sign = jnp.where(jax.random.uniform(k[3]) < 0.5, -1.0, 1.0)
        ang = jnp.where(big, jnp.pi, jnp.deg2rad(_u(k[4], m.small_angle_deg)))
        run_n = jax.random.randint(k[5], (), int(m.run_s[0] * hz), int(m.run_s[1] * hz) + 1)
        stop_n = jax.random.randint(k[6], (), int(m.stop_s[0] * hz), int(m.stop_s[1] * hz) + 1)
        turn_n = jnp.maximum(jnp.ceil(ang / rate * hz), 1.0)
        return vx, sign * rate, jnp.stack([run_n, stop_n, turn_n.astype(jnp.int32)]).astype(jnp.int32)

    @staticmethod
    def _man_cmd(phase, vx, rate):
        z = jnp.zeros(())
        return jnp.where(phase == 0, jnp.stack([vx, z, z]),
                         jnp.where(phase == 1, jnp.zeros(3), jnp.stack([z, z, rate]))).astype(jnp.float32)

    def _sample_period(self, key):
        if not self._resample:
            return jnp.int32(_NEVER)
        lo, hi = (int(t * OBS.policy_hz) for t in self._config.cmd_resample_s)
        return jax.random.randint(key, (), lo, max(hi, lo + 1)).astype(jnp.int32)

    # ---------------------------------------------------------------- API
    def reset(self, rng):
        k = jax.random.split(rng, 12)
        c, p, dr = self._config, self._config.push, self._config.dr
        data = mjx.make_data(self._mjx_model)
        data = data.replace(qpos=self._qpos0, qvel=jnp.zeros(self._mj_model.nv),
                            ctrl=self._ctrl(self._target(jnp.zeros(NUM_JOINTS))))
        data = mjx.forward(self._mjx_model, data)
        push_on = jax.random.uniform(k[1]) < p.prob
        if self._man_p > 0:
            man_on = jax.random.uniform(k[11]) < self._man_p
            m_vx, m_rate, m_n = self._man_plan(jax.random.fold_in(k[11], 1))
        else:
            man_on, m_vx, m_rate, m_n = jnp.bool_(False), jnp.float32(0.0), jnp.float32(0.0), jnp.ones(3, jnp.int32)
        t_push = jax.random.randint(k[2], (), int(p.t_range_s[0] * OBS.policy_hz), int(p.t_range_s[1] * OBS.policy_hz))
        if self._dr:
            fric, motor, mass = _u(k[7], dr.friction_range), _u(k[8], dr.motor_strength_range), _u(k[9], dr.mass_scale_range)
            delay = jax.random.randint(k[10], (), 0, self._lat_max + 1).astype(jnp.int32)
        else:
            fric = motor = mass = jnp.float32(1.0)
            delay = jnp.int32(0)
        info = {
            "step": jnp.int32(0),
            "rng": k[5],
            "cmd": jnp.where(man_on, self._man_cmd(jnp.int32(0), m_vx, m_rate), self._sample_cmd(k[0])),
            "next_cmd": jnp.where(man_on, _NEVER, self._sample_period(k[6])).astype(jnp.int32),
            "man_on": man_on, "man_phase": jnp.int32(0), "man_left": m_n[0],
            "man_vx": jnp.asarray(m_vx, jnp.float32), "man_rate": jnp.asarray(m_rate, jnp.float32), "man_n": m_n,
            "hist": jnp.zeros((OBS.history_len, NUM_JOINTS), jnp.float32),
            "last_act": jnp.zeros(NUM_JOINTS, jnp.float32),
            "act_buf": jnp.zeros((self._lat_max + 1, NUM_JOINTS), jnp.float32),
            "delay": delay, "fric": jnp.asarray(fric, jnp.float32), "motor": jnp.asarray(motor, jnp.float32),
            "mass": jnp.asarray(mass, jnp.float32),
            "next_push": jnp.where(push_on, t_push, _NEVER).astype(jnp.int32),
            "push_left": jnp.int32(0),
            "push_acc": _u(k[3], p.accel_range),
            "push_ang": jax.random.uniform(k[4], minval=0.0, maxval=2 * jnp.pi),   # body frame, 0 = forward
            "push_force_w": jnp.zeros(3),
            "air_time": jnp.zeros(len(self._foot_geoms)),
            "last_contact": jnp.ones(len(self._foot_geoms), bool),                 # starts standing
            "foot_prev": data.xpos[self._foot_bodies],
            "truncated": jnp.float32(0.0),
        }
        return mjx_env.State(data=data, obs=self.get_obs(data, info), reward=jnp.float32(0.0), done=jnp.float32(0.0),
                             metrics={k_: jnp.float32(0.0) for k_ in _METRICS}, info=info)

    def step(self, state, action):
        d, info = state.data, dict(state.info)
        c, dt = self._config, self._ctrl_dt
        a = jnp.clip(action, -1.0, 1.0)

        # ---- mid-episode command change: new random command / brake to zero / reverse ----
        rng, k = jax.random.split(info["rng"])
        info["rng"] = rng
        kf, ku, kt, km = jax.random.split(k, 4)
        u, cur = jax.random.uniform(ku), info["cmd"]
        new = jnp.where(u < c.cmd_p_brake, jnp.zeros(3),
                        jnp.where(u < c.cmd_p_brake + c.cmd_p_flip, -cur, self._sample_cmd(kf)))
        due = info["step"] == info["next_cmd"]
        legacy_cmd = jnp.where(due, new, cur)
        info["next_cmd"] = jnp.where(due, info["step"] + self._sample_period(kt), info["next_cmd"])
        if self._man_p > 0:
            # maneuver state machine: phase 0 run -> 1 stop -> 2 turn -> 0 (new random maneuver)
            left = info["man_left"] - 1
            adv = left <= 0
            nxt = (info["man_phase"] + 1) % 3
            new_run = adv & (nxt == 0)
            pvx, prate, pn = self._man_plan(km)
            info["man_vx"] = jnp.where(new_run, pvx, info["man_vx"])
            info["man_rate"] = jnp.where(new_run, prate, info["man_rate"])
            info["man_n"] = jnp.where(new_run, pn, info["man_n"])
            info["man_phase"] = jnp.where(adv, nxt, info["man_phase"])
            info["man_left"] = jnp.where(adv, info["man_n"][info["man_phase"]], left)
            man_cmd = self._man_cmd(info["man_phase"], info["man_vx"], info["man_rate"])
            info["cmd"] = jnp.where(info["man_on"], man_cmd, legacy_cmd)
        else:
            info["cmd"] = legacy_cmd

        # ---- action latency + motor strength ----
        buf = jnp.concatenate([a[None], info["act_buf"][:-1]], axis=0)
        info["act_buf"] = buf
        q_t = self._target(buf[info["delay"]])
        if self._dr:
            q = d.qpos[self._q_adr]
            q_t = q + info["motor"] * (q_t - q)          # == scaling the servo kp

        # ---- push (direction fixed in the body frame, same as gen_mpc_data.py) + payload ----
        R = d.xmat[self._base].reshape(3, 3)
        yaw = jnp.arctan2(R[1, 0], R[0, 0])
        start = info["step"] == info["next_push"]
        f_bf = info["push_acc"] * self._mass * jnp.stack([jnp.cos(info["push_ang"]), jnp.sin(info["push_ang"])])
        f_w = jnp.stack([jnp.cos(yaw) * f_bf[0] - jnp.sin(yaw) * f_bf[1], jnp.sin(yaw) * f_bf[0] + jnp.cos(yaw) * f_bf[1], 0.0])
        info["push_force_w"] = jnp.where(start, f_w, info["push_force_w"])
        left = jnp.where(start, self._push_steps, info["push_left"])
        force = jnp.where(left > 0, info["push_force_w"], 0.0)
        info["push_left"] = jnp.maximum(left - 1, 0)
        if self._dr:
            force = force + jnp.stack([0.0, 0.0, (info["mass"] - 1.0) * self._mass * self._gz])

        model = self._model_with_friction(info["fric"]) if self._dr else self._mjx_model
        d = d.replace(ctrl=self._ctrl(q_t), xfrc_applied=d.xfrc_applied.at[self._base, :3].set(force))
        d = jax.lax.fori_loop(0, self._n_sub, lambda _, x: mjx.step(model, x), d)

        prev_a = info["last_act"]
        info["hist"] = roll_history(jnp, info["hist"], a)      # history holds the COMMANDED actions
        info["last_act"] = a
        info["step"] = info["step"] + 1

        # ---- feet: contact, air time, slip ----
        contact = self._foot_contacts(d)
        first = contact & ~info["last_contact"]
        air_total = info["air_time"] + dt
        cmd = info["cmd"]
        moving = (jnp.linalg.norm(cmd[:2]) + jnp.abs(cmd[2])) > 1e-2
        r_air = jnp.sum(first * jnp.clip(air_total - c.air_time_target, -c.air_time_target, c.air_time_bonus_clip)) * moving
        foot_pos = d.xpos[self._foot_bodies]
        v_foot = (foot_pos - info["foot_prev"])[:, :2] / dt
        slip = jnp.sum((contact & info["last_contact"]) * jnp.sqrt(jnp.sum(v_foot ** 2, axis=-1) + 1e-8))
        info["air_time"] = air_total * (~contact)
        info["last_contact"] = contact
        info["foot_prev"] = foot_pos

        # ---- reward / termination ----
        R = d.xmat[self._base].reshape(3, 3)
        grav = R.T @ self._down
        v_b = R.T @ d.qvel[self._dof:self._dof + 3]
        wz = d.qvel[self._dof + 5]
        q = d.qpos[self._q_adr]
        lin_err = (v_b[0] - cmd[0]) ** 2 + (v_b[1] - cmd[1]) ** 2
        yaw_err = (wz - cmd[2]) ** 2
        # speed-relative tolerance (only if track_*_rel_ref > 0): a fixed sigma is far too tight at running speed / fast turns
        lin_den = 1.0 + (jnp.linalg.norm(cmd[:2]) / self._lin_ref) ** 2 if self._lin_ref > 0 else 1.0
        yaw_den = 1.0 + (jnp.abs(cmd[2]) / self._yaw_ref) ** 2 if self._yaw_ref > 0 else 1.0
        over = jnp.where(self._limited, jax.nn.relu(q - (self._hi - c.joint_limit_margin))
                         + jax.nn.relu((self._lo + c.joint_limit_margin) - q), 0.0)
        terms = {
            "track_lin": jnp.exp(-lin_err / (c.track_sigma_lin * lin_den)),
            "track_yaw": jnp.exp(-yaw_err / (c.track_sigma_yaw * yaw_den)),
            "upright": jnp.sum(grav[:2] ** 2),
            "action_rate": jnp.sum((a - prev_a) ** 2),
            "pose": jnp.sum((q - self._q_default) ** 2),
            "alive": jnp.float32(1.0),
            "feet_air_time": r_air,
            "foot_slip": slip,
            "torque": jnp.mean((d.actuator_force[self._aids] / self._f_scale) ** 2),
            "joint_limit": jnp.sum(over),
            "stand_still": jnp.sum(d.qvel[self._v_adr] ** 2) * (~moving),    # no stepping / joint motion on a zero command
        }
        bad = jnp.isnan(d.qpos).any() | jnp.isnan(d.qvel).any()
        tilt = -grav[2] < cfg.FALL_UP_Z
        low = d.qpos[self._qa + 2] < self._min_z
        terminated = bad | tilt | low
        truncated = (info["step"] >= c.episode_length) & ~terminated
        info["truncated"] = truncated.astype(jnp.float32)
        reward = sum(self._scales.get(k_, 0.0) * terms[k_] for k_ in _TERMS) + self._scales["termination"] * terminated
        reward = jnp.nan_to_num(reward).astype(jnp.float32)
        metrics = {k_: jnp.nan_to_num(v).astype(jnp.float32) for k_, v in terms.items()}
        metrics.update(lin_err=jnp.nan_to_num(jnp.sqrt(lin_err)), yaw_err=jnp.nan_to_num(jnp.sqrt(yaw_err)),
                       yaw_signed=jnp.nan_to_num(wz - cmd[2]).astype(jnp.float32),
                       term_tilt=tilt.astype(jnp.float32), term_height=low.astype(jnp.float32),
                       term_nan=bad.astype(jnp.float32))
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