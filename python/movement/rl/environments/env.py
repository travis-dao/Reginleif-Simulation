# environments/env.py

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PYTHON_DIR = Path(__file__).resolve().parents[3]
PROJECT_ROOT = PYTHON_DIR.parent

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from sim.util.build_sim_xml import build_sim_xml
import sim.config as cfg
from movement.motion.imu import Imu
from movement.rl.models.actor_critic import ModelConfig


GYRO_SCALE = 0.25
CMD_SCALE_LIN = 2.0
CMD_SCALE_ANG = 0.25
QD_SCALE = 0.05


class Env:
    """
    Single-robot MuJoCo environment for recurrent PPO.

    Actor:
        Observation: 45
        Action: 12

    Critic:
        Privileged observation: 20

    Control:
        Direct position actuator control.

    No MPC, gait controller, or teacher policy.
    """

    def __init__(
        self,
        q_default,
        action_scale_rad=1.0,
        policy_hz=50.0,
        episode_seconds=20.0,
        foot_body_names=None,
        ground_geom_name=None,
        seed=0,
        command_interval=(3.0, 6.0),
        stand_probability=0.15,
        command_vx=(-0.6, 0.6),
        command_vy=(-0.25, 0.25),
        command_yaw=(-0.8, 0.8),
    ):

        self.rng = np.random.default_rng(seed)

        self.q_default = np.asarray(q_default, dtype=np.float32).reshape(ModelConfig.ACT_DIM)

        self.action_scale = np.broadcast_to(np.asarray(action_scale_rad, dtype=np.float32), (ModelConfig.ACT_DIM,)).copy()

        self.policy_hz = float(policy_hz)
        self.episode_seconds = float(episode_seconds)

        self.command_interval = command_interval
        self.stand_probability = stand_probability

        self.command_vx = command_vx
        self.command_vy = command_vy
        self.command_yaw = command_yaw

        # -------------------------------------------------
        # Build MuJoCo model
        # -------------------------------------------------

        build_sim_xml(cfg.SRC, cfg.DST, cfg.START_HEIGHT, cfg.KP, cfg.FORCE)
        self.model = mujoco.MjModel.from_xml_path(cfg.DST)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        # -------------------------------------------------
        # Base body and free joint
        # -------------------------------------------------

        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")

        if self.body_id < 0:
            raise ValueError("MuJoCo body 'base' not found")

        base_joint = int(self.model.body_jntadr[self.body_id])

        if base_joint < 0:
            raise ValueError("Base body has no joint")

        if self.model.jnt_type[base_joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("Expected a free joint on base")

        self.base_qadr = int(self.model.jnt_qposadr[base_joint])

        # -------------------------------------------------
        # Resolve actuators in exact BC policy order
        # -------------------------------------------------

        self.actuator_ids = []
        self.joint_ids = []
        self.joint_names = []

        for leg in cfg.LEG_NAMES:
            for joint in cfg.JOINTS:

                actuator_name = f"{leg}_{joint}_act"

                aid = mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name,)

                if aid < 0:
                    raise KeyError(f"Actuator '{actuator_name}' not found")

                if self.model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_JOINT:
                    raise ValueError(f"{actuator_name} is not a joint actuator")

                jid = int(self.model.actuator_trnid[aid, 0])

                self.actuator_ids.append(aid)
                self.joint_ids.append(jid)
                self.joint_names.append(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid))

        self.actuator_ids = np.asarray(self.actuator_ids, dtype=np.int32)

        self.joint_ids = np.asarray(self.joint_ids, dtype=np.int32)

        assert len(self.actuator_ids) == ModelConfig.ACT_DIM

        self.q_adr = np.asarray([self.model.jnt_qposadr[j] for j in self.joint_ids])

        self.v_adr = np.asarray([self.model.jnt_dofadr[j]for j in self.joint_ids])

        # -------------------------------------------------
        # Actuator limits
        # -------------------------------------------------

        self.ctrl_limited = (self.model.actuator_ctrllimited[self.actuator_ids].copy())

        self.ctrl_range = (self.model.actuator_ctrlrange[self.actuator_ids].copy())

        # Validate default pose.
        for i in range(ModelConfig.ACT_DIM):
            if self.ctrl_limited[i]:
                lo, hi = self.ctrl_range[i]
                if not lo <= self.q_default[i] <= hi:
                    raise ValueError(f"Default position for {self.joint_names[i]} is outside limits")

        self.imu = Imu(self.model, self.data, "base")

        # -------------------------------------------------
        # Policy timing
        # -------------------------------------------------

        self.sim_dt = float(self.model.opt.timestep)

        self.decimation = max(1, int(round(1.0 / (self.policy_hz * self.sim_dt))))

        self.dt = self.decimation * self.sim_dt

        self.actual_policy_hz = 1.0 / self.dt

        self.max_episode_steps = int(self.episode_seconds / self.dt)

        # -------------------------------------------------
        # Privileged observation: foot identification
        # -------------------------------------------------

        if foot_body_names is None:
            raise ValueError("Pass the four actual foot body names from your MuJoCo XML.")

        if len(foot_body_names) != 4:
            raise ValueError("Expected four foot body names")

        self.foot_body_ids = []

        for name in foot_body_names:

            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

            if bid < 0:
                raise KeyError(f"Foot body '{name}' not found")

            self.foot_body_ids.append(bid)

        self.foot_body_ids = np.asarray(self.foot_body_ids)

        if ground_geom_name is None:
            raise ValueError("Pass the actual ground geom name from your MuJoCo XML.")

        self.ground_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, ground_geom_name)

        if self.ground_geom_id < 0:
            raise KeyError(f"Ground geom '{ground_geom_name}' not found")

        # Collect foot collision geoms, including descendants.
        self.foot_geom_ids = [set() for _ in range(4)]

        for gid in range(self.model.ngeom):

            geom_body = int(self.model.geom_bodyid[gid])

            for i, foot_body in enumerate(self.foot_body_ids):
                body = geom_body

                while body > 0:
                    if body == foot_body:
                        self.foot_geom_ids[i].add(gid)
                        break

                    body = int(self.model.body_parentid[body])

        # -------------------------------------------------
        # Episode state
        # -------------------------------------------------

        self.step_count = 0
        self.episode_reward = 0.0

        self.command = np.zeros(3, dtype=np.float32)

        self.command_timer = 0.0

        self.prev_action = np.zeros(ModelConfig.ACT_DIM, dtype=np.float32)

        self.last_q_target = self.q_default.copy()

        print(f"MuJoCo initialized | policy rate: {self.actual_policy_hz:.2f} Hz | sim dt: {self.sim_dt:.5f} s")
        print("Joint order:", self.joint_names)

    # =====================================================
    # Commands
    # =====================================================

    def _sample_command(self):
        if self.rng.random() < self.stand_probability:
            self.command[:] = 0.0

        else:
            self.command = np.array([
                self.rng.uniform(*self.command_vx),
                self.rng.uniform(*self.command_vy),
                self.rng.uniform(*self.command_yaw),
            ], dtype=np.float32)

        self.command_timer = self.rng.uniform(*self.command_interval)

    def _update_command(self):
        self.command_timer -= self.dt
        if self.command_timer <= 0.0:
            self._sample_command()

    # =====================================================
    # Actor observation: 45 dimensions
    # =====================================================

    def get_observation(self):
        d = self.data

        imu = self.imu.read()

        R = d.xmat[self.body_id].reshape(3, 3)

        # Gravity in body coordinates.
        gravity_body = (R.T @ np.array([0.0, 0.0, -1.0]))

        q = d.qpos[self.q_adr]
        qd = d.qvel[self.v_adr]

        frame = np.concatenate([
            np.asarray(
                imu.gyro_rad_s) * GYRO_SCALE, 
                gravity_body,
                [
                    self.command[0] * CMD_SCALE_LIN,
                    self.command[1] * CMD_SCALE_LIN,
                    self.command[2] * CMD_SCALE_ANG,
                ],
                q - self.q_default,
                qd * QD_SCALE,
                self.prev_action,
        ]).astype(np.float32)

        assert frame.shape == (ModelConfig.OBS_DIM,)
        return frame

    # =====================================================
    # Critic observation: 20 privileged dimensions
    # =====================================================

    def _get_base_velocity(self):
        velocity = np.zeros(6, dtype=np.float64)

        # MuJoCo spatial velocity:
        # [angular velocity, linear velocity].
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.body_id, velocity, 0)

        R = self.data.xmat[self.body_id].reshape(3, 3)

        linear_velocity_world = velocity[3:6]
        linear_velocity_body = R.T @ linear_velocity_world

        angular_velocity_world = velocity[:3]
        angular_velocity_body = R.T @ angular_velocity_world

        return linear_velocity_body, angular_velocity_body

    def _get_foot_contacts(self):
        contacts = np.zeros(4, dtype=np.float32)

        for i, foot_geoms in enumerate(self.foot_geom_ids):
            for j in range(self.data.ncon):
                contact = self.data.contact[j]
                g1 = int(contact.geom1)
                g2 = int(contact.geom2)
                if g1 == (self.ground_geom_id and g2 in foot_geoms) or (g2 == self.ground_geom_id and g1 in foot_geoms):
                    contacts[i] = 1.0
                    break

        return contacts

    def get_privileged_observation(self):
        d = self.data

        base_velocity, _ = self._get_base_velocity()
        base_height = float(d.xpos[self.body_id, 2])
        contacts = self._get_foot_contacts()

        R = d.xmat[self.body_id].reshape(3, 3)

        base_position = d.xpos[self.body_id]
        foot_positions = []

        for bid in self.foot_body_ids:
            position_world = d.xpos[bid] - base_position
            position_body = R.T @ position_world
            foot_positions.extend(position_body)

        privileged = np.concatenate([
            base_velocity, # 3
            [base_height], # 1
            contacts, # 4
            np.asarray(foot_positions), # 12
        ]).astype(np.float32)

        assert privileged.shape == (ModelConfig.PRIV_DIM, )
        return privileged

    # =====================================================
    # Reward
    # =====================================================

    def compute_reward(self, action):
        velocity_body, angular_velocity_body = self._get_base_velocity()

        # Desired linear velocity.
        velocity_error = (velocity_body[:2] - self.command[:2])

        velocity_reward = np.exp(-2.0 * np.sum(velocity_error**2))

        # Desired yaw rate.
        yaw_error = angular_velocity_body[2] - self.command[2]

        yaw_reward = np.exp( -2.0 * yaw_error**2)

        # Uprightness.
        R = self.data.xmat[self.body_id].reshape(3, 3)

        upright_reward = max(0.0, float(R[2, 2]))

        # Base height.
        height = self.data.xpos[self.body_id, 2]

        height_reward = np.exp(-10.0 * (height - cfg.STAND_HEIGHT)**2)

        # Penalize action changes.
        action_rate_penalty = np.mean((action - self.prev_action)**2)

        # Penalize excessive joint velocity.
        joint_velocity_penalty = np.mean(self.data.qvel[self.v_adr]**2)

        reward = (
            2.0 * velocity_reward
            + 0.5 * yaw_reward
            + 0.5 * upright_reward
            + 0.5 * height_reward
            - 0.02 * action_rate_penalty
            - 0.0005 * joint_velocity_penalty
        )

        return float(reward)

    # =====================================================
    # Termination
    # =====================================================

    def has_fallen(self):
        R = self.data.xmat[self.body_id].reshape(3, 3)

        up_z = R[2, 2]

        height = self.data.xpos[self.body_id, 2]

        return (up_z < cfg.FALL_UP_Z or height < cfg.FALL_HEIGHT)

    # =====================================================
    # Reset
    # =====================================================

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        # Restore the default joint pose.
        self.data.qpos[self.q_adr] = self.q_default

        # Initial actuator targets.
        self.data.ctrl[self.actuator_ids] = self.q_default

        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.episode_reward = 0.0

        self.prev_action.fill(0.0)

        self._sample_command()

        obs = self.get_observation()

        privileged_obs = (self.get_privileged_observation())

        return obs, privileged_obs

    # =====================================================
    # Environment step
    # =====================================================

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(ModelConfig.ACT_DIM)

        action = np.clip(action, -1.0, 1.0)

        # Convert normalized action to radians.
        q_target = (self.q_default + self.action_scale * action)

        # Enforce physical actuator limits.
        for i in range(ModelConfig.ACT_DIM):
            if self.ctrl_limited[i]:
                lo, hi = self.ctrl_range[i]
                q_target[i] = np.clip(q_target[i], lo, hi)

        # Apply directly to MuJoCo.
        self.data.ctrl[self.actuator_ids] = q_target

        # Advance simulation.
        for _ in range(self.decimation):
            mujoco.mj_step(self.model,self.data)
            self.imu.update()

        self.step_count += 1

        # Evaluate reward against the command
        # that was visible to the actor.
        reward = self.compute_reward(action)

        terminated = self.has_fallen()

        if terminated:
            reward -= 5.0

        truncated = self.step_count >= self.max_episode_steps and not terminated

        self.episode_reward += reward

        # Update previous action for next observation.
        self.prev_action = action.copy()

        self.last_q_target = q_target.copy()

        # Sample next command only after current reward.
        self._update_command()

        obs = self.get_observation()

        privileged_obs = self.get_privileged_observation()

        info = {
            "episode_reward": self.episode_reward,
            "episode_length": self.step_count,
            "command": self.command.copy(),
            "base_height": float(
                self.data.xpos[self.body_id, 2]
            ),
            "q_target": q_target.copy(),
        }

        return (
            obs,
            privileged_obs,
            reward,
            terminated,
            truncated,
            info,
        )

    def close(self):
        pass
