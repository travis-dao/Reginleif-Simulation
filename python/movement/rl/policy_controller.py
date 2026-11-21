"""Learned (BC / DAgger / PPO) controller: replaces LegController + Core on the same sim / servo interface.

    imu.update(dt)
    policy.set_command(v_cmd_mps, yaw_rate_cmd)      # whenever the command changes
    policy.step(dt, imu.read())                      # every tick: runs the GRU at policy_hz, writes servo targets via LegDriver
    mujoco.mj_step(model, data)

    policy.engage()                                  # once, when the robot is standing (before that, step() holds q_default)

Observation (45), identical to Env.get_observation / the BC dataset:
    gyro_body*GYRO_SCALE (3) | gravity_body (3) | [v_cmd, 0, yaw_rate_cmd]*CMD_SCALE (3) |
    q - q_default (12) | qd*QD_SCALE (12) | prev_action (12)
"""
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from movement.leg.leg_config import Theta3
from movement.leg.movement import LegDriver, RAD_TO_DEG
from movement.motion.imu import ImuData

from rl.models.actor_critic import ActorCritic, ModelConfig
from rl.environments.env import GYRO_SCALE, CMD_SCALE_LIN, CMD_SCALE_ANG, QD_SCALE   # single source of truth with training

NUM_LEGS = 4
JointState = Tuple[np.ndarray, np.ndarray]          # (q [12] rad, qd [12] rad/s), order leg0_coxa, leg0_femur, ..., leg3_tibia


def gravity_in_body(rpy_rad: np.ndarray) -> np.ndarray:
    """Unit gravity direction in the body frame from ZYX roll/pitch/yaw. Equals R.T @ [0, 0, -1]; yaw drops out."""
    r, p = float(rpy_rad[0]), float(rpy_rad[1])
    return np.array([np.sin(p), -np.sin(r) * np.cos(p), -np.cos(r) * np.cos(p)])


class PolicyController:
    def __init__(
        self,
        model,                                       # MjModel (used for the servo driver and the default joint reader)
        data,                                        # MjData
        checkpoint: str,
        policy_hz: float = 50.0,
        deterministic: bool = True,
        v_limit_mps: Optional[float] = None,         # clip commands to the range the policy was trained on
        yaw_limit_rad_s: Optional[float] = None,
        joint_state_fn: Optional[Callable[[], JointState]] = None,   # hardware: servo feedback; default: MuJoCo qpos/qvel
        device: str = "mps",
    ):
        self.device = torch.device(device)
        self.policy_period = 1.0 / policy_hz
        self.deterministic = deterministic
        self.v_limit, self.yaw_limit = v_limit_mps, yaw_limit_rad_s

        # ---- policy -------------------------------------------------------------------------------------------
        self.net = ActorCritic()
        self.net.load_state_dict(torch.load(checkpoint, map_location="cpu"))   # also restores q_default / action_scale buffers
        self.net.to(self.device).eval()
        self.q_default = self.net.q_default.cpu().numpy().astype(np.float64)
        self.hidden = self.net.actor.initial_state(1, self.device)
        self.prev_action = np.zeros(ModelConfig.ACT_DIM, np.float32)

        # ---- servo driver, same object LegController uses --------------------------------------------------------
        self.driver = LegDriver(model, data, debug=False)
        self.target_angles: List[Theta3] = [Theta3() for _ in range(NUM_LEGS)]
        self._joint_state_fn = joint_state_fn or self._mujoco_joint_reader(model, data)

        # ---- command (same attribute names as Core) ---------------------------------------------------------------------
        self.v_cmd_mps = 0.0
        self.yaw_rate_cmd = 0.0

        self.engaged = False
        self._t_since = 1e9                           # run the policy on the first step after engage()
        self.last_action: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ command
    def set_command(self, v_cmd_mps: float, yaw_rate_cmd: float) -> None:
        """Robot command: forward speed (m/s, negative = backward) and yaw rate (rad/s, + = left)."""
        if self.v_limit is not None:
            v_cmd_mps = float(np.clip(v_cmd_mps, -self.v_limit, self.v_limit))
        if self.yaw_limit is not None:
            yaw_rate_cmd = float(np.clip(yaw_rate_cmd, -self.yaw_limit, self.yaw_limit))
        self.v_cmd_mps, self.yaw_rate_cmd = float(v_cmd_mps), float(yaw_rate_cmd)

    # ------------------------------------------------------------------ policy
    def reset(self) -> None:
        """Clear the policy memory (GRU hidden state, prev_action)."""
        self.hidden = self.net.actor.initial_state(1, self.device)
        self.prev_action[:] = 0.0
        self.last_action = None
        self._t_since = 1e9

    def engage(self) -> None:
        """Hand the servos to the policy. Call when the robot is standing: the BC data starts from a standing robot
        with zero hidden state and zero prev_action."""
        self.reset()
        self.engaged = True

    def disengage(self) -> None:
        self.engaged = False

    def step(self, dt_s: float, imu_data: ImuData) -> Optional[np.ndarray]:
        """Control tick. Before engage(): holds the q_default stand pose. After: one GRU step every 1/policy_hz s
        (targets are held in between, like Env's decimation). Returns the new action when one was computed."""
        if not self.engaged:
            self._write_targets(self.q_default)
            return None
        self._t_since += dt_s
        if self._t_since < self.policy_period - 1e-9:
            return None
        if not imu_data.valid:                        # no usable orientation: keep the last targets, try again next tick
            return None
        self._t_since = 0.0

        obs = torch.as_tensor(self.build_observation(imu_data), dtype=torch.float32, device=self.device).view(1, 1, -1)
        with torch.no_grad():
            action, _, _, self.hidden = self.net.actor.sample_action(obs, self.hidden, deterministic=self.deterministic)
            q_target = self.net.actions_to_joint_targets(action.squeeze(1))
        action = action.view(-1).cpu().numpy()
        self._write_targets(q_target.view(-1).cpu().numpy().astype(np.float64))

        self.prev_action = action.astype(np.float32)
        self.last_action = action
        return action

    def build_observation(self, imu_data: ImuData) -> np.ndarray:
        """45-dim observation from IMU + joint feedback + command (matches Env.get_observation)."""
        q, qd = self._joint_state_fn()
        gyro_body = np.asarray(imu_data.gyro_rad_s, dtype=np.float64)           # already body frame
        gravity_body = gravity_in_body(np.asarray(imu_data.rpy_rad, dtype=np.float64))
        cmd = np.array([self.v_cmd_mps, 0.0, self.yaw_rate_cmd]) * np.array([CMD_SCALE_LIN, CMD_SCALE_LIN, CMD_SCALE_ANG])
        obs = np.concatenate([
            gyro_body * GYRO_SCALE,
            gravity_body,
            cmd,
            np.asarray(q, dtype=np.float64) - self.q_default,
            np.asarray(qd, dtype=np.float64) * QD_SCALE,
            self.prev_action,
        ]).astype(np.float32)
        assert obs.shape == (ModelConfig.OBS_DIM,)
        return obs

    # ------------------------------------------------------------------ servo output
    def _write_targets(self, q_target_rad: np.ndarray) -> None:
        deg = q_target_rad * RAD_TO_DEG
        for i in range(NUM_LEGS):
            self.target_angles[i] = Theta3(coxa=float(deg[3 * i]), femur=float(deg[3 * i + 1]), tibia=float(deg[3 * i + 2]))
            self.driver.set_angles(i, self.target_angles[i])

    # ------------------------------------------------------------------ joint feedback
    @staticmethod
    def _mujoco_joint_reader(model, data) -> Callable[[], JointState]:
        """Sim feedback: resolve each servo's joint through the same actuator names LegDriver writes to."""
        d = LegDriver(model, data)
        q_adr, v_adr = [], []
        for leg in LegDriver.LEG_NAMES:
            for j in LegDriver.JOINTS:
                jid = int(model.actuator_trnid[d.act_id[f"{leg}_{j}"], 0])
                q_adr.append(int(model.jnt_qposadr[jid]))
                v_adr.append(int(model.jnt_dofadr[jid]))
        return lambda: (data.qpos[q_adr].copy(), data.qvel[v_adr].copy())