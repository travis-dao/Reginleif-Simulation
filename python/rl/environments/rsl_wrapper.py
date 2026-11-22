"""rsl_rl (>=5.x) VecEnv adapter around a Playground/MJX env.
Batches with vmap, auto-resets finished envs, hands observations to torch via DLPack (no host copy on GPU).

rsl_rl contract used here (see rsl_rl/env/vec_env.py in your installed version):
  attrs : num_envs, num_actions, max_episode_length, episode_length_buf, device, cfg
  get_observations() -> TensorDict  with groups "policy" and "privileged"
  step(actions)      -> (obs TensorDict, rewards, dones, extras)   extras["time_outs"] marks truncations
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict


def _to_torch(x, device):
    return torch.utils.dlpack.from_dlpack(x).clone().to(device)


def _to_jax(t):
    return jax.dlpack.from_dlpack(t.detach().contiguous())


class PlaygroundVecEnv(VecEnv):
    def __init__(self, env, num_envs: int, seed: int = 0, device: str = "cuda:0", clip_actions: float = 1.0):
        self.env, self.num_envs, self.device = env, num_envs, torch.device(device)
        self.num_actions = env.action_size
        self.max_episode_length = int(env._config.episode_length)
        self.clip_actions = clip_actions
        self.cfg = {}
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._key = jax.random.PRNGKey(seed)
        self.last_done = self.last_metrics = None

        vreset, vstep = jax.vmap(env.reset), jax.vmap(env.step)
        self._vreset = jax.jit(vreset)

        def _step(state, action, key):
            nxt = vstep(state, action)
            done = nxt.done > 0.5
            fresh = vreset(jax.random.split(key, num_envs))
            sel = lambda f, n: jnp.where(done.reshape((-1,) + (1,) * (n.ndim - 1)), f, n)
            merged = jax.tree_util.tree_map(sel, fresh, nxt)       # finished envs continue from a fresh reset
            means = jax.tree_util.tree_map(jnp.mean, nxt.metrics)
            return merged, nxt.reward, nxt.done, nxt.info["truncated"], nxt.metrics, means

        self._step = jax.jit(_step)
        self._state = None
        self.reset()

    def reset(self):
        self._key, k = jax.random.split(self._key)
        self._state = self._vreset(jax.random.split(k, self.num_envs))
        self.episode_length_buf.zero_()
        return self.get_observations()

    def get_observations(self) -> TensorDict:
        return TensorDict({k: _to_torch(v, self.device) for k, v in self._state.obs.items()}, batch_size=[self.num_envs], device=self.device)

    def step(self, actions: torch.Tensor):
        a = actions.detach().clamp(-self.clip_actions, self.clip_actions).float()
        self._key, k = jax.random.split(self._key)
        self._state, rew, done, trunc, metrics, means = self._step(self._state, _to_jax(a), k)
        self.last_done, self.last_metrics = done, metrics            # per-env, used by eval.py
        dones = (_to_torch(done, self.device) > 0.5)
        self.episode_length_buf = torch.where(dones, torch.zeros_like(self.episode_length_buf), self.episode_length_buf + 1)
        extras = {"time_outs": _to_torch(trunc, self.device), "log": {f"Episode/{k_}": float(v) for k_, v in jax.device_get(means).items()}}
        return self.get_observations(), _to_torch(rew, self.device), dones.long(), extras