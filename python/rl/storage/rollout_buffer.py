
# storage/rollout_buffer.py

from __future__ import annotations

import torch
from types import SimpleNamespace


class RolloutBuffer:
    """
    Recurrent PPO rollout buffer.

    Stores:
        Actor observations
        Critic privileged observations
        Gaussian actions (before tanh)
        Old log probabilities
        Rewards
        Critic values
        Next-state values
        Termination/truncation flags
        GRU hidden states
        Episode-start masks

    Storage layout:
        [T, N, ...]

    T = rollout length
    N = number of environments
    """

    def __init__(
        self,
        num_steps,
        num_envs,
        obs_dim,
        privileged_dim,
        action_dim,
        hidden_dim,
        device,
        num_layers=1,
    ):

        self.num_steps = num_steps
        self.num_envs = num_envs

        self.obs_dim = obs_dim
        self.privileged_dim = privileged_dim
        self.action_dim = action_dim

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.device = torch.device(device)

        T = num_steps
        N = num_envs

        # -------------------------------------------------
        # Observations
        # -------------------------------------------------

        self.obs = torch.zeros(T, N, obs_dim, device=self.device)
        self.privileged_obs = torch.zeros(T, N, privileged_dim, device=self.device)

        # -------------------------------------------------
        # Actions and policy information
        # -------------------------------------------------

        # Gaussian samples BEFORE tanh.
        self.actions_z = torch.zeros(T, N, action_dim, device=self.device)
        self.log_probs = torch.zeros(T, N, device=self.device)

        # -------------------------------------------------
        # Rewards and values
        # -------------------------------------------------

        self.rewards = torch.zeros(T, N, device=self.device,)
        self.values = torch.zeros(T, N,device=self.device)

        # Value of the actual next observation.
        self.next_values = torch.zeros(T, N, device=self.device)

        # -------------------------------------------------
        # Episode boundaries
        # -------------------------------------------------

        self.terminated = torch.zeros(T, N, dtype=torch.bool, device=self.device)
        self.truncated = torch.zeros(T, N, dtype=torch.bool, device=self.device,)
        self.episode_starts = torch.zeros(T, N, dtype=torch.bool, device=self.device)

        # -------------------------------------------------
        # Recurrent hidden states
        # -------------------------------------------------

        # Hidden state BEFORE processing obs[t].
        #
        # Shape:
        # [T, layers, N, hidden_dim]

        self.hidden_states = torch.zeros(T, num_layers, N, hidden_dim, device=self.device)

        # -------------------------------------------------
        # GAE outputs
        # -------------------------------------------------

        self.advantages = torch.zeros(T, N, device=self.device)
        self.returns = torch.zeros(T, N, device=self.device,)

        self.ptr = 0
        self.full = False

    # =====================================================
    # Add one timestep
    # =====================================================

    def add(
        self,
        obs,
        privileged_obs,
        actions_z,
        log_probs,
        rewards,
        values,
        next_values,
        terminated,
        truncated,
        episode_starts,
        hidden_states,
    ):

        if self.ptr >= self.num_steps:
            raise RuntimeError("Rollout buffer is full")

        t = self.ptr

        # Observations.
        self.obs[t].copy_(obs.detach())
        self.privileged_obs[t].copy_(privileged_obs.detach())

        # Actions and log probabilities.
        self.actions_z[t].copy_(actions_z.detach())
        self.log_probs[t].copy_(log_probs.detach())

        # Rewards and values.
        self.rewards[t].copy_(rewards.detach().reshape(self.num_envs))
        self.values[t].copy_(values.detach().reshape(self.num_envs))
        self.next_values[t].copy_(next_values.detach().reshape(self.num_envs))

        # Episode boundaries.
        self.terminated[t].copy_(terminated.detach().bool().reshape(self.num_envs))
        self.truncated[t].copy_(truncated.detach().bool().reshape(self.num_envs))
        self.episode_starts[t].copy_(episode_starts.detach().bool().reshape(self.num_envs))
        # Hidden state before processing this observation.
        self.hidden_states[t].copy_(hidden_states.detach())

        self.ptr += 1

        if self.ptr == self.num_steps:
            self.full = True

    # =====================================================
    # Generalized Advantage Estimation
    # =====================================================

    def compute_returns_and_advantages(self, gamma=0.99, gae_lambda=0.95,):
        if not self.full:
            raise RuntimeError("Rollout buffer is not full")

        gae = torch.zeros(self.num_envs, device=self.device)

        for t in reversed(range(self.num_steps)):
            # True terminal states cannot bootstrap.
            bootstrap_mask = (~self.terminated[t]).float()

            # Both termination and truncation stop
            # GAE propagation into another episode.
            continuation_mask = (~(self.terminated[t] | self.truncated[t])).float()

            delta = (
                self.rewards[t]
                + gamma
                * self.next_values[t]
                * bootstrap_mask
                - self.values[t]
            )

            gae = (
                delta
                + gamma
                * gae_lambda
                * continuation_mask
                * gae
            )

            self.advantages[t] = gae

        self.returns = self.advantages + self.values

    # =====================================================
    # Recurrent minibatch generator
    # =====================================================

    def get_recurrent_minibatches(self, num_minibatches=8, sequence_length=32):
        if not self.full:
            raise RuntimeError("Rollout buffer is not full")

        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")

        if num_minibatches < 1:
            raise ValueError("num_minibatches must be positive")

        # -------------------------------------------------
        # Build contiguous sequence references.
        # -------------------------------------------------

        sequences = []

        for env_id in range(self.num_envs):
            for start in range(0, self.num_steps, sequence_length):
                end = min(start + sequence_length, self.num_steps)
                sequences.append((env_id, start, end))

        # Shuffle whole sequences.
        permutation = torch.randperm(len(sequences)).tolist()
        sequences = [sequences[i]for i in permutation]

        minibatch_size = max(1, (len(sequences) + num_minibatches - 1) // num_minibatches)

        # -------------------------------------------------
        # Generate minibatches.
        # -------------------------------------------------

        for batch_start in range(0, len(sequences), minibatch_size):
            batch_sequences = sequences[batch_start: batch_start + minibatch_size]

            B = len(batch_sequences)
            L = sequence_length

            # Allocate padded minibatch tensors.
            obs = torch.zeros(B, L, self.obs_dim, device=self.device)
            privileged_obs = torch.zeros(B, L, self.privileged_dim, device=self.device)
            actions_z = torch.zeros(B, L, self.action_dim, device=self.device)
            log_probs = torch.zeros(B, L, device=self.device)
            advantages = torch.zeros(B, L, device=self.device)
            returns = torch.zeros(B, L, device=self.device, )
            episode_starts = torch.zeros(B, L, dtype=torch.bool, device=self.device)

            # 1 = valid timestep, 0 = padding.
            valid_mask = torch.zeros(B, L, device=self.device)

            # Initial hidden state for every sequence.
            initial_hidden = torch.zeros(self.num_layers, B, self.hidden_dim, device=self.device)

            # -------------------------------------------------
            # Fill sequences.
            # -------------------------------------------------

            for b, (env_id, start, end) in enumerate(batch_sequences):
                length = end - start

                obs[b, :length] = self.obs[start:end, env_id]
                privileged_obs[b, :length] = self.privileged_obs[start:end, env_id]
                actions_z[b, :length] = self.actions_z[start:end, env_id]
                log_probs[b, :length] = self.log_probs[start:end, env_id]
                advantages[b, :length] = self.advantages[start:end, env_id]
                returns[b, :length] = self.returns[start:end, env_id]
                episode_starts[b, :length] = self.episode_starts[start:end, env_id]
                valid_mask[b, :length] = 1.0

                # Restore hidden state from rollout time.
                initial_hidden[:, b] = self.hidden_states[start, :, env_id]

            yield SimpleNamespace(
                obs=obs,
                privileged_obs=privileged_obs,
                actions_z=actions_z,
                log_probs=log_probs,
                advantages=advantages,
                returns=returns,
                episode_starts=episode_starts,
                initial_hidden=initial_hidden,
                valid_mask=valid_mask)

    # =====================================================
    # Clear for next rollout
    # =====================================================

    def clear(self):
        self.ptr = 0
        self.full = False