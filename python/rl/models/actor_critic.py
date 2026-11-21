import torch
import torch.nn as nn
from torch.distributions import Normal
from dataclasses import dataclass

@dataclass
class ModelConfig:
    OBS_DIM = 45
    H1_DIM = 128
    H2_DIM = 64
    ACT_DIM = 12
    PRIV_DIM = 20

"""
robot observations (45) -> GRU (128 hidden) -> linear + elu (64) -> mean + sd -> sample action (12)
"""
class RecurrentActor(nn.Module):
    def __init__(self, obs_dim=ModelConfig.OBS_DIM, hidden1_dim=ModelConfig.H1_DIM, hidden2_dim=ModelConfig.H2_DIM, act_dim=ModelConfig.ACT_DIM):
        super().__init__()

        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.action_dim = act_dim

        self.gru = nn.GRU(
            input_size=obs_dim,
            hidden_size=hidden1_dim,
            num_layers=1,
            batch_first=True
        )

        self.head = nn.Sequential(
            nn.Linear(in_features=hidden1_dim, out_features=hidden2_dim),
            nn.ELU(),
            nn.Linear(in_features=hidden2_dim, out_features=self.action_dim)
        )

        # learnable exploration noise ~ 1 val/joint
        self.log_std = nn.Parameter(torch.full((self.action_dim,), -0.5))

    def initial_state(self, batch_size: int, device):
        return torch.zeros(1, batch_size, self.hidden1_dim, device=device)

    def forward(self, obs, hidden=None, episode_starts=None):
        """
        B = batch size, T = # timesteps
        obs: [B, T, obs_dim]
        hidden: [1, B, hidden_dim]
        episode_starts: [B, T], True when hidden must be reset
        """

        if hidden is None:
            hidden = self.initial_state(obs.shape[0], obs.device)

        outputs = []

        # proccess 1 timestep a atime so individual envs can reset their hidden states
        for t in range(obs.shape[1]):
            if episode_starts is not None:
                keep = (~episode_starts[:, t]).float()
                hidden = hidden * keep.view(1, -1, 1)

            out, hidden = self.gru(obs[:, t:t + 1], hidden)

            outputs.append(out)

        features = torch.cat(outputs, dim=1)

        mean = self.head(features)

        std = self.log_std.clamp(-5.0, 2.0).exp()
        std = std.view(1, 1, -1).expand_as(mean)

        return mean, std, hidden

    def sample_action(self, obs, hidden=None, episode_starts=None, deterministic=False):
        mean, std, next_hidden = self.forward(obs, hidden, episode_starts)

        dist = Normal(mean, std)

        if (deterministic):
            z = mean
        else:
            z = dist.sample()

        action = torch.tanh(z)

        # log prob of each pre-tanh sample
        log_prob = (dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)).sum(dim=-1)

        return action, z, log_prob, next_hidden

    def evaluate_actions(self, obs, z, hidden=None, episode_starts=None):
        """
        re-evaluate previously sampled actions for PPO

        args:
            obs: [B, T, obs_dim]
            z: [B, T, action_dim] pre-tanh actions stored during rollout
            hidden: [1, B, hidden_dim] initial GRU hidden state
            episode_starts: [B, T] episode reset masks

        returns:
            log_prob: [B, T]
            entropy: [B, T]
            next_hidden: [1, B, hidden_dim]
            mean: [B, T, action_dim]
            std: [B, T, action_dim]
        """

        mean, std, next_hidden = self.forward(obs, hidden, episode_starts)

        dist = Normal(mean, std)

        # recalculate log prob of stored pre-tanh actions
        action = torch.tanh(z)

        log_prob = (dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)).sum(dim=-1)

        # gaussian entropy (before tanh transformation)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy, next_hidden, mean, std

"""
privileged obserations (12) -> linear + elu (128) -> linear + elu (64) -> est state val (1)
"""
class Critic(nn.Module):
    def __init__(self, privileged_dim, hidden_dims=(ModelConfig.H1_DIM, ModelConfig.H2_DIM)):
        super().__init__()

        layers = []
        in_dim = privileged_dim

        for dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, dim),
                nn.ELU()
            ])
            in_dim = dim

        layers.append(nn.Linear(in_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, privledged_obs):
        return self.network(privledged_obs).squeeze(-1)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim=ModelConfig.OBS_DIM, privileged_dim=ModelConfig.PRIV_DIM, action_dim=ModelConfig.ACT_DIM, hidden1_dim=ModelConfig.H1_DIM, hidden2_dim=ModelConfig.H2_DIM, q_default=None, action_scale=0.25):
        super().__init__()

        self.actor = RecurrentActor(obs_dim, hidden1_dim, hidden2_dim, action_dim)
        self.critic = Critic(privileged_dim)

        if q_default is None:
            q_default = torch.zeros(action_dim)

        self.register_buffer("q_default", torch.as_tensor(q_default, dtype=torch.float32))

        scale = torch.as_tensor(action_scale, dtype=torch.float32)
        if scale.ndim == 0:
            scale = scale.expand(action_dim).clone()

        self.register_buffer("action_scale", scale)

    def actions_to_joint_targets(self, actions):
        return self.q_default + self.action_scale * actions