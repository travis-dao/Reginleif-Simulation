
import torch
import torch.nn as nn

OBS_DIM = 45
H1_DIM = 128
H2_DIM = 64
ACT_DIM = 12

class Actor(nn.Module):
    """
    Recurrent actor:
        Input:  45-dim observation
        GRU:    45 -> 128
        Head:   128 -> 64 -> 12
        Output: joint position targets
    """

    def __init__(self, obs_dim=OBS_DIM, hidden1_dim=H1_DIM, hidden2_dim=H2_DIM, action_dim=ACT_DIM, action_scale=0.25,
        q_default=None, q_min=None, q_max=None,
    ):
        super().__init__()

        self.hidden_dim = hidden1_dim
        self.action_dim = action_dim

        self.gru = nn.GRU(input_size=obs_dim, hidden_size=hidden1_dim, num_layers=1, batch_first=True)

        self.head = nn.Sequential(
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.ELU(),
            nn.Linear(hidden2_dim, action_dim),
        )

        # Default joint positions
        if q_default is None:
            q_default = torch.zeros(action_dim)

        if q_min is None:
            q_min = torch.full((action_dim,), -float("inf"))

        if q_max is None:
            q_max = torch.full((action_dim,), float("inf"))

        self.register_buffer("q_default", torch.as_tensor(q_default, dtype=torch.float32))
        self.register_buffer("q_min", torch.as_tensor(q_min, dtype=torch.float32))
        self.register_buffer("q_max", torch.as_tensor(q_max, dtype=torch.float32))
        self.register_buffer("action_scale",
            torch.full((action_dim,), action_scale, dtype=torch.float32)
            if isinstance(action_scale, (float, int))
            else torch.as_tensor(action_scale, dtype=torch.float32)
        )

    def initial_state(self, batch_size, device=None):
        """Create zero-initialized GRU hidden state."""
        if device is None:
            device = self.q_default.device

        return torch.zeros(1, batch_size, self.hidden_dim, device=device)

    def forward(self, obs, hidden_state=None):
        """
        Args:
            obs:
                [B, 45]       single timestep
                [B, T, 45]    sequence of observations

            hidden_state:
                [1, B, 128]   previous GRU hidden state

        Returns:
            q_target:
                [B, 12] or [B, T, 12]

            next_hidden:
                [1, B, 128]
        """

        single_step = obs.dim() == 2

        if single_step:
            obs = obs.unsqueeze(1)

        if hidden_state is None:
            hidden_state = self.initial_state(obs.shape[0], obs.device)

        features, next_hidden = self.gru(obs, hidden_state)

        raw_action = self.head(features)

        # Bound the policy output
        action = torch.tanh(raw_action)

        # Convert normalized output into joint targets
        q_target = (self.q_default + self.action_scale * action)

        # Enforce servo limits
        q_target = torch.maximum(torch.minimum(q_target, self.q_max), self.q_min)

        if single_step:
            q_target = q_target.squeeze(1)

        return q_target, next_hidden


class Critic(nn.Module):
    """
    Feedforward critic using privileged observations.

    Input:
        True base velocity
        Base height
        Foot contacts
        Foot positions

    Output:
        State value V(s)
    """

    def __init__(self, privileged_dim, hidden_dims=(H1_DIM, H2_DIM)):
        super().__init__()

        layers = []
        input_dim = privileged_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ELU(),
            ])
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, privileged_obs):
        return self.network(privileged_obs)


class ActorCritic(nn.Module):
    """
    Combined recurrent actor and feedforward critic.
    """

    def __init__(self,
        obs_dim=OBS_DIM, privileged_dim=19, action_dim=ACT_DIM, hidden_dim=H1_DIM,
        action_scale=0.25, q_default=None, q_min=None, q_max=None,
    ):
        super().__init__()
        self.actor = Actor(obs_dim, hidden_dim, action_dim, action_scale, q_default, q_min, q_max)
        self.critic = Critic(privileged_dim)

    def forward(self, obs, privileged_obs, hidden_state=None):
        q_target, next_hidden = self.actor(obs, hidden_state)

        value = self.critic(privileged_obs)

        return q_target, value, next_hidden
