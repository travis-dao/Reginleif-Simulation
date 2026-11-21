import torch
import torch.nn as nn
from torch.distributions import Normal

class PPO:
    def __init__(self, model, lr=3e-4, clip_coef=0.2, value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5):
        self.model = model
        self.optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=lr)

        self.clip_coef = clip_coef
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    @staticmethod
    def masked_mean(x, mask):
        mask = mask.float()
        return (x * mask).sum() / mask.sum().clamp_min(1.0)

    def update(self, obs, privileged_obs, actions_z, old_log_probs, advantages, returns, initial_hidden, episode_starts, valid_mask=None):
        """
        Sequence minibatch inputs:

        obs:              [B, T, 45]
        privileged_obs:   [B, T, privileged_dim]
        actions_z:        [B, T, 12] (pre-tanh samples)
        old_log_probs:    [B, T]
        advantages:       [B, T]
        returns:          [B, T]
        initial_hidden:   [1, B, 128]
        episode_starts:   [B, T]
        valid_mask:       [B, T], optional padding mask

        Sequences retain their temporal order.
        """

        if valid_mask is None:
            valid_mask = torch.ones_like(advantages)

        mask = valid_mask.float()

        # normalize advatanges over valid timestep
        adv_mean = self.masked_mean(advantages, mask)

        adv_var = self.masked_mean((advantages - adv_mean).pow(2), mask)

        advantages = (advantages - adv_var) / torch.sqrt(adv_var + 1e-8)

        # re-eval actions using current actor
        mean, std, _ = self.model.actor(obs, initial_hidden, episode_starts)

        dist = Normal(mean, std)

        new_log_probs = dist.log_prob(actions_z).sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)

        # prob ratio btwn new and old policies
        ratio = torch.exp(new_log_probs - old_log_probs)

        # ppo clipped surrogate objectives
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * advantages

        actor_loss = -self.masked_mean(torch.minimum(surr1, surr2), mask)

        # critic loss
        values = self.model.critic(privileged_obs)

        critic_loss = self.masked_mean((values - returns).pow(2), mask)

        # exploration bonus
        entropy_bonus = self.masked_mean(entropy, mask)

        # combined objective
        loss = (actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy_bonus)

        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()

        return {
            "loss": loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy_bonus.item()
        }