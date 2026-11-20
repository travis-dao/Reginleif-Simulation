
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rl.models.actor_critic import ActorCritic


def main():
    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    model = ActorCritic().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4,
    )

    B = 4
    T = 32

    obs = torch.randn(
        B, T, 45, device=device
    )

    privileged_obs = torch.randn(
        B, T, 20, device=device
    )

    actions_z = torch.randn(
        B, T, 12, device=device
    )

    advantages = torch.randn(
        B, T, device=device
    )

    returns = torch.randn(
        B, T, device=device
    )

    old_log_probs = torch.randn(
        B, T, device=device
    )

    hidden = model.actor.initial_state(
        B, device
    )

    episode_starts = torch.zeros(
        B, T,
        dtype=torch.bool,
        device=device,
    )

    # Reset memory at the beginning of each sequence.
    episode_starts[:, 0] = True

    # Forward pass.
    #
    # Adapt these calls to your ActorCritic API.
    actions, log_probs, entropy, _, _ = (
        model.actor.evaluate_actions(
            obs,
            actions_z,
            hidden,
            episode_starts,
        )
    )

    values = model.critic(
        privileged_obs
    )

    # PPO clipped objective.
    ratio = torch.exp(
        log_probs - old_log_probs
    )

    surrogate1 = (
        ratio * advantages
    )

    surrogate2 = (
        torch.clamp(ratio, 0.8, 1.2)
        * advantages
    )

    actor_loss = -torch.minimum(
        surrogate1, surrogate2
    ).mean()

    critic_loss = (
        returns - values
    ).pow(2).mean()

    entropy_loss = -entropy.mean()

    loss = (
        actor_loss
        + 0.5 * critic_loss
        + 0.01 * entropy_loss
    )

    assert torch.isfinite(loss)

    # Backpropagation.
    optimizer.zero_grad()

    loss.backward()

    # Check gradients.
    actor_grad = 0.0
    critic_grad = 0.0

    for name, param in model.named_parameters():

        if param.grad is None:
            continue

        grad_norm = param.grad.norm().item()

        assert torch.isfinite(
            param.grad
        ).all()

        if "actor" in name:
            actor_grad += grad_norm

        elif "critic" in name:
            critic_grad += grad_norm

    print("\n--- GRADIENT TEST ---")
    print("Actor gradient:", actor_grad)
    print("Critic gradient:", critic_grad)

    assert actor_grad > 0
    assert critic_grad > 0

    # Optimizer step.
    optimizer.step()

    print("Backpropagation: PASSED")
    print("Optimizer update: PASSED")


if __name__ == "__main__":
    main()
