
import torch
import numpy as np

from models.actor_critic import ActorCritic, ModelConfig
from algorithms.ppo import PPO
from environments.env import Env
from storage.rollout_buffer import RolloutBuffer


def train():

    # 1. Configuration
    NUM_ENVS = 1
    ROLLOUT_STEPS = 128

    SEQUENCE_LENGTH = 16
    PPO_EPOCHS = 2
    NUM_MINIBATCHES = 4

    LEARNING_RATE = 3e-4
    GAMMA = 0.99
    GAE_LAMBDA = 0.95

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 2. Load BC dataset
    dataset = np.load("python/movement/rl/datasets/run1.npz")
    q_default = dataset["q_default"]

    # 3. Create environment
    env = Env(
        q_default=q_default,
        action_scale_rad=1.0,
        foot_body_names=("l0t", "l1t", "l2t", "l3t"),
        ground_geom_name="floor",
    )

    # 4. Create actor-critic
    model = ActorCritic().to(device)

    # 5. Create PPO
    ppo = PPO(model, lr=LEARNING_RATE)

    # 6. Create rollout buffer
    buffer = RolloutBuffer(
        num_steps=ROLLOUT_STEPS,
        num_envs=NUM_ENVS,
        obs_dim=ModelConfig.OBS_DIM,
        privileged_dim=ModelConfig.PRIV_DIM,
        action_dim=ModelConfig.ACT_DIM,
        hidden_dim=ModelConfig.H1_DIM,
        device=device
    )

    # 7. Initialize simulation
    obs, privileged_obs = env.reset()

    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)

    privileged_obs = torch.as_tensor(privileged_obs, dtype=torch.float32, device=device)

    hidden = model.actor.initial_state(NUM_ENVS, device)

    episode_starts = torch.ones(
        NUM_ENVS,
        dtype=torch.bool,
        device=device
    )

    # 8. Training loop
    for iteration in range(10000):

        buffer.clear()

        # Collect 128 transitions
        with torch.no_grad():

            for step in range(ROLLOUT_STEPS):

                # Save hidden state before processing observation
                hidden_before = hidden.detach().clone()

                # print("Observation shape:", obs.shape)
                # print("Hidden shape:", hidden.shape)
                # print("Episode starts shape:", episode_starts.shape)

                actions, z, log_probs, hidden = model.actor.sample_action(
                    obs.unsqueeze(0).unsqueeze(1),
                    hidden,
                    episode_starts.unsqueeze(0).unsqueeze(1)
                )

                values = model.critic(privileged_obs)

                # Convert policy action from MPS tensor to CPU NumPy
                env_action = (actions.squeeze(1).squeeze(0).detach().cpu().numpy())

                # Step MuJoCo environment
                next_obs, next_privileged, rewards, terminated, truncated, info = env.step(env_action)

                dones = terminated | truncated

                # Move outputs to training device
                next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
                next_privileged = torch.as_tensor(next_privileged, dtype=torch.float32, device=device)
                rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
                dones = torch.as_tensor(dones, dtype=torch.bool, device=device)
                terminated = torch.as_tensor(terminated, dtype=torch.bool, device=device)
                truncated = torch.as_tensor(truncated, dtype=torch.bool, device=device)
                next_values = model.critic(next_privileged)


                # Store transition
                buffer.add(
                    obs=obs,
                    privileged_obs=privileged_obs,
                    actions_z=z.squeeze(1),
                    log_probs=log_probs.squeeze(1),
                    rewards=rewards,
                    values=values,
                    next_values=next_values,
                    terminated=terminated,
                    truncated=truncated,
                    episode_starts=episode_starts,
                    hidden_states=hidden_before,
                )

                # Reset recurrent state after episode termination
                hidden = hidden * (~dones).float().view(1, NUM_ENVS, 1)

                obs = next_obs
                privileged_obs = next_privileged
                episode_starts = dones

            # Bootstrap final value
            last_value = model.critic(privileged_obs)

        # 9. Calculate GAE
        buffer.compute_returns_and_advantages(gamma=GAMMA, gae_lambda=GAE_LAMBDA)

        # 10. PPO optimization
        for epoch in range(PPO_EPOCHS):
            for batch in buffer.get_recurrent_minibatches(num_minibatches=NUM_MINIBATCHES, sequence_length=SEQUENCE_LENGTH):

                metrics = ppo.update(
                    obs=batch.obs,
                    privileged_obs=batch.privileged_obs,
                    actions_z=batch.actions_z,
                    old_log_probs=batch.log_probs,
                    advantages=batch.advantages,
                    returns=batch.returns,
                    initial_hidden=batch.initial_hidden,
                    episode_starts=batch.episode_starts,
                    valid_mask=batch.valid_mask,
                )

        # 11. Print metrics
        print(f"Iteration: {iteration}")
        print("Actor loss:", metrics["actor_loss"])
        print("Critic loss:", metrics["critic_loss"])
        print("Entropy:", metrics["entropy"])
        print("Mean reward:", buffer.rewards.mean().item())
        print("Mean advantage:", buffer.advantages.mean().item())

        # 12. Save checkpoint
        if iteration % 100 == 0:
            torch.save(model.state_dict(), f"python/movement/rl/checkpoints/checkpoint_{iteration}.pt")

    env.close()


if __name__ == "__main__":
    train()
