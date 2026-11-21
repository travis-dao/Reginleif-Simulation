import torch
import numpy as np

from rl.models.actor_critic import ActorCritic, ModelConfig
from rl.algorithms.ppo import PPO
from rl.environments.env import Env
from rl.storage.rollout_buffer import RolloutBuffer

def train():
    # 1. configs
    ROLLOUT_STEPS = 4096
    NUM_ENVS = 24

    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    # Load BC dataset for the reference pose and action scale.
    dataset = np.load("rl/data/run1.npz")

    q_default = dataset["q_default"]

    # 2. create env
    env = Env(
        q_default=q_default,
        action_scale_rad=1.0,
        foot_body_names=("l0t", "l1t", "l2t", "l3t"),
        ground_geom_name="floor",
    )

    # 3. create NN
    model = ActorCritic().to(device)

    # 4. create PPO
    ppo = PPO(model, lr=3e-4)

    # 5. create rollout buffer
    buffer = RolloutBuffer(
        num_steps=ROLLOUT_STEPS, 
        num_envs=NUM_ENVS, 
        obs_dim=ModelConfig.OBS_DIM, 
        privileged_dim=ModelConfig.PRIV_DIM, 
        action_dim=ModelConfig.ACT_DIM, 
        hidden_dim=ModelConfig.H1_DIM, 
        device=device)

    # 6. init sim
    obs, privileged_obs = env.reset()

    hidden = model.actor.initial_state(NUM_ENVS, device)

    episode_starts = torch.ones(NUM_ENVS, dtype=torch.bool, device=device)

    # 7. training loop

    for iteration in range(10000):
        buffer.clear()

        # collect rollout
        with torch.no_grad():
            for step in range(ROLLOUT_STEPS):

                #save hidden state before processing obs
                hidden_before = hidden.detach().clone()

                actions, z, log_probs, hidden = model.actor.sample_action(
                    obs.unsqueeze(0).unsqueeze(1),
                    hidden,
                    episode_starts.unsqueeze(0).unsqueeze(1)
                )

                values = model.critic(privileged_obs)

                q_targets = model.actions_to_joint_targets(actions.squeeze(1))

                next_obs, next_privileged, rewards, terminated, truncated, info = env.step(q_targets)
                next_values = model.critic(next_privileged)

                dones = terminated | truncated

                # Move outputs to training device
                next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
                next_privileged = torch.as_tensor(next_privileged, dtype=torch.float32, device=device)
                rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
                dones = torch.as_tensor(dones, dtype=torch.bool, device=device)
                terminated = torch.as_tensor(terminated, dtype=torch.bool, device=device)
                truncated = torch.as_tensor(truncated, dtype=torch.bool, device=device)
                next_values = model.critic(next_privileged)


                # store rollout data
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

                # reset hidden state on episode termination
                hidden = hidden * (~dones).float().view(1, NUM_ENVS, 1)

                obs = next_obs
                privileged_obs = next_privileged
                episode_starts = dones
            
            # bootstrap final val
            last_value = model.critic(privileged_obs)


        # 8. calc GAE
        buffer.compute_returns_and_advantages(last_value=last_value, gamma=0.99, gae_lambda=0.95)

        # 9. PPO optimization
        for epoch in range(5):
            for batch in buffer.get_recurrent_minibatches(num_minibatches=8):
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

        # 10. save checkpoint
        if iteration % 100 == 0:
            torch.save(model.state_dict(), f"checkpoint_{iteration}.pt")

            print(f"Iteration {iteration}: Loss = {metrics["loss"]:.4f}")

    env.close()

    pass

if __name__ == "__main__":
    train()