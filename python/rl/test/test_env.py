# python python/rl/test/test_env_start.py

import sys
from pathlib import Path

import numpy as np

# Add project root to Python path.
PYTHON_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PYTHON_DIR))

from rl.environments.env import Env
import sim.config as cfg
from rl.models.actor_critic import ModelConfig

def main():
    q_default = np.load("python/rl/data/q_default.npy").astype(np.float32)

    env = Env(
        q_default=q_default,
        action_scale_rad=1.0,
        foot_body_names=cfg.FOOT_BODIES,
        ground_geom_name=cfg.GROUND_GEOM,
    )

    obs, privileged_obs = env.reset()

    print("\n--- RESET TEST ---")
    print("Actor observation:", obs.shape)
    print("Critic observation:", privileged_obs.shape)

    assert obs.shape == (ModelConfig.OBS_DIM,)
    assert privileged_obs.shape == (ModelConfig.PRIV_DIM,)

    assert np.all(np.isfinite(obs))
    assert np.all(np.isfinite(privileged_obs))

    print("Observation test PASSED")

    # Test random actions.
    print("\n--- ACTION TEST ---")

    for step in range(500):
        action = np.random.uniform(-0.2, 0.2, size=12).astype(np.float32)

        obs, privileged_obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (ModelConfig.OBS_DIM,)
        assert privileged_obs.shape == (ModelConfig.PRIV_DIM,)

        assert np.all(np.isfinite(obs))
        assert np.all(np.isfinite(privileged_obs))
        assert np.isfinite(reward)

        # Verify actuator commands.
        actual_ctrl = env.data.ctrl[env.actuator_ids]

        assert np.all(np.isfinite(actual_ctrl))

        if step % 50 == 0:
            print(f"Step {step:4d} | Reward {reward:8.3f} | Height {info['base_height']:.3f} | Command {info['command']}")

        if terminated or truncated:
            print(f"Episode ended at step {step}. Terminated={terminated}, Truncated={truncated}")
            obs, privileged_obs = env.reset()

    env.close()
    print("\nEnvironment test PASSED")


if __name__ == "__main__":
    main()
