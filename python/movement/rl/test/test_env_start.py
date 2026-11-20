# python python/movement/rl/test/test_env.py

import sys
from pathlib import Path

import numpy as np

# Add project root to Python path.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from movement.rl.environments.env import Env, START_HEIGHT, STAND_HEIGHT, FALL_HEIGHT
from movement.rl.models.actor_critic import ModelConfig

# Replace these with the actual names in reginleif.xml.
FOOT_BODIES = ("l0t", "l1t", "l2t", "l3t")
GROUND_GEOM = "floor"

dataset = np.load(ROOT / "movement/rl/datasets/run1.npz")

q_default = dataset["q_default"]

env = Env(
    q_default=q_default,
    foot_body_names=FOOT_BODIES,
    ground_geom_name=GROUND_GEOM,
)

obs, privileged_obs = env.reset()

print("Initial base height:", env.data.xpos[env.body_id, 2])
print("Spawn height:", START_HEIGHT)
print("Standing height target:", STAND_HEIGHT)
print("Fall threshold:", FALL_HEIGHT)

for i in range(100):
    _, _, reward, terminated, truncated, info = env.step(
        np.zeros(ModelConfig.ACT_DIM, dtype=np.float32)
    )

    if i % 10 == 0:
        print(
            f"Step {i}: "
            f"height={info['base_height']:.3f} m, "
            f"reward={reward:.3f}, "
            f"terminated={terminated}"
        )

    if terminated or truncated:
        print("Episode ended at step", i)
        break

env.close()