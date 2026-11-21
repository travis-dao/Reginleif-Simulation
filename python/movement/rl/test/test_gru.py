# python python/movement/rl/test/test_gru.py

import sys
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rl.models.actor_critic import ActorCritic, ModelConfig


def main():

    device = torch.device(
        "mps" if torch.mps.is_available() else "cpu"
    )

    model = ActorCritic().to(device)
    model.eval()

    batch_size = 1

    hidden = model.actor.initial_state(batch_size, device)

    obs = torch.randn(batch_size, 1, ModelConfig.OBS_DIM, device=device,)

    print("\n--- GRU TEST ---")

    # First timestep.
    with torch.no_grad():
        action1, z1, log_prob1, hidden1 = model.actor.sample_action(obs, hidden, None)

    assert hidden1.shape == (1, 1, ModelConfig.H1_DIM)

    # Second timestep uses previous hidden state.
    with torch.no_grad():
        action2, z2, log_prob2, hidden2 = (model.actor.sample_action(obs, hidden1, None))

    assert not torch.allclose(hidden1, hidden2), "Hidden state did not change"

    print("Hidden state updates: PASSED")

    # Reset hidden state.
    hidden_reset = model.actor.initial_state(batch_size, device)

    assert torch.allclose(hidden_reset, torch.zeros_like(hidden_reset))

    print("Hidden state reset: PASSED")

    # Check output dimensions.
    assert action1.shape == (1, 1, ModelConfig.ACT_DIM)
    assert z1.shape == (1, 1, ModelConfig.ACT_DIM)

    assert torch.all(action1 <= 1.0)
    assert torch.all(action1 >= -1.0)

    print("Action dimensions and bounds: PASSED")


if __name__ == "__main__":
    main()
