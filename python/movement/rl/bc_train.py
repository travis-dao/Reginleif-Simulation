# GRU behavior cloning (stage 1).  Trains RecurrentActor only; saves a full ActorCritic state_dict.
import numpy as np
import torch

from models.actor_critic import ActorCritic, ModelConfig

DATA = "python/movement/rl/datasets/bc_run1.npz"
T, BURN_IN, BATCH = 256, 32, 32      # gait period is 1.8 s = 90 steps, so windows must span several periods
STEPS, LR = 20000, 1e-3
OBS_NOISE, PREV_ACT_NOISE = 0.01, 0.05
INIT_LOG_STD = -1.5                  # exploration std handed to PPO later (BC never trains it)


def load_episodes(path):
    d = np.load(path)
    idx = np.concatenate([[0], np.cumsum(d["ep_len"])])
    eps = [(torch.from_numpy(d["obs"][a:b]), torch.from_numpy(d["act"][a:b])) for a, b in zip(idx[:-1], idx[1:])]
    return eps, d["q_default"], float(d["action_scale"])


def sample_batch(eps, device):
    obs = torch.empty(BATCH, T, ModelConfig.OBS_DIM)
    act = torch.empty(BATCH, T, ModelConfig.ACT_DIM)
    mask = torch.ones(BATCH, T)
    starts = torch.zeros(BATCH, T, dtype=torch.bool)
    for i in range(BATCH):
        while True:
            o, a = eps[np.random.randint(len(eps))]
            if len(o) >= T:
                break
        s = np.random.randint(0, len(o) - T + 1)
        obs[i], act[i] = o[s:s + T], a[s:s + T]
        if s == 0:
            starts[i, 0] = True
        else:
            mask[i, :BURN_IN] = 0.0          # hidden began at zero mid-episode -> don't train on warm-up
    # input noise: discourages copying prev_action (causal confusion) and adds robustness
    obs[..., :-12] += OBS_NOISE * torch.randn_like(obs[..., :-12])
    obs[..., -12:] += PREV_ACT_NOISE * torch.randn_like(obs[..., -12:])
    return obs.to(device), act.to(device), mask.to(device), starts.to(device)


@torch.no_grad()
def validate(model, eps, device, n=64):
    model.eval()
    tot, cnt = 0.0, 0.0
    for _ in range(n // 16):
        obs, act, mask, starts = sample_batch_eval(eps, device)
        mean, _, _ = model.actor(obs, None, starts)
        err = (torch.tanh(mean) - act).pow(2).mean(-1)
        tot += (err * mask).sum().item(); cnt += mask.sum().item()
    model.train()
    return tot / cnt


def sample_batch_eval(eps, device):
    global BATCH, OBS_NOISE, PREV_ACT_NOISE
    saved = BATCH, OBS_NOISE, PREV_ACT_NOISE
    BATCH, OBS_NOISE, PREV_ACT_NOISE = 16, 0.0, 0.0
    try:
        return sample_batch(eps, device)
    finally:
        BATCH, OBS_NOISE, PREV_ACT_NOISE = saved


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    eps, q_default, action_scale = load_episodes(DATA)
    perm = np.random.RandomState(0).permutation(len(eps))
    n_val = max(1, len(eps) // 10)
    val_eps = [eps[i] for i in perm[:n_val]]
    train_eps = [eps[i] for i in perm[n_val:]]       # split by EPISODE, never by timestep

    model = ActorCritic(q_default=q_default, action_scale=action_scale).to(device)
    opt = torch.optim.Adam(model.actor.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)

    for step in range(STEPS):
        obs, act, mask, starts = sample_batch(train_eps, device)
        mean, _, _ = model.actor(obs, None, starts)
        pred = torch.tanh(mean)
        loss = ((pred - act).pow(2).mean(-1) * mask).sum() / mask.sum()

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0)
        opt.step(); sched.step()

        if step % 500 == 0:
            print(f"step {step}: train {loss.item():.5f}  val {validate(model, val_eps, device):.5f}")

    with torch.no_grad():
        model.actor.log_std.fill_(INIT_LOG_STD)
    torch.save(model.state_dict(), "python/movement/rl/policy/bc_policy.pt")


if __name__ == "__main__":
    main()