# GRU behavior cloning (stage 1) on the scenario .npz files from gen_test_data.py.
#   python python/rl/bc_train.py
# Trains RecurrentActor only; saves a full ActorCritic state_dict (bc_policy.pt).
import sys
from pathlib import Path

import numpy as np
import torch

PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))
from rl.models.actor_critic import ActorCritic, ModelConfig

DATA_DIR = "python/rl/data/mpc"
VAL_ACCELS = (3.0,)                   # hold out every pushed scenario with these push strengths (unseen-strength test)
T, BURN_IN, BATCH = 256, 32, 32       # gait period = 90 steps, so windows span ~3 periods
STEPS, LR = 4000, 1e-3
OBS_NOISE, PREV_ACT_NOISE = 0.01, 0.05
PUSH_WEIGHT = 5.0                     # loss weight on the 1.5 s after a push (the adaptation signal is a small residual)
PUSH_WINDOW_S = 1.5
PUSH_FOCUS = 0.5                      # fraction of training windows forced to contain a push response
INIT_LOG_STD = -1.5                   # exploration std handed to PPO later (BC never trains it)
VAL_EVERY = 500
COLD_START_FRAC = 0.25                # fraction of training windows that begin at t=0 of an episode (zero hidden, gait onset)
ONSET_STEPS = 100                     # first 2 s: standing -> gait onset; reported separately in validation

ZERO_PREV_ACT = True

def load_dataset(data_dir):
    eps, skipped, q_default, action_scale = [], 0, None, None
    for p in sorted(Path(data_dir).glob("*.npz")):
        d = np.load(p)
        hz = float(d["policy_hz"])
        n = len(d["obs"]) - (int(hz) if bool(d["fell"]) else 0)     # teacher was already failing in its last second
        if n < 50:
            skipped += 1
            continue
        idx = int(d["push_index"])
        idx = idx if 0 <= idx < n else -1
        post = int(PUSH_WINDOW_S * hz)
        w = np.ones(n, np.float32)
        if idx >= 0:
            w[idx:idx + post] = PUSH_WEIGHT
        qd, sc = d["q_default"].astype(np.float32), float(d["action_scale"])
        if q_default is None:
            q_default, action_scale = qd, sc
        assert np.allclose(qd, q_default) and sc == action_scale, f"{p.name}: q_default/action_scale differ from other files"
        eps.append(dict(name=p.stem, obs=torch.from_numpy(d["obs"][:n]), act=torch.from_numpy(d["act"][:n]),
                        w=torch.from_numpy(w), push_idx=idx, post=post, accel=float(d["push_accel"]),
                        command=str(d["command_name"]), fell=bool(d["fell"])))
    print(f"loaded {len(eps)} episodes from {data_dir} ({skipped} skipped as too short)")
    return eps, q_default, action_scale


def split(eps):
    val = [e for e in eps if e["push_idx"] >= 0 and e["accel"] in VAL_ACCELS]
    if not val:                                   # no matching scenarios: fall back to a random 10% of episodes
        rng = np.random.RandomState(0)
        val = [eps[i] for i in rng.permutation(len(eps))[:max(1, len(eps) // 10)]]
        print("VAL_ACCELS matched nothing; holding out a random 10% of episodes")
    val_names = {e["name"] for e in val}
    train = [e for e in eps if e["name"] not in val_names]
    print(f"train {len(train)} episodes, val {len(val)} episodes (held out by scenario)")
    return train, val


def sample_batch(eps, device):
    obs = torch.empty(BATCH, T, ModelConfig.OBS_DIM)
    act = torch.empty(BATCH, T, ModelConfig.ACT_DIM)
    w = torch.empty(BATCH, T)
    mask = torch.ones(BATCH, T)
    starts = torch.zeros(BATCH, T, dtype=torch.bool)
    lens = np.array([len(e["obs"]) for e in eps], dtype=np.float64)
    ok = lens >= T
    prob = ok * lens / (ok * lens).sum()
    for i in range(BATCH):
        e = eps[np.random.choice(len(eps), p=prob)]
        n = len(e["obs"])
        s = np.random.randint(0, n - T + 1)
        if np.random.rand() < COLD_START_FRAC:
            s = 0                                                      # train on exactly what validation/deployment starts with
        elif e["push_idx"] >= 0 and np.random.rand() < PUSH_FOCUS:
            lo = max(0, e["push_idx"] + e["post"] - T + 1)             # window must contain the whole push response...
            hi = min(n - T, e["push_idx"] - BURN_IN)                   # ...after the burn-in
            if lo <= hi:
                s = np.random.randint(lo, hi + 1)
        obs[i], act[i], w[i] = e["obs"][s:s + T], e["act"][s:s + T], e["w"][s:s + T]
        if s == 0:
            starts[i, 0] = True
        else:
            mask[i, :BURN_IN] = 0.0          # hidden began at zero mid-episode -> don't train on warm-up
    # input noise: discourages copying prev_action (causal confusion) and adds robustness
    obs[..., :-12] += OBS_NOISE * torch.randn_like(obs[..., :-12])
    if not ZERO_PREV_ACT:
        obs[..., -12:] += PREV_ACT_NOISE * torch.randn_like(obs[..., -12:])
    return obs.to(device), act.to(device), (mask * w).to(device), starts.to(device)


@torch.no_grad()
def validate(model, val, device):
    """Full held-out episodes from t=0 (hidden state starts at zero, exactly like deployment).
    Returns overall MSE plus MSE split into: before the push / the 1.5 s after it / the rest of the episode."""
    B, L = len(val), max(len(e["obs"]) for e in val)
    obs = torch.zeros(B, L, ModelConfig.OBS_DIM)
    act = torch.zeros(B, L, ModelConfig.ACT_DIM)
    valid = torch.zeros(B, L)
    for i, e in enumerate(val):
        n = len(e["obs"])
        obs[i, :n], act[i, :n], valid[i, :n] = e["obs"], e["act"], 1.0
    starts = torch.zeros(B, L, dtype=torch.bool)
    starts[:, 0] = True
    model.eval()
    mean, _, _ = model.actor(obs.to(device), None, starts.to(device))
    model.train()
    err = (torch.tanh(mean).cpu() - act).pow(2).mean(-1)                      # [B, L]

    t = torch.arange(L).view(1, L)
    p0 = torch.tensor([e["push_idx"] if e["push_idx"] >= 0 else L for e in val]).view(B, 1)
    p1 = p0 + torch.tensor([e["post"] for e in val]).view(B, 1)
    m = lambda mask: ((err * mask).sum() / mask.sum().clamp_min(1.0)).item()
    pre = valid * (t < p0)
    out = dict(overall=m(valid), onset=m(pre * (t < ONSET_STEPS)), pre=m(pre * (t >= ONSET_STEPS)),
               post=m(valid * ((t >= p0) & (t < p1))), late=m(valid * (t >= p1)))
    out["score"] = (out["onset"] + out["pre"] + out["post"] + out["late"]) / 4      # equal weight per segment

    # Copy baseline: predict act[t] from the previous ground-truth action act[t-1].
    # Exclude t=0 because the episode has no recorded previous action there.
    copy_pred = torch.zeros_like(act)
    copy_pred[:, 1:] = act[:, :-1]
    copy_err = (copy_pred - act).pow(2).mean(-1)
    copy_valid = valid.clone()
    copy_valid[:, 0] = 0.0
    out["copy_mse"] = ((copy_err * copy_valid).sum() / copy_valid.sum().clamp_min(1.0)).item()

    ep_err = (err * valid).sum(1) / valid.sum(1)                              # per-episode MSE
    def group(key):
        keys = sorted({str(key(e)) for e in val})
        return {k: (ep_err[[i for i, e in enumerate(val) if str(key(e)) == k]].mean().item(),
                    sum(str(key(e)) == k for e in val)) for k in keys}
    out["by_command"], out["by_fell"] = group(lambda e: e["command"]), group(lambda e: "teacher_fell" if e["fell"] else "teacher_ok")
    return out


def save(model, path):
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    sd["actor.log_std"].fill_(INIT_LOG_STD)
    torch.save(sd, path)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    eps, q_default, action_scale = load_dataset(DATA_DIR)
    if ZERO_PREV_ACT:
        for e in eps: e["obs"][:, -12:] = 0.0      # covers training AND validation
    train_eps, val_eps = split(eps)

    var = torch.cat([e["act"] for e in train_eps]).var(0).mean().item()
    print(f"baseline MSE (predict constant mean action): {var:.6f}   <- val MSE must be well below this")

    model = ActorCritic(q_default=q_default, action_scale=action_scale).to(device)
    opt = torch.optim.Adam(model.actor.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)

    best = float('inf')
    for step in range(STEPS):
        obs, act, weight, starts = sample_batch(train_eps, device)
        mean, _, _ = model.actor(obs, None, starts)
        err = (torch.tanh(mean) - act).pow(2).mean(-1)
        loss = (err * weight).sum() / weight.sum()

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0)
        opt.step(); sched.step()

        if step % VAL_EVERY == 0 or step == STEPS - 1:
            v = validate(model, val_eps, device)
            verdict = "COPY BASELINE MATCHES/BEATS MODEL" if v["copy_mse"] <= v["overall"] else "model beats copy baseline"
            print(f"step {step}: train {loss.item():.6f}  val {v['overall']:.6f}  "
                  f"copy-baseline {v['copy_mse']:.6f} ({verdict})  "
                  f"[onset {v['onset']:.6f}  pre-push {v['pre']:.6f}  post-push {v['post']:.6f}  late {v['late']:.6f}]  score {v['score']:.6f}")
            if v["score"] < best:
                best = v["score"]
                save(model, "python/rl/policy/bc_policy_best.pt")

    v = validate(model, val_eps, device)
    print("\nfinal val MSE per episode, by command:  ", {k: f"{e:.6f} (n={n})" for k, (e, n) in v["by_command"].items()})
    print("final val MSE per episode, by teacher outcome:", {k: f"{e:.6f} (n={n})" for k, (e, n) in v["by_fell"].items()})
    print(f"final copy-baseline MSE (act[t] = act[t-1]): {v['copy_mse']:.6f}")
    if v["copy_mse"] <= v["overall"]:
        print("WARNING: copy baseline matches or beats model overall val MSE; the model may mostly be learning to copy.")
    else:
        print("Model beats the copy baseline on overall validation MSE.")
    print(f"best segment-averaged val score {best:.6f} -> bc_policy_best.pt")
    save(model, "python/rl/policy/bc_policy.pt")
    print("saved bc_policy.pt (final)")


if __name__ == "__main__":
    main()