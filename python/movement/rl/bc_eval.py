# python python/movement/rl/bc_eval.py

# Closed-loop eval of the BC student under pushes. Offline action-MSE is NOT the success metric; this is.
import numpy as np
import torch

from models.actor_critic import ActorCritic
from environments.env import Env

PUSH_ACCELS = [0.0, 1.0, 2.0, 3.0, 4.0]    # m/s^2 on the base for 0.1 s (same as collector)
EPISODES, SECONDS, PUSH_T = 20, 12.0, 6.0
WIN = 50                                    # 1 s windows at 50 Hz


def touchdowns(c):                          # c: [T,4] bool -> count of 0->1 transitions per foot
    return int(((~c[:-1]) & c[1:]).sum())


def rollout(env, model, device, accel, vx):
    obs, priv = env.reset()
    env.command[:] = [vx, 0.0, 0.0]; env.command_timer = 1e9     # fixed command, no resampling
    h = model.actor.initial_state(1, device)
    start = torch.ones(1, dtype=torch.bool, device=device)
    mass = float(env.model.body_subtreemass[env.body_id])
    contacts, fell = [], False
    push_step = int(PUSH_T * env.policy_hz)
    for t in range(int(SECONDS * env.policy_hz)):
        o = torch.as_tensor(obs, device=device).view(1, 1, -1)
        with torch.no_grad():
            a, _, _, h = model.actor.sample_action(o, h, start.view(1, 1), deterministic=True)
        if accel > 0 and push_step <= t < push_step + 5:
            env.data.xfrc_applied[env.body_id, :3] = [accel * mass, 0.0, 0.0]
        else:
            env.data.xfrc_applied[env.body_id, :] = 0.0
        obs, priv, r, term, trunc, info = env.step(a.view(-1).cpu().numpy())
        contacts.append(np.asarray(env._get_foot_contacts()) > 0.5)
        start = torch.tensor([term or trunc], device=device)
        if term:
            fell = True; break
        if term or trunc:
            break
    c = np.asarray(contacts)
    pre = touchdowns(c[push_step - WIN:push_step]) if len(c) > push_step else np.nan
    post = touchdowns(c[push_step:push_step + WIN]) if len(c) > push_step + WIN else np.nan
    return fell, pre, post


def main(ckpt="python/movement/rl/policy/bc_policy.pt", q_default=None):
    device = torch.device("cpu")
    q_default = np.load("python/movement/rl/datasets/run1.npz")["q_default"]
    env = Env(q_default=q_default, action_scale_rad=1.0, foot_body_names=("l0t", "l1t", "l2t", "l3t"), ground_geom_name="floor")
    model = ActorCritic(q_default=q_default, action_scale=1.0)
    model.load_state_dict(torch.load(ckpt, map_location=device)); model.eval()

    print("accel | fall rate | touchdowns/s before -> after push")
    for acc in PUSH_ACCELS:
        res = [rollout(env, model, device, acc, vx=0.2) for _ in range(EPISODES)]
        falls = np.mean([r[0] for r in res])
        pre = np.nanmean([r[1] for r in res]); post = np.nanmean([r[2] for r in res])
        print(f"{acc:4.1f}  | {falls:8.0%}  | {pre:5.2f} -> {post:5.2f}")


if __name__ == "__main__":
    main()