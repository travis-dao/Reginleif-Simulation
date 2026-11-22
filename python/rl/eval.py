# Evaluate a BC checkpoint (bc_actor.pt) or an rsl_rl checkpoint (model_*.pt).
#   python python/rl/eval.py offline --ckpt python/rl/data/bc_actor.pt --data python/rl/data/mpc
#   python python/rl/eval.py rollout --ckpt logs/rsl_rl/<exp>/<run>/model_500.pt --manifest python/rl/data/mpc/manifest.json
#
# offline : one-step action MSE vs the teacher on the fixed test set (teacher-forced: history comes from the MPC), overall
#           and in the second after the push. Good for BC debugging, NOT a measure of closed-loop quality.
# rollout : closed loop in the Playground env over cfg["eval"] (command x push accel x push direction); survival + tracking.
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.config import CFG, OBS, NUM_JOINTS, rsl_rl_train_cfg, load_config
from rl.models import make_mlp


def load_policy(ckpt, cfg, device):
    """Returns fn(obs[B, obs_dim] tensor) -> actions tensor."""
    blob = torch.load(ckpt, map_location=device)
    if "bc_state_dict" in blob:
        net = make_mlp(blob["obs_dim"], blob["hidden_dims"], NUM_JOINTS, blob["activation"]).to(device)
        net.load_state_dict(blob["bc_state_dict"]); net.eval()
        return lambda obs: net(obs)
    from rsl_rl.runners import OnPolicyRunner                     # rsl_rl checkpoint
    from rl.environments.playground_env import QuadrupedEnv
    from rl.environments.rsl_wrapper import PlaygroundVecEnv
    env = PlaygroundVecEnv(QuadrupedEnv(), 1, device=device)       # only used by the runner to infer dimensions
    runner = OnPolicyRunner(env, rsl_rl_train_cfg(cfg), log_dir=None, device=device)
    runner.load(ckpt)
    pol = runner.get_inference_policy(device=device)
    return lambda obs: pol(TensorDict({"policy": obs}, batch_size=[obs.shape[0]], device=obs.device))


# ------------------------------------------------------------------ offline
def offline(args, policy, device):
    rows = defaultdict(list)
    win = int(args.window_s * OBS.policy_hz)
    for f in sorted(Path(args.data).glob("*.npz")):
        z = np.load(f)
        if len(z["obs"]) == 0:
            continue
        assert z["obs"].shape[1] == OBS.obs_dim, f"{f.name}: obs_dim {z['obs'].shape[1]} != {OBS.obs_dim}; regenerate data"
        with torch.no_grad():
            pred = policy(torch.from_numpy(z["obs"]).float().to(device)).cpu().numpy()
        err = ((pred - z["act"]) ** 2).mean(-1)
        pi = int(z["push_index"])
        rows[(float(z["push_accel"]))].append((err.mean(), err[pi:pi + win].mean() if pi >= 0 else np.nan))
    print(f"{'push accel':>10} {'n':>4} {'mse all':>10} {'mse post-push':>14}")
    for acc in sorted(rows):
        a = np.array(rows[acc]); print(f"{acc:10.1f} {len(a):4d} {a[:, 0].mean():10.5f} {np.nanmean(a[:, 1]) if acc > 0 else float('nan'):14.5f}")
    allv = np.array([v for r in rows.values() for v in r])
    print(f"overall action MSE: {allv[:, 0].mean():.5f}")


# ------------------------------------------------------------------ closed loop
def teacher_stats(manifest_path):
    """Per-episode MPC survival + tracking error, using the SAME definitions as the env metrics:
    lin_err = |v_body_xy - cmd_xy| (instantaneous), yaw_err = |gyro_z - cmd_yaw|."""
    d, out = Path(manifest_path).parent, []
    for m in json.load(open(manifest_path)):
        z = np.load(d / m["file"])
        n = len(z["obs"])
        if n == 0:
            continue
        cmd = z["cmd"]
        assert cmd.shape == (3,) and z["obs"].shape[1] == OBS.obs_dim, f"{m['file']}: stale data, regenerate with gen_mpc_data.py"
        yaw, v = z["base_yaw"], z["base_vel"]
        c, s = np.cos(yaw), np.sin(yaw)
        vxb, vyb = c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1]
        lin = np.sqrt((vxb - cmd[0]) ** 2 + (vyb - cmd[1]) ** 2)
        ys = z["obs"][:, 2] / OBS.gyro_scale - cmd[2]          # signed yaw-rate error (rad/s)
        yawe = np.abs(ys)
        out.append(dict(cmd=tuple(np.round(cmd, 4).tolist()), accel=float(z["push_accel"]), fell=bool(z["fell"]),
                        lin=float(lin.mean()), yaw=float(yawe.mean()),
                        yaw_bias=float(ys.mean()), yaw_std=float(ys.std())))
    return out


def rollout(args, policy, device, cfg):
    import jax
    import jax.numpy as jnp
    from rl.environments.playground_env import QuadrupedEnv
    from rl.environments.rsl_wrapper import PlaygroundVecEnv

    e = cfg["eval"]
    steps = int(e["episode_s"] * OBS.policy_hz)
    teacher = teacher_stats(args.manifest) if args.manifest else []
    if teacher:                                       # evaluate the student on the teacher's own commands
        cmds = sorted({t["cmd"] for t in teacher})
        if len(cmds) > args.n_cmds:
            cmds = [cmds[i] for i in np.linspace(0, len(cmds) - 1, args.n_cmds).round().astype(int)]
    else:
        cmds = [tuple(c) for c in e["cmds"]]
    grid = [(c, 0.0, 0.0) for c in cmds] + [(c, a, g) for c in cmds for a in e["accels"] for g in e["angles_deg"]]
    n = len(grid)
    print(f"{len(cmds)} commands x ({1} + {len(e['accels'])} accels x {len(e['angles_deg'])} angles) = {n} scenarios")
    env = QuadrupedEnv(config_overrides={"episode_length": steps + 5})   # no timeouts during eval: any done == fell
    vec = PlaygroundVecEnv(env, n, seed=0, device=device)

    cmd = jnp.asarray([g[0] for g in grid], jnp.float32)
    acc = jnp.asarray([g[1] for g in grid], jnp.float32)
    ang = jnp.deg2rad(jnp.asarray([g[2] for g in grid], jnp.float32))
    st = vec._state
    info = dict(st.info)
    info.update(cmd=cmd, push_acc=acc, push_ang=ang,
                next_push=jnp.where(acc > 0, int(e["push_t_s"] * OBS.policy_hz), 1 << 30).astype(jnp.int32))
    vec._state = st.replace(info=info, obs=jax.vmap(env.get_obs)(st.data, info))

    alive = np.ones(n, bool); lin = np.zeros(n); yaw = np.zeros(n); cnt = np.zeros(n); ys = np.zeros(n); ys2 = np.zeros(n)
    for t in range(steps):
        print(f"step {t}/{steps}")
        with torch.no_grad():
            act = policy(vec.get_observations()["policy"])
        vec.step(act)
        done = np.asarray(vec.last_done) > 0.5
        m = jax.device_get(vec.last_metrics)
        ok = alive & ~done
        lin[ok] += m["lin_err"][ok]; yaw[ok] += m["yaw_err"][ok]; cnt[ok] += 1
        ys[ok] += m["yaw_signed"][ok]; ys2[ok] += m["yaw_signed"][ok] ** 2
        alive &= ~done
    c_ = np.maximum(cnt, 1)
    lin, yaw, ybias = lin / c_, yaw / c_, ys / c_
    ystd = np.sqrt(np.maximum(ys2 / c_ - ybias ** 2, 0.0))

    # ---- survival vs push strength
    tcmds = set(cmds)
    print(f"\n{'push accel':>10} {'n':>4} {'student surv':>13} {'teacher surv':>13}")
    for a in sorted({g[1] for g in grid}):
        idx = [i for i, g in enumerate(grid) if g[1] == a]
        ts = [not t["fell"] for t in teacher if t["accel"] == a and t["cmd"] in tcmds]
        print(f"{a:10.1f} {len(idx):4d} {alive[idx].mean():13.2f} " + (f"{np.mean(ts):13.2f}" if ts else f"{'-':>13}"))

    # ---- nominal (unpushed) tracking per command: the BC 'bar'
    print(f"\nunpushed tracking per command  [vx, vy, yaw]")
    print(f"{'cmd':>50} {'surv':>5} {'lin stu':>8} {'lin mpc':>8} {'yaw stu':>8} {'yaw mpc':>8}")
    for c in cmds:
        i = next(i for i, g in enumerate(grid) if g[0] == c and g[1] == 0.0)
        t = [x for x in teacher if x["cmd"] == c and x["accel"] == 0.0]
        tl, ty = (f"{t[0]['lin']:8.4f}", f"{t[0]['yaw']:8.4f}") if t else (f"{'-':>8}", f"{'-':>8}")
        print(f"{str(list(c)):>50} {alive[i]:5d} {lin[i]:8.4f} {tl} {yaw[i]:8.4f} {ty}")
    print(f"\nyaw-rate error breakdown (signed, rad/s): bias = steady offset, std = oscillation")
    print(f"{'cmd':>50} {'bias stu':>9} {'bias mpc':>9} {'std stu':>8} {'std mpc':>8}")
    for c in cmds:
        i = next(i for i, g in enumerate(grid) if g[0] == c and g[1] == 0.0)
        t = [x for x in teacher if x["cmd"] == c and x["accel"] == 0.0]
        tb, tsd = (f"{t[0]['yaw_bias']:9.4f}", f"{t[0]['yaw_std']:8.4f}") if t else (f"{'-':>9}", f"{'-':>8}")
        print(f"{str(list(c)):>50} {ybias[i]:9.4f} {tb} {ystd[i]:8.4f} {tsd}")
    print(f"\noverall survival {alive.mean():.2f} over {n} scenarios ({steps} steps each)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["offline", "rollout"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="python/rl/data/mpc")
    ap.add_argument("--manifest", default=None, help="MPC test manifest.json -> adds teacher survival column")
    ap.add_argument("--window_s", type=float, default=1.0)
    ap.add_argument("--n_cmds", type=int, default=8, help="rollout: max number of teacher commands to evaluate (grid size = n_cmds x 17)")
    ap.add_argument("--config", default=None, help="model.yaml used for training (default: rl/configs/model.yaml)")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else CFG
    policy = load_policy(args.ckpt, cfg, args.device)
    offline(args, policy, args.device) if args.mode == "offline" else rollout(args, policy, args.device, cfg)


if __name__ == "__main__":
    main()