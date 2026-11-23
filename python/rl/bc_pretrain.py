# Behavior-clone the MPC: supervised obs -> action on the .npz files from gen_mpc_data.py.
#   python python/rl/bc_pretrain.py --train_dir python/rl/data/mpc_train_relabeled --val_dir python/rl/data/mpc_relabeled --out python/rl/data/bc_actor.pt
# The checkpoint is loaded into the rsl_rl PPO actor by train.py --bc_ckpt.
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.config import CFG, OBS, NUM_JOINTS
from rl.models import make_mlp


def load_dir(d):
    X, Y = [], []
    for f in sorted(Path(d).glob("*.npz")):
        z = np.load(f)
        if len(z["obs"]) == 0:
            continue
        if z["obs"].shape[1] != OBS.obs_dim or int(z["history_len"]) != OBS.history_len:
            raise ValueError(f"{f.name}: obs_dim={z['obs'].shape[1]} history_len={int(z['history_len'])} but model.yaml gives "
                             f"{OBS.obs_dim}/{OBS.history_len}. Regenerate the data (gen_mpc_data.py) or fix the config.")
        X.append(z["obs"]); Y.append(z["act"])
    if not X:
        raise FileNotFoundError(f"no usable .npz in {d}")
    return np.concatenate(X), np.concatenate(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", default=None, help="e.g. the fixed MPC test set; otherwise a random val_frac split is used")
    ap.add_argument("--out", default="python/rl/data/bc_actor.pt")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    bc, a = CFG["bc"], CFG["actor"]

    X, Y = load_dir(args.train_dir)
    if args.val_dir:
        Xv, Yv = load_dir(args.val_dir)
    else:
        idx = np.random.default_rng(0).permutation(len(X)); n = int(len(X) * bc["val_frac"])
        Xv, Yv, X, Y = X[idx[:n]], Y[idx[:n]], X[idx[n:]], Y[idx[n:]]
    print(f"train {len(X)} samples, val {len(Xv)} samples, obs_dim {OBS.obs_dim}")
    dev = torch.device(args.device)
    X, Y, Xv, Yv = (torch.from_numpy(t).float().to(dev) for t in (X, Y, Xv, Yv))

    net = make_mlp(OBS.obs_dim, a["hidden_dims"], NUM_JOINTS, a["activation"]).to(dev)
    with torch.no_grad():
        # v_y is always 0 in MPC data, so these weights get no gradient. Zero them so the BC policy ignores v_y and
        # PPO can start using it later without random-init noise leaking in.
        net[0].weight[:, OBS.vy_idx] = 0.0
    h0, noise_std, drop_p = OBS.hist_start, bc["history_noise_std"], bc["history_dropout_p"]
    opt = torch.optim.AdamW(net.parameters(), lr=bc["lr"], weight_decay=bc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=bc["epochs"])
    best = float("inf")
    for ep in range(bc["epochs"]):
        net.train()
        perm = torch.randperm(len(X), device=dev)
        for i in range(0, len(X), bc["batch_size"]):
            b = perm[i:i + bc["batch_size"]]
            xb = X[b].clone()
            if OBS.history_len > 0:                      # anti-copycat: perturb / hide the action history
                if noise_std > 0:
                    xb[:, h0:] += noise_std * torch.randn_like(xb[:, h0:])
                if drop_p > 0:
                    keep = (torch.rand(len(xb), 1, device=dev) >= drop_p).float()
                    xb[:, h0:] *= keep
            loss = nn.functional.mse_loss(net(xb), Y[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            val = nn.functional.mse_loss(net(Xv), Yv).item()
        if val < best:
            best = val
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"bc_state_dict": {k: v.cpu() for k, v in net.state_dict().items()}, "obs_dim": OBS.obs_dim,
                        "hidden_dims": list(a["hidden_dims"]), "activation": a["activation"],
                        "history_len": OBS.history_len, "val_mse": val}, args.out)
        print(f"epoch {ep + 1:3d}/{bc['epochs']}  train_last_batch {loss.item():.5f}  val_mse {val:.5f}  best {best:.5f}")
    print(f"saved best checkpoint -> {args.out}")


if __name__ == "__main__":
    main()