#!/usr/bin/env python3
"""Behavior-clone an MPC teacher into a recurrent GRU actor.

Actor: 45 -> GRU(128) -> 64 -> 12, with tanh-normalized joint actions.
Each dataset observation currently stores 3 x 45 stacked frames; this script uses
only the newest frame and learns temporal context in the GRU hidden state.

Put this file in python/movement/rl/ (next to datasets/) and run:
    python python/movement/rl/train.py --epochs 400

Outputs a PyTorch checkpoint and a C++ header containing GRU/head weights.
The C++ inference code must implement PyTorch's GRU equations and preserve/reset
its 128-element hidden state across policy ticks / episode boundaries.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
FRAME_DIM, HIDDEN, HEAD_HIDDEN, ACT_DIM = 45, 128, 64, 12
PRIVILEGED_DIM = 20  # base velocity(3), base height(1), contacts(4), foot positions(12)
META_EXACT = ["obs_history", "obs_frame_dim", "act_dim", "joint_names"]
META_FLOAT = ["gyro_scale", "cmd_scale_lin", "cmd_scale_ang", "qd_scale", "action_scale_rad", "policy_hz"]


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_datasets(data_dir: Path, skip_s: float):
    files = sorted(data_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"No .npz files found in {data_dir}")
    ref_meta, q_default = None, None
    X, Y, G = [], [], []
    for fi, f in enumerate(files):
        with np.load(f) as d:
            meta = json.loads(str(d["meta"]))
            if ref_meta is None:
                ref_meta, q_default = meta, d["q_default"].astype(np.float32)
            else:
                for k in META_EXACT:
                    if meta[k] != ref_meta[k]:
                        raise SystemExit(f"{f.name}: meta['{k}'] differs from {files[0].name}")
                for k in META_FLOAT:
                    if not math.isclose(meta[k], ref_meta[k], rel_tol=1e-6):
                        raise SystemExit(f"{f.name}: meta['{k}'] differs from {files[0].name}; recollect/re-export with consistent scales")
                if not np.allclose(d["q_default"], q_default, atol=1e-3):
                    print(f"  warning: {f.name} has a different q_default")
            if int(meta["obs_frame_dim"]) != FRAME_DIM:
                raise SystemExit(f"Expected obs_frame_dim={FRAME_DIM}, got {meta['obs_frame_dim']}")
            if int(meta["obs_history"]) * FRAME_DIM != d["obs"].shape[1]:
                raise SystemExit(f"Unexpected observation shape in {f.name}: {d['obs'].shape}")
            t = d["time"]
            keep = d["mpc_ok"].astype(bool) & ((t - t[0]) >= skip_s)
            # Stacked frames are oldest-first; feed only the newest 45-dim frame.
            X.append(d["obs"][keep, -FRAME_DIM:])
            Y.append(d["actions"][keep])
            G.append(fi * 100_000 + d["episode"][keep].astype(np.int64))
            print(f"  {f.name}: {int(keep.sum())}/{len(t)} samples kept")
    X, Y, G = np.concatenate(X).astype(np.float32), np.concatenate(Y).astype(np.float32), np.concatenate(G)
    assert X.shape[1] == FRAME_DIM and Y.shape[1] == ACT_DIM, (X.shape, Y.shape)
    return X, Y, G, ref_meta, q_default


def split(G: np.ndarray, val_frac: float, seed: int) -> np.ndarray:
    keys = np.unique(G)
    if len(keys) < 2:
        mask = np.zeros(len(G), dtype=bool)
        mask[int(len(G) * (1.0 - val_frac)):] = True
        return mask
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n_val = max(1, int(round(len(keys) * val_frac)))
    return np.isin(G, keys[:n_val])


def make_sequences(X, Y, G):
    """Split into episode sequences, preserving sample order."""
    seqs = []
    for key in np.unique(G):
        idx = np.flatnonzero(G == key)
        if len(idx):
            seqs.append((X[idx], Y[idx]))
    return seqs


def make_windows(sequences, length: int):
    """Fixed-length windows; final partial windows are zero-padded and masked."""
    windows = []
    for xs, ys in sequences:
        for start in range(0, len(xs), length):
            xw, yw = xs[start:start + length], ys[start:start + length]
            n = len(xw)
            if n == 0:
                continue
            xp = np.zeros((length, FRAME_DIM), np.float32)
            yp = np.zeros((length, ACT_DIM), np.float32)
            mask = np.zeros(length, np.float32)
            xp[:n], yp[:n], mask[:n] = xw, yw, 1.0
            windows.append((xp, yp, mask))
    return windows


# ----------------------------------------------------------------------------
# Actor and training-only critic
# ----------------------------------------------------------------------------
class Actor(nn.Module):
    def __init__(self, mean, std, clip: float):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.clip = float(clip)
        self.gru = nn.GRU(input_size=FRAME_DIM, hidden_size=HIDDEN, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(HIDDEN, HEAD_HIDDEN)
        self.fc2 = nn.Linear(HEAD_HIDDEN, ACT_DIM)
        self.act = nn.ELU()

    def normalize(self, x):
        return ((x - self.mean) / self.std).clamp(-self.clip, self.clip)

    def forward(self, x, hidden=None):
        # x: [batch, time, 45]; hidden: [1, batch, 128]
        x = self.normalize(x)
        features, hidden = self.gru(x, hidden)
        action = torch.tanh(self.fc2(self.act(self.fc1(features))))
        return action, hidden


class PrivilegedCritic(nn.Module):
    """Feedforward value critic for RL fine-tuning; not trained by BC-only data."""
    def __init__(self, privileged_dim: int = PRIVILEGED_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(privileged_dim, 128), nn.ELU(),
                                 nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 1))

    def forward(self, privileged_state):
        return self.net(privileged_state)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: Actor, windows, device, batch_size=128):
    model.eval()
    se = torch.zeros(ACT_DIM, device=device)
    ae = torch.zeros(ACT_DIM, device=device)
    count = 0.0
    for i in range(0, len(windows), batch_size):
        chunk = windows[i:i + batch_size]
        x = torch.from_numpy(np.stack([w[0] for w in chunk])).to(device)
        y = torch.from_numpy(np.stack([w[1] for w in chunk])).to(device)
        m = torch.from_numpy(np.stack([w[2] for w in chunk])).to(device).unsqueeze(-1)
        pred, _ = model(x)  # reset hidden state at each truncated window
        err = (pred - y) * m
        se += (err ** 2).sum((0, 1))
        ae += err.abs().sum((0, 1))
        count += m.sum().item()
    return (se.sum() / (count * ACT_DIM)).item(), (ae / count).cpu().numpy()


# ----------------------------------------------------------------------------
# C++ export
# ----------------------------------------------------------------------------
def _lit(v: float) -> str:
    s = f"{v:.9g}"
    if not any(c in s for c in ".en"):
        s += ".0"
    return s + "f"


def c_array(name: str, arr: np.ndarray, size_expr: str | None = None, per_line: int = 8) -> str:
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    assert np.all(np.isfinite(flat)), f"{name} has non-finite values"
    lines = [", ".join(_lit(float(v)) for v in flat[i:i + per_line]) for i in range(0, len(flat), per_line)]
    size = size_expr or str(len(flat))
    return f"constexpr float {name}[{size}] = {{\n    " + ",\n    ".join(lines) + "\n};\n"


def export_header(path: Path, model: Actor, meta: dict, q_default: np.ndarray):
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    out = [
        "#ifndef POLICY_WEIGHTS_h", "#define POLICY_WEIGHTS_h", "",
        "// Auto-generated GRU actor weights. Include from ONE translation unit only.",
        '#include "motion_config.h"', "",
        f"constexpr int POLICY_FRAME_DIM = {FRAME_DIM};",
        f"constexpr int POLICY_GRU_HIDDEN = {HIDDEN};",
        f"constexpr int POLICY_HEAD_HIDDEN = {HEAD_HIDDEN};",
        f"constexpr float POLICY_OBS_CLIP = {_lit(model.clip)};",
        f"constexpr float POLICY_ACTION_SCALE_RAD = {_lit(meta['action_scale_rad'])};",
        "// q_target_rad[j] = clamp(q_default[j] + POLICY_ACTION_SCALE_RAD * action[j], SERVO_MIN[j], SERVO_MAX[j]).",
        "// q_default [rad] = [" + ", ".join(f"{v:.6f}f" for v in q_default) + "]",
        "// Input frame layout: gyro(3), gravity(3), command(3), q-q_default(12), qd(12), previous action(12).",
        "// PyTorch GRU gate order: reset (r), update (z), new/candidate (n).",
        "// GRU equations: r=sigmoid(W_ir*x+b_ir+W_hr*h+b_hr); z=sigmoid(W_iz*x+b_iz+W_hz*h+b_hz);",
        "// n=tanh(W_in*x+b_in+r*(W_hn*h+b_hn)); h_new=(1-z)*n+z*h.",
        "// Weights are row-major [out][in]. Head: ELU -> linear -> tanh.", "",
        c_array("POLICY_OBS_MEAN", sd["mean"], "POLICY_FRAME_DIM"),
        c_array("POLICY_OBS_STD", sd["std"], "POLICY_FRAME_DIM"),
        c_array("POLICY_GRU_W_IH", sd["gru.weight_ih_l0"], "3 * POLICY_GRU_HIDDEN * POLICY_FRAME_DIM"),
        c_array("POLICY_GRU_W_HH", sd["gru.weight_hh_l0"], "3 * POLICY_GRU_HIDDEN * POLICY_GRU_HIDDEN"),
        c_array("POLICY_GRU_B_IH", sd["gru.bias_ih_l0"], "3 * POLICY_GRU_HIDDEN"),
        c_array("POLICY_GRU_B_HH", sd["gru.bias_hh_l0"], "3 * POLICY_GRU_HIDDEN"),
        c_array("POLICY_HEAD_W1", sd["fc1.weight"], "POLICY_HEAD_HIDDEN * POLICY_GRU_HIDDEN"),
        c_array("POLICY_HEAD_B1", sd["fc1.bias"], "POLICY_HEAD_HIDDEN"),
        c_array("POLICY_HEAD_W2", sd["fc2.weight"], "POLICY_ACT_DIM * POLICY_HEAD_HIDDEN"),
        c_array("POLICY_HEAD_B2", sd["fc2.bias"], "POLICY_ACT_DIM"),
        f'static_assert(POLICY_ACT_DIM == {ACT_DIM}, "Set POLICY_ACT_DIM to {ACT_DIM} in motion_config.h");',
        "#endif", ""
    ]
    path.write_text("\n".join(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=HERE / "datasets")
    ap.add_argument("--out-dir", type=Path, default=HERE / "runs")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=128, help="number of sequence windows per batch")
    ap.add_argument("--seq-len", type=int, default=32, help="truncated-BPTT window length")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--skip-s", type=float, default=1.5)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--noise", type=float, default=0.0, help="Gaussian noise on normalized actor inputs")
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.seq_len < 1 or args.batch < 1:
        raise SystemExit("--seq-len and --batch must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()
    print(f"Loading {args.data_dir}")
    X, Y, G, meta, q_default = load_datasets(args.data_dir, args.skip_s)
    val_mask = split(G, args.val_frac, args.seed)
    train_sequences = make_sequences(X[~val_mask], Y[~val_mask], G[~val_mask])
    val_sequences = make_sequences(X[val_mask], Y[val_mask], G[val_mask])
    train_windows = make_windows(train_sequences, args.seq_len)
    val_windows = make_windows(val_sequences, args.seq_len)
    if not train_windows or not val_windows:
        raise SystemExit("Need non-empty train and validation sequence windows; collect more episodes or lower --val-frac.")
    print(f"{len(X)} frames ({len(X) / meta['policy_hz'] / 60:.1f} min): train {sum(len(s[0]) for s in train_sequences)}, "
          f"val {sum(len(s[0]) for s in val_sequences)}, windows {len(train_windows)}/{len(val_windows)}, device {device}")

    train_x = np.concatenate([s[0] for s in train_sequences], axis=0)
    mean, std = train_x.mean(0), train_x.std(0)
    std = np.where(std < 1e-3, 1.0, std)
    model = Actor(mean, std, args.clip).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.02)
    base = float((Y[val_mask] ** 2).mean())
    print(f"baseline val MSE (always output zero action): {base:.5f}")

    run_dir = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    hist_f = open(run_dir / "history.csv", "w", newline="")
    hist = csv.writer(hist_f)
    hist.writerow(["epoch", "train_mse", "val_mse", "lr"])
    best, best_state, best_ep = float("inf"), None, 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(train_windows))
        tot, total_valid = 0.0, 0.0
        for start in range(0, len(order), args.batch):
            chunk = [train_windows[j] for j in order[start:start + args.batch]]
            xb = torch.from_numpy(np.stack([w[0] for w in chunk])).to(device)
            yb = torch.from_numpy(np.stack([w[1] for w in chunk])).to(device)
            mb = torch.from_numpy(np.stack([w[2] for w in chunk])).to(device).unsqueeze(-1)
            xn = model.normalize(xb)
            if args.noise > 0:
                xn = (xn + args.noise * torch.randn_like(xn)).clamp(-args.clip, args.clip)
            pred, _ = model.gru(xn)
            pred = torch.tanh(model.fc2(model.act(model.fc1(pred))))
            err2 = (pred - yb).pow(2) * mb
            loss = err2.sum() / (mb.sum() * ACT_DIM).clamp_min(1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            nvalid = mb.sum().item()
            tot += loss.item() * nvalid
            total_valid += nvalid
        sched.step()
        tr = tot / max(total_valid, 1.0)
        va, _ = evaluate(model, val_windows, device)
        hist.writerow([ep, tr, va, sched.get_last_lr()[0]])
        if va < best:
            best, best_ep, best_state = va, ep, copy.deepcopy(model.state_dict())
        if ep == 1 or ep % 10 == 0:
            print(f"epoch {ep:4d} | train {tr:.5f} | val {va:.5f} | best {best:.5f} @ {best_ep} | {time.time()-t0:.0f}s")
        if args.patience and ep - best_ep >= args.patience:
            print(f"early stop at epoch {ep} (best {best:.5f} @ {best_ep})")
            break
    hist_f.close()
    model.load_state_dict(best_state)
    va, mae = evaluate(model, val_windows, device)
    scale = meta["action_scale_rad"]
    print(f"\nbest val MSE {va:.5f} (epoch {best_ep}), {100 * (1 - va / base):.1f}% below baseline")
    print(f"val MAE per joint [rad]: {np.round(mae * scale, 4)} (mean {mae.mean()*scale:.4f} rad = {np.degrees(mae.mean()*scale):.2f} deg)")

    # Critic is deliberately not optimized here: this script performs behavior cloning,
    # and the dataset has no returns/value targets or privileged-state fields.
    critic = PrivilegedCritic()
    torch.save({"actor_state_dict": model.state_dict(), "critic_state_dict": critic.state_dict(),
                "clip": args.clip, "meta": meta, "q_default": q_default,
                "actor_arch": {"frame_dim": FRAME_DIM, "gru_hidden": HIDDEN, "head_hidden": HEAD_HIDDEN},
                "critic_arch": {"privileged_dim": PRIVILEGED_DIM, "hidden": [128, 64]},
                "critic_trained": False}, run_dir / "policy.pt")
    export_header(run_dir / "policy_weights.h", model, meta, q_default)
    print(f"parameters: actor={sum(p.numel() for p in model.parameters()):,}, critic={sum(p.numel() for p in critic.parameters()):,} (critic untrained)")
    print(f"saved -> {run_dir.resolve()}")


if __name__ == "__main__":
    main()
