#!/usr/bin/env python3
"""Behavior-clone the MPC teacher into the MLP policy (135 -> 256 -> 256 -> 12, tanh output).

Put this file in python/movement/rl/ (next to datasets/) and run:

    python python/movement/rl/train.py
    python python/movement/rl/train.py --epochs 400 --act elu --noise 0.02

Inputs : every *.npz in <this dir>/datasets, written by collect_data.py
Outputs: <this dir>/runs/<timestamp>/
            policy.pt           state_dict + normalization + meta (reload with Policy(...))
            policy_weights.h    drop-in replacement for the C++ header (POLICY_* names, motion_config.h)
            history.csv         per-epoch train/val loss

Pipeline (matches the C++ policy):  normalize -> clip -> fc1 -> act -> fc2 -> act -> fc3 -> tanh
Weights are exported row-major [out][in], i.e. PyTorch nn.Linear.weight flattened in C order.
Targets are the normalized teacher actions in [-1, 1]:  q_target = q_default + ACTION_SCALE * action.
"""

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

OBS_DIM, H1, H2, ACT_DIM = 135, 256, 256, 12

# Metadata that must be identical across all dataset files (it defines the obs/action encoding).
META_EXACT = ["obs_history", "obs_frame_dim", "act_dim", "joint_names"]
META_FLOAT = ["gyro_scale", "cmd_scale_lin", "cmd_scale_ang", "qd_scale", "action_scale_rad", "policy_hz"]

ACTIVATIONS = {"elu": nn.ELU, "tanh": nn.Tanh, "relu": nn.ReLU}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_datasets(data_dir: Path, skip_s: float):
    files = sorted(data_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"No .npz files found in {data_dir}")

    ref_meta, q_default = None, None
    X, Y, G, n_per_file = [], [], [], []
    for fi, f in enumerate(files):
        d = np.load(f)
        meta = json.loads(str(d["meta"]))
        if ref_meta is None:
            ref_meta, q_default = meta, d["q_default"].astype(np.float32)
        else:
            for k in META_EXACT:
                if meta[k] != ref_meta[k]:
                    raise SystemExit(f"{f.name}: meta['{k}'] differs from {files[0].name}")
            for k in META_FLOAT:
                if not math.isclose(meta[k], ref_meta[k], rel_tol=1e-6):
                    raise SystemExit(
                        f"{f.name}: meta['{k}']={meta[k]} differs from {files[0].name} ({ref_meta[k]}). "
                        "Obs/actions were encoded with different scales; re-export q_target with one scale "
                        "(the previous-action part of obs depends on it) or recollect.")
            if not np.allclose(d["q_default"], q_default, atol=1e-3):
                print(f"  warning: {f.name} has a different q_default")

        t = d["time"]
        keep = d["mpc_ok"].astype(bool) & ((t - t[0]) >= skip_s)
        X.append(d["obs"][keep])
        Y.append(d["actions"][keep])
        G.append(fi * 100_000 + d["episode"][keep].astype(np.int64))
        n_per_file.append(int(keep.sum()))
        print(f"  {f.name}: {int(keep.sum())}/{len(t)} samples kept")

    X, Y, G = np.concatenate(X).astype(np.float32), np.concatenate(Y).astype(np.float32), np.concatenate(G)
    assert X.shape[1] == OBS_DIM and Y.shape[1] == ACT_DIM, (X.shape, Y.shape)
    return X, Y, G, ref_meta, q_default


def split(G: np.ndarray, val_frac: float, seed: int) -> np.ndarray:
    """Boolean validation mask. Whole episodes are held out so history windows don't leak."""
    keys = np.unique(G)
    if len(keys) < 2:                                    # one chunk only: hold out the tail
        mask = np.zeros(len(G), dtype=bool)
        mask[int(len(G) * (1.0 - val_frac)):] = True
        return mask
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n_val = max(1, int(round(len(keys) * val_frac)))
    return np.isin(G, keys[:n_val])


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class Policy(nn.Module):
    def __init__(self, mean, std, clip: float, act: str):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.clip = float(clip)
        self.fc1, self.fc2, self.fc3 = nn.Linear(OBS_DIM, H1), nn.Linear(H1, H2), nn.Linear(H2, ACT_DIM)
        self.act = ACTIVATIONS[act]()

    def normalize(self, x):
        return ((x - self.mean) / self.std).clamp(-self.clip, self.clip)

    def head(self, xn):
        return torch.tanh(self.fc3(self.act(self.fc2(self.act(self.fc1(xn))))))

    def forward(self, x):
        return self.head(self.normalize(x))


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: Policy, X: torch.Tensor, Y: torch.Tensor, batch: int = 8192):
    model.eval()
    se, ae = torch.zeros(ACT_DIM, device=X.device), torch.zeros(ACT_DIM, device=X.device)
    for i in range(0, len(X), batch):
        err = model(X[i:i + batch]) - Y[i:i + batch]
        se += (err ** 2).sum(0)
        ae += err.abs().sum(0)
    n = len(X)
    return (se.sum() / (n * ACT_DIM)).item(), (ae / n).cpu().numpy()


# ----------------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------------
def _lit(v: float) -> str:
    s = f"{v:.9g}"
    if not any(c in s for c in ".en"):
        s += ".0"
    return s + "f"


def c_array(name: str, arr: np.ndarray, size_expr: str | None = None, per_line: int = 8) -> str:
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)         # C order == row-major [out][in]
    assert np.all(np.isfinite(flat)), f"{name} has non-finite values"
    lines = [", ".join(_lit(float(v)) for v in flat[i:i + per_line]) for i in range(0, len(flat), per_line)]
    size = size_expr if size_expr else str(len(flat))
    return f"constexpr float {name}[{size}] = {{\n    " + ",\n    ".join(lines) + "\n};\n"


def numpy_forward(x, mean, std, clip, W1, b1, W2, b2, W3, b3, act: str):
    """Reference forward from the exported flat arrays (what the C++ does)."""
    f = {"elu": lambda z: np.where(z > 0, z, np.expm1(np.minimum(z, 0))),
         "tanh": np.tanh, "relu": lambda z: np.maximum(z, 0)}[act]
    h = np.clip((x - mean) / std, -clip, clip)
    h = f(W1.reshape(H1, OBS_DIM) @ h + b1)
    h = f(W2.reshape(H2, H1) @ h + b2)
    return np.tanh(W3.reshape(ACT_DIM, H2) @ h + b3)


ACT_CODE = {"relu": 0, "elu": 1, "tanh": 2}      # POLICY_ACTIVATION in policy_weights.h


def export_header(path: Path, model: Policy, meta: dict, q_default: np.ndarray, act: str) -> None:
    """Drop-in policy_weights.h (same names/format as the dummy header)."""
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    out = [
        "#ifndef POLICY_WEIGHTS_h",
        "#define POLICY_WEIGHTS_h",
        "",
        "// Auto-generated by train.py: behavior-cloned MPC teacher. Include from ONE translation unit only.",
        "//",
        "// Must agree with motion_config.h (NOT defined here):",
        f"//   OBS_HISTORY = {meta['obs_history']}, OBS_FRAME_DIM = {meta['obs_frame_dim']}, "
        f"POLICY_OBS_DIM = {OBS_DIM}, POLICY_ACT_DIM = {ACT_DIM}, policy rate = {meta['policy_hz']:g} Hz",
        f"//   GYRO_SCALE = {meta['gyro_scale']:g}, CMD_SCALE_LIN = {meta['cmd_scale_lin']:g}, "
        f"CMD_SCALE_ANG = {meta['cmd_scale_ang']:g}, QD_SCALE = {meta['qd_scale']:g}",
        f"//   q_target [rad] = q_default + {meta['action_scale_rad']:g} * action",
        "//   q_default [rad] = [" + ", ".join(f"{v:.4f}" for v in q_default) + "]",
        "//   joint order: " + ", ".join(meta["joint_names"]),
        "//",
        "// Pipeline: obs (already multiplied by the fixed scales above) -> (obs - MEAN) / STD",
        "//           -> clip to +-POLICY_OBS_CLIP -> fc1 -> act -> fc2 -> act -> fc3 -> tanh",
        "// Weights are row-major [out][in], matching PyTorch nn.Linear.weight layout.",
        "",
        '#include "motion_config.h"',
        "",
        "// ---- architecture: 135 -> 256 -> 256 -> 12 ----",
        f"constexpr int POLICY_H1 = {H1};",
        f"constexpr int POLICY_H2 = {H2};",
        "",
        f"constexpr int POLICY_ACTIVATION = {ACT_CODE[act]};     // Hidden layers: 0 = ReLU, 1 = ELU, 2 = tanh ({act} here)",
        f"constexpr float POLICY_OBS_CLIP = {_lit(model.clip)};  // <= 0 disables clipping",
        "",
        f'static_assert(POLICY_OBS_DIM == {OBS_DIM}, "Set POLICY_OBS_DIM to {OBS_DIM} in motion_config.h");',
        f'static_assert(POLICY_ACT_DIM == {ACT_DIM}, "Set POLICY_ACT_DIM to {ACT_DIM} in motion_config.h");',
        "",
        "// ---- observation normalization (per input dimension, from the training split) ----",
        c_array("POLICY_OBS_MEAN", sd["mean"], "POLICY_OBS_DIM"),
        c_array("POLICY_OBS_STD", sd["std"], "POLICY_OBS_DIM"),
        "// ---- trained weights and biases ----",
        c_array("POLICY_W1", sd["fc1.weight"], "POLICY_H1 * POLICY_OBS_DIM"),
        c_array("POLICY_B1", sd["fc1.bias"], "POLICY_H1"),
        c_array("POLICY_W2", sd["fc2.weight"], "POLICY_H2 * POLICY_H1"),
        c_array("POLICY_B2", sd["fc2.bias"], "POLICY_H2"),
        c_array("POLICY_W3", sd["fc3.weight"], "POLICY_ACT_DIM * POLICY_H2"),
        c_array("POLICY_B3", sd["fc3.bias"], "POLICY_ACT_DIM"),
        "#endif",
        "",
    ]
    path.write_text("\n".join(out))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=HERE / "datasets")
    ap.add_argument("--out-dir", type=Path, default=HERE / "runs")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4, help="AdamW weight decay")
    ap.add_argument("--val-frac", type=float, default=0.1, help="fraction of episodes held out")
    ap.add_argument("--skip-s", type=float, default=1.5, help="drop this many seconds at the start of each file")
    ap.add_argument("--clip", type=float, default=5.0, help="clip normalized obs to +-clip")
    ap.add_argument("--noise", type=float, default=0.0, help="gaussian noise std added to normalized obs (train only)")
    ap.add_argument("--act", choices=list(ACTIVATIONS), default="elu",
                    help="hidden activation -- MUST match the C++ forward pass")
    ap.add_argument("--patience", type=int, default=40, help="early-stop after this many epochs without val improvement (0 = off)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()

    print(f"Loading {args.data_dir}")
    X, Y, G, meta, q_default = load_datasets(args.data_dir, args.skip_s)
    val_mask = split(G, args.val_frac, args.seed)
    Xtr, Ytr, Xva, Yva = X[~val_mask], Y[~val_mask], X[val_mask], Y[val_mask]
    print(f"{len(X)} samples ({len(X) / meta['policy_hz'] / 60:.1f} min): "
          f"train {len(Xtr)}, val {len(Xva)}, device {device}")
    if len(Xva) == 0:
        raise SystemExit("Empty validation set; collect more data or lower --val-frac.")

    # normalization from the training split; near-constant features (e.g. vy = 0) are left unscaled
    mean = Xtr.mean(0)
    std = Xtr.std(0)
    std = np.where(std < 1e-3, 1.0, std)

    model = Policy(mean, std, args.clip, args.act).to(device)
    Xtr_t, Ytr_t = torch.from_numpy(Xtr).to(device), torch.from_numpy(Ytr).to(device)
    Xva_t, Yva_t = torch.from_numpy(Xva).to(device), torch.from_numpy(Yva).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.02)
    lossf = nn.MSELoss()

    base = float((Yva ** 2).mean())
    print(f"baseline val MSE (always output the default pose): {base:.5f}")

    run_dir = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    hist_f = open(run_dir / "history.csv", "w", newline="")
    hist = csv.writer(hist_f)
    hist.writerow(["epoch", "train_mse", "val_mse", "lr"])

    best, best_state, best_ep = float("inf"), None, 0
    n = len(Xtr_t)
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            xn = model.normalize(Xtr_t[idx])
            if args.noise > 0:
                xn = xn + args.noise * torch.randn_like(xn)
            loss = lossf(model.head(xn), Ytr_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()

        tr = tot / n
        va, _ = evaluate(model, Xva_t, Yva_t)
        hist.writerow([ep, tr, va, sched.get_last_lr()[0]])
        if va < best:
            best, best_ep, best_state = va, ep, copy.deepcopy(model.state_dict())
        if ep == 1 or ep % 10 == 0:
            print(f"epoch {ep:4d} | train {tr:.5f} | val {va:.5f} | best {best:.5f} @ {best_ep} | {time.time() - t0:.0f}s")
        if args.patience and ep - best_ep >= args.patience:
            print(f"early stop at epoch {ep} (best {best:.5f} @ {best_ep})")
            break
    hist_f.close()

    model.load_state_dict(best_state)
    va, mae = evaluate(model, Xva_t, Yva_t)
    scale = meta["action_scale_rad"]
    print(f"\nbest val MSE {va:.5f} (epoch {best_ep}), {100 * (1 - va / base):.1f}% below the baseline")
    print(f"val MAE per joint [rad]: {np.round(mae * scale, 4)}   (mean {mae.mean() * scale:.4f} rad "
          f"= {np.degrees(mae.mean() * scale):.2f} deg)")

    # ---- save ----
    torch.save({"state_dict": model.state_dict(), "clip": args.clip, "act": args.act,
                "meta": meta, "q_default": q_default}, run_dir / "policy.pt")
    export_header(run_dir / "policy_weights.h", model, meta, q_default, args.act)

    # exported-array sanity check: numpy forward from the flat row-major arrays vs PyTorch
    model.eval().cpu()
    sd = {k: v.numpy().astype(np.float32) for k, v in model.state_dict().items()}
    xs = Xva[:256]
    ref = model(torch.from_numpy(xs)).detach().numpy()
    got = np.stack([numpy_forward(x, sd["mean"], sd["std"], model.clip,
                                  sd["fc1.weight"].reshape(-1), sd["fc1.bias"],
                                  sd["fc2.weight"].reshape(-1), sd["fc2.bias"],
                                  sd["fc3.weight"].reshape(-1), sd["fc3.bias"], args.act) for x in xs])
    print(f"export check: max |torch - flat-array forward| = {np.abs(ref - got).max():.2e}")
    print(f"saved -> {run_dir.resolve()}")


if __name__ == "__main__":
    main()