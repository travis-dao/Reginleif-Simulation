# python python/sim/util/log_plot.py --log python/sim/log/main.npz

import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot(args):
    d = np.load(args.log)
    t, F, S = d["t"], d["forces"], d["swing"]

    mask = (t >= 2.0) & (t <= 10.0)
    t, F, S = t[mask], F[mask], S[mask]

    leg_names = ("FL", "FR", "BR", "BL")
    force_names = ("Fx", "Fy", "Fz")

    # --- foot forces ---
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 7))

    for j, (ax, force_name) in enumerate(zip(axes, force_names)):
        for i, leg_name in enumerate(leg_names):
            ax.plot(t, F[:, i, j], label=leg_name)

        ax.set_ylabel(f"{force_name} (N)")
        ax.set_title(f"{force_name} — All Legs")
        ax.legend(ncol=4)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    # --- pitch and roll (separate figure) ---
    if "pitch" in d.files and "roll" in d.files:
        pitch = np.degrees(d["pitch"][mask])
        roll = np.degrees(d["roll"][mask])
        n_swing = S.sum(axis=1)                      # legs in swing at each sample

        fig2, (ax_p, ax_r, ax_n) = plt.subplots(
            3, 1, sharex=True, figsize=(9, 7),
            gridspec_kw={"height_ratios": [3, 3, 1]})

        ax_p.plot(t, pitch)
        ax_p.set_ylabel("Pitch (deg)")
        ax_p.set_title("Body pitch")

        ax_r.plot(t, roll, color="tab:orange")
        ax_r.set_ylabel("Roll (deg)")
        ax_r.set_title("Body roll")

        ax_n.step(t, 4 - n_swing, where="post", color="k")
        ax_n.set_ylabel("Stance legs")
        ax_n.set_yticks(range(0, 5))
        ax_n.set_xlabel("Time (s)")

        for ax in (ax_p, ax_r, ax_n):
            ax.grid(True, alpha=0.3)

    # --- foot offsets (separate figure) ---
    if "offsets" in d.files:
        O = d["offsets"][mask]
        offset_names = ("X", "Y", "Z")

        fig3, axes3 = plt.subplots(
            3, 1, sharex=True, figsize=(9, 7)
        )

        for j, (ax, offset_name) in enumerate(
            zip(axes3, offset_names)
        ):
            for i, leg_name in enumerate(leg_names):
                ax.plot(t, O[:, i, j], label=leg_name)

            ax.set_ylabel(f"{offset_name} (mm)")
            ax.set_title(f"{offset_name} Offset — All Legs")
            ax.legend(ncol=4)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
    else:
        print("log has no pitch/roll: add roll to Logger and rerun the sim")

    plt.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--log", type=str, required=True, help="path to log .npz file")

    args = ap.parse_args()
    plot(args)