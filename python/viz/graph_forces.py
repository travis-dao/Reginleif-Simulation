# Run with:
# python python/viz/graph_forces.py

import sys
from pathlib import Path
import argparse

# Add the python/ directory to the module search path.
PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = Path("python/logs/mpc_ground_forces.csv")

LEGS = ["FL", "FR", "BR", "BL"]

COLORS = {
    "FL": "tab:blue",
    "FR": "tab:orange",
    "BR": "tab:green",
    "BL": "tab:red",
}


def plot_leg_z_forces(csv_path: str, selected_legs: list[str]) -> None:
    df = pd.read_csv(csv_path)

    time = df["time_s"]
    swing_leg = df["swing_leg"]
    t = time.to_numpy()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Determine plot boundaries.
    if len(t) > 1:
        left_edge = t[0] - (t[1] - t[0]) / 2
        right_edge = t[-1] + (t[-1] - t[-2]) / 2
    else:
        left_edge = t[0] - 0.01
        right_edge = t[0] + 0.01

    # Shade each interval according to the swinging leg.
    start_idx = 0

    for i in range(1, len(df) + 1):
        changed = (
            i == len(df)
            or swing_leg.iloc[i] != swing_leg.iloc[start_idx]
        )

        if not changed:
            continue

        start_time = (
            left_edge
            if start_idx == 0
            else (t[start_idx - 1] + t[start_idx]) / 2
        )

        end_time = (
            right_edge
            if i == len(df)
            else (t[i - 1] + t[i]) / 2
        )

        leg = swing_leg.iloc[start_idx]

        ax.axvspan(
            start_time,
            end_time,
            color=COLORS.get(leg, "gray"),
            alpha=0.12,
            label=f"{leg} swinging",
        )

        start_idx = i

    # Plot only the requested legs.
    for leg in selected_legs:
        force_col = f"{leg}_Fz_N"

        if force_col not in df.columns:
            raise ValueError(f"Missing CSV column: {force_col}")

        ax.plot(
            time,
            df[force_col],
            color=COLORS[leg],
            linewidth=1.8,
            label=f"{leg} $F_z$",
        )

    ax.set_title("Vertical Force of Each Leg Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Vertical Force $F_z$ (N)")
    ax.grid(True, alpha=0.3)

    # Remove duplicate legend labels.
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), ncol=3)

    fig.tight_layout()
    plt.show()


def plot_total_z_force(csv_path):
    """Plot the sum of all four legs' vertical forces over time."""
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)

    # Ensure the force columns are numeric.
    legs = ["FL", "FR", "BR", "BL"]
    force_cols = [f"{leg}_Fz_N" for leg in legs]

    for col in force_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    time = pd.to_numeric(df["time_s"], errors="coerce")
    total_fz = df[force_cols].sum(axis=1, min_count=4)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(time, total_fz, color="black", linewidth=1.8,
            label="Total Fz")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Sum of vertical forces (N)")
    ax.set_title("Total Vertical Force of All Legs vs. Time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot vertical forces for selected robot legs."
    )

    parser.add_argument(
        "csv_path",
        nargs="?",
        default="python/logs/mpc_ground_forces.csv",
        help="Path to the robot CSV log (default: robot_log.csv)",
    )

    parser.add_argument(
        "--legs",
        nargs="+",
        choices=LEGS,
        default=LEGS,
        help="Legs to plot: FL FR BR BL (default: all legs)",
    )

    args = parser.parse_args()
    plot_leg_z_forces(args.csv_path, args.legs)
    plot_total_z_force(args.csv_path)