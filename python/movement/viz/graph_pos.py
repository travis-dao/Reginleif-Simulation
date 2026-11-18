
# Run with:
#   python python/movement/viz/graph_pos.py
#   python python/movement/viz/graph_pos.py --leg 0
#   python python/movement/viz/graph_pos.py --leg 2
#   python python/movement/viz/graph_pos.py --leg all

import argparse
import sys
from pathlib import Path

# Add the python/ directory to the module search path.
PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

import pandas as pd
import matplotlib.pyplot as plt

LOG_DIR = Path("python/movement/logs")

AXIS = ["x", "y", "z"]
PREFIX = ["target", "final"]

MAX_SAMPLES = 5000
START_TIME = 2
END_TIME = 6


def parse_args():
    parser = argparse.ArgumentParser(
        description="Graph foot position trajectories for one or all legs."
    )
    parser.add_argument(
        "--leg",
        choices=["0", "1", "2", "3", "all"],
        default="0",
        help="Leg to graph: 0, 1, 2, 3, or all (default: 0).",
    )
    return parser.parse_args()


def load_leg_data(leg_id):
    csv_path = LOG_DIR / f"leg{leg_id}_foot_pos.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV file: {csv_path}")

    df = pd.read_csv(csv_path, dtype=float)
    df = df[df["time_s"] >= START_TIME]
    df = df[df["time_s"] <= END_TIME]
    df = df.iloc[:MAX_SAMPLES].copy()

    return df


def main():
    args = parse_args()

    leg_ids = range(4) if args.leg == "all" else [int(args.leg)]
    leg_data = {leg_id: load_leg_data(leg_id) for leg_id in leg_ids}

    for axis in AXIS:
        fig, ax = plt.subplots()

        for leg_id, df in leg_data.items():
            time = df["time_s"]
            swinging = df["swinging"] != 0

            for prefix in PREFIX:
                col = f"{prefix}_{axis}"
                ax.plot(
                    time,
                    df[col],
                    linewidth=2.0,
                    label=f"Leg {leg_id} {prefix.capitalize()}",
                )

            # Mark swing periods for each leg.
            ax.fill_between(
                time,
                0,
                1,
                where=swinging,
                transform=ax.get_xaxis_transform(),
                alpha=0.08,
                label=f"Leg {leg_id} Swinging",
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"{axis.upper()} (mm, leg frame)")
        ax.set_title(f"{axis.upper()} Component of Foot Position")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
