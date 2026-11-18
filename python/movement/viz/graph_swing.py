
# Run with:
#   python python/movement/viz/graph_swing.py
#   python python/movement/viz/graph_swing.py --start 2 --end 6
#   python python/movement/viz/graph_swing.py --start 2
#   python python/movement/viz/graph_swing.py --end 6

import sys
import argparse
from pathlib import Path

# Add the python/ directory to the module search path.
PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from leg.leg_config import Movement

LOG_DIR = Path("python/movement/logs")

LEG_NAMES = {
    0: "Leg 0 (FL)",
    1: "Leg 1 (FR)",
    2: "Leg 2 (BR)",
    3: "Leg 3 (BL)",
}

T = Movement.SWING_DURATION_S
G = Movement.GAP_DURATION_S
DUTY = 1.0 - T / (3.0 * (T + G))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Graph leg swing timing and identify swing overlaps."
    )
    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help="Start time (default: 1 second before first swing).",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="End time (default: 1 second after final swing).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load all four CSVs before determining the plot time range.
    raw_data = {}

    for leg_id, name in LEG_NAMES.items():
        path = LOG_DIR / f"leg{leg_id}_foot_pos.csv"

        if not path.exists():
            raise FileNotFoundError(f"Missing CSV file: {path}")

        df = pd.read_csv(path)

        required = {"time_s", "swinging"}
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{path} is missing columns: {sorted(missing)}"
            )

        df = (
            df.sort_values("time_s")
            .drop_duplicates("time_s")
            .reset_index(drop=True)
        )

        if df.empty:
            raise ValueError(f"No samples found in {path}")

        raw_data[leg_id] = df

    # Find the first and final swing times across all four legs.
    first_swing_time = None
    final_swing_time = None

    for leg_id, df in raw_data.items():
        swing_times = df.loc[df["swinging"] != 0, "time_s"]

        if swing_times.empty:
            continue

        leg_first = float(swing_times.min())
        leg_final = float(swing_times.max())

        if first_swing_time is None or leg_first < first_swing_time:
            first_swing_time = leg_first

        if final_swing_time is None or leg_final > final_swing_time:
            final_swing_time = leg_final

    if first_swing_time is None:
        raise ValueError("No swing detected in any leg CSV.")

    # Automatically pad the first and final swings by one second.
    if args.start is None:
        args.start = first_swing_time - 1.0

    if args.end is None:
        args.end = final_swing_time + 1.0

    if args.end <= args.start:
        raise ValueError("--end must be greater than --start")

    print(f"First swing: {first_swing_time:.3f} s")
    print(f"Final swing: {final_swing_time:.3f} s")
    print(f"Plot range:  {args.start:.3f} - {args.end:.3f} s")

    # Filter the data to the selected time range.
    leg_data = {}

    for leg_id, df in raw_data.items():
        df = df[
            (df["time_s"] >= args.start)
            & (df["time_s"] <= args.end)
        ].copy()

        if df.empty:
            raise ValueError(
                f"No samples in selected time range for leg {leg_id}."
            )

        leg_data[leg_id] = df.set_index("time_s")["swinging"]

    # Align the four logs on a common time axis.
    all_times = np.unique(
        np.concatenate([
            series.index.to_numpy() for series in leg_data.values()
        ])
    )
    all_times.sort()

    swing = np.zeros((len(all_times), 4), dtype=bool)

    for leg_id, series in leg_data.items():
        # Forward-fill each leg's state onto the common time axis.
        aligned = series.reindex(all_times).ffill().fillna(0)
        swing[:, leg_id] = aligned.to_numpy() != 0

    overlap_count = swing.sum(axis=1)
    overlap = overlap_count >= 2

    # Print contiguous overlap intervals.
    print("\nSwing overlap intervals")
    print("=" * 60)

    overlap_indices = np.flatnonzero(overlap)

    if len(overlap_indices) == 0:
        print("No swing overlaps detected in the selected time range.")
    else:
        start_idx = overlap_indices[0]
        previous_idx = start_idx
        previous_legs = tuple(np.flatnonzero(swing[start_idx]))

        intervals = []

        for idx in overlap_indices[1:]:
            current_legs = tuple(np.flatnonzero(swing[idx]))

            if (
                idx != previous_idx + 1
                or current_legs != previous_legs
            ):
                intervals.append(
                    (start_idx, previous_idx, previous_legs)
                )
                start_idx = idx
                previous_legs = current_legs

            previous_idx = idx

        intervals.append(
            (start_idx, previous_idx, previous_legs)
        )

        for first, last, legs in intervals:
            names = ", ".join(LEG_NAMES[i] for i in legs)
            print(
                f"{all_times[first]:.4f} - "
                f"{all_times[last]:.4f} s: {names} swinging"
            )

    # Plot four swing timelines and an overlap-count timeline.
    fig, (ax, overlap_ax) = plt.subplots(
        2,
        1,
        figsize=(13, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )

    for leg_id, name in LEG_NAMES.items():
        ax.fill_between(
            all_times,
            leg_id - 0.35,
            leg_id + 0.35,
            where=swing[:, leg_id],
            step="post",
            alpha=0.8,
            label=name,
        )

    ax.set_yticks(range(4))
    ax.set_yticklabels([LEG_NAMES[i] for i in range(4)])
    ax.set_ylim(-0.5, 3.5)
    ax.set_ylabel("Leg")
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    # Show how many legs are swinging at each sample.
    overlap_ax.step(
        all_times,
        overlap_count,
        where="post",
        linewidth=1.8,
    )

    overlap_ax.axhline(
        1,
        linestyle="--",
        linewidth=1.0,
        label="Maximum for no overlap",
    )

    overlap_ax.fill_between(
        all_times,
        1,
        overlap_count,
        where=overlap,
        step="post",
        alpha=0.35,
        label="Overlap (2+ legs)",
    )

    overlap_ax.set_yticks([0, 1, 2, 3, 4])
    overlap_ax.set_ylim(-0.1, 4.2)
    overlap_ax.set_ylabel("Swinging\nlegs")
    overlap_ax.set_xlabel("Time (s)")
    overlap_ax.grid(True, axis="x", alpha=0.3)
    overlap_ax.legend(loc="upper right")
    overlap_ax.set_xlim(args.start, args.end)

    fig.suptitle(
        "Swing Timing and Overlap Detection",
        fontsize=16,
        y=0.98,
    )

    fig.text(
        0.5,
        0.90,
        f"Duty Factor = {DUTY:.3f}",
        ha="center",
        fontsize=12,
        color="gray",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    plt.show()


if __name__ == "__main__":
    main()
