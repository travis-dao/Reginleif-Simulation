# Run with:
# python python/viz/graph_orientation.py

import sys
from pathlib import Path

# Add the python/ directory to the module search path.
PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = Path("python/logs/mpc.csv")

COLS = ["roll", "pitch", "yaw"]
MAX_T_S = 10

# Leg colors: FL, FR, BR, BL
COLORS = {
    "FL": "tab:blue",
    "FR": "tab:orange",
    "BR": "tab:green",
    "BL": "tab:red",
}

df = pd.read_csv(CSV_PATH)
df = df[df["time_s"] < MAX_T_S].reset_index(drop=True)

for col in ["time_s", "swing_leg", *COLS]:
    if col not in df.columns:
        raise ValueError(f"Missing CSV column: {col}")

time = df["time_s"]
swing_leg = df["swing_leg"]
t = time.to_numpy()

fig, ax = plt.subplots(figsize=(12, 6))

if len(df) == 0:
    raise ValueError(f"No data found before {MAX_T_S} seconds.")

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

# Plot roll, pitch, and yaw.
for col in COLS:
    ax.plot(
        time,
        df[col] * 180.0 / np.pi,
        linewidth=1.8,
        label=col.capitalize(),
    )

ax.set_title("Body Orientation Over Time")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Orientation (deg)")
ax.grid(True, alpha=0.3)

# Remove duplicate legend labels.
handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax.legend(unique.values(), unique.keys(), ncol=4)

fig.tight_layout()
plt.show()