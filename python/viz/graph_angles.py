# Run with:
# python python/viz/graph_angles.py

import sys
from pathlib import Path

# Add the python/ directory to the module search path.
PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from leg.leg_config import Movement

CSV_PATH = Path("python/logs/leg0_trajectory_angles.csv")

LEG_COLORS = {
    "coxa": "",
    "femur": "",
    "tibia": ""
}

JOINT_ANGLES = ["coxa", "femur", "tibia"]
# JOINT_ANGLES = ["femur"]
REF = ["target", "real"]

MAX_SAMPLES = 2000

df = pd.read_csv(CSV_PATH, dtype=float)
df = df.iloc[:MAX_SAMPLES]  # limit number of samples to plot

time = df.iloc[:, 0] # time
df = df.iloc[:, 1:] # angles

for joint in JOINT_ANGLES:
    fig, ax = plt.subplots()

    for ref in REF:
        col = f"{ref}_{joint}"
        ax.plot(time, df[col], linewidth=2.0, label=ref.capitalize())

    ax.fill_between(time, df[f"{REF[0]}_{joint}"], 0, where=(df[df.columns[-1]] != 0), color="red", alpha=0.1, label="swinging")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint Angle (deg)")
    fig.suptitle(f"{joint.capitalize()} Joint Angles")
    ax.set_title(f"y-offset={Movement.SWING_Y_OFFSET_MM}mm, height={Movement.STEP_HEIGHT_MM}mm, length={Movement.STEP_LENGTH_MM}mm")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()