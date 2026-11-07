# Run with:
# python python/viz/graph_angles.py

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = Path("python/logs/leg0_trajectory_angles.csv")

LEG_COLORS = {
    "coxa": "",
    "femur": "",
    "tibia": ""
}

JOINT_ANGLES = ["coxa", "femur", "tibia"]
REF = ["target", "real"]

df = pd.read_csv(CSV_PATH, dtype=float)

time = df.iloc[:, 0] # time
df = df.iloc[:, 1:] # angles

for joint in JOINT_ANGLES:
    fig, ax = plt.subplots()

    for ref in REF:
        col = f"{ref}_{joint}"
        ax.plot(time, df[col], linewidth=2.0, label=ref.capitalize())

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint Angle (deg)")
    ax.set_title(f"{joint.capitalize()} Joint Angles")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()