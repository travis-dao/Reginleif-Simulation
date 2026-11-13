# Run with:
# python python/viz/graph_pos.py

import sys
from pathlib import Path

# Add the python/ directory to the module search path.
PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from leg.leg_config import Movement

CSV_PATH = Path("python/logs/leg0_foot_pos.csv")

AXIS = ["x", "y", "z"]
PREFIX = ["target", "final"]

MAX_SAMPLES = 5000

df = pd.read_csv(CSV_PATH, dtype=float)
df = df[df["time_s"] >= 1.5]
df = df.iloc[:MAX_SAMPLES]  # limit number of samples to plot

time = df.iloc[:, 0] # time
df = df.iloc[:, 1:] # angles

for axis in AXIS:
    fig, ax = plt.subplots()

    for pre in PREFIX:
        col = f"{pre}_{axis}"
        ax.plot(time, df[col], linewidth=2.0, label=pre.capitalize())
    
    swinging = df[df.columns[-1]] != 0

    ax.fill_between(
        time,
        0,
        1,
        where=swinging,
        transform=ax.get_xaxis_transform(),
        color="red",
        alpha=0.1,
        label="Swinging",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(" mm (leg frame)")
    fig.suptitle(f"{axis.capitalize()} Component of Foot Pos")
    # ax.set_title(f"")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()