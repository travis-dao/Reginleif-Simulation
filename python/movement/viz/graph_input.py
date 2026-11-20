
# Run with:
# python python/movement/viz/graph_input.py

import sys
from pathlib import Path

# Add the python/ directory to the module search path.
PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = Path("python/sim/log/input_command_mapping.csv")

df = pd.read_csv(CSV_PATH)

df = df[np.hypot(df["input_x"], df["input_y"]) >= 0.99]

# Joystick inputs define the X and Y axes.
x = df["input_x"]
y = df["input_y"]

# Each output variable is plotted on the Z axis.
outputs = [
    ("V", "Joystick Inputs -> V"),
    ("R", "Joystick Inputs -> R"),
    ("v_cmd", "Joystick Inputs -> v_cmd"),
    ("yaw_rate_cmd", "Joystick Inputs -> yaw_rate_cmd"),
]

fig = plt.figure(figsize=(14, 10))

for i, (z_col, title) in enumerate(outputs, start=1):
    ax = fig.add_subplot(2, 2, i, projection="3d")

    scatter = ax.scatter(
        x,
        y,
        df[z_col],
        c=df[z_col],
        cmap="viridis",
        s=20,
        alpha=0.8,
    )

    ax.set_xlabel("Joystick Input X")
    ax.set_ylabel("Joystick Input Y")
    ax.set_zlabel(z_col)
    ax.set_title(title)

    fig.colorbar(scatter, ax=ax, shrink=0.65, pad=0.12)

fig.suptitle("Joystick Input Mapping (3D)", fontsize=16)
fig.tight_layout()

plt.show()
