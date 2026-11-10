# Run with:
# python python/viz/animate_forces.py

import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

CSV_PATH = Path("python/logs/mpc_ground_forces.csv")

LEGS = ["FL", "FR", "BR", "BL"]

LEG_COLORS = {
    "FL": "tab:blue",
    "FR": "tab:orange",
    "BR": "tab:green",
    "BL": "tab:red",
}

# Force-arrow length in metres per newton.
FORCE_SCALE = 0.005

INTERVAL_MS = 30
PLAYBACK_SPEED = 0.25  # 0.25 = quarter speed, 0.5 = half speed, 1.0 = normal

# Length of the body's forward-direction arrow, in metres.
HEADING_LENGTH_M = 0.15


def clean(value):
    value = html.unescape(str(value)).strip()
    return re.sub(r"^\*+", "", value).strip()


# ------------------------------------------------------------
# Read and clean CSV
# ------------------------------------------------------------

df = pd.read_csv(CSV_PATH, dtype=str)
df.columns = [clean(c) for c in df.columns]

for col in df.columns:
    df[col] = df[col].map(clean)

# Remove repeated header rows, if present.
df = df[df["time_s"] != "time_s"].copy()

for col in df.columns:
    if col != "swing_leg":
        df[col] = pd.to_numeric(df[col], errors="coerce")

required = [
    "time_s",
    "com_x_m",
    "com_y_m",
    "com_z_m",
    "swing_leg",
]

for leg in LEGS:
    required += [
        f"{leg}_x_m",
        f"{leg}_y_m",
        f"{leg}_z_m",
        f"{leg}_Fx_N",
        f"{leg}_Fy_N",
        f"{leg}_Fz_N",
    ]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(
        "CSV is missing columns: "
        + ", ".join(missing)
        + "\nRegenerate the CSV using the updated Core.step()."
    )

# Yaw is optional, but is required to display body orientation.
has_yaw = "yaw_rad" in df.columns

if has_yaw:
    df["yaw_rad"] = pd.to_numeric(df["yaw_rad"], errors="coerce")

df = df.dropna(subset=required).sort_values("time_s")
df = df.reset_index(drop=True)

if df.empty:
    raise ValueError("No valid samples found in CSV.")

if has_yaw and df["yaw_rad"].isna().any():
    print("Warning: Some yaw values are invalid; heading may be unavailable.")

if not has_yaw:
    print(
        "Warning: CSV has no yaw_rad column. "
        "The body's forward direction cannot be displayed.\n"
        'Add "yaw_rad": float(s0.yaw) to the logged row in core.py '
        "and regenerate the CSV."
    )

# ------------------------------------------------------------
# Extract data
# ------------------------------------------------------------

# Shape: (frames, legs, xyz)
positions = np.stack(
    [
        df[[f"{leg}_{axis}_m" for axis in ["x", "y", "z"]]]
        .to_numpy(dtype=float)
        for leg in LEGS
    ],
    axis=1,
)

forces = np.stack(
    [
        df[[f"{leg}_F{axis}_N" for axis in ["x", "y", "z"]]]
        .to_numpy(dtype=float)
        for leg in LEGS
    ],
    axis=1,
)

times = df["time_s"].to_numpy(dtype=float)
swing_legs = df["swing_leg"].astype(str).str.upper().to_numpy()

com_positions = df[
    ["com_x_m", "com_y_m", "com_z_m"]
].to_numpy(dtype=float)

# print("CoM start:", com_positions[0])
# print("CoM end:  ", com_positions[-1])
# if has_yaw:
#     print("Yaw range (deg):", np.degrees(df["yaw_rad"].min()), np.degrees(df["yaw_rad"].max()))

# # Foot positions relative to CoM, over time
# rel = positions - com_positions[:, None, :]
# print("Foot-minus-CoM range per axis:", rel.min(axis=(0, 1)), rel.max(axis=(0, 1)))

# # Print total commanded foot displacement.
# print("\nNet commanded foot displacement:")

# for i, leg in enumerate(LEGS):
#     delta = positions[-1, i] - positions[0, i]

#     print(
#         f"{leg}: "
#         f"dx={delta[0]:+.4f} m, "
#         f"dy={delta[1]:+.4f} m, "
#         f"dz={delta[2]:+.4f} m"
#     )

# body_delta = com_positions[-1] - com_positions[0]

# print(
#     "\nBody/CoM displacement: "
#     f"dx={body_delta[0]:+.4f} m, "
#     f"dy={body_delta[1]:+.4f} m, "
#     f"dz={body_delta[2]:+.4f} m"
# )

# ------------------------------------------------------------
# Figure and fixed axes
# ------------------------------------------------------------

# Include both feet and the body when calculating axis limits.
all_points = np.concatenate(
    [positions.reshape(-1, 3), com_positions],
    axis=0,
)

lo = all_points.min(axis=0)
hi = all_points.max(axis=0)

center = (lo + hi) / 2
span = max(float(np.max(hi - lo)), 0.25)
half = span * 0.8

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

ax.set_xlim(center[0] - half, center[0] + half)
ax.set_ylim(center[1] - half, center[1] + half)
ax.set_zlim(
    min(0.0, center[2] - half),
    center[2] + half,
)

ax.set_xlabel("World X (m)")
ax.set_ylabel("World Y (m)")
ax.set_zlabel("World Z (m)")
ax.set_title("Quadruped Ground Reaction Forces")

# Keep the world axes visually equally scaled.
ax.set_box_aspect((1, 1, 1))

# Ground reference plane.
xx, yy = np.meshgrid(
    [center[0] - half, center[0] + half],
    [center[1] - half, center[1] + half],
)

ax.plot_surface(
    xx,
    yy,
    np.zeros_like(xx),
    alpha=0.08,
)

# ------------------------------------------------------------
# Animation artists
# ------------------------------------------------------------

com_marker, = ax.plot(
    [], [], [],
    "ko",
    markersize=7,
    label="Body CoM",
)

foot_markers = []
foot_labels = []
body_lines = []

for leg in LEGS:
    marker, = ax.plot(
        [], [], [],
        "o",
        color=LEG_COLORS[leg],
        markersize=7,
    )
    foot_markers.append(marker)

    label = ax.text(0, 0, 0, leg)
    foot_labels.append(label)

    line, = ax.plot(
        [], [], [],
        color=LEG_COLORS[leg],
        alpha=0.45,
        linewidth=1.5,
    )
    body_lines.append(line)

force_arrows = []
body_heading = None

title = ax.set_title("")


def update(frame):
    global force_arrows, body_heading

    # Remove force arrows from the previous frame.
    for arrow in force_arrows:
        arrow.remove()

    force_arrows = []

    # Remove the previous body-heading arrow.
    if body_heading is not None:
        body_heading.remove()
        body_heading = None

    p = positions[frame]
    f = forces[frame]
    com = com_positions[frame]

    # Update body marker.
    com_marker.set_data([com[0]], [com[1]])
    com_marker.set_3d_properties([com[2]])

    # Show the body's forward direction.
    # Assumption: body-forward is local +X, and yaw is a
    # conventional rotation about world Z.
    if has_yaw:
        yaw = df.loc[frame, "yaw_rad"]

        if np.isfinite(yaw):
            hx = HEADING_LENGTH_M * np.cos(yaw)
            hy = HEADING_LENGTH_M * np.sin(yaw)

            body_heading = ax.quiver(
                com[0],
                com[1],
                com[2],
                hx,
                hy,
                0.0,
                color="black",
                linewidth=2.5,
                arrow_length_ratio=0.2,
            )

    for i, leg in enumerate(LEGS):
        x, y, z = p[i]
        fx, fy, fz = f[i]

        # Foot marker.
        foot_markers[i].set_data([x], [y])
        foot_markers[i].set_3d_properties([z])

        # Foot label.
        foot_labels[i].set_position((x, y))
        foot_labels[i].set_3d_properties(z, zdir="z")

        # Line from body CoM to commanded foot position.
        body_lines[i].set_data(
            [com[0], x],
            [com[1], y],
        )
        body_lines[i].set_3d_properties([com[2], z])

        # Force arrow starts at the foot and points along
        # the logged world-frame force vector.
        arrow = ax.quiver(
            x,
            y,
            z,
            fx * FORCE_SCALE,
            fy * FORCE_SCALE,
            fz * FORCE_SCALE,
            color=LEG_COLORS[leg],
            linewidth=2,
            arrow_length_ratio=0.2,
        )

        force_arrows.append(arrow)

    swing = swing_legs[frame]

    heading_text = ""

    if has_yaw and np.isfinite(df.loc[frame, "yaw_rad"]):
        yaw = df.loc[frame, "yaw_rad"]
        heading_text = f" | Yaw: {np.degrees(yaw):.1f} deg"

    title.set_text(
        f"t = {times[frame]:.3f} s | "
        f"Airborne leg: {swing} | "
        f"Force scale: {FORCE_SCALE:g} m/N"
        f"{heading_text}"
    )

    return (
        [com_marker, *foot_markers, *foot_labels, *body_lines, title]
    )


ani = FuncAnimation(
    fig,
    update,
    frames=len(df),
    interval=INTERVAL_MS / PLAYBACK_SPEED,
    blit=False,
    repeat=False,
)

# ani.save("quadruped_forces.gif", writer="pillow", fps=30)

plt.tight_layout()
plt.show()

# Optional export:
# ani.save("quadruped_forces.gif", writer="pillow", fps=30)
# ani.save("quadruped_forces.mp4", writer="ffmpeg", fps=30)
