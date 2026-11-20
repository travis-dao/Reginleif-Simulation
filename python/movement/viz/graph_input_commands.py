import math
import matplotlib.pyplot as plt

points = [
    ('walk_forward_left_45deg', -0.7071067811865475, 0.7071067811865476),
    ('walk_forward_left_36deg', -0.5877852522924731, 0.8090169943749475),
    ('walk_forward_left_27deg', -0.4539904997395468, 0.8910065241883679),
    ('walk_forward_left_18deg', -0.3090169943749474, 0.9510565162951535),
    ('walk_forward_left_9deg', -0.15643446504023087, 0.9876883405951378),
    ('walk_forward', 0.0, 1.0),
    ('walk_forward_right_9deg', 0.15643446504023087, 0.9876883405951378),
    ('walk_forward_right_18deg', 0.3090169943749474, 0.9510565162951535),
    ('walk_forward_right_27deg', 0.4539904997395468, 0.8910065241883679),
    ('walk_forward_right_36deg', 0.5877852522924731, 0.8090169943749475),
    ('walk_forward_right_45deg', 0.7071067811865475, 0.7071067811865476),
    ('walk_backward_left_45deg', -0.7071067811865475, -0.7071067811865476),
    ('walk_backward_left_36deg', -0.5877852522924731, -0.8090169943749475),
    ('walk_backward_left_27deg', -0.4539904997395468, -0.8910065241883679),
    ('walk_backward_left_18deg', -0.3090169943749474, -0.9510565162951535),
    ('walk_backward_left_9deg', -0.15643446504023087, -0.9876883405951378),
    ('walk_backward', 0.0, -1.0),
    ('walk_backward_right_9deg', 0.15643446504023087, -0.9876883405951378),
    ('walk_backward_right_18deg', 0.3090169943749474, -0.9510565162951535),
    ('walk_backward_right_27deg', 0.4539904997395468, -0.8910065241883679),
    ('walk_backward_right_36deg', 0.5877852522924731, -0.8090169943749475),
    ('walk_backward_right_45deg', 0.7071067811865475, -0.7071067811865476),
]

# Extract coordinates and labels
labels = [p[0] for p in points]
x = [p[1] for p in points]
y = [p[2] for p in points]

fig, ax = plt.subplots(figsize=(9, 9))

# Plot points
ax.scatter(x, y, s=60, zorder=3)

# Annotate each point
for label, xi, yi in points:
    ax.annotate(
        label.replace("walk_", ""),
        (xi, yi),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=8,
    )

# Unit circle for reference
circle = plt.Circle(
    (0, 0), 1, fill=False, linestyle="--", linewidth=1
)
ax.add_patch(circle)

# Axes and grid
ax.axhline(0, color="gray", linewidth=0.8)
ax.axvline(0, color="gray", linewidth=0.8)
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.3, 1.3)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("Joystick Input X")
ax.set_ylabel("Joystick Input Y")
ax.set_title("Joystick Movement Input Points")
ax.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()
