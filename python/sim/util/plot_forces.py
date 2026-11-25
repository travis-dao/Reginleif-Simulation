import numpy as np
import matplotlib.pyplot as plt

d = np.load("python/sim/log/foot_forces.npz")
t, F, S = d["t"], d["forces"], d["swing"]

mask = t >= 2.0
t, F, S = t[mask], F[mask], S[mask]

leg_names = ("FL", "FR", "BR", "BL")
force_names = ("Fx", "Fy", "Fz")

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 7))

for j, (ax, force_name) in enumerate(zip(axes, force_names)):
    for i, leg_name in enumerate(leg_names):
        ax.plot(t, F[:, i, j], label=leg_name)

    ax.set_ylabel(f"{force_name} (N)")
    ax.set_title(f"{force_name} — All Legs")
    ax.legend(ncol=4)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time (s)")

plt.tight_layout()
plt.show()