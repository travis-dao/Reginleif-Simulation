# sim/util/force_logger.py
import numpy as np


class ForceLogger:
    """Logs MPC foot ground-reaction forces (world frame, N) over time.

    Call it every sim tick: force_log(core, data.time). A row is recorded only
    when core.last_result is a new, successful solve.
    """
    LEGS = ("FL", "FR", "BR", "BL")

    def __init__(self):
        self.t = []
        self.forces = []     # each (4, 3)
        self.swing = []      # each (4,) bool, True = leg airborne
        self.offsets = []    # each (4, 3), applied foot offsets (mm, leg frame)
        self._last = None

    def __call__(self, core, t: float) -> None:
        res = core.last_result
        if res is None or res is self._last or not res.success:
            return
        self._last = res

        swinging = set(core.lc.get_swing_legs())
        self.t.append(t)
        self.forces.append(np.array(res.u0.foot_ground_force, dtype=float).reshape(4, 3))
        self.swing.append(np.array([i in swinging for i in range(4)]))
        self.offsets.append(np.array([[o.x, o.y, o.z] for o in core._offsets]))

    def save(self, path: str = "foot_forces.npz") -> None:
        np.savez(path,
                 t=np.array(self.t),
                 forces=np.array(self.forces),     # (T, 4, 3)
                 swing=np.array(self.swing),       # (T, 4)
                 offsets=np.array(self.offsets))   # (T, 4, 3)
        print(f"saved {len(self.t)} force samples to {path}")

    def plot(self) -> None:
        import matplotlib.pyplot as plt

        if not self.t:
            print("no force samples logged")
            return
        t = np.array(self.t)
        F = np.array(self.forces)
        S = np.array(self.swing)

        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 9))
        for i, ax in enumerate(axes):
            ax.plot(t, F[:, i, 0], label="Fx")
            ax.plot(t, F[:, i, 1], label="Fy")
            ax.plot(t, F[:, i, 2], label="Fz")
            # shade the swing phases
            in_swing = False
            for k in range(len(t)):
                if S[k, i] and not in_swing:
                    t_start, in_swing = t[k], True
                elif not S[k, i] and in_swing:
                    ax.axvspan(t_start, t[k], color="gray", alpha=0.2)
                    in_swing = False
            if in_swing:
                ax.axvspan(t_start, t[-1], color="gray", alpha=0.2)
            ax.set_ylabel(f"{self.LEGS[i]} (N)")
            ax.grid(True)
        axes[0].legend(loc="upper right", ncol=3)
        axes[-1].set_xlabel("time (s)")
        fig.suptitle("MPC foot forces (gray = swing)")
        plt.tight_layout()
        plt.show()