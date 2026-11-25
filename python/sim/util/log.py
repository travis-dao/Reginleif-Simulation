# sim/util/logger.py
import numpy as np


class Logger:
    """Logs MPC foot ground-reaction forces (world frame, N) plus the body roll and
    pitch (rad) from core.state, over time.

    Call it every sim tick, right after core.step(): log(core, data.time).
    A row is recorded only when core.last_result is a new, successful solve.
    """
    LEGS = ("FL", "FR", "BR", "BL")

    def __init__(self):
        self.t = []
        self.forces = []     # each (4, 3)
        self.swing = []      # each (4,) bool, True = leg airborne
        self.offsets = []    # each (4, 3), applied foot offsets (mm, leg frame)
        self.roll = []        # rad, core.state.roll
        self.pitch = []      # rad, core.state.pitch
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
        self.roll.append(float(core.state.roll))
        self.pitch.append(float(core.state.pitch))

    def save(self, path: str = "log.npz") -> None:
        np.savez(path,
                 t=np.array(self.t),
                 forces=np.array(self.forces),     # (T, 4, 3)
                 swing=np.array(self.swing),       # (T, 4)
                 offsets=np.array(self.offsets),   # (T, 4, 3)
                 roll=np.array(self.roll),           # (T,) rad
                 pitch=np.array(self.pitch))       # (T,) rad
        print(f"saved {len(self.t)} samples to {path}")

    @staticmethod
    def _shade_swing(ax, t, swing_i) -> None:
        in_swing = False
        for k in range(len(t)):
            if swing_i[k] and not in_swing:
                t_start, in_swing = t[k], True
            elif not swing_i[k] and in_swing:
                ax.axvspan(t_start, t[k], color="gray", alpha=0.2)
                in_swing = False
        if in_swing:
            ax.axvspan(t_start, t[-1], color="gray", alpha=0.2)

    def plot(self) -> None:
        import matplotlib.pyplot as plt

        if not self.t:
            print("no samples logged")
            return
        t = np.array(self.t)
        F = np.array(self.forces)
        S = np.array(self.swing)

        # --- foot forces ---
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 9))
        for i, ax in enumerate(axes):
            ax.plot(t, F[:, i, 0], label="Fx")
            ax.plot(t, F[:, i, 1], label="Fy")
            ax.plot(t, F[:, i, 2], label="Fz")
            self._shade_swing(ax, t, S[:, i])
            ax.set_ylabel(f"{self.LEGS[i]} (N)")
            ax.grid(True)
        axes[0].legend(loc="upper right", ncol=3)
        axes[-1].set_xlabel("time (s)")
        fig.suptitle("MPC foot forces (gray = swing)")
        plt.tight_layout()

        # --- roll / pitch ---
        roll = np.degrees(np.unwrap(self.roll))
        pitch = np.degrees(self.pitch)
        fig2, (ax_y, ax_p) = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
        ax_y.plot(t, roll)
        ax_y.set_ylabel("roll (deg)")
        ax_p.plot(t, pitch)
        ax_p.set_ylabel("pitch (deg)")
        ax_p.set_xlabel("time (s)")
        for ax in (ax_y, ax_p):
            ax.grid(True)
            # shade the times when each leg is in swing (stacked alpha = more legs in swing)
            for i in range(4):
                self._shade_swing(ax, t, S[:, i])
        fig2.suptitle("Body roll / pitch (darker gray = more legs in swing)")
        plt.tight_layout()
        plt.show()