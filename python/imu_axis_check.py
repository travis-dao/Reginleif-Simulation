# mjpython python/imu_axis_check.py

import time
import math
import argparse
import numpy as np
import mujoco
import mujoco.viewer

from motion.state_estimator import make_imu
from build_sim_xml import build_sim_xml

SRC = "assembly/xml/reginleif.xml"
DST = "assembly/xml/reginleif_sim.xml"

KP = 30.0
FORCE = 9.4
START_HEIGHT = 0.5

PHASE_S = 8.0          # seconds per phase
ROLL_AMP = 30.0        # deg
PITCH_AMP = 30.0       # deg
YAW_AMP = 60.0         # deg
PRINT_HZ = 10.0
FRAME_DT = 1.0 / 60.0  # purely kinematic, so we just run at display rate

# (name, [roll_amp, pitch_amp, yaw_amp] in deg, [roll, pitch, yaw] frequency multipliers)
PHASES = [
    ("ROLL  (about body X)", [ROLL_AMP, 0, 0],               [1, 0, 0]),
    ("PITCH (about body Y)", [0, PITCH_AMP, 0],              [0, 1, 0]),
    ("YAW   (about body Z)", [0, 0, YAW_AMP],                [0, 0, 1]),
    ("COMBINED",             [ROLL_AMP / 2, PITCH_AMP / 2, YAW_AMP / 2], [1, 1.5, 0.5]),
]


def rpy_to_quat(roll, pitch, yaw):
    """ZYX (yaw-pitch-roll) convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll). Returns wxyz quat."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def draw_axes(viewer, model, data, body_id, length=0.25, width=0.01):
    """RGB arrows on the base: X=red, Y=green, Z=blue (body axes)."""
    scn = viewer.user_scn
    scn.ngeom = 0
    origin = data.xpos[body_id].copy()
    R = data.xmat[body_id].reshape(3, 3)
    colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]
    for i in range(3):
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                            np.zeros(9), np.array(colors[i], dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width,
                             origin, origin + R[:, i] * length)
        scn.ngeom += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hold-height", type=float, default=START_HEIGHT,
                        help="height (m) to hold the base at")
    args = parser.parse_args()

    build_sim_xml(SRC, DST, START_HEIGHT, KP, FORCE)
    model = mujoco.MjModel.from_xml_path(DST)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    jnt = model.body_jntadr[body_id]
    qadr = model.jnt_qposadr[jnt]
    vadr = model.jnt_dofadr[jnt]

    hold_pos = data.qpos[qadr:qadr + 3].copy()
    hold_pos[2] = args.hold_height

    print("Commanded rpy is applied to the base as R = Rz(yaw) @ Ry(pitch) @ Rx(roll).")
    print("Expect the IMU to track the commanded angle on the SAME axis with the SAME sign.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for name, amps, freqs in PHASES:
            print(f"\n=== {name} ===")
            max_meas = np.zeros(3)         # max |measured| per axis
            max_err = np.zeros(3)          # max |measured - commanded| per axis
            corr_num = np.zeros(3)         # for sign check
            last_print = -1.0
            t0 = time.time()

            while viewer.is_running():
                t = time.time() - t0
                if t > PHASE_S:
                    break

                # commanded angles (deg), smooth sweep
                cmd = np.array([amps[i] * math.sin(2 * math.pi * freqs[i] * t / (PHASE_S / 2))
                                for i in range(3)])

                # freeze: no mj_step, just place the base and recompute kinematics
                data.qpos[qadr:qadr + 3] = hold_pos
                data.qpos[qadr + 3:qadr + 7] = rpy_to_quat(*np.deg2rad(cmd))
                data.qvel[vadr:vadr + 6] = 0.0
                mujoco.mj_forward(model, data)

                imu = make_imu(model, data, body_id, undo_mount=False)
                meas = np.rad2deg(np.asarray(imu.rpy_rad, dtype=float))

                err = np.array([wrap_deg(meas[i] - cmd[i]) for i in range(3)])
                max_meas = np.maximum(max_meas, np.abs(meas))
                max_err = np.maximum(max_err, np.abs(err))
                corr_num += meas * cmd

                draw_axes(viewer, model, data, body_id)
                viewer.sync()

                if t - last_print >= 1.0 / PRINT_HZ:
                    last_print = t
                    print(
                        f"cmd rpy=[{cmd[0]:7.1f} {cmd[1]:7.1f} {cmd[2]:7.1f}] | "
                        f"imu rpy=[{meas[0]:7.1f} {meas[1]:7.1f} {meas[2]:7.1f}] | "
                        f"err=[{err[0]:6.1f} {err[1]:6.1f} {err[2]:6.1f}]",
                        flush=True,
                    )
                time.sleep(FRAME_DT)

            if not viewer.is_running():
                return

            # per-phase verdict
            axes = ["roll", "pitch", "yaw"]
            print("--- summary ---")
            for i, ax in enumerate(axes):
                if amps[i] > 0:
                    sign = "same sign" if corr_num[i] > 0 else "INVERTED"
                    ok = max_err[i] < 2.0
                    print(f"{ax:5s}: max err {max_err[i]:5.1f} deg, {sign} -> {'OK' if ok and corr_num[i] > 0 else 'CHECK AXIS/SIGN'}")
                else:
                    leak = max_meas[i]
                    print(f"{ax:5s}: leakage {leak:5.1f} deg -> {'OK' if leak < 2.0 else 'CROSSTALK (axes swapped or misframed?)'}")

        print("\nDone. Close the viewer to exit.")
        while viewer.is_running():
            time.sleep(0.1)


if __name__ == "__main__":
    main()