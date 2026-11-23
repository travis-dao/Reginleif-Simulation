# The MPC saturates its in-place turn rate (command +-0.667 rad/s -> it actually turns at ~0.34 rad/s).
# This rewrites the yaw command stored in turn-in-place episodes to the rate the MPC actually achieved, so
# "command = velocity in rad/s" is honest. Only the yaw-command column of the observation (and the stored cmd)
# changes; no MPC re-run is needed. The source directory is left untouched.
#
#   python python/rl/util/relabel_turn_cmd.py --src python/rl/data/mpc       --dst python/rl/data/mpc_relabeled
#   python python/rl/util/relabel_turn_cmd.py --src python/rl/data/mpc_train --dst python/rl/data/mpc_train_relabeled
# Use --yaw_rate on the second call (printed by the first) so test and train get the same value.
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rl.config import OBS

YAW_COL = 8        # obs layout: gyro(3) | gravity(3) | cmd [vx, vy, yaw](3) -> yaw is column 8


def is_turn(cmd, min_yaw):
    return abs(cmd[0]) < 1e-3 and abs(cmd[1]) < 1e-3 and abs(cmd[2]) > min_yaw


def measure_rate(src: Path, min_yaw: float, skip_s: float) -> float:
    rates = []
    for f in sorted(src.glob("*.npz")):
        z = np.load(f)
        if len(z["time"]) < 20 or not is_turn(z["cmd"], min_yaw) or float(z["push_accel"]) != 0.0 or bool(z["fell"]):
            continue
        t, yaw = z["time"], np.unwrap(z["base_yaw"])
        m = t > t[0] + skip_s                                    # skip the start-up transient
        rates.append(abs((yaw[m][-1] - yaw[m][0]) / (t[m][-1] - t[m][0])))
    if not rates:
        raise SystemExit("no unpushed turn-in-place episodes found to measure the achieved rate; pass --yaw_rate")
    return float(np.mean(rates))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--yaw_rate", type=float, default=None, help="rad/s; default: measured from unpushed turn episodes in --src")
    ap.add_argument("--min_yaw", type=float, default=0.5, help="|yaw cmd| above this with vx=vy=0 counts as turn-in-place")
    ap.add_argument("--skip_s", type=float, default=2.0, help="seconds ignored at the start when measuring the rate")
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    if src.resolve() == dst.resolve():
        raise SystemExit("--dst must differ from --src (this script never overwrites the source)")
    dst.mkdir(parents=True, exist_ok=True)

    rate = args.yaw_rate if args.yaw_rate is not None else measure_rate(src, args.min_yaw, args.skip_s)
    print(f"relabelling turn-in-place yaw commands to +-{rate:.4f} rad/s")
    n = 0
    for f in sorted(src.glob("*.npz")):
        z = dict(np.load(f))
        if is_turn(z["cmd"], args.min_yaw) and len(z["obs"]):
            new = float(np.sign(z["cmd"][2]) * rate)
            z["obs"] = z["obs"].copy()
            z["obs"][:, YAW_COL] = np.float32(new * OBS.cmd_scale_ang)
            z["cmd"] = z["cmd"].copy(); z["cmd"][2] = np.float32(new)
            z["yaw_cmd"] = np.asarray(new)
            n += 1
        np.savez_compressed(dst / f.name, **z)
    for extra in ("manifest.json", "spec.json"):
        if (src / extra).exists():
            shutil.copy(src / extra, dst / extra)
    print(f"{n} turn-in-place files relabelled, {len(list(src.glob('*.npz'))) - n} copied unchanged -> {dst}")


if __name__ == "__main__":
    main()