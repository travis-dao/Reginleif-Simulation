# python python/rl/util/test.py

import sys
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PYTHON_DIR))

import numpy as np, glob
for f in sorted(glob.glob("python/rl/data/mpc/*.npz")):
    z = np.load(f)
    if abs(z["cmd"][0]) < 1e-3 and abs(z["cmd"][2]) > 0.5 and float(z["push_accel"]) == 0.0 and len(z["time"]) > 10:
        yaw = np.unwrap(z["base_yaw"])
        print(f.split("/")[-1], "cmd yaw %.3f -> achieved %.3f rad/s" % (z["cmd"][2], (yaw[-1] - yaw[0]) / (z["time"][-1] - z["time"][0])))