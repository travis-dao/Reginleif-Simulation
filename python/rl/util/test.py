import sys
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PYTHON_DIR))


import numpy as np, glob
from rl.config import OBS
err_copy, err_zero, err_mean, n = 0, 0, 0, 0
acts = [np.load(f)["act"] for f in sorted(glob.glob("rl/data/mpc/*.npz")) if len(np.load(f)["obs"])]
mean = np.concatenate(acts).mean(0)
for a in acts:
    err_copy += ((a[1:] - a[:-1]) ** 2).mean(-1).sum()      # predict "same as last action"
    err_zero += (a ** 2).mean(-1).sum()                      # predict q_default
    err_mean += ((a - mean) ** 2).mean(-1).sum()
    n += len(a)
print("copy-last:", err_copy / n, " zeros:", err_zero / n, " mean:", err_mean / n)