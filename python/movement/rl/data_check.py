# python python/movement/rl/data_check.py

# import numpy as np, json
# d = np.load("python/movement/rl/dataset/dataset_20261118_191257.npz")
# s = json.loads(str(d["meta"]))["action_scale_rad"]
# q_rel = d["obs"][:, 90:135][:, 9:21]        # measured q - default (newest frame)
# err = q_rel - d["actions"] * s              # measured minus commanded
# t = d["time"]

# late = t > t[0] + 1.5                       # skip the landing transient
# print("max |q_rel| after 1.5 s:", np.abs(q_rel[late]).max(0).round(2))
# print("mean err (offset?)     :", err[late].mean(0).round(3))
# print("99th |err| (lag)       :", np.percentile(np.abs(err[late]), 99, axis=0).round(3))

import numpy as np, glob
for f in glob.glob("python/movement/rl/datasets/*.npz"):
    d = np.load(f)
    a, prev = d["actions"], d["obs"][:, 123:135]      # prev action in the newest frame
    ok = d["mpc_ok"]
    print(f.split("/")[-1], "copy-prev MSE:", ((a - prev)[ok] ** 2).mean())