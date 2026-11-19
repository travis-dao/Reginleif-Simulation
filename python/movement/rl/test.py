import numpy as np, torch, importlib.util, glob
ck = torch.load("python/movement/rl/runs/20261118_205831/policy.pt", map_location="cpu", weights_only=False)
spec = importlib.util.spec_from_file_location("bc", "python/movement/rl/train.py")
bc = importlib.util.module_from_spec(spec); spec.loader.exec_module(bc)
sd = ck["state_dict"]
net = bc.Policy(sd["mean"].numpy(), sd["std"].numpy(), ck["clip"], ck["act"])
net.load_state_dict(sd); net.eval()
pred = lambda x: net(torch.from_numpy(x).float()).detach().numpy()

d = np.load(glob.glob("python/movement/rl/datasets/*.npz")[0])
ok = d["mpc_ok"]; X, A = d["obs"][ok], d["actions"][ok]

# 1) does it react to the command?
X0 = X[np.abs(d["cmd"][ok][:, 0]) > 1e-3].copy()
Xz = X0.copy()
for f in range(3):
    Xz[:, f * 45 + 6: f * 45 + 9] = 0                      # zero cmd in all 3 frames
print("mean |action change| when command zeroed:", np.abs(pred(X0) - pred(Xz)).mean())
print("typical |action|:", np.abs(A).mean())

# 2) error at fast action changes (swing onsets) vs the rest
prev = X[:, 123:135]
jump = np.abs(A - prev).max(1)
hi = jump > np.percentile(jump, 95)
err = ((pred(X) - A) ** 2).mean(1)
print("MSE top-5% action changes:", err[hi].mean(), "| rest:", err[~hi].mean())
print("copy-prev MSE:", ((A - prev) ** 2).mean(), "| model MSE:", err.mean())