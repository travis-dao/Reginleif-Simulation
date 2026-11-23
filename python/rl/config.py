"""Loads rl/configs/model.yaml and exposes the observation spec + the rsl_rl runner config."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import yaml

RL_DIR = Path(__file__).resolve().parent
CONFIG_PATH = RL_DIR / "configs" / "model.yaml"
Q_DEFAULT_PATH = RL_DIR / "data" / "q_default.npy"
NUM_JOINTS = 12


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    """overrides: ["env.cmd_vy_range=[-0.1,0.1]", "ppo.learning_rate=5e-5", ...] (values parsed as YAML).
    Overriding `obs.*` changes the obs size and is not supported here (edit model.yaml and regenerate data)."""
    with open(path or CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, _, val = item.partition("=")
        node = cfg
        *parents, leaf = key.strip().split(".")
        for p in parents:
            node = node[p]
        if leaf not in node:
            raise KeyError(f"unknown config key: {key}")
        v = yaml.safe_load(val)
        if isinstance(v, str):                      # PyYAML reads "5e-5" as a string
            try:
                v = float(v)
            except ValueError:
                pass
        node[leaf] = v
    return cfg


@dataclass(frozen=True)
class ObsSpec:
    policy_hz: float
    history_len: int          # number of past actions in the observation
    action_scale: float
    gyro_scale: float
    cmd_scale_lin: float
    cmd_scale_ang: float
    qd_scale: float

    @classmethod
    def from_cfg(cls, cfg: dict) -> "ObsSpec":
        o, hz = cfg["obs"], float(cfg["policy_hz"])
        return cls(policy_hz=hz, history_len=int(round(o["action_history_s"] * hz)),
                   action_scale=float(o["action_scale"]), gyro_scale=float(o["gyro_scale"]),
                   cmd_scale_lin=float(o["cmd_scale_lin"]), cmd_scale_ang=float(o["cmd_scale_ang"]),
                   qd_scale=float(o["qd_scale"]))

    @property
    def obs_dim(self) -> int:
        return self.hist_start + self.history_len * NUM_JOINTS

    @property
    def vy_idx(self) -> int:           # obs column of the lateral command
        return 3 + 3 + 1

    @property
    def hist_start(self) -> int:       # first obs column of the action history
        return 3 + 3 + 3 + NUM_JOINTS + NUM_JOINTS


CFG = load_config()
OBS = ObsSpec.from_cfg(CFG)


def rsl_rl_train_cfg(cfg: dict | None = None) -> dict:
    """Runner config in the rsl_rl >= 5.x format (separate actor/critic models, obs_groups, distribution_cfg).
    Obs normalisation is off: observations are already scaled by ObsSpec, which keeps BC weights transferable."""
    cfg = cfg or CFG
    r, a, c = cfg["runner"], cfg["actor"], cfg["critic"]
    return {
        "seed": r["seed"],
        "num_steps_per_env": r["num_steps_per_env"],
        "max_iterations": r["max_iterations"],
        "save_interval": r["save_interval"],
        "experiment_name": r["experiment_name"],
        "run_name": r["run_name"],
        "logger": r["logger"],
        # actor sees only the deployable obs; critic additionally gets privileged info (body lin vel, base height)
        "obs_groups": {"actor": ["policy"], "critic": ["policy", "privileged"]},
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": list(a["hidden_dims"]),
            "activation": a["activation"],
            "obs_normalization": False,
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": a["init_std"], "std_type": "scalar"},
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": list(c["hidden_dims"]),
            "activation": c["activation"],
            "obs_normalization": False,
        },
        "algorithm": {"class_name": "PPO", **copy.deepcopy(cfg["ppo"])},
    }