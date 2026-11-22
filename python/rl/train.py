# PPO fine-tuning with rsl_rl in the MuJoCo Playground env, optionally initialised from the BC actor.
#   python python/rl/train.py --bc_ckpt python/rl/data/bc_actor.pt --num_envs 2048
# Needs: pip install "rsl-rl-lib>=5" mujoco-playground tensordict pyyaml  (+ jax[cuda] for GPU MJX)
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")   # let torch and JAX share the GPU

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.config import CONFIG_PATH, load_config, rsl_rl_train_cfg
from rl.environments.playground_env import QuadrupedEnv
from rl.environments.rsl_wrapper import PlaygroundVecEnv
from rl.models import load_bc_into_actor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_envs", type=int, default=64)
    ap.add_argument("--iterations", type=int, default=None, help="default: runner.max_iterations")
    ap.add_argument("--bc_ckpt", default=None, help="BC actor checkpoint from bc_pretrain.py")
    ap.add_argument("--resume", default=None, help="rsl_rl checkpoint (model_*.pt) to continue from")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--config", default=str(CONFIG_PATH))
    args = ap.parse_args()

    cfg = load_config(args.config)
    train_cfg = rsl_rl_train_cfg(cfg)
    env = PlaygroundVecEnv(QuadrupedEnv(), args.num_envs, seed=train_cfg["seed"], device=args.device)

    log_dir = Path("logs/rsl_rl") / train_cfg["experiment_name"] / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, log_dir / "model.yaml")          # eval.py / reproducibility

    runner = OnPolicyRunner(env, train_cfg, log_dir=str(log_dir), device=args.device)
    if args.resume:
        runner.load(args.resume)
    elif args.bc_ckpt:
        actor = getattr(runner.alg, "actor", None) or runner.alg.policy.actor    # 5.x / older layout
        load_bc_into_actor(actor, args.bc_ckpt, args.device)
    runner.learn(num_learning_iterations=args.iterations or train_cfg["max_iterations"])


if __name__ == "__main__":
    main()