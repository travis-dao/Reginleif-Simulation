# PPO fine-tuning with rsl_rl in the MuJoCo Playground env, initialised from the BC actor and (optionally) kept
# close to it on baseline-like commands by rl/bc_reg.py.
#
#   python python/rl/train.py --bc_ckpt python/rl/data/bc_actor.pt --num_envs 2048
#
# Curriculum = successive runs with config overrides, each resuming the previous run, e.g.
#   stage 1 (speed + braking/reversals, DR on):   python python/rl/train.py --bc_ckpt BC --iterations 800
#   stage 2 (add lateral):  python python/rl/train.py --bc_ckpt BC --resume logs/.../model_800.pt \\
#                               --set env.cmd_vy_range=[-0.08,0.08] --iterations 800
#   stage 3 (faster):       ... --set env.cmd_vx_range=[-0.2,0.2] env.cmd_vy_range=[-0.15,0.15]
# Needs: pip install "rsl-rl-lib>=5" mujoco-playground tensordict pyyaml  (+ jax[cuda] for GPU MJX)
# python python/rl/train.py --bc_ckpt python/rl/data/bc_actor.pt --num_envs 64 --iterations 100 --set bc_reg.lr=1e-4 bc_reg.steps_per_iter=10
# python python/rl/train.py --bc_ckpt python/rl/data/bc_actor_v2.pt --num_envs 64 --iterations 100
# tensorboard --logdir logs/rsl_rl
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")   # let torch and JAX share the GPU

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml
from rsl_rl.runners import OnPolicyRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.bc_reg import BCRegularizer
from rl.config import CONFIG_PATH, load_config, rsl_rl_train_cfg
from rl.environments.playground_env import QuadrupedEnv, default_config
from rl.environments.rsl_wrapper import PlaygroundVecEnv
from rl.models import load_bc_into_actor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_envs", type=int, default=64)
    ap.add_argument("--iterations", type=int, default=None, help="default: runner.max_iterations")
    ap.add_argument("--bc_ckpt", default=None, help="BC actor checkpoint: initial weights and the bc_reg reference")
    ap.add_argument("--resume", default=None, help="rsl_rl checkpoint (model_*.pt) to continue from (curriculum stages)")
    ap.add_argument("--no_bc_reg", action="store_true", help="disable the BC regulariser")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL", help="config overrides, e.g. env.cmd_vy_range=[-0.1,0.1]")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--out")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    train_cfg = rsl_rl_train_cfg(cfg)
    env = PlaygroundVecEnv(QuadrupedEnv(default_config(cfg)), args.num_envs, seed=train_cfg["seed"], device=args.device)

    # log_dir = Path("logs/rsl_rl") / train_cfg["experiment_name"] / datetime.now().strftime("%Y-%m-%d_%H")
    log_dir = Path("logs/rsl_rl") / args.out
    log_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, log_dir / "model.yaml")
    (log_dir / "overrides.yaml").write_text(yaml.safe_dump({"set": args.set, "resume": args.resume, "bc_ckpt": args.bc_ckpt}))

    runner = OnPolicyRunner(env, train_cfg, log_dir=str(log_dir), device=args.device)
    if args.resume:
        runner.load(args.resume)
    elif args.bc_ckpt:
        actor = getattr(runner.alg, "actor", None) or runner.alg.policy.actor    # 5.x / older layout
        load_bc_into_actor(actor, args.bc_ckpt, args.device)

    if cfg["bc_reg"]["enabled"] and not args.no_bc_reg:
        if not args.bc_ckpt:
            raise SystemExit("bc_reg is enabled but --bc_ckpt was not given (it is the frozen reference). "
                             "Pass --bc_ckpt or use --no_bc_reg.")
        BCRegularizer(runner, env, cfg, args.bc_ckpt, args.device)

    runner.learn(num_learning_iterations=args.iterations or train_cfg["max_iterations"])


if __name__ == "__main__":
    main()