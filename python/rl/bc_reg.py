"""Behavior regularisation toward the frozen BC (MPC-clone) baseline during PPO.

After every PPO update we take a few supervised steps that pull the actor's mean action toward the frozen baseline
network on the states PPO just visited, but ONLY where the command lies inside the MPC's range (v_y == 0, |v_x| and
|yaw| below the baseline limits). New skills (lateral, fast, braking beyond the baseline) are left to PPO.
This is DAgger-style distillation on the student's own state distribution, so it keeps push robustness that
pure reward-driven PPO tends to trade away.

Implemented as a hook on `runner.alg.update`, not as a change to rsl_rl, and the actor forward is computed from
its Linear layers (rl.models.actor_mean), so it does not depend on rsl_rl's model.forward() signature.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from rl.config import OBS, NUM_JOINTS
from rl.models import actor_mean, make_mlp


class BCRegularizer:
    def __init__(self, runner, env, cfg: dict, bc_ckpt: str, device: str):
        self.cfg, self.env, self.device, self.it = cfg["bc_reg"], env, torch.device(device), 0
        self.activation = cfg["actor"]["activation"]
        blob = torch.load(bc_ckpt, map_location=device)
        self.ref = make_mlp(blob["obs_dim"], blob["hidden_dims"], NUM_JOINTS, blob["activation"]).to(device)
        self.ref.load_state_dict(blob["bc_state_dict"])
        self.ref.eval().requires_grad_(False)

        self.actor = getattr(runner.alg, "actor", None) or runner.alg.policy.actor
        lins = [m for m in self.actor.modules() if isinstance(m, nn.Linear)]
        self.opt = torch.optim.Adam([p for m in lins for p in m.parameters()], lr=self.cfg["lr"])

        self._check_actor_forward(runner)
        env.log_obs = True
        orig = runner.alg.update

        def update_with_reg(*a, **k):
            out = orig(*a, **k)
            self.step()
            return out

        runner.alg.update = update_with_reg
        c = self.cfg
        # baseline-command mask in scaled-obs units: cmd columns are 6 (vx), 7 (vy), 8 (yaw)
        self.vx_max = c["baseline_vx_abs_max"] * OBS.cmd_scale_lin
        self.yaw_max = c["baseline_yaw_abs_max"] * OBS.cmd_scale_ang
        print(f"[bc_reg] enabled: coef={c['coef']} lr={c['lr']} steps/iter={c['steps_per_iter']} "
              f"baseline |vx|<={c['baseline_vx_abs_max']} |yaw|<={c['baseline_yaw_abs_max']} vy==0")

    def _check_actor_forward(self, runner):
        """Fail loudly if our functional forward disagrees with rsl_rl's actor (wrong layer order / normalisation)."""
        obs = self.env.get_observations()
        pol = runner.get_inference_policy(device=self.device)
        with torch.no_grad():
            a_ref, a_mine = pol(obs), actor_mean(self.actor, obs["policy"], self.activation)
        if not torch.allclose(a_ref, a_mine, atol=1e-4):
            raise RuntimeError("bc_reg: functional actor forward != rsl_rl inference policy "
                               f"(max abs diff {(a_ref - a_mine).abs().max().item():.2e}). Disable with bc_reg.enabled=false "
                               "or check actor.activation / obs_normalization.")

    def _coef(self):
        n = self.cfg["anneal_iters"]
        return self.cfg["coef"] * (max(0.0, 1.0 - self.it / n) if n > 0 else 1.0)

    def step(self):
        self.it += 1
        if not self.env.obs_log:
            return
        obs = torch.cat(self.env.obs_log)
        self.env.obs_log.clear()
        vx, vy, yaw = obs[:, 6], obs[:, 7], obs[:, 8]
        obs = obs[(vy.abs() < 1e-6) & (vx.abs() <= self.vx_max) & (yaw.abs() <= self.yaw_max)]
        coef = self._coef()
        if len(obs) == 0 or coef <= 0:
            return
        loss = torch.zeros(())
        for _ in range(self.cfg["steps_per_iter"]):
            xb = obs[torch.randint(len(obs), (min(self.cfg["batch_size"], len(obs)),), device=obs.device)]
            with torch.no_grad():
                target = self.ref(xb)
            loss = coef * nn.functional.mse_loss(actor_mean(self.actor, xb, self.activation), target)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        if self.it % 50 == 0:
            print(f"[bc_reg] iter {self.it}: loss {float(loss):.5f}  baseline-like states {len(obs)}  coef {coef:.3f}")