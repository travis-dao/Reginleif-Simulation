"""BC network (same architecture as the rsl_rl actor MLP) and weight transfer into the PPO actor."""
from __future__ import annotations

import torch
import torch.nn as nn

ACTIVATIONS = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "selu": nn.SELU, "lrelu": nn.LeakyReLU}


def make_mlp(in_dim: int, hidden_dims, out_dim: int, activation: str) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden_dims:
        layers += [nn.Linear(d, h), ACTIVATIONS[activation]()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def load_bc_into_actor(actor: nn.Module, ckpt_path: str, device="cpu") -> None:
    """Copy the BC Linear layers (in order) into the rsl_rl actor's Linear layers.
    Matched by order + shape instead of parameter names, so it survives rsl_rl renames; fails loudly on mismatch."""
    blob = torch.load(ckpt_path, map_location=device)
    src = [(k, v) for k, v in blob["bc_state_dict"].items()]
    src_w = [v for k, v in src if k.endswith("weight")]
    src_b = [v for k, v in src if k.endswith("bias")]
    dst = [m for m in actor.modules() if isinstance(m, nn.Linear)]
    if len(dst) != len(src_w):
        raise ValueError(f"BC net has {len(src_w)} Linear layers, rsl_rl actor has {len(dst)}. "
                         "Check actor.hidden_dims in model.yaml matches the BC checkpoint.")
    with torch.no_grad():
        for m, w, b in zip(dst, src_w, src_b):
            if m.weight.shape != w.shape:
                raise ValueError(f"shape mismatch {tuple(m.weight.shape)} vs {tuple(w.shape)} "
                                 "(obs size / hidden dims differ from the BC checkpoint)")
            m.weight.copy_(w.to(m.weight.device)); m.bias.copy_(b.to(m.bias.device))
    print(f"loaded BC weights from {ckpt_path} into {len(dst)} actor layers")