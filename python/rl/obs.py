"""ONE observation definition shared by the data generator (numpy) and the Playground env (jax.numpy).
Pass `xp=np` or `xp=jnp`. Everything broadcasts over leading batch dims."""
from __future__ import annotations

from rl.config import ObsSpec


def build_obs(xp, spec: ObsSpec, gyro, gravity_b, cmd, q, qd, q_default, action_hist):
    """gyro (…,3) rad/s body frame | gravity_b (…,3) = R.T @ [0,0,-1] | cmd (…,3) = [v_x m/s, v_y m/s, yaw rad/s]
    q, qd (…,12) | action_hist (…,H,12) ordered oldest -> newest (actions up to t-1)."""
    lead = action_hist.shape[:-2]
    cmd_scale = xp.asarray([spec.cmd_scale_lin, spec.cmd_scale_lin, spec.cmd_scale_ang], dtype=xp.float32)
    return xp.concatenate([
        gyro * spec.gyro_scale,
        gravity_b,
        cmd * cmd_scale,
        q - q_default,
        qd * spec.qd_scale,
        action_hist.reshape(lead + (-1,)),
    ], axis=-1).astype(xp.float32)


def roll_history(xp, hist, act):
    """Drop the oldest action, append `act` as the newest."""
    if hist.shape[-2] == 0:
        return hist
    return xp.concatenate([hist[..., 1:, :], act[..., None, :]], axis=-2)


def near_baseline_cmds(cmd, bank, tol):
    """cmd (N,3), bank (K,3), tol (3,) -> bool (N,): True where the command is within `tol` (per component) of ANY bank
    command. Works with numpy arrays and torch tensors."""
    return (abs(cmd[:, None, :] - bank[None]) <= tol).all(-1).any(-1)