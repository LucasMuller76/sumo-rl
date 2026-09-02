"""Construction of the centralized (agent-specific) global state for the critic.

For homogeneous agents (2x2, 4x4-Lucas) the global state for agent ``i`` is::

    s_i = concat_j(o_j)  ++  [optional normalized system_* scalars]  ++  onehot(i)

which is the "concat-local" (CL) state from Yu et al. 2021 plus the agent id that
lets a single shared critic emit a per-agent value. With ``critic="local"`` the
state degenerates to ``o_i`` (this is the IPPO ablation).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# Keys pulled from SumoEnvironment's info dict (already computed, no extra TraCI cost).
GLOBAL_SCALAR_KEYS = ("system_total_stopped", "system_total_waiting_time", "system_mean_speed")
N_GLOBAL_SCALARS = len(GLOBAL_SCALAR_KEYS)


def state_dim(obs_dim: int, n_agents: int, critic: str, use_global_scalars: bool) -> int:
    if critic == "local":
        return obs_dim
    d = n_agents * obs_dim + n_agents
    if use_global_scalars:
        d += N_GLOBAL_SCALARS
    return d


def build_state(
    obs: np.ndarray,
    g: Optional[np.ndarray],
    n_agents: int,
    critic: str,
    use_global_scalars: bool,
    g_mean: Optional[np.ndarray] = None,
    g_std: Optional[np.ndarray] = None,
) -> np.ndarray:
    """obs: ``[W, A, O]``; g: ``[W, N_GLOBAL_SCALARS]`` or None. Returns ``[W, A, state_dim]``."""
    obs = np.asarray(obs, np.float32)
    if critic == "local":
        return obs
    W, A, O = obs.shape
    joint = obs.reshape(W, A * O)
    joint = np.repeat(joint[:, None, :], A, axis=1)  # [W, A, A*O]
    parts = [joint]
    if use_global_scalars:
        gg = np.asarray(g, np.float32)
        if g_mean is not None:
            gg = (gg - np.asarray(g_mean, np.float32)) / np.asarray(g_std, np.float32)
        parts.append(np.repeat(gg[:, None, :], A, axis=1))  # [W, A, G]
    eye = np.broadcast_to(np.eye(A, dtype=np.float32), (W, A, A))
    parts.append(eye)
    return np.concatenate(parts, axis=-1).astype(np.float32)
