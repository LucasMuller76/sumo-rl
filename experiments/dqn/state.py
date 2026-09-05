"""Global-state construction.

Copied from ``experiments/mappo/state.py`` for ``GLOBAL_SCALAR_KEYS`` (needed by the
shared ``vec_sumo``) and extended with :func:`global_state` / :func:`global_state_dim`,
the team-level state the QMIX mixer conditions on: the concatenation of every agent's
local observation (+ optional normalized ``system_*`` scalars), ONE vector per timestep
(no per-agent one-hot -- the mixer sees the whole team at once).

``build_state`` (the MAPPO agent-specific state, ``s_i = concat_j(o_j) ++ onehot(i)``) is
kept unused for reference / parity.
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


def global_state_dim(obs_dim: int, n_agents: int, use_global_scalars: bool) -> int:
    """Width of the QMIX team state: concat of all local obs (+ optional system_* scalars)."""
    return n_agents * obs_dim + (N_GLOBAL_SCALARS if use_global_scalars else 0)


def global_state(
    obs: np.ndarray,
    g: Optional[np.ndarray] = None,
    use_global_scalars: bool = False,
    g_mean: Optional[np.ndarray] = None,
    g_std: Optional[np.ndarray] = None,
) -> np.ndarray:
    """obs: ``[..., A, O]`` -> team state ``[..., A*O (+ G)]`` (one vector per timestep)."""
    obs = np.asarray(obs, np.float32)
    lead = obs.shape[:-2]
    joint = obs.reshape(*lead, -1)  # [..., A*O]
    if not use_global_scalars or g is None:
        return joint
    gg = np.asarray(g, np.float32)
    if g_mean is not None:
        gg = (gg - np.asarray(g_mean, np.float32)) / np.asarray(g_std, np.float32)
    return np.concatenate([joint, gg], axis=-1).astype(np.float32)
