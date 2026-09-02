"""Actor / centralized-critic networks and running normalizers for MAPPO."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


_ACT = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}


def orthogonal_init(layer: nn.Linear, gain: float = np.sqrt(2)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


def _mlp(in_dim: int, hidden_dim: int, n_hidden: int, out_dim: int, activation: str, out_gain: float) -> nn.Sequential:
    act = _ACT[activation]
    layers: list[nn.Module] = []
    last = in_dim
    for _ in range(n_hidden):
        layers += [orthogonal_init(nn.Linear(last, hidden_dim)), act()]
        last = hidden_dim
    layers.append(orthogonal_init(nn.Linear(last, out_dim), gain=out_gain))
    return nn.Sequential(*layers)


class SharedActor(nn.Module):
    """Policy pi(a_i | o_i), parameters shared across all (homogeneous) agents."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128, n_hidden: int = 2, activation: str = "tanh"):
        super().__init__()
        self.net = _mlp(obs_dim, hidden_dim, n_hidden, n_actions, activation, out_gain=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def dist(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(obs))


class CentralizedCritic(nn.Module):
    """Value function V(s_i). ``s_i`` is the (agent-specific) global state built in state.py."""

    def __init__(self, state_dim: int, hidden_dim: int = 128, n_hidden: int = 2, activation: str = "tanh"):
        super().__init__()
        self.net = _mlp(state_dim, hidden_dim, n_hidden, 1, activation, out_gain=1.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class RunningMeanStd:
    """Parallel (Chan et al.) running mean/variance estimator. Shape () for scalars."""

    def __init__(self, shape=(), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = float(epsilon)

    def update(self, x) -> None:
        x = np.asarray(x, np.float64)
        x = x.reshape(-1) if self.mean.ndim == 0 else x.reshape(-1, *self.mean.shape)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        if batch_count == 0:
            return
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot
        self.var = m2 / tot
        self.count = tot

    @property
    def std(self):
        return np.sqrt(self.var) + 1e-8

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean = np.asarray(d["mean"], np.float64)
        self.var = np.asarray(d["var"], np.float64)
        self.count = float(d["count"])


class ValueNorm:
    """Normalize value targets with running return statistics (PopArt-lite, no weight rescale).

    The critic is trained to predict normalized returns; predictions are denormalized
    before they are used as bootstrap values / advantages.
    """

    def __init__(self):
        self.rms = RunningMeanStd(shape=())

    def update(self, returns: torch.Tensor) -> None:
        self.rms.update(returns.detach().cpu().numpy().reshape(-1))

    def _ms(self, ref: torch.Tensor):
        m = torch.as_tensor(self.rms.mean, dtype=torch.float32, device=ref.device)
        s = torch.as_tensor(self.rms.std, dtype=torch.float32, device=ref.device)
        return m, s

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        m, s = self._ms(x)
        return (x - m) / s

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        m, s = self._ms(x)
        return x * s + m

    def state_dict(self) -> dict:
        return self.rms.state_dict()

    def load_state_dict(self, d: dict) -> None:
        self.rms.load_state_dict(d)
