"""Shared dueling Q-network and the VDN / QMIX value-mixing modules.

The ``_mlp`` / ``orthogonal_init`` idiom matches ``experiments/mappo/networks.py``.
Execution is identical for every mixer: each agent acts on ``argmax_a Q(o_i, a)`` from
the shared Q-net; VDN/QMIX only change the training target (``Q_tot``).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


_ACT = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}


def orthogonal_init(layer: nn.Linear, gain: float = np.sqrt(2)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


def _trunk(in_dim: int, hidden_dim: int, n_hidden: int, activation: str) -> nn.Sequential:
    act = _ACT[activation]
    layers: list[nn.Module] = []
    last = in_dim
    for _ in range(n_hidden):
        layers += [orthogonal_init(nn.Linear(last, hidden_dim)), act()]
        last = hidden_dim
    return nn.Sequential(*layers)


class QNet(nn.Module):
    """Shared Q(o_i, a_i; theta). Accepts any leading shape ``[..., obs_dim]``.

    Dueling: ``Q = V(s) + (A(s,a) - mean_a A(s,a))``.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 128,
        n_hidden: int = 2,
        activation: str = "relu",
        dueling: bool = True,
    ):
        super().__init__()
        self.dueling = dueling
        self.trunk = _trunk(obs_dim, hidden_dim, n_hidden, activation)
        if dueling:
            self.value = orthogonal_init(nn.Linear(hidden_dim, 1), gain=1.0)
            self.adv = orthogonal_init(nn.Linear(hidden_dim, n_actions), gain=1.0)
        else:
            self.head = orthogonal_init(nn.Linear(hidden_dim, n_actions), gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.trunk(obs)
        if not self.dueling:
            return self.head(h)
        v = self.value(h)
        a = self.adv(h)
        return v + a - a.mean(dim=-1, keepdim=True)


class QMixer(nn.Module):
    """Monotonic mixing network (Rashid et al. 2018).

    ``forward(agent_qs[B, A], state[B, S]) -> q_tot[B]``. Non-negative mixing weights
    (via ``abs``) guarantee d q_tot / d q_i >= 0, so a per-agent argmax also maximises
    ``q_tot`` (decentralised execution stays correct).
    """

    def __init__(self, n_agents: int, state_dim: int, mixer_hidden: int = 32, hypernet_hidden: int = 64):
        super().__init__()
        self.n_agents = n_agents
        self.mixer_hidden = mixer_hidden

        self.hyper_w1 = nn.Sequential(
            orthogonal_init(nn.Linear(state_dim, hypernet_hidden)),
            nn.ReLU(),
            orthogonal_init(nn.Linear(hypernet_hidden, n_agents * mixer_hidden), gain=1.0),
        )
        self.hyper_w2 = nn.Sequential(
            orthogonal_init(nn.Linear(state_dim, hypernet_hidden)),
            nn.ReLU(),
            orthogonal_init(nn.Linear(hypernet_hidden, mixer_hidden), gain=1.0),
        )
        self.hyper_b1 = orthogonal_init(nn.Linear(state_dim, mixer_hidden), gain=1.0)
        self.hyper_b2 = nn.Sequential(
            orthogonal_init(nn.Linear(state_dim, hypernet_hidden)),
            nn.ReLU(),
            orthogonal_init(nn.Linear(hypernet_hidden, 1), gain=1.0),
        )

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        b = agent_qs.shape[0]
        qs = agent_qs.view(b, 1, self.n_agents)
        w1 = torch.abs(self.hyper_w1(state)).view(b, self.n_agents, self.mixer_hidden)
        b1 = self.hyper_b1(state).view(b, 1, self.mixer_hidden)
        hidden = F.elu(torch.bmm(qs, w1) + b1)  # [b, 1, mixer_hidden]
        w2 = torch.abs(self.hyper_w2(state)).view(b, self.mixer_hidden, 1)
        b2 = self.hyper_b2(state).view(b, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2  # [b, 1, 1]
        return q_tot.view(b)


def vdn_mix(agent_qs: torch.Tensor) -> torch.Tensor:
    """VDN: ``Q_tot = sum_i Q_i``. ``agent_qs[B, A] -> [B]``."""
    return agent_qs.sum(dim=-1)
