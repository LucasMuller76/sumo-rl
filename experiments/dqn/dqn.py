"""Shared-parameter multi-agent DQN agent (IDQN / VDN / QMIX).

Execution is always decentralised: each agent picks ``argmax_a Q(o_i, a)`` from the shared
Q-net. The ``mixer`` only changes the training target:
- ``none``: per-agent target ``y_i = R_i + gamma^m * Q_target(o'_i, a*_i)``  (IDQN + PS)
- ``vdn`` : ``Q_tot = sum_i Q_i``, team target on ``mean_i R_i``
- ``qmix``: monotonic mix of ``Q_i`` conditioned on the global team state
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import QMixer, QNet, vdn_mix


def epsilon_ladder(n_workers: int, eps_base: float, eps_alpha: float, eps_min: float) -> np.ndarray:
    """Ape-X per-worker epsilon: ``eps_i = eps_base ** (1 + i/(N-1) * alpha)``."""
    if n_workers == 1:
        return np.array([max(eps_min, eps_base)], np.float64)
    i = np.arange(n_workers, dtype=np.float64)
    eps = eps_base ** (1.0 + i / (n_workers - 1) * eps_alpha)
    return np.maximum(eps, eps_min)


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, n_agents: int, state_dim: int, cfg, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.mixer_kind = cfg.mixer

        def _q():
            return QNet(obs_dim, n_actions, cfg.hidden_dim, cfg.n_hidden, cfg.activation, cfg.dueling).to(device)

        self.q_net = _q()
        self.target_net = _q()
        self.target_net.load_state_dict(self.q_net.state_dict())
        params = list(self.q_net.parameters())

        self.mixer = self.target_mixer = None
        if cfg.mixer == "qmix":
            self.mixer = QMixer(n_agents, state_dim, cfg.mixer_hidden, cfg.hypernet_hidden).to(device)
            self.target_mixer = QMixer(n_agents, state_dim, cfg.mixer_hidden, cfg.hypernet_hidden).to(device)
            self.target_mixer.load_state_dict(self.mixer.state_dict())
            params += list(self.mixer.parameters())

        self.optimizer = torch.optim.Adam(params, lr=cfg.resolved_lr(), eps=cfg.adam_eps)
        self.updates_done = 0

    # ------------------------------------------------------------------ acting
    @torch.no_grad()
    def q_values(self, obs: np.ndarray) -> torch.Tensor:
        return self.q_net(torch.as_tensor(obs, dtype=torch.float32, device=self.device))

    @torch.no_grad()
    def act(self, obs: np.ndarray, eps_per_worker: np.ndarray, rng: np.random.Generator):
        """obs ``[W, A, O]`` -> (actions ``[W, A]`` int64, mean greedy Q for logging)."""
        q = self.q_net(torch.as_tensor(obs, dtype=torch.float32, device=self.device))  # [W, A, n_act]
        greedy = q.argmax(-1).cpu().numpy()
        rand = rng.integers(0, self.n_actions, size=greedy.shape)
        explore = rng.random(greedy.shape) < np.asarray(eps_per_worker)[:, None]
        actions = np.where(explore, rand, greedy).astype(np.int64)
        return actions, float(q.mean().item())

    @torch.no_grad()
    def act_deterministic(self, obs: np.ndarray) -> np.ndarray:
        """obs ``[W, A, O]`` -> ``[W, A]`` int64 (pure argmax; used by eval.run_episodes)."""
        q = self.q_net(torch.as_tensor(obs, dtype=torch.float32, device=self.device))
        return q.argmax(-1).cpu().numpy().reshape(obs.shape[:-1]).astype(np.int64)

    # ------------------------------------------------------------------ learning
    def update(self, batch: dict, beta: float = 1.0) -> dict:
        dev = self.device
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=dev)          # [B,A,O]
        action = torch.as_tensor(batch["action"], dtype=torch.int64, device=dev)      # [B,A]
        R = torch.as_tensor(batch["R"], dtype=torch.float32, device=dev)              # [B,A]
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=dev)  # [B,A,O]
        gamma_m = torch.as_tensor(batch["gamma_m"], dtype=torch.float32, device=dev)  # [B]
        weights = torch.as_tensor(batch["weights"], dtype=torch.float32, device=dev)  # [B]

        q_a = self.q_net(obs).gather(-1, action.unsqueeze(-1)).squeeze(-1)            # [B,A]

        with torch.no_grad():
            if self.cfg.double:
                next_act = self.q_net(next_obs).argmax(-1)                            # online
            else:
                next_act = self.target_net(next_obs).argmax(-1)
            next_q = self.target_net(next_obs).gather(-1, next_act.unsqueeze(-1)).squeeze(-1)  # [B,A]

        if self.mixer_kind == "none":
            y = R + gamma_m.unsqueeze(-1) * next_q                                    # [B,A], done_flag=0 always
            td = q_a - y                                                             # [B,A]
            loss = (self._huber(td).mean(dim=1) * weights).mean()
            td_abs = td.detach().abs().amax(dim=1)                                    # [B] priority = max_i|delta_i|
        else:
            if self.mixer_kind == "vdn":
                q_tot = vdn_mix(q_a)
                next_q_tot = vdn_mix(next_q)
            else:  # qmix
                gstate = torch.as_tensor(batch["gstate"], dtype=torch.float32, device=dev)
                next_gstate = torch.as_tensor(batch["next_gstate"], dtype=torch.float32, device=dev)
                q_tot = self.mixer(q_a, gstate)
                with torch.no_grad():
                    next_q_tot = self.target_mixer(next_q, next_gstate)
            r_team = R.mean(dim=1)                                                    # [B] mean over agents
            y = r_team + gamma_m * next_q_tot
            td = q_tot - y                                                           # [B]
            loss = (self._huber(td) * weights).mean()
            td_abs = td.detach().abs()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        params = list(self.q_net.parameters()) + (list(self.mixer.parameters()) if self.mixer else [])
        grad_norm = float(nn.utils.clip_grad_norm_(params, self.cfg.max_grad_norm))
        self.optimizer.step()
        self.updates_done += 1
        self._sync_target()

        return {
            "loss": float(loss.item()),
            "q_mean": float(q_a.mean().item()),
            "q_target_mean": float(next_q.mean().item()),
            "td_error_abs_mean": float(td_abs.mean().item()),
            "grad_norm": grad_norm,
            "td_abs": td_abs.cpu().numpy(),  # for PER priority update
        }

    def _huber(self, td: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(td, torch.zeros_like(td), reduction="none", delta=self.cfg.huber_delta)

    def _sync_target(self) -> None:
        if self.cfg.hard_target > 0:
            if self.updates_done % self.cfg.hard_target == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())
                if self.mixer is not None:
                    self.target_mixer.load_state_dict(self.mixer.state_dict())
            return
        tau = self.cfg.target_tau
        with torch.no_grad():
            for tp, p in zip(self.target_net.parameters(), self.q_net.parameters()):
                tp.mul_(1 - tau).add_(tau * p)
            if self.mixer is not None:
                for tp, p in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                    tp.mul_(1 - tau).add_(tau * p)

    # --------------------------------------------------------------- persistence
    def state_dict(self) -> dict:
        d = {
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "updates_done": self.updates_done,
        }
        if self.mixer is not None:
            d["mixer"] = self.mixer.state_dict()
            d["target_mixer"] = self.target_mixer.state_dict()
        return d

    def load_state_dict(self, d: dict) -> None:
        self.q_net.load_state_dict(d["q_net"])
        self.target_net.load_state_dict(d["target_net"])
        if d.get("optimizer") is not None:
            self.optimizer.load_state_dict(d["optimizer"])
        self.updates_done = int(d.get("updates_done", 0))
        if self.mixer is not None and d.get("mixer") is not None:
            self.mixer.load_state_dict(d["mixer"])
            self.target_mixer.load_state_dict(d["target_mixer"])
