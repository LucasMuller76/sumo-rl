"""MAPPO agent: shared actor + centralized critic, PPO-clip update with value normalization."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .buffer import RolloutBuffer
from .config import MAPPOConfig
from .networks import CentralizedCritic, SharedActor, ValueNorm


class MAPPO:
    def __init__(self, obs_dim: int, n_actions: int, state_dim: int, cfg: MAPPOConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.actor = SharedActor(obs_dim, n_actions, cfg.hidden_dim, cfg.n_hidden, cfg.activation).to(device)
        self.critic = CentralizedCritic(state_dim, cfg.hidden_dim, cfg.n_hidden, cfg.activation).to(device)
        self.params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=cfg.lr, eps=1e-5)
        self.value_norm = ValueNorm() if cfg.value_norm else None

    # ------------------------------------------------------------------ rollout
    @torch.no_grad()
    def act(self, obs: np.ndarray, state: np.ndarray, deterministic: bool = False):
        """obs: ``[W, A, O]``  state: ``[W, A, S]``. Returns flat ``[W*A]`` arrays."""
        obs_t = torch.as_tensor(obs.reshape(-1, obs.shape[-1]), dtype=torch.float32, device=self.device)
        state_t = torch.as_tensor(state.reshape(-1, state.shape[-1]), dtype=torch.float32, device=self.device)
        dist = self.actor.dist(obs_t)
        actions = dist.probs.argmax(-1) if deterministic else dist.sample()
        logp = dist.log_prob(actions)
        values = self.critic(state_t)
        if self.value_norm is not None:
            values = self.value_norm.denormalize(values)
        return actions.cpu().numpy(), logp.cpu().numpy(), values.cpu().numpy()

    @torch.no_grad()
    def get_values(self, state: np.ndarray) -> np.ndarray:
        state_t = torch.as_tensor(state.reshape(-1, state.shape[-1]), dtype=torch.float32, device=self.device)
        v = self.critic(state_t)
        if self.value_norm is not None:
            v = self.value_norm.denormalize(v)
        return v.cpu().numpy().reshape(state.shape[:-1])

    @torch.no_grad()
    def act_deterministic(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs.reshape(-1, obs.shape[-1]), dtype=torch.float32, device=self.device)
        a = self.actor.forward(obs_t).argmax(-1)
        return a.cpu().numpy().reshape(obs.shape[:-1])

    # ------------------------------------------------------------------- update
    def update(self, buf: RolloutBuffer, lr_now: float) -> dict:
        cfg = self.cfg
        for grp in self.optimizer.param_groups:
            grp["lr"] = lr_now

        data, n = buf.flat()
        returns = data["returns"]
        if self.value_norm is not None:
            self.value_norm.update(returns)
            norm_returns = self.value_norm.normalize(returns)
            old_values_norm = self.value_norm.normalize(data["old_values"])
        else:
            norm_returns = returns
            old_values_norm = data["old_values"]

        mb_size = max(1, n // cfg.n_minibatches)
        idx = np.arange(n)
        clipfracs: list[float] = []
        kls: list[float] = []
        pg_loss = v_loss = entropy = torch.tensor(0.0)
        grad_norm = 0.0
        epochs_run = 0

        for epoch in range(cfg.n_epochs):
            np.random.shuffle(idx)
            epoch_kls: list[float] = []
            for start in range(0, n, mb_size):
                mb = torch.as_tensor(idx[start : start + mb_size], device=self.device)
                obs, state, act = data["obs"][mb], data["states"][mb], data["actions"][mb]
                old_logp = data["old_logprobs"][mb]
                adv = data["advantages"][mb]
                mb_ret = norm_returns[mb]
                mb_old_v = old_values_norm[mb]

                dist = self.actor.dist(obs)
                new_logp = dist.log_prob(act)
                entropy = dist.entropy().mean()
                logratio = new_logp - old_logp
                ratio = logratio.exp()

                if cfg.norm_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                pg1 = -adv * ratio
                pg2 = -adv * torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range)
                pg_loss = torch.max(pg1, pg2).mean()

                new_v = self.critic(state)
                if cfg.clip_vloss:
                    v_unclipped = (new_v - mb_ret) ** 2
                    v_clip = mb_old_v + torch.clamp(new_v - mb_old_v, -cfg.clip_range, cfg.clip_range)
                    v_loss = 0.5 * torch.max(v_unclipped, (v_clip - mb_ret) ** 2).mean()
                else:
                    v_loss = 0.5 * ((new_v - mb_ret) ** 2).mean()

                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = float(nn.utils.clip_grad_norm_(self.params, cfg.max_grad_norm))
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean().item()
                    epoch_kls.append(approx_kl)
                    kls.append(approx_kl)
                    clipfracs.append((torch.abs(ratio - 1.0) > cfg.clip_range).float().mean().item())

            epochs_run = epoch + 1
            if cfg.kl_target is not None and float(np.mean(epoch_kls)) > cfg.kl_target:
                break

        y_pred = data["old_values"].cpu().numpy()
        y_true = data["returns"].cpu().numpy()
        var_y = float(np.var(y_true))
        ev = float("nan") if var_y == 0 else float(1.0 - np.var(y_true - y_pred) / var_y)

        return {
            "policy_loss": float(pg_loss.item()),
            "value_loss": float(v_loss.item()),
            "entropy": float(entropy.item()),
            "approx_kl": float(np.mean(kls)) if kls else 0.0,
            "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
            "grad_norm": grad_norm,
            "explained_variance": ev,
            "epochs_run": epochs_run,
            "value_norm_mean": float(self.value_norm.rms.mean) if self.value_norm else 0.0,
            "value_norm_std": float(self.value_norm.rms.std) if self.value_norm else 1.0,
        }

    # --------------------------------------------------------------- persistence
    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "value_norm": self.value_norm.state_dict() if self.value_norm else None,
        }

    def load_state_dict(self, d: dict) -> None:
        self.actor.load_state_dict(d["actor"])
        self.critic.load_state_dict(d["critic"])
        if d.get("optimizer") is not None:
            self.optimizer.load_state_dict(d["optimizer"])
        if self.value_norm is not None and d.get("value_norm") is not None:
            self.value_norm.load_state_dict(d["value_norm"])
