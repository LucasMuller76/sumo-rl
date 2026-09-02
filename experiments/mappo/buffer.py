"""On-policy rollout buffer with GAE and time-limit (truncation) bootstrapping.

SumoEnvironment never emits a true terminal state: every episode ends by time
limit (``sim_step >= sim_max_time``). So at every episode boundary we bootstrap
with ``V(s_T)`` of the final state and cut the GAE trace there.
"""

from __future__ import annotations

import numpy as np
import torch


class RolloutBuffer:
    def __init__(self, n_steps: int, n_workers: int, n_agents: int, obs_dim: int, state_dim: int, device):
        T, W, A = n_steps, n_workers, n_agents
        self.T, self.W, self.A = T, W, A
        self.device = device
        self.obs = np.zeros((T, W, A, obs_dim), np.float32)
        self.states = np.zeros((T, W, A, state_dim), np.float32)
        self.actions = np.zeros((T, W, A), np.int64)
        self.logprobs = np.zeros((T, W, A), np.float32)
        self.values = np.zeros((T, W, A), np.float32)  # denormalized (real scale)
        self.rewards = np.zeros((T, W, A), np.float32)
        self.dones = np.zeros((T, W), np.float32)  # 1 if the env auto-reset after step t
        self.term_values = np.zeros((T, W, A), np.float32)  # V(s_T) bootstrap at boundary steps
        self.advantages = np.zeros((T, W, A), np.float32)
        self.returns = np.zeros((T, W, A), np.float32)
        self.ptr = 0

    def reset(self) -> None:
        self.ptr = 0

    def add(self, obs, states, actions, logprobs, values, rewards, dones) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.states[i] = states
        self.actions[i] = actions
        self.logprobs[i] = logprobs
        self.values[i] = values
        self.rewards[i] = rewards
        self.dones[i] = dones
        self.term_values[i] = 0.0
        self.ptr += 1

    def set_term_values(self, t: int, term_values: np.ndarray) -> None:
        self.term_values[t] = term_values

    def compute_gae(self, last_values: np.ndarray, gamma: float, gae_lambda: float) -> None:
        """last_values: ``[W, A]`` bootstrap for step ``T-1`` when that step is not a boundary."""
        T = self.T
        adv = np.zeros_like(self.advantages)
        lastgaelam = np.zeros((self.W, self.A), np.float32)
        for t in reversed(range(T)):
            done = self.dones[t][:, None]  # [W, 1]
            next_values = last_values if t == T - 1 else self.values[t + 1]
            # at a boundary: bootstrap with the terminal-state value and stop the trace
            next_values = np.where(done > 0, self.term_values[t], next_values)
            nonterminal = 1.0 - done
            delta = self.rewards[t] + gamma * next_values - self.values[t]
            lastgaelam = delta + gamma * gae_lambda * nonterminal * lastgaelam
            adv[t] = lastgaelam
        self.advantages = adv
        self.returns = adv + self.values

    def flat(self) -> tuple[dict, int]:
        n = self.T * self.W * self.A
        to = lambda a, dt: torch.as_tensor(np.ascontiguousarray(a).reshape(n, -1) if a.ndim == 4 else
                                           np.ascontiguousarray(a).reshape(n), dtype=dt, device=self.device)
        data = {
            "obs": to(self.obs, torch.float32),
            "states": to(self.states, torch.float32),
            "actions": to(self.actions, torch.int64),
            "old_logprobs": to(self.logprobs, torch.float32),
            "old_values": to(self.values, torch.float32),
            "advantages": to(self.advantages, torch.float32),
            "returns": to(self.returns, torch.float32),
        }
        return data, n
