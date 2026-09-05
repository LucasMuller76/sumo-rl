"""n-step aggregation and (uniform / prioritized) replay over JOINT transitions.

A *joint* item bundles all A agents of one worker at one timestep, plus the QMIX team
state when a mixer is used. IDQN flattens ``[C, A, ...] -> [C*A, ...]`` at sample time;
VDN/QMIX consume the joint item whole. One storage path, switchable by ``--mixer``.

Episodes end only by time limit (never absorbing), so every stored transition carries
``done_flag = 0`` and bootstraps off its ``next_obs`` -- which at an episode boundary is
the *terminal* observation handed back by the vec-env, never the next episode's first obs.
"""

from __future__ import annotations

import collections
from typing import List, Optional

import numpy as np


class NStepAccumulator:
    """Per-worker n-step aggregation over joint env steps.

    ``add`` one joint step per worker per call; it returns a list of joint n-step items
    ready for the buffer. On a boundary (``done``) the worker's whole deque is drained as
    truncated ``m < n`` items (each bootstrapping off the terminal obs) and then cleared,
    so no item ever spans two episodes.
    """

    def __init__(self, n_workers: int, n_step: int, gamma: float):
        self.n = int(n_step)
        self.gamma = float(gamma)
        self._dq: List[collections.deque] = [collections.deque() for _ in range(n_workers)]

    def _emit_front(self, dq: collections.deque) -> dict:
        R = np.zeros_like(dq[0]["reward"])
        disc = 1.0
        m = 0
        boot_next = boot_ng = None
        for k in range(len(dq)):
            e = dq[k]
            R = R + disc * e["reward"]
            disc *= self.gamma
            m += 1
            boot_next, boot_ng = e["next_obs"], e["next_gstate"]
            if e["done"] or m == self.n:
                break
        head = dq[0]
        dq.popleft()
        return {
            "obs": head["obs"],
            "action": head["action"],
            "R": R.astype(np.float32),
            "next_obs": boot_next,
            "gamma_m": np.float32(self.gamma ** m),
            "gstate": head["gstate"],
            "next_gstate": boot_ng,
        }

    def add(self, w: int, obs, action, reward, next_obs, done: bool, gstate=None, next_gstate=None) -> List[dict]:
        dq = self._dq[w]
        dq.append(
            {"obs": obs, "action": action, "reward": np.asarray(reward, np.float32),
             "next_obs": next_obs, "done": bool(done), "gstate": gstate, "next_gstate": next_gstate}
        )
        out: List[dict] = []
        if bool(done):
            while dq:
                out.append(self._emit_front(dq))
        elif len(dq) >= self.n:
            out.append(self._emit_front(dq))
        return out


class _SumTree:
    """Array sum-tree; leaves hold p^alpha, internal nodes hold subtree sums."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity, dtype=np.float64)

    def update(self, leaf: int, value: float) -> None:
        i = leaf + self.capacity
        self.tree[i] = value
        i //= 2
        while i >= 1:
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]
            i //= 2

    def total(self) -> float:
        return float(self.tree[1])

    def retrieve(self, prefix: float) -> int:
        i = 1
        while i < self.capacity:
            left = 2 * i
            if prefix <= self.tree[left]:
                i = left
            else:
                prefix -= self.tree[left]
                i = left + 1
        return i - self.capacity


class JointReplay:
    """Uniform FIFO replay over joint items."""

    prioritized = False

    def __init__(self, capacity: int, n_agents: int, obs_dim: int, state_dim: int, rng: np.random.Generator):
        self.capacity = int(capacity)
        self.A = n_agents
        self.S = int(state_dim)
        self.rng = rng
        self.size = 0
        self.ptr = 0
        self.obs = np.zeros((capacity, n_agents, obs_dim), np.float32)
        self.action = np.zeros((capacity, n_agents), np.int64)
        self.R = np.zeros((capacity, n_agents), np.float32)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim), np.float32)
        self.gamma_m = np.zeros((capacity,), np.float32)
        if self.S:
            self.gstate = np.zeros((capacity, self.S), np.float32)
            self.next_gstate = np.zeros((capacity, self.S), np.float32)

    def __len__(self) -> int:
        return self.size

    def _write(self, i: int, it: dict) -> None:
        self.obs[i] = it["obs"]
        self.action[i] = it["action"]
        self.R[i] = it["R"]
        self.next_obs[i] = it["next_obs"]
        self.gamma_m[i] = it["gamma_m"]
        if self.S:
            self.gstate[i] = it["gstate"]
            self.next_gstate[i] = it["next_gstate"]

    def add_batch(self, items: List[dict]) -> None:
        for it in items:
            self._write(self.ptr, it)
            self._on_insert(self.ptr)
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def _on_insert(self, i: int) -> None:  # hook for PER
        pass

    def _gather(self, idx: np.ndarray, weights: np.ndarray) -> dict:
        out = {
            "obs": self.obs[idx],
            "action": self.action[idx],
            "R": self.R[idx],
            "next_obs": self.next_obs[idx],
            "gamma_m": self.gamma_m[idx],
            "idx": idx,
            "weights": weights.astype(np.float32),
        }
        if self.S:
            out["gstate"] = self.gstate[idx]
            out["next_gstate"] = self.next_gstate[idx]
        return out

    def sample(self, batch_size: int, beta: float = 1.0) -> dict:
        idx = self.rng.integers(0, self.size, size=batch_size)
        return self._gather(idx, np.ones(batch_size, np.float32))

    def update_priorities(self, idx: np.ndarray, td_abs: np.ndarray) -> None:
        pass


class PrioritizedJointReplay(JointReplay):
    """Proportional PER (sum-tree, stratified sampling, IS weights)."""

    prioritized = True

    def __init__(self, capacity, n_agents, obs_dim, state_dim, rng, alpha=0.6, eps=1e-6):
        super().__init__(capacity, n_agents, obs_dim, state_dim, rng)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.tree = _SumTree(capacity)
        self.max_p = 1.0  # max raw priority (|td| + eps) seen; new items get this

    def _on_insert(self, i: int) -> None:
        self.tree.update(i, self.max_p ** self.alpha)

    def sample(self, batch_size: int, beta: float = 1.0) -> dict:
        total = self.tree.total()
        seg = total / batch_size
        idx = np.empty(batch_size, np.int64)
        for j in range(batch_size):
            mass = self.rng.uniform(seg * j, seg * (j + 1))
            idx[j] = self.tree.retrieve(mass)
        p = self.tree.tree[idx + self.capacity] / max(total, 1e-12)  # P(i)
        w = (self.size * np.maximum(p, 1e-12)) ** (-beta)
        w /= w.max()
        return self._gather(idx, w)

    def update_priorities(self, idx: np.ndarray, td_abs: np.ndarray) -> None:
        prio = np.abs(td_abs) + self.eps
        self.max_p = max(self.max_p, float(prio.max()))
        pa = prio ** self.alpha
        for i, v in zip(idx, pa):
            self.tree.update(int(i), float(v))


def make_replay(cfg, scn, obs_dim: int, state_dim: int, rng: np.random.Generator) -> JointReplay:
    cap = cfg.resolved_replay_capacity(scn)
    if cfg.per:
        return PrioritizedJointReplay(cap, scn.n_agents, obs_dim, state_dim, rng, cfg.per_alpha, cfg.per_eps)
    return JointReplay(cap, scn.n_agents, obs_dim, state_dim, rng)
