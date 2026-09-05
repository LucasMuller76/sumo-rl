"""Scenario presets and DQN hyper-parameter configuration.

``ScenarioConfig``, ``SCENARIOS``, ``EVAL_SEED_BASE`` and the training-seed formula are
byte-identical to ``experiments/mappo/config.py`` so DQN and MAPPO evaluate under the
same conditions (same nets, episode length, held-out seeds, Fixed-Time baseline).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import List, Optional

import sumo_rl


_NETS = os.path.join(os.path.dirname(sumo_rl.__file__), "nets")

# Held-out evaluation seeds are EVAL_SEED_BASE + 1 .. EVAL_SEED_BASE + n_eval_seeds.
# Kept far away from the training seed range (seed * 10000 + worker * 100 + episode).
EVAL_SEED_BASE = 9_000_000


@dataclass
class ScenarioConfig:
    """Static description of a SUMO traffic-signal scenario (identical to MAPPO's)."""

    name: str
    net_file: str
    route_file: str
    num_seconds: int
    n_agents: int
    n_actions: int
    delta_time: int = 5
    yellow_time: int = 2
    min_green: int = 5
    begin_time: int = 0
    default_total_steps: int = 2_000_000  # agent-steps (summed over agents)


SCENARIOS = {
    "2x2": ScenarioConfig(
        name="2x2",
        net_file=os.path.join(_NETS, "2x2grid", "2x2.net.xml"),
        route_file=os.path.join(_NETS, "2x2grid", "2x2.rou.xml"),
        num_seconds=20_000,
        n_agents=4,
        n_actions=4,
        default_total_steps=2_000_000,
    ),
    "4x4": ScenarioConfig(
        name="4x4",
        net_file=os.path.join(_NETS, "4x4-Lucas", "4x4.net.xml"),
        route_file=os.path.join(_NETS, "4x4-Lucas", "4x4c1c2c1c2.rou.xml"),
        num_seconds=80_000,
        n_agents=16,
        n_actions=2,
        default_total_steps=6_000_000,
    ),
}


@dataclass
class DQNConfig:
    """All knobs for a DQN run. CLI flags in ``train_dqn.py`` map 1:1 onto these."""

    scenario: str = "2x2"
    seed: int = 0
    mixer: str = "none"  # "none" (IDQN+PS) | "vdn" | "qmix"
    per: bool = True  # prioritized replay (proportional, sum-tree)
    use_global_scalars: bool = False  # append normalized system_* scalars to the QMIX team state
    reward_fn: str = "diff-waiting-time"
    num_seconds: Optional[int] = None  # None -> scenario default
    total_steps: Optional[int] = None  # None -> scenario default

    # rollout / parallelism
    n_workers: int = 8
    collect_steps: int = 8  # env steps collected per worker per iteration
    sumo_backend: str = "auto"  # "auto" | "libsumo" | "traci"

    # replay
    replay_capacity: Optional[int] = None  # joint items (worker-steps); None -> 100k (2x2) / 150k (4x4)
    batch_size: int = 256
    replay_ratio: float = 4.0  # gradient updates per collected agent-transition
    learning_starts: Optional[int] = None  # random-action warmup, agent-steps; None -> 50k / 100k
    n_step: int = 3
    per_alpha: float = 0.6
    per_beta0: float = 0.4
    per_eps: float = 1e-6

    # optimisation
    lr: Optional[float] = None  # None -> 1e-4 if per else 2.5e-4
    anneal_lr: bool = False
    adam_eps: float = 1.5e-4
    gamma: float = 0.99
    max_grad_norm: float = 10.0
    huber_delta: float = 1.0
    target_tau: float = 0.005  # Polyak; ignored if hard_target > 0
    hard_target: int = 0  # if > 0: hard copy every N gradient updates instead of Polyak

    # exploration (Ape-X per-worker epsilon ladder, no global anneal)
    eps_base: float = 0.4
    eps_alpha: float = 7.0
    eps_min: float = 0.01

    # network
    hidden_dim: int = 128
    n_hidden: int = 2
    activation: str = "relu"
    dueling: bool = True
    double: bool = True
    mixer_hidden: int = 32
    hypernet_hidden: int = 64

    # eval / logging / io  (cadence is in AGENT-STEPS, not iterations)
    eval_every_steps: int = 120_000
    save_every_steps: int = 240_000
    n_eval_seeds: int = 5
    n_eval_workers: int = 5
    device: str = "cpu"
    torch_threads: int = 2
    run_name: Optional[str] = None
    out_root: str = "outputs/dqn"

    # ---- resolved getters -------------------------------------------------
    def resolved_num_seconds(self, scn: ScenarioConfig) -> int:
        return int(self.num_seconds if self.num_seconds is not None else scn.num_seconds)

    def resolved_total_steps(self, scn: ScenarioConfig) -> int:
        return int(self.total_steps if self.total_steps is not None else scn.default_total_steps)

    def resolved_lr(self) -> float:
        if self.lr is not None:
            return float(self.lr)
        return 1e-4 if self.per else 2.5e-4

    def resolved_replay_capacity(self, scn: ScenarioConfig) -> int:
        if self.replay_capacity is not None:
            return int(self.replay_capacity)
        return 100_000 if scn.n_agents <= 4 else 150_000

    def resolved_learning_starts(self, scn: ScenarioConfig) -> int:
        if self.learning_starts is not None:
            return int(self.learning_starts)
        return 50_000 if scn.n_agents <= 4 else 100_000

    def eval_seeds(self) -> List[int]:
        return [EVAL_SEED_BASE + i for i in range(1, self.n_eval_seeds + 1)]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DQNConfig":
        import dataclasses

        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})
