"""Scenario presets and MAPPO hyper-parameter configuration."""

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
    """Static description of a SUMO traffic-signal scenario."""

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
    default_total_steps: int = 3_000_000  # agent-steps (summed over agents)


SCENARIOS = {
    "2x2": ScenarioConfig(
        name="2x2",
        net_file=os.path.join(_NETS, "2x2grid", "2x2.net.xml"),
        route_file=os.path.join(_NETS, "2x2grid", "2x2.rou.xml"),
        num_seconds=20_000,
        n_agents=4,
        n_actions=4,
        default_total_steps=3_000_000,
    ),
    "4x4": ScenarioConfig(
        name="4x4",
        net_file=os.path.join(_NETS, "4x4-Lucas", "4x4.net.xml"),
        route_file=os.path.join(_NETS, "4x4-Lucas", "4x4c1c2c1c2.rou.xml"),
        num_seconds=80_000,
        n_agents=16,
        n_actions=2,
        default_total_steps=10_000_000,
    ),
}


@dataclass
class MAPPOConfig:
    """All knobs for a MAPPO run. CLI flags in ``train_mappo.py`` map 1:1 onto these."""

    scenario: str = "2x2"
    seed: int = 0
    critic: str = "centralized"  # "centralized" (MAPPO) | "local" (IPPO ablation)
    use_global_scalars: bool = False  # add normalized system_* scalars to the centralized state
    reward_fn: str = "diff-waiting-time"
    num_seconds: Optional[int] = None  # None -> scenario default
    total_steps: Optional[int] = None  # None -> scenario default

    # rollout / parallelism
    n_workers: int = 8
    n_steps: int = 512  # decision steps per worker per iteration
    sumo_backend: str = "auto"  # "auto" | "libsumo" | "traci"

    # optimisation
    lr: float = 5e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 10
    n_minibatches: int = 4
    norm_adv: bool = True
    value_norm: bool = True
    kl_target: Optional[float] = None  # early-stop epochs if approx_kl exceeds this

    # network
    hidden_dim: int = 128
    n_hidden: int = 2
    activation: str = "tanh"

    # eval / logging / io
    eval_every: int = 10  # iterations
    n_eval_seeds: int = 5
    n_eval_workers: int = 5
    save_every: int = 20  # iterations
    device: str = "cpu"
    run_name: Optional[str] = None
    out_root: str = "outputs/mappo"

    def resolved_num_seconds(self, scn: ScenarioConfig) -> int:
        return int(self.num_seconds if self.num_seconds is not None else scn.num_seconds)

    def resolved_total_steps(self, scn: ScenarioConfig) -> int:
        return int(self.total_steps if self.total_steps is not None else scn.default_total_steps)

    def eval_seeds(self) -> List[int]:
        return [EVAL_SEED_BASE + i for i in range(1, self.n_eval_seeds + 1)]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MAPPOConfig":
        import dataclasses

        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})
