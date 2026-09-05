"""Deterministic evaluation of a DQN policy and the Fixed-Time baseline.

Copied from ``experiments/mappo/eval.py`` with the policy parameter renamed
``mappo -> policy``; the KPI keys, aggregation and Fixed-Time path are byte-identical so
DQN and MAPPO numbers are directly comparable. One full episode per held-out seed;
SumoEnvironment writes its per-step KPI CSV automatically when the episode ends.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional, Sequence

import numpy as np

from .config import SCENARIOS, DQNConfig, ScenarioConfig
from .vec_sumo import SubprocSumoVecEnv, select_backend

_KPIS = ("system_total_waiting_time", "system_mean_waiting_time", "system_mean_speed", "system_total_stopped")


def _env_kwargs(scn: ScenarioConfig, cfg: DQNConfig, num_seconds: int, out_csv_prefix: Optional[str]) -> dict:
    return dict(
        net_file=scn.net_file,
        route_file=scn.route_file,
        num_seconds=num_seconds,
        delta_time=scn.delta_time,
        yellow_time=scn.yellow_time,
        min_green=scn.min_green,
        begin_time=scn.begin_time,
        reward_fn=cfg.reward_fn,
        add_system_info=True,
        add_per_agent_info=True,
        sumo_warnings=False,
        out_csv_name=out_csv_prefix,
    )


def _aggregate(steps: List[dict], seed: int) -> dict:
    arr = {k: np.array([s[k] for s in steps], np.float64) for k in _KPIS}
    return {
        "seed": seed,
        "n_steps": len(steps),
        "waiting_time_mean": float(arr["system_total_waiting_time"].mean()),
        "mean_wait_per_veh": float(arr["system_mean_waiting_time"].mean()),
        "mean_speed_mean": float(arr["system_mean_speed"].mean()),
        "stopped_mean": float(arr["system_total_stopped"].mean()),
        "arrived_end": float(steps[-1]["system_total_arrived"]) if steps else 0.0,
    }


def summarize(per_seed: List[dict]) -> dict:
    keys = ("waiting_time_mean", "mean_wait_per_veh", "mean_speed_mean", "stopped_mean", "arrived_end")
    out = {"n_seeds": len(per_seed), "seeds": [d["seed"] for d in per_seed]}
    for k in keys:
        v = np.array([d[k] for d in per_seed], np.float64)
        out[f"{k}_mean"] = float(v.mean())
        out[f"{k}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    out["per_seed"] = per_seed
    return out


def run_episodes(
    scn: ScenarioConfig,
    cfg: DQNConfig,
    seeds: Sequence[int],
    num_seconds: int,
    out_csv_prefix: Optional[str],
    policy=None,
    fixed_ts: bool = False,
    n_workers: Optional[int] = None,
) -> List[dict]:
    seeds = list(seeds)
    n_workers = min(n_workers or len(seeds), len(seeds))
    use_libsumo = select_backend(cfg.sumo_backend)
    env_kwargs = _env_kwargs(scn, cfg, num_seconds, out_csv_prefix)

    results: List[dict] = []
    for c0 in range(0, len(seeds), n_workers):
        chunk = seeds[c0 : c0 + n_workers]
        vec = SubprocSumoVecEnv(len(chunk), env_kwargs, use_libsumo, seed_base=0, fixed_ts=fixed_ts)
        obs, _g = vec.reset(seeds=chunk)
        acc: List[List[dict]] = [[] for _ in chunk]
        done_mask = np.zeros(len(chunk), bool)
        try:
            while not done_mask.all():
                if fixed_ts or policy is None:
                    actions = np.zeros((len(chunk), vec.n_agents), np.int64)
                else:
                    actions = policy.act_deterministic(obs)
                obs, _g, _r, dones, _to, _tg, step_info = vec.step(actions)
                for wi in range(len(chunk)):
                    if not done_mask[wi]:
                        acc[wi].append(step_info[wi])
                    if dones[wi]:
                        done_mask[wi] = True
        finally:
            vec.close()
        for wi, seed in enumerate(chunk):
            results.append(_aggregate(acc[wi], seed))
    return results


def run_fixed_time(scenario: str, cfg: DQNConfig, out_dir: str, num_seconds: Optional[int] = None) -> dict:
    scn = SCENARIOS[scenario]
    ns = int(num_seconds if num_seconds is not None else cfg.resolved_num_seconds(scn))
    seeds = cfg.eval_seeds()
    prefix = os.path.join(out_dir, "_fixed_time", "fixed")
    per_seed = run_episodes(scn, cfg, seeds, ns, prefix, policy=None, fixed_ts=True, n_workers=cfg.n_eval_workers)
    summary = summarize(per_seed)
    summary.update({"scenario": scenario, "num_seconds": ns, "reward_fn": cfg.reward_fn})
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "fixed_time_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _cli() -> None:
    p = argparse.ArgumentParser(description="DQN evaluation / Fixed-Time baseline")
    p.add_argument("--scenario", choices=list(SCENARIOS), required=True)
    p.add_argument("--fixed-time", action="store_true", help="run the Fixed-Time baseline")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="override eval seeds")
    p.add_argument("--n-eval-seeds", type=int, default=5)
    p.add_argument("--n-eval-workers", type=int, default=5)
    p.add_argument("--num-seconds", type=int, default=None)
    p.add_argument("--reward", dest="reward_fn", default="diff-waiting-time")
    p.add_argument("--sumo-backend", choices=["auto", "libsumo", "traci"], default="auto")
    p.add_argument("--out-root", default="outputs/dqn")
    p.add_argument("--checkpoint", default=None, help="DQN checkpoint .pt to evaluate (non-fixed-time)")
    args = p.parse_args()

    cfg = DQNConfig(
        scenario=args.scenario,
        reward_fn=args.reward_fn,
        n_eval_seeds=args.n_eval_seeds,
        n_eval_workers=args.n_eval_workers,
        sumo_backend=args.sumo_backend,
        num_seconds=args.num_seconds,
    )
    if args.seeds:
        cfg.n_eval_seeds = len(args.seeds)
    out_dir = os.path.join(args.out_root, args.scenario)

    if args.fixed_time:
        summary = run_fixed_time(args.scenario, cfg, out_dir, num_seconds=args.num_seconds)
        print(json.dumps({k: v for k, v in summary.items() if k != "per_seed"}, indent=2))
        return

    if not args.checkpoint:
        p.error("provide --checkpoint or --fixed-time")
    import torch

    from .dqn import DQNAgent
    from .state import global_state_dim

    scn = SCENARIOS[args.scenario]
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ck_cfg = DQNConfig.from_dict(ckpt["config"])
    seeds = args.seeds or ck_cfg.eval_seeds()
    ns = int(args.num_seconds or ck_cfg.resolved_num_seconds(scn))
    probe = SubprocSumoVecEnv(1, _env_kwargs(scn, ck_cfg, 200, None), select_backend(ck_cfg.sumo_backend))
    obs_dim, n_actions = probe.obs_dim, probe.n_actions
    probe.close()
    sd = global_state_dim(obs_dim, scn.n_agents, ck_cfg.use_global_scalars)
    agent = DQNAgent(obs_dim, n_actions, scn.n_agents, sd, ck_cfg, torch.device("cpu"))
    agent.load_state_dict(ckpt["agent"])
    prefix = os.path.join(out_dir, "eval_standalone", "dqn_eval")
    per_seed = run_episodes(scn, ck_cfg, seeds, ns, prefix, policy=agent, n_workers=args.n_eval_workers)
    print(json.dumps({k: v for k, v in summarize(per_seed).items() if k != "per_seed"}, indent=2))


if __name__ == "__main__":
    _cli()
