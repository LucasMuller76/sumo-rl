"""Train MAPPO (CTDE) for SUMO traffic-signal control on the 2x2 / 4x4 grids.

Example
-------
    python -m experiments.train_mappo --scenario 2x2 --seed 0 --n-workers 8
    python -m experiments.train_mappo --scenario 4x4 --seed 0 --n-workers 8 --eval-every 10

Outputs land in ``outputs/mappo/<scenario>/<run_name>/`` (config.json, train_log.csv,
eval_log.csv, checkpoints/, env_csv/, tensorboard/). The Fixed-Time baseline for the
scenario is written once to ``outputs/mappo/<scenario>/fixed_time_summary.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime

import numpy as np
import torch

from experiments.mappo.buffer import RolloutBuffer
from experiments.mappo.config import SCENARIOS, MAPPOConfig
from experiments.mappo.eval import run_episodes, run_fixed_time, summarize
from experiments.mappo.mappo import MAPPO
from experiments.mappo.networks import RunningMeanStd
from experiments.mappo.state import GLOBAL_SCALAR_KEYS, build_state, state_dim
from experiments.mappo.utils import (
    CsvLogger,
    TBWriter,
    git_hash,
    lib_versions,
    resolve_device,
    save_checkpoint,
    seed_everything,
)
from experiments.mappo.vec_sumo import SubprocSumoVecEnv, select_backend


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--scenario", choices=list(SCENARIOS), required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--critic", choices=["centralized", "local"], default="centralized",
                   help="'centralized' = MAPPO; 'local' = IPPO ablation")
    p.add_argument("--global-scalars", dest="use_global_scalars", action="store_true",
                   help="append normalized system_* scalars to the centralized state")
    p.add_argument("--reward", dest="reward_fn", default="diff-waiting-time",
                   choices=["diff-waiting-time", "pressure", "queue", "average-speed"])
    p.add_argument("--num-seconds", type=int, default=None, help="episode length (scenario default if unset)")
    p.add_argument("--total-steps", type=int, default=None, help="agent-steps budget (scenario default if unset)")

    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--sumo-backend", choices=["auto", "libsumo", "traci"], default="auto")

    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--no-anneal-lr", dest="anneal_lr", action="store_false")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--n-minibatches", type=int, default=4)
    p.add_argument("--no-value-norm", dest="value_norm", action="store_false")
    p.add_argument("--no-norm-adv", dest="norm_adv", action="store_false")
    p.add_argument("--kl-target", type=float, default=None)

    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--n-hidden", type=int, default=2)

    p.add_argument("--eval-every", type=int, default=10, help="iterations between evaluations")
    p.add_argument("--n-eval-seeds", type=int, default=5)
    p.add_argument("--n-eval-workers", type=int, default=5)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="cpu")
    p.add_argument("--torch-threads", type=int, default=2)
    p.add_argument("--run-name", default=None)
    p.add_argument("--out-root", default="outputs/mappo")
    p.add_argument("--resume", default=None, help="checkpoint .pt to resume from")
    p.add_argument("--skip-fixed-time", action="store_true", help="do not (re)run the Fixed-Time baseline")
    return p.parse_args()


def cfg_from_args(a: argparse.Namespace) -> MAPPOConfig:
    return MAPPOConfig(
        scenario=a.scenario, seed=a.seed, critic=a.critic, use_global_scalars=a.use_global_scalars,
        reward_fn=a.reward_fn, num_seconds=a.num_seconds, total_steps=a.total_steps,
        n_workers=a.n_workers, n_steps=a.n_steps, sumo_backend=a.sumo_backend,
        lr=a.lr, anneal_lr=a.anneal_lr, gamma=a.gamma, gae_lambda=a.gae_lambda,
        clip_range=a.clip_range, ent_coef=a.ent_coef, vf_coef=a.vf_coef, max_grad_norm=a.max_grad_norm,
        n_epochs=a.n_epochs, n_minibatches=a.n_minibatches, value_norm=a.value_norm, norm_adv=a.norm_adv,
        kl_target=a.kl_target, hidden_dim=a.hidden_dim, n_hidden=a.n_hidden,
        eval_every=a.eval_every, n_eval_seeds=a.n_eval_seeds, n_eval_workers=a.n_eval_workers,
        save_every=a.save_every, device=a.device, run_name=a.run_name, out_root=a.out_root,
    )


def make_env_kwargs(scn, cfg: MAPPOConfig, num_seconds: int, out_csv_name):
    return dict(
        net_file=scn.net_file, route_file=scn.route_file, num_seconds=num_seconds,
        delta_time=scn.delta_time, yellow_time=scn.yellow_time, min_green=scn.min_green,
        begin_time=scn.begin_time, reward_fn=cfg.reward_fn,
        add_system_info=True, add_per_agent_info=True, sumo_warnings=False,
        out_csv_name=out_csv_name,
    )


def main() -> None:
    args = parse_args()
    cfg = cfg_from_args(args)
    scn = SCENARIOS[cfg.scenario]

    seed_everything(cfg.seed)
    torch.set_num_threads(max(1, args.torch_threads))
    device = resolve_device(cfg.device)

    num_seconds = cfg.resolved_num_seconds(scn)
    total_steps = cfg.resolved_total_steps(scn)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = cfg.run_name or f"mappo_{cfg.scenario}_{cfg.critic}_r-{cfg.reward_fn}_s{cfg.seed}_{ts}"
    run_dir = os.path.join(cfg.out_root, cfg.scenario, run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    env_csv_dir = os.path.join(run_dir, "env_csv")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(env_csv_dir, exist_ok=True)

    use_libsumo = select_backend(cfg.sumo_backend)

    train_kwargs = make_env_kwargs(scn, cfg, num_seconds, out_csv_name=None)  # no CSV for training workers
    vec = SubprocSumoVecEnv(cfg.n_workers, train_kwargs, use_libsumo, seed_base=cfg.seed * 10_000)
    W, A = cfg.n_workers, vec.n_agents
    obs_dim, n_actions = vec.obs_dim, vec.n_actions
    if A != scn.n_agents:
        print(f"[warn] scenario preset n_agents={scn.n_agents} but env reports {A}; using {A}.")

    sd = state_dim(obs_dim, A, cfg.critic, cfg.use_global_scalars)
    mappo = MAPPO(obs_dim, n_actions, sd, cfg, device)
    buf = RolloutBuffer(cfg.n_steps, W, A, obs_dim, sd, device)
    g_rms = RunningMeanStd(shape=(len(GLOBAL_SCALAR_KEYS),)) if cfg.use_global_scalars else None

    steps_per_iter = cfg.n_steps * W * A
    n_iters = max(1, math.ceil(total_steps / steps_per_iter))
    ep_len = num_seconds // scn.delta_time

    start_iter, global_step = 0, 0
    best_wait = math.inf
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        mappo.load_state_dict(ck["mappo"])
        start_iter = int(ck.get("iteration", 0))
        global_step = int(ck.get("global_step", 0))
        if g_rms is not None and ck.get("g_rms") is not None:
            g_rms.load_state_dict(ck["g_rms"])
        print(f"[resume] from {args.resume} at iter {start_iter}, global_step {global_step}")

    config_blob = {
        "config": cfg.to_dict(),
        "resolved": {"num_seconds": num_seconds, "total_steps": total_steps, "n_iters": n_iters,
                     "steps_per_iter": steps_per_iter, "episode_len_steps": ep_len,
                     "obs_dim": obs_dim, "n_actions": n_actions, "n_agents": A, "state_dim": sd,
                     "device": str(device), "libsumo": use_libsumo, "agents": vec.agents},
        "git": git_hash(), "versions": lib_versions(),
        "net_file": scn.net_file, "route_file": scn.route_file,
        "started": ts, "run_dir": run_dir,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config_blob, f, indent=2)
    print(json.dumps(config_blob["resolved"], indent=2))

    train_log = CsvLogger(os.path.join(run_dir, "train_log.csv"))
    eval_log = CsvLogger(os.path.join(run_dir, "eval_log.csv"))
    tb = TBWriter(os.path.join(run_dir, "tensorboard"))

    # Fixed-Time baseline for this scenario (write once; shared across runs of the scenario).
    ft_path = os.path.join(cfg.out_root, cfg.scenario, "fixed_time_summary.json")
    if not args.skip_fixed_time and not os.path.exists(ft_path):
        print("[fixed-time] running baseline ...")
        ft = run_fixed_time(cfg.scenario, cfg, os.path.join(cfg.out_root, cfg.scenario), num_seconds=num_seconds)
        print(f"[fixed-time] waiting_time_mean = {ft['waiting_time_mean_mean']:.1f} "
              f"(+/- {ft['waiting_time_mean_std']:.1f}) over {ft['n_seeds']} seeds")
    ft_summary = None
    if os.path.exists(ft_path):
        with open(ft_path) as f:
            ft_summary = json.load(f)

    obs, g = vec.reset()
    t_start = time.time()

    ep_returns: list[float] = []
    ep_waits: list[float] = []
    cur_ret = np.zeros(W, np.float64)
    cur_wait_sum = np.zeros(W, np.float64)
    cur_len = np.zeros(W, np.int64)

    for it in range(start_iter, n_iters):
        buf.reset()
        frac = 1.0 - it / n_iters
        lr_now = cfg.lr * frac if cfg.anneal_lr else cfg.lr

        for t in range(cfg.n_steps):
            if g_rms is not None:
                g_rms.update(g)
                gm, gsd = g_rms.mean, g_rms.std
            else:
                gm = gsd = None
            state = build_state(obs, g, A, cfg.critic, cfg.use_global_scalars, gm, gsd)
            actions_flat, logp_flat, values_flat = mappo.act(obs, state)
            actions = actions_flat.reshape(W, A)

            next_obs, next_g, rewards, dones, term_obs, term_g, step_info = vec.step(actions)

            buf.add(
                obs=obs, states=state, actions=actions,
                logprobs=logp_flat.reshape(W, A), values=values_flat.reshape(W, A),
                rewards=rewards, dones=dones,
            )

            if dones.any():
                tv = np.zeros((W, A), np.float32)
                for w in np.nonzero(dones)[0]:
                    tobs = term_obs[w][None]  # [1, A, O]
                    tg = None if term_g[w] is None else term_g[w][None]
                    tstate = build_state(tobs, tg, A, cfg.critic, cfg.use_global_scalars, gm, gsd)
                    tv[w] = mappo.get_values(tstate)[0]
                buf.set_term_values(t, tv)

            cur_ret += rewards.sum(axis=1)
            cur_wait_sum += np.array([si["system_total_waiting_time"] for si in step_info])
            cur_len += 1
            for w in np.nonzero(dones)[0]:
                ep_returns.append(float(cur_ret[w]))
                ep_waits.append(float(cur_wait_sum[w] / max(1, cur_len[w])))
                cur_ret[w] = 0.0
                cur_wait_sum[w] = 0.0
                cur_len[w] = 0

            obs, g = next_obs, next_g
            global_step += W * A

        # bootstrap for the final (non-boundary) step
        if g_rms is not None:
            gm, gsd = g_rms.mean, g_rms.std
        else:
            gm = gsd = None
        last_state = build_state(obs, g, A, cfg.critic, cfg.use_global_scalars, gm, gsd)
        last_values = mappo.get_values(last_state)  # [W, A]
        buf.compute_gae(last_values, cfg.gamma, cfg.gae_lambda)

        stats = mappo.update(buf, lr_now)

        elapsed = time.time() - t_start
        sps = global_step / max(1e-9, elapsed)
        row = {
            "iter": it, "global_step": global_step, "wallclock_s": round(elapsed, 1),
            "sps": round(sps, 1), "lr": lr_now,
            "episodes_done": len(ep_returns),
            "ep_return_mean": float(np.mean(ep_returns[-W:])) if ep_returns else "",
            "ep_wait_mean": float(np.mean(ep_waits[-W:])) if ep_waits else "",
            "ep_len_steps": ep_len,
            **{k: round(v, 5) if isinstance(v, float) else v for k, v in stats.items()},
        }
        train_log.log(row)
        for k, v in row.items():
            if isinstance(v, (int, float)):
                tb.add(f"train/{k}", v, global_step)
        print(f"it {it:4d}/{n_iters}  step {global_step:>9d}  sps {sps:6.1f}  "
              f"pg {stats['policy_loss']:+.3f}  v {stats['value_loss']:.3f}  "
              f"ent {stats['entropy']:.3f}  kl {stats['approx_kl']:.4f}  ev {stats['explained_variance']:+.2f}")

        is_last = it == n_iters - 1
        if cfg.eval_every > 0 and (it % cfg.eval_every == 0 or is_last):
            prefix = os.path.join(env_csv_dir, f"mappo_eval_it{it}")
            per_seed = run_episodes(scn, cfg, cfg.eval_seeds(), num_seconds, prefix,
                                    mappo=mappo, n_workers=cfg.n_eval_workers)
            ev = summarize(per_seed)
            erow = {"iter": it, "global_step": global_step,
                    "waiting_time_mean": ev["waiting_time_mean_mean"], "waiting_time_std": ev["waiting_time_mean_std"],
                    "mean_wait_per_veh": ev["mean_wait_per_veh_mean"],
                    "mean_speed_mean": ev["mean_speed_mean_mean"], "mean_speed_std": ev["mean_speed_mean_std"],
                    "stopped_mean": ev["stopped_mean_mean"], "stopped_std": ev["stopped_mean_std"],
                    "arrived_end_mean": ev["arrived_end_mean"]}
            if ft_summary is not None:
                fw = ft_summary["waiting_time_mean_mean"]
                erow["fixed_time_waiting"] = fw
                erow["improvement_pct_vs_fixed"] = 100.0 * (fw - ev["waiting_time_mean_mean"]) / fw if fw else ""
            eval_log.log(erow)
            for k, v in erow.items():
                if isinstance(v, (int, float)):
                    tb.add(f"eval/{k}", v, global_step)
            imp = erow.get("improvement_pct_vs_fixed", "")
            print(f"    [eval] waiting {ev['waiting_time_mean_mean']:.1f} +/- {ev['waiting_time_mean_std']:.1f}"
                  + (f"   ({imp:+.1f}% vs Fixed-Time)" if isinstance(imp, float) else ""))

            if ev["waiting_time_mean_mean"] < best_wait:
                best_wait = ev["waiting_time_mean_mean"]
                save_checkpoint(os.path.join(ckpt_dir, "best.pt"), mappo, it, global_step, cfg, g_rms,
                                extra={"eval": {k: v for k, v in ev.items() if k != "per_seed"}})

        if cfg.save_every > 0 and (it % cfg.save_every == 0 or is_last):
            save_checkpoint(os.path.join(ckpt_dir, "last.pt"), mappo, it + 1, global_step, cfg, g_rms)

    vec.close()
    tb.close()
    print(f"done. best eval waiting_time_mean = {best_wait:.1f}. outputs in {run_dir}")


if __name__ == "__main__":
    main()
