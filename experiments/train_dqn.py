"""Train multi-agent DQN (IDQN+PS / VDN / QMIX) for SUMO traffic-signal control on 2x2 / 4x4.

Example
-------
    python -m experiments.train_dqn --scenario 2x2 --seed 0 --n-workers 8            # IDQN + PS
    python -m experiments.train_dqn --scenario 4x4 --seed 0 --n-workers 8 --mixer qmix
    python -m experiments.train_dqn --scenario 2x2 --seed 0 --no-per                  # textbook ablation

Synchronous parallel collection (``--n-workers`` spawned SUMO processes) + a single learner.
Evaluation follows the MAPPO protocol exactly (held-out seeds, deterministic argmax,
Fixed-Time baseline); eval/checkpoint cadence is in AGENT-STEPS so the x-axis lines up with
``outputs/mappo``. Outputs: ``outputs/dqn/<scenario>/<run_name>/``.
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

from experiments.dqn.config import SCENARIOS, DQNConfig
from experiments.dqn.dqn import DQNAgent, epsilon_ladder
from experiments.dqn.eval import run_episodes, run_fixed_time, summarize
from experiments.dqn.replay import NStepAccumulator, make_replay
from experiments.dqn.state import global_state, global_state_dim
from experiments.dqn.utils import (
    CsvLogger,
    TBWriter,
    git_hash,
    lib_versions,
    resolve_device,
    save_checkpoint,
    seed_everything,
)
from experiments.dqn.vec_sumo import SubprocSumoVecEnv, select_backend

_TRAIN_COLS = [
    "iter", "global_step", "wallclock_s", "sps", "epsilon_mean", "buffer_size", "updates_done",
    "lr", "loss", "q_mean", "q_target_mean", "td_error_abs_mean", "grad_norm",
    "ep_return_mean", "ep_wait_mean", "ep_len_steps",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--scenario", choices=list(SCENARIOS), required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mixer", choices=["none", "vdn", "qmix"], default="none",
                   help="'none' = IDQN + parameter sharing; 'vdn'/'qmix' = CTDE value mixing")
    p.add_argument("--no-per", dest="per", action="store_false", help="uniform replay instead of PER")
    p.add_argument("--global-scalars", dest="use_global_scalars", action="store_true",
                   help="append normalized system_* scalars to the QMIX team state")
    p.add_argument("--reward", dest="reward_fn", default="diff-waiting-time",
                   choices=["diff-waiting-time", "pressure", "queue", "average-speed"])
    p.add_argument("--num-seconds", type=int, default=None)
    p.add_argument("--total-steps", type=int, default=None, help="agent-steps budget (scenario default if unset)")

    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--collect-steps", type=int, default=8, help="env steps per worker per iteration")
    p.add_argument("--sumo-backend", choices=["auto", "libsumo", "traci"], default="auto")

    p.add_argument("--replay-capacity", type=int, default=None, help="joint items (worker-steps)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--replay-ratio", type=float, default=4.0, help="mean #times each transition is replayed")
    p.add_argument("--learning-starts", type=int, default=None, help="random-action warmup, agent-steps")
    p.add_argument("--n-step", type=int, default=3)
    p.add_argument("--per-alpha", type=float, default=0.6)
    p.add_argument("--per-beta0", type=float, default=0.4)

    p.add_argument("--lr", type=float, default=None, help="default 1e-4 (PER) / 2.5e-4 (--no-per)")
    p.add_argument("--anneal-lr", action="store_true")
    p.add_argument("--adam-eps", type=float, default=1.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--max-grad-norm", type=float, default=10.0)
    p.add_argument("--huber-delta", type=float, default=1.0)
    p.add_argument("--target-tau", type=float, default=0.005)
    p.add_argument("--hard-target", type=int, default=0, help="if >0, hard target copy every N updates")
    p.add_argument("--no-double", dest="double", action="store_false")
    p.add_argument("--no-dueling", dest="dueling", action="store_false")

    p.add_argument("--eps-base", type=float, default=0.4)
    p.add_argument("--eps-alpha", type=float, default=7.0)
    p.add_argument("--eps-min", type=float, default=0.01)

    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--n-hidden", type=int, default=2)

    p.add_argument("--eval-every-steps", type=int, default=120_000)
    p.add_argument("--save-every-steps", type=int, default=240_000)
    p.add_argument("--n-eval-seeds", type=int, default=5)
    p.add_argument("--n-eval-workers", type=int, default=5)
    p.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="cpu")
    p.add_argument("--torch-threads", type=int, default=2)
    p.add_argument("--run-name", default=None)
    p.add_argument("--out-root", default="outputs/dqn")
    p.add_argument("--resume", default=None)
    p.add_argument("--skip-fixed-time", action="store_true")
    return p.parse_args()


def cfg_from_args(a: argparse.Namespace) -> DQNConfig:
    return DQNConfig(
        scenario=a.scenario, seed=a.seed, mixer=a.mixer, per=a.per, use_global_scalars=a.use_global_scalars,
        reward_fn=a.reward_fn, num_seconds=a.num_seconds, total_steps=a.total_steps,
        n_workers=a.n_workers, collect_steps=a.collect_steps, sumo_backend=a.sumo_backend,
        replay_capacity=a.replay_capacity, batch_size=a.batch_size, replay_ratio=a.replay_ratio,
        learning_starts=a.learning_starts, n_step=a.n_step, per_alpha=a.per_alpha, per_beta0=a.per_beta0,
        lr=a.lr, anneal_lr=a.anneal_lr, adam_eps=a.adam_eps, gamma=a.gamma,
        max_grad_norm=a.max_grad_norm, huber_delta=a.huber_delta, target_tau=a.target_tau,
        hard_target=a.hard_target, double=a.double, dueling=a.dueling,
        eps_base=a.eps_base, eps_alpha=a.eps_alpha, eps_min=a.eps_min,
        hidden_dim=a.hidden_dim, n_hidden=a.n_hidden,
        eval_every_steps=a.eval_every_steps, save_every_steps=a.save_every_steps,
        n_eval_seeds=a.n_eval_seeds, n_eval_workers=a.n_eval_workers,
        device=a.device, torch_threads=a.torch_threads, run_name=a.run_name, out_root=a.out_root,
    )


def make_env_kwargs(scn, cfg: DQNConfig, num_seconds: int, out_csv_name):
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
    torch.set_num_threads(max(1, cfg.torch_threads))
    device = resolve_device(cfg.device)
    ss = np.random.SeedSequence(cfg.seed).spawn(3)
    act_rng = np.random.default_rng(ss[0])
    replay_rng = np.random.default_rng(ss[1])

    num_seconds = cfg.resolved_num_seconds(scn)
    total_steps = cfg.resolved_total_steps(scn)
    learning_starts = cfg.resolved_learning_starts(scn)
    lr = cfg.resolved_lr()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"{cfg.mixer}_{'per' if cfg.per else 'noper'}"
    run_name = cfg.run_name or f"dqn_{cfg.scenario}_{tag}_r-{cfg.reward_fn}_s{cfg.seed}_{ts}"
    run_dir = os.path.join(cfg.out_root, cfg.scenario, run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    env_csv_dir = os.path.join(run_dir, "env_csv")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(env_csv_dir, exist_ok=True)

    use_libsumo = select_backend(cfg.sumo_backend)
    vec = SubprocSumoVecEnv(cfg.n_workers, make_env_kwargs(scn, cfg, num_seconds, None), use_libsumo,
                            seed_base=cfg.seed * 10_000)
    W, A = cfg.n_workers, vec.n_agents
    obs_dim, n_actions = vec.obs_dim, vec.n_actions
    if A != scn.n_agents:
        print(f"[warn] scenario preset n_agents={scn.n_agents} but env reports {A}; using {A}.")

    team_sd = global_state_dim(obs_dim, A, cfg.use_global_scalars)
    replay_sd = team_sd if cfg.mixer == "qmix" else 0
    agent = DQNAgent(obs_dim, n_actions, A, team_sd, cfg, device)
    replay = make_replay(cfg, scn, obs_dim, replay_sd, replay_rng)
    nstep = NStepAccumulator(W, cfg.n_step, cfg.gamma)
    eps_ladder = epsilon_ladder(W, cfg.eps_base, cfg.eps_alpha, cfg.eps_min)

    updates_per_iter = max(1, round(cfg.replay_ratio * cfg.collect_steps * W * A / cfg.batch_size))
    steps_per_iter = cfg.collect_steps * W * A
    n_iters_est = max(1, math.ceil(total_steps / steps_per_iter))
    ep_len = num_seconds // scn.delta_time

    global_step, updates_done, best_wait = 0, 0, math.inf
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        agent.load_state_dict(ck["agent"])
        global_step = int(ck.get("global_step", 0))
        updates_done = int(ck.get("updates_done", agent.updates_done))
        print(f"[resume] {args.resume} @ global_step {global_step} (replay NOT restored -> approximate)")

    config_blob = {
        "config": cfg.to_dict(),
        "resolved": {"num_seconds": num_seconds, "total_steps": total_steps, "learning_starts": learning_starts,
                     "lr": lr, "updates_per_iter": updates_per_iter, "steps_per_iter": steps_per_iter,
                     "n_iters_est": n_iters_est, "episode_len_steps": ep_len, "replay_capacity": len(replay.gamma_m),
                     "obs_dim": obs_dim, "n_actions": n_actions, "n_agents": A, "team_state_dim": team_sd,
                     "eps_ladder": [round(float(e), 4) for e in eps_ladder],
                     "device": str(device), "libsumo": use_libsumo, "agents": vec.agents},
        "git": git_hash(), "versions": lib_versions(),
        "net_file": scn.net_file, "route_file": scn.route_file, "started": ts, "run_dir": run_dir,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config_blob, f, indent=2)
    print(json.dumps(config_blob["resolved"], indent=2))

    train_log = CsvLogger(os.path.join(run_dir, "train_log.csv"))
    eval_log = CsvLogger(os.path.join(run_dir, "eval_log.csv"))
    tb = TBWriter(os.path.join(run_dir, "tensorboard"))

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
    last_stats: dict = {}
    it = 0
    next_eval = cfg.eval_every_steps
    next_save = cfg.save_every_steps
    last_eval_step = -1

    def evaluate() -> None:
        nonlocal best_wait, last_eval_step
        if global_step == last_eval_step:
            return
        last_eval_step = global_step
        prefix = os.path.join(env_csv_dir, f"dqn_eval_step{global_step}")
        per_seed = run_episodes(scn, cfg, cfg.eval_seeds(), num_seconds, prefix,
                                policy=agent, n_workers=cfg.n_eval_workers)
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
        print(f"    [eval @ {global_step}] waiting {ev['waiting_time_mean_mean']:.1f} "
              f"+/- {ev['waiting_time_mean_std']:.1f}"
              + (f"   ({imp:+.1f}% vs Fixed-Time)" if isinstance(imp, float) else ""))
        if ev["waiting_time_mean_mean"] < best_wait:
            best_wait = ev["waiting_time_mean_mean"]
            save_checkpoint(os.path.join(ckpt_dir, "best.pt"), agent, global_step, cfg,
                            extra={"updates_done": updates_done,
                                   "eval": {k: v for k, v in ev.items() if k != "per_seed"}})

    while global_step < total_steps:
        it += 1
        # ---------------- collect ----------------
        for _ in range(cfg.collect_steps):
            gstate = global_state(obs, g, cfg.use_global_scalars) if cfg.mixer == "qmix" else None  # [W, S]
            if global_step < learning_starts:
                actions = act_rng.integers(0, n_actions, size=(W, A)).astype(np.int64)
            else:
                actions, _q = agent.act(obs, eps_ladder, act_rng)

            next_obs, next_g, rewards, dones, term_obs, term_g, step_info = vec.step(actions)

            done_idx = np.nonzero(dones)[0]
            nobs_all = next_obs.copy()
            ng_all = next_g.copy()
            for w in done_idx:
                nobs_all[w] = term_obs[w]
                if term_g[w] is not None:
                    ng_all[w] = term_g[w]
            ngstate = global_state(nobs_all, ng_all, cfg.use_global_scalars) if cfg.mixer == "qmix" else None

            for w in range(W):
                items = nstep.add(
                    w, obs[w], actions[w], rewards[w], nobs_all[w], bool(dones[w]),
                    None if gstate is None else gstate[w],
                    None if ngstate is None else ngstate[w],
                )
                if items:
                    replay.add_batch(items)

            cur_ret += rewards.sum(axis=1)
            cur_wait_sum += np.array([si["system_total_waiting_time"] for si in step_info])
            cur_len += 1
            for w in done_idx:
                ep_returns.append(float(cur_ret[w]))
                ep_waits.append(float(cur_wait_sum[w] / max(1, cur_len[w])))
                cur_ret[w] = cur_wait_sum[w] = 0.0
                cur_len[w] = 0

            obs, g = next_obs, next_g
            global_step += W * A

        # ---------------- learn ----------------
        if global_step >= learning_starts and len(replay) >= cfg.batch_size:
            frac = min(1.0, max(0.0, (global_step - learning_starts) / max(1, total_steps - learning_starts)))
            beta = cfg.per_beta0 + (1.0 - cfg.per_beta0) * frac
            if cfg.anneal_lr:
                for grp in agent.optimizer.param_groups:
                    grp["lr"] = lr * (1.0 - frac)
            for _ in range(updates_per_iter):
                batch = replay.sample(cfg.batch_size, beta)
                last_stats = agent.update(batch, beta)
                if replay.prioritized:
                    replay.update_priorities(batch["idx"], last_stats["td_abs"])
                updates_done += 1

        # ---------------- log ----------------
        elapsed = time.time() - t_start
        sps = global_step / max(1e-9, elapsed)
        cur_lr = agent.optimizer.param_groups[0]["lr"]
        row = {c: "" for c in _TRAIN_COLS}
        row.update({
            "iter": it, "global_step": global_step, "wallclock_s": round(elapsed, 1), "sps": round(sps, 1),
            "epsilon_mean": round(float(eps_ladder.mean()), 4), "buffer_size": len(replay),
            "updates_done": updates_done, "lr": cur_lr, "ep_len_steps": ep_len,
            "ep_return_mean": float(np.mean(ep_returns[-W:])) if ep_returns else "",
            "ep_wait_mean": float(np.mean(ep_waits[-W:])) if ep_waits else "",
        })
        if last_stats:
            row.update({
                "loss": round(last_stats["loss"], 5), "q_mean": round(last_stats["q_mean"], 5),
                "q_target_mean": round(last_stats["q_target_mean"], 5),
                "td_error_abs_mean": round(last_stats["td_error_abs_mean"], 5),
                "grad_norm": round(last_stats["grad_norm"], 5),
            })
        train_log.log(row)
        for k, v in row.items():
            if isinstance(v, (int, float)):
                tb.add(f"train/{k}", v, global_step)
        phase = "warmup" if global_step < learning_starts else "train"
        print(f"it {it:5d} [{phase}] step {global_step:>9d}  sps {sps:6.1f}  buf {len(replay):>7d}  "
              f"upd {updates_done:>7d}  loss {row['loss'] if row['loss'] != '' else '   nan'}  "
              f"q {row['q_mean'] if row['q_mean'] != '' else 'nan'}")

        # ---------------- eval / checkpoint (agent-step cadence) ----------------
        if global_step >= next_eval:
            evaluate()
            next_eval += cfg.eval_every_steps
        if global_step >= next_save:
            save_checkpoint(os.path.join(ckpt_dir, "last.pt"), agent, global_step, cfg,
                            extra={"updates_done": updates_done})
            next_save += cfg.save_every_steps

    evaluate()
    save_checkpoint(os.path.join(ckpt_dir, "last.pt"), agent, global_step, cfg, extra={"updates_done": updates_done})
    vec.close()
    tb.close()
    print(f"done. best eval waiting_time_mean = {best_wait:.1f}. outputs in {run_dir}")


if __name__ == "__main__":
    main()
