"""Subprocess-parallel SUMO environments for MAPPO rollouts.

Each worker owns one ``SumoEnvironment(single_agent=False)`` and one SUMO process.
When ``libsumo`` is available the worker sets ``LIBSUMO_AS_TRACI=1`` *before*
importing sumo_rl (the flag is read once at import) for a large speedup; libsumo
allows only one simulation per process, which is exactly the one-SUMO-per-worker
layout here. Without libsumo it falls back to the plain traci backend.

Workers auto-reset on episode end (time limit) and hand back the terminal
observation so the trainer can bootstrap the value target.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from typing import List, Optional, Sequence

import numpy as np

from .state import GLOBAL_SCALAR_KEYS

# system_* keys logged per rollout/eval step (subset of SumoEnvironment info)
_STEP_INFO_KEYS = (
    "step",
    "system_total_waiting_time",
    "system_mean_waiting_time",
    "system_mean_speed",
    "system_total_stopped",
    "system_total_arrived",
)


def select_backend(pref: str) -> bool:
    """Return True if the libsumo backend should be used."""
    try:
        import libsumo  # noqa: F401

        have = True
    except Exception:
        have = False
    if pref == "traci":
        return False
    if pref == "libsumo":
        if not have:
            raise RuntimeError("sumo_backend='libsumo' but the 'libsumo' package is not installed.")
        return True
    return have  # "auto"


def _scalars(d: dict) -> np.ndarray:
    return np.array([float(d.get(k, 0.0)) for k in GLOBAL_SCALAR_KEYS], np.float32)


def _step_info(d: dict) -> dict:
    return {k: float(d.get(k, 0.0)) for k in _STEP_INFO_KEYS}


def _worker(remote, parent_remote, env_kwargs: dict, use_libsumo: bool, seed_base: int, widx: int, fixed_ts: bool):
    parent_remote.close()
    if use_libsumo:
        os.environ["LIBSUMO_AS_TRACI"] = "1"
    else:
        os.environ.pop("LIBSUMO_AS_TRACI", None)

    from sumo_rl import SumoEnvironment

    env = SumoEnvironment(fixed_ts=fixed_ts, single_agent=False, **env_kwargs)
    agents: List[str] = list(env.ts_ids)
    obs_dim = int(env.observation_spaces(agents[0]).shape[0])
    n_actions = int(env.action_spaces(agents[0]).n)
    episode = 0
    obs_cache: dict = {}

    def _stack(obs_dict: dict) -> np.ndarray:
        for a, o in obs_dict.items():
            obs_cache[a] = np.asarray(o, np.float32)
        return np.stack([obs_cache[a] for a in agents]).astype(np.float32)

    def _do_reset(seed: Optional[int]) -> tuple:
        nonlocal episode
        episode += 1
        s = int(seed_base + widx * 100 + episode) if seed is None else int(seed)
        obs_cache.clear()
        obs_dict = env.reset(seed=s)
        return _stack(obs_dict), _scalars(env._get_system_info())

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "spaces":
                remote.send((obs_dim, n_actions, agents))
            elif cmd == "reset":
                remote.send(_do_reset(data))
            elif cmd == "step":
                if fixed_ts:
                    obs_dict, rew_dict, done_dict, info = env.step({})
                else:
                    act_map = {a: int(data[i]) for i, a in enumerate(agents)}
                    obs_dict, rew_dict, done_dict, info = env.step(act_map)
                done = bool(done_dict["__all__"])
                rewards = np.array([float(rew_dict.get(a, 0.0)) for a in agents], np.float32)
                g = _scalars(info)
                sinfo = _step_info(info)
                if done:
                    term_obs = _stack(obs_dict)
                    term_g = g
                    next_obs, next_g = _do_reset(None)  # SumoEnvironment.reset() writes the finished episode's CSV
                    remote.send((next_obs, next_g, rewards, True, term_obs, term_g, sinfo))
                else:
                    remote.send((_stack(obs_dict), g, rewards, False, None, None, sinfo))
            elif cmd == "save_csv":
                env.save_csv(data[0], data[1])
                remote.send(True)
            elif cmd == "close":
                env.close()
                remote.close()
                break
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        try:
            env.close()
        except Exception:
            pass


class SubprocSumoVecEnv:
    """Vectorized SUMO env over ``n_workers`` spawned processes."""

    def __init__(
        self,
        n_workers: int,
        env_kwargs: dict,
        use_libsumo: bool,
        seed_base: int = 0,
        fixed_ts: bool = False,
    ):
        ctx = mp.get_context("spawn")
        self.n_workers = n_workers
        self.closed = False
        pipes = [ctx.Pipe() for _ in range(n_workers)]
        self.remotes = [p[0] for p in pipes]
        work_remotes = [p[1] for p in pipes]
        self.procs = []
        for i, (wr, r) in enumerate(zip(work_remotes, self.remotes)):
            p = ctx.Process(
                target=_worker,
                args=(wr, r, env_kwargs, use_libsumo, seed_base, i, fixed_ts),
                daemon=True,
            )
            p.start()
            self.procs.append(p)
        for wr in work_remotes:
            wr.close()

        self.remotes[0].send(("spaces", None))
        self.obs_dim, self.n_actions, self.agents = self.remotes[0].recv()
        self.n_agents = len(self.agents)

    def reset(self, seeds: Optional[Sequence[int]] = None):
        for i, r in enumerate(self.remotes):
            r.send(("reset", None if seeds is None else int(seeds[i])))
        out = [r.recv() for r in self.remotes]
        obs = np.stack([o[0] for o in out])  # [W, A, obs_dim]
        g = np.stack([o[1] for o in out])  # [W, G]
        return obs, g

    def step(self, actions: np.ndarray):
        """actions: ``[W, A]`` ints. Returns next_obs, next_g, rewards, dones, term_obs, term_g, step_info."""
        for i, r in enumerate(self.remotes):
            r.send(("step", np.asarray(actions[i]).astype(np.int64)))
        out = [r.recv() for r in self.remotes]
        next_obs = np.stack([o[0] for o in out])
        next_g = np.stack([o[1] for o in out])
        rewards = np.stack([o[2] for o in out])
        dones = np.array([o[3] for o in out], np.float32)
        term_obs = [o[4] for o in out]  # list[None | [A, obs_dim]]
        term_g = [o[5] for o in out]
        step_info = [o[6] for o in out]
        return next_obs, next_g, rewards, dones, term_obs, term_g, step_info

    def save_csv(self, names: Sequence[str], episode: int) -> None:
        for i, r in enumerate(self.remotes):
            r.send(("save_csv", (names[i], episode)))
        for r in self.remotes:
            r.recv()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for r in self.remotes:
            try:
                r.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=15)
            if p.is_alive():
                p.terminate()
        for r in self.remotes:
            try:
                r.close()
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
