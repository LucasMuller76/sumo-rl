"""Infraestrutura compartilhada pelos experimentos de arquivo único (SUMO-RL + SB3).

Cada ``train_<algo>_<grid>.py`` só precisa definir um :class:`ExperimentSpec` e
uma função ``build_model(env, spec)``; toda a mecânica de ambiente, baselines,
avaliação e gráficos mora aqui — para que os resultados dos diferentes
algoritmos sejam diretamente comparáveis (mesmas seeds, mesmas métricas,
mesmos gráficos, mesma estrutura de saída em ``outputs/<grid>/<algo>/``).

Esta trilha é independente da trilha antiga em ``experiments/dqn/`` +
``experiments/mappo/`` (que tem seu próprio protocolo, seeds e saída em
``outputs/{dqn,mappo}/<cenário>/``) — não há necessidade de fazer as duas
coincidirem.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# --------------------------------------------------------------------------
# SUMO — precisa vir antes de qualquer import que dependa de traci/sumolib.
# --------------------------------------------------------------------------
if "SUMO_HOME" not in os.environ:
    sys.exit("Defina a variável de ambiente SUMO_HOME.")
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import supersuit as ss
import sumo_rl
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecMonitor


REPO_ROOT = Path(sumo_rl.__file__).resolve().parent.parent

EVAL_METRICS = [
    ("reward", "Recompensa"),
    ("waiting_time", "Tempo médio de espera (s)"),
    ("queue", "Fila média (veículos parados)"),
    ("mean_speed", "Velocidade média (m/s)"),
    ("throughput", "Throughput (veículos concluídos)"),
]


@dataclass
class ExperimentSpec:
    algo: str
    short: str
    grid_slug: str  # "2x2" ou "4x4" -- vira outputs/<grid_slug>/<short>/
    grid_label: str  # "grid 2x2" -- usado em títulos
    model_slug: str
    net_file: Path
    route_file: Path
    num_seconds: int = 3600
    total_timesteps: int = 500_000
    n_vec_envs: int = 6
    n_cpus: int = 6
    gamma: float = 0.99
    net_arch: list = field(default_factory=lambda: [256, 256])
    eval_episodes: int = 10

    @property
    def output_dir(self) -> Path:
        return REPO_ROOT / "outputs" / self.grid_slug / self.short

    @property
    def plots_dir(self) -> Path:
        return self.output_dir / "plots"

    @property
    def eval_seeds(self) -> list[int]:
        # Sementes de tráfego reservadas para avaliação. Como o treino usa
        # sumo_seed="random", nenhum cenário abaixo é visto durante o treino.
        return [90_000 + i for i in range(self.eval_episodes)]

    def check_files(self) -> None:
        for f in (self.net_file, self.route_file):
            if not Path(f).exists():
                sys.exit(f"Arquivo não encontrado: {f}")


# --------------------------------------------------------------------------
# Ambiente
# --------------------------------------------------------------------------
def make_env(spec: ExperimentSpec, use_gui: bool = False, fixed_ts: bool = False, sumo_seed="random"):
    spec.check_files()
    return sumo_rl.parallel_env(
        net_file=str(spec.net_file),
        route_file=str(spec.route_file),
        use_gui=use_gui,
        num_seconds=spec.num_seconds,
        sumo_seed=sumo_seed,
        add_system_info=True,
        add_per_agent_info=True,
        fixed_ts=fixed_ts,
    )


def make_training_env(spec: ExperimentSpec, run_dir: Path):
    # sumo_seed="random": cada cópia paralela e cada episódio pega um fluxo de
    # tráfego novo, para o modelo não decorar uma única realização.
    env = make_env(spec, sumo_seed="random")
    # agent_indicator: sem a identidade do agente na observação, o modelo
    # compartilhado não distingue as interseções (parameter sharing).
    env = ss.agent_indicator_v0(env, type_only=False)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env, num_vec_envs=spec.n_vec_envs, num_cpus=spec.n_cpus, base_class="stable_baselines3"
    )
    return VecMonitor(env, filename=str(run_dir / "monitor.csv"))


def load_model(model_cls, spec: ExperimentSpec, run_dir: Path):
    path = run_dir / f"{spec.model_slug}.zip"
    if not path.exists():
        sys.exit(f"Modelo não encontrado: {path}")
    return model_cls.load(str(run_dir / spec.model_slug), device="cpu")


def checkpoint_callback(spec: ExperimentSpec, run_dir: Path, denom: int | None = None) -> CheckpointCallback:
    denom = spec.n_vec_envs if denom is None else denom
    return CheckpointCallback(
        save_freq=max(50_000 // denom, 1),
        save_path=str(run_dir / "checkpoints"),
        name_prefix=spec.short,
    )


# --------------------------------------------------------------------------
# Treino (fluxo padrão -- MAPPO define o seu próprio, mais específico)
# --------------------------------------------------------------------------
def default_train(spec: ExperimentSpec, build_model: Callable, seed: int, denom: int | None = None) -> None:
    run_dir = spec.output_dir / f"seed_{seed}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    print(f"{spec.algo} — SUMO-RL — {spec.grid_label} — semente {seed} — {spec.total_timesteps:,} passos")

    # Semente aplicada manualmente: o ProcConcatVec do SuperSuit não implementa
    # env.seed(), então passar seed= ao modelo quebraria.
    set_random_seed(seed)

    env = make_training_env(spec, run_dir)
    try:
        model = build_model(env, spec)
        checkpoint = checkpoint_callback(spec, run_dir, denom=denom)
        model.learn(total_timesteps=spec.total_timesteps, callback=checkpoint, progress_bar=True)
        model.save(str(run_dir / spec.model_slug))
    finally:
        # Fecha os workers SUMO/SuperSuit mesmo se o treino falhar -- sem isso,
        # uma exceção aqui deixa subprocessos pendurados e o interpretador
        # trava na finalização (atexit do multiprocessing esperando por eles).
        env.close()

    plot_training_curve(spec)
    print(f"\nModelo salvo em {run_dir / spec.model_slug}.zip")


def make_default_train_fn(build_model: Callable, denom: int | None = None) -> Callable[[ExperimentSpec, int], None]:
    def _train(spec: ExperimentSpec, seed: int) -> None:
        default_train(spec, build_model, seed, denom=denom)

    return _train


# --------------------------------------------------------------------------
# Avaliação (idêntica para todos os algoritmos)
# --------------------------------------------------------------------------
def find_sumo_env(env):
    """Percorre os wrappers até achar o SumoEnvironment (tem .sumo e .traffic_signals)."""
    seen = set()
    cur = env
    for _ in range(30):
        if id(cur) in seen:
            break
        seen.add(id(cur))
        if getattr(cur, "traffic_signals", None) is not None and getattr(cur, "sumo", None) is not None:
            return cur
        for attr in ("env", "aec_env", "unwrapped"):
            nxt = getattr(cur, attr, None)
            if nxt is not None and nxt is not cur:
                cur = nxt
                break
        else:
            break
    return None


def run_episode(env, seed: int, policy_fn) -> dict:
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    waiting, queue, speed = [], [], []
    arrived = 0

    while env.agents:
        obs, rewards, _, _, infos = env.step(policy_fn(env, obs))
        total_reward += sum(float(r) for r in rewards.values())

        dicts = [i for i in infos.values() if isinstance(i, dict)]
        if dicts:
            waiting.append(np.mean([d.get("system_mean_waiting_time", 0.0) for d in dicts]))
            queue.append(np.mean([d.get("system_total_stopped", 0.0) for d in dicts]))
            speed.append(np.mean([d.get("system_mean_speed", 0.0) for d in dicts]))
            # system_total_arrived é um contador acumulado do SUMO-RL — basta
            # pegar o valor final do episódio (ler getArrivedNumber() por passo
            # subestima, porque cada passo do agente avança delta_time segundos).
            arrived = max(arrived, max(d.get("system_total_arrived", 0) for d in dicts))

    return {
        "reward": total_reward,
        "waiting_time": float(np.mean(waiting)) if waiting else 0.0,
        "queue": float(np.mean(queue)) if queue else 0.0,
        "mean_speed": float(np.mean(speed)) if speed else 0.0,
        "throughput": float(arrived),
    }


def fixed_policy(env, _obs):
    return {a: 0 for a in env.agents}


def maxpressure_policy(env, _obs):
    """Baseline max-pressure: cada semáforo escolhe a fase verde que maximiza
    (fila a montante - fila a jusante) nas conexões liberadas por ela."""
    base = find_sumo_env(env)
    tl, lane = base.sumo.trafficlight, base.sumo.lane
    signals = base.traffic_signals

    actions = {}
    for ts in env.agents:
        phases = signals[ts].green_phases
        links = tl.getControlledLinks(ts)
        best_idx, best_pressure = 0, None
        for idx, phase in enumerate(phases):
            state = phase.state
            pressure = 0.0
            for li, link in enumerate(links):
                if link and li < len(state) and state[li] in ("G", "g"):
                    in_lane, out_lane = link[0][0], link[0][1]
                    pressure += lane.getLastStepHaltingNumber(in_lane)
                    if out_lane:
                        pressure -= lane.getLastStepHaltingNumber(out_lane)
            if best_pressure is None or pressure > best_pressure:
                best_pressure, best_idx = pressure, idx
        actions[ts] = best_idx
    return actions


def model_policy(model):
    def policy(env, obs):
        return {a: int(np.asarray(model.predict(obs[a], deterministic=True)[0]).item()) for a in env.agents}

    return policy


def evaluate(spec: ExperimentSpec, model_cls) -> pd.DataFrame:
    seed_dirs = sorted(spec.output_dir.glob("seed_*"))
    if not seed_dirs:
        sys.exit(f"Nenhum modelo treinado em {spec.output_dir}. Rode antes: --train [--seed N]")

    rows = []

    # Baselines (tempo fixo e max-pressure): uma passada por semente de avaliação.
    fixed_env = make_env(spec, fixed_ts=True)
    mp_env = make_env(spec)
    for i, s in enumerate(spec.eval_seeds):
        r = run_episode(fixed_env, s, fixed_policy)
        r.update(episode=i + 1, controller="Fixed-Time", train_seed=-1)
        rows.append(r)
        r = run_episode(mp_env, s, maxpressure_policy)
        r.update(episode=i + 1, controller="Max-Pressure", train_seed=-1)
        rows.append(r)
    fixed_env.close()
    mp_env.close()

    # Modelo: uma passada por (semente de treino x semente de avaliação).
    for sd in seed_dirs:
        train_seed = int(sd.name.split("_")[1])
        model = load_model(model_cls, spec, sd)
        env = ss.agent_indicator_v0(make_env(spec), type_only=False)
        policy = model_policy(model)
        for i, s in enumerate(spec.eval_seeds):
            r = run_episode(env, s, policy)
            r.update(episode=i + 1, controller=spec.algo, train_seed=train_seed)
            rows.append(r)
        env.close()
        print(f"[{spec.algo}] semente de treino {train_seed}: avaliação concluída")

    df = pd.DataFrame(rows)
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(spec.output_dir / "evaluation.csv", index=False)
    plot_evaluation(spec, df)
    plot_comparison(spec, df)

    cols = [c for c, _ in EVAL_METRICS]
    print("\nResumo por controlador (média ± desvio sobre sementes e episódios):")
    print(df.groupby("controller")[cols].agg(["mean", "std"]))

    ref = df[df.controller == spec.algo][cols].mean()
    for name in ("Fixed-Time", "Max-Pressure"):
        base = df[df.controller == name][cols].mean()
        if base["waiting_time"] > 0:
            dw = (base["waiting_time"] - ref["waiting_time"]) / base["waiting_time"] * 100
            dt = (ref["throughput"] - base["throughput"]) / base["throughput"] * 100 if base["throughput"] else 0.0
            print(f"\n{spec.algo} vs {name}: espera {dw:+.1f}%  |  throughput {dt:+.1f}%")

    return df


# --------------------------------------------------------------------------
# Gráficos
# --------------------------------------------------------------------------
def plot_training_curve(spec: ExperimentSpec) -> None:
    monitors = sorted(spec.output_dir.glob("seed_*/monitor.csv"))
    if not monitors:
        return
    plt.figure(figsize=(10, 6))
    for m in monitors:
        try:
            d = pd.read_csv(m, skiprows=1)
        except Exception:
            continue
        if "r" in d.columns:
            plt.plot(d["r"].rolling(50, min_periods=1).mean(), label=m.parent.name)
    plt.xlabel("Episódio")
    plt.ylabel("Recompensa (média móvel de 50)")
    plt.title(f"{spec.algo} — Recompensa de treino — {spec.grid_label}")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    spec.plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(spec.plots_dir / f"{spec.short}_training_reward.png", dpi=200)
    plt.close()


def plot_evaluation(spec: ExperimentSpec, df: pd.DataFrame) -> None:
    sub = df[df.controller == spec.algo]
    grouped = sub.groupby("episode")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for ax, (col, label) in zip(axes, EVAL_METRICS):
        mean = grouped[col].mean()
        std = grouped[col].std().fillna(0.0)
        ax.plot(mean.index, mean.values, marker="o")
        ax.fill_between(mean.index, mean - std, mean + std, alpha=0.2)
        ax.set_xlabel("Episódio de avaliação")
        ax.set_ylabel(label)
        ax.set_title(f"{spec.algo} — {label}")
        ax.grid(alpha=0.25)
    axes[-1].axis("off")
    fig.suptitle(f"{spec.algo} — Métricas de avaliação — {spec.grid_label}", fontsize=15)
    plt.tight_layout()
    spec.plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(spec.plots_dir / f"{spec.short}_evaluation_metrics.png", dpi=200)
    plt.close()


def plot_comparison(spec: ExperimentSpec, df: pd.DataFrame) -> None:
    order = ["Fixed-Time", "Max-Pressure", spec.algo]
    summary = df.groupby("controller")[[c for c, _ in EVAL_METRICS]].mean().reindex(order)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for ax, (col, label) in zip(axes, EVAL_METRICS):
        ax.bar(summary.index, summary[col])
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].axis("off")
    fig.suptitle(f"Baselines vs {spec.algo} — {spec.grid_label}", fontsize=15)
    plt.tight_layout()
    spec.plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(spec.plots_dir / f"comparison_fixed_vs_{spec.short}.png", dpi=200)
    plt.close()


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------
def run_random(spec: ExperimentSpec, steps: int = 2500) -> None:
    env = make_env(spec, use_gui=True)
    env.reset(seed=spec.eval_seeds[0])
    for _ in range(steps):
        if not env.agents:
            break
        env.step({a: env.action_space(a).sample() for a in env.agents})
    env.close()


def render(spec: ExperimentSpec, model_cls, seed: int) -> None:
    sd = spec.output_dir / f"seed_{seed}"
    if not (sd / f"{spec.model_slug}.zip").exists():
        found = sorted(spec.output_dir.glob("seed_*"))
        if not found:
            sys.exit(f"Nenhum modelo treinado em {spec.output_dir}.")
        sd = found[0]
    model = load_model(model_cls, spec, sd)
    env = ss.agent_indicator_v0(make_env(spec, use_gui=True), type_only=False)
    obs, _ = env.reset(seed=spec.eval_seeds[0])
    policy = model_policy(model)
    while env.agents:
        obs, _, _, _, _ = env.step(policy(env, obs))
    env.close()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_cli(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--seed", type=int, default=42, help="semente da política; no --train grava em outputs/<grid>/<algo>/seed_<N>/"
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=None, help="sobrescreve o total de passos de treino (smoke tests)"
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=None, help="sobrescreve o número de episódios de avaliação"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--show", action="store_true", help="ações aleatórias com GUI (sanity check)")
    group.add_argument("--train", action="store_true", help="treina o modelo (semente de --seed)")
    group.add_argument(
        "--evaluate", action="store_true", help="avalia todas as sementes treinadas vs tempo fixo e max-pressure"
    )
    group.add_argument("--render", action="store_true", help="roda o modelo treinado com GUI")
    return parser


def run_cli(spec: ExperimentSpec, model_cls, train_fn: Callable[[ExperimentSpec, int], None]) -> None:
    parser = build_cli(f"{spec.algo} + SUMO-RL — {spec.grid_label}")
    args = parser.parse_args()

    if args.total_timesteps is not None:
        spec.total_timesteps = args.total_timesteps
    if args.eval_episodes is not None:
        spec.eval_episodes = args.eval_episodes

    if args.show:
        run_random(spec)
    elif args.train:
        train_fn(spec, args.seed)
    elif args.evaluate:
        evaluate(spec, model_cls)
    elif args.render:
        render(spec, model_cls, args.seed)
