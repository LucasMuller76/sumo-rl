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

Estudo de recompensas: os arquivos ``train_<algo>_<grid>_reward_<nome>.py``
(ex.: ``train_ppo_4x4_reward_queue.py``) fixam a recompensa de treino via
``run_cli(..., reward=...)``; ``--reward <nome>`` em qualquer
``train_<algo>_<grid>.py`` faz o mesmo para algoritmos sem arquivo próprio. Nos
dois casos a recompensa é normalizada com ``VecNormalize`` (para que escalas
diferentes não mudem o equilíbrio ator/crítico) e a saída vai para
``outputs/<grid>/rewards/<nome>/<algo>/`` — sem tocar nos resultados da
comparação entre algoritmos. ``plot_rewards.py`` compara as recompensas.

Todo treino do fluxo padrão grava, além do modelo, ``monitor.csv`` (recompensa
crua por episódio de agente), ``training_metrics.csv`` (métricas de tráfego por
episódio durante o treino -- comparáveis entre recompensas) e
``run_config.json`` (configuração completa da execução).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
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
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from sumo_rl.environment.traffic_signal import TrafficSignal


REPO_ROOT = Path(sumo_rl.__file__).resolve().parent.parent

# Recompensas registradas no SUMO-RL (diff-waiting-time, average-speed, queue,
# pressure, co2, ...). Lido do próprio registro para acompanhar novas funções.
REWARD_FNS = sorted(TrafficSignal.reward_fns)
DEFAULT_REWARD = "diff-waiting-time"

EVAL_METRICS = [
    ("reward", "Recompensa"),
    ("waiting_time", "Tempo médio de espera (s)"),
    ("queue", "Fila média (veículos parados)"),
    ("mean_speed", "Velocidade média (m/s)"),
    ("throughput", "Throughput (veículos concluídos)"),
    # As quatro abaixo vêm do TrafficProbe (lidas direto do SUMO). Avaliações
    # antigas não têm essas colunas -- os gráficos usam só as que existirem.
    ("wait_p95", "Espera acumulada por veículo, p95 (s)"),
    ("wait_max", "Maior espera acumulada de um veículo (s)"),
    ("co2_kg", "CO₂ estimado (kg por episódio)"),
    ("phase_switches", "Trocas de fase (por episódio)"),
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
    reward_fn: str = DEFAULT_REWARD
    norm_reward: bool = False
    # Subpasta entre o grid e o algoritmo. Vazia = comparação entre algoritmos
    # (outputs/<grid>/<algo>/); o estudo de recompensas usa "rewards/<nome>".
    results_subdir: str = ""

    @property
    def output_dir(self) -> Path:
        return REPO_ROOT / "outputs" / self.grid_slug / self.results_subdir / self.short

    @property
    def context_label(self) -> str:
        """Grid + recompensa, para títulos de gráficos."""
        if not self.results_subdir:
            return self.grid_label
        suffix = "" if self.norm_reward else ", sem normalização"
        return f"{self.grid_label} — recompensa {self.reward_fn}{suffix}"

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
        reward_fn=spec.reward_fn,
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
    env = VecMonitor(env, filename=str(run_dir / "monitor.csv"))
    if spec.norm_reward:
        # Normaliza só a recompensa (pelo desvio do retorno descontado), para que
        # funções com escalas muito diferentes -- diff-waiting-time ~0,01 por
        # passo, co2 na casa dos milhares -- não mudem o peso da perda do crítico
        # dentro do corte de gradiente (max_grad_norm). Fica DEPOIS do VecMonitor,
        # então monitor.csv continua com a recompensa crua. As observações não
        # são normalizadas: o modelo salvo roda na avaliação sem as estatísticas
        # do VecNormalize.
        env = VecNormalize(env, norm_obs=False, norm_reward=True, gamma=spec.gamma)
    return env


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


TRAINING_METRICS = ("waiting_time", "queue", "mean_speed", "throughput")


class TrafficMetricsCallback(BaseCallback):
    """Grava métricas de TRÁFEGO por episódio durante o treino em
    ``training_metrics.csv`` -- ao contrário da recompensa (cada função na sua
    escala), estas são comparáveis entre recompensas e entre algoritmos.

    As métricas são as mesmas da avaliação (``run_episode``), lidas do info do
    SUMO-RL: média por passo de ``system_mean_waiting_time``,
    ``system_total_stopped`` e ``system_mean_speed``, e o ``system_total_arrived``
    final. Diferença: aqui a política é a estocástica do treino e o tráfego vem
    de ``sumo_seed="random"``.

    Detalhes do vec env do SuperSuit:

    - o ``concat_vec_envs_v1`` organiza os sub-ambientes em blocos contíguos por
      cópia do SUMO (índice = cópia * agentes + agente), e todos os agentes de
      uma cópia recebem o mesmo info de sistema -- basta ler o primeiro de cada
      bloco;
    - no passo em que o episódio termina, o ``MarkovVectorEnv`` reinicia a cópia
      e mescla o info do reset POR CIMA do info do último passo. Esse info é
      descartado e o episódio é fechado com o que foi acumulado até ali (o
      throughput deixa de contar os últimos ``delta_time`` segundos, desprezível
      num episódio de 3600 s).
    """

    COLUMNS = ("timesteps", "copy", "episode", *TRAINING_METRICS)

    def __init__(self, path: Path, n_copies: int):
        super().__init__()
        self.path = Path(path)
        self.n_copies = n_copies

    def _on_training_start(self) -> None:
        num_envs = self.training_env.num_envs
        if num_envs % self.n_copies:
            raise ValueError(f"num_envs={num_envs} não é múltiplo do número de cópias ({self.n_copies})")
        self.stride = num_envs // self.n_copies
        self.buffers = [self._empty() for _ in range(self.n_copies)]
        self.episodes = [0] * self.n_copies
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="") as f:
            csv.writer(f).writerow(self.COLUMNS)

    @staticmethod
    def _empty() -> dict:
        return {"waiting": [], "queue": [], "speed": [], "arrived": 0}

    def _on_step(self) -> bool:
        infos, dones = self.locals.get("infos"), self.locals.get("dones")
        if infos is None or dones is None:
            return True
        for c in range(self.n_copies):
            i = c * self.stride
            if dones[i]:
                self._close_episode(c)
                continue
            info = infos[i]
            if not isinstance(info, dict) or "system_mean_waiting_time" not in info:
                continue
            buf = self.buffers[c]
            buf["waiting"].append(float(info["system_mean_waiting_time"]))
            buf["queue"].append(float(info.get("system_total_stopped", 0.0)))
            buf["speed"].append(float(info.get("system_mean_speed", 0.0)))
            buf["arrived"] = max(buf["arrived"], int(info.get("system_total_arrived", 0)))
        return True

    def _close_episode(self, c: int) -> None:
        buf, self.buffers[c] = self.buffers[c], self._empty()
        if not buf["waiting"]:
            return
        self.episodes[c] += 1
        row = (
            self.num_timesteps,
            c,
            self.episodes[c],
            float(np.mean(buf["waiting"])),
            float(np.mean(buf["queue"])),
            float(np.mean(buf["speed"])),
            buf["arrived"],
        )
        # Abre/fecha a cada episódio (raro: ~30 por cópia em 2M passos) para não
        # deixar arquivo aberto se o treino for interrompido.
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(row)


def _git_revision() -> str | None:
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except Exception:
        return None


# Hiperparâmetros lidos do modelo SB3 (os que não existirem no algoritmo são pulados).
_MODEL_HPARAMS = (
    "learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "clip_range",
    "clip_range_vf", "normalize_advantage", "ent_coef", "vf_coef", "max_grad_norm", "target_kl",
    "buffer_size", "learning_starts", "train_freq", "gradient_steps", "target_update_interval",
    "exploration_fraction", "exploration_initial_eps", "exploration_final_eps", "tau", "policy_kwargs",
)


def save_run_config(spec: ExperimentSpec, model, seed: int, run_dir: Path) -> None:
    """Grava run_config.json: tudo o que define a execução, para o relatório."""
    hparams = {}
    for name in _MODEL_HPARAMS:
        if not hasattr(model, name):
            continue
        value = getattr(model, name)
        if callable(value):  # schedules do SB3 (ex.: clip_range) -> valor inicial
            try:
                value = value(1.0)
            except Exception:
                continue
        hparams[name] = value
    config = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "git": _git_revision(),
        "seed": seed,
        "model_class": type(model).__name__,
        "n_envs": getattr(model, "n_envs", None),
        "spec": dataclasses.asdict(spec),
        "model_hparams": hparams,
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False, default=str))


# --------------------------------------------------------------------------
# Treino (fluxo padrão -- MAPPO define o seu próprio, mais específico)
# --------------------------------------------------------------------------
def default_train(spec: ExperimentSpec, build_model: Callable, seed: int, denom: int | None = None) -> None:
    run_dir = spec.output_dir / f"seed_{seed}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    print(f"{spec.algo} — SUMO-RL — {spec.context_label} — semente {seed} — {spec.total_timesteps:,} passos")

    # Semente aplicada manualmente: o ProcConcatVec do SuperSuit não implementa
    # env.seed(), então passar seed= ao modelo quebraria.
    set_random_seed(seed)

    env = make_training_env(spec, run_dir)
    try:
        model = build_model(env, spec)
        save_run_config(spec, model, seed, run_dir)
        callbacks = CallbackList(
            [
                checkpoint_callback(spec, run_dir, denom=denom),
                TrafficMetricsCallback(run_dir / "training_metrics.csv", n_copies=spec.n_vec_envs),
            ]
        )
        model.learn(total_timesteps=spec.total_timesteps, callback=callbacks, progress_bar=True)
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


class TrafficProbe:
    """Métricas lidas direto do SUMO a cada decisão (a cada delta_time s), que o
    info do SUMO-RL não traz e que separam recompensas diferentes:

    - espera acumulada por veículo (p95 e máximo): uma recompensa pode baixar a
      média deixando uma aproximação esquecida -- só a cauda mostra isso. Usa
      getAccumulatedWaitingTime, limitado à janela de waiting_time_memory do
      ambiente (1000 s por padrão);
    - CO₂: soma das emissões instantâneas (mg/s) x delta_time -- estimativa
      amostrada, igual para todos os controladores;
    - trocas de fase: mudanças no estado verde de cada semáforo entre duas
      amostras; amostras em amarelo ("y"/"u") são ignoradas, para o tempo fixo
      (que troca fora das decisões) ser contado do mesmo jeito.
    """

    def __init__(self, base):
        self.base = base
        self.veh_wait: dict[str, float] = {}
        self.co2_mg = 0.0
        self.last_green: dict[str, str] = {}
        self.switches = 0

    def sample(self) -> None:
        sumo = self.base.sumo
        dt = self.base.delta_time
        veh = sumo.vehicle
        for v in veh.getIDList():
            self.veh_wait[v] = max(self.veh_wait.get(v, 0.0), veh.getAccumulatedWaitingTime(v))
            self.co2_mg += veh.getCO2Emission(v) * dt
        tl = sumo.trafficlight
        for ts in self.base.ts_ids:
            state = tl.getRedYellowGreenState(ts)
            if "y" in state or "u" in state:
                continue
            prev = self.last_green.get(ts)
            if prev is not None and state != prev:
                self.switches += 1
            self.last_green[ts] = state

    def results(self) -> dict:
        waits = np.fromiter(self.veh_wait.values(), dtype=float)
        return {
            "wait_p95": float(np.percentile(waits, 95)) if waits.size else 0.0,
            "wait_max": float(waits.max()) if waits.size else 0.0,
            "co2_kg": self.co2_mg / 1e6,
            "phase_switches": float(self.switches),
        }


def run_episode(env, seed: int, policy_fn) -> dict:
    obs, _ = env.reset(seed=seed)
    base = find_sumo_env(env)
    probe = TrafficProbe(base) if base is not None else None
    total_reward = 0.0
    waiting, queue, speed = [], [], []
    arrived = 0

    while env.agents:
        obs, rewards, _, _, infos = env.step(policy_fn(env, obs))
        total_reward += sum(float(r) for r in rewards.values())
        if probe is not None:
            # A simulação só fecha no próximo reset, então ler o SUMO depois do
            # último passo é seguro.
            probe.sample()

        dicts = [i for i in infos.values() if isinstance(i, dict)]
        if dicts:
            waiting.append(np.mean([d.get("system_mean_waiting_time", 0.0) for d in dicts]))
            queue.append(np.mean([d.get("system_total_stopped", 0.0) for d in dicts]))
            speed.append(np.mean([d.get("system_mean_speed", 0.0) for d in dicts]))
            # system_total_arrived é um contador acumulado do SUMO-RL — basta
            # pegar o valor final do episódio (ler getArrivedNumber() por passo
            # subestima, porque cada passo do agente avança delta_time segundos).
            arrived = max(arrived, max(d.get("system_total_arrived", 0) for d in dicts))

    out = {
        "reward": total_reward,
        "waiting_time": float(np.mean(waiting)) if waiting else 0.0,
        "queue": float(np.mean(queue)) if queue else 0.0,
        "mean_speed": float(np.mean(speed)) if speed else 0.0,
        "throughput": float(arrived),
    }
    if probe is not None:
        out.update(probe.results())
    return out


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


def evaluate(spec: ExperimentSpec, model_cls, policy_factory: Callable = None) -> pd.DataFrame:
    # policy_factory permite que um algoritmo com mais de uma rede (ex.: IPPO,
    # uma rede por semáforo) roteie cada observação para o modelo certo. Todo o
    # resto da avaliação -- seeds, baselines, métricas, gráficos -- é idêntico.
    policy_factory = policy_factory or model_policy
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
        policy = policy_factory(model)
        for i, s in enumerate(spec.eval_seeds):
            r = run_episode(env, s, policy)
            r.update(episode=i + 1, controller=spec.algo, train_seed=train_seed)
            rows.append(r)
        env.close()
        print(f"[{spec.algo}] semente de treino {train_seed}: avaliação concluída")

    df = pd.DataFrame(rows)
    # Recompensa de treino deste experimento. A coluna "reward" (de todos os
    # controladores, inclusive baselines) está na escala desta função.
    df["reward_fn"] = spec.reward_fn
    df["norm_reward"] = spec.norm_reward
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(spec.output_dir / "evaluation.csv", index=False)
    plot_evaluation(spec, df)
    plot_comparison(spec, df)

    cols = [c for c, _ in available_metrics(df)]
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
def available_metrics(df: pd.DataFrame) -> list[tuple[str, str]]:
    """EVAL_METRICS presentes em df (avaliações antigas não têm as do TrafficProbe)."""
    return [(c, label) for c, label in EVAL_METRICS if c in df.columns]


def metric_grid(n: int, cell: tuple[float, float] = (5.0, 4.0)):
    """Figura com uma célula por métrica, 3 por linha; células sobrando ficam ocultas."""
    ncols = 3
    nrows = max(1, math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell[0] * ncols, cell[1] * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax in axes[n:]:
        ax.axis("off")
    return fig, axes


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
    plt.title(f"{spec.algo} — Recompensa de treino — {spec.context_label}")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    spec.plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(spec.plots_dir / f"{spec.short}_training_reward.png", dpi=200)
    plt.close()


def plot_evaluation(spec: ExperimentSpec, df: pd.DataFrame) -> None:
    sub = df[df.controller == spec.algo]
    grouped = sub.groupby("episode")
    metrics = available_metrics(sub)
    fig, axes = metric_grid(len(metrics))
    for ax, (col, label) in zip(axes, metrics):
        mean = grouped[col].mean()
        std = grouped[col].std().fillna(0.0)
        ax.plot(mean.index, mean.values, marker="o")
        ax.fill_between(mean.index, mean - std, mean + std, alpha=0.2)
        ax.set_xlabel("Episódio de avaliação")
        ax.set_ylabel(label)
        ax.set_title(f"{spec.algo} — {label}")
        ax.grid(alpha=0.25)
    fig.suptitle(f"{spec.algo} — Métricas de avaliação — {spec.context_label}", fontsize=15)
    plt.tight_layout()
    spec.plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(spec.plots_dir / f"{spec.short}_evaluation_metrics.png", dpi=200)
    plt.close()


def plot_comparison(spec: ExperimentSpec, df: pd.DataFrame) -> None:
    order = ["Fixed-Time", "Max-Pressure", spec.algo]
    metrics = available_metrics(df)
    summary = df.groupby("controller")[[c for c, _ in metrics]].mean().reindex(order)
    fig, axes = metric_grid(len(metrics), cell=(4.7, 4.0))
    for ax, (col, label) in zip(axes, metrics):
        ax.bar(summary.index, summary[col])
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"Baselines vs {spec.algo} — {spec.context_label}", fontsize=15)
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


def render(spec: ExperimentSpec, model_cls, seed: int, policy_factory: Callable = None) -> None:
    policy_factory = policy_factory or model_policy
    sd = spec.output_dir / f"seed_{seed}"
    if not (sd / f"{spec.model_slug}.zip").exists():
        found = sorted(spec.output_dir.glob("seed_*"))
        if not found:
            sys.exit(f"Nenhum modelo treinado em {spec.output_dir}.")
        sd = found[0]
    model = load_model(model_cls, spec, sd)
    env = ss.agent_indicator_v0(make_env(spec, use_gui=True), type_only=False)
    obs, _ = env.reset(seed=spec.eval_seeds[0])
    policy = policy_factory(model)
    while env.agents:
        obs, _, _, _, _ = env.step(policy(env, obs))
    env.close()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_cli(description: str, reward_flag: bool = True) -> argparse.ArgumentParser:
    """reward_flag=False: o script já fixa a recompensa (arquivos
    train_<algo>_<grid>_reward_<nome>.py), então --reward não é oferecido."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--seed", type=int, default=42, help="semente da política; no --train grava em <saída>/seed_<N>/"
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=None, help="sobrescreve o total de passos de treino (smoke tests)"
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=None, help="sobrescreve o número de episódios de avaliação"
    )
    if reward_flag:
        parser.add_argument(
            "--reward",
            choices=REWARD_FNS,
            default=None,
            help="estudo de recompensas: treina/avalia com esta recompensa, normalizada, em "
            "outputs/<grid>/rewards/<recompensa>/<algo>/ (repita a opção no --evaluate e no --render)",
        )
        norm_help = "com --reward: desliga a normalização da recompensa (saída em rewards/<recompensa>-raw/)"
    else:
        norm_help = "ablação: desliga a normalização da recompensa (saída em rewards/<recompensa>-raw/)"
    parser.add_argument("--no-norm-reward", action="store_true", help=norm_help)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--show", action="store_true", help="ações aleatórias com GUI (sanity check)")
    group.add_argument("--train", action="store_true", help="treina o modelo (semente de --seed)")
    group.add_argument(
        "--evaluate", action="store_true", help="avalia todas as sementes treinadas vs tempo fixo e max-pressure"
    )
    group.add_argument("--render", action="store_true", help="roda o modelo treinado com GUI")
    return parser


def configure_reward_study(spec: ExperimentSpec, reward_fn: str, normalize: bool = True) -> None:
    """Troca a recompensa de treino e isola a saída em
    outputs/<grid>/rewards/<recompensa>[-raw]/<algo>/. A comparação entre
    algoritmos (recompensa padrão, sem normalização, em outputs/<grid>/<algo>/)
    não é afetada."""
    spec.reward_fn = reward_fn
    spec.norm_reward = normalize
    spec.results_subdir = f"rewards/{reward_fn}" + ("" if normalize else "-raw")


def run_cli(
    spec: ExperimentSpec,
    model_cls,
    train_fn: Callable[[ExperimentSpec, int], None],
    policy_factory: Callable = None,
    reward: str | None = None,
) -> None:
    """reward: fixa a recompensa de treino deste script (estudo de recompensas,
    saída em outputs/<grid>/rewards/<reward>/<algo>/). Sem ela, o script é o da
    comparação entre algoritmos e aceita --reward na linha de comando."""
    if reward is not None and reward not in REWARD_FNS:
        sys.exit(f"Recompensa '{reward}' não registrada no SUMO-RL. Disponíveis: {', '.join(REWARD_FNS)}")

    description = f"{spec.algo} + SUMO-RL — {spec.grid_label}"
    if reward is not None:
        description += f" — recompensa {reward}"
    parser = build_cli(description, reward_flag=reward is None)
    args = parser.parse_args()

    if args.total_timesteps is not None:
        spec.total_timesteps = args.total_timesteps
    if args.eval_episodes is not None:
        spec.eval_episodes = args.eval_episodes
    chosen = reward if reward is not None else getattr(args, "reward", None)
    if chosen is not None:
        configure_reward_study(spec, chosen, normalize=not args.no_norm_reward)
    elif args.no_norm_reward:
        parser.error("--no-norm-reward só vale junto com --reward")

    if args.show:
        run_random(spec)
    elif args.train:
        train_fn(spec, args.seed)
    elif args.evaluate:
        evaluate(spec, model_cls, policy_factory=policy_factory)
    elif args.render:
        render(spec, model_cls, args.seed, policy_factory=policy_factory)
