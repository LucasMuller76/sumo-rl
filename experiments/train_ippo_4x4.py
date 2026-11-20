"""
IPPO + SUMO-RL — grid 4x4

IPPO (Independent PPO) com redes totalmente independentes: cada semáforo tem o
seu PRÓPRIO PPO (ator + crítico), treinado apenas com as próprias observações e
recompensas. Não há parameter sharing e não há crítico centralizado.

Diferença para os outros dois experimentos da mesma família:

    IPPO  (este)  -> N redes separadas,        crítico descentralizado
    PPO           -> 1 rede compartilhada,     crítico descentralizado
    MAPPO         -> 1 rede compartilhada,     crítico CENTRALIZADO (vê o estado global)

Ou seja: IPPO vs PPO isola o efeito do parameter sharing, e PPO vs MAPPO isola o
efeito do crítico centralizado.

Hiperparâmetros, ambiente, wrappers, seeds, métricas e gráficos são idênticos aos
do train_ppo_4x4.py -- a única variável que muda é "uma rede por agente" vs "uma
rede para todos". O orçamento (total_timesteps) conta interações com o ambiente,
então IPPO e PPO consomem exatamente a mesma quantidade de simulação; a diferença
é que aqui cada rede vê 1/N desses dados (que é justamente o custo do não-sharing).

O Stable-Baselines3 não treina N políticas num único ambiente multiagente, então
a coleta de rollouts é feita à mão aqui; o resto (GAE, clipping, minibatches) é o
PPO padrão do SB3, chamado por agente.

    python train_ippo_4x4.py --show                 # ações aleatórias com GUI (sanity check)
    python train_ippo_4x4.py --train [--seed N]     # treina uma semente (outputs/4x4/ippo/seed_N/)
    python train_ippo_4x4.py --evaluate             # avalia todas as sementes vs baselines
    python train_ippo_4x4.py --render [--seed N]    # roda o modelo treinado com GUI

Para um estudo com várias sementes: rode --train com --seed 0,1,2,3,4 e depois
--evaluate uma vez (agrega todas as sementes). Para comparar com os outros
algoritmos do mesmo grid, rode `python plot_compare.py --grid 4x4` depois de
avaliar cada um.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch as th
from tqdm import tqdm

import gymnasium as gym
import sumo_rl
from stable_baselines3 import PPO
from stable_baselines3.common.utils import configure_logger, explained_variance, obs_as_tensor, set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

import _common as common

# --------------------------------------------------------------------------
# Especificação do experimento
# --------------------------------------------------------------------------
_NETS = Path(sumo_rl.__file__).resolve().parent / "nets"

NUM_AGENTS = 16  # interseções semaforizadas (uma rede PPO para cada)

SPEC = common.ExperimentSpec(
    algo="IPPO",
    short="ippo",
    grid_slug="4x4",
    grid_label="grid 4x4",
    model_slug="ippo_4x4grid",
    net_file=_NETS / "RESCO" / "grid4x4" / "grid4x4.net.xml",
    route_file=_NETS / "RESCO" / "grid4x4" / "grid4x4_1.rou.xml",
    # 16 agentes (vs 4 no 2x2) precisam de mais passos pra convergir; validado
    # empiricamente -- com 500k (o default do 2x2) o explained_variance do
    # crítico ficava perto de zero, sinal de sub-treino.
    total_timesteps=2_000_000,
)

# Iguais ao train_ppo_4x4.py -- a única diferença deve ser o não-sharing.
LEARNING_RATE = 3e-4
N_STEPS = 1024
BATCH_SIZE = 256
N_EPOCHS = 10
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

CHECKPOINT_EVERY = 500_000  # passos de ambiente entre checkpoints


# --------------------------------------------------------------------------
# Uma rede PPO por agente
# --------------------------------------------------------------------------
class _SoloEnv(gym.Env):
    """Ambiente-fantasma: existe só para o SB3 inferir espaços e n_envs ao
    construir cada PPO. A coleta real acontece no SUMO, em train()."""

    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def _blank(self):
        return np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)

    def reset(self, *, seed=None, options=None):
        return self._blank(), {}

    def step(self, action):
        return self._blank(), 0.0, False, False, {}


def build_model(obs_space, action_space, spec: common.ExperimentSpec) -> PPO:
    """PPO de um agente. n_envs = n_vec_envs porque cada agente aparece uma vez
    em cada cópia paralela do ambiente."""
    dummy = DummyVecEnv([lambda: _SoloEnv(obs_space, action_space) for _ in range(spec.n_vec_envs)])
    return PPO(
        "MlpPolicy",
        dummy,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=spec.gamma,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        policy_kwargs={"net_arch": spec.net_arch},
        verbose=0,
        device="cpu",
    )


class IPPOEnsemble:
    """As N políticas independentes, salvas/carregadas como um único .zip para
    caber na mesma convenção de arquivo dos outros algoritmos."""

    def __init__(self, models, agent_ids):
        self.models = list(models)
        self.agent_ids = list(agent_ids)
        self._index = {a: i for i, a in enumerate(self.agent_ids)}

    @staticmethod
    def _as_zip(path) -> Path:
        path = Path(str(path))
        return path if path.suffix == ".zip" else path.with_suffix(".zip")

    def save(self, path) -> None:
        path = self._as_zip(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w") as bundle:
            bundle.writestr("agents.json", json.dumps(self.agent_ids))
            for i, model in enumerate(self.models):
                buf = io.BytesIO()
                model.save(buf)
                bundle.writestr(f"agent_{i}.zip", buf.getvalue())

    @classmethod
    def load(cls, path, device="cpu"):
        path = cls._as_zip(path)
        with zipfile.ZipFile(path, "r") as bundle:
            agent_ids = json.loads(bundle.read("agents.json"))
            models = [PPO.load(io.BytesIO(bundle.read(f"agent_{i}.zip")), device=device) for i in range(len(agent_ids))]
        return cls(models, agent_ids)

    def model_for(self, env, agent_id) -> PPO:
        """Resolve qual rede controla este semáforo. Usa os ids salvos no treino;
        se o ambiente expuser outra ordem, cai para a ordem do próprio ambiente."""
        if agent_id in self._index:
            return self.models[self._index[agent_id]]
        return self.models[list(env.possible_agents).index(agent_id)]


def ippo_policy(ensemble: IPPOEnsemble):
    """policy_factory do _common: roteia a observação de cada semáforo para a
    rede correspondente (o resto da avaliação é idêntico aos outros algoritmos)."""

    def policy(env, obs):
        actions = {}
        for a in env.agents:
            model = ensemble.model_for(env, a)
            actions[a] = int(np.asarray(model.predict(obs[a], deterministic=True)[0]).item())
        return actions

    return policy


# --------------------------------------------------------------------------
# Treino
# --------------------------------------------------------------------------
def _agent_ids(spec: common.ExperimentSpec) -> list[str]:
    """Ordem dos agentes usada pelo vec env (blocos contíguos por cópia)."""
    probe = common.make_env(spec)
    try:
        return list(probe.possible_agents)
    finally:
        probe.close()


def train(spec: common.ExperimentSpec, seed: int) -> None:
    run_dir = spec.output_dir / f"seed_{seed}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    print(f"{spec.algo} — SUMO-RL — {spec.grid_label} — semente {seed} — {spec.total_timesteps:,} passos")
    print(f"{NUM_AGENTS} redes independentes (sem parameter sharing)")

    set_random_seed(seed)
    agent_ids = _agent_ids(spec)
    if len(agent_ids) != NUM_AGENTS:
        raise RuntimeError(f"Esperava {NUM_AGENTS} semáforos, o ambiente tem {len(agent_ids)}: {agent_ids}")

    env = common.make_training_env(spec, run_dir)
    try:
        n_vec = spec.n_vec_envs
        if env.num_envs != n_vec * NUM_AGENTS:
            raise RuntimeError(f"Esperava {n_vec * NUM_AGENTS} sub-ambientes, veio {env.num_envs}")

        # Linhas do vetor achatado que pertencem a cada agente. O concat_vec_envs_v1
        # organiza em blocos contíguos por cópia: índice = cópia * NUM_AGENTS + agente.
        rows = [np.arange(n_vec) * NUM_AGENTS + i for i in range(NUM_AGENTS)]

        models = [build_model(env.observation_space, env.action_space, spec) for _ in range(NUM_AGENTS)]
        for model in models:
            model.set_logger(configure_logger(verbose=0))

        obs = env.reset()
        episode_starts = np.ones(env.num_envs, dtype=bool)
        dones = np.zeros(env.num_envs, dtype=bool)
        num_timesteps = 0
        next_ckpt = CHECKPOINT_EVERY
        iteration = 0
        recent_returns: list[float] = []

        with tqdm(total=spec.total_timesteps, unit="step", dynamic_ncols=True) as bar:
            while num_timesteps < spec.total_timesteps:
                for model in models:
                    model.policy.set_training_mode(False)
                    model.rollout_buffer.reset()

                for _ in range(N_STEPS):
                    actions = np.zeros(env.num_envs, dtype=np.int64)
                    cache = []
                    for i, model in enumerate(models):
                        local_obs = obs[rows[i]]
                        with th.no_grad():
                            act, val, log_prob = model.policy(obs_as_tensor(local_obs, model.device))
                        act = act.cpu().numpy().reshape(-1)
                        actions[rows[i]] = act
                        cache.append((local_obs, act, val, log_prob))

                    new_obs, rewards, dones, infos = env.step(actions)
                    num_timesteps += env.num_envs
                    bar.update(env.num_envs)

                    for info in infos:
                        if isinstance(info, dict) and "episode" in info:
                            recent_returns.append(float(info["episode"]["r"]))

                    for i, model in enumerate(models):
                        local_obs, act, val, log_prob = cache[i]
                        idx = rows[i]
                        local_rewards = np.asarray(rewards[idx], dtype=np.float32).copy()

                        # Bootstrap em truncamento, igual ao collect_rollouts do SB3.
                        for k, j in enumerate(idx):
                            info = infos[j]
                            if (
                                dones[j]
                                and isinstance(info, dict)
                                and info.get("terminal_observation") is not None
                                and info.get("TimeLimit.truncated", False)
                            ):
                                terminal_obs = model.policy.obs_to_tensor(info["terminal_observation"])[0]
                                with th.no_grad():
                                    terminal_value = model.policy.predict_values(terminal_obs)[0]
                                local_rewards[k] += model.gamma * float(terminal_value.item())

                        model.rollout_buffer.add(
                            local_obs, act.reshape(-1, 1), local_rewards, episode_starts[idx], val, log_prob
                        )

                    obs = new_obs
                    episode_starts = dones

                iteration += 1
                progress_remaining = max(1.0 - num_timesteps / spec.total_timesteps, 0.0)
                ev_scores = []
                for i, model in enumerate(models):
                    with th.no_grad():
                        last_values = model.policy.predict_values(obs_as_tensor(obs[rows[i]], model.device))
                    model.rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=dones[rows[i]])
                    model._current_progress_remaining = progress_remaining
                    model.train()
                    ev_scores.append(
                        explained_variance(
                            model.rollout_buffer.values.flatten(), model.rollout_buffer.returns.flatten()
                        )
                    )

                mean_return = float(np.mean(recent_returns[-100:])) if recent_returns else float("nan")
                bar.write(
                    f"iter {iteration:>3} | passos {num_timesteps:>9,} | "
                    f"ep_rew_mean {mean_return:>8.4f} | explained_variance {np.mean(ev_scores):>6.3f} "
                    f"(min {np.min(ev_scores):>6.3f} / max {np.max(ev_scores):>6.3f})"
                )

                if num_timesteps >= next_ckpt:
                    IPPOEnsemble(models, agent_ids).save(
                        run_dir / "checkpoints" / f"{spec.short}_{num_timesteps}_steps.zip"
                    )
                    next_ckpt += CHECKPOINT_EVERY

        IPPOEnsemble(models, agent_ids).save(run_dir / spec.model_slug)
    finally:
        # Fecha os workers SUMO/SuperSuit mesmo se o treino falhar -- sem isso,
        # uma exceção aqui deixa subprocessos pendurados e o interpretador
        # trava na finalização (atexit do multiprocessing esperando por eles).
        env.close()

    common.plot_training_curve(spec)
    print(f"\nModelo salvo em {run_dir / spec.model_slug}.zip")


if __name__ == "__main__":
    common.run_cli(SPEC, IPPOEnsemble, train, policy_factory=ippo_policy)
