"""
MAPPO + SUMO-RL — grid 4x4

MAPPO (Yu et al., 2021): treino centralizado, execução descentralizada (CTDE).

- Ator: MLP sobre a observação local do próprio semáforo + identidade do agente
  (parameter sharing). É o único braço usado em --evaluate e --render.
- Crítico: durante o TREINO vê o estado GLOBAL — a concatenação das observações
  de todos os semáforos da mesma cópia do ambiente, no mesmo instante.

O Stable-Baselines3 não suporta crítico centralizado nativamente, então este
arquivo estende PPO com um RolloutBuffer e uma policy próprios. O resto
(clipping, GAE, minibatches) é o PPO padrão. Ambiente, baselines, avaliação,
gráficos e CLI vêm de `_common.py` — os mesmos usados por DQN/PPO/QR-DQN/A2C,
para que os resultados sejam diretamente comparáveis.

    python train_mappo_4x4.py --show                 # ações aleatórias com GUI (sanity check)
    python train_mappo_4x4.py --train [--seed N]     # treina uma semente (outputs/4x4/mappo/seed_N/)
    python train_mappo_4x4.py --evaluate             # avalia todas as sementes vs baselines
    python train_mappo_4x4.py --render [--seed N]    # roda o modelo treinado com GUI
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym
import sumo_rl
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import explained_variance, obs_as_tensor, set_random_seed

import _common as common

# --------------------------------------------------------------------------
# Especificação do experimento
# --------------------------------------------------------------------------
NUM_AGENTS = 16  # interseções semaforizadas (usado para montar o estado global)

_NETS = Path(sumo_rl.__file__).resolve().parent / "nets"

SPEC = common.ExperimentSpec(
    algo="MAPPO",
    short="mappo",
    grid_slug="4x4",
    grid_label="grid 4x4",
    model_slug="mappo_4x4grid",
    net_file=_NETS / "RESCO" / "grid4x4" / "grid4x4.net.xml",
    route_file=_NETS / "RESCO" / "grid4x4" / "grid4x4_1.rou.xml",
    # 16 agentes (vs 4 no 2x2) precisam de mais passos pra convergir; validado
    # empiricamente -- com 500k (o default do 2x2) o explained_variance do
    # crítico ficava perto de zero, sinal de sub-treino.
    total_timesteps=2_000_000,
)

# Hiperparâmetros iguais ao PPO — a única diferença deve ser o crítico centralizado.
LEARNING_RATE = 3e-4
N_STEPS = 1024
BATCH_SIZE = 256
N_EPOCHS = 10
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5


# --------------------------------------------------------------------------
# Estado global (crítico centralizado)
# --------------------------------------------------------------------------
def compute_global_state(obs: np.ndarray, num_agents: int) -> np.ndarray:
    """obs: (n_total, obs_dim), n_total = N_VEC_ENVS * num_agents, em blocos
    contíguos de num_agents (organização do concat_vec_envs_v1). O estado
    global de cada linha é a concatenação das observações de todos os agentes
    da mesma cópia do ambiente."""
    n_total, obs_dim = obs.shape
    grouped = obs.reshape(n_total // num_agents, num_agents * obs_dim)
    return np.repeat(grouped, num_agents, axis=0)


class CentralizedRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    global_states: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class CentralizedRolloutBuffer(RolloutBuffer):
    """RolloutBuffer do SB3 + um array paralelo de estados globais, embaralhado
    junto com o resto em get() (mesmos índices)."""

    def __init__(self, *args, global_state_dim: int, **kwargs):
        self.global_state_dim = global_state_dim
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        super().reset()
        self.global_states = np.zeros(
            (self.buffer_size, self.n_envs, self.global_state_dim), dtype=np.float32
        )

    def add(self, obs, global_state, action, reward, episode_start, value, log_prob) -> None:
        self.global_states[self.pos] = np.array(global_state)
        super().add(obs, action, reward, episode_start, value, log_prob)

    def get(self, batch_size=None):
        assert self.full
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        if not self.generator_ready:
            names = ["observations", "global_states", "actions", "values", "log_probs", "advantages", "returns"]
            for name in names:
                self.__dict__[name] = self.swap_and_flatten(self.__dict__[name])
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start = 0
        while start < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start:start + batch_size])
            start += batch_size

    def _get_samples(self, batch_inds, env=None):
        data = (
            self.observations[batch_inds],
            self.global_states[batch_inds],
            self.actions[batch_inds].astype(np.float32, copy=False),
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return CentralizedRolloutBufferSamples(*tuple(map(self.to_torch, data)))


class CentralizedActorCriticPolicy(ActorCriticPolicy):
    """Ator (MLP sobre obs local) + crítico (MLP separado sobre o estado
    global). Sem o mlp_extractor compartilhado do SB3, porque os dois braços
    têm entradas de dimensões diferentes."""

    def __init__(self, observation_space, action_space, lr_schedule, global_state_dim, net_arch=None, **kwargs):
        self.global_state_dim = global_state_dim
        super().__init__(observation_space, action_space, lr_schedule, net_arch=net_arch, **kwargs)

    def _build_mlp_extractor(self) -> None:
        pass  # substituído por actor_net / critic_net em _build()

    @staticmethod
    def _make_mlp(input_dim: int, hidden: list) -> nn.Sequential:
        layers, last = [], input_dim
        for size in hidden:
            layers += [nn.Linear(last, size), nn.Tanh()]
            last = size
        return nn.Sequential(*layers)

    def _build(self, lr_schedule) -> None:
        hidden = self.net_arch if isinstance(self.net_arch, list) else self.net_arch.get("pi", [256, 256])
        obs_dim = int(np.prod(self.observation_space.shape))

        self.actor_net = self._make_mlp(obs_dim, hidden)
        self.critic_net = self._make_mlp(self.global_state_dim, hidden)
        latent_dim = hidden[-1] if hidden else obs_dim

        self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim)
        self.value_net = nn.Linear(latent_dim, 1)

        if self.ortho_init:
            for module, gain in [
                (self.actor_net, np.sqrt(2)),
                (self.critic_net, np.sqrt(2)),
                (self.action_net, 0.01),
                (self.value_net, 1),
            ]:
                module.apply(partial(self.init_weights, gain=gain))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _actor_latent(self, obs: th.Tensor) -> th.Tensor:
        return self.actor_net(obs.float())

    def _critic_latent(self, global_state: th.Tensor) -> th.Tensor:
        return self.critic_net(global_state.float())

    def forward(self, obs: th.Tensor, global_state: th.Tensor, deterministic: bool = False):
        latent_pi = self._actor_latent(obs)
        latent_vf = self._critic_latent(global_state)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def evaluate_actions(self, obs: th.Tensor, global_state: th.Tensor, actions: th.Tensor):
        latent_pi = self._actor_latent(obs)
        latent_vf = self._critic_latent(global_state)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: th.Tensor):
        # Usado por model.predict(): só o ator, sem estado global (execução descentralizada).
        latent_pi = self._actor_latent(obs)
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, global_state: th.Tensor) -> th.Tensor:
        latent_vf = self._critic_latent(global_state)
        return self.value_net(latent_vf)


class MAPPO(PPO):
    """PPO com crítico centralizado (CTDE). collect_rollouts() e train() são
    reescritos para carregar o estado global junto de cada transição."""

    def __init__(self, *args, num_agents: int = NUM_AGENTS, **kwargs):
        self.num_agents = num_agents
        super().__init__(*args, **kwargs)

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps) -> bool:
        assert self._last_obs is not None
        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                global_state_np = compute_global_state(self._last_obs, self.num_agents)
                global_state_tensor = obs_as_tensor(global_state_np, self.device)
                actions, values, log_probs = self.policy(obs_tensor, global_state_tensor)
            actions = actions.cpu().numpy()

            new_obs, rewards, dones, infos = env.step(actions)
            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1
            actions = actions.reshape(-1, 1)  # ação discreta

            rollout_buffer.add(
                self._last_obs, global_state_np, actions, rewards,
                self._last_episode_starts, values, log_probs,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            final_state = obs_as_tensor(compute_global_state(new_obs, self.num_agents), self.device)
            values = self.policy.predict_values(final_state)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = (
            self.clip_range_vf(self._current_progress_remaining)
            if self.clip_range_vf is not None
            else None
        )

        entropy_losses, pg_losses, value_losses, clip_fractions = [], [], [], []
        continue_training = True

        for _ in range(self.n_epochs):
            approx_kl_divs = []
            for rd in self.rollout_buffer.get(self.batch_size):
                actions = rd.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rd.observations, rd.global_states, actions
                )
                values = values.flatten()
                advantages = rd.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rd.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())
                clip_fractions.append(th.mean((th.abs(ratio - 1) > clip_range).float()).item())

                values_pred = (
                    values
                    if clip_range_vf is None
                    else rd.old_values + th.clamp(values - rd.old_values, -clip_range_vf, clip_range_vf)
                )
                value_loss = F.mse_loss(rd.returns, values_pred)
                value_losses.append(value_loss.item())

                entropy_loss = -th.mean(entropy) if entropy is not None else -th.mean(-log_prob)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with th.no_grad():
                    log_ratio = log_prob - rd.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)


# --------------------------------------------------------------------------
# Treino (específico: crítico centralizado precisa do estado global e de um
# denominador de checkpoint próprio; o resto vem de _common)
# --------------------------------------------------------------------------
def train(spec: common.ExperimentSpec, seed: int) -> None:
    run_dir = spec.output_dir / f"seed_{seed}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    print(f"{spec.algo} — SUMO-RL — {spec.grid_label} — semente {seed} — {spec.total_timesteps:,} passos")
    set_random_seed(seed)

    env = common.make_training_env(spec, run_dir)
    try:
        assert isinstance(env.action_space, gym.spaces.Discrete), (
            "Este MAPPO só cobre ações discretas (fases de semáforo)."
        )
        local_obs_dim = int(np.prod(env.observation_space.shape))
        global_state_dim = local_obs_dim * NUM_AGENTS

        model = MAPPO(
            policy=CentralizedActorCriticPolicy,
            env=env,
            num_agents=NUM_AGENTS,
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
            rollout_buffer_class=CentralizedRolloutBuffer,
            rollout_buffer_kwargs={"global_state_dim": global_state_dim},
            policy_kwargs={"global_state_dim": global_state_dim, "net_arch": spec.net_arch},
            verbose=1,
            device="cpu",
        )

        checkpoint = common.checkpoint_callback(spec, run_dir, denom=spec.n_vec_envs * NUM_AGENTS)
        model.learn(total_timesteps=spec.total_timesteps, callback=checkpoint, progress_bar=True)
        model.save(str(run_dir / spec.model_slug))
    finally:
        # Fecha os workers SUMO/SuperSuit mesmo se o treino falhar -- sem isso,
        # uma exceção aqui deixa subprocessos pendurados e o interpretador
        # trava na finalização (atexit do multiprocessing esperando por eles).
        env.close()

    common.plot_training_curve(spec)
    print(f"\nModelo salvo em {run_dir / spec.model_slug}.zip")


if __name__ == "__main__":
    common.run_cli(SPEC, MAPPO, train)
