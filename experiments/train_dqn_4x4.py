"""
DQN + SUMO-RL — grid 4x4

Um único DQN controla todos os semáforos do grid (parameter sharing).
O wrapper agent_indicator injeta a identidade do semáforo na observação,
para o modelo compartilhado saber qual interseção está controlando.

    python train_dqn_4x4.py --show                 # ações aleatórias com GUI (sanity check)
    python train_dqn_4x4.py --train [--seed N]     # treina uma semente (outputs/4x4/dqn/seed_N/)
    python train_dqn_4x4.py --evaluate             # avalia todas as sementes vs baselines
    python train_dqn_4x4.py --render [--seed N]    # roda o modelo treinado com GUI

Para um estudo com várias sementes: rode --train com --seed 0,1,2,3,4 e depois
--evaluate uma vez (agrega todas as sementes). Para comparar com os outros
algoritmos do mesmo grid, rode `python plot_compare.py --grid 4x4` depois de
avaliar cada um.
"""

from __future__ import annotations

from pathlib import Path

import sumo_rl
from stable_baselines3 import DQN

import _common as common

# --------------------------------------------------------------------------
# Especificação do experimento
# --------------------------------------------------------------------------
_NETS = Path(sumo_rl.__file__).resolve().parent / "nets"

SPEC = common.ExperimentSpec(
    algo="DQN",
    short="dqn",
    grid_slug="4x4",
    grid_label="grid 4x4",
    model_slug="dqn_4x4grid",
    net_file=_NETS / "RESCO" / "grid4x4" / "grid4x4.net.xml",
    route_file=_NETS / "RESCO" / "grid4x4" / "grid4x4_1.rou.xml",
    # 16 agentes (vs 4 no 2x2) precisam de mais passos pra convergir; validado
    # empiricamente -- com 500k (o default do 2x2) o explained_variance do
    # crítico ficava perto de zero, sinal de sub-treino.
    total_timesteps=2_000_000,
)


# --------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------
def build_model(env, spec: common.ExperimentSpec) -> DQN:
    return DQN(
        "MlpPolicy",
        env,
        learning_rate=5e-4,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=128,
        tau=1.0,
        gamma=spec.gamma,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=2_000,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.20,
        policy_kwargs={"net_arch": spec.net_arch},
        verbose=1,
        device="cpu",  # rede pequena: CPU é mais rápida que GPU aqui
    )


if __name__ == "__main__":
    common.run_cli(SPEC, DQN, common.make_default_train_fn(build_model))
