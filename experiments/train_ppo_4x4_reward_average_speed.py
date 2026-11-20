"""
PPO multiagente (parameter sharing) — grid 4x4 — recompensa `average-speed`

Recompensa (SUMO-RL):

    r_t = média, sobre os veículos nas faixas de entrada, de (velocidade / velocidade permitida)

Fica em [0, 1] e vale 1,0 quando não há veículos chegando. É sempre não negativa.
Incentiva manter o fluxo em movimento: veículos parados entram na média com 0.
Ponto de atenção: é uma média por veículo e não conta quantos veículos há, então
uma aproximação com fila longa pesa tanto quanto uma com poucos carros.

Configuração — idêntica em todos os train_ppo_4x4_reward_*.py; só REWARD muda:
PPO multiagente com parameter sharing (16 semáforos, uma rede compartilhada, a
identidade do agente na observação). Rede, hiperparâmetros, cenário (RESCO
grid4x4), 2M passos e 6 cópias paralelas do SUMO vêm de train_ppo_4x4.py; o
protocolo de avaliação (sementes, baselines, métricas) vem de _common.py. A
recompensa é normalizada com VecNormalize, para que escalas diferentes não mudem
o equilíbrio ator/crítico; o monitor.csv guarda a recompensa crua.

    python train_ppo_4x4_reward_average_speed.py --train [--seed N]   # outputs/4x4/rewards/average-speed/ppo/seed_N/
    python train_ppo_4x4_reward_average_speed.py --evaluate           # todas as sementes vs Fixed-Time e Max-Pressure
    python train_ppo_4x4_reward_average_speed.py --render [--seed N]  # modelo treinado com GUI
    python plot_rewards.py --grid 4x4 --algo ppo                       # comparação entre as recompensas

Estudo completo (todas as recompensas e sementes): bash run_reward_study_ppo_4x4.sh
"""

from __future__ import annotations

import dataclasses

from stable_baselines3 import PPO

import _common as common
import train_ppo_4x4 as base

REWARD = "average-speed"

if __name__ == "__main__":
    common.run_cli(
        dataclasses.replace(base.SPEC), PPO, common.make_default_train_fn(base.build_model), reward=REWARD
    )
