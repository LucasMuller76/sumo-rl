"""
PPO multiagente (parameter sharing) — grid 4x4 — recompensa `pressure`

Recompensa (SUMO-RL):

    r_t = (veículos nas faixas de saída) - (veículos nas faixas de entrada)

Conta veículos em movimento e parados, e a recompensa é maximizada. Incentiva
esvaziar as faixas de entrada.

Atenção: NÃO é a "pressure" da literatura de max-pressure/PressLight, que é
-|entrada - saída| calculada por movimento. A versão do SUMO-RL é um saldo
agregado da interseção e não penaliza acumular veículos na saída, que no grid é a
entrada do vizinho. Vale olhar o efeito disso na fila e na espera da rede inteira.
O baseline Max-Pressure da avaliação usa a definição clássica, por fase.

Configuração — idêntica em todos os train_ppo_4x4_reward_*.py; só REWARD muda:
PPO multiagente com parameter sharing (16 semáforos, uma rede compartilhada, a
identidade do agente na observação). Rede, hiperparâmetros, cenário (RESCO
grid4x4), 2M passos e 6 cópias paralelas do SUMO vêm de train_ppo_4x4.py; o
protocolo de avaliação (sementes, baselines, métricas) vem de _common.py. A
recompensa é normalizada com VecNormalize, para que escalas diferentes não mudem
o equilíbrio ator/crítico; o monitor.csv guarda a recompensa crua.

    python train_ppo_4x4_reward_pressure.py --train [--seed N]   # outputs/4x4/rewards/pressure/ppo/seed_N/
    python train_ppo_4x4_reward_pressure.py --evaluate           # todas as sementes vs Fixed-Time e Max-Pressure
    python train_ppo_4x4_reward_pressure.py --render [--seed N]  # modelo treinado com GUI
    python plot_rewards.py --grid 4x4 --algo ppo                  # comparação entre as recompensas

Estudo completo (todas as recompensas e sementes): bash run_reward_study_ppo_4x4.sh
"""

from __future__ import annotations

import dataclasses

from stable_baselines3 import PPO

import _common as common
import train_ppo_4x4 as base

REWARD = "pressure"

if __name__ == "__main__":
    common.run_cli(
        dataclasses.replace(base.SPEC), PPO, common.make_default_train_fn(base.build_model), reward=REWARD
    )
