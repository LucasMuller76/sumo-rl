"""
PPO multiagente (parameter sharing) — grid 4x4 — recompensa `diff-waiting-time`

Recompensa (SUMO-RL, padrão da biblioteca):

    r_t = W_{t-1} - W_t,   W_t = (espera acumulada dos veículos nas faixas de entrada) / 100

Positiva quando a espera acumulada na interseção diminui entre duas decisões,
negativa quando aumenta. A espera é contabilizada por faixa (um veículo que segue
para outra interseção não leva a espera junto). Escala típica: centésimos a
unidades por passo. Incentiva escoar quem está esperando há mais tempo.

É a mesma recompensa da comparação entre algoritmos (train_ppo_4x4.py), mas lá
SEM normalização. Por isso este é o braço de referência do estudo: os Δ% de
plot_rewards.py são calculados contra ele.

Configuração — idêntica em todos os train_ppo_4x4_reward_*.py; só REWARD muda:
PPO multiagente com parameter sharing (16 semáforos, uma rede compartilhada, a
identidade do agente na observação). Rede, hiperparâmetros, cenário (RESCO
grid4x4), 2M passos e 6 cópias paralelas do SUMO vêm de train_ppo_4x4.py; o
protocolo de avaliação (sementes, baselines, métricas) vem de _common.py. A
recompensa é normalizada com VecNormalize, para que escalas diferentes não mudem
o equilíbrio ator/crítico; o monitor.csv guarda a recompensa crua.

    python train_ppo_4x4_reward_diff_waiting_time.py --train [--seed N]   # outputs/4x4/rewards/diff-waiting-time/ppo/seed_N/
    python train_ppo_4x4_reward_diff_waiting_time.py --evaluate           # todas as sementes vs Fixed-Time e Max-Pressure
    python train_ppo_4x4_reward_diff_waiting_time.py --render [--seed N]  # modelo treinado com GUI
    python plot_rewards.py --grid 4x4 --algo ppo                           # comparação entre as recompensas

Estudo completo (todas as recompensas e sementes): bash run_reward_study_ppo_4x4.sh
"""

from __future__ import annotations

import dataclasses

from stable_baselines3 import PPO

import _common as common
import train_ppo_4x4 as base

REWARD = "diff-waiting-time"

if __name__ == "__main__":
    common.run_cli(
        dataclasses.replace(base.SPEC), PPO, common.make_default_train_fn(base.build_model), reward=REWARD
    )
