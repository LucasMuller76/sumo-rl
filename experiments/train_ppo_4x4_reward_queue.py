"""
PPO multiagente (parameter sharing) — grid 4x4 — recompensa `queue`

Recompensa (SUMO-RL):

    r_t = -(número de veículos parados nas faixas de entrada)

"Parado" segue a definição do SUMO: velocidade < 0,1 m/s. A recompensa é sempre
≤ 0, de zero a algumas dezenas negativas por interseção. Incentiva diretamente
filas curtas. Não olha há quanto tempo cada veículo está parado, então pode
segurar uma aproximação com pouca demanda por muito tempo; o wait_p95/wait_max
da avaliação mostra se isso acontece.

Configuração — idêntica em todos os train_ppo_4x4_reward_*.py; só REWARD muda:
PPO multiagente com parameter sharing (16 semáforos, uma rede compartilhada, a
identidade do agente na observação). Rede, hiperparâmetros, cenário (RESCO
grid4x4), 2M passos e 6 cópias paralelas do SUMO vêm de train_ppo_4x4.py; o
protocolo de avaliação (sementes, baselines, métricas) vem de _common.py. A
recompensa é normalizada com VecNormalize, para que escalas diferentes não mudem
o equilíbrio ator/crítico; o monitor.csv guarda a recompensa crua.

    python train_ppo_4x4_reward_queue.py --train [--seed N]   # outputs/4x4/rewards/queue/ppo/seed_N/
    python train_ppo_4x4_reward_queue.py --evaluate           # todas as sementes vs Fixed-Time e Max-Pressure
    python train_ppo_4x4_reward_queue.py --render [--seed N]  # modelo treinado com GUI
    python plot_rewards.py --grid 4x4 --algo ppo               # comparação entre as recompensas

Estudo completo (todas as recompensas e sementes): bash run_reward_study_ppo_4x4.sh
"""

from __future__ import annotations

import dataclasses

from stable_baselines3 import PPO

import _common as common
import train_ppo_4x4 as base

REWARD = "queue"

if __name__ == "__main__":
    common.run_cli(
        dataclasses.replace(base.SPEC), PPO, common.make_default_train_fn(base.build_model), reward=REWARD
    )
