#!/usr/bin/env bash
# Estudo de recompensas — PPO multiagente (parameter sharing), grid 4x4.
#
# Treina cada recompensa em cada semente, avalia e gera a comparação geral
# (plot_rewards.py). Pode ser interrompido e rodado de novo: sementes já
# treinadas (com o .zip final) são puladas.
#
#   bash run_reward_study_ppo_4x4.sh                         # 3 sementes (0 1 2), todas as recompensas
#   SEEDS="0 1 2 3 4" bash run_reward_study_ppo_4x4.sh       # outras sementes
#   REWARDS="queue pressure" bash run_reward_study_ppo_4x4.sh
#   PYTHON=python3 bash run_reward_study_ppo_4x4.sh          # outro interpretador
#
# Cada treino de 2M passos leva ~37 min (6 cópias do SUMO em paralelo), ou seja,
# ~7,5 h para 4 recompensas x 3 sementes.

set -euo pipefail

cd "$(dirname "$0")"
REPO_ROOT="$(cd .. && pwd)"

SEEDS="${SEEDS:-0 1 2}"
REWARDS="${REWARDS:-diff_waiting_time average_speed queue pressure}"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then PYTHON="$REPO_ROOT/.venv/bin/python"; else PYTHON=python; fi
fi

for reward in $REWARDS; do
  script="train_ppo_4x4_reward_${reward}.py"
  [[ -f "$script" ]] || { echo "Arquivo não encontrado: $script" >&2; exit 1; }
  out_name="${reward//_/-}"  # diff_waiting_time -> diff-waiting-time (nome no SUMO-RL)
  for seed in $SEEDS; do
    model="$REPO_ROOT/outputs/4x4/rewards/$out_name/ppo/seed_$seed/ppo_4x4grid.zip"
    if [[ -f "$model" ]]; then
      echo ">>> $out_name / semente $seed: já treinado, pulando"
      continue
    fi
    echo ">>> $out_name / semente $seed: treinando ($(date '+%H:%M'))"
    "$PYTHON" "$script" --train --seed "$seed"
  done
  echo ">>> $out_name: avaliando todas as sementes"
  "$PYTHON" "$script" --evaluate
done

echo ">>> comparação geral"
"$PYTHON" plot_rewards.py --grid 4x4 --algo ppo
