# Relatório — MAPPO (CTDE) para controle de semáforos: implementação e resultados no grid 2x2

**Data:** 2026-09-02 · **Repo:** sumo-rl @ `86f0fc4` · **Autor:** lucasmuller20082@gmail.com (com Claude Code)

---

## 1. Objetivo

Implementar **MAPPO de verdade** (Multi-Agent PPO com *Centralized Training + Decentralized
Execution*) para os grids 2x2 e 4x4 do sumo-rl e comparar contra o baseline **Fixed-Time**.

O que já existia no repositório **não** era MAPPO:

- `experiments/sb3_grid4x4.py` — PPO/SB3 com *parameter sharing* (crítico **descentralizado**).
- `experiments/ppo_4x4grid.py` — PPO/RLlib *shared policy*, `stop=100000` timesteps (~6 episódios).
- `experiments/ql_4x4grid*.py` — Q-learners tabulares independentes.

Nenhum script tinha *seeding*, checkpoints, env de avaliação separado ou comparação
automatizada com Fixed-Time. Tudo isso foi adicionado.

---

## 2. O que foi implementado

Pacote self-contained em **`experiments/mappo/`** (~1550 linhas, depende só de `torch`;
nada é importado pelo `sumo_rl` core).

| Arquivo | Papel |
|---|---|
| `config.py` | `MAPPOConfig` (todos os hiperparâmetros) + presets `SCENARIOS["2x2"|"4x4"]` (paths de rede/rota, `num_seconds`, orçamento). Seeds de avaliação held-out (`EVAL_SEED_BASE = 9_000_000`). |
| `networks.py` | `SharedActor` (MLP `π(a_i\|o_i)` compartilhado entre agentes), `CentralizedCritic` (MLP `V(s_i)`), init ortogonal, `RunningMeanStd`, `ValueNorm` (PopArt-lite). |
| `state.py` | Monta o **estado centralizado** `s_i = concat_j(o_j) ++ [escalares system_* opcionais] ++ onehot(i)`. Com `--critic local` degenera para `o_i` (= **IPPO**, ablação). |
| `buffer.py` | `RolloutBuffer [T,W,A,...]`, GAE **com bootstrap de truncation**. |
| `vec_sumo.py` | `SubprocSumoVecEnv` — N processos SUMO paralelos (`spawn`), backend `auto/libsumo/traci`. |
| `mappo.py` | `MAPPO`: `act` / `get_values` / `update` (PPO-clip + value-clip + entropia + adv-norm + KL early-stop opcional). |
| `eval.py` | Avaliação determinística + baseline Fixed-Time (`fixed_ts=True`), mesmos seeds. CLI `--fixed-time` / `--checkpoint`. |
| `utils.py` | Seeding, device, git hash + versões, `CsvLogger`, TensorBoard opcional, save/load de checkpoint (inclui estados de RNG). |
| `../train_mappo.py` | Entrypoint CLI: loop de coleta on-policy → GAE → update → log → avaliação periódica → checkpoint. |
| `../plot_mappo.py` | Gráficos (matplotlib puro, sem LaTeX): curvas de aprendizado com banda mean±std, barras MAPPO vs Fixed-Time, diagnóstico de treino. |
| `../../tests/test_mappo.py` | 4 testes: unitário do GAE-bootstrap, shape do estado, `update` finito, smoke end-to-end com SUMO. |

### Decisões técnicas principais

1. **CTDE.** Ator compartilhado decide a partir da observação local de cada semáforo
   (execução descentralizada). Crítico único e compartilhado enxerga o estado global —
   a concatenação de **todas** as observações locais + one-hot do agente (variante
   *agent-specific state* de Yu et al. 2021). Para o 2x2: `obs_dim = 21`, `state_dim = 88`.

2. **Bootstrap de time-limit.** O `SumoEnvironment` nunca emite estado terminal de verdade:
   todo episódio acaba por limite de tempo (`sim_step >= sim_max_time`). O buffer trata
   isso como *truncation* — no fim do episódio faz bootstrap com `V(s_T)` e corta o traço
   do GAE, em vez de zerar o retorno. (Testado em `test_gae_truncation_bootstrap`.)

3. **Driver do ambiente.** Usa `SumoEnvironment(single_agent=False)` direto (dicts de
   obs/reward/done), sem o wrapper PettingZoo AEC→parallel. Cada worker roda **um** SUMO
   em processo próprio via `multiprocessing spawn`, setando `LIBSUMO_AS_TRACI=1` **antes**
   de importar `sumo_rl` — `libsumo` dá ~8x por instância e permite só 1 sim por processo,
   que é exatamente o layout aqui. Fallback automático para `traci` se `libsumo` faltar.

4. **Estabilidade.** Value normalization (estatística corrida dos retornos), normalização
   de vantagem por minibatch, PPO value-clipping, `max_grad_norm=0.5`, init ortogonal,
   LR com decaimento linear. Foram as defesas contra o `diff-waiting-time` ser *spiky*.

5. **Reprodutibilidade.** `--seed S` semeia python/numpy/torch + `PYTHONHASHSEED`. Worker
   `w`, episódio `e` → `sumo_seed = S*10000 + w*100 + e`. Avaliação e Fixed-Time usam a
   **mesma** lista de seeds held-out (comparação pareada). `config.json` grava git hash +
   versões de todas as libs + SUMO. **Verificado:** dois runs com o mesmo seed produzem
   `train_log.csv` bit-idêntico em toda coluna de aprendizado (só diferem `wallclock`/`sps`).

6. **Hardware (Apple Silicon, 10 cores).** Gargalo = simulação SUMO (CPU, single-thread por
   instância). `--n-workers 8` deixa 2 cores para o trainer; `--torch-threads 2`. Device
   default = **CPU**: as MLPs são minúsculas (state_dim 88–195) e o overhead do MPS não
   compensa. O ganho real vem de workers paralelos + libsumo.

### Alterações fora do pacote

- `pyproject.toml` — extra opcional `[marl]` (`torch`, `pyyaml`, `tensorboard`, `matplotlib`).
- `README.md` — seção MAPPO com os comandos.
- `.gitignore` — `/outputs/mappo/` (checkpoints/plots/logs ficam locais).

---

## 3. Setup experimental — grid 2x2

| | |
|---|---|
| Rede / rota | `sumo_rl/nets/2x2grid/2x2.net.xml` + `2x2.rou.xml` (4 semáforos, `Discrete(4)`, 4 flows a probabilidade 0.1 — demanda uniforme e estacionária) |
| Episódio | `num_seconds = 20000` → 4000 passos de decisão/agente (`delta_time=5`, `yellow=2`, `min_green=5`) |
| Recompensa | `diff-waiting-time` (default do sumo-rl, usada por todos os experimentos existentes) |
| Orçamento | 3.0M agent-steps → 184 iterações de 16 384 amostras |
| PPO | `lr=5e-4` (decai linear), `γ=0.99`, `λ=0.95`, `clip=0.2` (+ value-clip), `ent_coef=0.01`, `vf_coef=0.5`, `max_grad_norm=0.5`, `n_epochs=10`, `n_minibatches=4` |
| Rede | MLP 2×128, `tanh`, init ortogonal · value normalization ON · adv-norm ON |
| Paralelismo | 8 workers SUMO (libsumo), device CPU |
| Avaliação | 5 seeds held-out (`9_000_001..5`), ações determinísticas (argmax), a cada 10 iterações |
| Wall-clock | ~64 min · ~708 steps/s médios |
| Versões | Python 3.12.14 · torch 2.14.0 · gymnasium 1.3.0 · pettingzoo 1.27.0 · sumo-rl 1.4.5 · SUMO 1.27.1 |

---

## 4. Resultados — 2x2

Média no episódio (20000s), 5 seeds held-out. Fixed-Time = programa semafórico embutido
da rede nos mesmos seeds.

| Métrica | Fixed-Time | MAPPO (final) | Δ |
|---|---|---|---|
| **Tempo de espera total do sistema (s)** | 236.0 ± 5.5 | **4.55 ± 0.13** | **−98.1%** |
| Espera média por veículo (s) | 8.54 | 0.24 | −97% |
| Velocidade média (m/s) | 6.31 | 10.41 | **+65%** |
| Veículos parados (média na rede) | 11.2 | 1.3 | −89% |
| Throughput (veículos servidos ao fim) | 7962 | 7973 | ≈ igual |

### Dinâmica do treino

- **Convergência muito rápida:** −97.5% já na iteração 10 (180k agent-steps). Praticamente
  estável a partir de ~iter 40 (670k steps); curva plana de iter 60 a 183.
- **Estável:** sem colapso de política; desvio entre os 5 seeds de avaliação ±0.13 s.
- **Diagnósticos saudáveis:** `entropy` 1.38 → 0.43 (política decidida, não colapsada num
  `Discrete(4)`); `approx_kl` ~1e-3 → ~0; `clipfrac` ~0.01; `value_loss` despenca;
  `explained_variance` sobe para ~0.21 (baixo mas positivo — com a fila zerada o retorno
  tem pouca variância para o crítico explicar).

Gráficos: `outputs/mappo/2x2/lc_waiting_*.png`, `lc_speed_*.png`, `bar_final_*.png` e
`outputs/mappo/2x2/<run>/plots/training_diag.png`.

### Leitura

O ganho é **real** — o throughput idêntico (~7970 veículos servidos) mostra que a política
não está "trapaceando" bloqueando a inserção de carros; ela serve os mesmos veículos com
fila praticamente inexistente (~1.3 parados na rede inteira a qualquer momento).

---

## 5. Limitações / ressalvas

- **O 2x2 é um cenário fácil.** 4 interseções, demanda uniforme/estacionária/moderada, rede
  sub-saturada. O Fixed-Time aqui é baseline fraco (ciclo fixo não adaptado). Chegar perto
  do ótimo é alcançável — **não esperar −98% no 4x4**.
- **3M steps é exagero para o 2x2** — dava para parar em ~1M (iter ~60). O orçamento de 10M
  do 4x4 faz mais sentido lá.
- **1 seed de treino.** A variância entre seeds de *avaliação* é mínima (±0.13), mas o ideal
  para as curvas finais é 3 seeds de treino.
- Ainda **não rodado:** ablação IPPO (`--critic local`) para isolar a contribuição do CTDE.
  No 2x2 provavelmente IPPO também resolve; a diferença deve aparecer no 4x4.
- Implementação assume **agentes homogêneos** e cadência de ação compartilhada (vale para
  2x2, 4x4-Lucas, RESCO grid4x4). Cologne/Ingolstadt precisariam de padding de observação.

---

## 6. Próximos passos

1. **4x4-Lucas** — o teste de verdade (16 agentes, demanda não-estacionária pesada, 80000s):
   `python -m experiments.train_mappo --scenario 4x4 --seed 0 --n-workers 8`
2. Seeds 1 e 2 no 2x2 para banda nas curvas.
3. Ablação **IPPO** (`--critic local`) nos dois cenários.
4. Opcional: `--reward pressure` como ablação de recompensa.

---

## Como reproduzir

```bash
pip install -e ".[marl]" && pip install libsumo
export SUMO_HOME=/Library/Frameworks/EclipseSUMO.framework/Versions/1.27.1/EclipseSUMO/share/sumo

python -m experiments.train_mappo --scenario 2x2 --seed 0 --n-workers 8
python experiments/plot_mappo.py --scenario 2x2 --runs "outputs/mappo/2x2/mappo_*"
```
