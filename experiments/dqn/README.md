# Multi-agent DQN (IDQN+PS / VDN / QMIX) for SUMO traffic-signal control

Custom, self-contained value-based counterpart to `experiments/mappo/`, for the `2x2` and
`4x4` grids. One **shared Q-network** across the homogeneous traffic signals; the
`--mixer` flag switches the training target:

- **`--mixer none` (headline)** — Independent DQN + parameter sharing. Per-agent target
  `y_i = R_i + gamma^m * Q_target(o'_i, a*_i)`. This is the value-based analogue of Lucas
  Alegre's independent tabular Q-learning (`experiments/ql_4x4grid.py`) and of MAPPO's
  decentralized execution.
- **`--mixer vdn`** — `Q_tot = sum_i Q_i`, team target on `mean_i R_i`.
- **`--mixer qmix`** — monotonic mixing network conditioned on the global team state
  (concat of all local observations). The CTDE counterpart to MAPPO's centralized critic.

Always on: **Double DQN**, **dueling** head, **3-step returns**, **Huber** loss,
grad-clip 10. **Prioritized replay** (proportional, sum-tree) on by default; `--no-per`
for the textbook comparison. Exploration is the **Ape-X per-worker epsilon ladder**
(`eps_i = eps_base ** (1 + i/(N-1) * alpha)`) plus a random-action warmup — no global
epsilon anneal.

Execution / evaluation is **decentralized and identical in all three modes**: each agent
acts on `argmax_a Q(o_i, a)` from the shared net (`DQNAgent.act_deterministic`).

## Install

```bash
pip install -e ".[rl]"            # torch, pyyaml, tensorboard, matplotlib
pip install libsumo              # optional, ~8x faster per SUMO worker
export SUMO_HOME=/path/to/sumo/share/sumo   # required by sumo-rl
```

## Run

```bash
# IDQN + parameter sharing (headline), 8 parallel SUMO workers
python -m experiments.train_dqn --scenario 2x2 --seed 0 --n-workers 8
python -m experiments.train_dqn --scenario 4x4 --seed 0 --n-workers 8

# QMIX (CTDE) and VDN
python -m experiments.train_dqn --scenario 4x4 --seed 0 --mixer qmix
python -m experiments.train_dqn --scenario 4x4 --seed 0 --mixer vdn

# textbook ablation: uniform replay
python -m experiments.train_dqn --scenario 2x2 --seed 0 --no-per

# Fixed-Time baseline only (written once per scenario, reused across runs)
python -m experiments.dqn.eval --scenario 2x2 --fixed-time

# standalone eval of a checkpoint on the held-out seeds
python -m experiments.dqn.eval --scenario 4x4 --checkpoint outputs/dqn/4x4/<run>/checkpoints/best.pt

# plots (matplotlib only); --compare overlays MAPPO runs on the same axes
python experiments/plot_dqn.py --scenario 4x4 --runs "outputs/dqn/4x4/dqn_qmix_*" \
    --compare "outputs/mappo/4x4/mappo_*"
```

3 training seeds per config for mean±std bands: run `--seed 0`, `1`, `2` as separate
processes, then point `plot_dqn.py --runs` at all of them.

## Methodology parity with MAPPO / comparison to Lucas Alegre

`config.SCENARIOS`, `EVAL_SEED_BASE = 9_000_000`, the training-seed formula
(`seed*10000 + worker*100 + episode`) and `eval.py` (KPI keys, aggregation, Fixed-Time)
are **byte-identical** to `experiments/mappo/` (copied verbatim, no import), so:

- Same nets (`2x2grid/2x2.net.xml`, `4x4-Lucas/4x4.net.xml`), reward `diff-waiting-time`,
  `delta_time=5` / `yellow_time=2` / `min_green=5`, episode length (20000 / 80000 s),
  held-out eval seeds, and 5 KPIs (total waiting time, mean wait per vehicle, mean speed,
  stopped vehicles, throughput).
- The Fixed-Time baseline is policy-free physics -- re-running it under `outputs/dqn/`
  should reproduce `outputs/mappo/<scn>/fixed_time_summary.json` within seed noise (a free
  check that "identical conditions" holds).

Differences vs Lucas Alegre's grid Q-learning: function approximation + parameter sharing
(vs 16 independent discretized Q-tables), Double/Dueling/n-step/PER (vs 1-step tabular
TD), annealed/laddered epsilon (vs fixed 0.05), full seeding, held-out deterministic
evaluation, a Fixed-Time baseline, and 8-way parallel collection. His only DQN
(`dqn_2way-single-intersection.py`) is single-agent SB3 vanilla.

## Hardware notes (Apple Silicon, 10 cores)

- Bottleneck is the SUMO simulation. `--n-workers 8` leaves 2 cores for the learner;
  `--torch-threads 2`.
- `--device cpu` default: the Q-net is a tiny MLP and MPS transfer overhead makes it
  slower than CPU here. `--device cuda` helps only on an NVIDIA box.
- Synchronous collection + a single-threaded learner + seeded RNG streams
  (`SeedSequence(seed).spawn`) -> reproducible run-to-run. The replay buffer is not
  checkpointed, so `--resume` is approximate (like the MAPPO trainer); a fully
  reproducible result needs one uninterrupted run. Results are comparable only at a fixed
  `--n-workers` (the epsilon ladder depends on the worker count).

## Outputs

```
outputs/dqn/<scenario>/
  fixed_time_summary.json
  _fixed_time/fixed_conn*_ep*.csv
  <run_name>/                       # dqn_<scenario>_<mixer>_<per|noper>_r-<reward>_s<seed>_<ts>
    config.json                     # full config + resolved dims + git + versions + eps ladder
    train_log.csv                   # per-iteration: loss / q_mean / td_error / epsilon / buffer_size / sps
    eval_log.csv                    # per-eval KPIs + % improvement vs Fixed-Time (SAME columns as MAPPO)
    checkpoints/{best.pt,last.pt}
    env_csv/dqn_eval_step*_conn*_ep*.csv
    tensorboard/
    plots/
```
