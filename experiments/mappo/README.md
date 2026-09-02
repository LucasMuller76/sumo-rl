# MAPPO (CTDE) for SUMO traffic-signal control

Custom, self-contained **Multi-Agent PPO** with **Centralized Training + Decentralized
Execution** for the `2x2` and `4x4` grids. Follows the recipe of Yu et al. 2021,
*"The Surprising Effectiveness of PPO in Cooperative MARL"*:

- **Shared actor** `pi(a_i | o_i)` across the (homogeneous) traffic signals.
- **Centralized critic** `V(s_i)` where `s_i = concat_j(o_j) ++ onehot(i)`
  (`+ normalized system_* scalars` with `--global-scalars`). With `--critic local`
  the critic sees only `o_i` — that is the **IPPO** ablation.
- GAE with **time-limit (truncation) bootstrapping**: SumoEnvironment never emits a
  true terminal state, so every episode boundary bootstraps with `V(s_T)`.
- PPO clip + value clipping, **value normalization** (running return stats),
  per-minibatch advantage normalization, orthogonal init, linear LR anneal.

Execution is decentralized: each signal acts from its own local observation
(`experiments/mappo/mappo.py::MAPPO.act_deterministic`).

## Install

```bash
pip install -e ".[marl]"          # torch, pyyaml, tensorboard, matplotlib
pip install libsumo               # optional but ~8x faster per SUMO worker
export SUMO_HOME=/path/to/sumo    # required by sumo-rl
```

`libsumo` allows one simulation per process — which is exactly the one-SUMO-per-worker
layout here. Each worker sets `LIBSUMO_AS_TRACI=1` before importing `sumo_rl`.
Without `libsumo` the code falls back to the plain `traci` backend (works, noisier
startup). Force a backend with `--sumo-backend {auto,libsumo,traci}`.

## Run

```bash
# 2x2 grid, 8 parallel SUMO workers
python -m experiments.train_mappo --scenario 2x2 --seed 0 --n-workers 8

# 4x4-Lucas grid
python -m experiments.train_mappo --scenario 4x4 --seed 0 --n-workers 8 --eval-every 10

# IPPO ablation (same code, decentralized critic)
python -m experiments.train_mappo --scenario 2x2 --seed 0 --critic local

# Fixed-Time baseline only (written once per scenario, reused across runs)
python -m experiments.mappo.eval --scenario 2x2 --fixed-time --n-eval-seeds 5

# Plots (matplotlib only, no LaTeX)
python experiments/plot_mappo.py --scenario 2x2 --runs "outputs/mappo/2x2/mappo_*"
```

Multiple seeds for mean±std curves: run `--seed 0,1,2` as separate processes, then
point `plot_mappo.py --runs` at all of them.

## Hardware notes (Apple Silicon, 10 cores)

- The wall-clock bottleneck is the SUMO simulation (CPU-bound, single-thread per
  instance). The speedup comes from **parallel workers + libsumo**, not the GPU.
- `--n-workers 8` leaves 2 cores for the trainer/OS; `--torch-threads 2` keeps the
  update from oversubscribing.
- `--device cpu` is the default: the actor/critic are tiny MLPs (obs ~11-21, state
  ~90-200, hidden 128) and MPS transfer overhead makes them slower than CPU here.
  `--device mps` is available to try; `--device cuda` helps only on an NVIDIA box.

## Outputs

```
outputs/mappo/<scenario>/
  fixed_time_summary.json           # Fixed-Time baseline (paired eval seeds)
  _fixed_time/fixed_conn*_ep*.csv
  <run_name>/
    config.json                     # full config + git hash + lib/SUMO versions
    train_log.csv                   # per-iteration losses / KL / EV / sps
    eval_log.csv                    # per-eval KPIs + % improvement vs Fixed-Time
    checkpoints/{best.pt,last.pt}
    env_csv/mappo_eval_it*_conn*_ep*.csv   # sumo-rl native per-step KPI CSVs
    tensorboard/
    plots/
```

`run_name = mappo_<scenario>_<critic>_r-<reward>_s<seed>_<timestamp>`.

## Reproducibility

`--seed S` seeds python / numpy / torch and `PYTHONHASHSEED`. Worker `w`, episode `e`
uses `sumo_seed = S*10000 + w*100 + e`. Evaluation uses fixed held-out seeds
(`config.EVAL_SEED_BASE + 1..n`), and the Fixed-Time baseline uses the *same* seeds
(paired comparison). `config.json` records the git hash and all versions.

## Scope / limitations

- Assumes **homogeneous agents** (same obs/action shape) — true for `2x2`,
  `4x4-Lucas`, RESCO `grid4x4`/`arterial4x4`. Heterogeneous scenarios
  (cologne/ingolstadt) would need per-agent observation padding in
  `state.build_state`.
- Assumes every signal acts on the same cadence (shared `delta_time`/`min_green`), so
  the per-step agent dict is always full; missing agents fall back to their last
  observation.
- Synchronous on-policy only (no async rollout).
