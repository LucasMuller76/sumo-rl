"""Plot DQN results (matplotlib only -- no seaborn / LaTeX).

    python experiments/plot_dqn.py --scenario 2x2 --runs "outputs/dqn/2x2/dqn_*"
    python experiments/plot_dqn.py --scenario 4x4 --runs "outputs/dqn/4x4/dqn_qmix_*" \
        --compare "outputs/mappo/4x4/mappo_*" --compare-label MAPPO

Reads ``eval_log.csv`` from each run dir and ``fixed_time_summary.json`` for the scenario
(schema is identical to the MAPPO output, so ``--compare`` can overlay MAPPO runs on the
same axes). Writes learning curves + a final-comparison bar chart under each run's
``plots/`` and a combined figure under ``outputs/dqn/<scenario>/``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_runs(patterns: list[str]) -> list[tuple[str, pd.DataFrame]]:
    runs = []
    for pat in patterns or []:
        for d in sorted(glob.glob(pat)):
            f = os.path.join(d, "eval_log.csv")
            if os.path.isfile(f):
                df = pd.read_csv(f)
                if len(df):
                    runs.append((d, df))
    return runs


def _fixed_time(scenario: str, out_root: str) -> dict | None:
    p = os.path.join(out_root, scenario, "fixed_time_summary.json")
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)


def _agg(runs, xcol: str, ycol: str):
    xs = np.array(sorted(set(np.concatenate([r[1][xcol].to_numpy() for r in runs]))), float)
    mat = np.vstack([np.interp(xs, df[xcol].to_numpy(float), df[ycol].to_numpy(float)) for _d, df in runs])
    return xs, mat.mean(0), mat.std(0)


def _series(ax, runs, metric, label, color, xcol="global_step"):
    for _d, df in runs:
        ax.plot(df[xcol], df[metric], alpha=0.25, lw=1, color=color)
    if len(runs) >= 2:
        x, m, s = _agg(runs, xcol, metric)
        ax.plot(x, m, color=color, lw=2.2, label=f"{label} (mean of {len(runs)})")
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.18)
    elif runs:
        ax.plot(runs[0][1][xcol], runs[0][1][metric], color=color, lw=2.2, label=label)


def learning_curve(series, ft, metric, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for runs, label, color in series:
        _series(ax, runs, metric, label, color)
    if ft is not None and metric == "waiting_time_mean":
        fv = ft["waiting_time_mean_mean"]
        ax.axhline(fv, color="tab:red", ls="--", lw=2, label="Fixed-Time")
        ax.fill_between(ax.get_xlim(), fv - ft["waiting_time_mean_std"], fv + ft["waiting_time_mean_std"],
                        color="tab:red", alpha=0.12)
    ax.set_xlabel("environment steps (agent-steps)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def bar_final(series, ft, out_path):
    metrics = [("waiting_time_mean", "Total waiting time (s)"),
               ("mean_speed_mean", "Mean speed (m/s)"),
               ("stopped_mean", "Stopped vehicles")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 3.8))
    for ax, (m, label) in zip(axes, metrics):
        names, heights, errs, colors = [], [], [], []
        for runs, lbl, color in series:
            if not runs:
                continue
            vals = np.array([df.sort_values("global_step")[m].iloc[-1] for _d, df in runs], float)
            names.append(lbl)
            heights.append(vals.mean())
            errs.append(vals.std() if len(vals) > 1 else 0.0)
            colors.append(color)
        if ft is not None:
            names.append("Fixed-Time")
            heights.append(ft[f"{m}_mean"])
            errs.append(ft[f"{m}_std"])
            colors.append("tab:red")
        ax.bar(names, heights, yerr=errs, capsize=5, color=colors)
        ax.set_title(label)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Final policy vs Fixed-Time")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def training_diag(run_dir, out_path):
    f = os.path.join(run_dir, "train_log.csv")
    if not os.path.isfile(f):
        return
    df = pd.read_csv(f)
    cols = [c for c in ["loss", "q_mean", "q_target_mean", "td_error_abs_mean", "grad_norm", "sps"] if c in df]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, c in zip(axes.ravel(), cols):
        sub = df[["global_step", c]].apply(pd.to_numeric, errors="coerce").dropna()
        ax.plot(sub["global_step"], sub[c], lw=1.3)
        ax.set_title(c)
        ax.grid(alpha=0.3)
        ax.set_xlabel("agent-steps")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", required=True)
    p.add_argument("--runs", nargs="+", required=True, help="glob(s) matching DQN run directories")
    p.add_argument("--compare", nargs="+", default=None, help="glob(s) of runs to overlay (e.g. MAPPO)")
    p.add_argument("--compare-label", default="MAPPO")
    p.add_argument("--label", default="DQN")
    p.add_argument("--out-root", default="outputs/dqn")
    p.add_argument("--compare-out-root", default="outputs/mappo")
    args = p.parse_args()

    runs = _load_runs(args.runs)
    if not runs:
        raise SystemExit(f"no run dirs with eval_log.csv matched {args.runs}")
    cmp_runs = _load_runs(args.compare)
    ft = _fixed_time(args.scenario, args.out_root) or _fixed_time(args.scenario, args.compare_out_root)
    print(f"{len(runs)} {args.label} run(s), {len(cmp_runs)} {args.compare_label} run(s); "
          f"fixed-time: {'found' if ft else 'MISSING'}")

    series = [(runs, args.label, "tab:blue")]
    if cmp_runs:
        series.append((cmp_runs, args.compare_label, "tab:green"))

    for d, _df in runs:
        os.makedirs(os.path.join(d, "plots"), exist_ok=True)
        training_diag(d, os.path.join(d, "plots", "training_diag.png"))

    combo = os.path.join(args.out_root, args.scenario)
    os.makedirs(combo, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    learning_curve(series, ft, "waiting_time_mean", "Total waiting time (s)",
                   os.path.join(combo, f"lc_waiting_{stamp}.png"))
    learning_curve(series, ft, "mean_speed_mean", "Mean speed (m/s)",
                   os.path.join(combo, f"lc_speed_{stamp}.png"))
    bar_final(series, ft, os.path.join(combo, f"bar_final_{stamp}.png"))
    learning_curve(series, ft, "waiting_time_mean", "Total waiting time (s)",
                   os.path.join(runs[0][0], "plots", "lc_waiting.png"))
    bar_final(series, ft, os.path.join(runs[0][0], "plots", "bar_final.png"))
    print(f"wrote plots to {combo}/ and {runs[0][0]}/plots/")


if __name__ == "__main__":
    main()
