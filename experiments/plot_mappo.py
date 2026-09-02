"""Plot MAPPO results (matplotlib only -- no seaborn / LaTeX).

    python experiments/plot_mappo.py --scenario 2x2 --runs "outputs/mappo/2x2/mappo_*"

Reads ``eval_log.csv`` from each run dir and ``fixed_time_summary.json`` for the
scenario, then writes learning curves + a final-comparison bar chart under each
run's ``plots/`` and a combined figure under ``outputs/mappo/<scenario>/``.
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
    for pat in patterns:
        for d in sorted(glob.glob(pat)):
            f = os.path.join(d, "eval_log.csv")
            if os.path.isfile(f):
                df = pd.read_csv(f)
                if len(df):
                    runs.append((d, df))
    return runs


def _fixed_time(scenario: str, out_root: str) -> dict | None:
    p = os.path.join(out_root, scenario, "fixed_time_summary.json")
    return json.load(open(p)) if os.path.isfile(p) else None


def _agg_across_runs(runs, xcol: str, ycol: str):
    """Interpolate every run onto the union of x values, return x, mean, std."""
    xs = sorted(set(np.concatenate([r[1][xcol].to_numpy() for r in runs])))
    xs = np.array(xs, float)
    mat = []
    for _d, df in runs:
        mat.append(np.interp(xs, df[xcol].to_numpy(float), df[ycol].to_numpy(float)))
    mat = np.vstack(mat)
    return xs, mat.mean(0), mat.std(0)


def learning_curve(runs, ft, metric, ylabel, out_path, xcol="global_step"):
    plt.figure(figsize=(7.2, 4.4))
    for d, df in runs:
        plt.plot(df[xcol], df[metric], alpha=0.28, lw=1, color="tab:blue")
    if len(runs) >= 2:
        x, m, s = _agg_across_runs(runs, xcol, metric)
        plt.plot(x, m, color="tab:blue", lw=2.2, label=f"MAPPO (mean of {len(runs)} seeds)")
        plt.fill_between(x, m - s, m + s, color="tab:blue", alpha=0.2)
    else:
        plt.plot(runs[0][1][xcol], runs[0][1][metric], color="tab:blue", lw=2.2, label="MAPPO")
    if ft is not None and metric in ("waiting_time_mean",):
        fv = ft["waiting_time_mean_mean"]
        plt.axhline(fv, color="tab:red", ls="--", lw=2, label="Fixed-Time")
        plt.fill_between(plt.xlim(), fv - ft["waiting_time_mean_std"], fv + ft["waiting_time_mean_std"],
                        color="tab:red", alpha=0.12)
    plt.xlabel("environment steps (agent-steps)")
    plt.ylabel(ylabel)
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def bar_final(runs, ft, out_path):
    metrics = [("waiting_time_mean", "Total waiting time (s)"),
               ("mean_speed_mean", "Mean speed (m/s)"),
               ("stopped_mean", "Stopped vehicles")]
    finals = {m: [df.sort_values("global_step")[m].iloc[-1] for _d, df in runs] for m, _ in metrics}
    fig, axes = plt.subplots(1, len(metrics), figsize=(11, 3.6))
    for ax, (m, label) in zip(axes, metrics):
        mvals = np.array(finals[m], float)
        bars = ["MAPPO"]
        heights = [mvals.mean()]
        errs = [mvals.std() if len(mvals) > 1 else 0.0]
        if ft is not None:
            key = {"waiting_time_mean": "waiting_time_mean", "mean_speed_mean": "mean_speed_mean",
                   "stopped_mean": "stopped_mean"}[m]
            bars.append("Fixed-Time")
            heights.append(ft[f"{key}_mean"])
            errs.append(ft[f"{key}_std"])
        ax.bar(bars, heights, yerr=errs, capsize=5, color=["tab:blue", "tab:red"][: len(bars)])
        ax.set_title(label)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("MAPPO (final) vs Fixed-Time")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def training_diag(run_dir, out_path):
    f = os.path.join(run_dir, "train_log.csv")
    if not os.path.isfile(f):
        return
    df = pd.read_csv(f)
    cols = [c for c in ["policy_loss", "value_loss", "entropy", "approx_kl", "explained_variance", "sps"] if c in df]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, c in zip(axes.ravel(), cols):
        ax.plot(df["global_step"], df[c], lw=1.4)
        ax.set_title(c)
        ax.grid(alpha=0.3)
        ax.set_xlabel("agent-steps")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", required=True)
    p.add_argument("--runs", nargs="+", required=True, help="glob(s) matching run directories")
    p.add_argument("--out-root", default="outputs/mappo")
    args = p.parse_args()

    runs = _load_runs(args.runs)
    if not runs:
        raise SystemExit(f"no run dirs with eval_log.csv matched {args.runs}")
    ft = _fixed_time(args.scenario, args.out_root)
    print(f"{len(runs)} run(s); fixed-time summary: {'found' if ft else 'MISSING'}")

    for d, _df in runs:
        pd_ = os.path.join(d, "plots")
        os.makedirs(pd_, exist_ok=True)
        training_diag(d, os.path.join(pd_, "training_diag.png"))

    combo = os.path.join(args.out_root, args.scenario)
    os.makedirs(combo, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    learning_curve(runs, ft, "waiting_time_mean", "Total waiting time (s)",
                   os.path.join(combo, f"lc_waiting_{stamp}.png"))
    learning_curve(runs, ft, "mean_speed_mean", "Mean speed (m/s)",
                   os.path.join(combo, f"lc_speed_{stamp}.png"))
    bar_final(runs, ft, os.path.join(combo, f"bar_final_{stamp}.png"))
    # also drop copies into the first run's plots/ for convenience
    learning_curve(runs, ft, "waiting_time_mean", "Total waiting time (s)",
                   os.path.join(runs[0][0], "plots", "lc_waiting.png"))
    bar_final(runs, ft, os.path.join(runs[0][0], "plots", "bar_final.png"))
    print(f"wrote plots to {combo}/ and {runs[0][0]}/plots/")


if __name__ == "__main__":
    main()
