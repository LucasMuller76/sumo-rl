"""Compara todos os algoritmos treinados num mesmo grid.

Lê os `evaluation.csv` já gerados por cada `train_<algo>_<grid>.py --evaluate`
(em `outputs/<grid>/<algo>/evaluation.csv`) e produz um resumo único, para
comparar DQN/PPO/QR-DQN/A2C/MAPPO lado a lado nas mesmas métricas.

    python plot_compare.py --grid 2x2
    python plot_compare.py --grid 4x4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

import _common as common


def load_all(grid: str) -> pd.DataFrame:
    grid_dir = common.REPO_ROOT / "outputs" / grid
    frames = []
    for csv_path in sorted(grid_dir.glob("*/evaluation.csv")):
        df = pd.read_csv(csv_path)
        df["run_dir"] = csv_path.parent.name
        frames.append(df)
    if not frames:
        raise SystemExit(
            f"Nenhum evaluation.csv encontrado em {grid_dir}/*/evaluation.csv. "
            f"Rode `train_<algo>_{grid}.py --evaluate` para os algoritmos que quiser comparar."
        )
    return pd.concat(frames, ignore_index=True)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por controlador (Fixed-Time, Max-Pressure, DQN, PPO, ...).

    Cada evaluation.csv repete os baselines (Fixed-Time/Max-Pressure) — como
    eles usam as mesmas EVAL_SEEDS em todo experimento, mantemos só a primeira
    cópia para não inflar a média com múltiplos runs idênticos.
    """
    cols = [c for c, _ in common.available_metrics(df)]
    baselines = df[df.controller.isin(["Fixed-Time", "Max-Pressure"])]
    baselines = baselines.drop_duplicates(subset=["controller", "episode"])
    algos = df[~df.controller.isin(["Fixed-Time", "Max-Pressure"])]
    combined = pd.concat([baselines, algos], ignore_index=True)
    return combined.groupby("controller")[cols].agg(["mean", "std"])


def plot_bars(summary: pd.DataFrame, grid: str, out_path: Path) -> None:
    order = sorted(summary.index, key=lambda c: (c not in ("Fixed-Time", "Max-Pressure"), c))
    metrics = [(c, label) for c, label in common.EVAL_METRICS if (c, "mean") in summary.columns]
    fig, axes = common.metric_grid(len(metrics))
    for ax, (col, label) in zip(axes, metrics):
        means = summary[(col, "mean")].reindex(order)
        stds = summary[(col, "std")].reindex(order).fillna(0.0)
        ax.bar(order, means, yerr=stds, capsize=3)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"Comparação entre algoritmos — grid {grid}", fontsize=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara os algoritmos já avaliados num mesmo grid")
    parser.add_argument("--grid", required=True, choices=["2x2", "4x4"])
    args = parser.parse_args()

    df = load_all(args.grid)
    summary = build_summary(df)

    out_dir = common.REPO_ROOT / "outputs" / args.grid
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "comparison_summary.csv"
    plot_path = out_dir / "comparison_bars.png"

    summary.to_csv(summary_path)
    plot_bars(summary, args.grid, plot_path)

    print(f"\nResumo comparativo — grid {args.grid} (média ± desvio por controlador):\n")
    print(summary)
    print(f"\nSalvo em {summary_path} e {plot_path}")


if __name__ == "__main__":
    main()
