"""Compara funções de recompensa para um mesmo algoritmo num mesmo grid.

Lê o que os arquivos do estudo de recompensas gravam em
`outputs/<grid>/rewards/<recompensa>/<algo>/` (ex.: `train_ppo_4x4_reward_queue.py`,
ou `train_<algo>_<grid>.py --reward <recompensa>`):

    evaluation.csv                  avaliação vs Fixed-Time e Max-Pressure (--evaluate)
    seed_N/training_metrics.csv     métricas de tráfego durante o treino
    seed_N/monitor.csv              recompensa crua de treino

e grava em `outputs/<grid>/rewards/`:

    <algo>_reward_summary.csv    média, desvio entre sementes e desvio entre episódios, por métrica
    <algo>_reward_bars.png       uma barra por recompensa + baselines, por métrica
    <algo>_reward_heatmap.png    ganho % sobre o Fixed-Time por métrica (azul = melhor, vermelho = pior)
    <algo>_reward_learning.png   métricas de tráfego ao longo do treino, todas as recompensas no mesmo eixo
    <algo>_reward_training.png   recompensa crua de treino, um painel por recompensa

A coluna "reward" fica FORA da comparação: cada execução mede a recompensa na
própria escala, então ela não diz nada entre funções diferentes. A comparação
usa só as métricas de tráfego, que são as mesmas para todos.

Barras de erro do algoritmo = desvio-padrão ENTRE SEMENTES DE TREINO (média de
cada semente -> desvio), que é a incerteza relevante para comparar configurações
de treino; com uma semente só, cai para o desvio entre episódios de avaliação.
Os baselines não têm semente de treino: as barras deles são o desvio entre
episódios.

    python plot_rewards.py --grid 4x4 --algo ppo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

import _common as common

BASELINES = ("Fixed-Time", "Max-Pressure")
REFERENCE = common.DEFAULT_REWARD

# Métricas em que menor é melhor -- definem o sinal do ganho % (positivo = melhor).
LOWER_IS_BETTER = {"waiting_time", "queue", "wait_p95", "wait_max", "co2_kg", "phase_switches"}

SHORT_LABELS = {
    "waiting_time": "Espera média",
    "queue": "Fila média",
    "mean_speed": "Velocidade média",
    "throughput": "Throughput",
    "wait_p95": "Espera p95",
    "wait_max": "Espera máxima",
    "co2_kg": "CO₂",
    "phase_switches": "Trocas de fase",
}

# --------------------------------------------------------------------------
# Estilo. Cor segue a RECOMPENSA (entidade), nunca a posição: cada recompensa tem
# um slot fixo da paleta categórica, o mesmo em todas as figuras. Paleta validada
# para daltonismo na ordem adjacente; os baselines ficam em cinzas neutros.
# --------------------------------------------------------------------------
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
REWARD_SLOTS = {"diff-waiting-time": 0, "average-speed": 1, "queue": 2, "pressure": 3, "co2": 4}
BASELINE_COLORS = {"Fixed-Time": "#b9b8b0", "Max-Pressure": "#6f6e69"}
BASELINE_LINES = {"Fixed-Time": "--", "Max-Pressure": ":"}
# Divergente azul (melhor) <-> vermelho (pior) com meio cinza neutro.
DIVERGING = LinearSegmentedColormap.from_list(
    "gain", ["#a32b2b", "#e34948", "#f3b9b5", "#f0efec", "#b7d3f6", "#3987e5", "#184f95"]
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "text.color": INK,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "font.size": 10,
    }
)


def base_reward(variant: str) -> str:
    return variant[: -len("-raw")] if variant.endswith("-raw") else variant


def variant_color(variant: str) -> str:
    name = base_reward(variant)
    if name in REWARD_SLOTS:
        return SERIES[REWARD_SLOTS[name]]
    # Recompensa nova (ex.: registrada depois): primeiro slot livre, em ordem
    # alfabética do nome -- estável entre figuras e execuções.
    known = set(REWARD_SLOTS)
    extras = sorted({n for n in common.REWARD_FNS if n not in known} | {name})
    return SERIES[(len(REWARD_SLOTS) + extras.index(name)) % len(SERIES)]


def is_raw(variant: str) -> bool:
    return variant.endswith("-raw")


def steps_label(x, _pos=None) -> str:
    if not x:
        return "0"
    return f"{x / 1e6:g}M" if abs(x) >= 1e6 else f"{x / 1e3:g}k"


def add_header(fig, title: str, handles, labels=None) -> None:
    """Título à esquerda e, logo abaixo, a legenda numa linha própria (nunca
    sobre o título)."""
    fig.suptitle(title, fontsize=15, x=0.01, y=0.995, ha="left", va="top")
    if handles:
        kwargs = {} if labels is None else {"labels": labels}
        fig.legend(
            handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=min(len(handles), 6),
            fontsize=9, handlelength=2.2, columnspacing=1.6, **kwargs,
        )


# --------------------------------------------------------------------------
# Dados
# --------------------------------------------------------------------------
def study_dir(grid: str) -> Path:
    return common.REPO_ROOT / "outputs" / grid / "rewards"


def load(grid: str, algo: str) -> pd.DataFrame:
    frames = []
    for csv_path in sorted(study_dir(grid).glob(f"*/{algo}/evaluation.csv")):
        df = pd.read_csv(csv_path)
        df["variant"] = csv_path.parent.parent.name  # ex.: "queue" ou "queue-raw"
        frames.append(df)
    if not frames:
        raise SystemExit(
            f"Nenhum evaluation.csv em {study_dir(grid)}/*/{algo}/. Rode antes "
            f"`train_{algo}_{grid}_reward_<recompensa>.py --evaluate` (ou `--evaluate --reward <recompensa>`)."
        )
    return pd.concat(frames, ignore_index=True)


def variant_order(variants) -> list[str]:
    """Referência primeiro, depois a ordem dos slots de cor, depois o resto em
    ordem alfabética -- fixa, para a posição de cada recompensa não mudar entre
    gráficos."""
    variants = set(variants)

    def key(v):
        name = base_reward(v)
        return (name != REFERENCE, REWARD_SLOTS.get(name, len(REWARD_SLOTS)), name, is_raw(v))

    return sorted(variants, key=key)


def traffic_metrics(df: pd.DataFrame) -> list[tuple[str, str]]:
    return [(c, label) for c, label in common.available_metrics(df) if c != "reward"]


def build_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, str]], dict]:
    """Uma linha por controlador: baselines + uma por recompensa ("PPO · queue").

    Colunas (métrica, estatística): mean, std_seeds, std_episodes, err (a usada
    nas barras). Devolve também as médias por semente de cada recompensa, para
    plotar os pontos individuais."""
    metrics = traffic_metrics(df)
    cols = [c for c, _ in metrics]

    # Os baselines rodam nas mesmas sementes em todo experimento e não dependem
    # da recompensa de treino (só a coluna "reward", que está fora): uma cópia basta.
    base = df[df.controller.isin(BASELINES)]
    base = base[base.variant == base.variant.iloc[0]]
    algos = df[~df.controller.isin(BASELINES)]
    algo_name = algos.controller.iloc[0]

    rows, seed_means, n_seeds = {}, {}, {}
    for name in BASELINES:
        sub = base[base.controller == name]
        if len(sub):
            rows[name] = {(c, "mean"): sub[c].mean() for c in cols} | {(c, "std_episodes"): sub[c].std() for c in cols}
            n_seeds[name] = 0
    for variant in variant_order(algos.variant):
        sub = algos[algos.variant == variant]
        per_seed = sub.groupby("train_seed")[cols].mean()
        label = f"{algo_name} · {variant}"
        rows[label] = (
            {(c, "mean"): sub[c].mean() for c in cols}
            | {(c, "std_seeds"): per_seed[c].std() for c in cols}
            | {(c, "std_episodes"): sub[c].std() for c in cols}
        )
        seed_means[label] = per_seed
        n_seeds[label] = len(per_seed)

    summary = pd.DataFrame.from_dict(rows, orient="index")
    for c in cols:
        if (c, "std_seeds") not in summary:
            summary[(c, "std_seeds")] = np.nan
        use_seeds = pd.Series(n_seeds).reindex(summary.index) >= 2
        summary[(c, "err")] = summary[(c, "std_seeds")].where(use_seeds, summary[(c, "std_episodes")]).fillna(0.0)
    summary[("train_seeds", "n")] = pd.Series(n_seeds).reindex(summary.index)
    ordered = [(c, s) for c in cols for s in ("mean", "std_seeds", "std_episodes", "err")]
    summary = summary[ordered + [("train_seeds", "n")]]
    summary.columns = pd.MultiIndex.from_tuples(summary.columns)
    return summary, metrics, seed_means


def gain_vs_fixed(summary: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Ganho % sobre o Fixed-Time, com sinal ajustado: positivo = melhor."""
    means = summary.xs("mean", axis=1, level=1)[cols]
    ref = means.loc["Fixed-Time"]
    gain = (means - ref) / ref.abs().replace(0, np.nan) * 100
    for c in cols:
        if c in LOWER_IS_BETTER:
            gain[c] = -gain[c]
    return gain.drop(index="Fixed-Time")


# --------------------------------------------------------------------------
# Gráficos
# --------------------------------------------------------------------------
def label_variant(label: str) -> str:
    return label.split(" · ")[-1]


def plot_bars(summary: pd.DataFrame, metrics, seed_means: dict, title: str, out_path: Path) -> None:
    labels = list(summary.index)
    colors = [BASELINE_COLORS.get(lab) or variant_color(label_variant(lab)) for lab in labels]
    hatches = ["//" if is_raw(label_variant(lab)) else "" for lab in labels]
    ticks = [label_variant(lab) for lab in labels]
    x = np.arange(len(labels))

    fig, axes = common.metric_grid(len(metrics), cell=(5.4, 4.2))
    for ax, (col, label) in zip(axes, metrics):
        bars = ax.bar(
            x, summary[(col, "mean")], width=0.72, color=colors, hatch=hatches, edgecolor=SURFACE, linewidth=1.5
        )
        for bar, hatch in zip(bars, hatches):
            if hatch:
                bar.set_edgecolor(INK_2)
        ax.errorbar(x, summary[(col, "mean")], yerr=summary[(col, "err")], fmt="none", ecolor=INK_2, capsize=3, lw=1)
        # Média de cada semente de treino como ponto: com poucas sementes, o
        # espalhamento real diz mais que a barra de erro sozinha.
        for i, lab in enumerate(labels):
            per_seed = seed_means.get(lab)
            if per_seed is not None and len(per_seed) >= 2:
                jitter = np.linspace(-0.15, 0.15, len(per_seed))
                ax.scatter(
                    x[i] + jitter, per_seed[col], s=18, color=INK, edgecolor=SURFACE, linewidth=0.8, zorder=3
                )
        ax.set_xticks(x, ticks, rotation=30, ha="right", fontsize=9)
        ax.set_title(label, fontsize=11)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)

    n = summary[("train_seeds", "n")]
    n_rl = sorted({int(v) for v in n[n > 0]})
    handles = [Patch(color=BASELINE_COLORS[b], label=b) for b in BASELINES if b in summary.index]
    handles.append(Line2D([], [], marker="o", ls="", color=INK, markersize=5, label="média de uma semente de treino"))
    if any(is_raw(label_variant(lab)) for lab in labels):
        handles.append(Patch(facecolor="white", edgecolor=INK_2, hatch="//", label="sem normalização (-raw)"))
    add_header(fig, title, handles)
    fig.text(
        0.01,
        0.005,
        f"Barras de erro: recompensas = desvio entre sementes de treino (n = {', '.join(map(str, n_rl))}; com 1 "
        "semente, desvio entre episódios); baselines = desvio entre episódios de avaliação.",
        fontsize=8,
        color=INK_2,
    )
    plt.tight_layout(rect=(0, 0.02, 1, 0.92))
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_heatmap(summary: pd.DataFrame, metrics, title: str, out_path: Path) -> None:
    cols = [c for c, _ in metrics]
    gain = gain_vs_fixed(summary, cols)
    rows = list(gain.index)

    fig, ax = plt.subplots(figsize=(1.35 * len(cols) + 3.0, 0.62 * len(rows) + 2.2))
    # Escala simétrica-log: ganhos pequenos (1-5%) continuam visíveis ao lado de
    # ganhos grandes (90%+) sem que uma métrica domine a escala de cor. Valores
    # além de ±100% saturam; o número na célula é sempre o real.
    norm = SymLogNorm(linthresh=5, vmin=-100, vmax=100, base=10)
    ax.imshow(gain.to_numpy(dtype=float), cmap=DIVERGING, norm=norm, aspect="auto")
    for i in range(len(rows)):
        for j, c in enumerate(cols):
            v = gain.iloc[i, j]
            if pd.isna(v):
                text, color = "—", INK_2
            else:
                text = f"{v:+.1f}%"
                color = "white" if abs(float(norm(np.clip(v, -100, 100))) - 0.5) > 0.32 else INK
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)
    ax.set_xticks(range(len(cols)), [SHORT_LABELS.get(c, c) for c in cols], rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), [label_variant(r) if " · " in r else r for r in rows])
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(len(cols) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(rows) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_title(title, fontsize=13, loc="left", pad=28)
    ax.text(
        0,
        1.02,
        "Ganho sobre o Fixed-Time (%) — positivo/azul = melhor, negativo/vermelho = pior (sinal já ajustado por métrica)",
        transform=ax.transAxes,
        fontsize=8.5,
        color=INK_2,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def load_learning(grid: str, algo: str) -> dict[str, list[pd.DataFrame]]:
    """Por recompensa, uma tabela por semente: uma linha por episódio de treino
    (média das cópias paralelas do SUMO)."""
    out: dict[str, list[pd.DataFrame]] = {}
    for path in sorted(study_dir(grid).glob(f"*/{algo}/seed_*/training_metrics.csv")):
        d = pd.read_csv(path)
        if d.empty:
            continue
        variant = path.parent.parent.parent.name
        per_episode = d.groupby("episode").agg(
            timesteps=("timesteps", "max"), **{m: (m, "mean") for m in common.TRAINING_METRICS}
        )
        out.setdefault(variant, []).append(per_episode)
    return out


def plot_learning(grid: str, algo: str, summary: pd.DataFrame | None, out_path: Path) -> bool:
    runs = load_learning(grid, algo)
    if not runs:
        return False
    labels = dict(common.EVAL_METRICS)
    metrics = list(common.TRAINING_METRICS)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    axes = axes.flatten()

    for ax, m in zip(axes, metrics):
        values = []
        for variant in variant_order(runs):
            seeds = runs[variant]
            # Alinha as sementes pelo índice do episódio (mesmo orçamento => mesmos
            # episódios); runs ainda em andamento entram até onde chegaram.
            stacked = pd.concat({i: s[m] for i, s in enumerate(seeds)}, axis=1)
            steps = pd.concat({i: s["timesteps"] for i, s in enumerate(seeds)}, axis=1).mean(axis=1)
            mean, std = stacked.mean(axis=1), stacked.std(axis=1).fillna(0.0)
            color = variant_color(variant)
            ls = "--" if is_raw(variant) else "-"
            # Marcadores só com poucos episódios (início do treino / smoke test),
            # quando a linha sozinha some.
            marker = "o" if len(mean) < 10 else None
            ax.plot(steps, mean, color=color, ls=ls, lw=2, marker=marker, ms=6, label=f"{variant} (n={len(seeds)})")
            if len(seeds) >= 2:
                ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15, lw=0)
            values.append(stacked.to_numpy().ravel())

        if summary is not None and (m, "mean") in summary.columns:
            for b in BASELINES:
                if b in summary.index:
                    ax.axhline(
                        summary.loc[b, (m, "mean")], color=BASELINE_COLORS[b], ls=BASELINE_LINES[b], lw=1.5,
                        label=f"{b} (avaliação)",
                    )
        allv = np.concatenate(values) if values else np.array([])
        allv = allv[np.isfinite(allv) & (allv > 0)]
        if m in ("waiting_time", "queue") and allv.size and allv.max() / max(allv.min(), 1e-9) > 20:
            ax.set_yscale("log")
            ax.set_ylabel(f"{labels.get(m, m)} — escala log")
        else:
            ax.set_ylabel(labels.get(m, m))
        ax.set_title(SHORT_LABELS.get(m, m), fontsize=11, loc="left")
        ax.set_xlabel("Passos de treino")
        ax.xaxis.set_major_formatter(FuncFormatter(steps_label))

    handles, legend_labels = axes[0].get_legend_handles_labels()
    add_header(fig, f"{algo.upper()} — métricas de tráfego durante o treino — grid {grid}", handles, legend_labels)
    fig.text(
        0.01,
        0.005,
        "Por episódio de treino (média das cópias paralelas); linha = média entre sementes, faixa = ± 1 desvio. "
        "Política estocástica e tráfego aleatório do treino; linhas tracejadas = baselines na avaliação.",
        fontsize=8,
        color=INK_2,
    )
    plt.tight_layout(rect=(0, 0.02, 1, 0.91))
    plt.savefig(out_path, dpi=200)
    plt.close()
    return True


def plot_training(grid: str, algo: str, out_path: Path) -> None:
    runs = [d for d in sorted(study_dir(grid).glob(f"*/{algo}")) if any(d.glob("seed_*/monitor.csv"))]
    if not runs:
        return
    by_variant = {r.parent.name: r for r in runs}
    variants = variant_order(by_variant)

    fig, axes = common.metric_grid(len(variants), cell=(5.4, 4.0))
    for ax, variant in zip(axes, variants):
        color = variant_color(variant)
        curves = []
        for m in sorted(by_variant[variant].glob("seed_*/monitor.csv")):
            d = pd.read_csv(m, skiprows=1)
            if d.empty:
                continue
            # x em passos de treino: cada linha do monitor é um episódio de UM
            # agente (l passos), então a soma acumulada de l dá os passos totais.
            curve = pd.Series(d["r"].rolling(200, min_periods=1).mean().to_numpy(), index=d["l"].cumsum().to_numpy())
            curves.append(curve)
        single = len(curves) == 1
        for curve in curves:
            ax.plot(curve.index, curve.values, color=color, lw=2 if single else 1, alpha=1.0 if single else 0.35)
        if len(curves) >= 2:
            n = min(len(c) for c in curves)
            mean = np.mean([c.values[:n] for c in curves], axis=0)
            ax.plot(curves[0].index[:n], mean, color=color, lw=2)
        ax.set_title(f"{variant} (n={len(curves)})", fontsize=11, loc="left")
        ax.set_xlabel("Passos de treino")
        ax.set_ylabel("Recompensa crua por episódio de agente")
        ax.xaxis.set_major_formatter(FuncFormatter(steps_label))
    add_header(
        fig,
        f"{algo.upper()} — recompensa de treino (crua, média móvel de 200) — cada painel na própria escala",
        [
            Line2D([], [], color=INK_2, lw=1, alpha=0.5, label="semente individual"),
            Line2D([], [], color=INK_2, lw=2, label="média das sementes"),
        ],
    )
    plt.tight_layout(rect=(0, 0, 1, 0.91))
    plt.savefig(out_path, dpi=200)
    plt.close()


# --------------------------------------------------------------------------
# Relatório
# --------------------------------------------------------------------------
def print_report(summary: pd.DataFrame, metrics) -> None:
    cols = [c for c, _ in metrics]
    means = summary.xs("mean", axis=1, level=1)[cols]
    print("\nMédia por controlador:\n")
    print(means.round(3).assign(sementes=summary[("train_seeds", "n")]).to_string())

    print("\nGanho % sobre o Fixed-Time (positivo = melhor):\n")
    print(gain_vs_fixed(summary, cols).round(1).to_string())

    algo_rows = [r for r in summary.index if r not in BASELINES]
    if "waiting_time" in cols and algo_rows:
        ranking = means.loc[algo_rows, "waiting_time"].sort_values()
        print("\nRanking por tempo médio de espera (menor é melhor):")
        for i, (label, v) in enumerate(ranking.items(), 1):
            print(f"  {i}. {label_variant(label):<20} {v:.3f} s")

    ref = next((label for label in summary.index if label.endswith(" · " + REFERENCE)), None)
    if ref is None:
        print(f"\n(sem a recompensa de referência '{REFERENCE}' -- Δ% contra ela não calculado)")
        return
    others = [r for r in algo_rows if r != ref]
    if others:
        print(f"\nΔ% contra {ref} (valor bruto; negativo = melhor nas métricas de 'menor é melhor'):\n")
        delta = (means.loc[others] - means.loc[ref]) / means.loc[ref].abs().replace(0, np.nan) * 100
        print(delta.round(1).to_string())
    print(
        "\nMenor é melhor: " + ", ".join(c for c in cols if c in LOWER_IS_BETTER)
        + " | maior é melhor: " + ", ".join(c for c in cols if c not in LOWER_IS_BETTER)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara recompensas de treino para um algoritmo")
    parser.add_argument("--grid", required=True, choices=["2x2", "4x4"])
    parser.add_argument("--algo", default="ppo", help="nome curto do algoritmo (pasta), ex.: ppo")
    args = parser.parse_args()

    out = study_dir(args.grid)
    prefix = out / f"{args.algo}_reward"
    df = load(args.grid, args.algo)
    summary, metrics, seed_means = build_summary(df)
    algo_label = args.algo.upper()

    summary.to_csv(f"{prefix}_summary.csv")
    plot_bars(summary, metrics, seed_means, f"{algo_label} — recompensas de treino — grid {args.grid}",
              Path(f"{prefix}_bars.png"))
    plot_heatmap(summary, metrics, f"{algo_label} — recompensas de treino vs Fixed-Time — grid {args.grid}",
                 Path(f"{prefix}_heatmap.png"))
    has_learning = plot_learning(args.grid, args.algo, summary, Path(f"{prefix}_learning.png"))
    plot_training(args.grid, args.algo, Path(f"{prefix}_training.png"))

    print_report(summary, metrics)
    written = ["summary.csv", "bars.png", "heatmap.png"] + (["learning.png"] if has_learning else []) + ["training.png"]
    print(f"\nSalvo em {out}/{args.algo}_reward_{{{','.join(written)}}}")
    if not has_learning:
        print("(sem training_metrics.csv -- treinos antigos não têm; o learning.png aparece nos novos)")


if __name__ == "__main__":
    main()
