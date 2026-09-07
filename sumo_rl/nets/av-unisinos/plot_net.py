#!/usr/bin/env python3
"""Renderiza a rede av-unisinos como PNG (não precisa de XQuartz/sumo-gui).

    python plot_net.py                 # salva em av-unisinos.png
    python plot_net.py -o /tmp/x.png

Lê o av-unisinos.net.xml compilado via sumolib e desenha faixa a faixa, para
o canteiro central entre as duas pistas ficar visível.
"""
from __future__ import annotations

import argparse
import os
import sys

if "SUMO_HOME" not in os.environ:
    sys.exit("Defina a variável de ambiente SUMO_HOME.")
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import sumolib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default=os.path.join(HERE, "av-unisinos.png"))
    ap.add_argument("--net", default=os.path.join(HERE, "av-unisinos.net.xml"))
    args = ap.parse_args()

    net = sumolib.net.readNet(args.net, withInternal=False)
    fig, ax = plt.subplots(figsize=(15, 4.2))

    for e in net.getEdges():
        color = "#1f6feb" if e.getID().startswith("EB") else "#d1242f"
        for lane in e.getLanes():
            xs, ys = zip(*lane.getShape())
            ax.plot(xs, ys, color=color, lw=13.0, solid_capstyle="butt", alpha=0.95, zorder=2)

    # canteiro central: faixa hachurada entre as duas pistas
    xmin, ymin, xmax, ymax = net.getBoundary()
    top = min(y for e in net.getEdges() if e.getID().startswith("WB")
              for ln in e.getLanes() for _, y in ln.getShape())
    bot = max(y for e in net.getEdges() if e.getID().startswith("EB")
              for ln in e.getLanes() for _, y in ln.getShape())
    ax.fill_between([xmin, xmax], bot + 1.6, top - 1.6, color="#dcedc8",
                    hatch="///", edgecolor="#7cb342", lw=0, zorder=1)
    ax.text(xmax * 0.62, (top + bot) / 2, "canteiro central",
            ha="center", va="center", fontsize=9, style="italic", color="#33691e", zorder=3)

    # um marcador por posição de semáforo (os 2 nós daquela posição = 1 TLS)
    seen: dict[str, float] = {}
    for n in net.getNodes():
        if n.getType() == "traffic_light":
            for conn in n.getConnections():
                tl = conn.getTLSID()
                if tl:
                    seen.setdefault(tl, n.getCoord()[0])
                    break
    for tl, x in sorted(seen.items()):
        ax.plot([x, x], [bot - 11, top + 11], color="black", lw=2.5, zorder=6)
        for y in (bot - 1.6, top + 1.6):
            ax.scatter([x], [y], s=230, c="#f5c518", edgecolors="black", linewidths=1.6, zorder=7)
        ax.annotate(tl, (x, top + 11), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=13, fontweight="bold")

    ax.annotate("", xy=(xmax * 0.46, bot - 7), xytext=(xmax * 0.28, bot - 7),
                arrowprops=dict(arrowstyle="-|>", color="#1f6feb", lw=2.6))
    ax.text(xmax * 0.37, bot - 20, "Oeste → Leste (EB, 3 faixas)",
            ha="center", color="#1f6feb", fontsize=10)
    ax.annotate("", xy=(xmax * 0.28, top + 7), xytext=(xmax * 0.46, top + 7),
                arrowprops=dict(arrowstyle="-|>", color="#d1242f", lw=2.6))
    ax.text(xmax * 0.37, top + 13, "Leste → Oeste (WB, 3 faixas)",
            ha="center", color="#d1242f", fontsize=10)

    ax.set_title("Av. Unisinos (simplificada) — duas pistas RETAS e paralelas · 3 faixas cada · "
                 "3 semáforos (10 s aberto / 10 s fechado)", fontsize=11.5, pad=16)
    # A avenida tem 1600 m de comprimento por ~35 m de largura: em escala real
    # nada seria legível, então a vertical é exagerada de propósito.
    ax.set_xlim(xmin - 60, xmax + 60)
    ax.set_ylim(bot - 34, top + 34)
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.15)
    ax.legend(handles=[
        Line2D([0], [0], color="#1f6feb", lw=7, label="Pista Oeste→Leste (EB)"),
        Line2D([0], [0], color="#d1242f", lw=7, label="Pista Leste→Oeste (WB)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#f5c518",
               markeredgecolor="k", markersize=13, label="Semáforo (tls_1/2/3)"),
    ], loc="upper center", ncol=3, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, -0.10))
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"figura salva em {args.out}")


if __name__ == "__main__":
    main()
