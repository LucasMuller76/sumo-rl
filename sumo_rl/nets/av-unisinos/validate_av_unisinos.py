#!/usr/bin/env python3
"""Validação da infraestrutura do cenário "Av. Unisinos" (sem demanda).

Sobe o SUMO via TraCI apenas com a rede (nenhum veículo/route) e imprime:

  - IDs dos traffic lights   -> deve ser tls_1, tls_2, tls_3
  - nº de edges / lanes / junctions
  - nº de faixas de cada pista               -> deve ser 3
  - comprimento de cada trecho               -> 90 / 380 / 340 / 90 m
  - fases do programa de cada semáforo       -> 10 s aberto / 10 s fechado
  - quantos links cada semáforo controla     -> 6 (3 faixas EB + 3 faixas WB)

Confere também que:
  - não existe nenhuma aresta fora das duas pistas (sem transversais);
  - todas as faixas são RETAS (y constante ao longo de toda a forma), ou seja,
    as duas pistas nunca convergem;
  - não há nenhum veículo (esta etapa é só infraestrutura).

Prova, via TraCI, que os três semáforos são acessíveis: lê o estado de cada um
(getRedYellowGreenState) e altera (setRedYellowGreenState), restaurando depois.

Uso:

    python validate_av_unisinos.py            # headless (sumo)
    python validate_av_unisinos.py --gui      # abre o sumo-gui (precisa XQuartz)

Sai com código != 0 se algo essencial falhar.
"""
from __future__ import annotations

import argparse
import os
import sys

if "SUMO_HOME" not in os.environ:
    sys.exit("Defina a variável de ambiente SUMO_HOME.")
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import traci  # noqa: E402
import sumolib  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SUMOCFG = os.path.join(HERE, "av-unisinos.sumocfg")
NETFILE = os.path.join(HERE, "av-unisinos.net.xml")

EXPECTED_TLS = ["tls_1", "tls_2", "tls_3"]
EB_EDGES = ["EB_w_t1", "EB_t1_t2", "EB_t2_t3", "EB_t3_e"]   # Oeste -> Leste
WB_EDGES = ["WB_e_t3", "WB_t3_t2", "WB_t2_t1", "WB_t1_w"]   # Leste -> Oeste
AVENUE_EDGES = EB_EDGES + WB_EDGES
EXPECTED_LANES = 3          # faixas de carro por pista
EXPECTED_LINKS_PER_TLS = 6  # 3 faixas EB + 3 faixas WB

# Distâncias aproximadas medidas no mapa (m). Tolerância de 5 m: são
# aproximações, não medidas exatas.
EXPECTED_SEGMENTS = [
    ("EB_w_t1", "início -> Sinaleira 1", 90),
    ("EB_t1_t2", "Sin. 1 -> Sinaleira 2", 380),
    ("EB_t2_t3", "Sin. 2 -> Sinaleira 3", 340),
    ("EB_t3_e", "Sin. 3 -> fim", 90),
]
SEGMENT_TOLERANCE = 5.0


def check_straight(problems: list[str]) -> None:
    """As duas pistas têm de ser retas: y constante em cada faixa."""
    net = sumolib.net.readNet(NETFILE, withInternal=False)
    print("retilineidade das pistas (y de cada faixa):")
    for group, name in ((EB_EDGES, "EB"), (WB_EDGES, "WB")):
        ys = set()
        for eid in group:
            for lane in net.getEdge(eid).getLanes():
                ys.update(round(y, 2) for _, y in lane.getShape())
                if len({round(y, 2) for _, y in lane.getShape()}) != 1:
                    problems.append(f"faixa {lane.getID()} não é reta (y varia)")
        print(f"  pista {name}: y = {sorted(ys)}  ({len(ys)} valores = {len(ys)} faixas)")
    top = min(y for e in WB_EDGES for ln in net.getEdge(e).getLanes() for _, y in ln.getShape())
    bot = max(y for e in EB_EDGES for ln in net.getEdge(e).getLanes() for _, y in ln.getShape())
    print(f"  canteiro central: {top - bot:.1f} m entre as duas pistas")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gui", action="store_true", help="abre o sumo-gui (precisa XQuartz)")
    args = ap.parse_args()

    binary = sumolib.checkBinary("sumo-gui" if args.gui else "sumo")
    traci.start([binary, "-c", SUMOCFG, "--no-step-log", "true", "--no-warnings", "true"])

    problems: list[str] = []
    try:
        tls_ids = list(traci.trafficlight.getIDList())
        print(f"traffic lights : {tls_ids}")
        if sorted(tls_ids) != sorted(EXPECTED_TLS):
            problems.append(f"TLS esperados {EXPECTED_TLS}, obtidos {tls_ids}")

        edges = [e for e in traci.edge.getIDList() if not e.startswith(":")]
        lanes = [ln for ln in traci.lane.getIDList() if not ln.startswith(":")]
        print(f"edges          : {len(edges)}")
        print(f"lanes          : {len(lanes)}")
        print(f"junctions      : {len(traci.junction.getIDList())}")
        extras = sorted(set(edges) - set(AVENUE_EDGES))
        if extras:
            problems.append(f"arestas inesperadas (transversais?): {extras}")

        print("faixas por trecho:")
        for e in AVENUE_EDGES:
            n = traci.edge.getLaneNumber(e)
            print(f"  {e:10} -> {n} faixa(s)")
            if n != EXPECTED_LANES:
                problems.append(f"{e} tem {n} faixas (esperado {EXPECTED_LANES})")

        print("programa de cada semáforo:")
        for tid in tls_ids:
            logic = traci.trafficlight.getAllProgramLogics(tid)[0]
            fases = " | ".join(f"{p.state}({int(p.duration)}s)" for p in logic.phases)
            n_links = len(traci.trafficlight.getControlledLinks(tid))
            print(f"  {tid}: {fases}  -> {n_links} links controlados")
            if n_links != EXPECTED_LINKS_PER_TLS:
                problems.append(f"{tid} controla {n_links} links (esperado {EXPECTED_LINKS_PER_TLS})")

        print("comprimento dos trechos (pista EB):")
        total = 0.0
        for eid, label, esperado in EXPECTED_SEGMENTS:
            comp = traci.lane.getLength(f"{eid}_0")
            total += comp
            print(f"  {label:24} {comp:6.1f} m  (esperado ~{esperado} m)")
            if abs(comp - esperado) > SEGMENT_TOLERANCE:
                problems.append(f"{eid}: {comp:.1f} m, esperado ~{esperado} m")
        print(f"  {'TOTAL do corredor':24} {total:6.1f} m")

        check_straight(problems)

        traci.simulationStep()
        n_veic = traci.vehicle.getIDCount()
        print(f"veículos na rede após 1 passo: {n_veic}")
        if n_veic != 0:
            problems.append(f"esperado 0 veículos, encontrados {n_veic}")

        print("acesso/alteração dos TLS via TraCI:")
        for tid in EXPECTED_TLS:
            if tid not in tls_ids:
                continue
            state = traci.trafficlight.getRedYellowGreenState(tid)
            all_red = "r" * len(state)
            traci.trafficlight.setRedYellowGreenState(tid, all_red)
            ok = traci.trafficlight.getRedYellowGreenState(tid) == all_red
            traci.trafficlight.setRedYellowGreenState(tid, state)
            print(f"  {tid}: estado='{state}' -> set all-red {'OK' if ok else 'FALHOU'}")
            if not ok:
                problems.append(f"{tid}: setRedYellowGreenState não teve efeito")

        # 40 passos = 2 ciclos completos (10 s aberto + 10 s fechado)
        for _ in range(40):
            traci.simulationStep()
    finally:
        traci.close()

    if problems:
        print("\nFALHAS:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("\nOK — infraestrutura validada.")


if __name__ == "__main__":
    main()
