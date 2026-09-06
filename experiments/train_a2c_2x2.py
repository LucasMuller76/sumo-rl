"""
A2C + SUMO-RL — grid 2x2

Um único A2C controla todos os semáforos do grid (parameter sharing).
O wrapper agent_indicator injeta a identidade do semáforo na observação,
para o modelo compartilhado saber qual interseção está controlando.

    python train_a2c_2x2.py --show                 # ações aleatórias com GUI (sanity check)
    python train_a2c_2x2.py --train [--seed N]     # treina uma semente (outputs/2x2/a2c/seed_N/)
    python train_a2c_2x2.py --evaluate             # avalia todas as sementes vs baselines
    python train_a2c_2x2.py --render [--seed N]    # roda o modelo treinado com GUI

Para um estudo com várias sementes: rode --train com --seed 0,1,2,3,4 e depois
--evaluate uma vez (agrega todas as sementes). Para comparar com os outros
algoritmos do mesmo grid, rode `python plot_compare.py --grid 2x2` depois de
avaliar cada um.
"""

from __future__ import annotations

from pathlib import Path

import sumo_rl
from stable_baselines3 import A2C

import _common as common

# --------------------------------------------------------------------------
# Especificação do experimento
# --------------------------------------------------------------------------
_NETS = Path(sumo_rl.__file__).resolve().parent / "nets"

SPEC = common.ExperimentSpec(
    algo="A2C",
    short="a2c",
    grid_slug="2x2",
    grid_label="grid 2x2",
    model_slug="a2c_2x2grid",
    net_file=_NETS / "2x2grid" / "2x2.net.xml",
    route_file=_NETS / "2x2grid" / "2x2.rou.xml",
)


# --------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------
def build_model(env, spec: common.ExperimentSpec) -> A2C:
    return A2C(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=128,
        gamma=spec.gamma,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        rms_prop_eps=1e-5,
        policy_kwargs={"net_arch": spec.net_arch},
        verbose=1,
        device="cpu",  # rede pequena: CPU é mais rápida que GPU aqui
    )


if __name__ == "__main__":
    common.run_cli(SPEC, A2C, common.make_default_train_fn(build_model))
