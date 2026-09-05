"""Multi-agent DQN (IDQN + parameter sharing, VDN, QMIX) for SUMO traffic-signal control.

Self-contained package used by ``experiments/train_dqn.py``. Depends on ``torch``
(install with ``pip install -e ".[rl]"``). It does NOT import from ``experiments/mappo``;
the shared infrastructure (``vec_sumo``, ``eval``, ``utils``, scenario presets) is copied
verbatim with byte-identical constants so the DQN and MAPPO experiments run under the same
conditions.
"""
