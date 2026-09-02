"""Tests for the MAPPO implementation in experiments/mappo/.

The GAE / state-shape tests are pure-numpy and always run. The end-to-end smoke
test needs SUMO (SUMO_HOME) and torch, and is skipped otherwise.
"""

import os
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

from experiments.mappo.buffer import RolloutBuffer  # noqa: E402
from experiments.mappo.state import build_state, state_dim  # noqa: E402


def test_state_dim_and_shape():
    W, A, O = 3, 4, 11
    obs = np.random.rand(W, A, O).astype(np.float32)
    g = np.random.rand(W, 3).astype(np.float32)

    s = build_state(obs, g, A, "centralized", use_global_scalars=False)
    assert s.shape == (W, A, state_dim(O, A, "centralized", False))
    assert s.shape[-1] == A * O + A

    s2 = build_state(obs, g, A, "centralized", use_global_scalars=True)
    assert s2.shape[-1] == state_dim(O, A, "centralized", True) == A * O + A + 3

    s3 = build_state(obs, g, A, "local", use_global_scalars=False)
    assert s3.shape == (W, A, O)

    # every agent's slice must contain the full joint observation
    joint = obs.reshape(W, A * O)
    for i in range(A):
        assert np.allclose(s[:, i, : A * O], joint)
    # agent id one-hot at the tail
    assert np.allclose(s[:, :, -A:], np.broadcast_to(np.eye(A, dtype=np.float32), (W, A, A)))


def test_gae_truncation_bootstrap():
    """One worker, one agent, T=3, an episode boundary at t=1 -> must bootstrap, not zero."""
    T, W, A = 3, 1, 1
    buf = RolloutBuffer(T, W, A, obs_dim=2, state_dim=2, device=torch.device("cpu"))
    rewards = [1.0, 2.0, 3.0]
    values = [0.5, 0.7, 0.9]
    term_value = 5.0  # V(s_T) at the boundary after step t=1
    for t in range(T):
        buf.add(
            obs=np.zeros((W, A, 2), np.float32), states=np.zeros((W, A, 2), np.float32),
            actions=np.zeros((W, A), np.int64), logprobs=np.zeros((W, A), np.float32),
            values=np.full((W, A), values[t], np.float32), rewards=np.full((W, A), rewards[t], np.float32),
            dones=np.array([1.0 if t == 1 else 0.0], np.float32),
        )
    buf.set_term_values(1, np.array([[term_value]], np.float32))

    gamma, lam = 0.99, 0.95
    last_values = np.array([[0.11]], np.float32)  # bootstrap for the final non-boundary step
    buf.compute_gae(last_values, gamma, lam)

    # hand computation
    d2 = rewards[2] + gamma * last_values[0, 0] - values[2]
    a2 = d2
    d1 = rewards[1] + gamma * term_value - values[1]  # boundary: bootstrap with term_value
    a1 = d1  # trace cut (nonterminal = 0)
    d0 = rewards[0] + gamma * values[1] - values[0]
    a0 = d0 + gamma * lam * a1

    adv = buf.advantages[:, 0, 0]
    np.testing.assert_allclose(adv, [a0, a1, a2], rtol=1e-5)
    np.testing.assert_allclose(buf.returns[:, 0, 0], np.array([a0, a1, a2]) + np.array(values), rtol=1e-5)

    # the boundary must NOT be treated as a true terminal (that would give d1 = r1 - v1)
    assert not np.isclose(a1, rewards[1] - values[1])


def test_mappo_update_runs():
    from experiments.mappo.config import MAPPOConfig
    from experiments.mappo.mappo import MAPPO

    W, A, O = 2, 4, 11
    cfg = MAPPOConfig(n_steps=6, n_workers=W, n_minibatches=2, n_epochs=2)
    sd = state_dim(O, A, "centralized", False)
    mappo = MAPPO(O, 2, sd, cfg, torch.device("cpu"))
    buf = RolloutBuffer(cfg.n_steps, W, A, O, sd, torch.device("cpu"))
    rng = np.random.default_rng(0)
    for t in range(cfg.n_steps):
        obs = rng.random((W, A, O)).astype(np.float32)
        st = build_state(obs, None, A, "centralized", False)
        af, lf, vf = mappo.act(obs, st)
        buf.add(obs=obs, states=st, actions=af.reshape(W, A), logprobs=lf.reshape(W, A),
                values=vf.reshape(W, A), rewards=rng.standard_normal((W, A)).astype(np.float32),
                dones=np.zeros(W, np.float32))
    buf.compute_gae(np.zeros((W, A), np.float32), cfg.gamma, cfg.gae_lambda)
    stats = mappo.update(buf, cfg.lr)
    for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "grad_norm"):
        assert np.isfinite(stats[k]), f"{k} not finite: {stats[k]}"


@pytest.mark.skipif("SUMO_HOME" not in os.environ, reason="needs SUMO")
def test_smoke_train(tmp_path):
    """2 workers, tiny budget, one eval -- checks the whole pipeline wires up."""
    from experiments import train_mappo

    argv = [
        "train_mappo", "--scenario", "2x2", "--seed", "0",
        "--n-workers", "2", "--n-steps", "8", "--num-seconds", "200",
        "--total-steps", str(8 * 2 * 4 * 2), "--n-epochs", "2", "--n-minibatches", "2",
        "--eval-every", "1", "--n-eval-seeds", "2", "--n-eval-workers", "2",
        "--save-every", "1", "--out-root", str(tmp_path), "--sumo-backend", "traci",
    ]
    old = sys.argv
    sys.argv = argv
    try:
        train_mappo.main()
    finally:
        sys.argv = old

    runs = list((tmp_path / "2x2").glob("mappo_*"))
    assert runs, "no run dir created"
    rd = runs[0]
    assert (rd / "config.json").is_file()
    assert (rd / "train_log.csv").is_file()
    assert (rd / "eval_log.csv").is_file()
    assert (rd / "checkpoints" / "last.pt").is_file()
    import pandas as pd

    tl = pd.read_csv(rd / "train_log.csv")
    assert len(tl) >= 1
    assert np.isfinite(tl["policy_loss"]).all()
    assert (tmp_path / "2x2" / "fixed_time_summary.json").is_file()
