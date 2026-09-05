"""Tests for the multi-agent DQN in experiments/dqn/.

T1/T2/T5 are pure-numpy and always run. T3 needs torch. T4 needs SUMO (SUMO_HOME).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.dqn.replay import JointReplay, NStepAccumulator, PrioritizedJointReplay, _SumTree  # noqa: E402


# --------------------------------------------------------------------------- T1
def test_nstep_boundary_bootstraps_on_terminal():
    """n=3 window, boundary after reward index 2 -> truncated m<n items bootstrapping
    off the terminal obs, deque cleared, nothing mixes across the episode boundary."""
    n, gamma = 3, 0.9
    acc = NStepAccumulator(n_workers=1, n_step=n, gamma=gamma)
    rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
    TERM = np.full((1, 2), -99.0, np.float32)  # sentinel terminal obs (A=1, O=2)
    NORMAL = lambda t: np.full((1, 2), t, np.float32)

    emitted = []
    for t, r in enumerate(rewards):
        done = t == 2
        nxt = TERM if done else NORMAL(t + 1)
        emitted += acc.add(0, obs=NORMAL(t), action=np.array([0]), reward=np.array([r], np.float32),
                           next_obs=nxt, done=done)

    assert len(emitted) == 3, "boundary must drain the whole deque"
    r0, r1, r2 = emitted
    np.testing.assert_allclose(r0["R"], [1 + gamma * 2 + gamma**2 * 3], rtol=1e-6)
    np.testing.assert_allclose(r1["R"], [2 + gamma * 3], rtol=1e-6)
    np.testing.assert_allclose(r2["R"], [3.0], rtol=1e-6)
    np.testing.assert_allclose([r0["gamma_m"], r1["gamma_m"], r2["gamma_m"]],
                               [gamma**3, gamma**2, gamma**1], rtol=1e-6)
    for it in emitted:
        assert np.array_equal(it["next_obs"], TERM), "must bootstrap off the terminal obs, not next episode"
        assert it["gamma_m"] > 0  # time-limit -> always bootstrap (done_flag is structurally 0)
    # r3 / r4 belong to the next episode and must not leak into r0
    assert float(r0["R"][0]) < 6.0


def test_nstep_steady_state_full_window():
    n, gamma = 3, 0.95
    acc = NStepAccumulator(1, n, gamma)
    emitted = []
    for t in range(4):  # no done: emits start once the deque first reaches n, then one per add
        emitted += acc.add(0, obs=np.full((1, 1), t, np.float32), action=np.array([1]),
                           reward=np.array([1.0], np.float32),
                           next_obs=np.full((1, 1), t + 1, np.float32), done=False)
    assert len(emitted) == 2  # emissions on the 3rd and 4th add
    for k, e in enumerate(emitted):
        np.testing.assert_allclose(e["R"], [1 + gamma + gamma**2], rtol=1e-6)
        np.testing.assert_allclose(e["gamma_m"], gamma**3, rtol=1e-6)
        # bootstrap obs is a normal next-obs (position k+n-1's next), never a terminal
        assert np.array_equal(e["next_obs"], np.full((1, 1), float(k + n), np.float32))


# --------------------------------------------------------------------------- T2
def test_sumtree_proportional_and_retrieve():
    tree = _SumTree(8)
    prios = np.arange(1, 9, dtype=float)
    for i, p in enumerate(prios):
        tree.update(i, p)
    assert tree.total() == prios.sum()
    rng = np.random.default_rng(0)
    counts = np.zeros(8)
    for _ in range(40000):
        counts[tree.retrieve(rng.uniform(0, tree.total()))] += 1
    freq = counts / counts.sum()
    np.testing.assert_allclose(freq, prios / prios.sum(), atol=0.02)


def test_per_sampling_and_is_weights():
    rng = np.random.default_rng(0)
    rb = PrioritizedJointReplay(64, n_agents=1, obs_dim=2, state_dim=0, rng=rng, alpha=0.6)
    items = [dict(obs=np.zeros((1, 2), np.float32), action=np.zeros(1, np.int64), R=np.zeros(1, np.float32),
                  next_obs=np.zeros((1, 2), np.float32), gamma_m=np.float32(0.9)) for _ in range(64)]
    rb.add_batch(items)
    rb.update_priorities(np.arange(64), np.arange(1, 65).astype(float))

    counts = np.zeros(64)
    for _ in range(400):
        b = rb.sample(64, beta=1.0)
        for i in b["idx"]:
            counts[i] += 1
        assert np.isclose(b["weights"].max(), 1.0)  # normalized by batch max
    # higher priority -> sampled more, and lower IS weight
    assert counts[63] > counts[0] * 3
    b = rb.sample(64, beta=1.0)
    hi = b["weights"][b["idx"] == b["idx"].max()]
    lo = b["weights"][b["idx"] == b["idx"].min()]
    if len(hi) and len(lo):
        assert hi.mean() <= lo.mean() + 1e-6


def test_uniform_replay_is_fifo():
    rb = JointReplay(8, n_agents=1, obs_dim=1, state_dim=0, rng=np.random.default_rng(0))
    for t in range(12):
        rb.add_batch([dict(obs=np.full((1, 1), t, np.float32), action=np.zeros(1, np.int64),
                           R=np.zeros(1, np.float32), next_obs=np.zeros((1, 1), np.float32),
                           gamma_m=np.float32(0.9))])
    assert len(rb) == 8
    assert sorted(rb.obs[:, 0, 0].tolist()) == [4, 5, 6, 7, 8, 9, 10, 11]  # oldest 4 evicted


# --------------------------------------------------------------------------- T3
def test_agent_update_all_mixers():
    torch = pytest.importorskip("torch")
    from experiments.dqn.config import DQNConfig
    from experiments.dqn.dqn import DQNAgent, epsilon_ladder
    from experiments.dqn.networks import QMixer
    from experiments.dqn.state import global_state_dim

    O, ACT, AG, B = 6, 3, 4, 16
    sd = global_state_dim(O, AG, False)
    rng = np.random.default_rng(0)

    def mk_batch(with_state):
        d = dict(
            obs=rng.random((B, AG, O)).astype(np.float32),
            action=rng.integers(0, ACT, (B, AG)).astype(np.int64),
            R=rng.standard_normal((B, AG)).astype(np.float32),
            next_obs=rng.random((B, AG, O)).astype(np.float32),
            gamma_m=np.full(B, 0.9**3, np.float32),
            weights=np.ones(B, np.float32),
            idx=np.arange(B),
        )
        if with_state:
            d["gstate"] = rng.random((B, sd)).astype(np.float32)
            d["next_gstate"] = rng.random((B, sd)).astype(np.float32)
        return d

    for mixer in ("none", "vdn", "qmix"):
        cfg = DQNConfig(scenario="2x2", mixer=mixer, hidden_dim=32, n_hidden=1, per=False, target_tau=0.1)
        agent = DQNAgent(O, ACT, AG, sd, cfg, torch.device("cpu"))
        tgt_before = torch.cat([p.flatten() for p in agent.target_net.parameters()]).clone()
        st = agent.update(mk_batch(mixer == "qmix"))
        for k in ("loss", "q_mean", "q_target_mean", "td_error_abs_mean", "grad_norm"):
            assert np.isfinite(st[k]), f"{mixer}: {k}={st[k]}"
        assert st["td_abs"].shape == (B,)
        tgt_after = torch.cat([p.flatten() for p in agent.target_net.parameters()])
        assert not torch.equal(tgt_before, tgt_after), f"{mixer}: Polyak target did not move"

    # QMIX monotonicity: raising any single agent's Q must not lower Q_tot (abs mixing weights)
    mix = QMixer(AG, sd, 8, 16)
    q = torch.randn(5, AG)
    s = torch.randn(5, sd)
    q_up = q.clone()
    q_up[:, 1] += 1.0
    assert (mix(q_up, s) >= mix(q, s) - 1e-4).all()

    # Double-DQN wiring runs both ways
    for dbl in (True, False):
        cfg = DQNConfig(scenario="2x2", mixer="none", hidden_dim=16, n_hidden=1, per=False, double=dbl)
        DQNAgent(O, ACT, AG, sd, cfg, torch.device("cpu")).update(mk_batch(False))


# --------------------------------------------------------------------------- T5
def test_epsilon_ladder_and_action_mask_determinism():
    from experiments.dqn.dqn import epsilon_ladder

    lad = epsilon_ladder(8, 0.4, 7.0, 0.01)
    assert lad.shape == (8,)
    assert abs(lad[0] - 0.4) < 1e-9
    assert np.all(np.diff(lad) <= 1e-12)  # monotone decreasing
    assert lad.min() >= 0.01 and lad.max() <= 1.0
    assert np.allclose(epsilon_ladder(1, 0.4, 7.0, 0.01), [0.4])

    torch = pytest.importorskip("torch")
    from experiments.dqn.config import DQNConfig
    from experiments.dqn.dqn import DQNAgent

    agent = DQNAgent(6, 3, 4, 24, DQNConfig(scenario="2x2", mixer="none", hidden_dim=16, n_hidden=1), torch.device("cpu"))
    obs = np.random.RandomState(0).rand(8, 4, 6).astype(np.float32)
    a1, _ = agent.act(obs, lad, np.random.default_rng(0))
    a2, _ = agent.act(obs, lad, np.random.default_rng(0))
    assert np.array_equal(a1, a2)


# --------------------------------------------------------------------------- T4
@pytest.mark.skipif("SUMO_HOME" not in os.environ, reason="needs SUMO")
@pytest.mark.parametrize("extra", [["--mixer", "qmix"], ["--mixer", "none", "--no-per"]])
def test_smoke_train(tmp_path, extra):
    pytest.importorskip("torch")
    from experiments import train_dqn

    argv = [
        "train_dqn", "--scenario", "2x2", "--seed", "0",
        "--n-workers", "2", "--collect-steps", "4", "--total-steps", "640",
        "--learning-starts", "64", "--num-seconds", "200", "--batch-size", "16",
        "--replay-capacity", "800", "--replay-ratio", "2",
        "--eval-every-steps", "200", "--save-every-steps", "200",
        "--n-eval-seeds", "2", "--n-eval-workers", "2",
        "--out-root", str(tmp_path), "--sumo-backend", "traci",
        *extra,
    ]
    old = sys.argv
    sys.argv = argv
    try:
        train_dqn.main()
    finally:
        sys.argv = old

    import pandas as pd

    runs = list((tmp_path / "2x2").glob("dqn_*"))
    assert runs, "no run dir created"
    rd = runs[0]
    for f in ("config.json", "train_log.csv", "eval_log.csv"):
        assert (rd / f).is_file(), f
    assert (rd / "checkpoints" / "last.pt").is_file()
    assert (tmp_path / "2x2" / "fixed_time_summary.json").is_file()
    tl = pd.read_csv(rd / "train_log.csv")
    assert "loss" in tl.columns
    losses = pd.to_numeric(tl["loss"], errors="coerce").dropna()
    assert len(losses) >= 1 and np.isfinite(losses.to_numpy()).all()
