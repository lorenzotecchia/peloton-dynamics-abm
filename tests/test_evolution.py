import random
import types

from peloton import evolution
from peloton.config import PelotonConfig


def _fake_model(finish_order, cfg):
    return types.SimpleNamespace(
        config=cfg, random=random.Random(0), finish_order=finish_order
    )


def _rider(uid, w_max10, alpha):
    return types.SimpleNamespace(
        unique_id=uid, w_max10=w_max10,
        coeffs={"coop": {"alpha": alpha}}, utility=0.0,
    )


def test_evolve_pulls_loser_toward_similar_winner():
    cfg = PelotonConfig(evo_noise=0.0, logit_lambda=0.5)
    winner = _rider(uid=0, w_max10=400.0, alpha=2.0)    # finishes first
    loser = _rider(uid=1, w_max10=400.0, alpha=0.0)     # same engine, finishes last
    # finish_order: winner then loser -> winner has higher utility.
    evolution.evolve([winner, loser], _fake_model([(0, 5), (1, 9)], cfg))

    # Normalised update: a fraction eta of the way toward the winner, no overshoot.
    assert 0.0 < loser.coeffs["coop"]["alpha"] < 2.0
    assert winner.coeffs["coop"]["alpha"] == 2.0        # nobody better -> unchanged


def test_evolution_does_not_diverge_over_many_generations():
    cfg = PelotonConfig(n_agents=30, n_teams=5, road_length=300.0,
                        evo_noise=0.0, seed=2)
    history = evolution.run_generations(n_generations=30, max_steps=200, config=cfg)
    # Convex-combination update + zero noise => coefficients stay bounded (the
    # old unnormalised rule blew past 1e5 within ~8 generations).
    assert abs(history[-1]["coop.delta_mean"]) < 10.0
    assert history[-1]["coop.delta_std"] < 10.0


def test_evolve_assigns_dnf_lowest_utility():
    cfg = PelotonConfig(evo_noise=0.0, logit_lambda=1.0)
    finisher = _rider(uid=0, w_max10=400.0, alpha=1.0)
    dnf = _rider(uid=1, w_max10=400.0, alpha=0.0)
    evolution.evolve([finisher, dnf], _fake_model([(0, 7)], cfg))   # only rider 0 finished
    assert finisher.utility > dnf.utility
    assert dnf.utility == 0.0


def test_run_generations_runs_and_returns_history():
    cfg = PelotonConfig(n_agents=6, n_teams=2, road_length=60.0, seed=0)
    history = evolution.run_generations(n_generations=3, max_steps=80, config=cfg)
    assert len(history) == 3
    assert [h["generation"] for h in history] == [0, 1, 2]
    assert all("n_finished" in h for h in history)


def test_run_generations_records_coefficient_trajectories():
    cfg = PelotonConfig(n_agents=8, n_teams=2, road_length=60.0, seed=0)
    history = evolution.run_generations(n_generations=4, max_steps=80, config=cfg)
    # Mean/std recorded for every coefficient, ready to plot for convergence.
    assert "coop.delta_mean" in history[0]
    assert "coop.delta_std" in history[0]
    assert history[0]["coop.delta_std"] >= 0.0
