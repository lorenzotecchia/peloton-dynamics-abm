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
    cfg = PelotonConfig(learning_rate=0.1, evo_noise=0.0, sim_scale=1.0)
    winner = _rider(uid=0, w_max10=400.0, alpha=2.0)    # finishes first
    loser = _rider(uid=1, w_max10=400.0, alpha=0.0)     # same engine, finishes last
    # finish_order: winner then loser -> winner has higher utility.
    evolution.evolve([winner, loser], _fake_model([(0, 5), (1, 9)], cfg))

    assert loser.coeffs["coop"]["alpha"] > 0.0          # moved toward winner's 2.0
    assert winner.coeffs["coop"]["alpha"] == 2.0        # nobody better -> unchanged


def test_evolve_assigns_dnf_lowest_utility():
    cfg = PelotonConfig(learning_rate=0.1, evo_noise=0.0)
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
