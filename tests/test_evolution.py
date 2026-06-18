from peloton import evolution
from peloton.config import PelotonConfig


def test_evolve_is_noop_for_now():
    class _Rider:
        def __init__(self):
            self.coeffs = {"a": 1.0}
            self.utility = 0.0

    riders = [_Rider(), _Rider()]
    before = [dict(r.coeffs) for r in riders]
    evolution.evolve(riders, model=None)
    assert [r.coeffs for r in riders] == before     # stub changes nothing


def test_run_generations_runs_and_returns_history():
    cfg = PelotonConfig(n_agents=6, n_teams=2, road_length=60.0, seed=0)
    history = evolution.run_generations(n_generations=3, max_steps=50, config=cfg)
    assert len(history) == 3
    assert [h["generation"] for h in history] == [0, 1, 2]
    assert all("n_finished" in h for h in history)
