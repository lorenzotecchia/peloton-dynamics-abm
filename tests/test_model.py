from peloton.config import PelotonConfig
from peloton.model import PelotonModel


def test_model_spawns_all_agents_across_teams():
    cfg = PelotonConfig(n_agents=12, n_teams=4, seed=1)
    model = PelotonModel(cfg)
    assert len(model.agents) == 12
    teams = {a.team_id for a in model.agents}
    assert teams == {0, 1, 2, 3}


def test_agents_advance_and_stay_in_road_bounds():
    cfg = PelotonConfig(n_agents=20, n_teams=5, road_length=300.0, seed=2)
    model = PelotonModel(cfg)
    start_x = {a.unique_id: a.pos[0] for a in model.agents}
    for _ in range(10):
        model.step()
    for a in model.agents:
        assert a.pos[0] >= start_x[a.unique_id]        # never moved backward overall
        assert 0.0 <= a.pos[1] <= cfg.road_width       # stays on the road laterally


def test_finishers_are_removed_and_counted():
    cfg = PelotonConfig(n_agents=10, n_teams=2, road_length=50.0,
                        base_speed=12.0, speed_noise=0.0, seed=3)
    model = PelotonModel(cfg)
    for _ in range(10):
        model.step()
    assert model.n_finished == 10
    assert len(model.agents) == 0          # finishers leave the road
    finished_ids = [uid for uid, _ in model.finish_order]
    assert len(finished_ids) == 10
    assert len(set(finished_ids)) == 10    # each rider finishes exactly once


def test_datacollector_records_mean_exposure():
    cfg = PelotonConfig(n_agents=15, n_teams=3, seed=4)
    model = PelotonModel(cfg)
    model.step()
    df = model.datacollector.get_model_vars_dataframe()
    assert "MeanExposure" in df.columns
    assert "Finished" in df.columns
    assert 0.0 <= df["MeanExposure"].iloc[-1] <= 1.0


def test_model_accepts_keyword_overrides_for_viz():
    model = PelotonModel(n_agents=8, n_teams=2, base_speed=10.0, draft_radius=2.5)
    assert len(model.agents) == 8
    assert model.config.n_teams == 2
    assert model.config.base_speed == 10.0
    assert model.config.draft_radius == 2.5


def test_resolve_config_preserves_rider_footprint():
    base = PelotonConfig(rider_length=2.5, rider_width=0.9)
    model = PelotonModel(config=base, n_agents=6)
    assert model.config.rider_length == 2.5
    assert model.config.rider_width == 0.9


def test_no_two_riders_ever_overlap():
    from peloton.physics import overlaps

    cfg = PelotonConfig(n_agents=30, n_teams=5, road_length=400.0, seed=11)
    model = PelotonModel(cfg)

    def assert_no_overlaps(step_no):
        agents = list(model.agents)
        for i, a in enumerate(agents):
            for b in agents[i + 1:]:
                assert not overlaps(
                    a.pos, b.pos,
                    rider_length=cfg.rider_length, rider_width=cfg.rider_width,
                ), f"step {step_no}: agents {a.unique_id} and {b.unique_id} overlap"

    assert_no_overlaps(0)                  # spawn grid is overlap-free
    for s in range(1, 16):
        model.step()
        assert_no_overlaps(s)


def test_race_stops_running_when_everyone_finished():
    cfg = PelotonConfig(n_agents=6, n_teams=2, road_length=40.0,
                        base_speed=12.0, speed_noise=0.0, seed=8)
    model = PelotonModel(cfg)
    assert model.running
    for _ in range(8):
        model.step()
    assert len(model.agents) == 0
    assert not model.running                       # autoplay stops at race end
