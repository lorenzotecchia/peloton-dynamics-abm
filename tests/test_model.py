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


def test_finishers_are_counted_and_parked():
    cfg = PelotonConfig(n_agents=10, n_teams=2, road_length=50.0,
                        base_speed=12.0, speed_noise=0.0, seed=3)
    model = PelotonModel(cfg)
    for _ in range(10):                                # 10*12 = 120 m > 50 m road
        model.step()
    assert model.n_finished == 10
    for a in model.agents:
        assert a.pos[0] >= cfg.road_length             # parked at/after the line


def test_datacollector_records_mean_exposure():
    cfg = PelotonConfig(n_agents=15, n_teams=3, seed=4)
    model = PelotonModel(cfg)
    model.step()
    df = model.datacollector.get_model_vars_dataframe()
    assert "MeanExposure" in df.columns
    assert "Finished" in df.columns
    assert 0.0 <= df["MeanExposure"].iloc[-1] <= 1.0
