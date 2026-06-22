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
    cfg = PelotonConfig(n_agents=10, n_teams=2, road_length=50.0, seed=3)
    model = PelotonModel(cfg)
    for _ in range(20):
        model.step()
    assert model.n_finished == 10
    assert len(model.agents) == 0          # finishers leave the road
    finished_ids = [uid for uid, _ in model.finish_order]
    assert len(finished_ids) == 10
    assert len(set(finished_ids)) == 10    # each rider finishes exactly once


def test_datacollector_records_emergent_metrics():
    cfg = PelotonConfig(n_agents=15, n_teams=3, seed=4)
    model = PelotonModel(cfg)
    model.step()
    df = model.datacollector.get_model_vars_dataframe()
    for col in ("MeanStamina", "NumGroups", "Breakaways", "MeanExposure"):
        assert col in df.columns
    assert 0.0 <= df["MeanStamina"].iloc[-1] <= 1.0
    assert df["NumGroups"].iloc[-1] >= 1                 # at least one pack while racing
    assert 0.0 <= df["MeanExposure"].iloc[-1] <= 1.0


def test_model_accepts_keyword_overrides_for_viz():
    model = PelotonModel(n_agents=8, n_teams=2, k_s=0.9, group_radius=2.5)
    assert len(model.agents) == 8
    assert model.config.n_teams == 2
    assert model.config.k_s == 0.9
    assert model.config.group_radius == 2.5


def test_reset_with_injected_scenario_kwarg():
    """SolaraViz's reset reconstructs the model with a ``scenario=`` kwarg (Mesa's
    experimental scenarios feature). PelotonModel doesn't use scenarios, but the
    constructor must not choke on the injected kwarg. Reproduces the reset-button
    ``TypeError: Unknown model parameter: 'scenario'``.
    """
    model = PelotonModel(n_agents=10, n_teams=2)
    # Mirror mesa.visualization.solara_viz.do_reset: the model's own scenario is
    # passed back into a fresh instance alongside the slider params.
    reset = PelotonModel(scenario=model.scenario, n_agents=15, n_teams=3)
    assert len(reset.agents) == 15
    assert reset.config.n_teams == 3


def test_resolve_config_preserves_rider_footprint():
    base = PelotonConfig(rider_length=2.5, rider_width=0.9)
    model = PelotonModel(config=base, n_agents=6)
    assert model.config.rider_length == 2.5
    assert model.config.rider_width == 0.9


def test_breakaways_occur_with_attack_prone_coeffs():
    from peloton import strategy

    cfg = PelotonConfig(n_agents=20, n_teams=2, road_length=600.0, seed=11)
    # Strong attack disposition: big speed weight, sharp sustain, low caution.
    pop = []
    for _ in range(cfg.n_agents):
        c = strategy.default_coeffs()
        c["weights"] = {"lambda_speed": 30.0, "lambda_energy": 0.5,
                        "lambda_risk": 0.0, "beta_loss": 1.0}
        c["sustain"] = {"bias": 2.0, "k_gap": 2.0}
        pop.append(c)
    model = PelotonModel(cfg, population=pop)
    seen_solo = False
    for _ in range(60):
        if not model.running:
            break
        model.step()
        if any(a.solo for a in model.agents):
            seen_solo = True
    assert seen_solo                              # at least one rider went off the front


def test_caught_attacker_rejoins_nearby_group():
    # A rider flagged as a breakaway that is back inside a pack (and not off the
    # front) must rejoin and draft, not stay locked solo.
    cfg = PelotonConfig(n_agents=3, n_teams=1, road_length=2000.0, seed=5)
    model = PelotonModel(cfg)
    for i, a in enumerate(sorted(model.agents, key=lambda a: a.unique_id)):
        model.space.move_agent(a, (100.0 + i * 0.5, 4.0))   # tight bunch, all < 3 m apart
    back = min(model.agents, key=lambda a: a.pos[0])         # not the frontmost
    back.solo = True
    model.step()
    assert back.solo is False                               # reabsorbed -> drafting again


def test_race_stops_running_when_everyone_finished():
    cfg = PelotonConfig(n_agents=6, n_teams=2, road_length=40.0, seed=8)
    model = PelotonModel(cfg)
    assert model.running
    for _ in range(15):
        model.step()
    assert len(model.agents) == 0
    assert not model.running                       # autoplay stops at race end
