from peloton.config import PelotonConfig


def test_defaults_are_sane():
    cfg = PelotonConfig()
    assert cfg.road_length > cfg.road_width      # road is long and thin
    assert cfg.n_agents >= cfg.n_teams           # at least one rider per team
    assert cfg.draft_radius > 0
    assert cfg.draft_lateral > 0


def test_is_frozen():
    cfg = PelotonConfig()
    try:
        cfg.n_agents = 5
        raised = False
    except Exception:
        raised = True
    assert raised, "PelotonConfig should be immutable (frozen)"


def test_rider_footprint_defaults():
    cfg = PelotonConfig()
    assert cfg.rider_length > cfg.rider_width      # bikes are long and narrow
    assert cfg.rider_width > 0
