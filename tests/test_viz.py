from peloton.viz import exposure_to_color, rider_color


def test_full_shelter_is_green_ish():
    r, g, b = exposure_to_color(0.0)
    assert g > r                       # sheltered -> green dominates

def test_full_exposure_is_red_ish():
    r, g, b = exposure_to_color(1.0)
    assert r > g                       # exposed -> red dominates

def test_color_channels_in_unit_range():
    for e in (0.0, 0.25, 0.5, 0.75, 1.0):
        r, g, b = exposure_to_color(e)
        for c in (r, g, b):
            assert 0.0 <= c <= 1.0


def test_rider_color_differs_by_team():
    c0 = rider_color(team_id=0, n_teams=5, exposure=0.5)
    c1 = rider_color(team_id=1, n_teams=5, exposure=0.5)
    assert c0 != c1                                  # different hue per team


def test_rider_color_brighter_when_exposed():
    sheltered = rider_color(team_id=2, n_teams=5, exposure=0.0)
    exposed = rider_color(team_id=2, n_teams=5, exposure=1.0)
    assert max(exposed) > max(sheltered)             # exposed = brighter


def test_rider_color_channels_in_unit_range():
    for team in range(5):
        for e in (0.0, 0.5, 1.0):
            c = rider_color(team_id=team, n_teams=5, exposure=e)
            assert len(c) == 3
            for ch in c:
                assert 0.0 <= ch <= 1.0


def test_rider_color_handles_zero_teams():
    c = rider_color(team_id=0, n_teams=0, exposure=0.5)   # must not divide by zero
    for ch in c:
        assert 0.0 <= ch <= 1.0


def _ellipses(ax):
    from matplotlib.patches import Ellipse
    return [p for p in ax.patches if isinstance(p, Ellipse)]


def test_draw_road_renders_one_true_scale_shape_per_rider():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from peloton.config import PelotonConfig
    from peloton.model import PelotonModel
    from peloton.viz import draw_road

    cfg = PelotonConfig(n_agents=12, n_teams=3, seed=5)
    model = PelotonModel(cfg)
    _, ax = plt.subplots()
    draw_road(model, ax)
    shapes = _ellipses(ax)
    assert len(shapes) == 12
    assert shapes[0].width == cfg.rider_length     # drawn in road metres,
    assert shapes[0].height == cfg.rider_width     # not screen points


def test_draw_road_camera_follows_the_peloton():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from peloton.config import PelotonConfig
    from peloton.model import PelotonModel
    from peloton.viz import draw_road, CAMERA_WINDOW, LEADER_MARGIN

    cfg = PelotonConfig(n_agents=20, n_teams=4, road_length=2000.0, seed=6)
    model = PelotonModel(cfg)
    for _ in range(20):
        model.step()
    _, ax = plt.subplots()
    draw_road(model, ax)

    xs = [a.pos[0] for a in model.agents]
    x_lo, x_hi = ax.get_xlim()
    assert abs((x_hi - x_lo) - CAMERA_WINDOW) < 1e-6        # fixed-width window
    assert abs(x_hi - (max(xs) + LEADER_MARGIN)) < 1e-6     # leader pinned near right
    assert x_hi - x_lo < 2000.0                             # not the whole road


def test_draw_road_handles_empty_race_without_crashing():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from peloton.config import PelotonConfig
    from peloton.model import PelotonModel
    from peloton.viz import draw_road

    cfg = PelotonConfig(n_agents=6, n_teams=2, road_length=40.0,
                        base_speed=12.0, speed_noise=0.0, seed=7)
    model = PelotonModel(cfg)
    for _ in range(8):
        model.step()
    assert len(model.agents) == 0                  # race is over
    _, ax = plt.subplots()
    draw_road(model, ax)                           # must not raise
    assert _ellipses(ax) == []
