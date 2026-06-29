import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from peloton.config import PelotonConfig
from peloton.model import PelotonModel
from peloton.viz import rider_color, draw_road_top3, draw_road_largest_pack


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
    return [p for p in ax.patches if isinstance(p, Ellipse)]


def test_camera_views_render_ellipses_and_zoom_in():
    cfg = PelotonConfig(n_agents=12, n_teams=3, road_length=2000.0, seed=5)
    model = PelotonModel(cfg)
    for view in (draw_road_top3, draw_road_largest_pack):
        _, ax = plt.subplots()
        view(model, ax)
        assert _ellipses(ax)                         # something is drawn
        x_lo, x_hi = ax.get_xlim()
        assert x_hi - x_lo < 2000.0                   # camera zoomed, not whole road


def test_camera_views_handle_empty_race_without_crashing():
    cfg = PelotonConfig(n_agents=6, n_teams=2, road_length=40.0, seed=7)
    model = PelotonModel(cfg)
    for _ in range(30):
        if not model.running:
            break
        model.step()
    assert len(model.agents) == 0                     # race is over
    for view in (draw_road_top3, draw_road_largest_pack):
        _, ax = plt.subplots()
        view(model, ax)                               # must not raise
        assert _ellipses(ax) == []
