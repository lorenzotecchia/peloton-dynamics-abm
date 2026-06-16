"""SolaraViz wiring: a true-scale road renderer and the exposure gradient.

Mesa's generic scatter view draws fixed-size screen dots, which mash together
on a 1000 m axis and crash on an empty space, so the road view is custom: riders
are bike-sized ellipses in road coordinates and the camera follows the bunch.
"""

import colorsys

import solara
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from mesa.visualization import SolaraViz, make_plot_component
from mesa.visualization.user_param import Slider
from mesa.visualization.utils import update_counter

from peloton.model import PelotonModel

_CAMERA_MARGIN = 10.0   # metres of road shown around the bunch
_MIN_WINDOW = 60.0      # never zoom tighter than this many metres


def exposure_to_color(exposure: float) -> tuple[float, float, float]:
    """Map exposure in [0, 1] to an RGB tuple: green (sheltered) -> red (exposed)."""
    e = max(0.0, min(1.0, exposure))
    return (e, 1.0 - e, 0.0)            # (r, g, b)


def rider_color(team_id: int, n_teams: int, exposure: float) -> tuple[float, float, float]:
    """Rider fill color: hue from team, brightness from wind exposure.

    Hue is spread across teams so groups are distinguishable; HSV ``value`` rises
    with exposure (sheltered riders are darker, exposed riders brighter), keeping
    the drafting signal the model actually produces today.
    """
    hue = team_id / max(n_teams, 1)
    value = 0.45 + 0.55 * max(0.0, min(1.0, exposure))
    return colorsys.hsv_to_rgb(hue, 0.85, value)


def draw_road(model, ax):
    """Render the road strip with riders at true physical scale.

    Safe for an empty race: once every rider has finished, the full road is
    shown with a banner instead of crashing on a zero-agent space.
    """
    cfg = model.config
    agents = list(model.agents)

    ax.axhspan(0.0, cfg.road_width, color="#9e9e9e")    # tarmac
    ax.set_ylim(-1.0, cfg.road_width + 1.0)
    ax.set_yticks([])
    ax.set_xlabel("distance (m)")

    if not agents:
        ax.set_xlim(0.0, cfg.road_length)
        ax.text(
            0.5, 0.5, "race finished",
            transform=ax.transAxes, ha="center", va="center", fontsize=14,
        )
        return

    xs = [a.pos[0] for a in agents]
    x_lo = min(xs) - _CAMERA_MARGIN
    x_hi = max(xs) + _CAMERA_MARGIN
    if x_hi - x_lo < _MIN_WINDOW:
        pad = (_MIN_WINDOW - (x_hi - x_lo)) / 2
        x_lo, x_hi = x_lo - pad, x_hi + pad
    ax.set_xlim(x_lo, x_hi)

    if x_lo <= cfg.road_length <= x_hi:
        ax.axvline(cfg.road_length, color="black", linestyle="--", linewidth=1)

    for agent in agents:
        ax.add_patch(
            Ellipse(
                agent.pos,
                width=cfg.rider_length,
                height=cfg.rider_width,
                facecolor=exposure_to_color(agent.exposure),
                edgecolor="black",
                linewidth=0.3,
            )
        )


@solara.component
def RoadView(model):
    """Solara component wrapping :func:`draw_road`; re-renders every model step."""
    update_counter.get()
    fig = Figure(figsize=(10, 2.5))
    ax = fig.add_subplot()
    draw_road(model, ax)
    solara.FigureMatplotlib(fig)


model_params = {
    "n_agents": Slider("Number of riders", value=30, min=5, max=100, step=5),
    "n_teams": Slider("Number of teams", value=5, min=1, max=10, step=1),
    "base_speed": Slider("Base speed", value=12.0, min=4.0, max=20.0, step=1.0),
    "draft_radius": Slider("Draft radius (m)", value=3.0, min=1.0, max=6.0, step=0.5),
}


def build_model(n_agents=30, n_teams=5, base_speed=12.0, draft_radius=3.0, config=None):
    """Factory used for the standalone app launch."""
    return PelotonModel(
        config=config,
        n_agents=n_agents,
        n_teams=n_teams,
        base_speed=base_speed,
        draft_radius=draft_radius,
    )


ExposurePlot = make_plot_component("MeanExposure")
FinishedPlot = make_plot_component("Finished")
