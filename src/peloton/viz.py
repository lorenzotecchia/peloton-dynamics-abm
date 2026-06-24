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

from peloton import group as grp_mod
from peloton.config import PelotonConfig
from peloton.model import PelotonModel

_cfg = PelotonConfig()

CAMERA_MARGIN     = 30.0   # padding (m) added on each side of the tracked span
MIN_CAMERA_WINDOW = 150.0  # minimum window width (m) so the view never collapses
DASH_PITCH = 10.0          # world-metres between dash starts on the centre line
DASH_LEN   = 4.0           # length of each centre-line dash (metres)


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

    # Leading pack: non-solo grouped riders, group with highest mean x and size > 1
    non_solo  = [a for a in agents if not a.solo and getattr(a, "break_cooldown", 0) == 0]
    pack_groups = [g for g in grp_mod.detect_groups(non_solo, model.config.group_radius)
                   if len(g) > 1]
    if pack_groups:
        lead_pack  = max(pack_groups, key=lambda g: sum(a.pos[0] for a in g) / len(g))
        pack_front = max(a.pos[0] for a in lead_pack)
        pack_rear  = min(a.pos[0] for a in lead_pack)
        puller     = max(lead_pack, key=lambda a: a.exposure)
        puller_id  = puller.unique_id
    else:
        pack_front = pack_rear = max(a.pos[0] for a in agents)
        puller_id  = None

    # Rank-1 rider: highest x of any agent still on the road
    rank1_x = max(a.pos[0] for a in agents)

    # Camera spans from the back of the leading pack to the rank-1 rider
    x_lo = pack_rear  - CAMERA_MARGIN
    x_hi = max(rank1_x, pack_front) + CAMERA_MARGIN
    if x_hi - x_lo < MIN_CAMERA_WINDOW:
        mid  = (x_lo + x_hi) / 2
        x_lo = mid - MIN_CAMERA_WINDOW / 2
        x_hi = mid + MIN_CAMERA_WINDOW / 2
    ax.set_xlim(x_lo, x_hi)

    # Scrolling centre-line: dashes at fixed world-x positions. As the window
    # follows the leader, they slide left under the riders.
    y_mid = cfg.road_width / 2.0
    import math
    k0 = math.ceil(x_lo / DASH_PITCH)
    k1 = math.floor(x_hi / DASH_PITCH)
    for k in range(k0, k1 + 1):
        dash_x = k * DASH_PITCH
        ax.plot(
            [dash_x, min(dash_x + DASH_LEN, x_hi)],
            [y_mid, y_mid],
            color="white", linewidth=1.0, solid_capstyle="butt",
        )

    if x_lo <= cfg.road_length <= x_hi:
        ax.axvline(cfg.road_length, color="black", linestyle="--", linewidth=1)

    for agent in agents:
        is_puller = agent.unique_id == puller_id
        exposure  = 1.0 if is_puller else agent.exposure
        # Draw the puller at the pack's nose so readers can track the rotation.
        draw_pos  = (pack_front, agent.pos[1]) if is_puller else agent.pos
        ax.add_patch(
            Ellipse(
                draw_pos,
                width=cfg.rider_length,
                height=cfg.rider_width,
                facecolor=rider_color(agent.team_id, cfg.n_teams, exposure),
                edgecolor="white" if is_puller else "black",
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
    "n_agents":     Slider("Number of riders",   value=_cfg.n_agents,     min=5,   max=100, step=5),
    "n_teams":      Slider("Number of teams",    value=_cfg.n_teams,      min=1,   max=10,  step=1),
    "k_s":          Slider("Pack speed coeff",   value=_cfg.k_s,          min=0.7, max=1.0, step=0.05),
    "group_radius": Slider("Group radius (m)",   value=_cfg.group_radius, min=1.0, max=6.0, step=0.5),
}


def build_model(n_agents=_cfg.n_agents, n_teams=_cfg.n_teams,
                k_s=_cfg.k_s, group_radius=_cfg.group_radius, config=None):
    """Factory used for the standalone app launch."""
    return PelotonModel(
        config=config,
        n_agents=n_agents,
        n_teams=n_teams,
        k_s=k_s,
        group_radius=group_radius,
    )


StaminaPlot = make_plot_component("MeanStamina")
GroupsPlot = make_plot_component("NumGroups")
BreakawaysPlot = make_plot_component("Breakaways")
