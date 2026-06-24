"""SolaraViz wiring: two road views and metric plots.

RoadViewTop3      — tracks rank-1, 2, 3 agents; zooms out when they are spread apart.
RoadViewLargestPack — tracks the largest group; highlights the current puller.
"""

import colorsys
import math

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

CAMERA_MARGIN     = 30.0   # padding (m) on each side of the tracked span
MIN_CAMERA_WINDOW = 150.0  # minimum window width (m)
DASH_PITCH        = 10.0   # world-metres between dash starts on the centre line
DASH_LEN          = 4.0    # length of each centre-line dash (metres)


def rider_color(team_id: int, n_teams: int, exposure: float) -> tuple[float, float, float]:
    """Hue from team, brightness from wind exposure (sheltered=dark, exposed=bright)."""
    hue   = team_id / max(n_teams, 1)
    value = 0.45 + 0.55 * max(0.0, min(1.0, exposure))
    return colorsys.hsv_to_rgb(hue, 0.85, value)


def _camera(x_vals: list[float]) -> tuple[float, float]:
    """Return (x_lo, x_hi) that spans x_vals with margin and minimum width."""
    x_lo = min(x_vals) - CAMERA_MARGIN
    x_hi = max(x_vals) + CAMERA_MARGIN
    if x_hi - x_lo < MIN_CAMERA_WINDOW:
        mid  = (x_lo + x_hi) / 2
        x_lo = mid - MIN_CAMERA_WINDOW / 2
        x_hi = mid + MIN_CAMERA_WINDOW / 2
    return x_lo, x_hi


def _render_road(model, ax, x_lo: float, x_hi: float, title: str,
                 puller_id=None, pack_front: float | None = None,
                 label_agents: dict | None = None):
    """Shared road renderer.

    puller_id    — unique_id of the pack puller; drawn at pack_front with brightness=1
    pack_front   — x position of pack nose (required when puller_id is set)
    label_agents — {unique_id: label_str} agents to annotate with a text label
    """
    cfg    = model.config
    agents = list(model.agents)

    ax.set_title(title, fontsize=9, pad=2)
    ax.axhspan(0.0, cfg.road_width, color="#9e9e9e")
    ax.set_ylim(-1.0, cfg.road_width + 1.0)
    ax.set_yticks([])
    ax.set_xlabel("distance (m)", fontsize=8)
    ax.set_xlim(x_lo, x_hi)

    # Scrolling centre-line dashes
    y_mid = cfg.road_width / 2.0
    k0 = math.ceil(x_lo / DASH_PITCH)
    k1 = math.floor(x_hi / DASH_PITCH)
    for k in range(k0, k1 + 1):
        dash_x = k * DASH_PITCH
        ax.plot([dash_x, min(dash_x + DASH_LEN, x_hi)], [y_mid, y_mid],
                color="white", linewidth=1.0, solid_capstyle="butt")

    if x_lo <= cfg.road_length <= x_hi:
        ax.axvline(cfg.road_length, color="black", linestyle="--", linewidth=1)

    label_agents = label_agents or {}
    for agent in agents:
        is_puller = agent.unique_id == puller_id
        exposure  = 1.0 if is_puller else agent.exposure
        draw_pos  = (pack_front, agent.pos[1]) if is_puller else agent.pos
        ax.add_patch(Ellipse(
            draw_pos,
            width=cfg.rider_length,
            height=cfg.rider_width,
            facecolor=rider_color(agent.team_id, cfg.n_teams, exposure),
            edgecolor="white" if is_puller else "black",
            linewidth=0.3,
        ))
        if agent.unique_id in label_agents:
            ax.text(draw_pos[0], draw_pos[1] + cfg.rider_width * 0.9,
                    label_agents[agent.unique_id],
                    ha="center", va="bottom", fontsize=6, fontweight="bold",
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.55, lw=0))


def draw_road_top3(model, ax):
    """Camera tracks rank-1, 2, 3 agents (by position); zooms out when spread."""
    cfg    = model.config
    agents = list(model.agents)

    ax.axhspan(0.0, cfg.road_width, color="#9e9e9e")
    ax.set_ylim(-1.0, cfg.road_width + 1.0)

    if not agents:
        ax.set_xlim(0.0, cfg.road_length)
        ax.set_title("Top 3 riders", fontsize=9, pad=2)
        ax.text(0.5, 0.5, "race finished", transform=ax.transAxes,
                ha="center", va="center", fontsize=14)
        return

    by_x     = sorted(agents, key=lambda a: a.pos[0], reverse=True)
    top3     = by_x[:3]
    x_lo, x_hi = _camera([a.pos[0] for a in top3])

    labels = {a.unique_id: f"R{i+1}" for i, a in enumerate(top3)}
    n_shown = min(3, len(agents))
    title = f"Top {n_shown} riders  |  R1 gap: {top3[0].pos[0] - top3[-1].pos[0]:.0f} m"
    _render_road(model, ax, x_lo, x_hi, title, label_agents=labels)


def draw_road_largest_pack(model, ax):
    """Camera tracks the largest group; highlights the current puller."""
    cfg    = model.config
    agents = list(model.agents)

    ax.axhspan(0.0, cfg.road_width, color="#9e9e9e")
    ax.set_ylim(-1.0, cfg.road_width + 1.0)

    if not agents:
        ax.set_xlim(0.0, cfg.road_length)
        ax.set_title("Largest pack", fontsize=9, pad=2)
        ax.text(0.5, 0.5, "race finished", transform=ax.transAxes,
                ha="center", va="center", fontsize=14)
        return

    non_solo    = [a for a in agents if not a.solo and getattr(a, "break_cooldown", 0) == 0]
    pack_groups = [g for g in grp_mod.detect_groups(non_solo, cfg.group_radius) if len(g) > 1]

    if pack_groups:
        largest    = max(pack_groups, key=lambda g: len(g))
        pack_front = max(a.pos[0] for a in largest)
        pack_rear  = min(a.pos[0] for a in largest)
        puller     = max(largest, key=lambda a: a.exposure)
        puller_id  = puller.unique_id
        x_lo, x_hi = _camera([pack_rear, pack_front])
        title = f"Largest pack  |  {len(largest)} riders  |  puller: agent {puller_id}"
    else:
        leader     = max(agents, key=lambda a: a.pos[0])
        pack_front = leader.pos[0]
        puller_id  = None
        x_lo, x_hi = _camera([pack_front])
        title = "Largest pack  |  no group found"

    _render_road(model, ax, x_lo, x_hi, title,
                 puller_id=puller_id, pack_front=pack_front)


@solara.component
def RoadViewTop3(model):
    """Tracks rank-1, 2, 3 agents; zooms out when they are spread apart."""
    update_counter.get()
    fig = Figure(figsize=(10, 2.5))
    ax  = fig.add_subplot()
    draw_road_top3(model, ax)
    solara.FigureMatplotlib(fig)


@solara.component
def RoadViewLargestPack(model):
    """Tracks the largest group and highlights the current puller."""
    update_counter.get()
    fig = Figure(figsize=(10, 2.5))
    ax  = fig.add_subplot()
    draw_road_largest_pack(model, ax)
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


StaminaPlot    = make_plot_component("MeanStamina")
GroupsPlot     = make_plot_component("NumGroups")
BreakawaysPlot = make_plot_component("Breakaways")
