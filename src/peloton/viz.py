"""SolaraViz wiring and the exposure -> color gradient."""

from mesa.visualization import SolaraViz, make_plot_component, make_space_component
from mesa.visualization.user_param import Slider

from peloton.model import PelotonModel


def exposure_to_color(exposure: float) -> tuple[float, float, float]:
    """Map exposure in [0, 1] to an RGB tuple: green (sheltered) -> red (exposed)."""
    e = max(0.0, min(1.0, exposure))
    return (e, 1.0 - e, 0.0)            # (r, g, b)


def agent_portrayal(agent):
    return {
        "color": exposure_to_color(agent.exposure),
        "size": 25,
        "marker": "o",
    }


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


SpaceGraph = make_space_component(agent_portrayal)
ExposurePlot = make_plot_component("MeanExposure")
FinishedPlot = make_plot_component("Finished")
