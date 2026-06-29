"""Launch the peloton visualization.

Run with:  uv run solara run run_app.py
"""

from mesa.visualization import SolaraViz

from peloton.viz import (
    BreakawaysPlot,
    GroupsPlot,
    RoadViewLargestPack,
    RoadViewTop3,
    StaminaPlot,
    build_model,
    model_params,
)

model = build_model()

page = SolaraViz(
    model,
    components=[RoadViewTop3, RoadViewLargestPack, StaminaPlot, GroupsPlot, BreakawaysPlot],
    model_params=model_params,
    name="Cycling Peloton MVP",
)
