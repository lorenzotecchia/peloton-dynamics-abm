"""Launch the peloton visualization.

Run with:  uv run solara run run_app.py
"""

from mesa.visualization import SolaraViz

from peloton.viz import (
    ExposurePlot,
    FinishedPlot,
    RoadView,
    build_model,
    model_params,
)

model = build_model()

page = SolaraViz(
    model,
    components=[RoadView, ExposurePlot, FinishedPlot],
    model_params=model_params,
    name="Cycling Peloton MVP",
)
