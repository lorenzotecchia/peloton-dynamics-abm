"""Launch the peloton visualization.

Run with:  uv run solara run run_app.py
"""

from mesa.visualization import SolaraViz

from peloton.viz import (
    BreakawaysPlot,
    GroupsPlot,
    RoadView,
    StaminaPlot,
    build_model,
    model_params,
)

import os
import json

# If a trained population was saved by the learning run, load it and pass it
# to the model so the Solara app visualizes that exact population.
population = None
pop_path = "population.json"
if os.path.exists(pop_path):
    try:
        with open(pop_path) as fh:
            population = json.load(fh)
    except Exception:
        population = None

model = build_model(population=population)

page = SolaraViz(
    model,
    components=[RoadView, StaminaPlot, GroupsPlot, BreakawaysPlot],
    model_params=model_params,
    name="Cycling Peloton MVP",
)
