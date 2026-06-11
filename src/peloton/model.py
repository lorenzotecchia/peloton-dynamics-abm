"""The peloton model: space, agent spawning, stepping, and data collection."""

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import ContinuousSpace

from peloton.agent import CyclistAgent
from peloton.config import PelotonConfig


def _mean_exposure(model: "PelotonModel") -> float:
    agents = list(model.agents)
    if not agents:
        return 0.0
    return sum(a.exposure for a in agents) / len(agents)


class PelotonModel(Model):
    """A road full of cyclists that drift into drafting formations."""

    def __init__(self, config: PelotonConfig | None = None):
        config = config or PelotonConfig()
        super().__init__(seed=config.seed)
        self.config = config
        self.n_finished = 0

        # Mesa's ContinuousSpace treats x_max/y_max as exclusive (out_of_bounds uses
        # coord >= max), so pad by a small epsilon to make road_length itself legal.
        self.space = ContinuousSpace(
            config.road_length + 1e-6, config.road_width + 1e-6, torus=False
        )

        # Spawn agents round-robin into teams, spread just behind the start line.
        for i in range(config.n_agents):
            agent = CyclistAgent(self, team_id=i % config.n_teams)
            x = self.random.uniform(0.0, 5.0)               # small start spread
            y = self.random.uniform(0.0, config.road_width)
            self.space.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "MeanExposure": _mean_exposure,
                "Finished": lambda m: m.n_finished,
            }
        )
        self.datacollector.collect(self)

    def step(self):
        # Movement clamps forward position to road_length, so finishers simply pin
        # at the line and never leave the space.
        self.agents.shuffle_do("step")
        self.n_finished = sum(
            1 for a in self.agents if a.pos[0] >= self.config.road_length
        )
        self.datacollector.collect(self)
