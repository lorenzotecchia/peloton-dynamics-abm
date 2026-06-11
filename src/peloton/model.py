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

        # x_max is exclusive in Mesa's ContinuousSpace (out_of_bounds uses x >= x_max),
        # so add a small epsilon so that road_length itself is a valid coordinate.
        self.space = ContinuousSpace(
            config.road_length + 1e-6, config.road_width + 1e-6, torus=False
        )

        # Spawn agents round-robin into teams, spread just behind the start line.
        for i in range(config.n_agents):
            agent = CyclistAgent(self, team_id=i % config.n_teams)
            x = self.random.uniform(0.0, 5.0)               # small start spread
            y = self.random.uniform(0.0, config.road_width)
            self.space.place_agent(agent, (x, y))

        self._finished_ids: set[int] = set()

        # Wrap move_agent so that any position past road_length is clamped,
        # preventing ContinuousSpace from raising an out-of-bounds exception.
        _real_move = self.space.move_agent
        _road_length = config.road_length

        def _clamping_move(agent, pos):
            x, y = pos
            if x > _road_length:
                x = _road_length
            _real_move(agent, (x, y))

        self.space.move_agent = _clamping_move  # type: ignore[method-assign]

        self.datacollector = DataCollector(
            model_reporters={
                "MeanExposure": _mean_exposure,
                "Finished": lambda m: m.n_finished,
            }
        )
        self.datacollector.collect(self)

    def step(self):
        for agent in list(self.agents):
            if agent.unique_id in self._finished_ids:
                continue          # parked riders do not advance
            agent.step()
            x, y = agent.pos
            if x >= self.config.road_length:
                self._finished_ids.add(agent.unique_id)
        self.n_finished = len(self._finished_ids)
        self.datacollector.collect(self)
