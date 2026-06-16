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

    def __init__(self, config: PelotonConfig | None = None, *, scenario=None, **overrides):
        # SolaraViz's reset injects a `scenario=` kwarg (Mesa's experimental
        # scenarios feature). We don't use scenarios, so consume and ignore it
        # here rather than let it reach _resolve_config, which strictly rejects
        # unknown keys (that guard still catches genuine slider-name typos).
        config = self._resolve_config(config, overrides)
        super().__init__(seed=config.seed)
        self.config = config
        self.n_finished = 0

        # Mesa's ContinuousSpace treats x_max/y_max as exclusive (out_of_bounds uses
        # coord >= max), so pad by a small epsilon to make road_length itself legal.
        self.space = ContinuousSpace(
            config.road_length + 1e-6, config.road_width + 1e-6, torus=False
        )

        self.finish_order: list[tuple[int, int]] = []

        # Spawn on a start grid with fixed clearance: non-overlapping by
        # construction. Jitter stays strictly under half the clearance so it
        # can never close the gap between neighbouring slots.
        gap = 0.2
        slot_w = config.rider_width + gap
        slot_l = config.rider_length + gap
        per_row = max(1, int(config.road_width // slot_w))
        jitter = gap / 2 - 0.01
        for i in range(config.n_agents):
            agent = CyclistAgent(self, team_id=i % config.n_teams)
            row, col = divmod(i, per_row)
            x = row * slot_l + self.random.uniform(0.0, jitter)
            y = col * slot_w + slot_w / 2 + self.random.uniform(-jitter, jitter)
            self.space.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "MeanExposure": _mean_exposure,
                "Finished": lambda m: m.n_finished,
            }
        )
        self.datacollector.collect(self)

    @staticmethod
    def _resolve_config(config: PelotonConfig | None, overrides: dict) -> PelotonConfig:
        """Build a config, applying any keyword overrides (used by SolaraViz sliders)."""
        base = config or PelotonConfig()
        if not overrides:
            return base
        fields = {
            "road_length": base.road_length,
            "road_width": base.road_width,
            "n_agents": base.n_agents,
            "n_teams": base.n_teams,
            "base_speed": base.base_speed,
            "speed_noise": base.speed_noise,
            "draft_radius": base.draft_radius,
            "draft_lateral": base.draft_lateral,
            "rider_length": base.rider_length,
            "rider_width": base.rider_width,
            "seed": base.seed,
        }
        for key, value in overrides.items():
            if key not in fields:
                raise TypeError(f"Unknown model parameter: {key!r}")
            if key in ("n_agents", "n_teams"):
                value = int(value)
            elif key != "seed":
                value = float(value)
            fields[key] = value
        return PelotonConfig(**fields)

    def step(self):
        self.agents.shuffle_do("step")
        self._remove_finishers()
        self.datacollector.collect(self)

    def _remove_finishers(self):
        """Riders that crossed the line leave the road (and stop blocking it).

        Same-step finishers are appended in agent-registration order, so ties in
        ``finish_order`` carry no ranking — a sprint-finish model must resolve
        them properly.
        """
        for agent in list(self.agents):
            if agent.pos[0] >= self.config.road_length:
                self.finish_order.append((agent.unique_id, self.steps))
                self.space.remove_agent(agent)
                agent.remove()
        self.n_finished = len(self.finish_order)
        if not len(self.agents):
            self.running = False        # race over: stop the viz autoplay
