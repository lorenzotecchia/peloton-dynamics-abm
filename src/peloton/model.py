"""The peloton model: space, agent spawning, stepping, and data collection."""

from dataclasses import asdict

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

    def __init__(self, config: PelotonConfig | None = None, *, scenario=None, rng=None,
                 population=None, **overrides):
        # SolaraViz's reset injects a `scenario=` kwarg (Mesa's experimental
        # scenarios feature). We don't use scenarios, so consume and ignore it
        # here rather than let it reach _resolve_config, which strictly rejects
        # unknown keys (that guard still catches genuine slider-name typos).
        #
        # mesa.batch_run injects a per-run seed under the kwarg name `rng`
        # (it only uses `seed` if `seed` is already a parameter). Route it to
        # our `seed` override so parallel replicates are reproducible.
        if rng is not None:
            overrides.setdefault("seed", rng)
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

        # Stable handle on every rider ever spawned, kept in spawn order and
        # never pruned (model.agents drops finishers). Evolution reads this to
        # carry coefficients across races.
        self.riders: list[CyclistAgent] = []

        # Spawn on a start grid with fixed clearance: non-overlapping by
        # construction. Jitter stays strictly under half the clearance so it
        # can never close the gap between neighbouring slots.
        gap = 0.2
        slot_w = config.rider_width + gap
        slot_l = config.rider_length + gap
        per_row = max(1, int(config.road_width // slot_w))
        jitter = gap / 2 - 0.01
        for i in range(config.n_agents):
            # Seed this rider's learned coefficients from the population, if any.
            coeffs = population[i] if population is not None else None
            agent = CyclistAgent(self, team_id=i % config.n_teams, coeffs=coeffs)
            self.riders.append(agent)
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
        """Build a config, applying any keyword overrides (used by SolaraViz sliders).

        Field names and types are read straight off the dataclass, so adding a
        knob to PelotonConfig makes it overridable here (and SA-targetable via
        sweep.py) with no edit. Unknown keys still raise, catching slider typos.
        """
        base = config or PelotonConfig()
        if not overrides:
            return base
        values = asdict(base)
        for key, value in overrides.items():
            if key not in values:
                raise TypeError(f"Unknown model parameter: {key!r}")
            if key in ("n_agents", "n_teams"):
                value = int(value)
            elif key != "seed":
                value = float(value)  # every other knob is a float; seed passes through
            values[key] = value
        return PelotonConfig(**values)

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
