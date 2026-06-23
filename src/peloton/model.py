"""The peloton model: space, agent spawning, stepping, and data collection."""

from dataclasses import asdict

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import ContinuousSpace

from peloton import energy, group, strategy
from peloton.agent import CyclistAgent
from peloton.config import PelotonConfig


def _mean_exposure(model: "PelotonModel") -> float:
    agents = list(model.agents)
    if not agents:
        return 0.0
    return sum(a.exposure for a in agents) / len(agents)


def _mean_stamina(model: "PelotonModel") -> float:
    """Mean remaining anaerobic capacity W'/W_full across riders still racing."""
    agents = list(model.agents)
    if not agents:
        return 0.0
    return sum(a.w_prime / a.w_full for a in agents if a.w_full) / len(agents)


def _num_groups(model: "PelotonModel") -> int:
    """How many packs the field has fragmented into (1 = one bunch)."""
    agents = list(model.agents)
    if not agents:
        return 0
    return len(group.detect_groups(agents, model.config.group_radius))


def _num_breakaways(model: "PelotonModel") -> int:
    """Riders currently off the front on their own (or chasing a breakaway)."""
    return sum(1 for a in model.agents if a.solo)


def _exposure(cf_eff: float, min_cf_eff: float) -> float:
    """Map effective drag factor to exposure (0..1) for the viz.

    ``min_cf_eff`` is the sheltered baseline for the current pack size.
    """
    span = 1.0 - min_cf_eff
    if span <= 0.0:
        return 1.0
    return max(0.0, min(1.0, (cf_eff - min_cf_eff) / span))


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
        # mesa wants `rng`, not `seed` (deprecated). An int rng seeds self.random
        # identically to the old seed= path; None draws from OS entropy.
        super().__init__(rng=config.seed)
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

        # Spawn on a start grid: just a tidy starting bunch (riders are points
        # now, so overlap is irrelevant — this only spreads them for the viz).
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
                "MeanStamina": _mean_stamina,     # energy depletion over the race
                "NumGroups": _num_groups,         # peloton fragmentation
                "Breakaways": _num_breakaways,    # riders off the front
                "MeanExposure": _mean_exposure,   # average drag factor (drafting depth)
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
        """Advance one ``dt``: detect packs, resolve breakaways, move everyone.

        Packs are re-detected each step from current positions, so a rider that
        escapes (or gets caught) regroups by geometry alone.

        To avoid recent breakaway members immediately rejoining their old pack
        we keep a short cooldown (break_cooldown) during which they are
        processed separately. However, we only move each agent once per step,
        so newly-broken agents are processed with the regular pack in this step,
        then excluded from regular packs on subsequent steps until cooldown expires.
        """
        cfg = self.config
        active = list(self.agents)

        # Decrement short-term breakaway cooldowns and clear solo when expired
        for a in active:
            if getattr(a, "break_cooldown", 0) > 0:
                a.break_cooldown -= 1
                if a.break_cooldown == 0:
                    a.solo = False

        # Process agents that are not in cooldown (either always-regular or just
        # had their cooldown expire). Agents that break THIS step will get
        # break_cooldown set here and will be excluded from regular packs NEXT step.
        eligible = [a for a in active if getattr(a, "break_cooldown", 0) == 0]
        moved = set()
        for members in group.detect_groups(eligible, cfg.group_radius):
            self._advance_group(members, cfg)
            moved.update(members)

        # Process recent breakers (from previous steps) that haven't been moved yet.
        # This is typically empty since agents break and are moved in the same step,
        # but handles edge cases if grouping logic changes.
        remaining = [a for a in active if a not in moved and getattr(a, "break_cooldown", 0) > 0]
        for members in group.detect_groups(remaining, cfg.group_radius):
            self._advance_group(members, cfg)
            moved.update(members)

        self._remove_finishers()
        self.datacollector.collect(self)

    def _advance_group(self, members, cfg):
        contribs = [strategy.contribution(m, members, cfg) for m in members]
        v_group = group.group_speed(members, contribs, cfg)
        cf = group.draft_factors(members, contribs, cfg)

        # Breakaway, then teammates deciding whether to chase it. Use a
        # short-lived cooldown so we can treat recent breakers as a separate
        # set (possibly forming their own new pack) and avoid permanent solo
        # tagging.
        broke = []
        for m in members:
            if getattr(m, "break_cooldown", 0) == 0 and not m.solo and self.random.random() < strategy.breakaway_prob(m, v_group, cfg):
                m.solo = True
                m.break_cooldown = cfg.breakaway_cooldown_steps
                broke.append(m)
        if broke:
            for m in members:
                if m not in broke and getattr(m, "break_cooldown", 0) == 0 and not m.solo and self.random.random() < strategy.follow_prob(m, broke, cfg):
                    m.solo = True
                    m.break_cooldown = cfg.breakaway_cooldown_steps

        # If multiple riders broke simultaneously, treat them as a new pack for
        # the purposes of speed/drafting for this timestep.
        if broke and len(broke) > 1:
            contribs_broke = [strategy.contribution(m, broke, cfg) for m in broke]
            v_broke = group.group_speed(broke, contribs_broke, cfg)
            cf_broke = group.draft_factors(broke, contribs_broke, cfg)
        else:
            v_broke = None
            cf_broke = []

        min_cf_eff = group.mean_draft_factor_for_group(len(members))
        for m, cf_pack in zip(members, cf):
            if m in broke:
                if v_broke is not None:
                    # collective breakaway speed with drafting among breakers
                    idx = broke.index(m)
                    v, cf_eff = v_broke, cf_broke[idx]
                else:
                    # single rider breakaway: exposed to full wind
                    v, cf_eff = cfg.breakaway_speed_frac * m.s_m, 1.0
            elif m.solo:
                # chasers / solo riders (no drafting)
                v, cf_eff = cfg.breakaway_speed_frac * m.s_m, 1.0
            else:
                v, cf_eff = v_group, cf_pack

            m.wind_power = cfg.k_aero * cf_eff * v**3   # aerodynamic drag at the speed passed to update_stamina
            energy.update_stamina(m, energy.power_required(v, cf_eff, cfg), cfg)
            if m.w_prime <= 0.0:
                v = min(v, m.s_cp*0.75)                 # exhausted: drop to sustainable speed
            new_x = min(m.pos[0] + v * cfg.dt, cfg.road_length)
            self.space.move_agent(m, (new_x, m.pos[1]))
            m.exposure = _exposure(cf_eff, min_cf_eff)

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
