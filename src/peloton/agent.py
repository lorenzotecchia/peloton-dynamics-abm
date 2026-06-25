"""The cyclist agent: physiological + strategic state.

Behaviour (grouping, drafting, pace, breakaways) is orchestrated pack-wise by
``PelotonModel.step`` — a single rider can't decide a pack's speed alone — so the
agent itself only holds state.
"""

from mesa import Agent

from peloton import energy, strategy


class CyclistAgent(Agent):
    """A single rider. State only; the model drives the dynamics."""

    def __init__(
        self,
        model,
        team_id: int,
        coeffs: dict | None = None,
        w_max10: float | None = None,
    ):
        super().__init__(model)
        self.team_id = team_id

        cfg = model.config
        # Heterogeneous engine; floor keeps the Gaussian tail physical.
        self.w_max10 = (
            w_max10
            if w_max10 is not None
            else max(50.0, model.random.gauss(cfg.w_max10_mean, cfg.w_max10_std))
        )
        # Derived physiology, all filled by init_physiology below.
        self.cp = self.s_m = self.s_cp = self.w_full = self.w_prime = 0.0
        energy.init_physiology(self, cfg)

        self.coeffs = (
            coeffs if coeffs is not None else strategy.default_coeffs(model.random)
        )
        self.solo = False  # True while a rider is off the front / chasing a breakaway (cleared after cooldown)
        self.break_cooldown = (
            0  # short-term memory to keep recent breakers separate from original packs
        )
        self.exposure = 1.0  # cf_eff-derived wind exposure, for the viz
        self.utility = 0.0  # race-outcome score, read by evolution
        self.wind_power = (
            0.0  # aerodynamic power (W) at the current step: k_aero * cf_eff * v^3
        )
