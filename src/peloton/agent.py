"""The cyclist agent: physiological + strategic state.

Strategy is a discrete EGT choice drawn from a mixed strategy vector
σ = (p_C, p_D, p_B) at the start of each race.  EGT payoffs accumulate
step-by-step from pairwise group interactions and drive replicator dynamics
between races (see evolution.py).
"""

from mesa import Agent

from peloton import energy
from peloton.strategy import Strategy, initial_mixed_strategy, sample_strategy


class CyclistAgent(Agent):
    """A single rider.  The model drives physics; the agent holds state only."""

    def __init__(self, model, team_id: int, mixed_strategy=None):
        super().__init__(model)
        self.team_id = team_id

        cfg = model.config
        self.w_max10 = max(50.0, model.random.gauss(cfg.w_max10_mean, cfg.w_max10_std))
        self.cp = self.s_m = self.s_cp = self.w_full = self.w_prime = 0.0
        energy.init_physiology(self, cfg)

        # Mixed strategy σ = [p_C, p_D, p_B]: probability over the three pure
        # strategies.  Supplied by evolution.run_generations after the first race;
        # otherwise initialised to the uniform prior.
        self.mixed_strategy: list[float] = (
            list(mixed_strategy) if mixed_strategy is not None
            else initial_mixed_strategy()
        )
        # Pure strategy chosen for this race by sampling from σ.
        self.strategy: Strategy = sample_strategy(self.mixed_strategy, model.random)

        self.solo = False
        self.break_cooldown = 0
        self.exposure = 1.0         # wind exposure fraction, for the viz
        self.egt_payoff = 0.0       # EGT payoff accumulated this race
        self.utility = 0.0          # fitness used by the evolution layer
