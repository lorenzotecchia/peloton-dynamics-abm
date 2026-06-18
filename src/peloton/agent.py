"""The cyclist agent: state plus per-step orchestration."""

from mesa import Agent

from peloton import energy, movement, strategy
from peloton.physics import exposure_for


class CyclistAgent(Agent):
    """A single rider. Holds state; delegates behaviour to the peloton modules."""

    def __init__(self, model, team_id: int, coeffs: dict | None = None):
        super().__init__(model)
        self.team_id = team_id
        self.energy = 100.0          # placeholder; energy.update_energy is a stub
        self.exposure = 1.0          # updated each step from drafting geometry
        self.action = None

        # --- Inert slots filled by the future stub layers. ---
        # Physiology (energy.py): w_max10 is heterogeneous now (framework, not
        # logic); the derived stamina vars stay None until the model lands.
        cfg = model.config
        self.w_max10 = model.random.gauss(cfg.w_max10_mean, cfg.w_max10_std)
        self.cp = None               # critical power
        self.w_full = None           # full anaerobic work capacity
        self.w_prime = None          # current anaerobic work capacity
        self.s_m = None              # speed at W_max10
        # Learning (strategy.py / evolution.py).
        self.coeffs = coeffs if coeffs is not None else {}  # learned game-theory coefficients
        self.utility = 0.0           # race-outcome score, read by evolution

    def step(self):
        cfg = self.model.config
        self.exposure = exposure_for(
            self, self.model,
            draft_radius=cfg.draft_radius,
            draft_lateral=cfg.draft_lateral,
        )
        self.action = strategy.decide_action(self, self.model)
        energy.update_energy(self, self.model)
        new_pos = movement.next_position(self, self.model)
        self.model.space.move_agent(self, new_pos)
