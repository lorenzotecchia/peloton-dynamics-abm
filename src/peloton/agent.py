"""The cyclist agent: state plus per-step orchestration."""

from mesa import Agent

from peloton import energy, movement, strategy
from peloton.physics import exposure_for


class CyclistAgent(Agent):
    """A single rider. Holds state; delegates behaviour to the peloton modules."""

    def __init__(self, model, team_id: int):
        super().__init__(model)
        self.team_id = team_id
        self.energy = 100.0          # placeholder; energy.update_energy is a stub
        self.exposure = 1.0          # updated each step from drafting geometry
        self.action = None

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
