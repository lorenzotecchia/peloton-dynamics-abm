from mesa import Model
from mesa.space import ContinuousSpace

from peloton.agent import CyclistAgent
from peloton.config import PelotonConfig


class _MiniModel(Model):
    """Just enough of a model to host one agent for wiring tests."""

    def __init__(self, seed=0):
        super().__init__(seed=seed)
        self.config = PelotonConfig()
        self.space = ContinuousSpace(
            self.config.road_length, self.config.road_width, torus=False
        )


def test_agent_has_team_and_initial_state():
    model = _MiniModel()
    agent = CyclistAgent(model, team_id=2)
    model.space.place_agent(agent, (0.0, 4.0))
    assert agent.team_id == 2
    assert agent.exposure == 1.0       # no neighbours yet -> fully exposed
    assert agent.action is None
    assert isinstance(agent.energy, float)


def test_step_updates_exposure_action_and_advances_x():
    model = _MiniModel()
    agent = CyclistAgent(model, team_id=0)
    model.space.place_agent(agent, (0.0, 4.0))
    agent.step()
    assert agent.action == "ride"
    assert 0.0 <= agent.exposure <= 1.0
    assert agent.pos[0] > 0.0          # advanced forward
