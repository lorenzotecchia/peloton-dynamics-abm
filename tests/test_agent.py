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


def test_agent_has_team_and_default_strategy_state():
    model = _MiniModel()
    agent = CyclistAgent(model, team_id=2)
    assert agent.team_id == 2
    assert agent.solo is False
    assert agent.utility == 0.0
    # Fresh agent carries an independent copy of the default coefficients.
    assert set(agent.coeffs) == {"coop", "leave", "follow"}
    assert agent.coeffs["coop"]["delta"] == 1.0


def test_agent_physiology_is_initialised_and_consistent():
    model = _MiniModel()
    agent = CyclistAgent(model, team_id=0)
    assert agent.w_max10 >= 50.0  # Gaussian floor holds
    assert agent.s_m > agent.s_cp > 0.0  # threshold above critical speed
    assert agent.cp == agent.w_max10 * model.config.cp_fraction
    assert agent.w_prime == agent.w_full > 0.0  # starts fully charged


def test_seeded_coeffs_are_used_when_provided():
    model = _MiniModel()
    seeded = {"coop": {"alpha": 5.0}}
    agent = CyclistAgent(model, team_id=1, coeffs=seeded)
    assert agent.coeffs is seeded
