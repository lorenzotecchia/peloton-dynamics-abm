from mesa import Model
from mesa.space import ContinuousSpace

from peloton.agent import CyclistAgent
from peloton.config import PelotonConfig
from peloton.strategy import Strategy, N_STRATEGIES


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
    assert agent.egt_payoff == 0.0
    # Default mixed strategy is the uniform prior.
    assert len(agent.mixed_strategy) == N_STRATEGIES
    assert abs(sum(agent.mixed_strategy) - 1.0) < 1e-10
    assert all(abs(p - 1.0 / N_STRATEGIES) < 1e-10 for p in agent.mixed_strategy)
    # Pure strategy is drawn from that distribution.
    assert isinstance(agent.strategy, Strategy)


def test_agent_physiology_is_initialised_and_consistent():
    model = _MiniModel()
    agent = CyclistAgent(model, team_id=0)
    assert agent.w_max10 >= 50.0                 # Gaussian floor holds
    assert agent.s_m > agent.s_cp > 0.0          # threshold above critical speed
    assert agent.cp == agent.w_max10 * model.config.cp_fraction
    assert agent.w_prime == agent.w_full > 0.0   # starts fully charged


def test_seeded_mixed_strategy_is_used_when_provided():
    model = _MiniModel()
    seeded = [0.6, 0.3, 0.1]
    agent = CyclistAgent(model, team_id=1, mixed_strategy=seeded)
    assert agent.mixed_strategy == seeded
    # Strategy drawn is one of the valid pure strategies.
    assert agent.strategy in list(Strategy)
