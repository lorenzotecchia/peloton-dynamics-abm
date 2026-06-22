import random
import types

from peloton import evolution
from peloton.config import PelotonConfig
from peloton.strategy import Strategy, N_STRATEGIES, STRATEGY_NAMES, sample_strategy


def _fake_model(cfg, steps=100):
    return types.SimpleNamespace(
        config=cfg,
        random=random.Random(0),
        finish_order=[],
        steps=steps,
    )


def _rider(uid, strategy, egt_payoff, mixed_strategy=None):
    ms = mixed_strategy if mixed_strategy is not None else [1.0 / N_STRATEGIES] * N_STRATEGIES
    return types.SimpleNamespace(
        unique_id=uid,
        strategy=strategy,
        mixed_strategy=list(ms),
        egt_payoff=egt_payoff,
        utility=0.0,
    )


# ── replicator dynamics unit tests ───────────────────────────────────────────

def test_replicator_increases_probability_of_higher_fitness_strategy():
    """When C earns more than D/B, p_C should grow after evolve."""
    cfg = PelotonConfig(learning_rate=0.5, evo_noise=0.0)
    rng = random.Random(42)
    # Start with a mixed population: half play C, half play D
    riders = [
        _rider(0, Strategy.COOPERATE, egt_payoff=10.0),
        _rider(1, Strategy.COOPERATE, egt_payoff=10.0),
        _rider(2, Strategy.DEFECT,    egt_payoff=2.0),
        _rider(3, Strategy.DEFECT,    egt_payoff=2.0),
    ]
    sigma_before = [r.mixed_strategy[int(Strategy.COOPERATE)] for r in riders]

    evolution.evolve(riders, _fake_model(cfg))

    sigma_after = [r.mixed_strategy[int(Strategy.COOPERATE)] for r in riders]
    # All agents should have increased their p_C (C dominated this race).
    assert all(after > before for before, after in zip(sigma_before, sigma_after))


def test_replicator_decreases_probability_of_lower_fitness_strategy():
    """When D earns nothing, p_D should shrink after evolve."""
    cfg = PelotonConfig(learning_rate=0.5, evo_noise=0.0)
    riders = [
        _rider(0, Strategy.COOPERATE, egt_payoff=8.0),
        _rider(1, Strategy.BREAKAWAY, egt_payoff=8.0),
        _rider(2, Strategy.DEFECT,    egt_payoff=1.0),
        _rider(3, Strategy.DEFECT,    egt_payoff=1.0),
    ]
    pd_before = riders[0].mixed_strategy[int(Strategy.DEFECT)]
    evolution.evolve(riders, _fake_model(cfg))
    pd_after  = riders[0].mixed_strategy[int(Strategy.DEFECT)]
    assert pd_after < pd_before


def test_mixed_strategy_remains_valid_probability_vector():
    """After evolve, every agent's σ must sum to 1 and have no negatives."""
    cfg = PelotonConfig(learning_rate=0.3, evo_noise=0.05)
    riders = [
        _rider(i, Strategy(i % N_STRATEGIES), egt_payoff=float(i))
        for i in range(9)
    ]
    evolution.evolve(riders, _fake_model(cfg))
    for r in riders:
        assert abs(sum(r.mixed_strategy) - 1.0) < 1e-10
        assert all(p >= 0.0 for p in r.mixed_strategy)


def test_evolve_sets_utility_to_egt_payoff():
    cfg = PelotonConfig(learning_rate=0.1, evo_noise=0.0)
    riders = [
        _rider(0, Strategy.COOPERATE, egt_payoff=7.5),
        _rider(1, Strategy.DEFECT,    egt_payoff=2.0),
    ]
    evolution.evolve(riders, _fake_model(cfg))
    assert riders[0].utility == 7.5
    assert riders[1].utility == 2.0


def test_mutation_prevents_strategy_extinction():
    """With evo_noise > 0 a strategy that went extinct (p=0) regains probability."""
    cfg = PelotonConfig(learning_rate=0.5, evo_noise=0.1)
    # Agent believes only C is possible
    riders = [
        _rider(0, Strategy.COOPERATE, egt_payoff=5.0,
               mixed_strategy=[1.0, 0.0, 0.0]),
    ]
    evolution.evolve(riders, _fake_model(cfg))
    # Mutation must have added mass back to D and B.
    assert riders[0].mixed_strategy[int(Strategy.DEFECT)]    > 0.0
    assert riders[0].mixed_strategy[int(Strategy.BREAKAWAY)] > 0.0


# ── run_generations integration tests ────────────────────────────────────────

def test_run_generations_runs_and_returns_history():
    cfg = PelotonConfig(n_agents=6, n_teams=2, road_length=60.0, seed=0)
    history = evolution.run_generations(n_generations=3, max_steps=80, config=cfg)
    assert len(history) == 3
    assert [h["generation"] for h in history] == [0, 1, 2]
    assert all("n_finished" in h for h in history)


def test_run_generations_records_strategy_trajectories():
    cfg = PelotonConfig(n_agents=8, n_teams=2, road_length=60.0, seed=0)
    history = evolution.run_generations(n_generations=4, max_steps=80, config=cfg)

    # Mean and std must be present for every strategy.
    for name in STRATEGY_NAMES:
        assert f"{name}_mean" in history[0], f"missing {name}_mean"
        assert f"{name}_std"  in history[0], f"missing {name}_std"
        assert f"{name}_freq" in history[0], f"missing {name}_freq"

    # Generation 0 is the untouched initial population — all agents have the
    # uniform prior σ = [1/3, 1/3, 1/3], so population std is exactly 0.
    for name in STRATEGY_NAMES:
        assert history[0][f"{name}_std"] == 0.0
        assert abs(history[0][f"{name}_mean"] - 1.0 / N_STRATEGIES) < 1e-10

    # Frequencies must sum to 1 each generation.
    for h in history:
        freq_sum = sum(h[f"{name}_freq"] for name in STRATEGY_NAMES)
        assert abs(freq_sum - 1.0) < 1e-10


def test_mixed_strategies_stay_bounded_over_many_generations():
    """Replicator + mutation must never push σ outside [0, 1] or sum ≠ 1."""
    cfg = PelotonConfig(n_agents=20, n_teams=2, road_length=200.0,
                        evo_noise=0.01, seed=3)
    history = evolution.run_generations(n_generations=20, max_steps=200, config=cfg)
    for name in STRATEGY_NAMES:
        assert 0.0 <= history[-1][f"{name}_mean"] <= 1.0
