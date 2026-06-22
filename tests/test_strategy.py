import random
import types

from peloton import strategy
from peloton.config import PelotonConfig
from peloton.strategy import (
    Strategy,
    N_STRATEGIES,
    STRATEGY_NAMES,
    initial_mixed_strategy,
    sample_strategy,
)

CFG = PelotonConfig()


def _rider(strat: Strategy):
    return types.SimpleNamespace(strategy=strat)


# ── pure-strategy behaviour tables ───────────────────────────────────────────

def test_contribution_values_are_ordered():
    """Cooperator pulls hardest; defector barely contributes."""
    c_coop = strategy.contribution(_rider(Strategy.COOPERATE), [], CFG)
    c_defe = strategy.contribution(_rider(Strategy.DEFECT),    [], CFG)
    c_brak = strategy.contribution(_rider(Strategy.BREAKAWAY), [], CFG)
    assert c_coop > c_brak > c_defe
    assert 0.0 < c_defe and c_coop <= 1.0


def test_breakaway_prob_ordered_by_strategy():
    """Breakaway specialists attempt escape most often."""
    p_coop = strategy.breakaway_prob(_rider(Strategy.COOPERATE), None, CFG)
    p_defe = strategy.breakaway_prob(_rider(Strategy.DEFECT),    None, CFG)
    p_brak = strategy.breakaway_prob(_rider(Strategy.BREAKAWAY), None, CFG)
    assert p_brak > p_defe >= p_coop
    assert 0.0 <= p_coop and p_brak <= 1.0


def test_follow_prob_in_unit_interval():
    for strat in Strategy:
        p = strategy.follow_prob(_rider(strat), [], CFG)
        assert 0.0 <= p <= 1.0


# ── group_payoffs ─────────────────────────────────────────────────────────────

def test_group_payoffs_length_matches_members():
    members = [_rider(Strategy.COOPERATE), _rider(Strategy.DEFECT)]
    payoffs = strategy.group_payoffs(members, CFG)
    assert len(payoffs) == 2


def test_group_payoffs_prisoner_dilemma_ordering():
    """In the C/D sub-game the payoff matrix satisfies T > R > P > S."""
    c = _rider(Strategy.COOPERATE)
    d = _rider(Strategy.DEFECT)

    # T: defector against cooperator
    T = strategy.group_payoffs([d, c], CFG)[0]
    # R: mutual cooperation
    R = strategy.group_payoffs([c, c], CFG)[0]
    # P: mutual defection
    P = strategy.group_payoffs([d, d], CFG)[0]
    # S: cooperator against defector
    S = strategy.group_payoffs([c, d], CFG)[0]

    assert T > R > P > S, f"T={T} R={R} P={P} S={S}"


def test_group_payoffs_solo_rider_is_zero():
    """A single rider has no opponent to interact with → zero payoff."""
    payoffs = strategy.group_payoffs([_rider(Strategy.BREAKAWAY)], CFG)
    assert payoffs == [0.0]


def test_group_payoffs_empty_group_is_empty():
    assert strategy.group_payoffs([], CFG) == []


def test_breakaway_gets_better_payoff_against_defector_than_cooperator():
    """B earns more against D than C (defectors don't chase the escape)."""
    b = _rider(Strategy.BREAKAWAY)
    c = _rider(Strategy.COOPERATE)
    d = _rider(Strategy.DEFECT)
    payoff_vs_c = strategy.group_payoffs([b, c], CFG)[0]
    payoff_vs_d = strategy.group_payoffs([b, d], CFG)[0]
    assert payoff_vs_d > payoff_vs_c


# ── mixed strategy utilities ──────────────────────────────────────────────────

def test_initial_mixed_strategy_is_uniform():
    ms = initial_mixed_strategy()
    assert len(ms) == N_STRATEGIES
    assert abs(sum(ms) - 1.0) < 1e-10
    assert all(abs(p - 1.0 / N_STRATEGIES) < 1e-10 for p in ms)


def test_sample_strategy_returns_valid_strategy():
    rng = random.Random(0)
    ms = [0.8, 0.1, 0.1]
    for _ in range(50):
        s = sample_strategy(ms, rng)
        assert isinstance(s, Strategy)
        assert s in list(Strategy)


def test_sample_strategy_respects_degenerate_distribution():
    """A pure-strategy distribution (weight=1 on one strategy) must always return it."""
    rng = random.Random(7)
    for target in Strategy:
        ms = [0.0] * N_STRATEGIES
        ms[int(target)] = 1.0
        for _ in range(20):
            assert sample_strategy(ms, rng) == target


def test_strategy_names_cover_all_strategies():
    assert len(STRATEGY_NAMES) == N_STRATEGIES
