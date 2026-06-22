"""Strategy layer: discrete EGT strategies and payoff matrix for the peloton.

Each cyclist carries a mixed strategy σ = (p_C, p_D, p_B) — a probability
vector over three pure strategies:

    C (COOPERATE) — pull at the front, share pace-setting work
    D (DEFECT)    — sit in / free-ride on others' effort
    B (BREAKAWAY) — attempt to escape the group solo

At the start of each race a pure strategy is drawn from σ.  During the race,
payoffs accumulate per step from pairwise interactions within each group using
the 3×3 payoff matrix defined in PelotonConfig.  Between races, replicator
dynamics update σ (see evolution.py).
"""

from enum import IntEnum

N_STRATEGIES = 3


class Strategy(IntEnum):
    COOPERATE = 0
    DEFECT    = 1
    BREAKAWAY = 2


# Short single-character labels used for CSV column headers and plots.
STRATEGY_NAMES = ["C", "D", "B"]

# ── Fixed per-strategy behaviour parameters ───────────────────────────────────
# These are the *deterministic* action profiles for each pure strategy.
# They replace the old continuous sigmoid coefficients.

# Contribution to group speed [0, 1]: fraction of max effort a rider puts in
# while in the pack.  Drives group_speed() and draft_factors() in group.py.
_CONTRIBUTION = [
    1.00,   # C — full effort at the front
    0.05,   # D — token effort, mostly sitting in
    0.50,   # B — half effort before the escape attempt
]

# Probability of attempting a solo breakaway each step.
_BREAKAWAY_PROB = [
    0.02,   # C — very rarely breaks away
    0.05,   # D — occasional opportunistic escape
    0.50,   # B — high chance each step
]

# Probability of chasing a breakaway that just formed.
_FOLLOW_PROB = [
    0.10,   # C — sometimes chases (cooperative instinct)
    0.15,   # D — slightly more likely to chase (opportunistic)
    0.20,   # B — most willing to follow another break
]


# ── Public API ────────────────────────────────────────────────────────────────

def contribution(agent, group, cfg) -> float:
    """Rider's contribution to group speed — determined by pure strategy."""
    return _CONTRIBUTION[int(agent.strategy)]


def breakaway_prob(agent, v_group, cfg) -> float:
    """Probability of attempting a solo breakaway this step."""
    return _BREAKAWAY_PROB[int(agent.strategy)]


def follow_prob(agent, breakaway_group, cfg) -> float:
    """Probability of chasing a breakaway that just formed."""
    return _FOLLOW_PROB[int(agent.strategy)]


def group_payoffs(members, cfg) -> list[float]:
    """Mean pairwise EGT payoff for each member from within-group interactions.

    Payoff to member i = mean over all opponents j ≠ i of A[s_i, s_j],
    where A is the 3×3 payoff matrix from the config.

    Solo breakaway groups (size 1) receive zero payoff — the rider escaped
    the interaction entirely.
    """
    n = len(members)
    if n <= 1:
        return [0.0] * n

    A = (
        (cfg.payoff_cc, cfg.payoff_cd, cfg.payoff_cb),
        (cfg.payoff_dc, cfg.payoff_dd, cfg.payoff_db),
        (cfg.payoff_bc, cfg.payoff_bd, cfg.payoff_bb),
    )
    payoffs = []
    for i, m in enumerate(members):
        si = int(m.strategy)
        total = sum(A[si][int(o.strategy)] for j, o in enumerate(members) if j != i)
        payoffs.append(total / (n - 1))
    return payoffs


def initial_mixed_strategy() -> list[float]:
    """Uniform prior: equal probability for every pure strategy."""
    return [1.0 / N_STRATEGIES] * N_STRATEGIES


def sample_strategy(mixed: list[float], rng) -> Strategy:
    """Draw a pure strategy from a mixed strategy probability vector."""
    r = rng.random()
    cumsum = 0.0
    for i, p in enumerate(mixed):
        cumsum += p
        if r < cumsum:
            return Strategy(i)
    return Strategy(N_STRATEGIES - 1)
