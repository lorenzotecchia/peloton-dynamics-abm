import random
import types


from peloton import strategy
from peloton.config import PelotonConfig

CFG = PelotonConfig(road_length=1000.0)


def _coeffs(**over):
    c = {
        "effort": {"alpha": 0.0, "beta_dist": 0.0, "gamma_team": 0.3,
                   "delta_energy": 1.0, "gamma_match": 0.0},
        "sustain": {"bias": 0.0, "k_gap": 0.5},
        "weights": {"lambda_speed": 1.0, "lambda_energy": 1.0,
                    "lambda_risk": 0.5, "beta_loss": 1.5},
    }
    for group, params in over.items():
        c[group].update(params)
    return c


def _rider(x=0.0, team=0, w_prime=500.0, w_full=20000.0, s_m=12.0,
           cp=280.0, s_cp=8.0, effort=0.5, coeffs=None):
    return types.SimpleNamespace(
        pos=(x, 0.0), team_id=team, w_prime=w_prime, w_full=w_full, s_m=s_m,
        cp=cp, s_cp=s_cp, effort=effort, coeffs=coeffs or _coeffs(),
    )


def test_sigmoid_is_stable_at_extreme_inputs():
    assert strategy.sigmoid(-1000.0) == 0.0
    assert strategy.sigmoid(1000.0) == 1.0
    assert strategy.sigmoid(0.0) == 0.5


def test_default_coeffs_has_new_groups():
    c = strategy.default_coeffs(random.Random(1))
    assert set(c) == {"effort", "sustain", "weights"}
    assert set(c["effort"]) == {"alpha", "beta_dist", "gamma_team",
                                "delta_energy", "gamma_match"}
    assert set(c["sustain"]) == {"bias", "k_gap"}
    assert set(c["weights"]) == {"lambda_speed", "lambda_energy",
                                 "lambda_risk", "beta_loss"}


def test_default_coeffs_are_reproducible_with_seeded_rng():
    assert strategy.default_coeffs(random.Random(2)) == strategy.default_coeffs(random.Random(2))


def test_default_coeffs_are_independent_draws():
    rng = random.Random(1)
    assert strategy.default_coeffs(rng) != strategy.default_coeffs(rng)


# --- sustain_probability ---

def test_sustain_prob_rises_as_target_slows():
    me = _rider()
    slow = strategy.sustain_probability(me, v_target=5.0, cfg=CFG)
    fast = strategy.sustain_probability(me, v_target=15.0, cfg=CFG)
    assert slow > fast
    assert 0.0 <= fast <= slow <= 1.0


# --- attack_prob ---

def test_attack_prob_in_unit_interval():
    me = _rider()
    p = strategy.attack_prob(me, [me], v_group=7.0, cfg=CFG)
    assert 0.0 <= p <= 1.0


def test_attack_prob_low_when_pack_already_faster():
    me = _rider()
    # pack rolling faster than this rider can solo -> no point attacking
    p = strategy.attack_prob(me, [me], v_group=12.0, cfg=CFG)
    assert p < 0.5


def test_attack_risk_term_is_linear_not_quadratic():
    # On a favourable attack, scaling the payoff-magnitude weights must NOT
    # collapse the probability (a variance penalty would; std stays linear).
    me = _rider(coeffs=_coeffs())
    v_group = 5.0  # below what the rider can solo -> favourable
    base = strategy.attack_prob(me, [me], v_group=v_group, cfg=CFG)
    big = _rider(coeffs=_coeffs(weights={"lambda_speed": 100.0, "lambda_energy": 100.0}))
    scaled = strategy.attack_prob(big, [big], v_group=v_group, cfg=CFG)
    assert base > 0.4          # favourable to begin with
    assert scaled > 0.5        # std penalty keeps it decisive, not crushed to ~0


# --- effort ---

def test_effort_in_unit_interval_and_rises_with_energy():
    fresh = _rider(w_prime=20000.0)
    spent = _rider(w_prime=10.0)
    c_fresh = strategy.effort(fresh, [fresh], CFG)
    c_spent = strategy.effort(spent, [spent], CFG)
    assert 0.0 < c_spent < c_fresh < 1.0


def test_effort_rises_with_teammates():
    me = _rider(team=0)
    mate = _rider(team=0)
    stranger = _rider(team=1)
    alone = strategy.effort(me, [me, stranger], CFG)
    withmate = strategy.effort(me, [me, mate, stranger], CFG)
    assert withmate > alone


def test_effort_matches_group_mean_when_gamma_match_positive():
    me = _rider(effort=0.2, coeffs=_coeffs(effort={"gamma_match": 2.0}))
    high = [_rider(effort=0.9), _rider(effort=0.9)]
    me_flat = _rider(effort=0.2, coeffs=_coeffs(effort={"gamma_match": 0.0}))
    pulled = strategy.effort(me, [me] + high, CFG)
    flat = strategy.effort(me_flat, [me_flat] + high, CFG)
    assert pulled > flat       # gamma_match pulls toward the higher group mean
