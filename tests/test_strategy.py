import random
import types


from peloton import strategy
from peloton.config import PelotonConfig

CFG = PelotonConfig(road_length=1000.0)


def _rider(x=0.0, team=0, w_prime=500.0, w_full=1000.0, s_m=12.0, coeffs=None):
    return types.SimpleNamespace(
        pos=(x, 0.0), team_id=team, w_prime=w_prime, w_full=w_full, s_m=s_m,
        coeffs=coeffs or {
            "coop":   {"alpha": 0.0, "beta": 0.0, "gamma": 0.3, "delta": 1.0},
            "leave":  {"alpha": -2.0, "beta": 0.0, "gamma": 0.5, "delta": 1.0},
            "follow": {"alpha": -1.0, "beta": 0.0, "gamma": 1.0, "delta": 1.0},
        },
    )


def test_sigmoid_is_stable_at_extreme_inputs():
    # Evolution can push coefficients large; sigmoid must not overflow math.exp.
    assert strategy.sigmoid(-1000.0) == 0.0
    assert strategy.sigmoid(1000.0) == 1.0
    assert strategy.sigmoid(0.0) == 0.5


def test_default_coeffs_are_independent_draws():
    rng = random.Random(1)
    a = strategy.default_coeffs(rng)
    b = strategy.default_coeffs(rng)
    assert a != b


def test_default_coeffs_are_reproducible_with_seeded_rng():
    a = strategy.default_coeffs(random.Random(2))
    b = strategy.default_coeffs(random.Random(2))
    assert a == b


def test_contribution_in_unit_interval_and_rises_with_energy():
    fresh = _rider(w_prime=1000.0)
    spent = _rider(w_prime=10.0)
    c_fresh = strategy.contribution(fresh, [fresh], CFG)
    c_spent = strategy.contribution(spent, [spent], CFG)
    assert 0.0 < c_spent < c_fresh < 1.0      # delta>0: more energy -> more willing


def test_contribution_rises_with_teammates_present():
    me = _rider(team=0)
    mate = _rider(team=0)
    stranger = _rider(team=1)
    alone = strategy.contribution(me, [me, stranger], CFG)
    withmate = strategy.contribution(me, [me, mate, stranger], CFG)
    assert withmate > alone                   # gamma>0 on teammate count


def test_breakaway_prob_rises_as_pack_slows():
    me = _rider(s_m=14.0)
    fast_pack = strategy.breakaway_prob(me, v_group=13.0, cfg=CFG)
    slow_pack = strategy.breakaway_prob(me, v_group=5.0, cfg=CFG)
    assert slow_pack > fast_pack
    assert 0.0 <= slow_pack <= 1.0


def test_follow_prob_rises_with_teammate_in_breakaway():
    me = _rider(team=0)
    mate_away = [_rider(team=0)]
    stranger_away = [_rider(team=1)]
    assert strategy.follow_prob(me, mate_away, CFG) > strategy.follow_prob(me, stranger_away, CFG)
