import types

import pytest

from peloton import group
from peloton.config import PelotonConfig

CFG = PelotonConfig(group_radius=3.0, k_s=0.8, draft_coefficient=0.62)


def _rider(x, s_sustain=12.0):
    return types.SimpleNamespace(pos=(x, 0.0), s_sustain=s_sustain)


def test_detect_groups_splits_on_gaps_and_chains_within_radius():
    a, b, c = _rider(0.0), _rider(2.0), _rider(4.0)   # chained: 0-2-4, gaps <=3
    far = _rider(20.0)
    groups = group.detect_groups([far, a, c, b], CFG.group_radius)
    assert len(groups) == 2
    big = max(groups, key=len)
    assert {id(r) for r in big} == {id(a), id(b), id(c)}
    assert any(len(g) == 1 and g[0] is far for g in groups)


def test_detect_groups_singletons_when_all_far():
    riders = [_rider(0.0), _rider(10.0), _rider(20.0)]
    assert len(group.detect_groups(riders, CFG.group_radius)) == 3


def test_group_speed_is_ks_weighted_average_of_sustainable_speeds():
    a, b = _rider(0.0, s_sustain=10.0), _rider(1.0, s_sustain=14.0)
    assert group.group_speed([a, b], [1.0, 1.0], CFG) == pytest.approx(0.8 * 12.0)


def test_group_speed_falls_back_to_slowest_when_no_contribution():
    a, b = _rider(0.0, s_sustain=10.0), _rider(1.0, s_sustain=14.0)
    assert group.group_speed([a, b], [0.0, 0.0], CFG) == pytest.approx(0.8 * 10.0)


def test_draft_factors_reward_the_bigger_contributor_with_more_wind():
    a, b = _rider(0.0), _rider(1.0)
    cf_a, cf_b = group.draft_factors([a, b], [0.9, 0.1], CFG)   # a leads most
    assert cf_a > cf_b                                # leader sees more wind
    for cf in (cf_a, cf_b):
        assert CFG.draft_coefficient <= cf <= 1.0
