import types

import pytest

from peloton import group
from peloton.config import PelotonConfig

CFG = PelotonConfig(group_radius=3.0, k_s=0.8, draft_coefficient=0.62, cohesion_visibility=20.0)


def _rider(x, s_m=12.0):
    return types.SimpleNamespace(pos=(x, 0.0), s_m=s_m)


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


def test_group_speed_is_ks_weighted_average_of_threshold_speeds():
    a, b = _rider(0.0, s_m=10.0), _rider(1.0, s_m=14.0)
    assert group.group_speed([a, b], [1.0, 1.0], CFG) == pytest.approx(0.8 * 12.0)


def test_group_speed_falls_back_to_slowest_when_no_contribution():
    a, b = _rider(0.0, s_m=10.0), _rider(1.0, s_m=14.0)
    assert group.group_speed([a, b], [0.0, 0.0], CFG) == pytest.approx(0.8 * 10.0)


def test_cohesion_boost_zero_when_no_riders_ahead():
    r = _rider(10.0)
    behind = _rider(5.0)
    assert group.cohesion_boost(r, [r, behind], CFG) == pytest.approx(0.0)


def test_cohesion_boost_zero_when_visible_rider_outside_window():
    r = _rider(0.0)
    far_ahead = _rider(100.0)   # 100 m > visibility=20
    assert group.cohesion_boost(r, [r, far_ahead], CFG) == pytest.approx(0.0)


def test_cohesion_boost_scales_with_relative_distance():
    r = _rider(0.0, s_m=10.0)
    # average ahead is at x=10, visibility=20 → w=0.5 → boost=5.0
    ahead = _rider(10.0)
    boost = group.cohesion_boost(r, [r, ahead], CFG)
    assert boost == pytest.approx(0.5 * 10.0)


def test_cohesion_boost_larger_when_centroid_farther_away():
    r = _rider(0.0, s_m=10.0)
    near = _rider(5.0)
    far = _rider(15.0)
    boost_near = group.cohesion_boost(r, [r, near], CFG)
    boost_far = group.cohesion_boost(r, [r, far], CFG)
    assert boost_far > boost_near


def test_draft_factors_physical_front_always_in_full_wind():
    a, b = _rider(0.0), _rider(1.0)   # b is physically in front
    # even with low contribution, b faces full wind because of position
    cf_a, cf_b = group.draft_factors([a, b], [0.9, 0.1], CFG)
    assert cf_b == pytest.approx(1.0)
    assert cf_a < 1.0


def test_draft_factors_higher_contribution_among_drafters_means_more_wind():
    # Three riders: c is front and pinned to 1.0; among the drafters, a > b contribution.
    a, b, c = _rider(0.0, s_m=12.0), _rider(1.0, s_m=12.0), _rider(2.0, s_m=12.0)
    cf_a, cf_b, cf_c = group.draft_factors([a, b, c], [0.7, 0.2, 0.1], CFG)
    assert cf_c == pytest.approx(1.0)   # physically front
    assert cf_a > cf_b                  # a contributes more → more secondary wind
    for cf in (cf_a, cf_b):
        assert CFG.draft_coefficient <= cf < 1.0
