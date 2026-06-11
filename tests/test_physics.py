import pytest

from peloton.physics import cf_draft


def test_cf_draft_at_zero_distance():
    # README formula: 0.62 - 0.0104*d + 0.0452*d^2, at d=0 -> 0.62
    assert cf_draft(0.0) == pytest.approx(0.62)


def test_cf_draft_increases_with_distance_within_range():
    # Within the relevant 0..3 m band, dropping further back reduces shelter
    # (drag multiplier rises back toward 1).
    assert cf_draft(3.0) > cf_draft(1.0)


def test_cf_draft_value_at_three_metres():
    # 0.62 - 0.0104*3 + 0.0452*9 = 0.62 - 0.0312 + 0.4068 = 0.9956
    assert cf_draft(3.0) == pytest.approx(0.9956)
