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


from peloton.physics import neighbors_ahead


class _FakeAgent:
    def __init__(self, pos):
        self.pos = pos


class _FakeSpace:
    """Returns every agent it was given; physics does the geometric filtering."""

    def __init__(self, agents):
        self._agents = agents

    def get_neighbors(self, pos, radius, include_center=True):
        return list(self._agents)


class _FakeModel:
    def __init__(self, agents):
        self.space = _FakeSpace(agents)


def test_neighbors_ahead_keeps_only_riders_in_front_and_in_cone():
    me = _FakeAgent((100.0, 4.0))
    in_front = _FakeAgent((101.5, 4.2))      # ahead (x bigger), within lateral cone
    behind = _FakeAgent((99.0, 4.0))         # behind -> excluded
    too_wide = _FakeAgent((101.0, 6.0))      # ahead but lateral gap 2.0 > draft_lateral
    me_again = _FakeAgent((100.0, 4.0))      # same pos as self -> not "ahead"

    model = _FakeModel([me, in_front, behind, too_wide, me_again])
    result = neighbors_ahead(
        me, model, draft_radius=3.0, draft_lateral=1.0
    )

    assert in_front in result
    assert behind not in result
    assert too_wide not in result
    assert me not in result


def test_neighbors_ahead_empty_when_alone():
    me = _FakeAgent((100.0, 4.0))
    model = _FakeModel([me])
    assert neighbors_ahead(me, model, draft_radius=3.0, draft_lateral=1.0) == []


from peloton.physics import exposure_for


def test_exposure_is_one_when_alone():
    me = _FakeAgent((100.0, 4.0))
    model = _FakeModel([me])
    assert exposure_for(me, model, draft_radius=3.0, draft_lateral=1.0) == 1.0


def test_exposure_is_lower_when_tucked_close_behind():
    me = _FakeAgent((100.0, 4.0))
    close = _FakeAgent((100.2, 4.0))     # ~0.2 m behind a wheel -> strong shelter
    model = _FakeModel([me, close])
    exp = exposure_for(me, model, draft_radius=3.0, draft_lateral=1.0)
    assert 0.0 <= exp < 0.5              # well sheltered


def test_exposure_in_unit_interval_and_grows_with_gap():
    me = _FakeAgent((100.0, 4.0))
    far = _FakeAgent((102.8, 4.0))       # near edge of draft range -> little shelter
    model_far = _FakeModel([me, far])
    near = _FakeAgent((100.2, 4.0))
    model_near = _FakeModel([me, near])

    exp_far = exposure_for(me, model_far, draft_radius=3.0, draft_lateral=1.0)
    exp_near = exposure_for(me, model_near, draft_radius=3.0, draft_lateral=1.0)

    assert 0.0 <= exp_near <= 1.0
    assert 0.0 <= exp_far <= 1.0
    assert exp_far > exp_near            # bigger gap = more exposed
