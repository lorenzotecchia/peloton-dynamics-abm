import random

import pytest

from peloton.config import PelotonConfig
from peloton.movement import next_position
from peloton.physics import overlaps


class _FakeAgent:
    def __init__(self, pos):
        self.pos = pos


class _FakeSpace:
    def __init__(self, agents):
        self._agents = agents

    def get_neighbors(self, pos, radius, include_center=True):
        return list(self._agents)


class _FakeModel:
    def __init__(self, agents, seed=0):
        self.space = _FakeSpace(agents)
        self.config = PelotonConfig(road_width=8.0, base_speed=12.0, speed_noise=0.5)
        self.random = random.Random(seed)


def test_moves_forward_by_about_base_speed():
    me = _FakeAgent((100.0, 4.0))
    model = _FakeModel([me])
    new_x, _ = next_position(me, model)
    assert 100.0 + 12.0 - 0.5 <= new_x <= 100.0 + 12.0 + 0.5


def test_lateral_position_stays_within_road():
    me = _FakeAgent((100.0, 0.1))
    model = _FakeModel([me])
    for _ in range(50):
        x, y = next_position(me, model)
        me.pos = (x, y)
        assert 0.0 <= y <= 8.0


def test_seeks_shelter_when_a_rider_is_ahead_on_one_side():
    # A shelter-giver sits just ahead at y=6. Starting between open air (y=2)
    # and the wheel at y=6, greedy seek-shelter should not drift further from it
    # on average. We assert the rule runs and yields an in-road y.
    me = _FakeAgent((100.0, 4.0))
    shelter = _FakeAgent((100.5, 6.0))
    model = _FakeModel([me, shelter])
    x, y = next_position(me, model)
    assert 0.0 <= y <= 8.0
    assert x > 100.0


def _model_with(agents, **cfg_kwargs):
    model = _FakeModel(agents)
    model.config = PelotonConfig(**cfg_kwargs)
    return model


def test_brakes_to_wheel_when_all_lanes_blocked():
    # A wall of three riders 2 m ahead covers every lateral candidate, so the
    # rider must brake to exactly one rider_length behind the wall.
    me = _FakeAgent((100.0, 4.0))
    wall = [
        _FakeAgent((102.0, 3.4)),
        _FakeAgent((102.0, 4.0)),
        _FakeAgent((102.0, 4.6)),
    ]
    model = _model_with(
        [me] + wall, road_width=8.0, base_speed=12.0, speed_noise=0.0
    )
    new_x, new_y = next_position(me, model)
    assert new_x == pytest.approx(102.0 - model.config.rider_length)
    assert 0.0 <= new_y <= 8.0


def test_never_moves_into_an_occupied_slot():
    # A rider sits ahead-diagonal; sliding right would overlap it. The rider
    # must end up overlap-free and must not have drifted into the occupied side.
    me = _FakeAgent((100.0, 4.0))
    other = _FakeAgent((101.0, 4.7))
    model = _model_with(
        [me, other], road_width=8.0, base_speed=12.0, speed_noise=0.0
    )
    new_pos = next_position(me, model)
    assert not overlaps(
        new_pos, other.pos, rider_length=1.8, rider_width=0.6
    )
    assert new_pos[1] < 4.1          # did not slide toward the occupied slot


def test_lone_rider_still_advances_at_full_speed():
    me = _FakeAgent((100.0, 4.0))
    model = _model_with([me], road_width=8.0, base_speed=12.0, speed_noise=0.0)
    new_x, _ = next_position(me, model)
    assert new_x == pytest.approx(112.0)


def test_lone_rider_holds_its_line():
    # No shelter anywhere and no blockers: every candidate ties on exposure and
    # progress. The rider must keep its line (within jitter), not drift to the
    # first-evaluated candidate and gutter along the road edge.
    me = _FakeAgent((100.0, 4.0))
    model = _model_with([me], road_width=8.0, base_speed=12.0, speed_noise=0.0)
    _, new_y = next_position(me, model)
    assert abs(new_y - 4.0) <= 0.09 + 1e-9       # only jitter, no lane drift
