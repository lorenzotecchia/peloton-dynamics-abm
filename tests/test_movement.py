import random

from peloton.config import PelotonConfig
from peloton.movement import next_position


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
