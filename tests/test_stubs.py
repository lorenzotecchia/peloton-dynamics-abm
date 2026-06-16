from peloton import energy, strategy


class _Agent:
    def __init__(self):
        self.energy = 100.0
        self.action = None


def test_decide_action_returns_ride_default():
    a = _Agent()
    assert strategy.decide_action(a, model=None) == "ride"


def test_update_energy_is_noop_for_now():
    a = _Agent()
    before = a.energy
    energy.update_energy(a, model=None)
    assert a.energy == before     # stub does not change energy yet
