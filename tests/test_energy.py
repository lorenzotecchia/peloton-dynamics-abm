import types
from dataclasses import replace

import pytest

from peloton import energy
from peloton.config import PelotonConfig

CFG = PelotonConfig()


def test_solo_speed_inverts_power_required():
    for v in (6.0, 9.0, 12.5, 15.0):
        p = energy.power_required(v, 1.0, CFG)
        assert energy.solo_speed(p, CFG) == pytest.approx(v, rel=1e-9)


def test_power_rises_with_speed_and_exposure():
    assert energy.power_required(12.0, 1.0, CFG) > energy.power_required(10.0, 1.0, CFG)
    # Leading (cf=1) costs more than drafting (cf<1) at the same speed.
    assert energy.power_required(12.0, 1.0, CFG) > energy.power_required(12.0, 0.62, CFG)


def test_solo_speed_zero_for_nonpositive_power():
    assert energy.solo_speed(0.0, CFG) == 0.0
    assert energy.solo_speed(-5.0, CFG) == 0.0


def test_critical_power_is_fraction_of_wmax():
    assert energy.critical_power(400.0, CFG) == pytest.approx(0.7 * 400.0)


def test_init_physiology_sets_positive_consistent_values():
    agent = types.SimpleNamespace(w_max10=400.0)
    energy.init_physiology(agent, CFG)
    assert agent.cp == pytest.approx(0.7 * 400.0)
    assert agent.s_m > agent.s_cp > 0.0          # threshold speed above critical speed
    assert agent.w_full > 0.0
    assert agent.w_prime == agent.w_full
    # s_m is the speed at W_max10 by construction.
    assert energy.power_required(agent.s_m, 1.0, CFG) == pytest.approx(400.0, rel=1e-9)


def test_update_stamina_drains_above_cp_and_recovers_below():
    # r=1 isolates the recovery formula from the (SA-tuned) default recovery_rate;
    # dt=1 keeps the arithmetic a clean per-second drain (default dt is 2.0).
    cfg = replace(CFG, recovery_rate=1.0, dt=1.0)
    agent = types.SimpleNamespace(cp=280.0, w_full=1000.0, w_prime=500.0)
    energy.update_stamina(agent, p_required=380.0, cfg=cfg)   # 100 W over CP, dt=1
    assert agent.w_prime == pytest.approx(400.0)
    energy.update_stamina(agent, p_required=230.0, cfg=cfg)   # 50 W under CP, r=1
    assert agent.w_prime == pytest.approx(450.0)


def test_update_stamina_clamps_to_full_and_zero():
    full = types.SimpleNamespace(cp=280.0, w_full=1000.0, w_prime=990.0)
    energy.update_stamina(full, p_required=0.0, cfg=CFG)      # huge recovery
    assert full.w_prime == 1000.0                            # capped at W_full

    spent = types.SimpleNamespace(cp=280.0, w_full=1000.0, w_prime=10.0)
    energy.update_stamina(spent, p_required=1000.0, cfg=CFG)  # huge drain
    assert spent.w_prime == 0.0                              # floored, never negative
