"""Evolutionary-stability check: can a mutant strategy invade the residents?

A strategy is (approximately) an ESS if a rare mutant playing something else
does no better than the residents. We don't prove it analytically — we evolve a
resident population, drop in a mutant at low frequency, race, and compare the
mutant's payoff to the residents'. One rider per team (n_teams == n_agents) so
each rider's utility is its own score, not a shared team total.

Status: currently XFAIL. The model is *not* ESS today — a solo rider rides its
full sustainable speed while a pack only manages ``k_s * s_sustain`` (k_s < 1),
so breaking away pays and an aggressive mutant out-scores the cooperative
residents. The assertion below encodes the ESS criterion; once the pack-speed
penalty no longer makes solo riding strictly better, this flips to XPASS and the
marker should be removed.
"""

import copy
import random
import statistics

import pytest

from peloton import evolution
from peloton.config import PelotonConfig
from peloton.model import PelotonModel

N_AGENTS = 20


def _cfg(seed):
    # One rider per team => utility is per-rider, the right granularity for ESS.
    return PelotonConfig(n_agents=N_AGENTS, n_teams=N_AGENTS, road_length=600.0, seed=seed)


def _race_utilities(cfg, population, max_steps=200):
    model = PelotonModel(cfg, population=population)
    for _ in range(max_steps):
        if not model.running:
            break
        model.step()
    evolution._assign_utilities(model.riders, model)
    # model.riders keeps spawn order == population order, so index 0 is the mutant.
    return [r.utility for r in model.riders]


@pytest.mark.xfail(reason="model not ESS: solo riders escape the k_s pack penalty", strict=False)
def test_evolved_residents_resist_random_mutant():
    # Evolve a resident strategy across several races.
    _hist, residents = evolution.run_generations(
        n_generations=12, max_steps=200, config=_cfg(seed=3)
    )
    assert residents is not None

    rng = random.Random(0)
    mutant_scores, resident_scores = [], []
    trials = 5
    for t in range(trials):
        pop = copy.deepcopy(residents)
        # Mutant = resident[0] perturbed hard in every coefficient.
        mutant = copy.deepcopy(residents[0])
        for grp, params in mutant.items():
            for p in params:
                mutant[grp][p] += rng.gauss(0.0, 3.0)
        pop[0] = mutant

        utils = _race_utilities(_cfg(seed=100 + t), pop)
        mutant_scores.append(utils[0])
        resident_scores.append(statistics.mean(utils[1:]))

    mutant_mean = statistics.mean(mutant_scores)
    resident_mean = statistics.mean(resident_scores)
    # If the evolved strategy is stable, the mutant can't out-score residents.
    assert mutant_mean <= resident_mean * 1.05, (
        f"mutant invaded: mutant={mutant_mean:.3f} > residents={resident_mean:.3f}"
    )
