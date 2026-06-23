"""Smoke test: GSA driver samples, simulates, and returns finite indices."""
import numpy as np

from peloton import gsa


def _check(method, index_cols):
    df = gsa.run_method(method, n=4, max_steps=30, replicates=1, processes=1)
    # one row per (metric, param)
    assert len(df) == len(gsa.METRICS) * gsa.PROBLEM["num_vars"]
    assert set(df["metric"]) == set(gsa.METRICS)
    for col in index_cols:
        assert df[col].notna().all(), f"{method} {col} has NaNs"
        assert np.isfinite(df[col]).all()


def test_morris():
    _check("morris", ["mu_star", "sigma"])


def test_sobol():
    _check("sobol", ["S1", "ST"])
