"""Smoke test: GSA driver samples, simulates, and returns finite indices."""
import numpy as np

from peloton import gsa


def _check(method, index_cols, tmp_path):
    # Short road + enough steps so races actually finish: the Total*/Mean* time
    # metrics are otherwise constant (all DNF), giving Sobol a zero-variance
    # column and NaN indices.
    df, _s2 = gsa.run_method(method, n=8, generations=2, max_steps=120, replicates=1,
                             processes=1, out_dir=tmp_path, base={"road_length": 80.0})
    # one row per (metric, param)
    assert len(df) == len(gsa.METRICS) * gsa.PROBLEMS[method]["num_vars"]
    assert set(df["metric"]) == set(gsa.METRICS)
    for col in index_cols:
        assert df[col].notna().all(), f"{method} {col} has NaNs"
        assert np.isfinite(df[col]).all()


def test_morris(tmp_path):
    _check("morris", ["mu_star", "sigma"], tmp_path)


def test_sobol(tmp_path):
    _check("sobol", ["S1", "ST"], tmp_path)
