"""The microstrip extractor's fitted ``beta`` is a diagnostic: S does not
depend on it.

The support matrix says so in public. This pins it: the same thru is solved
twice, the second time with the beta estimator replaced by one that returns
a value 20 % off, and S must come back bit-identical while the reported
``beta`` moves. S is built from V, I and the ANALYTIC Hammerstad-Jensen
reference impedance; if a de-embedding or a reference ever starts reading
the fitted beta, this goes red.
"""
from __future__ import annotations

import warnings

import numpy as np

from rfx.probes import msl_wave_decomp as _wd
from tests.unit.sparams.test_msl_sparse_dft import _thru_sim


def _solve():
    sim = _thru_sim("x", "uniform")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sim.compute_msl_s_matrix(n_freqs=3, num_periods=3)
    return np.asarray(res.S), np.asarray(res.beta)


def test_s_is_bit_identical_when_the_fitted_beta_is_wrong(monkeypatch):
    s_ref, beta_ref = _solve()
    real = _wd._estimate_beta

    def off_by_a_fifth(*args, **kwargs):
        beta, railed = real(*args, **kwargs)
        return 1.2 * beta, railed

    monkeypatch.setattr(_wd, "_estimate_beta", off_by_a_fifth)
    s_bad, beta_bad = _solve()
    assert not np.allclose(beta_bad, beta_ref, rtol=1e-3), \
        "the patched estimator did not reach the reported beta"
    np.testing.assert_array_equal(s_bad, s_ref)
