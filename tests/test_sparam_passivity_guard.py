"""Tier-1 correctness guard: waveguide/coax S-matrix extractors must
self-flag a non-physical (non-passive / non-finite) result.

This locks the wiring of ``rfx.validation.validate_port_smatrix`` into the
NON-MSL extractors via ``_warn_if_nonpassive_smatrix`` (rfx/api/_sparams.py).
Operationalizes the R5 "no surface-metric verdict" discipline: a passive
structure cannot have column power > 1, so |S11| > 1 means the extractor is
wrong — exactly the failure mode behind the multi-session WR-90 |S11| chase.

The guard is exercised at the helper level (cheap, no FDTD) for the warn /
raise / pass / NaN / tracer-safety contract.
"""
from types import SimpleNamespace

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.api._sparams import _warn_if_nonpassive_smatrix


def _result(s_params, freqs=None, names=("port0",)):
    s = np.asarray(s_params)
    n_f = s.shape[-1]
    if freqs is None:
        freqs = np.linspace(1e9, 2e9, n_f)
    return SimpleNamespace(
        s_params=s,
        freqs=np.asarray(freqs, dtype=float),
        port_names=names,
    )


def test_passive_smatrix_is_silent():
    """A physical |S11| <= 1 must NOT warn."""
    s = np.full((1, 1, 4), 0.5 + 0.0j)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning => test failure
        _warn_if_nonpassive_smatrix(
            _result(s), extractor="compute_waveguide_s_matrix"
        )


def test_nonpassive_smatrix_warns():
    """|S11| = 8.94 (the canonical WR-90 detour value) must warn."""
    s = np.zeros((1, 1, 4), dtype=complex)
    s[0, 0, :] = 8.94
    with pytest.warns(UserWarning, match="passivity"):
        _warn_if_nonpassive_smatrix(
            _result(s), extractor="compute_waveguide_s_matrix"
        )


def test_nonpassive_smatrix_raises_under_strict():
    """strict=True turns the non-physical result into a hard error so an
    automation loop fails fast instead of optimizing against garbage."""
    s = np.zeros((1, 1, 4), dtype=complex)
    s[0, 0, :] = 1.5
    with pytest.raises(ValueError, match="UNRELIABLE"):
        _warn_if_nonpassive_smatrix(
            _result(s), extractor="compute_coaxial_s_matrix", strict=True
        )


def test_nonfinite_smatrix_warns():
    """NaN/Inf in the S-matrix must surface, not pass silently."""
    s = np.full((1, 1, 4), 0.3 + 0.0j)
    s[0, 0, 2] = np.nan
    with pytest.warns(UserWarning):
        _warn_if_nonpassive_smatrix(
            _result(s), extractor="compute_waveguide_s_matrix"
        )


def test_small_passivity_overage_within_tol_is_silent():
    """Numerical Yee impedance mismatch (~3%, documented for the
    normalize=False strong-reflector path) must not false-positive: the
    default tol matches the MSL honesty guard (|S11| <= ~1.05, i.e. column
    power <= 1.10), so a |S11| ~ 1.04 stays silent."""
    s = np.full((1, 1, 3), 1.04 + 0.0j)  # column power 1.0816 < 1.10
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_nonpassive_smatrix(
            _result(s), extractor="compute_waveguide_s_matrix"
        )


def test_guard_is_tracer_safe_under_jax_grad():
    """Under jax.grad the S-matrix is an abstract tracer; the numpy-based
    guard MUST be skipped (never raise / convert), so AD through an extractor
    that calls it stays intact."""

    def f(x):
        # x stands in for a traced s_params produced inside an extractor.
        res = SimpleNamespace(
            s_params=x.reshape(1, 1, -1),
            freqs=np.linspace(1e9, 2e9, x.shape[0]),
            port_names=("port0",),
        )
        _warn_if_nonpassive_smatrix(res, extractor="compute_waveguide_s_matrix")
        return jnp.real(jnp.sum(x))

    g = jax.grad(f)(jnp.full((4,), 5.0))  # 5.0 => |S11|=5 > 1, but traced
    assert bool(jnp.all(jnp.isfinite(g)))


def test_normalize_aware_tol_tolerates_documented_overshoot():
    """compute_waveguide_s_matrix(normalize=False) has documented Yee-dispersion
    + band-edge |S11| overshoot (validated paths reach ~1.4); the loose tol used
    on that path must stay SILENT on a column-power ~2.0 (|S11|~1.41) result,
    while the tight tol used on normalize=True/"flux" still flags it. Gross
    extractor bugs (|S11|>>1) are caught under either tol."""
    s = np.zeros((1, 1, 3), dtype=complex)
    s[0, 0, :] = np.sqrt(2.0)  # column power 2.0  (|S11| = 1.414)
    # loose tol (the normalize=False path) -> silent
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_nonpassive_smatrix(
            _result(s), extractor="compute_waveguide_s_matrix", passivity_tol=2.0
        )
    # tight tol (the normalize=True/"flux" path) -> warns
    with pytest.warns(UserWarning, match="passivity"):
        _warn_if_nonpassive_smatrix(
            _result(s), extractor="compute_waveguide_s_matrix", passivity_tol=0.10
        )
    # a gross bug is caught even under the loose tol
    s[0, 0, :] = 8.94
    with pytest.warns(UserWarning, match="passivity"):
        _warn_if_nonpassive_smatrix(
            _result(s), extractor="compute_waveguide_s_matrix", passivity_tol=2.0
        )
