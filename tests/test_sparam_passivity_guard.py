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


# =============================================================================
# Item #5 (LLM-naive-usage audit) — SOFT over-unity advisory in the
# (documented-overshoot, extractor-broken] column-power gap.
#
# On the ``normalize=False`` waveguide path the passivity tol is loose (2.0 ->
# column-power hard limit 3.0, |S| <= 1.732 for a 1-port) to tolerate the
# DOCUMENTED single-run Yee/near-cutoff over-unity: a validated PEC short sits
# at column power ~2.0 there (see test_normalize_aware_tol_..._overshoot above
# and the battery test_pec_short_s11_magnitude). That left the window
# (~2.0, 3.0] UNGUARDED — a passive result materially above the documented
# envelope but below the hard limit returned silently. A SEPARATE, humble
# ADVISORY (never raise) now fires there. Floor is column power 2.25 (|S| ~ 1.5
# for a 1-port): above the ~2.0 documented envelope + the committed PEC-short
# with margin, below the tol=2.0 hard limit (3.0). The window is EMPTY on the
# tight-tol path (tol=0.10 -> hard limit 1.10 < 2.25), so the advisory only
# fires for normalize=False.  Message says "ADVISORY", NOT "UNRELIABLE".
# =============================================================================
def _soft_fired(rec):
    return any("ADVISORY" in str(w.message) for w in rec)


def _hard_fired(rec):
    return any("UNRELIABLE" in str(w.message) for w in rec)


def test_soft_advisory_fires_in_the_over_unity_gap():
    """A passive 1-port with column power in (2.25, 3.0] on the loose tol=2.0
    path must emit the SOFT advisory (warning, not the hard UNRELIABLE error)."""
    s = np.full((1, 1, 3), 1.58 + 0.0j)  # column power ~2.496, in (2.25, 3.0]
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _warn_if_nonpassive_smatrix(
            _result(s), extractor="compute_waveguide_s_matrix", passivity_tol=2.0
        )
    assert _soft_fired(rec), "expected the soft over-unity advisory in the gap"
    assert not _hard_fired(rec), "must NOT raise/flag the hard UNRELIABLE error"


def test_soft_advisory_silent_at_documented_envelope():
    """Column power == 2.0 (|S|=1.414, the documented normalize=False PEC-short
    envelope, locked silent by test_normalize_aware_tol_...) must NOT fire the
    soft advisory — the floor (2.25) sits above it with margin."""
    s = np.full((1, 1, 3), np.sqrt(2.0) + 0.0j)  # column power exactly 2.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning => failure
        _warn_if_nonpassive_smatrix(
            _result(s), extractor="compute_waveguide_s_matrix", passivity_tol=2.0
        )


def test_soft_advisory_silent_just_below_floor():
    """Column power 2.10 (< 2.25 floor) stays silent — margin for cross-machine
    float drift on the validated PEC-short (~2.00-2.005)."""
    s = np.full((1, 1, 3), np.sqrt(2.10) + 0.0j)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_nonpassive_smatrix(
            _result(s), extractor="compute_waveguide_s_matrix", passivity_tol=2.0
        )


def test_soft_advisory_never_fires_on_tight_tol_path():
    """On the tight tol=0.10 path (normalize='flux'/True) the window is empty
    (hard limit 1.10 < 2.25): a column power that would be in the gap is a HARD
    violation here, never the soft advisory."""
    s = np.full((1, 1, 3), 1.58 + 0.0j)  # column power ~2.496
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _warn_if_nonpassive_smatrix(
            _result(s), extractor="compute_waveguide_s_matrix", passivity_tol=0.10
        )
    assert not _soft_fired(rec), "soft advisory must not fire on the tight-tol path"
    assert _hard_fired(rec), "tight tol must flag this as the hard passivity error"


def test_gross_violation_still_hard_not_soft():
    """|S| >> 1 (column power > 3.0) stays the HARD UNRELIABLE error even under
    tol=2.0 — the soft advisory does not swallow gross extractor bugs."""
    s = np.full((1, 1, 3), 8.94 + 0.0j)  # column power ~79.9
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _warn_if_nonpassive_smatrix(
            _result(s), extractor="compute_waveguide_s_matrix", passivity_tol=2.0
        )
    assert _hard_fired(rec)
    assert not _soft_fired(rec)


@pytest.mark.slow
def test_soft_advisory_real_coarse_pec_short_witness():
    """REAL-geometry witness: a coarse (dx=2mm) WR-90 PEC-short on the
    normalize=False path lands at column power ~2.51 — above the ~2.0
    documented envelope, below the 3.0 hard limit — and used to return
    silently. It must now emit the soft advisory. The finer validated
    PEC-short (column power ~2.00) must stay silent (false-positive check)."""
    import jax.numpy as jnp
    from rfx import Box, Simulation

    DOMAIN = (0.12, 0.04, 0.02)

    def build(freqs, dx, cpml):
        freqs = np.asarray(freqs, float)
        f0 = float(freqs.mean())
        bw = max(0.2, min(0.8, (freqs[-1] - freqs[0]) / f0))
        sim = Simulation(freq_max=float(freqs[-1]), domain=DOMAIN,
                         boundary="cpml", cpml_layers=cpml, dx=dx)
        sim.add(Box((0.085, 0, 0), (0.087, DOMAIN[1], DOMAIN[2])), material="pec")
        pf = jnp.asarray(freqs)
        sim.add_waveguide_port(0.01, direction="+x", mode=(1, 0), mode_type="TE",
                               freqs=pf, f0=f0, bandwidth=bw,
                               waveform="modulated_gaussian", name="left")
        sim.add_waveguide_port(0.09, direction="-x", mode=(1, 0), mode_type="TE",
                               freqs=pf, f0=f0, bandwidth=bw,
                               waveform="modulated_gaussian", name="right")
        return sim

    # Witness: coarse mesh -> column power in the gap -> soft advisory fires.
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = build(np.linspace(4e9, 6e9, 6), dx=2e-3, cpml=8).\
            compute_waveguide_s_matrix(normalize=False, num_periods=30)
    cp = float(np.sum(np.abs(np.asarray(res.s_params)) ** 2, axis=0).max())
    assert 2.25 < cp <= 3.0, f"expected witness column power in the gap, got {cp:.4f}"
    assert _soft_fired(rec), f"coarse PEC-short (colpow {cp:.4f}) must emit the advisory"
    assert not _hard_fired(rec)

    # False-positive: the finer validated PEC-short (~2.00) stays silent.
    with warnings.catch_warnings(record=True) as rec2:
        warnings.simplefilter("always")
        res2 = build(np.linspace(5e9, 7e9, 6), dx=1e-3, cpml=10).\
            compute_waveguide_s_matrix(normalize=False, num_periods=40)
    cp2 = float(np.sum(np.abs(np.asarray(res2.s_params)) ** 2, axis=0).max())
    assert cp2 <= 2.25, f"validated PEC-short column power drifted into the gap: {cp2:.4f}"
    assert not _soft_fired(rec2), "validated PEC-short must NOT emit the advisory"
