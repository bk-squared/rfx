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
    return any("ADVISORY" in str(w.message) and "max column power" in str(w.message)
               for w in rec)


def _hard_fired(rec):
    return any("UNRELIABLE" in str(w.message) for w in rec)


def test_soft_advisory_filter_excludes_reciprocity_warning():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        warnings.warn("compute_waveguide_s_matrix: reciprocity ADVISORY")
    assert not _soft_fired(rec)


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


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("amplitude,expected", [
    (np.nextafter(1.5, 0.0), "silent"),
    (1.5, "silent"),
    (np.nextafter(1.5, np.inf), "soft"),
    (1.625, "soft"),
    (2.0, "hard"),
])
def test_advisory_policy_boundaries(amplitude, expected, strict):
    """Policy, not a PEC calibration: P=|b/a|^2, floor=1.5^2.

    Adjacent float64 amplitudes test the open lower boundary without fitting
    a simulated value. Strict mode must never promote a soft advisory.
    """
    s = np.full((1, 1, 3), amplitude, dtype=complex)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        if strict and expected == "hard":
            with pytest.raises(ValueError, match="UNRELIABLE"):
                _warn_if_nonpassive_smatrix(
                    _result(s), extractor="compute_waveguide_s_matrix",
                    passivity_tol=2.0, strict=strict)
            return
        _warn_if_nonpassive_smatrix(
            _result(s), extractor="compute_waveguide_s_matrix",
            passivity_tol=2.0, strict=strict)
    assert _soft_fired(rec) == (expected == "soft")
    assert _hard_fired(rec) == (expected == "hard")


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("normalize", [False, True, "flux"])
@pytest.mark.parametrize("power,loose_expected", [
    (1.0, "silent"), (2.25, "silent"), (2.5, "soft"),
    (3.0, "soft"), (3.25, "hard"),
])
def test_public_waveguide_advisory_policy(monkeypatch, normalize, strict,
                                         power, loose_expected):
    """Replace the retired coarse-short warning trigger with a public API gate.

    Inject at the numerical extractor boundary, leaving the public assembly,
    normalization dispatch, result epilogue and diagnostic code real. With
    incident a_j=1, the constructed column has outgoing power P=sum |b_i|^2.
    The declared policy is silent through 2.25, warn-only through 1+2, then
    hard; normalized paths instead have hard limit 1+0.10. No measured pin.
    """
    from tests._pec_short_advisory_fixture import build

    # Unit incident power; distribute outgoing power over two real entries.
    # Check exact float64 column sums so rounding cannot move a boundary case.
    diagonal = min(1.5, np.sqrt(power))
    cross = np.sqrt(power - diagonal**2)
    s = np.zeros((2, 2, 6), dtype=complex)
    s[0, 0] = s[1, 1] = diagonal
    s[0, 1] = s[1, 0] = cross
    np.testing.assert_array_equal(np.sum(np.abs(s)**2, axis=0), power)
    original = s.copy()
    calls = []
    target = {False: "extract_waveguide_s_matrix",
              True: "extract_waveguide_s_params_normalized",
              "flux": "extract_waveguide_s_matrix_flux"}[normalize]

    def extract(*args, **kwargs):
        calls.append(target)
        assert kwargs["return_settling"] is True
        return s, np.array([-80., -80.])

    # #980 Phase 2 moved compute_waveguide_s_matrix verbatim into
    # rfx/sparams/waveguide.py, so the extractor it calls is a global of
    # THAT module now. Patch where the name is looked up -- patching
    # "rfx.api._sparams.<target>" still succeeds (the name is re-exported
    # there) but would no longer be the binding the lane reads.
    monkeypatch.setattr("rfx.sparams.waveguide." + target, extract)
    expected = loose_expected if normalize is False else (
        "silent" if power <= 1.10 else "hard")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sim = build(np.linspace(4e9, 6e9, 6), dx=2e-3, cpml=8)
        if strict and expected == "hard":
            with pytest.raises(ValueError, match="UNRELIABLE"):
                sim.compute_waveguide_s_matrix(
                    normalize=normalize, strict_passivity=strict, num_periods=1)
        else:
            result = sim.compute_waveguide_s_matrix(
                normalize=normalize, strict_passivity=strict, num_periods=1)
            np.testing.assert_array_equal(result.s_params, original)
            assert _soft_fired(rec) == (expected == "soft")
            assert _hard_fired(rec) == (expected == "hard")
    assert calls == [target]
    np.testing.assert_array_equal(s, original)
