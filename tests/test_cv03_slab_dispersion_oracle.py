"""Estimator + oracle self-checks for the cv03 dispersion re-gate (#812).

These gate the INSTRUMENT, not the physics. They are E0 for the comparator:
the closed-form slab oracle is checked against an independent finite-difference
eigensolve that shares no line of code with it, and the two-wave estimator is
checked against synthetic lines whose beta is known exactly.

The pre-declared self-checks S1' and S2 of
``docs/design_notes/issue812_cv03_dispersion_regate_predeclaration.md`` live
here.  S3' (the two-wave residual) is enforced inside the crossval script
itself, because it is a property of the run, not of the comparator.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_COMPARATORS = Path(__file__).resolve().parent.parent / "validation" / "crossval" / "comparators"
sys.path.insert(0, str(_COMPARATORS))

from slab_te_dispersion import (  # noqa: E402
    slab_te0_neff, measure_neff_two_wave)

C0 = 2.998e8
A = 1.0e-6                      # the cv03 lattice constant


def _neff_fd(eps_core, eps_clad, d, k0, span=8.0, n=16001):
    """Independent oracle: 1-D Helmholtz eigenproblem for Ez(y).

    Shares nothing with ``slab_te0_neff`` -- no transcendental equation, no
    bisection, no analytic mode ansatz.  The largest eigenvalue of
    ``d2/dy2 + eps k0^2`` is ``beta^2`` of the fundamental bound mode.
    """
    from scipy.linalg import eigh_tridiagonal
    y = np.linspace(-span / 2, span / 2, n)
    h = y[1] - y[0]
    eps = np.where(np.abs(y) < d / 2, eps_core, eps_clad)
    diag = -2.0 / h ** 2 + eps * k0 ** 2
    off = np.ones(n - 1) / h ** 2
    w = eigh_tridiagonal(diag, off, select="i", select_range=(n - 1, n - 1),
                         eigvals_only=True)
    return float(np.sqrt(w[0]) / k0)


# ---------------------------------------------------------------- S2 (oracle)

@pytest.mark.parametrize("eps_core", [8.0, 10.0, 11.0, 12.0])
def test_s2_closed_form_matches_independent_fd_eigensolve(eps_core):
    """S2: closed form vs an independent FD eigensolve, within 3e-4."""
    k0 = 2.0 * np.pi * 0.15            # a = 1, c = 1 units
    closed = slab_te0_neff(eps_core, 1.0, 1.0, k0)
    fd = _neff_fd(eps_core, 1.0, 1.0, k0)
    assert abs(closed / fd - 1.0) < 3e-4, (
        f"eps={eps_core}: closed form {closed:.6f} vs FD {fd:.6f}")


def test_oracle_is_a_bound_mode_and_monotone_in_eps():
    """n_eff must lie strictly between the light lines and rise with eps."""
    k0 = 2.0 * np.pi * 0.15
    prev = 0.0
    for eps in (2.0, 4.0, 8.0, 12.0):
        n = slab_te0_neff(eps, 1.0, 1.0, k0)
        assert 1.0 < n < np.sqrt(eps)
        assert n > prev
        prev = n


def test_oracle_rejects_an_unbound_configuration():
    with pytest.raises(ValueError):
        slab_te0_neff(1.0, 12.0, 1.0, 2.0 * np.pi * 0.15)


# ------------------------------------------------------------- S1' (estimator)

@pytest.mark.parametrize("b_over_a", [0.0, 0.5, 0.9])
def test_s1p_two_wave_estimator_recovers_a_known_beta(b_over_a):
    """S1': synthetic two-wave line, recovered n_eff within 1e-9 relative."""
    f = 0.15 * C0 / A
    k0 = 2.0 * np.pi * f / C0
    n_true = slab_te0_neff(12.0, 1.0, A, k0)
    beta = n_true * k0
    x = np.arange(81) * (A / 10.0)
    y = (3.7 * np.exp(-1j * beta * x)
         + b_over_a * 3.7 * np.exp(1j * beta * x) * np.exp(0.7j))
    fit = measure_neff_two_wave(y[None, :], x, np.array([f]), c0=C0,
                                eps_core=12.0, eps_clad=1.0)[0]
    assert abs(fit.n_eff / n_true - 1.0) < 1e-9
    assert fit.rel_residual < 1e-9
    assert abs(fit.b_over_a - b_over_a) < 1e-9


def test_single_mode_phase_slope_is_biased_by_a_standing_wave():
    """Why the estimator was replaced (#812 design note section 7).

    On the very line the two-wave fit reads exactly, a plain unwrapped-phase
    slope -- the pre-declared estimator -- is wrong by more than half a percent,
    a third of the whole G1 budget, on synthetic data with no noise at all.
    """
    f = 0.15 * C0 / A
    k0 = 2.0 * np.pi * f / C0
    n_true = slab_te0_neff(12.0, 1.0, A, k0)
    beta = n_true * k0
    x = np.arange(81) * (A / 10.0)
    y = (np.exp(-1j * beta * x)
         + 0.53 * np.exp(1j * beta * x) * np.exp(0.7j))
    slope = np.polyfit(x, np.unwrap(np.angle(y)), 1)[0]
    n_slope = -slope / k0
    assert abs(n_slope / n_true - 1.0) > 5e-3
    two_wave = measure_neff_two_wave(y[None, :], x, np.array([f]), c0=C0,
                                     eps_core=12.0, eps_clad=1.0)[0]
    assert abs(two_wave.n_eff / n_true - 1.0) < 1e-9


# --------------------------------------------------- the gate can actually fail

def test_g1_would_fail_on_the_audit_defect():
    """A guide built at eps=8 must miss the eps=12 oracle by far more than 2%.

    This is the #812 falsifier expressed on synthetic data, so it runs in the
    unit suite with no FDTD: it fixes the arithmetic of the gate independently
    of the solver.
    """
    f = 0.15 * C0 / A
    k0 = 2.0 * np.pi * f / C0
    n_recipe = slab_te0_neff(12.0, 1.0, A, k0)
    x = np.arange(81) * (A / 10.0)
    for eps_defect, expect_at_least in ((11.0, 0.05), (10.0, 0.10), (8.0, 0.22)):
        n_defect = slab_te0_neff(eps_defect, 1.0, A, k0)
        y = np.exp(-1j * n_defect * k0 * x) + 0.5 * np.exp(1j * n_defect * k0 * x)
        fit = measure_neff_two_wave(y[None, :], x, np.array([f]), c0=C0,
                                    eps_core=eps_defect, eps_clad=1.0)[0]
        dev = abs(fit.n_eff / n_recipe - 1.0)
        assert dev > expect_at_least, (
            f"eps={eps_defect}: deviation {dev:.4f} from the eps=12 oracle")
        assert dev > 0.020, "G1's 2.0% gate must fire on every sweep point"


def test_estimator_rejects_malformed_input():
    f = np.array([0.15 * C0 / A])
    x = np.arange(81) * (A / 10.0)
    y = np.ones((1, 80), dtype=complex)
    with pytest.raises(ValueError):
        measure_neff_two_wave(y, x, f, c0=C0, eps_core=12.0, eps_clad=1.0)
    with pytest.raises(ValueError):
        measure_neff_two_wave(np.ones((1, 81), dtype=complex), x, f, c0=C0,
                              eps_core=1.0, eps_clad=12.0)


# ------------------------------------------- #812 round 2: numeric provenance
#
# The round-1 blocker was a comparison whose operands sat at different
# frequencies.  The corrected operands live in a committed JSON emitted by
# scripts/diagnostics/cv03_flux/build_cv03_matched_frequency.py; the design
# notes reference its keys instead of restating digits.  These tests make the
# artifact fail if it drifts from the code that produced it, and pin the two
# facts the correction asserts.

_ARTIFACT = (Path(__file__).resolve().parent.parent / "docs" / "design_notes"
             / "issue812_cv03_dispersion_matched_frequency.json")


def _built():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_cv03_mf",
        Path(__file__).resolve().parent.parent / "scripts" / "diagnostics"
        / "cv03_flux" / "build_cv03_matched_frequency.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build()


def test_matched_frequency_artifact_is_what_the_builder_emits():
    """The committed JSON must be byte-reproducible from committed code."""
    import json
    assert _ARTIFACT.is_file(), f"missing artifact {_ARTIFACT}"
    committed = json.loads(_ARTIFACT.read_text())
    assert committed == _built(), (
        "docs/design_notes/issue812_cv03_dispersion_matched_frequency.json is "
        "stale -- re-run scripts/diagnostics/cv03_flux/"
        "build_cv03_matched_frequency.py")


def test_round1_mismatch_mechanism_reproduces_and_inverts():
    """The published -0.026% must be reproducible ONLY at the wrong frequency.

    Two-sided: evaluating the analytic operand at ``fcen`` exactly regenerates
    round 1's published number, and evaluating it at the measured operand's own
    bin flips the sign.  That is the whole diagnosis, mechanised.
    """
    doc = _built()
    err = doc["round1_error"]
    assert err["reproduces_published_value"] == pytest.approx(
        err["published_dev_pct"], abs=5e-4)
    assert err["reproduces_published_value"] < 0.0
    assert err["corrected_dev_pct"] > 0.0
    assert doc["grid"]["fcen_minus_carrier_bin_c_over_a"] > 0.0
    # n_eff rises with f: that is why the mismatch could invert the sign.
    band = doc["oracle"]["band_n_eff"]
    assert all(b < a for b, a in zip(band, band[1:]))


def test_matched_frequency_deviations_are_one_signed_and_do_not_collapse():
    """The falsified conclusion, as a test.

    Round 1 concluded the recipe baseline's deviation was a domain/window
    artefact that the reflection-free run removes.  At matched frequency every
    configuration deviates with the same sign, and the reflection-free one is
    larger than the baseline, not smaller.
    """
    doc = _built()
    devs = doc["falsified_conclusion"]["dev_pct_matched_by_row"]
    assert all(v > 0.0 for v in devs.values())
    assert (devs["sx40_dft150_before_round_trip"]
            > devs["sx16_dft400_recipe_baseline"])
    assert doc["falsified_conclusion"]["reflection_free_minus_baseline_pct"] > 0.0


def test_column_semantics_cross_check_holds():
    """The recipe-baseline row must reproduce the committed run's own bin.

    This is what establishes that round 1's "measured n_eff" column is the
    carrier-bin two-wave fit, so that dividing it by the analytic value at that
    same bin is the right repair.
    """
    doc = _built()
    x = doc["column_semantics_cross_check"]
    assert x["abs_difference_pct"] < 5e-4, (
        "the recipe-baseline row no longer matches the per-bin deviation the "
        "results note recorded from the committed case run")
