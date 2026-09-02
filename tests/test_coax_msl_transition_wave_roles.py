"""Issue #589 hypothesis H1 -- wave-role convention of the coax<->MSL assembler.

W5 (design ``.scratch_589c/design.md``, section 2): plant analytic modal
voltages from the PHYSICAL convention -- the incident wave is the one that
travels TOWARD the junction (the reference plane), the outgoing wave the one
that travels AWAY from it -- and feed them through the UNMODIFIED
:func:`rfx.api._sparams._assemble_coax_msl_transition_from_voltages`.

Why this is not a strawman
--------------------------
The extractor
:func:`rfx.sources.coaxial_port.coaxial_line_reflection_from_plane_voltages`
defines ``forward_amp`` as "the wave travelling from the probe region toward
the reference plane" (its own docstring), and the repo's DFT kernel is
``exp(-j 2 pi f t)`` (``rfx/probes/probes.py``), so a +axis-travelling wave
has phase DEcreasing along the axis (``exp(-j beta z)``). Both facts are
pinned here as label-independent contracts
(``test_extractor_forward_amp_means_toward_reference_plane``) before the
assembler is exercised. In ``compute_coax_msl_transition`` both reference
planes sit AT the junction, on the DUT side of their probe ladders (coax
ref 2.5 mm above probes at 0.9-1.9 mm; MSL ref 1.0 mm below probes at
2.6-10.6 mm), so on this lane the extractor's ``forward_amp`` IS the
incident wave.

The pre-existing planted test
(``tests/test_coax_msl_transition.py::test_planted_voltages_recover_known_s_matrix_with_unequal_z0``)
builds its voltages with ``_voltages_from_ab``, which "inverts the
extractor's own extrapolation" -- i.e. it plants ``a`` on whichever
exponential branch the assembler READS as ``a``, so it passes for either
label assignment and cannot discriminate H1. The planting here uses only
the physical direction of travel and never consults the extractor.

Status of the W5 gate (measured 2026-09-01, unmodified assembler, this file)
---------------------------------------------------------------------------
RED, and the failure is exactly the label swap and nothing else::

    bin0: max|S_code-S_true|=1.476  max|S_code-inv(S_true)|=9.29e-14
    bin1: max|S_code-S_true|=1.15   max|S_code-inv(S_true)|=4.25e-14
    bin2: max|S_code-S_true|=1.7    max|S_code-inv(S_true)|=9.37e-14
    max fit_residual = 1.72e-15            (the pencil itself is clean)
    |a_code[0,1]|/|b_code[0,1]| = 11.1111  (= 1/0.09, the planted echo)
    label-swap counterfactual max|S-S_true| = 4.82e-14
    lam_min(I-S^H S): truth +0.326/+0.467/+0.533, code -1.59/-1.52/-3.29

i.e. BOTH ports' incident/outgoing roles are inverted, ``S_code =
inv(S_true)``, and re-solving with ``a`` and ``b`` exchanged returns
``S_true`` to 5e-14. The fix (swap ``forward_amp``/``backward_amp`` at the
four assignments in the assembler, plus flipping ``_voltages_from_ab``'s
planting branch in ``tests/test_coax_msl_transition.py``) is a PRODUCTION
change and is PI-gated (design open question 1); it is deliberately NOT
applied in the commit that adds this file. The gate is therefore committed
as ``xfail(strict=True)`` -- never as a weakened assertion -- so the fix PR
MUST remove the marker (an unexpected pass reds the suite). The two
mechanism pins
(``test_h1_diagnostic_unmodified_assembler_returns_inverse_of_s_true``,
``test_h1_diagnostic_label_swap_counterfactual_recovers_s_true``) MUST be
deleted in that same PR: after the fix they describe code that no longer
exists.

Independence note (review blocker 2): this file is a REGRESSION test of a
convention, not new evidence about the FDTD run. It cannot by itself decide
whether the physical junction is passive -- the label-free adjudicator for
that is the #589 flux box (W4), and the sign of the net flux on the coax
plane is the two-sided discriminator. What W5 does establish, offline and
reproducibly, is that IF each reference plane sits between its probe ladder
and the DUT (which is how ``compute_coax_msl_transition`` places both of
them), THEN the current assignment inverts the S-matrix.
"""
from __future__ import annotations

import numpy as np
import pytest

from rfx.api._sparams import _assemble_coax_msl_transition_from_voltages
from rfx.sources.coaxial_port import coaxial_line_reflection_from_plane_voltages

C0 = 299_792_458.0

# ---------------------------------------------------------------------------
# Attempt-3 plane geometry (tests/test_coax_msl_transition.py, DX = 100 um):
# coax probes k = 17..27 step 2 -> z = 0.9..1.9 mm, junction k = 33 -> 2.5 mm;
# MSL probes i = 34..114 step 10 -> x = 2.6..10.6 mm, junction x = 1.0 mm.
# ---------------------------------------------------------------------------
_Z_COAX_M = 0.9e-3 + 0.2e-3 * np.arange(6)
_REF_COAX_M = 2.5e-3
_X_MSL_M = 2.6e-3 + 1.0e-3 * np.arange(9)
_REF_MSL_M = 1.0e-3
_FREQS = np.array([6.0e9, 8.0e9, 10.0e9])
_EPS_COAX = 2.1          # PTFE (fixture EPS_COAX)
_EPS_EFF_MSL = 2.8327    # Hammerstad-Jensen eps_eff quoted in the attempt-2 predeclaration
_ALPHA_COAX = 2.0        # /m, small loss so the planted line is lossy (alpha > 0)
_ALPHA_MSL = 5.0         # /m
_Z0_COAX = 45.46         # analytic coax TEM Z0 the method normalises with
_Z0_MSL = 53.11          # analytic HJ microstrip Zc the method normalises with
_GAMMA_T = 0.09          # terminator echo on the NON-driven port (F2 estimate)


def _gammas():
    beta_c = 2.0 * np.pi * _FREQS * np.sqrt(_EPS_COAX) / C0
    beta_m = 2.0 * np.pi * _FREQS * np.sqrt(_EPS_EFF_MSL) / C0
    return _ALPHA_COAX + 1j * beta_c, _ALPHA_MSL + 1j * beta_m


def _s_true():
    """Passive, lossy, reciprocal S_true with distinct per-bin magnitudes/phases.

    Magnitudes are the design's own W5 values -- |S11| 0.5/0.45/0.4,
    |S21| 0.6/0.55/0.5, |S22| 0.3/0.35/0.2 -- but the PHASES are not free:
    with |S11|=0.5, |S21|=0.6, |S22|=0.3 the Frobenius norm is already
    0.25 + 2(0.36) + 0.09 = 1.06, so ``sigma_max <= 1`` requires
    ``|det S| = sigma_1 sigma_2 >= sqrt(1.06 - 1) = 0.245`` while
    ``|det|`` only ranges over ``|0.15 e^{i(p11+p22)} - 0.36 e^{2 i p21}|``
    in [0.21, 0.51]. The design's literal phase triple
    (arg S22 = 1.3/2.8/-0.3) lands at |det| = 0.238 in the first bin and is
    NOT passive (measured lam_min(I - S^H S) = -0.0159 at 6 GHz; a real
    finding about the design text, caught by the precondition test below).

    Fixed here by pinning arg(S22) to the ANTI-PHASE determinant choice,
    ``arg S11 + arg S22 = 2 arg S21 + pi``, which maximises |det| at 0.51 and
    leaves lam_min = 0.326/0.467/0.533. Magnitudes, reciprocity and the
    per-bin phase spread of the design are all preserved; only the free
    parameter that the design fixed arbitrarily is re-chosen. Passivity of
    the TRUTH is ASSERTED in the precondition test, never assumed.
    """
    p11 = np.array([0.7, -1.9, 2.4])
    p21 = np.array([-2.2, 0.4, -0.9])
    p22 = 2.0 * p21 + np.pi - p11        # -1.9584 / -0.4416 / -1.0584 rad
    s11 = np.array([0.50, 0.45, 0.40]) * np.exp(1j * p11)
    s21 = np.array([0.60, 0.55, 0.50]) * np.exp(1j * p21)
    s22 = np.array([0.30, 0.35, 0.20]) * np.exp(1j * p22)
    return np.array([[s11, s21], [s21, s22]])   # (2, 2, n_f), reciprocal


def _plant_ab_signal_flow(s, gamma_t):
    """Power waves a[j, i, f] / b[j, i, f] (measured port j, driven port i).

    Pure signal-flow identity (same construction as
    ``tests/test_coax_msl_transition.py::_plant_ab_power_wave``): unit
    incident wave at the driven port, terminator echo ``gamma_t`` re-injecting
    a fraction of the outgoing wave at the non-driven port. Label-free by
    construction: 'a' means the wave that ARRIVES at the junction.
    """
    n_f = s.shape[-1]
    a = np.zeros((2, 2, n_f), dtype=np.complex128)
    b = np.zeros((2, 2, n_f), dtype=np.complex128)
    s11, s12, s21, s22 = s[0, 0], s[0, 1], s[1, 0], s[1, 1]
    b2 = s21 / (1.0 - s22 * gamma_t)
    a[0, 0], a[1, 0] = 1.0, gamma_t * b2
    b[1, 0], b[0, 0] = b2, s11 + s12 * (gamma_t * b2)
    b1 = s12 / (1.0 - s11 * gamma_t)
    a[1, 1], a[0, 1] = 1.0, gamma_t * b1
    b[0, 1], b[1, 1] = b1, s22 + s21 * (gamma_t * b1)
    return a, b


def _physical_ladder_voltages(a, b, *, gamma, planes_m, ref_m, junction_above, z0):
    """V(axis) at the probe planes from the PHYSICAL direction of travel.

    ``junction_above=True`` (coax): the incident wave travels +axis toward
    the junction, ``V_a = sqrt(z0) a exp(-gamma (z - z_ref))``; the outgoing
    wave travels -axis, ``V_b = sqrt(z0) b exp(+gamma (z - z_ref))``.
    ``junction_above=False`` (MSL): the incident wave travels -axis toward
    the junction, ``V_a = sqrt(z0) a exp(+gamma (x - x_ref))``; the outgoing
    wave travels +axis, ``V_b = sqrt(z0) b exp(-gamma (x - x_ref))``.
    In both cases ``V(ref) = sqrt(z0) (a + b)`` and the incident wave DECAYS
    (alpha > 0) as it approaches the junction. Phase decreases along the
    direction of travel (``exp(-j beta s)``), matching the repo's
    ``exp(-j 2 pi f t)`` DFT kernel.
    """
    d = np.asarray(planes_m, dtype=np.float64) - float(ref_m)     # (n_planes,)
    g = np.asarray(gamma, dtype=np.complex128)                    # (n_f,)
    sign = -1.0 if junction_above else +1.0
    e_inc = np.exp(sign * np.multiply.outer(d, g))
    e_out = np.exp(-sign * np.multiply.outer(d, g))
    return np.sqrt(z0) * (e_inc * a[None, :] + e_out * b[None, :])


def _w5_fixture():
    s = _s_true()
    a, b = _plant_ab_signal_flow(s, _GAMMA_T)
    g_c, g_m = _gammas()
    v_coax = np.stack([
        _physical_ladder_voltages(
            a[0, drive], b[0, drive], gamma=g_c, planes_m=_Z_COAX_M,
            ref_m=_REF_COAX_M, junction_above=True, z0=_Z0_COAX,
        ) for drive in range(2)
    ])
    v_msl = np.stack([
        _physical_ladder_voltages(
            a[1, drive], b[1, drive], gamma=g_m, planes_m=_X_MSL_M,
            ref_m=_REF_MSL_M, junction_above=False, z0=_Z0_MSL,
        ) for drive in range(2)
    ])
    return s, a, b, v_coax, v_msl


def _lambda_min_passivity(s):
    """min eigenvalue of I - S^H S per frequency (>= 0 <=> passive)."""
    out = []
    for fi in range(s.shape[-1]):
        m = s[:, :, fi]
        out.append(np.linalg.eigvalsh(np.eye(2) - m.conj().T @ m).min())
    return np.asarray(out)


def _run_unmodified_assembler():
    s, a, b, v_coax, v_msl = _w5_fixture()
    s_code, cond_a, cond_a_eq, rec_resid, fit_resid, gamma_fit, a_code, b_code = (
        _assemble_coax_msl_transition_from_voltages(
            z_coax_planes_m=_Z_COAX_M, x_msl_planes_m=_X_MSL_M,
            ref_coax_m=_REF_COAX_M, ref_msl_m=_REF_MSL_M,
            v_coax_by_drive=v_coax, v_msl_by_drive=v_msl,
            z0_coax=_Z0_COAX, z0_msl=_Z0_MSL,
        )
    )
    return s, a, b, s_code, fit_resid, gamma_fit, a_code, b_code


# ---------------------------------------------------------------------------
# Label-independent contracts the W5 reading rests on
# ---------------------------------------------------------------------------

def test_planted_truth_is_passive_and_the_planting_is_self_consistent():
    """Preconditions: S_true passive; a/b satisfy b = S a for both drives;
    V(ref) = sqrt(z0)(a + b) on the planted ladders (extrapolated)."""
    s, a, b, v_coax, v_msl = _w5_fixture()
    lam = _lambda_min_passivity(s)
    assert np.all(lam > 0.0), lam
    # Reciprocal, and the magnitudes the design predeclared.
    np.testing.assert_allclose(s[0, 1], s[1, 0], atol=0.0)
    np.testing.assert_allclose(np.abs(s[0, 0]), [0.50, 0.45, 0.40])
    np.testing.assert_allclose(np.abs(s[1, 0]), [0.60, 0.55, 0.50])
    np.testing.assert_allclose(np.abs(s[1, 1]), [0.30, 0.35, 0.20])
    for fi in range(s.shape[-1]):
        np.testing.assert_allclose(s[:, :, fi] @ a[:, :, fi], b[:, :, fi], atol=1e-14)
    g_c, g_m = _gammas()
    # Extrapolate the planted two-wave field to the reference plane by hand.
    for drive in range(2):
        v_ref_c = _physical_ladder_voltages(
            a[0, drive], b[0, drive], gamma=g_c, planes_m=[_REF_COAX_M],
            ref_m=_REF_COAX_M, junction_above=True, z0=_Z0_COAX)[0]
        np.testing.assert_allclose(v_ref_c, np.sqrt(_Z0_COAX) * (a[0, drive] + b[0, drive]))
        v_ref_m = _physical_ladder_voltages(
            a[1, drive], b[1, drive], gamma=g_m, planes_m=[_REF_MSL_M],
            ref_m=_REF_MSL_M, junction_above=False, z0=_Z0_MSL)[0]
        np.testing.assert_allclose(v_ref_m, np.sqrt(_Z0_MSL) * (a[1, drive] + b[1, drive]))
    assert v_coax.shape == (2, 6, 3) and v_msl.shape == (2, 9, 3)


@pytest.mark.parametrize("fi", [0, 1, 2])
def test_extractor_forward_amp_means_toward_reference_plane(fi):
    """Frozen extractor contract, independent of any assembler.

    A single +z wave ``exp(-gamma z)`` (phase decreasing with z, the repo's
    ``exp(-j 2 pi f t)`` DFT convention) is ``forward_amp`` when the
    reference plane is ABOVE the ladder and ``backward_amp`` when it is
    BELOW; the pairwise phase slope of that wave is ``-beta``.
    """
    g_c, _ = _gammas()
    g = g_c[fi]
    z = _Z_COAX_M
    v_plus = np.exp(-g * (z - z.mean()))            # +z travelling wave
    slope = np.angle(v_plus[1:] * np.conj(v_plus[:-1])) / np.diff(z)
    np.testing.assert_allclose(slope, -g.imag, rtol=1e-9)

    above = coaxial_line_reflection_from_plane_voltages(z, v_plus, reference_plane_m=_REF_COAX_M)
    below = coaxial_line_reflection_from_plane_voltages(z, v_plus, reference_plane_m=0.0)
    assert abs(above.forward_amp) > 1e6 * abs(above.backward_amp), (above.forward_amp, above.backward_amp)
    assert abs(below.backward_amp) > 1e6 * abs(below.forward_amp), (below.forward_amp, below.backward_amp)
    np.testing.assert_allclose(above.gamma, g, rtol=1e-6)


# ---------------------------------------------------------------------------
# W5 -- the H1 gate (RED on the unmodified assembler; PI-gated fix)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    reason=(
        "#589 H1: _assemble_coax_msl_transition_from_voltages assigns "
        "a_inc = backward_amp / b_out = forward_amp (copied from the coax "
        "two-port lane whose reference planes sit on the FAR side of the "
        "probes); on this lane both reference planes sit at the junction so "
        "forward_amp IS the incident wave and S_code = inv(S_true). "
        "Measured RED 2026-09-01 (max|S_code - S_true| = 1.48/1.15/1.70 per "
        "bin; max|S_code - inv(S_true)| = 9.3e-14/4.3e-14/9.4e-14; "
        "max fit_residual 1.7e-15). Fix is a PRODUCTION change and is "
        "PI-gated (design open question 1); remove this marker in the fix PR. "
        "Lane: this is an offline numpy test with no slow/slow_physics "
        "marker, so the default fast suite (.github/workflows/pr-tests.yml "
        "fast-suite, plain pytest under pyproject addopts) collects it on "
        "every PR and strict=True reds that lane the day the fix lands "
        "without removing this marker."
    ),
)
def test_assembler_wave_roles_follow_the_junction_side_reference_plane():
    """W5: physical-convention planted voltages -> the assembler must return
    S_true, a passive S, and a SMALL incident wave at the non-driven coax
    port (the far-drive case: the backward wave dominates there)."""
    s, a, b, s_code, fit_resid, gamma_fit, a_code, b_code = _run_unmodified_assembler()
    assert np.all(fit_resid < 1e-9), fit_resid.max()     # the pencil itself is clean
    g_c, g_m = _gammas()
    np.testing.assert_allclose(gamma_fit[0], np.broadcast_to(g_c, gamma_fit[0].shape), rtol=1e-6)
    np.testing.assert_allclose(gamma_fit[1], np.broadcast_to(g_m, gamma_fit[1].shape), rtol=1e-6)

    np.testing.assert_allclose(s_code, s, atol=1e-9)
    lam = _lambda_min_passivity(s_code)
    assert np.all(lam >= -1e-12), lam
    assert np.all(np.abs(a_code[0, 1]) < np.abs(b_code[0, 1])), (
        np.abs(a_code[0, 1]), np.abs(b_code[0, 1]))


def test_h1_diagnostic_unmodified_assembler_returns_inverse_of_s_true():
    """DIAGNOSTIC PIN of the H1 mechanism (temporary -- DELETE in the fix PR,
    where it must fail): on the W5 planting the unmodified assembler returns
    exactly ``inv(S_true)`` -- the signature of BOTH ports having their
    incident/outgoing roles swapped (``B inv(A)`` with A and B exchanged).
    The recorded attempt-3 run shows the same symptom family: coax-ladder
    ``|a_inc/b_out| = 11.1/11.6/11.9`` at the NON-driven coax port under MSL
    drive, and ``inv(S_code)`` not passive (H1 alone is not the whole story
    -- see design H4/H8)."""
    s, a, b, s_code, fit_resid, gamma_fit, a_code, b_code = _run_unmodified_assembler()
    s_inv = np.stack([np.linalg.inv(s[:, :, fi]) for fi in range(s.shape[-1])], axis=-1)
    np.testing.assert_allclose(s_code, s_inv, atol=1e-9)
    # The two labels are exactly exchanged: a_code == sqrt-normalised b, b_code == a.
    np.testing.assert_allclose(a_code, b, atol=1e-9)
    np.testing.assert_allclose(b_code, a, atol=1e-9)
    # And the mislabelled 'incident' wave at the non-driven coax port under
    # MSL drive is the BIG one (1/gamma_t = 11.1), as in the attempt-3 JSON.
    ratio = np.abs(a_code[0, 1]) / np.abs(b_code[0, 1])
    np.testing.assert_allclose(ratio, 1.0 / _GAMMA_T, rtol=1e-9)
    # Not passive: a swapped passive S is generally over-unity.
    assert np.any(_lambda_min_passivity(s_code) < 0.0)


def test_h1_diagnostic_label_swap_counterfactual_recovers_s_true():
    """DIAGNOSTIC PIN (temporary -- DELETE in the fix PR): re-solving the
    assembler's OWN power waves with ``a`` and ``b`` exchanged returns
    ``S_true``.

    This is what makes the W5 red a statement about the LABELS rather than
    about the pencil, the Z0 normalization, the reference-plane
    extrapolation or the two-drive solve: every one of those stages is used
    unchanged here, and only the two roles are exchanged at the solve. It is
    also the exact algebraic content of the driver's report-only
    "label-swap counterfactual" table -- a PREDICTION of H1, never a
    measurement.
    """
    from rfx.sources.coaxial_port import solve_two_port_from_wave_amplitudes

    s, a, b, s_code, fit_resid, gamma_fit, a_code, b_code = _run_unmodified_assembler()
    swapped = solve_two_port_from_wave_amplitudes(
        b_code, a_code, cond_warn=1.0e30).s_params
    np.testing.assert_allclose(swapped, s, atol=1e-9)
    # ... and it is exactly the matrix inverse of what the code returned.
    s_code_inv = np.stack(
        [np.linalg.inv(s_code[:, :, fi]) for fi in range(s.shape[-1])], axis=-1)
    np.testing.assert_allclose(swapped, s_code_inv, atol=1e-9)
