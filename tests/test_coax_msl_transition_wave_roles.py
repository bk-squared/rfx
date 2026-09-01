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

Status of the W5 gate (was RED; GREEN since the #822 fix, measured 2026-09-01)
------------------------------------------------------------------------------
BEFORE the fix (unmodified assembler, ``a_inc = backward_amp``), RED, and
the failure was exactly the label swap and nothing else::

    bin0: max|S_code-S_true|=1.476  max|S_code-inv(S_true)|=9.29e-14
    bin1: max|S_code-S_true|=1.15   max|S_code-inv(S_true)|=4.25e-14
    bin2: max|S_code-S_true|=1.7    max|S_code-inv(S_true)|=9.37e-14
    max fit_residual = 1.72e-15            (the pencil itself is clean)
    |a_code[0,1]|/|b_code[0,1]| = 11.1111  (= 1/0.09, the planted echo)
    label-swap counterfactual max|S-S_true| = 4.82e-14
    lam_min(I-S^H S): truth +0.326/+0.467/+0.533, code -1.59/-1.52/-3.29

i.e. BOTH ports' incident/outgoing roles were inverted and ``S_code =
inv(S_true)``. AFTER the fix (``rfx/api/_sparams.py::_incident_outgoing``
applied at both ports of ``_assemble_coax_msl_transition_from_voltages``),
GREEN, and the same three numbers read the other way round::

    bin0: max|S_code-S_true|=4.82e-14  max|S_code-inv(S_true)|=1.476
    bin1: max|S_code-S_true|=2.81e-14  max|S_code-inv(S_true)|=1.150
    bin2: max|S_code-S_true|=2.67e-14  max|S_code-inv(S_true)|=1.700
    max fit_residual = 1.72e-15            (unchanged: same pencil)
    |a_code[0,1]|/|b_code[0,1]| = 0.09     (= the planted echo itself)
    lam_min(I-S^H S): code +0.326/+0.467/+0.533, equal to the truth

and the wrong-side control (both ``dut_sign`` flipped, CORRECTED
assembler) reproduces the old RED table exactly, as it must -- planting on
the wrong side and reading with the wrong constant are the same error::

    max|S-inv(S_true)| = 9.37e-14/3.63e-14/5.14e-14
    max|S-S_true|      = 1.48/1.15/1.70
    |a[0,1]|/|b[0,1]|  = 11.1111,  lam_min = -1.59/-1.52/-3.28

The gate was committed as ``xfail(strict=True)`` -- never as a weakened
assertion -- and the fix PR removed the marker (with the fix in place and
the marker still on, pytest reported ``[XPASS(strict)] #589 H1: ...``).

Disposition of the two mechanism pins, which this docstring originally
predeclared deleting BOTH of (changed deliberately, not silently; routed
to the PI as design open question 7):

* ``test_h1_diagnostic_unmodified_assembler_returns_inverse_of_s_true``
  is DELETED as predeclared -- it asserted a property of code that no
  longer exists, and it failed under the fix (measured max absolute
  difference 1.69997933) exactly as designed.
* ``test_h1_diagnostic_label_swap_counterfactual_recovers_s_true`` is
  INVERTED rather than deleted, as
  ``test_wrong_side_planting_returns_the_inverse_of_s_true``: plant BOTH
  ports on the wrong side of their reference planes and the CORRECTED
  assembler must return exactly ``inv(S_true)``. Keeping it costs nothing
  and buys the one thing a newly-green gate needs -- a non-firing control
  proving the gate can still fail.

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

from rfx.api._sparams import (
    _assemble_coax_msl_transition_from_voltages,
    _incident_outgoing,
)
from rfx.sources.coaxial_port import coaxial_line_reflection_from_plane_voltages
from tests._wave_convention import plant_ladder_voltages_physical

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

# Which side of each ladder the DUT (the coax<->MSL junction) sits on. NOT a
# free choice: asserted against the realized plane positions above by
# ``test_the_planted_dut_signs_are_the_fixture_geometry`` -- the coax
# junction (2.5 mm) is at LARGER z than its probes (0.9-1.9 mm), the MSL
# junction (1.0 mm) at SMALLER x than its probes (2.6-10.6 mm).
_DUT_SIGN_COAX = +1.0
_DUT_SIGN_MSL = -1.0


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


def _physical_ladder_voltages(a, b, *, gamma, planes_m, ref_m, dut_sign, z0):
    """V(axis) at the probe planes from the PHYSICAL direction of travel.

    Thin modal-voltage wrapper (the ``sqrt(z0)`` power-wave -> volt-wave
    scale) around the ONE frozen planting contract,
    :func:`tests._wave_convention.plant_ladder_voltages_physical`, shared
    with ``tests/test_coax_two_port_fdtd.py`` and
    ``tests/test_coax_msl_transition.py`` (issue #822). ``dut_sign = +1``
    (coax here): the junction lies at LARGER coordinate than the ladder, so
    the incident wave travels +axis toward it as ``exp(-gamma (z - z_ref))``
    and the outgoing wave travels -axis. ``dut_sign = -1`` (MSL here): the
    junction lies at smaller coordinate, so the two exponentials exchange.
    In both cases ``V(ref) = sqrt(z0) (a + b)`` and the incident wave DECAYS
    (alpha > 0) as it approaches the junction; phase decreases along the
    direction of travel (``exp(-j beta s)``), matching the repo's
    ``exp(-j 2 pi f t)`` DFT kernel.
    """
    return np.sqrt(z0) * plant_ladder_voltages_physical(
        a, b, gamma=gamma, planes_m=planes_m, ref_m=ref_m, dut_sign=dut_sign,
    )


def _w5_fixture(*, wrong_side=False):
    """Planted ladders. ``wrong_side=True`` flips ``dut_sign`` at BOTH ports
    (the non-firing control: the fixture, not the assembler, is wrong)."""
    s = _s_true()
    a, b = _plant_ab_signal_flow(s, _GAMMA_T)
    g_c, g_m = _gammas()
    flip = -1.0 if wrong_side else +1.0
    v_coax = np.stack([
        _physical_ladder_voltages(
            a[0, drive], b[0, drive], gamma=g_c, planes_m=_Z_COAX_M,
            ref_m=_REF_COAX_M, dut_sign=flip * _DUT_SIGN_COAX, z0=_Z0_COAX,
        ) for drive in range(2)
    ])
    v_msl = np.stack([
        _physical_ladder_voltages(
            a[1, drive], b[1, drive], gamma=g_m, planes_m=_X_MSL_M,
            ref_m=_REF_MSL_M, dut_sign=flip * _DUT_SIGN_MSL, z0=_Z0_MSL,
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


def _run_assembler(*, wrong_side=False):
    s, a, b, v_coax, v_msl = _w5_fixture(wrong_side=wrong_side)
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
            ref_m=_REF_COAX_M, dut_sign=_DUT_SIGN_COAX, z0=_Z0_COAX)[0]
        np.testing.assert_allclose(v_ref_c, np.sqrt(_Z0_COAX) * (a[0, drive] + b[0, drive]))
        v_ref_m = _physical_ladder_voltages(
            a[1, drive], b[1, drive], gamma=g_m, planes_m=[_REF_MSL_M],
            ref_m=_REF_MSL_M, dut_sign=_DUT_SIGN_MSL, z0=_Z0_MSL)[0]
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
# W5 -- the H1 gate (was RED as xfail(strict=True); GREEN since the #822 fix)
# ---------------------------------------------------------------------------

def test_assembler_wave_roles_follow_the_junction_side_reference_plane():
    """W5: physical-convention planted voltages -> the assembler must return
    S_true, a passive S, and a SMALL incident wave at the non-driven coax
    port (the far-drive case: the backward wave dominates there)."""
    s, a, b, s_code, fit_resid, gamma_fit, a_code, b_code = _run_assembler()
    assert np.all(fit_resid < 1e-9), fit_resid.max()     # the pencil itself is clean
    g_c, g_m = _gammas()
    np.testing.assert_allclose(gamma_fit[0], np.broadcast_to(g_c, gamma_fit[0].shape), rtol=1e-6)
    np.testing.assert_allclose(gamma_fit[1], np.broadcast_to(g_m, gamma_fit[1].shape), rtol=1e-6)

    np.testing.assert_allclose(s_code, s, atol=1e-9)
    lam = _lambda_min_passivity(s_code)
    assert np.all(lam >= -1e-12), lam
    assert np.all(np.abs(a_code[0, 1]) < np.abs(b_code[0, 1])), (
        np.abs(a_code[0, 1]), np.abs(b_code[0, 1]))


def test_wrong_side_planting_returns_the_inverse_of_s_true():
    """Non-firing control for the now-GREEN gate: the wrong convention must
    NOT also pass, and it must fail in the ONE way the physics predicts.

    Plant both ladders with ``dut_sign`` flipped -- i.e. claim the junction
    is on the far side of each ladder from where it actually is -- and feed
    them to the CORRECTED assembler. Both ports then have their
    incident/outgoing roles exchanged, so ``S = B inv(A)`` with ``A`` and
    ``B`` swapped, i.e. exactly ``inv(S_true)``. Asserting the exact inverse
    (not merely "not close") is what makes this a statement about the LABELS
    rather than about the pencil, the Z0 normalization, the reference-plane
    extrapolation or the two-drive solve: every one of those stages runs
    unchanged here.

    This test replaces
    ``test_h1_diagnostic_label_swap_counterfactual_recovers_s_true``, which
    described the pre-#822 assembler (see this module's docstring for the
    disposition of both former mechanism pins).
    """
    s, a, b, s_code, fit_resid, gamma_fit, a_code, b_code = _run_assembler(
        wrong_side=True)
    assert np.all(fit_resid < 1e-9), fit_resid.max()   # the pencil is still clean
    s_inv = np.stack([np.linalg.inv(s[:, :, fi]) for fi in range(s.shape[-1])], axis=-1)
    np.testing.assert_allclose(s_code, s_inv, atol=1e-9)
    # ... and it is NOT the truth: the two differ by O(1), so this control
    # genuinely fires (a degenerate S_true with inv(S_true) == S_true would
    # make the assertion above vacuous).
    for fi in range(s.shape[-1]):
        assert np.abs(s_inv[:, :, fi] - s[:, :, fi]).max() > 1.0
    # The exchanged roles are visible in the raw amplitudes too: the
    # mislabelled 'incident' wave at the non-driven coax port under the MSL
    # drive is the BIG one (1/gamma_t = 11.1), the symptom family recorded on
    # the attempt-3 run before the fix.
    ratio = np.abs(a_code[0, 1]) / np.abs(b_code[0, 1])
    np.testing.assert_allclose(ratio, 1.0 / _GAMMA_T, rtol=1e-9)


def test_the_planted_dut_signs_are_the_fixture_geometry():
    """``dut_sign`` is not a tuning knob: it is read off the fixture.

    The frozen planting contract
    (:func:`tests._wave_convention.plant_ladder_voltages_physical`) is
    parameterised by WHERE THE DUT IS, so this test derives both signs from
    the realized plane positions and the junction coordinates rather than
    trusting the literals -- the same check
    ``tests/test_coax_two_port_fdtd.py`` makes for its own two ladders.
    """
    assert _REF_COAX_M > _Z_COAX_M.max()      # junction above the coax ladder
    assert _REF_MSL_M < _X_MSL_M.min()        # junction below the MSL ladder
    assert _DUT_SIGN_COAX == np.sign(_REF_COAX_M - _Z_COAX_M.mean())
    assert _DUT_SIGN_MSL == np.sign(_REF_MSL_M - _X_MSL_M.mean())
    # And on this lane -- reference plane AT the junction, i.e. on the DUT
    # side of both ladders -- the production helper must resolve BOTH ports
    # to a = forward_amp (the bit the pre-#822 constant got backwards).
    class _Out:
        forward_amp, backward_amp = 1.0 + 0j, 2.0 + 0j
    for ref, planes, sign in ((_REF_COAX_M, _Z_COAX_M, _DUT_SIGN_COAX),
                              (_REF_MSL_M, _X_MSL_M, _DUT_SIGN_MSL)):
        assert _incident_outgoing(_Out(), ref_m=ref, planes_m=planes,
                                  dut_sign=sign) == (1.0 + 0j, 2.0 + 0j)
        # ... and flipping the DUT side flips the mapping, nothing else.
        assert _incident_outgoing(_Out(), ref_m=ref, planes_m=planes,
                                  dut_sign=-sign) == (2.0 + 0j, 1.0 + 0j)
