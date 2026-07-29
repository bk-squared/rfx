"""Two-drive 2x2 coax S-parameter solve (issue #489) — pure, no FDTD.

These tests pin the design decision behind `solve_two_port_from_wave_amplitudes`
as executable facts rather than prose:

  * the obvious rule `S[j,i] = b_j / a_i` has a HARD floor set by the terminator
    (measured 0.0800 with rfx's validated |Gamma_t| < 0.08 terminator), so it
    cannot be made accurate by improving the implementation;
  * the two-drive solve removes that floor entirely, staying exact to ~1e-13
    even at |Gamma_t| = 0.5;
  * which implementation defects the local witnesses DO and DO NOT separate.
    Two of the three families are invisible to both witnesses here, and that
    is recorded as an explicit gap rather than left for someone to discover:
    a per-port unit-modulus factor (sign flip / reference-plane shift) is a
    diagonal similarity, and a systematic both-drive a/b mislabel is a
    non-diagonal one that preserves |S21| = |S12| exactly. Only PASSIVITY
    catches the latter.

All fields are planted analytically from a signal-flow solve, so the "truth"
here is exact and independent of any rfx extraction path.
"""

import numpy as np
import pytest

from rfx.sources.coaxial_port import solve_two_port_from_wave_amplitudes

_FREQS = np.array([4e9, 6e9, 8e9, 10e9, 12e9])


def _plant(s_true, gamma_t, n_f):
    """Wave amplitudes at both ports for both drives, with terminator Gamma_t.

    Signal flow for a 2-port S with a terminator of reflection ``gamma_t`` at
    the NON-driven port and a unit incident wave at the driven port::

        drive 1:  a1 = 1,  a2 = gamma_t * b2
                  b2 = S21 / (1 - S22*gamma_t),  b1 = S11 + S12*b2*gamma_t

    and the mirror for drive 2. Returns ``(a, b)`` shaped
    ``(2, 2, n_f)`` indexed ``[measured_port, driven_port, freq]``.
    """
    s11, s12, s21, s22 = s_true
    a = np.zeros((2, 2, n_f), dtype=np.complex128)
    b = np.zeros((2, 2, n_f), dtype=np.complex128)
    # drive port 1
    b2 = s21 / (1.0 - s22 * gamma_t)
    a[0, 0], a[1, 0] = 1.0, gamma_t * b2
    b[1, 0], b[0, 0] = b2, s11 + s12 * (gamma_t * b2)
    # drive port 2
    b1 = s12 / (1.0 - s11 * gamma_t)
    a[1, 1], a[0, 1] = 1.0, gamma_t * b1
    b[0, 1], b[1, 1] = b1, s22 + s21 * (gamma_t * b1)
    return a, b


def _through(n_f):
    """Ideal lossless through: S11=S22=0, S21=S12=1 (phase absorbed)."""
    one = np.ones(n_f, dtype=np.complex128)
    return (0.0 * one, one, one, 0.0 * one)


def _shunt_resistor(r_ohm, z0, n_f):
    """Exact S of a shunt resistor R across a line of impedance Z0."""
    one = np.ones(n_f, dtype=np.complex128)
    y = z0 / (2.0 * r_ohm)
    s11 = -y / (1.0 + y) * one
    s21 = 1.0 / (1.0 + y) * one
    return (s11, s21, s21, s11)


def _asymmetric_reciprocal(z0, n_f):
    """A reciprocal but ASYMMETRIC 2-port: series Z1, shunt Y, series Z2.

    Needed because the through line (S22 = 0) and the shunt resistor
    (S11 = S22) are both degenerate for the defect family in
    ``test_both_drive_swap_gap_requires_the_downstream_passivity_handle`` — with those
    fixtures the defect happens to collapse to a singular matrix and gets
    caught by accident rather than by design.
    """
    one = np.ones(n_f, dtype=np.complex128)
    z1, z2, y = 30.0 + 0j, 12.0 + 0j, 1.0 / (80.0 + 0j)
    # ABCD of series-shunt-series (reciprocal: AD - BC = 1).
    a = 1.0 + z1 * y
    b = z1 + z2 + z1 * y * z2
    c = y
    d = 1.0 + y * z2
    den = a + b / z0 + c * z0 + d
    s11 = (a + b / z0 - c * z0 - d) / den
    s22 = (-a + b / z0 - c * z0 + d) / den
    s21 = 2.0 / den
    s12 = 2.0 * (a * d - b * c) / den
    return (s11 * one, s12 * one, s21 * one, s22 * one)


# ---------------------------------------------------------------------------
# Ground truth must itself be checked — every test below depends on it
# ---------------------------------------------------------------------------

def test_asymmetric_fixture_is_what_it_claims():
    """Verify the asymmetric DUT independently, by node analysis.

    The review that produced this fixture found the previous battery passing
    for the wrong reason: both older fixtures were DEGENERATE for the defect
    they were supposed to discriminate, and nothing said so. A fixture whose
    properties are assumed rather than checked is how that happens, so this
    one is derived a second way — plain node analysis of the series-shunt-
    series network — and compared against the ABCD formula it ships with.
    """
    z0 = 48.5914
    z1, z2, y = 30.0, 12.0, 1.0 / 80.0
    s11, s12, s21, s22 = (v[0] for v in _asymmetric_reciprocal(z0, 1))

    zp = 1.0 / (y + 1.0 / (z2 + z0))          # shunt 1/y || (z2 + Z0)
    zin = z1 + zp
    v1 = zin / (z0 + zin)                      # unit EMF behind series Z0
    v2 = (v1 * zp / (z1 + zp)) * z0 / (z2 + z0)
    zin2 = z2 + 1.0 / (y + 1.0 / (z1 + z0))
    np.testing.assert_allclose(s11.real, (zin - z0) / (zin + z0), atol=1e-12)
    np.testing.assert_allclose(s21.real, 2.0 * v2, atol=1e-12)
    np.testing.assert_allclose(s22.real, (zin2 - z0) / (zin2 + z0), atol=1e-12)

    assert np.isclose(s12, s21, atol=1e-14), "fixture must be reciprocal"
    assert not np.isclose(abs(s11), abs(s22)), "fixture must be ASYMMETRIC"
    assert abs(s11) ** 2 + abs(s21) ** 2 < 1.0, "fixture must be passive"


# ---------------------------------------------------------------------------
# Why the obvious rule was rejected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gamma_t", [0.02, 0.05, 0.08])
def test_single_ratio_rule_has_a_terminator_floor(gamma_t):
    """`S[j,i] = b_j/a_i` reads |1 + S11 - S21| = |Gamma_t| EXACTLY.

    This is the measurement that killed the round-1 design: on a through line
    the whole terminator reflection lands in S11, so the through-line identity
    residual equals the terminator reflection no matter how good the rest of
    the implementation is. rfx's validated matched terminator only reaches
    |Gamma_t| < 0.08, so a 0.05-level identity gate was unreachable BY
    CONSTRUCTION rather than by any defect.
    """
    n_f = len(_FREQS)
    a, b = _plant(_through(n_f), gamma_t, n_f)
    s11_ratio = b[0, 0] / a[0, 0]
    s21_ratio = b[1, 0] / a[0, 0]
    resid = np.abs(1.0 + s11_ratio - s21_ratio)
    np.testing.assert_allclose(resid, gamma_t, rtol=1e-12)


def test_two_drive_solve_removes_the_terminator_floor():
    """The same planted fields, solved properly, are exact at every Gamma_t."""
    n_f = len(_FREQS)
    for gamma_t in (0.0, 0.02, 0.08, 0.2, 0.5):
        a, b = _plant(_through(n_f), gamma_t, n_f)
        out = solve_two_port_from_wave_amplitudes(a, b)
        np.testing.assert_allclose(np.abs(out.s_params[1, 0]), 1.0, atol=1e-13)
        np.testing.assert_allclose(np.abs(out.s_params[0, 0]), 0.0, atol=1e-13)
        # Conditioning stays mild across the whole range — this is why no
        # terminator improvement is a prerequisite for issue #489.
        assert np.all(out.cond_a < 3.0), (gamma_t, out.cond_a)


def test_conditioning_grows_slowly_with_terminator_reflection():
    """Pins the measured cond(A) trend so a regression is visible."""
    n_f = 1
    conds = {}
    for gamma_t in (0.0, 0.08, 0.5):
        a, b = _plant(_through(n_f), gamma_t, n_f)
        conds[gamma_t] = float(solve_two_port_from_wave_amplitudes(a, b).cond_a[0])
    # Closed form for this planted through line: cond = (1+|Gt|)/(1-|Gt|).
    # Assert against that, not a hand-picked bound: the |Gt|=0.5 value is
    # EXACTLY 3.0, and a bare `< 3.0` passed only because this platform's SVD
    # rounds to 2.999999999999999 — a LAPACK/numpy change could flip it
    # (cross-machine float flake class, project_ci_slow_suite_fixes).
    for gt, c in conds.items():
        assert c == pytest.approx((1.0 + gt) / (1.0 - gt), rel=1e-9, abs=1e-9)


# ---------------------------------------------------------------------------
# Exact DUT recovery
# ---------------------------------------------------------------------------

def test_recovers_shunt_resistor_exactly_despite_terminator():
    """A known-mismatch DUT is recovered exactly, terminator and all."""
    n_f = len(_FREQS)
    z0, r = 48.5914, 25.0
    s_true = _shunt_resistor(r, z0, n_f)
    a, b = _plant(s_true, 0.08, n_f)
    out = solve_two_port_from_wave_amplitudes(a, b)
    np.testing.assert_allclose(out.s_params[0, 0], s_true[0], atol=1e-12)
    np.testing.assert_allclose(out.s_params[1, 0], s_true[2], atol=1e-12)
    # Reciprocity is a property of the recovered matrix, not an input.
    np.testing.assert_allclose(
        out.s_params[1, 0], out.s_params[0, 1], atol=1e-12
    )


# ---------------------------------------------------------------------------
# Defect discrimination — a gate that cannot fail a broken input is not a gate
# ---------------------------------------------------------------------------

def test_swapped_receive_amplitude_one_drive_is_caught():
    """Consuming the wrong amplitude at the receive array, ONE drive only.

    The receive array holds a large transmitted wave and a small one returning
    off its terminator; taking the small one is a likely wiring mistake. This
    one-sided variant collapses |S21| and is easy to catch.

    NOTE: the realistic SYSTEMATIC version of this mistake — the same mislabel
    applied to BOTH drives — behaves completely differently and is NOT caught
    here. See the next test; do not read this one as covering that.
    """
    n_f = len(_FREQS)
    a, b = _plant(_through(n_f), 0.08, n_f)
    good = solve_two_port_from_wave_amplitudes(a, b)
    a_bad, b_bad = a.copy(), b.copy()
    b_bad[1, 0], a_bad[1, 0] = a[1, 0], b[1, 0]   # swap at port 2, drive 1
    bad = solve_two_port_from_wave_amplitudes(a_bad, b_bad)
    assert np.all(np.abs(good.s_params[1, 0]) > 0.99)
    assert np.all(np.abs(bad.s_params[1, 0]) < 0.2), np.abs(bad.s_params[1, 0])


def test_both_drive_swap_gap_requires_the_downstream_passivity_handle():
    """DOCUMENTED GAP: the systematic a/b mislabel evades both local witnesses.

    A real implementation bug mislabels incident vs outgoing at one port
    CONSISTENTLY, i.e. for both drives. That is the non-diagonal transform
    ``S' = N M^-1`` with ``M = [[1,0],[S21,S22]]``, ``N = [[S11,S12],[0,1]]``,
    giving ``S'21 = -S21/S22`` and ``S'12 = S12/S22``. Because the two share a
    magnitude, **magnitude reciprocity is exactly blind to it**, and the
    matrix stays well conditioned, so the ill-conditioning guard is silent too.

    The through line (S22=0) and the shunt resistor (S11=S22) are DEGENERATE
    for this defect — it collapses to a singular matrix there and gets caught
    by accident. That accident is why the original battery looked complete.
    An asymmetric DUT exposes the truth.

    The independent handle is PASSIVITY, not anything in this module:
    ``S'22 = 1/S22`` explodes for any well-matched passive DUT. Whatever ships
    on top of this solve must route through the repo's passivity self-check.
    """
    n_f = len(_FREQS)
    s_true = _asymmetric_reciprocal(48.5914, n_f)
    assert not np.isclose(abs(s_true[0][0]), abs(s_true[3][0]))  # truly asymmetric
    a, b = _plant(s_true, 0.05, n_f)
    a_bad, b_bad = a.copy(), b.copy()
    a_bad[1, :], b_bad[1, :] = b[1, :].copy(), a[1, :].copy()
    bad = solve_two_port_from_wave_amplitudes(a_bad, b_bad)

    # 1. reciprocity: blind (this is the finding)
    np.testing.assert_allclose(
        np.abs(bad.s_params[1, 0]), np.abs(bad.s_params[0, 1]), atol=1e-9
    )
    # 2. conditioning guard: silent — nowhere near the 1e3 default
    assert np.all(bad.cond_a < 1.0e2), bad.cond_a
    # 3. the recovered S is nonetheless grossly wrong, and PASSIVITY sees it
    assert np.all(np.abs(bad.s_params[1, 0]) > 2.0)
    np.testing.assert_allclose(
        np.abs(bad.s_params[1, 1]), 1.0 / np.abs(s_true[3]), rtol=1e-9
    )
    col_power = np.abs(bad.s_params[0, 0]) ** 2 + np.abs(bad.s_params[1, 0]) ** 2
    # Measured 317 against a passive limit of 1 — the defect is not marginal,
    # it is two orders of magnitude out. The floor is well below the measured
    # value but far above 1, so it stays meaningful if the fixture is retuned
    # while still failing loudly if the defect ever stopped being gross. The
    # docstring quotes this number; a drift here should force re-reading it.
    assert np.all(col_power > 100.0), col_power


@pytest.mark.parametrize("scale", [1.05, 1.5, 0.8])
def test_reciprocity_catches_a_per_port_amplitude_error(scale):
    """A per-port CALIBRATION error is visible to the reciprocity witness.

    Scaling both amplitudes at one port by ``c`` is the similarity
    ``S -> D S D^-1`` with ``D = diag(1, c)``, which sends
    ``S21 -> c*S21`` and ``S12 -> S12/c``. The product is invariant but the
    RATIO moves by ``c^2``, so reciprocity sees it — including the 5%
    amplitude-calibration defect the design review flagged as untracked.
    """
    n_f = len(_FREQS)
    s_true = _shunt_resistor(25.0, 48.5914, n_f)
    a, b = _plant(s_true, 0.05, n_f)
    a_bad, b_bad = a.copy(), b.copy()
    a_bad[1, :] *= scale
    b_bad[1, :] *= scale
    bad = solve_two_port_from_wave_amplitudes(a_bad, b_bad)
    ratio = np.abs(bad.s_params[1, 0]) / np.abs(bad.s_params[0, 1])
    np.testing.assert_allclose(ratio, scale ** 2, rtol=1e-9)
    dev = np.abs(np.abs(bad.s_params[1, 0]) - np.abs(bad.s_params[0, 1]))
    assert np.all(dev > 0.04), dev


@pytest.mark.parametrize(
    "diag_factor", [-1.0, np.exp(-1j * 0.7), 1j], ids=["sign", "phase", "quarter"]
)
def test_reciprocity_is_BLIND_to_any_unit_modulus_per_port_factor(diag_factor):
    """The witness's blind spot, pinned so nobody over-trusts it.

    Any per-port factor of unit modulus — a sign flip from a wave-split
    orientation mistake, or a reference-plane shift — is a diagonal similarity
    ``D S D^-1`` with ``|D_jj| = 1``. It sends ``S21 -> c*S21`` and
    ``S12 -> S12/c`` with ``|c| = 1``, so BOTH magnitudes are untouched and
    reciprocity cannot see it at all.

    This test was written expecting the opposite and failed; the algebra above
    is why. Consequence for issue #489: reciprocity separates amplitude and
    orientation-SCALE defects, but a first gate must NOT be advertised as
    covering orientation sign or reference-plane errors — those need an
    independent handle (a known-phase DUT, or an invariance check).
    """
    n_f = len(_FREQS)
    s_true = _shunt_resistor(25.0, 48.5914, n_f)
    a, b = _plant(s_true, 0.05, n_f)
    good = solve_two_port_from_wave_amplitudes(a, b)
    a_bad, b_bad = a.copy(), b.copy()
    a_bad[1, :] *= diag_factor
    b_bad[1, :] *= diag_factor
    bad = solve_two_port_from_wave_amplitudes(a_bad, b_bad)
    np.testing.assert_allclose(
        np.abs(bad.s_params[1, 0]), np.abs(bad.s_params[0, 1]), atol=1e-12
    )
    # ...and it is invisible against the correct result too, not just internally.
    np.testing.assert_allclose(
        np.abs(bad.s_params[1, 0]), np.abs(good.s_params[1, 0]), atol=1e-12
    )


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def test_ill_conditioned_drives_warn_and_are_reported():
    """Two nearly identical drives must not silently produce a plausible S."""
    n_f = 2
    a = np.zeros((2, 2, n_f), dtype=np.complex128)
    b = np.zeros((2, 2, n_f), dtype=np.complex128)
    a[0, 0], a[1, 0] = 1.0, 1.0
    a[0, 1], a[1, 1] = 1.0, 1.0 + 1e-12    # second drive ~ first
    b[:, :] = 0.5
    with pytest.warns(UserWarning, match="nearly linearly dependent") as rec:
        out = solve_two_port_from_wave_amplitudes(a, b)
    assert np.all(out.cond_a > 1e3)
    # The warning must NOT read as "below this threshold is trustworthy" —
    # cond(A) also multiplies the caller's amplitude noise (review finding).
    msg = str(rec[0].message).lower()
    assert "degeneracy only" in msg and "not a reliability certificate" in msg


def test_shape_and_finiteness_contract():
    with pytest.raises(ValueError, match="same shape"):
        solve_two_port_from_wave_amplitudes(
            np.zeros((2, 2, 3), dtype=complex), np.zeros((2, 2, 4), dtype=complex)
        )
    with pytest.raises(ValueError, match=r"\(2, 2, n_freqs\)"):
        solve_two_port_from_wave_amplitudes(
            np.zeros((3, 2, 3), dtype=complex), np.zeros((3, 2, 3), dtype=complex)
        )
    # A NaN bin is reported as NaN, never quietly dropped or interpolated.
    a, b = _plant(_through(2), 0.05, 2)
    a[0, 0, 1] = np.nan
    out = solve_two_port_from_wave_amplitudes(a, b)
    assert np.all(np.isfinite(out.s_params[:, :, 0]))
    assert np.all(np.isnan(out.s_params[:, :, 1]))
