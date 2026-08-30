"""Tests for ``compute_coax_msl_transition`` (issue #489 leg 4).

Structure (mirrors ``tests/test_coax_two_port_fdtd.py`` and
``tests/test_coax_two_port_referee_header.py``, the two closest
precedents):

1. Pure-assembly tests (no FDTD) for
   :func:`rfx.api._sparams._assemble_coax_msl_transition_from_voltages` —
   PLANTED analytic voltages recover a KNOWN S-matrix, and a dedicated
   regression test proves the per-port ``sqrt(Z0)`` power-wave
   normalization is load-bearing (the pre-declared "impedance-convention
   mismatch" failure mode: skipping it silently corrupts the off-diagonal
   when the two ports' reference impedances differ).
2. A PREDECLARATION fill-contract test (UNRUN <=> no numbers, mirrors
   ``test_coax_two_port_referee_header.py``'s
   ``REPRODUCE_GATE_RECORD`` pattern) for the one committed FDTD fixture.
3. The FDTD fixture itself (``@pytest.mark.slow_physics``), asserting the
   predeclared witnesses at DIAGNOSTIC honesty level — this gate pins the
   MEASURED envelope of THIS fixture only, no claim beyond it.
4. ATTEMPT 2 (PI-directed, R2's own escape clause): attempt 1's own
   discriminant named a concrete implementation defect (the MSL probe
   ladder spans too small a fraction of a guided wavelength for the
   matrix-pencil wave split to resolve) — R2 permits exactly one retry
   when a prior attempt names such a defect. This section lengthens the
   MSL probe ladder and widens the MSL port's x-CPML clearance
   (INSTRUMENT changes only) while keeping the junction geometry
   byte-identical to attempt 1's committed fixture (asserted, not just
   claimed). Attempt 1's own predeclaration, fixture, and regression locks
   above are UNCHANGED and UNDELETED — they remain a valid, reproducible
   historical record of a deliberately short ladder.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfx.api._sparams import (
    _assemble_coax_msl_transition_from_voltages,
    _mixed_reciprocity_deviation,
)
from tests._gate_policy import gate_from_envelope


# ---------------------------------------------------------------------------
# Part 1 -- pure assembly (no FDTD)
# ---------------------------------------------------------------------------


def _plant_ab_power_wave(s_true, gamma_t, n_f):
    """Power-wave a/b at both ports for both drives, terminator ``gamma_t``.

    Identical signal-flow construction to
    ``tests/test_coax_two_port_fdtd.py::_plant_ab`` / the underlying
    ``tests/test_coax_two_port_solve.py::_plant`` (``a[j, i]`` / ``b[j, i]``
    indexed [measured_port, driven_port, freq]). Because this is a pure
    signal-flow identity with no Z0 anywhere, the resulting ``a``/``b`` are
    already POWER waves for ANY consistent choice of per-port reference
    impedance -- unlike the coax-coax precedent, this test needs that
    distinction because port 0 (coax) and port 1 (msl) get DIFFERENT
    reference impedances below.
    """
    s11, s12, s21, s22 = s_true
    a = np.zeros((2, 2, n_f), dtype=np.complex128)
    b = np.zeros((2, 2, n_f), dtype=np.complex128)
    b2 = s21 / (1.0 - s22 * gamma_t)
    a[0, 0], a[1, 0] = 1.0, gamma_t * b2
    b[1, 0], b[0, 0] = b2, s11 + s12 * (gamma_t * b2)
    b1 = s12 / (1.0 - s11 * gamma_t)
    a[1, 1], a[0, 1] = 1.0, gamma_t * b1
    b[0, 1], b[1, 1] = b1, s22 + s21 * (gamma_t * b1)
    return a, b


def _voltages_from_ab(a, b, *, gamma, planes_m, ref_m, load_below):
    """Invert the extractor's own extrapolation to build V(axis) from known a/b.

    Exact copy of ``tests/test_coax_two_port_fdtd.py::_voltages_from_ab``'s
    algebra (see that function's docstring for the derivation) -- kept as a
    local, self-contained copy rather than a cross-module import so this
    test file does not depend on another test file's internals staying
    stable. ``a``/``b`` here must already be VOLT-wave amplitudes (the
    ``coaxial_line_reflection_from_plane_voltages`` convention), not power
    waves -- callers multiply by ``sqrt(Z0)`` before calling this.
    """
    planes_m = np.asarray(planes_m, dtype=np.float64)
    a = np.atleast_1d(np.asarray(a, dtype=np.complex128))
    b = np.atleast_1d(np.asarray(b, dtype=np.complex128))
    gamma = np.broadcast_to(np.asarray(gamma, dtype=np.complex128), a.shape)
    p0 = float(planes_m.mean())
    pr = ref_m - p0
    if load_below:
        A = b * np.exp(-gamma * pr)
        B = a * np.exp(+gamma * pr)
    else:
        A = a * np.exp(-gamma * pr)
        B = b * np.exp(+gamma * pr)
    pc = planes_m - p0
    return (
        np.exp(np.multiply.outer(pc, gamma)) * A[None, :]
        + np.exp(-np.multiply.outer(pc, gamma)) * B[None, :]
    )


# A deliberately ASYMMETRIC (S11 != S22), reciprocal (S12 == S21) truth --
# same reasoning as the coax-coax precedent: a symmetric fixture cannot
# discriminate a systematic a/b mislabel, and per-frequency-varying values
# catch a fixed/transposed frequency-index bug in the assembly loop.
_S_TRUE = (
    np.array([0.12 - 0.04j, 0.18 + 0.02j, -0.03 - 0.09j]),   # S11 (coax)
    np.array([0.62 + 0.18j, 0.55 - 0.25j, 0.48 + 0.30j]),    # S12
    np.array([0.62 + 0.18j, 0.55 - 0.25j, 0.48 + 0.30j]),    # S21 == S12
    np.array([-0.08 - 0.05j, 0.06 + 0.10j, 0.11 - 0.02j]),   # S22 (msl)
)
_N_F = 3
_GAMMA_T = 0.05
_GAMMA_LINE = 1j * np.array([35.0, 47.0, 61.0])   # distinct per-frequency beta
_Z0_COAX = 50.0
_Z0_MSL = 74.3   # deliberately far from 50 ohm (Hammerstad-Jensen microstrip range)


def _s_matrix_per_freq(s_true, n_f):
    return [
        np.array([[s_true[0][fi], s_true[1][fi]], [s_true[2][fi], s_true[3][fi]]])
        for fi in range(n_f)
    ]


def _build_planted_fixture(z0_coax, z0_msl):
    """Common planted geometry shared by the tests below.

    Coax reference plane sits ABOVE the coax probe centroid (load_below=False,
    mirrors the ``ref_top_m`` case in the coax-coax precedent -- the coax
    stub's own junction reference plane sits past the end of its probe
    array, toward the DUT). The MSL reference plane sits BELOW the MSL probe
    centroid (load_below=True -- the junction is on the near side of the
    probe ladder, which steps away from the port's own feed toward the
    junction).
    """
    a_pow, b_pow = _plant_ab_power_wave(_S_TRUE, _GAMMA_T, _N_F)
    sqrt_z0 = np.array([np.sqrt(z0_coax), np.sqrt(z0_msl)])
    a_volt = a_pow * sqrt_z0[:, None, None]
    b_volt = b_pow * sqrt_z0[:, None, None]

    z_coax_planes_m = np.array([0.00, 0.01, 0.02, 0.03, 0.04, 0.05])
    ref_coax_m = 0.07  # above the probe centroid -> load_below=False
    # Strictly increasing, as coaxial_line_reflection_from_plane_voltages
    # requires -- compute_coax_msl_transition sorts its own MSL probe
    # ladder into this order before calling the extractor (a "-x"-facing
    # port's own ladder comes back DECREASING in x from
    # msl_probe_x_coords_n; see the sort in the real method).
    x_msl_planes_m = np.array([0.15, 0.16, 0.17, 0.18, 0.19, 0.20])
    ref_msl_m = 0.13   # below the probe centroid -> load_below=True

    v_coax_by_drive = np.stack([
        _voltages_from_ab(
            a_volt[0, drive], b_volt[0, drive], gamma=_GAMMA_LINE,
            planes_m=z_coax_planes_m, ref_m=ref_coax_m, load_below=False,
        )
        for drive in range(2)
    ], axis=0)
    v_msl_by_drive = np.stack([
        _voltages_from_ab(
            a_volt[1, drive], b_volt[1, drive], gamma=_GAMMA_LINE,
            planes_m=x_msl_planes_m, ref_m=ref_msl_m, load_below=True,
        )
        for drive in range(2)
    ], axis=0)
    return dict(
        z_coax_planes_m=z_coax_planes_m, x_msl_planes_m=x_msl_planes_m,
        ref_coax_m=ref_coax_m, ref_msl_m=ref_msl_m,
        v_coax_by_drive=v_coax_by_drive, v_msl_by_drive=v_msl_by_drive,
    )


def test_planted_voltages_recover_known_s_matrix_with_unequal_z0():
    """Full pure-assembly recovers the KNOWN S-matrix under UNEQUAL port Z0.

    This is the primary correctness witness for the "impedance-convention
    mismatch" pre-declared failure mode: port 0 (coax, 50 ohm) and port 1
    (msl, 74.3 ohm) have DIFFERENT reference impedances, so a correct
    power-wave assembly is required to recover ``_S_TRUE`` exactly on this
    noise-free synthetic field.
    """
    fx = _build_planted_fixture(_Z0_COAX, _Z0_MSL)
    s_params, cond_a, cond_a_eq, rec_resid, fit_resid, gamma_fit, a_inc, b_out = (
        _assemble_coax_msl_transition_from_voltages(
            **fx, z0_coax=_Z0_COAX, z0_msl=_Z0_MSL,
        )
    )
    s_mat_per_freq = _s_matrix_per_freq(_S_TRUE, _N_F)
    for fi in range(_N_F):
        np.testing.assert_allclose(s_params[:, :, fi], s_mat_per_freq[fi], atol=1e-9)
    assert np.all(np.isfinite(rec_resid)) and np.all(rec_resid < 1e-9)
    assert np.all(np.isfinite(fit_resid)) and np.all(fit_resid < 1e-9)
    assert np.all(cond_a < 3.0)
    assert np.all(cond_a_eq < 3.0)
    assert a_inc.shape == (2, 2, _N_F) and b_out.shape == (2, 2, _N_F)
    np.testing.assert_allclose(
        gamma_fit, np.broadcast_to(_GAMMA_LINE, gamma_fit.shape), atol=1e-6
    )


def test_unequal_z0_normalization_is_required_off_diagonal_only():
    """Regression witness: skipping the ``sqrt(Z0)`` step corrupts S12/S21 ONLY.

    Directly operationalizes the algebra in
    :func:`rfx.api._sparams._assemble_coax_msl_transition_from_voltages`'s
    own docstring: solving on RAW (un-normalized) volt-wave amplitudes
    leaves each diagonal entry exactly correct (``sqrt(Zi/Zi) = 1``) but
    scales each off-diagonal entry by ``sqrt(Zi/Zj)`` relative to the true
    power-wave value. This test calls the extractor's own two building
    blocks directly (bypassing the ``sqrt(Z0)`` division the function under
    test performs) to reproduce that DEFECT on the same planted fixture,
    and confirms: (a) it actually differs from the true S, (b) by the
    predicted closed-form factor, (c) the diagonal is untouched.
    """
    from rfx.sources.coaxial_port import (
        coaxial_line_reflection_from_plane_voltages,
        solve_two_port_from_wave_amplitudes,
    )

    fx = _build_planted_fixture(_Z0_COAX, _Z0_MSL)
    n_f = _N_F
    a_raw = np.zeros((2, 2, n_f), dtype=np.complex128)
    b_raw = np.zeros((2, 2, n_f), dtype=np.complex128)
    for drive_idx in range(2):
        for fi in range(n_f):
            out_c = coaxial_line_reflection_from_plane_voltages(
                fx["z_coax_planes_m"], fx["v_coax_by_drive"][drive_idx, :, fi],
                reference_plane_m=fx["ref_coax_m"],
            )
            out_m = coaxial_line_reflection_from_plane_voltages(
                fx["x_msl_planes_m"], fx["v_msl_by_drive"][drive_idx, :, fi],
                reference_plane_m=fx["ref_msl_m"],
            )
            a_raw[0, drive_idx, fi] = out_c.backward_amp   # NO sqrt(Z0) division
            b_raw[0, drive_idx, fi] = out_c.forward_amp
            a_raw[1, drive_idx, fi] = out_m.backward_amp
            b_raw[1, drive_idx, fi] = out_m.forward_amp

    solve = solve_two_port_from_wave_amplitudes(a_raw, b_raw)
    s_wrong = solve.s_params
    s_mat_per_freq = _s_matrix_per_freq(_S_TRUE, n_f)

    ratio = np.sqrt(_Z0_COAX / _Z0_MSL)  # predicted S'_01 = S_01 * sqrt(Z0/Z1)
    for fi in range(n_f):
        true = s_mat_per_freq[fi]
        # Diagonal: untouched by the missing normalization.
        np.testing.assert_allclose(s_wrong[0, 0, fi], true[0, 0], atol=1e-9)
        np.testing.assert_allclose(s_wrong[1, 1, fi], true[1, 1], atol=1e-9)
        # Off-diagonal: WRONG, and wrong by exactly the predicted factor.
        assert abs(s_wrong[0, 1, fi] - true[0, 1]) > 1e-3
        assert abs(s_wrong[1, 0, fi] - true[1, 0]) > 1e-3
        np.testing.assert_allclose(s_wrong[0, 1, fi], true[0, 1] * ratio, atol=1e-9)
        np.testing.assert_allclose(s_wrong[1, 0, fi], true[1, 0] / ratio, atol=1e-9)


def test_equal_z0_makes_normalization_a_no_op():
    """Sanity cross-check: when both ports share one Z0, the fix is inert.

    Confirms the ``sqrt(Z0)`` division degenerates to a no-op common-scale
    factor when the two families happen to share a reference impedance --
    exactly why the coax-coax two-port lane never needed this step.
    """
    fx = _build_planted_fixture(50.0, 50.0)
    s_params, *_ = _assemble_coax_msl_transition_from_voltages(
        **fx, z0_coax=50.0, z0_msl=50.0,
    )
    s_mat_per_freq = _s_matrix_per_freq(_S_TRUE, _N_F)
    for fi in range(_N_F):
        np.testing.assert_allclose(s_params[:, :, fi], s_mat_per_freq[fi], atol=1e-9)


def test_rejects_mismatched_frequency_axes():
    fx = _build_planted_fixture(_Z0_COAX, _Z0_MSL)
    fx["v_msl_by_drive"] = fx["v_msl_by_drive"][:, :, :-1]
    with pytest.raises(ValueError, match="frequency axis"):
        _assemble_coax_msl_transition_from_voltages(**fx, z0_coax=_Z0_COAX, z0_msl=_Z0_MSL)


def test_rejects_missing_drive_axis():
    fx = _build_planted_fixture(_Z0_COAX, _Z0_MSL)
    fx["v_coax_by_drive"] = fx["v_coax_by_drive"][0]
    with pytest.raises(ValueError, match="leading axis"):
        _assemble_coax_msl_transition_from_voltages(**fx, z0_coax=_Z0_COAX, z0_msl=_Z0_MSL)


@pytest.mark.parametrize("bad_z0", [0.0, -50.0, float("nan"), float("inf")])
def test_rejects_non_positive_finite_z0(bad_z0):
    fx = _build_planted_fixture(_Z0_COAX, _Z0_MSL)
    with pytest.raises(ValueError):
        _assemble_coax_msl_transition_from_voltages(**fx, z0_coax=bad_z0, z0_msl=_Z0_MSL)
    with pytest.raises(ValueError):
        _assemble_coax_msl_transition_from_voltages(**fx, z0_coax=_Z0_COAX, z0_msl=bad_z0)


# ---------------------------------------------------------------------------
# Part 2 -- PREDECLARATION (fill-contract; committed BEFORE the reported run)
# ---------------------------------------------------------------------------

PREDECLARATION = {
    "leg": "issue #489 leg 4 (coax<->MSL transition), first fixture",
    "fixture_choice": (
        "VERTICAL launch (coax axis along z, landing on a grounded "
        "substrate edge; MSL trace along x), NOT an in-plane collinear "
        "launch. Justification (rasterization arithmetic, decided BEFORE "
        "running): every coax 'line' primitive this method reuses "
        "(stamp_coaxial_line, build_coaxial_tem_plane_source_specs, "
        "coaxial_line_plane_voltage) is hardcoded to a z-propagation axis "
        "(circular cross-section in x-y, verified by reading "
        "rfx/sources/coaxial_port.py: no axis parameter exists, and "
        "build_coaxial_tem_plane_source_specs raises NotImplementedError "
        "for axis != 'z'). An in-plane (both-along-x) launch would need "
        "NEW x-axis analogues of all three -- more invasive, and violates "
        "the repo's 'imitate the canonical example' rule (there is no "
        "existing x-axis coax primitive to imitate). The vertical launch "
        "reuses every coax primitive UNCHANGED (mirrors "
        "compute_coaxial_two_port's own single-ended stub verbatim) and "
        "every MSL primitive UNCHANGED (ordinary Box/Cylinder geometry + "
        "add_msl_port, mirrors compute_mixed_s_matrix's own MSL "
        "consumption) -- the only NEW code this leg needed was the "
        "power-wave cross-family normalization "
        "(_assemble_coax_msl_transition_from_voltages) and the junction "
        "geometry itself (ground plane, clearance hole, pin-to-trace "
        "post), which the caller owns via ordinary sim.add(...) calls."
    ),
    "geometry_summary": (
        "dx=100um throughout (one grid, both families). Coax: pin=0.2mm, "
        "outer=0.6mm (PTFE eps_r=2.1 default) -> annulus 4.0 cells (above "
        "the 3.5-cell under-resolved floor). Ground plane: 1-cell PEC "
        "layer at node N_GND=25 (z=2.5mm), full x-y extent, with a "
        "clearance disk (radius = pin + 2 cells) carved around the pin "
        "axis. Substrate: RO4350B-like eps_r=3.66, 3 cells (300um) above "
        "the ground node. Trace: 1-cell PEC, width 600um, starting at "
        "junction_x=1.0mm and running to feed_x=4.0mm where the MSL port "
        "sits (direction='-x', facing back toward the junction, mirrors "
        "the #488 mixed-lane fixture's own convention). Pin-to-trace "
        "post: a PEC Cylinder (radius = pin radius) connecting the "
        "ground node through the substrate to the trace node. No z-"
        "boundary is placed on an exact node multiple -- Box corners use "
        "the CELL-MIDPOINT convention ((n +/- 0.5)*dx, per rfx/geometry/"
        "csg.py's own Box docstring); Cylinder heights use a DIFFERENT "
        "mechanism, a full stamp_coaxial_line-style +2*dz margin on each "
        "side (Cylinder's inclusive <= height check has the same knife-"
        "edge risk but is not corner-based, so the midpoint recipe does "
        "not literally apply -- margin is the analogous fix). See the "
        "module docstring note below on why exact multiples are unsafe "
        "on this grid."
    ),
    "healthy_envelope_predeclared": (
        "|S11| (coax) well below 1 (reflection, not total block); |S21| "
        "passivity-bounded (<=1 with the documented Yee/extraction "
        "envelope); reciprocity |S12-S21| within an envelope informed by "
        "#488's own measured 9-30% class on ITS mixed fixture (this "
        "fixture, with NO matching structure and a bare via-style post, "
        "was explicitly flagged in the task scoping as likely WORSE)."
    ),
    "falsifiers_predeclared": {
        "amplitude": (
            "|S| > 1 beyond the documented single-run Yee/near-cutoff "
            "envelope (test_sparam_passivity_guard's ~1.05-1.1 tight-path "
            "convention) = extraction defect (current sign/scale or "
            "reference plane) -- discriminate by inspecting recurrence_"
            "residual/fit_residual (large = the matrix-pencil fit itself "
            "is unreliable, e.g. multi-mode contamination near a source) "
            "and cond_a (large = the two-drive SYSTEM is ill-conditioned, "
            "independent of whether the fit itself is clean)."
        ),
        "reciprocity": (
            "|S12| vs |S21| disagreeing far outside the #488-informed "
            "envelope = reference-plane/normalization defect -- "
            "discriminate via cond_a: the two-drive solve's own docstring "
            "states cond(A) 'multiplies whatever noise is on the measured "
            "amplitudes', so a reciprocity blowout ACCOMPANIED BY cond_a "
            ">> 1000 points at near-degenerate drives (both ports seeing "
            "almost the same field -- e.g. both strongly reflecting their "
            "own port, so the two drives' incident-wave columns are "
            "nearly parallel), NOT a sign/normalization bug in the "
            "assembler itself (which is independently verified correct "
            "by Part 1's planted-voltage tests above)."
        ),
    },
    "status": "RUN",
    "post_review_correction": (
        "PR #581 adversarial review (finding B2) refuted the reciprocity "
        "discriminant's mechanism above by three independent checks on "
        "this fixture's own data: (i) cond_a is almost ENTIRELY column-"
        "norm ratio, not near-parallel columns -- after per-column "
        "equilibration cond_a_equilibrated = 1.0004/1.0001/1.0040 and the "
        "two incident-wave columns are near-ORTHOGONAL (normalized "
        "overlap 4e-4/7e-5/4e-3), the opposite of 'nearly parallel'; (ii) "
        "the 'both ports strongly reflecting' premise fails on this "
        "fixture's own |S22| = 0.000/0.000/0.500 (the MSL side is NOT "
        "strongly reflecting at the two lower bins); (iii) the signature "
        "that DOES match is the two-drive solve's OWN warning's second "
        "branch, 'one drive that failed to excite': the MSL drive's own "
        "incident amplitude a_inc[1,1,:] is 8.2e-14/1.3e-11/7.1e-17 "
        "against the coax drive's a_inc[0,0,:] of 5.8e-9/5.8e-8/3.0e-9 -- "
        "5 to 9 orders of magnitude apart. Superseded by "
        "POST_REVIEW_DISCRIMINANT below, which independently confirms the "
        "PREDECLARED extraction-class failure mode (not junction physics) "
        "is the better-supported explanation: the MSL ladder spans only "
        "0.34%-3.37% of the guided wavelength at these three frequencies, "
        "and the fitted Im(gamma) on the MSL array does not track the "
        "analytic Hammerstad-Jensen beta at all (ratios 4x-32x off, "
        "non-monotonic with frequency -- see the discriminant below and "
        "the class docstring on CoaxMSLTransitionResult)."
    ),
}


def _msl_guided_beta_and_ladder_fraction(freqs_hz, *, ladder_span_m):
    """Analytic Hammerstad-Jensen beta and ladder-span/lambda_g fraction.

    Pure function (no FDTD) -- the B3 discriminant computed from the SAME
    committed fixture's own geometry constants, independent of any run.
    """
    from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff
    from rfx.core.yee import EPS_0, MU_0

    c0 = 1.0 / np.sqrt(MU_0 * EPS_0)
    _, eps_eff_hj = hammerstad_jensen_z0_eps_eff(W_TRACE, H_SUB, EPS_SUB)
    beta = 2.0 * np.pi * np.asarray(freqs_hz) * np.sqrt(eps_eff_hj) / c0
    lambda_g = 2.0 * np.pi / beta
    return beta, ladder_span_m / lambda_g


def test_post_review_discriminant_msl_ladder_too_short_for_pencil_fit():
    """B3 discriminant (issue #581 review) -- no FDTD, pure geometry/analytic.

    The committed fixture's MSL probe ladder (probe_spacing_cells=2,
    probe_count=6 -> span = 2*5 = 10 cells = 1.000 mm) spans a tiny, and
    FREQUENCY-DEPENDENT, fraction of the guided wavelength at each measured
    bin. A matrix-pencil forward/backward decomposition needs the span to
    resolve a meaningful fraction of a cycle; this locks in that it does
    not, at any of the three measured frequencies -- independent evidence
    (alongside POST_REVIEW_DISCRIMINANT's fitted-gamma-vs-analytic-beta
    comparison, which needs the FDTD result and is reported in the PR body
    /the slow_physics test below) that the reciprocity/degeneracy finding
    is an MSL wave-extraction instrument-scoping limit, not a claim about
    the ladder span at published frequencies elsewhere in this file.
    """
    freqs = np.array([0.6e9, 3.3e9, 6.0e9])
    ladder_span_m = 2 * (6 - 1) * DX  # probe_spacing_cells * (probe_count-1) * dx
    beta, frac = _msl_guided_beta_and_ladder_fraction(freqs, ladder_span_m=ladder_span_m)
    assert np.all(frac < 0.05), (
        f"MSL ladder now spans {frac} of lambda_g at {freqs/1e9} GHz -- if "
        "this rises above the ~5% floor a matrix-pencil fit needs, the "
        "instrument-scoping finding may no longer apply; re-derive it "
        "before trusting a passing reciprocity assertion elsewhere."
    )


def test_predeclaration_has_required_fields():
    required = {
        "leg", "fixture_choice", "geometry_summary",
        "healthy_envelope_predeclared", "falsifiers_predeclared", "status",
    }
    missing = required - set(PREDECLARATION.keys())
    assert not missing, f"PREDECLARATION missing fields: {missing}"
    assert set(PREDECLARATION["falsifiers_predeclared"].keys()) == {
        "amplitude", "reciprocity",
    }


# ---------------------------------------------------------------------------
# Part 3 -- the one committed FDTD fixture (@pytest.mark.slow_physics)
# ---------------------------------------------------------------------------
#
# R3 self-audit / R2 accounting for this fixture (see PR body for the full
# form): ONE pre-declared physics attempt was spent. Two CONSTRUCTION
# defects were found and fixed before the declared attempt was read as a
# result (both are plumbing, not a mechanism-hypothesis loop, per the
# repo's own R2 distinction): (1) a missing z-layer of PEC connectivity
# between the coax stub and the ground-plane/pin-post geometry (an
# off-by-one in the original z-index bookkeeping), and (2) exact-multiple-
# of-dx Box/Cylinder corners tripping the float32 rasterization "knife
# edge" documented in rfx/geometry/csg.py's Box docstring (every boundary
# in this fixture happened to land exactly on a node plane -- the
# docstring's own worked recipe, cell-midpoint corners, is what fixed it).
# Neither fix changed the fixture's PHYSICAL dimensions or the intended
# junction topology; both are the kind of bug a green construction check
# would have caught before spending any FDTD time, and are recorded here
# only because they were found by actually running it, not by review.
#
# THE DECLARED RESULT: the fixture runs, is internally self-consistent
# (finite, deterministic, settles below -40 dB), but the RECIPROCITY
# falsifier fires -- badly (|S12| vs |S21| disagree by 94-100% across the
# three measured frequencies), with raw cond_a in the 1e3-1e7 range at
# every bin.
#
# ATTRIBUTION (CORRECTED after PR #581 adversarial review, findings B2/B3
# -- the ORIGINAL attribution in this comment, "near-degenerate two-drive
# amplification from strong junction reflection", was WRONG and is
# preserved only in PREDECLARATION["post_review_correction"] above as the
# historical record of what was first written and why it did not survive
# its own data):
#
# B2: raw cond_a is almost entirely a column-NORM-ratio artifact, not
# geometric near-parallelism. After per-column equilibration,
# cond_a_equilibrated = 1.0004/1.0001/1.0040 (see the assembler's own
# docstring) and the two incident-wave columns are near-ORTHOGONAL
# (normalized overlap 4e-4/7e-5/4e-3) -- the OPPOSITE of "nearly
# parallel". The "both ports strongly reflecting" premise also fails on
# this fixture's own |S22| = 0.000/0.000/0.500 (not uniformly near 1).
# What DOES match is the two-drive solve's own warning's SECOND branch,
# "one drive that failed to excite": a_inc[1,1,:] (MSL's own incident
# wave when MSL drives) is 8.2e-14/1.3e-11/7.1e-17 against
# a_inc[0,0,:] (coax's own incident wave when coax drives) of
# 5.8e-9/5.8e-8/3.0e-9 -- 5 to 9 orders of magnitude apart. This is a
# drive-AMPLITUDE mismatch between the two unrelated source
# constructions (coax TEM plane source vs MSL Ez injection), not
# evidence of near-degenerate DIRECTIONS.
#
# B3: the PREDECLARED extraction-class failure mode (not junction
# physics) is positively supported, not ruled out, by this fixture's own
# numbers. The MSL probe ladder spans only 1.000 mm = 0.34%-3.37% of the
# guided wavelength across the 3 measured frequencies (locked in by
# test_post_review_discriminant_msl_ladder_too_short_for_pencil_fit,
# analytic/no-FDTD) -- far too short for a matrix-pencil forward/backward
# split to resolve. The fitted Im(gamma) on the MSL array does not track
# analytic Hammerstad-Jensen beta (21.2/116.4/211.7 rad/m) at all: the
# coax-driven fit gives 673.0/853.6/885.0 rad/m (4-32x too high, and
# nearly FREQUENCY-FLAT across a 10x frequency span where the true beta
# is not -- "locked onto a spatial feature" of the fixed probe spacing,
# not the guided mode), and the msl-own-drive fit gives
# 4.5/36.2/2881.3 rad/m with Re(gamma) = 5619/5964/139 rad/m -- a decay
# length near 1 cell, i.e. not a propagating/decaying wave at all. Two of
# this array's own fit_residual values (0.0313, 0.0348) already cross the
# PREDECLARED "large = fit unreliable" line, and |S22| = 0.00000 at bins
# 0-1 is a b/a ratio of 1e-8 to 1e-11 -- below float32 relative precision,
# a degenerate split, which directly CONTRADICTS the via-reflector story
# (that story demands |S22| -> 1, a real near-total reflection, not a
# numerically-degenerate near-zero ratio). test_coax_msl_transition_
# first_fixture_diagnostic (below) locks this comparison in as an
# assertion, not just prose.
#
# HONEST STATEMENT (per review): degeneracy is observed. Whether the
# junction's own physical reflection or the MSL wave-extraction's own
# instrument-scoping limit (the PREDECLARED failure class) is the
# dominant cause is UNRESOLVED by this one fixture alone -- but the
# discriminant above POSITIVELY SUPPORTS the extraction-class
# explanation and does not support the junction-reflection story, so
# that is the better-supported reading, not a coin flip. This also NAMES
# the implementation defect that would justify a future SECOND R2
# attempt (a longer MSL probe ladder, >= 0.25 lambda_g at the lowest
# measured frequency -- NOT run in this PR).
#
# A NAMED THIRD candidate mechanism, not ruled out either (issue #581
# review B4): preflight's own x-CPML clearance warning on this fixture
# ("distance to nearest x-CPML = 200um < recommended 600um... Source-side
# CPML reflection may inflate |S11|") means the MSL port's own near-field
# measurement sits close enough to its absorbing boundary that CPML
# reflection could also be contaminating the MSL-side extraction,
# independent of the ladder-length limit above. Not disentangled from
# the ladder-length finding in this PR.
#
# Per R2, this STOPS here: no second geometry/probe-ladder attempt in
# this session -- the numeric re-derivations above are instruments on
# the ALREADY-COLLECTED run (same geometry, same n_steps; no new FDTD),
# not a second physics attempt. This is reported as an honest FIRST
# diagnostic measurement, not a validated transition -- the test below
# locks the measured (bad) reciprocity/passivity/discriminant numbers as
# a reproducibility witness, not as an accuracy claim.

DX = 100e-6
PIN_R = 0.2e-3
OUTER_R = 0.6e-3
EPS_COAX = 2.1
H_SUB = 300e-6
EPS_SUB = 3.66
W_TRACE = 600e-6
JUNCTION_X = 1.0e-3
FEED_X = 4.0e-3
LY = 3.4e-3
Y_C = LY / 2.0
N_GND, N_SUB_LO, N_SUB_HI, N_TRACE = 25, 26, 28, 29
JUNCTION_Z = N_GND * DX
LZ = JUNCTION_Z + H_SUB + DX + 1.0e-3
LX = FEED_X + 1.0e-3
FREQ_MAX = 6.0e9
CLEAR_R = PIN_R + 2 * DX
N_STEPS = 8000


def _half_cell_box_z(n_lo, n_hi):
    """z=(lo, hi) that rasterizes to EXACTLY grid nodes [n_lo, n_hi].

    Recipe from rfx/geometry/csg.py's ``Box`` docstring ("Rasterization
    convention"): interior corners belong on cell MIDPOINTS, ``(j+0.5)*dx``
    -- a corner placed exactly ON a node plane (the float32 double-
    rounding "knife edge" that docstring documents at length) has
    UNPREDICTABLE occupancy. Every boundary in this fixture is a clean
    multiple of ``dx`` before this correction (the obvious-looking
    choice), which is precisely the case the docstring warns is
    undecidable -- it silently dropped the ground-plane/pin-post
    connection's own z-layer on the first construction pass here (see the
    module-level comment above this fixture).
    """
    return (n_lo - 0.5) * DX, (n_hi + 0.5) * DX


def _margin_cylinder_z(n_lo, n_hi):
    """(center_z, height) covering nodes [n_lo, n_hi], one extra cell of
    margin per side -- mirrors ``stamp_coaxial_line``'s own
    ``height = (z_hi - z_lo) + 2*dz`` convention, applied here because
    ``Cylinder``'s inclusive (``<=``) height check has the identical
    node-plane knife-edge risk as ``Box``'s half-open check.
    """
    z_lo, z_hi = n_lo * DX, n_hi * DX
    return 0.5 * (z_lo + z_hi), (z_hi - z_lo) + 2 * DX


def _build_coax_msl_transition_sim():
    from rfx.api import Simulation
    from rfx.boundaries.spec import BoundarySpec
    from rfx.geometry.csg import Box, Cylinder

    sim = Simulation(
        freq_max=FREQ_MAX, domain=(LX, LY, LZ), dx=DX, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml", z="cpml"),
    )
    sim.add_material("sub", eps_r=EPS_SUB)
    sim.add_material("ptfe", eps_r=EPS_COAX)

    gnd_lo, gnd_hi = _half_cell_box_z(N_GND, N_GND)
    sim.add(Box((0.0, 0.0, gnd_lo), (LX, LY, gnd_hi)), material="pec")
    clr_c, clr_h = _margin_cylinder_z(N_GND, N_SUB_LO)
    sim.add(
        Cylinder(center=(JUNCTION_X, Y_C, clr_c), radius=CLEAR_R, height=clr_h, axis="z"),
        material="ptfe",
    )
    sub_lo, sub_hi = _half_cell_box_z(N_SUB_LO, N_SUB_HI)
    sim.add(Box((0.0, 0.0, sub_lo), (LX, LY, sub_hi)), material="sub")
    trc_lo, trc_hi = _half_cell_box_z(N_TRACE, N_TRACE)
    sim.add(
        Box((JUNCTION_X, Y_C - W_TRACE / 2, trc_lo), (LX, Y_C + W_TRACE / 2, trc_hi)),
        material="pec",
    )
    pin_c, pin_h = _margin_cylinder_z(N_GND, N_TRACE)
    sim.add(
        Cylinder(center=(JUNCTION_X, Y_C, pin_c), radius=PIN_R, height=pin_h, axis="z"),
        material="pec",
    )

    sim.add_coaxial_port(
        position=(JUNCTION_X, Y_C, N_GND * DX), face="bottom",
        pin_radius=PIN_R, outer_radius=OUTER_R, impedance=50.0,
    )
    sim.add_msl_port(
        position=(FEED_X, Y_C, N_SUB_LO * DX), width=W_TRACE, height=H_SUB,
        direction="-x", impedance=50.0, eps_r_sub=EPS_SUB,
    )
    return sim


def test_pin_axis_pec_connectivity_is_continuous():
    """Non-FDTD (~1s) connectivity check -- the claim behind N5/the module
    comment's "missing PEC z-layer" fix, committed as an artifact instead of
    living only in throwaway debugging output (issue #581 review N1).

    Reproduces exactly what compute_coax_msl_transition builds on the pin
    axis (registered geometry via _assemble_materials, THEN the coax stub's
    own stamp_coaxial_line, mirroring the method's own ordering) and asserts
    there is no gap: every z-index from the coax stub's near-source end
    through the trace's own layer is either registered PEC (pec_mask) or
    coax-stamped PEC-conductivity (sigma >= PEC_SIGMA/2), with no break.
    """
    from rfx.sources.coaxial_port import stamp_coaxial_line, PEC_SIGMA

    sim = _build_coax_msl_transition_sim()
    grid = sim._build_grid()
    materials, _, _, pec_mask, _, _, _ = sim._assemble_materials(grid)
    pec = np.asarray(pec_mask)

    port = sim._coaxial_ports[0]
    center_xy = (float(port.position[0]), float(port.position[1]))
    z_junction_idx = int(grid.position_to_index(port.position)[2])
    z_stub_lo = int(grid.pad_z_lo) + 2
    z_stub_hi = z_junction_idx - 1

    materials, _ = stamp_coaxial_line(
        grid, materials, center_xy=center_xy, z_lo_index=z_stub_lo,
        z_hi_index=z_stub_hi, pin_radius=PIN_R, outer_radius=OUTER_R,
    )
    sigma = np.asarray(materials.sigma)

    i0, j0 = grid.position_to_index((JUNCTION_X, Y_C, 0.0))[:2]
    trace_idx = int(grid.position_to_index((JUNCTION_X, Y_C, N_TRACE * DX))[2])

    registered_pec = pec[i0, j0, :]
    stamped_pec = sigma[i0, j0, :] >= 0.5 * PEC_SIGMA
    combined = registered_pec | stamped_pec

    span = combined[z_stub_lo:trace_idx + 1]
    assert np.all(span), (
        f"gap in pin-axis PEC/sigma connectivity between the coax stub "
        f"(z_stub_lo={z_stub_lo}) and the trace (idx={trace_idx}): missing "
        f"indices {z_stub_lo + np.where(~span)[0]}"
    )


# Pinned from gate_from_envelope(0.9928, quantum=100) = 1.49, computed ONCE
# from the max|S| measured when this test was last written/reviewed (issue
# #581 review B5): a gate computed LIVE from the value it bounds
# (``gate_from_envelope(max_abs_s)``) can never fail by construction, so it
# is frozen here as a literal instead, and capped at the PREDECLARED 1.10
# passivity floor (1.49 alone is ~40% above that floor and would provide no
# real protection).
PINNED_MEASURED_GATE = 1.49


def test_pinned_measured_gate_provenance():
    """PINNED_MEASURED_GATE's own derivation, kept checkable (issue #581
    review B5): frozen as a literal so the amplitude gate below cannot be
    tautological, but the literal's provenance (gate_from_envelope applied
    to the max|S| measured when it was pinned) stays self-verifying rather
    than a comment-only claim.
    """
    assert gate_from_envelope(0.9928, quantum=100) == PINNED_MEASURED_GATE


@pytest.mark.slow_physics
def test_coax_msl_transition_first_fixture_diagnostic():
    """The one committed #489-leg-4 fixture -- DIAGNOSTIC honesty level.

    See ``PREDECLARATION`` above and the module-level comment immediately
    preceding this test for the full R2/R3 accounting, INCLUDING THE
    POST-REVIEW ATTRIBUTION CORRECTION (issue #581 review B2/B3): the
    degeneracy this test locks in is NOT attributed to "near-degenerate
    two-drive amplification from junction reflection" (the original,
    refuted claim) -- it is better explained by the MSL probe ladder being
    far too short (0.34%-3.37% of a guided wavelength) for the matrix-pencil
    wave split to resolve, an instrument-scoping limit of THIS fixture's
    probe geometry, not a claim about the junction's own physics. This test
    asserts exactly what was (re-)measured: self-consistency (finite,
    settled) passes; the reciprocity/passivity/degeneracy findings are
    LOCKED as a reproducibility witness, not papered over; the corrected
    discriminants (column-equilibrated cond_a, fitted gamma vs analytic
    beta) are asserted too, not just claimed in prose. Runtime ~110s (two
    FDTD drives on a (67, 51, 56)-cell grid, 8000 steps each).
    """
    assert PREDECLARATION["status"] == "RUN"

    sim = _build_coax_msl_transition_sim()
    result = sim.compute_coax_msl_transition(
        junction_x=JUNCTION_X, eps_r_sub=EPS_SUB, n_steps=N_STEPS,
        n_freqs=3, probe_count=6, probe_start_cells=4, probe_spacing_cells=2,
        skip_preflight=True, strict_passivity=True,
    )

    assert result.s_params.shape == (2, 2, 3)
    assert np.all(np.isfinite(result.s_params))
    assert result.port_names == ("coax", "msl")
    assert result.a_inc.shape == (2, 2, 3) and result.b_out.shape == (2, 2, 3)

    # Settling: both drives clear the -40 dB ring-down rule (this fixture
    # measured -43.9 / -63.6 dB at N_STEPS=8000; a shorter, 1500-step
    # smoke run measured only -1.0 / -0.2 dB -- the record was genuinely
    # under-settled there, not a construction defect).
    assert np.all(np.asarray(result.settling_db) < -40.0), (
        f"settling_db {result.settling_db} did not clear -40 dB; the "
        "8000-step record may need lengthening."
    )

    # Amplitude falsifier: pinned literal (see PINNED_MEASURED_GATE above),
    # capped at the predeclared 1.10 passivity floor -- this fixture's own
    # diagonal peaks near 1 (strong coax-side reflection) but must not
    # cross the passivity-violation floor by more than extraction noise.
    # strict_passivity=True above additionally makes the hard
    # passivity/finiteness self-check RAISE (not warn) if this fixture's
    # own extraction ever crosses that check's tol=0.10 bound.
    max_abs_s = float(np.max(np.abs(result.s_params)))
    gate = min(PINNED_MEASURED_GATE, 1.10)
    assert max_abs_s <= gate, (
        f"max|S| {max_abs_s:.4f} exceeds the pinned gate {gate:.4f} -- a "
        "NEW amplitude excursion beyond what was measured when this test "
        "was written; re-run the falsifier diagnostic (recurrence_"
        "residual / fit_residual / cond_a_equilibrated) before trusting "
        "the change."
    )

    # Degeneracy witness -- CORRECTED (issue #581 review B2): raw cond_a
    # stays large (scale-mismatch artifact, see module comment), but the
    # column-EQUILIBRATED cond_a -- the one that actually discriminates
    # geometric near-parallelism on a mixed-family lane -- stays near 1,
    # confirming the two drives are NOT geometrically near-degenerate.
    assert np.all(np.asarray(result.cond_a) > 1.0e3), (
        f"cond_a {result.cond_a} dropped below the scale-mismatch floor "
        "this fixture measured -- re-evaluate the attribution."
    )
    assert np.all(np.asarray(result.cond_a_equilibrated) < 1.5), (
        f"cond_a_equilibrated {result.cond_a_equilibrated} rose above the "
        "near-unity floor this fixture measured -- the two drives may have "
        "become genuinely geometrically near-degenerate, which would "
        "revive (not refute) the original junction-reflection attribution; "
        "re-derive the finding before trusting either story."
    )

    # B3 discriminant, asserted (not just claimed in the PR body): the
    # fitted Im(gamma) on the MSL array does not track the analytic
    # Hammerstad-Jensen beta -- confirms the extraction-class (ladder too
    # short) explanation over the junction-physics one, per the module
    # comment. Checked both drives' own fits (they disagree with EACH
    # OTHER by 1-2 orders of magnitude too -- a second, independent
    # symptom of an unreliable split, not just a mismatch vs the analytic
    # value).
    freqs = np.asarray(result.freqs)
    ladder_span_m = 2 * (6 - 1) * DX
    beta_analytic, ladder_frac = _msl_guided_beta_and_ladder_fraction(
        freqs, ladder_span_m=ladder_span_m,
    )
    im_gamma_coax_driven = np.abs(result.gamma[1, 0, :].imag)
    im_gamma_msl_owndrive = np.abs(result.gamma[1, 1, :].imag)
    ratio_coax_driven = im_gamma_coax_driven / beta_analytic
    assert np.all(ratio_coax_driven > 3.0), (
        f"fitted Im(gamma) (coax-driven) / analytic beta = "
        f"{ratio_coax_driven} dropped near 1 -- the MSL matrix-pencil fit "
        "may have started tracking the true guided mode; re-derive the "
        "instrument-scoping attribution before trusting it."
    )
    # The two drives' own fits should still disagree by >1 order of
    # magnitude at at least one bin (measured: bin 0 gives 673.0 vs 4.5 --
    # ratio ~150x) -- an internal cross-check that neither fit is simply
    # "the right answer measured twice".
    cross_drive_ratio = np.maximum(
        im_gamma_coax_driven / np.maximum(im_gamma_msl_owndrive, 1e-30),
        im_gamma_msl_owndrive / np.maximum(im_gamma_coax_driven, 1e-30),
    )
    assert np.any(cross_drive_ratio > 10.0), (
        f"coax-driven vs msl-own-drive Im(gamma) ratios {cross_drive_ratio} "
        "no longer disagree by >10x anywhere -- the two independent fits "
        "may have converged, which would weaken the instrument-scoping "
        "finding; re-derive it."
    )

    # Reciprocity falsifier -- LOCKED, not silently accepted: this
    # fixture's own measured deviation is ~94-100% (see module comment for
    # the corrected attribution). Regression-lock the FINDING (it stays
    # badly non-reciprocal) so a future change that quietly "fixes" this
    # number gets flagged for review rather than assumed correct -- the
    # fix would need its own falsifier re-run (e.g. a longer MSL probe
    # ladder, >= 0.25 lambda_g -- a NEW, separately pre-declared R2
    # attempt, NOT run in this PR), not a silent gate change here.
    pair, worst_dev = _mixed_reciprocity_deviation(result.s_params)
    assert worst_dev > 0.5, (
        f"reciprocity deviation dropped to {worst_dev:.3f} between ports "
        f"{pair} -- the pre-declared falsifier no longer fires. Either "
        "the fixture's physics changed (re-derive the finding) or the "
        "assembler changed (re-run Part 1's planted-voltage tests, which "
        "independently pin the correct answer)."
    )


# ---------------------------------------------------------------------------
# Part 4 -- ATTEMPT 2: resolve attempt 1's own named instrument-scoping
# defect (PI-directed, R2's written escape clause). See PREDECLARATION_
# ATTEMPT2 below for the full form. Attempt 1's own PREDECLARATION, fixture,
# and regression locks above are UNCHANGED -- this section is purely
# additive.
# ---------------------------------------------------------------------------

PREDECLARATION_ATTEMPT2 = {
    "leg": "issue #489 leg 4, attempt 2 -- resolve attempt 1's MSL-ladder instrument-scoping limit",
    "authorization": (
        "R2's written escape clause: 'a second [attempt] needs a named new "
        "falsifier or an identified implementation defect in attempt 1, in "
        "writing'. Attempt 1 (PR #581, post-review) named exactly such a "
        "defect: the MSL probe ladder spanned only 0.34%-3.37% of the "
        "guided wavelength at the measured frequencies, too short for the "
        "matrix-pencil forward/backward split to resolve -- an instrument-"
        "scoping limit of the probe geometry, not a claim about attempt "
        "1's junction physics. PI-directed. This is the ONE attempt that "
        "defect authorizes; attempt 1's own two construction-debugging "
        "fixes (missing PEC layer, rasterization knife-edge) do not count "
        "against this budget (plumbing, not a mechanism-hypothesis loop, "
        "per the repo's own R2 distinction)."
    ),
    "instrument_changes": {
        "msl_probe_ladder": (
            "Lengthened from 1.000 mm (attempt 1: probe_spacing_cells=2, "
            "probe_count=6) to 8.000 mm (attempt 2: msl_probe_spacing_cells"
            "=10, msl_probe_count=9, msl_probe_start_cells=4 unchanged) -- "
            "the two families' probe-ladder parameters were coupled "
            "through ONE shared set of Simulation.compute_coax_msl_"
            "transition(...) kwargs through attempt 1, which silently "
            "forced the coax side's (necessarily short) ladder length onto "
            "the MSL side too. Fixed by adding independent "
            "msl_probe_count/msl_probe_start_cells/msl_probe_spacing_cells "
            "parameters (default None -> falls back to the coax-side "
            "value, so attempt 1's own committed call is byte-identical in "
            "behavior). This is new PUBLIC API surface -- "
            "docs/guides/api_symbol_inventory.json is regenerated in this "
            "PR (attempt 1's own CI-red lesson, review finding B1's "
            "sibling)."
        ),
        "frequency_band": (
            "Shifted from {0.6, 3.3, 6.0} GHz to {6.0, 8.0, 10.0} GHz. "
            "Forced by the ladder-length requirement itself: "
            "0.25*lambda_g at the OLD lowest bin (0.6 GHz) is 74.3 mm -- "
            "infeasible for a compact fixture. At 3.3/6.0/10.0 GHz it is "
            "13.5/7.4/4.5 mm respectively (eps_eff=2.8327, Hammerstad-"
            "Jensen, unchanged trace/substrate). Chose {6.0, 8.0, 10.0} "
            "GHz (not {3.3, 6.0, 10.0}) to keep the domain/runtime "
            "smaller (0.25*lambda_g at the new lowest bin, 6.0 GHz, is "
            "7.42 mm vs 13.5 mm at 3.3 GHz -- roughly half the ladder "
            "length and, since compute cost scales with cells*steps, "
            "considerably less than half the runtime) while still "
            "sharing ONE frequency (6.0 GHz) with attempt 1's own band "
            "for a direct cross-attempt comparison point. The default "
            "excitation (GaussianPulse, bandwidth=0.8, f0=freq_max/2) is "
            "re-centered by setting freq_max=16 GHz -> f0=8 GHz, "
            "spanning roughly 4.8-11.2 GHz and covering the new band "
            "with margin; this is an INSTRUMENT (excitation spectrum) "
            "change, not a junction change."
        ),
        "x_cpml_clearance": (
            "MSL port distance to its own nearest x-CPML face raised from "
            "attempt 1's ~200um (preflight-flagged, 'source-side CPML "
            "reflection may inflate |S11|' -- attempt 1's named third "
            "candidate mechanism, review finding B4) to 1500um (feed_x=11."
            "0mm, domain LX=12.5mm), well above the preflight-recommended "
            "600um (=2*h_sub). Verified BEFORE running any FDTD (preflight "
            "is pure NumPy, no FDTD): the x-CPML clearance line no longer "
            "appears in attempt 2's preflight output at all (was line 8/9 "
            "of attempt 1's nine; attempt 2 emits seven lines, all "
            "junction-resolution advisories that are EXPECTED to persist "
            "since the junction is unchanged -- see the byte-identical "
            "assertion below)."
        ),
        "coax_side": (
            "Unchanged: probe_count=6, probe_start_cells=4, "
            "probe_spacing_cells=2 (attempt 1's own values, now passed "
            "explicitly since these are the coax-only kwargs after the "
            "API split above). The coax stub's own z-extent is short by "
            "construction (between the near-source feed and the junction) "
            "and was never the limiting factor -- attempt 1's own "
            "recurrence_residual/fit_residual on the COAX array were "
            "small throughout (<0.008), unlike the MSL array."
        ),
    },
    "junction_byte_identical_fields": [
        "DX", "PIN_R", "OUTER_R", "EPS_COAX", "H_SUB", "EPS_SUB", "W_TRACE",
        "JUNCTION_X", "N_GND", "N_SUB_LO", "N_SUB_HI", "N_TRACE",
        "JUNCTION_Z", "CLEAR_R",
    ],
    "expected_if_extraction_class_confirmed": {
        "gamma_vs_beta": (
            "Fitted Im(gamma) on the MSL array (coax-driven fit, the more "
            "reliable of the two per attempt 1) tracks analytic "
            "Hammerstad-Jensen beta within a ratio band of [0.8, 1.3]. "
            "Justification for the band width: quasi-TEM microstrip "
            "dispersion (frequency-dependent eps_eff) is not captured by "
            "the static Hammerstad-Jensen formula used here, and typical "
            "quasi-static-vs-dispersive deviations for this W/H, eps_r "
            "class are documented elsewhere in this repo (compute_msl_s_"
            "matrix's own known limits) at up to ~10-20% -- [0.8, 1.3] "
            "gives headroom above that without being so wide it would "
            "also accept attempt 1's own 4-32x deviation."
        ),
        "reciprocity": (
            "Deviation collapses from attempt 1's 94-100% toward or below "
            "an acceptance of <=30% -- the #488 mixed lane's OWN measured "
            "envelope on its comparable (different-family, similar "
            "maturity) fixture was 9-30% (wave channel up to 55%); 30% "
            "sets this fixture's acceptance at the LOOSE end of that "
            "precedent, appropriate for a first resolved measurement on a "
            "brand-new lane rather than demanding #488's own best case."
        ),
        "s22": (
            "|S22| (MSL reflecting while MSL drives) becomes resolvable "
            "-- i.e. clearly above the float32 relative-precision noise "
            "floor (attempt 1 measured 1e-8 to 1e-11 at two of three "
            "bins, itself evidence of a degenerate split rather than a "
            "measured near-zero reflection). No specific target value is "
            "predeclared (the junction's true reflection is exactly the "
            "open physics question this whole leg exists to measure) -- "
            "only that it stop reading as numerical noise."
        ),
        "a_inc_amplitude_gap": (
            "No per-drive amplitude NORMALIZATION was added in this "
            "attempt (that would touch the assembler, not the "
            "extraction instrument, and attempt 1's own planted-voltage "
            "tests already prove the assembler's power-wave "
            "normalization does not need one). The gap should NARROW "
            "somewhat on physical grounds (a longer, better-resolved MSL "
            "ladder should recover a less-attenuated, more physically "
            "representative wave amplitude at the reference plane than a "
            "sub-wavelength-scale, poorly-resolved short ladder did), "
            "but is not required to close for the extraction-class "
            "verdict -- cond_a_equilibrated (not raw cond_a, per attempt "
            "1's own B2 finding) is the right degeneracy read either way, "
            "and is expected to remain near 1 (both attempts' drives are "
            "geometrically distinct regardless of amplitude)."
        ),
    },
    "falsifier": (
        "With a proper ladder (>=0.25 lambda_g at the lowest measured "
        "frequency) AND proper x-CPML clearance (>=600um), if reciprocity "
        "deviation STAYS above the declared 30% acceptance, OR fitted "
        "gamma still fails to track beta within the declared [0.8, 1.3] "
        "band -- the extraction-class explanation is REFUTED for the "
        "REMAINING excess, and the candidates return to (i) junction "
        "physics (a genuinely bad, high-reflection launch at this "
        "specific unmatched pin-to-trace geometry -- a legitimate, "
        "reportable physics finding) or (ii) the power-wave normalization "
        "failing IN SITU despite passing on planted voltages (attempt 1 "
        "verified it only against synthetic, noise-free voltages, not a "
        "real FDTD record). STOP on falsifier per R2: one attempt, "
        "report, no further iteration in this PR."
    ),
    "runtime_feasibility": (
        "Grid grows from (67, 51, 56) = 191,352 cells (attempt 1) to "
        "(142, 51, 56) = 405,552 cells (attempt 2, LX 5.0mm -> 12.5mm, "
        "dx unchanged) -- a 2.12x cell-count ratio (y, z unchanged). "
        "N_STEPS scaled from attempt 1's 8000 by roughly the LX ratio "
        "(12.5/5.0 = 2.5x) to 20000, on the reasoning that the dominant "
        "settling-time driver is round-trip transit time across the "
        "domain's largest dimension, which grew by that factor while dt "
        "is unchanged (dx unchanged). Estimated compute: attempt 1 spent "
        "110s (two drives, 8000 steps, 191,352 cells) => ~2.78e7 "
        "cell-steps/sec measured throughput. Attempt 2: 405,552 cells * "
        "20000 steps * 2 drives = 1.622e10 cell-steps => ~584s (~9.7 "
        "min) estimated, run locally (no VESSL needed for this scale). "
        "A 2000-step smoke run (construction-debugging only, not a "
        "physics attempt) measured 55.5s for two drives at 2000 steps "
        "each (0.01388s/step/drive), independently confirming the "
        "throughput estimate (0.01388 * 20000 * 2 = 555s, ~9.25 min) "
        "before the declared run below. CALIBRATION NOTE (same class as "
        "attempt 1's own 1500->8000 step calibration, not a new R2 "
        "attempt -- settling_db governs, not a guess): the first N_STEPS_2"
        "=20000 run (516.8s measured, matching the ~584s estimate) "
        "measured only -12.3/-10.7 dB settling, short of the -40 dB rule, "
        "so N_STEPS_2 was raised to 45000 (~1163s / ~19.4min estimated "
        "at the same measured throughput) for the run actually reported "
        "below."
    ),
    "status": "RUN",
    # RETRACTED (issue #585 adversarial review, finding B1) -- preserved
    # here as the historical record of what was first written, exactly
    # like PREDECLARATION["post_review_correction"] above preserves
    # attempt 1's own retracted attribution. Do not repeat this pattern:
    # this is the THIRD retracted attribution on this lane (near-
    # degenerate-drives in attempt 1; junction-reflection/amplitude-gap
    # here).
    "retracted_amplitude_gap_attribution": (
        "The reciprocity/|S22| miss was originally attributed here to the "
        "coax/MSL own-drive amplitude gap ('~1.8e7-3.3e7x apart... the "
        "signature that DOES match is a drive-amplitude mismatch'), with "
        "a locked `amplitude_ratio > 1.0e6` assertion. This is WRONG in a "
        "way that is mathematically provable, not just under-evidenced: "
        "per-drive (per-column) rescaling of the two-drive solve's "
        "incident-wave matrix A and outgoing-wave matrix B (A'=A@D, "
        "B'=B@D for any invertible diagonal D) leaves S=B@inv(A) EXACTLY "
        "invariant -- (AD)^-1 = D^-1 A^-1, so B'@inv(A') = B@D@D^-1@A^-1 "
        "= B@A^-1 = S identically (reviewer-verified numerically at this "
        "attempt's own measured gap, D=[1, 1.8e7]: |S'-S| ~ 3e-16, "
        "float64 noise floor). A per-drive amplitude difference CANNOT "
        "cause a reciprocity or magnitude defect in S by construction --"
        "the two-drive solve is exactly the machinery that removes that "
        "dependence. Separately, `amplitude_ratio` (a_inc[0,0,:]/"
        "a_inc[1,1,:], the own-drive incident amplitudes) turns out to "
        "equal raw `cond_a` to 8 significant figures on all 3 bins (both "
        "are dominated by the same column-norm ratio for a near-"
        "orthogonal 2x2 A) -- i.e. this was raw cond_a under a new name, "
        "the EXACT quantity this attempt's own docstring says NOT to "
        "read as a degeneracy witness on this lane ('per attempt 1's own "
        "B2 finding'). See `measured` below for the corrected read: "
        "column power (issue #585 review finding B5), which IS "
        "scaling-invariant and is the productive open question."
    ),
    "measured": {
        # Filled once, from BOTH calibration checkpoints (issue #585
        # review finding M2: disclose the refuting numbers too, not only
        # the confirming ones). Kept as a static literal (fill-contract
        # pattern, matches REPRODUCE_GATE_RECORD elsewhere in this repo)
        # rather than a runtime side-effect written by the test --
        # reviewable without running anything, and not order-dependent on
        # which test runs first. test_coax_msl_transition_attempt2_
        # lengthened_ladder independently RE-DERIVES and asserts against
        # these same numbers from a fresh FDTD run; it does not read this
        # dict.
        #
        # Run-length invariance test (issue #585 review finding B2/B4 --
        # the repo's own discriminator for "is this a settled physical
        # quantity or still an evolving transient", applied to the two
        # checkpoints 20000 -> 45000 steps):
        "n_steps_checkpoints": [20000, 45000],
        "gamma_ratio_coax_driven_by_checkpoint": [
            [1.085, 0.826, 0.976],   # 20000 steps
            [1.128, 0.854, 1.071],   # 45000 steps
        ],  # INVARIANT across checkpoints (both fully in [0.8, 1.3]) -- physical, CONFIRMED-provisional.
        "reciprocity_worst_dev_by_checkpoint": [0.824, 0.938],   # NOT invariant -- moved AWAY from the 0.30 acceptance.
        "s22_abs_by_checkpoint": [
            [0.043, 0.141, 0.451],   # 20000 steps
            [0.102, 0.109, 1.104],   # 45000 steps -- top bin grew 2.4x (0.451 -> 1.104)
        ],  # NOT invariant.
        "max_abs_s_by_checkpoint": [0.9933, 1.1038],   # crosses the 1.10 passivity-guard hard limit between checkpoints.
        "column_power_msl_driven_by_checkpoint": [
            [0.0018, 0.0199, 0.204],   # 20000 steps
            [0.0104, 0.0119, 1.218],   # 45000 steps
        ],  # Sum|S_ij|^2 over the MSL-driven column -- scaling-INVARIANT (unlike the retracted amplitude-gap read), NOT invariant across checkpoints either: mostly << 1 (power missing from a nominally-lossless structure) with one bin > 1 at the less-settled checkpoint (impossible for a passive structure -- an extraction artifact). THE open question this leg has not yet answered: where does the MSL column's power go (or come from)?
        "cond_a_equilibrated": [1.00238, 1.00244, 1.00549],   # 45000 steps; near 1 at both checkpoints -- invariant, confirms NOT geometric near-degeneracy.
        "settling_db_by_checkpoint": [[-12.3, -10.7], [-19.69, -17.94]],   # improving DIRECTION only; neither checkpoint clears -40 dB.
        "verdict": (
            "gamma-vs-beta: CONFIRMED, PROVISIONAL pending a settled run "
            "-- the one discriminant that PASSES the run-length "
            "invariance test (stable across both checkpoints, fully "
            "in-band both times). reciprocity, |S22|, max|S|: UNMEASURED "
            "AT THIS SETTLING, not 'refuted with cause identified' -- all "
            "three FAIL the run-length invariance test (still moving "
            "between checkpoints, in the WRONG direction for reciprocity "
            "and |S22|). The productive open question is the column-"
            "power witness (B5): the MSL-driven column loses the large "
            "majority of its power at both checkpoints, on a nominally "
            "lossless structure -- scaling-invariant, unlike the "
            "RETRACTED amplitude-gap attribution above. A fully settled "
            "run (predeclared UNRUN, see SETTLED_RUN_RECORD) is needed "
            "before any claim on reciprocity/|S22| can be made."
        ),
        # RESOLVED, VESSL 369367252283 (2026-08-06): the settled run this
        # 'verdict' string above calls for has now run and cleared -40 dB.
        # The two checkpoint numbers above are UNCHANGED (kept as the
        # historical record of what run-length invariance looked like
        # before settling) -- this key only points forward to where the
        # settled read lives, per the same additive pattern
        # PREDECLARATION["post_review_correction"] uses elsewhere in this
        # file. See SETTLED_RUN_RECORD["measured"]["verdict"] for the full
        # settled-run verdict (gamma-vs-beta CONFIRMED; passivity
        # ATTRIBUTED to truncation; reciprocity now MEASURED at 91.4% worst
        # deviation, not adjudicated between physical/instrument causes;
        # the column-power question SHARPENED with a named discriminating
        # check) -- do not re-derive it here.
        "settled_run_resolution": (
            "See SETTLED_RUN_RECORD (status RUN, VESSL 369367252283) for "
            "the full settled-run verdict -- this checkpoint-level "
            "'verdict' string above is preserved as history, not "
            "superseded in place."
        ),
    },
}


SETTLED_RUN_RECORD = {
    # Fill-contract pattern (mirrors REPRODUCE_GATE_RECORD in
    # tests/test_coax_two_port_referee_header.py): UNRUN <=> no numbers,
    # no log path. This is the settled-run predeclaration issue #585
    # review requires (finding B2/B4 + restructure item (c)): the local
    # runtime available to this session cannot reach the -40 dB rule at
    # attempt 2's domain size (extrapolated ~60 min locally from the two
    # measured checkpoints, 20000 -> -12.3/-10.7 dB and 45000 -> -19.7/
    # -17.9 dB) -- a VESSL run is the named next step, NOT attempted in
    # this PR.
    "leg": "issue #489 leg 4, attempt 2 -- the settled-run follow-up",
    "purpose": (
        "Resolve reciprocity/|S22|/max|S| from UNMEASURED-at-this-"
        "settling to an actual pass/fail read, by running long enough to "
        "clear the -40 dB ring-down rule on attempt 2's junction-"
        "identical, longer-ladder fixture (same geometry as "
        "_build_coax_msl_transition_sim_attempt2 in this file)."
    ),
    "target_n_steps": 135000,
    "target_n_steps_derivation": (
        "Extrapolated from the two measured local checkpoints (20000 "
        "steps -> -12.3/-10.7 dB, 45000 steps -> -19.69/-17.94 dB, see "
        "settling_db_by_checkpoint in PREDECLARATION_ATTEMPT2): 7.39 dB "
        "gained on port 0 and 7.24 dB gained on port 1 per +25000 steps "
        "in this range (an earlier revision of this text rounded these "
        "same two numbers to a looser '7.4-7.9 dB' band -- corrected "
        "here to the precise per-port deltas). Reaching -40 dB from the "
        "WORSE (less-negative) of the two checkpoints, -17.94 dB, needs "
        "another ~22.1 dB, i.e. roughly 3x the 25000-step increment "
        "already spent (25000 -> 45000): the formula 45000 + 3*25000 "
        "gives exactly 120000 steps, not a '120000-135000' range (an "
        "earlier revision of this text conflated the formula's output "
        "with the chosen target and stated it as a range). "
        "target_n_steps is set to 135000, not 120000 -- the extra "
        "15000 steps above the formula's 120000 are deliberate margin, "
        "not part of the linear extrapolation itself. This is a linear "
        "extrapolation of a process that may not actually be linear "
        "(ring-down decay is often closer to exponential in TIME, i.e. "
        "linear in STEPS on a dB scale, which is the assumption here) "
        "-- treat the target as a starting estimate to calibrate "
        "against, not a guarantee, exactly as attempt 2's own local "
        "calibration (20000 -> 45000) needed one correction already."
    ),
    "vessl_pattern": (
        "UPDATE (PR #588): a mounted-primary-checkout pattern was used "
        "instead of the branch-clone pattern this field originally "
        "predeclared (mirroring #561/#559) -- by the time this job can "
        "run, scripts/diagnostics/coax_msl_transition_settled_run.py and "
        "compute_coax_msl_transition are both already on main (PR #588 "
        "merged), so the mounted checkout already has everything needed "
        "and the branch-clone machinery (SHA pinning, fallback logic) "
        "would be pure overhead. Driver: "
        "scripts/diagnostics/coax_msl_transition_settled_run.py (imports "
        "the test module, reuses _build_coax_msl_transition_sim_attempt2, "
        "runs at --n-steps target_n_steps, reports the same witnesses "
        "this file's own attempt-2 test computes). YAML: "
        "scripts/vessl_coax_msl_transition_settled_run.yaml (mounted "
        "checkout, imitates scripts/vessl_gpu_suite.yaml / "
        "scripts/vessl_msl_f64_referee.yaml). Landed in PR #588; the run "
        "itself was deliberately NOT submitted by that PR -- submit only "
        "after it merges to main. Tracked by issue #589 (#489 is closed "
        "and does not track this lane's continuing work)."
    ),
    "tracking_issue": 589,
    "status": "RUN",
    "measured": {
        # Verbatim from the VESSL run's own result JSON (tracked copy below) --
        # do not hand-round these before comparing against a future re-run.
        "n_steps": 135000,
        "settling_db": [-45.9397630721533, -44.17124185980615],
        "settling_cleared_40db": True,  # CLEARED for the first time on this lane.
        "gamma_ratio_coax_driven": [1.1484731207266794, 0.858657230346571, 1.0510942174501197],
        "reciprocity_worst_deviation": {"pair": [0, 1], "value": 0.913919627745057},
        "s22_abs": [0.08078023177463328, 0.10478316571484025, 0.893672853684955],
        "max_abs_s": 0.9932631013833461,
        "col_msl_driven_power": [0.0065254707693085355, 0.010979588441163948, 0.7986516509686326],
        "passivity_guard_raised": False,
        "verdict": (
            "Both drives clear the -40 dB ring-down rule for the first time "
            "on this lane (settling_db -45.94 / -44.17 dB) -- reciprocity, "
            "|S22|, and max|S| are now genuinely MEASURED quantities on this "
            "fixture, not still-evolving transients, resolving the earlier "
            "'UNMEASURED AT THIS SETTLING' label.\n"
            "\n"
            "gamma-vs-beta: CONFIRMED, no longer provisional. Ratio "
            "1.148/0.859/1.051 at 6/8/10 GHz is a THIRD checkpoint (after "
            "20000 and 45000 steps) landing inside the predeclared [0.8, "
            "1.3] band -- three checkpoints in a row in-band, including the "
            "fully settled one, is the strongest confirmation this lane's "
            "own run-length-invariance discipline can produce.\n"
            "\n"
            "Passivity: ATTRIBUTED. max|S| = 0.9933 at full settling and "
            "the shared passivity guard (strict semantics) raises nothing "
            "on this result. The earlier passivity-guard-tripping readings "
            "(max|S| 1.1038 and MSL-driven column power 1.218, both >1 and "
            "both measured at the 45000-step checkpoint -- impossible for "
            "a passive structure) are now attributable to TRUNCATION: they "
            "were measured before ring-down had actually settled, not "
            "evidence of a real passivity violation or an extractor "
            "defect.\n"
            "\n"
            "Reciprocity: now MEASURED, not merely disclosed -- worst "
            "deviation 91.4% (pair (0, 1)) AT FULL SETTLING. This is NOT a "
            "still-evolving transient (the -40 dB rule is cleared) and is "
            "NOT explained by any of the THREE previously RETRACTED "
            "attributions on this lane (near-degenerate two-drive "
            "amplification; the drive-amplitude-gap 'explanation', proven "
            "mathematically impossible; and the MSL-ladder-too-short "
            "instrument-scoping limit, which attempt 2's own gamma-vs-beta "
            "pass shows is resolved, not the cause of THIS number). What "
            "91.4% IS: a real, settled property of THIS fixture's own "
            "measurement -- a passive reciprocal structure cannot "
            "physically have |S12| != |S21| by an order of magnitude, so "
            "this deviation measures either (a) genuine un-modeled "
            "loss/coupling asymmetry in how the extraction sees the two "
            "drives, or (b) an instrument limitation in the extraction "
            "itself that survives settling. Per this lane's own "
            "three-strikes retraction history, this is NOT adjudicated "
            "here -- see the sharpened open question below, which names a "
            "discriminating check instead of a fourth attribution.\n"
            "\n"
            "THE SHARPENED OPEN QUESTION (MSL-driven column power, at full "
            "settling): 0.00653 / 0.01098 / 0.79865 at 6/8/10 GHz -- "
            "~99.3% / ~98.9% / ~20.1% of the incident power is unaccounted "
            "for (neither transmitted nor reflected) at the two lower "
            "frequencies, dropping sharply by 10 GHz. Derived from this "
            "same result: |S12|^2 (col_msl_driven_power - |S22|^2) is "
            "~2.5e-8 / ~7.7e-8 / ~4.8e-7 at the three bins -- essentially "
            "nothing transmits to the coax side -- while |S22|^2 alone "
            "(0.0065 / 0.0110 / 0.7987) accounts for nearly all the "
            "retained power. Two named candidates, NOT adjudicated: "
            "(a) PHYSICAL -- the unmatched vertical coax-to-trace launch "
            "(no intermediate matching structure, per this leg's own "
            "scoping) radiates the MSL drive's power into the CPML "
            "absorber at low frequency; consistent with the frequency "
            "trend (coupling back to the ports improves sharply toward "
            "10 GHz). (b) INSTRUMENT -- the MSL-side b-wave (outgoing "
            "wave) extraction misses non-quasi-TEM content generated near "
            "the junction discontinuity, which a quasi-TEM matrix-pencil "
            "fit on the MSL probe ladder cannot resolve. A DISCRIMINATING "
            "check (named, not run): a closed-box variant of this fixture "
            "(PEC walls in place of CPML, no absorber) -- if the missing "
            "power reappears in the port accounting once there is nowhere "
            "to radiate, that supports (a); if it is still missing, that "
            "supports (b). Per this lane's three-retraction history, do "
            "not attribute without running this or an equivalent "
            "falsifier."
        ),
    },
    "log_path": (
        "scripts/diagnostics/_coax_msl_transition_settled_run_logs/"
        "settled_run_369367252283_run.log"
    ),
    "result_json_path": (
        "scripts/diagnostics/_coax_msl_transition_settled_run_logs/"
        "settled_run_369367252283_result.json"
    ),
    "vessl_run_id": "369367252283",
    "verified_on": "2026-08-06",
}


def test_settled_run_record_is_committed_unrun_and_self_consistent():
    """Fail-loud-honest invariant (mirrors test_coax_two_port_referee_
    header.py's own REPRODUCE_GATE_RECORD test): UNRUN <=> no numbers, no
    log path, no run id.
    """
    r = SETTLED_RUN_RECORD
    required = {
        "leg", "purpose", "target_n_steps", "target_n_steps_derivation",
        "vessl_pattern", "status", "measured", "log_path", "vessl_run_id",
    }
    missing = required - set(r.keys())
    assert not missing, f"SETTLED_RUN_RECORD missing fields: {missing}"
    if r["status"] == "UNRUN":
        assert r["measured"] is None
        assert r["log_path"] is None
        assert r["vessl_run_id"] is None
    else:
        assert r["measured"] is not None
        assert r["log_path"] is not None


def test_attempt2_predeclaration_has_required_fields():
    required = {
        "leg", "authorization", "instrument_changes",
        "junction_byte_identical_fields",
        "expected_if_extraction_class_confirmed", "falsifier",
        "runtime_feasibility", "status", "measured",
        "retracted_amplitude_gap_attribution",
    }
    missing = required - set(PREDECLARATION_ATTEMPT2.keys())
    assert not missing, f"PREDECLARATION_ATTEMPT2 missing fields: {missing}"
    assert set(PREDECLARATION_ATTEMPT2["instrument_changes"].keys()) == {
        "msl_probe_ladder", "frequency_band", "x_cpml_clearance", "coax_side",
    }
    # The retraction must actually retract -- not merely exist.
    assert "cond_a" in PREDECLARATION_ATTEMPT2["retracted_amplitude_gap_attribution"]
    assert "invariant" in PREDECLARATION_ATTEMPT2["retracted_amplitude_gap_attribution"]


# ---- attempt 2 geometry: INSTRUMENT constants only. Every junction
# constant below is the attempt-1 module-level name itself (DX, PIN_R,
# OUTER_R, EPS_COAX, H_SUB, EPS_SUB, W_TRACE, JUNCTION_X, N_GND, N_SUB_LO,
# N_SUB_HI, N_TRACE, JUNCTION_Z, CLEAR_R, Y_C, LY) -- reused directly, not
# redefined, so byte-identity is guaranteed by construction rather than by
# a separately-typed copy that could silently drift. Only the probe
# ladder, domain x-extent, and frequency band are new.
FEED_X_2 = 11.0e-3
LX_2 = 12.5e-3
LZ_2 = JUNCTION_Z + H_SUB + DX + 1.0e-3  # same formula as attempt 1 -- z-stack unchanged
FREQS_2 = np.array([6.0e9, 8.0e9, 10.0e9])
FREQ_MAX_2 = 16.0e9  # re-centers the default excitation (f0=freq_max/2=8GHz) on the new band
PROBE_START_2 = 4
PROBE_SPACING_2 = 10
PROBE_COUNT_2 = 9
# N_STEPS_2 was calibrated, same as attempt 1's own 1500->8000 step
# calibration (construction/instrument calibration, not a new R2 physics
# attempt -- the settling_db witness is what governs, not a guess): the
# predeclared estimate of 20000 (2.5x attempt 1's 8000, scaled by the LX
# ratio) measured only -12.3/-10.7 dB settling, well short of the -40 dB
# rule, even though the gamma-vs-beta discriminant it produced already
# landed inside the predeclared [0.8, 1.3] band at all 3 bins (ratios
# 1.085/0.826/0.976) -- encouraging, but not reportable without adequate
# settling per R5. Raised to 45000 (~2.25x more).
N_STEPS_2 = 45000


def _build_coax_msl_transition_sim_attempt2():
    """Attempt 2: same junction (reuses attempt 1's own module constants
    unchanged), longer MSL ladder, wider x-CPML clearance, higher band.
    """
    from rfx.api import Simulation
    from rfx.boundaries.spec import BoundarySpec
    from rfx.geometry.csg import Box, Cylinder

    sim = Simulation(
        freq_max=FREQ_MAX_2, domain=(LX_2, LY, LZ_2), dx=DX, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml", z="cpml"),
    )
    sim.add_material("sub", eps_r=EPS_SUB)
    sim.add_material("ptfe", eps_r=EPS_COAX)

    gnd_lo, gnd_hi = _half_cell_box_z(N_GND, N_GND)
    sim.add(Box((0.0, 0.0, gnd_lo), (LX_2, LY, gnd_hi)), material="pec")
    clr_c, clr_h = _margin_cylinder_z(N_GND, N_SUB_LO)
    sim.add(
        Cylinder(center=(JUNCTION_X, Y_C, clr_c), radius=CLEAR_R, height=clr_h, axis="z"),
        material="ptfe",
    )
    sub_lo, sub_hi = _half_cell_box_z(N_SUB_LO, N_SUB_HI)
    sim.add(Box((0.0, 0.0, sub_lo), (LX_2, LY, sub_hi)), material="sub")
    trc_lo, trc_hi = _half_cell_box_z(N_TRACE, N_TRACE)
    sim.add(
        Box((JUNCTION_X, Y_C - W_TRACE / 2, trc_lo), (LX_2, Y_C + W_TRACE / 2, trc_hi)),
        material="pec",
    )
    pin_c, pin_h = _margin_cylinder_z(N_GND, N_TRACE)
    sim.add(
        Cylinder(center=(JUNCTION_X, Y_C, pin_c), radius=PIN_R, height=pin_h, axis="z"),
        material="pec",
    )

    sim.add_coaxial_port(
        position=(JUNCTION_X, Y_C, N_GND * DX), face="bottom",
        pin_radius=PIN_R, outer_radius=OUTER_R, impedance=50.0,
    )
    sim.add_msl_port(
        position=(FEED_X_2, Y_C, N_SUB_LO * DX), width=W_TRACE, height=H_SUB,
        direction="-x", impedance=50.0, eps_r_sub=EPS_SUB,
    )
    return sim


def test_attempt2_junction_geometry_is_byte_identical_to_attempt1():
    """Asserts (not just claims) the "instrument changes only" predeclaration:
    compares attempt 2's registered coax/MSL junction parameters against
    attempt 1's own module constants, field by field.
    """
    sim1 = _build_coax_msl_transition_sim()
    sim2 = _build_coax_msl_transition_sim_attempt2()

    port1, port2 = sim1._coaxial_ports[0], sim2._coaxial_ports[0]
    assert port1.pin_radius == port2.pin_radius == PIN_R
    assert port1.outer_radius == port2.outer_radius == OUTER_R
    assert port1.position[0] == port2.position[0] == JUNCTION_X
    assert port1.position[1] == port2.position[1] == Y_C
    assert port1.position[2] == port2.position[2] == N_GND * DX
    assert port1.face == port2.face == "bottom"

    msl1, msl2 = sim1._msl_ports[0], sim2._msl_ports[0]
    assert msl1.width == msl2.width == W_TRACE
    assert msl1.height == msl2.height == H_SUB
    assert msl1.eps_r_sub == msl2.eps_r_sub == EPS_SUB
    assert msl1.position[2] == msl2.position[2] == N_SUB_LO * DX
    # feed_x (position[0]) and direction are the ONE MSL field allowed to
    # differ -- the probe ladder's own anchor, an instrument parameter.
    assert msl1.direction == msl2.direction == "-x"

    # Materials: same eps_r for both registered materials, same names.
    assert sim1._materials.keys() == sim2._materials.keys() == {"sub", "ptfe"}
    assert sim1._materials["sub"].eps_r == sim2._materials["sub"].eps_r == EPS_SUB
    assert sim1._materials["ptfe"].eps_r == sim2._materials["ptfe"].eps_r == EPS_COAX


# ---------------------------------------------------------------------------
# Issue #589 Step B -- attempt-2 WIDE fixture: a domain-clearance FALSIFIER
# for the named candidate (c) "absorber-proximity artifact", NOT attempt 3.
#
# Reconciliation with rfx/api/_spec.py's CoaxMSLTransitionResult docstring
# ("the next step is the settled VESSL run, not a third ladder/clearance
# change") and with PREDECLARATION_ATTEMPT2["x_cpml_clearance"]: Step A of
# the #589 plan IS that settled run re-taken with a fuller record dump
# (scripts/diagnostics/coax_msl_transition_settled_run.py, same
# _build_coax_msl_transition_sim_attempt2 fixture, same n_steps). Step B
# below is a FALSIFIER VARIANT run alongside it, not a replacement fixture
# and not a new attempt: the settled numbers stay pinned to attempt 2; the
# wide fixture only asks whether those numbers MOVE when the junction is
# taken away from the CPML (col_power / |S22| shift > 0.10 at 6 or 8 GHz
# => the committed numbers are absorber-proximity-limited; invariant within
# 0.05 (|S00| 0.02) => candidate (c) is falsified). Predeclared thresholds
# live in the #589 design record; this file adds no gate on them.
#
# Geometry deltas vs attempt 2 (everything else byte-identical, asserted by
# test_attempt2_wide_junction_cells_are_byte_identical_to_attempt2):
#   LY        3.4 -> 6.8 mm  (Y_C recentred: 1.7 -> 3.4 mm; substrate edge
#                             to trace edge 1.4 -> 3.1 mm on each +/-y side)
#   JUNCTION_X 1.0 -> 3.0 mm (junction to -x CPML inner edge: 10 -> 30 cells)
#   LX        12.5 -> 14.5 mm and FEED_X 11.0 -> 13.0 mm -- both move WITH
#             the junction, so the MSL ladder/feed geometry relative to the
#             junction and the +x CPML clearance (1.5 mm) are unchanged
#             (review item 8).
#   dx, cpml_layers=8, z-stack, FREQ_MAX_2, ladder recipe: unchanged.
# Cell count 163*85*56 padded (743,820 interior-equivalent per the design's
# arithmetic) vs attempt 2's 142*51*56 -- ~1.9x, so the same n_steps costs
# ~1.9x the GPU time; ring-down may be SLOWER (bigger substrate slab): a
# settling_db > -40 dB at 135000 steps makes B's numbers truncated and
# UNPINNABLE (report, do not pin; name a 2x-length rerun instead), and
# _warn_if_ringdown_truncated (#662) will say so verbatim in the run log.
#
# KNIFE-EDGE NOTE (why the trace box below is placed by half-cells): attempt
# 2 declares the trace as Box(y = Y_C -/+ W_TRACE/2) = 1.4/2.0 mm, both
# exactly on node planes -- the rfx/geometry/csg.py Box docstring's
# "knife edge". Measured on the committed fixture (_assemble_materials,
# pure NumPy): the realized pec_mask trace occupies y nodes 14..20
# INCLUSIVE at the trace z-layer (7 node rows; both edge nodes included by
# float32 rounding) and x nodes >= 11 -- the junction node x=10 itself is
# EXCLUDED from the trace box (it carries only the pin). The same declared
# construction recentred at (3.0, 3.4) mm (y edges 3.1/3.7 mm) rounds the
# OTHER way on all three edges: 5 node rows AND the junction x-column
# included -- 59 pec_mask cells differ, i.e. the naive wide build is NOT
# junction-identical (this is the fail-before-fix evidence for the
# structural test below; a second iteration with only the y edges pinned
# still differed at 3 cells, the junction x-column). The wide builder therefore places the trace box on cell
# MIDPOINTS (the _half_cell_box_z recipe, applied to x and y) so that it
# realizes EXACTLY attempt 2's own node set after the offset. The MSL
# port's registered width stays W_TRACE (600 um) in both fixtures. That
# attempt 2's declared 600 um trace realizes as 7 electric-wall node rows
# is itself a declared-vs-realized fidelity observation on the COMMITTED
# fixture (reported via --preflight's fidelity_report in the driver), not
# something this variant changes.
# ---------------------------------------------------------------------------
LY_2W = 6.8e-3
Y_C_2W = LY_2W / 2.0
JUNCTION_X_2W = 3.0e-3
FEED_X_2W = FEED_X_2 + (JUNCTION_X_2W - JUNCTION_X)   # 13.0 mm
LX_2W = LX_2 + (JUNCTION_X_2W - JUNCTION_X)           # 14.5 mm
# Node offsets of the wide fixture relative to attempt 2 (same dx on every
# axis; z unchanged).
WIDE_X_OFFSET_CELLS = int(round((JUNCTION_X_2W - JUNCTION_X) / DX))  # 20
WIDE_Y_OFFSET_CELLS = int(round((Y_C_2W - Y_C) / DX))                # 17
# Attempt 2's REALIZED trace node set (measured, see KNIFE-EDGE NOTE):
# y nodes [Y_C_NODE - 3, Y_C_NODE + 3] inclusive, x nodes >= JUNCTION_X_NODE + 1.
_TRACE_HALF_NODES = 3
_TRACE_X_START_OFFSET_NODES = 1
# Declared junction window for the byte-identity assertion, in attempt-2
# DOMAIN node indices (CPML pad excluded), all z: 0..4.0 mm in x (1.0 mm
# before the junction through 3.0 mm of trace/ladder start; covers the
# clearance disk r=0.4 mm, the coax outer radius 0.6 mm and the first MSL
# probe plane region) and the full attempt-2 y interior.
JUNCTION_WINDOW_ATTEMPT2 = {"x": (0, 40), "y": (0, 34)}


def _half_cell_box(n_lo, n_hi):
    """Same recipe as _half_cell_box_z, for any axis with spacing DX."""
    return (n_lo - 0.5) * DX, (n_hi + 0.5) * DX


def _build_coax_msl_transition_sim_attempt2_wide():
    """Step B (issue #589): attempt 2 with the junction moved away from the
    CPML on three sides -- a falsifier for candidate (c) absorber
    proximity, NOT a third attempt (see the block comment above; Step A,
    the settled run re-taken, is the claims-bearing measurement and this
    variant only tests whether its numbers are fixture-specific).

    Everything at and around the junction is attempt 2's own cells after a
    (+20, +17, 0) node offset; the trace box is placed by half-cells to
    pin attempt 2's measured realization (KNIFE-EDGE NOTE above).
    """
    from rfx.api import Simulation
    from rfx.boundaries.spec import BoundarySpec
    from rfx.geometry.csg import Box, Cylinder

    sim = Simulation(
        freq_max=FREQ_MAX_2, domain=(LX_2W, LY_2W, LZ_2), dx=DX, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml", z="cpml"),
    )
    sim.add_material("sub", eps_r=EPS_SUB)
    sim.add_material("ptfe", eps_r=EPS_COAX)

    gnd_lo, gnd_hi = _half_cell_box_z(N_GND, N_GND)
    sim.add(Box((0.0, 0.0, gnd_lo), (LX_2W, LY_2W, gnd_hi)), material="pec")
    clr_c, clr_h = _margin_cylinder_z(N_GND, N_SUB_LO)
    sim.add(
        Cylinder(center=(JUNCTION_X_2W, Y_C_2W, clr_c), radius=CLEAR_R, height=clr_h, axis="z"),
        material="ptfe",
    )
    sub_lo, sub_hi = _half_cell_box_z(N_SUB_LO, N_SUB_HI)
    sim.add(Box((0.0, 0.0, sub_lo), (LX_2W, LY_2W, sub_hi)), material="sub")
    trc_lo, trc_hi = _half_cell_box_z(N_TRACE, N_TRACE)
    # Trace: pin attempt 2's realized node set (x >= junction node + 1, y
    # within +/- _TRACE_HALF_NODES of the centre node) via half-cell placement.
    jx_node = int(round(JUNCTION_X_2W / DX))
    yc_node = int(round(Y_C_2W / DX))
    trc_x_lo, _ = _half_cell_box(jx_node + _TRACE_X_START_OFFSET_NODES, jx_node)
    trc_y_lo, trc_y_hi = _half_cell_box(yc_node - _TRACE_HALF_NODES, yc_node + _TRACE_HALF_NODES)
    sim.add(
        Box((trc_x_lo, trc_y_lo, trc_lo), (LX_2W, trc_y_hi, trc_hi)),
        material="pec",
    )
    pin_c, pin_h = _margin_cylinder_z(N_GND, N_TRACE)
    sim.add(
        Cylinder(center=(JUNCTION_X_2W, Y_C_2W, pin_c), radius=PIN_R, height=pin_h, axis="z"),
        material="pec",
    )

    sim.add_coaxial_port(
        position=(JUNCTION_X_2W, Y_C_2W, N_GND * DX), face="bottom",
        pin_radius=PIN_R, outer_radius=OUTER_R, impedance=50.0,
    )
    sim.add_msl_port(
        position=(FEED_X_2W, Y_C_2W, N_SUB_LO * DX), width=W_TRACE, height=H_SUB,
        direction="-x", impedance=50.0, eps_r_sub=EPS_SUB,
    )
    return sim


def _attempt2_wide_kwargs(n_steps):
    """compute_coax_msl_transition kwargs for the wide fixture: attempt 2's
    own kwargs with only junction_x moved (the ladder recipe is relative
    to the feed, which moved with the junction)."""
    kw = _attempt2_kwargs(n_steps)
    kw["junction_x"] = JUNCTION_X_2W
    return kw


def _junction_window_slices(grid, x_offset_cells, y_offset_cells):
    """Padded-array slices for JUNCTION_WINDOW_ATTEMPT2 on ``grid``, shifted
    by the given domain-node offsets (0, 0 for attempt 2 itself)."""
    x0, x1 = JUNCTION_WINDOW_ATTEMPT2["x"]
    y0, y1 = JUNCTION_WINDOW_ATTEMPT2["y"]
    px, py = int(grid.pad_x_lo), int(grid.pad_y_lo)
    return (
        slice(px + x_offset_cells + x0, px + x_offset_cells + x1),
        slice(py + y_offset_cells + y0, py + y_offset_cells + y1),
        slice(None),
    )


def test_attempt2_wide_junction_cells_are_byte_identical_to_attempt2():
    """Structural (build-only, no FDTD, ~seconds): the Step-B wide fixture
    assembles, and every material array (eps_r, sigma, mu_r) AND the PEC
    mask inside JUNCTION_WINDOW_ATTEMPT2 are np.array_equal to attempt 2's
    after the (+WIDE_X_OFFSET_CELLS, +WIDE_Y_OFFSET_CELLS) node offset.
    Also stamps the method's own coax stub (stamp_coaxial_line +
    stamp_coaxial_annular_resistor at compute_coax_msl_transition's own z
    indices) on both assembled grids. THAT part is NOT byte-identical and
    this test does not pretend it is: the stub's radial rasterization puts
    PIN_R (= 2 dx) and OUTER_R (= 6 dx) exactly on node radii, and the
    float rounding of (x - cx)^2 + (y - cy)^2 resolves differently at
    (3.0, 3.4) mm than at (1.0, 1.7) mm -- measured 2026-08-30: 26 cells
    differ, one PEC-shell node column at r = +6 dx (y) over the whole stub
    height and one annular-resistor cell at r = -2 dx (x). The assertion
    on the stub is therefore the structural statement that every
    differing cell sits ON a knife-edge radius (PIN_R, OUTER_R or the
    shell's inner radius) -- i.e. the difference is float rounding of an
    exactly-on-node radius, not a geometry error -- and the count is
    reported in the failure message. Whether a one-column shell asymmetry
    in the coax stub is acceptable for Step B is a PI call recorded in the
    #589 measurement report, not decided here; the method's stamping is
    production code and is not touched.

    Fail-before-fix evidence (recorded in the KNIFE-EDGE NOTE above): with
    the naive Box(y = Y_C_2W -/+ W_TRACE/2, x from JUNCTION_X_2W) trace the
    pec_mask differs at 59 cells in this window (trace edge rows and the
    junction x-column), so this assertion discriminates.
    """
    from rfx.sources.coaxial_port import (
        coaxial_tem_characteristic_impedance,
        stamp_coaxial_annular_resistor,
        stamp_coaxial_line,
    )

    sim2 = _build_coax_msl_transition_sim_attempt2()
    simw = _build_coax_msl_transition_sim_attempt2_wide()

    # Registered junction parameters: identical up to the declared offset.
    p2, pw = sim2._coaxial_ports[0], simw._coaxial_ports[0]
    assert p2.pin_radius == pw.pin_radius == PIN_R
    assert p2.outer_radius == pw.outer_radius == OUTER_R
    assert p2.face == pw.face == "bottom"
    assert p2.position[2] == pw.position[2] == N_GND * DX
    assert pw.position[0] == JUNCTION_X_2W and pw.position[1] == Y_C_2W
    m2, mw = sim2._msl_ports[0], simw._msl_ports[0]
    assert m2.width == mw.width == W_TRACE
    assert m2.height == mw.height == H_SUB
    assert m2.eps_r_sub == mw.eps_r_sub == EPS_SUB
    assert m2.direction == mw.direction == "-x"
    assert m2.position[2] == mw.position[2] == N_SUB_LO * DX
    # Feed-to-junction offset unchanged (review item 8).
    assert np.isclose(mw.position[0] - JUNCTION_X_2W, m2.position[0] - JUNCTION_X)
    # +x CPML clearance unchanged: LX - FEED_X.
    assert np.isclose(LX_2W - FEED_X_2W, LX_2 - FEED_X_2)

    g2 = sim2._build_grid()
    gw = simw._build_grid()
    assert (g2.pad_x_lo, g2.pad_y_lo, g2.pad_z_lo) == (gw.pad_x_lo, gw.pad_y_lo, gw.pad_z_lo)
    assert (g2.nx, g2.ny, g2.nz) == (142, 51, 56)
    assert (gw.nx, gw.ny, gw.nz) == (142 + WIDE_X_OFFSET_CELLS, 51 + 2 * WIDE_Y_OFFSET_CELLS, 56)
    assert WIDE_X_OFFSET_CELLS == 20 and WIDE_Y_OFFSET_CELLS == 17

    out2 = sim2._assemble_materials(g2)
    outw = simw._assemble_materials(gw)
    mats2, pec2 = out2[0], out2[3]
    matsw, pecw = outw[0], outw[3]
    s2 = _junction_window_slices(g2, 0, 0)
    sw = _junction_window_slices(gw, WIDE_X_OFFSET_CELLS, WIDE_Y_OFFSET_CELLS)

    for name in ("eps_r", "sigma", "mu_r"):
        a = np.asarray(getattr(mats2, name))[s2]
        b = np.asarray(getattr(matsw, name))[sw]
        assert a.dtype == b.dtype and a.shape == b.shape
        assert np.array_equal(a, b), (
            f"{name}: {int(np.sum(a != b))} junction-window cells differ "
            "between attempt 2 and the wide fixture"
        )
    assert pec2 is not None and pecw is not None
    a = np.asarray(pec2, dtype=bool)[s2]
    b = np.asarray(pecw, dtype=bool)[sw]
    assert np.array_equal(a, b), (
        f"pec_mask: {int(np.sum(a != b))} junction-window cells differ; "
        f"first at (x,y,z)={tuple(int(v) for v in np.argwhere(a != b)[0])}"
    )
    # The trace realization this variant pins (KNIFE-EDGE NOTE): attempt 2's
    # own pec_mask at the trace z-layer, a column well past the junction.
    pz = int(g2.pad_z_lo)
    yc_node = int(round(Y_C / DX))
    trace_rows = np.nonzero(a[30, :, pz + N_TRACE])[0]
    assert trace_rows.tolist() == list(
        range(yc_node - _TRACE_HALF_NODES, yc_node + _TRACE_HALF_NODES + 1)
    )
    jx_node = int(round(JUNCTION_X / DX))
    # Junction x-column at the trace layer carries only the pin (not the
    # 7-row trace); the trace starts one node further along +x.
    assert np.nonzero(a[jx_node, :, pz + N_TRACE])[0].tolist() != trace_rows.tolist()
    assert np.nonzero(a[jx_node + _TRACE_X_START_OFFSET_NODES, :, pz + N_TRACE])[0].tolist() == trace_rows.tolist()

    # Method-side coax stub: same z indices compute_coax_msl_transition uses.
    def _stub(sim, grid, mats):
        port = sim._coaxial_ports[0]
        z_j = int(grid.position_to_index(port.position)[2])
        z_lo = int(grid.pad_z_lo) + 2
        z_feed = z_lo + 1
        z_hi = z_j - 1
        cxy = (float(port.position[0]), float(port.position[1]))
        stamped, shell_inner = stamp_coaxial_line(
            grid, mats, center_xy=cxy, z_lo_index=z_lo, z_hi_index=z_hi,
            pin_radius=PIN_R, outer_radius=OUTER_R,
        )
        stamped = stamp_coaxial_annular_resistor(
            grid, stamped, center_xy=cxy, z_index=z_feed,
            pin_radius=PIN_R, outer_radius=OUTER_R,
            target_impedance=float(coaxial_tem_characteristic_impedance(PIN_R, OUTER_R)),
            shell_inner_radius=shell_inner,
        )
        return stamped, float(shell_inner)

    st2, shell_inner2 = _stub(sim2, g2, mats2)
    stw, shell_innerw = _stub(simw, gw, matsw)
    assert shell_inner2 == shell_innerw
    knife_radii = np.array([PIN_R, OUTER_R, shell_inner2]) / DX
    jx_node, yc_node = int(round(JUNCTION_X / DX)), int(round(Y_C / DX))
    n_diff_total = 0
    for name in ("eps_r", "sigma", "mu_r"):
        a = np.asarray(getattr(st2, name))[s2]
        b = np.asarray(getattr(stw, name))[sw]
        diff = np.argwhere(a != b)
        n_diff_total += len(diff)
        for (ix, iy, _iz) in diff:
            r_nodes = float(np.hypot(ix - jx_node, iy - yc_node))
            assert np.min(np.abs(knife_radii - r_nodes)) < 1e-6, (
                f"{name} after coax-stub stamping differs at window cell "
                f"(x,y,z)=({ix},{iy},{_iz}), r={r_nodes:.4f} dx -- NOT a "
                "knife-edge radius; the wide stub is geometrically different"
            )
    # n_diff_total is documented (26 knife-edge cells on 2026-08-30, see the
    # docstring), deliberately NOT pinned: 0 would be an improvement, not a bug.
    del n_diff_total


def test_attempt2_wide_kwargs_differ_from_attempt2_only_in_junction_x():
    k2 = _attempt2_kwargs(135000)
    kw = _attempt2_wide_kwargs(135000)
    assert set(k2) == set(kw)
    for key in k2:
        if key == "junction_x":
            assert k2[key] == JUNCTION_X and kw[key] == JUNCTION_X_2W
        elif key == "freqs":
            assert np.array_equal(k2[key], kw[key])
        else:
            assert k2[key] == kw[key], key


@pytest.mark.slow_physics
def test_coax_msl_transition_attempt2_instrument_verification():
    """Attempt 2: INSTRUMENT test, not a claims test (issue #585 review
    restructure). Same junction, longer MSL ladder + wider x-CPML
    clearance. Formerly named test_coax_msl_transition_attempt2_
    lengthened_ladder -- renamed because that name and its old body
    asserted claims-bearing physics numbers (reciprocity, |S22|) that a
    second review round showed were UNMEASURED at this settling, not
    passing. See ``PREDECLARATION_ATTEMPT2`` above for the full
    predeclaration.

    RESOLVED, VESSL 369367252283 (2026-08-06): ``SETTLED_RUN_RECORD``
    (status ``RUN``) now carries a fully settled read (both drives clear
    -40 dB) of THIS SAME fixture at ``n_steps=135000`` -- gamma-vs-beta is
    CONFIRMED (a third in-band checkpoint); the earlier passivity-guard
    trip is ATTRIBUTED to truncation (settled max|S|=0.9933, guard clean);
    reciprocity is now MEASURED at 91.4% worst deviation (not attributed
    to any of this lane's three retracted explanations); and the
    column-power open question is SHARPENED (mostly missing at 6/8 GHz,
    only ~20% missing at 10 GHz), with a named discriminating check
    (closed-box PEC-wall variant) as the next step. See
    ``SETTLED_RUN_RECORD["measured"]["verdict"]`` for the full accounting.
    THIS test's own numbers below are the 20000-/45000-step checkpoints
    that predate settling -- kept exactly as measured, as the historical
    record the settled run's own invariance argument rests on.

    Runtime ~20 min (two FDTD drives on a (142, 51, 56)-cell grid, 45000
    steps each).

    WHAT THIS TEST ASSERTS, AND WHY (issue #585 review restructure item
    (b)): only quantities that PASS the run-length invariance test
    (stable between the 20000- and 45000-step checkpoints) are asserted
    as pass/fail claims. Quantities that FAIL it are recorded (both
    checkpoints disclosed, issue #585 review finding M2) but NOT gated --
    asserting a tight bound on a value known to still be moving would
    itself be the "surface-metric gate binding an artifact" class this
    repo's own R5 rule exists to prevent.

    CONFIRMED, PROVISIONAL pending a settled run:
    * gamma-vs-beta ratio (coax-driven fit): stable at both checkpoints
      (20000: 1.085/0.826/0.976; 45000: 1.128/0.854/1.071), fully inside
      the predeclared [0.8, 1.3] band both times -- asserted as a real
      pass below.
    * cond_a_equilibrated: stable near 1 at both checkpoints -- confirms
      the two drives are NOT geometrically near-degenerate (this reading
      survives; see the RETRACTED amplitude-gap note below for what does
      NOT survive).

    UNMEASURED AT THIS SETTLING -- recorded, not gated (both checkpoints
    disclosed per review finding M2):
    * reciprocity worst deviation: 20000 steps 0.824 -> 45000 steps
      0.938 -- moved AWAY from any acceptance, not toward one.
    * |S22|: 20000 steps 0.451 -> 45000 steps 1.104 (2.4x) -- still
      growing, not settled.
    * max|S|: 20000 steps 0.9933 -> 45000 steps 1.1038 -- crosses the
      passivity-guard hard limit (1.10) BETWEEN checkpoints. The guard
      fires on the 45000-step run (verified below): "column power 1.218
      exceeds limit 1.1... UNRELIABLE... do not interpret as physics."
      strict_passivity semantics are exercised via that same shared
      guard function directly on this run's own result (equivalent to
      passing strict_passivity=True to the call itself, without a second
      ~20-minute FDTD run for a check that needs no FDTD).

    THE OPEN QUESTION (issue #585 review finding B5, replaces the
    RETRACTED amplitude-gap story below): column power on the MSL-driven
    column, Sum|S_ij|^2 over i for driven port msl. Scaling-INVARIANT
    (unlike drive amplitude -- see below), so this is real evidence, not
    a normalization artifact: 20000 steps [0.0018, 0.0199, 0.204], 45000
    steps [0.0104, 0.0119, 1.218] -- mostly far below 1 (power missing
    from a nominally lossless structure) with one bin above 1 at the
    less-settled checkpoint (impossible for a passive structure --
    itself evidence of non-convergence). Where does the MSL column's
    power go? Not answered by this attempt; recorded as the productive
    question a settled run should target.

    RETRACTED (issue #585 review finding B1 -- do not repeat, third
    retracted attribution on this lane): a prior revision of this test
    attributed the reciprocity miss to a ~1.8e7-3.3e7x coax/MSL drive-
    amplitude gap, with a locked ``amplitude_ratio > 1.0e6`` assertion.
    This is mathematically impossible: per-drive (column) rescaling of
    the two-drive solve leaves S EXACTLY invariant (A'=A@D, B'=B@D =>
    B'@inv(A')=B@A^-1=S for any invertible diagonal D -- verified at
    this attempt's own gap value, |S'-S| ~ 3e-16). The "amplitude_ratio"
    quantity is also, separately, raw ``cond_a`` under a new name
    (matches to 8 significant figures) -- the exact quantity this file's
    own docstrings already say not to read as a degeneracy witness on
    this lane. See ``PREDECLARATION_ATTEMPT2["retracted_amplitude_gap_
    attribution"]`` for the full record. The assertion is DELETED, not
    just unused.
    """
    assert PREDECLARATION_ATTEMPT2["status"] == "RUN"

    sim = _build_coax_msl_transition_sim_attempt2()
    result = sim.compute_coax_msl_transition(
        junction_x=JUNCTION_X, eps_r_sub=EPS_SUB, n_steps=N_STEPS_2,
        freqs=FREQS_2,
        probe_count=6, probe_start_cells=4, probe_spacing_cells=2,
        msl_probe_count=PROBE_COUNT_2, msl_probe_start_cells=PROBE_START_2,
        msl_probe_spacing_cells=PROBE_SPACING_2,
        skip_preflight=True, strict_passivity=False,
    )

    assert result.s_params.shape == (2, 2, 3)
    assert np.all(np.isfinite(result.s_params))
    assert result.port_names == ("coax", "msl")

    # --- CONFIRMED-provisional (invariant-verified) ------------------------

    # Settling DIRECTION only -- more negative than the 20000-step
    # checkpoint's own -12.3/-10.7 dB, not a claim of adequate settling
    # (neither of THIS test's own two checkpoints clears the -40 dB rule;
    # the settled n_steps=135000 run that does is SETTLED_RUN_RECORD,
    # status RUN, a separate recorded run, not re-run by this test).
    assert np.all(np.asarray(result.settling_db) < -15.0), (
        f"settling_db {result.settling_db} did not improve past the "
        "-15 dB floor (between the 20000-step checkpoint's -12.3/-10.7 "
        "dB and this run's own measured -19.7/-17.9 dB) -- a settling "
        "regression (unrelated code change) or a construction change; "
        "this is a monotonic-improvement floor, not a claim of adequate "
        "settling (neither checkpoint clears the -40 dB rule)."
    )

    # cond_a_equilibrated: stable near 1 at both checkpoints -- confirms
    # NOT geometric near-degeneracy (issue #581 review finding B2's own
    # correction, still holding).
    assert np.all(np.asarray(result.cond_a_equilibrated) < 1.5), (
        f"cond_a_equilibrated {result.cond_a_equilibrated} rose above the "
        "near-unity floor stable at both checkpoints -- the two drives "
        "may have become genuinely geometrically near-degenerate; "
        "re-derive the attribution."
    )

    # gamma-vs-beta (predeclared band [0.8, 1.3]): the ONE discriminant
    # that passes the run-length invariance test -- asserted as a real,
    # confirmed-provisional pass.
    from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff
    from rfx.core.yee import EPS_0, MU_0
    c0 = 1.0 / np.sqrt(MU_0 * EPS_0)
    _, eps_eff_hj = hammerstad_jensen_z0_eps_eff(W_TRACE, H_SUB, EPS_SUB)
    beta_analytic = 2.0 * np.pi * np.asarray(result.freqs) * np.sqrt(eps_eff_hj) / c0
    im_gamma_coax_driven = np.abs(result.gamma[1, 0, :].imag)
    gamma_ratio = im_gamma_coax_driven / beta_analytic
    gamma_ok = bool(np.all((gamma_ratio >= 0.8) & (gamma_ratio <= 1.3)))
    assert gamma_ok, (
        f"gamma_ratio {gamma_ratio} left the predeclared [0.8, 1.3] "
        "confirmation band, stable at both checkpoints until now -- the "
        "ladder-length fix's own headline result (the MSL matrix-pencil "
        "fit tracking the guided-wave beta) may no longer hold; "
        "re-derive the extraction-class verdict."
    )
    np.testing.assert_allclose(
        gamma_ratio,
        PREDECLARATION_ATTEMPT2["measured"]["gamma_ratio_coax_driven_by_checkpoint"][1],
        atol=0.05,
    )

    # --- The open question (B5): column power, scaling-invariant --------

    S = result.s_params  # local alias -- NOT result.a_inc, a different field
    col_msl_driven_power = np.abs(S[0, 1, :]) ** 2 + np.abs(S[1, 1, :]) ** 2
    _m_col = np.asarray(
        PREDECLARATION_ATTEMPT2["measured"]["column_power_msl_driven_by_checkpoint"][1]
    )
    np.testing.assert_allclose(col_msl_driven_power, _m_col, atol=0.01)
    # Pinned as the measured OPEN observation, not a pass/fail physics
    # claim: regression-lock that the finding itself (power missing from
    # the MSL-driven column) persists, so a future change that quietly
    # "fixes" this without investigation gets flagged for review.
    assert col_msl_driven_power[0] < 0.5 and col_msl_driven_power[1] < 0.5, (
        f"MSL-driven column power at the two lower bins {col_msl_driven_power[:2]} "
        "rose above the 0.5 floor measured on this attempt -- the "
        "'power going missing' open question may have resolved (or the "
        "extraction changed); re-derive the finding before assuming a fix."
    )

    # --- UNMEASURED at this settling: recorded, NOT gated -----------------
    # (issue #585 review findings B2/B4/M2: both checkpoints disclosed;
    # no pass/fail bound asserted on a value known to still be moving.)
    pair, worst_dev = _mixed_reciprocity_deviation(result.s_params)
    s22_abs = np.abs(result.s_params[1, 1, :])
    max_abs_s = float(np.max(np.abs(result.s_params)))
    # Sanity only (finite, not NaN/inf) -- explicitly NOT a physics gate.
    assert np.isfinite(worst_dev) and np.all(np.isfinite(s22_abs)) and np.isfinite(max_abs_s)

    # --- Passivity guard: proves the honest gate still fires (issue #585
    # review finding B3 -- no silent gate removal). Exercises the SAME
    # shared guard function compute_coax_msl_transition's own
    # strict_passivity=True would invoke, directly on this run's already-
    # obtained result -- equivalent verification without a second ~20-
    # minute FDTD run (the guard itself is a pure post-processing check,
    # no FDTD involved). On the real strict_passivity=True path,
    # compute_coax_msl_transition does not call _warn_if_nonpassive_smatrix
    # directly either -- it returns through the shared
    # rfx.api._sparams._finalize_sparam_result(result_obj,
    # extractor="compute_coax_msl_transition", strict=strict_passivity)
    # epilogue (also used by compute_waveguide_s_matrix and
    # compute_coaxial_s_matrix), which is the one call site that actually
    # invokes this guard function in production; calling it directly here
    # is a faithful stand-in, not a different code path (issue #585
    # final-verify, nit n2).
    from rfx.api._sparams import _warn_if_nonpassive_smatrix
    with pytest.raises(ValueError, match="UNRELIABLE"):
        _warn_if_nonpassive_smatrix(
            result, extractor="compute_coax_msl_transition", strict=True,
        )


# ---------------------------------------------------------------------------
# Part 5 -- #589 flux-adjudication instrument: the extra_flux_monitors= opt-in.
#
# The issue #589 pre-declaration (2026-08-07 comment) commits this PR to ONE
# witness before any adjudication run is interpreted: the S output must be
# BIT-IDENTICAL with and without monitors attached (the opt-in is an
# energy-witness channel, not an extractor change), and the registered-
# monitor guard must stay intact. The fast tests below pin the guard and the
# entry validation; the slow_physics test pins the bit-identity witness on
# the SAME attempt-2 fixture the adjudication run uses (step count is
# instrument calibration only -- bit-identity is step-count independent).
# ---------------------------------------------------------------------------

def _attempt2_kwargs(n_steps):
    return dict(
        junction_x=JUNCTION_X, eps_r_sub=EPS_SUB, n_steps=n_steps,
        freqs=FREQS_2,
        probe_count=6, probe_start_cells=4, probe_spacing_cells=2,
        msl_probe_count=PROBE_COUNT_2, msl_probe_start_cells=PROBE_START_2,
        msl_probe_spacing_cells=PROBE_SPACING_2,
        skip_preflight=True, strict_passivity=False,
    )


def _attempt2_scratch_flux_entries():
    """Flux entries built the documented way: a scratch Simulation sharing
    the attempt-2 domain, handing over its ._flux_monitors list."""
    from rfx.api import Simulation as _Sim
    scratch = _Sim(freq_max=FREQ_MAX_2, domain=(LX_2, LY, LZ_2), dx=DX,
                   boundary="cpml")
    scratch.add_flux_monitor(
        axis="x", coordinate=2.2e-3, freqs=FREQS_2,
        size=(1.2e-3, 0.7e-3), center=(Y_C, 2.8e-3), name="xplane_aperture",
    )
    scratch.add_flux_monitor(
        axis="z", coordinate=3.0e-3, freqs=FREQS_2,
        size=(1.3e-3, 1.2e-3), center=(2.15e-3, Y_C), name="ztop_patch",
    )
    return scratch._flux_monitors


def test_extra_flux_monitors_registered_guard_unchanged():
    """Monitors registered ON the sim still raise -- the opt-in did not
    loosen the builds-its-own guard."""
    sim = _build_coax_msl_transition_sim_attempt2()
    sim.add_flux_monitor(axis="x", coordinate=2.2e-3, freqs=FREQS_2)
    with pytest.raises(ValueError, match="flux monitors"):
        sim.compute_coax_msl_transition(**_attempt2_kwargs(1))


def test_extra_flux_monitors_entry_validation():
    """Malformed opt-in entries fail loudly BEFORE any FDTD work."""
    with pytest.raises(TypeError, match="add_flux_monitor"):
        _build_coax_msl_transition_sim_attempt2().compute_coax_msl_transition(
            **_attempt2_kwargs(1), extra_flux_monitors=[{"name": "not_an_entry"}],
        )

    from rfx.api import Simulation as _Sim
    off_domain = _Sim(freq_max=FREQ_MAX_2, domain=(50.0e-3, LY, LZ_2), dx=DX,
                      boundary="cpml")
    off_domain.add_flux_monitor(axis="x", coordinate=30.0e-3, freqs=FREQS_2,
                                name="outside_attempt2_x")
    with pytest.raises(ValueError, match="outside"):
        _build_coax_msl_transition_sim_attempt2().compute_coax_msl_transition(
            **_attempt2_kwargs(1),
            extra_flux_monitors=off_domain._flux_monitors,
        )

    dup = _Sim(freq_max=FREQ_MAX_2, domain=(LX_2, LY, LZ_2), dx=DX,
               boundary="cpml")
    dup.add_flux_monitor(axis="x", coordinate=2.2e-3, freqs=FREQS_2, name="d")
    dup.add_flux_monitor(axis="x", coordinate=2.3e-3, freqs=FREQS_2, name="d")
    with pytest.raises(ValueError, match="duplicate"):
        _build_coax_msl_transition_sim_attempt2().compute_coax_msl_transition(
            **_attempt2_kwargs(1), extra_flux_monitors=dup._flux_monitors,
        )


@pytest.mark.slow_physics
def test_extra_flux_monitors_do_not_perturb_s():
    """The #589 non-perturbation witness: S bit-identical with and without
    the opt-in monitors, on the attempt-2 fixture itself.

    Bit-identity (not allclose) is deliberate: the monitors add read-only
    DFT accumulation to the scan carry, and any drift here would mean the
    instrument perturbs the thing it measures. Measured before commit on
    CPU (2026-08-07): identical, and off/off repeatability also identical.
    If a future backend/XLA version reorders the fused field math under the
    added carry, this test reds -- that is a REAL finding about the witness
    (re-measure and re-declare on #589 before trusting adjudication runs),
    not a tolerance to loosen silently.
    """
    n_steps = 1000  # bit-identity is step-count independent; keep it cheap
    r_off = _build_coax_msl_transition_sim_attempt2().compute_coax_msl_transition(
        **_attempt2_kwargs(n_steps))
    r_on = _build_coax_msl_transition_sim_attempt2().compute_coax_msl_transition(
        **_attempt2_kwargs(n_steps),
        extra_flux_monitors=_attempt2_scratch_flux_entries(),
    )

    assert r_off.flux_monitors is None
    assert np.array_equal(np.asarray(r_off.s_params), np.asarray(r_on.s_params)), (
        "extra_flux_monitors perturbed the S output -- the #589 "
        "non-perturbation witness failed; do not run the adjudication "
        "until this is re-declared."
    )
    assert np.array_equal(np.asarray(r_off.settling_db),
                          np.asarray(r_on.settling_db))

    assert set(r_on.flux_monitors) == {"coax", "msl"}
    for drive_key, spectra in r_on.flux_monitors.items():
        assert set(spectra) == {"xplane_aperture", "ztop_patch"}
        for name, arr in spectra.items():
            assert arr.shape == (len(FREQS_2),), (drive_key, name, arr.shape)
            assert np.all(np.isfinite(arr)), (drive_key, name, arr)


def test_flux_spectrum_exact_f64_matches_default_on_healthy_magnitudes():
    """#589 C1 iteration 2: the exact_f64 path is the same Poynting sum in
    float64 — on magnitudes far from the float32 subnormal zone the two
    paths must agree to float32 roundoff; on subnormal-zone magnitudes the
    default path collapses (issue #304) while exact_f64 keeps the value.
    Synthetic monitor, no FDTD."""
    import numpy as np
    from rfx.probes.probes import FluxMonitor, flux_spectrum

    rng = np.random.default_rng(42)
    shape = (3, 8, 9)

    def _mon(scale):
        def c64(s):
            return (rng.normal(size=shape) * s
                    + 1j * rng.normal(size=shape) * s).astype(np.complex64)
        return FluxMonitor(
            axis=0, index=5,
            freqs=np.array([6.0e9, 8.0e9, 10.0e9]),
            e1_dft=c64(scale), e2_dft=c64(scale),
            h1_dft=c64(scale), h2_dft=c64(scale),
            dA=np.float32(1.0e-8),
            total_steps=100, window="rect", window_alpha=0.25,
        )

    healthy = _mon(1.0)
    a = np.asarray(flux_spectrum(healthy), dtype=np.float64)
    b = np.asarray(flux_spectrum(healthy, exact_f64=True), dtype=np.float64)
    assert np.allclose(a, b, rtol=5e-6, atol=0.0), (a, b)

    tiny = _mon(1.0e-15)  # products ~1e-30: inside the f32 flush zone
    t64 = np.asarray(flux_spectrum(tiny, exact_f64=True), dtype=np.float64)
    assert np.all(np.isfinite(t64)) and np.any(t64 != 0.0)
