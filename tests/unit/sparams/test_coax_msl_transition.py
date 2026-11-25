"""Tests for ``compute_coax_msl_transition`` (issue #489 leg 4).

#3062 clearance supersession: the frozen attempt records below describe
the former depth-buffered 2*h_sub advisory. Their values and declarations
remain historical; the current absorbing feed threshold is
max((4/3)*h_sub, 4*end_cell), measured from the realized feed node to the
realized interface without a depth buffer. It was measured on one board at
3 cells per h_sub, with 8-32 layers against a 12*h_sub reference; other
meshes and S below that threshold were not measured.

Structure (mirrors ``tests/unit/sparams/test_coax_two_port_smatrix.py``, the
closest precedent):

1. Pure-assembly tests (no FDTD) for
   :func:`rfx.api._sparams._assemble_coax_msl_transition_from_voltages` —
   PLANTED analytic voltages recover a KNOWN S-matrix, and a dedicated
   regression test proves the per-port ``sqrt(Z0)`` power-wave
   normalization is load-bearing (the pre-declared "impedance-convention
   mismatch" failure mode: skipping it silently corrupts the off-diagonal
   when the two ports' reference impedances differ).
2. A PREDECLARATION fill-contract test (UNRUN <=> no numbers, mirrors the
   repository's ``REPRODUCE_GATE_RECORD`` pattern) for the one committed
   FDTD fixture.
3. The FDTD fixture itself (``@pytest.mark.slow_physics``), asserting the
   predeclared witnesses at DIAGNOSTIC honesty level — this gate pins the
   MEASURED envelope of THIS fixture only, no claim beyond it.
4. ATTEMPT 2 (R2's own escape clause): attempt 1's own
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

#931 — NOT MIGRATED HERE, AND WHY (see
``tests/unit/sparams/_results_931/RECOMPUTE.md`` R6).

This is the only fixture in the S-parameter/port suite with a genuine
three-dimensional conductor stack, and the lattice ownership contract
touches every part of it:

* the ground plane and the trace are ``_half_cell_box_z(n, n)`` — a recipe
  that exists ONLY to make a Box rasterize to exactly one node plane. That
  is a sheet declaration written in the language of a volume, and §1.3
  gives it a first-class spelling: a zero-thickness Box with a static
  integer plane. Left as volumes they realize a wall on node 26 as well as
  25 — node 26 is ``N_SUB_LO``, the substrate's own bottom plane — which
  shorts the laminate's lower face and destroys the MSL launch;
* the pin ``Cylinder`` is a volume and gains its far end plane;
* ``_TRACE_Y_LO_OFFSET_NODES = -3`` / ``_TRACE_Y_HI_OFFSET_NODES = +2`` is a
  hard-coded compensation for the old rule dropping a footprint's hi row —
  exactly the class of local repair this branch deletes;
* every realized-count constant in the attempt-3 block (``N_GROUND_BOXES_3``,
  ``HOLE_CELLS_3``, ``ANNULUS_CELLS_3``, ``LIP_CELLS_3``, ``SHELL_CELLS_3``,
  ``TRACE_NODE_ROWS_2`` and the entity cell counts) is a measurement of the
  old realization, and the continuity checks read a CELL column that a
  sheet-declared ground plane no longer populates;
The attempt-3 and wide-trace tests were removed with the deprecation.

The PREDECLARATION / PREDECLARATION_ATTEMPT2 blocks and ``SETTLED_RUN_RECORD``
are frozen; the post-contract junction needs its OWN attempt predeclaration
and its own GPU run, the way the RASTERIZER NOTE already marks these records
as pre-#834. Editing the frozen numbers in place would be the thing that
discipline exists to prevent, so nothing here is touched until that
predeclaration is written.

#931 PHASE 2b — WHAT WAS DONE HERE, AND WHAT IS STILL OWED.

The eleven witnesses that went red were reading the primal CELL mask to
answer a question about a NODE PLANE. Those are different indices under the
contract, and this fixture's own ``_half_cell_box_z(n, n)`` recipe makes the
gap visible: the ground plane's cells sit at z index 32 (node 24) while the
junction node is 33 (node 25), so ``pec_mask[:, :, 33]`` read 2 of 36 annulus
cells. They now read the realization (``_realized_node_pec`` /
``_assemble_junction_realized``, both thin wrappers on
``realized_pec_edge_masks`` through ``tests/_realized_geometry.py``, with the
vectorized reader cross-checked against ``realized_wall_planes``' own
per-column form). Not one count, radius or window size changed.

Six of them are green on the realization channel and say exactly what they
always said: attempt 1 and attempt 2 short the launch (annulus 36/36 in-plane
PEC, wide ring 68/68, no open node in the 9x9 window), the coax shell contacts
ground 32/32 and the ground lip is 32/32 on both fixtures, and the trace is
untouched.

The attempt-3 and wide-trace tests were removed with the deprecation.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfx.api._sparams import (
    _assemble_coax_msl_transition_from_voltages,
    _mixed_reciprocity_deviation,
)
from tests._gate_policy import gate_from_envelope
from tests._wave_convention import plant_ladder_voltages_physical



def _plant_ab_power_wave(s_true, gamma_t, n_f):
    """Power-wave a/b at both ports for both drives, terminator ``gamma_t``.

    Identical signal-flow construction to
    ``tests/unit/sparams/test_coax_two_port_smatrix.py::_plant_ab`` / the underlying
    ``tests/unit/sparams/test_coax_two_port_smatrix.py::_plant`` (``a[j, i]`` / ``b[j, i]``
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



_S_TRUE = (
    np.array([0.12 - 0.04j, 0.18 + 0.02j, -0.03 - 0.09j]),
    np.array([0.62 + 0.18j, 0.55 - 0.25j, 0.48 + 0.30j]),
    np.array([0.62 + 0.18j, 0.55 - 0.25j, 0.48 + 0.30j]),
    np.array([-0.08 - 0.05j, 0.06 + 0.10j, 0.11 - 0.02j]),
)
_N_F = 3
_GAMMA_T = 0.05
_GAMMA_LINE = 1j * np.array([35.0, 47.0, 61.0])
_Z0_COAX = 50.0
_Z0_MSL = 74.3


def _s_matrix_per_freq(s_true, n_f):
    return [
        np.array([[s_true[0][fi], s_true[1][fi]], [s_true[2][fi], s_true[3][fi]]])
        for fi in range(n_f)
    ]


def _build_planted_fixture(z0_coax, z0_msl):
    """Common planted geometry shared by the tests below.

    Planted from GEOMETRY (#822), never from the extractor's labels: see
    ``tests/_wave_convention.py``. The two ``dut_sign`` values are the only
    place the geometry enters, and each is ASSERTED below against this
    fixture's own realized plane positions rather than written as a bare
    literal.

    This fixture mirrors ``compute_coax_msl_transition``'s real geometry,
    where BOTH reference planes are placed AT the junction and therefore sit
    on the DUT side of their probes:

      * coax port: probes at 0.00 - 0.05 m, reference plane at 0.07 m --
        ABOVE the ladder, with the DUT (the junction) further above, so the
        incident wave travels ``+z`` and ``dut_sign = +1``.
      * msl port: probes at 0.15 - 0.20 m, reference plane at 0.13 m --
        BELOW the ladder, with the DUT below it, so the incident wave
        travels ``-x`` and ``dut_sign = -1``.

    That is the OPPOSITE arrangement to ``compute_coaxial_two_port``, whose
    reference planes sit at the feeds, on the far side of the probes from
    the DUT. One constant cannot serve both lanes; the geometry is what
    tells them apart.
    """
    assert np.all(_S_TRUE[0] != _S_TRUE[3]) and np.array_equal(_S_TRUE[1], _S_TRUE[2]) and all(np.all(np.diff(values) != 0) for values in _S_TRUE), 'Asymmetric, reciprocal, frequency-varying truth distinguishes wave-role and frequency-index errors.'
    a_pow, b_pow = _plant_ab_power_wave(_S_TRUE, _GAMMA_T, _N_F)
    sqrt_z0 = np.array([np.sqrt(z0_coax), np.sqrt(z0_msl)])
    a_volt = a_pow * sqrt_z0[:, None, None]
    b_volt = b_pow * sqrt_z0[:, None, None]

    z_coax_planes_m = np.array([0.00, 0.01, 0.02, 0.03, 0.04, 0.05])
    ref_coax_m = 0.07
    x_msl_planes_m = np.array([0.15, 0.16, 0.17, 0.18, 0.19, 0.20])
    assert np.all(np.diff(x_msl_planes_m) > 0), 'MSL probe coordinates must be strictly increasing for the extractor.'
    ref_msl_m = 0.13

    dut_sign_coax = float(np.sign(ref_coax_m - z_coax_planes_m.mean()))
    dut_sign_msl = float(np.sign(ref_msl_m - x_msl_planes_m.mean()))
    assert dut_sign_coax == +1.0 and ref_coax_m > z_coax_planes_m.max()
    assert dut_sign_msl == -1.0 and ref_msl_m < x_msl_planes_m.min()

    assert len(np.unique(_GAMMA_LINE.imag)) == _N_F, 'Distinct per-frequency beta makes frequency-index errors observable.'
    v_coax_by_drive = np.stack([
        plant_ladder_voltages_physical(
            a_volt[0, drive], b_volt[0, drive], gamma=_GAMMA_LINE,
            planes_m=z_coax_planes_m, ref_m=ref_coax_m, dut_sign=dut_sign_coax,
        )
        for drive in range(2)
    ], axis=0)
    v_msl_by_drive = np.stack([
        plant_ladder_voltages_physical(
            a_volt[1, drive], b_volt[1, drive], gamma=_GAMMA_LINE,
            planes_m=x_msl_planes_m, ref_m=ref_msl_m, dut_sign=dut_sign_msl,
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
    (s_params, cond_a, cond_a_eq, rec_resid, fit_resid, gamma_fit, a_inc,
     b_out, *_appended) = _assemble_coax_msl_transition_from_voltages(
        **fx, z0_coax=_Z0_COAX, z0_msl=_Z0_MSL,
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
    On the coax<->MSL lane both reference planes sit at the junction, on the DUT side of their probes, so forward_amp is the incident wave.
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
            a_raw[0, drive_idx, fi] = out_c.forward_amp
            b_raw[0, drive_idx, fi] = out_c.backward_amp
            a_raw[1, drive_idx, fi] = out_m.forward_amp
            b_raw[1, drive_idx, fi] = out_m.backward_amp

    solve = solve_two_port_from_wave_amplitudes(a_raw, b_raw)
    s_wrong = solve.s_params
    s_mat_per_freq = _s_matrix_per_freq(_S_TRUE, n_f)

    ratio = np.sqrt(_Z0_COAX / _Z0_MSL)
    for fi in range(n_f):
        true = s_mat_per_freq[fi]
        np.testing.assert_allclose(s_wrong[0, 0, fi], true[0, 0], atol=1e-9)
        np.testing.assert_allclose(s_wrong[1, 1, fi], true[1, 1], atol=1e-9)
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
    ladder_span_m = MSL_PROBE_SPACING_CELLS * (MSL_PROBE_COUNT - 1) * DX
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



MSL_PROBE_SPACING_CELLS = 2
MSL_PROBE_COUNT = 6

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
        terminates=sim._geometry[-2].shape,
    )
    sim.add_msl_port(
        position=(FEED_X, Y_C, N_SUB_LO * DX), width=W_TRACE, height=H_SUB,
        direction="-x", impedance=50.0, eps_r_sub=EPS_SUB,
    )
    return sim


def test_attempt1_pin_axis_pec_column_is_continuous_because_ground_is_solid():
    """Non-FDTD (~1s) connectivity check -- the claim behind N5/the module
    comment's "missing PEC z-layer" fix, committed as an artifact instead of
    living only in throwaway debugging output (issue #581 review N1).
    Formerly named ``test_pin_axis_pec_connectivity_is_continuous``
    (renamed for issue #589; no other file references the old symbol).

    Reproduces exactly what compute_coax_msl_transition builds on the pin
    axis (registered geometry via _assemble_materials, THEN the coax stub's
    own stamp_coaxial_line, mirroring the method's own ordering) and asserts
    there is no gap: every z-index from the coax stub's near-source end
    through the trace's own layer is either registered PEC (pec_mask) or
    coax-stamped PEC-conductivity (sigma >= PEC_SIGMA/2), with no break.

    WHAT THIS DOES NOT CERTIFY (issue #589 root cause): the column is
    continuous in attempt 1/2 partly BECAUSE the ground plane is a solid
    PEC sheet at the junction node -- the later ``ptfe`` Cylinder only
    overwrites eps_r (``_assemble_materials`` ORs PEC masks and never
    clears one), so the pin passes through solid ground and the coax stub
    is terminated in a short (settled VESSL 369367257263: S00 = -0.9928 at
    6 GHz). Column continuity through a solid ground certifies nothing
    about an OPEN launch. The second assertion below pins that historical
    fact (all 36 clearance-annulus cells at the junction node are PEC) so
    this test reads as what it is: a PEC-column witness, not a launch
    certificate. The attempt-3 and wide-trace tests were removed with the deprecation.
    The pin and the outer conductor are returned as a cell mask rather than written into materials.sigma.
    """
    from rfx.sources.coaxial_port import stamp_coaxial_line, PEC_SIGMA

    sim = _build_coax_msl_transition_sim()
    grid, materials, pec, _i0, _j0, _kj = _assemble_junction_realized(sim)

    port = sim._coaxial_ports[0]
    center_xy = (float(port.position[0]), float(port.position[1]))
    z_junction_idx = int(grid.position_to_index(port.position)[2])
    z_stub_lo = int(grid.pad_z_lo) + 2
    z_stub_hi = z_junction_idx - 1

    materials, _, coax_cells = stamp_coaxial_line(
        grid, materials, center_xy=center_xy, z_lo_index=z_stub_lo,
        z_hi_index=z_stub_hi, pin_radius=PIN_R, outer_radius=OUTER_R,
    )
    sigma = np.asarray(materials.sigma)

    i0, j0 = grid.position_to_index((JUNCTION_X, Y_C, 0.0))[:2]
    trace_idx = int(grid.position_to_index((JUNCTION_X, Y_C, N_TRACE * DX))[2])

    registered_pec = pec[i0, j0, :]
    stamped_pec = (np.asarray(coax_cells)[i0, j0, :]
                   | (sigma[i0, j0, :] >= 0.5 * PEC_SIGMA))
    combined = registered_pec | stamped_pec

    span = combined[z_stub_lo:trace_idx + 1]
    assert np.all(span), (
        f"gap in pin-axis PEC/sigma connectivity between the coax stub "
        f"(z_stub_lo={z_stub_lo}) and the trace (idx={trace_idx}): missing "
        f"indices {z_stub_lo + np.where(~span)[0]}"
    )

    annulus = _lattice_ring_mask(grid, i0, j0, PIN_R_CELLS, CLEAR_R_CELLS)
    n_pec_in_annulus = int((pec[:, :, z_junction_idx] & annulus).sum())
    assert int(annulus.sum()) == ANNULUS_CELLS_3
    assert n_pec_in_annulus == ANNULUS_CELLS_3, (
        f"attempt 1's junction-node clearance annulus is {n_pec_in_annulus}/"
        f"{ANNULUS_CELLS_3} PEC -- this fixture is committed history as a "
        f"SHORTED launch; if this changed, attempt 1 was edited (forbidden)"
    )


PINNED_MEASURED_GATE = 1.49


def test_pinned_measured_gate_provenance():
    """PINNED_MEASURED_GATE's own derivation, kept checkable (issue #581
    review B5): frozen as a literal so the amplitude gate below cannot be
    tautological, but the literal's provenance (gate_from_envelope applied
    to the max|S| measured when it was pinned) stays self-verifying rather
    than a comment-only claim.
    The pinned literal is capped at the 1.10 passivity floor because 1.49 alone would provide no real protection.
    """
    assert gate_from_envelope(0.9928, quantum=100) == PINNED_MEASURED_GATE



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
        "1's junction physics. This is the ONE attempt that "
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
        "n_steps_checkpoints": [20000, 45000],
        "gamma_ratio_coax_driven_by_checkpoint": [
            [1.085, 0.826, 0.976],
            [1.128, 0.854, 1.071],
        ],
        "reciprocity_worst_dev_by_checkpoint": [0.824, 0.938],
        "s22_abs_by_checkpoint": [
            [0.043, 0.141, 0.451],
            [0.102, 0.109, 1.104],
        ],
        "max_abs_s_by_checkpoint": [0.9933, 1.1038],
        "column_power_msl_driven_by_checkpoint": [
            [0.0018, 0.0199, 0.204],
            [0.0104, 0.0119, 1.218],
        ],
        "cond_a_equilibrated": [1.00238, 1.00244, 1.00549],
        "settling_db_by_checkpoint": [[-12.3, -10.7], [-19.69, -17.94]],
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
        "settled_run_resolution": (
            "See SETTLED_RUN_RECORD (status RUN, VESSL 369367252283) for "
            "the full settled-run verdict -- this checkpoint-level "
            "'verdict' string above is preserved as history, not "
            "superseded in place."
        ),
    },
}


SETTLED_RUN_RECORD = {
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
        "n_steps": 135000,
        "settling_db": [-45.9397630721533, -44.17124185980615],
        "settling_cleared_40db": True,
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
    """Fail-loud-honest invariant (mirrors the repository's own
    REPRODUCE_GATE_RECORD tests): UNRUN <=> no numbers, no log path, no
    run id.
    SETTLED_RUN_RECORD['measured'] preserves the run's result JSON numbers verbatim for comparison against a future re-run.
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
    assert "cond_a" in PREDECLARATION_ATTEMPT2["retracted_amplitude_gap_attribution"]
    assert "invariant" in PREDECLARATION_ATTEMPT2["retracted_amplitude_gap_attribution"]


FEED_X_2 = 11.0e-3
LX_2 = 12.5e-3
LZ_2 = JUNCTION_Z + H_SUB + DX + 1.0e-3
FREQS_2 = np.array([6.0e9, 8.0e9, 10.0e9])
FREQ_MAX_2 = 16.0e9
PROBE_START_2 = 4
PROBE_SPACING_2 = 10
PROBE_COUNT_2 = 9


def _build_coax_msl_transition_sim_attempt2():
    """Attempt 2: same junction (reuses attempt 1's own module constants
    unchanged), longer MSL ladder, wider x-CPML clearance, higher band.
    FREQ_MAX_2 re-centers the default excitation at FREQ_MAX_2/2 = 8 GHz on the 6/8/10 GHz band.
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
        terminates=sim._geometry[-2].shape,
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
    assert msl1.direction == msl2.direction == "-x"

    assert sim1._materials.keys() == sim2._materials.keys() == {"sub", "ptfe"}
    assert sim1._materials["sub"].eps_r == sim2._materials["sub"].eps_r == EPS_SUB
    assert sim1._materials["ptfe"].eps_r == sim2._materials["ptfe"].eps_r == EPS_COAX


LY_2W = 6.8e-3
Y_C_2W = LY_2W / 2.0
JUNCTION_X_2W = 3.0e-3
FEED_X_2W = FEED_X_2 + (JUNCTION_X_2W - JUNCTION_X)
LX_2W = LX_2 + (JUNCTION_X_2W - JUNCTION_X)
_TRACE_Y_LO_OFFSET_NODES = -3
_TRACE_Y_HI_OFFSET_NODES = 2
_TRACE_X_START_OFFSET_NODES = 0


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
    jx_node = int(round(JUNCTION_X_2W / DX))
    yc_node = int(round(Y_C_2W / DX))
    trc_x_lo, _ = _half_cell_box(jx_node + _TRACE_X_START_OFFSET_NODES, jx_node)
    trc_y_lo, trc_y_hi = _half_cell_box(yc_node + _TRACE_Y_LO_OFFSET_NODES,
                                        yc_node + _TRACE_Y_HI_OFFSET_NODES)
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
        terminates=sim._geometry[-2].shape,
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


def _load_settled_run_driver():
    import importlib.util
    from pathlib import Path

    path = (Path(__file__).resolve().parents[3] / "scripts" / "diagnostics"
            / "coax_msl_transition_settled_run.py")
    spec = importlib.util.spec_from_file_location(
        "_coax_msl_transition_settled_run_driver_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_589_a3_falsified_requires_both_drives_and_both_residuals():
    """A3 predeclaration: '(b)-on-ladder FALSIFIED' only when the MSL-ladder
    fit_residual AND recurrence_residual are <= 0.02 (and <= 10x own 10 GHz)
    at ALL committed bins under BOTH drives; the MSL-drive residuals alone
    can only SUPPORT (b) (EXTRACTOR), never falsify it."""
    drv = _load_settled_run_driver()
    freqs = np.array([6.0e9, 8.0e9, 10.0e9])
    committed = [0, 1, 2]
    clean = np.full((2, 2, 3), 0.005)

    v = drv._classify_a3(clean, clean, freqs, committed)
    assert v.startswith("(b)-ON-LADDER FALSIFIED"), v
    assert "BOTH drives" in v and "recurrence" in v, v

    fit = clean.copy()
    fit[1, 0, :] = [0.593, 0.626, 0.784]
    v = drv._classify_a3(fit, clean, freqs, committed)
    assert "FALSIFIED" not in v.split(":")[0], v
    assert v.startswith("NON-CLOSING"), v
    assert "coax drive" in v and "NOT falsified" in v, v

    rec = clean.copy()
    rec[1, 1, 0] = 0.05
    v = drv._classify_a3(clean, rec, freqs, committed)
    assert v.startswith("EXTRACTOR ((b)-on-ladder supported)"), v
    assert "recurrence_residual" in v and "6 GHz" in v, v

    fit = clean.copy()
    fit[1, 1, :] = [0.015, 0.004, 0.001]
    v = drv._classify_a3(fit, clean, freqs, committed)
    assert v.startswith("EXTRACTOR") and "10x own 10 GHz" in v, v

    fit = clean.copy()
    fit[1, 0, :] = [0.015, 0.004, 0.001]
    v = drv._classify_a3(fit, clean, freqs, committed)
    assert v.startswith("NON-CLOSING") and "10x" in v, v

    fit = clean.copy()
    fit[0, :, :] = 0.9
    v = drv._classify_a3(fit, clean, freqs, committed)
    assert v.startswith("(b)-ON-LADDER FALSIFIED"), v

    fit = np.full((2, 2, 5), 0.005)
    fit[1, 1, 0] = 0.9
    v = drv._classify_a3(fit, np.full((2, 2, 5), 0.005),
                         np.array([5e9, 6e9, 8e9, 9e9, 10e9]), [1, 2, 4])
    assert v.startswith("(b)-ON-LADDER FALSIFIED") and "committed bins" in v, v


def _a0_baseline_dict(*, S, n_steps, fixture, x64, freqs_all, committed,
                      with_ext=True):
    """Build a baseline JSON dict in the driver's own output shape."""
    S = np.asarray(S)
    fc = np.asarray(freqs_all)[list(committed)]
    Sc = S[:, :, list(committed)]
    d = {
        "n_steps": n_steps,
        "freqs_hz": fc.tolist(),
        "s22_abs": np.abs(Sc[1, 1, :]).tolist(),
        "max_abs_s": float(np.max(np.abs(Sc))),
        "col_msl_driven_power": (np.abs(Sc[0, 1, :]) ** 2 + np.abs(Sc[1, 1, :]) ** 2).tolist(),
    }
    if with_ext:
        d["fixture"] = fixture
        d["ext_589"] = {
            "precision": {"jax_enable_x64": bool(x64)},
            "freqs_hz_all": np.asarray(freqs_all).tolist(),
            "committed_bin_indices": list(committed),
            "s_complex": {"re": S.real.tolist(), "im": S.imag.tolist()},
            "s_abs": {"S00": np.abs(S[0, 0, :]).tolist(), "S10": np.abs(S[1, 0, :]).tolist(),
                      "S01": np.abs(S[0, 1, :]).tolist(), "S11": np.abs(S[1, 1, :]).tolist()},
            "col_coax_driven_power": (np.abs(S[0, 0, :]) ** 2 + np.abs(S[1, 0, :]) ** 2).tolist(),
        }
    return d


def _a0_legacy_for(S, committed):
    Sc = np.asarray(S)[:, :, list(committed)]
    return {
        "s22_abs": np.abs(Sc[1, 1, :]).tolist(),
        "max_abs_s": float(np.max(np.abs(Sc))),
        "col_msl_driven_power": (np.abs(Sc[0, 1, :]) ** 2 + np.abs(Sc[1, 1, :]) ** 2).tolist(),
        "settling_db": [-45.0, -44.0],
        "reciprocity_worst_deviation": {"pair": [0, 1], "value": 0.9},
        "gamma_ratio_coax_driven": [1.1, 0.9, 1.0],
        "cond_a_equilibrated": [1.0, 1.0, 1.0],
    }


def _a0_reference_S():
    rng = np.random.default_rng(589)
    S = np.zeros((2, 2, 3), dtype=complex)
    S[0, 0, :] = 0.99 * np.exp(1j * rng.uniform(0, 6.28, 3))
    S[1, 1, :] = 0.21 * np.exp(1j * rng.uniform(0, 6.28, 3))
    S[1, 0, :] = 3.0e-6 * np.exp(1j * rng.uniform(0, 6.28, 3))
    S[0, 1, :] = 6.0e-5 * np.exp(1j * rng.uniform(0, 6.28, 3))
    return S


def test_589_a0_precision_mismatch_is_not_comparable_and_applies_f64_rule(tmp_path):
    """Review required change 2: an f64 run compared against an f32 baseline
    is NOT a reproduction test (no A0 tier), and the predeclared f64 rule
    (|S10| or |S01| moving > 2x => FLOOR; |S22|/|S00|/col_power within the
    A0 budget) is computed by the driver, not left as prose."""
    import json

    drv = _load_settled_run_driver()
    freqs = np.array([6.0e9, 8.0e9, 10.0e9])
    committed = [0, 1, 2]
    S_base = _a0_reference_S()
    base_path = tmp_path / "f32_baseline.json"
    base_path.write_text(json.dumps(_a0_baseline_dict(
        S=S_base, n_steps=300, fixture="attempt2", x64=False,
        freqs_all=freqs, committed=committed)))

    S_this = S_base.copy()
    S_this[1, 0, 0] *= 3.0
    S_this[0, 1, :] *= 1.5
    S_this[1, 1, :] *= (1.0 + 1.0e-4)
    a0 = drv._a0_compare(base_path, freqs_all=freqs, committed=committed, S=S_this,
                         n_steps=300, fixture="attempt2", legacy=_a0_legacy_for(S_this, committed),
                         x64_enabled=True)
    assert a0["status"] == "compared but NOT a reproduction test", a0["status"]
    assert a0["tier"].startswith("NOT COMPARABLE"), a0["tier"]
    assert "STOP" not in a0["tier"] and "reproduced" not in a0["tier"], a0["tier"]
    assert any("precision" in n for n in a0["notes"]), a0["notes"]
    assert a0["precision"]["this_run_jax_enable_x64"] is True
    assert a0["precision"]["baseline_jax_enable_x64"] is False
    f64 = a0["f64_replicate"]
    assert f64["applicable"] is True and f64["same_fixture_and_steps"] is True
    assert f64["verdict"].startswith("FLOOR"), f64["verdict"]
    assert "6 GHz" in f64["verdict"] and "S10" in f64["verdict"], f64["verdict"]
    assert np.isclose(f64["ratio_S10"][0], 3.0) and np.allclose(f64["ratio_S01"], 1.5)
    assert f64["magnitude_max_delta"] <= 1.0e-3
    assert "within the A0 reproduced tier" in f64["verdict"], f64["verdict"]

    S_this = S_base.copy()
    S_this[1, 1, :] = 0.41 * S_base[1, 1, :] / np.abs(S_base[1, 1, :])
    a0 = drv._a0_compare(base_path, freqs_all=freqs, committed=committed, S=S_this,
                         n_steps=300, fixture="attempt2", legacy=_a0_legacy_for(S_this, committed),
                         x64_enabled=True)
    f64 = a0["f64_replicate"]
    assert f64["verdict"].startswith("|S10|/|S01| within 2x"), f64["verdict"]
    assert "do NOT reproduce within the A0 budget" in f64["verdict"], f64["verdict"]
    assert f64["magnitude_max_delta"] > 1.0e-2
    assert a0["tier"].startswith("NOT COMPARABLE"), a0["tier"]

    tracked_like = tmp_path / "tracked_like.json"
    tracked_like.write_text(json.dumps(_a0_baseline_dict(
        S=S_base, n_steps=300, fixture="attempt2", x64=False,
        freqs_all=freqs, committed=committed, with_ext=False)))
    a0 = drv._a0_compare(tracked_like, freqs_all=freqs, committed=committed, S=S_base,
                         n_steps=300, fixture="attempt2", legacy=_a0_legacy_for(S_base, committed),
                         x64_enabled=True)
    assert a0["tier"].startswith("NOT COMPARABLE") and "precision" in a0["tier"], a0["tier"]
    assert "documented" in a0["precision"]["baseline_source"], a0["precision"]
    f64 = a0["f64_replicate"]
    assert f64["applicable"] is True and f64["ratio_S10"] is None
    assert "NOT computable" in f64["verdict"] and "s_abs" in f64["verdict"], f64["verdict"]
    assert f64["magnitude_max_delta"] == 0.0
    assert set(f64["magnitude_deltas"]) == {"S22", "col_msl_driven_power"}

    a0 = drv._a0_compare(base_path, freqs_all=freqs, committed=committed, S=S_base,
                         n_steps=300, fixture="attempt2", legacy=_a0_legacy_for(S_base, committed),
                         x64_enabled=False)
    assert a0["status"] == "compared" and a0["tier"].startswith("reproduced"), a0
    assert a0["f64_replicate"] is None

    a0 = drv._a0_compare(base_path, freqs_all=freqs, committed=committed, S=S_base,
                         n_steps=135000, fixture="attempt2", legacy=_a0_legacy_for(S_base, committed),
                         x64_enabled=False)
    assert a0["tier"].startswith("NOT COMPARABLE") and "n_steps" in a0["tier"], a0["tier"]
    assert "STOP" not in a0["tier"]
    assert a0["tier_value"] == 0.0 and "max_abs_delta_S_complex" in a0["deltas"]


def test_589_a0_indexes_dense_band_baseline_by_freqs_hz_all(tmp_path):
    """A baseline written with --freqs carries committed-only legacy arrays
    (freqs_hz) but all-bin ext_589 arrays (freqs_hz_all); the comparator must
    index each by its own frequency axis."""
    import json

    drv = _load_settled_run_driver()
    freqs_dense = np.array([5.0e9, 6.0e9, 7.0e9, 8.0e9, 10.0e9])
    committed_dense = [1, 3, 4]
    rng = np.random.default_rng(1)
    S_dense = rng.normal(size=(2, 2, 5)) + 1j * rng.normal(size=(2, 2, 5))
    base_path = tmp_path / "dense_f32.json"
    base_path.write_text(json.dumps(_a0_baseline_dict(
        S=S_dense, n_steps=300, fixture="attempt2", x64=False,
        freqs_all=freqs_dense, committed=committed_dense)))

    freqs = np.array([6.0e9, 8.0e9, 10.0e9])
    S_this = S_dense[:, :, committed_dense]
    a0 = drv._a0_compare(base_path, freqs_all=freqs, committed=[0, 1, 2], S=S_this,
                         n_steps=300, fixture="attempt2",
                         legacy=_a0_legacy_for(S_this, [0, 1, 2]), x64_enabled=False)
    assert a0["status"] == "compared", a0
    assert a0["deltas"]["max_abs_delta_S_complex"] == 0.0, a0["deltas"]
    assert a0["tier"].startswith("reproduced"), a0["tier"]



def _attempt2_kwargs(n_steps):
    return dict(
        junction_x=JUNCTION_X, eps_r_sub=EPS_SUB, n_steps=n_steps,
        freqs=FREQS_2,
        probe_count=6, probe_start_cells=4, probe_spacing_cells=2,
        msl_probe_count=PROBE_COUNT_2, msl_probe_start_cells=PROBE_START_2,
        msl_probe_spacing_cells=PROBE_SPACING_2,
        skip_preflight=True, strict_passivity=False,
    )



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

    tiny = _mon(1.0e-15)
    t64 = np.asarray(flux_spectrum(tiny, exact_f64=True), dtype=np.float64)
    assert np.all(np.isfinite(t64)) and np.any(t64 != 0.0)


PIN_R_CELLS = int(round(PIN_R / DX))
CLEAR_R_CELLS = int(round(CLEAR_R / DX))
SHELL_INNER_CELLS = int(round((OUTER_R - DX) / DX))
OUTER_R_CELLS = int(round(OUTER_R / DX))
N_GROUND_BOXES_3 = 20
HOLE_CELLS_3 = 49
ANNULUS_CELLS_3 = 36
LIP_CELLS_3 = 32
SHELL_CELLS_3 = 32
PIN_FOOTPRINT_CELLS_2 = 11
TRACE_NODE_ROWS_2 = 6
TRACE_NODE_ROWS_2_REALIZED = TRACE_NODE_ROWS_2 + 1
TRACE_W_REALIZED_M = TRACE_NODE_ROWS_2 * DX
HOLE_XOR_CELLS_3 = HOLE_CELLS_3 - PIN_FOOTPRINT_CELLS_2


def _integer_disk_row_half_widths(r_cells):
    """Integer-lattice disk di^2 + dj^2 <= r^2: row dj -> largest |di|."""
    import math
    return {dj: int(math.isqrt(r_cells * r_cells - dj * dj))
            for dj in range(-r_cells, r_cells + 1)}


def _ground_plane_boxes_with_clearance(lx, ly, jx_node, jy_node, k_node, r_cells):
    """PEC ground sheet at z node ``k_node`` tiled by half-cell Boxes so the
    integer-lattice disk di^2 + dj^2 <= r_cells^2 about (jx_node, jy_node)
    is NOT covered (the realized clearance hole).

    2 x-slabs (outer faces on 0.0 / lx exactly as attempt 2's single Box,
    inner faces half a cell outside the hole's x-band), 2 y-strips within
    the x-band, then per row |dj| with half-width w < r_cells two boxes
    filling x beyond w. Every inner face is (n +/- 0.5) * DX (#802).
    """
    from rfx.geometry.csg import Box

    R = r_cells
    z_lo, z_hi = _half_cell_box_z(k_node, k_node)
    rows = _integer_disk_row_half_widths(R)
    boxes = []
    boxes.append(Box((0.0, 0.0, z_lo), (_half_cell_box(jx_node - R - 1, jx_node - R - 1)[1], ly, z_hi)))
    boxes.append(Box((_half_cell_box(jx_node + R + 1, jx_node + R + 1)[0], 0.0, z_lo), (lx, ly, z_hi)))
    xl, xh = _half_cell_box(jx_node - R, jx_node + R)
    boxes.append(Box((xl, 0.0, z_lo), (xh, _half_cell_box(jy_node - R - 1, jy_node - R - 1)[1], z_hi)))
    boxes.append(Box((xl, _half_cell_box(jy_node + R + 1, jy_node + R + 1)[0], z_lo), (xh, ly, z_hi)))
    for dj in sorted(rows):
        w = rows[dj]
        if w >= R:
            continue
        yl, yh = _half_cell_box(jy_node + dj, jy_node + dj)
        boxes.append(Box((xl, yl, z_lo), (_half_cell_box(jx_node - w - 1, jx_node - w - 1)[1], yh, z_hi)))
        boxes.append(Box((_half_cell_box(jx_node + w + 1, jx_node + w + 1)[0], yl, z_lo), (xh, yh, z_hi)))
    return boxes


def _build_coax_msl_transition_sim_attempt3():
    """Attempt 3 (issue #589): attempt 2 with the single ground Box replaced
    by N_GROUND_BOXES_3 half-cell PEC Boxes that leave the predeclared 0.4 mm
    clearance disk OPEN at the junction node. Everything else -- PTFE
    Cylinder, substrate, trace, pin, both ports, domain, materials -- is
    attempt 2's own code path with attempt 2's own constants; compute
    kwargs are ``_attempt2_kwargs`` unchanged.
    """
    from rfx.api import Simulation
    from rfx.boundaries.spec import BoundarySpec
    from rfx.geometry.csg import Box, Cylinder

    sim = Simulation(snap="declared",
        freq_max=FREQ_MAX_2, domain=(LX_2, LY, LZ_2), dx=DX, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml", z="cpml"),
    )
    sim.add_material("sub", eps_r=EPS_SUB)
    sim.add_material("ptfe", eps_r=EPS_COAX)

    jx_node = int(round(JUNCTION_X / DX))
    jy_node = int(round(Y_C / DX))
    ground_boxes = _ground_plane_boxes_with_clearance(
        LX_2, LY, jx_node, jy_node, N_GND, CLEAR_R_CELLS)
    assert len(ground_boxes) == N_GROUND_BOXES_3
    for box in ground_boxes:
        sim.add(box, material="pec")
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
        terminates=sim._geometry[-2].shape,
    )
    sim.add_msl_port(
        position=(FEED_X_2, Y_C, N_SUB_LO * DX), width=W_TRACE, height=H_SUB,
        direction="-x", impedance=50.0, eps_r_sub=EPS_SUB,
    )
    return sim


PREDECLARATION_ATTEMPT3 = {
    "leg": "issue #589 attempt 3 -- realized ground-plane clearance hole (fixture fix for the attempt-1/2 short)",
    "authorization": (
        "R2's escape clause requires 'an identified implementation defect' "
        "in writing: the defect is the fixture's own -- attempts 1 and 2 "
        "declared the 0.4 mm clearance disk as a dielectric Cylinder AFTER "
        "the ground Box, which _assemble_materials (PEC-OR-only) cannot "
        "honour, so the pin was shorted to solid ground at the junction "
        "node (verified by direct inspection of the assembled pec_mask: "
        "36/36 clearance-annulus cells PEC; post-stamp node-25 open "
        "fraction 0.0). This is a construction defect of the same class as "
        "attempt 1's missing-PEC-layer / knife-edge fixes (plumbing, not a "
        "mechanism-hypothesis loop) and is what the settled runs measured "
        "(S00 = -0.9928 at 6 GHz, |S10| = -90 dB)."
    ),
    "geometry_change": (
        "ONLY the ground plane entity: one full-plane PEC Box -> 20 half-"
        "cell PEC Boxes generated from the integer-lattice disk di^2+dj^2 "
        "<= 16 (CLEAR_R_CELLS = 4 = pin + 2 cells, the SAME 0.4 mm clearance "
        "attempt 2 predeclared). pec_mask differs from attempt 2 in exactly "
        "38 cells, all at z node 25 (the 49-cell hole minus the pin's 11-"
        "cell realized footprint); eps_r/sigma arrays np.array_equal; PTFE "
        "Cylinder, substrate, trace, pin, both ports, domain, kwargs "
        "unchanged (asserted by test_attempt3_junction_is_attempt2_plus_"
        "hole_only). The 0.4 < r <= 0.5 mm ring under the coax dielectric "
        "stays PEC (a one-cell ground lip, 32/32) -- part of the "
        "predeclared clearance, flagged as a convention question "
        "(flush 0.5 mm aperture would be a NEW predeclaration)."
    ),
    "instrument_unchanged": (
        "MSL ladder (9 probes x 10 cells from cell 4), coax ladder (6 x 2 "
        "from 4), band {6, 8, 10} GHz, freq_max 16 GHz, domain 12.5 x 3.4 "
        "mm, dx 100 um, n_steps target 135000 (SETTLED_RUN_RECORD): all "
        "attempt 2's. The driver's --fixture attempt3 reuses _attempt2_"
        "kwargs verbatim; _a0_compare labels the run NOT COMPARABLE against "
        "the attempt-2 baseline by design (the baseline is the short)."
    ),
    "realization_contract_tests": [
        "test_attempt2_junction_is_shorted_by_registered_ground",
        "test_junction_pin_axis_is_pec_continuous_stub_to_trace[attempt2|attempt3]",
        "test_junction_coax_shell_contacts_ground[attempt2|attempt3]",
        "test_junction_trace_width_in_cells_is_attempt2s[attempt2|attempt3]",
    ],
    "expected_if_hole_was_the_whole_story": {
        "reasoning": (
            "Transmission-line arithmetic, brackets not points. Coax stub "
            "Z_TEM: the stamped shell realizes shell_inner = 0.5 mm "
            "(shell_thickness = min(dz, (b-a)/2) = 100 um), so the stub lies "
            "between Z_TEM(0.2, 0.6) = 45.46 ohm and Z_TEM(0.2, 0.5) = 37.91 "
            "ohm. Hammerstad-Jensen Z0 of the REALIZED 700 um trace (7 node "
            "rows) on 300 um / eps 3.66 = 48.27 ohm (declared 600 um: 53.11). "
            "Impedance step alone: |Gamma| in [0.030, 0.120] vs the realized "
            "trace ([0.077, 0.167] vs the declared width). Pin through the "
            "300 um substrate: 0.02-0.1 nH -> X_L <= 6 ohm at 10 GHz -> "
            "|Gamma| contribution <= 0.07, rising with frequency. Ground lip "
            "iris (0.4-0.5 mm, 42% node coverage of the coax dielectric): "
            "thin-iris estimate B/Y0 ~ 0.008 at 10 GHz -> |Gamma| ~ 0.004."
        ),
        "s00_abs": "0.05 - 0.25 (-26 to -12 dB) at 6/8/10 GHz, rising with frequency",
        "s10_abs": "0.95 - 1.0 (-0.5 to 0 dB)",
        "passivity": "|S00|^2 + |S10|^2 <= 1 within a few percent leakage",
        "reciprocity": "|S10 - S01| within the attempt-2 f32 replicate spread",
        "msl_ladder_residuals": (
            "fit/recurrence residuals fall from attempt 2's 0.19-0.25 (MSL "
            "drive) / 0.59-0.78 (coax drive) toward the 0.02 class under BOTH "
            "drives"
        ),
        "settling": "settling_db <= -40 dB on both drives at 135000 steps (else #662 warning -> report, do not pin)",
    },
    "falsifiers": {
        "a": "|S00| > 0.9 at all bins -> a second short/open remains (check pin nodes 26-28 and trace start node 11 vs pin nodes 9-12)",
        "b": "|S00| 0.3-0.8 with MSL residuals still ~0.2 -> the hole was necessary but the MSL-side instrument is an independent second layer; do not attribute to the junction",
        "c": "|S00| in band but |S10| far below sqrt(1 - |S00|^2) -> power unseen by both ports (substrate/pad leakage; read the driver's flux monitors)",
    },
    "measurement_command": (
        "cd <run-clone> && PYTHONPATH=$PWD JAX_ENABLE_X64=0 python scripts/"
        "diagnostics/coax_msl_transition_settled_run.py --fixture attempt3 "
        "--n-steps 135000 --preflight --baseline scripts/diagnostics/"
        "_coax_msl_transition_settled_run_logs/remeasure_369367257263_"
        "attempt2_x64-0_result.json --output <run-dir>/attempt3_x64-0_"
        "result.json (VESSL remilab-c0 lane, scripts/vessl_coax_msl_"
        "transition_remeasure.yaml FIXTURE=attempt3)"
    ),
    "preflight_noise_expected": (
        "The 20-Box ground makes fidelity_report 26 entities (repeats of the "
        "attempt-2 ground row's off-lattice-face/sheet-own-cell-live kinds) "
        "and adds ~35 'Gap between PEC structures' + 36 mesh_resolution "
        "advisories (pairs of ring boxes across the hole) -- decomposition "
        "artefacts, not physics; quantify in the record, do not gate."
    ),
    "status": "UNRUN",
    "measured": None,
    "log_path": None,
    "vessl_run_id": None,
}


def test_attempt3_predeclaration_is_committed_unrun_and_self_consistent():
    """Fill-contract invariant, same as SETTLED_RUN_RECORD's: UNRUN <=> no
    numbers, no log path, no run id. The prediction fields hold ranges as
    text, not measured values."""
    r = PREDECLARATION_ATTEMPT3
    required = {
        "leg", "authorization", "geometry_change", "instrument_unchanged",
        "realization_contract_tests", "expected_if_hole_was_the_whole_story",
        "falsifiers", "measurement_command", "preflight_noise_expected",
        "status", "measured", "log_path", "vessl_run_id",
    }
    missing = required - set(r.keys())
    assert not missing, f"PREDECLARATION_ATTEMPT3 missing fields: {missing}"
    assert set(r["falsifiers"]) == {"a", "b", "c"}
    if r["status"] == "UNRUN":
        assert r["measured"] is None
        assert r["log_path"] is None
        assert r["vessl_run_id"] is None
    else:
        assert r["measured"] is not None
        assert r["log_path"] is not None
    names = {n.split("[")[0] for n in r["realization_contract_tests"]}
    missing_tests = {n for n in names if n not in globals()}
    assert not missing_tests, missing_tests


_JUNCTION_FIXTURES = {
    "attempt2": _build_coax_msl_transition_sim_attempt2,
    "attempt3": _build_coax_msl_transition_sim_attempt3,
}


def _realized_node_pec(sim):
    """``(grid, node_pec)`` — ``node_pec[i, j, k]`` True when the lattice
    realizes node ``(i, j)`` as IN-PLANE PEC on z-node-plane ``k``."""
    from tests._realized_geometry import realized

    rz = realized(sim)
    mx = np.asarray(rz.edge_masks[0], dtype=bool)
    my = np.asarray(rz.edge_masks[1], dtype=bool)
    out = mx.copy()
    out[1:, :, :] |= mx[:-1, :, :]
    out |= my
    out[:, 1:, :] |= my[:, :-1, :]
    return rz.grid, out


def _assemble_junction_realized(sim):
    """``_assemble_junction``'s tuple with the REALIZED node-plane PEC in
    place of the primal cell mask: ``(grid, materials, node_pec, i0, j0, kj)``.

    The materials are still the assembled arrays — the coax stub's stamp is a
    sigma write and is read from them exactly as before.
    """
    grid = sim._build_grid()
    materials, _, _, _pec_mask, _, _, _ = sim._assemble_materials(
        grid, pec_sheets=[], pec_wires=[])
    g2, node_pec = _realized_node_pec(sim)
    assert (g2.nx, g2.ny, g2.nz) == (grid.nx, grid.ny, grid.nz)
    port = sim._coaxial_ports[0]
    i0, j0, kj = (int(v) for v in grid.position_to_index(port.position))
    return grid, materials, node_pec, i0, j0, kj


def test_realized_node_pec_reader_agrees_with_the_single_owner():
    """The vectorized reader IS ``realized_wall_planes``' per-column rule.

    A second hand-rolled scan over some mask is exactly the drift the
    single-owner rule (#931 §1.7) exists to stop, so the fast whole-plane
    form is checked against the owner's own ``ij=`` column form on a sample
    that covers the pin axis, the clearance annulus, the ground lip, the
    shell ring and open ground — on both committed junction fixtures.
    """
    from rfx.boundaries.pec import realized_wall_planes
    from tests._realized_geometry import realized

    for name, build in sorted(_JUNCTION_FIXTURES.items()):
        sim = build()
        grid, node_pec = _realized_node_pec(sim)
        edges = realized(sim).edge_masks
        port = sim._coaxial_ports[0]
        i0, j0, kj = (int(v) for v in grid.position_to_index(port.position))
        probes = [(i0, j0), (i0 + PIN_R_CELLS, j0), (i0, j0 + PIN_R_CELLS),
                  (i0 + 3, j0), (i0, j0 + 3), (i0 + CLEAR_R_CELLS, j0),
                  (i0, j0 + CLEAR_R_CELLS), (i0 + SHELL_INNER_CELLS, j0),
                  (i0 + OUTER_R_CELLS, j0), (i0 + 12, j0 + 9), (i0 - 7, j0 - 5)]
        for (i, j) in probes:
            want = kj in realized_wall_planes(edges, 2, ij=(i, j))
            assert bool(node_pec[i, j, kj]) == want, (
                f"{name}: the node-plane reader and realized_wall_planes "
                f"disagree at node ({i}, {j}) on plane {kj}: reader "
                f"{bool(node_pec[i, j, kj])}, owner {want}")



def _assemble_junction(sim):
    """(grid, materials, pec, i0, j0, k_junction) -- what compute_coax_msl_
    transition sees BEFORE its own coax-stub stamp."""
    grid = sim._build_grid()
    materials, _, _, pec_mask, _, _, _ = sim._assemble_materials(grid)
    port = sim._coaxial_ports[0]
    i0, j0, kj = (int(v) for v in grid.position_to_index(port.position))
    return grid, materials, np.asarray(pec_mask), i0, j0, kj


def _lattice_d2(grid, i0, j0):
    """Squared lattice distance (di^2 + dj^2, in cells) of every (i, j)
    node from the pin axis node -- an exact integer, so ring membership
    has no float knife edge (design review blocker 2)."""
    di = np.arange(grid.shape[0]) - i0
    dj = np.arange(grid.shape[1]) - j0
    return di[:, None] ** 2 + dj[None, :] ** 2


def _lattice_ring_mask(grid, i0, j0, r_in_cells, r_out_cells):
    """Cells with r_in < r <= r_out on the integer lattice."""
    d2 = _lattice_d2(grid, i0, j0)
    return (d2 > r_in_cells ** 2) & (d2 <= r_out_cells ** 2)


def _stamp_like_method(sim, grid, materials):
    """Apply compute_coax_msl_transition's own coax-stub stamps at its own
    z indices (z_stub_lo = pad_z_lo + 2, z_stub_hi = z_junction_idx - 1,
    resistor at z_feed = z_stub_lo + 1) -- returns (materials, shell_inner,
    coax_pec_cells, z_stub_lo, z_stub_hi).

    ``coax_pec_cells`` is where the pin and the outer conductor now live:
    ``stamp_coaxial_line`` returns them as a cell mask for
    ``realized_pec_edge_masks`` instead of writing ``sigma = PEC_SIGMA``, so a
    caller that inspects only ``materials.sigma`` no longer sees the conductor
    at all. The resistor is given the same mask, which is how the production
    lane threads it (``rfx/sparams/coax.py``, the
    ``stamp_coaxial_line`` / ``stamp_coaxial_annular_resistor`` pair).
    """
    from rfx.sources.coaxial_port import (
        stamp_coaxial_line, stamp_coaxial_annular_resistor,
        coaxial_tem_characteristic_impedance,
    )
    port = sim._coaxial_ports[0]
    center_xy = (float(port.position[0]), float(port.position[1]))
    z_junction_idx = int(grid.position_to_index(port.position)[2])
    z_stub_lo = int(grid.pad_z_lo) + 2
    z_stub_hi = z_junction_idx - 1
    materials, shell_inner, coax_pec_cells = stamp_coaxial_line(
        grid, materials, center_xy=center_xy, z_lo_index=z_stub_lo,
        z_hi_index=z_stub_hi, pin_radius=PIN_R, outer_radius=OUTER_R,
    )
    materials = stamp_coaxial_annular_resistor(
        grid, materials, center_xy=center_xy, z_index=z_stub_lo + 1,
        pin_radius=PIN_R, outer_radius=OUTER_R,
        target_impedance=float(coaxial_tem_characteristic_impedance(PIN_R, OUTER_R)),
        shell_inner_radius=shell_inner, pec_cell_mask=coax_pec_cells,
    )
    return materials, shell_inner, coax_pec_cells, z_stub_lo, z_stub_hi


def _assert_clearance_annulus_open(sim):
    """DISCRIMINATING assertion: at the junction node every lattice node
    with PIN_R < r <= CLEAR_R (36 nodes, all azimuths) is NOT realized as
    in-plane PEC. Fails on attempt 2 (36/36 PEC), passes on attempt 3 (0/36).

    #931: read on the REALIZED node plane (§1.9), not on the primal cell
    mask; the count and the radii are unchanged.
    """
    grid, _, pec, i0, j0, kj = _assemble_junction_realized(sim)
    annulus = _lattice_ring_mask(grid, i0, j0, PIN_R_CELLS, CLEAR_R_CELLS)
    assert int(annulus.sum()) == ANNULUS_CELLS_3
    n_pec = int((pec[:, :, kj] & annulus).sum())
    assert n_pec == 0, (
        f"clearance annulus PIN_R<r<=CLEAR_R at junction node {kj - grid.pad_z_lo} "
        f"has {n_pec}/{ANNULUS_CELLS_3} registered-PEC cells: the pin is "
        f"shorted to ground (issue #589 root cause)"
    )


def _post_stamp_junction_open_fraction(sim):
    """(node-24 annulus dielectric fraction, node-25 annulus OPEN fraction)
    after the method's own coax-stub stamps -- the DISCRIMINATING post-stamp
    read. 'Open' at the junction node = not registered PEC AND not
    sigma-PEC; 'dielectric' at node 24 = not sigma-PEC AND eps == EPS_COAX."""
    from rfx.sources.coaxial_port import PEC_SIGMA
    grid, materials, pec, i0, j0, kj = _assemble_junction_realized(sim)
    materials, _, _, _, z_stub_hi = _stamp_like_method(sim, grid, materials)
    assert z_stub_hi == kj - 1
    sigma = np.asarray(materials.sigma)
    eps = np.asarray(materials.eps_r)
    annulus = _lattice_ring_mask(grid, i0, j0, PIN_R_CELLS, CLEAR_R_CELLS)
    below = ((sigma[:, :, kj - 1][annulus] < 0.5 * PEC_SIGMA)
             & np.isclose(eps[:, :, kj - 1][annulus], EPS_COAX))
    at = ((~pec[:, :, kj][annulus]) & (sigma[:, :, kj][annulus] < 0.5 * PEC_SIGMA))
    return float(np.mean(below)), float(np.mean(at))


def _assert_method_stamp_leaves_junction_open(sim):
    frac_below, frac_at = _post_stamp_junction_open_fraction(sim)
    assert frac_below == 1.0, (
        f"node below the junction: annulus dielectric fraction {frac_below} != 1.0"
    )
    assert frac_at == 1.0, (
        f"junction node: clearance-annulus OPEN fraction after the method's "
        f"stamp is {frac_at} (1.0 = open launch; 0.0 = the #589 short)"
    )


def _node25_hole_window(sim):
    """(open_window, disk_window, pin_footprint_window) -- (2R+1)^2 boolean
    windows about the axis at the junction node, R = CLEAR_R_CELLS.

    #931: ``open_w`` is the complement of the REALIZED in-plane PEC nodes on
    the junction node plane, not of the primal cell mask.
    """
    grid, _, pec, i0, j0, kj = _assemble_junction_realized(sim)
    R = CLEAR_R_CELLS
    sl = (slice(i0 - R, i0 + R + 1), slice(j0 - R, j0 + R + 1))
    open_w = ~pec[:, :, kj][sl]
    disk_w = (_lattice_d2(grid, i0, j0) <= R * R)[sl]
    pin_entity = [e for e in sim._geometry
                  if e.material_name == "pec" and type(e.shape).__name__ == "Cylinder"]
    assert len(pin_entity) == 1
    pin_w = np.asarray(pin_entity[0].shape.mask(grid))[:, :, kj][sl]
    ptfe_entity = [e for e in sim._geometry if e.material_name == "ptfe"]
    assert len(ptfe_entity) == 1
    ptfe_w = np.asarray(ptfe_entity[0].shape.mask(grid))[:, :, kj][sl]
    return open_w, disk_w, pin_w, ptfe_w


def _assert_hole_matches_integer_disk(sim):
    """The 49-node disk is the integer-lattice set {(di, dj): di^2 + dj^2 <= 16}."""
    open_w, disk_w, pin_w, ptfe_w = _node25_hole_window(sim)
    assert int(disk_w.sum()) == HOLE_CELLS_3
    assert int(pin_w.sum()) == PIN_FOOTPRINT_CELLS_2
    assert np.all(pin_w <= disk_w)
    expected = disk_w & ~pin_w
    assert int(expected.sum()) == HOLE_XOR_CELLS_3
    assert np.array_equal(open_w, expected), (
        f"open cells at the junction node in the {2 * CLEAR_R_CELLS + 1}x"
        f"{2 * CLEAR_R_CELLS + 1} window: {int(open_w.sum())}, expected the "
        f"integer disk minus the pin footprint ({HOLE_XOR_CELLS_3})"
    )
    assert int((ptfe_w & ~disk_w).sum()) == 0
    assert 0 < int(ptfe_w.sum()) <= HOLE_CELLS_3



def test_attempt2_junction_is_shorted_by_registered_ground():
    """REGRESSION WITNESS on the COMMITTED attempt-2 builder (never edited):
    the fail-before-fix evidence for the three discriminating assertions
    above, committed as an artifact. Issue #589 root cause: the ground Box
    is a solid PEC sheet at the junction node; the later ``ptfe`` Cylinder
    cannot carve it (_assemble_materials is PEC-OR-only), so
      * 36/36 cells with PIN_R < r <= CLEAR_R are registered PEC,
      * 68/68 cells with PIN_R < r <= shell_inner (0.5 mm) are PEC,
      * after the method's own stamp the junction-node annulus is 0.0 open
        (the node below is 1.0 coax dielectric -- the stamp stops at
        z_junction_idx - 1, the sheet is the termination),
      * the 9x9 window has NO open cell,
    i.e. the coax stub is terminated in a short by REGISTERED geometry --
    what VESSL 369367257263 measured (S00 = -0.9928 at 6 GHz). If this test
    ever fails, attempt 2 was edited -- forbidden; attempt 2 is history."""
    sim2 = _build_coax_msl_transition_sim_attempt2()
    grid, _, pec, i0, j0, kj = _assemble_junction_realized(sim2)
    annulus = _lattice_ring_mask(grid, i0, j0, PIN_R_CELLS, CLEAR_R_CELLS)
    wide = _lattice_ring_mask(grid, i0, j0, PIN_R_CELLS, SHELL_INNER_CELLS)
    assert int(annulus.sum()) == ANNULUS_CELLS_3
    assert int(wide.sum()) == ANNULUS_CELLS_3 + LIP_CELLS_3
    assert int((pec[:, :, kj] & annulus).sum()) == ANNULUS_CELLS_3
    assert int((pec[:, :, kj] & wide).sum()) == ANNULUS_CELLS_3 + LIP_CELLS_3

    with pytest.raises(AssertionError, match="shorted to ground"):
        _assert_clearance_annulus_open(sim2)

    frac_below, frac_at = _post_stamp_junction_open_fraction(sim2)
    assert frac_below == 1.0
    assert frac_at == 0.0
    with pytest.raises(AssertionError, match="OPEN fraction"):
        _assert_method_stamp_leaves_junction_open(sim2)

    open_w, disk_w, pin_w, _ = _node25_hole_window(sim2)
    assert int(open_w.sum()) == 0
    assert int(disk_w.sum()) == HOLE_CELLS_3 and int(pin_w.sum()) == PIN_FOOTPRINT_CELLS_2
    with pytest.raises(AssertionError):
        _assert_hole_matches_integer_disk(sim2)


@pytest.mark.parametrize("fixture", sorted(_JUNCTION_FIXTURES))
def test_junction_pin_axis_is_pec_continuous_stub_to_trace(fixture):
    """INVARIANT (did-not-over-carve, pin side): registered pec_mask OR the
    coax stamper's own conductor cells OR stamped sigma >= PEC_SIGMA/2 on the
    axis for every z in [z_stub_lo, trace node], AND the axis cell at the
    junction node is REGISTERED PEC (the pin). Passes on attempt 2 too --
    through solid ground, so it does not distinguish pin from ground.
    The attempt-3 and wide-trace tests were removed with the deprecation.

    The coax cells are a third term because ``stamp_coaxial_line`` returns the
    pin and the outer conductor as a mask for ``realized_pec_edge_masks``
    instead of writing them into ``materials.sigma``; reading only ``sigma``
    here would look for the pin where it no longer is and report the whole
    stub as a gap. The invariant is unchanged and still holds -- measured
    continuous on both fixtures -- only the array it is read from moved.
    """
    from rfx.sources.coaxial_port import PEC_SIGMA
    sim = _JUNCTION_FIXTURES[fixture]()
    grid, materials, pec, i0, j0, kj = _assemble_junction_realized(sim)
    materials, _, coax_cells, z_stub_lo, _ = _stamp_like_method(
        sim, grid, materials)
    sigma = np.asarray(materials.sigma)
    trace_idx = int(grid.position_to_index((JUNCTION_X, Y_C, N_TRACE * DX))[2])
    column = (pec[i0, j0, :]
              | np.asarray(coax_cells)[i0, j0, :]
              | (sigma[i0, j0, :] >= 0.5 * PEC_SIGMA))
    span = column[z_stub_lo:trace_idx + 1]
    assert np.all(span), f"gap at indices {z_stub_lo + np.where(~span)[0]}"
    assert bool(pec[i0, j0, kj]), "axis cell at the junction node is not registered PEC (pin)"


@pytest.mark.parametrize("fixture", sorted(_JUNCTION_FIXTURES))
def test_junction_coax_shell_contacts_ground(fixture):
    """INVARIANT (did-not-over-carve, shell side): the realized outer
    conductor's footprint lands entirely on PEC ground at the junction node.
    The wall starts at the declared OUTER_R and grows outward, clamped by
    absorber clearance. The geometric ring between CLEAR_R_CELLS and
    SHELL_INNER_CELLS also remains PEC, forming the ground lip over the
    coax dielectric. The attempt-3 and wide-trace tests were removed with the deprecation.
    The ground lip contains 32 lattice nodes with 16 < di^2 + dj^2 <= 25."""
    sim = _JUNCTION_FIXTURES[fixture]()
    grid, materials, pec, i0, j0, kj = _assemble_junction_realized(sim)
    _, shell_inner, coax_cells, z_stub_lo, z_stub_hi = _stamp_like_method(
        sim, grid, materials)
    assert shell_inner == pytest.approx(OUTER_R, rel=0.0, abs=1e-15), (
        f"wall inner face {shell_inner*1e6:.2f} um is not the declared outer "
        f"radius {OUTER_R*1e6:.2f} um")

    dx_m = float(grid.dx)
    ii, jj = np.meshgrid(np.arange(coax_cells.shape[0]),
                         np.arange(coax_cells.shape[1]), indexing="ij")
    radius = np.hypot((ii - i0) * dx_m, (jj - j0) * dx_m)
    z_mid = (z_stub_lo + z_stub_hi) // 2
    shell = np.asarray(coax_cells)[:, :, z_mid] & (radius >= OUTER_R - 0.5 * dx_m)
    n_wall = int(shell.sum())
    assert n_wall > 0, "the stamper realized no outer conductor at all"
    n_shell = int((pec[:, :, kj] & shell).sum())
    assert n_shell == n_wall, (
        f"the outer conductor lands on ground at {n_shell}/{n_wall} of its "
        f"footprint cells ({100.0 * n_shell / n_wall:.1f} %); the rest overhang "
        f"the registered ground plane. Wall radii "
        f"{radius[shell].min()*1e6:.0f}-{radius[shell].max()*1e6:.0f} um against "
        f"a declared outer radius of {OUTER_R*1e6:.0f} um.")

    lip = _lattice_ring_mask(grid, i0, j0, CLEAR_R_CELLS, SHELL_INNER_CELLS)
    assert int(lip.sum()) == LIP_CELLS_3
    n_lip = int((pec[:, :, kj] & lip).sum())
    assert n_lip == LIP_CELLS_3, f"ground lip CLEAR_R<r<=shell_inner {n_lip}/{LIP_CELLS_3}"


@pytest.mark.parametrize("fixture", sorted(_JUNCTION_FIXTURES))
def test_junction_trace_width_in_cells_is_attempt2s(fixture):
    """INVARIANT (MSL side untouched): the contiguous realized in-plane-PEC
    node run across the trace at the trace node plane, 5 cells down the
    trace from the junction, is the same on both fixtures; the trace Box is
    the same code path in both.

    #931: the run is read on the REALIZED node plane (§1.9).  The trace Box's
    y span [Y_C - 300 um, Y_C + 300 um] is nodes 14..20, so the realized
    footprint is a CLOSED run of 7 nodes with 6 cells between them and a
    realized width of (7 - 1) x DX = 600.00 um -- the declared width, exactly.
    The pre-#834 float32 path realized 7 CELLS = 700 um (RASTERIZER NOTE and
    KNIFE-EDGE NOTE above); the exact-coordinate path realized 6 cells; the
    contract adds the closing rim node that bounds them.  The width in metres
    is the invariant and is asserted directly, so no reading of "how many
    indices" can drift it.
    """
    sim = _JUNCTION_FIXTURES[fixture]()
    grid, _, pec, i0, j0, kj = _assemble_junction_realized(sim)
    kt = int(grid.position_to_index((JUNCTION_X, Y_C, N_TRACE * DX))[2])
    row = pec[i0 + 5, :, kt]
    assert row[j0]
    lo = j0
    while lo - 1 >= 0 and row[lo - 1]:
        lo -= 1
    hi = j0
    while hi + 1 < row.size and row[hi + 1]:
        hi += 1
    assert hi - lo + 1 == TRACE_NODE_ROWS_2_REALIZED, (lo, hi, int(row.sum()))
    assert int(row.sum()) == TRACE_NODE_ROWS_2_REALIZED
    assert (hi - lo) * DX == pytest.approx(TRACE_W_REALIZED_M, rel=1e-12)
    assert TRACE_W_REALIZED_M == pytest.approx(W_TRACE, rel=1e-12), (
        "the realized trace width is the declared one; if this fails the "
        "fixture is no longer building the board it says it builds")


def test_attempt3_builder_entity_inventory_and_kwargs():
    """The attempt-3 builder registers exactly N_GROUND_BOXES_3 ground Boxes
    (+ the same 4 attempt-2 entities) and the driver reuses _attempt2_kwargs
    unchanged for it (no attempt-3 kwargs symbol exists by design)."""
    sim = _build_coax_msl_transition_sim_attempt3()
    assert len(sim._geometry) == N_GROUND_BOXES_3 + 4
    assert sum(1 for e in sim._geometry
               if e.material_name == "pec" and type(e.shape).__name__ == "Box") == N_GROUND_BOXES_3 + 1
    assert "_attempt3_kwargs" not in globals()
    kw = _attempt2_kwargs(135000)
    assert kw["junction_x"] == JUNCTION_X and kw["n_steps"] == 135000


FLUX_BOX_X_3 = (0.5e-3, 2.0e-3)
FLUX_BOX_Y_3 = (0.3e-3, 3.1e-3)
FLUX_BOX_Z_3 = (N_GND * DX, 3.6e-3)
FLUX_COAX_Z_3 = 2.2e-3
FLUX_COAX_PATCH_3 = 1.6e-3


def _attempt3_scratch_flux_entries():
    """Flux entries for the attempt-3 W4 box, built the documented way: a
    scratch Simulation sharing the attempt-3 domain, handing over its
    ``._flux_monitors`` list (same pattern as
    the attempt-2 scratch monitors; attempt 3 shares attempt 2's domain
    and dx, only the ground-plane entity differs).

    Report-only instrument: these monitors are passed through
    ``compute_coax_msl_transition(extra_flux_monitors=...)``, whose
    non-perturbation witness was retired with the deprecation.
    The +axis-positive closure convention is coax_z22 = msl_x20 + top_z36 - xlo_x05 - ylo_y03 + yhi_y31.
    """
    from rfx.api import Simulation as _Sim
    scratch = _Sim(freq_max=FREQ_MAX_2, domain=(LX_2, LY, LZ_2), dx=DX,
                   boundary="cpml")
    (x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi) = FLUX_BOX_X_3, FLUX_BOX_Y_3, FLUX_BOX_Z_3
    x_c, x_s = 0.5 * (x_lo + x_hi), x_hi - x_lo
    y_c, y_s = 0.5 * (y_lo + y_hi), y_hi - y_lo
    z_c, z_s = 0.5 * (z_lo + z_hi), z_hi - z_lo

    assert (5 + _attempt2_kwargs(0)["probe_start_cells"] + (_attempt2_kwargs(0)["probe_count"] - 1) * _attempt2_kwargs(0)["probe_spacing_cells"]) * DX < FLUX_COAX_Z_3 < JUNCTION_Z, (
        'The bottom flux face must lie above the highest coax probe and below its junction reference plane for the pencil comparison.'
    )

    assert 3 * DX < FLUX_COAX_Z_3 and FEED_X_2 > x_hi, (
        'The coax resistor and MSL loss sheet must remain outside the flux volume so its residual excludes dissipation.'
    )

    assert max(JUNCTION_X + CLEAR_R, 2 * JUNCTION_X - DX) < x_hi < float(np.min(_msl_ladder_x_coords(_build_coax_msl_transition_sim_attempt3(), count=PROBE_COUNT_2, start_cells=PROBE_START_2, spacing_cells=PROBE_SPACING_2))), (
        'The positive-x flux face must clear the junction footprint and precede the first realized MSL probe for their power comparison.'
    )

    assert FLUX_COAX_PATCH_3 >= 2 * OUTER_R, (
        'The coax flux patch must cover the entire dielectric cross-section carrying the coax power.'
    )

    scratch.add_flux_monitor(
        axis="z", coordinate=FLUX_COAX_Z_3, freqs=FREQS_2,
        size=(FLUX_COAX_PATCH_3, FLUX_COAX_PATCH_3), center=(JUNCTION_X, Y_C),
        name="coax_z22",
    )
    scratch.add_flux_monitor(
        axis="x", coordinate=x_hi, freqs=FREQS_2, size=(y_s, z_s),
        center=(y_c, z_c), name="msl_x20",
    )
    scratch.add_flux_monitor(
        axis="z", coordinate=z_hi, freqs=FREQS_2, size=(x_s, y_s),
        center=(x_c, y_c), name="top_z36",
    )
    scratch.add_flux_monitor(
        axis="x", coordinate=x_lo, freqs=FREQS_2, size=(y_s, z_s),
        center=(y_c, z_c), name="xlo_x05",
    )
    scratch.add_flux_monitor(
        axis="y", coordinate=y_lo, freqs=FREQS_2, size=(x_s, z_s),
        center=(x_c, z_c), name="ylo_y03",
    )
    scratch.add_flux_monitor(
        axis="y", coordinate=y_hi, freqs=FREQS_2, size=(x_s, z_s),
        center=(x_c, z_c), name="yhi_y31",
    )
    scratch.add_flux_monitor(
        axis="x", coordinate=x_hi, freqs=FREQS_2, name="msl_x20_full",
    )
    return scratch._flux_monitors


MSL_STANDOFF_H_SUB_MULTIPLE = 5.0
PROBE_COUNT_3B = 8
PROBE_START_3B = 15
PROBE_SPACING_3B = 10


def _msl_near_field_standoff_cells(dx, h_sub):
    """Minimum probe standoff in CELLS from an MSL launch/reference plane.

    Delegates to the PRODUCTION predicate,
    ``rfx.api._preflight.msl_source_near_field_standoff_cells`` -- the one
    the preflight advisory and ``compute_coax_msl_transition``'s own
    realized-ladder advisory call. This fixture must never carry a second
    copy of the rule: a test that re-implements the threshold it is
    checking cannot notice the day the two disagree. The equality with
    ``max(3, round(5*h_sub/dx))`` is asserted where it matters, in
    ``test_attempt3b_ladder_satisfies_the_standoff_at_both_ends``.
    The shipped 5*h carries 24% headroom above the earlier derived 4.0354*h floor.
    """
    from rfx.api._preflight import msl_source_near_field_standoff_cells

    return int(msl_source_near_field_standoff_cells(float(h_sub), float(dx)))


def _msl_ladder_x_coords(sim, *, count, start_cells, spacing_cells):
    """Realized MSL probe x-coordinates, ASCENDING -- built exactly the way
    ``compute_coax_msl_transition`` builds them (``msl_probe_x_coords_n`` on
    an ``MSLPort`` reconstructed from the registered entry, then sorted; see
    the ``xs_sorted`` step in ``rfx/api/_sparams.py``). Measured, never
    arithmetic on the recipe: the whole point of #823 is that the recipe and
    the realization were never compared.
    """
    from rfx.sources.msl_port import MSLPort, msl_probe_x_coords_n

    grid = sim._build_grid()
    pe = sim._msl_ports[0]
    x_feed, y_centre, z_lo = (float(c) for c in pe.position)
    port = MSLPort(
        feed_x=x_feed,
        y_lo=y_centre - pe.width / 2, y_hi=y_centre + pe.width / 2,
        z_lo=z_lo, z_hi=z_lo + pe.height,
        direction=pe.direction, impedance=pe.impedance, excitation=None,
    )
    return np.array(sorted(msl_probe_x_coords_n(
        grid, port, n_probes=count,
        n_offset_cells=start_cells, n_spacing_cells=spacing_cells,
    )), dtype=np.float64)


def _msl_standoff_violations(xs, *, feed_x, ref_x, dx, h_sub):
    """(n_violating_at_port, n_violating_at_reference_plane, min_d_port,
    min_d_ref, required_m) for a realized ladder."""
    required_m = _msl_near_field_standoff_cells(dx, h_sub) * dx
    d_port = np.abs(np.asarray(xs, dtype=np.float64) - float(feed_x))
    d_ref = np.abs(np.asarray(xs, dtype=np.float64) - float(ref_x))
    tol = 1e-12
    return (
        int(np.sum(d_port < required_m - tol)),
        int(np.sum(d_ref < required_m - tol)),
        float(d_port.min()), float(d_ref.min()), float(required_m),
    )


def _build_coax_msl_transition_sim_attempt3b():
    """Attempt 3b: attempt 3's fixture, UNCHANGED.

    Attempt 3b differs from attempt 3 in the ``msl_probe_*`` METHOD KWARGS
    only (``_attempt3b_kwargs``), so the fixture builder is attempt 3's own
    builder and the geometry is identical by CONSTRUCTION, not by copy. This
    wrapper exists so the driver's ``--fixture attempt3b`` names a builder of
    its own rather than silently reusing another label's, and so the identity
    is asserted rather than claimed -- see
    ``test_attempt3b_geometry_is_attempt3s_and_only_the_msl_probe_kwargs_differ``,
    which compares eps_r / sigma / mu_r / pec_mask with ``np.array_equal``
    over the WHOLE assembled domain.
    """
    return _build_coax_msl_transition_sim_attempt3()


def _attempt3b_kwargs(n_steps):
    """Attempt 3's compute kwargs (= ``_attempt2_kwargs``) with the three
    ``msl_probe_*`` values replaced by the compliant ladder. Nothing else."""
    kw = _attempt2_kwargs(n_steps)
    kw["msl_probe_count"] = PROBE_COUNT_3B
    kw["msl_probe_start_cells"] = PROBE_START_3B
    kw["msl_probe_spacing_cells"] = PROBE_SPACING_3B
    return kw


PREDECLARATION_ATTEMPT3B = {
    "leg": (
        "issue #823 attempt 3b -- attempt 3's geometry with a COMPLIANT MSL "
        "probe ladder (instrument change only)"
    ),
    "issue": "#823 (ladder), #822 (wave-role labels), #589 (lane tracker)",
    "fixture": (
        "attempt 3's, UNCHANGED (_build_coax_msl_transition_sim_attempt3b "
        "returns _build_coax_msl_transition_sim_attempt3()'s Simulation; "
        "eps_r/sigma/mu_r/pec_mask np.array_equal over the whole domain, "
        "asserted by test_attempt3b_geometry_is_attempt3s_and_only_the_msl_"
        "probe_kwargs_differ)"
    ),
    "this_is_a_new_predeclaration": (
        "THIS IS A NEW PREDECLARATION FOR A NEW FIXTURE LADDER, NOT A "
        "LOOSENED ONE. PREDECLARATION_ATTEMPT2, PREDECLARATION_ATTEMPT3 and "
        "SETTLED_RUN_RECORD are NOT edited, NOT relaxed and NOT reinterpreted "
        "by this work; they remain the committed record of what was predicted "
        "and measured for the ladders they describe. Attempt 3b is a "
        "different instrument, so it carries its own falsifiers and its own "
        "measured=None."
    ),
    "instrument_changes": {
        "wave_roles": (
            "#822 side-aware incident/outgoing mapping (a_inc = forward_amp "
            "at BOTH ports on this lane, because both reference planes sit at "
            "the junction, on the DUT side of their probes). Attempt 3's "
            "committed numbers were taken with the inverted mapping."
        ),
        "msl_ladder": (
            "msl_probe_count 9 -> 8, msl_probe_start_cells 4 -> 15, "
            "msl_probe_spacing_cells 10 (unchanged). Realized probes at "
            "x = 2.5 ... 9.5 mm; every probe at least 5*h_sub = 1.5 mm from "
            "BOTH the port feed plane (x = 11.0 mm) and the reference plane "
            "(the junction, x = 1.0 mm). At the inherited spacing of 10 cells "
            "this is the UNIQUE MAXIMAL-COUNT compliant ladder: the port end "
            "requires start >= 15 and the junction end requires "
            "start + 10*(count-1) <= 85, so count=9 is impossible and count=8 "
            "forces start=15 exactly. (Shorter ladders are also compliant -- "
            "count=7 with start in [15, 25] -- and are rejected only for "
            "carrying less information, not for violating the rule.) NO FREE "
            "PARAMETER WAS CHOSEN."
        ),
        "witness": (
            "ladder_split_gamma_dev / ladder_split_reflection_decades "
            "(disjoint-half refit of each ladder; report-only, new in this PR; "
            "opt-in with return_ladder_voltages=True, which the settled-run "
            "driver sets under --dump-ladders)."
        ),
        "advisory": (
            "the msl_port_geometry near-field standoff advisory must NOT fire "
            "on this ladder (it fires on attempts 1/2/3)."
        ),
    },
    "prediction_provenance": (
        "READ THIS BEFORE READING A FALSIFIER HIT. Every number in "
        "'predictions' below is read off the COMMITTED attempt-3 ladder dump "
        "(witnesses_369367257533_attempt3_x64-0_ladders.npz) over its clean "
        "8-probe window msl[0:8] = x 2.6 ... 9.6 mm. Attempt 3b's realized "
        "probes are x 2.5 ... 9.5 mm -- the SAME recipe spacing on DIFFERENT "
        "grid cells, one cell offset, because the 5*h_sub rule forces "
        "start=15 rather than the start=14 that would have made 3b a bit-exact "
        "subset of attempt 3's dumped probes. The predictions are therefore a "
        "ONE-CELL-SHIFTED PROXY, not an exact offline precomputation: a near-"
        "miss on F1 or F2 must be read against that shift before it is called "
        "a falsifier hit."
    ),
    "predictions": {
        "msl_fit_residual": (
            "<= 5e-3 at every committed bin under BOTH drives (attempt 3 "
            "measured 0.342 / 0.264 / 0.222 with the near-field probe "
            "included; the committed dump's clean 8-probe window measures "
            "3.7e-3 / 2.5e-3 / 1.8e-3)."
        ),
        "beta_over_beta_hj": (
            "0.86-0.88, flat across the band (declared-vs-realized trace "
            "width, #722) -- NOT 1.0."
        ),
        "abs_gamma_msl_junction": (
            "0.258 / 0.292 / 0.313 +/- 0.02 at 6/8/10 GHz (the clean-subset "
            "reading; stable to +/-0.003 across every window of the committed "
            "dump that excludes the near-field probe)."
        ),
        "abs_gamma_coax_junction": (
            "~0.93 (0.931 / 0.950 / 0.984 read off the corrected labels on "
            "the committed attempt-3 coax ladder), carrying a 10-18% "
            "alpha/referral uncertainty at 8/10 GHz -- quoted to two figures, "
            "with no reciprocity or passivity claim built on it."
        ),
        "ladder_split_reflection_decades": (
            "<= 0.05 on the MSL port array under both drives (attempt 3's own "
            "disjoint halves read 4.32-4.49)."
        ),
        "passivity": (
            "NOT predicted to pass. The label fix and the ladder fix are "
            "necessary, not sufficient: the coax ladder cannot identify alpha "
            "over its 1 mm span and the label-blind Z0 calibration ratio reads "
            "0.650 / 0.965 / 4.051 against a +/-10% band (#589). The clean-"
            "window proxy predicts lambda_min(I - S^H S) ~ -1.9 and an MSL-"
            "driven column power ~ 2.9. A passive result would be a SURPRISE "
            "and must be re-derived, not celebrated."
        ),
    },
    "falsifiers_predeclared": {
        "F1_ladder_not_clean": (
            "msl fit_residual > 5e-3 at any committed bin => the standoff rule "
            "is not the whole story; do not proceed to re-read the record."
        ),
        "F2_gamma_moved": (
            "|Gamma_msl| outside 0.258 / 0.292 / 0.313 +/- 0.02 => the clean-"
            "subset reading was window-dependent after all; #823's central "
            "claim fails."
        ),
        "F3_witness_disagrees": (
            "ladder_split_reflection_decades > 0.05 on a ladder that satisfies "
            "the standoff => the witness and the rule disagree; the rule is "
            "wrong, not the run."
        ),
        "F4_advisory_fires": (
            "the near-field advisory fires on this ladder => predicate and "
            "fixture disagree; a code defect, fix before quoting anything."
        ),
        "F5_beta_at_unity": (
            "beta/beta_HJ within 2% of 1.0 => the declared-vs-realized width "
            "attribution (#722) is wrong and must be retracted."
        ),
    },
    "realization_contract_tests": [
        "test_attempt3b_geometry_is_attempt3s_and_only_the_msl_probe_kwargs_differ",
        "test_attempt3b_ladder_satisfies_the_standoff_at_both_ends",
        "test_attempt3b_ladder_is_the_unique_maximal_count_compliant_ladder",
        "test_attempt3b_predeclaration_is_committed_unrun_and_self_consistent",
        "test_settled_run_driver_fixture_selection_is_explicit_for_every_label",
    ],
    "how_to_run": (
        "cd <run-clone> && PYTHONPATH=$PWD JAX_ENABLE_X64=0 python scripts/"
        "diagnostics/coax_msl_transition_settled_run.py --fixture attempt3b "
        "--n-steps 135000 --preflight --dump-ladders --flux --output "
        "<run-dir>/attempt3b_x64-0_result.json (VESSL remilab-c0, scripts/"
        "vessl_coax_msl_transition_remeasure.yaml FIXTURE=attempt3b "
        "DUMP_LADDERS=1 FLUX=1)"
    ),
    "post_review_correction": None,
    "status": "UNRUN",
    "measured": None,
    "log_path": None,
    "vessl_run_id": None,
}


def test_attempt3b_predeclaration_is_committed_unrun_and_self_consistent():
    """Fill-contract invariant, the same one SETTLED_RUN_RECORD and
    PREDECLARATION_ATTEMPT3 carry: UNRUN <=> no measured numbers, no log
    path, no run id; the prediction fields hold ranges as TEXT, not values;
    the falsifier key set is frozen; every named realization-contract test
    exists in this module.

    This is a NEW predeclaration. It does not, and must not, touch
    PREDECLARATION_ATTEMPT2/3 or SETTLED_RUN_RECORD -- asserted below by
    re-checking that those three are still UNRUN-or-filled on their own
    terms, so a future edit that "harmonises" them trips this test.
    """
    r = PREDECLARATION_ATTEMPT3B
    required = {
        "leg", "issue", "fixture", "this_is_a_new_predeclaration",
        "instrument_changes", "prediction_provenance", "predictions",
        "falsifiers_predeclared", "realization_contract_tests", "how_to_run",
        "post_review_correction", "status", "measured", "log_path",
        "vessl_run_id",
    }
    missing = required - set(r.keys())
    assert not missing, f"PREDECLARATION_ATTEMPT3B missing fields: {missing}"
    assert set(r["falsifiers_predeclared"]) == {
        "F1_ladder_not_clean", "F2_gamma_moved", "F3_witness_disagrees",
        "F4_advisory_fires", "F5_beta_at_unity",
    }
    assert set(r["instrument_changes"]) == {
        "wave_roles", "msl_ladder", "witness", "advisory",
    }
    assert set(r["predictions"]) == {
        "msl_fit_residual", "beta_over_beta_hj", "abs_gamma_msl_junction",
        "abs_gamma_coax_junction", "ladder_split_reflection_decades",
        "passivity",
    }
    if r["status"] == "UNRUN":
        assert r["measured"] is None
        assert r["log_path"] is None
        assert r["vessl_run_id"] is None
    else:
        assert r["measured"] is not None
        assert r["log_path"] is not None
    for key, val in r["predictions"].items():
        assert isinstance(val, str), (key, type(val))
    assert "NOT A LOOSENED ONE" in r["this_is_a_new_predeclaration"]
    assert "ONE-CELL-SHIFTED PROXY" in r["prediction_provenance"]
    names = {n.split("[")[0] for n in r["realization_contract_tests"]}
    missing_tests = {n for n in names if n not in globals()}
    assert not missing_tests, missing_tests
    assert SETTLED_RUN_RECORD["status"] == "RUN"
    assert SETTLED_RUN_RECORD["measured"] is not None
    assert PREDECLARATION_ATTEMPT2["status"] == "RUN"
    assert PREDECLARATION_ATTEMPT2["measured"] is not None
    assert PREDECLARATION_ATTEMPT3["status"] == "UNRUN"
    assert PREDECLARATION_ATTEMPT3["measured"] is None


def test_attempt3b_geometry_is_attempt3s_and_only_the_msl_probe_kwargs_differ():
    """The 'instrument change only' claim, ASSERTED not claimed.

    (a) Every assembled material array (eps_r, sigma, mu_r) AND the PEC mask
        are ``np.array_equal`` over the WHOLE domain between the attempt-3
        and attempt-3b builders -- not a window, the whole grid.
    (b) The registered coax and MSL port parameters are identical field by
        field, including the MSL feed x (the ladder's own anchor): attempt 3b
        moves the PROBES, never the PORT.
    (c) ``_attempt3b_kwargs`` differs from attempt 3's own kwargs
        (``_attempt2_kwargs``, which the driver reuses verbatim for
        attempt 3) in EXACTLY the three ``msl_probe_*`` keys.
    The spacing is inherited from attempt 2/3 so that the pencil conditioning is the instrument the earlier record characterised.
    """
    sim3 = _build_coax_msl_transition_sim_attempt3()
    sim3b = _build_coax_msl_transition_sim_attempt3b()

    g3, g3b = sim3._build_grid(), sim3b._build_grid()
    assert (g3.nx, g3.ny, g3.nz) == (g3b.nx, g3b.ny, g3b.nz)
    out3, out3b = sim3._assemble_materials(g3), sim3b._assemble_materials(g3b)
    mats3, pec3 = out3[0], out3[3]
    mats3b, pec3b = out3b[0], out3b[3]
    for name in ("eps_r", "sigma", "mu_r"):
        a = np.asarray(getattr(mats3, name))
        b = np.asarray(getattr(mats3b, name))
        assert a.dtype == b.dtype and a.shape == b.shape
        assert np.array_equal(a, b), (
            f"{name}: {int(np.sum(a != b))} cells differ between attempt 3 "
            "and attempt 3b -- attempt 3b is an INSTRUMENT change only"
        )
    assert pec3 is not None and pec3b is not None
    pa = np.asarray(pec3, dtype=bool)
    pb = np.asarray(pec3b, dtype=bool)
    assert np.array_equal(pa, pb), (
        f"pec_mask: {int(np.sum(pa != pb))} cells differ between attempt 3 "
        "and attempt 3b"
    )

    c3, c3b = sim3._coaxial_ports[0], sim3b._coaxial_ports[0]
    assert c3.pin_radius == c3b.pin_radius == PIN_R
    assert c3.outer_radius == c3b.outer_radius == OUTER_R
    assert c3.position == c3b.position and c3.face == c3b.face == "bottom"
    m3, m3b = sim3._msl_ports[0], sim3b._msl_ports[0]
    assert m3.width == m3b.width == W_TRACE
    assert m3.height == m3b.height == H_SUB
    assert m3.eps_r_sub == m3b.eps_r_sub == EPS_SUB
    assert m3.direction == m3b.direction == "-x"
    assert m3.position == m3b.position
    assert m3.position[0] == FEED_X_2

    k3 = _attempt2_kwargs(135000)
    k3b = _attempt3b_kwargs(135000)
    assert set(k3) == set(k3b)
    differing = {k for k in k3 if not np.array_equal(
        np.asarray(k3[k], dtype=object), np.asarray(k3b[k], dtype=object))}
    msl_probe_keys = {
        "msl_probe_count", "msl_probe_start_cells", "msl_probe_spacing_cells"}
    assert differing <= msl_probe_keys, differing
    assert differing == {"msl_probe_count", "msl_probe_start_cells"}, differing
    assert k3["msl_probe_spacing_cells"] == k3b["msl_probe_spacing_cells"] \
        == PROBE_SPACING_3B == PROBE_SPACING_2 == 10
    assert (k3["msl_probe_count"], k3["msl_probe_start_cells"]) == (
        PROBE_COUNT_2, PROBE_START_2) == (9, 4)
    assert (k3b["msl_probe_count"], k3b["msl_probe_start_cells"]) == (
        PROBE_COUNT_3B, PROBE_START_3B) == (8, 15)


def test_attempt3b_ladder_satisfies_the_standoff_at_both_ends():
    """The #823 ladder contract, measured on the REALIZED probe coordinates.

    Attempt 3b: every one of its 8 probes stands at least
    ``_msl_near_field_standoff_cells`` = 15 cells = 1.5 mm = 5.00 * h_sub
    from BOTH the MSL port feed plane (x = 11.0 mm) and the reference plane
    the ladder is referred to (the junction, x = 1.0 mm).

    FAIL-BEFORE-FIX / positive control, in the same test so the record shows
    the predicate discriminating: attempt 3's committed ladder VIOLATES the
    same predicate at the port end (2 of 9 probes, at d = 0.4 and 1.4 mm),
    and its worst standoff is 1.33 * h_sub. A predicate that passed both
    ladders would certify nothing.
    """
    sim3b = _build_coax_msl_transition_sim_attempt3b()
    xs_3b = _msl_ladder_x_coords(
        sim3b, count=PROBE_COUNT_3B, start_cells=PROBE_START_3B,
        spacing_cells=PROBE_SPACING_3B)
    assert xs_3b.shape == (PROBE_COUNT_3B,)
    np.testing.assert_allclose(
        xs_3b * 1e3, [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5], atol=1e-9)

    n_port, n_ref, d_port, d_ref, required_m = _msl_standoff_violations(
        xs_3b, feed_x=FEED_X_2, ref_x=JUNCTION_X, dx=DX, h_sub=H_SUB)
    assert _msl_near_field_standoff_cells(DX, H_SUB) == 15
    assert _msl_near_field_standoff_cells(DX, H_SUB) == max(
        3, int(round(MSL_STANDOFF_H_SUB_MULTIPLE * H_SUB / DX)))
    np.testing.assert_allclose(required_m, 1.5e-3, atol=1e-12)
    np.testing.assert_allclose(required_m / H_SUB, 5.0, atol=1e-9)
    assert (n_port, n_ref) == (0, 0), (
        f"attempt 3b violates the standoff: {n_port} probe(s) at the port "
        f"end, {n_ref} at the reference plane")
    np.testing.assert_allclose([d_port, d_ref], [1.5e-3, 1.5e-3], atol=1e-9)

    sim3 = _build_coax_msl_transition_sim_attempt3()
    xs_3 = _msl_ladder_x_coords(
        sim3, count=PROBE_COUNT_2, start_cells=PROBE_START_2,
        spacing_cells=PROBE_SPACING_2)
    np.testing.assert_allclose(
        xs_3 * 1e3, [2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6, 9.6, 10.6], atol=1e-9)
    n_port_3, n_ref_3, d_port_3, d_ref_3, _ = _msl_standoff_violations(
        xs_3, feed_x=FEED_X_2, ref_x=JUNCTION_X, dx=DX, h_sub=H_SUB)
    assert (n_port_3, n_ref_3) == (2, 0), (n_port_3, n_ref_3)
    np.testing.assert_allclose(d_port_3, 0.4e-3, atol=1e-9)
    np.testing.assert_allclose(d_port_3 / H_SUB, 4.0 / 3.0, atol=1e-6)
    np.testing.assert_allclose(d_ref_3, 1.6e-3, atol=1e-9)


def test_attempt3b_ladder_is_the_unique_maximal_count_compliant_ladder():
    """No free parameter was chosen. At the INHERITED spacing of 10 cells the
    compliant window (both ends at 15 cells from a plane 100 cells apart)
    admits count=8/start=15 and nothing longer; the enumeration is done on
    the REALIZED coordinates, not on arithmetic over the recipe.

    Shorter compliant ladders exist (count=7, start in [15, 25]) and are
    named here so 'unique' is never read as 'the only compliant recipe' --
    3b is the unique MAXIMAL-COUNT one.
    """
    sim = _build_coax_msl_transition_sim_attempt3b()
    compliant = []
    for count in range(3, 12):
        for start in range(3, 40):
            if start + PROBE_SPACING_3B * (count - 1) > 110:
                continue
            xs = _msl_ladder_x_coords(
                sim, count=count, start_cells=start,
                spacing_cells=PROBE_SPACING_3B)
            if xs.min() <= 0.0 or xs.max() >= LX_2:
                continue
            n_port, n_ref, *_ = _msl_standoff_violations(
                xs, feed_x=FEED_X_2, ref_x=JUNCTION_X, dx=DX, h_sub=H_SUB)
            if (n_port, n_ref) == (0, 0):
                compliant.append((count, start))
    assert compliant, "no compliant ladder at the inherited spacing"
    max_count = max(c for c, _ in compliant)
    assert max_count == PROBE_COUNT_3B
    assert [s for c, s in compliant if c == max_count] == [PROBE_START_3B]
    assert len([s for c, s in compliant if c == max_count - 1]) > 1
    xs_alt = _msl_ladder_x_coords(sim, count=9, start_cells=14, spacing_cells=9)
    n_port_alt, n_ref_alt, d_port_alt, _, _ = _msl_standoff_violations(
        xs_alt, feed_x=FEED_X_2, ref_x=JUNCTION_X, dx=DX, h_sub=H_SUB)
    assert (n_port_alt, n_ref_alt) == (1, 1)
    np.testing.assert_allclose(d_port_alt / H_SUB, 14.0 / 3.0, atol=1e-6)



_SYNTHETIC_NONPASSIVE_S = [[[1.2 + 0j], [1.2 + 0j]],
                           [[0.05 + 0j], [1.2 + 0j]]]

_SYNTHETIC_PASSIVE_S = [[[0.3 + 0j], [0.4 + 0j]],
                        [[0.4 + 0j], [0.3 + 0j]]]


def _transition_strict_passivity_default():
    """``strict_passivity``'s default as the PUBLIC method carries it."""
    import inspect
    from rfx.api import Simulation
    return inspect.signature(
        Simulation.compute_coax_msl_transition
    ).parameters["strict_passivity"].default


def _synthetic_transition_result(s_params):
    """The three fields ``_warn_if_nonpassive_smatrix`` reads off a result."""
    from types import SimpleNamespace
    return SimpleNamespace(
        s_params=np.asarray(s_params, dtype=complex),
        freqs=np.asarray([6.0e9]),
        port_names=("coax", "msl"),
    )


def test_transition_default_raises_on_a_nonpassive_s_and_false_returns_it():
    """The guard's own two outcomes, fed the method's signature default.

    SCOPE, stated because an earlier revision of this docstring overclaimed
    it: this pins the DEFAULT and the shared GUARD, not the path between
    them. It calls ``_finalize_sparam_result`` directly, so it stays green if
    the method stops forwarding ``strict_passivity`` to it. The wiring is
    pinned by ``test_transition_forwards_its_strict_passivity_to_the_guard``
    and the two end-to-end tests below.
    """
    from rfx.sparams._common import _finalize_sparam_result

    strict_default = _transition_strict_passivity_default()
    bad = _synthetic_transition_result(_SYNTHETIC_NONPASSIVE_S)
    assert np.all(np.sum(np.abs(bad.s_params) ** 2, axis=0) > 1.10), 'Both synthetic driven-port columns must exceed the guard limit to exercise refusal.'

    with pytest.raises(ValueError, match="UNRELIABLE"):
        _finalize_sparam_result(
            bad, extractor="compute_coax_msl_transition", strict=strict_default)

    with pytest.warns(UserWarning, match="UNRELIABLE"):
        returned = _finalize_sparam_result(
            bad, extractor="compute_coax_msl_transition", strict=False)
    assert returned is bad
    np.testing.assert_array_equal(
        returned.s_params, np.asarray(_SYNTHETIC_NONPASSIVE_S, dtype=complex))


def test_transition_default_returns_a_passive_s_unchanged():
    """The refusal is conditional on the guard, not on the flag alone."""
    from rfx.sparams._common import _finalize_sparam_result

    good = _synthetic_transition_result(_SYNTHETIC_PASSIVE_S)
    assert np.all(np.sum(np.abs(good.s_params) ** 2, axis=0) <= 1.10), 'Both synthetic driven-port columns must stay within the 1.10 column-power guard limit to detect unconditional refusal.'
    returned = _finalize_sparam_result(
        good, extractor="compute_coax_msl_transition",
        strict=_transition_strict_passivity_default())
    assert returned is good
    np.testing.assert_array_equal(
        returned.s_params, np.asarray(_SYNTHETIC_PASSIVE_S, dtype=complex))



def _finalize_call_in_transition_source():
    """The ``_finalize_sparam_result(...)`` Call node inside the method."""
    import ast
    import inspect
    import textwrap
    from rfx.api import Simulation

    tree = ast.parse(textwrap.dedent(
        inspect.getsource(Simulation.compute_coax_msl_transition)))
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "_finalize_sparam_result"
    ]
    assert len(calls) == 1, (
        f"expected exactly one _finalize_sparam_result call in "
        f"compute_coax_msl_transition, found {len(calls)}"
    )
    return calls[0]



_PLANTED_NONPASSIVE_S = _SYNTHETIC_NONPASSIVE_S


def _plant_nonpassive_assembler(monkeypatch, n_freqs):
    """Make the lane's assembler return a known non-passive S, nothing else.
    The assembler is the thinnest seam that leaves the whole tail real."""
    import rfx.sparams.coax as _coax

    s = np.repeat(np.asarray(_PLANTED_NONPASSIVE_S, dtype=complex), n_freqs,
                  axis=2)
    assert np.all(np.sum(np.abs(s) ** 2, axis=0) > 1.10), 'The planted assembler must exceed the column-power limit to exercise production-tail refusal.'
    per_freq = np.ones(n_freqs, dtype=float)
    resid = np.zeros((2, 2, n_freqs), dtype=float)
    cplx = np.zeros((2, 2, n_freqs), dtype=complex)

    def _planted(**kwargs):
        return (s, per_freq, per_freq, resid, resid,
                cplx, cplx.copy(), cplx.copy())

    monkeypatch.setattr(
        _coax, "_assemble_coax_msl_transition_from_voltages", _planted)
    return s


def _instrument_kwargs_without_the_flag(n_steps=1):
    """The vetted instrument fixture's kwargs, minus ``strict_passivity``.

    Removing it is the point: the call must take the method's own default.
    """
    from tests._coax_msl_instrument_fixture import instrument_kwargs

    kw = instrument_kwargs(n_steps)
    assert kw.pop("strict_passivity") is False, (
        "the shared fixture stopped pinning strict_passivity=False; this test "
        "removes that key to exercise the DEFAULT and needs to know it was there"
    )
    return kw


def test_default_call_refuses_a_nonpassive_s_and_says_how_to_get_it_back():
    """A default call on the real method: raises, and names the escape hatch."""
    from tests._coax_msl_instrument_fixture import build_instrument_junction
    from rfx.sparams.coax import COAX_MSL_TRANSITION_REFUSAL_HINT

    mp = pytest.MonkeyPatch()
    try:
        _plant_nonpassive_assembler(mp, n_freqs=3)
        sim = build_instrument_junction()
        with pytest.raises(ValueError, match="UNRELIABLE") as caught:
            sim.compute_coax_msl_transition(**_instrument_kwargs_without_the_flag())
    finally:
        mp.undo()

    message = str(caught.value)
    assert "strict_passivity=False" in message, message
    assert COAX_MSL_TRANSITION_REFUSAL_HINT in message, message
    assert "passivity_violation" in message, message
    assert isinstance(caught.value.__cause__, ValueError)
    assert COAX_MSL_TRANSITION_REFUSAL_HINT not in str(caught.value.__cause__)


def test_compute_s_matrix_route_into_this_lane_refuses_by_default():
    """The dispatcher forwards the default, so the lane refuses there too.

    ``sim.compute_s_matrix()`` picks this lane for 1 coaxial + 1 MSL port
    (rfx/sparams/dispatch.py) and forwards ``**kwargs`` verbatim, so a caller
    who never names the method still gets the refusal. The dispatch test file
    compares the two routes only with ``strict_passivity=False`` supplied, so
    nothing else covers the default here.
    """
    from tests._coax_msl_instrument_fixture import build_instrument_junction

    mp = pytest.MonkeyPatch()
    try:
        _plant_nonpassive_assembler(mp, n_freqs=3)
        sim = build_instrument_junction()
        with pytest.raises(ValueError, match="UNRELIABLE") as caught:
            sim.compute_s_matrix(**_instrument_kwargs_without_the_flag())
    finally:
        mp.undo()
    assert "strict_passivity=False" in str(caught.value)


def test_settled_run_driver_fixture_selection_is_explicit_for_every_label():
    """The driver's ``--fixture`` dispatch names EVERY label explicitly and
    raises on an unknown one.

    Fail-before-fix provenance: the dispatch used to be an if/elif chain
    whose ``else`` branch was attempt2_wide, so adding a new ``choices``
    entry without touching the chain would have run attempt2_wide's builder
    and kwargs under the new label -- a silent mislabel, exactly the class of
    defect this PR exists to fix (#822). It is now a total function,
    ``_select_fixture``, and this test pins each label to its own builder and
    kwargs by identity.
    """
    drv = _load_settled_run_driver()
    import sys

    t_self = sys.modules[__name__]

    expected = {
        "attempt2": (_build_coax_msl_transition_sim_attempt2, _attempt2_kwargs),
        "attempt2_wide": (_build_coax_msl_transition_sim_attempt2_wide,
                          _attempt2_wide_kwargs),
        "attempt3": (_build_coax_msl_transition_sim_attempt3, _attempt2_kwargs),
        "attempt3b": (_build_coax_msl_transition_sim_attempt3b, _attempt3b_kwargs),
    }
    assert set(drv.FIXTURE_CHOICES) == set(expected)
    for label, (build, kwargs_fn) in expected.items():
        sel = drv._select_fixture(t_self, label, 4242)
        assert sel.build is build, label
        assert sel.kwargs == kwargs_fn(4242), label
        assert sel.kwargs["n_steps"] == 4242
        assert set(sel.fixture_geom) == {
            "LX_m", "LY_m", "LZ_m", "junction_x_m", "feed_x_m", "y_c_m"}
        assert label in sel.banner
    s3 = drv._select_fixture(t_self, "attempt3", 4242)
    s3b = drv._select_fixture(t_self, "attempt3b", 4242)
    assert s3.fixture_geom == s3b.fixture_geom
    assert {k for k in s3.kwargs if not np.array_equal(
        np.asarray(s3.kwargs[k], dtype=object),
        np.asarray(s3b.kwargs[k], dtype=object))} == {
        "msl_probe_count", "msl_probe_start_cells"}
    with pytest.raises(ValueError, match="unknown --fixture"):
        drv._select_fixture(t_self, "attempt4", 4242)


def test_cross_family_transition_defaults_to_refusing_single_family_lanes_do_not():
    """The signature invariant: this cross-family lane refuses by default.

    Pinned in BOTH directions. The single-family coax lanes are a separate
    decision and keep ``False``; a blanket flip of every ``strict_passivity``
    in ``rfx/sparams/coax.py`` fails here rather than landing unnoticed.
    """
    import inspect
    from rfx.api import Simulation

    assert _transition_strict_passivity_default() is True, (
        "compute_coax_msl_transition must refuse a non-passive S by default "
        "(issue #838)"
    )
    for name in ("compute_coaxial_two_port",):
        default = inspect.signature(
            getattr(Simulation, name)).parameters["strict_passivity"].default
        assert default is False, (
            f"{name} is a single-family lane and was not part of the #838 "
            f"decision, but its strict_passivity default is {default!r}"
        )


@pytest.mark.parametrize('axis,coordinate,size,center', [
    ('z', .0003, None, None), ('x', .001, (.001, .0002), (.0017, .0004)),
])
def test_extra_flux_monitor_refuses_open_feed_end(axis, coordinate, size, center):
    from tests._coax_msl_instrument_fixture import build_instrument_junction, instrument_kwargs
    sim = build_instrument_junction()
    scratch = build_instrument_junction()
    scratch.add_flux_monitor(axis=axis, coordinate=coordinate, size=size, center=center, name='open-end')
    with pytest.raises(ValueError, match=r'open bottom feed end.*run\(\)/forward\(\) port has a shielded end'):
        sim.compute_coax_msl_transition(**instrument_kwargs(1), extra_flux_monitors=scratch._flux_monitors)
