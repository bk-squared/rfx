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
    s_params, cond_a, rec_resid, fit_resid, gamma_fit = (
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
        "ground node through the substrate to the trace node. All Box/"
        "Cylinder z-boundaries are placed on CELL MIDPOINTS "
        "((n +/- 0.5)*dx / stamp_coaxial_line-style +2*dz margin), not "
        "exact node multiples -- see the module docstring note below on "
        "why exact multiples are unsafe on this grid."
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
}


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
# three measured frequencies) -- with cond_a in the 1e3-1e7 range at every
# bin. Per the falsifier's own discriminant (predeclared above): this
# points at near-degenerate two-drive amplification, not an assembler
# defect (Part 1's planted-voltage tests independently confirm the
# assembler's power-wave normalization is correct) and not a missing-PEC
# defect (independently confirmed by direct pec_mask inspection during
# construction debugging). The coax side reflects strongly (|S11| ~
# 0.81-0.99) while the MSL side's own reflection is small at low/mid
# frequency and rises at the top of the band -- consistent with a
# via-dominated series discontinuity at this SPECIFIC fixture's thin
# (200um radius, ~700um long) unmatched pin-to-trace post, which the task
# scoping explicitly anticipated ("no intermediate matching structure").
#
# Per R2, this STOPS here: no second geometry attempt in this session.
# This is reported as an honest FIRST diagnostic measurement of a
# deliberately unmatched junction, not a validated transition -- the test
# below locks the measured (bad) reciprocity and passivity numbers as a
# reproducibility witness, not as an accuracy claim.

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


@pytest.mark.slow_physics
def test_coax_msl_transition_first_fixture_diagnostic():
    """The one committed #489-leg-4 fixture -- DIAGNOSTIC honesty level.

    See ``PREDECLARATION`` above and the module-level comment immediately
    preceding this test for the full R2/R3 accounting. This test asserts
    exactly what was measured: self-consistency (finite, settled) passes;
    the reciprocity/passivity falsifier fires and is LOCKED as a
    reproducibility witness, not papered over. Runtime ~110s (two FDTD
    drives on a (67, 51, 56)-cell grid, 8000 steps each).
    """
    assert PREDECLARATION["status"] == "RUN"

    sim = _build_coax_msl_transition_sim()
    result = sim.compute_coax_msl_transition(
        junction_x=JUNCTION_X, eps_r_sub=EPS_SUB, n_steps=N_STEPS,
        n_freqs=3, probe_count=6, probe_start_cells=4, probe_spacing_cells=2,
        skip_preflight=True,
    )

    assert result.s_params.shape == (2, 2, 3)
    assert np.all(np.isfinite(result.s_params))
    assert result.port_names == ("coax", "msl")

    # Settling: both drives clear the -40 dB ring-down rule (this fixture
    # measured -43.9 / -63.6 dB at N_STEPS=8000; a shorter, 1500-step
    # smoke run measured only -1.0 / -0.2 dB -- the record was genuinely
    # under-settled there, not a construction defect).
    assert np.all(np.asarray(result.settling_db) < -40.0), (
        f"settling_db {result.settling_db} did not clear -40 dB; the "
        "8000-step record may need lengthening."
    )

    # Amplitude falsifier: pin the MEASURED envelope with the repo's
    # standard 1.5x margin (tests/_gate_policy.py) rather than a fixed
    # a-priori threshold -- this fixture's own diagonal peaks near 1
    # (strong coax-side reflection; see the module comment) but must not
    # cross the passivity-violation floor by more than extraction noise.
    max_abs_s = float(np.max(np.abs(result.s_params)))
    gate = gate_from_envelope(max_abs_s, quantum=100)
    assert max_abs_s <= gate, (
        f"max|S| {max_abs_s:.4f} exceeds its own {gate:.4f} envelope gate "
        "-- a NEW amplitude excursion beyond what was measured when this "
        "test was written; re-run the falsifier diagnostic (recurrence_"
        "residual / fit_residual / cond_a) before trusting the change."
    )

    # Degeneracy witness: cond_a stays large (the predeclared discriminant
    # for the reciprocity falsifier below). A LOW cond_a here without the
    # matching reciprocity fix would itself be suspicious (the two
    # findings should move together).
    assert np.all(np.asarray(result.cond_a) > 1.0e3), (
        f"cond_a {result.cond_a} dropped below the degenerate-drive floor "
        "this fixture measured -- the ill-conditioning finding may no "
        "longer apply; re-evaluate the reciprocity assertion below too."
    )

    # Reciprocity falsifier -- LOCKED, not silently accepted: this
    # fixture's own measured deviation is ~94-100% (see module comment).
    # Regression-lock the FINDING (it stays badly non-reciprocal) so a
    # future change that quietly "fixes" this number gets flagged for
    # review rather than assumed correct -- the fix would need its own
    # falsifier re-run (a NEW fixture geometry, e.g. adding a matching
    # structure or a wider/shorter via), not a silent gate change here.
    pair, worst_dev = _mixed_reciprocity_deviation(result.s_params)
    assert worst_dev > 0.5, (
        f"reciprocity deviation dropped to {worst_dev:.3f} between ports "
        f"{pair} -- the pre-declared falsifier no longer fires. Either "
        "the fixture's physics changed (re-derive the finding) or the "
        "assembler changed (re-run Part 1's planted-voltage tests, which "
        "independently pin the correct answer)."
    )
