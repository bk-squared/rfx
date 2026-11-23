"""Mixed-family S-matrix lane (issue #488): guards + power-wave honesty.

Layer 1 (this file, fast suite):
  * the Kurokawa cross-impedance normalization unit falsifier — a synthetic
    unit power transfer across UNEQUAL reference impedances must read
    |S21| = |S12| = 1 (the issue-#460 sqrt(Z_j/Z_i) class caught directly,
    no FDTD run involved),
  * the delivered-power witness consistency on the same synthetic,
  * registration-guard contract (v1 envelope raises loudly),
  * one end-to-end PLUMBING smoke on a coarse probe-fed MSL line —
    asserts shapes/finiteness/witness bookkeeping only. It is NOT a
    physics gate: the claims-bearing fixture battery with the
    pre-declared falsifiers (passivity, reciprocity, openEMS referee)
    is the separate slow-lane battery.
"""


import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.api._sparams import _assemble_mixed_power_wave_s
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.probes.probes import DFTPlaneProbe
from rfx.sources.sources import (
    GaussianPulse, WirePort, _wire_port_cells, _wire_port_live_cells,
)
from tests._realized_geometry import (
    assert_sheet_planes, assert_wall_planes, realized,
)

_EPS_R = 3.66
_H_SUB = 254e-6
_W_TRACE = 600e-6
_DX = _H_SUB / 3.0



def _synthetic_unit_transfer(z0_lump=50.0, z0_msl=30.0, n_freqs=3):
    """Phasors for an ideal matched unit POWER transfer lumped <-> MSL.

    Both runs are constructed so the incident wave carries unit power-wave
    amplitude and the far port receives unit power-wave amplitude:

    * run 0 (lumped driven): FDTD-sign V/I with a = (-V + Z0*I)/(2*sqrt(Z0))
      = 1 and drive-port diagonal b = 0 (matched);
      MSL receive with (V0 - Z_msl*I)/2 = sqrt(Z_msl) i.e. b_pw = 1.
    * run 1 (MSL driven): a_msl = (V0 + Z_msl*I)/2/sqrt(Z_msl) = 1;
      lumped receive with (V - Z0*I)/(2*sqrt(Z0)) = 1.

    In VOLTAGE pseudo-waves the same data reads b_V/a_V = sqrt(Z_msl/Z0)
    != 1 — exactly the issue-#460 error a non-power-wave mixed lane
    would report.
    """
    s50, s30 = np.sqrt(z0_lump), np.sqrt(z0_msl)
    ones = np.ones(n_freqs, dtype=np.complex128)
    v_lw = np.zeros((2, 1, n_freqs), dtype=np.complex128)
    i_lw = np.zeros_like(v_lw)
    v0_msl = np.zeros_like(v_lw)
    i_msl = np.zeros_like(v_lw)
    v_lw[0, 0] = -s50 * ones
    i_lw[0, 0] = ones / s50
    v0_msl[0, 0] = s30 * ones
    i_msl[0, 0] = -ones / s30
    v0_msl[1, 0] = s30 * ones
    i_msl[1, 0] = ones / s30
    v_lw[1, 0] = s50 * ones
    i_lw[1, 0] = -ones / s50
    return v_lw, i_lw, v0_msl, i_msl


def test_power_wave_normalization_unequal_z0():
    """|S21| of a unit power transfer is 1 for UNEQUAL Z0 (issue #460).

    The voltage pseudo-wave ratio on the same phasors is
    sqrt(30/50) = 0.7746 — a mixed lane without the Kurokawa sqrt(Z)
    factors would report that instead.
    """
    z0_lump, z0_msl = 50.0, 30.0
    v_lw, i_lw, v0_msl, i_msl = _synthetic_unit_transfer(z0_lump, z0_msl)
    S, s21_power = _assemble_mixed_power_wave_s(
        v_lw, i_lw, v0_msl, i_msl,
        np.asarray([z0_lump]), np.asarray([1]), np.asarray([z0_msl]),
        wire_mode=False, drive_plan=[("lw", 0), ("msl", 0)],
    )
    S = np.asarray(S)
    pseudo_ratio = np.sqrt(z0_msl / z0_lump)
    assert abs(pseudo_ratio - 1.0) > 0.2
    assert np.isnan(S[1, 0, :]).all()
    np.testing.assert_allclose(np.abs(S[0, 1, :]), 1.0, atol=1e-5)
    assert np.isnan(S[0, 0, :]).all()


def test_reciprocity_on_synthetic_transfer():
    """S12 == S21 on the synthetic reciprocal transfer (both = 1).
    Reciprocity cannot compare a refused drive.
    """
    v_lw, i_lw, v0_msl, i_msl = _synthetic_unit_transfer()
    S, _ = _assemble_mixed_power_wave_s(
        v_lw, i_lw, v0_msl, i_msl,
        np.asarray([50.0]), np.asarray([1]), np.asarray([30.0]),
        wire_mode=False, drive_plan=[("lw", 0), ("msl", 0)],
    )
    S = np.asarray(S)
    assert np.isnan(S[:, 0]).all()
    np.testing.assert_allclose(np.abs(S[0, 1, :]), 1.0, atol=1e-5)


def test_power_witness_matches_synthetic():
    """Delivered-power |a| reconstruction reproduces the unit transfer.

    P_del = 0.5*Re(Z_in)*|I|^2 = 0.5 at the matched synthetic drive,
    so |a|_recon = sqrt(2*P_del/(1-|S11|^2)) = 1 and the witness equals
    |b_msl_pw| = 1 (issue #313 triangulation channel).
    The power witness inherits the diagonal refusal through its 1 - |S00|² denominator.
    """
    v_lw, i_lw, v0_msl, i_msl = _synthetic_unit_transfer()
    _, s21_power = _assemble_mixed_power_wave_s(
        v_lw, i_lw, v0_msl, i_msl,
        np.asarray([50.0]), np.asarray([1]), np.asarray([30.0]),
        wire_mode=False, drive_plan=[("lw", 0), ("msl", 0)],
    )
    assert np.isnan(s21_power[0, 0, :]).all()


def test_wire_mode_uses_per_cell_impedance():
    """Wire off-diagonals normalize by Z0/n_live (issue #318 convention).

    Rebuilding the run-0 synthetic with Z0c = 50/5 in the wave formulas
    must again read |S21| = 1 under wire_mode with n_live = 5; feeding the
    LUMPED-convention phasors instead mis-normalizes by sqrt(n_live).
    Wire drives share the refused per-cell frame, including their off-diagonals.
    """
    n_live, z0_lump, z0_msl = 5, 50.0, 30.0
    z0c = z0_lump / n_live
    sqc, s30 = np.sqrt(z0c), np.sqrt(z0_msl)
    ones = np.ones(3, dtype=np.complex128)
    v_lw = np.zeros((1, 1, 3), dtype=np.complex128)
    i_lw = np.zeros_like(v_lw)
    v0_msl = np.zeros_like(v_lw)
    i_msl = np.zeros_like(v_lw)
    v_lw[0, 0] = -sqc * ones
    i_lw[0, 0] = ones / sqc
    v0_msl[0, 0] = s30 * ones
    i_msl[0, 0] = -ones / s30
    S, _ = _assemble_mixed_power_wave_s(
        v_lw, i_lw, v0_msl, i_msl,
        np.asarray([z0_lump]), np.asarray([n_live]), np.asarray([z0_msl]),
        wire_mode=True, drive_plan=[("lw", 0)],
    )
    assert np.isnan(np.asarray(S)[:, 0, :]).all()



def _base_sim(**kw):
    """35 um foil is a SHEET on the laminate face, declared by a zero-thickness Box."""
    assert _H_SUB / _DX == 3.0, 'The board substrate height must be exactly three mesh cells.'
    lx, ly, lz = 8e-3, 3e-3, 754e-6
    kw.setdefault("snap", "declared")
    sim = Simulation(
        freq_max=5e9, domain=(lx, ly, lz), dx=_DX, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
        **kw,
    )
    sim.add_material("sub", eps_r=_EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (lx, ly, _H_SUB)), material="sub")
    y_c = ly / 2.0
    sim.add(Box((0.0, y_c - _W_TRACE / 2, _H_SUB),
                (lx, y_c + _W_TRACE / 2, _H_SUB)), material="pec")
    return sim, y_c


def _bisecting_sim():
    """The same board on a mesh that BISECTS the laminate face.

    dx = 80 um against h_sub = 254 um: h_sub/dx = 3.175, so no node sits
    on the substrate top. The contract does not hide that — the foil
    sheet lands on the nearest node (3, z = 240 um), 14 um below the face
    it was drawn on, and the assembly warns that the plane is buried in
    the dielectric. Kept deliberately off-lattice because the V-span test
    below is ABOUT the declared-vs-realized offset; every other fixture in
    this file is on-lattice.
    """
    lx, ly, lz = 8e-3, 3e-3, 754e-6
    sim = Simulation(snap="declared",
        freq_max=5e9, domain=(lx, ly, lz), dx=80e-6, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("sub", eps_r=_EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (lx, ly, _H_SUB)), material="sub")
    y_c = ly / 2.0
    sim.add(Box((0.0, y_c - _W_TRACE / 2, _H_SUB),
                (lx, y_c + _W_TRACE / 2, _H_SUB)), material="pec")
    return sim, y_c


def _add_msl(sim, y_c, x=5.5e-3, direction="-x", **kw):
    """Default facing is TOWARD the wire feed because a port facing away from the DUT measures launch+return mixed in its a-channel."""
    sim.add_msl_port(position=(x, y_c, 0.0), width=_W_TRACE,
                     height=_H_SUB, direction=direction, impedance=50.0,
                     waveform=GaussianPulse(f0=2.5e9, bandwidth=0.5), **kw)


def _add_feed(sim, y_c, x=2e-3, extent=None):
    """On the lattice, ``extent=_H_SUB`` lands the port's top NODE on the sheet plane, which is galvanic contact."""
    sim.add_port(position=(x, y_c, 0.0), component="ez",
                 impedance=50.0, extent=_H_SUB if extent is None else extent, terminates=sim._geometry[-1].shape)


def test_a_lumped_feed_reaches_the_lumped_branch_and_its_numbers_are_pinned():
    """The one-cell lumped feed refuses its unverified driven reflection."""
    sim, y_c = _base_sim()
    sim.add_port(position=(2e-3, y_c, 0.0), component="ez", impedance=50.0)
    _add_msl(sim, y_c, n_probe_offset=10, n_probe_spacing=4)

    with pytest.warns(UserWarning, match="refused lumped/wire-driven columns"):
        res = sim.compute_mixed_s_matrix(
            freqs=np.linspace(1e9, 2e9, 3), num_periods=2.0,
            skip_preflight=True, magnitude_channel="wave",
        )

    assert res.port_families == ("lumped", "msl"), (
        "this fixture no longer reaches the lumped branch, so it no longer "
        "covers what it was written for")
    S = np.asarray(res.S)
    assert S.shape == (2, 2, 3)
    assert np.isnan(S[0, 0]).all()
    assert np.isnan(S[1, 0]).all()
    assert np.isfinite(S[:, 1]).all()


def test_guard_requires_msl_port():
    sim, y_c = _base_sim()
    _add_feed(sim, y_c)
    with pytest.raises(ValueError, match="add_msl_port"):
        sim.compute_mixed_s_matrix(skip_preflight=True)


def test_guard_requires_lumped_or_wire_port():
    sim, y_c = _base_sim()
    _add_msl(sim, y_c)
    with pytest.raises(ValueError, match="add_port"):
        sim.compute_mixed_s_matrix(skip_preflight=True)


def test_guard_rejects_bare_source():
    sim, y_c = _base_sim()
    _add_feed(sim, y_c)
    _add_msl(sim, y_c)
    sim.add_source(position=(1e-3, y_c, 4e-4), component="ez",
                   waveform=GaussianPulse(f0=2.5e9, bandwidth=0.5))
    with pytest.raises(NotImplementedError, match="bare sources"):
        sim.compute_mixed_s_matrix(skip_preflight=True)


def test_guard_rejects_nonuniform_mesh():
    sim, y_c = _base_sim(dz_profile=np.full(10, 75.4e-6))
    _add_feed(sim, y_c)
    _add_msl(sim, y_c)
    with pytest.raises(NotImplementedError, match="uniform mesh"):
        sim.compute_mixed_s_matrix(skip_preflight=True)


def test_guard_rejects_mixed_lumped_and_wire():
    sim, y_c = _base_sim()
    _add_feed(sim, y_c)
    sim.add_port(position=(1.5e-3, y_c, 4e-4), component="ez",
                 impedance=50.0)
    _add_msl(sim, y_c)
    with pytest.raises(NotImplementedError, match="lumped \\+ wire"):
        sim.compute_mixed_s_matrix(skip_preflight=True)


def test_flux_magnitude_override_math():
    """Arch-A unit falsifier: flux ratios replace off-diagonal magnitudes.

    P_net = 0.75 at a drive with |S11| = 0.5 gives P_inc = 1.0; a receive
    plane carrying 0.36 toward the port must read |S21| = 0.6 with the
    WAVE phase preserved and the diagonal untouched — regardless of how
    wrong the wave-channel magnitude was (the #313 class).
    The MSL-driven column has P_inc = 1.0 from P_net = (-1)*(-0.99) = 0.99 and |S22|=0.1.
    """
    from rfx.api._sparams import _mixed_flux_magnitude_override
    n_freqs = 3
    S_wave = np.zeros((2, 2, n_freqs), dtype=np.complex64)
    S_wave[0, 0, :] = 0.5
    WRONG_MAGNITUDE_QUADRATURE_WAVE = 1.7j
    S_wave[1, 0, :] = WRONG_MAGNITUDE_QUADRATURE_WAVE
    S_wave[1, 1, :] = 0.1
    WRONG_MAGNITUDE_PI_PHASE_WAVE = -0.2
    S_wave[0, 1, :] = WRONG_MAGNITUDE_PI_PHASE_WAVE
    box_lw = np.zeros((2, 1, n_freqs))
    plane_msl = np.zeros((2, 1, n_freqs))
    LUMPED_DRIVE_NET_OUTWARD_POWER = 0.75
    box_lw[0, 0, :] = LUMPED_DRIVE_NET_OUTWARD_POWER
    MSL_RECEIVE_PLUS_X_POWER = 0.36
    plane_msl[0, 0, :] = MSL_RECEIVE_PLUS_X_POWER
    MSL_DRIVE_MINUS_X_POWER = -0.99
    plane_msl[1, 0, :] = MSL_DRIVE_MINUS_X_POWER
    LUMPED_RECEIVE_INWARD_BOX_FLUX = -0.25
    box_lw[1, 0, :] = LUMPED_RECEIVE_INWARD_BOX_FLUX
    S_out, ill, neg = _mixed_flux_magnitude_override(
        S_wave, box_lw, plane_msl, [("lw", 0), ("msl", 0)],
        msl_away_signs=[-1.0], n_lw=1,
    )
    S_out = np.asarray(S_out)
    np.testing.assert_allclose(S_out[0, 0, :], 0.5, atol=1e-6)
    np.testing.assert_allclose(np.abs(S_out[1, 0, :]), 0.6, rtol=1e-5)
    np.testing.assert_allclose(np.angle(S_out[1, 0, :]), np.pi / 2, atol=1e-5)
    np.testing.assert_allclose(np.abs(S_out[0, 1, :]), 0.5, rtol=1e-5)
    np.testing.assert_allclose(np.abs(np.angle(S_out[0, 1, :])), np.pi, atol=1e-5)
    assert not ill.any() and not neg.any()


def test_flux_column_power_is_an_identity_not_a_passivity_check():
    """ANTI-GATE LOCK (issue #488): do not gate passivity on this channel.

    With |S_ij|^2 = P_arr,i / (P_net,j / (1 - |S_jj|^2)), the column sum
    collapses to |S_jj|^2 + (P_arr/P_net)(1 - |S_jj|^2), which is exactly
    1 whenever the arriving power equals the net launched power — for ANY
    diagonal. A green column-power check therefore binds an algebraic
    identity, not physics (feedback: a gate can bind an artifact). The
    independent internal witness on the flux channel is RECIPROCITY,
    because S_ij and S_ji come from different runs.

    This test asserts the identity holds across wildly different
    diagonals so that anyone tempted to promote column power to a
    passivity gate here sees why it cannot discriminate.
    """
    from rfx.api._sparams import _mixed_flux_magnitude_override
    ARBITRARY_WRONG_WAVE_MAGNITUDE = 0.123
    for s11_mag in (0.03, 0.41, 0.90):
        S_wave = np.zeros((2, 2, 1), dtype=np.complex64)
        S_wave[0, 0, :] = s11_mag
        S_wave[1, 1, :] = 0.5
        S_wave[1, 0, :] = ARBITRARY_WRONG_WAVE_MAGNITUDE
        box_lw = np.zeros((2, 1, 1))
        plane_msl = np.zeros((2, 1, 1))
        box_lw[0, 0, :] = 0.4
        plane_msl[0, 0, :] = 0.4
        assert np.array_equal(plane_msl[0, 0, :], box_lw[0, 0, :]), 'The lossless input must receive all net launched power.'
        S_out, _, _ = _mixed_flux_magnitude_override(
            S_wave, box_lw, plane_msl, [("lw", 0)],
            msl_away_signs=[-1.0], n_lw=1,
        )
        col_power = float(
            np.abs(np.asarray(S_out)[0, 0, 0]) ** 2
            + np.abs(np.asarray(S_out)[1, 0, 0]) ** 2
        )
        assert col_power == pytest.approx(1.0, abs=1e-6), (
            f"identity broken for |S11|={s11_mag}: {col_power}"
        )
    S_wave = np.zeros((2, 2, 1), dtype=np.complex64)
    S_wave[0, 0, :] = 0.0
    box_lw = np.zeros((2, 1, 1))
    plane_msl = np.zeros((2, 1, 1))
    box_lw[0, 0, :] = 0.4
    plane_msl[0, 0, :] = 0.8
    assert np.array_equal(plane_msl[0, 0, :], 2 * box_lw[0, 0, :]), 'The gain input must receive twice the net launched power.'
    S_out, _, _ = _mixed_flux_magnitude_override(
        S_wave, box_lw, plane_msl, [("lw", 0)],
        msl_away_signs=[-1.0], n_lw=1,
    )
    assert float(np.abs(np.asarray(S_out)[1, 0, 0]) ** 2) > 1.5


def test_flux_override_masks_ill_conditioned_and_negative():
    from rfx.api._sparams import _mixed_flux_magnitude_override
    S_wave = np.zeros((2, 2, 2), dtype=np.complex64)
    NEAR_TOTAL_REFLECTION = 0.999
    S_wave[0, 0, :] = NEAR_TOTAL_REFLECTION
    box_lw = np.zeros((2, 1, 2))
    plane_msl = np.zeros((2, 1, 2))
    box_lw[0, 0, :] = -1e-3
    assert np.all(box_lw[0, 0, :] < 0), 'The driven input must have negative net box power.'
    _, ill, neg = _mixed_flux_magnitude_override(
        S_wave, box_lw, plane_msl, [("lw", 0), ("msl", 0)],
        msl_away_signs=[-1.0], n_lw=1,
    )
    assert ill[0].all() and neg[0].all()


def test_reciprocity_witness_catches_a_wrong_diagonal():
    """The runtime witness must catch what column power structurally cannot.

    A wrong driven-port diagonal corrupts the incident-power
    normalization `P_inc = P_net/(1-|S_jj|^2)` on that column only, so
    |S21| and |S12| stop agreeing. Column power cannot see it (it is an
    identity — see the anti-gate-lock test); reciprocity can. This is the
    exact failure class that reached review in the first revision of this
    lane.
    The SHIPPED DEFAULT tolerance is required so the warning a user actually gets would fire.
    """
    from rfx.api._sparams import (
        _mixed_flux_magnitude_override,
        _mixed_reciprocity_deviation,
    )
    box_lw = np.zeros((2, 1, 1))
    plane_msl = np.zeros((2, 1, 1))
    box_lw[0, 0, :] = 1.0
    plane_msl[0, 0, :] = 1.0
    plane_msl[1, 0, :] = -1.0
    box_lw[1, 0, :] = -1.0
    assert np.array_equal(box_lw, -box_lw[::-1]) and np.array_equal(plane_msl, box_lw), 'The symmetric lossless input must launch and receive equal power on both ports.'
    plan = [("lw", 0), ("msl", 0)]

    def run(s11, s22):
        S = np.zeros((2, 2, 1), dtype=np.complex64)
        S[0, 0, :], S[1, 1, :] = s11, s22
        out, _, _ = _mixed_flux_magnitude_override(
            S, box_lw, plane_msl, plan, msl_away_signs=[-1.0], n_lw=1,
        )
        return _mixed_reciprocity_deviation(out)[1]

    assert run(0.4, 0.4) == pytest.approx(0.0, abs=1e-5)
    import inspect

    default_tol = inspect.signature(
        Simulation.compute_mixed_s_matrix
    ).parameters["reciprocity_tol"].default
    assert run(0.4, 0.03) > default_tol, (
        f"a wrong diagonal deviates by {run(0.4, 0.03):.3f}, which the "
        f"shipped default tolerance {default_tol} would NOT warn about"
    )
    EARLIER_REAL_FIXTURE_RESIDUAL = 0.09
    assert default_tol < EARLIER_REAL_FIXTURE_RESIDUAL


def test_guard_requires_pec_ground_for_flux_channel():
    """Flux box omits its bottom face; that needs a PEC z_lo (review)."""
    from rfx.boundaries.spec import Boundary as _B, BoundarySpec as _BS
    lx, ly, lz = 8e-3, 3e-3, 754e-6
    sim = Simulation(
        freq_max=5e9, domain=(lx, ly, lz), dx=_DX, cpml_layers=8,
        boundary=_BS(x="cpml", y="cpml", z=_B(lo="cpml", hi="cpml")),
    )
    sim.add_material("sub", eps_r=_EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (lx, ly, _H_SUB)), material="sub")
    sim.add(Box((0.0, 0.0, 0.0), (95 * _DX, 36 * _DX, 0.0)), material="pec")
    y_c = ly / 2.0
    sim.add(Box((0.0, 14 * _DX, 3 * _DX),
                (95 * _DX, 21 * _DX, 4 * _DX)), material="pec")
    _add_feed(sim, y_c)
    _add_msl(sim, y_c, n_probe_offset=10, n_probe_spacing=4)
    with pytest.raises(NotImplementedError, match="PEC z_lo"):
        sim.compute_mixed_s_matrix(skip_preflight=True)
    sim.compute_mixed_s_matrix(
        freqs=np.linspace(1e9, 2e9, 2), num_periods=1.0,
        skip_preflight=True, magnitude_channel="wave",
    )


def test_guard_rejects_horizontal_port_on_flux_channel():
    """`pe.extent` is treated as a z height by the box builder (review)."""
    sim, y_c = _base_sim()
    sim.add_port(position=(2e-3, y_c, _H_SUB / 2), component="ex",
                 impedance=50.0, extent=_H_SUB)
    _add_msl(sim, y_c, n_probe_offset=10, n_probe_spacing=4)
    with pytest.raises(NotImplementedError, match="component='ez'"):
        sim.compute_mixed_s_matrix(skip_preflight=True)


def test_negative_arriving_power_is_tracked_not_only_driven():
    """A receive-side sign defect must be flagged, not silently |S|=0."""
    from rfx.api._sparams import _mixed_flux_magnitude_override
    S_wave = np.zeros((2, 2, 1), dtype=np.complex64)
    box_lw = np.zeros((1, 1, 1))
    plane_msl = np.zeros((1, 1, 1))
    HEALTHY_DRIVEN_NET_POWER = 1.0
    box_lw[0, 0, :] = HEALTHY_DRIVEN_NET_POWER
    MIS_SIGNED_RECEIVE_POWER = -1.0
    plane_msl[0, 0, :] = MIS_SIGNED_RECEIVE_POWER
    S_out, _, neg = _mixed_flux_magnitude_override(
        S_wave, box_lw, plane_msl, [("lw", 0)],
        msl_away_signs=[-1.0], n_lw=1,
    )
    assert neg[1].all(), "receive-side negative power went untracked"
    assert float(np.abs(np.asarray(S_out)[1, 0, 0])) == 0.0


def test_forward_returns_the_flux_monitors_it_builds():
    """A declared flux monitor is accumulated AND returned by plain forward()
    (#1373 D4); before, forward() skipped it because it could not return it."""
    import jax.numpy as jnp

    sim = Simulation(freq_max=10e9, domain=(4e-3, 2e-3, 1e-3),
                     dx=100e-6, cpml_layers=4)
    sim.add_port(position=(1e-3, 1e-3, 0.5e-3), component="ez",
                 impedance=50.0)
    sim.add_flux_monitor(axis="x", coordinate=2e-3,
                         freqs=jnp.asarray(np.array([5e9])), name="mid")
    result = sim.forward(n_steps=40, port_s11_freqs=np.array([5e9]),
                         skip_preflight=True)
    assert list(result.flux_monitors) == ["mid"]


def test_guard_rejects_cpml_adjacent_probe_ladder():
    """Regression lock for the #488 attempt-1 defect D3.

    A "+x"-facing MSL port at 5.5 mm in an 8 mm domain resolves its
    default probe ladder (offset 31, spacing 12 cells) past the domain
    edge; msl_probe_x_coords_n CLAMPS the coordinates, and attempt 1
    silently measured probe 0 next to the trace's OPEN end at the domain
    edge (a Box is not rasterized into the CPML padding), reading the
    open-stub standing-wave impedance -j*Z0*cot(beta*d) ~ 1400 ohm at
    1 GHz instead of the 48 ohm line. The mixed lane must refuse the
    registration the same way compute_msl_s_matrix does.
    """
    sim, y_c = _base_sim()
    _add_feed(sim, y_c)
    _add_msl(sim, y_c, direction="+x")
    with pytest.raises(ValueError, match="declared x-domain"):
        sim.compute_mixed_s_matrix(skip_preflight=True)



def _assert_drive_settling_witnesses(sim, result, periods, end_amplitudes):
    from rfx.measurement.record_end import source_end_step

    grid = sim._build_grid()
    n_steps = grid.num_timesteps(periods)
    ports = (*sim._ports, *sim._msl_ports)
    assert len(result.settling_witness) == len(ports) == len(end_amplitudes)
    for port, detail, expected in zip(ports, result.settling_witness, end_amplitudes):
        assert detail['status'] in {'pass', 'fail', 'undetermined'}
        if detail['status'] != 'pass':
            assert detail['reason']
        end, samples = source_end_step(
            [(port.waveform, 0.)], n_steps, grid.dt, return_detail=True)
        amplitudes, peak, _ = samples[0]
        end_amplitude = amplitudes[-1] / peak
        assert end_amplitude == pytest.approx(expected, rel=1e-4, abs=0.)
        if end_amplitude > 1e-6:
            assert end is None
            assert detail['status'] == 'undetermined'
            assert 'source end is unavailable' in detail['reason']


def test_mixed_probe_fed_msl_plumbing_smoke():
    """Coarse probe-fed MSL line runs end to end and fills the result.

    num_periods=4 is deliberately truncated — this asserts BOOKKEEPING
    (shapes, port order, finiteness, witnesses recorded), not physics.
    The claims-bearing battery with the pre-declared falsifiers is the
    slow-lane fixture battery (issue #488 arc).
    The absence of a ring-down warning on a 4-period open-domain record would itself be a bug.
    """
    sim, y_c = _base_sim()
    _add_feed(sim, y_c, x=2e-3)
    _add_msl(sim, y_c, x=5.5e-3, n_probe_offset=10, n_probe_spacing=4)
    freqs = np.linspace(1e9, 4e9, 5)
    with pytest.warns(UserWarning):
        res = sim.compute_mixed_s_matrix(
            freqs=freqs, num_periods=4.0, skip_preflight=True,
        )
    S = np.asarray(res.S)
    assert S.shape == (2, 2, 5)
    assert np.isnan(S[:, 0]).all()
    assert np.all(np.isfinite(S[:, 1]))
    assert res.port_families == ("wire", "msl")
    assert res.port_names[1] != "lw0" and res.port_names[0] == "lw0"
    assert res.z0_ref.shape == (2,)
    assert res.z0_ref[0] == pytest.approx(50.0)
    assert res.z0_ref[1] == pytest.approx(50.0)
    assert 20.0 < res.line_impedances[1] < 120.0
    assert res.s21_power_witness.shape == (1, 1, 5)
    assert np.isnan(res.s21_power_witness).all()
    assert res.magnitude_channel == "flux"
    assert res.S_wave is not None and res.S_wave.shape == S.shape
    _assert_drive_settling_witnesses(sim, res, 4., (0.0780033, 0.322213))
    assert len(sim._dft_planes) == 0
    assert len(sim._probes) == 0
    assert sim._msl_ports[0].excite is True


@pytest.mark.slow
def test_mixed_probe_fed_msl_smoke_slow_lane_exists():
    """Slow-lane smoke for compute_mixed_s_matrix (issue #520 leg 3).

    Before this test, NO test anywhere carried a slow/gpu marker while
    touching compute_mixed_s_matrix — the fast Layer-1c smoke above
    deliberately truncates at num_periods=4 and disclaims physics
    ("NOT PHYSICS... the claims-bearing fixture battery with the
    pre-declared falsifiers is the separate slow-lane battery"), but that
    battery was never built. It is still not built here either: this is a
    minimal smoke check, not the promised passivity/reciprocity/openEMS
    battery (that scope belongs to issue #488's arc; the adjacent
    single-ratio-vs-multi-drive-solve composition question is #517).

    Same real (non-monkeypatched) FDTD fixture as the fast smoke, run a bit
    further (num_periods=8 vs 4) so the record is less trivially truncated.
    Only proves the lane completes end to end and stays finite/bounded —
    not a physics claim.

    MEASURED BLIND (adversarial review of PR #553): this test PASSED
    unchanged under two independently injected defects — a sign flip on
    ``_b_msl`` (``rfx/api/_sparams.py:857``) and a revert of the pre-#511
    V-span anchor (``:3703``) — while the sibling fast Layer-1d test
    (``test_mixed_lane_v_span_reaches_the_rasterized_trace_on_bisecting_mesh``)
    correctly caught the V-span revert. This test gates LIVENESS (the lane
    completes and stays finite/bounded), not correctness. Adding
    discriminating physics assertions here is out of scope (#517/#488).

    MEASURED, this test's own run: ``UserWarning: compute_mixed_s_matrix:
    reciprocity deviation max 30.1% between |S[0,1]| and |S[1,0]|
    (tolerance 6%)``. ``_sparams.py:777`` documents this lane's known
    residual as ~9% (#488/#498); whether the 30.1% seen on THIS fixture is
    the same #498 residual at a different operating point, or a second
    contributor, is OPEN — not explained away here.
    """
    sim, y_c = _base_sim()
    _add_feed(sim, y_c)
    _add_msl(sim, y_c, n_probe_offset=10, n_probe_spacing=4)
    freqs = np.linspace(1e9, 4e9, 5)
    res = sim.compute_mixed_s_matrix(
        freqs=freqs, num_periods=8.0, skip_preflight=True,
    )
    S = np.asarray(res.S)
    assert S.shape == (2, 2, 5)
    assert np.isnan(S[:, 0]).all()
    assert np.all(np.isfinite(S[:, 1]))
    assert np.all(np.abs(S[:, 1]) < 1.5), f"gross blow-up: max|S|={np.max(np.abs(S[:, 1])):.3f}"
    _assert_drive_settling_witnesses(sim, res, 8., (4.13523e-21, 1.59902e-4))




def _fake_forward_mixed_z_profile(nz_markers, n_lw):
    """``_forward_from_materials`` stub: MSL Ez planes carry a fixed
    per-z-node profile; every registered probe plane gets it (retargets
    ``_fake_run_z_profile`` from test_msl_modal_voltage_and_wave_solve.py
    at the mixed lane's own scan hook).

    Hy/Hz carry a z-RAMP, not a uniform value: a uniform H makes the
    closed Ampere-loop current cancel to EXACTLY zero (bottom leg equals
    top leg, left leg equals right leg), which would make ``i_msl`` -- and
    everything downstream of it -- degenerate. Same trap documented next
    to ``_fake_run_drive_dependent`` in that file.
    """

    def fake_forward(self, grid, materials, debye_spec, lorentz_spec,
                     n_steps=None, checkpoint=False, pec_mask=None,
                     pec_sheets=(), pec_wires=(),
                     port_s11_freqs=None, _return_raw_port_sparams=False):
        del materials, debye_spec, lorentz_spec, n_steps, checkpoint, pec_mask
        del pec_sheets, pec_wires
        freqs = jnp.asarray(port_s11_freqs)
        n_f = int(freqs.shape[0])
        zramp = jnp.asarray(
            [1.0 + 0.6 * k for k in range(grid.nz)], dtype=jnp.complex64)
        planes = {}
        for entry in self._dft_planes:
            if entry.component == "ez":
                prof = jnp.asarray(
                    [complex(nz_markers.get(k, 0.0)) for k in range(grid.nz)],
                    dtype=jnp.complex64,
                )
                acc = jnp.broadcast_to(
                    prof[None, None, :], (n_f, grid.ny, grid.nz))
            elif entry.component == "hy":
                acc = jnp.broadcast_to(
                    ((0.018 + 0.004j) * zramp)[None, None, :],
                    (n_f, grid.ny, grid.nz))
            else:
                acc = jnp.broadcast_to(
                    ((0.005 + 0.001j) * zramp)[None, None, :],
                    (n_f, grid.ny, grid.nz))
            planes[entry.name] = DFTPlaneProbe(
                accumulator=acc, freqs=entry.freqs,
                component=entry.component, axis=0, index=0,
                total_steps=1, window="rect", window_alpha=0.25,
            )
        wire_accs = [
            (None, (np.full(n_f, 0.7 + 0.1j, dtype=np.complex128),
                   np.full(n_f, 0.02 - 0.01j, dtype=np.complex128)))
            for _ in range(n_lw)
        ]
        return {"lumped": wire_accs, "wire": wire_accs,
               "dft_planes": planes, "time_series": None}

    return fake_forward


def test_mixed_lane_v_span_reaches_the_realized_trace_on_bisecting_mesh(monkeypatch):
    """The mixed lane's MSL V must anchor on the REALIZED trace plane —
    read from the single owner — and on the bisecting mesh (dx = 80 um,
    h_sub/dx = 3.175) that plane is not where the declaration is.

    #931 §1.3: the foil is a sheet, and a sheet lands on the node plane
    nearest its declared mid-plane, so this board realizes the trace at
    node 3 (z = 240 um), 14 um BELOW the 254 um face it is drawn on. The
    span is therefore Ez edges 0..2 and sums to (1+1+1)*dz.

    The marker keeps its poison: node 3 carries 10 and node 4 carries
    -1000, so a span that over-reaches by one edge reads 13*dz and one
    that runs away reads a large negative — both loud. Before the
    contract the SAME fixture realized the trace at node 4 (a one-cell
    PEC Box's single wall sat on its far face) and this test pinned
    13*dz; the number moved because the realization did, and the
    discriminator moved with it rather than being re-tuned.

    The value is NOT recomputed here from round(h_sub/dx): the expected
    plane is read from ``realized_wall_planes`` so that if the lattice
    ever puts the trace somewhere else, this test states the new place
    instead of silently agreeing with a proxy that happens to match.
    The ``v0_msl`` axes are (n_runs, n_msl, n_freqs).
    """
    marker = {0: 1.0, 1: 1.0, 2: 1.0, 3: 10.0, 4: -1000.0}
    sim, y_c = _bisecting_sim()
    _add_feed(sim, y_c)
    _add_msl(sim, y_c, n_probe_offset=10, n_probe_spacing=4)
    n_lw = len(sim._ports)
    monkeypatch.setattr(Simulation, "_forward_from_materials",
                        _fake_forward_mixed_z_profile(marker, n_lw))

    result, diag = sim.compute_mixed_s_matrix(
        freqs=np.asarray([1.0e9]), magnitude_channel="wave",
        return_diagnostics=True, skip_preflight=True,
    )
    del result
    v0_msl = np.asarray(diag["v0_msl"])
    assert v0_msl.shape[1:] == (1, 1)

    BISECTING_MESH_EDGE_LENGTH_M = 80e-6
    dz = BISECTING_MESH_EDGE_LENGTH_M
    rz = realized(sim)
    k_trace = rz.sheet_planes[2][0]
    assert k_trace == 3, (
        f"the realized trace plane moved to {k_trace}; re-derive the span "
        "below from it before trusting the numbers")
    expected_correct = sum(marker[k] for k in range(k_trace)) * dz
    expected_overreach = expected_correct + marker[k_trace] * dz
    v_msl0 = v0_msl[:, 0, 0]
    np.testing.assert_allclose(v_msl0.real, expected_correct, rtol=1e-5)
    assert not np.isclose(v_msl0[0].real, expected_overreach, rtol=1e-3), (
        f"V over-reached the realized trace plane ({expected_overreach} "
        f"instead of {expected_correct}) — the #511/F2 off-by-one has "
        "returned in the mixed lane's own copy."
    )
    assert sim._ports[0].excite is True


def test_bisecting_mesh_reports_its_declared_vs_realized_offset():
    """The off-lattice board is REPORTED, not absorbed (#931 §1.3).

    Build-time witness, no solve: on the bisecting mesh the foil is drawn
    at z = 254 um and realized at node 3 = 240 um, and the substrate the
    same mesh realizes runs to node 4 = 320 um, so the sheet sits INSIDE
    the dielectric. That is the geometry the fixture did not describe;
    every other fixture in this file is on the lattice so it does not have
    it. Pinned here so the pair (on-lattice vs bisecting) stays visible.
    """
    sim, _y_c = _bisecting_sim()
    rz = realized(sim)
    k_sheet = rz.sheet_planes[2][0]
    assert k_sheet == 3
    z_nodes_dx = 80e-6
    declared, realized_z = _H_SUB, k_sheet * z_nodes_dx
    assert realized_z < declared
    assert abs(declared - realized_z) == pytest.approx(14e-6, abs=1e-9), (
        "the declared-vs-realized offset of the bisecting board is 14 um; "
        f"got {(declared - realized_z) * 1e6:.3f} um")
    on_lattice, _ = _base_sim()
    assert_sheet_planes(on_lattice, 2, [_H_SUB], what="on-lattice foil")
    assert_wall_planes(on_lattice, 2, [_H_SUB], what="on-lattice foil")



def _ground_truth_wire_port_n_live(sim, position, component, extent,
                                   impedance=50.0):
    """Independent ground-truth n_live: build grid + assembled pec_mask
    FRESH (not reusing anything preflight computed) and classify with the
    same primitive the assembler calls
    (rfx/api/_sparams.py:3426, compute_mixed_s_matrix's n_live_lw).
    """
    grid = sim._build_grid()
    axis = {"ex": 0, "ey": 1, "ez": 2}[component]
    end = list(position)
    end[axis] += extent
    wp = WirePort(start=tuple(position), end=tuple(end),
                  component=component, impedance=impedance)
    _, live_flags, n_live = _wire_port_live_cells(
        grid, wp, realized(sim).edge_masks)
    return n_live, len(live_flags)


def _old_buggy_dead_cell_classification(sim, position, component, extent,
                                        impedance=50.0):
    """Frozen, standalone reimplementation of the PRE-#544 advisory
    counting path (bounding-box-vs-cell-CENTER, closed interval). The live
    ``rfx/api/_preflight.py::_validate_cfg_port_inside_pec`` no longer
    contains this logic — kept here only so this test can demonstrate,
    on the exact fixture geometry, that it disagreed with ground truth
    (the #544 bug) without needing a pre-fix checkout.
    """
    grid = sim._build_grid()
    axis = {"ex": 0, "ey": 1, "ez": 2}[component]
    end = list(position)
    end[axis] += extent
    wp = WirePort(start=tuple(position), end=tuple(end),
                  component=component, impedance=impedance)
    d = (grid.dx, getattr(grid, "dy", grid.dx), getattr(grid, "dz", grid.dx))
    pad = (getattr(grid, "pad_x_lo", 0), getattr(grid, "pad_y_lo", 0),
           getattr(grid, "pad_z_lo", 0))
    cells = _wire_port_cells(grid, wp)
    centers = []
    for cell in cells:
        pos = [(cell[ax] - pad[ax]) * d[ax] for ax in range(3)]
        pos[axis] += 0.5 * d[axis]
        centers.append(tuple(pos))
    pec_bboxes = []
    for entry in sim._geometry:
        if entry.material_name != "pec":
            continue
        if not hasattr(entry.shape, "bounding_box"):
            continue
        c1, c2 = entry.shape.bounding_box()
        pec_bboxes.append((c1, c2))
    dead = [idx for idx, c in enumerate(centers)
            if any(all(c1[ax] <= c[ax] <= c2[ax] for ax in range(3))
                  for c1, c2 in pec_bboxes)]
    return len(cells) - len(dead), len(cells)


def test_wire_port_dead_cell_advisory_matches_ground_truth_on_488_fixture():
    """Regression lock (issue #544): on the #488 lane's own fixture the
    vertical feed spans the substrate and every one of its cells is
    genuinely live, so the dead-extent advisory must be silent.

    Before the fix, the advisory reported n_live/n = 3/4 ("terminates at
    50 ohm across its 3 live cells") because it compared a cell-CENTER
    (node + half-cell Yee offset) against the PEC bounding box instead of
    the real rasterization; the assembler's actual n_live_lw was always 4
    (PR #543 measured Z_in = Z0/4 = 12.5 ohm flat, matching n_live = 4 to
    5.8e-4).

    #931: the count is 3/3. The board is ON-LATTICE now (h_sub/dx
    3.175 -> 3) and the foil is a SHEET on node 3; the feed's extent
    ``h_sub`` = 3 dx is HALF-OPEN in edges (R8, 6d66ac65), so it drives
    exactly the three Ez edges between the ground and the foil, cells
    0..2, and every one of them is live — normal E through a sheet stays
    live by contract (§1.3). The 4/4 this test pinned before R8 counted a
    fourth edge ABOVE the foil (cell 3, spanning 3 dx .. 4 dx): the
    endpoint-inclusive rasterizer drove one edge past the declared end,
    and before the contract the same 4/4 came from ``apply_pec_mask``
    happening to preserve normal E inside a one-cell PEC body. The
    property under test is unchanged: no cell is dead, so the advisory
    stays silent, and the ground truth reads the realized edge set — the
    same primitive the assembler applies. Measured build-only 2026-09-07:
    extent 2 dx / 3 dx / 4 dx -> 2/2, 3/3, 4/4 (the 4 dx row is the old
    geometry, declared honestly).
    """
    sim, y_c = _base_sim()
    _add_feed(sim, y_c, x=2e-3)
    _add_msl(sim, y_c, x=5.5e-3, n_probe_offset=10, n_probe_spacing=4)

    n_live_truth, n = _ground_truth_wire_port_n_live(
        sim, position=(2e-3, y_c, 0.0), component="ez", extent=_H_SUB)
    assert (n_live_truth, n) == (3, 3), (
        f"ground truth changed ({n_live_truth}/{n}) -- re-derive the "
        "expected preflight behaviour before trusting the rest of this "
        "test"
    )

    issues = sim.preflight()
    codes = [getattr(s, "code", None) for s in issues]
    assert "wire_port_dead_extent_cells" not in codes, (
        "corrected advisory must be silent: ground truth is n_live/n=3/3 "
        "(issue #544): " + "\n".join(str(s) for s in issues)
    )
    assert "wire_port_midpoint_in_pec" not in codes, (
        "midpoint cell (index 1, z=85um) is live -- #314 must stay "
        "silent too: " + "\n".join(str(s) for s in issues)
    )
    assert "wire_port_dead_cell_classification_unavailable" not in codes, (
        "issue #544 review R1: the classification-unavailable note is "
        "reserved for NU meshes / a materials-build exception -- this "
        "fixture is a clean uniform mesh with a working build, so it "
        "must NOT fire here: " + "\n".join(str(s) for s in issues)
    )


def test_wire_port_dead_cell_advisory_stays_correct_on_an_already_agreeing_case():
    """No false-positive introduction: a case where the pre-#544 advisory
    and the assembler ALREADY agreed must still report the same n_live/n
    after the fix. dx=0.5mm, trace z in [0.9,1.4]mm (one cell thick: a
    VOLUME on cell 2 under #931 §1.1, its centre 1.25mm inside
    [0.9,1.4)mm), port extent=1.5mm -> 3 cells (centers 0.25/0.75/1.25mm;
    the extent is HALF-OPEN in edges, #931 R8 / 6d66ac65, so three edges
    on a 0.5mm mesh are 1.5mm of extent -- this fixture declared 1.0mm
    while the rasterizer drove one edge past the declared end, and 1.0mm
    now reads 2/2 with no dead cell, measured 2026-09-07). The thin-sheet
    midpoint is 1.15mm, UNAMBIGUOUSLY
    closer to node z=1.0mm (0.15mm away) than to node z=1.5mm (0.35mm
    away) -- not a tie, so the ground-truth node is deterministic
    regardless of platform/dtype rounding (issue #544 review item 5: an
    EARLIER version of this fixture used trace z in [1.0,1.5]mm, whose
    thin-sheet midpoint 1.25mm sits EXACTLY equidistant between nodes
    1.0mm and 1.5mm -- Box's own docstring calls that tie unpredictable
    across float32 double-rounding, so it was replaced). Cell index 2
    (z-node=1.0mm, cell center 1.25mm) is genuinely dead both under the
    old center-based check AND under the real node-based rasterization,
    and is NOT the midpoint cell (index 1, z-node=0.5mm).
    """
    dx = 0.5e-3
    sim = Simulation(freq_max=10e9, domain=(0.016, 0.010, 0.006),
                     boundary="cpml", cpml_layers=6, dx=dx)
    sim.add(Box((0.004, 0.003, 0.9e-3), (0.012, 0.007, 1.4e-3)),
            material="pec")
    sim.add_port(position=(0.008, 0.005, 0.0), component="ez",
                 impedance=50.0, extent=1.5e-3)

    n_live_truth, n = _ground_truth_wire_port_n_live(
        sim, position=(0.008, 0.005, 0.0), component="ez", extent=1.5e-3)
    n_live_old, n_old = _old_buggy_dead_cell_classification(
        sim, position=(0.008, 0.005, 0.0), component="ez", extent=1.5e-3)
    assert (n_live_truth, n) == (n_live_old, n_old) == (2, 3), (
        f"expected this fixture to be an ALREADY-AGREEING case (2/3 "
        f"either way); got ground_truth={n_live_truth}/{n} "
        f"old={n_live_old}/{n_old}"
    )

    issues = sim.preflight()
    dead = [s for s in issues
            if getattr(s, "code", None) == "wire_port_dead_extent_cells"]
    assert len(dead) == 1 and "2/3" in dead[0], (
        "advisory text must be unchanged on an already-agreeing fixture: "
        + "\n".join(str(s) for s in issues)
    )


def test_a_cell_occupancy_classifier_cannot_see_a_sheet_at_all():
    """Mutation-style falsifier (issue #544, re-derived at #931).

    The frozen pre-#544 method classifies a port cell by asking whether
    its CENTRE lies in a PEC bounding box. On the #488 fixture that method
    and the realized-edge ground truth now agree — both say every cell of
    the vertical feed is live — but for different reasons, and the
    agreement is an accident of this port's orientation: Ez is NORMAL to
    the foil, and normal E through a sheet stays live by contract.

    Turn the port ninety degrees and the two part company for good. An ex
    wire port lying IN the sheet plane, inside the trace footprint, drives
    a tangential edge: PEC by §1.3, so every cell of it is dead. A
    classifier that reads cell occupancy sees nothing at all, because a
    sheet owns NO cell — ``pec_mask`` is empty under the whole footprint.
    That is the general form of #544: an advisory that re-derives
    occupancy instead of reading the realization is blind to an entire
    conductor class, and no amount of tuning fixes it.
    The mask is checked rather than re-running the old classifier because a zero-thickness bbox makes its verdict turn on a float tie at the sheet plane.
    """
    import numpy as _np

    sim, y_c = _base_sim()
    _add_feed(sim, y_c, x=2e-3)
    _add_msl(sim, y_c, x=5.5e-3, n_probe_offset=10, n_probe_spacing=4)

    n_live_truth, n = _ground_truth_wire_port_n_live(
        sim, position=(2e-3, y_c, 0.0), component="ez", extent=_H_SUB)
    n_live_old, n_old = _old_buggy_dead_cell_classification(
        sim, position=(2e-3, y_c, 0.0), component="ez", extent=_H_SUB)
    assert (n_live_truth, n) == (n_live_old, n_old) == (3, 3)

    with pytest.raises(ValueError, match="ex edge shorted by realized PEC"):
        _ground_truth_wire_port_n_live(
            sim, position=(3e-3, y_c, _H_SUB), component="ex",
            extent=4 * _DX)

    rz = realized(sim)
    assert rz.pec_mask is None or not bool(_np.asarray(rz.pec_mask).any()), (
        "the foil is a sheet: it owns no cell, so any classifier reading "
        "pec_mask is blind to it")
    assert rz.sheet_planes[2], "the foil must still be declared as a sheet"

    issues = sim.preflight()
    codes = [getattr(s, "code", None) for s in issues]
    assert "wire_port_dead_extent_cells" not in codes, (
        "current advisory (sharing the ground-truth primitive) must NOT "
        "reproduce the old method's wrong classification"
    )


def test_wire_port_dead_cell_classification_unavailable_on_nu_mesh():
    """Issue #544 review R1: BLOCKING-1's fix (skip the classification on
    a non-uniform mesh rather than building a mismatched uniform
    substitute) had zero test coverage. On a dz_profile sim with a wire
    port, the new ``wire_port_dead_cell_classification_unavailable``
    advisory must fire (warning severity) and BOTH #314/#319 dead-cell
    advisories must stay silent (there is nothing safe to classify).
    """
    sim, y_c = _base_sim(dz_profile=np.full(10, 75.4e-6))
    _add_feed(sim, y_c, x=2e-3)

    issues = sim.preflight()
    codes = [getattr(s, "code", None) for s in issues]
    hits = [s for s in issues
            if getattr(s, "code", None)
            == "wire_port_dead_cell_classification_unavailable"]
    assert len(hits) == 1, (
        "expected exactly one classification-unavailable note on an NU "
        "sim with one wire port: " + "\n".join(str(s) for s in issues)
    )
    assert getattr(hits[0], "severity", None) == "warning", (
        f"classification-unavailable note must be warning-severity, got "
        f"{getattr(hits[0], 'severity', None)!r}"
    )
    assert "non-uniform mesh" in hits[0] and "dz_profile" in hits[0], (
        f"note must name the NU reason: {hits[0]}"
    )
    assert "wire_port_dead_extent_cells" not in codes, (
        "the uniform-only ground-truth path must not have run on an NU "
        "mesh: " + "\n".join(str(s) for s in issues)
    )
    assert "wire_port_midpoint_in_pec" not in codes, (
        "the uniform-only ground-truth path must not have run on an NU "
        "mesh: " + "\n".join(str(s) for s in issues)
    )


def test_wire_port_dead_cell_classification_unavailable_on_assemble_exception(
    monkeypatch,
):
    """Issue #544 review R1: the OTHER trigger for the same advisory --
    a materials-build exception on an otherwise-uniform sim (defensive
    fallback, review item 6) -- also had zero coverage. Monkeypatches
    ``_assemble_materials`` to raise a REALISTIC build failure
    (``ValueError`` -- matches what a malformed domain/grid config or
    ``_resolve_material``'s own ``KeyError`` class of failure looks like;
    NOT ``RuntimeError``/``TimeoutError``, which are the classes this
    advisory must NOT swallow -- see the sibling propagation test below)
    on a clean uniform #488-style fixture, and asserts the note fires
    carrying the exception text.
    """
    sim, y_c = _base_sim()
    _add_feed(sim, y_c, x=2e-3)
    _add_msl(sim, y_c, x=5.5e-3, n_probe_offset=10, n_probe_spacing=4)

    def _raise_materials_build(*_args, **_kwargs):
        raise ValueError(
            "synthetic materials-assembly failure (issue #544 review "
            "item 1 falsifier)"
        )

    monkeypatch.setattr(sim, "_assemble_materials", _raise_materials_build)

    issues = sim.preflight()
    hits = [s for s in issues
            if getattr(s, "code", None)
            == "wire_port_dead_cell_classification_unavailable"]
    assert len(hits) == 1, (
        "expected exactly one classification-unavailable note when "
        "_assemble_materials raises: " + "\n".join(str(s) for s in issues)
    )
    assert getattr(hits[0], "severity", None) == "warning", (
        f"classification-unavailable note must be warning-severity, got "
        f"{getattr(hits[0], 'severity', None)!r}"
    )
    assert "synthetic materials-assembly failure" in hits[0], (
        f"note must carry the caught exception's text verbatim: {hits[0]}"
    )


def test_wire_port_advisory_does_not_swallow_a_timeout_signal(monkeypatch):
    """Regression test for the studio-packaging CI failure on PR #555's
    original head (f3a3db7). The workflow, ``rfx.experiments`` and the studio
    test named below have since left this repository with Studio; they are
    history here, and the test itself uses only built-in exceptions.

    A broad ``except Exception`` in the new
    dead-cell classification code caught
    ``rfx.experiments.worker.RunTimedOut`` (at the time, a
    ``TimeoutError`` subclass) a SIGALRM handler raises asynchronously
    while ``_assemble_materials`` was in flight, silently converting the
    timeout into a "classification unavailable" advisory instead of
    letting it propagate -- so a 500,000-step experiment with a
    1-second timeout kept simulating instead of dying, hanging
    ``tests/studio/test_durable_worker_lifecycle.py::
    test_worker_timeout_is_durable_failed_outcome``'s subprocess wait
    (60s) with NO worker-side sign of the timeout ever having fired.
    Issue #482 later made ``RunTimedOut``/``RunCancelled`` derive from
    ``BaseException`` instead of ``TimeoutError``/``RuntimeError``, so
    this class of swallow is now impossible by construction; this test
    still guards the narrow-tuple advisory itself using plain builtins
    (below), independent of the worker's real exception hierarchy.

    Reproduces the class directly (no subprocess/signal needed): monkeypatch
    ``_assemble_materials`` to raise plain ``TimeoutError``/``RuntimeError``
    -- the advisory's ``except`` tuple must reject these regardless of
    what the worker's own exception classes derive from -- and assert
    ``preflight()`` does NOT catch it: it must propagate all the way out
    uncaught, exactly as it would have before this advisory existed.
    """
    sim, y_c = _base_sim()
    _add_feed(sim, y_c, x=2e-3)
    _add_msl(sim, y_c, x=5.5e-3, n_probe_offset=10, n_probe_spacing=4)

    def _raise_timeout(*_args, **_kwargs):
        raise TimeoutError("worker exceeded the experiment timeout")

    monkeypatch.setattr(sim, "_assemble_materials", _raise_timeout)
    with pytest.raises(TimeoutError, match="exceeded the experiment timeout"):
        sim.preflight()

    def _raise_cancel(*_args, **_kwargs):
        raise RuntimeError("worker received signal 15")

    monkeypatch.setattr(sim, "_assemble_materials", _raise_cancel)
    with pytest.raises(RuntimeError, match="worker received signal"):
        sim.preflight()



def _d5_gap_ground_truth(sim, position, component, extent, impedance=50.0):
    """Independent ground-truth premise check for the #556 signature.

    Returns ``(end_cell_live, wall_one_cell_beyond_the_end)`` for the
    high-index end of the extent, both read off the REALIZED edge set
    through the single owner (#931 §1.9). The second value used to ask
    whether the cell past the end was in ``pec_mask``; that question
    cannot see a sheet, which owns no cell, so it is now "is there a
    realized wall plane one cell past the port's end node, on the port's
    own column" — the same physical question, asked of the realization
    instead of an occupancy array.
    """
    from rfx.boundaries.pec import realized_wall_planes

    grid = sim._build_grid()
    axis = {"ex": 0, "ey": 1, "ez": 2}[component]
    end = list(position)
    end[axis] += extent
    wp = WirePort(start=tuple(position), end=tuple(end),
                  component=component, impedance=impedance)
    rz = realized(sim)
    cells, live_flags, _ = _wire_port_live_cells(grid, wp, rz.edge_masks)
    end_node = int(cells[-1][axis]) + 1
    ij = tuple(int(cells[-1][a]) for a in range(3) if a != axis)
    walls = realized_wall_planes(rz.edge_masks, axis, ij=ij,
                                 periodic=(False, False, False))
    return bool(live_flags[-1]), (end_node + 1) in walls


def test_wire_port_end_gap_advisory_fires_on_a_declared_one_cell_gap():
    """POSITIVE (issue #556), re-declared on the lattice (#931).

    The board is three cells thick, the foil sheet sits on node 3, and
    the feed is drawn ONE cell short of it (extent = 2*dx puts the port's
    cells at 0..1 and its top node at 2 — the extent is HALF-OPEN in
    edges, #931 R8 / 6d66ac65; it was declared ``dx`` while the
    rasterizer drove one edge past the declared end — so exactly one
    live Ez edge, cell 2, separates the feed from the conductor;
    measured 2026-09-07: (end_live, wall_one_cell_on) = (True, True) at
    2 dx, (True, False) at dx). Every port cell is live, so the
    #314/#319 dead-cell advisories are correctly silent -- and before
    #556 NOTHING warned, while the measured physics was capacitive-only
    coupling (|S21| rising with f, falsifier-ledger D5). The advisory
    must fire exactly once, on the +z side.

    Before the ownership contract this fixture got its gap for free: the
    board was off-lattice and the trace snapped one node above the feed's
    top. That is a rasterization accident, not a modelling statement, and
    the contract removes it -- so the fixture now ASKS for the gap. The
    defect it stands for is unchanged; only the way the fixture reaches it
    is honest about being deliberate.

    The advisory is identified by its structured finding code and severity.
    Its side has no structured field, so check the emitted +z-side token.
    """
    sim, y_c = _base_sim()
    _add_feed(sim, y_c, x=2e-3, extent=2 * _DX)
    _add_msl(sim, y_c, x=5.5e-3, n_probe_offset=10, n_probe_spacing=4)

    end_live, wall_one_cell_on = _d5_gap_ground_truth(
        sim, position=(2e-3, y_c, 0.0), component="ez", extent=2 * _DX)
    assert (end_live, wall_one_cell_on) == (True, True), (
        f"ground-truth premise changed (end_live={end_live}, "
        f"wall_one_cell_on={wall_one_cell_on}) -- the one-cell-gap "
        "signature no longer holds on this fixture; re-derive the "
        "expected preflight behaviour before trusting the rest of this "
        "test"
    )

    issues = sim.preflight()
    hits = [s for s in issues
            if getattr(s, "code", None) == "wire_port_end_gap_to_conductor"]
    assert len(hits) == 1, (
        "expected exactly one #556 end-gap advisory on the D5 fixture: "
        + "\n".join(str(s) for s in issues)
    )
    assert getattr(hits[0], "severity", None) == "warning", (
        f"#556 advisory must be advisory (warning) severity, got "
        f"{getattr(hits[0], 'severity', None)!r}"
    )
    assert str(hits[0]).count("+z-side") == 1
    codes = [getattr(s, "code", None) for s in issues]
    assert "wire_port_dead_extent_cells" not in codes
    assert "wire_port_midpoint_in_pec" not in codes


def test_wire_port_end_gap_advisory_silent_when_extent_reaches_conductor():
    """SILENT CONTROL 1 (issue #556): the feed drawn to the foil.

    ``extent = h_sub`` = 3*dx puts the port's top NODE on the foil's
    realized wall plane (node 3) -- galvanic contact, D5's own remedy.
    The extent is HALF-OPEN in edges (#931 R8, 6d66ac65): 3 dx drives
    cells 0..2 and ends on node 3, exactly the gap between ground and
    foil. This control used to declare ``2*dx`` and reached node 3 only
    because the rasterizer drove one edge past the declared end; under
    the corrected rule 2*dx ends on node 2, one cell SHORT of the foil,
    and is the positive case (measured 2026-09-07: extent 2 dx -> the
    #556 advisory fires once on the +z side; 3 dx -> silent, ground
    truth (end_live, wall_one_cell_on) = (True, False)). The #556 gap
    advisory must not fire.

    #931 changed the partition this control used to assert. Contact used
    to mean "the end cell is dead", which handed the case to the #319
    dead-extent advisory. Against a SHEET it does not: the port drives
    Ez, which is normal to the foil and stays live by contract (§1.3),
    so the end cell is live AND in contact. Neither advisory owns this
    case, and neither should -- a feed that reaches its conductor is
    correct geometry, and #929 records that telling the user to move a
    galvanic feed off the ground was the defect, not the finding.
    """
    sim, y_c = _base_sim()
    _add_feed(sim, y_c, x=2e-3, extent=_H_SUB)
    _add_msl(sim, y_c, x=5.5e-3, n_probe_offset=10, n_probe_spacing=4)

    end_live, wall_one_cell_on = _d5_gap_ground_truth(
        sim, position=(2e-3, y_c, 0.0), component="ez", extent=_H_SUB)
    assert (end_live, wall_one_cell_on) == (True, False), (
        "ground-truth premise changed: a feed drawn to the foil ends ON "
        "its wall plane, so its last cell is live (Ez is normal to a "
        "sheet) and there is no wall a further cell on")

    issues = sim.preflight()
    codes = [getattr(s, "code", None) for s in issues]
    assert "wire_port_end_gap_to_conductor" not in codes, (
        "#556 advisory must stay silent when the port reaches the "
        "conductor (galvanic contact): "
        + "\n".join(str(s) for s in issues)
    )
    assert "wire_port_dead_extent_cells" not in codes, (
        "contact with a SHEET leaves no dead cell, so #319 must be "
        "silent here too: " + "\n".join(str(s) for s in issues)
    )


def test_wire_port_end_gap_advisory_silent_in_open_vacuum():
    """SILENT CONTROL 2 (issue #556): a wire port ending in genuine open
    vacuum -- nothing PEC anywhere in the domain (dipole-feed-like
    situation, mis-gated-guard footgun class). The advisory must stay
    silent: a port that legitimately ends in open space is not
    one-cell-short of anything. Geometry mirrors
    tests/unit/preflight/test_inverse_design_preflight.py's _clean_sim (which pins
    preflight() == [] on it, so this control is double-locked).
    Eight CPML layers and an explicit wire radius keep the open-vacuum control fully clean.
    """
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02),
                     dx=0.02 / 15, boundary="cpml", cpml_layers=8)
    sim.add_port(position=(0.0093, 0.0093, 0.0093), component="ez",
                 impedance=50.0,
                 waveform=GaussianPulse(f0=5e9, bandwidth=0.9),
                 extent=0.004,
                 radius=0.1e-3)

    issues = sim.preflight()
    codes = [getattr(s, "code", None) for s in issues]
    assert "wire_port_end_gap_to_conductor" not in codes, (
        "#556 advisory fired on a wire port ending in open vacuum -- "
        "this is the false-positive class the gating must exclude: "
        + "\n".join(str(s) for s in issues)
    )
    assert issues == [], (
        "open-vacuum wire-port fixture must stay fully clean (it also "
        "backs test_zero_issue_summary_line_is_honest): "
        + "\n".join(str(s) for s in issues)
    )
