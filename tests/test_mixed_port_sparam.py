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

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.api._sparams import _assemble_mixed_power_wave_s
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.sources.sources import GaussianPulse

_EPS_R = 3.66
_H_SUB = 254e-6
_W_TRACE = 600e-6
_DX = 80e-6


# ---------------------------------------------------------------------------
# Layer 1a — pure power-wave assembly falsifiers (no FDTD)
# ---------------------------------------------------------------------------

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
    # run 0 — lumped driven: a=1 ((-V+Z0 I)/(2 sqrt(Z0))), diag b=0.
    v_lw[0, 0] = -s50 * ones
    i_lw[0, 0] = ones / s50
    # run 0 — MSL receives unit power wave: (V0 - Zm I)/2 = sqrt(Zm).
    v0_msl[0, 0] = s30 * ones
    i_msl[0, 0] = -ones / s30
    # run 1 — MSL driven: (V0 + Zm I)/2/sqrt(Zm) = 1.
    v0_msl[1, 0] = s30 * ones
    i_msl[1, 0] = ones / s30
    # run 1 — lumped receives unit power wave: (V - Z0 I)/(2 sqrt(Z0)) = 1.
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
    pseudo_ratio = np.sqrt(z0_msl / z0_lump)  # 0.7746 — the #460 error
    assert abs(pseudo_ratio - 1.0) > 0.2
    np.testing.assert_allclose(np.abs(S[1, 0, :]), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.abs(S[0, 0, :]), 0.0, atol=1e-6)


def test_reciprocity_on_synthetic_transfer():
    """S12 == S21 on the synthetic reciprocal transfer (both = 1)."""
    v_lw, i_lw, v0_msl, i_msl = _synthetic_unit_transfer()
    S, _ = _assemble_mixed_power_wave_s(
        v_lw, i_lw, v0_msl, i_msl,
        np.asarray([50.0]), np.asarray([1]), np.asarray([30.0]),
        wire_mode=False, drive_plan=[("lw", 0), ("msl", 0)],
    )
    S = np.asarray(S)
    np.testing.assert_allclose(
        np.abs(S[0, 1, :]), np.abs(S[1, 0, :]), atol=1e-5
    )


def test_power_witness_matches_synthetic():
    """Delivered-power |a| reconstruction reproduces the unit transfer.

    P_del = 0.5*Re(Z_in)*|I|^2 = 0.5 at the matched synthetic drive,
    so |a|_recon = sqrt(2*P_del/(1-|S11|^2)) = 1 and the witness equals
    |b_msl_pw| = 1 (issue #313 triangulation channel).
    """
    v_lw, i_lw, v0_msl, i_msl = _synthetic_unit_transfer()
    _, s21_power = _assemble_mixed_power_wave_s(
        v_lw, i_lw, v0_msl, i_msl,
        np.asarray([50.0]), np.asarray([1]), np.asarray([30.0]),
        wire_mode=False, drive_plan=[("lw", 0), ("msl", 0)],
    )
    np.testing.assert_allclose(s21_power[0, 0, :], 1.0, atol=1e-5)


def test_wire_mode_uses_per_cell_impedance():
    """Wire off-diagonals normalize by Z0/n_live (issue #318 convention).

    Rebuilding the run-0 synthetic with Z0c = 50/5 in the wave formulas
    must again read |S21| = 1 under wire_mode with n_live = 5; feeding the
    LUMPED-convention phasors instead mis-normalizes by sqrt(n_live).
    """
    n_live, z0_lump, z0_msl = 5, 50.0, 30.0
    z0c = z0_lump / n_live
    sqc, s30 = np.sqrt(z0c), np.sqrt(z0_msl)
    ones = np.ones(3, dtype=np.complex128)
    v_lw = np.zeros((1, 1, 3), dtype=np.complex128)
    i_lw = np.zeros_like(v_lw)
    v0_msl = np.zeros_like(v_lw)
    i_msl = np.zeros_like(v_lw)
    v_lw[0, 0] = -sqc * ones          # a = (-V + Z0c I)/(2 sqrt(Z0c)) = 1
    i_lw[0, 0] = ones / sqc
    v0_msl[0, 0] = s30 * ones          # b_pw = 1 at the MSL port
    i_msl[0, 0] = -ones / s30
    S, _ = _assemble_mixed_power_wave_s(
        v_lw, i_lw, v0_msl, i_msl,
        np.asarray([z0_lump]), np.asarray([n_live]), np.asarray([z0_msl]),
        wire_mode=True, drive_plan=[("lw", 0)],
    )
    np.testing.assert_allclose(np.abs(np.asarray(S)[1, 0, :]), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Layer 1b — registration-guard contract (v1 envelope)
# ---------------------------------------------------------------------------

def _base_sim(**kw):
    lx, ly, lz = 8e-3, 3e-3, 754e-6
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
                (lx, y_c + _W_TRACE / 2, _H_SUB + _DX)), material="pec")
    return sim, y_c


def _add_msl(sim, y_c, x=5.5e-3, direction="-x", **kw):
    # Default facing is TOWARD the wire feed (issue #488 attempt-1 defect
    # D1: a port facing away from the DUT measures launch+return mixed
    # in its a-channel and its probe ladder walks into the CPML).
    sim.add_msl_port(position=(x, y_c, 0.0), width=_W_TRACE,
                     height=_H_SUB, direction=direction, impedance=50.0,
                     waveform=GaussianPulse(f0=2.5e9, bandwidth=0.5), **kw)


def _add_feed(sim, y_c, x=2e-3):
    # Vertical probe feed: ez wire port ground-plane -> trace bottom.
    sim.add_port(position=(x, y_c, 0.0), component="ez",
                 impedance=50.0, extent=_H_SUB)


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
    _add_feed(sim, y_c)                       # wire (extent set)
    sim.add_port(position=(1.5e-3, y_c, 4e-4), component="ez",
                 impedance=50.0)              # lumped (no extent)
    _add_msl(sim, y_c)
    with pytest.raises(NotImplementedError, match="lumped \\+ wire"):
        sim.compute_mixed_s_matrix(skip_preflight=True)


def test_flux_magnitude_override_math():
    """Arch-A unit falsifier: flux ratios replace off-diagonal magnitudes.

    P_net = 0.75 at a drive with |S11| = 0.5 gives P_inc = 1.0; a receive
    plane carrying 0.36 toward the port must read |S21| = 0.6 with the
    WAVE phase preserved and the diagonal untouched — regardless of how
    wrong the wave-channel magnitude was (the #313 class).
    """
    from rfx.api._sparams import _mixed_flux_magnitude_override
    n_freqs = 3
    S_wave = np.zeros((2, 2, n_freqs), dtype=np.complex64)
    S_wave[0, 0, :] = 0.5                       # validated diagonal
    S_wave[1, 0, :] = 1.7j                      # wrong magnitude, known phase
    S_wave[1, 1, :] = 0.1
    S_wave[0, 1, :] = -0.2                      # wrong magnitude, phase pi
    box_lw = np.zeros((2, 1, n_freqs))
    plane_msl = np.zeros((2, 1, n_freqs))
    box_lw[0, 0, :] = 0.75                      # run 0: lw driven, net out
    plane_msl[0, 0, :] = 0.36                   # +x toward the "-x" port
    plane_msl[1, 0, :] = -0.99                  # run 1: msl driven, away=-x
    box_lw[1, 0, :] = -0.25                     # inward 0.25 at the lw box
    S_out, ill, neg = _mixed_flux_magnitude_override(
        S_wave, box_lw, plane_msl, [("lw", 0), ("msl", 0)],
        msl_away_signs=[-1.0], n_lw=1,
    )
    S_out = np.asarray(S_out)
    np.testing.assert_allclose(S_out[0, 0, :], 0.5, atol=1e-6)   # untouched
    np.testing.assert_allclose(np.abs(S_out[1, 0, :]), 0.6, rtol=1e-5)
    np.testing.assert_allclose(np.angle(S_out[1, 0, :]), np.pi / 2, atol=1e-5)
    # msl-driven column: P_net = (-1)*(-0.99) = 0.99, |S22|=0.1 ->
    # P_inc = 1.0; lw receive inward = 0.25 -> |S12| = 0.5, phase pi.
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
    for s11_mag in (0.03, 0.41, 0.90):
        S_wave = np.zeros((2, 2, 1), dtype=np.complex64)
        S_wave[0, 0, :] = s11_mag
        S_wave[1, 1, :] = 0.5
        S_wave[1, 0, :] = 0.123          # arbitrary wrong wave magnitude
        box_lw = np.zeros((2, 1, 1))
        plane_msl = np.zeros((2, 1, 1))
        box_lw[0, 0, :] = 0.4            # net launched
        plane_msl[0, 0, :] = 0.4         # all of it arrives (lossless)
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
    # Power GAIN is still detectable — that is all this quantity tests.
    S_wave = np.zeros((2, 2, 1), dtype=np.complex64)
    S_wave[0, 0, :] = 0.0
    box_lw = np.zeros((2, 1, 1))
    plane_msl = np.zeros((2, 1, 1))
    box_lw[0, 0, :] = 0.4
    plane_msl[0, 0, :] = 0.8             # twice as much arrives: gain
    S_out, _, _ = _mixed_flux_magnitude_override(
        S_wave, box_lw, plane_msl, [("lw", 0)],
        msl_away_signs=[-1.0], n_lw=1,
    )
    assert float(np.abs(np.asarray(S_out)[1, 0, 0]) ** 2) > 1.5


def test_flux_override_masks_ill_conditioned_and_negative():
    from rfx.api._sparams import _mixed_flux_magnitude_override
    S_wave = np.zeros((2, 2, 2), dtype=np.complex64)
    S_wave[0, 0, :] = 0.999                     # near-total reflection
    box_lw = np.zeros((2, 1, 2))
    plane_msl = np.zeros((2, 1, 2))
    box_lw[0, 0, :] = -1e-3                     # negative net at the drive
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
    """
    from rfx.api._sparams import (
        _mixed_flux_magnitude_override,
        _mixed_reciprocity_deviation,
    )
    # Symmetric lossless 2-port: both ports launch 1.0 and receive 1.0.
    box_lw = np.zeros((2, 1, 1))
    plane_msl = np.zeros((2, 1, 1))
    # "-x" MSL port: away_sign = -1, so the code reads net launched power
    # as -plane and arriving power as +plane.
    box_lw[0, 0, :] = 1.0            # run 0: lw driven, net outward
    plane_msl[0, 0, :] = 1.0         # ...arrives at the msl plane
    plane_msl[1, 0, :] = -1.0        # run 1: msl driven, net away
    box_lw[1, 0, :] = -1.0           # ...arrives back at lw (inward)
    plan = [("lw", 0), ("msl", 0)]

    def run(s11, s22):
        S = np.zeros((2, 2, 1), dtype=np.complex64)
        S[0, 0, :], S[1, 1, :] = s11, s22
        out, _, _ = _mixed_flux_magnitude_override(
            S, box_lw, plane_msl, plan, msl_away_signs=[-1.0], n_lw=1,
        )
        return _mixed_reciprocity_deviation(out)[1]

    # Matching diagonals -> reciprocity holds.
    assert run(0.4, 0.4) == pytest.approx(0.0, abs=1e-5)
    # A wrong diagonal on ONE port breaks it, and by a large margin.
    assert run(0.4, 0.03) > 0.05


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
    y_c = ly / 2.0
    sim.add(Box((0.0, y_c - _W_TRACE / 2, _H_SUB),
                (lx, y_c + _W_TRACE / 2, _H_SUB + _DX)), material="pec")
    _add_feed(sim, y_c)
    _add_msl(sim, y_c, n_probe_offset=10, n_probe_spacing=4)
    with pytest.raises(NotImplementedError, match="PEC z_lo"):
        sim.compute_mixed_s_matrix(skip_preflight=True)
    # The wave channel makes no such assumption and must still build.
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
    box_lw[0, 0, :] = 1.0            # healthy driven net power
    plane_msl[0, 0, :] = -1.0        # mis-signed: reads as arriving < 0
    S_out, _, neg = _mixed_flux_magnitude_override(
        S_wave, box_lw, plane_msl, [("lw", 0)],
        msl_away_signs=[-1.0], n_lw=1,
    )
    assert neg[1].all(), "receive-side negative power went untracked"
    assert float(np.abs(np.asarray(S_out)[1, 0, 0])) == 0.0


def test_forward_does_not_pay_for_flux_monitors(monkeypatch):
    """Registering flux monitors must not change the plain forward() lane.

    `ForwardResult` has never carried flux monitors, so accumulating them
    inside the differentiable scan would be pure cost (including AD-tape
    memory) with no visible result — a silent regression for existing
    AD callers. The build is therefore gated to the raw-hook consumer.
    """
    import jax.numpy as jnp
    import rfx.runners.uniform as _uni

    calls = []
    real = _uni.build_flux_monitor_cfgs
    monkeypatch.setattr(
        _uni, "build_flux_monitor_cfgs",
        lambda *a, **k: (calls.append(1), real(*a, **k))[1],
    )
    sim = Simulation(freq_max=10e9, domain=(4e-3, 2e-3, 1e-3),
                     dx=100e-6, cpml_layers=4)
    sim.add_port(position=(1e-3, 1e-3, 0.5e-3), component="ez",
                 impedance=50.0)
    sim.add_flux_monitor(axis="x", coordinate=2e-3,
                         freqs=jnp.asarray(np.array([5e9])))
    sim.forward(n_steps=40, port_s11_freqs=np.array([5e9]),
                skip_preflight=True)
    assert calls == [], "forward() built flux monitors it cannot return"


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


# ---------------------------------------------------------------------------
# Layer 1c — end-to-end plumbing smoke (NOT a physics gate)
# ---------------------------------------------------------------------------

def test_mixed_probe_fed_msl_plumbing_smoke():
    """Coarse probe-fed MSL line runs end to end and fills the result.

    num_periods=4 is deliberately truncated — this asserts BOOKKEEPING
    (shapes, port order, finiteness, witnesses recorded), not physics.
    The claims-bearing battery with the pre-declared falsifiers is the
    slow-lane fixture battery (issue #488 arc).
    """
    sim, y_c = _base_sim()
    _add_feed(sim, y_c, x=2e-3)
    _add_msl(sim, y_c, x=5.5e-3, n_probe_offset=10, n_probe_spacing=4)
    freqs = np.linspace(1e9, 4e9, 5)
    with pytest.warns(UserWarning):
        # The truncated record must surface the ring-down warning — its
        # absence on a 4-period open-domain record would itself be a bug.
        res = sim.compute_mixed_s_matrix(
            freqs=freqs, num_periods=4.0, skip_preflight=True,
        )
    S = np.asarray(res.S)
    assert S.shape == (2, 2, 5)
    assert np.all(np.isfinite(S))
    assert res.port_families == ("wire", "msl")
    assert res.port_names[1] != "lw0" and res.port_names[0] == "lw0"
    assert res.z0_ref.shape == (2,)
    assert res.z0_ref[0] == pytest.approx(50.0)
    assert 20.0 < res.z0_ref[1] < 120.0          # HJ analytic, sane band
    assert res.s21_power_witness.shape == (1, 1, 5)
    assert np.all(np.isfinite(res.s21_power_witness))
    assert res.magnitude_channel == "flux"
    assert res.S_wave is not None and res.S_wave.shape == S.shape
    assert res.settling_db is not None and np.all(np.isfinite(res.settling_db))
    # Registration state restored after the drive loop.
    assert len(sim._dft_planes) == 0
    assert len(sim._probes) == 0
    assert sim._msl_ports[0].excite is True
    assert sim._ports[0].excite is True
