"""Validation battery for the opt-in reference-plane port waves (issue #313).

Phase-1 lanes for ``add_port(reference_plane_cells=N)`` — the
REFERENCE-PLANE architecture promoted by the issue #313 Phase-0 verdict
(closed-box flux referee, 2026-07-10):

FAST (no FDTD / tiny FDTD):
  * extraction-math unit tests on synthetic two-wave line phasors
    (two-plane Zc invariant, measured beta, split, phase-only de-embed,
    byte-frozen legacy diagonals);
  * add_port opt-in validation (wire-only, explicit transverse direction);
  * plumbing contract on the canonical thru (two planes per opted port,
    Phase-0-exact plane geometry, accumulator shapes);
  * DEFAULT-PATH invariance: no opt-in => no plane registration, and
    forward() is bitwise unaffected by the opt-in (its diagonal path
    never touches the planes);
  * short-run run(compute_s_params=True) wiring: diagonals bitwise equal
    to the default path (byte-frozen), off-diagonals move (loud).

SLOW (canonical 16 mm thru, 4000 steps, production driver): the plane-path
measurement lane with gates anchored to the Phase-0 closed-box flux
referee. LOUD RE-BASELINE (issue #313 falsifier item 8): the committed
legacy battery |S21| regression-lock band [0.35, 0.85]
(tests/test_lumped_twoport_vi_validation_battery.py) is EXPECTED to fail
on the plane path — the shipped drive-side deflation kappa(f)=1.49-1.86
disappears and |S21| moves to the flux-implied ~0.96-1.0.  That legacy
band stays committed and green for the DEFAULT path; this module carries
the plane-path gates with their own referee-anchored labels, and asserts
explicitly that the plane path leaves the legacy band (proof the fix
moves the number).

Phase-0 referee numbers used as anchors (measured 2026-07-10, closed
6-face flux boxes; see issue #313 Phase-0 comment):
  |S21|_referee = sqrt(P_abs/P_launch) = 0.98431..1.00660 per bin
  reference-plane arch |S21| residual vs referee: -1.18%..-2.54%
  Zc two-plane (mid-line 13/19 mm planes): 47.85..48.63 ohm, Im/Re<=0.69%
  beta/(w/c): 1.048..1.061 (slow-wave, far above numerical dispersion)

HONESTY LABELS: the plane-path |S21| gates below are frozen from a
measured run of THIS implementation (per-port planes at N=3 / 2N=6 cells
outboard — nearer the ports than the Phase-0 mid-line Zc planes) and are
labelled with their distance to the Phase-0 referee.  They are
regression + referee-consistency gates on the canonical thru, NOT an
external cross-solver validation (the openEMS thru remains the final
gate per the issue #313 falsifier list).

No module-level x64 flip (documented complex64 envelope).
"""

from __future__ import annotations

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.sources.sources import GaussianPulse
from rfx.probes.probes import decompose_wire_s_matrix
from rfx.probes.refplane import (
    build_wire_refplane_specs,
    decompose_wire_s_matrix_with_reference_planes,
    refplane_beta,
    refplane_centered_current,
    refplane_split,
    refplane_zc_two_plane,
)

C0 = 299792458.0

# --- canonical thru fixture constants (byte-exact battery geometry) ---
_DX = 0.5e-3
_DOMAIN = (0.032, 0.020, 0.010)
_H = 1.0e-3
_W = 5.0e-3
_X1, _X2 = 0.008, 0.024
_Y_MID = _DOMAIN[1] / 2
_L = _X2 - _X1
_FREQS = np.linspace(3e9, 7e9, 9)
_N_STEPS = 4000
_PEC_FACES_ADVISORY_SNIPPET = "INFINITE PEC boundary"


def _build_thru(reference_plane_cells: int | None = None) -> Simulation:
    """The committed battery thru, optionally opted into the plane path."""
    sim = Simulation(
        freq_max=10e9, domain=_DOMAIN, dx=_DX,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
        cpml_layers=8,
    )
    sim.add(
        Box((_X1 - _DX, _Y_MID - _W / 2, _H),
            (_X2 + _DX, _Y_MID + _W / 2, _H + _DX)),
        material="pec",
    )
    pulse = GaussianPulse(f0=5e9, bandwidth=0.8)
    kw = {}
    if reference_plane_cells is not None:
        kw["reference_plane_cells"] = reference_plane_cells
    sim.add_port(position=(_X1, _Y_MID, 0.0), component="ez", impedance=50.0,
                 extent=_H, waveform=pulse, direction="-x", **kw)
    sim.add_port(position=(_X2, _Y_MID, 0.0), component="ez", impedance=50.0,
                 extent=_H, waveform=pulse, direction="+x", **kw)
    return sim


# ===========================================================================
# FAST — extraction math on synthetic two-wave line phasors
# ===========================================================================

def _synthetic_line(freqs, zc_true, beta, dt):
    """Helpers for a clean two-wave uniform line in DFT phasor convention.

    Forward (+x) wave ``f0 e^{-j beta x}``, backward ``g0 e^{+j beta x}``;
    V = f + g, I(+x) = (f - g)/Zc.  Raw loop currents are synthesized so
    that ``refplane_centered_current`` (average + exp(+j w dt/2)) returns
    the exact line current.
    """
    w = 2 * np.pi * np.asarray(freqs, dtype=np.float64)
    hcorr = np.exp(+1j * w * dt / 2)

    def field(x, f0, g0):
        f = f0 * np.exp(-1j * beta * x)
        g = g0 * np.exp(+1j * beta * x)
        return f + g, (f - g) / zc_true

    def raw(i_line):
        return i_line / hcorr, i_line / hcorr

    return field, raw


def test_zc_beta_split_deembed_recover_synthetic_line():
    """Two-plane Zc / measured beta / split / de-embed are exact on a
    clean two-wave line (the algebra the scan accumulators feed)."""
    freqs = _FREQS
    w = 2 * np.pi * freqs
    dt = 9.6e-13
    dx = _DX
    zc_true = 48.3
    beta_true = 1.055 * w / C0     # slow-wave class, per Phase 0
    n = 3
    d = n * dx
    field, raw = _synthetic_line(freqs, zc_true, beta_true, dt)

    f0 = (1.0 + 0.3j) * np.ones_like(w, dtype=np.complex128)
    g0 = 0.18 * np.exp(1j * 0.4) * f0          # backward (load reflection)

    v1, i1 = field(d, f0, g0)
    v2, i2 = field(2 * d, f0, g0)
    im1, ip1 = raw(i1)
    im2, ip2 = raw(i2)
    i1c = refplane_centered_current(im1, ip1, freqs, dt)
    i2c = refplane_centered_current(im2, ip2, freqs, dt)
    assert np.max(np.abs(i1c - i1)) < 1e-12

    zc = refplane_zc_two_plane(v1, i1c, v2, i2c)
    assert np.max(np.abs(zc - zc_true)) < 1e-9, (
        f"two-plane Zc invariant broke: {zc}")

    out1, in1 = refplane_split(v1, i1c, zc, outboard_sign=+1)
    out2, _ = refplane_split(v2, i2c, zc, outboard_sign=+1)
    # outgoing = the forward (+x) wave at the plane; incoming = backward
    assert np.max(np.abs(out1 - f0 * np.exp(-1j * beta_true * d))) < 1e-9
    assert np.max(np.abs(in1 - g0 * np.exp(+1j * beta_true * d))) < 1e-9

    beta = refplane_beta(out1, out2, d)
    assert np.max(np.abs(beta / beta_true - 1)) < 1e-12, (
        f"measured beta wrong: {beta / beta_true}")

    # phase-only de-embed back to the port plane (x=0)
    a_port = out1 * np.exp(+1j * beta * d)
    assert np.max(np.abs(a_port - f0)) < 1e-9


def _two_port_synthetic(gamma_load):
    """Full synthetic 2-port plane dataset (drive 0 + mirrored drive 1)."""
    freqs = _FREQS
    w = 2 * np.pi * freqs
    dt = 9.6e-13
    dx = _DX
    zc_true = 48.3
    beta_true = 1.055 * w / C0
    n = 3
    d = n * dx
    L = _L
    field, raw = _synthetic_line(freqs, zc_true, beta_true, dt)

    f0 = (1.0 + 0.3j) * np.ones_like(w, dtype=np.complex128)
    g0 = gamma_load * f0 * np.exp(-2j * beta_true * L)

    n_ports, n_freqs = 2, len(freqs)
    plane_v = np.zeros((n_ports, n_ports, 2, n_freqs), dtype=np.complex128)
    plane_im = np.zeros_like(plane_v)
    plane_ip = np.zeros_like(plane_v)

    def fill(j, p, slot, x, sign):
        v, i = field(x, f0, g0)
        im, ip = raw(sign * i)
        plane_v[j, p, slot] = v
        plane_im[j, p, slot] = im
        plane_ip[j, p, slot] = ip

    # drive 0: port0 (x=0, outboard +x) planes at d/2d; port1 at L-d/L-2d
    fill(0, 0, 0, d, +1)
    fill(0, 0, 1, 2 * d, +1)
    fill(0, 1, 0, L - d, +1)
    fill(0, 1, 1, L - 2 * d, +1)
    # drive 1: mirror x -> L - x; I(+x) flips sign under the mirror
    fill(1, 1, 0, d, -1)
    fill(1, 1, 1, 2 * d, -1)
    fill(1, 0, 0, L - d, -1)
    fill(1, 0, 1, L - 2 * d, -1)

    rng = np.random.default_rng(20260710)
    v_all = (rng.normal(size=(2, 2, n_freqs))
             + 1j * rng.normal(size=(2, 2, n_freqs)))
    i_all = (rng.normal(size=(2, 2, n_freqs))
             + 1j * rng.normal(size=(2, 2, n_freqs)))
    z0 = np.array([50.0, 50.0])
    counts = np.array([3, 3])
    return dict(
        freqs=freqs, dt=dt, dx=dx, n=n, beta_true=beta_true,
        zc_true=zc_true, L=L,
        plane_v=plane_v, plane_im=plane_im, plane_ip=plane_ip,
        v_all=v_all, i_all=i_all, z0=z0, counts=counts,
    )


def test_decompose_refplane_two_port_synthetic_matched():
    """Full wrapper on a matched synthetic thru: diagonals BITWISE equal
    the legacy decomposer; S21 = S12 = exp(-j beta L) at complex64
    rounding; measured Zc/beta recovered."""
    s = _two_port_synthetic(gamma_load=0.0)
    S_legacy = np.asarray(decompose_wire_s_matrix(
        s["v_all"], s["i_all"], s["z0"], s["counts"]), dtype=np.complex64)
    S, diag = decompose_wire_s_matrix_with_reference_planes(
        s["v_all"], s["i_all"], s["z0"], s["counts"],
        plane_v=s["plane_v"], plane_im=s["plane_im"], plane_ip=s["plane_ip"],
        plane_enabled=np.array([True, True]),
        plane_offsets=np.array([s["n"], s["n"]]),
        outboard_signs=np.array([1, -1]),
        freqs=s["freqs"], dt=s["dt"], dx=s["dx"],
        return_line_diagnostics=True,
    )
    # DIAGONAL: byte-frozen legacy path, always (issue #313 hard rule).
    assert S[0, 0].tobytes() == S_legacy[0, 0].tobytes()
    assert S[1, 1].tobytes() == S_legacy[1, 1].tobytes()
    for p in (0, 1):
        assert np.max(np.abs(diag["zc"][p] - s["zc_true"])) < 1e-6
        assert np.max(np.abs(diag["beta"][p] / s["beta_true"] - 1)) < 1e-9
    expect = np.exp(-1j * s["beta_true"] * s["L"])
    assert np.max(np.abs(S[1, 0] - expect)) < 1e-6
    assert np.max(np.abs(S[0, 1] - expect)) < 1e-6
    # R5 wave-pair diagnostics (Phase-0 falsifier F2 surface): the
    # at-plane pairs are exposed and the zero-free-parameter conservation
    # identity (|out|^2 - |inc|^2)/Re(Zc) is plane-invariant (exact on the
    # synthetic lossless line; matched load => incoming ~ 0 at the drive).
    d = s["n"] * s["dx"]
    out00, in00 = diag["plane_waves"][(0, 0, 0)]
    assert np.max(np.abs(in00)) < 1e-12
    assert np.max(np.abs(
        out00 - (1.0 + 0.3j) * np.exp(-1j * s["beta_true"] * d))) < 1e-9
    p_net_0 = (np.abs(out00) ** 2 - np.abs(in00) ** 2) / diag["zc"][0].real
    out10, in10 = diag["plane_waves"][(0, 1, 0)]
    p_net_1 = (np.abs(in10) ** 2 - np.abs(out10) ** 2) / diag["zc"][1].real
    # power flows drive -> load: at the receive plane the port-incoming
    # wave carries it, and on the lossless line the two planes agree.
    assert np.max(np.abs(p_net_1 / p_net_0 - 1)) < 1e-9


def test_decompose_refplane_two_port_synthetic_reflective():
    """Same wrapper with a reflective load (|Gamma|=0.18, the measured
    Phase-0 port mismatch class): the incoming/outgoing separation must
    still recover the exact thru transfer."""
    s = _two_port_synthetic(gamma_load=0.18 * np.exp(1j * 0.4))
    S = decompose_wire_s_matrix_with_reference_planes(
        s["v_all"], s["i_all"], s["z0"], s["counts"],
        plane_v=s["plane_v"], plane_im=s["plane_im"], plane_ip=s["plane_ip"],
        plane_enabled=np.array([True, True]),
        plane_offsets=np.array([s["n"], s["n"]]),
        outboard_signs=np.array([1, -1]),
        freqs=s["freqs"], dt=s["dt"], dx=s["dx"],
    )
    expect = np.exp(-1j * s["beta_true"] * s["L"])
    assert np.max(np.abs(S[1, 0] - expect)) < 1e-6
    assert np.max(np.abs(S[0, 1] - expect)) < 1e-6


def test_decompose_refplane_requires_both_ports_opted():
    """A pair with only ONE opted port keeps the legacy off-diagonals
    (F1 / partial composition intentionally OUT of this phase)."""
    s = _two_port_synthetic(gamma_load=0.0)
    S_legacy = np.asarray(decompose_wire_s_matrix(
        s["v_all"], s["i_all"], s["z0"], s["counts"]), dtype=np.complex64)
    S = decompose_wire_s_matrix_with_reference_planes(
        s["v_all"], s["i_all"], s["z0"], s["counts"],
        plane_v=s["plane_v"], plane_im=s["plane_im"], plane_ip=s["plane_ip"],
        plane_enabled=np.array([True, False]),
        plane_offsets=np.array([s["n"], 0]),
        outboard_signs=np.array([1, -1]),
        freqs=s["freqs"], dt=s["dt"], dx=s["dx"],
    )
    assert S.tobytes() == S_legacy.tobytes()


# ===========================================================================
# FAST — add_port opt-in validation
# ===========================================================================

def _base_sim():
    return Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=1e-3,
                      boundary="cpml", cpml_layers=6)


def test_add_port_refplane_rejects_lumped():
    with pytest.raises(NotImplementedError, match="wire ports"):
        _base_sim().add_port(position=(0.01, 0.01, 0.01), component="ez",
                             impedance=50.0, direction="-x",
                             reference_plane_cells=3)


def test_add_port_refplane_requires_direction():
    with pytest.raises(ValueError, match="explicit direction"):
        _base_sim().add_port(position=(0.01, 0.01, 0.0), component="ez",
                             impedance=50.0, extent=1e-3,
                             reference_plane_cells=3)


def test_add_port_refplane_rejects_direction_along_component():
    with pytest.raises(ValueError, match="transverse"):
        _base_sim().add_port(position=(0.01, 0.01, 0.0), component="ex",
                             impedance=50.0, extent=1e-3, direction="+x",
                             reference_plane_cells=3)


def test_add_port_refplane_rejects_nonpositive_n():
    with pytest.raises(ValueError, match=">= 1"):
        _base_sim().add_port(position=(0.01, 0.01, 0.0), component="ez",
                             impedance=50.0, extent=1e-3, direction="-x",
                             reference_plane_cells=0)


def test_nonuniform_lane_guard_fails_loudly():
    """Reference-plane ports on the NU/subgridded lanes raise instead of
    silently returning legacy off-diagonals."""
    sim = _build_thru(reference_plane_cells=3)
    with pytest.raises(NotImplementedError, match="uniform single-device"):
        sim._reject_refplane_ports_off_uniform_lane("non-uniform mesh", None)
    # explicit compute_s_params=False is allowed (no S-matrix requested)
    sim._reject_refplane_ports_off_uniform_lane("non-uniform mesh", False)
    # and a sim without the opt-in is never blocked
    _build_thru()._reject_refplane_ports_off_uniform_lane(
        "non-uniform mesh", None)


# ===========================================================================
# FAST — plumbing contract on the canonical thru (tiny FDTD)
# ===========================================================================

def _raw_drive(sim, n_steps=8, drive_idx=0):
    grid = sim._build_grid()
    mats, dsp, lsp, pm, _, _, _ = sim._assemble_materials(grid)
    return sim._forward_from_materials(
        grid, mats, dsp, lsp, n_steps=n_steps, checkpoint=False,
        pec_mask=pm, port_s11_freqs=_FREQS,
        _sparam_drive_idx=drive_idx, _return_raw_port_sparams=True)


def test_refplane_registers_two_planes_per_port_with_phase0_geometry():
    """TWO planes per opted port with the Phase-0-exact geometry.

    Hand-derived indices for the canonical thru (dx=0.5mm, CPML pad 8 on
    x/y, PEC z_lo): port 1 at x=8mm -> i=24; planes at 9.5/11.0mm ->
    27/30; port 2 at 24mm -> i=56; planes at 22.5/21.0mm -> 53/50.  Ampere
    loop legs half a cell outside the trace bbox (y 7.5..12.5mm -> j
    23..32 padded; z 1.0..1.5mm -> k 2): Hz columns at j=22/33 spanning
    k=[2,4), Hy rows at k=1/3 spanning j=[23,34) — the exact Phase-0
    probe layout (x=9.5mm plane: legs at y=7.25/12.75mm, z=0.75/1.75mm).
    """
    raw = _raw_drive(_build_thru(reference_plane_cells=3))
    rp = raw["wire_refplane"]
    assert rp is not None and len(rp) == 4, (
        "expected 2 ports x 2 planes registered")
    by_key = {(s.port_index, s.plane_slot): s for s, _ in rp}
    assert set(by_key) == {(0, 0), (0, 1), (1, 0), (1, 1)}

    expected_plane_index = {(0, 0): 27, (0, 1): 30, (1, 0): 53, (1, 1): 50}
    for key, spec in by_key.items():
        assert spec.plane_index == expected_plane_index[key], (
            f"plane index drifted for {key}: {spec.plane_index}")
        assert spec.line_axis == 0 and spec.comp_axis == 2
        assert spec.outboard_sign == (+1 if key[0] == 0 else -1)
        assert spec.n_cells_outboard == (3 if key[1] == 0 else 6)
        # GAP-trimmed V cells: the port extent is 3 cells (k=0,1,2) but
        # k=2 lies inside the PEC trace at the plane column and its Ez
        # edge is NOT mask-zeroed (measured |ez2/ez0| ~ 0.21) — the gap
        # line integral spans the 2 live cells, exactly the Phase-0
        # 2-cell integral that measured Zc = 47.9-48.6 ohm.
        assert (spec.e_lo, spec.e_hi) == (0, 2)
        assert spec.third_index == 28                # y = 10mm (padded)
        assert (spec.u_lo_leg, spec.u_hi_leg) == (22, 33)
        assert (spec.v_lo_leg, spec.v_hi_leg) == (1, 3)
        assert (spec.u_span_lo, spec.u_span_hi) == (23, 34)
        assert (spec.v_span_lo, spec.v_span_hi) == (2, 4)
        assert spec.hu_component == "hy" and spec.hv_component == "hz"

    for _, accs in rp:
        assert len(accs) == 3
        for a in accs:
            assert np.asarray(a).shape == (len(_FREQS),)
            assert np.all(np.isfinite(np.asarray(a)))


def test_default_path_registers_no_planes():
    raw = _raw_drive(_build_thru())
    assert raw["wire_refplane"] is None


def test_refplane_crossing_guard_rejects_planes_past_other_port():
    """N large enough to reach past the far port must fail loudly."""
    sim = _build_thru(reference_plane_cells=17)   # 2N=34 cells > 32-cell line
    with pytest.raises(ValueError, match="reach past another port"):
        _raw_drive(sim)


def test_refplane_requires_pec_trace_at_plane():
    """Planes pointing AWAY from the line (wrong direction) find no trace
    cross-section and must fail loudly, not measure vacuum."""
    sim = Simulation(
        freq_max=10e9, domain=_DOMAIN, dx=_DX,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
        cpml_layers=8,
    )
    sim.add(
        Box((_X1 - _DX, _Y_MID - _W / 2, _H),
            (_X2 + _DX, _Y_MID + _W / 2, _H + _DX)),
        material="pec",
    )
    pulse = GaussianPulse(f0=5e9, bandwidth=0.8)
    # direction "+x" on port 1: outboard becomes -x, off the trace end
    sim.add_port(position=(_X1, _Y_MID, 0.0), component="ez", impedance=50.0,
                 extent=_H, waveform=pulse, direction="+x",
                 reference_plane_cells=5)
    sim.add_port(position=(_X2, _Y_MID, 0.0), component="ez", impedance=50.0,
                 extent=_H, waveform=pulse, direction="+x",
                 reference_plane_cells=3)
    with pytest.raises(ValueError, match="no PEC conductor"):
        _raw_drive(sim)


def test_forward_bitwise_unaffected_by_optin():
    """forward(port_s11_freqs=...) is BITWISE identical with and without
    the opt-in: the plane registration is gated to the S-matrix driver
    path and forward()'s diagonal extraction never touches it."""
    f_default = _build_thru().forward(n_steps=64, port_s11_freqs=_FREQS,
                                      skip_preflight=True)
    f_opted = _build_thru(reference_plane_cells=3).forward(
        n_steps=64, port_s11_freqs=_FREQS, skip_preflight=True)
    a = np.asarray(f_default.s_params)
    b = np.asarray(f_opted.s_params)
    assert a.shape == b.shape and a.tobytes() == b.tobytes(), (
        "forward() S11 changed under reference_plane_cells — the opt-in "
        "must not touch the forward()/diagonal path")


def test_run_short_diagonals_byte_frozen_offdiagonals_move():
    """run(compute_s_params=True) wiring at 64 steps: the opted run's
    DIAGONALS are bitwise equal to the default run (byte-frozen legacy
    path) while the off-diagonals move (the plane path is actually live).
    64 steps is a plumbing witness, not physics — magnitudes are gated in
    the SLOW lane below."""
    r_default = _build_thru().run(
        n_steps=64, compute_s_params=True, s_param_freqs=_FREQS,
        skip_preflight=True)
    r_opted = _build_thru(reference_plane_cells=3).run(
        n_steps=64, compute_s_params=True, s_param_freqs=_FREQS,
        skip_preflight=True)
    S0 = np.asarray(r_default.s_params)
    S1 = np.asarray(r_opted.s_params)
    assert S0.shape == S1.shape == (2, 2, len(_FREQS))
    assert S1[0, 0].tobytes() == S0[0, 0].tobytes(), "S11 diagonal moved"
    assert S1[1, 1].tobytes() == S0[1, 1].tobytes(), "S22 diagonal moved"
    assert S1[1, 0].tobytes() != S0[1, 0].tobytes(), (
        "S21 did not move — plane path not live on run()")
    assert S1[0, 1].tobytes() != S0[0, 1].tobytes(), (
        "S12 did not move — plane path not live on run()")


def test_driver_vi_dump_with_planes_fails_loudly():
    from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan
    sim = _build_thru(reference_plane_cells=3)
    with pytest.raises(NotImplementedError, match="return_vi_dump"):
        compute_lumped_wire_s_matrix_via_scan(
            sim, _FREQS, n_steps=64, return_vi_dump=True)


# ===========================================================================
# SLOW physics battery (opt-in: -m slow_physics) — plane path on the thru
# ===========================================================================
#
# Fixture choice N=10: both planes (10 and 20 cells outboard, x=13/18 mm
# from port 1 and 19/14 mm from port 2) sit >= 10 cells from every port —
# the Phase-0 pre-registration rule for the two-plane Zc/beta measurement.
# The rejected first candidate N=3 (planes 3/6 cells) was measured
# near-field contaminated (R2 stop -> placement conformed to the Phase-0
# clean zone; one redesign, evidence-anchored). Current-code (gap-trimmed
# V) N=3 battery numbers, 2026-07-10: beta/(w/c) 1.16-1.20 (vs mid-line
# 1.046-1.059), Zc 52.0-53.1 ohm with Im/Re to 8.2%, |S21| closed-box
# referee residual -3.1% at 7 GHz, row energy to 1.019.  (The pre-trim
# driver had measured beta 1.21-1.25 / Im/Re 9.2% / resid -6.8% at the
# same placement — same class, same verdict.)
_REFPLANE_N = 10

# Phase-0 closed-box flux referee |S21| = sqrt(P_abs/P_launch), measured
# 2026-07-10 (leak-free 6-face boxes; issue #313 Phase-0 comment).
_REFEREE_S21 = np.array([1.0066, 1.0052, 1.0033, 1.0007, 0.99775,
                         0.99441, 0.99093, 0.98751, 0.98431])

# Measured plane-path values (THIS implementation, gap-trimmed V, N=10,
# 4000 steps, 2026-07-10):
#   |S21| = 0.98251..0.99840; |S21|/referee - 1 = -0.82%..-0.18% per bin
#   reciprocity rel <= 0.38%; Zc Re 47.94..48.62 ohm (both ports),
#   Im/Re <= 1.2%; beta/(w/c) = 1.0465..1.0589;
#   |arg(S21) + beta_meas*L| <= 8.4e-4 rad; max singular value of the
#   mixed matrix 1.0299 max; |S11|^2+|S21|^2 <= 1.0003.
# Gates carry margin over these measured values and are labelled with
# their referee anchors. They are referee-consistency + regression gates
# on the canonical thru, NOT external cross-solver validation.
_S21_REFEREE_RESID_BAND = (-0.025, 0.010)   # Phase-0 arch class <= 2.5%
_S21_LEGACY_BAND_HI = 0.85                  # committed legacy lock band hi
_RECIP_REL_MAX = 0.02
_ZC_RE_BAND = (46.0, 50.5)     # measured 47.9-48.6; Phase-0 mid-line
                               # 47.85-48.63; Phase-0 pair-dependence
                               # spread across plane pairs 44.5-51.0
_ZC_IM_OVER_RE_MAX = 0.03      # measured <= 0.012
_BETA_OVER_WC_BAND = (1.03, 1.08)   # measured 1.0465-1.0589. Mechanism
# check (2026-07-10, one dx/2 rerun at fixed physical geometry and
# identical physical plane locations on a shortened 12 mm line): the
# slow-wave excess is dx-STABLE — 0.0781 (dx=0.5mm) -> 0.0756 (dx=0.25mm),
# ratio 0.97, nowhere near the halving a staircase/dispersion owner
# would show. The beta/(w/c) = 1.05-1.08 slow wave is PHYSICAL for this
# open air-line-over-ground geometry (finite-thickness trace, non-TEM
# fringing), not a discretization artefact. (Measured Zc showed mild dx
# sensitivity, 48.5 -> 47.0 ohm at dx/2; beta did not.)
_PHASE_DEEMBED_MAX_RAD = 0.02  # measured <= 8.4e-4 (~24x margin)
_MIXED_SV_MAX = 1.05           # measured 1.0299 — see honesty label
_ENERGY_ROW_MAX = 1.02         # measured |S11|^2+|S21|^2 <= 1.0003


@pytest.fixture(scope="module")
def refplane_thru():
    """Opted thru through the production driver (~2.5 min);
    quotes preflight verbatim; returns (S complex128, diagnostics)."""
    from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan
    sim = _build_thru(reference_plane_cells=_REFPLANE_N)
    issues = [str(i) for i in sim.preflight()]
    for msg in issues:
        print(f"\n[refplane thru] preflight (verbatim): {msg}")
    # Exactly the one known advisory (the infinite ground plane IS the
    # microstrip return). Anything else = fixture drift, stop.
    assert len(issues) == 1 and _PEC_FACES_ADVISORY_SNIPPET in issues[0], (
        f"refplane thru preflight drifted from the baseline: {issues}")
    S, freqs, diag = compute_lumped_wire_s_matrix_via_scan(
        sim, _FREQS, n_steps=_N_STEPS, return_refplane_diagnostics=True)
    S = np.asarray(S).astype(np.complex128)
    assert S.shape == (2, 2, len(_FREQS))
    assert np.all(np.isfinite(S))
    with np.printoptions(precision=5, suppress=False):
        print(f"[refplane thru] |S21|={np.abs(S[1, 0])}")
        print(f"[refplane thru] |S21|/referee={np.abs(S[1, 0]) / _REFEREE_S21}")
        print(f"[refplane thru] Zc0={diag['zc'][0]}")
        print(f"[refplane thru] Zc1={diag['zc'][1]}")
        w = 2 * np.pi * _FREQS
        print(f"[refplane thru] beta0/(w/c)={diag['beta'][0] / (w / C0)}")
    return S, diag


@pytest.mark.slow_physics
def test_refplane_thru_s21_tracks_box_referee(refplane_thru):
    """Plane-path |S21| within the Phase-0 closed-box referee class.

    HONESTY LABEL: gate band [-2.5%, +1%] per bin against the Phase-0
    referee = the Phase-0 reference-plane-architecture residual class
    (arch |S21| landed within 2.5% of the box referee at all bins);
    measured here -0.82%..-0.18%. Referee-consistency on the canonical
    thru — the external openEMS thru remains the final gate (#313)."""
    S, _ = refplane_thru
    resid = np.abs(S[1, 0]) / _REFEREE_S21 - 1.0
    lo, hi = _S21_REFEREE_RESID_BAND
    assert np.all(resid >= lo) and np.all(resid <= hi), (
        f"plane-path |S21| left the Phase-0 referee class: resid={resid}")


@pytest.mark.slow_physics
def test_refplane_thru_leaves_legacy_lock_band_loudly(refplane_thru):
    """LOUD RE-BASELINE (issue #313 falsifier item 8): the plane path
    must move |S21| OUT of the committed legacy regression-lock band
    ([0.35, 0.85] in test_lumped_twoport_vi_validation_battery.py, the
    shipped kappa(f)=1.49-1.86 deflation envelope). If this fails, the
    plane path stopped moving the number and the whole lane is void."""
    S, _ = refplane_thru
    s21 = np.abs(S[1, 0])
    assert s21.min() > _S21_LEGACY_BAND_HI, (
        f"plane-path |S21| fell back into the legacy deflation band: "
        f"min={s21.min():.4f} <= {_S21_LEGACY_BAND_HI} — the drive-side "
        "kappa deflation returned (issue #313)")


@pytest.mark.slow_physics
def test_refplane_thru_reciprocity(refplane_thru):
    """S21 vs S12 on the symmetric thru (measured <= 0.38% rel)."""
    S, _ = refplane_thru
    rel = np.abs(S[1, 0] - S[0, 1]) / np.abs(S[1, 0])
    assert np.all(rel <= _RECIP_REL_MAX), f"reciprocity broke: {rel}"


@pytest.mark.slow_physics
def test_refplane_thru_measured_line_constants(refplane_thru):
    """Measured Zc and beta land in the Phase-0 mid-line class.

    Zc gate [46.0, 50.5] ohm: measured 47.94-48.62 here; Phase-0
    mid-line pair (13/19 mm) measured 47.85-48.63; the Phase-0 data's
    pair-to-pair spread across plane pairs is 44.5-51.0 ohm (the open
    radiating microstrip is not a perfect two-wave line), so the band is
    a placement-sensitive consistency gate, not a universal constant.
    beta/(w/c) gate [1.03, 1.08]: measured 1.0465-1.0589; Phase-0
    1.048-1.061 — the slow wave is attributed PHYSICAL for this open
    line by the one dx/2 mechanism check (excess dx-stable, ratio 0.97;
    see the band comment above), so this gate locks a physical measured
    class, not a discretization artefact."""
    _, diag = refplane_thru
    w = 2 * np.pi * _FREQS
    for p in (0, 1):
        zc = diag["zc"][p]
        assert np.all(zc.real >= _ZC_RE_BAND[0]) \
            and np.all(zc.real <= _ZC_RE_BAND[1]), (
                f"port {p} Zc left the measured class: {zc.real}")
        assert np.all(np.abs(zc.imag / zc.real) <= _ZC_IM_OVER_RE_MAX), (
            f"port {p} Zc Im/Re too large: {zc.imag / zc.real}")
        b = diag["beta"][p] / (w / C0)
        assert np.all(b >= _BETA_OVER_WC_BAND[0]) \
            and np.all(b <= _BETA_OVER_WC_BAND[1]), (
                f"port {p} beta/(w/c) left the measured class: {b}")


@pytest.mark.slow_physics
def test_refplane_thru_deembedded_phase_tracks_measured_beta(refplane_thru):
    """arg(S21) after de-embedding = -beta_meas*L to sub-milliradian
    (measured <= 8.4e-4 rad; gate 0.02 rad, ~24x margin). This is the
    de-embed self-consistency witness: the port-plane-referenced thru
    phase must equal the measured line phase over the port spacing."""
    S, diag = refplane_thru
    beta_avg = 0.5 * (diag["beta"][0] + diag["beta"][1])
    dev = np.angle(S[1, 0] * np.exp(1j * beta_avg * _L))
    assert np.all(np.abs(dev) <= _PHASE_DEEMBED_MAX_RAD), (
        f"de-embedded S21 phase left the measured-beta track: {dev}")


@pytest.mark.slow_physics
def test_refplane_thru_energy_and_passivity_labeled(refplane_thru):
    """Energy/passivity envelope of the MIXED matrix — honesty label.

    The matrix mixes byte-frozen legacy diagonals with plane-referenced
    off-diagonals (separately calibrated), and the Phase-0 box referee
    itself reads sqrt(P_abs/P_launch) up to 1.0066 at 3 GHz (~1-3%
    flux-accounting envelope: residual box leakage + finite DFT). Small
    >1 excursions of the singular values (measured max 1.0299) are that
    accounting envelope, NOT validated gain — this gate bounds the
    envelope, it does not certify passivity. Row energy
    |S11|^2+|S21|^2 measured <= 1.0003."""
    S, _ = refplane_thru
    sv = np.array([np.linalg.svd(S[:, :, k], compute_uv=False)[0]
                   for k in range(len(_FREQS))])
    assert np.all(sv <= _MIXED_SV_MAX), f"singular values blew up: {sv}"
    en = np.abs(S[0, 0]) ** 2 + np.abs(S[1, 0]) ** 2
    assert np.all(en <= _ENERGY_ROW_MAX), f"row energy blew up: {en}"
