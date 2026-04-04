"""Tests for rectangular waveguide port with analytical TE/TM mode profiles."""

import numpy as np
import jax.numpy as jnp
import pytest

from rfx.grid import C0
from rfx.core.yee import init_state, init_materials, update_e, update_h, EPS_0, MU_0
from rfx.boundaries.pec import apply_pec
from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h
from rfx.sources.waveguide_port import (
    WaveguidePort, init_waveguide_port, inject_waveguide_port,
    update_waveguide_port_probe, extract_waveguide_sparams,
    extract_waveguide_s11, extract_waveguide_s21, waveguide_plane_positions,
    cutoff_frequency,
)


def test_te10_cutoff_frequency():
    """TE10 cutoff for standard WR-90 waveguide (a=22.86mm, b=10.16mm)."""
    a = 22.86e-3
    b = 10.16e-3
    f_c = cutoff_frequency(a, b, 1, 0)
    # Analytical: f_c = c / (2*a) = 6.557 GHz
    f_c_exact = C0 / (2 * a)
    err = abs(f_c - f_c_exact) / f_c_exact
    print(f"\nTE10 cutoff: {f_c/1e9:.4f} GHz (exact: {f_c_exact/1e9:.4f} GHz)")
    assert err < 1e-10, f"Cutoff error {err}"


def test_te10_mode_profile_shape():
    """TE10 mode profile: Ez = (pi/a)*sin(pi*y/a), Ey = 0.

    With propagation along +x and width a in y, TE10 has Ez as the
    dominant transverse field (from d(Hx)/dy derivative).
    """
    from rfx.sources.waveguide_port import _te_mode_profiles

    a, b = 0.04, 0.02
    ny, nz = 20, 10
    dx = a / ny
    y = np.linspace(0.5 * dx, a - 0.5 * dx, ny)
    z = np.linspace(0.5 * dx, b - 0.5 * dx, nz)

    ey, ez, hy, hz = _te_mode_profiles(a, b, 1, 0, y, z)

    # TE10: Ey should be zero everywhere (n=0 → no Ey component)
    assert np.allclose(ey, 0, atol=1e-10), f"TE10 Ey not zero: max={np.max(np.abs(ey))}"

    # Ez should have sin(pi*y/a) shape along y, constant along z
    ez_mid = ez[:, nz // 2]  # slice at middle z
    expected_shape = np.sin(np.pi * y / a)
    # Normalize both for shape comparison
    ez_norm = ez_mid / np.max(np.abs(ez_mid))
    exp_norm = expected_shape / np.max(np.abs(expected_shape))
    assert np.allclose(np.abs(ez_norm), np.abs(exp_norm), atol=0.05), \
        "TE10 Ez profile shape mismatch"

    # H profiles: hy = -ez, hz = ey = 0
    assert np.allclose(hz, 0, atol=1e-10), "TE10 Hz not zero"
    assert np.allclose(hy, -ez, atol=1e-10), "TE10 Hy != -Ez"

    print(f"\nTE10 mode profile: Ez max={np.max(np.abs(ez)):.4f}, Ey max={np.max(np.abs(ey)):.6f}")


def test_te11_mode_profile_derivative_weights():
    """TE11 mode profile: Ey and Ez have different amplitudes when a != b.

    The derivative weights (n*pi/b) and (m*pi/a) give different peak
    amplitudes for Ey and Ez. For a=40mm, b=20mm:
        |Ey_max|/|Ez_max| = (pi/b) / (pi/a) = a/b = 2.0
    """
    from rfx.sources.waveguide_port import _te_mode_profiles

    a, b = 0.04, 0.02
    ny, nz = 40, 20
    dx = a / ny
    y = np.linspace(0.5 * dx, a - 0.5 * dx, ny)
    z = np.linspace(0.5 * dx, b - 0.5 * dx, nz)

    ey, ez, hy, hz = _te_mode_profiles(a, b, 1, 1, y, z)

    # Both components should be nonzero for TE11
    assert np.max(np.abs(ey)) > 0.01, "TE11 Ey should be nonzero"
    assert np.max(np.abs(ez)) > 0.01, "TE11 Ez should be nonzero"

    # Before normalization, |Ey_max|/|Ez_max| = (n*pi/b)/(m*pi/a) = a/b
    # After normalization (integral = 1), the ratio is preserved
    # Check the ratio of peak amplitudes
    ey_peak = np.max(np.abs(ey))
    ez_peak = np.max(np.abs(ez))
    ratio = ey_peak / ez_peak
    expected_ratio = a / b  # = 2.0

    print(f"\nTE11 mode: |Ey|_max={ey_peak:.4f}, |Ez|_max={ez_peak:.4f}")
    print(f"  Ratio |Ey/Ez| = {ratio:.3f} (expected {expected_ratio:.1f})")

    assert abs(ratio - expected_ratio) / expected_ratio < 0.1, \
        f"TE11 Ey/Ez ratio {ratio:.3f} != expected {expected_ratio:.1f}"

    # PEC BCs: Ey ∝ sin(nπz/b) → 0 at walls, Ez ∝ sin(mπy/a) → 0 at walls.
    # Cell centers don't land exactly on walls, so check that boundary values
    # are much smaller than peak (first/last cells are near the wall).
    ey_boundary_ratio = max(np.max(np.abs(ey[:, 0])), np.max(np.abs(ey[:, -1]))) / ey_peak
    ez_boundary_ratio = max(np.max(np.abs(ez[0, :])), np.max(np.abs(ez[-1, :]))) / ez_peak
    print(f"  Boundary ratios: Ey={ey_boundary_ratio:.3f}, Ez={ez_boundary_ratio:.3f}")
    assert ey_boundary_ratio < 0.15, f"TE11 Ey too large at z boundary: {ey_boundary_ratio:.3f}"
    assert ez_boundary_ratio < 0.15, f"TE11 Ez too large at y boundary: {ez_boundary_ratio:.3f}"


def test_tm_mode_invalid():
    """TM modes with m=0 or n=0 should raise ValueError."""
    from rfx.sources.waveguide_port import _tm_mode_profiles

    a, b = 0.04, 0.02
    y = np.linspace(0.001, a - 0.001, 10)
    z = np.linspace(0.001, b - 0.001, 5)

    with pytest.raises(ValueError):
        _tm_mode_profiles(a, b, 1, 0, y, z)

    with pytest.raises(ValueError):
        _tm_mode_profiles(a, b, 0, 1, y, z)


@pytest.mark.parametrize(
    ("mode_type", "mode", "freqs"),
    [
        ("TE", (1, 0), jnp.array([6.0e9, 8.0e9])),
        ("TM", (1, 1), jnp.array([11.0e9, 13.0e9])),
    ],
)
def test_waveguide_sparams_recover_traveling_waves(mode_type, mode, freqs):
    """S11/S21 should reconstruct known forward/backward modal waves."""
    port = WaveguidePort(
        x_index=5,
        y_slice=(0, 20),
        z_slice=(0, 10),
        a=0.04,
        b=0.02,
        mode=mode,
        mode_type=mode_type,
    )
    cfg = init_waveguide_port(
        port,
        dx=0.002,
        freqs=freqs,
        f0=float(freqs[0]),
        bandwidth=0.5,
    )

    omega = 2 * np.pi * np.array(freqs)
    kc = 2 * np.pi * cutoff_frequency(port.a, port.b, *mode) / C0
    beta = np.sqrt((omega / C0) ** 2 - kc ** 2)
    if mode_type == "TE":
        z_mode = omega * MU_0 / beta
    else:
        z_mode = beta / (omega * EPS_0)

    a_ref = np.array([1.0 + 0.2j, 0.7 - 0.1j], dtype=np.complex64)
    b_ref = np.array([0.15 - 0.05j, -0.10 + 0.08j], dtype=np.complex64)
    s21_expected = np.array([0.80 + 0.10j, 0.55 - 0.15j], dtype=np.complex64)
    a_probe = s21_expected * a_ref
    b_probe = np.zeros_like(a_probe)

    v_ref = a_ref + b_ref
    i_ref = (a_ref - b_ref) / z_mode
    v_probe = a_probe + b_probe
    i_probe = (a_probe - b_probe) / z_mode

    cfg = cfg._replace(
        v_ref_dft=jnp.array(v_ref),
        i_ref_dft=jnp.array(i_ref),
        v_probe_dft=jnp.array(v_probe),
        i_probe_dft=jnp.array(i_probe),
    )

    s11 = np.array(extract_waveguide_s11(cfg))
    s21 = np.array(extract_waveguide_s21(cfg))
    np.testing.assert_allclose(s11, b_ref / a_ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(s21, s21_expected, rtol=1e-5, atol=1e-6)


def test_waveguide_sparams_support_reference_plane_shifts():
    """De-embedding shifts should recover the desired source/probe planes."""
    freqs = jnp.array([6.0e9, 8.0e9])
    port = WaveguidePort(
        x_index=5,
        y_slice=(0, 20),
        z_slice=(0, 10),
        a=0.04,
        b=0.02,
        mode=(1, 0),
        mode_type="TE",
    )
    cfg = init_waveguide_port(port, dx=0.002, freqs=freqs, f0=6.0e9, bandwidth=0.5)

    omega = 2 * np.pi * np.array(freqs)
    kc = 2 * np.pi * cutoff_frequency(port.a, port.b, 1, 0) / C0
    beta = np.sqrt((omega / C0) ** 2 - kc ** 2)
    z_mode = omega * MU_0 / beta

    ref_plane_shift = -0.006   # desired plane is 6 mm upstream of stored ref plane
    probe_plane_shift = -0.010 # desired plane is 10 mm upstream of stored probe plane

    a_ref_target = np.array([1.0 + 0.2j, 0.7 - 0.1j], dtype=np.complex64)
    b_ref_target = np.array([0.10 - 0.04j, -0.05 + 0.03j], dtype=np.complex64)
    s21_target = np.array([0.85 - 0.15j, 0.60 + 0.05j], dtype=np.complex64)
    a_probe_target = s21_target * a_ref_target
    b_probe_target = np.zeros_like(a_probe_target)

    a_ref_meas = a_ref_target * np.exp(-1j * beta * (-ref_plane_shift))
    b_ref_meas = b_ref_target * np.exp(+1j * beta * (-ref_plane_shift))
    a_probe_meas = a_probe_target * np.exp(-1j * beta * (-probe_plane_shift))
    b_probe_meas = b_probe_target * np.exp(+1j * beta * (-probe_plane_shift))

    v_ref = a_ref_meas + b_ref_meas
    i_ref = (a_ref_meas - b_ref_meas) / z_mode
    v_probe = a_probe_meas + b_probe_meas
    i_probe = (a_probe_meas - b_probe_meas) / z_mode

    cfg = cfg._replace(
        v_ref_dft=jnp.array(v_ref),
        i_ref_dft=jnp.array(i_ref),
        v_probe_dft=jnp.array(v_probe),
        i_probe_dft=jnp.array(i_probe),
    )

    positions = waveguide_plane_positions(cfg)
    assert positions["source"] == pytest.approx(0.010)
    assert positions["reference"] == pytest.approx(0.016)
    assert positions["probe"] == pytest.approx(0.030)

    s11, s21 = extract_waveguide_sparams(
        cfg,
        ref_shift=ref_plane_shift,
        probe_shift=probe_plane_shift,
    )
    np.testing.assert_allclose(np.array(s11), b_ref_target / a_ref_target, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.array(s21), s21_target, rtol=1e-5, atol=1e-6)


class _WgGrid:
    """Minimal grid object for waveguide simulations.

    Unlike Grid, does NOT add CPML padding on y/z — the full y/z extent
    IS the waveguide with PEC walls at the boundary.
    """
    def __init__(self, length, a_wg, b_wg, dx, cpml_layers):
        self.dx = dx
        self.cpml_layers = cpml_layers
        self.dt = 0.99 * dx / (C0 * np.sqrt(3))
        # x: physical domain + CPML padding on both sides
        self.nx = int(np.ceil(length / dx)) + 1 + 2 * cpml_layers
        # y, z: exact waveguide dimensions (PEC at boundaries)
        self.ny = int(np.ceil(a_wg / dx)) + 1
        self.nz = int(np.ceil(b_wg / dx)) + 1
        self.shape = (self.nx, self.ny, self.nz)

    def num_timesteps(self, num_periods):
        return int(num_periods / (10e9 * self.dt))


def _run_waveguide_sim(a_wg, b_wg, length, f0, dx, nc, freqs,
                       num_periods=40, probe_offset=15, ref_offset=3):
    """Helper: run a PEC waveguide simulation and return port_cfg with S21.

    Grid: y/z = exact waveguide (PEC at boundary = walls).
    x = physical domain + CPML padding. PEC only on yz (not x).
    """
    grid = _WgGrid(length, a_wg, b_wg, dx, nc)
    dt = grid.dt

    # Actual waveguide dimensions from grid
    a_actual = (grid.ny - 1) * dx
    b_actual = (grid.nz - 1) * dx
    f_c = C0 / (2 * a_actual)

    port_x = nc + 5

    port = WaveguidePort(
        x_index=port_x,
        y_slice=(0, grid.ny), z_slice=(0, grid.nz),
        a=a_actual, b=b_actual,
        mode=(1, 0), mode_type="TE",
    )

    n_steps = grid.num_timesteps(num_periods=num_periods)
    port_cfg = init_waveguide_port(port, dx, freqs, f0=f0, bandwidth=0.5,
                                   amplitude=1.0, probe_offset=probe_offset,
                                   ref_offset=ref_offset,
                                   dft_total_steps=n_steps)

    state = init_state(grid.shape)
    materials = init_materials(grid.shape)
    cp, cs = init_cpml(grid)

    for step in range(n_steps):
        t = step * dt

        state = update_h(state, materials, dt, dx)
        state, cs = apply_cpml_h(state, cp, cs, grid, axes="x")
        state = update_e(state, materials, dt, dx)
        state, cs = apply_cpml_e(state, cp, cs, grid, axes="x")
        state = apply_pec(state, axes="yz")  # PEC on y/z only (waveguide walls)

        state = inject_waveguide_port(state, port_cfg, t, dt, dx)
        port_cfg = update_waveguide_port_probe(port_cfg, state, dt, dx)

    return port_cfg, grid, f_c, n_steps, dt


def test_waveguide_port_propagation():
    """TE10 mode launched above cutoff propagates and is received downstream.

    A PEC waveguide with CPML on x-ends. The full grid y/z extent is the
    waveguide (PEC at grid boundary = walls).
    TE10 above cutoff: |S21| should be close to 1 and never exceed 1.
    """
    a_wg = 0.04
    b_wg = 0.02
    length = 0.12
    f0 = 6e9
    dx = 0.002
    nc = 10

    freqs = jnp.linspace(4.5e9, 8e9, 25)

    port_cfg, grid, f_c, n_steps, dt = _run_waveguide_sim(
        a_wg, b_wg, length, f0, dx, nc, freqs,
        probe_offset=15, ref_offset=3,
    )

    assert f0 > f_c, f"Source {f0/1e9:.1f} GHz must be above cutoff {f_c/1e9:.2f} GHz"

    s21 = extract_waveguide_s21(port_cfg)
    s21_mag = np.abs(np.array(s21))

    # Above cutoff band
    f_arr = np.array(freqs)
    above_cutoff = f_arr > f_c * 1.3
    s21_above = s21_mag[above_cutoff]
    s21_db = 20 * np.log10(np.maximum(s21_above, 1e-10))
    mean_s21_db = np.mean(s21_db)

    print("\nWaveguide port TE10 propagation:")
    print(f"  f_cutoff = {f_c/1e9:.2f} GHz, f0 = {f0/1e9:.1f} GHz")
    print(f"  Grid: {grid.shape}, a_actual = {(grid.ny-1)*dx*1e3:.1f} mm")
    print(f"  Steps: {n_steps}")
    print(f"  |S21| above cutoff (mean): {np.mean(s21_above):.4f} ({mean_s21_db:.1f} dB)")
    print(f"  |S21| min/max: {np.min(s21_above):.4f} / {np.max(s21_above):.4f}")

    s11 = extract_waveguide_s11(port_cfg)
    s11_mag = np.abs(np.array(s11))
    # TE10 above cutoff: with tapered DFT accumulation, the extracted S21 band
    # should now be much smoother and stay near unity on average.
    assert mean_s21_db > -6, \
        f"Mean |S21| = {mean_s21_db:.1f} dB, expected > -6 dB"
    assert np.max(s21_above) < 1.5, \
        f"Max |S21| = {np.max(s21_above):.3f}, expected < 1.5 after windowing"
    assert np.min(s21_above) > 0.7, \
        f"Min |S21| = {np.min(s21_above):.3f}, expected > 0.7 after windowing"
    assert np.mean(s11_mag[above_cutoff]) < 0.6, \
        f"Mean |S11| = {np.mean(s11_mag[above_cutoff]):.3f}, expected < 0.6"


def test_te10_below_cutoff_evanescent():
    """Below cutoff, TE10 mode should be evanescent (|S21| -> 0).

    At f < f_cutoff, the mode cannot propagate. Modal voltage at the
    downstream probe should be much smaller than at the reference probe.
    """
    a_wg = 0.04
    b_wg = 0.02
    length = 0.10
    f0 = 2.5e9
    dx = 0.002
    nc = 10

    freqs = jnp.linspace(1e9, 3.0e9, 15)

    port_cfg, grid, f_c, n_steps, dt = _run_waveguide_sim(
        a_wg, b_wg, length, f0, dx, nc, freqs,
        probe_offset=15, ref_offset=3,
    )

    s21 = extract_waveguide_s21(port_cfg)
    s21_mag = np.abs(np.array(s21))

    f_arr = np.array(freqs)
    below_cutoff = f_arr < f_c * 0.7
    s21_below = s21_mag[below_cutoff]
    s21_below_db = 20 * np.log10(np.maximum(np.mean(s21_below), 1e-10))

    print("\nWaveguide port TE10 below cutoff:")
    print(f"  f_cutoff = {f_c/1e9:.2f} GHz, f0 = {f0/1e9:.1f} GHz")
    print(f"  |S21| below cutoff (mean): {np.mean(s21_below):.4f} ({s21_below_db:.1f} dB)")

    # Below cutoff: evanescent mode decays exponentially with distance.
    # With 12 cells between ref and probe (24mm) and f well below cutoff,
    # expect significant attenuation.
    assert s21_below_db < -3, \
        f"Mean |S21| below cutoff = {s21_below_db:.1f} dB, expected < -3 dB"
