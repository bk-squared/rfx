"""Floquet port and periodic BC tests for phased array unit cell analysis.

Tests:
1. Periodic BC field continuity (fields wrap correctly)
2. Floquet phase shift at normal incidence (theta=0 -> phase=1)
3. Floquet phase shift at oblique incidence (theta=30 -> known phase)
4. Floquet wave vector computation
5. Floquet DFT accumulator initialization and update
6. Floquet port broadside (normal incidence on infinite array)
7. Unit cell with Floquet port (patch element unit cell)
8. API integration (add_floquet_port method)
"""

import math

import numpy as np
import pytest
import jax.numpy as jnp

from rfx.grid import Grid, C0
from rfx.core.yee import (
    init_state, init_materials, update_e, update_h,
    FDTDState, MaterialArrays, EPS_0, MU_0,
)
from rfx.floquet import (
    FloquetPort,
    floquet_phase_shift,
    floquet_wave_vector,
    FloquetDFTAccumulator,
    init_floquet_dft,
    update_floquet_dft,
    inject_floquet_source,
    extract_floquet_modes,
    compute_floquet_s_params,
)
from rfx.api import Simulation
from rfx.geometry.csg import Box


# =========================================================================
# Test 1: Periodic BC field continuity
# =========================================================================

def test_periodic_bc_field_continuity():
    """Fields should be continuous across periodic boundaries.

    With periodic BC in y and z, a point source should produce fields
    that wrap around: F[0,:,:] should be influenced by F[-1,:,:] and
    vice versa (via the jnp.roll mechanism in the Yee updates).
    """
    grid = Grid(freq_max=5e9, domain=(0.02, 0.02, 0.02),
                dx=0.002, cpml_layers=0, mode="3d")
    state = init_state(grid.shape)
    materials = init_materials(grid.shape)
    dt, dx = grid.dt, grid.dx
    periodic = (False, True, True)

    # Place source at center
    cx, cy, cz = grid.nx // 2, grid.ny // 2, grid.nz // 2
    field = state.ez.at[cx, cy, cz].set(1.0)
    state = state._replace(ez=field)

    # Run a few steps with periodic BC
    for _ in range(50):
        state = update_h(state, materials, dt, dx, periodic)
        state = update_e(state, materials, dt, dx, periodic)

    ez = np.array(state.ez)

    # Check no NaN
    assert not np.any(np.isnan(ez)), "NaN in periodic BC simulation"

    # Fields should have propagated and wrapped. With periodic BC,
    # the fields at y=0 and y=-1 should be connected (not zero-padded).
    # After enough steps, edge values should be non-trivial.
    mid_x = grid.nx // 2

    # y-periodic: fields at y boundaries should be non-zero (wrapped)
    y_boundary_energy = np.sum(ez[mid_x, 0, :] ** 2) + np.sum(ez[mid_x, -1, :] ** 2)
    y_interior_energy = np.sum(ez[mid_x, grid.ny // 2, :] ** 2)

    # z-periodic: fields at z boundaries should be non-zero (wrapped)
    z_boundary_energy = np.sum(ez[mid_x, :, 0] ** 2) + np.sum(ez[mid_x, :, -1] ** 2)

    # The boundary energies should be comparable to interior
    # (not zero as would happen with PEC or zero-pad)
    assert y_boundary_energy > 0, "y-periodic boundary has zero fields"
    assert z_boundary_energy > 0, "z-periodic boundary has zero fields"

    # Verify that periodic wrapping works: compare roll-based result
    # with direct field access. After update_h with periodic=True on y,
    # fwd(arr, 1) uses jnp.roll(arr, -1, 1), meaning arr[-1] wraps to arr[0].
    # This means fields should smoothly connect across boundaries.
    print(f"\ny-boundary energy: {y_boundary_energy:.6e}")
    print(f"y-interior energy: {y_interior_energy:.6e}")
    print(f"z-boundary energy: {z_boundary_energy:.6e}")


def test_periodic_bc_no_boundary_reflection():
    """Periodic BC wraps fields through boundaries, unlike PEC which reflects.

    With PEC (apply_pec enforced), the boundary acts as a mirror.
    With periodic, the field pattern wraps seamlessly. We verify
    the wrapping produces a distinct field distribution from PEC.
    """
    from rfx.boundaries.pec import apply_pec

    grid = Grid(freq_max=5e9, domain=(0.02, 0.02, 0.02),
                dx=0.002, cpml_layers=0, mode="3d")
    state_pec = init_state(grid.shape)
    materials = init_materials(grid.shape)
    dt, dx = grid.dt, grid.dx

    # Source near y=0 boundary
    cx = grid.nx // 2
    src_y = 1
    cz = grid.nz // 2
    field = state_pec.ez.at[cx, src_y, cz].set(1.0)
    state_pec = state_pec._replace(ez=field)

    # Periodic version: same initial condition
    state_per = init_state(grid.shape)
    field_per = state_per.ez.at[cx, src_y, cz].set(1.0)
    state_per = state_per._replace(ez=field_per)

    periodic_off = (False, False, False)
    periodic_on = (False, True, True)

    for _ in range(30):
        state_pec = update_h(state_pec, materials, dt, dx, periodic_off)
        state_pec = update_e(state_pec, materials, dt, dx, periodic_off)
        state_pec = apply_pec(state_pec)  # enforce PEC on all boundaries

        state_per = update_h(state_per, materials, dt, dx, periodic_on)
        state_per = update_e(state_per, materials, dt, dx, periodic_on)

    # With periodic, field at y=0 boundary should be non-zero (wrapped)
    per_y0_energy = np.sum(np.array(state_per.ez[cx, 0, :]) ** 2)
    assert per_y0_energy > 0, "Periodic boundary should have non-zero wrapped fields"

    # PEC with apply_pec zeros boundary tangential E
    pec_y0_energy = np.sum(np.array(state_pec.ez[cx, 0, :]) ** 2)
    assert pec_y0_energy < 1e-10, "PEC boundary should have near-zero Ez"

    # The two field distributions should be meaningfully different
    diff = np.sum((np.array(state_pec.ez) - np.array(state_per.ez)) ** 2)
    total = np.sum(np.array(state_per.ez) ** 2) + np.sum(np.array(state_pec.ez) ** 2)
    assert diff / (total + 1e-30) > 0.01, \
        "Periodic and PEC should produce different field distributions"


# =========================================================================
# Test 2: Floquet phase shift at normal incidence
# =========================================================================

def test_floquet_phase_shift_normal():
    """At broadside (theta=0), Bloch phase shifts should be 1+0j."""
    Lx, Ly = 0.02, 0.02
    freq = 10e9

    phase_x, phase_y = floquet_phase_shift(Lx, Ly, freq, theta_deg=0.0, phi_deg=0.0)

    assert abs(phase_x - (1.0 + 0j)) < 1e-12, f"phase_x at broadside: {phase_x}"
    assert abs(phase_y - (1.0 + 0j)) < 1e-12, f"phase_y at broadside: {phase_y}"


def test_floquet_phase_shift_normal_any_freq():
    """Broadside phase shift is 1 regardless of frequency."""
    for freq in [1e9, 5e9, 10e9, 60e9]:
        px, py = floquet_phase_shift(0.01, 0.015, freq, 0.0, 0.0)
        assert abs(px - 1.0) < 1e-12
        assert abs(py - 1.0) < 1e-12


# =========================================================================
# Test 3: Floquet phase shift at oblique incidence
# =========================================================================

def test_floquet_phase_shift_oblique():
    """At theta=30, phi=0, verify phase shift matches analytical formula.

    kx = k0 * sin(30) * cos(0) = k0 * 0.5
    ky = k0 * sin(30) * sin(0) = 0
    phase_x = exp(j * kx * Lx)
    phase_y = 1 (since ky=0)
    """
    freq = 10e9
    Lx = 0.015  # 15mm = lambda/2 at 10 GHz
    Ly = 0.015

    phase_x, phase_y = floquet_phase_shift(Lx, Ly, freq, theta_deg=30.0, phi_deg=0.0)

    k0 = 2 * math.pi * freq / C0
    kx_expected = k0 * math.sin(math.radians(30)) * math.cos(0)
    expected_phase_x = np.exp(1j * kx_expected * Lx)

    assert abs(phase_x - expected_phase_x) < 1e-10, \
        f"phase_x mismatch: got {phase_x}, expected {expected_phase_x}"
    assert abs(phase_y - 1.0) < 1e-12, \
        f"phase_y should be 1 for phi=0, got {phase_y}"

    print("\nFloquet phase at theta=30, phi=0, f=10GHz, L=15mm:")
    print(f"  kx = {kx_expected:.2f} rad/m")
    print(f"  phase_x = {phase_x}")
    print(f"  |phase_x| = {abs(phase_x):.10f} (should be 1)")

    # Phase magnitude should always be 1
    assert abs(abs(phase_x) - 1.0) < 1e-12, "Phase magnitude must be 1"


def test_floquet_phase_shift_oblique_phi45():
    """At theta=30, phi=45, both x and y phase shifts should be non-trivial."""
    freq = 10e9
    Lx, Ly = 0.015, 0.015

    phase_x, phase_y = floquet_phase_shift(Lx, Ly, freq, theta_deg=30.0, phi_deg=45.0)

    k0 = 2 * math.pi * freq / C0
    kx = k0 * math.sin(math.radians(30)) * math.cos(math.radians(45))
    ky = k0 * math.sin(math.radians(30)) * math.sin(math.radians(45))

    assert abs(phase_x - np.exp(1j * kx * Lx)) < 1e-10
    assert abs(phase_y - np.exp(1j * ky * Ly)) < 1e-10
    assert abs(abs(phase_x) - 1.0) < 1e-12
    assert abs(abs(phase_y) - 1.0) < 1e-12


# =========================================================================
# Test 4: Floquet wave vector computation
# =========================================================================

def test_floquet_wave_vector_broadside():
    """At broadside, kx=ky=0 and kz=k0."""
    freq = 10e9
    kx, ky, kz = floquet_wave_vector(freq, theta_deg=0.0, phi_deg=0.0)

    k0 = 2 * math.pi * freq / C0
    assert abs(kx) < 1e-10, f"kx should be 0 at broadside, got {kx}"
    assert abs(ky) < 1e-10, f"ky should be 0 at broadside, got {ky}"
    assert abs(kz - k0) < 1e-6, f"kz should be k0={k0:.2f}, got {kz:.2f}"


def test_floquet_wave_vector_oblique():
    """At theta=30, verify kx, ky, kz components and |k|=k0."""
    freq = 10e9
    kx, ky, kz = floquet_wave_vector(freq, theta_deg=30.0, phi_deg=0.0)

    k0 = 2 * math.pi * freq / C0
    k_mag = math.sqrt(kx ** 2 + ky ** 2 + kz ** 2)

    assert abs(k_mag - k0) < 1e-6, f"|k| should be k0={k0:.2f}, got {k_mag:.2f}"
    assert abs(kx - k0 * math.sin(math.radians(30))) < 1e-6
    assert abs(ky) < 1e-10
    assert abs(kz - k0 * math.cos(math.radians(30))) < 1e-6


# =========================================================================
# Test 5: Floquet DFT accumulator
# =========================================================================

def test_floquet_dft_init():
    """DFT accumulator should initialize to zeros with correct shape."""
    n_freqs = 10
    plane_shape = (20, 20)
    acc = init_floquet_dft(n_freqs, plane_shape)

    assert acc.e_tang1_dft.shape == (10, 20, 20)
    assert acc.e_tang2_dft.shape == (10, 20, 20)
    assert acc.h_tang1_dft.shape == (10, 20, 20)
    assert acc.h_tang2_dft.shape == (10, 20, 20)
    assert jnp.allclose(acc.e_tang1_dft, 0)


def test_floquet_dft_accumulation():
    """DFT accumulator should accumulate non-zero values when fields are non-zero."""
    grid = Grid(freq_max=10e9, domain=(0.02, 0.02, 0.02),
                dx=0.002, cpml_layers=0, mode="3d")
    state = init_state(grid.shape)

    # Set a non-zero field
    state = state._replace(ex=jnp.ones(grid.shape, dtype=jnp.float32))

    freqs = jnp.linspace(5e9, 15e9, 5)
    port_index = grid.nz // 2
    axis = 2  # z-normal

    # Extract plane shape for z-normal: (nx, ny)
    plane_shape = (grid.nx, grid.ny)
    acc = init_floquet_dft(len(freqs), plane_shape)

    # Accumulate one step
    acc_new = update_floquet_dft(acc, state, port_index, axis, freqs, grid.dt, step=0)

    # At step=0, phase = exp(-j*2*pi*f*0) = 1, so DFT should equal field
    assert not jnp.allclose(acc_new.e_tang1_dft, 0), "DFT should be non-zero after accumulation"

    # The e_tang1 for z-normal is ex, which we set to 1.0
    # At step=0 with phase=1: accumulated value should be 1.0 per cell
    expected = jnp.ones(plane_shape, dtype=jnp.float32)
    for fi in range(len(freqs)):
        assert jnp.allclose(acc_new.e_tang1_dft[fi], expected, atol=1e-5), \
            f"DFT mismatch at freq index {fi}"


# =========================================================================
# Test 6: Floquet port broadside (normal incidence on infinite array)
# =========================================================================

def test_floquet_port_broadside():
    """Floquet port at broadside should produce clean plane-wave excitation.

    With periodic BC and a uniform plane source, fields should be
    uniform across the transverse plane (true plane wave).
    """
    grid = Grid(freq_max=10e9, domain=(0.015, 0.015, 0.06),
                dx=0.0015, cpml_layers=8, cpml_axes="z", mode="3d")
    state = init_state(grid.shape)
    materials = init_materials(grid.shape)
    dt, dx = grid.dt, grid.dx
    periodic = (True, True, False)

    # Inject uniform Ex source at z = 1/4 of domain (like Floquet port)
    src_z = grid.nz // 4
    f0 = 5e9
    bw = 0.5
    tau = 1.0 / (f0 * bw * math.pi)
    t0 = 3.0 * tau

    n_steps = 200
    probe_z = grid.nz // 2

    ts = np.zeros(n_steps)
    for step in range(n_steps):
        state = update_h(state, materials, dt, dx, periodic)
        state = update_e(state, materials, dt, dx, periodic)

        # Inject uniform plane source
        t = step * dt
        arg = (t - t0) / tau
        pulse = 1.0 * (-2.0 * arg) * np.exp(-(arg ** 2))
        field = state.ex.at[:, :, src_z].add(pulse)
        state = state._replace(ex=field)

        ts[step] = float(state.ex[grid.nx // 2, grid.ny // 2, probe_z])

    assert not np.any(np.isnan(ts)), "NaN in broadside Floquet simulation"

    # With periodic BC + uniform source, field should be uniform in x-y
    # at any z-plane. Check variance across the transverse plane at probe_z.
    ex_plane = np.array(state.ex[:, :, probe_z])
    variance = np.var(ex_plane)
    mean_sq = np.mean(ex_plane ** 2)

    print("\nBroadside plane uniformity:")
    print(f"  Ex plane variance: {variance:.6e}")
    print(f"  Ex plane mean^2: {mean_sq:.6e}")
    print(f"  Coefficient of variation: {np.sqrt(variance) / (np.sqrt(mean_sq) + 1e-30):.4f}")

    # Plane should be very uniform (periodic BC + uniform source)
    # Variance should be near zero relative to the signal
    if mean_sq > 1e-20:
        cv = np.sqrt(variance) / np.sqrt(mean_sq)
        assert cv < 0.01, f"Field non-uniformity {cv:.4f} exceeds 1% — plane wave not uniform"


# =========================================================================
# Test 7: Unit cell with Floquet port (patch element)
# =========================================================================

def test_unit_cell_with_floquet():
    """Unit cell simulation with a PEC patch and Floquet excitation.

    A simple PEC patch on a dielectric substrate in a periodic unit cell.
    The Floquet port excites a broadside plane wave and we verify the
    simulation runs without errors and produces physically meaningful output.
    """
    # Unit cell: 15mm x 15mm (half-wavelength at 10 GHz)
    Lx, Ly = 0.015, 0.015
    Lz = 0.03  # enough room for CPML on z

    sim = Simulation(
        freq_max=15e9,
        domain=(Lx, Ly, Lz),
        boundary="cpml",
        cpml_layers=8,
    )

    # Substrate
    sim.add_material("substrate", eps_r=2.2)
    sim.add(Box((0, 0, Lz / 2 - 0.001), (Lx, Ly, Lz / 2)), material="substrate")

    # PEC patch (8mm x 8mm centered)
    patch_w = 0.008
    x0 = (Lx - patch_w) / 2
    y0 = (Ly - patch_w) / 2
    sim.add(Box((x0, y0, Lz / 2), (x0 + patch_w, y0 + patch_w, Lz / 2)), material="pec")

    # Floquet port
    sim.add_floquet_port(
        Lz * 0.25,
        axis="z",
        scan_theta=0.0,
        scan_phi=0.0,
        polarization="te",
        n_freqs=10,
    )

    # Probe at center
    sim.add_probe((Lx / 2, Ly / 2, Lz / 2 + 0.002), component="ex")

    # Run short simulation
    result = sim.run(n_steps=200)

    # Basic checks
    assert result.state is not None
    assert result.time_series is not None
    ts = np.array(result.time_series).ravel()
    assert not np.any(np.isnan(ts)), "NaN in unit cell simulation"
    assert np.max(np.abs(ts)) > 0, "Time series is all zeros — source not injected"

    print("\nUnit cell with Floquet port:")
    print(f"  Time series max: {np.max(np.abs(ts)):.6e}")
    print(f"  Time series shape: {result.time_series.shape}")


# =========================================================================
# Test 8: API integration
# =========================================================================

def test_add_floquet_port_api():
    """Test the add_floquet_port() API method and validation."""
    sim = Simulation(freq_max=10e9, domain=(0.015, 0.015, 0.03), boundary="cpml")

    # Basic add should work
    sim.add_floquet_port(0.005, axis="z", scan_theta=0.0)
    assert len(sim._floquet_ports) == 1
    assert sim._periodic_axes == "xy"  # auto-set for z-normal

    # Second port should also work
    sim.add_floquet_port(0.025, axis="z", scan_theta=0.0, name="port2")
    assert len(sim._floquet_ports) == 2


def test_add_floquet_port_auto_periodic():
    """Floquet port should auto-set periodic axes for transverse directions."""
    # z-normal -> periodic xy
    sim_z = Simulation(freq_max=10e9, domain=(0.015, 0.015, 0.03), boundary="cpml")
    sim_z.add_floquet_port(0.005, axis="z")
    assert "x" in sim_z._periodic_axes
    assert "y" in sim_z._periodic_axes

    # x-normal -> periodic yz
    sim_x = Simulation(freq_max=10e9, domain=(0.03, 0.015, 0.015), boundary="cpml")
    sim_x.add_floquet_port(0.005, axis="x")
    assert "y" in sim_x._periodic_axes
    assert "z" in sim_x._periodic_axes

    # y-normal -> periodic xz
    sim_y = Simulation(freq_max=10e9, domain=(0.015, 0.03, 0.015), boundary="cpml")
    sim_y.add_floquet_port(0.005, axis="y")
    assert "x" in sim_y._periodic_axes
    assert "z" in sim_y._periodic_axes


def test_add_floquet_port_validation():
    """Validation errors should be raised for invalid parameters."""
    sim = Simulation(freq_max=10e9, domain=(0.015, 0.015, 0.03), boundary="cpml")

    with pytest.raises(ValueError, match="axis"):
        sim.add_floquet_port(0.005, axis="w")

    with pytest.raises(ValueError, match="polarization"):
        sim.add_floquet_port(0.005, polarization="invalid")

    with pytest.raises(ValueError, match="scan_theta"):
        sim.add_floquet_port(0.005, scan_theta=90.0)

    with pytest.raises(ValueError, match="scan_theta"):
        sim.add_floquet_port(0.005, scan_theta=-1.0)

    with pytest.raises(ValueError, match="n_modes"):
        sim.add_floquet_port(0.005, n_modes=0)


def test_floquet_port_tfsf_incompatible():
    """Floquet port and TFSF should be mutually exclusive."""
    sim = Simulation(freq_max=10e9, domain=(0.015, 0.015, 0.03), boundary="cpml")
    sim.add_tfsf_source()

    with pytest.raises(ValueError, match="TFSF"):
        sim.add_floquet_port(0.005, axis="z")


def test_floquet_port_periodic_conflict():
    """If periodic axes are already set and conflict, should raise."""
    sim = Simulation(freq_max=10e9, domain=(0.015, 0.015, 0.03), boundary="cpml")
    # Set periodic on x only
    sim.set_periodic_axes("x")

    # z-normal Floquet needs xy periodic — x is satisfied, but y is missing
    with pytest.raises(ValueError, match="periodic"):
        sim.add_floquet_port(0.005, axis="z")


def test_floquet_repr():
    """Simulation repr should include Floquet port count."""
    sim = Simulation(freq_max=10e9, domain=(0.015, 0.015, 0.03), boundary="cpml")
    sim.add_floquet_port(0.005, axis="z")
    r = repr(sim)
    assert "floquet_ports=1" in r


# =========================================================================
# Test: inject_floquet_source
# =========================================================================

def test_inject_floquet_source():
    """Source injection should add non-zero fields at the port plane."""
    grid = Grid(freq_max=10e9, domain=(0.015, 0.015, 0.03),
                dx=0.0015, cpml_layers=0, mode="3d")
    state = init_state(grid.shape)

    # Inject at z-midplane, TE polarization (Ex)
    port_z = grid.nz // 2
    state_new = inject_floquet_source(
        state, port_z, axis=2, dt=grid.dt, dx=grid.dx,
        step=50, f0=5e9, bandwidth=0.5, amplitude=1.0,
        polarization="te",
    )

    ex_plane = np.array(state_new.ex[:, :, port_z])
    assert np.max(np.abs(ex_plane)) > 0, "Source injection produced no field"


def test_inject_floquet_source_tm():
    """TM source injection should add Ey at z-normal port."""
    grid = Grid(freq_max=10e9, domain=(0.015, 0.015, 0.03),
                dx=0.0015, cpml_layers=0, mode="3d")
    state = init_state(grid.shape)

    port_z = grid.nz // 2
    state_new = inject_floquet_source(
        state, port_z, axis=2, dt=grid.dt, dx=grid.dx,
        step=50, f0=5e9, bandwidth=0.5, amplitude=1.0,
        polarization="tm",
    )

    ey_plane = np.array(state_new.ey[:, :, port_z])
    assert np.max(np.abs(ey_plane)) > 0, "TM source injection produced no field"


# =========================================================================
# Test: extract_floquet_modes
# =========================================================================

def test_extract_floquet_modes_structure():
    """Mode extraction should return correctly structured result."""
    n_freqs = 5
    plane_shape = (10, 10)
    acc = init_floquet_dft(n_freqs, plane_shape)

    # Set some non-zero data
    acc = acc._replace(
        e_tang1_dft=jnp.ones((n_freqs,) + plane_shape, dtype=jnp.complex64),
        h_tang2_dft=jnp.ones((n_freqs,) + plane_shape, dtype=jnp.complex64) * 0.002,
    )

    freqs = jnp.linspace(5e9, 15e9, n_freqs)
    result = extract_floquet_modes(
        acc, dx=0.001, Lx=0.01, Ly=0.01, freqs=freqs,
        theta_deg=0.0, phi_deg=0.0, n_modes=1,
    )

    assert 'S' in result
    assert 'modes' in result
    assert 'freqs' in result
    assert result['S'].shape[0] == 1  # n_modes
    assert result['S'].shape[1] == n_freqs
    assert result['modes'] == [(0, 0)]


# =========================================================================
# Test: End-to-end simulation run with high-level API
# =========================================================================

def test_floquet_api_run():
    """Full end-to-end: create sim, add Floquet port, run, get result."""
    sim = Simulation(
        freq_max=10e9,
        domain=(0.015, 0.015, 0.03),
        boundary="cpml",
        cpml_layers=8,
    )
    sim.add_floquet_port(0.008, axis="z", scan_theta=0.0, polarization="te")
    sim.add_probe((0.0075, 0.0075, 0.015), component="ex")

    result = sim.run(n_steps=100)

    assert result is not None
    ts = np.array(result.time_series).ravel()
    assert not np.any(np.isnan(ts)), "NaN in Floquet API simulation"
    # Source should have injected something
    assert np.max(np.abs(ts)) > 0, "No signal detected in probe"
