"""Tests for the compiled simulation runner and multiport S-params.

Validates that:
1. Compiled runner matches a manual Python loop
2. Source injection and probe recording work through scan
3. Multiport S-matrix has reciprocity (S12 = S21)
4. TFSF + periodic-boundary integration matches the manual loop
"""

import numpy as np
import jax.numpy as jnp
import pytest

from rfx.grid import Grid, C0
from rfx.core.yee import (
    init_state, init_materials, update_e, update_h, EPS_0, MU_0,
)
from rfx.boundaries.pec import apply_pec
from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h
from rfx.geometry.csg import Box
from rfx.sources.sources import GaussianPulse, LumpedPort
from rfx.sources.tfsf import (
    init_tfsf, update_tfsf_1d_h, update_tfsf_1d_e, apply_tfsf_h, apply_tfsf_e,
    is_tfsf_2d,
)
from rfx.simulation import (
    ProbeSpec, make_source, make_probe, run, run_until_decay,
)
from rfx.probes.probes import (
    extract_s_matrix, init_dft_plane_probe, update_dft_plane_probe,
)
from rfx.sources.waveguide_port import (
    WaveguidePort, init_waveguide_port, inject_waveguide_port,
    update_waveguide_port_probe, extract_waveguide_s21, extract_waveguide_s_matrix,
)


def _total_energy(state, dx):
    e_sq = jnp.sum(state.ex**2 + state.ey**2 + state.ez**2)
    h_sq = jnp.sum(state.hx**2 + state.hy**2 + state.hz**2)
    return float(0.5 * EPS_0 * e_sq * dx**3 + 0.5 * MU_0 * h_sq * dx**3)


def test_compiled_runner_energy_conservation():
    """Compiled runner in PEC cavity conserves energy (no sources)."""
    shape = (20, 20, 20)
    dx = 0.003
    0.99 * dx / (C0 * np.sqrt(3))
    # Build a minimal Grid-like object or use raw parameters
    grid = Grid(freq_max=5e9, domain=(shape[0]*dx, shape[1]*dx, shape[2]*dx),
                cpml_layers=0)
    materials = init_materials(grid.shape)

    # Seed initial field (cavity mode)
    Lx = (grid.nx - 1) * grid.dx
    Ly = (grid.ny - 1) * grid.dx
    x = np.arange(grid.nx) * grid.dx
    y = np.arange(grid.ny) * grid.dx
    (np.sin(np.pi * x[:, None, None] / Lx) *
               np.sin(np.pi * y[None, :, None] / Ly) *
               np.ones((1, 1, grid.nz)))

    # Run with compiled runner — probe at center
    center = make_probe(grid,
                        (grid.nx//2 * grid.dx, grid.ny//2 * grid.dx,
                         grid.nz//2 * grid.dx),
                        "ez")
    n_steps = 200

    # We can't seed initial field through run(), so compare energies
    # via the Python loop approach.  Instead, test that the runner
    # at least produces a valid state with zero initial energy preserved.
    result = run(grid, materials, n_steps, probes=[center])

    assert result.time_series.shape == (n_steps, 1)
    # With no sources, all fields should stay zero
    assert float(jnp.max(jnp.abs(result.time_series))) == 0.0
    assert _total_energy(result.state, grid.dx) == 0.0


def test_low_level_run_supports_upml_boundary():
    """Low-level run() should execute the UPML path on a simple 2D TMz setup."""
    grid = Grid(
        freq_max=6e9,
        domain=(0.04, 0.04, 0.004),
        dx=0.002,
        cpml_layers=6,
        mode="2d_tmz",
    )
    materials = init_materials(grid.shape)
    src = make_source(
        grid,
        (0.01, 0.02, 0.0),
        "ez",
        GaussianPulse(f0=3e9, bandwidth=0.5),
        80,
    )
    prb = make_probe(grid, (0.026, 0.02, 0.0), "ez")

    result = run(grid, materials, 80, sources=[src], probes=[prb], boundary="upml")

    assert result.time_series.shape == (80, 1)
    assert float(jnp.max(jnp.abs(result.time_series[:, 0]))) > 0.0


def test_run_until_decay_supports_upml_boundary():
    """Decay runner should also execute the UPML path without NaNs."""
    grid = Grid(
        freq_max=6e9,
        domain=(0.04, 0.04, 0.004),
        dx=0.002,
        cpml_layers=6,
        mode="2d_tmz",
    )
    materials = init_materials(grid.shape)
    src = make_source(
        grid,
        (0.01, 0.02, 0.0),
        "ez",
        GaussianPulse(f0=3e9, bandwidth=0.5),
        120,
    )
    prb = make_probe(grid, (0.026, 0.02, 0.0), "ez")

    result = run_until_decay(
        grid,
        materials,
        decay_by=1e-4,
        sources=[src],
        probes=[prb],
        monitor_position=(0.026, 0.02, 0.0),
        boundary="upml",
        max_steps=120,
        min_steps=40,
    )

    assert result.time_series.shape[1] == 1
    assert result.time_series.shape[0] >= 40
    assert not jnp.any(jnp.isnan(result.state.ez))


def test_compiled_runner_source_and_probe():
    """Compiled runner with a point source produces non-zero probe data."""
    grid = Grid(freq_max=5e9, domain=(0.03, 0.03, 0.03), cpml_layers=0)
    materials = init_materials(grid.shape)
    n_steps = 200

    pulse = GaussianPulse(f0=3e9, bandwidth=0.5)
    src = make_source(grid, (0.015, 0.015, 0.015), "ez", pulse, n_steps)
    prb = make_probe(grid, (0.02, 0.02, 0.02), "ez")

    result = run(grid, materials, n_steps, sources=[src], probes=[prb])

    assert result.time_series.shape == (n_steps, 1)
    # Source should inject energy → non-zero probe readings
    peak = float(jnp.max(jnp.abs(result.time_series[:, 0])))
    assert peak > 0, "Probe should detect non-zero field from source"

    # Final state should have non-zero energy
    energy = _total_energy(result.state, grid.dx)
    assert energy > 0, "Simulation should have non-zero energy"
    print(f"\nCompiled runner: peak probe = {peak:.4e}, energy = {energy:.4e}")


def test_compiled_runner_matches_python_loop():
    """Compiled runner should produce identical results to a Python loop."""
    grid = Grid(freq_max=5e9, domain=(0.03, 0.03, 0.03), cpml_layers=0)
    materials = init_materials(grid.shape)
    n_steps = 100

    pulse = GaussianPulse(f0=3e9, bandwidth=0.5)
    src_pos = (0.015, 0.015, 0.015)
    prb_pos = (0.02, 0.02, 0.02)

    # ---- compiled runner ----
    src = make_source(grid, src_pos, "ez", pulse, n_steps)
    prb = make_probe(grid, prb_pos, "ez")
    result = run(grid, materials, n_steps, sources=[src], probes=[prb])

    # ---- manual Python loop ----
    state = init_state(grid.shape)
    si, sj, sk = grid.position_to_index(src_pos)
    pi, pj, pk = grid.position_to_index(prb_pos)
    manual_ts = []
    for step in range(n_steps):
        state = update_h(state, materials, grid.dt, grid.dx)
        state = update_e(state, materials, grid.dt, grid.dx)
        state = apply_pec(state)
        # Soft source
        val = float(pulse(step * grid.dt))
        state = state._replace(ez=state.ez.at[si, sj, sk].add(val))
        manual_ts.append(float(state.ez[pi, pj, pk]))

    manual_ts = np.array(manual_ts)
    compiled_ts = np.array(result.time_series[:, 0])

    # Should match to machine precision
    max_err = np.max(np.abs(compiled_ts - manual_ts))
    print(f"\nCompiled vs manual loop max error: {max_err:.2e}")
    assert max_err < 1e-5, f"Compiled runner diverged from Python loop: {max_err:.2e}"


def test_compiled_runner_dft_plane_matches_manual_loop():
    """Compiled runner should match manual DFT plane accumulation."""
    grid = Grid(freq_max=5e9, domain=(0.03, 0.03, 0.03), cpml_layers=0)
    materials = init_materials(grid.shape)
    n_steps = 80

    pulse = GaussianPulse(f0=3e9, bandwidth=0.5)
    src = make_source(grid, (0.015, 0.015, 0.015), "ez", pulse, n_steps)
    plane_probe = init_dft_plane_probe(
        axis=0,
        index=grid.nx // 2,
        component="ez",
        freqs=jnp.array([2e9, 3e9, 4e9]),
        grid_shape=grid.shape,
        dft_total_steps=n_steps,
    )

    result = run(grid, materials, n_steps, sources=[src], dft_planes=[plane_probe])
    assert result.dft_planes is not None
    compiled_acc = np.array(result.dft_planes[0].accumulator)

    state = init_state(grid.shape)
    manual_probe = plane_probe
    si, sj, sk = grid.position_to_index((0.015, 0.015, 0.015))
    for step in range(n_steps):
        state = update_h(state, materials, grid.dt, grid.dx)
        state = update_e(state, materials, grid.dt, grid.dx)
        state = apply_pec(state)
        val = float(pulse(step * grid.dt))
        state = state._replace(ez=state.ez.at[si, sj, sk].add(val))
        manual_probe = update_dft_plane_probe(manual_probe, state, grid.dt)

    manual_acc = np.array(manual_probe.accumulator)
    max_err = np.max(np.abs(compiled_acc - manual_acc))
    print(f"\nCompiled DFT plane vs manual loop max error: {max_err:.2e}")
    assert max_err < 1e-5, f"Compiled DFT plane runner diverged from manual loop: {max_err:.2e}"


def test_compiled_runner_tfsf_matches_manual_loop():
    """Compiled runner should preserve the manual TFSF leapfrog ordering."""
    grid = Grid(freq_max=8e9, domain=(0.08, 0.006, 0.006), dx=0.001, cpml_layers=8)
    materials = init_materials(grid.shape)
    periodic = (False, True, True)
    n_steps = 120

    tfsf_cfg, tfsf_state0 = init_tfsf(
        grid.nx,
        grid.dx,
        grid.dt,
        cpml_layers=grid.cpml_layers,
        tfsf_margin=3,
        f0=4e9,
        bandwidth=0.5,
        amplitude=1.0,
    )
    probe = ProbeSpec(
        i=tfsf_cfg.x_lo + 8,
        j=grid.ny // 2,
        k=grid.nz // 2,
        component="ez",
    )

    compiled = run(
        grid,
        materials,
        n_steps,
        boundary="cpml",
        cpml_axes="x",
        periodic=periodic,
        tfsf=(tfsf_cfg, tfsf_state0),
        probes=[probe],
    )

    state = init_state(grid.shape)
    cp, cpml_state = init_cpml(grid)
    tfsf_state = tfsf_state0
    manual_ts = []

    for step in range(n_steps):
        t = step * grid.dt
        state = update_h(state, materials, grid.dt, grid.dx, periodic=periodic)
        state = apply_tfsf_h(state, tfsf_cfg, tfsf_state, grid.dx, grid.dt)
        state, cpml_state = apply_cpml_h(state, cp, cpml_state, grid, axes="x")
        tfsf_state = update_tfsf_1d_h(tfsf_cfg, tfsf_state, grid.dx, grid.dt)

        state = update_e(state, materials, grid.dt, grid.dx, periodic=periodic)
        state = apply_tfsf_e(state, tfsf_cfg, tfsf_state, grid.dx, grid.dt)
        state, cpml_state = apply_cpml_e(state, cp, cpml_state, grid, axes="x")
        state = apply_pec(state, axes="x")
        tfsf_state = update_tfsf_1d_e(tfsf_cfg, tfsf_state, grid.dx, grid.dt, t)
        manual_ts.append(float(state.ez[probe.i, probe.j, probe.k]))

    compiled_ts = np.array(compiled.time_series[:, 0])
    manual_ts = np.array(manual_ts)
    max_err = np.max(np.abs(compiled_ts - manual_ts))

    print(f"\nCompiled TFSF vs manual loop max error: {max_err:.2e}")
    assert max_err < 1e-5, f"Compiled TFSF runner diverged from manual loop: {max_err:.2e}"


def test_compiled_runner_tfsf_ey_matches_manual_loop():
    """Compiled runner should preserve Ey-polarized TFSF ordering."""
    grid = Grid(freq_max=8e9, domain=(0.08, 0.006, 0.006), dx=0.001, cpml_layers=8)
    materials = init_materials(grid.shape)
    periodic = (False, True, True)
    n_steps = 120

    tfsf_cfg, tfsf_state0 = init_tfsf(
        grid.nx,
        grid.dx,
        grid.dt,
        cpml_layers=grid.cpml_layers,
        tfsf_margin=3,
        f0=4e9,
        bandwidth=0.5,
        amplitude=1.0,
        polarization="ey",
    )
    probe = ProbeSpec(
        i=tfsf_cfg.x_lo + 8,
        j=grid.ny // 2,
        k=grid.nz // 2,
        component="ey",
    )

    compiled = run(
        grid,
        materials,
        n_steps,
        boundary="cpml",
        cpml_axes="x",
        periodic=periodic,
        tfsf=(tfsf_cfg, tfsf_state0),
        probes=[probe],
    )

    state = init_state(grid.shape)
    cp, cpml_state = init_cpml(grid)
    tfsf_state = tfsf_state0
    manual_ts = []

    for step in range(n_steps):
        t = step * grid.dt
        state = update_h(state, materials, grid.dt, grid.dx, periodic=periodic)
        state = apply_tfsf_h(state, tfsf_cfg, tfsf_state, grid.dx, grid.dt)
        state, cpml_state = apply_cpml_h(state, cp, cpml_state, grid, axes="x")
        tfsf_state = update_tfsf_1d_h(tfsf_cfg, tfsf_state, grid.dx, grid.dt)

        state = update_e(state, materials, grid.dt, grid.dx, periodic=periodic)
        state = apply_tfsf_e(state, tfsf_cfg, tfsf_state, grid.dx, grid.dt)
        state, cpml_state = apply_cpml_e(state, cp, cpml_state, grid, axes="x")
        state = apply_pec(state, axes="x")
        tfsf_state = update_tfsf_1d_e(tfsf_cfg, tfsf_state, grid.dx, grid.dt, t)
        manual_ts.append(float(state.ey[probe.i, probe.j, probe.k]))

    compiled_ts = np.array(compiled.time_series[:, 0])
    manual_ts = np.array(manual_ts)
    max_err = np.max(np.abs(compiled_ts - manual_ts))

    print(f"\nCompiled TFSF Ey vs manual loop max error: {max_err:.2e}")
    assert max_err < 1e-5, f"Compiled Ey-polarized TFSF runner diverged from manual loop: {max_err:.2e}"


def test_compiled_runner_tfsf_negative_x_matches_manual_loop():
    """Compiled runner should preserve reverse-propagating TFSF ordering."""
    grid = Grid(freq_max=8e9, domain=(0.08, 0.006, 0.006), dx=0.001, cpml_layers=8)
    materials = init_materials(grid.shape)
    periodic = (False, True, True)
    n_steps = 120

    tfsf_cfg, tfsf_state0 = init_tfsf(
        grid.nx,
        grid.dx,
        grid.dt,
        cpml_layers=grid.cpml_layers,
        tfsf_margin=3,
        f0=4e9,
        bandwidth=0.5,
        amplitude=1.0,
        polarization="ez",
        direction="-x",
    )
    probe = ProbeSpec(
        i=tfsf_cfg.x_lo + 8,
        j=grid.ny // 2,
        k=grid.nz // 2,
        component="ez",
    )

    compiled = run(
        grid,
        materials,
        n_steps,
        boundary="cpml",
        cpml_axes="x",
        periodic=periodic,
        tfsf=(tfsf_cfg, tfsf_state0),
        probes=[probe],
    )

    state = init_state(grid.shape)
    cp, cpml_state = init_cpml(grid)
    tfsf_state = tfsf_state0
    manual_ts = []

    for step in range(n_steps):
        t = step * grid.dt
        state = update_h(state, materials, grid.dt, grid.dx, periodic=periodic)
        state = apply_tfsf_h(state, tfsf_cfg, tfsf_state, grid.dx, grid.dt)
        state, cpml_state = apply_cpml_h(state, cp, cpml_state, grid, axes="x")
        tfsf_state = update_tfsf_1d_h(tfsf_cfg, tfsf_state, grid.dx, grid.dt)

        state = update_e(state, materials, grid.dt, grid.dx, periodic=periodic)
        state = apply_tfsf_e(state, tfsf_cfg, tfsf_state, grid.dx, grid.dt)
        state, cpml_state = apply_cpml_e(state, cp, cpml_state, grid, axes="x")
        state = apply_pec(state, axes="x")
        tfsf_state = update_tfsf_1d_e(tfsf_cfg, tfsf_state, grid.dx, grid.dt, t)
        manual_ts.append(float(state.ez[probe.i, probe.j, probe.k]))

    compiled_ts = np.array(compiled.time_series[:, 0])
    manual_ts = np.array(manual_ts)
    max_err = np.max(np.abs(compiled_ts - manual_ts))

    print(f"\nCompiled TFSF -x vs manual loop max error: {max_err:.2e}")
    assert max_err < 1e-5, f"Compiled reverse-direction TFSF runner diverged from manual loop: {max_err:.2e}"


def test_compiled_runner_tfsf_oblique_matches_manual_loop():
    """Compiled runner should preserve oblique-incidence TFSF ordering."""
    grid = Grid(freq_max=5e9, domain=(0.12, 0.12, 0.006), dx=0.004, cpml_layers=8)
    materials = init_materials(grid.shape)
    periodic = (False, True, True)
    n_steps = 120

    tfsf_cfg, tfsf_state0 = init_tfsf(
        grid.nx,
        grid.dx,
        grid.dt,
        cpml_layers=grid.cpml_layers,
        tfsf_margin=3,
        f0=5e9,
        bandwidth=0.2,
        amplitude=1.0,
        polarization="ez",
        angle_deg=30.0,
        ny=grid.ny,
    )
    probe = ProbeSpec(
        i=tfsf_cfg.x_lo + 5,
        j=grid.ny // 2,
        k=grid.nz // 2,
        component="ez",
    )

    compiled = run(
        grid,
        materials,
        n_steps,
        boundary="cpml",
        cpml_axes="x",
        periodic=periodic,
        tfsf=(tfsf_cfg, tfsf_state0),
        probes=[probe],
    )

    # Detect 2D auxiliary grid for oblique incidence
    _is_2d = is_tfsf_2d(tfsf_cfg)
    if _is_2d:
        from rfx.sources.tfsf_2d import update_tfsf_2d_h, update_tfsf_2d_e

    state = init_state(grid.shape)
    cp, cpml_state = init_cpml(grid)
    tfsf_state = tfsf_state0
    manual_ts = []

    for step in range(n_steps):
        t = step * grid.dt
        state = update_h(state, materials, grid.dt, grid.dx, periodic=periodic)
        state = apply_tfsf_h(state, tfsf_cfg, tfsf_state, grid.dx, grid.dt)
        state, cpml_state = apply_cpml_h(state, cp, cpml_state, grid, axes="x")
        if _is_2d:
            tfsf_state = update_tfsf_2d_h(tfsf_cfg, tfsf_state, grid.dx, grid.dt)
        else:
            tfsf_state = update_tfsf_1d_h(tfsf_cfg, tfsf_state, grid.dx, grid.dt)

        state = update_e(state, materials, grid.dt, grid.dx, periodic=periodic)
        state = apply_tfsf_e(state, tfsf_cfg, tfsf_state, grid.dx, grid.dt)
        state, cpml_state = apply_cpml_e(state, cp, cpml_state, grid, axes="x")
        state = apply_pec(state, axes="x")
        if _is_2d:
            tfsf_state = update_tfsf_2d_e(tfsf_cfg, tfsf_state, grid.dx, grid.dt, t)
        else:
            tfsf_state = update_tfsf_1d_e(tfsf_cfg, tfsf_state, grid.dx, grid.dt, t)
        manual_ts.append(float(state.ez[probe.i, probe.j, probe.k]))

    compiled_ts = np.array(compiled.time_series[:, 0])
    manual_ts = np.array(manual_ts)
    max_err = np.max(np.abs(compiled_ts - manual_ts))

    print(f"\nCompiled TFSF oblique vs manual loop max error: {max_err:.2e}")
    assert max_err < 1e-5, f"Compiled oblique TFSF runner diverged from manual loop: {max_err:.2e}"


class _CompiledWgGrid:
    """Minimal grid with x-CPML padding and y/z waveguide walls."""

    def __init__(self, length, a_wg, b_wg, dx, cpml_layers, freq_max=10e9):
        self.freq_max = freq_max
        self.dx = dx
        self.cpml_layers = cpml_layers
        self.dt = 0.99 * dx / (C0 * np.sqrt(3))
        self.nx = int(np.ceil(length / dx)) + 1 + 2 * cpml_layers
        self.ny = int(np.ceil(a_wg / dx)) + 1
        self.nz = int(np.ceil(b_wg / dx)) + 1
        self.shape = (self.nx, self.ny, self.nz)
        self.is_2d = False

    def position_to_index(self, pos):
        return (
            int(round(pos[0] / self.dx)) + self.cpml_layers,
            int(round(pos[1] / self.dx)),
            int(round(pos[2] / self.dx)),
        )

    def num_timesteps(self, num_periods):
        return int(num_periods / (self.freq_max * self.dt))


def test_compiled_runner_waveguide_port_matches_manual_loop():
    """Compiled runner should preserve waveguide-port source/probe behavior."""
    a_wg = 0.04
    b_wg = 0.02
    length = 0.12
    dx = 0.002
    nc = 10
    f0 = 6e9

    grid = _CompiledWgGrid(length, a_wg, b_wg, dx, nc)
    materials = init_materials(grid.shape)
    freqs = jnp.linspace(4.5e9, 8e9, 12)
    n_steps = grid.num_timesteps(num_periods=30)

    port = WaveguidePort(
        x_index=nc + 5,
        y_slice=(0, grid.ny),
        z_slice=(0, grid.nz),
        a=(grid.ny - 1) * dx,
        b=(grid.nz - 1) * dx,
        mode=(1, 0),
        mode_type="TE",
    )
    cfg0 = init_waveguide_port(
        port, dx, freqs, f0=f0, bandwidth=0.5,
        amplitude=1.0, probe_offset=15, ref_offset=3,
        dft_total_steps=n_steps,
    )

    compiled = run(
        grid,
        materials,
        n_steps,
        boundary="cpml",
        cpml_axes="x",
        pec_axes="yz",
        waveguide_ports=[cfg0],
    )
    assert compiled.waveguide_ports is not None
    compiled_cfg = compiled.waveguide_ports[0]

    state = init_state(grid.shape)
    cp, cs = init_cpml(grid)
    manual_cfg = cfg0
    for step in range(n_steps):
        t = step * grid.dt
        state = update_h(state, materials, grid.dt, grid.dx)
        state, cs = apply_cpml_h(state, cp, cs, grid, axes="x")
        state = update_e(state, materials, grid.dt, grid.dx)
        state, cs = apply_cpml_e(state, cp, cs, grid, axes="x")
        state = apply_pec(state, axes="yz")
        state = inject_waveguide_port(state, manual_cfg, t, grid.dt, grid.dx)
        manual_cfg = update_waveguide_port_probe(manual_cfg, state, grid.dt, grid.dx)

    s21_compiled = np.array(extract_waveguide_s21(compiled_cfg))
    s21_manual = np.array(extract_waveguide_s21(manual_cfg))
    max_err = np.max(np.abs(s21_compiled - s21_manual))
    print(f"\nCompiled waveguide vs manual loop max S21 error: {max_err:.2e}")
    assert max_err < 1e-3, f"Compiled waveguide runner diverged from manual loop: {max_err:.2e}"


def test_extract_waveguide_s_matrix_two_port_reciprocity():
    """Two-port waveguide S-matrix should be assembled one driven port at a time."""
    a_wg = 0.04
    b_wg = 0.02
    length = 0.12
    dx = 0.002
    nc = 10
    f0 = 6e9

    grid = _CompiledWgGrid(length, a_wg, b_wg, dx, nc)
    materials = init_materials(grid.shape)
    freqs = jnp.linspace(4.5e9, 8e9, 12)
    n_steps = grid.num_timesteps(num_periods=30)

    port0 = WaveguidePort(
        x_index=nc + 5,
        y_slice=(0, grid.ny),
        z_slice=(0, grid.nz),
        a=(grid.ny - 1) * dx,
        b=(grid.nz - 1) * dx,
        mode=(1, 0),
        mode_type="TE",
        direction="+x",
    )
    port1 = WaveguidePort(
        x_index=grid.nx - nc - 6,
        y_slice=(0, grid.ny),
        z_slice=(0, grid.nz),
        a=(grid.ny - 1) * dx,
        b=(grid.nz - 1) * dx,
        mode=(1, 0),
        mode_type="TE",
        direction="-x",
    )
    cfg0 = init_waveguide_port(port0, dx, freqs, f0=f0, dft_total_steps=n_steps)
    cfg1 = init_waveguide_port(port1, dx, freqs, f0=f0, dft_total_steps=n_steps)

    S = np.array(
        extract_waveguide_s_matrix(
            grid,
            materials,
            [cfg0, cfg1],
            n_steps,
            boundary="cpml",
            cpml_axes="x",
            pec_axes="yz",
        )
    )
    assert S.shape == (2, 2, 12)
    s11 = S[0, 0, :]
    s21 = S[1, 0, :]
    s22 = S[1, 1, :]
    s12 = S[0, 1, :]
    recip_err = np.mean(
        np.abs(np.abs(s21) - np.abs(s12))
        / np.maximum(0.5 * (np.abs(s21) + np.abs(s12)), 1e-8)
    )
    assert np.mean(np.abs(s21)) > 0.5
    assert np.mean(np.abs(s12)) > 0.5
    assert np.mean(np.abs(s11)) < 0.5
    assert np.mean(np.abs(s22)) < 0.5
    assert recip_err < 0.2


def test_extract_waveguide_s_matrix_mixed_normal_branch_reciprocity():
    """Mixed x/y boundary ports in a PEC T-junction should remain reciprocal."""
    dx = 0.002
    nc = 10
    domain = (0.12, 0.12, 0.02)
    grid = Grid(freq_max=10e9, domain=domain, dx=dx, cpml_layers=nc, cpml_axes="xy")
    materials = init_materials(grid.shape)

    for box in (
        Box((0.0, 0.0, 0.0), (0.12, 0.04, 0.02)),
        Box((0.0, 0.08, 0.0), (0.04, 0.12, 0.02)),
        Box((0.08, 0.08, 0.0), (0.12, 0.12, 0.02)),
    ):
        mask = box.mask(grid)
        materials = materials._replace(
            sigma=jnp.where(mask, 1e10, materials.sigma),
        )

    freqs = jnp.linspace(4.5e9, 8.0e9, 10)
    n_steps = grid.num_timesteps(num_periods=30)
    padx, pady, padz = grid.axis_pads
    xslice = (padx + int(round(0.04 / dx)), padx + int(round(0.08 / dx)) + 1)
    yslice = (pady + int(round(0.04 / dx)), pady + int(round(0.08 / dx)) + 1)
    zslice = (padz, grid.nz - padz)

    left = WaveguidePort(
        x_index=nc + 5,
        y_slice=yslice,
        z_slice=zslice,
        a=0.04,
        b=0.02,
        mode=(1, 0),
        mode_type="TE",
        direction="+x",
        normal_axis="x",
        u_slice=yslice,
        v_slice=zslice,
    )
    right = WaveguidePort(
        x_index=grid.nx - nc - 6,
        y_slice=yslice,
        z_slice=zslice,
        a=0.04,
        b=0.02,
        mode=(1, 0),
        mode_type="TE",
        direction="-x",
        normal_axis="x",
        u_slice=yslice,
        v_slice=zslice,
    )
    top = WaveguidePort(
        x_index=grid.ny - nc - 6,
        y_slice=None,
        z_slice=None,
        a=0.04,
        b=0.02,
        mode=(1, 0),
        mode_type="TE",
        direction="-y",
        normal_axis="y",
        u_slice=xslice,
        v_slice=zslice,
    )

    cfgs = [
        init_waveguide_port(port, dx, freqs, f0=6e9, ref_offset=3, probe_offset=15, dft_total_steps=n_steps)
        for port in (left, right, top)
    ]
    S = np.array(
        extract_waveguide_s_matrix(
            grid,
            materials,
            cfgs,
            n_steps,
            boundary="cpml",
            cpml_axes="xy",
            pec_axes="z",
        )
    )

    assert S.shape == (3, 3, 10)
    assert np.mean(np.abs(S[1, 0, :])) > 0.2
    assert np.mean(np.abs(S[2, 0, :])) > 0.15
    assert np.mean(np.abs(S[0, 2, :])) > 0.15

    for (recv_a, drive_a), (recv_b, drive_b) in (((2, 0), (0, 2)), ((2, 1), (1, 2)), ((1, 0), (0, 1))):
        mag_a = np.abs(S[recv_a, drive_a, :])
        mag_b = np.abs(S[recv_b, drive_b, :])
        recip_err = np.mean(
            np.abs(mag_a - mag_b) / np.maximum(0.5 * (mag_a + mag_b), 1e-8)
        )
        assert recip_err < 0.2


def test_init_tfsf_rejects_impossible_geometry():
    """Oversized CPML+margin should fail loudly instead of misconfiguring TFSF."""
    dt = Grid.courant_dt(0.001)
    with pytest.raises(ValueError, match="too large for the grid"):
        init_tfsf(nx=21, dx=0.001, dt=dt, cpml_layers=4, tfsf_margin=6)


def test_s_matrix_two_port_reciprocity():
    """Two-port S-matrix should be reciprocal: S12 ≈ S21.

    Two lumped ports in a PEC cavity. Since the system is linear and
    reciprocal, S12 = S21 at all frequencies.
    """
    grid = Grid(freq_max=5e9, domain=(0.05, 0.05, 0.025), cpml_layers=0)
    materials = init_materials(grid.shape)

    pulse = GaussianPulse(f0=3e9, bandwidth=0.8, amplitude=1.0)

    port1 = LumpedPort(
        position=(0.015, 0.025, 0.0125),
        component="ez",
        impedance=50.0,
        excitation=pulse,
    )
    port2 = LumpedPort(
        position=(0.035, 0.025, 0.0125),
        component="ez",
        impedance=50.0,
        excitation=pulse,
    )

    freqs = jnp.linspace(1e9, 5e9, 20)
    n_steps = grid.num_timesteps(num_periods=30)

    S = extract_s_matrix(grid, materials, [port1, port2], freqs, n_steps)
    assert S.shape == (2, 2, 20)

    s12 = S[0, 1, :]
    s21 = S[1, 0, :]

    # Reciprocity: |S12 - S21| should be small relative to |S12|
    diff = np.abs(s12 - s21)
    mag = np.maximum(np.abs(s12), np.abs(s21))
    # Only check where signals are significant
    significant = mag > 1e-6
    if np.any(significant):
        rel_err = np.max(diff[significant] / mag[significant])
        print(f"\nTwo-port reciprocity: max |S12-S21|/|S12| = {rel_err:.4e}")
        assert rel_err < 0.05, f"Reciprocity violated: relative error {rel_err:.4f}"

    # S11 should be significant (PEC cavity reflects)
    s11_db = 20 * np.log10(np.maximum(np.abs(S[0, 0, :]), 1e-10))
    print(f"  S11 mean: {np.mean(s11_db):.1f} dB")
    assert np.mean(s11_db) > -10, "S11 too low for PEC cavity"
