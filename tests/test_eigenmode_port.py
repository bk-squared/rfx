"""Tests for eigenmode solver integration with waveguide ports."""

import numpy as np
import jax.numpy as jnp

from rfx.grid import Grid
from rfx.core.yee import init_state, init_materials, update_e, update_h
from rfx.boundaries.pec import apply_pec
from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h
from rfx.eigenmode import solve_waveguide_modes
from rfx.sources.waveguide_port import (
    WaveguidePort, init_waveguide_port, inject_waveguide_port,
    update_waveguide_port_probe, extract_waveguide_sparams,
    cutoff_frequency,
)


def _run_waveguide(port, dx, freqs, f0, n_steps, grid):
    """Run a waveguide simulation and return S21."""
    cfg = init_waveguide_port(port, dx, freqs, f0=f0, probe_offset=15,
                              ref_offset=3, dft_total_steps=n_steps)
    state = init_state(grid.shape)
    materials = init_materials(grid.shape)
    cp, cs = init_cpml(grid)

    for step in range(n_steps):
        t = step * grid.dt
        state = update_h(state, materials, grid.dt, dx)
        state, cs = apply_cpml_h(state, cp, cs, grid, axes="x")
        state = apply_pec(state)
        state = update_e(state, materials, grid.dt, dx)
        state = inject_waveguide_port(state, cfg, t, grid.dt, dx)
        state, cs = apply_cpml_e(state, cp, cs, grid, axes="x")
        state = apply_pec(state)
        cfg = update_waveguide_port_probe(cfg, state, grid.dt, dx)

    _, s21 = extract_waveguide_sparams(cfg)
    return np.abs(np.array(s21))


def test_eigenmode_port_matches_analytical():
    """Eigenmode-derived port should produce same S21 as analytical for uniform guide."""
    a, b, dx, nc = 0.04, 0.02, 0.002, 10
    f0 = 6e9
    freqs = jnp.linspace(4.5e9, 8e9, 12)
    grid = Grid(freq_max=10e9, domain=(0.12, a, b), dx=dx, cpml_layers=nc, cpml_axes="x")
    n_steps = grid.num_timesteps(num_periods=30)

    ny, nz = grid.ny, grid.nz

    # Analytical port (standard)
    port_analytical = WaveguidePort(
        x_index=nc + 5, y_slice=(0, ny), z_slice=(0, nz),
        a=(ny - 1) * dx, b=(nz - 1) * dx,
        mode=(1, 0), mode_type="TE", direction="+x",
    )
    s21_analytical = _run_waveguide(port_analytical, dx, freqs, f0, n_steps, grid)

    # Eigenmode port — use port aperture size (ny, nz) not (ny-1, nz-1)
    a_wg = (ny - 1) * dx
    b_wg = (nz - 1) * dx
    # Eigenmode solver grid must match port aperture: ny × nz cells
    modes = solve_waveguide_modes(a_wg, b_wg, dx, np.array(freqs), n_modes=1)
    mode = modes[0]

    # Resize eigenmode profiles to match port aperture (ny, nz)
    # The eigenmode solver produces (nu, nv) where nu=round(a/dx), nv=round(b/dx)
    # The port aperture is (ny, nz) which may differ by 1 cell
    from scipy.ndimage import zoom
    target_shape = (ny, nz)
    def resize_profile(prof, target):
        if prof.shape == target:
            return prof
        zoom_factors = (target[0] / prof.shape[0], target[1] / prof.shape[1])
        return zoom(prof, zoom_factors, order=1)

    ey_resized = resize_profile(np.array(mode.ey_profile), target_shape)
    ez_resized = resize_profile(np.array(mode.ez_profile), target_shape)
    hy_resized = resize_profile(np.array(mode.hy_profile), target_shape)
    hz_resized = resize_profile(np.array(mode.hz_profile), target_shape)

    port_eigen = WaveguidePort(
        x_index=nc + 5, y_slice=(0, ny), z_slice=(0, nz),
        a=a_wg, b=b_wg,
        mode=mode.mode_indices, mode_type=mode.mode_type, direction="+x",
    )
    cfg_eigen = init_waveguide_port(port_eigen, dx, freqs, f0=f0,
                                     probe_offset=15, ref_offset=3,
                                     dft_total_steps=n_steps)
    cfg_eigen = cfg_eigen._replace(
        ey_profile=jnp.array(ey_resized, dtype=jnp.float32),
        ez_profile=jnp.array(ez_resized, dtype=jnp.float32),
        hy_profile=jnp.array(hy_resized, dtype=jnp.float32),
        hz_profile=jnp.array(hz_resized, dtype=jnp.float32),
        f_cutoff=float(mode.f_cutoff),
    )

    # Run with eigenmode profiles
    state = init_state(grid.shape)
    materials = init_materials(grid.shape)
    cp, cs = init_cpml(grid)
    for step in range(n_steps):
        t = step * grid.dt
        state = update_h(state, materials, grid.dt, dx)
        state, cs = apply_cpml_h(state, cp, cs, grid, axes="x")
        state = apply_pec(state)
        state = update_e(state, materials, grid.dt, dx)
        state = inject_waveguide_port(state, cfg_eigen, t, grid.dt, dx)
        state, cs = apply_cpml_e(state, cp, cs, grid, axes="x")
        state = apply_pec(state)
        cfg_eigen = update_waveguide_port_probe(cfg_eigen, state, grid.dt, dx)

    _, s21_eigen_raw = extract_waveguide_sparams(cfg_eigen)
    s21_eigen = np.abs(np.array(s21_eigen_raw))

    f_c = cutoff_frequency(a_wg, b_wg, 1, 0)
    above = np.array(freqs) > f_c * 1.2

    diff = np.mean(np.abs(s21_analytical[above] - s21_eigen[above]))
    print(f"\nEigenmode vs analytical S21:")
    print(f"  Analytical mean |S21|: {np.mean(s21_analytical[above]):.4f}")
    print(f"  Eigenmode mean |S21|:  {np.mean(s21_eigen[above]):.4f}")
    print(f"  Mean difference:       {diff:.4f}")

    assert diff < 0.15, f"Eigenmode S21 differs from analytical by {diff:.4f}"


def test_eigenmode_partially_filled_cutoff():
    """Partially-filled waveguide eigenmode should have shifted cutoff."""
    a, b, dx = 0.04, 0.02, 0.002
    freqs = np.array([5e9, 6e9, 7e9])

    # Uniform: cutoff = c/(2a) ≈ 3.75 GHz
    modes_uniform = solve_waveguide_modes(a, b, dx, freqs, n_modes=1)
    f_c_uniform = modes_uniform[0].f_cutoff

    # Half-filled with eps_r=4: cutoff should be lower
    ny = int(round(a / dx))
    nz = int(round(b / dx))
    eps_cross = np.ones((ny, nz))
    eps_cross[:ny // 2, :] = 4.0
    modes_filled = solve_waveguide_modes(a, b, dx, freqs, n_modes=1,
                                          eps_cross=eps_cross)
    f_c_filled = modes_filled[0].f_cutoff

    # Full eps_r=4: cutoff = c/(2a*sqrt(4)) ≈ 1.875 GHz
    f_c_full = cutoff_frequency(a, b, 1, 0) / np.sqrt(4.0)

    print(f"\nPartially-filled waveguide cutoff:")
    print(f"  Uniform (eps=1):   {f_c_uniform / 1e9:.3f} GHz")
    print(f"  Half-filled (eps=4): {f_c_filled / 1e9:.3f} GHz")
    print(f"  Full (eps=4):      {f_c_full / 1e9:.3f} GHz")

    assert f_c_filled < f_c_uniform, "Half-filled cutoff should be lower than uniform"
    assert f_c_filled > f_c_full, "Half-filled cutoff should be higher than full dielectric"
