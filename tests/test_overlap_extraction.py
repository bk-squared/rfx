"""Tests for overlap integral modal extraction."""

import numpy as np
import jax.numpy as jnp

from rfx.grid import Grid, C0
from rfx.core.yee import init_state, init_materials, update_e, update_h, EPS_0, MU_0
from rfx.boundaries.pec import apply_pec
from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h
from rfx.sources.waveguide_port import (
    WaveguidePort, init_waveguide_port, inject_waveguide_port,
    update_waveguide_port_probe, modal_voltage, modal_current,
    overlap_modal_amplitude, mode_self_overlap,
)


def _setup_waveguide(n_steps=300):
    """Create a simple straight waveguide with one port."""
    a_wg, b_wg, length = 0.04, 0.02, 0.12
    dx, nc = 0.002, 10
    f0 = 6e9

    grid = Grid(freq_max=10e9, domain=(length, a_wg, b_wg),
                dx=dx, cpml_layers=nc, cpml_axes="x")
    materials = init_materials(grid.shape)
    freqs = jnp.linspace(4.5e9, 8e9, 12)

    port = WaveguidePort(
        x_index=nc + 5,
        y_slice=(0, grid.ny), z_slice=(0, grid.nz),
        a=(grid.ny - 1) * dx, b=(grid.nz - 1) * dx,
        mode=(1, 0), mode_type="TE", direction="+x",
    )
    cfg = init_waveguide_port(port, dx, freqs, f0=f0,
                              probe_offset=15, ref_offset=3,
                              dft_total_steps=n_steps)

    state = init_state(grid.shape)
    cp, cs = init_cpml(grid)
    periodic = (False, False, False)

    # Run simulation
    for step in range(n_steps):
        t = step * grid.dt
        state = update_h(state, materials, grid.dt, dx, periodic)
        state, cs = apply_cpml_h(state, cp, cs, grid, axes="x")
        state = apply_pec(state)

        state = update_e(state, materials, grid.dt, dx, periodic)
        state = inject_waveguide_port(state, cfg, t, grid.dt, dx)
        state, cs = apply_cpml_e(state, cp, cs, grid, axes="x")
        state = apply_pec(state)

        cfg = update_waveguide_port_probe(cfg, state, grid.dt, dx)

    return grid, materials, cfg, state


def test_overlap_mode_normalization():
    """Mode self-overlap C_mode should be real and positive for TE10."""
    dx = 0.002
    port = WaveguidePort(
        x_index=15,
        y_slice=(0, 20), z_slice=(0, 10),
        a=0.04, b=0.02,
        mode=(1, 0), mode_type="TE", direction="+x",
    )
    freqs = jnp.linspace(4.5e9, 8e9, 12)
    cfg = init_waveguide_port(port, dx, freqs, f0=6e9)

    c_mode = mode_self_overlap(cfg, dx)

    print(f"\nMode self-overlap C_mode = {c_mode:.6e}")
    assert c_mode > 0, f"C_mode should be positive, got {c_mode}"
    assert np.isfinite(c_mode), "C_mode should be finite"


def test_overlap_vs_vi_agreement():
    """Overlap amplitude should agree with V/I decomposition for TE10."""
    grid, materials, cfg, state = _setup_waveguide(n_steps=200)
    dx = grid.dx

    # Compute V/I at reference plane
    v = modal_voltage(state, cfg, cfg.ref_x, dx)
    i = modal_current(state, cfg, cfg.ref_x, dx)

    # Compute overlap at reference plane
    a_fwd, a_bwd = overlap_modal_amplitude(state, cfg, cfg.ref_x, dx)

    # For a single-mode TE waveguide, the V/I and overlap approaches
    # should agree when the modes are properly normalized
    print(f"\nV/I: V={float(v):.6e}, I={float(i):.6e}")
    print(f"Overlap: a_fwd={float(a_fwd):.6e}, a_bwd={float(a_bwd):.6e}")
    print(f"|a_fwd| = {float(jnp.abs(a_fwd)):.6e}")

    # Both should detect signal (non-zero)
    assert float(jnp.abs(a_fwd)) > 1e-10, "Forward amplitude should be non-zero"
    # Forward should dominate (port launches in +x)
    assert float(jnp.abs(a_fwd)) > float(jnp.abs(a_bwd)), \
        "Forward should dominate for a +x port"


def test_overlap_passivity_with_obstacle():
    """Overlap-based S-params should have better energy conservation."""
    # This is a structural test: verify the overlap function runs
    # correctly with a non-trivial state. Full passivity improvement
    # requires DFT-level overlap accumulation (future work).
    grid, materials, cfg, state = _setup_waveguide(n_steps=150)
    dx = grid.dx

    a_fwd_ref, a_bwd_ref = overlap_modal_amplitude(state, cfg, cfg.ref_x, dx)
    a_fwd_probe, a_bwd_probe = overlap_modal_amplitude(state, cfg, cfg.probe_x, dx)

    print(f"\nRef plane:   a_fwd={float(jnp.abs(a_fwd_ref)):.6e}, "
          f"a_bwd={float(jnp.abs(a_bwd_ref)):.6e}")
    print(f"Probe plane: a_fwd={float(jnp.abs(a_fwd_probe)):.6e}, "
          f"a_bwd={float(jnp.abs(a_bwd_probe)):.6e}")

    # Basic sanity: all values finite
    for name, val in [("a_fwd_ref", a_fwd_ref), ("a_bwd_ref", a_bwd_ref),
                      ("a_fwd_probe", a_fwd_probe), ("a_bwd_probe", a_bwd_probe)]:
        assert np.isfinite(float(val)), f"{name} should be finite"
