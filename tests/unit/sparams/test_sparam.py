"""S-parameter extraction validation (lumped-port limiting cases).

Tests:
1. Lumped port in a PEC cavity: S11 near 0 dB in-band (full reflection).
   A lossless PEC cavity dissipates nothing, so |S11| ≈ 1; pinned as a
   loose limiting-case bound (mid-band mean |S11| > -3 dB).
2. Lumped port injects energy: E and H fields become non-zero after
   driving the port (excitation smoke test).

There is NO matched-load absorption gate (S11 < -20 dB) here: a lumped port
terminated in its own impedance in an open (CPML) domain is not exercised in
this file, so that absorption physics is not asserted.
"""

import numpy as np
import jax.numpy as jnp

from rfx.grid import Grid
from rfx.core.yee import init_state, init_materials, update_e, update_h
from rfx.boundaries.pec import apply_pec
from rfx.sources.sources import GaussianPulse, LumpedPort, setup_lumped_port, apply_lumped_port
from rfx.probes.probes import (
    init_sparam_probe, update_lumped_drive_ref_probe,
    update_sparam_probe, extract_s11,
)


def test_lumped_port_pec_cavity_s11():
    """S11 of a lumped port in a PEC cavity should be ~0 dB (full reflection).

    A lossless PEC cavity has no dissipation, so all injected power
    must be reflected back. |S11| ≈ 1 (0 dB) at all frequencies.
    """
    a, b, d = 0.05, 0.05, 0.025
    grid = Grid(freq_max=5e9, domain=(a, b, d), cpml_layers=0)
    state = init_state(grid.shape)
    materials = init_materials(grid.shape)

    # Lumped port at center
    port_pos = (a / 2, b / 2, d / 2)
    pulse = GaussianPulse(f0=3e9, bandwidth=0.8, amplitude=1.0)
    port = LumpedPort(
        position=port_pos,
        component="ez",
        impedance=50.0,
        excitation=pulse,
    )

    # Fold port impedance into materials
    materials = setup_lumped_port(grid, port, materials)

    # Frequency points for S-parameter extraction
    freqs = jnp.linspace(1e9, 5e9, 50)
    dt, dx = grid.dt, grid.dx
    num_steps = grid.num_timesteps(num_periods=60)
    sprobe = init_sparam_probe(grid, port, freqs, dft_total_steps=num_steps)

    for n in range(num_steps):
        t = n * dt
        state = update_h(state, materials, dt, dx)
        state = update_e(state, materials, dt, dx)
        state = apply_pec(state)
        # Both slots, the way the JIT scan uses them: the PRE-injection drive
        # sample into v_ref, the physical V/I AFTER injection
        # (scripts/diagnostics/lumped_port_known_load_line.py).  This loop
        # sampled only before injection while extract_s11 became the DRIVEN
        # terminal reflection, and the mismatched pair read |S11| up to 1.4468
        # on this lossless cavity — above unity, which a passive structure
        # cannot do.  On the two slots it reads 0.9774 … 0.9994.
        sprobe = update_lumped_drive_ref_probe(sprobe, state, grid, port, dt)
        state = apply_lumped_port(state, grid, port, t, materials)
        sprobe = update_sparam_probe(sprobe, state, grid, port, dt)

    s11 = extract_s11(sprobe, z0=50.0)
    s11_db = 20 * np.log10(np.maximum(np.abs(np.array(s11)), 1e-10))

    # In the excitation bandwidth (1-5 GHz), S11 should be near 0 dB
    # (PEC cavity = total reflection). Allow some tolerance for
    # numerical artifacts at band edges.
    mid_band = (np.array(freqs) > 1.5e9) & (np.array(freqs) < 4.5e9)
    s11_mid = s11_db[mid_band]

    print("\nS11 in PEC cavity (mid-band):")
    print(f"  Mean: {np.mean(s11_mid):.1f} dB")
    print(f"  Min:  {np.min(s11_mid):.1f} dB")
    print(f"  Max:  {np.max(s11_mid):.1f} dB")

    # Two-sided.  The lower bound is the original one: a PEC cavity reflects
    # most of what it is given.  The upper bound is the physics the module
    # docstring already claims and the old one-sided gate could not see — a
    # passive structure cannot return more than it receives, and a gate that
    # only looks down let |S11| = 1.4468 through.  The margin is the measured
    # rounding on this fixture: max |S11| reads 0.9994, so 1 + 1e-3 is the
    # float32 headroom and nothing more.
    assert np.mean(s11_mid) > -3.0, \
        f"Mean S11 {np.mean(s11_mid):.1f} dB too low for PEC cavity"
    s11_mag = np.abs(np.array(s11))
    assert np.max(s11_mag) <= 1.0 + 1e-3, (
        f"|S11| = {np.max(s11_mag):.5f} exceeds unity on a lossless PEC "
        "cavity, which a passive structure cannot do")


def test_lumped_port_injects_energy():
    """Lumped port should inject energy into the simulation."""
    grid = Grid(freq_max=3e9, domain=(0.05, 0.05, 0.025), cpml_layers=0)
    state = init_state(grid.shape)
    materials = init_materials(grid.shape)

    port_pos = (0.025, 0.025, 0.0125)
    pulse = GaussianPulse(f0=2e9, bandwidth=0.5, amplitude=1.0)
    port = LumpedPort(
        position=port_pos,
        component="ez",
        impedance=50.0,
        excitation=pulse,
    )

    # Fold port impedance into materials
    materials = setup_lumped_port(grid, port, materials)

    dt, dx = grid.dt, grid.dx

    # Run 100 steps with port excitation
    for n in range(100):
        t = n * dt
        state = update_h(state, materials, dt, dx)
        state = update_e(state, materials, dt, dx)
        state = apply_pec(state)
        state = apply_lumped_port(state, grid, port, t, materials)

    # Check that fields are non-zero
    total_e = float((state.ex**2 + state.ey**2 + state.ez**2).sum())
    total_h = float((state.hx**2 + state.hy**2 + state.hz**2).sum())

    print("\nAfter 100 steps with lumped port:")
    print(f"  Total E²: {total_e:.4e}")
    print(f"  Total H²: {total_h:.4e}")

    assert total_e > 0, "Lumped port did not inject any E-field energy"
    assert total_h > 0, "Lumped port did not inject any H-field energy"
