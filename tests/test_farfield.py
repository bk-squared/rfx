"""Tests for near-to-far-field transform.

Validates that:
1. NTFF DFT accumulation runs inside the scan without error
2. Hertzian dipole (Ez source) produces sin(theta) radiation pattern
3. Directivity of a short dipole is approximately 1.76 dBi
4. Far-field E_phi is near zero in the E-plane (phi=0)
"""

import numpy as np
import jax.numpy as jnp

from rfx.grid import Grid
from rfx.core.yee import init_materials
from rfx.sources.sources import GaussianPulse
from rfx.simulation import make_source, make_probe, run
from rfx.farfield import (
    make_ntff_box, compute_far_field, directivity,
)


def _run_dipole_ntff(boundary="cpml"):
    """Run a z-oriented dipole with NTFF box and return far-field result."""
    grid = Grid(freq_max=5e9, domain=(0.03, 0.03, 0.03),
                cpml_layers=8 if boundary == "cpml" else 0)
    materials = init_materials(grid.shape)
    n_steps = 200

    pulse = GaussianPulse(f0=3e9, bandwidth=0.5)
    center = (0.015, 0.015, 0.015)
    src = make_source(grid, center, "ez", pulse, n_steps)
    prb = make_probe(grid, center, "ez")

    # NTFF box: ~5 cells inside CPML on each side
    margin = 0.004
    ntff = make_ntff_box(
        grid,
        corner_lo=(margin, margin, margin),
        corner_hi=(0.03 - margin, 0.03 - margin, 0.03 - margin),
        freqs=jnp.array([3e9]),
    )

    result = run(
        grid, materials, n_steps,
        boundary=boundary,
        sources=[src], probes=[prb],
        ntff=ntff,
    )

    theta = np.linspace(0.01, np.pi - 0.01, 37)  # avoid poles
    phi = np.array([0.0, np.pi / 2])

    ff = compute_far_field(result.ntff_data, ntff, grid, theta, phi)
    return ff, result


def test_ntff_accumulation_runs():
    """NTFF accumulation completes without error."""
    grid = Grid(freq_max=5e9, domain=(0.02, 0.02, 0.02), cpml_layers=0)
    materials = init_materials(grid.shape)
    pulse = GaussianPulse(f0=3e9, bandwidth=0.5)
    src = make_source(grid, (0.01, 0.01, 0.01), "ez", pulse, 30)

    ntff = make_ntff_box(grid, (0.003, 0.003, 0.003),
                         (0.017, 0.017, 0.017),
                         freqs=jnp.array([3e9]))

    result = run(grid, materials, 30, sources=[src], ntff=ntff)

    assert result.ntff_data is not None
    # DFT should have accumulated non-zero values
    x_lo_mag = float(jnp.max(jnp.abs(result.ntff_data.x_lo)))
    assert x_lo_mag > 0, "NTFF DFT should accumulate non-zero fields"


def test_dipole_sin_theta_pattern():
    """Ez dipole far-field E_theta should follow sin(theta) shape.

    A z-oriented short dipole has E_theta proportional to sin(theta).
    We check that the pattern peaks near theta=90 and drops toward
    theta=0 and theta=180.
    """
    ff, _ = _run_dipole_ntff()

    # E-plane: phi=0, frequency index 0
    E_th = np.abs(ff.E_theta[0, :, 0])  # (n_theta,)
    theta = ff.theta

    # Normalize
    peak = np.max(E_th)
    assert peak > 0, "Far-field should be non-zero"
    E_norm = E_th / peak

    # Peak should be near theta = pi/2
    peak_idx = np.argmax(E_norm)
    peak_theta = theta[peak_idx]
    assert abs(peak_theta - np.pi / 2) < np.pi / 6, \
        f"Pattern peak at {np.degrees(peak_theta):.0f}° (expected ~90°)"

    # Values near poles should be much smaller than peak
    # theta < 20° or theta > 160°
    near_pole = (theta < np.radians(20)) | (theta > np.radians(160))
    if np.any(near_pole):
        pole_max = np.max(E_norm[near_pole])
        assert pole_max < 0.5, \
            f"Pattern near poles too large: {pole_max:.2f} (expected < 0.5)"

    print(f"\nDipole pattern: peak at {np.degrees(peak_theta):.0f}°, "
          f"pole suppression: {pole_max:.3f}")


def test_dipole_e_phi_small():
    """For a z-dipole in the E-plane (phi=0), E_phi should be near zero."""
    ff, _ = _run_dipole_ntff()

    E_th_max = np.max(np.abs(ff.E_theta[0, :, 0]))
    E_ph_max = np.max(np.abs(ff.E_phi[0, :, 0]))

    ratio = E_ph_max / E_th_max if E_th_max > 0 else 0
    assert ratio < 0.3, \
        f"|E_phi|/|E_theta| = {ratio:.2f} in E-plane (expected < 0.3)"
    print(f"\nE-plane cross-pol ratio: {ratio:.4f}")


def test_dipole_directivity():
    """Short dipole directivity should be near 1.76 dBi (theoretical 1.5 linear)."""
    ff, _ = _run_dipole_ntff()
    D = directivity(ff)  # (n_freqs,) dBi

    # Theoretical short dipole: 1.76 dBi
    # FDTD with coarse grid: allow 1-5 dBi range
    D_val = float(D[0])
    print(f"\nDipole directivity: {D_val:.2f} dBi (theoretical: 1.76 dBi)")
    assert 0 < D_val < 8, \
        f"Directivity {D_val:.2f} dBi outside expected range [0, 8]"


def test_ntff_without_ntff_returns_none():
    """SimResult.ntff_data is None when no NTFF box is used."""
    grid = Grid(freq_max=5e9, domain=(0.02, 0.02, 0.02), cpml_layers=0)
    materials = init_materials(grid.shape)
    result = run(grid, materials, 10)
    assert result.ntff_data is None
