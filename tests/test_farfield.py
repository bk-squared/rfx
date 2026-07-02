"""Tests for near-to-far-field transform.

Validates that:
1. NTFF DFT accumulation runs inside the scan without error
2. Hertzian dipole (Ez source) produces sin(theta) radiation pattern
3. Directivity of a short dipole is approximately 1.76 dBi
4. Far-field E_phi is near zero in the E-plane (phi=0)
"""

import numpy as np
import jax.numpy as jnp
import pytest

from rfx.grid import Grid
from rfx.core.yee import init_materials
from rfx.sources.sources import GaussianPulse
from rfx.simulation import make_source, make_probe, run
from rfx.farfield import (
    make_ntff_box, compute_far_field, directivity,
)
from rfx.optimize_objectives import maximize_directivity


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
    assert result.ntff_box == ntff
    assert result.grid is grid


def test_maximize_directivity_works_with_low_level_simresult():
    """Far-field objective should work on the low-level SimResult contract."""
    grid = Grid(freq_max=5e9, domain=(0.02, 0.02, 0.02), cpml_layers=0)
    materials = init_materials(grid.shape)
    pulse = GaussianPulse(f0=3e9, bandwidth=0.5)
    src = make_source(grid, (0.01, 0.01, 0.01), "ez", pulse, 30)

    ntff = make_ntff_box(
        grid,
        (0.003, 0.003, 0.003),
        (0.017, 0.017, 0.017),
        freqs=jnp.array([3e9]),
    )

    result = run(grid, materials, 30, sources=[src], ntff=ntff)
    obj = maximize_directivity(theta_target=np.pi / 2, phi_target=0.0)
    loss = obj(result)

    assert loss.shape == ()
    assert jnp.isfinite(loss)
    # Default is the log-ratio objective: loss = -(log U - log P) = log(4pi/D),
    # positive for a low-directivity source (D < 4pi) — not the old ratio-mode
    # negative value. Guard that the default IS the log-ratio path.
    assert float(loss) == float(
        maximize_directivity(theta_target=np.pi / 2, phi_target=0.0,
                             log_ratio=True)(result))


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
    """Short (Hertzian) dipole directivity should be near 1.76 dBi (1.5 linear).

    R5 measure-before-gate note: the shared ``_run_dipole_ntff`` fixture samples
    ``phi = [0, pi/2]`` (two points over a quarter circle), which is fine for the
    pattern-*shape* tests but UNDER-samples the ``directivity()`` P_rad solid-angle
    integral and inflates D by ~+3 dB (measured 4.78 dBi — a pure integration-grid
    artefact, not physics: E_theta is phi-flat to ~1% and peaks exactly at 90).
    We therefore re-integrate the *same* NTFF data on a proper full-2*pi sphere,
    which recovers the physical directivity (measured ~1.84 dBi, theory 1.76).
    """
    _, result = _run_dipole_ntff()

    # Full-sphere sampling so the P_rad integral is unbiased.
    theta = np.linspace(0.01, np.pi - 0.01, 37)
    phi = np.linspace(0.0, 2 * np.pi, 24, endpoint=False)
    ff = compute_far_field(result.ntff_data, result.ntff_box, result.grid,
                           theta, phi)
    D_val = float(directivity(ff)[0])

    print(f"\nShort-dipole directivity: {D_val:.3f} dBi (theory 1.76 dBi)")
    # Theory 1.76 dBi; +/-0.75 absorbs the coarse 3 mm / f0=3 GHz grid bias.
    assert abs(D_val - 1.76) < 0.75, \
        f"Short-dipole directivity {D_val:.3f} dBi outside 1.76 +/- 0.75 dBi"


def test_ntff_without_ntff_returns_none():
    """SimResult.ntff_data is None when no NTFF box is used."""
    grid = Grid(freq_max=5e9, domain=(0.02, 0.02, 0.02), cpml_layers=0)
    materials = init_materials(grid.shape)
    result = run(grid, materials, 10)
    assert result.ntff_data is None


def _run_halfwave_ntff(f0=3e9, freq_max=5e9, n_arm=7, n_steps=800,
                       domain=(0.08, 0.08, 0.14)):
    """Run a genuine center-fed ~lambda/2 PEC-wire dipole and return far-field.

    The radiator is a single-cell-wide PEC column along z with a one-cell feed
    gap at the center, driven by an ez soft source at the gap. Unlike a uniform
    imposed current line (which radiates 2.43 dBi, not 2.15), a real center-fed
    conductor self-develops the sinusoidal current of a half-wave dipole.
    """
    grid = Grid(freq_max=freq_max, domain=domain, cpml_layers=8)
    dx = grid.dx
    materials = init_materials(grid.shape)

    center = (domain[0] / 2, domain[1] / 2, domain[2] / 2)
    ic, jc, kc = grid.position_to_index(center)

    # PEC wire: arms of n_arm cells above/below a one-cell feed gap at kc.
    pec_mask = np.zeros(grid.shape, dtype=bool)
    pec_mask[ic, jc, kc - n_arm:kc + n_arm + 1] = True
    pec_mask[ic, jc, kc] = False  # feed gap (source cell must radiate)
    pec_mask = jnp.asarray(pec_mask)

    pulse = GaussianPulse(f0=f0, bandwidth=0.6)
    src = make_source(grid, center, "ez", pulse, n_steps)

    # NTFF box: enclose the wire while staying a few cells inside the 8-cell CPML.
    cpml_m = 8 * dx
    xy_m = cpml_m + 2 * dx
    z_m = cpml_m + 3 * dx
    ntff = make_ntff_box(
        grid,
        corner_lo=(xy_m, xy_m, z_m),
        corner_hi=(domain[0] - xy_m, domain[1] - xy_m, domain[2] - z_m),
        freqs=jnp.array([f0]),
    )

    result = run(grid, materials, n_steps, boundary="cpml",
                 sources=[src], ntff=ntff, pec_mask=pec_mask)

    theta = np.linspace(0.01, np.pi - 0.01, 73)
    phi = np.linspace(0.0, 2 * np.pi, 24, endpoint=False)
    ff = compute_far_field(result.ntff_data, ntff, grid, theta, phi)
    return ff, result


@pytest.mark.slow_physics
def test_halfwave_dipole_directivity():
    """Half-wave dipole directivity should be near 2.15 dBi (1.64 linear).

    Uses a genuine center-fed ~lambda/2 PEC wire (n_arm=7 -> L=0.450 lambda,
    the classic resonant length). Measured D = 2.380 dBi (full-2*pi sphere,
    n_steps-converged to 3 decimals). The coarse 3 mm / f0=3 GHz grid biases D
    slightly high (all lengths sit above theory; the offset shrinks with dx),
    which the +/-0.5 tolerance absorbs. A uniform imposed current line would
    give 2.43 dBi and be the wrong geometry, hence the real PEC conductor.
    """
    ff, _ = _run_halfwave_ntff()
    D_val = float(directivity(ff)[0])

    # Pattern-shape witness (R5): a half-wave dipole peaks broadside (theta=90).
    E_th = np.abs(ff.E_theta[0]).mean(axis=1)  # phi-averaged
    peak_theta = ff.theta[int(np.argmax(E_th))]
    assert abs(peak_theta - np.pi / 2) < np.radians(10), \
        f"Half-wave pattern peak at {np.degrees(peak_theta):.0f} (expected ~90)"

    print(f"\nHalf-wave dipole directivity: {D_val:.3f} dBi (theory 2.15 dBi)")
    assert abs(D_val - 2.15) < 0.5, \
        f"Half-wave dipole directivity {D_val:.3f} dBi outside 2.15 +/- 0.5 dBi"
